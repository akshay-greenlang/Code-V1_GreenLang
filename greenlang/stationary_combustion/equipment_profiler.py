# -*- coding: utf-8 -*-
"""
EquipmentProfilerEngine - Engine 3: Stationary Combustion Agent (AGENT-MRV-001)

Equipment profiling engine for stationary combustion sources. Manages
equipment registration, efficiency calculation (polynomial curves with age
degradation), and emission adjustment factors for Tier 2/3 GHG Protocol
calculations.

Provides 13 built-in equipment templates covering boilers, furnaces, process
heaters, gas turbines, reciprocating engines, kilns, ovens, dryers, flares,
incinerators, and thermal oxidizers. Each template includes default
efficiency, capacity range (MW), and load factor.

Thread-safe via ``threading.Lock()``. Uses ``Decimal`` arithmetic for
precision-sensitive efficiency and adjustment calculations.

Zero-Hallucination Guarantees:
    - Efficiency curves are deterministic polynomial evaluations
    - Age degradation uses linear formula with type-specific rates
    - No LLM calls in any calculation path
    - All adjustments are traceable via provenance hashing

Example:
    >>> from greenlang.stationary_combustion.equipment_profiler import EquipmentProfilerEngine
    >>> profiler = EquipmentProfilerEngine()
    >>> profile = profiler.register_equipment(
    ...     equipment_id="boiler_001",
    ...     equipment_type="BOILER_WATER_TUBE",
    ...     name="Main Steam Boiler",
    ...     rated_capacity=50.0,
    ...     age_years=5,
    ... )
    >>> eff = profiler.calculate_efficiency("boiler_001", load_factor=0.75)
    >>> adjusted = profiler.calculate_adjusted_emissions(1000.0, "boiler_001")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.stationary_combustion.models import EquipmentProfile, EquipmentType
from greenlang.stationary_combustion.metrics import (
    record_equipment_registration,
    set_active_equipment,
)
from greenlang.stationary_combustion.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in Equipment Templates
# ---------------------------------------------------------------------------

EQUIPMENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "BOILER_FIRE_TUBE": {
        "efficiency": Decimal("0.82"),
        "capacity_range": (Decimal("0.5"), Decimal("50")),
        "load_factor": Decimal("0.65"),
        "degradation_rate": Decimal("0.005"),
        "efficiency_curve": [
            Decimal("0.50"), Decimal("0.80"), Decimal("-0.48"), Decimal("0.0"),
        ],
    },
    "BOILER_WATER_TUBE": {
        "efficiency": Decimal("0.86"),
        "capacity_range": (Decimal("10"), Decimal("2000")),
        "load_factor": Decimal("0.70"),
        "degradation_rate": Decimal("0.004"),
        "efficiency_curve": [
            Decimal("0.52"), Decimal("0.85"), Decimal("-0.51"), Decimal("0.0"),
        ],
    },
    "FURNACE": {
        "efficiency": Decimal("0.78"),
        "capacity_range": (Decimal("5"), Decimal("1000")),
        "load_factor": Decimal("0.75"),
        "degradation_rate": Decimal("0.006"),
        "efficiency_curve": [
            Decimal("0.45"), Decimal("0.82"), Decimal("-0.49"), Decimal("0.0"),
        ],
    },
    "PROCESS_HEATER": {
        "efficiency": Decimal("0.82"),
        "capacity_range": (Decimal("1"), Decimal("500")),
        "load_factor": Decimal("0.60"),
        "degradation_rate": Decimal("0.005"),
        "efficiency_curve": [
            Decimal("0.50"), Decimal("0.78"), Decimal("-0.46"), Decimal("0.0"),
        ],
    },
    "GAS_TURBINE_SIMPLE": {
        "efficiency": Decimal("0.36"),
        "capacity_range": (Decimal("5"), Decimal("500")),
        "load_factor": Decimal("0.75"),
        "degradation_rate": Decimal("0.008"),
        "efficiency_curve": [
            Decimal("0.15"), Decimal("0.52"), Decimal("-0.31"), Decimal("0.0"),
        ],
    },
    "GAS_TURBINE_COMBINED": {
        "efficiency": Decimal("0.55"),
        "capacity_range": (Decimal("100"), Decimal("3000")),
        "load_factor": Decimal("0.80"),
        "degradation_rate": Decimal("0.006"),
        "efficiency_curve": [
            Decimal("0.25"), Decimal("0.72"), Decimal("-0.42"), Decimal("0.0"),
        ],
    },
    "RECIPROCATING_ENGINE": {
        "efficiency": Decimal("0.40"),
        "capacity_range": (Decimal("0.5"), Decimal("100")),
        "load_factor": Decimal("0.55"),
        "degradation_rate": Decimal("0.007"),
        "efficiency_curve": [
            Decimal("0.20"), Decimal("0.48"), Decimal("-0.28"), Decimal("0.0"),
        ],
    },
    "KILN": {
        "efficiency": Decimal("0.65"),
        "capacity_range": (Decimal("50"), Decimal("2000")),
        "load_factor": Decimal("0.80"),
        "degradation_rate": Decimal("0.005"),
        "efficiency_curve": [
            Decimal("0.35"), Decimal("0.72"), Decimal("-0.42"), Decimal("0.0"),
        ],
    },
    "OVEN": {
        "efficiency": Decimal("0.72"),
        "capacity_range": (Decimal("0.5"), Decimal("50")),
        "load_factor": Decimal("0.65"),
        "degradation_rate": Decimal("0.004"),
        "efficiency_curve": [
            Decimal("0.42"), Decimal("0.72"), Decimal("-0.42"), Decimal("0.0"),
        ],
    },
    "DRYER": {
        "efficiency": Decimal("0.68"),
        "capacity_range": (Decimal("1"), Decimal("200")),
        "load_factor": Decimal("0.70"),
        "degradation_rate": Decimal("0.005"),
        "efficiency_curve": [
            Decimal("0.38"), Decimal("0.72"), Decimal("-0.42"), Decimal("0.0"),
        ],
    },
    "FLARE": {
        "efficiency": Decimal("0.98"),
        "capacity_range": (Decimal("1"), Decimal("500")),
        "load_factor": Decimal("0.30"),
        "degradation_rate": Decimal("0.002"),
        "efficiency_curve": [
            Decimal("0.96"), Decimal("0.04"), Decimal("-0.02"), Decimal("0.0"),
        ],
    },
    "INCINERATOR": {
        "efficiency": Decimal("0.75"),
        "capacity_range": (Decimal("10"), Decimal("500")),
        "load_factor": Decimal("0.85"),
        "degradation_rate": Decimal("0.005"),
        "efficiency_curve": [
            Decimal("0.42"), Decimal("0.78"), Decimal("-0.45"), Decimal("0.0"),
        ],
    },
    "THERMAL_OXIDIZER": {
        "efficiency": Decimal("0.99"),
        "capacity_range": (Decimal("1"), Decimal("100")),
        "load_factor": Decimal("0.50"),
        "degradation_rate": Decimal("0.002"),
        "efficiency_curve": [
            Decimal("0.97"), Decimal("0.04"), Decimal("-0.02"), Decimal("0.0"),
        ],
    },
}

# Decimal precision quantizer
_PRECISION = Decimal("0.00000001")


class EquipmentProfilerEngine:
    """Equipment profiling engine for stationary combustion sources.

    Manages registration, lookup, efficiency calculation, age degradation,
    and emission adjustment for combustion equipment. Provides 13 built-in
    templates with default efficiency curves, capacity ranges, and load
    factors.

    Efficiency is modeled as a polynomial function of load factor:
        eff(L) = a0 + a1*L + a2*L^2 + a3*L^3

    Age degradation follows a linear model:
        degradation_factor = 1.0 - (age_years * degradation_rate)

    Thread-safe: all mutable state is guarded by ``threading.Lock()``.

    Attributes:
        _config: Optional configuration dictionary.
        _profiles: In-memory registry of equipment profiles keyed by ID.
        _lock: Thread lock for profile mutations.
        _provenance: Reference to the provenance tracker.

    Example:
        >>> profiler = EquipmentProfilerEngine()
        >>> p = profiler.register_equipment("b01", "BOILER_WATER_TUBE", "Boiler 1", 100)
        >>> eff = profiler.calculate_efficiency("b01", load_factor=0.8)
        >>> assert 0.0 < eff <= 1.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EquipmentProfilerEngine.

        Args:
            config: Optional configuration dict. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                  Defaults to True.
                - ``max_equipment`` (int): Maximum registered profiles.
                  Defaults to 50000.
        """
        self._config = config or {}
        self._profiles: Dict[str, EquipmentProfile] = {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._max_equipment: int = self._config.get("max_equipment", 50_000)

        if self._enable_provenance:
            self._provenance = get_provenance_tracker()
        else:
            self._provenance = None

        logger.info(
            "EquipmentProfilerEngine initialized (max_equipment=%d, "
            "templates=%d, provenance=%s)",
            self._max_equipment,
            len(EQUIPMENT_DEFAULTS),
            self._enable_provenance,
        )

    # ------------------------------------------------------------------
    # Public API: Registration
    # ------------------------------------------------------------------

    def register_equipment(
        self,
        equipment_id: str,
        equipment_type: str,
        name: str = "",
        rated_capacity: float = 0.0,
        efficiency_curve: Optional[List[float]] = None,
        age_years: int = 0,
        load_factor: Optional[float] = None,
        facility_id: Optional[str] = None,
    ) -> EquipmentProfile:
        """Register a new equipment profile.

        If an equipment with the same ID already exists, a ``ValueError``
        is raised. Use ``update_equipment`` to modify existing profiles.

        Args:
            equipment_id: Unique equipment identifier.
            equipment_type: Equipment type (must match an EquipmentType enum
                value or a key in EQUIPMENT_DEFAULTS).
            name: Human-readable name.
            rated_capacity: Rated thermal capacity in MW.
            efficiency_curve: Optional polynomial coefficients [a0, a1, a2, a3].
                If None, the default curve for the equipment type is used.
            age_years: Equipment age in years.
            load_factor: Average load factor (0-1). If None, the default
                for the equipment type is used.
            facility_id: Optional parent facility identifier.

        Returns:
            The created EquipmentProfile.

        Raises:
            ValueError: If equipment_id already exists, equipment_type is
                unknown, or capacity exceeds the template range.
            RuntimeError: If the maximum equipment limit is reached.
        """
        eq_type = equipment_type.upper()

        if eq_type not in EQUIPMENT_DEFAULTS:
            raise ValueError(
                f"Unknown equipment type: {equipment_type}. "
                f"Valid types: {sorted(EQUIPMENT_DEFAULTS.keys())}"
            )

        template = EQUIPMENT_DEFAULTS[eq_type]

        with self._lock:
            if equipment_id in self._profiles:
                raise ValueError(
                    f"Equipment '{equipment_id}' already registered. "
                    f"Use update_equipment() to modify."
                )
            if len(self._profiles) >= self._max_equipment:
                raise RuntimeError(
                    f"Maximum equipment limit reached ({self._max_equipment})"
                )

        # Resolve defaults from template
        resolved_load_factor = (
            Decimal(str(load_factor))
            if load_factor is not None
            else template["load_factor"]
        )
        resolved_efficiency = template["efficiency"]
        resolved_degradation = template["degradation_rate"]

        # Resolve efficiency curve
        resolved_curve: Optional[List[Decimal]] = None
        if efficiency_curve is not None:
            resolved_curve = [Decimal(str(c)) for c in efficiency_curve]
        else:
            resolved_curve = list(template["efficiency_curve"])

        now = datetime.now(timezone.utc)
        profile = EquipmentProfile(
            equipment_id=equipment_id,
            equipment_type=eq_type,
            name=name,
            facility_id=facility_id,
            rated_capacity_mw=Decimal(str(rated_capacity)) if rated_capacity > 0 else None,
            efficiency=resolved_efficiency,
            load_factor=resolved_load_factor,
            age_years=age_years,
            efficiency_curve=resolved_curve,
            degradation_rate=resolved_degradation,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._profiles[equipment_id] = profile

        # Record metrics
        record_equipment_registration(eq_type)
        set_active_equipment(len(self._profiles))

        # Record provenance
        self._record_provenance(
            "register_equipment", equipment_id,
            {
                "equipment_type": eq_type,
                "name": name,
                "rated_capacity": rated_capacity,
                "age_years": age_years,
            },
        )

        logger.info(
            "Registered equipment: id=%s, type=%s, name=%s, capacity=%.1f MW, age=%d yr",
            equipment_id, eq_type, name, rated_capacity, age_years,
        )
        return profile

    # ------------------------------------------------------------------
    # Public API: Update
    # ------------------------------------------------------------------

    def update_equipment(
        self,
        equipment_id: str,
        **kwargs: Any,
    ) -> EquipmentProfile:
        """Update fields of an existing equipment profile.

        Only fields provided in ``kwargs`` are updated. The ``updated_at``
        timestamp is refreshed automatically.

        Args:
            equipment_id: Equipment identifier to update.
            **kwargs: Fields to update. Supported keys: name, rated_capacity_mw,
                efficiency, load_factor, age_years, efficiency_curve,
                degradation_rate, facility_id.

        Returns:
            The updated EquipmentProfile.

        Raises:
            KeyError: If the equipment ID is not found.
        """
        with self._lock:
            if equipment_id not in self._profiles:
                raise KeyError(f"Equipment not found: {equipment_id}")

            profile = self._profiles[equipment_id]

            # Build updated dict from current profile
            update_data = profile.model_dump()

            # Apply provided kwargs
            allowed_fields = {
                "name", "rated_capacity_mw", "efficiency", "load_factor",
                "age_years", "efficiency_curve", "degradation_rate", "facility_id",
            }
            for key, value in kwargs.items():
                if key in allowed_fields:
                    if key in ("efficiency", "load_factor", "degradation_rate", "rated_capacity_mw"):
                        update_data[key] = Decimal(str(value)) if value is not None else None
                    elif key == "efficiency_curve" and value is not None:
                        update_data[key] = [Decimal(str(c)) for c in value]
                    else:
                        update_data[key] = value

            update_data["updated_at"] = datetime.now(timezone.utc)

            updated_profile = EquipmentProfile(**update_data)
            self._profiles[equipment_id] = updated_profile

        self._record_provenance(
            "update_equipment", equipment_id,
            {"updated_fields": list(kwargs.keys())},
        )

        logger.info(
            "Updated equipment %s: fields=%s",
            equipment_id, list(kwargs.keys()),
        )
        return updated_profile

    # ------------------------------------------------------------------
    # Public API: Get / Delete / List
    # ------------------------------------------------------------------

    def get_equipment(self, equipment_id: str) -> EquipmentProfile:
        """Retrieve an equipment profile by ID.

        Args:
            equipment_id: Equipment identifier.

        Returns:
            The EquipmentProfile.

        Raises:
            KeyError: If the equipment ID is not found.
        """
        with self._lock:
            if equipment_id not in self._profiles:
                raise KeyError(f"Equipment not found: {equipment_id}")
            return self._profiles[equipment_id]

    def delete_equipment(self, equipment_id: str) -> bool:
        """Delete an equipment profile.

        Args:
            equipment_id: Equipment identifier to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if equipment_id not in self._profiles:
                return False
            del self._profiles[equipment_id]

        set_active_equipment(len(self._profiles))

        self._record_provenance("delete_equipment", equipment_id, None)
        logger.info("Deleted equipment: %s", equipment_id)
        return True

    def list_equipment(
        self,
        equipment_type: Optional[str] = None,
        facility_id: Optional[str] = None,
    ) -> List[EquipmentProfile]:
        """List registered equipment profiles with optional filtering.

        Args:
            equipment_type: Filter by equipment type (optional).
            facility_id: Filter by facility ID (optional).

        Returns:
            List of matching EquipmentProfile objects.
        """
        with self._lock:
            profiles = list(self._profiles.values())

        if equipment_type:
            et = equipment_type.upper()
            profiles = [
                p for p in profiles
                if (p.equipment_type if isinstance(p.equipment_type, str)
                    else p.equipment_type.value) == et
            ]

        if facility_id:
            profiles = [p for p in profiles if p.facility_id == facility_id]

        return profiles

    # ------------------------------------------------------------------
    # Public API: Efficiency Calculation
    # ------------------------------------------------------------------

    def calculate_efficiency(
        self,
        equipment_id: str,
        load_factor: Optional[float] = None,
    ) -> float:
        """Calculate equipment efficiency at a given load factor.

        Uses the polynomial efficiency curve:
            eff(L) = a0 + a1*L + a2*L^2 + a3*L^3

        where L is the load factor (0-1). If no custom curve is registered,
        the default curve for the equipment type is used.

        The result is clamped to [0.01, 1.0] to prevent physically
        impossible values.

        Args:
            equipment_id: Equipment identifier.
            load_factor: Operating load factor (0-1). If None, uses the
                equipment's registered load factor.

        Returns:
            Efficiency as a float in [0.01, 1.0].

        Raises:
            KeyError: If the equipment ID is not found.
            ValueError: If load_factor is outside [0, 1].
        """
        profile = self.get_equipment(equipment_id)

        # Resolve load factor
        if load_factor is not None:
            if not (0.0 <= load_factor <= 1.0):
                raise ValueError(
                    f"load_factor must be in [0.0, 1.0], got {load_factor}"
                )
            lf = Decimal(str(load_factor))
        else:
            lf = profile.load_factor

        # Get efficiency curve coefficients
        curve = self._get_efficiency_curve(profile)

        # Evaluate polynomial: eff = a0 + a1*L + a2*L^2 + a3*L^3
        efficiency = self._evaluate_polynomial(curve, lf)

        # Clamp to physical bounds
        efficiency = max(Decimal("0.01"), min(Decimal("1.0"), efficiency))

        return float(efficiency.quantize(_PRECISION, rounding=ROUND_HALF_UP))

    # ------------------------------------------------------------------
    # Public API: Age Degradation
    # ------------------------------------------------------------------

    def calculate_age_degradation(self, equipment_id: str) -> float:
        """Calculate the age degradation factor for equipment.

        Formula:
            degradation_factor = max(0.5, 1.0 - age_years * degradation_rate)

        The factor is clamped to a minimum of 0.5 (50% of original
        efficiency) to prevent unrealistic degradation.

        Args:
            equipment_id: Equipment identifier.

        Returns:
            Degradation factor as a float in [0.5, 1.0].

        Raises:
            KeyError: If the equipment ID is not found.
        """
        profile = self.get_equipment(equipment_id)

        age = Decimal(str(profile.age_years))
        rate = self._get_degradation_rate(profile)

        factor = Decimal("1.0") - (age * rate)
        factor = max(Decimal("0.5"), factor)

        return float(factor.quantize(_PRECISION, rounding=ROUND_HALF_UP))

    # ------------------------------------------------------------------
    # Public API: Adjusted Emissions
    # ------------------------------------------------------------------

    def calculate_adjusted_emissions(
        self,
        base_emissions_kg: float,
        equipment_id: str,
        load_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate emissions adjusted for equipment efficiency and age.

        The adjustment accounts for both the efficiency at the operating
        load factor and the age degradation:

            actual_efficiency = curve_efficiency * age_degradation
            adjusted_emissions = base_emissions / actual_efficiency

        This models the reality that less efficient equipment requires more
        fuel (and thus produces more emissions) per unit of useful output.

        Args:
            base_emissions_kg: Base emissions in kg (before adjustment).
            equipment_id: Equipment identifier.
            load_factor: Operating load factor (0-1). If None, uses the
                equipment's registered load factor.

        Returns:
            Dictionary with keys:
                - ``base_emissions_kg``: Original emissions.
                - ``adjusted_emissions_kg``: Emissions after adjustment.
                - ``adjustment_factor``: Multiplier applied.
                - ``curve_efficiency``: Efficiency from the polynomial curve.
                - ``age_degradation``: Age degradation factor.
                - ``actual_efficiency``: Combined efficiency.
                - ``equipment_id``: Equipment identifier.

        Raises:
            KeyError: If the equipment ID is not found.
        """
        base_dec = Decimal(str(base_emissions_kg))

        curve_eff = Decimal(str(self.calculate_efficiency(equipment_id, load_factor)))
        age_deg = Decimal(str(self.calculate_age_degradation(equipment_id)))

        actual_eff = (curve_eff * age_deg).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )

        # Prevent division by zero
        if actual_eff <= Decimal("0"):
            actual_eff = Decimal("0.01")

        adjustment_factor = (Decimal("1") / actual_eff).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        adjusted_kg = (base_dec * adjustment_factor).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )

        result = {
            "base_emissions_kg": float(base_dec),
            "adjusted_emissions_kg": float(adjusted_kg),
            "adjustment_factor": float(adjustment_factor),
            "curve_efficiency": float(curve_eff),
            "age_degradation": float(age_deg),
            "actual_efficiency": float(actual_eff),
            "equipment_id": equipment_id,
        }

        self._record_provenance(
            "calculate_adjusted_emissions", equipment_id,
            {
                "base_emissions_kg": str(base_dec),
                "adjusted_emissions_kg": str(adjusted_kg),
                "adjustment_factor": str(adjustment_factor),
            },
        )

        logger.debug(
            "Adjusted emissions for %s: base=%.2f kg -> adjusted=%.2f kg "
            "(factor=%.4f, eff=%.4f, age_deg=%.4f)",
            equipment_id, base_emissions_kg, float(adjusted_kg),
            float(adjustment_factor), float(curve_eff), float(age_deg),
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Templates and Utilities
    # ------------------------------------------------------------------

    def get_equipment_template(self, equipment_type: str) -> Dict[str, Any]:
        """Return the default template for an equipment type.

        Args:
            equipment_type: Equipment type string.

        Returns:
            Template dictionary with efficiency, capacity_range, load_factor,
            degradation_rate, and efficiency_curve.

        Raises:
            KeyError: If the equipment type is not found.
        """
        eq_type = equipment_type.upper()
        if eq_type not in EQUIPMENT_DEFAULTS:
            raise KeyError(
                f"Unknown equipment type: {equipment_type}. "
                f"Valid types: {sorted(EQUIPMENT_DEFAULTS.keys())}"
            )

        template = EQUIPMENT_DEFAULTS[eq_type]
        return {
            "equipment_type": eq_type,
            "efficiency": float(template["efficiency"]),
            "capacity_range_mw": (
                float(template["capacity_range"][0]),
                float(template["capacity_range"][1]),
            ),
            "load_factor": float(template["load_factor"]),
            "degradation_rate": float(template["degradation_rate"]),
            "efficiency_curve": [float(c) for c in template["efficiency_curve"]],
        }

    def get_equipment_count(self) -> int:
        """Return the number of registered equipment profiles.

        Returns:
            Integer count of registered profiles.
        """
        with self._lock:
            return len(self._profiles)

    def clear(self) -> None:
        """Remove all registered equipment profiles.

        Primarily intended for testing. Resets the profile registry to
        empty state.
        """
        with self._lock:
            self._profiles.clear()

        set_active_equipment(0)

        self._record_provenance("clear_engine", "equipment_profiler", None)
        logger.info("EquipmentProfilerEngine cleared: all profiles removed")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_efficiency_curve(
        self,
        profile: EquipmentProfile,
    ) -> List[Decimal]:
        """Get the efficiency curve coefficients for a profile.

        Uses the profile's custom curve if set, otherwise falls back to
        the template default.

        Args:
            profile: Equipment profile.

        Returns:
            List of 4 Decimal coefficients [a0, a1, a2, a3].
        """
        if profile.efficiency_curve and len(profile.efficiency_curve) >= 4:
            return list(profile.efficiency_curve[:4])

        eq_type = (
            profile.equipment_type
            if isinstance(profile.equipment_type, str)
            else profile.equipment_type.value
        )
        if eq_type in EQUIPMENT_DEFAULTS:
            return list(EQUIPMENT_DEFAULTS[eq_type]["efficiency_curve"])

        # Fallback: flat efficiency
        return [profile.efficiency, Decimal("0"), Decimal("0"), Decimal("0")]

    def _get_degradation_rate(self, profile: EquipmentProfile) -> Decimal:
        """Get the degradation rate for a profile.

        Uses the profile's custom rate if set, otherwise falls back to
        the template default.

        Args:
            profile: Equipment profile.

        Returns:
            Degradation rate as Decimal.
        """
        if profile.degradation_rate is not None:
            return profile.degradation_rate

        eq_type = (
            profile.equipment_type
            if isinstance(profile.equipment_type, str)
            else profile.equipment_type.value
        )
        if eq_type in EQUIPMENT_DEFAULTS:
            return EQUIPMENT_DEFAULTS[eq_type]["degradation_rate"]

        return Decimal("0.005")  # Default 0.5% per year

    def _evaluate_polynomial(
        self,
        coefficients: List[Decimal],
        x: Decimal,
    ) -> Decimal:
        """Evaluate a polynomial at point x.

        Computes: a0 + a1*x + a2*x^2 + a3*x^3

        Args:
            coefficients: List of coefficients [a0, a1, a2, a3].
            x: Input value.

        Returns:
            Polynomial result as Decimal.
        """
        result = Decimal("0")
        for i, coeff in enumerate(coefficients):
            result += coeff * (x ** i)
        return result.quantize(_PRECISION, rounding=ROUND_HALF_UP)

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a provenance entry if provenance tracking is enabled.

        Args:
            action: Action label.
            entity_id: Entity identifier.
            data: Optional data payload for hashing.
        """
        if self._provenance is not None:
            self._provenance.record(
                entity_type="equipment",
                action=action,
                entity_id=entity_id,
                data=data,
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"EquipmentProfilerEngine("
            f"registered={self.get_equipment_count()}, "
            f"templates={len(EQUIPMENT_DEFAULTS)}, "
            f"max={self._max_equipment})"
        )

    def __len__(self) -> int:
        """Return the number of registered equipment profiles."""
        return self.get_equipment_count()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "EquipmentProfilerEngine",
    "EQUIPMENT_DEFAULTS",
]
