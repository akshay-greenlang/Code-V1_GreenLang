# -*- coding: utf-8 -*-
"""
EmissionCalculatorEngine - Core F-Gas Emission Calculation (Engine 2 of 7)

AGENT-MRV-SCOPE1-002: Refrigerants & F-Gas Agent

Core emission calculation engine implementing five GHG Protocol and EPA-
compliant calculation methodologies for Scope 1 refrigerant and fluorinated
gas emissions:

1. **Equipment-based** (GHG Protocol Chapter 8):
   Loss = Charge x Count x LeakRate
   Emissions_tCO2e = Loss x GWP / 1000

2. **Mass balance** (EPA 40 CFR Part 98 Subpart OO):
   Loss = BI + Purchases - Sales + Acq - Divest - EI - CapChange
   Loss = max(0, Loss)
   Emissions_tCO2e = Loss x GWP / 1000

3. **Screening** (Simplified estimation):
   Loss = TotalCharge x DefaultLeakRate
   Emissions_tCO2e = Loss x GWP / 1000

4. **Direct measurement** (Instrument-based):
   Emissions_tCO2e = MeasuredLoss x GWP / 1000

5. **Top-down** (Organizational aggregate):
   Emissions_tCO2e = (PurchasesTotal - RecoveredTotal) x GWP / 1000

Zero-Hallucination Guarantees:
    - All arithmetic uses Python Decimal for bit-perfect reproducibility.
    - No LLM involvement in any numeric path.
    - Every result carries a SHA-256 provenance hash.
    - Complete calculation trace for audit trail.
    - Same input always produces the same output (deterministic).

The engine does NOT import from refrigerant_database.py or
leak_rate_estimator.py. It receives GWP values and leak rates as
parameters. The pipeline engine coordinates between engines.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.refrigerants_fgas.emission_calculator import (
    ...     EmissionCalculatorEngine,
    ... )
    >>> from greenlang.refrigerants_fgas.models import (
    ...     CalculationMethod, RefrigerantType, EquipmentType,
    ... )
    >>> from decimal import Decimal
    >>> engine = EmissionCalculatorEngine()
    >>> results = engine.calculate_screening(
    ...     total_charge_kg=Decimal("200"),
    ...     ref_type=RefrigerantType.R_404A,
    ...     equip_type=EquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED,
    ...     leak_rate=Decimal("0.20"),
    ...     gwp=Decimal("3922"),
    ... )
    >>> print(results[0].emissions_tco2e)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-SCOPE1-002 Refrigerants & F-Gas Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["EmissionCalculatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.refrigerants_fgas.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.refrigerants_fgas.metrics import (
        PROMETHEUS_AVAILABLE as _METRICS_AVAILABLE,
        record_calculation as _record_calculation,
        record_emissions as _record_emissions,
        observe_calculation_duration as _observe_calculation_duration,
        record_batch as _record_batch,
        observe_batch_size as _observe_batch_size,
        set_active_calculations as _set_active_calculations,
    )
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calculation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]
    _record_batch = None  # type: ignore[assignment]
    _observe_batch_size = None  # type: ignore[assignment]
    _set_active_calculations = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Model imports
# ---------------------------------------------------------------------------

from greenlang.refrigerants_fgas.models import (
    CalculationMethod,
    CalculationResult,
    BatchCalculationResponse,
    EquipmentProfile,
    EquipmentType,
    EquipmentStatus,
    GasEmission,
    GWPSource,
    MassBalanceData,
    RefrigerantType,
    RefrigerantCategory,
)

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Conversion factor from kg to metric tonnes.
_KG_TO_TONNE = Decimal("0.001")

#: Default decimal precision for final emission values (3 decimal places).
_DEFAULT_PRECISION = 3

#: Default GWP source label when not otherwise specified.
_DEFAULT_GWP_SOURCE = "AR6"

#: Mapping from RefrigerantType to a human-readable gas name.
_GAS_NAMES: Dict[RefrigerantType, str] = {
    RefrigerantType.R_32: "R-32",
    RefrigerantType.R_125: "R-125",
    RefrigerantType.R_134A: "R-134a",
    RefrigerantType.R_143A: "R-143a",
    RefrigerantType.R_152A: "R-152a",
    RefrigerantType.R_227EA: "R-227ea",
    RefrigerantType.R_236FA: "R-236fa",
    RefrigerantType.R_245FA: "R-245fa",
    RefrigerantType.R_365MFC: "R-365mfc",
    RefrigerantType.R_23: "R-23",
    RefrigerantType.R_41: "R-41",
    RefrigerantType.R_404A: "R-404A",
    RefrigerantType.R_407A: "R-407A",
    RefrigerantType.R_407C: "R-407C",
    RefrigerantType.R_407F: "R-407F",
    RefrigerantType.R_410A: "R-410A",
    RefrigerantType.R_413A: "R-413A",
    RefrigerantType.R_417A: "R-417A",
    RefrigerantType.R_422D: "R-422D",
    RefrigerantType.R_427A: "R-427A",
    RefrigerantType.R_438A: "R-438A",
    RefrigerantType.R_448A: "R-448A",
    RefrigerantType.R_449A: "R-449A",
    RefrigerantType.R_452A: "R-452A",
    RefrigerantType.R_454B: "R-454B",
    RefrigerantType.R_507A: "R-507A",
    RefrigerantType.R_508B: "R-508B",
    RefrigerantType.R_1234YF: "R-1234yf",
    RefrigerantType.R_1234ZE: "R-1234ze",
    RefrigerantType.R_1233ZD: "R-1233zd",
    RefrigerantType.R_1336MZZ: "R-1336mzz",
    RefrigerantType.CF4: "CF4 (PFC-14)",
    RefrigerantType.C2F6: "C2F6 (PFC-116)",
    RefrigerantType.C3F8: "C3F8 (PFC-218)",
    RefrigerantType.C_C4F8: "c-C4F8 (PFC-318)",
    RefrigerantType.C4F10: "C4F10 (PFC-3-1-10)",
    RefrigerantType.C5F12: "C5F12 (PFC-4-1-12)",
    RefrigerantType.C6F14: "C6F14 (PFC-5-1-14)",
    RefrigerantType.SF6_GAS: "SF6",
    RefrigerantType.NF3_GAS: "NF3",
    RefrigerantType.SO2F2: "SO2F2",
    RefrigerantType.R_22: "R-22 (HCFC-22)",
    RefrigerantType.R_123: "R-123 (HCFC-123)",
    RefrigerantType.R_141B: "R-141b (HCFC-141b)",
    RefrigerantType.R_142B: "R-142b (HCFC-142b)",
    RefrigerantType.R_11: "R-11 (CFC-11)",
    RefrigerantType.R_12: "R-12 (CFC-12)",
    RefrigerantType.R_113: "R-113 (CFC-113)",
    RefrigerantType.R_114: "R-114 (CFC-114)",
    RefrigerantType.R_115: "R-115 (CFC-115)",
    RefrigerantType.R_502: "R-502",
    RefrigerantType.R_717: "R-717 (Ammonia)",
    RefrigerantType.R_744: "R-744 (CO2)",
    RefrigerantType.R_290: "R-290 (Propane)",
    RefrigerantType.R_600A: "R-600a (Isobutane)",
    RefrigerantType.CUSTOM: "Custom Refrigerant",
}

#: Mapping from RefrigerantType to its broad category for metrics recording.
_CATEGORY_MAP: Dict[RefrigerantType, str] = {
    RefrigerantType.R_32: "hfc",
    RefrigerantType.R_125: "hfc",
    RefrigerantType.R_134A: "hfc",
    RefrigerantType.R_143A: "hfc",
    RefrigerantType.R_152A: "hfc",
    RefrigerantType.R_227EA: "hfc",
    RefrigerantType.R_236FA: "hfc",
    RefrigerantType.R_245FA: "hfc",
    RefrigerantType.R_365MFC: "hfc",
    RefrigerantType.R_23: "hfc",
    RefrigerantType.R_41: "hfc",
    RefrigerantType.R_404A: "hfc_blend",
    RefrigerantType.R_407A: "hfc_blend",
    RefrigerantType.R_407C: "hfc_blend",
    RefrigerantType.R_407F: "hfc_blend",
    RefrigerantType.R_410A: "hfc_blend",
    RefrigerantType.R_413A: "hfc_blend",
    RefrigerantType.R_417A: "hfc_blend",
    RefrigerantType.R_422D: "hfc_blend",
    RefrigerantType.R_427A: "hfc_blend",
    RefrigerantType.R_438A: "hfc_blend",
    RefrigerantType.R_448A: "hfc_blend",
    RefrigerantType.R_449A: "hfc_blend",
    RefrigerantType.R_452A: "hfc_blend",
    RefrigerantType.R_454B: "hfc_blend",
    RefrigerantType.R_507A: "hfc_blend",
    RefrigerantType.R_508B: "hfc_blend",
    RefrigerantType.R_1234YF: "hfo",
    RefrigerantType.R_1234ZE: "hfo",
    RefrigerantType.R_1233ZD: "hfo",
    RefrigerantType.R_1336MZZ: "hfo",
    RefrigerantType.CF4: "pfc",
    RefrigerantType.C2F6: "pfc",
    RefrigerantType.C3F8: "pfc",
    RefrigerantType.C_C4F8: "pfc",
    RefrigerantType.C4F10: "pfc",
    RefrigerantType.C5F12: "pfc",
    RefrigerantType.C6F14: "pfc",
    RefrigerantType.SF6_GAS: "sf6",
    RefrigerantType.NF3_GAS: "nf3",
    RefrigerantType.SO2F2: "other",
    RefrigerantType.R_22: "hcfc",
    RefrigerantType.R_123: "hcfc",
    RefrigerantType.R_141B: "hcfc",
    RefrigerantType.R_142B: "hcfc",
    RefrigerantType.R_11: "cfc",
    RefrigerantType.R_12: "cfc",
    RefrigerantType.R_113: "cfc",
    RefrigerantType.R_114: "cfc",
    RefrigerantType.R_115: "cfc",
    RefrigerantType.R_502: "cfc",
    RefrigerantType.R_717: "natural",
    RefrigerantType.R_744: "natural",
    RefrigerantType.R_290: "natural",
    RefrigerantType.R_600A: "natural",
    RefrigerantType.CUSTOM: "other",
}

#: Default leak rates by equipment type for equipment-based calculations.
#: Fraction of charge lost per year.  Used when no custom leak_rate is
#: provided to calculate_equipment_based.
_DEFAULT_EQUIPMENT_LEAK_RATES: Dict[EquipmentType, Decimal] = {
    EquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED: Decimal("0.20"),
    EquipmentType.COMMERCIAL_REFRIGERATION_STANDALONE: Decimal("0.03"),
    EquipmentType.INDUSTRIAL_REFRIGERATION: Decimal("0.12"),
    EquipmentType.RESIDENTIAL_AC: Decimal("0.04"),
    EquipmentType.COMMERCIAL_AC: Decimal("0.06"),
    EquipmentType.CHILLERS_CENTRIFUGAL: Decimal("0.03"),
    EquipmentType.CHILLERS_SCREW: Decimal("0.04"),
    EquipmentType.HEAT_PUMPS: Decimal("0.04"),
    EquipmentType.TRANSPORT_REFRIGERATION: Decimal("0.22"),
    EquipmentType.SWITCHGEAR: Decimal("0.01"),
    EquipmentType.SEMICONDUCTOR: Decimal("0.08"),
    EquipmentType.FIRE_SUPPRESSION: Decimal("0.02"),
    EquipmentType.FOAM_BLOWING: Decimal("0.03"),
    EquipmentType.AEROSOLS: Decimal("1.00"),
    EquipmentType.SOLVENTS: Decimal("0.70"),
}


# ===========================================================================
# EmissionCalculatorEngine
# ===========================================================================


class EmissionCalculatorEngine:
    """Core F-gas emission calculation engine implementing five GHG Protocol
    and EPA-compliant methodologies.

    Provides deterministic, zero-hallucination emission calculations using
    Python Decimal arithmetic for bit-perfect reproducibility. Every
    calculation produces a SHA-256 provenance hash and a complete
    calculation trace for audit trails.

    Supported calculation methods:
        - EQUIPMENT_BASED: Per-equipment charge x leak rate x GWP.
        - MASS_BALANCE: Material balance equation per EPA 40 CFR Part 98.
        - SCREENING: Simplified estimation using average leak rates.
        - DIRECT_MEASUREMENT: Instrument-based measured losses.
        - TOP_DOWN: Organizational aggregate from purchase/recovery data.

    The engine receives GWP values and leak rates as parameters rather
    than importing from RefrigerantDatabaseEngine or LeakRateEstimatorEngine.
    The RefrigerantPipelineEngine coordinates between engines.

    Thread Safety:
        All mutable state (_calculation_history, _active_count) is
        protected by a reentrant lock. Concurrent callers are safe.

    Attributes:
        _config: Optional configuration dictionary.
        _calculation_history: List of all CalculationResult objects.
        _active_count: Number of currently in-progress calculations.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = EmissionCalculatorEngine()
        >>> result = engine.calculate_screening(
        ...     total_charge_kg=Decimal("200"),
        ...     ref_type=RefrigerantType.R_404A,
        ...     equip_type=EquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED,
        ...     leak_rate=Decimal("0.20"),
        ...     gwp=Decimal("3922"),
        ... )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the EmissionCalculatorEngine.

        Args:
            config: Optional configuration dictionary. If None, default
                settings are used. Supported keys:
                - ``precision``: Decimal places for final values (default 3).
                - ``gwp_source``: Default GWP source label (default "AR6").
                - ``max_history``: Maximum stored results (default 100000).
        """
        self._config: Dict[str, Any] = config or {}
        self._precision: int = self._config.get("precision", _DEFAULT_PRECISION)
        self._gwp_source_label: str = self._config.get("gwp_source", _DEFAULT_GWP_SOURCE)
        self._max_history: int = self._config.get("max_history", 100_000)
        self._calculation_history: List[CalculationResult] = []
        self._active_count: int = 0
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "EmissionCalculatorEngine initialized: "
            "precision=%d, gwp_source=%s, max_history=%d",
            self._precision,
            self._gwp_source_label,
            self._max_history,
        )

    # ------------------------------------------------------------------
    # Public API: Unified Calculate Entry Point
    # ------------------------------------------------------------------

    def calculate(
        self,
        method: CalculationMethod,
        **kwargs: Any,
    ) -> CalculationResult:
        """Route calculation to the appropriate method and return a
        CalculationResult.

        This is the primary entry point for all emission calculations.
        It dispatches to the specific method implementation based on the
        ``method`` parameter.

        Args:
            method: Calculation methodology to use.
            **kwargs: Method-specific keyword arguments. See the individual
                calculate_* methods for parameter details.

        Returns:
            CalculationResult with complete provenance.

        Raises:
            ValueError: If method is not supported or required parameters
                are missing.
        """
        t_start = time.monotonic()

        with self._lock:
            self._active_count += 1
        if _METRICS_AVAILABLE and _set_active_calculations is not None:
            try:
                _set_active_calculations(self._active_count)
            except Exception:
                pass

        try:
            if method == CalculationMethod.EQUIPMENT_BASED:
                gas_emissions = self._route_equipment_based(**kwargs)
            elif method == CalculationMethod.MASS_BALANCE:
                gas_emissions = self._route_mass_balance(**kwargs)
            elif method == CalculationMethod.SCREENING:
                gas_emissions = self._route_screening(**kwargs)
            elif method == CalculationMethod.DIRECT_MEASUREMENT:
                gas_emissions = self._route_direct_measurement(**kwargs)
            elif method == CalculationMethod.TOP_DOWN:
                gas_emissions = self._route_top_down(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported calculation method: {method}. "
                    f"Supported: {[m.value for m in CalculationMethod]}"
                )

            # Build CalculationResult
            total_loss_kg = sum(
                Decimal(str(g.loss_kg)) for g in gas_emissions
            )
            total_emissions_tco2e = sum(
                Decimal(str(g.emissions_tco2e)) for g in gas_emissions
            )

            total_loss_kg = self._apply_precision(total_loss_kg)
            total_emissions_tco2e = self._apply_precision(total_emissions_tco2e)

            # Build calculation trace
            trace: List[str] = []
            trace.append(f"Method: {method.value}")
            for g in gas_emissions:
                trace.append(
                    f"  {g.gas_name}: loss={g.loss_kg:.3f} kg, "
                    f"GWP={g.gwp_applied}, "
                    f"emissions={g.emissions_tco2e:.3f} tCO2e"
                )
            trace.append(
                f"Total: loss={total_loss_kg} kg, "
                f"emissions={total_emissions_tco2e} tCO2e"
            )

            # Build provenance hash
            provenance_data = {
                "method": method.value,
                "kwargs_keys": sorted(kwargs.keys()),
                "total_loss_kg": str(total_loss_kg),
                "total_emissions_tco2e": str(total_emissions_tco2e),
                "gas_count": len(gas_emissions),
            }
            provenance_hash = self._compute_hash(provenance_data)

            calc_id = f"calc_{uuid4().hex[:12]}"
            result = CalculationResult(
                calculation_id=calc_id,
                method=method,
                results=gas_emissions,
                total_loss_kg=float(total_loss_kg),
                total_emissions_tco2e=float(total_emissions_tco2e),
                blend_decomposition=False,
                provenance_hash=provenance_hash,
                timestamp=_utcnow(),
                calculation_trace=trace,
            )

            # Store in history
            with self._lock:
                if len(self._calculation_history) >= self._max_history:
                    self._calculation_history = self._calculation_history[
                        -(self._max_history // 2):
                    ]
                self._calculation_history.append(result)

            # Record provenance
            self._record_provenance(
                entity_type="calculation",
                action="calculate",
                entity_id=calc_id,
                data=provenance_data,
                metadata={"method": method.value},
            )

            # Record metrics
            elapsed = time.monotonic() - t_start
            self._record_calculation_metrics(
                method=method.value,
                gas_emissions=gas_emissions,
                status="completed",
                elapsed=elapsed,
            )

            logger.info(
                "Calculation completed: id=%s method=%s "
                "total_loss=%.3f kg total_emissions=%.3f tCO2e "
                "gases=%d in %.1fms",
                calc_id,
                method.value,
                total_loss_kg,
                total_emissions_tco2e,
                len(gas_emissions),
                elapsed * 1000,
            )

            return result

        except Exception:
            elapsed = time.monotonic() - t_start
            if _METRICS_AVAILABLE and _record_calculation is not None:
                try:
                    _record_calculation(method.value, "unknown", "failed")
                except Exception:
                    pass
            raise

        finally:
            with self._lock:
                self._active_count = max(0, self._active_count - 1)
            if _METRICS_AVAILABLE and _set_active_calculations is not None:
                try:
                    _set_active_calculations(self._active_count)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Public API: Equipment-Based Calculation
    # ------------------------------------------------------------------

    def calculate_equipment_based(
        self,
        profiles: List[EquipmentProfile],
        gwp_values: Dict[RefrigerantType, Decimal],
        leak_rates: Optional[Dict[str, Decimal]] = None,
    ) -> List[GasEmission]:
        """Calculate emissions using the equipment-based methodology.

        For each active equipment profile, computes:
            Loss_kg = charge_kg x equipment_count x leak_rate
            Emissions_tCO2e = Loss_kg x GWP / 1000

        Args:
            profiles: List of EquipmentProfile objects to calculate for.
                Only ACTIVE and MAINTENANCE status equipment is included.
            gwp_values: Dictionary mapping RefrigerantType to its GWP
                value as Decimal. The GWP for each profile's refrigerant
                must be present.
            leak_rates: Optional dictionary mapping equipment type value
                strings or equipment IDs to custom leak rate overrides
                as Decimal fractions. Falls back to the profile's
                custom_leak_rate, then to the default for the equipment
                type.

        Returns:
            List of GasEmission objects, one per equipment profile with
            non-zero emissions.

        Raises:
            ValueError: If a required GWP value is missing for a
                refrigerant type present in profiles.
        """
        t_start = time.monotonic()
        gas_emissions: List[GasEmission] = []
        trace: List[str] = [
            "Equipment-based calculation",
            f"Processing {len(profiles)} equipment profile(s)",
        ]

        for profile in profiles:
            # Skip non-active equipment
            if profile.status not in (
                EquipmentStatus.ACTIVE,
                EquipmentStatus.MAINTENANCE,
            ):
                trace.append(
                    f"  SKIP {profile.equipment_id}: "
                    f"status={profile.status.value}"
                )
                continue

            ref_type = profile.refrigerant_type

            # Resolve GWP
            gwp = gwp_values.get(ref_type)
            if gwp is None:
                raise ValueError(
                    f"GWP value not provided for refrigerant type "
                    f"'{ref_type.value}' (equipment_id={profile.equipment_id})"
                )
            gwp = Decimal(str(gwp))

            # Resolve leak rate
            leak_rate = self._resolve_leak_rate(
                profile=profile,
                leak_rates=leak_rates,
            )

            # Convert parameters to Decimal
            charge = Decimal(str(profile.charge_kg))
            count = Decimal(str(profile.equipment_count))

            # Calculate loss
            loss_kg = charge * count * leak_rate
            loss_kg = self._apply_precision(loss_kg)

            # Calculate emissions
            emissions_kg_co2e = loss_kg * gwp
            emissions_kg_co2e = self._apply_precision(emissions_kg_co2e)

            emissions_tco2e = loss_kg * gwp * _KG_TO_TONNE
            emissions_tco2e = self._apply_precision(emissions_tco2e)

            trace.append(
                f"  {profile.equipment_id} ({ref_type.value}): "
                f"charge={charge} kg x count={count} x "
                f"leak_rate={leak_rate} = loss={loss_kg} kg; "
                f"x GWP={gwp} / 1000 = {emissions_tco2e} tCO2e"
            )

            gas_name = _GAS_NAMES.get(ref_type, ref_type.value)
            emission = GasEmission(
                refrigerant_type=ref_type,
                gas_name=gas_name,
                loss_kg=float(loss_kg),
                gwp_applied=float(gwp),
                gwp_source=self._gwp_source_label,
                emissions_kg_co2e=float(emissions_kg_co2e),
                emissions_tco2e=float(emissions_tco2e),
                is_blend_component=False,
            )
            gas_emissions.append(emission)

        elapsed = time.monotonic() - t_start
        logger.debug(
            "Equipment-based calculation: %d profiles -> %d emissions in %.1fms",
            len(profiles),
            len(gas_emissions),
            elapsed * 1000,
        )

        return gas_emissions

    # ------------------------------------------------------------------
    # Public API: Mass Balance Calculation
    # ------------------------------------------------------------------

    def calculate_mass_balance(
        self,
        data: List[MassBalanceData],
        gwp_values: Dict[RefrigerantType, Decimal],
    ) -> List[GasEmission]:
        """Calculate emissions using the mass balance methodology.

        For each MassBalanceData record, computes:
            Loss = BI + Purchases - Sales + Acq - Divest - EI - CapChange
            Loss = max(0, Loss)
            Emissions_tCO2e = Loss x GWP / 1000

        Per EPA 40 CFR Part 98 Subpart OO equation OO-1.

        Args:
            data: List of MassBalanceData records, one per refrigerant
                type. Each record provides beginning/ending inventory,
                purchases, sales, acquisitions, divestitures, and
                capacity change.
            gwp_values: Dictionary mapping RefrigerantType to its GWP
                value as Decimal.

        Returns:
            List of GasEmission objects, one per MassBalanceData record
            with non-zero emissions.

        Raises:
            ValueError: If a required GWP value is missing for a
                refrigerant type present in data.
        """
        t_start = time.monotonic()
        gas_emissions: List[GasEmission] = []
        trace: List[str] = [
            "Mass balance calculation (EPA 40 CFR Part 98 Subpart OO)",
            f"Processing {len(data)} refrigerant balance(s)",
        ]

        for record in data:
            ref_type = record.refrigerant_type

            # Resolve GWP
            gwp = gwp_values.get(ref_type)
            if gwp is None:
                raise ValueError(
                    f"GWP value not provided for refrigerant type "
                    f"'{ref_type.value}' in mass balance data"
                )
            gwp = Decimal(str(gwp))

            # Convert to Decimal
            bi = Decimal(str(record.beginning_inventory_kg))
            purchases = Decimal(str(record.purchases_kg))
            sales = Decimal(str(record.sales_kg))
            acq = Decimal(str(record.acquisitions_kg))
            divest = Decimal(str(record.divestitures_kg))
            ei = Decimal(str(record.ending_inventory_kg))
            cap_change = Decimal(str(record.capacity_change_kg))

            # Mass balance equation
            loss_kg = bi + purchases - sales + acq - divest - ei - cap_change

            trace.append(
                f"  {ref_type.value}: BI={bi} + Purchases={purchases} "
                f"- Sales={sales} + Acq={acq} - Divest={divest} "
                f"- EI={ei} - CapChange={cap_change} = raw_loss={loss_kg}"
            )

            # Clamp to zero (negative loss means net accumulation)
            if loss_kg < Decimal("0"):
                trace.append(
                    f"    Clamped negative loss {loss_kg} to 0"
                )
                loss_kg = Decimal("0")

            loss_kg = self._apply_precision(loss_kg)

            # Calculate emissions
            emissions_kg_co2e = loss_kg * gwp
            emissions_kg_co2e = self._apply_precision(emissions_kg_co2e)

            emissions_tco2e = loss_kg * gwp * _KG_TO_TONNE
            emissions_tco2e = self._apply_precision(emissions_tco2e)

            trace.append(
                f"    loss={loss_kg} kg x GWP={gwp} / 1000 "
                f"= {emissions_tco2e} tCO2e"
            )

            gas_name = _GAS_NAMES.get(ref_type, ref_type.value)
            emission = GasEmission(
                refrigerant_type=ref_type,
                gas_name=gas_name,
                loss_kg=float(loss_kg),
                gwp_applied=float(gwp),
                gwp_source=self._gwp_source_label,
                emissions_kg_co2e=float(emissions_kg_co2e),
                emissions_tco2e=float(emissions_tco2e),
                is_blend_component=False,
            )
            gas_emissions.append(emission)

        elapsed = time.monotonic() - t_start
        logger.debug(
            "Mass balance calculation: %d records -> %d emissions in %.1fms",
            len(data),
            len(gas_emissions),
            elapsed * 1000,
        )

        return gas_emissions

    # ------------------------------------------------------------------
    # Public API: Screening Calculation
    # ------------------------------------------------------------------

    def calculate_screening(
        self,
        total_charge_kg: Decimal,
        ref_type: RefrigerantType,
        equip_type: EquipmentType,
        leak_rate: Decimal,
        gwp: Decimal,
    ) -> List[GasEmission]:
        """Calculate emissions using the screening methodology.

        Simplified estimation using total charge and default leak rate:
            Loss = TotalCharge x DefaultLeakRate
            Emissions_tCO2e = Loss x GWP / 1000

        Args:
            total_charge_kg: Total refrigerant charge in kilograms.
                Must be > 0.
            ref_type: Refrigerant type for the charge.
            equip_type: Equipment type classification for default rate
                selection and reporting.
            leak_rate: Annual leak rate as a Decimal fraction (0.0-1.0).
            gwp: GWP value as Decimal.

        Returns:
            List containing a single GasEmission object.

        Raises:
            ValueError: If total_charge_kg <= 0, leak_rate not in
                [0, 1], or gwp < 0.
        """
        t_start = time.monotonic()

        # Validate inputs
        total_charge_kg = Decimal(str(total_charge_kg))
        leak_rate = Decimal(str(leak_rate))
        gwp = Decimal(str(gwp))

        if total_charge_kg <= Decimal("0"):
            raise ValueError(
                f"total_charge_kg must be > 0, got {total_charge_kg}"
            )
        if leak_rate < Decimal("0") or leak_rate > Decimal("1"):
            raise ValueError(
                f"leak_rate must be in [0, 1], got {leak_rate}"
            )
        if gwp < Decimal("0"):
            raise ValueError(f"gwp must be >= 0, got {gwp}")

        # Calculate loss
        loss_kg = total_charge_kg * leak_rate
        loss_kg = self._apply_precision(loss_kg)

        # Calculate emissions
        emissions_kg_co2e = loss_kg * gwp
        emissions_kg_co2e = self._apply_precision(emissions_kg_co2e)

        emissions_tco2e = loss_kg * gwp * _KG_TO_TONNE
        emissions_tco2e = self._apply_precision(emissions_tco2e)

        gas_name = _GAS_NAMES.get(ref_type, ref_type.value)
        emission = GasEmission(
            refrigerant_type=ref_type,
            gas_name=gas_name,
            loss_kg=float(loss_kg),
            gwp_applied=float(gwp),
            gwp_source=self._gwp_source_label,
            emissions_kg_co2e=float(emissions_kg_co2e),
            emissions_tco2e=float(emissions_tco2e),
            is_blend_component=False,
        )

        elapsed = time.monotonic() - t_start
        logger.debug(
            "Screening calculation: charge=%.1f kg x leak_rate=%.4f "
            "x GWP=%s = %.3f tCO2e in %.1fms",
            total_charge_kg,
            leak_rate,
            gwp,
            emissions_tco2e,
            elapsed * 1000,
        )

        return [emission]

    # ------------------------------------------------------------------
    # Public API: Direct Measurement Calculation
    # ------------------------------------------------------------------

    def calculate_direct_measurement(
        self,
        loss_kg: Decimal,
        ref_type: RefrigerantType,
        gwp: Decimal,
    ) -> List[GasEmission]:
        """Calculate emissions from direct measurement data.

        Uses instrument-measured refrigerant loss:
            Emissions_tCO2e = MeasuredLoss x GWP / 1000

        Args:
            loss_kg: Measured refrigerant loss in kilograms. Must be >= 0.
            ref_type: Refrigerant type for the measured loss.
            gwp: GWP value as Decimal.

        Returns:
            List containing a single GasEmission object.

        Raises:
            ValueError: If loss_kg < 0 or gwp < 0.
        """
        t_start = time.monotonic()

        loss_kg = Decimal(str(loss_kg))
        gwp = Decimal(str(gwp))

        if loss_kg < Decimal("0"):
            raise ValueError(f"loss_kg must be >= 0, got {loss_kg}")
        if gwp < Decimal("0"):
            raise ValueError(f"gwp must be >= 0, got {gwp}")

        loss_kg = self._apply_precision(loss_kg)

        # Calculate emissions
        emissions_kg_co2e = loss_kg * gwp
        emissions_kg_co2e = self._apply_precision(emissions_kg_co2e)

        emissions_tco2e = loss_kg * gwp * _KG_TO_TONNE
        emissions_tco2e = self._apply_precision(emissions_tco2e)

        gas_name = _GAS_NAMES.get(ref_type, ref_type.value)
        emission = GasEmission(
            refrigerant_type=ref_type,
            gas_name=gas_name,
            loss_kg=float(loss_kg),
            gwp_applied=float(gwp),
            gwp_source=self._gwp_source_label,
            emissions_kg_co2e=float(emissions_kg_co2e),
            emissions_tco2e=float(emissions_tco2e),
            is_blend_component=False,
        )

        elapsed = time.monotonic() - t_start
        logger.debug(
            "Direct measurement calculation: loss=%.3f kg "
            "x GWP=%s = %.3f tCO2e in %.1fms",
            loss_kg,
            gwp,
            emissions_tco2e,
            elapsed * 1000,
        )

        return [emission]

    # ------------------------------------------------------------------
    # Public API: Top-Down Calculation
    # ------------------------------------------------------------------

    def calculate_top_down(
        self,
        purchases_kg: Decimal,
        recovered_kg: Decimal,
        ref_type: RefrigerantType,
        gwp: Decimal,
    ) -> List[GasEmission]:
        """Calculate emissions using the top-down methodology.

        Organizational aggregate from purchase and recovery data:
            Emissions_tCO2e = (PurchasesTotal - RecoveredTotal) x GWP / 1000

        Args:
            purchases_kg: Total refrigerant purchased in kilograms during
                the reporting period. Must be >= 0.
            recovered_kg: Total refrigerant recovered in kilograms during
                the reporting period. Must be >= 0.
            ref_type: Refrigerant type for the top-down calculation.
            gwp: GWP value as Decimal.

        Returns:
            List containing a single GasEmission object.

        Raises:
            ValueError: If purchases_kg < 0, recovered_kg < 0, or gwp < 0.
        """
        t_start = time.monotonic()

        purchases_kg = Decimal(str(purchases_kg))
        recovered_kg = Decimal(str(recovered_kg))
        gwp = Decimal(str(gwp))

        if purchases_kg < Decimal("0"):
            raise ValueError(
                f"purchases_kg must be >= 0, got {purchases_kg}"
            )
        if recovered_kg < Decimal("0"):
            raise ValueError(
                f"recovered_kg must be >= 0, got {recovered_kg}"
            )
        if gwp < Decimal("0"):
            raise ValueError(f"gwp must be >= 0, got {gwp}")

        # Net loss (clamp to zero if recovered exceeds purchases)
        net_loss_kg = purchases_kg - recovered_kg
        if net_loss_kg < Decimal("0"):
            net_loss_kg = Decimal("0")

        net_loss_kg = self._apply_precision(net_loss_kg)

        # Calculate emissions
        emissions_kg_co2e = net_loss_kg * gwp
        emissions_kg_co2e = self._apply_precision(emissions_kg_co2e)

        emissions_tco2e = net_loss_kg * gwp * _KG_TO_TONNE
        emissions_tco2e = self._apply_precision(emissions_tco2e)

        gas_name = _GAS_NAMES.get(ref_type, ref_type.value)
        emission = GasEmission(
            refrigerant_type=ref_type,
            gas_name=gas_name,
            loss_kg=float(net_loss_kg),
            gwp_applied=float(gwp),
            gwp_source=self._gwp_source_label,
            emissions_kg_co2e=float(emissions_kg_co2e),
            emissions_tco2e=float(emissions_tco2e),
            is_blend_component=False,
        )

        elapsed = time.monotonic() - t_start
        logger.debug(
            "Top-down calculation: purchases=%.1f - recovered=%.1f "
            "= net_loss=%.3f kg x GWP=%s = %.3f tCO2e in %.1fms",
            purchases_kg,
            recovered_kg,
            net_loss_kg,
            gwp,
            emissions_tco2e,
            elapsed * 1000,
        )

        return [emission]

    # ------------------------------------------------------------------
    # Public API: Batch Calculation
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        inputs: List[Dict[str, Any]],
    ) -> BatchCalculationResponse:
        """Execute a batch of calculations and aggregate results.

        Each input dictionary must contain a ``method`` key with a
        CalculationMethod enum value and the corresponding keyword
        arguments for that method.

        Args:
            inputs: List of dictionaries, each containing:
                - ``method``: CalculationMethod enum value.
                - All other keys are forwarded as kwargs to calculate().

        Returns:
            BatchCalculationResponse with per-calculation results,
            success/failure counts, and batch-level totals.
        """
        t_start = time.monotonic()
        batch_results: List[CalculationResult] = []
        success_count = 0
        failure_count = 0

        if _METRICS_AVAILABLE and _observe_batch_size is not None:
            try:
                methods_in_batch = set()
                for inp in inputs:
                    m = inp.get("method")
                    if isinstance(m, CalculationMethod):
                        methods_in_batch.add(m.value)
                batch_method = (
                    list(methods_in_batch)[0]
                    if len(methods_in_batch) == 1
                    else "mixed"
                )
                _observe_batch_size(batch_method, len(inputs))
            except Exception:
                pass

        for i, inp in enumerate(inputs):
            try:
                method = inp.pop("method") if "method" in inp else None
                if method is None:
                    raise ValueError(
                        f"Batch input[{i}] missing required 'method' key"
                    )
                if isinstance(method, str):
                    method = CalculationMethod(method)

                result = self.calculate(method=method, **inp)
                batch_results.append(result)
                success_count += 1
            except Exception as exc:
                failure_count += 1
                logger.warning(
                    "Batch calculation[%d] failed: %s", i, exc,
                )
                # Create a failed result placeholder
                failed_result = CalculationResult(
                    calculation_id=f"calc_{uuid4().hex[:12]}",
                    method=method if isinstance(method, CalculationMethod) else CalculationMethod.EQUIPMENT_BASED,
                    results=[],
                    total_loss_kg=0.0,
                    total_emissions_tco2e=0.0,
                    provenance_hash="",
                    timestamp=_utcnow(),
                    calculation_trace=[f"FAILED: {str(exc)}"],
                )
                batch_results.append(failed_result)

        # Compute batch totals
        total_emissions_tco2e = Decimal("0")
        for r in batch_results:
            total_emissions_tco2e += Decimal(str(r.total_emissions_tco2e))
        total_emissions_tco2e = self._apply_precision(total_emissions_tco2e)

        elapsed_ms = (time.monotonic() - t_start) * 1000

        # Batch provenance hash
        batch_prov_data = {
            "batch_size": len(inputs),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_emissions_tco2e": str(total_emissions_tco2e),
            "result_hashes": [r.provenance_hash for r in batch_results],
        }
        batch_hash = self._compute_hash(batch_prov_data)

        response = BatchCalculationResponse(
            results=batch_results,
            total_emissions_tco2e=float(total_emissions_tco2e),
            success_count=success_count,
            failure_count=failure_count,
            processing_time_ms=round(elapsed_ms, 1),
            provenance_hash=batch_hash,
        )

        # Record batch metrics
        if _METRICS_AVAILABLE and _record_batch is not None:
            try:
                status = "completed" if failure_count == 0 else "partial"
                _record_batch(status)
            except Exception:
                pass

        # Record provenance
        self._record_provenance(
            entity_type="calculation",
            action="batch",
            entity_id=f"batch_{uuid4().hex[:12]}",
            data=batch_prov_data,
            metadata={
                "batch_size": len(inputs),
                "success": success_count,
                "failure": failure_count,
            },
        )

        logger.info(
            "Batch calculation completed: %d inputs, %d success, %d failed, "
            "total=%.3f tCO2e in %.1fms",
            len(inputs),
            success_count,
            failure_count,
            total_emissions_tco2e,
            elapsed_ms,
        )

        return response

    # ------------------------------------------------------------------
    # Public API: Aggregation
    # ------------------------------------------------------------------

    def aggregate_results(
        self,
        results: List[CalculationResult],
        control_approach: str = "operational",
        share: Decimal = Decimal("1.0"),
    ) -> Dict[str, Any]:
        """Aggregate multiple calculation results with control approach
        and ownership share.

        Supports GHG Protocol organizational boundary approaches:
            - ``operational``: 100% of operations under operational control.
            - ``financial``: Proportional to financial interest (equity share).
            - ``equity``: Proportional to equity ownership.

        Args:
            results: List of CalculationResult objects to aggregate.
            control_approach: Boundary approach: "operational",
                "financial", or "equity". Defaults to "operational".
            share: Ownership or control share as Decimal (0 to 1).
                Defaults to 1.0 (100% for operational control).

        Returns:
            Dictionary containing:
                - ``total_emissions_tco2e``: Aggregated total after share.
                - ``total_loss_kg``: Aggregated total loss in kg.
                - ``control_approach``: The approach used.
                - ``share``: The share applied.
                - ``result_count``: Number of results aggregated.
                - ``by_refrigerant``: Per-refrigerant breakdown.
                - ``by_method``: Per-method breakdown.
                - ``provenance_hash``: SHA-256 hash of the aggregation.
        """
        share = Decimal(str(share))
        if share < Decimal("0") or share > Decimal("1"):
            raise ValueError(f"share must be in [0, 1], got {share}")

        valid_approaches = {"operational", "financial", "equity"}
        if control_approach not in valid_approaches:
            raise ValueError(
                f"control_approach must be one of {sorted(valid_approaches)}, "
                f"got '{control_approach}'"
            )

        total_emissions = Decimal("0")
        total_loss = Decimal("0")
        by_refrigerant: Dict[str, Dict[str, Decimal]] = {}
        by_method: Dict[str, Dict[str, Decimal]] = {}

        for result in results:
            for gas in result.results:
                ref_key = gas.refrigerant_type.value
                method_key = result.method.value

                loss = Decimal(str(gas.loss_kg))
                emissions = Decimal(str(gas.emissions_tco2e))

                total_emissions += emissions
                total_loss += loss

                # By refrigerant
                if ref_key not in by_refrigerant:
                    by_refrigerant[ref_key] = {
                        "loss_kg": Decimal("0"),
                        "emissions_tco2e": Decimal("0"),
                    }
                by_refrigerant[ref_key]["loss_kg"] += loss
                by_refrigerant[ref_key]["emissions_tco2e"] += emissions

                # By method
                if method_key not in by_method:
                    by_method[method_key] = {
                        "loss_kg": Decimal("0"),
                        "emissions_tco2e": Decimal("0"),
                    }
                by_method[method_key]["loss_kg"] += loss
                by_method[method_key]["emissions_tco2e"] += emissions

        # Apply ownership share
        total_emissions = self._apply_precision(total_emissions * share)
        total_loss = self._apply_precision(total_loss * share)

        by_ref_serialized: Dict[str, Dict[str, str]] = {}
        for ref_key, vals in by_refrigerant.items():
            by_ref_serialized[ref_key] = {
                "loss_kg": str(self._apply_precision(vals["loss_kg"] * share)),
                "emissions_tco2e": str(self._apply_precision(vals["emissions_tco2e"] * share)),
            }

        by_method_serialized: Dict[str, Dict[str, str]] = {}
        for method_key, vals in by_method.items():
            by_method_serialized[method_key] = {
                "loss_kg": str(self._apply_precision(vals["loss_kg"] * share)),
                "emissions_tco2e": str(self._apply_precision(vals["emissions_tco2e"] * share)),
            }

        # Provenance hash
        prov_data = {
            "control_approach": control_approach,
            "share": str(share),
            "result_count": len(results),
            "total_emissions_tco2e": str(total_emissions),
            "total_loss_kg": str(total_loss),
        }
        provenance_hash = self._compute_hash(prov_data)

        # Record provenance
        self._record_provenance(
            entity_type="calculation",
            action="aggregate",
            entity_id=f"agg_{uuid4().hex[:12]}",
            data=prov_data,
            metadata={"control_approach": control_approach},
        )

        logger.info(
            "Aggregation completed: %d results, approach=%s, share=%s, "
            "total=%.3f tCO2e",
            len(results),
            control_approach,
            share,
            total_emissions,
        )

        return {
            "total_emissions_tco2e": str(total_emissions),
            "total_loss_kg": str(total_loss),
            "control_approach": control_approach,
            "share": str(share),
            "result_count": len(results),
            "by_refrigerant": by_ref_serialized,
            "by_method": by_method_serialized,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Public API: History and Stats
    # ------------------------------------------------------------------

    def get_history(
        self,
        method: Optional[CalculationMethod] = None,
        limit: Optional[int] = None,
    ) -> List[CalculationResult]:
        """Return calculation history, optionally filtered.

        Args:
            method: Optional filter by calculation method.
            limit: Optional maximum number of recent entries.

        Returns:
            List of CalculationResult objects, oldest first.
        """
        with self._lock:
            entries = list(self._calculation_history)

        if method is not None:
            entries = [e for e in entries if e.method == method]

        if limit is not None and limit > 0 and len(entries) > limit:
            entries = entries[-limit:]

        return entries

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary with counts and operational statistics.
        """
        with self._lock:
            history_count = len(self._calculation_history)
            active = self._active_count

            by_method: Dict[str, int] = {}
            total_emissions = Decimal("0")
            for entry in self._calculation_history:
                method_key = entry.method.value
                by_method[method_key] = by_method.get(method_key, 0) + 1
                total_emissions += Decimal(str(entry.total_emissions_tco2e))

        return {
            "total_calculations": history_count,
            "active_calculations": active,
            "total_emissions_tco2e": str(
                self._apply_precision(total_emissions)
            ),
            "calculations_by_method": by_method,
            "precision": self._precision,
            "gwp_source": self._gwp_source_label,
        }

    def clear(self) -> None:
        """Clear calculation history.

        Intended for testing and engine reset scenarios.
        """
        with self._lock:
            self._calculation_history.clear()
            self._active_count = 0
        logger.info("EmissionCalculatorEngine cleared")

    # ------------------------------------------------------------------
    # Private: Route Helpers
    # ------------------------------------------------------------------

    def _route_equipment_based(self, **kwargs: Any) -> List[GasEmission]:
        """Route kwargs to calculate_equipment_based."""
        profiles = kwargs.get("profiles")
        gwp_values = kwargs.get("gwp_values")
        leak_rates = kwargs.get("leak_rates")
        if profiles is None or gwp_values is None:
            raise ValueError(
                "Equipment-based calculation requires 'profiles' and "
                "'gwp_values' parameters"
            )
        return self.calculate_equipment_based(
            profiles=profiles,
            gwp_values=gwp_values,
            leak_rates=leak_rates,
        )

    def _route_mass_balance(self, **kwargs: Any) -> List[GasEmission]:
        """Route kwargs to calculate_mass_balance."""
        data = kwargs.get("data")
        gwp_values = kwargs.get("gwp_values")
        if data is None or gwp_values is None:
            raise ValueError(
                "Mass balance calculation requires 'data' and "
                "'gwp_values' parameters"
            )
        return self.calculate_mass_balance(
            data=data,
            gwp_values=gwp_values,
        )

    def _route_screening(self, **kwargs: Any) -> List[GasEmission]:
        """Route kwargs to calculate_screening."""
        total_charge_kg = kwargs.get("total_charge_kg")
        ref_type = kwargs.get("ref_type")
        equip_type = kwargs.get("equip_type")
        leak_rate = kwargs.get("leak_rate")
        gwp = kwargs.get("gwp")
        if any(v is None for v in [total_charge_kg, ref_type, equip_type, leak_rate, gwp]):
            raise ValueError(
                "Screening calculation requires 'total_charge_kg', "
                "'ref_type', 'equip_type', 'leak_rate', and 'gwp' parameters"
            )
        return self.calculate_screening(
            total_charge_kg=Decimal(str(total_charge_kg)),
            ref_type=ref_type,
            equip_type=equip_type,
            leak_rate=Decimal(str(leak_rate)),
            gwp=Decimal(str(gwp)),
        )

    def _route_direct_measurement(self, **kwargs: Any) -> List[GasEmission]:
        """Route kwargs to calculate_direct_measurement."""
        loss_kg = kwargs.get("loss_kg")
        ref_type = kwargs.get("ref_type")
        gwp = kwargs.get("gwp")
        if any(v is None for v in [loss_kg, ref_type, gwp]):
            raise ValueError(
                "Direct measurement calculation requires 'loss_kg', "
                "'ref_type', and 'gwp' parameters"
            )
        return self.calculate_direct_measurement(
            loss_kg=Decimal(str(loss_kg)),
            ref_type=ref_type,
            gwp=Decimal(str(gwp)),
        )

    def _route_top_down(self, **kwargs: Any) -> List[GasEmission]:
        """Route kwargs to calculate_top_down."""
        purchases_kg = kwargs.get("purchases_kg")
        recovered_kg = kwargs.get("recovered_kg")
        ref_type = kwargs.get("ref_type")
        gwp = kwargs.get("gwp")
        if any(v is None for v in [purchases_kg, recovered_kg, ref_type, gwp]):
            raise ValueError(
                "Top-down calculation requires 'purchases_kg', "
                "'recovered_kg', 'ref_type', and 'gwp' parameters"
            )
        return self.calculate_top_down(
            purchases_kg=Decimal(str(purchases_kg)),
            recovered_kg=Decimal(str(recovered_kg)),
            ref_type=ref_type,
            gwp=Decimal(str(gwp)),
        )

    # ------------------------------------------------------------------
    # Private: Leak Rate Resolution
    # ------------------------------------------------------------------

    def _resolve_leak_rate(
        self,
        profile: EquipmentProfile,
        leak_rates: Optional[Dict[str, Decimal]] = None,
    ) -> Decimal:
        """Resolve leak rate for an equipment profile.

        Resolution order:
        1. Custom leak_rates dict keyed by equipment_id.
        2. Custom leak_rates dict keyed by equipment_type value.
        3. Profile's custom_leak_rate field.
        4. Default leak rate for the equipment type.

        Args:
            profile: Equipment profile to resolve for.
            leak_rates: Optional custom leak rates dictionary.

        Returns:
            Decimal leak rate as a fraction (0.0-1.0).
        """
        # 1. Lookup by equipment_id in custom dict
        if leak_rates is not None:
            if profile.equipment_id in leak_rates:
                return Decimal(str(leak_rates[profile.equipment_id]))

            # 2. Lookup by equipment_type value in custom dict
            et_key = profile.equipment_type.value
            if et_key in leak_rates:
                return Decimal(str(leak_rates[et_key]))

        # 3. Profile's custom_leak_rate
        if profile.custom_leak_rate is not None:
            return Decimal(str(profile.custom_leak_rate))

        # 4. Default for equipment type
        default_rate = _DEFAULT_EQUIPMENT_LEAK_RATES.get(
            profile.equipment_type
        )
        if default_rate is not None:
            return default_rate

        # Absolute fallback
        return Decimal("0.10")

    # ------------------------------------------------------------------
    # Private: Precision and Hashing
    # ------------------------------------------------------------------

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply regulatory rounding precision to a Decimal value.

        Uses ROUND_HALF_UP for consistency with GHG Protocol and
        regulatory reporting requirements.

        Args:
            value: Decimal value to round.

        Returns:
            Rounded Decimal value.
        """
        quantize_str = "0." + "0" * self._precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute SHA-256 hash for provenance data.

        Args:
            data: JSON-serializable data to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Private: Provenance and Metrics
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record provenance entry if provenance tracking is available."""
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type=entity_type,
                    action=action,
                    entity_id=entity_id,
                    data=data,
                    metadata=metadata,
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

    def _record_calculation_metrics(
        self,
        method: str,
        gas_emissions: List[GasEmission],
        status: str,
        elapsed: float,
    ) -> None:
        """Record Prometheus metrics for a completed calculation."""
        if not _METRICS_AVAILABLE:
            return

        # Record per-gas calculation metrics
        for gas in gas_emissions:
            ref_type_str = gas.refrigerant_type.value
            category = _CATEGORY_MAP.get(
                gas.refrigerant_type, "other"
            )

            if _record_calculation is not None:
                try:
                    _record_calculation(method, ref_type_str, status)
                except Exception:
                    pass

            if _record_emissions is not None:
                try:
                    _record_emissions(
                        ref_type_str, category, gas.emissions_kg_co2e
                    )
                except Exception:
                    pass

        # Record duration
        if _observe_calculation_duration is not None:
            try:
                _observe_calculation_duration(
                    "single_calculation", elapsed
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        with self._lock:
            hist_count = len(self._calculation_history)
            active = self._active_count
        return (
            f"EmissionCalculatorEngine("
            f"precision={self._precision}, "
            f"gwp_source={self._gwp_source_label!r}, "
            f"calculations={hist_count}, "
            f"active={active})"
        )

    def __len__(self) -> int:
        """Return the number of calculations performed."""
        with self._lock:
            return len(self._calculation_history)
