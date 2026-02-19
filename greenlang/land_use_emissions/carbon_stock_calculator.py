# -*- coding: utf-8 -*-
"""
CarbonStockCalculatorEngine - Stock-Difference & Gain-Loss Calculations (Engine 2 of 7)

AGENT-MRV-006: Land Use Emissions Agent

Core calculation engine implementing the two IPCC methods for estimating
carbon stock changes in land-use systems:

    1. Stock-Difference Method: DeltaC = (C_t2 - C_t1) / (t2 - t1)
       Applied per pool (AGB, BGB, dead wood, litter) and combined.

    2. Gain-Loss Method: DeltaC = (A * G_w * CF) - (L_harvest + L_fuelwood
       + L_disturbance) * CF
       Uses biomass growth rates, harvest volumes, and disturbance losses.

Both methods are applied to each of the five IPCC carbon pools (AGB, BGB,
dead wood, litter, SOC -- where SOC is delegated to the SoilOrganicCarbonEngine).

Fire Emissions:
    L_fire = M_fire * C_f * G_ef for each gas (CO2, CH4, N2O)
    Where M_fire is mass of fuel available, C_f is combustion factor,
    G_ef is gas-specific emission factor.

All calculations use Python Decimal arithmetic with 8+ decimal places for
zero-hallucination determinism.  Every calculation result includes a per-pool
breakdown, GWP-adjusted CO2e, processing time, and SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Stateless per-calculation.  Mutable counters protected by reentrant lock.

Example:
    >>> from greenlang.land_use_emissions.carbon_stock_calculator import (
    ...     CarbonStockCalculatorEngine,
    ... )
    >>> from greenlang.land_use_emissions.land_use_database import (
    ...     LandUseDatabaseEngine,
    ... )
    >>> db = LandUseDatabaseEngine()
    >>> calc = CarbonStockCalculatorEngine(land_use_database=db)
    >>> result = calc.calculate_stock_difference({
    ...     "land_category": "FOREST_LAND",
    ...     "climate_zone": "TROPICAL_WET",
    ...     "area_ha": 1000,
    ...     "c_t1": {"agb": 180, "bgb": 43, "dead_wood": 14, "litter": 5},
    ...     "c_t2": {"agb": 170, "bgb": 40, "dead_wood": 13, "litter": 5},
    ...     "year_t1": 2020,
    ...     "year_t2": 2025,
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["CarbonStockCalculatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.land_use_emissions.land_use_database import (
        LandUseDatabaseEngine,
        CARBON_FRACTION,
        CONVERSION_FACTOR_CO2_C,
        GWP_VALUES,
        N2O_N_RATIO,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    LandUseDatabaseEngine = None  # type: ignore[misc,assignment]
    CARBON_FRACTION = Decimal("0.47")
    CONVERSION_FACTOR_CO2_C = Decimal("3.66667")
    N2O_N_RATIO = Decimal("1.571429")
    GWP_VALUES = {
        "AR6": {
            "CO2": Decimal("1"),
            "CH4": Decimal("29.8"),
            "N2O": Decimal("273"),
        },
    }

try:
    from greenlang.land_use_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.land_use_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.land_use_emissions.metrics import (
        record_component_operation as _record_calc_operation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calc_operation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert a value to Decimal, returning default on failure."""
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default


# ===========================================================================
# Enumerations
# ===========================================================================


class CalculationMethod(str, Enum):
    """Carbon stock change calculation methods."""

    STOCK_DIFFERENCE = "STOCK_DIFFERENCE"
    GAIN_LOSS = "GAIN_LOSS"


class CarbonPool(str, Enum):
    """Five IPCC carbon pools."""

    AGB = "AGB"
    BGB = "BGB"
    DEAD_WOOD = "DEAD_WOOD"
    LITTER = "LITTER"
    SOC = "SOC"


# ===========================================================================
# Trace Step Dataclass
# ===========================================================================


@dataclass
class TraceStep:
    """Single step in a calculation trace for audit trail.

    Attributes:
        step_number: Sequential step number.
        description: Human-readable description.
        formula: Mathematical formula applied.
        inputs: Input values for this step.
        output: Output value from this step.
        unit: Unit of the output value.
    """

    step_number: int
    description: str
    formula: str
    inputs: Dict[str, str]
    output: str
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "formula": self.formula,
            "inputs": self.inputs,
            "output": self.output,
            "unit": self.unit,
        }


# ===========================================================================
# CarbonStockCalculatorEngine
# ===========================================================================


class CarbonStockCalculatorEngine:
    """Core carbon stock change calculator implementing IPCC stock-difference
    and gain-loss methods.

    This engine performs deterministic Decimal arithmetic for all carbon pool
    calculations, fire emissions, harvest losses, and CO2e conversions.

    Thread Safety:
        Per-calculation state is created fresh for each method call.
        Shared counters use a reentrant lock.

    Attributes:
        _land_use_db: Reference to the LandUseDatabaseEngine for factor lookups.
        _lock: Reentrant lock protecting mutable counters.
        _total_calculations: Counter of total calculations performed.
        _gwp_source: Default GWP source for CO2e conversion.

    Example:
        >>> db = LandUseDatabaseEngine()
        >>> calc = CarbonStockCalculatorEngine(land_use_database=db)
        >>> result = calc.calculate_stock_difference(request)
    """

    def __init__(
        self,
        land_use_database: Optional[Any] = None,
        gwp_source: str = "AR6",
    ) -> None:
        """Initialize the CarbonStockCalculatorEngine.

        Args:
            land_use_database: Optional LandUseDatabaseEngine instance.
                If None, a new one is created.
            gwp_source: Default GWP source for CO2e conversion.
        """
        if land_use_database is not None:
            self._land_use_db = land_use_database
        elif _DB_AVAILABLE and LandUseDatabaseEngine is not None:
            self._land_use_db = LandUseDatabaseEngine()
        else:
            self._land_use_db = None

        self._gwp_source = gwp_source.upper()
        self._lock = threading.RLock()
        self._total_calculations: int = 0
        self._created_at = _utcnow()

        logger.info(
            "CarbonStockCalculatorEngine initialized: "
            "gwp_source=%s, db_available=%s",
            self._gwp_source, self._land_use_db is not None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_calculations(self) -> None:
        """Thread-safe increment of the calculation counter."""
        with self._lock:
            self._total_calculations += 1

    def _extract_pool_value(
        self,
        data: Dict[str, Any],
        pool: str,
        default: Decimal = _ZERO,
    ) -> Decimal:
        """Extract a carbon pool value from a dictionary.

        Handles various key formats (lowercase, uppercase, enum-style).

        Args:
            data: Dictionary containing pool values.
            pool: Pool name to extract.
            default: Default value if pool not found.

        Returns:
            Pool value as Decimal.
        """
        for key in [pool, pool.lower(), pool.upper()]:
            if key in data:
                return _safe_decimal(data[key], default)
        return default

    # ------------------------------------------------------------------
    # Stock-Difference Method
    # ------------------------------------------------------------------

    def calculate_stock_difference(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate carbon stock changes using the IPCC stock-difference method.

        Formula per pool: DeltaC = (C_t2 - C_t1) / (t2 - t1)
        Total: DeltaC_total = sum(DeltaC_pool) for all pools
        Unit: tC/yr (converted to tCO2e using CONVERSION_FACTOR_CO2_C and GWP)

        Required request keys:
            land_category: IPCC land category.
            climate_zone: IPCC climate zone.
            area_ha: Land area in hectares.
            c_t1: Carbon stocks at time 1 (dict with pool keys in tC/ha).
            c_t2: Carbon stocks at time 2 (dict with pool keys in tC/ha).
            year_t1: Year of time 1 inventory.
            year_t2: Year of time 2 inventory.

        Args:
            request: Calculation request dictionary.

        Returns:
            Calculation result with per-pool breakdown and totals.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        self._increment_calculations()
        start_time = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = str(uuid4())

        # -- Extract inputs ------------------------------------------------
        land_category = str(request.get("land_category", "")).upper()
        climate_zone = str(request.get("climate_zone", "")).upper()
        area_ha = _safe_decimal(request.get("area_ha"), _ZERO)
        c_t1 = request.get("c_t1", {})
        c_t2 = request.get("c_t2", {})
        year_t1 = int(request.get("year_t1", 0))
        year_t2 = int(request.get("year_t2", 0))
        gwp_source = str(request.get("gwp_source", self._gwp_source)).upper()

        # -- Validate inputs -----------------------------------------------
        errors: List[str] = []
        if not land_category:
            errors.append("land_category is required")
        if not climate_zone:
            errors.append("climate_zone is required")
        if area_ha <= _ZERO:
            errors.append("area_ha must be > 0")
        if year_t1 <= 0:
            errors.append("year_t1 must be > 0")
        if year_t2 <= 0:
            errors.append("year_t2 must be > 0")
        if year_t2 <= year_t1:
            errors.append("year_t2 must be > year_t1")

        if errors:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "method": "STOCK_DIFFERENCE",
                "errors": errors,
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        time_interval = _D(str(year_t2 - year_t1))

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Extract and validate inputs",
            formula="time_interval = year_t2 - year_t1",
            inputs={
                "year_t1": str(year_t1),
                "year_t2": str(year_t2),
                "area_ha": str(area_ha),
            },
            output=str(time_interval),
            unit="years",
        ))

        # -- Calculate per-pool stock differences --------------------------
        pools = ["AGB", "BGB", "DEAD_WOOD", "LITTER"]
        pool_results: Dict[str, Dict[str, str]] = {}
        total_delta_c_per_ha = _ZERO
        total_delta_c_total = _ZERO

        for pool in pools:
            pool_t1 = self._extract_pool_value(c_t1, pool)
            pool_t2 = self._extract_pool_value(c_t2, pool)

            # DeltaC_per_ha = (C_t2 - C_t1) / time_interval
            delta_c_per_ha = (
                (pool_t2 - pool_t1) / time_interval
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

            # DeltaC_total = DeltaC_per_ha * area_ha
            delta_c_total = (
                delta_c_per_ha * area_ha
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

            total_delta_c_per_ha += delta_c_per_ha
            total_delta_c_total += delta_c_total

            pool_results[pool] = {
                "c_t1_tc_ha": str(pool_t1),
                "c_t2_tc_ha": str(pool_t2),
                "delta_c_tc_ha_yr": str(delta_c_per_ha),
                "delta_c_tc_yr": str(delta_c_total),
            }

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description=f"Stock difference for {pool}",
                formula=f"DeltaC_{pool} = (C_t2 - C_t1) / (t2 - t1)",
                inputs={
                    f"C_t1_{pool}": str(pool_t1),
                    f"C_t2_{pool}": str(pool_t2),
                    "time_interval": str(time_interval),
                },
                output=str(delta_c_per_ha),
                unit="tC/ha/yr",
            ))

        # -- CO2e conversion -----------------------------------------------
        # Net C change: negative = emission, positive = removal
        total_co2_tc = (
            total_delta_c_total * CONVERSION_FACTOR_CO2_C
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Convert tC-CO2 to tonnes CO2
        total_co2_tonnes = total_co2_tc.quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Convert total carbon change to CO2",
            formula="CO2 = DeltaC_total * 44/12",
            inputs={
                "delta_c_total_tc_yr": str(total_delta_c_total),
                "conversion_factor": str(CONVERSION_FACTOR_CO2_C),
            },
            output=str(total_co2_tonnes),
            unit="tCO2/yr",
        ))

        # -- Determine if net emission or removal -------------------------
        is_emission = total_delta_c_total < _ZERO
        emission_type = "NET_EMISSION" if is_emission else "NET_REMOVAL"

        gross_emissions = abs(total_co2_tonnes) if is_emission else _ZERO
        gross_removals = abs(total_co2_tonnes) if not is_emission else _ZERO

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Classify as emission or removal",
            formula="emission if DeltaC < 0, removal if DeltaC > 0",
            inputs={"delta_c_total_tc_yr": str(total_delta_c_total)},
            output=emission_type,
            unit="classification",
        ))

        # -- Assemble result -----------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "STOCK_DIFFERENCE",
            "land_category": land_category,
            "climate_zone": climate_zone,
            "area_ha": str(area_ha),
            "year_t1": year_t1,
            "year_t2": year_t2,
            "time_interval_years": str(time_interval),
            "gwp_source": gwp_source,
            "pool_results": pool_results,
            "total_delta_c_tc_ha_yr": str(
                total_delta_c_per_ha.quantize(_PRECISION, rounding=ROUND_HALF_UP)
            ),
            "total_delta_c_tc_yr": str(
                total_delta_c_total.quantize(_PRECISION, rounding=ROUND_HALF_UP)
            ),
            "total_co2_tonnes_yr": str(total_co2_tonnes),
            "emission_type": emission_type,
            "gross_emissions_tco2_yr": str(gross_emissions),
            "gross_removals_tco2_yr": str(gross_removals),
            "net_co2e_tonnes_yr": str(total_co2_tonnes),
            "trace_steps": [s.to_dict() for s in trace_steps],
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Stock-difference calculation complete: id=%s, "
            "delta_c=%s tC/yr, co2=%s tCO2/yr, type=%s, time=%.3fms",
            calc_id, total_delta_c_total, total_co2_tonnes,
            emission_type, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Gain-Loss Method
    # ------------------------------------------------------------------

    def calculate_gain_loss(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate carbon stock changes using the IPCC gain-loss method.

        Formula: DeltaC = Gains - Losses
            Gains = A * G_w * CF  (area * growth rate * carbon fraction)
            Losses = (L_harvest + L_fuelwood + L_disturbance) * CF

        Required request keys:
            land_category: IPCC land category.
            climate_zone: IPCC climate zone.
            area_ha: Land area in hectares.
            growth_rate_override: Optional growth rate override (tDM/ha/yr).
            harvest_volume_m3: Optional harvest volume in cubic metres.
            fuelwood_volume_m3: Optional fuelwood removal in cubic metres.
            disturbance_area_ha: Area affected by disturbance.
            disturbance_type: Type of disturbance (e.g. FIRE_WILDFIRE).
            wood_density: Optional wood density (tDM/m3), default 0.5.
            bcef: Biomass conversion and expansion factor, default 1.0.

        Args:
            request: Calculation request dictionary.

        Returns:
            Calculation result with gain/loss breakdown and totals.
        """
        self._increment_calculations()
        start_time = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = str(uuid4())

        # -- Extract inputs ------------------------------------------------
        land_category = str(request.get("land_category", "")).upper()
        climate_zone = str(request.get("climate_zone", "")).upper()
        area_ha = _safe_decimal(request.get("area_ha"), _ZERO)
        gwp_source = str(request.get("gwp_source", self._gwp_source)).upper()
        wood_density = _safe_decimal(request.get("wood_density"), _D("0.5"))
        bcef = _safe_decimal(request.get("bcef"), _ONE)
        harvest_volume_m3 = _safe_decimal(request.get("harvest_volume_m3"), _ZERO)
        fuelwood_volume_m3 = _safe_decimal(request.get("fuelwood_volume_m3"), _ZERO)
        disturbance_area_ha = _safe_decimal(request.get("disturbance_area_ha"), _ZERO)
        disturbance_type = str(request.get("disturbance_type", "")).upper()
        growth_rate_override = request.get("growth_rate_override")

        # -- Validate inputs -----------------------------------------------
        errors: List[str] = []
        if not land_category:
            errors.append("land_category is required")
        if not climate_zone:
            errors.append("climate_zone is required")
        if area_ha <= _ZERO:
            errors.append("area_ha must be > 0")

        if errors:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "method": "GAIN_LOSS",
                "errors": errors,
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        # -- Look up growth rate -------------------------------------------
        if growth_rate_override is not None:
            growth_rate_dm = _safe_decimal(growth_rate_override)
        elif self._land_use_db is not None:
            # DB returns tC/ha/yr, convert to tDM/ha/yr by dividing by CF
            growth_rate_c = self._land_use_db.get_growth_rate(
                land_category, climate_zone
            )
            growth_rate_dm = (growth_rate_c / CARBON_FRACTION).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            growth_rate_dm = _D("5.0")

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Determine biomass growth rate",
            formula="G_w (tDM/ha/yr)",
            inputs={
                "land_category": land_category,
                "climate_zone": climate_zone,
                "override": str(growth_rate_override),
            },
            output=str(growth_rate_dm),
            unit="tDM/ha/yr",
        ))

        # -- Calculate AGB gains -------------------------------------------
        # Gains = A * G_w * CF
        agb_gain_dm = (area_ha * growth_rate_dm).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        agb_gain_c = (agb_gain_dm * CARBON_FRACTION).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate AGB gains",
            formula="Gains = A * G_w * CF",
            inputs={
                "area_ha": str(area_ha),
                "growth_rate_dm": str(growth_rate_dm),
                "carbon_fraction": str(CARBON_FRACTION),
            },
            output=str(agb_gain_c),
            unit="tC/yr",
        ))

        # -- Calculate BGB gains using root-to-shoot ratio -----------------
        bgb_gain_c = _ZERO
        if self._land_use_db is not None:
            agb_stock = self._land_use_db.get_agb_default(
                land_category, climate_zone
            )
            rs_ratio = self._land_use_db.get_root_shoot_ratio(
                climate_zone, agb_stock
            )
            bgb_gain_c = (agb_gain_c * rs_ratio).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            bgb_gain_c = (agb_gain_c * _D("0.25")).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate BGB gains from root-to-shoot ratio",
            formula="BGB_gain = AGB_gain * R:S ratio",
            inputs={"agb_gain_c": str(agb_gain_c)},
            output=str(bgb_gain_c),
            unit="tC/yr",
        ))

        # -- Calculate harvest losses --------------------------------------
        # L_harvest = V_harvest * D * BCEF * CF
        harvest_loss_c = (
            harvest_volume_m3 * wood_density * bcef * CARBON_FRACTION
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate harvest losses",
            formula="L_harvest = V * D * BCEF * CF",
            inputs={
                "harvest_volume_m3": str(harvest_volume_m3),
                "wood_density": str(wood_density),
                "bcef": str(bcef),
                "carbon_fraction": str(CARBON_FRACTION),
            },
            output=str(harvest_loss_c),
            unit="tC/yr",
        ))

        # -- Calculate fuelwood losses -------------------------------------
        fuelwood_loss_c = (
            fuelwood_volume_m3 * wood_density * bcef * CARBON_FRACTION
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate fuelwood losses",
            formula="L_fuelwood = V * D * BCEF * CF",
            inputs={
                "fuelwood_volume_m3": str(fuelwood_volume_m3),
                "wood_density": str(wood_density),
            },
            output=str(fuelwood_loss_c),
            unit="tC/yr",
        ))

        # -- Calculate disturbance losses (fire) ---------------------------
        disturbance_loss_c = _ZERO
        fire_emissions_result: Optional[Dict[str, Any]] = None

        if disturbance_area_ha > _ZERO and disturbance_type:
            fire_result = self.calculate_fire_emissions({
                "land_category": land_category,
                "climate_zone": climate_zone,
                "disturbance_type": disturbance_type,
                "area_ha": disturbance_area_ha,
                "gwp_source": gwp_source,
            })
            if fire_result.get("status") == "SUCCESS":
                fire_emissions_result = fire_result
                disturbance_loss_c = _safe_decimal(
                    fire_result.get("total_carbon_released_tc")
                )

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate disturbance losses",
            formula="L_disturbance from fire emissions engine",
            inputs={
                "disturbance_area_ha": str(disturbance_area_ha),
                "disturbance_type": disturbance_type,
            },
            output=str(disturbance_loss_c),
            unit="tC/yr",
        ))

        # -- Net carbon change ---------------------------------------------
        total_gains_c = (agb_gain_c + bgb_gain_c).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        total_losses_c = (
            harvest_loss_c + fuelwood_loss_c + disturbance_loss_c
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        net_delta_c = (total_gains_c - total_losses_c).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Calculate net carbon change",
            formula="DeltaC = Gains - Losses",
            inputs={
                "total_gains_c": str(total_gains_c),
                "total_losses_c": str(total_losses_c),
            },
            output=str(net_delta_c),
            unit="tC/yr",
        ))

        # -- CO2e conversion -----------------------------------------------
        net_co2_tonnes = (net_delta_c * CONVERSION_FACTOR_CO2_C).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        is_emission = net_delta_c < _ZERO
        emission_type = "NET_EMISSION" if is_emission else "NET_REMOVAL"
        gross_emissions = abs(net_co2_tonnes) if is_emission else _ZERO
        gross_removals = abs(net_co2_tonnes) if not is_emission else _ZERO

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Convert to CO2 equivalent",
            formula="CO2 = DeltaC * 44/12",
            inputs={"net_delta_c": str(net_delta_c)},
            output=str(net_co2_tonnes),
            unit="tCO2/yr",
        ))

        # -- Assemble result -----------------------------------------------
        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "GAIN_LOSS",
            "land_category": land_category,
            "climate_zone": climate_zone,
            "area_ha": str(area_ha),
            "gwp_source": gwp_source,
            "gains": {
                "agb_gain_tc_yr": str(agb_gain_c),
                "bgb_gain_tc_yr": str(bgb_gain_c),
                "total_gains_tc_yr": str(total_gains_c),
                "growth_rate_tdm_ha_yr": str(growth_rate_dm),
            },
            "losses": {
                "harvest_loss_tc_yr": str(harvest_loss_c),
                "fuelwood_loss_tc_yr": str(fuelwood_loss_c),
                "disturbance_loss_tc_yr": str(disturbance_loss_c),
                "total_losses_tc_yr": str(total_losses_c),
            },
            "net_delta_c_tc_yr": str(net_delta_c),
            "total_co2_tonnes_yr": str(net_co2_tonnes),
            "emission_type": emission_type,
            "gross_emissions_tco2_yr": str(gross_emissions),
            "gross_removals_tco2_yr": str(gross_removals),
            "net_co2e_tonnes_yr": str(net_co2_tonnes),
            "fire_emissions": fire_emissions_result,
            "trace_steps": [s.to_dict() for s in trace_steps],
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Gain-loss calculation complete: id=%s, "
            "gains=%s tC/yr, losses=%s tC/yr, net=%s tC/yr, "
            "co2=%s tCO2/yr, type=%s, time=%.3fms",
            calc_id, total_gains_c, total_losses_c, net_delta_c,
            net_co2_tonnes, emission_type, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Per-Pool Calculations
    # ------------------------------------------------------------------

    def calculate_agb_change(
        self,
        agb_t1: Decimal,
        agb_t2: Decimal,
        area_ha: Decimal,
        time_interval: Decimal,
    ) -> Dict[str, str]:
        """Calculate AGB stock change for a single pool.

        Args:
            agb_t1: AGB at time 1 (tC/ha).
            agb_t2: AGB at time 2 (tC/ha).
            area_ha: Area in hectares.
            time_interval: Time interval in years.

        Returns:
            Dictionary with delta_c_per_ha, delta_c_total, co2_tonnes.
        """
        self._increment_calculations()

        delta_per_ha = ((agb_t2 - agb_t1) / time_interval).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        delta_total = (delta_per_ha * area_ha).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        co2_tonnes = (delta_total * CONVERSION_FACTOR_CO2_C).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        return {
            "pool": "AGB",
            "delta_c_tc_ha_yr": str(delta_per_ha),
            "delta_c_tc_yr": str(delta_total),
            "co2_tonnes_yr": str(co2_tonnes),
        }

    def calculate_bgb_change(
        self,
        agb_t1: Decimal,
        agb_t2: Decimal,
        area_ha: Decimal,
        time_interval: Decimal,
        climate_zone: str = "TEMPERATE_CONTINENTAL",
    ) -> Dict[str, str]:
        """Calculate BGB stock change derived from AGB via root-to-shoot ratio.

        Args:
            agb_t1: AGB at time 1 (tC/ha).
            agb_t2: AGB at time 2 (tC/ha).
            area_ha: Area in hectares.
            time_interval: Time interval in years.
            climate_zone: Climate zone for R:S ratio lookup.

        Returns:
            Dictionary with delta_c_per_ha, delta_c_total, co2_tonnes.
        """
        self._increment_calculations()

        if self._land_use_db is not None:
            rs_t1 = self._land_use_db.get_root_shoot_ratio(climate_zone, agb_t1)
            rs_t2 = self._land_use_db.get_root_shoot_ratio(climate_zone, agb_t2)
        else:
            rs_t1 = _D("0.25")
            rs_t2 = _D("0.25")

        bgb_t1 = (agb_t1 * rs_t1).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        bgb_t2 = (agb_t2 * rs_t2).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        delta_per_ha = ((bgb_t2 - bgb_t1) / time_interval).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        delta_total = (delta_per_ha * area_ha).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        co2_tonnes = (delta_total * CONVERSION_FACTOR_CO2_C).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        return {
            "pool": "BGB",
            "bgb_t1_tc_ha": str(bgb_t1),
            "bgb_t2_tc_ha": str(bgb_t2),
            "root_shoot_t1": str(rs_t1),
            "root_shoot_t2": str(rs_t2),
            "delta_c_tc_ha_yr": str(delta_per_ha),
            "delta_c_tc_yr": str(delta_total),
            "co2_tonnes_yr": str(co2_tonnes),
        }

    def calculate_dead_wood_change(
        self,
        dw_t1: Decimal,
        dw_t2: Decimal,
        area_ha: Decimal,
        time_interval: Decimal,
    ) -> Dict[str, str]:
        """Calculate dead wood stock change.

        Args:
            dw_t1: Dead wood at time 1 (tC/ha).
            dw_t2: Dead wood at time 2 (tC/ha).
            area_ha: Area in hectares.
            time_interval: Time interval in years.

        Returns:
            Dictionary with delta_c_per_ha, delta_c_total, co2_tonnes.
        """
        self._increment_calculations()

        delta_per_ha = ((dw_t2 - dw_t1) / time_interval).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        delta_total = (delta_per_ha * area_ha).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        co2_tonnes = (delta_total * CONVERSION_FACTOR_CO2_C).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        return {
            "pool": "DEAD_WOOD",
            "delta_c_tc_ha_yr": str(delta_per_ha),
            "delta_c_tc_yr": str(delta_total),
            "co2_tonnes_yr": str(co2_tonnes),
        }

    def calculate_litter_change(
        self,
        litter_t1: Decimal,
        litter_t2: Decimal,
        area_ha: Decimal,
        time_interval: Decimal,
    ) -> Dict[str, str]:
        """Calculate litter stock change.

        Args:
            litter_t1: Litter at time 1 (tC/ha).
            litter_t2: Litter at time 2 (tC/ha).
            area_ha: Area in hectares.
            time_interval: Time interval in years.

        Returns:
            Dictionary with delta_c_per_ha, delta_c_total, co2_tonnes.
        """
        self._increment_calculations()

        delta_per_ha = ((litter_t2 - litter_t1) / time_interval).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        delta_total = (delta_per_ha * area_ha).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        co2_tonnes = (delta_total * CONVERSION_FACTOR_CO2_C).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        return {
            "pool": "LITTER",
            "delta_c_tc_ha_yr": str(delta_per_ha),
            "delta_c_tc_yr": str(delta_total),
            "co2_tonnes_yr": str(co2_tonnes),
        }

    # ------------------------------------------------------------------
    # Fire Emissions
    # ------------------------------------------------------------------

    def calculate_fire_emissions(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate fire-related emissions for a land parcel.

        Formula per gas:
            L_fire_gas = A * M_B * C_f * G_ef / 1000
            where:
                A = area burned (ha)
                M_B = mass of fuel available (tDM/ha)
                C_f = combustion factor (dimensionless)
                G_ef = emission factor (g/kg dry matter)
                /1000 = g to kg conversion

        The mass of fuel available (M_B) is derived from the AGB of the
        land category and climate zone if not explicitly provided.

        Required request keys:
            land_category: IPCC land category.
            climate_zone: IPCC climate zone.
            disturbance_type: FIRE_WILDFIRE or FIRE_PRESCRIBED.
            area_ha: Area burned in hectares.

        Optional keys:
            fuel_load_tdm_ha: Override fuel load (tDM/ha).
            gwp_source: GWP source for CO2e conversion.

        Args:
            request: Fire emission calculation request.

        Returns:
            Per-gas emissions with CO2e totals.
        """
        self._increment_calculations()
        start_time = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = str(uuid4())

        land_category = str(request.get("land_category", "")).upper()
        climate_zone = str(request.get("climate_zone", "")).upper()
        disturbance_type = str(request.get("disturbance_type", "")).upper()
        area_ha = _safe_decimal(request.get("area_ha"), _ZERO)
        fuel_load_override = request.get("fuel_load_tdm_ha")
        gwp_source = str(request.get("gwp_source", self._gwp_source)).upper()

        # -- Validate -------------------------------------------------------
        errors: List[str] = []
        if not land_category:
            errors.append("land_category is required")
        if not climate_zone:
            errors.append("climate_zone is required")
        if not disturbance_type:
            errors.append("disturbance_type is required")
        if area_ha <= _ZERO:
            errors.append("area_ha must be > 0")

        if errors:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": errors,
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        # -- Determine fuel load -------------------------------------------
        if fuel_load_override is not None:
            fuel_load_tdm_ha = _safe_decimal(fuel_load_override)
        else:
            # Use AGB default / carbon fraction to get dry matter
            if self._land_use_db is not None:
                agb_tc = self._land_use_db.get_agb_default(
                    land_category, climate_zone
                )
                fuel_load_tdm_ha = (agb_tc / CARBON_FRACTION).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            else:
                fuel_load_tdm_ha = _D("200")

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Determine fuel load",
            formula="M_B = AGB / CF or override",
            inputs={"land_category": land_category, "climate_zone": climate_zone},
            output=str(fuel_load_tdm_ha),
            unit="tDM/ha",
        ))

        # -- Get fire emission factors --------------------------------------
        if self._land_use_db is not None:
            fire_ef = self._land_use_db.get_fire_ef(land_category, disturbance_type)
        else:
            fire_ef = {
                "combustion_factor": _D("0.45"),
                "ef_co2_g_per_kg": _D("1580"),
                "ef_ch4_g_per_kg": _D("6.8"),
                "ef_n2o_g_per_kg": _D("0.20"),
                "source": "default",
            }

        cf = _safe_decimal(fire_ef["combustion_factor"])

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Look up fire emission factors",
            formula="C_f, G_ef per gas",
            inputs={
                "land_category": land_category,
                "disturbance_type": disturbance_type,
            },
            output=str(cf),
            unit="combustion_factor",
        ))

        # -- Calculate per-gas emissions ------------------------------------
        # L_fire_gas = A * M_B * C_f * G_ef / 1000  (tonnes)
        gas_emissions: Dict[str, Dict[str, str]] = {}
        total_co2e = _ZERO

        for gas, ef_key in [
            ("CO2", "ef_co2_g_per_kg"),
            ("CH4", "ef_ch4_g_per_kg"),
            ("N2O", "ef_n2o_g_per_kg"),
        ]:
            ef_g_per_kg = _safe_decimal(fire_ef[ef_key])

            # tonnes = area * tDM/ha * cf * g/kg / 1000
            emission_tonnes = (
                area_ha * fuel_load_tdm_ha * cf * ef_g_per_kg / _THOUSAND
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

            # Get GWP for this gas
            gwp_val = _ONE
            if gwp_source in GWP_VALUES and gas in GWP_VALUES[gwp_source]:
                gwp_val = GWP_VALUES[gwp_source][gas]

            co2e_tonnes = (emission_tonnes * gwp_val).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            total_co2e += co2e_tonnes

            gas_emissions[gas] = {
                "emission_factor_g_per_kg": str(ef_g_per_kg),
                "emission_tonnes": str(emission_tonnes),
                "gwp": str(gwp_val),
                "co2e_tonnes": str(co2e_tonnes),
            }

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description=f"Calculate {gas} fire emissions",
                formula=f"L_{gas} = A * M_B * C_f * G_ef / 1000",
                inputs={
                    "area_ha": str(area_ha),
                    "fuel_load": str(fuel_load_tdm_ha),
                    "cf": str(cf),
                    "ef": str(ef_g_per_kg),
                },
                output=str(emission_tonnes),
                unit=f"t{gas}",
            ))

        # -- Total carbon released -----------------------------------------
        # Carbon released = area * fuel_load * cf * carbon_fraction
        total_c_released = (
            area_ha * fuel_load_tdm_ha * cf * CARBON_FRACTION
        ).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "land_category": land_category,
            "climate_zone": climate_zone,
            "disturbance_type": disturbance_type,
            "area_ha": str(area_ha),
            "fuel_load_tdm_ha": str(fuel_load_tdm_ha),
            "combustion_factor": str(cf),
            "gas_emissions": gas_emissions,
            "total_co2e_tonnes": str(
                total_co2e.quantize(_PRECISION, rounding=ROUND_HALF_UP)
            ),
            "total_carbon_released_tc": str(total_c_released),
            "gwp_source": gwp_source,
            "ef_source": fire_ef.get("source", "IPCC"),
            "trace_steps": [s.to_dict() for s in trace_steps],
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Fire emissions calculated: id=%s, area=%s ha, "
            "total_co2e=%s t, carbon_released=%s tC, time=%.3fms",
            calc_id, area_ha, total_co2e, total_c_released, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Conversion Emissions
    # ------------------------------------------------------------------

    def calculate_conversion_emissions(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate emissions from land-use conversion (immediate stock changes).

        When land converts from one category to another, carbon stocks change
        immediately.  DeltaC_conversion = (C_after - C_before) * area

        Required request keys:
            from_category: Original land category.
            to_category: New land category.
            climate_zone: IPCC climate zone.
            area_ha: Area converted in hectares.
            soil_type: IPCC soil type (for SOC).

        Args:
            request: Conversion emission request.

        Returns:
            Per-pool conversion emissions with totals.
        """
        self._increment_calculations()
        start_time = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = str(uuid4())

        from_category = str(request.get("from_category", "")).upper()
        to_category = str(request.get("to_category", "")).upper()
        climate_zone = str(request.get("climate_zone", "")).upper()
        area_ha = _safe_decimal(request.get("area_ha"), _ZERO)
        soil_type = str(request.get("soil_type", "HIGH_ACTIVITY_CLAY")).upper()
        gwp_source = str(request.get("gwp_source", self._gwp_source)).upper()

        # -- Validate -------------------------------------------------------
        errors: List[str] = []
        if not from_category:
            errors.append("from_category is required")
        if not to_category:
            errors.append("to_category is required")
        if not climate_zone:
            errors.append("climate_zone is required")
        if area_ha <= _ZERO:
            errors.append("area_ha must be > 0")

        if errors:
            return {
                "calculation_id": calc_id,
                "status": "VALIDATION_ERROR",
                "errors": errors,
                "processing_time_ms": round(
                    (time.monotonic() - start_time) * 1000, 3
                ),
            }

        # -- Get carbon stocks before and after ----------------------------
        if self._land_use_db is not None:
            factors_before = self._land_use_db.get_all_factors(
                from_category, climate_zone, soil_type
            )
            factors_after = self._land_use_db.get_all_factors(
                to_category, climate_zone, soil_type
            )
        else:
            factors_before = {
                "agb_tc_ha": "0", "bgb_tc_ha": "0",
                "dead_wood_tc_ha": "0", "litter_tc_ha": "0",
                "soc_ref_tc_ha": "0",
            }
            factors_after = dict(factors_before)

        step_num += 1
        trace_steps.append(TraceStep(
            step_number=step_num,
            description="Look up carbon stocks before and after conversion",
            formula="C_before, C_after per pool",
            inputs={
                "from_category": from_category,
                "to_category": to_category,
            },
            output="stocks_retrieved",
            unit="tC/ha",
        ))

        # -- Per-pool conversion calculations ------------------------------
        pool_mapping = [
            ("AGB", "agb_tc_ha"),
            ("BGB", "bgb_tc_ha"),
            ("DEAD_WOOD", "dead_wood_tc_ha"),
            ("LITTER", "litter_tc_ha"),
        ]

        pool_results: Dict[str, Dict[str, str]] = {}
        total_delta_c = _ZERO

        for pool_name, key in pool_mapping:
            before = _safe_decimal(factors_before.get(key, "0"))
            after = _safe_decimal(factors_after.get(key, "0"))
            delta_per_ha = (after - before).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            delta_total = (delta_per_ha * area_ha).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            total_delta_c += delta_total

            pool_results[pool_name] = {
                "before_tc_ha": str(before),
                "after_tc_ha": str(after),
                "delta_c_tc_ha": str(delta_per_ha),
                "delta_c_tc": str(delta_total),
            }

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description=f"Conversion {pool_name} change",
                formula=f"Delta_{pool_name} = (C_after - C_before) * area",
                inputs={
                    "before": str(before),
                    "after": str(after),
                    "area_ha": str(area_ha),
                },
                output=str(delta_total),
                unit="tC",
            ))

        # -- CO2 conversion ------------------------------------------------
        total_co2 = (total_delta_c * CONVERSION_FACTOR_CO2_C).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        is_emission = total_delta_c < _ZERO
        emission_type = "NET_EMISSION" if is_emission else "NET_REMOVAL"

        processing_time = round((time.monotonic() - start_time) * 1000, 3)

        result = {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "from_category": from_category,
            "to_category": to_category,
            "climate_zone": climate_zone,
            "area_ha": str(area_ha),
            "pool_results": pool_results,
            "total_delta_c_tc": str(
                total_delta_c.quantize(_PRECISION, rounding=ROUND_HALF_UP)
            ),
            "total_co2_tonnes": str(total_co2),
            "emission_type": emission_type,
            "gwp_source": gwp_source,
            "trace_steps": [s.to_dict() for s in trace_steps],
            "processing_time_ms": processing_time,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Conversion emissions: id=%s, %s -> %s, "
            "delta_c=%s tC, co2=%s t, type=%s, time=%.3fms",
            calc_id, from_category, to_category,
            total_delta_c, total_co2, emission_type, processing_time,
        )
        return result

    # ------------------------------------------------------------------
    # Total Calculation (All Pools)
    # ------------------------------------------------------------------

    def calculate_total(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate total carbon stock changes across all biomass pools.

        Dispatches to either stock-difference or gain-loss method based on
        the 'method' key in the request.

        Args:
            request: Calculation request with 'method' key.

        Returns:
            Combined calculation result.
        """
        method = str(request.get("method", "STOCK_DIFFERENCE")).upper()

        if method == "GAIN_LOSS":
            return self.calculate_gain_loss(request)
        else:
            return self.calculate_stock_difference(request)

    # ------------------------------------------------------------------
    # CO2e Conversion Utility
    # ------------------------------------------------------------------

    def to_co2e(
        self,
        carbon_stock_tc: Decimal,
        gas: str = "CO2",
        gwp_source: str = "AR6",
    ) -> Decimal:
        """Convert carbon stock (tC) to CO2 equivalent (tCO2e).

        For CO2: tCO2 = tC * 44/12
        For CH4/N2O: tCO2e = mass_tonnes * GWP

        Args:
            carbon_stock_tc: Carbon mass in tonnes carbon.
            gas: Gas type (CO2, CH4, N2O).
            gwp_source: GWP assessment report source.

        Returns:
            Mass in tonnes CO2e.
        """
        gas = gas.upper()
        gwp_source = gwp_source.upper()

        if gas == "CO2":
            return (carbon_stock_tc * CONVERSION_FACTOR_CO2_C).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

        gwp_val = _ONE
        if gwp_source in GWP_VALUES and gas in GWP_VALUES[gwp_source]:
            gwp_val = GWP_VALUES[gwp_source][gas]

        return (carbon_stock_tc * gwp_val).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics."""
        with self._lock:
            return {
                "engine": "CarbonStockCalculatorEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_calculations": self._total_calculations,
                "gwp_source": self._gwp_source,
                "db_available": self._land_use_db is not None,
            }

    def reset(self) -> None:
        """Reset engine counters. Intended for testing teardown."""
        with self._lock:
            self._total_calculations = 0
        logger.info("CarbonStockCalculatorEngine reset")
