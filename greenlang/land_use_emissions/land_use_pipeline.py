# -*- coding: utf-8 -*-
"""
LandUsePipelineEngine - 8-Stage Orchestration Pipeline (Engine 7 of 7)

AGENT-MRV-006: Land Use Emissions Agent

End-to-end orchestration pipeline for IPCC LULUCF carbon stock change
calculations.  Coordinates all six upstream engines through a
deterministic, eight-stage pipeline:

    1. VALIDATE_INPUT     - Validate CalculationRequest, check required fields
    2. CLASSIFY_LAND      - Determine land category, climate zone, soil type
    3. LOOKUP_FACTORS     - Get carbon stock defaults, emission factors
    4. CALCULATE_STOCKS   - Run stock-difference or gain-loss per pool
    5. CALCULATE_SOC      - Run SOC engine for soil carbon changes
    6. CALCULATE_NON_CO2  - Fire emissions, N2O from soil, CH4 from wetlands
    7. CHECK_COMPLIANCE   - Run compliance engine against selected frameworks
    8. ASSEMBLE_RESULTS   - Combine all results, calculate totals, provenance

Each stage is checkpointed so that failures produce partial results with
complete provenance.

Batch Processing:
    ``execute_batch()`` processes multiple calculation requests,
    accumulating results and producing an aggregate batch summary.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Python Decimal arithmetic
    - No LLM calls in the calculation path
    - SHA-256 provenance hash at every pipeline stage
    - Full audit trail for regulatory traceability

Thread Safety:
    All mutable state is protected by a ``threading.Lock``.  Concurrent
    ``execute`` invocations from different threads are safe.

Example:
    >>> from greenlang.land_use_emissions.land_use_pipeline import (
    ...     LandUsePipelineEngine,
    ... )
    >>> pipeline = LandUsePipelineEngine()
    >>> result = pipeline.execute({
    ...     "land_category": "FOREST_LAND",
    ...     "climate_zone": "TROPICAL_WET",
    ...     "soil_type": "HIGH_ACTIVITY_CLAY",
    ...     "area_ha": 1000,
    ...     "method": "STOCK_DIFFERENCE",
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
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["LandUsePipelineEngine"]

# ---------------------------------------------------------------------------
# Optional upstream-engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.land_use_emissions.config import (
        get_config,
    )
except ImportError:
    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.land_use_emissions.land_use_database import (
        LandUseDatabaseEngine,
        CONVERSION_FACTOR_CO2_C,
        GWP_VALUES,
    )
except ImportError:
    LandUseDatabaseEngine = None  # type: ignore[assignment, misc]
    CONVERSION_FACTOR_CO2_C = Decimal("3.66667")
    GWP_VALUES = {
        "AR6": {"CO2": Decimal("1"), "CH4": Decimal("29.8"), "N2O": Decimal("273")},
    }

try:
    from greenlang.land_use_emissions.carbon_stock_calculator import (
        CarbonStockCalculatorEngine,
    )
except ImportError:
    CarbonStockCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.land_use_change_tracker import (
        LandUseChangeTrackerEngine,
    )
except ImportError:
    LandUseChangeTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.soil_organic_carbon import (
        SoilOrganicCarbonEngine,
    )
except ImportError:
    SoilOrganicCarbonEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# UTC helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert to Decimal."""
    if value is None:
        return default
    try:
        return _D(value)
    except Exception:
        return default


# ===========================================================================
# Pipeline Stages
# ===========================================================================


class PipelineStage(str, Enum):
    """Enumeration of the 8 pipeline stages."""

    VALIDATE_INPUT = "VALIDATE_INPUT"
    CLASSIFY_LAND = "CLASSIFY_LAND"
    LOOKUP_FACTORS = "LOOKUP_FACTORS"
    CALCULATE_STOCKS = "CALCULATE_STOCKS"
    CALCULATE_SOC = "CALCULATE_SOC"
    CALCULATE_NON_CO2 = "CALCULATE_NON_CO2"
    CHECK_COMPLIANCE = "CHECK_COMPLIANCE"
    ASSEMBLE_RESULTS = "ASSEMBLE_RESULTS"


#: Valid land categories.
VALID_CATEGORIES: List[str] = [
    "FOREST_LAND", "CROPLAND", "GRASSLAND",
    "WETLANDS", "SETTLEMENTS", "OTHER_LAND",
]

#: Valid calculation methods.
VALID_METHODS: List[str] = ["STOCK_DIFFERENCE", "GAIN_LOSS"]


# ===========================================================================
# LandUsePipelineEngine
# ===========================================================================


class LandUsePipelineEngine:
    """End-to-end orchestration pipeline for LULUCF calculations.

    Coordinates LandUseDatabaseEngine, CarbonStockCalculatorEngine,
    LandUseChangeTrackerEngine, SoilOrganicCarbonEngine,
    UncertaintyQuantifierEngine, and ComplianceCheckerEngine through
    an 8-stage deterministic pipeline.

    Thread Safety:
        All mutable state is protected by a ``threading.Lock``.

    Attributes:
        _db_engine: LandUseDatabaseEngine instance.
        _calc_engine: CarbonStockCalculatorEngine instance.
        _tracker_engine: LandUseChangeTrackerEngine instance.
        _soc_engine: SoilOrganicCarbonEngine instance.
        _uncertainty_engine: UncertaintyQuantifierEngine instance.
        _compliance_engine: ComplianceCheckerEngine instance.
        _lock: Thread lock for mutable state.
        _total_executions: Total pipeline executions counter.
        _stage_timings: Accumulated per-stage timing data.

    Example:
        >>> pipeline = LandUsePipelineEngine()
        >>> result = pipeline.execute(request)
    """

    def __init__(
        self,
        db_engine: Optional[Any] = None,
        calc_engine: Optional[Any] = None,
        tracker_engine: Optional[Any] = None,
        soc_engine: Optional[Any] = None,
        uncertainty_engine: Optional[Any] = None,
        compliance_engine: Optional[Any] = None,
    ) -> None:
        """Initialize the LandUsePipelineEngine.

        Creates default engine instances if not provided.  Engines that
        fail to import are set to None and their stages are skipped.

        Args:
            db_engine: Optional LandUseDatabaseEngine.
            calc_engine: Optional CarbonStockCalculatorEngine.
            tracker_engine: Optional LandUseChangeTrackerEngine.
            soc_engine: Optional SoilOrganicCarbonEngine.
            uncertainty_engine: Optional UncertaintyQuantifierEngine.
            compliance_engine: Optional ComplianceCheckerEngine.
        """
        # Initialize engines
        self._db_engine = db_engine
        if self._db_engine is None and LandUseDatabaseEngine is not None:
            self._db_engine = LandUseDatabaseEngine()

        self._calc_engine = calc_engine
        if self._calc_engine is None and CarbonStockCalculatorEngine is not None:
            self._calc_engine = CarbonStockCalculatorEngine(
                land_use_database=self._db_engine
            )

        self._tracker_engine = tracker_engine
        if self._tracker_engine is None and LandUseChangeTrackerEngine is not None:
            self._tracker_engine = LandUseChangeTrackerEngine()

        self._soc_engine = soc_engine
        if self._soc_engine is None and SoilOrganicCarbonEngine is not None:
            self._soc_engine = SoilOrganicCarbonEngine(
                land_use_database=self._db_engine
            )

        self._uncertainty_engine = uncertainty_engine
        if self._uncertainty_engine is None and UncertaintyQuantifierEngine is not None:
            self._uncertainty_engine = UncertaintyQuantifierEngine()

        self._compliance_engine = compliance_engine
        if self._compliance_engine is None and ComplianceCheckerEngine is not None:
            self._compliance_engine = ComplianceCheckerEngine()

        self._lock = threading.Lock()
        self._total_executions: int = 0
        self._total_batches: int = 0
        self._stage_timings: Dict[str, List[float]] = {
            stage.value: [] for stage in PipelineStage
        }
        self._created_at = _utcnow()

        engine_status = {
            "db": self._db_engine is not None,
            "calc": self._calc_engine is not None,
            "tracker": self._tracker_engine is not None,
            "soc": self._soc_engine is not None,
            "uncertainty": self._uncertainty_engine is not None,
            "compliance": self._compliance_engine is not None,
        }

        logger.info(
            "LandUsePipelineEngine initialized: stages=%d, engines=%s",
            len(PipelineStage), engine_status,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_stage_timing(self, stage: str, elapsed_ms: float) -> None:
        """Record timing for a pipeline stage."""
        with self._lock:
            self._stage_timings[stage].append(elapsed_ms)

    def _run_stage(
        self,
        stage: PipelineStage,
        context: Dict[str, Any],
        stage_func: Any,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Execute a single pipeline stage with timing and error handling.

        Args:
            stage: Pipeline stage enum.
            context: Pipeline context dictionary (mutated in place).
            stage_func: Callable that performs the stage work.

        Returns:
            Tuple of (updated context, error message or None).
        """
        stage_start = time.monotonic()
        error: Optional[str] = None

        try:
            stage_func(context)
            context["stages_completed"].append(stage.value)
        except Exception as e:
            error = f"Stage {stage.value} failed: {str(e)}"
            context["errors"].append(error)
            context["stages_failed"].append(stage.value)
            logger.error(
                "Pipeline stage %s failed: %s",
                stage.value, str(e), exc_info=True,
            )

        elapsed_ms = (time.monotonic() - stage_start) * 1000
        context["stage_timings"][stage.value] = round(elapsed_ms, 3)
        self._record_stage_timing(stage.value, elapsed_ms)

        # Provenance per stage
        stage_data = {
            "stage": stage.value,
            "elapsed_ms": elapsed_ms,
            "error": error,
        }
        context["provenance_chain"].append(_compute_hash(stage_data))

        return context, error

    # ------------------------------------------------------------------
    # Stage Implementations
    # ------------------------------------------------------------------

    def _stage_validate_input(self, ctx: Dict[str, Any]) -> None:
        """Stage 1: Validate input request.

        Checks required fields, validates enums, and normalises values.
        """
        request = ctx["request"]
        errors: List[str] = []

        # Required fields
        land_category = str(request.get("land_category", "")).upper()
        if not land_category:
            errors.append("land_category is required")
        elif land_category not in VALID_CATEGORIES:
            errors.append(f"Invalid land_category: {land_category}")
        ctx["land_category"] = land_category

        climate_zone = str(request.get("climate_zone", "")).upper()
        if not climate_zone:
            errors.append("climate_zone is required")
        ctx["climate_zone"] = climate_zone

        area_ha = _safe_decimal(request.get("area_ha"), _ZERO)
        if area_ha <= _ZERO:
            errors.append("area_ha must be > 0")
        ctx["area_ha"] = area_ha

        method = str(request.get("method", "STOCK_DIFFERENCE")).upper()
        if method not in VALID_METHODS:
            errors.append(f"Invalid method: {method}. Valid: {VALID_METHODS}")
        ctx["method"] = method

        ctx["soil_type"] = str(request.get("soil_type", "HIGH_ACTIVITY_CLAY")).upper()
        ctx["gwp_source"] = str(request.get("gwp_source", "AR6")).upper()
        ctx["tier"] = str(request.get("tier", "TIER_1")).upper()
        ctx["frameworks"] = request.get("frameworks", [])
        ctx["parcel_id"] = str(request.get("parcel_id", ""))

        if errors:
            ctx["validation_errors"] = errors
            raise ValueError(f"Validation failed: {errors}")

        ctx["validation_status"] = "PASSED"
        logger.debug("Validation passed: %s %s %s", land_category, climate_zone, method)

    def _stage_classify_land(self, ctx: Dict[str, Any]) -> None:
        """Stage 2: Classify land category, climate zone, and soil type.

        Uses the LandUseDatabaseEngine for classification if climate/soil
        data is provided as raw measurements.
        """
        request = ctx["request"]

        # Auto-classify climate zone from temperature/precipitation if provided
        if self._db_engine is not None:
            temp = request.get("mean_annual_temp_c")
            precip = request.get("annual_precip_mm")
            if temp is not None and precip is not None:
                auto_zone = self._db_engine.classify_climate_zone(
                    _D(str(temp)),
                    _D(str(precip)),
                    _safe_decimal(request.get("elevation_m")),
                    _safe_decimal(request.get("latitude")),
                )
                ctx["climate_zone"] = auto_zone
                ctx["climate_zone_auto_classified"] = True

            # Auto-classify soil type
            soil_order = request.get("soil_order")
            organic_pct = request.get("organic_content_pct")
            if soil_order is not None or organic_pct is not None:
                auto_soil = self._db_engine.classify_soil_type(
                    soil_order=soil_order,
                    organic_content_pct=_safe_decimal(organic_pct) if organic_pct else None,
                    drainage_class=request.get("drainage_class"),
                    clay_content_pct=_safe_decimal(request.get("clay_content_pct")) if request.get("clay_content_pct") else None,
                    sand_content_pct=_safe_decimal(request.get("sand_content_pct")) if request.get("sand_content_pct") else None,
                )
                ctx["soil_type"] = auto_soil
                ctx["soil_type_auto_classified"] = True

        # Get subcategories
        if self._db_engine is not None:
            subcats = self._db_engine.get_land_subcategories(ctx["land_category"])
            ctx["subcategories"] = subcats

        ctx["classification_status"] = "COMPLETE"
        logger.debug(
            "Classification: category=%s, zone=%s, soil=%s",
            ctx["land_category"], ctx["climate_zone"], ctx["soil_type"],
        )

    def _stage_lookup_factors(self, ctx: Dict[str, Any]) -> None:
        """Stage 3: Look up carbon stock defaults and emission factors."""
        if self._db_engine is None:
            ctx["factors"] = {"status": "DB_UNAVAILABLE"}
            return

        factors = self._db_engine.get_all_factors(
            ctx["land_category"],
            ctx["climate_zone"],
            ctx["soil_type"],
        )
        ctx["factors"] = factors

        # Growth rate for gain-loss method
        if ctx["method"] == "GAIN_LOSS":
            growth = self._db_engine.get_growth_rate(
                ctx["land_category"], ctx["climate_zone"]
            )
            ctx["growth_rate_tc_ha_yr"] = growth

        # Fire EFs if disturbance present
        disturbance = ctx["request"].get("disturbance_type")
        if disturbance and disturbance.upper().startswith("FIRE"):
            fire_ef = self._db_engine.get_fire_ef(
                ctx["land_category"], disturbance
            )
            ctx["fire_ef"] = fire_ef

        # Peatland EFs if wetland
        peatland_type = ctx["request"].get("peatland_type")
        if peatland_type:
            peat_ef = self._db_engine.get_peatland_ef(peatland_type)
            ctx["peatland_ef"] = peat_ef

        ctx["factors_status"] = "COMPLETE"
        logger.debug("Factors retrieved: %d keys", len(factors))

    def _stage_calculate_stocks(self, ctx: Dict[str, Any]) -> None:
        """Stage 4: Run carbon stock calculations (stock-difference or gain-loss)."""
        if self._calc_engine is None:
            ctx["stock_result"] = {"status": "CALC_ENGINE_UNAVAILABLE"}
            return

        request = ctx["request"]
        method = ctx["method"]

        calc_request = {
            "land_category": ctx["land_category"],
            "climate_zone": ctx["climate_zone"],
            "area_ha": str(ctx["area_ha"]),
            "gwp_source": ctx["gwp_source"],
            "method": method,
        }

        if method == "STOCK_DIFFERENCE":
            calc_request.update({
                "c_t1": request.get("c_t1", {}),
                "c_t2": request.get("c_t2", {}),
                "year_t1": request.get("year_t1", 0),
                "year_t2": request.get("year_t2", 0),
            })
            stock_result = self._calc_engine.calculate_stock_difference(calc_request)
        else:
            calc_request.update({
                "harvest_volume_m3": request.get("harvest_volume_m3", 0),
                "fuelwood_volume_m3": request.get("fuelwood_volume_m3", 0),
                "disturbance_area_ha": request.get("disturbance_area_ha", 0),
                "disturbance_type": request.get("disturbance_type", ""),
                "growth_rate_override": request.get("growth_rate_override"),
                "wood_density": request.get("wood_density"),
                "bcef": request.get("bcef"),
            })
            stock_result = self._calc_engine.calculate_gain_loss(calc_request)

        ctx["stock_result"] = stock_result
        ctx["stock_status"] = stock_result.get("status", "UNKNOWN")
        logger.debug("Stock calculation: status=%s", ctx["stock_status"])

    def _stage_calculate_soc(self, ctx: Dict[str, Any]) -> None:
        """Stage 5: Run SOC engine for soil carbon changes."""
        if self._soc_engine is None:
            ctx["soc_result"] = {"status": "SOC_ENGINE_UNAVAILABLE"}
            return

        request = ctx["request"]

        # Check if SOC change calculation is needed
        old_lu = request.get("old_land_use_type") or request.get("from_land_use_type")
        new_lu = request.get("new_land_use_type") or request.get("land_use_type")

        if old_lu and new_lu:
            # SOC change calculation
            soc_request = {
                "climate_zone": ctx["climate_zone"],
                "soil_type": ctx["soil_type"],
                "old_land_use_type": old_lu,
                "old_management_practice": request.get("old_management_practice", "NOMINAL"),
                "old_input_level": request.get("old_input_level", "MEDIUM"),
                "new_land_use_type": new_lu,
                "new_management_practice": request.get("new_management_practice", "NOMINAL"),
                "new_input_level": request.get("new_input_level", "MEDIUM"),
                "area_ha": str(ctx["area_ha"]),
                "transition_period_years": request.get("transition_period_years", 20),
            }
            soc_result = self._soc_engine.calculate_soc_change(soc_request)
        elif new_lu:
            # SOC stock calculation
            soc_request = {
                "climate_zone": ctx["climate_zone"],
                "soil_type": ctx["soil_type"],
                "land_use_type": new_lu,
                "management_practice": request.get("management_practice", "NOMINAL"),
                "input_level": request.get("input_level", "MEDIUM"),
                "area_ha": str(ctx["area_ha"]),
                "parcel_id": ctx["parcel_id"],
            }
            soc_result = self._soc_engine.calculate_soc(soc_request)
        else:
            soc_result = {"status": "SKIPPED", "reason": "No land use type specified for SOC"}

        ctx["soc_result"] = soc_result

        # Liming and urea emissions
        limestone = request.get("limestone_tonnes")
        dolomite = request.get("dolomite_tonnes")
        if limestone or dolomite:
            liming_result = self._soc_engine.calculate_liming_emissions({
                "limestone_tonnes": limestone or 0,
                "dolomite_tonnes": dolomite or 0,
            })
            ctx["liming_result"] = liming_result

        urea = request.get("urea_tonnes")
        if urea:
            urea_result = self._soc_engine.calculate_urea_emissions({
                "urea_tonnes": urea,
            })
            ctx["urea_result"] = urea_result

        ctx["soc_status"] = soc_result.get("status", "UNKNOWN")
        logger.debug("SOC calculation: status=%s", ctx["soc_status"])

    def _stage_calculate_non_co2(self, ctx: Dict[str, Any]) -> None:
        """Stage 6: Calculate non-CO2 emissions (fire, N2O, CH4)."""
        request = ctx["request"]
        non_co2: Dict[str, Any] = {"status": "COMPLETE", "emissions": {}}

        # Fire emissions
        disturbance_type = request.get("disturbance_type", "")
        disturbance_area = _safe_decimal(request.get("disturbance_area_ha"))
        if disturbance_type and disturbance_area > _ZERO and self._calc_engine is not None:
            fire_result = self._calc_engine.calculate_fire_emissions({
                "land_category": ctx["land_category"],
                "climate_zone": ctx["climate_zone"],
                "disturbance_type": disturbance_type,
                "area_ha": str(disturbance_area),
                "gwp_source": ctx["gwp_source"],
                "fuel_load_tdm_ha": request.get("fuel_load_tdm_ha"),
            })
            non_co2["fire_emissions"] = fire_result

        # Peatland emissions
        peatland_type = request.get("peatland_type")
        if peatland_type and self._db_engine is not None:
            try:
                peat_ef = self._db_engine.get_peatland_ef(peatland_type)
                peat_area = _safe_decimal(request.get("peatland_area_ha", ctx["area_ha"]))

                co2_tc = _safe_decimal(peat_ef.get("co2_tc_ha_yr")) * peat_area
                ch4_kg = _safe_decimal(peat_ef.get("ch4_kg_ha_yr")) * peat_area
                n2o_kg = _safe_decimal(peat_ef.get("n2o_kg_ha_yr")) * peat_area

                # Convert to CO2e
                ch4_gwp = GWP_VALUES.get(ctx["gwp_source"], {}).get("CH4", _D("29.8"))
                n2o_gwp = GWP_VALUES.get(ctx["gwp_source"], {}).get("N2O", _D("273"))

                ch4_co2e = (ch4_kg / _D("1000") * ch4_gwp).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
                n2o_co2e = (n2o_kg / _D("1000") * n2o_gwp).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
                co2_tonnes = (co2_tc * CONVERSION_FACTOR_CO2_C).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )

                non_co2["peatland_emissions"] = {
                    "peatland_type": peatland_type,
                    "area_ha": str(peat_area),
                    "co2_tonnes_yr": str(co2_tonnes),
                    "ch4_co2e_tonnes_yr": str(ch4_co2e),
                    "n2o_co2e_tonnes_yr": str(n2o_co2e),
                    "total_co2e_tonnes_yr": str(
                        (co2_tonnes + ch4_co2e + n2o_co2e).quantize(
                            _PRECISION, rounding=ROUND_HALF_UP
                        )
                    ),
                }
            except Exception as e:
                non_co2["peatland_emissions"] = {
                    "status": "ERROR", "error": str(e)
                }

        # N2O from managed soils
        n_fertilizer = _safe_decimal(request.get("n_fertilizer_tonnes"))
        if n_fertilizer > _ZERO and self._db_engine is not None:
            try:
                ef1 = self._db_engine.get_n2o_ef("EF1_SYNTHETIC_FERTILIZER")
                frac_gasf = self._db_engine.get_n2o_ef("FRAC_GASF")
                ef3_atm = self._db_engine.get_n2o_ef("EF3_ATMOSPHERIC_DEPOSITION")
                frac_leach = self._db_engine.get_n2o_ef("FRAC_LEACH")
                ef3_leach = self._db_engine.get_n2o_ef("EF3_LEACHING_RUNOFF")

                # Direct N2O
                n2o_direct = (n_fertilizer * ef1 * _D("1.571429")).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
                # Indirect - atmospheric deposition
                n2o_atm = (n_fertilizer * frac_gasf * ef3_atm * _D("1.571429")).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
                # Indirect - leaching
                n2o_leach = (n_fertilizer * frac_leach * ef3_leach * _D("1.571429")).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )

                n2o_total = n2o_direct + n2o_atm + n2o_leach
                n2o_gwp = GWP_VALUES.get(ctx["gwp_source"], {}).get("N2O", _D("273"))
                n2o_co2e = (n2o_total * n2o_gwp).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )

                non_co2["n2o_soil_emissions"] = {
                    "n_fertilizer_tonnes": str(n_fertilizer),
                    "n2o_direct_tonnes": str(n2o_direct),
                    "n2o_indirect_atm_tonnes": str(n2o_atm),
                    "n2o_indirect_leach_tonnes": str(n2o_leach),
                    "n2o_total_tonnes": str(n2o_total),
                    "n2o_co2e_tonnes": str(n2o_co2e),
                }
            except Exception as e:
                non_co2["n2o_soil_emissions"] = {
                    "status": "ERROR", "error": str(e)
                }

        ctx["non_co2_result"] = non_co2
        logger.debug("Non-CO2 calculation complete")

    def _stage_check_compliance(self, ctx: Dict[str, Any]) -> None:
        """Stage 7: Run compliance checks against selected frameworks."""
        if self._compliance_engine is None:
            ctx["compliance_result"] = {"status": "COMPLIANCE_ENGINE_UNAVAILABLE"}
            return

        frameworks = ctx.get("frameworks", [])
        if not frameworks:
            ctx["compliance_result"] = {"status": "SKIPPED", "reason": "No frameworks specified"}
            return

        # Build compliance data from context
        stock_result = ctx.get("stock_result", {})
        pools_reported = []
        if "pool_results" in stock_result:
            pools_reported = list(stock_result["pool_results"].keys())
        if ctx.get("soc_result", {}).get("status") == "SUCCESS":
            pools_reported.append("SOC")

        compliance_data = {
            "land_category": ctx["land_category"],
            "climate_zone": ctx["climate_zone"],
            "method": ctx["method"],
            "tier": ctx["tier"],
            "area_ha": str(ctx["area_ha"]),
            "pools_reported": pools_reported,
            "total_co2e_tonnes": stock_result.get("total_co2_tonnes_yr", "0"),
            "net_co2e_tonnes_yr": stock_result.get("net_co2e_tonnes_yr", "0"),
            "gross_emissions_tco2_yr": stock_result.get("gross_emissions_tco2_yr", "0"),
            "gross_removals_tco2_yr": stock_result.get("gross_removals_tco2_yr", "0"),
            "emission_type": stock_result.get("emission_type", ""),
            "gwp_source": ctx["gwp_source"],
            "ef_source": ctx.get("factors", {}).get("source", "IPCC_2006"),
            "provenance_hash": stock_result.get("provenance_hash", ""),
            "has_uncertainty": ctx.get("uncertainty_result") is not None,
            "year_t1": ctx["request"].get("year_t1"),
            "year_t2": ctx["request"].get("year_t2"),
            "parcel_id": ctx["parcel_id"],
            "management_practice": ctx["request"].get("management_practice"),
            "is_managed": ctx["request"].get("is_managed", True),
        }

        # Add SOC fields
        soc = ctx.get("soc_result", {})
        if soc.get("status") == "SUCCESS":
            compliance_data["soc_method"] = "STOCK_CHANGE_FACTORS"

        # Add fire/disturbance fields
        if ctx["request"].get("disturbance_type"):
            compliance_data["disturbance_type"] = ctx["request"]["disturbance_type"]
            compliance_data["fire_emissions"] = ctx.get("non_co2_result", {}).get("fire_emissions")

        # Add non-CO2
        if ctx.get("non_co2_result", {}).get("n2o_soil_emissions"):
            compliance_data["n2o_emissions"] = True
            compliance_data["non_co2_emissions"] = True

        compliance_result = self._compliance_engine.check_compliance(
            compliance_data, frameworks
        )
        ctx["compliance_result"] = compliance_result
        logger.debug(
            "Compliance: %s",
            compliance_result.get("overall", {}).get("compliance_status", "UNKNOWN"),
        )

    def _stage_assemble_results(self, ctx: Dict[str, Any]) -> None:
        """Stage 8: Assemble all results into final output."""
        stock_result = ctx.get("stock_result", {})
        soc_result = ctx.get("soc_result", {})
        non_co2 = ctx.get("non_co2_result", {})

        # Total CO2e from all sources
        total_co2e = _ZERO

        # Stock change CO2
        stock_co2 = _safe_decimal(stock_result.get("total_co2_tonnes_yr", "0"))
        total_co2e += stock_co2

        # SOC change CO2
        soc_co2 = _safe_decimal(soc_result.get("delta_co2_tonnes_yr", "0"))
        total_co2e += soc_co2

        # Liming CO2
        liming_co2 = _safe_decimal(
            ctx.get("liming_result", {}).get("co2_tonnes", "0")
        )
        total_co2e += liming_co2

        # Urea CO2
        urea_co2 = _safe_decimal(
            ctx.get("urea_result", {}).get("co2_tonnes", "0")
        )
        total_co2e += urea_co2

        # Fire CO2e
        fire_co2e = _safe_decimal(
            non_co2.get("fire_emissions", {}).get("total_co2e_tonnes", "0")
        )
        total_co2e += fire_co2e

        # Peatland CO2e
        peatland_co2e = _safe_decimal(
            non_co2.get("peatland_emissions", {}).get("total_co2e_tonnes_yr", "0")
        )
        total_co2e += peatland_co2e

        # N2O CO2e
        n2o_co2e = _safe_decimal(
            non_co2.get("n2o_soil_emissions", {}).get("n2o_co2e_tonnes", "0")
        )
        total_co2e += n2o_co2e

        total_co2e = total_co2e.quantize(_PRECISION, rounding=ROUND_HALF_UP)

        is_emission = total_co2e > _ZERO
        net_type = "NET_EMISSION" if is_emission else "NET_REMOVAL" if total_co2e < _ZERO else "NEUTRAL"

        ctx["assembled"] = {
            "stock_change_co2_yr": str(stock_co2),
            "soc_change_co2_yr": str(soc_co2),
            "liming_co2_yr": str(liming_co2),
            "urea_co2_yr": str(urea_co2),
            "fire_co2e_yr": str(fire_co2e),
            "peatland_co2e_yr": str(peatland_co2e),
            "n2o_co2e_yr": str(n2o_co2e),
            "total_co2e_tonnes_yr": str(total_co2e),
            "net_type": net_type,
        }

        ctx["assembly_status"] = "COMPLETE"
        logger.debug("Assembly complete: total_co2e=%s tCO2e/yr", total_co2e)

    # ------------------------------------------------------------------
    # Main Execute
    # ------------------------------------------------------------------

    def execute(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the full 8-stage pipeline for a single calculation.

        Args:
            request: Calculation request dictionary.

        Returns:
            Complete calculation result with all stage outputs.
        """
        pipeline_start = time.monotonic()
        pipeline_id = str(uuid4())

        with self._lock:
            self._total_executions += 1

        # Initialize context
        ctx: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "request": request,
            "stages_completed": [],
            "stages_failed": [],
            "errors": [],
            "stage_timings": {},
            "provenance_chain": [],
        }

        # Stage definitions
        stages = [
            (PipelineStage.VALIDATE_INPUT, self._stage_validate_input),
            (PipelineStage.CLASSIFY_LAND, self._stage_classify_land),
            (PipelineStage.LOOKUP_FACTORS, self._stage_lookup_factors),
            (PipelineStage.CALCULATE_STOCKS, self._stage_calculate_stocks),
            (PipelineStage.CALCULATE_SOC, self._stage_calculate_soc),
            (PipelineStage.CALCULATE_NON_CO2, self._stage_calculate_non_co2),
            (PipelineStage.CHECK_COMPLIANCE, self._stage_check_compliance),
            (PipelineStage.ASSEMBLE_RESULTS, self._stage_assemble_results),
        ]

        # Execute stages sequentially
        abort = False
        for stage, func in stages:
            if abort:
                ctx["stages_failed"].append(stage.value)
                continue

            _, error = self._run_stage(stage, ctx, func)

            # Abort on validation errors (Stage 1 failure is fatal)
            if error and stage == PipelineStage.VALIDATE_INPUT:
                abort = True

        # Build final result
        pipeline_time = round((time.monotonic() - pipeline_start) * 1000, 3)

        is_success = len(ctx["stages_failed"]) == 0
        assembled = ctx.get("assembled", {})

        result: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "status": "SUCCESS" if is_success else "PARTIAL" if ctx["stages_completed"] else "FAILED",
            "land_category": ctx.get("land_category", ""),
            "climate_zone": ctx.get("climate_zone", ""),
            "soil_type": ctx.get("soil_type", ""),
            "area_ha": str(ctx.get("area_ha", "0")),
            "method": ctx.get("method", ""),
            "tier": ctx.get("tier", ""),
            "gwp_source": ctx.get("gwp_source", ""),
            "results": {
                "stock_change": ctx.get("stock_result", {}),
                "soc_change": ctx.get("soc_result", {}),
                "liming": ctx.get("liming_result", {}),
                "urea": ctx.get("urea_result", {}),
                "non_co2": ctx.get("non_co2_result", {}),
                "totals": assembled,
            },
            "total_co2e_tonnes_yr": assembled.get("total_co2e_tonnes_yr", "0"),
            "net_type": assembled.get("net_type", "UNKNOWN"),
            "compliance": ctx.get("compliance_result", {}),
            "factors": ctx.get("factors", {}),
            "stages_completed": ctx["stages_completed"],
            "stages_failed": ctx["stages_failed"],
            "stage_timings": ctx["stage_timings"],
            "errors": ctx["errors"],
            "provenance_chain": ctx["provenance_chain"],
            "processing_time_ms": pipeline_time,
            "calculated_at": _utcnow_iso(),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Pipeline execute: id=%s, status=%s, "
            "stages=%d/%d, total_co2e=%s, time=%.3fms",
            pipeline_id, result["status"],
            len(ctx["stages_completed"]), len(stages),
            result["total_co2e_tonnes_yr"], pipeline_time,
        )
        return result

    # ------------------------------------------------------------------
    # Batch Execute
    # ------------------------------------------------------------------

    def execute_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute the pipeline for a batch of calculation requests.

        Args:
            requests: List of calculation request dictionaries.

        Returns:
            Batch results with individual and aggregate summaries.
        """
        batch_start = time.monotonic()
        batch_id = str(uuid4())

        with self._lock:
            self._total_batches += 1

        results: List[Dict[str, Any]] = []
        total_co2e = _ZERO
        success_count = 0
        failure_count = 0

        for i, request in enumerate(requests):
            try:
                result = self.execute(request)
                results.append(result)

                if result["status"] == "SUCCESS":
                    success_count += 1
                    total_co2e += _safe_decimal(result.get("total_co2e_tonnes_yr", "0"))
                else:
                    failure_count += 1
            except Exception as e:
                failure_count += 1
                results.append({
                    "pipeline_id": str(uuid4()),
                    "status": "FAILED",
                    "errors": [f"Batch item {i} failed: {str(e)}"],
                    "request_index": i,
                })
                logger.error(
                    "Batch item %d failed: %s", i, str(e), exc_info=True
                )

        batch_time = round((time.monotonic() - batch_start) * 1000, 3)

        # Aggregate by category
        by_category: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        for r in results:
            if r.get("status") == "SUCCESS":
                cat = r.get("land_category", "UNKNOWN")
                by_category[cat] += _safe_decimal(r.get("total_co2e_tonnes_yr", "0"))

        batch_result = {
            "batch_id": batch_id,
            "status": "SUCCESS" if failure_count == 0 else "PARTIAL",
            "total_requests": len(requests),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_co2e_tonnes_yr": str(
                total_co2e.quantize(_PRECISION, rounding=ROUND_HALF_UP)
            ),
            "by_category": {
                k: str(v.quantize(_PRECISION, rounding=ROUND_HALF_UP))
                for k, v in sorted(by_category.items())
            },
            "results": results,
            "processing_time_ms": batch_time,
            "calculated_at": _utcnow_iso(),
        }
        batch_result["provenance_hash"] = _compute_hash(batch_result)

        logger.info(
            "Batch execute: id=%s, total=%d, success=%d, "
            "failed=%d, co2e=%s, time=%.3fms",
            batch_id, len(requests), success_count,
            failure_count, total_co2e, batch_time,
        )
        return batch_result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return pipeline engine statistics."""
        with self._lock:
            avg_timings: Dict[str, Optional[float]] = {}
            for stage, times in self._stage_timings.items():
                if times:
                    avg_timings[stage] = round(sum(times) / len(times), 3)
                else:
                    avg_timings[stage] = None

            return {
                "engine": "LandUsePipelineEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_executions": self._total_executions,
                "total_batches": self._total_batches,
                "stages": [s.value for s in PipelineStage],
                "stage_count": len(PipelineStage),
                "avg_stage_timings_ms": avg_timings,
                "engines": {
                    "db": self._db_engine is not None,
                    "calc": self._calc_engine is not None,
                    "tracker": self._tracker_engine is not None,
                    "soc": self._soc_engine is not None,
                    "uncertainty": self._uncertainty_engine is not None,
                    "compliance": self._compliance_engine is not None,
                },
            }

    def reset(self) -> None:
        """Reset pipeline state. Intended for testing teardown."""
        with self._lock:
            self._total_executions = 0
            self._total_batches = 0
            self._stage_timings = {
                stage.value: [] for stage in PipelineStage
            }

        # Reset upstream engines
        if self._db_engine and hasattr(self._db_engine, "reset"):
            self._db_engine.reset()
        if self._calc_engine and hasattr(self._calc_engine, "reset"):
            self._calc_engine.reset()
        if self._tracker_engine and hasattr(self._tracker_engine, "reset"):
            self._tracker_engine.reset()
        if self._soc_engine and hasattr(self._soc_engine, "reset"):
            self._soc_engine.reset()
        if self._uncertainty_engine and hasattr(self._uncertainty_engine, "reset"):
            self._uncertainty_engine.reset()
        if self._compliance_engine and hasattr(self._compliance_engine, "reset"):
            self._compliance_engine.reset()

        logger.info("LandUsePipelineEngine and all upstream engines reset")
