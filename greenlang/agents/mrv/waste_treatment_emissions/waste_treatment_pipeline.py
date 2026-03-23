# -*- coding: utf-8 -*-
"""
WasteTreatmentPipelineEngine - 8-Stage Orchestration Pipeline (Engine 7 of 7)

AGENT-MRV-007: On-site Waste Treatment Emissions Agent

End-to-end orchestration pipeline for IPCC Volume 5 (Waste) emission
calculations covering biological treatment, thermal treatment, and on-site
wastewater treatment.  Coordinates five upstream engines through a
deterministic, eight-stage pipeline:

    1. VALIDATE_INPUT      - Validate request, required fields, composition sums
    2. CLASSIFY_TREATMENT  - Determine treatment method(s), waste categories
    3. LOOKUP_FACTORS      - Get emission factors, DOC, MCF, carbon content, NCV
    4. CALCULATE_BIOLOGICAL - Run biological engine for composting/AD/MBT streams
    5. CALCULATE_THERMAL   - Run thermal engine for incineration/pyrolysis/gasification
    6. CALCULATE_WASTEWATER - Run wastewater engine for on-site treatment
    7. CHECK_COMPLIANCE    - Run compliance engine against selected frameworks
    8. ASSEMBLE_RESULTS    - Combine all results, fossil/biogenic CO2 split, provenance

Each stage is checkpointed so that failures produce partial results with
complete provenance.

Batch Processing:
    ``execute_batch()`` processes multiple calculation requests,
    accumulating results and producing an aggregate batch summary with
    per-treatment-method and per-waste-category breakdowns.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Python Decimal arithmetic
    - No LLM calls in the calculation path
    - SHA-256 provenance hash at every pipeline stage
    - Full audit trail for regulatory traceability (IPCC/GHG Protocol/CSRD)

Thread Safety:
    All mutable state is protected by a ``threading.Lock``.  Concurrent
    ``execute`` invocations from different threads are safe.

Gas Separation:
    The pipeline separates fossil CO2 from biogenic CO2.  IPCC and ETS
    frameworks exclude biogenic CO2 from Scope 1 totals.  The result
    structure provides both gross (fossil + biogenic) and net (fossil only)
    totals for flexible regulatory reporting.

Methane Recovery:
    Full methane lifecycle tracking: generated -> captured -> flared /
    utilized / vented -> emitted.  Energy recovery credits from WtE are
    calculated as displaced grid emissions.

Example:
    >>> from greenlang.agents.mrv.waste_treatment_emissions.waste_treatment_pipeline import (
    ...     WasteTreatmentPipelineEngine,
    ... )
    >>> pipeline = WasteTreatmentPipelineEngine()
    >>> result = pipeline.execute({
    ...     "tenant_id": "tenant_001",
    ...     "facility_id": "facility_abc",
    ...     "treatment_streams": [
    ...         {
    ...             "stream_id": "stream_01",
    ...             "treatment_method": "COMPOSTING",
    ...             "waste_category": "FOOD_WASTE",
    ...             "waste_mass_tonnes": 500,
    ...             "composition": {"organic": 85, "paper": 10, "other": 5},
    ...         },
    ...         {
    ...             "stream_id": "stream_02",
    ...             "treatment_method": "INCINERATION",
    ...             "waste_category": "MIXED_WASTE",
    ...             "waste_mass_tonnes": 1200,
    ...             "composition": {"plastic": 30, "paper": 25, "food": 20,
    ...                             "textile": 10, "wood": 10, "other": 5},
    ...         },
    ...     ],
    ...     "frameworks": ["GHG_PROTOCOL", "IPCC_2006"],
    ... })
    >>> assert result["status"] in ("SUCCESS", "PARTIAL")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-007)
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

__all__ = ["WasteTreatmentPipelineEngine"]

# ---------------------------------------------------------------------------
# Optional upstream-engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.waste_treatment_emissions.config import (
        get_config,
    )
except ImportError:
    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.waste_treatment_emissions.waste_treatment_database import (
        WasteTreatmentDatabaseEngine,
    )
except ImportError:
    WasteTreatmentDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.biological_treatment import (
        BiologicalTreatmentEngine,
    )
except ImportError:
    BiologicalTreatmentEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.thermal_treatment import (
        ThermalTreatmentEngine,
    )
except ImportError:
    ThermalTreatmentEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.wastewater_treatment import (
        WastewaterTreatmentEngine,
    )
except ImportError:
    WastewaterTreatmentEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.provenance import (
        ProvenanceTracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]


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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Any JSON-serializable object or Pydantic model.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
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
_THOUSAND = Decimal("1000")
_CO2_C_RATIO = Decimal("3.66667")  # 44/12 molecular weight ratio CO2/C
_N2O_N_RATIO = Decimal("1.571429")  # 44/28 molecular weight ratio N2O/N
_CH4_C_RATIO = Decimal("1.333333")  # 16/12 molecular weight ratio CH4/C


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal.

    Args:
        value: Numeric value (int, float, str, Decimal).

    Returns:
        Decimal representation of the value.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert to Decimal, returning a default on failure.

    Args:
        value: Value to convert.
        default: Fallback value if conversion fails.

    Returns:
        Decimal representation or the default.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except Exception:
        return default


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to pipeline precision.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal value.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# GWP values (IPCC AR6 100-year defaults)
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR6": {
        "CO2": _ONE,
        "CH4": Decimal("29.8"),
        "CH4_FOSSIL": Decimal("29.8"),
        "CH4_BIOGENIC": Decimal("27.0"),
        "N2O": Decimal("273"),
        "CO": Decimal("4.06"),
    },
    "AR5": {
        "CO2": _ONE,
        "CH4": Decimal("28"),
        "CH4_FOSSIL": Decimal("28"),
        "CH4_BIOGENIC": Decimal("28"),
        "N2O": Decimal("265"),
        "CO": Decimal("1.9"),
    },
    "AR4": {
        "CO2": _ONE,
        "CH4": Decimal("25"),
        "CH4_FOSSIL": Decimal("25"),
        "CH4_BIOGENIC": Decimal("25"),
        "N2O": Decimal("298"),
        "CO": Decimal("1.9"),
    },
}


# ===========================================================================
# Pipeline Stages
# ===========================================================================


class PipelineStage(str, Enum):
    """Enumeration of the 8 pipeline stages for waste treatment."""

    VALIDATE_INPUT = "VALIDATE_INPUT"
    CLASSIFY_TREATMENT = "CLASSIFY_TREATMENT"
    LOOKUP_FACTORS = "LOOKUP_FACTORS"
    CALCULATE_BIOLOGICAL = "CALCULATE_BIOLOGICAL"
    CALCULATE_THERMAL = "CALCULATE_THERMAL"
    CALCULATE_WASTEWATER = "CALCULATE_WASTEWATER"
    CHECK_COMPLIANCE = "CHECK_COMPLIANCE"
    ASSEMBLE_RESULTS = "ASSEMBLE_RESULTS"


# ---------------------------------------------------------------------------
# Valid enumerations
# ---------------------------------------------------------------------------

#: Valid waste categories per IPCC 2006 Vol 5 and 2019 Refinement.
VALID_WASTE_CATEGORIES: List[str] = [
    "MSW", "INDUSTRIAL_WASTE", "CONSTRUCTION_DEMOLITION",
    "ORGANIC_WASTE", "FOOD_WASTE", "YARD_GARDEN_WASTE",
    "PAPER", "CARDBOARD", "PLASTIC", "METAL", "GLASS",
    "TEXTILES", "WOOD", "RUBBER", "E_WASTE",
    "HAZARDOUS_WASTE", "MEDICAL_WASTE", "SLUDGE", "MIXED_WASTE",
]

#: Valid treatment methods.
VALID_TREATMENT_METHODS: List[str] = [
    "LANDFILL", "LANDFILL_GAS_CAPTURE",
    "INCINERATION", "INCINERATION_ENERGY_RECOVERY",
    "RECYCLING",
    "COMPOSTING", "ANAEROBIC_DIGESTION", "MBT",
    "PYROLYSIS", "GASIFICATION",
    "CHEMICAL_TREATMENT", "THERMAL_TREATMENT",
    "BIOLOGICAL_TREATMENT",
    "OPEN_BURNING", "OPEN_DUMPING",
    "WASTEWATER_TREATMENT",
]

#: Treatment methods classified as biological.
BIOLOGICAL_METHODS: frozenset = frozenset({
    "COMPOSTING", "ANAEROBIC_DIGESTION", "MBT",
    "BIOLOGICAL_TREATMENT",
})

#: Treatment methods classified as thermal.
THERMAL_METHODS: frozenset = frozenset({
    "INCINERATION", "INCINERATION_ENERGY_RECOVERY",
    "PYROLYSIS", "GASIFICATION",
    "OPEN_BURNING", "THERMAL_TREATMENT",
})

#: Treatment methods classified as wastewater.
WASTEWATER_METHODS: frozenset = frozenset({
    "WASTEWATER_TREATMENT",
})

#: Valid calculation methods.
VALID_CALCULATION_METHODS: List[str] = [
    "IPCC_DEFAULT", "IPCC_TIER_2", "IPCC_TIER_3",
    "FIRST_ORDER_DECAY", "MASS_BALANCE",
    "DIRECT_MEASUREMENT", "SPEND_BASED",
]

#: Valid GWP assessment report sources.
VALID_GWP_SOURCES: List[str] = ["AR6", "AR5", "AR4"]

#: Default emission factors for biological treatment (g/kg waste)
#: IPCC 2019 Table 5.1
BIOLOGICAL_DEFAULT_EFS: Dict[str, Dict[str, Decimal]] = {
    "COMPOSTING_WELL_MANAGED": {
        "ch4_g_per_kg": Decimal("4.0"),
        "n2o_g_per_kg": Decimal("0.24"),
    },
    "COMPOSTING_POORLY_MANAGED": {
        "ch4_g_per_kg": Decimal("10.0"),
        "n2o_g_per_kg": Decimal("0.6"),
    },
    "AD_VENTED": {
        "ch4_g_per_kg": Decimal("2.0"),
        "n2o_g_per_kg": _ZERO,
    },
    "AD_FLARED": {
        "ch4_g_per_kg": Decimal("0.8"),
        "n2o_g_per_kg": _ZERO,
    },
    "MBT_AEROBIC": {
        "ch4_g_per_kg": Decimal("4.0"),
        "n2o_g_per_kg": Decimal("0.3"),
    },
    "MBT_ANAEROBIC": {
        "ch4_g_per_kg": Decimal("2.0"),
        "n2o_g_per_kg": Decimal("0.1"),
    },
}

#: Default emission factors for thermal treatment (kg/Gg waste)
#: IPCC 2006 Vol 5 Table 5.3
THERMAL_DEFAULT_EFS: Dict[str, Dict[str, Decimal]] = {
    "STOKER_GRATE": {
        "n2o_kg_per_gg": Decimal("50"),
        "ch4_kg_per_gg": Decimal("0.2"),
    },
    "FLUIDIZED_BED": {
        "n2o_kg_per_gg": Decimal("56"),
        "ch4_kg_per_gg": Decimal("0.68"),
    },
    "ROTARY_KILN": {
        "n2o_kg_per_gg": Decimal("50"),
        "ch4_kg_per_gg": Decimal("0.2"),
    },
    "SEMI_CONTINUOUS": {
        "n2o_kg_per_gg": Decimal("60"),
        "ch4_kg_per_gg": Decimal("6.0"),
    },
    "BATCH_TYPE": {
        "n2o_kg_per_gg": Decimal("60"),
        "ch4_kg_per_gg": Decimal("60"),
    },
}

#: Fossil carbon fractions by waste component (IPCC Table 5.2)
FOSSIL_CARBON_FRACTIONS: Dict[str, Decimal] = {
    "FOOD_WASTE": _ZERO,
    "PAPER": Decimal("0.01"),
    "CARDBOARD": Decimal("0.01"),
    "PLASTIC": _ONE,
    "TEXTILES_SYNTHETIC": Decimal("0.80"),
    "TEXTILES_NATURAL": _ZERO,
    "RUBBER": Decimal("0.20"),
    "WOOD": _ZERO,
    "GARDEN_WASTE": _ZERO,
    "NAPPIES": Decimal("0.10"),
    "SLUDGE": _ZERO,
    "GLASS": _ZERO,
    "METAL": _ZERO,
}

#: Carbon content of waste (fraction of wet weight) by component
CARBON_CONTENT_WET: Dict[str, Decimal] = {
    "FOOD_WASTE": Decimal("0.15"),
    "PAPER": Decimal("0.375"),
    "CARDBOARD": Decimal("0.375"),
    "PLASTIC": Decimal("0.675"),
    "TEXTILES_SYNTHETIC": Decimal("0.45"),
    "TEXTILES_NATURAL": Decimal("0.45"),
    "RUBBER": Decimal("0.50"),
    "WOOD": Decimal("0.465"),
    "GARDEN_WASTE": Decimal("0.185"),
    "NAPPIES": Decimal("0.27"),
    "SLUDGE": Decimal("0.10"),
}

#: Wastewater MCF values by treatment system (IPCC Vol 5 Ch 6)
WASTEWATER_MCF: Dict[str, Decimal] = {
    "UNTREATED_DISCHARGE": Decimal("0.1"),
    "AEROBIC_WELL_MANAGED": _ZERO,
    "AEROBIC_OVERLOADED": Decimal("0.3"),
    "ANAEROBIC_NO_RECOVERY": Decimal("0.8"),
    "ANAEROBIC_WITH_RECOVERY": Decimal("0.8"),
    "ANAEROBIC_SHALLOW_LAGOON": Decimal("0.2"),
    "ANAEROBIC_DEEP_LAGOON": Decimal("0.8"),
    "SEPTIC_SYSTEM": Decimal("0.5"),
    "LATRINE_DRY": Decimal("0.1"),
    "LATRINE_WET": Decimal("0.7"),
}

#: CH4 producing capacity Bo (kg CH4 / kg organic parameter)
WASTEWATER_BO: Dict[str, Decimal] = {
    "BOD": Decimal("0.6"),
    "COD": Decimal("0.25"),
}


# ===========================================================================
# WasteTreatmentPipelineEngine
# ===========================================================================


class WasteTreatmentPipelineEngine:
    """End-to-end orchestration pipeline for on-site waste treatment emissions.

    Coordinates WasteTreatmentDatabaseEngine, BiologicalTreatmentEngine,
    ThermalTreatmentEngine, WastewaterTreatmentEngine, and
    ComplianceCheckerEngine through an 8-stage deterministic pipeline.

    The pipeline handles multi-stream treatment facilities where waste is
    routed to different treatment methods (composting, incineration,
    wastewater treatment) simultaneously.  Each stream is calculated
    independently and results are aggregated in the ASSEMBLE_RESULTS stage.

    Thread Safety:
        All mutable state is protected by a ``threading.Lock``.

    Attributes:
        _db_engine: WasteTreatmentDatabaseEngine instance.
        _bio_engine: BiologicalTreatmentEngine instance.
        _thermal_engine: ThermalTreatmentEngine instance.
        _ww_engine: WastewaterTreatmentEngine instance.
        _compliance_engine: ComplianceCheckerEngine instance.
        _provenance_tracker: ProvenanceTracker instance.
        _lock: Thread lock for mutable state.
        _total_executions: Total pipeline executions counter.
        _stage_timings: Accumulated per-stage timing data.

    Example:
        >>> pipeline = WasteTreatmentPipelineEngine()
        >>> result = pipeline.execute(request)
    """

    def __init__(
        self,
        db_engine: Optional[Any] = None,
        bio_engine: Optional[Any] = None,
        thermal_engine: Optional[Any] = None,
        ww_engine: Optional[Any] = None,
        compliance_engine: Optional[Any] = None,
        provenance_tracker: Optional[Any] = None,
    ) -> None:
        """Initialize the WasteTreatmentPipelineEngine.

        Creates default engine instances if not provided.  Engines that
        fail to import are set to None and their stages are skipped
        gracefully with a status indicator in the result.

        Args:
            db_engine: Optional WasteTreatmentDatabaseEngine.
            bio_engine: Optional BiologicalTreatmentEngine.
            thermal_engine: Optional ThermalTreatmentEngine.
            ww_engine: Optional WastewaterTreatmentEngine.
            compliance_engine: Optional ComplianceCheckerEngine.
            provenance_tracker: Optional ProvenanceTracker instance.
        """
        # Initialize database engine
        self._db_engine = db_engine
        if self._db_engine is None and WasteTreatmentDatabaseEngine is not None:
            try:
                self._db_engine = WasteTreatmentDatabaseEngine()
            except Exception:
                self._db_engine = None

        # Initialize biological treatment engine
        self._bio_engine = bio_engine
        if self._bio_engine is None and BiologicalTreatmentEngine is not None:
            try:
                self._bio_engine = BiologicalTreatmentEngine(
                    waste_database=self._db_engine
                )
            except Exception:
                self._bio_engine = None

        # Initialize thermal treatment engine
        self._thermal_engine = thermal_engine
        if self._thermal_engine is None and ThermalTreatmentEngine is not None:
            try:
                self._thermal_engine = ThermalTreatmentEngine(
                    waste_database=self._db_engine
                )
            except Exception:
                self._thermal_engine = None

        # Initialize wastewater treatment engine
        self._ww_engine = ww_engine
        if self._ww_engine is None and WastewaterTreatmentEngine is not None:
            try:
                self._ww_engine = WastewaterTreatmentEngine()
            except Exception:
                self._ww_engine = None

        # Initialize compliance engine
        self._compliance_engine = compliance_engine
        if self._compliance_engine is None and ComplianceCheckerEngine is not None:
            self._compliance_engine = ComplianceCheckerEngine()

        # Initialize provenance tracker
        self._provenance_tracker = provenance_tracker
        if self._provenance_tracker is None and ProvenanceTracker is not None:
            self._provenance_tracker = ProvenanceTracker()

        # Thread-safe counters and timing accumulators
        self._lock = threading.Lock()
        self._total_executions: int = 0
        self._total_batches: int = 0
        self._stage_timings: Dict[str, List[float]] = {
            stage.value: [] for stage in PipelineStage
        }
        self._created_at = _utcnow()

        engine_status = {
            "db": self._db_engine is not None,
            "bio": self._bio_engine is not None,
            "thermal": self._thermal_engine is not None,
            "ww": self._ww_engine is not None,
            "compliance": self._compliance_engine is not None,
            "provenance": self._provenance_tracker is not None,
        }

        logger.info(
            "WasteTreatmentPipelineEngine initialized: stages=%d, engines=%s",
            len(PipelineStage), engine_status,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_stage_timing(self, stage: str, elapsed_ms: float) -> None:
        """Record timing for a pipeline stage.

        Args:
            stage: Pipeline stage name.
            elapsed_ms: Elapsed time in milliseconds.
        """
        with self._lock:
            self._stage_timings[stage].append(elapsed_ms)

    def _run_stage(
        self,
        stage: PipelineStage,
        context: Dict[str, Any],
        stage_func: Any,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Execute a single pipeline stage with timing and error handling.

        Each stage is wrapped in timing, error capture, and provenance
        hash generation.  On success the stage name is appended to
        ``stages_completed``; on failure it goes to ``stages_failed``
        and the error message is recorded.

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

    def _get_gwp(
        self,
        gas: str,
        gwp_source: str,
    ) -> Decimal:
        """Look up GWP value for a gas and assessment report.

        Args:
            gas: Gas identifier (CO2, CH4, N2O, CH4_FOSSIL, CH4_BIOGENIC, CO).
            gwp_source: Assessment report (AR6, AR5, AR4).

        Returns:
            GWP-100 value as Decimal.
        """
        source_gwps = GWP_VALUES.get(gwp_source, GWP_VALUES["AR6"])
        return source_gwps.get(gas, _ONE)

    def _classify_stream_method(self, method: str) -> str:
        """Classify a treatment method into a category.

        Args:
            method: Treatment method string (uppercase).

        Returns:
            One of BIOLOGICAL, THERMAL, WASTEWATER, or OTHER.
        """
        if method in BIOLOGICAL_METHODS:
            return "BIOLOGICAL"
        if method in THERMAL_METHODS:
            return "THERMAL"
        if method in WASTEWATER_METHODS:
            return "WASTEWATER"
        return "OTHER"

    # ------------------------------------------------------------------
    # Stage 1: Validate Input
    # ------------------------------------------------------------------

    def _stage_validate_input(self, ctx: Dict[str, Any]) -> None:
        """Stage 1: Validate input request.

        Checks required fields, validates enums, normalises values,
        and verifies that waste composition percentages sum to 100%.

        Raises:
            ValueError: If validation fails with accumulated errors.
        """
        request = ctx["request"]
        errors: List[str] = []

        # Tenant ID
        tenant_id = str(request.get("tenant_id", "")).strip()
        if not tenant_id:
            errors.append("tenant_id is required")
        ctx["tenant_id"] = tenant_id

        # Facility ID (optional but recommended)
        ctx["facility_id"] = str(request.get("facility_id", "")).strip()

        # GWP source
        gwp_source = str(request.get("gwp_source", "AR6")).upper()
        if gwp_source not in VALID_GWP_SOURCES:
            errors.append(
                f"Invalid gwp_source: {gwp_source}. "
                f"Valid: {VALID_GWP_SOURCES}"
            )
        ctx["gwp_source"] = gwp_source

        # Calculation method
        calc_method = str(
            request.get("calculation_method", "IPCC_DEFAULT")
        ).upper()
        if calc_method not in VALID_CALCULATION_METHODS:
            errors.append(
                f"Invalid calculation_method: {calc_method}. "
                f"Valid: {VALID_CALCULATION_METHODS}"
            )
        ctx["calculation_method"] = calc_method

        # Frameworks
        ctx["frameworks"] = request.get("frameworks", [])

        # Reporting year
        ctx["reporting_year"] = request.get("reporting_year", _utcnow().year)

        # Treatment streams validation
        streams = request.get("treatment_streams", [])
        if not streams:
            errors.append("treatment_streams is required and must not be empty")

        validated_streams: List[Dict[str, Any]] = []
        for i, stream in enumerate(streams):
            stream_errors = self._validate_stream(stream, i)
            errors.extend(stream_errors)
            if not stream_errors:
                validated_streams.append(self._normalize_stream(stream))

        ctx["treatment_streams"] = validated_streams
        ctx["stream_count"] = len(validated_streams)

        # Wastewater parameters (optional top-level)
        ctx["wastewater_params"] = request.get("wastewater_params", {})

        # Methane recovery parameters (optional top-level)
        ctx["methane_recovery_params"] = request.get(
            "methane_recovery_params", {}
        )

        # Energy recovery parameters (optional top-level)
        ctx["energy_recovery_params"] = request.get(
            "energy_recovery_params", {}
        )

        if errors:
            ctx["validation_errors"] = errors
            raise ValueError(f"Validation failed: {errors}")

        ctx["validation_status"] = "PASSED"
        logger.debug(
            "Validation passed: %d streams, tenant=%s",
            len(validated_streams), tenant_id,
        )

    def _validate_stream(
        self, stream: Dict[str, Any], index: int
    ) -> List[str]:
        """Validate a single treatment stream.

        Args:
            stream: Stream dictionary from the request.
            index: Zero-based index of the stream in the array.

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []
        prefix = f"treatment_streams[{index}]"

        # Stream ID
        stream_id = str(stream.get("stream_id", "")).strip()
        if not stream_id:
            errors.append(f"{prefix}.stream_id is required")

        # Treatment method
        method = str(stream.get("treatment_method", "")).upper()
        if not method:
            errors.append(f"{prefix}.treatment_method is required")
        elif method not in VALID_TREATMENT_METHODS:
            errors.append(
                f"{prefix}.treatment_method '{method}' is invalid. "
                f"Valid: {VALID_TREATMENT_METHODS}"
            )

        # Waste category
        category = str(stream.get("waste_category", "")).upper()
        if not category:
            errors.append(f"{prefix}.waste_category is required")
        elif category not in VALID_WASTE_CATEGORIES:
            errors.append(
                f"{prefix}.waste_category '{category}' is invalid. "
                f"Valid: {VALID_WASTE_CATEGORIES}"
            )

        # Waste mass
        mass = _safe_decimal(stream.get("waste_mass_tonnes"))
        if mass <= _ZERO:
            errors.append(f"{prefix}.waste_mass_tonnes must be > 0")

        # Composition validation (must sum to ~100%)
        composition = stream.get("composition", {})
        if composition:
            total_pct = sum(
                _safe_decimal(v) for v in composition.values()
            )
            # Allow tolerance of +/- 1%
            if abs(total_pct - Decimal("100")) > _ONE:
                errors.append(
                    f"{prefix}.composition must sum to 100% "
                    f"(+/- 1% tolerance); got {total_pct}%"
                )

        return errors

    def _normalize_stream(self, stream: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a validated treatment stream into canonical form.

        Args:
            stream: Raw stream dictionary from the request.

        Returns:
            Normalized stream dictionary with uppercase enums and
            Decimal-converted numeric fields.
        """
        method = str(stream.get("treatment_method", "")).upper()
        category = str(stream.get("waste_category", "")).upper()
        mass = _safe_decimal(stream.get("waste_mass_tonnes"))

        normalized: Dict[str, Any] = {
            "stream_id": str(stream.get("stream_id", "")),
            "treatment_method": method,
            "waste_category": category,
            "waste_mass_tonnes": mass,
            "composition": stream.get("composition", {}),
            "method_class": self._classify_stream_method(method),
            "management_quality": str(
                stream.get("management_quality", "WELL_MANAGED")
            ).upper(),
            "technology_type": str(
                stream.get("technology_type", "")
            ).upper(),
            "moisture_content_pct": _safe_decimal(
                stream.get("moisture_content_pct")
            ),
            "dry_matter_fraction": _safe_decimal(
                stream.get("dry_matter_fraction"),
                default=Decimal("0.75"),
            ),
            "oxidation_factor": _safe_decimal(
                stream.get("oxidation_factor"),
                default=_ONE,
            ),
        }

        # Biological-specific fields
        if normalized["method_class"] == "BIOLOGICAL":
            normalized["volatile_solids_fraction"] = _safe_decimal(
                stream.get("volatile_solids_fraction"),
                default=Decimal("0.6"),
            )
            normalized["bmp_m3_per_tonne_vs"] = _safe_decimal(
                stream.get("bmp_m3_per_tonne_vs"),
                default=Decimal("300"),
            )
            normalized["digestion_efficiency"] = _safe_decimal(
                stream.get("digestion_efficiency"),
                default=Decimal("0.7"),
            )
            normalized["ch4_fraction_biogas"] = _safe_decimal(
                stream.get("ch4_fraction_biogas"),
                default=Decimal("0.55"),
            )
            normalized["biofilter_efficiency"] = _safe_decimal(
                stream.get("biofilter_efficiency"),
                default=_ZERO,
            )

        # Thermal-specific fields
        if normalized["method_class"] == "THERMAL":
            normalized["ncv_gj_per_tonne"] = _safe_decimal(
                stream.get("ncv_gj_per_tonne"),
                default=Decimal("10"),
            )
            normalized["energy_recovery_efficiency"] = _safe_decimal(
                stream.get("energy_recovery_efficiency"),
                default=_ZERO,
            )
            normalized["grid_ef_tco2e_per_gj"] = _safe_decimal(
                stream.get("grid_ef_tco2e_per_gj"),
                default=Decimal("0.1"),
            )
            normalized["open_burning_of"] = _safe_decimal(
                stream.get("open_burning_of"),
                default=Decimal("0.58"),
            )

        # Wastewater-specific fields
        if normalized["method_class"] == "WASTEWATER":
            normalized["organic_load_type"] = str(
                stream.get("organic_load_type", "BOD")
            ).upper()
            normalized["organic_load_kg_per_yr"] = _safe_decimal(
                stream.get("organic_load_kg_per_yr")
            )
            normalized["sludge_removed_kg"] = _safe_decimal(
                stream.get("sludge_removed_kg")
            )
            normalized["treatment_system"] = str(
                stream.get("treatment_system", "AEROBIC_WELL_MANAGED")
            ).upper()
            normalized["population_equivalent"] = _safe_decimal(
                stream.get("population_equivalent")
            )
            normalized["protein_consumption_kg_yr"] = _safe_decimal(
                stream.get("protein_consumption_kg_yr"),
                default=Decimal("25550"),
            )

        # Methane recovery fields (per-stream overrides)
        normalized["collection_efficiency"] = _safe_decimal(
            stream.get("collection_efficiency"),
            default=_ZERO,
        )
        normalized["flare_fraction"] = _safe_decimal(
            stream.get("flare_fraction"),
            default=_ZERO,
        )
        normalized["utilization_fraction"] = _safe_decimal(
            stream.get("utilization_fraction"),
            default=_ZERO,
        )
        normalized["vent_fraction"] = _safe_decimal(
            stream.get("vent_fraction"),
            default=_ZERO,
        )
        normalized["flare_destruction_efficiency"] = _safe_decimal(
            stream.get("flare_destruction_efficiency"),
            default=Decimal("0.98"),
        )
        normalized["utilization_conversion_efficiency"] = _safe_decimal(
            stream.get("utilization_conversion_efficiency"),
            default=Decimal("0.95"),
        )

        return normalized

    # ------------------------------------------------------------------
    # Stage 2: Classify Treatment
    # ------------------------------------------------------------------

    def _stage_classify_treatment(self, ctx: Dict[str, Any]) -> None:
        """Stage 2: Classify each treatment stream.

        Determines which calculation engine each stream maps to and
        identifies the facility type based on the mix of treatment methods.
        """
        streams = ctx["treatment_streams"]
        biological_streams: List[Dict[str, Any]] = []
        thermal_streams: List[Dict[str, Any]] = []
        wastewater_streams: List[Dict[str, Any]] = []
        other_streams: List[Dict[str, Any]] = []

        for stream in streams:
            method_class = stream["method_class"]
            if method_class == "BIOLOGICAL":
                biological_streams.append(stream)
            elif method_class == "THERMAL":
                thermal_streams.append(stream)
            elif method_class == "WASTEWATER":
                wastewater_streams.append(stream)
            else:
                other_streams.append(stream)

        ctx["biological_streams"] = biological_streams
        ctx["thermal_streams"] = thermal_streams
        ctx["wastewater_streams"] = wastewater_streams
        ctx["other_streams"] = other_streams

        # Determine facility type based on treatment mix
        facility_types = set()
        if biological_streams:
            facility_types.add("BIOLOGICAL_TREATMENT_FACILITY")
        if thermal_streams:
            facility_types.add("THERMAL_TREATMENT_FACILITY")
        if wastewater_streams:
            facility_types.add("WASTEWATER_TREATMENT_PLANT")
        if not facility_types:
            facility_types.add("GENERAL_WASTE_FACILITY")

        ctx["facility_types"] = sorted(facility_types)

        # Auto-classify waste categories present
        categories = sorted(set(
            s["waste_category"] for s in streams
        ))
        ctx["waste_categories_present"] = categories

        # Total waste mass
        total_mass = sum(s["waste_mass_tonnes"] for s in streams)
        ctx["total_waste_mass_tonnes"] = total_mass

        # Auto-classify using database engine if available
        if self._db_engine is not None:
            for stream in streams:
                try:
                    db_classification = self._db_engine.classify_waste(
                        stream["waste_category"],
                        stream.get("composition", {}),
                    )
                    stream["db_classification"] = db_classification
                except Exception as e:
                    stream["db_classification"] = {
                        "status": "ERROR", "error": str(e)
                    }

        ctx["classification_status"] = "COMPLETE"
        logger.debug(
            "Classification: bio=%d, thermal=%d, ww=%d, other=%d, "
            "total_mass=%s t",
            len(biological_streams), len(thermal_streams),
            len(wastewater_streams), len(other_streams),
            total_mass,
        )

    # ------------------------------------------------------------------
    # Stage 3: Lookup Factors
    # ------------------------------------------------------------------

    def _stage_lookup_factors(self, ctx: Dict[str, Any]) -> None:
        """Stage 3: Look up emission factors, DOC, MCF, NCV from database.

        Fetches per-stream emission factors.  Falls back to built-in
        IPCC defaults when the database engine is unavailable.
        """
        streams = ctx["treatment_streams"]
        factors_by_stream: Dict[str, Dict[str, Any]] = {}

        for stream in streams:
            stream_id = stream["stream_id"]
            method = stream["treatment_method"]
            category = stream["waste_category"]

            if self._db_engine is not None:
                try:
                    factors = self._db_engine.get_emission_factors(
                        treatment_method=method,
                        waste_category=category,
                    )
                    factors_by_stream[stream_id] = factors
                    continue
                except Exception as e:
                    logger.warning(
                        "DB factor lookup failed for stream %s: %s",
                        stream_id, str(e),
                    )

            # Fallback to built-in defaults
            factors_by_stream[stream_id] = self._get_default_factors(
                stream
            )

        ctx["factors_by_stream"] = factors_by_stream
        ctx["factors_status"] = "COMPLETE"
        logger.debug(
            "Factors retrieved for %d streams",
            len(factors_by_stream),
        )

    def _get_default_factors(
        self, stream: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get built-in IPCC default factors for a stream.

        Args:
            stream: Normalized stream dictionary.

        Returns:
            Dictionary of emission factors for the stream.
        """
        method = stream["treatment_method"]
        method_class = stream["method_class"]
        management = stream.get("management_quality", "WELL_MANAGED")

        factors: Dict[str, Any] = {
            "source": "IPCC_2006_DEFAULTS",
            "treatment_method": method,
        }

        if method_class == "BIOLOGICAL":
            bio_key = self._resolve_bio_ef_key(method, management)
            bio_efs = BIOLOGICAL_DEFAULT_EFS.get(
                bio_key,
                BIOLOGICAL_DEFAULT_EFS["COMPOSTING_WELL_MANAGED"],
            )
            factors.update({
                "ch4_g_per_kg": str(bio_efs["ch4_g_per_kg"]),
                "n2o_g_per_kg": str(bio_efs["n2o_g_per_kg"]),
            })

        elif method_class == "THERMAL":
            tech = stream.get("technology_type", "STOKER_GRATE")
            if not tech:
                tech = "STOKER_GRATE"
            thermal_efs = THERMAL_DEFAULT_EFS.get(
                tech,
                THERMAL_DEFAULT_EFS["STOKER_GRATE"],
            )
            factors.update({
                "n2o_kg_per_gg": str(thermal_efs["n2o_kg_per_gg"]),
                "ch4_kg_per_gg": str(thermal_efs["ch4_kg_per_gg"]),
            })

        elif method_class == "WASTEWATER":
            ts = stream.get("treatment_system", "AEROBIC_WELL_MANAGED")
            mcf = WASTEWATER_MCF.get(ts, _ZERO)
            org_type = stream.get("organic_load_type", "BOD")
            bo = WASTEWATER_BO.get(org_type, Decimal("0.6"))
            factors.update({
                "mcf": str(mcf),
                "bo": str(bo),
                "treatment_system": ts,
            })

        return factors

    def _resolve_bio_ef_key(
        self, method: str, management: str
    ) -> str:
        """Resolve the biological EF lookup key.

        Args:
            method: Treatment method (COMPOSTING, ANAEROBIC_DIGESTION, MBT).
            management: Management quality level.

        Returns:
            Key into BIOLOGICAL_DEFAULT_EFS.
        """
        if method == "COMPOSTING":
            if management == "POORLY_MANAGED":
                return "COMPOSTING_POORLY_MANAGED"
            return "COMPOSTING_WELL_MANAGED"
        if method == "ANAEROBIC_DIGESTION":
            if management in ("VENTED", "POORLY_MANAGED"):
                return "AD_VENTED"
            return "AD_FLARED"
        if method == "MBT":
            if management in ("ANAEROBIC", "ANAEROBIC_PRE_TREATMENT"):
                return "MBT_ANAEROBIC"
            return "MBT_AEROBIC"
        return "COMPOSTING_WELL_MANAGED"

    # ------------------------------------------------------------------
    # Stage 4: Calculate Biological
    # ------------------------------------------------------------------

    def _stage_calculate_biological(self, ctx: Dict[str, Any]) -> None:
        """Stage 4: Calculate biological treatment emissions.

        Handles composting CH4/N2O, anaerobic digestion biogas with
        methane recovery, and MBT emissions.  Uses the
        BiologicalTreatmentEngine if available, otherwise falls back
        to IPCC Tier 1 defaults.
        """
        bio_streams = ctx.get("biological_streams", [])
        if not bio_streams:
            ctx["biological_results"] = {
                "status": "SKIPPED",
                "reason": "No biological treatment streams",
            }
            return

        gwp_source = ctx["gwp_source"]
        ch4_gwp = self._get_gwp("CH4_BIOGENIC", gwp_source)
        n2o_gwp = self._get_gwp("N2O", gwp_source)

        stream_results: List[Dict[str, Any]] = []
        total_ch4 = _ZERO
        total_n2o = _ZERO
        total_ch4_co2e = _ZERO
        total_n2o_co2e = _ZERO
        total_ch4_generated = _ZERO
        total_ch4_captured = _ZERO
        total_ch4_flared = _ZERO
        total_ch4_utilized = _ZERO

        for stream in bio_streams:
            sid = stream["stream_id"]
            factors = ctx.get("factors_by_stream", {}).get(sid, {})

            if self._bio_engine is not None:
                try:
                    result = self._bio_engine.calculate(stream, factors)
                    stream_results.append(result)
                    total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
                    total_n2o += _safe_decimal(result.get("n2o_tonnes"))
                    total_ch4_co2e += _safe_decimal(result.get("ch4_co2e"))
                    total_n2o_co2e += _safe_decimal(result.get("n2o_co2e"))
                    total_ch4_generated += _safe_decimal(
                        result.get("ch4_generated_tonnes")
                    )
                    total_ch4_captured += _safe_decimal(
                        result.get("ch4_captured_tonnes")
                    )
                    total_ch4_flared += _safe_decimal(
                        result.get("ch4_flared_tonnes")
                    )
                    total_ch4_utilized += _safe_decimal(
                        result.get("ch4_utilized_tonnes")
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        "Bio engine failed for stream %s: %s, "
                        "falling back to defaults",
                        sid, str(e),
                    )

            # Fallback IPCC Tier 1 calculation
            result = self._calculate_bio_default(
                stream, factors, ch4_gwp, n2o_gwp
            )
            stream_results.append(result)
            total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
            total_n2o += _safe_decimal(result.get("n2o_tonnes"))
            total_ch4_co2e += _safe_decimal(result.get("ch4_co2e"))
            total_n2o_co2e += _safe_decimal(result.get("n2o_co2e"))
            total_ch4_generated += _safe_decimal(
                result.get("ch4_generated_tonnes")
            )
            total_ch4_captured += _safe_decimal(
                result.get("ch4_captured_tonnes")
            )
            total_ch4_flared += _safe_decimal(
                result.get("ch4_flared_tonnes")
            )
            total_ch4_utilized += _safe_decimal(
                result.get("ch4_utilized_tonnes")
            )

        ctx["biological_results"] = {
            "status": "COMPLETE",
            "stream_count": len(stream_results),
            "streams": stream_results,
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
            "total_ch4_co2e": str(_quantize(total_ch4_co2e)),
            "total_n2o_co2e": str(_quantize(total_n2o_co2e)),
            "total_co2e": str(_quantize(total_ch4_co2e + total_n2o_co2e)),
            "ch4_generated_tonnes": str(_quantize(total_ch4_generated)),
            "ch4_captured_tonnes": str(_quantize(total_ch4_captured)),
            "ch4_flared_tonnes": str(_quantize(total_ch4_flared)),
            "ch4_utilized_tonnes": str(_quantize(total_ch4_utilized)),
        }
        logger.debug(
            "Biological calc: %d streams, ch4=%s t, n2o=%s t",
            len(stream_results), total_ch4, total_n2o,
        )

    def _calculate_bio_default(
        self,
        stream: Dict[str, Any],
        factors: Dict[str, Any],
        ch4_gwp: Decimal,
        n2o_gwp: Decimal,
    ) -> Dict[str, Any]:
        """Calculate biological emissions using IPCC Tier 1 defaults.

        CH4_emitted = M_organic * EF_CH4 * (1 - R_CH4) - CH4_recovered
        N2O_emitted = M_organic * EF_N2O

        Args:
            stream: Normalized stream dictionary.
            factors: Emission factors for this stream.
            ch4_gwp: GWP for CH4.
            n2o_gwp: GWP for N2O.

        Returns:
            Dictionary of biological emission results.
        """
        mass_tonnes = stream["waste_mass_tonnes"]
        mass_kg = mass_tonnes * _THOUSAND

        # Emission factors (g/kg waste -> tonnes/tonne waste)
        ef_ch4 = _safe_decimal(factors.get("ch4_g_per_kg", "4.0"))
        ef_n2o = _safe_decimal(factors.get("n2o_g_per_kg", "0.24"))

        # CH4 generated (tonnes) = mass_kg * EF_CH4 / 1,000,000
        ch4_generated = _quantize(
            mass_kg * ef_ch4 / Decimal("1000000")
        )

        # Biofilter efficiency (composting) or recovery
        biofilter_eff = stream.get("biofilter_efficiency", _ZERO)
        ch4_after_biofilter = _quantize(
            ch4_generated * (_ONE - biofilter_eff)
        )

        # Methane recovery
        collection_eff = stream.get("collection_efficiency", _ZERO)
        ch4_captured = _quantize(ch4_after_biofilter * collection_eff)

        flare_frac = stream.get("flare_fraction", _ZERO)
        util_frac = stream.get("utilization_fraction", _ZERO)
        vent_frac = stream.get("vent_fraction", _ZERO)
        flare_dest_eff = stream.get(
            "flare_destruction_efficiency", Decimal("0.98")
        )
        util_conv_eff = stream.get(
            "utilization_conversion_efficiency", Decimal("0.95")
        )

        ch4_flared = _quantize(ch4_captured * flare_frac)
        ch4_utilized = _quantize(ch4_captured * util_frac)
        ch4_vented = _quantize(ch4_captured * vent_frac)

        # CH4 from flare that is NOT destroyed
        ch4_from_flare = _quantize(ch4_flared * (_ONE - flare_dest_eff))
        # CH4 from utilization that is NOT converted
        ch4_from_util = _quantize(
            ch4_utilized * (_ONE - util_conv_eff)
        )

        # Net CH4 emitted
        ch4_uncaptured = ch4_after_biofilter - ch4_captured
        ch4_emitted = _quantize(
            ch4_uncaptured + ch4_from_flare + ch4_from_util + ch4_vented
        )

        # N2O generated (tonnes)
        n2o_generated = _quantize(
            mass_kg * ef_n2o / Decimal("1000000")
        )

        # CO2e
        ch4_co2e = _quantize(ch4_emitted * ch4_gwp)
        n2o_co2e = _quantize(n2o_generated * n2o_gwp)
        total_co2e = _quantize(ch4_co2e + n2o_co2e)

        return {
            "stream_id": stream["stream_id"],
            "treatment_method": stream["treatment_method"],
            "waste_category": stream["waste_category"],
            "waste_mass_tonnes": str(mass_tonnes),
            "calculation_method": "IPCC_TIER_1_DEFAULT",
            "ch4_generated_tonnes": str(ch4_generated),
            "ch4_captured_tonnes": str(ch4_captured),
            "ch4_flared_tonnes": str(ch4_flared),
            "ch4_utilized_tonnes": str(ch4_utilized),
            "ch4_vented_tonnes": str(ch4_vented),
            "ch4_emitted_tonnes": str(ch4_emitted),
            "ch4_tonnes": str(ch4_emitted),
            "n2o_tonnes": str(n2o_generated),
            "ch4_co2e": str(ch4_co2e),
            "n2o_co2e": str(n2o_co2e),
            "total_co2e": str(total_co2e),
            "fossil_co2_tonnes": "0",
            "biogenic_co2_tonnes": "0",
            "ef_ch4_g_per_kg": str(ef_ch4),
            "ef_n2o_g_per_kg": str(ef_n2o),
            "gwp_ch4": str(ch4_gwp),
            "gwp_n2o": str(n2o_gwp),
            "provenance_hash": _compute_hash({
                "stream_id": stream["stream_id"],
                "mass": str(mass_tonnes),
                "ch4": str(ch4_emitted),
                "n2o": str(n2o_generated),
            }),
        }

    # ------------------------------------------------------------------
    # Stage 5: Calculate Thermal
    # ------------------------------------------------------------------

    def _stage_calculate_thermal(self, ctx: Dict[str, Any]) -> None:
        """Stage 5: Calculate thermal treatment emissions.

        Handles incineration (fossil/biogenic CO2 split), pyrolysis,
        gasification, and open burning.  Uses the ThermalTreatmentEngine
        if available, otherwise falls back to IPCC defaults with
        composition-based fossil/biogenic carbon separation.
        """
        thermal_streams = ctx.get("thermal_streams", [])
        if not thermal_streams:
            ctx["thermal_results"] = {
                "status": "SKIPPED",
                "reason": "No thermal treatment streams",
            }
            return

        gwp_source = ctx["gwp_source"]
        ch4_gwp = self._get_gwp("CH4_FOSSIL", gwp_source)
        n2o_gwp = self._get_gwp("N2O", gwp_source)

        stream_results: List[Dict[str, Any]] = []
        total_fossil_co2 = _ZERO
        total_biogenic_co2 = _ZERO
        total_ch4 = _ZERO
        total_n2o = _ZERO
        total_ch4_co2e = _ZERO
        total_n2o_co2e = _ZERO
        total_energy_recovered = _ZERO
        total_displaced_emissions = _ZERO

        for stream in thermal_streams:
            sid = stream["stream_id"]
            factors = ctx.get("factors_by_stream", {}).get(sid, {})

            if self._thermal_engine is not None:
                try:
                    result = self._thermal_engine.calculate(stream, factors)
                    stream_results.append(result)
                    total_fossil_co2 += _safe_decimal(
                        result.get("fossil_co2_tonnes")
                    )
                    total_biogenic_co2 += _safe_decimal(
                        result.get("biogenic_co2_tonnes")
                    )
                    total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
                    total_n2o += _safe_decimal(result.get("n2o_tonnes"))
                    total_ch4_co2e += _safe_decimal(
                        result.get("ch4_co2e")
                    )
                    total_n2o_co2e += _safe_decimal(
                        result.get("n2o_co2e")
                    )
                    total_energy_recovered += _safe_decimal(
                        result.get("energy_recovered_gj")
                    )
                    total_displaced_emissions += _safe_decimal(
                        result.get("displaced_emissions_tco2e")
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        "Thermal engine failed for stream %s: %s, "
                        "falling back to defaults",
                        sid, str(e),
                    )

            # Fallback IPCC Tier 2 calculation
            result = self._calculate_thermal_default(
                stream, factors, ch4_gwp, n2o_gwp
            )
            stream_results.append(result)
            total_fossil_co2 += _safe_decimal(
                result.get("fossil_co2_tonnes")
            )
            total_biogenic_co2 += _safe_decimal(
                result.get("biogenic_co2_tonnes")
            )
            total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
            total_n2o += _safe_decimal(result.get("n2o_tonnes"))
            total_ch4_co2e += _safe_decimal(result.get("ch4_co2e"))
            total_n2o_co2e += _safe_decimal(result.get("n2o_co2e"))
            total_energy_recovered += _safe_decimal(
                result.get("energy_recovered_gj")
            )
            total_displaced_emissions += _safe_decimal(
                result.get("displaced_emissions_tco2e")
            )

        ctx["thermal_results"] = {
            "status": "COMPLETE",
            "stream_count": len(stream_results),
            "streams": stream_results,
            "total_fossil_co2_tonnes": str(_quantize(total_fossil_co2)),
            "total_biogenic_co2_tonnes": str(_quantize(total_biogenic_co2)),
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
            "total_ch4_co2e": str(_quantize(total_ch4_co2e)),
            "total_n2o_co2e": str(_quantize(total_n2o_co2e)),
            "total_co2e_fossil_only": str(_quantize(
                total_fossil_co2 + total_ch4_co2e + total_n2o_co2e
            )),
            "total_co2e_gross": str(_quantize(
                total_fossil_co2 + total_biogenic_co2
                + total_ch4_co2e + total_n2o_co2e
            )),
            "energy_recovered_gj": str(_quantize(total_energy_recovered)),
            "displaced_emissions_tco2e": str(
                _quantize(total_displaced_emissions)
            ),
        }
        logger.debug(
            "Thermal calc: %d streams, fossil_co2=%s t, biogenic_co2=%s t",
            len(stream_results), total_fossil_co2, total_biogenic_co2,
        )

    def _calculate_thermal_default(
        self,
        stream: Dict[str, Any],
        factors: Dict[str, Any],
        ch4_gwp: Decimal,
        n2o_gwp: Decimal,
    ) -> Dict[str, Any]:
        """Calculate thermal emissions using IPCC Tier 2 defaults.

        CO2_fossil = Sum_j [ IW_j * CCW_j * FCF_j * OF_j ] * 44/12
        CO2_biogenic = Sum_j [ IW_j * CCW_j * (1-FCF_j) * OF_j ] * 44/12

        Args:
            stream: Normalized stream dictionary.
            factors: Emission factors for this stream.
            ch4_gwp: GWP for CH4.
            n2o_gwp: GWP for N2O.

        Returns:
            Dictionary of thermal emission results.
        """
        mass_tonnes = stream["waste_mass_tonnes"]
        method = stream["treatment_method"]
        composition = stream.get("composition", {})
        oxidation_factor = stream.get("oxidation_factor", _ONE)

        # Use open burning oxidation factor if applicable
        if method == "OPEN_BURNING":
            oxidation_factor = stream.get(
                "open_burning_of", Decimal("0.58")
            )

        fossil_co2 = _ZERO
        biogenic_co2 = _ZERO

        if composition:
            # Per-component calculation
            for component, pct in composition.items():
                comp_upper = component.upper().replace(" ", "_")
                frac = _safe_decimal(pct) / Decimal("100")
                comp_mass = mass_tonnes * frac

                cc = CARBON_CONTENT_WET.get(comp_upper, Decimal("0.15"))
                fcf = FOSSIL_CARBON_FRACTIONS.get(comp_upper, _ZERO)

                # Fossil CO2 from this component
                fossil_co2 += comp_mass * cc * fcf * oxidation_factor * _CO2_C_RATIO

                # Biogenic CO2 from this component
                biogenic_co2 += (
                    comp_mass * cc * (_ONE - fcf) * oxidation_factor * _CO2_C_RATIO
                )
        else:
            # Simplified: assume average carbon content and fossil fraction
            avg_cc = Decimal("0.25")
            avg_fcf = Decimal("0.40")
            fossil_co2 = (
                mass_tonnes * avg_cc * avg_fcf * oxidation_factor * _CO2_C_RATIO
            )
            biogenic_co2 = (
                mass_tonnes * avg_cc * (_ONE - avg_fcf) * oxidation_factor
                * _CO2_C_RATIO
            )

        fossil_co2 = _quantize(fossil_co2)
        biogenic_co2 = _quantize(biogenic_co2)

        # CH4 and N2O from thermal treatment
        tech = stream.get("technology_type", "STOKER_GRATE")
        if not tech:
            tech = "STOKER_GRATE"
        thermal_efs = THERMAL_DEFAULT_EFS.get(
            tech, THERMAL_DEFAULT_EFS["STOKER_GRATE"]
        )
        n2o_kg_per_gg = _safe_decimal(
            factors.get("n2o_kg_per_gg", thermal_efs["n2o_kg_per_gg"])
        )
        ch4_kg_per_gg = _safe_decimal(
            factors.get("ch4_kg_per_gg", thermal_efs["ch4_kg_per_gg"])
        )

        # Convert mass to Gg (1 Gg = 1000 tonnes)
        mass_gg = mass_tonnes / _THOUSAND

        ch4_tonnes = _quantize(mass_gg * ch4_kg_per_gg / _THOUSAND)
        n2o_tonnes = _quantize(mass_gg * n2o_kg_per_gg / _THOUSAND)

        ch4_co2e = _quantize(ch4_tonnes * ch4_gwp)
        n2o_co2e = _quantize(n2o_tonnes * n2o_gwp)

        # Energy recovery
        energy_recovered = _ZERO
        displaced_emissions = _ZERO
        if method in ("INCINERATION_ENERGY_RECOVERY", "GASIFICATION", "PYROLYSIS"):
            ncv = stream.get("ncv_gj_per_tonne", Decimal("10"))
            eta_recovery = stream.get(
                "energy_recovery_efficiency", Decimal("0.25")
            )
            grid_ef = stream.get(
                "grid_ef_tco2e_per_gj", Decimal("0.1")
            )
            energy_recovered = _quantize(mass_tonnes * ncv * eta_recovery)
            displaced_emissions = _quantize(energy_recovered * grid_ef)

        total_co2e = _quantize(
            fossil_co2 + ch4_co2e + n2o_co2e
        )

        return {
            "stream_id": stream["stream_id"],
            "treatment_method": method,
            "waste_category": stream["waste_category"],
            "waste_mass_tonnes": str(mass_tonnes),
            "calculation_method": "IPCC_TIER_2_DEFAULT",
            "fossil_co2_tonnes": str(fossil_co2),
            "biogenic_co2_tonnes": str(biogenic_co2),
            "ch4_tonnes": str(ch4_tonnes),
            "n2o_tonnes": str(n2o_tonnes),
            "ch4_co2e": str(ch4_co2e),
            "n2o_co2e": str(n2o_co2e),
            "total_co2e": str(total_co2e),
            "energy_recovered_gj": str(energy_recovered),
            "displaced_emissions_tco2e": str(displaced_emissions),
            "technology_type": tech,
            "oxidation_factor": str(oxidation_factor),
            "gwp_ch4": str(ch4_gwp),
            "gwp_n2o": str(n2o_gwp),
            "provenance_hash": _compute_hash({
                "stream_id": stream["stream_id"],
                "mass": str(mass_tonnes),
                "fossil_co2": str(fossil_co2),
                "biogenic_co2": str(biogenic_co2),
            }),
        }

    # ------------------------------------------------------------------
    # Stage 6: Calculate Wastewater
    # ------------------------------------------------------------------

    def _stage_calculate_wastewater(self, ctx: Dict[str, Any]) -> None:
        """Stage 6: Calculate wastewater treatment emissions.

        Handles on-site industrial wastewater CH4 and N2O using
        IPCC Vol 5 Ch 6 methods.  Uses the WastewaterTreatmentEngine
        if available, otherwise falls back to built-in calculations.
        """
        ww_streams = ctx.get("wastewater_streams", [])
        # Also check top-level wastewater params
        ww_params = ctx.get("wastewater_params", {})

        if not ww_streams and not ww_params:
            ctx["wastewater_results"] = {
                "status": "SKIPPED",
                "reason": "No wastewater treatment streams",
            }
            return

        gwp_source = ctx["gwp_source"]
        ch4_gwp = self._get_gwp("CH4_BIOGENIC", gwp_source)
        n2o_gwp = self._get_gwp("N2O", gwp_source)

        stream_results: List[Dict[str, Any]] = []
        total_ch4 = _ZERO
        total_n2o = _ZERO
        total_ch4_co2e = _ZERO
        total_n2o_co2e = _ZERO
        total_ch4_recovered = _ZERO

        for stream in ww_streams:
            sid = stream["stream_id"]
            factors = ctx.get("factors_by_stream", {}).get(sid, {})

            if self._ww_engine is not None:
                try:
                    result = self._ww_engine.calculate(stream, factors)
                    stream_results.append(result)
                    total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
                    total_n2o += _safe_decimal(result.get("n2o_tonnes"))
                    total_ch4_co2e += _safe_decimal(
                        result.get("ch4_co2e")
                    )
                    total_n2o_co2e += _safe_decimal(
                        result.get("n2o_co2e")
                    )
                    total_ch4_recovered += _safe_decimal(
                        result.get("ch4_recovered_tonnes")
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        "WW engine failed for stream %s: %s, "
                        "falling back to defaults",
                        sid, str(e),
                    )

            # Fallback IPCC Ch 6 calculation
            result = self._calculate_ww_default(
                stream, factors, ch4_gwp, n2o_gwp
            )
            stream_results.append(result)
            total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
            total_n2o += _safe_decimal(result.get("n2o_tonnes"))
            total_ch4_co2e += _safe_decimal(result.get("ch4_co2e"))
            total_n2o_co2e += _safe_decimal(result.get("n2o_co2e"))
            total_ch4_recovered += _safe_decimal(
                result.get("ch4_recovered_tonnes")
            )

        ctx["wastewater_results"] = {
            "status": "COMPLETE",
            "stream_count": len(stream_results),
            "streams": stream_results,
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
            "total_ch4_co2e": str(_quantize(total_ch4_co2e)),
            "total_n2o_co2e": str(_quantize(total_n2o_co2e)),
            "total_co2e": str(_quantize(total_ch4_co2e + total_n2o_co2e)),
            "ch4_recovered_tonnes": str(_quantize(total_ch4_recovered)),
        }
        logger.debug(
            "Wastewater calc: %d streams, ch4=%s t, n2o=%s t",
            len(stream_results), total_ch4, total_n2o,
        )

    def _calculate_ww_default(
        self,
        stream: Dict[str, Any],
        factors: Dict[str, Any],
        ch4_gwp: Decimal,
        n2o_gwp: Decimal,
    ) -> Dict[str, Any]:
        """Calculate wastewater emissions using IPCC Ch 6 defaults.

        CH4 = (TOW - S) * Bo * MCF * 0.001 - R
        N2O_plant = P * T_protein * F_NPR * F_NON_CON * EF_plant * 44/28

        Args:
            stream: Normalized stream dictionary.
            factors: Emission factors for this stream.
            ch4_gwp: GWP for CH4.
            n2o_gwp: GWP for N2O.

        Returns:
            Dictionary of wastewater emission results.
        """
        # TOW (total organic waste in wastewater kg/yr)
        tow = stream.get("organic_load_kg_per_yr", _ZERO)
        sludge = stream.get("sludge_removed_kg", _ZERO)
        organic_type = stream.get("organic_load_type", "BOD")
        treatment_system = stream.get(
            "treatment_system", "AEROBIC_WELL_MANAGED"
        )

        # Bo (max CH4 producing capacity)
        bo = _safe_decimal(
            factors.get("bo", WASTEWATER_BO.get(organic_type, Decimal("0.6")))
        )

        # MCF by treatment system
        mcf = _safe_decimal(
            factors.get("mcf", WASTEWATER_MCF.get(
                treatment_system, _ZERO
            ))
        )

        # CH4 calculation (IPCC Eq 6.1)
        # CH4 = (TOW - S) * Bo * MCF * 0.001 - R
        ch4_before_recovery = _quantize(
            (tow - sludge) * bo * mcf / _THOUSAND
        )
        ch4_before_recovery = max(ch4_before_recovery, _ZERO)

        # Methane recovery
        collection_eff = stream.get("collection_efficiency", _ZERO)
        ch4_recovered = _quantize(ch4_before_recovery * collection_eff)
        ch4_emitted = _quantize(ch4_before_recovery - ch4_recovered)
        ch4_emitted = max(ch4_emitted, _ZERO)

        # N2O calculation (IPCC Eq 6.7 - simplified)
        pop_equiv = stream.get("population_equivalent", _ZERO)
        n2o_tonnes = _ZERO
        if pop_equiv > _ZERO:
            protein_kg_yr = stream.get(
                "protein_consumption_kg_yr", Decimal("25550")
            )
            f_npr = Decimal("0.16")
            f_non_con = Decimal("1.1")
            ef_plant = Decimal("0.016")  # kg N2O-N / kg N

            # N2O (tonnes) = P * protein * F_NPR * F_NON_CON * EF * 44/28
            #                 / 1000 (convert kg to tonnes)
            n2o_tonnes = _quantize(
                pop_equiv * (protein_kg_yr / pop_equiv if pop_equiv > _ZERO else _ZERO)
                * f_npr * f_non_con * ef_plant * _N2O_N_RATIO / _THOUSAND
            )
        else:
            # Simplified: assume N2O proportional to organic load
            n2o_ef = Decimal("0.005")  # kg N2O-N / kg N
            n_load = tow * Decimal("0.05")  # approximate N as 5% of organic load
            n2o_tonnes = _quantize(
                n_load * n2o_ef * _N2O_N_RATIO / _THOUSAND
            )

        n2o_tonnes = max(n2o_tonnes, _ZERO)

        ch4_co2e = _quantize(ch4_emitted * ch4_gwp)
        n2o_co2e = _quantize(n2o_tonnes * n2o_gwp)
        total_co2e = _quantize(ch4_co2e + n2o_co2e)

        return {
            "stream_id": stream["stream_id"],
            "treatment_method": stream["treatment_method"],
            "waste_category": stream["waste_category"],
            "organic_load_type": organic_type,
            "treatment_system": treatment_system,
            "calculation_method": "IPCC_CH6_DEFAULT",
            "tow_kg_yr": str(tow),
            "sludge_removed_kg": str(sludge),
            "bo": str(bo),
            "mcf": str(mcf),
            "ch4_before_recovery_tonnes": str(ch4_before_recovery),
            "ch4_recovered_tonnes": str(ch4_recovered),
            "ch4_tonnes": str(ch4_emitted),
            "n2o_tonnes": str(n2o_tonnes),
            "ch4_co2e": str(ch4_co2e),
            "n2o_co2e": str(n2o_co2e),
            "total_co2e": str(total_co2e),
            "fossil_co2_tonnes": "0",
            "biogenic_co2_tonnes": "0",
            "gwp_ch4": str(ch4_gwp),
            "gwp_n2o": str(n2o_gwp),
            "provenance_hash": _compute_hash({
                "stream_id": stream["stream_id"],
                "tow": str(tow),
                "ch4": str(ch4_emitted),
                "n2o": str(n2o_tonnes),
            }),
        }

    # ------------------------------------------------------------------
    # Stage 7: Check Compliance
    # ------------------------------------------------------------------

    def _stage_check_compliance(self, ctx: Dict[str, Any]) -> None:
        """Stage 7: Run compliance checks against selected frameworks.

        Supports GHG_PROTOCOL, IPCC_2006, CSRD_ESRS_E1,
        EU_WASTE_DIRECTIVE, EPA_GHGRP, UK_SECR, and ISO_14064.
        """
        if self._compliance_engine is None:
            ctx["compliance_result"] = {
                "status": "COMPLIANCE_ENGINE_UNAVAILABLE",
            }
            return

        frameworks = ctx.get("frameworks", [])
        if not frameworks:
            ctx["compliance_result"] = {
                "status": "SKIPPED",
                "reason": "No frameworks specified",
            }
            return

        # Build compliance data from all calculation results
        bio = ctx.get("biological_results", {})
        thermal = ctx.get("thermal_results", {})
        ww = ctx.get("wastewater_results", {})

        compliance_data: Dict[str, Any] = {
            "tenant_id": ctx["tenant_id"],
            "facility_id": ctx.get("facility_id", ""),
            "reporting_year": ctx.get("reporting_year"),
            "gwp_source": ctx["gwp_source"],
            "calculation_method": ctx["calculation_method"],
            "stream_count": ctx.get("stream_count", 0),
            "treatment_methods": sorted(set(
                s["treatment_method"]
                for s in ctx.get("treatment_streams", [])
            )),
            "waste_categories": ctx.get("waste_categories_present", []),
            "total_waste_mass_tonnes": str(
                ctx.get("total_waste_mass_tonnes", _ZERO)
            ),
            # Biological results
            "biological_status": bio.get("status", "SKIPPED"),
            "biological_ch4_tonnes": bio.get("total_ch4_tonnes", "0"),
            "biological_n2o_tonnes": bio.get("total_n2o_tonnes", "0"),
            "biological_co2e": bio.get("total_co2e", "0"),
            # Thermal results
            "thermal_status": thermal.get("status", "SKIPPED"),
            "thermal_fossil_co2_tonnes": thermal.get(
                "total_fossil_co2_tonnes", "0"
            ),
            "thermal_biogenic_co2_tonnes": thermal.get(
                "total_biogenic_co2_tonnes", "0"
            ),
            "thermal_ch4_tonnes": thermal.get("total_ch4_tonnes", "0"),
            "thermal_n2o_tonnes": thermal.get("total_n2o_tonnes", "0"),
            # Wastewater results
            "wastewater_status": ww.get("status", "SKIPPED"),
            "wastewater_ch4_tonnes": ww.get("total_ch4_tonnes", "0"),
            "wastewater_n2o_tonnes": ww.get("total_n2o_tonnes", "0"),
            # Separation flags
            "has_fossil_biogenic_split": thermal.get("status") == "COMPLETE",
            "has_methane_recovery": _safe_decimal(
                bio.get("ch4_captured_tonnes", "0")
            ) > _ZERO,
            "has_energy_recovery": _safe_decimal(
                thermal.get("energy_recovered_gj", "0")
            ) > _ZERO,
        }

        try:
            compliance_result = self._compliance_engine.check_compliance(
                compliance_data, frameworks
            )
            ctx["compliance_result"] = compliance_result
        except Exception as e:
            ctx["compliance_result"] = {
                "status": "ERROR",
                "error": str(e),
            }
            logger.error(
                "Compliance check failed: %s", str(e), exc_info=True
            )

        logger.debug(
            "Compliance: %s",
            ctx["compliance_result"].get(
                "overall", {}
            ).get("compliance_status", "UNKNOWN"),
        )

    # ------------------------------------------------------------------
    # Stage 8: Assemble Results
    # ------------------------------------------------------------------

    def _stage_assemble_results(self, ctx: Dict[str, Any]) -> None:
        """Stage 8: Assemble all results into final output.

        Combines biological, thermal, and wastewater results into
        unified totals.  Separates fossil CO2 from biogenic CO2,
        calculates per-gas CO2e, and aggregates methane recovery
        and energy recovery data.
        """
        bio = ctx.get("biological_results", {})
        thermal = ctx.get("thermal_results", {})
        ww = ctx.get("wastewater_results", {})

        # ---- Fossil CO2 (only from thermal) ----
        total_fossil_co2 = _safe_decimal(
            thermal.get("total_fossil_co2_tonnes", "0")
        )

        # ---- Biogenic CO2 (only from thermal) ----
        total_biogenic_co2 = _safe_decimal(
            thermal.get("total_biogenic_co2_tonnes", "0")
        )

        # ---- Total CH4 from all sources ----
        total_ch4 = (
            _safe_decimal(bio.get("total_ch4_tonnes", "0"))
            + _safe_decimal(thermal.get("total_ch4_tonnes", "0"))
            + _safe_decimal(ww.get("total_ch4_tonnes", "0"))
        )

        # ---- Total N2O from all sources ----
        total_n2o = (
            _safe_decimal(bio.get("total_n2o_tonnes", "0"))
            + _safe_decimal(thermal.get("total_n2o_tonnes", "0"))
            + _safe_decimal(ww.get("total_n2o_tonnes", "0"))
        )

        # ---- Per-gas CO2e ----
        gwp_source = ctx["gwp_source"]
        ch4_gwp = self._get_gwp("CH4", gwp_source)
        n2o_gwp = self._get_gwp("N2O", gwp_source)

        co2_fossil_co2e = _quantize(total_fossil_co2)
        ch4_co2e = _quantize(total_ch4 * ch4_gwp)
        n2o_co2e = _quantize(total_n2o * n2o_gwp)

        # Total emissions (fossil CO2 + CH4 CO2e + N2O CO2e)
        # NOTE: biogenic CO2 excluded from Scope 1 total per GHG Protocol
        total_emissions_tco2e = _quantize(
            co2_fossil_co2e + ch4_co2e + n2o_co2e
        )

        # Gross total (including biogenic CO2 for full disclosure)
        gross_total_tco2e = _quantize(
            total_emissions_tco2e + total_biogenic_co2
        )

        # ---- Methane recovery lifecycle ----
        ch4_generated = _safe_decimal(
            bio.get("ch4_generated_tonnes", "0")
        )
        ch4_captured = _safe_decimal(
            bio.get("ch4_captured_tonnes", "0")
        )
        ch4_flared = _safe_decimal(
            bio.get("ch4_flared_tonnes", "0")
        )
        ch4_utilized = _safe_decimal(
            bio.get("ch4_utilized_tonnes", "0")
        )
        # CH4 emitted is the net CH4 from all sources
        ch4_emitted = total_ch4

        # Also add WW recovered CH4
        ch4_recovered_ww = _safe_decimal(
            ww.get("ch4_recovered_tonnes", "0")
        )
        total_ch4_captured = ch4_captured + ch4_recovered_ww

        # ---- Energy recovery ----
        energy_recovered_gj = _safe_decimal(
            thermal.get("energy_recovered_gj", "0")
        )
        displaced_emissions = _safe_decimal(
            thermal.get("displaced_emissions_tco2e", "0")
        )

        # ---- Per-stream breakdown ----
        all_stream_results: List[Dict[str, Any]] = []
        for source_key, source_data in [
            ("biological", bio),
            ("thermal", thermal),
            ("wastewater", ww),
        ]:
            if source_data.get("status") == "COMPLETE":
                for sr in source_data.get("streams", []):
                    sr_copy = dict(sr)
                    sr_copy["source_engine"] = source_key
                    all_stream_results.append(sr_copy)

        # ---- Calculation steps for audit trail ----
        calculation_steps: List[Dict[str, str]] = []
        calculation_steps.append({
            "step": "1_FOSSIL_CO2",
            "description": "Fossil CO2 from thermal treatment",
            "value_tonnes": str(total_fossil_co2),
        })
        calculation_steps.append({
            "step": "2_BIOGENIC_CO2",
            "description": "Biogenic CO2 from thermal treatment (memo item)",
            "value_tonnes": str(total_biogenic_co2),
        })
        calculation_steps.append({
            "step": "3_CH4",
            "description": "CH4 from all treatment streams",
            "value_tonnes": str(total_ch4),
            "value_co2e": str(ch4_co2e),
        })
        calculation_steps.append({
            "step": "4_N2O",
            "description": "N2O from all treatment streams",
            "value_tonnes": str(total_n2o),
            "value_co2e": str(n2o_co2e),
        })
        calculation_steps.append({
            "step": "5_TOTAL",
            "description": (
                "Total Scope 1 emissions "
                "(fossil CO2 + CH4 CO2e + N2O CO2e)"
            ),
            "value_tco2e": str(total_emissions_tco2e),
        })
        if displaced_emissions > _ZERO:
            calculation_steps.append({
                "step": "6_DISPLACED",
                "description": "Displaced grid emissions from energy recovery",
                "value_tco2e": str(displaced_emissions),
            })

        ctx["assembled"] = {
            "total_emissions_tco2e": str(total_emissions_tco2e),
            "gross_total_tco2e": str(gross_total_tco2e),
            "total_fossil_co2_tonnes": str(_quantize(total_fossil_co2)),
            "total_biogenic_co2_tonnes": str(_quantize(total_biogenic_co2)),
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
            "co2_fossil_co2e": str(co2_fossil_co2e),
            "ch4_co2e": str(ch4_co2e),
            "n2o_co2e": str(n2o_co2e),
            "ch4_generated_tonnes": str(_quantize(ch4_generated)),
            "ch4_captured_tonnes": str(_quantize(total_ch4_captured)),
            "ch4_flared_tonnes": str(_quantize(ch4_flared)),
            "ch4_utilized_tonnes": str(_quantize(ch4_utilized)),
            "ch4_emitted_tonnes": str(_quantize(ch4_emitted)),
            "energy_recovered_gj": str(_quantize(energy_recovered_gj)),
            "displaced_emissions_tco2e": str(_quantize(displaced_emissions)),
            "stream_results": all_stream_results,
            "calculation_steps": calculation_steps,
        }

        ctx["assembly_status"] = "COMPLETE"
        logger.debug(
            "Assembly complete: total=%s tCO2e (fossil_co2=%s, "
            "ch4_co2e=%s, n2o_co2e=%s)",
            total_emissions_tco2e, co2_fossil_co2e,
            ch4_co2e, n2o_co2e,
        )

    # ------------------------------------------------------------------
    # Main Execute
    # ------------------------------------------------------------------

    def execute(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the full 8-stage pipeline for a single calculation.

        Runs all stages sequentially.  Stage 1 (VALIDATE_INPUT) failure
        is fatal and aborts remaining stages.  Stages 4-6 (calculation
        engines) can fail independently without aborting the pipeline;
        partial results are still assembled.

        Args:
            request: Calculation request dictionary containing
                tenant_id, treatment_streams, and optional parameters.

        Returns:
            Complete calculation result dictionary with all stage
            outputs, per-gas totals, methane recovery lifecycle,
            energy recovery credits, compliance results, and
            SHA-256 provenance hash.
        """
        pipeline_start = time.monotonic()
        calculation_id = str(uuid4())

        with self._lock:
            self._total_executions += 1

        # Initialize context
        ctx: Dict[str, Any] = {
            "calculation_id": calculation_id,
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
            (PipelineStage.CLASSIFY_TREATMENT, self._stage_classify_treatment),
            (PipelineStage.LOOKUP_FACTORS, self._stage_lookup_factors),
            (PipelineStage.CALCULATE_BIOLOGICAL, self._stage_calculate_biological),
            (PipelineStage.CALCULATE_THERMAL, self._stage_calculate_thermal),
            (PipelineStage.CALCULATE_WASTEWATER, self._stage_calculate_wastewater),
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
            "calculation_id": calculation_id,
            "tenant_id": ctx.get("tenant_id", ""),
            "facility_id": ctx.get("facility_id", ""),
            "status": (
                "SUCCESS" if is_success
                else "PARTIAL" if ctx["stages_completed"]
                else "FAILED"
            ),
            "stages_completed": ctx["stages_completed"],
            "stages_failed": ctx["stages_failed"],

            # Totals
            "total_emissions_tco2e": assembled.get(
                "total_emissions_tco2e", "0"
            ),
            "gross_total_tco2e": assembled.get("gross_total_tco2e", "0"),
            "total_fossil_co2_tonnes": assembled.get(
                "total_fossil_co2_tonnes", "0"
            ),
            "total_biogenic_co2_tonnes": assembled.get(
                "total_biogenic_co2_tonnes", "0"
            ),
            "total_ch4_tonnes": assembled.get("total_ch4_tonnes", "0"),
            "total_n2o_tonnes": assembled.get("total_n2o_tonnes", "0"),

            # Per-gas CO2e
            "co2_fossil_co2e": assembled.get("co2_fossil_co2e", "0"),
            "ch4_co2e": assembled.get("ch4_co2e", "0"),
            "n2o_co2e": assembled.get("n2o_co2e", "0"),

            # Methane recovery lifecycle
            "ch4_generated_tonnes": assembled.get(
                "ch4_generated_tonnes", "0"
            ),
            "ch4_captured_tonnes": assembled.get(
                "ch4_captured_tonnes", "0"
            ),
            "ch4_flared_tonnes": assembled.get("ch4_flared_tonnes", "0"),
            "ch4_utilized_tonnes": assembled.get(
                "ch4_utilized_tonnes", "0"
            ),
            "ch4_emitted_tonnes": assembled.get(
                "ch4_emitted_tonnes", "0"
            ),

            # Energy recovery
            "energy_recovered_gj": assembled.get(
                "energy_recovered_gj", "0"
            ),
            "displaced_emissions_tco2e": assembled.get(
                "displaced_emissions_tco2e", "0"
            ),

            # Per-stream breakdown
            "stream_results": assembled.get("stream_results", []),

            # Engine results (detailed)
            "results": {
                "biological": ctx.get("biological_results", {}),
                "thermal": ctx.get("thermal_results", {}),
                "wastewater": ctx.get("wastewater_results", {}),
            },

            # Compliance
            "compliance": ctx.get("compliance_result", {}),

            # Context
            "gwp_source": ctx.get("gwp_source", "AR6"),
            "calculation_method": ctx.get("calculation_method", ""),
            "reporting_year": ctx.get("reporting_year"),
            "waste_categories_present": ctx.get(
                "waste_categories_present", []
            ),
            "facility_types": ctx.get("facility_types", []),
            "total_waste_mass_tonnes": str(
                ctx.get("total_waste_mass_tonnes", _ZERO)
            ),
            "stream_count": ctx.get("stream_count", 0),

            # Audit trail
            "calculation_steps": assembled.get("calculation_steps", []),
            "errors": ctx["errors"],
            "stage_timings": ctx["stage_timings"],
            "provenance_chain": ctx["provenance_chain"],
            "processing_time_ms": pipeline_time,
            "calculated_at": _utcnow_iso(),
        }

        # Final provenance hash over the entire result
        result["provenance_hash"] = f"sha256:{_compute_hash(result)}"

        # Record provenance if tracker available
        if self._provenance_tracker is not None:
            try:
                self._provenance_tracker.record(
                    entity_type="CALCULATION",
                    entity_id=calculation_id,
                    action="CALCULATE",
                    data=result,
                    metadata={
                        "tenant_id": ctx.get("tenant_id", ""),
                        "facility_id": ctx.get("facility_id", ""),
                        "status": result["status"],
                        "total_tco2e": result["total_emissions_tco2e"],
                    },
                )
            except Exception as e:
                logger.warning(
                    "Provenance recording failed: %s", str(e)
                )

        logger.info(
            "Pipeline execute: id=%s, status=%s, "
            "stages=%d/%d, total_tco2e=%s, time=%.3fms",
            calculation_id, result["status"],
            len(ctx["stages_completed"]), len(stages),
            result["total_emissions_tco2e"], pipeline_time,
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

        Processes each request sequentially through the full 8-stage
        pipeline.  Failures in individual requests do not abort the
        batch.  Produces aggregate summaries by treatment method and
        waste category.

        Args:
            requests: List of calculation request dictionaries.

        Returns:
            Batch results with individual results, aggregate totals
            by treatment method and waste category, and batch-level
            provenance hash.
        """
        batch_start = time.monotonic()
        batch_id = str(uuid4())

        with self._lock:
            self._total_batches += 1

        results: List[Dict[str, Any]] = []
        total_emissions = _ZERO
        total_fossil_co2 = _ZERO
        total_biogenic_co2 = _ZERO
        total_ch4 = _ZERO
        total_n2o = _ZERO
        total_energy_recovered = _ZERO
        total_displaced = _ZERO
        success_count = 0
        failure_count = 0

        for i, request in enumerate(requests):
            try:
                result = self.execute(request)
                results.append(result)

                if result["status"] in ("SUCCESS", "PARTIAL"):
                    success_count += 1
                    total_emissions += _safe_decimal(
                        result.get("total_emissions_tco2e", "0")
                    )
                    total_fossil_co2 += _safe_decimal(
                        result.get("total_fossil_co2_tonnes", "0")
                    )
                    total_biogenic_co2 += _safe_decimal(
                        result.get("total_biogenic_co2_tonnes", "0")
                    )
                    total_ch4 += _safe_decimal(
                        result.get("total_ch4_tonnes", "0")
                    )
                    total_n2o += _safe_decimal(
                        result.get("total_n2o_tonnes", "0")
                    )
                    total_energy_recovered += _safe_decimal(
                        result.get("energy_recovered_gj", "0")
                    )
                    total_displaced += _safe_decimal(
                        result.get("displaced_emissions_tco2e", "0")
                    )
                else:
                    failure_count += 1
            except Exception as e:
                failure_count += 1
                results.append({
                    "calculation_id": str(uuid4()),
                    "status": "FAILED",
                    "errors": [f"Batch item {i} failed: {str(e)}"],
                    "request_index": i,
                })
                logger.error(
                    "Batch item %d failed: %s", i, str(e), exc_info=True
                )

        batch_time = round((time.monotonic() - batch_start) * 1000, 3)

        # Aggregate by treatment method
        by_method: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        by_category: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        for r in results:
            if r.get("status") in ("SUCCESS", "PARTIAL"):
                for sr in r.get("stream_results", []):
                    method = sr.get("treatment_method", "UNKNOWN")
                    category = sr.get("waste_category", "UNKNOWN")
                    sr_co2e = _safe_decimal(sr.get("total_co2e", "0"))
                    by_method[method] += sr_co2e
                    by_category[category] += sr_co2e

        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "status": (
                "SUCCESS" if failure_count == 0
                else "PARTIAL" if success_count > 0
                else "FAILED"
            ),
            "total_requests": len(requests),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_emissions_tco2e": str(_quantize(total_emissions)),
            "total_fossil_co2_tonnes": str(_quantize(total_fossil_co2)),
            "total_biogenic_co2_tonnes": str(
                _quantize(total_biogenic_co2)
            ),
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
            "energy_recovered_gj": str(_quantize(total_energy_recovered)),
            "displaced_emissions_tco2e": str(_quantize(total_displaced)),
            "by_treatment_method": {
                k: str(_quantize(v))
                for k, v in sorted(by_method.items())
            },
            "by_waste_category": {
                k: str(_quantize(v))
                for k, v in sorted(by_category.items())
            },
            "results": results,
            "processing_time_ms": batch_time,
            "calculated_at": _utcnow_iso(),
        }
        batch_result["provenance_hash"] = f"sha256:{_compute_hash(batch_result)}"

        # Record batch provenance
        if self._provenance_tracker is not None:
            try:
                self._provenance_tracker.record(
                    entity_type="BATCH",
                    entity_id=batch_id,
                    action="CALCULATE",
                    data={
                        "total_requests": len(requests),
                        "success_count": success_count,
                        "failure_count": failure_count,
                        "total_tco2e": str(_quantize(total_emissions)),
                    },
                    metadata={"batch_id": batch_id},
                )
            except Exception as e:
                logger.warning(
                    "Batch provenance recording failed: %s", str(e)
                )

        logger.info(
            "Batch execute: id=%s, total=%d, success=%d, "
            "failed=%d, tco2e=%s, time=%.3fms",
            batch_id, len(requests), success_count,
            failure_count, total_emissions, batch_time,
        )
        return batch_result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return pipeline engine statistics.

        Returns:
            Dictionary containing engine version, uptime, execution
            counts, average stage timings, and engine availability.
        """
        with self._lock:
            avg_timings: Dict[str, Optional[float]] = {}
            for stage, times in self._stage_timings.items():
                if times:
                    avg_timings[stage] = round(
                        sum(times) / len(times), 3
                    )
                else:
                    avg_timings[stage] = None

            return {
                "engine": "WasteTreatmentPipelineEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_executions": self._total_executions,
                "total_batches": self._total_batches,
                "stages": [s.value for s in PipelineStage],
                "stage_count": len(PipelineStage),
                "avg_stage_timings_ms": avg_timings,
                "engines": {
                    "db": self._db_engine is not None,
                    "bio": self._bio_engine is not None,
                    "thermal": self._thermal_engine is not None,
                    "ww": self._ww_engine is not None,
                    "compliance": self._compliance_engine is not None,
                    "provenance": self._provenance_tracker is not None,
                },
            }

    def get_engine_health(self) -> Dict[str, Any]:
        """Return health status of the pipeline and upstream engines.

        Returns:
            Dictionary with overall health, per-engine status, and
            stage availability assessment.
        """
        engines_available = {
            "db": self._db_engine is not None,
            "bio": self._bio_engine is not None,
            "thermal": self._thermal_engine is not None,
            "ww": self._ww_engine is not None,
            "compliance": self._compliance_engine is not None,
            "provenance": self._provenance_tracker is not None,
        }

        # Stages that require specific engines
        stage_availability = {
            PipelineStage.VALIDATE_INPUT.value: True,
            PipelineStage.CLASSIFY_TREATMENT.value: True,
            PipelineStage.LOOKUP_FACTORS.value: True,  # has fallback
            PipelineStage.CALCULATE_BIOLOGICAL.value: True,  # has fallback
            PipelineStage.CALCULATE_THERMAL.value: True,  # has fallback
            PipelineStage.CALCULATE_WASTEWATER.value: True,  # has fallback
            PipelineStage.CHECK_COMPLIANCE.value: engines_available["compliance"],
            PipelineStage.ASSEMBLE_RESULTS.value: True,
        }

        all_critical_available = all(stage_availability.values())

        return {
            "healthy": True,  # Pipeline always works with fallbacks
            "all_engines_available": all(engines_available.values()),
            "all_stages_available": all_critical_available,
            "engines": engines_available,
            "stage_availability": stage_availability,
            "total_executions": self._total_executions,
            "total_batches": self._total_batches,
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
        for engine in [
            self._db_engine,
            self._bio_engine,
            self._thermal_engine,
            self._ww_engine,
            self._compliance_engine,
        ]:
            if engine is not None and hasattr(engine, "reset"):
                engine.reset()

        # Clear provenance
        if self._provenance_tracker is not None:
            if hasattr(self._provenance_tracker, "clear"):
                self._provenance_tracker.clear()
            elif hasattr(self._provenance_tracker, "clear_trail"):
                self._provenance_tracker.clear_trail()

        logger.info(
            "WasteTreatmentPipelineEngine and all upstream engines reset"
        )
