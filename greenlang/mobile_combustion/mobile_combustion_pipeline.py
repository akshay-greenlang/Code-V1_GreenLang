# -*- coding: utf-8 -*-
"""
Mobile Combustion Pipeline Engine (Engine 7) - AGENT-MRV-003

End-to-end orchestration pipeline for GHG Protocol Scope 1 mobile
combustion emissions calculations. Coordinates all six upstream engines
(vehicle database, emission calculator, fleet manager, distance estimator,
uncertainty quantifier, compliance checker) through a deterministic,
eight-stage pipeline:

    1. VALIDATE              - Input validation and normalization
    2. RESOLVE_VEHICLE       - Look up vehicle type and emission factors
    3. ESTIMATE_FUEL_OR_DISTANCE - Convert between fuel/distance as needed
    4. CALCULATE_EMISSIONS   - Apply emission calculation method
                               (fuel/distance/spend)
    5. APPLY_BIOFUEL_ADJUSTMENT - Separate biogenic/fossil CO2
    6. QUANTIFY_UNCERTAINTY  - Run Monte Carlo or analytical uncertainty
    7. CHECK_COMPLIANCE      - Validate against regulatory frameworks
    8. GENERATE_AUDIT        - Create provenance chain and audit trail

Each stage is checkpointed so that failures produce partial results with
complete provenance. Thread-safe execution with statistics counters.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Python arithmetic
    - No LLM calls in the calculation path
    - SHA-256 provenance hash at every pipeline stage
    - Full audit trail for regulatory traceability

Thread Safety:
    All mutable state is protected by a ``threading.Lock``. Concurrent
    ``run_pipeline`` invocations from different threads are safe.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional config import
# ---------------------------------------------------------------------------

try:
    from greenlang.mobile_combustion.config import (
        MobileCombustionConfig,
        get_config,
    )
except ImportError:
    MobileCombustionConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """No-op fallback when config module is unavailable."""
        return None


# ---------------------------------------------------------------------------
# Optional model imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.mobile_combustion.models import (
        CalculationInput,
        CalculationMethod,
        CalculationResult,
        CalculationTier,
        ComplianceCheckResult,
        ComplianceStatus,
        DistanceUnit,
        EmissionControlTechnology,
        EmissionFactorSource,
        EmissionGas,
        FleetAggregation,
        FuelType,
        GWPSource,
        MobileCombustionInput,
        MobileCombustionOutput,
        ReportingPeriod,
        UncertaintyResult,
        VehicleCategory,
        VehicleType,
    )
except ImportError:
    CalculationInput = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    CalculationTier = None  # type: ignore[assignment, misc]
    ComplianceCheckResult = None  # type: ignore[assignment, misc]
    ComplianceStatus = None  # type: ignore[assignment, misc]
    DistanceUnit = None  # type: ignore[assignment, misc]
    EmissionControlTechnology = None  # type: ignore[assignment, misc]
    EmissionFactorSource = None  # type: ignore[assignment, misc]
    EmissionGas = None  # type: ignore[assignment, misc]
    FleetAggregation = None  # type: ignore[assignment, misc]
    FuelType = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    MobileCombustionInput = None  # type: ignore[assignment, misc]
    MobileCombustionOutput = None  # type: ignore[assignment, misc]
    ReportingPeriod = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]
    VehicleCategory = None  # type: ignore[assignment, misc]
    VehicleType = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine
except ImportError:
    VehicleDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.emission_calculator import EmissionCalculatorEngine
except ImportError:
    EmissionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.fleet_manager import FleetManagerEngine
except ImportError:
    FleetManagerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.distance_estimator import DistanceEstimatorEngine
except ImportError:
    DistanceEstimatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.uncertainty_quantifier import UncertaintyQuantifierEngine
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.metrics import (
        PROMETHEUS_AVAILABLE,
        observe_calculation_duration,
        record_calculation,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def observe_calculation_duration(operation: str, seconds: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def record_calculation(method: str, vehicle_type: str, status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""


# ---------------------------------------------------------------------------
# Pipeline stage enum and constants
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    """Enumeration of the 8 mobile combustion pipeline stages.

    Each stage represents a discrete processing step with its own
    provenance hash, error handling, and timing instrumentation.
    """

    VALIDATE = "VALIDATE"
    RESOLVE_VEHICLE = "RESOLVE_VEHICLE"
    ESTIMATE_FUEL_OR_DISTANCE = "ESTIMATE_FUEL_OR_DISTANCE"
    CALCULATE_EMISSIONS = "CALCULATE_EMISSIONS"
    APPLY_BIOFUEL_ADJUSTMENT = "APPLY_BIOFUEL_ADJUSTMENT"
    QUANTIFY_UNCERTAINTY = "QUANTIFY_UNCERTAINTY"
    CHECK_COMPLIANCE = "CHECK_COMPLIANCE"
    GENERATE_AUDIT = "GENERATE_AUDIT"


PIPELINE_STAGES: List[str] = [stage.value for stage in PipelineStage]

# Stages whose failure does NOT abort the pipeline
NON_CRITICAL_STAGES: frozenset = frozenset({
    PipelineStage.QUANTIFY_UNCERTAINTY.value,
    PipelineStage.CHECK_COMPLIANCE.value,
    PipelineStage.GENERATE_AUDIT.value,
})

# Supported calculation methods
SUPPORTED_METHODS: List[str] = [
    "FUEL_BASED",
    "DISTANCE_BASED",
    "SPEND_BASED",
]

# Default GWP values (IPCC AR6 100-year)
DEFAULT_GWP_AR6: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 27.9,
    "N2O": 273.0,
}

DEFAULT_GWP_AR5: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 28.0,
    "N2O": 265.0,
}

DEFAULT_GWP_AR4: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 25.0,
    "N2O": 298.0,
}

GWP_TABLES: Dict[str, Dict[str, float]] = {
    "AR4": DEFAULT_GWP_AR4,
    "AR5": DEFAULT_GWP_AR5,
    "AR6": DEFAULT_GWP_AR6,
}

# EPA default emission factors for common vehicle fuels (kg per unit fuel)
# Source: EPA GHG Emission Factors Hub (2024)
# CO2 factors in kg/gallon, CH4 and N2O in g/mile (converted later)
FUEL_CO2_FACTORS_KG_PER_GALLON: Dict[str, float] = {
    "GASOLINE": 8.887,
    "DIESEL": 10.180,
    "LPG": 5.684,
    "CNG": 0.05444,      # kg/scf
    "LNG": 4.459,        # kg/gallon
    "ETHANOL": 5.746,    # kg/gallon (biogenic, tracked separately)
    "BIODIESEL": 9.460,  # kg/gallon (biogenic, tracked separately)
    "JET_FUEL": 9.750,
    "AVIATION_GASOLINE": 8.310,
    "MARINE_DIESEL": 10.210,
    "MARINE_RESIDUAL": 11.340,
    "E10": 8.573,        # 90% gasoline + 10% ethanol blend
    "E85": 6.216,        # 15% gasoline + 85% ethanol blend
    "B5": 10.144,        # 95% diesel + 5% biodiesel blend
    "B20": 10.036,       # 80% diesel + 20% biodiesel blend
    "B100": 9.460,       # 100% biodiesel
}

# Biofuel blend fossil fraction (for separating biogenic CO2)
BIOFUEL_FOSSIL_FRACTION: Dict[str, float] = {
    "GASOLINE": 1.0,
    "DIESEL": 1.0,
    "LPG": 1.0,
    "CNG": 1.0,
    "LNG": 1.0,
    "ETHANOL": 0.0,
    "BIODIESEL": 0.0,
    "JET_FUEL": 1.0,
    "AVIATION_GASOLINE": 1.0,
    "MARINE_DIESEL": 1.0,
    "MARINE_RESIDUAL": 1.0,
    "E10": 0.933,
    "E85": 0.215,
    "B5": 0.964,
    "B20": 0.858,
    "B100": 0.0,
    "SAF": 0.50,
}

# CH4 and N2O emission factors by vehicle type in g/mile
# Source: EPA Emission Factors for GHG Inventories (Table 2, 3, 4)
CH4_G_PER_MILE: Dict[str, Dict[str, float]] = {
    "PASSENGER_CAR_GASOLINE": {"CH4": 0.0113, "N2O": 0.0045},
    "PASSENGER_CAR_DIESEL": {"CH4": 0.0004, "N2O": 0.0010},
    "LIGHT_TRUCK_GASOLINE": {"CH4": 0.0143, "N2O": 0.0055},
    "LIGHT_TRUCK_DIESEL": {"CH4": 0.0005, "N2O": 0.0012},
    "HEAVY_TRUCK_DIESEL": {"CH4": 0.0051, "N2O": 0.0048},
    "BUS_DIESEL": {"CH4": 0.0171, "N2O": 0.0085},
    "BUS_CNG": {"CH4": 0.7370, "N2O": 0.0410},
    "MOTORCYCLE": {"CH4": 0.0373, "N2O": 0.0024},
    "AIRCRAFT": {"CH4": 0.0000, "N2O": 0.0000},
    "MARINE_VESSEL": {"CH4": 0.0010, "N2O": 0.0020},
    "LOCOMOTIVE": {"CH4": 0.0920, "N2O": 0.0260},
    "OFF_ROAD_VEHICLE": {"CH4": 0.0050, "N2O": 0.0040},
}

# Default fuel economy by vehicle type (L/100km)
DEFAULT_FUEL_ECONOMY_L_PER_100KM: Dict[str, float] = {
    "PASSENGER_CAR_GASOLINE": 8.9,
    "PASSENGER_CAR_DIESEL": 6.5,
    "LIGHT_TRUCK_GASOLINE": 11.8,
    "LIGHT_TRUCK_DIESEL": 9.8,
    "HEAVY_TRUCK_DIESEL": 35.0,
    "BUS_DIESEL": 30.0,
    "BUS_CNG": 45.0,
    "MOTORCYCLE": 4.5,
    "AIRCRAFT": 300.0,
    "MARINE_VESSEL": 500.0,
    "LOCOMOTIVE": 200.0,
    "OFF_ROAD_VEHICLE": 25.0,
}

# Distance unit conversion to km
DISTANCE_TO_KM: Dict[str, float] = {
    "KM": 1.0,
    "MILES": 1.60934,
    "NAUTICAL_MILES": 1.852,
    "METERS": 0.001,
}

# Volume conversion to gallons
VOLUME_TO_GALLONS: Dict[str, float] = {
    "GALLONS": 1.0,
    "LITERS": 0.264172,
    "LITRES": 0.264172,
    "BARRELS": 42.0,
    "CUBIC_FEET": 7.48052,
    "CUBIC_METERS": 264.172,
    "MCF": 1000.0 * 7.48052,
}

# Spend-based emission factors (kg CO2e per USD)
SPEND_BASED_FACTORS_KG_CO2E_PER_USD: Dict[str, float] = {
    "GASOLINE": 2.39,
    "DIESEL": 2.69,
    "JET_FUEL": 2.54,
    "MARINE_DIESEL": 2.72,
    "CNG": 1.75,
    "LNG": 2.10,
    "LPG": 1.65,
    "AVIATION_GASOLINE": 2.45,
    "DEFAULT": 2.50,
}

# Compliance framework requirements
COMPLIANCE_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    "GHG_PROTOCOL": [
        {
            "requirement_id": "GHG-MC-001",
            "requirement_name": "Scope 1 Mobile Combustion",
            "description": "Complete Scope 1 reporting for mobile sources",
        },
        {
            "requirement_id": "GHG-MC-002",
            "requirement_name": "Emission Factor Documentation",
            "description": "Document all emission factor sources and references",
        },
        {
            "requirement_id": "GHG-MC-003",
            "requirement_name": "Biogenic CO2 Separation",
            "description": "Separately report biogenic CO2 from biofuel blends",
        },
        {
            "requirement_id": "GHG-MC-004",
            "requirement_name": "Uncertainty Assessment",
            "description": "Quantify uncertainty in mobile source calculations",
        },
    ],
    "EPA_40CFR98": [
        {
            "requirement_id": "EPA-MC-001",
            "requirement_name": "Subpart C Mobile Combustion",
            "description": "EPA GHGRP Subpart C mobile source compliance",
        },
        {
            "requirement_id": "EPA-MC-002",
            "requirement_name": "Fuel Consumption Records",
            "description": "Maintain fuel purchase and consumption records",
        },
    ],
    "ISO_14064": [
        {
            "requirement_id": "ISO-MC-001",
            "requirement_name": "GHG Inventory - Transport",
            "description": "ISO 14064-1 Clause 5 mobile source quantification",
        },
        {
            "requirement_id": "ISO-MC-002",
            "requirement_name": "Data Quality Requirements",
            "description": "Data quality assessment for transport emissions",
        },
    ],
    "CSRD_ESRS_E1": [
        {
            "requirement_id": "ESRS-MC-001",
            "requirement_name": "Climate Change - Transport",
            "description": "ESRS E1 Scope 1 mobile emissions disclosure",
        },
    ],
    "EU_ETS": [
        {
            "requirement_id": "EU-ETS-MC-001",
            "requirement_name": "Aviation MRV",
            "description": "EU ETS aviation MRV compliance (if applicable)",
        },
    ],
    "UK_SECR": [
        {
            "requirement_id": "UK-SECR-MC-001",
            "requirement_name": "Transport Energy Use",
            "description": "UK SECR transport energy and emissions",
        },
    ],
}

# Uncertainty parameters by data quality tier
UNCERTAINTY_PARAMETERS: Dict[str, Dict[str, float]] = {
    "TIER_1": {
        "co2_relative_uncertainty": 0.05,
        "ch4_relative_uncertainty": 0.50,
        "n2o_relative_uncertainty": 0.50,
        "distance_relative_uncertainty": 0.10,
        "fuel_economy_relative_uncertainty": 0.15,
    },
    "TIER_2": {
        "co2_relative_uncertainty": 0.03,
        "ch4_relative_uncertainty": 0.30,
        "n2o_relative_uncertainty": 0.30,
        "distance_relative_uncertainty": 0.05,
        "fuel_economy_relative_uncertainty": 0.10,
    },
    "TIER_3": {
        "co2_relative_uncertainty": 0.01,
        "ch4_relative_uncertainty": 0.15,
        "n2o_relative_uncertainty": 0.15,
        "distance_relative_uncertainty": 0.02,
        "fuel_economy_relative_uncertainty": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _chain_hashes(prev_hash: str, current_data: Any) -> str:
    """Chain a previous provenance hash with new data.

    Args:
        prev_hash: Previous stage provenance hash.
        current_data: Current stage data to hash.

    Returns:
        New SHA-256 hex digest that chains previous and current.
    """
    current_hash = _compute_hash(current_data)
    combined = f"{prev_hash}:{current_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()


# ---------------------------------------------------------------------------
# StageResult dataclass
# ---------------------------------------------------------------------------


class StageResult:
    """Container for individual pipeline stage execution results.

    Attributes:
        stage_name: Pipeline stage identifier.
        status: Execution status (SUCCESS, FAILED, SKIPPED).
        duration_ms: Stage execution time in milliseconds.
        data: Stage-specific result data dictionary.
        error: Error message if stage failed (empty string otherwise).
        provenance_hash: SHA-256 hash for this stage.
    """

    __slots__ = (
        "stage_name", "status", "duration_ms", "data",
        "error", "provenance_hash",
    )

    def __init__(
        self,
        stage_name: str,
        status: str = "SUCCESS",
        duration_ms: float = 0.0,
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
    ) -> None:
        """Initialize a StageResult.

        Args:
            stage_name: Pipeline stage identifier.
            status: Execution status.
            duration_ms: Execution time in milliseconds.
            data: Stage-specific result data.
            error: Error message (empty on success).
            provenance_hash: SHA-256 provenance hash.
        """
        self.stage_name = stage_name
        self.status = status
        self.duration_ms = duration_ms
        self.data = data if data is not None else {}
        self.error = error
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all stage result fields.
        """
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------


class PipelineContext:
    """Mutable context carried between pipeline stages.

    Accumulates data as each stage completes, providing downstream stages
    with the outputs of upstream stages.

    Attributes:
        pipeline_id: Unique pipeline run identifier.
        calculation_id: Unique calculation identifier.
        input_data: Normalised input dictionary.
        vehicle_props: Resolved vehicle properties.
        emission_factors: Resolved emission factors.
        fuel_quantity_gallons: Fuel quantity in US gallons.
        distance_km: Distance in kilometres.
        calculation_method: Applied calculation method.
        calculation_result: Core emission calculation result.
        biofuel_adjustment: Biogenic/fossil CO2 split.
        uncertainty_result: Uncertainty quantification result.
        compliance_results: Compliance check results.
        audit_entries: Accumulated audit trail entries.
        provenance_chain: Ordered provenance hashes.
        stage_results: List of StageResult objects.
        errors: Accumulated errors.
        warnings: Accumulated warnings.
    """

    __slots__ = (
        "pipeline_id", "calculation_id", "input_data",
        "vehicle_props", "emission_factors",
        "fuel_quantity_gallons", "distance_km",
        "calculation_method", "calculation_result",
        "biofuel_adjustment", "uncertainty_result",
        "compliance_results", "audit_entries",
        "provenance_chain", "stage_results",
        "errors", "warnings",
    )

    def __init__(
        self,
        pipeline_id: str,
        calculation_id: str,
        input_data: Dict[str, Any],
    ) -> None:
        """Initialize a PipelineContext.

        Args:
            pipeline_id: Unique pipeline run identifier.
            calculation_id: Unique calculation identifier.
            input_data: Normalised input data dictionary.
        """
        self.pipeline_id = pipeline_id
        self.calculation_id = calculation_id
        self.input_data = input_data
        self.vehicle_props: Optional[Dict[str, Any]] = None
        self.emission_factors: Optional[Dict[str, Any]] = None
        self.fuel_quantity_gallons: Optional[float] = None
        self.distance_km: Optional[float] = None
        self.calculation_method: str = input_data.get(
            "calculation_method", "FUEL_BASED",
        )
        self.calculation_result: Optional[Dict[str, Any]] = None
        self.biofuel_adjustment: Optional[Dict[str, Any]] = None
        self.uncertainty_result: Optional[Dict[str, Any]] = None
        self.compliance_results: Optional[List[Dict[str, Any]]] = None
        self.audit_entries: List[Dict[str, Any]] = []
        self.provenance_chain: List[str] = []
        self.stage_results: List[StageResult] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []


# ===================================================================
# MobileCombustionPipelineEngine
# ===================================================================


class MobileCombustionPipelineEngine:
    """End-to-end orchestration pipeline for mobile combustion emissions.

    Coordinates all six upstream engines through an eight-stage pipeline
    with checkpointing, provenance tracking, and comprehensive error
    handling. Each pipeline run produces a deterministic SHA-256
    provenance hash for the complete execution chain.

    Thread-safe: all mutable state is protected by an internal lock.

    Attributes:
        config: MobileCombustionConfig instance or None.
        vehicle_db: Optional VehicleDatabaseEngine.
        calculator: Optional EmissionCalculatorEngine.
        fleet_manager: Optional FleetManagerEngine.
        distance_estimator: Optional DistanceEstimatorEngine.
        uncertainty: Optional UncertaintyQuantifierEngine.
        compliance: Optional ComplianceCheckerEngine.

    Example:
        >>> engine = MobileCombustionPipelineEngine()
        >>> result = engine.run_pipeline(input_data)
        >>> assert result["success"] is True
    """

    def __init__(
        self,
        config: Any = None,
        vehicle_db: Any = None,
        calculator: Any = None,
        fleet_manager: Any = None,
        distance_estimator: Any = None,
        uncertainty: Any = None,
        compliance: Any = None,
    ) -> None:
        """Initialize the MobileCombustionPipelineEngine.

        Wires all six upstream engines via dependency injection. Any
        engine set to ``None`` causes its pipeline stage to use built-in
        fallback logic rather than a hard failure.

        Args:
            config: MobileCombustionConfig instance or None.
            vehicle_db: VehicleDatabaseEngine instance or None.
            calculator: EmissionCalculatorEngine instance or None.
            fleet_manager: FleetManagerEngine instance or None.
            distance_estimator: DistanceEstimatorEngine instance or None.
            uncertainty: UncertaintyQuantifierEngine instance or None.
            compliance: ComplianceCheckerEngine instance or None.
        """
        self.config = config if config is not None else get_config()

        # Engine references
        self.vehicle_db = vehicle_db
        self.calculator = calculator
        self.fleet_manager = fleet_manager
        self.distance_estimator = distance_estimator
        self.uncertainty = uncertainty
        self.compliance = compliance

        # Thread-safe mutable state
        self._lock = threading.Lock()
        self._total_runs: int = 0
        self._successful_runs: int = 0
        self._failed_runs: int = 0
        self._total_duration_ms: float = 0.0
        self._last_run_at: Optional[str] = None
        self._pipeline_results: Dict[str, Dict[str, Any]] = {}
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

        # Stage timing aggregates
        self._stage_timings: Dict[str, List[float]] = {
            stage: [] for stage in PIPELINE_STAGES
        }

        logger.info(
            "MobileCombustionPipelineEngine initialized: "
            "vehicle_db=%s, calculator=%s, fleet_manager=%s, "
            "distance_estimator=%s, uncertainty=%s, compliance=%s",
            vehicle_db is not None,
            calculator is not None,
            fleet_manager is not None,
            distance_estimator is not None,
            uncertainty is not None,
            compliance is not None,
        )

    # ------------------------------------------------------------------
    # Public: Full pipeline execution - single calculation
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        input_data: Any,
    ) -> Dict[str, Any]:
        """Run the full eight-stage pipeline for a single calculation input.

        Each stage is checkpointed. If a critical stage fails, the
        pipeline aborts and returns partial results with provenance.
        Non-critical stages (uncertainty, compliance, audit) are
        skipped on failure without aborting the pipeline.

        Args:
            input_data: MobileCombustionInput (or dict) with vehicle_type,
                fuel_type, calculation_method, and fuel/distance/spend data.

        Returns:
            Dictionary with keys:
                - ``success``: bool indicating overall pipeline success.
                - ``pipeline_id``: Unique pipeline run identifier.
                - ``calculation_id``: Calculation identifier.
                - ``stage_results``: Per-stage execution details.
                - ``result``: Final calculation result dict.
                - ``pipeline_provenance_hash``: SHA-256 of entire pipeline.
                - ``total_duration_ms``: Wall-clock time in milliseconds.
                - ``stages_completed``: Count of successfully run stages.
                - ``stages_total``: Total number of pipeline stages (8).
        """
        pipeline_id = _new_uuid()
        t0 = time.perf_counter()

        # Normalise input to dict
        input_dict = self._normalise_input(input_data)
        calculation_id = input_dict.get("calculation_id", _new_uuid())
        input_dict["calculation_id"] = calculation_id

        # Create pipeline context
        ctx = PipelineContext(
            pipeline_id=pipeline_id,
            calculation_id=calculation_id,
            input_data=input_dict,
        )

        logger.info(
            "Pipeline %s started: calc_id=%s vehicle=%s method=%s",
            pipeline_id,
            calculation_id,
            input_dict.get("vehicle_type", "unknown"),
            input_dict.get("calculation_method", "FUEL_BASED"),
        )

        stages_completed = 0
        overall_success = True
        aborted = False

        # Execute each stage in order
        for stage_name in PIPELINE_STAGES:
            if aborted:
                stage_result = StageResult(
                    stage_name=stage_name,
                    status="SKIPPED",
                    error="Pipeline aborted due to prior critical failure",
                )
                ctx.stage_results.append(stage_result)
                continue

            stage_result = self._execute_stage(stage_name, ctx)
            ctx.stage_results.append(stage_result)

            if stage_result.status == "SUCCESS":
                stages_completed += 1
                self._record_stage_timing(
                    stage_name, stage_result.duration_ms,
                )
            elif stage_name in NON_CRITICAL_STAGES:
                logger.warning(
                    "Pipeline %s: non-critical stage %s failed: %s",
                    pipeline_id, stage_name, stage_result.error,
                )
            else:
                overall_success = False
                aborted = True
                logger.error(
                    "Pipeline %s: critical stage %s failed: %s",
                    pipeline_id, stage_name, stage_result.error,
                )

        # Compute pipeline-level provenance hash
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pipeline_provenance_hash = _compute_hash({
            "pipeline_id": pipeline_id,
            "calculation_id": calculation_id,
            "vehicle_type": input_dict.get("vehicle_type", ""),
            "stages_completed": stages_completed,
            "provenance_chain": ctx.provenance_chain,
            "total_duration_ms": elapsed_ms,
        })

        # Build final result from context
        final_result = self._build_final_result(ctx, elapsed_ms)

        # Update statistics
        self._update_stats(
            pipeline_id, overall_success, elapsed_ms,
            stages_completed, calculation_id,
        )

        # Record Prometheus metrics
        vehicle_type = input_dict.get("vehicle_type", "unknown")
        method = input_dict.get("calculation_method", "FUEL_BASED")
        record_calculation(
            method, vehicle_type,
            "success" if overall_success else "failure",
        )
        observe_calculation_duration(
            "single_pipeline", elapsed_ms / 1000.0,
        )

        result = {
            "success": overall_success,
            "pipeline_id": pipeline_id,
            "calculation_id": calculation_id,
            "stage_results": [
                sr.to_dict() for sr in ctx.stage_results
            ],
            "result": final_result,
            "pipeline_provenance_hash": pipeline_provenance_hash,
            "total_duration_ms": round(elapsed_ms, 3),
            "stages_completed": stages_completed,
            "stages_total": len(PIPELINE_STAGES),
            "errors": ctx.errors,
            "warnings": ctx.warnings,
            "timestamp": _utcnow_iso(),
        }

        logger.info(
            "Pipeline %s completed: success=%s stages=%d/%d "
            "duration=%.1fms",
            pipeline_id,
            overall_success,
            stages_completed,
            len(PIPELINE_STAGES),
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public: Batch pipeline execution
    # ------------------------------------------------------------------

    def run_batch_pipeline(
        self,
        inputs: List[Any],
        checkpoint_interval: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run the full pipeline for a batch of calculation inputs.

        Supports checkpointing at configurable intervals so that
        partial batch results are recoverable on failure.

        Args:
            inputs: List of MobileCombustionInput objects or dicts.
            checkpoint_interval: Save checkpoint every N calculations.

        Returns:
            List of individual pipeline result dictionaries.
        """
        batch_id = _new_uuid()
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        success_count = 0
        failure_count = 0
        total_co2e_kg = 0.0
        total_co2e_tonnes = 0.0

        logger.info(
            "Batch %s started: %d inputs, checkpoint_interval=%d",
            batch_id, len(inputs), checkpoint_interval,
        )

        for idx, inp in enumerate(inputs):
            try:
                pipeline_result = self.run_pipeline(inp)
                results.append(pipeline_result)

                if pipeline_result.get("success", False):
                    success_count += 1
                    result_data = pipeline_result.get("result", {})
                    total_co2e_kg += result_data.get(
                        "total_co2e_kg", 0.0,
                    )
                    total_co2e_tonnes += result_data.get(
                        "total_co2e_tonnes", 0.0,
                    )
                else:
                    failure_count += 1

                # Checkpoint at intervals
                if (idx + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(
                        batch_id, idx + 1, results,
                    )

            except Exception as exc:
                failure_count += 1
                logger.error(
                    "Batch %s: input %d failed: %s",
                    batch_id, idx, exc, exc_info=True,
                )
                results.append({
                    "success": False,
                    "pipeline_id": _new_uuid(),
                    "calculation_id": _new_uuid(),
                    "error": str(exc),
                    "input_index": idx,
                })

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        batch_provenance = _compute_hash({
            "batch_id": batch_id,
            "inputs_count": len(inputs),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_co2e_kg": total_co2e_kg,
        })

        # Wrap results in a summary envelope
        batch_summary: Dict[str, Any] = {
            "batch_id": batch_id,
            "total_co2e_kg": round(total_co2e_kg, 6),
            "total_co2e_tonnes": round(total_co2e_tonnes, 9),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(inputs),
            "processing_time_ms": round(elapsed_ms, 3),
            "provenance_hash": batch_provenance,
            "timestamp": _utcnow_iso(),
        }

        # Attach summary to each result
        for r in results:
            r["batch_id"] = batch_id

        logger.info(
            "Batch %s completed: %d/%d success, %.4f tCO2e, %.1fms",
            batch_id, success_count, len(inputs),
            total_co2e_tonnes, elapsed_ms,
        )

        # Return list with summary as first element
        return [batch_summary] + results

    # ------------------------------------------------------------------
    # Public: Pipeline statistics
    # ------------------------------------------------------------------

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Return pipeline-level aggregate statistics.

        Returns:
            Dictionary with total runs, avg duration, success rate,
            stage timing averages, and recent run summaries.
        """
        with self._lock:
            avg_duration = (
                self._total_duration_ms / self._total_runs
                if self._total_runs > 0
                else 0.0
            )
            success_rate = (
                self._successful_runs / self._total_runs * 100.0
                if self._total_runs > 0
                else 0.0
            )

            stage_avg_timings = {}
            for stage, timings in self._stage_timings.items():
                if timings:
                    stage_avg_timings[stage] = {
                        "avg_ms": round(
                            sum(timings) / len(timings), 3,
                        ),
                        "min_ms": round(min(timings), 3),
                        "max_ms": round(max(timings), 3),
                        "count": len(timings),
                    }

            return {
                "total_runs": self._total_runs,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "success_rate_pct": round(success_rate, 2),
                "total_duration_ms": round(
                    self._total_duration_ms, 3,
                ),
                "avg_duration_ms": round(avg_duration, 3),
                "last_run_at": self._last_run_at,
                "pipeline_stages": PIPELINE_STAGES,
                "stage_timings": stage_avg_timings,
                "recent_runs": list(
                    self._pipeline_results.values(),
                )[-10:],
                "timestamp": _utcnow_iso(),
            }

    def reset_stats(self) -> None:
        """Reset all pipeline statistics counters to zero.

        Thread-safe. Useful for testing and metric reset scenarios.
        """
        with self._lock:
            self._total_runs = 0
            self._successful_runs = 0
            self._failed_runs = 0
            self._total_duration_ms = 0.0
            self._last_run_at = None
            self._pipeline_results.clear()
            self._checkpoints.clear()
            for stage in self._stage_timings:
                self._stage_timings[stage] = []

        logger.info(
            "MobileCombustionPipelineEngine statistics reset",
        )

    # ==================================================================
    # Private: Stage execution dispatcher
    # ==================================================================

    def _execute_stage(
        self,
        stage_name: str,
        ctx: PipelineContext,
    ) -> StageResult:
        """Execute an individual pipeline stage.

        Delegates to the appropriate stage handler method based on
        stage name. Records provenance entry and timing for each stage.

        Args:
            stage_name: Pipeline stage name (one of PIPELINE_STAGES).
            ctx: Mutable PipelineContext carrying data between stages.

        Returns:
            StageResult with execution status, timing, and provenance.
        """
        stage_handlers = {
            PipelineStage.VALIDATE.value: self._stage_validate,
            PipelineStage.RESOLVE_VEHICLE.value: self._stage_resolve_vehicle,
            PipelineStage.ESTIMATE_FUEL_OR_DISTANCE.value: self._stage_estimate_fuel,
            PipelineStage.CALCULATE_EMISSIONS.value: self._stage_calculate,
            PipelineStage.APPLY_BIOFUEL_ADJUSTMENT.value: self._stage_biofuel_adjustment,
            PipelineStage.QUANTIFY_UNCERTAINTY.value: self._stage_uncertainty,
            PipelineStage.CHECK_COMPLIANCE.value: self._stage_compliance,
            PipelineStage.GENERATE_AUDIT.value: self._stage_audit,
        }

        handler = stage_handlers.get(stage_name)
        if handler is None:
            return StageResult(
                stage_name=stage_name,
                status="FAILED",
                error=f"Unknown pipeline stage: {stage_name}",
            )

        t0 = time.perf_counter()
        try:
            result = handler(ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.duration_ms = elapsed_ms

            # Record provenance entry
            ctx.provenance_chain.append(result.provenance_hash)

            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline stage %s failed: %s",
                stage_name, exc, exc_info=True,
            )
            return StageResult(
                stage_name=stage_name,
                status="FAILED",
                duration_ms=elapsed_ms,
                error=str(exc),
            )

    # ==================================================================
    # Stage 1: VALIDATE
    # ==================================================================

    def _stage_validate(self, ctx: PipelineContext) -> StageResult:
        """Stage 1: Validate and normalise all inputs.

        Checks that required fields are present, values are within valid
        ranges, and normalises string identifiers to uppercase.

        Args:
            ctx: Pipeline context with input_data.

        Returns:
            StageResult with validation status and any errors/warnings.
        """
        stage = PipelineStage.VALIDATE.value
        inp = ctx.input_data
        errors: List[str] = []
        warnings: List[str] = []

        # Validate calculation_method
        method = inp.get("calculation_method", "FUEL_BASED").upper()
        if method not in SUPPORTED_METHODS:
            errors.append(
                f"calculation_method '{method}' is not supported; "
                f"valid methods: {SUPPORTED_METHODS}"
            )
        inp["calculation_method"] = method
        ctx.calculation_method = method

        # Validate vehicle_type is present
        vehicle_type = inp.get("vehicle_type", "")
        if not vehicle_type:
            warnings.append(
                "vehicle_type not specified; using default"
            )
            vehicle_type = self._get_default_vehicle_type()
            inp["vehicle_type"] = vehicle_type

        # Validate fuel_type is present
        fuel_type = inp.get("fuel_type", "")
        if not fuel_type:
            warnings.append(
                "fuel_type not specified; using default"
            )
            fuel_type = self._get_default_fuel_type()
            inp["fuel_type"] = fuel_type

        # Method-specific validation
        if method == "FUEL_BASED":
            fuel_quantity = inp.get("fuel_quantity")
            fuel_unit = inp.get("fuel_unit", "GALLONS")
            if fuel_quantity is None:
                errors.append(
                    "fuel_quantity is required for FUEL_BASED method"
                )
            elif not isinstance(fuel_quantity, (int, float)):
                errors.append("fuel_quantity must be a number")
            elif fuel_quantity <= 0:
                errors.append("fuel_quantity must be > 0")

            if fuel_unit not in VOLUME_TO_GALLONS:
                errors.append(
                    f"fuel_unit '{fuel_unit}' is not supported; "
                    f"valid units: {list(VOLUME_TO_GALLONS.keys())}"
                )

        elif method == "DISTANCE_BASED":
            distance = inp.get("distance")
            distance_unit = inp.get("distance_unit", "KM")
            if distance is None:
                errors.append(
                    "distance is required for DISTANCE_BASED method"
                )
            elif not isinstance(distance, (int, float)):
                errors.append("distance must be a number")
            elif distance <= 0:
                errors.append("distance must be > 0")

            if distance_unit not in DISTANCE_TO_KM:
                errors.append(
                    f"distance_unit '{distance_unit}' is not "
                    f"supported; valid units: "
                    f"{list(DISTANCE_TO_KM.keys())}"
                )

        elif method == "SPEND_BASED":
            spend_amount = inp.get("spend_amount")
            spend_currency = inp.get("spend_currency", "USD")
            if spend_amount is None:
                errors.append(
                    "spend_amount is required for SPEND_BASED method"
                )
            elif not isinstance(spend_amount, (int, float)):
                errors.append("spend_amount must be a number")
            elif spend_amount <= 0:
                errors.append("spend_amount must be > 0")

            if spend_currency != "USD":
                warnings.append(
                    f"spend_currency '{spend_currency}' will be "
                    f"treated as USD equivalent"
                )

        # Validate GWP source
        gwp_source = inp.get("gwp_source", "AR6").upper()
        if gwp_source not in GWP_TABLES:
            warnings.append(
                f"gwp_source '{gwp_source}' not recognized; "
                f"using AR6"
            )
            gwp_source = "AR6"
        inp["gwp_source"] = gwp_source

        # Validate vehicle_id format (optional)
        vehicle_id = inp.get("vehicle_id")
        if vehicle_id is not None and not isinstance(vehicle_id, str):
            errors.append("vehicle_id must be a string")

        # Validate facility_id format (optional)
        facility_id = inp.get("facility_id")
        if facility_id is not None and not isinstance(facility_id, str):
            errors.append("facility_id must be a string")

        is_valid = len(errors) == 0
        ctx.errors.extend(errors)
        ctx.warnings.extend(warnings)

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": ctx.pipeline_id,
            "valid": is_valid,
            "error_count": len(errors),
            "method": method,
        })

        return StageResult(
            stage_name=stage,
            status="SUCCESS" if is_valid else "FAILED",
            data={
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "calculation_method": method,
                "vehicle_type": inp.get("vehicle_type", ""),
                "fuel_type": inp.get("fuel_type", ""),
            },
            error="; ".join(errors) if errors else "",
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Stage 2: RESOLVE_VEHICLE
    # ==================================================================

    def _stage_resolve_vehicle(
        self, ctx: PipelineContext,
    ) -> StageResult:
        """Stage 2: Look up vehicle type and resolve emission factors.

        Queries the vehicle database engine for the specified vehicle
        type to retrieve emission factors, fuel economy defaults, and
        vehicle category information.

        Args:
            ctx: Pipeline context with input_data.

        Returns:
            StageResult with resolved vehicle properties.
        """
        stage = PipelineStage.RESOLVE_VEHICLE.value
        inp = ctx.input_data
        vehicle_type = inp.get("vehicle_type", "")
        fuel_type = inp.get("fuel_type", "GASOLINE").upper()
        gwp_source = inp.get("gwp_source", "AR6")

        # Try engine-based vehicle resolution
        if self.vehicle_db is not None:
            try:
                props = self.vehicle_db.get_vehicle_type(vehicle_type)
                if props is not None:
                    if hasattr(props, "model_dump"):
                        props_dict = props.model_dump(mode="json")
                    elif isinstance(props, dict):
                        props_dict = props
                    else:
                        props_dict = {"vehicle_type": vehicle_type}

                    ctx.vehicle_props = props_dict

                    # Extract emission factors
                    factors = self.vehicle_db.get_emission_factors(
                        vehicle_type=vehicle_type,
                        fuel_type=fuel_type,
                        gwp_source=gwp_source,
                    )
                    if hasattr(factors, "model_dump"):
                        ctx.emission_factors = factors.model_dump(
                            mode="json",
                        )
                    elif isinstance(factors, dict):
                        ctx.emission_factors = factors

                    provenance_hash = _compute_hash({
                        "stage": stage,
                        "pipeline_id": ctx.pipeline_id,
                        "vehicle_type": vehicle_type,
                        "fuel_type": fuel_type,
                    })

                    return StageResult(
                        stage_name=stage,
                        status="SUCCESS",
                        data={
                            "vehicle_type": vehicle_type,
                            "fuel_type": fuel_type,
                            "source": "engine",
                        },
                        provenance_hash=provenance_hash,
                    )

            except (AttributeError, TypeError, KeyError) as exc:
                logger.warning(
                    "Vehicle DB lookup failed for %s: %s",
                    vehicle_type, exc,
                )

        # Fallback: use built-in defaults
        gwp_table = GWP_TABLES.get(gwp_source, DEFAULT_GWP_AR6)

        co2_factor = FUEL_CO2_FACTORS_KG_PER_GALLON.get(
            fuel_type, FUEL_CO2_FACTORS_KG_PER_GALLON.get(
                "GASOLINE", 8.887,
            ),
        )

        ch4_n2o = CH4_G_PER_MILE.get(
            vehicle_type, CH4_G_PER_MILE.get(
                "PASSENGER_CAR_GASOLINE",
                {"CH4": 0.0113, "N2O": 0.0045},
            ),
        )

        fuel_economy = DEFAULT_FUEL_ECONOMY_L_PER_100KM.get(
            vehicle_type, 8.9,
        )

        ctx.vehicle_props = {
            "vehicle_type": vehicle_type,
            "fuel_type": fuel_type,
            "fuel_economy_l_per_100km": fuel_economy,
            "category": self._infer_vehicle_category(vehicle_type),
            "stub": True,
        }

        ctx.emission_factors = {
            "co2_kg_per_gallon": co2_factor,
            "ch4_g_per_mile": ch4_n2o.get("CH4", 0.0),
            "n2o_g_per_mile": ch4_n2o.get("N2O", 0.0),
            "gwp_ch4": gwp_table.get("CH4", 27.9),
            "gwp_n2o": gwp_table.get("N2O", 273.0),
            "gwp_source": gwp_source,
            "source": "default",
        }

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": ctx.pipeline_id,
            "vehicle_type": vehicle_type,
            "fuel_type": fuel_type,
            "stub": True,
        })

        return StageResult(
            stage_name=stage,
            status="SUCCESS",
            data={
                "vehicle_type": vehicle_type,
                "fuel_type": fuel_type,
                "fuel_economy_l_per_100km": fuel_economy,
                "co2_kg_per_gallon": co2_factor,
                "source": "default",
            },
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Stage 3: ESTIMATE_FUEL_OR_DISTANCE
    # ==================================================================

    def _stage_estimate_fuel(
        self, ctx: PipelineContext,
    ) -> StageResult:
        """Stage 3: Convert between fuel quantity and distance as needed.

        For FUEL_BASED: converts fuel quantity to gallons and estimates
        distance for CH4/N2O calculations.
        For DISTANCE_BASED: converts distance to km and estimates fuel
        consumption using fuel economy.
        For SPEND_BASED: estimates fuel quantity from spend amount.

        Args:
            ctx: Pipeline context with input_data and vehicle_props.

        Returns:
            StageResult with estimated fuel/distance values.
        """
        stage = PipelineStage.ESTIMATE_FUEL_OR_DISTANCE.value
        inp = ctx.input_data
        method = ctx.calculation_method
        vehicle_props = ctx.vehicle_props or {}

        fuel_economy_l_per_100km = vehicle_props.get(
            "fuel_economy_l_per_100km",
            DEFAULT_FUEL_ECONOMY_L_PER_100KM.get(
                inp.get("vehicle_type", ""), 8.9,
            ),
        )

        # Override with custom fuel economy if provided
        custom_fuel_economy = inp.get("fuel_economy")
        custom_fuel_economy_unit = inp.get(
            "fuel_economy_unit", "L_PER_100KM",
        )
        if custom_fuel_economy is not None:
            fuel_economy_l_per_100km = self._convert_fuel_economy(
                custom_fuel_economy, custom_fuel_economy_unit,
            )

        # Try engine-based estimation
        if self.distance_estimator is not None:
            try:
                est_result = self.distance_estimator.estimate(
                    input_data=inp,
                    vehicle_props=vehicle_props,
                )
                if isinstance(est_result, dict):
                    ctx.fuel_quantity_gallons = est_result.get(
                        "fuel_quantity_gallons",
                    )
                    ctx.distance_km = est_result.get("distance_km")

                    provenance_hash = _compute_hash({
                        "stage": stage,
                        "pipeline_id": ctx.pipeline_id,
                        "fuel_gallons": ctx.fuel_quantity_gallons,
                        "distance_km": ctx.distance_km,
                        "source": "engine",
                    })

                    return StageResult(
                        stage_name=stage,
                        status="SUCCESS",
                        data={
                            "fuel_quantity_gallons": ctx.fuel_quantity_gallons,
                            "distance_km": ctx.distance_km,
                            "source": "engine",
                        },
                        provenance_hash=provenance_hash,
                    )

            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "DistanceEstimatorEngine failed: %s", exc,
                )

        # Fallback: built-in estimation
        if method == "FUEL_BASED":
            fuel_quantity = float(inp.get("fuel_quantity", 0.0))
            fuel_unit = inp.get("fuel_unit", "GALLONS")
            conversion = VOLUME_TO_GALLONS.get(fuel_unit, 1.0)
            fuel_gallons = fuel_quantity * conversion
            ctx.fuel_quantity_gallons = fuel_gallons

            # Estimate distance from fuel: gallons -> liters -> km
            liters = fuel_gallons / 0.264172
            if fuel_economy_l_per_100km > 0:
                distance_km = (liters / fuel_economy_l_per_100km) * 100.0
            else:
                distance_km = 0.0
            ctx.distance_km = distance_km

        elif method == "DISTANCE_BASED":
            distance = float(inp.get("distance", 0.0))
            distance_unit = inp.get("distance_unit", "KM")
            conversion = DISTANCE_TO_KM.get(distance_unit, 1.0)
            distance_km = distance * conversion
            ctx.distance_km = distance_km

            # Estimate fuel from distance: km -> liters -> gallons
            liters = (distance_km / 100.0) * fuel_economy_l_per_100km
            fuel_gallons = liters * 0.264172
            ctx.fuel_quantity_gallons = fuel_gallons

        elif method == "SPEND_BASED":
            spend_amount = float(inp.get("spend_amount", 0.0))
            fuel_type = inp.get("fuel_type", "GASOLINE").upper()

            # Estimate fuel quantity: approximate price per gallon
            fuel_prices_usd = {
                "GASOLINE": 3.50,
                "DIESEL": 3.80,
                "JET_FUEL": 5.50,
                "MARINE_DIESEL": 3.90,
                "CNG": 2.50,
                "LNG": 3.00,
                "LPG": 2.80,
                "AVIATION_GASOLINE": 6.00,
            }
            price_per_gallon = fuel_prices_usd.get(fuel_type, 3.50)
            fuel_gallons = spend_amount / price_per_gallon
            ctx.fuel_quantity_gallons = fuel_gallons

            # Estimate distance
            liters = fuel_gallons / 0.264172
            if fuel_economy_l_per_100km > 0:
                distance_km = (liters / fuel_economy_l_per_100km) * 100.0
            else:
                distance_km = 0.0
            ctx.distance_km = distance_km

        else:
            ctx.fuel_quantity_gallons = 0.0
            ctx.distance_km = 0.0

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": ctx.pipeline_id,
            "method": method,
            "fuel_gallons": ctx.fuel_quantity_gallons,
            "distance_km": ctx.distance_km,
        })

        return StageResult(
            stage_name=stage,
            status="SUCCESS",
            data={
                "calculation_method": method,
                "fuel_quantity_gallons": round(
                    ctx.fuel_quantity_gallons or 0.0, 6,
                ),
                "distance_km": round(ctx.distance_km or 0.0, 3),
                "fuel_economy_l_per_100km": fuel_economy_l_per_100km,
                "source": "default",
            },
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Stage 4: CALCULATE_EMISSIONS
    # ==================================================================

    def _stage_calculate(
        self, ctx: PipelineContext,
    ) -> StageResult:
        """Stage 4: Apply emission calculation method.

        Supports three calculation methods per GHG Protocol Chapter 7:
        - FUEL_BASED: fuel_quantity * CO2_factor + distance * CH4/N2O
        - DISTANCE_BASED: distance * fuel_economy * CO2_factor + CH4/N2O
        - SPEND_BASED: spend_amount * spend_emission_factor

        All calculations are deterministic (zero-hallucination).

        Args:
            ctx: Pipeline context with fuel/distance data and factors.

        Returns:
            StageResult with detailed emission calculation breakdown.
        """
        stage = PipelineStage.CALCULATE_EMISSIONS.value
        inp = ctx.input_data
        method = ctx.calculation_method
        factors = ctx.emission_factors or {}

        # Delegate to engine if available
        if self.calculator is not None:
            try:
                calc_result = self.calculator.calculate(
                    input_data=inp,
                    fuel_quantity_gallons=ctx.fuel_quantity_gallons,
                    distance_km=ctx.distance_km,
                    emission_factors=factors,
                    vehicle_props=ctx.vehicle_props,
                )
                if hasattr(calc_result, "model_dump"):
                    result_dict = calc_result.model_dump(mode="json")
                elif isinstance(calc_result, dict):
                    result_dict = calc_result
                else:
                    result_dict = {"total_co2e_kg": 0.0}

                ctx.calculation_result = result_dict

                provenance_hash = _compute_hash({
                    "stage": stage,
                    "pipeline_id": ctx.pipeline_id,
                    "total_co2e_kg": result_dict.get(
                        "total_co2e_kg", 0.0,
                    ),
                    "source": "engine",
                })

                return StageResult(
                    stage_name=stage,
                    status="SUCCESS",
                    data=result_dict,
                    provenance_hash=provenance_hash,
                )

            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(
                    "EmissionCalculatorEngine failed: %s", exc,
                )

        # Fallback: built-in deterministic calculation
        fuel_gallons = ctx.fuel_quantity_gallons or 0.0
        distance_km = ctx.distance_km or 0.0
        distance_miles = distance_km / 1.60934
        fuel_type = inp.get("fuel_type", "GASOLINE").upper()
        gwp_source = inp.get("gwp_source", "AR6")
        gwp_table = GWP_TABLES.get(gwp_source, DEFAULT_GWP_AR6)

        calculation_trace: List[str] = []

        if method == "FUEL_BASED":
            result_dict = self._calculate_fuel_based(
                fuel_gallons, distance_miles, fuel_type,
                factors, gwp_table, calculation_trace,
            )
        elif method == "DISTANCE_BASED":
            result_dict = self._calculate_distance_based(
                fuel_gallons, distance_miles, fuel_type,
                factors, gwp_table, calculation_trace,
            )
        elif method == "SPEND_BASED":
            result_dict = self._calculate_spend_based(
                inp, fuel_type, gwp_table, calculation_trace,
            )
        else:
            result_dict = {
                "total_co2e_kg": 0.0,
                "total_co2e_tonnes": 0.0,
                "error": f"Unknown calculation method: {method}",
            }

        result_dict["calculation_method"] = method
        result_dict["calculation_trace"] = calculation_trace
        result_dict["gwp_source"] = gwp_source

        ctx.calculation_result = result_dict

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": ctx.pipeline_id,
            "total_co2e_kg": result_dict.get("total_co2e_kg", 0.0),
            "method": method,
        })

        return StageResult(
            stage_name=stage,
            status="SUCCESS",
            data=result_dict,
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Stage 5: APPLY_BIOFUEL_ADJUSTMENT
    # ==================================================================

    def _stage_biofuel_adjustment(
        self, ctx: PipelineContext,
    ) -> StageResult:
        """Stage 5: Separate biogenic and fossil CO2 from biofuel blends.

        Per GHG Protocol guidance, biogenic CO2 from biofuel combustion
        (E10, E85, B5, B20, B100, SAF, ethanol, biodiesel) is reported
        separately and excluded from Scope 1 totals.

        Args:
            ctx: Pipeline context with calculation_result.

        Returns:
            StageResult with biogenic/fossil CO2 separation.
        """
        stage = PipelineStage.APPLY_BIOFUEL_ADJUSTMENT.value
        calc_result = ctx.calculation_result or {}
        inp = ctx.input_data
        fuel_type = inp.get("fuel_type", "GASOLINE").upper()

        fossil_fraction = BIOFUEL_FOSSIL_FRACTION.get(fuel_type, 1.0)
        biogenic_fraction = 1.0 - fossil_fraction

        total_co2_kg = calc_result.get("co2_kg", 0.0)
        total_co2e_kg = calc_result.get("total_co2e_kg", 0.0)
        ch4_co2e_kg = calc_result.get("ch4_co2e_kg", 0.0)
        n2o_co2e_kg = calc_result.get("n2o_co2e_kg", 0.0)

        # Only CO2 is split; CH4 and N2O are always fossil-origin
        fossil_co2_kg = total_co2_kg * fossil_fraction
        biogenic_co2_kg = total_co2_kg * biogenic_fraction

        # Adjusted total excludes biogenic CO2
        adjusted_co2e_kg = fossil_co2_kg + ch4_co2e_kg + n2o_co2e_kg
        adjusted_co2e_tonnes = adjusted_co2e_kg / 1000.0

        adjustment = {
            "fuel_type": fuel_type,
            "fossil_fraction": fossil_fraction,
            "biogenic_fraction": biogenic_fraction,
            "total_co2_kg": round(total_co2_kg, 6),
            "fossil_co2_kg": round(fossil_co2_kg, 6),
            "biogenic_co2_kg": round(biogenic_co2_kg, 6),
            "biogenic_co2_tonnes": round(biogenic_co2_kg / 1000.0, 9),
            "ch4_co2e_kg": round(ch4_co2e_kg, 6),
            "n2o_co2e_kg": round(n2o_co2e_kg, 6),
            "adjusted_co2e_kg": round(adjusted_co2e_kg, 6),
            "adjusted_co2e_tonnes": round(adjusted_co2e_tonnes, 9),
        }

        ctx.biofuel_adjustment = adjustment

        # Update calculation result with biofuel-adjusted values
        if ctx.calculation_result is not None:
            ctx.calculation_result["fossil_co2_kg"] = adjustment[
                "fossil_co2_kg"
            ]
            ctx.calculation_result["biogenic_co2_kg"] = adjustment[
                "biogenic_co2_kg"
            ]
            ctx.calculation_result["biogenic_co2_tonnes"] = adjustment[
                "biogenic_co2_tonnes"
            ]
            ctx.calculation_result["adjusted_co2e_kg"] = adjustment[
                "adjusted_co2e_kg"
            ]
            ctx.calculation_result["adjusted_co2e_tonnes"] = adjustment[
                "adjusted_co2e_tonnes"
            ]

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": ctx.pipeline_id,
            "fossil_co2_kg": adjustment["fossil_co2_kg"],
            "biogenic_co2_kg": adjustment["biogenic_co2_kg"],
        })

        return StageResult(
            stage_name=stage,
            status="SUCCESS",
            data=adjustment,
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Stage 6: QUANTIFY_UNCERTAINTY
    # ==================================================================

    def _stage_uncertainty(
        self, ctx: PipelineContext,
    ) -> StageResult:
        """Stage 6: Run Monte Carlo or analytical uncertainty quantification.

        Estimates the uncertainty range around the calculated emissions
        using either Monte Carlo simulation or analytical propagation,
        depending on engine availability and configuration.

        Args:
            ctx: Pipeline context with calculation_result.

        Returns:
            StageResult with uncertainty quantification results.
        """
        stage = PipelineStage.QUANTIFY_UNCERTAINTY.value
        calc_result = ctx.calculation_result or {}

        # Determine iterations
        iterations = 5000
        if self.config is not None:
            iterations = getattr(
                self.config, "monte_carlo_iterations", 5000,
            )

        # Try engine-based uncertainty
        if self.uncertainty is not None:
            try:
                unc_result = self.uncertainty.quantify(
                    calculation_result=calc_result,
                    iterations=iterations,
                )
                if hasattr(unc_result, "model_dump"):
                    unc_dict = unc_result.model_dump(mode="json")
                elif isinstance(unc_result, dict):
                    unc_dict = unc_result
                else:
                    unc_dict = {}

                ctx.uncertainty_result = unc_dict

                provenance_hash = _compute_hash({
                    "stage": stage,
                    "pipeline_id": ctx.pipeline_id,
                    "mean_co2e_kg": unc_dict.get(
                        "mean_co2e_kg", 0.0,
                    ),
                    "source": "engine",
                })

                return StageResult(
                    stage_name=stage,
                    status="SUCCESS",
                    data=unc_dict,
                    provenance_hash=provenance_hash,
                )

            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(
                    "UncertaintyQuantifierEngine failed: %s", exc,
                )

        # Fallback: analytical uncertainty estimation
        total_co2e_kg = calc_result.get("total_co2e_kg", 0.0)
        tier = ctx.input_data.get("tier", "TIER_1")
        params = UNCERTAINTY_PARAMETERS.get(
            tier, UNCERTAINTY_PARAMETERS["TIER_1"],
        )

        # Compute combined uncertainty using error propagation
        co2_unc = params["co2_relative_uncertainty"]
        ch4_unc = params["ch4_relative_uncertainty"]
        n2o_unc = params["n2o_relative_uncertainty"]

        co2_kg = calc_result.get("co2_kg", total_co2e_kg)
        ch4_co2e = calc_result.get("ch4_co2e_kg", 0.0)
        n2o_co2e = calc_result.get("n2o_co2e_kg", 0.0)

        # Root sum of squares for combined uncertainty
        if total_co2e_kg > 0:
            combined_abs_unc = math.sqrt(
                (co2_kg * co2_unc) ** 2
                + (ch4_co2e * ch4_unc) ** 2
                + (n2o_co2e * n2o_unc) ** 2
            )
            relative_unc = combined_abs_unc / total_co2e_kg
        else:
            combined_abs_unc = 0.0
            relative_unc = 0.0

        mean_co2e = total_co2e_kg
        std_co2e = combined_abs_unc
        p5 = max(0.0, mean_co2e - 1.645 * std_co2e)
        p95 = mean_co2e + 1.645 * std_co2e

        unc_dict = {
            "calculation_id": ctx.calculation_id,
            "method": "analytical",
            "mean_co2e_kg": round(mean_co2e, 6),
            "std_co2e_kg": round(std_co2e, 6),
            "p5_co2e_kg": round(p5, 6),
            "p95_co2e_kg": round(p95, 6),
            "confidence_interval_pct": 90.0,
            "relative_uncertainty_pct": round(
                relative_unc * 100.0, 2,
            ),
            "iterations": 0,
            "tier": tier,
            "data_quality_score": {"TIER_1": 3, "TIER_2": 4, "TIER_3": 5}.get(
                tier, 3,
            ),
        }

        ctx.uncertainty_result = unc_dict

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": ctx.pipeline_id,
            "mean_co2e_kg": unc_dict["mean_co2e_kg"],
            "method": "analytical",
        })

        return StageResult(
            stage_name=stage,
            status="SUCCESS",
            data=unc_dict,
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Stage 7: CHECK_COMPLIANCE
    # ==================================================================

    def _stage_compliance(
        self, ctx: PipelineContext,
    ) -> StageResult:
        """Stage 7: Validate against regulatory frameworks.

        Checks the calculation result against applicable regulatory
        framework requirements (GHG Protocol, EPA, ISO 14064, CSRD,
        EU ETS, UK SECR).

        Args:
            ctx: Pipeline context with calculation_result.

        Returns:
            StageResult with compliance check results.
        """
        stage = PipelineStage.CHECK_COMPLIANCE.value
        inp = ctx.input_data
        calc_result = ctx.calculation_result or {}

        framework = inp.get("regulatory_framework", "GHG_PROTOCOL")
        if isinstance(framework, str):
            frameworks = [framework]
        elif isinstance(framework, list):
            frameworks = framework
        else:
            frameworks = ["GHG_PROTOCOL"]

        # Try engine-based compliance check
        if self.compliance is not None:
            try:
                comp_result = self.compliance.check(
                    calculation_result=calc_result,
                    frameworks=frameworks,
                )
                if isinstance(comp_result, dict):
                    ctx.compliance_results = [comp_result]
                elif isinstance(comp_result, list):
                    ctx.compliance_results = comp_result
                else:
                    ctx.compliance_results = []

                provenance_hash = _compute_hash({
                    "stage": stage,
                    "pipeline_id": ctx.pipeline_id,
                    "frameworks": frameworks,
                    "source": "engine",
                })

                return StageResult(
                    stage_name=stage,
                    status="SUCCESS",
                    data={"compliance_results": ctx.compliance_results},
                    provenance_hash=provenance_hash,
                )

            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "ComplianceCheckerEngine failed: %s", exc,
                )

        # Fallback: built-in compliance check
        compliance_records: List[Dict[str, Any]] = []
        has_biogenic = ctx.biofuel_adjustment is not None
        has_uncertainty = ctx.uncertainty_result is not None

        for fw in frameworks:
            requirements = COMPLIANCE_REQUIREMENTS.get(fw, [])
            checks: List[Dict[str, Any]] = []

            for req in requirements:
                # Determine compliance based on data availability
                compliant = True
                notes = ""

                if "Biogenic" in req.get("description", ""):
                    compliant = has_biogenic
                    if not compliant:
                        notes = "Biogenic CO2 tracking not performed"

                elif "Uncertainty" in req.get("description", ""):
                    compliant = has_uncertainty
                    if not compliant:
                        notes = "Uncertainty analysis not performed"

                elif "Fuel Consumption" in req.get(
                    "requirement_name", "",
                ):
                    compliant = (
                        ctx.fuel_quantity_gallons is not None
                        and ctx.fuel_quantity_gallons > 0
                    )
                    if not compliant:
                        notes = "No fuel consumption data available"

                checks.append({
                    "requirement_id": req["requirement_id"],
                    "requirement_name": req["requirement_name"],
                    "compliant": compliant,
                    "notes": notes,
                })

            compliant_count = sum(
                1 for c in checks if c["compliant"]
            )
            total_reqs = len(checks)

            compliance_records.append({
                "framework": fw,
                "compliant": compliant_count == total_reqs,
                "compliant_count": compliant_count,
                "total_requirements": total_reqs,
                "checks": checks,
            })

        ctx.compliance_results = compliance_records

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": ctx.pipeline_id,
            "frameworks": frameworks,
            "records_count": len(compliance_records),
        })

        return StageResult(
            stage_name=stage,
            status="SUCCESS",
            data={
                "compliance_results": compliance_records,
                "frameworks_checked": frameworks,
            },
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Stage 8: GENERATE_AUDIT
    # ==================================================================

    def _stage_audit(self, ctx: PipelineContext) -> StageResult:
        """Stage 8: Create provenance chain and audit trail.

        Generates a complete audit trail entry with all pipeline inputs,
        outputs, stage results, and provenance chain for regulatory
        traceability.

        Args:
            ctx: Pipeline context with all accumulated results.

        Returns:
            StageResult with audit trail entry.
        """
        stage = PipelineStage.GENERATE_AUDIT.value
        calc_result = ctx.calculation_result or {}

        audit_entry = {
            "audit_id": _new_uuid(),
            "pipeline_id": ctx.pipeline_id,
            "calculation_id": ctx.calculation_id,
            "timestamp": _utcnow_iso(),
            "input_summary": {
                "vehicle_type": ctx.input_data.get(
                    "vehicle_type", "",
                ),
                "fuel_type": ctx.input_data.get("fuel_type", ""),
                "calculation_method": ctx.calculation_method,
                "gwp_source": ctx.input_data.get("gwp_source", "AR6"),
            },
            "output_summary": {
                "total_co2e_kg": calc_result.get(
                    "total_co2e_kg", 0.0,
                ),
                "total_co2e_tonnes": calc_result.get(
                    "total_co2e_tonnes", 0.0,
                ),
                "fossil_co2_kg": calc_result.get(
                    "fossil_co2_kg", 0.0,
                ),
                "biogenic_co2_kg": calc_result.get(
                    "biogenic_co2_kg", 0.0,
                ),
            },
            "stages_executed": [
                sr.stage_name for sr in ctx.stage_results
                if sr.status == "SUCCESS"
            ],
            "stages_failed": [
                sr.stage_name for sr in ctx.stage_results
                if sr.status == "FAILED"
            ],
            "provenance_chain": ctx.provenance_chain,
            "errors": ctx.errors,
            "warnings": ctx.warnings,
        }

        # Compute chain hash from entire audit entry
        chain_hash = _compute_hash(audit_entry)
        audit_entry["chain_hash"] = chain_hash

        ctx.audit_entries.append(audit_entry)

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": ctx.pipeline_id,
            "audit_id": audit_entry["audit_id"],
            "chain_hash": chain_hash,
        })

        return StageResult(
            stage_name=stage,
            status="SUCCESS",
            data=audit_entry,
            provenance_hash=provenance_hash,
        )

    # ==================================================================
    # Private calculation methods (zero-hallucination)
    # ==================================================================

    def _calculate_fuel_based(
        self,
        fuel_gallons: float,
        distance_miles: float,
        fuel_type: str,
        factors: Dict[str, Any],
        gwp_table: Dict[str, float],
        trace: List[str],
    ) -> Dict[str, Any]:
        """Fuel-based emission calculation.

        CO2 = fuel_gallons * CO2_factor (kg/gallon)
        CH4 = distance_miles * CH4_factor (g/mile) / 1000 * GWP_CH4
        N2O = distance_miles * N2O_factor (g/mile) / 1000 * GWP_N2O
        Total CO2e = CO2 + CH4_CO2e + N2O_CO2e

        Args:
            fuel_gallons: Fuel consumed in US gallons.
            distance_miles: Distance travelled in miles.
            fuel_type: Fuel type identifier.
            factors: Emission factor dictionary.
            gwp_table: GWP values dictionary.
            trace: Mutable trace list for recording steps.

        Returns:
            Dictionary with per-gas and total emission results.
        """
        co2_factor = factors.get(
            "co2_kg_per_gallon",
            FUEL_CO2_FACTORS_KG_PER_GALLON.get(fuel_type, 8.887),
        )
        ch4_factor = factors.get("ch4_g_per_mile", 0.0113)
        n2o_factor = factors.get("n2o_g_per_mile", 0.0045)
        gwp_ch4 = factors.get("gwp_ch4", gwp_table.get("CH4", 27.9))
        gwp_n2o = factors.get("gwp_n2o", gwp_table.get("N2O", 273.0))

        # CO2 calculation
        co2_kg = fuel_gallons * co2_factor
        trace.append(
            f"CO2 = {fuel_gallons:.4f} gal * "
            f"{co2_factor:.4f} kg/gal = {co2_kg:.4f} kg"
        )

        # CH4 calculation
        ch4_kg = (distance_miles * ch4_factor) / 1000.0
        ch4_co2e_kg = ch4_kg * gwp_ch4
        trace.append(
            f"CH4 = {distance_miles:.2f} mi * "
            f"{ch4_factor:.4f} g/mi / 1000 = {ch4_kg:.6f} kg; "
            f"CO2e = {ch4_kg:.6f} * {gwp_ch4:.1f} = "
            f"{ch4_co2e_kg:.4f} kg"
        )

        # N2O calculation
        n2o_kg = (distance_miles * n2o_factor) / 1000.0
        n2o_co2e_kg = n2o_kg * gwp_n2o
        trace.append(
            f"N2O = {distance_miles:.2f} mi * "
            f"{n2o_factor:.4f} g/mi / 1000 = {n2o_kg:.6f} kg; "
            f"CO2e = {n2o_kg:.6f} * {gwp_n2o:.1f} = "
            f"{n2o_co2e_kg:.4f} kg"
        )

        # Total
        total_co2e_kg = co2_kg + ch4_co2e_kg + n2o_co2e_kg
        total_co2e_tonnes = total_co2e_kg / 1000.0
        trace.append(
            f"Total CO2e = {co2_kg:.4f} + {ch4_co2e_kg:.4f} + "
            f"{n2o_co2e_kg:.4f} = {total_co2e_kg:.4f} kg "
            f"({total_co2e_tonnes:.6f} tonnes)"
        )

        return {
            "co2_kg": round(co2_kg, 6),
            "ch4_kg": round(ch4_kg, 9),
            "n2o_kg": round(n2o_kg, 9),
            "ch4_co2e_kg": round(ch4_co2e_kg, 6),
            "n2o_co2e_kg": round(n2o_co2e_kg, 6),
            "total_co2e_kg": round(total_co2e_kg, 6),
            "total_co2e_tonnes": round(total_co2e_tonnes, 9),
            "fuel_gallons": round(fuel_gallons, 6),
            "distance_miles": round(distance_miles, 3),
            "gas_emissions": [
                {"gas": "CO2", "kg": round(co2_kg, 6), "co2e_kg": round(co2_kg, 6)},
                {"gas": "CH4", "kg": round(ch4_kg, 9), "co2e_kg": round(ch4_co2e_kg, 6)},
                {"gas": "N2O", "kg": round(n2o_kg, 9), "co2e_kg": round(n2o_co2e_kg, 6)},
            ],
            "status": "SUCCESS",
        }

    def _calculate_distance_based(
        self,
        fuel_gallons: float,
        distance_miles: float,
        fuel_type: str,
        factors: Dict[str, Any],
        gwp_table: Dict[str, float],
        trace: List[str],
    ) -> Dict[str, Any]:
        """Distance-based emission calculation.

        Uses estimated fuel consumption from distance and fuel economy,
        then applies fuel-based factors.

        Args:
            fuel_gallons: Estimated fuel in US gallons (from distance).
            distance_miles: Distance in miles.
            fuel_type: Fuel type identifier.
            factors: Emission factor dictionary.
            gwp_table: GWP values dictionary.
            trace: Mutable trace list.

        Returns:
            Dictionary with emission results.
        """
        trace.append(
            f"Distance-based: estimated {fuel_gallons:.4f} gallons "
            f"from {distance_miles:.2f} miles"
        )
        result = self._calculate_fuel_based(
            fuel_gallons, distance_miles, fuel_type,
            factors, gwp_table, trace,
        )
        return result

    def _calculate_spend_based(
        self,
        inp: Dict[str, Any],
        fuel_type: str,
        gwp_table: Dict[str, float],
        trace: List[str],
    ) -> Dict[str, Any]:
        """Spend-based emission calculation.

        Uses expenditure-based emission factors (kg CO2e per USD).

        Args:
            inp: Input data dictionary with spend_amount.
            fuel_type: Fuel type identifier.
            gwp_table: GWP values dictionary.
            trace: Mutable trace list.

        Returns:
            Dictionary with emission results.
        """
        spend_amount = float(inp.get("spend_amount", 0.0))
        factor = SPEND_BASED_FACTORS_KG_CO2E_PER_USD.get(
            fuel_type,
            SPEND_BASED_FACTORS_KG_CO2E_PER_USD["DEFAULT"],
        )

        total_co2e_kg = spend_amount * factor
        total_co2e_tonnes = total_co2e_kg / 1000.0

        trace.append(
            f"Spend-based: ${spend_amount:.2f} * "
            f"{factor:.4f} kg CO2e/USD = "
            f"{total_co2e_kg:.4f} kg CO2e"
        )

        # Estimate CO2/CH4/N2O split (98.5% CO2, 0.5% CH4, 1% N2O)
        co2_kg = total_co2e_kg * 0.985
        ch4_co2e_kg = total_co2e_kg * 0.005
        n2o_co2e_kg = total_co2e_kg * 0.010
        ch4_kg = ch4_co2e_kg / gwp_table.get("CH4", 27.9)
        n2o_kg = n2o_co2e_kg / gwp_table.get("N2O", 273.0)

        return {
            "co2_kg": round(co2_kg, 6),
            "ch4_kg": round(ch4_kg, 9),
            "n2o_kg": round(n2o_kg, 9),
            "ch4_co2e_kg": round(ch4_co2e_kg, 6),
            "n2o_co2e_kg": round(n2o_co2e_kg, 6),
            "total_co2e_kg": round(total_co2e_kg, 6),
            "total_co2e_tonnes": round(total_co2e_tonnes, 9),
            "spend_amount_usd": round(spend_amount, 2),
            "spend_factor_kg_co2e_per_usd": factor,
            "gas_emissions": [
                {"gas": "CO2", "kg": round(co2_kg, 6), "co2e_kg": round(co2_kg, 6)},
                {"gas": "CH4", "kg": round(ch4_kg, 9), "co2e_kg": round(ch4_co2e_kg, 6)},
                {"gas": "N2O", "kg": round(n2o_kg, 9), "co2e_kg": round(n2o_co2e_kg, 6)},
            ],
            "status": "SUCCESS",
        }

    # ==================================================================
    # Private: Helper methods
    # ==================================================================

    def _normalise_input(self, input_data: Any) -> Dict[str, Any]:
        """Normalise input_data to a plain dictionary.

        Args:
            input_data: MobileCombustionInput, dict, or Pydantic model.

        Returns:
            Normalised dictionary.
        """
        if isinstance(input_data, dict):
            return dict(input_data)

        if hasattr(input_data, "model_dump"):
            return input_data.model_dump(mode="json")

        if hasattr(input_data, "__dict__"):
            return dict(input_data.__dict__)

        return {"raw_input": str(input_data)}

    def _build_final_result(
        self,
        ctx: PipelineContext,
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        """Build the final calculation result from pipeline context.

        Args:
            ctx: Completed pipeline context.
            elapsed_ms: Total pipeline duration in milliseconds.

        Returns:
            Comprehensive calculation result dictionary.
        """
        calc = ctx.calculation_result or {}
        biofuel = ctx.biofuel_adjustment or {}
        uncertainty = ctx.uncertainty_result or {}

        result = {
            "calculation_id": ctx.calculation_id,
            "status": "SUCCESS" if calc.get("status") == "SUCCESS" else "PARTIAL",
            "vehicle_type": ctx.input_data.get("vehicle_type", ""),
            "fuel_type": ctx.input_data.get("fuel_type", ""),
            "calculation_method": ctx.calculation_method,
            "gwp_source": ctx.input_data.get("gwp_source", "AR6"),
            "fuel_quantity_gallons": ctx.fuel_quantity_gallons or 0.0,
            "distance_km": ctx.distance_km or 0.0,
            "co2_kg": calc.get("co2_kg", 0.0),
            "ch4_kg": calc.get("ch4_kg", 0.0),
            "n2o_kg": calc.get("n2o_kg", 0.0),
            "ch4_co2e_kg": calc.get("ch4_co2e_kg", 0.0),
            "n2o_co2e_kg": calc.get("n2o_co2e_kg", 0.0),
            "total_co2e_kg": calc.get("total_co2e_kg", 0.0),
            "total_co2e_tonnes": calc.get("total_co2e_tonnes", 0.0),
            "fossil_co2_kg": biofuel.get("fossil_co2_kg", calc.get("co2_kg", 0.0)),
            "biogenic_co2_kg": biofuel.get("biogenic_co2_kg", 0.0),
            "biogenic_co2_tonnes": biofuel.get("biogenic_co2_tonnes", 0.0),
            "adjusted_co2e_kg": biofuel.get(
                "adjusted_co2e_kg", calc.get("total_co2e_kg", 0.0),
            ),
            "adjusted_co2e_tonnes": biofuel.get(
                "adjusted_co2e_tonnes", calc.get("total_co2e_tonnes", 0.0),
            ),
            "gas_emissions": calc.get("gas_emissions", []),
            "calculation_trace": calc.get("calculation_trace", []),
            "uncertainty": uncertainty,
            "compliance": ctx.compliance_results or [],
            "audit_entries": ctx.audit_entries,
            "provenance_hash": _compute_hash({
                "calculation_id": ctx.calculation_id,
                "total_co2e_kg": calc.get("total_co2e_kg", 0.0),
                "provenance_chain": ctx.provenance_chain,
            }),
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow_iso(),
            "facility_id": ctx.input_data.get("facility_id", ""),
            "vehicle_id": ctx.input_data.get("vehicle_id", ""),
        }

        return result

    def _update_stats(
        self,
        pipeline_id: str,
        success: bool,
        elapsed_ms: float,
        stages_completed: int,
        calculation_id: str,
    ) -> None:
        """Update thread-safe pipeline statistics.

        Args:
            pipeline_id: Pipeline run identifier.
            success: Whether the pipeline succeeded.
            elapsed_ms: Pipeline duration in milliseconds.
            stages_completed: Number of stages completed.
            calculation_id: Calculation identifier.
        """
        with self._lock:
            self._total_runs += 1
            if success:
                self._successful_runs += 1
            else:
                self._failed_runs += 1
            self._total_duration_ms += elapsed_ms
            self._last_run_at = _utcnow_iso()

            self._pipeline_results[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "calculation_id": calculation_id,
                "success": success,
                "duration_ms": round(elapsed_ms, 3),
                "stages_completed": stages_completed,
                "timestamp": self._last_run_at,
            }

            # Keep only last 100 results
            if len(self._pipeline_results) > 100:
                oldest_keys = list(self._pipeline_results.keys())[:-100]
                for key in oldest_keys:
                    del self._pipeline_results[key]

    def _record_stage_timing(
        self,
        stage_name: str,
        duration_ms: float,
    ) -> None:
        """Record stage timing for statistics.

        Args:
            stage_name: Pipeline stage identifier.
            duration_ms: Stage duration in milliseconds.
        """
        with self._lock:
            if stage_name in self._stage_timings:
                self._stage_timings[stage_name].append(duration_ms)
                # Keep only last 1000 timings per stage
                if len(self._stage_timings[stage_name]) > 1000:
                    self._stage_timings[stage_name] = (
                        self._stage_timings[stage_name][-1000:]
                    )

    def _save_checkpoint(
        self,
        batch_id: str,
        index: int,
        results: List[Dict[str, Any]],
    ) -> None:
        """Save a batch processing checkpoint.

        Args:
            batch_id: Batch identifier.
            index: Current processing index.
            results: Results processed so far.
        """
        with self._lock:
            self._checkpoints[batch_id] = {
                "batch_id": batch_id,
                "index": index,
                "results_count": len(results),
                "timestamp": _utcnow_iso(),
            }

        logger.debug(
            "Checkpoint saved for batch %s at index %d",
            batch_id, index,
        )

    def _convert_fuel_economy(
        self,
        value: float,
        unit: str,
    ) -> float:
        """Convert fuel economy to L/100km.

        Args:
            value: Fuel economy value.
            unit: Fuel economy unit.

        Returns:
            Fuel economy in L/100km.
        """
        unit = unit.upper()
        if unit == "L_PER_100KM":
            return value
        elif unit == "MPG_US":
            # MPG -> L/100km: 235.214 / MPG
            return 235.214 / value if value > 0 else 0.0
        elif unit == "MPG_UK":
            # UK MPG -> L/100km: 282.481 / MPG
            return 282.481 / value if value > 0 else 0.0
        elif unit == "KM_PER_L":
            # km/L -> L/100km: 100 / km_per_L
            return 100.0 / value if value > 0 else 0.0
        else:
            logger.warning(
                "Unknown fuel economy unit '%s'; treating as L/100km",
                unit,
            )
            return value

    def _infer_vehicle_category(
        self,
        vehicle_type: str,
    ) -> str:
        """Infer vehicle category from vehicle type identifier.

        Args:
            vehicle_type: Vehicle type string.

        Returns:
            Vehicle category string.
        """
        vehicle_type_upper = vehicle_type.upper()
        if "PASSENGER" in vehicle_type_upper:
            return "ON_ROAD_LIGHT_DUTY"
        elif "LIGHT_TRUCK" in vehicle_type_upper:
            return "ON_ROAD_LIGHT_DUTY"
        elif "HEAVY_TRUCK" in vehicle_type_upper:
            return "ON_ROAD_HEAVY_DUTY"
        elif "BUS" in vehicle_type_upper:
            return "ON_ROAD_HEAVY_DUTY"
        elif "MOTORCYCLE" in vehicle_type_upper:
            return "ON_ROAD_LIGHT_DUTY"
        elif "AIRCRAFT" in vehicle_type_upper:
            return "AVIATION"
        elif "MARINE" in vehicle_type_upper:
            return "MARINE"
        elif "LOCOMOTIVE" in vehicle_type_upper:
            return "RAIL"
        elif "OFF_ROAD" in vehicle_type_upper:
            return "OFF_ROAD"
        else:
            return "UNKNOWN"

    def _get_default_vehicle_type(self) -> str:
        """Get the default vehicle type from configuration.

        Returns:
            Default vehicle type string.
        """
        if self.config is not None:
            return getattr(
                self.config, "default_vehicle_type",
                "PASSENGER_CAR_GASOLINE",
            )
        return "PASSENGER_CAR_GASOLINE"

    def _get_default_fuel_type(self) -> str:
        """Get the default fuel type from configuration.

        Returns:
            Default fuel type string.
        """
        if self.config is not None:
            return getattr(
                self.config, "default_fuel_type", "GASOLINE",
            )
        return "GASOLINE"


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    "MobileCombustionPipelineEngine",
    "PipelineStage",
    "PipelineContext",
    "StageResult",
    "PIPELINE_STAGES",
    "SUPPORTED_METHODS",
    "FUEL_CO2_FACTORS_KG_PER_GALLON",
    "BIOFUEL_FOSSIL_FRACTION",
    "CH4_G_PER_MILE",
    "DEFAULT_FUEL_ECONOMY_L_PER_100KM",
    "DISTANCE_TO_KM",
    "VOLUME_TO_GALLONS",
    "SPEND_BASED_FACTORS_KG_CO2E_PER_USD",
    "GWP_TABLES",
    "COMPLIANCE_REQUIREMENTS",
    "UNCERTAINTY_PARAMETERS",
]
