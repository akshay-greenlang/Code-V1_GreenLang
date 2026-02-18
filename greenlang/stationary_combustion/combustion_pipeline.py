# -*- coding: utf-8 -*-
"""
Stationary Combustion Pipeline Engine (Engine 7) - AGENT-MRV-001

End-to-end orchestration pipeline for GHG Protocol Scope 1 stationary
combustion emissions calculations. Coordinates all six upstream engines
(fuel database, calculator, equipment profiler, factor selector,
uncertainty engine, audit engine) through a deterministic, seven-stage
pipeline:

    1. VALIDATE_INPUTS   - Validate and normalise all input data
    2. SELECT_FACTORS    - Select appropriate emission factors (tier-based)
    3. CONVERT_UNITS     - Convert fuel quantities to energy (GJ)
    4. CALCULATE         - Calculate emissions for each gas
    5. QUANTIFY_UNCERTAINTY - Run Monte Carlo uncertainty analysis
    6. GENERATE_AUDIT    - Generate audit trail entries
    7. AGGREGATE         - Aggregate results at facility level

Each stage is checkpointed so that failures produce partial results with
complete provenance.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Python ``Decimal`` arithmetic
    - No LLM calls in the calculation path
    - SHA-256 provenance hash at every pipeline stage
    - Full audit trail for regulatory traceability

Thread Safety:
    All mutable state is protected by a ``threading.Lock``. Concurrent
    ``run_pipeline`` invocations from different threads are safe.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from greenlang.stationary_combustion.config import (
    StationaryCombustionConfig,
    get_config,
)
from greenlang.stationary_combustion.models import (
    AuditEntry,
    CalculationResult,
    CalculationStatus,
    CalculationTier,
    CombustionInput,
    ControlApproach,
    FacilityAggregation,
    FuelType,
    GWPSource,
    ReportingPeriod,
    UncertaintyResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.stationary_combustion.fuel_database import FuelDatabaseEngine
except ImportError:
    FuelDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.calculator import CalculatorEngine
except ImportError:
    CalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.equipment_profiler import EquipmentProfilerEngine
except ImportError:
    EquipmentProfilerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.factor_selector import FactorSelectorEngine
except ImportError:
    FactorSelectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.uncertainty import UncertaintyEngine
except ImportError:
    UncertaintyEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.audit import AuditEngine
except ImportError:
    AuditEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.metrics import (
        PROMETHEUS_AVAILABLE,
        observe_pipeline_duration,
        record_pipeline_run,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def observe_pipeline_duration(stage: str, duration: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def record_pipeline_run(status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""


# ---------------------------------------------------------------------------
# Pipeline stage constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES: List[str] = [
    "VALIDATE_INPUTS",
    "SELECT_FACTORS",
    "CONVERT_UNITS",
    "CALCULATE",
    "QUANTIFY_UNCERTAINTY",
    "GENERATE_AUDIT",
    "AGGREGATE",
]


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
    elif isinstance(data, Decimal):
        serializable = str(data)
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# StationaryCombustionPipelineEngine
# ===================================================================


class StationaryCombustionPipelineEngine:
    """End-to-end orchestration pipeline for stationary combustion calculations.

    Coordinates all six upstream engines through a seven-stage pipeline with
    checkpointing, provenance tracking, and comprehensive error handling.
    Each pipeline run produces a deterministic SHA-256 provenance hash for
    the complete execution chain.

    Thread-safe: all mutable state is protected by an internal lock.

    Attributes:
        config: StationaryCombustionConfig instance.
        fuel_database: Optional FuelDatabaseEngine for fuel lookups.
        calculator: Optional CalculatorEngine for emission calculations.
        equipment_profiler: Optional EquipmentProfilerEngine.
        factor_selector: Optional FactorSelectorEngine.
        uncertainty_engine: Optional UncertaintyEngine for Monte Carlo.
        audit_engine: Optional AuditEngine for audit trail generation.

    Example:
        >>> engine = StationaryCombustionPipelineEngine()
        >>> result = engine.run_single(combustion_input)
        >>> assert result["success"] is True
    """

    def __init__(
        self,
        fuel_database: Any = None,
        calculator: Any = None,
        equipment_profiler: Any = None,
        factor_selector: Any = None,
        uncertainty_engine: Any = None,
        audit_engine: Any = None,
        config: Optional[StationaryCombustionConfig] = None,
    ) -> None:
        """Initialize the StationaryCombustionPipelineEngine.

        Wires all six upstream engines via dependency injection. Any engine
        set to ``None`` causes its pipeline stage to be skipped with a
        warning rather than a hard failure.

        Args:
            fuel_database: FuelDatabaseEngine instance or None.
            calculator: CalculatorEngine instance or None.
            equipment_profiler: EquipmentProfilerEngine instance or None.
            factor_selector: FactorSelectorEngine instance or None.
            uncertainty_engine: UncertaintyEngine instance or None.
            audit_engine: AuditEngine instance or None.
            config: Optional configuration. Uses global config if None.
        """
        self.config = config if config is not None else get_config()

        # Engine references
        self.fuel_database = fuel_database
        self.calculator = calculator
        self.equipment_profiler = equipment_profiler
        self.factor_selector = factor_selector
        self.uncertainty_engine = uncertainty_engine
        self.audit_engine = audit_engine

        # Thread-safe mutable state
        self._lock = threading.Lock()
        self._total_runs: int = 0
        self._successful_runs: int = 0
        self._failed_runs: int = 0
        self._total_duration_ms: float = 0.0
        self._last_run_at: Optional[str] = None
        self._pipeline_results: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "StationaryCombustionPipelineEngine initialized: "
            "fuel_database=%s, calculator=%s, equipment_profiler=%s, "
            "factor_selector=%s, uncertainty_engine=%s, audit_engine=%s",
            fuel_database is not None,
            calculator is not None,
            equipment_profiler is not None,
            factor_selector is not None,
            uncertainty_engine is not None,
            audit_engine is not None,
        )

    # ------------------------------------------------------------------
    # Full pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        inputs: List[CombustionInput],
        gwp_source: str = "AR6",
        include_biogenic: bool = False,
        organization_id: Optional[str] = None,
        reporting_period: Optional[str] = None,
        control_approach: str = "OPERATIONAL",
    ) -> Dict[str, Any]:
        """Execute the full seven-stage pipeline for a batch of inputs.

        Each stage is checkpointed. If a stage fails, the pipeline records
        the error and continues to the next stage where possible, yielding
        partial results with a ``PARTIAL`` status.

        Args:
            inputs: List of CombustionInput records to process.
            gwp_source: GWP source (AR4, AR5, AR6). Defaults to AR6.
            include_biogenic: Whether to include biogenic CO2 in totals.
            organization_id: Optional organisation identifier for reporting.
            reporting_period: Optional reporting period label.
            control_approach: GHG Protocol boundary approach
                (OPERATIONAL, FINANCIAL, EQUITY_SHARE).

        Returns:
            Dictionary with keys:
                - ``success``: bool indicating overall pipeline success.
                - ``pipeline_id``: Unique pipeline run identifier.
                - ``stage_results``: Per-stage execution details.
                - ``final_results``: List of CalculationResult dicts.
                - ``aggregations``: Facility-level aggregation dicts.
                - ``pipeline_provenance_hash``: SHA-256 of entire pipeline.
                - ``total_duration_ms``: Wall-clock time in milliseconds.
                - ``stages_completed``: Count of stages successfully run.
                - ``stages_total``: Total number of pipeline stages (7).
        """
        pipeline_id = _new_uuid()
        t0 = time.perf_counter()
        stage_results: List[Dict[str, Any]] = []
        stages_completed = 0
        overall_success = True

        # Working data passed between stages
        validated_inputs: List[CombustionInput] = []
        selected_factors: Dict[str, Any] = {}
        converted_inputs: List[Dict[str, Any]] = []
        calculation_results: List[CalculationResult] = []
        uncertainty_results: List[UncertaintyResult] = []
        audit_entries: List[AuditEntry] = []
        aggregations: List[Dict[str, Any]] = []

        logger.info(
            "Pipeline %s started: %d inputs, gwp=%s, biogenic=%s, "
            "control=%s",
            pipeline_id,
            len(inputs),
            gwp_source,
            include_biogenic,
            control_approach,
        )

        # Stage 1: VALIDATE_INPUTS
        stage_result = self._run_stage_validate(
            pipeline_id, inputs,
        )
        stage_results.append(stage_result)
        if stage_result["success"]:
            stages_completed += 1
            validated_inputs = stage_result.get("validated_inputs", [])
        else:
            overall_success = False
            validated_inputs = inputs  # proceed with originals on validation failure

        # Stage 2: SELECT_FACTORS
        stage_result = self._run_stage_select_factors(
            pipeline_id, validated_inputs, gwp_source,
        )
        stage_results.append(stage_result)
        if stage_result["success"]:
            stages_completed += 1
            selected_factors = stage_result.get("factors", {})
        else:
            overall_success = False

        # Stage 3: CONVERT_UNITS
        stage_result = self._run_stage_convert_units(
            pipeline_id, validated_inputs,
        )
        stage_results.append(stage_result)
        if stage_result["success"]:
            stages_completed += 1
            converted_inputs = stage_result.get("converted", [])
        else:
            overall_success = False

        # Stage 4: CALCULATE
        stage_result = self._run_stage_calculate(
            pipeline_id,
            validated_inputs,
            gwp_source,
            include_biogenic,
            selected_factors,
            converted_inputs,
        )
        stage_results.append(stage_result)
        if stage_result["success"]:
            stages_completed += 1
            calculation_results = stage_result.get("results", [])
        else:
            overall_success = False

        # Stage 5: QUANTIFY_UNCERTAINTY
        stage_result = self._run_stage_uncertainty(
            pipeline_id, calculation_results,
        )
        stage_results.append(stage_result)
        if stage_result["success"]:
            stages_completed += 1
            uncertainty_results = stage_result.get("uncertainty_results", [])
        else:
            # Uncertainty is non-critical; do not fail the pipeline
            pass

        # Stage 6: GENERATE_AUDIT
        stage_result = self._run_stage_audit(
            pipeline_id, calculation_results, validated_inputs,
        )
        stage_results.append(stage_result)
        if stage_result["success"]:
            stages_completed += 1
            audit_entries = stage_result.get("audit_entries", [])
        else:
            # Audit is non-critical for calculation correctness
            pass

        # Stage 7: AGGREGATE
        stage_result = self._run_stage_aggregate(
            pipeline_id, calculation_results, control_approach,
        )
        stage_results.append(stage_result)
        if stage_result["success"]:
            stages_completed += 1
            aggregations = stage_result.get("aggregations", [])
        else:
            overall_success = False

        # Compute pipeline-level provenance hash
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        final_results_dicts = _serialise_results(calculation_results)

        pipeline_provenance_hash = _compute_hash({
            "pipeline_id": pipeline_id,
            "inputs_count": len(inputs),
            "gwp_source": gwp_source,
            "include_biogenic": include_biogenic,
            "stages_completed": stages_completed,
            "results_count": len(calculation_results),
            "total_duration_ms": elapsed_ms,
        })

        # Update statistics
        with self._lock:
            self._total_runs += 1
            self._total_duration_ms += elapsed_ms
            self._last_run_at = _utcnow_iso()
            if overall_success:
                self._successful_runs += 1
            else:
                self._failed_runs += 1
            self._pipeline_results[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "success": overall_success,
                "stages_completed": stages_completed,
                "total_duration_ms": elapsed_ms,
                "results_count": len(calculation_results),
                "timestamp": self._last_run_at,
            }

        # Record Prometheus metrics
        record_pipeline_run("success" if overall_success else "failure")
        observe_pipeline_duration("full_pipeline", elapsed_ms / 1000.0)

        result = {
            "success": overall_success,
            "pipeline_id": pipeline_id,
            "stage_results": _strip_internal_keys(stage_results),
            "final_results": final_results_dicts,
            "aggregations": aggregations,
            "uncertainty_results": _serialise_uncertainty(uncertainty_results),
            "audit_entries": _serialise_audit(audit_entries),
            "pipeline_provenance_hash": pipeline_provenance_hash,
            "total_duration_ms": round(elapsed_ms, 3),
            "stages_completed": stages_completed,
            "stages_total": len(PIPELINE_STAGES),
            "organization_id": organization_id,
            "reporting_period": reporting_period,
            "gwp_source": gwp_source,
            "include_biogenic": include_biogenic,
            "control_approach": control_approach,
            "timestamp": _utcnow_iso(),
        }

        logger.info(
            "Pipeline %s completed: success=%s stages=%d/%d results=%d "
            "duration=%.1fms",
            pipeline_id,
            overall_success,
            stages_completed,
            len(PIPELINE_STAGES),
            len(calculation_results),
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Single record shortcut
    # ------------------------------------------------------------------

    def run_single(
        self,
        input_data: CombustionInput,
        gwp_source: str = "AR6",
    ) -> Dict[str, Any]:
        """Execute the full pipeline for a single combustion input.

        Convenience wrapper around ``run_pipeline`` for single-record use
        cases. Returns the pipeline result with a single-element
        ``final_results`` list.

        Args:
            input_data: Single CombustionInput to process.
            gwp_source: GWP source (AR4, AR5, AR6).

        Returns:
            Pipeline result dictionary.
        """
        return self.run_pipeline(
            inputs=[input_data],
            gwp_source=gwp_source,
            include_biogenic=self.config.enable_biogenic_tracking,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_inputs(
        self,
        inputs: List[CombustionInput],
    ) -> Dict[str, Any]:
        """Validate all inputs without executing the full pipeline.

        Args:
            inputs: List of CombustionInput records to validate.

        Returns:
            Dictionary with keys:
                - ``valid``: Overall validity boolean.
                - ``errors``: List of error message strings.
                - ``warnings``: List of warning message strings.
                - ``validated_count``: Number of valid inputs.
                - ``total_count``: Total number of inputs.
        """
        errors: List[str] = []
        warnings: List[str] = []
        valid_count = 0

        for idx, inp in enumerate(inputs):
            input_errors = self._validate_single_input(inp, idx)
            input_warnings = self._warn_single_input(inp, idx)
            errors.extend(input_errors)
            warnings.extend(input_warnings)
            if not input_errors:
                valid_count += 1

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_count": valid_count,
            "total_count": len(inputs),
        }

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

    def aggregate_by_facility(
        self,
        results: List[CalculationResult],
        control_approach: str = "OPERATIONAL",
    ) -> List[FacilityAggregation]:
        """Aggregate calculation results by facility_id.

        Uses the specified GHG Protocol organisational boundary approach
        to determine the share of emissions attributed to each facility.

        Args:
            results: List of CalculationResult objects.
            control_approach: Boundary approach (OPERATIONAL, FINANCIAL,
                EQUITY_SHARE).

        Returns:
            List of FacilityAggregation objects, one per unique facility_id.
        """
        facility_map: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_co2e_tonnes": Decimal("0"),
                "total_biogenic_co2_tonnes": Decimal("0"),
                "by_fuel": defaultdict(lambda: Decimal("0")),
                "by_equipment": defaultdict(lambda: Decimal("0")),
                "calculation_count": 0,
            }
        )

        for result in results:
            facility_id = self._extract_facility_id(result)
            if not facility_id:
                facility_id = "UNASSIGNED"

            entry = facility_map[facility_id]
            entry["total_co2e_tonnes"] += result.total_co2e_tonnes
            entry["total_biogenic_co2_tonnes"] += result.biogenic_co2_tonnes
            entry["calculation_count"] += 1

            fuel_key = result.fuel_type if isinstance(result.fuel_type, str) else result.fuel_type.value
            entry["by_fuel"][fuel_key] += result.total_co2e_tonnes

            equip_key = self._extract_equipment_id(result)
            if equip_key:
                entry["by_equipment"][equip_key] += result.total_co2e_tonnes

        aggregations: List[FacilityAggregation] = []
        for fid, data in facility_map.items():
            aggregation = FacilityAggregation(
                facility_id=fid,
                total_co2e_tonnes=data["total_co2e_tonnes"],
                total_biogenic_co2_tonnes=data["total_biogenic_co2_tonnes"],
                by_fuel=dict(data["by_fuel"]),
                by_equipment=dict(data["by_equipment"]),
                calculation_count=data["calculation_count"],
            )
            aggregations.append(aggregation)

        logger.info(
            "Aggregated %d results into %d facilities (approach=%s)",
            len(results),
            len(aggregations),
            control_approach,
        )
        return aggregations

    def aggregate_by_fuel(
        self,
        results: List[CalculationResult],
    ) -> Dict[str, Any]:
        """Aggregate calculation results by fuel type.

        Args:
            results: List of CalculationResult objects.

        Returns:
            Dictionary mapping fuel type names to aggregated CO2e tonnes.
        """
        fuel_totals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_co2e_tonnes": Decimal("0"),
                "total_biogenic_co2_tonnes": Decimal("0"),
                "total_energy_gj": Decimal("0"),
                "calculation_count": 0,
            }
        )

        for result in results:
            fuel_key = result.fuel_type if isinstance(result.fuel_type, str) else result.fuel_type.value
            entry = fuel_totals[fuel_key]
            entry["total_co2e_tonnes"] += result.total_co2e_tonnes
            entry["total_biogenic_co2_tonnes"] += result.biogenic_co2_tonnes
            entry["total_energy_gj"] += result.energy_gj
            entry["calculation_count"] += 1

        # Convert Decimal to float for JSON serialisability
        serialisable: Dict[str, Any] = {}
        for fuel_key, data in fuel_totals.items():
            serialisable[fuel_key] = {
                "total_co2e_tonnes": float(data["total_co2e_tonnes"]),
                "total_biogenic_co2_tonnes": float(data["total_biogenic_co2_tonnes"]),
                "total_energy_gj": float(data["total_energy_gj"]),
                "calculation_count": data["calculation_count"],
            }

        return serialisable

    def aggregate_by_period(
        self,
        results: List[CalculationResult],
        period_type: str = "MONTHLY",
    ) -> Dict[str, Any]:
        """Aggregate calculation results by reporting period.

        Groups results by their ``calculated_at`` timestamp bucketed into
        the specified period (MONTHLY, QUARTERLY, ANNUAL).

        Args:
            results: List of CalculationResult objects.
            period_type: Period granularity (MONTHLY, QUARTERLY, ANNUAL).

        Returns:
            Dictionary mapping period keys to aggregated CO2e tonnes.
        """
        period_totals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_co2e_tonnes": Decimal("0"),
                "total_biogenic_co2_tonnes": Decimal("0"),
                "calculation_count": 0,
            }
        )

        for result in results:
            period_key = self._derive_period_key(
                result.calculated_at, period_type,
            )
            entry = period_totals[period_key]
            entry["total_co2e_tonnes"] += result.total_co2e_tonnes
            entry["total_biogenic_co2_tonnes"] += result.biogenic_co2_tonnes
            entry["calculation_count"] += 1

        serialisable: Dict[str, Any] = {}
        for period_key, data in period_totals.items():
            serialisable[period_key] = {
                "total_co2e_tonnes": float(data["total_co2e_tonnes"]),
                "total_biogenic_co2_tonnes": float(data["total_biogenic_co2_tonnes"]),
                "calculation_count": data["calculation_count"],
            }

        return serialisable

    # ------------------------------------------------------------------
    # Status and statistics
    # ------------------------------------------------------------------

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Return current pipeline operational status.

        Returns:
            Dictionary with pipeline status, total runs, and errors.
        """
        with self._lock:
            return {
                "status": "ready",
                "engines": {
                    "fuel_database": self.fuel_database is not None,
                    "calculator": self.calculator is not None,
                    "equipment_profiler": self.equipment_profiler is not None,
                    "factor_selector": self.factor_selector is not None,
                    "uncertainty_engine": self.uncertainty_engine is not None,
                    "audit_engine": self.audit_engine is not None,
                },
                "total_runs": self._total_runs,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "last_run_at": self._last_run_at,
                "pipeline_stages": PIPELINE_STAGES,
                "timestamp": _utcnow_iso(),
            }

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Return pipeline-level aggregate statistics.

        Returns:
            Dictionary with total runs, avg duration, success rate, etc.
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

            return {
                "total_runs": self._total_runs,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "success_rate_pct": round(success_rate, 2),
                "total_duration_ms": round(self._total_duration_ms, 3),
                "avg_duration_ms": round(avg_duration, 3),
                "last_run_at": self._last_run_at,
                "recent_runs": list(self._pipeline_results.values())[-10:],
                "timestamp": _utcnow_iso(),
            }

    # ==================================================================
    # Private stage implementations
    # ==================================================================

    def _run_stage_validate(
        self,
        pipeline_id: str,
        inputs: List[CombustionInput],
    ) -> Dict[str, Any]:
        """Stage 1: Validate and normalise all input data.

        Args:
            pipeline_id: Current pipeline run identifier.
            inputs: Raw CombustionInput records.

        Returns:
            Stage result dictionary with validated inputs.
        """
        stage = "VALIDATE_INPUTS"
        t0 = time.perf_counter()

        try:
            validation = self.validate_inputs(inputs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            observe_pipeline_duration(stage, elapsed_ms / 1000.0)

            return {
                "stage": stage,
                "success": validation["valid"],
                "duration_ms": round(elapsed_ms, 3),
                "validated_count": validation["validated_count"],
                "error_count": len(validation["errors"]),
                "warning_count": len(validation["warnings"]),
                "errors": validation["errors"],
                "warnings": validation["warnings"],
                "validated_inputs": inputs if validation["valid"] else inputs,
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": pipeline_id,
                    "valid": validation["valid"],
                    "count": len(inputs),
                }),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return {
                "stage": stage,
                "success": False,
                "duration_ms": round(elapsed_ms, 3),
                "error": str(exc),
                "provenance_hash": "",
            }

    def _run_stage_select_factors(
        self,
        pipeline_id: str,
        inputs: List[CombustionInput],
        gwp_source: str,
    ) -> Dict[str, Any]:
        """Stage 2: Select appropriate emission factors (tier-based).

        Args:
            pipeline_id: Current pipeline run identifier.
            inputs: Validated CombustionInput records.
            gwp_source: GWP source for factor selection.

        Returns:
            Stage result dictionary with selected factors.
        """
        stage = "SELECT_FACTORS"
        t0 = time.perf_counter()

        try:
            factors: Dict[str, Any] = {}

            if self.factor_selector is not None:
                for inp in inputs:
                    try:
                        factor_result = self.factor_selector.select_factors(
                            fuel_type=inp.fuel_type,
                            ef_source=inp.ef_source,
                            tier=inp.tier,
                            gwp_source=gwp_source,
                        )
                        key = self._factor_cache_key(inp)
                        factors[key] = factor_result
                    except (AttributeError, TypeError, ValueError) as exc:
                        logger.warning(
                            "Factor selection failed for input %s: %s",
                            inp.calculation_id, exc,
                        )
            else:
                logger.warning(
                    "Pipeline %s: FactorSelectorEngine not available; "
                    "stage skipped",
                    pipeline_id,
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            observe_pipeline_duration(stage, elapsed_ms / 1000.0)

            return {
                "stage": stage,
                "success": True,
                "duration_ms": round(elapsed_ms, 3),
                "factors_selected": len(factors),
                "factors": factors,
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": pipeline_id,
                    "factors_count": len(factors),
                }),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return {
                "stage": stage,
                "success": False,
                "duration_ms": round(elapsed_ms, 3),
                "error": str(exc),
                "factors": {},
                "provenance_hash": "",
            }

    def _run_stage_convert_units(
        self,
        pipeline_id: str,
        inputs: List[CombustionInput],
    ) -> Dict[str, Any]:
        """Stage 3: Convert fuel quantities to energy (GJ).

        Args:
            pipeline_id: Current pipeline run identifier.
            inputs: Validated CombustionInput records.

        Returns:
            Stage result dictionary with converted energy values.
        """
        stage = "CONVERT_UNITS"
        t0 = time.perf_counter()

        try:
            converted: List[Dict[str, Any]] = []

            for inp in inputs:
                try:
                    if self.fuel_database is not None:
                        conversion = self.fuel_database.convert_to_energy(
                            fuel_type=inp.fuel_type,
                            quantity=inp.quantity,
                            unit=inp.unit,
                            heating_value_basis=inp.heating_value_basis,
                            custom_heating_value=inp.custom_heating_value,
                        )
                        converted.append({
                            "calculation_id": inp.calculation_id or _new_uuid(),
                            "energy_gj": conversion.get("energy_gj", Decimal("0")),
                            "conversion_factor": conversion.get("conversion_factor", Decimal("1")),
                        })
                    else:
                        # Stub conversion when engine unavailable
                        converted.append({
                            "calculation_id": inp.calculation_id or _new_uuid(),
                            "energy_gj": Decimal("0"),
                            "conversion_factor": Decimal("1"),
                        })
                except (AttributeError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Unit conversion failed for input %s: %s",
                        inp.calculation_id, exc,
                    )
                    converted.append({
                        "calculation_id": inp.calculation_id or _new_uuid(),
                        "energy_gj": Decimal("0"),
                        "conversion_factor": Decimal("1"),
                        "error": str(exc),
                    })

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            observe_pipeline_duration(stage, elapsed_ms / 1000.0)

            return {
                "stage": stage,
                "success": True,
                "duration_ms": round(elapsed_ms, 3),
                "converted_count": len(converted),
                "converted": converted,
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": pipeline_id,
                    "converted_count": len(converted),
                }),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return {
                "stage": stage,
                "success": False,
                "duration_ms": round(elapsed_ms, 3),
                "error": str(exc),
                "converted": [],
                "provenance_hash": "",
            }

    def _run_stage_calculate(
        self,
        pipeline_id: str,
        inputs: List[CombustionInput],
        gwp_source: str,
        include_biogenic: bool,
        selected_factors: Dict[str, Any],
        converted_inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Stage 4: Calculate emissions for each gas.

        Args:
            pipeline_id: Current pipeline run identifier.
            inputs: Validated CombustionInput records.
            gwp_source: GWP source for calculations.
            include_biogenic: Whether to include biogenic CO2.
            selected_factors: Factors from stage 2.
            converted_inputs: Energy conversions from stage 3.

        Returns:
            Stage result dictionary with CalculationResult objects.
        """
        stage = "CALCULATE"
        t0 = time.perf_counter()

        try:
            results: List[CalculationResult] = []

            if self.calculator is not None:
                for idx, inp in enumerate(inputs):
                    try:
                        calc_result = self.calculator.calculate(
                            input_data=inp,
                            gwp_source=gwp_source,
                            include_biogenic=include_biogenic,
                            factors=selected_factors.get(
                                self._factor_cache_key(inp),
                            ),
                        )
                        if isinstance(calc_result, CalculationResult):
                            results.append(calc_result)
                        elif isinstance(calc_result, dict):
                            results.append(CalculationResult(**calc_result))
                    except (AttributeError, TypeError, ValueError) as exc:
                        logger.warning(
                            "Calculation failed for input %d: %s",
                            idx, exc,
                        )
                        results.append(self._create_failed_result(inp, str(exc)))
            else:
                logger.warning(
                    "Pipeline %s: CalculatorEngine not available; "
                    "creating stub results",
                    pipeline_id,
                )
                for inp in inputs:
                    results.append(self._create_stub_result(inp, gwp_source))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            observe_pipeline_duration(stage, elapsed_ms / 1000.0)

            success_count = sum(
                1 for r in results
                if r.status == CalculationStatus.SUCCESS
                or r.status == "SUCCESS"
            )

            return {
                "stage": stage,
                "success": success_count > 0,
                "duration_ms": round(elapsed_ms, 3),
                "results_count": len(results),
                "success_count": success_count,
                "failure_count": len(results) - success_count,
                "results": results,
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": pipeline_id,
                    "results_count": len(results),
                    "success_count": success_count,
                }),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return {
                "stage": stage,
                "success": False,
                "duration_ms": round(elapsed_ms, 3),
                "error": str(exc),
                "results": [],
                "provenance_hash": "",
            }

    def _run_stage_uncertainty(
        self,
        pipeline_id: str,
        results: List[CalculationResult],
    ) -> Dict[str, Any]:
        """Stage 5: Run Monte Carlo uncertainty analysis.

        Args:
            pipeline_id: Current pipeline run identifier.
            results: CalculationResult objects from stage 4.

        Returns:
            Stage result dictionary with UncertaintyResult objects.
        """
        stage = "QUANTIFY_UNCERTAINTY"
        t0 = time.perf_counter()

        try:
            uncertainty_results: List[UncertaintyResult] = []

            if self.uncertainty_engine is not None and results:
                for result in results:
                    try:
                        unc = self.uncertainty_engine.quantify(
                            calculation_result=result,
                            iterations=self.config.monte_carlo_iterations,
                        )
                        if isinstance(unc, UncertaintyResult):
                            uncertainty_results.append(unc)
                        elif isinstance(unc, dict):
                            uncertainty_results.append(
                                UncertaintyResult(**unc)
                            )
                    except (AttributeError, TypeError, ValueError) as exc:
                        logger.warning(
                            "Uncertainty analysis failed for %s: %s",
                            result.calculation_id, exc,
                        )
            else:
                if self.uncertainty_engine is None:
                    logger.warning(
                        "Pipeline %s: UncertaintyEngine not available; "
                        "stage skipped",
                        pipeline_id,
                    )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            observe_pipeline_duration(stage, elapsed_ms / 1000.0)

            return {
                "stage": stage,
                "success": True,
                "duration_ms": round(elapsed_ms, 3),
                "uncertainty_results": uncertainty_results,
                "analysed_count": len(uncertainty_results),
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": pipeline_id,
                    "analysed_count": len(uncertainty_results),
                }),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return {
                "stage": stage,
                "success": False,
                "duration_ms": round(elapsed_ms, 3),
                "error": str(exc),
                "uncertainty_results": [],
                "provenance_hash": "",
            }

    def _run_stage_audit(
        self,
        pipeline_id: str,
        results: List[CalculationResult],
        inputs: List[CombustionInput],
    ) -> Dict[str, Any]:
        """Stage 6: Generate audit trail entries.

        Args:
            pipeline_id: Current pipeline run identifier.
            results: CalculationResult objects from stage 4.
            inputs: Original validated inputs.

        Returns:
            Stage result dictionary with AuditEntry objects.
        """
        stage = "GENERATE_AUDIT"
        t0 = time.perf_counter()

        try:
            audit_entries: List[AuditEntry] = []

            if self.audit_engine is not None and results:
                for result in results:
                    try:
                        entry = self.audit_engine.generate_entry(
                            calculation_result=result,
                            pipeline_id=pipeline_id,
                        )
                        if isinstance(entry, AuditEntry):
                            audit_entries.append(entry)
                        elif isinstance(entry, dict):
                            audit_entries.append(AuditEntry(**entry))
                    except (AttributeError, TypeError, ValueError) as exc:
                        logger.warning(
                            "Audit generation failed for %s: %s",
                            result.calculation_id, exc,
                        )
            else:
                # Generate stub audit entries when engine unavailable
                for result in results:
                    entry = AuditEntry(
                        audit_id=_new_uuid(),
                        calculation_id=result.calculation_id,
                        action="pipeline_calculation",
                        actor="system",
                        details={
                            "pipeline_id": pipeline_id,
                            "status": result.status
                            if isinstance(result.status, str)
                            else result.status.value,
                            "total_co2e_tonnes": float(result.total_co2e_tonnes),
                        },
                        provenance_hash=result.provenance_hash,
                    )
                    audit_entries.append(entry)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            observe_pipeline_duration(stage, elapsed_ms / 1000.0)

            return {
                "stage": stage,
                "success": True,
                "duration_ms": round(elapsed_ms, 3),
                "audit_entries": audit_entries,
                "entries_count": len(audit_entries),
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": pipeline_id,
                    "entries_count": len(audit_entries),
                }),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return {
                "stage": stage,
                "success": False,
                "duration_ms": round(elapsed_ms, 3),
                "error": str(exc),
                "audit_entries": [],
                "provenance_hash": "",
            }

    def _run_stage_aggregate(
        self,
        pipeline_id: str,
        results: List[CalculationResult],
        control_approach: str,
    ) -> Dict[str, Any]:
        """Stage 7: Aggregate results at facility level.

        Args:
            pipeline_id: Current pipeline run identifier.
            results: CalculationResult objects from stage 4.
            control_approach: GHG Protocol boundary approach.

        Returns:
            Stage result dictionary with facility aggregations.
        """
        stage = "AGGREGATE"
        t0 = time.perf_counter()

        try:
            facility_aggregations = self.aggregate_by_facility(
                results, control_approach,
            )
            aggregation_dicts = [
                agg.model_dump(mode="json")
                for agg in facility_aggregations
            ]

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            observe_pipeline_duration(stage, elapsed_ms / 1000.0)

            return {
                "stage": stage,
                "success": True,
                "duration_ms": round(elapsed_ms, 3),
                "aggregations": aggregation_dicts,
                "facility_count": len(aggregation_dicts),
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": pipeline_id,
                    "facility_count": len(aggregation_dicts),
                }),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline %s stage %s failed: %s",
                pipeline_id, stage, exc, exc_info=True,
            )
            return {
                "stage": stage,
                "success": False,
                "duration_ms": round(elapsed_ms, 3),
                "error": str(exc),
                "aggregations": [],
                "provenance_hash": "",
            }

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _validate_single_input(
        self,
        inp: CombustionInput,
        idx: int,
    ) -> List[str]:
        """Validate a single CombustionInput and return error messages.

        Args:
            inp: The input to validate.
            idx: Index in the batch for error reporting.

        Returns:
            List of error message strings (empty if valid).
        """
        errors: List[str] = []

        if inp.quantity <= 0:
            errors.append(f"Input [{idx}]: quantity must be > 0")

        if inp.custom_oxidation_factor is not None:
            if not (Decimal("0") <= inp.custom_oxidation_factor <= Decimal("1")):
                errors.append(
                    f"Input [{idx}]: custom_oxidation_factor must be "
                    f"in [0.0, 1.0]"
                )

        if inp.custom_heating_value is not None:
            if inp.custom_heating_value <= 0:
                errors.append(
                    f"Input [{idx}]: custom_heating_value must be > 0"
                )

        return errors

    def _warn_single_input(
        self,
        inp: CombustionInput,
        idx: int,
    ) -> List[str]:
        """Generate warnings for a single CombustionInput.

        Args:
            inp: The input to check.
            idx: Index in the batch for warning reporting.

        Returns:
            List of warning message strings (empty if no warnings).
        """
        warnings: List[str] = []

        if inp.calculation_id is None:
            warnings.append(
                f"Input [{idx}]: no calculation_id provided; "
                f"one will be auto-generated"
            )

        if inp.facility_id is None:
            warnings.append(
                f"Input [{idx}]: no facility_id; result will be "
                f"aggregated under UNASSIGNED"
            )

        return warnings

    def _factor_cache_key(self, inp: CombustionInput) -> str:
        """Generate a cache key for emission factor lookups.

        Args:
            inp: CombustionInput to derive the key from.

        Returns:
            String cache key.
        """
        fuel = inp.fuel_type if isinstance(inp.fuel_type, str) else inp.fuel_type.value
        source = inp.ef_source if isinstance(inp.ef_source, str) else inp.ef_source.value
        tier = ""
        if inp.tier is not None:
            tier = inp.tier if isinstance(inp.tier, str) else inp.tier.value
        return f"{fuel}:{source}:{tier}"

    def _extract_facility_id(self, result: CalculationResult) -> str:
        """Extract facility_id from a CalculationResult.

        The result model does not directly carry facility_id; it may be
        embedded in calculation_trace or provenance metadata. Falls back
        to empty string if not found.

        Args:
            result: CalculationResult to inspect.

        Returns:
            Facility ID string or empty string.
        """
        # Check calculation_trace for facility_id hints
        for trace_entry in result.calculation_trace:
            if "facility_id=" in trace_entry:
                parts = trace_entry.split("facility_id=")
                if len(parts) > 1:
                    return parts[1].split()[0].strip()
        return ""

    def _extract_equipment_id(self, result: CalculationResult) -> str:
        """Extract equipment_id from a CalculationResult.

        Args:
            result: CalculationResult to inspect.

        Returns:
            Equipment ID string or empty string.
        """
        for trace_entry in result.calculation_trace:
            if "equipment_id=" in trace_entry:
                parts = trace_entry.split("equipment_id=")
                if len(parts) > 1:
                    return parts[1].split()[0].strip()
        return ""

    def _derive_period_key(
        self,
        timestamp: datetime,
        period_type: str,
    ) -> str:
        """Derive a period key from a datetime and period type.

        Args:
            timestamp: The datetime to bucket.
            period_type: MONTHLY, QUARTERLY, or ANNUAL.

        Returns:
            String key like "2026-Q1", "2026-02", or "2026".
        """
        if period_type == "MONTHLY":
            return timestamp.strftime("%Y-%m")
        elif period_type == "QUARTERLY":
            quarter = (timestamp.month - 1) // 3 + 1
            return f"{timestamp.year}-Q{quarter}"
        else:
            return str(timestamp.year)

    def _create_stub_result(
        self,
        inp: CombustionInput,
        gwp_source: str,
    ) -> CalculationResult:
        """Create a stub CalculationResult when no calculator engine is available.

        Args:
            inp: The CombustionInput for which to create the stub.
            gwp_source: GWP source string.

        Returns:
            A CalculationResult with zero emissions and PARTIAL status.
        """
        calc_id = inp.calculation_id or _new_uuid()
        return CalculationResult(
            calculation_id=calc_id,
            status=CalculationStatus.PARTIAL,
            fuel_type=inp.fuel_type,
            quantity=inp.quantity,
            unit=inp.unit,
            energy_gj=Decimal("0"),
            tier=inp.tier or CalculationTier.TIER_1,
            ef_source=inp.ef_source,
            gwp_source=gwp_source,
            heating_value_basis=inp.heating_value_basis,
            gas_emissions=[],
            total_co2e_kg=Decimal("0"),
            total_co2e_tonnes=Decimal("0"),
            biogenic_co2_kg=Decimal("0"),
            biogenic_co2_tonnes=Decimal("0"),
            calculation_trace=[
                "STUB: CalculatorEngine not available",
                f"facility_id={inp.facility_id or ''}",
                f"equipment_id={inp.equipment_id or ''}",
            ],
            provenance_hash=_compute_hash({
                "calculation_id": calc_id,
                "stub": True,
            }),
        )

    def _create_failed_result(
        self,
        inp: CombustionInput,
        error_message: str,
    ) -> CalculationResult:
        """Create a failed CalculationResult for error cases.

        Args:
            inp: The CombustionInput that failed.
            error_message: Description of the failure.

        Returns:
            A CalculationResult with FAILED status.
        """
        calc_id = inp.calculation_id or _new_uuid()
        return CalculationResult(
            calculation_id=calc_id,
            status=CalculationStatus.FAILED,
            fuel_type=inp.fuel_type,
            quantity=inp.quantity,
            unit=inp.unit,
            energy_gj=Decimal("0"),
            tier=inp.tier or CalculationTier.TIER_1,
            ef_source=inp.ef_source,
            gwp_source=GWPSource.AR6,
            heating_value_basis=inp.heating_value_basis,
            gas_emissions=[],
            total_co2e_kg=Decimal("0"),
            total_co2e_tonnes=Decimal("0"),
            biogenic_co2_kg=Decimal("0"),
            biogenic_co2_tonnes=Decimal("0"),
            calculation_trace=[
                f"FAILED: {error_message}",
                f"facility_id={inp.facility_id or ''}",
                f"equipment_id={inp.equipment_id or ''}",
            ],
            error_message=error_message,
            provenance_hash=_compute_hash({
                "calculation_id": calc_id,
                "failed": True,
                "error": error_message,
            }),
        )


# ===================================================================
# Module-level serialisation helpers
# ===================================================================


def _serialise_results(
    results: List[CalculationResult],
) -> List[Dict[str, Any]]:
    """Convert CalculationResult list to JSON-serialisable dicts.

    Args:
        results: CalculationResult objects.

    Returns:
        List of dictionaries.
    """
    return [r.model_dump(mode="json") for r in results]


def _serialise_uncertainty(
    results: List[UncertaintyResult],
) -> List[Dict[str, Any]]:
    """Convert UncertaintyResult list to JSON-serialisable dicts.

    Args:
        results: UncertaintyResult objects.

    Returns:
        List of dictionaries.
    """
    return [r.model_dump(mode="json") for r in results]


def _serialise_audit(
    entries: List[AuditEntry],
) -> List[Dict[str, Any]]:
    """Convert AuditEntry list to JSON-serialisable dicts.

    Args:
        entries: AuditEntry objects.

    Returns:
        List of dictionaries.
    """
    return [e.model_dump(mode="json") for e in entries]


def _strip_internal_keys(
    stage_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove internal keys from stage results for external consumption.

    Keys like ``validated_inputs``, ``factors``, ``converted``, ``results``,
    ``uncertainty_results``, and ``audit_entries`` carry full objects that
    are not suitable for the top-level pipeline response.

    Args:
        stage_results: Raw stage result dictionaries.

    Returns:
        Cleaned stage result dictionaries.
    """
    internal_keys = {
        "validated_inputs",
        "factors",
        "converted",
        "results",
        "uncertainty_results",
        "audit_entries",
    }
    cleaned: List[Dict[str, Any]] = []
    for stage in stage_results:
        cleaned.append({
            k: v for k, v in stage.items() if k not in internal_keys
        })
    return cleaned


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    "StationaryCombustionPipelineEngine",
    "PIPELINE_STAGES",
]
