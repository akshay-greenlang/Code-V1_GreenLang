# -*- coding: utf-8 -*-
"""
Refrigerant Pipeline Engine (Engine 7) - AGENT-MRV-002

End-to-end orchestration pipeline for GHG Protocol Scope 1 refrigerant
and fluorinated gas emissions calculations. Coordinates all six upstream
engines (refrigerant database, emission calculator, equipment registry,
leak rate estimator, uncertainty quantifier, compliance tracker) through
a deterministic, eight-stage pipeline:

    1. VALIDATE          - Validate all inputs (refrigerant types exist,
                           charges positive, methods valid)
    2. LOOKUP_REFRIGERANT - Look up refrigerant properties and GWP from
                           database
    3. ESTIMATE_LEAK_RATE - Get leak rate (custom or default with
                           adjustments)
    4. CALCULATE         - Route to appropriate method (equipment_based,
                           mass_balance, screening, direct, top_down)
    5. DECOMPOSE_BLENDS  - Decompose blend emissions into component gases
    6. QUANTIFY_UNCERTAINTY - Run Monte Carlo or analytical uncertainty
    7. CHECK_COMPLIANCE  - Check against applicable regulatory frameworks
    8. GENERATE_AUDIT    - Generate audit trail with provenance chain

Each stage is checkpointed so that failures produce partial results with
complete provenance. Thread-safe execution with statistics counters.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Python arithmetic
    - No LLM calls in the calculation path
    - SHA-256 provenance hash at every pipeline stage
    - Full audit trail for regulatory traceability

Thread Safety:
    All mutable state is protected by a ``threading.Lock``. Concurrent
    ``run`` invocations from different threads are safe.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-002 Refrigerants & F-Gas (GL-MRV-SCOPE1-002)
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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional config import
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.config import (
        RefrigerantsFGasConfig,
        get_config,
    )
except ImportError:
    RefrigerantsFGasConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """No-op fallback when config module is unavailable."""
        return None


# ---------------------------------------------------------------------------
# Optional model imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.models import (
        CalculationInput,
        CalculationMethod,
        CalculationResult,
        CalculationStatus,
        ComplianceRecord,
        EquipmentProfile,
        GasEmission,
        GWPSource,
        LeakRateProfile,
        MassBalanceData,
        RefrigerantProperties,
        RefrigerantType,
        RegulatoryFramework,
        ServiceEvent,
        UncertaintyResult,
    )
except ImportError:
    CalculationInput = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    CalculationStatus = None  # type: ignore[assignment, misc]
    ComplianceRecord = None  # type: ignore[assignment, misc]
    EquipmentProfile = None  # type: ignore[assignment, misc]
    GasEmission = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    LeakRateProfile = None  # type: ignore[assignment, misc]
    MassBalanceData = None  # type: ignore[assignment, misc]
    RefrigerantProperties = None  # type: ignore[assignment, misc]
    RefrigerantType = None  # type: ignore[assignment, misc]
    RegulatoryFramework = None  # type: ignore[assignment, misc]
    ServiceEvent = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.refrigerant_database import RefrigerantDatabaseEngine
except ImportError:
    RefrigerantDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.emission_calculator import EmissionCalculatorEngine
except ImportError:
    EmissionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.equipment_registry import EquipmentRegistryEngine
except ImportError:
    EquipmentRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.leak_rate_estimator import LeakRateEstimatorEngine
except ImportError:
    LeakRateEstimatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.uncertainty_quantifier import UncertaintyQuantifierEngine
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.compliance_tracker import ComplianceTrackerEngine
except ImportError:
    ComplianceTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.metrics import (
        PROMETHEUS_AVAILABLE,
        observe_calculation_duration,
        record_calculation,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def observe_calculation_duration(operation: str, seconds: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def record_calculation(method: str, refrigerant_type: str, status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""


# ---------------------------------------------------------------------------
# Pipeline stage constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES: List[str] = [
    "VALIDATE",
    "LOOKUP_REFRIGERANT",
    "ESTIMATE_LEAK_RATE",
    "CALCULATE",
    "DECOMPOSE_BLENDS",
    "QUANTIFY_UNCERTAINTY",
    "CHECK_COMPLIANCE",
    "GENERATE_AUDIT",
]

# Supported calculation methods
SUPPORTED_METHODS: List[str] = [
    "equipment_based",
    "mass_balance",
    "screening",
    "direct",
    "top_down",
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
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# RefrigerantPipelineEngine
# ===================================================================


class RefrigerantPipelineEngine:
    """End-to-end orchestration pipeline for refrigerant emissions calculations.

    Coordinates all six upstream engines through an eight-stage pipeline with
    checkpointing, provenance tracking, and comprehensive error handling.
    Each pipeline run produces a deterministic SHA-256 provenance hash for
    the complete execution chain.

    Thread-safe: all mutable state is protected by an internal lock.

    Attributes:
        config: RefrigerantsFGasConfig instance or None.
        refrigerant_database: Optional RefrigerantDatabaseEngine.
        emission_calculator: Optional EmissionCalculatorEngine.
        equipment_registry: Optional EquipmentRegistryEngine.
        leak_rate_estimator: Optional LeakRateEstimatorEngine.
        uncertainty_quantifier: Optional UncertaintyQuantifierEngine.
        compliance_tracker: Optional ComplianceTrackerEngine.

    Example:
        >>> engine = RefrigerantPipelineEngine()
        >>> result = engine.run(calculation_input)
        >>> assert result["success"] is True
    """

    def __init__(
        self,
        refrigerant_database: Any = None,
        emission_calculator: Any = None,
        equipment_registry: Any = None,
        leak_rate_estimator: Any = None,
        uncertainty_quantifier: Any = None,
        compliance_tracker: Any = None,
        config: Any = None,
    ) -> None:
        """Initialize the RefrigerantPipelineEngine.

        Wires all six upstream engines via dependency injection. Any engine
        set to ``None`` causes its pipeline stage to be skipped with a
        warning rather than a hard failure.

        Args:
            refrigerant_database: RefrigerantDatabaseEngine instance or None.
            emission_calculator: EmissionCalculatorEngine instance or None.
            equipment_registry: EquipmentRegistryEngine instance or None.
            leak_rate_estimator: LeakRateEstimatorEngine instance or None.
            uncertainty_quantifier: UncertaintyQuantifierEngine instance or None.
            compliance_tracker: ComplianceTrackerEngine instance or None.
            config: Optional configuration. Uses global config if None.
        """
        self.config = config if config is not None else get_config()

        # Engine references
        self.refrigerant_database = refrigerant_database
        self.emission_calculator = emission_calculator
        self.equipment_registry = equipment_registry
        self.leak_rate_estimator = leak_rate_estimator
        self.uncertainty_quantifier = uncertainty_quantifier
        self.compliance_tracker = compliance_tracker

        # Thread-safe mutable state
        self._lock = threading.Lock()
        self._total_runs: int = 0
        self._successful_runs: int = 0
        self._failed_runs: int = 0
        self._total_duration_ms: float = 0.0
        self._last_run_at: Optional[str] = None
        self._pipeline_results: Dict[str, Dict[str, Any]] = {}

        # Checkpoint store for batch pipeline recovery
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "RefrigerantPipelineEngine initialized: "
            "refrigerant_database=%s, emission_calculator=%s, "
            "equipment_registry=%s, leak_rate_estimator=%s, "
            "uncertainty_quantifier=%s, compliance_tracker=%s",
            refrigerant_database is not None,
            emission_calculator is not None,
            equipment_registry is not None,
            leak_rate_estimator is not None,
            uncertainty_quantifier is not None,
            compliance_tracker is not None,
        )

    # ------------------------------------------------------------------
    # Full pipeline execution - single calculation
    # ------------------------------------------------------------------

    def run(self, input_data: Any) -> Dict[str, Any]:
        """Run the full eight-stage pipeline for a single calculation input.

        Each stage is checkpointed. If a stage fails, the pipeline records
        the error and continues to the next stage where possible, yielding
        partial results with complete provenance.

        Args:
            input_data: CalculationInput (or dict) with refrigerant_type,
                charge_kg, method, and optional parameters.

        Returns:
            Dictionary with keys:
                - ``success``: bool indicating overall pipeline success.
                - ``pipeline_id``: Unique pipeline run identifier.
                - ``calculation_id``: Calculation identifier.
                - ``stage_results``: Per-stage execution details.
                - ``result``: Final calculation result dict.
                - ``pipeline_provenance_hash``: SHA-256 of entire pipeline.
                - ``total_duration_ms``: Wall-clock time in milliseconds.
                - ``stages_completed``: Count of stages successfully run.
                - ``stages_total``: Total number of pipeline stages (8).
        """
        pipeline_id = _new_uuid()
        t0 = time.perf_counter()
        stage_results: List[Dict[str, Any]] = []
        stages_completed = 0
        overall_success = True

        # Normalise input to dict for internal processing
        input_dict = self._normalise_input(input_data)
        calculation_id = input_dict.get(
            "calculation_id", _new_uuid(),
        )
        input_dict["calculation_id"] = calculation_id

        # Pipeline context carried between stages
        context: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "calculation_id": calculation_id,
            "input": input_dict,
            "refrigerant_props": None,
            "gwp_value": None,
            "leak_rate": None,
            "calculation_result": None,
            "blend_decomposition": None,
            "uncertainty_result": None,
            "compliance_records": None,
            "audit_entries": [],
            "provenance_chain": [],
        }

        logger.info(
            "Pipeline %s started: calc_id=%s ref_type=%s method=%s",
            pipeline_id,
            calculation_id,
            input_dict.get("refrigerant_type", "unknown"),
            input_dict.get("method", "equipment_based"),
        )

        # Execute each stage in order
        for stage in PIPELINE_STAGES:
            stage_result = self.run_stage(stage, context)
            stage_results.append(stage_result)

            if stage_result.get("success", False):
                stages_completed += 1
            else:
                # Non-critical stages do not fail the pipeline
                if stage in ("QUANTIFY_UNCERTAINTY", "CHECK_COMPLIANCE", "GENERATE_AUDIT"):
                    logger.warning(
                        "Pipeline %s: non-critical stage %s failed: %s",
                        pipeline_id, stage,
                        stage_result.get("error", "unknown"),
                    )
                else:
                    overall_success = False
                    logger.error(
                        "Pipeline %s: critical stage %s failed: %s",
                        pipeline_id, stage,
                        stage_result.get("error", "unknown"),
                    )

        # Compute pipeline-level provenance hash
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pipeline_provenance_hash = _compute_hash({
            "pipeline_id": pipeline_id,
            "calculation_id": calculation_id,
            "refrigerant_type": input_dict.get("refrigerant_type", ""),
            "stages_completed": stages_completed,
            "total_duration_ms": elapsed_ms,
        })

        # Build final result from context
        final_result = self._build_final_result(context, elapsed_ms)

        # Update statistics
        self._update_stats(
            pipeline_id, overall_success, elapsed_ms,
            stages_completed, calculation_id,
        )

        # Record Prometheus metrics
        ref_type = input_dict.get("refrigerant_type", "unknown")
        method = input_dict.get("method", "equipment_based")
        record_calculation(
            method, ref_type,
            "success" if overall_success else "failure",
        )
        observe_calculation_duration(
            "single_calculation", elapsed_ms / 1000.0,
        )

        result = {
            "success": overall_success,
            "pipeline_id": pipeline_id,
            "calculation_id": calculation_id,
            "stage_results": self._strip_internal_keys(stage_results),
            "result": final_result,
            "pipeline_provenance_hash": pipeline_provenance_hash,
            "total_duration_ms": round(elapsed_ms, 3),
            "stages_completed": stages_completed,
            "stages_total": len(PIPELINE_STAGES),
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
    # Batch pipeline execution
    # ------------------------------------------------------------------

    def run_batch(
        self,
        inputs: List[Any],
        checkpoint_interval: int = 10,
    ) -> Dict[str, Any]:
        """Run the full pipeline for a batch of calculation inputs.

        Supports checkpointing at configurable intervals so that partial
        batch results are recoverable on failure.

        Args:
            inputs: List of CalculationInput objects or dicts.
            checkpoint_interval: Save checkpoint every N calculations.

        Returns:
            Dictionary with keys:
                - ``batch_id``: Unique batch identifier.
                - ``results``: List of individual pipeline results.
                - ``total_emissions_kg_co2e``: Aggregate emissions.
                - ``total_emissions_tco2e``: Aggregate emissions in tonnes.
                - ``success_count``: Number of successful calculations.
                - ``failure_count``: Number of failed calculations.
                - ``processing_time_ms``: Total batch processing time.
                - ``provenance_hash``: SHA-256 batch provenance hash.
        """
        batch_id = _new_uuid()
        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        success_count = 0
        failure_count = 0
        total_emissions_kg = 0.0
        total_emissions_tco2e = 0.0

        logger.info(
            "Batch %s started: %d inputs, checkpoint_interval=%d",
            batch_id, len(inputs), checkpoint_interval,
        )

        for idx, inp in enumerate(inputs):
            try:
                pipeline_result = self.run(inp)
                results.append(pipeline_result)

                if pipeline_result.get("success", False):
                    success_count += 1
                    result_data = pipeline_result.get("result", {})
                    total_emissions_kg += result_data.get(
                        "total_emissions_kg_co2e", 0.0,
                    )
                    total_emissions_tco2e += result_data.get(
                        "total_emissions_tco2e", 0.0,
                    )
                else:
                    failure_count += 1

                # Checkpoint at intervals
                if (idx + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(batch_id, idx + 1, results)

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
            "total_emissions_kg": total_emissions_kg,
        })

        batch_result = {
            "batch_id": batch_id,
            "results": results,
            "total_emissions_kg_co2e": round(total_emissions_kg, 6),
            "total_emissions_tco2e": round(total_emissions_tco2e, 9),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(inputs),
            "processing_time_ms": round(elapsed_ms, 3),
            "provenance_hash": batch_provenance,
            "timestamp": _utcnow_iso(),
        }

        logger.info(
            "Batch %s completed: %d/%d success, %.4f tCO2e, %.1fms",
            batch_id, success_count, len(inputs),
            total_emissions_tco2e, elapsed_ms,
        )
        return batch_result

    # ------------------------------------------------------------------
    # Individual stage execution
    # ------------------------------------------------------------------

    def run_stage(
        self,
        stage: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run an individual pipeline stage.

        Delegates to the appropriate stage handler method based on stage
        name. Records provenance entry and timing for each stage.

        Args:
            stage: Pipeline stage name (one of PIPELINE_STAGES).
            context: Mutable pipeline context dict carrying data between
                stages.

        Returns:
            Stage result dictionary with success status, timing, and
            provenance hash.
        """
        stage_handlers = {
            "VALIDATE": self._run_stage_validate,
            "LOOKUP_REFRIGERANT": self._run_stage_lookup_refrigerant,
            "ESTIMATE_LEAK_RATE": self._run_stage_estimate_leak_rate,
            "CALCULATE": self._run_stage_calculate,
            "DECOMPOSE_BLENDS": self._run_stage_decompose_blends,
            "QUANTIFY_UNCERTAINTY": self._run_stage_quantify_uncertainty,
            "CHECK_COMPLIANCE": self._run_stage_check_compliance,
            "GENERATE_AUDIT": self._run_stage_generate_audit,
        }

        handler = stage_handlers.get(stage)
        if handler is None:
            return {
                "stage": stage,
                "success": False,
                "duration_ms": 0.0,
                "error": f"Unknown pipeline stage: {stage}",
                "provenance_hash": "",
            }

        t0 = time.perf_counter()
        try:
            result = handler(context)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result["duration_ms"] = round(elapsed_ms, 3)

            # Record provenance entry
            provenance_entry = {
                "stage": stage,
                "success": result.get("success", False),
                "duration_ms": result["duration_ms"],
                "provenance_hash": result.get("provenance_hash", ""),
                "timestamp": _utcnow_iso(),
            }
            context["provenance_chain"].append(provenance_entry)

            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Pipeline stage %s failed: %s",
                stage, exc, exc_info=True,
            )
            return {
                "stage": stage,
                "success": False,
                "duration_ms": round(elapsed_ms, 3),
                "error": str(exc),
                "provenance_hash": "",
            }

    # ------------------------------------------------------------------
    # Facility aggregation
    # ------------------------------------------------------------------

    def aggregate_facility(
        self,
        results: List[Dict[str, Any]],
        control_approach: str = "OPERATIONAL",
        share: float = 1.0,
    ) -> Dict[str, Any]:
        """Aggregate pipeline results at facility level.

        Supports three GHG Protocol organisational boundary approaches:
        operational control, financial control, and equity share.

        Args:
            results: List of pipeline result dictionaries.
            control_approach: Boundary approach (OPERATIONAL, FINANCIAL,
                EQUITY_SHARE). Defaults to OPERATIONAL.
            share: Ownership or control share fraction (0.0-1.0). Only
                applicable for FINANCIAL and EQUITY_SHARE approaches.
                Defaults to 1.0.

        Returns:
            Dictionary with aggregated emissions by facility, equipment
            type, and refrigerant type.
        """
        # Determine share multiplier based on control approach
        if control_approach == "OPERATIONAL":
            share_multiplier = 1.0
        elif control_approach in ("FINANCIAL", "EQUITY_SHARE"):
            share_multiplier = max(0.0, min(1.0, share))
        else:
            share_multiplier = 1.0
            logger.warning(
                "Unknown control_approach '%s'; defaulting to "
                "OPERATIONAL (share=1.0)",
                control_approach,
            )

        facility_totals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_emissions_kg_co2e": 0.0,
                "total_emissions_tco2e": 0.0,
                "by_refrigerant": defaultdict(float),
                "by_equipment_type": defaultdict(float),
                "calculation_count": 0,
                "calculation_ids": [],
            },
        )

        for pipeline_result in results:
            result_data = pipeline_result.get("result", {})
            facility_id = result_data.get("facility_id", "UNASSIGNED")
            calc_id = pipeline_result.get("calculation_id", "")

            emissions_kg = result_data.get(
                "total_emissions_kg_co2e", 0.0,
            ) * share_multiplier
            emissions_tco2e = result_data.get(
                "total_emissions_tco2e", 0.0,
            ) * share_multiplier

            entry = facility_totals[facility_id]
            entry["total_emissions_kg_co2e"] += emissions_kg
            entry["total_emissions_tco2e"] += emissions_tco2e
            entry["calculation_count"] += 1
            entry["calculation_ids"].append(calc_id)

            ref_type = result_data.get("refrigerant_type", "unknown")
            entry["by_refrigerant"][ref_type] += emissions_tco2e

            equip_type = result_data.get("equipment_type", "unknown")
            entry["by_equipment_type"][equip_type] += emissions_tco2e

        # Build aggregation response
        aggregations = []
        for facility_id, data in facility_totals.items():
            aggregations.append({
                "facility_id": facility_id,
                "total_emissions_kg_co2e": round(
                    data["total_emissions_kg_co2e"], 6,
                ),
                "total_emissions_tco2e": round(
                    data["total_emissions_tco2e"], 9,
                ),
                "by_refrigerant": dict(data["by_refrigerant"]),
                "by_equipment_type": dict(data["by_equipment_type"]),
                "calculation_count": data["calculation_count"],
                "calculation_ids": data["calculation_ids"],
                "control_approach": control_approach,
                "share": share_multiplier,
                "provenance_hash": _compute_hash({
                    "facility_id": facility_id,
                    "total_emissions_tco2e": data["total_emissions_tco2e"],
                    "control_approach": control_approach,
                }),
            })

        grand_total_tco2e = sum(
            a["total_emissions_tco2e"] for a in aggregations
        )

        logger.info(
            "Aggregated %d results into %d facilities "
            "(approach=%s, share=%.2f): %.4f tCO2e",
            len(results), len(aggregations),
            control_approach, share_multiplier,
            grand_total_tco2e,
        )

        return {
            "aggregations": aggregations,
            "total_facilities": len(aggregations),
            "grand_total_tco2e": round(grand_total_tco2e, 9),
            "control_approach": control_approach,
            "share": share_multiplier,
            "provenance_hash": _compute_hash({
                "aggregation_count": len(aggregations),
                "grand_total_tco2e": grand_total_tco2e,
            }),
            "timestamp": _utcnow_iso(),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Return pipeline-level aggregate statistics.

        Returns:
            Dictionary with total runs, avg duration, success rate,
            and recent run summaries.
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
                "pipeline_stages": PIPELINE_STAGES,
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

        logger.info("RefrigerantPipelineEngine statistics reset")

    # ==================================================================
    # Private stage implementations
    # ==================================================================

    def _run_stage_validate(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 1: Validate all inputs.

        Checks that refrigerant type exists, charge is positive, and
        calculation method is valid.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result dictionary with validation details.
        """
        stage = "VALIDATE"
        input_data = context["input"]
        errors: List[str] = []
        warnings: List[str] = []

        # Validate refrigerant_type is present
        ref_type = input_data.get("refrigerant_type", "")
        if not ref_type:
            errors.append("refrigerant_type is required")

        # Validate charge_kg is positive
        charge_kg = input_data.get("charge_kg", 0.0)
        if not isinstance(charge_kg, (int, float)):
            errors.append("charge_kg must be a number")
        elif charge_kg <= 0:
            errors.append("charge_kg must be greater than 0")

        # Validate method is supported
        method = input_data.get("method", "equipment_based")
        if method not in SUPPORTED_METHODS:
            errors.append(
                f"method '{method}' is not supported; "
                f"valid methods: {SUPPORTED_METHODS}"
            )

        # Validate GWP source if specified
        gwp_source = input_data.get("gwp_source", "AR6")
        valid_gwp_sources = ["AR4", "AR5", "AR6", "AR6_20yr"]
        if gwp_source not in valid_gwp_sources:
            warnings.append(
                f"gwp_source '{gwp_source}' is not standard; "
                f"using AR6 as fallback"
            )

        # Validate mass_balance specific fields
        if method == "mass_balance":
            mass_balance = input_data.get("mass_balance_data")
            if mass_balance is None:
                errors.append(
                    "mass_balance_data is required for mass_balance method"
                )

        # Validate equipment_id for equipment_based method
        if method == "equipment_based":
            if not input_data.get("equipment_id") and not input_data.get("equipment_type"):
                warnings.append(
                    "equipment_based method without equipment_id or "
                    "equipment_type; using default leak rate"
                )

        # Validate leak_rate_pct if custom
        custom_leak = input_data.get("custom_leak_rate_pct")
        if custom_leak is not None:
            if not isinstance(custom_leak, (int, float)):
                errors.append("custom_leak_rate_pct must be a number")
            elif custom_leak < 0 or custom_leak > 100:
                errors.append(
                    "custom_leak_rate_pct must be between 0 and 100"
                )

        # Validate facility_id format (optional)
        facility_id = input_data.get("facility_id")
        if facility_id is not None and not isinstance(facility_id, str):
            errors.append("facility_id must be a string")

        is_valid = len(errors) == 0

        provenance_hash = _compute_hash({
            "stage": stage,
            "pipeline_id": context["pipeline_id"],
            "valid": is_valid,
            "error_count": len(errors),
        })

        return {
            "stage": stage,
            "success": is_valid,
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "provenance_hash": provenance_hash,
        }

    def _run_stage_lookup_refrigerant(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 2: Look up refrigerant properties and GWP from database.

        Queries the refrigerant database engine for the specified
        refrigerant type to retrieve GWP values, blend components,
        and physical properties.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result dictionary with refrigerant properties.
        """
        stage = "LOOKUP_REFRIGERANT"
        input_data = context["input"]
        ref_type = input_data.get("refrigerant_type", "")
        gwp_source = input_data.get("gwp_source", "AR6")

        if self.refrigerant_database is not None:
            try:
                props = self.refrigerant_database.get_refrigerant(ref_type)
                if props is None:
                    return {
                        "stage": stage,
                        "success": False,
                        "error": f"Refrigerant type '{ref_type}' not found",
                        "provenance_hash": "",
                    }

                if hasattr(props, "model_dump"):
                    props_dict = props.model_dump(mode="json")
                elif isinstance(props, dict):
                    props_dict = props
                else:
                    props_dict = {"refrigerant_type": ref_type}

                context["refrigerant_props"] = props_dict

                # Extract GWP value for the specified source
                gwp_value = self._extract_gwp(props_dict, gwp_source)
                context["gwp_value"] = gwp_value

                return {
                    "stage": stage,
                    "success": True,
                    "refrigerant_type": ref_type,
                    "gwp_value": gwp_value,
                    "gwp_source": gwp_source,
                    "is_blend": props_dict.get("is_blend", False),
                    "category": props_dict.get("category", ""),
                    "provenance_hash": _compute_hash({
                        "stage": stage,
                        "pipeline_id": context["pipeline_id"],
                        "refrigerant_type": ref_type,
                        "gwp_value": gwp_value,
                    }),
                }

            except (AttributeError, TypeError, KeyError) as exc:
                logger.warning(
                    "Refrigerant lookup failed for %s: %s",
                    ref_type, exc,
                )

        # Stub lookup when engine unavailable
        logger.warning(
            "RefrigerantDatabaseEngine not available; "
            "using stub GWP for %s",
            ref_type,
        )
        stub_gwp = self._get_stub_gwp(ref_type, gwp_source)
        context["refrigerant_props"] = {
            "refrigerant_type": ref_type,
            "gwp_100yr": stub_gwp,
            "is_blend": False,
            "stub": True,
        }
        context["gwp_value"] = stub_gwp

        return {
            "stage": stage,
            "success": True,
            "refrigerant_type": ref_type,
            "gwp_value": stub_gwp,
            "gwp_source": gwp_source,
            "is_blend": False,
            "category": "UNKNOWN",
            "message": "Used stub GWP value",
            "provenance_hash": _compute_hash({
                "stage": stage,
                "pipeline_id": context["pipeline_id"],
                "refrigerant_type": ref_type,
                "gwp_value": stub_gwp,
                "stub": True,
            }),
        }

    def _run_stage_estimate_leak_rate(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 3: Get leak rate (custom or default with adjustments).

        If a custom leak rate is provided in the input, uses that directly.
        Otherwise queries the leak rate estimator engine for a default
        rate based on equipment type, age, and climate factors.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result dictionary with estimated leak rate.
        """
        stage = "ESTIMATE_LEAK_RATE"
        input_data = context["input"]

        # Check for custom leak rate
        custom_leak = input_data.get("custom_leak_rate_pct")
        if custom_leak is not None:
            leak_rate_pct = float(custom_leak)
            context["leak_rate"] = leak_rate_pct

            return {
                "stage": stage,
                "success": True,
                "leak_rate_pct": leak_rate_pct,
                "source": "custom",
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": context["pipeline_id"],
                    "leak_rate_pct": leak_rate_pct,
                    "source": "custom",
                }),
            }

        # Try engine-based leak rate estimation
        equipment_type = input_data.get("equipment_type", "")
        equipment_id = input_data.get("equipment_id", "")
        age_years = input_data.get("age_years", 0)

        if self.leak_rate_estimator is not None:
            try:
                lr_result = self.leak_rate_estimator.estimate(
                    equipment_type=equipment_type,
                    equipment_id=equipment_id,
                    age_years=age_years,
                    refrigerant_type=input_data.get(
                        "refrigerant_type", "",
                    ),
                )
                if hasattr(lr_result, "model_dump"):
                    lr_dict = lr_result.model_dump(mode="json")
                elif isinstance(lr_result, dict):
                    lr_dict = lr_result
                else:
                    lr_dict = {"leak_rate_pct": float(lr_result)}

                leak_rate_pct = lr_dict.get("leak_rate_pct", 5.0)
                context["leak_rate"] = leak_rate_pct

                return {
                    "stage": stage,
                    "success": True,
                    "leak_rate_pct": leak_rate_pct,
                    "source": "engine",
                    "equipment_type": equipment_type,
                    "adjustments": lr_dict.get("adjustments", {}),
                    "provenance_hash": _compute_hash({
                        "stage": stage,
                        "pipeline_id": context["pipeline_id"],
                        "leak_rate_pct": leak_rate_pct,
                        "source": "engine",
                    }),
                }

            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(
                    "Leak rate estimation failed: %s", exc,
                )

        # Default leak rate based on equipment type
        default_rates = {
            "COMMERCIAL_REFRIGERATION": 15.0,
            "INDUSTRIAL_REFRIGERATION": 10.0,
            "RESIDENTIAL_AC": 4.0,
            "COMMERCIAL_AC": 6.0,
            "CHILLER_CENTRIFUGAL": 2.0,
            "CHILLER_SCREW": 3.0,
            "CHILLER_RECIPROCATING": 5.0,
            "HEAT_PUMP": 3.0,
            "TRANSPORT_REFRIGERATION": 15.0,
            "MOBILE_AC": 12.0,
            "SWITCHGEAR": 0.5,
            "FOAM_BLOWING": 3.5,
            "FIRE_SUPPRESSION": 1.0,
            "AEROSOL": 50.0,
            "SEMICONDUCTOR": 5.0,
        }
        leak_rate_pct = default_rates.get(
            equipment_type.upper(), 5.0,
        )

        # Apply age adjustment factor
        age_factor = 1.0 + (float(age_years) * 0.02)
        leak_rate_pct = min(leak_rate_pct * age_factor, 100.0)

        context["leak_rate"] = leak_rate_pct

        return {
            "stage": stage,
            "success": True,
            "leak_rate_pct": round(leak_rate_pct, 4),
            "source": "default",
            "equipment_type": equipment_type,
            "age_years": age_years,
            "age_factor": round(age_factor, 4),
            "provenance_hash": _compute_hash({
                "stage": stage,
                "pipeline_id": context["pipeline_id"],
                "leak_rate_pct": leak_rate_pct,
                "source": "default",
            }),
        }

    def _run_stage_calculate(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 4: Route to appropriate calculation method.

        Supports five calculation methods per GHG Protocol Chapter 8:
        - equipment_based: charge * leak_rate * GWP
        - mass_balance: (purchases - recovery - inventory_change) * GWP
        - screening: activity_data * screening_factor * GWP
        - direct: measured_emissions * GWP
        - top_down: total_charge * leak_rate * GWP

        All calculations are deterministic (zero-hallucination).

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result dictionary with calculation result.
        """
        stage = "CALCULATE"
        input_data = context["input"]
        method = input_data.get("method", "equipment_based")
        gwp_value = context.get("gwp_value", 0.0)

        # Delegate to engine if available
        if self.emission_calculator is not None:
            try:
                calc_result = self.emission_calculator.calculate(
                    input_data=input_data,
                    gwp_value=gwp_value,
                    leak_rate_pct=context.get("leak_rate", 5.0),
                    refrigerant_props=context.get(
                        "refrigerant_props", {},
                    ),
                )
                if hasattr(calc_result, "model_dump"):
                    result_dict = calc_result.model_dump(mode="json")
                elif isinstance(calc_result, dict):
                    result_dict = calc_result
                else:
                    result_dict = {"total_emissions_kg_co2e": 0.0}

                context["calculation_result"] = result_dict

                return {
                    "stage": stage,
                    "success": True,
                    "method": method,
                    "total_emissions_kg_co2e": result_dict.get(
                        "total_emissions_kg_co2e", 0.0,
                    ),
                    "total_emissions_tco2e": result_dict.get(
                        "total_emissions_tco2e", 0.0,
                    ),
                    "provenance_hash": _compute_hash({
                        "stage": stage,
                        "pipeline_id": context["pipeline_id"],
                        "method": method,
                        "emissions": result_dict.get(
                            "total_emissions_kg_co2e", 0.0,
                        ),
                    }),
                }

            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(
                    "EmissionCalculatorEngine failed: %s", exc,
                )

        # Deterministic fallback calculation (zero-hallucination)
        result_dict = self._calculate_fallback(
            input_data, gwp_value, context,
        )
        context["calculation_result"] = result_dict

        return {
            "stage": stage,
            "success": True,
            "method": method,
            "total_emissions_kg_co2e": result_dict.get(
                "total_emissions_kg_co2e", 0.0,
            ),
            "total_emissions_tco2e": result_dict.get(
                "total_emissions_tco2e", 0.0,
            ),
            "calculation_trace": result_dict.get(
                "calculation_trace", [],
            ),
            "provenance_hash": _compute_hash({
                "stage": stage,
                "pipeline_id": context["pipeline_id"],
                "method": method,
                "emissions": result_dict.get(
                    "total_emissions_kg_co2e", 0.0,
                ),
            }),
        }

    def _run_stage_decompose_blends(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 5: Decompose blend emissions into component gases.

        If the refrigerant is a blend (e.g. R-404A, R-410A, R-407C),
        decomposes total emissions into per-component gas emissions
        using weight fractions and individual GWP values.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result dictionary with blend decomposition.
        """
        stage = "DECOMPOSE_BLENDS"
        props = context.get("refrigerant_props", {})
        is_blend = props.get("is_blend", False)

        if not is_blend:
            # Pure refrigerant - no decomposition needed
            return {
                "stage": stage,
                "success": True,
                "is_blend": False,
                "message": "Not a blend; decomposition skipped",
                "components": [],
                "provenance_hash": _compute_hash({
                    "stage": stage,
                    "pipeline_id": context["pipeline_id"],
                    "is_blend": False,
                }),
            }

        calc_result = context.get("calculation_result", {})
        total_emissions_kg = calc_result.get(
            "total_emissions_kg_co2e", 0.0,
        )
        total_emissions_raw_kg = calc_result.get(
            "emissions_kg", 0.0,
        )

        # Try engine-based decomposition
        if self.refrigerant_database is not None:
            try:
                ref_type = context["input"].get(
                    "refrigerant_type", "",
                )
                components = self.refrigerant_database.get_blend_components(
                    ref_type,
                )
                if components:
                    decomposed = self._decompose_with_components(
                        components, total_emissions_raw_kg,
                        context.get("gwp_value", 0.0),
                    )
                    context["blend_decomposition"] = decomposed

                    return {
                        "stage": stage,
                        "success": True,
                        "is_blend": True,
                        "components": decomposed,
                        "component_count": len(decomposed),
                        "provenance_hash": _compute_hash({
                            "stage": stage,
                            "pipeline_id": context["pipeline_id"],
                            "components": len(decomposed),
                        }),
                    }
            except (AttributeError, TypeError, KeyError) as exc:
                logger.warning(
                    "Blend decomposition via engine failed: %s", exc,
                )

        # Stub decomposition using blend components from props
        blend_components = props.get("components", [])
        if blend_components:
            decomposed = self._decompose_with_components(
                blend_components, total_emissions_raw_kg,
                context.get("gwp_value", 0.0),
            )
        else:
            decomposed = [{
                "gas": props.get("refrigerant_type", "unknown"),
                "weight_fraction": 1.0,
                "emissions_kg_co2e": total_emissions_kg,
                "gwp": context.get("gwp_value", 0.0),
            }]

        context["blend_decomposition"] = decomposed

        return {
            "stage": stage,
            "success": True,
            "is_blend": True,
            "components": decomposed,
            "component_count": len(decomposed),
            "provenance_hash": _compute_hash({
                "stage": stage,
                "pipeline_id": context["pipeline_id"],
                "components": len(decomposed),
            }),
        }

    def _run_stage_quantify_uncertainty(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 6: Run Monte Carlo or analytical uncertainty.

        Quantifies the uncertainty in the emission calculation using
        Monte Carlo simulation or analytical error propagation.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result dictionary with uncertainty analysis.
        """
        stage = "QUANTIFY_UNCERTAINTY"
        calc_result = context.get("calculation_result", {})

        if self.uncertainty_quantifier is not None:
            try:
                iterations = 5000
                if self.config is not None:
                    iterations = getattr(
                        self.config, "monte_carlo_iterations",
                        5000,
                    )

                unc_result = self.uncertainty_quantifier.quantify(
                    calculation_result=calc_result,
                    iterations=iterations,
                )
                if hasattr(unc_result, "model_dump"):
                    unc_dict = unc_result.model_dump(mode="json")
                elif isinstance(unc_result, dict):
                    unc_dict = unc_result
                else:
                    unc_dict = {}

                context["uncertainty_result"] = unc_dict

                return {
                    "stage": stage,
                    "success": True,
                    "method": unc_dict.get("method", "monte_carlo"),
                    "mean_co2e_kg": unc_dict.get("mean_co2e_kg", 0.0),
                    "std_co2e_kg": unc_dict.get("std_co2e_kg", 0.0),
                    "p5_co2e_kg": unc_dict.get("p5_co2e_kg", 0.0),
                    "p95_co2e_kg": unc_dict.get("p95_co2e_kg", 0.0),
                    "confidence_interval_pct": unc_dict.get(
                        "confidence_interval_pct", 95.0,
                    ),
                    "iterations": unc_dict.get("iterations", iterations),
                    "data_quality_score": unc_dict.get(
                        "data_quality_score", 3,
                    ),
                    "provenance_hash": _compute_hash({
                        "stage": stage,
                        "pipeline_id": context["pipeline_id"],
                        "mean": unc_dict.get("mean_co2e_kg", 0.0),
                    }),
                }

            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(
                    "Uncertainty quantification failed: %s", exc,
                )

        # Stub uncertainty when engine unavailable
        total_emissions = calc_result.get(
            "total_emissions_kg_co2e", 0.0,
        )
        # Apply default 25% relative uncertainty (IPCC Tier 1)
        relative_unc = 0.25
        std_estimate = total_emissions * relative_unc

        unc_dict = {
            "method": "analytical_stub",
            "mean_co2e_kg": total_emissions,
            "std_co2e_kg": round(std_estimate, 6),
            "p5_co2e_kg": round(
                total_emissions - 1.645 * std_estimate, 6,
            ),
            "p95_co2e_kg": round(
                total_emissions + 1.645 * std_estimate, 6,
            ),
            "confidence_interval_pct": 90.0,
            "iterations": 0,
            "data_quality_score": 3,
            "relative_uncertainty_pct": relative_unc * 100,
        }
        context["uncertainty_result"] = unc_dict

        return {
            "stage": stage,
            "success": True,
            **unc_dict,
            "message": "UncertaintyQuantifierEngine not available; "
                       "using analytical stub",
            "provenance_hash": _compute_hash({
                "stage": stage,
                "pipeline_id": context["pipeline_id"],
                "mean": total_emissions,
            }),
        }

    def _run_stage_check_compliance(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 7: Check against applicable regulatory frameworks.

        Evaluates the calculation against relevant compliance frameworks
        including GHG Protocol, EPA 40 CFR Part 98, EU F-Gas Regulation,
        Kigali Amendment, and ISO 14064.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result dictionary with compliance check results.
        """
        stage = "CHECK_COMPLIANCE"
        calc_result = context.get("calculation_result", {})
        input_data = context["input"]
        ref_props = context.get("refrigerant_props", {})

        frameworks = input_data.get("regulatory_frameworks", [
            "GHG_PROTOCOL",
            "EPA_40CFR98_DD",
            "EU_FGAS_2024",
            "KIGALI_AMENDMENT",
            "ISO_14064",
        ])

        if self.compliance_tracker is not None:
            try:
                compliance_results = self.compliance_tracker.check(
                    calculation_result=calc_result,
                    refrigerant_props=ref_props,
                    frameworks=frameworks,
                )
                if isinstance(compliance_results, list):
                    records = compliance_results
                elif isinstance(compliance_results, dict):
                    records = compliance_results.get("records", [])
                else:
                    records = []

                serialized = []
                for rec in records:
                    if hasattr(rec, "model_dump"):
                        serialized.append(rec.model_dump(mode="json"))
                    elif isinstance(rec, dict):
                        serialized.append(rec)

                context["compliance_records"] = serialized

                compliant_count = sum(
                    1 for r in serialized
                    if r.get("compliant", False)
                )

                return {
                    "stage": stage,
                    "success": True,
                    "frameworks_checked": len(serialized),
                    "compliant_count": compliant_count,
                    "non_compliant_count": len(serialized) - compliant_count,
                    "overall_compliant": compliant_count == len(serialized),
                    "records": serialized,
                    "provenance_hash": _compute_hash({
                        "stage": stage,
                        "pipeline_id": context["pipeline_id"],
                        "compliant_count": compliant_count,
                    }),
                }

            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(
                    "Compliance check failed: %s", exc,
                )

        # Stub compliance check when engine unavailable
        stub_records = []
        for fw in frameworks:
            stub_records.append({
                "framework": fw,
                "compliant": True,
                "requirements_met": [],
                "requirements_gap": [],
                "message": "ComplianceTrackerEngine not available; "
                           "defaulting to compliant",
            })

        context["compliance_records"] = stub_records

        return {
            "stage": stage,
            "success": True,
            "frameworks_checked": len(stub_records),
            "compliant_count": len(stub_records),
            "non_compliant_count": 0,
            "overall_compliant": True,
            "records": stub_records,
            "message": "Stub compliance check; engine unavailable",
            "provenance_hash": _compute_hash({
                "stage": stage,
                "pipeline_id": context["pipeline_id"],
                "compliant_count": len(stub_records),
            }),
        }

    def _run_stage_generate_audit(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 8: Generate audit trail with provenance chain.

        Creates a comprehensive audit trail capturing every stage
        execution, input data, calculation results, and compliance
        outcomes with SHA-256 provenance hashing.

        Args:
            context: Pipeline context dictionary.

        Returns:
            Stage result dictionary with audit trail entries.
        """
        stage = "GENERATE_AUDIT"
        input_data = context["input"]
        calc_result = context.get("calculation_result", {})

        audit_entries: List[Dict[str, Any]] = []

        # Build provenance chain hash from all stage hashes
        chain_hashes = [
            entry.get("provenance_hash", "")
            for entry in context.get("provenance_chain", [])
            if entry.get("provenance_hash")
        ]
        chain_hash = _compute_hash(chain_hashes)

        # Entry 1: Input record
        audit_entries.append({
            "audit_id": _new_uuid(),
            "calculation_id": context["calculation_id"],
            "pipeline_id": context["pipeline_id"],
            "action": "input_recorded",
            "actor": "system",
            "details": {
                "refrigerant_type": input_data.get(
                    "refrigerant_type", "",
                ),
                "charge_kg": input_data.get("charge_kg", 0.0),
                "method": input_data.get("method", "equipment_based"),
                "gwp_source": input_data.get("gwp_source", "AR6"),
            },
            "provenance_hash": _compute_hash({
                "action": "input_recorded",
                "calculation_id": context["calculation_id"],
            }),
            "timestamp": _utcnow_iso(),
        })

        # Entry 2: Calculation result
        audit_entries.append({
            "audit_id": _new_uuid(),
            "calculation_id": context["calculation_id"],
            "pipeline_id": context["pipeline_id"],
            "action": "calculation_completed",
            "actor": "system",
            "details": {
                "total_emissions_kg_co2e": calc_result.get(
                    "total_emissions_kg_co2e", 0.0,
                ),
                "total_emissions_tco2e": calc_result.get(
                    "total_emissions_tco2e", 0.0,
                ),
                "gwp_value": context.get("gwp_value", 0.0),
                "leak_rate_pct": context.get("leak_rate", 0.0),
                "method": input_data.get("method", "equipment_based"),
            },
            "provenance_hash": _compute_hash({
                "action": "calculation_completed",
                "emissions": calc_result.get(
                    "total_emissions_kg_co2e", 0.0,
                ),
            }),
            "timestamp": _utcnow_iso(),
        })

        # Entry 3: Uncertainty (if available)
        unc_result = context.get("uncertainty_result")
        if unc_result:
            audit_entries.append({
                "audit_id": _new_uuid(),
                "calculation_id": context["calculation_id"],
                "pipeline_id": context["pipeline_id"],
                "action": "uncertainty_quantified",
                "actor": "system",
                "details": {
                    "method": unc_result.get("method", ""),
                    "mean_co2e_kg": unc_result.get("mean_co2e_kg", 0.0),
                    "std_co2e_kg": unc_result.get("std_co2e_kg", 0.0),
                    "confidence_interval_pct": unc_result.get(
                        "confidence_interval_pct", 0.0,
                    ),
                },
                "provenance_hash": _compute_hash({
                    "action": "uncertainty_quantified",
                    "mean": unc_result.get("mean_co2e_kg", 0.0),
                }),
                "timestamp": _utcnow_iso(),
            })

        # Entry 4: Compliance (if available)
        compliance_records = context.get("compliance_records", [])
        if compliance_records:
            audit_entries.append({
                "audit_id": _new_uuid(),
                "calculation_id": context["calculation_id"],
                "pipeline_id": context["pipeline_id"],
                "action": "compliance_checked",
                "actor": "system",
                "details": {
                    "frameworks_checked": len(compliance_records),
                    "compliant_count": sum(
                        1 for r in compliance_records
                        if r.get("compliant", False)
                    ),
                },
                "provenance_hash": _compute_hash({
                    "action": "compliance_checked",
                    "count": len(compliance_records),
                }),
                "timestamp": _utcnow_iso(),
            })

        # Entry 5: Pipeline provenance chain
        audit_entries.append({
            "audit_id": _new_uuid(),
            "calculation_id": context["calculation_id"],
            "pipeline_id": context["pipeline_id"],
            "action": "provenance_chain_sealed",
            "actor": "system",
            "details": {
                "chain_hash": chain_hash,
                "stage_count": len(context.get("provenance_chain", [])),
                "stages": [
                    e.get("stage", "")
                    for e in context.get("provenance_chain", [])
                ],
            },
            "provenance_hash": chain_hash,
            "timestamp": _utcnow_iso(),
        })

        context["audit_entries"] = audit_entries

        return {
            "stage": stage,
            "success": True,
            "entries_count": len(audit_entries),
            "chain_hash": chain_hash,
            "audit_entries": audit_entries,
            "provenance_hash": _compute_hash({
                "stage": stage,
                "pipeline_id": context["pipeline_id"],
                "entries_count": len(audit_entries),
                "chain_hash": chain_hash,
            }),
        }

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _normalise_input(self, input_data: Any) -> Dict[str, Any]:
        """Normalise input data to a plain dictionary.

        Args:
            input_data: CalculationInput (Pydantic model) or dict.

        Returns:
            Plain dictionary representation of the input.
        """
        if hasattr(input_data, "model_dump"):
            return input_data.model_dump(mode="json")
        if isinstance(input_data, dict):
            return dict(input_data)
        return {"raw": str(input_data)}

    def _calculate_fallback(
        self,
        input_data: Dict[str, Any],
        gwp_value: float,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deterministic fallback calculation when engine is unavailable.

        Implements all five calculation methods using only Python
        arithmetic (zero-hallucination).

        Args:
            input_data: Normalised input dictionary.
            gwp_value: GWP value for the refrigerant.
            context: Pipeline context for leak rate and properties.

        Returns:
            Dictionary with calculation result.
        """
        method = input_data.get("method", "equipment_based")
        charge_kg = float(input_data.get("charge_kg", 0.0))
        leak_rate_pct = context.get("leak_rate", 5.0)
        trace: List[str] = []

        if method == "equipment_based":
            return self._calc_equipment_based(
                charge_kg, leak_rate_pct, gwp_value,
                input_data, trace,
            )

        if method == "mass_balance":
            return self._calc_mass_balance(
                gwp_value, input_data, trace,
            )

        if method == "screening":
            return self._calc_screening(
                gwp_value, input_data, trace,
            )

        if method == "direct":
            return self._calc_direct(
                gwp_value, input_data, trace,
            )

        if method == "top_down":
            return self._calc_top_down(
                charge_kg, leak_rate_pct, gwp_value,
                input_data, trace,
            )

        # Unknown method - should not reach here after validation
        trace.append(f"Unknown method: {method}")
        return {
            "calculation_id": input_data.get("calculation_id", ""),
            "method": method,
            "status": "FAILED",
            "total_emissions_kg_co2e": 0.0,
            "total_emissions_tco2e": 0.0,
            "emissions_kg": 0.0,
            "gwp_value": gwp_value,
            "calculation_trace": trace,
            "error": f"Unknown calculation method: {method}",
        }

    def _calc_equipment_based(
        self,
        charge_kg: float,
        leak_rate_pct: float,
        gwp_value: float,
        input_data: Dict[str, Any],
        trace: List[str],
    ) -> Dict[str, Any]:
        """Equipment-based calculation: charge * leak_rate * GWP.

        Args:
            charge_kg: Equipment refrigerant charge in kg.
            leak_rate_pct: Annual leak rate percentage.
            gwp_value: Global warming potential value.
            input_data: Full input dictionary.
            trace: Mutable calculation trace list.

        Returns:
            Calculation result dictionary.
        """
        leak_rate_frac = leak_rate_pct / 100.0
        emissions_kg = charge_kg * leak_rate_frac
        emissions_co2e_kg = emissions_kg * gwp_value
        emissions_co2e_tonnes = emissions_co2e_kg / 1000.0

        trace.append(
            f"Equipment-based: {charge_kg} kg * "
            f"{leak_rate_pct}% leak rate = "
            f"{emissions_kg:.6f} kg refrigerant"
        )
        trace.append(
            f"CO2e: {emissions_kg:.6f} kg * "
            f"GWP {gwp_value} = "
            f"{emissions_co2e_kg:.6f} kg CO2e"
        )

        return self._build_calc_result(
            input_data, "equipment_based", emissions_kg,
            emissions_co2e_kg, emissions_co2e_tonnes,
            gwp_value, trace,
        )

    def _calc_mass_balance(
        self,
        gwp_value: float,
        input_data: Dict[str, Any],
        trace: List[str],
    ) -> Dict[str, Any]:
        """Mass-balance calculation.

        emissions = (inventory_start + purchases - recovery - inventory_end)

        Args:
            gwp_value: Global warming potential value.
            input_data: Full input dictionary.
            trace: Mutable calculation trace list.

        Returns:
            Calculation result dictionary.
        """
        mb = input_data.get("mass_balance_data", {})
        inventory_start = float(mb.get("inventory_start_kg", 0.0))
        purchases = float(mb.get("purchases_kg", 0.0))
        recovery = float(mb.get("recovery_kg", 0.0))
        inventory_end = float(mb.get("inventory_end_kg", 0.0))

        emissions_kg = (
            inventory_start + purchases - recovery - inventory_end
        )
        emissions_kg = max(emissions_kg, 0.0)
        emissions_co2e_kg = emissions_kg * gwp_value
        emissions_co2e_tonnes = emissions_co2e_kg / 1000.0

        trace.append(
            f"Mass balance: ({inventory_start} + {purchases} - "
            f"{recovery} - {inventory_end}) = "
            f"{emissions_kg:.6f} kg refrigerant"
        )
        trace.append(
            f"CO2e: {emissions_kg:.6f} kg * "
            f"GWP {gwp_value} = "
            f"{emissions_co2e_kg:.6f} kg CO2e"
        )

        return self._build_calc_result(
            input_data, "mass_balance", emissions_kg,
            emissions_co2e_kg, emissions_co2e_tonnes,
            gwp_value, trace,
        )

    def _calc_screening(
        self,
        gwp_value: float,
        input_data: Dict[str, Any],
        trace: List[str],
    ) -> Dict[str, Any]:
        """Screening calculation: activity_data * screening_factor * GWP.

        Args:
            gwp_value: Global warming potential value.
            input_data: Full input dictionary.
            trace: Mutable calculation trace list.

        Returns:
            Calculation result dictionary.
        """
        activity_data = float(
            input_data.get("activity_data", 0.0),
        )
        screening_factor = float(
            input_data.get("screening_factor", 0.01),
        )

        emissions_kg = activity_data * screening_factor
        emissions_co2e_kg = emissions_kg * gwp_value
        emissions_co2e_tonnes = emissions_co2e_kg / 1000.0

        trace.append(
            f"Screening: {activity_data} * "
            f"{screening_factor} = "
            f"{emissions_kg:.6f} kg refrigerant"
        )
        trace.append(
            f"CO2e: {emissions_kg:.6f} kg * "
            f"GWP {gwp_value} = "
            f"{emissions_co2e_kg:.6f} kg CO2e"
        )

        return self._build_calc_result(
            input_data, "screening", emissions_kg,
            emissions_co2e_kg, emissions_co2e_tonnes,
            gwp_value, trace,
        )

    def _calc_direct(
        self,
        gwp_value: float,
        input_data: Dict[str, Any],
        trace: List[str],
    ) -> Dict[str, Any]:
        """Direct measurement calculation: measured_emissions * GWP.

        Args:
            gwp_value: Global warming potential value.
            input_data: Full input dictionary.
            trace: Mutable calculation trace list.

        Returns:
            Calculation result dictionary.
        """
        measured_kg = float(
            input_data.get("measured_emissions_kg", 0.0),
        )

        emissions_kg = measured_kg
        emissions_co2e_kg = emissions_kg * gwp_value
        emissions_co2e_tonnes = emissions_co2e_kg / 1000.0

        trace.append(
            f"Direct: measured {measured_kg:.6f} kg refrigerant"
        )
        trace.append(
            f"CO2e: {emissions_kg:.6f} kg * "
            f"GWP {gwp_value} = "
            f"{emissions_co2e_kg:.6f} kg CO2e"
        )

        return self._build_calc_result(
            input_data, "direct", emissions_kg,
            emissions_co2e_kg, emissions_co2e_tonnes,
            gwp_value, trace,
        )

    def _calc_top_down(
        self,
        charge_kg: float,
        leak_rate_pct: float,
        gwp_value: float,
        input_data: Dict[str, Any],
        trace: List[str],
    ) -> Dict[str, Any]:
        """Top-down calculation: total_charge * leak_rate * GWP.

        Uses facility-level total charge and average leak rate.

        Args:
            charge_kg: Total facility refrigerant charge in kg.
            leak_rate_pct: Average annual leak rate percentage.
            gwp_value: Global warming potential value.
            input_data: Full input dictionary.
            trace: Mutable calculation trace list.

        Returns:
            Calculation result dictionary.
        """
        num_units = int(input_data.get("num_units", 1))
        total_charge = charge_kg * num_units
        leak_rate_frac = leak_rate_pct / 100.0

        emissions_kg = total_charge * leak_rate_frac
        emissions_co2e_kg = emissions_kg * gwp_value
        emissions_co2e_tonnes = emissions_co2e_kg / 1000.0

        trace.append(
            f"Top-down: {charge_kg} kg * {num_units} units = "
            f"{total_charge} kg total charge"
        )
        trace.append(
            f"Emissions: {total_charge} kg * "
            f"{leak_rate_pct}% = "
            f"{emissions_kg:.6f} kg refrigerant"
        )
        trace.append(
            f"CO2e: {emissions_kg:.6f} kg * "
            f"GWP {gwp_value} = "
            f"{emissions_co2e_kg:.6f} kg CO2e"
        )

        return self._build_calc_result(
            input_data, "top_down", emissions_kg,
            emissions_co2e_kg, emissions_co2e_tonnes,
            gwp_value, trace,
        )

    def _build_calc_result(
        self,
        input_data: Dict[str, Any],
        method: str,
        emissions_kg: float,
        emissions_co2e_kg: float,
        emissions_co2e_tonnes: float,
        gwp_value: float,
        trace: List[str],
    ) -> Dict[str, Any]:
        """Build a standardised calculation result dictionary.

        Args:
            input_data: Input dictionary.
            method: Calculation method name.
            emissions_kg: Raw refrigerant emissions in kg.
            emissions_co2e_kg: CO2e emissions in kg.
            emissions_co2e_tonnes: CO2e emissions in tonnes.
            gwp_value: GWP value used.
            trace: Calculation trace entries.

        Returns:
            Standardised result dictionary.
        """
        return {
            "calculation_id": input_data.get("calculation_id", ""),
            "refrigerant_type": input_data.get(
                "refrigerant_type", "",
            ),
            "charge_kg": float(input_data.get("charge_kg", 0.0)),
            "method": method,
            "status": "SUCCESS",
            "emissions_kg": round(emissions_kg, 6),
            "total_emissions_kg_co2e": round(emissions_co2e_kg, 6),
            "total_emissions_tco2e": round(emissions_co2e_tonnes, 9),
            "gwp_value": gwp_value,
            "gwp_source": input_data.get("gwp_source", "AR6"),
            "facility_id": input_data.get("facility_id", ""),
            "equipment_type": input_data.get("equipment_type", ""),
            "equipment_id": input_data.get("equipment_id", ""),
            "calculation_trace": trace,
            "provenance_hash": _compute_hash({
                "calculation_id": input_data.get(
                    "calculation_id", "",
                ),
                "emissions_co2e_kg": emissions_co2e_kg,
                "method": method,
            }),
            "calculated_at": _utcnow_iso(),
        }

    def _extract_gwp(
        self,
        props: Dict[str, Any],
        gwp_source: str,
    ) -> float:
        """Extract GWP value from refrigerant properties.

        Tries multiple key formats to find the GWP value for the
        specified source.

        Args:
            props: Refrigerant properties dictionary.
            gwp_source: GWP source (AR4, AR5, AR6, AR6_20yr).

        Returns:
            GWP value as float. Returns 0.0 if not found.
        """
        # Direct key match
        gwp_key_variants = [
            f"gwp_{gwp_source.lower()}",
            f"gwp_{gwp_source.lower()}_100yr",
            f"gwp_{gwp_source}",
            "gwp_100yr",
            "gwp",
        ]
        for key in gwp_key_variants:
            if key in props:
                return float(props[key])

        # Check nested gwp_values dict
        gwp_values = props.get("gwp_values", {})
        if isinstance(gwp_values, dict):
            for key in gwp_key_variants:
                if key in gwp_values:
                    return float(gwp_values[key])
            if gwp_source in gwp_values:
                return float(gwp_values[gwp_source])

        return 0.0

    def _get_stub_gwp(
        self,
        ref_type: str,
        gwp_source: str,
    ) -> float:
        """Get a stub GWP value for common refrigerant types.

        Used when the refrigerant database engine is unavailable.
        Values are from IPCC AR6 100-year GWPs.

        Args:
            ref_type: Refrigerant type identifier.
            gwp_source: GWP source for reference.

        Returns:
            Stub GWP value.
        """
        stub_gwps = {
            "R_32": 771.0,
            "R_125": 3740.0,
            "R_134A": 1530.0,
            "R_143A": 5810.0,
            "R_152A": 164.0,
            "R_227EA": 3600.0,
            "R_245FA": 962.0,
            "R_404A": 4728.0,
            "R_407A": 2434.0,
            "R_407C": 1908.0,
            "R_410A": 2256.0,
            "R_507A": 4427.0,
            "R_508B": 13860.0,
            "R_1234YF": 0.501,
            "R_1234ZE": 1.37,
            "SF6": 25200.0,
            "NF3": 17400.0,
            "CF4": 7380.0,
            "C2F6": 12400.0,
            "HFC_23": 14600.0,
            "R_22": 1960.0,
            "R_11": 6230.0,
            "R_12": 12500.0,
            "R_290": 0.072,
            "R_600A": 0.072,
            "R_717": 0.0,
            "R_744": 1.0,
        }
        # Normalise key
        key = ref_type.upper().replace("-", "_")
        return stub_gwps.get(key, 0.0)

    def _decompose_with_components(
        self,
        components: Any,
        total_raw_emissions_kg: float,
        blend_gwp: float,
    ) -> List[Dict[str, Any]]:
        """Decompose blend emissions into per-component gases.

        Args:
            components: List of blend component dicts or Pydantic models.
            total_raw_emissions_kg: Total raw refrigerant emissions in kg.
            blend_gwp: Blended GWP value for normalisation.

        Returns:
            List of per-component emission dictionaries.
        """
        decomposed: List[Dict[str, Any]] = []

        for comp in components:
            if hasattr(comp, "model_dump"):
                comp_dict = comp.model_dump(mode="json")
            elif isinstance(comp, dict):
                comp_dict = comp
            else:
                continue

            gas = comp_dict.get(
                "gas", comp_dict.get("refrigerant_type", "unknown"),
            )
            weight_fraction = float(
                comp_dict.get("weight_fraction", 0.0),
            )
            component_gwp = float(comp_dict.get("gwp", 0.0))

            # Per-component emissions
            component_emissions_kg = (
                total_raw_emissions_kg * weight_fraction
            )
            component_co2e_kg = component_emissions_kg * component_gwp

            decomposed.append({
                "gas": gas,
                "weight_fraction": weight_fraction,
                "emissions_kg": round(component_emissions_kg, 6),
                "gwp": component_gwp,
                "emissions_kg_co2e": round(component_co2e_kg, 6),
            })

        return decomposed

    def _build_final_result(
        self,
        context: Dict[str, Any],
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        """Build the final result dictionary from pipeline context.

        Args:
            context: Pipeline context dictionary.
            elapsed_ms: Total pipeline execution time in ms.

        Returns:
            Final result dictionary for the pipeline response.
        """
        calc_result = context.get("calculation_result", {})
        input_data = context["input"]

        result = {
            "calculation_id": context["calculation_id"],
            "refrigerant_type": input_data.get(
                "refrigerant_type", "",
            ),
            "charge_kg": float(input_data.get("charge_kg", 0.0)),
            "method": input_data.get("method", "equipment_based"),
            "status": calc_result.get("status", "SUCCESS"),
            "gwp_value": context.get("gwp_value", 0.0),
            "gwp_source": input_data.get("gwp_source", "AR6"),
            "leak_rate_pct": context.get("leak_rate", 0.0),
            "emissions_kg": calc_result.get("emissions_kg", 0.0),
            "total_emissions_kg_co2e": calc_result.get(
                "total_emissions_kg_co2e", 0.0,
            ),
            "total_emissions_tco2e": calc_result.get(
                "total_emissions_tco2e", 0.0,
            ),
            "facility_id": input_data.get("facility_id", ""),
            "equipment_type": input_data.get("equipment_type", ""),
            "equipment_id": input_data.get("equipment_id", ""),
            "blend_decomposition": context.get(
                "blend_decomposition",
            ),
            "uncertainty": context.get("uncertainty_result"),
            "compliance": context.get("compliance_records"),
            "calculation_trace": calc_result.get(
                "calculation_trace", [],
            ),
            "provenance_hash": calc_result.get(
                "provenance_hash", "",
            ),
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": _utcnow_iso(),
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
            elapsed_ms: Execution duration in ms.
            stages_completed: Number of stages completed.
            calculation_id: Calculation identifier.
        """
        with self._lock:
            self._total_runs += 1
            self._total_duration_ms += elapsed_ms
            self._last_run_at = _utcnow_iso()

            if success:
                self._successful_runs += 1
            else:
                self._failed_runs += 1

            self._pipeline_results[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "calculation_id": calculation_id,
                "success": success,
                "stages_completed": stages_completed,
                "total_duration_ms": round(elapsed_ms, 3),
                "timestamp": self._last_run_at,
            }

    def _save_checkpoint(
        self,
        batch_id: str,
        processed_count: int,
        results: List[Dict[str, Any]],
    ) -> None:
        """Save a batch pipeline checkpoint for recovery.

        Args:
            batch_id: Batch identifier.
            processed_count: Number of inputs processed so far.
            results: Results collected so far.
        """
        with self._lock:
            self._checkpoints[batch_id] = {
                "batch_id": batch_id,
                "processed_count": processed_count,
                "results_count": len(results),
                "timestamp": _utcnow_iso(),
            }

        logger.debug(
            "Batch %s checkpoint saved: %d processed",
            batch_id, processed_count,
        )

    def _strip_internal_keys(
        self,
        stage_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Remove internal keys from stage results for external consumption.

        Keys like ``audit_entries``, ``records``, and ``components`` carry
        full objects that are returned via context in the final result.

        Args:
            stage_results: Raw stage result dictionaries.

        Returns:
            Cleaned stage result dictionaries.
        """
        internal_keys = {
            "audit_entries",
            "records",
            "components",
        }
        cleaned: List[Dict[str, Any]] = []
        for stage in stage_results:
            cleaned.append({
                k: v for k, v in stage.items()
                if k not in internal_keys
            })
        return cleaned


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    "RefrigerantPipelineEngine",
    "PIPELINE_STAGES",
    "SUPPORTED_METHODS",
]
