# -*- coding: utf-8 -*-
"""
ValidationPipelineEngine - AGENT-DATA-019: Validation Rule Engine

Engine 7 of 7 -- End-to-end pipeline orchestration for the Validation Rule
Engine.

This module implements the ValidationPipelineEngine, which orchestrates
all six upstream engines (RuleRegistryEngine, RuleComposerEngine,
RuleEvaluatorEngine, ConflictDetectorEngine, RulePackEngine,
ValidationReporterEngine) into a coherent pipeline workflow:

    Stage 1 REGISTER           - Apply a rule pack or register individual
                                  rules into the rule registry
    Stage 2 COMPOSE            - Build or validate rule set composition
                                  (compound rules, dependency ordering)
    Stage 3 EVALUATE           - Execute rules against the supplied data
                                  (single-dataset or batch evaluation)
    Stage 4 DETECT_CONFLICTS   - Run conflict detection across the active
                                  rule set for contradictions, overlaps,
                                  and redundancies
    Stage 5 REPORT             - Generate a validation report (summary,
                                  detailed, compliance, trend, executive)
    Stage 6 AUDIT              - Record audit trail entries with SHA-256
                                  provenance hashes for the pipeline run
    Stage 7 NOTIFY             - Generate a notification payload for
                                  alerting subsystems (OBS-004 bridge,
                                  Slack, PagerDuty, email)

Design Principles:
    - Zero-hallucination: all pass rates, failure counts, conflict counts,
      and scoring verdicts come from deterministic engine calls -- never
      from LLM inference.  Every numeric aggregation uses pure Python
      arithmetic.
    - Provenance: every pipeline run produces a SHA-256 provenance chain
      anchored to the shared ProvenanceTracker genesis hash.
    - Thread-safety: a single threading.Lock guards the pipeline run store,
      notification history, and aggregate statistics so concurrent callers
      never corrupt state.
    - Graceful degradation: each upstream engine is imported with a
      try/except guard; missing engines produce clear error messages rather
      than cryptic AttributeErrors at runtime.
    - Auditability: the pipeline result dictionary captures every stage
      outcome, timing, and error detail for compliance reporting.

Example:
    >>> from greenlang.validation_rule_engine.validation_pipeline import (
    ...     ValidationPipelineEngine,
    ... )
    >>> engine = ValidationPipelineEngine()
    >>> result = engine.run_pipeline(
    ...     data=[{"emissions": 120.5, "unit": "tCO2e"}],
    ...     rule_set_id="ghg-completeness",
    ... )
    >>> assert result["pipeline_id"] is not None
    >>> assert len(result["stages_completed"]) >= 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful imports for provenance, config, metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.provenance import ProvenanceTracker
    _PROVENANCE_AVAILABLE = True
except Exception:  # noqa: BLE001
    ProvenanceTracker = None  # type: ignore[assignment, misc]
    _PROVENANCE_AVAILABLE = False
    logger.warning(
        "ProvenanceTracker not available; provenance import failed."
    )

try:
    from greenlang.validation_rule_engine.config import get_config
    _CONFIG_AVAILABLE = True
except Exception:  # noqa: BLE001
    get_config = None  # type: ignore[assignment, misc]
    _CONFIG_AVAILABLE = False

try:
    from greenlang.validation_rule_engine.metrics import (
        observe_processing_duration,
        observe_evaluation_duration,
        PROMETHEUS_AVAILABLE,
        record_rule_registered,
        record_rule_set_created,
        record_evaluation,
        record_evaluation_failure,
        record_conflict_detected,
        record_report_generated,
        set_active_rules,
        set_active_rule_sets,
        set_pass_rate,
    )
    _METRICS_AVAILABLE = True
except Exception:  # noqa: BLE001
    observe_processing_duration = None  # type: ignore[assignment]
    observe_evaluation_duration = None  # type: ignore[assignment]
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    record_rule_registered = None  # type: ignore[assignment]
    record_rule_set_created = None  # type: ignore[assignment]
    record_evaluation = None  # type: ignore[assignment]
    record_evaluation_failure = None  # type: ignore[assignment]
    record_conflict_detected = None  # type: ignore[assignment]
    record_report_generated = None  # type: ignore[assignment]
    set_active_rules = None  # type: ignore[assignment]
    set_active_rule_sets = None  # type: ignore[assignment]
    set_pass_rate = None  # type: ignore[assignment]
    _METRICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Graceful imports for the 6 upstream engines
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.rule_registry import (
        RuleRegistryEngine,
    )
    _RULE_REGISTRY_AVAILABLE = True
except Exception:  # noqa: BLE001
    RuleRegistryEngine = None  # type: ignore[assignment, misc]
    _RULE_REGISTRY_AVAILABLE = False
    logger.warning(
        "RuleRegistryEngine not available; rule_registry import failed. "
        "Rule registration will use stub fallback."
    )

try:
    from greenlang.validation_rule_engine.rule_composer import (
        RuleComposerEngine,
    )
    _RULE_COMPOSER_AVAILABLE = True
except Exception:  # noqa: BLE001
    RuleComposerEngine = None  # type: ignore[assignment, misc]
    _RULE_COMPOSER_AVAILABLE = False
    logger.warning(
        "RuleComposerEngine not available; rule_composer import failed. "
        "Rule composition will use stub fallback."
    )

try:
    from greenlang.validation_rule_engine.rule_evaluator import (
        RuleEvaluatorEngine,
    )
    _RULE_EVALUATOR_AVAILABLE = True
except Exception:  # noqa: BLE001
    RuleEvaluatorEngine = None  # type: ignore[assignment, misc]
    _RULE_EVALUATOR_AVAILABLE = False
    logger.warning(
        "RuleEvaluatorEngine not available; rule_evaluator import failed. "
        "Rule evaluation will use stub fallback."
    )

try:
    from greenlang.validation_rule_engine.conflict_detector import (
        ConflictDetectorEngine,
    )
    _CONFLICT_DETECTOR_AVAILABLE = True
except Exception:  # noqa: BLE001
    ConflictDetectorEngine = None  # type: ignore[assignment, misc]
    _CONFLICT_DETECTOR_AVAILABLE = False
    logger.warning(
        "ConflictDetectorEngine not available; conflict_detector import "
        "failed. Conflict detection will use stub fallback."
    )

try:
    from greenlang.validation_rule_engine.rule_pack import RulePackEngine
    _RULE_PACK_AVAILABLE = True
except Exception:  # noqa: BLE001
    RulePackEngine = None  # type: ignore[assignment, misc]
    _RULE_PACK_AVAILABLE = False
    logger.warning(
        "RulePackEngine not available; rule_pack import failed. "
        "Regulatory rule packs will use stub fallback."
    )

try:
    from greenlang.validation_rule_engine.validation_reporter import (
        ValidationReporterEngine,
    )
    _VALIDATION_REPORTER_AVAILABLE = True
except Exception:  # noqa: BLE001
    ValidationReporterEngine = None  # type: ignore[assignment, misc]
    _VALIDATION_REPORTER_AVAILABLE = False
    logger.warning(
        "ValidationReporterEngine not available; validation_reporter "
        "import failed. Report generation will use stub fallback."
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    "register",
    "compose",
    "evaluate",
    "detect_conflicts",
    "report",
    "audit",
    "notify",
]

_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_PARTIAL = "partial"

_NOTIFICATION_SEVERITY_MAP = {
    "critical": "P1",
    "high": "P2",
    "medium": "P3",
    "low": "P4",
    "info": "P5",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string (microseconds zeroed)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_id(prefix: str = "pipe") -> str:
    """Generate a short unique identifier with a descriptive prefix.

    Args:
        prefix: Short string prepended to the UUID hex fragment.

    Returns:
        Identifier string of the form ``"{prefix}-{12-hex-chars}"``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds from a ``time.monotonic()`` start.

    Args:
        start: ``time.monotonic()`` value captured before the operation.

    Returns:
        Elapsed duration in milliseconds, rounded to two decimal places.
    """
    return round((time.monotonic() - start) * 1000.0, 2)


def _sha256(payload: Any) -> str:
    """Compute a SHA-256 hex digest for any JSON-serialisable payload.

    Serialises ``payload`` to canonical JSON (sorted keys, ``str``
    default for non-serialisable types) before hashing so that
    equivalent structures always produce the same digest.

    Args:
        payload: Any JSON-serialisable object or primitive.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).
    """
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _safe_observe_processing(operation: str, seconds: float) -> None:
    """Observe processing duration with graceful fallback.

    Args:
        operation: Operation type label for the metric.
        seconds: Duration in seconds.
    """
    if _METRICS_AVAILABLE and observe_processing_duration is not None:
        try:
            observe_processing_duration(operation, seconds)
        except Exception:  # noqa: BLE001
            pass


def _safe_observe_evaluation(operation: str, seconds: float) -> None:
    """Observe evaluation duration with graceful fallback.

    Args:
        operation: Operation type label for the metric.
        seconds: Duration in seconds.
    """
    if _METRICS_AVAILABLE and observe_evaluation_duration is not None:
        try:
            observe_evaluation_duration(operation, seconds)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# ValidationPipelineEngine
# ---------------------------------------------------------------------------


class ValidationPipelineEngine:
    """End-to-end pipeline orchestrator for the GreenLang Validation Rule Engine.

    Coordinates all six upstream engines through a deterministic pipeline
    workflow: register -> compose -> evaluate -> detect_conflicts -> report
    -> audit -> notify.  Every stage outcome is captured in the pipeline
    result dictionary for compliance reporting and auditability.

    Key design decisions:

    - **Zero-hallucination**: pass rates, failure counts, conflict counts,
      and all scoring verdicts come exclusively from upstream engine calls.
      No LLM inference is used in any stage.
    - **Provenance**: every pipeline run appends chain-hashed entries to the
      shared ProvenanceTracker.
    - **Thread-safety**: ``self._lock`` serialises writes to the pipeline
      run store, notification history, and aggregate statistics while
      individual engine calls remain stateless.
    - **Graceful degradation**: missing engines trigger stub behaviour that
      surfaces a clear ``"engine_unavailable"`` error rather than an
      AttributeError deep inside stage logic.

    Attributes:
        _registry: Engine 1 - RuleRegistryEngine (or None).
        _composer: Engine 2 - RuleComposerEngine (or None).
        _evaluator: Engine 3 - RuleEvaluatorEngine (or None).
        _conflict_detector: Engine 4 - ConflictDetectorEngine (or None).
        _rule_pack: Engine 5 - RulePackEngine (or None).
        _reporter: Engine 6 - ValidationReporterEngine (or None).
        _provenance: SHA-256 chain-hashing provenance tracker.
        _pipeline_runs: Mapping of pipeline_id to pipeline result dict.
        _notifications: List of notification payloads generated by the
            notify stage.
        _lock: Mutex protecting concurrent writes to shared state.

    Example:
        >>> engine = ValidationPipelineEngine()
        >>> result = engine.run_pipeline(
        ...     data=[{"field": "value"}],
        ...     rule_set_id="test-set",
        ... )
        >>> print(result["pipeline_id"])
        pipe-abc123def456
    """

    def __init__(
        self,
        registry: Any = None,
        composer: Any = None,
        evaluator: Any = None,
        conflict_detector: Any = None,
        rule_pack: Any = None,
        reporter: Any = None,
        provenance: Any = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialise the pipeline engine and all six upstream engines.

        Each upstream engine can be injected via constructor parameters for
        testing.  When ``None`` is passed (the default), the engine is
        instantiated from its module if available, otherwise the attribute
        is set to ``None`` and the corresponding stage will return an
        ``"engine_unavailable"`` error.

        The ProvenanceTracker is shared across all engines to maintain a
        single provenance chain for the entire validation rule engine agent.

        Args:
            registry: Optional pre-built RuleRegistryEngine instance.
            composer: Optional pre-built RuleComposerEngine instance.
            evaluator: Optional pre-built RuleEvaluatorEngine instance.
            conflict_detector: Optional pre-built ConflictDetectorEngine
                instance.
            rule_pack: Optional pre-built RulePackEngine instance.
            reporter: Optional pre-built ValidationReporterEngine instance.
            provenance: Optional pre-built ProvenanceTracker instance.
            genesis_hash: Optional genesis hash for provenance tracker
                creation when no ``provenance`` is given.
        """
        # Provenance tracker -- shared across all engines
        if provenance is not None:
            self._provenance = provenance
        elif genesis_hash is not None and _PROVENANCE_AVAILABLE and ProvenanceTracker is not None:
            self._provenance = ProvenanceTracker(genesis_hash=genesis_hash)
        elif _PROVENANCE_AVAILABLE and ProvenanceTracker is not None:
            self._provenance = ProvenanceTracker()
        else:
            self._provenance = None  # type: ignore[assignment]

        # Engine 1 -- Rule Registry
        # Constructor: RuleRegistryEngine(provenance=None)
        if registry is not None:
            self._registry = registry
        elif _RULE_REGISTRY_AVAILABLE and RuleRegistryEngine is not None:
            self._registry = RuleRegistryEngine(provenance=self._provenance)
        else:
            self._registry = None

        # Engine 2 -- Rule Composer
        # Constructor: RuleComposerEngine(registry, provenance=None, ...)
        if composer is not None:
            self._composer = composer
        elif (
            _RULE_COMPOSER_AVAILABLE
            and RuleComposerEngine is not None
            and self._registry is not None
        ):
            self._composer = RuleComposerEngine(
                registry=self._registry,
                provenance=self._provenance,
            )
        else:
            self._composer = None

        # Engine 3 -- Rule Evaluator
        # Constructor: RuleEvaluatorEngine(registry, composer=None, provenance=None)
        if evaluator is not None:
            self._evaluator = evaluator
        elif _RULE_EVALUATOR_AVAILABLE and RuleEvaluatorEngine is not None:
            self._evaluator = RuleEvaluatorEngine(
                registry=self._registry,
                composer=self._composer,
                provenance=self._provenance,
            )
        else:
            self._evaluator = None

        # Engine 4 -- Conflict Detector
        # Constructor: ConflictDetectorEngine(registry, provenance=None)
        # Requires registry to be non-None
        if conflict_detector is not None:
            self._conflict_detector = conflict_detector
        elif (
            _CONFLICT_DETECTOR_AVAILABLE
            and ConflictDetectorEngine is not None
            and self._registry is not None
        ):
            self._conflict_detector = ConflictDetectorEngine(
                registry=self._registry,
                provenance=self._provenance,
            )
        else:
            self._conflict_detector = None

        # Engine 5 -- Rule Pack
        # Constructor: RulePackEngine(provenance=None) -- module may not exist yet
        if rule_pack is not None:
            self._rule_pack = rule_pack
        elif _RULE_PACK_AVAILABLE and RulePackEngine is not None:
            self._rule_pack = RulePackEngine(self._provenance)
        else:
            self._rule_pack = None

        # Engine 6 -- Validation Reporter
        # Constructor: ValidationReporterEngine(provenance=None)
        if reporter is not None:
            self._reporter = reporter
        elif (
            _VALIDATION_REPORTER_AVAILABLE
            and ValidationReporterEngine is not None
        ):
            self._reporter = ValidationReporterEngine(
                provenance=self._provenance,
            )
        else:
            self._reporter = None

        # Pipeline state
        self._pipeline_runs: Dict[str, Dict[str, Any]] = {}
        self._notifications: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        logger.info(
            "ValidationPipelineEngine initialised: "
            "registry=%s composer=%s evaluator=%s conflict_detector=%s "
            "rule_pack=%s reporter=%s provenance=%s",
            "ok" if self._registry else "UNAVAILABLE",
            "ok" if self._composer else "UNAVAILABLE",
            "ok" if self._evaluator else "UNAVAILABLE",
            "ok" if self._conflict_detector else "UNAVAILABLE",
            "ok" if self._rule_pack else "UNAVAILABLE",
            "ok" if self._reporter else "UNAVAILABLE",
            "ok" if self._provenance else "UNAVAILABLE",
        )

    # ------------------------------------------------------------------
    # Properties exposing sub-engines (read-only)
    # ------------------------------------------------------------------

    @property
    def registry(self) -> Any:
        """Return the RuleRegistryEngine instance, or None if unavailable."""
        return self._registry

    @property
    def composer(self) -> Any:
        """Return the RuleComposerEngine instance, or None if unavailable."""
        return self._composer

    @property
    def evaluator(self) -> Any:
        """Return the RuleEvaluatorEngine instance, or None if unavailable."""
        return self._evaluator

    @property
    def conflict_detector(self) -> Any:
        """Return the ConflictDetectorEngine instance, or None if unavailable."""
        return self._conflict_detector

    @property
    def rule_pack(self) -> Any:
        """Return the RulePackEngine instance, or None if unavailable."""
        return self._rule_pack

    @property
    def reporter(self) -> Any:
        """Return the ValidationReporterEngine instance, or None if unavailable."""
        return self._reporter

    @property
    def provenance(self) -> Any:
        """Return the shared ProvenanceTracker instance, or None."""
        return self._provenance

    # ------------------------------------------------------------------
    # Public API -- primary pipeline entry point
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        rule_set_id: Optional[str] = None,
        pack_name: Optional[str] = None,
        stages: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the complete validation pipeline end-to-end.

        Executes stages sequentially: register -> compose -> evaluate ->
        detect_conflicts -> report -> audit -> notify.  Individual stages
        can be limited via the ``stages`` parameter; when ``None`` is
        passed all seven stages execute.

        Args:
            data: Optional list of data records (dictionaries) to validate.
                Each record represents a row of data to be checked against
                the active rule set.  When ``None``, the evaluate stage
                is skipped.
            rule_set_id: Optional rule set identifier to evaluate against.
                When provided, the compose stage validates the rule set
                exists and the evaluate stage uses it.
            pack_name: Optional regulatory rule pack name to apply during
                the register stage (e.g. ``"ghg_protocol"``,
                ``"csrd_esrs"``, ``"eudr"``, ``"soc2"``).
            stages: Optional list of stage names to execute.  When
                ``None``, all seven stages run.  Unrecognised stage names
                are silently ignored.  Valid values: ``"register"``,
                ``"compose"``, ``"evaluate"``, ``"detect_conflicts"``,
                ``"report"``, ``"audit"``, ``"notify"``.
            parameters: Optional dictionary of additional parameters
                forwarded to individual stages.  Supported keys include:
                ``"report_type"`` (str), ``"report_format"`` (str),
                ``"pass_threshold"`` (float), ``"warn_threshold"`` (float),
                ``"short_circuit"`` (bool), ``"notification_channels"``
                (list of str).

        Returns:
            Pipeline result dictionary with keys:
            ``pipeline_id``, ``stages_completed``, ``stages_skipped``,
            ``results``, ``evaluation_summary``, ``conflicts``,
            ``report_id``, ``duration_ms``, ``stage_timings``,
            ``errors``, ``started_at``, ``completed_at``,
            ``provenance_hash``, ``status``.

        Example:
            >>> engine = ValidationPipelineEngine()
            >>> result = engine.run_pipeline(
            ...     data=[{"emissions": 120.5}],
            ...     rule_set_id="ghg-completeness",
            ... )
            >>> assert "pipeline_id" in result
        """
        pipeline_id = _new_id("pipe")
        started_at = _utcnow_iso()
        pipeline_start = time.monotonic()
        params = parameters if parameters is not None else {}

        # Determine which stages to execute
        active_stages = set(stages) if stages is not None else set(PIPELINE_STAGES)

        logger.info(
            "Pipeline %s starting: rule_set_id=%s pack_name=%s "
            "stages=%s data_rows=%d",
            pipeline_id,
            rule_set_id,
            pack_name,
            sorted(active_stages),
            len(data) if data else 0,
        )

        result: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "stages_completed": [],
            "stages_skipped": [],
            "results": {},
            "evaluation_summary": None,
            "conflicts": None,
            "report_id": None,
            "duration_ms": 0.0,
            "stage_timings": {},
            "errors": [],
            "started_at": started_at,
            "completed_at": None,
            "provenance_hash": None,
            "status": _STATUS_COMPLETED,
        }

        try:
            # Stage 1: REGISTER
            result = self._execute_register_stage(
                result, pipeline_id, active_stages, pack_name, params,
            )

            # Stage 2: COMPOSE
            result = self._execute_compose_stage(
                result, pipeline_id, active_stages, rule_set_id, params,
            )

            # Stage 3: EVALUATE
            result = self._execute_evaluate_stage(
                result, pipeline_id, active_stages, data, rule_set_id,
                params,
            )

            # Stage 4: DETECT_CONFLICTS
            result = self._execute_detect_conflicts_stage(
                result, pipeline_id, active_stages, rule_set_id, params,
            )

            # Stage 5: REPORT
            result = self._execute_report_stage(
                result, pipeline_id, active_stages, params,
            )

            # Stage 6: AUDIT
            result = self._execute_audit_stage(
                result, pipeline_id, active_stages, params,
            )

            # Stage 7: NOTIFY
            result = self._execute_notify_stage(
                result, pipeline_id, active_stages, params,
            )

        except Exception as exc:
            logger.error(
                "Pipeline %s failed with unexpected error: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )
            result["errors"].append(f"Pipeline error: {str(exc)}")
            result["status"] = _STATUS_FAILED

        # Finalise
        elapsed = _elapsed_ms(pipeline_start)
        result["duration_ms"] = elapsed
        result["completed_at"] = _utcnow_iso()

        if result["errors"] and result["status"] != _STATUS_FAILED:
            result["status"] = _STATUS_PARTIAL

        # Record provenance for the pipeline run
        provenance_hash = self._record_pipeline_provenance(pipeline_id, result)
        result["provenance_hash"] = provenance_hash

        # Observe total pipeline duration
        _safe_observe_processing("pipeline_run", elapsed / 1000.0)
        _safe_observe_evaluation("full_pipeline", elapsed / 1000.0)

        # Update gauges
        self._update_gauges()

        # Store the pipeline run
        with self._lock:
            self._pipeline_runs[pipeline_id] = result

        logger.info(
            "Pipeline %s completed: status=%s stages_completed=%d "
            "errors=%d duration_ms=%.2f",
            pipeline_id,
            result["status"],
            len(result["stages_completed"]),
            len(result["errors"]),
            elapsed,
        )

        return result

    # ------------------------------------------------------------------
    # Public API -- batch pipeline
    # ------------------------------------------------------------------

    def run_batch_pipeline(
        self,
        datasets: List[Dict[str, Any]],
        rule_set_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the validation pipeline across multiple datasets.

        Each dataset entry should contain a ``"dataset_id"`` key and a
        ``"data"`` key holding the list of record dictionaries to
        validate.  The pipeline is run once per dataset and an aggregated
        summary is produced at the end.

        Args:
            datasets: List of dataset dictionaries.  Each must contain at
                minimum a ``"data"`` key with a list of record dicts.
                An optional ``"dataset_id"`` key provides a human-readable
                label for the dataset; when omitted, a UUID is generated.
            rule_set_id: Rule set identifier to evaluate against for
                every dataset in the batch.
            parameters: Optional dictionary of additional parameters
                forwarded to each individual pipeline run.  See
                :meth:`run_pipeline` for supported keys.

        Returns:
            Batch result dictionary with keys:
            ``batch_id``, ``total_datasets``, ``succeeded``, ``failed``,
            ``dataset_results``, ``summary``, ``duration_ms``,
            ``started_at``, ``completed_at``, ``provenance_hash``,
            ``status``.

        Example:
            >>> batch_result = engine.run_batch_pipeline(
            ...     datasets=[
            ...         {"dataset_id": "ds1", "data": [{"x": 1}]},
            ...         {"dataset_id": "ds2", "data": [{"x": 2}]},
            ...     ],
            ...     rule_set_id="quality-checks",
            ... )
            >>> assert batch_result["total_datasets"] == 2
        """
        batch_id = _new_id("batch")
        started_at = _utcnow_iso()
        batch_start = time.monotonic()
        params = parameters if parameters is not None else {}

        logger.info(
            "Batch pipeline %s starting: rule_set_id=%s datasets=%d",
            batch_id,
            rule_set_id,
            len(datasets),
        )

        dataset_results: List[Dict[str, Any]] = []
        succeeded = 0
        failed = 0
        total_pass_rate = 0.0
        total_evaluations = 0
        total_failures = 0
        total_conflicts = 0
        all_errors: List[str] = []

        for ds_entry in datasets:
            dataset_id = ds_entry.get("dataset_id", _new_id("ds"))
            data = ds_entry.get("data", [])

            try:
                run_result = self.run_pipeline(
                    data=data,
                    rule_set_id=rule_set_id,
                    parameters=params,
                )

                ds_summary = {
                    "dataset_id": dataset_id,
                    "pipeline_id": run_result.get("pipeline_id"),
                    "status": run_result.get("status"),
                    "duration_ms": run_result.get("duration_ms", 0.0),
                    "evaluation_summary": run_result.get("evaluation_summary"),
                    "conflicts": run_result.get("conflicts"),
                    "errors": run_result.get("errors", []),
                }
                dataset_results.append(ds_summary)

                if run_result.get("status") == _STATUS_FAILED:
                    failed += 1
                    all_errors.extend(run_result.get("errors", []))
                else:
                    succeeded += 1

                # Aggregate evaluation metrics
                eval_summary = run_result.get("evaluation_summary")
                if isinstance(eval_summary, dict):
                    pr = eval_summary.get("pass_rate")
                    if isinstance(pr, (int, float)):
                        total_pass_rate += float(pr)
                        total_evaluations += 1
                    fc = eval_summary.get("fail_count", 0)
                    if isinstance(fc, int):
                        total_failures += fc

                # Aggregate conflict counts
                conflicts = run_result.get("conflicts")
                if isinstance(conflicts, dict):
                    cc = conflicts.get("conflict_count", 0)
                    if isinstance(cc, int):
                        total_conflicts += cc

            except Exception as exc:
                failed += 1
                error_msg = (
                    f"Dataset '{dataset_id}' pipeline failed: {str(exc)}"
                )
                all_errors.append(error_msg)
                dataset_results.append({
                    "dataset_id": dataset_id,
                    "pipeline_id": None,
                    "status": _STATUS_FAILED,
                    "duration_ms": 0.0,
                    "evaluation_summary": None,
                    "conflicts": None,
                    "errors": [error_msg],
                })
                logger.error(
                    "Batch %s dataset '%s' failed: %s",
                    batch_id,
                    dataset_id,
                    exc,
                    exc_info=True,
                )

        # Compute aggregated summary
        avg_pass_rate = (
            total_pass_rate / total_evaluations
            if total_evaluations > 0
            else 0.0
        )

        elapsed = _elapsed_ms(batch_start)

        if failed > 0 and succeeded == 0:
            batch_status = _STATUS_FAILED
        elif failed > 0:
            batch_status = _STATUS_PARTIAL
        else:
            batch_status = _STATUS_COMPLETED

        # Provenance for the batch run
        batch_provenance_payload = {
            "batch_id": batch_id,
            "total_datasets": len(datasets),
            "succeeded": succeeded,
            "failed": failed,
            "avg_pass_rate": avg_pass_rate,
        }
        provenance_hash = self._record_batch_provenance(
            batch_id, batch_provenance_payload,
        )

        _safe_observe_processing("batch_pipeline", elapsed / 1000.0)

        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "total_datasets": len(datasets),
            "succeeded": succeeded,
            "failed": failed,
            "dataset_results": dataset_results,
            "summary": {
                "avg_pass_rate": round(avg_pass_rate, 4),
                "total_failures": total_failures,
                "total_conflicts": total_conflicts,
                "datasets_evaluated": total_evaluations,
            },
            "duration_ms": elapsed,
            "started_at": started_at,
            "completed_at": _utcnow_iso(),
            "provenance_hash": provenance_hash,
            "status": batch_status,
            "errors": all_errors,
        }

        logger.info(
            "Batch pipeline %s completed: status=%s succeeded=%d "
            "failed=%d avg_pass_rate=%.4f duration_ms=%.2f",
            batch_id,
            batch_status,
            succeeded,
            failed,
            avg_pass_rate,
            elapsed,
        )

        return batch_result

    # ------------------------------------------------------------------
    # Public API -- pipeline run queries
    # ------------------------------------------------------------------

    def get_pipeline_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a pipeline run record by its unique identifier.

        Args:
            run_id: Unique pipeline run identifier.

        Returns:
            Pipeline result dictionary, or ``None`` if no run with the
            given identifier exists.

        Example:
            >>> run = engine.get_pipeline_run("pipe-abc123def456")
            >>> if run:
            ...     print(run["status"])
        """
        with self._lock:
            return self._pipeline_runs.get(run_id)

    def list_pipeline_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return a list of pipeline run records, most recent first.

        Args:
            limit: Maximum number of records to return.  Defaults to 100.

        Returns:
            List of pipeline result dictionaries sorted by ``started_at``
            descending (most recent first).

        Raises:
            ValueError: If ``limit`` is negative.

        Example:
            >>> runs = engine.list_pipeline_runs(limit=10)
            >>> print(len(runs))
        """
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")

        with self._lock:
            all_runs = list(self._pipeline_runs.values())

        all_runs.sort(
            key=lambda r: r.get("started_at", ""),
            reverse=True,
        )
        return all_runs[:limit]

    # ------------------------------------------------------------------
    # Public API -- health and statistics
    # ------------------------------------------------------------------

    def get_health(self) -> Dict[str, Any]:
        """Return engine health status and sub-engine availability.

        Checks the availability of each of the six upstream engines plus
        the provenance tracker and collects current aggregate statistics
        from the pipeline run store.

        Returns:
            Health status dictionary with keys:
            ``status``, ``engines``, ``engines_available``,
            ``engines_total``, ``pipeline_stats``, ``checked_at``.

        Example:
            >>> health = engine.get_health()
            >>> print(health["status"])
            healthy
        """
        engines = {
            "registry": self._registry is not None,
            "composer": self._composer is not None,
            "evaluator": self._evaluator is not None,
            "conflict_detector": self._conflict_detector is not None,
            "rule_pack": self._rule_pack is not None,
            "reporter": self._reporter is not None,
            "provenance": self._provenance is not None,
        }

        available_count = sum(1 for v in engines.values() if v)
        total_count = len(engines)

        if available_count == total_count:
            status = "healthy"
        elif available_count > 0:
            status = "degraded"
        else:
            status = "unhealthy"

        # Collect pipeline run stats
        pipeline_stats = self._get_run_summary_stats()

        # Collect sub-engine stats
        engine_stats = self._collect_engine_stats()

        return {
            "status": status,
            "engines": engines,
            "engines_available": available_count,
            "engines_total": total_count,
            "pipeline_stats": pipeline_stats,
            "engine_stats": engine_stats,
            "checked_at": _utcnow_iso(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics from all engines and pipeline runs.

        Computes statistics deterministically from the in-memory run store
        and engine state.  All numeric aggregations use pure Python
        arithmetic -- no LLM inference.

        Returns:
            Statistics dictionary with keys:
            ``total_runs``, ``succeeded``, ``failed``, ``partial``,
            ``by_status``, ``avg_duration_ms``, ``min_duration_ms``,
            ``max_duration_ms``, ``success_rate``, ``stage_timings``,
            ``total_notifications``, ``registry_stats``,
            ``evaluator_stats``, ``provenance_entry_count``,
            ``computed_at``.

        Example:
            >>> stats = engine.get_statistics()
            >>> print(stats["total_runs"])
            5
        """
        with self._lock:
            runs = list(self._pipeline_runs.values())
            notification_count = len(self._notifications)

        total = len(runs)
        by_status: Dict[str, int] = {}
        durations: List[float] = []
        stage_timing_sums: Dict[str, float] = {}
        stage_timing_counts: Dict[str, int] = {}

        for run in runs:
            status = run.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            dur = run.get("duration_ms")
            if isinstance(dur, (int, float)):
                durations.append(float(dur))

            # Accumulate per-stage timings
            st = run.get("stage_timings", {})
            if isinstance(st, dict):
                for stage_name, stage_ms in st.items():
                    if isinstance(stage_ms, (int, float)):
                        stage_timing_sums[stage_name] = (
                            stage_timing_sums.get(stage_name, 0.0)
                            + float(stage_ms)
                        )
                        stage_timing_counts[stage_name] = (
                            stage_timing_counts.get(stage_name, 0) + 1
                        )

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0

        completed = by_status.get(_STATUS_COMPLETED, 0)
        success_rate = completed / total if total > 0 else 0.0

        # Average stage timings
        avg_stage_timings: Dict[str, float] = {}
        for stage_name, total_ms in stage_timing_sums.items():
            count = stage_timing_counts.get(stage_name, 1)
            avg_stage_timings[stage_name] = round(total_ms / count, 2)

        # Collect sub-engine statistics
        registry_stats = self._get_registry_stats()
        evaluator_stats = self._get_evaluator_stats()

        provenance_count = 0
        if self._provenance is not None:
            try:
                provenance_count = self._provenance.entry_count
            except Exception:  # noqa: BLE001
                pass

        return {
            "total_runs": total,
            "succeeded": by_status.get(_STATUS_COMPLETED, 0),
            "failed": by_status.get(_STATUS_FAILED, 0),
            "partial": by_status.get(_STATUS_PARTIAL, 0),
            "by_status": by_status,
            "avg_duration_ms": round(avg_duration, 2),
            "min_duration_ms": round(min_duration, 2),
            "max_duration_ms": round(max_duration, 2),
            "success_rate": round(success_rate, 4),
            "stage_timings": avg_stage_timings,
            "total_notifications": notification_count,
            "registry_stats": registry_stats,
            "evaluator_stats": evaluator_stats,
            "provenance_entry_count": provenance_count,
            "computed_at": _utcnow_iso(),
        }

    # ------------------------------------------------------------------
    # Public API -- reset / clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all engine state, pipeline runs, and notifications.

        Clears the pipeline run store, notification list, resets the
        provenance tracker, and calls ``reset()`` on all sub-engines
        that support it.  Primarily intended for testing to prevent
        state leakage between test cases.

        Example:
            >>> engine.clear()
            >>> assert engine.get_statistics()["total_runs"] == 0
        """
        with self._lock:
            self._pipeline_runs.clear()
            self._notifications.clear()

        # Reset provenance
        if self._provenance is not None:
            try:
                self._provenance.reset()
            except Exception as exc:
                logger.warning("clear: provenance reset failed: %s", exc)

        # Reset sub-engines
        for name, eng in (
            ("registry", self._registry),
            ("composer", self._composer),
            ("evaluator", self._evaluator),
            ("conflict_detector", self._conflict_detector),
            ("rule_pack", self._rule_pack),
            ("reporter", self._reporter),
        ):
            if eng is not None and hasattr(eng, "reset"):
                try:
                    eng.reset()
                except Exception as exc:
                    logger.warning(
                        "clear: engine %s reset failed: %s", name, exc,
                    )

        # Reset gauges
        if _METRICS_AVAILABLE:
            if set_active_rules is not None:
                try:
                    set_active_rules(0)
                except Exception:  # noqa: BLE001
                    pass
            if set_active_rule_sets is not None:
                try:
                    set_active_rule_sets(0)
                except Exception:  # noqa: BLE001
                    pass
            if set_pass_rate is not None:
                try:
                    set_pass_rate(0.0)
                except Exception:  # noqa: BLE001
                    pass

        logger.info("ValidationPipelineEngine: full clear/reset complete")

    # ------------------------------------------------------------------
    # Internal: stage execution methods
    # ------------------------------------------------------------------

    def _execute_register_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        active_stages: set,
        pack_name: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Stage 1: REGISTER -- apply rule pack or register rules.

        When ``pack_name`` is provided, the rule pack engine is invoked to
        apply the specified regulatory rule pack (GHG Protocol, CSRD/ESRS,
        EUDR, SOC 2, or a custom pack).  Otherwise the stage records that
        registration was acknowledged with no new rules applied.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            active_stages: Set of stage names that should execute.
            pack_name: Optional rule pack name to apply.
            params: Additional parameters dictionary.

        Returns:
            Updated pipeline result dictionary.
        """
        if "register" not in active_stages:
            result["stages_skipped"].append("register")
            return result

        stage_start = time.monotonic()

        try:
            register_result: Dict[str, Any] = {
                "rules_applied": 0,
                "pack_name": pack_name,
                "details": None,
            }

            if pack_name is not None:
                if self._rule_pack is not None:
                    try:
                        pack_result = self._rule_pack.apply_pack(pack_name)
                        if isinstance(pack_result, dict):
                            register_result["rules_applied"] = pack_result.get(
                                "rules_applied", pack_result.get("rules_count", 0),
                            )
                            register_result["details"] = pack_result
                        else:
                            register_result["rules_applied"] = 0
                            register_result["details"] = {"raw": str(pack_result)}

                        # Record metrics for each rule in the pack
                        if (
                            _METRICS_AVAILABLE
                            and record_rule_registered is not None
                        ):
                            try:
                                record_rule_registered(
                                    "pack", pack_name,
                                )
                            except Exception:  # noqa: BLE001
                                pass

                    except Exception as exc:
                        error_msg = (
                            f"Failed to apply rule pack '{pack_name}': "
                            f"{str(exc)}"
                        )
                        result["errors"].append(error_msg)
                        register_result["details"] = {"error": str(exc)}
                        logger.warning(
                            "Pipeline %s register stage: pack apply "
                            "failed: %s",
                            pipeline_id,
                            exc,
                        )
                elif self._registry is not None:
                    register_result["details"] = {
                        "warning": (
                            "RulePackEngine unavailable; "
                            "cannot apply rule pack"
                        ),
                    }
                    result["errors"].append(
                        "RulePackEngine unavailable; "
                        f"cannot apply pack '{pack_name}'"
                    )
                else:
                    register_result["details"] = {
                        "error": "engine_unavailable",
                        "message": (
                            "Both RulePackEngine and RuleRegistryEngine "
                            "are unavailable"
                        ),
                    }
                    result["errors"].append(
                        "Register stage: no engines available"
                    )
            else:
                # No pack requested -- registration acknowledged
                register_result["details"] = {
                    "message": "No rule pack specified; "
                    "registration stage acknowledged",
                }

                # Check if individual rules were passed in params
                rules_to_register = params.get("rules", [])
                if rules_to_register and self._registry is not None:
                    registered_count = self._bulk_register_rules(
                        rules_to_register, result,
                    )
                    register_result["rules_applied"] = registered_count

            result["results"]["register"] = register_result
            result["stages_completed"].append("register")

        except Exception as exc:
            result["errors"].append(f"Register stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s register stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed = _elapsed_ms(stage_start)
        result["stage_timings"]["register"] = elapsed
        _safe_observe_processing("stage_register", elapsed / 1000.0)

        logger.info(
            "Pipeline %s stage register: pack=%s duration_ms=%.2f",
            pipeline_id,
            pack_name,
            elapsed,
        )
        return result

    def _execute_compose_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        active_stages: set,
        rule_set_id: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Stage 2: COMPOSE -- build or validate rule set composition.

        When ``rule_set_id`` is provided, the composer validates that the
        rule set exists, resolves compound rules, determines evaluation
        order via dependency graph analysis, and reports the composition
        structure.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            active_stages: Set of stage names that should execute.
            rule_set_id: Optional rule set identifier to compose/validate.
            params: Additional parameters dictionary.

        Returns:
            Updated pipeline result dictionary.
        """
        if "compose" not in active_stages:
            result["stages_skipped"].append("compose")
            return result

        stage_start = time.monotonic()

        try:
            compose_result: Dict[str, Any] = {
                "rule_set_id": rule_set_id,
                "rule_count": 0,
                "compound_rules": 0,
                "evaluation_order": [],
                "validation_status": "unknown",
                "details": None,
            }

            if rule_set_id is not None and self._composer is not None:
                try:
                    # Retrieve and validate rule set composition
                    rule_set_data = self._composer.get_rule_set(rule_set_id)
                    if rule_set_data is not None and isinstance(rule_set_data, dict):
                        rule_ids = rule_set_data.get("rule_ids", [])
                        compound_ids = rule_set_data.get(
                            "compound_rule_ids", [],
                        )
                        compose_result["rule_count"] = len(rule_ids)
                        compose_result["compound_rules"] = len(compound_ids)
                        compose_result["validation_status"] = "valid"
                        compose_result["details"] = rule_set_data

                        # Record metric for rules per set
                        if (
                            _METRICS_AVAILABLE
                            and record_rule_set_created is not None
                        ):
                            try:
                                record_rule_set_created("pipeline")
                            except Exception:  # noqa: BLE001
                                pass
                    else:
                        compose_result["validation_status"] = "not_found"
                        compose_result["details"] = {
                            "message": f"Rule set '{rule_set_id}' not found",
                        }

                except Exception as exc:
                    error_msg = (
                        f"Compose stage failed for rule_set '{rule_set_id}'"
                        f": {str(exc)}"
                    )
                    result["errors"].append(error_msg)
                    compose_result["validation_status"] = "error"
                    compose_result["details"] = {"error": str(exc)}
                    logger.warning(
                        "Pipeline %s compose stage: validation failed: %s",
                        pipeline_id,
                        exc,
                    )

            elif rule_set_id is not None and self._composer is None:
                compose_result["validation_status"] = "skipped"
                compose_result["details"] = {
                    "warning": "RuleComposerEngine unavailable",
                }
                result["errors"].append(
                    "Compose stage: RuleComposerEngine unavailable"
                )

            else:
                compose_result["validation_status"] = "skipped"
                compose_result["details"] = {
                    "message": "No rule_set_id specified; "
                    "compose stage skipped",
                }

            result["results"]["compose"] = compose_result
            result["stages_completed"].append("compose")

        except Exception as exc:
            result["errors"].append(f"Compose stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s compose stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed = _elapsed_ms(stage_start)
        result["stage_timings"]["compose"] = elapsed
        _safe_observe_processing("stage_compose", elapsed / 1000.0)

        logger.info(
            "Pipeline %s stage compose: rule_set=%s duration_ms=%.2f",
            pipeline_id,
            rule_set_id,
            elapsed,
        )
        return result

    def _execute_evaluate_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        active_stages: set,
        data: Optional[List[Dict[str, Any]]],
        rule_set_id: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Stage 3: EVALUATE -- execute rules against data.

        When ``data`` and ``rule_set_id`` are provided, the evaluator
        engine validates every record in the dataset against each rule in
        the rule set and produces per-rule and per-record results plus an
        aggregated evaluation summary.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            active_stages: Set of stage names that should execute.
            data: Optional list of data records to validate.
            rule_set_id: Optional rule set to evaluate against.
            params: Additional parameters dictionary.

        Returns:
            Updated pipeline result dictionary.
        """
        if "evaluate" not in active_stages:
            result["stages_skipped"].append("evaluate")
            return result

        stage_start = time.monotonic()

        try:
            evaluate_result: Dict[str, Any] = {
                "evaluation_id": _new_id("eval"),
                "rule_set_id": rule_set_id,
                "records_evaluated": 0,
                "rules_executed": 0,
                "pass_count": 0,
                "fail_count": 0,
                "warn_count": 0,
                "skip_count": 0,
                "pass_rate": 0.0,
                "verdict": "unknown",
                "details": None,
            }

            if data is not None and self._evaluator is not None:
                try:
                    # Determine thresholds
                    pass_threshold = params.get(
                        "pass_threshold", 0.95,
                    )
                    warn_threshold = params.get(
                        "warn_threshold", 0.80,
                    )
                    short_circuit = params.get(
                        "short_circuit", False,
                    )

                    # Build a rule set dict for the evaluator
                    rule_set_dict = self._resolve_rule_set(
                        rule_set_id,
                        pass_threshold,
                        warn_threshold,
                        short_circuit,
                    )

                    eval_output = self._evaluator.evaluate_rule_set(
                        rule_set=rule_set_dict,
                        data=data,
                    )

                    if isinstance(eval_output, dict):
                        evaluate_result["records_evaluated"] = eval_output.get(
                            "records_evaluated", len(data),
                        )
                        evaluate_result["rules_executed"] = eval_output.get(
                            "rules_evaluated", 0,
                        )
                        evaluate_result["pass_count"] = eval_output.get(
                            "rules_passed", 0,
                        )
                        evaluate_result["fail_count"] = eval_output.get(
                            "rules_failed", 0,
                        )
                        evaluate_result["warn_count"] = eval_output.get(
                            "warn_count", 0,
                        )
                        evaluate_result["skip_count"] = eval_output.get(
                            "skip_count", 0,
                        )
                        evaluate_result["pass_rate"] = eval_output.get(
                            "overall_pass_rate", 0.0,
                        )
                        sla_result = eval_output.get("sla_result", "unknown")
                        evaluate_result["verdict"] = sla_result
                        evaluate_result["details"] = eval_output

                    # Record evaluation metrics
                    self._record_evaluation_metrics(evaluate_result)

                except Exception as exc:
                    error_msg = (
                        f"Evaluate stage failed: {str(exc)}"
                    )
                    result["errors"].append(error_msg)
                    evaluate_result["verdict"] = "error"
                    evaluate_result["details"] = {"error": str(exc)}
                    logger.warning(
                        "Pipeline %s evaluate stage: evaluation "
                        "failed: %s",
                        pipeline_id,
                        exc,
                    )

            elif data is not None and self._evaluator is None:
                evaluate_result["verdict"] = "skipped"
                evaluate_result["details"] = {
                    "warning": "RuleEvaluatorEngine unavailable",
                }
                result["errors"].append(
                    "Evaluate stage: RuleEvaluatorEngine unavailable"
                )

            else:
                evaluate_result["verdict"] = "skipped"
                evaluate_result["records_evaluated"] = 0
                evaluate_result["details"] = {
                    "message": "No data provided; evaluate stage skipped",
                }

            result["results"]["evaluate"] = evaluate_result
            result["evaluation_summary"] = {
                "evaluation_id": evaluate_result["evaluation_id"],
                "records_evaluated": evaluate_result["records_evaluated"],
                "rules_executed": evaluate_result["rules_executed"],
                "pass_count": evaluate_result["pass_count"],
                "fail_count": evaluate_result["fail_count"],
                "warn_count": evaluate_result["warn_count"],
                "skip_count": evaluate_result["skip_count"],
                "pass_rate": evaluate_result["pass_rate"],
                "verdict": evaluate_result["verdict"],
            }
            result["stages_completed"].append("evaluate")

        except Exception as exc:
            result["errors"].append(f"Evaluate stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s evaluate stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed = _elapsed_ms(stage_start)
        result["stage_timings"]["evaluate"] = elapsed
        _safe_observe_evaluation("rule_set", elapsed / 1000.0)

        logger.info(
            "Pipeline %s stage evaluate: records=%d verdict=%s "
            "duration_ms=%.2f",
            pipeline_id,
            result.get("evaluation_summary", {}).get(
                "records_evaluated", 0,
            ) if isinstance(result.get("evaluation_summary"), dict) else 0,
            result.get("evaluation_summary", {}).get(
                "verdict", "unknown",
            ) if isinstance(result.get("evaluation_summary"), dict) else "unknown",
            elapsed,
        )
        return result

    def _execute_detect_conflicts_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        active_stages: set,
        rule_set_id: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Stage 4: DETECT_CONFLICTS -- run conflict detection.

        Analyses the active rule set for contradictions (rules with no
        valid intersection), overlaps (partially intersecting ranges),
        redundancies (one rule subsumed by another), and severity
        inconsistencies (same condition with different severity levels).

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            active_stages: Set of stage names that should execute.
            rule_set_id: Optional rule set to analyse for conflicts.
            params: Additional parameters dictionary.

        Returns:
            Updated pipeline result dictionary.
        """
        if "detect_conflicts" not in active_stages:
            result["stages_skipped"].append("detect_conflicts")
            return result

        stage_start = time.monotonic()

        try:
            conflict_result: Dict[str, Any] = {
                "conflict_count": 0,
                "contradictions": 0,
                "overlaps": 0,
                "redundancies": 0,
                "severity_inconsistencies": 0,
                "conflicts": [],
                "resolution_suggestions": [],
                "details": None,
            }

            if self._conflict_detector is not None:
                try:
                    # Resolve rule IDs for scope filtering when a
                    # rule_set_id is provided
                    rule_ids = self._resolve_rule_ids_for_conflict(
                        rule_set_id,
                    )

                    detection_output = self._conflict_detector.detect_all_conflicts(
                        rule_ids=rule_ids,
                    )

                    if isinstance(detection_output, dict):
                        type_dist = detection_output.get(
                            "type_distribution", {},
                        )
                        total = detection_output.get(
                            "total_conflicts", 0,
                        )
                        conflict_result["conflict_count"] = total
                        conflict_result["contradictions"] = type_dist.get(
                            "range_contradiction", 0,
                        )
                        conflict_result["overlaps"] = type_dist.get(
                            "range_overlap", 0,
                        )
                        conflict_result["redundancies"] = type_dist.get(
                            "redundancy", 0,
                        )
                        conflict_result["severity_inconsistencies"] = (
                            type_dist.get("severity_inconsistency", 0)
                        )
                        conflict_result["conflicts"] = detection_output.get(
                            "conflicts", [],
                        )
                        conflict_result["resolution_suggestions"] = (
                            detection_output.get("recommendations", [])
                        )
                        conflict_result["details"] = detection_output

                    # Record conflict metrics
                    self._record_conflict_metrics(conflict_result)

                except Exception as exc:
                    error_msg = (
                        f"Conflict detection failed: {str(exc)}"
                    )
                    result["errors"].append(error_msg)
                    conflict_result["details"] = {"error": str(exc)}
                    logger.warning(
                        "Pipeline %s detect_conflicts stage: "
                        "detection failed: %s",
                        pipeline_id,
                        exc,
                    )
            else:
                conflict_result["details"] = {
                    "warning": "ConflictDetectorEngine unavailable",
                }

            result["results"]["detect_conflicts"] = conflict_result
            result["conflicts"] = {
                "conflict_count": conflict_result["conflict_count"],
                "contradictions": conflict_result["contradictions"],
                "overlaps": conflict_result["overlaps"],
                "redundancies": conflict_result["redundancies"],
                "severity_inconsistencies": (
                    conflict_result["severity_inconsistencies"]
                ),
            }
            result["stages_completed"].append("detect_conflicts")

        except Exception as exc:
            result["errors"].append(
                f"Detect conflicts stage failed: {str(exc)}"
            )
            logger.error(
                "Pipeline %s detect_conflicts stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed = _elapsed_ms(stage_start)
        result["stage_timings"]["detect_conflicts"] = elapsed
        _safe_observe_processing("stage_detect_conflicts", elapsed / 1000.0)

        logger.info(
            "Pipeline %s stage detect_conflicts: conflicts=%d "
            "duration_ms=%.2f",
            pipeline_id,
            result.get("conflicts", {}).get("conflict_count", 0)
            if isinstance(result.get("conflicts"), dict) else 0,
            elapsed,
        )
        return result

    def _execute_report_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        active_stages: set,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Stage 5: REPORT -- generate validation report.

        Generates a validation report from the evaluation results and
        conflict analysis.  Supports five report types (summary, detailed,
        compliance, trend, executive) and five output formats (text, JSON,
        HTML, Markdown, CSV).

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            active_stages: Set of stage names that should execute.
            params: Additional parameters with optional ``report_type``
                and ``report_format`` keys.

        Returns:
            Updated pipeline result dictionary.
        """
        if "report" not in active_stages:
            result["stages_skipped"].append("report")
            return result

        stage_start = time.monotonic()

        try:
            report_type = params.get("report_type", "summary")
            report_format = params.get("report_format", "json")
            report_id = _new_id("rpt")

            report_result: Dict[str, Any] = {
                "report_id": report_id,
                "report_type": report_type,
                "report_format": report_format,
                "content": None,
                "details": None,
            }

            if self._reporter is not None:
                try:
                    # Build evaluation results list for the reporter
                    evaluation_results = self._build_evaluation_results_list(
                        result,
                    )

                    report_output = self._reporter.generate_report(
                        report_type=report_type,
                        format=report_format,
                        evaluation_results=evaluation_results,
                    )

                    if isinstance(report_output, dict):
                        report_result["content"] = report_output.get(
                            "content", report_output,
                        )
                        report_result["details"] = report_output
                        # Use reporter's report_id if available
                        report_result["report_id"] = report_output.get(
                            "report_id", report_id,
                        )
                    else:
                        report_result["content"] = report_output
                        report_result["details"] = {
                            "raw": str(report_output),
                        }

                    # Record report metric
                    if (
                        _METRICS_AVAILABLE
                        and record_report_generated is not None
                    ):
                        try:
                            record_report_generated(
                                report_type, report_format,
                            )
                        except Exception:  # noqa: BLE001
                            pass

                except Exception as exc:
                    error_msg = (
                        f"Report generation failed: {str(exc)}"
                    )
                    result["errors"].append(error_msg)
                    report_result["details"] = {"error": str(exc)}
                    logger.warning(
                        "Pipeline %s report stage: generation "
                        "failed: %s",
                        pipeline_id,
                        exc,
                    )
            else:
                # Reporter unavailable -- build minimal inline report
                report_result["content"] = self._build_inline_report(result)
                report_result["details"] = {
                    "warning": "ValidationReporterEngine unavailable; "
                    "inline report generated",
                }

            result["results"]["report"] = report_result
            result["report_id"] = report_result["report_id"]
            result["stages_completed"].append("report")

        except Exception as exc:
            result["errors"].append(f"Report stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s report stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed = _elapsed_ms(stage_start)
        result["stage_timings"]["report"] = elapsed
        _safe_observe_processing("stage_report", elapsed / 1000.0)

        logger.info(
            "Pipeline %s stage report: type=%s format=%s "
            "duration_ms=%.2f",
            pipeline_id,
            params.get("report_type", "summary"),
            params.get("report_format", "json"),
            elapsed,
        )
        return result

    def _execute_audit_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        active_stages: set,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Stage 6: AUDIT -- record audit trail with provenance.

        Records a comprehensive audit trail entry for the pipeline run,
        capturing all stage outcomes, timings, and evaluation results with
        a SHA-256 provenance hash for tamper-evident compliance reporting.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            active_stages: Set of stage names that should execute.
            params: Additional parameters dictionary.

        Returns:
            Updated pipeline result dictionary.
        """
        if "audit" not in active_stages:
            result["stages_skipped"].append("audit")
            return result

        stage_start = time.monotonic()

        try:
            audit_entry: Dict[str, Any] = {
                "audit_id": _new_id("aud"),
                "pipeline_id": pipeline_id,
                "timestamp": _utcnow_iso(),
                "stages_completed": list(result.get("stages_completed", [])),
                "stages_skipped": list(result.get("stages_skipped", [])),
                "error_count": len(result.get("errors", [])),
                "evaluation_verdict": None,
                "conflict_count": 0,
                "report_id": result.get("report_id"),
                "audit_hash": None,
            }

            # Extract evaluation verdict
            eval_summary = result.get("evaluation_summary")
            if isinstance(eval_summary, dict):
                audit_entry["evaluation_verdict"] = eval_summary.get(
                    "verdict", "unknown",
                )

            # Extract conflict count
            conflicts = result.get("conflicts")
            if isinstance(conflicts, dict):
                audit_entry["conflict_count"] = conflicts.get(
                    "conflict_count", 0,
                )

            # Compute audit hash over the current pipeline state
            audit_payload = {
                "pipeline_id": pipeline_id,
                "stages_completed": audit_entry["stages_completed"],
                "evaluation_verdict": audit_entry["evaluation_verdict"],
                "conflict_count": audit_entry["conflict_count"],
                "error_count": audit_entry["error_count"],
                "timestamp": audit_entry["timestamp"],
            }
            audit_entry["audit_hash"] = _sha256(audit_payload)

            # Record provenance entry for the audit action
            if self._provenance is not None:
                try:
                    self._provenance.record(
                        "audit",
                        audit_entry["audit_id"],
                        "audit_recorded",
                        metadata=audit_payload,
                    )
                except Exception:  # noqa: BLE001
                    pass

            result["results"]["audit"] = audit_entry
            result["stages_completed"].append("audit")

        except Exception as exc:
            result["errors"].append(f"Audit stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s audit stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed = _elapsed_ms(stage_start)
        result["stage_timings"]["audit"] = elapsed
        _safe_observe_processing("stage_audit", elapsed / 1000.0)

        logger.info(
            "Pipeline %s stage audit: duration_ms=%.2f",
            pipeline_id,
            elapsed,
        )
        return result

    def _execute_notify_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        active_stages: set,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Stage 7: NOTIFY -- generate notification payload.

        Generates a structured notification payload suitable for delivery
        to alerting subsystems (OBS-004 bridge, Slack, PagerDuty, email,
        Microsoft Teams).  The notification includes the evaluation verdict,
        failure counts, conflict counts, and configurable severity mapping.

        Notifications are stored in-memory for later retrieval and can be
        forwarded to external channels by the API layer.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            active_stages: Set of stage names that should execute.
            params: Additional parameters with optional
                ``notification_channels`` key.

        Returns:
            Updated pipeline result dictionary.
        """
        if "notify" not in active_stages:
            result["stages_skipped"].append("notify")
            return result

        stage_start = time.monotonic()

        try:
            notification = self._build_notification_payload(
                pipeline_id, result, params,
            )

            # Store notification
            with self._lock:
                self._notifications.append(notification)

            result["results"]["notify"] = notification
            result["stages_completed"].append("notify")

        except Exception as exc:
            result["errors"].append(f"Notify stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s notify stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed = _elapsed_ms(stage_start)
        result["stage_timings"]["notify"] = elapsed
        _safe_observe_processing("stage_notify", elapsed / 1000.0)

        logger.info(
            "Pipeline %s stage notify: duration_ms=%.2f",
            pipeline_id,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: engine method adapters
    # ------------------------------------------------------------------

    def _resolve_rule_set(
        self,
        rule_set_id: Optional[str],
        pass_threshold: float,
        warn_threshold: float,
        short_circuit: bool,
    ) -> Dict[str, Any]:
        """Resolve a rule set dict suitable for RuleEvaluatorEngine.

        If a ``rule_set_id`` is provided and the composer has the set
        registered, the set definition is retrieved.  Otherwise, a
        minimal stub set is built from all active rules in the registry.

        Args:
            rule_set_id: Optional rule set identifier to look up.
            pass_threshold: SLA pass rate threshold (0.0 to 1.0).
            warn_threshold: SLA warning rate threshold (0.0 to 1.0).
            short_circuit: Whether to enable AND short-circuit evaluation.

        Returns:
            Rule set dictionary with keys ``set_id``, ``set_name``,
            ``rules``, ``sla_pass_rate``, ``sla_warn_rate``,
            ``short_circuit``.
        """
        rules_list: List[Dict[str, Any]] = []

        # Try to retrieve from composer first
        if rule_set_id is not None and self._composer is not None:
            try:
                rule_set_data = self._composer.get_rule_set(rule_set_id)
                if rule_set_data is not None and isinstance(rule_set_data, dict):
                    # Resolve individual rule definitions from the registry
                    rule_ids = rule_set_data.get("rule_ids", [])
                    if self._registry is not None:
                        for rid in rule_ids:
                            try:
                                rule_def = self._registry.get_rule(rid)
                                if rule_def is not None:
                                    rules_list.append(rule_def)
                            except Exception:  # noqa: BLE001
                                pass

                    return {
                        "set_id": rule_set_id,
                        "set_name": rule_set_data.get("name", rule_set_id),
                        "rules": rules_list,
                        "sla_pass_rate": pass_threshold,
                        "sla_warn_rate": warn_threshold,
                        "short_circuit": short_circuit,
                    }
            except Exception:  # noqa: BLE001
                pass

        # Fallback: use all active rules from the registry
        if self._registry is not None:
            try:
                all_rules = self._registry.list_rules()
                if isinstance(all_rules, dict):
                    rules_list = all_rules.get("rules", [])
                elif isinstance(all_rules, list):
                    rules_list = all_rules
            except Exception:  # noqa: BLE001
                pass

        return {
            "set_id": rule_set_id or "pipeline-adhoc",
            "set_name": rule_set_id or "Pipeline Ad-Hoc Rule Set",
            "rules": rules_list,
            "sla_pass_rate": pass_threshold,
            "sla_warn_rate": warn_threshold,
            "short_circuit": short_circuit,
        }

    def _resolve_rule_ids_for_conflict(
        self,
        rule_set_id: Optional[str],
    ) -> Optional[List[str]]:
        """Resolve rule IDs for conflict detection scope.

        When a ``rule_set_id`` is provided, returns the rule IDs belonging
        to that set so conflict detection is scoped.  When ``None``,
        returns ``None`` so the detector analyses all active rules.

        Args:
            rule_set_id: Optional rule set identifier.

        Returns:
            List of rule ID strings, or ``None`` for all-rules analysis.
        """
        if rule_set_id is None:
            return None

        if self._composer is not None:
            try:
                rule_set_data = self._composer.get_rule_set(rule_set_id)
                if rule_set_data is not None and isinstance(rule_set_data, dict):
                    return rule_set_data.get("rule_ids", None)
            except Exception:  # noqa: BLE001
                pass

        return None

    def _build_evaluation_results_list(
        self,
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build an evaluation results list for the reporter engine.

        Extracts per-rule evaluation results from the pipeline result
        dictionary and packages them in the format expected by
        ``ValidationReporterEngine.generate_report()``.

        Args:
            result: Current pipeline result dictionary.

        Returns:
            List of per-rule evaluation result dictionaries.
        """
        eval_data = result.get("results", {}).get("evaluate", {})
        if not isinstance(eval_data, dict):
            return []

        details = eval_data.get("details")
        if isinstance(details, dict):
            # Try to extract per-rule results from the evaluator output
            per_rule = details.get("per_rule_results", [])
            if per_rule:
                return per_rule

        # Build a minimal results list from the evaluation summary
        eval_summary = result.get("evaluation_summary", {})
        if isinstance(eval_summary, dict):
            return [{
                "rule_id": "pipeline_summary",
                "status": eval_summary.get("verdict", "unknown"),
                "severity": "medium",
                "pass_count": eval_summary.get("pass_count", 0),
                "fail_count": eval_summary.get("fail_count", 0),
            }]

        return []

    # ------------------------------------------------------------------
    # Internal: notification builder
    # ------------------------------------------------------------------

    def _build_notification_payload(
        self,
        pipeline_id: str,
        result: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a structured notification payload from pipeline results.

        The notification payload is designed to be directly consumable by
        the OBS-004 Alerting & Notification Platform and includes all
        information needed for alert routing, severity mapping, and
        channel delivery.

        Args:
            pipeline_id: Pipeline run identifier.
            result: Current pipeline result dictionary.
            params: Additional parameters with optional channel config.

        Returns:
            Notification payload dictionary.
        """
        channels = params.get("notification_channels", ["default"])

        # Determine notification severity from evaluation verdict and
        # conflict count
        eval_summary = result.get("evaluation_summary")
        verdict = "unknown"
        fail_count = 0
        pass_rate = 0.0

        if isinstance(eval_summary, dict):
            verdict = eval_summary.get("verdict", "unknown")
            fail_count = eval_summary.get("fail_count", 0)
            pass_rate = eval_summary.get("pass_rate", 0.0)

        conflict_count = 0
        conflicts = result.get("conflicts")
        if isinstance(conflicts, dict):
            conflict_count = conflicts.get("conflict_count", 0)

        severity = self._determine_notification_severity(
            verdict, fail_count, conflict_count,
        )

        # Build human-readable summary
        summary_lines = [
            f"Validation Pipeline {pipeline_id}",
            f"Verdict: {verdict}",
            f"Pass Rate: {pass_rate:.1%}" if isinstance(pass_rate, float) else f"Pass Rate: {pass_rate}",
            f"Failures: {fail_count}",
        ]
        if conflict_count > 0:
            summary_lines.append(f"Conflicts: {conflict_count}")

        error_count = len(result.get("errors", []))
        if error_count > 0:
            summary_lines.append(f"Errors: {error_count}")

        summary_text = " | ".join(summary_lines)

        notification: Dict[str, Any] = {
            "notification_id": _new_id("ntf"),
            "pipeline_id": pipeline_id,
            "timestamp": _utcnow_iso(),
            "severity": severity,
            "priority": _NOTIFICATION_SEVERITY_MAP.get(severity, "P5"),
            "channels": channels,
            "summary": summary_text,
            "evaluation_verdict": verdict,
            "pass_rate": pass_rate,
            "fail_count": fail_count,
            "conflict_count": conflict_count,
            "error_count": error_count,
            "stages_completed": result.get("stages_completed", []),
            "report_id": result.get("report_id"),
            "requires_action": verdict in ("fail", "error") or conflict_count > 0,
        }

        logger.debug(
            "Notification built: id=%s severity=%s channels=%s",
            notification["notification_id"],
            severity,
            channels,
        )
        return notification

    def _determine_notification_severity(
        self,
        verdict: str,
        fail_count: int,
        conflict_count: int,
    ) -> str:
        """Determine notification severity from evaluation results.

        Uses a deterministic mapping based on verdict, failure count, and
        conflict count.  No LLM inference is involved.

        Args:
            verdict: Evaluation verdict string.
            fail_count: Number of rule failures.
            conflict_count: Number of rule conflicts detected.

        Returns:
            Severity string: ``"critical"``, ``"high"``, ``"medium"``,
            ``"low"``, or ``"info"``.
        """
        if verdict == "error":
            return "critical"
        if verdict == "fail" and fail_count > 10:
            return "critical"
        if verdict == "fail":
            return "high"
        if conflict_count > 5:
            return "high"
        if conflict_count > 0:
            return "medium"
        if verdict == "warn":
            return "medium"
        if verdict == "pass":
            return "info"
        return "low"

    # ------------------------------------------------------------------
    # Internal: report builders
    # ------------------------------------------------------------------

    def _build_report_input(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a report input dictionary from the current pipeline state.

        Extracts evaluation summary, conflict analysis, and stage timings
        from the pipeline result dictionary and packages them into a
        format suitable for the ValidationReporterEngine.

        Args:
            result: Current pipeline result dictionary.

        Returns:
            Report input dictionary.
        """
        return {
            "pipeline_id": result.get("pipeline_id"),
            "evaluation_summary": result.get("evaluation_summary"),
            "conflicts": result.get("conflicts"),
            "stage_timings": result.get("stage_timings", {}),
            "errors": result.get("errors", []),
            "started_at": result.get("started_at"),
        }

    def _build_inline_report(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a minimal inline report when the reporter is unavailable.

        Generates a basic report dictionary from the pipeline results
        without relying on the ValidationReporterEngine.  All data is
        deterministically extracted from existing pipeline state.

        Args:
            result: Current pipeline result dictionary.

        Returns:
            Inline report dictionary.
        """
        eval_summary = result.get("evaluation_summary", {})
        conflicts = result.get("conflicts", {})

        verdict = "unknown"
        pass_rate = 0.0
        fail_count = 0
        records_evaluated = 0

        if isinstance(eval_summary, dict):
            verdict = eval_summary.get("verdict", "unknown")
            pass_rate = eval_summary.get("pass_rate", 0.0)
            fail_count = eval_summary.get("fail_count", 0)
            records_evaluated = eval_summary.get("records_evaluated", 0)

        conflict_count = 0
        if isinstance(conflicts, dict):
            conflict_count = conflicts.get("conflict_count", 0)

        return {
            "report_type": "inline_summary",
            "format": "json",
            "generated_at": _utcnow_iso(),
            "pipeline_id": result.get("pipeline_id"),
            "verdict": verdict,
            "pass_rate": pass_rate,
            "fail_count": fail_count,
            "records_evaluated": records_evaluated,
            "conflict_count": conflict_count,
            "error_count": len(result.get("errors", [])),
            "stages_completed": len(result.get("stages_completed", [])),
            "duration_ms": sum(
                v for v in result.get("stage_timings", {}).values()
                if isinstance(v, (int, float))
            ),
            "report_hash": _sha256({
                "verdict": verdict,
                "pass_rate": pass_rate,
                "fail_count": fail_count,
                "conflict_count": conflict_count,
            }),
        }

    # ------------------------------------------------------------------
    # Internal: bulk rule registration
    # ------------------------------------------------------------------

    def _bulk_register_rules(
        self,
        rules: List[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> int:
        """Register multiple rules via the RuleRegistryEngine.

        Args:
            rules: List of rule definition dictionaries.
            result: Pipeline result dictionary (errors are appended).

        Returns:
            Number of rules successfully registered.
        """
        registered = 0

        if self._registry is None:
            return registered

        for rule_def in rules:
            try:
                self._registry.register_rule(
                    rule_id=rule_def.get("rule_id"),
                    name=rule_def.get("name", ""),
                    rule_type=rule_def.get("rule_type", "CUSTOM"),
                    column=rule_def.get("column", ""),
                    operator=rule_def.get("operator", "EQUALS"),
                    expected_value=rule_def.get("expected_value"),
                    severity=rule_def.get("severity", "MEDIUM"),
                    description=rule_def.get("description", ""),
                    tags=rule_def.get("tags", []),
                )
                registered += 1

                if _METRICS_AVAILABLE and record_rule_registered is not None:
                    try:
                        record_rule_registered(
                            rule_def.get("rule_type", "CUSTOM"),
                            rule_def.get("severity", "MEDIUM"),
                        )
                    except Exception:  # noqa: BLE001
                        pass

            except Exception as exc:
                rule_id = rule_def.get("rule_id", "unknown")
                result["errors"].append(
                    f"Failed to register rule '{rule_id}': {str(exc)}"
                )
                logger.warning(
                    "_bulk_register_rules: failed for '%s': %s",
                    rule_id,
                    exc,
                )

        return registered

    # ------------------------------------------------------------------
    # Internal: metrics recording helpers
    # ------------------------------------------------------------------

    def _record_evaluation_metrics(
        self,
        evaluate_result: Dict[str, Any],
    ) -> None:
        """Record Prometheus metrics for an evaluation result.

        Args:
            evaluate_result: Evaluation result dictionary containing
                pass_count, fail_count, and verdict.
        """
        if not _METRICS_AVAILABLE:
            return

        verdict = evaluate_result.get("verdict", "unknown")

        if record_evaluation is not None:
            try:
                record_evaluation(verdict, "rule_set")
            except Exception:  # noqa: BLE001
                pass

        fail_count = evaluate_result.get("fail_count", 0)
        if fail_count > 0 and record_evaluation_failure is not None:
            try:
                record_evaluation_failure("mixed")
            except Exception:  # noqa: BLE001
                pass

    def _record_conflict_metrics(
        self,
        conflict_result: Dict[str, Any],
    ) -> None:
        """Record Prometheus metrics for conflict detection results.

        Args:
            conflict_result: Conflict detection result dictionary.
        """
        if not _METRICS_AVAILABLE or record_conflict_detected is None:
            return

        conflict_types = [
            ("contradictions", "contradiction"),
            ("overlaps", "overlap"),
            ("redundancies", "redundancy"),
            ("severity_inconsistencies", "severity_inconsistency"),
        ]

        for field_name, metric_label in conflict_types:
            count = conflict_result.get(field_name, 0)
            if count > 0:
                try:
                    for _ in range(count):
                        record_conflict_detected(metric_label)
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Internal: provenance helpers
    # ------------------------------------------------------------------

    def _record_pipeline_provenance(
        self,
        pipeline_id: str,
        result: Dict[str, Any],
    ) -> Optional[str]:
        """Record a provenance entry for a completed pipeline run.

        Args:
            pipeline_id: Pipeline run identifier.
            result: Pipeline result dictionary to hash.

        Returns:
            SHA-256 provenance hash, or ``None`` if provenance is
            unavailable.
        """
        if self._provenance is None:
            return _sha256(result)

        try:
            entry = self._provenance.record(
                entity_type="evaluation",
                entity_id=pipeline_id,
                action="evaluation_completed",
                metadata={
                    "status": result.get("status"),
                    "stages_completed": result.get("stages_completed", []),
                    "evaluation_verdict": (
                        result.get("evaluation_summary", {}).get(
                            "verdict", "unknown",
                        )
                        if isinstance(result.get("evaluation_summary"), dict)
                        else "unknown"
                    ),
                    "conflict_count": (
                        result.get("conflicts", {}).get("conflict_count", 0)
                        if isinstance(result.get("conflicts"), dict)
                        else 0
                    ),
                    "duration_ms": result.get("duration_ms", 0),
                    "error_count": len(result.get("errors", [])),
                },
            )
            return entry.hash_value
        except Exception as exc:
            logger.warning(
                "_record_pipeline_provenance: failed for %s: %s",
                pipeline_id,
                exc,
            )
            return _sha256(result)

    def _record_batch_provenance(
        self,
        batch_id: str,
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Record a provenance entry for a completed batch pipeline run.

        Args:
            batch_id: Batch run identifier.
            payload: Batch summary payload to hash.

        Returns:
            SHA-256 provenance hash, or ``None`` if provenance is
            unavailable.
        """
        if self._provenance is None:
            return _sha256(payload)

        try:
            entry = self._provenance.record(
                entity_type="evaluation",
                entity_id=batch_id,
                action="batch_evaluation_completed",
                metadata=payload,
            )
            return entry.hash_value
        except Exception as exc:
            logger.warning(
                "_record_batch_provenance: failed for %s: %s",
                batch_id,
                exc,
            )
            return _sha256(payload)

    # ------------------------------------------------------------------
    # Internal: sub-engine statistics helpers
    # ------------------------------------------------------------------

    def _get_registry_stats(self) -> Dict[str, Any]:
        """Collect statistics from the rule registry engine.

        Returns:
            Dictionary with rule counts and type breakdown, or a status
            indicator when the engine is unavailable.
        """
        if self._registry is None:
            return {"status": "unavailable"}

        try:
            if hasattr(self._registry, "get_statistics"):
                return self._registry.get_statistics()
            if hasattr(self._registry, "list_rules"):
                rules = self._registry.list_rules()
                if isinstance(rules, dict):
                    return {"total_rules": len(rules.get("rules", []))}
                if isinstance(rules, list):
                    return {"total_rules": len(rules)}
            return {"total_rules": 0}
        except Exception as exc:
            logger.warning("_get_registry_stats failed: %s", exc)
            return {"status": "error", "reason": str(exc)}

    def _get_evaluator_stats(self) -> Dict[str, Any]:
        """Collect statistics from the rule evaluator engine.

        Returns:
            Dictionary with evaluation counts and pass rate, or a status
            indicator when the engine is unavailable.
        """
        if self._evaluator is None:
            return {"status": "unavailable"}

        try:
            if hasattr(self._evaluator, "get_statistics"):
                return self._evaluator.get_statistics()
            return {"total_evaluations": 0}
        except Exception as exc:
            logger.warning("_get_evaluator_stats failed: %s", exc)
            return {"status": "error", "reason": str(exc)}

    def _get_run_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics from pipeline run history.

        Returns:
            Dictionary with total runs, success/failure counts, and
            average duration.
        """
        with self._lock:
            runs = list(self._pipeline_runs.values())

        total = len(runs)
        if total == 0:
            return {
                "total_runs": 0,
                "succeeded": 0,
                "failed": 0,
                "partial": 0,
                "avg_duration_ms": 0.0,
            }

        succeeded = sum(
            1 for r in runs if r.get("status") == _STATUS_COMPLETED
        )
        failed = sum(
            1 for r in runs if r.get("status") == _STATUS_FAILED
        )
        partial = sum(
            1 for r in runs if r.get("status") == _STATUS_PARTIAL
        )

        durations = [
            float(r.get("duration_ms", 0))
            for r in runs
            if isinstance(r.get("duration_ms"), (int, float))
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_runs": total,
            "succeeded": succeeded,
            "failed": failed,
            "partial": partial,
            "avg_duration_ms": round(avg_duration, 2),
        }

    def _collect_engine_stats(self) -> Dict[str, Any]:
        """Collect availability and basic statistics from each engine.

        Returns:
            Dictionary mapping engine name to availability and stats.
        """
        engine_stats: Dict[str, Any] = {}

        for name, engine in (
            ("registry", self._registry),
            ("composer", self._composer),
            ("evaluator", self._evaluator),
            ("conflict_detector", self._conflict_detector),
            ("rule_pack", self._rule_pack),
            ("reporter", self._reporter),
        ):
            if engine is None:
                engine_stats[name] = {"available": False}
                continue

            stats: Dict[str, Any] = {"available": True}

            if hasattr(engine, "get_statistics"):
                try:
                    engine_detail = engine.get_statistics()
                    if isinstance(engine_detail, dict):
                        stats.update(engine_detail)
                except Exception:  # noqa: BLE001
                    stats["stats_error"] = True

            engine_stats[name] = stats

        return engine_stats

    # ------------------------------------------------------------------
    # Internal: gauge update helper
    # ------------------------------------------------------------------

    def _update_gauges(self) -> None:
        """Update Prometheus gauges for active rules, rule sets, pass rate."""
        if not _METRICS_AVAILABLE:
            return

        # Update active rules gauge
        if set_active_rules is not None and self._registry is not None:
            try:
                stats = self._get_registry_stats()
                if isinstance(stats, dict):
                    count = stats.get("total_rules", stats.get("active_rules", 0))
                    if isinstance(count, int):
                        set_active_rules(count)
            except Exception:  # noqa: BLE001
                pass

        # Update active rule sets gauge
        if set_active_rule_sets is not None and self._composer is not None:
            try:
                if hasattr(self._composer, "get_statistics"):
                    stats = self._composer.get_statistics()
                    if isinstance(stats, dict):
                        count = stats.get(
                            "total_rule_sets",
                            stats.get("active_rule_sets", 0),
                        )
                        if isinstance(count, int):
                            set_active_rule_sets(count)
            except Exception:  # noqa: BLE001
                pass

        # Update pass rate gauge from most recent run
        if set_pass_rate is not None:
            try:
                with self._lock:
                    if self._pipeline_runs:
                        # Get the most recent run
                        latest = max(
                            self._pipeline_runs.values(),
                            key=lambda r: r.get("started_at", ""),
                        )
                        eval_summary = latest.get("evaluation_summary")
                        if isinstance(eval_summary, dict):
                            rate = eval_summary.get("pass_rate", 0.0)
                            if isinstance(rate, (int, float)):
                                set_pass_rate(float(rate))
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ValidationPipelineEngine",
    "PIPELINE_STAGES",
]
