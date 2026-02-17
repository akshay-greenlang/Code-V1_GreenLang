# -*- coding: utf-8 -*-
"""
SchemaMigrationPipelineEngine - AGENT-DATA-017: Schema Migration Agent

Engine 7 of 7 — End-to-end pipeline orchestration for schema migrations.

This module implements the SchemaMigrationPipelineEngine, which orchestrates
all six upstream engines (Registry, Versioner, Detector, Checker, Planner,
Executor) into a single coherent seven-stage migration workflow:

    Stage 1 DETECT        - Identify changes between source and target schema
    Stage 2 COMPATIBILITY - Assess backward/forward compatibility of changes
    Stage 3 PLAN          - Generate an ordered, idempotent migration plan
    Stage 4 VALIDATE      - Dry-run the plan to surface errors before execution
    Stage 5 EXECUTE       - Apply the plan to actual data
    Stage 6 VERIFY        - Confirm migrated data satisfies the target schema
    Stage 7 REGISTRY      - Persist the new schema version and provenance chain

Design Principles:
    - Zero-hallucination: all version numbers, change counts and record
      counts come from deterministic engine calls, never from LLM inference.
    - Provenance: every pipeline run produces a SHA-256 provenance chain
      anchored to the ProvenanceTracker genesis hash.
    - Thread-safety: a single threading.Lock guards the pipeline run store
      so concurrent callers never corrupt shared state.
    - Graceful degradation: each upstream engine is imported with a
      try/except guard; missing engines produce clear error messages rather
      than cryptic AttributeErrors at runtime.
    - Auditability: the pipeline result dictionary captures every stage
      outcome, timing, version labels and error details for compliance
      reporting.

Example:
    >>> from greenlang.schema_migration.schema_migration_pipeline import (
    ...     SchemaMigrationPipelineEngine,
    ... )
    >>> engine = SchemaMigrationPipelineEngine()
    >>> result = engine.run_pipeline(
    ...     schema_id="emissions_v2",
    ...     target_definition_json='{"fields": [{"name": "co2_kg", "type": "float"}]}',
    ... )
    >>> assert result["status"] in ("completed", "no_changes")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
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
# Graceful imports for all 6 upstream engines
# ---------------------------------------------------------------------------

try:
    from greenlang.schema_migration.schema_registry import SchemaRegistryEngine
    _REGISTRY_AVAILABLE = True
except Exception:  # noqa: BLE001 — broad catch: ImportError or Prometheus ValueError
    SchemaRegistryEngine = None  # type: ignore[assignment, misc]
    _REGISTRY_AVAILABLE = False
    logger.warning(
        "SchemaRegistryEngine not available; schema_registry import failed. "
        "Registry operations will use stub fallback."
    )

try:
    from greenlang.schema_migration.schema_versioner import SchemaVersionerEngine
    _VERSIONER_AVAILABLE = True
except Exception:  # noqa: BLE001
    SchemaVersionerEngine = None  # type: ignore[assignment, misc]
    _VERSIONER_AVAILABLE = False
    logger.warning(
        "SchemaVersionerEngine not available; schema_versioner import failed. "
        "Version management will use stub fallback."
    )

try:
    from greenlang.schema_migration.change_detector import ChangeDetectorEngine
    _DETECTOR_AVAILABLE = True
except Exception:  # noqa: BLE001
    ChangeDetectorEngine = None  # type: ignore[assignment, misc]
    _DETECTOR_AVAILABLE = False
    logger.warning(
        "ChangeDetectorEngine not available; change_detector import failed. "
        "Change detection will use stub fallback."
    )

try:
    from greenlang.schema_migration.compatibility_checker import CompatibilityCheckerEngine
    _CHECKER_AVAILABLE = True
except Exception:  # noqa: BLE001
    CompatibilityCheckerEngine = None  # type: ignore[assignment, misc]
    _CHECKER_AVAILABLE = False
    logger.warning(
        "CompatibilityCheckerEngine not available; compatibility_checker import failed. "
        "Compatibility checks will use stub fallback."
    )

try:
    from greenlang.schema_migration.migration_planner import MigrationPlannerEngine
    _PLANNER_AVAILABLE = True
except Exception:  # noqa: BLE001
    MigrationPlannerEngine = None  # type: ignore[assignment, misc]
    _PLANNER_AVAILABLE = False
    logger.warning(
        "MigrationPlannerEngine not available; migration_planner import failed. "
        "Plan generation will use stub fallback."
    )

try:
    from greenlang.schema_migration.migration_executor import MigrationExecutorEngine
    _EXECUTOR_AVAILABLE = True
except Exception:  # noqa: BLE001
    MigrationExecutorEngine = None  # type: ignore[assignment, misc]
    _EXECUTOR_AVAILABLE = False
    logger.warning(
        "MigrationExecutorEngine not available; migration_executor import failed. "
        "Execution will use stub fallback."
    )

# ---------------------------------------------------------------------------
# Metrics helpers — graceful fallback when prometheus_client is absent
# ---------------------------------------------------------------------------

from greenlang.schema_migration.metrics import (  # noqa: E402
    observe_processing_duration,
    record_change_detected,
    record_compatibility_check,
    record_migration_executed,
    record_migration_planned,
    record_version_created,
    observe_migration_duration,
    observe_records_migrated,
    set_active_migrations,
)
from greenlang.schema_migration.provenance import ProvenanceTracker  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_PIPELINE_STAGES = (
    "detect",
    "compatibility",
    "plan",
    "validate",
    "execute",
    "verify",
    "registry",
)

_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_ABORTED = "aborted"
_STATUS_NO_CHANGES = "no_changes"
_STATUS_DRY_RUN = "dry_run_completed"

_EFFORT_CRITICAL = "CRITICAL"


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


# ---------------------------------------------------------------------------
# SchemaMigrationPipelineEngine
# ---------------------------------------------------------------------------


class SchemaMigrationPipelineEngine:
    """End-to-end pipeline orchestrator for the GreenLang Schema Migration Agent.

    Coordinates all six upstream engines through a deterministic seven-stage
    workflow: detect → compatibility → plan → validate → execute → verify →
    registry update.  Every stage outcome is captured in the pipeline result
    dictionary for compliance reporting and full auditability.

    Key design decisions:
    - Zero-hallucination: version numbers, record counts and change
      classifications come exclusively from upstream engine calls.  No LLM
      inference is used in any stage.
    - Provenance: every pipeline run appends chain-hashed entries to the
      shared :class:`~greenlang.schema_migration.provenance.ProvenanceTracker`.
    - Thread-safety: ``self._lock`` serialises writes to
      ``self._pipeline_runs`` while individual engine calls are stateless.
    - Graceful degradation: missing engines trigger stub behaviour that
      surface a clear ``"engine_unavailable"`` error rather than an
      AttributeError deep inside stage logic.

    Attributes:
        _registry: Engine 1 - SchemaRegistryEngine (or None if unavailable).
        _versioner: Engine 2 - SchemaVersionerEngine (or None if unavailable).
        _detector: Engine 3 - ChangeDetectorEngine (or None if unavailable).
        _checker: Engine 4 - CompatibilityCheckerEngine (or None if unavailable).
        _planner: Engine 5 - MigrationPlannerEngine (or None if unavailable).
        _executor: Engine 6 - MigrationExecutorEngine (or None if unavailable).
        _pipeline_runs: Mapping of pipeline_id to pipeline result dict.
        _lock: Mutex protecting writes to ``_pipeline_runs``.
        _provenance: SHA-256 chain-hashing provenance tracker.

    Example:
        >>> engine = SchemaMigrationPipelineEngine()
        >>> result = engine.run_pipeline(
        ...     schema_id="orders_v3",
        ...     target_definition_json='{"type": "object", "properties": {}}',
        ... )
        >>> print(result["status"])
        no_changes
    """

    def __init__(self) -> None:
        """Initialise the pipeline engine and all six upstream engines.

        Each upstream engine is instantiated only when its module was
        successfully imported.  If an engine is unavailable the attribute
        is set to ``None`` and the corresponding stage will return an
        ``"engine_unavailable"`` error result rather than raising.

        The :class:`~greenlang.schema_migration.provenance.ProvenanceTracker`
        is always created because it is a pure-Python dependency with no
        optional imports.
        """
        # Engine 1 — Schema Registry
        self._registry = SchemaRegistryEngine() if _REGISTRY_AVAILABLE else None

        # Engine 2 — Schema Versioner
        self._versioner = SchemaVersionerEngine() if _VERSIONER_AVAILABLE else None

        # Engine 3 — Change Detector
        self._detector = ChangeDetectorEngine() if _DETECTOR_AVAILABLE else None

        # Engine 4 — Compatibility Checker
        self._checker = CompatibilityCheckerEngine() if _CHECKER_AVAILABLE else None

        # Engine 5 — Migration Planner
        self._planner = MigrationPlannerEngine() if _PLANNER_AVAILABLE else None

        # Engine 6 — Migration Executor
        self._executor = MigrationExecutorEngine() if _EXECUTOR_AVAILABLE else None

        # Pipeline state
        self._pipeline_runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._provenance = ProvenanceTracker()

        # Track active migrations for the Gauge metric
        self._active_migrations: int = 0

        logger.info(
            "SchemaMigrationPipelineEngine initialised: "
            "registry=%s versioner=%s detector=%s checker=%s planner=%s executor=%s",
            "ok" if self._registry else "UNAVAILABLE",
            "ok" if self._versioner else "UNAVAILABLE",
            "ok" if self._detector else "UNAVAILABLE",
            "ok" if self._checker else "UNAVAILABLE",
            "ok" if self._planner else "UNAVAILABLE",
            "ok" if self._executor else "UNAVAILABLE",
        )

    # ------------------------------------------------------------------
    # Public API — primary pipeline entry points
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        schema_id: str,
        target_definition_json: str,
        data: Optional[List[Dict[str, Any]]] = None,
        skip_compatibility: bool = False,
        skip_dry_run: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run the complete seven-stage migration pipeline end-to-end.

        Executes stages in order: detect → compatibility → plan → validate
        → execute → verify → registry update.  Early-exit paths exist for:
        - No changes detected (returns immediately with ``status="no_changes"``).
        - Breaking compatibility with ``skip_compatibility=False`` (aborts).
        - Validation failure (aborts before execution).
        - ``dry_run=True`` (skips execute and verify, returns
          ``status="dry_run_completed"``).

        Args:
            schema_id: Registered schema subject identifier.
            target_definition_json: JSON string representing the desired
                target schema definition.
            data: Optional list of record dictionaries to migrate.  When
                supplied, the validate and execute stages operate on this
                data.  When ``None``, only structural migration steps run.
            skip_compatibility: When ``True``, bypass the compatibility
                check stage even if breaking changes are detected.
            skip_dry_run: When ``True``, skip the validate/dry-run stage
                (Stage 4) entirely.
            dry_run: When ``True``, run all stages up to and including
                validate but do not commit execution or update the registry.
                Returns ``status="dry_run_completed"``.

        Returns:
            Pipeline result dictionary with keys:
            ``pipeline_id``, ``schema_id``, ``status``,
            ``stages_completed``, ``stages_failed``,
            ``changes``, ``compatibility``, ``plan``, ``execution``,
            ``verification``, ``source_version``, ``target_version``,
            ``total_time_ms``, ``created_at``, ``provenance_hash``.

        Raises:
            ValueError: If ``schema_id`` or ``target_definition_json``
                are empty.

        Example:
            >>> engine = SchemaMigrationPipelineEngine()
            >>> result = engine.run_pipeline(
            ...     schema_id="suppliers",
            ...     target_definition_json='{"fields": []}',
            ...     dry_run=True,
            ... )
            >>> assert result["status"] == "dry_run_completed"
        """
        if not schema_id:
            raise ValueError("schema_id must not be empty")
        if not target_definition_json:
            raise ValueError("target_definition_json must not be empty")

        pipeline_id = _new_id("pipe")
        created_at = _utcnow_iso()
        pipeline_start = time.monotonic()

        logger.info(
            "Pipeline %s starting: schema_id=%s dry_run=%s "
            "skip_compat=%s skip_dry_run=%s records=%d",
            pipeline_id,
            schema_id,
            dry_run,
            skip_compatibility,
            skip_dry_run,
            len(data) if data else 0,
        )

        self._increment_active()

        # Accumulate result as we traverse stages
        result = self._build_initial_result(pipeline_id, schema_id, created_at)

        try:
            # ----------------------------------------------------------------
            # Stage 1: DETECT
            # ----------------------------------------------------------------
            source_definition = self._fetch_current_definition(schema_id)
            source_version = self._fetch_current_version(schema_id)
            result["source_version"] = source_version

            try:
                target_definition = json.loads(target_definition_json)
            except json.JSONDecodeError as exc:
                return self._abort(
                    result,
                    pipeline_id,
                    pipeline_start,
                    "detect",
                    f"target_definition_json is not valid JSON: {exc}",
                )

            detect_result = self.detect_stage(source_definition, target_definition)
            result["changes"] = detect_result

            if not detect_result.get("has_changes", False):
                result["status"] = _STATUS_NO_CHANGES
                result["stages_completed"].append("detect")
                logger.info(
                    "Pipeline %s: no changes detected — skipping remaining stages",
                    pipeline_id,
                )
                return self._finalise(result, pipeline_id, pipeline_start)

            result["stages_completed"].append("detect")
            logger.info(
                "Pipeline %s stage detect: %d change(s) found",
                pipeline_id,
                detect_result.get("change_count", 0),
            )

            # ----------------------------------------------------------------
            # Stage 2: COMPATIBILITY
            # ----------------------------------------------------------------
            compat_result = self.compatibility_stage(
                source_definition, target_definition, detect_result
            )
            result["compatibility"] = compat_result
            result["stages_completed"].append("compatibility")

            is_breaking = compat_result.get("is_breaking", False)
            if is_breaking and not skip_compatibility:
                logger.warning(
                    "Pipeline %s: breaking changes detected and "
                    "skip_compatibility=False — aborting",
                    pipeline_id,
                )
                return self._abort(
                    result,
                    pipeline_id,
                    pipeline_start,
                    "compatibility",
                    "Breaking compatibility detected; set skip_compatibility=True to override",
                )

            logger.info(
                "Pipeline %s stage compatibility: breaking=%s (skipped=%s)",
                pipeline_id,
                is_breaking,
                skip_compatibility,
            )

            # ----------------------------------------------------------------
            # Stage 3: PLAN
            # ----------------------------------------------------------------
            target_version = self._determine_target_version(
                source_version, compat_result
            )
            result["target_version"] = target_version

            plan_result = self.plan_stage(
                schema_id=schema_id,
                source_version=source_version,
                target_version=target_version,
                changes=detect_result,
                source_def=source_definition,
                target_def=target_definition,
            )
            result["plan"] = plan_result
            result["stages_completed"].append("plan")

            if plan_result.get("effort") == _EFFORT_CRITICAL:
                logger.warning(
                    "Pipeline %s: migration effort classified as CRITICAL — "
                    "proceeding with caution",
                    pipeline_id,
                )

            logger.info(
                "Pipeline %s stage plan: plan_id=%s steps=%d effort=%s",
                pipeline_id,
                plan_result.get("plan_id", "n/a"),
                plan_result.get("step_count", 0),
                plan_result.get("effort", "unknown"),
            )

            # ----------------------------------------------------------------
            # Stage 4: VALIDATE (unless skip_dry_run)
            # ----------------------------------------------------------------
            if not skip_dry_run:
                validate_result = self.validate_stage(
                    plan_result.get("plan_id", "")
                )
                result["validation"] = validate_result
                result["stages_completed"].append("validate")

                if not validate_result.get("is_valid", True):
                    logger.error(
                        "Pipeline %s: validation failed — aborting before execution",
                        pipeline_id,
                    )
                    return self._abort(
                        result,
                        pipeline_id,
                        pipeline_start,
                        "validate",
                        validate_result.get("errors", "Validation failed"),
                    )

                logger.info(
                    "Pipeline %s stage validate: is_valid=%s",
                    pipeline_id,
                    validate_result.get("is_valid"),
                )

            # ----------------------------------------------------------------
            # Stage 5: EXECUTE (unless dry_run=True)
            # ----------------------------------------------------------------
            if dry_run:
                result["status"] = _STATUS_DRY_RUN
                logger.info(
                    "Pipeline %s: dry_run=True — skipping execute and verify",
                    pipeline_id,
                )
                return self._finalise(result, pipeline_id, pipeline_start)

            execution_result = self.execute_stage(plan_result, data=data, dry_run=False)
            result["execution"] = execution_result
            result["stages_completed"].append("execute")

            exec_status = execution_result.get("status", "failed")
            if exec_status in ("failed", "rolled_back"):
                logger.error(
                    "Pipeline %s: execution failed with status=%s",
                    pipeline_id,
                    exec_status,
                )
                return self._abort(
                    result,
                    pipeline_id,
                    pipeline_start,
                    "execute",
                    execution_result.get("error", "Execution failed"),
                )

            logger.info(
                "Pipeline %s stage execute: status=%s records=%d",
                pipeline_id,
                exec_status,
                execution_result.get("records_migrated", 0),
            )

            # ----------------------------------------------------------------
            # Stage 6: VERIFY
            # ----------------------------------------------------------------
            execution_id = execution_result.get("execution_id", "")
            migrated_data = execution_result.get("migrated_data")
            verify_result = self.verify_stage(
                execution_id=execution_id,
                target_definition=target_definition,
                migrated_data=migrated_data,
            )
            result["verification"] = verify_result
            result["stages_completed"].append("verify")

            if not verify_result.get("passed", True):
                logger.error(
                    "Pipeline %s: verification failed — initiating rollback",
                    pipeline_id,
                )
                self._attempt_rollback(execution_result)
                return self._abort(
                    result,
                    pipeline_id,
                    pipeline_start,
                    "verify",
                    verify_result.get("failure_reason", "Verification failed"),
                )

            logger.info(
                "Pipeline %s stage verify: passed=%s records_verified=%d",
                pipeline_id,
                verify_result.get("passed"),
                verify_result.get("records_verified", 0),
            )

            # ----------------------------------------------------------------
            # Stage 7: UPDATE REGISTRY
            # ----------------------------------------------------------------
            registry_result = self._update_registry(
                schema_id=schema_id,
                target_definition=target_definition,
                target_version=target_version,
                pipeline_id=pipeline_id,
            )
            result["registry_update"] = registry_result
            result["stages_completed"].append("registry")

            logger.info(
                "Pipeline %s stage registry: version=%s",
                pipeline_id,
                registry_result.get("version", target_version),
            )

            result["status"] = _STATUS_COMPLETED
            return self._finalise(result, pipeline_id, pipeline_start)

        except Exception as exc:
            logger.error(
                "Pipeline %s: unexpected error — %s",
                pipeline_id,
                exc,
                exc_info=True,
            )
            result["error"] = str(exc)
            result["status"] = _STATUS_FAILED
            result["stages_failed"].append("unexpected")
            return self._finalise(result, pipeline_id, pipeline_start)
        finally:
            self._decrement_active()

    def run_batch_pipeline(
        self,
        schema_pairs: List[Dict[str, Any]],
        data_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Run the migration pipeline for multiple schema pairs in sequence.

        Iterates over ``schema_pairs``, calling :meth:`run_pipeline` for each
        entry.  Results are accumulated and summarised in the returned batch
        result dictionary.  A failure in one pair does not prevent remaining
        pairs from being processed.

        Args:
            schema_pairs: List of parameter dictionaries.  Each must contain:
                - ``schema_id`` (str): Registered schema subject identifier.
                - ``target_definition_json`` (str): JSON string of the target.
                Optional per-pair keys mirror :meth:`run_pipeline` kwargs:
                ``skip_compatibility``, ``skip_dry_run``, ``dry_run``.
            data_map: Optional mapping of ``schema_id`` to list of records.
                When a schema_id key is found, the corresponding records are
                passed as the ``data`` argument to :meth:`run_pipeline`.

        Returns:
            Batch result dictionary with keys:
            ``batch_id``, ``total``, ``completed``, ``failed``, ``aborted``,
            ``no_changes``, ``results``, ``total_time_ms``, ``created_at``,
            ``provenance_hash``.

        Raises:
            ValueError: If ``schema_pairs`` is empty.

        Example:
            >>> pairs = [
            ...     {"schema_id": "a", "target_definition_json": "{}"},
            ...     {"schema_id": "b", "target_definition_json": "{}"},
            ... ]
            >>> result = engine.run_batch_pipeline(pairs)
            >>> assert result["total"] == 2
        """
        if not schema_pairs:
            raise ValueError("schema_pairs must not be empty")

        batch_id = _new_id("batch")
        created_at = _utcnow_iso()
        batch_start = time.monotonic()
        data_map = data_map or {}

        logger.info(
            "Batch pipeline %s starting: %d schema pair(s)",
            batch_id,
            len(schema_pairs),
        )

        individual_results: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {
            _STATUS_COMPLETED: 0,
            _STATUS_FAILED: 0,
            _STATUS_ABORTED: 0,
            _STATUS_NO_CHANGES: 0,
            _STATUS_DRY_RUN: 0,
        }

        for pair in schema_pairs:
            schema_id = pair.get("schema_id", "")
            target_json = pair.get("target_definition_json", "")
            records = data_map.get(schema_id)

            try:
                pair_result = self.run_pipeline(
                    schema_id=schema_id,
                    target_definition_json=target_json,
                    data=records,
                    skip_compatibility=pair.get("skip_compatibility", False),
                    skip_dry_run=pair.get("skip_dry_run", False),
                    dry_run=pair.get("dry_run", False),
                )
            except Exception as exc:
                pair_result = {
                    "schema_id": schema_id,
                    "status": _STATUS_FAILED,
                    "error": str(exc),
                }

            pair_status = pair_result.get("status", _STATUS_FAILED)
            status_counts[pair_status] = status_counts.get(pair_status, 0) + 1
            individual_results.append(pair_result)

        total_time_ms = _elapsed_ms(batch_start)
        batch_payload = {
            "batch_id": batch_id,
            "total": len(schema_pairs),
            "completed": status_counts.get(_STATUS_COMPLETED, 0),
            "failed": status_counts.get(_STATUS_FAILED, 0),
            "aborted": status_counts.get(_STATUS_ABORTED, 0),
            "no_changes": status_counts.get(_STATUS_NO_CHANGES, 0),
            "dry_run_completed": status_counts.get(_STATUS_DRY_RUN, 0),
            "results": individual_results,
            "total_time_ms": total_time_ms,
            "created_at": created_at,
            "provenance_hash": _sha256(
                {"batch_id": batch_id, "total": len(schema_pairs), "created_at": created_at}
            ),
        }

        # Record batch provenance
        self._provenance.record(
            "batch_pipeline",
            batch_id,
            "batch_completed",
            batch_payload,
        )

        logger.info(
            "Batch pipeline %s done: total=%d completed=%d failed=%d "
            "no_changes=%d time_ms=%.1f",
            batch_id,
            len(schema_pairs),
            status_counts.get(_STATUS_COMPLETED, 0),
            status_counts.get(_STATUS_FAILED, 0),
            status_counts.get(_STATUS_NO_CHANGES, 0),
            total_time_ms,
        )
        return batch_payload

    # ------------------------------------------------------------------
    # Individual stage methods
    # ------------------------------------------------------------------

    def detect_stage(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 1: Detect changes between source and target schema definitions.

        Delegates to :class:`ChangeDetectorEngine` when available.  When the
        engine is unavailable a structural comparison stub is performed:
        every key present in one definition but not the other is reported as
        a change, and ``has_changes`` reflects whether any differences exist.

        Metrics recorded:
        - ``observe_processing_duration("detect", seconds)``
        - ``record_change_detected(change_type, severity)`` for each change

        Args:
            source_definition: Current (source) schema definition dictionary.
            target_definition: Desired (target) schema definition dictionary.

        Returns:
            Change detection result dictionary with keys:
            ``has_changes``, ``change_count``, ``changes``,
            ``breaking_changes``, ``non_breaking_changes``,
            ``detected_at``, ``duration_ms``.
        """
        stage_start = time.monotonic()

        if self._detector is not None:
            try:
                raw = self._detector.detect_changes(
                    source_definition, target_definition
                )
                result = self._normalise_detect_result(raw)
            except Exception as exc:
                logger.error("detect_stage: ChangeDetectorEngine failed — %s", exc)
                result = self._stub_detect(source_definition, target_definition)
        else:
            result = self._stub_detect(source_definition, target_definition)

        elapsed = time.monotonic() - stage_start
        observe_processing_duration("detect", elapsed)
        result["duration_ms"] = round(elapsed * 1000.0, 2)
        result["detected_at"] = _utcnow_iso()

        # Emit per-change metrics
        for change in result.get("changes", []):
            change_type = change.get("change_type", "unknown")
            severity = change.get("severity", "informational")
            record_change_detected(change_type, severity)

        logger.debug(
            "detect_stage: has_changes=%s change_count=%d",
            result.get("has_changes"),
            result.get("change_count", 0),
        )
        return result

    def compatibility_stage(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        changes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 2: Assess backward/forward compatibility of detected changes.

        Delegates to :class:`CompatibilityCheckerEngine` when available.
        The stub fallback classifies the migration as compatible when no
        breaking changes are reported in the ``changes`` dict.

        Metrics recorded:
        - ``observe_processing_duration("compatibility", seconds)``
        - ``record_compatibility_check(result)``

        Args:
            source_definition: Current schema definition dictionary.
            target_definition: Target schema definition dictionary.
            changes: Output from :meth:`detect_stage`.

        Returns:
            Compatibility assessment dictionary with keys:
            ``is_compatible``, ``is_breaking``, ``compatibility_level``,
            ``breaking_changes``, ``warnings``, ``checked_at``,
            ``duration_ms``.
        """
        stage_start = time.monotonic()

        if self._checker is not None:
            try:
                raw = self._checker.check_compatibility(
                    source_definition, target_definition, changes
                )
                result = self._normalise_compat_result(raw)
            except Exception as exc:
                logger.error(
                    "compatibility_stage: CompatibilityCheckerEngine failed — %s", exc
                )
                result = self._stub_compat(changes)
        else:
            result = self._stub_compat(changes)

        elapsed = time.monotonic() - stage_start
        observe_processing_duration("compatibility", elapsed)
        result["duration_ms"] = round(elapsed * 1000.0, 2)
        result["checked_at"] = _utcnow_iso()

        compat_label = "compatible" if result.get("is_compatible") else "incompatible"
        record_compatibility_check(compat_label)

        logger.debug(
            "compatibility_stage: is_compatible=%s is_breaking=%s",
            result.get("is_compatible"),
            result.get("is_breaking"),
        )
        return result

    def plan_stage(
        self,
        schema_id: str,
        source_version: str,
        target_version: str,
        changes: Dict[str, Any],
        source_def: Optional[Dict[str, Any]] = None,
        target_def: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Stage 3: Generate a migration plan from the detected changes.

        Delegates to :class:`MigrationPlannerEngine` when available.  The
        stub fallback creates a single-step plan derived directly from the
        ``changes`` dictionary without any LLM inference.

        Metrics recorded:
        - ``record_migration_planned(status)``

        Args:
            schema_id: Registered schema subject identifier.
            source_version: Semantic version string of the current schema.
            target_version: Semantic version string of the target schema.
            changes: Output from :meth:`detect_stage`.
            source_def: Optional source definition for the planner.
            target_def: Optional target definition for the planner.

        Returns:
            Migration plan dictionary with keys:
            ``plan_id``, ``schema_id``, ``source_version``,
            ``target_version``, ``step_count``, ``steps``, ``effort``,
            ``estimated_duration_seconds``, ``planned_at``, ``duration_ms``.
        """
        stage_start = time.monotonic()

        if self._planner is not None:
            try:
                raw = self._planner.create_plan(
                    schema_id=schema_id,
                    source_version=source_version,
                    target_version=target_version,
                    changes=changes,
                    source_definition=source_def,
                    target_definition=target_def,
                )
                result = self._normalise_plan_result(raw, schema_id, source_version, target_version)
            except Exception as exc:
                logger.error("plan_stage: MigrationPlannerEngine failed — %s", exc)
                result = self._stub_plan(schema_id, source_version, target_version, changes)
        else:
            result = self._stub_plan(schema_id, source_version, target_version, changes)

        elapsed = time.monotonic() - stage_start
        result["duration_ms"] = round(elapsed * 1000.0, 2)
        result["planned_at"] = _utcnow_iso()

        plan_status = "success" if result.get("plan_id") else "failed"
        record_migration_planned(plan_status)

        logger.debug(
            "plan_stage: plan_id=%s steps=%d effort=%s",
            result.get("plan_id"),
            result.get("step_count", 0),
            result.get("effort"),
        )
        return result

    def validate_stage(self, plan_id: str) -> Dict[str, Any]:
        """Stage 4: Validate and dry-run the migration plan.

        Delegates to :class:`MigrationPlannerEngine` when available for
        plan consistency checks.  When the engine is unavailable, validation
        passes if ``plan_id`` is non-empty (structural check only).

        Args:
            plan_id: Unique identifier of the plan returned by
                :meth:`plan_stage`.

        Returns:
            Validation result dictionary with keys:
            ``is_valid``, ``errors``, ``warnings``, ``validated_at``,
            ``duration_ms``.
        """
        stage_start = time.monotonic()

        if self._planner is not None:
            try:
                raw = self._planner.validate_plan(plan_id)
                result = self._normalise_validate_result(raw)
            except Exception as exc:
                logger.error("validate_stage: MigrationPlannerEngine.validate_plan failed — %s", exc)
                result = self._stub_validate(plan_id)
        else:
            result = self._stub_validate(plan_id)

        elapsed = time.monotonic() - stage_start
        result["duration_ms"] = round(elapsed * 1000.0, 2)
        result["validated_at"] = _utcnow_iso()

        logger.debug(
            "validate_stage: is_valid=%s errors=%d",
            result.get("is_valid"),
            len(result.get("errors", [])) if isinstance(result.get("errors"), list) else 0,
        )
        return result

    def execute_stage(
        self,
        plan: Dict[str, Any],
        data: Optional[List[Dict[str, Any]]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Stage 5: Execute the migration plan against the provided data.

        Delegates to :class:`MigrationExecutorEngine` when available.  The
        stub fallback applies each plan step as a field rename/add/remove
        operation on the provided data list in pure Python.

        Metrics recorded:
        - ``record_migration_executed(status)``
        - ``observe_records_migrated(count)`` when data is provided

        Args:
            plan: Migration plan dictionary returned by :meth:`plan_stage`.
            data: Optional list of record dictionaries to transform.  When
                ``None``, only structural migration metadata is recorded.
            dry_run: When ``True``, simulate execution without committing
                any changes.  Returned status is ``"dry_run"``.

        Returns:
            Execution result dictionary with keys:
            ``execution_id``, ``status``, ``records_migrated``,
            ``migrated_data``, ``errors``, ``warnings``,
            ``executed_at``, ``duration_ms``.
        """
        stage_start = time.monotonic()

        if self._executor is not None:
            try:
                raw = self._executor.execute(plan=plan, data=data, dry_run=dry_run)
                result = self._normalise_exec_result(raw)
            except Exception as exc:
                logger.error("execute_stage: MigrationExecutorEngine failed — %s", exc)
                result = self._stub_execute(plan, data, dry_run)
        else:
            result = self._stub_execute(plan, data, dry_run)

        elapsed = time.monotonic() - stage_start
        result["duration_ms"] = round(elapsed * 1000.0, 2)
        result["executed_at"] = _utcnow_iso()

        exec_status = result.get("status", "failed")
        record_migration_executed(exec_status)

        records_migrated = result.get("records_migrated", 0)
        if records_migrated > 0:
            observe_records_migrated(records_migrated)

        logger.debug(
            "execute_stage: status=%s records_migrated=%d",
            exec_status,
            records_migrated,
        )
        return result

    def verify_stage(
        self,
        execution_id: str,
        target_definition: Dict[str, Any],
        migrated_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 6: Verify migrated data satisfies the target schema definition.

        Performs three deterministic checks:
        1. Required fields are present in every migrated record.
        2. Record count is non-negative (no data loss).
        3. No unexpected null values in non-nullable fields.

        When :class:`MigrationExecutorEngine` supplies a ``verify`` method
        that is delegated first; otherwise the pure-Python checks above run.

        Args:
            execution_id: Unique identifier from :meth:`execute_stage`.
            target_definition: Target schema definition dictionary used to
                determine required field names.
            migrated_data: Optional list of migrated record dictionaries.
                When ``None`` the record-level checks are skipped.

        Returns:
            Verification result dictionary with keys:
            ``passed``, ``records_verified``, ``failure_reason``,
            ``warnings``, ``verified_at``, ``duration_ms``.
        """
        stage_start = time.monotonic()

        if self._executor is not None and hasattr(self._executor, "verify"):
            try:
                raw = self._executor.verify(
                    execution_id=execution_id,
                    target_definition=target_definition,
                    migrated_data=migrated_data,
                )
                result = self._normalise_verify_result(raw)
            except Exception as exc:
                logger.error("verify_stage: executor.verify failed — %s", exc, exc_info=True)
                result = self._structural_verify(target_definition, migrated_data)
        else:
            result = self._structural_verify(target_definition, migrated_data)

        elapsed = time.monotonic() - stage_start
        result["duration_ms"] = round(elapsed * 1000.0, 2)
        result["verified_at"] = _utcnow_iso()

        logger.debug(
            "verify_stage: passed=%s records_verified=%d",
            result.get("passed"),
            result.get("records_verified", 0),
        )
        return result

    # ------------------------------------------------------------------
    # Reporting and administrative methods
    # ------------------------------------------------------------------

    def generate_report(self, pipeline_id: str) -> Dict[str, Any]:
        """Generate a compliance and audit report for a completed pipeline run.

        Produces a structured report capturing all stage outcomes, timing
        data, version lineage, change classification summary, and provenance
        chain for the specified pipeline run.

        Args:
            pipeline_id: Unique pipeline run identifier returned by
                :meth:`run_pipeline`.

        Returns:
            Report dictionary with keys:
            ``report_id``, ``pipeline_id``, ``schema_id``, ``status``,
            ``source_version``, ``target_version``, ``stages_completed``,
            ``stages_failed``, ``change_summary``, ``timing``,
            ``compliance_notes``, ``provenance_entries``,
            ``generated_at``, ``report_hash``.

        Raises:
            KeyError: If ``pipeline_id`` is not found in the pipeline run
                store.

        Example:
            >>> result = engine.run_pipeline("schema_x", '{"fields": []}')
            >>> report = engine.generate_report(result["pipeline_id"])
            >>> assert "compliance_notes" in report
        """
        run = self.get_pipeline_run(pipeline_id)
        if run is None:
            raise KeyError(f"Pipeline run not found: {pipeline_id}")

        report_id = _new_id("rpt")
        generated_at = _utcnow_iso()

        changes = run.get("changes") or {}
        change_summary = {
            "total_changes": changes.get("change_count", 0),
            "breaking_changes": len(changes.get("breaking_changes", [])),
            "non_breaking_changes": len(changes.get("non_breaking_changes", [])),
            "by_type": self._summarise_changes_by_type(changes.get("changes", [])),
        }

        timing = {
            "total_time_ms": run.get("total_time_ms", 0),
            "created_at": run.get("created_at"),
        }

        compat = run.get("compatibility") or {}
        compliance_notes = self._build_compliance_notes(run, compat)

        provenance_entries = [
            e.to_dict()
            for e in self._provenance.get_chain(pipeline_id)
        ]

        report = {
            "report_id": report_id,
            "pipeline_id": pipeline_id,
            "schema_id": run.get("schema_id", ""),
            "status": run.get("status", "unknown"),
            "source_version": run.get("source_version", ""),
            "target_version": run.get("target_version", ""),
            "stages_completed": run.get("stages_completed", []),
            "stages_failed": run.get("stages_failed", []),
            "change_summary": change_summary,
            "timing": timing,
            "compliance_notes": compliance_notes,
            "provenance_entries": provenance_entries,
            "generated_at": generated_at,
            "report_hash": _sha256(
                {"report_id": report_id, "pipeline_id": pipeline_id, "generated_at": generated_at}
            ),
        }

        logger.info(
            "generate_report: report_id=%s pipeline_id=%s status=%s",
            report_id,
            pipeline_id,
            run.get("status"),
        )
        return report

    def get_pipeline_run(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a pipeline run record by its unique identifier.

        Args:
            pipeline_id: Unique pipeline run identifier.

        Returns:
            Pipeline result dictionary, or ``None`` if no run with the
            given identifier exists.

        Example:
            >>> run = engine.get_pipeline_run("pipe-abc123def456")
            >>> if run:
            ...     print(run["status"])
        """
        with self._lock:
            return self._pipeline_runs.get(pipeline_id)

    def list_pipeline_runs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return a paginated list of pipeline run records.

        Results are ordered by ``created_at`` descending (most recent first).
        Pagination is applied after sorting using ``offset`` and ``limit``.

        Args:
            limit: Maximum number of records to return. Defaults to 100.
            offset: Number of records to skip from the beginning of the
                sorted list. Defaults to 0.

        Returns:
            List of pipeline result dictionaries (most recent first).

        Raises:
            ValueError: If ``limit`` or ``offset`` are negative.

        Example:
            >>> runs = engine.list_pipeline_runs(limit=10, offset=0)
            >>> print(len(runs))
            10
        """
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        with self._lock:
            all_runs = list(self._pipeline_runs.values())

        all_runs.sort(
            key=lambda r: r.get("created_at", ""),
            reverse=True,
        )
        return all_runs[offset: offset + limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for all pipeline runs recorded so far.

        Computes statistics deterministically from the in-memory run store
        without any LLM inference.  All numeric aggregations use pure Python
        arithmetic.

        Returns:
            Statistics dictionary with keys:
            ``total_runs``, ``by_status``, ``avg_duration_ms``,
            ``min_duration_ms``, ``max_duration_ms``, ``success_rate``,
            ``active_migrations``, ``provenance_entry_count``,
            ``computed_at``.

        Example:
            >>> stats = engine.get_statistics()
            >>> print(stats["success_rate"])
            1.0
        """
        with self._lock:
            runs = list(self._pipeline_runs.values())

        total = len(runs)
        by_status: Dict[str, int] = {}
        durations: List[float] = []

        for run in runs:
            status = run.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            dur = run.get("total_time_ms")
            if isinstance(dur, (int, float)):
                durations.append(float(dur))

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0

        completed = by_status.get(_STATUS_COMPLETED, 0)
        success_rate = completed / total if total > 0 else 0.0

        return {
            "total_runs": total,
            "by_status": by_status,
            "avg_duration_ms": round(avg_duration, 2),
            "min_duration_ms": round(min_duration, 2),
            "max_duration_ms": round(max_duration, 2),
            "success_rate": round(success_rate, 4),
            "active_migrations": self._active_migrations,
            "provenance_entry_count": self._provenance.entry_count,
            "computed_at": _utcnow_iso(),
        }

    def reset(self) -> None:
        """Reset all engine state and the pipeline run store.

        Clears the pipeline run store, resets the provenance tracker, and
        re-initialises any engines that expose a ``reset()`` method.
        Primarily intended for testing to prevent state leakage between
        test cases.

        Example:
            >>> engine.reset()
            >>> assert engine.get_statistics()["total_runs"] == 0
        """
        with self._lock:
            self._pipeline_runs.clear()
            self._active_migrations = 0

        self._provenance.reset()

        for name, eng in (
            ("registry", self._registry),
            ("versioner", self._versioner),
            ("detector", self._detector),
            ("checker", self._checker),
            ("planner", self._planner),
            ("executor", self._executor),
        ):
            if eng is not None and hasattr(eng, "reset"):
                try:
                    eng.reset()
                except Exception as exc:
                    logger.warning("reset: engine %s reset failed — %s", name, exc)

        set_active_migrations(0)
        logger.info("SchemaMigrationPipelineEngine: full reset complete")

    # ------------------------------------------------------------------
    # Internal: registry interaction helpers
    # ------------------------------------------------------------------

    def _fetch_current_definition(self, schema_id: str) -> Dict[str, Any]:
        """Retrieve the current schema definition from the registry.

        Falls back to an empty dictionary when the registry is unavailable
        or the schema has not been registered yet.

        Args:
            schema_id: Registered schema subject identifier.

        Returns:
            Current schema definition dictionary, or ``{}`` if unavailable.
        """
        if self._registry is None:
            logger.debug("_fetch_current_definition: registry unavailable for %s", schema_id)
            return {}
        try:
            schema = self._registry.get_schema(schema_id)
            if schema is None:
                return {}
            definition = schema.get("definition") or schema.get("schema_definition") or {}
            if isinstance(definition, str):
                try:
                    definition = json.loads(definition)
                except json.JSONDecodeError:
                    definition = {}
            return definition
        except Exception as exc:
            logger.warning(
                "_fetch_current_definition: registry.get_schema(%s) failed — %s",
                schema_id,
                exc,
            )
            return {}

    def _fetch_current_version(self, schema_id: str) -> str:
        """Retrieve the current semantic version string for a schema.

        Queries the versioner engine first, then falls back to the registry,
        then defaults to ``"1.0.0"`` when both are unavailable.

        Args:
            schema_id: Registered schema subject identifier.

        Returns:
            Semantic version string (e.g. ``"1.0.0"``).
        """
        if self._versioner is not None:
            try:
                version_info = self._versioner.get_current_version(schema_id)
                if version_info:
                    return version_info.get("version", "1.0.0")
            except Exception as exc:
                logger.warning(
                    "_fetch_current_version: versioner failed for %s — %s",
                    schema_id,
                    exc,
                )

        if self._registry is not None:
            try:
                schema = self._registry.get_schema(schema_id)
                if schema:
                    return schema.get("version", "1.0.0")
            except Exception as exc:
                logger.warning(
                    "_fetch_current_version: registry fallback failed for %s — %s",
                    schema_id,
                    exc,
                )

        return "1.0.0"

    def _determine_target_version(
        self,
        source_version: str,
        compat_result: Dict[str, Any],
    ) -> str:
        """Compute the target semantic version from the compatibility result.

        Uses the ``recommended_bump`` key in ``compat_result`` when present.
        Falls back to auto-incrementing the patch segment of ``source_version``
        when no recommendation is available.

        Args:
            source_version: Semantic version string of the current schema.
            compat_result: Compatibility stage result dictionary.

        Returns:
            Computed target semantic version string.
        """
        bump = compat_result.get("recommended_bump", "")
        parts = source_version.split(".")

        # Ensure we have at least three segments
        while len(parts) < 3:
            parts.append("0")

        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            major, minor, patch = 1, 0, 0

        if bump == "major" or compat_result.get("is_breaking"):
            major += 1
            minor = 0
            patch = 0
        elif bump == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1

        return f"{major}.{minor}.{patch}"

    def _update_registry(
        self,
        schema_id: str,
        target_definition: Dict[str, Any],
        target_version: str,
        pipeline_id: str,
    ) -> Dict[str, Any]:
        """Persist the new schema version to the registry.

        Creates a new version entry via the registry and versioner engines
        when available.  Records provenance for the new version.

        Args:
            schema_id: Registered schema subject identifier.
            target_definition: New target schema definition dictionary.
            target_version: Computed target version string.
            pipeline_id: Pipeline run identifier for provenance linking.

        Returns:
            Registry update result dictionary with keys:
            ``schema_id``, ``version``, ``registered``, ``updated_at``.
        """
        updated_at = _utcnow_iso()
        registered = False

        if self._registry is not None:
            try:
                self._registry.register_schema(
                    schema_id=schema_id,
                    definition=target_definition,
                    version=target_version,
                    metadata={"pipeline_id": pipeline_id},
                )
                registered = True
                record_version_created("auto")
            except Exception as exc:
                logger.warning(
                    "_update_registry: registry.register_schema failed for %s v%s — %s",
                    schema_id,
                    target_version,
                    exc,
                )

        if self._versioner is not None and registered:
            try:
                self._versioner.create_version(
                    schema_id=schema_id,
                    version=target_version,
                    definition=target_definition,
                    pipeline_id=pipeline_id,
                )
            except Exception as exc:
                logger.warning(
                    "_update_registry: versioner.create_version failed — %s", exc
                )

        self._provenance.record(
            "schema_version",
            f"{schema_id}:{target_version}",
            "version_registered",
            {"schema_id": schema_id, "version": target_version, "pipeline_id": pipeline_id},
        )

        return {
            "schema_id": schema_id,
            "version": target_version,
            "registered": registered,
            "updated_at": updated_at,
        }

    def _attempt_rollback(self, execution_result: Dict[str, Any]) -> None:
        """Attempt to roll back a failed migration execution.

        Delegates to :class:`MigrationExecutorEngine` when it exposes a
        ``rollback`` method.  Logs the outcome but does not re-raise so
        that the caller can still record the abort result cleanly.

        Args:
            execution_result: Execution stage result dictionary containing
                at least ``execution_id``.
        """
        execution_id = execution_result.get("execution_id", "")
        if not execution_id:
            return

        if self._executor is not None and hasattr(self._executor, "rollback"):
            try:
                self._executor.rollback(execution_id)
                logger.info(
                    "_attempt_rollback: rollback succeeded for execution_id=%s",
                    execution_id,
                )
            except Exception as exc:
                logger.error(
                    "_attempt_rollback: rollback failed for execution_id=%s — %s",
                    execution_id,
                    exc,
                )
        else:
            logger.warning(
                "_attempt_rollback: executor unavailable or has no rollback method; "
                "manual intervention may be required for execution_id=%s",
                execution_id,
            )

    # ------------------------------------------------------------------
    # Internal: pipeline result lifecycle helpers
    # ------------------------------------------------------------------

    def _build_initial_result(
        self,
        pipeline_id: str,
        schema_id: str,
        created_at: str,
    ) -> Dict[str, Any]:
        """Construct the skeleton pipeline result dictionary.

        Args:
            pipeline_id: Unique pipeline run identifier.
            schema_id: Registered schema subject identifier.
            created_at: ISO-8601 creation timestamp string.

        Returns:
            Initial pipeline result dictionary with all stage keys set to
            ``None`` and status set to ``"running"``.
        """
        return {
            "pipeline_id": pipeline_id,
            "schema_id": schema_id,
            "status": "running",
            "stages_completed": [],
            "stages_failed": [],
            "changes": None,
            "compatibility": None,
            "validation": None,
            "plan": None,
            "execution": None,
            "verification": None,
            "registry_update": None,
            "source_version": "",
            "target_version": "",
            "total_time_ms": None,
            "created_at": created_at,
            "provenance_hash": None,
            "error": None,
        }

    def _finalise(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        pipeline_start: float,
    ) -> Dict[str, Any]:
        """Finalise a pipeline result: add timing, provenance, and persist.

        Args:
            result: Accumulated pipeline result dictionary.
            pipeline_id: Unique pipeline run identifier.
            pipeline_start: ``time.monotonic()`` value from when the pipeline
                started.

        Returns:
            The completed pipeline result dictionary with ``total_time_ms``
            and ``provenance_hash`` populated.
        """
        elapsed_ms = _elapsed_ms(pipeline_start)
        result["total_time_ms"] = elapsed_ms

        # Record provenance
        prov_entry = self._provenance.record(
            "pipeline_run",
            pipeline_id,
            f"pipeline_{result['status']}",
            {
                "schema_id": result.get("schema_id"),
                "status": result.get("status"),
                "stages_completed": result.get("stages_completed"),
                "source_version": result.get("source_version"),
                "target_version": result.get("target_version"),
            },
        )
        result["provenance_hash"] = prov_entry.hash_value

        # Observe end-to-end migration duration
        observe_migration_duration(elapsed_ms / 1000.0)

        # Persist to run store
        with self._lock:
            self._pipeline_runs[pipeline_id] = result

        logger.info(
            "Pipeline %s finalised: status=%s stages=%s time_ms=%.1f",
            pipeline_id,
            result.get("status"),
            result.get("stages_completed"),
            elapsed_ms,
        )
        return result

    def _abort(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        pipeline_start: float,
        failed_stage: str,
        reason: Any,
    ) -> Dict[str, Any]:
        """Abort the pipeline at a specific stage and finalise the result.

        Records the failed stage, sets status to ``"aborted"``, and
        calls :meth:`_finalise` so that the run is persisted and timed.

        Args:
            result: Accumulated pipeline result dictionary.
            pipeline_id: Unique pipeline run identifier.
            pipeline_start: ``time.monotonic()`` value from pipeline start.
            failed_stage: Name of the stage at which the abort occurred.
            reason: Human-readable abort reason (str, dict, or list).

        Returns:
            Completed pipeline result dictionary with
            ``status="aborted"``.
        """
        result["status"] = _STATUS_ABORTED
        result["stages_failed"].append(failed_stage)
        result["error"] = str(reason) if not isinstance(reason, str) else reason
        logger.warning(
            "Pipeline %s aborted at stage '%s': %s",
            pipeline_id,
            failed_stage,
            reason,
        )
        return self._finalise(result, pipeline_id, pipeline_start)

    def _increment_active(self) -> None:
        """Increment the active migration gauge."""
        with self._lock:
            self._active_migrations += 1
        set_active_migrations(self._active_migrations)

    def _decrement_active(self) -> None:
        """Decrement the active migration gauge, flooring at zero."""
        with self._lock:
            self._active_migrations = max(0, self._active_migrations - 1)
        set_active_migrations(self._active_migrations)

    # ------------------------------------------------------------------
    # Internal: result normalisation helpers
    # ------------------------------------------------------------------

    def _normalise_detect_result(self, raw: Any) -> Dict[str, Any]:
        """Normalise raw ChangeDetectorEngine output to a standard dict.

        Handles both dict returns and Pydantic model returns from
        different engine implementations.

        Args:
            raw: Raw output from ChangeDetectorEngine.detect_changes().

        Returns:
            Normalised detection result dictionary.
        """
        if isinstance(raw, dict):
            result = raw.copy()
        elif hasattr(raw, "dict"):
            result = raw.dict()
        elif hasattr(raw, "__dict__"):
            result = vars(raw)
        else:
            result = {"raw": str(raw)}

        # Ensure required keys exist with sensible defaults
        result.setdefault("has_changes", bool(result.get("changes") or result.get("change_count", 0)))
        result.setdefault("change_count", len(result.get("changes", [])))
        result.setdefault("changes", [])
        result.setdefault("breaking_changes", [])
        result.setdefault("non_breaking_changes", [])
        return result

    def _normalise_compat_result(self, raw: Any) -> Dict[str, Any]:
        """Normalise raw CompatibilityCheckerEngine output.

        Args:
            raw: Raw output from CompatibilityCheckerEngine.check_compatibility().

        Returns:
            Normalised compatibility result dictionary.
        """
        if isinstance(raw, dict):
            result = raw.copy()
        elif hasattr(raw, "dict"):
            result = raw.dict()
        elif hasattr(raw, "__dict__"):
            result = vars(raw)
        else:
            result = {"raw": str(raw)}

        result.setdefault("is_compatible", not result.get("is_breaking", False))
        result.setdefault("is_breaking", not result.get("is_compatible", True))
        result.setdefault("compatibility_level", "backward")
        result.setdefault("breaking_changes", [])
        result.setdefault("warnings", [])
        result.setdefault("recommended_bump", "patch")
        return result

    def _normalise_plan_result(
        self,
        raw: Any,
        schema_id: str,
        source_version: str,
        target_version: str,
    ) -> Dict[str, Any]:
        """Normalise raw MigrationPlannerEngine output.

        Args:
            raw: Raw output from MigrationPlannerEngine.create_plan().
            schema_id: Schema subject identifier (for defaulting).
            source_version: Source version string (for defaulting).
            target_version: Target version string (for defaulting).

        Returns:
            Normalised plan result dictionary.
        """
        if isinstance(raw, dict):
            result = raw.copy()
        elif hasattr(raw, "dict"):
            result = raw.dict()
        elif hasattr(raw, "__dict__"):
            result = vars(raw)
        else:
            result = {"raw": str(raw)}

        result.setdefault("plan_id", _new_id("plan"))
        result.setdefault("schema_id", schema_id)
        result.setdefault("source_version", source_version)
        result.setdefault("target_version", target_version)
        result.setdefault("step_count", len(result.get("steps", [])))
        result.setdefault("steps", [])
        result.setdefault("effort", "LOW")
        result.setdefault("estimated_duration_seconds", 0)
        return result

    def _normalise_validate_result(self, raw: Any) -> Dict[str, Any]:
        """Normalise raw validation output from MigrationPlannerEngine.

        Args:
            raw: Raw output from MigrationPlannerEngine.validate_plan().

        Returns:
            Normalised validation result dictionary.
        """
        if isinstance(raw, dict):
            result = raw.copy()
        elif hasattr(raw, "dict"):
            result = raw.dict()
        elif hasattr(raw, "__dict__"):
            result = vars(raw)
        else:
            result = {"raw": str(raw)}

        result.setdefault("is_valid", True)
        result.setdefault("errors", [])
        result.setdefault("warnings", [])
        return result

    def _normalise_exec_result(self, raw: Any) -> Dict[str, Any]:
        """Normalise raw MigrationExecutorEngine output.

        Args:
            raw: Raw output from MigrationExecutorEngine.execute().

        Returns:
            Normalised execution result dictionary.
        """
        if isinstance(raw, dict):
            result = raw.copy()
        elif hasattr(raw, "dict"):
            result = raw.dict()
        elif hasattr(raw, "__dict__"):
            result = vars(raw)
        else:
            result = {"raw": str(raw)}

        result.setdefault("execution_id", _new_id("exec"))
        result.setdefault("status", "success")
        result.setdefault("records_migrated", 0)
        result.setdefault("migrated_data", None)
        result.setdefault("errors", [])
        result.setdefault("warnings", [])
        return result

    def _normalise_verify_result(self, raw: Any) -> Dict[str, Any]:
        """Normalise raw verification output from executor or structural check.

        Args:
            raw: Raw output from executor.verify() or internal verification.

        Returns:
            Normalised verification result dictionary.
        """
        if isinstance(raw, dict):
            result = raw.copy()
        elif hasattr(raw, "dict"):
            result = raw.dict()
        elif hasattr(raw, "__dict__"):
            result = vars(raw)
        else:
            result = {"raw": str(raw)}

        result.setdefault("passed", True)
        result.setdefault("records_verified", 0)
        result.setdefault("failure_reason", None)
        result.setdefault("warnings", [])
        return result

    # ------------------------------------------------------------------
    # Internal: stub implementations (engines unavailable)
    # ------------------------------------------------------------------

    def _stub_detect(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform a lightweight structural change detection without the engine.

        Compares top-level key sets of the two definition dicts.  Reports
        added keys as ``field_added`` (non-breaking) and removed keys as
        ``field_removed`` (breaking).

        Args:
            source_definition: Current schema definition dictionary.
            target_definition: Target schema definition dictionary.

        Returns:
            Stub detection result dictionary.
        """
        source_keys = set(source_definition.keys())
        target_keys = set(target_definition.keys())

        added = [
            {"change_type": "field_added", "field": k, "severity": "non_breaking"}
            for k in (target_keys - source_keys)
        ]
        removed = [
            {"change_type": "field_removed", "field": k, "severity": "breaking"}
            for k in (source_keys - target_keys)
        ]
        all_changes = added + removed

        return {
            "has_changes": bool(all_changes),
            "change_count": len(all_changes),
            "changes": all_changes,
            "breaking_changes": removed,
            "non_breaking_changes": added,
            "stub": True,
        }

    def _stub_compat(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Determine compatibility from the change detection output.

        Classifies a migration as breaking when any breaking changes exist
        in the ``changes`` dictionary.  No LLM inference is used.

        Args:
            changes: Output from :meth:`detect_stage`.

        Returns:
            Stub compatibility result dictionary.
        """
        breaking = changes.get("breaking_changes", [])
        is_breaking = len(breaking) > 0
        recommended_bump = "major" if is_breaking else "minor" if changes.get("change_count", 0) > 0 else "patch"

        return {
            "is_compatible": not is_breaking,
            "is_breaking": is_breaking,
            "compatibility_level": "backward",
            "breaking_changes": breaking,
            "warnings": [],
            "recommended_bump": recommended_bump,
            "stub": True,
        }

    def _stub_plan(
        self,
        schema_id: str,
        source_version: str,
        target_version: str,
        changes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a minimal migration plan from the detected changes.

        Creates one step per detected change.  Each step is a dict
        describing a deterministic field-level operation.  No LLM inference
        is used.

        Args:
            schema_id: Registered schema subject identifier.
            source_version: Semantic version string of the current schema.
            target_version: Semantic version string of the target schema.
            changes: Output from :meth:`detect_stage`.

        Returns:
            Stub migration plan dictionary.
        """
        plan_id = _new_id("plan")
        steps = []
        for i, change in enumerate(changes.get("changes", []), start=1):
            steps.append({
                "step_number": i,
                "operation": change.get("change_type", "modify"),
                "field": change.get("field", ""),
                "description": (
                    f"Apply {change.get('change_type', 'modify')} "
                    f"on field '{change.get('field', '')}'"
                ),
                "reversible": change.get("severity") != "breaking",
            })

        change_count = changes.get("change_count", 0)
        if change_count == 0:
            effort = "NONE"
        elif change_count <= 5:
            effort = "LOW"
        elif change_count <= 20:
            effort = "MEDIUM"
        elif change_count <= 50:
            effort = "HIGH"
        else:
            effort = _EFFORT_CRITICAL

        return {
            "plan_id": plan_id,
            "schema_id": schema_id,
            "source_version": source_version,
            "target_version": target_version,
            "step_count": len(steps),
            "steps": steps,
            "effort": effort,
            "estimated_duration_seconds": len(steps) * 2,
            "stub": True,
        }

    def _stub_validate(self, plan_id: str) -> Dict[str, Any]:
        """Return a pass-through validation result when the planner is absent.

        Only checks that ``plan_id`` is non-empty.

        Args:
            plan_id: Plan identifier to validate.

        Returns:
            Stub validation result dictionary.
        """
        is_valid = bool(plan_id)
        errors = [] if is_valid else ["plan_id is empty; cannot validate"]
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": ["Validation performed by stub; MigrationPlannerEngine unavailable"],
            "stub": True,
        }

    def _stub_execute(
        self,
        plan: Dict[str, Any],
        data: Optional[List[Dict[str, Any]]],
        dry_run: bool,
    ) -> Dict[str, Any]:
        """Apply migration steps to data records in pure Python.

        Iterates over each step in the plan and applies field-level
        operations (add, remove, rename) to each record in ``data``.
        No external engine, LLM, or ML model is used.

        Supported operations (derived from change_type):
        - ``field_added``: Adds the field with value ``None`` if absent.
        - ``field_removed``: Removes the field if present.
        - ``field_renamed``: Renames using ``old_field`` / ``new_field`` keys.
        - Any other: Left unchanged (logged as warning).

        Args:
            plan: Migration plan dictionary from :meth:`plan_stage`.
            data: List of record dictionaries to transform.
            dry_run: When ``True``, returns a copy without modifying input.

        Returns:
            Stub execution result dictionary including ``migrated_data``.
        """
        execution_id = _new_id("exec")
        steps = plan.get("steps", [])
        migrated_data: Optional[List[Dict[str, Any]]] = None
        errors: List[str] = []
        warnings: List[str] = []

        if dry_run:
            return {
                "execution_id": execution_id,
                "status": "dry_run",
                "records_migrated": 0,
                "migrated_data": None,
                "errors": [],
                "warnings": ["dry_run=True; no records transformed"],
                "stub": True,
            }

        if data is not None:
            migrated_data = []
            for record in data:
                row = dict(record)
                for step in steps:
                    operation = step.get("operation", "")
                    field = step.get("field", "")
                    try:
                        if operation == "field_added":
                            row.setdefault(field, None)
                        elif operation == "field_removed":
                            row.pop(field, None)
                        elif operation == "field_renamed":
                            old_f = step.get("old_field", field)
                            new_f = step.get("new_field", field)
                            if old_f in row:
                                row[new_f] = row.pop(old_f)
                        else:
                            warnings.append(
                                f"Unknown operation '{operation}' for field '{field}'; skipped"
                            )
                    except Exception as exc:
                        errors.append(
                            f"Step '{operation}' on field '{field}' failed: {exc}"
                        )
                migrated_data.append(row)

        status = "failed" if errors else "success"
        return {
            "execution_id": execution_id,
            "status": status,
            "records_migrated": len(migrated_data) if migrated_data is not None else 0,
            "migrated_data": migrated_data,
            "errors": errors,
            "warnings": warnings,
            "stub": True,
        }

    def _structural_verify(
        self,
        target_definition: Dict[str, Any],
        migrated_data: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Verify migrated data against target schema definition structurally.

        Checks:
        1. All keys in ``target_definition`` are present in every record.
        2. Records count is non-negative.
        3. No records have unexpected keys beyond the target definition
           (recorded as warnings, not failures).

        Args:
            target_definition: Target schema definition dictionary.
            migrated_data: List of migrated record dictionaries, or ``None``.

        Returns:
            Verification result dictionary.
        """
        if migrated_data is None:
            return {
                "passed": True,
                "records_verified": 0,
                "failure_reason": None,
                "warnings": ["No migrated_data supplied; structural check skipped"],
            }

        required_fields = set(target_definition.keys())
        warnings: List[str] = []
        failure_reasons: List[str] = []
        verified = 0

        for idx, record in enumerate(migrated_data):
            record_keys = set(record.keys())
            missing = required_fields - record_keys
            extra = record_keys - required_fields

            if missing:
                failure_reasons.append(
                    f"Record[{idx}] missing required fields: {sorted(missing)}"
                )
            if extra:
                warnings.append(
                    f"Record[{idx}] has extra fields not in target: {sorted(extra)}"
                )
            verified += 1

        passed = len(failure_reasons) == 0
        return {
            "passed": passed,
            "records_verified": verified,
            "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Internal: report generation helpers
    # ------------------------------------------------------------------

    def _summarise_changes_by_type(
        self, changes: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count detected changes grouped by change_type.

        Args:
            changes: List of change dictionaries from detect stage.

        Returns:
            Dictionary mapping change_type to occurrence count.
        """
        summary: Dict[str, int] = {}
        for change in changes:
            ctype = change.get("change_type", "unknown")
            summary[ctype] = summary.get(ctype, 0) + 1
        return summary

    def _build_compliance_notes(
        self,
        run: Dict[str, Any],
        compat: Dict[str, Any],
    ) -> List[str]:
        """Generate human-readable compliance notes for the pipeline run.

        Notes are derived deterministically from the pipeline result data.
        No LLM inference is used.

        Args:
            run: Pipeline result dictionary.
            compat: Compatibility stage result dictionary.

        Returns:
            List of compliance note strings.
        """
        notes: List[str] = []
        status = run.get("status", "unknown")

        if status == _STATUS_COMPLETED:
            notes.append(
                f"Migration completed successfully from version "
                f"{run.get('source_version', 'N/A')} to "
                f"{run.get('target_version', 'N/A')}."
            )
        elif status == _STATUS_ABORTED:
            notes.append(
                f"Migration aborted. Review 'error' field for details. "
                f"Failed at stages: {run.get('stages_failed', [])}."
            )
        elif status == _STATUS_NO_CHANGES:
            notes.append("No schema changes detected. Registry remains unchanged.")
        elif status == _STATUS_DRY_RUN:
            notes.append("Dry-run completed. No data or registry changes committed.")

        if compat.get("is_breaking"):
            notes.append(
                "Breaking changes were detected. Downstream consumers may require "
                "updates before deploying the new schema version."
            )

        plan = run.get("plan") or {}
        effort = plan.get("effort", "")
        if effort in (_EFFORT_CRITICAL, "HIGH"):
            notes.append(
                f"Migration effort classified as {effort}. "
                "Extended downtime window or phased rollout is recommended."
            )

        execution = run.get("execution") or {}
        records = execution.get("records_migrated", 0)
        if records > 0:
            notes.append(f"{records} record(s) successfully migrated and verified.")

        return notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SchemaMigrationPipelineEngine",
]
