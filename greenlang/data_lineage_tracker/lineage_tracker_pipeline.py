# -*- coding: utf-8 -*-
"""
LineageTrackerPipelineEngine - AGENT-DATA-018: Data Lineage Tracker

Engine 7 of 7 -- End-to-end pipeline orchestration for data lineage tracking.

This module implements the LineageTrackerPipelineEngine, which orchestrates
all six upstream engines (AssetRegistryEngine, TransformationTrackerEngine,
LineageGraphEngine, ImpactAnalyzerEngine, LineageValidatorEngine,
LineageReporterEngine) into a coherent pipeline workflow:

    Stage 1 REGISTER         - Bulk-register data assets into the asset registry
    Stage 2 CAPTURE          - Record data transformation events
    Stage 3 BUILD_GRAPH      - Synchronise the lineage graph from registry
                               and tracker state
    Stage 4 VALIDATE         - Run lineage validation checks for completeness,
                               cycles, orphans, and coverage
    Stage 5 ANALYZE          - Execute impact analysis on the lineage graph
    Stage 6 REPORT           - Generate lineage reports (visualization, audit
                               trail, compliance, data flow)
    Stage 7 DETECT_CHANGES   - Snapshot the graph and detect structural changes
                               since the last snapshot

Design Principles:
    - Zero-hallucination: all node counts, edge counts, coverage ratios and
      validation verdicts come from deterministic engine calls -- never from
      LLM inference.  Every numeric aggregation uses pure Python arithmetic.
    - Provenance: every pipeline run produces a SHA-256 provenance chain
      anchored to the shared ProvenanceTracker genesis hash.
    - Thread-safety: a single threading.Lock guards the pipeline run store,
      snapshots, and change events so concurrent callers never corrupt state.
    - Graceful degradation: each upstream engine is imported with a
      try/except guard; missing engines produce clear error messages rather
      than cryptic AttributeErrors at runtime.
    - Auditability: the pipeline result dictionary captures every stage
      outcome, timing, and error details for compliance reporting.

Example:
    >>> from greenlang.data_lineage_tracker.lineage_tracker_pipeline import (
    ...     LineageTrackerPipelineEngine,
    ... )
    >>> engine = LineageTrackerPipelineEngine()
    >>> result = engine.run_pipeline(scope="full")
    >>> assert result["pipeline_id"] is not None
    >>> assert len(result["stages_completed"]) >= 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
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
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker
    _PROVENANCE_AVAILABLE = True
except Exception:  # noqa: BLE001
    ProvenanceTracker = None  # type: ignore[assignment, misc]
    _PROVENANCE_AVAILABLE = False
    logger.warning(
        "ProvenanceTracker not available; provenance import failed."
    )

try:
    from greenlang.data_lineage_tracker.config import get_config
    _CONFIG_AVAILABLE = True
except Exception:  # noqa: BLE001
    get_config = None  # type: ignore[assignment, misc]
    _CONFIG_AVAILABLE = False

try:
    from greenlang.data_lineage_tracker.metrics import (
        observe_processing_duration,
        PROMETHEUS_AVAILABLE,
        set_graph_node_count,
        set_graph_edge_count,
        record_asset_registered,
        record_transformation_captured,
        record_edge_created,
        record_validation,
        record_report_generated,
        record_change_event,
    )
    _METRICS_AVAILABLE = True
except Exception:  # noqa: BLE001
    observe_processing_duration = None  # type: ignore[assignment]
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    set_graph_node_count = None  # type: ignore[assignment]
    set_graph_edge_count = None  # type: ignore[assignment]
    record_asset_registered = None  # type: ignore[assignment]
    record_transformation_captured = None  # type: ignore[assignment]
    record_edge_created = None  # type: ignore[assignment]
    record_validation = None  # type: ignore[assignment]
    record_report_generated = None  # type: ignore[assignment]
    record_change_event = None  # type: ignore[assignment]
    _METRICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Graceful imports for the 6 upstream engines
# ---------------------------------------------------------------------------

try:
    from greenlang.data_lineage_tracker.asset_registry import AssetRegistryEngine
    _ASSET_REGISTRY_AVAILABLE = True
except Exception:  # noqa: BLE001
    AssetRegistryEngine = None  # type: ignore[assignment, misc]
    _ASSET_REGISTRY_AVAILABLE = False
    logger.warning(
        "AssetRegistryEngine not available; asset_registry import failed. "
        "Asset registration will use stub fallback."
    )

try:
    from greenlang.data_lineage_tracker.transformation_tracker import (
        TransformationTrackerEngine,
    )
    _TRANSFORMATION_TRACKER_AVAILABLE = True
except Exception:  # noqa: BLE001
    TransformationTrackerEngine = None  # type: ignore[assignment, misc]
    _TRANSFORMATION_TRACKER_AVAILABLE = False
    logger.warning(
        "TransformationTrackerEngine not available; transformation_tracker "
        "import failed. Transformation capture will use stub fallback."
    )

try:
    from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
    _LINEAGE_GRAPH_AVAILABLE = True
except Exception:  # noqa: BLE001
    LineageGraphEngine = None  # type: ignore[assignment, misc]
    _LINEAGE_GRAPH_AVAILABLE = False
    logger.warning(
        "LineageGraphEngine not available; lineage_graph import failed. "
        "Graph operations will use stub fallback."
    )

try:
    from greenlang.data_lineage_tracker.impact_analyzer import ImpactAnalyzerEngine
    _IMPACT_ANALYZER_AVAILABLE = True
except Exception:  # noqa: BLE001
    ImpactAnalyzerEngine = None  # type: ignore[assignment, misc]
    _IMPACT_ANALYZER_AVAILABLE = False
    logger.warning(
        "ImpactAnalyzerEngine not available; impact_analyzer import failed. "
        "Impact analysis will use stub fallback."
    )

try:
    from greenlang.data_lineage_tracker.lineage_validator import (
        LineageValidatorEngine,
    )
    _LINEAGE_VALIDATOR_AVAILABLE = True
except Exception:  # noqa: BLE001
    LineageValidatorEngine = None  # type: ignore[assignment, misc]
    _LINEAGE_VALIDATOR_AVAILABLE = False
    logger.warning(
        "LineageValidatorEngine not available; lineage_validator import failed. "
        "Validation will use stub fallback."
    )

try:
    from greenlang.data_lineage_tracker.lineage_reporter import (
        LineageReporterEngine,
    )
    _LINEAGE_REPORTER_AVAILABLE = True
except Exception:  # noqa: BLE001
    LineageReporterEngine = None  # type: ignore[assignment, misc]
    _LINEAGE_REPORTER_AVAILABLE = False
    logger.warning(
        "LineageReporterEngine not available; lineage_reporter import failed. "
        "Report generation will use stub fallback."
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    "register",
    "capture",
    "build_graph",
    "validate",
    "analyze",
    "report",
    "detect_changes",
]

_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_PARTIAL = "partial"


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


def _safe_observe(operation: str, seconds: float) -> None:
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


# ---------------------------------------------------------------------------
# LineageTrackerPipelineEngine
# ---------------------------------------------------------------------------


class LineageTrackerPipelineEngine:
    """End-to-end pipeline orchestrator for the GreenLang Data Lineage Tracker.

    Coordinates all six upstream engines through a deterministic pipeline
    workflow: register -> capture -> build_graph -> validate -> analyze ->
    report -> detect_changes.  Every stage outcome is captured in the
    pipeline result dictionary for compliance reporting and auditability.

    Key design decisions:

    - **Zero-hallucination**: node counts, edge counts, coverage ratios,
      and validation verdicts come exclusively from upstream engine calls.
      No LLM inference is used in any stage.
    - **Provenance**: every pipeline run appends chain-hashed entries to the
      shared ProvenanceTracker.
    - **Thread-safety**: ``self._lock`` serialises writes to the pipeline
      run store, snapshot list, and change event list while individual
      engine calls remain stateless.
    - **Graceful degradation**: missing engines trigger stub behaviour that
      surfaces a clear ``"engine_unavailable"`` error rather than an
      AttributeError deep inside stage logic.

    Attributes:
        _asset_registry: Engine 1 - AssetRegistryEngine (or None).
        _transformation_tracker: Engine 2 - TransformationTrackerEngine (or None).
        _lineage_graph: Engine 3 - LineageGraphEngine (or None).
        _impact_analyzer: Engine 4 - ImpactAnalyzerEngine (or None).
        _lineage_validator: Engine 5 - LineageValidatorEngine (or None).
        _lineage_reporter: Engine 6 - LineageReporterEngine (or None).
        _provenance: SHA-256 chain-hashing provenance tracker.
        _snapshots: Point-in-time graph snapshots for change detection.
        _change_events: Detected structural changes between snapshots.
        _pipeline_runs: Mapping of pipeline_id to pipeline result dict.
        _lock: Mutex protecting concurrent writes to shared state.

    Example:
        >>> engine = LineageTrackerPipelineEngine()
        >>> result = engine.run_pipeline(scope="full")
        >>> print(result["pipeline_id"])
        pipe-abc123def456
    """

    def __init__(
        self,
        asset_registry: Any = None,
        transformation_tracker: Any = None,
        lineage_graph: Any = None,
        impact_analyzer: Any = None,
        lineage_validator: Any = None,
        lineage_reporter: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialise the pipeline engine and all six upstream engines.

        Each upstream engine can be injected via constructor parameters for
        testing.  When ``None`` is passed (the default), the engine is
        instantiated from its module if available, otherwise the attribute
        is set to ``None`` and the corresponding stage will return an
        ``"engine_unavailable"`` error.

        The ProvenanceTracker is shared across all engines to maintain a
        single provenance chain for the entire lineage tracker agent.

        Args:
            asset_registry: Optional pre-built AssetRegistryEngine instance.
            transformation_tracker: Optional pre-built
                TransformationTrackerEngine instance.
            lineage_graph: Optional pre-built LineageGraphEngine instance.
            impact_analyzer: Optional pre-built ImpactAnalyzerEngine instance.
            lineage_validator: Optional pre-built LineageValidatorEngine
                instance.
            lineage_reporter: Optional pre-built LineageReporterEngine
                instance.
            provenance: Optional pre-built ProvenanceTracker instance.
        """
        # Provenance tracker -- shared across all engines
        if provenance is not None:
            self._provenance = provenance
        elif _PROVENANCE_AVAILABLE and ProvenanceTracker is not None:
            self._provenance = ProvenanceTracker()
        else:
            self._provenance = None  # type: ignore[assignment]

        # Engine 1 -- Asset Registry
        if asset_registry is not None:
            self._asset_registry = asset_registry
        elif _ASSET_REGISTRY_AVAILABLE and AssetRegistryEngine is not None:
            self._asset_registry = AssetRegistryEngine(self._provenance)
        else:
            self._asset_registry = None

        # Engine 2 -- Transformation Tracker
        if transformation_tracker is not None:
            self._transformation_tracker = transformation_tracker
        elif (
            _TRANSFORMATION_TRACKER_AVAILABLE
            and TransformationTrackerEngine is not None
        ):
            self._transformation_tracker = TransformationTrackerEngine(
                self._provenance,
            )
        else:
            self._transformation_tracker = None

        # Engine 3 -- Lineage Graph
        if lineage_graph is not None:
            self._lineage_graph = lineage_graph
        elif _LINEAGE_GRAPH_AVAILABLE and LineageGraphEngine is not None:
            self._lineage_graph = LineageGraphEngine(self._provenance)
        else:
            self._lineage_graph = None

        # Engine 4 -- Impact Analyzer
        if impact_analyzer is not None:
            self._impact_analyzer = impact_analyzer
        elif (
            _IMPACT_ANALYZER_AVAILABLE
            and ImpactAnalyzerEngine is not None
            and self._lineage_graph is not None
        ):
            self._impact_analyzer = ImpactAnalyzerEngine(
                self._lineage_graph, self._provenance,
            )
        else:
            self._impact_analyzer = None

        # Engine 5 -- Lineage Validator
        if lineage_validator is not None:
            self._lineage_validator = lineage_validator
        elif (
            _LINEAGE_VALIDATOR_AVAILABLE
            and LineageValidatorEngine is not None
            and self._lineage_graph is not None
        ):
            self._lineage_validator = LineageValidatorEngine(
                self._lineage_graph, self._provenance,
            )
        else:
            self._lineage_validator = None

        # Engine 6 -- Lineage Reporter
        if lineage_reporter is not None:
            self._lineage_reporter = lineage_reporter
        elif (
            _LINEAGE_REPORTER_AVAILABLE
            and LineageReporterEngine is not None
            and self._lineage_graph is not None
        ):
            self._lineage_reporter = LineageReporterEngine(
                self._lineage_graph, self._provenance,
            )
        else:
            self._lineage_reporter = None

        # Pipeline state
        self._snapshots: List[Dict[str, Any]] = []
        self._change_events: List[Dict[str, Any]] = []
        self._pipeline_runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        logger.info(
            "LineageTrackerPipelineEngine initialised: "
            "asset_registry=%s transformation_tracker=%s lineage_graph=%s "
            "impact_analyzer=%s lineage_validator=%s lineage_reporter=%s "
            "provenance=%s",
            "ok" if self._asset_registry else "UNAVAILABLE",
            "ok" if self._transformation_tracker else "UNAVAILABLE",
            "ok" if self._lineage_graph else "UNAVAILABLE",
            "ok" if self._impact_analyzer else "UNAVAILABLE",
            "ok" if self._lineage_validator else "UNAVAILABLE",
            "ok" if self._lineage_reporter else "UNAVAILABLE",
            "ok" if self._provenance else "UNAVAILABLE",
        )

    # ------------------------------------------------------------------
    # Properties exposing sub-engines (read-only)
    # ------------------------------------------------------------------

    @property
    def asset_registry(self) -> Any:
        """Return the AssetRegistryEngine instance, or None if unavailable."""
        return self._asset_registry

    @property
    def transformation_tracker(self) -> Any:
        """Return the TransformationTrackerEngine instance, or None."""
        return self._transformation_tracker

    @property
    def lineage_graph(self) -> Any:
        """Return the LineageGraphEngine instance, or None."""
        return self._lineage_graph

    @property
    def impact_analyzer(self) -> Any:
        """Return the ImpactAnalyzerEngine instance, or None."""
        return self._impact_analyzer

    @property
    def lineage_validator(self) -> Any:
        """Return the LineageValidatorEngine instance, or None."""
        return self._lineage_validator

    @property
    def lineage_reporter(self) -> Any:
        """Return the LineageReporterEngine instance, or None."""
        return self._lineage_reporter

    @property
    def provenance(self) -> Any:
        """Return the shared ProvenanceTracker instance, or None."""
        return self._provenance

    # ------------------------------------------------------------------
    # Public API -- primary pipeline entry point
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        pipeline_id: Optional[str] = None,
        scope: str = "full",
        register_assets: bool = True,
        capture_transformations: bool = True,
        build_graph: bool = True,
        validate: bool = True,
        generate_report: bool = True,
        report_type: str = "visualization",
        report_format: str = "json",
        asset_metadata: Optional[List[Dict[str, Any]]] = None,
        transformation_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the complete lineage tracker pipeline end-to-end.

        Executes stages sequentially: register -> capture -> build_graph ->
        validate -> analyze -> report -> detect_changes.  Individual stages
        can be skipped via boolean flags or by setting ``scope`` to a subset.

        Args:
            pipeline_id: Optional custom pipeline run identifier.  When
                ``None``, a UUID-based identifier is auto-generated.
            scope: Pipeline scope.  ``"full"`` runs all stages.
                ``"register_only"`` runs only asset registration.
                ``"validate_only"`` runs only validation.
                ``"report_only"`` runs only report generation.
            register_assets: When ``True``, execute the register stage
                using ``asset_metadata`` if provided.
            capture_transformations: When ``True``, execute the capture
                stage using ``transformation_events`` if provided.
            build_graph: When ``True``, synchronise the lineage graph
                from registry and tracker state.
            validate: When ``True``, run lineage validation checks.
            generate_report: When ``True``, generate a lineage report.
            report_type: Type of report to generate.  One of
                ``"visualization"``, ``"audit_trail"``, ``"compliance"``,
                ``"data_flow"``, ``"quality_summary"``.
            report_format: Output format for the report.  One of
                ``"json"``, ``"html"``, ``"csv"``, ``"markdown"``.
            asset_metadata: Optional list of asset metadata dicts for
                bulk registration.  Each dict must contain at minimum
                ``"qualified_name"`` and ``"asset_type"`` keys.
            transformation_events: Optional list of transformation event
                dicts for bulk capture.

        Returns:
            Pipeline result dictionary with keys:
            ``pipeline_id``, ``scope``, ``stages_completed``,
            ``stages_skipped``, ``assets_registered``,
            ``transformations_captured``, ``edges_created``,
            ``validation_result``, ``report``, ``duration_ms``,
            ``errors``, ``started_at``, ``completed_at``,
            ``provenance_hash``, ``status``.

        Example:
            >>> engine = LineageTrackerPipelineEngine()
            >>> result = engine.run_pipeline(scope="full")
            >>> assert "pipeline_id" in result
        """
        if pipeline_id is None:
            pipeline_id = _new_id("pipe")

        started_at = _utcnow_iso()
        pipeline_start = time.monotonic()

        logger.info(
            "Pipeline %s starting: scope=%s register=%s capture=%s "
            "build_graph=%s validate=%s report=%s report_type=%s",
            pipeline_id,
            scope,
            register_assets,
            capture_transformations,
            build_graph,
            validate,
            generate_report,
            report_type,
        )

        result: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "scope": scope,
            "stages_completed": [],
            "stages_skipped": [],
            "assets_registered": 0,
            "transformations_captured": 0,
            "edges_created": 0,
            "validation_result": None,
            "report": None,
            "duration_ms": 0,
            "errors": [],
            "started_at": started_at,
            "completed_at": None,
            "provenance_hash": None,
            "status": _STATUS_COMPLETED,
        }

        # Determine which stages to run based on scope
        should_register = register_assets and scope in ("full", "register_only")
        should_capture = capture_transformations and scope in ("full",)
        should_build = build_graph and scope in ("full",)
        should_validate = validate and scope in ("full", "validate_only")
        should_analyze = scope in ("full",)
        should_report = generate_report and scope in ("full", "report_only")
        should_detect = scope in ("full",)

        try:
            # Stage 1: REGISTER
            result = self._execute_register_stage(
                result, pipeline_id, should_register, asset_metadata,
            )

            # Stage 2: CAPTURE
            result = self._execute_capture_stage(
                result, pipeline_id, should_capture, transformation_events,
            )

            # Stage 3: BUILD_GRAPH
            result = self._execute_build_graph_stage(
                result, pipeline_id, should_build,
            )

            # Stage 4: VALIDATE
            result = self._execute_validate_stage(
                result, pipeline_id, should_validate,
            )

            # Stage 5: ANALYZE
            result = self._execute_analyze_stage(
                result, pipeline_id, should_analyze,
            )

            # Stage 6: REPORT
            result = self._execute_report_stage(
                result, pipeline_id, should_report, report_type, report_format,
            )

            # Stage 7: DETECT_CHANGES
            result = self._execute_detect_changes_stage(
                result, pipeline_id, should_detect,
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
        _safe_observe("pipeline_run", elapsed / 1000.0)

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
    # Public API -- bulk registration
    # ------------------------------------------------------------------

    def register_assets_from_metadata(
        self,
        metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Bulk-register data assets from a list of metadata dictionaries.

        Each entry must contain at minimum ``qualified_name`` and
        ``asset_type``.  Additional fields (``description``,
        ``classification``, ``owner``, ``tags``, ``schema``) are
        forwarded to the asset registry engine.

        Args:
            metadata: List of asset metadata dictionaries.

        Returns:
            Summary dictionary with keys ``registered``, ``failed``,
            ``errors``.

        Example:
            >>> result = engine.register_assets_from_metadata([
            ...     {"qualified_name": "db.schema.table1", "asset_type": "table"},
            ...     {"qualified_name": "db.schema.table2", "asset_type": "table"},
            ... ])
            >>> assert result["registered"] >= 0
        """
        stage_start = time.monotonic()
        registered = 0
        failed = 0
        errors: List[str] = []

        if self._asset_registry is None:
            return {
                "registered": 0,
                "failed": len(metadata),
                "errors": ["AssetRegistryEngine is not available"],
            }

        for entry in metadata:
            qualified_name = entry.get("qualified_name", "")
            asset_type = entry.get("asset_type", "unknown")

            if not qualified_name:
                failed += 1
                errors.append("Missing 'qualified_name' in metadata entry")
                continue

            try:
                self._asset_registry.register_asset(
                    qualified_name=qualified_name,
                    asset_type=asset_type,
                    description=entry.get("description", ""),
                    classification=entry.get("classification", "internal"),
                    owner=entry.get("owner", ""),
                    tags=entry.get("tags", []),
                    schema_ref=entry.get("schema"),
                )
                registered += 1

                if (
                    _METRICS_AVAILABLE
                    and record_asset_registered is not None
                ):
                    record_asset_registered(
                        asset_type, entry.get("classification", "internal"),
                    )

            except Exception as exc:
                failed += 1
                errors.append(
                    f"Failed to register '{qualified_name}': {str(exc)}"
                )
                logger.warning(
                    "register_assets_from_metadata: failed for '%s': %s",
                    qualified_name,
                    exc,
                )

        elapsed = _elapsed_ms(stage_start)
        _safe_observe("bulk_register", elapsed / 1000.0)

        logger.info(
            "register_assets_from_metadata: registered=%d failed=%d "
            "duration_ms=%.2f",
            registered,
            failed,
            elapsed,
        )

        return {
            "registered": registered,
            "failed": failed,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Public API -- bulk transformation capture
    # ------------------------------------------------------------------

    def capture_transformations_from_events(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Bulk-record transformation events into the transformation tracker.

        Each event dictionary should contain the transformation metadata
        such as ``transformation_type``, ``agent_id``, ``input_assets``,
        ``output_assets``, and optional ``parameters``.

        Args:
            events: List of transformation event dictionaries.

        Returns:
            Summary dictionary with keys ``captured``, ``failed``,
            ``errors``.

        Example:
            >>> result = engine.capture_transformations_from_events([
            ...     {
            ...         "transformation_type": "filter",
            ...         "agent_id": "data-quality-profiler",
            ...         "input_assets": ["raw.table1"],
            ...         "output_assets": ["clean.table1"],
            ...     },
            ... ])
            >>> assert result["captured"] >= 0
        """
        stage_start = time.monotonic()
        captured = 0
        failed = 0
        errors: List[str] = []

        if self._transformation_tracker is None:
            return {
                "captured": 0,
                "failed": len(events),
                "errors": ["TransformationTrackerEngine is not available"],
            }

        for event in events:
            transformation_type = event.get("transformation_type", "unknown")
            agent_id = event.get("agent_id", "unknown")

            try:
                self._transformation_tracker.record_transformation(
                    transformation_type=transformation_type,
                    agent_id=agent_id,
                    pipeline_id=event.get("pipeline_id", "default"),
                    source_asset_ids=event.get("input_assets", []),
                    target_asset_ids=event.get("output_assets", []),
                    parameters=event.get("parameters"),
                    description=event.get("description", ""),
                )
                captured += 1

                if (
                    _METRICS_AVAILABLE
                    and record_transformation_captured is not None
                ):
                    record_transformation_captured(transformation_type, agent_id)

            except Exception as exc:
                failed += 1
                errors.append(
                    f"Failed to capture transformation "
                    f"(type={transformation_type}): {str(exc)}"
                )
                logger.warning(
                    "capture_transformations_from_events: failed for "
                    "type='%s': %s",
                    transformation_type,
                    exc,
                )

        elapsed = _elapsed_ms(stage_start)
        _safe_observe("bulk_capture", elapsed / 1000.0)

        logger.info(
            "capture_transformations_from_events: captured=%d failed=%d "
            "duration_ms=%.2f",
            captured,
            failed,
            elapsed,
        )

        return {
            "captured": captured,
            "failed": failed,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Public API -- graph construction
    # ------------------------------------------------------------------

    def build_graph_from_registry(self) -> Dict[str, Any]:
        """Synchronise the lineage graph from asset registry and tracker.

        Reads all registered assets from the asset registry and adds them
        as nodes in the lineage graph.  Then reads all recorded
        transformations from the transformation tracker and creates
        directed edges for each input -> output relationship.

        Returns:
            Summary dictionary with keys ``nodes_added`` and
            ``edges_added``.

        Example:
            >>> result = engine.build_graph_from_registry()
            >>> print(result["nodes_added"], result["edges_added"])
            5 3
        """
        stage_start = time.monotonic()
        nodes_added = 0
        edges_added = 0

        # Sync nodes from asset registry
        if self._asset_registry is not None and self._lineage_graph is not None:
            try:
                assets = self._asset_registry.list_assets()
                if isinstance(assets, dict):
                    assets = assets.get("assets", [])

                for asset in assets:
                    asset_id = ""
                    if isinstance(asset, dict):
                        asset_id = asset.get(
                            "qualified_name",
                            asset.get("asset_id", ""),
                        )
                    elif isinstance(asset, str):
                        asset_id = asset

                    if not asset_id:
                        continue

                    try:
                        asset_type = (
                            asset.get("asset_type", "unknown")
                            if isinstance(asset, dict)
                            else "unknown"
                        )
                        self._lineage_graph.add_node(
                            asset_id=asset_id,
                            qualified_name=asset_id,
                            asset_type=asset_type,
                            metadata=asset if isinstance(asset, dict) else {},
                        )
                        nodes_added += 1
                    except Exception:  # noqa: BLE001
                        pass  # Node may already exist

            except Exception as exc:
                logger.warning(
                    "build_graph_from_registry: asset sync failed: %s", exc,
                )

        # Sync edges from transformation tracker
        if (
            self._transformation_tracker is not None
            and self._lineage_graph is not None
        ):
            try:
                transformations = self._transformation_tracker.list_transformations()
                if isinstance(transformations, dict):
                    transformations = transformations.get("transformations", [])

                for txn in transformations:
                    if not isinstance(txn, dict):
                        continue

                    input_assets = txn.get("source_asset_ids", txn.get("input_assets", []))
                    output_assets = txn.get("target_asset_ids", txn.get("output_assets", []))
                    edge_type = txn.get(
                        "transformation_type", "derived_from",
                    )

                    # Normalise edge_type to a valid value accepted by
                    # LineageGraphEngine (dataset_level | column_level).
                    valid_edge_types = {"dataset_level", "column_level"}
                    normalised_edge_type = (
                        edge_type if edge_type in valid_edge_types
                        else "dataset_level"
                    )

                    for inp in input_assets:
                        for out in output_assets:
                            try:
                                self._lineage_graph.add_edge(
                                    source_asset_id=inp,
                                    target_asset_id=out,
                                    edge_type=normalised_edge_type,
                                )
                                edges_added += 1

                                if (
                                    _METRICS_AVAILABLE
                                    and record_edge_created is not None
                                ):
                                    record_edge_created(edge_type)

                            except Exception:  # noqa: BLE001
                                pass  # Edge may already exist

            except Exception as exc:
                logger.warning(
                    "build_graph_from_registry: transformation sync failed: %s",
                    exc,
                )

        # Update graph gauges
        self._update_graph_gauges()

        elapsed = _elapsed_ms(stage_start)
        _safe_observe("build_graph", elapsed / 1000.0)

        logger.info(
            "build_graph_from_registry: nodes_added=%d edges_added=%d "
            "duration_ms=%.2f",
            nodes_added,
            edges_added,
            elapsed,
        )

        return {
            "nodes_added": nodes_added,
            "edges_added": edges_added,
        }

    # ------------------------------------------------------------------
    # Public API -- change detection
    # ------------------------------------------------------------------

    def detect_changes(self) -> Dict[str, Any]:
        """Take a new snapshot and detect structural changes since the last.

        Compares the current lineage graph state against the most recent
        stored snapshot to identify added/removed nodes and edges.
        Detected changes are recorded as change events and stored for
        later retrieval.

        Returns:
            Change detection result dictionary with keys:
            ``previous_snapshot``, ``current_snapshot``, ``changes``,
            ``change_count``.

        Example:
            >>> result = engine.detect_changes()
            >>> print(result["change_count"])
            0
        """
        stage_start = time.monotonic()

        current_snapshot = self._take_snapshot()
        current_snapshot_id = current_snapshot.get("snapshot_id", "")

        previous_snapshot_id = ""
        changes: List[Dict[str, Any]] = []

        with self._lock:
            if len(self._snapshots) >= 2:
                previous = self._snapshots[-2]
                previous_snapshot_id = previous.get("snapshot_id", "")
                changes = self._compare_snapshots(previous, current_snapshot)
            elif len(self._snapshots) == 1:
                previous_snapshot_id = "genesis"
                changes = self._build_initial_changes(current_snapshot)

        # Store change events
        if changes:
            change_event = {
                "event_id": _new_id("chg"),
                "previous_snapshot": previous_snapshot_id,
                "current_snapshot": current_snapshot_id,
                "changes": changes,
                "change_count": len(changes),
                "detected_at": _utcnow_iso(),
            }
            with self._lock:
                self._change_events.append(change_event)

            # Record provenance for change detection
            if self._provenance is not None:
                try:
                    self._provenance.record(
                        "change_event",
                        change_event["event_id"],
                        "change_detected",
                        metadata={"change_count": len(changes)},
                    )
                except Exception:  # noqa: BLE001
                    pass

            # Record metrics for each change
            if _METRICS_AVAILABLE and record_change_event is not None:
                for change in changes:
                    try:
                        record_change_event(
                            change.get("change_type", "unknown"),
                            change.get("severity", "informational"),
                        )
                    except Exception:  # noqa: BLE001
                        pass

        elapsed = _elapsed_ms(stage_start)
        _safe_observe("detect_changes", elapsed / 1000.0)

        logger.info(
            "detect_changes: previous=%s current=%s changes=%d "
            "duration_ms=%.2f",
            previous_snapshot_id or "none",
            current_snapshot_id,
            len(changes),
            elapsed,
        )

        return {
            "previous_snapshot": previous_snapshot_id,
            "current_snapshot": current_snapshot_id,
            "changes": changes,
            "change_count": len(changes),
        }

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

    def get_change_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recorded change events, most recent first.

        Args:
            limit: Maximum number of events to return.  Defaults to 100.

        Returns:
            List of change event dictionaries sorted by ``detected_at``
            descending (most recent first).

        Raises:
            ValueError: If ``limit`` is negative.

        Example:
            >>> events = engine.get_change_events(limit=50)
            >>> for event in events:
            ...     print(event["change_count"])
        """
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")

        with self._lock:
            events = list(self._change_events)

        events.sort(
            key=lambda e: e.get("detected_at", ""),
            reverse=True,
        )
        return events[:limit]

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Return all stored lineage graph snapshots.

        Returns:
            List of snapshot dictionaries in chronological order
            (oldest first).

        Example:
            >>> snapshots = engine.get_snapshots()
            >>> print(len(snapshots))
        """
        with self._lock:
            return list(self._snapshots)

    # ------------------------------------------------------------------
    # Public API -- health and statistics
    # ------------------------------------------------------------------

    def get_health(self) -> Dict[str, Any]:
        """Return engine health status and lineage graph statistics.

        Checks the availability of each upstream engine and collects
        current graph node and edge counts.

        Returns:
            Health status dictionary with keys:
            ``status``, ``engines``, ``graph_stats``, ``checked_at``.

        Example:
            >>> health = engine.get_health()
            >>> print(health["status"])
            healthy
        """
        engines = {
            "asset_registry": self._asset_registry is not None,
            "transformation_tracker": self._transformation_tracker is not None,
            "lineage_graph": self._lineage_graph is not None,
            "impact_analyzer": self._impact_analyzer is not None,
            "lineage_validator": self._lineage_validator is not None,
            "lineage_reporter": self._lineage_reporter is not None,
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

        graph_stats = self._get_graph_stats()

        return {
            "status": status,
            "engines": engines,
            "engines_available": available_count,
            "engines_total": total_count,
            "graph_stats": graph_stats,
            "checked_at": _utcnow_iso(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics from all engines and pipeline runs.

        Computes statistics deterministically from the in-memory run store
        and engine state.  All numeric aggregations use pure Python
        arithmetic -- no LLM inference.

        Returns:
            Statistics dictionary with keys:
            ``total_pipeline_runs``, ``by_status``, ``avg_duration_ms``,
            ``min_duration_ms``, ``max_duration_ms``, ``success_rate``,
            ``total_snapshots``, ``total_change_events``,
            ``graph_stats``, ``asset_registry_stats``,
            ``transformation_tracker_stats``, ``provenance_entry_count``,
            ``computed_at``.

        Example:
            >>> stats = engine.get_statistics()
            >>> print(stats["total_pipeline_runs"])
            5
        """
        with self._lock:
            runs = list(self._pipeline_runs.values())
            snapshot_count = len(self._snapshots)
            change_event_count = len(self._change_events)

        total = len(runs)
        by_status: Dict[str, int] = {}
        durations: List[float] = []

        for run in runs:
            status = run.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            dur = run.get("duration_ms")
            if isinstance(dur, (int, float)):
                durations.append(float(dur))

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0

        completed = by_status.get(_STATUS_COMPLETED, 0)
        success_rate = completed / total if total > 0 else 0.0

        # Collect sub-engine statistics
        asset_stats = self._get_asset_registry_stats()
        txn_stats = self._get_transformation_tracker_stats()
        graph_stats = self._get_graph_stats()

        provenance_count = 0
        if self._provenance is not None:
            try:
                provenance_count = self._provenance.entry_count
            except Exception:  # noqa: BLE001
                pass

        return {
            "total_pipeline_runs": total,
            "by_status": by_status,
            "avg_duration_ms": round(avg_duration, 2),
            "min_duration_ms": round(min_duration, 2),
            "max_duration_ms": round(max_duration, 2),
            "success_rate": round(success_rate, 4),
            "total_snapshots": snapshot_count,
            "total_change_events": change_event_count,
            "graph_stats": graph_stats,
            "asset_registry_stats": asset_stats,
            "transformation_tracker_stats": txn_stats,
            "provenance_entry_count": provenance_count,
            "computed_at": _utcnow_iso(),
        }

    # ------------------------------------------------------------------
    # Public API -- reset / clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all engine state, pipeline runs, snapshots, and events.

        Clears the pipeline run store, snapshot list, change event list,
        resets the provenance tracker, and calls ``reset()`` on all
        sub-engines that support it.  Primarily intended for testing to
        prevent state leakage between test cases.

        Example:
            >>> engine.clear()
            >>> assert engine.get_statistics()["total_pipeline_runs"] == 0
        """
        with self._lock:
            self._pipeline_runs.clear()
            self._snapshots.clear()
            self._change_events.clear()

        # Reset provenance
        if self._provenance is not None:
            try:
                self._provenance.reset()
            except Exception as exc:
                logger.warning("clear: provenance reset failed: %s", exc)

        # Reset sub-engines
        for name, eng in (
            ("asset_registry", self._asset_registry),
            ("transformation_tracker", self._transformation_tracker),
            ("lineage_graph", self._lineage_graph),
            ("impact_analyzer", self._impact_analyzer),
            ("lineage_validator", self._lineage_validator),
            ("lineage_reporter", self._lineage_reporter),
        ):
            if eng is not None and hasattr(eng, "reset"):
                try:
                    eng.reset()
                except Exception as exc:
                    logger.warning(
                        "clear: engine %s reset failed: %s", name, exc,
                    )

        # Reset graph gauges
        if _METRICS_AVAILABLE:
            if set_graph_node_count is not None:
                try:
                    set_graph_node_count(0)
                except Exception:  # noqa: BLE001
                    pass
            if set_graph_edge_count is not None:
                try:
                    set_graph_edge_count(0)
                except Exception:  # noqa: BLE001
                    pass

        logger.info("LineageTrackerPipelineEngine: full clear/reset complete")

    # ------------------------------------------------------------------
    # Internal: stage execution methods
    # ------------------------------------------------------------------

    def _execute_register_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        should_run: bool,
        asset_metadata: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Execute the register stage of the pipeline.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            should_run: Whether this stage should execute.
            asset_metadata: Optional list of asset metadata dicts.

        Returns:
            Updated pipeline result dictionary.
        """
        if not should_run:
            result["stages_skipped"].append("register")
            return result

        stage_start = time.monotonic()

        try:
            if asset_metadata:
                reg_result = self.register_assets_from_metadata(asset_metadata)
                result["assets_registered"] = reg_result.get("registered", 0)
                if reg_result.get("errors"):
                    result["errors"].extend(reg_result["errors"])
            else:
                result["assets_registered"] = 0

            result["stages_completed"].append("register")

        except Exception as exc:
            result["errors"].append(f"Register stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s register stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed_s = (time.monotonic() - stage_start)
        _safe_observe("stage_register", elapsed_s)

        logger.info(
            "Pipeline %s stage register: assets=%d duration_ms=%.2f",
            pipeline_id,
            result["assets_registered"],
            elapsed_s * 1000.0,
        )
        return result

    def _execute_capture_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        should_run: bool,
        transformation_events: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Execute the capture stage of the pipeline.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            should_run: Whether this stage should execute.
            transformation_events: Optional list of transformation dicts.

        Returns:
            Updated pipeline result dictionary.
        """
        if not should_run:
            result["stages_skipped"].append("capture")
            return result

        stage_start = time.monotonic()

        try:
            if transformation_events:
                cap_result = self.capture_transformations_from_events(
                    transformation_events,
                )
                result["transformations_captured"] = cap_result.get(
                    "captured", 0,
                )
                if cap_result.get("errors"):
                    result["errors"].extend(cap_result["errors"])
            else:
                result["transformations_captured"] = 0

            result["stages_completed"].append("capture")

        except Exception as exc:
            result["errors"].append(f"Capture stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s capture stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed_s = (time.monotonic() - stage_start)
        _safe_observe("stage_capture", elapsed_s)

        logger.info(
            "Pipeline %s stage capture: transformations=%d duration_ms=%.2f",
            pipeline_id,
            result["transformations_captured"],
            elapsed_s * 1000.0,
        )
        return result

    def _execute_build_graph_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        should_run: bool,
    ) -> Dict[str, Any]:
        """Execute the build_graph stage of the pipeline.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            should_run: Whether this stage should execute.

        Returns:
            Updated pipeline result dictionary.
        """
        if not should_run:
            result["stages_skipped"].append("build_graph")
            return result

        stage_start = time.monotonic()

        try:
            graph_result = self.build_graph_from_registry()
            result["edges_created"] = graph_result.get("edges_added", 0)
            result["stages_completed"].append("build_graph")

        except Exception as exc:
            result["errors"].append(f"Build graph stage failed: {str(exc)}")
            logger.error(
                "Pipeline %s build_graph stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed_s = (time.monotonic() - stage_start)
        _safe_observe("stage_build_graph", elapsed_s)

        logger.info(
            "Pipeline %s stage build_graph: edges=%d duration_ms=%.2f",
            pipeline_id,
            result["edges_created"],
            elapsed_s * 1000.0,
        )
        return result

    def _execute_validate_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        should_run: bool,
    ) -> Dict[str, Any]:
        """Execute the validate stage of the pipeline.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            should_run: Whether this stage should execute.

        Returns:
            Updated pipeline result dictionary.
        """
        if not should_run:
            result["stages_skipped"].append("validate")
            return result

        stage_start = time.monotonic()

        try:
            if self._lineage_validator is not None:
                validation_result = self._lineage_validator.validate()
                result["validation_result"] = validation_result

                # Record validation metric
                if _METRICS_AVAILABLE and record_validation is not None:
                    verdict = "pass"
                    if isinstance(validation_result, dict):
                        verdict = validation_result.get("verdict", "pass")
                    try:
                        record_validation(verdict)
                    except Exception:  # noqa: BLE001
                        pass
            else:
                result["validation_result"] = {
                    "verdict": "skipped",
                    "reason": "LineageValidatorEngine is not available",
                }

            result["stages_completed"].append("validate")

        except Exception as exc:
            result["errors"].append(f"Validate stage failed: {str(exc)}")
            result["validation_result"] = {
                "verdict": "error",
                "reason": str(exc),
            }
            logger.error(
                "Pipeline %s validate stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed_s = (time.monotonic() - stage_start)
        _safe_observe("stage_validate", elapsed_s)

        logger.info(
            "Pipeline %s stage validate: result=%s duration_ms=%.2f",
            pipeline_id,
            result["validation_result"].get("verdict", "unknown")
            if isinstance(result["validation_result"], dict)
            else "unknown",
            elapsed_s * 1000.0,
        )
        return result

    def _execute_analyze_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        should_run: bool,
    ) -> Dict[str, Any]:
        """Execute the analyze (impact analysis) stage of the pipeline.

        This stage is informational and does not block subsequent stages.
        It performs a broad downstream impact analysis on all root nodes
        in the lineage graph.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            should_run: Whether this stage should execute.

        Returns:
            Updated pipeline result dictionary.
        """
        if not should_run:
            result["stages_skipped"].append("analyze")
            return result

        stage_start = time.monotonic()

        try:
            if self._impact_analyzer is not None:
                analysis = self._impact_analyzer.analyze_all()
                if "analysis" not in result:
                    result["analysis"] = analysis
            else:
                if "analysis" not in result:
                    result["analysis"] = {
                        "status": "skipped",
                        "reason": "ImpactAnalyzerEngine is not available",
                    }

            result["stages_completed"].append("analyze")

        except Exception as exc:
            result["errors"].append(f"Analyze stage failed: {str(exc)}")
            result["analysis"] = {
                "status": "error",
                "reason": str(exc),
            }
            logger.error(
                "Pipeline %s analyze stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed_s = (time.monotonic() - stage_start)
        _safe_observe("stage_analyze", elapsed_s)

        logger.info(
            "Pipeline %s stage analyze: duration_ms=%.2f",
            pipeline_id,
            elapsed_s * 1000.0,
        )
        return result

    def _execute_report_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        should_run: bool,
        report_type: str,
        report_format: str,
    ) -> Dict[str, Any]:
        """Execute the report generation stage of the pipeline.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            should_run: Whether this stage should execute.
            report_type: Type of report to generate.
            report_format: Output format for the report.

        Returns:
            Updated pipeline result dictionary.
        """
        if not should_run:
            result["stages_skipped"].append("report")
            return result

        stage_start = time.monotonic()

        try:
            if self._lineage_reporter is not None:
                report = self._lineage_reporter.generate_report(
                    report_type=report_type,
                    format=report_format,
                )
                result["report"] = report

                # Record report metric
                if _METRICS_AVAILABLE and record_report_generated is not None:
                    try:
                        record_report_generated(report_type, report_format)
                    except Exception:  # noqa: BLE001
                        pass
            else:
                result["report"] = {
                    "status": "skipped",
                    "reason": "LineageReporterEngine is not available",
                }

            result["stages_completed"].append("report")

        except Exception as exc:
            result["errors"].append(f"Report stage failed: {str(exc)}")
            result["report"] = {
                "status": "error",
                "reason": str(exc),
            }
            logger.error(
                "Pipeline %s report stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed_s = (time.monotonic() - stage_start)
        _safe_observe("stage_report", elapsed_s)

        logger.info(
            "Pipeline %s stage report: type=%s format=%s duration_ms=%.2f",
            pipeline_id,
            report_type,
            report_format,
            elapsed_s * 1000.0,
        )
        return result

    def _execute_detect_changes_stage(
        self,
        result: Dict[str, Any],
        pipeline_id: str,
        should_run: bool,
    ) -> Dict[str, Any]:
        """Execute the detect_changes stage of the pipeline.

        Args:
            result: Accumulating pipeline result dictionary.
            pipeline_id: Current pipeline run identifier.
            should_run: Whether this stage should execute.

        Returns:
            Updated pipeline result dictionary.
        """
        if not should_run:
            result["stages_skipped"].append("detect_changes")
            return result

        stage_start = time.monotonic()

        try:
            change_result = self.detect_changes()
            result["change_detection"] = change_result
            result["stages_completed"].append("detect_changes")

        except Exception as exc:
            result["errors"].append(
                f"Detect changes stage failed: {str(exc)}"
            )
            result["change_detection"] = {
                "status": "error",
                "reason": str(exc),
            }
            logger.error(
                "Pipeline %s detect_changes stage failed: %s",
                pipeline_id,
                exc,
                exc_info=True,
            )

        elapsed_s = (time.monotonic() - stage_start)
        _safe_observe("stage_detect_changes", elapsed_s)

        logger.info(
            "Pipeline %s stage detect_changes: duration_ms=%.2f",
            pipeline_id,
            elapsed_s * 1000.0,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: snapshot and change detection helpers
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> Dict[str, Any]:
        """Capture a point-in-time snapshot of the lineage graph.

        Reads all node IDs and edge tuples from the lineage graph and
        stores them as a snapshot with a unique identifier and timestamp.

        Returns:
            Snapshot dictionary with keys: ``snapshot_id``, ``timestamp``,
            ``node_ids``, ``edge_tuples``, ``node_count``, ``edge_count``,
            ``hash``.
        """
        snapshot_id = _new_id("snap")
        timestamp = _utcnow_iso()

        node_ids: List[str] = []
        edge_tuples: List[List[str]] = []

        if self._lineage_graph is not None:
            try:
                nodes = self._lineage_graph.list_nodes()
                if isinstance(nodes, dict):
                    nodes = nodes.get("nodes", [])
                for node in nodes:
                    if isinstance(node, dict):
                        node_ids.append(
                            node.get("node_id", node.get("id", "")),
                        )
                    elif isinstance(node, str):
                        node_ids.append(node)
            except Exception as exc:
                logger.warning(
                    "_take_snapshot: list_nodes failed: %s", exc,
                )

            try:
                edges = self._lineage_graph.list_edges()
                if isinstance(edges, dict):
                    edges = edges.get("edges", [])
                for edge in edges:
                    if isinstance(edge, dict):
                        source = edge.get("source_id", edge.get("source", ""))
                        target = edge.get("target_id", edge.get("target", ""))
                        etype = edge.get("edge_type", "unknown")
                        edge_tuples.append([source, target, etype])
                    elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                        edge_tuples.append(list(edge[:3]))
            except Exception as exc:
                logger.warning(
                    "_take_snapshot: list_edges failed: %s", exc,
                )

        snapshot_data = {
            "node_ids": sorted(node_ids),
            "edge_tuples": sorted(
                edge_tuples, key=lambda e: (e[0] if e else "", e[1] if len(e) > 1 else ""),
            ),
        }

        snapshot = {
            "snapshot_id": snapshot_id,
            "timestamp": timestamp,
            "node_ids": snapshot_data["node_ids"],
            "edge_tuples": snapshot_data["edge_tuples"],
            "node_count": len(node_ids),
            "edge_count": len(edge_tuples),
            "hash": _sha256(snapshot_data),
        }

        with self._lock:
            self._snapshots.append(snapshot)

        logger.debug(
            "_take_snapshot: id=%s nodes=%d edges=%d",
            snapshot_id,
            len(node_ids),
            len(edge_tuples),
        )
        return snapshot

    def _compare_snapshots(
        self,
        previous: Dict[str, Any],
        current: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare two snapshots and return a list of structural changes.

        Detects nodes and edges that were added or removed between the
        two snapshot states.

        Args:
            previous: The earlier snapshot dictionary.
            current: The later snapshot dictionary.

        Returns:
            List of change dictionaries, each with keys ``change_type``,
            ``entity_type``, ``entity_id``, ``severity``.
        """
        changes: List[Dict[str, Any]] = []

        prev_nodes = set(previous.get("node_ids", []))
        curr_nodes = set(current.get("node_ids", []))

        # Nodes added
        for node_id in sorted(curr_nodes - prev_nodes):
            changes.append({
                "change_type": "asset_added",
                "entity_type": "node",
                "entity_id": node_id,
                "severity": "informational",
            })

        # Nodes removed
        for node_id in sorted(prev_nodes - curr_nodes):
            changes.append({
                "change_type": "asset_removed",
                "entity_type": "node",
                "entity_id": node_id,
                "severity": "high",
            })

        # Edge comparison
        prev_edge_set = set()
        for edge in previous.get("edge_tuples", []):
            if isinstance(edge, (list, tuple)):
                prev_edge_set.add(tuple(edge))

        curr_edge_set = set()
        for edge in current.get("edge_tuples", []):
            if isinstance(edge, (list, tuple)):
                curr_edge_set.add(tuple(edge))

        # Edges added
        for edge in sorted(curr_edge_set - prev_edge_set):
            changes.append({
                "change_type": "edge_added",
                "entity_type": "edge",
                "entity_id": f"{edge[0]}->{edge[1]}" if len(edge) >= 2 else str(edge),
                "severity": "informational",
            })

        # Edges removed
        for edge in sorted(prev_edge_set - curr_edge_set):
            changes.append({
                "change_type": "edge_removed",
                "entity_type": "edge",
                "entity_id": f"{edge[0]}->{edge[1]}" if len(edge) >= 2 else str(edge),
                "severity": "medium",
            })

        return changes

    def _build_initial_changes(
        self,
        snapshot: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build change events for the first snapshot (all items are new).

        Args:
            snapshot: The initial snapshot dictionary.

        Returns:
            List of change dictionaries representing newly detected
            nodes and edges.
        """
        changes: List[Dict[str, Any]] = []

        for node_id in snapshot.get("node_ids", []):
            changes.append({
                "change_type": "asset_added",
                "entity_type": "node",
                "entity_id": node_id,
                "severity": "informational",
            })

        for edge in snapshot.get("edge_tuples", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                changes.append({
                    "change_type": "edge_added",
                    "entity_type": "edge",
                    "entity_id": f"{edge[0]}->{edge[1]}",
                    "severity": "informational",
                })

        return changes

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
                entity_type="pipeline_run",
                entity_id=pipeline_id,
                action="pipeline_completed",
                metadata={
                    "status": result.get("status"),
                    "stages_completed": result.get("stages_completed", []),
                    "assets_registered": result.get("assets_registered", 0),
                    "transformations_captured": result.get(
                        "transformations_captured", 0,
                    ),
                    "edges_created": result.get("edges_created", 0),
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

    # ------------------------------------------------------------------
    # Internal: sub-engine statistics helpers
    # ------------------------------------------------------------------

    def _get_graph_stats(self) -> Dict[str, Any]:
        """Collect current node and edge counts from the lineage graph.

        Returns:
            Dictionary with ``node_count`` and ``edge_count``.
        """
        node_count = 0
        edge_count = 0

        if self._lineage_graph is not None:
            try:
                stats = self._lineage_graph.get_statistics()
                if isinstance(stats, dict):
                    node_count = stats.get("node_count", 0)
                    edge_count = stats.get("edge_count", 0)
            except AttributeError:
                # Fallback: try list methods
                try:
                    nodes = self._lineage_graph.list_nodes()
                    if isinstance(nodes, dict):
                        node_count = len(nodes.get("nodes", []))
                    elif isinstance(nodes, list):
                        node_count = len(nodes)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    edges = self._lineage_graph.list_edges()
                    if isinstance(edges, dict):
                        edge_count = len(edges.get("edges", []))
                    elif isinstance(edges, list):
                        edge_count = len(edges)
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:
                logger.warning("_get_graph_stats failed: %s", exc)

        return {
            "node_count": node_count,
            "edge_count": edge_count,
        }

    def _get_asset_registry_stats(self) -> Dict[str, Any]:
        """Collect statistics from the asset registry engine.

        Returns:
            Dictionary with asset count and type breakdown.
        """
        if self._asset_registry is None:
            return {"status": "unavailable"}

        try:
            if hasattr(self._asset_registry, "get_statistics"):
                return self._asset_registry.get_statistics()
            assets = self._asset_registry.list_assets()
            if isinstance(assets, dict):
                return {"total_assets": len(assets.get("assets", []))}
            if isinstance(assets, list):
                return {"total_assets": len(assets)}
            return {"total_assets": 0}
        except Exception as exc:
            logger.warning("_get_asset_registry_stats failed: %s", exc)
            return {"status": "error", "reason": str(exc)}

    def _get_transformation_tracker_stats(self) -> Dict[str, Any]:
        """Collect statistics from the transformation tracker engine.

        Returns:
            Dictionary with transformation count and type breakdown.
        """
        if self._transformation_tracker is None:
            return {"status": "unavailable"}

        try:
            if hasattr(self._transformation_tracker, "get_statistics"):
                return self._transformation_tracker.get_statistics()
            txns = self._transformation_tracker.list_transformations()
            if isinstance(txns, dict):
                return {
                    "total_transformations": len(
                        txns.get("transformations", []),
                    ),
                }
            if isinstance(txns, list):
                return {"total_transformations": len(txns)}
            return {"total_transformations": 0}
        except Exception as exc:
            logger.warning(
                "_get_transformation_tracker_stats failed: %s", exc,
            )
            return {"status": "error", "reason": str(exc)}

    # ------------------------------------------------------------------
    # Internal: graph gauge update helper
    # ------------------------------------------------------------------

    def _update_graph_gauges(self) -> None:
        """Update Prometheus gauges for graph node and edge counts."""
        if not _METRICS_AVAILABLE:
            return

        stats = self._get_graph_stats()

        if set_graph_node_count is not None:
            try:
                set_graph_node_count(stats.get("node_count", 0))
            except Exception:  # noqa: BLE001
                pass

        if set_graph_edge_count is not None:
            try:
                set_graph_edge_count(stats.get("edge_count", 0))
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LineageTrackerPipelineEngine",
    "PIPELINE_STAGES",
]
