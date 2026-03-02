# -*- coding: utf-8 -*-
"""
Audit Trail & Lineage Prometheus Metrics - AGENT-MRV-030

14 Prometheus metrics with gl_atl_ prefix for monitoring
the Audit Trail & Lineage Agent (GL-MRV-X-042).

This module provides Prometheus metrics tracking for audit trail event
recording, hash chain verification, lineage graph construction, evidence
packaging, compliance coverage, change detection, and pipeline orchestration.

Thread-safe singleton pattern with graceful fallback if prometheus_client
is not installed.

Metrics prefix: gl_atl_

14 Prometheus Metrics:
    1.  gl_atl_events_total                         - Counter: total audit events recorded
    2.  gl_atl_event_recording_duration_seconds      - Histogram: event recording latency
    3.  gl_atl_chain_length                          - Gauge: current chain length per org
    4.  gl_atl_chain_verifications_total              - Counter: chain verifications
    5.  gl_atl_lineage_nodes_total                   - Counter: lineage nodes created
    6.  gl_atl_lineage_edges_total                   - Counter: lineage edges created
    7.  gl_atl_lineage_depth                         - Histogram: lineage traversal depth
    8.  gl_atl_evidence_packages_total               - Counter: evidence packages created
    9.  gl_atl_evidence_completeness_score            - Histogram: evidence completeness
    10. gl_atl_compliance_coverage_pct               - Gauge: compliance coverage per org
    11. gl_atl_changes_detected_total                - Counter: changes detected
    12. gl_atl_recalculations_triggered_total        - Counter: recalculations triggered
    13. gl_atl_pipeline_duration_seconds             - Histogram: pipeline execution latency
    14. gl_atl_pipeline_executions_total             - Counter: pipeline executions

Example:
    >>> metrics = AuditTrailLineageMetrics()
    >>> metrics.record_event(event_type="calculation_completed", scope="scope_1")
    >>> metrics.record_pipeline_execution(
    ...     status="SUCCESS", duration=0.045,
    ...     event_type="calculation_completed", scope="scope_1",
    ... )

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-042
Date: March 2026
"""

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful Prometheus import -- fall back to no-op stubs when the client
# library is not installed, ensuring the agent still operates correctly.
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not available, metrics will be no-ops")
    PROMETHEUS_AVAILABLE = False

    class _NoOpMetric:
        """No-op metric stub for environments without prometheus_client."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

        def set(self, value: float = 0) -> None:
            pass

        def observe(self, amount: float = 0) -> None:
            pass

        def info(self, data: Optional[Dict[str, str]] = None) -> None:
            pass

    Counter = _NoOpMetric  # type: ignore[misc,assignment]
    Histogram = _NoOpMetric  # type: ignore[misc,assignment]
    Gauge = _NoOpMetric  # type: ignore[misc,assignment]
    Info = _NoOpMetric  # type: ignore[misc,assignment]


# ===========================================================================
# Enumerations -- Audit Trail & Lineage domain label sets
# ===========================================================================


class EventTypeLabel(str, Enum):
    """Audit event type labels."""

    CALCULATION_COMPLETED = "calculation_completed"
    CALCULATION_UPDATED = "calculation_updated"
    DATA_INGESTED = "data_ingested"
    DATA_VALIDATED = "data_validated"
    EMISSION_FACTOR_APPLIED = "emission_factor_applied"
    EMISSION_FACTOR_UPDATED = "emission_factor_updated"
    AGGREGATION_COMPLETED = "aggregation_completed"
    REPORT_GENERATED = "report_generated"
    CORRECTION_APPLIED = "correction_applied"
    ASSUMPTION_CHANGED = "assumption_changed"
    METHODOLOGY_CHANGED = "methodology_changed"
    RECALCULATION_TRIGGERED = "recalculation_triggered"
    AUDIT_EVENT = "audit_event"
    OTHER = "other"


class ScopeLabel(str, Enum):
    """GHG emission scope labels."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    CROSS_CUTTING = "cross_cutting"


class PipelineStatusLabel(str, Enum):
    """Pipeline execution status labels."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    VALIDATION_ERROR = "validation_error"


class EdgeTypeLabel(str, Enum):
    """Lineage edge type labels."""

    DERIVED_FROM = "derived_from"
    DEPENDS_ON = "depends_on"
    AGGREGATED_INTO = "aggregated_into"
    CALCULATION_DEPENDENCY = "calculation_dependency"
    DATA_SOURCE = "data_source"


class ChangeTypeLabel(str, Enum):
    """Change detection type labels."""

    EMISSION_FACTOR = "emission_factor"
    ACTIVITY_DATA = "activity_data"
    METHODOLOGY = "methodology"
    ASSUMPTION = "assumption"
    BOUNDARY = "boundary"
    ALLOCATION = "allocation"


class VerificationResultLabel(str, Enum):
    """Chain verification result labels."""

    VALID = "valid"
    INVALID = "invalid"


class PipelineStageLabel(str, Enum):
    """Pipeline stage labels for duration tracking."""

    VALIDATE = "validate"
    CLASSIFY = "classify"
    RECORD = "record"
    LINK = "link"
    TRACE = "trace"
    DETECT = "detect"
    VERIFY = "verify"
    PACKAGE = "package"
    COMPLIANCE = "compliance"
    SEAL = "seal"


# ===========================================================================
# AuditTrailLineageMetrics -- Thread-safe Singleton
# ===========================================================================


class AuditTrailLineageMetrics:
    """
    Thread-safe singleton metrics collector for Audit Trail & Lineage (MRV-030).

    Provides 14 Prometheus metrics for tracking audit event recording,
    hash chain verification, lineage graph construction, evidence packaging,
    compliance coverage, change detection, and pipeline orchestration.

    All metrics use the ``gl_atl_`` prefix to ensure namespace isolation
    within the GreenLang Prometheus ecosystem.

    Example:
        >>> metrics = AuditTrailLineageMetrics()
        >>> metrics.record_event(
        ...     event_type="calculation_completed", scope="scope_1",
        ... )
        >>> metrics.record_chain_verification(valid=True, chain_length=42)
    """

    _instance: Optional["AuditTrailLineageMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AuditTrailLineageMetrics":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._start_time: datetime = datetime.utcnow()
        self._stats_lock: threading.Lock = threading.Lock()
        self._in_memory_stats: Dict[str, Any] = {
            "events": 0,
            "chain_verifications": 0,
            "lineage_nodes": 0,
            "lineage_edges": 0,
            "evidence_packages": 0,
            "changes_detected": 0,
            "recalculations": 0,
            "pipeline_executions": 0,
            "errors": 0,
        }

        self._init_metrics()

        logger.info(
            "AuditTrailLineageMetrics initialized (Prometheus: %s)",
            "available" if PROMETHEUS_AVAILABLE else "unavailable",
        )

    def _init_metrics(self) -> None:
        """
        Initialize all 14 Prometheus metrics with gl_atl_ prefix.

        When Prometheus is available, metric registration may fail if the
        metrics were previously registered. In that case we unregister
        and re-register to obtain fresh collector objects.
        """
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """Create metric, unregistering prior collector on conflict."""
                try:
                    return metric_cls(name, *args, **kwargs)
                except ValueError:
                    try:
                        REGISTRY.unregister(REGISTRY._names_to_collectors.get(name))
                    except Exception:
                        for collector in list(REGISTRY._names_to_collectors.values()):
                            try:
                                REGISTRY.unregister(collector)
                            except Exception:
                                pass
                    return metric_cls(name, *args, **kwargs)
        else:
            def _safe_create(metric_cls: type, name: str, *args: Any, **kwargs: Any) -> Any:
                """No-op stub creation (Prometheus not available)."""
                return metric_cls(name, *args, **kwargs)

        # ------------------------------------------------------------------
        # 1. gl_atl_events_total (Counter)
        #    Total audit events recorded by event_type and scope.
        # ------------------------------------------------------------------
        self.events_total = _safe_create(
            Counter,
            "gl_atl_events_total",
            "Total audit events recorded",
            ["event_type", "scope"],
        )

        # ------------------------------------------------------------------
        # 2. gl_atl_event_recording_duration_seconds (Histogram)
        #    Duration of audit event recording operations.
        # ------------------------------------------------------------------
        self.event_recording_duration_seconds = _safe_create(
            Histogram,
            "gl_atl_event_recording_duration_seconds",
            "Duration of audit event recording in seconds",
            ["event_type"],
            buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        # ------------------------------------------------------------------
        # 3. gl_atl_chain_length (Gauge)
        #    Current chain length per organization.
        # ------------------------------------------------------------------
        self.chain_length = _safe_create(
            Gauge,
            "gl_atl_chain_length",
            "Current audit chain length per organization",
            ["organization_id"],
        )

        # ------------------------------------------------------------------
        # 4. gl_atl_chain_verifications_total (Counter)
        #    Total chain verifications by result (valid/invalid).
        # ------------------------------------------------------------------
        self.chain_verifications_total = _safe_create(
            Counter,
            "gl_atl_chain_verifications_total",
            "Total hash chain verifications performed",
            ["result"],
        )

        # ------------------------------------------------------------------
        # 5. gl_atl_lineage_nodes_total (Counter)
        #    Total lineage graph nodes created by scope.
        # ------------------------------------------------------------------
        self.lineage_nodes_total = _safe_create(
            Counter,
            "gl_atl_lineage_nodes_total",
            "Total lineage graph nodes created",
            ["scope"],
        )

        # ------------------------------------------------------------------
        # 6. gl_atl_lineage_edges_total (Counter)
        #    Total lineage graph edges created by edge_type.
        # ------------------------------------------------------------------
        self.lineage_edges_total = _safe_create(
            Counter,
            "gl_atl_lineage_edges_total",
            "Total lineage graph edges created",
            ["edge_type"],
        )

        # ------------------------------------------------------------------
        # 7. gl_atl_lineage_depth (Histogram)
        #    Distribution of lineage graph traversal depths.
        # ------------------------------------------------------------------
        self.lineage_depth = _safe_create(
            Histogram,
            "gl_atl_lineage_depth",
            "Lineage graph traversal depth distribution",
            ["scope"],
            buckets=(1, 2, 5, 10, 15, 20, 30, 40, 50),
        )

        # ------------------------------------------------------------------
        # 8. gl_atl_evidence_packages_total (Counter)
        #    Total evidence packages created by scope.
        # ------------------------------------------------------------------
        self.evidence_packages_total = _safe_create(
            Counter,
            "gl_atl_evidence_packages_total",
            "Total evidence packages created",
            ["scope"],
        )

        # ------------------------------------------------------------------
        # 9. gl_atl_evidence_completeness_score (Histogram)
        #    Distribution of evidence completeness scores (0.0-1.0).
        # ------------------------------------------------------------------
        self.evidence_completeness_score = _safe_create(
            Histogram,
            "gl_atl_evidence_completeness_score",
            "Evidence completeness score distribution",
            ["scope"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
        )

        # ------------------------------------------------------------------
        # 10. gl_atl_compliance_coverage_pct (Gauge)
        #     Compliance coverage percentage per organization.
        # ------------------------------------------------------------------
        self.compliance_coverage_pct = _safe_create(
            Gauge,
            "gl_atl_compliance_coverage_pct",
            "Compliance coverage percentage per organization",
            ["organization_id"],
        )

        # ------------------------------------------------------------------
        # 11. gl_atl_changes_detected_total (Counter)
        #     Total changes detected by event_type.
        # ------------------------------------------------------------------
        self.changes_detected_total = _safe_create(
            Counter,
            "gl_atl_changes_detected_total",
            "Total changes detected requiring review",
            ["event_type"],
        )

        # ------------------------------------------------------------------
        # 12. gl_atl_recalculations_triggered_total (Counter)
        #     Total recalculations triggered by scope and category.
        # ------------------------------------------------------------------
        self.recalculations_triggered_total = _safe_create(
            Counter,
            "gl_atl_recalculations_triggered_total",
            "Total recalculations triggered by change detection",
            ["scope", "category"],
        )

        # ------------------------------------------------------------------
        # 13. gl_atl_pipeline_duration_seconds (Histogram)
        #     Pipeline execution duration by status.
        # ------------------------------------------------------------------
        self.pipeline_duration_seconds = _safe_create(
            Histogram,
            "gl_atl_pipeline_duration_seconds",
            "Audit trail pipeline execution duration in seconds",
            ["status"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # ------------------------------------------------------------------
        # 14. gl_atl_pipeline_executions_total (Counter)
        #     Total pipeline executions by status, event_type, and scope.
        # ------------------------------------------------------------------
        self.pipeline_executions_total = _safe_create(
            Counter,
            "gl_atl_pipeline_executions_total",
            "Total audit trail pipeline executions",
            ["status", "event_type", "scope"],
        )

        # ------------------------------------------------------------------
        # Agent info metric (non-numeric metadata)
        # ------------------------------------------------------------------
        self.agent_info = _safe_create(
            Info,
            "gl_atl_agent",
            "Audit Trail & Lineage Agent metadata",
        )
        if PROMETHEUS_AVAILABLE:
            try:
                self.agent_info.info({
                    "agent_id": "GL-MRV-X-042",
                    "version": "1.0.0",
                    "scope": "cross_cutting",
                    "description": "Audit trail and calculation lineage tracking",
                    "engines": "7",
                    "pipeline_stages": "10",
                })
            except Exception:
                pass

    # ======================================================================
    # Primary recording methods
    # ======================================================================

    def record_event(
        self,
        event_type: str,
        scope: str,
        duration: Optional[float] = None,
    ) -> None:
        """
        Record an audit event.

        Args:
            event_type: Audit event type.
            scope: GHG emission scope.
            duration: Event recording duration in seconds (optional).
        """
        try:
            event_type = self._validate_enum_value(
                event_type, EventTypeLabel, EventTypeLabel.OTHER.value
            )
            scope = self._validate_enum_value(
                scope, ScopeLabel, ScopeLabel.CROSS_CUTTING.value
            )

            self.events_total.labels(
                event_type=event_type, scope=scope
            ).inc()

            if duration is not None and duration > 0:
                self.event_recording_duration_seconds.labels(
                    event_type=event_type
                ).observe(duration)

            with self._stats_lock:
                self._in_memory_stats["events"] += 1

        except Exception as e:
            logger.error(
                "Failed to record event metrics: %s", e, exc_info=True
            )

    def record_chain_verification(
        self,
        valid: bool,
        chain_length: int = 0,
        organization_id: Optional[str] = None,
    ) -> None:
        """
        Record a chain verification operation.

        Args:
            valid: Whether the chain passed verification.
            chain_length: Number of events in the chain.
            organization_id: Organization identifier (for chain_length gauge).
        """
        try:
            result_label = (
                VerificationResultLabel.VALID.value
                if valid
                else VerificationResultLabel.INVALID.value
            )
            self.chain_verifications_total.labels(result=result_label).inc()

            if organization_id and chain_length > 0:
                self.chain_length.labels(
                    organization_id=organization_id
                ).set(chain_length)

            with self._stats_lock:
                self._in_memory_stats["chain_verifications"] += 1

        except Exception as e:
            logger.error(
                "Failed to record chain verification: %s", e, exc_info=True
            )

    def record_lineage_node(self, scope: str) -> None:
        """
        Record a lineage graph node creation.

        Args:
            scope: GHG emission scope.
        """
        try:
            scope = self._validate_enum_value(
                scope, ScopeLabel, ScopeLabel.CROSS_CUTTING.value
            )
            self.lineage_nodes_total.labels(scope=scope).inc()

            with self._stats_lock:
                self._in_memory_stats["lineage_nodes"] += 1

        except Exception as e:
            logger.error(
                "Failed to record lineage node: %s", e, exc_info=True
            )

    def record_lineage_edge(self, edge_type: str) -> None:
        """
        Record a lineage graph edge creation.

        Args:
            edge_type: Type of lineage edge.
        """
        try:
            edge_type = self._validate_enum_value(
                edge_type, EdgeTypeLabel,
                EdgeTypeLabel.CALCULATION_DEPENDENCY.value,
            )
            self.lineage_edges_total.labels(edge_type=edge_type).inc()

            with self._stats_lock:
                self._in_memory_stats["lineage_edges"] += 1

        except Exception as e:
            logger.error(
                "Failed to record lineage edge: %s", e, exc_info=True
            )

    def record_lineage_depth(self, scope: str, depth: int) -> None:
        """
        Record a lineage graph traversal depth.

        Args:
            scope: GHG emission scope.
            depth: Traversal depth value.
        """
        try:
            scope = self._validate_enum_value(
                scope, ScopeLabel, ScopeLabel.CROSS_CUTTING.value
            )
            self.lineage_depth.labels(scope=scope).observe(depth)

        except Exception as e:
            logger.error(
                "Failed to record lineage depth: %s", e, exc_info=True
            )

    def record_evidence_package(
        self,
        completeness_score: float = 0.0,
        scope: str = "cross_cutting",
    ) -> None:
        """
        Record an evidence package creation.

        Args:
            completeness_score: Evidence completeness score (0.0-1.0).
            scope: GHG emission scope.
        """
        try:
            scope = self._validate_enum_value(
                scope, ScopeLabel, ScopeLabel.CROSS_CUTTING.value
            )
            self.evidence_packages_total.labels(scope=scope).inc()

            if completeness_score > 0:
                self.evidence_completeness_score.labels(
                    scope=scope
                ).observe(completeness_score)

            with self._stats_lock:
                self._in_memory_stats["evidence_packages"] += 1

        except Exception as e:
            logger.error(
                "Failed to record evidence package: %s", e, exc_info=True
            )

    def record_compliance_coverage(
        self,
        coverage_pct: float,
        organization_id: str,
    ) -> None:
        """
        Record compliance coverage percentage.

        Args:
            coverage_pct: Coverage percentage (0-100).
            organization_id: Organization identifier.
        """
        try:
            self.compliance_coverage_pct.labels(
                organization_id=organization_id
            ).set(coverage_pct)

        except Exception as e:
            logger.error(
                "Failed to record compliance coverage: %s", e, exc_info=True
            )

    def record_change(
        self,
        event_type: str,
        count: int = 1,
    ) -> None:
        """
        Record detected changes.

        Args:
            event_type: Type of change event.
            count: Number of changes detected.
        """
        try:
            event_type = self._validate_enum_value(
                event_type, EventTypeLabel, EventTypeLabel.OTHER.value
            )
            self.changes_detected_total.labels(
                event_type=event_type
            ).inc(count)

            with self._stats_lock:
                self._in_memory_stats["changes_detected"] += count

        except Exception as e:
            logger.error(
                "Failed to record change: %s", e, exc_info=True
            )

    def record_recalculation(
        self,
        scope: str,
        category: str,
    ) -> None:
        """
        Record a recalculation trigger.

        Args:
            scope: GHG emission scope.
            category: Emission category.
        """
        try:
            scope = self._validate_enum_value(
                scope, ScopeLabel, ScopeLabel.CROSS_CUTTING.value
            )
            self.recalculations_triggered_total.labels(
                scope=scope, category=category
            ).inc()

            with self._stats_lock:
                self._in_memory_stats["recalculations"] += 1

        except Exception as e:
            logger.error(
                "Failed to record recalculation: %s", e, exc_info=True
            )

    def record_pipeline_execution(
        self,
        status: str,
        duration: float,
        event_type: str = "audit_event",
        scope: str = "cross_cutting",
    ) -> None:
        """
        Record a pipeline execution.

        Args:
            status: Pipeline outcome status.
            duration: Pipeline duration in seconds.
            event_type: Audit event type processed.
            scope: GHG emission scope.
        """
        try:
            status = self._validate_enum_value(
                status, PipelineStatusLabel,
                PipelineStatusLabel.FAILED.value,
            )
            event_type = self._validate_enum_value(
                event_type, EventTypeLabel, EventTypeLabel.OTHER.value
            )
            scope = self._validate_enum_value(
                scope, ScopeLabel, ScopeLabel.CROSS_CUTTING.value
            )

            self.pipeline_executions_total.labels(
                status=status, event_type=event_type, scope=scope
            ).inc()

            if duration > 0:
                self.pipeline_duration_seconds.labels(
                    status=status
                ).observe(duration)

            with self._stats_lock:
                self._in_memory_stats["pipeline_executions"] += 1

        except Exception as e:
            logger.error(
                "Failed to record pipeline execution: %s", e, exc_info=True
            )

    def record_pipeline_stage(self, stage: str, duration: float) -> None:
        """
        Record individual pipeline stage duration.

        Uses the pipeline_duration_seconds histogram with a stage-prefixed
        status label for per-stage granularity.

        Args:
            stage: Pipeline stage name.
            duration: Stage duration in seconds.
        """
        try:
            stage = self._validate_enum_value(
                stage, PipelineStageLabel, PipelineStageLabel.VALIDATE.value
            )
            self.pipeline_duration_seconds.labels(
                status=f"stage_{stage}"
            ).observe(duration)

        except Exception as e:
            logger.error(
                "Failed to record pipeline stage: %s", e, exc_info=True
            )

    # ======================================================================
    # Utility methods
    # ======================================================================

    @staticmethod
    def _validate_enum_value(
        value: str, enum_cls: type, default: str
    ) -> str:
        """
        Validate and normalize a label value against an enum.

        Args:
            value: Raw label value.
            enum_cls: Enum class to validate against.
            default: Default value if validation fails.

        Returns:
            Validated label value string.
        """
        if value is None:
            return default
        value_lower = str(value).lower()
        valid_values = {e.value for e in enum_cls}
        if value_lower in valid_values:
            return value_lower
        return default

    def get_stats(self) -> Dict[str, Any]:
        """
        Get in-memory statistics summary.

        Returns:
            Dictionary of current statistics.
        """
        with self._stats_lock:
            stats = self._in_memory_stats.copy()

        stats["uptime_seconds"] = (
            datetime.utcnow() - self._start_time
        ).total_seconds()
        stats["prometheus_available"] = PROMETHEUS_AVAILABLE
        return stats

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Primarily for test teardown. Clears the singleton so a fresh
        instance is created on next access.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None
                logger.info("AuditTrailLineageMetrics singleton reset")


# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

_metrics_instance: Optional[AuditTrailLineageMetrics] = None
_metrics_lock: threading.Lock = threading.Lock()


def get_metrics() -> AuditTrailLineageMetrics:
    """
    Get the singleton AuditTrailLineageMetrics instance.

    Thread-safe accessor for the global metrics instance.

    Returns:
        AuditTrailLineageMetrics singleton instance.

    Example:
        >>> from greenlang.audit_trail_lineage.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_event(
        ...     event_type="calculation_completed", scope="scope_1",
        ... )
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = AuditTrailLineageMetrics()

    return _metrics_instance


def reset_metrics() -> None:
    """
    Reset the singleton metrics instance for testing purposes.

    Example:
        >>> from greenlang.audit_trail_lineage.metrics import reset_metrics
        >>> reset_metrics()
    """
    global _metrics_instance
    with _metrics_lock:
        _metrics_instance = None
    AuditTrailLineageMetrics.reset()


# ===========================================================================
# Context manager helpers
# ===========================================================================


@contextmanager
def track_event(
    event_type: str = "audit_event",
    scope: str = "cross_cutting",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks an audit event recording lifecycle.

    Automatically measures wall-clock duration and records the event
    when the context exits.

    Args:
        event_type: Audit event type.
        scope: GHG emission scope.

    Yields:
        Mutable context dict. Set additional fields inside the block.

    Example:
        >>> with track_event("calculation_completed", "scope_1") as ctx:
        ...     result = record_audit_event(data)
        ...     ctx["event_id"] = result.event_id
    """
    context: Dict[str, Any] = {
        "event_type": event_type,
        "scope": scope,
        "status": "success",
    }

    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "error"
        raise
    finally:
        duration = time.monotonic() - start
        metrics = get_metrics()
        if context["status"] == "success":
            metrics.record_event(
                event_type=context["event_type"],
                scope=context["scope"],
                duration=duration,
            )
        else:
            with metrics._stats_lock:
                metrics._in_memory_stats["errors"] += 1


@contextmanager
def track_pipeline(
    event_type: str = "audit_event",
    scope: str = "cross_cutting",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that tracks a full pipeline execution lifecycle.

    Automatically measures wall-clock duration and records the pipeline
    execution when the context exits.

    Args:
        event_type: Audit event type being processed.
        scope: GHG emission scope.

    Yields:
        Mutable context dict. Set ``context['status']`` inside the block.

    Example:
        >>> with track_pipeline("calculation_completed", "scope_1") as ctx:
        ...     result = pipeline.execute(...)
        ...     ctx["status"] = result["status"]
    """
    context: Dict[str, Any] = {
        "event_type": event_type,
        "scope": scope,
        "status": "success",
    }

    start = time.monotonic()

    try:
        yield context
    except Exception:
        context["status"] = "failed"
        raise
    finally:
        duration = time.monotonic() - start
        metrics = get_metrics()
        metrics.record_pipeline_execution(
            status=context["status"],
            duration=duration,
            event_type=context["event_type"],
            scope=context["scope"],
        )


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # Availability flag
    "PROMETHEUS_AVAILABLE",
    # Enums
    "EventTypeLabel",
    "ScopeLabel",
    "PipelineStatusLabel",
    "EdgeTypeLabel",
    "ChangeTypeLabel",
    "VerificationResultLabel",
    "PipelineStageLabel",
    # Singleton class
    "AuditTrailLineageMetrics",
    # Module-level accessors
    "get_metrics",
    "reset_metrics",
    # Context managers
    "track_event",
    "track_pipeline",
]
