# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-001: Supply Chain Mapping Master

15 Prometheus metrics for supply chain mapping agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_scm_`` prefix (GreenLang EUDR Supply
Chain Mapper) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_eudr_scm_graphs_created_total           (Counter)
    2.  gl_eudr_scm_nodes_added_total              (Counter, labels: node_type)
    3.  gl_eudr_scm_edges_added_total              (Counter)
    4.  gl_eudr_scm_tier_discovery_total           (Counter)
    5.  gl_eudr_scm_trace_operations_total         (Counter, labels: direction)
    6.  gl_eudr_scm_risk_propagations_total        (Counter)
    7.  gl_eudr_scm_gaps_detected_total            (Counter, labels: gap_type, severity)
    8.  gl_eudr_scm_gaps_resolved_total            (Counter)
    9.  gl_eudr_scm_dds_exports_total              (Counter)
    10. gl_eudr_scm_processing_duration_seconds    (Histogram, labels: operation)
    11. gl_eudr_scm_graph_query_duration_seconds   (Histogram)
    12. gl_eudr_scm_errors_total                   (Counter, labels: operation)
    13. gl_eudr_scm_active_graphs                  (Gauge)
    14. gl_eudr_scm_total_nodes                    (Gauge)
    15. gl_eudr_scm_compliance_readiness_avg       (Gauge)

Label Values Reference:
    node_type:
        producer, collector, processor, trader, importer,
        certifier, warehouse, port.
    direction:
        forward, backward.
    gap_type:
        missing_geolocation, missing_polygon, broken_custody_chain,
        unverified_actor, missing_tier, mass_balance_discrepancy,
        missing_certification, stale_data, orphan_node,
        missing_documentation.
    severity:
        critical, high, medium, low.
    operation:
        graph_create, node_add, edge_add, tier_discover,
        trace_forward, trace_backward, risk_propagate,
        gap_analyze, dds_export, graph_query, batch_trace,
        provenance_hash, snapshot_create.

Example:
    >>> from greenlang.agents.eudr.supply_chain_mapper.metrics import (
    ...     record_graph_created,
    ...     record_node_added,
    ...     observe_processing_duration,
    ...     set_active_graphs,
    ... )
    >>> record_graph_created()
    >>> record_node_added("producer")
    >>> observe_processing_duration("graph_create", 0.125)
    >>> set_active_graphs(42)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; "
        "supply chain mapper metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(name: str, doc: str, labelnames: list = None):
        """Create a Counter or retrieve existing one to avoid registry collisions."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            # Last resort: create in isolated registry (still functional)
            from prometheus_client import CollectorRegistry
            return Counter(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

    def _safe_histogram(name: str, doc: str, labelnames: list = None,
                        buckets: tuple = ()):
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [], **kw)
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(), **kw,
            )

    def _safe_gauge(name: str, doc: str, labelnames: list = None):
        """Create a Gauge or retrieve existing one."""
        try:
            return Gauge(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Gauge(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

# ---------------------------------------------------------------------------
# Metric definitions (15 metrics per PRD Section 7.6)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Supply chain graphs created
    scm_graphs_created_total = _safe_counter(
        "gl_eudr_scm_graphs_created_total",
        "Total supply chain graphs created",
    )

    # 2. Nodes added to graphs by node type
    scm_nodes_added_total = _safe_counter(
        "gl_eudr_scm_nodes_added_total",
        "Total nodes added to supply chain graphs by node type",
        labelnames=["node_type"],
    )

    # 3. Edges added to graphs
    scm_edges_added_total = _safe_counter(
        "gl_eudr_scm_edges_added_total",
        "Total custody transfer edges added to supply chain graphs",
    )

    # 4. Multi-tier discovery operations
    scm_tier_discovery_total = _safe_counter(
        "gl_eudr_scm_tier_discovery_total",
        "Total multi-tier recursive discovery operations performed",
    )

    # 5. Forward/backward trace operations by direction
    scm_trace_operations_total = _safe_counter(
        "gl_eudr_scm_trace_operations_total",
        "Total forward and backward trace operations performed",
        labelnames=["direction"],
    )

    # 6. Risk propagation runs
    scm_risk_propagations_total = _safe_counter(
        "gl_eudr_scm_risk_propagations_total",
        "Total risk propagation runs across supply chain graphs",
    )

    # 7. Gaps detected by type and severity
    scm_gaps_detected_total = _safe_counter(
        "gl_eudr_scm_gaps_detected_total",
        "Total compliance gaps detected by type and severity",
        labelnames=["gap_type", "severity"],
    )

    # 8. Gaps resolved
    scm_gaps_resolved_total = _safe_counter(
        "gl_eudr_scm_gaps_resolved_total",
        "Total compliance gaps resolved",
    )

    # 9. DDS export operations
    scm_dds_exports_total = _safe_counter(
        "gl_eudr_scm_dds_exports_total",
        "Total DDS export operations performed",
    )

    # 10. Processing operation latency by operation type
    scm_processing_duration_seconds = _safe_histogram(
        "gl_eudr_scm_processing_duration_seconds",
        "Duration of supply chain mapping operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 11. Graph query latency
    scm_graph_query_duration_seconds = _safe_histogram(
        "gl_eudr_scm_graph_query_duration_seconds",
        "Duration of supply chain graph query operations in seconds",
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        ),
    )

    # 12. Errors by operation type
    scm_errors_total = _safe_counter(
        "gl_eudr_scm_errors_total",
        "Total errors encountered by operation type",
        labelnames=["operation"],
    )

    # 13. Currently active supply chain graphs
    scm_active_graphs = _safe_gauge(
        "gl_eudr_scm_active_graphs",
        "Number of currently active supply chain graphs",
    )

    # 14. Total nodes across all graphs
    scm_total_nodes = _safe_gauge(
        "gl_eudr_scm_total_nodes",
        "Total number of nodes across all active supply chain graphs",
    )

    # 15. Average compliance readiness score
    scm_compliance_readiness_avg = _safe_gauge(
        "gl_eudr_scm_compliance_readiness_avg",
        "Average compliance readiness score across all active graphs",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    scm_graphs_created_total = None               # type: ignore[assignment]
    scm_nodes_added_total = None                   # type: ignore[assignment]
    scm_edges_added_total = None                   # type: ignore[assignment]
    scm_tier_discovery_total = None                # type: ignore[assignment]
    scm_trace_operations_total = None              # type: ignore[assignment]
    scm_risk_propagations_total = None             # type: ignore[assignment]
    scm_gaps_detected_total = None                 # type: ignore[assignment]
    scm_gaps_resolved_total = None                 # type: ignore[assignment]
    scm_dds_exports_total = None                   # type: ignore[assignment]
    scm_processing_duration_seconds = None         # type: ignore[assignment]
    scm_graph_query_duration_seconds = None        # type: ignore[assignment]
    scm_errors_total = None                        # type: ignore[assignment]
    scm_active_graphs = None                       # type: ignore[assignment]
    scm_total_nodes = None                         # type: ignore[assignment]
    scm_compliance_readiness_avg = None            # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_graph_created() -> None:
    """Record a supply chain graph creation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    scm_graphs_created_total.inc()


def record_node_added(node_type: str) -> None:
    """Record a node addition event.

    Args:
        node_type: Type of node added (producer, collector, processor,
            trader, importer, certifier, warehouse, port).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_nodes_added_total.labels(node_type=node_type).inc()


def record_edge_added() -> None:
    """Record a custody transfer edge addition event."""
    if not PROMETHEUS_AVAILABLE:
        return
    scm_edges_added_total.inc()


def record_tier_discovery() -> None:
    """Record a multi-tier recursive discovery operation."""
    if not PROMETHEUS_AVAILABLE:
        return
    scm_tier_discovery_total.inc()


def record_trace_operation(direction: str) -> None:
    """Record a forward or backward trace operation.

    Args:
        direction: Trace direction ('forward' or 'backward').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_trace_operations_total.labels(direction=direction).inc()


def record_risk_propagation() -> None:
    """Record a risk propagation run."""
    if not PROMETHEUS_AVAILABLE:
        return
    scm_risk_propagations_total.inc()


def record_gap_detected(gap_type: str, severity: str) -> None:
    """Record a compliance gap detection event.

    Args:
        gap_type: Classification of the gap (missing_geolocation,
            missing_polygon, broken_custody_chain, etc.).
        severity: Severity level (critical, high, medium, low).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_gaps_detected_total.labels(
        gap_type=gap_type,
        severity=severity,
    ).inc()


def record_gap_resolved() -> None:
    """Record a compliance gap resolution event."""
    if not PROMETHEUS_AVAILABLE:
        return
    scm_gaps_resolved_total.inc()


def record_dds_export() -> None:
    """Record a DDS export operation."""
    if not PROMETHEUS_AVAILABLE:
        return
    scm_dds_exports_total.inc()


def observe_processing_duration(operation: str, seconds: float) -> None:
    """Record the duration of a processing operation.

    Args:
        operation: Type of operation being measured (graph_create,
            node_add, edge_add, tier_discover, trace_forward,
            trace_backward, risk_propagate, gap_analyze,
            dds_export, batch_trace, provenance_hash,
            snapshot_create).
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_processing_duration_seconds.labels(operation=operation).observe(
        seconds
    )


def observe_graph_query_duration(seconds: float) -> None:
    """Record the duration of a graph query operation.

    Args:
        seconds: Query wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_graph_query_duration_seconds.observe(seconds)


def record_error(operation: str) -> None:
    """Record an error event by operation type.

    Args:
        operation: Type of operation that failed (graph_create,
            node_add, edge_add, risk_propagate, gap_analyze,
            dds_export, trace_forward, trace_backward, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_errors_total.labels(operation=operation).inc()


def set_active_graphs(count: int) -> None:
    """Set the gauge for currently active supply chain graphs.

    Args:
        count: Number of active graphs. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_active_graphs.set(count)


def set_total_nodes(count: int) -> None:
    """Set the gauge for total nodes across all graphs.

    Args:
        count: Total node count. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_total_nodes.set(count)


def set_compliance_readiness_avg(score: float) -> None:
    """Set the gauge for average compliance readiness score.

    Args:
        score: Average compliance readiness (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scm_compliance_readiness_avg.set(score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "scm_graphs_created_total",
    "scm_nodes_added_total",
    "scm_edges_added_total",
    "scm_tier_discovery_total",
    "scm_trace_operations_total",
    "scm_risk_propagations_total",
    "scm_gaps_detected_total",
    "scm_gaps_resolved_total",
    "scm_dds_exports_total",
    "scm_processing_duration_seconds",
    "scm_graph_query_duration_seconds",
    "scm_errors_total",
    "scm_active_graphs",
    "scm_total_nodes",
    "scm_compliance_readiness_avg",
    # Helper functions
    "record_graph_created",
    "record_node_added",
    "record_edge_added",
    "record_tier_discovery",
    "record_trace_operation",
    "record_risk_propagation",
    "record_gap_detected",
    "record_gap_resolved",
    "record_dds_export",
    "observe_processing_duration",
    "observe_graph_query_duration",
    "record_error",
    "set_active_graphs",
    "set_total_nodes",
    "set_compliance_readiness_avg",
]
