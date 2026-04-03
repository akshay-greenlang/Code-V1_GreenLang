# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-018: Data Lineage Tracker

12 Prometheus metrics for data lineage tracking service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_dlt_operations_total (Counter, labels: type, tenant_id)
    2.  gl_dlt_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_dlt_validation_errors_total (Counter, labels: severity, type)
    4.  gl_dlt_batch_jobs_total (Counter, labels: status)
    5.  gl_dlt_active_jobs (Gauge)
    6.  gl_dlt_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_dlt_assets_registered_total (Counter, labels: asset_type, classification)
    8.  gl_dlt_transformations_captured_total (Counter, labels: transformation_type, agent_id)
    9.  gl_dlt_edges_created_total (Counter, labels: edge_type)
    10. gl_dlt_graph_traversal_duration_seconds (Histogram, buckets: traversal-scale)
    11. gl_dlt_graph_node_count (Gauge)
    12. gl_dlt_graph_edge_count (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_dlt",
    "Data Lineage Tracker",
    duration_buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10),
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

dlt_assets_registered_total = m.create_custom_counter(
    "assets_registered_total",
    "Total data assets registered in the lineage graph",
    labelnames=["asset_type", "classification"],
)

dlt_transformations_captured_total = m.create_custom_counter(
    "transformations_captured_total",
    "Total data transformations captured in the lineage graph",
    labelnames=["transformation_type", "agent_id"],
)

dlt_edges_created_total = m.create_custom_counter(
    "edges_created_total",
    "Total lineage edges created between data assets",
    labelnames=["edge_type"],
)

dlt_impact_analyses_total = m.create_custom_counter(
    "impact_analyses_total",
    "Total impact analyses performed on the lineage graph",
    labelnames=["direction", "severity"],
)

dlt_validations_total = m.create_custom_counter(
    "validations_total",
    "Total lineage validation checks performed",
    labelnames=["result"],
)

dlt_reports_generated_total = m.create_custom_counter(
    "reports_generated_total",
    "Total lineage reports generated",
    labelnames=["report_type", "format"],
)

dlt_change_events_total = m.create_custom_counter(
    "change_events_total",
    "Total lineage change events detected and recorded",
    labelnames=["change_type", "severity"],
)

dlt_quality_scores_computed_total = m.create_custom_counter(
    "quality_scores_computed_total",
    "Total lineage quality scores computed for data assets",
    labelnames=["score_tier"],
)

dlt_graph_traversal_duration_seconds = m.create_custom_histogram(
    "graph_traversal_duration_seconds",
    "Duration of lineage graph traversal operations in seconds",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30),
)

dlt_processing_duration_seconds = m.create_custom_histogram(
    "processing_duration_seconds_detail",
    "Data lineage tracker engine operation processing duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10),
    labelnames=["operation"],
)

dlt_graph_node_count = m.create_custom_gauge(
    "graph_node_count",
    "Current number of nodes in the data lineage graph",
)

dlt_graph_edge_count = m.create_custom_gauge(
    "graph_edge_count",
    "Current number of edges in the data lineage graph",
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_asset_registered(asset_type: str, classification: str) -> None:
    """Record a data asset registration event.

    Args:
        asset_type: Type of data asset registered
            (dataset, table, column, file, stream, api_endpoint, model,
            report, dashboard, metric).
        classification: Data classification level
            (public, internal, confidential, restricted, pii).
    """
    m.safe_inc(
        dlt_assets_registered_total, 1,
        asset_type=asset_type, classification=classification,
    )


def record_transformation_captured(
    transformation_type: str, agent_id: str
) -> None:
    """Record a data transformation capture event.

    Args:
        transformation_type: Type of transformation captured
            (filter, aggregate, join, map, enrich, normalize, validate,
            deduplicate, impute, classify, extract, convert).
        agent_id: Identifier of the agent that performed the transformation
            (e.g. ``"data-quality-profiler"``, ``"excel-normalizer"``).
    """
    m.safe_inc(
        dlt_transformations_captured_total, 1,
        transformation_type=transformation_type, agent_id=agent_id,
    )


def record_edge_created(edge_type: str) -> None:
    """Record a lineage edge creation event.

    Args:
        edge_type: Type of lineage edge created
            (derived_from, consumed_by, produced_by, transformed_by,
            validated_by, enriched_by, copied_from, merged_from).
    """
    m.safe_inc(dlt_edges_created_total, 1, edge_type=edge_type)


def record_impact_analysis(direction: str, severity: str) -> None:
    """Record an impact analysis execution event.

    Args:
        direction: Traversal direction for impact analysis
            (upstream, downstream, bidirectional).
        severity: Highest severity level found during analysis
            (critical, high, medium, low, none).
    """
    m.safe_inc(dlt_impact_analyses_total, 1, direction=direction, severity=severity)


def record_validation(result: str) -> None:
    """Record a lineage validation result.

    Args:
        result: Validation outcome
            (pass, fail, warning, error, skipped).
    """
    m.safe_inc(dlt_validations_total, 1, result=result)


def record_report_generated(report_type: str, format: str) -> None:
    """Record a lineage report generation event.

    Args:
        report_type: Type of lineage report generated
            (full_lineage, impact_analysis, compliance, audit_trail,
            data_flow, quality_summary, change_history).
        format: Output format of the generated report
            (json, html, pdf, csv, markdown, graphviz).
    """
    m.safe_inc(
        dlt_reports_generated_total, 1,
        report_type=report_type, format=format,
    )


def record_change_event(change_type: str, severity: str) -> None:
    """Record a lineage change event.

    Args:
        change_type: Type of change detected in the lineage graph
            (asset_added, asset_removed, asset_modified, edge_added,
            edge_removed, schema_changed, classification_changed,
            owner_changed, status_changed).
        severity: Severity level of the change
            (critical, high, medium, low, informational).
    """
    m.safe_inc(
        dlt_change_events_total, 1,
        change_type=change_type, severity=severity,
    )


def record_quality_score(score_tier: str) -> None:
    """Record a lineage quality score computation event.

    Args:
        score_tier: Quality score tier computed for the data asset
            (excellent, good, fair, poor, critical).
    """
    m.safe_inc(dlt_quality_scores_computed_total, 1, score_tier=score_tier)


def observe_graph_traversal_duration(seconds: float) -> None:
    """Record the duration of a lineage graph traversal operation.

    Args:
        seconds: Graph traversal wall-clock time in seconds.
    """
    m.safe_observe(dlt_graph_traversal_duration_seconds, seconds)


def observe_processing_duration(operation: str, seconds: float) -> None:
    """Record processing duration for a single engine operation.

    Args:
        operation: Operation type label
            (asset_register, transformation_capture, edge_create,
            impact_analyze, validate, report_generate, change_detect,
            quality_score, graph_query, export, import).
        seconds: Operation duration in seconds.
    """
    m.safe_observe(dlt_processing_duration_seconds, seconds, operation=operation)


def set_graph_node_count(count: int) -> None:
    """Set the gauge for current number of nodes in the lineage graph.

    Args:
        count: Number of nodes currently in the lineage graph.
    """
    m.safe_set(dlt_graph_node_count, count)


def set_graph_edge_count(count: int) -> None:
    """Set the gauge for current number of edges in the lineage graph.

    Args:
        count: Number of edges currently in the lineage graph.
    """
    m.safe_set(dlt_graph_edge_count, count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "dlt_assets_registered_total",
    "dlt_transformations_captured_total",
    "dlt_edges_created_total",
    "dlt_impact_analyses_total",
    "dlt_validations_total",
    "dlt_reports_generated_total",
    "dlt_change_events_total",
    "dlt_quality_scores_computed_total",
    "dlt_graph_traversal_duration_seconds",
    "dlt_processing_duration_seconds",
    "dlt_graph_node_count",
    "dlt_graph_edge_count",
    # Helper functions
    "record_asset_registered",
    "record_transformation_captured",
    "record_edge_created",
    "record_impact_analysis",
    "record_validation",
    "record_report_generated",
    "record_change_event",
    "record_quality_score",
    "observe_graph_traversal_duration",
    "observe_processing_duration",
    "set_graph_node_count",
    "set_graph_edge_count",
]
