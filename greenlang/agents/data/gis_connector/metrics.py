# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-006: GIS/Mapping Connector (GL-DATA-GEO-001)

12 Prometheus metrics for GIS connector service monitoring with graceful
fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_gis_connector_operations_total (Counter, labels: type, tenant_id)
    2.  gl_gis_connector_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_gis_connector_validation_errors_total (Counter, labels: severity, type)
    4.  gl_gis_connector_batch_jobs_total (Counter, labels: status)
    5.  gl_gis_connector_active_jobs (Gauge)
    6.  gl_gis_connector_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_gis_connector_format_conversions_total (Counter, labels: source_format, target_format)
    8.  gl_gis_connector_crs_transformations_total (Counter, labels: source_crs, target_crs)
    9.  gl_gis_connector_spatial_queries_total (Counter, labels: query_type, status)
    10. gl_gis_connector_geocoding_requests_total (Counter, labels: direction, status)
    11. gl_gis_connector_features_processed_total (Counter, labels: layer, operation)
    12. gl_gis_connector_data_volume_bytes (Histogram, labels: format)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector
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
    "gl_gis_connector",
    "GIS Connector",
    duration_buckets=(
        0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
    ),
)

# Backward-compat aliases
gis_connector_operations_total = m.operations_total
gis_connector_operation_duration_seconds = m.processing_duration
gis_connector_processing_errors_total = m.validation_errors_total

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

gis_connector_format_conversions_total = m.create_custom_counter(
    "format_conversions_total",
    "Total format conversion operations",
    labelnames=["source_format", "target_format"],
)

gis_connector_crs_transformations_total = m.create_custom_counter(
    "crs_transformations_total",
    "Total CRS transformation operations",
    labelnames=["source_crs", "target_crs"],
)

gis_connector_spatial_queries_total = m.create_custom_counter(
    "spatial_queries_total",
    "Total spatial query operations",
    labelnames=["query_type", "status"],
)

gis_connector_geocoding_requests_total = m.create_custom_counter(
    "geocoding_requests_total",
    "Total geocoding requests",
    labelnames=["direction", "status"],
)

gis_connector_features_processed_total = m.create_custom_counter(
    "features_processed_total",
    "Total features processed",
    labelnames=["layer", "operation"],
)

gis_connector_active_layers = m.create_custom_gauge(
    "active_layers",
    "Number of currently active layers",
)

gis_connector_layer_features_count = m.create_custom_gauge(
    "layer_features_count",
    "Number of features per layer",
    labelnames=["layer"],
)

gis_connector_cache_hit_rate = m.create_custom_gauge(
    "cache_hit_rate",
    "Cache hit rate (0-1) per cache type",
    labelnames=["cache_type"],
)

gis_connector_data_volume_bytes = m.create_custom_histogram(
    "data_volume_bytes",
    "Data volume processed in bytes",
    buckets=(
        256, 1024, 4096, 16384, 65536, 262144,
        1048576, 4194304, 16777216, 67108864,
    ),
    labelnames=["format"],
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_operation(
    operation: str,
    format: str,
    status: str,
    duration: float = 0.0,
) -> None:
    """Record a GIS operation event.

    Args:
        operation: Operation type (format_parse, crs_transform, etc.).
        format: Data format or geometry type.
        status: Result status (success, error).
        duration: Execution duration in seconds.
    """
    m.record_operation(duration, type=f"{operation}:{format}", tenant_id=status)


def record_format_conversion(
    source_format: str,
    target_format: str,
) -> None:
    """Record a format conversion event.

    Args:
        source_format: Source data format.
        target_format: Target data format.
    """
    m.safe_inc(
        gis_connector_format_conversions_total, 1,
        source_format=source_format, target_format=target_format,
    )


def record_crs_transformation(
    source_crs: str,
    target_crs: str,
) -> None:
    """Record a CRS transformation event.

    Args:
        source_crs: Source CRS identifier.
        target_crs: Target CRS identifier.
    """
    m.safe_inc(
        gis_connector_crs_transformations_total, 1,
        source_crs=source_crs, target_crs=target_crs,
    )


def record_spatial_query(
    query_type: str,
    status: str,
) -> None:
    """Record a spatial query event.

    Args:
        query_type: Query type (distance, area, contains, etc.).
        status: Result status (success, error).
    """
    m.safe_inc(
        gis_connector_spatial_queries_total, 1,
        query_type=query_type, status=status,
    )


def record_geocoding_request(
    direction: str,
    status: str,
) -> None:
    """Record a geocoding request event.

    Args:
        direction: Direction (forward, reverse).
        status: Result status (success, no_results, error).
    """
    m.safe_inc(
        gis_connector_geocoding_requests_total, 1,
        direction=direction, status=status,
    )


def record_features_processed(
    layer: str,
    operation: str,
) -> None:
    """Record features processed event.

    Args:
        layer: Layer identifier.
        operation: Operation type (add, update, delete).
    """
    m.safe_inc(
        gis_connector_features_processed_total, 1,
        layer=layer, operation=operation,
    )


def record_processing_error(
    operation: str,
    error_type: str,
) -> None:
    """Record a processing error event.

    Args:
        operation: Operation that failed.
        error_type: Type of error encountered.
    """
    m.record_validation_error(severity=operation, type=error_type)


def update_active_layers(count: int) -> None:
    """Update the active layers gauge.

    Args:
        count: Current number of active layers.
    """
    m.safe_set(gis_connector_active_layers, count)


def update_layer_features(layer: str, count: int) -> None:
    """Update the feature count gauge for a layer.

    Args:
        layer: Layer identifier.
        count: Current feature count.
    """
    m.safe_set(gis_connector_layer_features_count, count, layer=layer)


def update_cache_hit_rate(cache_type: str, rate: float) -> None:
    """Update the cache hit rate gauge.

    Args:
        cache_type: Cache type (geocoding, format_parse, etc.).
        rate: Hit rate as float 0-1.
    """
    m.safe_set(gis_connector_cache_hit_rate, rate, cache_type=cache_type)


def record_data_volume(format: str, size_bytes: int) -> None:
    """Record a data volume observation.

    Args:
        format: Data format identifier.
        size_bytes: Data size in bytes.
    """
    m.safe_observe(gis_connector_data_volume_bytes, size_bytes, format=format)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "gis_connector_operations_total",
    "gis_connector_operation_duration_seconds",
    "gis_connector_format_conversions_total",
    "gis_connector_crs_transformations_total",
    "gis_connector_spatial_queries_total",
    "gis_connector_geocoding_requests_total",
    "gis_connector_features_processed_total",
    "gis_connector_active_layers",
    "gis_connector_layer_features_count",
    "gis_connector_processing_errors_total",
    "gis_connector_cache_hit_rate",
    "gis_connector_data_volume_bytes",
    # Helper functions
    "record_operation",
    "record_format_conversion",
    "record_crs_transformation",
    "record_spatial_query",
    "record_geocoding_request",
    "record_features_processed",
    "record_processing_error",
    "update_active_layers",
    "update_layer_features",
    "update_cache_hit_rate",
    "record_data_volume",
]
