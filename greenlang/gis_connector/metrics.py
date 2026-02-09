# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-006: GIS/Mapping Connector (GL-DATA-GEO-001)

12 Prometheus metrics for GIS connector service monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_gis_connector_operations_total (Counter, labels: operation, format, status)
    2.  gl_gis_connector_operation_duration_seconds (Histogram, labels: operation, format)
    3.  gl_gis_connector_format_conversions_total (Counter, labels: source_format, target_format)
    4.  gl_gis_connector_crs_transformations_total (Counter, labels: source_crs, target_crs)
    5.  gl_gis_connector_spatial_queries_total (Counter, labels: query_type, status)
    6.  gl_gis_connector_geocoding_requests_total (Counter, labels: direction, status)
    7.  gl_gis_connector_features_processed_total (Counter, labels: layer, operation)
    8.  gl_gis_connector_active_layers (Gauge)
    9.  gl_gis_connector_layer_features_count (Gauge, labels: layer)
    10. gl_gis_connector_processing_errors_total (Counter, labels: operation, error_type)
    11. gl_gis_connector_cache_hit_rate (Gauge, labels: cache_type)
    12. gl_gis_connector_data_volume_bytes (Histogram, labels: format)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector
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
        "prometheus_client not installed; GIS connector metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Total operations by operation, format, and status
    gis_connector_operations_total = Counter(
        "gl_gis_connector_operations_total",
        "Total GIS connector operations executed",
        labelnames=["operation", "format", "status"],
    )

    # 2. Operation duration histogram by operation and format
    gis_connector_operation_duration_seconds = Histogram(
        "gl_gis_connector_operation_duration_seconds",
        "GIS connector operation duration in seconds",
        labelnames=["operation", "format"],
        buckets=(
            0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
        ),
    )

    # 3. Format conversions by source and target format
    gis_connector_format_conversions_total = Counter(
        "gl_gis_connector_format_conversions_total",
        "Total format conversion operations",
        labelnames=["source_format", "target_format"],
    )

    # 4. CRS transformations by source and target CRS
    gis_connector_crs_transformations_total = Counter(
        "gl_gis_connector_crs_transformations_total",
        "Total CRS transformation operations",
        labelnames=["source_crs", "target_crs"],
    )

    # 5. Spatial queries by query type and status
    gis_connector_spatial_queries_total = Counter(
        "gl_gis_connector_spatial_queries_total",
        "Total spatial query operations",
        labelnames=["query_type", "status"],
    )

    # 6. Geocoding requests by direction and status
    gis_connector_geocoding_requests_total = Counter(
        "gl_gis_connector_geocoding_requests_total",
        "Total geocoding requests",
        labelnames=["direction", "status"],
    )

    # 7. Features processed by layer and operation
    gis_connector_features_processed_total = Counter(
        "gl_gis_connector_features_processed_total",
        "Total features processed",
        labelnames=["layer", "operation"],
    )

    # 8. Active layers gauge
    gis_connector_active_layers = Gauge(
        "gl_gis_connector_active_layers",
        "Number of currently active layers",
    )

    # 9. Layer feature count gauge by layer
    gis_connector_layer_features_count = Gauge(
        "gl_gis_connector_layer_features_count",
        "Number of features per layer",
        labelnames=["layer"],
    )

    # 10. Processing errors by operation and error type
    gis_connector_processing_errors_total = Counter(
        "gl_gis_connector_processing_errors_total",
        "Total GIS connector processing errors",
        labelnames=["operation", "error_type"],
    )

    # 11. Cache hit rate gauge by cache type
    gis_connector_cache_hit_rate = Gauge(
        "gl_gis_connector_cache_hit_rate",
        "Cache hit rate (0-1) per cache type",
        labelnames=["cache_type"],
    )

    # 12. Data volume histogram by format
    gis_connector_data_volume_bytes = Histogram(
        "gl_gis_connector_data_volume_bytes",
        "Data volume processed in bytes",
        labelnames=["format"],
        buckets=(
            256, 1024, 4096, 16384, 65536, 262144,
            1048576, 4194304, 16777216, 67108864,
        ),
    )

else:
    # No-op placeholders
    gis_connector_operations_total = None  # type: ignore[assignment]
    gis_connector_operation_duration_seconds = None  # type: ignore[assignment]
    gis_connector_format_conversions_total = None  # type: ignore[assignment]
    gis_connector_crs_transformations_total = None  # type: ignore[assignment]
    gis_connector_spatial_queries_total = None  # type: ignore[assignment]
    gis_connector_geocoding_requests_total = None  # type: ignore[assignment]
    gis_connector_features_processed_total = None  # type: ignore[assignment]
    gis_connector_active_layers = None  # type: ignore[assignment]
    gis_connector_layer_features_count = None  # type: ignore[assignment]
    gis_connector_processing_errors_total = None  # type: ignore[assignment]
    gis_connector_cache_hit_rate = None  # type: ignore[assignment]
    gis_connector_data_volume_bytes = None  # type: ignore[assignment]


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
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_operations_total.labels(
        operation=operation, format=format, status=status,
    ).inc()
    if duration > 0:
        gis_connector_operation_duration_seconds.labels(
            operation=operation, format=format,
        ).observe(duration)


def record_format_conversion(
    source_format: str,
    target_format: str,
) -> None:
    """Record a format conversion event.

    Args:
        source_format: Source data format.
        target_format: Target data format.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_format_conversions_total.labels(
        source_format=source_format, target_format=target_format,
    ).inc()


def record_crs_transformation(
    source_crs: str,
    target_crs: str,
) -> None:
    """Record a CRS transformation event.

    Args:
        source_crs: Source CRS identifier.
        target_crs: Target CRS identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_crs_transformations_total.labels(
        source_crs=source_crs, target_crs=target_crs,
    ).inc()


def record_spatial_query(
    query_type: str,
    status: str,
) -> None:
    """Record a spatial query event.

    Args:
        query_type: Query type (distance, area, contains, etc.).
        status: Result status (success, error).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_spatial_queries_total.labels(
        query_type=query_type, status=status,
    ).inc()


def record_geocoding_request(
    direction: str,
    status: str,
) -> None:
    """Record a geocoding request event.

    Args:
        direction: Direction (forward, reverse).
        status: Result status (success, no_results, error).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_geocoding_requests_total.labels(
        direction=direction, status=status,
    ).inc()


def record_features_processed(
    layer: str,
    operation: str,
) -> None:
    """Record features processed event.

    Args:
        layer: Layer identifier.
        operation: Operation type (add, update, delete).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_features_processed_total.labels(
        layer=layer, operation=operation,
    ).inc()


def record_processing_error(
    operation: str,
    error_type: str,
) -> None:
    """Record a processing error event.

    Args:
        operation: Operation that failed.
        error_type: Type of error encountered.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_processing_errors_total.labels(
        operation=operation, error_type=error_type,
    ).inc()


def update_active_layers(count: int) -> None:
    """Update the active layers gauge.

    Args:
        count: Current number of active layers.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_active_layers.set(count)


def update_layer_features(layer: str, count: int) -> None:
    """Update the feature count gauge for a layer.

    Args:
        layer: Layer identifier.
        count: Current feature count.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_layer_features_count.labels(layer=layer).set(count)


def update_cache_hit_rate(cache_type: str, rate: float) -> None:
    """Update the cache hit rate gauge.

    Args:
        cache_type: Cache type (geocoding, format_parse, etc.).
        rate: Hit rate as float 0-1.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_cache_hit_rate.labels(cache_type=cache_type).set(rate)


def record_data_volume(format: str, size_bytes: int) -> None:
    """Record a data volume observation.

    Args:
        format: Data format identifier.
        size_bytes: Data size in bytes.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gis_connector_data_volume_bytes.labels(format=format).observe(size_bytes)


__all__ = [
    "PROMETHEUS_AVAILABLE",
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
