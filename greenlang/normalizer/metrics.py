# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-003: Unit & Reference Normalizer

12 Prometheus metrics for normalizer monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_normalizer_conversions_total (Counter)
    2.  gl_normalizer_conversion_duration_seconds (Histogram)
    3.  gl_normalizer_entity_resolutions_total (Counter)
    4.  gl_normalizer_resolution_duration_seconds (Histogram)
    5.  gl_normalizer_dimension_errors_total (Counter)
    6.  gl_normalizer_gwp_conversions_total (Counter)
    7.  gl_normalizer_batch_size (Histogram)
    8.  gl_normalizer_vocabulary_entries (Gauge)
    9.  gl_normalizer_cache_hits_total (Counter)
    10. gl_normalizer_cache_misses_total (Counter)
    11. gl_normalizer_active_conversions (Gauge)
    12. gl_normalizer_custom_factors (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Optional

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
        "prometheus_client not installed; normalizer metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Conversion count
    normalizer_conversions_total = Counter(
        "gl_normalizer_conversions_total",
        "Total unit conversions performed",
        labelnames=["dimension", "result"],
    )

    # 2. Conversion duration
    normalizer_conversion_duration_seconds = Histogram(
        "gl_normalizer_conversion_duration_seconds",
        "Unit conversion duration in seconds",
        labelnames=["dimension"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )

    # 3. Entity resolution count
    normalizer_entity_resolutions_total = Counter(
        "gl_normalizer_entity_resolutions_total",
        "Total entity resolutions performed",
        labelnames=["entity_type", "confidence_level"],
    )

    # 4. Resolution duration
    normalizer_resolution_duration_seconds = Histogram(
        "gl_normalizer_resolution_duration_seconds",
        "Entity resolution duration in seconds",
        labelnames=["entity_type"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )

    # 5. Dimension errors
    normalizer_dimension_errors_total = Counter(
        "gl_normalizer_dimension_errors_total",
        "Total cross-dimension conversion errors",
    )

    # 6. GWP conversions
    normalizer_gwp_conversions_total = Counter(
        "gl_normalizer_gwp_conversions_total",
        "Total GWP-based GHG conversions",
        labelnames=["gas_type", "gwp_version"],
    )

    # 7. Batch size
    normalizer_batch_size = Histogram(
        "gl_normalizer_batch_size",
        "Number of items in batch conversion requests",
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
    )

    # 8. Vocabulary entries
    normalizer_vocabulary_entries = Gauge(
        "gl_normalizer_vocabulary_entries",
        "Number of entries in entity vocabulary",
        labelnames=["entity_type"],
    )

    # 9. Cache hits
    normalizer_cache_hits_total = Counter(
        "gl_normalizer_cache_hits_total",
        "Total normalizer cache hits",
    )

    # 10. Cache misses
    normalizer_cache_misses_total = Counter(
        "gl_normalizer_cache_misses_total",
        "Total normalizer cache misses",
    )

    # 11. Active conversions
    normalizer_active_conversions = Gauge(
        "gl_normalizer_active_conversions",
        "Number of currently active conversions",
    )

    # 12. Custom factors
    normalizer_custom_factors = Gauge(
        "gl_normalizer_custom_factors",
        "Number of custom conversion factors per tenant",
        labelnames=["tenant_id"],
    )

else:
    # No-op placeholders
    normalizer_conversions_total = None  # type: ignore[assignment]
    normalizer_conversion_duration_seconds = None  # type: ignore[assignment]
    normalizer_entity_resolutions_total = None  # type: ignore[assignment]
    normalizer_resolution_duration_seconds = None  # type: ignore[assignment]
    normalizer_dimension_errors_total = None  # type: ignore[assignment]
    normalizer_gwp_conversions_total = None  # type: ignore[assignment]
    normalizer_batch_size = None  # type: ignore[assignment]
    normalizer_vocabulary_entries = None  # type: ignore[assignment]
    normalizer_cache_hits_total = None  # type: ignore[assignment]
    normalizer_cache_misses_total = None  # type: ignore[assignment]
    normalizer_active_conversions = None  # type: ignore[assignment]
    normalizer_custom_factors = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_conversion(
    dimension: str, result: str, duration_seconds: float,
) -> None:
    """Record a unit conversion completion.

    Args:
        dimension: Unit dimension (e.g., "mass", "energy").
        result: Conversion result ("success" or "error").
        duration_seconds: Conversion duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_conversions_total.labels(dimension=dimension, result=result).inc()
    normalizer_conversion_duration_seconds.labels(dimension=dimension).observe(
        duration_seconds,
    )


def record_entity_resolution(
    entity_type: str, confidence_level: str, duration_seconds: float,
) -> None:
    """Record an entity resolution completion.

    Args:
        entity_type: Entity type (fuel, material, process).
        confidence_level: Confidence level of the match.
        duration_seconds: Resolution duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_entity_resolutions_total.labels(
        entity_type=entity_type, confidence_level=confidence_level,
    ).inc()
    normalizer_resolution_duration_seconds.labels(
        entity_type=entity_type,
    ).observe(duration_seconds)


def record_dimension_error() -> None:
    """Record a cross-dimension conversion error."""
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_dimension_errors_total.inc()


def record_gwp_conversion(gas_type: str, gwp_version: str) -> None:
    """Record a GWP-based GHG conversion.

    Args:
        gas_type: Greenhouse gas type.
        gwp_version: IPCC AR version used.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_gwp_conversions_total.labels(
        gas_type=gas_type, gwp_version=gwp_version,
    ).inc()


def record_batch(batch_size: int) -> None:
    """Record a batch conversion request size.

    Args:
        batch_size: Number of items in the batch.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_batch_size.observe(batch_size)


def update_vocabulary_entries(entity_type: str, count: int) -> None:
    """Set the vocabulary entries gauge for an entity type.

    Args:
        entity_type: Entity type (fuel, material, process).
        count: Number of vocabulary entries.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_vocabulary_entries.labels(entity_type=entity_type).set(count)


def record_cache_hit() -> None:
    """Record a normalizer cache hit."""
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_cache_hits_total.inc()


def record_cache_miss() -> None:
    """Record a normalizer cache miss."""
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_cache_misses_total.inc()


def update_active_conversions(delta: int) -> None:
    """Update the active conversions gauge.

    Args:
        delta: Amount to change (positive to increment, negative to decrement).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        normalizer_active_conversions.inc(delta)
    elif delta < 0:
        normalizer_active_conversions.dec(abs(delta))


def update_custom_factors(tenant_id: str, count: int) -> None:
    """Set the custom factors gauge for a tenant.

    Args:
        tenant_id: Tenant identifier.
        count: Number of custom factors.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    normalizer_custom_factors.labels(tenant_id=tenant_id).set(count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "normalizer_conversions_total",
    "normalizer_conversion_duration_seconds",
    "normalizer_entity_resolutions_total",
    "normalizer_resolution_duration_seconds",
    "normalizer_dimension_errors_total",
    "normalizer_gwp_conversions_total",
    "normalizer_batch_size",
    "normalizer_vocabulary_entries",
    "normalizer_cache_hits_total",
    "normalizer_cache_misses_total",
    "normalizer_active_conversions",
    "normalizer_custom_factors",
    # Helper functions
    "record_conversion",
    "record_entity_resolution",
    "record_dimension_error",
    "record_gwp_conversion",
    "record_batch",
    "update_vocabulary_entries",
    "record_cache_hit",
    "record_cache_miss",
    "update_active_conversions",
    "update_custom_factors",
]
