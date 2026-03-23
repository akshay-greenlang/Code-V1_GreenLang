# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-002: GreenLang Schema Compiler & Validator

12 Prometheus metrics for schema validation monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_schema_validations_total (Counter)
    2.  gl_schema_validation_duration_seconds (Histogram)
    3.  gl_schema_compilation_duration_seconds (Histogram)
    4.  gl_schema_errors_total (Counter)
    5.  gl_schema_warnings_total (Counter)
    6.  gl_schema_fixes_applied_total (Counter)
    7.  gl_schema_cache_hits_total (Counter)
    8.  gl_schema_cache_misses_total (Counter)
    9.  gl_schema_batch_size (Histogram)
    10. gl_schema_payload_bytes (Histogram)
    11. gl_schema_active_validations (Gauge)
    12. gl_schema_registered_schemas (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-002 Schema Compiler & Validator
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
        "prometheus_client not installed; schema metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Schema validation count
    schema_validations_total = Counter(
        "gl_schema_validations_total",
        "Total schema validations",
        labelnames=["schema_id", "result"],
    )

    # 2. Schema validation duration
    schema_validation_duration_seconds = Histogram(
        "gl_schema_validation_duration_seconds",
        "Schema validation duration in seconds",
        labelnames=["schema_id"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )

    # 3. Schema compilation duration
    schema_compilation_duration_seconds = Histogram(
        "gl_schema_compilation_duration_seconds",
        "Schema compilation duration in seconds",
        labelnames=["schema_id"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )

    # 4. Schema errors
    schema_errors_total = Counter(
        "gl_schema_errors_total",
        "Total schema validation errors",
        labelnames=["error_code"],
    )

    # 5. Schema warnings
    schema_warnings_total = Counter(
        "gl_schema_warnings_total",
        "Total schema validation warnings",
        labelnames=["warning_code"],
    )

    # 6. Fixes applied
    schema_fixes_applied_total = Counter(
        "gl_schema_fixes_applied_total",
        "Total fix suggestions applied",
        labelnames=["safety_level"],
    )

    # 7. Cache hits
    schema_cache_hits_total = Counter(
        "gl_schema_cache_hits_total",
        "Total schema cache hits",
        labelnames=["schema_id"],
    )

    # 8. Cache misses
    schema_cache_misses_total = Counter(
        "gl_schema_cache_misses_total",
        "Total schema cache misses",
        labelnames=["schema_id"],
    )

    # 9. Batch size
    schema_batch_size = Histogram(
        "gl_schema_batch_size",
        "Number of payloads in batch validation requests",
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
    )

    # 10. Payload bytes
    schema_payload_bytes = Histogram(
        "gl_schema_payload_bytes",
        "Payload size in bytes",
        buckets=(
            100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000,
        ),
    )

    # 11. Active validations
    schema_active_validations = Gauge(
        "gl_schema_active_validations",
        "Number of currently active schema validations",
    )

    # 12. Registered schemas
    schema_registered_schemas = Gauge(
        "gl_schema_registered_schemas",
        "Number of schemas registered in the schema registry",
    )

else:
    # No-op placeholders
    schema_validations_total = None  # type: ignore[assignment]
    schema_validation_duration_seconds = None  # type: ignore[assignment]
    schema_compilation_duration_seconds = None  # type: ignore[assignment]
    schema_errors_total = None  # type: ignore[assignment]
    schema_warnings_total = None  # type: ignore[assignment]
    schema_fixes_applied_total = None  # type: ignore[assignment]
    schema_cache_hits_total = None  # type: ignore[assignment]
    schema_cache_misses_total = None  # type: ignore[assignment]
    schema_batch_size = None  # type: ignore[assignment]
    schema_payload_bytes = None  # type: ignore[assignment]
    schema_active_validations = None  # type: ignore[assignment]
    schema_registered_schemas = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_validation(
    schema_id: str, result: str, duration_seconds: float,
) -> None:
    """Record a schema validation completion.

    Args:
        schema_id: Schema identifier.
        result: Validation result ("valid" or "invalid").
        duration_seconds: Validation duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_validations_total.labels(schema_id=schema_id, result=result).inc()
    schema_validation_duration_seconds.labels(schema_id=schema_id).observe(
        duration_seconds,
    )


def record_compilation(
    schema_id: str, duration_seconds: float,
) -> None:
    """Record a schema compilation completion.

    Args:
        schema_id: Schema identifier.
        duration_seconds: Compilation duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_compilation_duration_seconds.labels(schema_id=schema_id).observe(
        duration_seconds,
    )


def record_error(error_code: str) -> None:
    """Record a schema validation error.

    Args:
        error_code: GreenLang error code (e.g., "GLSCHEMA-E100").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_errors_total.labels(error_code=error_code).inc()


def record_warning(warning_code: str) -> None:
    """Record a schema validation warning.

    Args:
        warning_code: GreenLang warning code (e.g., "GLSCHEMA-W200").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_warnings_total.labels(warning_code=warning_code).inc()


def record_fix_applied(safety_level: str) -> None:
    """Record a fix suggestion applied.

    Args:
        safety_level: Safety level of the fix ("safe", "needs_review", "unsafe").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_fixes_applied_total.labels(safety_level=safety_level).inc()


def record_cache_hit(schema_id: str) -> None:
    """Record a schema cache hit.

    Args:
        schema_id: Schema identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_cache_hits_total.labels(schema_id=schema_id).inc()


def record_cache_miss(schema_id: str) -> None:
    """Record a schema cache miss.

    Args:
        schema_id: Schema identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_cache_misses_total.labels(schema_id=schema_id).inc()


def record_batch(batch_size: int) -> None:
    """Record a batch validation request size.

    Args:
        batch_size: Number of payloads in the batch.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_batch_size.observe(batch_size)


def record_payload_bytes(size_bytes: int) -> None:
    """Record payload size in bytes.

    Args:
        size_bytes: Payload size in bytes.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_payload_bytes.observe(size_bytes)


def update_active_validations(delta: int) -> None:
    """Update the active validations gauge.

    Args:
        delta: Amount to change (positive to increment, negative to decrement).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        schema_active_validations.inc(delta)
    elif delta < 0:
        schema_active_validations.dec(abs(delta))


def update_registered_schemas(count: int) -> None:
    """Set the registered schemas gauge to an absolute value.

    Args:
        count: Current number of registered schemas.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    schema_registered_schemas.set(count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "schema_validations_total",
    "schema_validation_duration_seconds",
    "schema_compilation_duration_seconds",
    "schema_errors_total",
    "schema_warnings_total",
    "schema_fixes_applied_total",
    "schema_cache_hits_total",
    "schema_cache_misses_total",
    "schema_batch_size",
    "schema_payload_bytes",
    "schema_active_validations",
    "schema_registered_schemas",
    # Helper functions
    "record_validation",
    "record_compilation",
    "record_error",
    "record_warning",
    "record_fix_applied",
    "record_cache_hit",
    "record_cache_miss",
    "record_batch",
    "record_payload_bytes",
    "update_active_validations",
    "update_registered_schemas",
]
