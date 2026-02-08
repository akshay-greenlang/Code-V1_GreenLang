# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-008: Reproducibility Agent

12 Prometheus metrics for reproducibility service monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_reproducibility_verifications_total (Counter, labels: status)
    2.  gl_reproducibility_verification_duration_seconds (Histogram)
    3.  gl_reproducibility_hash_computations_total (Counter, labels: type)
    4.  gl_reproducibility_hash_mismatches_total (Counter)
    5.  gl_reproducibility_drift_detections_total (Counter, labels: severity)
    6.  gl_reproducibility_drift_percentage (Gauge)
    7.  gl_reproducibility_replays_total (Counter, labels: result)
    8.  gl_reproducibility_replay_duration_seconds (Histogram)
    9.  gl_reproducibility_non_determinism_sources_total (Counter, labels: source)
    10. gl_reproducibility_environment_mismatches_total (Counter)
    11. gl_reproducibility_cache_hits_total (Counter)
    12. gl_reproducibility_cache_misses_total (Counter)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
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
        "prometheus_client not installed; reproducibility metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Verifications count by status
    reproducibility_verifications_total = Counter(
        "gl_reproducibility_verifications_total",
        "Total reproducibility verification runs",
        labelnames=["status"],
    )

    # 2. Verification duration
    reproducibility_verification_duration_seconds = Histogram(
        "gl_reproducibility_verification_duration_seconds",
        "Reproducibility verification duration in seconds",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )

    # 3. Hash computations by artifact type
    reproducibility_hash_computations_total = Counter(
        "gl_reproducibility_hash_computations_total",
        "Total hash computations performed",
        labelnames=["type"],
    )

    # 4. Hash mismatches
    reproducibility_hash_mismatches_total = Counter(
        "gl_reproducibility_hash_mismatches_total",
        "Total hash mismatches detected",
    )

    # 5. Drift detections by severity
    reproducibility_drift_detections_total = Counter(
        "gl_reproducibility_drift_detections_total",
        "Total drift detections performed",
        labelnames=["severity"],
    )

    # 6. Current drift percentage gauge
    reproducibility_drift_percentage = Gauge(
        "gl_reproducibility_drift_percentage",
        "Most recent drift percentage detected",
    )

    # 7. Replays by result
    reproducibility_replays_total = Counter(
        "gl_reproducibility_replays_total",
        "Total replay executions",
        labelnames=["result"],
    )

    # 8. Replay duration
    reproducibility_replay_duration_seconds = Histogram(
        "gl_reproducibility_replay_duration_seconds",
        "Replay execution duration in seconds",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # 9. Non-determinism sources detected
    reproducibility_non_determinism_sources_total = Counter(
        "gl_reproducibility_non_determinism_sources_total",
        "Total non-determinism source detections",
        labelnames=["source"],
    )

    # 10. Environment mismatches
    reproducibility_environment_mismatches_total = Counter(
        "gl_reproducibility_environment_mismatches_total",
        "Total environment mismatches detected",
    )

    # 11. Cache hits
    reproducibility_cache_hits_total = Counter(
        "gl_reproducibility_cache_hits_total",
        "Total reproducibility hash cache hits",
    )

    # 12. Cache misses
    reproducibility_cache_misses_total = Counter(
        "gl_reproducibility_cache_misses_total",
        "Total reproducibility hash cache misses",
    )

else:
    # No-op placeholders
    reproducibility_verifications_total = None  # type: ignore[assignment]
    reproducibility_verification_duration_seconds = None  # type: ignore[assignment]
    reproducibility_hash_computations_total = None  # type: ignore[assignment]
    reproducibility_hash_mismatches_total = None  # type: ignore[assignment]
    reproducibility_drift_detections_total = None  # type: ignore[assignment]
    reproducibility_drift_percentage = None  # type: ignore[assignment]
    reproducibility_replays_total = None  # type: ignore[assignment]
    reproducibility_replay_duration_seconds = None  # type: ignore[assignment]
    reproducibility_non_determinism_sources_total = None  # type: ignore[assignment]
    reproducibility_environment_mismatches_total = None  # type: ignore[assignment]
    reproducibility_cache_hits_total = None  # type: ignore[assignment]
    reproducibility_cache_misses_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_verification(status: str, duration_seconds: float) -> None:
    """Record a verification run with its status and duration.

    Args:
        status: Verification status (pass, fail, warning, skipped).
        duration_seconds: Verification duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_verifications_total.labels(status=status).inc()
    reproducibility_verification_duration_seconds.observe(duration_seconds)


def record_hash_computation(artifact_type: str) -> None:
    """Record a hash computation for a given artifact type.

    Args:
        artifact_type: Type of artifact hashed (input, output, config, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_hash_computations_total.labels(type=artifact_type).inc()


def record_hash_mismatch() -> None:
    """Record a hash mismatch detection."""
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_hash_mismatches_total.inc()


def record_drift(severity: str, drift_pct: float) -> None:
    """Record a drift detection event.

    Args:
        severity: Drift severity (none, minor, moderate, critical).
        drift_pct: Drift percentage detected.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_drift_detections_total.labels(severity=severity).inc()
    reproducibility_drift_percentage.set(drift_pct)


def record_replay(result: str, duration_seconds: float) -> None:
    """Record a replay execution.

    Args:
        result: Replay result (pass, fail).
        duration_seconds: Replay duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_replays_total.labels(result=result).inc()
    reproducibility_replay_duration_seconds.observe(duration_seconds)


def record_non_determinism(source: str) -> None:
    """Record a non-determinism source detection.

    Args:
        source: Name of the non-determinism source.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_non_determinism_sources_total.labels(source=source).inc()


def record_environment_mismatch() -> None:
    """Record an environment mismatch detection."""
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_environment_mismatches_total.inc()


def record_cache_hit() -> None:
    """Record a reproducibility cache hit."""
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_cache_hits_total.inc()


def record_cache_miss() -> None:
    """Record a reproducibility cache miss."""
    if not PROMETHEUS_AVAILABLE:
        return
    reproducibility_cache_misses_total.inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "reproducibility_verifications_total",
    "reproducibility_verification_duration_seconds",
    "reproducibility_hash_computations_total",
    "reproducibility_hash_mismatches_total",
    "reproducibility_drift_detections_total",
    "reproducibility_drift_percentage",
    "reproducibility_replays_total",
    "reproducibility_replay_duration_seconds",
    "reproducibility_non_determinism_sources_total",
    "reproducibility_environment_mismatches_total",
    "reproducibility_cache_hits_total",
    "reproducibility_cache_misses_total",
    # Helper functions
    "record_verification",
    "record_hash_computation",
    "record_hash_mismatch",
    "record_drift",
    "record_replay",
    "record_non_determinism",
    "record_environment_mismatch",
    "record_cache_hit",
    "record_cache_miss",
]
