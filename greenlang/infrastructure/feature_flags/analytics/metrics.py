# -*- coding: utf-8 -*-
"""
Feature Flags Prometheus Metrics - INFRA-008

Defines all Prometheus metrics for the GreenLang feature flag system
and provides helper functions for recording evaluation events, cache
events, and flag state changes.

Metrics are organized by subsystem:
    - Evaluation: Total evaluations, evaluation duration
    - Cache: L1/L2 hit and miss counters
    - State: Flag lifecycle state gauges, kill switch status
    - Stale: Stale flag detection gauges
    - Storage: Backend error counters

All metrics use the ``ff_`` prefix (feature flags) to avoid collisions
with other GreenLang Prometheus metrics.

Example:
    >>> from greenlang.infrastructure.feature_flags.analytics.metrics import (
    ...     record_evaluation, record_cache_event, update_flag_state,
    ... )
    >>> record_evaluation("my-flag", enabled=True, environment="prod", tenant="acme")
    >>> record_cache_event("l1", hit=True)
    >>> update_flag_state("my-flag", "active")
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore[misc, assignment]
    Gauge = None  # type: ignore[misc, assignment]
    Histogram = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:

    ff_evaluation_total = Counter(
        "ff_evaluation_total",
        "Total number of feature flag evaluations",
        labelnames=["flag_key", "result", "environment", "tenant"],
    )
    """Counter: Total flag evaluations. Labels: flag_key, result (true/false), environment, tenant."""

    ff_evaluation_duration_seconds = Histogram(
        "ff_evaluation_duration_seconds",
        "Duration of feature flag evaluations in seconds",
        labelnames=["flag_key", "cache_layer"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    )
    """Histogram: Evaluation duration. Labels: flag_key, cache_layer (l1/l2/db/default)."""

    ff_flag_state = Gauge(
        "ff_flag_state",
        "Current state of a feature flag (1=active, 0=inactive)",
        labelnames=["flag_key", "status"],
    )
    """Gauge: Current flag state. Labels: flag_key, status."""

    ff_cache_hit_total = Counter(
        "ff_cache_hit_total",
        "Total cache hits for feature flag lookups",
        labelnames=["layer"],
    )
    """Counter: Cache hits. Labels: layer (l1/l2)."""

    ff_cache_miss_total = Counter(
        "ff_cache_miss_total",
        "Total cache misses for feature flag lookups",
        labelnames=["layer"],
    )
    """Counter: Cache misses. Labels: layer (l1/l2)."""

    ff_kill_switch_active = Gauge(
        "ff_kill_switch_active",
        "Whether the kill switch is active for a flag (1=killed, 0=normal)",
        labelnames=["flag_key"],
    )
    """Gauge: Kill switch state. Labels: flag_key."""

    ff_stale_flags_total = Gauge(
        "ff_stale_flags_total",
        "Total number of stale feature flags",
        labelnames=["environment"],
    )
    """Gauge: Stale flag count. Labels: environment."""

    ff_storage_errors_total = Counter(
        "ff_storage_errors_total",
        "Total storage backend errors",
        labelnames=["backend"],
    )
    """Counter: Storage errors. Labels: backend (memory/redis/postgres)."""

else:
    # Provide no-op stubs when prometheus_client is not installed
    logger.info(
        "prometheus_client not available; feature flag metrics will be no-ops"
    )

    class _NoOpMetric:
        """No-op metric stub when prometheus_client is not installed."""

        def labels(self, *args, **kwargs):  # noqa: ANN
            """Return self for chaining."""
            return self

        def inc(self, amount=1):  # noqa: ANN
            """No-op increment."""

        def dec(self, amount=1):  # noqa: ANN
            """No-op decrement."""

        def set(self, value):  # noqa: ANN
            """No-op set."""

        def observe(self, amount):  # noqa: ANN
            """No-op observe."""

    ff_evaluation_total = _NoOpMetric()  # type: ignore[assignment]
    ff_evaluation_duration_seconds = _NoOpMetric()  # type: ignore[assignment]
    ff_flag_state = _NoOpMetric()  # type: ignore[assignment]
    ff_cache_hit_total = _NoOpMetric()  # type: ignore[assignment]
    ff_cache_miss_total = _NoOpMetric()  # type: ignore[assignment]
    ff_kill_switch_active = _NoOpMetric()  # type: ignore[assignment]
    ff_stale_flags_total = _NoOpMetric()  # type: ignore[assignment]
    ff_storage_errors_total = _NoOpMetric()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_evaluation(
    flag_key: str,
    enabled: bool,
    environment: str = "dev",
    tenant: str = "",
    cache_layer: str = "default",
    duration_seconds: float = 0.0,
) -> None:
    """Record a feature flag evaluation event.

    Updates the evaluation counter and duration histogram.

    Args:
        flag_key: The flag that was evaluated.
        enabled: Whether the flag resolved to enabled.
        environment: Deployment environment.
        tenant: Tenant identifier (empty string if not applicable).
        cache_layer: Which cache layer served the result.
        duration_seconds: Evaluation duration in seconds.
    """
    result_str = "true" if enabled else "false"
    tenant_label = tenant or "unknown"

    try:
        ff_evaluation_total.labels(
            flag_key=flag_key,
            result=result_str,
            environment=environment,
            tenant=tenant_label,
        ).inc()

        if duration_seconds > 0:
            ff_evaluation_duration_seconds.labels(
                flag_key=flag_key,
                cache_layer=cache_layer,
            ).observe(duration_seconds)
    except Exception as exc:
        logger.debug("Failed to record evaluation metric: %s", exc)


def record_cache_event(
    layer: str,
    hit: bool,
) -> None:
    """Record a cache hit or miss event.

    Args:
        layer: Cache layer name (l1, l2).
        hit: True if cache hit, False if miss.
    """
    try:
        if hit:
            ff_cache_hit_total.labels(layer=layer).inc()
        else:
            ff_cache_miss_total.labels(layer=layer).inc()
    except Exception as exc:
        logger.debug("Failed to record cache metric: %s", exc)


def update_flag_state(
    flag_key: str,
    flag_status: str,
) -> None:
    """Update the flag state gauge.

    Sets the gauge to 1 for the current status and 0 for all others.

    Args:
        flag_key: The flag key.
        flag_status: Current lifecycle status string.
    """
    all_statuses = ["draft", "active", "rolled_out", "permanent", "archived", "killed"]
    try:
        for s in all_statuses:
            value = 1.0 if s == flag_status else 0.0
            ff_flag_state.labels(flag_key=flag_key, status=s).set(value)
    except Exception as exc:
        logger.debug("Failed to update flag state metric: %s", exc)


def record_kill_switch(
    flag_key: str,
    active: bool,
) -> None:
    """Record kill switch activation/deactivation.

    Args:
        flag_key: The flag key.
        active: True if kill switch is active.
    """
    try:
        ff_kill_switch_active.labels(flag_key=flag_key).set(1.0 if active else 0.0)
    except Exception as exc:
        logger.debug("Failed to record kill switch metric: %s", exc)


def record_storage_error(
    backend: str,
) -> None:
    """Record a storage backend error.

    Args:
        backend: Backend name (memory, redis, postgres).
    """
    try:
        ff_storage_errors_total.labels(backend=backend).inc()
    except Exception as exc:
        logger.debug("Failed to record storage error metric: %s", exc)


def update_stale_count(
    environment: str,
    count: int,
) -> None:
    """Update the stale flags gauge.

    Args:
        environment: Deployment environment.
        count: Number of stale flags.
    """
    try:
        ff_stale_flags_total.labels(environment=environment).set(float(count))
    except Exception as exc:
        logger.debug("Failed to update stale count metric: %s", exc)


__all__ = [
    "ff_cache_hit_total",
    "ff_cache_miss_total",
    "ff_evaluation_duration_seconds",
    "ff_evaluation_total",
    "ff_flag_state",
    "ff_kill_switch_active",
    "ff_stale_flags_total",
    "ff_storage_errors_total",
    "record_cache_event",
    "record_evaluation",
    "record_kill_switch",
    "record_storage_error",
    "update_flag_state",
    "update_stale_count",
]
