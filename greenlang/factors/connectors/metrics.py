# -*- coding: utf-8 -*-
"""
Prometheus metrics for factor connectors (F060).

Exposes connector request counts, latency histograms, error rates,
and quota usage as Prometheus metrics compatible with OBS-001.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Try to import prometheus_client; degrade gracefully if not installed.
try:
    from prometheus_client import Counter, Gauge, Histogram

    CONNECTOR_REQUESTS = Counter(
        "greenlang_factors_connector_requests_total",
        "Total connector API requests",
        ["connector_id", "operation", "status"],
    )
    CONNECTOR_LATENCY = Histogram(
        "greenlang_factors_connector_latency_seconds",
        "Connector request latency in seconds",
        ["connector_id", "operation"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    CONNECTOR_ERRORS = Counter(
        "greenlang_factors_connector_errors_total",
        "Total connector errors by type",
        ["connector_id", "error_type"],
    )
    CONNECTOR_FACTORS_FETCHED = Counter(
        "greenlang_factors_connector_factors_fetched_total",
        "Total factors fetched via connectors",
        ["connector_id"],
    )
    CONNECTOR_QUOTA_REMAINING = Gauge(
        "greenlang_factors_connector_quota_remaining",
        "Remaining API quota for connector",
        ["connector_id"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


@dataclass
class InMemoryMetrics:
    """Fallback metrics when prometheus_client is not installed."""

    requests: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, int] = field(default_factory=dict)
    latencies: Dict[str, list] = field(default_factory=dict)
    factors_fetched: Dict[str, int] = field(default_factory=dict)
    quota_remaining: Dict[str, int] = field(default_factory=dict)


_fallback = InMemoryMetrics()


def record_request(connector_id: str, operation: str, success: bool = True) -> None:
    """Record a connector request."""
    status = "ok" if success else "error"
    if _PROMETHEUS_AVAILABLE:
        CONNECTOR_REQUESTS.labels(connector_id=connector_id, operation=operation, status=status).inc()
    else:
        key = f"{connector_id}:{operation}:{status}"
        _fallback.requests[key] = _fallback.requests.get(key, 0) + 1


def record_latency(connector_id: str, operation: str, seconds: float) -> None:
    """Record request latency."""
    if _PROMETHEUS_AVAILABLE:
        CONNECTOR_LATENCY.labels(connector_id=connector_id, operation=operation).observe(seconds)
    else:
        key = f"{connector_id}:{operation}"
        _fallback.latencies.setdefault(key, []).append(seconds)


def record_error(connector_id: str, error_type: str) -> None:
    """Record a connector error."""
    if _PROMETHEUS_AVAILABLE:
        CONNECTOR_ERRORS.labels(connector_id=connector_id, error_type=error_type).inc()
    else:
        key = f"{connector_id}:{error_type}"
        _fallback.errors[key] = _fallback.errors.get(key, 0) + 1


def record_factors_fetched(connector_id: str, count: int) -> None:
    """Record number of factors fetched."""
    if _PROMETHEUS_AVAILABLE:
        CONNECTOR_FACTORS_FETCHED.labels(connector_id=connector_id).inc(count)
    else:
        _fallback.factors_fetched[connector_id] = _fallback.factors_fetched.get(connector_id, 0) + count


def set_quota_remaining(connector_id: str, remaining: int) -> None:
    """Update the remaining quota gauge."""
    if _PROMETHEUS_AVAILABLE:
        CONNECTOR_QUOTA_REMAINING.labels(connector_id=connector_id).set(remaining)
    else:
        _fallback.quota_remaining[connector_id] = remaining


@contextmanager
def track_connector_call(
    connector_id: str,
    operation: str,
) -> Generator[None, None, None]:
    """Context manager that tracks request count, latency, and errors."""
    start = time.monotonic()
    try:
        yield
        elapsed = time.monotonic() - start
        record_request(connector_id, operation, success=True)
        record_latency(connector_id, operation, elapsed)
    except Exception as exc:
        elapsed = time.monotonic() - start
        record_request(connector_id, operation, success=False)
        record_latency(connector_id, operation, elapsed)
        record_error(connector_id, type(exc).__name__)
        raise


def get_fallback_metrics() -> InMemoryMetrics:
    """Return in-memory metrics (for testing / non-Prometheus environments)."""
    return _fallback
