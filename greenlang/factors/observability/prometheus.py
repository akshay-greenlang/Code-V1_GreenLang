# -*- coding: utf-8 -*-
"""
Prometheus ASGI middleware and /metrics endpoint for the Factors API (F070).

Provides automatic request count and latency recording for all requests
to ``/api/v1/factors/*`` via an ASGI middleware, plus convenience helpers
for recording search results, match scores, QA failures, and edition
gauge updates.

The ``/metrics`` endpoint exposes all Prometheus metrics in the standard
text exposition format consumed by the Prometheus scraper.

Metrics emitted (aligned with ``factors-catalog.json`` dashboard):
  greenlang_factors_api_requests_total          - Counter
  greenlang_factors_api_latency_seconds         - Histogram
  greenlang_factors_search_results_count        - Histogram
  greenlang_factors_match_score_top1            - Histogram
  greenlang_factors_edition_factor_count        - Gauge
  greenlang_factors_qa_gate_failures_total      - Counter

Example::

    from fastapi import FastAPI
    from greenlang.factors.observability.prometheus import (
        FactorsMetricsMiddleware,
        metrics_endpoint,
    )

    app = FastAPI()
    app.add_middleware(FactorsMetricsMiddleware)
    app.add_route("/metrics", metrics_endpoint)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        generate_latest,
    )

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

from greenlang.factors.observability.prometheus_exporter import (
    FactorsMetrics,
    get_factors_metrics,
)

# ---------------------------------------------------------------------------
# Path prefix matched by the middleware
# ---------------------------------------------------------------------------

_FACTORS_API_PREFIX = "/api/v1/factors"


# ---------------------------------------------------------------------------
# ASGI Middleware
# ---------------------------------------------------------------------------


class FactorsMetricsMiddleware:
    """ASGI middleware that records request count and latency for Factors API.

    Only requests whose path starts with ``/api/v1/factors`` are instrumented.
    Other requests pass through unmodified for zero overhead.

    The middleware captures metrics even when the downstream handler raises
    an exception, ensuring error responses are always counted.

    Args:
        app: The ASGI application to wrap.

    Example::

        app = FastAPI()
        app.add_middleware(FactorsMetricsMiddleware)
    """

    def __init__(self, app: Any) -> None:
        """Initialize middleware wrapping the given ASGI application."""
        self.app = app
        self._metrics: FactorsMetrics = get_factors_metrics()
        logger.info("FactorsMetricsMiddleware initialized (prometheus=%s)", _PROM_AVAILABLE)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        """Process an ASGI request, recording metrics for Factors API paths.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if not path.startswith(_FACTORS_API_PREFIX):
            await self.app(scope, receive, send)
            return

        method: str = scope.get("method", "GET")
        route_template = _normalize_path(path)
        start = time.monotonic()
        status_code = 500  # default to 500 in case of unhandled error

        async def _send_wrapper(message: dict) -> None:
            """Intercept the response start message to capture status code."""
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        except Exception:
            status_code = 500
            raise
        finally:
            latency = time.monotonic() - start
            self._metrics.record_api_request(method, route_template, status_code, latency)
            if latency > 1.0:
                logger.warning(
                    "Slow Factors API request: method=%s path=%s latency=%.3fs status=%d",
                    method,
                    route_template,
                    latency,
                    status_code,
                )


def _normalize_path(path: str) -> str:
    """Normalize a request path to a route template to avoid cardinality explosion.

    Replaces UUID-like and numeric path segments with placeholders so that
    ``/api/v1/factors/abc-123`` becomes ``/api/v1/factors/{factor_id}``.

    Args:
        path: The raw request path.

    Returns:
        Normalized path string suitable for use as a Prometheus label value.
    """
    parts = path.rstrip("/").split("/")
    # /api/v1/factors -> keep as-is
    # /api/v1/factors/{id} -> replace id with {factor_id}
    # /api/v1/factors/{id}/audit-bundle -> replace id with {factor_id}
    # /api/v1/factors/search -> keep as-is
    # /api/v1/factors/search/v2 -> keep as-is
    # /api/v1/factors/search/facets -> keep as-is

    # Known static sub-paths that should NOT be treated as factor IDs
    _STATIC_SEGMENTS = {
        "search", "match", "export", "coverage", "health", "metrics",
    }

    if len(parts) <= 4:
        # /api/v1/factors or shorter
        return path.rstrip("/") or "/"

    # parts[4] is the segment after /api/v1/factors/
    segment = parts[4]
    if segment in _STATIC_SEGMENTS:
        return path.rstrip("/")

    # Assume it is a factor_id -- replace with template
    parts[4] = "{factor_id}"
    return "/".join(parts)


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------


async def metrics_endpoint(request: Any) -> Any:
    """ASGI/Starlette endpoint that returns Prometheus metrics in text format.

    Generates the standard Prometheus text exposition from the default
    collector registry. Intended to be mounted at ``/metrics``.

    Args:
        request: The incoming Starlette/FastAPI Request object.

    Returns:
        A Starlette Response with ``text/plain`` content type and the
        serialized Prometheus metrics payload.
    """
    # Ensure the singleton is initialized so metrics are registered
    get_factors_metrics()

    if not _PROM_AVAILABLE:
        from starlette.responses import PlainTextResponse

        return PlainTextResponse(
            "# prometheus_client not installed\n",
            status_code=503,
            media_type="text/plain",
        )

    from starlette.responses import Response as StarletteResponse

    body = generate_latest()
    return StarletteResponse(
        content=body,
        status_code=200,
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Convenience helpers (module-level functions)
# ---------------------------------------------------------------------------


def record_search_results(count: int) -> None:
    """Record the number of results returned by a search request.

    Args:
        count: Number of search results returned.
    """
    get_factors_metrics().record_search_results(count)


def record_match_score(score: float) -> None:
    """Record the top-1 match confidence score.

    Args:
        score: Confidence score between 0.0 and 1.0.
    """
    get_factors_metrics().record_match_score(score)


def record_qa_failure(gate_name: str = "default") -> None:
    """Record a QA gate failure.

    Args:
        gate_name: Name of the QA gate that failed.
    """
    get_factors_metrics().record_qa_failure(gate_name)


def update_edition_gauge(edition_id: str, count: int, status: str = "certified") -> None:
    """Update the edition factor count gauge.

    Args:
        edition_id: Edition identifier.
        count: Number of factors in this edition.
        status: Factor status category (e.g. ``certified``, ``preview``).
    """
    get_factors_metrics().set_edition_factor_count(edition_id, status, count)


__all__ = [
    "FactorsMetricsMiddleware",
    "metrics_endpoint",
    "record_search_results",
    "record_match_score",
    "record_qa_failure",
    "update_edition_gauge",
]
