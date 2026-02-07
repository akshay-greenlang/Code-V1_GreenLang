# -*- coding: utf-8 -*-
"""
Factory Middleware - Request context injection and cost tracking.

Provides two ASGI middleware classes:
    - FactoryContextMiddleware: Injects tenant_id, correlation_id, and
      factory context into every request's state.
    - CostTrackingMiddleware: Measures request duration and tags each
      response with cost/timing headers.

Usage with FastAPI:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.agent_factory.api.middleware import (
    ...     FactoryContextMiddleware, CostTrackingMiddleware,
    ... )
    >>> app = FastAPI()
    >>> app.add_middleware(FactoryContextMiddleware)
    >>> app.add_middleware(CostTrackingMiddleware)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACTORY_VERSION = "1.0.0"

# Header names
HEADER_CORRELATION_ID = "x-correlation-id"
HEADER_TENANT_ID = "x-tenant-id"
HEADER_FACTORY_VERSION = "x-agent-factory-version"
HEADER_REQUEST_COST = "x-request-cost-usd"
HEADER_REQUEST_DURATION = "x-request-duration-ms"


# ---------------------------------------------------------------------------
# Tenant extraction helpers
# ---------------------------------------------------------------------------

def _extract_tenant_id(request: Request) -> str:
    """Extract tenant ID from the request.

    Priority:
        1. x-tenant-id header
        2. Authorization JWT claim (tenant_id)
        3. Query parameter tenant_id
        4. "default" fallback

    Args:
        request: Incoming HTTP request.

    Returns:
        Extracted tenant identifier.
    """
    # 1. Header
    tenant = request.headers.get(HEADER_TENANT_ID)
    if tenant:
        return tenant

    # 2. JWT claim (lightweight extraction without full JWT validation)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            import base64
            import json

            token = auth_header[7:]
            parts = token.split(".")
            if len(parts) >= 2:
                # Decode payload (add padding)
                payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
                payload = json.loads(base64.urlsafe_b64decode(payload_b64))
                tenant = payload.get("tenant_id") or payload.get("tid")
                if tenant:
                    return str(tenant)
        except Exception:
            pass  # Fall through to next method

    # 3. Query parameter
    tenant = request.query_params.get("tenant_id")
    if tenant:
        return tenant

    # 4. Default
    return "default"


def _extract_correlation_id(request: Request) -> str:
    """Extract or generate a correlation ID.

    Args:
        request: Incoming HTTP request.

    Returns:
        Existing or newly-generated correlation ID.
    """
    existing = request.headers.get(HEADER_CORRELATION_ID)
    if existing:
        return existing
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# FactoryContextMiddleware
# ---------------------------------------------------------------------------

class FactoryContextMiddleware(BaseHTTPMiddleware):
    """Inject factory context into every request.

    Adds the following to request.state:
        - tenant_id: Extracted tenant identifier.
        - correlation_id: Request correlation ID.
        - factory_version: Current factory version string.

    Also adds response headers:
        - x-agent-factory-version
        - x-correlation-id

    Example:
        >>> app.add_middleware(FactoryContextMiddleware)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process the request and inject context."""
        tenant_id = _extract_tenant_id(request)
        correlation_id = _extract_correlation_id(request)

        # Attach to request state for downstream access via request.state
        request.state.tenant_id = tenant_id
        request.state.correlation_id = correlation_id
        request.state.factory_version = FACTORY_VERSION

        logger.debug(
            "Request context: tenant=%s correlation=%s path=%s",
            tenant_id,
            correlation_id,
            request.url.path,
        )

        response = await call_next(request)

        # Add response headers
        response.headers[HEADER_FACTORY_VERSION] = FACTORY_VERSION
        response.headers[HEADER_CORRELATION_ID] = correlation_id

        return response


# ---------------------------------------------------------------------------
# CostTrackingMiddleware
# ---------------------------------------------------------------------------

# Cost model: base cost per request + per-ms cost.
# These values are configurable via environment or config.
_BASE_COST_USD = 0.000_001  # $0.001 per 1000 requests
_PER_MS_COST_USD = 0.000_000_01  # $0.01 per 1M ms


class CostTrackingMiddleware(BaseHTTPMiddleware):
    """Track API request cost and duration.

    Adds response headers:
        - x-request-duration-ms: Wall-clock duration in milliseconds.
        - x-request-cost-usd: Estimated cost in USD.

    Cost model uses a simple base + per-millisecond formula suitable for
    internal charge-back.

    Example:
        >>> app.add_middleware(CostTrackingMiddleware)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Measure request duration and estimate cost."""
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        cost_usd = _BASE_COST_USD + (elapsed_ms * _PER_MS_COST_USD)

        response.headers[HEADER_REQUEST_DURATION] = f"{elapsed_ms:.2f}"
        response.headers[HEADER_REQUEST_COST] = f"{cost_usd:.10f}"

        logger.debug(
            "Request completed: path=%s duration=%.2fms cost=$%.10f",
            request.url.path,
            elapsed_ms,
            cost_usd,
        )

        return response


__all__ = [
    "CostTrackingMiddleware",
    "FactoryContextMiddleware",
    "FACTORY_VERSION",
]
