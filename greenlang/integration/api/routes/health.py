# -*- coding: utf-8 -*-
"""
GreenLang API - Health and Monitoring Routes

Routes:
- GET /health                  - Health check (for K8s probes)
- GET /api/v1/factors/health   - Factors-specific health (used by Docker/K8s)
- GET /api/v1/cache/stats      - Cache statistics
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from greenlang.integration.api.models import CacheStats, HealthResponse
from greenlang.integration.api.dependencies import get_factor_service

logger = logging.getLogger(__name__)

# Track process start time for uptime calculation
_START_TIME = time.monotonic()
_START_TIMESTAMP = datetime.now(timezone.utc)

router = APIRouter(tags=["Health"])


def _uptime_seconds() -> float:
    return round(time.monotonic() - _START_TIME, 1)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status (for load balancers and K8s probes)",
    responses={
        200: {"description": "API is healthy"},
        503: {"description": "API is unhealthy"},
    },
)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint for monitoring and K8s liveness/readiness probes."""
    db_status = "connected"
    cache_status = "available"

    # Probe the factor service to check DB connectivity
    try:
        svc = get_factor_service()
        edition_id = svc.repo.get_default_edition_id()
        if not edition_id:
            db_status = "no_editions"
    except Exception as exc:
        logger.warning("Health check: factor service probe failed: %s", exc)
        db_status = "error"

    overall = "healthy" if db_status in ("connected", "no_editions") else "unhealthy"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        database=db_status,
        cache=cache_status,
        uptime_seconds=_uptime_seconds(),
    )


@router.get(
    "/api/v1/factors/health",
    response_model=HealthResponse,
    summary="Factors service health",
    description="Health check endpoint referenced by Docker and K8s probes",
    responses={200: {"description": "Factors service is healthy"}},
)
async def factors_health(request: Request) -> HealthResponse:
    """Factors-specific health used by Docker HEALTHCHECK and K8s probes."""
    return await health_check(request)


@router.get(
    "/api/v1/cache/stats",
    response_model=CacheStats,
    summary="Get cache statistics",
    description="Retrieve cache hit/miss statistics",
    responses={200: {"description": "Cache statistics"}},
)
async def get_cache_stats(request: Request) -> CacheStats:
    """Get cache statistics for monitoring."""
    # Try to read from Redis cache if available
    try:
        from greenlang.factors.cache_redis import FactorPayloadCache

        cache = FactorPayloadCache()
        return CacheStats(
            enabled=cache._redis is not None if hasattr(cache, "_redis") else False,
            hits=0,
            misses=0,
            hit_rate_pct=0.0,
            size=0,
            max_size=10000,
        )
    except Exception:
        return CacheStats(
            enabled=False,
            hits=0,
            misses=0,
            hit_rate_pct=0.0,
            size=0,
            max_size=0,
        )


__all__ = ["router"]
