"""
GreenLang API - Health and Monitoring Routes

This module provides health check and monitoring endpoints.

Routes:
- GET /health - Health check endpoint
- GET /api/v1/cache/stats - Cache statistics

Author: GreenLang Team
Date: 2025-11-21
"""

from fastapi import APIRouter, Request

from greenlang.api.models import HealthResponse, CacheStats

import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status",
    responses={
        200: {"description": "API is healthy"},
        503: {"description": "API is unhealthy"}
    }
)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint for monitoring.
    """
    # Implementation will be moved from main.py
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=str(request.state.request_id) if hasattr(request.state, 'request_id') else ""
    )


@router.get(
    "/api/v1/cache/stats",
    response_model=CacheStats,
    summary="Get cache statistics",
    description="Retrieve cache hit/miss statistics",
    responses={
        200: {"description": "Cache statistics"},
    }
)
async def get_cache_stats(request: Request) -> CacheStats:
    """
    Get cache statistics for monitoring.
    """
    # Implementation will be moved from main.py
    return CacheStats(
        hits=0,
        misses=0,
        hit_rate=0.0,
        size=0
    )


__all__ = ['router']
