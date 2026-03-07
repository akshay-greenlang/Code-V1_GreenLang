# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-003 Satellite Monitoring API

Aggregates all route modules under the ``/v1/eudr-sat`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (28 endpoints):
    imagery_routes:      4 endpoints (POST search, POST download,
                                      GET scene, GET availability)
    analysis_routes:     6 endpoints (POST ndvi, POST baseline, GET baseline,
                                      POST change-detect, POST fusion,
                                      GET history)
    monitoring_routes:   6 endpoints (POST schedule, GET schedule, PUT schedule,
                                      DELETE schedule, GET results, POST execute)
    alert_routes:        4 endpoints (GET list, GET detail, PUT acknowledge,
                                      GET summary)
    evidence_routes:     3 endpoints (POST package, GET package, GET download)
    batch_routes:        4 endpoints (POST submit, GET results, GET progress,
                                      DELETE cancel)
    + health check:      1 endpoint  (GET health)
    -------
    Total: 28 unique endpoints + health = 29 base

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-satellite:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from greenlang.agents.eudr.satellite_monitoring.api.alert_routes import (
    router as alert_router,
)
from greenlang.agents.eudr.satellite_monitoring.api.analysis_routes import (
    router as analysis_router,
)
from greenlang.agents.eudr.satellite_monitoring.api.batch_routes import (
    router as batch_router,
)
from greenlang.agents.eudr.satellite_monitoring.api.evidence_routes import (
    router as evidence_router,
)
from greenlang.agents.eudr.satellite_monitoring.api.imagery_routes import (
    router as imagery_router,
)
from greenlang.agents.eudr.satellite_monitoring.api.monitoring_routes import (
    router as monitoring_router,
)
from greenlang.agents.eudr.satellite_monitoring.api.schemas import (
    HealthResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-sat prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-sat",
    tags=["EUDR Satellite Monitoring"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(
    imagery_router,
    prefix="/imagery",
    tags=["Satellite Imagery"],
)
router.include_router(
    analysis_router,
    prefix="/analysis",
    tags=["Satellite Analysis"],
)
router.include_router(
    monitoring_router,
    prefix="/monitoring",
    tags=["Continuous Monitoring"],
)
router.include_router(
    alert_router,
    prefix="/alerts",
    tags=["Satellite Alerts"],
)
router.include_router(
    evidence_router,
    prefix="/evidence",
    tags=["Evidence Packages"],
)
router.include_router(
    batch_router,
    prefix="/batch",
    tags=["Batch Analysis"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check EUDR Satellite Monitoring API health.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring.

    Returns service identity, version, and current timestamp.
    No authentication required.
    """
    return HealthResponse(
        status="healthy",
        agent_id="GL-EUDR-SAT-003",
        agent_name="EUDR Satellite Monitoring Agent",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


def get_router() -> APIRouter:
    """Return the EUDR Satellite Monitoring API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.satellite_monitoring.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all satellite monitoring endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
