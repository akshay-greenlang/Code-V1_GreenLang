# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-002 Geolocation Verification API

Aggregates all route modules under the ``/v1/eudr-geo`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (20+ endpoints):
    coordinate_routes:     2 endpoints (POST single, POST batch)
    polygon_routes:        2 endpoints (POST verify, POST repair)
    verification_routes:   7 endpoints (POST/GET protected-areas, POST/GET deforestation,
                                        POST/GET plot, GET history)
    batch_routes:          4 endpoints (POST submit, GET status, GET progress, DELETE cancel)
    scoring_routes:        4 endpoints (GET score, GET history, GET summary, PUT weights)
    compliance_routes:     3 endpoints (POST report, GET report, GET summary)
    + health check:        1 endpoint (GET health)
    -------
    Total: 23 unique endpoints + health = 24 base

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-geolocation:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from greenlang.agents.eudr.geolocation_verification.api.batch_routes import (
    router as batch_router,
)
from greenlang.agents.eudr.geolocation_verification.api.compliance_routes import (
    router as compliance_router,
)
from greenlang.agents.eudr.geolocation_verification.api.coordinate_routes import (
    router as coordinate_router,
)
from greenlang.agents.eudr.geolocation_verification.api.polygon_routes import (
    router as polygon_router,
)
from greenlang.agents.eudr.geolocation_verification.api.scoring_routes import (
    router as scoring_router,
)
from greenlang.agents.eudr.geolocation_verification.api.verification_routes import (
    router as verification_router,
)
from greenlang.agents.eudr.geolocation_verification.api.schemas import (
    HealthResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-geo prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-geo",
    tags=["EUDR Geolocation Verification"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
# Coordinate and polygon verification routes share the /verify prefix
router.include_router(
    coordinate_router,
    prefix="/verify",
    tags=["Coordinate Validation"],
)
router.include_router(
    polygon_router,
    prefix="/verify",
    tags=["Polygon Verification"],
)
router.include_router(
    verification_router,
    prefix="/verify",
    tags=["Verification"],
)
router.include_router(
    batch_router,
    prefix="/verify",
    tags=["Batch Verification"],
)
router.include_router(
    scoring_router,
    tags=["Accuracy Scoring"],
)
router.include_router(
    compliance_router,
    tags=["Compliance Reporting"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check EUDR Geolocation Verification API health.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring.

    Returns service identity, version, and current timestamp.
    No authentication required.
    """
    return HealthResponse(
        status="healthy",
        agent_id="GL-EUDR-GEO-002",
        agent_name="EUDR Geolocation Verification Agent",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


def get_router() -> APIRouter:
    """Return the EUDR Geolocation Verification API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.geolocation_verification.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all geolocation verification endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
