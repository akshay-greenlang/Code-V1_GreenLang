# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-004 Forest Cover Analysis API

Aggregates all route modules under the ``/v1/eudr-fca`` prefix and
provides ``get_eudr_fca_router()`` for integration with the GreenLang
platform.

Route Module Summary (30 endpoints):
    density_routes:         5 endpoints (POST analyze, POST batch,
                                         GET {id}, GET {id}/history,
                                         POST compare)
    classification_routes:  4 endpoints (POST /, POST batch,
                                         GET {id}, GET types)
    historical_routes:      5 endpoints (POST reconstruct, POST batch,
                                         GET {id}, POST compare,
                                         GET {id}/sources)
    verification_routes:    5 endpoints (POST /, POST batch,
                                         GET {id}, GET {id}/evidence,
                                         POST complete)
    analysis_routes:        5 endpoints (POST height, POST fragmentation,
                                         POST biomass, GET {id}/profile,
                                         POST compare)
    report_routes:          4 endpoints (POST generate, GET {id},
                                         GET {id}/download, POST batch)
    + health check:         1 endpoint  (GET health)
    + version:              1 endpoint  (GET version)
    -------
    Total: 30 unique endpoints + health + version = 32 base

Auth & RBAC:
    All endpoints (except health and version) require JWT auth via
    SEC-001 and check eudr-fca:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from greenlang.agents.eudr.forest_cover_analysis.api.analysis_routes import (
    router as analysis_router,
)
from greenlang.agents.eudr.forest_cover_analysis.api.classification_routes import (
    router as classification_router,
)
from greenlang.agents.eudr.forest_cover_analysis.api.density_routes import (
    router as density_router,
)
from greenlang.agents.eudr.forest_cover_analysis.api.historical_routes import (
    router as historical_router,
)
from greenlang.agents.eudr.forest_cover_analysis.api.report_routes import (
    router as report_router,
)
from greenlang.agents.eudr.forest_cover_analysis.api.schemas import (
    HealthResponse,
    VersionResponse,
)
from greenlang.agents.eudr.forest_cover_analysis.api.verification_routes import (
    router as verification_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-fca prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-fca",
    tags=["EUDR Forest Cover Analysis"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(
    density_router,
    prefix="/density",
    tags=["Canopy Density"],
)
router.include_router(
    classification_router,
    prefix="/classify",
    tags=["Forest Classification"],
)
router.include_router(
    historical_router,
    prefix="/historical",
    tags=["Historical Reconstruction"],
)
router.include_router(
    verification_router,
    prefix="/verify",
    tags=["Deforestation-Free Verification"],
)
router.include_router(
    analysis_router,
    prefix="/analysis",
    tags=["Forest Analysis"],
)
router.include_router(
    report_router,
    prefix="/reports",
    tags=["Compliance Reports"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check EUDR Forest Cover Analysis API health.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring.

    Returns service identity, version, and current timestamp.
    No authentication required.
    """
    return HealthResponse(
        status="healthy",
        agent_id="GL-EUDR-FCA-004",
        agent_name="EUDR Forest Cover Analysis Agent",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="API version information",
    description="Get API version, available engines, and configuration.",
    tags=["System"],
)
async def get_version() -> VersionResponse:
    """Return API version information and available capabilities.

    Returns agent identity, available analysis engines, supported
    commodities, FAO thresholds, and EUDR cutoff date.
    No authentication required.
    """
    return VersionResponse(
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_eudr_fca_router() -> APIRouter:
    """Return the EUDR Forest Cover Analysis API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.forest_cover_analysis.api import (
        ...     get_eudr_fca_router,
        ... )
        >>> app.include_router(get_eudr_fca_router(), prefix="/api")

    Returns:
        Configured APIRouter with all forest cover analysis endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_eudr_fca_router",
]
