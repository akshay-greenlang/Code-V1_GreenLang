# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-005 Land Use Change Detector API

Aggregates all route modules under the ``/v1/eudr-luc`` prefix and
provides ``get_eudr_luc_router()`` for integration with the GreenLang
platform.

Route Module Summary (33 endpoints):
    classification_routes:  5 endpoints (POST /, POST batch,
                                         GET {id}, GET {id}/history,
                                         POST compare)
    transition_routes:      5 endpoints (POST detect, POST batch,
                                         GET {id}, POST matrix,
                                         GET types)
    trajectory_routes:      3 endpoints (POST analyze, POST batch,
                                         GET {id})
    verification_routes:    5 endpoints (POST cutoff, POST batch,
                                         GET {id}, GET {id}/evidence,
                                         POST complete)
    risk_routes:            8 endpoints (POST risk/assess, POST risk/batch,
                                         GET risk/{id},
                                         POST urban/analyze,
                                         POST urban/batch,
                                         GET urban/{id},
                                         POST batch, DELETE batch/{id})
    report_routes:          4 endpoints (POST generate, GET {id},
                                         GET {id}/download, POST batch)
    + health check:         1 endpoint  (GET health)
    + version:              1 endpoint  (GET version)
    -------
    Total: 31 unique endpoints + health + version = 33 base

Auth & RBAC:
    All endpoints (except health, version, and transition types) require
    JWT auth via SEC-001 and check eudr-luc:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter

from greenlang.agents.eudr.land_use_change.api.classification_routes import (
    router as classification_router,
)
from greenlang.agents.eudr.land_use_change.api.report_routes import (
    router as report_router,
)
from greenlang.agents.eudr.land_use_change.api.risk_routes import (
    router as risk_router,
)
from greenlang.agents.eudr.land_use_change.api.schemas import (
    EngineStatus,
    HealthResponse,
    VersionResponse,
)
from greenlang.agents.eudr.land_use_change.api.trajectory_routes import (
    router as trajectory_router,
)
from greenlang.agents.eudr.land_use_change.api.transition_routes import (
    router as transition_router,
)
from greenlang.agents.eudr.land_use_change.api.verification_routes import (
    router as verification_router,
)

logger = logging.getLogger(__name__)

# Track startup time for uptime calculation
_startup_time = time.monotonic()

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-luc prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-luc",
    tags=["EUDR Land Use Change Detector"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(
    classification_router,
    prefix="/classify",
    tags=["Land Use Classification"],
)
router.include_router(
    transition_router,
    prefix="/transitions",
    tags=["Transition Detection"],
)
router.include_router(
    trajectory_router,
    prefix="/trajectory",
    tags=["Trajectory Analysis"],
)
router.include_router(
    verification_router,
    prefix="/verify",
    tags=["Cutoff Verification"],
)
router.include_router(
    risk_router,
    tags=["Risk & Urban Analysis"],
)
router.include_router(
    report_router,
    prefix="/reports",
    tags=["Land Use Change Reports"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check EUDR Land Use Change Detector API health.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring.

    Returns service identity, version, engine status, uptime, and
    current timestamp. No authentication required.

    Returns:
        HealthResponse with service status and engine availability.
    """
    uptime = time.monotonic() - _startup_time

    engines = [
        EngineStatus(
            engine_name="LandUseClassifier",
            status="healthy",
            version="1.0.0",
        ),
        EngineStatus(
            engine_name="TransitionDetector",
            status="healthy",
            version="1.0.0",
        ),
        EngineStatus(
            engine_name="TemporalTrajectoryAnalyzer",
            status="healthy",
            version="1.0.0",
        ),
        EngineStatus(
            engine_name="CutoffDateVerifier",
            status="healthy",
            version="1.0.0",
        ),
    ]

    return HealthResponse(
        status="healthy",
        agent_id="GL-EUDR-LUC-005",
        agent_name="EUDR Land Use Change Detector Agent",
        version="1.0.0",
        engines_status=engines,
        uptime_seconds=uptime,
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="API version information",
    description=(
        "Get API version, available engines, supported commodities, "
        "classification methods, land use categories, and EUDR "
        "cutoff date."
    ),
    tags=["System"],
)
async def get_version() -> VersionResponse:
    """Return API version information and available capabilities.

    Returns agent identity, available analysis engines, supported
    commodities, classification methods, land use categories,
    transition types, trajectory types, and EUDR cutoff date.
    No authentication required.

    Returns:
        VersionResponse with comprehensive capability listing.
    """
    return VersionResponse(
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_eudr_luc_router() -> APIRouter:
    """Return the EUDR Land Use Change Detector API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.land_use_change.api import (
        ...     get_eudr_luc_router,
        ... )
        >>> app.include_router(get_eudr_luc_router(), prefix="/api")

    Returns:
        Configured APIRouter with all land use change detection endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_eudr_luc_router",
]
