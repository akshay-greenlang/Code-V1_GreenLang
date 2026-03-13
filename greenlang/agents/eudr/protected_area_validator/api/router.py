# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-022 Protected Area Validator API

Aggregates all route modules under the ``/v1/eudr-pav`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (32 endpoints + health):
    protected_area_routes: 6 endpoints (POST create, GET list, GET detail, PUT update, DELETE archive, POST search)
    overlap_routes:        5 endpoints (POST detect, POST analyze, POST bulk, GET by-plot, GET by-area)
    buffer_zone_routes:    4 endpoints (POST monitor, GET violations, POST analyze, POST bulk)
    designation_routes:    3 endpoints (POST validate, GET status, GET history)
    risk_routes:           4 endpoints (POST score, GET heatmap, GET summary, GET proximity-alerts)
    violation_routes:      4 endpoints (POST detect, GET list, PUT resolve, PUT escalate)
    compliance_routes:     3 endpoints (POST assess, GET report, GET audit-trail)
    paddd_routes:          3 endpoints (POST monitor, GET events, POST impact-assessment)
    + health check:        1 endpoint  (GET health)
    -------
    Total: 32 unique endpoints + health = 33

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-pav:* permissions (18 total) via SEC-002.

RBAC Permissions (18):
    eudr-pav:protected-areas:read
    eudr-pav:protected-areas:create
    eudr-pav:protected-areas:update
    eudr-pav:protected-areas:delete
    eudr-pav:overlap:read
    eudr-pav:overlap:analyze
    eudr-pav:buffer-zones:read
    eudr-pav:buffer-zones:monitor
    eudr-pav:designation:read
    eudr-pav:designation:validate
    eudr-pav:risk:read
    eudr-pav:risk:score
    eudr-pav:violations:read
    eudr-pav:violations:detect
    eudr-pav:violations:resolve
    eudr-pav:compliance:read
    eudr-pav:paddd:read
    eudr-pav:paddd:monitor

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 Protected Area Validator (GL-EUDR-PAV-022)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from greenlang.agents.eudr.protected_area_validator.api.protected_area_routes import (
    router as protected_area_router,
)
from greenlang.agents.eudr.protected_area_validator.api.overlap_routes import (
    router as overlap_router,
)
from greenlang.agents.eudr.protected_area_validator.api.buffer_zone_routes import (
    router as buffer_zone_router,
)
from greenlang.agents.eudr.protected_area_validator.api.designation_routes import (
    router as designation_router,
)
from greenlang.agents.eudr.protected_area_validator.api.risk_routes import (
    router as risk_router,
)
from greenlang.agents.eudr.protected_area_validator.api.violation_routes import (
    router as violation_router,
)
from greenlang.agents.eudr.protected_area_validator.api.compliance_routes import (
    router as compliance_router,
)
from greenlang.agents.eudr.protected_area_validator.api.paddd_routes import (
    router as paddd_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-pav prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-pav",
    tags=["eudr-protected-area-validator"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(protected_area_router)
router.include_router(overlap_router)
router.include_router(buffer_zone_router)
router.include_router(designation_router)
router.include_router(risk_router)
router.include_router(violation_router)
router.include_router(compliance_router)
router.include_router(paddd_router)


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    tags=["health"],
    summary="Protected Area Validator health check",
    description="Returns service health status for AGENT-EUDR-022.",
)
async def health_check() -> dict:
    """Return health status for the Protected Area Validator agent.

    Returns:
        Dictionary with status, agent ID, and component name.
    """
    return {
        "status": "healthy",
        "agent_id": "GL-EUDR-PAV-022",
        "agent": "EUDR-022",
        "component": "protected-area-validator",
        "version": "1.0.0",
    }


def get_router() -> APIRouter:
    """Return the EUDR Protected Area Validator API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.protected_area_validator.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all protected area validator endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
