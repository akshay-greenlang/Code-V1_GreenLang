# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-021 Indigenous Rights Checker API

Aggregates all route modules under the ``/v1/eudr-irc`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (25 endpoints + health):
    territory_routes:     5 endpoints (POST create, GET list, GET detail, PUT update, DELETE archive)
    fpic_routes:          4 endpoints (POST verify, GET documents, GET document detail, POST score)
    overlap_routes:       4 endpoints (POST analyze, GET by-plot, GET by-territory, POST bulk)
    consultation_routes:  3 endpoints (POST create, GET list, GET detail)
    violation_routes:     4 endpoints (POST detect, GET list, GET detail, PUT resolve)
    registry_routes:      3 endpoints (POST register, GET list, GET detail)
    compliance_routes:    2 endpoints (GET report, POST assess)
    + health check:       1 endpoint  (GET health)
    -------
    Total: 25 unique endpoints + health = 26

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-irc:* permissions (14 total) via SEC-002.

RBAC Permissions (14):
    eudr-irc:territories:read
    eudr-irc:territories:create
    eudr-irc:territories:update
    eudr-irc:territories:delete
    eudr-irc:fpic:read
    eudr-irc:fpic:verify
    eudr-irc:overlap:read
    eudr-irc:overlap:analyze
    eudr-irc:consultations:read
    eudr-irc:consultations:create
    eudr-irc:violations:read
    eudr-irc:violations:detect
    eudr-irc:violations:resolve
    eudr-irc:compliance:read

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from greenlang.agents.eudr.indigenous_rights_checker.api.territory_routes import (
    router as territory_router,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.fpic_routes import (
    router as fpic_router,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.overlap_routes import (
    router as overlap_router,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.consultation_routes import (
    router as consultation_router,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.violation_routes import (
    router as violation_router,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.registry_routes import (
    router as registry_router,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.compliance_routes import (
    router as compliance_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-irc prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-irc",
    tags=["eudr-indigenous-rights-checker"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(territory_router)
router.include_router(fpic_router)
router.include_router(overlap_router)
router.include_router(consultation_router)
router.include_router(violation_router)
router.include_router(registry_router)
router.include_router(compliance_router)


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    tags=["health"],
    summary="Indigenous Rights Checker health check",
    description="Returns service health status for AGENT-EUDR-021.",
)
async def health_check() -> dict:
    """Return health status for the Indigenous Rights Checker agent.

    Returns:
        Dictionary with status, agent ID, and component name.
    """
    return {
        "status": "healthy",
        "agent_id": "GL-EUDR-IRC-021",
        "agent": "EUDR-021",
        "component": "indigenous-rights-checker",
        "version": "1.0.0",
    }


def get_router() -> APIRouter:
    """Return the EUDR Indigenous Rights Checker API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.indigenous_rights_checker.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all indigenous rights checker endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
