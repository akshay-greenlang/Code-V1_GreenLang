# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-024 Third-Party Audit Manager API

Aggregates all route modules under the ``/v1/eudr-tam`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (~39 domain endpoints + health/stats):
    audit_routes:       6 endpoints  (POST, GET list, GET detail, POST schedule,
                                      POST start, POST complete)
    auditor_routes:     5 endpoints  (POST register, GET list, GET detail,
                                      POST match, POST qualification)
    checklist_routes:   3 endpoints  (GET checklists, POST custom, POST progress)
    evidence_routes:    3 endpoints  (POST upload, GET list, DELETE remove)
    nc_routes:          5 endpoints  (POST create, GET list, GET detail,
                                      POST classify, POST root-cause)
    car_routes:         6 endpoints  (POST issue, GET list, GET detail,
                                      POST submit-plan, POST verify, POST close)
    certificate_routes: 4 endpoints  (POST create, GET list, GET supplier certs,
                                      POST validate-eudr)
    report_routes:      4 endpoints  (POST generate, GET list, GET detail,
                                      GET download)
    authority_routes:   5 endpoints  (POST create, GET list, POST respond,
                                      GET compliance-rate, GET nc-trends)
    + health check:     1 endpoint   (GET /health)
    + stats:            1 endpoint   (GET /stats)
    -------
    Total: ~43 endpoints

Auth & RBAC (22 permissions):
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-tam:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from greenlang.agents.eudr.third_party_audit_manager.api.audit_routes import (
    router as audit_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.auditor_routes import (
    router as auditor_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.checklist_routes import (
    router as checklist_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.evidence_routes import (
    router as evidence_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.nc_routes import (
    router as nc_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.car_routes import (
    router as car_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.certificate_routes import (
    router as certificate_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.report_routes import (
    router as report_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.authority_routes import (
    router as authority_router,
)
from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    get_current_user,
    get_analytics_engine,
    rate_limit_standard,
    require_permission,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-tam prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-tam",
    tags=["eudr-third-party-audit-manager"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(audit_router)
router.include_router(auditor_router)
router.include_router(checklist_router)
router.include_router(evidence_router)
router.include_router(nc_router)
router.include_router(car_router)
router.include_router(certificate_router)
router.include_router(report_router)
router.include_router(authority_router)


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    tags=["System"],
    summary="Third-Party Audit Manager health check",
    description="Returns service health status for AGENT-EUDR-024.",
)
async def health_check() -> dict:
    """Return health status for the Third-Party Audit Manager agent.

    Returns:
        Dictionary with status, agent ID, and component name.
    """
    return {
        "status": "healthy",
        "agent_id": "GL-EUDR-TAM-024",
        "agent": "EUDR-024",
        "component": "third-party-audit-manager",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/stats",
    tags=["System"],
    summary="Service statistics",
    description="Returns aggregate service statistics for AGENT-EUDR-024.",
)
async def service_stats(
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rl: None = Depends(rate_limit_standard),
    analytics_engine: object = Depends(get_analytics_engine),
) -> dict:
    """Return aggregate service statistics.

    Args:
        user: Authenticated user with analytics:read permission.
        analytics_engine: AuditAnalyticsEngine singleton.

    Returns:
        Dictionary with service statistics (active audits, open CARs, etc.).
    """
    try:
        if hasattr(analytics_engine, "get_dashboard_summary"):
            summary = await analytics_engine.get_dashboard_summary()
            return {
                "status": "ok",
                "agent_id": "GL-EUDR-TAM-024",
                "stats": summary,
            }
    except Exception:
        logger.exception("Failed to retrieve service stats")

    return {
        "status": "ok",
        "agent_id": "GL-EUDR-TAM-024",
        "stats": {},
    }


def get_router() -> APIRouter:
    """Return the EUDR Third-Party Audit Manager API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.third_party_audit_manager.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all audit manager endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
