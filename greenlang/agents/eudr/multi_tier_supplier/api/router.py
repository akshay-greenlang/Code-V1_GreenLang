# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-008 Multi-Tier Supplier Tracker API

Aggregates all route modules under the ``/v1/eudr-mst`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (32 endpoints + health):
    discovery_routes:   4 endpoints (POST discover, POST batch, POST from-declaration, POST from-questionnaire)
    profile_routes:     6 endpoints (POST create, GET get, PUT update, DELETE deactivate, POST search, POST batch)
    tier_routes:        8 endpoints (GET tier/{id}, POST assess, GET visibility, GET gaps,
                                     POST relationship, PUT relationship/{id}, GET relationships/{id},
                                     POST relationships/history)
    compliance_routes:  8 endpoints (POST risk/assess, POST risk/propagate, GET risk/{id}, POST risk/batch,
                                     POST compliance/check, GET compliance/{id}, POST compliance/batch,
                                     GET compliance/alerts)
    report_routes:      5 endpoints (POST audit, POST tier-summary, POST gaps, GET report/{id},
                                     GET report/{id}/download)
    batch_routes:       3 endpoints (POST batch, GET batch/{id}, DELETE batch/{id})
    + health check:     1 endpoint (GET health)
    -------
    Total: 32 unique endpoints + health = 33

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-mst:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from greenlang.agents.eudr.multi_tier_supplier.api.discovery_routes import (
    router as discovery_router,
)
from greenlang.agents.eudr.multi_tier_supplier.api.profile_routes import (
    router as profile_router,
)
from greenlang.agents.eudr.multi_tier_supplier.api.tier_routes import (
    router as tier_router,
)
from greenlang.agents.eudr.multi_tier_supplier.api.compliance_routes import (
    router as compliance_router,
)
from greenlang.agents.eudr.multi_tier_supplier.api.report_routes import (
    router as report_router,
)
from greenlang.agents.eudr.multi_tier_supplier.api.batch_routes import (
    router as batch_router,
)
from greenlang.agents.eudr.multi_tier_supplier.api.schemas import (
    HealthResponseSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-mst prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-mst",
    tags=["EUDR Multi-Tier Supplier Tracker"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(
    discovery_router,
    prefix="/discover",
    tags=["Supplier Discovery"],
)
router.include_router(
    profile_router,
    prefix="/suppliers",
    tags=["Supplier Profiles"],
)
router.include_router(
    tier_router,
    tags=["Tier Depth & Relationships"],
)
router.include_router(
    compliance_router,
    tags=["Risk & Compliance"],
)
router.include_router(
    report_router,
    prefix="/reports",
    tags=["Reporting"],
)
router.include_router(
    batch_router,
    prefix="/batch",
    tags=["Batch Jobs"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponseSchema,
    summary="Health check",
    description="Check EUDR Multi-Tier Supplier Tracker API health.",
    tags=["System"],
)
async def health_check() -> HealthResponseSchema:
    """Health check endpoint for load balancers and monitoring.

    Returns service identity, version, and current timestamp.
    No authentication required.
    """
    return HealthResponseSchema(
        status="healthy",
        agent_id="GL-EUDR-MST-008",
        agent_name="EUDR Multi-Tier Supplier Tracker",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


def get_router() -> APIRouter:
    """Return the Multi-Tier Supplier Tracker API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.multi_tier_supplier.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all multi-tier supplier endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
