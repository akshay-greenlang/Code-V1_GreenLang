# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-020 Deforestation Alert System API

Aggregates all route modules under the ``/v1/eudr-das`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (30+ endpoints):
    satellite_routes:  4 endpoints (POST detect, POST scan, GET sources, GET imagery)
    alert_routes:      6 endpoints (GET list, GET detail, POST create, POST batch, GET summary, GET statistics)
    severity_routes:   4 endpoints (POST classify, POST reclassify, GET thresholds, GET distribution)
    buffer_routes:     5 endpoints (POST create, PUT update, POST check, GET violations, GET zones)
    cutoff_routes:     4 endpoints (POST verify, POST batch-verify, GET evidence, GET timeline)
    baseline_routes:   4 endpoints (POST establish, POST compare, PUT update, GET coverage)
    workflow_routes:   6 endpoints (POST triage, POST assign, POST investigate, POST resolve, POST escalate, GET sla)
    compliance_routes: 4 endpoints (POST assess, GET affected-products, GET recommendations, POST remediation)
    + health check:    1 endpoint (GET health)
    -------
    Total: 38 unique endpoints + health = 39

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-deforestation-alert:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from greenlang.agents.eudr.deforestation_alert_system.api.satellite_routes import (
    router as satellite_router,
)
from greenlang.agents.eudr.deforestation_alert_system.api.alert_routes import (
    router as alert_router,
)
from greenlang.agents.eudr.deforestation_alert_system.api.severity_routes import (
    router as severity_router,
)
from greenlang.agents.eudr.deforestation_alert_system.api.buffer_routes import (
    router as buffer_router,
)
from greenlang.agents.eudr.deforestation_alert_system.api.cutoff_routes import (
    router as cutoff_router,
)
from greenlang.agents.eudr.deforestation_alert_system.api.baseline_routes import (
    router as baseline_router,
)
from greenlang.agents.eudr.deforestation_alert_system.api.workflow_routes import (
    router as workflow_router,
)
from greenlang.agents.eudr.deforestation_alert_system.api.compliance_routes import (
    router as compliance_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-das prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-das",
    tags=["eudr-deforestation-alert-system"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(satellite_router)
router.include_router(alert_router)
router.include_router(severity_router)
router.include_router(buffer_router)
router.include_router(cutoff_router)
router.include_router(baseline_router)
router.include_router(workflow_router)
router.include_router(compliance_router)


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    tags=["health"],
    summary="Deforestation Alert System health check",
    description="Returns service health status for AGENT-EUDR-020.",
)
async def health_check() -> dict:
    """Return health status for the Deforestation Alert System agent.

    Returns:
        Dictionary with status, agent ID, and component name.
    """
    return {
        "status": "healthy",
        "agent_id": "GL-EUDR-DAS-020",
        "agent": "EUDR-020",
        "component": "deforestation-alert-system",
        "version": "1.0.0",
    }


def get_router() -> APIRouter:
    """Return the EUDR Deforestation Alert System API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.deforestation_alert_system.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all deforestation alert system endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
