# -*- coding: utf-8 -*-
"""
Supplier Risk Scorer API Router - AGENT-EUDR-017

Main router aggregating 8 domain-specific sub-routers plus a health
endpoint for the Supplier Risk Scorer Agent.

Prefix: /v1/eudr-srs
Tags: eudr-supplier-risk-scorer

Sub-routers:
    - supplier_routes: Supplier risk assessment (6 endpoints)
    - due_diligence_routes: Due diligence tracking (5 endpoints)
    - documentation_routes: Documentation analysis (5 endpoints)
    - certification_routes: Certification validation (5 endpoints)
    - geographic_routes: Geographic sourcing (5 endpoints)
    - network_routes: Supplier network analysis (5 endpoints)
    - monitoring_routes: Continuous monitoring (5 endpoints)
    - report_routes: Risk reporting (6 endpoints)
    + health (1 endpoint) = 43 total

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-srs:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017, Section 7.4
Agent ID: GL-EUDR-SRS-017
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

# ---------------------------------------------------------------------------
# Sub-router imports (try/except for import safety during startup)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.supplier_routes import (
        router as supplier_router,
    )
except ImportError:
    supplier_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.due_diligence_routes import (
        router as due_diligence_router,
    )
except ImportError:
    due_diligence_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.documentation_routes import (
        router as documentation_router,
    )
except ImportError:
    documentation_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.certification_routes import (
        router as certification_router,
    )
except ImportError:
    certification_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.geographic_routes import (
        router as geographic_router,
    )
except ImportError:
    geographic_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.network_routes import (
        router as network_router,
    )
except ImportError:
    network_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.monitoring_routes import (
        router as monitoring_router,
    )
except ImportError:
    monitoring_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.supplier_risk_scorer.api.report_routes import (
        router as report_router,
    )
except ImportError:
    report_router = None  # type: ignore[assignment]

from greenlang.agents.eudr.supplier_risk_scorer.api.schemas import (
    HealthSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup time for uptime tracking
# ---------------------------------------------------------------------------

_startup_time = datetime.now(timezone.utc).replace(microsecond=0)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-srs prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-srs",
    tags=["eudr-supplier-risk-scorer"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers that were successfully imported
if supplier_router is not None:
    router.include_router(supplier_router)
if due_diligence_router is not None:
    router.include_router(due_diligence_router)
if documentation_router is not None:
    router.include_router(documentation_router)
if certification_router is not None:
    router.include_router(certification_router)
if geographic_router is not None:
    router.include_router(geographic_router)
if network_router is not None:
    router.include_router(network_router)
if monitoring_router is not None:
    router.include_router(monitoring_router)
if report_router is not None:
    router.include_router(report_router)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthSchema,
    summary="Health check",
    description=(
        "Check EUDR Supplier Risk Scorer API health and component "
        "status. No authentication required."
    ),
    tags=["System"],
)
async def health_check() -> HealthSchema:
    """Health check endpoint for load balancers and monitoring.

    Returns:
        HealthSchema with service health and component status.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    uptime_seconds = (now - _startup_time).total_seconds()

    return HealthSchema(
        uptime_seconds=uptime_seconds,
        checked_at=now,
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the EUDR Supplier Risk Scorer API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.supplier_risk_scorer.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all supplier risk scorer endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
