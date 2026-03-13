# -*- coding: utf-8 -*-
"""
Country Risk Evaluator API Router - AGENT-EUDR-016

Main router aggregating 8 domain-specific sub-routers plus a health
endpoint for the Country Risk Evaluator Agent.

Prefix: /v1/eudr-cre
Tags: eudr-country-risk-evaluator

Sub-routers:
    - country_routes: Country risk assessment (5 endpoints)
    - commodity_routes: Commodity risk analysis (4 endpoints)
    - hotspot_routes: Deforestation hotspot detection (5 endpoints)
    - governance_routes: Governance evaluation (4 endpoints)
    - due_diligence_routes: Due diligence classification (5 endpoints)
    - trade_flow_routes: Trade flow analysis (5 endpoints)
    - report_routes: Report generation (5 endpoints)
    - regulatory_routes: Regulatory update tracking (4 endpoints)
    + health (1 endpoint) = 38 total

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-cre:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016, Section 7.4
Agent ID: GL-EUDR-CRE-016
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
    from greenlang.agents.eudr.country_risk_evaluator.api.country_routes import (
        router as country_router,
    )
except ImportError:
    country_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.country_risk_evaluator.api.commodity_routes import (
        router as commodity_router,
    )
except ImportError:
    commodity_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.country_risk_evaluator.api.hotspot_routes import (
        router as hotspot_router,
    )
except ImportError:
    hotspot_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.country_risk_evaluator.api.governance_routes import (
        router as governance_router,
    )
except ImportError:
    governance_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.country_risk_evaluator.api.due_diligence_routes import (
        router as due_diligence_router,
    )
except ImportError:
    due_diligence_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.country_risk_evaluator.api.trade_flow_routes import (
        router as trade_flow_router,
    )
except ImportError:
    trade_flow_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.country_risk_evaluator.api.report_routes import (
        router as report_router,
    )
except ImportError:
    report_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.country_risk_evaluator.api.regulatory_routes import (
        router as regulatory_router,
    )
except ImportError:
    regulatory_router = None  # type: ignore[assignment]

from greenlang.agents.eudr.country_risk_evaluator.api.schemas import (
    HealthSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup time for uptime tracking
# ---------------------------------------------------------------------------

_startup_time = datetime.now(timezone.utc).replace(microsecond=0)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-cre prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-cre",
    tags=["eudr-country-risk-evaluator"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers that were successfully imported
if country_router is not None:
    router.include_router(country_router)
if commodity_router is not None:
    router.include_router(commodity_router)
if hotspot_router is not None:
    router.include_router(hotspot_router)
if governance_router is not None:
    router.include_router(governance_router)
if due_diligence_router is not None:
    router.include_router(due_diligence_router)
if trade_flow_router is not None:
    router.include_router(trade_flow_router)
if report_router is not None:
    router.include_router(report_router)
if regulatory_router is not None:
    router.include_router(regulatory_router)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthSchema,
    summary="Health check",
    description=(
        "Check EUDR Country Risk Evaluator API health and component "
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
    """Return the EUDR Country Risk Evaluator API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.country_risk_evaluator.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all country risk evaluator endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
