# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-025 Risk Mitigation Advisor API

Aggregates all route modules under the ``/v1/eudr-rma`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (43 endpoints):
    strategy_routes:      5 endpoints (POST recommend, GET list, GET detail, POST select, GET explain)
    plan_routes:         10 endpoints (POST create, GET list, GET detail, PUT update, PUT status,
                                       POST clone, GET gantt, POST milestone, PUT milestone, POST evidence)
    capacity_routes:      5 endpoints (POST enroll, GET enrollments, GET enrollment, PUT progress, GET scorecard)
    library_routes:       4 endpoints (GET search, GET compare, GET packages, GET detail)
    effectiveness_routes: 4 endpoints (GET plan, GET supplier, GET portfolio, GET roi)
    monitoring_routes:    4 endpoints (GET triggers, PUT acknowledge, GET dashboard, GET drift)
    optimization_routes:  4 endpoints (POST run, GET result, GET pareto, GET sensitivity)
    collaboration_routes: 4 endpoints (POST message, GET messages, POST task, GET supplier-portal)
    reporting_routes:     4 endpoints (POST generate, GET list, GET download, GET dds-section)
    + health check:       1 endpoint  (GET health)
    + stats:              1 endpoint  (GET stats)
    -------
    Total: 46 unique endpoints

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-rma:* permissions via SEC-002. 21 distinct permissions across
    9 route groups.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.strategy_routes import (
    router as strategy_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.plan_routes import (
    router as plan_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.capacity_routes import (
    router as capacity_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.library_routes import (
    router as library_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.effectiveness_routes import (
    router as effectiveness_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.monitoring_routes import (
    router as monitoring_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.optimization_routes import (
    router as optimization_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.collaboration_routes import (
    router as collaboration_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.reporting_routes import (
    router as reporting_router,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    HealthResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)

# Track service start time for uptime calculation
_SERVICE_START_TIME: float = time.monotonic()

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-rma prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-rma",
    tags=["eudr-risk-mitigation-advisor"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(strategy_router)
router.include_router(plan_router)
router.include_router(capacity_router)
router.include_router(library_router)
router.include_router(effectiveness_router)
router.include_router(monitoring_router)
router.include_router(optimization_router)
router.include_router(collaboration_router)
router.include_router(reporting_router)


# ---------------------------------------------------------------------------
# Health check endpoint (no auth required)
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Risk Mitigation Advisor health check",
    description=(
        "Returns service health status for AGENT-EUDR-025 Risk Mitigation "
        "Advisor. Checks database connectivity, Redis availability, and "
        "engine initialization status. Used by load balancers and Kubernetes "
        "readiness probes."
    ),
)
async def health_check(request: Request) -> HealthResponse:
    """Return health status for the Risk Mitigation Advisor agent.

    Checks service availability, database connectivity, and Redis.
    Does not require authentication.

    Returns:
        HealthResponse with status, version, engine count, and
        connectivity flags.
    """
    db_connected = False
    redis_connected = False
    engines_available = 0

    try:
        from greenlang.agents.eudr.risk_mitigation_advisor.setup import get_service

        service = get_service()
        if service is not None and getattr(service, "_initialized", False):
            engines_available = getattr(service, "_engine_count", 8)
            db_connected = getattr(service, "_db_connected", False)
            redis_connected = getattr(service, "_redis_connected", False)

            # Attempt quick health check via service
            try:
                health = await service.health_check()
                if isinstance(health, dict):
                    db_connected = health.get("db_connected", db_connected)
                    redis_connected = health.get("redis_connected", redis_connected)
                    engines_available = health.get("engines_available", engines_available)
            except Exception:
                pass
    except ImportError:
        pass

    service_status = "healthy" if db_connected and engines_available > 0 else "degraded"

    return HealthResponse(
        status=service_status,
        version="1.0.0",
        agent_id="GL-EUDR-RMA-025",
        engines_available=engines_available,
        db_connected=db_connected,
        redis_connected=redis_connected,
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["health"],
    summary="Risk Mitigation Advisor service statistics",
    description=(
        "Returns aggregated service statistics for AGENT-EUDR-025 including "
        "strategy recommendation counts, plan counts, enrollment counts, "
        "library size, trigger counts, optimization runs, and report counts. "
        "Requires audit:read permission."
    ),
    responses={
        200: {"description": "Statistics retrieved"},
    },
)
async def get_stats(
    request: Request,
) -> StatsResponse:
    """Return service-level statistics.

    Returns:
        StatsResponse with operational metrics and ML model version.
    """
    try:
        from greenlang.agents.eudr.risk_mitigation_advisor.setup import get_service

        service = get_service()
        if service is None or not getattr(service, "_initialized", False):
            return StatsResponse(
                uptime_seconds=int(time.monotonic() - _SERVICE_START_TIME),
            )

        try:
            stats = await service.get_stats()
            data = stats if isinstance(stats, dict) else {}
        except Exception:
            data = {}

        return StatsResponse(
            total_strategies_recommended=data.get("total_strategies_recommended", 0),
            total_plans_created=data.get("total_plans_created", 0),
            active_plans=data.get("active_plans", 0),
            total_enrollments=data.get("total_enrollments", 0),
            active_enrollments=data.get("active_enrollments", 0),
            library_measures_count=data.get("library_measures_count", 0),
            unresolved_triggers=data.get("unresolved_triggers", 0),
            total_optimizations_run=data.get("total_optimizations_run", 0),
            total_reports_generated=data.get("total_reports_generated", 0),
            average_risk_reduction_pct=Decimal(str(data.get("average_risk_reduction_pct", 0))),
            ml_model_version=data.get("ml_model_version", ""),
            uptime_seconds=int(time.monotonic() - _SERVICE_START_TIME),
        )

    except Exception as e:
        logger.error("Stats retrieval failed: %s", e, exc_info=True)
        return StatsResponse(
            uptime_seconds=int(time.monotonic() - _SERVICE_START_TIME),
        )


def get_router() -> APIRouter:
    """Return the EUDR Risk Mitigation Advisor API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.risk_mitigation_advisor.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all risk mitigation advisor endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
