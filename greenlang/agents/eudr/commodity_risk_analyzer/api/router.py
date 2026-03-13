# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-018 Commodity Risk Analyzer API

Aggregates all route modules under the ``/v1/eudr-cra`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (40+ endpoints):
    commodity_routes:       6 endpoints (POST profile, POST batch, GET risk, GET history, GET compare, GET summary)
    derived_product_routes: 5 endpoints (POST analyze, GET chain, GET risk, GET mapping, POST trace)
    price_routes:           5 endpoints (GET current, GET history, GET volatility, GET disruptions, POST forecast)
    production_routes:      5 endpoints (POST forecast, GET yield, GET climate-impact, GET seasonal, GET summary)
    substitution_routes:    5 endpoints (POST detect, GET history, GET alerts, POST verify, GET patterns)
    regulatory_routes:      5 endpoints (GET requirements, POST check, GET penalty, GET updates, GET docs)
    due_diligence_routes:   5 endpoints (POST initiate, GET status, POST evidence, GET pending, POST complete)
    portfolio_routes:       4 endpoints (POST analyze, GET concentration, GET diversification, GET summary)
    + health check:         1 endpoint (GET health)
    -------
    Total: 41 unique endpoints + health = 42

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-commodity-risk:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from greenlang.agents.eudr.commodity_risk_analyzer.api.commodity_routes import (
    router as commodity_router,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.derived_product_routes import (
    router as derived_product_router,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.price_routes import (
    router as price_router,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.production_routes import (
    router as production_router,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.substitution_routes import (
    router as substitution_router,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.regulatory_routes import (
    router as regulatory_router,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.due_diligence_routes import (
    router as due_diligence_router,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.portfolio_routes import (
    router as portfolio_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-cra prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-cra",
    tags=["eudr-commodity-risk-analyzer"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(commodity_router)
router.include_router(derived_product_router)
router.include_router(price_router)
router.include_router(production_router)
router.include_router(substitution_router)
router.include_router(regulatory_router)
router.include_router(due_diligence_router)
router.include_router(portfolio_router)


def get_router() -> APIRouter:
    """Return the EUDR Commodity Risk Analyzer API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.commodity_risk_analyzer.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all commodity risk analyzer endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
