# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-019 Corruption Index Monitor API

Aggregates all route modules under the ``/v1/eudr-cim`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (40+ endpoints):
    cpi_routes:           6 endpoints (GET score, GET history, GET rankings, GET regional, POST batch, GET summary)
    wgi_routes:           5 endpoints (GET indicators, GET history, GET dimension, POST compare, GET rankings)
    bribery_routes:       5 endpoints (POST assess, GET risk, GET sectors, GET high-risk, GET sector-analysis)
    institutional_routes: 5 endpoints (GET quality, GET governance, POST assess, GET forest-governance, POST compare)
    trend_routes:         5 endpoints (POST analyze, GET trajectory, POST prediction, GET improving, GET deteriorating)
    correlation_routes:   5 endpoints (POST analyze, GET deforestation, POST regression, GET heatmap, GET causal-pathways)
    alert_routes:         5 endpoints (GET list, GET detail, POST configure, POST acknowledge, GET summary)
    compliance_routes:    4 endpoints (POST assess-impact, GET impact, GET dd-recommendations, GET classifications)
    + health check:       1 endpoint (GET health)
    -------
    Total: 41 unique endpoints + health = 42

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-corruption-index:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from greenlang.agents.eudr.corruption_index_monitor.api.cpi_routes import (
    router as cpi_router,
)
from greenlang.agents.eudr.corruption_index_monitor.api.wgi_routes import (
    router as wgi_router,
)
from greenlang.agents.eudr.corruption_index_monitor.api.bribery_routes import (
    router as bribery_router,
)
from greenlang.agents.eudr.corruption_index_monitor.api.institutional_routes import (
    router as institutional_router,
)
from greenlang.agents.eudr.corruption_index_monitor.api.trend_routes import (
    router as trend_router,
)
from greenlang.agents.eudr.corruption_index_monitor.api.correlation_routes import (
    router as correlation_router,
)
from greenlang.agents.eudr.corruption_index_monitor.api.alert_routes import (
    router as alert_router,
)
from greenlang.agents.eudr.corruption_index_monitor.api.compliance_routes import (
    router as compliance_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-cim prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-cim",
    tags=["eudr-corruption-index-monitor"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(cpi_router)
router.include_router(wgi_router)
router.include_router(bribery_router)
router.include_router(institutional_router)
router.include_router(trend_router)
router.include_router(correlation_router)
router.include_router(alert_router)
router.include_router(compliance_router)


def get_router() -> APIRouter:
    """Return the EUDR Corruption Index Monitor API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.corruption_index_monitor.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all corruption index monitor endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
