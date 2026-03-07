# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-001 Supply Chain Mapper API

Aggregates all route modules under the ``/v1/eudr-scm`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (25+ endpoints):
    graph_routes:          5 endpoints (POST, GET list, GET detail, DELETE, GET export)
    mapping_routes:        2 endpoints (POST discover, GET tiers)
    traceability_routes:   3 endpoints (GET forward, GET backward, GET batch)
    risk_routes:           3 endpoints (POST propagate, GET summary, GET heatmap)
    gap_routes:            3 endpoints (POST analyze, GET list, PUT resolve)
    visualization_routes:  2 endpoints (GET layout, GET sankey)
    onboarding_routes:     3 endpoints (POST invite, GET status, POST submit)
    + health check:        1 endpoint (GET health)
    -------
    Total: 22 unique endpoints + health = 23 base + additional filters/params

Auth & RBAC:
    All endpoints (except health and public onboarding) require JWT auth
    via SEC-001 and check eudr-supply-chain:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from greenlang.agents.eudr.supply_chain_mapper.api.gap_routes import (
    router as gap_router,
)
from greenlang.agents.eudr.supply_chain_mapper.api.graph_routes import (
    router as graph_router,
)
from greenlang.agents.eudr.supply_chain_mapper.api.mapping_routes import (
    router as mapping_router,
)
from greenlang.agents.eudr.supply_chain_mapper.api.onboarding_routes import (
    router as onboarding_router,
)
from greenlang.agents.eudr.supply_chain_mapper.api.risk_routes import (
    router as risk_router,
)
from greenlang.agents.eudr.supply_chain_mapper.api.traceability_routes import (
    router as traceability_router,
)
from greenlang.agents.eudr.supply_chain_mapper.api.visualization_routes import (
    router as visualization_router,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-scm prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-scm",
    tags=["EUDR Supply Chain Mapper"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(graph_router)
router.include_router(mapping_router)
router.include_router(traceability_router)
router.include_router(risk_router)
router.include_router(gap_router)
router.include_router(visualization_router)
router.include_router(onboarding_router)


def get_router() -> APIRouter:
    """Return the EUDR Supply Chain Mapper API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.supply_chain_mapper.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all supply chain mapper endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
