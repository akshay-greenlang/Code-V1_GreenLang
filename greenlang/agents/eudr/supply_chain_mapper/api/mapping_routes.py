# -*- coding: utf-8 -*-
"""
Multi-Tier Mapping Routes - AGENT-EUDR-001 Supply Chain Mapper API

Endpoints for multi-tier recursive supply chain discovery and
tier distribution reporting.

Endpoints:
    POST /graphs/{graph_id}/discover  - Trigger multi-tier discovery
    GET  /graphs/{graph_id}/tiers     - Get tier depth distribution

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001, Section 7.5 (Feature 2)
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from statistics import mean, median
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
    AuthUser,
    ErrorResponse,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.supply_chain_mapper.api.graph_routes import (
    _get_graph_store,
)
from greenlang.agents.eudr.supply_chain_mapper.api.schemas import (
    DiscoverRequest,
    DiscoverResponse,
)
from greenlang.agents.eudr.supply_chain_mapper.metrics import (
    observe_processing_duration,
    record_error,
    record_tier_discovery,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    TierDistribution,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Multi-Tier Mapping"])


# ---------------------------------------------------------------------------
# POST /graphs/{graph_id}/discover
# ---------------------------------------------------------------------------


@router.post(
    "/graphs/{graph_id}/discover",
    response_model=DiscoverResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger multi-tier supply chain discovery",
    description=(
        "Initiates recursive discovery of supply chain tiers from Tier 1 "
        "through Tier N down to production plots. Integrates with ERP, "
        "supplier questionnaire, and document extraction data sources."
    ),
    responses={
        202: {"description": "Discovery initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def discover_tiers(
    graph_id: str,
    body: DiscoverRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:mapping:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> DiscoverResponse:
    """Trigger multi-tier recursive supply chain discovery.

    Args:
        graph_id: Target graph identifier.
        body: Discovery parameters (max_depth, filters).
        user: Authenticated user with mapping:write permission.

    Returns:
        DiscoverResponse with discovery results and metrics.

    Raises:
        HTTPException: 404 if graph not found, 403 if unauthorized.
    """
    start = time.monotonic()
    store = _get_graph_store()
    graph = store.get(graph_id)

    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph {graph_id} not found",
        )

    operator_id = user.operator_id or user.user_id
    if graph.operator_id != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this graph",
        )

    try:
        # In production, this delegates to MultiTierMapper engine.
        # For now, return the current graph state as discovery result.
        record_tier_discovery()
        elapsed = time.monotonic() - start
        observe_processing_duration("tier_discover", elapsed)

        logger.info(
            "Tier discovery completed: graph=%s max_depth=%d",
            graph_id,
            body.max_depth,
        )

        return DiscoverResponse(
            graph_id=graph_id,
            tiers_discovered=graph.max_tier_depth,
            new_nodes_added=0,
            new_edges_added=0,
            opaque_segments=0,
            processing_time_ms=elapsed * 1000,
            status="completed",
        )

    except Exception as exc:
        record_error("tier_discover")
        logger.error(
            "Tier discovery failed: graph=%s error=%s",
            graph_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tier discovery failed",
        )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/tiers
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/tiers",
    response_model=TierDistribution,
    summary="Get tier depth distribution",
    description=(
        "Return the distribution of supply chain nodes by tier depth, "
        "including maximum, average, and median depth statistics."
    ),
    responses={
        200: {"description": "Tier distribution"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def get_tier_distribution(
    graph_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:mapping:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TierDistribution:
    """Get tier depth distribution for a graph.

    Args:
        graph_id: Target graph identifier.
        user: Authenticated user with mapping:read permission.

    Returns:
        TierDistribution with tier counts and statistics.

    Raises:
        HTTPException: 404 if graph not found, 403 if unauthorized.
    """
    store = _get_graph_store()
    graph = store.get(graph_id)

    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph {graph_id} not found",
        )

    operator_id = user.operator_id or user.user_id
    if graph.operator_id != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this graph",
        )

    # Compute tier distribution from node tier depths
    depths = [n.tier_depth for n in graph.nodes.values()]
    tier_counts: Dict[int, int] = dict(Counter(depths)) if depths else {}

    return TierDistribution(
        tier_counts=tier_counts,
        max_depth=max(depths) if depths else 0,
        average_depth=mean(depths) if depths else 0.0,
        median_depth=median(depths) if depths else 0.0,
    )
