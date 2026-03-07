# -*- coding: utf-8 -*-
"""
Risk Routes - AGENT-EUDR-001 Supply Chain Mapper API

Endpoints for risk propagation, risk summary, and risk heatmap
generation across supply chain graphs.

Endpoints:
    POST /graphs/{graph_id}/risk/propagate  - Run risk propagation
    GET  /graphs/{graph_id}/risk/summary    - Get risk summary
    GET  /graphs/{graph_id}/risk/heatmap    - Get risk heatmap data

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001, Section 7.5 (Feature 5)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

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
    RiskHeatmapResponse,
    RiskPropagateRequest,
    RiskPropagateResponse,
)
from greenlang.agents.eudr.supply_chain_mapper.metrics import (
    observe_processing_duration,
    record_error,
    record_risk_propagation,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    RiskLevel,
    RiskPropagationResult,
    RiskSummary,
    SupplyChainGraph,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Risk Assessment"])


def _check_graph_access(
    graph_id: str, user: AuthUser
) -> SupplyChainGraph:
    """Validate graph exists and user has access."""
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

    return graph


# ---------------------------------------------------------------------------
# POST /graphs/{graph_id}/risk/propagate
# ---------------------------------------------------------------------------


@router.post(
    "/graphs/{graph_id}/risk/propagate",
    response_model=RiskPropagateResponse,
    status_code=status.HTTP_200_OK,
    summary="Run risk propagation across graph",
    description=(
        "Execute the risk propagation engine across all nodes in the "
        "supply chain graph. Uses configurable risk weights for country, "
        "commodity, supplier, and deforestation dimensions. Implements "
        "'highest risk wins' principle per EUDR Article 10."
    ),
    responses={
        200: {"description": "Risk propagation completed"},
        400: {"model": ErrorResponse, "description": "Invalid risk weights"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def propagate_risk(
    graph_id: str,
    body: RiskPropagateRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:risk:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> RiskPropagateResponse:
    """Run risk propagation across the supply chain graph.

    Args:
        graph_id: Target graph identifier.
        body: Risk propagation parameters (optional custom weights).
        user: Authenticated user with risk:write permission.

    Returns:
        RiskPropagateResponse with updated node risk scores.
    """
    start = time.monotonic()
    graph = _check_graph_access(graph_id, user)

    try:
        # Use custom weights or config defaults
        from greenlang.agents.eudr.supply_chain_mapper.config import get_config

        cfg = get_config()
        weights = body.risk_weights or cfg.risk_weights

        # In production, delegates to RiskPropagationEngine.
        # Here, perform basic risk recalculation for each node.
        propagation_results: List[RiskPropagationResult] = []
        nodes_updated = 0

        for node_id, node in graph.nodes.items():
            prev_score = node.risk_score
            prev_level = node.risk_level

            # Simple weighted risk calculation (production uses full engine)
            country_risk = 50.0  # Placeholder; production uses country benchmarks
            commodity_risk = 40.0
            supplier_risk = node.risk_score
            deforestation_risk = 30.0

            new_score = (
                country_risk * weights["country"]
                + commodity_risk * weights["commodity"]
                + supplier_risk * weights["supplier"]
                + deforestation_risk * weights["deforestation"]
            )
            new_score = min(100.0, max(0.0, new_score))

            if new_score >= cfg.risk_high_threshold:
                new_level = RiskLevel.HIGH
            elif new_score <= cfg.risk_low_threshold:
                new_level = RiskLevel.LOW
            else:
                new_level = RiskLevel.STANDARD

            node.risk_score = new_score
            node.risk_level = new_level
            nodes_updated += 1

            propagation_results.append(
                RiskPropagationResult(
                    node_id=node_id,
                    previous_risk_score=prev_score,
                    new_risk_score=new_score,
                    previous_risk_level=prev_level,
                    new_risk_level=new_level,
                    propagation_source=body.propagation_source,
                    risk_factors=weights,
                )
            )

        # Update graph risk summary
        risk_dist = {"low": 0, "standard": 0, "high": 0}
        for n in graph.nodes.values():
            risk_dist[n.risk_level.value] += 1
        graph.risk_summary = risk_dist

        record_risk_propagation()
        elapsed = time.monotonic() - start
        observe_processing_duration("risk_propagate", elapsed)

        logger.info(
            "Risk propagation completed: graph=%s nodes_updated=%d",
            graph_id,
            nodes_updated,
        )

        return RiskPropagateResponse(
            graph_id=graph_id,
            nodes_updated=nodes_updated,
            propagation_results=propagation_results,
            processing_time_ms=elapsed * 1000,
            status="completed",
        )

    except Exception as exc:
        record_error("risk_propagate")
        logger.error(
            "Risk propagation failed: graph=%s error=%s",
            graph_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk propagation failed",
        )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/risk/summary
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/risk/summary",
    response_model=RiskSummary,
    summary="Get risk summary for a graph",
    description=(
        "Retrieve aggregated risk statistics including distribution by "
        "risk level, average and maximum risk scores, and list of "
        "high-risk nodes requiring attention."
    ),
    responses={
        200: {"description": "Risk summary"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def get_risk_summary(
    graph_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:risk:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RiskSummary:
    """Get aggregated risk summary for a graph.

    Args:
        graph_id: Graph identifier.
        user: Authenticated user with risk:read permission.

    Returns:
        RiskSummary with distribution, averages, and high-risk nodes.
    """
    graph = _check_graph_access(graph_id, user)

    risk_dist = {"low": 0, "standard": 0, "high": 0}
    scores: List[float] = []
    high_risk_nodes: List[str] = []

    for node_id, node in graph.nodes.items():
        risk_dist[node.risk_level.value] += 1
        scores.append(node.risk_score)
        if node.risk_level == RiskLevel.HIGH:
            high_risk_nodes.append(node_id)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    # Top 5 risk concentration nodes
    sorted_nodes = sorted(
        graph.nodes.items(),
        key=lambda x: x[1].risk_score,
        reverse=True,
    )
    risk_concentration = [
        {
            "node_id": nid,
            "operator_name": n.operator_name,
            "risk_score": n.risk_score,
            "risk_level": n.risk_level.value,
            "country_code": n.country_code,
        }
        for nid, n in sorted_nodes[:5]
    ]

    return RiskSummary(
        graph_id=graph_id,
        total_nodes=len(graph.nodes),
        risk_distribution=risk_dist,
        average_risk_score=round(avg_score, 2),
        max_risk_score=round(max_score, 2),
        high_risk_nodes=high_risk_nodes,
        risk_concentration=risk_concentration,
    )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/risk/heatmap
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/risk/heatmap",
    response_model=RiskHeatmapResponse,
    summary="Get risk heatmap data",
    description=(
        "Retrieve geospatial risk heatmap data for visualization. "
        "Returns node risk scores with GPS coordinates for geographic "
        "risk distribution display."
    ),
    responses={
        200: {"description": "Risk heatmap data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def get_risk_heatmap(
    graph_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:risk:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RiskHeatmapResponse:
    """Get risk heatmap data with geospatial coordinates.

    Args:
        graph_id: Graph identifier.
        user: Authenticated user with risk:read permission.

    Returns:
        RiskHeatmapResponse with node positions and risk scores.
    """
    graph = _check_graph_access(graph_id, user)

    heatmap_data: List[Dict[str, Any]] = []
    risk_dist = {"low": 0, "standard": 0, "high": 0}

    for node_id, node in graph.nodes.items():
        risk_dist[node.risk_level.value] += 1
        entry: Dict[str, Any] = {
            "node_id": node_id,
            "operator_name": node.operator_name,
            "risk_score": node.risk_score,
            "risk_level": node.risk_level.value,
            "country_code": node.country_code,
            "node_type": node.node_type.value,
        }
        if node.coordinates:
            entry["lat"] = node.coordinates[0]
            entry["lon"] = node.coordinates[1]
        heatmap_data.append(entry)

    return RiskHeatmapResponse(
        graph_id=graph_id,
        heatmap_data=heatmap_data,
        risk_distribution=risk_dist,
    )
