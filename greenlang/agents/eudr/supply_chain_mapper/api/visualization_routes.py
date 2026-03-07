# -*- coding: utf-8 -*-
"""
Visualization Routes - AGENT-EUDR-001 Supply Chain Mapper API

Endpoints for generating graph layout data and Sankey diagram data
for frontend rendering (D3.js, vis-network, Cytoscape.js).

Endpoints:
    GET /graphs/{graph_id}/layout  - Get graph layout positions
    GET /graphs/{graph_id}/sankey  - Get Sankey flow diagram data

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001, Section 7.5 (Feature 7)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
    AuthUser,
    ErrorResponse,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.supply_chain_mapper.api.graph_routes import (
    _get_graph_store,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    GraphLayoutData,
    RiskLevel,
    SankeyData,
    SupplyChainGraph,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Visualization"])

# Node type color mapping for visualization
NODE_COLORS: Dict[str, str] = {
    "producer": "#4CAF50",
    "collector": "#2196F3",
    "processor": "#FF9800",
    "trader": "#9C27B0",
    "importer": "#F44336",
    "certifier": "#00BCD4",
    "warehouse": "#795548",
    "port": "#607D8B",
}

# Risk level colors
RISK_COLORS: Dict[str, str] = {
    "low": "#4CAF50",
    "standard": "#FFC107",
    "high": "#F44336",
}


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
# GET /graphs/{graph_id}/layout
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/layout",
    response_model=GraphLayoutData,
    summary="Get graph layout positions",
    description=(
        "Generate node positions and edge paths for rendering the "
        "supply chain graph. Supports force-directed and hierarchical "
        "layout algorithms."
    ),
    responses={
        200: {"description": "Graph layout data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def get_layout(
    graph_id: str,
    request: Request,
    algorithm: str = Query(
        default="hierarchical",
        description="Layout algorithm: force_directed, hierarchical, radial",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:visualization:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> GraphLayoutData:
    """Generate graph layout positions for visualization.

    Args:
        graph_id: Graph identifier.
        algorithm: Layout algorithm to use.
        user: Authenticated user with visualization:read permission.

    Returns:
        GraphLayoutData with node positions, edge paths, and styles.
    """
    graph = _check_graph_access(graph_id, user)

    node_positions: Dict[str, Tuple[float, float]] = {}
    node_styles: Dict[str, Dict[str, Any]] = {}
    edge_paths: Dict[str, List[Tuple[float, float]]] = {}
    edge_styles: Dict[str, Dict[str, Any]] = {}

    if algorithm == "hierarchical":
        # Arrange nodes by tier depth (left-to-right, top-to-bottom)
        tier_groups: Dict[int, List[str]] = {}
        for node_id, node in graph.nodes.items():
            tier = node.tier_depth
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append(node_id)

        x_spacing = 200.0
        y_spacing = 120.0

        for tier, node_ids in sorted(tier_groups.items()):
            x = tier * x_spacing
            for i, node_id in enumerate(node_ids):
                y = i * y_spacing
                node_positions[node_id] = (x, y)
    else:
        # Simple circular/force-directed layout
        n = len(graph.nodes)
        radius = max(200.0, n * 30.0)
        center_x, center_y = 500.0, 500.0

        for i, node_id in enumerate(graph.nodes):
            angle = 2 * math.pi * i / max(n, 1)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            node_positions[node_id] = (round(x, 2), round(y, 2))

    # Generate node styles
    for node_id, node in graph.nodes.items():
        node_styles[node_id] = {
            "color": NODE_COLORS.get(node.node_type.value, "#9E9E9E"),
            "border_color": RISK_COLORS.get(
                node.risk_level.value, "#FFC107"
            ),
            "size": 30 if node.node_type.value == "importer" else 20,
            "shape": "diamond" if node.node_type.value == "producer" else "circle",
            "label": node.operator_name[:30],
            "tooltip": (
                f"{node.operator_name}\n"
                f"Type: {node.node_type.value}\n"
                f"Country: {node.country_code}\n"
                f"Risk: {node.risk_level.value} ({node.risk_score:.1f})"
            ),
        }

    # Generate edge paths (straight lines between node positions)
    for edge_id, edge in graph.edges.items():
        source_pos = node_positions.get(edge.source_node_id)
        target_pos = node_positions.get(edge.target_node_id)
        if source_pos and target_pos:
            edge_paths[edge_id] = [source_pos, target_pos]
            edge_styles[edge_id] = {
                "color": "#757575",
                "width": min(5.0, float(edge.quantity) / 1000.0 + 1.0),
                "label": f"{edge.quantity} {edge.unit}",
                "dashed": edge.custody_model.value == "mass_balance",
            }

    # Compute viewport
    if node_positions:
        all_x = [p[0] for p in node_positions.values()]
        all_y = [p[1] for p in node_positions.values()]
        viewport = {
            "min_x": min(all_x) - 50,
            "min_y": min(all_y) - 50,
            "max_x": max(all_x) + 50,
            "max_y": max(all_y) + 50,
        }
    else:
        viewport = {"min_x": 0.0, "min_y": 0.0, "max_x": 1000.0, "max_y": 1000.0}

    return GraphLayoutData(
        graph_id=graph_id,
        layout_algorithm=algorithm,
        node_positions=node_positions,
        edge_paths=edge_paths,
        node_styles=node_styles,
        edge_styles=edge_styles,
        viewport=viewport,
    )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/sankey
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/sankey",
    response_model=SankeyData,
    summary="Get Sankey flow diagram data",
    description=(
        "Generate Sankey diagram data showing commodity flow volumes "
        "through the supply chain. Nodes are grouped by tier depth "
        "and connected by weighted links."
    ),
    responses={
        200: {"description": "Sankey diagram data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def get_sankey(
    graph_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:visualization:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SankeyData:
    """Generate Sankey diagram data for commodity flow visualization.

    Args:
        graph_id: Graph identifier.
        user: Authenticated user with visualization:read permission.

    Returns:
        SankeyData with nodes and weighted links.
    """
    graph = _check_graph_access(graph_id, user)

    # Build node index for Sankey
    node_index: Dict[str, int] = {}
    sankey_nodes: List[Dict[str, Any]] = []

    for i, (node_id, node) in enumerate(graph.nodes.items()):
        node_index[node_id] = i
        sankey_nodes.append(
            {
                "id": i,
                "node_id": node_id,
                "name": node.operator_name[:40],
                "node_type": node.node_type.value,
                "tier_depth": node.tier_depth,
                "country_code": node.country_code,
                "color": NODE_COLORS.get(node.node_type.value, "#9E9E9E"),
            }
        )

    # Build links from edges
    sankey_links: List[Dict[str, Any]] = []
    for edge in graph.edges.values():
        source_idx = node_index.get(edge.source_node_id)
        target_idx = node_index.get(edge.target_node_id)
        if source_idx is not None and target_idx is not None:
            sankey_links.append(
                {
                    "source": source_idx,
                    "target": target_idx,
                    "value": float(edge.quantity),
                    "commodity": edge.commodity.value,
                    "unit": edge.unit,
                    "custody_model": edge.custody_model.value,
                }
            )

    return SankeyData(
        graph_id=graph_id,
        nodes=sankey_nodes,
        links=sankey_links,
    )
