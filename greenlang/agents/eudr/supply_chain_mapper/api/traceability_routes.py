# -*- coding: utf-8 -*-
"""
Traceability Routes - AGENT-EUDR-001 Supply Chain Mapper API

Endpoints for forward/backward trace operations and batch-level
traceability queries through the supply chain graph.

Endpoints:
    GET /graphs/{graph_id}/trace/forward/{node_id}   - Forward trace
    GET /graphs/{graph_id}/trace/backward/{node_id}  - Backward trace
    GET /graphs/{graph_id}/trace/batch/{batch_id}    - Batch trace

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001, Section 7.5 (Feature 3 & 4)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from decimal import Decimal
from typing import List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

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
    BatchTraceResponse,
)
from greenlang.agents.eudr.supply_chain_mapper.metrics import (
    observe_processing_duration,
    record_error,
    record_trace_operation,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    SupplyChainGraph,
    TraceResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Traceability"])


def _check_graph_access(
    graph_id: str, user: AuthUser
) -> SupplyChainGraph:
    """Validate graph exists and user has access.

    Args:
        graph_id: Graph identifier.
        user: Authenticated user.

    Returns:
        The SupplyChainGraph if accessible.

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized.
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

    return graph


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/trace/forward/{node_id}
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/trace/forward/{node_id}",
    response_model=TraceResult,
    summary="Forward trace from a node",
    description=(
        "Trace forward (downstream) from a node to discover all downstream "
        "actors, edges, and eventual importers/consumers. Returns visited "
        "nodes, edges, trace depth, and completeness status."
    ),
    responses={
        200: {"description": "Forward trace result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph or node not found"},
    },
)
async def trace_forward(
    graph_id: str,
    node_id: str,
    request: Request,
    max_depth: int = Query(
        default=50, ge=1, le=100, description="Maximum trace depth"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:trace:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TraceResult:
    """Perform forward trace from a node through downstream edges.

    Args:
        graph_id: Graph identifier.
        node_id: Starting node for the trace.
        max_depth: Maximum depth to trace.
        user: Authenticated user with trace:read permission.

    Returns:
        TraceResult with visited nodes, edges, and completeness.
    """
    start = time.monotonic()
    graph = _check_graph_access(graph_id, user)

    if node_id not in graph.nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} not found in graph {graph_id}",
        )

    # BFS forward traversal (source -> target direction)
    visited_nodes: List[str] = []
    visited_edges: List[str] = []
    broken_at: List[str] = []
    queue: deque = deque([(node_id, 0)])
    seen: Set[str] = {node_id}
    max_reached = 0

    while queue:
        current, depth = queue.popleft()
        visited_nodes.append(current)
        max_reached = max(max_reached, depth)

        if depth >= max_depth:
            continue

        # Find outgoing edges from current node
        outgoing = [
            e
            for e in graph.edges.values()
            if e.source_node_id == current
        ]

        if not outgoing and depth > 0:
            # Leaf node with no outgoing edges
            pass

        for edge in outgoing:
            visited_edges.append(edge.edge_id)
            target = edge.target_node_id
            if target not in seen:
                if target in graph.nodes:
                    seen.add(target)
                    queue.append((target, depth + 1))
                else:
                    broken_at.append(target)

    elapsed = time.monotonic() - start
    record_trace_operation("forward")
    observe_processing_duration("trace_forward", elapsed)

    return TraceResult(
        trace_id=str(uuid.uuid4()),
        direction="forward",
        start_node_id=node_id,
        visited_nodes=visited_nodes,
        visited_edges=visited_edges,
        origin_plot_ids=[],
        trace_depth=max_reached,
        is_complete=len(broken_at) == 0,
        broken_at=broken_at,
        processing_time_ms=elapsed * 1000,
    )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/trace/backward/{node_id}
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/trace/backward/{node_id}",
    response_model=TraceResult,
    summary="Backward trace from a node",
    description=(
        "Trace backward (upstream) from a node to discover all upstream "
        "suppliers, producers, and origin production plots. Essential for "
        "EUDR compliance verification of plot-to-product traceability."
    ),
    responses={
        200: {"description": "Backward trace result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph or node not found"},
    },
)
async def trace_backward(
    graph_id: str,
    node_id: str,
    request: Request,
    max_depth: int = Query(
        default=50, ge=1, le=100, description="Maximum trace depth"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:trace:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TraceResult:
    """Perform backward trace from a node through upstream edges.

    Args:
        graph_id: Graph identifier.
        node_id: Starting node for the backward trace.
        max_depth: Maximum depth to trace.
        user: Authenticated user with trace:read permission.

    Returns:
        TraceResult with visited nodes, edges, origin plot IDs, and completeness.
    """
    start = time.monotonic()
    graph = _check_graph_access(graph_id, user)

    if node_id not in graph.nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} not found in graph {graph_id}",
        )

    # BFS backward traversal (target -> source direction)
    visited_nodes: List[str] = []
    visited_edges: List[str] = []
    origin_plot_ids: List[str] = []
    broken_at: List[str] = []
    total_quantity = Decimal("0")
    queue: deque = deque([(node_id, 0)])
    seen: Set[str] = {node_id}
    max_reached = 0

    while queue:
        current, depth = queue.popleft()
        visited_nodes.append(current)
        max_reached = max(max_reached, depth)

        # Collect plot IDs from producer nodes
        node = graph.nodes.get(current)
        if node and node.node_type.value == "producer" and node.plot_ids:
            origin_plot_ids.extend(node.plot_ids)

        if depth >= max_depth:
            continue

        # Find incoming edges to current node
        incoming = [
            e
            for e in graph.edges.values()
            if e.target_node_id == current
        ]

        for edge in incoming:
            visited_edges.append(edge.edge_id)
            total_quantity += edge.quantity
            source = edge.source_node_id
            if source not in seen:
                if source in graph.nodes:
                    seen.add(source)
                    queue.append((source, depth + 1))
                else:
                    broken_at.append(source)

    elapsed = time.monotonic() - start
    record_trace_operation("backward")
    observe_processing_duration("trace_backward", elapsed)

    return TraceResult(
        trace_id=str(uuid.uuid4()),
        direction="backward",
        start_node_id=node_id,
        visited_nodes=visited_nodes,
        visited_edges=visited_edges,
        origin_plot_ids=list(set(origin_plot_ids)),
        trace_depth=max_reached,
        total_quantity=total_quantity if total_quantity > 0 else None,
        is_complete=len(broken_at) == 0,
        broken_at=broken_at,
        processing_time_ms=elapsed * 1000,
    )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/trace/batch/{batch_id}
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/trace/batch/{batch_id}",
    response_model=BatchTraceResponse,
    summary="Trace a batch through the supply chain",
    description=(
        "Find all edges and nodes associated with a specific batch/lot "
        "number, tracing the commodity flow from origin to destination."
    ),
    responses={
        200: {"description": "Batch trace result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def trace_batch(
    graph_id: str,
    batch_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:trace:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchTraceResponse:
    """Trace a batch/lot through the supply chain graph.

    Args:
        graph_id: Graph identifier.
        batch_id: Batch/lot number to trace.
        user: Authenticated user with trace:read permission.

    Returns:
        BatchTraceResponse with edges, origin/destination nodes, and quantity.
    """
    start = time.monotonic()
    graph = _check_graph_access(graph_id, user)

    # Find all edges with matching batch number
    batch_edges = [
        e
        for e in graph.edges.values()
        if e.batch_number == batch_id
    ]

    if not batch_edges:
        return BatchTraceResponse(
            batch_id=batch_id,
            graph_id=graph_id,
            edges=[],
            origin_nodes=[],
            destination_nodes=[],
            total_quantity=None,
            is_complete=False,
        )

    # Collect source and target nodes
    source_ids = {e.source_node_id for e in batch_edges}
    target_ids = {e.target_node_id for e in batch_edges}
    origin_nodes = list(source_ids - target_ids)
    destination_nodes = list(target_ids - source_ids)

    # Total quantity
    total_qty = sum((e.quantity for e in batch_edges), Decimal("0"))

    # Determine custody model (all should be same for a batch)
    custody = batch_edges[0].custody_model.value if batch_edges else None

    elapsed = time.monotonic() - start
    record_trace_operation("forward")
    observe_processing_duration("batch_trace", elapsed)

    return BatchTraceResponse(
        batch_id=batch_id,
        graph_id=graph_id,
        edges=[
            {
                "edge_id": e.edge_id,
                "source_node_id": e.source_node_id,
                "target_node_id": e.target_node_id,
                "commodity": e.commodity.value,
                "quantity": str(e.quantity),
                "unit": e.unit,
                "transfer_date": e.transfer_date.isoformat(),
            }
            for e in batch_edges
        ],
        origin_nodes=origin_nodes,
        destination_nodes=destination_nodes,
        total_quantity=str(total_qty),
        custody_model=custody,
        is_complete=True,
    )
