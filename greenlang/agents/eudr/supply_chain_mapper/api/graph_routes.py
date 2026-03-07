# -*- coding: utf-8 -*-
"""
Graph CRUD Routes - AGENT-EUDR-001 Supply Chain Mapper API

Endpoints for creating, listing, retrieving, deleting, and exporting
supply chain graphs. Each graph represents a single operator's view
of one EUDR commodity supply chain.

Endpoints:
    POST   /graphs              - Create a new supply chain graph
    GET    /graphs              - List graphs with pagination and filters
    GET    /graphs/{graph_id}   - Get full graph details by ID
    DELETE /graphs/{graph_id}   - Delete a graph
    GET    /graphs/{graph_id}/export - Export graph as DDS data

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001, Section 7.5
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_current_user,
    get_pagination,
    rate_limit_export,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.supply_chain_mapper.api.schemas import (
    GraphCreateRequest,
    GraphCreateResponse,
    GraphDeleteResponse,
    GraphListResponse,
    GraphSummary,
    HealthResponse,
    PaginatedMeta,
)
from greenlang.agents.eudr.supply_chain_mapper.metrics import (
    observe_processing_duration,
    record_dds_export,
    record_error,
    record_graph_created,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    DDSExportData,
    EUDRCommodity,
    SupplyChainGraph,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Supply Chain Graphs"])

# ---------------------------------------------------------------------------
# In-memory graph store (replaced by database in production)
# ---------------------------------------------------------------------------

_graph_store: dict[str, SupplyChainGraph] = {}


def _get_graph_store() -> dict[str, SupplyChainGraph]:
    """Return the graph store singleton. Replaceable for testing."""
    return _graph_store


# ---------------------------------------------------------------------------
# POST /graphs
# ---------------------------------------------------------------------------


@router.post(
    "/graphs",
    response_model=GraphCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new supply chain graph",
    description=(
        "Create a new supply chain graph for a specific EUDR commodity. "
        "The graph is owned by the authenticated user's operator."
    ),
    responses={
        201: {"description": "Graph created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_graph(
    request: Request,
    body: GraphCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:graphs:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> GraphCreateResponse:
    """Create a new supply chain graph for an EUDR commodity.

    Args:
        body: Graph creation parameters (commodity, optional name).
        user: Authenticated user with graphs:create permission.

    Returns:
        GraphCreateResponse with the new graph ID and metadata.
    """
    start = time.monotonic()
    try:
        graph_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        graph = SupplyChainGraph(
            graph_id=graph_id,
            operator_id=user.operator_id or user.user_id,
            commodity=body.commodity,
            graph_name=body.graph_name,
            created_at=now,
            updated_at=now,
        )

        store = _get_graph_store()
        store[graph_id] = graph

        record_graph_created()
        elapsed = time.monotonic() - start
        observe_processing_duration("graph_create", elapsed)

        logger.info(
            "Graph created: id=%s commodity=%s operator=%s",
            graph_id,
            body.commodity.value,
            user.operator_id or user.user_id,
        )

        return GraphCreateResponse(
            graph_id=graph_id,
            operator_id=graph.operator_id,
            commodity=body.commodity,
            graph_name=body.graph_name,
            status="created",
            created_at=now,
        )

    except Exception as exc:
        record_error("graph_create")
        logger.error("Failed to create graph: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create supply chain graph",
        )


# ---------------------------------------------------------------------------
# GET /graphs
# ---------------------------------------------------------------------------


@router.get(
    "/graphs",
    response_model=GraphListResponse,
    summary="List supply chain graphs",
    description=(
        "List all supply chain graphs accessible to the authenticated user, "
        "with optional commodity filter and pagination."
    ),
    responses={
        200: {"description": "List of graphs"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_graphs(
    request: Request,
    commodity: Optional[EUDRCommodity] = Query(
        None, description="Filter by EUDR commodity"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:graphs:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> GraphListResponse:
    """List supply chain graphs with pagination and optional commodity filter.

    Args:
        commodity: Optional commodity filter.
        pagination: Limit and offset for pagination.
        user: Authenticated user with graphs:read permission.

    Returns:
        Paginated list of graph summaries.
    """
    store = _get_graph_store()
    operator_id = user.operator_id or user.user_id

    # Filter graphs by operator and optional commodity
    all_graphs = [
        g for g in store.values()
        if g.operator_id == operator_id
        and (commodity is None or g.commodity == commodity)
    ]

    total = len(all_graphs)
    page = all_graphs[pagination.offset: pagination.offset + pagination.limit]

    summaries = [
        GraphSummary(
            graph_id=g.graph_id,
            operator_id=g.operator_id,
            commodity=g.commodity,
            graph_name=g.graph_name,
            total_nodes=g.total_nodes,
            total_edges=g.total_edges,
            max_tier_depth=g.max_tier_depth,
            traceability_score=g.traceability_score,
            compliance_readiness=g.compliance_readiness,
            version=g.version,
            created_at=g.created_at,
            updated_at=g.updated_at,
        )
        for g in page
    ]

    return GraphListResponse(
        graphs=summaries,
        meta=PaginatedMeta(
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
            has_more=(pagination.offset + pagination.limit) < total,
        ),
    )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}",
    response_model=SupplyChainGraph,
    summary="Get graph details",
    description="Retrieve full supply chain graph details including nodes, edges, and gaps.",
    responses={
        200: {"description": "Graph details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def get_graph(
    graph_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:graphs:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SupplyChainGraph:
    """Get full supply chain graph by ID.

    Args:
        graph_id: Unique graph identifier.
        user: Authenticated user with graphs:read permission.

    Returns:
        Complete SupplyChainGraph with all nodes, edges, and gaps.

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

    return graph


# ---------------------------------------------------------------------------
# DELETE /graphs/{graph_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/graphs/{graph_id}",
    response_model=GraphDeleteResponse,
    summary="Delete a supply chain graph",
    description="Permanently delete a supply chain graph and all associated data.",
    responses={
        200: {"description": "Graph deleted"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def delete_graph(
    graph_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:graphs:delete")
    ),
    _rate: None = Depends(rate_limit_write),
) -> GraphDeleteResponse:
    """Delete a supply chain graph.

    Args:
        graph_id: Unique graph identifier.
        user: Authenticated user with graphs:delete permission.

    Returns:
        Confirmation of deletion.

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
            detail="Not authorized to delete this graph",
        )

    del store[graph_id]

    logger.info("Graph deleted: id=%s operator=%s", graph_id, operator_id)

    return GraphDeleteResponse(
        graph_id=graph_id,
        status="deleted",
        deleted_at=datetime.now(timezone.utc).replace(microsecond=0),
    )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/export
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/export",
    response_model=DDSExportData,
    summary="Export graph as DDS data",
    description=(
        "Export supply chain graph data formatted for Due Diligence "
        "Statement (DDS) inclusion per EUDR Article 4(2)(f)."
    ),
    responses={
        200: {"description": "DDS export data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def export_graph(
    graph_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:graphs:export")
    ),
    _rate: None = Depends(rate_limit_export),
) -> DDSExportData:
    """Export graph data for DDS filing.

    Args:
        graph_id: Unique graph identifier.
        user: Authenticated user with graphs:export permission.

    Returns:
        DDSExportData formatted for Due Diligence Statement.

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
            detail="Not authorized to export this graph",
        )

    # Collect origin countries from producer nodes
    origin_countries = list(
        {
            n.country_code
            for n in graph.nodes.values()
            if n.node_type.value == "producer"
        }
    )

    # Count origin plots
    origin_plots = sum(
        len(n.plot_ids)
        for n in graph.nodes.values()
        if n.node_type.value == "producer"
    )

    # Determine overall risk level from max node risk
    max_risk = max(
        (n.risk_score for n in graph.nodes.values()), default=0.0
    )
    from greenlang.agents.eudr.supply_chain_mapper.models import RiskLevel

    if max_risk >= 70:
        overall_risk = RiskLevel.HIGH
    elif max_risk <= 30:
        overall_risk = RiskLevel.LOW
    else:
        overall_risk = RiskLevel.STANDARD

    export_data = DDSExportData(
        graph_id=graph.graph_id,
        operator_id=graph.operator_id,
        commodity=graph.commodity,
        total_supply_chain_actors=graph.total_nodes,
        tier_depth=graph.max_tier_depth,
        traceability_score=graph.traceability_score,
        origin_countries=origin_countries,
        origin_plot_count=origin_plots,
        custody_transfers_count=graph.total_edges,
        risk_level=overall_risk,
        compliance_readiness=graph.compliance_readiness,
        supply_chain_summary={
            "total_producers": sum(
                1
                for n in graph.nodes.values()
                if n.node_type.value == "producer"
            ),
            "total_processors": sum(
                1
                for n in graph.nodes.values()
                if n.node_type.value == "processor"
            ),
            "total_traders": sum(
                1
                for n in graph.nodes.values()
                if n.node_type.value == "trader"
            ),
        },
    )

    record_dds_export()
    elapsed = time.monotonic() - start
    observe_processing_duration("dds_export", elapsed)

    logger.info("Graph exported for DDS: id=%s", graph_id)

    return export_data


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check EUDR Supply Chain Mapper API health.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse()
