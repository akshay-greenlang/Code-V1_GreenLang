# -*- coding: utf-8 -*-
"""
Gap Analysis Routes - AGENT-EUDR-001 Supply Chain Mapper API

Endpoints for triggering gap analysis, listing detected gaps, and
resolving individual compliance gaps in supply chain graphs.

Endpoints:
    POST /graphs/{graph_id}/gaps/analyze       - Run gap analysis
    GET  /graphs/{graph_id}/gaps               - List gaps with filters
    PUT  /graphs/{graph_id}/gaps/{gap_id}/resolve - Resolve a gap

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001, Section 7.5 (Feature 6)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.supply_chain_mapper.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.supply_chain_mapper.api.graph_routes import (
    _get_graph_store,
)
from greenlang.agents.eudr.supply_chain_mapper.api.schemas import (
    GapAnalyzeRequest,
    GapListResponse,
    GapResolveRequest,
    GapResolveResponse,
    PaginatedMeta,
)
from greenlang.agents.eudr.supply_chain_mapper.metrics import (
    observe_processing_duration,
    record_error,
    record_gap_detected,
    record_gap_resolved,
)
from greenlang.agents.eudr.supply_chain_mapper.models import (
    GAP_ARTICLE_MAP,
    GAP_SEVERITY_MAP,
    GapAnalysisResult,
    GapSeverity,
    GapType,
    SupplyChainGap,
    SupplyChainGraph,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Gap Analysis"])


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
# POST /graphs/{graph_id}/gaps/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/graphs/{graph_id}/gaps/analyze",
    response_model=GapAnalysisResult,
    summary="Run gap analysis on a graph",
    description=(
        "Execute comprehensive gap analysis across the supply chain "
        "graph to detect compliance gaps including missing geolocation, "
        "missing polygons, broken custody chains, unverified actors, "
        "missing tiers, mass balance discrepancies, and more. "
        "Each gap type maps to a specific EUDR article."
    ),
    responses={
        200: {"description": "Gap analysis result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def analyze_gaps(
    graph_id: str,
    body: GapAnalyzeRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:gaps:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> GapAnalysisResult:
    """Run gap analysis across the supply chain graph.

    Detects 10 gap types per EUDR articles 4, 9, 10, and 31.

    Args:
        graph_id: Target graph identifier.
        body: Gap analysis parameters.
        user: Authenticated user with gaps:write permission.

    Returns:
        GapAnalysisResult with detected gaps and remediation priorities.
    """
    start = time.monotonic()
    graph = _check_graph_access(graph_id, user)

    try:
        detected_gaps: List[SupplyChainGap] = []

        # Detect missing geolocation for producer nodes
        for node_id, node in graph.nodes.items():
            if node.node_type.value == "producer":
                if not node.coordinates:
                    gap = SupplyChainGap(
                        gap_id=str(uuid.uuid4()),
                        gap_type=GapType.MISSING_GEOLOCATION,
                        severity=GapSeverity.CRITICAL,
                        affected_node_id=node_id,
                        description=(
                            f"Producer node '{node.operator_name}' "
                            f"lacks GPS coordinates (EUDR Article 9)"
                        ),
                        remediation=(
                            "Add GPS coordinates via satellite imagery "
                            "or field survey"
                        ),
                        eudr_article="Article 9",
                    )
                    detected_gaps.append(gap)

                # Check for missing plot IDs
                if not node.plot_ids:
                    gap = SupplyChainGap(
                        gap_id=str(uuid.uuid4()),
                        gap_type=GapType.MISSING_POLYGON,
                        severity=GapSeverity.CRITICAL,
                        affected_node_id=node_id,
                        description=(
                            f"Producer node '{node.operator_name}' "
                            f"has no linked production plots"
                        ),
                        remediation=(
                            "Link production plot IDs with polygon "
                            "boundary data per Article 9(1)(d)"
                        ),
                        eudr_article="Article 9(1)(d)",
                    )
                    detected_gaps.append(gap)

            # Check for unverified actors
            if node.compliance_status.value == "pending_verification":
                gap = SupplyChainGap(
                    gap_id=str(uuid.uuid4()),
                    gap_type=GapType.UNVERIFIED_ACTOR,
                    severity=GapSeverity.HIGH,
                    affected_node_id=node_id,
                    description=(
                        f"Node '{node.operator_name}' has not been "
                        f"verified for EUDR compliance"
                    ),
                    remediation=(
                        "Complete identity verification and compliance "
                        "documentation per Article 10"
                    ),
                    eudr_article="Article 10",
                )
                detected_gaps.append(gap)

        # Check for orphan nodes (no edges)
        source_ids = {e.source_node_id for e in graph.edges.values()}
        target_ids = {e.target_node_id for e in graph.edges.values()}
        connected = source_ids | target_ids

        for node_id, node in graph.nodes.items():
            if node_id not in connected and len(graph.nodes) > 1:
                gap = SupplyChainGap(
                    gap_id=str(uuid.uuid4()),
                    gap_type=GapType.ORPHAN_NODE,
                    severity=GapSeverity.LOW,
                    affected_node_id=node_id,
                    description=(
                        f"Node '{node.operator_name}' has no connections "
                        f"(orphan node)"
                    ),
                    remediation=(
                        "Connect this node to the supply chain or "
                        "remove if not relevant"
                    ),
                    eudr_article="Internal",
                )
                detected_gaps.append(gap)

        # Filter by severity if requested
        if body.severity_filter:
            severity_order = {
                GapSeverity.CRITICAL: 0,
                GapSeverity.HIGH: 1,
                GapSeverity.MEDIUM: 2,
                GapSeverity.LOW: 3,
            }
            filter_level = severity_order.get(body.severity_filter, 3)
            detected_gaps = [
                g
                for g in detected_gaps
                if severity_order.get(g.severity, 3) <= filter_level
            ]

        # Filter out resolved gaps unless requested
        if not body.include_resolved:
            detected_gaps = [g for g in detected_gaps if not g.is_resolved]

        # Update graph gaps
        graph.gaps = detected_gaps

        # Calculate compliance readiness
        if graph.nodes:
            total_checks = len(graph.nodes) * 3  # 3 checks per node
            gaps_count = len(detected_gaps)
            readiness = max(
                0.0, 100.0 * (1 - gaps_count / max(total_checks, 1))
            )
            graph.compliance_readiness = round(readiness, 2)

        # Count by severity
        gaps_by_severity: Dict[str, int] = {
            "critical": 0, "high": 0, "medium": 0, "low": 0
        }
        gaps_by_type: Dict[str, int] = {}
        for g in detected_gaps:
            gaps_by_severity[g.severity.value] += 1
            gaps_by_type[g.gap_type.value] = (
                gaps_by_type.get(g.gap_type.value, 0) + 1
            )
            record_gap_detected(g.gap_type.value, g.severity.value)

        # Prioritize remediation: critical first, then high, etc.
        priority_sorted = sorted(
            detected_gaps,
            key=lambda g: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    g.severity.value, 4
                )
            ),
        )

        elapsed = time.monotonic() - start
        observe_processing_duration("gap_analyze", elapsed)

        logger.info(
            "Gap analysis completed: graph=%s gaps=%d readiness=%.1f%%",
            graph_id,
            len(detected_gaps),
            graph.compliance_readiness,
        )

        return GapAnalysisResult(
            graph_id=graph_id,
            total_gaps=len(detected_gaps),
            gaps_by_severity=gaps_by_severity,
            gaps_by_type=gaps_by_type,
            compliance_readiness=graph.compliance_readiness,
            gaps=detected_gaps,
            remediation_priority=[g.gap_id for g in priority_sorted],
        )

    except HTTPException:
        raise
    except Exception as exc:
        record_error("gap_analyze")
        logger.error(
            "Gap analysis failed: graph=%s error=%s",
            graph_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gap analysis failed",
        )


# ---------------------------------------------------------------------------
# GET /graphs/{graph_id}/gaps
# ---------------------------------------------------------------------------


@router.get(
    "/graphs/{graph_id}/gaps",
    response_model=GapListResponse,
    summary="List gaps for a graph",
    description=(
        "List compliance gaps detected in the supply chain graph, "
        "with optional filters by severity, type, and resolution status."
    ),
    responses={
        200: {"description": "Paginated gap list"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Graph not found"},
    },
)
async def list_gaps(
    graph_id: str,
    request: Request,
    severity: Optional[GapSeverity] = Query(
        None, description="Filter by severity"
    ),
    gap_type: Optional[GapType] = Query(
        None, description="Filter by gap type"
    ),
    is_resolved: Optional[bool] = Query(
        None, description="Filter by resolution status"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:gaps:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> GapListResponse:
    """List compliance gaps with filters and pagination.

    Args:
        graph_id: Graph identifier.
        severity: Optional severity filter.
        gap_type: Optional gap type filter.
        is_resolved: Optional resolution status filter.
        pagination: Pagination parameters.
        user: Authenticated user with gaps:read permission.

    Returns:
        Paginated list of gaps.
    """
    graph = _check_graph_access(graph_id, user)

    # Apply filters
    gaps = list(graph.gaps)

    if severity is not None:
        gaps = [g for g in gaps if g.severity == severity]

    if gap_type is not None:
        gaps = [g for g in gaps if g.gap_type == gap_type]

    if is_resolved is not None:
        gaps = [g for g in gaps if g.is_resolved == is_resolved]

    total = len(gaps)
    page = gaps[pagination.offset: pagination.offset + pagination.limit]

    return GapListResponse(
        gaps=page,
        meta=PaginatedMeta(
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
            has_more=(pagination.offset + pagination.limit) < total,
        ),
    )


# ---------------------------------------------------------------------------
# PUT /graphs/{graph_id}/gaps/{gap_id}/resolve
# ---------------------------------------------------------------------------


@router.put(
    "/graphs/{graph_id}/gaps/{gap_id}/resolve",
    response_model=GapResolveResponse,
    summary="Resolve a compliance gap",
    description=(
        "Mark a specific compliance gap as resolved with resolution notes "
        "and optional evidence references. Updates compliance readiness score."
    ),
    responses={
        200: {"description": "Gap resolved"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Gap not found"},
    },
)
async def resolve_gap(
    graph_id: str,
    gap_id: str,
    body: GapResolveRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-supply-chain:gaps:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> GapResolveResponse:
    """Resolve a compliance gap with resolution notes and evidence.

    Args:
        graph_id: Graph identifier.
        gap_id: Gap identifier to resolve.
        body: Resolution details (notes, evidence IDs).
        user: Authenticated user with gaps:write permission.

    Returns:
        GapResolveResponse confirming resolution and updated readiness.
    """
    graph = _check_graph_access(graph_id, user)

    # Find the gap
    target_gap: Optional[SupplyChainGap] = None
    for gap in graph.gaps:
        if gap.gap_id == gap_id:
            target_gap = gap
            break

    if target_gap is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Gap {gap_id} not found in graph {graph_id}",
        )

    if target_gap.is_resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gap {gap_id} is already resolved",
        )

    # Mark as resolved
    now = datetime.now(timezone.utc).replace(microsecond=0)
    target_gap.is_resolved = True
    target_gap.resolved_at = now
    target_gap.remediation = body.resolution_notes

    # Recalculate compliance readiness
    unresolved = [g for g in graph.gaps if not g.is_resolved]
    if graph.nodes:
        total_checks = len(graph.nodes) * 3
        readiness = max(
            0.0, 100.0 * (1 - len(unresolved) / max(total_checks, 1))
        )
        graph.compliance_readiness = round(readiness, 2)

    record_gap_resolved()

    logger.info(
        "Gap resolved: gap_id=%s graph=%s by=%s",
        gap_id,
        graph_id,
        user.user_id,
    )

    return GapResolveResponse(
        gap_id=gap_id,
        graph_id=graph_id,
        status="resolved",
        resolved_at=now,
        compliance_readiness=graph.compliance_readiness,
    )
