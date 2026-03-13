# -*- coding: utf-8 -*-
"""
Authority & Analytics Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for competent authority interaction management and analytics
per EUDR Articles 14-16, 18-23.  Combines authority liaison (interactions,
inspection responses) with analytics (compliance rate, NC trends).

Endpoints (5):
    POST /authority-interactions              - Log new authority interaction
    GET  /authority-interactions              - List authority interactions
    POST /authority-inspections/respond       - Respond to inspection request
    GET  /analytics/compliance-rate           - Get compliance rate trend
    GET  /analytics/nc-trends                - Get non-conformance trends

RBAC Permissions:
    eudr-tam:authority:create   - Log new authority interactions
    eudr-tam:authority:read     - View authority interactions
    eudr-tam:authority:respond  - Submit inspection responses
    eudr-tam:analytics:read    - View analytics (compliance rates, NC trends)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_analytics_engine,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    ComplianceRatesResponse,
    ComplianceRateEntry,
    ErrorResponse,
    FindingTrendEntry,
    FindingTrendsResponse,
    InteractionCreateRequest,
    InteractionCreateResponse,
    InteractionEntry,
    InteractionListResponse,
    InteractionStatusEnum,
    InteractionTypeEnum,
    InteractionUpdateRequest,
    MetadataSchema,
    NCSeverityEnum,
    PaginatedMeta,
    ProvenanceInfo,
    ResponseSLAStatusEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authority & Analytics"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /authority-interactions
# ---------------------------------------------------------------------------


@router.post(
    "/authority-interactions",
    response_model=InteractionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log new authority interaction",
    description=(
        "Log a new competent authority interaction per EUDR Articles 14-16 "
        "and 18-23. Supports interaction types: document_request, "
        "inspection_notification, unannounced_inspection, penalty_notice, "
        "suspension_order, reinstatement, general_inquiry. Tracks SLA "
        "compliance for response deadlines."
    ),
    responses={
        201: {"description": "Authority interaction logged successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def create_authority_interaction(
    request: Request,
    body: InteractionCreateRequest,
    user: AuthUser = Depends(require_permission("eudr-tam:authority:create")),
    _rl: None = Depends(rate_limit_write),
    analytics_engine: object = Depends(get_analytics_engine),
) -> InteractionCreateResponse:
    """Log a new competent authority interaction.

    Records all interactions with EU Member State competent authorities
    including document requests, inspection notifications, and
    enforcement measures.  Automatically computes response SLA status.

    Args:
        body: Interaction creation request (operator, authority, type, dates).
        user: Authenticated user with authority:create permission.
        analytics_engine: AuditAnalyticsEngine singleton.

    Returns:
        InteractionCreateResponse with the logged interaction record.

    Raises:
        HTTPException: 400 if validation fails, 401/403 if unauthorized.
    """
    start = time.monotonic()
    try:
        logger.info(
            "Creating authority interaction: operator=%s authority=%s "
            "type=%s state=%s user=%s",
            body.operator_id,
            body.authority_name,
            body.interaction_type.value,
            body.member_state,
            user.user_id,
        )

        interaction_data = body.model_dump()
        interaction_data["created_by"] = user.user_id

        result: Dict[str, Any] = {}
        if hasattr(analytics_engine, "create_authority_interaction"):
            result = await analytics_engine.create_authority_interaction(
                operator_id=body.operator_id,
                authority_name=body.authority_name,
                member_state=body.member_state.upper(),
                interaction_type=body.interaction_type.value,
                received_date=str(body.received_date),
                response_deadline=str(body.response_deadline),
                internal_tasks=body.internal_tasks,
                created_by=user.user_id,
            )
        else:
            interaction_hash = hashlib.sha256(
                f"{body.operator_id}{body.interaction_type.value}"
                f"{time.time()}".encode()
            ).hexdigest()
            result = {
                "interaction_id": interaction_hash[:36],
                "provenance_hash": interaction_hash,
            }

        interaction = InteractionEntry(
            interaction_id=result.get("interaction_id", ""),
            operator_id=body.operator_id,
            authority_name=body.authority_name,
            member_state=body.member_state.upper(),
            interaction_type=body.interaction_type,
            received_date=body.received_date,
            response_deadline=body.response_deadline,
            internal_tasks=body.internal_tasks,
            provenance_hash=result.get("provenance_hash", ""),
        )

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(
            f"authority:{body.operator_id}:{body.interaction_type.value}",
            interaction.interaction_id,
        )

        logger.info(
            "Authority interaction created: id=%s type=%s state=%s",
            interaction.interaction_id,
            body.interaction_type.value,
            body.member_state,
        )

        return InteractionCreateResponse(
            interaction=interaction,
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create authority interaction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create authority interaction",
        )


# ---------------------------------------------------------------------------
# GET /authority-interactions
# ---------------------------------------------------------------------------


@router.get(
    "/authority-interactions",
    response_model=InteractionListResponse,
    summary="List authority interactions",
    description=(
        "Retrieve a paginated list of competent authority interactions "
        "with optional filters for member state, interaction type, and "
        "SLA status. Returns response deadline tracking information."
    ),
    responses={
        200: {"description": "Authority interactions listed successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_authority_interactions(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:authority:read")),
    _rl: None = Depends(rate_limit_standard),
    pagination: PaginationParams = Depends(get_pagination),
    analytics_engine: object = Depends(get_analytics_engine),
    member_state: Optional[str] = Query(
        None, description="Filter by EU member state (2-char ISO code)"
    ),
    interaction_type: Optional[InteractionTypeEnum] = Query(
        None, alias="type", description="Filter by interaction type"
    ),
    interaction_status: Optional[InteractionStatusEnum] = Query(
        None, alias="status", description="Filter by interaction status"
    ),
) -> InteractionListResponse:
    """List competent authority interactions with pagination and filters.

    Args:
        user: Authenticated user with authority:read permission.
        pagination: Standard limit/offset parameters.
        analytics_engine: AuditAnalyticsEngine singleton.
        member_state: Optional filter by EU member state.
        interaction_type: Optional filter by interaction type.
        interaction_status: Optional filter by status.

    Returns:
        Paginated list of authority interaction records.
    """
    start = time.monotonic()
    try:
        filters: Dict[str, Any] = {}
        if member_state:
            filters["member_state"] = member_state.upper()
        if interaction_type:
            filters["interaction_type"] = interaction_type.value
        if interaction_status:
            filters["status"] = interaction_status.value

        interactions: List[Dict[str, Any]] = []
        total = 0
        if hasattr(analytics_engine, "list_authority_interactions"):
            result = await analytics_engine.list_authority_interactions(
                member_state=member_state,
                status=interaction_status.value if interaction_status else None,
                interaction_type=(
                    interaction_type.value if interaction_type else None
                ),
                limit=pagination.limit,
                offset=pagination.offset,
            )
            raw_interactions = result.get("interactions", [])
            total = result.get("total", len(raw_interactions))
            for item in raw_interactions:
                interactions.append(
                    InteractionEntry(
                        interaction_id=item.get("interaction_id", ""),
                        operator_id=item.get("operator_id", ""),
                        authority_name=item.get("authority_name", ""),
                        member_state=item.get("member_state", ""),
                        interaction_type=InteractionTypeEnum(
                            item.get("interaction_type", "document_request")
                        ),
                        received_date=item.get("received_date"),
                        response_deadline=item.get("response_deadline"),
                        response_sla_status=ResponseSLAStatusEnum(
                            item.get("response_sla_status", "on_track")
                        ),
                        status=InteractionStatusEnum(
                            item.get("status", "open")
                        ),
                        provenance_hash=item.get("provenance_hash", ""),
                    )
                )

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance("authority_list", f"total:{total}")

        return InteractionListResponse(
            interactions=interactions,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list authority interactions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve authority interactions",
        )


# ---------------------------------------------------------------------------
# POST /authority-inspections/respond
# ---------------------------------------------------------------------------


@router.post(
    "/authority-inspections/respond",
    response_model=InteractionCreateResponse,
    summary="Respond to authority inspection request",
    description=(
        "Submit an operator's response to a competent authority inspection "
        "request or document request. Records the response date, links the "
        "evidence package, and updates the SLA status. Per EUDR Article 18-23, "
        "operators must respond within the deadlines set by the authority."
    ),
    responses={
        200: {"description": "Inspection response submitted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        404: {"model": ErrorResponse, "description": "Interaction not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def respond_to_inspection(
    request: Request,
    body: InteractionUpdateRequest,
    user: AuthUser = Depends(require_permission("eudr-tam:authority:respond")),
    _rl: None = Depends(rate_limit_write),
    analytics_engine: object = Depends(get_analytics_engine),
    interaction_id: str = Query(
        ..., description="UUID of the authority interaction to respond to"
    ),
) -> InteractionCreateResponse:
    """Submit response to an authority inspection or document request.

    Updates the interaction with response details including evidence
    package reference, response timestamp, and any authority decision.
    Recalculates SLA compliance status based on response timing.

    Args:
        body: Update request with response details.
        user: Authenticated user with authority:respond permission.
        analytics_engine: AuditAnalyticsEngine singleton.
        interaction_id: UUID of the interaction being responded to.

    Returns:
        Updated interaction record with new SLA status.

    Raises:
        HTTPException: 404 if interaction not found, 400 if invalid.
    """
    start = time.monotonic()
    try:
        logger.info(
            "Responding to authority inspection: interaction=%s user=%s",
            interaction_id,
            user.user_id,
        )

        updates = body.model_dump(exclude_none=True)
        updates["responded_by"] = user.user_id

        result: Optional[Dict[str, Any]] = None
        if hasattr(analytics_engine, "update_authority_interaction"):
            result = await analytics_engine.update_authority_interaction(
                interaction_id=interaction_id,
                updates=updates,
                updated_by=user.user_id,
            )
        else:
            # Fallback: return a synthetic response
            result = {
                "interaction_id": interaction_id,
                "operator_id": "",
                "authority_name": "",
                "member_state": "",
                "interaction_type": "document_request",
                "status": "responded",
            }

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Authority interaction {interaction_id} not found",
            )

        interaction = InteractionEntry(
            interaction_id=interaction_id,
            operator_id=result.get("operator_id", ""),
            authority_name=result.get("authority_name", ""),
            member_state=result.get("member_state", ""),
            interaction_type=InteractionTypeEnum(
                result.get("interaction_type", "document_request")
            ),
            received_date=result.get("received_date"),
            response_deadline=result.get("response_deadline"),
            response_submitted_at=result.get("response_submitted_at"),
            authority_decision=result.get("authority_decision"),
            status=InteractionStatusEnum(
                result.get("status", "responded")
            ),
            provenance_hash=result.get("provenance_hash", ""),
        )

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(
            f"authority_respond:{interaction_id}", user.user_id
        )

        logger.info(
            "Authority inspection response submitted: interaction=%s",
            interaction_id,
        )

        return InteractionCreateResponse(
            interaction=interaction,
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to respond to inspection %s: %s", interaction_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit inspection response",
        )


# ---------------------------------------------------------------------------
# GET /analytics/compliance-rate
# ---------------------------------------------------------------------------


@router.get(
    "/analytics/compliance-rate",
    response_model=ComplianceRatesResponse,
    summary="Get compliance rate trend",
    description=(
        "Retrieve compliance rate trend data aggregated by period "
        "(monthly, quarterly, or yearly). Shows the ratio of compliant "
        "audits to total audits, with critical NC rate breakdowns. "
        "Supports filtering by country, commodity, and certification scheme."
    ),
    responses={
        200: {"description": "Compliance rate data retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_compliance_rate(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rl: None = Depends(rate_limit_heavy),
    analytics_engine: object = Depends(get_analytics_engine),
    period: str = Query(
        "quarterly",
        description="Aggregation period: monthly, quarterly, yearly",
    ),
    country_code: Optional[str] = Query(
        None, description="Filter by ISO 3166-1 alpha-2 country code"
    ),
    commodity: Optional[str] = Query(
        None, description="Filter by commodity type"
    ),
    scheme: Optional[str] = Query(
        None, description="Filter by certification scheme"
    ),
) -> ComplianceRatesResponse:
    """Get compliance rate trend analytics.

    Computes the audit compliance rate across time periods using
    deterministic aggregation from audit outcome data.

    Args:
        user: Authenticated user with analytics:read permission.
        analytics_engine: AuditAnalyticsEngine singleton.
        period: Aggregation period (monthly, quarterly, yearly).
        country_code: Optional country filter.
        commodity: Optional commodity filter.
        scheme: Optional certification scheme filter.

    Returns:
        ComplianceRatesResponse with per-period compliance rate data.
    """
    start = time.monotonic()
    try:
        filters: Dict[str, Any] = {"period": period}
        if country_code:
            filters["country_code"] = country_code.upper()
        if commodity:
            filters["commodity"] = commodity
        if scheme:
            filters["scheme"] = scheme

        rates: List[Dict[str, Any]] = []
        overall = Decimal("0.00")
        if hasattr(analytics_engine, "get_compliance_rates"):
            result = await analytics_engine.get_compliance_rates(
                filters=filters
            )
            rates = result.get("rates", [])
            overall = Decimal(
                str(result.get("overall_compliance_rate", "0.00"))
            )

        rate_entries = [
            ComplianceRateEntry(
                period=r.get("period", ""),
                total_audits=r.get("total_audits", 0),
                compliant_audits=r.get("compliant_audits", 0),
                compliance_rate=Decimal(
                    str(r.get("compliance_rate", "0.00"))
                ),
                critical_nc_rate=Decimal(
                    str(r.get("critical_nc_rate", "0.00"))
                ),
            )
            for r in rates
        ]

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(
            f"compliance_rate:{period}", len(rate_entries)
        )

        return ComplianceRatesResponse(
            rates=rate_entries,
            overall_compliance_rate=overall,
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get compliance rates: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance rate analytics",
        )


# ---------------------------------------------------------------------------
# GET /analytics/nc-trends
# ---------------------------------------------------------------------------


@router.get(
    "/analytics/nc-trends",
    response_model=FindingTrendsResponse,
    summary="Get non-conformance trend analytics",
    description=(
        "Retrieve non-conformance (NC) trend data aggregated by period "
        "and severity level (critical, major, minor, observation). "
        "Supports filtering by country, commodity, and severity. Used "
        "for identifying patterns in audit findings across the supply chain."
    ),
    responses={
        200: {"description": "NC trend data retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_nc_trends(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rl: None = Depends(rate_limit_heavy),
    analytics_engine: object = Depends(get_analytics_engine),
    period: str = Query(
        "quarterly",
        description="Aggregation period: monthly, quarterly, yearly",
    ),
    severity: Optional[NCSeverityEnum] = Query(
        None, description="Filter by NC severity level"
    ),
    country_code: Optional[str] = Query(
        None, description="Filter by ISO 3166-1 alpha-2 country code"
    ),
    commodity: Optional[str] = Query(
        None, description="Filter by commodity type"
    ),
) -> FindingTrendsResponse:
    """Get non-conformance trend analytics.

    Aggregates NC data across time periods by severity, providing
    trend analysis for identifying systemic supply chain issues.

    Args:
        user: Authenticated user with analytics:read permission.
        analytics_engine: AuditAnalyticsEngine singleton.
        period: Aggregation period (monthly, quarterly, yearly).
        severity: Optional NC severity filter.
        country_code: Optional country filter.
        commodity: Optional commodity filter.

    Returns:
        FindingTrendsResponse with per-period NC trend data.
    """
    start = time.monotonic()
    try:
        filters: Dict[str, Any] = {"period": period}
        if severity:
            filters["severity"] = severity.value
        if country_code:
            filters["country_code"] = country_code.upper()
        if commodity:
            filters["commodity"] = commodity

        trends: List[Dict[str, Any]] = []
        total_findings = 0
        period_range = ""
        if hasattr(analytics_engine, "get_finding_trends"):
            result = await analytics_engine.get_finding_trends(
                filters=filters
            )
            trends = result.get("trends", [])
            total_findings = result.get("total_findings", 0)
            period_range = result.get("period_range", "")

        trend_entries = [
            FindingTrendEntry(
                period=t.get("period", ""),
                severity=NCSeverityEnum(
                    t.get("severity", "minor")
                ),
                count=t.get("count", 0),
                country_code=t.get("country_code"),
                commodity=t.get("commodity"),
            )
            for t in trends
        ]

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(
            f"nc_trends:{period}", total_findings
        )

        return FindingTrendsResponse(
            trends=trend_entries,
            total_findings=total_findings,
            period_range=period_range,
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get NC trends: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve non-conformance trend analytics",
        )
