# -*- coding: utf-8 -*-
"""
PADDD Monitoring Routes - AGENT-EUDR-022 Protected Area Validator API

Endpoints for monitoring PADDD (Protected Area Downgrading, Downsizing,
Degazettement) events that affect protected areas near supply chain plots,
listing events, and performing impact assessments.

Endpoints:
    POST /paddd/monitor            - Monitor PADDD events
    GET  /paddd/events             - List PADDD events
    POST /paddd/impact-assessment  - PADDD impact assessment

Auth: eudr-pav:paddd:{create|read}

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022, PADDDMonitor Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.protected_area_validator.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_paddd_monitor,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.protected_area_validator.api.schemas import (
    ErrorResponse,
    MetadataSchema,
    PADDDEventEntry,
    PADDDEventTypeEnum,
    PADDDEventsResponse,
    PADDDImpactAssessmentRequest,
    PADDDImpactAssessmentResponse,
    PADDDMonitorRequest,
    PADDDMonitorResponse,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paddd", tags=["PADDD Monitoring"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _parse_paddd_event(e: dict) -> PADDDEventEntry:
    """Parse a raw PADDD event result into a PADDDEventEntry schema."""
    return PADDDEventEntry(
        event_id=e.get("event_id", ""),
        area_id=e.get("area_id", ""),
        area_name=e.get("area_name", ""),
        country_code=e.get("country_code", ""),
        event_type=PADDDEventTypeEnum(e.get("event_type", "downgrade")),
        event_date=e.get("event_date"),
        enacted_date=e.get("enacted_date"),
        area_affected_km2=Decimal(str(e.get("area_affected_km2", 0)))
        if e.get("area_affected_km2") is not None else None,
        percentage_affected=Decimal(str(e.get("percentage_affected", 0)))
        if e.get("percentage_affected") is not None else None,
        legal_mechanism=e.get("legal_mechanism"),
        description=e.get("description"),
        source=e.get("source"),
        is_reversed=e.get("is_reversed", False),
    )


# ---------------------------------------------------------------------------
# POST /paddd/monitor
# ---------------------------------------------------------------------------


@router.post(
    "/monitor",
    response_model=PADDDMonitorResponse,
    status_code=status.HTTP_200_OK,
    summary="Monitor PADDD events",
    description=(
        "Monitor PADDD (Downgrading, Downsizing, Degazettement) events "
        "affecting protected areas near supply chain plots or in specified "
        "countries. Returns active and historical PADDD events with "
        "area impact details."
    ),
    responses={
        200: {"description": "PADDD monitoring completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def monitor_paddd(
    request: Request,
    body: PADDDMonitorRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:paddd:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> PADDDMonitorResponse:
    """Monitor PADDD events for protected areas.

    Args:
        body: Monitoring request with area/plot IDs or spatial parameters.
        user: Authenticated user with paddd:create permission.

    Returns:
        PADDDMonitorResponse with detected events.
    """
    start = time.monotonic()

    try:
        engine = get_paddd_monitor()
        result = engine.monitor(
            plot_ids=body.plot_ids,
            area_ids=body.area_ids,
            country_codes=[c.upper() for c in body.country_codes] if body.country_codes else None,
            latitude=float(body.center.latitude) if body.center else None,
            longitude=float(body.center.longitude) if body.center else None,
            radius_km=float(body.radius_km) if body.radius_km else None,
            event_types=[t.value for t in body.event_types] if body.event_types else None,
            since_date=body.since_date,
        )

        events = [_parse_paddd_event(e) for e in result.get("events", [])]

        areas_affected = len({e.area_id for e in events})
        active_events = sum(1 for e in events if not e.is_reversed)

        by_type: dict = {}
        for e in events:
            key = e.event_type.value
            by_type[key] = by_type.get(key, 0) + 1

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"paddd_monitor:{body.plot_ids or body.area_ids or body.country_codes}",
            str(len(events)),
        )

        logger.info(
            "PADDD monitoring: events=%d areas=%d active=%d operator=%s",
            len(events),
            areas_affected,
            active_events,
            user.operator_id or user.user_id,
        )

        return PADDDMonitorResponse(
            events=events,
            total_events=len(events),
            areas_affected=areas_affected,
            active_events=active_events,
            by_event_type=by_type,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["PADDDMonitor", "PADDDtracker.org", "WDPA"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("PADDD monitoring failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PADDD monitoring failed",
        )


# ---------------------------------------------------------------------------
# GET /paddd/events
# ---------------------------------------------------------------------------


@router.get(
    "/events",
    response_model=PADDDEventsResponse,
    summary="List PADDD events",
    description=(
        "Retrieve a paginated list of PADDD events with optional filters "
        "for country, event type, area ID, and date range."
    ),
    responses={
        200: {"description": "PADDD events listed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_paddd_events(
    request: Request,
    country_code: Optional[str] = Query(None, description="Filter by country code"),
    event_type: Optional[PADDDEventTypeEnum] = Query(None, description="Filter by event type"),
    area_id: Optional[str] = Query(None, description="Filter by area ID"),
    include_reversed: bool = Query(True, description="Include reversed events"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-pav:paddd:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PADDDEventsResponse:
    """List PADDD events with optional filters.

    Args:
        country_code: Optional country filter.
        event_type: Optional event type filter.
        area_id: Optional area ID filter.
        include_reversed: Whether to include reversed events.
        pagination: Pagination parameters.
        user: Authenticated user with paddd:read permission.

    Returns:
        PADDDEventsResponse with paginated events.
    """
    start = time.monotonic()

    try:
        engine = get_paddd_monitor()
        result = engine.list_events(
            country_code=country_code.upper() if country_code else None,
            event_type=event_type.value if event_type else None,
            area_id=area_id,
            include_reversed=include_reversed,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        events = [_parse_paddd_event(e) for e in result.get("events", [])]
        total = result.get("total", len(events))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance("list_paddd_events", str(total))

        logger.info(
            "PADDD events listed: total=%d operator=%s",
            total,
            user.operator_id or user.user_id,
        )

        return PADDDEventsResponse(
            events=events,
            total_events=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["PADDDMonitor", "PADDDtracker.org"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("PADDD events listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PADDD events listing failed",
        )


# ---------------------------------------------------------------------------
# POST /paddd/impact-assessment
# ---------------------------------------------------------------------------


@router.post(
    "/impact-assessment",
    response_model=PADDDImpactAssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="PADDD impact assessment on supply chain",
    description=(
        "Assess the impact of PADDD events on supply chain plots. Evaluates "
        "how downgrading, downsizing, or degazettement of protected areas "
        "changes risk scores and compliance status for affected plots."
    ),
    responses={
        200: {"description": "Impact assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_paddd_impact(
    request: Request,
    body: PADDDImpactAssessmentRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:paddd:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> PADDDImpactAssessmentResponse:
    """Assess PADDD impact on supply chain.

    Args:
        body: Impact assessment request.
        user: Authenticated user with paddd:create permission.

    Returns:
        PADDDImpactAssessmentResponse with impact analysis.
    """
    start = time.monotonic()

    try:
        engine = get_paddd_monitor()
        result = engine.assess_impact(
            event_id=body.event_id,
            area_id=body.area_id,
            plot_ids=body.plot_ids,
            include_supply_chain_impact=body.include_supply_chain_impact,
            include_risk_reassessment=body.include_risk_reassessment,
            assessed_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not perform impact assessment. Specify event_id or area_id.",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"paddd_impact:{body.event_id or body.area_id}",
            str(result.get("plots_affected", 0)),
        )

        logger.info(
            "PADDD impact assessed: event=%s area=%s plots_affected=%d operator=%s",
            body.event_id,
            body.area_id,
            result.get("plots_affected", 0),
            user.operator_id or user.user_id,
        )

        return PADDDImpactAssessmentResponse(
            assessment_id=result.get("assessment_id", ""),
            event_id=body.event_id,
            area_id=body.area_id,
            plots_affected=result.get("plots_affected", 0),
            risk_change_summary=result.get("risk_change_summary", {}),
            supply_chain_impact=result.get("supply_chain_impact"),
            recommendations=result.get("recommendations", []),
            assessment_rationale=result.get("assessment_rationale", ""),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["PADDDMonitor", "RiskScorer", "ComplianceAssessor"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("PADDD impact assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PADDD impact assessment failed",
        )
