# -*- coding: utf-8 -*-
"""
Adaptive Management / Monitoring Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for continuous monitoring and adaptive management including
trigger event listing, event acknowledgement, real-time monitoring dashboard,
and plan drift analysis.

Endpoints (4):
    GET /monitoring/triggers                           - List active trigger events
    PUT /monitoring/triggers/{event_id}/acknowledge    - Acknowledge a trigger event
    GET /monitoring/dashboard                          - Real-time monitoring dashboard
    GET /monitoring/drift/{plan_id}                    - Plan drift analysis

RBAC Permissions:
    eudr-rma:monitoring:read         - View triggers and dashboard
    eudr-rma:monitoring:acknowledge  - Acknowledge trigger events

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 6: Continuous Monitoring
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_rma_service,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_uuid,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    AcknowledgeRequest,
    DriftAnalysisResponse,
    ErrorResponse,
    MonitoringDashboardResponse,
    PaginatedMeta,
    TriggerEventEntry,
    TriggerListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Adaptive Management"])


def _trigger_dict_to_entry(t: Dict[str, Any]) -> TriggerEventEntry:
    """Convert trigger event dictionary to TriggerEventEntry."""
    return TriggerEventEntry(
        event_id=t.get("event_id", ""),
        plan_id=t.get("plan_id"),
        trigger_type=t.get("trigger_type", ""),
        source_agent=t.get("source_agent", ""),
        severity=t.get("severity", "medium"),
        description=t.get("description", ""),
        risk_data=t.get("risk_data", {}),
        recommended_adjustment=t.get("recommended_adjustment", {}),
        adjustment_type=t.get("adjustment_type"),
        acknowledged=t.get("acknowledged", False),
        acknowledged_by=t.get("acknowledged_by"),
        acknowledged_at=t.get("acknowledged_at"),
        resolved=t.get("resolved", False),
        resolved_at=t.get("resolved_at"),
        detected_at=t.get("detected_at"),
    )


# ---------------------------------------------------------------------------
# GET /monitoring/triggers
# ---------------------------------------------------------------------------


@router.get(
    "/triggers",
    response_model=TriggerListResponse,
    summary="List active trigger events",
    description=(
        "Retrieve monitoring trigger events detected by the adaptive management "
        "engine. Events originate from 9 upstream risk agents (EUDR-016 through "
        "EUDR-024) when risk signals exceed trigger thresholds for active "
        "mitigation plans. Supports filters by severity, source agent, "
        "acknowledged/resolved status, and plan ID."
    ),
    responses={200: {"description": "Triggers listed"}},
)
async def list_triggers(
    request: Request,
    severity: Optional[str] = Query(None, description="Filter by severity: critical, high, medium, low"),
    source_agent: Optional[str] = Query(None, description="Filter by source agent (e.g., EUDR-020)"),
    plan_id: Optional[str] = Query(None, description="Filter by affected plan ID"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged status"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-rma:monitoring:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> TriggerListResponse:
    """List trigger events with filters."""
    try:
        result = await service.list_trigger_events(
            operator_id=user.operator_id,
            severity=severity,
            source_agent=source_agent,
            plan_id=plan_id,
            acknowledged=acknowledged,
            resolved=resolved,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        triggers_raw = result.get("triggers", []) if isinstance(result, dict) else []
        total = result.get("total", 0) if isinstance(result, dict) else 0
        triggers = [_trigger_dict_to_entry(t) for t in triggers_raw]

        return TriggerListResponse(
            triggers=triggers,
            meta=PaginatedMeta(
                total=total, limit=pagination.limit, offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except Exception as e:
        logger.error("Trigger list failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve triggers")


# ---------------------------------------------------------------------------
# PUT /monitoring/triggers/{event_id}/acknowledge
# ---------------------------------------------------------------------------


@router.put(
    "/triggers/{event_id}/acknowledge",
    response_model=TriggerEventEntry,
    summary="Acknowledge a trigger event",
    description=(
        "Acknowledge receipt of a trigger event and optionally record "
        "immediate action taken. Required within the SLA window defined "
        "by the trigger response matrix (4h for critical, 24h for high, "
        "48h for medium)."
    ),
    responses={
        200: {"description": "Trigger acknowledged"},
        404: {"model": ErrorResponse, "description": "Trigger event not found"},
    },
)
async def acknowledge_trigger(
    request: Request,
    event_id: str,
    body: AcknowledgeRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:monitoring:acknowledge")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> TriggerEventEntry:
    """Acknowledge a trigger event."""
    validate_uuid(event_id, "event_id")

    try:
        result = await service.acknowledge_trigger(
            event_id=event_id,
            acknowledged_by=user.user_id,
            notes=body.notes,
            action_taken=body.action_taken,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trigger event {event_id} not found",
            )

        logger.info("Trigger acknowledged: event_id=%s user=%s", event_id, user.user_id)
        return _trigger_dict_to_entry(result if isinstance(result, dict) else {})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Trigger acknowledge failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to acknowledge trigger")


# ---------------------------------------------------------------------------
# GET /monitoring/dashboard
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard",
    response_model=MonitoringDashboardResponse,
    summary="Get real-time monitoring dashboard data",
    description=(
        "Retrieve aggregated monitoring dashboard data including active "
        "plan counts, unresolved trigger counts by severity and source, "
        "plans at risk, recent trigger events, and next Article 8(3) "
        "annual review date."
    ),
    responses={200: {"description": "Dashboard data retrieved"}},
)
async def get_monitoring_dashboard(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-rma:monitoring:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> MonitoringDashboardResponse:
    """Get monitoring dashboard data."""
    try:
        result = await service.get_monitoring_dashboard(
            operator_id=user.operator_id,
        )

        data = result if isinstance(result, dict) else {}
        recent_triggers = [_trigger_dict_to_entry(t) for t in data.get("recent_triggers", [])]

        return MonitoringDashboardResponse(
            active_plans_count=data.get("active_plans_count", 0),
            unresolved_triggers_count=data.get("unresolved_triggers_count", 0),
            critical_triggers_count=data.get("critical_triggers_count", 0),
            triggers_by_severity=data.get("triggers_by_severity", {}),
            triggers_by_source=data.get("triggers_by_source", {}),
            plans_at_risk=data.get("plans_at_risk", []),
            recent_triggers=recent_triggers,
            next_annual_review=data.get("next_annual_review"),
        )

    except Exception as e:
        logger.error("Monitoring dashboard failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get dashboard data")


# ---------------------------------------------------------------------------
# GET /monitoring/drift/{plan_id}
# ---------------------------------------------------------------------------


@router.get(
    "/drift/{plan_id}",
    response_model=DriftAnalysisResponse,
    summary="Get plan drift analysis",
    description=(
        "Analyze the drift between planned risk reduction trajectory "
        "and actual risk reduction trajectory for a specific plan. "
        "Positive drift = better than planned. Negative drift = "
        "mitigation not keeping pace with expectations."
    ),
    responses={
        200: {"description": "Drift analysis generated"},
        404: {"model": ErrorResponse, "description": "Plan not found or insufficient data"},
    },
)
async def get_drift_analysis(
    request: Request,
    plan_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:monitoring:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> DriftAnalysisResponse:
    """Get plan drift analysis."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.get_plan_drift(
            plan_id=plan_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No drift data for plan {plan_id}",
            )

        data = result if isinstance(result, dict) else {}
        return DriftAnalysisResponse(
            plan_id=plan_id,
            plan_name=data.get("plan_name", ""),
            planned_trajectory=data.get("planned_trajectory", []),
            actual_trajectory=data.get("actual_trajectory", []),
            drift_pct=Decimal(str(data.get("drift_pct", 0))),
            drift_direction=data.get("drift_direction", "on_track"),
            recommendation=data.get("recommendation", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Drift analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate drift analysis")
