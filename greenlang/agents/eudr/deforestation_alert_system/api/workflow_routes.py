# -*- coding: utf-8 -*-
"""
Alert Workflow Routes - AGENT-EUDR-020 Deforestation Alert System API

Endpoints for alert workflow management with auto-triage, investigation,
resolution, and escalation states. Configurable SLA deadlines: triage 4h,
investigation 48h, resolution 168h. Up to 3 escalation levels.

Endpoints:
    POST /workflow/triage       - Triage an alert
    POST /workflow/assign       - Assign alert to investigator
    POST /workflow/investigate  - Start investigation
    POST /workflow/resolve      - Resolve an alert
    POST /workflow/escalate     - Escalate an alert
    GET  /workflow/sla          - Get SLA status for alerts

Workflow: NEW -> TRIAGED -> ASSIGNED -> INVESTIGATING -> RESOLVED/ESCALATED
SLA: Triage 4h, Investigation 48h, Resolution 168h (7 days)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020, AlertWorkflowEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.deforestation_alert_system.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_workflow_engine,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.deforestation_alert_system.api.schemas import (
    AlertStatusEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    SLAEntry,
    SLAStatusEnum,
    WorkflowActionEnum,
    WorkflowAssignRequest,
    WorkflowEscalateRequest,
    WorkflowInvestigateRequest,
    WorkflowPriorityEnum,
    WorkflowResolveRequest,
    WorkflowSLAResponse,
    WorkflowTransitionResponse,
    WorkflowTriageRequest,
    WorkflowTriageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflow", tags=["Alert Workflow"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /workflow/triage
# ---------------------------------------------------------------------------


@router.post(
    "/triage",
    response_model=WorkflowTriageResponse,
    status_code=status.HTTP_200_OK,
    summary="Triage a deforestation alert",
    description=(
        "Triage a deforestation alert by assigning priority and optionally "
        "auto-assigning to an investigator. Transitions alert status from "
        "NEW to TRIAGED. SLA: triage must occur within 4 hours."
    ),
    responses={
        200: {"description": "Alert triaged successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def triage_alert(
    request: Request,
    body: WorkflowTriageRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:workflow:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> WorkflowTriageResponse:
    """Triage a deforestation alert.

    Args:
        body: Triage request.
        user: Authenticated user with workflow:create permission.

    Returns:
        WorkflowTriageResponse with triage result.
    """
    start = time.monotonic()

    try:
        engine = get_workflow_engine()
        result = engine.triage(
            alert_id=body.alert_id,
            priority=body.priority.value,
            notes=body.notes,
            auto_assign=body.auto_assign,
            assignee_id=body.assignee_id,
            triaged_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {body.alert_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"triage:{body.alert_id}:{body.priority.value}",
            str(result.get("new_status", "triaged")),
        )

        logger.info(
            "Alert triaged: alert_id=%s priority=%s operator=%s",
            body.alert_id,
            body.priority.value,
            user.operator_id or user.user_id,
        )

        return WorkflowTriageResponse(
            alert_id=body.alert_id,
            previous_status=AlertStatusEnum(result.get("previous_status", "new")),
            new_status=AlertStatusEnum(result.get("new_status", "triaged")),
            priority=body.priority,
            triaged_by=user.user_id,
            sla_deadline=result.get("sla_deadline"),
            assigned_to=result.get("assigned_to"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AlertWorkflowEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert triage failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert triage failed",
        )


# ---------------------------------------------------------------------------
# POST /workflow/assign
# ---------------------------------------------------------------------------


@router.post(
    "/assign",
    response_model=WorkflowTransitionResponse,
    status_code=status.HTTP_200_OK,
    summary="Assign alert to investigator",
    description=(
        "Assign a triaged alert to a specific investigator. Transitions "
        "alert status from TRIAGED to ASSIGNED."
    ),
    responses={
        200: {"description": "Alert assigned"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def assign_alert(
    request: Request,
    body: WorkflowAssignRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:workflow:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> WorkflowTransitionResponse:
    """Assign an alert to an investigator.

    Args:
        body: Assignment request.
        user: Authenticated user with workflow:update permission.

    Returns:
        WorkflowTransitionResponse with assignment result.
    """
    start = time.monotonic()

    try:
        engine = get_workflow_engine()
        result = engine.assign(
            alert_id=body.alert_id,
            assignee_id=body.assignee_id,
            priority=body.priority.value if body.priority else None,
            notes=body.notes,
            due_date=body.due_date,
            assigned_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {body.alert_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"assign:{body.alert_id}:{body.assignee_id}",
            str(result.get("new_status", "assigned")),
        )

        logger.info(
            "Alert assigned: alert_id=%s assignee=%s operator=%s",
            body.alert_id,
            body.assignee_id,
            user.operator_id or user.user_id,
        )

        return WorkflowTransitionResponse(
            alert_id=body.alert_id,
            action=WorkflowActionEnum.ASSIGN,
            previous_status=AlertStatusEnum(result.get("previous_status", "triaged")),
            new_status=AlertStatusEnum(result.get("new_status", "assigned")),
            performed_by=user.user_id,
            sla_deadline=result.get("sla_deadline"),
            notes=body.notes,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AlertWorkflowEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert assignment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert assignment failed",
        )


# ---------------------------------------------------------------------------
# POST /workflow/investigate
# ---------------------------------------------------------------------------


@router.post(
    "/investigate",
    response_model=WorkflowTransitionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start investigation on an alert",
    description=(
        "Begin investigation on an assigned alert. Transitions status from "
        "ASSIGNED to INVESTIGATING. SLA: investigation must complete within 48h."
    ),
    responses={
        200: {"description": "Investigation started"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
    },
)
async def investigate_alert(
    request: Request,
    body: WorkflowInvestigateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:workflow:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> WorkflowTransitionResponse:
    """Start investigation on an alert.

    Args:
        body: Investigation request.
        user: Authenticated user with workflow:update permission.

    Returns:
        WorkflowTransitionResponse with investigation start result.
    """
    start = time.monotonic()

    try:
        engine = get_workflow_engine()
        result = engine.investigate(
            alert_id=body.alert_id,
            investigation_type=body.investigation_type,
            notes=body.notes,
            evidence_urls=body.evidence_urls,
            investigated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {body.alert_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"investigate:{body.alert_id}",
            str(result.get("new_status", "investigating")),
        )

        logger.info(
            "Investigation started: alert_id=%s type=%s operator=%s",
            body.alert_id,
            body.investigation_type or "general",
            user.operator_id or user.user_id,
        )

        return WorkflowTransitionResponse(
            alert_id=body.alert_id,
            action=WorkflowActionEnum.INVESTIGATE,
            previous_status=AlertStatusEnum(result.get("previous_status", "assigned")),
            new_status=AlertStatusEnum(result.get("new_status", "investigating")),
            performed_by=user.user_id,
            sla_deadline=result.get("sla_deadline"),
            notes=body.notes,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AlertWorkflowEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Investigation start failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Investigation start failed",
        )


# ---------------------------------------------------------------------------
# POST /workflow/resolve
# ---------------------------------------------------------------------------


@router.post(
    "/resolve",
    response_model=WorkflowTransitionResponse,
    status_code=status.HTTP_200_OK,
    summary="Resolve a deforestation alert",
    description=(
        "Resolve an alert after investigation with findings, root cause, "
        "and remediation actions. Transitions to RESOLVED or FALSE_POSITIVE."
    ),
    responses={
        200: {"description": "Alert resolved"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
    },
)
async def resolve_alert(
    request: Request,
    body: WorkflowResolveRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:workflow:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> WorkflowTransitionResponse:
    """Resolve a deforestation alert.

    Args:
        body: Resolution request.
        user: Authenticated user with workflow:update permission.

    Returns:
        WorkflowTransitionResponse with resolution result.
    """
    start = time.monotonic()

    try:
        engine = get_workflow_engine()
        result = engine.resolve(
            alert_id=body.alert_id,
            resolution=body.resolution,
            root_cause=body.root_cause,
            findings=body.findings,
            evidence_urls=body.evidence_urls,
            remediation_actions=[a.value for a in body.remediation_actions]
            if body.remediation_actions else None,
            is_false_positive=body.is_false_positive,
            resolved_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {body.alert_id}",
            )

        new_status = "false_positive" if body.is_false_positive else "resolved"

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"resolve:{body.alert_id}:{body.resolution}",
            str(result.get("new_status", new_status)),
        )

        logger.info(
            "Alert resolved: alert_id=%s resolution=%s false_positive=%s operator=%s",
            body.alert_id,
            body.resolution,
            body.is_false_positive,
            user.operator_id or user.user_id,
        )

        return WorkflowTransitionResponse(
            alert_id=body.alert_id,
            action=WorkflowActionEnum.RESOLVE,
            previous_status=AlertStatusEnum(result.get("previous_status", "investigating")),
            new_status=AlertStatusEnum(result.get("new_status", new_status)),
            performed_by=user.user_id,
            notes=body.findings,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AlertWorkflowEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert resolution failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert resolution failed",
        )


# ---------------------------------------------------------------------------
# POST /workflow/escalate
# ---------------------------------------------------------------------------


@router.post(
    "/escalate",
    response_model=WorkflowTransitionResponse,
    status_code=status.HTTP_200_OK,
    summary="Escalate a deforestation alert",
    description=(
        "Escalate an alert to a higher level (up to 3 levels). Triggers "
        "additional notification and shortened SLA deadlines."
    ),
    responses={
        200: {"description": "Alert escalated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
        409: {"model": ErrorResponse, "description": "Max escalation level reached"},
    },
)
async def escalate_alert(
    request: Request,
    body: WorkflowEscalateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:workflow:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> WorkflowTransitionResponse:
    """Escalate a deforestation alert.

    Args:
        body: Escalation request.
        user: Authenticated user with workflow:update permission.

    Returns:
        WorkflowTransitionResponse with escalation result.
    """
    start = time.monotonic()

    try:
        engine = get_workflow_engine()
        result = engine.escalate(
            alert_id=body.alert_id,
            escalation_level=body.escalation_level,
            reason=body.reason,
            escalate_to=body.escalate_to,
            requires_external_review=body.requires_external_review,
            escalated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {body.alert_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"escalate:{body.alert_id}:level-{body.escalation_level}",
            str(result.get("new_status", "escalated")),
        )

        logger.info(
            "Alert escalated: alert_id=%s level=%d reason=%s operator=%s",
            body.alert_id,
            body.escalation_level,
            body.reason[:50],
            user.operator_id or user.user_id,
        )

        return WorkflowTransitionResponse(
            alert_id=body.alert_id,
            action=WorkflowActionEnum.ESCALATE,
            previous_status=AlertStatusEnum(result.get("previous_status", "investigating")),
            new_status=AlertStatusEnum(result.get("new_status", "escalated")),
            performed_by=user.user_id,
            sla_deadline=result.get("sla_deadline"),
            notes=body.reason,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AlertWorkflowEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Alert escalation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert escalation failed",
        )


# ---------------------------------------------------------------------------
# GET /workflow/sla
# ---------------------------------------------------------------------------


@router.get(
    "/sla",
    response_model=WorkflowSLAResponse,
    summary="Get SLA status for alerts",
    description=(
        "Retrieve SLA compliance status for active alerts including time "
        "remaining, breach status, and overall compliance rate. "
        "SLA targets: triage 4h, investigation 48h, resolution 168h."
    ),
    responses={
        200: {"description": "SLA status retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_sla_status(
    request: Request,
    sla_status: Optional[SLAStatusEnum] = Query(None, description="Filter by SLA status"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:workflow:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> WorkflowSLAResponse:
    """Get SLA status for active alerts.

    Args:
        sla_status: Optional SLA status filter.
        pagination: Pagination parameters.
        user: Authenticated user with workflow:read permission.

    Returns:
        WorkflowSLAResponse with SLA status data.
    """
    start = time.monotonic()

    try:
        engine = get_workflow_engine()
        result = engine.get_sla_status(
            sla_status=sla_status.value if sla_status else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        entries = []
        for entry in result.get("sla_entries", []):
            entries.append(
                SLAEntry(
                    alert_id=entry.get("alert_id", ""),
                    current_status=AlertStatusEnum(entry.get("current_status", "new")),
                    sla_stage=entry.get("sla_stage", "triage"),
                    deadline=entry.get("deadline"),
                    sla_status=SLAStatusEnum(entry.get("sla_status", "on_track")),
                    hours_remaining=Decimal(str(entry.get("hours_remaining", 0)))
                    if entry.get("hours_remaining") is not None else None,
                    hours_elapsed=Decimal(str(entry.get("hours_elapsed", 0)))
                    if entry.get("hours_elapsed") is not None else None,
                    escalation_level=entry.get("escalation_level", 0),
                )
            )

        total = result.get("total_tracked", len(entries))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"sla_status:{sla_status}", str(total)
        )

        logger.info(
            "SLA status retrieved: total=%d on_track=%d at_risk=%d breached=%d operator=%s",
            total,
            result.get("on_track_count", 0),
            result.get("at_risk_count", 0),
            result.get("breached_count", 0),
            user.operator_id or user.user_id,
        )

        return WorkflowSLAResponse(
            sla_entries=entries,
            total_tracked=total,
            on_track_count=result.get("on_track_count", 0),
            at_risk_count=result.get("at_risk_count", 0),
            breached_count=result.get("breached_count", 0),
            sla_compliance_rate=Decimal(str(result.get("sla_compliance_rate", 0)))
            if result.get("sla_compliance_rate") is not None else None,
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
                data_sources=["AlertWorkflowEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("SLA status retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SLA status retrieval failed",
        )
