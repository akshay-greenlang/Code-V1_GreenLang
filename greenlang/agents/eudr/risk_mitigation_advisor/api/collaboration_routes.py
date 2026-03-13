# -*- coding: utf-8 -*-
"""
Stakeholder Collaboration Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for multi-stakeholder collaboration including plan conversation
threads, task assignment and tracking, and supplier self-service portal
data aggregation.

Endpoints (4):
    POST /collaboration/{plan_id}/messages           - Post message to plan thread
    GET  /collaboration/{plan_id}/messages           - Get plan conversation thread
    POST /collaboration/{plan_id}/tasks              - Assign task to stakeholder
    GET  /collaboration/supplier-portal/{supplier_id} - Supplier self-service portal

RBAC Permissions:
    eudr-rma:collaboration:participate - Post/read messages in plan threads
    eudr-rma:collaboration:manage      - Create and assign collaboration tasks
    eudr-rma:supplier-portal:access    - Access supplier self-service portal

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 8: Stakeholder Collaboration Hub
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
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
    CreateTaskRequest,
    EnrollmentEntry,
    ErrorResponse,
    MessageEntry,
    MessageListResponse,
    PaginatedMeta,
    PlanEntry,
    PostMessageRequest,
    SupplierPortalResponse,
    TaskEntry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collaboration", tags=["Stakeholder Collaboration"])


def _message_dict_to_entry(m: Dict[str, Any]) -> MessageEntry:
    """Convert message dictionary to MessageEntry schema."""
    return MessageEntry(
        message_id=m.get("message_id", ""),
        plan_id=m.get("plan_id", ""),
        sender_id=m.get("sender_id", ""),
        sender_role=m.get("sender_role", ""),
        message_type=m.get("message_type", "text"),
        content=m.get("content", ""),
        attachments=m.get("attachments", []),
        mentions=m.get("mentions", []),
        read_by=m.get("read_by", []),
        sent_at=m.get("sent_at"),
    )


def _task_dict_to_entry(t: Dict[str, Any]) -> TaskEntry:
    """Convert task dictionary to TaskEntry schema."""
    return TaskEntry(
        task_id=t.get("task_id", ""),
        plan_id=t.get("plan_id", ""),
        title=t.get("title", ""),
        description=t.get("description", ""),
        assigned_to=t.get("assigned_to", ""),
        assigned_role=t.get("assigned_role", ""),
        due_date=t.get("due_date"),
        priority=t.get("priority", "medium"),
        status=t.get("status", "pending"),
        created_by=t.get("created_by", ""),
        created_at=t.get("created_at"),
    )


# ---------------------------------------------------------------------------
# POST /collaboration/{plan_id}/messages
# ---------------------------------------------------------------------------


@router.post(
    "/{plan_id}/messages",
    response_model=MessageEntry,
    status_code=status.HTTP_201_CREATED,
    summary="Post message to plan collaboration thread",
    description=(
        "Post a message to the collaboration thread for a remediation plan. "
        "Supports text messages, task update notifications, evidence upload "
        "notifications, and system notifications. Messages are delivered to "
        "all plan stakeholders (operators, suppliers, auditors, certifiers) "
        "according to their notification preferences."
    ),
    responses={
        201: {"description": "Message posted successfully"},
        404: {"model": ErrorResponse, "description": "Plan not found"},
    },
)
async def post_message(
    request: Request,
    plan_id: str,
    body: PostMessageRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:collaboration:participate")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> MessageEntry:
    """Post a message to a plan's collaboration thread."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.post_collaboration_message(
            plan_id=plan_id,
            sender_id=user.user_id,
            content=body.content,
            message_type=body.message_type,
            attachments=body.attachments,
            mentions=body.mentions,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plan {plan_id} not found",
            )

        logger.info(
            "Message posted: plan_id=%s sender=%s type=%s",
            plan_id, user.user_id, body.message_type,
        )
        return _message_dict_to_entry(result if isinstance(result, dict) else {})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Post message failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to post message",
        )


# ---------------------------------------------------------------------------
# GET /collaboration/{plan_id}/messages
# ---------------------------------------------------------------------------


@router.get(
    "/{plan_id}/messages",
    response_model=MessageListResponse,
    summary="Get plan conversation thread",
    description=(
        "Retrieve the collaboration message thread for a remediation plan. "
        "Messages are returned in chronological order with pagination. "
        "Includes text messages, task update notifications, and system events."
    ),
    responses={
        200: {"description": "Messages retrieved"},
        404: {"model": ErrorResponse, "description": "Plan not found"},
    },
)
async def get_messages(
    request: Request,
    plan_id: str,
    message_type: Optional[str] = Query(
        None,
        description="Filter by message type: text, task_update, evidence_upload, system_notification",
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-rma:collaboration:participate")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> MessageListResponse:
    """Get messages for a plan collaboration thread."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.get_collaboration_messages(
            plan_id=plan_id,
            operator_id=user.operator_id,
            message_type=message_type,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plan {plan_id} not found",
            )

        data = result if isinstance(result, dict) else {}
        messages_raw = data.get("messages", [])
        total = data.get("total", 0)
        messages = [_message_dict_to_entry(m) for m in messages_raw]

        return MessageListResponse(
            messages=messages,
            meta=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get messages failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve messages",
        )


# ---------------------------------------------------------------------------
# POST /collaboration/{plan_id}/tasks
# ---------------------------------------------------------------------------


@router.post(
    "/{plan_id}/tasks",
    response_model=TaskEntry,
    status_code=status.HTTP_201_CREATED,
    summary="Assign task to stakeholder",
    description=(
        "Create and assign a collaboration task to a stakeholder within a "
        "remediation plan. Tasks can be assigned to operators, suppliers, "
        "auditors, certifiers, or other roles. Supports priority levels "
        "(low, medium, high, urgent) and optional due dates."
    ),
    responses={
        201: {"description": "Task created and assigned"},
        400: {"model": ErrorResponse, "description": "Invalid task parameters"},
        404: {"model": ErrorResponse, "description": "Plan not found"},
    },
)
async def create_task(
    request: Request,
    plan_id: str,
    body: CreateTaskRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:collaboration:manage")),
    _rate: None = Depends(rate_limit_write),
    service: Any = Depends(get_rma_service),
) -> TaskEntry:
    """Create and assign a collaboration task."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.create_collaboration_task(
            plan_id=plan_id,
            title=body.title,
            description=body.description,
            assigned_to=body.assigned_to,
            assigned_role=body.assigned_role,
            due_date=body.due_date,
            priority=body.priority,
            created_by=user.user_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plan {plan_id} not found",
            )

        logger.info(
            "Task created: plan_id=%s assigned_to=%s priority=%s user=%s",
            plan_id, body.assigned_to, body.priority, user.user_id,
        )
        return _task_dict_to_entry(result if isinstance(result, dict) else {})

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Create task failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create task",
        )


# ---------------------------------------------------------------------------
# GET /collaboration/supplier-portal/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/supplier-portal/{supplier_id}",
    response_model=SupplierPortalResponse,
    summary="Get supplier self-service portal data",
    description=(
        "Retrieve aggregated self-service portal data for a supplier including "
        "active remediation plans, capacity building enrollments, pending tasks, "
        "risk score history, recent messages, and evidence upload URL. Designed "
        "for the supplier-facing portal UI to display a comprehensive dashboard "
        "of their EUDR mitigation status."
    ),
    responses={
        200: {"description": "Portal data retrieved"},
        404: {"model": ErrorResponse, "description": "Supplier not found or no data"},
    },
)
async def get_supplier_portal(
    request: Request,
    supplier_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:supplier-portal:access")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> SupplierPortalResponse:
    """Get supplier self-service portal data."""
    try:
        result = await service.get_supplier_portal_data(
            supplier_id=supplier_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No portal data found for supplier {supplier_id}",
            )

        data = result if isinstance(result, dict) else {}

        # Convert nested objects
        active_plans = [
            PlanEntry(**p) if isinstance(p, dict) else p
            for p in data.get("active_plans", [])
        ]
        capacity_building = [
            EnrollmentEntry(**e) if isinstance(e, dict) else e
            for e in data.get("capacity_building", [])
        ]
        pending_tasks = [_task_dict_to_entry(t) for t in data.get("pending_tasks", [])]
        recent_messages = [_message_dict_to_entry(m) for m in data.get("recent_messages", [])]

        return SupplierPortalResponse(
            supplier_id=supplier_id,
            active_plans=active_plans,
            capacity_building=capacity_building,
            pending_tasks=pending_tasks,
            risk_score_history=data.get("risk_score_history", []),
            recent_messages=recent_messages,
            evidence_upload_url=data.get("evidence_upload_url", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Supplier portal data failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supplier portal data",
        )
