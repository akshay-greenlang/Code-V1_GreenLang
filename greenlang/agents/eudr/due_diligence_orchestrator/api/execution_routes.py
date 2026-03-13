# -*- coding: utf-8 -*-
"""
Execution Control Routes - AGENT-EUDR-026 Due Diligence Orchestrator API

Endpoints for controlling the execution lifecycle of due diligence
workflows: starting, pausing, resuming, cancelling, and rolling back
workflow execution. Each operation transitions the workflow state
machine and creates an audit trail entry.

Endpoints (5):
    POST /workflows/{id}/start    - Start executing a created workflow
    POST /workflows/{id}/pause    - Pause a running workflow
    POST /workflows/{id}/resume   - Resume from checkpoint
    POST /workflows/{id}/cancel   - Cancel a workflow
    POST /workflows/{id}/rollback - Rollback to a previous checkpoint

RBAC Permissions:
    eudr-ddo:workflows:manage  - Pause, resume, cancel, rollback

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_ddo_service,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    ResumeWorkflowRequest,
    StartWorkflowRequest,
    WorkflowStatus,
    WorkflowStatusResponse,
    _utcnow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Execution Control"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    """Request body for starting a workflow."""

    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Initial input data for Phase 1 agents",
    )
    priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Execution priority (1=highest, 5=lowest)",
    )


class ResumeRequest(BaseModel):
    """Request body for resuming a workflow."""

    checkpoint_id: Optional[str] = Field(
        default=None,
        description="Specific checkpoint to resume from (None = latest)",
    )
    retry_failed: bool = Field(
        default=True,
        description="Whether to retry previously failed agents",
    )


class CancelRequest(BaseModel):
    """Request body for cancelling a workflow."""

    reason: str = Field(
        default="User-initiated cancellation",
        max_length=2000,
        description="Cancellation reason for audit trail",
    )


class RollbackRequest(BaseModel):
    """Request body for rolling back a workflow."""

    checkpoint_id: str = Field(
        ...,
        description="Target checkpoint ID to rollback to",
    )
    reason: str = Field(
        default="",
        max_length=2000,
        description="Rollback reason for audit trail",
    )


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/start -- Start workflow execution
# ---------------------------------------------------------------------------


@router.post(
    "/{workflow_id}/start",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Start workflow execution",
    description=(
        "Start executing a created due diligence workflow. Transitions "
        "the workflow from CREATED to VALIDATING to RUNNING and begins "
        "executing the first layer of agents in the DAG."
    ),
    responses={
        200: {"description": "Workflow started successfully"},
        400: {"model": ErrorResponse, "description": "Workflow cannot be started"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def start_workflow(
    request: Request,
    workflow_id: str,
    body: Optional[StartRequest] = None,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:manage")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> WorkflowStatusResponse:
    """Start executing a workflow.

    Validates the workflow DAG, initializes the execution timeline,
    and begins executing the first layer of agents. The workflow
    must be in CREATED status to start.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        body: Optional start parameters.
        user: Authenticated and authorized user.

    Returns:
        WorkflowStatusResponse with running status.

    Raises:
        HTTPException: 400/409 if workflow cannot be started.
    """
    logger.info(
        "start_workflow: user=%s workflow_id=%s",
        user.user_id,
        workflow_id,
    )

    service = get_ddo_service()

    # Verify workflow exists and is in valid state
    try:
        current = service.get_workflow_status(workflow_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    if current.status != WorkflowStatus.CREATED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot start workflow in {current.status.value} status. "
                f"Only CREATED workflows can be started."
            ),
        )

    start_request = StartWorkflowRequest(
        workflow_id=workflow_id,
        input_data=body.input_data if body else {},
        priority=body.priority if body else 3,
    )

    try:
        return service.start_workflow(start_request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/pause -- Pause workflow
# ---------------------------------------------------------------------------


@router.post(
    "/{workflow_id}/pause",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Pause workflow execution",
    description=(
        "Pause a running workflow. Creates a checkpoint and stops "
        "dispatching new agent invocations. In-flight agents will "
        "complete but their results are saved for resume."
    ),
    responses={
        200: {"description": "Workflow paused successfully"},
        400: {"model": ErrorResponse, "description": "Workflow cannot be paused"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def pause_workflow(
    request: Request,
    workflow_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:manage")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> WorkflowStatusResponse:
    """Pause a running workflow.

    Saves a checkpoint of the current state and transitions the
    workflow to PAUSED status. Use resume to continue execution
    from the checkpoint.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        user: Authenticated and authorized user.

    Returns:
        WorkflowStatusResponse with paused status.

    Raises:
        HTTPException: 409 if workflow is not in a pausable state.
    """
    logger.info(
        "pause_workflow: user=%s workflow_id=%s",
        user.user_id,
        workflow_id,
    )

    service = get_ddo_service()

    try:
        current = service.get_workflow_status(workflow_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    if current.status != WorkflowStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot pause workflow in {current.status.value} status. "
                f"Only RUNNING workflows can be paused."
            ),
        )

    try:
        return service.pause_workflow(workflow_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/resume -- Resume workflow
# ---------------------------------------------------------------------------


@router.post(
    "/{workflow_id}/resume",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Resume workflow execution",
    description=(
        "Resume a paused or gate-failed workflow from the latest "
        "checkpoint (or a specific checkpoint). Completed agents "
        "are not re-executed; execution continues from where it stopped."
    ),
    responses={
        200: {"description": "Workflow resumed successfully"},
        400: {"model": ErrorResponse, "description": "Workflow cannot be resumed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
        409: {"model": ErrorResponse, "description": "Invalid state transition"},
    },
)
async def resume_workflow(
    request: Request,
    workflow_id: str,
    body: Optional[ResumeRequest] = None,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:manage")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> WorkflowStatusResponse:
    """Resume a paused or gate-failed workflow.

    Restores the workflow state from the latest checkpoint and continues
    execution from the last successful point. Optionally specify a
    specific checkpoint to resume from.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        body: Optional resume parameters.
        user: Authenticated and authorized user.

    Returns:
        WorkflowStatusResponse with running status.

    Raises:
        HTTPException: 409 if workflow is not in a resumable state.
    """
    logger.info(
        "resume_workflow: user=%s workflow_id=%s checkpoint=%s",
        user.user_id,
        workflow_id,
        body.checkpoint_id if body else "latest",
    )

    service = get_ddo_service()

    try:
        current = service.get_workflow_status(workflow_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    resumable_states = {
        WorkflowStatus.PAUSED,
        WorkflowStatus.GATE_FAILED,
        WorkflowStatus.TERMINATED,
    }
    if current.status not in resumable_states:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot resume workflow in {current.status.value} status. "
                f"Only PAUSED, GATE_FAILED, or TERMINATED workflows can be resumed."
            ),
        )

    resume_request = ResumeWorkflowRequest(
        workflow_id=workflow_id,
        checkpoint_id=body.checkpoint_id if body else None,
        retry_failed=body.retry_failed if body else True,
    )

    try:
        return service.resume_workflow(resume_request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/cancel -- Cancel workflow
# ---------------------------------------------------------------------------


@router.post(
    "/{workflow_id}/cancel",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel workflow execution",
    description=(
        "Cancel a workflow. Terminates all running agents and marks "
        "the workflow as cancelled. This is a terminal state -- cancelled "
        "workflows cannot be resumed."
    ),
    responses={
        200: {"description": "Workflow cancelled successfully"},
        400: {"model": ErrorResponse, "description": "Workflow cannot be cancelled"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
        409: {"model": ErrorResponse, "description": "Workflow already in terminal state"},
    },
)
async def cancel_workflow(
    request: Request,
    workflow_id: str,
    body: Optional[CancelRequest] = None,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:manage")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> WorkflowStatusResponse:
    """Cancel a workflow.

    Transitions the workflow to CANCELLED terminal state. All in-flight
    agent invocations are terminated, and the cancellation is recorded
    in the audit trail.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        body: Optional cancellation parameters.
        user: Authenticated and authorized user.

    Returns:
        WorkflowStatusResponse with cancelled status.

    Raises:
        HTTPException: 409 if workflow is already in a terminal state.
    """
    logger.info(
        "cancel_workflow: user=%s workflow_id=%s reason=%s",
        user.user_id,
        workflow_id,
        body.reason if body else "User-initiated",
    )

    service = get_ddo_service()

    try:
        current = service.get_workflow_status(workflow_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    terminal_states = {
        WorkflowStatus.COMPLETED,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.TERMINATED,
    }
    if current.status in terminal_states:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Workflow is already in terminal state: {current.status.value}. "
                f"Cannot cancel."
            ),
        )

    reason = body.reason if body else "User-initiated cancellation"
    try:
        return service.cancel_workflow(workflow_id, reason=reason)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/rollback -- Rollback to checkpoint
# ---------------------------------------------------------------------------


@router.post(
    "/{workflow_id}/rollback",
    status_code=status.HTTP_200_OK,
    summary="Rollback to a previous checkpoint",
    description=(
        "Rollback workflow state to a specific previous checkpoint. "
        "All agent execution records and state changes after the target "
        "checkpoint are discarded. The workflow is paused at the rollback "
        "point and must be explicitly resumed."
    ),
    responses={
        200: {"description": "Workflow rolled back successfully"},
        400: {"model": ErrorResponse, "description": "Invalid rollback parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow or checkpoint not found"},
    },
)
async def rollback_workflow(
    request: Request,
    workflow_id: str,
    body: RollbackRequest,
    user: AuthUser = Depends(require_permission("eudr-ddo:checkpoints:rollback")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> Dict[str, Any]:
    """Rollback workflow to a previous checkpoint.

    Restores the workflow state to the specified checkpoint, discarding
    all subsequent execution records and state transitions. The workflow
    transitions to PAUSED at the rollback point.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        body: Rollback parameters with target checkpoint ID.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with rollback confirmation and checkpoint details.

    Raises:
        HTTPException: 404 if workflow or checkpoint not found.
    """
    logger.info(
        "rollback_workflow: user=%s workflow_id=%s checkpoint=%s reason=%s",
        user.user_id,
        workflow_id,
        body.checkpoint_id,
        body.reason,
    )

    service = get_ddo_service()

    # Verify workflow exists
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    try:
        rollback_result = service._state_manager.rollback_to_checkpoint(
            workflow_id=workflow_id,
            checkpoint_id=body.checkpoint_id,
            reason=body.reason or f"Rollback by {user.user_id}",
            actor=user.user_id,
        )

        return {
            "workflow_id": workflow_id,
            "status": "paused",
            "rolled_back_to_checkpoint": body.checkpoint_id,
            "rolled_back_at": _utcnow().isoformat(),
            "rolled_back_by": user.user_id,
            "reason": body.reason,
            "checkpoint_phase": rollback_result.get("phase", "unknown") if isinstance(rollback_result, dict) else "unknown",
            "checkpoint_sequence": rollback_result.get("sequence_number", 0) if isinstance(rollback_result, dict) else 0,
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
