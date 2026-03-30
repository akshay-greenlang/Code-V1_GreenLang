# -*- coding: utf-8 -*-
"""
Checkpoint Management Routes - AGENT-EUDR-026 Due Diligence Orchestrator API

Endpoints for managing workflow checkpoints. Checkpoints capture the
complete workflow state after every agent completion and quality gate
evaluation, enabling resume-from-any-point and rollback capabilities.
Each checkpoint includes a cumulative SHA-256 provenance hash for
tamper detection and audit trail integrity.

Endpoints (3):
    GET  /workflows/{id}/checkpoints    - List checkpoints for a workflow
    POST /workflows/{id}/checkpoints    - Create a manual checkpoint
    GET  /checkpoints/{checkpoint_id}   - Get checkpoint details by ID

RBAC Permissions:
    eudr-ddo:checkpoints:read     - View checkpoint data
    eudr-ddo:checkpoints:rollback - Create manual checkpoints

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_ddo_service,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    WorkflowCheckpoint,
    _utcnow,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Checkpoint Management"])


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/checkpoints -- List checkpoints
# ---------------------------------------------------------------------------


@router.get(
    "/workflows/{workflow_id}/checkpoints",
    status_code=status.HTTP_200_OK,
    summary="List checkpoints for a workflow",
    description=(
        "List all checkpoints for a workflow ordered by sequence number. "
        "Each checkpoint captures the complete workflow state after an "
        "agent completion or quality gate evaluation."
    ),
    responses={
        200: {"description": "Checkpoints retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def list_checkpoints(
    request: Request,
    workflow_id: str,
    pagination: PaginationParams = Depends(get_pagination),
    phase: Optional[str] = Query(
        default=None,
        description="Filter by phase: information_gathering, risk_assessment, risk_mitigation, package_generation",
    ),
    user: AuthUser = Depends(require_permission("eudr-ddo:checkpoints:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """List checkpoints for a workflow.

    Returns all checkpoints ordered by sequence number, with optional
    filtering by phase and pagination support.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        pagination: Pagination parameters.
        phase: Optional phase filter.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with checkpoints list and pagination metadata.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "list_checkpoints: user=%s workflow_id=%s phase=%s",
        user.user_id,
        workflow_id,
        phase,
    )

    service = get_ddo_service()
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    checkpoints = state.checkpoints or []

    # Filter by phase if specified
    if phase:
        checkpoints = [
            cp for cp in checkpoints
            if (cp.phase.value if hasattr(cp.phase, "value") else str(cp.phase)) == phase
        ]

    # Sort by sequence number
    checkpoints = sorted(checkpoints, key=lambda cp: cp.sequence_number)

    # Apply pagination
    total = len(checkpoints)
    paginated = checkpoints[pagination.offset: pagination.offset + pagination.limit]

    checkpoint_summaries = []
    for cp in paginated:
        checkpoint_summaries.append({
            "checkpoint_id": cp.checkpoint_id,
            "sequence_number": cp.sequence_number,
            "phase": cp.phase.value if hasattr(cp.phase, "value") else str(cp.phase),
            "agent_id": cp.agent_id,
            "gate_id": cp.gate_id,
            "agents_completed": sum(
                1 for s in cp.agent_statuses.values()
                if s == "completed"
            ),
            "cumulative_provenance_hash": cp.cumulative_provenance_hash[:16] + "..." if cp.cumulative_provenance_hash else None,
            "created_at": cp.created_at.isoformat() if cp.created_at else None,
            "created_by": cp.created_by,
        })

    return {
        "workflow_id": workflow_id,
        "checkpoints": checkpoint_summaries,
        "meta": {
            "total": total,
            "limit": pagination.limit,
            "offset": pagination.offset,
            "has_more": (pagination.offset + pagination.limit) < total,
        },
    }


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/checkpoints -- Create manual checkpoint
# ---------------------------------------------------------------------------


@router.post(
    "/workflows/{workflow_id}/checkpoints",
    status_code=status.HTTP_201_CREATED,
    summary="Create a manual checkpoint",
    description=(
        "Create a manual checkpoint for the current workflow state. "
        "Manual checkpoints supplement the automatic checkpoints created "
        "after each agent completion and quality gate evaluation. Useful "
        "before making configuration changes or risky operations."
    ),
    responses={
        201: {"description": "Checkpoint created successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def create_checkpoint(
    request: Request,
    workflow_id: str,
    reason: Optional[str] = Query(
        default=None,
        description="Reason for creating the manual checkpoint",
    ),
    user: AuthUser = Depends(require_permission("eudr-ddo:checkpoints:rollback")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> Dict[str, Any]:
    """Create a manual checkpoint.

    Captures the current workflow state as a checkpoint with a cumulative
    provenance hash. The checkpoint can be used for rollback or resume
    operations.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        reason: Optional reason for creating the checkpoint.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with checkpoint details.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "create_checkpoint: user=%s workflow_id=%s reason=%s",
        user.user_id,
        workflow_id,
        reason,
    )

    service = get_ddo_service()
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    try:
        checkpoint = service._state_manager.create_checkpoint(
            workflow_id=workflow_id,
            reason=reason or f"Manual checkpoint by {user.user_id}",
        )

        cp_id = checkpoint.checkpoint_id if hasattr(checkpoint, "checkpoint_id") else str(checkpoint)
        cp_seq = checkpoint.sequence_number if hasattr(checkpoint, "sequence_number") else 0
        cp_hash = checkpoint.cumulative_provenance_hash if hasattr(checkpoint, "cumulative_provenance_hash") else ""

        return {
            "checkpoint_id": cp_id,
            "workflow_id": workflow_id,
            "sequence_number": cp_seq,
            "type": "manual",
            "reason": reason,
            "cumulative_provenance_hash": cp_hash,
            "created_at": utcnow().isoformat(),
            "created_by": user.user_id,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create checkpoint: {str(exc)[:500]}",
        )


# ---------------------------------------------------------------------------
# GET /checkpoints/{checkpoint_id} -- Get checkpoint details
# ---------------------------------------------------------------------------


@router.get(
    "/checkpoints/{checkpoint_id}",
    status_code=status.HTTP_200_OK,
    summary="Get checkpoint details by ID",
    description=(
        "Get the full details of a specific checkpoint including all "
        "agent statuses, output references, quality gate results, and "
        "the cumulative SHA-256 provenance hash at that point in time."
    ),
    responses={
        200: {"description": "Checkpoint retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Checkpoint not found"},
    },
)
async def get_checkpoint(
    request: Request,
    checkpoint_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:checkpoints:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get checkpoint details by ID.

    Returns the complete checkpoint state including all agent statuses
    at the time of the checkpoint, output references, quality gate
    results, and the cumulative provenance hash for integrity
    verification.

    Args:
        request: FastAPI request object.
        checkpoint_id: Unique checkpoint identifier.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with full checkpoint state details.

    Raises:
        HTTPException: 404 if checkpoint not found.
    """
    logger.info(
        "get_checkpoint: user=%s checkpoint_id=%s",
        user.user_id,
        checkpoint_id,
    )

    service = get_ddo_service()

    # Search across all workflows for this checkpoint
    checkpoint = service._state_manager.get_checkpoint(checkpoint_id)
    if checkpoint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint {checkpoint_id} not found",
        )

    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "workflow_id": checkpoint.workflow_id,
        "sequence_number": checkpoint.sequence_number,
        "phase": checkpoint.phase.value if hasattr(checkpoint.phase, "value") else str(checkpoint.phase),
        "agent_id": checkpoint.agent_id,
        "gate_id": checkpoint.gate_id,
        "agent_statuses": checkpoint.agent_statuses,
        "agent_outputs": checkpoint.agent_outputs,
        "quality_gate_results": checkpoint.quality_gate_results,
        "cumulative_provenance_hash": checkpoint.cumulative_provenance_hash,
        "created_at": checkpoint.created_at.isoformat() if checkpoint.created_at else None,
        "created_by": checkpoint.created_by,
    }
