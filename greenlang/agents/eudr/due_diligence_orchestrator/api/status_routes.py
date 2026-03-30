# -*- coding: utf-8 -*-
"""
Status Monitoring Routes - AGENT-EUDR-026 Due Diligence Orchestrator API

Endpoints for real-time monitoring of workflow execution status, progress
tracking at the per-agent granularity, phase-level status across the
three due diligence phases, and estimated time to completion (ETA).

Endpoints (4):
    GET /workflows/{id}/status       - Get workflow execution status
    GET /workflows/{id}/progress     - Get detailed per-agent progress
    GET /workflows/{id}/phase-status - Get per-phase completion status
    GET /workflows/{id}/eta          - Get estimated time to completion

RBAC Permissions:
    eudr-ddo:workflows:read  - View status, progress, phase-status, ETA

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_ddo_service,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    AGENT_NAMES,
    ALL_EUDR_AGENTS,
    AgentExecutionStatus,
    DueDiligencePhase,
    PHASE_1_AGENTS,
    PHASE_2_AGENTS,
    PHASE_3_AGENTS,
    WorkflowProgressResponse,
    WorkflowStatus,
    WorkflowStatusResponse,
    _utcnow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Status Monitoring"])


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/status -- Workflow execution status
# ---------------------------------------------------------------------------


@router.get(
    "/{workflow_id}/status",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get workflow execution status",
    description=(
        "Get the current execution status of a workflow including "
        "overall status, current phase, agent completion counts, "
        "and processing time."
    ),
    responses={
        200: {"description": "Status retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def get_workflow_status(
    request: Request,
    workflow_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> WorkflowStatusResponse:
    """Get workflow execution status.

    Returns the current overall status, phase, agent completion metrics,
    and estimated time to completion for the specified workflow.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        user: Authenticated and authorized user.

    Returns:
        WorkflowStatusResponse with current execution status.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "get_workflow_status: user=%s workflow_id=%s",
        user.user_id,
        workflow_id,
    )

    try:
        service = get_ddo_service()
        return service.get_workflow_status(workflow_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/progress -- Per-agent progress
# ---------------------------------------------------------------------------


@router.get(
    "/{workflow_id}/progress",
    response_model=WorkflowProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Get detailed per-agent progress",
    description=(
        "Get detailed progress information including the execution "
        "status of each individual EUDR agent (25 agents), quality "
        "gate results, composite risk score, and execution timeline."
    ),
    responses={
        200: {"description": "Progress retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def get_workflow_progress(
    request: Request,
    workflow_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> WorkflowProgressResponse:
    """Get detailed per-agent progress.

    Returns the execution status of each of the 25 EUDR agents, quality
    gate evaluation results, and execution timeline data for Gantt chart
    visualization.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        user: Authenticated and authorized user.

    Returns:
        WorkflowProgressResponse with per-agent status details.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "get_workflow_progress: user=%s workflow_id=%s",
        user.user_id,
        workflow_id,
    )

    try:
        service = get_ddo_service()
        return service.get_workflow_progress(workflow_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/phase-status -- Per-phase status
# ---------------------------------------------------------------------------


@router.get(
    "/{workflow_id}/phase-status",
    status_code=status.HTTP_200_OK,
    summary="Get per-phase completion status",
    description=(
        "Get the completion status of each due diligence phase: "
        "Phase 1 (Information Gathering, Art. 9), "
        "Phase 2 (Risk Assessment, Art. 10), "
        "Phase 3 (Risk Mitigation, Art. 11), "
        "and Package Generation (Art. 12)."
    ),
    responses={
        200: {"description": "Phase status retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def get_phase_status(
    request: Request,
    workflow_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get per-phase completion status.

    Computes the completion status and agent metrics for each of the
    four workflow phases. Includes per-phase agent counts, quality gate
    status, and completeness scores.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with per-phase status breakdown.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "get_phase_status: user=%s workflow_id=%s",
        user.user_id,
        workflow_id,
    )

    service = get_ddo_service()
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    def _phase_summary(phase_name: str, agent_ids: List[str], gate_id: Optional[str]) -> Dict[str, Any]:
        """Build summary for a single phase."""
        completed = 0
        running = 0
        failed = 0
        pending = 0

        for aid in agent_ids:
            record = state.agent_executions.get(aid)
            if record is None:
                pending += 1
            elif record.status == AgentExecutionStatus.COMPLETED:
                completed += 1
            elif record.status == AgentExecutionStatus.RUNNING:
                running += 1
            elif record.status in (
                AgentExecutionStatus.FAILED,
                AgentExecutionStatus.CIRCUIT_BROKEN,
                AgentExecutionStatus.TIMED_OUT,
            ):
                failed += 1
            else:
                pending += 1

        total = len(agent_ids)
        progress = (Decimal(str(completed)) / Decimal(str(total)) * Decimal("100")).quantize(
            Decimal("0.01")
        ) if total > 0 else Decimal("0")

        # Quality gate status
        gate_result = None
        if gate_id and gate_id in state.quality_gates:
            gate_eval = state.quality_gates[gate_id]
            gate_result = {
                "gate_id": gate_id,
                "result": gate_eval.result.value if hasattr(gate_eval.result, "value") else str(gate_eval.result),
                "score": str(gate_eval.weighted_score),
                "threshold": str(gate_eval.threshold),
            }

        return {
            "phase": phase_name,
            "agents_total": total,
            "agents_completed": completed,
            "agents_running": running,
            "agents_failed": failed,
            "agents_pending": pending,
            "progress_pct": str(progress),
            "quality_gate": gate_result,
            "agent_details": [
                {
                    "agent_id": aid,
                    "agent_name": AGENT_NAMES.get(aid, aid),
                    "status": (
                        state.agent_executions[aid].status.value
                        if aid in state.agent_executions
                        else "pending"
                    ),
                }
                for aid in agent_ids
            ],
        }

    phases = {
        "information_gathering": _phase_summary(
            "Information Gathering (Article 9)",
            PHASE_1_AGENTS,
            "QG-1",
        ),
        "risk_assessment": _phase_summary(
            "Risk Assessment (Article 10)",
            PHASE_2_AGENTS,
            "QG-2",
        ),
        "risk_mitigation": _phase_summary(
            "Risk Mitigation (Article 11)",
            PHASE_3_AGENTS,
            "QG-3",
        ),
    }

    return {
        "workflow_id": workflow_id,
        "status": state.status.value if hasattr(state.status, "value") else str(state.status),
        "current_phase": state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase),
        "phases": phases,
        "retrieved_at": utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/eta -- Estimated time to completion
# ---------------------------------------------------------------------------


@router.get(
    "/{workflow_id}/eta",
    status_code=status.HTTP_200_OK,
    summary="Get estimated time to completion",
    description=(
        "Get the estimated time to completion (ETA) for a running "
        "workflow. Calculated using critical path analysis on the "
        "remaining uncompleted agents in the DAG."
    ),
    responses={
        200: {"description": "ETA retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def get_workflow_eta(
    request: Request,
    workflow_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get estimated time to completion.

    Calculates the remaining execution time based on the critical path
    of uncompleted agents in the DAG, using historical execution
    durations for estimation.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with ETA in seconds, completion estimate, and analysis.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "get_workflow_eta: user=%s workflow_id=%s",
        user.user_id,
        workflow_id,
    )

    service = get_ddo_service()
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    # Completed agents
    completed_agents = {
        aid for aid, rec in state.agent_executions.items()
        if rec.status == AgentExecutionStatus.COMPLETED
    }

    total_agents = len(state.agent_executions) or len(ALL_EUDR_AGENTS)
    remaining_count = total_agents - len(completed_agents)

    # Use the parallel engine's ETA calculator
    eta_seconds = state.eta_seconds or 0

    # Estimate completion time if ETA is available
    if eta_seconds > 0:
        from datetime import timedelta
        estimated_completion = (utcnow() + timedelta(seconds=eta_seconds)).isoformat()
    else:
        estimated_completion = None

    return {
        "workflow_id": workflow_id,
        "status": state.status.value if hasattr(state.status, "value") else str(state.status),
        "eta_seconds": eta_seconds,
        "estimated_completion": estimated_completion,
        "agents_completed": len(completed_agents),
        "agents_remaining": remaining_count,
        "agents_total": total_agents,
        "progress_pct": str(state.progress_pct),
        "current_phase": state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase),
        "calculated_at": utcnow().isoformat(),
    }
