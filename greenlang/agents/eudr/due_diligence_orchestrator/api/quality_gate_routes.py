# -*- coding: utf-8 -*-
"""
Quality Gate Routes - AGENT-EUDR-026 Due Diligence Orchestrator API

Endpoints for managing the three quality gates that enforce phase
transition requirements per EUDR Article 8:
    QG-1: Information Gathering Completeness (Art. 9 -> Art. 10)
    QG-2: Risk Assessment Coverage (Art. 10 -> Art. 11)
    QG-3: Mitigation Adequacy (Art. 11 -> Art. 12)

Endpoints (3):
    GET  /workflows/{id}/gates                    - Get all quality gate results
    POST /workflows/{id}/gates/{gate_id}/override - Override a failed gate
    GET  /workflows/{id}/gates/{gate_id}/details  - Get detailed gate evaluation

RBAC Permissions:
    eudr-ddo:gates:read     - View quality gate evaluations
    eudr-ddo:gates:override - Override failed quality gates

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
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
    QualityGateId,
    QualityGateResponse,
    QualityGateResultEnum,
    _utcnow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Quality Gate Management"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class GateOverrideRequest(BaseModel):
    """Request to override a failed quality gate."""

    justification: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description=(
            "Mandatory justification for overriding the quality gate. "
            "Must explain why proceeding is acceptable despite the "
            "gate failure. Recorded in audit trail."
        ),
    )


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/gates -- List all quality gates
# ---------------------------------------------------------------------------


@router.get(
    "/{workflow_id}/gates",
    status_code=status.HTTP_200_OK,
    summary="Get all quality gate results",
    description=(
        "Get the evaluation results for all three quality gates "
        "(QG-1, QG-2, QG-3) for a workflow. Returns the result, "
        "score, threshold, and individual check details for each gate."
    ),
    responses={
        200: {"description": "Quality gates retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def get_quality_gates(
    request: Request,
    workflow_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:gates:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get all quality gate evaluation results.

    Returns the status and evaluation details for each of the three
    quality gates: QG-1 (Information Gathering), QG-2 (Risk Assessment),
    and QG-3 (Mitigation Adequacy).

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with gate results, overall pass status, and details.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "get_quality_gates: user=%s workflow_id=%s",
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

    gates = {}
    gate_ids = [QualityGateId.QG1, QualityGateId.QG2, QualityGateId.QG3]

    for gate_id in gate_ids:
        gate_key = gate_id.value
        evaluation = state.quality_gates.get(gate_key)

        if evaluation is not None:
            gates[gate_key] = {
                "gate_id": gate_key,
                "gate_name": _gate_name(gate_id),
                "result": evaluation.result.value if hasattr(evaluation.result, "value") else str(evaluation.result),
                "weighted_score": str(evaluation.weighted_score),
                "threshold": str(evaluation.threshold),
                "checks_count": len(evaluation.checks),
                "checks_passed": sum(1 for c in evaluation.checks if c.passed),
                "override_justification": evaluation.override_justification,
                "override_by": evaluation.override_by,
                "evaluated_at": evaluation.evaluated_at.isoformat() if evaluation.evaluated_at else None,
            }
        else:
            gates[gate_key] = {
                "gate_id": gate_key,
                "gate_name": _gate_name(gate_id),
                "result": "pending",
                "weighted_score": None,
                "threshold": None,
                "checks_count": 0,
                "checks_passed": 0,
                "override_justification": None,
                "override_by": None,
                "evaluated_at": None,
            }

    # Determine overall quality gate status
    evaluated_gates = [g for g in gates.values() if g["result"] != "pending"]
    all_passed = all(
        g["result"] in ("passed", "overridden")
        for g in evaluated_gates
    ) if evaluated_gates else False

    return {
        "workflow_id": workflow_id,
        "gates": gates,
        "all_gates_passed": all_passed,
        "gates_evaluated": len(evaluated_gates),
        "gates_total": 3,
        "retrieved_at": _utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/gates/{gate_id}/override -- Override gate
# ---------------------------------------------------------------------------


@router.post(
    "/{workflow_id}/gates/{gate_id}/override",
    response_model=QualityGateResponse,
    status_code=status.HTTP_200_OK,
    summary="Override a failed quality gate",
    description=(
        "Override a failed quality gate with a mandatory justification. "
        "This allows the workflow to proceed to the next phase despite "
        "the gate failure. The override is recorded in the audit trail "
        "with full provenance for regulatory inspection."
    ),
    responses={
        200: {"description": "Quality gate overridden successfully"},
        400: {"model": ErrorResponse, "description": "Invalid override request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow or gate not found"},
        409: {"model": ErrorResponse, "description": "Gate has not failed"},
    },
)
async def override_quality_gate(
    request: Request,
    workflow_id: str,
    gate_id: str,
    body: GateOverrideRequest,
    user: AuthUser = Depends(require_permission("eudr-ddo:gates:override")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> QualityGateResponse:
    """Override a failed quality gate.

    Allows a compliance officer or admin to override a failed gate
    with documented justification. The override, justification, and
    user identity are permanently recorded in the audit trail.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        gate_id: Quality gate identifier (QG-1, QG-2, QG-3).
        body: Override request with justification.
        user: Authenticated and authorized user.

    Returns:
        QualityGateResponse with override status.

    Raises:
        HTTPException: 400/409 if gate cannot be overridden.
    """
    logger.warning(
        "override_quality_gate: user=%s workflow_id=%s gate=%s justification=%s",
        user.user_id,
        workflow_id,
        gate_id,
        body.justification[:100],
    )

    # Validate gate_id
    valid_gates = {"QG-1", "QG-2", "QG-3"}
    if gate_id not in valid_gates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid gate_id: {gate_id}. Valid values: {', '.join(sorted(valid_gates))}",
        )

    service = get_ddo_service()

    # Verify workflow exists
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    # Verify gate has failed
    gate_eval = state.quality_gates.get(gate_id)
    if gate_eval is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quality gate {gate_id} has not been evaluated for workflow {workflow_id}",
        )

    if gate_eval.result == QualityGateResultEnum.PASSED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Quality gate {gate_id} has already passed. Override not needed.",
        )

    if gate_eval.result == QualityGateResultEnum.OVERRIDDEN:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Quality gate {gate_id} has already been overridden.",
        )

    try:
        return service.override_quality_gate(
            workflow_id=workflow_id,
            gate_id=gate_id,
            justification=body.justification,
            override_by=user.user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id}/gates/{gate_id}/details -- Gate details
# ---------------------------------------------------------------------------


@router.get(
    "/{workflow_id}/gates/{gate_id}/details",
    status_code=status.HTTP_200_OK,
    summary="Get detailed quality gate evaluation",
    description=(
        "Get the full evaluation details for a specific quality gate "
        "including all individual checks, weights, scores, pass/fail "
        "status, remediation guidance for failed checks, and provenance."
    ),
    responses={
        200: {"description": "Gate details retrieved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid gate ID"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow or gate not found"},
    },
)
async def get_gate_details(
    request: Request,
    workflow_id: str,
    gate_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:gates:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get detailed quality gate evaluation results.

    Returns the complete evaluation details for a gate including
    each individual check, its weight, measured score, threshold,
    pass/fail determination, and remediation guidance for any
    failing checks.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        gate_id: Quality gate identifier (QG-1, QG-2, QG-3).
        user: Authenticated and authorized user.

    Returns:
        Dictionary with full gate evaluation details and checks.

    Raises:
        HTTPException: 404 if workflow or gate evaluation not found.
    """
    logger.info(
        "get_gate_details: user=%s workflow_id=%s gate=%s",
        user.user_id,
        workflow_id,
        gate_id,
    )

    valid_gates = {"QG-1", "QG-2", "QG-3"}
    if gate_id not in valid_gates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid gate_id: {gate_id}. Valid values: {', '.join(sorted(valid_gates))}",
        )

    service = get_ddo_service()
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    evaluation = state.quality_gates.get(gate_id)
    if evaluation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quality gate {gate_id} has not been evaluated for workflow {workflow_id}",
        )

    # Build detailed check results
    checks = []
    for check in evaluation.checks:
        checks.append({
            "check_id": check.check_id,
            "name": check.name,
            "description": check.description,
            "weight": str(check.weight),
            "measured_value": str(check.measured_value),
            "threshold": str(check.threshold),
            "passed": check.passed,
            "source_agents": check.source_agents,
            "remediation": check.remediation,
            "evidence": check.evidence,
        })

    return {
        "workflow_id": workflow_id,
        "gate_id": gate_id,
        "gate_name": _gate_name(QualityGateId(gate_id)),
        "evaluation_id": evaluation.evaluation_id,
        "result": evaluation.result.value if hasattr(evaluation.result, "value") else str(evaluation.result),
        "weighted_score": str(evaluation.weighted_score),
        "threshold": str(evaluation.threshold),
        "phase_from": evaluation.phase_from.value if hasattr(evaluation.phase_from, "value") else str(evaluation.phase_from),
        "phase_to": evaluation.phase_to.value if hasattr(evaluation.phase_to, "value") else str(evaluation.phase_to),
        "checks": checks,
        "checks_total": len(checks),
        "checks_passed": sum(1 for c in checks if c["passed"]),
        "checks_failed": sum(1 for c in checks if not c["passed"]),
        "override_justification": evaluation.override_justification,
        "override_by": evaluation.override_by,
        "evaluated_at": evaluation.evaluated_at.isoformat() if evaluation.evaluated_at else None,
        "provenance_hash": evaluation.provenance_hash,
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _gate_name(gate_id: QualityGateId) -> str:
    """Return human-readable quality gate name."""
    names = {
        QualityGateId.QG1: "Information Gathering Completeness (Art. 9 -> Art. 10)",
        QualityGateId.QG2: "Risk Assessment Coverage (Art. 10 -> Art. 11)",
        QualityGateId.QG3: "Mitigation Adequacy (Art. 11 -> Art. 12)",
    }
    return names.get(gate_id, str(gate_id))
