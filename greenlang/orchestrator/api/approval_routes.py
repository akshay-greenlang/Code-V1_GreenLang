# -*- coding: utf-8 -*-
"""
FR-043: Approval API Routes for GL-FOUND-X-001
==============================================

FastAPI routes for the signed approvals/attestations feature.

Endpoints:
    - POST /v1/runs/{run_id}/steps/{step_id}/approve - Submit approval
    - GET /v1/runs/{run_id}/approvals - List pending approvals
    - GET /v1/approvals/{approval_id} - Get approval status

Author: GreenLang Team
Version: 1.0.0
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from greenlang.orchestrator.api.deps import (
    AuthContext,
    RequestTrace,
    get_api_key,
    get_request_trace,
    get_orchestrator,
    get_approval_workflow,
)
from greenlang.orchestrator.api.models import (
    ErrorDetail,
    ErrorResponse,
    ApprovalStatusEnum,
    ApprovalDecisionEnum,
    ApprovalSubmitRequest,
    ApprovalRequestResponse,
    ApprovalAttestationResponse,
    ApprovalListResponse,
    ApprovalSubmitResponse,
)
from greenlang.orchestrator.governance.approvals import (
    ApprovalDecision,
    ApprovalStatus,
    SignatureUtils,
)

logger = logging.getLogger(__name__)

# Create approval router
approval_router = APIRouter(prefix="/approvals", tags=["Approvals"])

# Error codes
APPROVAL_NOT_FOUND = "GL-E-APR-001"
APPROVAL_ALREADY_DECIDED = "GL-E-APR-002"
APPROVAL_EXPIRED = "GL-E-APR-003"
SIGNATURE_INVALID = "GL-E-APR-004"


def create_error_response(error_type: str, message: str, details=None, trace_id=None):
    """Create standardized error response."""
    return ErrorResponse(
        error=error_type,
        message=message,
        details=details or [],
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc),
    )


def _to_approval_response(request) -> ApprovalRequestResponse:
    """Convert ApprovalRequest to API response."""
    attestation = None
    if request.attestation:
        attestation = ApprovalAttestationResponse(
            approver_id=request.attestation.approver_id,
            approver_name=request.attestation.approver_name,
            approver_role=request.attestation.approver_role,
            decision=ApprovalDecisionEnum(request.attestation.decision.value),
            reason=request.attestation.reason,
            timestamp=request.attestation.timestamp,
            signature=request.attestation.signature[:32] + "..." if request.attestation.signature else "",
            attestation_hash=request.attestation.attestation_hash,
            signature_valid=None,
        )

    return ApprovalRequestResponse(
        request_id=request.request_id,
        run_id=request.run_id,
        step_id=request.step_id,
        approval_type=request.approval_type.value,
        reason=request.reason,
        requested_by=request.requested_by,
        requested_at=request.requested_at,
        deadline=request.deadline,
        status=ApprovalStatusEnum(request.status.value),
        attestation=attestation,
        provenance_hash=request.provenance_hash,
    )


@approval_router.post(
    "/{approval_id}/decide",
    response_model=ApprovalSubmitResponse,
    summary="Submit approval decision",
    description="Submit an approval or rejection with cryptographic signature.",
    responses={
        200: {"description": "Approval submitted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Approval not found"},
        409: {"model": ErrorResponse, "description": "Already decided"},
    },
)
async def submit_approval_by_id(
    approval_id: str,
    request: ApprovalSubmitRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    workflow=Depends(get_approval_workflow),
) -> ApprovalSubmitResponse:
    """Submit an approval decision with Ed25519 signature."""
    logger.info(f"Submitting approval: {approval_id} [{trace.trace_id}]")

    try:
        decision = ApprovalDecision.APPROVED if request.decision == ApprovalDecisionEnum.APPROVED else ApprovalDecision.REJECTED
        private_key = SignatureUtils.base64_to_bytes(request.signature)
        public_key = SignatureUtils.base64_to_bytes(request.public_key)

        attestation = await workflow.submit_approval(
            approval_id=approval_id,
            approver_id=auth.user_id or "unknown",
            decision=decision,
            private_key=private_key,
            public_key=public_key,
            reason=request.reason,
            approver_name=request.approver_name,
            approver_role=request.approver_role,
        )

        return ApprovalSubmitResponse(
            request_id=approval_id,
            status=ApprovalStatusEnum.APPROVED if decision == ApprovalDecision.APPROVED else ApprovalStatusEnum.REJECTED,
            attestation=ApprovalAttestationResponse(
                approver_id=attestation.approver_id,
                approver_name=attestation.approver_name,
                approver_role=attestation.approver_role,
                decision=ApprovalDecisionEnum(attestation.decision.value),
                reason=attestation.reason,
                timestamp=attestation.timestamp,
                signature=attestation.signature[:32] + "...",
                attestation_hash=attestation.attestation_hash,
                signature_valid=True,
            ),
            message=f"Approval {decision.value} successfully recorded",
        )

    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response("not_found", error_msg, [ErrorDetail(code=APPROVAL_NOT_FOUND, message=error_msg)], trace.trace_id).model_dump(),
            )
        elif "already decided" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=create_error_response("conflict", error_msg, [ErrorDetail(code=APPROVAL_ALREADY_DECIDED, message=error_msg)], trace.trace_id).model_dump(),
            )
        elif "expired" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=create_error_response("conflict", error_msg, [ErrorDetail(code=APPROVAL_EXPIRED, message=error_msg)], trace.trace_id).model_dump(),
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=create_error_response("bad_request", error_msg, trace_id=trace.trace_id).model_dump())
    except Exception as e:
        logger.error(f"Failed to submit approval: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=create_error_response("internal_error", "Failed to submit approval", trace_id=trace.trace_id).model_dump())


@approval_router.get(
    "/{approval_id}",
    response_model=ApprovalRequestResponse,
    summary="Get approval status",
    description="Get the status and details of an approval request.",
    responses={
        200: {"description": "Approval details"},
        404: {"model": ErrorResponse, "description": "Approval not found"},
    },
)
async def get_approval(
    approval_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    workflow=Depends(get_approval_workflow),
) -> ApprovalRequestResponse:
    """Get approval request details."""
    logger.debug(f"Getting approval: {approval_id} [{trace.trace_id}]")

    try:
        request = await workflow.get_approval(approval_id)
        if not request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response("not_found", f"Approval not found: {approval_id}", [ErrorDetail(code=APPROVAL_NOT_FOUND, message=f"Approval {approval_id} not found")], trace.trace_id).model_dump(),
            )
        return _to_approval_response(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get approval: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=create_error_response("internal_error", "Failed to get approval", trace_id=trace.trace_id).model_dump())


@approval_router.get(
    "/{approval_id}/verify",
    response_model=dict,
    summary="Verify attestation signature",
    description="Cryptographically verify the signature on an approval attestation.",
    responses={
        200: {"description": "Verification result"},
        404: {"model": ErrorResponse, "description": "Approval not found"},
    },
)
async def verify_attestation(
    approval_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    workflow=Depends(get_approval_workflow),
) -> dict:
    """Verify the cryptographic signature on an attestation."""
    logger.debug(f"Verifying attestation: {approval_id} [{trace.trace_id}]")

    try:
        is_valid = await workflow.verify_attestation(approval_id)
        return {"approval_id": approval_id, "signature_valid": is_valid, "verified_at": datetime.now(timezone.utc).isoformat()}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=create_error_response("not_found", str(e), trace_id=trace.trace_id).model_dump())
    except Exception as e:
        logger.error(f"Failed to verify attestation: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=create_error_response("internal_error", "Failed to verify attestation", trace_id=trace.trace_id).model_dump())


# Run-scoped approval endpoints
run_approval_router = APIRouter(tags=["Run Approvals"])


@run_approval_router.get(
    "/runs/{run_id}/approvals",
    response_model=ApprovalListResponse,
    summary="List pending approvals for run",
    description="Get all pending approval requests for a pipeline run.",
    responses={
        200: {"description": "List of approvals"},
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
)
async def list_run_approvals(
    run_id: str,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    orchestrator=Depends(get_orchestrator),
    workflow=Depends(get_approval_workflow),
) -> ApprovalListResponse:
    """List all pending approvals for a run."""
    logger.debug(f"Listing approvals for run: {run_id} [{trace.trace_id}]")

    try:
        run = await orchestrator.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=create_error_response("not_found", f"Run not found: {run_id}", trace_id=trace.trace_id).model_dump())

        requests = await workflow.get_pending_approvals(run_id)
        return ApprovalListResponse(
            approvals=[_to_approval_response(r) for r in requests],
            total=len(requests),
            run_id=run_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list approvals: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=create_error_response("internal_error", "Failed to list approvals", trace_id=trace.trace_id).model_dump())


@run_approval_router.post(
    "/runs/{run_id}/steps/{step_id}/approve",
    response_model=ApprovalSubmitResponse,
    summary="Submit step approval",
    description="Submit an approval decision for a specific step with cryptographic signature.",
    responses={
        200: {"description": "Approval submitted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Approval not found"},
        409: {"model": ErrorResponse, "description": "Already decided or expired"},
    },
)
async def submit_step_approval(
    run_id: str,
    step_id: str,
    request: ApprovalSubmitRequest,
    auth: AuthContext = Depends(get_api_key),
    trace: RequestTrace = Depends(get_request_trace),
    workflow=Depends(get_approval_workflow),
) -> ApprovalSubmitResponse:
    """Submit approval for a specific step."""
    logger.info(f"Submitting step approval: {run_id}/{step_id} [{trace.trace_id}]")

    try:
        approval_request = await workflow.get_step_approval(run_id, step_id)
        if not approval_request:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=create_error_response("not_found", f"No approval pending for step {step_id}", [ErrorDetail(code=APPROVAL_NOT_FOUND, message=f"No approval found for run {run_id} step {step_id}")], trace.trace_id).model_dump())

        decision = ApprovalDecision.APPROVED if request.decision == ApprovalDecisionEnum.APPROVED else ApprovalDecision.REJECTED
        private_key = SignatureUtils.base64_to_bytes(request.signature)
        public_key = SignatureUtils.base64_to_bytes(request.public_key)

        attestation = await workflow.submit_approval(
            approval_id=approval_request.request_id,
            approver_id=auth.user_id or "unknown",
            decision=decision,
            private_key=private_key,
            public_key=public_key,
            reason=request.reason,
            approver_name=request.approver_name,
            approver_role=request.approver_role,
        )

        return ApprovalSubmitResponse(
            request_id=approval_request.request_id,
            status=ApprovalStatusEnum.APPROVED if decision == ApprovalDecision.APPROVED else ApprovalStatusEnum.REJECTED,
            attestation=ApprovalAttestationResponse(
                approver_id=attestation.approver_id,
                approver_name=attestation.approver_name,
                approver_role=attestation.approver_role,
                decision=ApprovalDecisionEnum(attestation.decision.value),
                reason=attestation.reason,
                timestamp=attestation.timestamp,
                signature=attestation.signature[:32] + "...",
                attestation_hash=attestation.attestation_hash,
                signature_valid=True,
            ),
            message=f"Step {step_id} approval {decision.value} successfully recorded",
        )

    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        if "expired" in error_msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=create_error_response("conflict", error_msg, [ErrorDetail(code=APPROVAL_EXPIRED, message=error_msg)], trace.trace_id).model_dump())
        elif "already decided" in error_msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=create_error_response("conflict", error_msg, [ErrorDetail(code=APPROVAL_ALREADY_DECIDED, message=error_msg)], trace.trace_id).model_dump())
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=create_error_response("bad_request", error_msg, trace_id=trace.trace_id).model_dump())
    except Exception as e:
        logger.error(f"Failed to submit step approval: {e} [{trace.trace_id}]", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=create_error_response("internal_error", "Failed to submit approval", trace_id=trace.trace_id).model_dump())


__all__ = ["approval_router", "run_approval_router"]
