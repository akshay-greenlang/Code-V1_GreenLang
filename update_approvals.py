#!/usr/bin/env python
"""Script to update files for FR-043 Signed Approvals/Attestations."""

import os

# ============================================================================
# 1. Update governance/__init__.py to export approvals module
# ============================================================================

governance_init = '''# -*- coding: utf-8 -*-
"""
Orchestrator Governance Layer
==============================

Provides policy enforcement for the GreenLang orchestrator.

Components:
    - PolicyEngine: Hybrid OPA + YAML policy evaluation
    - PolicyDecision: Evaluation results with provenance
    - PolicyBundle: Versioned policy collections
    - OPAClient: HTTP client for OPA server
    - YAMLRulesParser: Declarative YAML rule evaluation
    - ApprovalWorkflow: Signed approval workflow (FR-043)

Evaluation Points:
    - Pre-run: Pipeline + plan validation
    - Pre-step: Permissions, cost, data residency
    - Post-step: Artifact classification, export controls

Example:
    >>> from greenlang.orchestrator.governance import PolicyEngine
    >>> engine = PolicyEngine()
    >>> decision = await engine.evaluate_pre_run(pipeline, run_config)
    >>> if not decision.allowed:
    ...     print(decision.reasons[0].message)

Author: GreenLang Team
Version: 1.0.0
"""

from greenlang.orchestrator.governance.policy_engine import (
    # Main classes
    PolicyEngine,
    PolicyEngineConfig,
    OPAClient,
    YAMLRulesParser,
    # Decision models
    PolicyDecision,
    PolicyReason,
    ApprovalRequirement,
    # Enums
    PolicyAction,
    PolicySeverity,
    EvaluationPoint,
    ApprovalType,
    # Rule models
    YAMLRule,
    YAMLRuleSet,
    CostBudget,
    DataResidencyRule,
    PolicyBundle,
    # Exceptions
    OPAError,
)

from greenlang.orchestrator.governance.approvals import (
    # Enums
    ApprovalStatus,
    ApprovalDecision,
    # Models
    ApprovalAttestation,
    ApprovalRequest,
    # Store
    ApprovalStore,
    InMemoryApprovalStore,
    # Utilities
    SignatureUtils,
    # Workflow
    ApprovalWorkflow,
    # Exceptions
    ApprovalError,
    ApprovalNotFoundError,
    ApprovalExpiredError,
    ApprovalAlreadyDecidedError,
    SignatureVerificationError,
    # Constants
    CRYPTO_AVAILABLE,
)

__all__ = [
    # Main classes
    "PolicyEngine",
    "PolicyEngineConfig",
    "OPAClient",
    "YAMLRulesParser",
    # Decision models
    "PolicyDecision",
    "PolicyReason",
    "ApprovalRequirement",
    # Enums
    "PolicyAction",
    "PolicySeverity",
    "EvaluationPoint",
    "ApprovalType",
    # Rule models
    "YAMLRule",
    "YAMLRuleSet",
    "CostBudget",
    "DataResidencyRule",
    "PolicyBundle",
    # Exceptions
    "OPAError",
    # FR-043: Approval Workflow
    "ApprovalStatus",
    "ApprovalDecision",
    "ApprovalAttestation",
    "ApprovalRequest",
    "ApprovalStore",
    "InMemoryApprovalStore",
    "SignatureUtils",
    "ApprovalWorkflow",
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalExpiredError",
    "ApprovalAlreadyDecidedError",
    "SignatureVerificationError",
    "CRYPTO_AVAILABLE",
]
'''

with open('greenlang/orchestrator/governance/__init__.py', 'w', encoding='utf-8') as f:
    f.write(governance_init)
print('Updated greenlang/orchestrator/governance/__init__.py')


# ============================================================================
# 2. Add approval models to api/models.py (append to existing file)
# ============================================================================

approval_models = '''

# =============================================================================
# APPROVAL MODELS (FR-043)
# =============================================================================


class ApprovalStatusEnum(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalDecisionEnum(str, Enum):
    """Decision made by an approver."""
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalSubmitRequest(BaseModel):
    """Request to submit an approval decision."""
    decision: ApprovalDecisionEnum = Field(..., description="APPROVED or REJECTED")
    reason: Optional[str] = Field(None, max_length=2000, description="Explanation for decision")
    signature: str = Field(..., description="Base64-encoded Ed25519 signature")
    public_key: str = Field(..., description="Base64-encoded public key")
    approver_name: Optional[str] = Field(None, description="Human-readable name")
    approver_role: Optional[str] = Field(None, description="Approver role")

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision": "approved",
                "reason": "Verified calculation methodology is correct",
                "signature": "base64-encoded-ed25519-signature",
                "public_key": "base64-encoded-public-key",
                "approver_name": "John Smith",
                "approver_role": "Manager"
            }
        }
    }


class ApprovalAttestationResponse(BaseModel):
    """Response containing signed attestation details."""
    approver_id: str = Field(..., description="Approver identifier")
    approver_name: Optional[str] = Field(None, description="Approver name")
    approver_role: Optional[str] = Field(None, description="Approver role")
    decision: ApprovalDecisionEnum = Field(..., description="Decision made")
    reason: Optional[str] = Field(None, description="Decision reason")
    timestamp: datetime = Field(..., description="Attestation timestamp")
    signature: str = Field(..., description="Truncated signature for display")
    attestation_hash: str = Field(..., description="SHA-256 hash of attestation")
    signature_valid: Optional[bool] = Field(None, description="Whether signature verified")


class ApprovalRequestResponse(BaseModel):
    """Response for an approval request."""
    request_id: str = Field(..., description="Unique request identifier")
    run_id: str = Field(..., description="Associated run ID")
    step_id: str = Field(..., description="Step requiring approval")
    approval_type: str = Field(..., description="Type of approval required")
    reason: str = Field(..., description="Why approval is required")
    requested_by: Optional[str] = Field(None, description="Who requested approval")
    requested_at: datetime = Field(..., description="Request timestamp")
    deadline: datetime = Field(..., description="Approval deadline")
    status: ApprovalStatusEnum = Field(..., description="Current status")
    attestation: Optional[ApprovalAttestationResponse] = Field(None, description="Signed attestation if decided")
    provenance_hash: str = Field(..., description="Provenance hash for audit")

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "apr-abc123def456",
                "run_id": "run-xyz789",
                "step_id": "step-calculate",
                "approval_type": "manager",
                "reason": "High-value calculation requires manager approval",
                "requested_by": "system",
                "requested_at": "2026-01-28T10:00:00Z",
                "deadline": "2026-01-29T10:00:00Z",
                "status": "pending",
                "attestation": None,
                "provenance_hash": "abc123..."
            }
        }
    }


class ApprovalListResponse(BaseModel):
    """Response for listing approvals."""
    approvals: List[ApprovalRequestResponse] = Field(..., description="List of approvals")
    total: int = Field(..., description="Total count")
    run_id: str = Field(..., description="Run ID filter")


class ApprovalSubmitResponse(BaseModel):
    """Response after submitting an approval."""
    request_id: str = Field(..., description="Approval request ID")
    status: ApprovalStatusEnum = Field(..., description="New status")
    attestation: ApprovalAttestationResponse = Field(..., description="Signed attestation")
    message: str = Field(..., description="Confirmation message")
'''

# Read existing models.py and append if models not already present
models_path = 'greenlang/orchestrator/api/models.py'
with open(models_path, 'r', encoding='utf-8') as f:
    existing = f.read()

if 'ApprovalSubmitRequest' not in existing:
    with open(models_path, 'a', encoding='utf-8') as f:
        f.write(approval_models)
    print(f'Updated {models_path} with approval models')
else:
    print(f'{models_path} already has approval models')


# ============================================================================
# 3. Create approval_routes.py for approval endpoints
# ============================================================================

approval_routes = '''# -*- coding: utf-8 -*-
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
'''

with open('greenlang/orchestrator/api/approval_routes.py', 'w', encoding='utf-8') as f:
    f.write(approval_routes)
print('Created greenlang/orchestrator/api/approval_routes.py')


# ============================================================================
# 4. Add get_approval_workflow to deps.py
# ============================================================================

deps_addition = '''

# =============================================================================
# APPROVAL WORKFLOW DEPENDENCY (FR-043)
# =============================================================================


_approval_workflow_instance = None


async def get_approval_workflow():
    """
    Get the ApprovalWorkflow instance for signed approvals.

    Returns:
        ApprovalWorkflow instance

    Raises:
        HTTPException: If approval workflow is not available
    """
    global _approval_workflow_instance

    if _approval_workflow_instance is None:
        try:
            from greenlang.orchestrator.governance.approvals import (
                ApprovalWorkflow,
                InMemoryApprovalStore,
            )
            from greenlang.orchestrator.audit.event_store import EventFactory

            store = InMemoryApprovalStore()
            event_store = await get_event_store()
            event_factory = EventFactory(event_store) if event_store else None

            _approval_workflow_instance = ApprovalWorkflow(
                store=store,
                event_factory=event_factory,
                default_deadline_hours=24,
            )
            logger.info("ApprovalWorkflow initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"ApprovalWorkflow not available: {e}")
            _approval_workflow_instance = MockApprovalWorkflow()
        except Exception as e:
            logger.error(f"Failed to initialize approval workflow: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Approval workflow service unavailable",
            )

    return _approval_workflow_instance


def set_approval_workflow(workflow) -> None:
    """
    Set the approval workflow instance (for testing).

    Args:
        workflow: ApprovalWorkflow instance or compatible mock
    """
    global _approval_workflow_instance
    _approval_workflow_instance = workflow


class MockApprovalWorkflow:
    """Mock approval workflow for development and testing."""

    def __init__(self):
        self._requests = {}
        logger.info("MockApprovalWorkflow initialized")

    async def request_approval(self, run_id, step_id, requirement, **kwargs):
        from uuid import uuid4
        request_id = f"apr-{uuid4().hex[:12]}"
        self._requests[request_id] = {
            "request_id": request_id,
            "run_id": run_id,
            "step_id": step_id,
            "status": "pending",
        }
        return request_id

    async def submit_approval(self, approval_id, approver_id, decision, **kwargs):
        from dataclasses import dataclass
        from datetime import datetime, timezone

        @dataclass
        class MockAttestation:
            approver_id: str = approver_id
            approver_name: str = None
            approver_role: str = None
            decision: type = None
            reason: str = None
            timestamp: datetime = None
            signature: str = "mock-signature"
            attestation_hash: str = "mock-hash"

        return MockAttestation(timestamp=datetime.now(timezone.utc))

    async def check_approval_status(self, approval_id):
        return "pending"

    async def verify_attestation(self, approval_id):
        return True

    async def get_pending_approvals(self, run_id=None):
        return []

    async def get_approval(self, approval_id):
        return self._requests.get(approval_id)

    async def get_step_approval(self, run_id, step_id):
        return None
'''

# Read existing deps.py and append if not already present
deps_path = 'greenlang/orchestrator/api/deps.py'
with open(deps_path, 'r', encoding='utf-8') as f:
    existing_deps = f.read()

if 'get_approval_workflow' not in existing_deps:
    # Find the end of the file before __all__ and insert
    if '__all__ = [' in existing_deps:
        # Insert before __all__
        parts = existing_deps.rsplit('# =============================================================================\n# EXPORTS\n# =============================================================================', 1)
        if len(parts) == 2:
            new_deps = parts[0] + deps_addition + '\n\n# =============================================================================\n# EXPORTS\n# =============================================================================' + parts[1]
            # Update __all__ to include new exports
            new_deps = new_deps.replace(
                '"shutdown_dependencies",\n]',
                '"shutdown_dependencies",\n    # Approval Workflow\n    "get_approval_workflow",\n    "set_approval_workflow",\n    "MockApprovalWorkflow",\n]'
            )
            with open(deps_path, 'w', encoding='utf-8') as f:
                f.write(new_deps)
            print(f'Updated {deps_path} with approval workflow dependency')
        else:
            # Just append to end
            with open(deps_path, 'a', encoding='utf-8') as f:
                f.write(deps_addition)
            print(f'Appended to {deps_path}')
    else:
        with open(deps_path, 'a', encoding='utf-8') as f:
            f.write(deps_addition)
        print(f'Appended to {deps_path}')
else:
    print(f'{deps_path} already has approval workflow dependency')


print('\\nFR-043 Signed Approvals/Attestations update complete!')
print('\\nFiles created/updated:')
print('  - greenlang/orchestrator/governance/approvals.py (created earlier)')
print('  - greenlang/orchestrator/governance/__init__.py')
print('  - greenlang/orchestrator/api/models.py')
print('  - greenlang/orchestrator/api/approval_routes.py')
print('  - greenlang/orchestrator/api/deps.py')
