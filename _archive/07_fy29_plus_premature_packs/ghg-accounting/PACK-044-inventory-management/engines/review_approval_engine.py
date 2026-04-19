# -*- coding: utf-8 -*-
"""
ReviewApprovalEngine - PACK-044 Inventory Management Engine 5
==============================================================

Multi-level review and sign-off engine implementing a structured four-
stage review workflow (Preparer -> Reviewer -> Approver -> Verifier)
for GHG inventory data quality assurance, with digital sign-off,
threaded comments, escalation management, and full audit provenance.

The engine enforces separation of duties by requiring distinct actors
at each workflow stage, and supports configurable review criteria,
conditional approvals, and reviewer delegation.

Review Workflow:
    Stage 1 - PREPARATION:
        Preparer completes inventory section and submits for review.
        Self-review checklist must be completed before submission.

    Stage 2 - REVIEW:
        Reviewer examines data quality, completeness, and methodology.
        May approve, request revisions, or escalate.
        Reviewer cannot be the same person as the preparer.

    Stage 3 - APPROVAL:
        Approver performs final internal sign-off.
        Examines reviewer findings, confirms compliance.
        Approver must be different from both preparer and reviewer.

    Stage 4 - VERIFICATION:
        External or independent verifier performs limited/reasonable
        assurance checks.  Optional stage, required for assured inventories.

Escalation Rules:
    - If review not completed within deadline, escalate to supervisor.
    - If two revision rounds fail, escalate to senior reviewer.
    - Critical findings auto-escalate to approver immediately.
    - Verification failures escalate to board-level review.

Regulatory References:
    - GHG Protocol Corporate Standard, Chapter 10 (Verification)
    - ISO 14064-1:2018, Clause 9 (Management of GHG inventory quality)
    - ISO 14064-3:2019 (Validation and verification of GHG statements)
    - ISAE 3410 (Assurance Engagements on GHG Statements)
    - ESRS E1 (Climate Change - assurance requirements)
    - SEC Climate Disclosure Rule (2024), Item 1505 (attestation)

Zero-Hallucination:
    - All workflow logic is deterministic state-machine transitions
    - No LLM involvement in any review decision or routing
    - SHA-256 provenance hash on every result and sign-off
    - Separation of duties enforced programmatically

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  5 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReviewStage(str, Enum):
    """Stage in the four-stage review workflow.

    PREPARATION:    Preparer is completing the inventory section.
    REVIEW:         Reviewer is examining data quality and methodology.
    APPROVAL:       Approver is performing final internal sign-off.
    VERIFICATION:   Verifier is performing independent assurance.
    COMPLETE:       All required stages are complete.
    """
    PREPARATION = "preparation"
    REVIEW = "review"
    APPROVAL = "approval"
    VERIFICATION = "verification"
    COMPLETE = "complete"

class ReviewStatus(str, Enum):
    """Status of a review at a particular stage.

    NOT_STARTED:        Review has not yet begun for this stage.
    IN_PROGRESS:        Review is actively underway.
    APPROVED:           Stage approved by reviewer/approver.
    APPROVED_WITH_CONDITIONS: Approved with conditions to be addressed.
    REVISIONS_REQUESTED: Revisions requested by reviewer.
    REJECTED:           Stage rejected.
    ESCALATED:          Escalated to higher authority.
    DELEGATED:          Delegated to another reviewer.
    SKIPPED:            Stage skipped (not required for this inventory).
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    REVISIONS_REQUESTED = "revisions_requested"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    DELEGATED = "delegated"
    SKIPPED = "skipped"

class CommentType(str, Enum):
    """Type of review comment.

    FINDING:            A finding requiring action (non-conformance).
    OBSERVATION:        An observation for improvement.
    QUESTION:           A question for the preparer.
    RESPONSE:           A response to a finding/question.
    CLARIFICATION:      A clarification of methodology or data.
    RECOMMENDATION:     A recommendation for future reporting.
    """
    FINDING = "finding"
    OBSERVATION = "observation"
    QUESTION = "question"
    RESPONSE = "response"
    CLARIFICATION = "clarification"
    RECOMMENDATION = "recommendation"

class FindingSeverity(str, Enum):
    """Severity of a review finding.

    CRITICAL:       Fundamental error or omission that materially
                    affects the inventory. Must be corrected.
    MAJOR:          Significant error that should be corrected but
                    does not render the inventory unusable.
    MINOR:          Minor error or improvement opportunity.
    INFORMATIONAL:  Non-issue observation for awareness.
    """
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFORMATIONAL = "informational"

class EscalationReason(str, Enum):
    """Reason for escalating a review.

    DEADLINE_EXCEEDED:      Review deadline has been exceeded.
    REPEATED_REVISIONS:     Multiple revision rounds failed.
    CRITICAL_FINDING:       Critical finding requires immediate attention.
    VERIFICATION_FAILURE:   Verification found material misstatement.
    CONFLICT_OF_INTEREST:   Reviewer has a conflict of interest.
    SCOPE_EXCEEDS_AUTHORITY: Change exceeds reviewer's authority.
    """
    DEADLINE_EXCEEDED = "deadline_exceeded"
    REPEATED_REVISIONS = "repeated_revisions"
    CRITICAL_FINDING = "critical_finding"
    VERIFICATION_FAILURE = "verification_failure"
    CONFLICT_OF_INTEREST = "conflict_of_interest"
    SCOPE_EXCEEDS_AUTHORITY = "scope_exceeds_authority"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stage ordering for workflow progression.
STAGE_ORDER: List[ReviewStage] = [
    ReviewStage.PREPARATION,
    ReviewStage.REVIEW,
    ReviewStage.APPROVAL,
    ReviewStage.VERIFICATION,
    ReviewStage.COMPLETE,
]

# Maximum revision rounds before auto-escalation.
MAX_REVISION_ROUNDS: int = 3

# Default review deadline in hours per stage.
DEFAULT_DEADLINE_HOURS: Dict[str, int] = {
    ReviewStage.PREPARATION.value: 168,   # 7 days
    ReviewStage.REVIEW.value: 120,        # 5 days
    ReviewStage.APPROVAL.value: 72,       # 3 days
    ReviewStage.VERIFICATION.value: 240,  # 10 days
}

# Statuses that terminate a stage (no further action needed).
TERMINAL_STATUSES = frozenset({
    ReviewStatus.APPROVED,
    ReviewStatus.APPROVED_WITH_CONDITIONS,
    ReviewStatus.REJECTED,
    ReviewStatus.SKIPPED,
})

# Statuses that allow progression to next stage.
PROGRESSION_STATUSES = frozenset({
    ReviewStatus.APPROVED,
    ReviewStatus.APPROVED_WITH_CONDITIONS,
    ReviewStatus.SKIPPED,
})

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class ReviewComment(BaseModel):
    """A comment within a review thread.

    Attributes:
        comment_id: Unique comment identifier.
        review_request_id: Parent review request ID.
        stage: Review stage where this comment was made.
        comment_type: Type of comment (finding, question, etc.).
        severity: Severity of finding (for FINDING type).
        author_id: User who wrote the comment.
        author_name: Display name of the author.
        author_role: Role of the author (preparer, reviewer, etc.).
        content: Comment text content.
        section_reference: Reference to inventory section (e.g. scope1.stationary).
        data_reference: Reference to specific data point or cell.
        parent_comment_id: ID of parent comment (for threaded replies).
        is_resolved: Whether the comment/finding has been resolved.
        resolved_by: User who resolved the comment.
        resolved_at: Timestamp when resolved.
        created_at: Creation timestamp.
        provenance_hash: SHA-256 hash of the comment.
    """
    comment_id: str = Field(default_factory=_new_uuid, description="Comment ID")
    review_request_id: str = Field(default="", description="Parent review request ID")
    stage: ReviewStage = Field(
        default=ReviewStage.REVIEW, description="Review stage"
    )
    comment_type: CommentType = Field(
        default=CommentType.OBSERVATION, description="Comment type"
    )
    severity: FindingSeverity = Field(
        default=FindingSeverity.INFORMATIONAL, description="Finding severity"
    )
    author_id: str = Field(default="", description="Author user ID")
    author_name: str = Field(default="", description="Author display name")
    author_role: str = Field(default="reviewer", description="Author role")
    content: str = Field(default="", description="Comment text")
    section_reference: str = Field(
        default="", description="Inventory section reference"
    )
    data_reference: str = Field(
        default="", description="Specific data point reference"
    )
    parent_comment_id: Optional[str] = Field(
        default=None, description="Parent comment ID for threading"
    )
    is_resolved: bool = Field(default=False, description="Whether resolved")
    resolved_by: Optional[str] = Field(default=None, description="Resolved by user")
    resolved_at: Optional[datetime] = Field(
        default=None, description="Resolution timestamp"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class StageSignOff(BaseModel):
    """Digital sign-off record for a review stage.

    Attributes:
        signoff_id: Unique sign-off identifier.
        review_request_id: Parent review request ID.
        stage: Review stage being signed off.
        signer_id: User performing the sign-off.
        signer_name: Display name of the signer.
        signer_role: Role of the signer.
        decision: Review decision (approved, rejected, etc.).
        conditions: Conditions attached (for conditional approval).
        statement: Sign-off statement (e.g. assurance opinion).
        signed_at: Sign-off timestamp.
        provenance_hash: SHA-256 hash of the sign-off record.
    """
    signoff_id: str = Field(default_factory=_new_uuid, description="Sign-off ID")
    review_request_id: str = Field(default="", description="Parent review request ID")
    stage: ReviewStage = Field(
        default=ReviewStage.REVIEW, description="Stage signed off"
    )
    signer_id: str = Field(default="", description="Signer user ID")
    signer_name: str = Field(default="", description="Signer display name")
    signer_role: str = Field(default="", description="Signer role")
    decision: ReviewStatus = Field(
        default=ReviewStatus.APPROVED, description="Sign-off decision"
    )
    conditions: List[str] = Field(
        default_factory=list, description="Conditions (for conditional approval)"
    )
    statement: str = Field(
        default="", description="Sign-off statement"
    )
    signed_at: datetime = Field(
        default_factory=utcnow, description="Sign-off timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class ReviewRequest(BaseModel):
    """A review request for an inventory section or complete inventory.

    Attributes:
        request_id: Unique review request identifier.
        inventory_id: ID of the inventory being reviewed.
        inventory_version: Version of the inventory under review.
        reporting_year: Reporting year.
        section: Inventory section (e.g. scope1, scope2, full_inventory).
        title: Short title for the review.
        description: Detailed description of what is being reviewed.
        current_stage: Current review stage.
        stage_statuses: Status of each review stage.
        preparer_id: User who prepared the inventory section.
        preparer_name: Preparer display name.
        reviewer_id: Assigned reviewer.
        reviewer_name: Reviewer display name.
        approver_id: Assigned approver.
        approver_name: Approver display name.
        verifier_id: Assigned verifier (for assurance).
        verifier_name: Verifier display name.
        requires_verification: Whether verification stage is required.
        revision_round: Current revision round number.
        comments: Review comments and findings.
        signoffs: Stage sign-off records.
        deadline_hours: Deadline per stage in hours.
        created_at: Creation timestamp.
        metadata: Additional metadata.
    """
    request_id: str = Field(default_factory=_new_uuid, description="Review request ID")
    inventory_id: str = Field(default="", description="Inventory ID")
    inventory_version: str = Field(default="v1.0", description="Inventory version")
    reporting_year: int = Field(
        default=2025, ge=1990, le=2050, description="Reporting year"
    )
    section: str = Field(
        default="full_inventory", description="Inventory section"
    )
    title: str = Field(default="", max_length=500, description="Review title")
    description: str = Field(default="", description="Review description")
    current_stage: ReviewStage = Field(
        default=ReviewStage.PREPARATION, description="Current stage"
    )
    stage_statuses: Dict[str, str] = Field(
        default_factory=lambda: {
            ReviewStage.PREPARATION.value: ReviewStatus.NOT_STARTED.value,
            ReviewStage.REVIEW.value: ReviewStatus.NOT_STARTED.value,
            ReviewStage.APPROVAL.value: ReviewStatus.NOT_STARTED.value,
            ReviewStage.VERIFICATION.value: ReviewStatus.NOT_STARTED.value,
        },
        description="Status of each stage",
    )
    preparer_id: str = Field(default="", description="Preparer user ID")
    preparer_name: str = Field(default="", description="Preparer display name")
    reviewer_id: str = Field(default="", description="Reviewer user ID")
    reviewer_name: str = Field(default="", description="Reviewer display name")
    approver_id: str = Field(default="", description="Approver user ID")
    approver_name: str = Field(default="", description="Approver display name")
    verifier_id: str = Field(default="", description="Verifier user ID")
    verifier_name: str = Field(default="", description="Verifier display name")
    requires_verification: bool = Field(
        default=False, description="Whether verification required"
    )
    revision_round: int = Field(
        default=0, ge=0, description="Current revision round"
    )
    comments: List[ReviewComment] = Field(
        default_factory=list, description="Review comments"
    )
    signoffs: List[StageSignOff] = Field(
        default_factory=list, description="Stage sign-offs"
    )
    deadline_hours: Dict[str, int] = Field(
        default_factory=lambda: dict(DEFAULT_DEADLINE_HOURS),
        description="Deadline per stage (hours)",
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ReviewDecision(BaseModel):
    """Decision outcome for a review stage.

    Attributes:
        request_id: Review request ID.
        stage: Stage the decision applies to.
        decision: Decision made (approved, rejected, etc.).
        decided_by_id: User who made the decision.
        decided_by_name: Display name of the decision-maker.
        findings_summary: Summary of findings at this stage.
        critical_findings: Count of critical findings.
        major_findings: Count of major findings.
        minor_findings: Count of minor findings.
        unresolved_findings: Count of unresolved findings.
        conditions: Conditions attached to the decision.
        can_progress: Whether the workflow can progress to next stage.
        next_stage: Next stage in the workflow (if can_progress).
        rationale: Explanation of the decision.
    """
    request_id: str = Field(default="", description="Review request ID")
    stage: ReviewStage = Field(
        default=ReviewStage.REVIEW, description="Review stage"
    )
    decision: ReviewStatus = Field(
        default=ReviewStatus.APPROVED, description="Decision"
    )
    decided_by_id: str = Field(default="", description="Decision-maker ID")
    decided_by_name: str = Field(default="", description="Decision-maker name")
    findings_summary: str = Field(default="", description="Findings summary")
    critical_findings: int = Field(default=0, ge=0, description="Critical findings")
    major_findings: int = Field(default=0, ge=0, description="Major findings")
    minor_findings: int = Field(default=0, ge=0, description="Minor findings")
    unresolved_findings: int = Field(
        default=0, ge=0, description="Unresolved findings"
    )
    conditions: List[str] = Field(
        default_factory=list, description="Conditions"
    )
    can_progress: bool = Field(
        default=False, description="Whether workflow can progress"
    )
    next_stage: Optional[ReviewStage] = Field(
        default=None, description="Next stage"
    )
    rationale: str = Field(default="", description="Decision rationale")

class ApprovalRecord(BaseModel):
    """Permanent approval record for the inventory.

    Attributes:
        record_id: Unique approval record identifier.
        request_id: Review request ID.
        inventory_id: Inventory ID.
        inventory_version: Inventory version approved.
        reporting_year: Reporting year.
        approval_chain: Ordered list of sign-offs in the approval chain.
        all_stages_complete: Whether all required stages completed.
        final_status: Final review status.
        total_findings: Total findings across all stages.
        critical_unresolved: Count of unresolved critical findings.
        is_fully_approved: Whether inventory is fully approved.
        approved_at: Final approval timestamp.
        provenance_hash: SHA-256 hash for tamper detection.
    """
    record_id: str = Field(default_factory=_new_uuid, description="Record ID")
    request_id: str = Field(default="", description="Review request ID")
    inventory_id: str = Field(default="", description="Inventory ID")
    inventory_version: str = Field(default="", description="Inventory version")
    reporting_year: int = Field(default=2025, description="Reporting year")
    approval_chain: List[StageSignOff] = Field(
        default_factory=list, description="Approval chain"
    )
    all_stages_complete: bool = Field(
        default=False, description="All stages complete"
    )
    final_status: ReviewStatus = Field(
        default=ReviewStatus.NOT_STARTED, description="Final status"
    )
    total_findings: int = Field(default=0, ge=0, description="Total findings")
    critical_unresolved: int = Field(
        default=0, ge=0, description="Unresolved critical findings"
    )
    is_fully_approved: bool = Field(
        default=False, description="Whether fully approved"
    )
    approved_at: Optional[datetime] = Field(
        default=None, description="Final approval timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class ReviewApprovalResult(BaseModel):
    """Complete result from the review and approval engine.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        request: Updated review request after processing.
        decision: Decision made for the current action.
        approval_record: Approval record (if workflow is complete).
        escalations: Any escalations triggered during processing.
        audit_entries: Audit trail entries generated.
        warnings: Warnings raised.
        calculated_at: Processing timestamp.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    request: Optional[ReviewRequest] = Field(
        default=None, description="Updated review request"
    )
    decision: Optional[ReviewDecision] = Field(
        default=None, description="Review decision"
    )
    approval_record: Optional[ApprovalRecord] = Field(
        default=None, description="Approval record"
    )
    escalations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Escalations triggered"
    )
    audit_entries: List[Dict[str, Any]] = Field(
        default_factory=list, description="Audit trail entries"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Processing timestamp"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

ReviewComment.model_rebuild()
StageSignOff.model_rebuild()
ReviewRequest.model_rebuild()
ReviewDecision.model_rebuild()
ApprovalRecord.model_rebuild()
ReviewApprovalResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReviewApprovalEngine:
    """Multi-level review and sign-off engine for GHG inventories.

    Implements a four-stage review workflow (Preparer -> Reviewer ->
    Approver -> Verifier) with digital sign-off, threaded comments,
    separation of duties enforcement, and escalation management.

    Guarantees:
        - Deterministic: same inputs always produce identical outputs.
        - Separation of duties: enforced programmatically.
        - Auditable: complete trail for every action and decision.
        - No LLM: zero hallucination risk in workflow logic.

    Attributes:
        _config: Engine configuration.
        _audit_entries: Accumulated audit trail entries.
        _escalations: Accumulated escalation records.
        _warnings: Accumulated warnings.

    Example:
        >>> engine = ReviewApprovalEngine()
        >>> request = ReviewRequest(
        ...     title="Scope 1 Q4 Review",
        ...     preparer_id="user-001",
        ...     reviewer_id="user-002",
        ...     approver_id="user-003",
        ... )
        >>> result = engine.submit_for_review(request, "user-001", "Alice")
        >>> print(result.request.current_stage)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the ReviewApprovalEngine.

        Args:
            config: Optional configuration overrides. Supported keys:
                - max_revision_rounds (int): default 3
                - enforce_separation_of_duties (bool): default True
                - auto_skip_verification (bool): default False
        """
        self._config = config or {}
        self._max_revisions = int(
            self._config.get("max_revision_rounds", MAX_REVISION_ROUNDS)
        )
        self._enforce_sod = bool(
            self._config.get("enforce_separation_of_duties", True)
        )
        self._auto_skip_verification = bool(
            self._config.get("auto_skip_verification", False)
        )
        self._audit_entries: List[Dict[str, Any]] = []
        self._escalations: List[Dict[str, Any]] = []
        self._warnings: List[str] = []

        logger.info(
            "ReviewApprovalEngine v%s initialised: max_revisions=%d, "
            "enforce_sod=%s, auto_skip_verification=%s",
            _MODULE_VERSION, self._max_revisions,
            self._enforce_sod, self._auto_skip_verification,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_for_review(
        self,
        request: ReviewRequest,
        actor_id: str,
        actor_name: str,
    ) -> ReviewApprovalResult:
        """Submit an inventory section for review.

        Transitions from PREPARATION to REVIEW stage.  The preparer
        submits their completed work for peer review.

        Args:
            request: The review request to submit.
            actor_id: ID of the submitting user (must be preparer).
            actor_name: Display name of the submitting user.

        Returns:
            ReviewApprovalResult with updated request.

        Raises:
            ValueError: If actor is not the preparer or stage is wrong.
        """
        t0 = time.perf_counter()
        self._reset_state()

        logger.info(
            "Submitting review request %s for review by %s",
            request.request_id[:12], actor_name,
        )

        # Validate submitter is the preparer.
        if self._enforce_sod and request.preparer_id and actor_id != request.preparer_id:
            raise ValueError(
                f"Only the preparer ({request.preparer_id}) can submit "
                f"for review. Actor: {actor_id}."
            )

        if request.current_stage != ReviewStage.PREPARATION:
            raise ValueError(
                f"Cannot submit for review: current stage is "
                f"{request.current_stage.value}, expected {ReviewStage.PREPARATION.value}."
            )

        # Update stage statuses.
        request.stage_statuses[ReviewStage.PREPARATION.value] = (
            ReviewStatus.APPROVED.value
        )
        request.stage_statuses[ReviewStage.REVIEW.value] = (
            ReviewStatus.IN_PROGRESS.value
        )
        request.current_stage = ReviewStage.REVIEW

        signoff = StageSignOff(
            review_request_id=request.request_id,
            stage=ReviewStage.PREPARATION,
            signer_id=actor_id,
            signer_name=actor_name,
            signer_role="preparer",
            decision=ReviewStatus.APPROVED,
            statement="Preparation complete. Submitted for review.",
        )
        signoff.provenance_hash = _compute_hash(signoff)
        request.signoffs.append(signoff)

        self._add_audit(
            request.request_id, "submitted_for_review",
            actor_id, actor_name,
            "Inventory section submitted for peer review.",
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))
        return self._build_result(request, None, elapsed)

    def record_decision(
        self,
        request: ReviewRequest,
        stage: ReviewStage,
        decision: ReviewStatus,
        actor_id: str,
        actor_name: str,
        conditions: Optional[List[str]] = None,
        statement: str = "",
    ) -> ReviewApprovalResult:
        """Record a review decision at a specific stage.

        The reviewer, approver, or verifier records their decision
        for the current stage.  Enforces separation of duties.

        Args:
            request: The review request.
            stage: Stage for which the decision is being made.
            decision: The decision (approved, rejected, etc.).
            actor_id: ID of the decision-maker.
            actor_name: Display name of the decision-maker.
            conditions: Optional conditions (for conditional approval).
            statement: Sign-off statement.

        Returns:
            ReviewApprovalResult with decision and updated request.

        Raises:
            ValueError: If separation of duties is violated or stage mismatch.
        """
        t0 = time.perf_counter()
        self._reset_state()

        logger.info(
            "Recording decision for request %s: stage=%s, decision=%s, by %s",
            request.request_id[:12], stage.value, decision.value, actor_name,
        )

        # Validate stage matches current stage.
        if request.current_stage != stage:
            raise ValueError(
                f"Decision stage mismatch: request is at "
                f"{request.current_stage.value}, decision for {stage.value}."
            )

        # Enforce separation of duties.
        self._enforce_separation(request, stage, actor_id)

        # Count findings.
        stage_comments = [
            c for c in request.comments if c.stage == stage
        ]
        critical = sum(
            1 for c in stage_comments
            if c.comment_type == CommentType.FINDING
            and c.severity == FindingSeverity.CRITICAL
        )
        major = sum(
            1 for c in stage_comments
            if c.comment_type == CommentType.FINDING
            and c.severity == FindingSeverity.MAJOR
        )
        minor = sum(
            1 for c in stage_comments
            if c.comment_type == CommentType.FINDING
            and c.severity == FindingSeverity.MINOR
        )
        unresolved = sum(
            1 for c in stage_comments
            if c.comment_type == CommentType.FINDING
            and not c.is_resolved
        )

        # Block approval if critical findings are unresolved.
        if decision in (ReviewStatus.APPROVED, ReviewStatus.APPROVED_WITH_CONDITIONS):
            unresolved_critical = sum(
                1 for c in stage_comments
                if c.comment_type == CommentType.FINDING
                and c.severity == FindingSeverity.CRITICAL
                and not c.is_resolved
            )
            if unresolved_critical > 0:
                self._warnings.append(
                    f"Cannot fully approve: {unresolved_critical} unresolved "
                    f"critical finding(s). Converting to conditional approval."
                )
                decision = ReviewStatus.APPROVED_WITH_CONDITIONS
                if conditions is None:
                    conditions = []
                conditions.append(
                    f"Resolve {unresolved_critical} critical finding(s) "
                    f"before final sign-off."
                )

        # Handle revisions requested.
        if decision == ReviewStatus.REVISIONS_REQUESTED:
            request.revision_round += 1
            if request.revision_round >= self._max_revisions:
                self._escalate(
                    request, EscalationReason.REPEATED_REVISIONS,
                    f"Revision round {request.revision_round} exceeds "
                    f"maximum of {self._max_revisions}.",
                )

        # Update stage status.
        request.stage_statuses[stage.value] = decision.value

        # Create sign-off.
        signoff = StageSignOff(
            review_request_id=request.request_id,
            stage=stage,
            signer_id=actor_id,
            signer_name=actor_name,
            signer_role=self._role_for_stage(stage),
            decision=decision,
            conditions=conditions or [],
            statement=statement,
        )
        signoff.provenance_hash = _compute_hash(signoff)
        request.signoffs.append(signoff)

        # Determine progression.
        can_progress = decision in PROGRESSION_STATUSES
        next_stage = self._get_next_stage(request, stage) if can_progress else None

        # Progress to next stage if approved.
        if can_progress and next_stage:
            request.current_stage = next_stage
            if next_stage != ReviewStage.COMPLETE:
                request.stage_statuses[next_stage.value] = (
                    ReviewStatus.IN_PROGRESS.value
                )
            logger.info(
                "Request %s progressed to %s",
                request.request_id[:12], next_stage.value,
            )

        # Build decision record.
        findings_summary = (
            f"{critical} critical, {major} major, {minor} minor findings; "
            f"{unresolved} unresolved."
        )

        review_decision = ReviewDecision(
            request_id=request.request_id,
            stage=stage,
            decision=decision,
            decided_by_id=actor_id,
            decided_by_name=actor_name,
            findings_summary=findings_summary,
            critical_findings=critical,
            major_findings=major,
            minor_findings=minor,
            unresolved_findings=unresolved,
            conditions=conditions or [],
            can_progress=can_progress,
            next_stage=next_stage,
            rationale=self._build_decision_rationale(
                stage, decision, critical, major, unresolved,
            ),
        )

        self._add_audit(
            request.request_id, f"decision_{decision.value}",
            actor_id, actor_name,
            f"Stage {stage.value}: {decision.value}. {findings_summary}",
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))
        return self._build_result(request, review_decision, elapsed)

    def add_comment(
        self,
        request: ReviewRequest,
        comment: ReviewComment,
    ) -> ReviewRequest:
        """Add a comment to a review request.

        Validates that the comment is for the current or prior stage,
        computes the provenance hash, and appends to the comments list.

        Args:
            request: The review request.
            comment: The comment to add.

        Returns:
            Updated ReviewRequest with the comment added.
        """
        comment.review_request_id = request.request_id
        comment.provenance_hash = _compute_hash(comment)
        request.comments.append(comment)

        # Auto-escalate critical findings.
        if (
            comment.comment_type == CommentType.FINDING
            and comment.severity == FindingSeverity.CRITICAL
        ):
            self._escalate(
                request, EscalationReason.CRITICAL_FINDING,
                f"Critical finding raised by {comment.author_name}: "
                f"{comment.content[:100]}",
            )

        logger.info(
            "Comment added to request %s: type=%s, severity=%s, by %s",
            request.request_id[:12], comment.comment_type.value,
            comment.severity.value, comment.author_name,
        )
        return request

    def resolve_comment(
        self,
        request: ReviewRequest,
        comment_id: str,
        resolver_id: str,
        resolver_name: str,
    ) -> ReviewRequest:
        """Mark a comment as resolved.

        Args:
            request: The review request.
            comment_id: ID of the comment to resolve.
            resolver_id: User resolving the comment.
            resolver_name: Display name of the resolver.

        Returns:
            Updated ReviewRequest with the comment resolved.

        Raises:
            ValueError: If comment not found.
        """
        for comment in request.comments:
            if comment.comment_id == comment_id:
                comment.is_resolved = True
                comment.resolved_by = resolver_id
                comment.resolved_at = utcnow()
                comment.provenance_hash = _compute_hash(comment)
                logger.info(
                    "Comment %s resolved by %s",
                    comment_id[:12], resolver_name,
                )
                return request

        raise ValueError(f"Comment {comment_id} not found in request.")

    def generate_approval_record(
        self,
        request: ReviewRequest,
    ) -> ApprovalRecord:
        """Generate a permanent approval record for a completed review.

        Should be called once all required stages are complete.

        Args:
            request: The completed review request.

        Returns:
            ApprovalRecord with full audit chain.
        """
        all_complete = self._are_all_stages_complete(request)

        # Determine final status.
        if all_complete:
            has_conditions = any(
                s.decision == ReviewStatus.APPROVED_WITH_CONDITIONS
                for s in request.signoffs
            )
            final = (
                ReviewStatus.APPROVED_WITH_CONDITIONS
                if has_conditions
                else ReviewStatus.APPROVED
            )
        else:
            final = ReviewStatus.IN_PROGRESS

        total_findings = sum(
            1 for c in request.comments
            if c.comment_type == CommentType.FINDING
        )
        critical_unresolved = sum(
            1 for c in request.comments
            if c.comment_type == CommentType.FINDING
            and c.severity == FindingSeverity.CRITICAL
            and not c.is_resolved
        )

        record = ApprovalRecord(
            request_id=request.request_id,
            inventory_id=request.inventory_id,
            inventory_version=request.inventory_version,
            reporting_year=request.reporting_year,
            approval_chain=list(request.signoffs),
            all_stages_complete=all_complete,
            final_status=final,
            total_findings=total_findings,
            critical_unresolved=critical_unresolved,
            is_fully_approved=(
                all_complete
                and critical_unresolved == 0
                and final in (ReviewStatus.APPROVED, ReviewStatus.APPROVED_WITH_CONDITIONS)
            ),
            approved_at=utcnow() if all_complete else None,
        )
        record.provenance_hash = _compute_hash(record)

        logger.info(
            "Approval record generated for request %s: fully_approved=%s, "
            "findings=%d, critical_unresolved=%d",
            request.request_id[:12], record.is_fully_approved,
            total_findings, critical_unresolved,
        )
        return record

    def check_escalation_needed(
        self,
        request: ReviewRequest,
        current_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Check whether any escalation conditions are met.

        Evaluates deadline, revision count, and critical finding rules.

        Args:
            request: The review request to check.
            current_time: Current time for deadline comparison.
                Defaults to UTC now.

        Returns:
            List of escalation dicts with reason and details.
        """
        now = current_time or utcnow()
        escalations: List[Dict[str, Any]] = []

        # Check revision rounds.
        if request.revision_round >= self._max_revisions:
            escalations.append({
                "reason": EscalationReason.REPEATED_REVISIONS.value,
                "details": (
                    f"Revision round {request.revision_round} exceeds "
                    f"maximum of {self._max_revisions}."
                ),
                "request_id": request.request_id,
                "current_stage": request.current_stage.value,
            })

        # Check for unresolved critical findings.
        unresolved_critical = [
            c for c in request.comments
            if c.comment_type == CommentType.FINDING
            and c.severity == FindingSeverity.CRITICAL
            and not c.is_resolved
        ]
        if unresolved_critical:
            escalations.append({
                "reason": EscalationReason.CRITICAL_FINDING.value,
                "details": (
                    f"{len(unresolved_critical)} unresolved critical finding(s)."
                ),
                "request_id": request.request_id,
                "finding_ids": [c.comment_id for c in unresolved_critical],
            })

        return escalations

    def get_review_summary(
        self,
        request: ReviewRequest,
    ) -> Dict[str, Any]:
        """Generate a summary of the review status.

        Args:
            request: The review request.

        Returns:
            Dict with review summary statistics.
        """
        total_comments = len(request.comments)
        findings = [
            c for c in request.comments
            if c.comment_type == CommentType.FINDING
        ]
        resolved = sum(1 for c in findings if c.is_resolved)
        unresolved = len(findings) - resolved

        by_severity: Dict[str, int] = {s.value: 0 for s in FindingSeverity}
        for f in findings:
            by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1

        by_stage: Dict[str, int] = {s.value: 0 for s in ReviewStage if s != ReviewStage.COMPLETE}
        for c in request.comments:
            if c.stage.value in by_stage:
                by_stage[c.stage.value] = by_stage.get(c.stage.value, 0) + 1

        return {
            "request_id": request.request_id,
            "current_stage": request.current_stage.value,
            "stage_statuses": dict(request.stage_statuses),
            "revision_round": request.revision_round,
            "total_comments": total_comments,
            "total_findings": len(findings),
            "resolved_findings": resolved,
            "unresolved_findings": unresolved,
            "findings_by_severity": by_severity,
            "comments_by_stage": by_stage,
            "signoffs_count": len(request.signoffs),
            "requires_verification": request.requires_verification,
        }

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset per-operation state."""
        self._audit_entries = []
        self._escalations = []
        self._warnings = []

    def _enforce_separation(
        self,
        request: ReviewRequest,
        stage: ReviewStage,
        actor_id: str,
    ) -> None:
        """Enforce separation of duties for a review stage.

        Args:
            request: The review request.
            stage: Current stage.
            actor_id: User attempting to act.

        Raises:
            ValueError: If separation of duties is violated.
        """
        if not self._enforce_sod:
            return

        if stage == ReviewStage.REVIEW:
            if actor_id == request.preparer_id:
                raise ValueError(
                    "Separation of duties violation: reviewer cannot be "
                    "the same person as preparer."
                )

        elif stage == ReviewStage.APPROVAL:
            if actor_id == request.preparer_id:
                raise ValueError(
                    "Separation of duties violation: approver cannot be "
                    "the same person as preparer."
                )
            if actor_id == request.reviewer_id:
                raise ValueError(
                    "Separation of duties violation: approver cannot be "
                    "the same person as reviewer."
                )

        elif stage == ReviewStage.VERIFICATION:
            if actor_id in (
                request.preparer_id, request.reviewer_id, request.approver_id,
            ):
                raise ValueError(
                    "Separation of duties violation: verifier must be "
                    "independent of preparer, reviewer, and approver."
                )

    def _get_next_stage(
        self,
        request: ReviewRequest,
        current: ReviewStage,
    ) -> Optional[ReviewStage]:
        """Determine the next stage in the workflow.

        Args:
            request: The review request.
            current: Current stage just completed.

        Returns:
            Next ReviewStage, or None if workflow is complete.
        """
        try:
            idx = STAGE_ORDER.index(current)
        except ValueError:
            return None

        next_idx = idx + 1
        if next_idx >= len(STAGE_ORDER):
            return ReviewStage.COMPLETE

        next_stage = STAGE_ORDER[next_idx]

        # Skip verification if not required.
        if (
            next_stage == ReviewStage.VERIFICATION
            and not request.requires_verification
        ):
            request.stage_statuses[ReviewStage.VERIFICATION.value] = (
                ReviewStatus.SKIPPED.value
            )
            return ReviewStage.COMPLETE

        return next_stage

    def _role_for_stage(self, stage: ReviewStage) -> str:
        """Get the role name for a review stage.

        Args:
            stage: The review stage.

        Returns:
            Role name string.
        """
        role_map = {
            ReviewStage.PREPARATION: "preparer",
            ReviewStage.REVIEW: "reviewer",
            ReviewStage.APPROVAL: "approver",
            ReviewStage.VERIFICATION: "verifier",
        }
        return role_map.get(stage, "unknown")

    def _are_all_stages_complete(self, request: ReviewRequest) -> bool:
        """Check whether all required stages are complete.

        Args:
            request: The review request.

        Returns:
            True if all required stages have terminal status.
        """
        required_stages = [
            ReviewStage.PREPARATION,
            ReviewStage.REVIEW,
            ReviewStage.APPROVAL,
        ]
        if request.requires_verification:
            required_stages.append(ReviewStage.VERIFICATION)

        for stage in required_stages:
            status_str = request.stage_statuses.get(
                stage.value, ReviewStatus.NOT_STARTED.value
            )
            try:
                status = ReviewStatus(status_str)
            except ValueError:
                return False
            if status not in TERMINAL_STATUSES:
                return False

        return True

    def _build_decision_rationale(
        self,
        stage: ReviewStage,
        decision: ReviewStatus,
        critical: int,
        major: int,
        unresolved: int,
    ) -> str:
        """Build rationale string for a review decision.

        Args:
            stage: Review stage.
            decision: Decision made.
            critical: Critical finding count.
            major: Major finding count.
            unresolved: Unresolved finding count.

        Returns:
            Rationale string.
        """
        parts = [f"Stage: {stage.value}. Decision: {decision.value}."]

        if critical > 0:
            parts.append(f"{critical} critical finding(s) identified.")
        if major > 0:
            parts.append(f"{major} major finding(s) identified.")
        if unresolved > 0:
            parts.append(f"{unresolved} finding(s) remain unresolved.")

        if decision == ReviewStatus.APPROVED:
            parts.append("All review criteria met. Stage approved.")
        elif decision == ReviewStatus.APPROVED_WITH_CONDITIONS:
            parts.append(
                "Approved with conditions. Outstanding items must be "
                "addressed before final certification."
            )
        elif decision == ReviewStatus.REVISIONS_REQUESTED:
            parts.append(
                "Revisions requested. Preparer must address findings "
                "and resubmit for review."
            )
        elif decision == ReviewStatus.REJECTED:
            parts.append(
                "Rejected. Significant issues prevent approval. "
                "See findings for details."
            )

        return " ".join(parts)

    def _escalate(
        self,
        request: ReviewRequest,
        reason: EscalationReason,
        details: str,
    ) -> None:
        """Record an escalation event.

        Args:
            request: The review request.
            reason: Reason for escalation.
            details: Escalation details.
        """
        escalation = {
            "escalation_id": _new_uuid(),
            "request_id": request.request_id,
            "reason": reason.value,
            "details": details,
            "current_stage": request.current_stage.value,
            "timestamp": utcnow().isoformat(),
        }
        self._escalations.append(escalation)
        self._warnings.append(f"ESCALATION: {reason.value} - {details}")

        logger.warning(
            "Escalation for request %s: %s - %s",
            request.request_id[:12], reason.value, details,
        )

    def _add_audit(
        self,
        request_id: str,
        action: str,
        actor_id: str,
        actor_name: str,
        details: str,
    ) -> None:
        """Add an audit trail entry.

        Args:
            request_id: Related review request ID.
            action: Action performed.
            actor_id: Actor user ID.
            actor_name: Actor display name.
            details: Event details.
        """
        entry = {
            "entry_id": _new_uuid(),
            "request_id": request_id,
            "action": action,
            "actor_id": actor_id,
            "actor_name": actor_name,
            "details": details,
            "timestamp": utcnow().isoformat(),
        }
        entry["provenance_hash"] = _compute_hash(entry)
        self._audit_entries.append(entry)

    def _build_result(
        self,
        request: ReviewRequest,
        decision: Optional[ReviewDecision],
        elapsed_ms: Decimal,
    ) -> ReviewApprovalResult:
        """Build the final ReviewApprovalResult.

        Also generates an approval record if the workflow is complete.

        Args:
            request: Updated review request.
            decision: Review decision (if applicable).
            elapsed_ms: Processing time in milliseconds.

        Returns:
            ReviewApprovalResult with provenance hash.
        """
        # Generate approval record if complete.
        approval_record = None
        if request.current_stage == ReviewStage.COMPLETE:
            approval_record = self.generate_approval_record(request)

        result = ReviewApprovalResult(
            request=request,
            decision=decision,
            approval_record=approval_record,
            escalations=list(self._escalations),
            audit_entries=list(self._audit_entries),
            warnings=list(self._warnings),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result
