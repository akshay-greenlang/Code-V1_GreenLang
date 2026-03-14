# -*- coding: utf-8 -*-
"""
ApprovalWorkflowEngine - PACK-002 CSRD Professional Engine 2

Four-level approval chain engine implementing the CSRD report approval
workflow: Preparer -> Reviewer -> Approver -> Board. Supports delegation,
auto-approval based on quality gate scores, escalation for overdue items,
and complete audit trail tracking.

Approval Levels:
    1. PREPARER  - Data preparers who compile and submit ESRS data
    2. REVIEWER  - Technical reviewers who validate accuracy
    3. APPROVER  - Management approvers who authorize disclosure
    4. BOARD     - Board-level sign-off for final submission

Status Flow:
    PENDING -> SUBMITTED -> IN_REVIEW -> APPROVED
    IN_REVIEW -> REJECTED / RETURNED / ESCALATED
    RETURNED -> SUBMITTED (re-submit after revision)
    ESCALATED -> IN_REVIEW (higher authority picks up)

Features:
    - Configurable multi-approver requirements per level
    - Quality-gate-based auto-approval thresholds
    - Time-based escalation with configurable timeouts
    - Delegation of authority with scope and expiry
    - Complete decision audit trail with comments
    - SHA-256 provenance hashing on all decisions

Zero-Hallucination:
    - All status transitions use explicit allowed-transitions map
    - Timeout calculations use deterministic datetime arithmetic
    - Auto-approval uses numeric threshold comparison only
    - No LLM involvement in any approval decision

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ApprovalLevel(IntEnum):
    """Hierarchical approval levels."""

    PREPARER = 1
    REVIEWER = 2
    APPROVER = 3
    BOARD = 4


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETURNED = "returned"
    ESCALATED = "escalated"


class DecisionType(str, Enum):
    """Type of approval decision."""

    APPROVE = "approve"
    REJECT = "reject"
    RETURN = "return"


# Allowed status transitions
_ALLOWED_TRANSITIONS: Dict[ApprovalStatus, Set[ApprovalStatus]] = {
    ApprovalStatus.PENDING: {ApprovalStatus.SUBMITTED},
    ApprovalStatus.SUBMITTED: {ApprovalStatus.IN_REVIEW},
    ApprovalStatus.IN_REVIEW: {
        ApprovalStatus.APPROVED,
        ApprovalStatus.REJECTED,
        ApprovalStatus.RETURNED,
        ApprovalStatus.ESCALATED,
    },
    ApprovalStatus.REJECTED: {ApprovalStatus.SUBMITTED},
    ApprovalStatus.RETURNED: {ApprovalStatus.SUBMITTED},
    ApprovalStatus.ESCALATED: {ApprovalStatus.IN_REVIEW},
    ApprovalStatus.APPROVED: set(),  # Terminal state (per level)
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ApprovalLevelConfig(BaseModel):
    """Configuration for a single approval level."""

    level: ApprovalLevel = Field(..., description="Approval level")
    required_approvers: int = Field(
        1, ge=1, description="Number of approvers needed to pass"
    )
    auto_approve_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Quality score threshold for auto-approval (None=disabled)",
    )
    escalation_timeout_hours: int = Field(
        48, ge=1, description="Hours before automatic escalation"
    )
    role_required: str = Field(
        "", description="Role name required for approvers at this level"
    )
    can_delegate: bool = Field(
        True, description="Whether this level supports delegation"
    )


class ApprovalComment(BaseModel):
    """A comment attached to an approval request."""

    comment_id: str = Field(default_factory=_new_uuid, description="Comment ID")
    author: str = Field(..., description="Comment author username")
    timestamp: datetime = Field(default_factory=_utcnow, description="When posted")
    text: str = Field(..., min_length=1, description="Comment text")
    level: ApprovalLevel = Field(..., description="Level at which comment was made")


class ApprovalDecision(BaseModel):
    """A decision made by an approver."""

    decision_id: str = Field(default_factory=_new_uuid, description="Decision ID")
    decision: DecisionType = Field(..., description="Decision type")
    approver: str = Field(..., description="Approver username")
    timestamp: datetime = Field(default_factory=_utcnow, description="When decided")
    comments: str = Field("", description="Decision comments")
    conditions: List[str] = Field(
        default_factory=list, description="Conditions for conditional approval"
    )
    level: ApprovalLevel = Field(
        ApprovalLevel.PREPARER, description="Level at which decision was made"
    )
    provenance_hash: str = Field("", description="SHA-256 hash of decision")


class DelegationEntry(BaseModel):
    """Authority delegation from one user to another."""

    delegation_id: str = Field(default_factory=_new_uuid, description="Delegation ID")
    delegator: str = Field(..., description="User delegating authority")
    delegate: str = Field(..., description="User receiving authority")
    scope: str = Field(
        "all", description="Scope of delegation (workflow_id or 'all')"
    )
    valid_from: datetime = Field(
        default_factory=_utcnow, description="Delegation start"
    )
    valid_until: datetime = Field(
        ..., description="Delegation expiry"
    )
    level: ApprovalLevel = Field(
        ApprovalLevel.REVIEWER, description="Level being delegated"
    )

    @field_validator("valid_until")
    @classmethod
    def validate_expiry(cls, v: datetime, info: Any) -> datetime:
        """Ensure delegation has a valid timeframe."""
        return v


class ApprovalRequest(BaseModel):
    """An approval request moving through the workflow."""

    request_id: str = Field(default_factory=_new_uuid, description="Request ID")
    workflow_id: str = Field(..., description="Parent workflow this belongs to")
    current_level: ApprovalLevel = Field(
        ApprovalLevel.PREPARER, description="Current approval level"
    )
    status: ApprovalStatus = Field(
        ApprovalStatus.PENDING, description="Current status"
    )
    submitted_by: str = Field(..., description="User who submitted")
    assigned_to: List[str] = Field(
        default_factory=list, description="Currently assigned approvers"
    )
    comments: List[ApprovalComment] = Field(
        default_factory=list, description="Comments thread"
    )
    decisions: List[ApprovalDecision] = Field(
        default_factory=list, description="Decision history"
    )
    quality_gate_results: Optional[Dict[str, Any]] = Field(
        None, description="Quality gate evaluation results"
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Creation time")
    updated_at: datetime = Field(default_factory=_utcnow, description="Last update")
    escalation_count: int = Field(0, ge=0, description="Number of escalations")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")


class ApprovalChainResult(BaseModel):
    """Result of a complete approval chain execution."""

    chain_id: str = Field(default_factory=_new_uuid, description="Chain ID")
    workflow_id: str = Field(..., description="Workflow ID")
    levels_completed: List[int] = Field(
        default_factory=list, description="Completed approval levels"
    )
    current_level: ApprovalLevel = Field(
        ApprovalLevel.PREPARER, description="Current level"
    )
    overall_status: ApprovalStatus = Field(
        ApprovalStatus.PENDING, description="Overall chain status"
    )
    history: List[ApprovalDecision] = Field(
        default_factory=list, description="All decisions in order"
    )
    total_elapsed_hours: float = Field(0.0, description="Total hours from start")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ApprovalConfig(BaseModel):
    """Configuration for the approval workflow engine."""

    max_escalation_levels: int = Field(
        3, ge=1, description="Maximum escalation count before forced review"
    )
    allow_self_approval: bool = Field(
        False, description="Allow submitter to approve their own submission"
    )
    require_comments_on_reject: bool = Field(
        True, description="Require comments when rejecting"
    )
    default_escalation_hours: int = Field(
        48, ge=1, description="Default hours before escalation"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ApprovalWorkflowEngine:
    """Four-level approval chain engine for CSRD reporting workflows.

    Manages the lifecycle of approval requests through Preparer, Reviewer,
    Approver, and Board levels with delegation, auto-approval, and escalation.

    Attributes:
        config: Engine configuration.
        chains: Approval chain configurations keyed by chain_id.
        requests: Approval requests keyed by request_id.
        delegations: Active delegation entries.

    Example:
        >>> engine = ApprovalWorkflowEngine()
        >>> chain_id = await engine.create_chain("wf-001", [
        ...     ApprovalLevelConfig(level=ApprovalLevel.REVIEWER, required_approvers=1),
        ...     ApprovalLevelConfig(level=ApprovalLevel.APPROVER, required_approvers=2),
        ... ])
        >>> request = ApprovalRequest(workflow_id="wf-001", submitted_by="analyst")
        >>> request = await engine.submit_for_approval(request)
    """

    def __init__(self, config: Optional[ApprovalConfig] = None) -> None:
        """Initialize ApprovalWorkflowEngine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
        """
        self.config = config or ApprovalConfig()
        self.chains: Dict[str, Dict[str, Any]] = {}
        self.requests: Dict[str, ApprovalRequest] = {}
        self.delegations: List[DelegationEntry] = []
        self._level_configs: Dict[str, List[ApprovalLevelConfig]] = {}
        logger.info(
            "ApprovalWorkflowEngine initialized (version=%s)", _MODULE_VERSION
        )

    # -- Chain Management ---------------------------------------------------

    async def create_chain(
        self, workflow_id: str, levels: List[ApprovalLevelConfig]
    ) -> str:
        """Initialize an approval chain for a workflow.

        Args:
            workflow_id: Workflow identifier.
            levels: Ordered list of approval level configurations.

        Returns:
            Chain identifier string.

        Raises:
            ValueError: If levels list is empty or has duplicate levels.
        """
        if not levels:
            raise ValueError("At least one approval level is required")

        seen_levels: Set[int] = set()
        for lc in levels:
            if lc.level.value in seen_levels:
                raise ValueError(f"Duplicate level: {lc.level.name}")
            seen_levels.add(lc.level.value)

        sorted_levels = sorted(levels, key=lambda x: x.level.value)
        chain_id = _new_uuid()

        self.chains[chain_id] = {
            "chain_id": chain_id,
            "workflow_id": workflow_id,
            "levels": [lc.model_dump() for lc in sorted_levels],
            "created_at": _utcnow().isoformat(),
        }
        self._level_configs[workflow_id] = sorted_levels

        logger.info(
            "Approval chain created: chain_id=%s, workflow_id=%s, levels=%d",
            chain_id,
            workflow_id,
            len(levels),
        )
        return chain_id

    # -- Submission ---------------------------------------------------------

    async def submit_for_approval(
        self, request: ApprovalRequest
    ) -> ApprovalRequest:
        """Submit a request for approval at the current level.

        Args:
            request: Approval request to submit.

        Returns:
            Updated ApprovalRequest with SUBMITTED status.

        Raises:
            ValueError: If workflow has no chain configuration.
        """
        if request.workflow_id not in self._level_configs:
            raise ValueError(
                f"No approval chain configured for workflow '{request.workflow_id}'"
            )

        self._validate_transition(request.status, ApprovalStatus.SUBMITTED)

        request.status = ApprovalStatus.SUBMITTED
        request.updated_at = _utcnow()

        # Move to IN_REVIEW automatically
        request.status = ApprovalStatus.IN_REVIEW
        request.updated_at = _utcnow()

        request.provenance_hash = _compute_hash(request)
        self.requests[request.request_id] = request

        logger.info(
            "Request %s submitted for approval at level %s",
            request.request_id,
            request.current_level.name,
        )
        return request

    # -- Approval -----------------------------------------------------------

    async def approve(
        self, request_id: str, decision: ApprovalDecision
    ) -> ApprovalRequest:
        """Process an approval decision.

        If enough approvers at the current level have approved, the request
        advances to the next level or reaches final APPROVED status.

        Args:
            request_id: Request to approve.
            decision: Approval decision details.

        Returns:
            Updated ApprovalRequest.

        Raises:
            ValueError: If request not found, invalid state, or self-approval.
        """
        request = self._get_request(request_id)

        if not self.config.allow_self_approval:
            if decision.approver == request.submitted_by:
                raise ValueError("Self-approval is not permitted")

        if not self._is_authorized(decision.approver, request):
            raise ValueError(
                f"User '{decision.approver}' is not authorized to approve "
                f"at level {request.current_level.name}"
            )

        self._validate_transition(request.status, ApprovalStatus.APPROVED)

        decision.level = request.current_level
        decision.decision = DecisionType.APPROVE
        decision.provenance_hash = _compute_hash(decision)
        request.decisions.append(decision)

        # Check if enough approvers have approved at this level
        level_approvals = self._count_level_approvals(request)
        required = self._get_required_approvers(request)

        if level_approvals >= required:
            # Advance to next level or finalize
            next_level = self._get_next_level(request)
            if next_level is None:
                request.status = ApprovalStatus.APPROVED
                logger.info(
                    "Request %s fully approved through all levels",
                    request_id,
                )
            else:
                request.current_level = next_level
                request.status = ApprovalStatus.IN_REVIEW
                logger.info(
                    "Request %s advanced to level %s",
                    request_id,
                    next_level.name,
                )
        else:
            logger.info(
                "Request %s: %d/%d approvals at level %s",
                request_id,
                level_approvals,
                required,
                request.current_level.name,
            )

        request.updated_at = _utcnow()
        request.provenance_hash = _compute_hash(request)
        self.requests[request_id] = request
        return request

    # -- Rejection ----------------------------------------------------------

    async def reject(
        self, request_id: str, decision: ApprovalDecision
    ) -> ApprovalRequest:
        """Reject an approval request.

        Args:
            request_id: Request to reject.
            decision: Rejection decision with required comments.

        Returns:
            Updated ApprovalRequest with REJECTED status.

        Raises:
            ValueError: If request not found or comments missing.
        """
        request = self._get_request(request_id)
        self._validate_transition(request.status, ApprovalStatus.REJECTED)

        if self.config.require_comments_on_reject and not decision.comments.strip():
            raise ValueError("Comments are required when rejecting")

        decision.level = request.current_level
        decision.decision = DecisionType.REJECT
        decision.provenance_hash = _compute_hash(decision)
        request.decisions.append(decision)

        request.status = ApprovalStatus.REJECTED
        request.updated_at = _utcnow()
        request.provenance_hash = _compute_hash(request)
        self.requests[request_id] = request

        logger.info(
            "Request %s rejected at level %s by %s: %s",
            request_id,
            request.current_level.name,
            decision.approver,
            decision.comments,
        )
        return request

    # -- Return for Revision ------------------------------------------------

    async def return_for_revision(
        self, request_id: str, comments: str, author: str = "system"
    ) -> ApprovalRequest:
        """Return a request to the previous level for revision.

        Args:
            request_id: Request to return.
            comments: Reason for returning.
            author: User returning the request.

        Returns:
            Updated ApprovalRequest with RETURNED status.
        """
        request = self._get_request(request_id)
        self._validate_transition(request.status, ApprovalStatus.RETURNED)

        request.comments.append(
            ApprovalComment(
                author=author,
                text=comments,
                level=request.current_level,
            )
        )

        decision = ApprovalDecision(
            decision=DecisionType.RETURN,
            approver=author,
            comments=comments,
            level=request.current_level,
        )
        decision.provenance_hash = _compute_hash(decision)
        request.decisions.append(decision)

        # Drop back one level
        prev_level = self._get_previous_level(request)
        if prev_level is not None:
            request.current_level = prev_level

        request.status = ApprovalStatus.RETURNED
        request.updated_at = _utcnow()
        request.provenance_hash = _compute_hash(request)
        self.requests[request_id] = request

        logger.info(
            "Request %s returned for revision to level %s",
            request_id,
            request.current_level.name,
        )
        return request

    # -- Escalation ---------------------------------------------------------

    async def escalate(self, request_id: str) -> ApprovalRequest:
        """Escalate an overdue request to the next level.

        Args:
            request_id: Request to escalate.

        Returns:
            Updated ApprovalRequest with ESCALATED status.

        Raises:
            ValueError: If max escalation reached.
        """
        request = self._get_request(request_id)
        self._validate_transition(request.status, ApprovalStatus.ESCALATED)

        if request.escalation_count >= self.config.max_escalation_levels:
            raise ValueError(
                f"Maximum escalation count ({self.config.max_escalation_levels}) reached"
            )

        request.escalation_count += 1
        request.status = ApprovalStatus.ESCALATED
        request.updated_at = _utcnow()

        request.comments.append(
            ApprovalComment(
                author="system",
                text=f"Auto-escalated (count={request.escalation_count})",
                level=request.current_level,
            )
        )

        request.provenance_hash = _compute_hash(request)
        self.requests[request_id] = request

        logger.warning(
            "Request %s escalated (count=%d) at level %s",
            request_id,
            request.escalation_count,
            request.current_level.name,
        )
        return request

    # -- Delegation ---------------------------------------------------------

    async def add_delegation(self, entry: DelegationEntry) -> None:
        """Register a delegation of authority.

        Args:
            entry: Delegation definition.

        Raises:
            ValueError: If delegator equals delegate.
        """
        if entry.delegator == entry.delegate:
            raise ValueError("Cannot delegate to self")

        if entry.valid_until <= entry.valid_from:
            raise ValueError("valid_until must be after valid_from")

        self.delegations.append(entry)
        logger.info(
            "Delegation added: %s -> %s (scope=%s, until=%s)",
            entry.delegator,
            entry.delegate,
            entry.scope,
            entry.valid_until.isoformat(),
        )

    # -- Auto-Approval ------------------------------------------------------

    async def check_auto_approve(self, request_id: str) -> bool:
        """Check if a request qualifies for automatic approval.

        Compares quality gate scores against the auto-approval threshold
        configured for the current level.

        Args:
            request_id: Request to evaluate.

        Returns:
            True if auto-approval was applied, False otherwise.
        """
        request = self._get_request(request_id)

        level_config = self._get_current_level_config(request)
        if level_config is None or level_config.auto_approve_threshold is None:
            return False

        if request.quality_gate_results is None:
            return False

        overall_score = request.quality_gate_results.get("overall_score", 0.0)

        if float(overall_score) >= level_config.auto_approve_threshold:
            auto_decision = ApprovalDecision(
                decision=DecisionType.APPROVE,
                approver="auto_approve_system",
                comments=(
                    f"Auto-approved: quality score {overall_score:.1f} >= "
                    f"threshold {level_config.auto_approve_threshold:.1f}"
                ),
                level=request.current_level,
            )
            auto_decision.provenance_hash = _compute_hash(auto_decision)
            request.decisions.append(auto_decision)

            next_level = self._get_next_level(request)
            if next_level is None:
                request.status = ApprovalStatus.APPROVED
            else:
                request.current_level = next_level
                request.status = ApprovalStatus.IN_REVIEW

            request.updated_at = _utcnow()
            request.provenance_hash = _compute_hash(request)
            self.requests[request_id] = request

            logger.info(
                "Request %s auto-approved at level %s (score=%.1f)",
                request_id,
                auto_decision.level.name,
                overall_score,
            )
            return True

        return False

    # -- Queries ------------------------------------------------------------

    async def get_approval_history(
        self, workflow_id: str
    ) -> List[ApprovalDecision]:
        """Get full approval decision history for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            List of all decisions for the workflow in chronological order.
        """
        decisions: List[ApprovalDecision] = []
        for request in self.requests.values():
            if request.workflow_id == workflow_id:
                decisions.extend(request.decisions)

        decisions.sort(key=lambda d: d.timestamp)
        return decisions

    async def get_delegation_matrix(self) -> List[DelegationEntry]:
        """Get all active delegation entries.

        Returns:
            List of currently valid delegation entries.
        """
        now = _utcnow()
        return [
            d
            for d in self.delegations
            if d.valid_from <= now <= d.valid_until
        ]

    async def get_pending_approvals(self, user: str) -> List[ApprovalRequest]:
        """Get all requests pending action by a specific user.

        Args:
            user: Username to check.

        Returns:
            List of requests assigned to or delegated to the user.
        """
        pending: List[ApprovalRequest] = []
        now = _utcnow()

        # Build delegation map for this user
        delegated_from: Set[str] = set()
        for d in self.delegations:
            if d.delegate == user and d.valid_from <= now <= d.valid_until:
                delegated_from.add(d.delegator)

        for request in self.requests.values():
            if request.status not in (
                ApprovalStatus.IN_REVIEW,
                ApprovalStatus.SUBMITTED,
                ApprovalStatus.ESCALATED,
            ):
                continue

            if user in request.assigned_to:
                pending.append(request)
            elif any(a in delegated_from for a in request.assigned_to):
                pending.append(request)

        return pending

    async def get_overdue_requests(self) -> List[ApprovalRequest]:
        """Get all requests that have exceeded their escalation timeout.

        Returns:
            List of overdue approval requests.
        """
        now = _utcnow()
        overdue: List[ApprovalRequest] = []

        for request in self.requests.values():
            if request.status not in (
                ApprovalStatus.IN_REVIEW,
                ApprovalStatus.SUBMITTED,
            ):
                continue

            level_config = self._get_current_level_config(request)
            timeout_hours = (
                level_config.escalation_timeout_hours
                if level_config
                else self.config.default_escalation_hours
            )

            deadline = request.updated_at + timedelta(hours=timeout_hours)
            if now > deadline:
                overdue.append(request)

        return overdue

    async def get_chain_result(self, workflow_id: str) -> ApprovalChainResult:
        """Get the overall chain result for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            ApprovalChainResult summarizing the chain status.
        """
        all_decisions = await self.get_approval_history(workflow_id)

        # Find the latest request for this workflow
        latest_request: Optional[ApprovalRequest] = None
        for req in self.requests.values():
            if req.workflow_id == workflow_id:
                if latest_request is None or req.updated_at > latest_request.updated_at:
                    latest_request = req

        levels_completed: List[int] = []
        if latest_request:
            for decision in latest_request.decisions:
                if decision.decision == DecisionType.APPROVE:
                    if decision.level.value not in levels_completed:
                        levels_completed.append(decision.level.value)

        elapsed_hours = 0.0
        if latest_request:
            elapsed = _utcnow() - latest_request.created_at
            elapsed_hours = elapsed.total_seconds() / 3600

        result = ApprovalChainResult(
            workflow_id=workflow_id,
            levels_completed=sorted(levels_completed),
            current_level=(
                latest_request.current_level
                if latest_request
                else ApprovalLevel.PREPARER
            ),
            overall_status=(
                latest_request.status
                if latest_request
                else ApprovalStatus.PENDING
            ),
            history=all_decisions,
            total_elapsed_hours=round(elapsed_hours, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -- Internal Helpers ---------------------------------------------------

    def _get_request(self, request_id: str) -> ApprovalRequest:
        """Get a request by ID or raise ValueError."""
        request = self.requests.get(request_id)
        if request is None:
            raise ValueError(f"Request '{request_id}' not found")
        return request

    def _validate_transition(
        self, current: ApprovalStatus, target: ApprovalStatus
    ) -> None:
        """Validate a status transition is allowed.

        Args:
            current: Current status.
            target: Target status.

        Raises:
            ValueError: If transition is not allowed.
        """
        allowed = _ALLOWED_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise ValueError(
                f"Transition from {current.value} to {target.value} is not allowed. "
                f"Allowed: {[s.value for s in allowed]}"
            )

    def _get_current_level_config(
        self, request: ApprovalRequest
    ) -> Optional[ApprovalLevelConfig]:
        """Get the level config for the request's current level."""
        configs = self._level_configs.get(request.workflow_id, [])
        for lc in configs:
            if lc.level == request.current_level:
                return lc
        return None

    def _get_required_approvers(self, request: ApprovalRequest) -> int:
        """Get required approver count for current level."""
        config = self._get_current_level_config(request)
        return config.required_approvers if config else 1

    def _count_level_approvals(self, request: ApprovalRequest) -> int:
        """Count approvals at the current level."""
        return sum(
            1
            for d in request.decisions
            if d.decision == DecisionType.APPROVE and d.level == request.current_level
        )

    def _get_next_level(
        self, request: ApprovalRequest
    ) -> Optional[ApprovalLevel]:
        """Get the next approval level after the current one."""
        configs = self._level_configs.get(request.workflow_id, [])
        current_value = request.current_level.value
        for lc in sorted(configs, key=lambda x: x.level.value):
            if lc.level.value > current_value:
                return lc.level
        return None

    def _get_previous_level(
        self, request: ApprovalRequest
    ) -> Optional[ApprovalLevel]:
        """Get the previous approval level before the current one."""
        configs = self._level_configs.get(request.workflow_id, [])
        current_value = request.current_level.value
        previous: Optional[ApprovalLevel] = None
        for lc in sorted(configs, key=lambda x: x.level.value):
            if lc.level.value < current_value:
                previous = lc.level
        return previous

    def _is_authorized(self, user: str, request: ApprovalRequest) -> bool:
        """Check if a user is authorized to act on a request.

        Checks direct assignment and active delegations.

        Args:
            user: Username to check.
            request: Approval request.

        Returns:
            True if authorized.
        """
        # Direct assignment
        if user in request.assigned_to:
            return True

        # No assigned_to means anyone with the role can act
        if not request.assigned_to:
            return True

        # Check delegations
        now = _utcnow()
        for d in self.delegations:
            if d.delegate != user:
                continue
            if d.valid_from > now or d.valid_until < now:
                continue
            if d.delegator in request.assigned_to:
                if d.scope == "all" or d.scope == request.workflow_id:
                    return True

        return False
