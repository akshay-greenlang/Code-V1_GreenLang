# -*- coding: utf-8 -*-
"""
Methodology review workflow (F023).

Extends the basic Q3 review queue with structured review assignments,
10-point methodology checklist tracking, batch review decisions, and
review lifecycle management.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

# Import consensus + SLA pieces lazily at module scope so the existing state
# machine keeps running even if callers only want the original behaviour.
from greenlang.factors.quality.consensus import (  # noqa: F401 (re-exported)
    ConsensusConfig,
    ConsensusResult,
    ConsensusStatus,
    DissentCaptureRequiredError,
    InsufficientConsensusError,
    ReviewerVote,
    VoteDecision,
    evaluate_consensus,
    load_votes as _load_votes,
    record_vote as _record_vote,
    tier_based_requirements,
)
from greenlang.factors.quality.sla import (  # noqa: F401 (re-exported)
    SLAExpiredError,
    SLAStage,
    SLATimer,
    SLATimerStatus,
    complete_timer as _complete_sla_timer,
    get_timer_for_factor as _get_sla_timer,
    start_sla_timer as _start_sla_timer,
)

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class ReviewPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 10-point methodology review checklist
METHODOLOGY_CHECKLIST = [
    {"id": "C01", "label": "Source provenance verified", "description": "Source URL, publication, and version confirmed"},
    {"id": "C02", "label": "GWP basis correct", "description": "AR5/AR6 GWP values match source documentation"},
    {"id": "C03", "label": "Gas vectors complete", "description": "CO2, CH4, N2O present with correct units"},
    {"id": "C04", "label": "Unit conversion verified", "description": "Activity unit matches source; conversion factors correct"},
    {"id": "C05", "label": "Geography mapping accurate", "description": "ISO-3166 code correct for source geography"},
    {"id": "C06", "label": "Scope/boundary alignment", "description": "Scope 1/2/3 and boundary match GHG Protocol definitions"},
    {"id": "C07", "label": "Temporal validity window", "description": "valid_from/valid_to dates match source publication period"},
    {"id": "C08", "label": "DQS scores justified", "description": "Data quality scores reflect actual data provenance"},
    {"id": "C09", "label": "License compliance", "description": "Factor distribution rights match source license terms"},
    {"id": "C10", "label": "Cross-source plausibility", "description": "Values within expected range compared to similar sources"},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ChecklistItem:
    """A single checklist item with pass/fail status."""

    item_id: str
    label: str
    passed: Optional[bool] = None  # None = not yet reviewed
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "passed": self.passed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChecklistItem":
        return cls(
            item_id=data["item_id"],
            label=data.get("label", ""),
            passed=data.get("passed"),
            notes=data.get("notes", ""),
        )


@dataclass
class ReviewAssignment:
    """A methodology review assignment for a set of factors."""

    review_id: str
    edition_id: str
    factor_ids: List[str]
    source_id: str
    reviewer: str
    status: ReviewStatus = ReviewStatus.PENDING
    priority: ReviewPriority = ReviewPriority.MEDIUM
    checklist: List[ChecklistItem] = field(default_factory=list)
    decision: Optional[str] = None  # approved | rejected | needs_revision
    decision_notes: str = ""
    created_at: str = field(default_factory=_utc_now)
    due_date: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.checklist:
            self.checklist = [
                ChecklistItem(item_id=c["id"], label=c["label"])
                for c in METHODOLOGY_CHECKLIST
            ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "edition_id": self.edition_id,
            "factor_ids": self.factor_ids,
            "source_id": self.source_id,
            "reviewer": self.reviewer,
            "status": self.status.value,
            "priority": self.priority.value,
            "checklist": [c.to_dict() for c in self.checklist],
            "decision": self.decision,
            "decision_notes": self.decision_notes,
            "created_at": self.created_at,
            "due_date": self.due_date,
            "completed_at": self.completed_at,
            "checklist_complete": self.checklist_complete,
            "checklist_pass_count": self.checklist_pass_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewAssignment":
        checklist = [ChecklistItem.from_dict(c) for c in data.get("checklist", [])]
        return cls(
            review_id=data["review_id"],
            edition_id=data["edition_id"],
            factor_ids=data.get("factor_ids", []),
            source_id=data.get("source_id", ""),
            reviewer=data.get("reviewer", ""),
            status=ReviewStatus(data.get("status", "pending")),
            priority=ReviewPriority(data.get("priority", "medium")),
            checklist=checklist,
            decision=data.get("decision"),
            decision_notes=data.get("decision_notes", ""),
            created_at=data.get("created_at", ""),
            due_date=data.get("due_date"),
            completed_at=data.get("completed_at"),
        )

    @property
    def checklist_complete(self) -> bool:
        """True if all checklist items have been reviewed (passed is not None)."""
        return all(c.passed is not None for c in self.checklist)

    @property
    def checklist_pass_count(self) -> int:
        return sum(1 for c in self.checklist if c.passed is True)

    @property
    def checklist_fail_count(self) -> int:
        return sum(1 for c in self.checklist if c.passed is False)

    @property
    def all_passed(self) -> bool:
        return self.checklist_complete and self.checklist_fail_count == 0


def create_review(
    edition_id: str,
    factor_ids: List[str],
    source_id: str,
    reviewer: str,
    *,
    priority: ReviewPriority = ReviewPriority.MEDIUM,
    due_date: Optional[str] = None,
) -> ReviewAssignment:
    """Create a new methodology review assignment."""
    review = ReviewAssignment(
        review_id=str(uuid.uuid4()),
        edition_id=edition_id,
        factor_ids=factor_ids,
        source_id=source_id,
        reviewer=reviewer,
        priority=priority,
        due_date=due_date,
    )
    logger.info(
        "Created review %s: edition=%s source=%s factors=%d reviewer=%s",
        review.review_id, edition_id, source_id, len(factor_ids), reviewer,
    )
    return review


def update_checklist_item(
    review: ReviewAssignment,
    item_id: str,
    passed: bool,
    notes: str = "",
) -> bool:
    """
    Update a checklist item on a review.

    Returns True if the item was found and updated.
    """
    for item in review.checklist:
        if item.item_id == item_id:
            item.passed = passed
            if notes:
                item.notes = notes
            logger.debug("Updated checklist item %s on review %s: passed=%s", item_id, review.review_id, passed)
            return True
    return False


def submit_decision(
    review: ReviewAssignment,
    decision: str,
    notes: str = "",
) -> ReviewAssignment:
    """
    Submit a review decision (approved/rejected/needs_revision).

    Raises ValueError if checklist is incomplete.
    """
    if decision not in ("approved", "rejected", "needs_revision"):
        raise ValueError(f"Invalid decision: {decision!r}")

    if decision == "approved" and not review.checklist_complete:
        raise ValueError("Cannot approve: checklist is incomplete")

    review.decision = decision
    review.decision_notes = notes
    review.completed_at = _utc_now()

    if decision == "approved":
        review.status = ReviewStatus.APPROVED
    elif decision == "rejected":
        review.status = ReviewStatus.REJECTED
    else:
        review.status = ReviewStatus.NEEDS_REVISION

    logger.info(
        "Review %s decision: %s (%d/%d checklist passed)",
        review.review_id, decision, review.checklist_pass_count, len(review.checklist),
    )
    return review


def batch_review(
    reviews: Sequence[ReviewAssignment],
    decision: str,
    notes: str = "",
    *,
    auto_approve_checklist: bool = False,
) -> List[ReviewAssignment]:
    """
    Apply the same decision to multiple reviews.

    Args:
        reviews: List of ReviewAssignment objects.
        decision: Decision to apply.
        notes: Notes for the decision.
        auto_approve_checklist: If True, mark all checklist items as passed before approving.

    Returns:
        List of updated reviews (skipping any that raise ValueError).
    """
    updated = []
    for review in reviews:
        if auto_approve_checklist and decision == "approved":
            for item in review.checklist:
                if item.passed is None:
                    item.passed = True
        try:
            submit_decision(review, decision, notes)
            updated.append(review)
        except ValueError as exc:
            logger.warning("Batch review skipped %s: %s", review.review_id, exc)
    logger.info("Batch review: %d/%d reviews updated with decision=%s", len(updated), len(reviews), decision)
    return updated


def save_review_to_sqlite(
    conn: sqlite3.Connection,
    review: ReviewAssignment,
) -> None:
    """Persist a review assignment to the qa_reviews table."""
    payload = review.to_dict()
    for fid in review.factor_ids:
        conn.execute(
            """
            INSERT INTO qa_reviews (review_id, edition_id, factor_id, status, payload_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(review_id) DO UPDATE SET
                status=excluded.status,
                payload_json=excluded.payload_json
            """,
            (review.review_id, review.edition_id, fid, review.status.value,
             json.dumps(payload, sort_keys=True, default=str)),
        )


# ---------------------------------------------------------------------------
# Workflow lifecycle integration (GAP-14 + GAP-15)
# ---------------------------------------------------------------------------


class WorkflowState(str, Enum):
    """High-level factor lifecycle state used by consensus + SLA layers."""

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


# Stage that the OUTGOING state owns (used to complete a timer when leaving).
_STATE_STAGE_MAP: Dict[WorkflowState, SLAStage] = {
    WorkflowState.DRAFT: SLAStage.INITIAL_REVIEW,
    WorkflowState.UNDER_REVIEW: SLAStage.DETAILED_REVIEW,
    WorkflowState.APPROVED: SLAStage.FINAL_APPROVAL,
    WorkflowState.PUBLISHED: SLAStage.DEPRECATION_NOTICE,
}


# Stage the INCOMING state should start a timer for.
_STATE_NEW_TIMER: Dict[WorkflowState, Optional[SLAStage]] = {
    WorkflowState.DRAFT: None,
    WorkflowState.UNDER_REVIEW: SLAStage.DETAILED_REVIEW,
    WorkflowState.APPROVED: SLAStage.FINAL_APPROVAL,
    WorkflowState.PUBLISHED: SLAStage.DEPRECATION_NOTICE,
    WorkflowState.DEPRECATED: None,
    WorkflowState.REJECTED: None,
}


_ALLOWED_TRANSITIONS: Dict[WorkflowState, set] = {
    WorkflowState.DRAFT: {WorkflowState.UNDER_REVIEW, WorkflowState.REJECTED},
    WorkflowState.UNDER_REVIEW: {
        WorkflowState.APPROVED,
        WorkflowState.DRAFT,  # needs_revision
        WorkflowState.REJECTED,
    },
    WorkflowState.APPROVED: {
        WorkflowState.PUBLISHED,
        WorkflowState.REJECTED,
    },
    WorkflowState.PUBLISHED: {WorkflowState.DEPRECATED},
    WorkflowState.DEPRECATED: set(),
    WorkflowState.REJECTED: set(),
}


def _coerce_state(value: Any) -> WorkflowState:
    if isinstance(value, WorkflowState):
        return value
    return WorkflowState(str(value).lower())


@dataclass
class TransitionOutcome:
    """Result of a workflow transition request."""

    factor_id: str
    from_state: WorkflowState
    to_state: WorkflowState
    allowed: bool
    consensus: Optional[ConsensusResult]
    completed_timer: Optional[SLATimer]
    new_timer: Optional[SLATimer]
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "allowed": self.allowed,
            "consensus": self.consensus.to_dict() if self.consensus else None,
            "completed_timer": (
                self.completed_timer.to_dict() if self.completed_timer else None
            ),
            "new_timer": self.new_timer.to_dict() if self.new_timer else None,
            "reason": self.reason,
        }


def advance_workflow_state(
    conn: sqlite3.Connection,
    factor_id: str,
    from_state: Any,
    to_state: Any,
    *,
    tier: str,
    factor_type: str = "",
    config: Optional[ConsensusConfig] = None,
    factor_author: Optional[str] = None,
    now: Optional[datetime] = None,
    require_consensus: bool = True,
) -> TransitionOutcome:
    """Advance a factor through its lifecycle with consensus + SLA checks.

    This function extends (does not replace) the existing
    :func:`submit_decision` machinery -- callers can still use the old API
    for the per-review checklist flow while letting this helper govern the
    factor-level state transitions consumed by the API routes.

    Args:
        conn: SQLite connection that backs both consensus and SLA tables.
        factor_id: Factor being moved.
        from_state / to_state: Lifecycle states.
        tier: Tenant tier (``community`` / ``pro`` / ``enterprise`` /
            ``enterprise_cbam``).
        factor_type: Optional factor type used to pick regulatory policy.
        config: Explicit consensus config (overrides ``tier_based_requirements``).
        factor_author: Used to reject self-approval.
        now: Override for deterministic tests.
        require_consensus: If False, skip consensus evaluation (used for
            deprecation/rejection flows).

    Raises:
        InsufficientConsensusError: If consensus is required and not met.
        SLAExpiredError: If the outgoing stage's timer has auto-rejected.
        ValueError: For illegal state transitions.
    """
    src = _coerce_state(from_state)
    dst = _coerce_state(to_state)
    now = now or datetime.now(timezone.utc)

    if dst not in _ALLOWED_TRANSITIONS.get(src, set()):
        raise ValueError(
            "Illegal state transition %s -> %s for factor %s"
            % (src.value, dst.value, factor_id)
        )

    # 1. SLA gate on outgoing stage.
    outgoing_stage = _STATE_STAGE_MAP.get(src)
    completed_timer: Optional[SLATimer] = None
    if outgoing_stage is not None:
        existing = _get_sla_timer(conn, factor_id, outgoing_stage)
        if existing is not None and existing.status == SLATimerStatus.EXPIRED:
            raise SLAExpiredError(factor_id, outgoing_stage.value, existing.deadline)

    # 2. Consensus gate (only on positive transitions).
    consensus_result: Optional[ConsensusResult] = None
    advancing = dst in (
        WorkflowState.UNDER_REVIEW,
        WorkflowState.APPROVED,
        WorkflowState.PUBLISHED,
    )
    if require_consensus and advancing:
        cfg = config or tier_based_requirements(tier=tier, factor_type=factor_type)
        votes = _load_votes(conn, factor_id)
        consensus_result = evaluate_consensus(
            factor_id, votes, cfg, factor_author=factor_author
        )
        if consensus_result.status != ConsensusStatus.APPROVED:
            logger.info(
                "Transition blocked for factor=%s (%s): consensus=%s reason=%s",
                factor_id, dst.value, consensus_result.status.value,
                consensus_result.reason,
            )
            raise InsufficientConsensusError(
                factor_id,
                "Consensus not met for %s -> %s: %s"
                % (src.value, dst.value, consensus_result.reason),
                met_requirements=consensus_result.met_requirements,
                missing=[
                    r.role
                    for r in cfg.reviewer_requirements
                    if consensus_result.met_requirements.get(r.role, 0)
                    < r.min_count
                ],
            )

    # 3. Complete outgoing timer.
    if outgoing_stage is not None:
        completed_timer = _complete_sla_timer(conn, factor_id, outgoing_stage, now=now)

    # 4. Start timer for incoming stage (if any).
    new_stage = _STATE_NEW_TIMER.get(dst)
    new_timer: Optional[SLATimer] = None
    if new_stage is not None:
        new_timer = _start_sla_timer(conn, factor_id, new_stage, tier, now=now)

    logger.info(
        "Transition factor=%s %s -> %s completed_timer=%s new_timer=%s",
        factor_id, src.value, dst.value,
        completed_timer.timer_id if completed_timer else None,
        new_timer.timer_id if new_timer else None,
    )
    return TransitionOutcome(
        factor_id=factor_id,
        from_state=src,
        to_state=dst,
        allowed=True,
        consensus=consensus_result,
        completed_timer=completed_timer,
        new_timer=new_timer,
        reason="ok",
    )


def submit_vote(
    conn: sqlite3.Connection,
    vote: ReviewerVote,
) -> None:
    """Persist a reviewer's vote via the consensus engine."""
    _record_vote(conn, vote)


def load_reviews_from_sqlite(
    conn: sqlite3.Connection,
    edition_id: str,
    *,
    status: Optional[str] = None,
    reviewer: Optional[str] = None,
) -> List[ReviewAssignment]:
    """Load review assignments from qa_reviews table."""
    clauses = ["edition_id = ?"]
    params: List[Any] = [edition_id]
    if status:
        clauses.append("status = ?")
        params.append(status)

    where = " AND ".join(clauses)
    cur = conn.execute(
        f"SELECT DISTINCT review_id, payload_json FROM qa_reviews WHERE {where} ORDER BY review_id",
        params,
    )
    reviews = []
    seen: set = set()
    for row in cur.fetchall():
        rid = row[0] if isinstance(row, (list, tuple)) else row["review_id"]
        if rid in seen:
            continue
        seen.add(rid)
        payload_str = row[1] if isinstance(row, (list, tuple)) else row["payload_json"]
        data = json.loads(payload_str)
        review = ReviewAssignment.from_dict(data)
        if reviewer and review.reviewer != reviewer:
            continue
        reviews.append(review)
    return reviews
