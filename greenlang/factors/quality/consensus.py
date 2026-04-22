# -*- coding: utf-8 -*-
"""
Multi-reviewer consensus engine (GAP-14).

Provides an N-of-M approval engine used by the methodology review workflow
to decide whether a factor may advance between lifecycle states.  Supports
four consensus rules:

- ``ANY_OF_N``   -- a single approver is sufficient
- ``N_OF_M``     -- a configurable subset of reviewers must approve
- ``UNANIMOUS``  -- every named reviewer must approve
- ``WEIGHTED``   -- role-weighted voting (e.g., methodology lead weight 2)

The engine exposes both in-memory evaluation (pure function) and SQLite-backed
persistence so the Factors review queue, the FactorsApprovalQueue.tsx surface,
and the QA dashboard share a single source of truth for dissent and quorum.

This module is intentionally stand-alone so it can be imported by
:mod:`review_workflow` without introducing a circular dependency with the
existing state machine.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions (re-exported via review_workflow; not added to greenlang.exceptions)
# ---------------------------------------------------------------------------


class InsufficientConsensusError(RuntimeError):
    """Raised when a state transition is attempted before consensus is reached."""

    def __init__(
        self,
        factor_id: str,
        message: str,
        *,
        met_requirements: Optional[Mapping[str, int]] = None,
        missing: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(message)
        self.factor_id = factor_id
        self.met_requirements = dict(met_requirements or {})
        self.missing = list(missing or [])


class DissentCaptureRequiredError(ValueError):
    """Raised when a REJECT/ABSTAIN vote is cast without a dissent note."""


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------


class ConsensusRule(str, Enum):
    """Supported consensus rules."""

    ANY_OF_N = "any_of_n"
    N_OF_M = "n_of_m"
    UNANIMOUS = "unanimous"
    WEIGHTED = "weighted"


class VoteDecision(str, Enum):
    """Individual reviewer vote outcomes."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class ConsensusStatus(str, Enum):
    """Overall consensus evaluation status."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INSUFFICIENT_QUORUM = "INSUFFICIENT_QUORUM"


# Known reviewer roles. Using a closed set keeps policy predictable; unknown
# roles still vote but cannot satisfy role-specific requirements.
REVIEWER_ROLES: Tuple[str, ...] = (
    "methodology_lead",
    "qa_lead",
    "compliance_lead",
    "legal_lead",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewerRequirement:
    """Minimum number of approvals for a given reviewer role."""

    role: str
    min_count: int
    weight: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "min_count": self.min_count, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewerRequirement":
        return cls(
            role=str(data["role"]),
            min_count=int(data.get("min_count", 1)),
            weight=int(data.get("weight", 1)),
        )


@dataclass(frozen=True)
class ConsensusConfig:
    """Configuration describing the consensus rule for a factor/tier."""

    rule: ConsensusRule
    reviewer_requirements: Tuple[ReviewerRequirement, ...]
    quorum: int
    allow_self_approval: bool = False
    dissent_capture_required: bool = True
    sla_hours: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule.value,
            "reviewer_requirements": [r.to_dict() for r in self.reviewer_requirements],
            "quorum": self.quorum,
            "allow_self_approval": self.allow_self_approval,
            "dissent_capture_required": self.dissent_capture_required,
            "sla_hours": self.sla_hours,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConsensusConfig":
        reqs_raw = data.get("reviewer_requirements", [])
        reqs = tuple(ReviewerRequirement.from_dict(r) for r in reqs_raw)
        return cls(
            rule=ConsensusRule(data.get("rule", ConsensusRule.ANY_OF_N.value)),
            reviewer_requirements=reqs,
            quorum=int(data.get("quorum", 1)),
            allow_self_approval=bool(data.get("allow_self_approval", False)),
            dissent_capture_required=bool(data.get("dissent_capture_required", True)),
            sla_hours=(
                int(data["sla_hours"]) if data.get("sla_hours") is not None else None
            ),
        )


@dataclass
class ReviewerVote:
    """Single reviewer decision on a factor."""

    vote_id: str
    factor_id: str
    reviewer_id: str
    reviewer_role: str
    decision: VoteDecision
    rationale: Optional[str] = None
    dissent_notes: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0)
    )
    weight: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vote_id": self.vote_id,
            "factor_id": self.factor_id,
            "reviewer_id": self.reviewer_id,
            "reviewer_role": self.reviewer_role,
            "decision": self.decision.value,
            "rationale": self.rationale,
            "dissent_notes": self.dissent_notes,
            "timestamp": self.timestamp.isoformat(),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewerVote":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = datetime.now(timezone.utc).replace(microsecond=0)
        return cls(
            vote_id=str(data["vote_id"]),
            factor_id=str(data["factor_id"]),
            reviewer_id=str(data["reviewer_id"]),
            reviewer_role=str(data["reviewer_role"]),
            decision=VoteDecision(str(data["decision"]).upper()),
            rationale=data.get("rationale"),
            dissent_notes=data.get("dissent_notes"),
            timestamp=timestamp,
            weight=int(data.get("weight", 1)),
        )


@dataclass
class DissentNote:
    """Extracted dissent note for QA dashboards."""

    factor_id: str
    reviewer_id: str
    reviewer_role: str
    decision: VoteDecision
    notes: str
    recorded_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "reviewer_id": self.reviewer_id,
            "reviewer_role": self.reviewer_role,
            "decision": self.decision.value,
            "notes": self.notes,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass
class ConsensusResult:
    """Outcome of evaluating a vote set against a consensus configuration."""

    factor_id: str
    consensus_rule: ConsensusRule
    status: ConsensusStatus
    votes: List[ReviewerVote]
    met_requirements: Dict[str, int]
    dissent_captured: bool
    decided_at: Optional[datetime] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "consensus_rule": self.consensus_rule.value,
            "status": self.status.value,
            "votes": [v.to_dict() for v in self.votes],
            "met_requirements": dict(self.met_requirements),
            "dissent_captured": self.dissent_captured,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "reason": self.reason,
            "approval_count": self.approval_count,
            "rejection_count": self.rejection_count,
        }

    @property
    def approval_count(self) -> int:
        return sum(1 for v in self.votes if v.decision == VoteDecision.APPROVE)

    @property
    def rejection_count(self) -> int:
        return sum(1 for v in self.votes if v.decision == VoteDecision.REJECT)

    @property
    def is_approved(self) -> bool:
        return self.status == ConsensusStatus.APPROVED


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _validate_vote(
    vote: ReviewerVote,
    *,
    config: ConsensusConfig,
    factor_author: Optional[str] = None,
) -> None:
    """Sanity-check a single vote against config constraints.

    Raises:
        ValueError: If self-approval is attempted but not allowed.
        DissentCaptureRequiredError: If a REJECT/ABSTAIN vote has no dissent note.
    """
    if (
        not config.allow_self_approval
        and factor_author
        and vote.reviewer_id == factor_author
    ):
        raise ValueError(
            "Self-approval not permitted for factor %s by reviewer %s"
            % (vote.factor_id, vote.reviewer_id)
        )
    if (
        config.dissent_capture_required
        and vote.decision in (VoteDecision.REJECT, VoteDecision.ABSTAIN)
        and not (vote.dissent_notes or "").strip()
    ):
        raise DissentCaptureRequiredError(
            "Dissent notes required for %s vote by %s on factor %s"
            % (vote.decision.value, vote.reviewer_id, vote.factor_id)
        )


def _dedupe_votes(votes: Sequence[ReviewerVote]) -> List[ReviewerVote]:
    """Keep the most recent vote per reviewer (reviewers may update their vote)."""
    latest: Dict[str, ReviewerVote] = {}
    for vote in votes:
        existing = latest.get(vote.reviewer_id)
        if existing is None or vote.timestamp >= existing.timestamp:
            latest[vote.reviewer_id] = vote
    return list(latest.values())


def _role_counts(
    approvals: Sequence[ReviewerVote],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for vote in approvals:
        counts[vote.reviewer_role] = counts.get(vote.reviewer_role, 0) + 1
    return counts


def _role_weight(
    approvals: Sequence[ReviewerVote],
) -> Dict[str, int]:
    weights: Dict[str, int] = {}
    for vote in approvals:
        weights[vote.reviewer_role] = weights.get(vote.reviewer_role, 0) + int(vote.weight)
    return weights


def _dissent_captured(
    votes: Sequence[ReviewerVote],
    *,
    required: bool = True,
) -> bool:
    """True if every REJECT/ABSTAIN vote has non-empty dissent notes.

    When ``required`` is False (config.dissent_capture_required=False),
    we treat dissent-capture as vacuously satisfied — there is no
    obligation to capture dissent, so the flag cannot fail.  This matches
    the contract expected by ``evaluate_consensus`` callers where a
    config explicitly opts out of dissent capture.
    """
    if not required:
        return True
    for vote in votes:
        if vote.decision in (VoteDecision.REJECT, VoteDecision.ABSTAIN):
            if not (vote.dissent_notes or "").strip():
                return False
    return True


def evaluate_consensus(
    factor_id: str,
    votes: Sequence[ReviewerVote],
    config: ConsensusConfig,
    *,
    factor_author: Optional[str] = None,
) -> ConsensusResult:
    """Evaluate a set of votes against a consensus configuration.

    The function is pure -- it does not write to the database and does not
    mutate any inputs.  Latest vote per reviewer wins.

    Args:
        factor_id: Identifier of the factor being decided.
        votes: All reviewer votes observed so far.
        config: Consensus configuration (rule, quorum, dissent requirement).
        factor_author: Optional author id, used to enforce the
            ``allow_self_approval=False`` rule.

    Returns:
        A :class:`ConsensusResult` describing the decision.
    """
    normalized = _dedupe_votes(votes)

    # Vote-level validation (non-fatal: collected into result via reason text)
    reason_parts: List[str] = []
    valid_votes: List[ReviewerVote] = []
    for vote in normalized:
        try:
            _validate_vote(vote, config=config, factor_author=factor_author)
            valid_votes.append(vote)
        except (ValueError, DissentCaptureRequiredError) as exc:
            logger.warning(
                "Rejecting vote %s on factor %s: %s",
                vote.vote_id, factor_id, exc,
            )
            reason_parts.append(str(exc))

    approvals = [v for v in valid_votes if v.decision == VoteDecision.APPROVE]
    rejections = [v for v in valid_votes if v.decision == VoteDecision.REJECT]
    met = _role_counts(approvals)

    dissent_ok = _dissent_captured(
        valid_votes, required=config.dissent_capture_required
    )

    # A single rejection short-circuits UNANIMOUS and triggers REJECTED.
    if rejections:
        logger.info(
            "Consensus for factor %s: REJECTED (%d rejection(s))",
            factor_id, len(rejections),
        )
        return ConsensusResult(
            factor_id=factor_id,
            consensus_rule=config.rule,
            status=ConsensusStatus.REJECTED,
            votes=valid_votes,
            met_requirements=met,
            dissent_captured=dissent_ok,
            decided_at=max((v.timestamp for v in rejections), default=None),
            reason="; ".join(reason_parts) or "one or more reviewers rejected",
        )

    # Quorum check is universal.
    if len(approvals) < max(0, int(config.quorum)):
        return ConsensusResult(
            factor_id=factor_id,
            consensus_rule=config.rule,
            status=ConsensusStatus.INSUFFICIENT_QUORUM,
            votes=valid_votes,
            met_requirements=met,
            dissent_captured=dissent_ok,
            reason="need quorum=%d, have %d approvals"
            % (config.quorum, len(approvals)),
        )

    status = _apply_rule(config, approvals, met)
    if status == ConsensusStatus.APPROVED and not dissent_ok and valid_votes:
        # If dissent is required but missing on any non-approve vote, hold pending.
        return ConsensusResult(
            factor_id=factor_id,
            consensus_rule=config.rule,
            status=ConsensusStatus.PENDING,
            votes=valid_votes,
            met_requirements=met,
            dissent_captured=False,
            reason="dissent notes missing on non-approve vote",
        )

    decided_at: Optional[datetime]
    if status == ConsensusStatus.APPROVED and approvals:
        decided_at = max(v.timestamp for v in approvals)
    else:
        decided_at = None

    logger.info(
        "Consensus for factor %s: %s (rule=%s approvals=%d quorum=%d)",
        factor_id, status.value, config.rule.value, len(approvals), config.quorum,
    )
    return ConsensusResult(
        factor_id=factor_id,
        consensus_rule=config.rule,
        status=status,
        votes=valid_votes,
        met_requirements=met,
        dissent_captured=dissent_ok,
        decided_at=decided_at,
        reason="; ".join(reason_parts),
    )


def _apply_rule(
    config: ConsensusConfig,
    approvals: Sequence[ReviewerVote],
    met: Mapping[str, int],
) -> ConsensusStatus:
    """Apply the configured rule to an approval set that already passes quorum."""
    requirements = config.reviewer_requirements

    if config.rule == ConsensusRule.ANY_OF_N:
        return (
            ConsensusStatus.APPROVED if approvals else ConsensusStatus.INSUFFICIENT_QUORUM
        )

    if config.rule == ConsensusRule.N_OF_M:
        for req in requirements:
            if met.get(req.role, 0) < req.min_count:
                return ConsensusStatus.PENDING
        return ConsensusStatus.APPROVED

    if config.rule == ConsensusRule.UNANIMOUS:
        # No rejections have reached this branch (short-circuited above),
        # so the question is whether every required role has a quorum.
        required_total = sum(r.min_count for r in requirements) or len(requirements)
        if len(approvals) < required_total:
            return ConsensusStatus.PENDING
        for req in requirements:
            if met.get(req.role, 0) < max(1, req.min_count):
                return ConsensusStatus.PENDING
        return ConsensusStatus.APPROVED

    if config.rule == ConsensusRule.WEIGHTED:
        role_weight = _role_weight(approvals)
        for req in requirements:
            needed = req.min_count * max(1, req.weight)
            if role_weight.get(req.role, 0) < needed:
                return ConsensusStatus.PENDING
        return ConsensusStatus.APPROVED

    logger.warning("Unknown consensus rule %r; defaulting to PENDING", config.rule)
    return ConsensusStatus.PENDING


# ---------------------------------------------------------------------------
# Tier-based default configurations
# ---------------------------------------------------------------------------


# Regulatory factor types that require an explicit compliance_lead sign-off.
REGULATORY_FACTOR_TYPES: Tuple[str, ...] = (
    "cbam",
    "eudr",
    "csrd",
    "regulatory",
)


def tier_based_requirements(
    tier: str,
    factor_type: str = "",
) -> ConsensusConfig:
    """Return the default consensus configuration for a (tier, factor_type) pair.

    The defaults follow the GAP-14 matrix:

    - **Community**: one reviewer (``ANY_OF_N`` / quorum=1), 72h SLA.
    - **Pro**: ``N_OF_M`` 2-of-3 methodology leads, 48h SLA.
    - **Enterprise (custom)**: three reviewers across roles.
    - **Enterprise + regulatory factor (CBAM, EUDR, CSRD)**: compliance_lead
      required in addition to the enterprise trio.
    """
    tier_norm = (tier or "community").lower().strip()
    ftype = (factor_type or "").lower().strip()

    if tier_norm == "community":
        return ConsensusConfig(
            rule=ConsensusRule.ANY_OF_N,
            reviewer_requirements=(ReviewerRequirement(role="methodology_lead", min_count=1),),
            quorum=1,
            allow_self_approval=False,
            dissent_capture_required=True,
            sla_hours=72,
        )

    if tier_norm == "pro":
        return ConsensusConfig(
            rule=ConsensusRule.N_OF_M,
            reviewer_requirements=(
                ReviewerRequirement(role="methodology_lead", min_count=2),
            ),
            quorum=2,
            allow_self_approval=False,
            dissent_capture_required=True,
            sla_hours=48,
        )

    if tier_norm == "enterprise":
        base_reqs: List[ReviewerRequirement] = [
            ReviewerRequirement(role="methodology_lead", min_count=1, weight=2),
            ReviewerRequirement(role="qa_lead", min_count=1, weight=1),
            ReviewerRequirement(role="legal_lead", min_count=1, weight=1),
        ]
        quorum = 3
        sla_hours = 72
        if ftype in REGULATORY_FACTOR_TYPES:
            base_reqs.append(
                ReviewerRequirement(role="compliance_lead", min_count=1, weight=2)
            )
            quorum = 4
            sla_hours = 48  # Tighter SLA for regulatory flows.
        return ConsensusConfig(
            rule=ConsensusRule.WEIGHTED,
            reviewer_requirements=tuple(base_reqs),
            quorum=quorum,
            allow_self_approval=False,
            dissent_capture_required=True,
            sla_hours=sla_hours,
        )

    # Unknown tier -- conservative default matching community.
    logger.warning("Unknown tier %r, falling back to community defaults", tier)
    return tier_based_requirements("community", factor_type)


# ---------------------------------------------------------------------------
# Persistence helpers (SQLite in-process; Postgres schema lives in migration)
# ---------------------------------------------------------------------------


CONSENSUS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS factors_review_consensus_configs (
    config_id TEXT PRIMARY KEY,
    factor_type TEXT,
    tier TEXT,
    rule TEXT NOT NULL,
    reviewer_requirements_json TEXT NOT NULL,
    quorum INTEGER NOT NULL,
    allow_self_approval INTEGER NOT NULL DEFAULT 0,
    dissent_capture_required INTEGER NOT NULL DEFAULT 1,
    sla_hours INTEGER,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS factors_review_votes (
    vote_id TEXT PRIMARY KEY,
    factor_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('APPROVE','REJECT','ABSTAIN')),
    rationale TEXT,
    dissent_notes TEXT,
    weight INTEGER DEFAULT 1,
    voted_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_votes_factor ON factors_review_votes(factor_id);
CREATE INDEX IF NOT EXISTS idx_votes_reviewer ON factors_review_votes(reviewer_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_votes_factor_reviewer
    ON factors_review_votes(factor_id, reviewer_id);
"""


def ensure_consensus_schema(conn: sqlite3.Connection) -> None:
    """Create the consensus tables (idempotent)."""
    conn.executescript(CONSENSUS_SCHEMA_SQL)


def record_vote(conn: sqlite3.Connection, vote: ReviewerVote) -> None:
    """Persist (or upsert) a reviewer vote for a factor."""
    ensure_consensus_schema(conn)
    conn.execute(
        """
        INSERT INTO factors_review_votes (
            vote_id, factor_id, reviewer_id, reviewer_role,
            decision, rationale, dissent_notes, weight, voted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(factor_id, reviewer_id) DO UPDATE SET
            vote_id = excluded.vote_id,
            reviewer_role = excluded.reviewer_role,
            decision = excluded.decision,
            rationale = excluded.rationale,
            dissent_notes = excluded.dissent_notes,
            weight = excluded.weight,
            voted_at = excluded.voted_at
        """,
        (
            vote.vote_id,
            vote.factor_id,
            vote.reviewer_id,
            vote.reviewer_role,
            vote.decision.value,
            vote.rationale,
            vote.dissent_notes,
            int(vote.weight),
            vote.timestamp.isoformat(),
        ),
    )
    logger.debug(
        "Recorded vote %s for factor=%s reviewer=%s decision=%s",
        vote.vote_id, vote.factor_id, vote.reviewer_id, vote.decision.value,
    )


def build_vote(
    factor_id: str,
    reviewer_id: str,
    reviewer_role: str,
    decision: str,
    *,
    rationale: Optional[str] = None,
    dissent_notes: Optional[str] = None,
    weight: int = 1,
    timestamp: Optional[datetime] = None,
) -> ReviewerVote:
    """Convenience factory for a :class:`ReviewerVote` with a fresh UUID."""
    return ReviewerVote(
        vote_id=str(uuid.uuid4()),
        factor_id=factor_id,
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        decision=VoteDecision(decision.upper()),
        rationale=rationale,
        dissent_notes=dissent_notes,
        timestamp=timestamp or datetime.now(timezone.utc).replace(microsecond=0),
        weight=int(weight),
    )


def load_votes(conn: sqlite3.Connection, factor_id: str) -> List[ReviewerVote]:
    """Load all votes for a factor ordered by timestamp."""
    ensure_consensus_schema(conn)
    cur = conn.execute(
        """
        SELECT vote_id, factor_id, reviewer_id, reviewer_role, decision,
               rationale, dissent_notes, weight, voted_at
        FROM factors_review_votes
        WHERE factor_id = ?
        ORDER BY voted_at ASC
        """,
        (factor_id,),
    )
    votes: List[ReviewerVote] = []
    for row in cur.fetchall():
        votes.append(
            ReviewerVote(
                vote_id=row[0],
                factor_id=row[1],
                reviewer_id=row[2],
                reviewer_role=row[3],
                decision=VoteDecision(row[4]),
                rationale=row[5],
                dissent_notes=row[6],
                weight=int(row[7] or 1),
                timestamp=datetime.fromisoformat(row[8]),
            )
        )
    return votes


def get_pending_votes(
    conn: sqlite3.Connection,
    reviewer_id: str,
    *,
    factor_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return factor ids that are missing a vote from ``reviewer_id``.

    Used by the reviewer inbox / FactorsApprovalQueue.tsx surface to highlight
    "waiting on you" items.
    """
    ensure_consensus_schema(conn)
    if factor_ids is None:
        return []
    wanted = list(factor_ids)
    if not wanted:
        return []
    placeholders = ",".join(["?"] * len(wanted))
    cur = conn.execute(
        f"""
        SELECT factor_id FROM factors_review_votes
        WHERE reviewer_id = ? AND factor_id IN ({placeholders})
        """,
        (reviewer_id, *wanted),
    )
    voted = {row[0] for row in cur.fetchall()}
    return [fid for fid in wanted if fid not in voted]


def dissent_report(
    conn: sqlite3.Connection,
    factor_id: str,
) -> List[DissentNote]:
    """Return all dissent notes (non-APPROVE votes) for a factor."""
    ensure_consensus_schema(conn)
    cur = conn.execute(
        """
        SELECT factor_id, reviewer_id, reviewer_role, decision,
               COALESCE(dissent_notes, ''), voted_at
        FROM factors_review_votes
        WHERE factor_id = ? AND decision IN ('REJECT', 'ABSTAIN')
        ORDER BY voted_at ASC
        """,
        (factor_id,),
    )
    notes: List[DissentNote] = []
    for row in cur.fetchall():
        notes.append(
            DissentNote(
                factor_id=row[0],
                reviewer_id=row[1],
                reviewer_role=row[2],
                decision=VoteDecision(row[3]),
                notes=row[4],
                recorded_at=datetime.fromisoformat(row[5]),
            )
        )
    return notes


def save_config(
    conn: sqlite3.Connection,
    *,
    factor_type: str,
    tier: str,
    config: ConsensusConfig,
) -> str:
    """Persist a :class:`ConsensusConfig` for a (factor_type, tier) pair."""
    ensure_consensus_schema(conn)
    config_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO factors_review_consensus_configs (
            config_id, factor_type, tier, rule, reviewer_requirements_json,
            quorum, allow_self_approval, dissent_capture_required, sla_hours, active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (
            config_id,
            factor_type,
            tier,
            config.rule.value,
            json.dumps([r.to_dict() for r in config.reviewer_requirements]),
            int(config.quorum),
            1 if config.allow_self_approval else 0,
            1 if config.dissent_capture_required else 0,
            config.sla_hours,
        ),
    )
    return config_id


__all__ = [
    "InsufficientConsensusError",
    "DissentCaptureRequiredError",
    "ConsensusRule",
    "ConsensusStatus",
    "VoteDecision",
    "ReviewerRequirement",
    "ConsensusConfig",
    "ReviewerVote",
    "DissentNote",
    "ConsensusResult",
    "REVIEWER_ROLES",
    "REGULATORY_FACTOR_TYPES",
    "CONSENSUS_SCHEMA_SQL",
    "evaluate_consensus",
    "tier_based_requirements",
    "record_vote",
    "build_vote",
    "load_votes",
    "get_pending_votes",
    "dissent_report",
    "save_config",
    "ensure_consensus_schema",
]
