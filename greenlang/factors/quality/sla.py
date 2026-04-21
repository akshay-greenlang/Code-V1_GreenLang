# -*- coding: utf-8 -*-
"""
Approval SLA enforcement (GAP-15).

Implements SLA timers around the methodology review state machine
(``DRAFT -> UNDER_REVIEW -> APPROVED -> PUBLISHED``) and feeds escalation
decisions to :mod:`greenlang.factors.quality.escalation`.

A timer is started for each lifecycle stage with a tier-specific duration,
a 75%-warning point, and optional hard-deadline auto-reject.  Timers are
persisted to SQLite in-process; the Postgres schema for production lives in
``V446__factors_review_sla.sql``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SLAExpiredError(RuntimeError):
    """Raised when a workflow transition is blocked by an expired SLA timer."""

    def __init__(self, factor_id: str, stage: str, expired_at: datetime) -> None:
        super().__init__(
            "SLA expired for factor %s at stage %s (deadline=%s)"
            % (factor_id, stage, expired_at.isoformat())
        )
        self.factor_id = factor_id
        self.stage = stage
        self.expired_at = expired_at


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SLAStage(str, Enum):
    """Review lifecycle stages that carry an SLA timer."""

    INITIAL_REVIEW = "initial_review"          # DRAFT -> UNDER_REVIEW
    DETAILED_REVIEW = "detailed_review"        # UNDER_REVIEW -> APPROVED
    FINAL_APPROVAL = "final_approval"          # APPROVED -> PUBLISHED
    DEPRECATION_NOTICE = "deprecation_notice"  # PUBLISHED -> DEPRECATED


class SLATimerStatus(str, Enum):
    """Runtime status of an SLA timer."""

    ACTIVE = "ACTIVE"
    WARNED = "WARNED"
    ESCALATED = "ESCALATED"
    EXPIRED = "EXPIRED"
    COMPLETED = "COMPLETED"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SLAPolicy:
    """Tier/stage-specific SLA policy."""

    stage: SLAStage
    duration_hours: int
    tier: str
    warning_at_pct: float = 0.75
    escalation_level: int = 1
    auto_reject_after_hours: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "tier": self.tier,
            "duration_hours": self.duration_hours,
            "warning_at_pct": self.warning_at_pct,
            "escalation_level": self.escalation_level,
            "auto_reject_after_hours": self.auto_reject_after_hours,
        }


@dataclass
class SLATimer:
    """Active SLA timer for a factor at a specific stage."""

    timer_id: str
    factor_id: str
    stage: SLAStage
    started_at: datetime
    deadline: datetime
    warning_at: datetime
    status: SLATimerStatus = SLATimerStatus.ACTIVE
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    completed_at: Optional[datetime] = None
    auto_reject_after_hours: Optional[int] = None
    tier: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timer_id": self.timer_id,
            "factor_id": self.factor_id,
            "stage": self.stage.value,
            "started_at": self.started_at.isoformat(),
            "deadline": self.deadline.isoformat(),
            "warning_at": self.warning_at.isoformat(),
            "status": self.status.value,
            "escalation_history": list(self.escalation_history),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "auto_reject_after_hours": self.auto_reject_after_hours,
            "tier": self.tier,
        }

    def is_overdue(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.deadline

    def is_warning(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.warning_at and now < self.deadline

    def should_auto_reject(self, now: Optional[datetime] = None) -> bool:
        if self.auto_reject_after_hours is None:
            return False
        now = now or datetime.now(timezone.utc)
        cutoff = self.started_at + timedelta(hours=int(self.auto_reject_after_hours))
        return now >= cutoff


# ---------------------------------------------------------------------------
# Default policies
# ---------------------------------------------------------------------------


def _policy_key(stage: SLAStage, tier: str) -> Tuple[str, str]:
    return (stage.value, tier.lower())


# Defaults per spec: community (72h/168h/168h), pro (48h/120h/120h),
# enterprise (24h/72h/72h), enterprise-cbam (48h/96h/48h).
DEFAULT_SLA_POLICIES: Dict[Tuple[str, str], SLAPolicy] = {
    _policy_key(SLAStage.INITIAL_REVIEW, "community"): SLAPolicy(
        stage=SLAStage.INITIAL_REVIEW, tier="community", duration_hours=72,
        escalation_level=1,
    ),
    _policy_key(SLAStage.DETAILED_REVIEW, "community"): SLAPolicy(
        stage=SLAStage.DETAILED_REVIEW, tier="community", duration_hours=168,
        escalation_level=1,
    ),
    _policy_key(SLAStage.FINAL_APPROVAL, "community"): SLAPolicy(
        stage=SLAStage.FINAL_APPROVAL, tier="community", duration_hours=168,
        escalation_level=2,
    ),
    _policy_key(SLAStage.DEPRECATION_NOTICE, "community"): SLAPolicy(
        stage=SLAStage.DEPRECATION_NOTICE, tier="community", duration_hours=720,
        escalation_level=1,
    ),

    _policy_key(SLAStage.INITIAL_REVIEW, "pro"): SLAPolicy(
        stage=SLAStage.INITIAL_REVIEW, tier="pro", duration_hours=48,
        escalation_level=1,
    ),
    _policy_key(SLAStage.DETAILED_REVIEW, "pro"): SLAPolicy(
        stage=SLAStage.DETAILED_REVIEW, tier="pro", duration_hours=120,
        escalation_level=2,
    ),
    _policy_key(SLAStage.FINAL_APPROVAL, "pro"): SLAPolicy(
        stage=SLAStage.FINAL_APPROVAL, tier="pro", duration_hours=120,
        escalation_level=2,
    ),
    _policy_key(SLAStage.DEPRECATION_NOTICE, "pro"): SLAPolicy(
        stage=SLAStage.DEPRECATION_NOTICE, tier="pro", duration_hours=360,
        escalation_level=2,
    ),

    _policy_key(SLAStage.INITIAL_REVIEW, "enterprise"): SLAPolicy(
        stage=SLAStage.INITIAL_REVIEW, tier="enterprise", duration_hours=24,
        escalation_level=2,
    ),
    _policy_key(SLAStage.DETAILED_REVIEW, "enterprise"): SLAPolicy(
        stage=SLAStage.DETAILED_REVIEW, tier="enterprise", duration_hours=72,
        escalation_level=2,
    ),
    _policy_key(SLAStage.FINAL_APPROVAL, "enterprise"): SLAPolicy(
        stage=SLAStage.FINAL_APPROVAL, tier="enterprise", duration_hours=72,
        escalation_level=3,
    ),
    _policy_key(SLAStage.DEPRECATION_NOTICE, "enterprise"): SLAPolicy(
        stage=SLAStage.DEPRECATION_NOTICE, tier="enterprise", duration_hours=240,
        escalation_level=2,
    ),

    # Enterprise with CBAM / regulatory flag has dedicated tier key.
    _policy_key(SLAStage.INITIAL_REVIEW, "enterprise_cbam"): SLAPolicy(
        stage=SLAStage.INITIAL_REVIEW, tier="enterprise_cbam", duration_hours=48,
        escalation_level=2,
        auto_reject_after_hours=240,
    ),
    _policy_key(SLAStage.DETAILED_REVIEW, "enterprise_cbam"): SLAPolicy(
        stage=SLAStage.DETAILED_REVIEW, tier="enterprise_cbam", duration_hours=96,
        escalation_level=3,
        auto_reject_after_hours=336,
    ),
    _policy_key(SLAStage.FINAL_APPROVAL, "enterprise_cbam"): SLAPolicy(
        stage=SLAStage.FINAL_APPROVAL, tier="enterprise_cbam", duration_hours=48,
        escalation_level=3,
        auto_reject_after_hours=168,
    ),
    _policy_key(SLAStage.DEPRECATION_NOTICE, "enterprise_cbam"): SLAPolicy(
        stage=SLAStage.DEPRECATION_NOTICE, tier="enterprise_cbam", duration_hours=168,
        escalation_level=3,
    ),
}


def get_policy(stage: SLAStage, tier: str) -> SLAPolicy:
    """Look up the default :class:`SLAPolicy` for a (stage, tier) pair.

    Falls back to the ``community`` policy for unknown tiers and logs a warning.
    """
    key = _policy_key(stage, tier)
    policy = DEFAULT_SLA_POLICIES.get(key)
    if policy is None:
        logger.warning(
            "No SLA policy found for stage=%s tier=%s; falling back to community",
            stage.value, tier,
        )
        return DEFAULT_SLA_POLICIES[_policy_key(stage, "community")]
    return policy


# ---------------------------------------------------------------------------
# Persistence schema
# ---------------------------------------------------------------------------


SLA_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS factors_sla_policies (
    policy_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    tier TEXT NOT NULL,
    duration_hours INTEGER NOT NULL,
    warning_at_pct REAL DEFAULT 0.75,
    escalation_level INTEGER DEFAULT 1,
    auto_reject_after_hours INTEGER,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE(stage, tier)
);
CREATE TABLE IF NOT EXISTS factors_sla_timers (
    timer_id TEXT PRIMARY KEY,
    factor_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    tier TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    deadline TEXT NOT NULL,
    warning_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN
        ('ACTIVE','WARNED','ESCALATED','EXPIRED','COMPLETED')),
    escalation_history_json TEXT DEFAULT '[]',
    auto_reject_after_hours INTEGER,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sla_timer_factor
    ON factors_sla_timers(factor_id);
CREATE INDEX IF NOT EXISTS idx_sla_timer_deadline_status
    ON factors_sla_timers(deadline, status);
"""


def ensure_sla_schema(conn: sqlite3.Connection) -> None:
    """Create SLA tables (idempotent)."""
    conn.executescript(SLA_SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Timer management
# ---------------------------------------------------------------------------


def _compute_deadlines(
    started_at: datetime,
    policy: SLAPolicy,
) -> Tuple[datetime, datetime]:
    """Return (deadline, warning_at) for the given start time and policy."""
    duration = timedelta(hours=int(policy.duration_hours))
    deadline = started_at + duration
    warn_secs = max(0.0, min(1.0, float(policy.warning_at_pct))) * duration.total_seconds()
    warning_at = started_at + timedelta(seconds=warn_secs)
    return deadline, warning_at


def start_sla_timer(
    conn: sqlite3.Connection,
    factor_id: str,
    stage: SLAStage,
    tier: str,
    *,
    now: Optional[datetime] = None,
    policy: Optional[SLAPolicy] = None,
) -> SLATimer:
    """Create and persist a new SLA timer for a factor/stage."""
    ensure_sla_schema(conn)
    now = now or datetime.now(timezone.utc)
    policy = policy or get_policy(stage, tier)
    deadline, warning_at = _compute_deadlines(now, policy)
    timer = SLATimer(
        timer_id=str(uuid.uuid4()),
        factor_id=factor_id,
        stage=stage,
        started_at=now,
        deadline=deadline,
        warning_at=warning_at,
        status=SLATimerStatus.ACTIVE,
        auto_reject_after_hours=policy.auto_reject_after_hours,
        tier=policy.tier,
    )
    _insert_timer(conn, timer)
    logger.info(
        "Started SLA timer %s for factor=%s stage=%s tier=%s deadline=%s",
        timer.timer_id, factor_id, stage.value, policy.tier, deadline.isoformat(),
    )
    return timer


def _insert_timer(conn: sqlite3.Connection, timer: SLATimer) -> None:
    conn.execute(
        """
        INSERT INTO factors_sla_timers (
            timer_id, factor_id, stage, tier, started_at, deadline, warning_at,
            status, escalation_history_json, auto_reject_after_hours, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timer.timer_id,
            timer.factor_id,
            timer.stage.value,
            timer.tier,
            timer.started_at.isoformat(),
            timer.deadline.isoformat(),
            timer.warning_at.isoformat(),
            timer.status.value,
            json.dumps(timer.escalation_history),
            timer.auto_reject_after_hours,
            timer.completed_at.isoformat() if timer.completed_at else None,
        ),
    )


def _row_to_timer(row: Mapping[str, Any]) -> SLATimer:
    history_raw = row["escalation_history_json"] if isinstance(row, dict) else row[8]
    try:
        history = json.loads(history_raw) if history_raw else []
    except (TypeError, ValueError):
        history = []
    completed_at_raw = row["completed_at"] if isinstance(row, dict) else row[10]
    return SLATimer(
        timer_id=row[0] if not isinstance(row, dict) else row["timer_id"],
        factor_id=row[1] if not isinstance(row, dict) else row["factor_id"],
        stage=SLAStage(row[2] if not isinstance(row, dict) else row["stage"]),
        tier=row[3] if not isinstance(row, dict) else (row["tier"] or ""),
        started_at=datetime.fromisoformat(
            row[4] if not isinstance(row, dict) else row["started_at"]
        ),
        deadline=datetime.fromisoformat(
            row[5] if not isinstance(row, dict) else row["deadline"]
        ),
        warning_at=datetime.fromisoformat(
            row[6] if not isinstance(row, dict) else row["warning_at"]
        ),
        status=SLATimerStatus(
            row[7] if not isinstance(row, dict) else row["status"]
        ),
        escalation_history=history,
        auto_reject_after_hours=(
            row[9] if not isinstance(row, dict) else row["auto_reject_after_hours"]
        ),
        completed_at=(
            datetime.fromisoformat(completed_at_raw) if completed_at_raw else None
        ),
    )


def get_timer_for_factor(
    conn: sqlite3.Connection,
    factor_id: str,
    stage: SLAStage,
    *,
    include_completed: bool = False,
) -> Optional[SLATimer]:
    """Return the most recent timer for ``(factor_id, stage)``."""
    ensure_sla_schema(conn)
    status_filter = "" if include_completed else "AND status != 'COMPLETED'"
    cur = conn.execute(
        f"""
        SELECT timer_id, factor_id, stage, tier, started_at, deadline, warning_at,
               status, escalation_history_json, auto_reject_after_hours, completed_at
        FROM factors_sla_timers
        WHERE factor_id = ? AND stage = ? {status_filter}
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (factor_id, stage.value),
    )
    row = cur.fetchone()
    return _row_to_timer(row) if row else None


def _update_timer_status(
    conn: sqlite3.Connection,
    timer: SLATimer,
    new_status: SLATimerStatus,
    *,
    event: Optional[Dict[str, Any]] = None,
    completed_at: Optional[datetime] = None,
) -> SLATimer:
    timer.status = new_status
    if event is not None:
        timer.escalation_history.append(event)
    if completed_at is not None:
        timer.completed_at = completed_at
    conn.execute(
        """
        UPDATE factors_sla_timers
        SET status = ?, escalation_history_json = ?, completed_at = ?
        WHERE timer_id = ?
        """,
        (
            timer.status.value,
            json.dumps(timer.escalation_history),
            timer.completed_at.isoformat() if timer.completed_at else None,
            timer.timer_id,
        ),
    )
    return timer


def warn_overdue(
    conn: sqlite3.Connection,
    timer: SLATimer,
    *,
    now: Optional[datetime] = None,
) -> SLATimer:
    """Transition an ACTIVE timer to WARNED (first escalation stage)."""
    now = now or datetime.now(timezone.utc)
    if timer.status != SLATimerStatus.ACTIVE:
        logger.debug("warn_overdue skipped: timer %s status=%s", timer.timer_id, timer.status.value)
        return timer
    event = {
        "type": "warning",
        "level": 1,
        "at": now.isoformat(),
    }
    logger.info(
        "Warning SLA timer=%s factor=%s stage=%s",
        timer.timer_id, timer.factor_id, timer.stage.value,
    )
    return _update_timer_status(conn, timer, SLATimerStatus.WARNED, event=event)


def escalate_overdue(
    conn: sqlite3.Connection,
    timer: SLATimer,
    *,
    now: Optional[datetime] = None,
    max_level: int = 4,
) -> SLATimer:
    """Bump a timer to ESCALATED and append an entry to its history."""
    now = now or datetime.now(timezone.utc)
    if timer.status == SLATimerStatus.COMPLETED:
        return timer
    current_level = sum(
        1 for e in timer.escalation_history if e.get("type") == "escalation"
    )
    next_level = min(current_level + 1, max_level)
    event = {
        "type": "escalation",
        "level": next_level,
        "at": now.isoformat(),
    }
    logger.info(
        "Escalating SLA timer=%s factor=%s stage=%s to level %d",
        timer.timer_id, timer.factor_id, timer.stage.value, next_level,
    )
    return _update_timer_status(conn, timer, SLATimerStatus.ESCALATED, event=event)


def auto_reject_stale(
    conn: sqlite3.Connection,
    timer: SLATimer,
    *,
    now: Optional[datetime] = None,
) -> SLATimer:
    """Mark a timer as EXPIRED because the auto-reject cutoff has passed."""
    now = now or datetime.now(timezone.utc)
    event = {
        "type": "auto_reject",
        "at": now.isoformat(),
        "auto_reject_after_hours": timer.auto_reject_after_hours,
    }
    logger.warning(
        "Auto-rejecting stale SLA timer=%s factor=%s stage=%s",
        timer.timer_id, timer.factor_id, timer.stage.value,
    )
    return _update_timer_status(
        conn, timer, SLATimerStatus.EXPIRED,
        event=event, completed_at=now,
    )


def complete_timer(
    conn: sqlite3.Connection,
    factor_id: str,
    stage: SLAStage,
    *,
    now: Optional[datetime] = None,
) -> Optional[SLATimer]:
    """Mark the newest active timer for ``(factor_id, stage)`` as COMPLETED."""
    now = now or datetime.now(timezone.utc)
    timer = get_timer_for_factor(conn, factor_id, stage)
    if timer is None or timer.status == SLATimerStatus.COMPLETED:
        return None
    event = {"type": "completed", "at": now.isoformat()}
    return _update_timer_status(
        conn, timer, SLATimerStatus.COMPLETED,
        event=event, completed_at=now,
    )


def check_timers(
    conn: sqlite3.Connection,
    *,
    now: Optional[datetime] = None,
) -> List[SLATimer]:
    """Scan active timers and return those needing action.

    The function is pure-inspection: callers (cron/scheduler) decide what to
    do with the result.  Returned timers are annotated with an in-memory
    ``_action`` attribute (one of ``warn``, ``escalate``, ``auto_reject``).
    """
    ensure_sla_schema(conn)
    now = now or datetime.now(timezone.utc)
    cur = conn.execute(
        """
        SELECT timer_id, factor_id, stage, tier, started_at, deadline, warning_at,
               status, escalation_history_json, auto_reject_after_hours, completed_at
        FROM factors_sla_timers
        WHERE status != 'COMPLETED'
        """,
    )
    needing: List[SLATimer] = []
    for row in cur.fetchall():
        timer = _row_to_timer(row)
        action = _next_action(timer, now=now)
        if action is None:
            continue
        setattr(timer, "_action", action)
        needing.append(timer)
    return needing


def _next_action(timer: SLATimer, *, now: datetime) -> Optional[str]:
    """Determine the next required action for a timer at ``now``."""
    if timer.status == SLATimerStatus.COMPLETED:
        return None
    if timer.should_auto_reject(now) and timer.status != SLATimerStatus.EXPIRED:
        return "auto_reject"
    if timer.is_overdue(now):
        # Overdue timers must at least be escalated; WARNED moves to ESCALATED,
        # ESCALATED stays in the queue for level bumps until auto-reject.
        if timer.status in (SLATimerStatus.ACTIVE, SLATimerStatus.WARNED):
            return "escalate"
        # Timers already in ESCALATED state continue bumping levels each tick.
        return "escalate"
    if timer.is_warning(now) and timer.status == SLATimerStatus.ACTIVE:
        return "warn"
    return None


# ---------------------------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------------------------


def sla_dashboard_metrics(
    conn: sqlite3.Connection,
    tenant_id: Optional[str] = None,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Summarise timer state for a tenant's approval queue.

    ``tenant_id`` is accepted for forward-compatibility; the in-memory SQLite
    schema does not segment by tenant, but the Postgres migration's dashboard
    views do.  Unknown tenants receive aggregate metrics.
    """
    ensure_sla_schema(conn)
    now = now or datetime.now(timezone.utc)
    cur = conn.execute(
        """
        SELECT status, COUNT(*) FROM factors_sla_timers GROUP BY status
        """,
    )
    status_counts: Dict[str, int] = {s.value: 0 for s in SLATimerStatus}
    total = 0
    for row in cur.fetchall():
        status_counts[row[0]] = int(row[1])
        total += int(row[1])

    overdue = 0
    warning = 0
    cur = conn.execute(
        """
        SELECT warning_at, deadline, status FROM factors_sla_timers
        WHERE status != 'COMPLETED'
        """,
    )
    for row in cur.fetchall():
        warn_at = datetime.fromisoformat(row[0])
        deadline = datetime.fromisoformat(row[1])
        if now >= deadline:
            overdue += 1
        elif now >= warn_at:
            warning += 1

    active = status_counts.get(SLATimerStatus.ACTIVE.value, 0) + status_counts.get(
        SLATimerStatus.WARNED.value, 0
    )
    completed = status_counts.get(SLATimerStatus.COMPLETED.value, 0)
    total_decided = completed + status_counts.get(SLATimerStatus.EXPIRED.value, 0)
    compliance_pct = (
        (completed / total_decided) * 100.0 if total_decided else 100.0
    )
    return {
        "tenant_id": tenant_id,
        "as_of": now.isoformat(),
        "total_timers": total,
        "status_counts": status_counts,
        "active_timers": active,
        "warning_timers": warning,
        "overdue_timers": overdue,
        "compliance_pct": round(compliance_pct, 2),
    }


__all__ = [
    "SLAExpiredError",
    "SLAStage",
    "SLATimerStatus",
    "SLAPolicy",
    "SLATimer",
    "DEFAULT_SLA_POLICIES",
    "SLA_SCHEMA_SQL",
    "ensure_sla_schema",
    "get_policy",
    "start_sla_timer",
    "get_timer_for_factor",
    "warn_overdue",
    "escalate_overdue",
    "auto_reject_stale",
    "complete_timer",
    "check_timers",
    "sla_dashboard_metrics",
]
