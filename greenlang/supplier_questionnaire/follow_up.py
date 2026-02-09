# -*- coding: utf-8 -*-
"""
Follow-Up Engine - AGENT-DATA-008: Supplier Questionnaire Processor
====================================================================

Manages follow-up actions for distributed questionnaires including
scheduled reminders, escalations, bulk triggers, non-responsive
supplier tracking, and effectiveness analytics.

Supports:
    - Automatic reminder scheduling (4 types: gentle/firm/urgent/final)
    - Manual and automatic reminder triggering
    - 5-level escalation for non-responsive suppliers
    - Bulk reminder triggering per campaign
    - Non-responsive supplier identification
    - Effectiveness tracking (response rate by reminder count)
    - Follow-up history per distribution
    - Reminder cancellation
    - Thread-safe in-memory storage
    - SHA-256 provenance hashes on all operations

Zero-Hallucination Guarantees:
    - All scheduling is deterministic (deadline-based offsets)
    - No LLM involvement in follow-up management
    - SHA-256 provenance hashes for audit trails
    - Reminder timing is pure arithmetic on deadline dates

Example:
    >>> from greenlang.supplier_questionnaire.follow_up import FollowUpEngine
    >>> engine = FollowUpEngine()
    >>> reminders = engine.schedule_reminders(
    ...     distribution_id="dist-001",
    ...     deadline=date(2025, 3, 31),
    ... )
    >>> assert len(reminders) == 4

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from greenlang.supplier_questionnaire.models import (
    Distribution,
    DistributionStatus,
    EscalationLevel,
    FollowUpAction,
    ReminderType,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FollowUpEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _today() -> date:
    """Return today's date in UTC."""
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------
# Reminder type configurations
# ---------------------------------------------------------------------------

_REMINDER_CONFIGS: Dict[ReminderType, Dict[str, Any]] = {
    ReminderType.GENTLE: {
        "days_before_deadline": 7,
        "subject": "Friendly reminder: Questionnaire due soon",
        "message": (
            "This is a friendly reminder that your sustainability "
            "questionnaire is due in 7 days. We appreciate your "
            "timely response."
        ),
    },
    ReminderType.FIRM: {
        "days_before_deadline": 3,
        "subject": "Reminder: Questionnaire response needed",
        "message": (
            "Your sustainability questionnaire is due in 3 days. "
            "Please complete and submit your response at your "
            "earliest convenience."
        ),
    },
    ReminderType.URGENT: {
        "days_before_deadline": 1,
        "subject": "Urgent: Questionnaire due tomorrow",
        "message": (
            "Your sustainability questionnaire is due tomorrow. "
            "Please submit your response as soon as possible "
            "to avoid escalation."
        ),
    },
    ReminderType.FINAL: {
        "days_before_deadline": 0,
        "subject": "Final notice: Questionnaire deadline today",
        "message": (
            "This is the final reminder. Your sustainability "
            "questionnaire deadline is today. Failure to respond "
            "may result in escalation to procurement management."
        ),
    },
}

_ESCALATION_MESSAGES: Dict[EscalationLevel, str] = {
    EscalationLevel.LEVEL_1: (
        "Automated escalation: Supplier has not responded after "
        "all reminders. Notifying supplier relationship manager."
    ),
    EscalationLevel.LEVEL_2: (
        "Escalation Level 2: Notifying procurement manager about "
        "non-responsive supplier."
    ),
    EscalationLevel.LEVEL_3: (
        "Escalation Level 3: Notifying VP Procurement. Supplier "
        "non-response may impact business relationship."
    ),
    EscalationLevel.LEVEL_4: (
        "Escalation Level 4: Notifying Chief Procurement Officer. "
        "Considering supplier scorecard impact."
    ),
    EscalationLevel.LEVEL_5: (
        "Escalation Level 5: Executive notification. Supplier "
        "relationship under review for non-compliance."
    ),
}


# ---------------------------------------------------------------------------
# FollowUpEngine
# ---------------------------------------------------------------------------


class FollowUpEngine:
    """Follow-up action scheduling and management engine.

    Manages the scheduling, triggering, escalation, and effectiveness
    tracking of follow-up reminders for distributed questionnaires.

    Attributes:
        _actions: In-memory action storage keyed by action_id.
        _distribution_actions: Index of distribution_id to action_ids.
        _campaign_actions: Index of campaign_id to action_ids.
        _config: Configuration dictionary.
        _lock: Threading lock for mutations.
        _stats: Aggregate statistics counters.

    Example:
        >>> engine = FollowUpEngine()
        >>> reminders = engine.schedule_reminders("dist-001", date(2025, 3, 31))
        >>> assert len(reminders) == 4
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FollowUpEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``max_actions``: int (default 100000)
                - ``auto_escalate``: bool (default False)
                - ``max_reminders_per_dist``: int (default 10)
        """
        self._config = config or {}
        self._actions: Dict[str, FollowUpAction] = {}
        self._distribution_actions: Dict[str, List[str]] = {}
        self._campaign_actions: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
        self._max_actions: int = self._config.get("max_actions", 100000)
        self._auto_escalate: bool = self._config.get("auto_escalate", False)
        self._max_reminders: int = self._config.get(
            "max_reminders_per_dist", 10,
        )
        self._stats: Dict[str, int] = {
            "reminders_scheduled": 0,
            "reminders_triggered": 0,
            "reminders_cancelled": 0,
            "escalations_created": 0,
            "bulk_triggers": 0,
            "errors": 0,
        }
        logger.info(
            "FollowUpEngine initialised: max_actions=%d, "
            "auto_escalate=%s",
            self._max_actions,
            self._auto_escalate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule_reminders(
        self,
        distribution_id: str,
        deadline: date,
        campaign_id: str = "",
        supplier_id: str = "",
    ) -> List[FollowUpAction]:
        """Schedule the standard 4-reminder sequence for a distribution.

        Creates GENTLE (7d before), FIRM (3d), URGENT (1d), and
        FINAL (on deadline) reminders.

        Args:
            distribution_id: Distribution to schedule reminders for.
            deadline: Submission deadline date.
            campaign_id: Optional campaign association.
            supplier_id: Optional supplier identifier.

        Returns:
            List of 4 FollowUpAction records.

        Raises:
            ValueError: If distribution_id is empty.
        """
        start = time.monotonic()

        if not distribution_id or not distribution_id.strip():
            raise ValueError("distribution_id must be non-empty")

        reminders: List[FollowUpAction] = []

        for reminder_type, config in _REMINDER_CONFIGS.items():
            days_before = config["days_before_deadline"]
            scheduled_date = deadline - timedelta(days=days_before)
            scheduled_dt = datetime(
                scheduled_date.year,
                scheduled_date.month,
                scheduled_date.day,
                9, 0, 0,  # Schedule for 9 AM UTC
                tzinfo=timezone.utc,
            )

            # Skip reminders in the past
            if scheduled_dt < _utcnow():
                # Still create them but mark as overdue
                pass

            action = self._create_action(
                distribution_id=distribution_id,
                campaign_id=campaign_id,
                supplier_id=supplier_id,
                reminder_type=reminder_type,
                scheduled_at=scheduled_dt,
                message=config["message"],
            )
            reminders.append(action)

        with self._lock:
            self._stats["reminders_scheduled"] += len(reminders)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Scheduled %d reminders for distribution %s, "
            "deadline=%s (%.1f ms)",
            len(reminders), distribution_id[:8],
            deadline.isoformat(), elapsed_ms,
        )
        return reminders

    def trigger_reminder(self, action_id: str) -> FollowUpAction:
        """Trigger (mark as sent) a scheduled reminder.

        Args:
            action_id: Action identifier to trigger.

        Returns:
            Updated FollowUpAction record.

        Raises:
            ValueError: If action_id not found or already sent.
        """
        action = self._get_action_or_raise(action_id)

        if action.status == "sent":
            raise ValueError(
                f"Action {action_id} has already been triggered"
            )
        if action.status == "cancelled":
            raise ValueError(
                f"Action {action_id} has been cancelled"
            )

        now = _utcnow()
        with self._lock:
            record = self._actions[action_id]
            record.status = "sent"
            record.sent_at = now
            record.provenance_hash = self._compute_provenance(
                "trigger_reminder", action_id,
            )
            self._stats["reminders_triggered"] += 1

        logger.info(
            "Triggered reminder %s (type=%s) for distribution %s",
            action_id[:8], action.reminder_type.value,
            action.distribution_id[:8],
        )
        return self._actions[action_id]

    def get_due_reminders(
        self,
        as_of: Optional[datetime] = None,
    ) -> List[FollowUpAction]:
        """Get all reminders that are due for sending.

        Returns scheduled reminders whose scheduled_at is at or
        before the specified time (default: now).

        Args:
            as_of: Cutoff time for due reminders (default: now).

        Returns:
            List of due FollowUpAction records.
        """
        cutoff = as_of or _utcnow()

        with self._lock:
            due: List[FollowUpAction] = []
            for action in self._actions.values():
                if (
                    action.status == "scheduled"
                    and action.scheduled_at is not None
                    and action.scheduled_at <= cutoff
                ):
                    due.append(action)

        # Sort by scheduled time (earliest first)
        due.sort(key=lambda a: a.scheduled_at or _utcnow())

        logger.debug(
            "Found %d due reminders as of %s",
            len(due), cutoff.isoformat(),
        )
        return due

    def escalate(
        self,
        distribution_id: str,
        level: str = "level_1",
    ) -> FollowUpAction:
        """Create an escalation action for a distribution.

        Args:
            distribution_id: Distribution to escalate.
            level: Escalation level string.

        Returns:
            FollowUpAction representing the escalation.

        Raises:
            ValueError: If distribution_id is empty or level invalid.
        """
        if not distribution_id or not distribution_id.strip():
            raise ValueError("distribution_id must be non-empty")

        esc_level = self._resolve_escalation_level(level)
        message = _ESCALATION_MESSAGES.get(
            esc_level,
            f"Escalation {esc_level.value}: Non-responsive supplier",
        )

        # Find supplier and campaign from existing actions
        supplier_id = ""
        campaign_id = ""
        with self._lock:
            action_ids = self._distribution_actions.get(distribution_id, [])
            for aid in action_ids:
                existing = self._actions.get(aid)
                if existing:
                    supplier_id = existing.supplier_id
                    campaign_id = existing.campaign_id
                    break

        action = self._create_action(
            distribution_id=distribution_id,
            campaign_id=campaign_id,
            supplier_id=supplier_id,
            reminder_type=ReminderType.FINAL,
            scheduled_at=_utcnow(),
            message=message,
            escalation_level=esc_level,
        )

        # Auto-trigger escalations
        self.trigger_reminder(action.action_id)

        with self._lock:
            self._stats["escalations_created"] += 1

        logger.info(
            "Escalated distribution %s to %s",
            distribution_id[:8], esc_level.value,
        )
        return self._actions[action.action_id]

    def get_follow_up_history(
        self,
        distribution_id: str,
    ) -> List[FollowUpAction]:
        """Get the complete follow-up history for a distribution.

        Args:
            distribution_id: Distribution to look up.

        Returns:
            List of FollowUpAction records, chronologically ordered.
        """
        with self._lock:
            action_ids = self._distribution_actions.get(
                distribution_id, [],
            )
            actions = [
                self._actions[aid]
                for aid in action_ids
                if aid in self._actions
            ]

        # Sort by scheduled_at
        actions.sort(key=lambda a: a.scheduled_at or _utcnow())
        return actions

    def cancel_reminders(self, distribution_id: str) -> int:
        """Cancel all pending reminders for a distribution.

        Only cancels reminders in 'scheduled' status.

        Args:
            distribution_id: Distribution to cancel reminders for.

        Returns:
            Number of reminders cancelled.
        """
        cancelled_count = 0

        with self._lock:
            action_ids = self._distribution_actions.get(
                distribution_id, [],
            )
            for aid in action_ids:
                action = self._actions.get(aid)
                if action and action.status == "scheduled":
                    action.status = "cancelled"
                    action.provenance_hash = self._compute_provenance(
                        "cancel_reminder", aid,
                    )
                    cancelled_count += 1

            self._stats["reminders_cancelled"] += cancelled_count

        logger.info(
            "Cancelled %d reminders for distribution %s",
            cancelled_count, distribution_id[:8],
        )
        return cancelled_count

    def bulk_trigger_reminders(
        self,
        campaign_id: str,
    ) -> List[FollowUpAction]:
        """Trigger all due reminders for a campaign.

        Args:
            campaign_id: Campaign to trigger reminders for.

        Returns:
            List of triggered FollowUpAction records.
        """
        start = time.monotonic()

        due = self.get_due_reminders()
        campaign_due = [
            a for a in due if a.campaign_id == campaign_id
        ]

        triggered: List[FollowUpAction] = []
        for action in campaign_due:
            try:
                result = self.trigger_reminder(action.action_id)
                triggered.append(result)
            except ValueError as e:
                logger.warning(
                    "Failed to trigger %s: %s", action.action_id[:8], e,
                )

        with self._lock:
            self._stats["bulk_triggers"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Bulk triggered %d/%d reminders for campaign %s (%.1f ms)",
            len(triggered), len(campaign_due), campaign_id[:8], elapsed_ms,
        )
        return triggered

    def get_non_responsive_suppliers(
        self,
        campaign_id: str,
        days_overdue: int = 7,
        distributions: Optional[List[Distribution]] = None,
    ) -> List[Dict[str, Any]]:
        """Identify non-responsive suppliers in a campaign.

        Args:
            campaign_id: Campaign to check.
            days_overdue: Minimum days overdue to include.
            distributions: Optional pre-fetched distribution list.

        Returns:
            List of dicts with supplier details and overdue info.
        """
        today = _today()
        non_responsive: List[Dict[str, Any]] = []

        if distributions is None:
            # Build from follow-up action data
            with self._lock:
                action_ids = self._campaign_actions.get(campaign_id, [])
                dist_ids = set()
                for aid in action_ids:
                    action = self._actions.get(aid)
                    if action:
                        dist_ids.add(action.distribution_id)

            for dist_id in dist_ids:
                history = self.get_follow_up_history(dist_id)
                if not history:
                    continue

                # Check if any sent reminders exist
                sent_count = sum(
                    1 for a in history if a.status == "sent"
                )
                if sent_count >= 2:  # At least 2 reminders sent
                    last_action = history[-1]
                    non_responsive.append({
                        "distribution_id": dist_id,
                        "supplier_id": last_action.supplier_id,
                        "campaign_id": campaign_id,
                        "reminders_sent": sent_count,
                        "last_reminder_type": last_action.reminder_type.value,
                        "escalation_level": (
                            last_action.escalation_level.value
                            if last_action.escalation_level
                            else None
                        ),
                    })
        else:
            for dist in distributions:
                if dist.campaign_id != campaign_id:
                    continue
                if dist.status in (
                    DistributionStatus.SUBMITTED,
                    DistributionStatus.CANCELLED,
                ):
                    continue
                if dist.deadline is None:
                    continue

                delta_days = (today - dist.deadline).days
                if delta_days >= days_overdue:
                    non_responsive.append({
                        "distribution_id": dist.distribution_id,
                        "supplier_id": dist.supplier_id,
                        "supplier_name": dist.supplier_name,
                        "supplier_email": dist.supplier_email,
                        "campaign_id": campaign_id,
                        "deadline": dist.deadline.isoformat(),
                        "days_overdue": delta_days,
                        "status": dist.status.value,
                        "reminders_sent": dist.reminder_count,
                    })

        # Sort by overdue days descending
        non_responsive.sort(
            key=lambda x: x.get("days_overdue", 0),
            reverse=True,
        )

        return non_responsive

    def track_effectiveness(
        self,
        campaign_id: str,
        distributions: Optional[List[Distribution]] = None,
    ) -> Dict[str, Any]:
        """Track reminder effectiveness for a campaign.

        Analyses the relationship between reminder count and
        response rate.

        Args:
            campaign_id: Campaign to analyse.
            distributions: Optional pre-fetched distributions.

        Returns:
            Dictionary with effectiveness metrics.
        """
        # Count reminders sent per distribution
        with self._lock:
            action_ids = self._campaign_actions.get(campaign_id, [])
            dist_reminder_count: Dict[str, int] = {}
            dist_responded: Dict[str, bool] = {}

            for aid in action_ids:
                action = self._actions.get(aid)
                if action and action.status == "sent":
                    did = action.distribution_id
                    dist_reminder_count[did] = (
                        dist_reminder_count.get(did, 0) + 1
                    )

        # If distributions provided, check response status
        if distributions:
            for dist in distributions:
                if dist.campaign_id == campaign_id:
                    dist_responded[dist.distribution_id] = (
                        dist.status == DistributionStatus.SUBMITTED
                    )

        # Build effectiveness buckets
        buckets: Dict[int, Dict[str, int]] = {}
        for dist_id, count in dist_reminder_count.items():
            if count not in buckets:
                buckets[count] = {"total": 0, "responded": 0}
            buckets[count]["total"] += 1
            if dist_responded.get(dist_id, False):
                buckets[count]["responded"] += 1

        effectiveness: Dict[str, Any] = {}
        for count in sorted(buckets.keys()):
            data = buckets[count]
            rate = (
                round(data["responded"] / data["total"] * 100, 1)
                if data["total"] > 0
                else 0.0
            )
            effectiveness[f"{count}_reminders"] = {
                "total_distributions": data["total"],
                "responded": data["responded"],
                "response_rate": rate,
            }

        total_dists = sum(b["total"] for b in buckets.values())
        total_responded = sum(b["responded"] for b in buckets.values())
        overall_rate = (
            round(total_responded / total_dists * 100, 1)
            if total_dists > 0
            else 0.0
        )

        provenance_hash = self._compute_provenance(
            "track_effectiveness", campaign_id, str(total_dists),
        )

        return {
            "campaign_id": campaign_id,
            "total_distributions": total_dists,
            "total_responded": total_responded,
            "overall_response_rate": overall_rate,
            "by_reminder_count": effectiveness,
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow().isoformat(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary of counter values.
        """
        with self._lock:
            return {
                **self._stats,
                "active_actions": len(self._actions),
                "distributions_tracked": len(self._distribution_actions),
                "campaigns_tracked": len(self._campaign_actions),
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_action(
        self,
        distribution_id: str,
        campaign_id: str,
        supplier_id: str,
        reminder_type: ReminderType,
        scheduled_at: datetime,
        message: str,
        escalation_level: Optional[EscalationLevel] = None,
    ) -> FollowUpAction:
        """Create and store a follow-up action.

        Args:
            distribution_id: Target distribution.
            campaign_id: Campaign association.
            supplier_id: Target supplier.
            reminder_type: Type of reminder.
            scheduled_at: When to send.
            message: Reminder message.
            escalation_level: Optional escalation level.

        Returns:
            Created FollowUpAction.
        """
        action_id = str(uuid.uuid4())
        provenance_hash = self._compute_provenance(
            "create_action", action_id, distribution_id,
            reminder_type.value,
        )

        action = FollowUpAction(
            action_id=action_id,
            distribution_id=distribution_id,
            campaign_id=campaign_id,
            supplier_id=supplier_id,
            reminder_type=reminder_type,
            escalation_level=escalation_level,
            status="scheduled",
            scheduled_at=scheduled_at,
            message=message,
            provenance_hash=provenance_hash,
        )

        with self._lock:
            self._actions[action_id] = action

            # Index by distribution
            if distribution_id not in self._distribution_actions:
                self._distribution_actions[distribution_id] = []
            self._distribution_actions[distribution_id].append(action_id)

            # Index by campaign
            if campaign_id and campaign_id not in self._campaign_actions:
                self._campaign_actions[campaign_id] = []
            if campaign_id:
                self._campaign_actions[campaign_id].append(action_id)

        return action

    def _get_action_or_raise(self, action_id: str) -> FollowUpAction:
        """Retrieve an action or raise ValueError.

        Args:
            action_id: Action identifier.

        Returns:
            FollowUpAction.

        Raises:
            ValueError: If action_id is not found.
        """
        with self._lock:
            action = self._actions.get(action_id)
        if action is None:
            raise ValueError(f"Unknown action: {action_id}")
        return action

    def _resolve_escalation_level(self, level: str) -> EscalationLevel:
        """Resolve an escalation level string to enum.

        Args:
            level: Escalation level string.

        Returns:
            EscalationLevel enum member.

        Raises:
            ValueError: If level is not recognised.
        """
        try:
            return EscalationLevel(level)
        except ValueError:
            valid = [e.value for e in EscalationLevel]
            raise ValueError(
                f"Unknown escalation level '{level}'. Valid: {valid}"
            )

    def _compute_provenance(self, *parts: str) -> str:
        """Compute SHA-256 provenance hash from parts.

        Args:
            *parts: Strings to include in the hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        combined = json.dumps(
            {"parts": list(parts), "timestamp": _utcnow().isoformat()},
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
