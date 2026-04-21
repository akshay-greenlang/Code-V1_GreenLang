# -*- coding: utf-8 -*-
"""
Escalation dispatch for overdue factor approvals (GAP-15).

Routes overdue and stale SLA timers to the existing Slack / email / webhook
plumbing in :mod:`greenlang.factors.notifications` and supports a daily
digest roll-up for ops teams.

Four escalation levels are recognised:

1. primary reviewer
2. team lead
3. methodology lead
4. methodology lead's manager

Escalation targets are supplied via :class:`EscalationTargets` rather than
hard-coded, so the module can be used both in production (with real
addresses from a directory service) and in tests (with in-memory stubs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from greenlang.factors.quality.sla import SLAStage, SLATimer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EscalationTargets:
    """Routing targets for the four escalation levels.

    Each field holds ordered addresses/channels for the level.  Email and
    Slack entries are both optional per level.
    """

    level1_emails: List[str] = field(default_factory=list)
    level1_slack: Optional[str] = None
    level2_emails: List[str] = field(default_factory=list)
    level2_slack: Optional[str] = None
    level3_emails: List[str] = field(default_factory=list)
    level3_slack: Optional[str] = None
    level4_emails: List[str] = field(default_factory=list)
    level4_slack: Optional[str] = None

    def for_level(self, level: int) -> Dict[str, Any]:
        level = max(1, min(4, int(level)))
        return {
            "emails": list(getattr(self, f"level{level}_emails")),
            "slack_channel": getattr(self, f"level{level}_slack"),
        }


@dataclass
class EscalationEvent:
    """Single escalation notification attempt (for audit + digest)."""

    timer_id: str
    factor_id: str
    stage: SLAStage
    level: int
    reason: str
    channels: List[str]
    recipients: List[str]
    dispatched_at: datetime
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timer_id": self.timer_id,
            "factor_id": self.factor_id,
            "stage": self.stage.value,
            "level": self.level,
            "reason": self.reason,
            "channels": list(self.channels),
            "recipients": list(self.recipients),
            "dispatched_at": self.dispatched_at.isoformat(),
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


# Stage-level templates keyed by (stage, reason).  Reason values are
# ``warning`` (75% mark), ``overdue`` (deadline passed), ``auto_reject``
# (hard cutoff), ``digest`` (daily roll-up).
TEMPLATES: Dict[str, Dict[str, str]] = {
    "warning": {
        "subject": "[Factors SLA] Warning: {factor_id} stage={stage} 75% elapsed",
        "body": (
            "Heads up: the SLA timer for factor {factor_id} at stage {stage} "
            "has reached 75% of its duration.\n\n"
            "Deadline: {deadline}\nStarted: {started_at}\nLevel: {level}\n\n"
            "Please take action before the timer expires."
        ),
    },
    "overdue": {
        "subject": "[Factors SLA] OVERDUE: {factor_id} stage={stage}",
        "body": (
            "The SLA timer for factor {factor_id} at stage {stage} has expired.\n\n"
            "Deadline was: {deadline}\nStarted: {started_at}\nLevel: {level}\n\n"
            "Escalating to level {level}. Immediate action required."
        ),
    },
    "auto_reject": {
        "subject": "[Factors SLA] Auto-rejected: {factor_id} stage={stage}",
        "body": (
            "Factor {factor_id} has exceeded the hard deadline for stage {stage} "
            "and has been auto-rejected by policy.\n\n"
            "Please triage and reopen if appropriate."
        ),
    },
    "digest": {
        "subject": "[Factors SLA] Daily digest: {overdue_count} overdue, {warning_count} warnings",
        "body": (
            "Daily SLA digest ({as_of}):\n\n"
            "- Overdue timers: {overdue_count}\n"
            "- Timers approaching deadline: {warning_count}\n"
            "- Escalations dispatched (last 24h): {escalation_count}\n\n"
            "Details:\n{details}"
        ),
    },
}


def render_template(
    reason: str,
    context: Mapping[str, Any],
) -> Dict[str, str]:
    """Render (subject, body) for a given reason key."""
    template = TEMPLATES.get(reason, TEMPLATES["overdue"])
    try:
        subject = template["subject"].format_map(_SafeDict(context))
        body = template["body"].format_map(_SafeDict(context))
    except (KeyError, ValueError) as exc:
        logger.warning("Template render failure for reason=%s: %s", reason, exc)
        subject = f"[Factors SLA] {reason}"
        body = str(dict(context))
    return {"subject": subject, "body": body}


class _SafeDict(dict):
    """Dict that returns ``{key}`` for missing keys so templates never crash."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


# Callable signature matches :func:`greenlang.factors.notifications.webhook_notifier.send_slack_notification`.
SlackSender = Callable[..., bool]
EmailSender = Callable[..., bool]
WebhookSender = Callable[..., bool]


@dataclass
class EscalationDispatcher:
    """Routes escalation events to Slack, email, and optional webhooks.

    The dispatcher is dependency-injected so tests can swap in recording stubs
    without touching the real network.  In production, ``slack_sender`` and
    ``email_sender`` are bound to the functions in
    :mod:`greenlang.factors.notifications.webhook_notifier`.
    """

    targets: EscalationTargets
    slack_sender: Optional[SlackSender] = None
    email_sender: Optional[EmailSender] = None
    webhook_sender: Optional[WebhookSender] = None
    slack_webhook_url: Optional[str] = None
    smtp_config: Optional[Mapping[str, Any]] = None
    from_address: str = "greenlang-factors@greenlang.io"
    audit: List[EscalationEvent] = field(default_factory=list)

    # -- dispatch ----------------------------------------------------------

    def dispatch(
        self,
        timer: SLATimer,
        *,
        reason: str,
        level: int,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> EscalationEvent:
        """Dispatch a single escalation event.

        Args:
            timer: The SLA timer being escalated.
            reason: Template key (``warning`` / ``overdue`` / ``auto_reject``).
            level: Escalation level (1..4).
            extra_context: Additional template variables.

        Returns:
            The recorded :class:`EscalationEvent`.
        """
        ctx: Dict[str, Any] = {
            "factor_id": timer.factor_id,
            "stage": timer.stage.value,
            "started_at": timer.started_at.isoformat(),
            "deadline": timer.deadline.isoformat(),
            "level": level,
            "tier": timer.tier,
        }
        if extra_context:
            ctx.update(extra_context)

        rendered = render_template(reason, ctx)
        subject = rendered["subject"]
        body = rendered["body"]

        target = self.targets.for_level(level)
        recipients: List[str] = list(target["emails"])
        slack_channel: Optional[str] = target["slack_channel"]

        channels: List[str] = []
        success = True
        error: Optional[str] = None

        if recipients and self.email_sender is not None:
            try:
                ok = self._send_email(recipients, subject, body)
                channels.append("email")
                success = success and ok
                if not ok:
                    error = "email_sender returned False"
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Email escalation failed: %s", exc, exc_info=True)
                success = False
                error = str(exc)

        if slack_channel and self.slack_sender is not None and self.slack_webhook_url:
            try:
                slack_text = f"*{subject}*\n{body}"
                ok = self.slack_sender(
                    self.slack_webhook_url, slack_text, channel=slack_channel
                )
                channels.append("slack")
                success = success and bool(ok)
                if not ok:
                    error = error or "slack_sender returned False"
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Slack escalation failed: %s", exc, exc_info=True)
                success = False
                error = str(exc)

        if self.webhook_sender is not None:
            try:
                ok = self.webhook_sender(
                    {"subject": subject, "body": body, "context": ctx}
                )
                channels.append("webhook")
                success = success and bool(ok)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Webhook escalation failed: %s", exc, exc_info=True)
                success = False
                error = str(exc)

        event = EscalationEvent(
            timer_id=timer.timer_id,
            factor_id=timer.factor_id,
            stage=timer.stage,
            level=level,
            reason=reason,
            channels=channels,
            recipients=recipients,
            dispatched_at=datetime.now(timezone.utc),
            success=success,
            error=error,
        )
        self.audit.append(event)
        logger.info(
            "Escalation dispatched factor=%s stage=%s level=%d reason=%s success=%s",
            timer.factor_id, timer.stage.value, level, reason, success,
        )
        return event

    def _send_email(
        self,
        recipients: Sequence[str],
        subject: str,
        body: str,
    ) -> bool:
        """Call the injected email sender with the SMTP config."""
        if self.email_sender is None:
            return False
        smtp = dict(self.smtp_config or {})
        return bool(
            self.email_sender(
                smtp.get("smtp_host"),
                int(smtp.get("smtp_port", 587) or 587),
                smtp.get("smtp_user"),
                smtp.get("smtp_password"),
                self.from_address,
                list(recipients),
                subject,
                body,
            )
        )

    # -- digest ------------------------------------------------------------

    def daily_digest(
        self,
        overdue: Sequence[SLATimer],
        warnings: Sequence[SLATimer],
        *,
        digest_level: int = 3,
        as_of: Optional[datetime] = None,
    ) -> EscalationEvent:
        """Emit the daily digest to the ops channel/email list."""
        as_of = as_of or datetime.now(timezone.utc)
        details_lines: List[str] = []
        for t in list(overdue)[:25]:
            details_lines.append(
                f"  * OVERDUE factor={t.factor_id} stage={t.stage.value} "
                f"deadline={t.deadline.isoformat()}"
            )
        for t in list(warnings)[:25]:
            details_lines.append(
                f"  * warning factor={t.factor_id} stage={t.stage.value} "
                f"warn_at={t.warning_at.isoformat()}"
            )
        ctx = {
            "as_of": as_of.isoformat(),
            "overdue_count": len(overdue),
            "warning_count": len(warnings),
            "escalation_count": len(self.audit),
            "details": "\n".join(details_lines) or "  (none)",
        }
        rendered = render_template("digest", ctx)

        # Synthetic timer just so dispatch() can reuse the same plumbing.
        placeholder = SLATimer(
            timer_id="digest",
            factor_id="digest",
            stage=SLAStage.INITIAL_REVIEW,
            started_at=as_of,
            deadline=as_of,
            warning_at=as_of,
        )
        target = self.targets.for_level(digest_level)
        channels: List[str] = []
        success = True
        error: Optional[str] = None

        if target["emails"] and self.email_sender is not None:
            try:
                ok = self._send_email(
                    target["emails"], rendered["subject"], rendered["body"]
                )
                channels.append("email")
                success = success and ok
            except Exception as exc:  # pragma: no cover - defensive
                success = False
                error = str(exc)

        if (
            target["slack_channel"]
            and self.slack_sender is not None
            and self.slack_webhook_url
        ):
            try:
                slack_text = f"*{rendered['subject']}*\n{rendered['body']}"
                ok = self.slack_sender(
                    self.slack_webhook_url,
                    slack_text,
                    channel=target["slack_channel"],
                )
                channels.append("slack")
                success = success and bool(ok)
            except Exception as exc:  # pragma: no cover - defensive
                success = False
                error = str(exc)

        event = EscalationEvent(
            timer_id=placeholder.timer_id,
            factor_id=placeholder.factor_id,
            stage=placeholder.stage,
            level=digest_level,
            reason="digest",
            channels=channels,
            recipients=list(target["emails"]),
            dispatched_at=as_of,
            success=success,
            error=error,
        )
        self.audit.append(event)
        logger.info(
            "Daily digest dispatched overdue=%d warnings=%d success=%s",
            len(overdue), len(warnings), success,
        )
        return event


__all__ = [
    "EscalationTargets",
    "EscalationEvent",
    "EscalationDispatcher",
    "TEMPLATES",
    "render_template",
]
