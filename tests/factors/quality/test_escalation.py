# -*- coding: utf-8 -*-
"""Tests for escalation dispatch (GAP-15)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import pytest

from greenlang.factors.quality.escalation import (
    EscalationDispatcher,
    EscalationEvent,
    EscalationTargets,
    TEMPLATES,
    render_template,
)
from greenlang.factors.quality.sla import SLAStage, SLATimer, SLATimerStatus


T0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _SlackStub:
    def __init__(self, succeed: bool = True) -> None:
        self.calls: List[Tuple[str, str, Dict[str, Any]]] = []
        self.succeed = succeed

    def __call__(self, url: str, message: str, **kwargs: Any) -> bool:
        self.calls.append((url, message, kwargs))
        return self.succeed


class _EmailStub:
    def __init__(self, succeed: bool = True) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.succeed = succeed

    def __call__(
        self,
        smtp_host,
        smtp_port,
        smtp_user,
        smtp_password,
        from_addr,
        to_addrs,
        subject,
        body,
    ) -> bool:
        self.calls.append(
            {
                "smtp_host": smtp_host,
                "smtp_port": smtp_port,
                "from": from_addr,
                "to": list(to_addrs),
                "subject": subject,
                "body": body,
            }
        )
        return self.succeed


def _make_timer(
    *,
    factor_id: str = "EF:TEST:1",
    stage: SLAStage = SLAStage.DETAILED_REVIEW,
    deadline_offset_hours: int = 120,
    tier: str = "pro",
) -> SLATimer:
    return SLATimer(
        timer_id="timer-1",
        factor_id=factor_id,
        stage=stage,
        started_at=T0,
        deadline=T0 + timedelta(hours=deadline_offset_hours),
        warning_at=T0 + timedelta(hours=int(deadline_offset_hours * 0.75)),
        status=SLATimerStatus.ACTIVE,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def test_all_templates_have_subject_and_body():
    for reason, tpl in TEMPLATES.items():
        assert "subject" in tpl
        assert "body" in tpl


def test_render_template_substitutes_context():
    ctx = {
        "factor_id": "EF:1",
        "stage": "detailed_review",
        "deadline": "2026-04-05T00:00:00Z",
        "started_at": "2026-04-01T00:00:00Z",
        "level": 2,
    }
    out = render_template("overdue", ctx)
    assert "EF:1" in out["subject"]
    assert "detailed_review" in out["body"]
    assert "Level: 2" in out["body"]


def test_render_template_safe_on_missing_keys():
    out = render_template("warning", {"factor_id": "only-id"})
    assert "only-id" in out["subject"]
    # Missing keys remain as placeholders, do not raise.
    assert "{stage}" in out["subject"] or "{stage}" in out["body"]


def test_render_template_unknown_reason_falls_back():
    out = render_template("nonsense", {"factor_id": "EF:1", "stage": "x",
                                        "deadline": "", "started_at": "", "level": 1})
    assert out["subject"]  # non-empty


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _dispatcher(
    targets: EscalationTargets,
    *,
    slack_ok: bool = True,
    email_ok: bool = True,
) -> Tuple[EscalationDispatcher, _SlackStub, _EmailStub]:
    slack = _SlackStub(slack_ok)
    email = _EmailStub(email_ok)
    d = EscalationDispatcher(
        targets=targets,
        slack_sender=slack,
        email_sender=email,
        slack_webhook_url="https://hooks.example/xyz",
        smtp_config={"smtp_host": "smtp.example", "smtp_port": 587},
        from_address="noreply@greenlang.io",
    )
    return d, slack, email


def test_dispatch_sends_email_and_slack_for_level1():
    targets = EscalationTargets(
        level1_emails=["reviewer@example.com"],
        level1_slack="#factors-reviewers",
    )
    d, slack, email = _dispatcher(targets)
    timer = _make_timer()
    event = d.dispatch(timer, reason="overdue", level=1)
    assert event.success is True
    assert "email" in event.channels
    assert "slack" in event.channels
    assert email.calls[0]["to"] == ["reviewer@example.com"]
    assert "#factors-reviewers" in str(slack.calls[0][2])


def test_dispatch_records_failure_when_sender_returns_false():
    targets = EscalationTargets(
        level2_emails=["team-lead@example.com"],
    )
    d, _slack, _email = _dispatcher(targets, email_ok=False)
    timer = _make_timer()
    event = d.dispatch(timer, reason="overdue", level=2)
    assert event.success is False
    assert event.error is not None


def test_dispatch_level_escalation_targets_routed_correctly():
    targets = EscalationTargets(
        level1_emails=["l1@x"],
        level2_emails=["l2@x"],
        level3_emails=["l3@x"],
        level4_emails=["l4@x"],
    )
    d, _slack, email = _dispatcher(targets)
    timer = _make_timer()
    for level in (1, 2, 3, 4):
        d.dispatch(timer, reason="overdue", level=level)
    levels = [c["to"][0] for c in email.calls]
    assert levels == ["l1@x", "l2@x", "l3@x", "l4@x"]


def test_dispatch_skips_channels_with_no_targets():
    targets = EscalationTargets(level1_emails=[], level1_slack=None)
    d, slack, email = _dispatcher(targets)
    timer = _make_timer()
    event = d.dispatch(timer, reason="warning", level=1)
    # No channels configured -> success vacuous, no calls made.
    assert email.calls == []
    assert slack.calls == []
    assert event.channels == []


def test_dispatch_records_audit_trail():
    targets = EscalationTargets(level1_emails=["a@x"])
    d, _slack, _email = _dispatcher(targets)
    timer = _make_timer()
    d.dispatch(timer, reason="overdue", level=1)
    d.dispatch(timer, reason="warning", level=1)
    assert len(d.audit) == 2
    assert d.audit[0].reason == "overdue"
    assert d.audit[1].reason == "warning"


def test_event_to_dict_round_trip():
    event = EscalationEvent(
        timer_id="t1",
        factor_id="EF:1",
        stage=SLAStage.DETAILED_REVIEW,
        level=2,
        reason="overdue",
        channels=["email"],
        recipients=["a@x"],
        dispatched_at=T0,
        success=True,
    )
    payload = event.to_dict()
    assert payload["stage"] == "detailed_review"
    assert payload["level"] == 2
    assert payload["recipients"] == ["a@x"]


# ---------------------------------------------------------------------------
# Daily digest
# ---------------------------------------------------------------------------


def test_daily_digest_includes_counts_and_details():
    targets = EscalationTargets(
        level3_emails=["ops@greenlang.io"],
        level3_slack="#factors-ops",
    )
    d, slack, email = _dispatcher(targets)
    overdue = [_make_timer(factor_id="EF:over:1")]
    warnings = [_make_timer(factor_id="EF:warn:1"), _make_timer(factor_id="EF:warn:2")]
    event = d.daily_digest(overdue, warnings, digest_level=3, as_of=T0)
    assert event.reason == "digest"
    assert event.success
    # One email with counts embedded.
    assert len(email.calls) == 1
    assert "Overdue timers: 1" in email.calls[0]["body"]
    assert "approaching deadline: 2" in email.calls[0]["body"]


def test_daily_digest_with_nothing_still_emits_event():
    targets = EscalationTargets(level3_emails=["ops@greenlang.io"])
    d, _slack, email = _dispatcher(targets)
    event = d.daily_digest([], [], digest_level=3, as_of=T0)
    assert event.reason == "digest"
    assert "Overdue timers: 0" in email.calls[0]["body"]


def test_targets_for_level_clamps_invalid_level():
    targets = EscalationTargets(level1_emails=["l1@x"], level4_emails=["l4@x"])
    assert targets.for_level(0)["emails"] == ["l1@x"]
    assert targets.for_level(99)["emails"] == ["l4@x"]
