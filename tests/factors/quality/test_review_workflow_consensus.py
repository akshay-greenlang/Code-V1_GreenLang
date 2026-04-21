# -*- coding: utf-8 -*-
"""
Integration tests: review_workflow state machine + consensus + SLA (GAP-14 + 15).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from greenlang.factors.quality.consensus import (
    ConsensusStatus,
    build_vote,
    ensure_consensus_schema,
    record_vote,
)
from greenlang.factors.quality.review_workflow import (
    InsufficientConsensusError,
    SLAExpiredError,
    SLAStage,
    SLATimerStatus,
    WorkflowState,
    advance_workflow_state,
    submit_vote,
)
from greenlang.factors.quality.sla import (
    auto_reject_stale,
    ensure_sla_schema,
    get_timer_for_factor,
    start_sla_timer,
)


T0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    ensure_consensus_schema(c)
    ensure_sla_schema(c)
    try:
        yield c
    finally:
        c.close()


def _cast(conn, factor_id, reviewer_id, role, decision="APPROVE", **kwargs):
    vote = build_vote(
        factor_id=factor_id,
        reviewer_id=reviewer_id,
        reviewer_role=role,
        decision=decision,
        **kwargs,
    )
    record_vote(conn, vote)
    return vote


# ---------------------------------------------------------------------------
# Happy path: community tier end-to-end
# ---------------------------------------------------------------------------


def test_community_happy_path_draft_to_published(conn):
    factor_id = "EF:COMMUNITY:1"
    # DRAFT -> UNDER_REVIEW: one methodology_lead approval.
    _cast(conn, factor_id, "alice", "methodology_lead")
    outcome1 = advance_workflow_state(
        conn, factor_id,
        WorkflowState.DRAFT, WorkflowState.UNDER_REVIEW,
        tier="community", now=T0,
    )
    assert outcome1.allowed
    assert outcome1.consensus is not None
    assert outcome1.consensus.status == ConsensusStatus.APPROVED
    assert outcome1.new_timer is not None
    assert outcome1.new_timer.stage == SLAStage.DETAILED_REVIEW

    # UNDER_REVIEW -> APPROVED (still any_of_n in community).
    outcome2 = advance_workflow_state(
        conn, factor_id,
        WorkflowState.UNDER_REVIEW, WorkflowState.APPROVED,
        tier="community", now=T0 + timedelta(hours=4),
    )
    assert outcome2.allowed
    assert outcome2.completed_timer is not None
    assert outcome2.completed_timer.status == SLATimerStatus.COMPLETED
    assert outcome2.new_timer is not None
    assert outcome2.new_timer.stage == SLAStage.FINAL_APPROVAL

    # APPROVED -> PUBLISHED.
    outcome3 = advance_workflow_state(
        conn, factor_id,
        WorkflowState.APPROVED, WorkflowState.PUBLISHED,
        tier="community", now=T0 + timedelta(hours=8),
    )
    assert outcome3.allowed
    assert outcome3.completed_timer.status == SLATimerStatus.COMPLETED
    # PUBLISHED starts a deprecation_notice timer.
    assert outcome3.new_timer.stage == SLAStage.DEPRECATION_NOTICE


# ---------------------------------------------------------------------------
# Consensus gate blocks advancement
# ---------------------------------------------------------------------------


def test_pro_tier_requires_two_approvals(conn):
    factor_id = "EF:PRO:1"
    # One approval is not enough for pro (2-of-3 methodology leads).
    _cast(conn, factor_id, "alice", "methodology_lead")
    with pytest.raises(InsufficientConsensusError) as excinfo:
        advance_workflow_state(
            conn, factor_id,
            WorkflowState.DRAFT, WorkflowState.UNDER_REVIEW,
            tier="pro", now=T0,
        )
    assert excinfo.value.factor_id == factor_id
    assert excinfo.value.met_requirements.get("methodology_lead", 0) == 1

    # Add the second approval -> advances.
    _cast(conn, factor_id, "bob", "methodology_lead")
    outcome = advance_workflow_state(
        conn, factor_id,
        WorkflowState.DRAFT, WorkflowState.UNDER_REVIEW,
        tier="pro", now=T0,
    )
    assert outcome.allowed


def test_enterprise_cbam_requires_compliance_lead(conn):
    factor_id = "EF:CBAM:1"
    # Methodology + QA + legal approvals aren't enough without compliance_lead.
    _cast(conn, factor_id, "methods@x", "methodology_lead", weight=2)
    _cast(conn, factor_id, "qa@x", "qa_lead", weight=1)
    _cast(conn, factor_id, "legal@x", "legal_lead", weight=1)
    with pytest.raises(InsufficientConsensusError):
        advance_workflow_state(
            conn, factor_id,
            WorkflowState.DRAFT, WorkflowState.UNDER_REVIEW,
            tier="enterprise", factor_type="cbam", now=T0,
        )

    _cast(conn, factor_id, "comp@x", "compliance_lead", weight=2)
    outcome = advance_workflow_state(
        conn, factor_id,
        WorkflowState.DRAFT, WorkflowState.UNDER_REVIEW,
        tier="enterprise", factor_type="cbam", now=T0,
    )
    assert outcome.allowed


def test_rejection_short_circuits_transition(conn):
    factor_id = "EF:REJECT:1"
    _cast(conn, factor_id, "alice", "methodology_lead")
    _cast(
        conn,
        factor_id,
        "bob",
        "qa_lead",
        decision="REJECT",
        dissent_notes="methodology flawed",
    )
    # community tier only needs 1 approver but a single REJECT still flips status.
    with pytest.raises(InsufficientConsensusError):
        advance_workflow_state(
            conn, factor_id,
            WorkflowState.DRAFT, WorkflowState.UNDER_REVIEW,
            tier="community", now=T0,
        )


# ---------------------------------------------------------------------------
# SLA gate
# ---------------------------------------------------------------------------


def test_sla_expired_blocks_transition(conn):
    factor_id = "EF:EXPIRED:1"
    # Seed a timer and force it to EXPIRED via auto_reject_stale.
    timer = start_sla_timer(
        conn, factor_id, SLAStage.DETAILED_REVIEW, "enterprise_cbam", now=T0
    )
    auto_reject_stale(conn, timer, now=T0 + timedelta(hours=500))

    _cast(conn, factor_id, "alice", "methodology_lead")
    with pytest.raises(SLAExpiredError):
        advance_workflow_state(
            conn, factor_id,
            WorkflowState.UNDER_REVIEW, WorkflowState.APPROVED,
            tier="enterprise_cbam", now=T0 + timedelta(hours=600),
        )


def test_illegal_transition_rejected(conn):
    with pytest.raises(ValueError):
        advance_workflow_state(
            conn, "EF:WHATEVER",
            WorkflowState.DRAFT, WorkflowState.PUBLISHED,
            tier="community",
        )


# ---------------------------------------------------------------------------
# SLA completion + timer handoff
# ---------------------------------------------------------------------------


def test_transition_completes_outgoing_timer_and_starts_new(conn):
    factor_id = "EF:HANDOFF:1"
    _cast(conn, factor_id, "alice", "methodology_lead")
    advance_workflow_state(
        conn, factor_id,
        WorkflowState.DRAFT, WorkflowState.UNDER_REVIEW,
        tier="community", now=T0,
    )
    # A detailed_review timer is active now.
    active = get_timer_for_factor(conn, factor_id, SLAStage.DETAILED_REVIEW)
    assert active is not None
    assert active.status == SLATimerStatus.ACTIVE

    advance_workflow_state(
        conn, factor_id,
        WorkflowState.UNDER_REVIEW, WorkflowState.APPROVED,
        tier="community", now=T0 + timedelta(hours=4),
    )
    # Detailed review now completed, final_approval active.
    completed = get_timer_for_factor(
        conn, factor_id, SLAStage.DETAILED_REVIEW, include_completed=True
    )
    assert completed is not None
    assert completed.status == SLATimerStatus.COMPLETED
    final_timer = get_timer_for_factor(conn, factor_id, SLAStage.FINAL_APPROVAL)
    assert final_timer is not None
    assert final_timer.status == SLATimerStatus.ACTIVE


def test_submit_vote_stores_vote(conn):
    vote = build_vote(
        factor_id="EF:SUBMIT:1",
        reviewer_id="alice",
        reviewer_role="methodology_lead",
        decision="APPROVE",
    )
    submit_vote(conn, vote)
    row = conn.execute(
        "SELECT reviewer_id, decision FROM factors_review_votes WHERE factor_id=?",
        ("EF:SUBMIT:1",),
    ).fetchone()
    assert row == ("alice", "APPROVE")


def test_self_approval_blocked_in_integration(conn):
    factor_id = "EF:SELF:1"
    _cast(conn, factor_id, "author@x", "methodology_lead")
    with pytest.raises(InsufficientConsensusError):
        advance_workflow_state(
            conn, factor_id,
            WorkflowState.DRAFT, WorkflowState.UNDER_REVIEW,
            tier="community", factor_author="author@x", now=T0,
        )


# ---------------------------------------------------------------------------
# Existing review_workflow surface still works.
# ---------------------------------------------------------------------------


def test_existing_create_review_flow_untouched():
    from greenlang.factors.quality.review_workflow import (
        METHODOLOGY_CHECKLIST,
        create_review,
        submit_decision,
        update_checklist_item,
    )

    review = create_review(
        "2026.04.0", ["EF:COMMUNITY:1"], "epa", "alice@greenlang.io"
    )
    assert len(review.checklist) == len(METHODOLOGY_CHECKLIST)
    for item in review.checklist:
        update_checklist_item(review, item.item_id, passed=True)
    finalised = submit_decision(review, "approved", notes="all good")
    assert finalised.decision == "approved"
