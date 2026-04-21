# -*- coding: utf-8 -*-
"""Tests for the multi-reviewer consensus engine (GAP-14)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from greenlang.factors.quality.consensus import (
    CONSENSUS_SCHEMA_SQL,
    ConsensusConfig,
    ConsensusRule,
    ConsensusStatus,
    DissentCaptureRequiredError,
    ReviewerRequirement,
    ReviewerVote,
    VoteDecision,
    build_vote,
    dissent_report,
    ensure_consensus_schema,
    evaluate_consensus,
    get_pending_votes,
    load_votes,
    record_vote,
    save_config,
    tier_based_requirements,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    ensure_consensus_schema(c)
    try:
        yield c
    finally:
        c.close()


def _ts(offset_minutes: int = 0) -> datetime:
    return datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=offset_minutes
    )


def _vote(
    reviewer_id: str,
    role: str,
    decision: str,
    *,
    factor_id: str = "EF:TEST:1",
    dissent: str = "",
    rationale: str = "",
    weight: int = 1,
    offset: int = 0,
) -> ReviewerVote:
    return build_vote(
        factor_id=factor_id,
        reviewer_id=reviewer_id,
        reviewer_role=role,
        decision=decision,
        rationale=rationale or None,
        dissent_notes=dissent or None,
        weight=weight,
        timestamp=_ts(offset),
    )


# ---------------------------------------------------------------------------
# Rule-level evaluation
# ---------------------------------------------------------------------------


def test_any_of_n_approves_with_single_vote():
    cfg = ConsensusConfig(
        rule=ConsensusRule.ANY_OF_N,
        reviewer_requirements=(ReviewerRequirement(role="methodology_lead", min_count=1),),
        quorum=1,
    )
    votes = [_vote("alice", "methodology_lead", "APPROVE")]
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert result.status == ConsensusStatus.APPROVED
    assert result.approval_count == 1
    assert result.is_approved


def test_any_of_n_pending_when_no_approvals():
    cfg = ConsensusConfig(
        rule=ConsensusRule.ANY_OF_N,
        reviewer_requirements=(ReviewerRequirement(role="methodology_lead", min_count=1),),
        quorum=1,
    )
    result = evaluate_consensus("EF:TEST:1", [], cfg)
    assert result.status == ConsensusStatus.INSUFFICIENT_QUORUM


def test_n_of_m_2_of_3_methodology_leads_approves():
    cfg = ConsensusConfig(
        rule=ConsensusRule.N_OF_M,
        reviewer_requirements=(ReviewerRequirement(role="methodology_lead", min_count=2),),
        quorum=2,
    )
    votes = [
        _vote("alice", "methodology_lead", "APPROVE"),
        _vote("bob", "methodology_lead", "APPROVE"),
    ]
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert result.status == ConsensusStatus.APPROVED
    assert result.met_requirements["methodology_lead"] == 2


def test_n_of_m_pending_with_only_one_approval():
    cfg = ConsensusConfig(
        rule=ConsensusRule.N_OF_M,
        reviewer_requirements=(ReviewerRequirement(role="methodology_lead", min_count=2),),
        quorum=2,
    )
    votes = [_vote("alice", "methodology_lead", "APPROVE")]
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert result.status == ConsensusStatus.INSUFFICIENT_QUORUM


def test_unanimous_requires_all_roles():
    cfg = ConsensusConfig(
        rule=ConsensusRule.UNANIMOUS,
        reviewer_requirements=(
            ReviewerRequirement(role="methodology_lead", min_count=1),
            ReviewerRequirement(role="qa_lead", min_count=1),
            ReviewerRequirement(role="legal_lead", min_count=1),
        ),
        quorum=3,
    )
    # Only two of three present -> PENDING.
    votes = [
        _vote("alice", "methodology_lead", "APPROVE"),
        _vote("bob", "qa_lead", "APPROVE"),
    ]
    pending = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert pending.status == ConsensusStatus.INSUFFICIENT_QUORUM

    # Add the third role -> APPROVED.
    votes.append(_vote("carol", "legal_lead", "APPROVE"))
    approved = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert approved.status == ConsensusStatus.APPROVED


def test_unanimous_rejected_on_single_reject():
    cfg = ConsensusConfig(
        rule=ConsensusRule.UNANIMOUS,
        reviewer_requirements=(
            ReviewerRequirement(role="methodology_lead", min_count=1),
            ReviewerRequirement(role="qa_lead", min_count=1),
        ),
        quorum=2,
    )
    votes = [
        _vote("alice", "methodology_lead", "APPROVE"),
        _vote("bob", "qa_lead", "REJECT", dissent="bad methodology"),
    ]
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert result.status == ConsensusStatus.REJECTED
    assert result.rejection_count == 1


def test_weighted_voting_counts_role_weights():
    cfg = ConsensusConfig(
        rule=ConsensusRule.WEIGHTED,
        reviewer_requirements=(
            ReviewerRequirement(role="methodology_lead", min_count=1, weight=2),
            ReviewerRequirement(role="qa_lead", min_count=1, weight=1),
        ),
        quorum=2,
    )
    votes = [
        _vote("alice", "methodology_lead", "APPROVE", weight=2),
        _vote("bob", "qa_lead", "APPROVE", weight=1),
    ]
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert result.status == ConsensusStatus.APPROVED


def test_weighted_pending_when_weight_insufficient():
    cfg = ConsensusConfig(
        rule=ConsensusRule.WEIGHTED,
        reviewer_requirements=(
            ReviewerRequirement(role="methodology_lead", min_count=2, weight=2),
        ),
        quorum=2,
    )
    # Only one methodology_lead even though quorum met => PENDING.
    votes = [
        _vote("alice", "methodology_lead", "APPROVE", weight=2),
        _vote("bob", "qa_lead", "APPROVE", weight=1),
    ]
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert result.status == ConsensusStatus.PENDING


# ---------------------------------------------------------------------------
# Dissent + self-approval
# ---------------------------------------------------------------------------


def test_dissent_capture_required_for_reject_without_notes():
    cfg = ConsensusConfig(
        rule=ConsensusRule.ANY_OF_N,
        reviewer_requirements=(ReviewerRequirement(role="methodology_lead", min_count=1),),
        quorum=1,
        dissent_capture_required=True,
    )
    votes = [
        _vote("alice", "methodology_lead", "APPROVE"),
        _vote("bob", "qa_lead", "REJECT"),  # no dissent notes
    ]
    # The reject vote is dropped by validation -> result ignores it and status is APPROVED.
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    # Only the valid approve should remain.
    assert len(result.votes) == 1
    assert result.status == ConsensusStatus.APPROVED


def test_dissent_allowed_when_not_required():
    cfg = ConsensusConfig(
        rule=ConsensusRule.UNANIMOUS,
        reviewer_requirements=(
            ReviewerRequirement(role="methodology_lead", min_count=1),
            ReviewerRequirement(role="qa_lead", min_count=1),
        ),
        quorum=2,
        dissent_capture_required=False,
    )
    votes = [
        _vote("alice", "methodology_lead", "APPROVE"),
        _vote("bob", "qa_lead", "ABSTAIN"),  # no dissent notes OK
    ]
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    assert result.dissent_captured is True  # no blocking non-approve votes


def test_self_approval_rejected_by_default():
    cfg = ConsensusConfig(
        rule=ConsensusRule.ANY_OF_N,
        reviewer_requirements=(ReviewerRequirement(role="methodology_lead", min_count=1),),
        quorum=1,
    )
    votes = [_vote("author@x", "methodology_lead", "APPROVE")]
    result = evaluate_consensus(
        "EF:TEST:1", votes, cfg, factor_author="author@x"
    )
    # Self-approval vote is filtered -> no valid approvals -> insufficient quorum.
    assert result.status == ConsensusStatus.INSUFFICIENT_QUORUM
    assert result.votes == []


def test_self_approval_allowed_when_configured():
    cfg = ConsensusConfig(
        rule=ConsensusRule.ANY_OF_N,
        reviewer_requirements=(ReviewerRequirement(role="methodology_lead", min_count=1),),
        quorum=1,
        allow_self_approval=True,
    )
    votes = [_vote("author@x", "methodology_lead", "APPROVE")]
    result = evaluate_consensus("EF:TEST:1", votes, cfg, factor_author="author@x")
    assert result.status == ConsensusStatus.APPROVED


# ---------------------------------------------------------------------------
# Tier-based defaults
# ---------------------------------------------------------------------------


def test_community_tier_defaults():
    cfg = tier_based_requirements("community")
    assert cfg.rule == ConsensusRule.ANY_OF_N
    assert cfg.quorum == 1
    assert cfg.sla_hours == 72


def test_pro_tier_defaults_2_of_3():
    cfg = tier_based_requirements("pro")
    assert cfg.rule == ConsensusRule.N_OF_M
    assert cfg.reviewer_requirements[0].min_count == 2
    assert cfg.sla_hours == 48


def test_enterprise_tier_defaults_three_roles():
    cfg = tier_based_requirements("enterprise")
    roles = {r.role for r in cfg.reviewer_requirements}
    assert {"methodology_lead", "qa_lead", "legal_lead"}.issubset(roles)
    assert cfg.quorum == 3
    assert cfg.rule == ConsensusRule.WEIGHTED


def test_enterprise_cbam_requires_compliance_lead():
    cfg = tier_based_requirements("enterprise", factor_type="cbam")
    roles = {r.role for r in cfg.reviewer_requirements}
    assert "compliance_lead" in roles
    assert cfg.quorum == 4
    assert cfg.sla_hours == 48


def test_unknown_tier_falls_back_to_community():
    cfg = tier_based_requirements("ultra-platinum")
    community = tier_based_requirements("community")
    assert cfg.rule == community.rule
    assert cfg.quorum == community.quorum


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_record_vote_roundtrip(conn):
    vote = _vote("alice", "methodology_lead", "APPROVE")
    record_vote(conn, vote)
    loaded = load_votes(conn, "EF:TEST:1")
    assert len(loaded) == 1
    assert loaded[0].reviewer_id == "alice"
    assert loaded[0].decision == VoteDecision.APPROVE


def test_record_vote_upsert_latest_wins(conn):
    first = _vote("alice", "methodology_lead", "APPROVE", offset=0)
    second = _vote("alice", "methodology_lead", "REJECT", dissent="changed mind", offset=60)
    record_vote(conn, first)
    record_vote(conn, second)
    loaded = load_votes(conn, "EF:TEST:1")
    assert len(loaded) == 1
    assert loaded[0].decision == VoteDecision.REJECT


def test_get_pending_votes_returns_unvoted(conn):
    record_vote(conn, _vote("alice", "methodology_lead", "APPROVE", factor_id="EF:A"))
    pending = get_pending_votes(
        conn, "alice", factor_ids=["EF:A", "EF:B", "EF:C"]
    )
    assert pending == ["EF:B", "EF:C"]


def test_dissent_report_lists_only_non_approvals(conn):
    record_vote(conn, _vote("alice", "methodology_lead", "APPROVE"))
    record_vote(
        conn, _vote("bob", "qa_lead", "REJECT", dissent="values out of range")
    )
    record_vote(
        conn, _vote("carol", "legal_lead", "ABSTAIN", dissent="license unclear")
    )
    notes = dissent_report(conn, "EF:TEST:1")
    assert len(notes) == 2
    reviewers = {n.reviewer_id for n in notes}
    assert reviewers == {"bob", "carol"}
    assert all(n.notes for n in notes)


def test_save_config_persists_policy(conn):
    cfg = tier_based_requirements("pro")
    cid = save_config(conn, factor_type="", tier="pro", config=cfg)
    row = conn.execute(
        "SELECT rule, quorum FROM factors_review_consensus_configs WHERE config_id=?",
        (cid,),
    ).fetchone()
    assert row is not None
    assert row[0] == "n_of_m"
    assert row[1] == 2


def test_schema_sql_constants_are_non_empty():
    # Simple guard so the migration-side SQL stays aligned.
    assert "factors_review_votes" in CONSENSUS_SCHEMA_SQL
    assert "factors_review_consensus_configs" in CONSENSUS_SCHEMA_SQL


# ---------------------------------------------------------------------------
# ReviewerVote (de)serialisation
# ---------------------------------------------------------------------------


def test_reviewer_vote_to_from_dict_roundtrip():
    vote = _vote("alice", "methodology_lead", "APPROVE", rationale="ok")
    payload = vote.to_dict()
    restored = ReviewerVote.from_dict(payload)
    assert restored.reviewer_id == vote.reviewer_id
    assert restored.decision == vote.decision
    assert restored.timestamp == vote.timestamp


def test_invalid_decision_raises():
    with pytest.raises(ValueError):
        build_vote(
            factor_id="EF:TEST:1",
            reviewer_id="alice",
            reviewer_role="methodology_lead",
            decision="MAYBE",  # invalid
        )


def test_consensus_result_to_dict_contains_counts():
    cfg = tier_based_requirements("pro")
    votes = [
        _vote("a", "methodology_lead", "APPROVE"),
        _vote("b", "methodology_lead", "APPROVE"),
    ]
    result = evaluate_consensus("EF:TEST:1", votes, cfg)
    d = result.to_dict()
    assert d["approval_count"] == 2
    assert d["status"] == "APPROVED"
    assert d["consensus_rule"] == "n_of_m"
