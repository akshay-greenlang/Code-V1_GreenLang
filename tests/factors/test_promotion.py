# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.quality.promotion."""

from __future__ import annotations

from greenlang.factors.quality.promotion import PromotionState, transition


def test_promotion_to_certified():
    nxt, reason = transition(
        PromotionState.PREVIEW,
        qa_pass=True,
        methodology_signed=True,
        legal_ok=True,
        approval_gate_ok=True,
    )
    assert nxt == PromotionState.CERTIFIED
    assert reason == "promoted"


def test_promotion_from_preview_stays_without_qa():
    nxt, reason = transition(
        PromotionState.PREVIEW,
        qa_pass=False,
        methodology_signed=True,
        legal_ok=True,
        approval_gate_ok=True,
    )
    assert nxt == PromotionState.PREVIEW
    assert "qa" in reason.lower()


def test_promotion_blocked_without_methodology():
    nxt, reason = transition(
        PromotionState.PREVIEW,
        qa_pass=True,
        methodology_signed=False,
        legal_ok=True,
        approval_gate_ok=True,
    )
    assert nxt == PromotionState.PREVIEW
    assert "methodology" in reason.lower()


def test_rejected_is_terminal():
    nxt, reason = transition(
        PromotionState.REJECTED,
        qa_pass=True,
        methodology_signed=True,
        legal_ok=True,
        approval_gate_ok=True,
    )
    assert nxt == PromotionState.REJECTED
    assert reason == "terminal"


def test_invalid_gate_blocks():
    nxt, reason = transition(
        PromotionState.PREVIEW,
        qa_pass=True,
        methodology_signed=True,
        legal_ok=True,
        approval_gate_ok=False,
    )
    assert nxt == PromotionState.PREVIEW
    assert "approval" in reason.lower()


def test_promotion_states_enum():
    assert PromotionState.PREVIEW.value == "preview"
    assert PromotionState.CERTIFIED.value == "certified"
    assert PromotionState.REJECTED.value == "rejected"
    assert len(PromotionState) == 3
