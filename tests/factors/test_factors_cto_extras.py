# -*- coding: utf-8 -*-
"""Smoke tests for CTO backlog helpers."""

from __future__ import annotations

from greenlang.factors.quality.promotion import PromotionState, transition
from greenlang.factors.watch.doc_diff import diff_text_versions, fingerprint_text


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


def test_doc_diff():
    assert fingerprint_text("a") != fingerprint_text("b")
    changed, msg = diff_text_versions("x", "y")
    assert changed and "delta" in msg
