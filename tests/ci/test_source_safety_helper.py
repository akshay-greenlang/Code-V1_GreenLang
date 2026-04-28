# -*- coding: utf-8 -*-
"""Direct unit tests for greenlang/factors/ingestion/source_safety.py.

Phase 3 Wave 3.0 — Block 7 gate-4 runtime helper. The helper is also
exercised indirectly by the IngestionPipelineRunner; these tests pin the
contract.
"""
from __future__ import annotations

import pytest

from greenlang.factors.ingestion.source_safety import (
    BLOCKING_STATUSES,
    SourceNotApprovedForEnvError,
    assert_source_safe_for_env,
    is_source_safe_for_env,
    parse_release_milestone,
)


# ---------------------------------------------------------------------------
# parse_release_milestone
# ---------------------------------------------------------------------------

def test_parse_release_milestone_v0_1():
    assert parse_release_milestone("v0.1") == (0, 1)


def test_parse_release_milestone_v2_5():
    # Spec: "v2.5-rc1" -> (2,5) OR None per implementation. Our parser
    # rejects suffixed milestones (returns None).
    out = parse_release_milestone("v2.5-rc1")
    assert out in (None, (2, 5))


def test_parse_release_milestone_garbage_returns_none():
    assert parse_release_milestone(None) is None
    assert parse_release_milestone("") is None
    assert parse_release_milestone("0.1") is None
    assert parse_release_milestone("vNotAVersion") is None
    assert parse_release_milestone(42) is None


# ---------------------------------------------------------------------------
# is_source_safe_for_env
# ---------------------------------------------------------------------------

def test_is_safe_alpha_v01_in_production_is_true():
    entry = {
        "source_id": "epa_hub",
        "status": "alpha_v0_1",
        "release_milestone": "v0.1",
    }
    assert is_source_safe_for_env(entry, "production") is True


def test_is_safe_pending_legal_in_production_is_false():
    entry = {
        "source_id": "pending",
        "status": "pending_legal_review",
        "release_milestone": "v0.1",
    }
    assert is_source_safe_for_env(entry, "production") is False


def test_is_safe_blocked_in_production_is_false():
    entry = {
        "source_id": "blk",
        "status": "blocked",
        "release_milestone": "v0.1",
    }
    assert is_source_safe_for_env(entry, "production") is False


def test_is_safe_v05_milestone_in_production_is_false():
    entry = {
        "source_id": "demo",
        "status": "alpha_v0_1",
        "release_milestone": "v0.5",
    }
    assert is_source_safe_for_env(entry, "production") is False


def test_is_safe_in_dev_always_true():
    entry = {
        "source_id": "anything",
        "status": "pending_legal_review",
        "release_milestone": "v9.9",
    }
    assert is_source_safe_for_env(entry, "dev") is True


def test_is_safe_in_staging_always_true():
    entry = {
        "source_id": "demo",
        "status": "blocked",
        "release_milestone": "v2.0",
    }
    assert is_source_safe_for_env(entry, "staging") is True


# ---------------------------------------------------------------------------
# assert_source_safe_for_env
# ---------------------------------------------------------------------------

def test_assert_raises_on_pending_legal_with_reason_status_blocked():
    entry = {
        "source_id": "pending",
        "status": "pending_legal_review",
        "release_milestone": "v0.1",
    }
    with pytest.raises(SourceNotApprovedForEnvError) as ei:
        assert_source_safe_for_env(entry, "production")
    err = ei.value
    # IngestionError stores details; the helper sets reason='status_blocked'.
    details = getattr(err, "details", {}) or {}
    assert details.get("reason") == "status_blocked"


def test_assert_raises_on_future_milestone_with_reason_release_milestone_too_late():
    entry = {
        "source_id": "demo",
        "status": "alpha_v0_1",
        "release_milestone": "v0.5",
    }
    with pytest.raises(SourceNotApprovedForEnvError) as ei:
        assert_source_safe_for_env(entry, "production")
    details = getattr(ei.value, "details", {}) or {}
    assert details.get("reason") == "release_milestone_too_late"


def test_assert_raises_on_missing_milestone_with_reason_release_milestone_missing():
    entry = {
        "source_id": "demo",
        "status": "alpha_v0_1",
        "release_milestone": None,
    }
    with pytest.raises(SourceNotApprovedForEnvError) as ei:
        assert_source_safe_for_env(entry, "production")
    details = getattr(ei.value, "details", {}) or {}
    assert details.get("reason") == "release_milestone_missing"


def test_assert_no_raise_in_dev():
    entry = {
        "source_id": "anything",
        "status": "blocked",
        "release_milestone": "v9.9",
    }
    # Should not raise.
    assert_source_safe_for_env(entry, "dev")


def test_assert_raises_on_unknown_env():
    entry = {
        "source_id": "demo",
        "status": "alpha_v0_1",
        "release_milestone": "v0.1",
    }
    with pytest.raises(SourceNotApprovedForEnvError) as ei:
        assert_source_safe_for_env(entry, "qa")
    details = getattr(ei.value, "details", {}) or {}
    assert details.get("reason") == "unknown_env"


def test_blocking_statuses_constant_contains_pending_and_blocked():
    assert "pending_legal_review" in BLOCKING_STATUSES
    assert "blocked" in BLOCKING_STATUSES
