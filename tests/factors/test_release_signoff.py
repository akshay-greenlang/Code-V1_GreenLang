# -*- coding: utf-8 -*-
"""Tests for release signoff workflow (F024)."""

from __future__ import annotations

import pytest

from greenlang.factors.quality.release_signoff import (
    ReleaseSignoff,
    SignoffItem,
    approve_release,
    release_signoff_checklist,
)


def _manifest(**kw):
    base = {
        "edition_id": "2024.04.0",
        "status": "pending",
        "changelog": ["Test change"],
        "factor_count": 100,
    }
    base.update(kw)
    return base


def _qa_report(passed=100, failed=0, total=100):
    return {"total_factors": total, "total_passed": passed, "total_failed": failed}


def _dedup_report(human_review=0):
    return {"human_review": human_review, "exact_duplicates": 0, "near_duplicates": 0}


def _consistency_report(reviews=0):
    return {"total_reviews": reviews, "total_activities": 10, "total_ok": 10}


# ---- SignoffItem ----

def test_signoff_item_to_dict():
    item = SignoffItem(item_id="S1", label="QA gates", ok=True, detail="All passed")
    d = item.to_dict()
    assert d["item_id"] == "S1"
    assert d["ok"] is True


# ---- release_signoff_checklist ----

def test_signoff_has_9_items():
    signoff = release_signoff_checklist("2024.04.0", _manifest())
    assert len(signoff.items) == 9


def test_signoff_all_pass():
    signoff = release_signoff_checklist(
        "2024.04.0",
        _manifest(),
        qa_report=_qa_report(),
        dedup_report=_dedup_report(),
        consistency_report=_consistency_report(),
        changelog_reviewed=True,
        methodology_signed=True,
        legal_confirmed=True,
        regression_passed=True,
        load_test_passed=True,
        gold_eval_precision=0.92,
    )
    assert signoff.passed_count == 9
    assert signoff.failed_count == 0
    assert signoff.all_required_passed
    assert signoff.ready_for_release


def test_signoff_qa_fail():
    signoff = release_signoff_checklist(
        "2024.04.0",
        _manifest(),
        qa_report=_qa_report(passed=90, failed=10),
    )
    s1 = next(i for i in signoff.items if i.item_id == "S1")
    assert not s1.ok
    assert "90/100" in s1.detail


def test_signoff_dedup_issues():
    signoff = release_signoff_checklist(
        "2024.04.0",
        _manifest(),
        dedup_report=_dedup_report(human_review=5),
    )
    s2 = next(i for i in signoff.items if i.item_id == "S2")
    assert not s2.ok
    assert "5" in s2.detail


def test_signoff_consistency_issues():
    signoff = release_signoff_checklist(
        "2024.04.0",
        _manifest(),
        consistency_report=_consistency_report(reviews=3),
    )
    s3 = next(i for i in signoff.items if i.item_id == "S3")
    assert not s3.ok


def test_signoff_changelog_not_reviewed():
    signoff = release_signoff_checklist("2024.04.0", _manifest(), changelog_reviewed=False)
    s4 = next(i for i in signoff.items if i.item_id == "S4")
    assert not s4.ok


def test_signoff_empty_changelog():
    signoff = release_signoff_checklist("2024.04.0", _manifest(changelog=[]))
    s4 = next(i for i in signoff.items if i.item_id == "S4")
    assert not s4.ok


def test_signoff_methodology_not_signed():
    signoff = release_signoff_checklist("2024.04.0", _manifest(), methodology_signed=False)
    s5 = next(i for i in signoff.items if i.item_id == "S5")
    assert not s5.ok


def test_signoff_legal_not_confirmed():
    signoff = release_signoff_checklist("2024.04.0", _manifest(), legal_confirmed=False)
    s6 = next(i for i in signoff.items if i.item_id == "S6")
    assert not s6.ok


def test_signoff_regression_not_run():
    signoff = release_signoff_checklist("2024.04.0", _manifest())
    s7 = next(i for i in signoff.items if i.item_id == "S7")
    assert not s7.ok
    assert s7.severity == "recommended"


def test_signoff_regression_passed():
    signoff = release_signoff_checklist("2024.04.0", _manifest(), regression_passed=True)
    s7 = next(i for i in signoff.items if i.item_id == "S7")
    assert s7.ok


def test_signoff_load_test_not_run():
    signoff = release_signoff_checklist("2024.04.0", _manifest())
    s8 = next(i for i in signoff.items if i.item_id == "S8")
    assert not s8.ok
    assert s8.severity == "recommended"


def test_signoff_gold_eval_above_threshold():
    signoff = release_signoff_checklist("2024.04.0", _manifest(), gold_eval_precision=0.90)
    s9 = next(i for i in signoff.items if i.item_id == "S9")
    assert s9.ok
    assert "0.900" in s9.detail


def test_signoff_gold_eval_below_threshold():
    signoff = release_signoff_checklist("2024.04.0", _manifest(), gold_eval_precision=0.70)
    s9 = next(i for i in signoff.items if i.item_id == "S9")
    assert not s9.ok
    assert "below" in s9.detail


def test_signoff_recommended_dont_block_release():
    """Recommended items (S7, S8, S9) don't block release."""
    signoff = release_signoff_checklist(
        "2024.04.0",
        _manifest(),
        qa_report=_qa_report(),
        dedup_report=_dedup_report(),
        consistency_report=_consistency_report(),
        changelog_reviewed=True,
        methodology_signed=True,
        legal_confirmed=True,
        # S7, S8, S9 not provided -> recommended, not blocking
    )
    assert signoff.all_required_passed
    assert signoff.ready_for_release
    assert signoff.failed_count == 3  # S7, S8, S9


def test_signoff_to_dict():
    signoff = release_signoff_checklist("2024.04.0", _manifest())
    d = signoff.to_dict()
    assert d["edition_id"] == "2024.04.0"
    assert d["total_items"] == 9
    assert "passed_items" in d
    assert "failed_items" in d


# ---- approve_release ----

def test_approve_release_success():
    signoff = release_signoff_checklist(
        "2024.04.0",
        _manifest(),
        qa_report=_qa_report(),
        dedup_report=_dedup_report(),
        consistency_report=_consistency_report(),
        changelog_reviewed=True,
        methodology_signed=True,
        legal_confirmed=True,
    )
    result = approve_release(signoff, "alice@greenlang.io", notes="LGTM")
    assert result.approved
    assert result.approver == "alice@greenlang.io"
    assert result.approved_at is not None
    assert result.notes == "LGTM"


def test_approve_release_blocked():
    signoff = release_signoff_checklist("2024.04.0", _manifest())
    with pytest.raises(ValueError, match="required items failed"):
        approve_release(signoff, "alice@greenlang.io")


def test_approve_release_force():
    signoff = release_signoff_checklist("2024.04.0", _manifest())
    result = approve_release(signoff, "admin@greenlang.io", force=True, notes="Emergency release")
    assert result.approved


# ---- ReleaseSignoff properties ----

def test_ready_for_release_false_when_empty():
    signoff = ReleaseSignoff(edition_id="test")
    assert not signoff.ready_for_release
