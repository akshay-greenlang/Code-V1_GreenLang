# -*- coding: utf-8 -*-
"""Tests for methodology review workflow (F023)."""

from __future__ import annotations

import pytest

from greenlang.factors.quality.review_workflow import (
    METHODOLOGY_CHECKLIST,
    ChecklistItem,
    ReviewAssignment,
    ReviewPriority,
    ReviewStatus,
    batch_review,
    create_review,
    submit_decision,
    update_checklist_item,
)


# ---- METHODOLOGY_CHECKLIST ----

def test_checklist_has_10_items():
    assert len(METHODOLOGY_CHECKLIST) == 10


def test_checklist_ids_unique():
    ids = [c["id"] for c in METHODOLOGY_CHECKLIST]
    assert len(ids) == len(set(ids))


# ---- ChecklistItem ----

def test_checklist_item_to_dict():
    item = ChecklistItem(item_id="C01", label="Source verified", passed=True, notes="OK")
    d = item.to_dict()
    assert d["item_id"] == "C01"
    assert d["passed"] is True


def test_checklist_item_from_dict():
    d = {"item_id": "C01", "label": "Test", "passed": False, "notes": "Fail"}
    item = ChecklistItem.from_dict(d)
    assert item.item_id == "C01"
    assert item.passed is False


# ---- ReviewAssignment ----

def test_create_review_default_checklist():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice@greenlang.io")
    assert len(review.checklist) == 10
    assert review.status == ReviewStatus.PENDING
    assert review.reviewer == "alice@greenlang.io"
    assert len(review.factor_ids) == 1


def test_review_checklist_initially_incomplete():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    assert not review.checklist_complete
    assert review.checklist_pass_count == 0


def test_review_to_dict():
    review = create_review("2024.04.0", ["EF:EPA:1", "EF:EPA:2"], "epa", "bob")
    d = review.to_dict()
    assert d["edition_id"] == "2024.04.0"
    assert d["source_id"] == "epa"
    assert len(d["factor_ids"]) == 2
    assert len(d["checklist"]) == 10
    assert d["checklist_complete"] is False


def test_review_from_dict():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    d = review.to_dict()
    restored = ReviewAssignment.from_dict(d)
    assert restored.review_id == review.review_id
    assert restored.edition_id == review.edition_id
    assert len(restored.checklist) == 10


def test_review_priority():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice", priority=ReviewPriority.CRITICAL)
    assert review.priority == ReviewPriority.CRITICAL


def test_review_due_date():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice", due_date="2024-05-01")
    assert review.due_date == "2024-05-01"


# ---- update_checklist_item ----

def test_update_checklist_item_found():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    ok = update_checklist_item(review, "C01", True, "Verified")
    assert ok
    assert review.checklist[0].passed is True
    assert review.checklist[0].notes == "Verified"


def test_update_checklist_item_not_found():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    ok = update_checklist_item(review, "C99", True)
    assert not ok


def test_update_all_checklist_items():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    for item in review.checklist:
        update_checklist_item(review, item.item_id, True)
    assert review.checklist_complete
    assert review.checklist_pass_count == 10
    assert review.all_passed


# ---- submit_decision ----

def test_submit_approved():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    # Complete checklist first
    for item in review.checklist:
        item.passed = True
    submit_decision(review, "approved", "All checks pass")
    assert review.status == ReviewStatus.APPROVED
    assert review.decision == "approved"
    assert review.completed_at is not None


def test_submit_rejected():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    submit_decision(review, "rejected", "Data quality issues")
    assert review.status == ReviewStatus.REJECTED
    assert review.decision == "rejected"


def test_submit_needs_revision():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    submit_decision(review, "needs_revision", "Fix unit conversion")
    assert review.status == ReviewStatus.NEEDS_REVISION


def test_submit_approved_incomplete_raises():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    with pytest.raises(ValueError, match="incomplete"):
        submit_decision(review, "approved")


def test_submit_invalid_decision_raises():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    with pytest.raises(ValueError, match="Invalid decision"):
        submit_decision(review, "maybe")


# ---- batch_review ----

def test_batch_approve_with_auto_checklist():
    reviews = [
        create_review("2024.04.0", [f"EF:EPA:{i}"], "epa", "alice")
        for i in range(3)
    ]
    updated = batch_review(reviews, "approved", "Batch OK", auto_approve_checklist=True)
    assert len(updated) == 3
    assert all(r.status == ReviewStatus.APPROVED for r in updated)


def test_batch_reject():
    reviews = [
        create_review("2024.04.0", [f"EF:EPA:{i}"], "epa", "alice")
        for i in range(2)
    ]
    updated = batch_review(reviews, "rejected", "Source invalid")
    assert len(updated) == 2
    assert all(r.status == ReviewStatus.REJECTED for r in updated)


def test_batch_approve_without_auto_checklist_skips():
    reviews = [
        create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice"),
    ]
    updated = batch_review(reviews, "approved", "OK", auto_approve_checklist=False)
    assert len(updated) == 0  # Skipped because checklist incomplete


# ---- ReviewStatus / ReviewPriority enums ----

def test_review_statuses():
    assert len(ReviewStatus) == 5
    assert ReviewStatus.PENDING.value == "pending"
    assert ReviewStatus.APPROVED.value == "approved"


def test_review_priorities():
    assert len(ReviewPriority) == 4
    assert ReviewPriority.CRITICAL.value == "critical"


# ---- Edge cases ----

def test_checklist_fail_count():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    review.checklist[0].passed = True
    review.checklist[1].passed = False
    assert review.checklist_pass_count == 1
    assert review.checklist_fail_count == 1
    assert not review.all_passed


def test_empty_factor_ids():
    review = create_review("2024.04.0", [], "epa", "alice")
    assert len(review.factor_ids) == 0


def test_review_id_is_uuid():
    review = create_review("2024.04.0", ["EF:EPA:1"], "epa", "alice")
    import uuid
    uuid.UUID(review.review_id)  # Should not raise
