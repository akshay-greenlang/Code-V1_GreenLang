# -*- coding: utf-8 -*-
"""
Methodology review workflow (F023).

Extends the basic Q3 review queue with structured review assignments,
10-point methodology checklist tracking, batch review decisions, and
review lifecycle management.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class ReviewPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 10-point methodology review checklist
METHODOLOGY_CHECKLIST = [
    {"id": "C01", "label": "Source provenance verified", "description": "Source URL, publication, and version confirmed"},
    {"id": "C02", "label": "GWP basis correct", "description": "AR5/AR6 GWP values match source documentation"},
    {"id": "C03", "label": "Gas vectors complete", "description": "CO2, CH4, N2O present with correct units"},
    {"id": "C04", "label": "Unit conversion verified", "description": "Activity unit matches source; conversion factors correct"},
    {"id": "C05", "label": "Geography mapping accurate", "description": "ISO-3166 code correct for source geography"},
    {"id": "C06", "label": "Scope/boundary alignment", "description": "Scope 1/2/3 and boundary match GHG Protocol definitions"},
    {"id": "C07", "label": "Temporal validity window", "description": "valid_from/valid_to dates match source publication period"},
    {"id": "C08", "label": "DQS scores justified", "description": "Data quality scores reflect actual data provenance"},
    {"id": "C09", "label": "License compliance", "description": "Factor distribution rights match source license terms"},
    {"id": "C10", "label": "Cross-source plausibility", "description": "Values within expected range compared to similar sources"},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ChecklistItem:
    """A single checklist item with pass/fail status."""

    item_id: str
    label: str
    passed: Optional[bool] = None  # None = not yet reviewed
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "passed": self.passed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChecklistItem":
        return cls(
            item_id=data["item_id"],
            label=data.get("label", ""),
            passed=data.get("passed"),
            notes=data.get("notes", ""),
        )


@dataclass
class ReviewAssignment:
    """A methodology review assignment for a set of factors."""

    review_id: str
    edition_id: str
    factor_ids: List[str]
    source_id: str
    reviewer: str
    status: ReviewStatus = ReviewStatus.PENDING
    priority: ReviewPriority = ReviewPriority.MEDIUM
    checklist: List[ChecklistItem] = field(default_factory=list)
    decision: Optional[str] = None  # approved | rejected | needs_revision
    decision_notes: str = ""
    created_at: str = field(default_factory=_utc_now)
    due_date: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.checklist:
            self.checklist = [
                ChecklistItem(item_id=c["id"], label=c["label"])
                for c in METHODOLOGY_CHECKLIST
            ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "edition_id": self.edition_id,
            "factor_ids": self.factor_ids,
            "source_id": self.source_id,
            "reviewer": self.reviewer,
            "status": self.status.value,
            "priority": self.priority.value,
            "checklist": [c.to_dict() for c in self.checklist],
            "decision": self.decision,
            "decision_notes": self.decision_notes,
            "created_at": self.created_at,
            "due_date": self.due_date,
            "completed_at": self.completed_at,
            "checklist_complete": self.checklist_complete,
            "checklist_pass_count": self.checklist_pass_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewAssignment":
        checklist = [ChecklistItem.from_dict(c) for c in data.get("checklist", [])]
        return cls(
            review_id=data["review_id"],
            edition_id=data["edition_id"],
            factor_ids=data.get("factor_ids", []),
            source_id=data.get("source_id", ""),
            reviewer=data.get("reviewer", ""),
            status=ReviewStatus(data.get("status", "pending")),
            priority=ReviewPriority(data.get("priority", "medium")),
            checklist=checklist,
            decision=data.get("decision"),
            decision_notes=data.get("decision_notes", ""),
            created_at=data.get("created_at", ""),
            due_date=data.get("due_date"),
            completed_at=data.get("completed_at"),
        )

    @property
    def checklist_complete(self) -> bool:
        """True if all checklist items have been reviewed (passed is not None)."""
        return all(c.passed is not None for c in self.checklist)

    @property
    def checklist_pass_count(self) -> int:
        return sum(1 for c in self.checklist if c.passed is True)

    @property
    def checklist_fail_count(self) -> int:
        return sum(1 for c in self.checklist if c.passed is False)

    @property
    def all_passed(self) -> bool:
        return self.checklist_complete and self.checklist_fail_count == 0


def create_review(
    edition_id: str,
    factor_ids: List[str],
    source_id: str,
    reviewer: str,
    *,
    priority: ReviewPriority = ReviewPriority.MEDIUM,
    due_date: Optional[str] = None,
) -> ReviewAssignment:
    """Create a new methodology review assignment."""
    review = ReviewAssignment(
        review_id=str(uuid.uuid4()),
        edition_id=edition_id,
        factor_ids=factor_ids,
        source_id=source_id,
        reviewer=reviewer,
        priority=priority,
        due_date=due_date,
    )
    logger.info(
        "Created review %s: edition=%s source=%s factors=%d reviewer=%s",
        review.review_id, edition_id, source_id, len(factor_ids), reviewer,
    )
    return review


def update_checklist_item(
    review: ReviewAssignment,
    item_id: str,
    passed: bool,
    notes: str = "",
) -> bool:
    """
    Update a checklist item on a review.

    Returns True if the item was found and updated.
    """
    for item in review.checklist:
        if item.item_id == item_id:
            item.passed = passed
            if notes:
                item.notes = notes
            logger.debug("Updated checklist item %s on review %s: passed=%s", item_id, review.review_id, passed)
            return True
    return False


def submit_decision(
    review: ReviewAssignment,
    decision: str,
    notes: str = "",
) -> ReviewAssignment:
    """
    Submit a review decision (approved/rejected/needs_revision).

    Raises ValueError if checklist is incomplete.
    """
    if decision not in ("approved", "rejected", "needs_revision"):
        raise ValueError(f"Invalid decision: {decision!r}")

    if decision == "approved" and not review.checklist_complete:
        raise ValueError("Cannot approve: checklist is incomplete")

    review.decision = decision
    review.decision_notes = notes
    review.completed_at = _utc_now()

    if decision == "approved":
        review.status = ReviewStatus.APPROVED
    elif decision == "rejected":
        review.status = ReviewStatus.REJECTED
    else:
        review.status = ReviewStatus.NEEDS_REVISION

    logger.info(
        "Review %s decision: %s (%d/%d checklist passed)",
        review.review_id, decision, review.checklist_pass_count, len(review.checklist),
    )
    return review


def batch_review(
    reviews: Sequence[ReviewAssignment],
    decision: str,
    notes: str = "",
    *,
    auto_approve_checklist: bool = False,
) -> List[ReviewAssignment]:
    """
    Apply the same decision to multiple reviews.

    Args:
        reviews: List of ReviewAssignment objects.
        decision: Decision to apply.
        notes: Notes for the decision.
        auto_approve_checklist: If True, mark all checklist items as passed before approving.

    Returns:
        List of updated reviews (skipping any that raise ValueError).
    """
    updated = []
    for review in reviews:
        if auto_approve_checklist and decision == "approved":
            for item in review.checklist:
                if item.passed is None:
                    item.passed = True
        try:
            submit_decision(review, decision, notes)
            updated.append(review)
        except ValueError as exc:
            logger.warning("Batch review skipped %s: %s", review.review_id, exc)
    logger.info("Batch review: %d/%d reviews updated with decision=%s", len(updated), len(reviews), decision)
    return updated


def save_review_to_sqlite(
    conn: sqlite3.Connection,
    review: ReviewAssignment,
) -> None:
    """Persist a review assignment to the qa_reviews table."""
    payload = review.to_dict()
    for fid in review.factor_ids:
        conn.execute(
            """
            INSERT INTO qa_reviews (review_id, edition_id, factor_id, status, payload_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(review_id) DO UPDATE SET
                status=excluded.status,
                payload_json=excluded.payload_json
            """,
            (review.review_id, review.edition_id, fid, review.status.value,
             json.dumps(payload, sort_keys=True, default=str)),
        )


def load_reviews_from_sqlite(
    conn: sqlite3.Connection,
    edition_id: str,
    *,
    status: Optional[str] = None,
    reviewer: Optional[str] = None,
) -> List[ReviewAssignment]:
    """Load review assignments from qa_reviews table."""
    clauses = ["edition_id = ?"]
    params: List[Any] = [edition_id]
    if status:
        clauses.append("status = ?")
        params.append(status)

    where = " AND ".join(clauses)
    cur = conn.execute(
        f"SELECT DISTINCT review_id, payload_json FROM qa_reviews WHERE {where} ORDER BY review_id",
        params,
    )
    reviews = []
    seen: set = set()
    for row in cur.fetchall():
        rid = row[0] if isinstance(row, (list, tuple)) else row["review_id"]
        if rid in seen:
            continue
        seen.add(rid)
        payload_str = row[1] if isinstance(row, (list, tuple)) else row["payload_json"]
        data = json.loads(payload_str)
        review = ReviewAssignment.from_dict(data)
        if reviewer and review.reviewer != reviewer:
            continue
        reviews.append(review)
    return reviews
