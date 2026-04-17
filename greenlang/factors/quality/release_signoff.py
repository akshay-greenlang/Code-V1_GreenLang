# -*- coding: utf-8 -*-
"""
Release signoff workflow (F024).

Full 9-point release signoff checklist for promoting an edition to stable.
Each check is independently verifiable and produces a structured result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignoffItem:
    """A single signoff checklist item."""

    item_id: str
    label: str
    ok: bool
    detail: str
    severity: str = "required"  # "required" | "recommended"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "ok": self.ok,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass
class ReleaseSignoff:
    """Full release signoff result for an edition."""

    edition_id: str
    items: List[SignoffItem] = field(default_factory=list)
    approver: str = ""
    approved: bool = False
    approved_at: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edition_id": self.edition_id,
            "items": [i.to_dict() for i in self.items],
            "total_items": len(self.items),
            "passed_items": self.passed_count,
            "failed_items": self.failed_count,
            "all_required_passed": self.all_required_passed,
            "approver": self.approver,
            "approved": self.approved,
            "approved_at": self.approved_at,
            "notes": self.notes,
        }

    @property
    def passed_count(self) -> int:
        return sum(1 for i in self.items if i.ok)

    @property
    def failed_count(self) -> int:
        return sum(1 for i in self.items if not i.ok)

    @property
    def all_required_passed(self) -> bool:
        """True if all required (non-recommended) items passed."""
        return all(i.ok for i in self.items if i.severity == "required")

    @property
    def ready_for_release(self) -> bool:
        return self.all_required_passed and len(self.items) > 0


def release_signoff_checklist(
    edition_id: str,
    manifest: Dict[str, Any],
    *,
    qa_report: Optional[Dict[str, Any]] = None,
    dedup_report: Optional[Dict[str, Any]] = None,
    consistency_report: Optional[Dict[str, Any]] = None,
    changelog_reviewed: bool = False,
    methodology_signed: bool = False,
    legal_confirmed: bool = False,
    regression_passed: Optional[bool] = None,
    load_test_passed: Optional[bool] = None,
    gold_eval_precision: Optional[float] = None,
) -> ReleaseSignoff:
    """
    Build a full 9-point release signoff checklist.

    Args:
        edition_id: Edition being released.
        manifest: Edition manifest dict.
        qa_report: BatchQAReport.to_dict() result (optional).
        dedup_report: DedupReport.to_dict() result (optional).
        consistency_report: ConsistencyReport.to_dict() result (optional).
        changelog_reviewed: Whether changelog has been reviewed.
        methodology_signed: Whether methodology lead signed off.
        legal_confirmed: Whether legal confirmed licenses.
        regression_passed: Whether regression test (compare_editions) passed.
        load_test_passed: Whether load test (p95 < 500ms) passed.
        gold_eval_precision: Precision@1 from gold eval (>= 0.85 required).

    Returns:
        ReleaseSignoff with all 9 checklist items.
    """
    signoff = ReleaseSignoff(edition_id=edition_id)

    # S1: All Q1-Q6 gates pass
    if qa_report:
        all_passed = qa_report.get("total_failed", 0) == 0 and qa_report.get("total_factors", 0) > 0
        total = qa_report.get("total_factors", 0)
        passed = qa_report.get("total_passed", 0)
        signoff.items.append(SignoffItem(
            item_id="S1",
            label="QA gates pass for all factors",
            ok=all_passed,
            detail=f"{passed}/{total} factors passed Q1-Q6 gates",
        ))
    else:
        signoff.items.append(SignoffItem(
            item_id="S1",
            label="QA gates pass for all factors",
            ok=False,
            detail="QA report not provided",
        ))

    # S2: No unresolved duplicate pairs
    if dedup_report:
        human_review = dedup_report.get("human_review", 0)
        signoff.items.append(SignoffItem(
            item_id="S2",
            label="No unresolved duplicate pairs",
            ok=human_review == 0,
            detail=f"{human_review} duplicate pairs need human review" if human_review else "No duplicates requiring review",
        ))
    else:
        signoff.items.append(SignoffItem(
            item_id="S2",
            label="No unresolved duplicate pairs",
            ok=False,
            detail="Dedup report not provided",
        ))

    # S3: Cross-source consistency reviewed
    if consistency_report:
        reviews_needed = consistency_report.get("total_reviews", 0)
        signoff.items.append(SignoffItem(
            item_id="S3",
            label="Cross-source consistency reviewed",
            ok=reviews_needed == 0,
            detail=f"{reviews_needed} activities need consistency review" if reviews_needed else "All cross-source checks passed",
        ))
    else:
        signoff.items.append(SignoffItem(
            item_id="S3",
            label="Cross-source consistency reviewed",
            ok=False,
            detail="Consistency report not provided",
        ))

    # S4: Changelog reviewed and approved
    has_changelog = bool(manifest.get("changelog"))
    signoff.items.append(SignoffItem(
        item_id="S4",
        label="Changelog reviewed and approved",
        ok=has_changelog and changelog_reviewed,
        detail="Changelog reviewed" if (has_changelog and changelog_reviewed) else "Changelog not reviewed or empty",
    ))

    # S5: Methodology lead signed off
    signoff.items.append(SignoffItem(
        item_id="S5",
        label="Methodology lead signed off",
        ok=methodology_signed,
        detail="Methodology lead approved" if methodology_signed else "Awaiting methodology lead approval",
    ))

    # S6: Legal confirmed licenses for all new sources
    signoff.items.append(SignoffItem(
        item_id="S6",
        label="Legal confirmed source licenses",
        ok=legal_confirmed,
        detail="Legal review complete" if legal_confirmed else "Awaiting legal confirmation",
    ))

    # S7: Regression test passed (compare_editions)
    if regression_passed is not None:
        signoff.items.append(SignoffItem(
            item_id="S7",
            label="Regression test passed",
            ok=regression_passed,
            detail="Edition comparison shows expected changes" if regression_passed else "Regression test failed",
        ))
    else:
        signoff.items.append(SignoffItem(
            item_id="S7",
            label="Regression test passed",
            ok=False,
            detail="Regression test not run",
            severity="recommended",
        ))

    # S8: Load test passed (p95 < 500ms)
    if load_test_passed is not None:
        signoff.items.append(SignoffItem(
            item_id="S8",
            label="Load test passed (p95 < 500ms)",
            ok=load_test_passed,
            detail="Load test p95 within threshold" if load_test_passed else "Load test failed",
        ))
    else:
        signoff.items.append(SignoffItem(
            item_id="S8",
            label="Load test passed (p95 < 500ms)",
            ok=False,
            detail="Load test not run",
            severity="recommended",
        ))

    # S9: Gold eval precision >= 0.85
    if gold_eval_precision is not None:
        passed = gold_eval_precision >= 0.85
        signoff.items.append(SignoffItem(
            item_id="S9",
            label="Gold eval precision >= 0.85",
            ok=passed,
            detail=f"Precision@1 = {gold_eval_precision:.3f}" + (" (above threshold)" if passed else " (below threshold)"),
        ))
    else:
        signoff.items.append(SignoffItem(
            item_id="S9",
            label="Gold eval precision >= 0.85",
            ok=False,
            detail="Gold eval not run",
            severity="recommended",
        ))

    logger.info(
        "Release signoff: edition=%s passed=%d/%d all_required=%s",
        edition_id, signoff.passed_count, len(signoff.items), signoff.all_required_passed,
    )
    return signoff


def approve_release(
    signoff: ReleaseSignoff,
    approver: str,
    *,
    force: bool = False,
    notes: str = "",
) -> ReleaseSignoff:
    """
    Record release approval.

    Args:
        signoff: The signoff to approve.
        approver: Email/ID of the approver.
        force: If True, allow approval even if not all required items pass.
        notes: Approval notes.

    Raises:
        ValueError: If not ready for release and force is False.
    """
    if not signoff.ready_for_release and not force:
        failed = [i.item_id for i in signoff.items if not i.ok and i.severity == "required"]
        raise ValueError(f"Cannot approve: required items failed: {failed}")

    signoff.approved = True
    signoff.approver = approver
    signoff.approved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    signoff.notes = notes

    logger.info(
        "Release approved: edition=%s approver=%s force=%s",
        signoff.edition_id, approver, force,
    )
    return signoff
