# -*- coding: utf-8 -*-
"""
Change detection pipeline (F051).

Processes watch results that flagged a content change:
1. Download the new artifact from the source URL.
2. Run the appropriate parser to extract factor dicts.
3. Compare extracted factors against the current edition.
4. Classify changes (numeric, metadata, policy, parser-break).
5. Generate a structured ChangeReport.
6. Optionally trigger notifications (methodology lead for policy changes).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.factors.watch.change_classification import classify_change

logger = logging.getLogger(__name__)


@dataclass
class ChangedFactor:
    """A single factor that differs between old and new artifact."""

    factor_id: str
    change_kind: str  # "added", "removed", "modified"
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    field_changes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeReport:
    """Structured report of all changes detected in a source update."""

    source_id: str
    timestamp: str
    edition_id: Optional[str] = None
    change_type: str = "content_changed"
    artifact_hash_old: Optional[str] = None
    artifact_hash_new: Optional[str] = None
    before_count: int = 0
    after_count: int = 0
    added: List[ChangedFactor] = field(default_factory=list)
    removed: List[ChangedFactor] = field(default_factory=list)
    modified: List[ChangedFactor] = field(default_factory=list)
    requires_human_review: bool = False
    review_reason: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.removed) + len(self.modified)

    @property
    def has_breaking_changes(self) -> bool:
        return len(self.removed) > 0 or any(
            f.field_changes.get("unit_changed") for f in self.modified
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "edition_id": self.edition_id,
            "change_type": self.change_type,
            "artifact_hash_old": self.artifact_hash_old,
            "artifact_hash_new": self.artifact_hash_new,
            "before_count": self.before_count,
            "after_count": self.after_count,
            "added_count": len(self.added),
            "removed_count": len(self.removed),
            "modified_count": len(self.modified),
            "total_changes": self.total_changes,
            "has_breaking_changes": self.has_breaking_changes,
            "requires_human_review": self.requires_human_review,
            "review_reason": self.review_reason,
            "errors": self.errors,
        }


def _extract_factor_map(factors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index factors by factor_id for fast comparison."""
    result: Dict[str, Dict[str, Any]] = {}
    for f in factors:
        fid = f.get("factor_id") or f.get("id") or ""
        if fid:
            result[fid] = f
    return result


def _compute_factor_hash(factor: Dict[str, Any]) -> str:
    """Deterministic hash of a factor's numeric + structural fields."""
    key_fields = sorted(
        (k, str(v)) for k, v in factor.items()
        if k not in ("_raw", "metadata", "last_updated", "ingested_at")
    )
    raw = "|".join(f"{k}={v}" for k, v in key_fields)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect_source_change(
    source_id: str,
    old_factors: List[Dict[str, Any]],
    new_factors: List[Dict[str, Any]],
    *,
    edition_id: Optional[str] = None,
    artifact_hash_old: Optional[str] = None,
    artifact_hash_new: Optional[str] = None,
) -> ChangeReport:
    """
    Compare old vs new factor lists from a source and produce a ChangeReport.

    Args:
        source_id: Source identifier.
        old_factors: Factors currently in catalog for this source.
        new_factors: Factors parsed from the new artifact.
        edition_id: Current edition for context.
        artifact_hash_old: SHA-256 of the previous raw artifact.
        artifact_hash_new: SHA-256 of the new raw artifact.

    Returns:
        ChangeReport with added/removed/modified factors.
    """
    report = ChangeReport(
        source_id=source_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        edition_id=edition_id,
        artifact_hash_old=artifact_hash_old,
        artifact_hash_new=artifact_hash_new,
        before_count=len(old_factors),
        after_count=len(new_factors),
    )

    old_map = _extract_factor_map(old_factors)
    new_map = _extract_factor_map(new_factors)

    old_ids = set(old_map.keys())
    new_ids = set(new_map.keys())

    # Added factors
    for fid in sorted(new_ids - old_ids):
        nf = new_map[fid]
        report.added.append(ChangedFactor(
            factor_id=fid,
            change_kind="added",
            new_value=nf.get("co2e_total"),
        ))

    # Removed factors
    for fid in sorted(old_ids - new_ids):
        of = old_map[fid]
        report.removed.append(ChangedFactor(
            factor_id=fid,
            change_kind="removed",
            old_value=of.get("co2e_total"),
        ))

    # Modified factors
    for fid in sorted(old_ids & new_ids):
        of = old_map[fid]
        nf = new_map[fid]
        old_hash = _compute_factor_hash(of)
        new_hash = _compute_factor_hash(nf)
        if old_hash != new_hash:
            field_changes: Dict[str, Any] = {}
            # Check key field differences
            for key in ("co2e_total", "unit", "scope", "geography", "fuel_type"):
                ov = of.get(key)
                nv = nf.get(key)
                if ov != nv:
                    field_changes[key] = {"old": ov, "new": nv}
                    if key == "unit":
                        field_changes["unit_changed"] = True

            change_class = classify_change(
                old_hash=old_hash,
                new_hash=new_hash,
                old_row=of,
                new_row=nf,
            )
            report.modified.append(ChangedFactor(
                factor_id=fid,
                change_kind="modified",
                old_value=of.get("co2e_total"),
                new_value=nf.get("co2e_total"),
                field_changes=field_changes,
            ))

    # Determine if human review is needed
    if report.has_breaking_changes:
        report.requires_human_review = True
        report.review_reason = "Breaking changes detected (factors removed or units changed)"
    elif len(report.removed) > len(old_factors) * 0.1:
        report.requires_human_review = True
        report.review_reason = f"More than 10% of factors removed ({len(report.removed)}/{len(old_factors)})"
    elif report.total_changes > len(old_factors) * 0.5:
        report.requires_human_review = True
        report.review_reason = f"More than 50% of factors changed ({report.total_changes}/{len(old_factors)})"

    # Classify overall change
    if report.total_changes == 0:
        report.change_type = "no_change"
    elif report.has_breaking_changes:
        report.change_type = "breaking_change"
    elif report.added and not report.removed and not report.modified:
        report.change_type = "additions_only"
    else:
        report.change_type = "content_changed"

    logger.info(
        "Change detection for %s: added=%d removed=%d modified=%d type=%s review=%s",
        source_id, len(report.added), len(report.removed), len(report.modified),
        report.change_type, report.requires_human_review,
    )
    return report
