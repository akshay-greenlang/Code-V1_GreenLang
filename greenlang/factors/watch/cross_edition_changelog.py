# -*- coding: utf-8 -*-
"""Cross-edition changelog aggregation.

``changelog_draft.draft_changelog`` produces a pairwise diff between
two editions.  Customers and auditors routinely want the aggregate
"what changed between release X and release Z" across every
intermediate release.  This module stitches pairwise diffs together
into a single chronological report and, separately, a per-factor
history timeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from greenlang.factors.catalog_repository import FactorCatalogRepository
from greenlang.factors.watch.changelog_draft import draft_changelog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregate changelog across N editions
# ---------------------------------------------------------------------------


@dataclass
class CrossEditionSection:
    """One pairwise diff inside a cross-edition report."""

    left_edition_id: str
    right_edition_id: str
    lines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_edition_id": self.left_edition_id,
            "right_edition_id": self.right_edition_id,
            "lines": list(self.lines),
        }


@dataclass
class CrossEditionChangelog:
    """Full chronological report."""

    editions: List[str] = field(default_factory=list)
    sections: List[CrossEditionSection] = field(default_factory=list)

    def to_text(self) -> str:
        out: List[str] = []
        if self.editions:
            out.append("cross-edition changelog: " + " -> ".join(self.editions))
            out.append("")
        for section in self.sections:
            out.extend(section.lines)
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "editions": list(self.editions),
            "sections": [s.to_dict() for s in self.sections],
        }


def build_cross_edition_changelog(
    repo: FactorCatalogRepository,
    editions: Sequence[str],
) -> CrossEditionChangelog:
    """Build a single report covering pairwise diffs between every
    consecutive pair of editions in ``editions`` (order-preserving).

    Fewer than two editions → empty report.  Missing editions are
    surfaced as explicit error sections so the caller does not have
    to re-run the diff to understand why coverage stops.
    """
    if len(editions) < 2:
        return CrossEditionChangelog(editions=list(editions))

    report = CrossEditionChangelog(editions=list(editions))
    for left, right in zip(editions, editions[1:]):
        section = CrossEditionSection(left_edition_id=left, right_edition_id=right)
        try:
            section.lines = list(draft_changelog(left, right, repo))
        except Exception as exc:
            logger.warning(
                "cross-edition diff failed: %s -> %s: %s", left, right, exc
            )
            section.lines = [
                f"edition diff {left} -> {right}",
                f"error: {exc}",
            ]
        report.sections.append(section)
    return report


# ---------------------------------------------------------------------------
# Per-factor timeline across editions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorHistoryPoint:
    """One appearance of a factor inside an edition's manifest."""

    edition_id: str
    content_hash: str
    factor_status: str


@dataclass
class FactorHistory:
    """Chronological history for a single factor across editions."""

    factor_id: str
    points: List[FactorHistoryPoint] = field(default_factory=list)
    first_seen_in: Optional[str] = None
    last_seen_in: Optional[str] = None
    total_editions: int = 0
    unique_content_hashes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "first_seen_in": self.first_seen_in,
            "last_seen_in": self.last_seen_in,
            "total_editions": self.total_editions,
            "unique_content_hashes": self.unique_content_hashes,
            "points": [
                {
                    "edition_id": p.edition_id,
                    "content_hash": p.content_hash,
                    "factor_status": p.factor_status,
                }
                for p in self.points
            ],
        }


def summarize_factor_history(
    repo: FactorCatalogRepository,
    factor_id: str,
    editions: Sequence[str],
) -> FactorHistory:
    """Return the chronological appearance history of one factor.

    Walks ``editions`` in the given order (treat as oldest → newest)
    and records every edition that contains the factor along with its
    content hash and status.  The resulting object tells you how many
    revisions a factor has gone through and when it first appeared or
    disappeared.
    """
    history = FactorHistory(factor_id=factor_id)
    seen_hashes: set[str] = set()
    for edition_id in editions:
        try:
            summaries = repo.list_factor_summaries(edition_id)
        except Exception as exc:
            logger.debug(
                "history: cannot read edition %s: %s", edition_id, exc
            )
            continue
        # Build an index just for this edition.
        for s in summaries:
            if s.get("factor_id") != factor_id:
                continue
            point = FactorHistoryPoint(
                edition_id=edition_id,
                content_hash=str(s.get("content_hash", "")),
                factor_status=str(s.get("factor_status", "certified")),
            )
            history.points.append(point)
            if point.content_hash:
                seen_hashes.add(point.content_hash)
            if history.first_seen_in is None:
                history.first_seen_in = edition_id
            history.last_seen_in = edition_id
            break
    history.total_editions = len(history.points)
    history.unique_content_hashes = len(seen_hashes)
    return history


# ---------------------------------------------------------------------------
# Release notes helper
# ---------------------------------------------------------------------------


def collect_release_notes(
    repo: FactorCatalogRepository,
    editions: Sequence[str],
) -> List[Tuple[str, List[str]]]:
    """Return ``(edition_id, changelog_lines)`` for every edition that
    carries a changelog in its manifest.  Useful for the hosted docs
    page ("release notes") and for the status API.
    """
    out: List[Tuple[str, List[str]]] = []
    for edition_id in editions:
        try:
            lines = list(repo.get_changelog(edition_id))
        except Exception as exc:
            logger.debug(
                "release notes: cannot read changelog for %s: %s",
                edition_id, exc,
            )
            continue
        if lines:
            out.append((edition_id, lines))
    return out


__all__ = [
    "CrossEditionSection",
    "CrossEditionChangelog",
    "FactorHistoryPoint",
    "FactorHistory",
    "build_cross_edition_changelog",
    "summarize_factor_history",
    "collect_release_notes",
]
