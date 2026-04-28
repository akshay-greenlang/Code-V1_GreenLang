# -*- coding: utf-8 -*-
"""Deterministic JSON + Markdown diff export for methodology-lead review.

The Phase 3 plan §"Dedupe / supersede / diff rules" makes the staging diff
the SOLE artefact a methodology lead reads to approve a publish. This
module wraps :class:`~greenlang.factors.release.alpha_publisher.StagingDiff`
with a richer change-record shape (per-attribute deltas, parser-version
shifts, licence/methodology callouts) and serialises it to byte-identical
JSON / Markdown so two runs of the same pipeline produce a stable diff.

Determinism contract
--------------------
Every list inside :class:`RunDiff` is sorted by URN before serialisation.
``serialize_json`` writes with ``sort_keys=True`` and a fixed key set.
``serialize_markdown`` emits sections in a fixed order (added, removed,
changed, supersedes, parser-version-changes, licence-changes,
methodology-changes, summary). A second call against the same diff MUST
produce a byte-identical string — this property is asserted by the
Phase 3 snapshot tests (task #31).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


__all__ = [
    "ChangeRecord",
    "RunDiff",
    "serialize_json",
    "serialize_markdown",
    "from_staging_diff",
]


# ---------------------------------------------------------------------------
# ChangeRecord + RunDiff dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeRecord:
    """One per-attribute change between a production and a staging record.

    The Phase 3 plan §"Dedupe / supersede / diff rules" lists the exact
    set of attributes a reviewer cares about: value, unit, boundary,
    geography, methodology, licence, citation, parser-version. The
    runner emits one ``ChangeRecord`` per (urn, attribute) pair so the
    Markdown table can show one row per change rather than one row per
    record.

    Note
    ----
    ``frozen=True`` makes ChangeRecord hashable so a runner can dedupe
    duplicate (urn, attribute) emissions without an explicit set comp.
    """

    urn: str
    attribute: str
    old_value: Optional[str]
    new_value: Optional[str]


@dataclass
class RunDiff:
    """Comprehensive run-level diff used as the publish-approval artefact.

    Phase 3 plan §"Dedupe / supersede / diff rules" requires the diff
    surface five counter buckets (added / removed / changed / unchanged /
    superseded) PLUS first-class call-outs for parser-version, licence,
    and methodology changes — these three are the highest-risk surfaces
    a methodology lead reviews and they get their own sections in the
    Markdown output.
    """

    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    changed: List[ChangeRecord] = field(default_factory=list)
    supersedes: List[Tuple[str, str]] = field(default_factory=list)
    unchanged_count: int = 0
    parser_version_changes: List[ChangeRecord] = field(default_factory=list)
    licence_changes: List[ChangeRecord] = field(default_factory=list)
    methodology_changes: List[ChangeRecord] = field(default_factory=list)
    source_urn: Optional[str] = None
    source_version: Optional[str] = None
    run_id: Optional[str] = None

    def is_empty(self) -> bool:
        """Return True when every diff bucket is empty.

        Reviewers see this from the CLI summary; an empty diff after a
        steady-state period almost always indicates a parser regression.
        """
        return not (
            self.added
            or self.removed
            or self.changed
            or self.supersedes
            or self.parser_version_changes
            or self.licence_changes
            or self.methodology_changes
        )

    def total_changes(self) -> int:
        """Sum across every change bucket — used in the Markdown summary."""
        return (
            len(self.added)
            + len(self.removed)
            + len(self.changed)
            + len(self.supersedes)
            + len(self.parser_version_changes)
            + len(self.licence_changes)
            + len(self.methodology_changes)
        )


# ---------------------------------------------------------------------------
# Conversion from the legacy StagingDiff
# ---------------------------------------------------------------------------


def from_staging_diff(
    staging_diff: Any,
    *,
    run_id: Optional[str] = None,
    source_urn: Optional[str] = None,
    source_version: Optional[str] = None,
    production_records: Optional[Dict[str, Dict[str, Any]]] = None,
    staging_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> RunDiff:
    """Lift a legacy :class:`StagingDiff` into a Phase 3 :class:`RunDiff`.

    The Phase 2 ``StagingDiff`` carries additions / removals / changes /
    unchanged but does not break out per-attribute deltas. When the runner
    has access to both side records (``production_records`` and
    ``staging_records``, both keyed by URN), this helper computes the
    per-attribute :class:`ChangeRecord` list for the high-risk fields
    (value, unit, boundary, geography_urn, methodology, licence,
    parser_version, citation).

    When ``production_records`` and ``staging_records`` are ``None`` the
    return value is structurally complete but the per-attribute change
    list is empty — the runner falls back to the URN-only view.
    """
    diff = RunDiff(
        run_id=run_id,
        source_urn=source_urn,
        source_version=source_version,
    )

    # -- additions / removals / supersedes ---------------------------------
    additions = list(getattr(staging_diff, "additions", []) or [])
    removals = list(getattr(staging_diff, "removals", []) or [])
    changes = list(getattr(staging_diff, "changes", []) or [])
    unchanged = int(getattr(staging_diff, "unchanged", 0) or 0)

    diff.added = sorted(
        rec.get("urn", "") for rec in additions if isinstance(rec, dict)
    )
    diff.removed = sorted(str(u) for u in removals)
    diff.supersedes = sorted(
        (str(old), str(new)) for old, new in changes
    )
    diff.unchanged_count = unchanged

    # -- per-attribute deltas (only for supersede pairs we have records for)
    if production_records and staging_records:
        attribute_targets = (
            "value",
            "unit",
            "boundary",
            "geography_urn",
            "methodology",
            "licence",
            "parser_version",
            "citation",
        )
        for old_urn, new_urn in diff.supersedes:
            old_rec = production_records.get(old_urn) or {}
            new_rec = staging_records.get(new_urn) or {}
            for attr in attribute_targets:
                old_val = _flatten(old_rec, attr)
                new_val = _flatten(new_rec, attr)
                if old_val == new_val:
                    continue
                cr = ChangeRecord(
                    urn=new_urn,
                    attribute=attr,
                    old_value=_to_text(old_val),
                    new_value=_to_text(new_val),
                )
                diff.changed.append(cr)
                if attr == "parser_version":
                    diff.parser_version_changes.append(cr)
                elif attr == "licence":
                    diff.licence_changes.append(cr)
                elif attr == "methodology":
                    diff.methodology_changes.append(cr)

        diff.changed.sort(key=lambda c: (c.urn, c.attribute))
        diff.parser_version_changes.sort(key=lambda c: c.urn)
        diff.licence_changes.sort(key=lambda c: c.urn)
        diff.methodology_changes.sort(key=lambda c: c.urn)

    return diff


def _flatten(rec: Dict[str, Any], attr: str) -> Any:
    """Return ``rec[attr]`` with simple dotted-path lookup support.

    ``parser_version`` lives under ``extraction.parser_version`` in the
    factor record schema; this helper keeps the caller terse.
    """
    if attr == "parser_version":
        ext = rec.get("extraction") or {}
        if isinstance(ext, dict):
            return ext.get("parser_version")
        return None
    if attr == "citation":
        ext = rec.get("extraction") or {}
        if isinstance(ext, dict):
            return ext.get("source_publication") or rec.get("citation")
        return rec.get("citation")
    if attr == "methodology":
        return rec.get("methodology") or rec.get("methodology_urn")
    return rec.get(attr)


def _to_text(value: Any) -> Optional[str]:
    """Convert a record value to a stable text form for the diff output."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def serialize_json(diff: RunDiff) -> Dict[str, Any]:
    """Serialise a :class:`RunDiff` to a deterministic JSON-able dict.

    Lists are sorted (by URN, then by attribute for changes) BEFORE the
    dict is returned. The caller is expected to pass the result to
    ``json.dumps(..., sort_keys=True, indent=2)`` to get a canonical
    on-disk representation; the snapshot tests use exactly that.
    """
    return {
        "run_id": diff.run_id,
        "source_urn": diff.source_urn,
        "source_version": diff.source_version,
        "summary": {
            "added": len(diff.added),
            "removed": len(diff.removed),
            "changed": len(diff.changed),
            "supersedes": len(diff.supersedes),
            "unchanged": diff.unchanged_count,
            "parser_version_changes": len(diff.parser_version_changes),
            "licence_changes": len(diff.licence_changes),
            "methodology_changes": len(diff.methodology_changes),
            "total_changes": diff.total_changes(),
        },
        "added": list(diff.added),
        "removed": list(diff.removed),
        "changed": [
            {
                "urn": c.urn,
                "attribute": c.attribute,
                "old_value": c.old_value,
                "new_value": c.new_value,
            }
            for c in diff.changed
        ],
        "supersedes": [
            {"old_urn": old, "new_urn": new} for old, new in diff.supersedes
        ],
        "parser_version_changes": [
            {
                "urn": c.urn,
                "old_value": c.old_value,
                "new_value": c.new_value,
            }
            for c in diff.parser_version_changes
        ],
        "licence_changes": [
            {
                "urn": c.urn,
                "old_value": c.old_value,
                "new_value": c.new_value,
            }
            for c in diff.licence_changes
        ],
        "methodology_changes": [
            {
                "urn": c.urn,
                "old_value": c.old_value,
                "new_value": c.new_value,
            }
            for c in diff.methodology_changes
        ],
    }


def serialize_markdown(diff: RunDiff) -> str:
    """Serialise a :class:`RunDiff` to the methodology-lead review document.

    Emits sections in a fixed order so two runs against the same diff
    produce byte-identical output (Phase 3 snapshot test contract).

    Section order
    -------------
    1. Heading + summary table.
    2. Added URNs (count + first 50).
    3. Removed URNs (count + ALL — removals are the highest-risk delta).
    4. Supersede pairs.
    5. Per-attribute changes table.
    6. Parser-version changes (callout).
    7. Licence changes (callout).
    8. Methodology changes (callout).

    A reviewer who scans top-to-bottom sees the highest-impact deltas
    (removals, methodology, licence) without scrolling to find them.
    """
    lines: List[str] = []
    lines.append("# Ingestion Run Diff")
    lines.append("")
    lines.append("- run_id: `%s`" % (diff.run_id or "(unset)"))
    lines.append("- source_urn: `%s`" % (diff.source_urn or "(unset)"))
    lines.append("- source_version: `%s`" % (diff.source_version or "(unset)"))
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|---|---|")
    lines.append("| Added | %d |" % len(diff.added))
    lines.append("| Removed | %d |" % len(diff.removed))
    lines.append("| Changed (per-attribute) | %d |" % len(diff.changed))
    lines.append("| Supersedes | %d |" % len(diff.supersedes))
    lines.append("| Unchanged | %d |" % diff.unchanged_count)
    lines.append("| Parser-version changes | %d |" % len(diff.parser_version_changes))
    lines.append("| Licence changes | %d |" % len(diff.licence_changes))
    lines.append("| Methodology changes | %d |" % len(diff.methodology_changes))
    lines.append("")

    # 2. Added.
    lines.append("## Added (%d)" % len(diff.added))
    lines.append("")
    if diff.added:
        for urn in diff.added[:50]:
            lines.append("- `%s`" % urn)
        if len(diff.added) > 50:
            lines.append("- ... %d more not shown" % (len(diff.added) - 50))
    else:
        lines.append("_(none)_")
    lines.append("")

    # 3. Removed — show ALL because removals are the largest review risk.
    lines.append("## Removed (%d)" % len(diff.removed))
    lines.append("")
    if diff.removed:
        for urn in diff.removed:
            lines.append("- `%s`" % urn)
    else:
        lines.append("_(none)_")
    lines.append("")

    # 4. Supersedes.
    lines.append("## Supersedes (%d)" % len(diff.supersedes))
    lines.append("")
    if diff.supersedes:
        lines.append("| Old URN | New URN |")
        lines.append("|---|---|")
        for old, new in diff.supersedes:
            lines.append("| `%s` | `%s` |" % (old, new))
    else:
        lines.append("_(none)_")
    lines.append("")

    # 5. Per-attribute changes.
    lines.append("## Changed attributes (%d)" % len(diff.changed))
    lines.append("")
    if diff.changed:
        lines.append("| URN | Attribute | Old | New |")
        lines.append("|---|---|---|---|")
        for cr in diff.changed:
            lines.append(
                "| `%s` | %s | %s | %s |"
                % (
                    cr.urn,
                    cr.attribute,
                    _md_cell(cr.old_value),
                    _md_cell(cr.new_value),
                )
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    # 6. Parser-version changes.
    lines.append("## Parser-version changes (%d)" % len(diff.parser_version_changes))
    lines.append("")
    if diff.parser_version_changes:
        lines.append("| URN | Old | New |")
        lines.append("|---|---|---|")
        for cr in diff.parser_version_changes:
            lines.append(
                "| `%s` | %s | %s |"
                % (cr.urn, _md_cell(cr.old_value), _md_cell(cr.new_value))
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    # 7. Licence changes.
    lines.append("## Licence changes (%d)" % len(diff.licence_changes))
    lines.append("")
    if diff.licence_changes:
        lines.append("| URN | Old | New |")
        lines.append("|---|---|---|")
        for cr in diff.licence_changes:
            lines.append(
                "| `%s` | %s | %s |"
                % (cr.urn, _md_cell(cr.old_value), _md_cell(cr.new_value))
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    # 8. Methodology changes.
    lines.append("## Methodology changes (%d)" % len(diff.methodology_changes))
    lines.append("")
    if diff.methodology_changes:
        lines.append("| URN | Old | New |")
        lines.append("|---|---|---|")
        for cr in diff.methodology_changes:
            lines.append(
                "| `%s` | %s | %s |"
                % (cr.urn, _md_cell(cr.old_value), _md_cell(cr.new_value))
            )
    else:
        lines.append("_(none)_")
    lines.append("")

    return "\n".join(lines)


def _md_cell(value: Optional[str]) -> str:
    """Render a value as a Markdown table cell, escaping pipes and newlines."""
    if value is None:
        return "_null_"
    text = str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > 80:
        text = text[:77] + "..."
    if text == "":
        return "_empty_"
    return "`%s`" % text
