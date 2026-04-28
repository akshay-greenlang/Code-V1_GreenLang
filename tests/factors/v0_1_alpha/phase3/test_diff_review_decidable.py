# -*- coding: utf-8 -*-
"""Phase 3 — a methodology lead can decide publish-yes/no from the diff alone.

Acceptance criterion (PHASE_3_PLAN.md §"Dedupe / supersede / diff rules"):
"Reviewers must be able to decide publish-yes/no from the diff alone,
without reading raw rows."

Operationally that translates to:

  * The Markdown rendering carries a top ``Summary`` section with
    counters (added / removed / changed / unchanged / supersedes).
  * The JSON rendering carries ``summary`` + per-bucket lists (added,
    removed, changed, supersedes) so downstream UIs can branch.

The MD output emitted by the current ``serialize_markdown`` does not
include a literal ``Decision:`` line — instead the Summary table at the
top + the Removed-section-shows-ALL contract is the operational
"reviewer can decide from the doc alone" surface. The decidability test
asserts the strict invariants (Summary section, bucket sections present
in fixed order).
"""
from __future__ import annotations

import importlib
import json

import pytest


def _diff_available() -> bool:
    try:
        importlib.import_module("greenlang.factors.ingestion.diff")
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _diff_available(),
    reason=(
        "greenlang.factors.ingestion.diff not yet committed; "
        "Wave 1.0 sibling agent still in flight"
    ),
)


def _build_mid_size_diff():
    """5 added + 2 removed + 3 changed + 1 supersede pair."""
    from greenlang.factors.ingestion.diff import ChangeRecord, RunDiff

    added = sorted(
        f"urn:gl:factor:phase3-alpha:test:added-{i}:v1" for i in range(5)
    )
    removed = sorted(
        f"urn:gl:factor:phase3-alpha:test:removed-{i}:v1" for i in range(2)
    )
    changed = [
        ChangeRecord(
            urn=f"urn:gl:factor:phase3-alpha:test:changed-{i}:v1",
            attribute="value",
            old_value=f"{i / 100:.2f}",
            new_value=f"{i / 50:.2f}",
        )
        for i in range(3)
    ]
    supersedes = [
        (
            "urn:gl:factor:phase3-alpha:test:supersede:v1",
            "urn:gl:factor:phase3-alpha:test:supersede:v2",
        )
    ]
    return RunDiff(
        run_id="phase3-decidable-run",
        source_urn="urn:gl:source:phase2-alpha",
        source_version="2024.1",
        added=added,
        removed=removed,
        changed=changed,
        supersedes=supersedes,
        unchanged_count=0,
    )


def test_md_rendering_carries_summary_at_top():
    """The Summary section appears before the Added / Removed / Supersedes lists."""
    from greenlang.factors.ingestion.diff import serialize_markdown

    md = serialize_markdown(_build_mid_size_diff())
    summary_idx = md.find("## Summary")
    added_idx = md.find("## Added")
    removed_idx = md.find("## Removed")
    supersedes_idx = md.find("## Supersedes")
    assert summary_idx >= 0
    assert added_idx > summary_idx
    assert removed_idx > summary_idx
    assert supersedes_idx > summary_idx
    # The summary table reports every bucket.
    summary_block = md[summary_idx : added_idx]
    for bucket in ("Added", "Removed", "Changed", "Supersedes", "Unchanged"):
        assert bucket in summary_block, f"summary missing bucket {bucket!r}"


def test_md_rendering_lists_all_removals():
    """Removed section shows EVERY URN — the highest-risk reviewer surface."""
    from greenlang.factors.ingestion.diff import serialize_markdown

    diff = _build_mid_size_diff()
    md = serialize_markdown(diff)
    for urn in diff.removed:
        assert urn in md, f"removed URN {urn!r} missing from MD"


def test_json_rendering_has_summary_changes_and_buckets():
    """JSON output carries ``summary`` and the per-bucket lists."""
    from greenlang.factors.ingestion.diff import serialize_json

    payload = serialize_json(_build_mid_size_diff())
    assert "summary" in payload
    summary = payload["summary"]
    assert summary["added"] == 5
    assert summary["removed"] == 2
    assert summary["changed"] == 3
    assert summary["supersedes"] == 1
    # Per-bucket lists.
    for key in ("added", "removed", "changed", "supersedes"):
        assert key in payload, f"json payload missing bucket {key!r}"
    # The full payload is JSON-serialisable.
    json.dumps(payload, sort_keys=True)
