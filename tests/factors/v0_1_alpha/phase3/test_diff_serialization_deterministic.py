# -*- coding: utf-8 -*-
"""Phase 3 — diff serialization is byte-deterministic.

Acceptance: a synthetic ``RunDiff`` rendered twice (md and json) produces
byte-identical output. This is the lever that makes diff review
auditable and the CI snapshot gate possible — non-determinism here would
mean different commits produce different "expected" goldens.

Reference: ``docs/factors/PHASE_3_PLAN.md`` §"Dedupe / supersede / diff
rules" + ``PHASE_3_EXIT_CHECKLIST.md`` Block 4 ("Diff (MD + JSON)
shows: ...").
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


def _build_synthetic_diff(n: int = 5):
    """Construct a synthetic RunDiff with N additions for determinism check."""
    from greenlang.factors.ingestion.diff import ChangeRecord, RunDiff

    return RunDiff(
        run_id="phase3-deterministic-run",
        source_urn="urn:gl:source:phase2-alpha",
        source_version="2024.1",
        added=sorted(
            f"urn:gl:factor:phase3-alpha:test:row-{i}:v1" for i in range(n)
        ),
        removed=[],
        changed=[
            ChangeRecord(
                urn="urn:gl:factor:phase3-alpha:test:row-1:v1",
                attribute="value",
                old_value="0.10",
                new_value="0.20",
            )
        ],
        supersedes=[],
        unchanged_count=2,
    )


def test_diff_md_serialization_byte_identical_across_runs():
    """``serialize_markdown(diff)`` is byte-deterministic across two calls."""
    from greenlang.factors.ingestion.diff import serialize_markdown

    diff = _build_synthetic_diff(n=5)
    md_a = serialize_markdown(diff)
    md_b = serialize_markdown(diff)
    assert md_a == md_b
    assert isinstance(md_a, str)
    assert "Summary" in md_a


def test_diff_json_serialization_byte_identical_across_runs():
    """``serialize_json(diff)`` is byte-deterministic across two calls."""
    from greenlang.factors.ingestion.diff import serialize_json

    diff = _build_synthetic_diff(n=5)
    json_a = serialize_json(diff)
    json_b = serialize_json(diff)
    assert json_a == json_b
    text_a = json.dumps(json_a, sort_keys=True, indent=2)
    text_b = json.dumps(json_b, sort_keys=True, indent=2)
    assert text_a == text_b
    # Round-trip parse to confirm structured payload.
    parsed = json.loads(text_a)
    assert "summary" in parsed
    assert "added" in parsed
    assert "removed" in parsed
    assert "changed" in parsed
