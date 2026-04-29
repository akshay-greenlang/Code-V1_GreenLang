# -*- coding: utf-8 -*-
"""Phase 3 Block 6 — branch-coverage tests for ``ingestion/diff.py``.

Targets the residual ~4pp gap left by the existing happy-path tests:

* ``from_staging_diff`` corner cases — empty staging, empty production,
  records-only-on-one-side, all attributes equal (no change), supersede
  pair where one side is missing.
* ``serialize_markdown`` — the ``_md_cell`` escaping of pipes / newlines /
  long values / empty strings, and the ``Added > 50`` truncation branch.
* ``RunDiff.is_empty()`` / ``RunDiff.total_changes()`` corner cases.
* ``_flatten`` callsites for ``parser_version`` / ``citation`` /
  ``methodology`` with non-dict ``extraction`` payloads.
"""
from __future__ import annotations

from types import SimpleNamespace

from greenlang.factors.ingestion.diff import (
    ChangeRecord,
    RunDiff,
    from_staging_diff,
    serialize_json,
    serialize_markdown,
)


# ---------------------------------------------------------------------------
# RunDiff convenience accessors
# ---------------------------------------------------------------------------


def test_rundiff_is_empty_on_default():
    diff = RunDiff()
    assert diff.is_empty() is True
    assert diff.total_changes() == 0


def test_rundiff_is_empty_with_only_unchanged_count():
    """``unchanged_count > 0`` does NOT count as a change."""
    diff = RunDiff(unchanged_count=12)
    assert diff.is_empty() is True
    assert diff.total_changes() == 0


def test_rundiff_total_changes_counts_every_bucket():
    diff = RunDiff(
        added=["urn:1"],
        removed=["urn:2", "urn:3"],
        changed=[ChangeRecord("urn:4", "value", "1", "2")],
        supersedes=[("a", "b"), ("c", "d"), ("e", "f")],
        parser_version_changes=[ChangeRecord("urn:4", "parser_version", "1.0", "2.0")],
        licence_changes=[ChangeRecord("urn:4", "licence", "x", "y")],
        methodology_changes=[ChangeRecord("urn:4", "methodology", "m1", "m2")],
    )
    assert diff.is_empty() is False
    # 1 added + 2 removed + 1 changed + 3 supersedes + 1 pv + 1 lic + 1 meth = 10
    assert diff.total_changes() == 10


def test_rundiff_is_empty_only_methodology_change():
    diff = RunDiff(methodology_changes=[ChangeRecord("u", "methodology", None, "x")])
    assert diff.is_empty() is False


def test_rundiff_is_empty_only_licence_change():
    diff = RunDiff(licence_changes=[ChangeRecord("u", "licence", None, "x")])
    assert diff.is_empty() is False


def test_rundiff_is_empty_only_parser_version_change():
    diff = RunDiff(parser_version_changes=[ChangeRecord("u", "parser_version", None, "x")])
    assert diff.is_empty() is False


# ---------------------------------------------------------------------------
# from_staging_diff edge cases
# ---------------------------------------------------------------------------


def test_from_staging_diff_handles_completely_empty_inputs():
    """Both production and staging empty + no additions/removals/changes."""
    sd = SimpleNamespace(additions=[], removals=[], changes=[], unchanged=0)
    diff = from_staging_diff(sd, run_id="r1", source_urn="s", source_version="1")
    assert diff.is_empty() is True
    assert diff.run_id == "r1"
    assert diff.source_urn == "s"
    assert diff.source_version == "1"


def test_from_staging_diff_handles_attrs_returning_none():
    """When the legacy diff lacks attributes ``getattr`` returns None — coalesce."""
    sd = SimpleNamespace()
    diff = from_staging_diff(sd)
    assert diff.added == []
    assert diff.removed == []
    assert diff.supersedes == []
    assert diff.unchanged_count == 0


def test_from_staging_diff_skips_per_attr_changes_when_records_missing():
    """Without prod+staging records, supersede pairs are kept but no per-attr deltas."""
    sd = SimpleNamespace(
        additions=[{"urn": "u:new"}],
        removals=["u:old"],
        changes=[("urn:old", "urn:new")],
        unchanged=2,
    )
    diff = from_staging_diff(sd, run_id="r")
    assert diff.added == ["u:new"]
    assert diff.removed == ["u:old"]
    assert diff.supersedes == [("urn:old", "urn:new")]
    assert diff.unchanged_count == 2
    assert diff.changed == []


def test_from_staging_diff_emits_parser_version_change():
    sd = SimpleNamespace(
        additions=[],
        removals=[],
        changes=[("urn:old", "urn:new")],
        unchanged=0,
    )
    prod = {"urn:old": {"extraction": {"parser_version": "1.0"}}}
    stag = {"urn:new": {"extraction": {"parser_version": "2.0"}}}
    diff = from_staging_diff(
        sd, production_records=prod, staging_records=stag,
    )
    assert len(diff.parser_version_changes) == 1
    pv = diff.parser_version_changes[0]
    assert pv.old_value == "1.0"
    assert pv.new_value == "2.0"
    # Also lands in ``changed`` (per-attribute table).
    assert any(c.attribute == "parser_version" for c in diff.changed)


def test_from_staging_diff_emits_licence_change():
    sd = SimpleNamespace(
        additions=[], removals=[], changes=[("o", "n")], unchanged=0,
    )
    prod = {"o": {"licence": "CC-BY-4.0"}}
    stag = {"n": {"licence": "OGL-3.0"}}
    diff = from_staging_diff(sd, production_records=prod, staging_records=stag)
    assert len(diff.licence_changes) == 1
    assert diff.licence_changes[0].old_value == "CC-BY-4.0"
    assert diff.licence_changes[0].new_value == "OGL-3.0"


def test_from_staging_diff_emits_methodology_change_via_methodology_urn():
    sd = SimpleNamespace(
        additions=[], removals=[], changes=[("o", "n")], unchanged=0,
    )
    # Only ``methodology_urn`` set on staging side — _flatten falls through.
    prod = {"o": {"methodology": "m_old"}}
    stag = {"n": {"methodology_urn": "m_new"}}
    diff = from_staging_diff(sd, production_records=prod, staging_records=stag)
    assert len(diff.methodology_changes) == 1
    assert diff.methodology_changes[0].old_value == "m_old"
    assert diff.methodology_changes[0].new_value == "m_new"


def test_from_staging_diff_skips_unchanged_attributes():
    sd = SimpleNamespace(
        additions=[], removals=[], changes=[("o", "n")], unchanged=0,
    )
    prod = {"o": {"value": 1.0, "unit": "kg"}}
    stag = {"n": {"value": 1.0, "unit": "kg"}}
    diff = from_staging_diff(sd, production_records=prod, staging_records=stag)
    # No attribute differs → no per-attribute changes.
    assert diff.changed == []


def test_from_staging_diff_handles_extraction_being_non_dict():
    """``extraction`` is sometimes a non-dict (e.g. legacy str)."""
    sd = SimpleNamespace(
        additions=[], removals=[], changes=[("o", "n")], unchanged=0,
    )
    prod = {"o": {"extraction": "not-a-dict"}}
    stag = {"n": {"extraction": "still-not-a-dict"}}
    diff = from_staging_diff(sd, production_records=prod, staging_records=stag)
    # _flatten returns None for both sides → no parser_version delta.
    assert diff.parser_version_changes == []


def test_from_staging_diff_citation_attribute_path():
    """``citation`` is read from ``extraction.source_publication`` first."""
    sd = SimpleNamespace(
        additions=[], removals=[], changes=[("o", "n")], unchanged=0,
    )
    prod = {"o": {"extraction": {"source_publication": "Pub-A"}}}
    stag = {"n": {"extraction": {"source_publication": "Pub-B"}}}
    diff = from_staging_diff(sd, production_records=prod, staging_records=stag)
    cites = [c for c in diff.changed if c.attribute == "citation"]
    assert len(cites) == 1
    assert cites[0].old_value == "Pub-A"
    assert cites[0].new_value == "Pub-B"


def test_from_staging_diff_falls_back_to_top_level_citation_when_extraction_non_dict():
    sd = SimpleNamespace(
        additions=[], removals=[], changes=[("o", "n")], unchanged=0,
    )
    prod = {"o": {"extraction": None, "citation": "C-old"}}
    stag = {"n": {"extraction": None, "citation": "C-new"}}
    diff = from_staging_diff(sd, production_records=prod, staging_records=stag)
    cites = [c for c in diff.changed if c.attribute == "citation"]
    assert len(cites) == 1
    assert cites[0].old_value == "C-old"
    assert cites[0].new_value == "C-new"


# ---------------------------------------------------------------------------
# serialize_json + serialize_markdown formatting branches
# ---------------------------------------------------------------------------


def test_serialize_json_empty_diff_is_stable():
    diff = RunDiff(run_id="r", source_urn="s", source_version="1")
    payload = serialize_json(diff)
    assert payload["summary"] == {
        "added": 0,
        "removed": 0,
        "changed": 0,
        "supersedes": 0,
        "unchanged": 0,
        "parser_version_changes": 0,
        "licence_changes": 0,
        "methodology_changes": 0,
        "total_changes": 0,
    }
    assert payload["added"] == []


def test_serialize_markdown_truncates_added_list_after_50():
    """The ``Added (N)`` section caps at 50 + emits a ``... N more`` line."""
    diff = RunDiff(added=[f"urn:gl:test:{i:04d}" for i in range(75)])
    out = serialize_markdown(diff)
    # First-50 should appear.
    assert "urn:gl:test:0000" in out
    assert "urn:gl:test:0049" in out
    # 50th-onwards should be truncated.
    assert "urn:gl:test:0050" not in out
    assert "... 25 more not shown" in out


def test_serialize_markdown_md_cell_escapes_pipes_and_newlines():
    """A ChangeRecord with ``|`` and newlines must be Markdown-safe."""
    diff = RunDiff(
        changed=[ChangeRecord("urn:1", "value", "old|with|pipes\nand\nnewlines", "new")]
    )
    out = serialize_markdown(diff)
    # Pipes escaped, newlines collapsed to spaces.
    assert "\\|" in out
    assert "old\nand\nnewlines" not in out


def test_serialize_markdown_md_cell_truncates_long_values():
    """Values > 80 chars are truncated with ``...``."""
    long_val = "x" * 200
    diff = RunDiff(changed=[ChangeRecord("urn:1", "value", long_val, "new")])
    out = serialize_markdown(diff)
    # Long string is truncated; ellipsis present.
    assert "..." in out
    assert "x" * 200 not in out


def test_serialize_markdown_md_cell_renders_empty_string_marker():
    diff = RunDiff(changed=[ChangeRecord("urn:1", "value", "", "non-empty")])
    out = serialize_markdown(diff)
    assert "_empty_" in out


def test_serialize_markdown_md_cell_renders_null_marker():
    diff = RunDiff(changed=[ChangeRecord("urn:1", "value", None, None)])
    out = serialize_markdown(diff)
    assert "_null_" in out


def test_serialize_markdown_handles_dict_value_via_to_text():
    """``_to_text`` JSON-encodes dict / list values."""
    diff = RunDiff()
    sd = SimpleNamespace(
        additions=[], removals=[], changes=[("o", "n")], unchanged=0,
    )
    prod = {"o": {"value": {"nested": 1}}}
    stag = {"n": {"value": {"nested": 2}}}
    diff = from_staging_diff(sd, production_records=prod, staging_records=stag)
    out = serialize_markdown(diff)
    # JSON-encoded dict appears in the change row.
    assert "nested" in out


def test_serialize_markdown_sections_render_in_fixed_order():
    """All eight sections must appear in the canonical order."""
    diff = RunDiff(
        run_id="r1",
        source_urn="src",
        source_version="v1",
        added=["a"],
        removed=["b"],
        supersedes=[("o", "n")],
        changed=[ChangeRecord("u", "value", "1", "2")],
        parser_version_changes=[ChangeRecord("u", "parser_version", "1.0", "2.0")],
        licence_changes=[ChangeRecord("u", "licence", "x", "y")],
        methodology_changes=[ChangeRecord("u", "methodology", "m1", "m2")],
        unchanged_count=3,
    )
    out = serialize_markdown(diff)
    sections = (
        "## Summary",
        "## Added",
        "## Removed",
        "## Supersedes",
        "## Changed attributes",
        "## Parser-version changes",
        "## Licence changes",
        "## Methodology changes",
    )
    last = -1
    for s in sections:
        idx = out.find(s)
        assert idx > last, f"section {s!r} missing or out-of-order in markdown"
        last = idx
