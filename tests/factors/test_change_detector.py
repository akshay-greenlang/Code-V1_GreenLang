# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.watch.change_detector (F051)."""

from __future__ import annotations

import pytest

from greenlang.factors.watch.change_detector import (
    ChangedFactor,
    ChangeReport,
    detect_source_change,
)


def _make_factor(fid: str, co2e: float = 1.0, unit: str = "kg_co2e", **kwargs) -> dict:
    return {"factor_id": fid, "co2e_total": co2e, "unit": unit, **kwargs}


class TestChangedFactor:
    def test_basic(self):
        cf = ChangedFactor(factor_id="EF:1", change_kind="added", new_value=1.23)
        assert cf.factor_id == "EF:1"
        assert cf.change_kind == "added"
        assert cf.old_value is None
        assert cf.new_value == 1.23


class TestChangeReport:
    def test_empty_report(self):
        r = ChangeReport(source_id="test", timestamp="2026-04-17T00:00:00Z")
        assert r.total_changes == 0
        assert r.has_breaking_changes is False

    def test_with_removed(self):
        r = ChangeReport(
            source_id="test",
            timestamp="2026-04-17T00:00:00Z",
            removed=[ChangedFactor(factor_id="EF:1", change_kind="removed")],
        )
        assert r.total_changes == 1
        assert r.has_breaking_changes is True

    def test_with_unit_change(self):
        r = ChangeReport(
            source_id="test",
            timestamp="2026-04-17T00:00:00Z",
            modified=[
                ChangedFactor(
                    factor_id="EF:1",
                    change_kind="modified",
                    field_changes={"unit_changed": True},
                ),
            ],
        )
        assert r.has_breaking_changes is True

    def test_to_dict(self):
        r = ChangeReport(
            source_id="epa",
            timestamp="2026-04-17T00:00:00Z",
            before_count=100,
            after_count=105,
            added=[ChangedFactor(factor_id="EF:new", change_kind="added")],
        )
        d = r.to_dict()
        assert d["source_id"] == "epa"
        assert d["added_count"] == 1
        assert d["removed_count"] == 0
        assert d["total_changes"] == 1
        assert d["has_breaking_changes"] is False


class TestDetectSourceChange:
    def test_no_changes(self):
        old = [_make_factor("EF:1", 1.0), _make_factor("EF:2", 2.0)]
        new = [_make_factor("EF:1", 1.0), _make_factor("EF:2", 2.0)]
        report = detect_source_change("test", old, new)
        assert report.change_type == "no_change"
        assert report.total_changes == 0

    def test_additions_only(self):
        old = [_make_factor("EF:1", 1.0)]
        new = [_make_factor("EF:1", 1.0), _make_factor("EF:2", 2.0)]
        report = detect_source_change("test", old, new)
        assert report.change_type == "additions_only"
        assert len(report.added) == 1
        assert report.added[0].factor_id == "EF:2"

    def test_removals(self):
        old = [_make_factor("EF:1", 1.0), _make_factor("EF:2", 2.0)]
        new = [_make_factor("EF:1", 1.0)]
        report = detect_source_change("test", old, new)
        assert report.has_breaking_changes is True
        assert len(report.removed) == 1
        assert report.removed[0].factor_id == "EF:2"

    def test_modifications(self):
        old = [_make_factor("EF:1", 1.0)]
        new = [_make_factor("EF:1", 1.5)]
        report = detect_source_change("test", old, new)
        assert len(report.modified) == 1
        assert report.modified[0].old_value == 1.0
        assert report.modified[0].new_value == 1.5

    def test_unit_change_is_breaking(self):
        old = [_make_factor("EF:1", 1.0, unit="kg_co2e")]
        new = [_make_factor("EF:1", 1.0, unit="g_co2e")]
        report = detect_source_change("test", old, new)
        assert report.has_breaking_changes is True
        assert report.modified[0].field_changes.get("unit_changed") is True

    def test_human_review_on_mass_removal(self):
        old = [_make_factor(f"EF:{i}", float(i)) for i in range(100)]
        new = [_make_factor(f"EF:{i}", float(i)) for i in range(80)]
        report = detect_source_change("test", old, new)
        assert report.requires_human_review is True
        assert "removed" in (report.review_reason or "").lower()

    def test_human_review_on_mass_change(self):
        old = [_make_factor(f"EF:{i}", float(i)) for i in range(10)]
        new = [_make_factor(f"EF:{i}", float(i) + 100) for i in range(10)]
        report = detect_source_change("test", old, new)
        assert report.requires_human_review is True

    def test_mixed_changes(self):
        old = [_make_factor("EF:1", 1.0), _make_factor("EF:2", 2.0)]
        new = [_make_factor("EF:1", 1.5), _make_factor("EF:3", 3.0)]
        report = detect_source_change("test", old, new)
        assert len(report.added) == 1
        assert len(report.removed) == 1
        assert len(report.modified) == 1
        assert report.change_type == "breaking_change"

    def test_edition_id_propagated(self):
        report = detect_source_change(
            "epa", [], [], edition_id="2026.04.0",
            artifact_hash_old="aaa", artifact_hash_new="bbb",
        )
        assert report.edition_id == "2026.04.0"
        assert report.artifact_hash_old == "aaa"
        assert report.artifact_hash_new == "bbb"

    def test_before_after_counts(self):
        old = [_make_factor(f"EF:{i}", float(i)) for i in range(50)]
        new = [_make_factor(f"EF:{i}", float(i)) for i in range(60)]
        report = detect_source_change("test", old, new)
        assert report.before_count == 50
        assert report.after_count == 60
