# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.watch submodules."""

from __future__ import annotations

from greenlang.factors.watch.change_classification import classify_change
from greenlang.factors.watch.changelog_draft import draft_changelog_lines
from greenlang.factors.watch.doc_diff import diff_text_versions, fingerprint_text


def test_doc_diff():
    assert fingerprint_text("a") != fingerprint_text("b")
    changed, msg = diff_text_versions("x", "y")
    assert changed and "delta" in msg


def test_fingerprint_text_deterministic():
    h1 = fingerprint_text("same input")
    h2 = fingerprint_text("same input")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_change_classification_numeric():
    result = classify_change(
        old_hash="aaa", new_hash="bbb",
        old_row={"x": 1}, new_row={"x": 1},
    )
    assert result == "numeric_or_vectors"


def test_change_classification_metadata():
    result = classify_change(
        old_hash="aaa", new_hash="aaa",
        old_row={"x": 1}, new_row={"x": 2},
    )
    assert result == "metadata"


def test_change_classification_docs_only():
    result = classify_change(
        old_hash="aaa", new_hash="aaa",
        old_row={"x": 1}, new_row={"x": 1},
    )
    assert result == "docs_only"


def test_changelog_draft_format():
    compare = {
        "left_edition_id": "v1",
        "right_edition_id": "v2",
        "added_factor_ids": ["a", "b"],
        "removed_factor_ids": [],
        "changed_factor_ids": ["c"],
    }
    lines = draft_changelog_lines(compare)
    assert isinstance(lines, list)
    assert len(lines) >= 3
    assert "v1" in lines[0]
    assert "v2" in lines[0]
    assert "added: 2" in lines[1]
