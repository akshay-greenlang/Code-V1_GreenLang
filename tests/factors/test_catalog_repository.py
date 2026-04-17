# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.catalog_repository (SQLite + Memory)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_sqlite_ingest_and_list(sqlite_catalog):
    eid = sqlite_catalog.get_default_edition_id()
    assert eid
    rows, total = sqlite_catalog.list_factors(eid, page=1, limit=10)
    assert total >= 1
    assert len(rows) == min(10, total)


def test_sqlite_search_and_coverage(sqlite_catalog):
    eid = sqlite_catalog.get_default_edition_id()
    hits = sqlite_catalog.search_factors(eid, query="dies", limit=5)
    assert isinstance(hits, list)
    cov = sqlite_catalog.coverage_stats(eid)
    assert cov["total_factors"] > 0
    facets = sqlite_catalog.search_facets(eid, max_values=40)
    assert facets["edition_id"] == eid
    assert "factor_status" in facets["facets"]


def test_sqlite_get_factor_by_id(sqlite_catalog):
    eid = sqlite_catalog.get_default_edition_id()
    summaries = sqlite_catalog.list_factor_summaries(eid)
    assert len(summaries) > 0
    fid = summaries[0]["factor_id"]
    rec = sqlite_catalog.get_factor(eid, fid)
    assert rec is not None
    assert rec.factor_id == fid


def test_sqlite_get_factor_missing_returns_none(sqlite_catalog):
    eid = sqlite_catalog.get_default_edition_id()
    rec = sqlite_catalog.get_factor(eid, "EF:NONEXISTENT:x:y:z")
    assert rec is None


def test_sqlite_resolve_edition_default(sqlite_catalog):
    eid = sqlite_catalog.resolve_edition(None)
    assert eid


def test_sqlite_resolve_edition_unknown_raises(sqlite_catalog):
    with pytest.raises(ValueError, match="Unknown"):
        sqlite_catalog.resolve_edition("does-not-exist-xyz")


def test_sqlite_list_factor_summaries(sqlite_catalog):
    eid = sqlite_catalog.get_default_edition_id()
    summaries = sqlite_catalog.list_factor_summaries(eid)
    assert isinstance(summaries, list)
    assert len(summaries) > 0
    row = summaries[0]
    assert "factor_id" in row
    assert "content_hash" in row


def test_memory_repo_list(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    rows, total = memory_catalog.list_factors(eid, page=1, limit=10)
    assert total > 0
    assert len(rows) == min(10, total)


def test_memory_repo_search(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    hits = memory_catalog.search_factors(eid, query="diesel", limit=5)
    assert isinstance(hits, list)


def test_memory_repo_get_factor(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    summaries = memory_catalog.list_factor_summaries(eid)
    fid = summaries[0]["factor_id"]
    rec = memory_catalog.get_factor(eid, fid)
    assert rec is not None
    assert rec.factor_id == fid


def test_memory_repo_coverage_stats(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    cov = memory_catalog.coverage_stats(eid)
    assert cov["total_factors"] > 0


def test_memory_repo_facets(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    facets = memory_catalog.search_facets(eid)
    assert "factor_status" in facets["facets"]
