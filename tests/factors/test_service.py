# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.service."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from greenlang.factors.service import FactorCatalogService, resolve_edition_id


def test_from_environment_sqlite(tmp_path, monkeypatch):
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    from greenlang.factors.etl.ingest import ingest_builtin_database

    dbfile = tmp_path / "fc_env.sqlite"
    ingest_builtin_database(dbfile, "env-edition", label="env")
    monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))
    db = EmissionFactorDatabase(enable_cache=False)
    svc = FactorCatalogService.from_environment(db)
    assert svc.repo.get_default_edition_id() == "env-edition"


def test_from_environment_memory_fallback(monkeypatch):
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository

    monkeypatch.delenv("GL_FACTORS_SQLITE_PATH", raising=False)
    db = EmissionFactorDatabase(enable_cache=False)
    svc = FactorCatalogService.from_environment(db)
    assert isinstance(svc.repo, MemoryFactorCatalogRepository)


def test_compare_editions_identical(factor_service):
    eid = factor_service.repo.get_default_edition_id()
    diff = factor_service.compare_editions(eid, eid)
    assert diff["added_factor_ids"] == []
    assert diff["removed_factor_ids"] == []
    assert diff["changed_factor_ids"] == []
    assert diff["unchanged_count"] > 0


def test_compare_editions_detects_added(tmp_path, monkeypatch):
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
    from greenlang.factors.etl.ingest import ingest_builtin_database

    dbfile = tmp_path / "cmp.sqlite"
    db = EmissionFactorDatabase(enable_cache=False)
    all_factors = list(db.factors.values())

    # Edition A has all factors
    ingest_builtin_database(dbfile, "ed-all", label="all")

    # Edition B has only first 5
    repo = SqliteFactorCatalogRepository(dbfile)
    from greenlang.factors.edition_manifest import build_manifest_for_factors

    subset = all_factors[:5]
    m = build_manifest_for_factors("ed-subset", "stable", subset)
    repo.upsert_edition("ed-subset", "stable", "subset", m.to_dict(), m.changelog)
    repo.insert_factors("ed-subset", subset)

    svc = FactorCatalogService(repo)
    diff = svc.compare_editions("ed-subset", "ed-all")
    assert len(diff["added_factor_ids"]) > 0


def test_replacement_chain_single(factor_service):
    eid = factor_service.repo.get_default_edition_id()
    summaries = factor_service.repo.list_factor_summaries(eid)
    fid = summaries[0]["factor_id"]
    chain = factor_service.replacement_chain(eid, fid)
    assert chain == [fid]


def test_replacement_chain_cycle_protection(factor_service):
    eid = factor_service.repo.get_default_edition_id()
    # Non-existent factor should produce short chain
    chain = factor_service.replacement_chain(eid, "EF:NONEXIST:x:2024:v1")
    assert chain == ["EF:NONEXIST:x:2024:v1"]


def test_resolve_edition_header_priority(sqlite_catalog):
    eid = sqlite_catalog.get_default_edition_id()
    resolved, source = resolve_edition_id(sqlite_catalog, eid, None)
    assert resolved == eid
    assert source == "header"


def test_resolve_edition_force_override(sqlite_catalog, monkeypatch):
    eid = sqlite_catalog.get_default_edition_id()
    monkeypatch.setenv("GL_FACTORS_FORCE_EDITION", eid)
    resolved, source = resolve_edition_id(sqlite_catalog, None, None)
    assert resolved == eid
    assert source == "rollback_override"
