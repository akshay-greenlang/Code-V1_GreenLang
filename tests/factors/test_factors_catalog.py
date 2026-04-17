# -*- coding: utf-8 -*-
"""Tests for GreenLang Factors FY27 catalog, editions, and API."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


def test_edition_manifest_fingerprint_is_deterministic():
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    from greenlang.factors.edition_manifest import build_manifest_for_factors

    db = EmissionFactorDatabase(enable_cache=False)
    factors = list(db.factors.values())
    m1 = build_manifest_for_factors("test-edition", "stable", factors, changelog=["line a"])
    m2 = build_manifest_for_factors("test-edition", "stable", factors, changelog=["line a"])
    assert m1.manifest_fingerprint() == m2.manifest_fingerprint()


def test_sqlite_ingest_and_list(tmp_path: Path):
    from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
    from greenlang.factors.etl.ingest import ingest_builtin_database

    dbfile = tmp_path / "fc.sqlite"
    n = ingest_builtin_database(dbfile, "e2e-test-edition", label="test")
    assert n > 0
    repo = SqliteFactorCatalogRepository(dbfile)
    rows, total = repo.list_factors(repo.get_default_edition_id(), page=1, limit=10)
    # Primary key is (edition_id, factor_id); duplicate factor_ids collapse to one row.
    assert total <= n
    assert total >= 1
    assert len(rows) == min(10, total)


def test_sqlite_search_and_coverage(tmp_path: Path):
    from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
    from greenlang.factors.etl.ingest import ingest_builtin_database

    dbfile = tmp_path / "fc2.sqlite"
    ingest_builtin_database(dbfile, "e2e-search", label="test")
    repo = SqliteFactorCatalogRepository(dbfile)
    eid = repo.get_default_edition_id()
    hits = repo.search_factors(eid, query="dies", limit=5)
    assert isinstance(hits, list)
    cov = repo.coverage_stats(eid)
    assert cov["total_factors"] > 0
    facets = repo.search_facets(eid, max_values=40)
    assert facets["edition_id"] == eid
    assert "factor_status" in facets["facets"]
    assert sum(facets["facets"]["factor_status"].values()) == cov["total_factors"]


def test_factors_api_editions_and_list(monkeypatch):
    monkeypatch.setenv("GL_ENV", "test")
    monkeypatch.delenv("GL_FACTORS_SQLITE_PATH", raising=False)
    from fastapi.testclient import TestClient

    from greenlang.integration.api import main as api_main

    with TestClient(api_main.app, raise_server_exceptions=True) as client:
        r = client.get("/api/v1/editions")
        assert r.status_code == 200
        body = r.json()
        assert "editions" in body and body["default_edition_id"]

        r2 = client.get("/api/v1/factors", params={"limit": 5})
        assert r2.status_code == 200
        data = r2.json()
        assert data.get("edition_id")
        assert "X-Factors-Edition" in r2.headers


def test_factors_api_unknown_edition_returns_404(monkeypatch):
    monkeypatch.setenv("GL_ENV", "test")
    monkeypatch.delenv("GL_FACTORS_SQLITE_PATH", raising=False)
    from fastapi.testclient import TestClient

    from greenlang.integration.api import main as api_main

    with TestClient(api_main.app, raise_server_exceptions=True) as client:
        r = client.get("/api/v1/factors", params={"edition": "does-not-exist-xyz"})
        assert r.status_code == 404


def test_sqlite_catalog_load_via_env(tmp_path: Path, monkeypatch):
    from greenlang.factors.etl.ingest import ingest_builtin_database
    from greenlang.factors.service import FactorCatalogService
    from greenlang.data.emission_factor_database import EmissionFactorDatabase

    dbfile = tmp_path / "fc_env.sqlite"
    ingest_builtin_database(dbfile, "env-edition", label="env")
    monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))
    svc = FactorCatalogService.from_environment(EmissionFactorDatabase(enable_cache=False))
    assert svc.repo.get_default_edition_id() == "env-edition"


def test_policy_pending_roundtrip(tmp_path: Path):
    from greenlang.factors.policy_mapping import (
        clear_pending_edition,
        read_pending_edition,
        write_pending_edition,
    )

    dbf = tmp_path / "x.sqlite"
    dbf.write_bytes(b"")
    write_pending_edition(
        dbf,
        edition_id="pending-1",
        reason="policy",
        factor_ids=["EF:US:diesel:2024:v1"],
        policy_rule_ids=["RULE_1"],
    )
    payload = read_pending_edition(dbf)
    assert payload and payload["proposed_edition_id"] == "pending-1"
    clear_pending_edition(dbf)
    assert read_pending_edition(dbf) is None


def test_inventory_json_structure():
    from greenlang.factors.inventory import collect_inventory

    data = collect_inventory()
    assert "sources" in data and isinstance(data["sources"], list)
    dumped = json.dumps(data, sort_keys=True)
    assert "emission_factor_database_v2" in dumped
    reg = data.get("source_registry") or {}
    assert reg.get("entries", 0) > 0
    assert reg.get("validation_issues") == []


def test_source_registry_validate():
    from greenlang.factors.source_registry import load_source_registry, validate_registry

    assert validate_registry(load_source_registry()) == []


def test_factors_api_source_registry_and_facets(monkeypatch):
    monkeypatch.setenv("GL_ENV", "test")
    monkeypatch.delenv("GL_FACTORS_SQLITE_PATH", raising=False)
    from fastapi.testclient import TestClient

    from greenlang.integration.api import main as api_main

    with TestClient(api_main.app, raise_server_exceptions=True) as client:
        r = client.get("/api/v1/factors/source-registry")
        assert r.status_code == 200
        body = r.json()
        assert body["sources"]
        row = body["sources"][0]
        for key in (
            "citation_text",
            "derivative_works_allowed",
            "commercial_use_allowed",
            "public_bulk_export_allowed",
            "approval_required_for_certified",
        ):
            assert key in row

        f = client.get("/api/v1/factors/search/facets")
        assert f.status_code == 200
        fb = f.json()
        assert "edition_id" in fb and "facets" in fb
        assert "factor_status" in fb["facets"]


def test_compare_editions_sqlite(tmp_path: Path, monkeypatch):
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    from greenlang.factors.etl.ingest import ingest_builtin_database
    from greenlang.factors.service import FactorCatalogService

    dbfile = tmp_path / "cmp.sqlite"
    ingest_builtin_database(dbfile, "cmp-edition", label="cmp")
    monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))
    svc = FactorCatalogService.from_environment(EmissionFactorDatabase(enable_cache=False))
    diff = svc.compare_editions("cmp-edition", "cmp-edition")
    assert diff["added_factor_ids"] == []
    assert diff["removed_factor_ids"] == []
    assert diff["changed_factor_ids"] == []
    assert diff["unchanged_count"] == len(svc.repo.list_factor_summaries("cmp-edition"))
