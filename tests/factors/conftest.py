# -*- coding: utf-8 -*-
"""Shared pytest fixtures for GreenLang Factors test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def emission_db():
    """Session-scoped EmissionFactorDatabase with cache disabled."""
    from greenlang.data.emission_factor_database import EmissionFactorDatabase

    return EmissionFactorDatabase(enable_cache=False)


@pytest.fixture()
def sample_factor(emission_db):
    """First factor from the built-in database."""
    return next(iter(emission_db.factors.values()))


@pytest.fixture()
def sample_factor_dict(sample_factor):
    """Sample factor serialized to dict (for QA tests)."""
    return sample_factor.to_dict()


@pytest.fixture()
def sqlite_catalog(tmp_path, emission_db):
    """Ingested SQLite DB + SqliteFactorCatalogRepository."""
    from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
    from greenlang.factors.etl.ingest import ingest_builtin_database

    dbfile = tmp_path / "test_catalog.sqlite"
    ingest_builtin_database(dbfile, "test-edition", label="conftest")
    repo = SqliteFactorCatalogRepository(dbfile)
    return repo


@pytest.fixture()
def memory_catalog(emission_db):
    """MemoryFactorCatalogRepository wrapping built-in DB."""
    from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository

    return MemoryFactorCatalogRepository("memory-v1", "memory-test", emission_db)


@pytest.fixture()
def factor_service(sqlite_catalog):
    """FactorCatalogService wrapping SQLite catalog."""
    from greenlang.factors.service import FactorCatalogService

    return FactorCatalogService(sqlite_catalog)


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    """FastAPI TestClient with GL_ENV=test and a pre-ingested SQLite catalog."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GL_ENV", "test")
    dbfile = tmp_path / "api_test.sqlite"
    monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))

    from greenlang.factors.etl.ingest import ingest_builtin_database

    ingest_builtin_database(dbfile, "api-test-edition", label="api-test")

    from greenlang.integration.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def source_registry():
    """Loaded SourceRegistryEntry list from default YAML."""
    from greenlang.factors.source_registry import load_source_registry

    return load_source_registry()


@pytest.fixture()
def gold_eval_cases():
    """Load gold eval smoke cases from fixtures."""
    path = FIXTURES_DIR / "gold_eval_smoke.json"
    return json.loads(path.read_text(encoding="utf-8"))
