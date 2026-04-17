# -*- coding: utf-8 -*-
"""OpenAPI contract anchors for Factors CTO backlog (A5)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


REQUIRED_PATHS = frozenset(
    {
        "/api/v1/editions",
        "/api/v1/editions/compare",
        "/api/v1/factors",
        "/api/v1/factors/search",
        "/api/v1/factors/search/v2",
        "/api/v1/factors/search/facets",
        "/api/v1/factors/match",
        "/api/v1/factors/source-registry",
    }
)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("GL_ENV", "test")
    dbfile = tmp_path / "fc.sqlite"
    monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    from greenlang.factors.etl.ingest import ingest_builtin_database

    ingest_builtin_database(dbfile, "test-edition", label="contract")
    from greenlang.integration.api.main import app

    with TestClient(app) as c:
        yield c


def test_openapi_contains_factors_routes(client: TestClient) -> None:
    spec = client.get("/api/openapi.json").json()
    paths = set(spec.get("paths") or {})
    missing = sorted(REQUIRED_PATHS - paths)
    assert not missing, f"OpenAPI missing paths: {missing}"
