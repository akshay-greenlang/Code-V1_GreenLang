# -*- coding: utf-8 -*-
"""Tests for the /api/v1/factors/{factor_id}/quality endpoint."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterator

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository


def _load_factors_router():
    """Load the factors router directly, bypassing routes/__init__."""
    repo_root = Path(__file__).resolve().parents[3]
    factors_path = (
        repo_root
        / "greenlang"
        / "integration"
        / "api"
        / "routes"
        / "factors.py"
    )
    spec = importlib.util.spec_from_file_location(
        "greenlang_factors_router_under_test_quality", str(factors_path)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def client() -> Iterator[TestClient]:
    router_module = _load_factors_router()

    db = EmissionFactorDatabase()
    edition_id = "2027.01"
    repo = MemoryFactorCatalogRepository(edition_id, "test-quality", db)

    class _Svc:
        def __init__(self, r):
            self.repo = r

    def _fake_user():
        return {"user_id": "test-user", "tier": "pro"}

    def _fake_svc():
        return _Svc(repo)

    from greenlang.integration.api.dependencies import (
        get_current_user,
        get_factor_service,
    )

    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_factor_service] = _fake_svc

    with TestClient(app) as c:
        yield c


def _any_factor_id(client: TestClient) -> str:
    """Pick a factor that exists in the fixture repo."""
    resp = client.get("/api/v1/factors?limit=1", headers={"X-Factors-Edition": "2027.01"})
    assert resp.status_code == 200, resp.text
    factors = resp.json().get("factors", [])
    assert factors, "fixture must have at least one factor"
    return factors[0]["factor_id"]


def test_quality_endpoint_returns_fqs_payload(client: TestClient) -> None:
    factor_id = _any_factor_id(client)
    resp = client.get(
        f"/api/v1/factors/{factor_id}/quality",
        headers={"X-Factors-Edition": "2027.01"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["factor_id"] == factor_id
    assert body["edition_id"] == "2027.01"
    # Composite FQS surface.
    fqs = body["fqs"]
    assert set(fqs) == {
        "composite_fqs",
        "rating",
        "promotion_eligibility",
        "components",
        "formula_version",
        "weights",
    }
    assert 0 <= fqs["composite_fqs"] <= 100
    assert fqs["rating"] in {"excellent", "good", "fair", "poor"}
    assert fqs["promotion_eligibility"] in {"certified", "preview", "connector_only"}
    # 5 components with CTO alias + both scales.
    assert len(fqs["components"]) == 5
    names = {c["name"] for c in fqs["components"]}
    assert names == {
        "temporal",
        "geographical",
        "technological",
        "representativeness",
        "methodological",
    }
    for c in fqs["components"]:
        assert 1 <= c["score_5"] <= 5
        assert 20 <= c["score_100"] <= 100


def test_quality_endpoint_404_on_unknown_factor(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/factors/DOES_NOT_EXIST/quality",
        headers={"X-Factors-Edition": "2027.01"},
    )
    assert resp.status_code == 404


def test_quality_endpoint_sets_edition_headers(client: TestClient) -> None:
    factor_id = _any_factor_id(client)
    resp = client.get(
        f"/api/v1/factors/{factor_id}/quality",
        headers={"X-Factors-Edition": "2027.01"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-GreenLang-Edition") == "2027.01"
    assert resp.headers.get("X-Factors-Edition") == "2027.01"
    assert "ETag" in resp.headers
    assert "Cache-Control" in resp.headers
