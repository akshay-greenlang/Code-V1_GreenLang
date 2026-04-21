# -*- coding: utf-8 -*-
"""GAP-2 — Tests for the Factors ``/explain`` family of endpoints.

Covers:
    * GET  /api/v1/factors/{factor_id}/explain
    * POST /api/v1/factors/resolve-explain
    * GET  /api/v1/factors/{factor_id}/alternates

Tier gate: Pro, Consulting, Enterprise, or Internal.  Community → 403.

The tests spin up a minimal FastAPI app that includes only the factors
router so we don't depend on the broader integration/api app (which pulls
in optional connector modules at import-time).  The router's dependencies
(``get_current_user``, ``get_factor_service``) are overridden with
deterministic stubs backed by the built-in ``EmissionFactorDatabase``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository


# ---------------------------------------------------------------------------
# Router loading — bypass routes/__init__.py to avoid unrelated broken imports.
# ---------------------------------------------------------------------------


def _load_factors_router():
    """Load the factors router module directly, bypassing ``routes/__init__``.

    The sibling ``routes/calculations.py`` has an unrelated ModuleNotFoundError
    on ``greenlang.api.models`` that trips any ``from greenlang.integration.
    api.routes import …`` import chain.  Loading ``factors.py`` by file path
    isolates us from that breakage.
    """
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
        "greenlang_factors_router_under_test", str(factors_path)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["greenlang_factors_router_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures — fake factor service + per-tier auth stubs.
# ---------------------------------------------------------------------------


class _FakeFactorService:
    """Minimal stand-in for ``FactorCatalogService`` used by the router."""

    def __init__(self, repo: MemoryFactorCatalogRepository):
        self.repo = repo


@pytest.fixture(scope="module")
def emission_db() -> EmissionFactorDatabase:
    return EmissionFactorDatabase(enable_cache=False)


@pytest.fixture(scope="module")
def memory_repo(emission_db: EmissionFactorDatabase) -> MemoryFactorCatalogRepository:
    return MemoryFactorCatalogRepository("test-explain-v1", "test", emission_db)


@pytest.fixture(scope="module")
def factor_service(memory_repo) -> _FakeFactorService:
    return _FakeFactorService(memory_repo)


@pytest.fixture(scope="module")
def sample_factor(memory_repo) -> Any:
    factors, _ = memory_repo.list_factors("test-explain-v1", page=1, limit=1)
    return factors[0]


@pytest.fixture(scope="module")
def sample_factor_id(sample_factor) -> str:
    return sample_factor.factor_id


@pytest.fixture(scope="module")
def factors_router_module():
    return _load_factors_router()


# The auth stub is per-test so we can swap tier without rebuilding the app.
_CURRENT_TIER: Dict[str, str] = {"tier": "pro"}


def _make_app(factor_service: _FakeFactorService, factors_router_module) -> FastAPI:
    """Build a mini FastAPI app with the factors router mounted + deps stubbed."""
    from greenlang.integration.api.dependencies import (
        get_current_user,
        get_factor_service,
    )

    app = FastAPI()
    app.include_router(factors_router_module.router)

    async def _stub_user() -> dict:
        return {
            "user_id": "test-user",
            "tenant_id": "test-tenant",
            "tier": _CURRENT_TIER["tier"],
        }

    def _stub_service():
        return factor_service

    # Also override the auth dependency used by the router module when it
    # was loaded by file path — it imports the same symbol from dependencies.
    app.dependency_overrides[get_current_user] = _stub_user
    app.dependency_overrides[get_factor_service] = _stub_service
    return app


@pytest.fixture()
def client(
    factor_service: _FakeFactorService, factors_router_module
) -> Iterator[TestClient]:
    _CURRENT_TIER["tier"] = "pro"  # reset per test
    app = _make_app(factor_service, factors_router_module)
    with TestClient(app) as c:
        yield c


def _set_tier(tier: str) -> None:
    _CURRENT_TIER["tier"] = tier


# ---------------------------------------------------------------------------
# GET /{factor_id}/explain
# ---------------------------------------------------------------------------


class TestExplainEndpointOk:
    def test_200_for_valid_factor_id(self, client, sample_factor_id):
        _set_tier("pro")
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/explain")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["chosen_factor_id"] == sample_factor_id
        assert 1 <= body["fallback_rank"] <= 7
        assert body["step_label"]
        assert body["why_chosen"]
        assert "assumptions" in body and isinstance(body["assumptions"], list)
        assert "gas_breakdown" in body
        assert "uncertainty" in body

    def test_headers_include_edition_and_method_profile(self, client, sample_factor_id):
        _set_tier("pro")
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/explain")
        assert resp.status_code == 200
        assert resp.headers.get("X-GreenLang-Edition") == "test-explain-v1"
        assert resp.headers.get("X-GreenLang-Method-Profile")
        assert resp.headers.get("X-Factors-Edition") == "test-explain-v1"

    def test_etag_roundtrip_returns_304(self, client, sample_factor_id):
        _set_tier("pro")
        r1 = client.get(f"/api/v1/factors/{sample_factor_id}/explain")
        assert r1.status_code == 200
        etag = r1.headers.get("ETag")
        assert etag and etag.startswith('"')

        r2 = client.get(
            f"/api/v1/factors/{sample_factor_id}/explain",
            headers={"If-None-Match": etag},
        )
        assert r2.status_code == 304

    def test_gas_breakdown_components_separated(self, client, sample_factor_id):
        """Non-negotiable: every gas component must be returned separately.

        The CTO spec forbids rolling the breakdown into a single co2e figure.
        """
        _set_tier("pro")
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/explain")
        assert resp.status_code == 200
        gb = resp.json()["gas_breakdown"]
        for key in (
            "co2_kg",
            "ch4_kg",
            "n2o_kg",
            "hfcs_kg",
            "pfcs_kg",
            "sf6_kg",
            "nf3_kg",
            "biogenic_co2_kg",
            "co2e_total_kg",
            "gwp_basis",
        ):
            assert key in gb, f"Missing gas component: {key!r}"

    def test_alternates_limit_default_five(self, client, sample_factor_id):
        _set_tier("pro")
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/explain")
        assert resp.status_code == 200
        # With a single-factor engine, alternates is 0; limit parameter
        # still must be honoured as a cap, not a floor.
        assert len(resp.json()["alternates"]) <= 5

    def test_custom_method_profile_override_echoes_in_header(
        self, client, sample_factor_id
    ):
        """If the override is compatible with the factor, the 200 echoes it.

        If incompatible (method-pack selection rule rejects the factor),
        the endpoint returns 422 rather than a mis-resolved payload — we
        accept either as correct behaviour as long as the server doesn't
        return a 500.
        """
        _set_tier("pro")
        resp = client.get(
            f"/api/v1/factors/{sample_factor_id}/explain",
            params={"method_profile": "corporate_scope1"},
        )
        assert resp.status_code in (200, 422)
        if resp.status_code == 200:
            assert (
                resp.headers["X-GreenLang-Method-Profile"] == "corporate_scope1"
            )


class TestExplainEndpointErrors:
    def test_404_for_unknown_factor_id(self, client):
        _set_tier("pro")
        resp = client.get("/api/v1/factors/EF:NOPE:xxx:9999:v0/explain")
        assert resp.status_code == 404

    def test_403_for_community_tier(self, client, sample_factor_id):
        _set_tier("community")
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/explain")
        assert resp.status_code == 403

    @pytest.mark.parametrize("tier", ["pro", "consulting", "enterprise", "internal"])
    def test_200_for_allowed_tiers(self, client, sample_factor_id, tier):
        _set_tier(tier)
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/explain")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /resolve-explain
# ---------------------------------------------------------------------------


class TestResolveExplainEndpoint:
    def _base_body(self) -> Dict[str, Any]:
        return {
            "activity": "diesel combustion stationary",
            "method_profile": "corporate_scope1",
            "jurisdiction": "US",
            "reporting_date": "2026-06-01",
        }

    def test_200_with_full_activity_context(self, client):
        _set_tier("pro")
        resp = client.post("/api/v1/factors/resolve-explain", json=self._base_body())
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["chosen_factor_id"]
        assert body["method_profile"] == "corporate_scope1"
        assert 1 <= body["fallback_rank"] <= 7
        assert "alternates" in body
        assert "gas_breakdown" in body
        assert "uncertainty" in body

    def test_alternates_limit_clamped_to_twenty(self, client):
        _set_tier("pro")
        # Request 999, must be clamped.  FastAPI's `le=20` validator will
        # reject at the edge; we test the clamp logic at 20 exactly.
        resp = client.post(
            "/api/v1/factors/resolve-explain",
            json=self._base_body(),
            params={"limit": 20},
        )
        assert resp.status_code == 200
        assert len(resp.json()["alternates"]) <= 20

    def test_alternates_limit_rejected_above_twenty(self, client):
        _set_tier("pro")
        resp = client.post(
            "/api/v1/factors/resolve-explain",
            json=self._base_body(),
            params={"limit": 21},
        )
        # FastAPI returns 422 for out-of-range query params.
        assert resp.status_code == 422

    def test_headers_edition_and_method_profile(self, client):
        _set_tier("pro")
        resp = client.post("/api/v1/factors/resolve-explain", json=self._base_body())
        assert resp.status_code == 200
        assert resp.headers["X-GreenLang-Edition"] == "test-explain-v1"
        assert resp.headers["X-GreenLang-Method-Profile"] == "corporate_scope1"

    def test_400_for_invalid_body(self, client):
        _set_tier("pro")
        # Missing required method_profile field.
        resp = client.post(
            "/api/v1/factors/resolve-explain",
            json={"activity": "diesel"},
        )
        assert resp.status_code == 400

    def test_400_for_blank_activity(self, client):
        _set_tier("pro")
        resp = client.post(
            "/api/v1/factors/resolve-explain",
            json={"activity": "   ", "method_profile": "corporate_scope1"},
        )
        assert resp.status_code == 400

    def test_403_for_community_tier(self, client):
        _set_tier("community")
        resp = client.post("/api/v1/factors/resolve-explain", json=self._base_body())
        assert resp.status_code == 403

    def test_gas_breakdown_separated(self, client):
        """Same CTO non-negotiable as GET /explain."""
        _set_tier("pro")
        resp = client.post("/api/v1/factors/resolve-explain", json=self._base_body())
        assert resp.status_code == 200
        gb = resp.json()["gas_breakdown"]
        for key in (
            "co2_kg",
            "ch4_kg",
            "n2o_kg",
            "hfcs_kg",
            "pfcs_kg",
            "sf6_kg",
            "nf3_kg",
            "biogenic_co2_kg",
            "co2e_total_kg",
        ):
            assert key in gb


# ---------------------------------------------------------------------------
# GET /{factor_id}/alternates
# ---------------------------------------------------------------------------


class TestAlternatesEndpoint:
    def test_200_for_valid_factor(self, client, sample_factor_id):
        _set_tier("pro")
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/alternates")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["factor_id"] == sample_factor_id
        assert "alternates" in body
        assert isinstance(body["alternates"], list)

    def test_404_for_unknown_factor(self, client):
        _set_tier("pro")
        resp = client.get("/api/v1/factors/EF:NOPE:xxx:9999:v0/alternates")
        assert resp.status_code == 404

    def test_403_for_community_tier(self, client, sample_factor_id):
        _set_tier("community")
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/alternates")
        assert resp.status_code == 403

    def test_limit_query_clamped(self, client, sample_factor_id):
        _set_tier("pro")
        resp = client.get(
            f"/api/v1/factors/{sample_factor_id}/alternates",
            params={"limit": 20},
        )
        assert resp.status_code == 200
        assert len(resp.json()["alternates"]) <= 20

    def test_limit_above_twenty_rejected(self, client, sample_factor_id):
        _set_tier("pro")
        resp = client.get(
            f"/api/v1/factors/{sample_factor_id}/alternates",
            params={"limit": 25},
        )
        assert resp.status_code == 422

    def test_etag_matched_returns_304(self, client, sample_factor_id):
        _set_tier("pro")
        r1 = client.get(f"/api/v1/factors/{sample_factor_id}/alternates")
        assert r1.status_code == 200
        etag = r1.headers["ETag"]
        r2 = client.get(
            f"/api/v1/factors/{sample_factor_id}/alternates",
            headers={"If-None-Match": etag},
        )
        assert r2.status_code == 304

    def test_headers_present(self, client, sample_factor_id):
        _set_tier("pro")
        resp = client.get(f"/api/v1/factors/{sample_factor_id}/alternates")
        assert resp.status_code == 200
        assert resp.headers["X-GreenLang-Edition"]
        assert resp.headers["X-GreenLang-Method-Profile"]


# ---------------------------------------------------------------------------
# Service-layer unit tests for the explain helpers.
# ---------------------------------------------------------------------------


class TestExplainHelpers:
    def test_clamp_alternates_limit_defaults(self):
        from greenlang.factors.api_endpoints import (
            EXPLAIN_ALTERNATES_DEFAULT,
            EXPLAIN_ALTERNATES_MAX,
            clamp_alternates_limit,
        )

        assert clamp_alternates_limit(None) == EXPLAIN_ALTERNATES_DEFAULT
        assert clamp_alternates_limit(0) == 1
        assert clamp_alternates_limit(-5) == 1
        assert clamp_alternates_limit(1000) == EXPLAIN_ALTERNATES_MAX
        assert clamp_alternates_limit(5) == 5
        assert clamp_alternates_limit("not-a-number") == EXPLAIN_ALTERNATES_DEFAULT

    def test_default_method_profile_respects_explicit_field(self):
        from greenlang.factors.api_endpoints import default_method_profile_for_factor
        from types import SimpleNamespace

        f = SimpleNamespace(method_profile="freight_iso_14083")
        assert default_method_profile_for_factor(f) == "freight_iso_14083"

    def test_default_method_profile_derives_from_scope(self):
        from greenlang.factors.api_endpoints import default_method_profile_for_factor
        from types import SimpleNamespace

        f2 = SimpleNamespace(
            method_profile=None, scope=SimpleNamespace(value="2")
        )
        assert (
            default_method_profile_for_factor(f2)
            == "corporate_scope2_location_based"
        )

    def test_build_factor_explain_returns_none_for_missing(self, memory_repo):
        from greenlang.factors.api_endpoints import build_factor_explain

        assert (
            build_factor_explain(memory_repo, "test-explain-v1", "EF:NOPE:xxx:9999:v0")
            is None
        )

    def test_build_factor_explain_alternates_clamped(self, memory_repo, sample_factor_id):
        from greenlang.factors.api_endpoints import build_factor_explain

        out = build_factor_explain(
            memory_repo, "test-explain-v1", sample_factor_id, alternates_limit=3
        )
        assert out is not None
        assert len(out["alternates"]) <= 3

    def test_compute_explain_etag_stable(self):
        from greenlang.factors.api_endpoints import compute_explain_etag

        payload = {
            "chosen_factor_id": "EF:X:1",
            "factor_version": "v1",
            "method_profile": "corporate_scope1",
            "alternates": [],
        }
        e1 = compute_explain_etag(payload, "edition-1")
        e2 = compute_explain_etag(payload, "edition-1")
        e3 = compute_explain_etag(payload, "edition-2")
        assert e1 == e2
        assert e1 != e3
        assert e1.startswith('"') and e1.endswith('"')

    def test_build_resolution_explain_from_request(self, memory_repo):
        from greenlang.factors.api_endpoints import build_resolution_explain

        out = build_resolution_explain(
            memory_repo,
            "test-explain-v1",
            {
                "activity": "diesel combustion",
                "method_profile": "corporate_scope1",
                "jurisdiction": "US",
            },
            alternates_limit=5,
        )
        assert out["method_profile"] == "corporate_scope1"
        assert out["chosen_factor_id"]
        # Non-negotiable: every gas component separate.
        for key in (
            "co2_kg",
            "ch4_kg",
            "n2o_kg",
            "hfcs_kg",
            "pfcs_kg",
            "sf6_kg",
            "nf3_kg",
            "biogenic_co2_kg",
        ):
            assert key in out["gas_breakdown"]
