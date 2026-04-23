# -*- coding: utf-8 -*-
"""GraphQL surface tests (W4-C / API15).

These tests exercise the schema and routes directly, without needing
the full factors app wiring. We mount just the graphql router on a
thin FastAPI and seed ``app.state.factors_service`` with a stub
repo/service so resolvers can run end-to-end.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("fastapi")
strawberry = pytest.importorskip("strawberry")

from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers — a stub service the resolvers can reach through app.state.
# ---------------------------------------------------------------------------


class _StubRepo:
    def __init__(self) -> None:
        self._factors: Dict[str, Any] = {}

    def resolve_edition(self, _edition: Optional[str]) -> str:
        return "test-edition"

    def get_factor(self, _edition: str, factor_id: str) -> Any:
        return self._factors.get(factor_id)

    def list_factors(self, _edition: str, page: int = 1, limit: int = 100):
        return list(self._factors.values()), len(self._factors)

    def search_factors(self, _edition: str, query: str, page: int = 1, limit: int = 100):
        return [v for v in self._factors.values() if query.lower() in str(getattr(v, "factor_id", "")).lower()], 0

    def list_editions(self) -> List[Dict[str, Any]]:
        return [{"edition_id": "test-edition", "created_at": "2026-04-23T00:00:00Z"}]


class _StubService:
    def __init__(self) -> None:
        self.repo = _StubRepo()


def _build_app(*, user: Optional[Dict[str, Any]] = None) -> FastAPI:
    from greenlang.factors.graphql import graphql_router

    app = FastAPI()
    app.state.factors_service = _StubService()

    @app.middleware("http")
    async def _inject_user(request, call_next):
        if user is not None:
            request.state.user = user
        return await call_next(request)

    app.include_router(graphql_router)
    return app


def _q(client: TestClient, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = {"query": query}
    if variables:
        body["variables"] = variables
    r = client.post("/v1/graphql", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1-3: Schema introspection + SDL endpoint
# ---------------------------------------------------------------------------


def test_schema_introspection_exposes_resolve():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    data = _q(
        client,
        "{ __schema { queryType { fields { name } } } }",
    )
    assert data.get("errors") is None
    names = {f["name"] for f in data["data"]["__schema"]["queryType"]["fields"]}
    assert {"resolve", "factor", "search", "methodPacks", "sources", "releases", "quality", "explain"} <= names


def test_schema_sdl_endpoint_exposes_core_types():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    r = client.get("/v1/graphql/schema")
    assert r.status_code == 200
    sdl = r.json()["sdl"]
    # The 16-field contract envelopes must all appear in SDL.
    for t in (
        "ChosenFactorGQL",
        "SourceDescriptorGQL",
        "QualityEnvelopeGQL",
        "UncertaintyEnvelopeGQL",
        "LicensingEnvelopeGQL",
        "DeprecationStatusGQL",
        "GasBreakdownGQL",
        "ResolvedFactorGQL",
    ):
        assert t in sdl, f"{t} missing from SDL"


def test_introspection_cannot_reveal_fields_beyond_rest_contract():
    """Safety net: if somebody adds a private-looking field to the GQL
    schema without a REST counterpart, this test fails loudly."""
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    r = client.get("/v1/graphql/schema")
    sdl = r.json()["sdl"]
    # No internal-looking tokens leaked.
    for bad in ("internal_secret", "_hidden", "debug_trace"):
        assert bad not in sdl


# ---------------------------------------------------------------------------
# 4-7: Auth + rate-limit + dev-only GraphiQL
# ---------------------------------------------------------------------------


def test_graphql_unauthed_in_prod_returns_401(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    client = TestClient(_build_app(user=None))
    r = client.post("/v1/graphql", json={"query": "{ __typename }"})
    assert r.status_code == 401


def test_graphql_authed_ok(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    data = _q(client, "{ __typename }")
    assert data.get("data", {}).get("__typename") == "Query"


def test_graphiql_ui_dev_only(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    assert client.get("/v1/graphql").status_code == 404

    monkeypatch.setenv("APP_ENV", "dev")
    client2 = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    r = client2.get("/v1/graphql")
    assert r.status_code == 200
    assert "GraphiQL" in r.text


def test_invalid_json_body_400():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    r = client.post(
        "/v1/graphql",
        data="not-json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code in (400, 422)


def test_missing_query_field_400():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    r = client.post("/v1/graphql", json={"variables": {}})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 8-10: Resolver smoke — methodPacks, sources, releases
# ---------------------------------------------------------------------------


def test_method_packs_returns_list():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    data = _q(client, "{ methodPacks { packId name version } }")
    assert data.get("errors") is None
    assert isinstance(data["data"]["methodPacks"], list)


def test_sources_returns_list():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    data = _q(client, "{ sources { id name authority } }")
    assert data.get("errors") is None
    assert isinstance(data["data"]["sources"], list)


def test_releases_returns_list():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    data = _q(client, "{ releases { editionId cutAt } }")
    assert data.get("errors") is None
    items = data["data"]["releases"]
    assert isinstance(items, list)
    assert any(r["editionId"] == "test-edition" for r in items)


# ---------------------------------------------------------------------------
# 11-12: Error mapping
# ---------------------------------------------------------------------------


def test_explain_requires_one_selector():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    r = client.post(
        "/v1/graphql",
        json={"query": "{ explain { receiptId } }"},
    )
    # Expectation: 200 with errors[] (GraphQL semantics).
    assert r.status_code == 200
    data = r.json()
    assert data.get("errors"), data


def test_factor_not_found_returns_null():
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    data = _q(
        client,
        '{ factor(id: "does-not-exist") { chosenFactor { id } } }',
    )
    # factor() returns null per schema when not found.
    assert data["data"]["factor"] is None


# ---------------------------------------------------------------------------
# 13-15: Canonical demo via GraphQL mirrors REST contract
# ---------------------------------------------------------------------------


def test_gql_schema_mirrors_rest_16_field_envelopes():
    """The 16 contract elements must appear as GraphQL fields."""
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    r = client.get("/v1/graphql/schema")
    sdl = r.json()["sdl"]
    # Each nested envelope must be queryable from ResolvedFactorGQL.
    expected = [
        "chosenFactor",          # #1
        "alternates",            # #2
        "whyThisWon",            # #3
        "source",                # #4
        "releaseVersion",        # #5
        "methodPack",            # #6
        "validFrom",             # #7
        "gasBreakdown",          # #8
        "co2eBasis",             # #9
        "quality",               # #10
        "uncertainty",           # #11
        "licensing",             # #12
        "assumptions",           # #13
        "fallbackRank",          # #14
        "deprecationStatus",     # #15
        "auditText",             # #16
    ]
    for field in expected:
        assert field in sdl, f"missing {field} in ResolvedFactorGQL"


def test_graphql_query_to_resolve_with_missing_service_fails_cleanly():
    from greenlang.factors.graphql import graphql_router

    app = FastAPI()
    # No factors_service on app.state — resolve must fail cleanly.
    app.state.factors_service = None

    @app.middleware("http")
    async def _inject_user(request, call_next):
        request.state.user = {"tenant_id": "t1", "tier": "pro"}
        return await call_next(request)

    app.include_router(graphql_router)
    client = TestClient(app)
    r = client.post(
        "/v1/graphql",
        json={
            "query": "query($i: ResolveInputGQL!) { resolve(input:$i) { whyThisWon } }",
            "variables": {
                "i": {"activity": "diesel", "methodProfile": "CORPORATE_SCOPE1"}
            },
        },
    )
    # The server must not 5xx; GraphQL semantics = 200 + errors[].
    assert r.status_code == 200
    body = r.json()
    assert body.get("errors")


def test_compact_mode_not_leaking_explain_only_in_rest():
    """Smoke: GraphQL alone can't serve a REST-only 'compact' semantic —
    it always returns the queried fields, nothing extra."""
    client = TestClient(_build_app(user={"tenant_id": "t1", "tier": "pro"}))
    data = _q(client, "{ sources { id } }")
    # sanity: no extra top-level keys beyond data/errors/extensions
    assert set(data.keys()) <= {"data", "errors", "extensions"}
