# -*- coding: utf-8 -*-
"""
Wave C / WS4-T2 — full behavioural contract for the v0.1 Alpha API.

Hardens the Wave A/B route-table check
(``test_factors_app_alpha_routes.py``) into a richer, behavioural contract
that asserts:

  * the OpenAPI spec exposes EXACTLY the 5 alpha endpoints (+ 410 catch-all)
    and nothing else (no resolve / explain / batch / coverage / fqs / edition
    / admin / billing / oem / graphql / method-pack routes);
  * every alpha operation is HTTP GET only (alpha is read-only);
  * every alpha 200 carries a typed response model registered in
    ``components.schemas``;
  * /v1/healthz is reachable un-authenticated;
  * 404/400 errors are returned with the stable v0.1 error body shape
    (top-level ``error`` / ``message`` / ``urn`` keys, NOT FastAPI's default
    ``{"detail": ...}`` wrapper);
  * the path converter accepts both raw and percent-encoded URNs;
  * /v1/sources reflects exactly the ``alpha_v0_1: true`` rows from
    ``data/source_registry.yaml`` (currently 6);
  * /v1/packs filtering by ``source_urn`` works;
  * /api/v1/* is hard-410-Goned for every HTTP method (alpha is read-only,
    no legacy POST/PUT/DELETE migration grace path);
  * resolve / explain / batch / method-packs endpoints are 404 under alpha;
  * every successful response advertises ``X-GL-Release-Profile: alpha-v0.1``.

Plus an OpenAPI snapshot test that fails when the live spec drifts from the
checked-in fixture, with a clear regenerate hint.

Authentication note:
    The factors app's ``AuthMeteringMiddleware`` protects ``/v1/*`` paths
    other than the listed PUBLIC_PATHS (``/v1/healthz``, ``/v1/health``).
    For every test below that hits a protected route, the ``alpha_client``
    fixture installs a community-tier API key via ``GL_FACTORS_API_KEYS``.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Constants — the WS4 frozen contract.
# ---------------------------------------------------------------------------

# The 5 alpha endpoints, as they appear in the OpenAPI spec. FastAPI
# renders ``{urn:path}`` as ``{urn}`` in OpenAPI (the converter is dropped),
# so the spec key for the URN-lookup route is ``/v1/factors/{urn}``.
ALPHA_OPENAPI_PATHS = {
    "/v1/healthz",
    "/v1/factors",
    "/v1/factors/{urn}",
    # Phase 2 / WS2 (2026-04-27): canonical alias resolver. Returns the
    # same FactorV0_1 shape as /v1/factors/{urn}, with `urn` as the
    # primary id and `factor_id_alias` as a sibling.
    "/v1/factors/by-alias/{legacy_id}",
    "/v1/sources",
    "/v1/packs",
}

# /api/v1/{path:path} catch-all is mounted with ``include_in_schema=False``
# so the deprecated path does NOT appear in OpenAPI. The taskcreate spec
# notes this divergence is acceptable: keeping it out of the spec means
# alpha clients see a clean 5-endpoint surface.
ALPHA_OPENAPI_PATHS_PLUS_DEPRECATED = ALPHA_OPENAPI_PATHS  # alias for clarity

ALPHA_PROFILE = "alpha-v0.1"

# Snapshot fixture path (Deliverable 3).
SNAPSHOT_PATH = Path(__file__).parent / "openapi_alpha_v0_1.json"

# Common test API key — populated into ``GL_FACTORS_API_KEYS`` so the
# auth middleware lets us through to the protected routes.
_TEST_API_KEY = "gl_alpha_contract_test_key_1"
_TEST_API_KEYS_JSON = json.dumps([
    {
        "key_id": "alpha-contract-test",
        "key": _TEST_API_KEY,
        "tier": "community",
        "tenant_id": "tenant-alpha-test",
        "user_id": "ws4-test-user",
        "active": True,
    }
])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def alpha_app(monkeypatch, tmp_path):
    """Build a fresh factors app under ``GL_FACTORS_RELEASE_PROFILE=alpha-v0.1``."""
    pytest.importorskip("fastapi")

    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", ALPHA_PROFILE)
    monkeypatch.setenv("GL_ENV", "test")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    # Install a test API key so the auth middleware lets us through.
    monkeypatch.setenv("GL_FACTORS_API_KEYS", _TEST_API_KEYS_JSON)

    # Force the default validator to reload the new env var. Tests that
    # build the app must see the freshly-injected key.
    try:
        from greenlang.factors import api_auth as _api_auth
        _api_auth._default_validator = None  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - import-only
        pass

    # Seed a small SQLite catalog so /v1/factors and /v1/factors/{urn}
    # have data to return. Failure is non-fatal — most contract checks are
    # surface-shape, not data-shape.
    try:
        from greenlang.factors.etl.ingest import ingest_builtin_database

        dbfile = tmp_path / "alpha_contract.sqlite"
        monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))
        ingest_builtin_database(dbfile, "alpha-contract-test", label="alpha-contract")
    except Exception:  # pragma: no cover - surface-only path
        pass

    from greenlang.factors.factors_app import create_factors_app

    return create_factors_app(
        enable_admin=True,
        enable_billing=True,
        enable_oem=True,
        enable_metrics=True,
    )


@pytest.fixture()
def alpha_client(alpha_app):
    """TestClient that auto-attaches the test API key on every call."""
    from fastapi.testclient import TestClient

    return TestClient(alpha_app, headers={"X-API-Key": _TEST_API_KEY})


@pytest.fixture()
def openapi_spec(alpha_client):
    """Live OpenAPI spec served by the alpha app."""
    resp = alpha_client.get("/openapi.json")
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# 1. OpenAPI surface — exactly the 5 alpha endpoints.
# ---------------------------------------------------------------------------


def test_openapi_documents_only_5_alpha_endpoints(openapi_spec):
    """OpenAPI ``paths`` must equal exactly the 5 alpha endpoints.

    The ``/api/v1/{path:path}`` 410-Gone catch-all is mounted with
    ``include_in_schema=False`` and therefore MUST NOT appear in OpenAPI.
    """
    paths = set(openapi_spec.get("paths", {}).keys())
    assert paths == ALPHA_OPENAPI_PATHS, (
        f"OpenAPI alpha surface drift. Got {sorted(paths)}, "
        f"expected {sorted(ALPHA_OPENAPI_PATHS)}."
    )


def test_openapi_alpha_endpoints_all_get_only(openapi_spec):
    """Every operation under the 5 alpha paths must be HTTP GET."""
    allowed_methods = {"get", "parameters"}  # 'parameters' is a path-level key
    forbidden = {"post", "put", "patch", "delete"}

    for path, ops in openapi_spec.get("paths", {}).items():
        if path not in ALPHA_OPENAPI_PATHS:
            continue
        op_methods = set(ops.keys()) - {"parameters"}
        assert op_methods == {"get"}, (
            f"Path {path} declares non-GET operations {op_methods}; "
            "alpha is read-only."
        )
        for method in forbidden:
            assert method not in ops, (
                f"Path {path} declares forbidden method {method!r} in alpha."
            )


def test_openapi_responses_carry_typed_models(openapi_spec):
    """Each alpha endpoint declares a 200 response with a $ref into components.schemas."""
    components = openapi_spec.get("components", {}).get("schemas", {})
    assert components, "OpenAPI components.schemas is empty"

    for path in ALPHA_OPENAPI_PATHS:
        ops = openapi_spec["paths"][path]
        get_op = ops["get"]
        responses = get_op.get("responses", {})
        assert "200" in responses, f"{path} has no 200 response"
        content = responses["200"].get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})

        # Either a direct $ref, or a $ref nested under allOf/oneOf/items.
        ref = schema.get("$ref")
        if not ref:
            for nest_key in ("allOf", "oneOf", "anyOf"):
                if nest_key in schema and schema[nest_key]:
                    ref = schema[nest_key][0].get("$ref")
                    if ref:
                        break
        assert ref, f"{path} 200 response has no $ref schema: {schema!r}"

        # $ref must resolve to a real schema in components.schemas.
        assert ref.startswith("#/components/schemas/"), (
            f"{path} 200 response $ref is not local: {ref!r}"
        )
        schema_name = ref.split("/")[-1]
        assert schema_name in components, (
            f"{path} 200 response references unknown schema {schema_name!r}"
        )


# ---------------------------------------------------------------------------
# 2. Behaviour — /v1/healthz is unauthenticated.
# ---------------------------------------------------------------------------


def test_v1_healthz_returns_200_unauthenticated(alpha_app):
    """Health probes MUST work without any credentials. Liveness must
    never be gated by tenant auth — kube-probe is a happy path."""
    from fastapi.testclient import TestClient

    # No headers → no API key → must still be 200.
    bare_client = TestClient(alpha_app)
    resp = bare_client.get("/v1/healthz")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "greenlang-factors"
    assert body["release_profile"] == ALPHA_PROFILE


# ---------------------------------------------------------------------------
# 3. Errors — typed body, no FastAPI ``{"detail": ...}`` wrapper.
# ---------------------------------------------------------------------------


def test_v1_factors_404_returns_typed_error_with_urn(alpha_client):
    """404 responses must carry the stable v0.1 error shape at top-level.

    Specifically: ``{"error": "factor_not_found", "urn": "...", "message": "..."}``
    NOT FastAPI's default ``{"detail": {...}}`` envelope.
    """
    bogus_urn = "urn:gl:factor:nope:nope:nope:v1"
    resp = alpha_client.get(f"/v1/factors/{bogus_urn}")
    assert resp.status_code == 404, resp.text
    body = resp.json()

    # Top-level keys, no FastAPI ``detail`` wrapper.
    assert "detail" not in body, (
        f"404 leaks FastAPI's default detail wrapper: {body!r}"
    )
    assert body["error"] == "factor_not_found"
    assert body["urn"] == bogus_urn
    assert isinstance(body["message"], str) and body["message"]


def test_v1_factors_400_on_bad_category_filter(alpha_client):
    """Invalid ``category`` query param returns 400 with the alpha enum."""
    resp = alpha_client.get("/v1/factors?category=invalid")
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "detail" not in body
    assert body["error"] == "invalid_category"
    assert "invalid" in body["message"]
    assert isinstance(body["allowed"], list)
    assert "scope1" in body["allowed"]
    assert "fuel" in body["allowed"]


# ---------------------------------------------------------------------------
# 4. Path converter — raw vs URL-encoded URN are equivalent.
# ---------------------------------------------------------------------------


def test_v1_factors_url_encoded_urn_roundtrip(alpha_client):
    """``GET /v1/factors/{percent_encoded_urn}`` must resolve the same factor as
    ``GET /v1/factors/{raw_urn}``. FastAPI's ``{urn:path}`` converter
    captures both forms identically."""
    # Pick a real factor by listing first.
    list_resp = alpha_client.get("/v1/factors?limit=10")
    assert list_resp.status_code == 200, list_resp.text
    rows = list_resp.json().get("data", [])
    if not rows:
        pytest.skip("Catalog empty — no factor to roundtrip.")

    # Prefer the legacy alias (these are the only ones the SQLite repo
    # actually keys on). Falling back to the synthesized URN would be
    # 404 by design (the synthesized URN isn't a primary key).
    target = rows[0].get("factor_id_alias") or rows[0]["urn"]

    raw = alpha_client.get(f"/v1/factors/{target}")
    encoded = alpha_client.get(
        f"/v1/factors/{urllib.parse.quote(target, safe='')}"
    )

    assert raw.status_code == encoded.status_code, (
        f"raw={raw.status_code} encoded={encoded.status_code} for {target!r}"
    )
    if raw.status_code == 200:
        assert raw.json()["urn"] == encoded.json()["urn"], (
            "URN-encoded lookup returned a different factor than raw lookup."
        )


# ---------------------------------------------------------------------------
# 5. Sources / Packs — registry-driven contract.
# ---------------------------------------------------------------------------


def test_v1_sources_returns_only_alpha_v0_1_sources(alpha_client):
    """``/v1/sources`` reflects exactly the ``alpha_v0_1: true`` rows from
    the source registry (currently 6: ipcc_2006_nggi, desnz_ghg_conversion,
    epa_hub, egrid, india_cea_co2_baseline, cbam_default_values)."""
    from greenlang.factors.source_registry import alpha_v0_1_sources

    expected = alpha_v0_1_sources()  # registry-truth
    resp = alpha_client.get("/v1/sources")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == len(expected), (
        f"/v1/sources returned {body['count']} rows, registry has "
        f"{len(expected)} alpha_v0_1=true rows."
    )

    returned_ids = {row["source_id"] for row in body["data"]}
    assert returned_ids == set(expected.keys()), (
        f"source_id set drift: got={sorted(returned_ids)} "
        f"expected={sorted(expected.keys())}"
    )


def test_v1_packs_filter_by_source_urn(alpha_client):
    """``?source_urn=...`` filters /v1/packs to a single source."""
    # Resolve a real source URN from the registry rather than guessing
    # at the slugged form.
    sources = alpha_client.get("/v1/sources").json()["data"]
    assert sources, "registry returned no alpha sources"
    target_urn = sources[0]["urn"]

    full = alpha_client.get("/v1/packs").json()
    filtered = alpha_client.get(f"/v1/packs?source_urn={target_urn}").json()

    assert filtered["count"] >= 1
    assert filtered["count"] < full["count"] or full["count"] == 1
    for pack in filtered["data"]:
        assert pack["source_urn"] == target_urn, (
            f"filter leaked pack with source_urn={pack['source_urn']!r} "
            f"(expected {target_urn!r})."
        )


# ---------------------------------------------------------------------------
# 6. /api/v1 catch-all — 410 Gone for every method.
# ---------------------------------------------------------------------------


def test_api_v1_legacy_path_returns_410_with_alpha_endpoints_list(alpha_client):
    """Legacy /api/v1/* GET → 410, body advertises the 5 alpha endpoints."""
    resp = alpha_client.get("/api/v1/factors")
    assert resp.status_code == 410, resp.text
    body = resp.json()
    assert body["error"] == "endpoint_gone"
    listed = set(body["alpha_endpoints"])
    expected = {
        "/v1/healthz",
        "/v1/factors",
        "/v1/factors/{urn}",
        # Phase 2 / WS2 (2026-04-27): alias resolver added to the public
        # endpoint catalogue. The router exposes the same FactorV0_1
        # response shape as /v1/factors/{urn}.
        "/v1/factors/by-alias/{legacy_id}",
        "/v1/sources",
        "/v1/packs",
    }
    assert listed == expected, (
        f"410 body advertises wrong endpoints: {listed} vs {expected}"
    )
    assert body["requested_path"] == "/api/v1/factors"


@pytest.mark.parametrize(
    "method", ["post", "put", "patch", "delete"]
)
def test_api_v1_legacy_post_also_410(alpha_client, method):
    """Non-GET methods to /api/v1/* must also return 410. Alpha is read-only —
    we never want a legacy client thinking POST is a graceful path."""
    fn = getattr(alpha_client, method)
    resp = fn("/api/v1/factors")
    assert resp.status_code == 410, (
        f"/api/v1/factors via {method.upper()} returned {resp.status_code} "
        f"(expected 410). Body: {resp.text[:200]}"
    )
    assert resp.json()["error"] == "endpoint_gone"


# ---------------------------------------------------------------------------
# 7. Beta+ surfaces are 404 under alpha.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/v1/resolve",
        "/v1/resolve/diesel",
    ],
)
def test_resolve_endpoint_returns_404_under_alpha_profile(alpha_client, path):
    """GET /v1/resolve and POST /v1/resolve are 404 under alpha."""
    get_resp = alpha_client.get(path)
    post_resp = alpha_client.post(path, json={})
    assert get_resp.status_code == 404, (
        f"GET {path} returned {get_resp.status_code} under alpha (expected 404)."
    )
    assert post_resp.status_code == 404, (
        f"POST {path} returned {post_resp.status_code} under alpha (expected 404)."
    )


def test_explain_endpoint_returns_404(alpha_client):
    """/v1/explain* is 404 under alpha (explain surface is beta-v0.5+)."""
    for p in ("/v1/explain", "/v1/factors/EF:US:diesel:2024:v1/explain"):
        resp = alpha_client.get(p)
        assert resp.status_code == 404, (
            f"{p} returned {resp.status_code} under alpha (expected 404)."
        )


def test_batch_endpoint_returns_404(alpha_client):
    """/v1/batch is 404 under alpha (batch surface is beta-v0.5+)."""
    resp = alpha_client.post("/v1/batch", json={"factor_ids": []})
    assert resp.status_code == 404
    resp = alpha_client.get("/v1/batch")
    assert resp.status_code == 404


def test_method_packs_endpoint_returns_404(alpha_client):
    """/v1/method-packs is 404 under alpha — coverage surface is beta+."""
    for p in (
        "/v1/method-packs",
        "/v1/method-packs/coverage",
        "/v1/coverage",
        "/v1/quality/fqs",
        "/v1/editions/builtin-v1.0.0",
    ):
        resp = alpha_client.get(p)
        assert resp.status_code == 404, (
            f"{p} returned {resp.status_code} under alpha (expected 404)."
        )


# ---------------------------------------------------------------------------
# 8. Release-profile header on every successful response.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/v1/healthz",
        "/v1/factors",
        "/v1/sources",
        "/v1/packs",
    ],
)
def test_alpha_response_includes_release_profile_header(alpha_client, path):
    """Every successful alpha response advertises ``X-GL-Release-Profile``."""
    resp = alpha_client.get(path)
    assert resp.status_code == 200, resp.text
    profile = resp.headers.get("X-GL-Release-Profile")
    assert profile == ALPHA_PROFILE, (
        f"{path} did not stamp X-GL-Release-Profile (got {profile!r})."
    )


def test_alpha_release_profile_header_on_410(alpha_client):
    """The 410 catch-all also advertises the release profile so legacy
    clients can detect they hit a profile-locked surface."""
    resp = alpha_client.get("/api/v1/factors")
    assert resp.status_code == 410
    assert resp.headers.get("X-GL-Release-Profile") == ALPHA_PROFILE


def test_alpha_release_profile_header_on_400(alpha_client):
    """The typed 400 also advertises the release profile."""
    resp = alpha_client.get("/v1/factors?category=invalid")
    assert resp.status_code == 400
    assert resp.headers.get("X-GL-Release-Profile") == ALPHA_PROFILE


def test_alpha_release_profile_header_on_404(alpha_client):
    """The typed 404 also advertises the release profile."""
    resp = alpha_client.get("/v1/factors/urn:gl:factor:nope:nope:nope:v1")
    assert resp.status_code == 404
    assert resp.headers.get("X-GL-Release-Profile") == ALPHA_PROFILE


# ---------------------------------------------------------------------------
# 9. OpenAPI snapshot test — fail loudly on drift.
# ---------------------------------------------------------------------------


def _normalize_spec(spec: dict) -> dict:
    """Strip volatile fields (server-binding URLs, build SHA description tail).

    The OpenAPI spec embeds the FastAPI ``servers`` block dynamically per
    request host; that's irrelevant to the contract. Everything else
    (paths, operations, schemas) is the contract surface.
    """
    out = dict(spec)
    out.pop("servers", None)
    return out


def test_openapi_alpha_v0_1_matches_snapshot(openapi_spec):
    """OpenAPI alpha spec must match the checked-in snapshot byte-for-byte
    (after dropping the volatile ``servers`` block).

    To regenerate after an intentional contract change:

        pytest tests/factors/v0_1_alpha/test_alpha_api_contract.py \\
            --update-openapi-snapshot

    or set the env var ``UPDATE_OPENAPI_SNAPSHOT=1`` once and re-run.
    """
    live = _normalize_spec(openapi_spec)

    update_flag = (
        os.environ.get("UPDATE_OPENAPI_SNAPSHOT") == "1"
        or os.environ.get("PYTEST_UPDATE_OPENAPI_SNAPSHOT") == "1"
    )

    if update_flag or not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(live, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if update_flag:
            pytest.skip(
                f"openapi snapshot regenerated at {SNAPSHOT_PATH}. "
                "Re-run without UPDATE_OPENAPI_SNAPSHOT=1 to validate."
            )

    saved = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert live == saved, (
        "OpenAPI alpha spec drifted from the saved snapshot.\n\n"
        f"Snapshot path: {SNAPSHOT_PATH}\n"
        "If this drift is intentional, regenerate the snapshot via:\n"
        "    UPDATE_OPENAPI_SNAPSHOT=1 pytest "
        "tests/factors/v0_1_alpha/test_alpha_api_contract.py "
        "::test_openapi_alpha_v0_1_matches_snapshot\n"
    )
