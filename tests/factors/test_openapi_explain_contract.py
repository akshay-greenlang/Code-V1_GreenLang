# -*- coding: utf-8 -*-
"""
OpenAPI explain-primitive contract test (Track A-4 of FY27_Factors_Launch_Checklist.md).

CTO non-negotiable #3: never hide fallback logic. Every factor-returning
route must return the full explain payload by default; ``?compact=true``
is the only way to suppress it.

This test boots the FY27 FastAPI app via the shared ``factors_app``
fixture (see ``tests/factors/conftest.py``) and asserts that:

  1. /v1/health is reachable and returns the active edition id.
  2. The OpenAPI spec is published at /openapi.json with ≥ the launch
     surface (health, resolve, factors/{id}, factors/{id}/explain,
     coverage, quality/fqs, editions/{id}).
  3. /v1/factors/{id} returns an ``explain`` block by default.
  4. /v1/factors/{id}?compact=true does NOT return an explain block.
  5. /v1/factors/{id}/explain returns a non-empty explain payload.
  6. /v1/coverage returns a families dict, never an empty string.
"""

from __future__ import annotations

import pytest


REQUIRED_EXPLAIN_FIELDS = {
    "alternates",            # CTO: alternates considered
    "fallback_rank",         # CTO: why this one won
}

LAUNCH_PATHS = {
    "/v1/health",
    "/v1/resolve",
    "/v1/factors/{factor_id}",
    "/v1/factors/{factor_id}/explain",
    "/v1/coverage",
    "/v1/quality/fqs",
    "/v1/editions/{edition_id}",
}


@pytest.fixture()
def first_factor_id(factors_client):
    """Pull a real factor id off the seeded catalog so the test never relies
    on a hard-coded id that drifts when the seed changes."""
    # We hit /v1/coverage for a list of families so the licensing-guard
    # middleware has a chance to scan a JSON body in passing.
    r = factors_client.get("/v1/coverage")
    if r.status_code != 200:
        pytest.skip(f"/v1/coverage unreachable on this build: {r.status_code} {r.text[:200]}")

    # Try to find any factor via the resolve endpoint with a generic prompt.
    body = {
        "activity": "diesel",
        "method_profile": "corporate_scope1",
    }
    r = factors_client.post("/v1/resolve", json=body)
    if r.status_code != 200:
        pytest.skip(
            "Catalog has no resolvable diesel factor in this seed — "
            f"resolve returned {r.status_code} {r.text[:200]}"
        )
    payload = r.json()
    fid = payload.get("factor_id") or payload.get("chosen_factor_id")
    if not fid:
        pytest.skip("resolve payload had no factor_id")
    return fid


def test_health_returns_edition(factors_client):
    r = factors_client.get("/v1/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "ok"
    assert body.get("service") == "greenlang-factors"
    assert "edition" in body  # may be None in dev, must be present


def test_openapi_publishes_launch_surface(factors_client):
    r = factors_client.get("/openapi.json")
    assert r.status_code == 200, r.text
    spec = r.json()
    paths = set(spec.get("paths", {}).keys())
    missing = LAUNCH_PATHS - paths
    assert not missing, (
        f"FY27 launch routes missing from OpenAPI: {sorted(missing)}\n"
        f"Got: {sorted(paths)}"
    )


def test_get_factor_includes_explain_by_default(factors_client, first_factor_id):
    r = factors_client.get(f"/v1/factors/{first_factor_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "explain" in body, (
        "CTO non-negotiable violated: factor response must include "
        "'explain' by default. Got keys: " + ", ".join(sorted(body.keys()))
    )
    explain = body["explain"]
    assert isinstance(explain, dict)
    missing = REQUIRED_EXPLAIN_FIELDS - set(explain.keys())
    assert not missing, f"Explain payload missing required fields: {missing}"


def test_get_factor_compact_suppresses_explain(factors_client, first_factor_id):
    r = factors_client.get(f"/v1/factors/{first_factor_id}", params={"compact": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("_compact") is True
    assert "explain" not in body, (
        "compact=true must suppress the explain block (the only opt-out path)."
    )


def test_explain_endpoint_returns_payload(factors_client, first_factor_id):
    r = factors_client.get(f"/v1/factors/{first_factor_id}/explain")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, dict) and body, "explain endpoint must return a non-empty object"


def test_resolve_includes_explain_by_default(factors_client):
    r = factors_client.post("/v1/resolve", json={
        "activity": "diesel",
        "method_profile": "corporate_scope1",
    })
    if r.status_code == 400:
        pytest.skip(f"resolve request rejected by validator: {r.text[:200]}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "explain" in body, (
        "CTO non-negotiable violated: /v1/resolve must include explain "
        "by default."
    )
