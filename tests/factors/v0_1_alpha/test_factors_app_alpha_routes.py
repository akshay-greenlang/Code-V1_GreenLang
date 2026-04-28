# -*- coding: utf-8 -*-
"""
WS11-T1: alpha-mode route minimality test.

Asserts that when ``GL_FACTORS_RELEASE_PROFILE=alpha-v0.1`` is set, the
FastAPI app constructed by ``create_factors_app()`` exposes ONLY the five
alpha-allowed public endpoints (plus stock framework paths).

The five alpha-allowed endpoints (per CTO doc §19.1):
  * GET /v1/healthz   (currently /v1/health — task #13 will rename)
  * GET /v1/factors
  * GET /v1/factors/{urn}
  * GET /v1/sources
  * GET /v1/packs

Excluded from the assertion: /openapi.json, /docs, /redoc, /metrics,
HEAD-only synthetics, and any websocket / static mounts.
"""

from __future__ import annotations

import pytest


# Stock framework / observability paths we don't gate.
_FRAMEWORK_PATHS = {
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/metrics",
}

# Alpha-allowed v1 routes — the FINAL contract after task #13 landed.
# /v1/factors/{urn:path} is FastAPI's url-encoded URN capture form and is
# the canonical alpha path (URNs contain colons, which need :path).
# /api/v1/{path:path} is the 410-Gone catch-all that explicitly tells
# legacy /api/v1 callers the alpha contract uses /v1/...
_ALPHA_ALLOWED = {
    "/v1/healthz",
    "/v1/factors",
    "/v1/factors/{urn:path}",
    # Phase 2 / WS2 (2026-04-27): canonical alias resolver — same shape
    # as /v1/factors/{urn} with `urn` as the primary id.
    "/v1/factors/by-alias/{legacy_id:path}",
    "/v1/sources",
    "/v1/packs",
    "/api/v1/{path:path}",  # 410-Gone deprecation catch-all (alpha-only)
}


@pytest.fixture()
def alpha_app(monkeypatch, tmp_path):
    """Build a fresh factors app with ``alpha-v0.1`` release profile."""
    pytest.importorskip("fastapi")

    # Force alpha profile and a clean test environment.
    monkeypatch.setenv("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
    monkeypatch.setenv("GL_ENV", "test")  # avoid prod-only signing assert
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    # Seed a small catalog so the ``/v1/factors`` route can boot. Failure
    # to seed is non-fatal — the route may still mount, we only check
    # *which* routes exist, not that they return 200.
    try:
        from greenlang.factors.etl.ingest import ingest_builtin_database

        dbfile = tmp_path / "alpha_routes.sqlite"
        monkeypatch.setenv("GL_FACTORS_SQLITE_PATH", str(dbfile))
        ingest_builtin_database(dbfile, "alpha-routes-test", label="alpha-routes")
    except Exception:  # pragma: no cover — surface-only path
        pass

    from greenlang.factors.factors_app import create_factors_app

    return create_factors_app(
        enable_admin=True,
        enable_billing=True,
        enable_oem=True,
        enable_metrics=True,
    )


def _public_route_paths(app) -> set:
    """Return the set of mounted *templated* paths, ignoring framework paths."""
    paths = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if not isinstance(path, str):
            continue
        if path in _FRAMEWORK_PATHS:
            continue
        # Skip any private /internal paths.
        if path.startswith("/_"):
            continue
        paths.add(path)
    return paths


def test_alpha_only_exposes_five_endpoints(alpha_app):
    """The alpha-mode app must expose ONLY the five always-on endpoints."""
    public = _public_route_paths(alpha_app)
    unexpected = public - _ALPHA_ALLOWED
    assert not unexpected, (
        "alpha-v0.1 mode leaked non-alpha routes: "
        f"{sorted(unexpected)}. Expected only {sorted(_ALPHA_ALLOWED)}."
    )


def test_alpha_keeps_health_and_factors_endpoints(alpha_app):
    """Alpha mode must still expose healthz + factors lookups by URN."""
    public = _public_route_paths(alpha_app)
    assert "/v1/healthz" in public, public
    assert "/v1/factors/{urn:path}" in public, public
    assert "/v1/factors" in public, public
    assert "/v1/sources" in public, public
    assert "/v1/packs" in public, public


def test_alpha_api_v1_returns_410_gone(alpha_app):
    """Legacy /api/v1/* must return 410 Gone with the canonical alpha hint."""
    from fastapi.testclient import TestClient

    client = TestClient(alpha_app)
    resp = client.get("/api/v1/factors")
    assert resp.status_code == 410, resp.text
    body = resp.json()
    assert body["error"] == "endpoint_gone"
    assert "/v1/factors" in body["alpha_endpoints"]
    assert "/v1/healthz" in body["alpha_endpoints"]


def test_alpha_healthz_returns_release_profile(alpha_app):
    """/v1/healthz must surface release_profile + schema_id (smoke)."""
    from fastapi.testclient import TestClient

    client = TestClient(alpha_app)
    resp = client.get("/v1/healthz")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["service"] == "greenlang-factors"
    assert body["release_profile"] == "alpha-v0.1"
    assert "factor_record_v0_1" in body["schema_id"]


def test_alpha_does_not_mount_resolve(alpha_app):
    """Beta-only resolve / batch / coverage / fqs / edition routes are gone."""
    public = _public_route_paths(alpha_app)
    forbidden_prefixes = (
        "/v1/resolve",
        "/v1/batch",
        "/v1/coverage",
        "/v1/quality/fqs",
        "/v1/editions",
        "/v1/explain",
        "/v1/admin",
        "/v1/billing",
        "/v1/oem",
        "/v1/graphql",
    )
    leaked = [
        p for p in public
        if any(p == pre or p.startswith(pre + "/") or p.startswith(pre + "{") for pre in forbidden_prefixes)
    ]
    assert not leaked, (
        f"alpha-v0.1 mode leaked beta+ routes: {leaked}"
    )
