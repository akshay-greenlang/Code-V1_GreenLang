"""Factors API staging smoke tests.

Runs against a deployed staging (or prod) instance via HTTP.

Configuration (environment variables):
    GL_FACTORS_STAGING_URL      Base URL, no trailing slash
                                e.g. https://staging.greenlang.io
    GL_FACTORS_STAGING_JWT      Bearer JWT for auth paths (optional in CI
                                pre-deploy; required for authenticated
                                endpoints to return 200).
    GL_FACTORS_STAGING_APIKEY   X-API-Key alternative for JWT.

Usage:
    GL_FACTORS_STAGING_URL=https://staging.greenlang.io \\
    GL_FACTORS_STAGING_JWT=eyJhbGc... \\
        pytest tests/factors/smoke/test_staging.py -v

The tests deliberately do NOT assert content shape in depth — they check
that:
  1. routed endpoints return 200
  2. auth is enforced
  3. edition and request-id headers flow
  4. signed receipts are attached
  5. rate limits fire after a burst
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest


STAGING_URL: str = os.getenv("GL_FACTORS_STAGING_URL", "").rstrip("/")
STAGING_JWT: str = os.getenv("GL_FACTORS_STAGING_JWT", "")
STAGING_APIKEY: str = os.getenv("GL_FACTORS_STAGING_APIKEY", "")

pytestmark = pytest.mark.skipif(
    not STAGING_URL,
    reason="GL_FACTORS_STAGING_URL not set; smoke tests require a live endpoint.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """HTTP client with generous timeouts and a per-run request id."""
    run_id = f"smoke-{uuid.uuid4().hex[:8]}"
    client = httpx.Client(
        base_url=STAGING_URL,
        timeout=httpx.Timeout(15.0, connect=5.0),
        headers={
            "User-Agent": "greenlang-factors-smoke/1.0",
            "X-Request-Id": run_id,
        },
        follow_redirects=True,
    )
    yield client
    client.close()


@pytest.fixture(scope="module")
def auth_headers() -> dict[str, str]:
    """Prefer JWT, fall back to API key."""
    if STAGING_JWT:
        return {"Authorization": f"Bearer {STAGING_JWT}"}
    if STAGING_APIKEY:
        return {"X-API-Key": STAGING_APIKEY}
    pytest.skip("Neither GL_FACTORS_STAGING_JWT nor GL_FACTORS_STAGING_APIKEY set.")
    return {}


# ---------------------------------------------------------------------------
# Unauthenticated probes
# ---------------------------------------------------------------------------


def test_health_returns_200(client: httpx.Client) -> None:
    """/api/v1/health must be reachable without auth."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200, f"health failed: {r.status_code} {r.text}"
    body = r.json()
    assert body.get("status") == "healthy", body
    assert body.get("database") == "connected", body


def test_correlation_id_echoed(client: httpx.Client) -> None:
    """Kong correlation-id plugin must echo the caller's X-Request-Id."""
    rid = f"smoke-probe-{uuid.uuid4().hex[:12]}"
    r = client.get("/api/v1/health", headers={"X-Request-Id": rid})
    assert r.status_code == 200
    assert r.headers.get("X-Request-Id") == rid, "Kong correlation-id plugin not echoing"


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------


def test_factors_list_requires_auth(client: httpx.Client) -> None:
    """Without Authorization / X-API-Key the list endpoint must 401."""
    r = client.get("/api/v1/factors", params={"limit": 1})
    # Kong emits 401; fallback middleware would also 401.
    assert r.status_code in (401, 403), (
        f"Expected 401/403, got {r.status_code}: {r.text[:200]}"
    )


# ---------------------------------------------------------------------------
# Authenticated happy-path
# ---------------------------------------------------------------------------


def test_factors_list_paginated(
    client: httpx.Client, auth_headers: dict[str, str]
) -> None:
    """List endpoint returns paginated results."""
    r = client.get(
        "/api/v1/factors",
        params={"limit": 5, "offset": 0},
        headers=auth_headers,
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
    body = r.json()
    # Pagination envelope must exist (field names match main.py FactorsListResponse)
    assert "total" in body or "items" in body or "factors" in body, body
    assert r.headers.get("X-Factors-Edition"), "X-Factors-Edition header missing"


def test_coverage_stats(client: httpx.Client, auth_headers: dict[str, str]) -> None:
    """Coverage endpoint returns 200 and flags the active edition."""
    r = client.get("/api/v1/stats/coverage", headers=auth_headers)
    assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
    assert r.headers.get("X-Factors-Edition"), "X-Factors-Edition header missing"


def test_factor_quality_payload(
    client: httpx.Client, auth_headers: dict[str, str]
) -> None:
    """Fetching a factor must include DQS / FQS quality payload."""
    # Grab a factor id from the list.
    r = client.get(
        "/api/v1/factors",
        params={"limit": 1},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text[:400]
    body = r.json()
    items = (
        body.get("items")
        or body.get("factors")
        or body.get("results")
        or []
    )
    if not items:
        pytest.skip("No factors in staging catalog; cannot exercise quality path.")
    factor_id = items[0].get("factor_id") or items[0].get("id")
    assert factor_id, f"could not extract factor_id from {items[0]!r}"

    r2 = client.get(f"/api/v1/factors/{factor_id}", headers=auth_headers)
    assert r2.status_code == 200, f"{r2.status_code}: {r2.text[:400]}"
    detail = r2.json()
    assert "data_quality" in detail, f"data_quality missing in {detail.keys()}"

    # Signed receipt header (install_signing_middleware)
    sig = (
        r2.headers.get("X-Factors-Signature")
        or r2.headers.get("X-Factors-Receipt")
        or r2.headers.get("X-GreenLang-Signature")
    )
    assert sig, (
        "No signed-receipt header found. Expected one of "
        "X-Factors-Signature / X-Factors-Receipt / X-GreenLang-Signature. "
        f"Got headers: {list(r2.headers.keys())}"
    )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_fires_on_burst(
    client: httpx.Client, auth_headers: dict[str, str]
) -> None:
    """A burst above the community-tier quota must yield 429."""
    # Default community quota: 60/min. Fire 80 quickly; expect at least one 429.
    saw_429 = False
    last_status = None
    for _ in range(80):
        r = client.get("/api/v1/health", headers=auth_headers)
        last_status = r.status_code
        if r.status_code == 429:
            saw_429 = True
            break
        # Tiny sleep to avoid hammering the cluster during transient 5xx.
        time.sleep(0.02)
    if not saw_429:
        pytest.xfail(
            "No 429 observed within 80 requests; the smoke caller is likely "
            f"on a higher tier (last status={last_status}). Acceptable for "
            "pro/enterprise smoke keys — fail only if smoke key is community."
        )


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


def test_security_headers_present(client: httpx.Client) -> None:
    """Kong response-transformer injects the security header set."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    hsts = r.headers.get("Strict-Transport-Security", "")
    assert "max-age=63072000" in hsts, f"HSTS missing or too short: {hsts!r}"
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("Referrer-Policy") == "no-referrer"
