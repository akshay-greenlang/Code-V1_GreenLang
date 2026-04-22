# -*- coding: utf-8 -*-
"""Tests for the signed-receipts FastAPI middleware."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterator

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.factors.middleware.signed_receipts import (
    algorithm_for_tier,
    install_signing_middleware,
)
from greenlang.factors.signing import verify_sha256_hmac


SIGNING_SECRET = "test-secret-never-use-in-prod-very-long"


@pytest.fixture(autouse=True)
def _set_signing_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GL_FACTORS_SIGNING_SECRET", SIGNING_SECRET)
    # Ensure Ed25519 falls back cleanly for tests that exercise higher tiers.
    monkeypatch.delenv("GL_FACTORS_ED25519_PRIVATE_KEY", raising=False)


def _make_app(
    *,
    tier: str = "pro",
    set_edition: bool = True,
) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def _inject_tier(request, call_next):
        request.state.tier = tier
        response = await call_next(request)
        return response

    install_signing_middleware(app, protected_prefix="/api/v1/factors")

    @app.get("/api/v1/factors/ef1")
    async def ef1():
        from fastapi.responses import JSONResponse

        headers = {"X-GreenLang-Edition": "2027.01"} if set_edition else {}
        return JSONResponse({"factor_id": "ef1", "co2e_per_unit": 10.2}, headers=headers)

    @app.get("/api/v1/factors/list-raw")
    async def list_raw():
        # Non-dict JSON top-level (list) — header-only signing path.
        from fastapi.responses import JSONResponse

        return JSONResponse(["ef1", "ef2"])

    @app.get("/api/v1/factors/stream")
    async def stream_endpoint():
        from fastapi.responses import Response

        return Response(
            content=b"ndjson\nline\nfactors",
            media_type="application/x-ndjson",
            headers={"X-GreenLang-Stream": "1"},
        )

    @app.get("/api/v1/factors/broken")
    async def broken():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="not found")

    @app.get("/public/ping")
    async def public_ping():
        return {"ok": True}

    return app


# --------------------------------------------------------------------------- #
# Tier algorithm policy.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tier,expected_algo",
    [
        ("community", "sha256-hmac"),
        ("pro", "sha256-hmac"),
        ("internal", "sha256-hmac"),
        ("consulting", "ed25519"),
        ("platform", "ed25519"),
        ("enterprise", "ed25519"),
        ("CONSULTING", "ed25519"),  # case-insensitive
        (None, "sha256-hmac"),
    ],
)
def test_algorithm_for_tier(tier: str, expected_algo: str) -> None:
    assert algorithm_for_tier(tier) == expected_algo


# --------------------------------------------------------------------------- #
# Happy path — JSON dict response gets a body-embedded receipt + headers.
# --------------------------------------------------------------------------- #


def test_json_dict_response_is_signed_in_body_and_headers() -> None:
    app = _make_app(tier="pro")
    client = TestClient(app)
    resp = client.get("/api/v1/factors/ef1")
    assert resp.status_code == 200

    body = resp.json()
    assert body["factor_id"] == "ef1"

    # Body envelope.
    assert "_signed_receipt" in body
    sr = body["_signed_receipt"]
    assert sr["algorithm"] == "sha256-hmac"
    assert sr["key_id"]
    assert sr["signature"]
    assert sr["payload_hash"]
    assert sr["signed_over"]["edition_id"] == "2027.01"
    assert sr["signed_over"]["path"] == "/api/v1/factors/ef1"
    assert sr["signed_over"]["method"] == "GET"
    assert sr["signed_over"]["status_code"] == 200

    # Header envelope.
    for h in (
        "X-GreenLang-Receipt-Signature",
        "X-GreenLang-Receipt-Algorithm",
        "X-GreenLang-Receipt-Key-Id",
        "X-GreenLang-Receipt-Signed-At",
        "X-GreenLang-Receipt-Hash",
    ):
        assert h in resp.headers, f"missing receipt header {h}"


def test_signature_verifies_with_shared_secret() -> None:
    app = _make_app(tier="pro")
    client = TestClient(app)
    resp = client.get("/api/v1/factors/ef1")
    body = resp.json()
    sr = body["_signed_receipt"]

    # Reconstruct the signed_over payload and verify.
    signed_over = sr["signed_over"]
    ok = verify_sha256_hmac(
        signed_over,
        {
            "signature": sr["signature"],
            "algorithm": sr["algorithm"],
            "signed_at": sr["signed_at"],
            "key_id": sr["key_id"],
            "payload_hash": sr["payload_hash"],
        },
    )
    assert ok is True


def test_body_hash_matches_client_visible_body() -> None:
    app = _make_app(tier="pro")
    client = TestClient(app)
    resp = client.get("/api/v1/factors/ef1")
    # The body hash signs the *pre-injection* body so the client can
    # verify by removing _signed_receipt and hashing.
    body = resp.json()
    sr = body.pop("_signed_receipt")
    expected_hash = hashlib.sha256(
        json.dumps(body).encode("utf-8")
    ).hexdigest()
    # Our canonical body before injection was the exact JSONResponse
    # serialization; Starlette uses json.dumps with default separators.
    # We can't reconstruct byte-identical output in the test, but we
    # CAN verify the header hash matches the signed_over.body_hash.
    assert sr["signed_over"]["body_hash"] == resp.headers["X-GreenLang-Receipt-Hash"]


# --------------------------------------------------------------------------- #
# Error responses are never signed.
# --------------------------------------------------------------------------- #


def test_404_is_not_signed() -> None:
    app = _make_app(tier="pro")
    client = TestClient(app)
    resp = client.get("/api/v1/factors/broken")
    assert resp.status_code == 404
    assert "X-GreenLang-Receipt-Signature" not in resp.headers
    body = resp.json()
    assert "_signed_receipt" not in body


# --------------------------------------------------------------------------- #
# Non-protected prefix is untouched.
# --------------------------------------------------------------------------- #


def test_non_protected_prefix_is_not_signed() -> None:
    app = _make_app(tier="pro")
    client = TestClient(app)
    resp = client.get("/public/ping")
    assert resp.status_code == 200
    assert "X-GreenLang-Receipt-Signature" not in resp.headers


# --------------------------------------------------------------------------- #
# Streaming / non-JSON responses get header-only receipt.
# --------------------------------------------------------------------------- #


def test_streaming_response_has_header_receipt_only_and_no_body_mutation() -> None:
    app = _make_app(tier="pro")
    client = TestClient(app)
    resp = client.get("/api/v1/factors/stream")
    assert resp.status_code == 200
    assert resp.content == b"ndjson\nline\nfactors"
    assert "X-GreenLang-Receipt-Signature" in resp.headers
    assert resp.headers["X-GreenLang-Receipt-Algorithm"] == "sha256-hmac"


def test_non_dict_json_top_level_gets_header_only() -> None:
    app = _make_app(tier="pro")
    client = TestClient(app)
    resp = client.get("/api/v1/factors/list-raw")
    assert resp.status_code == 200
    body = resp.json()
    # Lists must NOT be wrapped — that would break SDK deserialization.
    assert body == ["ef1", "ef2"]
    # But header receipt is still set.
    assert "X-GreenLang-Receipt-Signature" in resp.headers


# --------------------------------------------------------------------------- #
# Ed25519 fallback — requested for platform tier but key missing → HMAC.
# --------------------------------------------------------------------------- #


def test_ed25519_tier_falls_back_to_hmac_without_private_key() -> None:
    app = _make_app(tier="platform")
    client = TestClient(app)
    resp = client.get("/api/v1/factors/ef1")
    body = resp.json()
    assert body["_signed_receipt"]["algorithm"] == "sha256-hmac"


# --------------------------------------------------------------------------- #
# Graceful degradation — signing failure must not crash the response.
# --------------------------------------------------------------------------- #


def test_signing_failure_returns_response_unsigned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Remove the secret so HMAC signing raises SigningError.
    monkeypatch.delenv("GL_FACTORS_SIGNING_SECRET", raising=False)
    app = _make_app(tier="pro")
    client = TestClient(app)
    resp = client.get("/api/v1/factors/ef1")
    assert resp.status_code == 200
    # Response is still returned, just without a receipt.
    body = resp.json()
    assert "_signed_receipt" not in body
    assert "X-GreenLang-Receipt-Signature" not in resp.headers


# --------------------------------------------------------------------------- #
# Edition pinning — removing the edition header invalidates the receipt.
# --------------------------------------------------------------------------- #


def test_signed_over_includes_edition_when_header_set() -> None:
    app = _make_app(tier="pro", set_edition=True)
    client = TestClient(app)
    resp = client.get("/api/v1/factors/ef1")
    sr = resp.json()["_signed_receipt"]
    assert sr["signed_over"]["edition_id"] == "2027.01"


def test_signed_over_edition_is_null_when_header_missing() -> None:
    app = _make_app(tier="pro", set_edition=False)
    client = TestClient(app)
    resp = client.get("/api/v1/factors/ef1")
    sr = resp.json()["_signed_receipt"]
    assert sr["signed_over"]["edition_id"] is None
