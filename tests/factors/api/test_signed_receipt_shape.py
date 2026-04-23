# -*- coding: utf-8 -*-
"""
Wave 2 — Signed-receipt envelope shape tests.

These tests pin the 4-rename in the SignedReceiptsMiddleware:

    1. ``_signed_receipt``  -> ``signed_receipt`` (response key injection)
    2. ``algorithm``        -> ``alg`` (canonical field name)
    3. ``signed_over``      -> ``payload_hash`` (canonical field name)
    4. New field: ``receipt_id`` (per-response UUIDv4)
    5. New field: ``verification_key_hint`` (16-hex-char fingerprint)

The old key names (``_signed_receipt``, ``algorithm``, ``signed_over``) are
now READ-only aliases: the middleware no longer WRITES them.  These tests
assert that the rename took effect and that the signature is still valid
under the published verification key.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from greenlang.factors.middleware.signed_receipts import SignedReceiptsMiddleware
from greenlang.factors.signing import verify_sha256_hmac


SIGNING_SECRET = "test-secret-signed-receipt-shape-never-use-in-prod"


@pytest.fixture(autouse=True)
def _set_signing_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GL_FACTORS_SIGNING_SECRET", SIGNING_SECRET)
    monkeypatch.delenv("GL_FACTORS_ED25519_PRIVATE_KEY", raising=False)


def _make_app(tier: str = "pro") -> FastAPI:
    """Minimal FastAPI app that returns a dict payload on /v1/thing."""
    app = FastAPI()

    @app.middleware("http")
    async def _inject_tier(request, call_next):
        request.state.tier = tier
        return await call_next(request)

    app.add_middleware(SignedReceiptsMiddleware)

    @app.get("/v1/thing")
    async def _thing():
        return JSONResponse(
            {"factor_id": "demo-1", "co2e_per_unit": 0.712},
            headers={"X-GreenLang-Edition": "factors-ga-2027.04.0"},
        )

    return app


# ---------------------------------------------------------------------------
# New canonical key names ARE emitted.
# ---------------------------------------------------------------------------


class TestCanonicalKeyNames:
    def test_signed_receipt_key_present(self) -> None:
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        assert resp.status_code == 200
        body = resp.json()
        assert "signed_receipt" in body

    def test_receipt_has_four_required_keys(self) -> None:
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]
        for key in ("receipt_id", "signature", "verification_key_hint", "alg"):
            assert key in sr, f"missing required key {key!r}"

    def test_receipt_id_is_uuid4_shape(self) -> None:
        import uuid

        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]
        # uuid.UUID parses any UUID variant; we only require a valid UUID string.
        u = uuid.UUID(sr["receipt_id"])
        assert str(u) == sr["receipt_id"]

    def test_verification_key_hint_is_16_hex(self) -> None:
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]
        hint = sr["verification_key_hint"]
        assert isinstance(hint, str)
        assert len(hint) == 16
        int(hint, 16)  # must be hex

    def test_alg_value_is_known_algorithm(self) -> None:
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]
        assert sr["alg"] in ("sha256-hmac", "ed25519")

    def test_payload_hash_is_hex_sha256(self) -> None:
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]
        assert "payload_hash" in sr
        # hex sha-256 is 64 lowercase hex chars
        assert len(sr["payload_hash"]) == 64
        int(sr["payload_hash"], 16)

    def test_receipt_ids_are_unique_per_response(self) -> None:
        client = TestClient(_make_app())
        ids = set()
        for _ in range(5):
            resp = client.get("/v1/thing")
            ids.add(resp.json()["signed_receipt"]["receipt_id"])
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Old / deprecated key names are NO LONGER emitted (rename took).
# ---------------------------------------------------------------------------


class TestOldKeyNamesNotEmitted:
    def test_underscore_signed_receipt_key_absent(self) -> None:
        """Legacy ``_signed_receipt`` top-level key must NOT be emitted."""
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        body = resp.json()
        assert "_signed_receipt" not in body

    def test_algorithm_field_absent_in_receipt(self) -> None:
        """Legacy ``algorithm`` field inside receipt must NOT be emitted.

        The canonical name is ``alg`` now.  We still allow the legacy
        ``key_id`` alias during the one-release deprecation window, but
        ``algorithm`` as a top-level receipt property is gone.
        """
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]
        assert "algorithm" not in sr

    def test_signed_over_as_canonical_field_renamed_to_payload_hash(self) -> None:
        """The CANONICAL payload hash key is ``payload_hash`` (hex string).

        ``signed_over`` remains a legacy debug block during the deprecation
        window but the SDK's canonical read path is ``payload_hash``.
        """
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]
        assert "payload_hash" in sr
        # hex-shaped.
        assert isinstance(sr["payload_hash"], str)
        assert len(sr["payload_hash"]) == 64


# ---------------------------------------------------------------------------
# Signature verifies under the published verification key.
# ---------------------------------------------------------------------------


class TestSignatureVerifies:
    def test_hmac_signature_verifies(self) -> None:
        """Signature must verify using the same HMAC secret used to sign."""
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]

        # Reconstruct the `signed_over` block the middleware signed.
        assert "signed_over" in sr, (
            "signed_over is kept as a legacy companion for one release; "
            "its removal is a SEPARATE breaking change"
        )
        signed_over = sr["signed_over"]
        ok = verify_sha256_hmac(
            signed_over,
            {
                "signature": sr["signature"],
                "algorithm": sr["alg"],
                "signed_at": sr["signed_at"],
                "key_id": sr.get("key_id") or sr["verification_key_hint"],
                "payload_hash": sr["payload_hash"],
            },
        )
        assert ok is True

    def test_bad_secret_fails_verification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A different secret must NOT verify the signature."""
        client = TestClient(_make_app())
        resp = client.get("/v1/thing")
        sr = resp.json()["signed_receipt"]
        signed_over = sr["signed_over"]
        ok = verify_sha256_hmac(
            signed_over,
            {
                "signature": sr["signature"],
                "algorithm": sr["alg"],
                "signed_at": sr["signed_at"],
                "key_id": sr.get("key_id") or sr["verification_key_hint"],
                "payload_hash": sr["payload_hash"],
            },
            secret="wrong-secret",
        )
        assert ok is False
