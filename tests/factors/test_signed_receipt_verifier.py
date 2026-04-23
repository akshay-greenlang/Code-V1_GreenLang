# -*- coding: utf-8 -*-
"""
Tests for the offline signed-receipt verifier shipped in v1.0 of the
Factors Python SDK.

These tests exercise:
    * HMAC-SHA256 happy path (round-trips with the server's signing path)
    * Tampered payload rejected
    * Missing receipt rejected
    * Future-timestamp drift rejected
    * Unknown algorithm rejected
    * The verifier strips the receipt block before re-hashing
    * Bytes / string / dict input shapes all work

Ed25519 verification is *not* exercised here because it requires the
optional `cryptography` package and a live JWKS document; that path is
exercised in the Ed25519-specific integration test.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from greenlang.factors.signing import sign_sha256_hmac
from greenlang.factors.sdk.python.verify import (
    ReceiptVerificationError,
    verify_receipt,
)


SECRET = "unit-test-secret-do-not-use-in-prod"


def _make_receipt(payload: dict, *, secret: str = SECRET) -> dict:
    """Sign ``payload`` with the production HMAC primitive."""
    return sign_sha256_hmac(payload, secret=secret).to_dict()


def _make_response(payload: dict, *, secret: str = SECRET) -> dict:
    """Build a server-style response: payload + receipt sibling."""
    receipt = _make_receipt(payload, secret=secret)
    return {**payload, "receipt": receipt}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_verify_receipt_round_trips_a_valid_response() -> None:
    payload = {"factor_id": "ef:co2:diesel:us:2026", "co2e_per_unit": 10.21, "unit": "gal"}
    response = _make_response(payload)

    summary = verify_receipt(response, secret=SECRET)

    assert summary["verified"] is True
    assert summary["algorithm"] == "sha256-hmac"
    assert summary["payload_hash"]
    assert summary["key_id"] == "gl-factors-v1"


def test_verify_receipt_accepts_a_json_string() -> None:
    payload = {"a": 1, "b": [2, 3]}
    response_str = json.dumps(_make_response(payload))
    summary = verify_receipt(response_str, secret=SECRET)
    assert summary["verified"] is True


def test_verify_receipt_accepts_raw_bytes() -> None:
    payload = {"a": 1}
    response_bytes = json.dumps(_make_response(payload)).encode("utf-8")
    summary = verify_receipt(response_bytes, secret=SECRET)
    assert summary["verified"] is True


def test_verify_receipt_strips_receipt_block_before_hashing() -> None:
    """The verifier must hash the response *without* the receipt sibling.

    If it didn't, the receipt's `payload_hash` would never match because
    the receipt itself would be part of the canonicalised body.
    """
    payload = {"x": 1, "y": 2}
    response = _make_response(payload)
    # Prove it: passing the payload alone (no receipt) must fail with
    # "missing receipt" rather than "hash mismatch".
    with pytest.raises(ReceiptVerificationError, match="receipt"):
        verify_receipt(payload, secret=SECRET)
    # And the round-trip continues to verify with the receipt attached.
    assert verify_receipt(response, secret=SECRET)["verified"] is True


def test_verify_receipt_finds_receipt_under_meta_envelope() -> None:
    """Some endpoints emit the receipt under `meta.receipt`."""
    payload = {"meta": {"run_id": "r-42"}, "factor_id": "x"}
    receipt = _make_receipt(payload)
    response = {
        "meta": {**payload["meta"], "receipt": receipt},
        "factor_id": payload["factor_id"],
    }
    summary = verify_receipt(response, secret=SECRET)
    assert summary["verified"] is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_verify_receipt_rejects_missing_receipt() -> None:
    with pytest.raises(ReceiptVerificationError, match="does not contain a receipt"):
        verify_receipt({"factor_id": "x"}, secret=SECRET)


def test_verify_receipt_rejects_tampered_payload() -> None:
    payload = {"a": 1}
    response = _make_response(payload)
    response["a"] = 999  # mutate after signing
    with pytest.raises(ReceiptVerificationError, match="hash mismatch|modified"):
        verify_receipt(response, secret=SECRET)


def test_verify_receipt_rejects_wrong_secret() -> None:
    response = _make_response({"a": 1})
    with pytest.raises(ReceiptVerificationError, match="signature does not match"):
        verify_receipt(response, secret="WRONG_SECRET")


def test_verify_receipt_rejects_future_timestamp() -> None:
    payload = {"x": 1}
    receipt = _make_receipt(payload)
    # Push signed_at one hour into the future.
    receipt["signed_at"] = (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).isoformat()
    response = {**payload, "receipt": receipt}
    with pytest.raises(ReceiptVerificationError, match="future"):
        verify_receipt(response, secret=SECRET, future_tolerance_sec=10)


def test_verify_receipt_rejects_unknown_algorithm() -> None:
    payload = {"x": 1}
    receipt = _make_receipt(payload)
    receipt["algorithm"] = "made-up-algorithm"
    response = {**payload, "receipt": receipt}
    with pytest.raises(ReceiptVerificationError, match="Unknown receipt algorithm"):
        verify_receipt(response, secret=SECRET)


def test_verify_receipt_rejects_missing_signature_field() -> None:
    payload = {"x": 1}
    receipt = _make_receipt(payload)
    receipt.pop("signature", None)
    response = {**payload, "receipt": receipt}
    with pytest.raises(ReceiptVerificationError, match="signature"):
        verify_receipt(response, secret=SECRET)


def test_verify_receipt_requires_secret_for_hmac_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GL_FACTORS_SIGNING_SECRET", raising=False)
    response = _make_response({"a": 1})
    with pytest.raises(ReceiptVerificationError, match="requires a secret"):
        verify_receipt(response)


def test_verify_receipt_uses_env_var_secret_when_kwarg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _make_response({"a": 1})
    monkeypatch.setenv("GL_FACTORS_SIGNING_SECRET", SECRET)
    summary = verify_receipt(response)
    assert summary["verified"] is True


def test_verify_receipt_rejects_invalid_json_bytes() -> None:
    with pytest.raises(ReceiptVerificationError, match="JSON"):
        verify_receipt(b"not a json document", secret=SECRET)


def test_verify_receipt_rejects_invalid_json_string() -> None:
    with pytest.raises(ReceiptVerificationError, match="JSON"):
        verify_receipt("{broken", secret=SECRET)


# ---------------------------------------------------------------------------
# Sanity: the SDK client surface uses the same verifier
# ---------------------------------------------------------------------------


def test_client_verify_receipt_method_delegates_to_verifier() -> None:
    from greenlang.factors.sdk.python.client import FactorsClient

    payload = {"a": 1}
    response = _make_response(payload)
    # Use a no-op transport (we never touch the network for verification).
    client = FactorsClient(base_url="https://example.invalid", verify_greenlang_cert=False)
    try:
        summary = client.verify_receipt(response, secret=SECRET)
        assert summary["verified"] is True
    finally:
        client.close()
