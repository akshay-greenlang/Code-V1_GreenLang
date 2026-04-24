# -*- coding: utf-8 -*-
"""SDK v1.2 tests — Wave 2 / 2a / 2.5 envelope handling.

Covers:
    * Signed-receipt key renames (canonical ``signed_receipt`` + ``alg`` +
      ``payload_hash``) and one-release back-compat fallback for the legacy
      ``_signed_receipt`` / ``algorithm`` / ``signed_over`` spellings.
    * New typed envelope models surfaced on :class:`ResolvedFactor`:
      :class:`ChosenFactor`, :class:`SourceDescriptor`,
      :class:`QualityEnvelope`, :class:`UncertaintyEnvelope`,
      :class:`LicensingEnvelope`, :class:`DeprecationStatus`,
      :class:`SignedReceipt`.
    * Wave 2.5 ``audit_text`` + ``audit_text_draft`` fields.
    * :class:`FactorCannotResolveSafelyError` mapping from 422 responses.
    * End-to-end ``verify_receipt()`` happy path + tampered-payload path
      on a canonical Wave 2a demo response.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import warnings
from typing import Any, Dict

import httpx
import pytest

from greenlang.factors.sdk.python import (
    ChosenFactor,
    DeprecationStatus,
    FactorCannotResolveSafelyError,
    FactorsClient,
    LicensingEnvelope,
    QualityEnvelope,
    ReceiptVerificationError,
    ResolvedFactor,
    SignedReceipt,
    SourceDescriptor,
    UncertaintyEnvelope,
    __version__,
    verify_receipt,
)
from greenlang.factors.sdk.python.errors import error_from_response
from greenlang.factors.sdk.python.models import ResolutionRequest


BASE_URL = "https://factors.test"
API_PREFIX = "/api/v1"
HMAC_SECRET = "unit-test-secret-do-not-use-in-prod"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_hash(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _make_signed_response(
    payload: Dict[str, Any],
    *,
    wave2a: bool = True,
    secret: str = HMAC_SECRET,
) -> Dict[str, Any]:
    """Return ``payload`` plus a signed-receipt block.

    When ``wave2a=True`` the receipt uses the new canonical keys
    (``signed_receipt`` / ``alg`` / ``payload_hash``). When False, the
    legacy spellings (``_signed_receipt`` / ``algorithm`` / ``signed_over``)
    are emitted to exercise the SDK's deprecation fallback path.
    """
    payload_hash = _canonical_hash(payload)
    sig = hmac.new(
        secret.encode("utf-8"), payload_hash.encode("ascii"), hashlib.sha256
    ).digest()
    sig_b64 = base64.b64encode(sig).decode("ascii")

    if wave2a:
        receipt = {
            "receipt_id": "11111111-1111-1111-1111-111111111111",
            "signature": sig_b64,
            "verification_key_hint": "abc123def456aa11",
            "alg": "sha256-hmac",
            "payload_hash": payload_hash,
            "signed_at": "2026-04-23T00:00:00+00:00",
            "key_id": "gl-factors-v1",
        }
        return {**payload, "signed_receipt": receipt}

    # Legacy shape — exercises the SDK's back-compat reader.
    legacy_receipt = {
        "signature": sig_b64,
        "algorithm": "sha256-hmac",  # deprecated alias for ``alg``
        "signed_over": payload_hash,  # deprecated alias for ``payload_hash``
        "signed_at": "2026-04-23T00:00:00+00:00",
        "key_id": "gl-factors-v1",
    }
    return {**payload, "_signed_receipt": legacy_receipt}


# ---------------------------------------------------------------------------
# Version bump
# ---------------------------------------------------------------------------


def test_sdk_version_is_1_2_0() -> None:
    assert __version__ == "1.3.0"


# ---------------------------------------------------------------------------
# Wave 2 typed envelope models
# ---------------------------------------------------------------------------


def test_chosen_factor_envelope_parses() -> None:
    cf = ChosenFactor.model_validate(
        {
            "factor_id": "ef:co2:diesel:us:2026",
            "factor_version": "2026.2",
            "release_version": "corporate_scope1.v3",
            "method_profile": "corporate_scope1",
            "method_pack_id": "corporate_scope1",
            "pack_id": "corporate_scope1",
            "co2e_per_unit": 10.21,
            "unit": "gal",
            "geography": "US",
            "scope": "scope_1",
        }
    )
    assert cf.factor_id == "ef:co2:diesel:us:2026"
    assert cf.release_version == "corporate_scope1.v3"
    assert cf.pack_id == "corporate_scope1"


def test_quality_envelope_surfaces_composite_fqs() -> None:
    q = QualityEnvelope.model_validate({"composite_fqs_0_100": 87.5})
    assert q.composite_fqs_0_100 == 87.5


def test_licensing_envelope_parses_upstream_chain() -> None:
    lic = LicensingEnvelope.model_validate(
        {
            "license": "Open Government Licence v3.0",
            "license_class": "certified",
            "redistribution_class": "redistributable",
            "upstream_licenses": ["OGL-UK-3.0", "CC-BY-4.0"],
            "attribution": "UK BEIS 2024",
            "restrictions": [],
        }
    )
    assert lic.license_class == "certified"
    assert lic.upstream_licenses == ["OGL-UK-3.0", "CC-BY-4.0"]


def test_deprecation_status_envelope_parses() -> None:
    ds = DeprecationStatus.model_validate(
        {
            "status": "scheduled",
            "effective_from": "2027-01-01",
            "replacement_factor_id": "ef:co2:diesel:us:2027",
            "reason": "Superseded by 2027 AR6 GWP revision",
        }
    )
    assert ds.status == "scheduled"
    assert ds.replacement_factor_id == "ef:co2:diesel:us:2027"


# ---------------------------------------------------------------------------
# Signed-receipt key renames + back-compat
# ---------------------------------------------------------------------------


def test_signed_receipt_reads_canonical_keys() -> None:
    sr = SignedReceipt.from_response_dict(
        {
            "receipt_id": "r1",
            "signature": "deadbeef",
            "verification_key_hint": "abc123",
            "alg": "sha256-hmac",
            "payload_hash": "c0ffee",
        }
    )
    assert sr.alg == "sha256-hmac"
    assert sr.payload_hash == "c0ffee"
    assert sr.verification_key_hint == "abc123"


def test_signed_receipt_falls_back_to_legacy_algorithm_key() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sr = SignedReceipt.from_response_dict(
            {
                "signature": "deadbeef",
                "algorithm": "sha256-hmac",  # deprecated
                "payload_hash": "c0ffee",
            }
        )
    assert sr.alg == "sha256-hmac"
    assert any(
        issubclass(warning.category, DeprecationWarning)
        and "algorithm" in str(warning.message)
        for warning in w
    ), "expected a DeprecationWarning on legacy 'algorithm' key"


def test_signed_receipt_falls_back_to_legacy_signed_over_envelope() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sr = SignedReceipt.from_response_dict(
            {
                "signature": "deadbeef",
                "alg": "sha256-hmac",
                "signed_over": {
                    "body_hash": "c0ffee",
                    "path": "/api/v1/factors/foo",
                },
            }
        )
    assert sr.payload_hash == "c0ffee"
    assert any(
        issubclass(warning.category, DeprecationWarning)
        and "signed_over" in str(warning.message)
        for warning in w
    )


def test_resolved_factor_reads_canonical_top_level_key() -> None:
    payload = {
        "chosen_factor_id": "ef:co2:diesel:us:2026",
        "factor_id": "ef:co2:diesel:us:2026",
        "signed_receipt": {
            "receipt_id": "r1",
            "signature": "deadbeef",
            "verification_key_hint": "abc123",
            "alg": "sha256-hmac",
            "payload_hash": "c0ffee",
        },
    }
    rf = ResolvedFactor.model_validate(payload)
    assert rf.signed_receipt is not None
    assert rf.signed_receipt.alg == "sha256-hmac"
    assert rf.signed_receipt.payload_hash == "c0ffee"


def test_resolved_factor_falls_back_to_legacy_underscore_key() -> None:
    payload = {
        "chosen_factor_id": "ef:co2:diesel:us:2026",
        "_signed_receipt": {
            "signature": "deadbeef",
            "algorithm": "sha256-hmac",
            "signed_over": "c0ffee",
        },
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rf = ResolvedFactor.model_validate(payload)
    assert rf.signed_receipt is not None
    assert rf.signed_receipt.alg == "sha256-hmac"
    assert rf.signed_receipt.payload_hash == "c0ffee"
    # At least one DeprecationWarning should have fired.
    assert any(
        issubclass(warning.category, DeprecationWarning) for warning in w
    )


# ---------------------------------------------------------------------------
# Wave 2.5 audit_text
# ---------------------------------------------------------------------------


def test_resolved_factor_surfaces_audit_text_draft() -> None:
    rf = ResolvedFactor.model_validate(
        {
            "chosen_factor_id": "ef:co2:diesel:us:2026",
            "audit_text": (
                "Selected EPA AP-42 stationary diesel factor because the "
                "request specified scope_1 stationary combustion in US."
            ),
            "audit_text_draft": True,
        }
    )
    assert rf.audit_text is not None
    assert rf.audit_text.startswith("Selected EPA AP-42")
    assert rf.audit_text_draft is True


def test_resolved_factor_audit_text_absent_defaults_to_none() -> None:
    rf = ResolvedFactor.model_validate({"chosen_factor_id": "foo"})
    assert rf.audit_text is None
    assert rf.audit_text_draft is None


# ---------------------------------------------------------------------------
# FactorCannotResolveSafelyError
# ---------------------------------------------------------------------------


def test_error_from_response_maps_factor_cannot_resolve_safely() -> None:
    body = {
        "detail": "No candidate meets the safety floor",
        "error_code": "factor_cannot_resolve_safely",
        "details": {
            "pack_id": "corporate_scope1",
            "method_profile": "corporate_scope1",
            "evaluated_candidates_count": 7,
        },
    }
    exc = error_from_response(
        status_code=422,
        url="https://factors.test/api/v1/factors/resolve-explain",
        body=body,
    )
    assert isinstance(exc, FactorCannotResolveSafelyError)
    assert exc.pack_id == "corporate_scope1"
    assert exc.method_profile == "corporate_scope1"
    assert exc.evaluated_candidates_count == 7


def test_client_raises_factor_cannot_resolve_safely_on_422() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": "Safety floor not met",
                "error_code": "factor_cannot_resolve_safely",
                "details": {
                    "pack_id": "product_carbon",
                    "method_profile": "product_carbon",
                    "evaluated_candidates_count": 3,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        req = ResolutionRequest(
            activity="battery cathode NMC811",
            method_profile="product_carbon",
        )
        with pytest.raises(FactorCannotResolveSafelyError) as excinfo:
            client.resolve(req)
    exc = excinfo.value
    assert exc.pack_id == "product_carbon"
    assert exc.evaluated_candidates_count == 3


# ---------------------------------------------------------------------------
# End-to-end verify_receipt on a canonical Wave 2a demo response
# ---------------------------------------------------------------------------


def test_verify_receipt_canonical_wave2a_response() -> None:
    payload = {
        "chosen_factor_id": "ef:co2:diesel:us:2026",
        "co2e_per_unit": 10.21,
        "quality": {"composite_fqs_0_100": 92.5},
    }
    response = _make_signed_response(payload, wave2a=True)
    summary = verify_receipt(response, secret=HMAC_SECRET)
    assert summary["verified"] is True
    assert summary["alg"] == "sha256-hmac"
    assert summary["algorithm"] == "sha256-hmac"  # back-compat
    assert summary["receipt_id"] == "11111111-1111-1111-1111-111111111111"
    assert summary["verification_key_hint"] == "abc123def456aa11"


def test_verify_receipt_legacy_keys_still_verifies_with_warning() -> None:
    payload = {"chosen_factor_id": "ef:co2:diesel:us:2026", "co2e_per_unit": 10.21}
    response = _make_signed_response(payload, wave2a=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        summary = verify_receipt(response, secret=HMAC_SECRET)
    assert summary["verified"] is True
    assert any(
        issubclass(warning.category, DeprecationWarning) for warning in w
    ), "legacy keys should emit a DeprecationWarning"


def test_verify_receipt_raises_on_tampered_payload() -> None:
    payload = {"chosen_factor_id": "ef:co2:diesel:us:2026", "co2e_per_unit": 10.21}
    response = _make_signed_response(payload, wave2a=True)
    # Tamper with the payload AFTER signing — the hash should no longer match.
    response["co2e_per_unit"] = 99.99
    with pytest.raises(ReceiptVerificationError):
        verify_receipt(response, secret=HMAC_SECRET)


def test_verify_receipt_raises_on_tampered_signature() -> None:
    payload = {"chosen_factor_id": "ef:co2:diesel:us:2026"}
    response = _make_signed_response(payload, wave2a=True)
    # Flip a byte in the signature.
    sig = response["signed_receipt"]["signature"]
    tampered = ("A" if sig[0] != "A" else "B") + sig[1:]
    response["signed_receipt"]["signature"] = tampered
    with pytest.raises(ReceiptVerificationError):
        verify_receipt(response, secret=HMAC_SECRET)


# ---------------------------------------------------------------------------
# ResolvedFactor surfaces new envelope fields end-to-end via the mock client
# ---------------------------------------------------------------------------


def test_resolve_surfaces_full_wave2_envelope() -> None:
    resolved_payload = {
        "chosen_factor_id": "ef:co2:diesel:us:2026",
        "chosen_factor": {
            "factor_id": "ef:co2:diesel:us:2026",
            "factor_version": "2026.2",
            "release_version": "corporate_scope1.v3",
            "method_profile": "corporate_scope1",
            "pack_id": "corporate_scope1",
            "co2e_per_unit": 10.21,
            "unit": "gal",
        },
        "release_version": "corporate_scope1.v3",
        "source": {
            "source_id": "epa_ghg_2026",
            "organization": "EPA",
            "license_class": "certified",
        },
        "quality": {"composite_fqs_0_100": 92.5},
        "uncertainty": {"ci_95": 0.12, "distribution": "lognormal"},
        "licensing": {
            "license_class": "certified",
            "redistribution_class": "redistributable",
            "upstream_licenses": ["US-Government-Work"],
        },
        "deprecation_status": {
            "status": "current",
            "effective_from": "2026-01-01",
        },
        "audit_text": "Selected EPA AP-42 diesel factor.",
        "audit_text_draft": False,
        "signed_receipt": {
            "receipt_id": "22222222-2222-2222-2222-222222222222",
            "signature": "deadbeef",
            "verification_key_hint": "face1234cafe5678",
            "alg": "sha256-hmac",
            "payload_hash": "c0ffee",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=resolved_payload)

    transport = httpx.MockTransport(handler)
    with FactorsClient(
        base_url=BASE_URL, transport=transport, verify_greenlang_cert=False
    ) as client:
        req = ResolutionRequest(
            activity="diesel stationary", method_profile="corporate_scope1"
        )
        rf = client.resolve(req)

    # Typed envelopes populated
    assert isinstance(rf.chosen_factor, ChosenFactor)
    assert rf.chosen_factor.release_version == "corporate_scope1.v3"
    assert isinstance(rf.source, SourceDescriptor)
    assert rf.source.source_id == "epa_ghg_2026"
    assert isinstance(rf.quality, QualityEnvelope)
    assert rf.quality.composite_fqs_0_100 == 92.5
    assert isinstance(rf.uncertainty, UncertaintyEnvelope)
    assert rf.uncertainty.ci_95 == 0.12
    assert isinstance(rf.licensing, LicensingEnvelope)
    assert rf.licensing.redistribution_class == "redistributable"
    # Wave 2.5 narrative
    assert rf.audit_text == "Selected EPA AP-42 diesel factor."
    assert rf.audit_text_draft is False
    # Wave 2a receipt (typed)
    assert isinstance(rf.signed_receipt, SignedReceipt)
    assert rf.signed_receipt.alg == "sha256-hmac"
    assert rf.signed_receipt.receipt_id == "22222222-2222-2222-2222-222222222222"
