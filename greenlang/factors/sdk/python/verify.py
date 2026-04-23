# -*- coding: utf-8 -*-
"""Offline signed-receipt verifier for the Factors SDK.

Every API response from the GreenLang Factors service can carry a signed
receipt that proves what the server returned, when, and to whom. This
module verifies those receipts entirely offline so customers can audit
their reports months or years after the original request -- no network
round-trip back to the GreenLang service required.

Two algorithms are supported:

* **HMAC-SHA256** (default for Community / Developer Pro tiers). Symmetric
  key shared via the ``GL_FACTORS_SIGNING_SECRET`` environment variable.
* **Ed25519** (Consulting / Platform / Enterprise). Asymmetric, verified
  against a JSON Web Key Set (JWKS) document fetched from
  ``https://api.greenlang.io/.well-known/jwks.json`` (or a customer-side
  copy passed via ``jwks_url``).

The verifier is purely defensive: an unsigned response always raises
:class:`ReceiptVerificationError` so a missing signature can never be
confused with a successful verification.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

#: Maximum drift (seconds) allowed between ``signed_at`` and the local
#: clock when verifying. Receipts older than this are still considered
#: valid -- this guard only triggers on receipts with future timestamps,
#: which would indicate clock skew or a forged signature.
DEFAULT_FUTURE_TOLERANCE_SEC: int = 600


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ReceiptVerificationError(Exception):
    """Receipt verification failed (signature, hash, format, or freshness)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_hash(payload: Any) -> str:
    """Compute the SHA-256 hex digest of a canonical JSON serialisation.

    Mirrors the server implementation in
    :func:`greenlang.factors.signing._canonical_hash` so client and
    server produce identical hashes for the same logical payload.
    """
    body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _b64url_decode(data: str) -> bytes:
    """Decode a base64url string with padding tolerance."""
    s = data.strip().replace("-", "+").replace("_", "/")
    padding = (-len(s)) % 4
    if padding:
        s = s + "=" * padding
    return base64.b64decode(s.encode("ascii"))


def _b64_decode(data: str) -> bytes:
    """Decode a standard base64 string (also tolerates base64url)."""
    s = data.strip()
    if "-" in s or "_" in s:
        return _b64url_decode(s)
    padding = (-len(s)) % 4
    if padding:
        s = s + "=" * padding
    return base64.b64decode(s.encode("ascii"))


#: Top-level response keys that may carry the signed receipt. Wave 2a
#: canonicalised ``signed_receipt``; ``_signed_receipt`` is kept for one
#: release so older servers still verify. ``receipt`` is the legacy
#: pre-1.0 shape.
_RECEIPT_KEY_CANDIDATES: tuple = ("signed_receipt", "_signed_receipt", "receipt")

#: Deprecated receipt-key aliases -> canonical names. The verifier reads
#: both; a DeprecationWarning fires if a fallback is taken.
_LEGACY_RECEIPT_ALIASES: Dict[str, str] = {
    "algorithm": "alg",
    "signed_over": "payload_hash",
}


def _warn_deprecated_receipt_key(legacy: str, canonical: str) -> None:
    import warnings

    warnings.warn(
        f"Signed receipt key {legacy!r} is deprecated; server should emit "
        f"{canonical!r}. SDK fallback will be removed in v2.0.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def _normalize_receipt(raw: Any) -> Optional[Dict[str, Any]]:
    """Return a new receipt dict with Wave 2a canonical keys.

    Reads ``alg`` then falls back to ``algorithm``; reads ``payload_hash``
    then falls back to ``signed_over`` (either a string hex or the
    legacy ``{body_hash, ...}`` envelope). Emits a DeprecationWarning on
    every fallback read.
    """
    if not isinstance(raw, dict):
        return None
    out = dict(raw)
    if "alg" not in out and "algorithm" in out:
        _warn_deprecated_receipt_key("algorithm", "alg")
        out["alg"] = out.get("algorithm")
    if "payload_hash" not in out and "signed_over" in out:
        _warn_deprecated_receipt_key("signed_over", "payload_hash")
        so = out.get("signed_over")
        if isinstance(so, dict):
            out["payload_hash"] = so.get("body_hash") or so.get("payload_hash")
        elif isinstance(so, str):
            out["payload_hash"] = so
    return out


def _extract_receipt(response: Any) -> Optional[Dict[str, Any]]:
    """Locate the receipt block inside a parsed response payload.

    Wave 2a top-level key is ``signed_receipt``. ``_signed_receipt`` is
    still accepted for one release; a DeprecationWarning fires on fallback.
    Also accepts the legacy ``receipt`` sibling, ``meta.receipt`` envelope,
    and a top-level receipt-shaped dict.
    """
    if response is None:
        return None
    if isinstance(response, dict):
        # Canonical and legacy top-level keys, in preference order.
        for key in _RECEIPT_KEY_CANDIDATES:
            if key in response and isinstance(response[key], dict):
                if key == "_signed_receipt":
                    _warn_deprecated_receipt_key("_signed_receipt", "signed_receipt")
                return _normalize_receipt(response[key])
        meta = response.get("meta")
        if isinstance(meta, dict):
            for key in _RECEIPT_KEY_CANDIDATES:
                if key in meta and isinstance(meta[key], dict):
                    if key == "_signed_receipt":
                        _warn_deprecated_receipt_key(
                            "_signed_receipt", "signed_receipt"
                        )
                    return _normalize_receipt(meta[key])
        # Top-level receipt-shaped dict (either new or legacy).
        if "signature" in response and ("alg" in response or "algorithm" in response):
            return _normalize_receipt(response)
    return None


def _extract_payload(response: Any) -> Any:
    """Strip the receipt block out of ``response`` before re-hashing.

    Handles every receipt-placement variant (canonical ``signed_receipt``,
    deprecated ``_signed_receipt``, legacy ``receipt`` sibling, and
    ``meta.receipt``) so the re-hash matches what the server signed.
    """
    if isinstance(response, dict):
        present = [k for k in _RECEIPT_KEY_CANDIDATES if k in response]
        if present:
            return {k: v for k, v in response.items() if k not in present}
        meta = response.get("meta")
        if isinstance(meta, dict):
            meta_present = [k for k in _RECEIPT_KEY_CANDIDATES if k in meta]
            if meta_present:
                cleaned_meta = {
                    k: v for k, v in meta.items() if k not in meta_present
                }
                cleaned = dict(response)
                cleaned["meta"] = cleaned_meta
                return cleaned
    return response


def _parse_iso_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a tz-aware UTC datetime.

    Tolerates ``"Z"`` suffix and offset-less strings (interpreted UTC).
    """
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# JWKS fetch + cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _JWKSEntry:
    """Single JWK entry minimally parsed for Ed25519 verification."""

    kid: str
    kty: str
    crv: Optional[str]
    x: Optional[str]
    alg: Optional[str]


_JWKS_CACHE: Dict[str, List[_JWKSEntry]] = {}


def _fetch_jwks(jwks_url: str) -> List[_JWKSEntry]:
    """Fetch and parse a JWKS document, with simple in-process caching."""
    if jwks_url in _JWKS_CACHE:
        return _JWKS_CACHE[jwks_url]
    if not re.match(r"^https?://", jwks_url):
        raise ReceiptVerificationError(f"Invalid JWKS URL: {jwks_url!r}")
    req = urllib.request.Request(jwks_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            doc = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ReceiptVerificationError(
            f"Failed to fetch JWKS from {jwks_url}: {exc}"
        ) from exc
    keys: List[_JWKSEntry] = []
    for key in doc.get("keys", []) or []:
        if not isinstance(key, dict):
            continue
        keys.append(
            _JWKSEntry(
                kid=str(key.get("kid", "")),
                kty=str(key.get("kty", "")),
                crv=key.get("crv"),
                x=key.get("x"),
                alg=key.get("alg"),
            )
        )
    _JWKS_CACHE[jwks_url] = keys
    return keys


def _select_jwk(keys: List[_JWKSEntry], kid: str) -> _JWKSEntry:
    """Return the JWK matching ``kid`` or raise."""
    for k in keys:
        if k.kid == kid:
            return k
    raise ReceiptVerificationError(
        f"No JWK with kid={kid!r} (available: {[k.kid for k in keys]})"
    )


# ---------------------------------------------------------------------------
# Signature verification primitives
# ---------------------------------------------------------------------------


def _verify_hmac(
    payload_hash_hex: str,
    signature_b64: str,
    *,
    secret: bytes,
) -> bool:
    """Constant-time HMAC-SHA256 verification."""
    expected = hmac.new(secret, payload_hash_hex.encode("ascii"), hashlib.sha256).digest()
    expected_b64 = base64.b64encode(expected).decode("ascii")
    return hmac.compare_digest(expected_b64, signature_b64.strip())


def _verify_ed25519(
    payload_hash_hex: str,
    signature_b64: str,
    *,
    public_key_x_b64url: str,
) -> bool:
    """Ed25519 verification using the ``cryptography`` package.

    Raises :class:`ReceiptVerificationError` if ``cryptography`` is not
    installed -- callers can either install the optional dependency or
    fall back to HMAC-SHA256 receipts.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ReceiptVerificationError(
            "Ed25519 receipt verification requires the optional "
            "`cryptography` package: pip install cryptography"
        ) from exc

    try:
        public_bytes = _b64url_decode(public_key_x_b64url)
        signature = _b64_decode(signature_b64)
        key = Ed25519PublicKey.from_public_bytes(public_bytes)
        key.verify(signature, payload_hash_hex.encode("ascii"))
        return True
    except InvalidSignature:
        return False
    except Exception as exc:  # noqa: BLE001
        raise ReceiptVerificationError(f"Ed25519 verification error: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_receipt(
    response: Union[Dict[str, Any], str, bytes],
    *,
    secret: Optional[Union[str, bytes]] = None,
    jwks_url: Optional[str] = None,
    algorithm: Optional[str] = None,
    future_tolerance_sec: int = DEFAULT_FUTURE_TOLERANCE_SEC,
) -> Dict[str, Any]:
    """Verify a signed-receipt-bearing response **offline**.

    Args:
        response: The full server response. May be passed as a parsed
            ``dict``, a JSON string, or raw bytes.
        secret: Shared secret for HMAC-SHA256 receipts. Defaults to the
            ``GL_FACTORS_SIGNING_SECRET`` environment variable when
            ``algorithm`` resolves to ``sha256-hmac``.
        jwks_url: JWKS document URL for Ed25519 receipts. Defaults to
            ``https://api.greenlang.io/.well-known/jwks.json`` when
            unspecified and the receipt algorithm is Ed25519.
        algorithm: Optional explicit algorithm override. When omitted the
            receipt's ``algorithm`` field is trusted.
        future_tolerance_sec: Maximum number of seconds the receipt's
            ``signed_at`` timestamp may be ahead of the local clock.

    Returns:
        A dict summarising the verified receipt::

            {
              "verified": True,
              "algorithm": "sha256-hmac" | "ed25519",
              "key_id": "...",
              "signed_at": "<iso>",
              "payload_hash": "<sha256 hex>",
              "edition_id": "..." | None,   # mirrored when present
              "factor_ids": [...] | None,
              "caller_id": "..." | None,
            }

    Raises:
        ReceiptVerificationError: When verification fails for any reason
            (missing receipt, bad signature, bad hash, future timestamp,
            unknown algorithm, or missing key material).
    """
    if isinstance(response, (bytes, bytearray)):
        try:
            response = json.loads(bytes(response).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReceiptVerificationError(
                f"Could not parse response bytes as JSON: {exc}"
            ) from exc
    elif isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ReceiptVerificationError(
                f"Could not parse response string as JSON: {exc}"
            ) from exc

    receipt = _extract_receipt(response)
    if receipt is None:
        raise ReceiptVerificationError(
            "Response does not contain a receipt block. Either the "
            "server did not sign it (Community tier) or the receipt was "
            "stripped in transit."
        )

    payload = _extract_payload(response)
    expected_hash = _canonical_hash(payload)

    actual_hash = str(receipt.get("payload_hash") or "")
    if not hmac.compare_digest(expected_hash, actual_hash):
        raise ReceiptVerificationError(
            "Payload hash mismatch: response body has been modified "
            "since the receipt was issued."
        )

    # Wave 2a: canonical field is ``alg``; legacy ``algorithm`` is still
    # accepted (``_normalize_receipt`` copies it forward with a warning).
    receipt_algorithm = (
        algorithm or receipt.get("alg") or receipt.get("algorithm") or ""
    ).lower().strip()
    if not receipt_algorithm:
        raise ReceiptVerificationError(
            "Receipt is missing the algorithm field (expected 'alg')."
        )

    signature = str(receipt.get("signature") or "")
    if not signature:
        raise ReceiptVerificationError("Receipt is missing the signature field.")

    key_id = str(
        receipt.get("key_id")
        or receipt.get("verification_key_hint")
        or ""
    )
    signed_at_str = str(receipt.get("signed_at") or "")

    if signed_at_str:
        try:
            signed_at = _parse_iso_utc(signed_at_str)
        except ValueError as exc:
            raise ReceiptVerificationError(
                f"Receipt signed_at is not a valid ISO-8601 timestamp: {signed_at_str!r}"
            ) from exc
        now = datetime.now(timezone.utc)
        drift = (signed_at - now).total_seconds()
        if drift > future_tolerance_sec:
            raise ReceiptVerificationError(
                f"Receipt signed_at is in the future by {drift:.0f}s "
                f"(tolerance={future_tolerance_sec}s)."
            )

    if receipt_algorithm == "sha256-hmac":
        secret_bytes = _resolve_secret(secret)
        if not _verify_hmac(expected_hash, signature, secret=secret_bytes):
            raise ReceiptVerificationError(
                "HMAC-SHA256 receipt signature does not match."
            )
    elif receipt_algorithm == "ed25519":
        url = jwks_url or os.getenv(
            "GL_FACTORS_JWKS_URL",
            "https://api.greenlang.io/.well-known/jwks.json",
        )
        keys = _fetch_jwks(url)
        if not key_id:
            raise ReceiptVerificationError(
                "Ed25519 receipt missing key_id; cannot select JWK."
            )
        jwk = _select_jwk(keys, key_id)
        if jwk.kty.upper() != "OKP" or (jwk.crv or "").lower() != "ed25519":
            raise ReceiptVerificationError(
                f"JWK kid={key_id!r} is not an Ed25519 key (kty={jwk.kty}, crv={jwk.crv})."
            )
        if not jwk.x:
            raise ReceiptVerificationError(
                f"JWK kid={key_id!r} is missing the public-key 'x' parameter."
            )
        if not _verify_ed25519(expected_hash, signature, public_key_x_b64url=jwk.x):
            raise ReceiptVerificationError(
                "Ed25519 receipt signature does not match."
            )
    else:
        raise ReceiptVerificationError(
            f"Unknown receipt algorithm: {receipt_algorithm!r}"
        )

    # Optional: surface signed scope fields when present so the caller
    # can confirm the receipt covers what they expect. ``algorithm`` is
    # retained for backwards compatibility; ``alg`` mirrors the Wave 2a
    # canonical spelling.
    summary: Dict[str, Any] = {
        "verified": True,
        "algorithm": receipt_algorithm,
        "alg": receipt_algorithm,
        "key_id": key_id,
        "receipt_id": receipt.get("receipt_id"),
        "verification_key_hint": receipt.get("verification_key_hint"),
        "signed_at": signed_at_str,
        "payload_hash": expected_hash,
        "edition_id": receipt.get("edition_id"),
        "factor_ids": receipt.get("factor_ids"),
        "caller_id": receipt.get("caller_id"),
    }
    return summary


def _resolve_secret(secret: Optional[Union[str, bytes]]) -> bytes:
    """Return the HMAC secret as bytes, defaulting to the env var."""
    if secret is None:
        env = os.getenv("GL_FACTORS_SIGNING_SECRET", "")
        if not env:
            raise ReceiptVerificationError(
                "HMAC receipt verification requires a secret. Pass "
                "`secret=` or set GL_FACTORS_SIGNING_SECRET."
            )
        return env.encode("utf-8")
    if isinstance(secret, str):
        return secret.encode("utf-8")
    return bytes(secret)


__all__ = [
    "ReceiptVerificationError",
    "verify_receipt",
    "DEFAULT_FUTURE_TOLERANCE_SEC",
]
