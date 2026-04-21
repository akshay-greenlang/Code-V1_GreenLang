# -*- coding: utf-8 -*-
"""Webhook signature verifier for Factors webhooks.

Mirrors the server-side
:func:`greenlang.factors.webhooks.sign_webhook_payload`: HMAC-SHA256
over the *canonical* JSON form of the payload with ``sort_keys=True``.

Two call styles are supported:

1. Payload-dict mode (recommended when you already parsed JSON)::

       from greenlang.factors.sdk.python import verify_webhook
       verify_webhook(payload_dict, signature, secret)

2. Raw-bytes mode (for framework integrations that give you the raw body)::

       verify_webhook_bytes(raw_body, signature, secret)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Mapping, Optional, Union

logger = logging.getLogger(__name__)


class WebhookVerificationError(Exception):
    """Raised when a webhook signature does not match."""


def compute_signature(payload: Mapping[str, Any], secret: str) -> str:
    """Compute the canonical HMAC-SHA256 hex signature.

    Matches :func:`greenlang.factors.webhooks.sign_webhook_payload` exactly.
    """
    body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def compute_signature_bytes(body: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 over pre-serialised canonical bytes."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_webhook(
    payload: Union[Mapping[str, Any], bytes, str],
    signature: str,
    secret: str,
    *,
    scheme: Optional[str] = None,
) -> bool:
    """Constant-time signature verification.

    Accepts either a dict payload (re-canonicalised before hashing) or
    raw bytes/str body (hashed as-is).  The ``signature`` may optionally
    carry a scheme prefix (``"sha256="``) which is stripped.

    Returns True on match, False otherwise — never raises.
    """
    if not signature or not secret:
        return False

    sig = signature.strip()
    if "=" in sig:
        prefix, _, sig_value = sig.partition("=")
        if scheme and prefix.lower() != scheme.lower():
            return False
    else:
        sig_value = sig

    if isinstance(payload, Mapping):
        expected = compute_signature(payload, secret)
    elif isinstance(payload, (bytes, bytearray)):
        expected = compute_signature_bytes(bytes(payload), secret)
    elif isinstance(payload, str):
        expected = compute_signature_bytes(payload.encode("utf-8"), secret)
    else:
        return False

    return hmac.compare_digest(expected, sig_value)


def verify_webhook_bytes(body: bytes, signature: str, secret: str) -> bool:
    """Verify signature over raw request body bytes."""
    return verify_webhook(body, signature, secret)


def verify_webhook_strict(
    payload: Union[Mapping[str, Any], bytes, str],
    signature: str,
    secret: str,
) -> None:
    """Same as :func:`verify_webhook` but raises on mismatch."""
    if not verify_webhook(payload, signature, secret):
        raise WebhookVerificationError("Webhook signature verification failed")


def parse_signature_header(header_value: str) -> Dict[str, str]:
    """Parse comma-separated key=value signature headers (Stripe-style).

    Example input::

        "t=1700000000,v1=abc123..."

    Returns a dict of components; useful if GreenLang rolls out a
    versioned signature scheme later without breaking the SDK.
    """
    out: Dict[str, str] = {}
    if not header_value:
        return out
    for chunk in header_value.split(","):
        if "=" in chunk:
            k, _, v = chunk.partition("=")
            out[k.strip()] = v.strip()
    return out


__all__ = [
    "WebhookVerificationError",
    "compute_signature",
    "compute_signature_bytes",
    "verify_webhook",
    "verify_webhook_bytes",
    "verify_webhook_strict",
    "parse_signature_header",
]
