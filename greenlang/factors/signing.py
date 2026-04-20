# -*- coding: utf-8 -*-
"""
Signed result receipts (Phase F6).

Every API response can be signed so customers can prove what we returned
and when.  Two algorithms supported:

- **SHA-256 HMAC** (default, symmetric) — fast, requires shared secret.
  Suitable for the open-core tier: ``GL_FACTORS_SIGNING_SECRET`` env var.
- **Ed25519** (asymmetric) — customers verify with GreenLang's public
  key; we keep the private key in Vault.

Receipt format::

    {
      "signature": "<base64>",
      "algorithm": "sha256-hmac" | "ed25519",
      "signed_at": "<iso>",
      "key_id": "<rotating-kid>",
      "payload_hash": "<sha256 of canonical response>"
    }

Callers attach the receipt to the response header or body.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Receipt:
    signature: str
    algorithm: str
    signed_at: str
    key_id: str
    payload_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "algorithm": self.algorithm,
            "signed_at": self.signed_at,
            "key_id": self.key_id,
            "payload_hash": self.payload_hash,
        }


class SigningError(RuntimeError):
    pass


def _canonical_hash(payload: Any) -> str:
    """SHA-256 over a canonical JSON serialisation of ``payload``."""
    body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def sign_sha256_hmac(
    payload: Any,
    *,
    secret: Optional[str] = None,
    key_id: str = "gl-factors-v1",
) -> Receipt:
    """Sign with HMAC-SHA256 using the shared secret."""
    key = secret or os.getenv("GL_FACTORS_SIGNING_SECRET", "")
    if not key:
        raise SigningError(
            "GL_FACTORS_SIGNING_SECRET env var not set; "
            "provide `secret` kwarg or configure the environment."
        )
    payload_hash = _canonical_hash(payload)
    mac = hmac.new(key.encode("utf-8"), payload_hash.encode("ascii"), hashlib.sha256)
    sig = base64.b64encode(mac.digest()).decode("ascii")
    return Receipt(
        signature=sig,
        algorithm="sha256-hmac",
        signed_at=datetime.now(timezone.utc).isoformat(),
        key_id=key_id,
        payload_hash=payload_hash,
    )


def verify_sha256_hmac(
    payload: Any,
    receipt: Dict[str, Any],
    *,
    secret: Optional[str] = None,
) -> bool:
    """Verify a receipt against ``payload``.  Constant-time comparison."""
    key = secret or os.getenv("GL_FACTORS_SIGNING_SECRET", "")
    if not key:
        raise SigningError("GL_FACTORS_SIGNING_SECRET not set for verification")
    expected_hash = _canonical_hash(payload)
    if receipt.get("payload_hash") != expected_hash:
        return False
    mac = hmac.new(key.encode("utf-8"), expected_hash.encode("ascii"), hashlib.sha256)
    expected_sig = base64.b64encode(mac.digest()).decode("ascii")
    provided = str(receipt.get("signature") or "")
    # Constant-time compare to avoid timing oracles.
    return hmac.compare_digest(expected_sig, provided)


def sign_ed25519(
    payload: Any,
    *,
    private_key_pem: Optional[str] = None,
    key_id: str = "gl-factors-ed25519",
) -> Receipt:
    """Sign with Ed25519 (requires ``cryptography``)."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover
        raise SigningError(
            "ed25519 signing requires the `cryptography` package"
        ) from exc

    pem = private_key_pem or os.getenv("GL_FACTORS_ED25519_PRIVATE_KEY", "")
    if not pem:
        raise SigningError(
            "Ed25519 private key not provided via argument or "
            "GL_FACTORS_ED25519_PRIVATE_KEY env var."
        )
    key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningError("provided key is not Ed25519")

    payload_hash = _canonical_hash(payload)
    sig = key.sign(payload_hash.encode("ascii"))
    return Receipt(
        signature=base64.b64encode(sig).decode("ascii"),
        algorithm="ed25519",
        signed_at=datetime.now(timezone.utc).isoformat(),
        key_id=key_id,
        payload_hash=payload_hash,
    )


__all__ = [
    "Receipt",
    "SigningError",
    "sign_ed25519",
    "sign_sha256_hmac",
    "verify_sha256_hmac",
]
