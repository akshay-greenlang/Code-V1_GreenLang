# -*- coding: utf-8 -*-
"""Authentication providers for the Factors SDK.

Three paths are supported, matching the server side:

1. :class:`JWTAuth`     — ``Authorization: Bearer <jwt>``
2. :class:`APIKeyAuth`  — ``X-API-Key: <key>``
3. :class:`HMACAuth`    — request-signing wrapper for Pro+ tiers

All auth providers implement ``__call__(headers) -> headers`` so the
transport layer can compose them transparently.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)


class AuthProvider:
    """Protocol-style base: mutate headers in place (or return a new dict).

    Implementations MUST NOT log the secret or token.
    """

    def apply(
        self,
        headers: MutableMapping[str, str],
        *,
        method: str = "GET",
        path: str = "",
        body: Optional[bytes] = None,
    ) -> MutableMapping[str, str]:  # pragma: no cover - trivial passthrough
        raise NotImplementedError


# ---------------------------------------------------------------------------
# JWT Bearer
# ---------------------------------------------------------------------------


@dataclass
class JWTAuth(AuthProvider):
    """Attach ``Authorization: Bearer <token>`` to every request."""

    token: str

    def apply(
        self,
        headers: MutableMapping[str, str],
        *,
        method: str = "GET",
        path: str = "",
        body: Optional[bytes] = None,
    ) -> MutableMapping[str, str]:
        if not self.token:
            raise ValueError("JWTAuth requires a non-empty token")
        headers["Authorization"] = "Bearer " + self.token
        return headers


# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------


@dataclass
class APIKeyAuth(AuthProvider):
    """Attach ``X-API-Key: <key>`` to every request.

    The server-side validator is :class:`greenlang.factors.api_auth.APIKeyValidator`
    (see MEMORY).  Keys look like ``gl_fac_<32-char token>``; the SDK
    does not validate the prefix so rotation formats remain flexible.
    """

    api_key: str
    header_name: str = "X-API-Key"

    def apply(
        self,
        headers: MutableMapping[str, str],
        *,
        method: str = "GET",
        path: str = "",
        body: Optional[bytes] = None,
    ) -> MutableMapping[str, str]:
        if not self.api_key:
            raise ValueError("APIKeyAuth requires a non-empty api_key")
        headers[self.header_name] = self.api_key
        return headers


# ---------------------------------------------------------------------------
# HMAC request signing (Pro+ tiers)
# ---------------------------------------------------------------------------


@dataclass
class HMACAuth(AuthProvider):
    """HMAC-SHA256 request signing for Pro+ tiers.

    Composes *around* a primary auth provider (typically APIKeyAuth or
    JWTAuth).  The signature string covers::

        method\\n path\\n x-gl-timestamp\\n x-gl-nonce\\n sha256_hex(body)

    and is placed in ``X-GL-Signature: sha256=<hex>``.  The server side
    regenerates the signature using the tenant's shared secret and
    compares via :func:`hmac.compare_digest`.
    """

    api_key_id: str
    secret: str
    primary: Optional[AuthProvider] = None
    signature_header: str = "X-GL-Signature"
    timestamp_header: str = "X-GL-Timestamp"
    nonce_header: str = "X-GL-Nonce"
    key_id_header: str = "X-GL-Key-Id"

    def apply(
        self,
        headers: MutableMapping[str, str],
        *,
        method: str = "GET",
        path: str = "",
        body: Optional[bytes] = None,
    ) -> MutableMapping[str, str]:
        if self.primary is not None:
            self.primary.apply(headers, method=method, path=path, body=body)

        timestamp = str(int(time.time()))
        nonce = base64.urlsafe_b64encode(
            hashlib.sha256(f"{timestamp}:{self.api_key_id}:{path}".encode("utf-8")).digest()
        ).decode("ascii")[:22]

        body_digest = hashlib.sha256(body or b"").hexdigest()
        canonical = "\n".join(
            [method.upper(), path, timestamp, nonce, body_digest]
        ).encode("utf-8")
        digest = hmac.new(
            self.secret.encode("utf-8"), canonical, hashlib.sha256
        ).hexdigest()

        headers[self.key_id_header] = self.api_key_id
        headers[self.timestamp_header] = timestamp
        headers[self.nonce_header] = nonce
        headers[self.signature_header] = "sha256=" + digest
        return headers


def compose_auth_headers(
    auth: Optional[AuthProvider],
    base_headers: Mapping[str, str],
    *,
    method: str,
    path: str,
    body: Optional[bytes] = None,
) -> Dict[str, str]:
    """Helper for transport: copy ``base_headers`` and apply ``auth``."""
    out: Dict[str, str] = dict(base_headers)
    if auth is not None:
        auth.apply(out, method=method, path=path, body=body)
    return out


__all__ = [
    "AuthProvider",
    "JWTAuth",
    "APIKeyAuth",
    "HMACAuth",
    "compose_auth_headers",
]
