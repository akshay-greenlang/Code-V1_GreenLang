# -*- coding: utf-8 -*-
"""
FastAPI middleware — attach a signed receipt to every 2xx Factors response.

Rationale
---------
CTO non-negotiable (2026-04-22): every API response must carry proof of
what GreenLang returned and when, so regulated customers can defend
their claims in audit and so OEM partners can redistribute signed
artifacts to their own customers.  ``greenlang/factors/signing.py``
already implements both HMAC-SHA256 (symmetric, cheap) and Ed25519
(asymmetric, customer-verifiable).  This middleware wires that engine
into every FastAPI response on the protected prefix so no route has to
remember to sign.

Policy
------
* **Only sign 2xx.**  Error responses (4xx/5xx) are not signed.  The
  auditor semantics are "GreenLang stands behind this answer"; we do
  not stand behind errors.
* **Only sign JSON bodies.**  Streaming / NDJSON / binary responses
  receive a header-only receipt (so the client can still verify the
  content hash of the raw bytes) but are NOT body-mutated.
* **Algorithm per tier:**

    =================== ==============
    Tier                Algorithm
    =================== ==============
    community           sha256-hmac
    pro                 sha256-hmac
    consulting          ed25519 (fallback sha256-hmac)
    platform            ed25519 (fallback sha256-hmac)
    enterprise          ed25519 (fallback sha256-hmac)
    internal            sha256-hmac
    =================== ==============

  Fallback semantics: if Ed25519 is requested but the private key is
  not configured (``GL_FACTORS_ED25519_PRIVATE_KEY`` unset), we emit
  an HMAC receipt instead and log a warning exactly once per process
  start.  This keeps dev and CI environments running without gifting
  customers silent Ed25519-missing failures.
* **Edition-pin into the receipt.**  If the response carries
  ``X-GreenLang-Edition`` (set by the explain / resolve / quality /
  detail routes), the signed payload includes ``edition_id`` so
  customers can independently verify "this response was built from
  edition X at time Y".  Tampering with the edition header after the
  fact invalidates the signature.
* **Receipt placement:**

  - JSON body responses → receipt injected as top-level
    ``_signed_receipt`` object; response body rewritten.
  - Non-JSON / streaming responses → receipt delivered as four HTTP
    headers (``X-GreenLang-Receipt-Signature``, ``-Algorithm``,
    ``-Key-Id``, ``-Signed-At``) plus an ``X-GreenLang-Receipt-Hash``
    that is the SHA-256 of the response body the client will see.

Non-negotiables enforced
------------------------
* Signing never crashes the response.  On any signing error the
  response is returned unmodified and an audit-log line is emitted.
* Constant-time signature generation (via HMAC / Ed25519 library
  primitives); no custom byte-comparisons in this module.
* Secret rotation: receipts carry ``key_id`` so clients verify
  against the correct key.  Rotation happens by updating the env
  var (HMAC) or the Vault mount (Ed25519); no code change needed.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from greenlang.factors.signing import (
    Receipt,
    SigningError,
    sign_ed25519,
    sign_sha256_hmac,
)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tier → algorithm policy.
# --------------------------------------------------------------------------- #

_ED25519_TIERS = frozenset({"consulting", "platform", "enterprise"})
_HMAC_TIERS = frozenset({"community", "pro", "internal"})

ALL_TIERS = _ED25519_TIERS | _HMAC_TIERS

#: One-process-lifetime flag so we warn once (not per request) when the
#: Ed25519 key is missing but requested by tier policy.
_ed25519_fallback_warned = False


def algorithm_for_tier(tier: Optional[str]) -> str:
    """Return ``"ed25519"`` or ``"sha256-hmac"`` for a tier label."""
    if tier is None:
        return "sha256-hmac"
    t = str(tier).lower()
    if t in _ED25519_TIERS:
        return "ed25519"
    return "sha256-hmac"


# --------------------------------------------------------------------------- #
# Signing core.
# --------------------------------------------------------------------------- #


def _sign_payload(payload: Any, *, tier: Optional[str]) -> Optional[Receipt]:
    """
    Produce a :class:`Receipt` for ``payload`` honouring tier policy.

    Returns ``None`` on any signing error (caller then returns the
    response unsigned rather than failing).
    """
    global _ed25519_fallback_warned

    algo = algorithm_for_tier(tier)
    try:
        if algo == "ed25519":
            if os.getenv("GL_FACTORS_ED25519_PRIVATE_KEY"):
                return sign_ed25519(payload)
            if not _ed25519_fallback_warned:
                logger.warning(
                    "Ed25519 requested for tier=%r but "
                    "GL_FACTORS_ED25519_PRIVATE_KEY is not set; "
                    "falling back to sha256-hmac receipts.",
                    tier,
                )
                _ed25519_fallback_warned = True
            # fall through to HMAC
        return sign_sha256_hmac(payload)
    except SigningError as exc:
        logger.warning("Factors signing skipped (SigningError): %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Factors signing skipped (unexpected): %s", exc)
        return None


def _build_signed_payload(
    *,
    body_hash: str,
    edition_id: Optional[str],
    path: str,
    method: str,
    status_code: int,
) -> Dict[str, Any]:
    """
    The minimal object the receipt signs over.

    Using a stable canonical shape means:
    * the receipt never drifts when we add new response fields;
    * the client can verify by reconstructing *only* the body hash +
      edition + path + method + status, no need to round-trip the full
      response payload.
    """
    return {
        "body_hash": body_hash,
        "edition_id": edition_id,
        "path": path,
        "method": method.upper(),
        "status_code": int(status_code),
    }


def _body_hash(raw: bytes) -> str:
    """SHA-256 of the raw response bytes as seen by the client."""
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def _decode_json(raw: bytes) -> Optional[Any]:
    """Try to decode JSON; return ``None`` if not JSON."""
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _attach_receipt_headers(
    headers: Dict[str, str],
    *,
    receipt: Receipt,
    body_hash: str,
) -> None:
    """Set the four standard receipt headers + the body hash header."""
    headers["X-GreenLang-Receipt-Signature"] = receipt.signature
    headers["X-GreenLang-Receipt-Algorithm"] = receipt.algorithm
    headers["X-GreenLang-Receipt-Key-Id"] = receipt.key_id
    headers["X-GreenLang-Receipt-Signed-At"] = receipt.signed_at
    headers["X-GreenLang-Receipt-Hash"] = body_hash


def _inject_receipt_into_json(
    body_json: Any,
    *,
    receipt: Receipt,
    body_hash: str,
    edition_id: Optional[str],
    path: str,
    method: str,
    status_code: int,
) -> Any:
    """
    Return ``body_json`` with a top-level ``_signed_receipt`` attached.

    We only inject on dict top-levels.  Lists and scalars get the
    header-only treatment; we won't wrap them in a container because
    that would break SDK deserialization contracts.
    """
    if isinstance(body_json, dict):
        signed = dict(body_json)
        signed["_signed_receipt"] = {
            **receipt.to_dict(),
            "signed_over": _build_signed_payload(
                body_hash=body_hash,
                edition_id=edition_id,
                path=path,
                method=method,
                status_code=status_code,
            ),
        }
        return signed
    return body_json


# --------------------------------------------------------------------------- #
# FastAPI installation.
# --------------------------------------------------------------------------- #


def install_signing_middleware(
    app: Any,
    *,
    protected_prefix: str = "/api/v1/factors",
    tier_resolver: Optional[Callable[[Any], Optional[str]]] = None,
) -> None:
    """
    Attach the signed-receipts middleware to a FastAPI ``app``.

    Args:
        app: FastAPI application instance.
        protected_prefix: Only requests whose path starts with this
            prefix are signed.  Health / ready endpoints stay unsigned
            by default.
        tier_resolver: Optional callable ``(request) -> tier``.  Defaults
            to reading ``request.state.tier`` (populated by the
            ``auth_metering`` middleware if installed) then falling
            back to the JWT / API-key claims on ``request.state.user``.
    """
    from fastapi import Request
    from fastapi.responses import Response

    resolve_tier = tier_resolver or _default_tier_resolver

    @app.middleware("http")
    async def factors_signing_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if not path.startswith(protected_prefix):
            return await call_next(request)

        response = await call_next(request)

        # Only sign success responses.
        if not (200 <= response.status_code < 300):
            return response

        tier = resolve_tier(request)

        # Capture the body.  FastAPI delivers most responses via
        # streaming; we re-materialize them so the client sees exactly
        # what we sign.  For genuine SSE / WebSocket / NDJSON streams
        # the route should set ``X-GreenLang-Stream: 1`` which we honor
        # by sending a header-only receipt without rewriting the body.
        body_chunks: list[bytes] = []
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            body_chunks.append(chunk)
        raw = b"".join(body_chunks)

        edition_id = response.headers.get("X-GreenLang-Edition") or response.headers.get(
            "X-Factors-Edition"
        )

        body_hash = _body_hash(raw)
        signed_over = _build_signed_payload(
            body_hash=body_hash,
            edition_id=edition_id,
            path=path,
            method=request.method,
            status_code=response.status_code,
        )
        receipt = _sign_payload(signed_over, tier=tier)
        if receipt is None:
            # Return the response unsigned — never crash on signing.
            return Response(
                content=raw,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        # Header-only receipt is always applied.
        new_headers = dict(response.headers)
        _attach_receipt_headers(new_headers, receipt=receipt, body_hash=body_hash)

        # Stream-preserving: if the route opted out of body rewriting,
        # return the bytes as-is.
        if response.headers.get("X-GreenLang-Stream"):
            return Response(
                content=raw,
                status_code=response.status_code,
                headers=new_headers,
                media_type=response.media_type,
            )

        # JSON bodies get the receipt injected into the envelope too.
        body_json = _decode_json(raw)
        if body_json is None or not isinstance(body_json, dict):
            return Response(
                content=raw,
                status_code=response.status_code,
                headers=new_headers,
                media_type=response.media_type,
            )

        signed_body = _inject_receipt_into_json(
            body_json,
            receipt=receipt,
            body_hash=body_hash,
            edition_id=edition_id,
            path=path,
            method=request.method,
            status_code=response.status_code,
        )
        new_body = json.dumps(signed_body, default=str).encode("utf-8")
        # Content-Length must match the new body exactly.
        new_headers.pop("content-length", None)
        new_headers.pop("Content-Length", None)
        new_headers["content-length"] = str(len(new_body))
        return Response(
            content=new_body,
            status_code=response.status_code,
            headers=new_headers,
            media_type="application/json",
        )

    logger.info(
        "Factors signing middleware installed: prefix=%s", protected_prefix
    )


def _default_tier_resolver(request: Any) -> Optional[str]:
    """Pull the tier from the request state, falling back to JWT claims."""
    tier = getattr(getattr(request, "state", None), "tier", None)
    if tier:
        return str(tier).lower()
    user = getattr(getattr(request, "state", None), "user", None)
    if isinstance(user, Mapping):
        t = user.get("tier")
        if t:
            return str(t).lower()
    return None


__all__ = [
    "algorithm_for_tier",
    "install_signing_middleware",
    "ALL_TIERS",
]


# ---------------------------------------------------------------------------
# Class-shape middleware (BaseHTTPMiddleware) for `app.add_middleware(...)`.
# Mirrors install_signing_middleware but applies to /v1 by default and
# also emits the canonical `X-GL-Signature` header alias requested by the
# CTO (the existing X-GreenLang-Receipt-* names are kept for compatibility).
# ---------------------------------------------------------------------------

from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTPMiddleware
from starlette.responses import Response as _StarletteResponse


class SignedReceiptsMiddleware(_BaseHTTPMiddleware):
    """Sign every 2xx JSON response on /v1 with HMAC or Ed25519 per tier.

    Unlike :func:`install_signing_middleware` (which uses an inner
    decorator), this is a Starlette ``BaseHTTPMiddleware`` subclass so it
    composes cleanly with edition-pin / auth-metering / rate-limit /
    licensing-guard middlewares via ``app.add_middleware``.
    """

    BYPASS = {"/v1/health", "/openapi.json", "/docs", "/redoc", "/metrics", "/"}

    def __init__(self, app, *, protected_prefix: str = "/v1",
                 tier_resolver: Optional[Callable[[Any], Optional[str]]] = None) -> None:
        super().__init__(app)
        self._prefix = protected_prefix
        self._resolve_tier = tier_resolver or _default_tier_resolver

    async def dispatch(self, request, call_next) -> _StarletteResponse:
        path = request.url.path
        if path in self.BYPASS or not path.startswith(self._prefix):
            return await call_next(request)

        response = await call_next(request)
        if not (200 <= response.status_code < 300):
            return response

        tier = self._resolve_tier(request)
        # Materialise the body so we sign exactly what the client sees.
        body_chunks: list[bytes] = []
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            body_chunks.append(chunk)
        raw = b"".join(body_chunks)

        edition_id = response.headers.get("X-GreenLang-Edition") or response.headers.get(
            "X-Factors-Edition"
        )
        body_hash = _body_hash(raw)
        signed_over = _build_signed_payload(
            body_hash=body_hash, edition_id=edition_id,
            path=path, method=request.method, status_code=response.status_code,
        )
        receipt = _sign_payload(signed_over, tier=tier)

        new_headers = dict(response.headers)
        if receipt is not None:
            _attach_receipt_headers(new_headers, receipt=receipt, body_hash=body_hash)
            # CTO-canonical alias
            new_headers["X-GL-Signature"] = receipt.signature
            new_headers["X-GL-Signature-Algorithm"] = receipt.algorithm
            new_headers["X-GL-Signature-Key-Id"] = receipt.key_id

        # Inject into JSON body (if applicable) so the client can verify
        # offline using the SDK's verify_receipt() helper.
        if (
            receipt is not None
            and not response.headers.get("X-GreenLang-Stream")
        ):
            body_json = _decode_json(raw)
            if isinstance(body_json, dict):
                signed_body = _inject_receipt_into_json(
                    body_json, receipt=receipt, body_hash=body_hash,
                    edition_id=edition_id, path=path,
                    method=request.method, status_code=response.status_code,
                )
                raw = json.dumps(signed_body, default=str).encode("utf-8")
                new_headers.pop("content-length", None)
                new_headers.pop("Content-Length", None)
                new_headers["content-length"] = str(len(raw))
                return _StarletteResponse(
                    content=raw, status_code=response.status_code,
                    headers=new_headers, media_type="application/json",
                )

        return _StarletteResponse(
            content=raw, status_code=response.status_code,
            headers=new_headers, media_type=response.media_type,
        )


# Append the new symbol to __all__ for clean imports.
__all__ = list(__all__) + ["SignedReceiptsMiddleware"]
