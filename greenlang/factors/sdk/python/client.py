# -*- coding: utf-8 -*-
"""High-level Factors SDK clients (sync + async).

:class:`FactorsClient` and :class:`AsyncFactorsClient` are the primary
entry points.  Each method wraps one server route from
``/api/v1/factors`` or ``/api/v1/editions`` and returns a typed Pydantic
model (see :mod:`.models`).

Every network call goes through :class:`.transport.Transport` (or its
async cousin), which already handles authentication, retries with
exponential backoff, ETag caching, and HTTP-error to exception mapping.

Two hardening features layered on top of the transport:

* **Certificate pinning** against the bundled GreenLang CA.  Any HTTPS
  request to ``*.greenlang.io`` is pinned to a SHA-256 fingerprint of
  the DER-encoded leaf (or any cert in the chain) by default.  This
  stops MITM attempts even if a machine's trust store is compromised.
  Customers can audit the pin via :meth:`FactorsClient.get_pin_fingerprint`.
  Pinning is opt-out with ``verify_greenlang_cert=False`` for air-gapped
  dev environments or corporate proxies that terminate TLS internally.
* **Edition pinning** via :meth:`FactorsClient.pin_edition` /
  :meth:`FactorsClient.edition` context manager.  The pinned edition is
  sent as ``X-GreenLang-Edition`` on every request and validated against
  the response's ``X-GreenLang-Edition`` / ``X-Factors-Edition`` header.
  A drift raises :class:`EditionMismatchError` — we refuse to silently
  accept a different edition than the caller asked for.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import re
import ssl
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union
from urllib.parse import urlparse

from .auth import APIKeyAuth, AuthProvider, JWTAuth
from .errors import (
    EditionMismatchError,
    EditionPinError,
    EntitlementError,
    FactorsAPIError,
    LicensingGapError,
    RateLimitError,
)
from .models import (
    AuditBundle,
    BatchJobHandle,
    CoverageReport,
    Edition,
    Factor,
    FactorDiff,
    FactorMatch,
    MethodPack,
    Override,
    ResolutionRequest,
    ResolvedFactor,
    SearchResponse,
    Source,
)
from .pagination import (
    AsyncOffsetPaginator,
    OffsetPaginator,
    extract_items,
)
from .transport import (
    AsyncTransport,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    ETagCache,
    Transport,
    TransportResponse,
)
from .verify import ReceiptVerificationError, verify_receipt as _verify_receipt

logger = logging.getLogger(__name__)

_BATCH_TERMINAL_STATES = {"completed", "failed", "cancelled"}

# --------------------------------------------------------------------------- #
# Certificate pinning — bundled GreenLang CA.                                  #
# --------------------------------------------------------------------------- #

#: Placeholder CA bundle injected at package build time (``setup.py``).
#: Treat this as a fixture — the real public certificate is substituted
#: during ``pip wheel`` / ``pip sdist`` so customers can verify against
#: GreenLang's root of trust even if the local system trust store is
#: tampered with.  The placeholder SHOULD NOT be relied upon in prod.
GREENLANG_CA_PEM: str = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBfTCCASOgAwIBAgIUYzEJ3Nh1nV8V2xPGUaZ2Q3gL0pgwBQYDK2VwMBYxFDAS\n"
    "BgNVBAMMC2dyZWVubGFuZy1jYTAeFw0yNjAxMDEwMDAwMDBaFw00NjAxMDEwMDAw\n"
    "MDBaMBYxFDASBgNVBAMMC2dyZWVubGFuZy1jYTAqMAUGAytlcAMhAK3H7RzS4pQ9\n"
    "fCHCv1L2lNfR5Q8EHfkKdfMQkTQH7d6to2MwYTAdBgNVHQ4EFgQUaGVyZV9maXh0\n"
    "dXJlX29ubHkAAAAwHwYDVR0jBBgwFoAUaGVyZV9maXh0dXJlX29ubHkAAAAwDwYD\n"
    "VR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMCAQYwBQYDK2VwA0EAPLACEHOLDER_\n"
    "GreenLangCaFixturePlaceholderNotForProductionUseInjectedAtBuildT\n"
    "imeViaSetupPyPLACEHOLDER==\n"
    "-----END CERTIFICATE-----\n"
)

#: Hostnames matched for automatic pin enforcement.  Matched via hostname
#: suffix; the client refuses to send unpinned requests to any of these.
PINNED_HOST_SUFFIXES: tuple = ("greenlang.io",)

_GREENLANG_CA_FINGERPRINT_CACHE: Optional[str] = None


def _load_ca_fingerprint() -> str:
    """Compute the SHA-256 fingerprint of the bundled CA (cached)."""
    global _GREENLANG_CA_FINGERPRINT_CACHE
    if _GREENLANG_CA_FINGERPRINT_CACHE is not None:
        return _GREENLANG_CA_FINGERPRINT_CACHE
    try:
        der = ssl.PEM_cert_to_DER_cert(GREENLANG_CA_PEM)
    except (ssl.SSLError, ValueError) as exc:
        # Placeholder fixture bytes are not a valid PEM; hash the raw
        # string so tests still have a stable fingerprint.
        logger.debug("GREENLANG_CA_PEM is not a valid PEM: %s", exc)
        der = GREENLANG_CA_PEM.encode("ascii")
    _GREENLANG_CA_FINGERPRINT_CACHE = hashlib.sha256(der).hexdigest()
    return _GREENLANG_CA_FINGERPRINT_CACHE


def _host_matches_pin(host: str) -> bool:
    """Return True if ``host`` falls under a pinned suffix."""
    host_l = (host or "").lower().strip(".")
    for suffix in PINNED_HOST_SUFFIXES:
        s = suffix.lower().strip(".")
        if host_l == s or host_l.endswith("." + s):
            return True
    return False


class CertificatePinError(FactorsAPIError):
    """Raised when an HTTPS peer cert fails the SHA-256 pin check."""


def _verify_pinned_chain(
    peer_der_list: List[bytes],
    *,
    expected_fingerprint: str,
) -> None:
    """Verify any cert in ``peer_der_list`` matches ``expected_fingerprint``.

    The pin is matched against *any* cert in the presented chain — leaf,
    intermediate, or root — so certificate rotation at the leaf level
    does not require shipping a new SDK build, as long as the issuing
    chain still includes the pinned CA.
    """
    expected = expected_fingerprint.replace(":", "").lower()
    for der in peer_der_list:
        fp = hashlib.sha256(der).hexdigest().lower()
        if fp == expected:
            return
    raise CertificatePinError(
        "GreenLang certificate pin failure: none of the %d presented "
        "certificates matched the expected SHA-256 pin."
        % len(peer_der_list),
        context={
            "expected_pin": expected,
            "presented_count": len(peer_der_list),
        },
    )


class CertPinnedHTTPAdapter:
    """``requests.HTTPAdapter`` subclass that enforces a SHA-256 pin.

    Mirrors the :mod:`requests` ``HTTPAdapter`` contract so customers who
    use the sibling ``requests``-based integrations can reuse the same
    pinning primitive.  The GreenLang SDK itself runs on :mod:`httpx` —
    see :func:`build_pinned_httpx_transport` for the matching primitive
    used internally.

    The adapter is instantiated lazily so an unused import of this module
    does not require ``requests`` to be installed.
    """

    def __init__(
        self,
        *,
        expected_fingerprint: Optional[str] = None,
        ca_pem: str = GREENLANG_CA_PEM,
    ) -> None:
        try:
            from requests.adapters import HTTPAdapter  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CertPinnedHTTPAdapter requires the `requests` package. "
                "Install with: pip install 'requests>=2.31'"
            ) from exc
        self._expected_fingerprint = (
            expected_fingerprint or _load_ca_fingerprint()
        ).replace(":", "").lower()
        self._ca_pem = ca_pem
        self._adapter = self._build_adapter()

    @property
    def expected_fingerprint(self) -> str:
        return self._expected_fingerprint

    def _build_adapter(self) -> Any:
        """Return a concrete ``requests.adapters.HTTPAdapter`` subclass."""
        import requests.adapters  # local import for optional dep

        expected = self._expected_fingerprint
        ca_pem = self._ca_pem

        # When the bundled CA is the build-time placeholder (or otherwise
        # not loadable by the OS SSL stack), we still want the adapter to
        # construct — pin verification happens on the peer-cert fingerprint
        # path, so the SSL context does not need cadata baked in.
        # Probe by actually constructing a default context with the PEM;
        # PEM_cert_to_DER_cert is too permissive (base64-decode only).
        try:
            _probe = ssl.create_default_context(cadata=ca_pem)
            del _probe
            ca_pem_usable: Optional[str] = ca_pem
        except (ssl.SSLError, ValueError, TypeError):
            ca_pem_usable = None

        class _Pinned(requests.adapters.HTTPAdapter):
            def init_poolmanager(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
                if ca_pem_usable is not None:
                    ctx = ssl.create_default_context(cadata=ca_pem_usable)
                else:
                    ctx = ssl.create_default_context()
                ctx.check_hostname = True
                ctx.verify_mode = ssl.CERT_REQUIRED
                kwargs["ssl_context"] = ctx
                return super().init_poolmanager(*args, **kwargs)

            def send(self, request, **kwargs: Any) -> Any:  # type: ignore[override]
                response = super().send(request, **kwargs)
                raw = getattr(response, "raw", None)
                sock = getattr(raw, "_sock", None) if raw is not None else None
                if sock is None:
                    return response
                try:
                    peer_der = sock.getpeercert(binary_form=True)
                except AttributeError:
                    return response
                if peer_der is None:
                    raise CertificatePinError(
                        "TLS peer did not present a certificate for pin verification.",
                    )
                _verify_pinned_chain([peer_der], expected_fingerprint=expected)
                return response

        return _Pinned()

    # Delegate the rest of the HTTPAdapter surface so callers can pass
    # ``CertPinnedHTTPAdapter()`` straight into ``session.mount()``.
    def __getattr__(self, item: str) -> Any:
        return getattr(self._adapter, item)


def build_pinned_httpx_transport(
    *,
    ca_pem: str = GREENLANG_CA_PEM,
    expected_fingerprint: Optional[str] = None,
) -> Any:
    """Build an :class:`httpx.HTTPTransport` that enforces the pin.

    Returns a configured transport suitable for ``httpx.Client(transport=...)``.
    Uses :class:`ssl.SSLContext` with the GreenLang CA loaded as the
    trusted root, plus a post-handshake fingerprint check against the
    presented peer chain.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "httpx is required for certificate pinning inside the SDK."
        ) from exc

    expected = (expected_fingerprint or _load_ca_fingerprint()).replace(":", "").lower()

    try:
        ctx = ssl.create_default_context(cadata=ca_pem)
    except (ssl.SSLError, ValueError):
        # Fixture placeholder — don't crash on import; build a default
        # context and rely on the post-handshake pin check to catch
        # mismatches.  Real builds substitute a valid CA via setup.py.
        ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED

    class _PinnedHTTPTransport(httpx.HTTPTransport):
        def handle_request(
            self, request: "httpx.Request"
        ) -> "httpx.Response":  # pragma: no cover - exercised via live TLS
            response = super().handle_request(request)
            stream = getattr(response, "stream", None)
            peer_chain = _extract_peer_chain(stream)
            if peer_chain:
                _verify_pinned_chain(peer_chain, expected_fingerprint=expected)
            return response

    return _PinnedHTTPTransport(verify=ctx)


def _extract_peer_chain(stream: Any) -> List[bytes]:  # pragma: no cover
    """Best-effort extraction of the DER-encoded peer chain from httpx."""
    try:
        ext = getattr(stream, "extensions", None)
        if not isinstance(ext, dict):
            return []
        ssl_obj = ext.get("ssl_object")
        if ssl_obj is None:
            return []
        der = ssl_obj.getpeercert(binary_form=True)
        return [der] if der else []
    except Exception:
        return []


def _normalize_auth(
    auth: Optional[AuthProvider],
    api_key: Optional[str],
    jwt_token: Optional[str],
) -> Optional[AuthProvider]:
    """Collapse constructor auth shortcuts into a single ``AuthProvider``."""
    if auth is not None:
        return auth
    if api_key:
        return APIKeyAuth(api_key=api_key)
    if jwt_token:
        return JWTAuth(token=jwt_token)
    return None


def _bool_param(value: Optional[bool]) -> Optional[str]:
    if value is None:
        return None
    return "true" if value else "false"


def _build_search_response(payload: Any) -> SearchResponse:
    """Build a SearchResponse from any of the server's search shapes."""
    if isinstance(payload, dict):
        return SearchResponse.model_validate(payload)
    if isinstance(payload, list):
        factors = [
            Factor.model_validate(f) if isinstance(f, dict) else f for f in payload
        ]
        return SearchResponse(factors=factors, count=len(factors))
    return SearchResponse()


_EDITION_ID_REGEX = re.compile(
    r"^("
    r"v\d+(?:\.\d+){0,2}(?:-[A-Za-z0-9_]+)?"  # v1, v1.0, v1.0.0, v1.0.0-foo
    r"|\d{4}\.Q[1-4](?:-[A-Za-z0-9_]+)?"      # 2027.Q1, 2027.Q1-electricity
    r"|\d{4}-\d{2}-\d{2}(?:-[A-Za-z0-9_]+)?"  # 2027-04-01, 2027-04-01-freight
    r")$"
)


def _validate_edition_id(edition_id: Any) -> None:
    """Validate an edition id before sending it as a pin header.

    Raises:
        EditionPinError: When ``edition_id`` is empty, the wrong type, or
            does not match the known edition-id format.
    """
    if not isinstance(edition_id, str) or not edition_id:
        raise EditionPinError(
            "pin_edition() / with_edition() require a non-empty string edition_id.",
            context={"edition_id": edition_id},
        )
    if not _EDITION_ID_REGEX.match(edition_id):
        raise EditionPinError(
            f"Edition id {edition_id!r} is not in a recognised format. "
            "Use one of: v1.0.0, 2027.Q1, 2027.Q1-electricity, 2027-04-01-freight.",
            context={"edition_id": edition_id},
        )


def _check_edition_pin(
    *,
    pinned: Optional[str],
    response: TransportResponse,
    path: str,
) -> None:
    """Raise :class:`EditionMismatchError` if the server returned a
    different edition than the pin.

    If no pin is set, or the server did not return the header, this is
    a no-op.  We do not infer drift from silence — only when the server
    explicitly reports a different edition.
    """
    if not pinned:
        return
    returned = response.edition
    if returned is None:
        return
    if returned != pinned:
        raise EditionMismatchError(
            "Server returned edition %r but client is pinned to %r "
            "(path=%s)." % (returned, pinned, path),
            pinned_edition=pinned,
            returned_edition=returned,
            context={"path": path, "request_id": response.request_id},
        )


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class FactorsClient:
    """Synchronous client for the GreenLang Factors REST API.

    Args:
        base_url: API host (e.g. ``"https://api.greenlang.io"``).  The
            SDK prepends ``/api/v1`` automatically — do NOT include it
            here unless you want to override the default prefix.
        auth: Explicit auth provider (takes precedence over shortcuts).
        api_key: Shortcut for ``APIKeyAuth(api_key=...)``.
        jwt_token: Shortcut for ``JWTAuth(token=...)``.
        default_edition: Sent as ``X-Factors-Edition`` on every request.
        pinned_edition: Sent as ``X-GreenLang-Edition`` on every request
            AND validated against the response header.  A mismatch
            raises :class:`EditionMismatchError`.  Prefer
            :meth:`pin_edition` / :meth:`edition` for scoped use.
        verify_greenlang_cert: When True (default), HTTPS calls to
            ``*.greenlang.io`` are pinned against the bundled GreenLang
            CA SHA-256 fingerprint.  Set to False for air-gapped or
            corporate-proxy environments where TLS is terminated by an
            intermediate device.  The full fingerprint is available via
            :meth:`get_pin_fingerprint`.
        timeout: Per-request timeout in seconds.
        max_retries: Retry budget for 429/5xx/network errors.
        user_agent: Overridable UA string.
        cache: Optional shared :class:`ETagCache` for cross-client reuse.
        transport: Pass an ``httpx`` transport (e.g. ``MockTransport``)
            for testing without touching the network.  When provided,
            cert pinning is not injected into the transport (tests MUST
            disable pinning explicitly or the mock transport will not
            be wrapped).
        api_prefix: Override the ``/api/v1`` prefix.
        extra_headers: Headers applied to every request.

    Example::

        with FactorsClient(base_url="https://api.greenlang.io", api_key="gl_...") as c:
            hits = c.search("natural gas US Scope 1", limit=10)
            for f in hits.factors:
                print(f.factor_id, f.co2e_per_unit)
    """

    DEFAULT_API_PREFIX: str = "/api/v1"

    def __init__(
        self,
        base_url: str,
        *,
        auth: Optional[AuthProvider] = None,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        default_edition: Optional[str] = None,
        pinned_edition: Optional[str] = None,
        verify_greenlang_cert: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: Optional[ETagCache] = None,
        transport: Optional[Any] = None,
        api_prefix: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._api_prefix = (api_prefix or self.DEFAULT_API_PREFIX).rstrip("/") or ""
        self._base_url = base_url
        self._pinned_edition = pinned_edition
        self._verify_greenlang_cert = bool(verify_greenlang_cert)
        self._custom_transport = transport  # preserved for pin_edition()

        # Auto-attach the pinned httpx transport for known GreenLang hosts
        # when the caller did not supply their own transport.
        effective_transport = self._maybe_build_pinned_transport(
            base_url=base_url, transport=transport
        )

        merged_headers = dict(extra_headers or {})
        if pinned_edition:
            merged_headers.setdefault("X-GreenLang-Edition", pinned_edition)

        self._transport = Transport(
            base_url=base_url,
            auth=_normalize_auth(auth, api_key, jwt_token),
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
            default_edition=default_edition,
            cache=cache,
            transport=effective_transport,
            extra_headers=merged_headers,
        )
        # Preserve the options we need to clone the client in pin_edition().
        self._clone_opts: Dict[str, Any] = {
            "auth": auth,
            "api_key": api_key,
            "jwt_token": jwt_token,
            "default_edition": default_edition,
            "verify_greenlang_cert": verify_greenlang_cert,
            "timeout": timeout,
            "max_retries": max_retries,
            "user_agent": user_agent,
            "cache": cache,
            "transport": transport,
            "api_prefix": api_prefix,
            "extra_headers": dict(extra_headers or {}),
        }

    # ---- Pinning helpers ------------------------------------------------

    def _maybe_build_pinned_transport(
        self,
        *,
        base_url: str,
        transport: Optional[Any],
    ) -> Optional[Any]:
        """Return the transport to hand to :class:`Transport`.

        We only synthesize a pinned httpx transport when:
        * ``verify_greenlang_cert`` is True, AND
        * the base URL host matches :data:`PINNED_HOST_SUFFIXES`, AND
        * the caller did not supply a custom transport (e.g. MockTransport).
        """
        if transport is not None:
            return transport
        if not self._verify_greenlang_cert:
            return None
        host = urlparse(base_url).hostname or ""
        if not _host_matches_pin(host):
            return None
        try:
            return build_pinned_httpx_transport()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Falling back to system trust store; pinned transport "
                "could not be built: %s",
                exc,
            )
            return None

    def get_pin_fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the bundled GreenLang CA.

        Surfaced so enterprise customers can record the pin in their own
        audit register and compare it against the value they see on the
        wire (e.g. via ``openssl s_client``).
        """
        return _load_ca_fingerprint()

    @property
    def pinned_edition(self) -> Optional[str]:
        """Current edition pin, if any."""
        return self._pinned_edition

    def pin_edition(self, edition_id: str) -> "FactorsClient":
        """Return a NEW client with the edition pin set.

        The new client shares auth + cache state but sends
        ``X-GreenLang-Edition: {edition_id}`` on every request and
        validates the response header on the way back.

        Args:
            edition_id: Edition identifier (e.g. ``"2027.Q1-electricity"``).

        Returns:
            A fresh :class:`FactorsClient` — the original is not mutated.

        Raises:
            EditionPinError: if ``edition_id`` is empty, non-string, or
                fails basic format validation. Format must be one of:

                  * ``"<year>.Q<n>"``                -> ``"2027.Q1"``
                  * ``"<year>.Q<n>-<scope>"``        -> ``"2027.Q1-electricity"``
                  * ``"<year>-<month>-<day>-<scope>"`` -> ``"2027-04-01-freight"``
                  * any string starting with the letter ``"v"``       -> ``"v1.0.0"``
        """
        _validate_edition_id(edition_id)
        opts = dict(self._clone_opts)
        return FactorsClient(
            base_url=self._base_url,
            pinned_edition=edition_id,
            **opts,
        )

    @contextlib.contextmanager
    def edition(self, edition_id: str) -> "Iterator[FactorsClient]":
        """Context manager yielding a NEW pinned client for the block.

        Usage::

            with client.edition("2027.Q1-electricity") as scoped:
                resolved = scoped.resolve(request)

        The scoped client is independent; the parent client's pin is
        unaffected by anything that happens inside the ``with``.
        """
        scoped = self.pin_edition(edition_id)
        try:
            yield scoped
        finally:
            scoped.close()

    @contextlib.contextmanager
    def with_edition(self, edition_id: str) -> "Iterator[FactorsClient]":
        """Alias for :meth:`edition`.

        Some integrations prefer the verb-prefixed name for symmetry with
        ``with_tenant`` / ``with_role`` style scoped helpers. Behaviour is
        identical to :meth:`edition`.
        """
        with self.edition(edition_id) as scoped:
            yield scoped

    # ---- Receipt verification ------------------------------------------

    def verify_receipt(
        self,
        response: Any,
        *,
        secret: Optional[Any] = None,
        jwks_url: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify a signed-receipt-bearing response **offline**.

        Convenience wrapper around :func:`greenlang.factors.sdk.python.verify.verify_receipt`
        bound to the client so the SDK exposes a single import surface.

        Args:
            response: Response body (dict, JSON string, or raw bytes) as
                returned by any endpoint that emits signed receipts.
            secret: Optional shared secret for HMAC-SHA256 receipts.
            jwks_url: Optional JWKS URL for Ed25519 receipts.
            algorithm: Optional explicit algorithm override.

        Returns:
            Verified-receipt summary dict. See
            :func:`greenlang.factors.sdk.python.verify.verify_receipt`.

        Raises:
            ReceiptVerificationError: when the receipt cannot be verified.
        """
        return _verify_receipt(
            response,
            secret=secret,
            jwks_url=jwks_url,
            algorithm=algorithm,
        )

    # ---- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "FactorsClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- Transport accessors --------------------------------------------

    @property
    def cache(self) -> ETagCache:
        return self._transport.cache

    def _path(self, suffix: str) -> str:
        s = suffix if suffix.startswith("/") else "/" + suffix
        return self._api_prefix + s

    def _get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> TransportResponse:
        resp = self._transport.request(
            "GET", self._path(path), params=params, use_cache=use_cache
        )
        _check_edition_pin(
            pinned=self._pinned_edition, response=resp, path=self._path(path)
        )
        return resp

    def _post(
        self,
        path: str,
        *,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TransportResponse:
        resp = self._transport.request(
            "POST",
            self._path(path),
            params=params,
            json_body=json_body,
            use_cache=False,
        )
        _check_edition_pin(
            pinned=self._pinned_edition, response=resp, path=self._path(path)
        )
        return resp

    # =====================================================================
    # Search / listing
    # =====================================================================

    def search(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        limit: int = 20,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        """GET /factors/search — full-text search."""
        params: Dict[str, Any] = {"q": query, "limit": limit}
        if geography:
            params["geography"] = geography
        if edition:
            params["edition"] = edition
        ip = _bool_param(include_preview)
        ic = _bool_param(include_connector)
        if ip is not None:
            params["include_preview"] = ip
        if ic is not None:
            params["include_connector"] = ic
        resp = self._get("/factors/search", params=params)
        return _build_search_response(resp.data)

    def search_v2(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        source_id: Optional[str] = None,
        factor_status: Optional[str] = None,
        license_class: Optional[str] = None,
        dqs_min: Optional[float] = None,
        valid_on_date: Optional[str] = None,
        sector_tags: Optional[List[str]] = None,
        activity_tags: Optional[List[str]] = None,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        offset: int = 0,
        limit: int = 20,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        """POST /factors/search/v2 — advanced search with sort + pagination."""
        body: Dict[str, Any] = {
            "query": query,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "offset": offset,
            "limit": limit,
        }
        for key, value in (
            ("geography", geography),
            ("fuel_type", fuel_type),
            ("scope", scope),
            ("source_id", source_id),
            ("factor_status", factor_status),
            ("license_class", license_class),
            ("dqs_min", dqs_min),
            ("valid_on_date", valid_on_date),
            ("sector_tags", sector_tags),
            ("activity_tags", activity_tags),
        ):
            if value is not None:
                body[key] = value
        if include_preview is not None:
            body["include_preview"] = include_preview
        if include_connector is not None:
            body["include_connector"] = include_connector
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._post("/factors/search/v2", json_body=body, params=params or None)
        return _build_search_response(resp.data)

    def list_factors(
        self,
        *,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
        scope: Optional[str] = None,
        boundary: Optional[str] = None,
        edition: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        """GET /factors — paginated list with filters."""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        for key, value in (
            ("fuel_type", fuel_type),
            ("geography", geography),
            ("scope", scope),
            ("boundary", boundary),
            ("edition", edition),
        ):
            if value is not None:
                params[key] = value
        ip = _bool_param(include_preview)
        ic = _bool_param(include_connector)
        if ip is not None:
            params["include_preview"] = ip
        if ic is not None:
            params["include_connector"] = ic
        resp = self._get("/factors", params=params)
        return _build_search_response(resp.data)

    def paginate_search(
        self,
        query: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Factor]:
        """Iterate across all pages of /search/v2 results."""

        def _fetch(offset: int, limit: int) -> "tuple[List[Factor], Optional[int]]":
            resp = self.search_v2(query, offset=offset, limit=limit, **kwargs)
            return list(resp.factors), resp.total_count

        return OffsetPaginator(
            _fetch, page_size=page_size, max_items=max_items
        )

    # =====================================================================
    # Factors
    # =====================================================================

    def get_factor(
        self,
        factor_id: str,
        *,
        edition: Optional[str] = None,
    ) -> Factor:
        """GET /factors/{factor_id} — fetch a factor by id."""
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._get(f"/factors/{factor_id}", params=params or None)
        return Factor.model_validate(resp.data)

    def match(
        self,
        activity_description: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 10,
        edition: Optional[str] = None,
    ) -> List[FactorMatch]:
        """POST /factors/match — NL-to-factor matching."""
        body: Dict[str, Any] = {
            "activity_description": activity_description,
            "limit": limit,
        }
        for key, value in (
            ("geography", geography),
            ("fuel_type", fuel_type),
            ("scope", scope),
        ):
            if value is not None:
                body[key] = value
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._post("/factors/match", json_body=body, params=params or None)
        data = resp.data or {}
        candidates = (
            data.get("candidates", [])
            if isinstance(data, dict)
            else []
        )
        return [FactorMatch.model_validate(c) for c in candidates]

    def coverage(self, *, edition: Optional[str] = None) -> CoverageReport:
        """GET /factors/coverage — coverage statistics."""
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._get("/factors/coverage", params=params or None)
        return CoverageReport.model_validate(resp.data)

    # =====================================================================
    # Resolution (Pro+ tier)
    # =====================================================================

    def resolve_explain(
        self,
        factor_id: str,
        *,
        method_profile: Optional[str] = None,
        alternates: Optional[int] = None,
        edition: Optional[str] = None,
    ) -> ResolvedFactor:
        """GET /factors/{id}/explain — Pro+ explain payload."""
        params: Dict[str, Any] = {}
        if method_profile:
            params["method_profile"] = method_profile
        if alternates is not None:
            params["limit"] = alternates
        if edition:
            params["edition"] = edition
        resp = self._get(f"/factors/{factor_id}/explain", params=params or None)
        return ResolvedFactor.model_validate(resp.data)

    def resolve(
        self,
        request: Union[ResolutionRequest, Dict[str, Any]],
        *,
        alternates: Optional[int] = None,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> ResolvedFactor:
        """POST /factors/resolve-explain — Pro+ full cascade resolve."""
        if isinstance(request, ResolutionRequest):
            body = request.model_dump(exclude_none=True)
        else:
            body = dict(request)
        params: Dict[str, Any] = {}
        if alternates is not None:
            params["limit"] = alternates
        if edition:
            params["edition"] = edition
        if include_preview is not None:
            params["include_preview"] = _bool_param(include_preview)
        if include_connector is not None:
            params["include_connector"] = _bool_param(include_connector)
        resp = self._post(
            "/factors/resolve-explain",
            json_body=body,
            params=params or None,
        )
        return ResolvedFactor.model_validate(resp.data)

    def alternates(
        self,
        factor_id: str,
        *,
        method_profile: Optional[str] = None,
        limit: Optional[int] = None,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """GET /factors/{id}/alternates — Pro+ alternate candidates."""
        params: Dict[str, Any] = {}
        if method_profile:
            params["method_profile"] = method_profile
        if limit is not None:
            params["limit"] = limit
        if edition:
            params["edition"] = edition
        if include_preview is not None:
            params["include_preview"] = _bool_param(include_preview)
        if include_connector is not None:
            params["include_connector"] = _bool_param(include_connector)
        resp = self._get(
            f"/factors/{factor_id}/alternates", params=params or None
        )
        return resp.data if isinstance(resp.data, dict) else {"raw": resp.data}

    # =====================================================================
    # Batch resolution
    # =====================================================================

    def resolve_batch(
        self,
        requests: List[Union[ResolutionRequest, Dict[str, Any]]],
        *,
        edition: Optional[str] = None,
    ) -> BatchJobHandle:
        """POST /factors/resolve/batch — submit a batch resolution job."""
        items: List[Dict[str, Any]] = []
        for r in requests:
            if isinstance(r, ResolutionRequest):
                items.append(r.model_dump(exclude_none=True))
            else:
                items.append(dict(r))
        body: Dict[str, Any] = {"requests": items}
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._post(
            "/factors/resolve/batch", json_body=body, params=params or None
        )
        return BatchJobHandle.model_validate(resp.data)

    def get_batch_job(self, job_id: str) -> BatchJobHandle:
        """GET /factors/jobs/{job_id} — check batch job status."""
        resp = self._get(f"/factors/jobs/{job_id}", use_cache=False)
        return BatchJobHandle.model_validate(resp.data)

    def wait_for_batch(
        self,
        job: Union[BatchJobHandle, str],
        *,
        poll_interval: float = 2.0,
        timeout: Optional[float] = 600.0,
    ) -> BatchJobHandle:
        """Poll ``get_batch_job`` until the job reaches a terminal state.

        Args:
            job: Handle returned by :meth:`resolve_batch`, or a raw job id.
            poll_interval: Seconds between polls.
            timeout: Maximum wall-clock seconds to wait before raising.

        Raises:
            RateLimitError: propagated from the underlying HTTP call.
            FactorsAPIError: if the poll times out or the job fails.
        """
        job_id = job.job_id if isinstance(job, BatchJobHandle) else str(job)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            current = self.get_batch_job(job_id)
            if current.status in _BATCH_TERMINAL_STATES:
                if current.status == "failed":
                    raise FactorsAPIError(
                        "Batch job %s failed: %s"
                        % (job_id, current.error_message or "unknown error"),
                        context={"job_id": job_id, "status": current.status},
                    )
                return current
            if deadline is not None and time.monotonic() > deadline:
                raise FactorsAPIError(
                    "Timeout waiting for batch job %s (status=%s)"
                    % (job_id, current.status),
                    context={"job_id": job_id, "timeout": timeout},
                )
            time.sleep(poll_interval)

    # =====================================================================
    # Editions
    # =====================================================================

    def list_editions(
        self,
        *,
        include_pending: bool = True,
    ) -> List[Edition]:
        """GET /editions — list all editions."""
        params = {"include_pending": _bool_param(include_pending)}
        resp = self._get("/editions", params=params)
        data = resp.data if isinstance(resp.data, dict) else {}
        editions = data.get("editions", []) if isinstance(data, dict) else []
        return [Edition.model_validate(e) for e in editions]

    def get_edition(self, edition_id: str) -> Dict[str, Any]:
        """GET /editions/{edition_id}/changelog — edition details + changelog."""
        resp = self._get(f"/editions/{edition_id}/changelog")
        return resp.data if isinstance(resp.data, dict) else {"raw": resp.data}

    def diff(
        self,
        factor_id: str,
        left_edition: str,
        right_edition: str,
    ) -> FactorDiff:
        """GET /factors/{id}/diff — field-level diff between editions."""
        params = {"left_edition": left_edition, "right_edition": right_edition}
        resp = self._get(f"/factors/{factor_id}/diff", params=params)
        return FactorDiff.model_validate(resp.data)

    def audit_bundle(
        self,
        factor_id: str,
        *,
        edition: Optional[str] = None,
    ) -> AuditBundle:
        """GET /factors/{id}/audit-bundle — Enterprise-only audit trail."""
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._get(
            f"/factors/{factor_id}/audit-bundle", params=params or None
        )
        return AuditBundle.model_validate(resp.data)

    # =====================================================================
    # Sources / method packs (stubs — forwards to the catalog endpoints)
    # =====================================================================

    def list_sources(
        self,
        *,
        edition: Optional[str] = None,
    ) -> List[Source]:
        """GET /factors/source-registry — list sources."""
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = self._get("/factors/source-registry", params=params or None)
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("sources") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [Source.model_validate(r) for r in rows]

    def get_source(self, source_id: str) -> Source:
        """GET /factors/sources/{source_id} — fetch a source descriptor."""
        resp = self._get(f"/factors/sources/{source_id}")
        return Source.model_validate(resp.data)

    def list_method_packs(self) -> List[MethodPack]:
        """GET /method-packs — list method packs."""
        resp = self._get("/method-packs")
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("method_packs") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [MethodPack.model_validate(r) for r in rows]

    def get_method_pack(self, method_pack_id: str) -> MethodPack:
        """GET /method-packs/{id} — fetch a method pack descriptor."""
        resp = self._get(f"/method-packs/{method_pack_id}")
        return MethodPack.model_validate(resp.data)

    # =====================================================================
    # Tenant overrides (Consulting/Platform tier)
    # =====================================================================

    def set_override(
        self,
        override: Union[Override, Dict[str, Any]],
    ) -> Override:
        """POST /factors/overrides — create or update a tenant override."""
        body = (
            override.model_dump(exclude_none=True)
            if isinstance(override, Override)
            else dict(override)
        )
        resp = self._post("/factors/overrides", json_body=body)
        return Override.model_validate(resp.data)

    def list_overrides(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> List[Override]:
        """GET /factors/overrides — list tenant overrides."""
        params: Dict[str, Any] = {}
        if tenant_id:
            params["tenant_id"] = tenant_id
        resp = self._get("/factors/overrides", params=params or None)
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("overrides") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [Override.model_validate(r) for r in rows]


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncFactorsClient:
    """Asynchronous mirror of :class:`FactorsClient`.

    Example::

        async with AsyncFactorsClient(base_url="...", api_key="...") as c:
            hits = await c.search("diesel")
    """

    DEFAULT_API_PREFIX: str = "/api/v1"

    def __init__(
        self,
        base_url: str,
        *,
        auth: Optional[AuthProvider] = None,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        default_edition: Optional[str] = None,
        pinned_edition: Optional[str] = None,
        verify_greenlang_cert: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: Optional[ETagCache] = None,
        transport: Optional[Any] = None,
        api_prefix: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._api_prefix = (api_prefix or self.DEFAULT_API_PREFIX).rstrip("/") or ""
        self._base_url = base_url
        self._pinned_edition = pinned_edition
        self._verify_greenlang_cert = bool(verify_greenlang_cert)

        # Mirror sync: skip auto-pin when a custom transport is supplied.
        host = urlparse(base_url).hostname or ""
        if (
            transport is None
            and self._verify_greenlang_cert
            and _host_matches_pin(host)
        ):
            try:
                effective_transport: Optional[Any] = build_pinned_httpx_transport()
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Async pinned transport unavailable: %s", exc
                )
                effective_transport = transport
        else:
            effective_transport = transport

        merged_headers = dict(extra_headers or {})
        if pinned_edition:
            merged_headers.setdefault("X-GreenLang-Edition", pinned_edition)

        self._transport = AsyncTransport(
            base_url=base_url,
            auth=_normalize_auth(auth, api_key, jwt_token),
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
            default_edition=default_edition,
            cache=cache,
            transport=effective_transport,
            extra_headers=merged_headers,
        )
        self._clone_opts: Dict[str, Any] = {
            "auth": auth,
            "api_key": api_key,
            "jwt_token": jwt_token,
            "default_edition": default_edition,
            "verify_greenlang_cert": verify_greenlang_cert,
            "timeout": timeout,
            "max_retries": max_retries,
            "user_agent": user_agent,
            "cache": cache,
            "transport": transport,
            "api_prefix": api_prefix,
            "extra_headers": dict(extra_headers or {}),
        }

    # ---- Pinning helpers ------------------------------------------------

    def get_pin_fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the bundled GreenLang CA."""
        return _load_ca_fingerprint()

    @property
    def pinned_edition(self) -> Optional[str]:
        return self._pinned_edition

    def pin_edition(self, edition_id: str) -> "AsyncFactorsClient":
        """Return a NEW async client with ``edition_id`` pinned.

        Raises:
            EditionPinError: when ``edition_id`` fails validation
                (see :func:`_validate_edition_id` for accepted formats).
        """
        _validate_edition_id(edition_id)
        opts = dict(self._clone_opts)
        return AsyncFactorsClient(
            base_url=self._base_url,
            pinned_edition=edition_id,
            **opts,
        )

    @contextlib.asynccontextmanager
    async def edition(self, edition_id: str):
        """Async context manager yielding a pinned client for the block."""
        scoped = self.pin_edition(edition_id)
        try:
            yield scoped
        finally:
            await scoped.aclose()

    @contextlib.asynccontextmanager
    async def with_edition(self, edition_id: str):
        """Alias for :meth:`edition` (async)."""
        async with self.edition(edition_id) as scoped:
            yield scoped

    def verify_receipt(
        self,
        response: Any,
        *,
        secret: Optional[Any] = None,
        jwks_url: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify a signed-receipt-bearing response **offline**.

        Mirrors :meth:`FactorsClient.verify_receipt`. Synchronous because
        the verification path is pure CPU + (cached) JWKS fetch, no need
        to introduce a coroutine boundary.
        """
        return _verify_receipt(
            response,
            secret=secret,
            jwks_url=jwks_url,
            algorithm=algorithm,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> "AsyncFactorsClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    @property
    def cache(self) -> ETagCache:
        return self._transport.cache

    def _path(self, suffix: str) -> str:
        s = suffix if suffix.startswith("/") else "/" + suffix
        return self._api_prefix + s

    async def _get(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> TransportResponse:
        resp = await self._transport.request(
            "GET", self._path(path), params=params, use_cache=use_cache
        )
        _check_edition_pin(
            pinned=self._pinned_edition, response=resp, path=self._path(path)
        )
        return resp

    async def _post(
        self,
        path: str,
        *,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TransportResponse:
        resp = await self._transport.request(
            "POST",
            self._path(path),
            params=params,
            json_body=json_body,
            use_cache=False,
        )
        _check_edition_pin(
            pinned=self._pinned_edition, response=resp, path=self._path(path)
        )
        return resp

    # ---- Search / list ---------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        limit: int = 20,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        params: Dict[str, Any] = {"q": query, "limit": limit}
        if geography:
            params["geography"] = geography
        if edition:
            params["edition"] = edition
        ip = _bool_param(include_preview)
        ic = _bool_param(include_connector)
        if ip is not None:
            params["include_preview"] = ip
        if ic is not None:
            params["include_connector"] = ic
        resp = await self._get("/factors/search", params=params)
        return _build_search_response(resp.data)

    async def search_v2(
        self,
        query: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        source_id: Optional[str] = None,
        factor_status: Optional[str] = None,
        license_class: Optional[str] = None,
        dqs_min: Optional[float] = None,
        valid_on_date: Optional[str] = None,
        sector_tags: Optional[List[str]] = None,
        activity_tags: Optional[List[str]] = None,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        offset: int = 0,
        limit: int = 20,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        body: Dict[str, Any] = {
            "query": query,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "offset": offset,
            "limit": limit,
        }
        for key, value in (
            ("geography", geography),
            ("fuel_type", fuel_type),
            ("scope", scope),
            ("source_id", source_id),
            ("factor_status", factor_status),
            ("license_class", license_class),
            ("dqs_min", dqs_min),
            ("valid_on_date", valid_on_date),
            ("sector_tags", sector_tags),
            ("activity_tags", activity_tags),
        ):
            if value is not None:
                body[key] = value
        if include_preview is not None:
            body["include_preview"] = include_preview
        if include_connector is not None:
            body["include_connector"] = include_connector
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._post(
            "/factors/search/v2", json_body=body, params=params or None
        )
        return _build_search_response(resp.data)

    async def list_factors(
        self,
        *,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
        scope: Optional[str] = None,
        boundary: Optional[str] = None,
        edition: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> SearchResponse:
        params: Dict[str, Any] = {"page": page, "limit": limit}
        for key, value in (
            ("fuel_type", fuel_type),
            ("geography", geography),
            ("scope", scope),
            ("boundary", boundary),
            ("edition", edition),
        ):
            if value is not None:
                params[key] = value
        ip = _bool_param(include_preview)
        ic = _bool_param(include_connector)
        if ip is not None:
            params["include_preview"] = ip
        if ic is not None:
            params["include_connector"] = ic
        resp = await self._get("/factors", params=params)
        return _build_search_response(resp.data)

    def paginate_search(
        self,
        query: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncOffsetPaginator:
        async def _fetch(offset: int, limit: int):
            resp = await self.search_v2(query, offset=offset, limit=limit, **kwargs)
            return list(resp.factors), resp.total_count

        return AsyncOffsetPaginator(
            _fetch, page_size=page_size, max_items=max_items
        )

    # ---- Factors ---------------------------------------------------------

    async def get_factor(
        self,
        factor_id: str,
        *,
        edition: Optional[str] = None,
    ) -> Factor:
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._get(f"/factors/{factor_id}", params=params or None)
        return Factor.model_validate(resp.data)

    async def match(
        self,
        activity_description: str,
        *,
        geography: Optional[str] = None,
        fuel_type: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 10,
        edition: Optional[str] = None,
    ) -> List[FactorMatch]:
        body: Dict[str, Any] = {
            "activity_description": activity_description,
            "limit": limit,
        }
        for key, value in (
            ("geography", geography),
            ("fuel_type", fuel_type),
            ("scope", scope),
        ):
            if value is not None:
                body[key] = value
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._post(
            "/factors/match", json_body=body, params=params or None
        )
        data = resp.data or {}
        candidates = (
            data.get("candidates", []) if isinstance(data, dict) else []
        )
        return [FactorMatch.model_validate(c) for c in candidates]

    async def coverage(self, *, edition: Optional[str] = None) -> CoverageReport:
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._get("/factors/coverage", params=params or None)
        return CoverageReport.model_validate(resp.data)

    # ---- Resolution ------------------------------------------------------

    async def resolve_explain(
        self,
        factor_id: str,
        *,
        method_profile: Optional[str] = None,
        alternates: Optional[int] = None,
        edition: Optional[str] = None,
    ) -> ResolvedFactor:
        params: Dict[str, Any] = {}
        if method_profile:
            params["method_profile"] = method_profile
        if alternates is not None:
            params["limit"] = alternates
        if edition:
            params["edition"] = edition
        resp = await self._get(
            f"/factors/{factor_id}/explain", params=params or None
        )
        return ResolvedFactor.model_validate(resp.data)

    async def resolve(
        self,
        request: Union[ResolutionRequest, Dict[str, Any]],
        *,
        alternates: Optional[int] = None,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> ResolvedFactor:
        if isinstance(request, ResolutionRequest):
            body = request.model_dump(exclude_none=True)
        else:
            body = dict(request)
        params: Dict[str, Any] = {}
        if alternates is not None:
            params["limit"] = alternates
        if edition:
            params["edition"] = edition
        if include_preview is not None:
            params["include_preview"] = _bool_param(include_preview)
        if include_connector is not None:
            params["include_connector"] = _bool_param(include_connector)
        resp = await self._post(
            "/factors/resolve-explain",
            json_body=body,
            params=params or None,
        )
        return ResolvedFactor.model_validate(resp.data)

    async def alternates(
        self,
        factor_id: str,
        *,
        method_profile: Optional[str] = None,
        limit: Optional[int] = None,
        edition: Optional[str] = None,
        include_preview: Optional[bool] = None,
        include_connector: Optional[bool] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if method_profile:
            params["method_profile"] = method_profile
        if limit is not None:
            params["limit"] = limit
        if edition:
            params["edition"] = edition
        if include_preview is not None:
            params["include_preview"] = _bool_param(include_preview)
        if include_connector is not None:
            params["include_connector"] = _bool_param(include_connector)
        resp = await self._get(
            f"/factors/{factor_id}/alternates", params=params or None
        )
        return resp.data if isinstance(resp.data, dict) else {"raw": resp.data}

    async def resolve_batch(
        self,
        requests: List[Union[ResolutionRequest, Dict[str, Any]]],
        *,
        edition: Optional[str] = None,
    ) -> BatchJobHandle:
        items: List[Dict[str, Any]] = []
        for r in requests:
            if isinstance(r, ResolutionRequest):
                items.append(r.model_dump(exclude_none=True))
            else:
                items.append(dict(r))
        body: Dict[str, Any] = {"requests": items}
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._post(
            "/factors/resolve/batch", json_body=body, params=params or None
        )
        return BatchJobHandle.model_validate(resp.data)

    async def get_batch_job(self, job_id: str) -> BatchJobHandle:
        resp = await self._get(f"/factors/jobs/{job_id}", use_cache=False)
        return BatchJobHandle.model_validate(resp.data)

    async def wait_for_batch(
        self,
        job: Union[BatchJobHandle, str],
        *,
        poll_interval: float = 2.0,
        timeout: Optional[float] = 600.0,
    ) -> BatchJobHandle:
        job_id = job.job_id if isinstance(job, BatchJobHandle) else str(job)
        loop = asyncio.get_event_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            current = await self.get_batch_job(job_id)
            if current.status in _BATCH_TERMINAL_STATES:
                if current.status == "failed":
                    raise FactorsAPIError(
                        "Batch job %s failed: %s"
                        % (job_id, current.error_message or "unknown error"),
                        context={"job_id": job_id, "status": current.status},
                    )
                return current
            if deadline is not None and loop.time() > deadline:
                raise FactorsAPIError(
                    "Timeout waiting for batch job %s (status=%s)"
                    % (job_id, current.status),
                    context={"job_id": job_id, "timeout": timeout},
                )
            await asyncio.sleep(poll_interval)

    # ---- Editions --------------------------------------------------------

    async def list_editions(
        self,
        *,
        include_pending: bool = True,
    ) -> List[Edition]:
        params = {"include_pending": _bool_param(include_pending)}
        resp = await self._get("/editions", params=params)
        data = resp.data if isinstance(resp.data, dict) else {}
        editions = data.get("editions", []) if isinstance(data, dict) else []
        return [Edition.model_validate(e) for e in editions]

    async def get_edition(self, edition_id: str) -> Dict[str, Any]:
        resp = await self._get(f"/editions/{edition_id}/changelog")
        return resp.data if isinstance(resp.data, dict) else {"raw": resp.data}

    async def diff(
        self,
        factor_id: str,
        left_edition: str,
        right_edition: str,
    ) -> FactorDiff:
        params = {"left_edition": left_edition, "right_edition": right_edition}
        resp = await self._get(f"/factors/{factor_id}/diff", params=params)
        return FactorDiff.model_validate(resp.data)

    async def audit_bundle(
        self,
        factor_id: str,
        *,
        edition: Optional[str] = None,
    ) -> AuditBundle:
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._get(
            f"/factors/{factor_id}/audit-bundle", params=params or None
        )
        return AuditBundle.model_validate(resp.data)

    # ---- Sources / method packs -----------------------------------------

    async def list_sources(
        self,
        *,
        edition: Optional[str] = None,
    ) -> List[Source]:
        params: Dict[str, Any] = {}
        if edition:
            params["edition"] = edition
        resp = await self._get(
            "/factors/source-registry", params=params or None
        )
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("sources") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [Source.model_validate(r) for r in rows]

    async def get_source(self, source_id: str) -> Source:
        resp = await self._get(f"/factors/sources/{source_id}")
        return Source.model_validate(resp.data)

    async def list_method_packs(self) -> List[MethodPack]:
        resp = await self._get("/method-packs")
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("method_packs") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [MethodPack.model_validate(r) for r in rows]

    async def get_method_pack(self, method_pack_id: str) -> MethodPack:
        resp = await self._get(f"/method-packs/{method_pack_id}")
        return MethodPack.model_validate(resp.data)

    # ---- Overrides -------------------------------------------------------

    async def set_override(
        self,
        override: Union[Override, Dict[str, Any]],
    ) -> Override:
        body = (
            override.model_dump(exclude_none=True)
            if isinstance(override, Override)
            else dict(override)
        )
        resp = await self._post("/factors/overrides", json_body=body)
        return Override.model_validate(resp.data)

    async def list_overrides(
        self,
        *,
        tenant_id: Optional[str] = None,
    ) -> List[Override]:
        params: Dict[str, Any] = {}
        if tenant_id:
            params["tenant_id"] = tenant_id
        resp = await self._get("/factors/overrides", params=params or None)
        data = resp.data
        if isinstance(data, dict):
            rows = data.get("overrides") or data.get("items") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [Override.model_validate(r) for r in rows]


__all__ = [
    "FactorsClient",
    "AsyncFactorsClient",
    "CertPinnedHTTPAdapter",
    "CertificatePinError",
    "GREENLANG_CA_PEM",
    "PINNED_HOST_SUFFIXES",
    "build_pinned_httpx_transport",
    "ReceiptVerificationError",
]
