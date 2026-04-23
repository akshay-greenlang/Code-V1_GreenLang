# -*- coding: utf-8 -*-
"""
OpenID Connect Relying Party (RP) middleware for the Factors API (SEC-5).

Supports:

* OIDC Discovery document (``/.well-known/openid-configuration``).
* Authorization Code flow with PKCE (S256).
* Tenant-scoped client credentials per :class:`SSOTenantConfig`.
* ID token validation: issuer, audience, exp, iat, nonce.
* JWKS key discovery with in-memory caching.
* Attribute / claim mapping -> GreenLang JWT.

Providers verified via fixtures: Okta, Azure AD, Auth0, OneLogin, generic OIDC.

Exposed routes (tenant-scoped):

  * ``GET  /v1/sso/oidc/{tenant_id}/login``      — start auth code flow.
  * ``GET  /v1/sso/oidc/{tenant_id}/callback``   — exchange + id-token verify.

Network calls (discovery, token, JWKS) go through an injectable
:class:`OIDCHttpClient`; tests swap it for a stub so no live IdP is needed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from fastapi import Request  # noqa: F401 (used by route handler annotations)
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment]

from greenlang.factors.middleware.sso_config import (
    SSOConfigRegistry,
    SSOTenantConfig,
    default_registry,
)
from greenlang.factors.middleware.sso_saml import _mint_jwt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OIDCError(Exception):
    """Raised when an OIDC flow step fails."""


class OIDCConfigError(OIDCError):
    """Tenant OIDC config missing or malformed."""


class OIDCVerificationError(OIDCError):
    """ID token verification failed."""


# ---------------------------------------------------------------------------
# HTTP client (injectable for tests)
# ---------------------------------------------------------------------------


class OIDCHttpClient:
    """Tiny HTTP client used for discovery / token / jwks calls.

    The class deliberately avoids a hard ``requests`` dependency so the
    Factors image stays small; tests inject a fake that captures calls.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def get_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers=headers or {"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def post_form(
        self,
        url: str,
        data: Dict[str, str],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        encoded = urllib.parse.urlencode(data).encode("utf-8")
        h = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(url, data=encoded, headers=h, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Discovery + JWKS cache
# ---------------------------------------------------------------------------


@dataclass
class OIDCDiscovery:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    userinfo_endpoint: Optional[str] = None
    end_session_endpoint: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class _DiscoveryCache:
    """Per-process cache of OIDC discovery + JWKS documents.

    TTL defaults to 15 minutes; force-refresh via :meth:`invalidate`.
    """

    def __init__(self, http: OIDCHttpClient, ttl_seconds: int = 900) -> None:
        self._http = http
        self._ttl = ttl_seconds
        self._disco: Dict[str, Tuple[float, OIDCDiscovery]] = {}
        self._jwks: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def discovery(self, metadata_url: str) -> OIDCDiscovery:
        now = time.time()
        cached = self._disco.get(metadata_url)
        if cached and now - cached[0] < self._ttl:
            return cached[1]
        raw = self._http.get_json(metadata_url)
        disco = OIDCDiscovery(
            issuer=raw["issuer"],
            authorization_endpoint=raw["authorization_endpoint"],
            token_endpoint=raw["token_endpoint"],
            jwks_uri=raw["jwks_uri"],
            userinfo_endpoint=raw.get("userinfo_endpoint"),
            end_session_endpoint=raw.get("end_session_endpoint"),
            raw=raw,
        )
        self._disco[metadata_url] = (now, disco)
        return disco

    def jwks(self, jwks_uri: str) -> Dict[str, Any]:
        now = time.time()
        cached = self._jwks.get(jwks_uri)
        if cached and now - cached[0] < self._ttl:
            return cached[1]
        raw = self._http.get_json(jwks_uri)
        self._jwks[jwks_uri] = (now, raw)
        return raw

    def invalidate(self, metadata_url: Optional[str] = None) -> None:
        if metadata_url is None:
            self._disco.clear()
            self._jwks.clear()
        else:
            self._disco.pop(metadata_url, None)


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def generate_pkce_pair() -> Tuple[str, str]:
    """Return (code_verifier, code_challenge) for S256 PKCE."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    return verifier, challenge


# ---------------------------------------------------------------------------
# State/nonce store (in-memory; swappable)
# ---------------------------------------------------------------------------


class _StateStore:
    """In-memory short-lived state store for CSRF/nonce binding.

    For multi-pod deploys, swap this for a Redis-backed store via
    :func:`install_oidc_routes(state_store=...)`. Keys expire after 10 min.
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def put(self, state: str, payload: Dict[str, Any]) -> None:
        self._store[state] = (time.time(), payload)

    def pop(self, state: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        rec = self._store.pop(state, None)
        if rec is None:
            return None
        ts, payload = rec
        if now - ts > self._ttl:
            return None
        return payload

    def gc(self) -> int:
        now = time.time()
        stale = [k for k, (ts, _) in self._store.items() if now - ts > self._ttl]
        for k in stale:
            self._store.pop(k, None)
        return len(stale)


# ---------------------------------------------------------------------------
# ID token verification
# ---------------------------------------------------------------------------


def verify_id_token(
    id_token: str,
    cfg: SSOTenantConfig,
    discovery: OIDCDiscovery,
    jwks: Dict[str, Any],
    expected_nonce: Optional[str] = None,
    leeway: int = 60,
) -> Dict[str, Any]:
    """Verify an OIDC ID token and return its claims.

    Checks: signature, issuer, audience, exp, iat, nbf, nonce.
    """
    try:
        import jwt  # type: ignore
        from jwt.algorithms import RSAAlgorithm  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise OIDCError("PyJWT[crypto] is required to verify ID tokens") from exc

    unverified = jwt.get_unverified_header(id_token)
    kid = unverified.get("kid")
    alg = unverified.get("alg") or "RS256"

    key = None
    for jwk in jwks.get("keys", []):
        if kid and jwk.get("kid") == kid:
            key = RSAAlgorithm.from_jwk(json.dumps(jwk))
            break
    if key is None and jwks.get("keys"):
        key = RSAAlgorithm.from_jwk(json.dumps(jwks["keys"][0]))
    if key is None:
        raise OIDCVerificationError("No matching JWKS key for kid=%r" % kid)

    try:
        claims = jwt.decode(
            id_token,
            key=key,
            algorithms=[alg],
            audience=cfg.client_id,
            issuer=discovery.issuer,
            leeway=leeway,
        )
    except Exception as exc:  # noqa: BLE001
        raise OIDCVerificationError(f"ID token validation failed: {exc}") from exc

    if expected_nonce and claims.get("nonce") != expected_nonce:
        raise OIDCVerificationError("nonce mismatch")

    return claims


# ---------------------------------------------------------------------------
# Claim mapping
# ---------------------------------------------------------------------------


def extract_user_claims_oidc(
    cfg: SSOTenantConfig, id_token_claims: Dict[str, Any]
) -> Dict[str, Any]:
    """Translate OIDC ID-token claims into GreenLang user claims."""

    def _pick(target: str) -> Any:
        source = cfg.attribute_mappings.get(target, target)
        return id_token_claims.get(source)

    email = _pick("email") or id_token_claims.get("email")
    if not email:
        raise OIDCError("ID token missing 'email' claim")

    if cfg.allowed_email_domains:
        domain = email.rsplit("@", 1)[-1].lower()
        allowed = [d.lower().lstrip("@") for d in cfg.allowed_email_domains]
        if domain not in allowed:
            raise OIDCError(
                f"Email domain {domain!r} not permitted for tenant "
                f"{cfg.tenant_id!r} (allowed={allowed})"
            )

    groups = _pick("groups") or []
    if isinstance(groups, str):
        groups = [groups]

    return {
        "sub": email,
        "email": email,
        "first_name": _pick("first_name") or "",
        "last_name": _pick("last_name") or "",
        "groups": list(groups),
        "tenant_id": cfg.tenant_id,
        "tier": cfg.default_tier,
        "role": cfg.default_role,
        "auth_method": "oidc",
        "idp": id_token_claims.get("iss", "oidc"),
    }


# ---------------------------------------------------------------------------
# FastAPI installer
# ---------------------------------------------------------------------------


def install_oidc_routes(
    app,
    *,
    prefix: str = "/v1/sso/oidc",
    registry: Optional[SSOConfigRegistry] = None,
    http_client: Optional[OIDCHttpClient] = None,
    state_store: Optional[_StateStore] = None,
    discovery_cache: Optional[_DiscoveryCache] = None,
    on_login: Optional[Callable[[Dict[str, Any]], None]] = None,
    external_base_url: Optional[str] = None,
) -> None:
    """Mount OIDC RP routes on a FastAPI app (authorization_code + PKCE)."""
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse, RedirectResponse

    reg = registry or default_registry()
    http = http_client or OIDCHttpClient()
    store = state_store or _StateStore()
    cache = discovery_cache or _DiscoveryCache(http)
    base = (
        external_base_url
        or os.getenv("GL_FACTORS_EXTERNAL_BASE_URL")
        or "https://factors.greenlang.com"
    ).rstrip("/")

    def _callback_url(tenant_id: str) -> str:
        return f"{base}{prefix}/{tenant_id}/callback"

    def _tenant_cfg(tenant_id: str) -> SSOTenantConfig:
        cfg = reg.get(tenant_id)
        if cfg is None or cfg.protocol != "oidc" or not cfg.enabled:
            raise HTTPException(status_code=404, detail="SSO not configured for tenant")
        if not cfg.idp_metadata_url or not cfg.client_id or not cfg.client_secret:
            raise HTTPException(
                status_code=500,
                detail="OIDC tenant config is incomplete (metadata_url/client_id/client_secret)",
            )
        return cfg

    # ------------------------------------------------------------------
    @app.get(prefix + "/{tenant_id}/login")
    def oidc_login(tenant_id: str, relay_state: Optional[str] = None):
        cfg = _tenant_cfg(tenant_id)
        disco = cache.discovery(cfg.idp_metadata_url)  # type: ignore[arg-type]
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(24)
        verifier, challenge = generate_pkce_pair()

        store.put(
            state,
            {
                "tenant_id": tenant_id,
                "nonce": nonce,
                "verifier": verifier,
                "relay_state": relay_state,
                "created_at": time.time(),
            },
        )

        params = {
            "response_type": "code",
            "client_id": cfg.client_id,
            "redirect_uri": _callback_url(tenant_id),
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        sep = "&" if "?" in disco.authorization_endpoint else "?"
        url = disco.authorization_endpoint + sep + urllib.parse.urlencode(params)
        return RedirectResponse(url, status_code=302)

    # ------------------------------------------------------------------
    @app.get(prefix + "/{tenant_id}/callback")
    def oidc_callback(tenant_id: str, code: str, state: str, request: Request):
        payload = store.pop(state)
        if not payload or payload["tenant_id"] != tenant_id:
            raise HTTPException(status_code=400, detail="Invalid or expired state")

        cfg = _tenant_cfg(tenant_id)
        disco = cache.discovery(cfg.idp_metadata_url)  # type: ignore[arg-type]

        token_resp = http.post_form(
            disco.token_endpoint,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _callback_url(tenant_id),
                "client_id": cfg.client_id or "",
                "client_secret": cfg.client_secret or "",
                "code_verifier": payload["verifier"],
            },
        )
        id_token = token_resp.get("id_token")
        if not id_token:
            raise HTTPException(status_code=401, detail="IdP did not return id_token")

        jwks = cache.jwks(disco.jwks_uri)
        try:
            id_claims = verify_id_token(
                id_token, cfg, disco, jwks, expected_nonce=payload["nonce"]
            )
            user_claims = extract_user_claims_oidc(cfg, id_claims)
        except OIDCError as exc:
            logger.warning("OIDC callback failed tenant=%s: %s", tenant_id, exc)
            raise HTTPException(status_code=401, detail=f"OIDC error: {exc}")

        if on_login is not None:
            try:
                on_login(user_claims)
            except Exception as exc:  # noqa: BLE001
                logger.error("on_login callback failed: %s", exc)

        token = _mint_jwt(user_claims)
        resp = JSONResponse(
            {
                "token": token,
                "relay_state": payload.get("relay_state"),
                "user": {
                    "email": user_claims["email"],
                    "tenant_id": user_claims["tenant_id"],
                    "role": user_claims["role"],
                    "tier": user_claims["tier"],
                },
            }
        )
        resp.set_cookie(
            "gl_factors_token",
            token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=3600,
        )
        return resp

    logger.info("OIDC RP routes installed at %s", prefix)


__all__ = [
    "OIDCError",
    "OIDCConfigError",
    "OIDCVerificationError",
    "OIDCHttpClient",
    "OIDCDiscovery",
    "generate_pkce_pair",
    "verify_id_token",
    "extract_user_claims_oidc",
    "install_oidc_routes",
]
