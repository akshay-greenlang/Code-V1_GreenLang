# -*- coding: utf-8 -*-
"""
SAML 2.0 Service Provider (SP) middleware for the Factors API (SEC-5).

Responsibilities:

* Accept SP-initiated ``/v1/sso/saml/{tenant_id}/login`` (builds AuthnRequest
  and redirects to the IdP SSO URL).
* Accept IdP-initiated and SP-initiated HTTP-POST responses at
  ``/v1/sso/saml/{tenant_id}/acs`` (Assertion Consumer Service).
* Validate the SAML assertion (signature + audience + NotOnOrAfter).
* Apply the tenant's attribute mapping and optional JIT provisioning.
* Mint a GreenLang JWT via the existing SEC-001 auth layer and return it
  to the browser as an HTTP-only cookie + JSON body so API clients and the
  Operator Console can use either.

This module does **not** re-implement JWT issuance. It integrates with the
existing auth layer (``greenlang.integration.api.dependencies``) that owns
the signing secret and the claim schema. When that module is not
installed, the handlers fall back to an HS256 token minted with
``GL_FACTORS_SSO_JWT_SECRET`` so tests can exercise the full flow.

The SAML protocol layer uses ``python3-saml`` if installed, otherwise the
minimal in-tree verifier :class:`_MinimalSAMLVerifier` which supports
signed-assertion validation and attribute extraction — enough for Okta,
Azure AD, Auth0, and OneLogin. Production deployments are expected to ship
``python3-saml`` (gated behind the ``security`` optional dependency group).

Providers verified via fixtures in ``tests/factors/fixtures/sso/``:

    okta, azure_ad, auth0, onelogin, generic_saml
"""
from __future__ import annotations

import base64
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

try:
    from fastapi import Request, Response  # noqa: F401 (used by route handler annotations)
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]

from greenlang.factors.middleware.sso_config import (
    SSOConfigRegistry,
    SSOTenantConfig,
    default_registry,
)

logger = logging.getLogger(__name__)

# SAML namespaces (kept local — no dependency on python3-saml for parsing).
_NS = {
    "saml": "urn:oasis:names:tc:SAML:2.0:assertion",
    "samlp": "urn:oasis:names:tc:SAML:2.0:protocol",
    "ds": "http://www.w3.org/2000/09/xmldsig#",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SAMLError(Exception):
    """Raised when a SAML assertion cannot be validated."""


class SAMLConfigError(SAMLError):
    """Raised when tenant SAML config is missing / invalid."""


class SAMLSignatureError(SAMLError):
    """Assertion signature failed to validate."""


class SAMLAudienceError(SAMLError):
    """Assertion audience does not match our SP entity id."""


class SAMLExpiredError(SAMLError):
    """Assertion is expired or not yet valid."""


# ---------------------------------------------------------------------------
# Minimal in-tree verifier (used when python3-saml is not installed)
# ---------------------------------------------------------------------------


class _MinimalSAMLVerifier:
    """Bare-bones SAML 2.0 Response parser + validator.

    Feature coverage:

      * HTTP-POST binding, base64-wrapped ``SAMLResponse``.
      * Signed ``<Response>`` or signed ``<Assertion>`` (either is fine).
      * Audience restriction check against ``sp_entity_id``.
      * NotBefore / NotOnOrAfter freshness check with ``clock_skew_seconds``.
      * Attribute extraction from ``<AttributeStatement>``.

    NOT covered (use ``python3-saml`` for these):
      * HTTP-Redirect binding signatures.
      * Encrypted assertions.
      * Name-ID format translation.

    All missing features fail closed (raise :class:`SAMLError`) rather than
    silently accepting the assertion.
    """

    def __init__(self, clock_skew_seconds: int = 30) -> None:
        self.clock_skew = clock_skew_seconds

    # ------------------------------------------------------------------
    def parse(self, raw_response_b64: str) -> ET.Element:
        try:
            xml_bytes = base64.b64decode(raw_response_b64, validate=False)
        except Exception as exc:
            raise SAMLError(f"SAMLResponse is not valid base64: {exc}") from exc
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise SAMLError(f"SAMLResponse is not valid XML: {exc}") from exc
        return root

    # ------------------------------------------------------------------
    def verify_signature(
        self,
        response: ET.Element,
        idp_cert_pem: Optional[str],
    ) -> None:
        """Verify the XML signature embedded in the Response or Assertion.

        If ``idp_cert_pem`` is None, signature verification is skipped with
        a loud warning. This mode is **only** acceptable in tests — callers
        get a ``SAMLSignatureError`` in production config via
        :func:`validate_assertion`.
        """
        if not idp_cert_pem:
            logger.warning("SAML signature check skipped (no idp_cert_pem)")
            return

        # Prefer python3-saml if available; fall back to xmlsec if present.
        try:
            from signxml import XMLVerifier  # type: ignore

            XMLVerifier().verify(response, x509_cert=idp_cert_pem)
            return
        except ImportError:
            pass

        try:
            import xmlsec  # type: ignore

            # xmlsec requires lxml elements; re-parse via lxml if available.
            from lxml import etree as _et  # type: ignore

            xml_bytes = ET.tostring(response)
            doc = _et.fromstring(xml_bytes)
            sig_node = xmlsec.tree.find_node(doc, xmlsec.constants.NodeSignature)
            if sig_node is None:
                raise SAMLSignatureError("No <Signature> element found")
            ctx = xmlsec.SignatureContext()
            key = xmlsec.Key.from_memory(
                idp_cert_pem, xmlsec.constants.KeyDataFormatCertPem
            )
            ctx.key = key
            ctx.verify(sig_node)
            return
        except ImportError:
            pass

        raise SAMLSignatureError(
            "No XML signature library available; install 'signxml' or 'xmlsec' "
            "to validate SAML signatures."
        )

    # ------------------------------------------------------------------
    def validate_assertion(
        self,
        response: ET.Element,
        sp_entity_id: str,
        expected_destination: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the parsed assertion fields or raise :class:`SAMLError`."""
        assertion = response.find(".//saml:Assertion", _NS)
        if assertion is None:
            raise SAMLError("No <Assertion> in SAMLResponse")

        # --- Audience ----------------------------------------------------
        audiences = [
            a.text for a in assertion.findall(".//saml:AudienceRestriction/saml:Audience", _NS)
            if a.text
        ]
        if audiences and sp_entity_id not in audiences:
            raise SAMLAudienceError(
                f"Assertion audience {audiences!r} does not include our "
                f"sp_entity_id={sp_entity_id!r}"
            )

        # --- Time window -------------------------------------------------
        now = datetime.now(timezone.utc)
        for cond in assertion.findall("./saml:Conditions", _NS):
            nb = cond.get("NotBefore")
            naoa = cond.get("NotOnOrAfter")
            if nb:
                nb_dt = _parse_iso(nb)
                if nb_dt and now + timedelta(seconds=self.clock_skew) < nb_dt:
                    raise SAMLExpiredError(f"Assertion not yet valid (NotBefore={nb})")
            if naoa:
                naoa_dt = _parse_iso(naoa)
                if naoa_dt and now - timedelta(seconds=self.clock_skew) >= naoa_dt:
                    raise SAMLExpiredError(
                        f"Assertion expired (NotOnOrAfter={naoa})"
                    )

        # --- Destination (for SP-initiated flows) -----------------------
        dest = response.get("Destination")
        if expected_destination and dest and dest != expected_destination:
            raise SAMLError(
                f"Response Destination={dest!r} does not match "
                f"expected={expected_destination!r}"
            )

        # --- NameID + attributes -----------------------------------------
        name_id_el = assertion.find(".//saml:Subject/saml:NameID", _NS)
        name_id = name_id_el.text if name_id_el is not None else None

        attributes: Dict[str, List[str]] = {}
        for attr in assertion.findall(".//saml:Attribute", _NS):
            name = attr.get("Name") or attr.get("FriendlyName")
            if not name:
                continue
            vals = [
                (v.text or "")
                for v in attr.findall("./saml:AttributeValue", _NS)
            ]
            attributes[name] = [v for v in vals if v]

        return {
            "name_id": name_id,
            "attributes": attributes,
            "assertion_id": assertion.get("ID"),
            "issuer": (
                assertion.find("./saml:Issuer", _NS).text
                if assertion.find("./saml:Issuer", _NS) is not None
                else None
            ),
        }


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        # Python's fromisoformat doesn't grok trailing Z until 3.11;
        # normalize to +00:00 so 3.10 CI still passes.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SP / Protocol helpers
# ---------------------------------------------------------------------------


def build_authn_request(
    cfg: SSOTenantConfig,
    acs_url: str,
    relay_state: Optional[str] = None,
) -> Tuple[str, str]:
    """Build an SP-initiated AuthnRequest, base64-deflate encoded.

    Returns:
        Tuple of ``(redirect_url, request_id)``. Callers should stash
        ``request_id`` in the session keyed by ``RelayState`` so the ACS
        handler can tie the response back to the original request.
    """
    if not cfg.idp_sso_url:
        raise SAMLConfigError(
            f"tenant={cfg.tenant_id}: missing idp_sso_url in SSO config"
        )

    request_id = "_" + uuid.uuid4().hex
    issue_instant = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    authn_xml = (
        '<samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" '
        'xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" '
        f'ID="{request_id}" Version="2.0" '
        f'IssueInstant="{issue_instant}" '
        f'Destination="{cfg.idp_sso_url}" '
        'ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" '
        f'AssertionConsumerServiceURL="{acs_url}">'
        f"<saml:Issuer>{cfg.sp_entity_id}</saml:Issuer>"
        '<samlp:NameIDPolicy Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress" '
        'AllowCreate="true"/>'
        "</samlp:AuthnRequest>"
    )

    import zlib

    deflated = zlib.compress(authn_xml.encode("utf-8"))[2:-4]
    b64 = base64.b64encode(deflated).decode("ascii")

    params = {"SAMLRequest": b64}
    if relay_state:
        params["RelayState"] = relay_state

    sep = "&" if "?" in cfg.idp_sso_url else "?"
    return cfg.idp_sso_url + sep + urlencode(params), request_id


# ---------------------------------------------------------------------------
# Attribute mapping + user assembly
# ---------------------------------------------------------------------------


def extract_user_claims(
    cfg: SSOTenantConfig,
    parsed_assertion: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the tenant's attribute mapping to produce a GreenLang user
    claim dict ready for JWT minting.

    Validates:
      * Presence of email / NameID.
      * Email domain allowlist (if configured).
    """
    attrs = parsed_assertion.get("attributes", {}) or {}
    name_id = parsed_assertion.get("name_id")

    def _pick(target: str) -> Optional[str]:
        source = cfg.attribute_mappings.get(target, target)
        # Special-case NameID because it's not inside AttributeStatement.
        if source == "NameID":
            return name_id
        vals = attrs.get(source)
        if not vals:
            return None
        return vals[0]

    email = _pick("email") or name_id
    if not email:
        raise SAMLError("Assertion is missing an email / NameID")

    if cfg.allowed_email_domains:
        domain = email.rsplit("@", 1)[-1].lower()
        allowed = [d.lower().lstrip("@") for d in cfg.allowed_email_domains]
        if domain not in allowed:
            raise SAMLError(
                f"Email domain {domain!r} not permitted for tenant "
                f"{cfg.tenant_id!r} (allowed={allowed})"
            )

    groups_val = attrs.get(cfg.attribute_mappings.get("groups", "groups"), [])
    return {
        "sub": email,
        "email": email,
        "first_name": _pick("first_name") or "",
        "last_name": _pick("last_name") or "",
        "groups": groups_val,
        "tenant_id": cfg.tenant_id,
        "tier": cfg.default_tier,
        "role": cfg.default_role,
        "auth_method": "saml",
        "idp": cfg.idp_entity_id or "saml",
    }


# ---------------------------------------------------------------------------
# JWT minting (integrates with existing SEC-001)
# ---------------------------------------------------------------------------


def _mint_jwt(claims: Dict[str, Any], ttl_seconds: int = 3600) -> str:
    """Mint a JWT with the existing SEC-001 signing secret (preferred),
    falling back to ``GL_FACTORS_SSO_JWT_SECRET`` for isolated tests."""
    try:
        import jwt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SAMLError("PyJWT is required to mint SSO JWTs") from exc

    try:
        from greenlang.integration.api.dependencies import (
            JWT_ALGORITHM,
            JWT_SECRET,
        )
        secret = JWT_SECRET
        algo = JWT_ALGORITHM
    except Exception:  # noqa: BLE001
        secret = os.getenv("GL_FACTORS_SSO_JWT_SECRET") or os.getenv("JWT_SECRET")
        if not secret:
            raise SAMLError(
                "No JWT secret available. Set GL_FACTORS_SSO_JWT_SECRET or install "
                "greenlang.integration.api.dependencies."
            )
        algo = "HS256"

    now = int(time.time())
    payload = dict(claims)
    payload.setdefault("iat", now)
    payload.setdefault("nbf", now)
    payload["exp"] = now + ttl_seconds
    payload["jti"] = secrets.token_hex(16)
    return jwt.encode(payload, secret, algorithm=algo)


# ---------------------------------------------------------------------------
# FastAPI installer
# ---------------------------------------------------------------------------


def install_saml_routes(
    app,
    *,
    prefix: str = "/v1/sso/saml",
    registry: Optional[SSOConfigRegistry] = None,
    verifier: Optional[_MinimalSAMLVerifier] = None,
    on_login: Optional[Callable[[Dict[str, Any]], None]] = None,
    external_base_url: Optional[str] = None,
    strict_signatures: Optional[bool] = None,
) -> None:
    """Mount SAML 2.0 SP routes on a FastAPI app.

    Exposed routes (all tenant-scoped):

      * ``GET  {prefix}/{tenant_id}/metadata`` — SP metadata XML.
      * ``GET  {prefix}/{tenant_id}/login``    — SP-initiated redirect.
      * ``POST {prefix}/{tenant_id}/acs``      — Assertion Consumer Service
        (accepts both SP-initiated and IdP-initiated responses).

    Args:
        app: FastAPI application.
        prefix: URL prefix.
        registry: SSO config registry (defaults to process-wide).
        verifier: SAML verifier (defaults to :class:`_MinimalSAMLVerifier`).
        on_login: Optional callback invoked on successful login with the
            full claim dict — use this to trigger JIT provisioning.
        external_base_url: Base URL under which the SP is reachable, for
            metadata / ACS URL assembly. Defaults to
            ``GL_FACTORS_EXTERNAL_BASE_URL`` env or ``https://factors.greenlang.com``.
        strict_signatures: if True, missing IdP cert is a hard error. Defaults
            to True when ``APP_ENV`` is staging/production, False otherwise.
    """
    from fastapi import HTTPException, Request, Response
    from fastapi.responses import JSONResponse, RedirectResponse

    reg = registry or default_registry()
    ver = verifier or _MinimalSAMLVerifier()
    base = (
        external_base_url
        or os.getenv("GL_FACTORS_EXTERNAL_BASE_URL")
        or "https://factors.greenlang.com"
    ).rstrip("/")

    if strict_signatures is None:
        env = (os.getenv("APP_ENV") or os.getenv("GL_ENV") or "").lower()
        strict_signatures = env in {"staging", "production", "prod"}

    def _acs_url(tenant_id: str) -> str:
        return f"{base}{prefix}/{tenant_id}/acs"

    def _tenant_cfg(tenant_id: str) -> SSOTenantConfig:
        cfg = reg.get(tenant_id)
        if cfg is None or cfg.protocol != "saml" or not cfg.enabled:
            raise HTTPException(status_code=404, detail="SSO not configured for tenant")
        return cfg

    # ------------------------------------------------------------------
    @app.get(prefix + "/{tenant_id}/metadata")
    def saml_metadata(tenant_id: str) -> Response:
        cfg = _tenant_cfg(tenant_id)
        acs = _acs_url(tenant_id)
        xml = (
            '<?xml version="1.0"?>'
            '<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata" '
            f'entityID="{cfg.sp_entity_id}">'
            '<md:SPSSODescriptor AuthnRequestsSigned="false" WantAssertionsSigned="true" '
            'protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">'
            '<md:AssertionConsumerService '
            'Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" '
            f'Location="{acs}" index="0" isDefault="true"/>'
            "</md:SPSSODescriptor>"
            "</md:EntityDescriptor>"
        )
        return Response(content=xml, media_type="application/samlmetadata+xml")

    # ------------------------------------------------------------------
    @app.get(prefix + "/{tenant_id}/login")
    def saml_login(tenant_id: str, relay_state: Optional[str] = None):
        cfg = _tenant_cfg(tenant_id)
        redirect_url, request_id = build_authn_request(
            cfg, acs_url=_acs_url(tenant_id), relay_state=relay_state
        )
        logger.info(
            "SAML SP-initiated login tenant=%s request_id=%s", tenant_id, request_id
        )
        return RedirectResponse(redirect_url, status_code=302)

    # ------------------------------------------------------------------
    @app.post(prefix + "/{tenant_id}/acs")
    async def saml_acs(tenant_id: str, request: Request):
        form = await request.form()
        saml_response = form.get("SAMLResponse")
        relay_state = form.get("RelayState")
        if not saml_response:
            raise HTTPException(status_code=400, detail="Missing SAMLResponse")

        cfg = _tenant_cfg(tenant_id)

        # Strict mode: missing cert is fatal.
        if strict_signatures and not cfg.idp_public_cert_pem:
            raise HTTPException(
                status_code=500,
                detail="SAML IdP cert not configured for tenant (strict mode)",
            )

        try:
            root = ver.parse(saml_response)
            ver.verify_signature(root, cfg.idp_public_cert_pem)
            parsed = ver.validate_assertion(
                root,
                sp_entity_id=cfg.sp_entity_id or "",
                expected_destination=_acs_url(tenant_id),
            )
            claims = extract_user_claims(cfg, parsed)
        except SAMLError as exc:
            logger.warning("SAML ACS failed for tenant=%s: %s", tenant_id, exc)
            raise HTTPException(status_code=401, detail=f"SAML error: {exc}")

        if on_login is not None:
            try:
                on_login(claims)
            except Exception as exc:  # noqa: BLE001
                logger.error("on_login callback failed: %s", exc)

        token = _mint_jwt(claims)
        resp = JSONResponse(
            {
                "token": token,
                "relay_state": relay_state,
                "user": {
                    "email": claims["email"],
                    "tenant_id": claims["tenant_id"],
                    "role": claims["role"],
                    "tier": claims["tier"],
                },
            }
        )
        # Also expose as an HTTP-only cookie for browser consumers.
        resp.set_cookie(
            "gl_factors_token",
            token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=3600,
        )
        return resp

    logger.info("SAML SP routes installed at %s", prefix)


__all__ = [
    "SAMLError",
    "SAMLConfigError",
    "SAMLSignatureError",
    "SAMLAudienceError",
    "SAMLExpiredError",
    "_MinimalSAMLVerifier",
    "build_authn_request",
    "extract_user_claims",
    "install_saml_routes",
]
