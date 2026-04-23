# -*- coding: utf-8 -*-
"""Tests for the OIDC RP middleware (SEC-5)."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import pytest

from greenlang.factors.middleware.sso_config import (
    SSOConfigRegistry,
    SSOTenantConfig,
)
from greenlang.factors.middleware.sso_oidc import (
    OIDCVerificationError,
    extract_user_claims_oidc,
    generate_pkce_pair,
    install_oidc_routes,
    verify_id_token,
)


# ---------------------------------------------------------------------------
# Fake IdP with RS256 signing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair():
    jwt = pytest.importorskip("jwt")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key()
    priv_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    pub_numbers = public.public_numbers()

    def _b64url_uint(n: int) -> str:
        h = hex(n)[2:]
        if len(h) % 2:
            h = "0" + h
        return base64.urlsafe_b64encode(bytes.fromhex(h)).rstrip(b"=").decode("ascii")

    jwk = {
        "kty": "RSA",
        "kid": "test-key-1",
        "alg": "RS256",
        "use": "sig",
        "n": _b64url_uint(pub_numbers.n),
        "e": _b64url_uint(pub_numbers.e),
    }
    return priv_pem, jwk


def _make_id_token(
    priv_pem: bytes,
    *,
    iss: str,
    aud: str,
    email: str = "bob@globex.com",
    nonce: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    kid: str = "test-key-1",
) -> str:
    jwt = pytest.importorskip("jwt")
    now = int(time.time())
    payload = {
        "iss": iss,
        "aud": aud,
        "sub": email,
        "email": email,
        "given_name": "Bob",
        "family_name": "Barker",
        "iat": now,
        "exp": now + 300,
    }
    if nonce:
        payload["nonce"] = nonce
    if extra:
        payload.update(extra)
    return jwt.encode(payload, priv_pem, algorithm="RS256", headers={"kid": kid})


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------


def test_pkce_pair_is_rfc7636_compliant():
    verifier, challenge = generate_pkce_pair()
    # S256: challenge == base64url(sha256(verifier))
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert challenge == expected
    assert 43 <= len(verifier) <= 128


# ---------------------------------------------------------------------------
# ID token verification
# ---------------------------------------------------------------------------


def test_verify_id_token_accepts_valid(rsa_keypair):
    priv, jwk = rsa_keypair
    cfg = SSOTenantConfig(
        tenant_id="globex", protocol="oidc", client_id="client-abc"
    )
    from greenlang.factors.middleware.sso_oidc import OIDCDiscovery

    disco = OIDCDiscovery(
        issuer="https://idp.test",
        authorization_endpoint="https://idp.test/authorize",
        token_endpoint="https://idp.test/token",
        jwks_uri="https://idp.test/jwks",
    )
    token = _make_id_token(priv, iss="https://idp.test", aud="client-abc", nonce="abc")
    claims = verify_id_token(token, cfg, disco, {"keys": [jwk]}, expected_nonce="abc")
    assert claims["email"] == "bob@globex.com"


def test_verify_id_token_rejects_bad_nonce(rsa_keypair):
    priv, jwk = rsa_keypair
    cfg = SSOTenantConfig(
        tenant_id="globex", protocol="oidc", client_id="client-abc"
    )
    from greenlang.factors.middleware.sso_oidc import OIDCDiscovery

    disco = OIDCDiscovery(
        issuer="https://idp.test",
        authorization_endpoint="https://idp.test/authorize",
        token_endpoint="https://idp.test/token",
        jwks_uri="https://idp.test/jwks",
    )
    token = _make_id_token(priv, iss="https://idp.test", aud="client-abc", nonce="abc")
    with pytest.raises(OIDCVerificationError):
        verify_id_token(token, cfg, disco, {"keys": [jwk]}, expected_nonce="wrong")


@pytest.mark.parametrize(
    "provider,groups_claim",
    [
        ("okta", "groups"),
        ("azure_ad", "groups"),
        ("auth0", "https://globex.example/groups"),
        ("onelogin", "memberOf"),
        ("generic_oidc", "groups"),
    ],
)
def test_claim_mapping_per_provider(provider, groups_claim):
    cfg = SSOTenantConfig(
        tenant_id="globex",
        protocol="oidc",
        attribute_mappings={
            "email": "email",
            "first_name": "given_name",
            "last_name": "family_name",
            "groups": groups_claim,
        },
    )
    claims = {
        "iss": f"https://{provider}.example.com",
        "email": "bob@globex.com",
        "given_name": "Bob",
        "family_name": "Barker",
        groups_claim: ["admins"],
    }
    mapped = extract_user_claims_oidc(cfg, claims)
    assert mapped["email"] == "bob@globex.com"
    assert mapped["groups"] == ["admins"]
    assert mapped["auth_method"] == "oidc"


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------


class _FakeHttp:
    def __init__(self, disco: Dict[str, Any], token_resp: Dict[str, Any], jwks: Dict[str, Any]):
        self._disco = disco
        self._token_resp = token_resp
        self._jwks = jwks
        self.calls = []

    def get_json(self, url, headers=None):
        self.calls.append(("GET", url))
        if "openid-configuration" in url:
            return self._disco
        if "jwks" in url:
            return self._jwks
        raise RuntimeError(f"unexpected GET {url}")

    def post_form(self, url, data, headers=None):
        self.calls.append(("POST", url, data))
        return self._token_resp


@pytest.fixture()
def oidc_app(monkeypatch, rsa_keypair):
    fastapi = pytest.importorskip("fastapi")
    priv, jwk = rsa_keypair
    monkeypatch.setenv("GL_FACTORS_SSO_JWT_SECRET", "unit-test-secret")

    disco = {
        "issuer": "https://idp.test",
        "authorization_endpoint": "https://idp.test/authorize",
        "token_endpoint": "https://idp.test/token",
        "jwks_uri": "https://idp.test/jwks",
    }
    id_token = _make_id_token(
        priv, iss="https://idp.test", aud="client-abc", nonce="__placeholder__"
    )
    http = _FakeHttp(
        disco=disco,
        token_resp={"id_token": id_token, "access_token": "at", "token_type": "Bearer"},
        jwks={"keys": [jwk]},
    )

    reg = SSOConfigRegistry()
    reg.put(
        SSOTenantConfig(
            tenant_id="globex",
            protocol="oidc",
            idp_metadata_url="https://idp.test/.well-known/openid-configuration",
            client_id="client-abc",
            client_secret="shh",
        )
    )

    app = fastapi.FastAPI()
    install_oidc_routes(
        app,
        registry=reg,
        http_client=http,
        external_base_url="https://factors.greenlang.com",
    )
    return app, http, priv, jwk


def test_oidc_login_redirects_with_pkce(oidc_app):
    tc = pytest.importorskip("fastapi.testclient")
    app, http, _, _ = oidc_app
    client = tc.TestClient(app, follow_redirects=False)
    resp = client.get("/v1/sso/oidc/globex/login")
    assert resp.status_code in (302, 307)
    qs = parse_qs(urlparse(resp.headers["location"]).query)
    assert qs["code_challenge_method"] == ["S256"]
    assert qs["client_id"] == ["client-abc"]


def test_oidc_callback_state_must_match(oidc_app):
    tc = pytest.importorskip("fastapi.testclient")
    app, *_ = oidc_app
    client = tc.TestClient(app)
    resp = client.get("/v1/sso/oidc/globex/callback", params={"code": "x", "state": "bogus"})
    assert resp.status_code == 400
