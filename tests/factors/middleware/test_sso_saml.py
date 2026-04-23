# -*- coding: utf-8 -*-
"""Tests for the SAML 2.0 SP middleware (SEC-5)."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import pytest

from greenlang.factors.middleware.sso_config import (
    SSOConfigRegistry,
    SSOTenantConfig,
)
from greenlang.factors.middleware.sso_saml import (
    SAMLAudienceError,
    SAMLExpiredError,
    _MinimalSAMLVerifier,
    build_authn_request,
    extract_user_claims,
    install_saml_routes,
)


# ---------------------------------------------------------------------------
# SAML response fixture builder (shared with provider fixtures)
# ---------------------------------------------------------------------------


def _saml_response(
    *,
    audience: str = "urn:greenlang:factors:sp:acme",
    email: str = "alice@acme.com",
    first: str = "Alice",
    last: str = "Anderson",
    groups: str = "analysts",
    not_before_offset: int = -60,
    not_on_or_after_offset: int = 300,
    destination: str = "https://factors.greenlang.com/v1/sso/saml/acme/acs",
) -> str:
    now = datetime.now(timezone.utc)
    nb = (now + timedelta(seconds=not_before_offset)).strftime("%Y-%m-%dT%H:%M:%SZ")
    naoa = (now + timedelta(seconds=not_on_or_after_offset)).strftime("%Y-%m-%dT%H:%M:%SZ")
    xml = f"""<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
  xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" ID="_r" Version="2.0"
  IssueInstant="{now.strftime('%Y-%m-%dT%H:%M:%SZ')}" Destination="{destination}">
  <saml:Assertion ID="_a" Version="2.0" IssueInstant="{now.strftime('%Y-%m-%dT%H:%M:%SZ')}">
    <saml:Issuer>https://idp.example.com</saml:Issuer>
    <saml:Subject>
      <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">{email}</saml:NameID>
    </saml:Subject>
    <saml:Conditions NotBefore="{nb}" NotOnOrAfter="{naoa}">
      <saml:AudienceRestriction>
        <saml:Audience>{audience}</saml:Audience>
      </saml:AudienceRestriction>
    </saml:Conditions>
    <saml:AttributeStatement>
      <saml:Attribute Name="urn:oid:2.5.4.42"><saml:AttributeValue>{first}</saml:AttributeValue></saml:Attribute>
      <saml:Attribute Name="urn:oid:2.5.4.4"><saml:AttributeValue>{last}</saml:AttributeValue></saml:Attribute>
      <saml:Attribute Name="http://schemas.xmlsoap.org/claims/Group"><saml:AttributeValue>{groups}</saml:AttributeValue></saml:Attribute>
    </saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>"""
    return base64.b64encode(xml.encode("utf-8")).decode("ascii")


# ---------------------------------------------------------------------------
# _MinimalSAMLVerifier
# ---------------------------------------------------------------------------


def test_minimal_verifier_accepts_valid_assertion():
    ver = _MinimalSAMLVerifier()
    root = ver.parse(_saml_response())
    parsed = ver.validate_assertion(root, sp_entity_id="urn:greenlang:factors:sp:acme")
    assert parsed["name_id"] == "alice@acme.com"
    attrs = parsed["attributes"]
    assert attrs["urn:oid:2.5.4.42"] == ["Alice"]
    assert attrs["urn:oid:2.5.4.4"] == ["Anderson"]


def test_minimal_verifier_rejects_wrong_audience():
    ver = _MinimalSAMLVerifier()
    root = ver.parse(_saml_response(audience="urn:someone:else"))
    with pytest.raises(SAMLAudienceError):
        ver.validate_assertion(root, sp_entity_id="urn:greenlang:factors:sp:acme")


def test_minimal_verifier_rejects_expired_assertion():
    ver = _MinimalSAMLVerifier(clock_skew_seconds=5)
    root = ver.parse(_saml_response(not_on_or_after_offset=-120))
    with pytest.raises(SAMLExpiredError):
        ver.validate_assertion(root, sp_entity_id="urn:greenlang:factors:sp:acme")


def test_minimal_verifier_rejects_not_yet_valid():
    ver = _MinimalSAMLVerifier(clock_skew_seconds=5)
    root = ver.parse(_saml_response(not_before_offset=600))
    with pytest.raises(SAMLExpiredError):
        ver.validate_assertion(root, sp_entity_id="urn:greenlang:factors:sp:acme")


# ---------------------------------------------------------------------------
# Attribute mapping (per-IdP fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "idp_name,group_attr",
    [
        ("okta", "http://schemas.xmlsoap.org/claims/Group"),
        ("azure_ad", "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups"),
        ("auth0", "https://auth0.com/groups"),
        ("onelogin", "memberOf"),
        ("generic_saml", "http://schemas.xmlsoap.org/claims/Group"),
    ],
)
def test_attribute_mapping_per_provider(idp_name, group_attr):
    cfg = SSOTenantConfig(
        tenant_id="acme",
        protocol="saml",
        attribute_mappings={
            "email": "NameID",
            "first_name": "urn:oid:2.5.4.42",
            "last_name": "urn:oid:2.5.4.4",
            "groups": group_attr,
        },
    )
    parsed = {
        "name_id": "alice@acme.com",
        "attributes": {
            "urn:oid:2.5.4.42": ["Alice"],
            "urn:oid:2.5.4.4": ["Anderson"],
            group_attr: ["admins", "analysts"],
        },
    }
    claims = extract_user_claims(cfg, parsed)
    assert claims["email"] == "alice@acme.com"
    assert claims["first_name"] == "Alice"
    assert claims["last_name"] == "Anderson"
    assert claims["groups"] == ["admins", "analysts"]
    assert claims["tenant_id"] == "acme"
    assert claims["auth_method"] == "saml"


def test_email_domain_allowlist_enforced():
    cfg = SSOTenantConfig(
        tenant_id="acme",
        protocol="saml",
        allowed_email_domains=["acme.com"],
    )
    parsed = {"name_id": "mallory@evilcorp.com", "attributes": {}}
    with pytest.raises(Exception) as exc:
        extract_user_claims(cfg, parsed)
    assert "not permitted" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# AuthnRequest builder
# ---------------------------------------------------------------------------


def test_build_authn_request_returns_redirect():
    cfg = SSOTenantConfig(
        tenant_id="acme",
        protocol="saml",
        idp_sso_url="https://idp.example.com/sso",
        sp_entity_id="urn:greenlang:factors:sp:acme",
    )
    url, req_id = build_authn_request(
        cfg, acs_url="https://factors.greenlang.com/v1/sso/saml/acme/acs"
    )
    assert url.startswith("https://idp.example.com/sso")
    assert "SAMLRequest=" in url
    assert req_id.startswith("_")


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_with_saml(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    monkeypatch.setenv("GL_FACTORS_SSO_JWT_SECRET", "unit-test-secret")
    app = fastapi.FastAPI()
    reg = SSOConfigRegistry()
    reg.put(
        SSOTenantConfig(
            tenant_id="acme",
            protocol="saml",
            idp_sso_url="https://idp.example.com/sso",
            sp_entity_id="urn:greenlang:factors:sp:acme",
            idp_public_cert_pem=None,  # signature skipped in non-strict mode
        )
    )
    install_saml_routes(
        app,
        registry=reg,
        external_base_url="https://factors.greenlang.com",
        strict_signatures=False,
    )
    return app


def test_metadata_endpoint_returns_samlmetadata(app_with_saml):
    starlette_testclient = pytest.importorskip("fastapi.testclient")
    client = starlette_testclient.TestClient(app_with_saml)
    resp = client.get("/v1/sso/saml/acme/metadata")
    assert resp.status_code == 200
    assert "application/samlmetadata+xml" in resp.headers["content-type"]
    assert "urn:greenlang:factors:sp:acme" in resp.text


def test_login_redirects_to_idp(app_with_saml):
    tc = pytest.importorskip("fastapi.testclient")
    client = tc.TestClient(app_with_saml, follow_redirects=False)
    resp = client.get("/v1/sso/saml/acme/login")
    assert resp.status_code in (302, 307)
    assert resp.headers["location"].startswith("https://idp.example.com/sso?")


def test_acs_accepts_valid_response_and_mints_jwt(app_with_saml):
    tc = pytest.importorskip("fastapi.testclient")
    client = tc.TestClient(app_with_saml)
    payload = {"SAMLResponse": _saml_response()}
    resp = client.post("/v1/sso/saml/acme/acs", data=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "token" in body and body["user"]["email"] == "alice@acme.com"


def test_acs_rejects_wrong_audience(app_with_saml):
    tc = pytest.importorskip("fastapi.testclient")
    client = tc.TestClient(app_with_saml)
    resp = client.post(
        "/v1/sso/saml/acme/acs",
        data={"SAMLResponse": _saml_response(audience="urn:evilcorp")},
    )
    assert resp.status_code == 401


def test_unknown_tenant_returns_404(app_with_saml):
    tc = pytest.importorskip("fastapi.testclient")
    client = tc.TestClient(app_with_saml)
    resp = client.get("/v1/sso/saml/unknown/metadata")
    assert resp.status_code == 404
