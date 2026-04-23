# -*- coding: utf-8 -*-
"""
Tests for the OEM white-label onboarding lifecycle (Track C-5).

Coverage:
* OEM provisions multiple sub-tenants successfully.
* Subset enforcement: a sub-tenant cannot escalate above the parent
  OEM's redistribution grant.
* The license-class redistribution guard
  (``check_oem_redistribution``) returns False for factors outside the
  parent grant even when the OEM listed those classes against a
  sub-tenant.
* The OEM API returns branding metadata in responses for OEM callers.

The tests run against the in-process OEM registry; no real DB / Stripe
/ Vault round-trip is required. The fixture below resets the registry
between tests so behaviour is deterministic regardless of order.
"""
from __future__ import annotations

from typing import Iterator

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.factors.entitlements import (
    EntitlementError,
    check_oem_redistribution,
)
from greenlang.factors.onboarding.api import oem_router
from greenlang.factors.onboarding.branding_config import BrandingConfig
from greenlang.factors.onboarding.partner_setup import (
    OEM_ELIGIBLE_PARENT_PLANS,
    OEM_GRANT_CLASSES,
    OemError,
    _reset_oem_registry,
    create_oem_partner,
    get_oem_partner,
    get_redistribution_grant,
    list_subtenants,
    provision_subtenant,
    revoke_subtenant,
    update_branding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Reset the in-process OEM registry between every test."""
    _reset_oem_registry()
    yield
    _reset_oem_registry()


@pytest.fixture()
def api_client() -> TestClient:
    """Return a TestClient that mounts only the OEM router."""
    app = FastAPI()
    app.include_router(oem_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Lifecycle smoke
# ---------------------------------------------------------------------------


def test_create_oem_partner_returns_grant_and_api_key():
    partner = create_oem_partner(
        name="Acme Sustainability",
        contact_email="ops@acme.example.com",
        redistribution_grants=["open", "public_us_government"],
        parent_plan="consulting_platform",
    )
    assert partner.id.startswith("oem_acme-sustainability_")
    assert partner.api_key.startswith("gl_oem_")
    assert partner.parent_plan == "consulting_platform"
    assert set(partner.grant.allowed_classes) == {"open", "public_us_government"}
    assert get_oem_partner(partner.id) is partner


def test_create_oem_rejects_unknown_plan():
    with pytest.raises(OemError):
        create_oem_partner(
            name="X", contact_email="x@y.com",
            redistribution_grants=["open"],
            parent_plan="hobbyist",
        )


def test_create_oem_rejects_unknown_grant_class():
    with pytest.raises(OemError):
        create_oem_partner(
            name="X", contact_email="x@y.com",
            redistribution_grants=["definitely-not-a-class"],
            parent_plan="platform",
        )


def test_create_oem_rejects_empty_grants():
    with pytest.raises(OemError):
        create_oem_partner(
            name="X", contact_email="x@y.com",
            redistribution_grants=[],
            parent_plan="platform",
        )


def test_create_oem_rejects_invalid_email():
    with pytest.raises(OemError):
        create_oem_partner(
            name="X", contact_email="not-an-email",
            redistribution_grants=["open"],
            parent_plan="platform",
        )


def test_oem_provisions_two_subtenants():
    partner = create_oem_partner(
        name="Acme",
        contact_email="ops@acme.example.com",
        redistribution_grants=["open", "public_us_government"],
        parent_plan="consulting_platform",
    )
    sub1 = provision_subtenant(partner.id, "Tenant One", None, ["open"])
    sub2 = provision_subtenant(
        partner.id, "Tenant Two", None, ["public_us_government"]
    )
    assert sub1.id != sub2.id
    assert sub1.entitlements == ["open"]
    assert sub2.entitlements == ["public_us_government"]
    listing = list_subtenants(partner.id)
    assert {s.id for s in listing} == {sub1.id, sub2.id}


def test_subtenant_with_empty_entitlements_is_allowed():
    partner = create_oem_partner(
        name="Acme", contact_email="a@b.com",
        redistribution_grants=["open"],
        parent_plan="platform",
    )
    sub = provision_subtenant(partner.id, "EmptySub", None, [])
    assert sub.entitlements == []
    assert sub.active


def test_subtenant_revocation_is_idempotent():
    partner = create_oem_partner(
        name="Acme", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    sub = provision_subtenant(partner.id, "Sub", None, ["open"])
    assert revoke_subtenant(partner.id, sub.id) is True
    assert revoke_subtenant(partner.id, sub.id) is False


def test_revoke_unknown_subtenant_returns_false():
    partner = create_oem_partner(
        name="Acme", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    assert revoke_subtenant(partner.id, "not-a-real-sub") is False


def test_get_redistribution_grant_returns_full_payload():
    partner = create_oem_partner(
        name="Acme", contact_email="a@b.com",
        redistribution_grants=["open", "uk_open_government"],
        parent_plan="platform",
    )
    grant = get_redistribution_grant(partner.id)
    assert grant.oem_id == partner.id
    assert grant.parent_plan == "platform"
    assert "open" in grant.allowed_classes


# ---------------------------------------------------------------------------
# Subset / license enforcement
# ---------------------------------------------------------------------------


def test_oem_with_open_grant_cannot_provision_licensed_subtenant():
    """Sub-tenant entitlement outside the parent grant raises EntitlementError."""
    partner = create_oem_partner(
        name="OpenOnly",
        contact_email="ops@openonly.example.com",
        redistribution_grants=["open"],
        parent_plan="consulting_platform",
    )
    with pytest.raises(EntitlementError):
        provision_subtenant(
            partner.id,
            "ShouldFail",
            None,
            ["licensed"],  # 'licensed' is not in the parent grant
        )


def test_subtenant_with_unknown_entitlement_raises_entitlement_error():
    partner = create_oem_partner(
        name="X", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    with pytest.raises(EntitlementError):
        provision_subtenant(partner.id, "Sub", None, ["bogus_class"])


def test_check_oem_redistribution_open_grant_admits_open_factor():
    partner = create_oem_partner(
        name="OpenCo", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    factor = {"license_class": "open"}
    assert check_oem_redistribution(partner.id, factor) is True


def test_check_oem_redistribution_open_grant_denies_pcaf_factor():
    partner = create_oem_partner(
        name="OpenCo", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    # PCAF attribution is not in the open-only OEM grant.
    factor = {"license_class": "pcaf_attribution"}
    assert check_oem_redistribution(partner.id, factor) is False


def test_check_oem_redistribution_handles_nested_licensing_block():
    """The guard probes the canonical_v2 ``licensing.redistribution`` shape."""
    partner = create_oem_partner(
        name="GovCo", contact_email="a@b.com",
        redistribution_grants=["public_us_government"],
        parent_plan="platform",
    )
    factor = {"licensing": {"license_class": "public_us_government"}}
    assert check_oem_redistribution(partner.id, factor) is True
    other = {"licensing": {"license_class": "uk_open_government"}}
    assert check_oem_redistribution(partner.id, other) is False


def test_check_oem_redistribution_unknown_oem_denies():
    """Unknown OEM ids fail closed."""
    assert check_oem_redistribution("not-a-real-oem", {"license_class": "open"}) is False


def test_check_oem_redistribution_missing_license_class_denies():
    """Factors missing a license class are denied (operator must fix catalog)."""
    partner = create_oem_partner(
        name="X", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    assert check_oem_redistribution(partner.id, {"co2e_per_unit": 1.0}) is False
    assert check_oem_redistribution(partner.id, None) is False


def test_check_oem_redistribution_blank_oem_denies():
    """Empty / falsy oem_id always denies."""
    assert check_oem_redistribution("", {"license_class": "open"}) is False


def test_subtenant_negative_path_when_oem_grant_does_not_include_class(api_client: TestClient):
    """Even if the OEM accidentally lists ``licensed`` against a sub-tenant
    in the API call, the parent grant ('open' only) blocks it with HTTP 403.
    """
    partner = create_oem_partner(
        name="OpenOnly", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="consulting_platform",
    )
    resp = api_client.post(
        "/v1/oem/subtenants",
        headers={"X-OEM-Id": partner.id},
        json={
            "name": "Sneaky Tenant",
            "entitlements": ["licensed"],  # not granted to the parent OEM
        },
    )
    assert resp.status_code == 403
    body = resp.json()
    assert "exceed OEM grant" in body["detail"] or "OEM grant" in body["detail"]


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_signup_route_returns_partner_with_api_key(api_client: TestClient):
    resp = api_client.post(
        "/v1/oem/signup",
        json={
            "name": "Acme Platform",
            "contact_email": "ops@acme.example.com",
            "redistribution_grants": ["open"],
            "parent_plan": "consulting_platform",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Acme Platform"
    assert body["api_key"].startswith("gl_oem_")
    assert "id" in body and body["id"].startswith("oem_")


def test_me_route_requires_oem_id_header(api_client: TestClient):
    resp = api_client.get("/v1/oem/me")
    assert resp.status_code == 401


def test_me_route_returns_oem_context_for_oem_caller(api_client: TestClient):
    partner = create_oem_partner(
        name="Acme", contact_email="ops@acme.example.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    update_branding(partner.id, BrandingConfig(
        logo_url="https://cdn.acme.com/logo.svg",
        primary_color="#0A66C2",
        powered_by_text="Powered by GreenLang",
    ))
    resp = api_client.get("/v1/oem/me", headers={"X-OEM-Id": partner.id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == partner.id
    assert body["branding"]["logo_url"] == "https://cdn.acme.com/logo.svg"
    assert body["branding"]["powered_by_text"] == "Powered by GreenLang"
    # The api_key MUST NOT be echoed on subsequent reads.
    assert "api_key" not in body
    assert "api_key_prefix" not in body


def test_branding_route_round_trip(api_client: TestClient):
    partner = create_oem_partner(
        name="Acme", contact_email="ops@acme.example.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    payload = {
        "branding": {
            "logo_url": "https://cdn.acme.com/logo.png",
            "primary_color": "#123456",
            "secondary_color": "#FEDCBA",
            "support_email": "support@acme.example.com",
            "support_url": "https://acme.example.com/support",
            "custom_domain": "factors.acme.com",
            "attribution_required": True,
            "powered_by_text": "Powered by GreenLang",
        }
    }
    resp = api_client.post(
        "/v1/oem/branding",
        headers={"X-OEM-Id": partner.id},
        json=payload,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["oem_id"] == partner.id
    assert body["branding"]["logo_url"] == "https://cdn.acme.com/logo.png"
    assert body["branding"]["primary_color"] == "#123456"


def test_subtenants_list_route(api_client: TestClient):
    partner = create_oem_partner(
        name="Acme", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    api_client.post(
        "/v1/oem/subtenants",
        headers={"X-OEM-Id": partner.id},
        json={"name": "Sub One", "entitlements": ["open"]},
    )
    api_client.post(
        "/v1/oem/subtenants",
        headers={"X-OEM-Id": partner.id},
        json={"name": "Sub Two", "entitlements": ["open"]},
    )
    resp = api_client.get(
        "/v1/oem/subtenants", headers={"X-OEM-Id": partner.id}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    names = {s["name"] for s in body["subtenants"]}
    assert names == {"Sub One", "Sub Two"}


def test_subtenant_delete_route(api_client: TestClient):
    partner = create_oem_partner(
        name="Acme", contact_email="a@b.com",
        redistribution_grants=["open"], parent_plan="platform",
    )
    create = api_client.post(
        "/v1/oem/subtenants",
        headers={"X-OEM-Id": partner.id},
        json={"name": "Sub", "entitlements": ["open"]},
    )
    sub_id = create.json()["id"]
    delete = api_client.delete(
        f"/v1/oem/subtenants/{sub_id}",
        headers={"X-OEM-Id": partner.id},
    )
    assert delete.status_code == 200
    assert delete.json()["revoked"] is True
    # Second delete is 404 (idempotent flag is on the helper, not the route).
    again = api_client.delete(
        f"/v1/oem/subtenants/{sub_id}",
        headers={"X-OEM-Id": partner.id},
    )
    assert again.status_code == 404


def test_redistribution_route_returns_grant(api_client: TestClient):
    partner = create_oem_partner(
        name="Acme", contact_email="a@b.com",
        redistribution_grants=["open", "uk_open_government"],
        parent_plan="platform",
    )
    resp = api_client.get(
        "/v1/oem/redistribution", headers={"X-OEM-Id": partner.id}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["oem_id"] == partner.id
    assert set(body["allowed_classes"]) == {"open", "uk_open_government"}
    assert "open" in body["known_classes"]
    assert "platform" in body["eligible_parent_plans"]


def test_signup_route_validates_grant_class_at_pydantic_layer(api_client: TestClient):
    """Unknown grant classes surface as 422 from the lifecycle helper."""
    resp = api_client.post(
        "/v1/oem/signup",
        json={
            "name": "Bad Co",
            "contact_email": "a@b.com",
            "redistribution_grants": ["does-not-exist"],
            "parent_plan": "platform",
        },
    )
    assert resp.status_code == 422


def test_acceptance_smoke_flow():
    """Mirrors the README acceptance snippet end-to-end."""
    p = create_oem_partner("Test", "a@b.com", ["open"], "consulting_platform")
    s = provision_subtenant(p.id, "sub1", None, ["open"])
    assert s.id
    assert s.id.startswith(p.id)
    assert "open" in p.grant.allowed_classes
    # The grant lookup should be stable.
    assert get_redistribution_grant(p.id) is p.grant


def test_oem_grant_classes_are_a_superset_of_source_registry_classes():
    """Sanity check: every license_class published by the source registry
    has a matching marketing/grant slug in OEM_GRANT_CLASSES."""
    expected_subset = {
        "open",
        "public_us_government",
        "uk_open_government",
        "public_in_government",
        "public_international",
        "eu_publication",
        "academic_research",
        "wri_wbcsd_terms",
        "smart_freight_terms",
        "registry_terms",
        "pcaf_attribution",
        "greenlang_terms",
        "commercial_connector",
    }
    assert expected_subset.issubset(set(OEM_GRANT_CLASSES))
    assert "consulting_platform" in OEM_ELIGIBLE_PARENT_PLANS
