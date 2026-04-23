# -*- coding: utf-8 -*-
"""Tests for SCIM 2.0 endpoints (SEC-5)."""
from __future__ import annotations

import pytest

from greenlang.factors.middleware.scim import (
    InMemorySCIMStore,
    SCIMTokenStore,
    SCIM_SCHEMA_PATCH,
    SCIM_SCHEMA_USER,
    install_scim_routes,
)


@pytest.fixture()
def scim_app():
    fastapi = pytest.importorskip("fastapi")
    app = fastapi.FastAPI()
    store = InMemorySCIMStore()
    tokens = SCIMTokenStore()
    tokens.set("acme", "tok-acme")
    tokens.set("globex", "tok-globex")
    install_scim_routes(app, store=store, token_store=tokens)
    return app, store, tokens


@pytest.fixture()
def client(scim_app):
    tc = pytest.importorskip("fastapi.testclient")
    app, store, tokens = scim_app
    c = tc.TestClient(app)
    return c, store, tokens


def _auth(tok: str):
    return {"Authorization": f"Bearer {tok}"}


def test_service_provider_config_advertises_bulk_and_patch(client):
    c, _, _ = client
    resp = c.get("/v1/scim/acme/v2/ServiceProviderConfig", headers=_auth("tok-acme"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["patch"]["supported"] is True
    assert body["bulk"]["supported"] is True


def test_unauthenticated_requests_rejected(client):
    c, _, _ = client
    resp = c.get("/v1/scim/acme/v2/Users")
    assert resp.status_code == 401


def test_wrong_tenant_token_rejected(client):
    c, _, _ = client
    resp = c.get("/v1/scim/acme/v2/Users", headers=_auth("tok-globex"))
    assert resp.status_code == 401


def test_user_provision_and_lookup(client):
    c, _, _ = client
    payload = {
        "schemas": [SCIM_SCHEMA_USER],
        "userName": "alice@acme.com",
        "active": True,
        "displayName": "Alice Anderson",
        "name": {"givenName": "Alice", "familyName": "Anderson"},
        "emails": [{"value": "alice@acme.com", "primary": True}],
    }
    resp = c.post("/v1/scim/acme/v2/Users", json=payload, headers=_auth("tok-acme"))
    assert resp.status_code == 201, resp.text
    uid = resp.json()["id"]

    # Lookup by id
    got = c.get(f"/v1/scim/acme/v2/Users/{uid}", headers=_auth("tok-acme"))
    assert got.status_code == 200
    assert got.json()["userName"] == "alice@acme.com"


def test_duplicate_username_conflicts(client):
    c, _, _ = client
    payload = {
        "schemas": [SCIM_SCHEMA_USER],
        "userName": "bob@acme.com",
    }
    c.post("/v1/scim/acme/v2/Users", json=payload, headers=_auth("tok-acme"))
    resp = c.post("/v1/scim/acme/v2/Users", json=payload, headers=_auth("tok-acme"))
    assert resp.status_code == 409


def test_filter_by_username(client):
    c, _, _ = client
    for name in ("alice", "bob", "eve"):
        c.post(
            "/v1/scim/acme/v2/Users",
            json={"schemas": [SCIM_SCHEMA_USER], "userName": f"{name}@acme.com"},
            headers=_auth("tok-acme"),
        )
    resp = c.get(
        '/v1/scim/acme/v2/Users?filter=userName eq "alice@acme.com"',
        headers=_auth("tok-acme"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["totalResults"] == 1
    assert body["Resources"][0]["userName"] == "alice@acme.com"


def test_filter_with_and_operator(client):
    c, _, _ = client
    c.post(
        "/v1/scim/acme/v2/Users",
        json={
            "schemas": [SCIM_SCHEMA_USER],
            "userName": "carol@acme.com",
            "active": True,
            "emails": [{"value": "carol@acme.com"}],
        },
        headers=_auth("tok-acme"),
    )
    resp = c.get(
        '/v1/scim/acme/v2/Users?filter=active eq "true" and emails.value co "carol"',
        headers=_auth("tok-acme"),
    )
    assert resp.status_code == 200
    assert resp.json()["totalResults"] == 1


def test_suspend_via_patch(client):
    c, _, _ = client
    created = c.post(
        "/v1/scim/acme/v2/Users",
        json={"schemas": [SCIM_SCHEMA_USER], "userName": "dan@acme.com", "active": True},
        headers=_auth("tok-acme"),
    ).json()
    uid = created["id"]
    resp = c.patch(
        f"/v1/scim/acme/v2/Users/{uid}",
        json={
            "schemas": [SCIM_SCHEMA_PATCH],
            "Operations": [{"op": "replace", "path": "active", "value": False}],
        },
        headers=_auth("tok-acme"),
    )
    assert resp.status_code == 200
    assert resp.json()["active"] is False


def test_deprovision_soft_deletes_user(client):
    c, store, _ = client
    created = c.post(
        "/v1/scim/acme/v2/Users",
        json={"schemas": [SCIM_SCHEMA_USER], "userName": "eve@acme.com"},
        headers=_auth("tok-acme"),
    ).json()
    uid = created["id"]
    resp = c.delete(f"/v1/scim/acme/v2/Users/{uid}", headers=_auth("tok-acme"))
    assert resp.status_code == 204
    got = c.get(f"/v1/scim/acme/v2/Users/{uid}", headers=_auth("tok-acme"))
    assert got.status_code == 404


def test_tenant_isolation(client):
    c, _, _ = client
    c.post(
        "/v1/scim/acme/v2/Users",
        json={"schemas": [SCIM_SCHEMA_USER], "userName": "x@acme.com"},
        headers=_auth("tok-acme"),
    )
    resp = c.get("/v1/scim/globex/v2/Users", headers=_auth("tok-globex"))
    assert resp.status_code == 200
    assert resp.json()["totalResults"] == 0


def test_group_membership(client):
    c, _, _ = client
    u = c.post(
        "/v1/scim/acme/v2/Users",
        json={"schemas": [SCIM_SCHEMA_USER], "userName": "frank@acme.com"},
        headers=_auth("tok-acme"),
    ).json()
    g = c.post(
        "/v1/scim/acme/v2/Groups",
        json={"schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
              "displayName": "analysts"},
        headers=_auth("tok-acme"),
    ).json()
    resp = c.patch(
        f"/v1/scim/acme/v2/Groups/{g['id']}",
        json={
            "schemas": [SCIM_SCHEMA_PATCH],
            "Operations": [
                {"op": "add", "path": "members", "value": [{"value": u["id"]}]}
            ],
        },
        headers=_auth("tok-acme"),
    )
    assert resp.status_code == 200
    got = c.get(f"/v1/scim/acme/v2/Groups/{g['id']}", headers=_auth("tok-acme")).json()
    assert len(got["members"]) == 1 and got["members"][0]["value"] == u["id"]


def test_bulk_endpoint(client):
    c, _, _ = client
    bulk = {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:BulkRequest"],
        "Operations": [
            {
                "method": "POST",
                "path": "/Users",
                "bulkId": "bulk-1",
                "data": {"schemas": [SCIM_SCHEMA_USER], "userName": "gina@acme.com"},
            },
            {
                "method": "POST",
                "path": "/Users",
                "bulkId": "bulk-2",
                "data": {"schemas": [SCIM_SCHEMA_USER], "userName": "harry@acme.com"},
            },
        ],
    }
    resp = c.post("/v1/scim/acme/v2/Bulk", json=bulk, headers=_auth("tok-acme"))
    assert resp.status_code == 200
    ops = resp.json()["Operations"]
    assert len(ops) == 2 and all(op["status"] == "201" for op in ops)
