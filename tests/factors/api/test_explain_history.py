# -*- coding: utf-8 -*-
"""Hosted explain log tests (W4-C / API13)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.factors.api_v1_routes import api_v1_router
from greenlang.factors.explain_history import (
    ExplainHistoryStore,
    can_subscribe,
    retention_days_for_tier,
    set_default_store,
)


def _build(user: Optional[Dict[str, Any]]) -> TestClient:
    app = FastAPI()

    @app.middleware("http")
    async def _inject(request, call_next):
        if user is not None:
            request.state.user = user
        return await call_next(request)

    app.include_router(api_v1_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_store():
    set_default_store(ExplainHistoryStore(None))
    yield


# ---------------------------------------------------------------------------
# Retention rules
# ---------------------------------------------------------------------------


def test_retention_rules():
    assert retention_days_for_tier("community") is None  # cannot subscribe
    assert retention_days_for_tier("pro") == 90
    assert retention_days_for_tier("platform") == 365
    assert retention_days_for_tier("enterprise") is None  # indefinite


def test_community_cannot_subscribe():
    assert can_subscribe("community") is False
    assert can_subscribe("pro") is True
    assert can_subscribe("enterprise") is True


# ---------------------------------------------------------------------------
# Subscribe / unsubscribe endpoints
# ---------------------------------------------------------------------------


def test_subscribe_endpoint_community_forbidden():
    client = _build({"tenant_id": "t1", "tier": "community"})
    r = client.post("/v1/explain/subscribe")
    assert r.status_code == 403


def test_subscribe_endpoint_pro_ok():
    client = _build({"tenant_id": "t1", "tier": "pro"})
    r = client.post("/v1/explain/subscribe")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["tier"] == "pro"
    assert body["retention_days"] == 90


def test_unsubscribe_endpoint_idempotent():
    client = _build({"tenant_id": "t1", "tier": "pro"})
    client.post("/v1/explain/subscribe")
    r = client.delete("/v1/explain/subscribe")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# History listing + tenant isolation
# ---------------------------------------------------------------------------


def test_list_history_requires_subscription():
    client = _build({"tenant_id": "t1", "tier": "community"})
    r = client.get("/v1/explain/history")
    assert r.status_code == 403


def test_tenant_isolation_on_list(monkeypatch):
    store = ExplainHistoryStore(None)
    set_default_store(store)

    store.subscribe(tenant_id="t-alpha", tier="pro")
    store.subscribe(tenant_id="t-beta", tier="pro")

    store.record(tenant_id="t-alpha", tier="pro", payload={"x": 1}, factor_id="f1")
    store.record(tenant_id="t-beta", tier="pro", payload={"y": 2}, factor_id="f2")

    alpha = _build({"tenant_id": "t-alpha", "tier": "pro"})
    r = alpha.get("/v1/explain/history")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["factor_id"] == "f1"


def test_get_by_receipt_cross_tenant_denied():
    store = ExplainHistoryStore(None)
    set_default_store(store)

    store.subscribe(tenant_id="t-alpha", tier="pro")
    rid = store.record(tenant_id="t-alpha", tier="pro", payload={"z": 1}, factor_id="fX")
    assert rid is not None

    beta = _build({"tenant_id": "t-beta", "tier": "pro"})
    # Subscribe beta first so endpoint doesn't 403 on the tier guard.
    store.subscribe(tenant_id="t-beta", tier="pro")
    r = beta.get(f"/v1/explain/history/{rid}")
    assert r.status_code == 404  # scoped lookup returns None → 404


def test_unsubscribed_tenant_not_recorded():
    store = ExplainHistoryStore(None)
    set_default_store(store)

    # No subscribe call.
    rid = store.record(tenant_id="t-silent", tier="pro", payload={"a": 1})
    assert rid is None
    assert store.list_history(tenant_id="t-silent") == []


# ---------------------------------------------------------------------------
# Enterprise purge
# ---------------------------------------------------------------------------


def test_enterprise_purge_allowed_others_denied():
    store = ExplainHistoryStore(None)
    set_default_store(store)
    store.subscribe(tenant_id="t-ent", tier="enterprise")
    store.record(tenant_id="t-ent", tier="enterprise", payload={"k": "v"})

    pro = _build({"tenant_id": "t-ent", "tier": "pro"})
    assert pro.post("/v1/explain/history/purge").status_code == 403

    ent = _build({"tenant_id": "t-ent", "tier": "enterprise"})
    r = ent.post("/v1/explain/history/purge")
    assert r.status_code == 200
    assert r.json()["deleted"] >= 1


def test_gc_expired_deletes_old_rows():
    store = ExplainHistoryStore(None)
    set_default_store(store)
    store.subscribe(tenant_id="t-pro", tier="pro")
    # Force an expired row by writing directly to the store's conn.
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO factors_explain_history "
        "(receipt_id, tenant_id, factor_id, edition_id, stored_at, expires_at, payload_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("exr_old", "t-pro", None, None, "2000-01-01T00:00:00Z", "2000-01-02T00:00:00Z", "{}"),
    )
    assert store.gc_expired() >= 1


# ---------------------------------------------------------------------------
# Unauthed
# ---------------------------------------------------------------------------


def test_unauthed_401():
    client = _build(user=None)
    r = client.get("/v1/explain/history")
    assert r.status_code == 401
