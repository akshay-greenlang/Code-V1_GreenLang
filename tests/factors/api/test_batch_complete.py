# -*- coding: utf-8 -*-
"""Batch API tests (W4-C / API11).

Covers:
  * CSV + JSON submission paths
  * Tier rate-limit enforcement (Community → 403, Pro daily cap, etc.)
  * 10k-row hard cap
  * Status / results / errors endpoints
  * Tenant isolation on GET
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.factors.api_v1_routes import (
    BATCH_DAILY_CAPS,
    BATCH_MAX_ROWS_PER_SUBMIT,
    api_v1_router,
)
from greenlang.factors.batch_jobs import (
    BatchJobStatus,
    SQLiteBatchJobQueue,
    get_default_queue,
    set_default_queue,
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
def _reset_queue():
    set_default_queue(SQLiteBatchJobQueue(None))
    yield


# ---------------------------------------------------------------------------
# 1. Tier caps constants
# ---------------------------------------------------------------------------


def test_tier_cap_constants():
    assert BATCH_DAILY_CAPS["community"] == 0
    assert BATCH_DAILY_CAPS["pro"] == 10_000
    assert BATCH_DAILY_CAPS["platform"] == 100_000
    assert BATCH_DAILY_CAPS["enterprise"] is None
    assert BATCH_MAX_ROWS_PER_SUBMIT == 10_000


# ---------------------------------------------------------------------------
# 2. Community tier cannot submit
# ---------------------------------------------------------------------------


def test_community_tier_forbidden():
    client = _build({"tenant_id": "t1", "tier": "community"})
    r = client.post("/v1/batch/resolve", json={"requests": [{"activity": "x"}]})
    assert r.status_code == 403
    assert r.json()["detail"]["error"] == "tier_forbidden"


# ---------------------------------------------------------------------------
# 3. JSON submission works (Pro tier)
# ---------------------------------------------------------------------------


def test_json_submission_returns_batch_id():
    client = _build({"tenant_id": "t1", "tier": "pro", "user_id": "u1"})
    rows = [{"activity": f"row-{i}"} for i in range(3)]
    r = client.post("/v1/batch/resolve", json={"requests": rows})
    assert r.status_code == 202, r.text
    payload = r.json()
    assert payload["status"] == "queued"
    assert payload["batch_id"]
    assert payload["request_count"] == 3
    assert payload["status_url"].endswith(payload["batch_id"])


# ---------------------------------------------------------------------------
# 4. CSV submission parses rows
# ---------------------------------------------------------------------------


def test_csv_submission_parses_rows():
    client = _build({"tenant_id": "t1", "tier": "pro", "user_id": "u1"})
    csv_body = "activity,method_profile\ndiesel,CORPORATE_SCOPE1\npetrol,CORPORATE_SCOPE1\n"
    r = client.post(
        "/v1/batch/resolve",
        content=csv_body.encode(),
        headers={"content-type": "text/csv"},
    )
    assert r.status_code == 202
    assert r.json()["request_count"] == 2


# ---------------------------------------------------------------------------
# 5. Unsupported content type → 415
# ---------------------------------------------------------------------------


def test_unsupported_content_type():
    client = _build({"tenant_id": "t1", "tier": "pro", "user_id": "u1"})
    r = client.post(
        "/v1/batch/resolve",
        content=b"<xml/>",
        headers={"content-type": "application/xml"},
    )
    assert r.status_code == 415


# ---------------------------------------------------------------------------
# 6. Empty payload → 400
# ---------------------------------------------------------------------------


def test_empty_submission_rejected():
    client = _build({"tenant_id": "t1", "tier": "pro", "user_id": "u1"})
    r = client.post("/v1/batch/resolve", json={"requests": []})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 7. 10k-row hard cap
# ---------------------------------------------------------------------------


def test_too_many_rows_413():
    client = _build({"tenant_id": "t1", "tier": "enterprise", "user_id": "u1"})
    rows = [{"activity": f"a{i}"} for i in range(10_001)]
    r = client.post("/v1/batch/resolve", json={"requests": rows})
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# 8. Daily cap for Pro tier enforced on second submission
# ---------------------------------------------------------------------------


def test_pro_daily_cap_enforced(monkeypatch):
    # Shrink pro cap to a small number so the test is fast.
    from greenlang.factors import api_v1_routes as mod
    monkeypatch.setitem(mod.BATCH_DAILY_CAPS, "pro", 10)

    client = _build({"tenant_id": "t-pro", "tier": "pro", "user_id": "u"})
    rows1 = [{"activity": f"a{i}"} for i in range(8)]
    r1 = client.post("/v1/batch/resolve", json={"requests": rows1})
    assert r1.status_code == 202

    rows2 = [{"activity": f"b{i}"} for i in range(5)]
    r2 = client.post("/v1/batch/resolve", json={"requests": rows2})
    assert r2.status_code == 429
    assert r2.json()["detail"]["error"] == "batch_daily_cap"


# ---------------------------------------------------------------------------
# 9. Tenant isolation on status
# ---------------------------------------------------------------------------


def test_tenant_isolation_on_status_read():
    submitter = _build({"tenant_id": "t-alpha", "tier": "pro", "user_id": "u-alpha"})
    r = submitter.post("/v1/batch/resolve", json={"requests": [{"activity": "a"}]})
    assert r.status_code == 202
    batch_id = r.json()["batch_id"]

    # Another tenant cannot read this batch.
    other = _build({"tenant_id": "t-beta", "tier": "pro", "user_id": "u-beta"})
    r2 = other.get(f"/v1/batch/{batch_id}")
    assert r2.status_code == 403


# ---------------------------------------------------------------------------
# 10. Status endpoint returns queued shape
# ---------------------------------------------------------------------------


def test_status_endpoint_shape():
    client = _build({"tenant_id": "t1", "tier": "pro", "user_id": "u1"})
    r = client.post("/v1/batch/resolve", json={"requests": [{"activity": "a"}]})
    batch_id = r.json()["batch_id"]
    r2 = client.get(f"/v1/batch/{batch_id}")
    assert r2.status_code == 200
    body = r2.json()
    for k in (
        "batch_id",
        "status",
        "progress",
        "request_count",
        "submitted_at",
        "result_file_url",
        "error_file_url",
        "signed_receipt_manifest_url",
    ):
        assert k in body


# ---------------------------------------------------------------------------
# 11. Unknown batch → 404
# ---------------------------------------------------------------------------


def test_unknown_batch_404():
    client = _build({"tenant_id": "t1", "tier": "pro", "user_id": "u1"})
    r = client.get("/v1/batch/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 12. Unauthed caller → 401
# ---------------------------------------------------------------------------


def test_unauthed_request_401():
    client = _build(user=None)
    r = client.post("/v1/batch/resolve", json={"requests": [{"activity": "a"}]})
    assert r.status_code == 401
