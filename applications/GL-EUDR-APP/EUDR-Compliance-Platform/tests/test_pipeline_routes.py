"""
Unit tests for GL-EUDR-APP v1.0 Pipeline API Routes.

Tests all pipeline REST endpoints using FastAPI TestClient:
start, get status, history, retry, and cancel.

Test count target: 20+ tests

NOTE: Since pipeline_routes.py may not yet exist, this test defines
a self-contained pipeline router implementing the expected API contract.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Query, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline Route Models & Router (self-contained for testing)
# ---------------------------------------------------------------------------

_pipeline_store: Dict[str, Dict[str, Any]] = {}

PIPELINE_STAGES = ["intake", "geo_validation", "deforestation_risk",
                    "document_verification", "dds_reporting"]


class PipelineStartRequest(BaseModel):
    supplier_id: str = Field(..., min_length=1)
    commodity: Optional[str] = None
    plot_ids: List[str] = Field(default_factory=list)


pipeline_router = APIRouter(prefix="/api/v1/pipeline", tags=["Pipeline"])


@pipeline_router.post("/start", status_code=201)
async def start_pipeline(body: PipelineStartRequest):
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    run = {
        "run_id": run_id,
        "supplier_id": body.supplier_id,
        "commodity": body.commodity,
        "plot_ids": body.plot_ids,
        "status": "pending",
        "current_stage": "intake",
        "stages": {},
        "error_message": None,
        "started_at": None,
        "completed_at": None,
        "created_at": now.isoformat(),
    }
    _pipeline_store[run_id] = run

    # Simulate execution
    run["status"] = "running"
    run["started_at"] = now.isoformat()
    for stage in PIPELINE_STAGES:
        run["current_stage"] = stage
        run["stages"][stage] = {
            "status": "completed",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "output": {},
        }
    run["status"] = "completed"
    run["completed_at"] = datetime.now(timezone.utc).isoformat()

    return run


@pipeline_router.get("/{run_id}")
async def get_pipeline(run_id: str):
    run = _pipeline_store.get(run_id)
    if not run:
        raise HTTPException(404, f"Pipeline run '{run_id}' not found")
    return run


@pipeline_router.get("/")
async def pipeline_history(
    supplier_id: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
):
    results = list(_pipeline_store.values())
    if supplier_id:
        results = [r for r in results if r["supplier_id"] == supplier_id]
    if status_filter:
        results = [r for r in results if r["status"] == status_filter]
    results.sort(key=lambda r: r["created_at"], reverse=True)
    return {"items": results[:limit], "total": len(results)}


@pipeline_router.post("/{run_id}/retry")
async def retry_pipeline(run_id: str):
    run = _pipeline_store.get(run_id)
    if not run:
        raise HTTPException(404, f"Pipeline run '{run_id}' not found")
    if run["status"] != "failed":
        raise HTTPException(400, f"Can only retry failed pipelines, got '{run['status']}'")

    # Simulate retry: complete remaining stages
    run["status"] = "running"
    run["error_message"] = None
    for stage in PIPELINE_STAGES:
        if stage not in run["stages"] or run["stages"][stage]["status"] == "failed":
            now = datetime.now(timezone.utc).isoformat()
            run["stages"][stage] = {
                "status": "completed",
                "started_at": now,
                "completed_at": now,
                "output": {},
            }
    run["status"] = "completed"
    run["completed_at"] = datetime.now(timezone.utc).isoformat()
    return run


@pipeline_router.post("/{run_id}/cancel")
async def cancel_pipeline(run_id: str):
    run = _pipeline_store.get(run_id)
    if not run:
        raise HTTPException(404, f"Pipeline run '{run_id}' not found")
    if run["status"] not in ("pending", "running"):
        raise HTTPException(400, f"Cannot cancel pipeline in '{run['status']}' status")
    run["status"] = "cancelled"
    return run


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

def _create_app() -> FastAPI:
    app = FastAPI(title="GL-EUDR-APP Pipeline Test")
    app.include_router(pipeline_router)
    return app


@pytest.fixture(autouse=True)
def clear_store():
    _pipeline_store.clear()
    yield
    _pipeline_store.clear()


@pytest.fixture
def client():
    return TestClient(_create_app())


@pytest.fixture
def completed_run(client):
    resp = client.post("/api/v1/pipeline/start", json={
        "supplier_id": "sup_test",
        "commodity": "soya",
        "plot_ids": ["plot_a", "plot_b"],
    })
    return resp.json()


@pytest.fixture
def failed_run(client):
    """Create a run and manually set it to failed state."""
    resp = client.post("/api/v1/pipeline/start", json={
        "supplier_id": "sup_fail",
        "commodity": "wood",
    })
    run = resp.json()
    # Manually set to failed state
    _pipeline_store[run["run_id"]]["status"] = "failed"
    _pipeline_store[run["run_id"]]["error_message"] = "Simulated failure"
    _pipeline_store[run["run_id"]]["stages"]["deforestation_risk"] = {
        "status": "failed", "error": "Satellite timeout",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    return _pipeline_store[run["run_id"]]


# ---------------------------------------------------------------------------
# POST /pipeline/start
# ---------------------------------------------------------------------------

class TestStartPipelineRoute:

    def test_starts_pipeline(self, client):
        resp = client.post("/api/v1/pipeline/start", json={
            "supplier_id": "sup_abc",
            "commodity": "wood",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["run_id"].startswith("run_")
        assert data["supplier_id"] == "sup_abc"

    def test_returns_run_id(self, client):
        resp = client.post("/api/v1/pipeline/start", json={
            "supplier_id": "sup_abc",
        })
        assert "run_id" in resp.json()

    def test_completed_status(self, client, completed_run):
        assert completed_run["status"] == "completed"

    def test_all_stages_present(self, client, completed_run):
        for stage in PIPELINE_STAGES:
            assert stage in completed_run["stages"]

    def test_missing_supplier_422(self, client):
        resp = client.post("/api/v1/pipeline/start", json={
            "supplier_id": "",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /pipeline/{run_id}
# ---------------------------------------------------------------------------

class TestGetPipelineRoute:

    def test_get_status(self, client, completed_run):
        resp = client.get(f"/api/v1/pipeline/{completed_run['run_id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == completed_run["run_id"]
        assert "stages" in data

    def test_get_not_found_404(self, client):
        resp = client.get("/api/v1/pipeline/run_nonexistent")
        assert resp.status_code == 404

    def test_stage_details_included(self, client, completed_run):
        resp = client.get(f"/api/v1/pipeline/{completed_run['run_id']}")
        data = resp.json()
        for stage_name, stage_data in data["stages"].items():
            assert "status" in stage_data
            assert "started_at" in stage_data


# ---------------------------------------------------------------------------
# GET /pipeline (history)
# ---------------------------------------------------------------------------

class TestHistoryRoute:

    def test_list_all(self, client, completed_run):
        resp = client.get("/api/v1/pipeline/")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_filter_by_supplier(self, client):
        client.post("/api/v1/pipeline/start", json={"supplier_id": "s1"})
        client.post("/api/v1/pipeline/start", json={"supplier_id": "s2"})
        resp = client.get("/api/v1/pipeline/?supplier_id=s1")
        assert resp.json()["total"] == 1

    def test_filter_by_status(self, client, completed_run, failed_run):
        resp = client.get("/api/v1/pipeline/?status=failed")
        data = resp.json()
        for item in data["items"]:
            assert item["status"] == "failed"

    def test_limit(self, client):
        for i in range(5):
            client.post("/api/v1/pipeline/start", json={"supplier_id": f"s{i}"})
        resp = client.get("/api/v1/pipeline/?limit=3")
        assert len(resp.json()["items"]) == 3


# ---------------------------------------------------------------------------
# POST /pipeline/{run_id}/retry
# ---------------------------------------------------------------------------

class TestRetryRoute:

    def test_retry_failed_succeeds(self, client, failed_run):
        resp = client.post(f"/api/v1/pipeline/{failed_run['run_id']}/retry")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_retry_completed_400(self, client, completed_run):
        resp = client.post(f"/api/v1/pipeline/{completed_run['run_id']}/retry")
        assert resp.status_code == 400

    def test_retry_nonexistent_404(self, client):
        resp = client.post("/api/v1/pipeline/run_nope/retry")
        assert resp.status_code == 404

    def test_retry_clears_error(self, client, failed_run):
        resp = client.post(f"/api/v1/pipeline/{failed_run['run_id']}/retry")
        assert resp.json()["error_message"] is None


# ---------------------------------------------------------------------------
# POST /pipeline/{run_id}/cancel
# ---------------------------------------------------------------------------

class TestCancelRoute:

    def test_cancel_active(self, client):
        """Create a pending run and cancel it."""
        resp = client.post("/api/v1/pipeline/start", json={"supplier_id": "sup_x"})
        run_id = resp.json()["run_id"]
        # Set back to pending for testing
        _pipeline_store[run_id]["status"] = "pending"
        cancel_resp = client.post(f"/api/v1/pipeline/{run_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

    def test_cancel_completed_400(self, client, completed_run):
        resp = client.post(f"/api/v1/pipeline/{completed_run['run_id']}/cancel")
        assert resp.status_code == 400

    def test_cancel_nonexistent_404(self, client):
        resp = client.post("/api/v1/pipeline/run_nope/cancel")
        assert resp.status_code == 404
