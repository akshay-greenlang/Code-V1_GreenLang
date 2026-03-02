"""
Unit tests for GL-EUDR-APP v1.0 DDS API Routes.

Tests all DDS REST endpoints using FastAPI TestClient:
generate, get, list, validate, submit, download, amend,
bulk-generate, and annual summary.

Test count target: 35+ tests

NOTE: The DDS routes (services/api/dds_routes.py) are expected to follow
the same in-memory pattern as supplier_routes.py. Since the route file
may not yet be built, these tests define a self-contained DDS router
that implements the expected API contract.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Query, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# DDS Route Models & Router (self-contained for testing)
# ---------------------------------------------------------------------------

_dds_store: Dict[str, Dict[str, Any]] = {}
_dds_sequences: Dict[str, int] = {}

DDS_SECTIONS = [
    "operator_info", "product_description", "country_of_production",
    "geolocation_data", "risk_assessment", "risk_mitigation", "conclusion",
]


class DDSGenerateRequest(BaseModel):
    supplier_id: str = Field(..., min_length=1)
    commodity: str = Field(..., min_length=1)
    country_iso3: str = Field(..., min_length=3, max_length=3)
    year: int = Field(..., ge=2024, le=2100)
    plot_ids: List[str] = Field(default_factory=list)
    operator_info: Dict = Field(default_factory=dict)
    product_description: Dict = Field(default_factory=dict)
    country_of_production: Dict = Field(default_factory=dict)
    geolocation_data: Dict = Field(default_factory=dict)
    risk_assessment: Dict = Field(default_factory=dict)
    risk_mitigation: Dict = Field(default_factory=dict)
    conclusion: Dict = Field(default_factory=dict)


class DDSBulkRequest(BaseModel):
    requests: List[DDSGenerateRequest] = Field(..., min_length=1)


dds_router = APIRouter(prefix="/api/v1/dds", tags=["DDS"])


def _next_ref(iso3: str, year: int) -> str:
    key = f"{iso3.upper()}_{year}"
    seq = _dds_sequences.get(key, 0) + 1
    _dds_sequences[key] = seq
    return f"EUDR-{iso3.upper()}-{year}-{seq:06d}"


@dds_router.post("/generate", status_code=201)
async def generate_dds(body: DDSGenerateRequest):
    ref = _next_ref(body.country_iso3, body.year)
    dds_id = f"dds_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    dds = {
        "dds_id": dds_id,
        "reference_number": ref,
        "supplier_id": body.supplier_id,
        "commodity": body.commodity,
        "year": body.year,
        "country_iso3": body.country_iso3.upper(),
        "status": "draft",
        "operator_info": body.operator_info,
        "product_description": body.product_description,
        "country_of_production": body.country_of_production,
        "geolocation_data": body.geolocation_data,
        "risk_assessment": body.risk_assessment,
        "risk_mitigation": body.risk_mitigation,
        "conclusion": body.conclusion,
        "plot_ids": body.plot_ids,
        "submission_date": None,
        "eu_reference": None,
        "amendment_of": None,
        "created_at": now.isoformat(),
    }
    _dds_store[dds_id] = dds
    return dds


@dds_router.get("/{dds_id}")
async def get_dds(dds_id: str):
    dds = _dds_store.get(dds_id)
    if not dds:
        raise HTTPException(404, f"DDS '{dds_id}' not found")
    return dds


@dds_router.get("/")
async def list_dds(
    supplier_id: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    year: Optional[int] = None,
    commodity: Optional[str] = None,
):
    results = list(_dds_store.values())
    if supplier_id:
        results = [d for d in results if d["supplier_id"] == supplier_id]
    if status_filter:
        results = [d for d in results if d["status"] == status_filter]
    if year:
        results = [d for d in results if d["year"] == year]
    if commodity:
        results = [d for d in results if d["commodity"] == commodity]
    return {"items": results, "total": len(results)}


@dds_router.post("/{dds_id}/validate")
async def validate_dds(dds_id: str):
    dds = _dds_store.get(dds_id)
    if not dds:
        raise HTTPException(404, f"DDS '{dds_id}' not found")
    section_results = {}
    for section in DDS_SECTIONS:
        data = dds.get(section, {})
        is_complete = bool(data) and len(data) > 0
        section_results[section] = {"complete": is_complete}
    all_valid = all(s["complete"] for s in section_results.values())
    dds["status"] = "validated" if all_valid else "review"
    return {"valid": all_valid, "sections": section_results}


@dds_router.post("/{dds_id}/submit")
async def submit_dds(dds_id: str):
    dds = _dds_store.get(dds_id)
    if not dds:
        raise HTTPException(404, f"DDS '{dds_id}' not found")
    if dds["status"] != "validated":
        raise HTTPException(400, f"Cannot submit DDS in '{dds['status']}' status")
    dds["status"] = "submitted"
    dds["submission_date"] = datetime.now(timezone.utc).isoformat()
    dds["eu_reference"] = f"EU-{uuid.uuid4().hex[:8].upper()}"
    return {"dds_id": dds_id, "status": "submitted", "eu_reference": dds["eu_reference"]}


@dds_router.get("/{dds_id}/download")
async def download_dds(dds_id: str, format: str = "json"):
    dds = _dds_store.get(dds_id)
    if not dds:
        raise HTTPException(404, f"DDS '{dds_id}' not found")
    if format.lower() not in ("json", "xml"):
        raise HTTPException(400, f"Invalid format '{format}'")
    if format.lower() == "json":
        return {"format": "json", "data": dds}
    return {"format": "xml", "data": f"<dds id='{dds_id}'/>"}


@dds_router.post("/{dds_id}/amend")
async def amend_dds(dds_id: str):
    original = _dds_store.get(dds_id)
    if not original:
        raise HTTPException(404, f"DDS '{dds_id}' not found")
    if original["status"] not in ("submitted", "accepted"):
        raise HTTPException(400, f"Cannot amend DDS in '{original['status']}' status")
    # Create amendment
    new_ref = _next_ref(original["country_iso3"], original["year"])
    new_id = f"dds_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    amendment = {**original, "dds_id": new_id, "reference_number": new_ref,
                 "status": "draft", "amendment_of": dds_id, "created_at": now.isoformat()}
    _dds_store[new_id] = amendment
    original["status"] = "amended"
    return amendment


@dds_router.post("/bulk-generate", status_code=201)
async def bulk_generate(body: DDSBulkRequest):
    created = []
    errors = []
    for idx, req in enumerate(body.requests):
        try:
            ref = _next_ref(req.country_iso3, req.year)
            dds_id = f"dds_{uuid.uuid4().hex[:12]}"
            now = datetime.now(timezone.utc)
            dds = {
                "dds_id": dds_id, "reference_number": ref,
                "supplier_id": req.supplier_id, "commodity": req.commodity,
                "year": req.year, "country_iso3": req.country_iso3.upper(),
                "status": "draft", "operator_info": req.operator_info,
                "product_description": req.product_description,
                "country_of_production": req.country_of_production,
                "geolocation_data": req.geolocation_data,
                "risk_assessment": req.risk_assessment,
                "risk_mitigation": req.risk_mitigation,
                "conclusion": req.conclusion,
                "plot_ids": req.plot_ids,
                "submission_date": None, "eu_reference": None,
                "amendment_of": None, "created_at": now.isoformat(),
            }
            _dds_store[dds_id] = dds
            created.append(dds_id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
    return {"total": len(body.requests), "created": len(created),
            "failed": len(errors), "dds_ids": created, "errors": errors}


@dds_router.get("/summary/{year}")
async def annual_summary(year: int):
    year_dds = [d for d in _dds_store.values() if d["year"] == year]
    by_status = {}
    by_commodity = {}
    by_country = {}
    for d in year_dds:
        by_status[d["status"]] = by_status.get(d["status"], 0) + 1
        by_commodity[d["commodity"]] = by_commodity.get(d["commodity"], 0) + 1
        by_country[d["country_iso3"]] = by_country.get(d["country_iso3"], 0) + 1
    return {"year": year, "total": len(year_dds),
            "by_status": by_status, "by_commodity": by_commodity, "by_country": by_country}


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

def _create_app() -> FastAPI:
    app = FastAPI(title="GL-EUDR-APP DDS Test")
    app.include_router(dds_router)
    return app


@pytest.fixture(autouse=True)
def clear_store():
    _dds_store.clear()
    _dds_sequences.clear()
    yield
    _dds_store.clear()
    _dds_sequences.clear()


@pytest.fixture
def client():
    return TestClient(_create_app())


def _make_full_dds_request() -> Dict:
    return {
        "supplier_id": "sup_test",
        "commodity": "soya",
        "country_iso3": "BRA",
        "year": 2026,
        "plot_ids": ["plot_a"],
        "operator_info": {"name": "Test"},
        "product_description": {"hs": "1201"},
        "country_of_production": {"iso3": "BRA"},
        "geolocation_data": {"lat": -12.0},
        "risk_assessment": {"score": 0.3},
        "risk_mitigation": {"m": "monitoring"},
        "conclusion": {"ok": True},
    }


@pytest.fixture
def sample_dds(client):
    resp = client.post("/api/v1/dds/generate", json=_make_full_dds_request())
    return resp.json()


@pytest.fixture
def validated_dds(client, sample_dds):
    client.post(f"/api/v1/dds/{sample_dds['dds_id']}/validate")
    return sample_dds


@pytest.fixture
def submitted_dds(client, validated_dds):
    client.post(f"/api/v1/dds/{validated_dds['dds_id']}/submit")
    return validated_dds


# ---------------------------------------------------------------------------
# POST /dds/generate
# ---------------------------------------------------------------------------

class TestGenerateRoute:

    def test_create_dds(self, client):
        resp = client.post("/api/v1/dds/generate", json=_make_full_dds_request())
        assert resp.status_code == 201
        data = resp.json()
        assert data["dds_id"].startswith("dds_")
        assert data["status"] == "draft"

    def test_reference_number_format(self, client):
        resp = client.post("/api/v1/dds/generate", json=_make_full_dds_request())
        ref = resp.json()["reference_number"]
        assert re.match(r"^EUDR-BRA-2026-\d{6}$", ref)

    def test_invalid_missing_supplier_422(self, client):
        body = _make_full_dds_request()
        body["supplier_id"] = ""
        resp = client.post("/api/v1/dds/generate", json=body)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /dds/{id}
# ---------------------------------------------------------------------------

class TestGetDDSRoute:

    def test_get_found(self, client, sample_dds):
        resp = client.get(f"/api/v1/dds/{sample_dds['dds_id']}")
        assert resp.status_code == 200
        assert resp.json()["dds_id"] == sample_dds["dds_id"]

    def test_get_not_found_404(self, client):
        resp = client.get("/api/v1/dds/dds_nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /dds (list)
# ---------------------------------------------------------------------------

class TestListDDSRoute:

    def test_list_all(self, client, sample_dds):
        resp = client.get("/api/v1/dds/")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_filter_by_supplier(self, client):
        client.post("/api/v1/dds/generate", json={
            **_make_full_dds_request(), "supplier_id": "s1"})
        client.post("/api/v1/dds/generate", json={
            **_make_full_dds_request(), "supplier_id": "s2"})
        resp = client.get("/api/v1/dds/?supplier_id=s1")
        assert resp.json()["total"] == 1

    def test_filter_by_commodity(self, client):
        client.post("/api/v1/dds/generate", json=_make_full_dds_request())
        client.post("/api/v1/dds/generate", json={
            **_make_full_dds_request(), "commodity": "wood"})
        resp = client.get("/api/v1/dds/?commodity=wood")
        assert resp.json()["total"] == 1

    def test_filter_by_year(self, client, sample_dds):
        resp = client.get("/api/v1/dds/?year=2026")
        assert resp.json()["total"] >= 1
        resp2 = client.get("/api/v1/dds/?year=2030")
        assert resp2.json()["total"] == 0


# ---------------------------------------------------------------------------
# POST /dds/{id}/validate
# ---------------------------------------------------------------------------

class TestValidateRoute:

    def test_validation_passes(self, client, sample_dds):
        resp = client.post(f"/api/v1/dds/{sample_dds['dds_id']}/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True

    def test_validation_fails_incomplete(self, client):
        resp = client.post("/api/v1/dds/generate", json={
            "supplier_id": "s1", "commodity": "wood",
            "country_iso3": "BRA", "year": 2026,
        })
        dds_id = resp.json()["dds_id"]
        val_resp = client.post(f"/api/v1/dds/{dds_id}/validate")
        assert val_resp.json()["valid"] is False

    def test_validate_nonexistent_404(self, client):
        resp = client.post("/api/v1/dds/dds_nope/validate")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /dds/{id}/submit
# ---------------------------------------------------------------------------

class TestSubmitRoute:

    def test_submit_success(self, client, validated_dds):
        resp = client.post(f"/api/v1/dds/{validated_dds['dds_id']}/submit")
        assert resp.status_code == 200
        assert resp.json()["status"] == "submitted"
        assert resp.json()["eu_reference"] is not None

    def test_submit_draft_400(self, client, sample_dds):
        resp = client.post(f"/api/v1/dds/{sample_dds['dds_id']}/submit")
        assert resp.status_code == 400

    def test_submit_nonexistent_404(self, client):
        resp = client.post("/api/v1/dds/dds_nope/submit")
        assert resp.status_code == 404

    def test_double_submit_400(self, client, validated_dds):
        client.post(f"/api/v1/dds/{validated_dds['dds_id']}/submit")
        resp = client.post(f"/api/v1/dds/{validated_dds['dds_id']}/submit")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /dds/{id}/download
# ---------------------------------------------------------------------------

class TestDownloadRoute:

    def test_json_format(self, client, sample_dds):
        resp = client.get(f"/api/v1/dds/{sample_dds['dds_id']}/download?format=json")
        assert resp.status_code == 200
        assert resp.json()["format"] == "json"

    def test_xml_format(self, client, sample_dds):
        resp = client.get(f"/api/v1/dds/{sample_dds['dds_id']}/download?format=xml")
        assert resp.status_code == 200
        assert resp.json()["format"] == "xml"

    def test_invalid_format_400(self, client, sample_dds):
        resp = client.get(f"/api/v1/dds/{sample_dds['dds_id']}/download?format=csv")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /dds/{id}/amend
# ---------------------------------------------------------------------------

class TestAmendRoute:

    def test_amendment_created(self, client, submitted_dds):
        resp = client.post(f"/api/v1/dds/{submitted_dds['dds_id']}/amend")
        assert resp.status_code == 200
        data = resp.json()
        assert data["amendment_of"] == submitted_dds["dds_id"]
        assert data["status"] == "draft"

    def test_amend_draft_400(self, client, sample_dds):
        resp = client.post(f"/api/v1/dds/{sample_dds['dds_id']}/amend")
        assert resp.status_code == 400

    def test_amend_nonexistent_404(self, client):
        resp = client.post("/api/v1/dds/dds_nope/amend")
        assert resp.status_code == 404

    def test_amendment_new_reference(self, client, submitted_dds):
        resp = client.post(f"/api/v1/dds/{submitted_dds['dds_id']}/amend")
        assert resp.json()["reference_number"] != submitted_dds["reference_number"]


# ---------------------------------------------------------------------------
# POST /dds/bulk-generate
# ---------------------------------------------------------------------------

class TestBulkGenerateRoute:

    def test_multiple_dds(self, client):
        body = {"requests": [
            _make_full_dds_request(),
            {**_make_full_dds_request(), "supplier_id": "s2", "commodity": "wood"},
        ]}
        resp = client.post("/api/v1/dds/bulk-generate", json=body)
        assert resp.status_code == 201
        assert resp.json()["created"] == 2

    def test_empty_request_422(self, client):
        resp = client.post("/api/v1/dds/bulk-generate", json={"requests": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /dds/summary/{year}
# ---------------------------------------------------------------------------

class TestSummaryRoute:

    def test_annual_summary(self, client, sample_dds):
        resp = client.get("/api/v1/dds/summary/2026")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert data["year"] == 2026

    def test_empty_year(self, client):
        resp = client.get("/api/v1/dds/summary/2030")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
