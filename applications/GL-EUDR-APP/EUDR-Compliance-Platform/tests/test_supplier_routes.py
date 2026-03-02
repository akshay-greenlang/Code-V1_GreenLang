"""
Unit tests for GL-EUDR-APP v1.0 Supplier API Routes.

Tests all supplier REST endpoints using FastAPI TestClient:
POST, GET, PUT, DELETE, bulk import, compliance, risk.

Test count target: 30+ tests
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.supplier_routes import router, _suppliers


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

def _create_app() -> FastAPI:
    app = FastAPI(title="GL-EUDR-APP Test")
    app.include_router(router)
    return app


@pytest.fixture(autouse=True)
def clear_store():
    """Clear in-memory store before each test."""
    _suppliers.clear()
    yield
    _suppliers.clear()


@pytest.fixture
def client():
    app = _create_app()
    return TestClient(app)


@pytest.fixture
def sample_supplier(client):
    """Create and return a sample supplier via the API."""
    resp = client.post("/api/v1/suppliers/", json={
        "name": "Amazonia Timber Co.",
        "country": "BR",
        "tax_id": "12.345.678/0001-90",
        "commodities": ["timber"],
    })
    # If timber is invalid (it should be "wood"), adjust:
    if resp.status_code == 422:
        resp = client.post("/api/v1/suppliers/", json={
            "name": "Amazonia Timber Co.",
            "country": "BR",
            "tax_id": "12.345.678/0001-90",
            "commodities": ["wood"],
        })
    return resp.json()


# ---------------------------------------------------------------------------
# POST /suppliers
# ---------------------------------------------------------------------------

class TestCreateSupplierRoute:

    def test_create_valid(self, client):
        resp = client.post("/api/v1/suppliers/", json={
            "name": "Test Corp",
            "country": "BR",
            "commodities": ["soya", "wood"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Corp"
        assert data["compliance_status"] == "pending"
        assert data["supplier_id"].startswith("sup_")

    def test_create_invalid_data_422(self, client):
        resp = client.post("/api/v1/suppliers/", json={
            "name": "",  # empty name violates min_length
            "country": "BR",
            "commodities": ["soya"],
        })
        assert resp.status_code == 422

    def test_create_invalid_commodity_422(self, client):
        resp = client.post("/api/v1/suppliers/", json={
            "name": "Bad Corp",
            "country": "BR",
            "commodities": ["bananas"],
        })
        assert resp.status_code == 422

    def test_create_with_address(self, client):
        resp = client.post("/api/v1/suppliers/", json={
            "name": "With Address Corp",
            "country": "BR",
            "commodities": ["cocoa"],
            "address": {"city": "Manaus", "country": "BR"},
        })
        assert resp.status_code == 201
        assert resp.json()["address"]["city"] == "Manaus"

    def test_create_missing_commodities_422(self, client):
        resp = client.post("/api/v1/suppliers/", json={
            "name": "No Commodities",
            "country": "BR",
            "commodities": [],
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /suppliers/{id}
# ---------------------------------------------------------------------------

class TestGetSupplierRoute:

    def test_get_found(self, client, sample_supplier):
        sid = sample_supplier["supplier_id"]
        resp = client.get(f"/api/v1/suppliers/{sid}")
        assert resp.status_code == 200
        assert resp.json()["supplier_id"] == sid

    def test_get_not_found_404(self, client):
        resp = client.get("/api/v1/suppliers/sup_nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /suppliers/{id}
# ---------------------------------------------------------------------------

class TestUpdateSupplierRoute:

    def test_update_valid(self, client, sample_supplier):
        sid = sample_supplier["supplier_id"]
        resp = client.put(f"/api/v1/suppliers/{sid}", json={
            "name": "Updated Name",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"

    def test_update_not_found_404(self, client):
        resp = client.put("/api/v1/suppliers/sup_nonexistent", json={
            "name": "Ghost",
        })
        assert resp.status_code == 404

    def test_update_commodities(self, client, sample_supplier):
        sid = sample_supplier["supplier_id"]
        resp = client.put(f"/api/v1/suppliers/{sid}", json={
            "commodities": ["rubber", "coffee"],
        })
        assert resp.status_code == 200
        assert set(resp.json()["commodities"]) == {"rubber", "coffee"}


# ---------------------------------------------------------------------------
# GET /suppliers (list)
# ---------------------------------------------------------------------------

class TestListSuppliersRoute:

    def test_list_with_pagination(self, client):
        for i in range(5):
            client.post("/api/v1/suppliers/", json={
                "name": f"Supplier {i}",
                "country": "BR",
                "commodities": ["wood"],
            })
        resp = client.get("/api/v1/suppliers/?page=1&limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 3
        assert data["total"] == 5
        assert data["total_pages"] == 2

    def test_filter_by_country(self, client):
        client.post("/api/v1/suppliers/", json={
            "name": "Brazil Co", "country": "BR", "commodities": ["wood"],
        })
        client.post("/api/v1/suppliers/", json={
            "name": "German Co", "country": "DE", "commodities": ["cocoa"],
        })
        resp = client.get("/api/v1/suppliers/?country=BR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["country"] == "BR"

    def test_filter_by_commodity(self, client):
        client.post("/api/v1/suppliers/", json={
            "name": "Soya Co", "country": "BR", "commodities": ["soya"],
        })
        client.post("/api/v1/suppliers/", json={
            "name": "Wood Co", "country": "BR", "commodities": ["wood"],
        })
        resp = client.get("/api/v1/suppliers/?commodity=soya")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_search_by_name(self, client):
        client.post("/api/v1/suppliers/", json={
            "name": "Amazonia Corp", "country": "BR", "commodities": ["wood"],
        })
        client.post("/api/v1/suppliers/", json={
            "name": "Other Co", "country": "BR", "commodities": ["wood"],
        })
        resp = client.get("/api/v1/suppliers/?search=amazonia")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_empty_list(self, client):
        resp = client.get("/api/v1/suppliers/")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# POST /suppliers/bulk-import
# ---------------------------------------------------------------------------

class TestBulkImportRoute:

    def test_valid_records(self, client):
        resp = client.post("/api/v1/suppliers/bulk-import", json={
            "records": [
                {"name": "A", "country": "BR", "commodities": ["wood"]},
                {"name": "B", "country": "ID", "commodities": ["oil_palm"]},
            ]
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_submitted"] == 2
        assert data["total_created"] == 2

    def test_empty_list_rejected(self, client):
        resp = client.post("/api/v1/suppliers/bulk-import", json={
            "records": []
        })
        # min_length=1 in BulkImportRequest should cause 422
        assert resp.status_code == 422

    def test_mixed_valid_invalid(self, client):
        resp = client.post("/api/v1/suppliers/bulk-import", json={
            "records": [
                {"name": "Valid", "country": "BR", "commodities": ["wood"]},
                {"name": "Invalid", "country": "BR", "commodities": ["bananas"]},
            ]
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_created"] == 1
        assert data["total_failed"] == 1


# ---------------------------------------------------------------------------
# GET /suppliers/{id}/compliance
# ---------------------------------------------------------------------------

class TestComplianceRoute:

    def test_compliance_status(self, client, sample_supplier):
        sid = sample_supplier["supplier_id"]
        resp = client.get(f"/api/v1/suppliers/{sid}/compliance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["compliance_status"] == "pending"
        assert len(data["issues"]) > 0

    def test_compliance_not_found(self, client):
        resp = client.get("/api/v1/suppliers/sup_nonexistent/compliance")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /suppliers/{id}/risk
# ---------------------------------------------------------------------------

class TestRiskRoute:

    def test_risk_summary(self, client, sample_supplier):
        sid = sample_supplier["supplier_id"]
        resp = client.get(f"/api/v1/suppliers/{sid}/risk")
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["risk_score"] <= 100
        assert data["overall_risk_level"] in ("low", "medium", "high", "critical")

    def test_risk_not_found(self, client):
        resp = client.get("/api/v1/suppliers/sup_nonexistent/risk")
        assert resp.status_code == 404

    def test_high_risk_country_score(self, client):
        resp = client.post("/api/v1/suppliers/", json={
            "name": "Brazil Soy Co",
            "country": "BR",
            "commodities": ["soya", "cattle"],
        })
        sid = resp.json()["supplier_id"]
        risk_resp = client.get(f"/api/v1/suppliers/{sid}/risk")
        data = risk_resp.json()
        # BR is high-risk, soya/cattle are high commodities
        assert data["country_risk"] == "high"
        assert data["risk_score"] >= 50


# ---------------------------------------------------------------------------
# DELETE /suppliers/{id}
# ---------------------------------------------------------------------------

class TestDeleteSupplierRoute:

    def test_delete_existing(self, client, sample_supplier):
        sid = sample_supplier["supplier_id"]
        resp = client.delete(f"/api/v1/suppliers/{sid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        # Verify it is gone
        get_resp = client.get(f"/api/v1/suppliers/{sid}")
        assert get_resp.status_code == 404

    def test_delete_not_found_404(self, client):
        resp = client.delete("/api/v1/suppliers/sup_nonexistent")
        assert resp.status_code == 404
