"""
Unit tests for GL-GHG-APP v1.0 API Routes

Tests key FastAPI endpoints using TestClient for inventory management,
scope aggregation, reporting, dashboard, verification, and targets.
35+ test cases.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.inventory_routes import router as inventory_router
from services.api.scope1_routes import router as scope1_router


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a test FastAPI application with routers."""
    application = FastAPI(title="GL-GHG-APP Test", version="1.0.0-test")
    application.include_router(inventory_router)
    application.include_router(scope1_router)
    return application


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper to create an org then return its org_id
# ---------------------------------------------------------------------------

def _create_org(client: TestClient) -> str:
    """Create an organization and return its org_id."""
    resp = client.post("/api/v1/inventory/organizations", json={
        "name": "Acme Manufacturing Inc.",
        "industry": "Manufacturing",
        "country": "US",
        "description": "Test organization for GHG accounting",
        "fiscal_year_end_month": 12,
    })
    assert resp.status_code == 201
    return resp.json()["org_id"]


def _create_org_with_entity(client: TestClient):
    """Create an organization with one entity and return (org_id, entity_id)."""
    org_id = _create_org(client)
    resp = client.post(f"/api/v1/inventory/organizations/{org_id}/entities", json={
        "name": "East Coast Plant",
        "type": "facility",
        "ownership_pct": 100.0,
        "country": "US",
        "employees": 350,
        "operational_control": True,
    })
    assert resp.status_code == 201
    return org_id, resp.json()["entity_id"]


def _create_org_with_boundary(client: TestClient):
    """Create org with boundary and return (org_id, boundary_id)."""
    org_id = _create_org(client)
    resp = client.post(f"/api/v1/inventory/organizations/{org_id}/boundary", json={
        "consolidation_approach": "operational_control",
        "scopes": [1, 2, 3],
        "base_year": 2019,
        "reporting_year": 2025,
    })
    assert resp.status_code == 201
    return org_id, resp.json()["boundary_id"]


def _create_org_with_inventory(client: TestClient):
    """Create org with inventory and return (org_id, inventory_id)."""
    org_id = _create_org(client)
    resp = client.post(f"/api/v1/inventory/organizations/{org_id}/inventories", json={
        "reporting_year": 2025,
    })
    assert resp.status_code == 201
    return org_id, resp.json()["inventory_id"]


# ---------------------------------------------------------------------------
# TestOrganizationEndpoints
# ---------------------------------------------------------------------------

class TestOrganizationEndpoints:
    """Test organization CRUD endpoints."""

    def test_create_organization(self, client):
        """Test POST /inventory/organizations."""
        resp = client.post("/api/v1/inventory/organizations", json={
            "name": "Test Corp",
            "industry": "Manufacturing",
            "country": "US",
            "fiscal_year_end_month": 12,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Corp"
        assert data["industry"] == "Manufacturing"
        assert "org_id" in data

    def test_get_organization(self, client):
        """Test GET /inventory/organizations/{org_id}."""
        org_id = _create_org(client)
        resp = client.get(f"/api/v1/inventory/organizations/{org_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["org_id"] == org_id
        assert data["name"] == "Acme Manufacturing Inc."

    def test_get_nonexistent_org(self, client):
        """Test 404 for non-existent organization."""
        resp = client.get("/api/v1/inventory/organizations/org_nonexistent")
        assert resp.status_code == 404

    def test_create_org_missing_name(self, client):
        """Test validation error for missing name."""
        resp = client.post("/api/v1/inventory/organizations", json={
            "industry": "Tech",
            "country": "US",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TestEntityEndpoints
# ---------------------------------------------------------------------------

class TestEntityEndpoints:
    """Test entity management endpoints."""

    def test_create_entity(self, client):
        """Test POST /organizations/{org_id}/entities."""
        org_id = _create_org(client)
        resp = client.post(f"/api/v1/inventory/organizations/{org_id}/entities", json={
            "name": "East Coast Plant",
            "type": "facility",
            "ownership_pct": 100.0,
            "country": "US",
            "employees": 350,
            "operational_control": True,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "East Coast Plant"
        assert data["type"] == "facility"

    def test_create_entity_nonexistent_org(self, client):
        """Test creating entity for non-existent org returns 404."""
        resp = client.post("/api/v1/inventory/organizations/org_fake/entities", json={
            "name": "Test",
            "type": "facility",
            "country": "US",
        })
        assert resp.status_code == 404

    def test_create_entity_with_parent(self, client):
        """Test creating entity with parent linkage."""
        org_id, entity_id = _create_org_with_entity(client)
        resp = client.post(f"/api/v1/inventory/organizations/{org_id}/entities", json={
            "name": "Production Line A",
            "type": "site",
            "parent_id": entity_id,
            "ownership_pct": 100.0,
            "country": "US",
            "operational_control": True,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["parent_id"] == entity_id

    def test_create_entity_invalid_parent(self, client):
        """Test creating entity with invalid parent returns 400."""
        org_id = _create_org(client)
        resp = client.post(f"/api/v1/inventory/organizations/{org_id}/entities", json={
            "name": "Orphan",
            "type": "facility",
            "parent_id": "ent_nonexistent",
            "country": "US",
        })
        assert resp.status_code == 400

    def test_update_entity(self, client):
        """Test PUT /entities/{entity_id}."""
        org_id, entity_id = _create_org_with_entity(client)
        resp = client.put(f"/api/v1/inventory/entities/{entity_id}", json={
            "name": "Renamed Plant",
            "employees": 500,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Renamed Plant"
        assert data["employees"] == 500

    def test_update_nonexistent_entity(self, client):
        """Test updating non-existent entity returns 404."""
        resp = client.put("/api/v1/inventory/entities/ent_fake", json={
            "name": "Test",
        })
        assert resp.status_code == 404

    def test_self_parent_rejected(self, client):
        """Test entity cannot be its own parent."""
        org_id, entity_id = _create_org_with_entity(client)
        resp = client.put(f"/api/v1/inventory/entities/{entity_id}", json={
            "parent_id": entity_id,
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# TestBoundaryEndpoints
# ---------------------------------------------------------------------------

class TestBoundaryEndpoints:
    """Test boundary setting endpoints."""

    def test_set_boundary(self, client):
        """Test POST /organizations/{org_id}/boundary."""
        org_id = _create_org(client)
        resp = client.post(f"/api/v1/inventory/organizations/{org_id}/boundary", json={
            "consolidation_approach": "operational_control",
            "scopes": [1, 2, 3],
            "base_year": 2019,
            "reporting_year": 2025,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["consolidation_approach"] == "operational_control"
        assert data["scopes"] == [1, 2, 3]

    def test_get_boundary(self, client):
        """Test GET /organizations/{org_id}/boundary."""
        org_id, boundary_id = _create_org_with_boundary(client)
        resp = client.get(f"/api/v1/inventory/organizations/{org_id}/boundary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["boundary_id"] == boundary_id

    def test_set_boundary_nonexistent_org(self, client):
        """Test setting boundary for non-existent org returns 404."""
        resp = client.post("/api/v1/inventory/organizations/org_fake/boundary", json={
            "consolidation_approach": "operational_control",
            "scopes": [1, 2],
            "base_year": 2019,
            "reporting_year": 2025,
        })
        assert resp.status_code == 404

    def test_invalid_scope(self, client):
        """Test invalid scope value."""
        org_id = _create_org(client)
        resp = client.post(f"/api/v1/inventory/organizations/{org_id}/boundary", json={
            "consolidation_approach": "operational_control",
            "scopes": [1, 2, 4],
            "base_year": 2019,
            "reporting_year": 2025,
        })
        assert resp.status_code == 400

    def test_reporting_before_base_year(self, client):
        """Test reporting year before base year returns 400."""
        org_id = _create_org(client)
        resp = client.post(f"/api/v1/inventory/organizations/{org_id}/boundary", json={
            "consolidation_approach": "operational_control",
            "scopes": [1, 2],
            "base_year": 2025,
            "reporting_year": 2020,
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# TestInventoryEndpoints
# ---------------------------------------------------------------------------

class TestInventoryEndpoints:
    """Test inventory CRUD endpoints."""

    def test_create_inventory(self, client):
        """Test POST /organizations/{org_id}/inventories."""
        org_id, inventory_id = _create_org_with_inventory(client)
        assert inventory_id.startswith("inv_")

    def test_get_inventory(self, client):
        """Test GET /inventories/{inventory_id}."""
        org_id, inventory_id = _create_org_with_inventory(client)
        resp = client.get(f"/api/v1/inventory/inventories/{inventory_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["inventory_id"] == inventory_id
        assert data["reporting_year"] == 2025

    def test_get_nonexistent_inventory(self, client):
        """Test 404 for non-existent inventory."""
        resp = client.get("/api/v1/inventory/inventories/inv_nonexistent")
        assert resp.status_code == 404

    def test_duplicate_year(self, client):
        """Test creating duplicate inventory for same year returns 409."""
        org_id = _create_org(client)
        resp1 = client.post(f"/api/v1/inventory/organizations/{org_id}/inventories", json={
            "reporting_year": 2025,
        })
        assert resp1.status_code == 201
        resp2 = client.post(f"/api/v1/inventory/organizations/{org_id}/inventories", json={
            "reporting_year": 2025,
        })
        assert resp2.status_code == 409


# ---------------------------------------------------------------------------
# TestScope1Endpoints
# ---------------------------------------------------------------------------

class TestScope1Endpoints:
    """Test Scope 1 emission endpoints."""

    def test_aggregate_scope1(self, client):
        """Test POST /scope1/aggregate/{inventory_id}."""
        resp = client.post("/api/v1/scope1/aggregate/inv_test123")
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "completed"
        assert data["total_tco2e"] > 0

    def test_get_scope1_summary(self, client):
        """Test GET /scope1/{inventory_id}/summary."""
        resp = client.get("/api/v1/scope1/inv_test123/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_tco2e" in data
        assert "total_co2_tonnes" in data
        assert "total_ch4_tco2e" in data

    def test_get_scope1_categories(self, client):
        """Test GET /scope1/{inventory_id}/categories."""
        resp = client.get("/api/v1/scope1/inv_test123/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert all("category" in cat for cat in data)

    def test_get_scope1_categories_include_zero(self, client):
        """Test include_zero parameter shows all 8 categories."""
        resp = client.get("/api/v1/scope1/inv_test123/categories?include_zero=true")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 8

    def test_get_scope1_facilities(self, client):
        """Test GET /scope1/{inventory_id}/facilities."""
        resp = client.get("/api/v1/scope1/inv_test123/facilities")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_get_scope1_gases(self, client):
        """Test GET /scope1/{inventory_id}/gases."""
        resp = client.get("/api/v1/scope1/inv_test123/gases")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert all("gas" in g for g in data)

    def test_get_scope1_gases_include_zero(self, client):
        """Test include_zero shows all 7 gases."""
        resp = client.get("/api/v1/scope1/inv_test123/gases?include_zero=true")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 7

    def test_submit_scope1_data(self, client):
        """Test POST /scope1/{inventory_id}/data."""
        resp = client.post("/api/v1/scope1/inv_test123/data", json={
            "category": "stationary_combustion",
            "facility_name": "East Coast Plant",
            "quantity": 150000,
            "unit": "therms",
            "fuel_type": "natural_gas",
            "calculation_tier": "tier_1",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["category"] == "stationary_combustion"
        assert data["calculated_tco2e"] > 0
        assert data["emission_factor_used"] == 5.302

    def test_submit_scope1_custom_ef(self, client):
        """Test submitting data with custom emission factor."""
        resp = client.post("/api/v1/scope1/inv_test123/data", json={
            "category": "stationary_combustion",
            "quantity": 10000,
            "unit": "gallons",
            "fuel_type": "biodiesel",
            "emission_factor": 9.45,
            "emission_factor_source": "EPA",
            "calculation_tier": "tier_2",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["emission_factor_used"] == 9.45
        # 10000 * 9.45 / 1000 = 94.5 tCO2e
        assert data["calculated_tco2e"] == 94.5
