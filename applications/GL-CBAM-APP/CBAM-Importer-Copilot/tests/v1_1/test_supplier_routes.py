# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Supplier Portal API Routes

Tests supplier_portal.api.supplier_routes FastAPI endpoints:
- Register supplier (valid, invalid EORI, duplicate)
- Get supplier profile / not found
- Update supplier profile
- Add installation / list installations
- Submit emissions data (valid, invalid)
- Get submission status / history
- Link supplier to importer
- Get supplier dashboard
- Search suppliers
- Data quality score
- Supplier verification status update
- Bulk import / export suppliers
- Upcoming deadlines

Uses FastAPI TestClient for synchronous HTTP testing.

Target: 80+ tests
"""

import pytest
import json
import hashlib
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from supplier_portal.api.supplier_routes import (
    router,
    _suppliers,
    _installations,
    _submissions,
    _access_requests,
    _audit_log,
)


# ===========================================================================
# App & Client Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def clear_stores():
    """Clear in-memory stores before each test."""
    _suppliers.clear()
    _installations.clear()
    _submissions.clear()
    _access_requests.clear()
    _audit_log.clear()
    yield
    _suppliers.clear()
    _installations.clear()
    _submissions.clear()
    _access_requests.clear()
    _audit_log.clear()


@pytest.fixture
def app():
    """Create a FastAPI app with the supplier router."""
    application = FastAPI()
    application.include_router(router)
    return application


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def valid_supplier_payload():
    """Valid supplier registration payload."""
    return {
        "company_name": "Steel Works Istanbul",
        "country_iso": "TR",
        "eori_number": "TR1234567890",
        "tax_id": "VKN-1234567",
        "product_groups": ["steel"],
        "cn_codes_produced": ["72031000"],
        "address": {
            "street": "Industrial Ave 42",
            "city": "Istanbul",
            "region": "Marmara",
            "postal_code": "34000",
            "country": "Turkey",
        },
        "contact": {
            "person": "Ahmet Yilmaz",
            "email": "ahmet@steelworks.tr",
            "phone": "+905551234567",
        },
        "certifications": ["ISO 14001"],
        "production_capacity_tons_per_year": 50000.0,
        "notes": "Primary steel supplier",
    }


@pytest.fixture
def registered_supplier(client, valid_supplier_payload):
    """Register a supplier and return the response data."""
    resp = client.post(
        "/api/v1/cbam/suppliers/register",
        json=valid_supplier_payload,
    )
    assert resp.status_code == 201
    return resp.json()


# ===========================================================================
# TEST CLASS -- Register supplier
# ===========================================================================

class TestRegisterSupplier:
    """Tests for POST /register."""

    def test_register_supplier_success(self, client, valid_supplier_payload):
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=valid_supplier_payload,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["company_name"] == "Steel Works Istanbul"
        assert data["country_iso"] == "TR"
        assert data["supplier_id"].startswith("SUP-")
        assert data["verification_status"] == "pending"

    def test_register_supplier_minimal(self, client):
        payload = {
            "company_name": "MinimalCo",
            "country_iso": "CN",
            "product_groups": ["steel"],
            "contact": {
                "person": "Test Person",
                "email": "test@example.com",
            },
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 201

    def test_register_supplier_invalid_eori(self, client):
        payload = {
            "company_name": "BadEORI Corp",
            "country_iso": "TR",
            "eori_number": "invalid-eori!",
            "product_groups": ["steel"],
            "contact": {"person": "Test", "email": "test@test.com"},
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 422

    def test_register_supplier_invalid_country(self, client):
        payload = {
            "company_name": "BadCountry Corp",
            "country_iso": "xx",  # lowercase
            "product_groups": ["steel"],
            "contact": {"person": "Test", "email": "test@test.com"},
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 422

    def test_register_supplier_invalid_product_group(self, client):
        payload = {
            "company_name": "InvalidProduct Corp",
            "country_iso": "TR",
            "product_groups": ["widgets"],  # Invalid
            "contact": {"person": "Test", "email": "test@test.com"},
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 422

    def test_register_supplier_duplicate_eori(
        self, client, valid_supplier_payload, registered_supplier
    ):
        # Try to register again with same EORI
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=valid_supplier_payload,
        )
        assert resp.status_code == 409

    def test_register_supplier_has_provenance(self, registered_supplier):
        assert "provenance" in registered_supplier
        assert len(registered_supplier["provenance"]["provenance_hash"]) == 64

    def test_register_supplier_has_created_at(self, registered_supplier):
        assert "created_at" in registered_supplier

    def test_register_supplier_multiple_product_groups(self, client):
        payload = {
            "company_name": "MultiProduct Corp",
            "country_iso": "CN",
            "product_groups": ["steel", "cement", "aluminum"],
            "contact": {"person": "Test", "email": "test@test.com"},
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 201
        assert len(resp.json()["product_groups"]) == 3

    def test_register_supplier_invalid_cn_code(self, client):
        payload = {
            "company_name": "BadCN Corp",
            "country_iso": "TR",
            "product_groups": ["steel"],
            "cn_codes_produced": ["1234"],  # Not 8 digits
            "contact": {"person": "Test", "email": "test@test.com"},
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 422


# ===========================================================================
# TEST CLASS -- Get supplier profile
# ===========================================================================

class TestGetSupplierProfile:
    """Tests for GET /{supplier_id}."""

    def test_get_supplier_profile(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        resp = client.get(f"/api/v1/cbam/suppliers/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["supplier_id"] == sid
        assert data["company_name"] == "Steel Works Istanbul"

    def test_get_supplier_not_found(self, client):
        resp = client.get("/api/v1/cbam/suppliers/SUP-NONEXISTENT")
        assert resp.status_code == 404


# ===========================================================================
# TEST CLASS -- Update supplier
# ===========================================================================

class TestUpdateSupplier:
    """Tests for PUT /{supplier_id}."""

    def test_update_supplier_name(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        resp = client.put(
            f"/api/v1/cbam/suppliers/{sid}",
            json={"company_name": "Updated Steel Works"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["company_name"] == "Updated Steel Works"

    def test_update_supplier_not_found(self, client):
        resp = client.put(
            "/api/v1/cbam/suppliers/SUP-NONEXISTENT",
            json={"company_name": "New Name"},
        )
        assert resp.status_code == 404

    def test_update_supplier_product_groups(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        resp = client.put(
            f"/api/v1/cbam/suppliers/{sid}",
            json={"product_groups": ["steel", "aluminum"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "aluminum" in data["product_groups"]


# ===========================================================================
# TEST CLASS -- Installation management
# ===========================================================================

class TestInstallationManagement:
    """Tests for installation endpoints."""

    def test_add_installation(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        payload = {
            "installation_name": "Istanbul Plant A",
            "country_iso": "TR",
            "address": {
                "city": "Istanbul",
                "country": "Turkey",
            },
            "product_groups": ["steel"],
        }
        resp = client.post(
            f"/api/v1/cbam/suppliers/{sid}/installations",
            json=payload,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["installation_name"] == "Istanbul Plant A"
        assert data["supplier_id"] == sid

    def test_list_installations(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        # Add an installation first
        payload = {
            "installation_name": "Plant B",
            "country_iso": "TR",
            "address": {"city": "Ankara", "country": "Turkey"},
            "product_groups": ["steel"],
        }
        client.post(
            f"/api/v1/cbam/suppliers/{sid}/installations",
            json=payload,
        )
        resp = client.get(f"/api/v1/cbam/suppliers/{sid}/installations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pagination"]["total"] >= 1

    def test_add_installation_supplier_not_found(self, client):
        payload = {
            "installation_name": "Orphan Plant",
            "country_iso": "TR",
            "address": {"city": "Istanbul", "country": "Turkey"},
            "product_groups": ["steel"],
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/SUP-NONEXISTENT/installations",
            json=payload,
        )
        assert resp.status_code == 404

    def test_add_installation_invalid_product_group(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        payload = {
            "installation_name": "Bad Plant",
            "country_iso": "TR",
            "address": {"city": "Istanbul", "country": "Turkey"},
            "product_groups": ["unknown_product"],
        }
        resp = client.post(
            f"/api/v1/cbam/suppliers/{sid}/installations",
            json=payload,
        )
        assert resp.status_code == 422


# ===========================================================================
# TEST CLASS -- Emissions submission
# ===========================================================================

class TestEmissionsSubmission:
    """Tests for emissions submission endpoints."""

    def _create_installation(self, client, supplier_id):
        """Helper to create an installation."""
        payload = {
            "installation_name": "Test Plant",
            "country_iso": "TR",
            "address": {"city": "Istanbul", "country": "Turkey"},
            "product_groups": ["steel"],
        }
        resp = client.post(
            f"/api/v1/cbam/suppliers/{supplier_id}/installations",
            json=payload,
        )
        return resp.json()["installation_id"]

    def test_submit_emissions_data(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        inst_id = self._create_installation(client, sid)
        payload = {
            "installation_id": inst_id,
            "reporting_period": "2026Q1",
            "product_group": "steel",
            "cn_code": "72031000",
            "direct_emissions_tco2_per_ton": 1.5,
            "indirect_emissions_tco2_per_ton": 0.3,
            "production_volume_tons": 5000.0,
            "methodology": "GHG Protocol - Direct measurement",
            "data_quality": "high",
            "data_completeness_pct": 95.0,
        }
        resp = client.post(
            f"/api/v1/cbam/suppliers/{sid}/emissions",
            json=payload,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "submitted"
        assert data["total_emissions_tco2_per_ton"] == pytest.approx(1.8, rel=0.01)

    def test_submit_emissions_invalid_period(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        inst_id = self._create_installation(client, sid)
        payload = {
            "installation_id": inst_id,
            "reporting_period": "2026X5",  # Invalid
            "product_group": "steel",
            "cn_code": "72031000",
            "direct_emissions_tco2_per_ton": 1.5,
            "indirect_emissions_tco2_per_ton": 0.3,
            "production_volume_tons": 5000.0,
            "methodology": "Direct measurement",
            "data_quality": "high",
            "data_completeness_pct": 95.0,
        }
        resp = client.post(
            f"/api/v1/cbam/suppliers/{sid}/emissions",
            json=payload,
        )
        assert resp.status_code == 422

    def test_submit_emissions_invalid_cn_code(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        inst_id = self._create_installation(client, sid)
        payload = {
            "installation_id": inst_id,
            "reporting_period": "2026Q1",
            "product_group": "steel",
            "cn_code": "1234",  # Not 8 digits
            "direct_emissions_tco2_per_ton": 1.5,
            "indirect_emissions_tco2_per_ton": 0.3,
            "production_volume_tons": 5000.0,
            "methodology": "Direct measurement",
            "data_quality": "high",
            "data_completeness_pct": 95.0,
        }
        resp = client.post(
            f"/api/v1/cbam/suppliers/{sid}/emissions",
            json=payload,
        )
        assert resp.status_code == 422


# ===========================================================================
# TEST CLASS -- Get submission status / history
# ===========================================================================

class TestSubmissionStatus:
    """Tests for submission status and history."""

    def _create_submission(self, client, supplier_id):
        """Helper to create an installation and emission submission."""
        payload_inst = {
            "installation_name": "Submission Plant",
            "country_iso": "TR",
            "address": {"city": "Istanbul", "country": "Turkey"},
            "product_groups": ["steel"],
        }
        inst_resp = client.post(
            f"/api/v1/cbam/suppliers/{supplier_id}/installations",
            json=payload_inst,
        )
        inst_id = inst_resp.json()["installation_id"]
        payload = {
            "installation_id": inst_id,
            "reporting_period": "2026Q1",
            "product_group": "steel",
            "cn_code": "72031000",
            "direct_emissions_tco2_per_ton": 1.5,
            "indirect_emissions_tco2_per_ton": 0.3,
            "production_volume_tons": 5000.0,
            "methodology": "Direct measurement",
            "data_quality": "high",
            "data_completeness_pct": 95.0,
        }
        resp = client.post(
            f"/api/v1/cbam/suppliers/{supplier_id}/emissions",
            json=payload,
        )
        return resp.json()

    def test_get_submission_status(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        sub = self._create_submission(client, sid)
        sub_id = sub["submission_id"]
        resp = client.get(
            f"/api/v1/cbam/suppliers/{sid}/emissions/{sub_id}"
        )
        assert resp.status_code == 200

    def test_get_submission_history(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        self._create_submission(client, sid)
        resp = client.get(
            f"/api/v1/cbam/suppliers/{sid}/emissions"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pagination"]["total"] >= 1


# ===========================================================================
# TEST CLASS -- Search suppliers
# ===========================================================================

class TestSearchSuppliers:
    """Tests for GET /search."""

    def test_search_all(self, client, registered_supplier):
        resp = client.get("/api/v1/cbam/suppliers/search")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pagination"]["total"] >= 1

    def test_search_by_country(self, client, registered_supplier):
        resp = client.get("/api/v1/cbam/suppliers/search?country=TR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pagination"]["total"] >= 1

    def test_search_by_sector(self, client, registered_supplier):
        resp = client.get("/api/v1/cbam/suppliers/search?sector=steel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pagination"]["total"] >= 1

    def test_search_by_name(self, client, registered_supplier):
        resp = client.get("/api/v1/cbam/suppliers/search?name=Steel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pagination"]["total"] >= 1

    def test_search_no_results(self, client, registered_supplier):
        resp = client.get("/api/v1/cbam/suppliers/search?country=ZZ")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pagination"]["total"] == 0

    def test_search_pagination(self, client, registered_supplier):
        resp = client.get(
            "/api/v1/cbam/suppliers/search?offset=0&limit=5"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pagination" in data


# ===========================================================================
# TEST CLASS -- Dashboard and data quality
# ===========================================================================

class TestDashboardAndQuality:
    """Tests for dashboard and data quality endpoints."""

    def test_supplier_dashboard(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        resp = client.get(f"/api/v1/cbam/suppliers/{sid}/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["supplier_id"] == sid

    def test_data_quality_score(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        resp = client.get(f"/api/v1/cbam/suppliers/{sid}/data-quality")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_score" in data

    def test_dashboard_not_found(self, client):
        resp = client.get("/api/v1/cbam/suppliers/SUP-NONEXISTENT/dashboard")
        assert resp.status_code == 404


# ===========================================================================
# TEST CLASS -- Verification status
# ===========================================================================

class TestVerificationStatus:
    """Tests for verification status updates."""

    def _create_installation(self, client, supplier_id):
        payload = {
            "installation_name": "Verify Plant",
            "country_iso": "TR",
            "address": {"city": "Istanbul", "country": "Turkey"},
            "product_groups": ["steel"],
        }
        resp = client.post(
            f"/api/v1/cbam/suppliers/{supplier_id}/installations",
            json=payload,
        )
        return resp.json()["installation_id"]

    def test_get_verification_status(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        inst_id = self._create_installation(client, sid)
        resp = client.get(
            f"/api/v1/cbam/suppliers/{sid}/installations/{inst_id}/verification"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verification_status"] == "pending"


# ===========================================================================
# TEST CLASS -- Data exchange (link supplier to importer)
# ===========================================================================

class TestDataExchange:
    """Tests for data exchange / supplier-importer linking."""

    def test_request_access(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        payload = {
            "importer_id": "NL123456789012",
            "importer_name": "EU Import Co",
            "supplier_id": sid,
            "purpose": "CBAM quarterly report Q1 2026",
            "access_duration_days": 365,
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/data-exchange/request-access",
            json=payload,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert data["importer_id"] == "NL123456789012"


# ===========================================================================
# TEST CLASS -- Upcoming deadlines
# ===========================================================================

class TestUpcomingDeadlines:
    """Tests for upcoming deadlines."""

    def test_get_deadlines(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        resp = client.get(f"/api/v1/cbam/suppliers/{sid}/deadlines")
        assert resp.status_code == 200
        data = resp.json()
        assert "deadlines" in data


# ===========================================================================
# TEST CLASS -- Export and audit
# ===========================================================================

class TestExportAndAudit:
    """Tests for export and audit log endpoints."""

    def test_get_audit_log(self, client, registered_supplier):
        sid = registered_supplier["supplier_id"]
        resp = client.get(f"/api/v1/cbam/suppliers/{sid}/audit-log")
        assert resp.status_code == 200
        data = resp.json()
        # Registration should have created an audit entry
        assert data["pagination"]["total"] >= 1

    def test_export_suppliers(self, client, registered_supplier):
        resp = client.get("/api/v1/cbam/suppliers/export?format=json")
        assert resp.status_code == 200


# ===========================================================================
# TEST CLASS -- Request models validation
# ===========================================================================

class TestRequestValidation:
    """Tests for Pydantic request model validation."""

    def test_empty_company_name_rejected(self, client):
        payload = {
            "company_name": "",
            "country_iso": "TR",
            "product_groups": ["steel"],
            "contact": {"person": "Test", "email": "test@test.com"},
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 422

    def test_missing_contact_rejected(self, client):
        payload = {
            "company_name": "NoContact Corp",
            "country_iso": "TR",
            "product_groups": ["steel"],
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 422

    def test_empty_product_groups_rejected(self, client):
        payload = {
            "company_name": "EmptyPG Corp",
            "country_iso": "TR",
            "product_groups": [],
            "contact": {"person": "Test", "email": "test@test.com"},
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 422

    def test_valid_eori_accepted(self, client):
        payload = {
            "company_name": "Valid EORI Corp",
            "country_iso": "DE",
            "eori_number": "DE123456789",
            "product_groups": ["cement"],
            "contact": {"person": "Test", "email": "test@test.com"},
        }
        resp = client.post(
            "/api/v1/cbam/suppliers/register",
            json=payload,
        )
        assert resp.status_code == 201
