# -*- coding: utf-8 -*-
"""
Unit tests for Land Use Emissions REST API Router - AGENT-MRV-006

Tests all 20 REST endpoints defined in the router via FastAPI TestClient.
Validates request/response contracts, status codes, error handling,
and query parameter behavior using mocked LandUseEmissionsService.

Target: 85%+ coverage of api/router.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.land_use_emissions.api.router import create_router


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def mock_service():
    """Create a fully-mocked LandUseEmissionsService."""
    svc = MagicMock(name="LandUseEmissionsService")
    svc._calculations = []
    svc._soc_assessments = []
    svc._compliance_results = []
    svc._total_calculations = 0
    return svc


@pytest.fixture
def app_client(mock_service):
    """Create a FastAPI TestClient with the router and mocked service.

    The router's ``_get_service`` closure calls
    ``from greenlang.land_use_emissions.setup import get_service``
    at request time. We patch that module-level function so every
    endpoint resolves to our mock.
    """
    app = FastAPI()
    router = create_router()
    app.include_router(router)

    with patch(
        "greenlang.land_use_emissions.setup.get_service",
        return_value=mock_service,
    ):
        client = TestClient(app, raise_server_exceptions=False)
        yield client, mock_service


# ===================================================================
# Sample payloads
# ===================================================================


def _calc_request(**overrides) -> Dict[str, Any]:
    """Return a valid SingleCalculationRequest body."""
    base = {
        "parcel_id": "parcel-001",
        "from_category": "forest_land",
        "to_category": "cropland",
        "area_ha": 100.0,
        "climate_zone": "tropical_moist",
        "soil_type": "high_activity_clay",
    }
    base.update(overrides)
    return base


def _batch_request(**overrides) -> Dict[str, Any]:
    """Return a valid BatchCalculationBody body."""
    base = {
        "calculations": [
            _calc_request(),
            _calc_request(area_ha=50.0),
        ],
    }
    base.update(overrides)
    return base


def _carbon_stock_body(**overrides) -> Dict[str, Any]:
    """Return a valid CarbonStockBody."""
    base = {
        "parcel_id": "parcel-001",
        "pool": "above_ground_biomass",
        "stock_tc_ha": 180.0,
        "measurement_date": "2025-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


def _parcel_body(**overrides) -> Dict[str, Any]:
    """Return a valid LandParcelBody."""
    base = {
        "name": "Test Forest",
        "area_ha": 100.0,
        "land_category": "forest_land",
        "climate_zone": "tropical_moist",
        "soil_type": "high_activity_clay",
        "latitude": -3.5,
        "longitude": 28.8,
        "tenant_id": "tenant_001",
    }
    base.update(overrides)
    return base


def _parcel_update_body(**overrides) -> Dict[str, Any]:
    """Return a valid LandParcelUpdateBody."""
    base = {"name": "Updated Name"}
    base.update(overrides)
    return base


def _transition_body(**overrides) -> Dict[str, Any]:
    """Return a valid TransitionBody."""
    base = {
        "parcel_id": "parcel-001",
        "from_category": "forest_land",
        "to_category": "cropland",
        "transition_date": "2025-06-15",
        "area_ha": 50.0,
        "transition_type": "conversion",
    }
    base.update(overrides)
    return base


def _soc_body(**overrides) -> Dict[str, Any]:
    """Return a valid SOCAssessmentBody."""
    base = {
        "parcel_id": "parcel-001",
        "climate_zone": "tropical_moist",
        "soil_type": "high_activity_clay",
        "land_category": "forest_land",
    }
    base.update(overrides)
    return base


def _compliance_body(**overrides) -> Dict[str, Any]:
    """Return a valid ComplianceCheckBody."""
    base = {
        "calculation_id": "lu_calc_abc123",
        "frameworks": ["GHG_PROTOCOL", "IPCC"],
    }
    base.update(overrides)
    return base


def _uncertainty_body(**overrides) -> Dict[str, Any]:
    """Return a valid UncertaintyBody."""
    base = {
        "calculation_id": "lu_calc_abc123",
        "iterations": 5000,
        "seed": 42,
        "confidence_level": 95.0,
    }
    base.update(overrides)
    return base


# ===================================================================
# Standard mock responses
# ===================================================================

_MOCK_CALC_RESPONSE_DICT = {
    "success": True,
    "calculation_id": "lu_calc_test001",
    "from_category": "forest_land",
    "to_category": "cropland",
    "method": "stock_difference",
    "tier": "tier_1",
    "total_co2e_tonnes": 42.5,
    "removals_co2e_tonnes": 5.0,
    "net_co2e_tonnes": 37.5,
    "emissions_by_pool": {"AGB": 30.0},
    "emissions_by_gas": {"CO2": 42.5},
    "area_ha": 100.0,
    "uncertainty_pct": None,
    "provenance_hash": "a" * 64,
    "processing_time_ms": 1.23,
    "timestamp": "2026-02-19T00:00:00+00:00",
}


def _mock_calc_response():
    """Create a MagicMock that behaves like CalculateResponse."""
    resp = MagicMock()
    resp.model_dump.return_value = dict(_MOCK_CALC_RESPONSE_DICT)
    resp.success = True
    resp.total_co2e_tonnes = 42.5
    resp.removals_co2e_tonnes = 5.0
    return resp


_MOCK_BATCH_RESPONSE_DICT = {
    "success": True,
    "batch_id": "lu_batch_test001",
    "total_calculations": 2,
    "successful": 2,
    "failed": 0,
    "total_co2e_tonnes": 85.0,
    "total_removals_tonnes": 10.0,
    "net_co2e_tonnes": 75.0,
    "results": [_MOCK_CALC_RESPONSE_DICT, _MOCK_CALC_RESPONSE_DICT],
    "processing_time_ms": 5.5,
}


def _mock_batch_response():
    """Create a MagicMock that behaves like BatchCalculateResponse."""
    resp = MagicMock()
    resp.model_dump.return_value = dict(_MOCK_BATCH_RESPONSE_DICT)
    return resp


PREFIX = "/api/v1/land-use-emissions"


# ===================================================================
# Test class: POST /calculations (Endpoint 1)
# ===================================================================


class TestPostCalculations:
    """Tests for POST /calculations."""

    def test_valid_request_returns_201(self, app_client):
        client, svc = app_client
        svc.calculate.return_value = _mock_calc_response()
        resp = client.post(f"{PREFIX}/calculations", json=_calc_request())
        assert resp.status_code == 201
        data = resp.json()
        assert data["success"] is True
        assert data["calculation_id"] == "lu_calc_test001"

    def test_with_optional_fields_returns_201(self, app_client):
        client, svc = app_client
        svc.calculate.return_value = _mock_calc_response()
        body = _calc_request(
            tier="tier_2",
            method="gain_loss",
            gwp_source="AR6",
            pools=["above_ground_biomass"],
            management_practice="no_till",
            input_level="high",
            include_fire=True,
            include_n2o=True,
            include_peatland=True,
            disturbance_type="fire",
            peatland_status="drained",
            transition_years=20,
            reference_year=2025,
            tenant_id="t1",
        )
        resp = client.post(f"{PREFIX}/calculations", json=body)
        assert resp.status_code == 201

    def test_missing_required_field_returns_422(self, app_client):
        client, svc = app_client
        body = _calc_request()
        del body["parcel_id"]
        resp = client.post(f"{PREFIX}/calculations", json=body)
        assert resp.status_code == 422

    def test_invalid_area_zero_returns_422(self, app_client):
        client, svc = app_client
        body = _calc_request(area_ha=0)
        resp = client.post(f"{PREFIX}/calculations", json=body)
        assert resp.status_code == 422

    def test_negative_area_returns_422(self, app_client):
        client, svc = app_client
        body = _calc_request(area_ha=-10)
        resp = client.post(f"{PREFIX}/calculations", json=body)
        assert resp.status_code == 422

    def test_value_error_returns_422(self, app_client):
        client, svc = app_client
        svc.calculate.side_effect = ValueError("bad input")
        resp = client.post(f"{PREFIX}/calculations", json=_calc_request())
        assert resp.status_code == 422

    def test_unexpected_error_returns_500(self, app_client):
        client, svc = app_client
        svc.calculate.side_effect = RuntimeError("boom")
        resp = client.post(f"{PREFIX}/calculations", json=_calc_request())
        assert resp.status_code == 500


# ===================================================================
# Test class: POST /calculations/batch (Endpoint 2)
# ===================================================================


class TestPostBatchCalculations:
    """Tests for POST /calculations/batch."""

    def test_valid_batch_returns_201(self, app_client):
        client, svc = app_client
        svc.calculate_batch.return_value = _mock_batch_response()
        resp = client.post(
            f"{PREFIX}/calculations/batch", json=_batch_request(),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_calculations"] == 2

    def test_batch_with_gwp_and_tenant(self, app_client):
        client, svc = app_client
        svc.calculate_batch.return_value = _mock_batch_response()
        body = _batch_request(gwp_source="AR5", tenant_id="t1")
        resp = client.post(f"{PREFIX}/calculations/batch", json=body)
        assert resp.status_code == 201

    def test_empty_calculations_list_returns_422(self, app_client):
        client, svc = app_client
        resp = client.post(
            f"{PREFIX}/calculations/batch",
            json={"calculations": []},
        )
        assert resp.status_code == 422

    def test_missing_calculations_returns_422(self, app_client):
        client, svc = app_client
        resp = client.post(f"{PREFIX}/calculations/batch", json={})
        assert resp.status_code == 422

    def test_batch_value_error_returns_422(self, app_client):
        client, svc = app_client
        svc.calculate_batch.side_effect = ValueError("bad")
        resp = client.post(
            f"{PREFIX}/calculations/batch", json=_batch_request(),
        )
        assert resp.status_code == 422

    def test_batch_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.calculate_batch.side_effect = RuntimeError("fail")
        resp = client.post(
            f"{PREFIX}/calculations/batch", json=_batch_request(),
        )
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /calculations/{id} (Endpoint 3)
# ===================================================================


class TestGetCalculationById:
    """Tests for GET /calculations/{calc_id}."""

    def test_existing_returns_200(self, app_client):
        client, svc = app_client
        svc._calculations = [
            {"calculation_id": "lu_calc_abc", "total_co2e_tonnes": 10.0},
        ]
        resp = client.get(f"{PREFIX}/calculations/lu_calc_abc")
        assert resp.status_code == 200
        assert resp.json()["calculation_id"] == "lu_calc_abc"

    def test_missing_returns_404(self, app_client):
        client, svc = app_client
        svc._calculations = []
        resp = client.get(f"{PREFIX}/calculations/lu_calc_nope")
        assert resp.status_code == 404

    def test_multiple_calculations_finds_correct_one(self, app_client):
        client, svc = app_client
        svc._calculations = [
            {"calculation_id": "lu_calc_a", "val": 1},
            {"calculation_id": "lu_calc_b", "val": 2},
            {"calculation_id": "lu_calc_c", "val": 3},
        ]
        resp = client.get(f"{PREFIX}/calculations/lu_calc_b")
        assert resp.status_code == 200
        assert resp.json()["val"] == 2


# ===================================================================
# Test class: GET /calculations (Endpoint 4)
# ===================================================================


class TestListCalculations:
    """Tests for GET /calculations."""

    def test_list_empty_returns_200(self, app_client):
        client, svc = app_client
        svc._calculations = []
        resp = client.get(f"{PREFIX}/calculations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["calculations"] == []

    def test_list_with_data_returns_200(self, app_client):
        client, svc = app_client
        svc._calculations = [
            {"calculation_id": "c1", "to_category": "cropland",
             "from_category": "forest_land", "method": "stock_difference",
             "tenant_id": "t1"},
        ]
        resp = client.get(f"{PREFIX}/calculations")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_list_filter_by_land_category(self, app_client):
        client, svc = app_client
        svc._calculations = [
            {"calculation_id": "c1", "to_category": "cropland",
             "from_category": "forest_land", "method": "sd", "tenant_id": "t1"},
            {"calculation_id": "c2", "to_category": "grassland",
             "from_category": "wetland", "method": "sd", "tenant_id": "t1"},
        ]
        resp = client.get(
            f"{PREFIX}/calculations", params={"land_category": "cropland"},
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_list_filter_by_method(self, app_client):
        client, svc = app_client
        svc._calculations = [
            {"calculation_id": "c1", "method": "stock_difference"},
            {"calculation_id": "c2", "method": "gain_loss"},
        ]
        resp = client.get(
            f"{PREFIX}/calculations", params={"method": "gain_loss"},
        )
        assert resp.json()["total"] == 1

    def test_list_filter_by_tenant_id(self, app_client):
        client, svc = app_client
        svc._calculations = [
            {"calculation_id": "c1", "tenant_id": "t1"},
            {"calculation_id": "c2", "tenant_id": "t2"},
        ]
        resp = client.get(
            f"{PREFIX}/calculations", params={"tenant_id": "t1"},
        )
        assert resp.json()["total"] == 1

    def test_list_pagination(self, app_client):
        client, svc = app_client
        svc._calculations = [{"calculation_id": f"c{i}"} for i in range(10)]
        resp = client.get(
            f"{PREFIX}/calculations",
            params={"page": 2, "page_size": 3},
        )
        data = resp.json()
        assert data["total"] == 10
        assert len(data["calculations"]) == 3
        assert data["page"] == 2

    def test_list_invalid_page_returns_422(self, app_client):
        client, svc = app_client
        resp = client.get(
            f"{PREFIX}/calculations", params={"page": 0},
        )
        assert resp.status_code == 422

    def test_list_page_size_too_large_returns_422(self, app_client):
        client, svc = app_client
        resp = client.get(
            f"{PREFIX}/calculations", params={"page_size": 101},
        )
        assert resp.status_code == 422


# ===================================================================
# Test class: DELETE /calculations/{id} (Endpoint 5)
# ===================================================================


class TestDeleteCalculation:
    """Tests for DELETE /calculations/{calc_id}."""

    def test_existing_returns_200(self, app_client):
        client, svc = app_client
        svc._calculations = [{"calculation_id": "lu_calc_del1"}]
        resp = client.delete(f"{PREFIX}/calculations/lu_calc_del1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["calculation_id"] == "lu_calc_del1"

    def test_missing_returns_404(self, app_client):
        client, svc = app_client
        svc._calculations = []
        resp = client.delete(f"{PREFIX}/calculations/nope")
        assert resp.status_code == 404

    def test_delete_decrements_total(self, app_client):
        client, svc = app_client
        svc._calculations = [{"calculation_id": "c1"}]
        svc._total_calculations = 1
        resp = client.delete(f"{PREFIX}/calculations/c1")
        assert resp.status_code == 200
        assert svc._total_calculations == 0


# ===================================================================
# Test class: POST /carbon-stocks (Endpoint 6)
# ===================================================================


class TestPostCarbonStocks:
    """Tests for POST /carbon-stocks."""

    def test_valid_returns_201(self, app_client):
        client, svc = app_client
        svc.record_carbon_stock.return_value = {
            "snapshot_id": "cs_test001",
            "parcel_id": "parcel-001",
            "pool": "above_ground_biomass",
            "stock_tc_ha": 180.0,
        }
        resp = client.post(
            f"{PREFIX}/carbon-stocks", json=_carbon_stock_body(),
        )
        assert resp.status_code == 201
        assert resp.json()["snapshot_id"] == "cs_test001"

    def test_missing_field_returns_422(self, app_client):
        client, svc = app_client
        body = _carbon_stock_body()
        del body["parcel_id"]
        resp = client.post(f"{PREFIX}/carbon-stocks", json=body)
        assert resp.status_code == 422

    def test_negative_stock_returns_422(self, app_client):
        client, svc = app_client
        body = _carbon_stock_body(stock_tc_ha=-10)
        resp = client.post(f"{PREFIX}/carbon-stocks", json=body)
        assert resp.status_code == 422

    def test_with_optional_fields(self, app_client):
        client, svc = app_client
        svc.record_carbon_stock.return_value = {"snapshot_id": "cs_opt"}
        body = _carbon_stock_body(
            tier="tier_2",
            source="national_inventory",
            uncertainty_pct=15.0,
            notes="Field measurement",
        )
        resp = client.post(f"{PREFIX}/carbon-stocks", json=body)
        assert resp.status_code == 201

    def test_value_error_returns_422(self, app_client):
        client, svc = app_client
        svc.record_carbon_stock.side_effect = ValueError("bad")
        resp = client.post(
            f"{PREFIX}/carbon-stocks", json=_carbon_stock_body(),
        )
        assert resp.status_code == 422

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.record_carbon_stock.side_effect = RuntimeError("fail")
        resp = client.post(
            f"{PREFIX}/carbon-stocks", json=_carbon_stock_body(),
        )
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /carbon-stocks/{parcel_id} (Endpoint 7)
# ===================================================================


class TestGetCarbonStocks:
    """Tests for GET /carbon-stocks/{parcel_id}."""

    def test_returns_200(self, app_client):
        client, svc = app_client
        svc.get_carbon_stocks.return_value = {
            "parcel_id": "p1",
            "snapshots": [],
            "total": 0,
            "page": 1,
            "page_size": 20,
        }
        resp = client.get(f"{PREFIX}/carbon-stocks/p1")
        assert resp.status_code == 200
        assert resp.json()["parcel_id"] == "p1"

    def test_with_pool_filter(self, app_client):
        client, svc = app_client
        svc.get_carbon_stocks.return_value = {
            "parcel_id": "p1", "snapshots": [], "total": 0,
            "page": 1, "page_size": 20,
        }
        resp = client.get(
            f"{PREFIX}/carbon-stocks/p1",
            params={"pool": "above_ground_biomass"},
        )
        assert resp.status_code == 200

    def test_with_pagination(self, app_client):
        client, svc = app_client
        svc.get_carbon_stocks.return_value = {
            "parcel_id": "p1", "snapshots": [], "total": 0,
            "page": 2, "page_size": 5,
        }
        resp = client.get(
            f"{PREFIX}/carbon-stocks/p1",
            params={"page": 2, "page_size": 5},
        )
        assert resp.status_code == 200

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.get_carbon_stocks.side_effect = RuntimeError("fail")
        resp = client.get(f"{PREFIX}/carbon-stocks/p1")
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /carbon-stocks/{parcel_id}/summary (Endpoint 8)
# ===================================================================


class TestGetCarbonStocksSummary:
    """Tests for GET /carbon-stocks/{parcel_id}/summary."""

    def test_returns_200_empty(self, app_client):
        client, svc = app_client
        svc.get_carbon_stocks.return_value = {
            "parcel_id": "p1", "snapshots": [], "total": 0,
            "page": 1, "page_size": 10000,
        }
        resp = client.get(f"{PREFIX}/carbon-stocks/p1/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["parcel_id"] == "p1"
        assert data["total_stock_tc_ha"] == 0
        assert data["pool_count"] == 0

    def test_summary_aggregates_latest_per_pool(self, app_client):
        client, svc = app_client
        svc.get_carbon_stocks.return_value = {
            "parcel_id": "p1",
            "snapshots": [
                {"pool": "above_ground_biomass", "stock_tc_ha": 180.0,
                 "measurement_date": "2025-06-01"},
                {"pool": "above_ground_biomass", "stock_tc_ha": 170.0,
                 "measurement_date": "2025-01-01"},
                {"pool": "below_ground_biomass", "stock_tc_ha": 40.0,
                 "measurement_date": "2025-06-01"},
            ],
            "total": 3,
            "page": 1,
            "page_size": 10000,
        }
        resp = client.get(f"{PREFIX}/carbon-stocks/p1/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pool_count"] == 2
        # 180.0 (latest AGB) + 40.0 (BGB)
        assert data["total_stock_tc_ha"] == pytest.approx(220.0)

    def test_summary_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.get_carbon_stocks.side_effect = RuntimeError("fail")
        resp = client.get(f"{PREFIX}/carbon-stocks/p1/summary")
        assert resp.status_code == 500


# ===================================================================
# Test class: POST /land-parcels (Endpoint 9)
# ===================================================================


class TestPostLandParcels:
    """Tests for POST /land-parcels."""

    def test_valid_returns_201(self, app_client):
        client, svc = app_client
        svc.register_parcel.return_value = {
            "parcel_id": "parcel_test01",
            "name": "Test Forest",
        }
        resp = client.post(f"{PREFIX}/land-parcels", json=_parcel_body())
        assert resp.status_code == 201
        assert resp.json()["parcel_id"] == "parcel_test01"

    def test_missing_name_returns_422(self, app_client):
        client, svc = app_client
        body = _parcel_body()
        del body["name"]
        resp = client.post(f"{PREFIX}/land-parcels", json=body)
        assert resp.status_code == 422

    def test_missing_tenant_id_returns_422(self, app_client):
        client, svc = app_client
        body = _parcel_body()
        del body["tenant_id"]
        resp = client.post(f"{PREFIX}/land-parcels", json=body)
        assert resp.status_code == 422

    def test_invalid_latitude_returns_422(self, app_client):
        client, svc = app_client
        body = _parcel_body(latitude=91.0)
        resp = client.post(f"{PREFIX}/land-parcels", json=body)
        assert resp.status_code == 422

    def test_invalid_longitude_returns_422(self, app_client):
        client, svc = app_client
        body = _parcel_body(longitude=181.0)
        resp = client.post(f"{PREFIX}/land-parcels", json=body)
        assert resp.status_code == 422

    def test_zero_area_returns_422(self, app_client):
        client, svc = app_client
        body = _parcel_body(area_ha=0)
        resp = client.post(f"{PREFIX}/land-parcels", json=body)
        assert resp.status_code == 422

    def test_with_optional_fields_returns_201(self, app_client):
        client, svc = app_client
        svc.register_parcel.return_value = {"parcel_id": "parcel_opt"}
        body = _parcel_body(
            country_code="US",
            management_practice="no_till",
            input_level="high",
            peatland_status="natural",
        )
        resp = client.post(f"{PREFIX}/land-parcels", json=body)
        assert resp.status_code == 201

    def test_value_error_returns_422(self, app_client):
        client, svc = app_client
        svc.register_parcel.side_effect = ValueError("missing fields")
        resp = client.post(f"{PREFIX}/land-parcels", json=_parcel_body())
        assert resp.status_code == 422

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.register_parcel.side_effect = RuntimeError("fail")
        resp = client.post(f"{PREFIX}/land-parcels", json=_parcel_body())
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /land-parcels (Endpoint 10)
# ===================================================================


class TestListLandParcels:
    """Tests for GET /land-parcels."""

    def test_returns_200(self, app_client):
        client, svc = app_client
        svc.list_parcels.return_value = {
            "parcels": [], "total": 0, "page": 1, "page_size": 20,
        }
        resp = client.get(f"{PREFIX}/land-parcels")
        assert resp.status_code == 200

    def test_with_filters(self, app_client):
        client, svc = app_client
        svc.list_parcels.return_value = {
            "parcels": [{"parcel_id": "p1"}],
            "total": 1, "page": 1, "page_size": 20,
        }
        resp = client.get(
            f"{PREFIX}/land-parcels",
            params={
                "tenant_id": "t1",
                "land_category": "forest_land",
                "climate_zone": "tropical_moist",
            },
        )
        assert resp.status_code == 200

    def test_pagination(self, app_client):
        client, svc = app_client
        svc.list_parcels.return_value = {
            "parcels": [], "total": 50, "page": 3, "page_size": 10,
        }
        resp = client.get(
            f"{PREFIX}/land-parcels",
            params={"page": 3, "page_size": 10},
        )
        assert resp.status_code == 200
        assert resp.json()["page"] == 3

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.list_parcels.side_effect = RuntimeError("fail")
        resp = client.get(f"{PREFIX}/land-parcels")
        assert resp.status_code == 500


# ===================================================================
# Test class: PUT /land-parcels/{id} (Endpoint 11)
# ===================================================================


class TestUpdateLandParcel:
    """Tests for PUT /land-parcels/{parcel_id}."""

    def test_valid_update_returns_200(self, app_client):
        client, svc = app_client
        svc.update_parcel.return_value = {
            "parcel_id": "p1", "name": "Updated",
        }
        resp = client.put(
            f"{PREFIX}/land-parcels/p1",
            json=_parcel_update_body(),
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_not_found_returns_404(self, app_client):
        client, svc = app_client
        svc.update_parcel.return_value = None
        resp = client.put(
            f"{PREFIX}/land-parcels/nonexistent",
            json=_parcel_update_body(),
        )
        assert resp.status_code == 404

    def test_all_optional_fields(self, app_client):
        client, svc = app_client
        svc.update_parcel.return_value = {"parcel_id": "p1"}
        body = {
            "name": "New",
            "area_ha": 200.0,
            "land_category": "cropland",
            "management_practice": "reduced_tillage",
            "input_level": "low",
            "peatland_status": "rewetted",
            "country_code": "GB",
        }
        resp = client.put(f"{PREFIX}/land-parcels/p1", json=body)
        assert resp.status_code == 200

    def test_empty_body_returns_200(self, app_client):
        client, svc = app_client
        svc.update_parcel.return_value = {"parcel_id": "p1"}
        resp = client.put(f"{PREFIX}/land-parcels/p1", json={})
        assert resp.status_code == 200

    def test_value_error_returns_422(self, app_client):
        client, svc = app_client
        svc.update_parcel.side_effect = ValueError("bad")
        resp = client.put(
            f"{PREFIX}/land-parcels/p1",
            json=_parcel_update_body(),
        )
        assert resp.status_code == 422

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.update_parcel.side_effect = RuntimeError("fail")
        resp = client.put(
            f"{PREFIX}/land-parcels/p1",
            json=_parcel_update_body(),
        )
        assert resp.status_code == 500


# ===================================================================
# Test class: POST /transitions (Endpoint 12)
# ===================================================================


class TestPostTransitions:
    """Tests for POST /transitions."""

    def test_valid_returns_201(self, app_client):
        client, svc = app_client
        svc.record_transition.return_value = {
            "transition_id": "tr_test001",
            "parcel_id": "parcel-001",
        }
        resp = client.post(
            f"{PREFIX}/transitions", json=_transition_body(),
        )
        assert resp.status_code == 201
        assert resp.json()["transition_id"] == "tr_test001"

    def test_missing_parcel_id_returns_422(self, app_client):
        client, svc = app_client
        body = _transition_body()
        del body["parcel_id"]
        resp = client.post(f"{PREFIX}/transitions", json=body)
        assert resp.status_code == 422

    def test_zero_area_returns_422(self, app_client):
        client, svc = app_client
        body = _transition_body(area_ha=0)
        resp = client.post(f"{PREFIX}/transitions", json=body)
        assert resp.status_code == 422

    def test_with_optional_fields(self, app_client):
        client, svc = app_client
        svc.record_transition.return_value = {"transition_id": "tr_opt"}
        body = _transition_body(
            disturbance_type="fire",
            notes="Wildfire event",
        )
        resp = client.post(f"{PREFIX}/transitions", json=body)
        assert resp.status_code == 201

    def test_value_error_returns_422(self, app_client):
        client, svc = app_client
        svc.record_transition.side_effect = ValueError("bad")
        resp = client.post(
            f"{PREFIX}/transitions", json=_transition_body(),
        )
        assert resp.status_code == 422

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.record_transition.side_effect = RuntimeError("fail")
        resp = client.post(
            f"{PREFIX}/transitions", json=_transition_body(),
        )
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /transitions (Endpoint 13)
# ===================================================================


class TestListTransitions:
    """Tests for GET /transitions."""

    def test_returns_200(self, app_client):
        client, svc = app_client
        svc.get_transitions.return_value = {
            "transitions": [], "total": 0, "page": 1, "page_size": 20,
        }
        resp = client.get(f"{PREFIX}/transitions")
        assert resp.status_code == 200

    def test_with_all_filters(self, app_client):
        client, svc = app_client
        svc.get_transitions.return_value = {
            "transitions": [], "total": 0, "page": 1, "page_size": 20,
        }
        resp = client.get(
            f"{PREFIX}/transitions",
            params={
                "parcel_id": "p1",
                "from_category": "forest_land",
                "to_category": "cropland",
                "transition_type": "conversion",
                "page": 2,
                "page_size": 5,
            },
        )
        assert resp.status_code == 200

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.get_transitions.side_effect = RuntimeError("fail")
        resp = client.get(f"{PREFIX}/transitions")
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /transitions/matrix (Endpoint 14)
# ===================================================================


class TestGetTransitionMatrix:
    """Tests for GET /transitions/matrix."""

    def test_returns_200(self, app_client):
        client, svc = app_client
        svc.get_transition_matrix.return_value = {
            "categories": ["forest_land", "cropland"],
            "matrix": {},
            "total_area_ha": 0,
            "total_transitions": 0,
        }
        resp = client.get(f"{PREFIX}/transitions/matrix")
        assert resp.status_code == 200

    def test_with_tenant_filter(self, app_client):
        client, svc = app_client
        svc.get_transition_matrix.return_value = {
            "categories": [], "matrix": {},
            "total_area_ha": 0, "total_transitions": 0,
        }
        resp = client.get(
            f"{PREFIX}/transitions/matrix",
            params={"tenant_id": "t1"},
        )
        assert resp.status_code == 200

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.get_transition_matrix.side_effect = RuntimeError("fail")
        resp = client.get(f"{PREFIX}/transitions/matrix")
        assert resp.status_code == 500


# ===================================================================
# Test class: POST /soc-assessments (Endpoint 15)
# ===================================================================


class TestPostSOCAssessments:
    """Tests for POST /soc-assessments."""

    def test_valid_returns_201(self, app_client):
        client, svc = app_client
        svc.assess_soc.return_value = {
            "assessment_id": "soc_test001",
            "parcel_id": "parcel-001",
            "soc_current": 50.0,
        }
        resp = client.post(
            f"{PREFIX}/soc-assessments", json=_soc_body(),
        )
        assert resp.status_code == 201
        assert resp.json()["assessment_id"] == "soc_test001"

    def test_missing_field_returns_422(self, app_client):
        client, svc = app_client
        body = _soc_body()
        del body["parcel_id"]
        resp = client.post(f"{PREFIX}/soc-assessments", json=body)
        assert resp.status_code == 422

    def test_with_all_optional_fields(self, app_client):
        client, svc = app_client
        svc.assess_soc.return_value = {"assessment_id": "soc_opt"}
        body = _soc_body(
            management_practice="no_till",
            input_level="high",
            depth_cm=30,
            transition_years=20,
            previous_land_category="grassland",
            previous_management="full_tillage",
            previous_input_level="low",
        )
        resp = client.post(f"{PREFIX}/soc-assessments", json=body)
        assert resp.status_code == 201

    def test_invalid_depth_returns_422(self, app_client):
        client, svc = app_client
        body = _soc_body(depth_cm=0)
        resp = client.post(f"{PREFIX}/soc-assessments", json=body)
        assert resp.status_code == 422

    def test_invalid_transition_years_returns_422(self, app_client):
        client, svc = app_client
        body = _soc_body(transition_years=101)
        resp = client.post(f"{PREFIX}/soc-assessments", json=body)
        assert resp.status_code == 422

    def test_value_error_returns_422(self, app_client):
        client, svc = app_client
        svc.assess_soc.side_effect = ValueError("bad")
        resp = client.post(
            f"{PREFIX}/soc-assessments", json=_soc_body(),
        )
        assert resp.status_code == 422

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.assess_soc.side_effect = RuntimeError("fail")
        resp = client.post(
            f"{PREFIX}/soc-assessments", json=_soc_body(),
        )
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /soc-assessments/{parcel_id} (Endpoint 16)
# ===================================================================


class TestGetSOCAssessments:
    """Tests for GET /soc-assessments/{parcel_id}."""

    def test_returns_200_empty(self, app_client):
        client, svc = app_client
        svc._soc_assessments = []
        resp = client.get(f"{PREFIX}/soc-assessments/p1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["parcel_id"] == "p1"
        assert data["total"] == 0

    def test_returns_200_with_data(self, app_client):
        client, svc = app_client
        svc._soc_assessments = [
            {"parcel_id": "p1", "assessment_id": "soc_a1"},
            {"parcel_id": "p2", "assessment_id": "soc_a2"},
            {"parcel_id": "p1", "assessment_id": "soc_a3"},
        ]
        resp = client.get(f"{PREFIX}/soc-assessments/p1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_pagination(self, app_client):
        client, svc = app_client
        svc._soc_assessments = [
            {"parcel_id": "p1", "assessment_id": f"soc_{i}"}
            for i in range(10)
        ]
        resp = client.get(
            f"{PREFIX}/soc-assessments/p1",
            params={"page": 2, "page_size": 3},
        )
        data = resp.json()
        assert data["total"] == 10
        assert len(data["assessments"]) == 3


# ===================================================================
# Test class: POST /compliance/check (Endpoint 17)
# ===================================================================


class TestPostComplianceCheck:
    """Tests for POST /compliance/check."""

    def test_valid_returns_200(self, app_client):
        client, svc = app_client
        svc.check_compliance.return_value = {
            "id": "comp_test001",
            "success": True,
            "frameworks_checked": 2,
            "compliant": 2,
            "non_compliant": 0,
            "partial": 0,
            "results": [],
        }
        resp = client.post(
            f"{PREFIX}/compliance/check", json=_compliance_body(),
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_empty_body_returns_200(self, app_client):
        client, svc = app_client
        svc.check_compliance.return_value = {
            "id": "comp_empty", "success": True,
            "frameworks_checked": 6, "compliant": 0,
            "non_compliant": 0, "partial": 0, "results": [],
        }
        resp = client.post(f"{PREFIX}/compliance/check", json={})
        assert resp.status_code == 200

    def test_with_tenant_id(self, app_client):
        client, svc = app_client
        svc.check_compliance.return_value = {
            "id": "comp_tid", "success": True,
            "frameworks_checked": 1, "compliant": 0,
            "non_compliant": 0, "partial": 0, "results": [],
        }
        body = _compliance_body(tenant_id="t1")
        resp = client.post(f"{PREFIX}/compliance/check", json=body)
        assert resp.status_code == 200

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.check_compliance.side_effect = RuntimeError("fail")
        resp = client.post(
            f"{PREFIX}/compliance/check", json=_compliance_body(),
        )
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /compliance/{id} (Endpoint 18)
# ===================================================================


class TestGetComplianceResult:
    """Tests for GET /compliance/{compliance_id}."""

    def test_existing_returns_200(self, app_client):
        client, svc = app_client
        svc._compliance_results = [
            {"id": "comp_a1", "success": True, "frameworks_checked": 2},
        ]
        resp = client.get(f"{PREFIX}/compliance/comp_a1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "comp_a1"

    def test_missing_returns_404(self, app_client):
        client, svc = app_client
        svc._compliance_results = []
        resp = client.get(f"{PREFIX}/compliance/comp_nope")
        assert resp.status_code == 404

    def test_multiple_finds_correct(self, app_client):
        client, svc = app_client
        svc._compliance_results = [
            {"id": "comp_a1", "val": 1},
            {"id": "comp_a2", "val": 2},
        ]
        resp = client.get(f"{PREFIX}/compliance/comp_a2")
        assert resp.status_code == 200
        assert resp.json()["val"] == 2


# ===================================================================
# Test class: POST /uncertainty (Endpoint 19)
# ===================================================================


class TestPostUncertainty:
    """Tests for POST /uncertainty."""

    def test_valid_returns_200(self, app_client):
        client, svc = app_client
        svc.run_uncertainty.return_value = {
            "success": True,
            "calculation_id": "lu_calc_abc123",
            "method": "monte_carlo",
            "iterations": 5000,
            "mean_co2e_tonnes": 42.5,
            "std_dev_tonnes": 5.0,
            "ci_lower": 32.7,
            "ci_upper": 52.3,
            "confidence_level": 95.0,
            "coefficient_of_variation": 11.76,
            "provenance_hash": "b" * 64,
        }
        resp = client.post(
            f"{PREFIX}/uncertainty", json=_uncertainty_body(),
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_missing_calculation_id_returns_422(self, app_client):
        client, svc = app_client
        body = {"iterations": 1000}
        resp = client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 422

    def test_zero_iterations_returns_422(self, app_client):
        client, svc = app_client
        body = _uncertainty_body(iterations=0)
        resp = client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 422

    def test_too_many_iterations_returns_422(self, app_client):
        client, svc = app_client
        body = _uncertainty_body(iterations=2_000_000)
        resp = client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 422

    def test_confidence_level_out_of_range_returns_422(self, app_client):
        client, svc = app_client
        body = _uncertainty_body(confidence_level=100.0)
        resp = client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 422

    def test_confidence_level_zero_returns_422(self, app_client):
        client, svc = app_client
        body = _uncertainty_body(confidence_level=0)
        resp = client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 422

    def test_value_error_returns_422(self, app_client):
        client, svc = app_client
        svc.run_uncertainty.side_effect = ValueError("bad")
        resp = client.post(
            f"{PREFIX}/uncertainty", json=_uncertainty_body(),
        )
        assert resp.status_code == 422

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.run_uncertainty.side_effect = RuntimeError("fail")
        resp = client.post(
            f"{PREFIX}/uncertainty", json=_uncertainty_body(),
        )
        assert resp.status_code == 500


# ===================================================================
# Test class: GET /aggregations (Endpoint 20)
# ===================================================================


class TestGetAggregations:
    """Tests for GET /aggregations."""

    def test_valid_returns_200(self, app_client):
        client, svc = app_client
        svc.aggregate.return_value = {
            "groups": {},
            "total_co2e_tonnes": 0,
            "total_removals_tonnes": 0,
            "net_co2e_tonnes": 0,
            "area_ha": 0,
            "calculation_count": 0,
            "period": "annual",
            "tenant_id": "t1",
        }
        resp = client.get(
            f"{PREFIX}/aggregations", params={"tenant_id": "t1"},
        )
        assert resp.status_code == 200

    def test_missing_tenant_id_returns_422(self, app_client):
        client, svc = app_client
        resp = client.get(f"{PREFIX}/aggregations")
        assert resp.status_code == 422

    def test_with_all_query_params(self, app_client):
        client, svc = app_client
        svc.aggregate.return_value = {
            "groups": {},
            "total_co2e_tonnes": 0,
            "total_removals_tonnes": 0,
            "net_co2e_tonnes": 0,
            "area_ha": 0,
            "calculation_count": 0,
            "period": "quarterly",
            "tenant_id": "t1",
        }
        resp = client.get(
            f"{PREFIX}/aggregations",
            params={
                "tenant_id": "t1",
                "period": "quarterly",
                "group_by": "land_category,climate_zone",
                "date_from": "2025-01-01",
                "date_to": "2025-12-31",
                "land_categories": "forest_land,cropland",
            },
        )
        assert resp.status_code == 200

    def test_group_by_parsed_as_list(self, app_client):
        client, svc = app_client
        svc.aggregate.return_value = {
            "groups": {}, "total_co2e_tonnes": 0,
            "total_removals_tonnes": 0, "net_co2e_tonnes": 0,
            "area_ha": 0, "calculation_count": 0,
            "period": "annual", "tenant_id": "t1",
        }
        resp = client.get(
            f"{PREFIX}/aggregations",
            params={
                "tenant_id": "t1",
                "group_by": "land_category,climate_zone",
            },
        )
        assert resp.status_code == 200
        # Verify aggregate was called with parsed group_by list
        call_args = svc.aggregate.call_args[0][0]
        assert call_args["group_by"] == ["land_category", "climate_zone"]

    def test_land_categories_parsed_as_list(self, app_client):
        client, svc = app_client
        svc.aggregate.return_value = {
            "groups": {}, "total_co2e_tonnes": 0,
            "total_removals_tonnes": 0, "net_co2e_tonnes": 0,
            "area_ha": 0, "calculation_count": 0,
            "period": "annual", "tenant_id": "t1",
        }
        resp = client.get(
            f"{PREFIX}/aggregations",
            params={
                "tenant_id": "t1",
                "land_categories": "forest_land,wetland",
            },
        )
        assert resp.status_code == 200
        call_args = svc.aggregate.call_args[0][0]
        assert call_args["land_categories"] == ["forest_land", "wetland"]

    def test_server_error_returns_500(self, app_client):
        client, svc = app_client
        svc.aggregate.side_effect = RuntimeError("fail")
        resp = client.get(
            f"{PREFIX}/aggregations", params={"tenant_id": "t1"},
        )
        assert resp.status_code == 500


# ===================================================================
# Test class: Router creation and configuration
# ===================================================================


class TestRouterCreation:
    """Tests for create_router() factory function."""

    def test_create_router_returns_api_router(self):
        from fastapi import APIRouter
        router = create_router()
        assert isinstance(router, APIRouter)

    def test_create_router_has_correct_prefix(self):
        router = create_router()
        assert router.prefix == "/api/v1/land-use-emissions"

    def test_create_router_has_tag(self):
        router = create_router()
        assert "Land Use Emissions" in router.tags

    def test_create_router_raises_without_fastapi(self):
        import greenlang.land_use_emissions.api.router as mod
        original = mod.FASTAPI_AVAILABLE
        mod.FASTAPI_AVAILABLE = False
        try:
            with pytest.raises(RuntimeError, match="FastAPI is required"):
                create_router()
        finally:
            mod.FASTAPI_AVAILABLE = original

    def test_router_has_20_routes(self):
        router = create_router()
        # Each route registers at least one method
        assert len(router.routes) >= 20


# ===================================================================
# Test class: 503 service unavailable
# ===================================================================


class TestServiceUnavailable:
    """Tests for 503 when service is not initialized."""

    def test_503_when_service_is_none(self):
        """Endpoints return 503 when get_service returns None."""
        app = FastAPI()
        router = create_router()
        app.include_router(router)

        with patch(
            "greenlang.land_use_emissions.setup.get_service",
            return_value=None,
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(f"{PREFIX}/calculations")
            assert resp.status_code == 503


# ===================================================================
# Test class: Edge cases and contract validation
# ===================================================================


class TestEdgeCases:
    """Additional edge-case and contract validation tests."""

    def test_calc_request_all_fields_none_still_valid(self, app_client):
        """Minimal required fields suffice; optional fields default."""
        client, svc = app_client
        svc.calculate.return_value = _mock_calc_response()
        body = _calc_request()
        resp = client.post(f"{PREFIX}/calculations", json=body)
        assert resp.status_code == 201

    def test_empty_json_body_post_calculations_returns_422(self, app_client):
        """POST /calculations with empty body returns 422."""
        client, svc = app_client
        resp = client.post(f"{PREFIX}/calculations", json={})
        assert resp.status_code == 422

    def test_empty_json_body_post_transitions_returns_422(self, app_client):
        """POST /transitions with empty body returns 422."""
        client, svc = app_client
        resp = client.post(f"{PREFIX}/transitions", json={})
        assert resp.status_code == 422

    def test_empty_json_body_post_soc_returns_422(self, app_client):
        """POST /soc-assessments with empty body returns 422."""
        client, svc = app_client
        resp = client.post(f"{PREFIX}/soc-assessments", json={})
        assert resp.status_code == 422

    def test_empty_json_body_post_carbon_stocks_returns_422(self, app_client):
        """POST /carbon-stocks with empty body returns 422."""
        client, svc = app_client
        resp = client.post(f"{PREFIX}/carbon-stocks", json={})
        assert resp.status_code == 422

    def test_empty_json_body_post_land_parcels_returns_422(self, app_client):
        """POST /land-parcels with empty body returns 422."""
        client, svc = app_client
        resp = client.post(f"{PREFIX}/land-parcels", json={})
        assert resp.status_code == 422

    def test_country_code_max_length_enforced(self, app_client):
        """Country code longer than 2 characters returns 422."""
        client, svc = app_client
        body = _parcel_body(country_code="USA")
        resp = client.post(f"{PREFIX}/land-parcels", json=body)
        assert resp.status_code == 422

    def test_negative_seed_returns_422(self, app_client):
        """Negative seed in uncertainty body returns 422."""
        client, svc = app_client
        body = _uncertainty_body(seed=-1)
        resp = client.post(f"{PREFIX}/uncertainty", json=body)
        assert resp.status_code == 422

    def test_uncertainty_pct_over_100_returns_422(self, app_client):
        """Carbon stock uncertainty_pct > 100 returns 422."""
        client, svc = app_client
        body = _carbon_stock_body(uncertainty_pct=101)
        resp = client.post(f"{PREFIX}/carbon-stocks", json=body)
        assert resp.status_code == 422

    def test_parcel_name_too_long_returns_422(self, app_client):
        """Parcel name exceeding 500 characters returns 422."""
        client, svc = app_client
        body = _parcel_body(name="X" * 501)
        resp = client.post(f"{PREFIX}/land-parcels", json=body)
        assert resp.status_code == 422

    def test_transition_notes_too_long_returns_422(self, app_client):
        """Transition notes exceeding 2000 characters returns 422."""
        client, svc = app_client
        body = _transition_body(notes="N" * 2001)
        resp = client.post(f"{PREFIX}/transitions", json=body)
        assert resp.status_code == 422

    def test_soc_depth_over_300_returns_422(self, app_client):
        """SOC depth_cm > 300 returns 422."""
        client, svc = app_client
        body = _soc_body(depth_cm=301)
        resp = client.post(f"{PREFIX}/soc-assessments", json=body)
        assert resp.status_code == 422
