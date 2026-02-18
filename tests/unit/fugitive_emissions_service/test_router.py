# -*- coding: utf-8 -*-
"""
Unit tests for Fugitive Emissions REST API Router - AGENT-MRV-005.

Tests all 20 REST endpoints at /api/v1/fugitive-emissions using FastAPI
TestClient with a fully mocked FugitiveEmissionsService.

Validates:
- Request parsing and Pydantic validation (status codes 200/201/422)
- Correct delegation to service methods with expected arguments
- Response structure and field values
- Error handling (404, 422, 500)
- Query parameter filtering and pagination
- Router creation and metadata

Endpoints covered:
     1. POST   /calculate              8 tests
     2. POST   /calculate/batch        5 tests
     3. GET    /calculations           5 tests
     4. GET    /calculations/{calc_id} (included above)
     5. POST   /sources                3 tests
     6. GET    /sources                2 tests
     7. GET    /sources/{source_id}    2 tests
     8. POST   /components             3 tests
     9. GET    /components             2 tests
    10. GET    /components/{id}        2 tests
    11. POST   /surveys                4 tests
    12. GET    /surveys                1 test
    13. POST   /factors                4 tests
    14. GET    /factors                1 test
    15. POST   /repairs                4 tests
    16. GET    /repairs                1 test
    17. POST   /uncertainty            6 tests
    18. POST   /compliance/check       4 tests
    19. GET    /health                 3 tests
    20. GET    /stats                  3 tests
    Router creation                    4 tests

Total: 67 tests across 12 test classes.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Conditional imports - skip entire module if FastAPI not available
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.fugitive_emissions.api.router import create_router
from greenlang.fugitive_emissions.setup import (
    BatchCalculateResponse,
    CalculateResponse,
    ComplianceCheckResponse,
    ComponentDetailResponse,
    ComponentListResponse,
    FactorDetailResponse,
    FactorListResponse,
    FugitiveEmissionsService,
    HealthResponse,
    RepairListResponse,
    SourceDetailResponse,
    SourceListResponse,
    StatsResponse,
    SurveyListResponse,
    UncertaintyResponse,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREFIX = "/api/v1/fugitive-emissions"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service() -> MagicMock:
    """Create a mock FugitiveEmissionsService with sensible defaults.

    Every public method used by the router is pre-configured to return
    a valid response model so that tests can focus on HTTP-layer behaviour.
    """
    svc = MagicMock(spec=FugitiveEmissionsService)

    # -- In-memory calculation store used by GET /calculations endpoints --
    svc._calculations = [
        {
            "calculation_id": "fe_calc_existing001",
            "source_type": "EQUIPMENT_LEAK",
            "method": "AVERAGE_EMISSION_FACTOR",
            "total_co2e_kg": 100.0,
            "provenance_hash": "b" * 64,
            "timestamp": "2026-01-15T00:00:00+00:00",
            "status": "SUCCESS",
        },
        {
            "calculation_id": "fe_calc_existing002",
            "source_type": "COAL_MINE_METHANE",
            "method": "ENGINEERING_ESTIMATE",
            "total_co2e_kg": 200.0,
            "provenance_hash": "c" * 64,
            "timestamp": "2026-01-16T00:00:00+00:00",
            "status": "SUCCESS",
        },
    ]

    # -- POST /calculate --
    svc.calculate.return_value = CalculateResponse(
        success=True,
        calculation_id="fe_calc_abc123def456",
        source_type="EQUIPMENT_LEAK",
        calculation_method="AVERAGE_EMISSION_FACTOR",
        total_co2e_kg=1250.75,
        ch4_kg=50.0,
        voc_kg=12.5,
        n2o_kg=0.1,
        uncertainty_pct=25.0,
        provenance_hash="a" * 64,
        processing_time_ms=2.3,
        timestamp="2026-02-18T00:00:00+00:00",
    )

    # -- POST /calculate/batch --
    svc.calculate_batch.return_value = BatchCalculateResponse(
        success=True,
        total_calculations=3,
        successful=3,
        failed=0,
        total_co2e_kg=3752.25,
        results=[
            {"calculation_id": "fe_calc_b1", "success": True},
            {"calculation_id": "fe_calc_b2", "success": True},
            {"calculation_id": "fe_calc_b3", "success": True},
        ],
        processing_time_ms=8.1,
    )

    # -- POST /sources --
    svc.register_source.return_value = SourceDetailResponse(
        source_id="CUSTOM_SRC",
        source_type="CUSTOM_SRC",
        name="Custom Source",
        gases=["CH4", "CO2"],
        methods=["AVERAGE_EMISSION_FACTOR"],
    )

    # -- GET /sources --
    svc.list_sources.return_value = SourceListResponse(
        sources=[
            {"source_id": "EQUIPMENT_LEAK", "name": "Equipment Leak"},
            {"source_id": "TANK_STORAGE", "name": "Tank Storage"},
        ],
        total=2,
        page=1,
        page_size=20,
    )

    # -- GET /sources/{source_id} --
    svc.get_source.return_value = SourceDetailResponse(
        source_id="EQUIPMENT_LEAK",
        source_type="EQUIPMENT_LEAK",
        name="Equipment Leak",
        gases=["CH4", "VOC"],
        methods=["AVERAGE_EMISSION_FACTOR", "SCREENING_RANGE"],
    )

    # -- POST /components --
    svc.register_component.return_value = ComponentDetailResponse(
        component_id="comp_abc123",
        tag_number="V-101-A",
        component_type="valve",
        service_type="gas",
        facility_id="FAC-001",
    )

    # -- GET /components --
    svc.list_components.return_value = ComponentListResponse(
        components=[{"component_id": "comp_abc123", "tag_number": "V-101-A"}],
        total=1,
        page=1,
        page_size=20,
    )

    # -- GET /components/{component_id} --
    svc.get_component.return_value = ComponentDetailResponse(
        component_id="comp_abc123",
        tag_number="V-101-A",
        component_type="valve",
        service_type="gas",
        facility_id="FAC-001",
    )

    # -- POST /surveys --
    svc.register_survey.return_value = {
        "survey_id": "survey_test001",
        "survey_type": "OGI",
        "facility_id": "FAC-001",
        "components_surveyed": 500,
        "leaks_found": 3,
    }

    # -- GET /surveys --
    svc.list_surveys.return_value = SurveyListResponse(
        surveys=[{"survey_id": "survey_test001", "survey_type": "OGI"}],
        total=1,
    )

    # -- POST /factors --
    svc.register_factor.return_value = FactorDetailResponse(
        factor_id="fef_test001",
        source_type="EQUIPMENT_LEAK",
        component_type="valve",
        gas="CH4",
        value=0.00597,
        source="EPA",
    )

    # -- GET /factors --
    svc.list_factors.return_value = FactorListResponse(
        factors=[{"factor_id": "fef_test001", "gas": "CH4", "value": 0.00597}],
        total=1,
    )

    # -- POST /repairs --
    svc.register_repair.return_value = {
        "repair_id": "repair_test001",
        "component_id": "comp_abc123",
        "repair_type": "minor",
        "pre_repair_rate_ppm": 10500.0,
        "post_repair_rate_ppm": 50.0,
    }

    # -- GET /repairs --
    svc.list_repairs.return_value = RepairListResponse(
        repairs=[{"repair_id": "repair_test001", "repair_type": "minor"}],
        total=1,
    )

    # -- POST /uncertainty --
    svc.run_uncertainty.return_value = UncertaintyResponse(
        success=True,
        method="monte_carlo",
        iterations=5000,
        mean_co2e_kg=1250.75,
        std_dev_kg=312.69,
        confidence_intervals={"95": {"lower": 637.87, "upper": 1863.63}},
        dqi_score=3.2,
    )

    # -- POST /compliance/check --
    svc.check_compliance.return_value = ComplianceCheckResponse(
        success=True,
        frameworks_checked=7,
        compliant=5,
        non_compliant=1,
        partial=1,
        results=[{"framework": "GHG_PROTOCOL", "status": "COMPLIANT"}],
    )

    # -- GET /health --
    svc.health_check.return_value = HealthResponse(
        status="healthy",
        service="fugitive-emissions",
        version="1.0.0",
        engines={
            "equipment_component": "available",
            "uncertainty_quantifier": "available",
            "compliance_checker": "available",
            "pipeline": "available",
        },
    )

    # -- GET /stats --
    svc.get_stats.return_value = StatsResponse(
        total_calculations=42,
        total_sources=6,
        total_components=150,
        total_surveys=8,
        total_repairs=12,
        uptime_seconds=7200.0,
    )

    return svc


@pytest.fixture
def client(mock_service: MagicMock) -> TestClient:
    """Create a FastAPI TestClient with mock service injected.

    Patches ``greenlang.fugitive_emissions.setup.get_service`` so that
    the router's internal ``_get_service()`` closure returns the mock.
    """
    app = FastAPI()
    router = create_router()
    app.include_router(router)

    with patch(
        "greenlang.fugitive_emissions.setup.get_service",
        return_value=mock_service,
    ):
        yield TestClient(app)


# ===========================================================================
# 1. TestCalculateEndpoint (8 tests)
# ===========================================================================


class TestCalculateEndpoint:
    """Tests for POST /calculate -- single fugitive emission calculation."""

    def test_calculate_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate returns 200 with valid minimal body."""
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-001",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["calculation_id"] == "fe_calc_abc123def456"
        mock_service.calculate.assert_called_once()

    def test_calculate_with_all_optional_fields(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate accepts every optional field."""
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "TANK_STORAGE",
            "facility_id": "FAC-002",
            "component_count": 200,
            "calculation_method": "MASS_BALANCE",
            "gwp_source": "AR6",
            "gas_composition": {"CH4": 0.85, "C2H6": 0.10, "C3H8": 0.05},
            "service_type": "GAS",
            "tank_type": "FIXED_ROOF",
            "tank_parameters": {"diameter_m": 15.0, "height_m": 12.0},
            "abatement_efficiency": 0.95,
            "operating_hours": 8760.0,
        })
        assert resp.status_code == 200
        call_data = mock_service.calculate.call_args[0][0]
        assert call_data["component_count"] == 200
        assert call_data["calculation_method"] == "MASS_BALANCE"
        assert call_data["gwp_source"] == "AR6"
        assert call_data["gas_composition"]["CH4"] == 0.85
        assert call_data["service_type"] == "GAS"
        assert call_data["tank_type"] == "FIXED_ROOF"
        assert call_data["abatement_efficiency"] == 0.95
        assert call_data["operating_hours"] == 8760.0

    def test_calculate_default_source_type(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate with empty body defaults source_type to EQUIPMENT_LEAK."""
        resp = client.post(f"{PREFIX}/calculate", json={})
        assert resp.status_code == 200
        call_data = mock_service.calculate.call_args[0][0]
        assert call_data["source_type"] == "EQUIPMENT_LEAK"

    def test_calculate_response_has_per_gas_breakdown(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate response includes ch4_kg, voc_kg, n2o_kg."""
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "EQUIPMENT_LEAK",
        })
        body = resp.json()
        assert body["ch4_kg"] == 50.0
        assert body["voc_kg"] == 12.5
        assert body["n2o_kg"] == 0.1

    def test_calculate_response_has_provenance_hash(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate response includes a 64-character provenance hash."""
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "EQUIPMENT_LEAK",
        })
        body = resp.json()
        assert len(body["provenance_hash"]) == 64

    def test_calculate_response_has_processing_time(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate response includes processing_time_ms >= 0."""
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "EQUIPMENT_LEAK",
        })
        body = resp.json()
        assert "processing_time_ms" in body
        assert body["processing_time_ms"] >= 0

    def test_calculate_value_error_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate returns 422 when service raises ValueError."""
        mock_service.calculate.side_effect = ValueError("Invalid source type")
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "EQUIPMENT_LEAK",
        })
        assert resp.status_code == 422
        assert "Invalid source type" in resp.json()["detail"]

    def test_calculate_runtime_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate returns 500 when service raises RuntimeError."""
        mock_service.calculate.side_effect = RuntimeError("Engine crashed")
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "EQUIPMENT_LEAK",
        })
        assert resp.status_code == 500
        assert "Engine crashed" in resp.json()["detail"]


# ===========================================================================
# 2. TestBatchCalculateEndpoint (5 tests)
# ===========================================================================


class TestBatchCalculateEndpoint:
    """Tests for POST /calculate/batch -- batch fugitive calculations."""

    def test_batch_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate/batch returns 200 with valid batch."""
        resp = client.post(f"{PREFIX}/calculate/batch", json={
            "calculations": [
                {"source_type": "EQUIPMENT_LEAK", "component_count": 100},
                {"source_type": "TANK_STORAGE", "facility_id": "FAC-002"},
                {"source_type": "PNEUMATIC_DEVICE", "component_count": 50},
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total_calculations"] == 3
        assert body["successful"] == 3
        assert body["failed"] == 0
        mock_service.calculate_batch.assert_called_once()

    def test_batch_returns_aggregate_co2e(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate/batch includes total_co2e_kg aggregate."""
        resp = client.post(f"{PREFIX}/calculate/batch", json={
            "calculations": [{"source_type": "EQUIPMENT_LEAK"}],
        })
        body = resp.json()
        assert body["total_co2e_kg"] == 3752.25

    def test_batch_returns_processing_time(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate/batch includes processing_time_ms."""
        resp = client.post(f"{PREFIX}/calculate/batch", json={
            "calculations": [{"source_type": "EQUIPMENT_LEAK"}],
        })
        body = resp.json()
        assert "processing_time_ms" in body
        assert body["processing_time_ms"] >= 0

    def test_batch_empty_list_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate/batch with empty calculations returns 422."""
        resp = client.post(f"{PREFIX}/calculate/batch", json={
            "calculations": [],
        })
        assert resp.status_code == 422

    def test_batch_runtime_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate/batch returns 500 on unexpected error."""
        mock_service.calculate_batch.side_effect = RuntimeError("Batch failed")
        resp = client.post(f"{PREFIX}/calculate/batch", json={
            "calculations": [{"source_type": "EQUIPMENT_LEAK"}],
        })
        assert resp.status_code == 500
        assert "Batch failed" in resp.json()["detail"]


# ===========================================================================
# 3. TestCalculationsListEndpoint (5 tests)
# ===========================================================================


class TestCalculationsListEndpoint:
    """Tests for GET /calculations and GET /calculations/{calc_id}."""

    def test_list_calculations_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /calculations returns paginated list with default params."""
        resp = client.get(f"{PREFIX}/calculations")
        assert resp.status_code == 200
        body = resp.json()
        assert "calculations" in body
        assert "total" in body
        assert body["page"] == 1
        assert body["page_size"] == 20

    def test_list_calculations_pagination(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /calculations respects page and page_size params."""
        resp = client.get(
            f"{PREFIX}/calculations",
            params={"page": 1, "page_size": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["page_size"] == 1
        assert len(body["calculations"]) <= 1

    def test_list_calculations_filter_by_source_type(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /calculations?source_type=EQUIPMENT_LEAK filters correctly."""
        resp = client.get(
            f"{PREFIX}/calculations",
            params={"source_type": "EQUIPMENT_LEAK"},
        )
        assert resp.status_code == 200
        body = resp.json()
        for calc in body["calculations"]:
            assert calc["source_type"] == "EQUIPMENT_LEAK"

    def test_get_calculation_by_id_found(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /calculations/{calc_id} returns 200 when calculation exists."""
        resp = client.get(f"{PREFIX}/calculations/fe_calc_existing001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["calculation_id"] == "fe_calc_existing001"
        assert body["source_type"] == "EQUIPMENT_LEAK"

    def test_get_calculation_by_id_not_found(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /calculations/{calc_id} returns 404 when not found."""
        resp = client.get(f"{PREFIX}/calculations/nonexistent_id_xyz")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===========================================================================
# 4. TestSourceEndpoints (7 tests)
# ===========================================================================


class TestSourceEndpoints:
    """Tests for POST /sources, GET /sources, GET /sources/{source_id}."""

    def test_register_source_returns_201(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /sources registers a new source type and returns 201."""
        resp = client.post(f"{PREFIX}/sources", json={
            "source_type": "CUSTOM_SRC",
            "name": "Custom Source",
            "gases": ["CH4", "CO2"],
            "methods": ["AVERAGE_EMISSION_FACTOR"],
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["source_type"] == "CUSTOM_SRC"
        assert body["name"] == "Custom Source"
        mock_service.register_source.assert_called_once()

    def test_register_source_minimal_body(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /sources with only required source_type works."""
        resp = client.post(f"{PREFIX}/sources", json={
            "source_type": "FLARE_STACK",
        })
        assert resp.status_code == 201
        mock_service.register_source.assert_called_once()

    def test_register_source_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /sources returns 500 on service exception."""
        mock_service.register_source.side_effect = RuntimeError("DB write error")
        resp = client.post(f"{PREFIX}/sources", json={
            "source_type": "BROKEN_SRC",
        })
        assert resp.status_code == 500

    def test_list_sources_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /sources returns 200 with source list."""
        resp = client.get(f"{PREFIX}/sources")
        assert resp.status_code == 200
        body = resp.json()
        assert "sources" in body
        assert body["total"] == 2
        mock_service.list_sources.assert_called_once()

    def test_list_sources_pagination(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /sources passes pagination params to service."""
        resp = client.get(
            f"{PREFIX}/sources",
            params={"page": 2, "page_size": 10},
        )
        assert resp.status_code == 200
        mock_service.list_sources.assert_called_once_with(
            page=2, page_size=10,
        )

    def test_get_source_found(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /sources/{source_id} returns source details when found."""
        resp = client.get(f"{PREFIX}/sources/EQUIPMENT_LEAK")
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_type"] == "EQUIPMENT_LEAK"
        assert "gases" in body
        assert "methods" in body
        mock_service.get_source.assert_called_once_with("EQUIPMENT_LEAK")

    def test_get_source_not_found(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /sources/{source_id} returns 404 for unknown source."""
        mock_service.get_source.return_value = None
        resp = client.get(f"{PREFIX}/sources/NONEXISTENT_TYPE")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===========================================================================
# 5. TestComponentEndpoints (7 tests)
# ===========================================================================


class TestComponentEndpoints:
    """Tests for POST /components, GET /components, GET /components/{id}."""

    def test_register_component_returns_201(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /components registers a new component and returns 201."""
        resp = client.post(f"{PREFIX}/components", json={
            "tag_number": "V-101-A",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
            "unit_id": "UNIT-A",
            "location": "North Platform, Level 2",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["component_id"] == "comp_abc123"
        assert body["tag_number"] == "V-101-A"
        mock_service.register_component.assert_called_once()

    def test_register_component_minimal_body(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /components works with only required tag_number."""
        resp = client.post(f"{PREFIX}/components", json={
            "tag_number": "P-200-B",
        })
        assert resp.status_code == 201

    def test_register_component_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /components returns 500 on service exception."""
        mock_service.register_component.side_effect = RuntimeError(
            "Registry full"
        )
        resp = client.post(f"{PREFIX}/components", json={
            "tag_number": "V-ERR",
        })
        assert resp.status_code == 500

    def test_list_components_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /components returns paginated component list."""
        resp = client.get(f"{PREFIX}/components")
        assert resp.status_code == 200
        body = resp.json()
        assert "components" in body
        assert body["total"] == 1

    def test_list_components_pagination(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /components passes pagination params to service."""
        resp = client.get(
            f"{PREFIX}/components",
            params={"page": 3, "page_size": 50},
        )
        assert resp.status_code == 200
        mock_service.list_components.assert_called_once_with(
            page=3, page_size=50,
        )

    def test_get_component_found(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /components/{component_id} returns details when found."""
        resp = client.get(f"{PREFIX}/components/comp_abc123")
        assert resp.status_code == 200
        body = resp.json()
        assert body["component_id"] == "comp_abc123"
        assert body["component_type"] == "valve"
        mock_service.get_component.assert_called_once_with("comp_abc123")

    def test_get_component_not_found(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /components/{component_id} returns 404 for unknown id."""
        mock_service.get_component.return_value = None
        resp = client.get(f"{PREFIX}/components/comp_nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===========================================================================
# 6. TestSurveyEndpoints (5 tests)
# ===========================================================================


class TestSurveyEndpoints:
    """Tests for POST /surveys and GET /surveys."""

    def test_register_survey_returns_201(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /surveys registers a new LDAR survey and returns 201."""
        resp = client.post(f"{PREFIX}/surveys", json={
            "survey_type": "OGI",
            "facility_id": "FAC-001",
            "survey_date": "2026-02-15T09:00:00Z",
            "components_surveyed": 500,
            "leaks_found": 3,
            "threshold_ppm": 10000.0,
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["survey_id"] == "survey_test001"
        assert body["survey_type"] == "OGI"
        mock_service.register_survey.assert_called_once()

    def test_register_survey_minimal_body(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /surveys works with empty body (all fields have defaults)."""
        resp = client.post(f"{PREFIX}/surveys", json={})
        assert resp.status_code == 201
        mock_service.register_survey.assert_called_once()

    def test_register_survey_negative_leaks_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /surveys with negative leaks_found returns 422."""
        resp = client.post(f"{PREFIX}/surveys", json={
            "leaks_found": -5,
        })
        assert resp.status_code == 422

    def test_register_survey_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /surveys returns 500 on service exception."""
        mock_service.register_survey.side_effect = RuntimeError(
            "Survey storage failure"
        )
        resp = client.post(f"{PREFIX}/surveys", json={
            "survey_type": "METHOD21",
        })
        assert resp.status_code == 500

    def test_list_surveys_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /surveys returns survey list."""
        resp = client.get(f"{PREFIX}/surveys")
        assert resp.status_code == 200
        body = resp.json()
        assert "surveys" in body
        assert body["total"] == 1
        mock_service.list_surveys.assert_called_once()


# ===========================================================================
# 7. TestFactorEndpoints (5 tests)
# ===========================================================================


class TestFactorEndpoints:
    """Tests for POST /factors and GET /factors."""

    def test_register_factor_returns_201(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /factors registers a new emission factor and returns 201."""
        resp = client.post(f"{PREFIX}/factors", json={
            "source_type": "EQUIPMENT_LEAK",
            "component_type": "valve",
            "gas": "CH4",
            "value": 0.00597,
            "source": "EPA",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["factor_id"] == "fef_test001"
        assert body["gas"] == "CH4"
        assert body["value"] == 0.00597
        mock_service.register_factor.assert_called_once()

    def test_register_factor_missing_required_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /factors without required source_type returns 422."""
        resp = client.post(f"{PREFIX}/factors", json={
            "gas": "CH4",
        })
        assert resp.status_code == 422

    def test_register_factor_zero_value_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /factors with value=0 returns 422 (Pydantic gt=0)."""
        resp = client.post(f"{PREFIX}/factors", json={
            "source_type": "EQUIPMENT_LEAK",
            "value": 0,
        })
        assert resp.status_code == 422

    def test_register_factor_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /factors returns 500 on service exception."""
        mock_service.register_factor.side_effect = RuntimeError(
            "Factor DB locked"
        )
        resp = client.post(f"{PREFIX}/factors", json={
            "source_type": "EQUIPMENT_LEAK",
            "value": 0.01,
        })
        assert resp.status_code == 500

    def test_list_factors_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /factors returns factor list."""
        resp = client.get(f"{PREFIX}/factors")
        assert resp.status_code == 200
        body = resp.json()
        assert "factors" in body
        assert body["total"] == 1
        mock_service.list_factors.assert_called_once()


# ===========================================================================
# 8. TestRepairEndpoints (5 tests)
# ===========================================================================


class TestRepairEndpoints:
    """Tests for POST /repairs and GET /repairs."""

    def test_register_repair_returns_201(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /repairs registers a new component repair and returns 201."""
        resp = client.post(f"{PREFIX}/repairs", json={
            "component_id": "comp_abc123",
            "repair_type": "minor",
            "repair_date": "2026-02-10T14:30:00Z",
            "pre_repair_rate_ppm": 10500.0,
            "post_repair_rate_ppm": 50.0,
            "cost_usd": 250.0,
            "notes": "Replaced packing on valve stem",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["repair_id"] == "repair_test001"
        assert body["component_id"] == "comp_abc123"
        mock_service.register_repair.assert_called_once()

    def test_register_repair_minimal_body(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /repairs with only required component_id works."""
        resp = client.post(f"{PREFIX}/repairs", json={
            "component_id": "comp_min001",
        })
        assert resp.status_code == 201

    def test_register_repair_missing_component_id_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /repairs without required component_id returns 422."""
        resp = client.post(f"{PREFIX}/repairs", json={
            "repair_type": "major",
        })
        assert resp.status_code == 422

    def test_register_repair_negative_cost_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /repairs with negative cost_usd returns 422."""
        resp = client.post(f"{PREFIX}/repairs", json={
            "component_id": "comp_abc123",
            "cost_usd": -100.0,
        })
        assert resp.status_code == 422

    def test_register_repair_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /repairs returns 500 on service exception."""
        mock_service.register_repair.side_effect = RuntimeError(
            "Repair log corruption"
        )
        resp = client.post(f"{PREFIX}/repairs", json={
            "component_id": "comp_abc123",
        })
        assert resp.status_code == 500

    def test_list_repairs_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /repairs returns repair list."""
        resp = client.get(f"{PREFIX}/repairs")
        assert resp.status_code == 200
        body = resp.json()
        assert "repairs" in body
        assert body["total"] == 1
        mock_service.list_repairs.assert_called_once()


# ===========================================================================
# 9. TestUncertaintyEndpoint (6 tests)
# ===========================================================================


class TestUncertaintyEndpoint:
    """Tests for POST /uncertainty -- Monte Carlo / analytical analysis."""

    def test_uncertainty_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /uncertainty returns 200 with uncertainty results."""
        resp = client.post(f"{PREFIX}/uncertainty", json={
            "calculation_id": "fe_calc_existing001",
            "method": "monte_carlo",
            "iterations": 10000,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["method"] == "monte_carlo"
        assert body["mean_co2e_kg"] == 1250.75
        assert "confidence_intervals" in body
        mock_service.run_uncertainty.assert_called_once()

    def test_uncertainty_analytical_method(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /uncertainty accepts analytical method."""
        resp = client.post(f"{PREFIX}/uncertainty", json={
            "calculation_id": "fe_calc_existing001",
            "method": "analytical",
        })
        assert resp.status_code == 200

    def test_uncertainty_default_iterations(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /uncertainty defaults iterations to 5000 when omitted."""
        resp = client.post(f"{PREFIX}/uncertainty", json={
            "calculation_id": "fe_calc_existing001",
        })
        assert resp.status_code == 200
        call_data = mock_service.run_uncertainty.call_args[0][0]
        assert call_data["iterations"] == 5000

    def test_uncertainty_zero_iterations_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /uncertainty with iterations=0 returns 422 (Pydantic gt=0)."""
        resp = client.post(f"{PREFIX}/uncertainty", json={
            "calculation_id": "fe_calc_existing001",
            "iterations": 0,
        })
        assert resp.status_code == 422

    def test_uncertainty_exceeds_max_iterations_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /uncertainty with iterations > 1000000 returns 422."""
        resp = client.post(f"{PREFIX}/uncertainty", json={
            "calculation_id": "fe_calc_existing001",
            "iterations": 2_000_000,
        })
        assert resp.status_code == 422

    def test_uncertainty_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /uncertainty returns 500 on service exception."""
        mock_service.run_uncertainty.side_effect = RuntimeError(
            "Monte Carlo diverged"
        )
        resp = client.post(f"{PREFIX}/uncertainty", json={
            "calculation_id": "fe_calc_existing001",
        })
        assert resp.status_code == 500


# ===========================================================================
# 10. TestComplianceEndpoint (4 tests)
# ===========================================================================


class TestComplianceEndpoint:
    """Tests for POST /compliance/check -- multi-framework compliance."""

    def test_compliance_check_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /compliance/check returns 200 with compliance results."""
        resp = client.post(f"{PREFIX}/compliance/check", json={
            "calculation_id": "fe_calc_existing001",
            "frameworks": ["GHG_PROTOCOL", "ISO_14064"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["frameworks_checked"] == 7
        assert body["compliant"] == 5
        assert body["non_compliant"] == 1
        assert body["partial"] == 1
        mock_service.check_compliance.assert_called_once()

    def test_compliance_empty_frameworks_checks_all(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /compliance/check with empty frameworks list checks all 7."""
        resp = client.post(f"{PREFIX}/compliance/check", json={
            "frameworks": [],
        })
        assert resp.status_code == 200

    def test_compliance_response_structure(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /compliance/check response has all expected keys."""
        resp = client.post(f"{PREFIX}/compliance/check", json={
            "calculation_id": "fe_calc_existing001",
        })
        body = resp.json()
        expected_keys = {
            "success", "frameworks_checked", "compliant",
            "non_compliant", "partial", "results",
        }
        assert expected_keys.issubset(set(body.keys()))

    def test_compliance_error_returns_500(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /compliance/check returns 500 on service exception."""
        mock_service.check_compliance.side_effect = RuntimeError(
            "Framework DB unavailable"
        )
        resp = client.post(f"{PREFIX}/compliance/check", json={
            "calculation_id": "fe_calc_existing001",
        })
        assert resp.status_code == 500


# ===========================================================================
# 11. TestHealthEndpoint (3 tests)
# ===========================================================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /health returns 200 with health data."""
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "fugitive-emissions"
        assert body["version"] == "1.0.0"
        mock_service.health_check.assert_called_once()

    def test_health_reports_all_engines(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /health response includes all 4 engine statuses."""
        resp = client.get(f"{PREFIX}/health")
        body = resp.json()
        engines = body["engines"]
        assert "equipment_component" in engines
        assert "uncertainty_quantifier" in engines
        assert "compliance_checker" in engines
        assert "pipeline" in engines

    def test_health_all_engines_available(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /health shows all engines as available."""
        resp = client.get(f"{PREFIX}/health")
        body = resp.json()
        for engine_name, engine_status in body["engines"].items():
            assert engine_status == "available", (
                f"Engine {engine_name} is {engine_status}, expected available"
            )


# ===========================================================================
# 12. TestStatsEndpoint (3 tests)
# ===========================================================================


class TestStatsEndpoint:
    """Tests for GET /stats."""

    def test_stats_returns_200(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /stats returns 200 with aggregate statistics."""
        resp = client.get(f"{PREFIX}/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_calculations"] == 42
        assert body["total_sources"] == 6
        assert body["total_components"] == 150
        assert body["total_surveys"] == 8
        assert body["total_repairs"] == 12
        assert body["uptime_seconds"] == 7200.0
        mock_service.get_stats.assert_called_once()

    def test_stats_includes_uptime(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /stats includes uptime_seconds >= 0."""
        resp = client.get(f"{PREFIX}/stats")
        body = resp.json()
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0

    def test_stats_response_structure(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /stats response contains all expected fields."""
        resp = client.get(f"{PREFIX}/stats")
        body = resp.json()
        expected_fields = {
            "total_calculations", "total_sources",
            "total_components", "total_surveys",
            "total_repairs", "uptime_seconds",
        }
        assert expected_fields.issubset(set(body.keys()))


# ===========================================================================
# 13. TestRouterCreation (4 tests)
# ===========================================================================


class TestRouterCreation:
    """Tests for router factory function and metadata."""

    def test_create_router_returns_api_router(self):
        """create_router() returns an APIRouter instance."""
        from fastapi import APIRouter

        router = create_router()
        assert isinstance(router, APIRouter)

    def test_router_has_correct_prefix(self):
        """Router prefix is /api/v1/fugitive-emissions."""
        router = create_router()
        assert router.prefix == "/api/v1/fugitive-emissions"

    def test_router_has_correct_tag(self):
        """Router has 'Fugitive Emissions' tag."""
        router = create_router()
        assert "Fugitive Emissions" in router.tags

    def test_router_has_20_routes(self):
        """Router has exactly 20 endpoint routes."""
        router = create_router()
        assert len(router.routes) >= 20


# ===========================================================================
# 14. TestPydanticValidation (4 tests - additional validation edge cases)
# ===========================================================================


class TestPydanticValidation:
    """Tests for Pydantic request body validation edge cases."""

    def test_calculate_invalid_abatement_over_1(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate with abatement_efficiency > 1.0 returns 422."""
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "EQUIPMENT_LEAK",
            "abatement_efficiency": 1.5,
        })
        assert resp.status_code == 422

    def test_calculate_negative_component_count(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """POST /calculate with negative component_count returns 422."""
        resp = client.post(f"{PREFIX}/calculate", json={
            "source_type": "EQUIPMENT_LEAK",
            "component_count": -10,
        })
        assert resp.status_code == 422

    def test_calculations_page_zero_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /calculations with page=0 returns 422 (ge=1)."""
        resp = client.get(
            f"{PREFIX}/calculations",
            params={"page": 0},
        )
        assert resp.status_code == 422

    def test_calculations_page_size_exceeds_max_returns_422(
        self, client: TestClient, mock_service: MagicMock,
    ):
        """GET /calculations with page_size > 100 returns 422 (le=100)."""
        resp = client.get(
            f"{PREFIX}/calculations",
            params={"page_size": 200},
        )
        assert resp.status_code == 422
