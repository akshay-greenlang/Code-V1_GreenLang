# -*- coding: utf-8 -*-
"""
Unit tests for Climate Hazard Connector REST API Router - AGENT-DATA-020

Tests all 20 FastAPI endpoints defined in api/router.py using mock
services and the FastAPI TestClient.

Target: 85%+ code coverage across all endpoints and edge cases.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import FastAPI / httpx for TestClient usage
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from greenlang.climate_hazard.api.router import router


# =========================================================================
# Skip all tests if FastAPI is not available
# =========================================================================


pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed; skipping router tests",
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_service():
    """Create a comprehensive mock ClimateHazardService."""
    svc = MagicMock()

    # Source operations
    svc.register_source.return_value = {
        "source_id": "src_001",
        "name": "Test Source",
        "source_type": "noaa",
        "hazard_types": ["flood"],
        "status": "active",
        "region": "US",
        "description": "Test",
        "metadata": {},
        "created_at": "2026-02-17T00:00:00+00:00",
        "updated_at": "2026-02-17T00:00:00+00:00",
        "provenance_hash": "a" * 64,
    }
    svc.list_sources.return_value = [
        {"source_id": "src_001", "name": "S1"},
        {"source_id": "src_002", "name": "S2"},
    ]
    svc.get_source.return_value = {
        "source_id": "src_001",
        "name": "Test Source",
        "source_type": "noaa",
    }

    # Hazard data operations
    svc.ingest_hazard_data.return_value = {
        "record_id": "rec_001",
        "source_id": "src_001",
        "hazard_type": "flood",
        "value": 75.5,
        "provenance_hash": "b" * 64,
    }
    svc.query_hazard_data.return_value = [
        {"record_id": "rec_001", "hazard_type": "flood"},
    ]
    svc.list_hazard_events.return_value = [
        {"event_id": "evt_001", "hazard_type": "flood", "severity": "high"},
    ]

    # Risk index operations
    svc.calculate_risk_index.return_value = {
        "index_id": "idx_001",
        "composite_score": 55.0,
        "risk_classification": "medium",
        "provenance_hash": "c" * 64,
    }
    svc.calculate_multi_hazard.return_value = {
        "assessment_id": "mh_001",
        "composite_score": 62.0,
        "risk_classification": "high",
        "provenance_hash": "d" * 64,
    }
    svc.compare_locations.return_value = {
        "comparison_id": "cmp_001",
        "rankings": [
            {"location_id": "loc_001", "score": 70.0},
            {"location_id": "loc_002", "score": 40.0},
        ],
        "provenance_hash": "e" * 64,
    }

    # Scenario operations
    svc.project_scenario.return_value = {
        "projection_id": "proj_001",
        "projected_value": 85.0,
        "change_percent": 22.0,
        "provenance_hash": "f" * 64,
    }
    svc.list_scenarios.return_value = [
        {"scenario": "SSP2-4.5", "time_horizon": "MID_TERM"},
    ]

    # Asset operations
    svc.register_asset.return_value = {
        "asset_id": "asset_001",
        "name": "London HQ",
        "asset_type": "office",
        "provenance_hash": "a1" * 32,
    }
    svc.list_assets.return_value = [
        {"asset_id": "asset_001", "name": "London HQ"},
    ]

    # Exposure operations
    svc.assess_exposure.return_value = {
        "exposure_id": "exp_001",
        "exposure_score": 60.0,
        "exposure_level": "high",
        "provenance_hash": "b1" * 32,
    }
    svc.assess_portfolio_exposure.return_value = {
        "portfolio_id": "pf_001",
        "overall_exposure_score": 55.0,
        "provenance_hash": "c1" * 32,
    }

    # Vulnerability operations
    svc.score_vulnerability.return_value = {
        "vulnerability_id": "vuln_001",
        "vulnerability_score": 65.0,
        "vulnerability_level": "high",
        "provenance_hash": "d1" * 32,
    }

    # Report operations
    svc.generate_report.return_value = {
        "report_id": "rpt_001",
        "report_type": "physical_risk",
        "format": "json",
        "content": "Report content",
        "provenance_hash": "e1" * 32,
    }
    svc.get_report.return_value = {
        "report_id": "rpt_001",
        "report_type": "physical_risk",
        "content": "Report content",
    }

    # Pipeline operations
    svc.run_pipeline.return_value = {
        "pipeline_id": "pipe_001",
        "overall_status": "completed",
        "stages_completed": 7,
        "provenance_hash": "f1" * 32,
    }

    # Health
    svc.get_health.return_value = {
        "status": "healthy",
        "engines": {
            "hazard_database": "available",
            "risk_index": "available",
            "scenario_projector": "available",
            "exposure_assessor": "available",
            "vulnerability_scorer": "available",
            "compliance_reporter": "available",
            "hazard_pipeline": "available",
        },
        "engines_available": 7,
        "engines_total": 7,
        "timestamp": "2026-02-17T00:00:00+00:00",
    }

    return svc


@pytest.fixture
def client(mock_service) -> TestClient:
    """Create a FastAPI TestClient with the router and mocked service."""
    app = FastAPI()
    app.include_router(router)

    with patch(
        "greenlang.climate_hazard.api.router._get_service",
        return_value=mock_service,
    ):
        yield TestClient(app)


@pytest.fixture
def client_no_service() -> TestClient:
    """Create a TestClient that simulates service not initialized."""
    app = FastAPI()
    app.include_router(router)

    # Import HTTPException from FastAPI
    from fastapi import HTTPException

    def raise_503():
        raise HTTPException(
            status_code=503,
            detail="Climate Hazard Connector service not initialized",
        )

    with patch(
        "greenlang.climate_hazard.api.router._get_service",
        side_effect=raise_503,
    ):
        yield TestClient(app)


# =========================================================================
# Router basics
# =========================================================================


class TestRouterBasics:
    """Tests for router configuration and existence."""

    def test_router_exists(self):
        assert router is not None

    def test_router_has_prefix(self):
        assert router.prefix == "/api/v1/climate-hazard"

    def test_router_has_tag(self):
        assert "Climate Hazard" in router.tags

    def test_router_has_routes(self):
        assert len(router.routes) >= 20


# =========================================================================
# 1. POST /sources
# =========================================================================


class TestRegisterSource:
    """Tests for POST /sources endpoint."""

    def test_register_source_success(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/sources",
            json={"name": "Test Source", "source_type": "noaa"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_id"] == "src_001"
        mock_service.register_source.assert_called_once()

    def test_register_source_minimal(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/sources",
            json={"name": "Minimal"},
        )
        assert response.status_code == 200

    def test_register_source_service_unavailable(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/sources",
            json={"name": "Test"},
        )
        assert response.status_code == 503


# =========================================================================
# 2. GET /sources
# =========================================================================


class TestListSources:
    """Tests for GET /sources endpoint."""

    def test_list_sources(self, client, mock_service):
        response = client.get("/api/v1/climate-hazard/sources")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_list_sources_with_filters(self, client, mock_service):
        response = client.get(
            "/api/v1/climate-hazard/sources",
            params={"hazard_type": "flood", "status": "active", "limit": 50},
        )
        assert response.status_code == 200

    def test_list_sources_with_pagination(self, client, mock_service):
        response = client.get(
            "/api/v1/climate-hazard/sources",
            params={"limit": 10, "offset": 0},
        )
        assert response.status_code == 200


# =========================================================================
# 3. GET /sources/{source_id}
# =========================================================================


class TestGetSource:
    """Tests for GET /sources/{source_id} endpoint."""

    def test_get_source_found(self, client, mock_service):
        response = client.get("/api/v1/climate-hazard/sources/src_001")
        assert response.status_code == 200
        data = response.json()
        assert data["source_id"] == "src_001"

    def test_get_source_not_found(self, client, mock_service):
        mock_service.get_source.return_value = None
        response = client.get("/api/v1/climate-hazard/sources/nonexistent")
        assert response.status_code == 404


# =========================================================================
# 4. POST /hazard-data/ingest
# =========================================================================


class TestIngestHazardData:
    """Tests for POST /hazard-data/ingest endpoint."""

    def test_ingest_success(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/hazard-data/ingest",
            json={
                "source_id": "src_001",
                "hazard_type": "flood",
                "location_id": "loc_001",
                "value": 75.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["record_id"] == "rec_001"

    def test_ingest_minimal(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/hazard-data/ingest",
            json={"hazard_type": "flood"},
        )
        assert response.status_code == 200


# =========================================================================
# 5. GET /hazard-data
# =========================================================================


class TestQueryHazardData:
    """Tests for GET /hazard-data endpoint."""

    def test_query_all(self, client, mock_service):
        response = client.get("/api/v1/climate-hazard/hazard-data")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_query_with_filters(self, client, mock_service):
        response = client.get(
            "/api/v1/climate-hazard/hazard-data",
            params={
                "hazard_type": "flood",
                "source_id": "src_001",
                "location_id": "loc_001",
            },
        )
        assert response.status_code == 200


# =========================================================================
# 6. GET /hazard-data/events
# =========================================================================


class TestListHazardEvents:
    """Tests for GET /hazard-data/events endpoint."""

    def test_list_events(self, client, mock_service):
        response = client.get("/api/v1/climate-hazard/hazard-data/events")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_list_events_with_filters(self, client, mock_service):
        response = client.get(
            "/api/v1/climate-hazard/hazard-data/events",
            params={"hazard_type": "flood", "severity": "high"},
        )
        assert response.status_code == 200


# =========================================================================
# 7. POST /risk-index/calculate
# =========================================================================


class TestCalculateRiskIndex:
    """Tests for POST /risk-index/calculate endpoint."""

    def test_calculate_success(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/risk-index/calculate",
            json={
                "location_id": "loc_001",
                "hazard_type": "flood",
                "scenario": "SSP2-4.5",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["composite_score"] == 55.0

    def test_calculate_minimal(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/risk-index/calculate",
            json={"hazard_type": "flood"},
        )
        assert response.status_code == 200


# =========================================================================
# 8. POST /risk-index/multi-hazard
# =========================================================================


class TestCalculateMultiHazard:
    """Tests for POST /risk-index/multi-hazard endpoint."""

    def test_multi_hazard(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/risk-index/multi-hazard",
            json={
                "location_id": "loc_001",
                "hazard_types": ["flood", "drought"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["composite_score"] == 62.0


# =========================================================================
# 9. POST /risk-index/compare
# =========================================================================


class TestCompareLocations:
    """Tests for POST /risk-index/compare endpoint."""

    def test_compare(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/risk-index/compare",
            json={
                "location_ids": ["loc_001", "loc_002"],
                "hazard_type": "flood",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["rankings"]) == 2


# =========================================================================
# 10. POST /scenarios/project
# =========================================================================


class TestProjectScenario:
    """Tests for POST /scenarios/project endpoint."""

    def test_project_success(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/scenarios/project",
            json={
                "location_id": "loc_001",
                "hazard_type": "flood",
                "scenario": "SSP2-4.5",
                "time_horizon": "MID_TERM",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["projected_value"] == 85.0


# =========================================================================
# 11. GET /scenarios
# =========================================================================


class TestListScenarios:
    """Tests for GET /scenarios endpoint."""

    def test_list_scenarios(self, client, mock_service):
        response = client.get("/api/v1/climate-hazard/scenarios")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_scenarios_with_filters(self, client, mock_service):
        response = client.get(
            "/api/v1/climate-hazard/scenarios",
            params={"scenario": "SSP2-4.5", "time_horizon": "MID_TERM"},
        )
        assert response.status_code == 200


# =========================================================================
# 12. POST /assets
# =========================================================================


class TestRegisterAsset:
    """Tests for POST /assets endpoint."""

    def test_register_asset(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/assets",
            json={
                "name": "London HQ",
                "asset_type": "office",
                "location_id": "loc_london",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["asset_id"] == "asset_001"


# =========================================================================
# 13. GET /assets
# =========================================================================


class TestListAssets:
    """Tests for GET /assets endpoint."""

    def test_list_assets(self, client, mock_service):
        response = client.get("/api/v1/climate-hazard/assets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_list_assets_with_filters(self, client, mock_service):
        response = client.get(
            "/api/v1/climate-hazard/assets",
            params={"asset_type": "office", "location_id": "loc_london"},
        )
        assert response.status_code == 200


# =========================================================================
# 14. POST /exposure/assess
# =========================================================================


class TestAssessExposure:
    """Tests for POST /exposure/assess endpoint."""

    def test_assess_exposure(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/exposure/assess",
            json={
                "asset_id": "asset_001",
                "hazard_type": "flood",
                "scenario": "SSP2-4.5",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["exposure_score"] == 60.0


# =========================================================================
# 15. POST /exposure/portfolio
# =========================================================================


class TestAssessPortfolioExposure:
    """Tests for POST /exposure/portfolio endpoint."""

    def test_assess_portfolio(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/exposure/portfolio",
            json={
                "asset_ids": ["asset_001", "asset_002"],
                "hazard_type": "flood",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["overall_exposure_score"] == 55.0


# =========================================================================
# 16. POST /vulnerability/score
# =========================================================================


class TestScoreVulnerability:
    """Tests for POST /vulnerability/score endpoint."""

    def test_score_vulnerability(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/vulnerability/score",
            json={
                "entity_id": "entity_001",
                "hazard_type": "flood",
                "sector": "manufacturing",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["vulnerability_score"] == 65.0


# =========================================================================
# 17. POST /reports/generate
# =========================================================================


class TestGenerateReport:
    """Tests for POST /reports/generate endpoint."""

    def test_generate_report(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/reports/generate",
            json={
                "report_type": "physical_risk",
                "format": "json",
                "framework": "tcfd",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["report_id"] == "rpt_001"

    def test_generate_report_minimal(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/reports/generate",
            json={},
        )
        assert response.status_code == 200


# =========================================================================
# 18. GET /reports/{report_id}
# =========================================================================


class TestGetReport:
    """Tests for GET /reports/{report_id} endpoint."""

    def test_get_report_found(self, client, mock_service):
        response = client.get("/api/v1/climate-hazard/reports/rpt_001")
        assert response.status_code == 200
        data = response.json()
        assert data["report_id"] == "rpt_001"

    def test_get_report_not_found(self, client, mock_service):
        mock_service.get_report.return_value = None
        response = client.get("/api/v1/climate-hazard/reports/nonexistent")
        assert response.status_code == 404


# =========================================================================
# 19. POST /pipeline/run
# =========================================================================


class TestRunPipeline:
    """Tests for POST /pipeline/run endpoint."""

    def test_run_pipeline(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/pipeline/run",
            json={
                "assets": [
                    {
                        "asset_id": "a1",
                        "name": "HQ",
                        "asset_type": "office",
                        "location": {"lat": 51.5, "lon": -0.13},
                    }
                ],
                "hazard_types": ["flood"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_id"] == "pipe_001"

    def test_run_pipeline_minimal(self, client, mock_service):
        response = client.post(
            "/api/v1/climate-hazard/pipeline/run",
            json={},
        )
        assert response.status_code == 200


# =========================================================================
# 20. GET /health
# =========================================================================


class TestHealthCheck:
    """Tests for GET /health endpoint."""

    def test_health_check(self, client, mock_service):
        response = client.get("/api/v1/climate-hazard/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["engines_available"] == 7

    def test_health_check_degraded(self, client, mock_service):
        mock_service.get_health.return_value = {
            "status": "degraded",
            "engines": {},
            "engines_available": 3,
            "engines_total": 7,
            "timestamp": "2026-02-17T00:00:00+00:00",
        }
        response = client.get("/api/v1/climate-hazard/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"


# =========================================================================
# Service unavailability tests (across multiple endpoints)
# =========================================================================


class TestServiceUnavailable:
    """Tests for service unavailability (503) across endpoints."""

    def test_list_sources_503(self, client_no_service):
        response = client_no_service.get("/api/v1/climate-hazard/sources")
        assert response.status_code == 503

    def test_ingest_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/hazard-data/ingest",
            json={"hazard_type": "flood"},
        )
        assert response.status_code == 503

    def test_query_data_503(self, client_no_service):
        response = client_no_service.get("/api/v1/climate-hazard/hazard-data")
        assert response.status_code == 503

    def test_list_events_503(self, client_no_service):
        response = client_no_service.get("/api/v1/climate-hazard/hazard-data/events")
        assert response.status_code == 503

    def test_calculate_risk_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/risk-index/calculate",
            json={"hazard_type": "flood"},
        )
        assert response.status_code == 503

    def test_multi_hazard_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/risk-index/multi-hazard",
            json={"hazard_types": ["flood"]},
        )
        assert response.status_code == 503

    def test_compare_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/risk-index/compare",
            json={"location_ids": ["loc_001"]},
        )
        assert response.status_code == 503

    def test_project_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/scenarios/project",
            json={"scenario": "SSP2-4.5"},
        )
        assert response.status_code == 503

    def test_list_scenarios_503(self, client_no_service):
        response = client_no_service.get("/api/v1/climate-hazard/scenarios")
        assert response.status_code == 503

    def test_register_asset_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/assets",
            json={"name": "Test"},
        )
        assert response.status_code == 503

    def test_list_assets_503(self, client_no_service):
        response = client_no_service.get("/api/v1/climate-hazard/assets")
        assert response.status_code == 503

    def test_assess_exposure_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/exposure/assess",
            json={"asset_id": "a1"},
        )
        assert response.status_code == 503

    def test_portfolio_exposure_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/exposure/portfolio",
            json={"asset_ids": ["a1"]},
        )
        assert response.status_code == 503

    def test_vulnerability_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/vulnerability/score",
            json={"entity_id": "e1"},
        )
        assert response.status_code == 503

    def test_generate_report_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/reports/generate",
            json={},
        )
        assert response.status_code == 503

    def test_get_report_503(self, client_no_service):
        response = client_no_service.get(
            "/api/v1/climate-hazard/reports/rpt_001",
        )
        assert response.status_code == 503

    def test_run_pipeline_503(self, client_no_service):
        response = client_no_service.post(
            "/api/v1/climate-hazard/pipeline/run",
            json={},
        )
        assert response.status_code == 503

    def test_health_503(self, client_no_service):
        response = client_no_service.get("/api/v1/climate-hazard/health")
        assert response.status_code == 503


# =========================================================================
# Endpoint method verification
# =========================================================================


class TestEndpointMethodVerification:
    """Verify each endpoint calls the correct service method with args."""

    def test_register_source_passes_body(self, client, mock_service):
        body = {"name": "S1", "source_type": "copernicus", "region": "EU"}
        client.post("/api/v1/climate-hazard/sources", json=body)
        mock_service.register_source.assert_called_once_with(**body)

    def test_list_sources_passes_params(self, client, mock_service):
        client.get(
            "/api/v1/climate-hazard/sources",
            params={"hazard_type": "flood", "status": "active", "limit": 50, "offset": 5},
        )
        mock_service.list_sources.assert_called_once_with(
            hazard_type="flood",
            status="active",
            limit=50,
            offset=5,
        )

    def test_get_source_passes_id(self, client, mock_service):
        client.get("/api/v1/climate-hazard/sources/src_123")
        mock_service.get_source.assert_called_once_with("src_123")

    def test_ingest_passes_body(self, client, mock_service):
        body = {"source_id": "s1", "hazard_type": "flood", "value": 42.0}
        client.post("/api/v1/climate-hazard/hazard-data/ingest", json=body)
        mock_service.ingest_hazard_data.assert_called_once_with(**body)

    def test_risk_calculate_passes_body(self, client, mock_service):
        body = {"location_id": "loc_001", "hazard_type": "flood"}
        client.post("/api/v1/climate-hazard/risk-index/calculate", json=body)
        mock_service.calculate_risk_index.assert_called_once_with(**body)

    def test_multi_hazard_passes_body(self, client, mock_service):
        body = {"location_id": "loc_001", "hazard_types": ["flood", "storm"]}
        client.post("/api/v1/climate-hazard/risk-index/multi-hazard", json=body)
        mock_service.calculate_multi_hazard.assert_called_once_with(**body)

    def test_compare_passes_body(self, client, mock_service):
        body = {"location_ids": ["l1", "l2"], "hazard_type": "flood"}
        client.post("/api/v1/climate-hazard/risk-index/compare", json=body)
        mock_service.compare_locations.assert_called_once_with(**body)

    def test_project_passes_body(self, client, mock_service):
        body = {"location_id": "l1", "scenario": "SSP2-4.5"}
        client.post("/api/v1/climate-hazard/scenarios/project", json=body)
        mock_service.project_scenario.assert_called_once_with(**body)

    def test_register_asset_passes_body(self, client, mock_service):
        body = {"name": "A1", "asset_type": "office"}
        client.post("/api/v1/climate-hazard/assets", json=body)
        mock_service.register_asset.assert_called_once_with(**body)

    def test_assess_exposure_passes_body(self, client, mock_service):
        body = {"asset_id": "a1", "hazard_type": "flood"}
        client.post("/api/v1/climate-hazard/exposure/assess", json=body)
        mock_service.assess_exposure.assert_called_once_with(**body)

    def test_portfolio_passes_body(self, client, mock_service):
        body = {"asset_ids": ["a1", "a2"], "hazard_type": "flood"}
        client.post("/api/v1/climate-hazard/exposure/portfolio", json=body)
        mock_service.assess_portfolio_exposure.assert_called_once_with(**body)

    def test_vulnerability_passes_body(self, client, mock_service):
        body = {"entity_id": "e1", "hazard_type": "flood"}
        client.post("/api/v1/climate-hazard/vulnerability/score", json=body)
        mock_service.score_vulnerability.assert_called_once_with(**body)

    def test_generate_report_passes_body(self, client, mock_service):
        body = {"report_type": "physical_risk", "format": "json"}
        client.post("/api/v1/climate-hazard/reports/generate", json=body)
        mock_service.generate_report.assert_called_once_with(**body)

    def test_get_report_passes_id(self, client, mock_service):
        client.get("/api/v1/climate-hazard/reports/rpt_001")
        mock_service.get_report.assert_called_once_with("rpt_001")

    def test_pipeline_passes_body(self, client, mock_service):
        body = {"stages": ["ingest"]}
        client.post("/api/v1/climate-hazard/pipeline/run", json=body)
        mock_service.run_pipeline.assert_called_once_with(**body)

    def test_health_calls_get_health(self, client, mock_service):
        client.get("/api/v1/climate-hazard/health")
        mock_service.get_health.assert_called_once()
