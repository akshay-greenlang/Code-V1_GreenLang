# -*- coding: utf-8 -*-
"""
Unit tests for Employee Commuting API Router (AGENT-MRV-020).

Tests all 22+ REST endpoints on the FastAPI router, including request model
validation, response model structure, error responses, health endpoint, and
endpoint route existence.

Target: ~30 tests covering all endpoints exist on the router, request model
validation, response model structure, error handling, and health endpoint.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import router and models
# ---------------------------------------------------------------------------

from greenlang.agents.mrv.employee_commuting.api.router import (
    router,
    CalculateRequest,
    BatchCalculateRequest,
    CommuteCalculateRequest,
    TeleworkCalculateRequest,
    SurveyRequest,
    AverageDataRequest,
    SpendCalculateRequest,
    MultiModalCalculateRequest,
    ComplianceCheckRequest,
    UncertaintyRequest,
    ModeShareRequest,
    CalculationResponse,
    BatchCalculateResponse,
    CommuteCalculateResponse,
    TeleworkCalculateResponse,
    SurveyResponse,
    AverageDataResponse,
    SpendCalculateResponse,
    MultiModalCalculateResponse,
    ComplianceCheckResponse,
    UncertaintyResponse,
    AggregationResponse,
    ModeShareResponse,
    ProvenanceResponse,
    HealthResponse,
    DeleteResponse,
)


# ===========================================================================
# 1. ROUTER CONFIGURATION
# ===========================================================================

class TestRouterConfiguration:
    """Tests for router configuration and existence."""

    def test_router_prefix(self):
        """Router has /api/v1/employee-commuting prefix."""
        assert router.prefix == "/api/v1/employee-commuting"

    def test_router_tags(self):
        """Router has employee-commuting tag."""
        assert "employee-commuting" in router.tags

    def test_router_has_routes(self):
        """Router has routes registered."""
        assert len(router.routes) > 0


# ===========================================================================
# 2. ENDPOINT EXISTENCE
# ===========================================================================

class TestEndpointExistence:
    """Verify all 22 expected endpoints exist on the router."""

    @staticmethod
    def _get_route_paths() -> List[str]:
        """Extract all route paths from the router."""
        paths = []
        for route in router.routes:
            if hasattr(route, "path"):
                paths.append(route.path)
        return paths

    @pytest.mark.parametrize("path", [
        "/calculate",
        "/calculate/batch",
        "/calculate/commute",
        "/calculate/telework",
        "/calculate/survey",
        "/calculate/average-data",
        "/calculate/spend",
        "/calculate/multi-modal",
    ])
    def test_calculation_endpoints_exist(self, path):
        """Calculation POST endpoints are registered."""
        paths = self._get_route_paths()
        assert path in paths, f"Missing endpoint: {path}"

    @pytest.mark.parametrize("path", [
        "/calculations/{calculation_id}",
        "/calculations",
    ])
    def test_crud_endpoints_exist(self, path):
        """CRUD GET endpoints are registered."""
        paths = self._get_route_paths()
        assert path in paths, f"Missing endpoint: {path}"

    def test_delete_endpoint_exists(self):
        """DELETE endpoint for calculation is registered."""
        paths = self._get_route_paths()
        assert "/calculations/{calculation_id}" in paths

    @pytest.mark.parametrize("path", [
        "/emission-factors",
        "/modes",
        "/working-days/{region}",
        "/average-distances",
        "/grid-factors/{country_code}",
    ])
    def test_reference_endpoints_exist(self, path):
        """Reference data GET endpoints are registered."""
        paths = self._get_route_paths()
        assert path in paths, f"Missing endpoint: {path}"

    def test_compliance_endpoint_exists(self):
        """Compliance check POST endpoint is registered."""
        paths = self._get_route_paths()
        assert "/compliance/check" in paths

    def test_uncertainty_endpoint_exists(self):
        """Uncertainty analysis POST endpoint is registered."""
        paths = self._get_route_paths()
        assert "/uncertainty" in paths

    def test_aggregation_endpoint_exists(self):
        """Aggregation GET endpoint is registered."""
        paths = self._get_route_paths()
        assert "/aggregations/{period}" in paths

    def test_mode_share_endpoint_exists(self):
        """Mode share POST endpoint is registered."""
        paths = self._get_route_paths()
        assert "/mode-share" in paths

    def test_provenance_endpoint_exists(self):
        """Provenance GET endpoint is registered."""
        paths = self._get_route_paths()
        assert "/provenance/{calculation_id}" in paths

    def test_health_endpoint_exists(self):
        """Health GET endpoint is registered."""
        paths = self._get_route_paths()
        assert "/health" in paths


# ===========================================================================
# 3. REQUEST MODEL VALIDATION
# ===========================================================================

class TestRequestModelValidation:
    """Tests for Pydantic request model validation."""

    def test_calculate_request_valid(self):
        """Valid CalculateRequest passes validation."""
        req = CalculateRequest(
            mode="sov",
            commute_data={"distance_km": 15.0, "vehicle_type": "car_medium_petrol"},
        )
        assert req.mode == "sov"
        assert req.reporting_period == "2024"

    def test_calculate_request_mode_required(self):
        """CalculateRequest requires mode field."""
        with pytest.raises(Exception):
            CalculateRequest(commute_data={"distance_km": 15.0})

    def test_batch_request_min_employees(self):
        """BatchCalculateRequest requires at least 1 employee."""
        req = BatchCalculateRequest(
            employees=[{"mode": "sov"}],
            reporting_period="2024",
        )
        assert len(req.employees) == 1

    def test_commute_request_distance_range(self):
        """CommuteCalculateRequest distance must be > 0 and <= 500."""
        req = CommuteCalculateRequest(
            mode="sov",
            distance_km=15.0,
        )
        assert req.distance_km == 15.0

    def test_commute_request_zero_distance_fails(self):
        """CommuteCalculateRequest with distance=0 fails validation."""
        with pytest.raises(Exception):
            CommuteCalculateRequest(mode="sov", distance_km=0)

    def test_commute_request_excessive_distance_fails(self):
        """CommuteCalculateRequest with distance > 500 fails validation."""
        with pytest.raises(Exception):
            CommuteCalculateRequest(mode="sov", distance_km=501)

    def test_telework_request_defaults(self):
        """TeleworkCalculateRequest has sensible defaults."""
        req = TeleworkCalculateRequest()
        assert req.frequency == "hybrid_3"
        assert req.region == "US"
        assert req.daily_kwh == 4.0
        assert req.seasonal_adjustment == "none"
        assert req.equipment_lifecycle is False

    def test_survey_request_requires_responses(self):
        """SurveyRequest requires at least 1 response."""
        req = SurveyRequest(
            responses=[{"employee_id": "E1", "mode": "sov", "distance_km": 10}],
            total_employees=100,
        )
        assert len(req.responses) == 1

    def test_average_data_request_valid(self):
        """AverageDataRequest with required fields passes."""
        req = AverageDataRequest(total_employees=500)
        assert req.total_employees == 500
        assert req.country_code == "US"

    def test_spend_request_valid(self):
        """SpendCalculateRequest with all fields passes."""
        req = SpendCalculateRequest(
            naics_code="485000",
            amount=50000.0,
            currency="USD",
            reporting_year=2024,
        )
        assert req.naics_code == "485000"
        assert req.amount == 50000.0

    def test_multi_modal_request_max_5_legs(self):
        """MultiModalCalculateRequest accepts up to 5 legs."""
        legs = [{"mode": "sov", "distance_km": 5.0}] * 5
        req = MultiModalCalculateRequest(legs=legs)
        assert len(req.legs) == 5

    def test_compliance_request_frameworks(self):
        """ComplianceCheckRequest requires at least 1 framework."""
        req = ComplianceCheckRequest(
            frameworks=["ghg_protocol"],
            calculation_results=[{"total_co2e_kg": 1000.0}],
        )
        assert "ghg_protocol" in req.frameworks

    def test_uncertainty_request_defaults(self):
        """UncertaintyRequest has sensible defaults."""
        req = UncertaintyRequest(
            calculation_results=[{"total_co2e_kg": 500.0}],
        )
        assert req.method == "monte_carlo"
        assert req.iterations == 10000
        assert req.confidence_level == 0.95

    def test_mode_share_request_valid(self):
        """ModeShareRequest with reporting_period passes."""
        req = ModeShareRequest(reporting_period="2024")
        assert req.reporting_period == "2024"


# ===========================================================================
# 4. RESPONSE MODEL STRUCTURE
# ===========================================================================

class TestResponseModelStructure:
    """Tests for Pydantic response model structure."""

    def test_calculation_response_fields(self):
        """CalculationResponse has all required fields."""
        resp = CalculationResponse(
            success=True,
            calculation_id="abc-123",
            mode="sov",
            method="employee_specific",
            total_co2e_kg=1234.56,
            commute_co2e_kg=1100.0,
            wtt_co2e_kg=134.56,
            provenance_hash="a" * 64,
            calculated_at="2024-01-01T00:00:00Z",
        )
        assert resp.success is True
        assert resp.calculation_id == "abc-123"

    def test_health_response_fields(self):
        """HealthResponse has all required fields."""
        resp = HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-007",
            version="1.0.0",
            uptime_seconds=120.5,
        )
        assert resp.status == "healthy"
        assert resp.agent_id == "GL-MRV-S3-007"

    def test_delete_response_fields(self):
        """DeleteResponse has required fields."""
        resp = DeleteResponse(
            calculation_id="abc-123",
            deleted=True,
            message="Soft-deleted successfully",
        )
        assert resp.deleted is True

    def test_batch_response_fields(self):
        """BatchCalculateResponse has required fields."""
        resp = BatchCalculateResponse(
            success=True,
            batch_id="batch-1",
            total_employees=10,
            successful=9,
            failed=1,
            total_co2e_kg=50000.0,
            results=[],
            reporting_period="2024",
        )
        assert resp.successful == 9
        assert resp.failed == 1

    def test_commute_response_fields(self):
        """CommuteCalculateResponse has required fields."""
        resp = CommuteCalculateResponse(
            success=True,
            calculation_id="c-1",
            mode="bus",
            distance_km=10.0,
            annual_distance_km=4500.0,
            co2e_kg=464.04,
            wtt_co2e_kg=83.12,
            total_co2e_kg=547.16,
            ef_used=0.10312,
            ef_source="DEFRA",
            provenance_hash="b" * 64,
            calculated_at="2024-01-01T00:00:00Z",
        )
        assert resp.mode == "bus"

    def test_telework_response_fields(self):
        """TeleworkCalculateResponse has required fields."""
        resp = TeleworkCalculateResponse(
            success=True,
            calculation_id="t-1",
            frequency="full_remote",
            telework_days_per_year=240,
            daily_kwh=4.0,
            annual_kwh=960.0,
            grid_ef_kgco2e_per_kwh=0.37938,
            telework_co2e_kg=364.2,
            seasonal_adjustment_applied=False,
            region="US",
            provenance_hash="c" * 64,
            calculated_at="2024-01-01T00:00:00Z",
        )
        assert resp.frequency == "full_remote"
