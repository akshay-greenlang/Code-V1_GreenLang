# -*- coding: utf-8 -*-
"""
Unit tests for API Router (api/router.py)

AGENT-MRV-012: Cooling Purchase Agent

Tests FastAPI router endpoints for cooling purchase calculations including
electric chillers, absorption chillers, district cooling, free cooling, TES,
technology specs, emission factors, uncertainty quantification, and compliance
checking using FastAPI TestClient.

Target: 30 tests, ~300 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal

import pytest

# Try importing FastAPI and TestClient
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from greenlang.cooling_purchase.api.router import router
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    pytestmark = pytest.mark.skip("fastapi or cooling_purchase not available")


if FASTAPI_AVAILABLE:
    # Create test app
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/cooling-purchase")
    client = TestClient(app)


# ===========================================================================
# 1. Calculate Electric Chiller Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestCalculateElectricChiller:
    """Test POST /calculate/electric endpoint."""

    def test_calculate_electric_returns_200(self):
        """Test electric chiller calculation returns 200."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/electric",
            json={
                "cooling_kwh_th": "100000",
                "cop": "5.5",
                "grid_ef_kgco2e_kwh": "0.45",
                "tier": "TIER_2",
                "gwp_source": "AR6",
            },
        )
        assert response.status_code == 200

    def test_calculate_electric_returns_result(self):
        """Test electric chiller returns calculation result."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/electric",
            json={
                "cooling_kwh_th": "100000",
                "cop": "5.5",
                "grid_ef_kgco2e_kwh": "0.45",
            },
        )
        data = response.json()
        assert "emissions_kgco2e" in data
        assert float(data["emissions_kgco2e"]) > 0

    def test_calculate_electric_invalid_request_returns_422(self):
        """Test invalid request returns 422."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/electric",
            json={
                "cooling_kwh_th": "invalid",
            },
        )
        assert response.status_code == 422


# ===========================================================================
# 2. Calculate Absorption Cooling Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestCalculateAbsorption:
    """Test POST /calculate/absorption endpoint."""

    def test_calculate_absorption_returns_200(self):
        """Test absorption cooling calculation returns 200."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/absorption",
            json={
                "cooling_kwh_th": "80000",
                "cop_thermal": "1.2",
                "heat_source": "natural_gas",
                "heat_ef_kgco2e_kwh": "0.25",
                "tier": "TIER_2",
                "gwp_source": "AR6",
            },
        )
        assert response.status_code == 200

    def test_calculate_absorption_returns_result(self):
        """Test absorption cooling returns calculation result."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/absorption",
            json={
                "cooling_kwh_th": "80000",
                "cop_thermal": "1.2",
                "heat_source": "natural_gas",
                "heat_ef_kgco2e_kwh": "0.25",
            },
        )
        data = response.json()
        assert "emissions_kgco2e" in data


# ===========================================================================
# 3. Calculate District Cooling Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestCalculateDistrict:
    """Test POST /calculate/district endpoint."""

    def test_calculate_district_returns_200(self):
        """Test district cooling calculation returns 200."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/district",
            json={
                "cooling_kwh_th": "100000",
                "region": "singapore",
                "distribution_loss_pct": "5.0",
                "tier": "TIER_1",
                "gwp_source": "AR6",
            },
        )
        assert response.status_code == 200

    def test_calculate_district_returns_result(self):
        """Test district cooling returns calculation result."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/district",
            json={
                "cooling_kwh_th": "100000",
                "region": "singapore",
            },
        )
        data = response.json()
        assert "emissions_kgco2e" in data


# ===========================================================================
# 4. Calculate Free Cooling Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestCalculateFreeCooling:
    """Test POST /calculate/free-cooling endpoint."""

    def test_calculate_free_cooling_returns_200(self):
        """Test free cooling calculation returns 200."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/free-cooling",
            json={
                "cooling_kwh_th": "50000",
                "source": "seawater",
                "grid_ef_kgco2e_kwh": "0.40",
                "tier": "TIER_2",
                "gwp_source": "AR6",
            },
        )
        assert response.status_code == 200

    def test_calculate_free_cooling_seawater(self):
        """Test free cooling with seawater source."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/free-cooling",
            json={
                "cooling_kwh_th": "50000",
                "source": "seawater",
                "grid_ef_kgco2e_kwh": "0.40",
            },
        )
        data = response.json()
        assert data["technology"] == "SEAWATER_FREE"


# ===========================================================================
# 5. Calculate TES Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestCalculateTES:
    """Test POST /calculate/tes endpoint."""

    def test_calculate_tes_returns_200(self):
        """Test TES calculation returns 200."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/tes",
            json={
                "cooling_kwh_th": "80000",
                "tes_type": "ice",
                "capacity_kwh_th": "20000",
                "cop_charge": "3.0",
                "grid_ef_charge_kgco2e_kwh": "0.30",
                "grid_ef_peak_kgco2e_kwh": "0.60",
                "tier": "TIER_2",
                "gwp_source": "AR6",
            },
        )
        assert response.status_code == 200

    def test_calculate_tes_returns_savings(self):
        """Test TES returns emission savings."""
        response = client.post(
            "/api/v1/cooling-purchase/calculate/tes",
            json={
                "cooling_kwh_th": "80000",
                "tes_type": "ice",
                "capacity_kwh_th": "20000",
                "cop_charge": "3.0",
                "grid_ef_charge_kgco2e_kwh": "0.30",
                "grid_ef_peak_kgco2e_kwh": "0.60",
            },
        )
        data = response.json()
        assert "tes_savings_kgco2e" in data


# ===========================================================================
# 6. Technologies Endpoint Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestTechnologiesEndpoint:
    """Test GET /technologies endpoints."""

    def test_get_technologies_returns_list(self):
        """Test GET /technologies returns list of technologies."""
        response = client.get("/api/v1/cooling-purchase/technologies")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_technology_by_id_returns_spec(self):
        """Test GET /technologies/{tech_id} returns technology spec."""
        response = client.get(
            "/api/v1/cooling-purchase/technologies/WATER_COOLED_CENTRIFUGAL"
        )
        assert response.status_code == 200
        data = response.json()
        assert "cop_min" in data
        assert "cop_max" in data
        assert "iplv" in data

    def test_get_invalid_technology_returns_404(self):
        """Test GET /technologies/{invalid_id} returns 404."""
        response = client.get(
            "/api/v1/cooling-purchase/technologies/INVALID_TECH"
        )
        # Might return 404 or 422 depending on implementation
        assert response.status_code in [404, 422]


# ===========================================================================
# 7. Emission Factors Endpoint Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestEmissionFactorsEndpoint:
    """Test GET /factors endpoints."""

    def test_get_district_ef_singapore(self):
        """Test GET /factors/district/{region} for Singapore."""
        response = client.get("/api/v1/cooling-purchase/factors/district/singapore")
        assert response.status_code == 200
        data = response.json()
        assert "emission_factor_kgco2e_kwh" in data

    def test_get_district_ef_dubai(self):
        """Test GET /factors/district/{region} for Dubai."""
        response = client.get("/api/v1/cooling-purchase/factors/district/dubai")
        assert response.status_code == 200

    def test_get_refrigerants_list(self):
        """Test GET /factors/refrigerants returns list."""
        response = client.get("/api/v1/cooling-purchase/factors/refrigerants")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


# ===========================================================================
# 8. Uncertainty Endpoint Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestUncertaintyEndpoint:
    """Test POST /uncertainty endpoint."""

    def test_post_uncertainty_returns_result(self):
        """Test POST /uncertainty returns uncertainty result."""
        response = client.post(
            "/api/v1/cooling-purchase/uncertainty",
            json={
                "total_emissions_kgco2e": "10000",
                "cooling_kwh_th": "100000",
                "cop": "5.0",
                "tier": "TIER_1",
                "technology": "WATER_COOLED_CENTRIFUGAL",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "uncertainty_pct" in data


# ===========================================================================
# 9. Compliance Endpoint Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestComplianceEndpoint:
    """Test POST /compliance/check endpoint."""

    def test_post_compliance_check_returns_results(self):
        """Test POST /compliance/check returns compliance results."""
        response = client.post(
            "/api/v1/cooling-purchase/compliance/check",
            json={
                "technology": "WATER_COOLED_CENTRIFUGAL",
                "cooling_kwh_th": "100000",
                "emissions_kgco2e": "10000",
                "cop": "5.5",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "framework_results" in data

    def test_get_compliance_frameworks_returns_7(self):
        """Test GET /compliance/frameworks returns 7 frameworks."""
        response = client.get("/api/v1/cooling-purchase/compliance/frameworks")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 7


# ===========================================================================
# 10. Health Endpoint Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_get_health_returns_status(self):
        """Test GET /health returns health status."""
        response = client.get("/api/v1/cooling-purchase/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


# ===========================================================================
# 11. Batch Endpoint Tests
# ===========================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestBatchEndpoint:
    """Test POST /batch endpoint."""

    def test_batch_with_empty_list(self):
        """Test batch with empty list."""
        response = client.post(
            "/api/v1/cooling-purchase/batch",
            json={
                "calculations": [],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_calculations"] == 0

    def test_batch_with_single_calculation(self):
        """Test batch with single calculation."""
        response = client.post(
            "/api/v1/cooling-purchase/batch",
            json={
                "calculations": [
                    {
                        "type": "electric",
                        "cooling_kwh_th": "100000",
                        "cop": "5.5",
                        "grid_ef_kgco2e_kwh": "0.45",
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_calculations"] == 1

    def test_batch_with_multiple_calculations(self):
        """Test batch with multiple calculations."""
        response = client.post(
            "/api/v1/cooling-purchase/batch",
            json={
                "calculations": [
                    {
                        "type": "electric",
                        "cooling_kwh_th": "100000",
                        "cop": "5.5",
                        "grid_ef_kgco2e_kwh": "0.45",
                    },
                    {
                        "type": "district",
                        "cooling_kwh_th": "80000",
                        "region": "singapore",
                    },
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_calculations"] == 2
