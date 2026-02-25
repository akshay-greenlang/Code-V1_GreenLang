"""
Unit tests for Upstream Transportation FastAPI router endpoints.

Tests all API endpoints for upstream transportation emission calculations:
- POST /calculate
- POST /calculate/batch
- GET /calculations
- GET /calculations/{id}
- DELETE /calculations/{id}
- POST /transport-chains
- GET /transport-chains
- GET /transport-chains/{id}
- GET /emission-factors
- GET /emission-factors/{id}
- POST /emission-factors
- POST /classify
- POST /compliance/check
- GET /compliance/{id}
- POST /uncertainty
- GET /aggregations
- GET /hot-spots
- POST /export
- GET /health
- GET /stats

Tests:
- Successful requests (200, 201)
- Validation errors (422)
- Not found errors (404)
- Bad requests (400)
- Authentication/authorization
"""

import pytest
from decimal import Decimal
from datetime import datetime
from fastapi.testclient import TestClient

from greenlang.main import app
from greenlang.mrv.upstream_transportation.models import (
    CalculationRequest,
    TransportMode,
    VehicleType,
    FuelType,
)


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    # In real tests, this would use actual JWT token
    return {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }


@pytest.fixture
def calculation_payload():
    """Sample calculation request payload."""
    return {
        "calculation_type": "distance_based",
        "mode": "ROAD",
        "vehicle_type": "TRUCK_ARTICULATED_GT33T",
        "distance_km": "500",
        "mass_tonnes": "20.0",
        "origin": "Warehouse A",
        "destination": "Customer B",
        "scope": "WTW",
    }


@pytest.fixture
def transport_chain_payload():
    """Sample transport chain payload."""
    return {
        "chain_id": "CHAIN-001",
        "legs": [
            {
                "mode": "ROAD",
                "vehicle_type": "TRUCK_RIGID_GT17T",
                "distance_km": "100",
                "mass_tonnes": "15.0",
                "origin": "Factory",
                "destination": "Port A",
            },
            {
                "mode": "MARITIME",
                "vehicle_type": "CONTAINER_SHIP_2000_8000TEU",
                "distance_km": "8000",
                "mass_tonnes": "15.0",
                "origin": "Port A",
                "destination": "Port B",
            },
        ],
        "hubs": [
            {"type": "port", "location": "Port A"},
            {"type": "port", "location": "Port B"},
        ],
    }


# ============================================================================
# POST /calculate
# ============================================================================


def test_post_calculate(client, auth_headers, calculation_payload):
    """Test POST /calculate endpoint."""
    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.json()
    assert "calculation_id" in data
    assert "co2e_kg" in data
    assert float(data["co2e_kg"]) > 0
    assert data["mode"] == "ROAD"


# ============================================================================
# POST /calculate/batch
# ============================================================================


def test_post_calculate_batch(client, auth_headers, calculation_payload):
    """Test POST /calculate/batch endpoint."""
    batch_payload = {
        "calculations": [
            calculation_payload,
            {
                "calculation_type": "distance_based",
                "mode": "AIR",
                "distance_km": "2000",
                "mass_tonnes": "5.0",
            },
        ]
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate/batch",
        json=batch_payload,
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert all("calculation_id" in r for r in data["results"])


# ============================================================================
# GET /calculations
# ============================================================================


def test_get_calculations(client, auth_headers, calculation_payload):
    """Test GET /calculations endpoint."""
    # Create a calculation first
    client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )

    # Get calculations
    response = client.get(
        "/api/v1/mrv/upstream-transportation/calculations",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "calculations" in data
    assert len(data["calculations"]) > 0


# ============================================================================
# GET /calculations/{id}
# ============================================================================


def test_get_calculation_by_id(client, auth_headers, calculation_payload):
    """Test GET /calculations/{id} endpoint."""
    # Create calculation
    create_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = create_response.json()["calculation_id"]

    # Get calculation
    response = client.get(
        f"/api/v1/mrv/upstream-transportation/calculations/{calc_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["calculation_id"] == calc_id


# ============================================================================
# DELETE /calculations/{id}
# ============================================================================


def test_delete_calculation(client, auth_headers, calculation_payload):
    """Test DELETE /calculations/{id} endpoint."""
    # Create calculation
    create_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = create_response.json()["calculation_id"]

    # Delete calculation
    response = client.delete(
        f"/api/v1/mrv/upstream-transportation/calculations/{calc_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is True


# ============================================================================
# POST /transport-chains
# ============================================================================


def test_post_transport_chains(client, auth_headers, transport_chain_payload):
    """Test POST /transport-chains endpoint."""
    response = client.post(
        "/api/v1/mrv/upstream-transportation/transport-chains",
        json=transport_chain_payload,
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.json()
    assert data["chain_id"] == "CHAIN-001"
    assert len(data["legs"]) == 2


# ============================================================================
# GET /transport-chains
# ============================================================================


def test_get_transport_chains(client, auth_headers, transport_chain_payload):
    """Test GET /transport-chains endpoint."""
    # Create chain first
    client.post(
        "/api/v1/mrv/upstream-transportation/transport-chains",
        json=transport_chain_payload,
        headers=auth_headers,
    )

    # Get chains
    response = client.get(
        "/api/v1/mrv/upstream-transportation/transport-chains",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "chains" in data
    assert len(data["chains"]) > 0


# ============================================================================
# GET /transport-chains/{id}
# ============================================================================


def test_get_transport_chain_by_id(client, auth_headers, transport_chain_payload):
    """Test GET /transport-chains/{id} endpoint."""
    # Create chain
    create_response = client.post(
        "/api/v1/mrv/upstream-transportation/transport-chains",
        json=transport_chain_payload,
        headers=auth_headers,
    )
    chain_id = create_response.json()["chain_id"]

    # Get chain
    response = client.get(
        f"/api/v1/mrv/upstream-transportation/transport-chains/{chain_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["chain_id"] == chain_id


# ============================================================================
# GET /emission-factors
# ============================================================================


def test_get_emission_factors(client, auth_headers):
    """Test GET /emission-factors endpoint."""
    response = client.get(
        "/api/v1/mrv/upstream-transportation/emission-factors",
        params={"mode": "ROAD", "vehicle_type": "TRUCK_ARTICULATED_GT33T"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "emission_factors" in data
    assert len(data["emission_factors"]) > 0


# ============================================================================
# GET /emission-factors/{id}
# ============================================================================


def test_get_emission_factor_by_id(client, auth_headers):
    """Test GET /emission-factors/{id} endpoint."""
    # Get list first to get an ID
    list_response = client.get(
        "/api/v1/mrv/upstream-transportation/emission-factors",
        headers=auth_headers,
    )
    ef_id = list_response.json()["emission_factors"][0]["ef_id"]

    # Get specific EF
    response = client.get(
        f"/api/v1/mrv/upstream-transportation/emission-factors/{ef_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ef_id"] == ef_id


# ============================================================================
# POST /emission-factors
# ============================================================================


def test_post_custom_emission_factor(client, auth_headers):
    """Test POST /emission-factors endpoint (create custom)."""
    custom_ef_payload = {
        "mode": "ROAD",
        "vehicle_type": "TRUCK_CUSTOM",
        "kg_co2e_per_tonne_km": "0.085",
        "source": "company_specific",
        "scope": "WTW",
        "year": 2023,
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/emission-factors",
        json=custom_ef_payload,
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.json()
    assert "ef_id" in data
    assert data["source"] == "company_specific"


# ============================================================================
# POST /classify
# ============================================================================


def test_post_classify(client, auth_headers):
    """Test POST /classify endpoint."""
    classify_payload = {
        "description": "Truck delivery from warehouse to customer",
        "origin": "Warehouse A",
        "destination": "Customer B",
        "distance_km": "500",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/classify",
        json=classify_payload,
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "ROAD"
    assert "vehicle_type" in data
    assert data["confidence"] > 0.7


# ============================================================================
# POST /compliance/check
# ============================================================================


def test_post_compliance_check(client, auth_headers, calculation_payload):
    """Test POST /compliance/check endpoint."""
    # Create calculation first
    calc_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = calc_response.json()["calculation_id"]

    # Check compliance
    compliance_payload = {
        "calculation_id": calc_id,
        "framework": "GHG_PROTOCOL",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/compliance/check",
        json=compliance_payload,
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["framework"] == "GHG_PROTOCOL"
    assert "status" in data
    assert "score" in data


# ============================================================================
# GET /compliance/{id}
# ============================================================================


def test_get_compliance_result(client, auth_headers, calculation_payload):
    """Test GET /compliance/{id} endpoint."""
    # Create calculation
    calc_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = calc_response.json()["calculation_id"]

    # Check compliance
    client.post(
        "/api/v1/mrv/upstream-transportation/compliance/check",
        json={"calculation_id": calc_id, "framework": "GHG_PROTOCOL"},
        headers=auth_headers,
    )

    # Get compliance result
    response = client.get(
        f"/api/v1/mrv/upstream-transportation/compliance/{calc_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "framework" in data


# ============================================================================
# POST /uncertainty
# ============================================================================


def test_post_uncertainty(client, auth_headers, calculation_payload):
    """Test POST /uncertainty endpoint."""
    # Create calculation
    calc_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = calc_response.json()["calculation_id"]

    # Calculate uncertainty
    uncertainty_payload = {"calculation_id": calc_id}

    response = client.post(
        "/api/v1/mrv/upstream-transportation/uncertainty",
        json=uncertainty_payload,
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "uncertainty_percent" in data
    assert float(data["uncertainty_percent"]) > 0


# ============================================================================
# GET /aggregations
# ============================================================================


def test_get_aggregations(client, auth_headers, calculation_payload):
    """Test GET /aggregations endpoint."""
    # Create calculations
    client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )

    # Get aggregations
    response = client.get(
        "/api/v1/mrv/upstream-transportation/aggregations",
        params={"group_by": "mode,vehicle_type"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "by_mode" in data


# ============================================================================
# GET /hot-spots
# ============================================================================


def test_get_hot_spots(client, auth_headers, calculation_payload):
    """Test GET /hot-spots endpoint."""
    # Create calculation
    client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )

    # Get hot spots
    response = client.get(
        "/api/v1/mrv/upstream-transportation/hot-spots",
        params={"top_n": 10, "metric": "co2e_kg"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "hot_spots" in data
    assert len(data["hot_spots"]) > 0


# ============================================================================
# POST /export
# ============================================================================


def test_post_export(client, auth_headers, calculation_payload):
    """Test POST /export endpoint."""
    # Create calculation
    calc_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = calc_response.json()["calculation_id"]

    # Export report
    export_payload = {
        "calculation_ids": [calc_id],
        "format": "json",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/export",
        json=export_payload,
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "calculations" in data
    assert len(data["calculations"]) == 1


# ============================================================================
# GET /health
# ============================================================================


def test_get_health(client):
    """Test GET /health endpoint."""
    response = client.get("/api/v1/mrv/upstream-transportation/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert "components" in data


# ============================================================================
# GET /stats
# ============================================================================


def test_get_stats(client, auth_headers, calculation_payload):
    """Test GET /stats endpoint."""
    # Create calculation
    client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )

    # Get stats
    response = client.get(
        "/api/v1/mrv/upstream-transportation/stats",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "total_calculations" in data
    assert data["total_calculations"] > 0


# ============================================================================
# Error Cases
# ============================================================================


def test_calculate_invalid_request_422(client, auth_headers):
    """Test POST /calculate with invalid request returns 422."""
    invalid_payload = {
        "calculation_type": "distance_based",
        "mode": "ROAD",
        # Missing required fields (distance_km, mass_tonnes)
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=invalid_payload,
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_calculate_missing_required_field(client, auth_headers):
    """Test missing required field returns 422."""
    incomplete_payload = {
        "calculation_type": "distance_based",
        "mode": "ROAD",
        "distance_km": "500",
        # Missing mass_tonnes
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=incomplete_payload,
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_get_nonexistent_calculation_404(client, auth_headers):
    """Test GET nonexistent calculation returns 404."""
    response = client.get(
        "/api/v1/mrv/upstream-transportation/calculations/NONEXISTENT-ID",
        headers=auth_headers,
    )

    assert response.status_code == 404


def test_batch_too_large_400(client, auth_headers, calculation_payload):
    """Test batch too large returns 400."""
    # Create batch with 1000 calculations (over limit)
    large_batch = {
        "calculations": [calculation_payload for _ in range(1000)]
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate/batch",
        json=large_batch,
        headers=auth_headers,
    )

    # Should reject (assuming 100 limit)
    assert response.status_code == 400


def test_invalid_mode_422(client, auth_headers):
    """Test invalid transport mode returns 422."""
    invalid_payload = {
        "calculation_type": "distance_based",
        "mode": "INVALID_MODE",  # Invalid
        "distance_km": "500",
        "mass_tonnes": "20.0",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=invalid_payload,
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_negative_distance_422(client, auth_headers):
    """Test negative distance returns 422."""
    invalid_payload = {
        "calculation_type": "distance_based",
        "mode": "ROAD",
        "distance_km": "-500",  # Negative
        "mass_tonnes": "20.0",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=invalid_payload,
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_zero_mass_422(client, auth_headers):
    """Test zero mass returns 422."""
    invalid_payload = {
        "calculation_type": "distance_based",
        "mode": "ROAD",
        "distance_km": "500",
        "mass_tonnes": "0",  # Zero
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=invalid_payload,
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_delete_nonexistent_404(client, auth_headers):
    """Test delete nonexistent calculation returns 404."""
    response = client.delete(
        "/api/v1/mrv/upstream-transportation/calculations/NONEXISTENT-ID",
        headers=auth_headers,
    )

    assert response.status_code == 404


def test_unauthorized_request_401(client, calculation_payload):
    """Test unauthorized request returns 401."""
    # No auth headers
    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
    )

    assert response.status_code == 401


def test_invalid_framework_compliance_422(client, auth_headers, calculation_payload):
    """Test invalid framework for compliance returns 422."""
    calc_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = calc_response.json()["calculation_id"]

    invalid_compliance = {
        "calculation_id": calc_id,
        "framework": "INVALID_FRAMEWORK",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/compliance/check",
        json=invalid_compliance,
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_export_invalid_format_422(client, auth_headers, calculation_payload):
    """Test export with invalid format returns 422."""
    calc_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = calc_response.json()["calculation_id"]

    invalid_export = {
        "calculation_ids": [calc_id],
        "format": "invalid_format",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/export",
        json=invalid_export,
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_get_calculations_pagination(client, auth_headers, calculation_payload):
    """Test GET /calculations with pagination."""
    # Create multiple calculations
    for _ in range(5):
        client.post(
            "/api/v1/mrv/upstream-transportation/calculate",
            json=calculation_payload,
            headers=auth_headers,
        )

    # Page 1
    response1 = client.get(
        "/api/v1/mrv/upstream-transportation/calculations",
        params={"limit": 2, "offset": 0},
        headers=auth_headers,
    )

    assert response1.status_code == 200
    assert len(response1.json()["calculations"]) == 2

    # Page 2
    response2 = client.get(
        "/api/v1/mrv/upstream-transportation/calculations",
        params={"limit": 2, "offset": 2},
        headers=auth_headers,
    )

    assert response2.status_code == 200
    assert len(response2.json()["calculations"]) == 2


def test_get_calculations_filter_by_mode(client, auth_headers, calculation_payload):
    """Test GET /calculations filtered by mode."""
    # Create calculation
    client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )

    # Filter by ROAD
    response = client.get(
        "/api/v1/mrv/upstream-transportation/calculations",
        params={"mode": "ROAD"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert all(c["mode"] == "ROAD" for c in data["calculations"])


def test_get_emission_factors_filter_by_source(client, auth_headers):
    """Test GET /emission-factors filtered by source."""
    response = client.get(
        "/api/v1/mrv/upstream-transportation/emission-factors",
        params={"source": "DEFRA"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert all(ef["source"] == "DEFRA" for ef in data["emission_factors"])


def test_post_calculate_with_allocation(client, auth_headers):
    """Test POST /calculate with allocation."""
    allocation_payload = {
        "calculation_type": "distance_based",
        "mode": "ROAD",
        "vehicle_type": "TRUCK_ARTICULATED_GT33T",
        "distance_km": "500",
        "mass_tonnes": "10.0",
        "total_load_mass_tonnes": "20.0",
        "allocation_method": "MASS",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=allocation_payload,
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.json()
    assert float(data["allocation_factor"]) == 0.5


def test_post_calculate_fuel_based(client, auth_headers):
    """Test POST /calculate with fuel-based calculation."""
    fuel_payload = {
        "calculation_type": "fuel_based",
        "mode": "ROAD",
        "fuel_type": "DIESEL",
        "fuel_consumed_liters": "500",
        "distance_km": "2000",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=fuel_payload,
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.json()
    assert float(data["co2e_kg"]) > 0
    assert data["data_quality_tier"] == "TIER_1"  # Fuel is Tier 1


def test_post_calculate_spend_based(client, auth_headers):
    """Test POST /calculate with spend-based calculation."""
    spend_payload = {
        "calculation_type": "spend_based",
        "spend_amount": "15000.00",
        "spend_currency": "USD",
        "spend_year": 2023,
        "naics_code": "484121",  # Trucking
        "transport_mode": "ROAD",
    }

    response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=spend_payload,
        headers=auth_headers,
    )

    assert response.status_code == 201
    data = response.json()
    assert float(data["co2e_kg"]) > 0
    assert data["data_quality_tier"] == "TIER_3"  # Spend is Tier 3


def test_export_multiple_formats(client, auth_headers, calculation_payload):
    """Test export in multiple formats."""
    calc_response = client.post(
        "/api/v1/mrv/upstream-transportation/calculate",
        json=calculation_payload,
        headers=auth_headers,
    )
    calc_id = calc_response.json()["calculation_id"]

    formats = ["json", "csv", "excel"]

    for fmt in formats:
        export_payload = {
            "calculation_ids": [calc_id],
            "format": fmt,
        }

        response = client.post(
            "/api/v1/mrv/upstream-transportation/export",
            json=export_payload,
            headers=auth_headers,
        )

        assert response.status_code == 200
