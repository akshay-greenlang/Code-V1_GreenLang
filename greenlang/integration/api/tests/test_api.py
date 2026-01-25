# -*- coding: utf-8 -*-
"""
greenlang/api/tests/test_api.py

Integration tests for GreenLang Emission Factor API.

Run with: pytest greenlang/api/tests/test_api.py -v --cov=greenlang.api --cov-report=html
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from greenlang.api.main import app


# ==================== TEST FIXTURES ====================

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


# ==================== HEALTH CHECK TESTS ====================

def test_root_endpoint(client):
    """Test root endpoint returns welcome message"""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "GreenLang Emission Factor API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "operational"


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["database"] == "connected"
    assert "uptime_seconds" in data


# ==================== FACTOR QUERY TESTS ====================

def test_list_factors(client):
    """Test listing all factors"""
    response = client.get("/api/v1/factors")
    assert response.status_code == 200

    data = response.json()
    assert "factors" in data
    assert "total_count" in data
    assert "page" in data
    assert data["total_count"] > 0
    assert len(data["factors"]) > 0


def test_list_factors_with_filters(client):
    """Test listing factors with filters"""
    response = client.get("/api/v1/factors?fuel_type=diesel&geography=US")
    assert response.status_code == 200

    data = response.json()
    assert data["total_count"] > 0

    # Verify all results match filters
    for factor in data["factors"]:
        assert factor["fuel_type"] == "diesel"
        assert factor["geography"] == "US"


def test_list_factors_pagination(client):
    """Test pagination"""
    # Get first page
    response1 = client.get("/api/v1/factors?page=1&limit=2")
    assert response1.status_code == 200
    data1 = response1.json()

    # Get second page
    response2 = client.get("/api/v1/factors?page=2&limit=2")
    assert response2.status_code == 200
    data2 = response2.json()

    # Verify pagination
    assert data1["page"] == 1
    assert data2["page"] == 2
    assert len(data1["factors"]) <= 2
    assert len(data2["factors"]) <= 2


def test_get_factor_by_id(client):
    """Test getting specific factor by ID"""
    # First get list of factors
    response = client.get("/api/v1/factors")
    factors = response.json()["factors"]

    # Get first factor by ID
    factor_id = factors[0]["factor_id"]
    response = client.get(f"/api/v1/factors/{factor_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["factor_id"] == factor_id
    assert "co2_per_unit" in data
    assert "ch4_per_unit" in data
    assert "n2o_per_unit" in data
    assert "data_quality" in data
    assert "source" in data


def test_get_factor_not_found(client):
    """Test getting non-existent factor"""
    response = client.get("/api/v1/factors/INVALID_ID")
    assert response.status_code == 404

    data = response.json()
    assert "error" in data or "detail" in data


def test_search_factors(client):
    """Test searching factors"""
    response = client.get("/api/v1/factors/search?q=diesel")
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "diesel"
    assert "factors" in data
    assert data["count"] > 0
    assert "search_time_ms" in data


def test_search_factors_with_geography_filter(client):
    """Test searching with geography filter"""
    response = client.get("/api/v1/factors/search?q=electricity&geography=US")
    assert response.status_code == 200

    data = response.json()
    assert data["count"] > 0

    # Verify all results are from US
    for factor in data["factors"]:
        assert factor["geography"] == "US"


def test_get_factors_by_fuel_type(client):
    """Test getting factors by fuel type"""
    response = client.get("/api/v1/factors/category/diesel")
    assert response.status_code == 200

    data = response.json()
    assert data["total_count"] > 0

    # Verify all are diesel
    for factor in data["factors"]:
        assert factor["fuel_type"] == "diesel"


def test_get_factors_by_scope(client):
    """Test getting factors by scope"""
    response = client.get("/api/v1/factors/scope/1")
    assert response.status_code == 200

    data = response.json()
    assert data["total_count"] > 0

    # Verify all are scope 1
    for factor in data["factors"]:
        assert factor["scope"] == "1"


# ==================== CALCULATION TESTS ====================

def test_calculate_emissions(client):
    """Test single emission calculation"""
    request = {
        "fuel_type": "diesel",
        "activity_amount": 100,
        "activity_unit": "gallons",
        "geography": "US",
        "scope": "1",
        "boundary": "combustion"
    }

    response = client.post("/api/v1/calculate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "calculation_id" in data
    assert data["emissions_kg_co2e"] > 0
    assert data["emissions_tonnes_co2e"] > 0
    assert "emissions_by_gas" in data
    assert "CO2" in data["emissions_by_gas"]
    assert "CH4" in data["emissions_by_gas"]
    assert "N2O" in data["emissions_by_gas"]
    assert "factor_used" in data
    assert data["factor_used"]["fuel_type"] == "diesel"


def test_calculate_emissions_natural_gas(client):
    """Test natural gas calculation"""
    request = {
        "fuel_type": "natural_gas",
        "activity_amount": 500,
        "activity_unit": "therms",
        "geography": "US"
    }

    response = client.post("/api/v1/calculate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["emissions_kg_co2e"] > 0
    assert data["factor_used"]["fuel_type"] == "natural_gas"


def test_calculate_emissions_electricity(client):
    """Test electricity calculation"""
    request = {
        "fuel_type": "electricity",
        "activity_amount": 1000,
        "activity_unit": "kWh",
        "geography": "US",
        "scope": "2"
    }

    response = client.post("/api/v1/calculate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["emissions_kg_co2e"] > 0
    assert data["factor_used"]["scope"] == "2"


def test_calculate_emissions_invalid_fuel(client):
    """Test calculation with invalid fuel type"""
    request = {
        "fuel_type": "invalid_fuel_xyz",
        "activity_amount": 100,
        "activity_unit": "gallons",
        "geography": "US"
    }

    response = client.post("/api/v1/calculate", json=request)
    assert response.status_code == 404


def test_calculate_emissions_negative_amount(client):
    """Test calculation with negative activity amount"""
    request = {
        "fuel_type": "diesel",
        "activity_amount": -100,
        "activity_unit": "gallons",
        "geography": "US"
    }

    response = client.post("/api/v1/calculate", json=request)
    assert response.status_code == 422  # Validation error


def test_batch_calculation(client):
    """Test batch calculation"""
    batch_request = {
        "calculations": [
            {
                "fuel_type": "diesel",
                "activity_amount": 100,
                "activity_unit": "gallons",
                "geography": "US"
            },
            {
                "fuel_type": "natural_gas",
                "activity_amount": 500,
                "activity_unit": "therms",
                "geography": "US"
            },
            {
                "fuel_type": "electricity",
                "activity_amount": 1000,
                "activity_unit": "kWh",
                "geography": "US",
                "scope": "2"
            }
        ]
    }

    response = client.post("/api/v1/calculate/batch", json=batch_request)
    assert response.status_code == 200

    data = response.json()
    assert "batch_id" in data
    assert data["total_emissions_kg_co2e"] > 0
    assert data["count"] == 3
    assert len(data["calculations"]) == 3

    # Verify each calculation has emissions
    for calc in data["calculations"]:
        assert calc["emissions_kg_co2e"] > 0


def test_batch_calculation_max_limit(client):
    """Test batch calculation exceeds max limit"""
    batch_request = {
        "calculations": [
            {
                "fuel_type": "diesel",
                "activity_amount": 1,
                "activity_unit": "gallons",
                "geography": "US"
            }
        ] * 101  # 101 calculations (exceeds max of 100)
    }

    response = client.post("/api/v1/calculate/batch", json=batch_request)
    assert response.status_code == 422  # Validation error


def test_scope1_calculation(client):
    """Test Scope 1 calculation endpoint"""
    request = {
        "fuel_type": "natural_gas",
        "consumption": 500,
        "unit": "therms",
        "geography": "US"
    }

    response = client.post("/api/v1/calculate/scope1", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["emissions_kg_co2e"] > 0
    assert "gas_breakdown" in data
    assert data["gas_breakdown"]["CO2"] > 0


def test_scope2_calculation(client):
    """Test Scope 2 calculation endpoint"""
    request = {
        "electricity_kwh": 10000,
        "geography": "US"
    }

    response = client.post("/api/v1/calculate/scope2", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["emissions_kg_co2e"] > 0
    assert "gas_breakdown" in data


def test_scope2_calculation_market_based(client):
    """Test Scope 2 calculation with market-based factor"""
    request = {
        "electricity_kwh": 10000,
        "geography": "US",
        "market_based_factor": 0.0  # Renewable energy
    }

    response = client.post("/api/v1/calculate/scope2", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["emissions_kg_co2e"] == 0.0


def test_scope3_calculation(client):
    """Test Scope 3 calculation endpoint (not yet implemented)"""
    request = {
        "category": "business_travel",
        "activity_data": {
            "miles_driven": 1000
        },
        "geography": "US"
    }

    response = client.post("/api/v1/calculate/scope3", json=request)
    assert response.status_code == 501  # Not implemented


# ==================== STATISTICS TESTS ====================

def test_get_statistics(client):
    """Test getting API statistics"""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200

    data = response.json()
    assert data["version"] == "1.0.0"
    assert data["total_factors"] > 0
    assert "calculations_today" in data
    assert "cache_stats" in data
    assert "uptime_seconds" in data


def test_get_coverage_stats(client):
    """Test getting coverage statistics"""
    response = client.get("/api/v1/stats/coverage")
    assert response.status_code == 200

    data = response.json()
    assert data["total_factors"] > 0
    assert data["geographies"] > 0
    assert data["fuel_types"] > 0
    assert "scopes" in data
    assert "boundaries" in data
    assert "by_geography" in data
    assert "by_fuel_type" in data


# ==================== OPENAPI TESTS ====================

def test_openapi_schema(client):
    """Test OpenAPI schema is available"""
    response = client.get("/api/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "GreenLang Emission Factor API"
    assert schema["info"]["version"] == "1.0.0"


def test_swagger_docs(client):
    """Test Swagger UI is available"""
    response = client.get("/api/docs")
    assert response.status_code == 200


def test_redoc(client):
    """Test ReDoc is available"""
    response = client.get("/api/redoc")
    assert response.status_code == 200


# ==================== ERROR HANDLING TESTS ====================

def test_404_endpoint(client):
    """Test non-existent endpoint returns 404"""
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404


# ==================== PERFORMANCE TESTS ====================

def test_response_time_list_factors(client):
    """Test response time for listing factors"""
    import time

    start = time.time()
    response = client.get("/api/v1/factors")
    elapsed_ms = (time.time() - start) * 1000

    assert response.status_code == 200
    assert elapsed_ms < 100  # Should be under 100ms


def test_response_time_calculation(client):
    """Test response time for calculation"""
    import time

    request = {
        "fuel_type": "diesel",
        "activity_amount": 100,
        "activity_unit": "gallons",
        "geography": "US"
    }

    start = time.time()
    response = client.post("/api/v1/calculate", json=request)
    elapsed_ms = (time.time() - start) * 1000

    assert response.status_code == 200
    assert elapsed_ms < 50  # Should be under 50ms (95th percentile target)


# ==================== HEADERS TESTS ====================

def test_response_headers(client):
    """Test response includes proper headers"""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time" in response.headers


# ==================== CORS TESTS ====================

def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.options(
        "/api/v1/factors",
        headers={"Origin": "http://localhost:3000"}
    )

    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers or response.status_code == 200
