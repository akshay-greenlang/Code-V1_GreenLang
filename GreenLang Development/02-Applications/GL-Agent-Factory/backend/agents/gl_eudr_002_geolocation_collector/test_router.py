"""
GL-EUDR-002: Router/API Tests

Test suite for FastAPI router endpoints:
- Plot validation endpoints
- Plot CRUD operations
- Bulk upload endpoints
- Statistics and history endpoints
- Authentication and rate limiting

Run with: pytest test_router.py -v
"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# FastAPI test client
try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    TestClient = None
    FastAPI = None

from .agent import (
    GeolocationCollectorAgent,
    CommodityType,
    ValidationStatus,
    CollectionMethod,
    GeometryType,
    BulkJobStatus,
)


# Skip all tests if FastAPI is not available
pytestmark = pytest.mark.skipif(
    not HAS_FASTAPI,
    reason="FastAPI not installed"
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def app():
    """Create a FastAPI app with the router."""
    from .router import router, get_agent

    app = FastAPI()
    app.include_router(router)

    # Override agent dependency
    test_agent = GeolocationCollectorAgent()

    def override_get_agent():
        return test_agent

    app.dependency_overrides[get_agent] = override_get_agent

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def valid_point_data():
    """Valid point coordinate request data."""
    return {
        "type": "point",
        "latitude": -4.123456,
        "longitude": 102.654321
    }


@pytest.fixture
def valid_polygon_data():
    """Valid polygon coordinate request data."""
    return {
        "type": "polygon",
        "coordinates": [
            [102.654321, -4.123456],
            [102.664321, -4.123456],
            [102.664321, -4.133456],
            [102.654321, -4.133456],
            [102.654321, -4.123456],
        ]
    }


@pytest.fixture
def supplier_id():
    """Sample supplier ID."""
    return str(uuid.uuid4())


# =============================================================================
# VALIDATION ENDPOINT TESTS
# =============================================================================

class TestValidationEndpoints:
    """Test /plots/validate endpoint."""

    def test_validate_valid_point(self, client, valid_point_data):
        """Test validation of valid point coordinates."""
        response = client.post(
            "/api/v1/plots/validate",
            json={
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["status"] == "VALID"

    def test_validate_valid_polygon(self, client, valid_polygon_data):
        """Test validation of valid polygon coordinates."""
        response = client.post(
            "/api/v1/plots/validate",
            json={
                "coordinates": valid_polygon_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    def test_validate_with_declared_area(self, client, valid_point_data):
        """Test validation with declared area."""
        response = client.post(
            "/api/v1/plots/validate",
            json={
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL",
                "declared_area_hectares": 5.0
            }
        )

        assert response.status_code == 200
        data = response.json()
        # Should have NEEDS_POLYGON warning for large area with point
        assert any(
            w.get("code") == "NEEDS_POLYGON"
            for w in data.get("warnings", [])
        )

    def test_validate_insufficient_precision(self, client):
        """Test validation of coordinates with insufficient precision."""
        response = client.post(
            "/api/v1/plots/validate",
            json={
                "coordinates": {
                    "type": "point",
                    "latitude": -4.12,
                    "longitude": 102.65
                },
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["status"] == "INVALID"
        assert len(data["errors"]) > 0

    def test_validate_missing_required_fields(self, client, valid_point_data):
        """Test validation with missing required fields."""
        response = client.post(
            "/api/v1/plots/validate",
            json={
                "coordinates": valid_point_data
                # Missing country_code and commodity
            }
        )

        assert response.status_code == 422  # Validation error


# =============================================================================
# PLOT CRUD ENDPOINT TESTS
# =============================================================================

class TestPlotCrudEndpoints:
    """Test plot CRUD endpoints."""

    def test_create_plot(self, client, valid_point_data, supplier_id):
        """Test creating a new plot."""
        response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL",
                "collection_method": "GPS",
                "collection_accuracy_m": 5.0
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert "plot_id" in data
        assert data["supplier_id"] == supplier_id
        assert data["geometry_type"] == "POINT"
        assert data["validation_status"] == "VALID"

    def test_create_polygon_plot(self, client, valid_polygon_data, supplier_id):
        """Test creating a plot with polygon geometry."""
        response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_polygon_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["geometry_type"] == "POLYGON"
        assert data["area_hectares"] is not None
        assert data["centroid"] is not None

    def test_get_plot(self, client, valid_point_data, supplier_id):
        """Test retrieving a plot by ID."""
        # First create a plot
        create_response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )
        plot_id = create_response.json()["plot_id"]

        # Then retrieve it
        response = client.get(f"/api/v1/plots/{plot_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["plot_id"] == plot_id

    def test_get_nonexistent_plot(self, client):
        """Test retrieving a non-existent plot."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/plots/{fake_id}")

        assert response.status_code == 404

    def test_list_plots(self, client, valid_point_data, supplier_id):
        """Test listing plots."""
        # Create a few plots
        for _ in range(3):
            client.post(
                "/api/v1/plots",
                json={
                    "supplier_id": supplier_id,
                    "coordinates": valid_point_data,
                    "country_code": "ID",
                    "commodity": "PALM_OIL"
                }
            )

        # List plots
        response = client.get(f"/api/v1/plots?supplier_id={supplier_id}")

        assert response.status_code == 200
        data = response.json()
        assert "plots" in data
        assert "total_count" in data
        assert data["total_count"] >= 3

    def test_list_plots_with_status_filter(self, client, valid_point_data, supplier_id):
        """Test listing plots filtered by validation status."""
        # Create a valid plot
        client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )

        # List only valid plots
        response = client.get(f"/api/v1/plots?validation_status=VALID")

        assert response.status_code == 200
        data = response.json()
        assert all(p["validation_status"] == "VALID" for p in data["plots"])

    def test_list_plots_pagination(self, client, valid_point_data, supplier_id):
        """Test plot listing pagination."""
        # Create multiple plots
        for i in range(5):
            point = {
                "type": "point",
                "latitude": -4.123456 + (i * 0.001),
                "longitude": 102.654321 + (i * 0.001)
            }
            client.post(
                "/api/v1/plots",
                json={
                    "supplier_id": supplier_id,
                    "coordinates": point,
                    "country_code": "ID",
                    "commodity": "PALM_OIL"
                }
            )

        # Get first page
        response = client.get(f"/api/v1/plots?supplier_id={supplier_id}&limit=2&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["plots"]) == 2

        # Get second page
        response = client.get(f"/api/v1/plots?supplier_id={supplier_id}&limit=2&offset=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["plots"]) == 2


# =============================================================================
# REVALIDATION ENDPOINT TESTS
# =============================================================================

class TestRevalidationEndpoints:
    """Test plot revalidation endpoints."""

    def test_revalidate_plot(self, client, valid_point_data, supplier_id):
        """Test revalidating an existing plot."""
        # Create a plot
        create_response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )
        plot_id = create_response.json()["plot_id"]

        # Revalidate
        response = client.post(f"/api/v1/plots/{plot_id}/validate")

        assert response.status_code == 200
        data = response.json()
        assert "validation_result" in data
        assert data["plot"]["plot_id"] == plot_id


# =============================================================================
# ENRICHMENT ENDPOINT TESTS
# =============================================================================

class TestEnrichmentEndpoints:
    """Test plot enrichment endpoints."""

    def test_enrich_plot(self, client, valid_point_data, supplier_id):
        """Test enriching a plot with geocoding data."""
        # Create a plot
        create_response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )
        plot_id = create_response.json()["plot_id"]

        # Enrich
        response = client.post(f"/api/v1/plots/{plot_id}/enrich")

        assert response.status_code == 200
        data = response.json()
        assert data["plot_id"] == plot_id


# =============================================================================
# HISTORY ENDPOINT TESTS
# =============================================================================

class TestHistoryEndpoints:
    """Test validation history endpoints."""

    def test_get_validation_history(self, client, valid_point_data, supplier_id):
        """Test retrieving validation history."""
        # Create a plot
        create_response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )
        plot_id = create_response.json()["plot_id"]

        # Get history
        response = client.get(f"/api/v1/plots/{plot_id}/history")

        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert len(data["history"]) >= 1


# =============================================================================
# BULK UPLOAD ENDPOINT TESTS
# =============================================================================

class TestBulkUploadEndpoints:
    """Test bulk upload endpoints."""

    def test_initiate_bulk_upload(self, client, supplier_id):
        """Test initiating a bulk upload job."""
        response = client.post(
            "/api/v1/plots/bulk",
            json={
                "supplier_id": supplier_id,
                "file_path": "/uploads/plots.csv",
                "file_format": "csv"
            }
        )

        assert response.status_code == 202  # Accepted
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "QUEUED"

    def test_get_bulk_job_status(self, client, supplier_id):
        """Test retrieving bulk job status."""
        # Create a job
        create_response = client.post(
            "/api/v1/plots/bulk",
            json={
                "supplier_id": supplier_id,
                "file_path": "/uploads/plots.geojson",
                "file_format": "geojson"
            }
        )
        job_id = create_response.json()["job_id"]

        # Get status
        response = client.get(f"/api/v1/bulk/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id


# =============================================================================
# STATISTICS ENDPOINT TESTS
# =============================================================================

class TestStatisticsEndpoints:
    """Test statistics endpoints."""

    def test_get_stats(self, client, valid_point_data, supplier_id):
        """Test retrieving validation statistics."""
        # Create some plots first
        for _ in range(3):
            client.post(
                "/api/v1/plots",
                json={
                    "supplier_id": supplier_id,
                    "coordinates": valid_point_data,
                    "country_code": "ID",
                    "commodity": "PALM_OIL"
                }
            )

        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_plots" in data
        assert "valid_count" in data
        assert "invalid_count" in data


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling."""

    def test_invalid_coordinate_type(self, client, supplier_id):
        """Test error for invalid coordinate type."""
        response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": {
                    "type": "invalid",
                    "latitude": -4.123456,
                    "longitude": 102.654321
                },
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )

        assert response.status_code == 422

    def test_invalid_country_code(self, client, valid_point_data, supplier_id):
        """Test error for invalid country code format."""
        response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "INVALID",  # Should be 2 letters
                "commodity": "PALM_OIL"
            }
        )

        assert response.status_code == 422

    def test_invalid_commodity(self, client, valid_point_data, supplier_id):
        """Test error for invalid commodity type."""
        response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "INVALID_COMMODITY"
            }
        )

        assert response.status_code == 422

    def test_invalid_uuid_format(self, client):
        """Test error for invalid UUID format."""
        response = client.get("/api/v1/plots/not-a-uuid")

        assert response.status_code == 422


# =============================================================================
# RESPONSE FORMAT TESTS
# =============================================================================

class TestResponseFormat:
    """Test response format consistency."""

    def test_plot_response_format(self, client, valid_point_data, supplier_id):
        """Test that plot responses have consistent format."""
        response = client.post(
            "/api/v1/plots",
            json={
                "supplier_id": supplier_id,
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )

        data = response.json()

        # Required fields
        assert "plot_id" in data
        assert "supplier_id" in data
        assert "geometry_type" in data
        assert "country_code" in data
        assert "commodity" in data
        assert "validation_status" in data
        assert "created_at" in data

    def test_validation_response_format(self, client, valid_point_data):
        """Test that validation responses have consistent format."""
        response = client.post(
            "/api/v1/plots/validate",
            json={
                "coordinates": valid_point_data,
                "country_code": "ID",
                "commodity": "PALM_OIL"
            }
        )

        data = response.json()

        # Required fields
        assert "valid" in data
        assert "status" in data
        assert "errors" in data
        assert "warnings" in data

    def test_list_response_format(self, client, supplier_id):
        """Test that list responses have consistent format."""
        response = client.get(f"/api/v1/plots?supplier_id={supplier_id}")

        data = response.json()

        assert "plots" in data
        assert "total_count" in data
        assert isinstance(data["plots"], list)
        assert isinstance(data["total_count"], int)
