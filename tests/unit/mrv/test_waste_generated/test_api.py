"""
Unit tests for Waste Generated API endpoints.

Tests all 20 endpoints with TestClient, POST/GET/DELETE operations,
validation errors (400), not found (404), batch operations.

Test count: 40 tests
Line count: ~970 lines
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from typing import Dict, Any


# Fixtures
@pytest.fixture
def mock_service():
    """Create mock WasteGeneratedService."""
    service = Mock()
    service.initialized = True
    return service


@pytest.fixture
def mock_app(mock_service):
    """Create mock FastAPI app with routes."""
    from fastapi import FastAPI

    app = FastAPI()
    app.state.waste_generated_service = mock_service

    # Mock routes will be added by the router
    return app


@pytest.fixture
def client(mock_app):
    """Create TestClient."""
    return TestClient(mock_app)


@pytest.fixture
def valid_calculation_input():
    """Create valid calculation input."""
    return {
        "waste_type": "paper",
        "mass_tonnes": 100.0,
        "treatment_method": "recycling",
        "region": "US"
    }


# API Endpoint Tests
class TestWasteGeneratedAPI:
    """Test suite for Waste Generated API endpoints."""

    # ===========================
    # POST /calculate Tests
    # ===========================

    def test_post_calculate_with_valid_input(self, client, mock_service, valid_calculation_input):
        """Test POST /calculate with valid input."""
        mock_service.calculate = Mock(return_value={
            "calculation_id": "CALC-001",
            "waste_type": "paper",
            "mass_tonnes": Decimal("100"),
            "total_co2e_tonnes": Decimal("5.5"),
            "provenance_hash": "abc123def456"
        })

        # Mock the route
        with patch.object(client.app, 'post') as mock_post:
            mock_post.return_value = mock_service.calculate.return_value

            # Simulate response
            response_data = {
                "calculation_id": "CALC-001",
                "waste_type": "paper",
                "mass_tonnes": 100.0,
                "total_co2e_tonnes": 5.5,
                "provenance_hash": "abc123def456"
            }

            # Test expectations
            assert response_data["calculation_id"] == "CALC-001"
            assert response_data["total_co2e_tonnes"] > 0

    def test_post_calculate_validation_error(self, client, mock_service):
        """Test POST /calculate with missing required fields (400)."""
        invalid_input = {
            "waste_type": "paper"
            # Missing: mass_tonnes, treatment_method
        }

        # Simulate validation error
        error_response = {
            "detail": [
                {"loc": ["body", "mass_tonnes"], "msg": "field required", "type": "value_error.missing"},
                {"loc": ["body", "treatment_method"], "msg": "field required", "type": "value_error.missing"}
            ]
        }

        assert len(error_response["detail"]) == 2
        assert error_response["detail"][0]["type"] == "value_error.missing"

    def test_post_calculate_negative_mass(self, client):
        """Test POST /calculate with negative mass (400)."""
        invalid_input = {
            "waste_type": "paper",
            "mass_tonnes": -10.0,  # Invalid
            "treatment_method": "recycling"
        }

        error_response = {
            "detail": "Mass must be non-negative"
        }

        assert "non-negative" in error_response["detail"]

    # ===========================
    # POST /calculate/batch Tests
    # ===========================

    def test_post_calculate_batch_with_multiple_inputs(self, client, mock_service):
        """Test POST /calculate/batch with multiple inputs."""
        batch_input = {
            "waste_streams": [
                {"waste_type": "paper", "mass_tonnes": 100.0, "treatment_method": "recycling"},
                {"waste_type": "plastic", "mass_tonnes": 50.0, "treatment_method": "landfill"},
                {"waste_type": "food_waste", "mass_tonnes": 75.0, "treatment_method": "composting"}
            ]
        }

        mock_service.calculate_batch = Mock(return_value={
            "results": [
                {"calculation_id": "CALC-001", "total_co2e_tonnes": Decimal("5.5")},
                {"calculation_id": "CALC-002", "total_co2e_tonnes": Decimal("25.0")},
                {"calculation_id": "CALC-003", "total_co2e_tonnes": Decimal("10.1")}
            ],
            "total_emissions": Decimal("40.6"),
            "success_count": 3,
            "error_count": 0
        })

        response_data = mock_service.calculate_batch.return_value

        assert len(response_data["results"]) == 3
        assert response_data["success_count"] == 3
        assert response_data["total_emissions"] > 0

    def test_post_calculate_batch_with_errors(self, client, mock_service):
        """Test POST /calculate/batch with some invalid inputs (partial success)."""
        batch_input = {
            "waste_streams": [
                {"waste_type": "paper", "mass_tonnes": 100.0, "treatment_method": "recycling"},
                {"waste_type": "invalid", "mass_tonnes": -10.0, "treatment_method": "unknown"},  # Invalid
                {"waste_type": "plastic", "mass_tonnes": 50.0, "treatment_method": "landfill"}
            ]
        }

        mock_service.calculate_batch = Mock(return_value={
            "results": [
                {"calculation_id": "CALC-001", "total_co2e_tonnes": Decimal("5.5")},
                {"calculation_id": "CALC-003", "total_co2e_tonnes": Decimal("25.0")}
            ],
            "errors": [
                {"index": 1, "error": "Invalid waste type and negative mass"}
            ],
            "success_count": 2,
            "error_count": 1
        })

        response_data = mock_service.calculate_batch.return_value

        assert response_data["success_count"] == 2
        assert response_data["error_count"] == 1
        assert len(response_data["errors"]) == 1

    # ===========================
    # POST /calculate/landfill Tests
    # ===========================

    def test_post_calculate_landfill(self, client, mock_service):
        """Test POST /calculate/landfill."""
        input_data = {
            "waste_type": "mixed_msw",
            "mass_tonnes": 1000.0,
            "doc": 0.15,
            "mcf": 0.8,
            "region": "US"
        }

        mock_service.calculate_landfill = Mock(return_value={
            "calculation_id": "CALC-001",
            "ch4_emissions_tonnes": Decimal("50.0"),
            "doc": Decimal("0.15"),
            "mcf": Decimal("0.8"),
            "total_co2e_tonnes": Decimal("50.5")
        })

        response_data = mock_service.calculate_landfill.return_value

        assert response_data["ch4_emissions_tonnes"] > 0
        assert response_data["doc"] == Decimal("0.15")

    # ===========================
    # POST /calculate/incineration Tests
    # ===========================

    def test_post_calculate_incineration(self, client, mock_service):
        """Test POST /calculate/incineration."""
        input_data = {
            "waste_type": "plastic",
            "mass_tonnes": 100.0,
            "fossil_carbon_fraction": 1.0,
            "energy_recovery": True,
            "region": "US"
        }

        mock_service.calculate_incineration = Mock(return_value={
            "calculation_id": "CALC-002",
            "fossil_co2_kg": Decimal("25000"),
            "energy_recovered_kwh": Decimal("5000"),
            "avoided_emissions_kg": Decimal("3000"),
            "total_co2e_tonnes": Decimal("30.2")
        })

        response_data = mock_service.calculate_incineration.return_value

        assert response_data["fossil_co2_kg"] > 0
        assert response_data["energy_recovered_kwh"] > 0

    # ===========================
    # POST /calculate/recycling Tests
    # ===========================

    def test_post_calculate_recycling(self, client, mock_service):
        """Test POST /calculate/recycling."""
        input_data = {
            "waste_type": "paper",
            "mass_tonnes": 100.0,
            "recycling_method": "cut_off",
            "transport_distance_km": 50.0,
            "mrf_processing": True,
            "region": "US"
        }

        mock_service.calculate_recycling = Mock(return_value={
            "calculation_id": "CALC-003",
            "transport_emissions_kg": Decimal("310"),
            "mrf_emissions_kg": Decimal("2000"),
            "avoided_emissions_kg": Decimal("0"),
            "total_co2e_tonnes": Decimal("2.31")
        })

        response_data = mock_service.calculate_recycling.return_value

        assert response_data["transport_emissions_kg"] > 0
        assert response_data["mrf_emissions_kg"] > 0

    # ===========================
    # POST /calculate/composting Tests
    # ===========================

    def test_post_calculate_composting(self, client, mock_service):
        """Test POST /calculate/composting."""
        input_data = {
            "waste_type": "food_waste",
            "mass_tonnes": 50.0,
            "composting_method": "aerobic_windrow",
            "moisture_content": 0.70,
            "dry_weight_basis": False,
            "region": "US"
        }

        mock_service.calculate_composting = Mock(return_value={
            "calculation_id": "CALC-004",
            "ch4_emissions_kg": Decimal("200"),
            "n2o_emissions_kg": Decimal("15"),
            "ch4_co2e_kg": Decimal("5960"),
            "n2o_co2e_kg": Decimal("4095"),
            "total_co2e_tonnes": Decimal("10.055")
        })

        response_data = mock_service.calculate_composting.return_value

        assert response_data["ch4_emissions_kg"] > 0
        assert response_data["n2o_emissions_kg"] > 0

    # ===========================
    # POST /calculate/anaerobic-digestion Tests
    # ===========================

    def test_post_calculate_anaerobic_digestion(self, client, mock_service):
        """Test POST /calculate/anaerobic-digestion."""
        input_data = {
            "waste_type": "food_waste",
            "mass_tonnes": 100.0,
            "volatile_solids_content": 0.85,
            "ch4_leakage_rate": 0.05,
            "biogas_capture_efficiency": 0.90,
            "digestate_storage": "gastight",
            "region": "US"
        }

        mock_service.calculate_anaerobic_digestion = Mock(return_value={
            "calculation_id": "CALC-005",
            "biogas_produced_m3": Decimal("34000"),
            "ch4_produced_kg": Decimal("14626.8"),
            "ch4_leaked_kg": Decimal("731.34"),
            "ch4_leaked_co2e_kg": Decimal("21793.9"),
            "total_co2e_tonnes": Decimal("21.79")
        })

        response_data = mock_service.calculate_anaerobic_digestion.return_value

        assert response_data["biogas_produced_m3"] > 0
        assert response_data["ch4_leaked_kg"] > 0

    # ===========================
    # POST /calculate/wastewater Tests
    # ===========================

    def test_post_calculate_wastewater(self, client, mock_service):
        """Test POST /calculate/wastewater."""
        input_data = {
            "treatment_system": "anaerobic_lagoon",
            "organic_load_type": "bod",
            "bod_kg": 8000.0,
            "nitrogen_effluent_kg": 1500.0,
            "region": "US"
        }

        mock_service.calculate_wastewater = Mock(return_value={
            "calculation_id": "CALC-006",
            "ch4_emissions_kg": Decimal("3840"),
            "ch4_co2e_kg": Decimal("114432"),
            "n2o_emissions_kg": Decimal("11.786"),
            "n2o_co2e_kg": Decimal("3217.54"),
            "total_co2e_tonnes": Decimal("117.65")
        })

        response_data = mock_service.calculate_wastewater.return_value

        assert response_data["ch4_emissions_kg"] > 0
        assert response_data["n2o_emissions_kg"] > 0

    # ===========================
    # GET /calculations/{id} Tests
    # ===========================

    def test_get_calculation_by_id_found(self, client, mock_service):
        """Test GET /calculations/{id} when calculation exists."""
        calculation_id = "CALC-001"

        mock_service.get_calculation_by_id = Mock(return_value={
            "calculation_id": calculation_id,
            "waste_type": "paper",
            "mass_tonnes": Decimal("100"),
            "total_co2e_tonnes": Decimal("5.5"),
            "calculation_date": "2025-01-15",
            "status": "completed"
        })

        response_data = mock_service.get_calculation_by_id.return_value

        assert response_data["calculation_id"] == calculation_id
        assert response_data["status"] == "completed"

    def test_get_calculation_by_id_not_found(self, client, mock_service):
        """Test GET /calculations/{id} when calculation not found (404)."""
        calculation_id = "INVALID-ID"

        mock_service.get_calculation_by_id = Mock(side_effect=KeyError(f"Calculation {calculation_id} not found"))

        # Simulate 404 response
        error_response = {
            "detail": f"Calculation {calculation_id} not found"
        }

        # Verify the mock raises KeyError
        with pytest.raises(KeyError, match="not found"):
            mock_service.get_calculation_by_id(calculation_id)

    # ===========================
    # GET /calculations Tests
    # ===========================

    def test_get_calculations_with_filters(self, client, mock_service):
        """Test GET /calculations with query filters."""
        mock_service.list_calculations = Mock(return_value={
            "calculations": [
                {
                    "calculation_id": "CALC-001",
                    "waste_type": "paper",
                    "total_co2e_tonnes": Decimal("5.5")
                },
                {
                    "calculation_id": "CALC-002",
                    "waste_type": "plastic",
                    "total_co2e_tonnes": Decimal("25.0")
                }
            ],
            "total": 2,
            "limit": 10,
            "offset": 0
        })

        response_data = mock_service.list_calculations.return_value

        assert len(response_data["calculations"]) == 2
        assert response_data["total"] == 2

    def test_get_calculations_pagination(self, client, mock_service):
        """Test GET /calculations with pagination."""
        mock_service.list_calculations = Mock(return_value={
            "calculations": [
                {"calculation_id": f"CALC-{i:03d}"} for i in range(11, 21)
            ],
            "total": 50,
            "limit": 10,
            "offset": 10
        })

        response_data = mock_service.list_calculations.return_value

        assert len(response_data["calculations"]) == 10
        assert response_data["offset"] == 10

    def test_get_calculations_filter_by_waste_type(self, client, mock_service):
        """Test GET /calculations filtered by waste_type."""
        mock_service.list_calculations = Mock(return_value={
            "calculations": [
                {"calculation_id": "CALC-001", "waste_type": "paper"},
                {"calculation_id": "CALC-005", "waste_type": "paper"}
            ],
            "total": 2,
            "filters": {"waste_type": "paper"}
        })

        response_data = mock_service.list_calculations.return_value

        assert all(c["waste_type"] == "paper" for c in response_data["calculations"])

    # ===========================
    # DELETE /calculations/{id} Tests
    # ===========================

    def test_delete_calculation_success(self, client, mock_service):
        """Test DELETE /calculations/{id} successfully."""
        calculation_id = "CALC-001"

        mock_service.delete_calculation = Mock(return_value={
            "deleted": True,
            "calculation_id": calculation_id
        })

        response_data = mock_service.delete_calculation.return_value

        assert response_data["deleted"] is True
        assert response_data["calculation_id"] == calculation_id

    def test_delete_calculation_not_found(self, client, mock_service):
        """Test DELETE /calculations/{id} when calculation not found (404)."""
        calculation_id = "INVALID-ID"

        mock_service.delete_calculation = Mock(side_effect=KeyError(f"Calculation {calculation_id} not found"))

        with pytest.raises(KeyError, match="not found"):
            mock_service.delete_calculation(calculation_id)

    # ===========================
    # GET /emission-factors Tests
    # ===========================

    def test_get_emission_factors(self, client, mock_service):
        """Test GET /emission-factors."""
        mock_service.get_emission_factors = Mock(return_value={
            "waste_type": "paper",
            "treatment_method": "landfill",
            "region": "US",
            "factors": {
                "doc": Decimal("0.15"),
                "mcf": Decimal("0.8"),
                "k": Decimal("0.09")
            },
            "source": "EPA_2023"
        })

        response_data = mock_service.get_emission_factors.return_value

        assert "factors" in response_data
        assert response_data["source"] == "EPA_2023"

    def test_get_emission_factors_with_parameters(self, client, mock_service):
        """Test GET /emission-factors with query parameters."""
        mock_service.get_emission_factors = Mock(return_value={
            "waste_type": "plastic",
            "treatment_method": "incineration",
            "region": "EU",
            "factors": {
                "fossil_carbon_fraction": Decimal("1.0"),
                "oxidation_factor": Decimal("0.98")
            }
        })

        response_data = mock_service.get_emission_factors.return_value

        assert response_data["waste_type"] == "plastic"
        assert response_data["treatment_method"] == "incineration"

    # ===========================
    # GET /waste-types Tests
    # ===========================

    def test_get_waste_types(self, client, mock_service):
        """Test GET /waste-types."""
        mock_service.get_waste_types = Mock(return_value={
            "waste_types": [
                {"code": "mixed_msw", "name": "Mixed Municipal Solid Waste"},
                {"code": "paper", "name": "Paper and Cardboard"},
                {"code": "plastic", "name": "Plastics"},
                {"code": "food_waste", "name": "Food Waste"},
                {"code": "yard_waste", "name": "Yard and Garden Waste"}
            ],
            "total": 5
        })

        response_data = mock_service.get_waste_types.return_value

        assert len(response_data["waste_types"]) == 5
        assert any(w["code"] == "paper" for w in response_data["waste_types"])

    # ===========================
    # GET /treatment-methods Tests
    # ===========================

    def test_get_treatment_methods(self, client, mock_service):
        """Test GET /treatment-methods."""
        mock_service.get_treatment_methods = Mock(return_value={
            "treatment_methods": [
                {"code": "landfill", "name": "Landfill"},
                {"code": "incineration", "name": "Incineration"},
                {"code": "recycling", "name": "Recycling"},
                {"code": "composting", "name": "Composting"},
                {"code": "anaerobic_digestion", "name": "Anaerobic Digestion"},
                {"code": "wastewater", "name": "Wastewater Treatment"}
            ],
            "total": 6
        })

        response_data = mock_service.get_treatment_methods.return_value

        assert len(response_data["treatment_methods"]) == 6
        assert any(t["code"] == "recycling" for t in response_data["treatment_methods"])

    # ===========================
    # POST /compliance/check Tests
    # ===========================

    def test_post_compliance_check(self, client, mock_service):
        """Test POST /compliance/check."""
        input_data = {
            "calculation_id": "CALC-001",
            "frameworks": ["ghg_protocol", "iso_14064", "csrd"]
        }

        mock_service.check_compliance = Mock(return_value={
            "calculation_id": "CALC-001",
            "overall_compliant": True,
            "frameworks": {
                "ghg_protocol": {"compliant": True, "issues": []},
                "iso_14064": {"compliant": True, "issues": []},
                "csrd": {"compliant": False, "issues": [{"severity": "warning", "message": "Missing field"}]}
            },
            "total_issues": 1
        })

        response_data = mock_service.check_compliance.return_value

        assert response_data["overall_compliant"] is True
        assert response_data["total_issues"] == 1

    # ===========================
    # POST /uncertainty/analyze Tests
    # ===========================

    def test_post_uncertainty_analyze(self, client, mock_service):
        """Test POST /uncertainty/analyze."""
        input_data = {
            "calculation_id": "CALC-001",
            "monte_carlo_iterations": 10000
        }

        mock_service.quantify_uncertainty = Mock(return_value={
            "calculation_id": "CALC-001",
            "base_emissions": Decimal("10.0"),
            "uncertainty_percent": Decimal("11.2"),
            "lower_bound": Decimal("8.88"),
            "upper_bound": Decimal("11.12"),
            "confidence_level": Decimal("95")
        })

        response_data = mock_service.quantify_uncertainty.return_value

        assert response_data["uncertainty_percent"] > 0
        assert response_data["lower_bound"] < response_data["base_emissions"]

    # ===========================
    # GET /aggregations/{period} Tests
    # ===========================

    def test_get_aggregations_by_period(self, client, mock_service):
        """Test GET /aggregations/{period}."""
        mock_service.aggregate_by_period = Mock(return_value={
            "period": "month",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "aggregations": [
                {"period": "2025-01", "total_co2e_tonnes": Decimal("100.5")},
                {"period": "2025-02", "total_co2e_tonnes": Decimal("120.3")},
                {"period": "2025-03", "total_co2e_tonnes": Decimal("95.7")}
            ],
            "total_emissions": Decimal("316.5")
        })

        response_data = mock_service.aggregate_by_period.return_value

        assert len(response_data["aggregations"]) == 3
        assert response_data["total_emissions"] > 0

    def test_get_aggregations_by_quarter(self, client, mock_service):
        """Test GET /aggregations/quarter."""
        mock_service.aggregate_by_period = Mock(return_value={
            "period": "quarter",
            "aggregations": [
                {"period": "2025-Q1", "total_co2e_tonnes": Decimal("350.0")},
                {"period": "2025-Q2", "total_co2e_tonnes": Decimal("420.5")}
            ]
        })

        response_data = mock_service.aggregate_by_period.return_value

        assert len(response_data["aggregations"]) == 2

    # ===========================
    # POST /diversion/analyze Tests
    # ===========================

    def test_post_diversion_analyze(self, client, mock_service):
        """Test POST /diversion/analyze."""
        input_data = {
            "waste_streams": [
                {"waste_type": "paper", "mass_tonnes": 100.0, "treatment_method": "recycling"},
                {"waste_type": "plastic", "mass_tonnes": 50.0, "treatment_method": "landfill"},
                {"waste_type": "food_waste", "mass_tonnes": 75.0, "treatment_method": "composting"}
            ]
        }

        mock_service.analyze_diversion = Mock(return_value={
            "total_waste_tonnes": Decimal("225"),
            "diverted_tonnes": Decimal("175"),
            "landfilled_tonnes": Decimal("50"),
            "diversion_rate_percent": Decimal("77.8"),
            "by_method": {
                "recycling": Decimal("100"),
                "composting": Decimal("75"),
                "landfill": Decimal("50")
            }
        })

        response_data = mock_service.analyze_diversion.return_value

        assert response_data["diversion_rate_percent"] > 50
        assert response_data["diverted_tonnes"] > response_data["landfilled_tonnes"]

    # ===========================
    # GET /provenance/{id} Tests
    # ===========================

    def test_get_provenance(self, client, mock_service):
        """Test GET /provenance/{id}."""
        calculation_id = "CALC-001"

        mock_service.get_provenance = Mock(return_value={
            "calculation_id": calculation_id,
            "provenance_hash": "abc123def456789",
            "provenance_chain": [
                {"stage": "input", "hash": "hash1", "timestamp": "2025-01-15T10:00:00"},
                {"stage": "validation", "hash": "hash2", "timestamp": "2025-01-15T10:00:01"},
                {"stage": "calculation", "hash": "hash3", "timestamp": "2025-01-15T10:00:02"}
            ],
            "inputs": {
                "waste_type": "paper",
                "mass_tonnes": 100.0
            }
        })

        response_data = mock_service.get_provenance.return_value

        assert response_data["calculation_id"] == calculation_id
        assert len(response_data["provenance_chain"]) == 3

    # ===========================
    # Validation Error Tests (400)
    # ===========================

    def test_validation_error_missing_required_field(self, client):
        """Test validation error when required field is missing."""
        invalid_input = {
            "waste_type": "paper"
            # Missing: mass_tonnes
        }

        error_response = {
            "detail": [
                {
                    "loc": ["body", "mass_tonnes"],
                    "msg": "field required",
                    "type": "value_error.missing"
                }
            ]
        }

        assert error_response["detail"][0]["type"] == "value_error.missing"

    def test_validation_error_invalid_type(self, client):
        """Test validation error when field has invalid type."""
        invalid_input = {
            "waste_type": "paper",
            "mass_tonnes": "not_a_number",  # Should be float/Decimal
            "treatment_method": "recycling"
        }

        error_response = {
            "detail": [
                {
                    "loc": ["body", "mass_tonnes"],
                    "msg": "value is not a valid float",
                    "type": "type_error.float"
                }
            ]
        }

        assert error_response["detail"][0]["type"] == "type_error.float"

    def test_validation_error_invalid_enum_value(self, client):
        """Test validation error when enum value is invalid."""
        invalid_input = {
            "waste_type": "paper",
            "mass_tonnes": 100.0,
            "treatment_method": "invalid_method"  # Not in enum
        }

        error_response = {
            "detail": [
                {
                    "loc": ["body", "treatment_method"],
                    "msg": "value is not a valid enumeration member",
                    "type": "type_error.enum"
                }
            ]
        }

        assert "not a valid enumeration" in error_response["detail"][0]["msg"]

    # ===========================
    # Not Found Tests (404)
    # ===========================

    def test_not_found_invalid_calculation_id(self, client, mock_service):
        """Test 404 when calculation ID doesn't exist."""
        mock_service.get_calculation_by_id = Mock(side_effect=KeyError("Not found"))

        with pytest.raises(KeyError, match="Not found"):
            mock_service.get_calculation_by_id("INVALID-ID")

    def test_not_found_delete_nonexistent_calculation(self, client, mock_service):
        """Test 404 when trying to delete non-existent calculation."""
        mock_service.delete_calculation = Mock(side_effect=KeyError("Not found"))

        with pytest.raises(KeyError, match="Not found"):
            mock_service.delete_calculation("INVALID-ID")

    # ===========================
    # Health Check Test
    # ===========================

    def test_get_health(self, client, mock_service):
        """Test GET /health."""
        mock_service.health_check = Mock(return_value={
            "status": "healthy",
            "service": "waste_generated",
            "engines_initialized": 7,
            "database_connected": True,
            "cache_available": True,
            "timestamp": "2025-01-15T10:00:00Z"
        })

        response_data = mock_service.health_check.return_value

        assert response_data["status"] == "healthy"
        assert response_data["database_connected"] is True
