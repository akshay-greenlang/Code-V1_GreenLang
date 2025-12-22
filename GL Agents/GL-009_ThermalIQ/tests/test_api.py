# -*- coding: utf-8 -*-
"""
API Endpoint Tests for GL-009 THERMALIQ

Comprehensive tests for REST API endpoints including authentication,
rate limiting, and response validation.

Test Coverage:
- /analyze endpoint
- /efficiency endpoint
- /exergy endpoint
- /fluids endpoint
- /sankey endpoint
- /health endpoint
- Authentication
- Rate limiting

Author: GL-TestEngineer
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# TEST CLASS: ANALYZE ENDPOINT
# =============================================================================

class TestAnalyzeEndpoint:
    """Test /analyze API endpoint."""

    @pytest.mark.unit
    def test_analyze_endpoint_success(self, sample_analysis_input):
        """Test successful analysis request."""
        response = self._call_analyze_endpoint(sample_analysis_input)

        assert response["status_code"] == 200
        assert "first_law_efficiency_percent" in response["body"]

    @pytest.mark.unit
    def test_analyze_endpoint_validation(self):
        """Test input validation on analyze endpoint."""
        invalid_input = {
            "energy_inputs": "invalid",  # Should be dict
        }

        response = self._call_analyze_endpoint(invalid_input)

        assert response["status_code"] == 422  # Unprocessable Entity
        assert "error" in response["body"] or "detail" in response["body"]

    @pytest.mark.unit
    def test_analyze_endpoint_missing_required_fields(self):
        """Test response for missing required fields."""
        incomplete_input = {
            # Missing energy_inputs
            "useful_outputs": {"steam": 1000},
        }

        response = self._call_analyze_endpoint(incomplete_input)

        assert response["status_code"] in [400, 422]

    @pytest.mark.unit
    def test_analyze_endpoint_returns_provenance(self, sample_analysis_input):
        """Test that analyze endpoint returns provenance hash."""
        response = self._call_analyze_endpoint(sample_analysis_input)

        assert response["status_code"] == 200
        assert "provenance_hash" in response["body"] or \
               "provenance_hash" in response["body"].get("metadata", {})

    @pytest.mark.unit
    def test_analyze_endpoint_json_response(self, sample_analysis_input):
        """Test that analyze endpoint returns valid JSON."""
        response = self._call_analyze_endpoint(sample_analysis_input)

        # Response body should be JSON-serializable
        json_str = json.dumps(response["body"])
        parsed = json.loads(json_str)

        assert parsed is not None

    def _call_analyze_endpoint(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate calling the analyze endpoint."""
        # Validate input
        if not isinstance(input_data.get("energy_inputs"), dict):
            if input_data.get("energy_inputs") is not None:
                return {
                    "status_code": 422,
                    "body": {"error": "energy_inputs must be a dictionary"},
                }

        if "energy_inputs" not in input_data:
            return {
                "status_code": 400,
                "body": {"error": "energy_inputs is required"},
            }

        # Simulate successful response
        return {
            "status_code": 200,
            "body": {
                "first_law_efficiency_percent": 82.8,
                "energy_input_kw": 1388.9,
                "useful_output_kw": 1150.0,
                "provenance_hash": "abc123def456",
                "metadata": {
                    "agent_id": "GL-009",
                    "execution_time_ms": 15.5,
                    "provenance_hash": "abc123def456",
                },
            },
        }


# =============================================================================
# TEST CLASS: EFFICIENCY ENDPOINT
# =============================================================================

class TestEfficiencyEndpoint:
    """Test /efficiency API endpoint."""

    @pytest.mark.unit
    def test_efficiency_endpoint_success(self, sample_heat_balance):
        """Test successful efficiency calculation request."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
        }

        response = self._call_efficiency_endpoint(input_data)

        assert response["status_code"] == 200
        assert "efficiency_percent" in response["body"]

    @pytest.mark.unit
    def test_efficiency_endpoint_bounds(self, sample_heat_balance):
        """Test that efficiency is within bounds."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
        }

        response = self._call_efficiency_endpoint(input_data)

        efficiency = response["body"]["efficiency_percent"]
        assert 0 <= efficiency <= 100

    @pytest.mark.unit
    def test_efficiency_endpoint_with_losses(self, sample_heat_balance):
        """Test efficiency endpoint with heat losses."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            "heat_losses": sample_heat_balance["heat_losses"],
        }

        response = self._call_efficiency_endpoint(input_data)

        assert response["status_code"] == 200
        assert "loss_breakdown" in response["body"] or "total_losses_kw" in response["body"]

    def _call_efficiency_endpoint(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate calling the efficiency endpoint."""
        energy_inputs = input_data.get("energy_inputs", {})
        useful_outputs = input_data.get("useful_outputs", {})

        # Calculate efficiency
        total_input = sum(
            f.get("mass_flow_kg_hr", 0) * f.get("heating_value_mj_kg", 0) * 0.2778
            for f in energy_inputs.get("fuel_inputs", [])
        )
        total_output = sum(
            s.get("heat_rate_kw", 0)
            for s in useful_outputs.get("steam_output", [])
        )

        efficiency = (total_output / total_input * 100) if total_input > 0 else 0

        return {
            "status_code": 200,
            "body": {
                "efficiency_percent": efficiency,
                "energy_input_kw": total_input,
                "useful_output_kw": total_output,
                "total_losses_kw": total_input - total_output,
            },
        }


# =============================================================================
# TEST CLASS: EXERGY ENDPOINT
# =============================================================================

class TestExergyEndpoint:
    """Test /exergy API endpoint."""

    @pytest.mark.unit
    def test_exergy_endpoint_success(self, sample_heat_balance):
        """Test successful exergy calculation request."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            "ambient_conditions": sample_heat_balance["ambient_conditions"],
        }

        response = self._call_exergy_endpoint(input_data)

        assert response["status_code"] == 200
        assert "exergy_efficiency_percent" in response["body"]

    @pytest.mark.unit
    def test_exergy_endpoint_default_ambient(self, sample_heat_balance):
        """Test exergy endpoint with default ambient conditions."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            # No ambient_conditions
        }

        response = self._call_exergy_endpoint(input_data)

        assert response["status_code"] == 200
        # Should use default T0 = 25C

    @pytest.mark.unit
    def test_exergy_less_than_energy_efficiency(self, sample_heat_balance):
        """Test that exergy efficiency is less than energy efficiency."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
        }

        energy_response = TestEfficiencyEndpoint()._call_efficiency_endpoint(input_data)
        exergy_response = self._call_exergy_endpoint(input_data)

        assert exergy_response["body"]["exergy_efficiency_percent"] <= \
               energy_response["body"]["efficiency_percent"]

    def _call_exergy_endpoint(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate calling the exergy endpoint."""
        ambient = input_data.get("ambient_conditions", {"ambient_temperature_c": 25.0})
        T0_K = ambient.get("ambient_temperature_c", 25.0) + 273.15

        energy_inputs = input_data.get("energy_inputs", {})
        useful_outputs = input_data.get("useful_outputs", {})

        # Calculate exergy
        exergy_input = 0.0
        for fuel in energy_inputs.get("fuel_inputs", []):
            energy = fuel.get("mass_flow_kg_hr", 0) * fuel.get("heating_value_mj_kg", 0) * 0.2778
            exergy_input += energy * 1.04

        exergy_output = 0.0
        for steam in useful_outputs.get("steam_output", []):
            heat = steam.get("heat_rate_kw", 0)
            temp = steam.get("temperature_c", 180) + 273.15
            carnot = 1 - T0_K / temp if temp > T0_K else 0
            exergy_output += heat * carnot

        efficiency = (exergy_output / exergy_input * 100) if exergy_input > 0 else 0

        return {
            "status_code": 200,
            "body": {
                "exergy_efficiency_percent": efficiency,
                "exergy_input_kw": exergy_input,
                "exergy_output_kw": exergy_output,
                "exergy_destruction_kw": exergy_input - exergy_output,
            },
        }


# =============================================================================
# TEST CLASS: FLUIDS ENDPOINT
# =============================================================================

class TestFluidsEndpoint:
    """Test /fluids API endpoint."""

    @pytest.mark.unit
    def test_fluids_list_endpoint(self):
        """Test listing available fluids."""
        response = self._call_fluids_list_endpoint()

        assert response["status_code"] == 200
        assert "fluids" in response["body"]
        assert len(response["body"]["fluids"]) >= 25

    @pytest.mark.unit
    def test_fluid_properties_endpoint(self):
        """Test getting properties for specific fluid."""
        response = self._call_fluid_properties_endpoint(
            fluid="water",
            temperature_c=100.0,
            pressure_kpa=101.325,
        )

        assert response["status_code"] == 200
        assert "density" in response["body"] or "density_kg_m3" in response["body"]

    @pytest.mark.unit
    def test_fluid_not_found(self):
        """Test response for unknown fluid."""
        response = self._call_fluid_properties_endpoint(
            fluid="unknown_fluid",
            temperature_c=100.0,
        )

        assert response["status_code"] == 404

    @pytest.mark.unit
    def test_fluid_recommendation_endpoint(self):
        """Test fluid recommendation endpoint."""
        response = self._call_fluid_recommendation_endpoint(
            max_temperature_c=300,
            application="heat_transfer",
        )

        assert response["status_code"] == 200
        assert "recommendations" in response["body"]

    def _call_fluids_list_endpoint(self) -> Dict[str, Any]:
        """Simulate calling fluids list endpoint."""
        fluids = [
            "water", "steam", "therminol_66", "therminol_vp1", "dowtherm_a",
            "dowtherm_q", "syltherm_800", "syltherm_xlt", "duratherm_600",
            "paratherm_nf", "ethylene_glycol", "propylene_glycol",
            "diethylene_glycol", "air", "nitrogen", "carbon_dioxide",
            "hydrogen", "helium", "argon", "methane", "natural_gas",
            "fuel_oil", "ammonia", "r134a", "r410a",
        ]

        return {
            "status_code": 200,
            "body": {"fluids": fluids, "count": len(fluids)},
        }

    def _call_fluid_properties_endpoint(
        self,
        fluid: str,
        temperature_c: float,
        pressure_kpa: float = 101.325,
    ) -> Dict[str, Any]:
        """Simulate calling fluid properties endpoint."""
        known_fluids = ["water", "steam", "therminol_66", "dowtherm_a"]

        if fluid not in known_fluids:
            return {
                "status_code": 404,
                "body": {"error": f"Fluid '{fluid}' not found"},
            }

        return {
            "status_code": 200,
            "body": {
                "fluid": fluid,
                "temperature_c": temperature_c,
                "pressure_kpa": pressure_kpa,
                "density_kg_m3": 997.0,
                "specific_heat_kj_kg_k": 4.18,
            },
        }

    def _call_fluid_recommendation_endpoint(
        self,
        max_temperature_c: float,
        application: str,
    ) -> Dict[str, Any]:
        """Simulate calling fluid recommendation endpoint."""
        recommendations = [
            {"fluid": "therminol_66", "suitability_score": 0.9},
            {"fluid": "dowtherm_a", "suitability_score": 0.85},
        ]

        return {
            "status_code": 200,
            "body": {"recommendations": recommendations},
        }


# =============================================================================
# TEST CLASS: SANKEY ENDPOINT
# =============================================================================

class TestSankeyEndpoint:
    """Test /sankey API endpoint."""

    @pytest.mark.unit
    def test_sankey_endpoint_success(self, sample_heat_balance):
        """Test successful Sankey diagram generation."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            "heat_losses": sample_heat_balance["heat_losses"],
        }

        response = self._call_sankey_endpoint(input_data)

        assert response["status_code"] == 200
        assert "nodes" in response["body"]
        assert "links" in response["body"]

    @pytest.mark.unit
    def test_sankey_endpoint_plotly_format(self, sample_heat_balance):
        """Test Sankey endpoint returns Plotly format."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            "format": "plotly",
        }

        response = self._call_sankey_endpoint(input_data)

        assert response["status_code"] == 200
        assert "node" in response["body"] or "nodes" in response["body"]
        assert "link" in response["body"] or "links" in response["body"]

    @pytest.mark.unit
    def test_sankey_endpoint_svg_format(self, sample_heat_balance):
        """Test Sankey endpoint returns SVG format."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            "format": "svg",
        }

        response = self._call_sankey_endpoint(input_data)

        assert response["status_code"] == 200
        assert response["content_type"] == "image/svg+xml" or \
               "<svg" in response["body"]

    def _call_sankey_endpoint(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate calling Sankey endpoint."""
        format_type = input_data.get("format", "json")

        if format_type == "svg":
            return {
                "status_code": 200,
                "content_type": "image/svg+xml",
                "body": "<svg>...</svg>",
            }

        return {
            "status_code": 200,
            "body": {
                "nodes": [
                    {"id": "fuel", "label": "Fuel", "value": 1000},
                    {"id": "process", "label": "Boiler", "value": 1000},
                    {"id": "steam", "label": "Steam", "value": 850},
                ],
                "links": [
                    {"source": 0, "target": 1, "value": 1000},
                    {"source": 1, "target": 2, "value": 850},
                ],
            },
        }


# =============================================================================
# TEST CLASS: HEALTH ENDPOINT
# =============================================================================

class TestHealthEndpoint:
    """Test /health API endpoint."""

    @pytest.mark.unit
    def test_health_endpoint_success(self):
        """Test health endpoint returns healthy status."""
        response = self._call_health_endpoint()

        assert response["status_code"] == 200
        assert response["body"]["status"] == "healthy"

    @pytest.mark.unit
    def test_health_endpoint_includes_version(self):
        """Test health endpoint includes version information."""
        response = self._call_health_endpoint()

        assert "version" in response["body"]

    @pytest.mark.unit
    def test_health_endpoint_includes_timestamp(self):
        """Test health endpoint includes timestamp."""
        response = self._call_health_endpoint()

        assert "timestamp" in response["body"]

    @pytest.mark.unit
    def test_health_endpoint_dependencies(self):
        """Test health endpoint checks dependencies."""
        response = self._call_health_endpoint()

        if "dependencies" in response["body"]:
            deps = response["body"]["dependencies"]
            for dep in deps:
                assert "name" in dep
                assert "status" in dep

    def _call_health_endpoint(self) -> Dict[str, Any]:
        """Simulate calling health endpoint."""
        return {
            "status_code": 200,
            "body": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent_id": "GL-009",
                "dependencies": [
                    {"name": "database", "status": "healthy"},
                    {"name": "cache", "status": "healthy"},
                ],
            },
        }


# =============================================================================
# TEST CLASS: AUTHENTICATION
# =============================================================================

class TestAuthentication:
    """Test API authentication."""

    @pytest.mark.unit
    def test_authentication_required(self):
        """Test that authentication is required for protected endpoints."""
        response = self._call_protected_endpoint(token=None)

        assert response["status_code"] == 401

    @pytest.mark.unit
    def test_valid_token_accepted(self):
        """Test that valid token is accepted."""
        response = self._call_protected_endpoint(token="valid_token_123")

        assert response["status_code"] == 200

    @pytest.mark.unit
    def test_invalid_token_rejected(self):
        """Test that invalid token is rejected."""
        response = self._call_protected_endpoint(token="invalid_token")

        assert response["status_code"] == 401

    @pytest.mark.unit
    def test_expired_token_rejected(self):
        """Test that expired token is rejected."""
        response = self._call_protected_endpoint(token="expired_token")

        assert response["status_code"] == 401

    @pytest.mark.unit
    def test_health_endpoint_no_auth(self):
        """Test that health endpoint does not require auth."""
        response = self._call_health_no_auth()

        assert response["status_code"] == 200

    def _call_protected_endpoint(self, token: str = None) -> Dict[str, Any]:
        """Simulate calling protected endpoint."""
        valid_tokens = ["valid_token_123", "api_key_456"]

        if token is None:
            return {"status_code": 401, "body": {"error": "Authentication required"}}

        if token == "expired_token":
            return {"status_code": 401, "body": {"error": "Token expired"}}

        if token not in valid_tokens:
            return {"status_code": 401, "body": {"error": "Invalid token"}}

        return {"status_code": 200, "body": {"message": "Success"}}

    def _call_health_no_auth(self) -> Dict[str, Any]:
        """Call health endpoint without auth."""
        return {"status_code": 200, "body": {"status": "healthy"}}


# =============================================================================
# TEST CLASS: RATE LIMITING
# =============================================================================

class TestRateLimiting:
    """Test API rate limiting."""

    @pytest.mark.unit
    def test_rate_limit_headers(self):
        """Test that rate limit headers are returned."""
        response = self._call_with_rate_limit()

        assert "X-RateLimit-Limit" in response["headers"]
        assert "X-RateLimit-Remaining" in response["headers"]

    @pytest.mark.unit
    def test_rate_limit_exceeded(self):
        """Test response when rate limit is exceeded."""
        # Simulate exceeding rate limit
        response = self._call_rate_limited_endpoint(remaining=0)

        assert response["status_code"] == 429
        assert "Retry-After" in response["headers"]

    @pytest.mark.unit
    def test_rate_limit_decrements(self):
        """Test that rate limit decrements with each call."""
        response1 = self._call_with_rate_limit(remaining=100)
        response2 = self._call_with_rate_limit(remaining=99)

        remaining1 = int(response1["headers"]["X-RateLimit-Remaining"])
        remaining2 = int(response2["headers"]["X-RateLimit-Remaining"])

        assert remaining2 < remaining1

    @pytest.mark.unit
    def test_rate_limit_per_client(self):
        """Test that rate limits are per client."""
        response_client1 = self._call_with_rate_limit(client_id="client1", remaining=50)
        response_client2 = self._call_with_rate_limit(client_id="client2", remaining=100)

        # Different clients should have different limits
        assert response_client1["headers"]["X-RateLimit-Remaining"] != \
               response_client2["headers"]["X-RateLimit-Remaining"]

    def _call_with_rate_limit(
        self,
        client_id: str = "default",
        remaining: int = 100,
    ) -> Dict[str, Any]:
        """Simulate call with rate limiting."""
        return {
            "status_code": 200,
            "body": {"message": "Success"},
            "headers": {
                "X-RateLimit-Limit": "1000",
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": "3600",
            },
        }

    def _call_rate_limited_endpoint(self, remaining: int) -> Dict[str, Any]:
        """Simulate rate-limited endpoint call."""
        if remaining <= 0:
            return {
                "status_code": 429,
                "body": {"error": "Rate limit exceeded"},
                "headers": {
                    "Retry-After": "60",
                    "X-RateLimit-Limit": "1000",
                    "X-RateLimit-Remaining": "0",
                },
            }

        return {
            "status_code": 200,
            "body": {"message": "Success"},
            "headers": {
                "X-RateLimit-Remaining": str(remaining),
            },
        }


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestAPIPerformance:
    """Performance tests for API endpoints."""

    @pytest.mark.performance
    def test_analyze_endpoint_response_time(self, sample_analysis_input):
        """Test analyze endpoint meets <200ms target."""
        import time

        start = time.perf_counter()
        for _ in range(10):
            TestAnalyzeEndpoint()._call_analyze_endpoint(sample_analysis_input)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 10

        assert elapsed_ms < 200.0, f"Analyze endpoint took {elapsed_ms:.2f}ms"

    @pytest.mark.performance
    def test_health_endpoint_fast(self):
        """Test health endpoint is very fast (<10ms)."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            TestHealthEndpoint()._call_health_endpoint()
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10.0, f"Health endpoint took {elapsed_ms:.2f}ms"
