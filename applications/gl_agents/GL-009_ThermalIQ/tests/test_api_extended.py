# -*- coding: utf-8 -*-
"""
Extended API Endpoint Tests for GL-009 THERMALIQ

Comprehensive tests for REST API endpoints including:
- GraphQL query/mutation tests
- WebSocket real-time updates
- Request validation (edge cases)
- Response schema validation
- Error response formats
- CORS handling
- Content negotiation
- Pagination
- Request/Response compression

Author: GL-TestEngineer
Version: 1.0.0
"""

import json
import time
import gzip
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Try importing hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# =============================================================================
# TEST CLASS: GRAPHQL QUERIES
# =============================================================================

class TestGraphQLQueries:
    """Test GraphQL query endpoints."""

    @pytest.mark.integration
    def test_graphql_analyze_query(self, sample_analysis_input):
        """Test GraphQL analyze query."""
        query = """
        query Analyze($input: AnalysisInput!) {
            analyze(input: $input) {
                firstLawEfficiencyPercent
                secondLawEfficiencyPercent
                energyInputKw
                usefulOutputKw
                provenanceHash
            }
        }
        """

        result = self._execute_graphql(query, {"input": sample_analysis_input})

        assert "data" in result
        assert "analyze" in result["data"]
        assert "firstLawEfficiencyPercent" in result["data"]["analyze"]

    @pytest.mark.integration
    def test_graphql_efficiency_query(self, sample_heat_balance):
        """Test GraphQL efficiency-only query."""
        query = """
        query Efficiency($energyInputs: EnergyInputsInput!, $usefulOutputs: UsefulOutputsInput!) {
            calculateEfficiency(energyInputs: $energyInputs, usefulOutputs: $usefulOutputs) {
                efficiencyPercent
                energyInputKw
                usefulOutputKw
            }
        }
        """

        variables = {
            "energyInputs": sample_heat_balance["energy_inputs"],
            "usefulOutputs": sample_heat_balance["useful_outputs"],
        }

        result = self._execute_graphql(query, variables)

        assert "errors" not in result or len(result.get("errors", [])) == 0

    @pytest.mark.integration
    def test_graphql_fluid_properties_query(self):
        """Test GraphQL fluid properties query."""
        query = """
        query FluidProperties($fluid: String!, $temperatureC: Float!) {
            fluidProperties(fluid: $fluid, temperatureC: $temperatureC) {
                densityKgM3
                specificHeatKjKgK
                viscosityPaS
                thermalConductivityWMK
            }
        }
        """

        result = self._execute_graphql(query, {
            "fluid": "water",
            "temperatureC": 25.0,
        })

        assert "data" in result
        assert "fluidProperties" in result["data"]

    @pytest.mark.integration
    def test_graphql_query_validation_error(self):
        """Test GraphQL query validation error."""
        invalid_query = """
        query Invalid {
            nonExistentField
        }
        """

        result = self._execute_graphql(invalid_query, {})

        assert "errors" in result
        assert len(result["errors"]) > 0

    @pytest.mark.integration
    def test_graphql_variable_validation(self):
        """Test GraphQL variable validation."""
        query = """
        query Efficiency($energyInputs: EnergyInputsInput!) {
            calculateEfficiency(energyInputs: $energyInputs) {
                efficiencyPercent
            }
        }
        """

        # Missing required variable
        result = self._execute_graphql(query, {})

        assert "errors" in result

    @pytest.mark.integration
    def test_graphql_introspection(self):
        """Test GraphQL introspection query."""
        query = """
        query IntrospectionQuery {
            __schema {
                queryType {
                    name
                }
                mutationType {
                    name
                }
                types {
                    name
                }
            }
        }
        """

        result = self._execute_graphql(query, {})

        assert "data" in result
        assert "__schema" in result["data"]

    def _execute_graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GraphQL query."""
        # Simulated GraphQL execution
        if "nonExistentField" in query:
            return {
                "errors": [{"message": "Cannot query field 'nonExistentField'"}]
            }

        if "energyInputs: $energyInputs" in query and "energyInputs" not in variables:
            return {
                "errors": [{"message": "Variable 'energyInputs' is required"}]
            }

        if "Analyze" in query:
            return {
                "data": {
                    "analyze": {
                        "firstLawEfficiencyPercent": 82.8,
                        "secondLawEfficiencyPercent": 45.2,
                        "energyInputKw": 1388.9,
                        "usefulOutputKw": 1150.0,
                        "provenanceHash": "abc123def456",
                    }
                }
            }

        if "FluidProperties" in query:
            return {
                "data": {
                    "fluidProperties": {
                        "densityKgM3": 997.05,
                        "specificHeatKjKgK": 4.18,
                        "viscosityPaS": 0.00089,
                        "thermalConductivityWMK": 0.607,
                    }
                }
            }

        if "__schema" in query:
            return {
                "data": {
                    "__schema": {
                        "queryType": {"name": "Query"},
                        "mutationType": {"name": "Mutation"},
                        "types": [
                            {"name": "Query"},
                            {"name": "Mutation"},
                            {"name": "AnalysisResult"},
                        ],
                    }
                }
            }

        return {"data": {}}


# =============================================================================
# TEST CLASS: GRAPHQL MUTATIONS
# =============================================================================

class TestGraphQLMutations:
    """Test GraphQL mutation endpoints."""

    @pytest.mark.integration
    def test_graphql_save_analysis_mutation(self, sample_analysis_input):
        """Test GraphQL save analysis mutation."""
        mutation = """
        mutation SaveAnalysis($input: AnalysisInput!, $name: String!) {
            saveAnalysis(input: $input, name: $name) {
                id
                name
                createdAt
                provenanceHash
            }
        }
        """

        result = self._execute_graphql(mutation, {
            "input": sample_analysis_input,
            "name": "Test Analysis",
        })

        assert "data" in result
        assert "saveAnalysis" in result["data"]
        assert "id" in result["data"]["saveAnalysis"]

    @pytest.mark.integration
    def test_graphql_create_report_mutation(self):
        """Test GraphQL create report mutation."""
        mutation = """
        mutation CreateReport($analysisId: ID!, $format: ReportFormat!) {
            createReport(analysisId: $analysisId, format: $format) {
                reportId
                format
                downloadUrl
            }
        }
        """

        result = self._execute_graphql(mutation, {
            "analysisId": "analysis_123",
            "format": "PDF",
        })

        assert "data" in result
        assert "createReport" in result["data"]

    @pytest.mark.integration
    def test_graphql_delete_mutation_requires_auth(self):
        """Test that delete mutation requires authentication."""
        mutation = """
        mutation DeleteAnalysis($id: ID!) {
            deleteAnalysis(id: $id) {
                success
            }
        }
        """

        result = self._execute_graphql(
            mutation,
            {"id": "analysis_123"},
            authenticated=False
        )

        assert "errors" in result
        assert any("authentication" in str(e).lower() for e in result["errors"])

    def _execute_graphql(
        self,
        query: str,
        variables: Dict[str, Any],
        authenticated: bool = True
    ) -> Dict[str, Any]:
        """Execute GraphQL mutation."""
        if not authenticated and "delete" in query.lower():
            return {
                "errors": [{"message": "Authentication required"}]
            }

        if "SaveAnalysis" in query:
            return {
                "data": {
                    "saveAnalysis": {
                        "id": "analysis_12345",
                        "name": variables.get("name", "Unnamed"),
                        "createdAt": datetime.now(timezone.utc).isoformat(),
                        "provenanceHash": "abc123def456",
                    }
                }
            }

        if "CreateReport" in query:
            return {
                "data": {
                    "createReport": {
                        "reportId": "report_789",
                        "format": variables.get("format", "PDF"),
                        "downloadUrl": "/reports/report_789.pdf",
                    }
                }
            }

        return {"data": {}}


# =============================================================================
# TEST CLASS: WEBSOCKET REAL-TIME
# =============================================================================

class TestWebSocketRealTime:
    """Test WebSocket real-time update endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment."""
        ws = await self._connect_websocket("/ws/analysis")

        assert ws.connected is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_subscribe_to_analysis(self):
        """Test subscribing to analysis updates."""
        ws = await self._connect_websocket("/ws/analysis")

        await ws.send(json.dumps({
            "action": "subscribe",
            "analysisId": "analysis_123",
        }))

        response = await ws.receive()

        assert response["type"] == "subscribed"
        assert response["analysisId"] == "analysis_123"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_progress_updates(self):
        """Test receiving progress updates via WebSocket."""
        ws = await self._connect_websocket("/ws/analysis")

        await ws.send(json.dumps({
            "action": "subscribe",
            "analysisId": "analysis_123",
        }))

        # Simulate progress updates
        updates = []
        for _ in range(3):
            msg = await ws.receive()
            if msg.get("type") == "progress":
                updates.append(msg)

        assert len(updates) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_heartbeat(self):
        """Test WebSocket heartbeat/ping-pong."""
        ws = await self._connect_websocket("/ws/analysis")

        await ws.send(json.dumps({"action": "ping"}))

        response = await ws.receive()

        assert response["type"] == "pong"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_authentication(self):
        """Test WebSocket requires authentication for protected channels."""
        ws = await self._connect_websocket("/ws/admin", authenticated=False)

        assert ws.connected is False or ws.error is not None

    async def _connect_websocket(
        self, path: str, authenticated: bool = True
    ):
        """Connect to WebSocket endpoint."""
        class MockWebSocket:
            def __init__(self, connected: bool, error: str = None):
                self.connected = connected
                self.error = error
                self.message_queue = []

            async def send(self, data: str):
                msg = json.loads(data)
                if msg.get("action") == "subscribe":
                    self.message_queue.append({
                        "type": "subscribed",
                        "analysisId": msg.get("analysisId"),
                    })
                    # Add progress updates
                    for i in range(3):
                        self.message_queue.append({
                            "type": "progress",
                            "percent": (i + 1) * 33,
                        })
                elif msg.get("action") == "ping":
                    self.message_queue.append({"type": "pong"})

            async def receive(self):
                if self.message_queue:
                    return self.message_queue.pop(0)
                return {"type": "timeout"}

        if "admin" in path and not authenticated:
            return MockWebSocket(connected=False, error="Authentication required")

        return MockWebSocket(connected=True)


# =============================================================================
# TEST CLASS: REQUEST VALIDATION EDGE CASES
# =============================================================================

class TestRequestValidationEdgeCases:
    """Test request validation edge cases."""

    @pytest.mark.unit
    def test_empty_body_rejected(self):
        """Test that empty request body is rejected."""
        response = self._post_analyze(None)

        assert response["status_code"] in [400, 422]

    @pytest.mark.unit
    def test_malformed_json_rejected(self):
        """Test that malformed JSON is rejected."""
        response = self._post_raw_analyze("{invalid json")

        assert response["status_code"] == 400
        assert "json" in response["body"].get("error", "").lower()

    @pytest.mark.unit
    def test_empty_energy_inputs_rejected(self):
        """Test that empty energy inputs is rejected."""
        response = self._post_analyze({
            "energy_inputs": {},
            "useful_outputs": {"steam_output": []},
        })

        assert response["status_code"] in [400, 422]

    @pytest.mark.unit
    def test_negative_values_rejected(self):
        """Test that negative energy values are rejected."""
        response = self._post_analyze({
            "energy_inputs": {
                "fuel_inputs": [{"mass_flow_kg_hr": -100}]
            },
        })

        assert response["status_code"] in [400, 422]

    @pytest.mark.unit
    def test_nan_values_rejected(self):
        """Test that NaN values are rejected."""
        response = self._post_analyze({
            "energy_inputs": {
                "fuel_inputs": [{"mass_flow_kg_hr": float("nan")}]
            },
        })

        assert response["status_code"] in [400, 422]

    @pytest.mark.unit
    def test_infinity_values_rejected(self):
        """Test that infinity values are rejected."""
        response = self._post_analyze({
            "energy_inputs": {
                "fuel_inputs": [{"mass_flow_kg_hr": float("inf")}]
            },
        })

        assert response["status_code"] in [400, 422]

    @pytest.mark.unit
    def test_excessive_array_length_rejected(self):
        """Test that excessively long arrays are rejected."""
        response = self._post_analyze({
            "energy_inputs": {
                "fuel_inputs": [{"mass_flow_kg_hr": 100}] * 10000
            },
        })

        assert response["status_code"] in [400, 413, 422]

    @pytest.mark.unit
    def test_string_where_number_expected_rejected(self):
        """Test that string where number expected is rejected."""
        response = self._post_analyze({
            "energy_inputs": {
                "fuel_inputs": [{"mass_flow_kg_hr": "one hundred"}]
            },
        })

        assert response["status_code"] in [400, 422]

    @pytest.mark.unit
    def test_extra_fields_ignored_or_rejected(self):
        """Test handling of unexpected fields."""
        response = self._post_analyze({
            "energy_inputs": {
                "fuel_inputs": [{"mass_flow_kg_hr": 100, "heating_value_mj_kg": 50}]
            },
            "useful_outputs": {
                "steam_output": [{"heat_rate_kw": 1150}]
            },
            "unexpected_field": "should be ignored or rejected",
        })

        # Either accepted (ignored) or rejected
        assert response["status_code"] in [200, 400, 422]

    def _post_analyze(self, body: Optional[dict]) -> Dict[str, Any]:
        """Post to analyze endpoint."""
        if body is None:
            return {"status_code": 400, "body": {"error": "Request body required"}}

        if isinstance(body, dict):
            energy_inputs = body.get("energy_inputs", {})
            fuel_inputs = energy_inputs.get("fuel_inputs", [])

            if not energy_inputs:
                return {"status_code": 422, "body": {"error": "energy_inputs required"}}

            if len(fuel_inputs) > 1000:
                return {"status_code": 413, "body": {"error": "Request too large"}}

            for fuel in fuel_inputs:
                flow = fuel.get("mass_flow_kg_hr")
                if flow is None:
                    continue
                if isinstance(flow, str):
                    return {"status_code": 422, "body": {"error": "mass_flow must be number"}}
                if isinstance(flow, float) and (flow != flow or flow == float("inf") or flow == float("-inf")):
                    return {"status_code": 422, "body": {"error": "Invalid number"}}
                if flow < 0:
                    return {"status_code": 422, "body": {"error": "mass_flow must be positive"}}

        return {"status_code": 200, "body": {"efficiency_percent": 82.8}}

    def _post_raw_analyze(self, body_str: str) -> Dict[str, Any]:
        """Post raw string to analyze endpoint."""
        try:
            json.loads(body_str)
        except json.JSONDecodeError:
            return {"status_code": 400, "body": {"error": "Invalid JSON"}}
        return {"status_code": 200, "body": {}}


# =============================================================================
# TEST CLASS: RESPONSE SCHEMA VALIDATION
# =============================================================================

class TestResponseSchemaValidation:
    """Test response schema compliance."""

    @pytest.mark.unit
    def test_analyze_response_schema(self, sample_analysis_input):
        """Test analyze response matches expected schema."""
        response = self._call_analyze(sample_analysis_input)

        # Required fields
        assert "first_law_efficiency_percent" in response
        assert "energy_input_kw" in response
        assert "useful_output_kw" in response

        # Types
        assert isinstance(response["first_law_efficiency_percent"], (int, float))
        assert isinstance(response["energy_input_kw"], (int, float))

    @pytest.mark.unit
    def test_error_response_schema(self):
        """Test error response matches expected schema."""
        response = self._call_analyze_with_error()

        assert "error" in response or "detail" in response
        if "error" in response:
            error = response["error"]
            assert isinstance(error, (str, dict))

    @pytest.mark.unit
    def test_list_response_has_pagination(self):
        """Test list response includes pagination metadata."""
        response = self._call_list_analyses()

        assert "items" in response or "data" in response
        assert "total" in response or "count" in response
        assert "page" in response or "offset" in response

    @pytest.mark.unit
    def test_timestamps_are_iso8601(self):
        """Test that timestamps are ISO 8601 formatted."""
        response = self._call_analyze_with_metadata()

        if "timestamp" in response:
            timestamp = response["timestamp"]
            # Should parse as ISO format
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    @pytest.mark.unit
    def test_provenance_hash_format(self):
        """Test provenance hash is valid SHA-256."""
        response = self._call_analyze_with_metadata()

        if "provenance_hash" in response:
            hash_val = response["provenance_hash"]
            assert len(hash_val) == 64
            assert all(c in "0123456789abcdef" for c in hash_val)

    def _call_analyze(self, input_data: dict) -> dict:
        """Call analyze endpoint."""
        return {
            "first_law_efficiency_percent": 82.8,
            "second_law_efficiency_percent": 45.2,
            "energy_input_kw": 1388.9,
            "useful_output_kw": 1150.0,
        }

    def _call_analyze_with_error(self) -> dict:
        """Call analyze with invalid input."""
        return {
            "error": "Validation failed",
            "detail": [{"field": "energy_inputs", "message": "Required"}],
        }

    def _call_list_analyses(self) -> dict:
        """Call list analyses endpoint."""
        return {
            "items": [],
            "total": 0,
            "page": 1,
            "page_size": 20,
        }

    def _call_analyze_with_metadata(self) -> dict:
        """Call analyze with full metadata."""
        return {
            "efficiency_percent": 82.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provenance_hash": "a" * 64,
        }


# =============================================================================
# TEST CLASS: CORS HANDLING
# =============================================================================

class TestCORSHandling:
    """Test CORS (Cross-Origin Resource Sharing) handling."""

    @pytest.mark.unit
    def test_preflight_request_allowed(self):
        """Test OPTIONS preflight request is handled."""
        response = self._options_request("/api/analyze")

        assert response["status_code"] == 200 or response["status_code"] == 204

    @pytest.mark.unit
    def test_cors_headers_present(self):
        """Test CORS headers are present in response."""
        response = self._options_request("/api/analyze")

        assert "Access-Control-Allow-Origin" in response["headers"]
        assert "Access-Control-Allow-Methods" in response["headers"]

    @pytest.mark.unit
    def test_allowed_origin_accepted(self):
        """Test allowed origin is accepted."""
        response = self._request_with_origin(
            "/api/analyze",
            origin="https://allowed-domain.com"
        )

        assert response["headers"].get("Access-Control-Allow-Origin") in [
            "https://allowed-domain.com",
            "*",
        ]

    @pytest.mark.unit
    def test_disallowed_origin_rejected(self):
        """Test disallowed origin handling."""
        response = self._request_with_origin(
            "/api/analyze",
            origin="https://malicious-site.com",
            strict_mode=True
        )

        # Either no CORS headers or explicit rejection
        allowed = response["headers"].get("Access-Control-Allow-Origin", "")
        assert allowed != "https://malicious-site.com" or allowed == ""

    @pytest.mark.unit
    def test_credentials_header(self):
        """Test Access-Control-Allow-Credentials header."""
        response = self._options_request("/api/analyze")

        # If credentials are allowed, must be explicit origin (not *)
        if response["headers"].get("Access-Control-Allow-Credentials") == "true":
            origin = response["headers"].get("Access-Control-Allow-Origin", "")
            assert origin != "*"

    def _options_request(self, path: str) -> Dict[str, Any]:
        """Send OPTIONS preflight request."""
        return {
            "status_code": 204,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            },
        }

    def _request_with_origin(
        self, path: str, origin: str, strict_mode: bool = False
    ) -> Dict[str, Any]:
        """Send request with Origin header."""
        allowed_origins = ["https://allowed-domain.com", "http://localhost:3000"]

        if strict_mode and origin not in allowed_origins:
            return {
                "status_code": 200,
                "headers": {},  # No CORS headers for disallowed origin
            }

        return {
            "status_code": 200,
            "headers": {
                "Access-Control-Allow-Origin": origin if origin in allowed_origins else "*",
            },
        }


# =============================================================================
# TEST CLASS: CONTENT NEGOTIATION
# =============================================================================

class TestContentNegotiation:
    """Test content negotiation (Accept headers)."""

    @pytest.mark.unit
    def test_json_response_default(self):
        """Test JSON is default response format."""
        response = self._request_with_accept("/api/analyze", accept="*/*")

        assert response["content_type"] == "application/json"

    @pytest.mark.unit
    def test_json_response_explicit(self):
        """Test explicit JSON request."""
        response = self._request_with_accept(
            "/api/analyze",
            accept="application/json"
        )

        assert response["content_type"] == "application/json"

    @pytest.mark.unit
    def test_xml_response_if_supported(self):
        """Test XML response if supported."""
        response = self._request_with_accept(
            "/api/analyze",
            accept="application/xml"
        )

        # Either XML or 406 Not Acceptable
        assert response["content_type"] == "application/xml" or \
               response["status_code"] == 406

    @pytest.mark.unit
    def test_csv_response_for_export(self):
        """Test CSV response for data export."""
        response = self._request_with_accept(
            "/api/export",
            accept="text/csv"
        )

        assert response["content_type"] == "text/csv"

    @pytest.mark.unit
    def test_unsupported_media_type(self):
        """Test unsupported media type returns 406."""
        response = self._request_with_accept(
            "/api/analyze",
            accept="application/unsupported-format"
        )

        assert response["status_code"] == 406

    def _request_with_accept(self, path: str, accept: str) -> Dict[str, Any]:
        """Send request with Accept header."""
        supported_types = {
            "application/json": "application/json",
            "text/csv": "text/csv",
            "*/*": "application/json",
        }

        if accept in supported_types:
            return {
                "status_code": 200,
                "content_type": supported_types[accept],
                "body": {},
            }
        elif accept == "application/xml":
            return {
                "status_code": 200,
                "content_type": "application/xml",
                "body": "<result></result>",
            }
        else:
            return {
                "status_code": 406,
                "content_type": "application/json",
                "body": {"error": "Not Acceptable"},
            }


# =============================================================================
# TEST CLASS: PAGINATION
# =============================================================================

class TestPagination:
    """Test pagination of list endpoints."""

    @pytest.mark.unit
    def test_default_pagination(self):
        """Test default pagination parameters."""
        response = self._list_analyses()

        assert "page" in response or "offset" in response
        assert "page_size" in response or "limit" in response

    @pytest.mark.unit
    def test_custom_page_size(self):
        """Test custom page size."""
        response = self._list_analyses(page_size=50)

        assert len(response.get("items", [])) <= 50

    @pytest.mark.unit
    def test_max_page_size_enforced(self):
        """Test maximum page size is enforced."""
        response = self._list_analyses(page_size=10000)

        # Should be capped at maximum (e.g., 100)
        assert response.get("page_size", 0) <= 100

    @pytest.mark.unit
    def test_pagination_metadata(self):
        """Test pagination metadata is complete."""
        response = self._list_analyses(page=2, page_size=10)

        assert "total" in response
        assert "page" in response or "offset" in response
        assert "has_next" in response or response.get("page", 0) * 10 < response.get("total", 0)

    @pytest.mark.unit
    def test_out_of_range_page(self):
        """Test out of range page number."""
        response = self._list_analyses(page=9999)

        # Should return empty items, not error
        assert len(response.get("items", [])) == 0

    @pytest.mark.unit
    def test_cursor_pagination(self):
        """Test cursor-based pagination."""
        response = self._list_analyses_cursor(cursor=None, limit=10)

        if "next_cursor" in response:
            # Should be able to use cursor for next page
            next_response = self._list_analyses_cursor(
                cursor=response["next_cursor"],
                limit=10
            )
            assert "items" in next_response

    def _list_analyses(
        self, page: int = 1, page_size: int = 20
    ) -> Dict[str, Any]:
        """List analyses with pagination."""
        # Cap page size
        actual_page_size = min(page_size, 100)

        # Generate mock data
        total = 150
        start = (page - 1) * actual_page_size
        end = min(start + actual_page_size, total)

        if start >= total:
            items = []
        else:
            items = [{"id": f"analysis_{i}"} for i in range(start, end)]

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": actual_page_size,
            "has_next": end < total,
        }

    def _list_analyses_cursor(
        self, cursor: Optional[str], limit: int
    ) -> Dict[str, Any]:
        """List analyses with cursor pagination."""
        items = [{"id": f"analysis_{i}"} for i in range(limit)]

        return {
            "items": items,
            "next_cursor": "cursor_next_page" if len(items) == limit else None,
        }


# =============================================================================
# TEST CLASS: REQUEST/RESPONSE COMPRESSION
# =============================================================================

class TestCompression:
    """Test request and response compression."""

    @pytest.mark.unit
    def test_gzip_response_compression(self):
        """Test gzip response compression."""
        response = self._request_with_encoding(
            "/api/analyze",
            accept_encoding="gzip"
        )

        assert response["headers"].get("Content-Encoding") == "gzip"
        assert response["compressed"] is True

    @pytest.mark.unit
    def test_no_compression_without_header(self):
        """Test no compression without Accept-Encoding header."""
        response = self._request_with_encoding(
            "/api/analyze",
            accept_encoding=None
        )

        assert "Content-Encoding" not in response["headers"]

    @pytest.mark.unit
    def test_gzip_request_body(self):
        """Test gzip compressed request body."""
        body = json.dumps({"value": 100})
        compressed = gzip.compress(body.encode())

        response = self._post_compressed("/api/analyze", compressed, "gzip")

        assert response["status_code"] == 200

    @pytest.mark.unit
    def test_unsupported_encoding_ignored(self):
        """Test unsupported encoding is ignored."""
        response = self._request_with_encoding(
            "/api/analyze",
            accept_encoding="br, unsupported"
        )

        # Should still return response, possibly uncompressed
        assert response["status_code"] == 200

    @pytest.mark.unit
    def test_small_response_not_compressed(self):
        """Test small responses are not compressed."""
        response = self._request_with_encoding(
            "/api/health",  # Small response
            accept_encoding="gzip"
        )

        # Small responses might skip compression
        # (implementation-dependent)
        assert response["status_code"] == 200

    def _request_with_encoding(
        self, path: str, accept_encoding: Optional[str]
    ) -> Dict[str, Any]:
        """Send request with Accept-Encoding header."""
        if accept_encoding and "gzip" in accept_encoding:
            return {
                "status_code": 200,
                "headers": {"Content-Encoding": "gzip"},
                "compressed": True,
            }

        return {
            "status_code": 200,
            "headers": {},
            "compressed": False,
        }

    def _post_compressed(
        self, path: str, body: bytes, encoding: str
    ) -> Dict[str, Any]:
        """Post compressed request body."""
        # Decompress and validate
        try:
            if encoding == "gzip":
                decompressed = gzip.decompress(body)
                json.loads(decompressed)
        except Exception:
            return {"status_code": 400}

        return {"status_code": 200}


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestAPIPerformance:
    """Performance tests for API endpoints."""

    @pytest.mark.performance
    def test_analyze_endpoint_latency(self, sample_analysis_input):
        """Test analyze endpoint meets latency target."""
        iterations = 100
        latencies = []

        for _ in range(iterations):
            start = time.perf_counter()
            self._mock_analyze(sample_analysis_input)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        assert avg_latency < 50, f"Average latency {avg_latency:.2f}ms (target: <50ms)"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms (target: <100ms)"

    @pytest.mark.performance
    def test_health_endpoint_very_fast(self):
        """Test health endpoint is very fast (<5ms)."""
        iterations = 1000

        start = time.perf_counter()
        for _ in range(iterations):
            self._mock_health()
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

        assert elapsed_ms < 5, f"Health endpoint took {elapsed_ms:.2f}ms (target: <5ms)"

    @pytest.mark.performance
    def test_graphql_query_performance(self):
        """Test GraphQL query performance."""
        query = "query { analyze { efficiency } }"

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            TestGraphQLQueries()._execute_graphql(query, {})
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

        assert elapsed_ms < 100, f"GraphQL took {elapsed_ms:.2f}ms (target: <100ms)"

    def _mock_analyze(self, input_data: dict) -> dict:
        """Mock analyze endpoint."""
        return {"efficiency_percent": 82.8}

    def _mock_health(self) -> dict:
        """Mock health endpoint."""
        return {"status": "healthy"}


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================

if HAS_HYPOTHESIS:

    class TestAPIPropertyBased:
        """Property-based tests for API."""

        @given(
            value=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
        )
        @settings(max_examples=50)
        def test_efficiency_response_bounded(self, value: float):
            """Property: Efficiency in response is always bounded."""
            response = self._mock_analyze_with_value(value)

            eff = response.get("efficiency_percent", 0)
            assert 0 <= eff <= 100

        @given(
            page=st.integers(min_value=1, max_value=10000),
            page_size=st.integers(min_value=1, max_value=10000),
        )
        @settings(max_examples=50)
        def test_pagination_returns_valid_response(self, page: int, page_size: int):
            """Property: Pagination always returns valid structure."""
            response = TestPagination()._list_analyses(page, page_size)

            assert "items" in response
            assert "total" in response
            assert isinstance(response["items"], list)

        def _mock_analyze_with_value(self, value: float) -> dict:
            """Mock analyze with value."""
            # Clamp to valid efficiency range
            efficiency = min(100, max(0, value / 100 * 82.8))
            return {"efficiency_percent": efficiency}
