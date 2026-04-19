# -*- coding: utf-8 -*-
"""
gRPC Service Contract Tests for GreenLang

Tests gRPC service contracts to ensure client and server agree on
method signatures, request/response schemas, and error handling.

Test Coverage:
- CalculationService contracts
- AgentService contracts
- StreamingService contracts
- Error handling contracts
- Metadata contracts

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .conftest import GrpcContractVerifier, GrpcMethodContract, ContractVerificationResult


# ==============================================================================
# gRPC Method Definitions
# ==============================================================================

# CalculationService Methods
CALCULATE_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["input_data", "calculation_type"],
    "properties": {
        "input_data": {
            "type": "object",
            "required": ["fuel_type", "quantity", "unit"],
            "properties": {
                "fuel_type": {"type": "string"},
                "quantity": {"type": "number", "minimum": 0},
                "unit": {"type": "string"},
                "region": {"type": "string"},
            },
        },
        "calculation_type": {
            "type": "string",
            "enum": ["scope1", "scope2", "scope3"],
        },
        "options": {
            "type": "object",
            "properties": {
                "include_provenance": {"type": "boolean"},
                "emission_factor_source": {"type": "string"},
                "gwp_version": {"type": "string"},
            },
        },
    },
}

CALCULATE_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["result", "provenance_hash", "execution_time_ms"],
    "properties": {
        "result": {
            "type": "object",
            "required": ["emissions_kg_co2e"],
            "properties": {
                "emissions_kg_co2e": {"type": "number"},
                "emissions_tonnes_co2e": {"type": "number"},
                "emission_factor": {"type": "number"},
                "emission_factor_unit": {"type": "string"},
            },
        },
        "provenance_hash": {"type": "string", "minLength": 64},
        "execution_time_ms": {"type": "number", "minimum": 0},
        "metadata": {"type": "object"},
    },
}

BATCH_CALCULATE_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["calculations"],
    "properties": {
        "calculations": {
            "type": "array",
            "items": CALCULATE_REQUEST_SCHEMA,
            "minItems": 1,
            "maxItems": 1000,
        },
        "options": {
            "type": "object",
            "properties": {
                "parallel": {"type": "boolean"},
                "fail_fast": {"type": "boolean"},
                "timeout_seconds": {"type": "integer"},
            },
        },
    },
}

BATCH_CALCULATE_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["results", "summary"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "status"],
                "properties": {
                    "index": {"type": "integer"},
                    "status": {"type": "string", "enum": ["success", "failed"]},
                    "result": CALCULATE_RESPONSE_SCHEMA,
                    "error": {"type": "object"},
                },
            },
        },
        "summary": {
            "type": "object",
            "required": ["total", "succeeded", "failed"],
            "properties": {
                "total": {"type": "integer"},
                "succeeded": {"type": "integer"},
                "failed": {"type": "integer"},
                "total_execution_time_ms": {"type": "number"},
            },
        },
    },
}

# AgentService Methods
INVOKE_AGENT_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["agent_id", "input"],
    "properties": {
        "agent_id": {"type": "string"},
        "input": {"type": "object"},
        "options": {
            "type": "object",
            "properties": {
                "timeout_seconds": {"type": "integer"},
                "include_trace": {"type": "boolean"},
            },
        },
    },
}

INVOKE_AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["agent_id", "execution_id", "result"],
    "properties": {
        "agent_id": {"type": "string"},
        "execution_id": {"type": "string"},
        "result": {"type": "object"},
        "trace": {
            "type": "object",
            "properties": {
                "steps": {"type": "array"},
                "tool_calls": {"type": "array"},
            },
        },
        "execution_time_ms": {"type": "number"},
    },
}

# Streaming Methods
STREAM_CALCULATIONS_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["calculation"],
    "properties": {
        "calculation": CALCULATE_REQUEST_SCHEMA,
        "sequence_number": {"type": "integer"},
    },
}

STREAM_CALCULATIONS_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["sequence_number", "status"],
    "properties": {
        "sequence_number": {"type": "integer"},
        "status": {"type": "string"},
        "result": CALCULATE_RESPONSE_SCHEMA,
        "error": {"type": "object"},
    },
}


# ==============================================================================
# Error Schema Definitions
# ==============================================================================

GRPC_ERROR_SCHEMA = {
    "type": "object",
    "required": ["code", "message"],
    "properties": {
        "code": {
            "type": "string",
            "enum": [
                "INVALID_ARGUMENT",
                "NOT_FOUND",
                "PERMISSION_DENIED",
                "RESOURCE_EXHAUSTED",
                "INTERNAL",
                "UNAVAILABLE",
                "DEADLINE_EXCEEDED",
            ],
        },
        "message": {"type": "string"},
        "details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "field": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
    },
}


# ==============================================================================
# CalculationService Contract Tests
# ==============================================================================

class TestCalculationServiceContracts:
    """Test CalculationService gRPC contracts."""

    @pytest.fixture
    def verifier(self) -> GrpcContractVerifier:
        """Create verifier with CalculationService contracts."""
        verifier = GrpcContractVerifier()

        verifier.register_method(GrpcMethodContract(
            service_name="CalculationService",
            method_name="Calculate",
            request_type="CalculateRequest",
            response_type="CalculateResponse",
            request_schema=CALCULATE_REQUEST_SCHEMA,
            response_schema=CALCULATE_RESPONSE_SCHEMA,
        ))

        verifier.register_method(GrpcMethodContract(
            service_name="CalculationService",
            method_name="BatchCalculate",
            request_type="BatchCalculateRequest",
            response_type="BatchCalculateResponse",
            request_schema=BATCH_CALCULATE_REQUEST_SCHEMA,
            response_schema=BATCH_CALCULATE_RESPONSE_SCHEMA,
        ))

        return verifier

    @pytest.mark.grpc_contract
    def test_calculate_request_valid(self, verifier: GrpcContractVerifier):
        """Test valid Calculate request."""
        request = {
            "input_data": {
                "fuel_type": "natural_gas",
                "quantity": 1000.0,
                "unit": "therms",
                "region": "US",
            },
            "calculation_type": "scope1",
            "options": {
                "include_provenance": True,
                "emission_factor_source": "EPA",
            },
        }

        result = verifier.verify_request("CalculationService", "Calculate", request)

        assert result.passed, f"Request validation failed: {result.errors}"

    @pytest.mark.grpc_contract
    def test_calculate_request_missing_required(self, verifier: GrpcContractVerifier):
        """Test Calculate request missing required fields."""
        request = {
            "input_data": {
                "fuel_type": "natural_gas",
                # Missing: quantity, unit
            },
            # Missing: calculation_type
        }

        result = verifier.verify_request("CalculationService", "Calculate", request)

        assert not result.passed
        assert any("calculation_type" in e for e in result.errors)

    @pytest.mark.grpc_contract
    def test_calculate_response_valid(self, verifier: GrpcContractVerifier):
        """Test valid Calculate response."""
        response = {
            "result": {
                "emissions_kg_co2e": 5300.0,
                "emissions_tonnes_co2e": 5.3,
                "emission_factor": 5.3,
                "emission_factor_unit": "kg CO2e/therm",
            },
            "provenance_hash": "a" * 64,
            "execution_time_ms": 45.2,
            "metadata": {"region": "US"},
        }

        result = verifier.verify_response("CalculationService", "Calculate", response)

        assert result.passed, f"Response validation failed: {result.errors}"

    @pytest.mark.grpc_contract
    def test_batch_calculate_request_valid(self, verifier: GrpcContractVerifier):
        """Test valid BatchCalculate request."""
        request = {
            "calculations": [
                {
                    "input_data": {
                        "fuel_type": "natural_gas",
                        "quantity": 1000.0,
                        "unit": "therms",
                    },
                    "calculation_type": "scope1",
                },
                {
                    "input_data": {
                        "fuel_type": "diesel",
                        "quantity": 500.0,
                        "unit": "gallons",
                    },
                    "calculation_type": "scope1",
                },
            ],
            "options": {
                "parallel": True,
                "fail_fast": False,
            },
        }

        result = verifier.verify_request("CalculationService", "BatchCalculate", request)

        assert result.passed, f"Request validation failed: {result.errors}"

    @pytest.mark.grpc_contract
    def test_batch_calculate_response_valid(self, verifier: GrpcContractVerifier):
        """Test valid BatchCalculate response."""
        response = {
            "results": [
                {
                    "index": 0,
                    "status": "success",
                    "result": {
                        "result": {"emissions_kg_co2e": 5300.0},
                        "provenance_hash": "a" * 64,
                        "execution_time_ms": 40.0,
                    },
                },
                {
                    "index": 1,
                    "status": "success",
                    "result": {
                        "result": {"emissions_kg_co2e": 1340.0},
                        "provenance_hash": "b" * 64,
                        "execution_time_ms": 35.0,
                    },
                },
            ],
            "summary": {
                "total": 2,
                "succeeded": 2,
                "failed": 0,
                "total_execution_time_ms": 75.0,
            },
        }

        result = verifier.verify_response("CalculationService", "BatchCalculate", response)

        assert result.passed, f"Response validation failed: {result.errors}"


# ==============================================================================
# AgentService Contract Tests
# ==============================================================================

class TestAgentServiceContracts:
    """Test AgentService gRPC contracts."""

    @pytest.fixture
    def verifier(self) -> GrpcContractVerifier:
        """Create verifier with AgentService contracts."""
        verifier = GrpcContractVerifier()

        verifier.register_method(GrpcMethodContract(
            service_name="AgentService",
            method_name="InvokeAgent",
            request_type="InvokeAgentRequest",
            response_type="InvokeAgentResponse",
            request_schema=INVOKE_AGENT_REQUEST_SCHEMA,
            response_schema=INVOKE_AGENT_RESPONSE_SCHEMA,
        ))

        return verifier

    @pytest.mark.grpc_contract
    def test_invoke_agent_request_valid(self, verifier: GrpcContractVerifier):
        """Test valid InvokeAgent request."""
        request = {
            "agent_id": "fuel_agent",
            "input": {
                "fuel_type": "natural_gas",
                "quantity": 1000.0,
                "unit": "therms",
            },
            "options": {
                "timeout_seconds": 30,
                "include_trace": True,
            },
        }

        result = verifier.verify_request("AgentService", "InvokeAgent", request)

        assert result.passed, f"Request validation failed: {result.errors}"

    @pytest.mark.grpc_contract
    def test_invoke_agent_response_valid(self, verifier: GrpcContractVerifier):
        """Test valid InvokeAgent response."""
        response = {
            "agent_id": "fuel_agent",
            "execution_id": "exec_abc123",
            "result": {
                "emissions_kg_co2e": 5300.0,
                "recommendation": "Consider switching to renewable energy",
            },
            "trace": {
                "steps": [
                    {"name": "parse_input", "duration_ms": 5},
                    {"name": "lookup_factor", "duration_ms": 10},
                    {"name": "calculate", "duration_ms": 20},
                ],
                "tool_calls": [
                    {"tool": "emission_factor_lookup", "success": True},
                ],
            },
            "execution_time_ms": 35.0,
        }

        result = verifier.verify_response("AgentService", "InvokeAgent", response)

        assert result.passed, f"Response validation failed: {result.errors}"

    @pytest.mark.grpc_contract
    def test_invoke_agent_minimal_response(self, verifier: GrpcContractVerifier):
        """Test minimal valid InvokeAgent response."""
        response = {
            "agent_id": "fuel_agent",
            "execution_id": "exec_xyz789",
            "result": {},
        }

        result = verifier.verify_response("AgentService", "InvokeAgent", response)

        assert result.passed


# ==============================================================================
# Streaming Service Contract Tests
# ==============================================================================

class TestStreamingServiceContracts:
    """Test streaming gRPC contracts."""

    @pytest.fixture
    def verifier(self) -> GrpcContractVerifier:
        """Create verifier with streaming contracts."""
        verifier = GrpcContractVerifier()

        verifier.register_method(GrpcMethodContract(
            service_name="StreamingService",
            method_name="StreamCalculations",
            request_type="StreamCalculationsRequest",
            response_type="StreamCalculationsResponse",
            request_schema=STREAM_CALCULATIONS_REQUEST_SCHEMA,
            response_schema=STREAM_CALCULATIONS_RESPONSE_SCHEMA,
        ))

        return verifier

    @pytest.mark.grpc_contract
    def test_stream_request_valid(self, verifier: GrpcContractVerifier):
        """Test valid streaming request."""
        request = {
            "calculation": {
                "input_data": {
                    "fuel_type": "natural_gas",
                    "quantity": 1000.0,
                    "unit": "therms",
                },
                "calculation_type": "scope1",
            },
            "sequence_number": 0,
        }

        result = verifier.verify_request("StreamingService", "StreamCalculations", request)

        assert result.passed

    @pytest.mark.grpc_contract
    def test_stream_response_success(self, verifier: GrpcContractVerifier):
        """Test successful streaming response."""
        response = {
            "sequence_number": 0,
            "status": "completed",
            "result": {
                "result": {"emissions_kg_co2e": 5300.0},
                "provenance_hash": "a" * 64,
                "execution_time_ms": 45.0,
            },
        }

        result = verifier.verify_response("StreamingService", "StreamCalculations", response)

        assert result.passed

    @pytest.mark.grpc_contract
    def test_stream_response_error(self, verifier: GrpcContractVerifier):
        """Test streaming response with error."""
        response = {
            "sequence_number": 1,
            "status": "failed",
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "Invalid fuel type",
            },
        }

        result = verifier.verify_response("StreamingService", "StreamCalculations", response)

        assert result.passed


# ==============================================================================
# Error Handling Contract Tests
# ==============================================================================

class TestGrpcErrorContracts:
    """Test gRPC error handling contracts."""

    @pytest.fixture
    def verifier(self) -> GrpcContractVerifier:
        """Create verifier with error schema."""
        verifier = GrpcContractVerifier()

        verifier.register_method(GrpcMethodContract(
            service_name="ErrorContract",
            method_name="ValidateError",
            request_type="ErrorRequest",
            response_type="ErrorResponse",
            request_schema={"type": "object"},
            response_schema=GRPC_ERROR_SCHEMA,
        ))

        return verifier

    @pytest.mark.grpc_contract
    def test_invalid_argument_error(self, verifier: GrpcContractVerifier):
        """Test INVALID_ARGUMENT error format."""
        error = {
            "code": "INVALID_ARGUMENT",
            "message": "Field 'quantity' must be positive",
            "details": [
                {
                    "type": "FieldViolation",
                    "field": "quantity",
                    "description": "Value -100 is not positive",
                },
            ],
        }

        result = verifier.verify_response("ErrorContract", "ValidateError", error)

        assert result.passed

    @pytest.mark.grpc_contract
    def test_not_found_error(self, verifier: GrpcContractVerifier):
        """Test NOT_FOUND error format."""
        error = {
            "code": "NOT_FOUND",
            "message": "Calculation 'calc_nonexistent' not found",
            "details": [
                {
                    "type": "ResourceInfo",
                    "field": "calculation_id",
                    "description": "calc_nonexistent",
                },
            ],
        }

        result = verifier.verify_response("ErrorContract", "ValidateError", error)

        assert result.passed

    @pytest.mark.grpc_contract
    def test_resource_exhausted_error(self, verifier: GrpcContractVerifier):
        """Test RESOURCE_EXHAUSTED (rate limit) error format."""
        error = {
            "code": "RESOURCE_EXHAUSTED",
            "message": "Rate limit exceeded",
            "details": [
                {
                    "type": "RetryInfo",
                    "field": "retry_after_seconds",
                    "description": "60",
                },
            ],
        }

        result = verifier.verify_response("ErrorContract", "ValidateError", error)

        assert result.passed

    @pytest.mark.grpc_contract
    def test_deadline_exceeded_error(self, verifier: GrpcContractVerifier):
        """Test DEADLINE_EXCEEDED (timeout) error format."""
        error = {
            "code": "DEADLINE_EXCEEDED",
            "message": "Request timed out after 30 seconds",
        }

        result = verifier.verify_response("ErrorContract", "ValidateError", error)

        assert result.passed


# ==============================================================================
# Metadata Contract Tests
# ==============================================================================

class TestGrpcMetadataContracts:
    """Test gRPC metadata contracts."""

    METADATA_SCHEMA = {
        "type": "object",
        "properties": {
            "x-request-id": {"type": "string"},
            "x-correlation-id": {"type": "string"},
            "x-client-version": {"type": "string"},
            "authorization": {"type": "string"},
            "x-api-key": {"type": "string"},
        },
    }

    @pytest.mark.grpc_contract
    def test_required_request_metadata(self):
        """Test required request metadata fields."""
        required_metadata = {
            "x-request-id": "req_abc123",
            "x-client-version": "2.0.0",
        }

        # Validate required fields present
        assert "x-request-id" in required_metadata
        assert "x-client-version" in required_metadata

    @pytest.mark.grpc_contract
    def test_authentication_metadata(self):
        """Test authentication metadata format."""
        auth_metadata = {
            "authorization": "Bearer eyJhbGciOiJIUzI1NiIs...",
        }

        # Validate Bearer token format
        assert auth_metadata["authorization"].startswith("Bearer ")

    @pytest.mark.grpc_contract
    def test_response_metadata(self):
        """Test response metadata contract."""
        response_metadata = {
            "x-request-id": "req_abc123",
            "x-execution-time-ms": "45",
            "x-server-version": "2.0.0",
            "x-rate-limit-remaining": "99",
            "x-rate-limit-reset": "1733572800",
        }

        # Validate response metadata
        assert "x-request-id" in response_metadata
        assert "x-execution-time-ms" in response_metadata


# ==============================================================================
# Service Health Contract Tests
# ==============================================================================

class TestGrpcHealthContracts:
    """Test gRPC health check contracts (standard grpc.health.v1)."""

    HEALTH_CHECK_REQUEST_SCHEMA = {
        "type": "object",
        "properties": {
            "service": {"type": "string"},
        },
    }

    HEALTH_CHECK_RESPONSE_SCHEMA = {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["UNKNOWN", "SERVING", "NOT_SERVING", "SERVICE_UNKNOWN"],
            },
        },
    }

    @pytest.fixture
    def verifier(self) -> GrpcContractVerifier:
        """Create verifier with health check contracts."""
        verifier = GrpcContractVerifier()

        verifier.register_method(GrpcMethodContract(
            service_name="grpc.health.v1.Health",
            method_name="Check",
            request_type="HealthCheckRequest",
            response_type="HealthCheckResponse",
            request_schema=self.HEALTH_CHECK_REQUEST_SCHEMA,
            response_schema=self.HEALTH_CHECK_RESPONSE_SCHEMA,
        ))

        return verifier

    @pytest.mark.grpc_contract
    def test_health_check_serving(self, verifier: GrpcContractVerifier):
        """Test health check SERVING response."""
        response = {"status": "SERVING"}

        result = verifier.verify_response("grpc.health.v1.Health", "Check", response)

        assert result.passed

    @pytest.mark.grpc_contract
    def test_health_check_not_serving(self, verifier: GrpcContractVerifier):
        """Test health check NOT_SERVING response."""
        response = {"status": "NOT_SERVING"}

        result = verifier.verify_response("grpc.health.v1.Health", "Check", response)

        assert result.passed

    @pytest.mark.grpc_contract
    def test_health_check_specific_service(self, verifier: GrpcContractVerifier):
        """Test health check for specific service."""
        request = {"service": "CalculationService"}

        result = verifier.verify_request("grpc.health.v1.Health", "Check", request)

        assert result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "grpc_contract"])
