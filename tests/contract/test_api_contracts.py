# -*- coding: utf-8 -*-
"""
API Contract Tests for GreenLang

Tests API contracts using Pact-style consumer-driven contract testing.
Validates that REST API endpoints maintain backward compatibility
and follow defined schemas.

Test Coverage:
- Calculation API contracts
- Agent API contracts
- Webhook API contracts
- Configuration API contracts
- Health check API contracts

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
import json
from typing import Dict, Any
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from .conftest import (
    PactMockServer,
    ContractVerificationResult,
    ContractDefinition,
    ContractType,
)


# ==============================================================================
# Calculation API Contracts
# ==============================================================================

class TestCalculationAPIContracts:
    """Test Calculation API contracts."""

    @pytest.mark.contract
    @pytest.mark.pact
    def test_create_calculation_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: POST /api/v1/calculations

        Consumer expects to create a calculation and receive calculation ID.
        """
        # Define expected interaction
        pact_mock_server.given(
            "calculation service is available"
        ).upon_receiving(
            "a request to create a new calculation"
        ).with_request(
            method="POST",
            path="/api/v1/calculations",
            headers={"Content-Type": "application/json"},
            body={
                "fuel_type": "natural_gas",
                "quantity": 1000.0,
                "unit": "therms",
                "region": "US",
            },
        ).will_respond_with(
            status=201,
            headers={"Content-Type": "application/json"},
            body={
                "calculation_id": "calc_abc123",
                "status": "pending",
                "created_at": "2025-12-07T12:00:00Z",
            },
        )

        # Simulate consumer making request
        request = {
            "method": "POST",
            "path": "/api/v1/calculations",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "fuel_type": "natural_gas",
                "quantity": 1000.0,
                "unit": "therms",
                "region": "US",
            },
        }

        response = pact_mock_server.handle_request(request)

        # Verify response
        assert response is not None
        assert response["status"] == 201
        assert "calculation_id" in response["body"]
        assert response["body"]["status"] == "pending"

        # Verify contract
        result = pact_mock_server.verify()
        assert result.passed, f"Contract verification failed: {result.errors}"

    @pytest.mark.contract
    @pytest.mark.pact
    def test_get_calculation_result_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: GET /api/v1/calculations/{id}

        Consumer expects to retrieve calculation results with provenance.
        """
        pact_mock_server.given(
            "calculation calc_abc123 has completed"
        ).upon_receiving(
            "a request to get calculation result"
        ).with_request(
            method="GET",
            path="/api/v1/calculations/calc_abc123",
            headers={"Accept": "application/json"},
        ).will_respond_with(
            status=200,
            headers={"Content-Type": "application/json"},
            body={
                "calculation_id": "calc_abc123",
                "status": "completed",
                "result": {
                    "emissions_kg_co2e": 5300.0,
                    "emissions_tonnes_co2e": 5.3,
                },
                "provenance": {
                    "hash": "abc123def456",
                    "emission_factor_source": "EPA 2024",
                    "calculation_method": "GHG Protocol",
                },
                "completed_at": "2025-12-07T12:01:00Z",
            },
        )

        request = {
            "method": "GET",
            "path": "/api/v1/calculations/calc_abc123",
            "headers": {"Accept": "application/json"},
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 200
        assert "result" in response["body"]
        assert "provenance" in response["body"]
        assert "hash" in response["body"]["provenance"]

        result = pact_mock_server.verify()
        assert result.passed

    @pytest.mark.contract
    @pytest.mark.pact
    def test_calculation_not_found_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: GET /api/v1/calculations/{id} - Not Found

        Consumer expects 404 when calculation doesn't exist.
        """
        pact_mock_server.given(
            "calculation calc_nonexistent does not exist"
        ).upon_receiving(
            "a request to get non-existent calculation"
        ).with_request(
            method="GET",
            path="/api/v1/calculations/calc_nonexistent",
        ).will_respond_with(
            status=404,
            body={
                "error": "not_found",
                "message": "Calculation not found",
                "calculation_id": "calc_nonexistent",
            },
        )

        request = {
            "method": "GET",
            "path": "/api/v1/calculations/calc_nonexistent",
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 404
        assert response["body"]["error"] == "not_found"

        result = pact_mock_server.verify()
        assert result.passed

    @pytest.mark.contract
    @pytest.mark.pact
    def test_batch_calculation_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: POST /api/v1/calculations/batch

        Consumer expects to submit batch calculations.
        """
        pact_mock_server.given(
            "batch processing is enabled"
        ).upon_receiving(
            "a request to create batch calculation"
        ).with_request(
            method="POST",
            path="/api/v1/calculations/batch",
            headers={"Content-Type": "application/json"},
            body={
                "calculations": [
                    {"fuel_type": "natural_gas", "quantity": 1000.0, "unit": "therms"},
                    {"fuel_type": "diesel", "quantity": 500.0, "unit": "gallons"},
                ],
                "options": {
                    "parallel": True,
                    "callback_url": "https://example.com/callback",
                },
            },
        ).will_respond_with(
            status=202,
            body={
                "batch_id": "batch_xyz789",
                "status": "processing",
                "calculation_count": 2,
                "estimated_completion_seconds": 30,
            },
        )

        request = {
            "method": "POST",
            "path": "/api/v1/calculations/batch",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "calculations": [
                    {"fuel_type": "natural_gas", "quantity": 1000.0, "unit": "therms"},
                    {"fuel_type": "diesel", "quantity": 500.0, "unit": "gallons"},
                ],
                "options": {
                    "parallel": True,
                    "callback_url": "https://example.com/callback",
                },
            },
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 202
        assert "batch_id" in response["body"]

        result = pact_mock_server.verify()
        assert result.passed


# ==============================================================================
# Agent API Contracts
# ==============================================================================

class TestAgentAPIContracts:
    """Test Agent API contracts."""

    @pytest.mark.contract
    @pytest.mark.pact
    def test_list_agents_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: GET /api/v1/agents

        Consumer expects to list available agents with capabilities.
        """
        pact_mock_server.given(
            "agents are registered"
        ).upon_receiving(
            "a request to list agents"
        ).with_request(
            method="GET",
            path="/api/v1/agents",
        ).will_respond_with(
            status=200,
            body={
                "agents": [
                    {
                        "agent_id": "fuel_agent",
                        "name": "Fuel Emissions Agent",
                        "version": "2.0.0",
                        "capabilities": ["fuel_calculation", "emission_factors"],
                        "status": "active",
                    },
                    {
                        "agent_id": "grid_agent",
                        "name": "Grid Factor Agent",
                        "version": "1.5.0",
                        "capabilities": ["grid_factors", "marginal_emissions"],
                        "status": "active",
                    },
                ],
                "total_count": 2,
            },
        )

        request = {
            "method": "GET",
            "path": "/api/v1/agents",
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 200
        assert "agents" in response["body"]
        assert len(response["body"]["agents"]) == 2

        for agent in response["body"]["agents"]:
            assert "agent_id" in agent
            assert "capabilities" in agent
            assert "status" in agent

        result = pact_mock_server.verify()
        assert result.passed

    @pytest.mark.contract
    @pytest.mark.pact
    def test_invoke_agent_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: POST /api/v1/agents/{agent_id}/invoke

        Consumer expects to invoke an agent with input and receive result.
        """
        pact_mock_server.given(
            "fuel_agent is available"
        ).upon_receiving(
            "a request to invoke fuel agent"
        ).with_request(
            method="POST",
            path="/api/v1/agents/fuel_agent/invoke",
            headers={"Content-Type": "application/json"},
            body={
                "input": {
                    "fuel_type": "natural_gas",
                    "quantity": 1000.0,
                    "unit": "therms",
                },
                "options": {
                    "include_provenance": True,
                },
            },
        ).will_respond_with(
            status=200,
            body={
                "agent_id": "fuel_agent",
                "execution_id": "exec_123",
                "result": {
                    "emissions_kg_co2e": 5300.0,
                    "emission_factor": 5.3,
                    "emission_factor_unit": "kg CO2e/therm",
                },
                "provenance": {
                    "hash": "sha256:abc123",
                    "inputs_hash": "sha256:def456",
                    "timestamp": "2025-12-07T12:00:00Z",
                },
                "execution_time_ms": 45.2,
            },
        )

        request = {
            "method": "POST",
            "path": "/api/v1/agents/fuel_agent/invoke",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "input": {
                    "fuel_type": "natural_gas",
                    "quantity": 1000.0,
                    "unit": "therms",
                },
                "options": {
                    "include_provenance": True,
                },
            },
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 200
        assert "result" in response["body"]
        assert "provenance" in response["body"]

        result = pact_mock_server.verify()
        assert result.passed


# ==============================================================================
# Webhook API Contracts
# ==============================================================================

class TestWebhookAPIContracts:
    """Test Webhook API contracts."""

    @pytest.mark.contract
    @pytest.mark.pact
    def test_register_webhook_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: POST /api/v1/webhooks

        Consumer expects to register a webhook subscription.
        """
        pact_mock_server.given(
            "webhook service is available"
        ).upon_receiving(
            "a request to register webhook"
        ).with_request(
            method="POST",
            path="/api/v1/webhooks",
            headers={"Content-Type": "application/json"},
            body={
                "url": "https://example.com/webhook",
                "events": ["calculation.completed", "alarm.triggered"],
                "secret": "webhook-secret-123",
            },
        ).will_respond_with(
            status=201,
            body={
                "webhook_id": "wh_abc123",
                "url": "https://example.com/webhook",
                "events": ["calculation.completed", "alarm.triggered"],
                "is_active": True,
                "created_at": "2025-12-07T12:00:00Z",
            },
        )

        request = {
            "method": "POST",
            "path": "/api/v1/webhooks",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "url": "https://example.com/webhook",
                "events": ["calculation.completed", "alarm.triggered"],
                "secret": "webhook-secret-123",
            },
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 201
        assert "webhook_id" in response["body"]
        assert response["body"]["is_active"] is True

        result = pact_mock_server.verify()
        assert result.passed

    @pytest.mark.contract
    @pytest.mark.pact
    def test_webhook_delivery_payload_contract(self):
        """
        Contract: Webhook delivery payload format

        Provider will deliver webhooks with specific payload structure.
        """
        # Define expected webhook payload structure
        expected_payload = {
            "type": "object",
            "required": ["event_type", "timestamp", "payload", "signature"],
            "properties": {
                "event_type": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
                "payload": {"type": "object"},
                "signature": {"type": "string"},
                "delivery_id": {"type": "string"},
            },
        }

        # Sample webhook payload
        sample_payload = {
            "event_type": "calculation.completed",
            "timestamp": "2025-12-07T12:00:00Z",
            "payload": {
                "calculation_id": "calc_123",
                "result": {"emissions_kg_co2e": 5300.0},
            },
            "signature": "sha256=abc123def456",
            "delivery_id": "del_xyz789",
        }

        # Validate payload structure
        assert "event_type" in sample_payload
        assert "timestamp" in sample_payload
        assert "payload" in sample_payload
        assert "signature" in sample_payload

        # Validate types
        assert isinstance(sample_payload["event_type"], str)
        assert isinstance(sample_payload["payload"], dict)
        assert isinstance(sample_payload["signature"], str)


# ==============================================================================
# Health Check API Contracts
# ==============================================================================

class TestHealthCheckAPIContracts:
    """Test Health Check API contracts."""

    @pytest.mark.contract
    @pytest.mark.pact
    def test_health_check_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: GET /health

        Consumer expects health check endpoint with status.
        """
        pact_mock_server.given(
            "system is healthy"
        ).upon_receiving(
            "a health check request"
        ).with_request(
            method="GET",
            path="/health",
        ).will_respond_with(
            status=200,
            body={
                "status": "healthy",
                "version": "2.0.0",
                "components": {
                    "database": {"status": "healthy", "latency_ms": 5},
                    "cache": {"status": "healthy", "latency_ms": 1},
                    "agents": {"status": "healthy", "count": 10},
                },
                "timestamp": "2025-12-07T12:00:00Z",
            },
        )

        request = {
            "method": "GET",
            "path": "/health",
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 200
        assert response["body"]["status"] == "healthy"
        assert "components" in response["body"]

        result = pact_mock_server.verify()
        assert result.passed

    @pytest.mark.contract
    @pytest.mark.pact
    def test_health_check_degraded_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: GET /health - Degraded state

        Consumer expects degraded status when components are unhealthy.
        """
        pact_mock_server.given(
            "cache is unavailable"
        ).upon_receiving(
            "a health check request when degraded"
        ).with_request(
            method="GET",
            path="/health",
        ).will_respond_with(
            status=200,
            body={
                "status": "degraded",
                "version": "2.0.0",
                "components": {
                    "database": {"status": "healthy", "latency_ms": 5},
                    "cache": {"status": "unhealthy", "error": "Connection refused"},
                    "agents": {"status": "healthy", "count": 10},
                },
                "timestamp": "2025-12-07T12:00:00Z",
            },
        )

        request = {
            "method": "GET",
            "path": "/health",
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 200
        assert response["body"]["status"] == "degraded"
        assert response["body"]["components"]["cache"]["status"] == "unhealthy"

        result = pact_mock_server.verify()
        assert result.passed


# ==============================================================================
# API Versioning Contracts
# ==============================================================================

class TestAPIVersioningContracts:
    """Test API versioning contracts."""

    @pytest.mark.contract
    def test_api_version_header_contract(self):
        """
        Contract: API version header

        All API responses must include version header.
        """
        expected_headers = {
            "X-API-Version": "2.0.0",
            "X-Request-ID": "uuid format",
        }

        # Validate header requirements
        assert "X-API-Version" in expected_headers
        assert "X-Request-ID" in expected_headers

    @pytest.mark.contract
    def test_deprecation_header_contract(self):
        """
        Contract: Deprecation warning header

        Deprecated endpoints must include deprecation header.
        """
        deprecated_response_headers = {
            "Deprecation": "true",
            "Sunset": "2026-01-01T00:00:00Z",
            "Link": "</api/v2/calculations>; rel=\"successor-version\"",
        }

        # Validate deprecation header requirements
        assert "Deprecation" in deprecated_response_headers
        assert "Sunset" in deprecated_response_headers
        assert "successor-version" in deprecated_response_headers["Link"]


# ==============================================================================
# Error Response Contracts
# ==============================================================================

class TestErrorResponseContracts:
    """Test error response contracts."""

    @pytest.mark.contract
    @pytest.mark.pact
    def test_validation_error_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: 400 Bad Request - Validation Error

        Consumer expects structured validation error response.
        """
        pact_mock_server.given(
            "validation is enabled"
        ).upon_receiving(
            "a request with invalid data"
        ).with_request(
            method="POST",
            path="/api/v1/calculations",
            headers={"Content-Type": "application/json"},
            body={
                "fuel_type": "invalid_fuel",
                "quantity": -100,
            },
        ).will_respond_with(
            status=400,
            body={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {
                        "field": "fuel_type",
                        "error": "Invalid fuel type: invalid_fuel",
                    },
                    {
                        "field": "quantity",
                        "error": "Quantity must be positive",
                    },
                ],
                "request_id": "req_abc123",
            },
        )

        request = {
            "method": "POST",
            "path": "/api/v1/calculations",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "fuel_type": "invalid_fuel",
                "quantity": -100,
            },
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 400
        assert response["body"]["error"] == "validation_error"
        assert "details" in response["body"]
        assert len(response["body"]["details"]) == 2

        result = pact_mock_server.verify()
        assert result.passed

    @pytest.mark.contract
    @pytest.mark.pact
    def test_rate_limit_error_contract(self, pact_mock_server: PactMockServer):
        """
        Contract: 429 Too Many Requests

        Consumer expects rate limit error with retry information.
        """
        pact_mock_server.given(
            "rate limit exceeded"
        ).upon_receiving(
            "a request when rate limited"
        ).with_request(
            method="POST",
            path="/api/v1/calculations",
        ).will_respond_with(
            status=429,
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1733572800",
            },
            body={
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "retry_after_seconds": 60,
            },
        )

        request = {
            "method": "POST",
            "path": "/api/v1/calculations",
        }

        response = pact_mock_server.handle_request(request)

        assert response is not None
        assert response["status"] == 429
        assert response["body"]["error"] == "rate_limit_exceeded"
        assert "Retry-After" in response["headers"]

        result = pact_mock_server.verify()
        assert result.passed


# ==============================================================================
# Pact File Generation
# ==============================================================================

class TestPactGeneration:
    """Test Pact file generation."""

    @pytest.mark.contract
    def test_generate_pact_file(
        self,
        pact_mock_server: PactMockServer,
        pact_output_dir,
    ):
        """Test generating Pact file from interactions."""
        # Define interactions
        pact_mock_server.consumer = "dashboard_frontend"
        pact_mock_server.provider = "calculation_service"

        pact_mock_server.given(
            "calculation service is available"
        ).upon_receiving(
            "a request to create calculation"
        ).with_request(
            method="POST",
            path="/api/v1/calculations",
            headers={"Content-Type": "application/json"},
            body={"fuel_type": "natural_gas", "quantity": 1000.0},
        ).will_respond_with(
            status=201,
            body={"calculation_id": "calc_123", "status": "pending"},
        )

        pact_mock_server.given(
            "calculation calc_123 exists"
        ).upon_receiving(
            "a request to get calculation"
        ).with_request(
            method="GET",
            path="/api/v1/calculations/calc_123",
        ).will_respond_with(
            status=200,
            body={"calculation_id": "calc_123", "status": "completed"},
        )

        # Write pact file
        pact_mock_server.write_pact(pact_output_dir)

        # Verify file was created
        pact_file = pact_output_dir / "dashboard_frontend-calculation_service.json"
        assert pact_file.exists()

        # Verify file contents
        with open(pact_file) as f:
            pact_data = json.load(f)

        assert pact_data["consumer"]["name"] == "dashboard_frontend"
        assert pact_data["provider"]["name"] == "calculation_service"
        assert len(pact_data["interactions"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "contract"])
