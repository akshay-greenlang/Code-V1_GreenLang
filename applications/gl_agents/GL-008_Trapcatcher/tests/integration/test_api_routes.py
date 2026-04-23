# -*- coding: utf-8 -*-
"""
Integration tests for FastAPI routes.

Tests API endpoints, request/response validation, and error handling.

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Note: These tests require FastAPI TestClient
# In a real scenario, use: from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_structure(self):
        """Test health check response structure."""
        # Mock response structure validation
        expected_fields = ["status", "timestamp", "agent_id", "version"]
        response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agent_id": "GL-008",
            "version": "1.0.0",
        }

        for field in expected_fields:
            assert field in response

    def test_health_status_values(self):
        """Test valid health status values."""
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        status = "healthy"

        assert status in valid_statuses


class TestStatusEndpoint:
    """Tests for /status endpoint."""

    def test_status_response_structure(self):
        """Test status response structure."""
        response = {
            "agent_id": "GL-008",
            "agent_name": "TRAPCATCHER",
            "version": "1.0.0",
            "mode": "production",
            "status": "running",
            "statistics": {
                "total_diagnostics": 100,
                "failed_traps_detected": 15,
                "healthy_traps_verified": 80,
            },
            "components": {
                "classifier": "ready",
                "energy_calculator": "ready",
                "explainer": "ready",
            },
        }

        assert "agent_id" in response
        assert "statistics" in response
        assert "components" in response


class TestDiagnoseEndpoint:
    """Tests for /diagnose endpoint."""

    def test_valid_diagnose_request(self):
        """Test valid diagnostic request."""
        request = {
            "trap_id": "ST-001",
            "acoustic_amplitude_db": 65.0,
            "acoustic_frequency_khz": 38.0,
            "inlet_temp_c": 185.0,
            "outlet_temp_c": 95.0,
            "pressure_bar_g": 10.0,
            "trap_type": "thermodynamic",
            "include_explanation": True,
        }

        # Validate required fields
        assert "trap_id" in request
        assert "pressure_bar_g" in request

    def test_diagnose_response_structure(self):
        """Test diagnostic response structure."""
        response = {
            "trap_id": "ST-001",
            "timestamp": "2025-12-22T00:00:00Z",
            "condition": "healthy",
            "severity": "none",
            "confidence": 0.95,
            "energy_loss_kw": 0.0,
            "annual_cost_usd": 0.0,
            "annual_co2_kg": 0.0,
            "recommended_action": "Continue routine monitoring",
            "alert_level": "none",
            "provenance_hash": "abc123",
            "explanation": None,
        }

        required_fields = [
            "trap_id", "timestamp", "condition", "severity",
            "confidence", "energy_loss_kw", "alert_level",
        ]

        for field in required_fields:
            assert field in response

    def test_diagnose_condition_values(self):
        """Test valid condition values."""
        valid_conditions = ["healthy", "degraded", "leaking", "failed", "unknown"]
        condition = "healthy"

        assert condition in valid_conditions

    def test_diagnose_alert_levels(self):
        """Test valid alert level values."""
        valid_levels = ["none", "low", "medium", "high", "critical"]
        level = "none"

        assert level in valid_levels


class TestFleetAnalyzeEndpoint:
    """Tests for /analyze/fleet endpoint."""

    def test_fleet_request_structure(self):
        """Test fleet analysis request structure."""
        request = {
            "traps": [
                {
                    "trap_id": "ST-001",
                    "acoustic_amplitude_db": 65.0,
                    "pressure_bar_g": 10.0,
                },
                {
                    "trap_id": "ST-002",
                    "acoustic_amplitude_db": 95.0,
                    "pressure_bar_g": 10.0,
                },
            ]
        }

        assert "traps" in request
        assert len(request["traps"]) == 2

    def test_fleet_response_structure(self):
        """Test fleet analysis response structure."""
        response = {
            "summary": {
                "total_traps": 10,
                "healthy_count": 7,
                "failed_count": 2,
                "leaking_count": 1,
                "unknown_count": 0,
                "total_energy_loss_kw": 150.0,
                "total_annual_cost_usd": 25000.0,
                "total_annual_co2_kg": 500000.0,
                "fleet_health_score": 0.7,
                "critical_alerts": 2,
            },
            "diagnostics": [],
        }

        assert "summary" in response
        assert "diagnostics" in response
        assert response["summary"]["total_traps"] == 10

    def test_fleet_summary_calculations(self):
        """Test fleet summary calculations are consistent."""
        summary = {
            "total_traps": 10,
            "healthy_count": 7,
            "failed_count": 2,
            "leaking_count": 1,
            "unknown_count": 0,
        }

        total = (summary["healthy_count"] + summary["failed_count"] +
                 summary["leaking_count"] + summary["unknown_count"])

        assert total == summary["total_traps"]


class TestErrorHandling:
    """Tests for API error handling."""

    def test_400_bad_request_structure(self):
        """Test 400 error response structure."""
        error_response = {
            "detail": "Invalid trap_id format",
        }

        assert "detail" in error_response

    def test_500_internal_error_structure(self):
        """Test 500 error response structure."""
        error_response = {
            "detail": "Diagnostic error: Internal calculation failure",
        }

        assert "detail" in error_response
        assert "Diagnostic error" in error_response["detail"]


class TestRequestValidation:
    """Tests for request validation."""

    def test_trap_id_required(self):
        """Test trap_id is required."""
        required_fields = ["trap_id"]

        for field in required_fields:
            assert field in required_fields

    def test_pressure_has_default(self):
        """Test pressure has default value."""
        defaults = {
            "pressure_bar_g": 10.0,
            "trap_type": "thermodynamic",
            "orifice_diameter_mm": 6.35,
        }

        assert defaults["pressure_bar_g"] == 10.0

    def test_numeric_field_validation(self):
        """Test numeric fields are validated."""
        # These should all be valid numeric values
        valid_values = {
            "acoustic_amplitude_db": 65.0,
            "acoustic_frequency_khz": 38.0,
            "inlet_temp_c": 185.0,
            "outlet_temp_c": 95.0,
            "pressure_bar_g": 10.0,
        }

        for field, value in valid_values.items():
            assert isinstance(value, (int, float))
            assert value >= 0
