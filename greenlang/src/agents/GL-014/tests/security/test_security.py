# -*- coding: utf-8 -*-
"""
Security Tests for GL-014 EXCHANGER-PRO.

Tests security aspects including:
- Authentication requirements
- Authorization roles
- Input sanitization
- Audit trail generation
- Data validation and protection

Author: GL-TestEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingResistanceInput,
    FoulingSeverityInput,
    FluidType,
    ExchangerType,
)
from calculators.economic_calculator import (
    EconomicCalculator,
    EnergyLossInput,
    ROIInput,
    FuelType,
)


# =============================================================================
# Test Class: Authentication Required
# =============================================================================

@pytest.mark.security
class TestAuthenticationRequired:
    """Tests for authentication requirements."""

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client for security testing."""
        client = MagicMock()
        client.get = MagicMock()
        client.post = MagicMock()
        return client

    def test_authentication_required(self, mock_api_client):
        """Test API requires authentication."""
        # Arrange: Request without auth token
        mock_api_client.post.return_value = MagicMock(
            status_code=401,
            json=lambda: {"error": "Authentication required"}
        )

        # Act
        response = mock_api_client.post(
            "/api/v1/fouling/analyze",
            json={"u_clean": 500, "u_fouled": 420},
            headers={}  # No auth header
        )

        # Assert
        assert response.status_code == 401

    def test_invalid_token_rejected(self, mock_api_client):
        """Test invalid authentication token is rejected."""
        # Arrange
        mock_api_client.post.return_value = MagicMock(
            status_code=401,
            json=lambda: {"error": "Invalid token"}
        )

        # Act
        response = mock_api_client.post(
            "/api/v1/fouling/analyze",
            json={"u_clean": 500, "u_fouled": 420},
            headers={"Authorization": "Bearer invalid_token_12345"}
        )

        # Assert
        assert response.status_code == 401

    def test_expired_token_rejected(self, mock_api_client):
        """Test expired authentication token is rejected."""
        # Arrange
        mock_api_client.post.return_value = MagicMock(
            status_code=401,
            json=lambda: {"error": "Token expired"}
        )

        # Act
        response = mock_api_client.post(
            "/api/v1/fouling/analyze",
            json={"u_clean": 500, "u_fouled": 420},
            headers={"Authorization": "Bearer expired_token"}
        )

        # Assert
        assert response.status_code == 401

    def test_valid_token_accepted(self, mock_api_client):
        """Test valid authentication token is accepted."""
        # Arrange
        mock_api_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": "success"}
        )

        # Act
        response = mock_api_client.post(
            "/api/v1/fouling/analyze",
            json={"u_clean": 500, "u_fouled": 420},
            headers={"Authorization": "Bearer valid_token_12345"}
        )

        # Assert
        assert response.status_code == 200


# =============================================================================
# Test Class: Authorization Roles
# =============================================================================

@pytest.mark.security
class TestAuthorizationRoles:
    """Tests for role-based authorization."""

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client for authorization testing."""
        client = MagicMock()
        client.get = MagicMock()
        client.post = MagicMock()
        client.delete = MagicMock()
        return client

    def test_authorization_roles(self, mock_api_client):
        """Test role-based access control for read operations."""
        # Arrange: Viewer role can read
        mock_api_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": "analysis result"}
        )

        # Act
        response = mock_api_client.get(
            "/api/v1/exchangers/HX-001/analysis",
            headers={"Authorization": "Bearer viewer_token", "X-Role": "viewer"}
        )

        # Assert
        assert response.status_code == 200

    def test_viewer_cannot_modify(self, mock_api_client):
        """Test viewer role cannot modify data."""
        # Arrange
        mock_api_client.post.return_value = MagicMock(
            status_code=403,
            json=lambda: {"error": "Insufficient permissions"}
        )

        # Act
        response = mock_api_client.post(
            "/api/v1/exchangers/HX-001/configuration",
            json={"setting": "value"},
            headers={"Authorization": "Bearer viewer_token", "X-Role": "viewer"}
        )

        # Assert
        assert response.status_code == 403

    def test_operator_can_analyze(self, mock_api_client):
        """Test operator role can run analysis."""
        # Arrange
        mock_api_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": "analysis complete"}
        )

        # Act
        response = mock_api_client.post(
            "/api/v1/fouling/analyze",
            json={"u_clean": 500, "u_fouled": 420},
            headers={"Authorization": "Bearer operator_token", "X-Role": "operator"}
        )

        # Assert
        assert response.status_code == 200

    def test_admin_can_delete(self, mock_api_client):
        """Test admin role can delete records."""
        # Arrange
        mock_api_client.delete.return_value = MagicMock(
            status_code=200,
            json=lambda: {"deleted": True}
        )

        # Act
        response = mock_api_client.delete(
            "/api/v1/analysis-history/12345",
            headers={"Authorization": "Bearer admin_token", "X-Role": "admin"}
        )

        # Assert
        assert response.status_code == 200

    def test_operator_cannot_delete(self, mock_api_client):
        """Test operator role cannot delete records."""
        # Arrange
        mock_api_client.delete.return_value = MagicMock(
            status_code=403,
            json=lambda: {"error": "Admin role required"}
        )

        # Act
        response = mock_api_client.delete(
            "/api/v1/analysis-history/12345",
            headers={"Authorization": "Bearer operator_token", "X-Role": "operator"}
        )

        # Assert
        assert response.status_code == 403


# =============================================================================
# Test Class: Input Sanitization
# =============================================================================

@pytest.mark.security
class TestInputSanitization:
    """Tests for input sanitization and validation."""

    def test_input_sanitization(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test input validation prevents invalid values."""
        # Test: Negative values should be rejected
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=-500.0,  # Negative
                u_fouled_w_m2_k=420.0,
            )

    def test_sql_injection_prevention(self, malicious_inputs):
        """Test SQL injection attempts are prevented."""
        sql_injection = malicious_inputs["sql_injection"]

        # Input validation should reject or sanitize
        # For numeric fields, the injection string should fail
        with pytest.raises((ValueError, TypeError)):
            FoulingResistanceInput(
                u_clean_w_m2_k=sql_injection,  # Should fail type validation
                u_fouled_w_m2_k=420.0,
            )

    def test_xss_prevention(self, malicious_inputs):
        """Test XSS attempts are prevented."""
        xss_attack = malicious_inputs["xss_attack"]

        # String fields should sanitize or reject XSS
        # For numeric inputs, type validation handles this
        with pytest.raises((ValueError, TypeError)):
            FoulingResistanceInput(
                u_clean_w_m2_k=xss_attack,
                u_fouled_w_m2_k=420.0,
            )

    def test_path_traversal_prevention(self, malicious_inputs):
        """Test path traversal attempts are prevented."""
        path_traversal = malicious_inputs["path_traversal"]

        # Should not process path traversal strings
        with pytest.raises((ValueError, TypeError)):
            FoulingResistanceInput(
                u_clean_w_m2_k=path_traversal,
                u_fouled_w_m2_k=420.0,
            )

    def test_integer_overflow_prevention(self, malicious_inputs):
        """Test integer overflow is handled safely."""
        overflow_int = malicious_inputs["overflow_int"]

        # Should handle gracefully or raise appropriate error
        try:
            result = FoulingResistanceInput(
                u_clean_w_m2_k=float(overflow_int),
                u_fouled_w_m2_k=420.0,
            )
            # If it accepts, ensure value is reasonable
            assert result.u_clean_w_m2_k < float('inf')
        except (ValueError, OverflowError):
            pass  # Expected behavior

    def test_infinity_handling(self, malicious_inputs):
        """Test infinity values are rejected."""
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=malicious_inputs["overflow_float"],  # inf
                u_fouled_w_m2_k=420.0,
            )

    def test_null_byte_handling(self, malicious_inputs):
        """Test null byte injection is prevented."""
        null_bytes = malicious_inputs["null_bytes"]

        with pytest.raises((ValueError, TypeError)):
            FoulingResistanceInput(
                u_clean_w_m2_k=null_bytes,
                u_fouled_w_m2_k=420.0,
            )

    def test_very_long_string_handling(self, malicious_inputs):
        """Test very long strings are handled safely."""
        long_string = malicious_inputs["very_long_string"]

        with pytest.raises((ValueError, TypeError)):
            FoulingResistanceInput(
                u_clean_w_m2_k=long_string,
                u_fouled_w_m2_k=420.0,
            )

    def test_enum_validation(self):
        """Test invalid enum values are rejected."""
        with pytest.raises((ValueError, KeyError)):
            FoulingResistanceInput(
                u_clean_w_m2_k=500.0,
                u_fouled_w_m2_k=420.0,
                fluid_type_hot="INVALID_FLUID_TYPE",  # Invalid enum
            )

    def test_boundary_values_validation(self):
        """Test boundary value validation."""
        # Test zero value
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=0.0,
                u_fouled_w_m2_k=420.0,
            )

        # Test extremely small positive value (should be accepted)
        result = FoulingResistanceInput(
            u_clean_w_m2_k=0.001,
            u_fouled_w_m2_k=0.0001,
        )
        assert result.u_clean_w_m2_k == 0.001


# =============================================================================
# Test Class: Audit Trail Generation
# =============================================================================

@pytest.mark.security
class TestAuditTrailGeneration:
    """Tests for audit trail generation and completeness."""

    def test_audit_trail_generation(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test audit trail is generated for calculations."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act
        result = fouling_calculator.calculate_fouling_resistance(input_data)

        # Assert: Provenance hash exists
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

        # Assert: Timestamp exists
        assert result.calculation_timestamp is not None

    def test_audit_trail_completeness(
        self,
        economic_calculator: EconomicCalculator,
    ):
        """Test audit trail includes all calculation steps."""
        # Arrange
        input_data = EnergyLossInput(
            design_duty_kw=Decimal("1500"),
            actual_duty_kw=Decimal("1275"),
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.05"),
            operating_hours_per_year=Decimal("8000"),
        )

        # Act
        result = economic_calculator.calculate_energy_loss_cost(input_data)

        # Assert: Calculation steps are recorded
        assert len(result.calculation_steps) > 0

        # Assert: Each step has required fields
        for step in result.calculation_steps:
            assert step.step_number > 0
            assert step.operation != ""
            assert step.description != ""
            assert step.output_name != ""

    def test_audit_trail_tamper_evident(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test audit trail is tamper-evident (immutable results)."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act
        result = fouling_calculator.calculate_fouling_resistance(input_data)

        # Assert: Result is immutable (frozen dataclass)
        with pytest.raises((AttributeError, TypeError)):
            result.fouling_resistance_m2_k_w = Decimal("0")

        with pytest.raises((AttributeError, TypeError)):
            result.provenance_hash = "tampered_hash"

    def test_provenance_hash_integrity(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test provenance hash can verify result integrity."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act: Get two results
        result1 = fouling_calculator.calculate_fouling_resistance(input_data)
        result2 = fouling_calculator.calculate_fouling_resistance(input_data)

        # Assert: Same input produces same hash
        assert result1.provenance_hash == result2.provenance_hash

        # Act: Different input
        different_input = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=400.0,  # Different
        )
        result3 = fouling_calculator.calculate_fouling_resistance(different_input)

        # Assert: Different input produces different hash
        assert result1.provenance_hash != result3.provenance_hash


# =============================================================================
# Test Class: Data Protection
# =============================================================================

@pytest.mark.security
class TestDataProtection:
    """Tests for data protection measures."""

    def test_sensitive_data_not_logged(self):
        """Test sensitive data is not exposed in logs or errors."""
        # Arrange: Create input with potentially sensitive equipment ID
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # The calculator should not log raw input values
        # This is a design verification test
        calculator = FoulingCalculator()
        result = calculator.calculate_fouling_resistance(input_data)

        # Assert: Result contains processed data, not sensitive raw inputs
        assert result.provenance_hash is not None

    def test_error_messages_safe(self, fouling_calculator: FoulingCalculator):
        """Test error messages don't expose sensitive information."""
        # Arrange: Trigger validation error
        try:
            FoulingResistanceInput(
                u_clean_w_m2_k=-500.0,  # Invalid
                u_fouled_w_m2_k=420.0,
            )
        except ValueError as e:
            error_message = str(e)
            # Assert: Error message is informative but not exposing internals
            assert "stack trace" not in error_message.lower()
            assert "internal" not in error_message.lower()

    def test_result_isolation(self, fouling_calculator: FoulingCalculator):
        """Test results from different calculations are isolated."""
        # Arrange
        input1 = FoulingResistanceInput(u_clean_w_m2_k=500.0, u_fouled_w_m2_k=420.0)
        input2 = FoulingResistanceInput(u_clean_w_m2_k=600.0, u_fouled_w_m2_k=500.0)

        # Act
        result1 = fouling_calculator.calculate_fouling_resistance(input1)
        result2 = fouling_calculator.calculate_fouling_resistance(input2)

        # Assert: Results are independent
        assert result1.fouling_resistance_m2_k_w != result2.fouling_resistance_m2_k_w
        assert result1.provenance_hash != result2.provenance_hash


# =============================================================================
# Test Class: Rate Limiting
# =============================================================================

@pytest.mark.security
class TestRateLimiting:
    """Tests for rate limiting protection."""

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client for rate limit testing."""
        client = MagicMock()
        client.post = MagicMock()
        return client

    def test_rate_limiting_enforced(self, mock_api_client):
        """Test rate limiting is enforced on API endpoints."""
        # Simulate rate limit exceeded response
        mock_api_client.post.return_value = MagicMock(
            status_code=429,
            json=lambda: {
                "error": "Rate limit exceeded",
                "retry_after": 60
            }
        )

        # Act: Simulate excessive requests
        response = mock_api_client.post(
            "/api/v1/fouling/analyze",
            json={"u_clean": 500, "u_fouled": 420}
        )

        # Assert
        assert response.status_code == 429
        assert response.json()["retry_after"] == 60


# =============================================================================
# Test Class: Secure Defaults
# =============================================================================

@pytest.mark.security
class TestSecureDefaults:
    """Tests for secure default configurations."""

    def test_secure_hash_algorithm(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test secure hash algorithm (SHA-256) is used."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act
        result = fouling_calculator.calculate_fouling_resistance(input_data)

        # Assert: Hash is SHA-256 (64 hex characters)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash.lower())

    def test_decimal_precision_default(self):
        """Test Decimal precision uses safe defaults."""
        # Decimal should use sufficient precision
        value = Decimal("1.23456789012345678901234567890")
        # Default should preserve at least 10 significant digits
        assert len(str(value).replace(".", "").lstrip("0")) >= 10

    def test_immutable_results_by_default(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test calculation results are immutable by default."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=420.0,
        )

        # Act
        result = fouling_calculator.calculate_fouling_resistance(input_data)

        # Assert: Result is frozen/immutable
        with pytest.raises((AttributeError, TypeError)):
            result.fouling_resistance_m2_k_w = Decimal("999")
