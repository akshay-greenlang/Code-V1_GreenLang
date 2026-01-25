# -*- coding: utf-8 -*-
"""
Comprehensive Tests for Factor Broker Exceptions
GL-VCCI Scope 3 Platform - Phase 6

Tests for all exception classes:
- FactorNotFoundException
- LicenseViolationException
- CacheException
- SourceUnavailableException
- ValidationError
- RateLimitExceededError

Test Count: 15 tests
Target Coverage: 95%

Version: 1.0.0
"""

import pytest
from typing import Dict, Any

import sys
sys.path.insert(0, '/c/Users/aksha/Code-V1_GreenLang/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform')

from services.factor_broker.exceptions import (
    FactorBrokerError,
    FactorNotFoundError,
    LicenseViolationError,
    RateLimitExceededError,
    SourceUnavailableError,
    ValidationError,
    CacheError,
    DataQualityError,
    ProxyCalculationError,
    ConfigurationError,
    get_exception_for_status_code,
    HTTP_EXCEPTION_MAP
)


class TestFactorBrokerError:
    """Test base FactorBrokerError class."""

    def test_base_error_initialization(self):
        """Test base error initializes correctly."""
        error = FactorBrokerError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )

        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details["key"] == "value"

    def test_base_error_to_dict(self):
        """Test base error converts to dict correctly."""
        error = FactorBrokerError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"]["key"] == "value"

    def test_base_error_with_original_exception(self):
        """Test base error wraps original exception."""
        original = ValueError("Original error")
        error = FactorBrokerError(
            message="Wrapped error",
            error_code="WRAPPED_ERROR",
            original_exception=original
        )

        error_dict = error.to_dict()
        assert "original_error" in error_dict
        assert "Original error" in error_dict["original_error"]


class TestFactorNotFoundError:
    """Test FactorNotFoundError exception."""

    def test_factor_not_found_basic(self):
        """Test basic FactorNotFoundError."""
        error = FactorNotFoundError(
            product="Steel",
            region="US",
            gwp_standard="AR6"
        )

        assert "Steel" in str(error)
        assert "US" in str(error)
        assert error.error_code == "FACTOR_NOT_FOUND"

    def test_factor_not_found_with_tried_sources(self):
        """Test FactorNotFoundError includes tried sources."""
        error = FactorNotFoundError(
            product="Steel",
            region="US",
            gwp_standard="AR6",
            tried_sources=["ecoinvent", "desnz_uk", "epa_us"]
        )

        assert error.details["tried_sources"] == ["ecoinvent", "desnz_uk", "epa_us"]

    def test_factor_not_found_with_suggestions(self):
        """Test FactorNotFoundError includes suggestions."""
        error = FactorNotFoundError(
            product="Steal",  # Typo
            region="US",
            gwp_standard="AR6",
            suggestions=["Steel", "Stainless Steel"]
        )

        assert "Did you mean" in str(error)
        assert error.details["suggestions"] == ["Steel", "Stainless Steel"]


class TestLicenseViolationError:
    """Test LicenseViolationError exception."""

    def test_license_violation_basic(self):
        """Test basic LicenseViolationError."""
        error = LicenseViolationError(
            violation_type="bulk_export",
            license_source="ecoinvent"
        )

        assert "bulk_export" in str(error)
        assert "ecoinvent" in str(error)
        assert error.error_code == "LICENSE_VIOLATION"

    def test_license_violation_with_details(self):
        """Test LicenseViolationError with additional details."""
        error = LicenseViolationError(
            violation_type="cache_ttl_exceeded",
            license_source="ecoinvent",
            details_dict={"ttl_hours": 48, "max_allowed_hours": 24}
        )

        assert error.details["ttl_hours"] == 48
        assert error.details["max_allowed_hours"] == 24
        assert "ecoinvent license prohibits" in error.details["compliance_note"]


class TestRateLimitExceededError:
    """Test RateLimitExceededError exception."""

    def test_rate_limit_basic(self):
        """Test basic RateLimitExceededError."""
        error = RateLimitExceededError(
            source="ecoinvent",
            limit=1000
        )

        assert "ecoinvent" in str(error)
        assert "1000" in str(error)
        assert error.error_code == "RATE_LIMIT_EXCEEDED"

    def test_rate_limit_with_retry_after(self):
        """Test RateLimitExceededError with retry_after."""
        error = RateLimitExceededError(
            source="epa_us",
            limit=500,
            retry_after_seconds=60
        )

        assert "Retry after 60 seconds" in str(error)
        assert error.details["retry_after_seconds"] == 60


class TestSourceUnavailableError:
    """Test SourceUnavailableError exception."""

    def test_source_unavailable_basic(self):
        """Test basic SourceUnavailableError."""
        error = SourceUnavailableError(
            source="ecoinvent",
            reason="API timeout"
        )

        assert "ecoinvent" in str(error)
        assert "API timeout" in str(error)
        assert error.error_code == "SOURCE_UNAVAILABLE"

    def test_source_unavailable_with_original_exception(self):
        """Test SourceUnavailableError wraps original exception."""
        original = ConnectionError("Network unreachable")
        error = SourceUnavailableError(
            source="epa_us",
            reason="Connection failed",
            original_exception=original
        )

        assert error.original_exception == original


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_basic(self):
        """Test basic ValidationError."""
        error = ValidationError(
            field="region",
            value="usa",
            reason="Region code must be 2 characters"
        )

        assert "region" in str(error)
        assert error.error_code == "VALIDATION_ERROR"

    def test_validation_error_with_valid_values(self):
        """Test ValidationError with valid values list."""
        error = ValidationError(
            field="gwp_standard",
            value="AR4",
            reason="Invalid GWP standard",
            valid_values=["AR5", "AR6"]
        )

        assert "Valid values: AR5, AR6" in str(error)
        assert error.details["valid_values"] == ["AR5", "AR6"]


class TestCacheError:
    """Test CacheError exception."""

    def test_cache_error_basic(self):
        """Test basic CacheError."""
        error = CacheError(
            operation="get",
            reason="Redis connection failed"
        )

        assert "get" in str(error)
        assert "Redis connection failed" in str(error)
        assert error.error_code == "CACHE_ERROR"

    def test_cache_error_with_original_exception(self):
        """Test CacheError wraps original exception."""
        original = ConnectionRefusedError("Redis server not responding")
        error = CacheError(
            operation="set",
            reason="Connection refused",
            original_exception=original
        )

        assert error.original_exception == original


class TestDataQualityError:
    """Test DataQualityError exception."""

    def test_data_quality_error_basic(self):
        """Test basic DataQualityError."""
        error = DataQualityError(
            factor_id="proxy_factor_001",
            quality_score=40,
            min_threshold=50
        )

        assert "proxy_factor_001" in str(error)
        assert "40/100" in str(error)
        assert error.error_code == "DATA_QUALITY_WARNING"

    def test_data_quality_error_with_recommendation(self):
        """Test DataQualityError with recommendation."""
        error = DataQualityError(
            factor_id="low_quality_factor",
            quality_score=35,
            min_threshold=50,
            recommendation="Request primary data from supplier"
        )

        assert "Request primary data from supplier" in str(error)


class TestHTTPExceptionMapping:
    """Test HTTP status code to exception mapping."""

    def test_http_404_maps_to_factor_not_found(self):
        """Test HTTP 404 maps to FactorNotFoundError."""
        exception = get_exception_for_status_code(404, "Factor not found")

        assert isinstance(exception, FactorNotFoundError)

    def test_http_429_maps_to_rate_limit(self):
        """Test HTTP 429 maps to RateLimitExceededError."""
        exception = get_exception_for_status_code(429, "Rate limit exceeded")

        assert isinstance(exception, RateLimitExceededError)

    def test_http_503_maps_to_source_unavailable(self):
        """Test HTTP 503 maps to SourceUnavailableError."""
        exception = get_exception_for_status_code(503, "Service unavailable")

        assert isinstance(exception, SourceUnavailableError)

    def test_http_400_maps_to_validation_error(self):
        """Test HTTP 400 maps to ValidationError."""
        exception = get_exception_for_status_code(400, "Bad request")

        assert isinstance(exception, ValidationError)

    def test_http_500_maps_to_base_error(self):
        """Test HTTP 500 maps to base FactorBrokerError."""
        exception = get_exception_for_status_code(500, "Internal server error")

        assert isinstance(exception, FactorBrokerError)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
