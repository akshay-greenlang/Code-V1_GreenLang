"""
Unit tests for SDK exceptions.
"""

import pytest

from gl_normalizer.exceptions import (
    NormalizerError,
    ConfigurationError,
    ValidationError,
    ConversionError,
    ResolutionError,
    VocabularyError,
    AuditError,
    APIError,
    RateLimitError,
    TimeoutError,
    ServiceUnavailableError,
    ConnectionError,
    JobError,
    raise_for_error_response,
)


class TestNormalizerError:
    """Test base NormalizerError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = NormalizerError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code is None
        assert error.details == {}

    def test_error_with_code(self) -> None:
        """Test error with code."""
        error = NormalizerError("Something went wrong", code="GLNORM-E100")
        assert str(error) == "[GLNORM-E100] Something went wrong"
        assert error.code == "GLNORM-E100"

    def test_error_with_details(self) -> None:
        """Test error with details."""
        error = NormalizerError(
            "Something went wrong",
            code="GLNORM-E100",
            details={"field": "unit", "value": "invalid"},
        )
        assert error.details == {"field": "unit", "value": "invalid"}

    def test_error_repr(self) -> None:
        """Test error repr."""
        error = NormalizerError("Test", code="CODE", details={"key": "value"})
        repr_str = repr(error)
        assert "NormalizerError" in repr_str
        assert "Test" in repr_str
        assert "CODE" in repr_str


class TestConfigurationError:
    """Test ConfigurationError."""

    def test_configuration_error(self) -> None:
        """Test configuration error."""
        error = ConfigurationError("API key is required")
        assert isinstance(error, NormalizerError)
        assert error.message == "API key is required"


class TestValidationError:
    """Test ValidationError."""

    def test_basic_validation_error(self) -> None:
        """Test basic validation error."""
        error = ValidationError(
            code="GLNORM-E100",
            message="Unit parse failed",
        )
        assert error.code == "GLNORM-E100"
        assert error.message == "Unit parse failed"

    def test_validation_error_with_details(self) -> None:
        """Test validation error with all details."""
        error = ValidationError(
            code="GLNORM-E200",
            message="Dimension mismatch",
            path="/measurements/0",
            expected={"dimension": "energy"},
            actual={"dimension": "mass"},
            hint="Use energy units like kWh",
        )
        assert error.path == "/measurements/0"
        assert error.expected == {"dimension": "energy"}
        assert error.actual == {"dimension": "mass"}
        assert error.hint == "Use energy units like kWh"


class TestConversionError:
    """Test ConversionError."""

    def test_conversion_error(self) -> None:
        """Test conversion error."""
        error = ConversionError(
            code="GLNORM-E301",
            message="Missing reference conditions",
            from_unit="Nm3",
            to_unit="m3",
            hint="Provide temperature and pressure",
        )
        assert error.code == "GLNORM-E301"
        assert error.from_unit == "Nm3"
        assert error.to_unit == "m3"
        assert error.hint == "Provide temperature and pressure"


class TestResolutionError:
    """Test ResolutionError."""

    def test_resolution_error_not_found(self) -> None:
        """Test resolution error when not found."""
        error = ResolutionError(
            code="GLNORM-E400",
            message="Reference not found",
            entity_type="fuel",
            raw_name="Unknown Fuel",
        )
        assert error.entity_type == "fuel"
        assert error.raw_name == "Unknown Fuel"
        assert error.candidates == []

    def test_resolution_error_ambiguous(self) -> None:
        """Test resolution error when ambiguous."""
        error = ResolutionError(
            code="GLNORM-E401",
            message="Reference ambiguous",
            entity_type="fuel",
            raw_name="Gas",
            candidates=[
                {"reference_id": "GL-FUEL-NATGAS", "score": 0.8},
                {"reference_id": "GL-FUEL-LPG", "score": 0.75},
            ],
        )
        assert len(error.candidates) == 2

    def test_resolution_error_low_confidence(self) -> None:
        """Test resolution error with low confidence."""
        error = ResolutionError(
            code="GLNORM-E403",
            message="Low confidence match",
            entity_type="fuel",
            raw_name="Some Fuel",
            confidence=0.65,
        )
        assert error.confidence == 0.65


class TestVocabularyError:
    """Test VocabularyError."""

    def test_vocabulary_error(self) -> None:
        """Test vocabulary error."""
        error = VocabularyError(
            code="GLNORM-E500",
            message="Vocabulary version mismatch",
            vocabulary_version="2025.01.0",
        )
        assert error.vocabulary_version == "2025.01.0"


class TestAuditError:
    """Test AuditError."""

    def test_audit_error(self) -> None:
        """Test audit error."""
        error = AuditError("Audit write failed", code="GLNORM-E600")
        assert isinstance(error, NormalizerError)
        assert error.code == "GLNORM-E600"


class TestAPIError:
    """Test APIError."""

    def test_api_error(self) -> None:
        """Test API error."""
        error = APIError("Server error", status_code=500)
        assert error.status_code == 500
        assert error.response_body is None

    def test_api_error_with_response(self) -> None:
        """Test API error with response body."""
        error = APIError(
            "Bad request",
            status_code=400,
            response_body='{"error": "invalid input"}',
        )
        assert error.status_code == 400
        assert error.response_body == '{"error": "invalid input"}'


class TestRateLimitError:
    """Test RateLimitError."""

    def test_rate_limit_error(self) -> None:
        """Test rate limit error."""
        error = RateLimitError("Rate limit exceeded", retry_after=60.0)
        assert error.retry_after == 60.0
        assert error.status_code == 429
        assert error.code == "GLNORM-E900"

    def test_rate_limit_error_default_retry(self) -> None:
        """Test rate limit error default retry after."""
        error = RateLimitError("Rate limit exceeded")
        assert error.retry_after == 60.0


class TestTimeoutError:
    """Test TimeoutError."""

    def test_timeout_error(self) -> None:
        """Test timeout error."""
        error = TimeoutError("Request timed out", timeout=30.0)
        assert error.timeout == 30.0
        assert error.code == "GLNORM-E901"


class TestServiceUnavailableError:
    """Test ServiceUnavailableError."""

    def test_service_unavailable_error(self) -> None:
        """Test service unavailable error."""
        error = ServiceUnavailableError("Service temporarily unavailable")
        assert error.status_code == 503
        assert error.code == "GLNORM-E903"


class TestConnectionError:
    """Test ConnectionError."""

    def test_connection_error(self) -> None:
        """Test connection error."""
        error = ConnectionError("Failed to connect")
        assert isinstance(error, NormalizerError)


class TestJobError:
    """Test JobError."""

    def test_job_error(self) -> None:
        """Test job error."""
        error = JobError(
            "Job failed",
            job_id="job-abc123",
            job_status="failed",
        )
        assert error.job_id == "job-abc123"
        assert error.job_status == "failed"


class TestRaiseForErrorResponse:
    """Test raise_for_error_response function."""

    def test_validation_error_e1xx(self) -> None:
        """Test raises ValidationError for E1xx codes."""
        response = {
            "errors": [
                {
                    "code": "GLNORM-E100",
                    "message": "Unit parse failed",
                    "path": "/measurements/0",
                }
            ]
        }
        with pytest.raises(ValidationError) as exc_info:
            raise_for_error_response(response, 400)
        assert exc_info.value.code == "GLNORM-E100"

    def test_validation_error_e2xx(self) -> None:
        """Test raises ValidationError for E2xx codes."""
        response = {
            "errors": [
                {
                    "code": "GLNORM-E200",
                    "message": "Dimension mismatch",
                    "expected": {"dimension": "energy"},
                    "actual": {"dimension": "mass"},
                    "hint": {"suggestion": "Use energy units"},
                }
            ]
        }
        with pytest.raises(ValidationError) as exc_info:
            raise_for_error_response(response, 400)
        assert exc_info.value.code == "GLNORM-E200"
        assert exc_info.value.hint == "Use energy units"

    def test_conversion_error_e3xx(self) -> None:
        """Test raises ConversionError for E3xx codes."""
        response = {
            "errors": [
                {
                    "code": "GLNORM-E301",
                    "message": "Missing reference conditions",
                    "actual": {"unit": "Nm3"},
                    "expected": {"unit": "m3"},
                }
            ]
        }
        with pytest.raises(ConversionError) as exc_info:
            raise_for_error_response(response, 422)
        assert exc_info.value.code == "GLNORM-E301"

    def test_resolution_error_e4xx(self) -> None:
        """Test raises ResolutionError for E4xx codes."""
        response = {
            "errors": [
                {
                    "code": "GLNORM-E400",
                    "message": "Reference not found",
                    "actual": {"entity_type": "fuel", "raw_name": "Unknown"},
                }
            ],
            "candidates": [],
        }
        with pytest.raises(ResolutionError) as exc_info:
            raise_for_error_response(response, 404)
        assert exc_info.value.code == "GLNORM-E400"

    def test_vocabulary_error_e5xx(self) -> None:
        """Test raises VocabularyError for E5xx codes."""
        response = {
            "errors": [
                {
                    "code": "GLNORM-E500",
                    "message": "Vocabulary version mismatch",
                }
            ],
            "vocabulary_version": "2025.01.0",
        }
        with pytest.raises(VocabularyError) as exc_info:
            raise_for_error_response(response, 422)
        assert exc_info.value.code == "GLNORM-E500"

    def test_audit_error_e6xx(self) -> None:
        """Test raises AuditError for E6xx codes."""
        response = {
            "errors": [
                {
                    "code": "GLNORM-E600",
                    "message": "Audit write failed",
                }
            ]
        }
        with pytest.raises(AuditError) as exc_info:
            raise_for_error_response(response, 500)
        assert exc_info.value.code == "GLNORM-E600"

    def test_rate_limit_error_429(self) -> None:
        """Test raises RateLimitError for 429 status."""
        response = {
            "errors": [{"code": "GLNORM-E900", "message": "Rate limit exceeded"}],
            "retry_after": 30.0,
        }
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_error_response(response, 429)
        assert exc_info.value.retry_after == 30.0

    def test_service_unavailable_error_503(self) -> None:
        """Test raises ServiceUnavailableError for 503 status."""
        response = {
            "errors": [{"code": "GLNORM-E903", "message": "Service unavailable"}]
        }
        with pytest.raises(ServiceUnavailableError):
            raise_for_error_response(response, 503)

    def test_timeout_error_e901(self) -> None:
        """Test raises TimeoutError for E901 code."""
        response = {
            "errors": [{"code": "GLNORM-E901", "message": "Operation timed out"}]
        }
        with pytest.raises(TimeoutError) as exc_info:
            raise_for_error_response(response, 504)
        assert exc_info.value.code == "GLNORM-E901"

    def test_generic_api_error(self) -> None:
        """Test raises APIError for unknown codes."""
        response = {
            "errors": [{"code": "UNKNOWN", "message": "Unknown error"}]
        }
        with pytest.raises(APIError) as exc_info:
            raise_for_error_response(response, 500)
        assert exc_info.value.status_code == 500
