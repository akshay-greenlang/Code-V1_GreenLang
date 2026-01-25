# -*- coding: utf-8 -*-
"""
Factor Broker Custom Exceptions
GL-VCCI Scope 3 Platform

Custom exception classes for Factor Broker service with detailed error codes
and helpful messages for debugging and user feedback.

Version: 1.0.0
"""

from typing import Optional, Dict, Any


class FactorBrokerError(Exception):
    """
    Base exception for all Factor Broker errors.

    All custom exceptions in the Factor Broker service inherit from this base class,
    allowing for easy exception handling at different granularity levels.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "FACTOR_BROKER_ERROR",
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize FactorBrokerError.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for logging/monitoring
            details: Additional context about the error
            original_exception: Original exception if this wraps another error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.

        Returns:
            Dictionary containing error information
        """
        result = {
            "error": self.error_code,
            "message": self.message,
        }

        if self.details:
            result["details"] = self.details

        if self.original_exception:
            result["original_error"] = str(self.original_exception)

        return result


class FactorNotFoundError(FactorBrokerError):
    """
    Exception raised when an emission factor cannot be found in any source.

    This error is raised after all cascading sources (ecoinvent, DESNZ, EPA, proxy)
    have been attempted and none could provide a factor for the requested product.
    """

    def __init__(
        self,
        product: str,
        region: str,
        gwp_standard: str,
        tried_sources: Optional[list] = None,
        suggestions: Optional[list] = None
    ):
        """
        Initialize FactorNotFoundError.

        Args:
            product: Product name that was searched
            region: Region code that was searched
            gwp_standard: GWP standard that was used
            tried_sources: List of sources that were attempted
            suggestions: List of similar product names for fuzzy matching
        """
        tried_sources = tried_sources or []
        suggestions = suggestions or []

        message = (
            f"Emission factor not found for product '{product}' "
            f"in region '{region}' (GWP: {gwp_standard})"
        )

        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"

        details = {
            "product": product,
            "region": region,
            "gwp_standard": gwp_standard,
            "tried_sources": tried_sources,
            "suggestions": suggestions
        }

        super().__init__(
            message=message,
            error_code="FACTOR_NOT_FOUND",
            details=details
        )


class LicenseViolationError(FactorBrokerError):
    """
    Exception raised when an operation would violate license terms.

    This is critical for ecoinvent compliance - prevents bulk redistribution,
    caching beyond 24 hours, or any other license violations.
    """

    def __init__(
        self,
        violation_type: str,
        license_source: str = "ecoinvent",
        details_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LicenseViolationError.

        Args:
            violation_type: Type of license violation
            license_source: Data source with license restrictions
            details_dict: Additional details about the violation
        """
        message = (
            f"License violation detected: {violation_type} "
            f"(source: {license_source})"
        )

        details = {
            "violation_type": violation_type,
            "license_source": license_source,
            "compliance_note": (
                "ecoinvent license prohibits bulk redistribution and "
                "caching beyond 24 hours"
            )
        }

        if details_dict:
            details.update(details_dict)

        super().__init__(
            message=message,
            error_code="LICENSE_VIOLATION",
            details=details
        )


class RateLimitExceededError(FactorBrokerError):
    """
    Exception raised when API rate limit is exceeded.

    Different sources have different rate limits:
    - ecoinvent: 1000 requests/minute
    - DESNZ: typically no limit
    - EPA: varies by endpoint
    """

    def __init__(
        self,
        source: str,
        limit: int,
        retry_after_seconds: Optional[int] = None
    ):
        """
        Initialize RateLimitExceededError.

        Args:
            source: Data source that rate limited the request
            limit: Rate limit threshold (requests per minute)
            retry_after_seconds: Seconds to wait before retrying
        """
        message = (
            f"Rate limit exceeded for {source} "
            f"(limit: {limit} requests/minute)"
        )

        if retry_after_seconds:
            message += f". Retry after {retry_after_seconds} seconds."

        details = {
            "source": source,
            "limit": limit,
            "retry_after_seconds": retry_after_seconds
        }

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class SourceUnavailableError(FactorBrokerError):
    """
    Exception raised when a data source is unavailable.

    This can be due to network issues, API downtime, authentication failures,
    or other connectivity problems.
    """

    def __init__(
        self,
        source: str,
        reason: str,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize SourceUnavailableError.

        Args:
            source: Data source that is unavailable
            reason: Human-readable reason for unavailability
            original_exception: Original exception from connection attempt
        """
        message = f"Data source '{source}' is unavailable: {reason}"

        details = {
            "source": source,
            "reason": reason
        }

        super().__init__(
            message=message,
            error_code="SOURCE_UNAVAILABLE",
            details=details,
            original_exception=original_exception
        )


class ValidationError(FactorBrokerError):
    """
    Exception raised when input validation fails.

    This covers invalid product names, region codes, GWP standards,
    units, or any other validation failures.
    """

    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        valid_values: Optional[list] = None
    ):
        """
        Initialize ValidationError.

        Args:
            field: Field name that failed validation
            value: Invalid value that was provided
            reason: Reason why validation failed
            valid_values: List of valid values if applicable
        """
        message = f"Validation failed for '{field}': {reason}"

        if valid_values:
            message += f". Valid values: {', '.join(map(str, valid_values))}"

        details = {
            "field": field,
            "value": value,
            "reason": reason,
            "valid_values": valid_values
        }

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class CacheError(FactorBrokerError):
    """
    Exception raised when cache operations fail.

    This can be due to Redis connection failures, serialization errors,
    or other cache-related issues.
    """

    def __init__(
        self,
        operation: str,
        reason: str,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize CacheError.

        Args:
            operation: Cache operation that failed (get, set, delete, etc.)
            reason: Reason for the failure
            original_exception: Original exception from cache operation
        """
        message = f"Cache operation '{operation}' failed: {reason}"

        details = {
            "operation": operation,
            "reason": reason
        }

        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details,
            original_exception=original_exception
        )


class DataQualityError(FactorBrokerError):
    """
    Exception raised when factor data quality is below acceptable threshold.

    This is used to warn users when proxy factors or low-quality data
    is being used, suggesting they seek better quality data.
    """

    def __init__(
        self,
        factor_id: str,
        quality_score: int,
        min_threshold: int = 50,
        recommendation: Optional[str] = None
    ):
        """
        Initialize DataQualityError.

        Args:
            factor_id: Factor identifier
            quality_score: Actual quality score (0-100)
            min_threshold: Minimum acceptable quality score
            recommendation: Recommendation for improving data quality
        """
        message = (
            f"Data quality for factor '{factor_id}' is below threshold "
            f"({quality_score}/100, minimum: {min_threshold}/100)"
        )

        if recommendation:
            message += f". Recommendation: {recommendation}"

        details = {
            "factor_id": factor_id,
            "quality_score": quality_score,
            "min_threshold": min_threshold,
            "recommendation": recommendation or (
                "Consider supplier engagement for Tier 1 data collection"
            )
        }

        super().__init__(
            message=message,
            error_code="DATA_QUALITY_WARNING",
            details=details
        )


class ProxyCalculationError(FactorBrokerError):
    """
    Exception raised when proxy factor calculation fails.

    This can happen when there is insufficient data to calculate
    a category average or when the category is not recognized.
    """

    def __init__(
        self,
        product: str,
        category: Optional[str],
        reason: str
    ):
        """
        Initialize ProxyCalculationError.

        Args:
            product: Product for which proxy calculation failed
            category: Category used for proxy calculation
            reason: Reason for the failure
        """
        message = (
            f"Proxy calculation failed for product '{product}' "
            f"(category: {category or 'unknown'}): {reason}"
        )

        details = {
            "product": product,
            "category": category,
            "reason": reason
        }

        super().__init__(
            message=message,
            error_code="PROXY_CALCULATION_ERROR",
            details=details
        )


class ConfigurationError(FactorBrokerError):
    """
    Exception raised when service configuration is invalid.

    This covers missing environment variables, invalid settings,
    or other configuration issues.
    """

    def __init__(
        self,
        config_key: str,
        reason: str,
        resolution: Optional[str] = None
    ):
        """
        Initialize ConfigurationError.

        Args:
            config_key: Configuration key that is invalid
            reason: Reason why configuration is invalid
            resolution: Suggested resolution
        """
        message = f"Configuration error for '{config_key}': {reason}"

        if resolution:
            message += f". Resolution: {resolution}"

        details = {
            "config_key": config_key,
            "reason": reason,
            "resolution": resolution
        }

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


# Mapping of HTTP status codes to exception classes
HTTP_EXCEPTION_MAP = {
    400: ValidationError,
    404: FactorNotFoundError,
    429: RateLimitExceededError,
    500: FactorBrokerError,
    503: SourceUnavailableError,
}


def get_exception_for_status_code(
    status_code: int,
    default_message: str = "Unknown error"
) -> FactorBrokerError:
    """
    Get appropriate exception class for HTTP status code.

    Args:
        status_code: HTTP status code
        default_message: Default message if no specific mapping exists

    Returns:
        Instance of appropriate exception class
    """
    exception_class = HTTP_EXCEPTION_MAP.get(status_code, FactorBrokerError)

    if exception_class == ValidationError:
        return exception_class(
            field="request",
            value=None,
            reason=default_message
        )
    elif exception_class == FactorNotFoundError:
        return exception_class(
            product="unknown",
            region="unknown",
            gwp_standard="AR6"
        )
    elif exception_class == RateLimitExceededError:
        return exception_class(
            source="unknown",
            limit=0
        )
    elif exception_class == SourceUnavailableError:
        return exception_class(
            source="unknown",
            reason=default_message
        )
    else:
        return exception_class(message=default_message)
