"""
Workday RaaS Connector Custom Exceptions
GL-VCCI Scope 3 Platform

Custom exception classes for Workday connector with detailed error codes
and helpful messages for debugging and user feedback.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

from typing import Optional, Dict, Any


class WorkdayConnectorError(Exception):
    """
    Base exception for all Workday connector errors.

    All custom exceptions in the Workday connector inherit from this base class,
    allowing for easy exception handling at different granularity levels.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "WORKDAY_CONNECTOR_ERROR",
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize WorkdayConnectorError.

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


class WorkdayConnectionError(WorkdayConnectorError):
    """
    Exception raised when connection to Workday fails.

    This can be due to network issues, incorrect configuration,
    or Workday system unavailability.
    """

    def __init__(
        self,
        endpoint: str,
        reason: str,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize WorkdayConnectionError.

        Args:
            endpoint: Workday endpoint that failed to connect
            reason: Human-readable reason for connection failure
            original_exception: Original exception from connection attempt
        """
        message = f"Failed to connect to Workday endpoint '{endpoint}': {reason}"

        details = {
            "endpoint": endpoint,
            "reason": reason
        }

        super().__init__(
            message=message,
            error_code="WORKDAY_CONNECTION_ERROR",
            details=details,
            original_exception=original_exception
        )


class WorkdayAuthenticationError(WorkdayConnectorError):
    """
    Exception raised when authentication to Workday fails.

    This includes OAuth 2.0 token acquisition failures, expired tokens,
    invalid credentials, or insufficient permissions.
    """

    def __init__(
        self,
        auth_type: str = "OAuth 2.0",
        reason: str = "Authentication failed",
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize WorkdayAuthenticationError.

        Args:
            auth_type: Type of authentication that failed
            reason: Human-readable reason for authentication failure
            original_exception: Original exception from auth attempt
        """
        message = f"Workday authentication failed ({auth_type}): {reason}"

        details = {
            "auth_type": auth_type,
            "reason": reason,
            "resolution": "Check client credentials and token URL configuration"
        }

        super().__init__(
            message=message,
            error_code="WORKDAY_AUTHENTICATION_ERROR",
            details=details,
            original_exception=original_exception
        )


class WorkdayRateLimitError(WorkdayConnectorError):
    """
    Exception raised when Workday API rate limit is exceeded.

    Workday has rate limits on RaaS API calls to prevent system overload.
    """

    def __init__(
        self,
        endpoint: str,
        limit: int = 10,
        retry_after_seconds: Optional[int] = None
    ):
        """
        Initialize WorkdayRateLimitError.

        Args:
            endpoint: Workday endpoint that rate limited the request
            limit: Rate limit threshold (requests per minute)
            retry_after_seconds: Seconds to wait before retrying
        """
        message = (
            f"Rate limit exceeded for Workday endpoint '{endpoint}' "
            f"(limit: {limit} requests/minute)"
        )

        if retry_after_seconds:
            message += f". Retry after {retry_after_seconds} seconds."

        details = {
            "endpoint": endpoint,
            "limit": limit,
            "retry_after_seconds": retry_after_seconds
        }

        super().__init__(
            message=message,
            error_code="WORKDAY_RATE_LIMIT_ERROR",
            details=details
        )


class WorkdayDataError(WorkdayConnectorError):
    """
    Exception raised when Workday data is invalid or unexpected.

    This includes RaaS response parsing errors, schema validation failures,
    missing required fields, or malformed responses.
    """

    def __init__(
        self,
        data_type: str,
        reason: str,
        entity_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize WorkdayDataError.

        Args:
            data_type: Type of data that failed (e.g., "ExpenseReport", "CommuteData")
            reason: Reason for the data error
            entity_id: ID of the entity if applicable
            original_exception: Original exception from data parsing
        """
        message = f"Workday data error for {data_type}: {reason}"

        if entity_id:
            message += f" (entity ID: {entity_id})"

        details = {
            "data_type": data_type,
            "reason": reason,
            "entity_id": entity_id
        }

        super().__init__(
            message=message,
            error_code="WORKDAY_DATA_ERROR",
            details=details,
            original_exception=original_exception
        )


class WorkdayTimeoutError(WorkdayConnectorError):
    """
    Exception raised when Workday request times out.

    This can happen with large reports or slow Workday system performance.
    """

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: int,
        operation: str = "request"
    ):
        """
        Initialize WorkdayTimeoutError.

        Args:
            endpoint: Workday endpoint that timed out
            timeout_seconds: Timeout threshold that was exceeded
            operation: Operation that timed out
        """
        message = (
            f"Workday {operation} to '{endpoint}' timed out "
            f"after {timeout_seconds} seconds"
        )

        details = {
            "endpoint": endpoint,
            "timeout_seconds": timeout_seconds,
            "operation": operation,
            "resolution": "Consider increasing timeout or reducing report size"
        }

        super().__init__(
            message=message,
            error_code="WORKDAY_TIMEOUT_ERROR",
            details=details
        )


class WorkdayConfigurationError(WorkdayConnectorError):
    """
    Exception raised when Workday configuration is invalid.

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
        Initialize WorkdayConfigurationError.

        Args:
            config_key: Configuration key that is invalid
            reason: Reason why configuration is invalid
            resolution: Suggested resolution
        """
        message = f"Workday configuration error for '{config_key}': {reason}"

        if resolution:
            message += f". Resolution: {resolution}"

        details = {
            "config_key": config_key,
            "reason": reason,
            "resolution": resolution
        }

        super().__init__(
            message=message,
            error_code="WORKDAY_CONFIGURATION_ERROR",
            details=details
        )


# Mapping of HTTP status codes to exception classes
HTTP_EXCEPTION_MAP = {
    400: WorkdayDataError,
    401: WorkdayAuthenticationError,
    403: WorkdayAuthenticationError,
    404: WorkdayDataError,
    429: WorkdayRateLimitError,
    500: WorkdayConnectionError,
    502: WorkdayConnectionError,
    503: WorkdayConnectionError,
    504: WorkdayTimeoutError,
}


def get_exception_for_status_code(
    status_code: int,
    endpoint: str = "unknown",
    default_message: str = "Unknown error"
) -> WorkdayConnectorError:
    """
    Get appropriate exception class for HTTP status code.

    Args:
        status_code: HTTP status code
        endpoint: Workday endpoint that returned the error
        default_message: Default message if no specific mapping exists

    Returns:
        Instance of appropriate exception class
    """
    exception_class = HTTP_EXCEPTION_MAP.get(status_code, WorkdayConnectorError)

    if exception_class == WorkdayDataError:
        return exception_class(
            data_type="response",
            reason=f"HTTP {status_code}: {default_message}"
        )
    elif exception_class == WorkdayAuthenticationError:
        return exception_class(
            reason=f"HTTP {status_code}: {default_message}"
        )
    elif exception_class == WorkdayRateLimitError:
        return exception_class(
            endpoint=endpoint
        )
    elif exception_class == WorkdayTimeoutError:
        return exception_class(
            endpoint=endpoint,
            timeout_seconds=60
        )
    elif exception_class == WorkdayConnectionError:
        return exception_class(
            endpoint=endpoint,
            reason=f"HTTP {status_code}: {default_message}"
        )
    else:
        return exception_class(message=default_message)
