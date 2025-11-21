# -*- coding: utf-8 -*-
"""
SAP S/4HANA Connector Custom Exceptions
GL-VCCI Scope 3 Platform

Custom exception classes for SAP connector with detailed error codes
and helpful messages for debugging and user feedback.

Version: 1.0.0
Phase: 4 (Weeks 19-22)
Date: 2025-11-06
"""

from typing import Optional, Dict, Any


class SAPConnectorError(Exception):
    """
    Base exception for all SAP connector errors.

    All custom exceptions in the SAP connector inherit from this base class,
    allowing for easy exception handling at different granularity levels.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "SAP_CONNECTOR_ERROR",
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize SAPConnectorError.

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


class SAPConnectionError(SAPConnectorError):
    """
    Exception raised when connection to SAP S/4HANA fails.

    This can be due to network issues, incorrect configuration,
    or SAP system unavailability.
    """

    def __init__(
        self,
        endpoint: str,
        reason: str,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize SAPConnectionError.

        Args:
            endpoint: SAP endpoint that failed to connect
            reason: Human-readable reason for connection failure
            original_exception: Original exception from connection attempt
        """
        message = f"Failed to connect to SAP endpoint '{endpoint}': {reason}"

        details = {
            "endpoint": endpoint,
            "reason": reason
        }

        super().__init__(
            message=message,
            error_code="SAP_CONNECTION_ERROR",
            details=details,
            original_exception=original_exception
        )


class SAPAuthenticationError(SAPConnectorError):
    """
    Exception raised when authentication to SAP S/4HANA fails.

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
        Initialize SAPAuthenticationError.

        Args:
            auth_type: Type of authentication that failed
            reason: Human-readable reason for authentication failure
            original_exception: Original exception from auth attempt
        """
        message = f"SAP authentication failed ({auth_type}): {reason}"

        details = {
            "auth_type": auth_type,
            "reason": reason,
            "resolution": "Check client credentials and token URL configuration"
        }

        super().__init__(
            message=message,
            error_code="SAP_AUTHENTICATION_ERROR",
            details=details,
            original_exception=original_exception
        )


class SAPRateLimitError(SAPConnectorError):
    """
    Exception raised when SAP API rate limit is exceeded.

    Different SAP modules and endpoints may have different rate limits.
    This exception includes retry-after information when available.
    """

    def __init__(
        self,
        endpoint: str,
        limit: int = 10,
        retry_after_seconds: Optional[int] = None
    ):
        """
        Initialize SAPRateLimitError.

        Args:
            endpoint: SAP endpoint that rate limited the request
            limit: Rate limit threshold (requests per minute)
            retry_after_seconds: Seconds to wait before retrying
        """
        message = (
            f"Rate limit exceeded for SAP endpoint '{endpoint}' "
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
            error_code="SAP_RATE_LIMIT_ERROR",
            details=details
        )


class SAPDataError(SAPConnectorError):
    """
    Exception raised when SAP data is invalid or unexpected.

    This includes OData parsing errors, schema validation failures,
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
        Initialize SAPDataError.

        Args:
            data_type: Type of data that failed (e.g., "PurchaseOrder", "Vendor")
            reason: Reason for the data error
            entity_id: ID of the entity if applicable
            original_exception: Original exception from data parsing
        """
        message = f"SAP data error for {data_type}: {reason}"

        if entity_id:
            message += f" (entity ID: {entity_id})"

        details = {
            "data_type": data_type,
            "reason": reason,
            "entity_id": entity_id
        }

        super().__init__(
            message=message,
            error_code="SAP_DATA_ERROR",
            details=details,
            original_exception=original_exception
        )


class SAPTimeoutError(SAPConnectorError):
    """
    Exception raised when SAP request times out.

    This can happen with large data queries or slow SAP system performance.
    """

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: int,
        operation: str = "request"
    ):
        """
        Initialize SAPTimeoutError.

        Args:
            endpoint: SAP endpoint that timed out
            timeout_seconds: Timeout threshold that was exceeded
            operation: Operation that timed out
        """
        message = (
            f"SAP {operation} to '{endpoint}' timed out "
            f"after {timeout_seconds} seconds"
        )

        details = {
            "endpoint": endpoint,
            "timeout_seconds": timeout_seconds,
            "operation": operation,
            "resolution": "Consider increasing timeout or paginating the query"
        }

        super().__init__(
            message=message,
            error_code="SAP_TIMEOUT_ERROR",
            details=details
        )


class SAPConfigurationError(SAPConnectorError):
    """
    Exception raised when SAP configuration is invalid.

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
        Initialize SAPConfigurationError.

        Args:
            config_key: Configuration key that is invalid
            reason: Reason why configuration is invalid
            resolution: Suggested resolution
        """
        message = f"SAP configuration error for '{config_key}': {reason}"

        if resolution:
            message += f". Resolution: {resolution}"

        details = {
            "config_key": config_key,
            "reason": reason,
            "resolution": resolution
        }

        super().__init__(
            message=message,
            error_code="SAP_CONFIGURATION_ERROR",
            details=details
        )


# Mapping of HTTP status codes to exception classes
HTTP_EXCEPTION_MAP = {
    400: SAPDataError,
    401: SAPAuthenticationError,
    403: SAPAuthenticationError,
    404: SAPDataError,
    429: SAPRateLimitError,
    500: SAPConnectionError,
    502: SAPConnectionError,
    503: SAPConnectionError,
    504: SAPTimeoutError,
}


def get_exception_for_status_code(
    status_code: int,
    endpoint: str = "unknown",
    default_message: str = "Unknown error"
) -> SAPConnectorError:
    """
    Get appropriate exception class for HTTP status code.

    Args:
        status_code: HTTP status code
        endpoint: SAP endpoint that returned the error
        default_message: Default message if no specific mapping exists

    Returns:
        Instance of appropriate exception class
    """
    exception_class = HTTP_EXCEPTION_MAP.get(status_code, SAPConnectorError)

    if exception_class == SAPDataError:
        return exception_class(
            data_type="response",
            reason=f"HTTP {status_code}: {default_message}"
        )
    elif exception_class == SAPAuthenticationError:
        return exception_class(
            reason=f"HTTP {status_code}: {default_message}"
        )
    elif exception_class == SAPRateLimitError:
        return exception_class(
            endpoint=endpoint
        )
    elif exception_class == SAPTimeoutError:
        return exception_class(
            endpoint=endpoint,
            timeout_seconds=30
        )
    elif exception_class == SAPConnectionError:
        return exception_class(
            endpoint=endpoint,
            reason=f"HTTP {status_code}: {default_message}"
        )
    else:
        return exception_class(message=default_message)
