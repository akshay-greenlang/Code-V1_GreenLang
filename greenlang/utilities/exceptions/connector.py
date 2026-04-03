"""
GreenLang Connector Exceptions - External Connector Errors

This module provides exception classes for connector-related errors including
network, authentication, rate limiting, and configuration failures.

Features:
- Connector configuration errors
- Authentication and authorization failures
- Network and timeout errors
- Rate limiting with retry-after tracking
- Request validation errors
- Security policy violations

Author: GreenLang Team
Date: 2026-04-02
"""

from typing import Any, Dict, List, Optional

from greenlang.exceptions.base import GreenLangException


class ConnectorException(GreenLangException):
    """Base exception for connector-related errors.

    Raised when an external connector encounters an error during operation.
    Carries connector-specific metadata such as connector name, HTTP status
    code, request URL, and request ID.

    Example:
        >>> raise ConnectorException(
        ...     message="Connector operation failed",
        ...     connector_name="grid/electricitymaps",
        ...     status_code=500,
        ...     url="https://api.electricitymaps.com/v3/zones"
        ... )
    """
    ERROR_PREFIX = "GL_CONNECTOR"

    def __init__(
        self,
        message: str,
        connector_name: Optional[str] = None,
        status_code: Optional[int] = None,
        url: Optional[str] = None,
        request_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize connector exception.

        Args:
            message: Error message
            connector_name: Identifier for the connector (e.g., "grid/electricitymaps")
            status_code: HTTP status code if applicable
            url: Target URL of the failed request
            request_id: Request identifier for debugging
            original_error: Wrapped original exception
            agent_name: Name of agent using the connector
            context: Additional error context
        """
        context = context or {}
        if connector_name:
            context["connector_name"] = connector_name
        if status_code is not None:
            context["status_code"] = status_code
        if url:
            context["url"] = url
        if request_id:
            context["request_id"] = request_id
        if original_error:
            context["original_error"] = str(original_error)
            context["original_error_type"] = type(original_error).__name__
        super().__init__(message, agent_name=agent_name, context=context)
        self.connector_name = connector_name
        self.status_code = status_code
        self.url = url
        self.request_id = request_id
        self.original_error = original_error


class ConnectorConfigError(ConnectorException):
    """Connector configuration is invalid or incomplete.

    Raised when connector configuration is missing required fields such
    as API keys, endpoints, or authentication credentials.

    Example:
        >>> raise ConnectorConfigError(
        ...     message="Missing API key",
        ...     connector_name="grid/electricitymaps",
        ...     context={"required_env": "ELECTRICITYMAPS_API_KEY"}
        ... )
    """
    pass


class ConnectorAuthError(ConnectorException):
    """Connector authentication or authorization failed.

    Raised when authentication fails, credentials are invalid, or the
    connector lacks sufficient permissions.

    Example:
        >>> raise ConnectorAuthError(
        ...     message="API key expired",
        ...     connector_name="grid/electricitymaps",
        ...     status_code=401
        ... )
    """
    pass


class ConnectorNetworkError(ConnectorException):
    """Network communication error.

    Raised when the network request fails due to DNS resolution,
    connection refused, TLS errors, or unreachable hosts.

    Example:
        >>> raise ConnectorNetworkError(
        ...     message="DNS resolution failed for api.example.com",
        ...     connector_name="erp/sap",
        ...     url="https://api.example.com/v1/data"
        ... )
    """
    pass


class ConnectorTimeoutError(ConnectorException):
    """Connector request timed out.

    Raised when a request exceeds the configured timeout threshold.

    Example:
        >>> raise ConnectorTimeoutError(
        ...     message="Request timed out after 30s",
        ...     connector_name="erp/sap",
        ...     timeout_seconds=30.0
        ... )
    """

    def __init__(
        self,
        message: str,
        connector_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize timeout error.

        Args:
            message: Error message
            connector_name: Connector identifier
            timeout_seconds: Configured timeout limit
            **kwargs: Additional arguments passed to ConnectorException
        """
        context = kwargs.pop("context", None) or {}
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        super().__init__(message, connector_name=connector_name, context=context, **kwargs)


class ConnectorRateLimitError(ConnectorException):
    """API rate limit exceeded.

    Raised when the connector hits the rate limit of an external API.
    Includes retry-after information when available.

    Example:
        >>> raise ConnectorRateLimitError(
        ...     message="Rate limit exceeded",
        ...     connector_name="grid/electricitymaps",
        ...     retry_after_seconds=60,
        ...     limit=100
        ... )
    """

    def __init__(
        self,
        message: str,
        connector_name: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            connector_name: Connector identifier
            retry_after_seconds: Seconds until retry is allowed
            limit: Rate limit value
            **kwargs: Additional arguments passed to ConnectorException
        """
        context = kwargs.pop("context", None) or {}
        if retry_after_seconds is not None:
            context["retry_after_seconds"] = retry_after_seconds
        if limit is not None:
            context["limit"] = limit
        super().__init__(message, connector_name=connector_name, context=context, **kwargs)


class ConnectorNotFoundError(ConnectorException):
    """Requested resource not found.

    Raised when the requested resource does not exist in the external service.

    Example:
        >>> raise ConnectorNotFoundError(
        ...     message="Region code 'XX' not found",
        ...     connector_name="grid/electricitymaps",
        ...     status_code=404
        ... )
    """
    pass


class ConnectorValidationError(ConnectorException):
    """Request or response data failed validation.

    Raised when payload data fails schema validation, Pydantic validation,
    or other data integrity checks.

    Example:
        >>> raise ConnectorValidationError(
        ...     message="Response schema mismatch",
        ...     connector_name="erp/sap",
        ...     validation_errors=[{"field": "amount", "error": "expected float"}]
        ... )
    """

    def __init__(
        self,
        message: str,
        connector_name: Optional[str] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            connector_name: Connector identifier
            validation_errors: List of validation error details
            **kwargs: Additional arguments passed to ConnectorException
        """
        context = kwargs.pop("context", None) or {}
        if validation_errors:
            context["validation_errors"] = validation_errors
        super().__init__(message, connector_name=connector_name, context=context, **kwargs)


class ConnectorSecurityError(ConnectorException):
    """Connector operation violates security policy.

    Raised when a connector operation is blocked by egress controls,
    domain allowlists, TLS requirements, or other security policies.

    Example:
        >>> raise ConnectorSecurityError(
        ...     message="Egress blocked: domain not in allowlist",
        ...     connector_name="erp/sap",
        ...     policy_violated="egress_allowlist"
        ... )
    """

    def __init__(
        self,
        message: str,
        connector_name: Optional[str] = None,
        policy_violated: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize security error.

        Args:
            message: Error message
            connector_name: Connector identifier
            policy_violated: Name of the violated security policy
            **kwargs: Additional arguments passed to ConnectorException
        """
        context = kwargs.pop("context", None) or {}
        if policy_violated:
            context["policy_violated"] = policy_violated
        super().__init__(message, connector_name=connector_name, context=context, **kwargs)


class ConnectorServerError(ConnectorException):
    """Upstream server returned a 5xx error.

    Raised when the external service has an internal error. Usually retriable.

    Example:
        >>> raise ConnectorServerError(
        ...     message="Upstream service returned 503",
        ...     connector_name="grid/electricitymaps",
        ...     status_code=503
        ... )
    """
    pass


__all__ = [
    'ConnectorException',
    'ConnectorConfigError',
    'ConnectorAuthError',
    'ConnectorNetworkError',
    'ConnectorTimeoutError',
    'ConnectorRateLimitError',
    'ConnectorNotFoundError',
    'ConnectorValidationError',
    'ConnectorSecurityError',
    'ConnectorServerError',
]
