"""
GreenLang Integration Exceptions - External Service Integration Errors

This module provides exception classes for integration-related errors
including emission factor lookups, entity resolution, external service
calls, and API client failures.

Features:
- Emission factor lookup failures
- Entity resolution errors
- External service communication errors
- API client errors
- Rate limiting for integrated services

Author: GreenLang Team
Date: 2026-04-02
"""

from typing import Any, Dict, Optional

from greenlang.exceptions.base import GreenLangException


class IntegrationException(GreenLangException):
    """Base exception for integration-related errors.

    Raised when external service integrations fail including factor
    brokers, entity resolution services, and third-party APIs.
    """
    ERROR_PREFIX = "GL_INTEGRATION"


class EmissionFactorError(IntegrationException):
    """Emission factor lookup or retrieval failed.

    Raised when the factor broker cannot find, retrieve, or validate
    an emission factor from its configured data sources.

    Example:
        >>> raise EmissionFactorError(
        ...     message="Emission factor not found for fuel type",
        ...     factor_source="EPA",
        ...     query={"fuel_type": "jet_fuel", "year": 2025, "region": "US"}
        ... )
    """

    def __init__(
        self,
        message: str,
        factor_source: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize emission factor error.

        Args:
            message: Error message
            factor_source: Source of the emission factor (EPA, DEFRA, IPCC)
            query: Query parameters used for the lookup
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if factor_source:
            context["factor_source"] = factor_source
        if query:
            context["query"] = query
        super().__init__(message, agent_name=agent_name, context=context)


class EntityResolutionError(IntegrationException):
    """Entity resolution or matching failed.

    Raised when the system cannot resolve an entity (supplier, material,
    facility) against master data.

    Example:
        >>> raise EntityResolutionError(
        ...     message="Cannot resolve supplier name to master record",
        ...     entity_type="supplier",
        ...     entity_value="Acme Corp Ltd.",
        ...     context={"candidates": ["Acme Corporation", "Acme Co."]}
        ... )
    """

    def __init__(
        self,
        message: str,
        entity_type: Optional[str] = None,
        entity_value: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize entity resolution error.

        Args:
            message: Error message
            entity_type: Type of entity (supplier, material, facility)
            entity_value: Value that could not be resolved
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if entity_type:
            context["entity_type"] = entity_type
        if entity_value:
            context["entity_value"] = entity_value
        super().__init__(message, agent_name=agent_name, context=context)


class ExternalServiceError(IntegrationException):
    """External service call failed.

    Raised when a call to an external service (ERP, CRM, data provider)
    fails for reasons other than authentication or rate limiting.

    Example:
        >>> raise ExternalServiceError(
        ...     message="SAP connection failed",
        ...     service_name="SAP S/4HANA",
        ...     operation="material_master_lookup",
        ...     cause=original_exception
        ... )
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize external service error.

        Args:
            message: Error message
            service_name: Name of the external service
            operation: Operation that was being performed
            cause: Original exception
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if service_name:
            context["service_name"] = service_name
        if operation:
            context["operation"] = operation
        if cause:
            context["cause"] = str(cause)
            context["cause_type"] = type(cause).__name__
        super().__init__(message, agent_name=agent_name, context=context)


class APIClientError(IntegrationException):
    """API client request failed.

    Raised when an outbound API call fails due to client-side issues
    such as invalid request format, missing parameters, or serialization
    errors.

    Example:
        >>> raise APIClientError(
        ...     message="Invalid request payload",
        ...     endpoint="/api/v1/factors",
        ...     http_method="POST",
        ...     status_code=400
        ... )
    """

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        http_method: Optional[str] = None,
        status_code: Optional[int] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize API client error.

        Args:
            message: Error message
            endpoint: API endpoint that was called
            http_method: HTTP method used (GET, POST, PUT, DELETE)
            status_code: HTTP status code returned
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if endpoint:
            context["endpoint"] = endpoint
        if http_method:
            context["http_method"] = http_method
        if status_code is not None:
            context["status_code"] = status_code
        super().__init__(message, agent_name=agent_name, context=context)


class RateLimitError(IntegrationException):
    """External service rate limit exceeded.

    Raised when an integrated service rejects the request due to rate
    limiting. Distinct from ConnectorRateLimitError in that this covers
    higher-level service integrations, not raw connector calls.

    Example:
        >>> raise RateLimitError(
        ...     message="DEFRA API rate limit exceeded",
        ...     service_name="DEFRA",
        ...     retry_after_seconds=120
        ... )
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            service_name: Name of the rate-limited service
            retry_after_seconds: Seconds until retry is allowed
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if service_name:
            context["service_name"] = service_name
        if retry_after_seconds is not None:
            context["retry_after_seconds"] = retry_after_seconds
        super().__init__(message, agent_name=agent_name, context=context)


__all__ = [
    'IntegrationException',
    'EmissionFactorError',
    'EntityResolutionError',
    'ExternalServiceError',
    'APIClientError',
    'RateLimitError',
]
