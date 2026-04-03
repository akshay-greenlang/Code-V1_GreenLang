"""
GreenLang Exception Utilities - Helper Functions for Error Handling

This module provides utility functions for exception handling and formatting.

Features:
- Exception chain formatting
- Retry logic determination
- Error context extraction

Author: GreenLang Team
Date: 2025-11-21
"""

from greenlang.exceptions.base import GreenLangException
from greenlang.exceptions.agent import ValidationError, TimeoutError
from greenlang.exceptions.workflow import PolicyViolation, DAGError, ResourceError
from greenlang.exceptions.data import InvalidSchema, DataAccessError

# Connector exceptions (retriable: timeout, network, rate limit, server)
from greenlang.exceptions.connector import (
    ConnectorTimeoutError,
    ConnectorNetworkError,
    ConnectorRateLimitError,
    ConnectorServerError,
    ConnectorConfigError,
    ConnectorAuthError,
    ConnectorValidationError,
    ConnectorSecurityError,
)

# Security exceptions (non-retriable: auth, authz, encryption, PII)
from greenlang.exceptions.security import (
    AuthenticationError,
    AuthorizationError,
    EncryptionError,
    PIIViolationError,
    EgressBlockedError,
    SecretAccessError,
    CertificateError,
)

# Integration exceptions (retriable: external service, rate limit)
from greenlang.exceptions.integration import (
    ExternalServiceError,
    RateLimitError,
    EmissionFactorError,
    EntityResolutionError,
    APIClientError,
)

# Infrastructure exceptions (retriable: database, cache, storage, queue, circuit)
from greenlang.exceptions.infrastructure import (
    DatabaseError,
    CacheError,
    StorageError,
    QueueError,
    CircuitBreakerOpenError,
    BulkheadFullError,
    RetryExhaustedError,
)

# Compliance exceptions (non-retriable: violations, provenance, audit)
from greenlang.exceptions.compliance import (
    EUDRViolationError,
    CSRDViolationError,
    CBAMViolationError,
    RegulatoryDeadlineError,
    AuditTrailError,
    ProvenanceError,
)

# Calculation exceptions (non-retriable: calc errors, unit, factor, methodology)
from greenlang.exceptions.calculation import (
    EmissionCalculationError,
    UnitConversionError,
    FactorNotFoundError,
    MethodologyError,
    BoundaryError,
)


def format_exception_chain(exc: Exception) -> str:
    """Format exception chain for logging/display.

    Args:
        exc: Exception to format

    Returns:
        Formatted string with full exception chain
    """
    lines = []
    current = exc

    while current is not None:
        if isinstance(current, GreenLangException):
            lines.append(str(current))
            lines.append(f"  Context: {current.context}")
        else:
            lines.append(f"{type(current).__name__}: {current}")

        # Get cause
        current = getattr(current, "__cause__", None)

    return "\n".join(lines)


def is_retriable(exc: Exception) -> bool:
    """Check if exception is retriable.

    Retriable exceptions are transient failures where retrying the
    operation may succeed: timeouts, network errors, rate limits,
    server errors, database/cache/storage/queue failures, and circuit
    breaker rejections.

    Non-retriable exceptions are deterministic failures where retrying
    will produce the same result: validation errors, schema errors,
    policy violations, auth failures, calculation errors, compliance
    violations, and exhausted retries.

    Args:
        exc: Exception to check

    Returns:
        True if operation should be retried
    """
    # Retriable: transient failures that may succeed on retry
    retriable_types = (
        # Original
        TimeoutError,
        ResourceError,
        DataAccessError,
        # Connector: network, timeout, rate limit, server errors
        ConnectorTimeoutError,
        ConnectorNetworkError,
        ConnectorRateLimitError,
        ConnectorServerError,
        # Integration: external service, rate limit
        ExternalServiceError,
        RateLimitError,
        # Infrastructure: database, cache, storage, queue, circuit breaker, bulkhead
        DatabaseError,
        CacheError,
        StorageError,
        QueueError,
        CircuitBreakerOpenError,
        BulkheadFullError,
        # Security: secret access (Vault may be temporarily unavailable)
        SecretAccessError,
        # Security: certificate (may be mid-rotation)
        CertificateError,
    )

    # Non-retriable: deterministic failures that will not change on retry
    non_retriable_types = (
        # Original
        ValidationError,
        InvalidSchema,
        PolicyViolation,
        DAGError,
        # Connector: config, auth, validation, security
        ConnectorConfigError,
        ConnectorAuthError,
        ConnectorValidationError,
        ConnectorSecurityError,
        # Security: auth, authz, encryption, PII, egress
        AuthenticationError,
        AuthorizationError,
        EncryptionError,
        PIIViolationError,
        EgressBlockedError,
        # Integration: factor lookup, entity resolution, API client
        EmissionFactorError,
        EntityResolutionError,
        APIClientError,
        # Compliance: all violations and provenance errors
        EUDRViolationError,
        CSRDViolationError,
        CBAMViolationError,
        RegulatoryDeadlineError,
        AuditTrailError,
        ProvenanceError,
        # Calculation: all calculation errors
        EmissionCalculationError,
        UnitConversionError,
        FactorNotFoundError,
        MethodologyError,
        BoundaryError,
        # Infrastructure: retry exhausted (already retried max times)
        RetryExhaustedError,
    )

    if isinstance(exc, retriable_types):
        return True
    if isinstance(exc, non_retriable_types):
        return False

    # Unknown exceptions: don't retry by default
    return False


__all__ = [
    'format_exception_chain',
    'is_retriable',
]
