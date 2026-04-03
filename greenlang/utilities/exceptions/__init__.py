"""
GreenLang Custom Exception Hierarchy.

This module provides a comprehensive exception hierarchy for GreenLang with rich
error context for debugging, monitoring, and user feedback.

Exception Hierarchy:
    GreenLangException (base)
    ├── AgentException
    │   ├── ValidationError
    │   ├── ExecutionError
    │   ├── TimeoutError
    │   └── ConfigurationError
    ├── WorkflowException
    │   ├── DAGError
    │   ├── PolicyViolation
    │   ├── ResourceError
    │   └── OrchestrationError
    ├── DataException
    │   ├── InvalidSchema
    │   ├── MissingData
    │   ├── CorruptedData
    │   └── DataAccessError
    ├── ConnectorException
    │   ├── ConnectorConfigError
    │   ├── ConnectorAuthError
    │   ├── ConnectorNetworkError
    │   ├── ConnectorTimeoutError
    │   ├── ConnectorRateLimitError
    │   ├── ConnectorNotFoundError
    │   ├── ConnectorValidationError
    │   ├── ConnectorSecurityError
    │   └── ConnectorServerError
    ├── SecurityException
    │   ├── AuthenticationError
    │   ├── AuthorizationError
    │   ├── EncryptionError
    │   ├── SecretAccessError
    │   ├── PIIViolationError
    │   ├── EgressBlockedError
    │   └── CertificateError
    ├── IntegrationException
    │   ├── EmissionFactorError
    │   ├── EntityResolutionError
    │   ├── ExternalServiceError
    │   ├── APIClientError
    │   └── RateLimitError
    ├── InfrastructureException
    │   ├── DatabaseError
    │   ├── CacheError
    │   ├── StorageError
    │   ├── QueueError
    │   ├── CircuitBreakerOpenError
    │   ├── BulkheadFullError
    │   └── RetryExhaustedError
    ├── ComplianceException
    │   ├── EUDRViolationError
    │   ├── CSRDViolationError
    │   ├── CBAMViolationError
    │   ├── RegulatoryDeadlineError
    │   ├── AuditTrailError
    │   └── ProvenanceError
    └── CalculationException
        ├── EmissionCalculationError
        ├── UnitConversionError
        ├── FactorNotFoundError
        ├── MethodologyError
        └── BoundaryError

All exceptions include rich context:
- error_code: Unique error identifier
- agent_name: Name of agent that raised the error
- context: Dictionary with error-specific details
- timestamp: When the error occurred
- traceback: Full stack trace for debugging

Organization:
- base: Base exception class with rich context
- agent: Agent-related exceptions
- workflow: Workflow orchestration exceptions
- data: Data access and validation exceptions
- connector: External connector exceptions
- security: Authentication, authorization, and security exceptions
- integration: External service integration exceptions
- infrastructure: Database, cache, storage, and resilience exceptions
- compliance: Regulatory compliance and audit trail exceptions
- calculation: Emission calculation and measurement exceptions
- utils: Exception utilities and helpers

Example:
    >>> from greenlang.exceptions import ValidationError
    >>> raise ValidationError(
    ...     message="Invalid fuel type",
    ...     agent_name="FuelAgent",
    ...     context={"fuel_type": "invalid", "valid_types": ["natural_gas", "coal"]}
    ... )

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

# Import from separated modules
from greenlang.utilities.exceptions.base import GreenLangException

from greenlang.utilities.exceptions.agent import (
    AgentException,
    ValidationError,
    ExecutionError,
    TimeoutError,
    ConfigurationError,
)

from greenlang.utilities.exceptions.workflow import (
    WorkflowException,
    DAGError,
    PolicyViolation,
    ResourceError,
    OrchestrationError,
)

from greenlang.utilities.exceptions.data import (
    DataException,
    InvalidSchema,
    MissingData,
    CorruptedData,
    DataAccessError,
)

from greenlang.utilities.exceptions.connector import (
    ConnectorException,
    ConnectorConfigError,
    ConnectorAuthError,
    ConnectorNetworkError,
    ConnectorTimeoutError,
    ConnectorRateLimitError,
    ConnectorNotFoundError,
    ConnectorValidationError,
    ConnectorSecurityError,
    ConnectorServerError,
)

from greenlang.utilities.exceptions.security import (
    SecurityException,
    AuthenticationError,
    AuthorizationError,
    EncryptionError,
    SecretAccessError,
    PIIViolationError,
    EgressBlockedError,
    CertificateError,
)

from greenlang.utilities.exceptions.integration import (
    IntegrationException,
    EmissionFactorError,
    EntityResolutionError,
    ExternalServiceError,
    APIClientError,
    RateLimitError,
)

from greenlang.utilities.exceptions.infrastructure import (
    InfrastructureException,
    DatabaseError,
    CacheError,
    StorageError,
    QueueError,
    CircuitBreakerOpenError,
    BulkheadFullError,
    RetryExhaustedError,
)

from greenlang.utilities.exceptions.compliance import (
    ComplianceException,
    EUDRViolationError,
    CSRDViolationError,
    CBAMViolationError,
    RegulatoryDeadlineError,
    AuditTrailError,
    ProvenanceError,
)

from greenlang.utilities.exceptions.calculation import (
    CalculationException,
    EmissionCalculationError,
    UnitConversionError,
    FactorNotFoundError,
    MethodologyError,
    BoundaryError,
)

from greenlang.utilities.exceptions.utils import (
    format_exception_chain,
    is_retriable,
)


__all__ = [
    # Base
    'GreenLangException',

    # Agent Exceptions
    'AgentException',
    'ValidationError',
    'ExecutionError',
    'TimeoutError',
    'ConfigurationError',

    # Workflow Exceptions
    'WorkflowException',
    'DAGError',
    'PolicyViolation',
    'ResourceError',
    'OrchestrationError',

    # Data Exceptions
    'DataException',
    'InvalidSchema',
    'MissingData',
    'CorruptedData',
    'DataAccessError',

    # Connector Exceptions
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

    # Security Exceptions
    'SecurityException',
    'AuthenticationError',
    'AuthorizationError',
    'EncryptionError',
    'SecretAccessError',
    'PIIViolationError',
    'EgressBlockedError',
    'CertificateError',

    # Integration Exceptions
    'IntegrationException',
    'EmissionFactorError',
    'EntityResolutionError',
    'ExternalServiceError',
    'APIClientError',
    'RateLimitError',

    # Infrastructure Exceptions
    'InfrastructureException',
    'DatabaseError',
    'CacheError',
    'StorageError',
    'QueueError',
    'CircuitBreakerOpenError',
    'BulkheadFullError',
    'RetryExhaustedError',

    # Compliance Exceptions
    'ComplianceException',
    'EUDRViolationError',
    'CSRDViolationError',
    'CBAMViolationError',
    'RegulatoryDeadlineError',
    'AuditTrailError',
    'ProvenanceError',

    # Calculation Exceptions
    'CalculationException',
    'EmissionCalculationError',
    'UnitConversionError',
    'FactorNotFoundError',
    'MethodologyError',
    'BoundaryError',

    # Utils
    'format_exception_chain',
    'is_retriable',
]
