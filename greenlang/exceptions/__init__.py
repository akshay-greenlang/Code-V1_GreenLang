# -*- coding: utf-8 -*-
"""GreenLang Custom Exception Hierarchy.

This module provides a comprehensive exception hierarchy for GreenLang with rich
error context for debugging, monitoring, and user feedback.

DEPRECATED: This module is now organized into submodules for better separation of concerns.
Please import from the specific submodules:
- greenlang.exceptions.base - Base exception class
- greenlang.exceptions.agent - Agent-related exceptions
- greenlang.exceptions.workflow - Workflow orchestration exceptions
- greenlang.exceptions.data - Data access and validation exceptions
- greenlang.exceptions.connector - External connector exceptions
- greenlang.exceptions.security - Authentication, authorization, and security exceptions
- greenlang.exceptions.integration - External service integration exceptions
- greenlang.exceptions.infrastructure - Database, cache, storage, and resilience exceptions
- greenlang.exceptions.compliance - Regulatory compliance and audit trail exceptions
- greenlang.exceptions.calculation - Emission calculation and measurement exceptions
- greenlang.exceptions.factors - Factor catalog, governance, and matching exceptions
- greenlang.exceptions.utils - Exception utilities

This file provides backward-compatible re-exports.

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
    ├── CalculationException
    │   ├── EmissionCalculationError
    │   ├── UnitConversionError
    │   ├── FactorNotFoundError
    │   ├── MethodologyError
    │   └── BoundaryError
    └── FactorsException
        ├── FactorValidationError
        ├── FactorIngestionError
        ├── FactorGovernanceError
        ├── FactorEditionError
        ├── FactorLicenseError
        ├── FactorMatchingError
        ├── SourceRegistryError
        ├── ParserError
        ├── QualityGateError
        └── WatchError

All exceptions include rich context:
- error_code: Unique error identifier
- agent_name: Name of agent that raised the error
- context: Dictionary with error-specific details
- timestamp: When the error occurred
- traceback: Full stack trace for debugging

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

import warnings

# Show deprecation warning when importing from this module
warnings.warn(
    "Importing from greenlang.exceptions is deprecated. "
    "Please import from specific submodules: "
    "greenlang.exceptions.base, greenlang.exceptions.agent, "
    "greenlang.exceptions.workflow, greenlang.exceptions.data, "
    "greenlang.exceptions.connector, greenlang.exceptions.security, "
    "greenlang.exceptions.integration, greenlang.exceptions.infrastructure, "
    "greenlang.exceptions.compliance, greenlang.exceptions.calculation, "
    "greenlang.exceptions.utils",
    DeprecationWarning,
    stacklevel=2
)

# Backward-compatible re-exports from utilities.exceptions
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
    BillingProviderError,
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

from greenlang.utilities.exceptions.factors import (
    FactorsException,
    FactorValidationError,
    FactorIngestionError,
    FactorGovernanceError,
    FactorEditionError,
    FactorLicenseError,
    FactorMatchingError,
    SourceRegistryError,
    ParserError,
    QualityGateError,
    WatchError,
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
    'BillingProviderError',

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

    # Factors Exceptions
    'FactorsException',
    'FactorValidationError',
    'FactorIngestionError',
    'FactorGovernanceError',
    'FactorEditionError',
    'FactorLicenseError',
    'FactorMatchingError',
    'SourceRegistryError',
    'ParserError',
    'QualityGateError',
    'WatchError',

    # Utils
    'format_exception_chain',
    'is_retriable',
]
