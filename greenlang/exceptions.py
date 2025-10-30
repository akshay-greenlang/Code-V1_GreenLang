"""GreenLang Custom Exception Hierarchy.

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
    └── DataException
        ├── InvalidSchema
        ├── MissingData
        ├── CorruptedData
        └── DataAccessError

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

from typing import Any, Dict, Optional
from datetime import datetime
import traceback as tb
import json


# ==============================================================================
# Base Exception
# ==============================================================================

class GreenLangException(Exception):
    """Base exception for all GreenLang errors.

    Provides rich error context for debugging, monitoring, and user feedback.

    Attributes:
        message: Human-readable error message
        error_code: Unique error identifier (e.g., "GL_AGENT_VALIDATION_001")
        agent_name: Name of agent that raised the error (optional)
        context: Dictionary with error-specific details
        timestamp: When the error occurred
        traceback_str: Full stack trace for debugging
    """

    # Base error code prefix
    ERROR_PREFIX = "GL"

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize GreenLang exception with rich context.

        Args:
            message: Human-readable error message
            error_code: Unique error identifier (auto-generated if not provided)
            agent_name: Name of agent that raised the error
            context: Dictionary with error-specific details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.agent_name = agent_name
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback_str = "".join(tb.format_stack()[:-1])

    def _generate_error_code(self) -> str:
        """Generate unique error code based on exception class.

        Returns:
            Error code like "GL_AGENT_VALIDATION_001"
        """
        class_name = self.__class__.__name__
        # Convert CamelCase to SCREAMING_SNAKE_CASE
        import re
        error_type = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).upper()
        return f"{self.ERROR_PREFIX}_{error_type}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization.

        Returns:
            Dictionary with all error details
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "agent_name": self.agent_name,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback_str,
        }

    def to_json(self) -> str:
        """Convert exception to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """String representation with error code and message."""
        parts = [f"[{self.error_code}]"]
        if self.agent_name:
            parts.append(f"Agent: {self.agent_name}")
        parts.append(self.message)
        return " - ".join(parts)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"agent_name='{self.agent_name}')"
        )


# ==============================================================================
# Agent Exceptions
# ==============================================================================

class AgentException(GreenLangException):
    """Base exception for agent-related errors.

    Raised when an agent encounters an error during execution.
    """
    ERROR_PREFIX = "GL_AGENT"


class ValidationError(AgentException):
    """Input validation failed.

    Raised when agent input does not meet validation requirements.

    Example:
        >>> raise ValidationError(
        ...     message="Missing required field: fuel_type",
        ...     agent_name="FuelAgent",
        ...     context={"input": {"amount": 100}, "missing_fields": ["fuel_type"]}
        ... )
    """

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        invalid_fields: Optional[Dict[str, str]] = None,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            agent_name: Name of agent
            context: Error context
            invalid_fields: Dictionary of field_name -> reason
        """
        if invalid_fields:
            context = context or {}
            context["invalid_fields"] = invalid_fields
        super().__init__(message, agent_name=agent_name, context=context)


class ExecutionError(AgentException):
    """Agent execution failed.

    Raised when an agent encounters an error during execution.

    Example:
        >>> raise ExecutionError(
        ...     message="Failed to calculate emissions",
        ...     agent_name="FuelAgent",
        ...     context={"step": "calculate", "input": {...}, "cause": "Division by zero"}
        ... )
    """

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        step: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize execution error.

        Args:
            message: Error message
            agent_name: Name of agent
            context: Error context
            step: Execution step where error occurred
            cause: Original exception that caused this error
        """
        context = context or {}
        if step:
            context["step"] = step
        if cause:
            context["cause"] = str(cause)
            context["cause_type"] = type(cause).__name__
        super().__init__(message, agent_name=agent_name, context=context)


class TimeoutError(AgentException):
    """Agent execution timed out.

    Raised when an agent exceeds its execution time limit.

    Example:
        >>> raise TimeoutError(
        ...     message="Execution timed out after 30 seconds",
        ...     agent_name="FuelAgent",
        ...     context={"timeout_seconds": 30, "elapsed_seconds": 31.5}
        ... )
    """

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
    ):
        """Initialize timeout error.

        Args:
            message: Error message
            agent_name: Name of agent
            context: Error context
            timeout_seconds: Configured timeout limit
            elapsed_seconds: Actual execution time
        """
        context = context or {}
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        if elapsed_seconds:
            context["elapsed_seconds"] = elapsed_seconds
        super().__init__(message, agent_name=agent_name, context=context)


class ConfigurationError(AgentException):
    """Agent configuration is invalid.

    Raised when an agent is misconfigured or missing required configuration.

    Example:
        >>> raise ConfigurationError(
        ...     message="Missing API key configuration",
        ...     agent_name="FuelAgent",
        ...     context={"config_key": "OPENAI_API_KEY", "config_file": ".env"}
        ... )
    """
    pass


# ==============================================================================
# Workflow Exceptions
# ==============================================================================

class WorkflowException(GreenLangException):
    """Base exception for workflow-related errors.

    Raised when workflow orchestration encounters an error.
    """
    ERROR_PREFIX = "GL_WORKFLOW"


class DAGError(WorkflowException):
    """Workflow DAG (Directed Acyclic Graph) is invalid.

    Raised when workflow DAG contains cycles, missing dependencies, or other
    structural issues.

    Example:
        >>> raise DAGError(
        ...     message="Cycle detected in workflow DAG",
        ...     context={"cycle": ["step1", "step2", "step3", "step1"]}
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        invalid_nodes: Optional[list] = None,
    ):
        """Initialize DAG error.

        Args:
            message: Error message
            context: Error context
            workflow_id: ID of workflow with invalid DAG
            invalid_nodes: List of problematic nodes
        """
        context = context or {}
        if workflow_id:
            context["workflow_id"] = workflow_id
        if invalid_nodes:
            context["invalid_nodes"] = invalid_nodes
        super().__init__(message, context=context)


class PolicyViolation(WorkflowException):
    """Workflow violates policy constraints.

    Raised when workflow execution violates security, compliance, or resource
    policies.

    Example:
        >>> raise PolicyViolation(
        ...     message="Workflow exceeds maximum execution time policy",
        ...     context={"policy": "max_execution_time", "limit": 300, "actual": 350}
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        policy_name: Optional[str] = None,
        violation_details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize policy violation.

        Args:
            message: Error message
            context: Error context
            policy_name: Name of violated policy
            violation_details: Details about the violation
        """
        context = context or {}
        if policy_name:
            context["policy_name"] = policy_name
        if violation_details:
            context["violation_details"] = violation_details
        super().__init__(message, context=context)


class ResourceError(WorkflowException):
    """Workflow resource constraint exceeded.

    Raised when workflow exceeds resource limits (CPU, memory, disk, network).

    Example:
        >>> raise ResourceError(
        ...     message="Memory limit exceeded",
        ...     context={"resource": "memory", "limit_mb": 1024, "used_mb": 1500}
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        resource_type: Optional[str] = None,
        limit: Optional[float] = None,
        used: Optional[float] = None,
    ):
        """Initialize resource error.

        Args:
            message: Error message
            context: Error context
            resource_type: Type of resource (cpu, memory, disk, network)
            limit: Resource limit
            used: Actual resource usage
        """
        context = context or {}
        if resource_type:
            context["resource_type"] = resource_type
        if limit:
            context["limit"] = limit
        if used:
            context["used"] = used
        super().__init__(message, context=context)


class OrchestrationError(WorkflowException):
    """Workflow orchestration failed.

    Raised when workflow orchestrator encounters an error coordinating agent
    execution.

    Example:
        >>> raise OrchestrationError(
        ...     message="Failed to coordinate parallel agent execution",
        ...     context={"failed_agents": ["agent1", "agent2"], "reason": "Deadlock"}
        ... )
    """
    pass


# ==============================================================================
# Data Exceptions
# ==============================================================================

class DataException(GreenLangException):
    """Base exception for data-related errors.

    Raised when data access, validation, or processing fails.
    """
    ERROR_PREFIX = "GL_DATA"


class InvalidSchema(DataException):
    """Data schema is invalid.

    Raised when data does not conform to expected schema.

    Example:
        >>> raise InvalidSchema(
        ...     message="Input data does not match schema",
        ...     context={
        ...         "expected_schema": {"fuel_type": "string", "amount": "number"},
        ...         "actual_data": {"fuel_type": 123, "amount": "invalid"},
        ...         "errors": ["fuel_type: expected string, got int"]
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        actual_data: Optional[Dict[str, Any]] = None,
        schema_errors: Optional[list] = None,
    ):
        """Initialize schema error.

        Args:
            message: Error message
            context: Error context
            expected_schema: Expected schema definition
            actual_data: Actual data that failed validation
            schema_errors: List of schema validation errors
        """
        context = context or {}
        if expected_schema:
            context["expected_schema"] = expected_schema
        if actual_data:
            context["actual_data"] = actual_data
        if schema_errors:
            context["schema_errors"] = schema_errors
        super().__init__(message, context=context)


class MissingData(DataException):
    """Required data is missing.

    Raised when required data fields or resources are not found.

    Example:
        >>> raise MissingData(
        ...     message="Required emission factor not found",
        ...     context={
        ...         "data_type": "emission_factor",
        ...         "query": {"fuel_type": "natural_gas", "country": "US"},
        ...         "available_factors": ["coal", "diesel"]
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        data_type: Optional[str] = None,
        missing_fields: Optional[list] = None,
    ):
        """Initialize missing data error.

        Args:
            message: Error message
            context: Error context
            data_type: Type of missing data
            missing_fields: List of missing field names
        """
        context = context or {}
        if data_type:
            context["data_type"] = data_type
        if missing_fields:
            context["missing_fields"] = missing_fields
        super().__init__(message, context=context)


class CorruptedData(DataException):
    """Data is corrupted or malformed.

    Raised when data integrity checks fail or data is malformed.

    Example:
        >>> raise CorruptedData(
        ...     message="Checksum verification failed",
        ...     context={
        ...         "file_path": "/data/emissions.json",
        ...         "expected_checksum": "abc123",
        ...         "actual_checksum": "def456"
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        data_source: Optional[str] = None,
        corruption_details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize corrupted data error.

        Args:
            message: Error message
            context: Error context
            data_source: Source of corrupted data
            corruption_details: Details about the corruption
        """
        context = context or {}
        if data_source:
            context["data_source"] = data_source
        if corruption_details:
            context["corruption_details"] = corruption_details
        super().__init__(message, context=context)


class DataAccessError(DataException):
    """Data access failed.

    Raised when data cannot be accessed (permissions, network, etc.).

    Example:
        >>> raise DataAccessError(
        ...     message="Failed to access database",
        ...     context={
        ...         "database": "emissions_db",
        ...         "operation": "SELECT",
        ...         "error": "Connection timeout"
        ...     }
        ... )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        data_source: Optional[str] = None,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize data access error.

        Args:
            message: Error message
            context: Error context
            data_source: Data source that failed
            operation: Operation that failed (read, write, delete)
            cause: Original exception
        """
        context = context or {}
        if data_source:
            context["data_source"] = data_source
        if operation:
            context["operation"] = operation
        if cause:
            context["cause"] = str(cause)
            context["cause_type"] = type(cause).__name__
        super().__init__(message, context=context)


# ==============================================================================
# Exception Utilities
# ==============================================================================

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

    Args:
        exc: Exception to check

    Returns:
        True if operation should be retried
    """
    # Retriable: timeout, resource errors
    retriable_types = (TimeoutError, ResourceError, DataAccessError)

    # Non-retriable: validation, schema, policy violations
    non_retriable_types = (ValidationError, InvalidSchema, PolicyViolation, DAGError)

    if isinstance(exc, retriable_types):
        return True
    if isinstance(exc, non_retriable_types):
        return False

    # Unknown exceptions: don't retry by default
    return False
