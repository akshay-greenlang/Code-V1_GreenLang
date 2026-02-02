"""
GreenLang Agent Exceptions - Agent Execution Errors

This module provides exception classes for agent-related errors.

Features:
- Validation errors with field-level details
- Execution errors with step tracking
- Timeout errors with timing information
- Configuration errors

Author: GreenLang Team
Date: 2025-11-21
"""

from typing import Any, Dict, Optional

from greenlang.exceptions.base import GreenLangException


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


__all__ = [
    'AgentException',
    'ValidationError',
    'ExecutionError',
    'TimeoutError',
    'ConfigurationError',
]
