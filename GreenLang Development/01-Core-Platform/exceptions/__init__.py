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

import warnings

# Show deprecation warning when importing from this module
warnings.warn(
    "Importing from greenlang.exceptions is deprecated. "
    "Please import from specific submodules: "
    "greenlang.exceptions.base, greenlang.exceptions.agent, "
    "greenlang.exceptions.workflow, greenlang.exceptions.data, "
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

    # Utils
    'format_exception_chain',
    'is_retriable',
]
