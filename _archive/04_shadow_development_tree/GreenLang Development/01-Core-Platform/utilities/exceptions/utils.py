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


__all__ = [
    'format_exception_chain',
    'is_retriable',
]
