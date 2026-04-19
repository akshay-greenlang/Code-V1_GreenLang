"""
GreenLang Base Exception - Foundation for All Exceptions

This module provides the base exception class with rich error context.

Features:
- Unique error code generation
- Timestamp tracking
- Context dictionary for error details
- JSON serialization
- Stack trace capture

Author: GreenLang Team
Date: 2025-11-21
"""

from typing import Any, Dict, Optional
import traceback as tb
import json
import re

from greenlang.utilities.determinism.clock import DeterministicClock


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
        self.timestamp = DeterministicClock.now()
        self.traceback_str = "".join(tb.format_stack()[:-1])

    def _generate_error_code(self) -> str:
        """Generate unique error code based on exception class.

        Returns:
            Error code like "GL_AGENT_VALIDATION_001"
        """
        class_name = self.__class__.__name__
        # Convert CamelCase to SCREAMING_SNAKE_CASE
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


__all__ = [
    'GreenLangException',
]
