"""
Lifecycle hooks for agent execution.

Provides pre/post hooks for validation and execution phases.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class LifecycleHooks(ABC, Generic[InT, OutT]):
    """Base class for lifecycle hooks."""

    async def pre_validate(self, input_data: InT, context: dict) -> InT:
        """
        Hook called before input validation.

        Use this to:
        - Transform input data
        - Add context-specific defaults
        - Log input received

        Args:
            input_data: Raw input data
            context: Execution context

        Returns:
            Potentially modified input data
        """
        return input_data

    async def post_validate(self, validated_input: InT, context: dict) -> InT:
        """
        Hook called after input validation passes.

        Use this to:
        - Enrich validated data
        - Start timers or metrics
        - Initialize resources

        Args:
            validated_input: Validated input data
            context: Execution context

        Returns:
            Potentially enriched input data
        """
        return validated_input

    async def pre_execute(self, input_data: InT, context: dict) -> InT:
        """
        Hook called before main execution.

        Use this to:
        - Set up execution environment
        - Initialize calculators or validators
        - Record execution start

        Args:
            input_data: Validated input data
            context: Execution context

        Returns:
            Input data ready for execution
        """
        return input_data

    async def post_execute(self, output_data: OutT, context: dict) -> OutT:
        """
        Hook called after execution completes.

        Use this to:
        - Transform output data
        - Add metadata
        - Clean up resources

        Args:
            output_data: Raw output data
            context: Execution context

        Returns:
            Potentially modified output data
        """
        return output_data

    async def on_error(self, error: Exception, context: dict) -> None:
        """
        Hook called when an error occurs.

        Use this to:
        - Log errors with context
        - Clean up resources
        - Send alerts

        Args:
            error: The exception that occurred
            context: Execution context
        """
        pass
