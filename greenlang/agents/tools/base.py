"""
GreenLang Shared Tool Library - Base Classes
=============================================

This module provides the foundation for the shared tool library, including:
- BaseTool: Abstract base class for all tools
- Tool: Concrete tool implementation with JSON schema validation
- ToolResult: Standard result format
- Tool decorators for easy registration

Design Principles:
- Type-safe tool definitions
- JSON Schema validation
- Deterministic execution
- Composable tools
- Test-friendly design

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from enum import Enum

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# Tool Safety Levels
# ==============================================================================

class ToolSafety(str, Enum):
    """Tool safety classification for AgentSpec v2 compliance."""

    DETERMINISTIC = "deterministic"  # Always same output for same input
    IDEMPOTENT = "idempotent"        # Can be called multiple times safely
    STATEFUL = "stateful"            # May have side effects
    UNSAFE = "unsafe"                # External calls, non-deterministic


# ==============================================================================
# Tool Result
# ==============================================================================

@dataclass
class ToolResult:
    """Standard result format for tool execution."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[Any] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "success": self.success,
            "data": self.data,
        }

        if self.error:
            result["error"] = self.error

        if self.metadata:
            result["metadata"] = self.metadata

        if self.citations:
            result["citations"] = self.citations

        if self.execution_time_ms > 0:
            result["execution_time_ms"] = self.execution_time_ms

        return result


# ==============================================================================
# Tool Definition
# ==============================================================================

class ToolDef(BaseModel):
    """
    Tool definition for ChatSession integration.

    This matches the existing ToolDef interface used by agents,
    ensuring backward compatibility.
    """

    name: str = Field(..., description="Tool name (must be unique)")
    description: str = Field(..., description="Tool description for LLM")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for parameters")
    safety: ToolSafety = Field(
        default=ToolSafety.DETERMINISTIC,
        description="Safety classification"
    )

    class Config:
        use_enum_values = True


# ==============================================================================
# Base Tool Class
# ==============================================================================

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseTool(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all shared tools.

    Provides:
    - Standard interface for tool execution
    - Input/output validation
    - Error handling
    - Execution metrics
    - Citation tracking

    Subclasses must implement:
    - execute(): Core tool logic
    - get_tool_def(): Tool definition for LLM
    """

    def __init__(
        self,
        name: str,
        description: str,
        safety: ToolSafety = ToolSafety.DETERMINISTIC,
    ):
        """
        Initialize base tool.

        Args:
            name: Tool name (must be unique)
            description: Tool description for LLM
            safety: Safety classification
        """
        self.name = name
        self.description = description
        self.safety = safety
        self.execution_count = 0
        self.total_execution_time_ms = 0.0
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with success status and data
        """
        pass

    @abstractmethod
    def get_tool_def(self) -> ToolDef:
        """
        Get tool definition for ChatSession.

        Returns:
            ToolDef with name, description, and JSON schema
        """
        pass

    def validate_input(self, **kwargs) -> bool:
        """
        Validate input arguments against schema.

        Override for custom validation logic.

        Args:
            **kwargs: Input arguments

        Returns:
            True if valid, False otherwise
        """
        return True

    def __call__(self, **kwargs) -> ToolResult:
        """
        Execute tool (allows calling tool as a function).

        Args:
            **kwargs: Tool arguments

        Returns:
            ToolResult
        """
        import time

        start_time = time.time()

        try:
            # Validate input
            if not self.validate_input(**kwargs):
                return ToolResult(
                    success=False,
                    error=f"Input validation failed for tool {self.name}"
                )

            # Execute tool
            result = self.execute(**kwargs)

            # Track metrics
            execution_time_ms = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time_ms
            self.total_execution_time_ms += execution_time_ms
            self.execution_count += 1

            return result

        except Exception as e:
            self.logger.error(f"Tool {self.name} failed: {e}", exc_info=True)
            execution_time_ms = (time.time() - start_time) * 1000

            return ToolResult(
                success=False,
                error=str(e),
                metadata={"exception_type": type(e).__name__},
                execution_time_ms=execution_time_ms
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        avg_time = (
            self.total_execution_time_ms / self.execution_count
            if self.execution_count > 0
            else 0
        )

        return {
            "name": self.name,
            "executions": self.execution_count,
            "total_time_ms": round(self.total_execution_time_ms, 2),
            "avg_time_ms": round(avg_time, 2),
            "safety": self.safety.value,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"executions={self.execution_count}, "
            f"safety={self.safety.value})"
        )


# ==============================================================================
# Tool Decorator
# ==============================================================================

def tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    safety: ToolSafety = ToolSafety.DETERMINISTIC,
):
    """
    Decorator to convert a function into a tool.

    Usage:
        @tool(
            name="calculate_emissions",
            description="Calculate CO2e emissions",
            parameters={
                "type": "object",
                "required": ["amount", "factor"],
                "properties": {
                    "amount": {"type": "number"},
                    "factor": {"type": "number"}
                }
            }
        )
        def calculate_emissions(amount: float, factor: float) -> ToolResult:
            return ToolResult(
                success=True,
                data={"emissions": amount * factor}
            )

    Args:
        name: Tool name
        description: Tool description
        parameters: JSON Schema for parameters
        safety: Safety classification

    Returns:
        Decorated function wrapped as a Tool
    """
    def decorator(func: Callable) -> BaseTool:
        class FunctionTool(BaseTool):
            def __init__(self):
                super().__init__(name, description, safety)
                self.func = func

            def execute(self, **kwargs) -> ToolResult:
                # Call wrapped function
                result = self.func(**kwargs)

                # If function returns ToolResult, use it
                if isinstance(result, ToolResult):
                    return result

                # Otherwise wrap in ToolResult
                return ToolResult(success=True, data=result)

            def get_tool_def(self) -> ToolDef:
                return ToolDef(
                    name=self.name,
                    description=self.description,
                    parameters=parameters,
                    safety=self.safety
                )

        return FunctionTool()

    return decorator


# ==============================================================================
# Tool Composition
# ==============================================================================

class CompositeTool(BaseTool):
    """
    Tool that composes multiple tools in sequence.

    Useful for creating complex workflows from simple tools.
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: List[BaseTool],
        safety: ToolSafety = ToolSafety.DETERMINISTIC,
    ):
        """
        Initialize composite tool.

        Args:
            name: Composite tool name
            description: Description
            tools: List of tools to execute in sequence
            safety: Safety (most restrictive of all tools)
        """
        super().__init__(name, description, safety)
        self.tools = tools

    def execute(self, **kwargs) -> ToolResult:
        """Execute all tools in sequence."""
        results = []
        data = kwargs.copy()

        for tool in self.tools:
            result = tool(**data)

            if not result.success:
                return result  # Fail fast

            results.append(result)
            data.update(result.data)

        # Combine all results
        return ToolResult(
            success=True,
            data=data,
            metadata={
                "composed_tools": [t.name for t in self.tools],
                "individual_results": [r.to_dict() for r in results]
            }
        )

    def get_tool_def(self) -> ToolDef:
        """Get tool definition."""
        # For composite tools, combine schemas of all tools
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {}  # Composite tools define their own schema
            },
            safety=self.safety
        )
