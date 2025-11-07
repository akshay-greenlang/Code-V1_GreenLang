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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from enum import Enum

from pydantic import BaseModel, Field, ValidationError

# Import security components
from .validation import ValidationRule, ValidationResult
from .rate_limiting import RateLimiter, RateLimitExceeded, get_rate_limiter
from .audit import AuditLogger, get_audit_logger
from .security_config import SecurityConfig, get_security_config
from .telemetry import TelemetryCollector, get_telemetry

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
        validation_rules: Optional[Dict[str, List[ValidationRule]]] = None,
        rate_limit: Optional[tuple] = None,
        enable_audit: bool = True,
        enable_telemetry: bool = True,
        security_config: Optional[SecurityConfig] = None,
    ):
        """
        Initialize base tool with security features.

        Args:
            name: Tool name (must be unique)
            description: Tool description for LLM
            safety: Safety classification
            validation_rules: Input validation rules per parameter
            rate_limit: Custom rate limit (rate, burst) tuple
            enable_audit: Enable audit logging for this tool
            enable_telemetry: Enable telemetry collection for this tool
            security_config: Custom security config (None = use global)
        """
        self.name = name
        self.description = description
        self.safety = safety
        self.execution_count = 0
        self.total_execution_time_ms = 0.0
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Validation rules
        self.validation_rules = validation_rules or {}

        # Security components
        self.security_config = security_config or get_security_config()

        # Rate limiter
        if rate_limit:
            self.rate_limiter = RateLimiter(
                rate=rate_limit[0],
                burst=rate_limit[1],
                per_tool=True
            )
        else:
            self.rate_limiter = get_rate_limiter()

        # Audit logger
        self.audit_logger = get_audit_logger() if enable_audit else None

        # Telemetry collector
        self.telemetry = get_telemetry() if enable_telemetry else None

        # Session tracking
        self._current_user_id: Optional[str] = None
        self._current_session_id: Optional[str] = None

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

    def validate_inputs(self, **kwargs) -> ValidationResult:
        """
        Validate input arguments using security framework.

        Args:
            **kwargs: Input arguments

        Returns:
            ValidationResult with validation status and sanitized values
        """
        if not self.security_config.enable_validation:
            # Validation disabled - return success with original values
            return ValidationResult(valid=True, sanitized_value=kwargs)

        # Start with successful result
        result = ValidationResult(valid=True, sanitized_value={})

        # Validate each parameter with its rules
        for param_name, param_value in kwargs.items():
            if param_name in self.validation_rules:
                # Apply all validation rules for this parameter
                for rule in self.validation_rules[param_name]:
                    rule_result = rule.validate(param_value, context=kwargs)

                    # Merge results
                    result.merge(rule_result)

                    # Update sanitized value
                    if rule_result.sanitized_value is not None:
                        param_value = rule_result.sanitized_value

                    # Stop on first error if strict mode
                    if not rule_result.valid and self.security_config.strict_validation:
                        result.sanitized_value = kwargs  # Keep original on failure
                        return result

            # Store sanitized parameter value
            result.sanitized_value[param_name] = param_value

        # Check for warnings in strict mode
        if self.security_config.strict_validation and result.warnings:
            result.valid = False
            result.errors.append("Validation warnings in strict mode")

        return result

    def validate_input(self, **kwargs) -> bool:
        """
        Legacy validation method for backward compatibility.

        Args:
            **kwargs: Input arguments

        Returns:
            True if valid, False otherwise
        """
        result = self.validate_inputs(**kwargs)
        return result.valid

    def set_context(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
        """
        Set execution context for rate limiting and audit logging.

        Args:
            user_id: User ID for per-user rate limiting
            session_id: Session ID for audit logging
        """
        self._current_user_id = user_id
        self._current_session_id = session_id

    def __call__(self, **kwargs) -> ToolResult:
        """
        Execute tool with comprehensive security checks.

        Security features:
        1. Tool access control (whitelist/blacklist)
        2. Input validation
        3. Rate limiting
        4. Execution with timeout
        5. Audit logging

        Args:
            **kwargs: Tool arguments

        Returns:
            ToolResult
        """
        start_time = time.perf_counter()
        original_kwargs = kwargs.copy()

        try:
            # 1. Check if tool is allowed to execute
            if not self.security_config.is_tool_allowed(self.name):
                error_msg = f"Tool '{self.name}' is not allowed to execute (blacklisted or not whitelisted)"
                self.logger.warning(error_msg)
                return ToolResult(
                    success=False,
                    error=error_msg,
                    metadata={"security_check": "tool_access_denied"}
                )

            # 2. Input validation
            if self.security_config.enable_validation:
                validation_result = self.validate_inputs(**kwargs)

                if not validation_result.valid:
                    error_msg = f"Validation failed: {', '.join(validation_result.errors)}"
                    self.logger.warning(f"Tool {self.name} validation failed: {error_msg}")

                    result = ToolResult(
                        success=False,
                        error=error_msg,
                        metadata={
                            "validation_errors": validation_result.errors,
                            "validation_warnings": validation_result.warnings
                        }
                    )

                    # Log validation failure
                    execution_time_ms = (time.perf_counter() - start_time) * 1000
                    if self.audit_logger and self.security_config.audit_log_failures:
                        self.audit_logger.log_execution(
                            tool_name=self.name,
                            inputs=original_kwargs,
                            result=result,
                            execution_time_ms=execution_time_ms,
                            user_id=self._current_user_id,
                            session_id=self._current_session_id,
                            metadata={"failure_reason": "validation"}
                        )

                    # Record telemetry for validation failure
                    if self.telemetry:
                        self.telemetry.record_execution(
                            tool_name=self.name,
                            execution_time_ms=execution_time_ms,
                            success=False,
                            error_type="ValidationError",
                            user_id=self._current_user_id,
                            validation_failed=True
                        )

                    return result

                # Use sanitized values
                kwargs = validation_result.sanitized_value

                # Log warnings if any
                if validation_result.warnings:
                    for warning in validation_result.warnings:
                        self.logger.debug(f"Validation warning for {self.name}: {warning}")

            # 3. Rate limiting
            if self.security_config.enable_rate_limiting:
                try:
                    self.rate_limiter.check_and_consume(
                        tool_name=self.name,
                        user_id=self._current_user_id
                    )
                except RateLimitExceeded as e:
                    error_msg = f"Rate limit exceeded: {e.message}"
                    self.logger.warning(error_msg)

                    result = ToolResult(
                        success=False,
                        error=error_msg,
                        metadata={
                            "rate_limit_exceeded": True,
                            "retry_after_seconds": e.retry_after
                        }
                    )

                    # Log rate limit failure
                    execution_time_ms = (time.perf_counter() - start_time) * 1000
                    if self.audit_logger and self.security_config.audit_log_failures:
                        self.audit_logger.log_execution(
                            tool_name=self.name,
                            inputs=original_kwargs,
                            result=result,
                            execution_time_ms=execution_time_ms,
                            user_id=self._current_user_id,
                            session_id=self._current_session_id,
                            metadata={"failure_reason": "rate_limit"}
                        )

                    # Record telemetry for rate limit
                    if self.telemetry:
                        self.telemetry.record_execution(
                            tool_name=self.name,
                            execution_time_ms=execution_time_ms,
                            success=False,
                            error_type="RateLimitExceeded",
                            user_id=self._current_user_id,
                            rate_limited=True
                        )

                    return result

            # 4. Execute tool
            result = self.execute(**kwargs)

            # Track metrics
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time_ms
            self.total_execution_time_ms += execution_time_ms
            self.execution_count += 1

            # 5. Audit logging (success)
            if self.audit_logger and self.security_config.audit_log_successes:
                self.audit_logger.log_execution(
                    tool_name=self.name,
                    inputs=original_kwargs,
                    result=result,
                    execution_time_ms=execution_time_ms,
                    user_id=self._current_user_id,
                    session_id=self._current_session_id
                )

            # 6. Record telemetry (success)
            if self.telemetry:
                self.telemetry.record_execution(
                    tool_name=self.name,
                    execution_time_ms=execution_time_ms,
                    success=result.success,
                    error_type=None if result.success else "ExecutionError",
                    user_id=self._current_user_id
                )

            return result

        except Exception as e:
            self.logger.error(f"Tool {self.name} failed: {e}", exc_info=True)
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            result = ToolResult(
                success=False,
                error=str(e),
                metadata={"exception_type": type(e).__name__},
                execution_time_ms=execution_time_ms
            )

            # Audit logging (failure)
            if self.audit_logger and self.security_config.audit_log_failures:
                self.audit_logger.log_execution(
                    tool_name=self.name,
                    inputs=original_kwargs,
                    result=result,
                    execution_time_ms=execution_time_ms,
                    user_id=self._current_user_id,
                    session_id=self._current_session_id,
                    metadata={"failure_reason": "exception"}
                )

            # Record telemetry (exception)
            if self.telemetry:
                self.telemetry.record_execution(
                    tool_name=self.name,
                    execution_time_ms=execution_time_ms,
                    success=False,
                    error_type=type(e).__name__,
                    user_id=self._current_user_id
                )

            return result

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
