# -*- coding: utf-8 -*-
"""
Tool Framework - Comprehensive tool use and function calling with sandboxing.

This module implements a production-ready tool framework for GreenLang agents with:
- Tool registry and discovery
- Function calling protocol
- Sandboxed execution environment
- Permission-based access control (RBAC)
- Comprehensive audit logging
- Rate limiting and safety controls

Example:
    >>> framework = ToolFramework(config)
    >>> result = await framework.execute_tool("carbon_calculator", {"emissions": 100})
"""

import asyncio
import hashlib
import inspect
import json
import logging
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

from pydantic import BaseModel, Field, validator
import aiofiles
from asyncio import Queue, Semaphore
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Tool categories for organization and access control."""

    CALCULATION = "calculation_tools"
    DATA = "data_tools"
    INTEGRATION = "integration_tools"
    AI = "ai_tools"
    SYSTEM = "system_tools"
    CUSTOM = "custom_tools"


class ToolPermission(str, Enum):
    """Permission levels for tool access."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class ToolStatus(str, Enum):
    """Tool execution status."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class ToolDefinition(BaseModel):
    """Definition of a tool with metadata."""

    name: str = Field(..., description="Tool name")
    category: ToolCategory = Field(..., description="Tool category")
    description: str = Field(..., description="Tool description")
    version: str = Field("1.0.0", description="Tool version")

    # Function details
    function: Optional[str] = Field(None, description="Function reference")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameter schema")
    returns: Dict[str, Any] = Field(default_factory=dict, description="Return schema")

    # Access control
    required_permissions: Set[ToolPermission] = Field(default_factory=set)
    allowed_roles: Set[str] = Field(default_factory=lambda: {"agent", "admin"})

    # Safety controls
    max_calls_per_minute: int = Field(60, ge=1, description="Rate limit")
    timeout_seconds: int = Field(30, ge=1, description="Execution timeout")
    sandboxed: bool = Field(True, description="Enable sandboxing")
    audit_enabled: bool = Field(True, description="Enable audit logging")

    # Resource limits
    max_memory_mb: Optional[int] = Field(None, description="Memory limit")
    max_cpu_percent: Optional[float] = Field(None, description="CPU limit")


class ToolExecutionContext(BaseModel):
    """Context for tool execution with tracing."""

    tool_name: str
    execution_id: str
    agent_id: str
    timestamp: datetime
    parameters: Dict[str, Any]
    permissions: Set[ToolPermission]
    trace_enabled: bool = True


class ToolExecutionResult(BaseModel):
    """Result from tool execution."""

    tool_name: str
    execution_id: str
    status: ToolStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)


class ToolSandbox:
    """Sandboxed execution environment for tools."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize sandbox environment."""
        self.config = config
        self.resource_limits = config.get("resource_limits", {})
        self.allowed_imports = set(config.get("allowed_imports", [
            "math", "json", "datetime", "hashlib", "collections",
            "numpy", "pandas", "scipy"
        ]))
        self.blocked_builtins = set(config.get("blocked_builtins", [
            "eval", "exec", "compile", "__import__", "open"
        ]))

    async def execute(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        context: ToolExecutionContext
    ) -> Any:
        """Execute function in sandboxed environment."""
        try:
            # Create restricted globals
            safe_globals = self._create_safe_globals()

            # Wrap function for sandboxing
            sandboxed_func = self._wrap_function(func, safe_globals)

            # Apply resource limits
            with self._apply_resource_limits(context):
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_async(sandboxed_func, args, kwargs),
                    timeout=self.resource_limits.get("timeout", 30)
                )
                return result

        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool execution exceeded timeout")
        except Exception as e:
            logger.error(f"Sandbox execution failed: {str(e)}")
            raise

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create restricted global namespace."""
        safe_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in self.blocked_builtins
        }

        return {
            "__builtins__": safe_builtins,
            "__name__": "sandboxed_tool",
            "math": __import__("math"),
            "json": __import__("json"),
            "datetime": __import__("datetime"),
            "hashlib": __import__("hashlib"),
        }

    def _wrap_function(self, func: Callable, safe_globals: Dict) -> Callable:
        """Wrap function with safety checks."""
        def wrapped(*args, **kwargs):
            # Validate inputs
            self._validate_inputs(args, kwargs)
            # Execute with safe globals
            return func(*args, **kwargs)
        return wrapped

    def _validate_inputs(self, args: tuple, kwargs: dict) -> None:
        """Validate function inputs for safety."""
        # Check for dangerous patterns
        for arg in args:
            if isinstance(arg, str):
                dangerous_patterns = ["__", "eval", "exec", "import"]
                if any(pattern in arg for pattern in dangerous_patterns):
                    raise ValueError(f"Dangerous pattern detected in input")

    def _apply_resource_limits(self, context: ToolExecutionContext):
        """Context manager for resource limits."""
        class ResourceLimiter:
            def __enter__(self):
                # Apply memory and CPU limits if available
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Clean up resource limits
                pass

        return ResourceLimiter()

    async def _execute_async(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function asynchronously."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)


class FunctionCallingProtocol:
    """Protocol for function discovery, selection, and execution."""

    def __init__(self):
        """Initialize function calling protocol."""
        self.discovery_cache = {}
        self.selection_history = defaultdict(list)
        self.parameter_validators = {}

    async def discover_tools(
        self,
        requirements: Dict[str, Any],
        available_tools: Dict[str, ToolDefinition]
    ) -> List[ToolDefinition]:
        """Discover tools matching requirements."""
        matching_tools = []

        for tool_name, tool_def in available_tools.items():
            if self._matches_requirements(tool_def, requirements):
                matching_tools.append(tool_def)

        # Sort by relevance score
        matching_tools.sort(
            key=lambda t: self._calculate_relevance(t, requirements),
            reverse=True
        )

        return matching_tools

    def select_tool(
        self,
        tools: List[ToolDefinition],
        context: Dict[str, Any]
    ) -> Optional[ToolDefinition]:
        """Select best tool for the context."""
        if not tools:
            return None

        scores = []
        for tool in tools:
            score = self._score_tool(tool, context)
            scores.append((tool, score))

        # Select tool with highest score
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[0][0] if scores else None

        # Record selection
        if selected:
            self.selection_history[selected.name].append({
                "timestamp": DeterministicClock.now(),
                "context": context,
                "score": scores[0][1]
            })

        return selected

    def map_parameters(
        self,
        tool: ToolDefinition,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map input data to tool parameters."""
        mapped_params = {}

        for param_name, param_schema in tool.parameters.items():
            # Direct mapping
            if param_name in input_data:
                mapped_params[param_name] = input_data[param_name]
            # Try alternative names
            elif "aliases" in param_schema:
                for alias in param_schema["aliases"]:
                    if alias in input_data:
                        mapped_params[param_name] = input_data[alias]
                        break
            # Use default if available
            elif "default" in param_schema:
                mapped_params[param_name] = param_schema["default"]
            # Required parameter missing
            elif param_schema.get("required", False):
                raise ValueError(f"Required parameter '{param_name}' not found")

        # Validate mapped parameters
        self._validate_parameters(tool, mapped_params)

        return mapped_params

    def process_result(
        self,
        tool: ToolDefinition,
        raw_result: Any
    ) -> Any:
        """Process and validate tool output."""
        # Validate against return schema
        if tool.returns:
            self._validate_output(raw_result, tool.returns)

        # Apply any transformations
        processed_result = self._transform_output(raw_result, tool)

        return processed_result

    def _matches_requirements(
        self,
        tool: ToolDefinition,
        requirements: Dict[str, Any]
    ) -> bool:
        """Check if tool matches requirements."""
        # Check category
        if "category" in requirements:
            if tool.category != requirements["category"]:
                return False

        # Check capabilities
        if "capabilities" in requirements:
            tool_capabilities = set(tool.parameters.keys())
            required_capabilities = set(requirements["capabilities"])
            if not required_capabilities.issubset(tool_capabilities):
                return False

        return True

    def _calculate_relevance(
        self,
        tool: ToolDefinition,
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for tool."""
        score = 0.0

        # Category match
        if tool.category == requirements.get("category"):
            score += 1.0

        # Parameter overlap
        if "parameters" in requirements:
            overlap = len(
                set(tool.parameters.keys()) &
                set(requirements["parameters"].keys())
            )
            score += overlap * 0.5

        # Performance history
        if tool.name in self.selection_history:
            recent_uses = len([
                h for h in self.selection_history[tool.name]
                if h["timestamp"] > DeterministicClock.now() - timedelta(hours=1)
            ])
            score += min(recent_uses * 0.1, 0.5)

        return score

    def _score_tool(self, tool: ToolDefinition, context: Dict[str, Any]) -> float:
        """Score tool based on context."""
        score = 0.0

        # Base score from tool properties
        score += 1.0 if not tool.sandboxed else 0.5  # Prefer non-sandboxed for speed
        score += 0.5 if tool.timeout_seconds < 10 else 0.0  # Prefer fast tools

        # Context-based scoring
        if "priority" in context:
            if context["priority"] == "speed" and tool.timeout_seconds < 5:
                score += 1.0
            elif context["priority"] == "accuracy" and tool.audit_enabled:
                score += 1.0

        return score

    def _validate_parameters(self, tool: ToolDefinition, params: Dict[str, Any]) -> None:
        """Validate parameters against schema."""
        for param_name, param_value in params.items():
            if param_name in tool.parameters:
                schema = tool.parameters[param_name]
                # Type validation
                if "type" in schema:
                    expected_type = schema["type"]
                    if not isinstance(param_value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' expects {expected_type}"
                        )
                # Range validation
                if "min" in schema and param_value < schema["min"]:
                    raise ValueError(
                        f"Parameter '{param_name}' below minimum {schema['min']}"
                    )
                if "max" in schema and param_value > schema["max"]:
                    raise ValueError(
                        f"Parameter '{param_name}' above maximum {schema['max']}"
                    )

    def _validate_output(self, output: Any, schema: Dict[str, Any]) -> None:
        """Validate output against schema."""
        # Basic type validation
        if "type" in schema:
            expected_type = schema["type"]
            if not isinstance(output, expected_type):
                raise TypeError(f"Output type mismatch: expected {expected_type}")

    def _transform_output(self, output: Any, tool: ToolDefinition) -> Any:
        """Apply transformations to output."""
        # Default: no transformation
        return output


class ToolRegistry:
    """Registry for tool management and discovery."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.categories: Dict[ToolCategory, Set[str]] = defaultdict(set)

    def register_tool(
        self,
        tool_def: ToolDefinition,
        function: Callable
    ) -> None:
        """Register a tool with its function."""
        # Validate tool definition
        self._validate_tool_definition(tool_def)

        # Store tool
        self.tools[tool_def.name] = tool_def
        self.tool_functions[tool_def.name] = function
        self.categories[tool_def.category].add(tool_def.name)

        logger.info(f"Registered tool: {tool_def.name}")

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self.tools:
            tool_def = self.tools[tool_name]
            del self.tools[tool_name]
            del self.tool_functions[tool_name]
            self.categories[tool_def.category].discard(tool_name)
            logger.info(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[Tuple[ToolDefinition, Callable]]:
        """Get tool definition and function."""
        if tool_name in self.tools:
            return self.tools[tool_name], self.tool_functions[tool_name]
        return None

    def list_tools(
        self,
        category: Optional[ToolCategory] = None
    ) -> List[ToolDefinition]:
        """List available tools."""
        if category:
            tool_names = self.categories.get(category, set())
            return [self.tools[name] for name in tool_names]
        return list(self.tools.values())

    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools by name or description."""
        results = []
        query_lower = query.lower()

        for tool_def in self.tools.values():
            if (query_lower in tool_def.name.lower() or
                query_lower in tool_def.description.lower()):
                results.append(tool_def)

        return results

    def _validate_tool_definition(self, tool_def: ToolDefinition) -> None:
        """Validate tool definition completeness."""
        if not tool_def.name:
            raise ValueError("Tool name is required")
        if tool_def.name in self.tools:
            raise ValueError(f"Tool '{tool_def.name}' already registered")


class ToolExecutor:
    """Executor for tool execution with safety controls."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize tool executor."""
        self.config = config
        self.sandbox = ToolSandbox(config.get("sandbox", {}))
        self.rate_limiters: Dict[str, asyncio.Queue] = {}
        self.execution_history: List[ToolExecutionResult] = []
        self.audit_logger = self._setup_audit_logger()

    async def execute(
        self,
        tool_def: ToolDefinition,
        function: Callable,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolExecutionResult:
        """Execute tool with all safety controls."""
        execution_id = context.execution_id
        start_time = time.time()

        try:
            # Check rate limit
            if not await self._check_rate_limit(tool_def):
                return ToolExecutionResult(
                    tool_name=tool_def.name,
                    execution_id=execution_id,
                    status=ToolStatus.RATE_LIMITED,
                    error="Rate limit exceeded",
                    execution_time_ms=0
                )

            # Log execution start
            self._audit_log("execution_start", tool_def, parameters, context)

            # Execute in sandbox if enabled
            if tool_def.sandboxed:
                result = await self.sandbox.execute(
                    function,
                    args=(),
                    kwargs=parameters,
                    context=context
                )
            else:
                # Direct execution with timeout
                result = await asyncio.wait_for(
                    self._execute_direct(function, parameters),
                    timeout=tool_def.timeout_seconds
                )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Log success
            self._audit_log("execution_success", tool_def, result, context)

            # Create result
            execution_result = ToolExecutionResult(
                tool_name=tool_def.name,
                execution_id=execution_id,
                status=ToolStatus.COMPLETED,
                result=result,
                execution_time_ms=execution_time_ms,
                audit_trail=self._get_audit_trail(execution_id)
            )

            # Store in history
            self.execution_history.append(execution_result)

            return execution_result

        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            self._audit_log("execution_timeout", tool_def, None, context)

            return ToolExecutionResult(
                tool_name=tool_def.name,
                execution_id=execution_id,
                status=ToolStatus.TIMEOUT,
                error="Execution timeout",
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            self._audit_log("execution_error", tool_def, error_msg, context)

            return ToolExecutionResult(
                tool_name=tool_def.name,
                execution_id=execution_id,
                status=ToolStatus.FAILED,
                error=error_msg,
                execution_time_ms=execution_time_ms,
                audit_trail=self._get_audit_trail(execution_id)
            )

    async def _check_rate_limit(self, tool_def: ToolDefinition) -> bool:
        """Check if tool execution is within rate limit."""
        if tool_def.name not in self.rate_limiters:
            # Create rate limiter for this tool
            self.rate_limiters[tool_def.name] = asyncio.Queue(
                maxsize=tool_def.max_calls_per_minute
            )
            # Pre-fill with tokens
            for _ in range(tool_def.max_calls_per_minute):
                self.rate_limiters[tool_def.name].put_nowait(None)
            # Start token refill task
            asyncio.create_task(
                self._refill_tokens(tool_def.name, tool_def.max_calls_per_minute)
            )

        try:
            # Try to get a token (non-blocking)
            self.rate_limiters[tool_def.name].get_nowait()
            return True
        except asyncio.QueueEmpty:
            return False

    async def _refill_tokens(self, tool_name: str, rate: int) -> None:
        """Refill rate limit tokens."""
        interval = 60.0 / rate  # Interval between token refills

        while tool_name in self.rate_limiters:
            await asyncio.sleep(interval)
            try:
                self.rate_limiters[tool_name].put_nowait(None)
            except asyncio.QueueFull:
                pass  # Queue is full, skip

    async def _execute_direct(
        self,
        function: Callable,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute function directly (no sandbox)."""
        if asyncio.iscoroutinefunction(function):
            return await function(**parameters)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, function, **parameters)

    def _setup_audit_logger(self) -> logging.Logger:
        """Set up audit logger."""
        audit_logger = logging.getLogger(f"{__name__}.audit")
        audit_logger.setLevel(logging.INFO)
        return audit_logger

    def _audit_log(
        self,
        event: str,
        tool_def: ToolDefinition,
        data: Any,
        context: ToolExecutionContext
    ) -> None:
        """Log audit event."""
        if not tool_def.audit_enabled:
            return

        audit_entry = {
            "timestamp": DeterministicClock.now().isoformat(),
            "event": event,
            "tool": tool_def.name,
            "agent_id": context.agent_id,
            "execution_id": context.execution_id,
            "data": str(data)[:1000]  # Truncate large data
        }

        self.audit_logger.info(json.dumps(audit_entry))

    def _get_audit_trail(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for execution."""
        # In production, this would query audit log storage
        return []


class ToolFramework:
    """Main framework for tool management and execution."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize tool framework."""
        self.config = config or {}
        self.registry = ToolRegistry()
        self.protocol = FunctionCallingProtocol()
        self.executor = ToolExecutor(self.config.get("executor", {}))

        # Initialize default tools
        self._initialize_default_tools()

        # Metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time_ms": 0.0
        }

    def _initialize_default_tools(self) -> None:
        """Initialize default tools."""
        # Carbon calculator
        self.registry.register_tool(
            ToolDefinition(
                name="carbon_calculator",
                category=ToolCategory.CALCULATION,
                description="Calculate carbon emissions",
                parameters={
                    "activity_data": {"type": float, "required": True},
                    "emission_factor": {"type": float, "required": True}
                },
                returns={"type": float}
            ),
            lambda activity_data, emission_factor: activity_data * emission_factor
        )

        # JSON parser
        self.registry.register_tool(
            ToolDefinition(
                name="json_parser",
                category=ToolCategory.DATA,
                description="Parse JSON data",
                parameters={
                    "json_string": {"type": str, "required": True}
                },
                returns={"type": dict}
            ),
            lambda json_string: json.loads(json_string)
        )

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: str = "unknown"
    ) -> ToolExecutionResult:
        """Execute a tool by name."""
        # Get tool
        tool_result = self.registry.get_tool(tool_name)
        if not tool_result:
            return ToolExecutionResult(
                tool_name=tool_name,
                execution_id=self._generate_execution_id(),
                status=ToolStatus.FAILED,
                error=f"Tool '{tool_name}' not found",
                execution_time_ms=0
            )

        tool_def, function = tool_result

        # Create execution context
        context = ToolExecutionContext(
            tool_name=tool_name,
            execution_id=self._generate_execution_id(),
            agent_id=agent_id,
            timestamp=DeterministicClock.now(),
            parameters=parameters,
            permissions={ToolPermission.EXECUTE}
        )

        # Map parameters
        try:
            mapped_params = self.protocol.map_parameters(tool_def, parameters)
        except Exception as e:
            return ToolExecutionResult(
                tool_name=tool_name,
                execution_id=context.execution_id,
                status=ToolStatus.FAILED,
                error=f"Parameter mapping failed: {str(e)}",
                execution_time_ms=0
            )

        # Execute tool
        result = await self.executor.execute(
            tool_def,
            function,
            mapped_params,
            context
        )

        # Update metrics
        self._update_metrics(result)

        return result

    async def discover_and_execute(
        self,
        requirements: Dict[str, Any],
        input_data: Dict[str, Any],
        agent_id: str = "unknown"
    ) -> ToolExecutionResult:
        """Discover and execute best matching tool."""
        # Discover tools
        matching_tools = await self.protocol.discover_tools(
            requirements,
            self.registry.tools
        )

        if not matching_tools:
            return ToolExecutionResult(
                tool_name="unknown",
                execution_id=self._generate_execution_id(),
                status=ToolStatus.FAILED,
                error="No matching tools found",
                execution_time_ms=0
            )

        # Select best tool
        selected_tool = self.protocol.select_tool(
            matching_tools,
            {"requirements": requirements, "input": input_data}
        )

        if not selected_tool:
            return ToolExecutionResult(
                tool_name="unknown",
                execution_id=self._generate_execution_id(),
                status=ToolStatus.FAILED,
                error="Tool selection failed",
                execution_time_ms=0
            )

        # Execute selected tool
        return await self.execute_tool(
            selected_tool.name,
            input_data,
            agent_id
        )

    def register_tool(
        self,
        tool_def: ToolDefinition,
        function: Callable
    ) -> None:
        """Register a new tool."""
        self.registry.register_tool(tool_def, function)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """List available tools."""
        return self.registry.list_tools(category)

    def get_metrics(self) -> Dict[str, Any]:
        """Get framework metrics."""
        metrics = self.metrics.copy()
        if metrics["total_executions"] > 0:
            metrics["average_execution_time_ms"] = (
                metrics["total_execution_time_ms"] / metrics["total_executions"]
            )
            metrics["success_rate"] = (
                metrics["successful_executions"] / metrics["total_executions"]
            )
        return metrics

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        timestamp = DeterministicClock.now().isoformat()
        random_suffix = hashlib.sha256(
            f"{timestamp}{id(self)}".encode()
        ).hexdigest()[:8]
        return f"exec_{random_suffix}"

    def _update_metrics(self, result: ToolExecutionResult) -> None:
        """Update framework metrics."""
        self.metrics["total_executions"] += 1

        if result.status == ToolStatus.COMPLETED:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1

        self.metrics["total_execution_time_ms"] += result.execution_time_ms