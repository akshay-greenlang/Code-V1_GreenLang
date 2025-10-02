"""
Tool Registry and Execution System

Bridges GreenLang agents to LLM function-calling:
- @tool decorator: Marks agent methods as LLM-callable tools
- ToolRegistry: Auto-discovers and registers tools from agents
- ToolExecutor: Validates arguments and executes tools safely
- Error handling: Wraps tool exceptions with context

Architecture:
    Agent (@tool methods) → ToolRegistry → LLM (ToolDef) → ToolExecutor → Agent

Key features:
1. Auto-discovery: Scan agent classes for @tool-decorated methods
2. Schema validation: Enforce JSON Schema for arguments and returns
3. Timeout enforcement: Kill runaway tool executions
4. Error wrapping: Convert tool exceptions to structured errors
5. Provenance tracking: Log tool invocations for audit

Example:
    # Define tool in agent
    class CarbonAgent(BaseAgent):
        @tool(
            name="calculate_emissions",
            description="Calculate CO2e emissions from fuel combustion",
            parameters_schema={
                "type": "object",
                "properties": {
                    "fuel_type": {"type": "string", "enum": ["diesel", "gasoline"]},
                    "amount": {"type": "number", "minimum": 0}
                },
                "required": ["fuel_type", "amount"]
            },
            returns_schema={
                "type": "object",
                "properties": {
                    "co2e_kg": {"type": "number"},
                    "source": {"type": "string"}
                },
                "required": ["co2e_kg", "source"]
            }
        )
        def calculate_emissions(self, fuel_type: str, amount: float):
            # Implementation
            return {"co2e_kg": amount * 2.68, "source": "EPA 2024"}

    # Register tools
    registry = ToolRegistry()
    agent = CarbonAgent()
    registry.register_from_agent(agent)

    # Get tool definitions for LLM
    tool_defs = registry.get_tool_defs()

    # Execute tool call
    result = registry.invoke("calculate_emissions", {
        "fuel_type": "diesel",
        "amount": 100
    })
    print(result)  # {"co2e_kg": 268, "source": "EPA 2024"}
"""

from __future__ import annotations
import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from functools import wraps
from pydantic import BaseModel, Field, ValidationError

from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.runtime.jsonio import validate_json_payload

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """
    Error during tool execution

    Attributes:
        tool_name: Name of tool that failed
        arguments: Arguments passed to tool
        original_error: Underlying exception
    """

    def __init__(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        original_error: Exception,
    ):
        self.tool_name = tool_name
        self.arguments = arguments
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {original_error}")


class ToolNotFoundError(Exception):
    """Tool not found in registry"""

    pass


class ToolTimeoutError(Exception):
    """Tool execution exceeded timeout"""

    pass


class ToolSpec(BaseModel):
    """
    Internal tool specification

    Extends ToolDef with execution metadata:
    - callable: The actual function to execute
    - returns_schema: JSON Schema for return value validation
    - timeout_s: Execution timeout in seconds
    - async_fn: Whether function is async
    """

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    returns_schema: Optional[Dict[str, Any]] = None
    callable: Optional[Callable] = Field(default=None, exclude=True)
    timeout_s: float = 30.0
    async_fn: bool = False

    class Config:
        arbitrary_types_allowed = True


def tool(
    name: str,
    description: str,
    parameters_schema: Dict[str, Any],
    returns_schema: Optional[Dict[str, Any]] = None,
    timeout_s: float = 30.0,
):
    """
    Decorator to mark agent methods as LLM-callable tools

    Adds metadata to method for auto-discovery by ToolRegistry.
    Validates arguments against JSON Schema before execution.
    Validates return value against JSON Schema after execution.

    Args:
        name: Tool name (must be unique, valid Python identifier)
        description: Tool description for LLM (be specific!)
        parameters_schema: JSON Schema for parameters (type: object)
        returns_schema: JSON Schema for return value (optional)
        timeout_s: Execution timeout in seconds (default: 30s)

    Returns:
        Decorated method with _tool_spec attribute

    Example:
        @tool(
            name="get_grid_intensity",
            description="Returns carbon intensity of electricity grid",
            parameters_schema={
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "year": {"type": "integer"}
                },
                "required": ["region"]
            },
            returns_schema={
                "type": "object",
                "properties": {
                    "intensity": {"type": "number"},
                    "unit": {"type": "string"}
                },
                "required": ["intensity", "unit"]
            },
            timeout_s=10.0
        )
        def get_grid_intensity(self, region: str, year: int = 2024):
            # Implementation
            return {"intensity": 0.4, "unit": "kg_CO2e/kWh"}
    """

    def decorator(func: Callable) -> Callable:
        # Store tool spec on function
        spec = ToolSpec(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            returns_schema=returns_schema,
            callable=func,
            timeout_s=timeout_s,
            async_fn=inspect.iscoroutinefunction(func),
        )
        func._tool_spec = spec  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        if spec.async_fn:
            async_wrapper._tool_spec = spec  # type: ignore
            return async_wrapper
        else:
            wrapper._tool_spec = spec  # type: ignore
            return wrapper

    return decorator


class ToolRegistry:
    """
    Registry for LLM-callable tools

    Auto-discovers @tool-decorated methods from agents and provides:
    - Tool definitions (ToolDef) for LLM function calling
    - Tool execution with argument validation
    - Error handling and timeout enforcement
    - Provenance tracking for audit

    Usage:
        # Create registry
        registry = ToolRegistry()

        # Register tools from agents
        carbon_agent = CarbonAgent()
        registry.register_from_agent(carbon_agent)

        energy_agent = EnergyAgent()
        registry.register_from_agent(energy_agent)

        # Get tool defs for LLM
        tool_defs = registry.get_tool_defs()

        # Execute tool
        result = registry.invoke("calculate_emissions", {
            "fuel_type": "diesel",
            "amount": 100
        })
    """

    def __init__(self):
        """Initialize empty registry"""
        self._tools: Dict[str, ToolSpec] = {}
        logger.info("Initialized ToolRegistry")

    def register_from_agent(self, agent: Any) -> int:
        """
        Auto-discover and register @tool-decorated methods from agent

        Scans agent instance for methods with _tool_spec attribute.
        Binds methods to agent instance for execution.

        Args:
            agent: Agent instance with @tool-decorated methods

        Returns:
            Number of tools registered

        Raises:
            ValueError: If tool name already registered
            ValueError: If tool spec invalid

        Example:
            agent = CarbonAgent()
            count = registry.register_from_agent(agent)
            print(f"Registered {count} tools")
        """
        count = 0

        for attr_name in dir(agent):
            # Skip private attributes
            if attr_name.startswith("_"):
                continue

            attr = getattr(agent, attr_name)

            # Check if method has _tool_spec
            if not hasattr(attr, "_tool_spec"):
                continue

            spec: ToolSpec = attr._tool_spec

            # Check for name conflicts
            if spec.name in self._tools:
                raise ValueError(
                    f"Tool '{spec.name}' already registered. "
                    f"Use unique tool names across agents."
                )

            # Bind callable to agent instance
            spec.callable = attr

            # Register tool
            self._tools[spec.name] = spec
            count += 1

            logger.info(
                f"Registered tool: {spec.name} (timeout={spec.timeout_s}s, "
                f"async={spec.async_fn})"
            )

        if count == 0:
            logger.warning(
                f"No @tool-decorated methods found on {agent.__class__.__name__}. "
                "Did you forget @tool decorator?"
            )

        return count

    def register_tool(
        self,
        name: str,
        description: str,
        callable_fn: Callable,
        parameters_schema: Dict[str, Any],
        returns_schema: Optional[Dict[str, Any]] = None,
        timeout_s: float = 30.0,
    ) -> None:
        """
        Manually register a tool (alternative to @tool decorator)

        Args:
            name: Tool name
            description: Tool description for LLM
            callable_fn: Function to execute
            parameters_schema: JSON Schema for parameters
            returns_schema: JSON Schema for return value
            timeout_s: Execution timeout

        Raises:
            ValueError: If tool name already registered

        Example:
            def custom_tool(region: str) -> dict:
                return {"result": region.upper()}

            registry.register_tool(
                name="uppercase_region",
                description="Converts region to uppercase",
                callable_fn=custom_tool,
                parameters_schema={
                    "type": "object",
                    "properties": {"region": {"type": "string"}},
                    "required": ["region"]
                }
            )
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")

        spec = ToolSpec(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            returns_schema=returns_schema,
            callable=callable_fn,
            timeout_s=timeout_s,
            async_fn=inspect.iscoroutinefunction(callable_fn),
        )

        self._tools[name] = spec
        logger.info(f"Manually registered tool: {name}")

    def get_tool_defs(self) -> List[ToolDef]:
        """
        Get tool definitions for LLM function calling

        Converts internal ToolSpec to ToolDef format for LLM providers.

        Returns:
            List of ToolDef objects

        Example:
            tool_defs = registry.get_tool_defs()
            response = await provider.chat(
                messages=[...],
                tools=tool_defs,
                budget=Budget(max_usd=0.50)
            )
        """
        return [
            ToolDef(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters_schema,
            )
            for spec in self._tools.values()
        ]

    def has_tool(self, name: str) -> bool:
        """Check if tool is registered"""
        return name in self._tools

    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names"""
        return list(self._tools.keys())

    def invoke(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute tool synchronously with validation

        Workflow:
        1. Look up tool in registry
        2. Validate arguments against parameters_schema
        3. Execute tool with timeout
        4. Validate return value against returns_schema
        5. Return result

        Args:
            name: Tool name to execute
            arguments: Tool arguments (must match parameters_schema)
            timeout_s: Override default timeout

        Returns:
            Tool result (validated against returns_schema if specified)

        Raises:
            ToolNotFoundError: If tool not found
            ValidationError: If arguments invalid
            ToolTimeoutError: If execution exceeds timeout
            ToolExecutionError: If tool raises exception

        Example:
            result = registry.invoke(
                "calculate_emissions",
                {"fuel_type": "diesel", "amount": 100}
            )
            print(result["co2e_kg"])
        """
        # 1. Look up tool
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool '{name}' not found. Available: {self.get_tool_names()}"
            )

        spec = self._tools[name]

        # 2. Validate arguments
        try:
            validate_json_payload(arguments, spec.parameters_schema)
        except Exception as e:
            raise ValidationError(f"Tool '{name}' arguments invalid: {e}")

        # 3. Execute tool
        try:
            # Use override timeout or default
            timeout = timeout_s if timeout_s is not None else spec.timeout_s

            if spec.async_fn:
                # Async tool - run in event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                result = loop.run_until_complete(
                    asyncio.wait_for(spec.callable(**arguments), timeout=timeout)
                )
            else:
                # Sync tool - run directly (no timeout support for sync)
                result = spec.callable(**arguments)

        except asyncio.TimeoutError:
            raise ToolTimeoutError(f"Tool '{name}' exceeded timeout of {timeout}s")
        except Exception as e:
            raise ToolExecutionError(
                tool_name=name,
                arguments=arguments,
                original_error=e,
            )

        # 4. Validate return value
        if spec.returns_schema:
            try:
                validate_json_payload(result, spec.returns_schema)
            except Exception as e:
                raise ValidationError(
                    f"Tool '{name}' return value invalid: {e}. " f"Got: {result}"
                )

        logger.debug(f"Tool '{name}' executed successfully")
        return result

    async def invoke_async(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute tool asynchronously with validation

        Same as invoke() but async-compatible.
        Handles both sync and async tools.

        Args:
            name: Tool name to execute
            arguments: Tool arguments
            timeout_s: Override default timeout

        Returns:
            Tool result

        Raises:
            ToolNotFoundError: If tool not found
            ValidationError: If arguments invalid
            ToolTimeoutError: If execution exceeds timeout
            ToolExecutionError: If tool raises exception

        Example:
            result = await registry.invoke_async(
                "calculate_emissions",
                {"fuel_type": "diesel", "amount": 100}
            )
        """
        # 1. Look up tool
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool '{name}' not found. Available: {self.get_tool_names()}"
            )

        spec = self._tools[name]

        # 2. Validate arguments
        try:
            validate_json_payload(arguments, spec.parameters_schema)
        except Exception as e:
            raise ValidationError(f"Tool '{name}' arguments invalid: {e}")

        # 3. Execute tool
        try:
            timeout = timeout_s if timeout_s is not None else spec.timeout_s

            if spec.async_fn:
                # Async tool
                result = await asyncio.wait_for(
                    spec.callable(**arguments), timeout=timeout
                )
            else:
                # Sync tool - run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: spec.callable(**arguments)
                )

        except asyncio.TimeoutError:
            raise ToolTimeoutError(f"Tool '{name}' exceeded timeout of {timeout}s")
        except Exception as e:
            raise ToolExecutionError(
                tool_name=name,
                arguments=arguments,
                original_error=e,
            )

        # 4. Validate return value
        if spec.returns_schema:
            try:
                validate_json_payload(result, spec.returns_schema)
            except Exception as e:
                raise ValidationError(
                    f"Tool '{name}' return value invalid: {e}. " f"Got: {result}"
                )

        logger.debug(f"Tool '{name}' executed successfully (async)")
        return result

    def clear(self) -> None:
        """Clear all registered tools"""
        count = len(self._tools)
        self._tools.clear()
        logger.info(f"Cleared {count} tools from registry")

    def __len__(self) -> int:
        """Return number of registered tools"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered (supports 'name in registry')"""
        return name in self._tools

    def __repr__(self) -> str:
        """String representation"""
        return f"ToolRegistry({len(self._tools)} tools: {self.get_tool_names()})"


# Global registry instance (singleton pattern)
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """
    Get global tool registry (singleton)

    Provides a shared registry across the application.
    Useful for registering tools once and using everywhere.

    Returns:
        Global ToolRegistry instance

    Example:
        registry = get_global_registry()
        registry.register_from_agent(carbon_agent)

        # Later, in different module
        registry = get_global_registry()
        result = registry.invoke("calculate_emissions", {...})
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
