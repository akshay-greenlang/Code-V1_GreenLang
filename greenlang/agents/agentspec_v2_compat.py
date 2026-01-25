# -*- coding: utf-8 -*-
"""
AgentSpec v2 Backward Compatibility Wrapper
============================================

This module provides compatibility wrappers to make existing agents
AgentSpec v2 compliant without modifying their source code.

Key Features:
- Zero-code migration: Wrap existing agents with v2 compliance
- Full lifecycle support: Maps old run() to new lifecycle
- Schema validation: Adds pack.yaml validation on top of existing logic
- Citation preservation: Maintains existing citation tracking
- Drop-in replacement: Same interface, enhanced capabilities

Usage Example:
    >>> from greenlang.agents.fuel_agent_ai import FuelAgentAI
    >>> from greenlang.agents.agentspec_v2_compat import wrap_agent_v2
    >>>
    >>> # Original agent
    >>> original_agent = FuelAgentAI()
    >>>
    >>> # Wrap with v2 compliance
    >>> v2_agent = wrap_agent_v2(
    ...     original_agent,
    ...     pack_path=Path("packs/fuel_ai")
    ... )
    >>>
    >>> # Use exactly like before, but now with v2 validation!
    >>> result = v2_agent.run({"fuel_type": "natural_gas", "amount": 1000})

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from pathlib import Path
from typing import Any, Dict, Generic, TypeVar

from greenlang.agents.agentspec_v2_base import (
    AgentExecutionContext,
    AgentSpecV2Base,
)
from greenlang.types import AgentResult

InT = TypeVar("InT")
OutT = TypeVar("OutT")


# ==============================================================================
# Backward Compatibility Wrapper
# ==============================================================================

class AgentSpecV2Wrapper(AgentSpecV2Base[InT, OutT]):
    """
    Wrapper that makes any existing agent AgentSpec v2 compliant.

    This wrapper:
    - Takes an existing agent that implements Agent[InT, OutT] protocol
    - Adds AgentSpec v2 lifecycle on top
    - Validates input/output against pack.yaml (if provided)
    - Preserves all existing functionality
    - Adds zero overhead when validation disabled

    The wrapped agent's run() method becomes the execute_impl() in the
    new lifecycle.

    Architecture:
        ┌─────────────────────────────┐
        │  AgentSpecV2Wrapper         │
        │  - Lifecycle management     │
        │  - Schema validation        │
        └────────────┬────────────────┘
                     │ wraps
        ┌────────────▼────────────────┐
        │  Existing Agent             │
        │  - run() method             │
        │  - validate() method        │
        └─────────────────────────────┘
    """

    def __init__(
        self,
        delegate_agent: Any,  # The existing agent to wrap
        pack_path: Path | None = None,
        enable_validation: bool = True,
        enable_metrics: bool = True,
        enable_citations: bool = True,
    ):
        """
        Initialize wrapper around existing agent.

        Args:
            delegate_agent: Existing agent implementing Agent[InT, OutT]
            pack_path: Path to pack.yaml for v2 validation (optional)
            enable_validation: Enable schema validation
            enable_metrics: Enable metrics collection
            enable_citations: Enable citation tracking
        """
        self.delegate = delegate_agent

        # Extract agent_id from delegate if available
        agent_id = getattr(delegate_agent, "agent_id", None) or delegate_agent.__class__.__name__

        # Initialize base class
        super().__init__(
            pack_path=pack_path,
            agent_id=agent_id,
            enable_metrics=enable_metrics,
            enable_citations=enable_citations,
            enable_validation=enable_validation,
        )

    def execute_impl(self, validated_input: InT, context: AgentExecutionContext) -> OutT:
        """
        Execute by delegating to the wrapped agent's run() method.

        Args:
            validated_input: Input that has passed v2 validation
            context: Execution context

        Returns:
            Agent output
        """
        # Delegate to existing agent's run() method
        result = self.delegate.run(validated_input)

        # Extract data from result
        if isinstance(result, AgentResult):
            if not result.success:
                raise RuntimeError(result.error or "Agent execution failed")

            output = result.data

            # Capture citations from result if present
            if self.enable_citations and "citations" in output:
                context.citations = output["citations"]

            return output
        else:
            # If agent returns raw data instead of AgentResult
            return result

    def validate_input_impl(self, input_data: InT, context: AgentExecutionContext) -> InT:
        """
        Custom validation by delegating to agent's validate() method if available.

        Args:
            input_data: Input to validate
            context: Execution context

        Returns:
            Validated input
        """
        # Delegate to existing agent's validate() method if it exists
        if hasattr(self.delegate, "validate"):
            is_valid = self.delegate.validate(input_data)
            if not is_valid:
                raise ValueError(f"Agent {self.agent_id} validate() returned False")

        return input_data


# ==============================================================================
# Convenience Factory Function
# ==============================================================================

def wrap_agent_v2(
    agent: Any,
    pack_path: Path | None = None,
    enable_validation: bool = True,
    enable_metrics: bool = True,
    enable_citations: bool = True,
) -> AgentSpecV2Wrapper:
    """
    Wrap an existing agent with AgentSpec v2 compliance.

    This is the main entry point for backward compatibility. It takes
    any existing agent and makes it AgentSpec v2 compliant by adding:
    - Lifecycle management
    - Schema validation (if pack_path provided)
    - Enhanced error handling
    - Metrics collection
    - Citation tracking

    Args:
        agent: Existing agent to wrap (must have run() method)
        pack_path: Optional path to pack.yaml for validation
        enable_validation: Enable schema validation
        enable_metrics: Enable metrics collection
        enable_citations: Enable citation tracking

    Returns:
        AgentSpec v2 compliant wrapper

    Example:
        >>> from greenlang.agents.fuel_agent_ai import FuelAgentAI
        >>> original = FuelAgentAI()
        >>> wrapped = wrap_agent_v2(original, pack_path=Path("packs/fuel_ai"))
        >>> result = wrapped.run({"fuel_type": "natural_gas", "amount": 1000})
    """
    if not hasattr(agent, "run"):
        raise TypeError(
            f"Agent {agent.__class__.__name__} must have a run() method "
            "to be wrapped with AgentSpec v2"
        )

    return AgentSpecV2Wrapper(
        delegate_agent=agent,
        pack_path=pack_path,
        enable_validation=enable_validation,
        enable_metrics=enable_metrics,
        enable_citations=enable_citations,
    )


# ==============================================================================
# Decorator for Easy Migration
# ==============================================================================

def agentspec_v2(
    pack_path: Path | None = None,
    enable_validation: bool = True,
    enable_metrics: bool = True,
    enable_citations: bool = True,
):
    """
    Decorator to make a class AgentSpec v2 compliant.

    This decorator wraps the class's __init__ to return an
    AgentSpecV2Wrapper instance instead.

    Usage:
        >>> @agentspec_v2(pack_path=Path("packs/my_agent"))
        ... class MyAgent:
        ...     def run(self, payload):
        ...         return AgentResult(success=True, data={"result": 42})

    Args:
        pack_path: Path to pack.yaml
        enable_validation: Enable schema validation
        enable_metrics: Enable metrics collection
        enable_citations: Enable citation tracking

    Returns:
        Decorated class that returns wrapped instances
    """
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Call original __init__
            original_init(self, *args, **kwargs)

            # Wrap instance
            wrapped = wrap_agent_v2(
                self,
                pack_path=pack_path,
                enable_validation=enable_validation,
                enable_metrics=enable_metrics,
                enable_citations=enable_citations,
            )

            # Replace instance with wrapped version
            self.__class__ = wrapped.__class__
            self.__dict__ = wrapped.__dict__

        cls.__init__ = new_init
        return cls

    return decorator


# ==============================================================================
# Migration Helpers
# ==============================================================================

def _python_type_to_dtype(type_hint: Any) -> str:
    """Convert Python type hint to AgentSpec dtype."""
    import typing
    type_str = str(type_hint).lower()

    if type_hint is None or type_hint == type(None):
        return "null"
    elif type_hint in (int, "int"):
        return "integer"
    elif type_hint in (float, "float"):
        return "number"
    elif type_hint in (str, "str"):
        return "string"
    elif type_hint in (bool, "bool"):
        return "boolean"
    elif "list" in type_str or "array" in type_str:
        return "array"
    elif "dict" in type_str or "mapping" in type_str:
        return "object"
    else:
        return "string"  # Default to string for unknown types


def _extract_inputs_from_signature(sig) -> dict:
    """Extract input field definitions from method signature."""
    import inspect
    inputs = {}

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        dtype = "string"
        if param.annotation != inspect.Parameter.empty:
            dtype = _python_type_to_dtype(param.annotation)

        field_def = {
            "dtype": dtype,
            "unit": "1",
            "description": f"Input parameter: {name}",
            "required": param.default == inspect.Parameter.empty,
        }

        if param.default != inspect.Parameter.empty:
            field_def["default"] = param.default

        inputs[name] = field_def

    return inputs if inputs else {"_empty": {"dtype": "null", "description": "No inputs required"}}


def _extract_outputs_from_signature(sig, method) -> dict:
    """Extract output field definitions from return annotation and docstring."""
    import inspect

    outputs = {}
    return_annotation = sig.return_annotation

    if return_annotation != inspect.Signature.empty:
        dtype = _python_type_to_dtype(return_annotation)
        outputs["result"] = {
            "dtype": dtype,
            "unit": "1",
            "description": f"Return value of type {return_annotation}",
        }
    else:
        # Default output for methods without return annotation
        outputs["result"] = {
            "dtype": "object",
            "unit": "1",
            "description": "Agent execution result",
        }

    return outputs


def create_pack_yaml_for_agent(
    agent: Any,
    output_path: Path,
    agent_id: str,
    version: str = "1.0.0",
    summary: str | None = None,
) -> None:
    """
    Generate a basic pack.yaml for an existing agent.

    This helper analyzes an agent's run() method signature and creates
    a minimal pack.yaml that can be refined later.

    Args:
        agent: Agent to analyze
        output_path: Where to write pack.yaml
        agent_id: Agent identifier (e.g., "fuel-agent-ai")
        version: Version string
        summary: Agent description

    Example:
        >>> from greenlang.agents.fuel_agent_ai import FuelAgentAI
        >>> create_pack_yaml_for_agent(
        ...     FuelAgentAI(),
        ...     Path("packs/fuel_ai/pack.yaml"),
        ...     "fuel-agent-ai",
        ...     "1.0.0",
        ...     "AI-powered fuel emissions calculator"
        ... )
    """
    import inspect
    import yaml

    # Get run() method signature
    run_method = getattr(agent, "run", None)
    if not run_method:
        raise ValueError(f"Agent {agent.__class__.__name__} has no run() method")

    sig = inspect.signature(run_method)

    # Basic pack structure
    pack_data = {
        "schema_version": "2.0.0",
        "id": agent_id,
        "name": agent.__class__.__name__,
        "version": version,
        "summary": summary or f"AgentSpec v2 pack for {agent.__class__.__name__}",
        "metadata": {
            "tags": ["emissions", "greenlang"],
            "owners": ["greenlang-team"],
            "license": "MIT",
        },
        "compute": {
            "entrypoint": f"python://greenlang.agents.{agent.__class__.__module__}:{agent.__class__.__name__}.run",
            "deterministic": True,
            "timeout_seconds": 30,
            "memory_limit_mb": 512,
            "inputs": _extract_inputs_from_signature(sig),
            "outputs": _extract_outputs_from_signature(sig, run_method),
            "factors": [],
        },
        "provenance": {
            "pin_ef": True,
            "gwp_set": "AR6GWP100",
            "record": ["seed", "tool_calls", "ef_cids"],
        },
    }

    # Write pack.yaml
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(pack_data, f, default_flow_style=False, sort_keys=False)

    print(f"Created pack.yaml at {output_path}")
    print(f"Note: Review generated input/output fields for accuracy")
