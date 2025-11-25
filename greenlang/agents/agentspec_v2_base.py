# -*- coding: utf-8 -*-
"""
GreenLang AgentSpec v2 Foundation
==================================

This module provides the unified base class for AgentSpec v2 compliant agents.

Design Goals:
- Generic typing: Agent[Input, Output] with type safety
- Standard lifecycle: initialize → validate → execute → finalize
- Schema validation: Automated input/output checking against pack.yaml
- Backward compatibility: Wraps existing agents without code changes
- Citation integration: Built-in citation tracking
- Production ready: Comprehensive error handling and logging

Architecture:
    ┌─────────────────────────────────────┐
    │   AgentSpecV2Base[InT, OutT]        │
    │   - Lifecycle management             │
    │   - Schema validation                │
    │   - Citation tracking                │
    └──────────────┬──────────────────────┘
                   │ extends
    ┌──────────────▼──────────────────────┐
    │   Concrete Agents                    │
    │   - FuelAgentAI                      │
    │   - CarbonAgentAI                    │
    │   - GridFactorAgentAI                │
    └──────────────────────────────────────┘

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Callable

import yaml
from pydantic import BaseModel, Field, ValidationError

from greenlang.specs.agentspec_v2 import AgentSpecV2
from greenlang.specs.errors import GLValidationError, GLVErr
from greenlang.types import AgentResult
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


# ==============================================================================
# Type Variables for Generic Agent Base Class
# ==============================================================================

InT = TypeVar("InT")  # Input type (Dict, TypedDict, or Pydantic model)
OutT = TypeVar("OutT")  # Output type (Dict, TypedDict, or Pydantic model)


# ==============================================================================
# AgentSpec v2 Lifecycle States
# ==============================================================================

class AgentLifecycleState:
    """Agent lifecycle state machine."""

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    VALIDATING = "validating"
    EXECUTING = "executing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


# ==============================================================================
# AgentSpec v2 Execution Context
# ==============================================================================

class AgentExecutionContext(BaseModel):
    """Context object passed through agent lifecycle."""

    execution_id: str = Field(..., description="Unique execution identifier")
    start_time: datetime = Field(default_factory=datetime.now)
    state: str = Field(default=AgentLifecycleState.UNINITIALIZED)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Any] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ==============================================================================
# AgentSpec v2 Base Class
# ==============================================================================

class AgentSpecV2Base(ABC, Generic[InT, OutT]):
    """
    Unified base class for AgentSpec v2 compliant agents.

    This class provides:
    - Standard lifecycle methods (initialize, validate, execute, finalize)
    - Automated schema validation against pack.yaml
    - Citation tracking integration
    - Comprehensive error handling
    - Metrics collection
    - Backward compatibility with existing Agent[InT, OutT] protocol

    Usage Example:
        >>> class MyAgent(AgentSpecV2Base[MyInput, MyOutput]):
        ...     def execute_impl(self, validated_input: MyInput, context: AgentExecutionContext) -> MyOutput:
        ...         # Your agent logic here
        ...         return MyOutput(result="success")
        ...
        >>> agent = MyAgent()
        >>> result = agent.run({"param": "value"})

    Lifecycle Flow:
        1. initialize() - Setup resources, load pack.yaml
        2. validate_input() - Schema validation against pack.yaml
        3. execute_impl() - Core agent logic (implemented by subclass)
        4. validate_output() - Schema validation of results
        5. finalize() - Cleanup resources, prepare final result
    """

    def __init__(
        self,
        pack_path: Optional[Path] = None,
        agent_id: Optional[str] = None,
        enable_metrics: bool = True,
        enable_citations: bool = True,
        enable_validation: bool = True,
    ):
        """
        Initialize AgentSpec v2 base.

        Args:
            pack_path: Path to pack directory containing pack.yaml (optional)
            agent_id: Agent identifier (auto-detected if pack_path provided)
            enable_metrics: Enable execution metrics collection
            enable_citations: Enable citation tracking
            enable_validation: Enable input/output schema validation
        """
        self.pack_path = pack_path
        self.agent_id = agent_id or self.__class__.__name__
        self.enable_metrics = enable_metrics
        self.enable_citations = enable_citations
        self.enable_validation = enable_validation

        # Spec and config
        self.spec: Optional[AgentSpecV2] = None
        self._lifecycle_hooks: Dict[str, List[Callable]] = {
            "pre_initialize": [],
            "post_initialize": [],
            "pre_validate": [],
            "post_validate": [],
            "pre_execute": [],
            "post_execute": [],
            "pre_finalize": [],
            "post_finalize": [],
        }

        # Execution state
        self._state = AgentLifecycleState.UNINITIALIZED
        self._execution_count = 0
        self._total_execution_time_ms = 0.0

        # Logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Run initialization
        self.initialize()

    # ==========================================================================
    # Lifecycle Methods (Public API)
    # ==========================================================================

    def initialize(self) -> None:
        """
        Initialize agent resources and load pack.yaml.

        This method:
        1. Loads pack.yaml if pack_path provided
        2. Validates pack against AgentSpec v2 schema
        3. Sets up agent configuration
        4. Calls initialize_impl() for custom initialization

        Override initialize_impl() for custom initialization logic.
        """
        self._run_hooks("pre_initialize")

        try:
            # Load pack.yaml if provided
            if self.pack_path:
                self._load_pack_yaml()

            # Call custom initialization
            self.initialize_impl()

            self._state = AgentLifecycleState.INITIALIZED
            self.logger.info(f"{self.agent_id} initialized successfully")

        except Exception as e:
            self._state = AgentLifecycleState.FAILED
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            raise

        self._run_hooks("post_initialize")

    def validate_input(self, input_data: InT, context: AgentExecutionContext) -> InT:
        """
        Validate input data against AgentSpec v2 schema.

        This method:
        1. Checks required fields are present
        2. Validates data types match schema
        3. Checks constraints (ge, le, enum, etc.)
        4. Calls validate_input_impl() for custom validation

        Args:
            input_data: Input data to validate
            context: Execution context

        Returns:
            Validated input data (possibly transformed)

        Raises:
            GLValidationError: If validation fails
        """
        self._run_hooks("pre_validate")
        context.state = AgentLifecycleState.VALIDATING

        try:
            # Schema validation (if enabled and spec available)
            if self.enable_validation and self.spec:
                self._validate_input_schema(input_data)

            # Custom validation
            validated_input = self.validate_input_impl(input_data, context)

            self.logger.debug(f"Input validation passed for {self.agent_id}")
            return validated_input

        except Exception as e:
            context.errors.append(f"Input validation failed: {str(e)}")
            context.state = AgentLifecycleState.FAILED
            raise

        finally:
            self._run_hooks("post_validate")

    def execute(self, validated_input: InT, context: AgentExecutionContext) -> OutT:
        """
        Execute agent logic.

        This method:
        1. Calls execute_impl() with validated input
        2. Tracks execution time
        3. Handles errors and logging

        Args:
            validated_input: Validated input data
            context: Execution context

        Returns:
            Agent output

        Raises:
            Exception: If execution fails
        """
        self._run_hooks("pre_execute")
        context.state = AgentLifecycleState.EXECUTING

        start_time = time.time()

        try:
            # Execute agent logic
            output = self.execute_impl(validated_input, context)

            # Track metrics
            if self.enable_metrics:
                execution_time_ms = (time.time() - start_time) * 1000
                self._total_execution_time_ms += execution_time_ms
                context.metadata["execution_time_ms"] = execution_time_ms

            self.logger.info(
                f"{self.agent_id} executed successfully "
                f"(took {context.metadata.get('execution_time_ms', 0):.2f}ms)"
            )

            return output

        except Exception as e:
            context.errors.append(f"Execution failed: {str(e)}")
            context.state = AgentLifecycleState.FAILED
            self.logger.error(f"Execution failed: {e}", exc_info=True)
            raise

        finally:
            self._run_hooks("post_execute")

    def validate_output(self, output: OutT, context: AgentExecutionContext) -> OutT:
        """
        Validate output data against AgentSpec v2 schema.

        Args:
            output: Output data to validate
            context: Execution context

        Returns:
            Validated output

        Raises:
            GLValidationError: If validation fails
        """
        try:
            # Schema validation (if enabled and spec available)
            if self.enable_validation and self.spec:
                self._validate_output_schema(output)

            # Custom output validation
            validated_output = self.validate_output_impl(output, context)

            self.logger.debug(f"Output validation passed for {self.agent_id}")
            return validated_output

        except Exception as e:
            context.errors.append(f"Output validation failed: {str(e)}")
            context.state = AgentLifecycleState.FAILED
            raise

    def finalize(self, result: AgentResult[OutT], context: AgentExecutionContext) -> AgentResult[OutT]:
        """
        Finalize execution and prepare result.

        This method:
        1. Adds citations to result (if enabled)
        2. Adds execution metadata
        3. Calls finalize_impl() for custom finalization
        4. Cleans up resources

        Args:
            result: Agent result to finalize
            context: Execution context

        Returns:
            Finalized result
        """
        self._run_hooks("pre_finalize")
        context.state = AgentLifecycleState.FINALIZING

        try:
            # Add citations to result
            if self.enable_citations and context.citations:
                result.data["citations"] = context.citations

            # Add execution metadata
            result.metadata.update({
                "agent_id": self.agent_id,
                "execution_id": context.execution_id,
                "execution_time_ms": context.metadata.get("execution_time_ms", 0),
                "lifecycle_state": context.state,
            })

            # Custom finalization
            result = self.finalize_impl(result, context)

            context.state = AgentLifecycleState.COMPLETED
            self.logger.info(f"{self.agent_id} finalized successfully")

            return result

        except Exception as e:
            context.errors.append(f"Finalization failed: {str(e)}")
            context.state = AgentLifecycleState.FAILED
            raise

        finally:
            self._run_hooks("post_finalize")

    def run(self, payload: InT) -> AgentResult[OutT]:
        """
        Execute complete agent lifecycle.

        This is the main entry point that orchestrates:
        1. Initialize (if needed)
        2. Validate input
        3. Execute
        4. Validate output
        5. Finalize

        Args:
            payload: Input data conforming to InT type

        Returns:
            AgentResult with output data and metadata
        """
        import uuid

        # Create execution context
        context = AgentExecutionContext(
            execution_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            start_time=DeterministicClock.now(),
        )

        self._execution_count += 1

        try:
            # Ensure initialized
            if self._state == AgentLifecycleState.UNINITIALIZED:
                self.initialize()

            # Validate input
            validated_input = self.validate_input(payload, context)

            # Execute
            output = self.execute(validated_input, context)

            # Validate output
            validated_output = self.validate_output(output, context)

            # Create result
            result = AgentResult(
                success=True,
                data=validated_output if isinstance(validated_output, dict) else validated_output.__dict__,
                timestamp=DeterministicClock.now(),
            )

            # Finalize
            result = self.finalize(result, context)

            return result

        except GLValidationError as e:
            # Handle validation errors
            self.logger.error(f"Validation error: {e.message}")
            return AgentResult(
                success=False,
                error=f"Validation error ({e.code}): {e.message}",
                metadata={"validation_path": e.path, "code": e.code},
                timestamp=DeterministicClock.now(),
            )

        except Exception as e:
            # Handle other errors
            self.logger.error(f"Agent execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"errors": context.errors, "state": context.state},
                timestamp=DeterministicClock.now(),
            )

    # ==========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # ==========================================================================

    @abstractmethod
    def execute_impl(self, validated_input: InT, context: AgentExecutionContext) -> OutT:
        """
        Core agent logic - MUST be implemented by subclasses.

        Args:
            validated_input: Input data that has passed validation
            context: Execution context with metadata, citations, etc.

        Returns:
            Agent output conforming to OutT type
        """
        pass

    # ==========================================================================
    # Optional Override Methods
    # ==========================================================================

    def initialize_impl(self) -> None:
        """Custom initialization logic. Override if needed."""
        pass

    def validate_input_impl(self, input_data: InT, context: AgentExecutionContext) -> InT:
        """Custom input validation logic. Override if needed."""
        return input_data

    def validate_output_impl(self, output: OutT, context: AgentExecutionContext) -> OutT:
        """Custom output validation logic. Override if needed."""
        return output

    def finalize_impl(self, result: AgentResult[OutT], context: AgentExecutionContext) -> AgentResult[OutT]:
        """Custom finalization logic. Override if needed."""
        return result

    # ==========================================================================
    # Lifecycle Hooks
    # ==========================================================================

    def add_lifecycle_hook(self, hook_name: str, callback: Callable) -> None:
        """Add a lifecycle hook callback."""
        if hook_name in self._lifecycle_hooks:
            self._lifecycle_hooks[hook_name].append(callback)
        else:
            raise ValueError(f"Unknown hook: {hook_name}")

    def _run_hooks(self, hook_name: str) -> None:
        """Run all registered hooks for a lifecycle event."""
        for callback in self._lifecycle_hooks.get(hook_name, []):
            try:
                callback(self)
            except Exception as e:
                self.logger.warning(f"Hook {hook_name} failed: {e}")

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _load_pack_yaml(self) -> None:
        """Load and validate pack.yaml."""
        if not self.pack_path:
            return

        pack_yaml_path = self.pack_path / "pack.yaml"
        if not pack_yaml_path.exists():
            raise FileNotFoundError(f"pack.yaml not found at {pack_yaml_path}")

        with open(pack_yaml_path, "r") as f:
            pack_data = yaml.safe_load(f)

        # Validate against AgentSpec v2
        try:
            self.spec = AgentSpecV2(**pack_data)
            self.agent_id = self.spec.id
            self.logger.info(f"Loaded pack.yaml for {self.agent_id} (v{self.spec.version})")
        except ValidationError as e:
            raise GLValidationError(
                GLVErr.AI_SCHEMA_INVALID,
                f"pack.yaml validation failed: {e}",
                ["pack.yaml"]
            )

    def _validate_input_schema(self, input_data: InT) -> None:
        """Validate input against AgentSpec v2 schema."""
        if not self.spec or not self.spec.compute:
            return

        input_dict = input_data if isinstance(input_data, dict) else input_data.__dict__

        # Check required fields
        for field_name, field_spec in self.spec.compute.inputs.items():
            if field_spec.required and field_name not in input_dict:
                raise GLValidationError(
                    GLVErr.MISSING_FIELD,
                    f"Required input field '{field_name}' is missing",
                    ["compute", "inputs", field_name]
                )

            # Validate constraints if field present
            if field_name in input_dict:
                value = input_dict[field_name]
                self._check_constraints(field_name, value, field_spec)

    def _validate_output_schema(self, output: OutT) -> None:
        """Validate output against AgentSpec v2 schema."""
        if not self.spec or not self.spec.compute:
            return

        output_dict = output if isinstance(output, dict) else output.__dict__

        # Check all required outputs are present
        for field_name in self.spec.compute.outputs.keys():
            if field_name not in output_dict:
                raise GLValidationError(
                    GLVErr.MISSING_FIELD,
                    f"Required output field '{field_name}' is missing",
                    ["compute", "outputs", field_name]
                )

    def _check_constraints(self, field_name: str, value: Any, field_spec: Any) -> None:
        """Check value against field constraints."""
        # Greater than or equal
        if hasattr(field_spec, "ge") and field_spec.ge is not None:
            if value < field_spec.ge:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"{field_name} must be >= {field_spec.ge}, got {value}",
                    ["compute", "inputs", field_name]
                )

        # Less than or equal
        if hasattr(field_spec, "le") and field_spec.le is not None:
            if value > field_spec.le:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"{field_name} must be <= {field_spec.le}, got {value}",
                    ["compute", "inputs", field_name]
                )

        # Enum constraint
        if hasattr(field_spec, "enum") and field_spec.enum is not None:
            if value not in field_spec.enum:
                raise GLValidationError(
                    GLVErr.CONSTRAINT,
                    f"{field_name} must be one of {field_spec.enum}, got {value}",
                    ["compute", "inputs", field_name]
                )

    # ==========================================================================
    # Metrics and Statistics
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        avg_time = (
            self._total_execution_time_ms / self._execution_count
            if self._execution_count > 0
            else 0
        )

        return {
            "agent_id": self.agent_id,
            "executions": self._execution_count,
            "total_time_ms": round(self._total_execution_time_ms, 2),
            "avg_time_ms": round(avg_time, 2),
            "state": self._state,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"agent_id={self.agent_id}, "
            f"executions={self._execution_count}, "
            f"state={self._state})"
        )
