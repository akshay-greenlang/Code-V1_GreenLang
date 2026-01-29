# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Steps
============================

GL-FOUND-X-001 / GL-FOUND-X-002: Built-in step types for the orchestrator.

This module provides custom step types that extend the orchestrator's
capabilities beyond simple agent execution. Steps can perform specialized
operations like validation, transformation, aggregation, etc.

Available Steps:
    - ValidateStep: Validates payloads against GreenLang schemas

Step Registration:
    Steps must be registered with the step registry to be used in pipelines.
    Use register_all_steps() to register all built-in steps.

Example:
    >>> from greenlang.orchestrator.steps import (
    ...     ValidateStep,
    ...     ValidateStepConfig,
    ...     ValidateStepInput,
    ...     register_all_steps,
    ... )
    >>> register_all_steps()
    >>> step = ValidateStep(ValidateStepConfig(schema_uri="gl://schemas/test@1.0.0"))
    >>> result = await step.execute(inputs, context)

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 6.2
"""

from greenlang.orchestrator.steps.validate_step import (
    # Models
    ValidateStepConfig,
    ValidateStepInput,
    ValidateStepOutput,
    ValidateStepFinding,
    # Step implementation
    ValidateStep,
    # Exception
    ValidationFailedError,
    # Registration
    register_validate_step,
    # Factory
    create_validate_step,
    # Constants
    STEP_TYPE as VALIDATE_STEP_TYPE,
    STEP_VERSION as VALIDATE_STEP_VERSION,
)

# Step registry
_STEP_REGISTRY: dict = {}


def get_step_registry() -> dict:
    """
    Get the step registry.

    Returns:
        Dictionary mapping step_type to step class.
    """
    return _STEP_REGISTRY


def register_step(step_type: str, step_class: type) -> None:
    """
    Register a step type with the registry.

    Args:
        step_type: The step type identifier (e.g., "validate").
        step_class: The step class to register.

    Raises:
        ValueError: If step_type is already registered.
    """
    if step_type in _STEP_REGISTRY:
        raise ValueError(f"Step type '{step_type}' is already registered")
    _STEP_REGISTRY[step_type] = step_class


def get_step_class(step_type: str) -> type:
    """
    Get a step class by type.

    Args:
        step_type: The step type identifier.

    Returns:
        The registered step class.

    Raises:
        KeyError: If step_type is not registered.
    """
    if step_type not in _STEP_REGISTRY:
        raise KeyError(f"Unknown step type: '{step_type}'")
    return _STEP_REGISTRY[step_type]


def register_all_steps() -> None:
    """
    Register all built-in step types with the registry.

    This should be called during orchestrator initialization to make
    all built-in steps available for use in pipelines.
    """
    # Register validate step
    register_validate_step()


__all__ = [
    # Models
    "ValidateStepConfig",
    "ValidateStepInput",
    "ValidateStepOutput",
    "ValidateStepFinding",
    # Step implementation
    "ValidateStep",
    # Exception
    "ValidationFailedError",
    # Registration
    "register_validate_step",
    # Factory
    "create_validate_step",
    # Constants
    "VALIDATE_STEP_TYPE",
    "VALIDATE_STEP_VERSION",
    # Registry functions
    "get_step_registry",
    "register_step",
    "get_step_class",
    "register_all_steps",
]

__version__ = "1.0.0"
