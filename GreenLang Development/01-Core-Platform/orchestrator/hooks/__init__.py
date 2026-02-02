# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Hooks
============================

GL-FOUND-X-001 / GL-FOUND-X-002: Pipeline hook system for the orchestrator.

This module provides hooks that can be registered to execute at various
points in the pipeline lifecycle:

- Pre-run hooks: Execute before pipeline starts (validation, setup)
- Post-run hooks: Execute after pipeline completes (cleanup, reporting)
- Pre-step hooks: Execute before each step
- Post-step hooks: Execute after each step

Available Hooks:
    - PreRunValidationHook: Validates pipeline inputs against declared schemas

Example:
    >>> from greenlang.orchestrator.hooks import (
    ...     PreRunValidationHook,
    ...     PipelineValidationConfig,
    ...     create_validation_hook,
    ... )
    >>> hook = create_validation_hook(pipeline_config)
    >>> result = await hook.validate(inputs, context)

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 6.3
"""

from greenlang.orchestrator.hooks.validation_hook import (
    # Models
    InputSchemaSpec,
    PipelineValidationConfig,
    InputValidationResult,
    PreRunValidationResult,
    # Hook implementation
    PreRunValidationHook,
    # Exception
    ValidationHookError,
    # Factory functions
    create_validation_hook,
    extract_input_schemas,
)

__all__ = [
    # Models
    "InputSchemaSpec",
    "PipelineValidationConfig",
    "InputValidationResult",
    "PreRunValidationResult",
    # Hook implementation
    "PreRunValidationHook",
    # Exception
    "ValidationHookError",
    # Factory functions
    "create_validation_hook",
    "extract_input_schemas",
]

__version__ = "1.0.0"
