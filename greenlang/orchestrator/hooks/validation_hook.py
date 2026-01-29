# -*- coding: utf-8 -*-
"""
Pre-Run Validation Hook for Pipeline Execution
===============================================

GL-FOUND-X-002: Schema Compiler & Validator - Task 6.3

This module implements the pre-run validation hook for the GreenLang orchestrator.
The hook validates all pipeline inputs against declared schemas before the pipeline
starts executing, ensuring data quality and early failure detection.

Key Features:
    - Validates pipeline inputs against declared schemas
    - Supports multiple validation profiles (strict, standard, permissive)
    - Optional input normalization for downstream processing
    - Configurable failure behavior (fail on warnings or errors only)
    - Complete audit trail with validation results
    - Async-first design for non-blocking validation

Configuration Example (YAML):
    ```yaml
    pipeline:
      id: emissions-calculation
      validation:
        enabled: true
        profile: strict
        fail_on_warnings: false
      inputs:
        data:
          schema: gl://schemas/emissions/activity@1.3.0
          required: true
        config:
          schema: gl://schemas/config/calculation@1.0.0
          required: false
    ```

Example Usage:
    >>> from greenlang.orchestrator.hooks.validation_hook import (
    ...     PreRunValidationHook,
    ...     PipelineValidationConfig,
    ...     InputSchemaSpec,
    ... )
    >>> config = PipelineValidationConfig(
    ...     enabled=True,
    ...     profile=ValidationProfile.STRICT,
    ...     fail_on_warnings=False,
    ... )
    >>> input_schemas = {
    ...     "data": InputSchemaSpec(
    ...         schema="gl://schemas/emissions/activity@1.3.0",
    ...         required=True,
    ...     ),
    ... }
    >>> hook = PreRunValidationHook(config, input_schemas)
    >>> result = await hook.validate(inputs, context)
    >>> if not result.valid:
    ...     raise ValidationHookError(result)

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 6.3
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

# Schema validator imports
from greenlang.schema.validator.core import SchemaValidator
from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.models.config import (
    ValidationOptions,
    ValidationProfile,
    CoercionPolicy,
    UnknownFieldPolicy,
)
from greenlang.schema.models.report import ValidationReport

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


class InputSchemaSpec(BaseModel):
    """
    Schema specification for a pipeline input.

    Defines the schema reference and validation requirements for a single
    pipeline input. Each input can have its own schema and optionally
    override the pipeline-level validation profile.

    Attributes:
        schema: Schema reference URI (e.g., "gl://schemas/emissions/activity@1.3.0").
        required: Whether this input is required for pipeline execution.
        profile: Optional validation profile override for this input.
        description: Optional human-readable description of the input.

    Example:
        >>> spec = InputSchemaSpec(
        ...     schema="gl://schemas/emissions/activity@1.3.0",
        ...     required=True,
        ...     profile=ValidationProfile.STRICT,
        ... )
    """

    schema: str = Field(
        ...,
        min_length=1,
        description="Schema reference URI (e.g., gl://schemas/emissions/activity@1.3.0)"
    )

    required: bool = Field(
        default=True,
        description="Whether this input is required for pipeline execution"
    )

    profile: Optional[ValidationProfile] = Field(
        default=None,
        description="Optional validation profile override for this input"
    )

    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional human-readable description of the input"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "schema": "gl://schemas/emissions/activity@1.3.0",
                    "required": True,
                },
                {
                    "schema": "gl://schemas/config/calculation@1.0.0",
                    "required": False,
                    "profile": "strict",
                }
            ]
        }
    }

    @field_validator("schema")
    @classmethod
    def validate_schema_uri(cls, v: str) -> str:
        """
        Validate schema URI format.

        Args:
            v: The schema URI to validate.

        Returns:
            The validated schema URI.

        Raises:
            ValueError: If URI format is invalid.
        """
        if not v.startswith("gl://schemas/"):
            raise ValueError(
                f"Invalid schema URI '{v}'. Must start with 'gl://schemas/'."
            )
        if "@" not in v:
            raise ValueError(
                f"Invalid schema URI '{v}'. Must include version after '@'."
            )
        return v

    def to_schema_ref(self) -> SchemaRef:
        """
        Convert schema URI to SchemaRef object.

        Returns:
            SchemaRef parsed from the URI.
        """
        return SchemaRef.from_uri(self.schema)


class PipelineValidationConfig(BaseModel):
    """
    Validation configuration for a pipeline.

    Defines the global validation settings for a pipeline, including
    whether validation is enabled, the default profile, and failure behavior.

    Attributes:
        enabled: Whether pre-run validation is enabled.
        profile: Default validation profile for all inputs.
        fail_on_warnings: Whether to fail the pipeline if warnings are found.
        normalize_inputs: Whether to normalize inputs to canonical form.
        max_validation_time_seconds: Maximum time allowed for validation.

    Example:
        >>> config = PipelineValidationConfig(
        ...     enabled=True,
        ...     profile=ValidationProfile.STRICT,
        ...     fail_on_warnings=False,
        ... )
    """

    enabled: bool = Field(
        default=True,
        description="Whether pre-run validation is enabled"
    )

    profile: ValidationProfile = Field(
        default=ValidationProfile.STANDARD,
        description="Default validation profile for all inputs"
    )

    fail_on_warnings: bool = Field(
        default=False,
        description="Whether to fail the pipeline if warnings are found"
    )

    normalize_inputs: bool = Field(
        default=True,
        description="Whether to normalize inputs to canonical form"
    )

    max_validation_time_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
        description="Maximum time allowed for validation (seconds)"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "enabled": True,
                    "profile": "standard",
                    "fail_on_warnings": False,
                    "normalize_inputs": True,
                },
                {
                    "enabled": True,
                    "profile": "strict",
                    "fail_on_warnings": True,
                    "normalize_inputs": True,
                    "max_validation_time_seconds": 30.0,
                }
            ]
        }
    }

    def to_validation_options(self) -> ValidationOptions:
        """
        Convert to ValidationOptions for the schema validator.

        Returns:
            ValidationOptions configured from this config.
        """
        return ValidationOptions(
            profile=self.profile,
            normalize=self.normalize_inputs,
            emit_patches=True,
            max_errors=100,
            fail_fast=False,
        )


class InputValidationResult(BaseModel):
    """
    Result of validating a single input.

    Contains the validation outcome for one pipeline input, including
    the validation report and optionally the normalized value.

    Attributes:
        input_name: Name of the input being validated.
        valid: Whether the input passed validation.
        report: Full validation report (if validation succeeded).
        normalized_value: Normalized input value (if normalization enabled).
        error: Error message (if validation failed to run).
        validation_time_ms: Time taken for validation in milliseconds.

    Example:
        >>> result = InputValidationResult(
        ...     input_name="data",
        ...     valid=True,
        ...     report=validation_report,
        ...     normalized_value={"energy": {"value": 100, "unit": "kWh"}},
        ... )
    """

    input_name: str = Field(
        ...,
        min_length=1,
        description="Name of the input being validated"
    )

    valid: bool = Field(
        ...,
        description="Whether the input passed validation"
    )

    report: Optional[ValidationReport] = Field(
        default=None,
        description="Full validation report (if validation completed)"
    )

    normalized_value: Optional[Any] = Field(
        default=None,
        description="Normalized input value (if normalization enabled)"
    )

    error: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Error message (if validation failed to run)"
    )

    validation_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Time taken for validation in milliseconds"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    def error_count(self) -> int:
        """Get the number of validation errors."""
        if self.report is None:
            return 1 if self.error else 0
        return self.report.summary.error_count

    def warning_count(self) -> int:
        """Get the number of validation warnings."""
        if self.report is None:
            return 0
        return self.report.summary.warning_count

    def has_findings(self) -> bool:
        """Check if there are any validation findings."""
        if self.report is None:
            return self.error is not None
        return self.report.summary.total_findings() > 0


class PreRunValidationResult(BaseModel):
    """
    Result of pre-run validation for all pipeline inputs.

    Contains the aggregate validation results for all pipeline inputs,
    including individual results and summary statistics.

    Attributes:
        valid: Whether all required inputs passed validation.
        input_results: Validation results for each input.
        errors: List of error messages from validation.
        total_findings: Total number of validation findings across all inputs.
        total_validation_time_ms: Total time taken for validation.
        provenance_hash: SHA-256 hash for audit trail.

    Example:
        >>> result = PreRunValidationResult(
        ...     valid=True,
        ...     input_results={
        ...         "data": InputValidationResult(input_name="data", valid=True),
        ...     },
        ...     errors=[],
        ...     total_findings=0,
        ... )
    """

    valid: bool = Field(
        ...,
        description="Whether all required inputs passed validation"
    )

    input_results: Dict[str, InputValidationResult] = Field(
        default_factory=dict,
        description="Validation results for each input"
    )

    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages from validation"
    )

    total_findings: int = Field(
        default=0,
        ge=0,
        description="Total number of validation findings across all inputs"
    )

    total_validation_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total time taken for validation in milliseconds"
    )

    provenance_hash: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of validation results for audit trail"
    )

    validated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when validation completed"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    def get_normalized_inputs(self) -> Dict[str, Any]:
        """
        Get all normalized input values.

        Returns:
            Dictionary of input names to normalized values.
        """
        return {
            name: result.normalized_value
            for name, result in self.input_results.items()
            if result.normalized_value is not None
        }

    def get_failed_inputs(self) -> List[str]:
        """
        Get list of input names that failed validation.

        Returns:
            List of failed input names.
        """
        return [
            name for name, result in self.input_results.items()
            if not result.valid
        ]

    def total_errors(self) -> int:
        """Get total number of errors across all inputs."""
        return sum(r.error_count() for r in self.input_results.values())

    def total_warnings(self) -> int:
        """Get total number of warnings across all inputs."""
        return sum(r.warning_count() for r in self.input_results.values())

    def format_summary(self) -> str:
        """
        Format a human-readable summary.

        Returns:
            Summary string.
        """
        status = "PASSED" if self.valid else "FAILED"
        input_count = len(self.input_results)
        failed_count = len(self.get_failed_inputs())
        return (
            f"Pre-run validation {status}: "
            f"{input_count - failed_count}/{input_count} inputs valid, "
            f"{self.total_findings} findings, "
            f"{self.total_validation_time_ms:.1f}ms"
        )


# =============================================================================
# EXCEPTION
# =============================================================================


class ValidationHookError(Exception):
    """
    Raised when pre-run validation fails.

    This exception is raised when pipeline inputs fail validation and the
    pipeline should not proceed. It contains the full validation result
    for debugging and audit purposes.

    Attributes:
        result: The PreRunValidationResult that caused the failure.

    Example:
        >>> try:
        ...     if not result.valid:
        ...         raise ValidationHookError(result)
        ... except ValidationHookError as e:
        ...     print(f"Validation failed: {e.result.errors}")
    """

    def __init__(self, result: PreRunValidationResult):
        """
        Initialize ValidationHookError.

        Args:
            result: The validation result that caused the failure.
        """
        self.result = result
        error_count = result.total_errors()
        warning_count = result.total_warnings()
        message = (
            f"Pre-run validation failed: {error_count} error(s), "
            f"{warning_count} warning(s)"
        )
        if result.errors:
            message += f". First error: {result.errors[0]}"
        super().__init__(message)


# =============================================================================
# PRE-RUN VALIDATION HOOK
# =============================================================================


class PreRunValidationHook:
    """
    Pre-run validation hook for pipeline execution.

    Validates all pipeline inputs against declared schemas before the
    pipeline starts executing. This ensures data quality and provides
    early failure detection before expensive computations begin.

    Thread Safety:
        The hook is thread-safe for concurrent validation calls.

    Attributes:
        config: Pipeline validation configuration.
        input_schemas: Schema specifications for each input.

    Example:
        >>> config = PipelineValidationConfig(
        ...     enabled=True,
        ...     profile=ValidationProfile.STRICT,
        ... )
        >>> input_schemas = {
        ...     "data": InputSchemaSpec(schema="gl://schemas/activity@1.0.0"),
        ... }
        >>> hook = PreRunValidationHook(config, input_schemas)
        >>> result = await hook.validate(inputs, context)
        >>> if not result.valid:
        ...     raise ValidationHookError(result)
    """

    def __init__(
        self,
        config: PipelineValidationConfig,
        input_schemas: Dict[str, InputSchemaSpec],
    ):
        """
        Initialize PreRunValidationHook.

        Args:
            config: Pipeline validation configuration.
            input_schemas: Schema specifications for each input.
        """
        self.config = config
        self.input_schemas = input_schemas
        self._validator: Optional[SchemaValidator] = None

        logger.debug(
            f"PreRunValidationHook initialized: "
            f"enabled={config.enabled}, "
            f"profile={config.profile.value}, "
            f"inputs={list(input_schemas.keys())}"
        )

    async def validate(
        self,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> PreRunValidationResult:
        """
        Validate all pipeline inputs before execution.

        Validates each input against its declared schema and returns
        a comprehensive result including all findings, normalized values,
        and provenance information.

        Args:
            inputs: Pipeline input values (input_name -> value).
            context: Pipeline context (may contain tenant_id, user_id, etc.).

        Returns:
            PreRunValidationResult with all validation results.

        Example:
            >>> inputs = {
            ...     "data": {"energy": 100, "unit": "kWh"},
            ...     "config": {"method": "market-based"},
            ... }
            >>> context = {"tenant_id": "tenant-123", "run_id": "run-456"}
            >>> result = await hook.validate(inputs, context)
        """
        start_time = time.perf_counter()
        validated_at = datetime.utcnow()

        logger.info(
            f"Starting pre-run validation: "
            f"{len(self.input_schemas)} input schemas, "
            f"{len(inputs)} inputs provided"
        )

        # Check if validation is disabled
        if not self.config.enabled:
            logger.info("Pre-run validation disabled, skipping")
            return PreRunValidationResult(
                valid=True,
                input_results={},
                errors=[],
                total_findings=0,
                total_validation_time_ms=0.0,
                validated_at=validated_at,
            )

        # Validate all inputs
        input_results: Dict[str, InputValidationResult] = {}
        errors: List[str] = []
        total_findings = 0

        # Create validation tasks for concurrent execution
        validation_tasks = []
        input_names = []

        for name, spec in self.input_schemas.items():
            input_names.append(name)
            value = inputs.get(name)
            validation_tasks.append(
                self._validate_input_with_timeout(name, value, spec)
            )

        # Execute validations concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*validation_tasks, return_exceptions=True),
                timeout=self.config.max_validation_time_seconds,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Validation timeout exceeded: {self.config.max_validation_time_seconds}s"
            )
            total_time_ms = (time.perf_counter() - start_time) * 1000
            return PreRunValidationResult(
                valid=False,
                input_results={},
                errors=[
                    f"Validation timeout exceeded: {self.config.max_validation_time_seconds}s"
                ],
                total_findings=0,
                total_validation_time_ms=total_time_ms,
                validated_at=validated_at,
            )

        # Process results
        for name, result in zip(input_names, results):
            if isinstance(result, Exception):
                # Validation task raised an exception
                error_msg = f"Validation failed for input '{name}': {str(result)}"
                logger.error(error_msg)
                input_results[name] = InputValidationResult(
                    input_name=name,
                    valid=False,
                    error=error_msg,
                )
                errors.append(error_msg)
            else:
                input_results[name] = result
                if not result.valid:
                    if result.error:
                        errors.append(result.error)
                    elif result.report:
                        for finding in result.report.findings:
                            if finding.is_error():
                                errors.append(
                                    f"[{name}] {finding.code}: {finding.message}"
                                )
                if result.report:
                    total_findings += result.report.summary.total_findings()

        # Determine overall validity
        valid = self._determine_overall_validity(input_results, errors)

        # Calculate total time
        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Compute provenance hash
        provenance_hash = self._compute_provenance_hash(
            inputs, input_results, context
        )

        result = PreRunValidationResult(
            valid=valid,
            input_results=input_results,
            errors=errors,
            total_findings=total_findings,
            total_validation_time_ms=total_time_ms,
            provenance_hash=provenance_hash,
            validated_at=validated_at,
        )

        logger.info(
            f"Pre-run validation completed: {result.format_summary()}"
        )

        return result

    async def _validate_input_with_timeout(
        self,
        name: str,
        value: Any,
        spec: InputSchemaSpec,
    ) -> InputValidationResult:
        """
        Validate a single input with individual timeout handling.

        Args:
            name: Input name.
            value: Input value (may be None).
            spec: Schema specification for this input.

        Returns:
            InputValidationResult for this input.
        """
        return await self._validate_input(name, value, spec)

    async def _validate_input(
        self,
        name: str,
        value: Any,
        spec: InputSchemaSpec,
    ) -> InputValidationResult:
        """
        Validate a single input against its schema.

        Args:
            name: Input name.
            value: Input value (may be None).
            spec: Schema specification for this input.

        Returns:
            InputValidationResult for this input.
        """
        start_time = time.perf_counter()

        logger.debug(f"Validating input '{name}' against schema '{spec.schema}'")

        # Check if required input is missing
        if value is None:
            if spec.required:
                error_msg = f"Required input '{name}' is missing"
                logger.warning(error_msg)
                return InputValidationResult(
                    input_name=name,
                    valid=False,
                    error=error_msg,
                    validation_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            else:
                # Optional input not provided - valid
                logger.debug(f"Optional input '{name}' not provided, skipping validation")
                return InputValidationResult(
                    input_name=name,
                    valid=True,
                    validation_time_ms=(time.perf_counter() - start_time) * 1000,
                )

        try:
            # Get or create validator
            validator = self._get_validator()

            # Build validation options
            options = self._build_options(spec)

            # Parse schema reference
            schema_ref = spec.to_schema_ref()

            # Run validation (synchronous call wrapped for async)
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(
                None,
                lambda: validator.validate(value, schema_ref, options),
            )

            # Extract normalized value if available
            normalized_value = None
            if self.config.normalize_inputs and report.normalized_payload is not None:
                normalized_value = report.normalized_payload

            validation_time_ms = (time.perf_counter() - start_time) * 1000

            return InputValidationResult(
                input_name=name,
                valid=report.valid,
                report=report,
                normalized_value=normalized_value,
                validation_time_ms=validation_time_ms,
            )

        except Exception as e:
            logger.error(f"Validation error for input '{name}': {e}", exc_info=True)
            return InputValidationResult(
                input_name=name,
                valid=False,
                error=f"Validation error: {str(e)}",
                validation_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _get_validator(self) -> SchemaValidator:
        """
        Get or create validator instance.

        Returns:
            SchemaValidator instance configured with base options.
        """
        if self._validator is None:
            # Create validator with default configuration
            self._validator = SchemaValidator(
                options=self.config.to_validation_options(),
            )
            logger.debug("Created new SchemaValidator instance")
        return self._validator

    def _build_options(
        self,
        spec: InputSchemaSpec,
    ) -> ValidationOptions:
        """
        Build validation options for an input.

        Combines pipeline-level configuration with input-specific overrides.

        Args:
            spec: Schema specification for the input.

        Returns:
            ValidationOptions configured for this input.
        """
        # Start with pipeline defaults
        profile = self.config.profile

        # Apply input-level profile override if specified
        if spec.profile is not None:
            profile = spec.profile

        return ValidationOptions(
            profile=profile,
            normalize=self.config.normalize_inputs,
            emit_patches=True,
            max_errors=100,
            fail_fast=False,
        )

    def _determine_overall_validity(
        self,
        input_results: Dict[str, InputValidationResult],
        errors: List[str],
    ) -> bool:
        """
        Determine overall validation validity.

        Args:
            input_results: Results for each input.
            errors: List of error messages.

        Returns:
            True if validation passed, False otherwise.
        """
        # Check for any validation errors
        has_errors = any(not r.valid for r in input_results.values())

        if has_errors:
            return False

        # Check for warnings if fail_on_warnings is enabled
        if self.config.fail_on_warnings:
            has_warnings = any(
                r.warning_count() > 0 for r in input_results.values()
            )
            if has_warnings:
                return False

        return True

    def _should_fail(self, result: PreRunValidationResult) -> bool:
        """
        Check if pipeline should fail based on validation results.

        Args:
            result: The validation result to check.

        Returns:
            True if pipeline should fail, False otherwise.
        """
        if not result.valid:
            return True

        if self.config.fail_on_warnings and result.total_warnings() > 0:
            return True

        return False

    def _compute_provenance_hash(
        self,
        inputs: Dict[str, Any],
        results: Dict[str, InputValidationResult],
        context: Dict[str, Any],
    ) -> str:
        """
        Compute SHA-256 hash for audit trail.

        Creates a provenance hash that includes:
        - Input schemas used
        - Validation configuration
        - Result summaries

        Args:
            inputs: Original input values.
            results: Validation results.
            context: Pipeline context.

        Returns:
            SHA-256 hash string (64 characters).
        """
        import json

        # Build provenance content
        provenance_data = {
            "config": {
                "enabled": self.config.enabled,
                "profile": self.config.profile.value,
                "fail_on_warnings": self.config.fail_on_warnings,
                "normalize_inputs": self.config.normalize_inputs,
            },
            "schemas": {
                name: spec.schema
                for name, spec in self.input_schemas.items()
            },
            "results": {
                name: {
                    "valid": result.valid,
                    "error_count": result.error_count(),
                    "warning_count": result.warning_count(),
                }
                for name, result in results.items()
            },
            "context_keys": sorted(context.keys()),
        }

        # Compute hash
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_validation_hook(
    pipeline_config: Dict[str, Any],
) -> Optional[PreRunValidationHook]:
    """
    Create validation hook from pipeline configuration.

    Parses the pipeline configuration dictionary and creates a
    PreRunValidationHook if validation is enabled.

    Args:
        pipeline_config: Pipeline configuration dictionary. Expected format:
            ```yaml
            pipeline:
              validation:
                enabled: true
                profile: strict
                fail_on_warnings: false
              inputs:
                data:
                  schema: gl://schemas/emissions/activity@1.3.0
                  required: true
            ```

    Returns:
        PreRunValidationHook instance, or None if validation is disabled.

    Example:
        >>> config = {
        ...     "pipeline": {
        ...         "validation": {"enabled": True, "profile": "strict"},
        ...         "inputs": {
        ...             "data": {"schema": "gl://schemas/activity@1.0.0"}
        ...         }
        ...     }
        ... }
        >>> hook = create_validation_hook(config)
    """
    # Extract pipeline section
    pipeline = pipeline_config.get("pipeline", pipeline_config)

    # Extract validation config
    validation_dict = pipeline.get("validation", {})

    # Check if validation is explicitly disabled
    if not validation_dict.get("enabled", True):
        logger.info("Validation disabled in pipeline configuration")
        return None

    # Parse validation config
    profile_str = validation_dict.get("profile", "standard")
    try:
        profile = ValidationProfile(profile_str)
    except ValueError:
        logger.warning(f"Invalid profile '{profile_str}', using standard")
        profile = ValidationProfile.STANDARD

    config = PipelineValidationConfig(
        enabled=validation_dict.get("enabled", True),
        profile=profile,
        fail_on_warnings=validation_dict.get("fail_on_warnings", False),
        normalize_inputs=validation_dict.get("normalize_inputs", True),
        max_validation_time_seconds=validation_dict.get(
            "max_validation_time_seconds", 60.0
        ),
    )

    # Extract input schemas
    input_schemas = extract_input_schemas(pipeline_config)

    # Check if there are any schemas to validate
    if not input_schemas:
        logger.info("No input schemas defined, validation hook not needed")
        return None

    return PreRunValidationHook(config, input_schemas)


def extract_input_schemas(
    pipeline_config: Dict[str, Any],
) -> Dict[str, InputSchemaSpec]:
    """
    Extract input schema specifications from pipeline config.

    Parses the inputs section of the pipeline configuration and
    creates InputSchemaSpec objects for each input that has a
    schema defined.

    Args:
        pipeline_config: Pipeline configuration dictionary. Expected format:
            ```yaml
            pipeline:
              inputs:
                data:
                  schema: gl://schemas/emissions/activity@1.3.0
                  required: true
                config:
                  schema: gl://schemas/config/calculation@1.0.0
                  required: false
            ```

    Returns:
        Dictionary of input names to InputSchemaSpec objects.

    Example:
        >>> config = {
        ...     "pipeline": {
        ...         "inputs": {
        ...             "data": {"schema": "gl://schemas/activity@1.0.0"}
        ...         }
        ...     }
        ... }
        >>> schemas = extract_input_schemas(config)
        >>> print(schemas["data"].schema)
        gl://schemas/activity@1.0.0
    """
    result: Dict[str, InputSchemaSpec] = {}

    # Extract pipeline section
    pipeline = pipeline_config.get("pipeline", pipeline_config)

    # Extract inputs section
    inputs = pipeline.get("inputs", {})

    for input_name, input_def in inputs.items():
        # Skip if no schema defined
        if not isinstance(input_def, dict):
            continue

        schema_uri = input_def.get("schema")
        if not schema_uri:
            continue

        # Parse profile if specified
        profile = None
        profile_str = input_def.get("profile")
        if profile_str:
            try:
                profile = ValidationProfile(profile_str)
            except ValueError:
                logger.warning(
                    f"Invalid profile '{profile_str}' for input '{input_name}'"
                )

        try:
            spec = InputSchemaSpec(
                schema=schema_uri,
                required=input_def.get("required", True),
                profile=profile,
                description=input_def.get("description"),
            )
            result[input_name] = spec
            logger.debug(
                f"Extracted schema spec for input '{input_name}': {schema_uri}"
            )
        except ValueError as e:
            logger.error(
                f"Invalid schema spec for input '{input_name}': {e}"
            )

    return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================


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
