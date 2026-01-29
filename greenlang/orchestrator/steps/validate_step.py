# -*- coding: utf-8 -*-
"""
ValidateStep - Schema Validation Step for GL-FOUND-X-001 Orchestrator
======================================================================

This module implements the ValidateStep for the GreenLang orchestrator,
enabling schema validation as a first-class step type in pipelines.
ValidateStep integrates with the SchemaValidator from GL-FOUND-X-002
to validate payloads against GreenLang schemas.

Key Features:
    - Schema validation against GreenLang schemas (gl://schemas/*)
    - Configurable validation profiles (strict, standard, permissive)
    - Optional payload normalization
    - Fail-on-warnings option for strict pipelines
    - Full provenance tracking with SHA-256 hashes
    - Deterministic execution (zero-hallucination)

Example:
    >>> from greenlang.orchestrator.steps import ValidateStep, ValidateStepConfig
    >>> config = ValidateStepConfig(
    ...     schema="gl://schemas/activity@1.0.0",
    ...     profile="strict",
    ...     fail_on_warnings=True,
    ...     normalize=True,
    ... )
    >>> step = ValidateStep(config)
    >>> result = await step.execute(
    ...     inputs={"payload": {"energy": 100, "unit": "kWh"}},
    ...     context=execution_context,
    ... )
    >>> if result.valid:
    ...     print("Validation passed!")

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-001: Orchestrator - Task 6.2 Validate Step Type
GL-FOUND-X-002: Schema Compiler & Validator Integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Schema validator integration
from greenlang.schema.validator.core import SchemaValidator
from greenlang.schema.models.config import (
    ValidationOptions,
    ValidationProfile,
    UnknownFieldPolicy,
    CoercionPolicy,
)
from greenlang.schema.models.finding import Finding, Severity
from greenlang.schema.models.report import ValidationReport
from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.registry.resolver import SchemaRegistry

# NOTE: Step registry functions are defined in __init__.py
# We avoid importing them here to prevent circular imports.
# The register_validate_step function will import them at runtime.


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

STEP_TYPE = "validate"
STEP_VERSION = "1.0.0"

# Error codes for validation step
ERROR_CODES = {
    "INVALID_CONFIG": "GL-STEP-E001",
    "MISSING_PAYLOAD": "GL-STEP-E002",
    "SCHEMA_NOT_FOUND": "GL-STEP-E003",
    "VALIDATION_FAILED": "GL-STEP-E004",
    "INTERNAL_ERROR": "GL-STEP-E500",
}


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ValidationFailedError(Exception):
    """
    Exception raised when validation fails and fail_on_warnings or errors are present.

    Attributes:
        error_count: Number of validation errors.
        warning_count: Number of validation warnings.
        findings: List of validation findings.
        message: Human-readable error message.
    """

    def __init__(
        self,
        message: str,
        error_count: int = 0,
        warning_count: int = 0,
        findings: Optional[List[Finding]] = None,
    ):
        """
        Initialize ValidationFailedError.

        Args:
            message: Human-readable error message.
            error_count: Number of validation errors.
            warning_count: Number of validation warnings.
            findings: List of validation findings.
        """
        super().__init__(message)
        self.message = message
        self.error_count = error_count
        self.warning_count = warning_count
        self.findings = findings or []

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"{self.message} "
            f"(errors={self.error_count}, warnings={self.warning_count})"
        )


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class ValidateStepConfig(BaseModel):
    """
    Configuration for ValidateStep.

    Defines how the validation step should behave, including which schema
    to validate against, the validation profile, and error handling options.

    Attributes:
        schema_uri: GreenLang schema reference (e.g., "gl://schemas/activity@1.0.0").
        profile: Validation profile (strict, standard, permissive).
        fail_on_warnings: Whether to fail the step if warnings are present.
        normalize: Whether to normalize the payload to canonical form.
        max_errors: Maximum number of errors to collect before stopping.
        unknown_field_policy: How to handle unknown fields.
        coercion_policy: Type coercion policy during normalization.

    Example:
        >>> config = ValidateStepConfig(
        ...     schema_uri="gl://schemas/activity@1.0.0",
        ...     profile="strict",
        ...     fail_on_warnings=True,
        ...     normalize=True,
        ... )
    """

    schema_uri: str = Field(
        ...,
        min_length=1,
        max_length=512,
        alias="schema",
        description="GreenLang schema reference URI (e.g., 'gl://schemas/activity@1.0.0')"
    )

    profile: str = Field(
        default="standard",
        description="Validation profile: 'strict', 'standard', or 'permissive'"
    )

    fail_on_warnings: bool = Field(
        default=False,
        description="Whether to fail the step if warnings are present"
    )

    normalize: bool = Field(
        default=True,
        description="Whether to normalize the payload to canonical form"
    )

    max_errors: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of errors to collect"
    )

    unknown_field_policy: str = Field(
        default="warn",
        description="Policy for unknown fields: 'error', 'warn', or 'ignore'"
    )

    coercion_policy: str = Field(
        default="safe",
        description="Type coercion policy: 'off', 'safe', or 'aggressive'"
    )

    # Optional: schema registry configuration
    registry_url: Optional[str] = Field(
        default=None,
        description="URL to schema registry (if different from default)"
    )

    # Optional: timeout for validation
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum time allowed for validation"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "schema": "gl://schemas/activity@1.0.0",
                    "profile": "standard",
                    "fail_on_warnings": False,
                    "normalize": True,
                    "max_errors": 100,
                },
                {
                    "schema": "gl://schemas/emissions@2.1.0",
                    "profile": "strict",
                    "fail_on_warnings": True,
                    "normalize": True,
                    "max_errors": 50,
                    "unknown_field_policy": "error",
                },
            ]
        }
    }

    @field_validator("schema_uri")
    @classmethod
    def validate_schema_uri(cls, v: str) -> str:
        """
        Validate schema URI format.

        Args:
            v: The schema URI string.

        Returns:
            The validated schema URI.

        Raises:
            ValueError: If URI format is invalid.
        """
        # Allow both gl:// URIs and simple schema@version format
        if not (v.startswith("gl://") or "@" in v):
            raise ValueError(
                f"Invalid schema URI '{v}'. "
                "Expected format: 'gl://schemas/name@version' or 'name@version'"
            )
        return v

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, v: str) -> str:
        """
        Validate profile value.

        Args:
            v: The profile string.

        Returns:
            The validated profile (lowercase).

        Raises:
            ValueError: If profile is not recognized.
        """
        valid_profiles = {"strict", "standard", "permissive"}
        v_lower = v.lower()
        if v_lower not in valid_profiles:
            raise ValueError(
                f"Invalid profile '{v}'. Must be one of: {', '.join(valid_profiles)}"
            )
        return v_lower

    @field_validator("unknown_field_policy")
    @classmethod
    def validate_unknown_field_policy(cls, v: str) -> str:
        """Validate unknown_field_policy value."""
        valid_policies = {"error", "warn", "ignore"}
        v_lower = v.lower()
        if v_lower not in valid_policies:
            raise ValueError(
                f"Invalid unknown_field_policy '{v}'. "
                f"Must be one of: {', '.join(valid_policies)}"
            )
        return v_lower

    @field_validator("coercion_policy")
    @classmethod
    def validate_coercion_policy(cls, v: str) -> str:
        """Validate coercion_policy value."""
        valid_policies = {"off", "safe", "aggressive"}
        v_lower = v.lower()
        if v_lower not in valid_policies:
            raise ValueError(
                f"Invalid coercion_policy '{v}'. "
                f"Must be one of: {', '.join(valid_policies)}"
            )
        return v_lower

    def to_validation_options(self) -> ValidationOptions:
        """
        Convert to ValidationOptions for SchemaValidator.

        Returns:
            ValidationOptions configured from this config.
        """
        # Map string profile to enum
        profile_map = {
            "strict": ValidationProfile.STRICT,
            "standard": ValidationProfile.STANDARD,
            "permissive": ValidationProfile.PERMISSIVE,
        }

        # Map string policies to enums
        unknown_map = {
            "error": UnknownFieldPolicy.ERROR,
            "warn": UnknownFieldPolicy.WARN,
            "ignore": UnknownFieldPolicy.IGNORE,
        }

        coercion_map = {
            "off": CoercionPolicy.OFF,
            "safe": CoercionPolicy.SAFE,
            "aggressive": CoercionPolicy.AGGRESSIVE,
        }

        return ValidationOptions(
            profile=profile_map[self.profile],
            normalize=self.normalize,
            max_errors=self.max_errors,
            unknown_field_policy=unknown_map[self.unknown_field_policy],
            coercion_policy=coercion_map[self.coercion_policy],
        )


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================


class ValidateStepInput(BaseModel):
    """
    Input data model for ValidateStep.

    Contains the payload to validate. The payload can be provided as
    a dictionary or as a JSON/YAML string.

    Attributes:
        payload: The data to validate against the schema.
        payload_id: Optional identifier for the payload (for tracking).
        metadata: Optional metadata about the payload.

    Example:
        >>> input_data = ValidateStepInput(
        ...     payload={"energy": 100, "unit": "kWh"},
        ...     payload_id="activity-2024-001",
        ... )
    """

    payload: Union[Dict[str, Any], str] = Field(
        ...,
        description="The payload to validate (dict or JSON/YAML string)"
    )

    payload_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional identifier for the payload"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata about the payload"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, v: Union[Dict[str, Any], str]) -> Union[Dict[str, Any], str]:
        """
        Validate payload is not empty.

        Args:
            v: The payload value.

        Returns:
            The validated payload.

        Raises:
            ValueError: If payload is empty.
        """
        if v is None:
            raise ValueError("Payload cannot be None")

        if isinstance(v, str) and not v.strip():
            raise ValueError("Payload string cannot be empty")

        if isinstance(v, dict) and len(v) == 0:
            raise ValueError("Payload dictionary cannot be empty")

        return v


class ValidateStepFinding(BaseModel):
    """
    Simplified finding model for step output.

    Contains the essential information from a validation finding
    in a format suitable for pipeline outputs.

    Attributes:
        code: Error code (e.g., "GLSCHEMA-E100").
        severity: Severity level ("error", "warning", "info").
        path: JSON Pointer to the issue location.
        message: Human-readable description.
    """

    code: str = Field(..., description="Error code")
    severity: str = Field(..., description="Severity level")
    path: str = Field(..., description="JSON Pointer path")
    message: str = Field(..., description="Description")

    @classmethod
    def from_finding(cls, finding: Finding) -> "ValidateStepFinding":
        """
        Create from a Finding model.

        Args:
            finding: The Finding to convert.

        Returns:
            ValidateStepFinding instance.
        """
        return cls(
            code=finding.code,
            severity=finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity),
            path=finding.path,
            message=finding.message,
        )


class ValidateStepOutput(BaseModel):
    """
    Output data model for ValidateStep.

    Contains the complete validation result including validity status,
    normalized payload, findings, and provenance information.

    Attributes:
        valid: Whether the payload passed validation.
        normalized: The normalized payload (if normalization was enabled).
        findings: List of validation findings (errors, warnings, info).
        error_count: Number of error-level findings.
        warning_count: Number of warning-level findings.
        info_count: Number of info-level findings.
        schema_ref: The schema reference used.
        schema_hash: SHA-256 hash of the compiled schema.
        provenance_hash: SHA-256 hash of input + output for audit trail.
        processing_time_ms: Time taken to process validation.
        payload_id: Identifier of the validated payload (if provided).

    Example:
        >>> output = ValidateStepOutput(
        ...     valid=True,
        ...     normalized={"energy": {"value": 100, "unit": "kWh"}},
        ...     findings=[],
        ...     error_count=0,
        ...     warning_count=0,
        ...     info_count=0,
        ...     schema_ref="gl://schemas/activity@1.0.0",
        ...     schema_hash="abc123...",
        ...     provenance_hash="def456...",
        ...     processing_time_ms=15.5,
        ... )
    """

    valid: bool = Field(
        ...,
        description="Whether the payload passed validation"
    )

    normalized: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Normalized payload (if normalization enabled)"
    )

    findings: List[ValidateStepFinding] = Field(
        default_factory=list,
        description="List of validation findings"
    )

    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of error-level findings"
    )

    warning_count: int = Field(
        default=0,
        ge=0,
        description="Number of warning-level findings"
    )

    info_count: int = Field(
        default=0,
        ge=0,
        description="Number of info-level findings"
    )

    schema_ref: str = Field(
        ...,
        description="Schema reference used for validation"
    )

    schema_hash: str = Field(
        ...,
        description="SHA-256 hash of the compiled schema"
    )

    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )

    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )

    payload_id: Optional[str] = Field(
        default=None,
        description="Identifier of the validated payload"
    )

    validated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of validation"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return self.error_count > 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return self.warning_count > 0

    def total_findings(self) -> int:
        """Get total number of findings."""
        return self.error_count + self.warning_count + self.info_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline outputs."""
        return self.model_dump()


# =============================================================================
# VALIDATE STEP IMPLEMENTATION
# =============================================================================


class ValidateStep:
    """
    ValidateStep - Schema validation step for orchestrator pipelines.

    This step validates payloads against GreenLang schemas using the
    SchemaValidator from GL-FOUND-X-002. It integrates seamlessly with
    the orchestrator's step execution model.

    Key Features:
        - Zero-hallucination: Uses deterministic schema validation
        - Provenance tracking: SHA-256 hashes for audit trails
        - Configurable profiles: strict, standard, permissive
        - Optional normalization: canonical form output
        - Fail-on-warnings: for strict pipeline requirements

    Thread Safety:
        The ValidateStep is thread-safe for concurrent execution.
        Each execute() call creates its own validation context.

    Attributes:
        config: Step configuration
        _validator: SchemaValidator instance (lazy-initialized)
        _schema_registry: Optional schema registry

    Example:
        >>> config = ValidateStepConfig(
        ...     schema="gl://schemas/activity@1.0.0",
        ...     profile="strict",
        ... )
        >>> step = ValidateStep(config)
        >>> result = await step.execute(
        ...     inputs={"payload": {"energy": 100}},
        ...     context=ctx,
        ... )
    """

    # Step type identifier (used for registration)
    STEP_TYPE = STEP_TYPE
    VERSION = STEP_VERSION

    def __init__(
        self,
        config: ValidateStepConfig,
        schema_registry: Optional[SchemaRegistry] = None,
    ):
        """
        Initialize ValidateStep.

        Args:
            config: Step configuration.
            schema_registry: Optional schema registry for resolving schemas.
        """
        self.config = config
        self._schema_registry = schema_registry
        self._validator: Optional[SchemaValidator] = None

        logger.debug(
            f"ValidateStep initialized with schema={config.schema_uri}, "
            f"profile={config.profile}"
        )

    @property
    def validator(self) -> SchemaValidator:
        """
        Get or create SchemaValidator (lazy initialization).

        Returns:
            Configured SchemaValidator instance.
        """
        if self._validator is None:
            validation_options = self.config.to_validation_options()
            self._validator = SchemaValidator(
                schema_registry=self._schema_registry,
                options=validation_options,
            )
            logger.debug("SchemaValidator created")
        return self._validator

    async def execute(
        self,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidateStepOutput:
        """
        Execute the validation step.

        This is the main entry point for step execution. It validates
        the payload against the configured schema and returns the result.

        Args:
            inputs: Dictionary containing 'payload' key with data to validate.
            context: Optional execution context with run metadata.

        Returns:
            ValidateStepOutput containing validation results.

        Raises:
            ValidationFailedError: If validation fails and errors are present.
            ValueError: If inputs are missing required data.

        Example:
            >>> result = await step.execute(
            ...     inputs={"payload": {"energy": 100, "unit": "kWh"}},
            ...     context={"run_id": "run-123"},
            ... )
        """
        start_time = time.perf_counter()
        context = context or {}

        logger.info(
            f"ValidateStep executing: schema={self.config.schema_uri}, "
            f"run_id={context.get('run_id', 'N/A')}"
        )

        try:
            # Extract and validate input
            step_input = self._extract_input(inputs)

            # Perform validation
            report = self._validate_payload(step_input.payload)

            # Calculate processing time
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance(
                step_input,
                report,
                context,
            )

            # Convert findings
            findings = [
                ValidateStepFinding.from_finding(f)
                for f in report.findings
            ]

            # Create output
            output = ValidateStepOutput(
                valid=report.valid,
                normalized=report.normalized_payload if self.config.normalize else None,
                findings=findings,
                error_count=report.summary.error_count,
                warning_count=report.summary.warning_count,
                info_count=report.summary.info_count,
                schema_ref=str(report.schema_ref),
                schema_hash=report.schema_hash,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
                payload_id=step_input.payload_id,
            )

            # Check fail conditions
            self._check_fail_conditions(output, report)

            logger.info(
                f"ValidateStep completed: valid={output.valid}, "
                f"errors={output.error_count}, warnings={output.warning_count}, "
                f"time_ms={processing_time_ms:.2f}"
            )

            return output

        except ValidationFailedError:
            # Re-raise validation errors
            raise

        except Exception as e:
            logger.error(f"ValidateStep failed: {str(e)}", exc_info=True)
            raise ValidationFailedError(
                message=f"Validation step failed: {str(e)}",
                error_count=1,
                warning_count=0,
            ) from e

    def _extract_input(self, inputs: Dict[str, Any]) -> ValidateStepInput:
        """
        Extract and validate input from inputs dictionary.

        Args:
            inputs: Raw inputs dictionary.

        Returns:
            Validated ValidateStepInput.

        Raises:
            ValueError: If required inputs are missing.
        """
        if "payload" not in inputs:
            raise ValueError(
                f"Missing required input 'payload'. "
                f"Received keys: {list(inputs.keys())}"
            )

        return ValidateStepInput(
            payload=inputs["payload"],
            payload_id=inputs.get("payload_id"),
            metadata=inputs.get("metadata"),
        )

    def _validate_payload(
        self,
        payload: Union[Dict[str, Any], str],
    ) -> ValidationReport:
        """
        Validate payload against schema.

        This method performs the actual validation using SchemaValidator.
        It handles both dict and string payloads.

        Args:
            payload: The payload to validate.

        Returns:
            ValidationReport from SchemaValidator.
        """
        # Parse schema reference
        schema_ref = self._parse_schema_ref(self.config.schema_uri)

        # Get validation options
        options = self.config.to_validation_options()

        # Perform validation
        report = self.validator.validate(
            payload=payload,
            schema_ref=schema_ref,
            options=options,
        )

        return report

    def _parse_schema_ref(self, schema_uri: str) -> Union[SchemaRef, str]:
        """
        Parse schema URI into SchemaRef or string.

        Args:
            schema_uri: Schema URI string.

        Returns:
            SchemaRef object or URI string.
        """
        # If it's a gl:// URI, parse it
        if schema_uri.startswith("gl://"):
            return schema_uri

        # If it's name@version format, convert to gl:// URI
        if "@" in schema_uri:
            parts = schema_uri.rsplit("@", 1)
            return f"gl://schemas/{parts[0]}@{parts[1]}"

        # Return as-is
        return schema_uri

    def _calculate_provenance(
        self,
        step_input: ValidateStepInput,
        report: ValidationReport,
        context: Dict[str, Any],
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        The provenance hash includes:
        - Input payload hash
        - Schema hash
        - Validation options
        - Execution context (run_id, timestamp)

        Args:
            step_input: The step input.
            report: The validation report.
            context: Execution context.

        Returns:
            SHA-256 hex digest.
        """
        # Build provenance data
        provenance_data = {
            "input_hash": self._hash_payload(step_input.payload),
            "schema_hash": report.schema_hash,
            "schema_ref": str(report.schema_ref),
            "config": {
                "profile": self.config.profile,
                "fail_on_warnings": self.config.fail_on_warnings,
                "normalize": self.config.normalize,
            },
            "context": {
                "run_id": context.get("run_id"),
                "step_id": context.get("step_id"),
            },
            "result_valid": report.valid,
            "error_count": report.summary.error_count,
            "warning_count": report.summary.warning_count,
        }

        # Serialize deterministically
        json_str = json.dumps(provenance_data, sort_keys=True, default=str)

        return hashlib.sha256(json_str.encode()).hexdigest()

    def _hash_payload(self, payload: Union[Dict[str, Any], str]) -> str:
        """
        Calculate SHA-256 hash of payload.

        Args:
            payload: The payload to hash.

        Returns:
            SHA-256 hex digest.
        """
        if isinstance(payload, str):
            return hashlib.sha256(payload.encode()).hexdigest()

        json_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _check_fail_conditions(
        self,
        output: ValidateStepOutput,
        report: ValidationReport,
    ) -> None:
        """
        Check if the step should fail based on configuration.

        Args:
            output: The step output.
            report: The validation report.

        Raises:
            ValidationFailedError: If fail conditions are met.
        """
        # Always fail on errors
        if output.error_count > 0:
            raise ValidationFailedError(
                message=f"Validation failed with {output.error_count} error(s)",
                error_count=output.error_count,
                warning_count=output.warning_count,
                findings=report.findings,
            )

        # Optionally fail on warnings
        if self.config.fail_on_warnings and output.warning_count > 0:
            raise ValidationFailedError(
                message=f"Validation failed with {output.warning_count} warning(s) "
                        f"(fail_on_warnings=True)",
                error_count=output.error_count,
                warning_count=output.warning_count,
                findings=report.findings,
            )

    def clear_cache(self) -> None:
        """
        Clear the validator's schema cache.

        Call this to force re-compilation of schemas on next validation.
        """
        if self._validator:
            self._validator.clear_cache()
            logger.debug("ValidateStep cache cleared")


# =============================================================================
# STEP REGISTRATION
# =============================================================================


def register_validate_step() -> None:
    """
    Register ValidateStep with the orchestrator step registry.

    This function should be called during orchestrator initialization
    to make the validate step type available for use in pipelines.

    Example:
        >>> from greenlang.orchestrator.steps import register_validate_step
        >>> register_validate_step()
    """
    # Import at runtime to avoid circular imports
    from greenlang.orchestrator.steps import register_step

    try:
        register_step(STEP_TYPE, ValidateStep)
        logger.info(f"Registered step type: {STEP_TYPE}")
    except ValueError:
        # Already registered, skip
        logger.debug(f"Step type {STEP_TYPE} already registered")


def create_validate_step(
    schema: str,
    profile: str = "standard",
    fail_on_warnings: bool = False,
    normalize: bool = True,
    **kwargs,
) -> ValidateStep:
    """
    Factory function to create a ValidateStep.

    Provides a convenient way to create a ValidateStep with
    common configuration options.

    Args:
        schema: GreenLang schema reference URI.
        profile: Validation profile (strict, standard, permissive).
        fail_on_warnings: Whether to fail on warnings.
        normalize: Whether to normalize the payload.
        **kwargs: Additional configuration options.

    Returns:
        Configured ValidateStep instance.

    Example:
        >>> step = create_validate_step(
        ...     schema="gl://schemas/activity@1.0.0",
        ...     profile="strict",
        ...     fail_on_warnings=True,
        ... )
    """
    config = ValidateStepConfig(
        schema_uri=schema,
        profile=profile,
        fail_on_warnings=fail_on_warnings,
        normalize=normalize,
        **kwargs,
    )
    return ValidateStep(config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================


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
    "STEP_TYPE",
    "STEP_VERSION",
    "ERROR_CODES",
]

__version__ = STEP_VERSION
