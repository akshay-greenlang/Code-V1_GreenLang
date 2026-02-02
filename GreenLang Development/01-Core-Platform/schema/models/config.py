# -*- coding: utf-8 -*-
"""
Validation Configuration Models
===============================

Pydantic models for validation configuration and options.

This module provides:
- ValidationProfile: Enum for validation strictness levels
- CoercionPolicy: Enum for type coercion behavior
- UnknownFieldPolicy: Enum for handling unknown fields
- PatchLevel: Enum for fix suggestion safety thresholds
- ValidationOptions: Complete configuration for validation runs

Example:
    >>> options = ValidationOptions(
    ...     profile=ValidationProfile.STRICT,
    ...     normalize=True,
    ...     emit_patches=True
    ... )
    >>> print(options.profile)
    ValidationProfile.STRICT

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ValidationProfile(str, Enum):
    """
    Validation strictness profile.

    Defines how strictly validation rules are enforced:
    - STRICT: All warnings become errors, unknown fields rejected
    - STANDARD: Standard validation with warnings and errors
    - PERMISSIVE: Only critical errors, maximum flexibility

    Example:
        >>> profile = ValidationProfile.STRICT
        >>> print(profile.value)
        strict
    """

    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"

    def is_strict(self) -> bool:
        """Check if this is strict mode."""
        return self == ValidationProfile.STRICT

    def is_permissive(self) -> bool:
        """Check if this is permissive mode."""
        return self == ValidationProfile.PERMISSIVE


class CoercionPolicy(str, Enum):
    """
    Type coercion policy during normalization.

    Controls how the normalizer handles type mismatches:
    - OFF: No automatic type coercion
    - SAFE: Only safe, exact coercions (e.g., "42" -> 42)
    - AGGRESSIVE: More permissive coercions (e.g., "true" -> True)

    Example:
        >>> policy = CoercionPolicy.SAFE
        >>> print(policy.value)
        safe
    """

    OFF = "off"
    SAFE = "safe"
    AGGRESSIVE = "aggressive"

    def allows_coercion(self) -> bool:
        """Check if any coercion is allowed."""
        return self != CoercionPolicy.OFF

    def is_aggressive(self) -> bool:
        """Check if aggressive coercion is enabled."""
        return self == CoercionPolicy.AGGRESSIVE


class UnknownFieldPolicy(str, Enum):
    """
    Policy for handling unknown fields in payloads.

    Controls behavior when fields not defined in schema are encountered:
    - ERROR: Unknown fields cause validation to fail
    - WARN: Unknown fields generate warnings but don't fail
    - IGNORE: Unknown fields are silently ignored

    Example:
        >>> policy = UnknownFieldPolicy.WARN
        >>> print(policy.value)
        warn
    """

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"

    def is_error(self) -> bool:
        """Check if unknown fields cause errors."""
        return self == UnknownFieldPolicy.ERROR

    def is_silent(self) -> bool:
        """Check if unknown fields are silently ignored."""
        return self == UnknownFieldPolicy.IGNORE


class PatchLevel(str, Enum):
    """
    Safety level threshold for fix suggestions.

    Controls which fix suggestions are included in the response:
    - SAFE: Only include suggestions that are safe to auto-apply
    - NEEDS_REVIEW: Include suggestions that may need human review
    - UNSAFE: Include all suggestions, including potentially unsafe ones

    Example:
        >>> level = PatchLevel.SAFE
        >>> print(level.value)
        safe
    """

    SAFE = "safe"
    NEEDS_REVIEW = "needs_review"
    UNSAFE = "unsafe"

    def allows_review(self) -> bool:
        """Check if needs_review level suggestions are allowed."""
        return self in (PatchLevel.NEEDS_REVIEW, PatchLevel.UNSAFE)

    def allows_unsafe(self) -> bool:
        """Check if unsafe level suggestions are allowed."""
        return self == PatchLevel.UNSAFE


class ValidationOptions(BaseModel):
    """
    Configuration options for schema validation.

    This model encapsulates all configurable aspects of a validation run,
    including strictness, normalization behavior, error limits, and more.

    Attributes:
        profile: Validation strictness profile (strict/standard/permissive).
        normalize: Whether to normalize the payload to canonical form.
        emit_patches: Whether to generate fix suggestions for validation errors.
        patch_level: Maximum safety level for fix suggestions.
        max_errors: Maximum number of errors to report (0 = unlimited).
        fail_fast: Stop validation at first error.
        unit_system: Unit system for conversions (SI, Imperial, etc.).
        unknown_field_policy: How to handle fields not in schema.
        coercion_policy: How to handle type mismatches during normalization.
        timezone: Default timezone for datetime normalization.
        locale: Locale for number/date formatting.

    Example:
        >>> options = ValidationOptions(
        ...     profile=ValidationProfile.STRICT,
        ...     normalize=True,
        ...     max_errors=50
        ... )
        >>> print(options.profile)
        ValidationProfile.STRICT
    """

    profile: ValidationProfile = Field(
        default=ValidationProfile.STANDARD,
        description="Validation strictness profile"
    )

    normalize: bool = Field(
        default=True,
        description="Whether to normalize payload to canonical form"
    )

    emit_patches: bool = Field(
        default=True,
        description="Whether to generate fix suggestions"
    )

    patch_level: PatchLevel = Field(
        default=PatchLevel.SAFE,
        description="Maximum safety level for fix suggestions"
    )

    max_errors: int = Field(
        default=100,
        ge=0,
        le=10000,
        description="Maximum errors to report (0 = unlimited)"
    )

    fail_fast: bool = Field(
        default=False,
        description="Stop validation at first error"
    )

    unit_system: str = Field(
        default="SI",
        min_length=1,
        max_length=32,
        description="Unit system for conversions (SI, Imperial)"
    )

    unknown_field_policy: UnknownFieldPolicy = Field(
        default=UnknownFieldPolicy.WARN,
        description="How to handle unknown fields"
    )

    coercion_policy: CoercionPolicy = Field(
        default=CoercionPolicy.SAFE,
        description="Type coercion policy during normalization"
    )

    timezone: Optional[str] = Field(
        default="UTC",
        max_length=64,
        description="Default timezone for datetime normalization"
    )

    locale: Optional[str] = Field(
        default="en_US",
        max_length=32,
        description="Locale for number/date formatting"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "profile": "standard",
                    "normalize": True,
                    "emit_patches": True,
                    "patch_level": "safe",
                    "max_errors": 100,
                    "fail_fast": False,
                    "unit_system": "SI",
                    "unknown_field_policy": "warn",
                    "coercion_policy": "safe"
                },
                {
                    "profile": "strict",
                    "normalize": True,
                    "emit_patches": True,
                    "patch_level": "needs_review",
                    "max_errors": 50,
                    "fail_fast": True,
                    "unit_system": "SI",
                    "unknown_field_policy": "error",
                    "coercion_policy": "off"
                }
            ]
        }
    }

    @field_validator("unit_system")
    @classmethod
    def validate_unit_system(cls, v: str) -> str:
        """
        Validate unit system is recognized.

        Args:
            v: The unit system value.

        Returns:
            The validated unit system (uppercase).

        Raises:
            ValueError: If unit system is not recognized.
        """
        valid_systems = {"SI", "IMPERIAL", "CGS", "CUSTOM"}
        v_upper = v.upper()
        if v_upper not in valid_systems:
            raise ValueError(
                f"Invalid unit_system '{v}'. Must be one of: {', '.join(valid_systems)}"
            )
        return v_upper

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate timezone format.

        Args:
            v: The timezone string (may be None).

        Returns:
            The validated timezone string.

        Note:
            Full timezone validation requires pytz or zoneinfo,
            this just does basic format checking.
        """
        if v is None:
            return v

        # Basic format validation
        # Valid formats: UTC, America/New_York, Europe/London, etc.
        if not v or len(v) > 64:
            raise ValueError(f"Invalid timezone '{v}'")

        return v

    @field_validator("locale")
    @classmethod
    def validate_locale(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate locale format.

        Args:
            v: The locale string (may be None).

        Returns:
            The validated locale string.
        """
        if v is None:
            return v

        # Basic format validation: xx or xx_XX
        import re
        if not re.match(r"^[a-z]{2}(_[A-Z]{2})?$", v):
            raise ValueError(
                f"Invalid locale '{v}'. Expected format: 'en' or 'en_US'"
            )

        return v

    def with_profile(self, profile: ValidationProfile) -> "ValidationOptions":
        """
        Create a copy with a different profile.

        Args:
            profile: The new validation profile.

        Returns:
            A new ValidationOptions with the specified profile.
        """
        return self.model_copy(update={"profile": profile})

    def with_strict_mode(self) -> "ValidationOptions":
        """
        Create a copy configured for strict mode.

        Returns:
            A new ValidationOptions with strict settings.
        """
        return self.model_copy(update={
            "profile": ValidationProfile.STRICT,
            "unknown_field_policy": UnknownFieldPolicy.ERROR,
            "coercion_policy": CoercionPolicy.OFF,
        })

    def with_permissive_mode(self) -> "ValidationOptions":
        """
        Create a copy configured for permissive mode.

        Returns:
            A new ValidationOptions with permissive settings.
        """
        return self.model_copy(update={
            "profile": ValidationProfile.PERMISSIVE,
            "unknown_field_policy": UnknownFieldPolicy.IGNORE,
            "coercion_policy": CoercionPolicy.AGGRESSIVE,
        })

    @classmethod
    def strict(cls) -> "ValidationOptions":
        """
        Create strict validation options.

        Returns:
            ValidationOptions configured for strict validation.

        Example:
            >>> options = ValidationOptions.strict()
            >>> print(options.profile)
            ValidationProfile.STRICT
        """
        return cls(
            profile=ValidationProfile.STRICT,
            unknown_field_policy=UnknownFieldPolicy.ERROR,
            coercion_policy=CoercionPolicy.OFF,
        )

    @classmethod
    def permissive(cls) -> "ValidationOptions":
        """
        Create permissive validation options.

        Returns:
            ValidationOptions configured for permissive validation.

        Example:
            >>> options = ValidationOptions.permissive()
            >>> print(options.profile)
            ValidationProfile.PERMISSIVE
        """
        return cls(
            profile=ValidationProfile.PERMISSIVE,
            unknown_field_policy=UnknownFieldPolicy.IGNORE,
            coercion_policy=CoercionPolicy.AGGRESSIVE,
        )

    @classmethod
    def default(cls) -> "ValidationOptions":
        """
        Create default validation options.

        Returns:
            ValidationOptions with standard defaults.

        Example:
            >>> options = ValidationOptions.default()
            >>> print(options.profile)
            ValidationProfile.STANDARD
        """
        return cls()
