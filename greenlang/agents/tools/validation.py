# -*- coding: utf-8 -*-
"""
GreenLang Tool Input Validation Framework
==========================================

Production-grade input validation system for tool security.

Features:
- Comprehensive validation rules (range, type, enum, regex, custom)
- Composite validators for complex validation logic
- Input sanitization and normalization
- Detailed error and warning reporting
- Thread-safe operation
- Minimal performance overhead

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
from enum import Enum

logger = logging.getLogger(__name__)


# ==============================================================================
# Validation Result
# ==============================================================================

@dataclass
class ValidationResult:
    """
    Result of validation with errors, warnings, and sanitized values.

    Attributes:
        valid: True if validation passed (no errors)
        errors: List of validation error messages
        warnings: List of validation warning messages
        sanitized_value: The input value after sanitization/normalization
        metadata: Additional validation metadata
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    def merge(self, other: ValidationResult) -> None:
        """
        Merge another validation result into this one.

        Args:
            other: Another ValidationResult to merge
        """
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if other.errors:
            self.valid = False

        # Merge metadata
        self.metadata.update(other.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }

        if self.metadata:
            result["metadata"] = self.metadata

        return result


# ==============================================================================
# Base Validation Rule
# ==============================================================================

class ValidationRule(ABC):
    """
    Abstract base class for validation rules.

    All validators must implement the validate() method which checks
    a value against specific criteria and returns a ValidationResult.
    """

    def __init__(self, error_message: Optional[str] = None):
        """
        Initialize validation rule.

        Args:
            error_message: Custom error message (optional)
        """
        self.error_message = error_message
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a value.

        Args:
            value: Value to validate
            context: Additional context for validation (e.g., other parameter values)

        Returns:
            ValidationResult with validation status and details
        """
        pass

    def _create_result(
        self,
        valid: bool,
        value: Any,
        error: Optional[str] = None,
        warning: Optional[str] = None
    ) -> ValidationResult:
        """Helper to create ValidationResult."""
        result = ValidationResult(valid=valid, sanitized_value=value)

        if error:
            result.add_error(self.error_message or error)

        if warning:
            result.add_warning(warning)

        return result


# ==============================================================================
# Range Validator
# ==============================================================================

class RangeValidator(ValidationRule):
    """
    Validate that a numeric value is within a specified range.

    Example:
        >>> validator = RangeValidator(min_value=0, max_value=100)
        >>> result = validator.validate(50)
        >>> assert result.valid == True
    """

    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Initialize range validator.

        Args:
            min_value: Minimum allowed value (None = no minimum)
            max_value: Maximum allowed value (None = no maximum)
            min_inclusive: Whether minimum is inclusive (default: True)
            max_inclusive: Whether maximum is inclusive (default: True)
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.min_value = min_value
        self.max_value = max_value
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate that value is within range."""
        # Type check
        if not isinstance(value, (int, float)):
            return self._create_result(
                False,
                value,
                error=f"Value must be numeric, got {type(value).__name__}"
            )

        # Check minimum
        if self.min_value is not None:
            if self.min_inclusive:
                if value < self.min_value:
                    return self._create_result(
                        False,
                        value,
                        error=f"Value {value} is below minimum {self.min_value}"
                    )
            else:
                if value <= self.min_value:
                    return self._create_result(
                        False,
                        value,
                        error=f"Value {value} must be greater than {self.min_value}"
                    )

        # Check maximum
        if self.max_value is not None:
            if self.max_inclusive:
                if value > self.max_value:
                    return self._create_result(
                        False,
                        value,
                        error=f"Value {value} exceeds maximum {self.max_value}"
                    )
            else:
                if value >= self.max_value:
                    return self._create_result(
                        False,
                        value,
                        error=f"Value {value} must be less than {self.max_value}"
                    )

        return self._create_result(True, value)


# ==============================================================================
# Type Validator
# ==============================================================================

class TypeValidator(ValidationRule):
    """
    Validate that a value is of the expected type.

    Example:
        >>> validator = TypeValidator(expected_type=int)
        >>> result = validator.validate(42)
        >>> assert result.valid == True
    """

    def __init__(
        self,
        expected_type: Union[type, tuple],
        coerce: bool = False,
        error_message: Optional[str] = None
    ):
        """
        Initialize type validator.

        Args:
            expected_type: Expected type or tuple of types
            coerce: Attempt to coerce to expected type (default: False)
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.expected_type = expected_type
        self.coerce = coerce

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate value type."""
        # Check type
        if isinstance(value, self.expected_type):
            return self._create_result(True, value)

        # Try coercion if enabled
        if self.coerce:
            try:
                # Handle tuple of types - try first type
                target_type = (
                    self.expected_type[0]
                    if isinstance(self.expected_type, tuple)
                    else self.expected_type
                )

                coerced_value = target_type(value)
                return self._create_result(
                    True,
                    coerced_value,
                    warning=f"Value coerced from {type(value).__name__} to {target_type.__name__}"
                )
            except (ValueError, TypeError) as e:
                return self._create_result(
                    False,
                    value,
                    error=f"Cannot coerce {type(value).__name__} to {target_type.__name__}: {e}"
                )

        # Type mismatch
        expected_name = (
            " or ".join(t.__name__ for t in self.expected_type)
            if isinstance(self.expected_type, tuple)
            else self.expected_type.__name__
        )

        return self._create_result(
            False,
            value,
            error=f"Expected type {expected_name}, got {type(value).__name__}"
        )


# ==============================================================================
# Enum Validator
# ==============================================================================

class EnumValidator(ValidationRule):
    """
    Validate that a value is one of a set of allowed values.

    Example:
        >>> validator = EnumValidator(allowed_values=["red", "green", "blue"])
        >>> result = validator.validate("red")
        >>> assert result.valid == True
    """

    def __init__(
        self,
        allowed_values: Union[List[Any], Set[Any]],
        case_sensitive: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Initialize enum validator.

        Args:
            allowed_values: Set or list of allowed values
            case_sensitive: Whether string comparison is case-sensitive
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.allowed_values = set(allowed_values)
        self.case_sensitive = case_sensitive

        # Create lowercase mapping if case-insensitive
        if not case_sensitive:
            self.lowercase_map = {
                str(v).lower(): v for v in allowed_values if isinstance(v, str)
            }

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate value is in allowed set."""
        # Direct match
        if value in self.allowed_values:
            return self._create_result(True, value)

        # Case-insensitive string match
        if not self.case_sensitive and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in self.lowercase_map:
                normalized_value = self.lowercase_map[lower_value]
                return self._create_result(
                    True,
                    normalized_value,
                    warning=f"Value normalized from '{value}' to '{normalized_value}'"
                )

        # Value not allowed
        return self._create_result(
            False,
            value,
            error=f"Value '{value}' not in allowed values: {sorted(self.allowed_values)}"
        )


# ==============================================================================
# Regex Validator
# ==============================================================================

class RegexValidator(ValidationRule):
    """
    Validate that a string matches a regular expression pattern.

    Example:
        >>> validator = RegexValidator(pattern=r'^[A-Z]{2}[0-9]{4}$')
        >>> result = validator.validate("AB1234")
        >>> assert result.valid == True
    """

    def __init__(
        self,
        pattern: str,
        flags: int = 0,
        error_message: Optional[str] = None
    ):
        """
        Initialize regex validator.

        Args:
            pattern: Regular expression pattern
            flags: Regex flags (e.g., re.IGNORECASE)
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.pattern = pattern
        self.flags = flags
        self.compiled_pattern = re.compile(pattern, flags)

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate value matches regex pattern."""
        # Convert to string
        if not isinstance(value, str):
            return self._create_result(
                False,
                value,
                error=f"Value must be a string for regex validation, got {type(value).__name__}"
            )

        # Check pattern match
        if self.compiled_pattern.match(value):
            return self._create_result(True, value)

        return self._create_result(
            False,
            value,
            error=f"Value '{value}' does not match pattern '{self.pattern}'"
        )


# ==============================================================================
# Custom Validator
# ==============================================================================

class CustomValidator(ValidationRule):
    """
    Validate using a custom validation function.

    Example:
        >>> def is_even(value):
        ...     return value % 2 == 0, "Value must be even"
        >>> validator = CustomValidator(validation_fn=is_even)
        >>> result = validator.validate(4)
        >>> assert result.valid == True
    """

    def __init__(
        self,
        validation_fn: Callable[[Any, Optional[Dict[str, Any]]], Union[bool, tuple]],
        error_message: Optional[str] = None
    ):
        """
        Initialize custom validator.

        Args:
            validation_fn: Function that takes (value, context) and returns:
                - bool: True if valid, False otherwise
                - tuple: (is_valid, error_message) for detailed errors
            error_message: Default error message if not provided by function
        """
        super().__init__(error_message)
        self.validation_fn = validation_fn

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate using custom function."""
        try:
            result = self.validation_fn(value, context)

            # Handle bool return
            if isinstance(result, bool):
                if result:
                    return self._create_result(True, value)
                else:
                    return self._create_result(
                        False,
                        value,
                        error=self.error_message or "Custom validation failed"
                    )

            # Handle tuple return (is_valid, message)
            elif isinstance(result, tuple) and len(result) >= 2:
                is_valid, message = result[0], result[1]
                if is_valid:
                    return self._create_result(True, value)
                else:
                    return self._create_result(
                        False,
                        value,
                        error=self.error_message or message
                    )

            else:
                return self._create_result(
                    False,
                    value,
                    error="Custom validation function returned invalid result"
                )

        except Exception as e:
            self.logger.error(f"Custom validation failed: {e}", exc_info=True)
            return self._create_result(
                False,
                value,
                error=f"Custom validation raised exception: {e}"
            )


# ==============================================================================
# Composite Validator
# ==============================================================================

class CompositeValidator(ValidationRule):
    """
    Combine multiple validators with AND or OR logic.

    Example:
        >>> validators = [
        ...     TypeValidator(int),
        ...     RangeValidator(min_value=0, max_value=100)
        ... ]
        >>> validator = CompositeValidator(validators, mode="all")
        >>> result = validator.validate(50)
        >>> assert result.valid == True
    """

    def __init__(
        self,
        validators: List[ValidationRule],
        mode: str = "all",
        error_message: Optional[str] = None
    ):
        """
        Initialize composite validator.

        Args:
            validators: List of validators to combine
            mode: Combination mode:
                - "all": All validators must pass (AND logic)
                - "any": At least one validator must pass (OR logic)
            error_message: Custom error message
        """
        super().__init__(error_message)

        if mode not in ("all", "any"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'all' or 'any'")

        self.validators = validators
        self.mode = mode

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate using composite logic."""
        if self.mode == "all":
            return self._validate_all(value, context)
        else:
            return self._validate_any(value, context)

    def _validate_all(self, value: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """All validators must pass (AND logic)."""
        final_result = ValidationResult(valid=True, sanitized_value=value)
        current_value = value

        for validator in self.validators:
            result = validator.validate(current_value, context)

            # Merge errors and warnings
            final_result.merge(result)

            # Update current value with sanitized value
            if result.sanitized_value is not None:
                current_value = result.sanitized_value

            # Short-circuit on first error
            if not result.valid:
                final_result.valid = False
                final_result.sanitized_value = value  # Keep original on failure
                return final_result

        # All passed - use final sanitized value
        final_result.sanitized_value = current_value
        return final_result

    def _validate_any(self, value: Any, context: Optional[Dict[str, Any]]) -> ValidationResult:
        """At least one validator must pass (OR logic)."""
        all_errors = []
        all_warnings = []

        for validator in self.validators:
            result = validator.validate(value, context)

            if result.valid:
                # Found a passing validator
                return result

            # Collect errors and warnings
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        # None passed
        return ValidationResult(
            valid=False,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_value=value
        )
