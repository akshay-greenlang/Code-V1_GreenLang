# -*- coding: utf-8 -*-
"""
Validation Framework for GreenLang Agents.

This module provides validation utilities for ensuring data integrity,
thermodynamic consistency, and business rule compliance across GreenLang agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, TypeVar, Generic
import logging
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ValidationSeverity(str, Enum):
    """Validation result severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Validation categories for classification."""
    THERMODYNAMIC = "thermodynamic"
    ECONOMIC = "economic"
    OPERATIONAL = "operational"
    DATA_QUALITY = "data_quality"
    COMPLIANCE = "compliance"
    SAFETY = "safety"
    CONFIGURATION = "configuration"


class ValidationError(Exception):
    """
    Custom exception for validation failures.

    Attributes:
        message: Error message
        code: Error code for programmatic handling
        field: Field that failed validation
        value: Value that failed validation
        severity: Error severity level
        category: Validation category
    """

    def __init__(
        self,
        message: str,
        code: str = "VALIDATION_ERROR",
        field: Optional[str] = None,
        value: Any = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        category: ValidationCategory = ValidationCategory.DATA_QUALITY,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.field = field
        self.value = value
        self.severity = severity
        self.category = category
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "message": self.message,
            "code": self.code,
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationResult:
    """
    Container for validation results.

    Attributes:
        is_valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        info: List of informational messages
        metadata: Additional validation metadata
        validated_at: Timestamp of validation
        validator_id: Identifier of the validator
    """
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    validator_id: Optional[str] = None

    def add_error(self, error: ValidationError):
        """Add an error to the result."""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.is_valid = False
        elif error.severity == ValidationSeverity.CRITICAL:
            self.errors.append(error)
            self.is_valid = False
        elif error.severity == ValidationSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info.append(error)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.metadata.update(other.metadata)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
            "metadata": self.metadata,
            "validated_at": self.validated_at.isoformat(),
            "validator_id": self.validator_id,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }

    @classmethod
    def success(cls, message: str = "Validation passed", metadata: Optional[Dict[str, Any]] = None) -> "ValidationResult":
        """Create a successful validation result."""
        result = cls(is_valid=True, metadata=metadata or {})
        result.add_error(ValidationError(
            message=message,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.DATA_QUALITY,
        ))
        return result

    @classmethod
    def failure(cls, message: str, code: str = "VALIDATION_FAILED", **kwargs) -> "ValidationResult":
        """Create a failed validation result."""
        result = cls(is_valid=False)
        result.add_error(ValidationError(
            message=message,
            code=code,
            severity=ValidationSeverity.ERROR,
            **kwargs
        ))
        return result


@dataclass
class ValidationContext:
    """
    Context for validation operations.

    Attributes:
        strict_mode: Whether to use strict validation
        max_errors: Maximum errors before stopping
        categories: Categories to validate
        skip_categories: Categories to skip
        custom_rules: Custom validation rules
    """
    strict_mode: bool = False
    max_errors: int = 100
    categories: Optional[List[ValidationCategory]] = None
    skip_categories: Optional[List[ValidationCategory]] = None
    custom_rules: Dict[str, Callable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_validate_category(self, category: ValidationCategory) -> bool:
        """Check if a category should be validated."""
        if self.skip_categories and category in self.skip_categories:
            return False
        if self.categories:
            return category in self.categories
        return True


class Validator(Generic[T]):
    """
    Generic validator class for data validation.

    This class provides a fluent interface for building validation chains
    and executing validations against data.

    Example:
        >>> validator = Validator("temperature")
        >>> validator.required().min_value(0).max_value(1000)
        >>> result = validator.validate(data)
    """

    def __init__(self, name: str, category: ValidationCategory = ValidationCategory.DATA_QUALITY):
        """
        Initialize the validator.

        Args:
            name: Name of the validator
            category: Default validation category
        """
        self.name = name
        self.category = category
        self._rules: List[Dict[str, Any]] = []
        self._context: Optional[ValidationContext] = None

    def with_context(self, context: ValidationContext) -> "Validator[T]":
        """Set validation context."""
        self._context = context
        return self

    def required(self, message: Optional[str] = None) -> "Validator[T]":
        """Add required validation rule."""
        self._rules.append({
            "type": "required",
            "message": message or f"{self.name} is required",
        })
        return self

    def min_value(self, min_val: float, message: Optional[str] = None) -> "Validator[T]":
        """Add minimum value validation rule."""
        self._rules.append({
            "type": "min_value",
            "value": min_val,
            "message": message or f"{self.name} must be at least {min_val}",
        })
        return self

    def max_value(self, max_val: float, message: Optional[str] = None) -> "Validator[T]":
        """Add maximum value validation rule."""
        self._rules.append({
            "type": "max_value",
            "value": max_val,
            "message": message or f"{self.name} must be at most {max_val}",
        })
        return self

    def range(self, min_val: float, max_val: float, message: Optional[str] = None) -> "Validator[T]":
        """Add range validation rule."""
        self._rules.append({
            "type": "range",
            "min": min_val,
            "max": max_val,
            "message": message or f"{self.name} must be between {min_val} and {max_val}",
        })
        return self

    def positive(self, message: Optional[str] = None) -> "Validator[T]":
        """Add positive number validation rule."""
        self._rules.append({
            "type": "positive",
            "message": message or f"{self.name} must be positive",
        })
        return self

    def non_negative(self, message: Optional[str] = None) -> "Validator[T]":
        """Add non-negative validation rule."""
        self._rules.append({
            "type": "non_negative",
            "message": message or f"{self.name} must be non-negative",
        })
        return self

    def custom(self, func: Callable[[Any], bool], message: str, code: str = "CUSTOM_VALIDATION") -> "Validator[T]":
        """Add custom validation rule."""
        self._rules.append({
            "type": "custom",
            "func": func,
            "message": message,
            "code": code,
        })
        return self

    def validate(self, value: Any) -> ValidationResult:
        """
        Execute validation against a value.

        Args:
            value: Value to validate

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(validator_id=self.name)

        for rule in self._rules:
            try:
                rule_result = self._apply_rule(rule, value)
                if not rule_result:
                    result.add_error(ValidationError(
                        message=rule.get("message", "Validation failed"),
                        code=rule.get("code", f"VALIDATION_{rule['type'].upper()}"),
                        field=self.name,
                        value=value,
                        category=self.category,
                    ))

                    # Check if we should stop on max errors
                    if self._context and len(result.errors) >= self._context.max_errors:
                        break

            except Exception as e:
                result.add_error(ValidationError(
                    message=f"Validation rule failed: {str(e)}",
                    code="VALIDATION_EXCEPTION",
                    field=self.name,
                    value=value,
                    category=self.category,
                ))

        return result

    def _apply_rule(self, rule: Dict[str, Any], value: Any) -> bool:
        """Apply a single validation rule."""
        rule_type = rule["type"]

        if rule_type == "required":
            return value is not None

        if value is None:
            return True  # Skip other validations if value is None

        if rule_type == "min_value":
            return value >= rule["value"]

        if rule_type == "max_value":
            return value <= rule["value"]

        if rule_type == "range":
            return rule["min"] <= value <= rule["max"]

        if rule_type == "positive":
            return value > 0

        if rule_type == "non_negative":
            return value >= 0

        if rule_type == "custom":
            return rule["func"](value)

        return True


def validate(validator: Validator, raise_on_error: bool = True):
    """
    Decorator for validating function inputs.

    Args:
        validator: Validator to apply
        raise_on_error: Whether to raise exception on validation failure

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract value from args/kwargs based on validator name
            value = kwargs.get(validator.name)
            if value is None and args:
                value = args[0]

            result = validator.validate(value)
            if not result.is_valid:
                if raise_on_error:
                    raise result.errors[0] if result.errors else ValidationError("Validation failed")
                return result

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Thermodynamic validation helpers
def validate_temperature(value: float, unit: str = "C", field_name: str = "temperature") -> ValidationResult:
    """Validate temperature value."""
    validator = Validator(field_name, ValidationCategory.THERMODYNAMIC)
    if unit == "C":
        validator.min_value(-273.15, "Temperature cannot be below absolute zero")
    elif unit == "K":
        validator.min_value(0, "Temperature cannot be below 0 K")
    return validator.validate(value)


def validate_pressure(value: float, unit: str = "bar", field_name: str = "pressure") -> ValidationResult:
    """Validate pressure value."""
    validator = Validator(field_name, ValidationCategory.THERMODYNAMIC)
    validator.positive("Pressure must be positive")
    return validator.validate(value)


def validate_flow_rate(value: float, field_name: str = "flow_rate") -> ValidationResult:
    """Validate flow rate value."""
    validator = Validator(field_name, ValidationCategory.THERMODYNAMIC)
    validator.non_negative("Flow rate cannot be negative")
    return validator.validate(value)


def validate_efficiency(value: float, field_name: str = "efficiency") -> ValidationResult:
    """Validate efficiency value (0-1 range)."""
    validator = Validator(field_name, ValidationCategory.THERMODYNAMIC)
    validator.range(0.0, 1.0, "Efficiency must be between 0 and 1")
    return validator.validate(value)


__all__ = [
    'ValidationSeverity',
    'ValidationCategory',
    'ValidationError',
    'ValidationResult',
    'ValidationContext',
    'Validator',
    'validate',
    'validate_temperature',
    'validate_pressure',
    'validate_flow_rate',
    'validate_efficiency',
]
