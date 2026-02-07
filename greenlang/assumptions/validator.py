# -*- coding: utf-8 -*-
"""
Assumption Validator - AGENT-FOUND-004: Assumptions Registry

Validates assumption values against data type constraints and
user-defined validation rules. Supports min/max range checks,
allowed value lists, regex patterns, and custom validator functions.

Zero-Hallucination Guarantees:
    - All validation is deterministic and rule-based
    - No LLM or ML models involved in validation logic
    - Complete audit of which rules were checked

Example:
    >>> from greenlang.assumptions.validator import AssumptionValidator
    >>> validator = AssumptionValidator()
    >>> result = validator.validate_value("float", 10.5, [rule1, rule2])
    >>> print(result.is_valid)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
Status: Production Ready
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from greenlang.assumptions.models import (
    Assumption,
    AssumptionDataType,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
)
from greenlang.assumptions.metrics import (
    record_validation,
    record_validation_failure,
)

logger = logging.getLogger(__name__)


class AssumptionValidator:
    """Validates assumption values against type constraints and rules.

    Supports deterministic validation through:
    - Data type checking (float, int, string, boolean, percentage, ratio, etc.)
    - Min/max range validation
    - Allowed values lists
    - Regex pattern matching for strings
    - Custom validator functions registered by name

    Attributes:
        _custom_validators: Registry of named custom validator functions.

    Example:
        >>> validator = AssumptionValidator()
        >>> validator.register_custom_validator("positive", lambda v: v > 0)
        >>> result = validator.validate_value("float", -1.0, [rule_with_custom])
        >>> assert not result.is_valid
    """

    def __init__(self) -> None:
        """Initialize AssumptionValidator."""
        self._custom_validators: Dict[str, Callable[[Any], bool]] = {}
        logger.info("AssumptionValidator initialized")

    def validate(
        self,
        assumption: Assumption,
        value: Optional[Any] = None,
    ) -> ValidationResult:
        """Validate an assumption's current or provided value.

        Args:
            assumption: The assumption to validate.
            value: Optional value to validate. Uses current_value if None.

        Returns:
            ValidationResult with errors, warnings, and info.
        """
        val = value if value is not None else assumption.current_value
        return self.validate_value(
            data_type=assumption.data_type.value
            if isinstance(assumption.data_type, AssumptionDataType)
            else assumption.data_type,
            value=val,
            rules=assumption.validation_rules,
        )

    def validate_value(
        self,
        data_type: str,
        value: Any,
        rules: Optional[List[ValidationRule]] = None,
    ) -> ValidationResult:
        """Validate a value against data type and rules.

        Args:
            data_type: Expected data type string (e.g., "float", "integer").
            value: The value to validate.
            rules: Optional list of validation rules to apply.

        Returns:
            ValidationResult with errors, warnings, and info.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: List[str] = []
        rules_checked: List[str] = []

        # Step 1: Check data type
        dt = AssumptionDataType(data_type) if isinstance(data_type, str) else data_type
        if not self.check_data_type(dt, value):
            errors.append(
                f"Value type does not match expected {data_type}"
            )

        # Step 2: Check rules
        for rule in (rules or []):
            rules_checked.append(rule.rule_id)
            self._check_rule(rule, value, errors, warnings, info)

        # Record metrics
        is_valid = len(errors) == 0
        record_validation("pass" if is_valid else "fail")
        if not is_valid:
            for rule_id in rules_checked:
                record_validation_failure(rule_id)

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            rules_checked=rules_checked,
        )

    def register_custom_validator(
        self,
        name: str,
        func: Callable[[Any], bool],
    ) -> None:
        """Register a custom validation function by name.

        Args:
            name: Name to reference the validator in rules.
            func: Function that takes a value and returns True if valid.
        """
        self._custom_validators[name] = func
        logger.info("Registered custom validator: %s", name)

    def check_data_type(
        self,
        data_type: AssumptionDataType,
        value: Any,
    ) -> bool:
        """Check if a value matches the expected data type.

        Args:
            data_type: Expected data type.
            value: Value to check.

        Returns:
            True if value matches the data type.
        """
        type_checks: Dict[AssumptionDataType, Callable[[Any], bool]] = {
            AssumptionDataType.FLOAT: lambda v: isinstance(v, (int, float)),
            AssumptionDataType.INTEGER: lambda v: (
                isinstance(v, int)
                or (isinstance(v, float) and v.is_integer())
            ),
            AssumptionDataType.STRING: lambda v: isinstance(v, str),
            AssumptionDataType.BOOLEAN: lambda v: isinstance(v, bool),
            AssumptionDataType.PERCENTAGE: lambda v: (
                isinstance(v, (int, float)) and 0 <= v <= 100
            ),
            AssumptionDataType.RATIO: lambda v: (
                isinstance(v, (int, float)) and 0 <= v <= 1
            ),
            AssumptionDataType.DATE: lambda v: isinstance(v, (str, datetime)),
            AssumptionDataType.LIST_FLOAT: lambda v: (
                isinstance(v, list)
                and all(isinstance(x, (int, float)) for x in v)
            ),
            AssumptionDataType.LIST_STRING: lambda v: (
                isinstance(v, list)
                and all(isinstance(x, str) for x in v)
            ),
            AssumptionDataType.DICT: lambda v: isinstance(v, dict),
        }

        check_func = type_checks.get(data_type)
        if check_func:
            return check_func(value)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_rule(
        self,
        rule: ValidationRule,
        value: Any,
        errors: List[str],
        warnings: List[str],
        info: List[str],
    ) -> None:
        """Apply a single validation rule to a value.

        Args:
            rule: The validation rule to check.
            value: The value to validate.
            errors: Accumulator for error messages.
            warnings: Accumulator for warning messages.
            info: Accumulator for info messages.
        """
        try:
            self._check_min_value(rule, value, errors, warnings, info)
            self._check_max_value(rule, value, errors, warnings, info)
            self._check_allowed_values(rule, value, errors, warnings, info)
            self._check_regex_pattern(rule, value, errors, warnings, info)
            self._check_custom_validator(rule, value, errors, warnings, info)
        except Exception as e:
            errors.append(f"Validation rule {rule.rule_id} failed: {str(e)}")

    def _check_min_value(
        self,
        rule: ValidationRule,
        value: Any,
        errors: List[str],
        warnings: List[str],
        info: List[str],
    ) -> None:
        """Check min_value constraint."""
        if rule.min_value is not None and isinstance(value, (int, float)):
            if value < rule.min_value:
                msg = f"Value {value} is below minimum {rule.min_value}"
                self._append_by_severity(rule.severity, msg, errors, warnings, info)

    def _check_max_value(
        self,
        rule: ValidationRule,
        value: Any,
        errors: List[str],
        warnings: List[str],
        info: List[str],
    ) -> None:
        """Check max_value constraint."""
        if rule.max_value is not None and isinstance(value, (int, float)):
            if value > rule.max_value:
                msg = f"Value {value} is above maximum {rule.max_value}"
                self._append_by_severity(rule.severity, msg, errors, warnings, info)

    def _check_allowed_values(
        self,
        rule: ValidationRule,
        value: Any,
        errors: List[str],
        warnings: List[str],
        info: List[str],
    ) -> None:
        """Check allowed_values constraint."""
        if rule.allowed_values is not None:
            if value not in rule.allowed_values:
                msg = f"Value {value} is not in allowed values: {rule.allowed_values}"
                self._append_by_severity(rule.severity, msg, errors, warnings, info)

    def _check_regex_pattern(
        self,
        rule: ValidationRule,
        value: Any,
        errors: List[str],
        warnings: List[str],
        info: List[str],
    ) -> None:
        """Check regex_pattern constraint."""
        if rule.regex_pattern and isinstance(value, str):
            if not re.match(rule.regex_pattern, value):
                msg = f"Value '{value}' does not match pattern {rule.regex_pattern}"
                self._append_by_severity(rule.severity, msg, errors, warnings, info)

    def _check_custom_validator(
        self,
        rule: ValidationRule,
        value: Any,
        errors: List[str],
        warnings: List[str],
        info: List[str],
    ) -> None:
        """Check custom_validator constraint."""
        if rule.custom_validator and rule.custom_validator in self._custom_validators:
            validator_func = self._custom_validators[rule.custom_validator]
            if not validator_func(value):
                msg = (
                    f"Custom validation '{rule.custom_validator}' "
                    f"failed for value {value}"
                )
                self._append_by_severity(rule.severity, msg, errors, warnings, info)

    @staticmethod
    def _append_by_severity(
        severity: ValidationSeverity,
        msg: str,
        errors: List[str],
        warnings: List[str],
        info: List[str],
    ) -> None:
        """Append message to the appropriate list based on severity.

        Args:
            severity: The severity level.
            msg: The message to append.
            errors: Error message list.
            warnings: Warning message list.
            info: Info message list.
        """
        if severity == ValidationSeverity.ERROR:
            errors.append(msg)
        elif severity == ValidationSeverity.WARNING:
            warnings.append(msg)
        else:
            info.append(msg)


__all__ = [
    "AssumptionValidator",
]
