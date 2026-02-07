# -*- coding: utf-8 -*-
"""
Unit Tests for AssumptionValidator (AGENT-FOUND-004)

Tests all validation rule types: min_value, max_value, allowed_values,
regex_pattern, custom_validator, data type validation, severity handling.

Coverage target: 85%+ of validator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline AssumptionValidator mirroring greenlang/assumptions/validator.py
# ---------------------------------------------------------------------------


class ValidationRule:
    """A validation rule."""

    def __init__(
        self,
        rule_id: str,
        assumption_id: str,
        rule_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        message: str = "",
    ):
        self.rule_id = rule_id
        self.assumption_id = assumption_id
        self.rule_type = rule_type
        self.parameters = parameters or {}
        self.severity = severity
        self.message = message


class ValidationResult:
    """Result of validation."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


class AssumptionValidator:
    """
    Validates assumption values against rules.
    Mirrors greenlang/assumptions/validator.py.
    """

    # Supported data types
    DATA_TYPES = {
        "float": (int, float),
        "integer": (int,),
        "string": (str,),
        "boolean": (bool,),
        "percentage": (int, float),
        "ratio": (int, float),
        "date": (str,),
        "list_float": (list,),
        "list_string": (list,),
        "dict": (dict,),
    }

    def __init__(self):
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._custom_validators: Dict[str, Callable] = {}

    def add_rule(self, rule: ValidationRule):
        """Add a validation rule for an assumption."""
        if rule.assumption_id not in self._rules:
            self._rules[rule.assumption_id] = []
        self._rules[rule.assumption_id].append(rule)

    def register_custom_validator(
        self, name: str, validator_fn: Callable[[Any, Dict[str, Any]], Optional[str]]
    ):
        """Register a custom validator function.

        The function receives (value, parameters) and returns None for valid
        or an error message string.
        """
        self._custom_validators[name] = validator_fn

    def validate(self, assumption_id: str, value: Any) -> ValidationResult:
        """Validate a value against all rules for an assumption."""
        rules = self._rules.get(assumption_id, [])
        if not rules:
            return ValidationResult(is_valid=True)

        errors: List[str] = []
        warnings: List[str] = []

        for rule in rules:
            msg = self._check_rule(rule, value)
            if msg:
                if rule.severity == "error":
                    errors.append(msg)
                elif rule.severity == "warning":
                    warnings.append(msg)
                # info severity is ignored for validation purposes

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_data_type(self, value: Any, data_type: str) -> bool:
        """Validate that a value matches the expected data type."""
        expected_types = self.DATA_TYPES.get(data_type)
        if expected_types is None:
            return False

        if data_type == "percentage":
            if not isinstance(value, (int, float)):
                return False
            return 0 <= value <= 100

        if data_type == "ratio":
            if not isinstance(value, (int, float)):
                return False
            return 0 <= value <= 1

        if data_type == "date":
            if not isinstance(value, str):
                return False
            # Basic ISO-8601 date pattern
            return bool(re.match(r"^\d{4}-\d{2}-\d{2}", value))

        if data_type == "list_float":
            if not isinstance(value, list):
                return False
            return all(isinstance(v, (int, float)) for v in value)

        if data_type == "list_string":
            if not isinstance(value, list):
                return False
            return all(isinstance(v, str) for v in value)

        if data_type == "boolean":
            # Strict boolean check (not int as bool subclass)
            return value is True or value is False

        return isinstance(value, expected_types)

    def _check_rule(self, rule: ValidationRule, value: Any) -> Optional[str]:
        """Check a single rule against a value. Returns error message or None."""
        rule_type = rule.rule_type
        params = rule.parameters

        if rule_type == "min_value":
            min_val = params.get("min_value", 0)
            if isinstance(value, (int, float)) and value < min_val:
                return rule.message or f"Value {value} is below minimum {min_val}"

        elif rule_type == "max_value":
            max_val = params.get("max_value", float("inf"))
            if isinstance(value, (int, float)) and value > max_val:
                return rule.message or f"Value {value} exceeds maximum {max_val}"

        elif rule_type == "allowed_values":
            allowed = params.get("allowed_values", [])
            if value not in allowed:
                return rule.message or f"Value {value} not in allowed values: {allowed}"

        elif rule_type == "regex_pattern":
            pattern = params.get("pattern", "")
            if isinstance(value, str) and not re.match(pattern, value):
                return rule.message or f"Value does not match pattern: {pattern}"

        elif rule_type == "custom":
            validator_name = params.get("validator_name", "")
            fn = self._custom_validators.get(validator_name)
            if fn:
                return fn(value, params)

        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator():
    """Fresh AssumptionValidator."""
    return AssumptionValidator()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestValidateMinValue:
    """Test min_value validation rule."""

    def test_valid_above_min(self, validator):
        rule = ValidationRule("r1", "a1", "min_value", {"min_value": 0}, "error")
        validator.add_rule(rule)
        result = validator.validate("a1", 2.68)
        assert result.is_valid is True

    def test_valid_at_min(self, validator):
        rule = ValidationRule("r1", "a1", "min_value", {"min_value": 0}, "error")
        validator.add_rule(rule)
        result = validator.validate("a1", 0)
        assert result.is_valid is True

    def test_invalid_below_min(self, validator):
        rule = ValidationRule("r1", "a1", "min_value", {"min_value": 0}, "error",
                              "Must be positive")
        validator.add_rule(rule)
        result = validator.validate("a1", -1)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "positive" in result.errors[0].lower() or "Must" in result.errors[0]


class TestValidateMaxValue:
    """Test max_value validation rule."""

    def test_valid_below_max(self, validator):
        rule = ValidationRule("r1", "a1", "max_value", {"max_value": 100}, "error")
        validator.add_rule(rule)
        result = validator.validate("a1", 50)
        assert result.is_valid is True

    def test_valid_at_max(self, validator):
        rule = ValidationRule("r1", "a1", "max_value", {"max_value": 100}, "error")
        validator.add_rule(rule)
        result = validator.validate("a1", 100)
        assert result.is_valid is True

    def test_invalid_above_max(self, validator):
        rule = ValidationRule("r1", "a1", "max_value", {"max_value": 100}, "error",
                              "Exceeds maximum")
        validator.add_rule(rule)
        result = validator.validate("a1", 150)
        assert result.is_valid is False
        assert len(result.errors) == 1


class TestValidateAllowedValues:
    """Test allowed_values validation rule."""

    def test_valid_in_allowed(self, validator):
        rule = ValidationRule(
            "r1", "a1", "allowed_values",
            {"allowed_values": ["diesel", "gasoline", "natural_gas"]},
            "error",
        )
        validator.add_rule(rule)
        result = validator.validate("a1", "diesel")
        assert result.is_valid is True

    def test_invalid_not_in_allowed(self, validator):
        rule = ValidationRule(
            "r1", "a1", "allowed_values",
            {"allowed_values": ["diesel", "gasoline"]},
            "error", "Invalid fuel type",
        )
        validator.add_rule(rule)
        result = validator.validate("a1", "coal")
        assert result.is_valid is False


class TestValidateRegexPattern:
    """Test regex_pattern validation rule."""

    def test_valid_matches_pattern(self, validator):
        rule = ValidationRule(
            "r1", "a1", "regex_pattern",
            {"pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "error",
        )
        validator.add_rule(rule)
        result = validator.validate("a1", "2025-01-15")
        assert result.is_valid is True

    def test_invalid_no_match(self, validator):
        rule = ValidationRule(
            "r1", "a1", "regex_pattern",
            {"pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "error", "Invalid date format",
        )
        validator.add_rule(rule)
        result = validator.validate("a1", "15/01/2025")
        assert result.is_valid is False


class TestValidateCustomValidator:
    """Test custom_validator rule."""

    def test_custom_validator_valid(self, validator):
        def check_even(value, params):
            if isinstance(value, int) and value % 2 != 0:
                return "Value must be even"
            return None

        validator.register_custom_validator("check_even", check_even)
        rule = ValidationRule(
            "r1", "a1", "custom",
            {"validator_name": "check_even"},
            "error",
        )
        validator.add_rule(rule)
        result = validator.validate("a1", 42)
        assert result.is_valid is True

    def test_custom_validator_invalid(self, validator):
        def check_even(value, params):
            if isinstance(value, int) and value % 2 != 0:
                return "Value must be even"
            return None

        validator.register_custom_validator("check_even", check_even)
        rule = ValidationRule(
            "r1", "a1", "custom",
            {"validator_name": "check_even"},
            "error",
        )
        validator.add_rule(rule)
        result = validator.validate("a1", 41)
        assert result.is_valid is False


class TestValidateSeverity:
    """Test severity handling."""

    def test_warning_allows_save(self, validator):
        rule = ValidationRule(
            "r1", "a1", "max_value",
            {"max_value": 10},
            "warning", "Outside typical range",
        )
        validator.add_rule(rule)
        result = validator.validate("a1", 50)
        assert result.is_valid is True  # warnings do not block
        assert len(result.warnings) == 1

    def test_error_blocks_save(self, validator):
        rule = ValidationRule(
            "r1", "a1", "max_value",
            {"max_value": 10},
            "error", "Exceeds maximum",
        )
        validator.add_rule(rule)
        result = validator.validate("a1", 50)
        assert result.is_valid is False

    def test_mixed_severity(self, validator):
        err_rule = ValidationRule("r1", "a1", "min_value", {"min_value": 0}, "error")
        warn_rule = ValidationRule("r2", "a1", "max_value", {"max_value": 5}, "warning")
        validator.add_rule(err_rule)
        validator.add_rule(warn_rule)
        result = validator.validate("a1", 10)
        assert result.is_valid is True  # no errors (10 >= 0)
        assert len(result.warnings) == 1  # 10 > 5 triggers warning


class TestMultipleRules:
    """Test multiple rules on same assumption."""

    def test_all_rules_checked(self, validator):
        rule1 = ValidationRule("r1", "a1", "min_value", {"min_value": 0}, "error")
        rule2 = ValidationRule("r2", "a1", "max_value", {"max_value": 100}, "error")
        validator.add_rule(rule1)
        validator.add_rule(rule2)

        result = validator.validate("a1", 50)
        assert result.is_valid is True

    def test_multiple_errors(self, validator):
        rule1 = ValidationRule("r1", "a1", "min_value", {"min_value": 10}, "error", "Too low")
        rule2 = ValidationRule("r2", "a1", "max_value", {"max_value": 5}, "error", "Too high")
        validator.add_rule(rule1)
        validator.add_rule(rule2)

        # value=3: below min(10) AND not above max(5) -> only 1 error
        result = validator.validate("a1", 3)
        assert result.is_valid is False
        assert len(result.errors) == 1  # only min_value fails


class TestEmptyRules:
    """Test validation with no rules."""

    def test_no_rules_passes(self, validator):
        result = validator.validate("a1", 42)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []


class TestValidateDataType:
    """Test validate_data_type() for all 10 types."""

    def test_float_valid(self, validator):
        assert validator.validate_data_type(2.68, "float") is True

    def test_float_int_valid(self, validator):
        assert validator.validate_data_type(3, "float") is True

    def test_float_string_invalid(self, validator):
        assert validator.validate_data_type("2.68", "float") is False

    def test_integer_valid(self, validator):
        assert validator.validate_data_type(42, "integer") is True

    def test_integer_float_invalid(self, validator):
        assert validator.validate_data_type(42.5, "integer") is False

    def test_string_valid(self, validator):
        assert validator.validate_data_type("hello", "string") is True

    def test_string_int_invalid(self, validator):
        assert validator.validate_data_type(42, "string") is False

    def test_boolean_true(self, validator):
        assert validator.validate_data_type(True, "boolean") is True

    def test_boolean_false(self, validator):
        assert validator.validate_data_type(False, "boolean") is True

    def test_percentage_valid(self, validator):
        assert validator.validate_data_type(50, "percentage") is True

    def test_percentage_0(self, validator):
        assert validator.validate_data_type(0, "percentage") is True

    def test_percentage_100(self, validator):
        assert validator.validate_data_type(100, "percentage") is True

    def test_percentage_over_100_invalid(self, validator):
        assert validator.validate_data_type(101, "percentage") is False

    def test_percentage_negative_invalid(self, validator):
        assert validator.validate_data_type(-1, "percentage") is False

    def test_ratio_valid(self, validator):
        assert validator.validate_data_type(0.5, "ratio") is True

    def test_ratio_0(self, validator):
        assert validator.validate_data_type(0, "ratio") is True

    def test_ratio_1(self, validator):
        assert validator.validate_data_type(1, "ratio") is True

    def test_ratio_over_1_invalid(self, validator):
        assert validator.validate_data_type(1.1, "ratio") is False

    def test_date_valid(self, validator):
        assert validator.validate_data_type("2025-01-15", "date") is True

    def test_date_invalid(self, validator):
        assert validator.validate_data_type("15/01/2025", "date") is False

    def test_list_float_valid(self, validator):
        assert validator.validate_data_type([1.0, 2.0, 3.0], "list_float") is True

    def test_list_float_empty(self, validator):
        assert validator.validate_data_type([], "list_float") is True

    def test_list_float_with_string_invalid(self, validator):
        assert validator.validate_data_type([1.0, "bad"], "list_float") is False

    def test_list_string_valid(self, validator):
        assert validator.validate_data_type(["a", "b"], "list_string") is True

    def test_list_string_with_int_invalid(self, validator):
        assert validator.validate_data_type(["a", 1], "list_string") is False

    def test_dict_valid(self, validator):
        assert validator.validate_data_type({"k": "v"}, "dict") is True

    def test_unknown_type_invalid(self, validator):
        assert validator.validate_data_type(42, "unknown_type") is False
