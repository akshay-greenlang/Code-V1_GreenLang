# -*- coding: utf-8 -*-
"""
Comprehensive tests for GreenLang Business Rules Engine.

Tests cover:
- All 12 rule operators (==, !=, >, >=, <, <=, in, not_in, contains, regex, is_null, not_null)
- Rule sets and conditional rules
- Nested field path support
- Rule enabling/disabling
- Custom error messages
- Severity levels
"""

import pytest
from greenlang.validation.rules import (
    RulesEngine,
    Rule,
    RuleSet,
    RuleOperator
)
from greenlang.validation.framework import ValidationSeverity


# Test Fixtures
@pytest.fixture
def engine():
    """Create a fresh rules engine."""
    return RulesEngine()


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "name": "John Doe",
        "age": 25,
        "email": "john@example.com",
        "status": "active",
        "tags": ["developer", "python"],
        "score": 85.5,
        "address": {
            "city": "New York",
            "zipcode": "10001"
        }
    }


# Rule Creation Tests
class TestRuleCreation:
    """Test Rule model creation."""

    def test_basic_rule_creation(self):
        """Test creating a basic rule."""
        rule = Rule(
            name="check_age",
            field="age",
            operator=RuleOperator.GREATER_EQUAL,
            value=18
        )
        assert rule.name == "check_age"
        assert rule.field == "age"
        assert rule.operator == RuleOperator.GREATER_EQUAL
        assert rule.value == 18

    def test_rule_with_message(self):
        """Test rule with custom message."""
        rule = Rule(
            name="check_email",
            field="email",
            operator=RuleOperator.NOT_NULL,
            message="Email is required"
        )
        assert rule.message == "Email is required"

    def test_rule_with_severity(self):
        """Test rule with custom severity."""
        rule = Rule(
            name="check_optional",
            field="phone",
            operator=RuleOperator.NOT_NULL,
            severity=ValidationSeverity.WARNING
        )
        assert rule.severity == ValidationSeverity.WARNING

    def test_rule_disabled(self):
        """Test creating disabled rule."""
        rule = Rule(
            name="disabled",
            field="test",
            operator=RuleOperator.EQUALS,
            value="test",
            enabled=False
        )
        assert rule.enabled is False

    def test_rule_with_condition(self):
        """Test rule with condition."""
        rule = Rule(
            name="conditional",
            field="age",
            operator=RuleOperator.GREATER_EQUAL,
            value=18,
            condition="exists:email"
        )
        assert rule.condition == "exists:email"


# RuleSet Tests
class TestRuleSet:
    """Test RuleSet functionality."""

    def test_ruleset_creation(self):
        """Test creating a rule set."""
        rules = [
            Rule(name="r1", field="f1", operator=RuleOperator.NOT_NULL),
            Rule(name="r2", field="f2", operator=RuleOperator.NOT_NULL)
        ]

        ruleset = RuleSet(
            name="required_fields",
            rules=rules,
            description="Check required fields"
        )

        assert ruleset.name == "required_fields"
        assert len(ruleset.rules) == 2
        assert ruleset.description == "Check required fields"

    def test_empty_ruleset(self):
        """Test creating empty rule set."""
        ruleset = RuleSet(name="empty")
        assert len(ruleset.rules) == 0


# Operator Tests
class TestRuleOperators:
    """Test all rule operators."""

    def test_equals_operator(self, engine, sample_data):
        """Test EQUALS operator."""
        rule = Rule(
            name="check_status",
            field="status",
            operator=RuleOperator.EQUALS,
            value="active"
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test failure
        data = sample_data.copy()
        data["status"] = "inactive"
        result = engine.validate(data)
        assert result.valid is False

    def test_not_equals_operator(self, engine, sample_data):
        """Test NOT_EQUALS operator."""
        rule = Rule(
            name="not_deleted",
            field="status",
            operator=RuleOperator.NOT_EQUALS,
            value="deleted"
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test failure
        data = sample_data.copy()
        data["status"] = "deleted"
        result = engine.validate(data)
        assert result.valid is False

    def test_greater_than_operator(self, engine, sample_data):
        """Test GREATER_THAN operator."""
        rule = Rule(
            name="min_age",
            field="age",
            operator=RuleOperator.GREATER_THAN,
            value=18
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test failure
        data = sample_data.copy()
        data["age"] = 18
        result = engine.validate(data)
        assert result.valid is False

    def test_greater_equal_operator(self, engine, sample_data):
        """Test GREATER_EQUAL operator."""
        rule = Rule(
            name="min_age",
            field="age",
            operator=RuleOperator.GREATER_EQUAL,
            value=18
        )
        engine.add_rule(rule)

        # Test exact value
        data = sample_data.copy()
        data["age"] = 18
        result = engine.validate(data)
        assert result.valid is True

        # Test failure
        data["age"] = 17
        result = engine.validate(data)
        assert result.valid is False

    def test_less_than_operator(self, engine, sample_data):
        """Test LESS_THAN operator."""
        rule = Rule(
            name="max_score",
            field="score",
            operator=RuleOperator.LESS_THAN,
            value=100
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test failure
        data = sample_data.copy()
        data["score"] = 100
        result = engine.validate(data)
        assert result.valid is False

    def test_less_equal_operator(self, engine, sample_data):
        """Test LESS_EQUAL operator."""
        rule = Rule(
            name="max_score",
            field="score",
            operator=RuleOperator.LESS_EQUAL,
            value=100
        )
        engine.add_rule(rule)

        # Test exact value
        data = sample_data.copy()
        data["score"] = 100
        result = engine.validate(data)
        assert result.valid is True

        # Test failure
        data["score"] = 101
        result = engine.validate(data)
        assert result.valid is False

    def test_in_operator(self, engine, sample_data):
        """Test IN operator."""
        rule = Rule(
            name="valid_status",
            field="status",
            operator=RuleOperator.IN,
            value=["active", "pending", "completed"]
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test failure
        data = sample_data.copy()
        data["status"] = "invalid"
        result = engine.validate(data)
        assert result.valid is False

    def test_not_in_operator(self, engine, sample_data):
        """Test NOT_IN operator."""
        rule = Rule(
            name="not_blocked",
            field="status",
            operator=RuleOperator.NOT_IN,
            value=["blocked", "suspended", "deleted"]
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test failure
        data = sample_data.copy()
        data["status"] = "blocked"
        result = engine.validate(data)
        assert result.valid is False

    def test_contains_operator(self, engine, sample_data):
        """Test CONTAINS operator."""
        rule = Rule(
            name="has_python",
            field="tags",
            operator=RuleOperator.CONTAINS,
            value="python"
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test failure
        data = sample_data.copy()
        data["tags"] = ["developer", "java"]
        result = engine.validate(data)
        assert result.valid is False

    def test_is_null_operator(self, engine):
        """Test IS_NULL operator."""
        rule = Rule(
            name="optional_field",
            field="optional",
            operator=RuleOperator.IS_NULL
        )
        engine.add_rule(rule)

        # Test with null value
        data = {"optional": None}
        result = engine.validate(data)
        assert result.valid is True

        # Test with non-null value
        data = {"optional": "value"}
        result = engine.validate(data)
        assert result.valid is False

    def test_not_null_operator(self, engine, sample_data):
        """Test NOT_NULL operator."""
        rule = Rule(
            name="required_name",
            field="name",
            operator=RuleOperator.NOT_NULL
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test with null value
        data = sample_data.copy()
        data["name"] = None
        result = engine.validate(data)
        assert result.valid is False

    def test_regex_operator(self, engine, sample_data):
        """Test REGEX operator."""
        rule = Rule(
            name="email_format",
            field="email",
            operator=RuleOperator.REGEX,
            value=r"^[\w\.-]+@[\w\.-]+\.\w+$"
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

        # Test failure
        data = sample_data.copy()
        data["email"] = "invalid-email"
        result = engine.validate(data)
        assert result.valid is False


# Nested Field Path Tests
class TestNestedFieldPaths:
    """Test nested field path support."""

    def test_simple_nested_path(self, engine, sample_data):
        """Test accessing nested field."""
        rule = Rule(
            name="check_city",
            field="address.city",
            operator=RuleOperator.EQUALS,
            value="New York"
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is True

    def test_nested_path_not_found(self, engine, sample_data):
        """Test accessing non-existent nested field."""
        rule = Rule(
            name="check_country",
            field="address.country",
            operator=RuleOperator.NOT_NULL
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is False

    def test_deep_nested_path(self, engine):
        """Test deeply nested path."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42
                    }
                }
            }
        }

        rule = Rule(
            name="deep_check",
            field="level1.level2.level3.value",
            operator=RuleOperator.EQUALS,
            value=42
        )
        engine.add_rule(rule)

        result = engine.validate(data)
        assert result.valid is True

    def test_get_field_value_simple(self, engine):
        """Test getting simple field value."""
        data = {"name": "John"}
        value = engine.get_field_value(data, "name")
        assert value == "John"

    def test_get_field_value_nested(self, engine, sample_data):
        """Test getting nested field value."""
        value = engine.get_field_value(sample_data, "address.city")
        assert value == "New York"

    def test_get_field_value_missing(self, engine):
        """Test getting missing field value."""
        data = {"name": "John"}
        value = engine.get_field_value(data, "missing.field")
        assert value is None


# Conditional Rules Tests
class TestConditionalRules:
    """Test conditional rule execution."""

    def test_condition_exists(self, engine, sample_data):
        """Test exists condition."""
        rule = Rule(
            name="check_age_if_email",
            field="age",
            operator=RuleOperator.GREATER_EQUAL,
            value=18,
            condition="exists:email"
        )
        engine.add_rule(rule)

        # Email exists, rule should run
        result = engine.validate(sample_data)
        assert result.valid is True

        # No email, rule should be skipped
        data = {"age": 10}  # Would fail if rule runs
        result = engine.validate(data)
        assert result.valid is True  # Skipped

    def test_evaluate_condition(self, engine, sample_data):
        """Test condition evaluation."""
        # Exists condition - field present
        assert engine.evaluate_condition(sample_data, "exists:email") is True

        # Exists condition - field missing
        assert engine.evaluate_condition(sample_data, "exists:missing") is False


# RuleSet Tests
class TestRuleSetValidation:
    """Test validation with rule sets."""

    def test_validate_with_ruleset(self, engine, sample_data):
        """Test validation using rule set."""
        rules = [
            Rule(name="r1", field="name", operator=RuleOperator.NOT_NULL),
            Rule(name="r2", field="age", operator=RuleOperator.GREATER_EQUAL, value=18)
        ]
        ruleset = RuleSet(name="basic_checks", rules=rules)

        engine.add_rule_set(ruleset)

        result = engine.validate(sample_data, rule_set_name="basic_checks")
        assert result.valid is True

    def test_ruleset_not_found(self, engine, sample_data):
        """Test validation with non-existent rule set."""
        result = engine.validate(sample_data, rule_set_name="missing")
        assert result.valid is False
        assert any("not found" in e.message for e in result.errors)

    def test_disabled_ruleset(self, engine, sample_data):
        """Test validation with disabled rule set."""
        rules = [
            Rule(name="r1", field="invalid", operator=RuleOperator.NOT_NULL)
        ]
        ruleset = RuleSet(name="checks", rules=rules, enabled=False)

        engine.add_rule_set(ruleset)

        result = engine.validate(sample_data, rule_set_name="checks")
        assert result.valid is True
        assert "skipped" in result.metadata


# Engine Management Tests
class TestEngineManagement:
    """Test rules engine management."""

    def test_add_rule(self, engine):
        """Test adding a rule."""
        rule = Rule(
            name="test_rule",
            field="field",
            operator=RuleOperator.NOT_NULL
        )
        engine.add_rule(rule)

        assert len(engine.rules) == 1
        assert engine.rules[0] == rule

    def test_remove_rule(self, engine):
        """Test removing a rule."""
        rule = Rule(
            name="test_rule",
            field="field",
            operator=RuleOperator.NOT_NULL
        )
        engine.add_rule(rule)

        engine.remove_rule("test_rule")
        assert len(engine.rules) == 0

    def test_get_rule_names(self, engine):
        """Test getting rule names."""
        rule1 = Rule(name="r1", field="f1", operator=RuleOperator.NOT_NULL)
        rule2 = Rule(name="r2", field="f2", operator=RuleOperator.NOT_NULL)

        engine.add_rule(rule1)
        engine.add_rule(rule2)

        names = engine.get_rule_names()
        assert "r1" in names
        assert "r2" in names

    def test_enable_rule(self, engine, sample_data):
        """Test enabling a rule."""
        rule = Rule(
            name="test",
            field="invalid",
            operator=RuleOperator.NOT_NULL,
            enabled=False
        )
        engine.add_rule(rule)

        # Rule disabled, should pass
        result = engine.validate(sample_data)
        assert result.valid is True

        # Enable rule
        engine.enable_rule("test")

        # Now should fail
        result = engine.validate(sample_data)
        assert result.valid is False

    def test_disable_rule(self, engine, sample_data):
        """Test disabling a rule."""
        rule = Rule(
            name="test",
            field="invalid",
            operator=RuleOperator.NOT_NULL
        )
        engine.add_rule(rule)

        # Rule enabled, should fail
        result = engine.validate(sample_data)
        assert result.valid is False

        # Disable rule
        engine.disable_rule("test")

        # Now should pass
        result = engine.validate(sample_data)
        assert result.valid is True

    def test_load_rules_from_dict(self, engine):
        """Test loading rules from dictionary."""
        rules_config = [
            {
                "name": "r1",
                "field": "f1",
                "operator": "not_null"
            },
            {
                "name": "r2",
                "field": "f2",
                "operator": ">=",
                "value": 10
            }
        ]

        engine.load_rules_from_dict(rules_config)

        assert len(engine.rules) == 2
        assert engine.rules[0].name == "r1"
        assert engine.rules[1].name == "r2"


# Error Message Tests
class TestErrorMessages:
    """Test error message customization."""

    def test_custom_error_message(self, engine, sample_data):
        """Test custom error message."""
        rule = Rule(
            name="age_check",
            field="age",
            operator=RuleOperator.LESS_THAN,
            value=18,
            message="User must be under 18"
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is False
        assert "User must be under 18" in result.errors[0].message

    def test_default_error_message(self, engine, sample_data):
        """Test default error message."""
        rule = Rule(
            name="age_check",
            field="age",
            operator=RuleOperator.LESS_THAN,
            value=18
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert result.valid is False
        assert "age_check" in result.errors[0].message


# Metadata Tests
class TestValidationMetadata:
    """Test validation metadata."""

    def test_metadata_rules_evaluated(self, engine, sample_data):
        """Test metadata contains rules evaluated count."""
        rule = Rule(name="r1", field="name", operator=RuleOperator.NOT_NULL)
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert "rules_evaluated" in result.metadata
        assert result.metadata["rules_evaluated"] == 1

    def test_error_location(self, engine, sample_data):
        """Test error contains location information."""
        rule = Rule(
            name="check",
            field="address.city",
            operator=RuleOperator.EQUALS,
            value="Boston"
        )
        engine.add_rule(rule)

        result = engine.validate(sample_data)
        assert len(result.errors) == 1
        assert result.errors[0].location is not None


# Edge Cases
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_rule_on_missing_field(self, engine):
        """Test rule on missing field."""
        rule = Rule(
            name="missing",
            field="nonexistent",
            operator=RuleOperator.EQUALS,
            value="test"
        )
        engine.add_rule(rule)

        result = engine.validate({"other": "field"})
        assert result.valid is False

    def test_type_mismatch_comparison(self, engine):
        """Test comparison with type mismatch."""
        rule = Rule(
            name="compare",
            field="value",
            operator=RuleOperator.GREATER_THAN,
            value=10
        )
        engine.add_rule(rule)

        # String vs number comparison
        result = engine.validate({"value": "not a number"})
        assert result.valid is False

    def test_empty_data(self, engine):
        """Test validation with empty data."""
        rule = Rule(
            name="check",
            field="field",
            operator=RuleOperator.NOT_NULL
        )
        engine.add_rule(rule)

        result = engine.validate({})
        assert result.valid is False

    def test_disabled_rule_not_evaluated(self, engine):
        """Test that disabled rules are not evaluated."""
        rule = Rule(
            name="disabled",
            field="invalid",
            operator=RuleOperator.NOT_NULL,
            enabled=False
        )
        engine.add_rule(rule)

        result = engine.validate({})
        assert result.valid is True
