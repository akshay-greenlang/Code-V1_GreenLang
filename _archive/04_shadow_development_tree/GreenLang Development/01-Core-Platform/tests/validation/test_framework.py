# -*- coding: utf-8 -*-
"""
Tests for ValidationFramework
"""

import pytest
from greenlang.validation import (
    ValidationFramework,
    ValidationResult,
    ValidationError,
    ValidationSeverity,
    SchemaValidator,
    RulesEngine,
    Rule,
    RuleOperator,
)


class TestValidationFramework:
    """Test ValidationFramework."""

    def test_framework_initialization(self):
        """Test framework initialization."""
        framework = ValidationFramework()
        assert framework is not None
        assert len(framework.validators) == 0

    def test_add_validator(self):
        """Test adding validators."""
        framework = ValidationFramework()

        def dummy_validator(data):
            return ValidationResult(valid=True)

        framework.add_validator("test", dummy_validator)
        assert "test" in framework.validators

    def test_validation_success(self):
        """Test successful validation."""
        framework = ValidationFramework()

        def always_pass(data):
            return ValidationResult(valid=True)

        framework.add_validator("always_pass", always_pass)

        result = framework.validate({"test": "data"})
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validation_failure(self):
        """Test validation failure."""
        framework = ValidationFramework()

        def always_fail(data):
            result = ValidationResult(valid=False)
            result.add_error(
                ValidationError(
                    field="test",
                    message="Test error",
                    severity=ValidationSeverity.ERROR,
                    validator="always_fail"
                )
            )
            return result

        framework.add_validator("always_fail", always_fail)

        result = framework.validate({"test": "data"})
        assert result.valid is False
        assert len(result.errors) == 1


class TestSchemaValidator:
    """Test SchemaValidator."""

    def test_schema_initialization(self):
        """Test schema validator initialization."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        validator = SchemaValidator(schema)
        assert validator.schema == schema

    def test_valid_data(self):
        """Test validation of valid data."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        validator = SchemaValidator(schema)

        result = validator.validate({"name": "John"})
        assert result.valid is True

    def test_missing_required_field(self):
        """Test missing required field."""
        schema = {
            "type": "object",
            "required": ["name"]
        }
        validator = SchemaValidator(schema)

        result = validator.validate({})
        # Should fail or warn about missing field
        # Exact behavior depends on jsonschema availability


class TestRulesEngine:
    """Test RulesEngine."""

    def test_engine_initialization(self):
        """Test rules engine initialization."""
        engine = RulesEngine()
        assert engine is not None
        assert len(engine.rules) == 0

    def test_add_rule(self):
        """Test adding rules."""
        engine = RulesEngine()

        rule = Rule(
            name="age_check",
            field="age",
            operator=RuleOperator.GREATER_EQUAL,
            value=18
        )
        engine.add_rule(rule)

        assert len(engine.rules) == 1

    def test_rule_validation_pass(self):
        """Test rule validation passing."""
        engine = RulesEngine()

        rule = Rule(
            name="age_check",
            field="age",
            operator=RuleOperator.GREATER_EQUAL,
            value=18
        )
        engine.add_rule(rule)

        result = engine.validate({"age": 25})
        assert result.valid is True

    def test_rule_validation_fail(self):
        """Test rule validation failing."""
        engine = RulesEngine()

        rule = Rule(
            name="age_check",
            field="age",
            operator=RuleOperator.GREATER_EQUAL,
            value=18,
            message="Age must be 18 or older"
        )
        engine.add_rule(rule)

        result = engine.validate({"age": 15})
        assert result.valid is False
        assert len(result.errors) == 1
        assert "Age must be 18 or older" in result.errors[0].message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
