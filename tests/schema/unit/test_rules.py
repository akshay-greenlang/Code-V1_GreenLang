# -*- coding: utf-8 -*-
"""
Unit Tests for Rule Engine Integration (GL-FOUND-X-002 Task 2.4).

Tests cover:
    - RuleExpression model validation
    - Rule model validation
    - ExpressionEvaluator for all operators
    - RuleValidator for cross-field validation
    - Built-in rule templates
    - Factory functions

Author: GreenLang Framework Team
Version: 0.1.0
"""

from datetime import datetime
from typing import Any, Dict, List

import pytest

from greenlang.schema.compiler.ir import RuleBindingIR, SchemaIR
from greenlang.schema.models.config import ValidationOptions, ValidationProfile
from greenlang.schema.models.finding import Severity
from greenlang.schema.validator.rules import (
    ARITHMETIC_OPERATORS,
    COMPARISON_OPERATORS,
    ERROR_CODE_CONDITIONAL_REQUIRED,
    ERROR_CODE_CONSISTENCY_ERROR,
    ERROR_CODE_RULE_VIOLATION,
    EXTENDED_OPERATORS,
    NULL_CHECK_OPERATORS,
    ExpressionEvaluator,
    Rule,
    RuleExpression,
    RuleValidator,
    create_conditional_required_rule,
    create_range_rule,
    create_sum_consistency_rule,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def evaluator() -> ExpressionEvaluator:
    """Create an ExpressionEvaluator instance."""
    return ExpressionEvaluator()


@pytest.fixture
def sample_payload() -> Dict[str, Any]:
    """Create a sample payload for testing."""
    return {
        "fuel_type": "gas",
        "methane_slip": 0.5,
        "energy_consumption": 100,
        "unit": "kWh",
        "scope1": 10,
        "scope2": 20,
        "scope3": 30,
        "total_emissions": 60,
        "temperature": 25.5,
        "transport_mode": "car",
        "values": [1, 2, 3, 4, 5],
        "nested": {
            "deep": {
                "value": 42
            }
        }
    }


@pytest.fixture
def schema_ir() -> SchemaIR:
    """Create a minimal SchemaIR for testing."""
    return SchemaIR(
        schema_id="test/rules",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        rule_bindings=[],
    )


@pytest.fixture
def validation_options() -> ValidationOptions:
    """Create ValidationOptions for testing."""
    return ValidationOptions()


@pytest.fixture
def rule_validator(schema_ir: SchemaIR, validation_options: ValidationOptions) -> RuleValidator:
    """Create a RuleValidator instance."""
    return RuleValidator(schema_ir, validation_options)


# =============================================================================
# RULE EXPRESSION MODEL TESTS
# =============================================================================


class TestRuleExpression:
    """Tests for RuleExpression model."""

    def test_simple_comparison_expression(self):
        """Test creating a simple comparison expression."""
        expr = RuleExpression(
            operator="==",
            field="/fuel_type",
            value="gas"
        )
        assert expr.operator == "=="
        assert expr.field == "/fuel_type"
        assert expr.value == "gas"
        assert expr.is_comparison()
        assert not expr.is_logical()

    def test_logical_expression(self):
        """Test creating a logical expression."""
        expr = RuleExpression(
            operator="and",
            operands=[
                RuleExpression(operator=">=", field="/value", value=0),
                RuleExpression(operator="<=", field="/value", value=100),
            ]
        )
        assert expr.operator == "and"
        assert len(expr.operands) == 2
        assert expr.is_logical()
        assert not expr.is_comparison()

    def test_aggregate_expression(self):
        """Test creating an aggregate expression."""
        expr = RuleExpression(
            operator="sum",
            fields=["/scope1", "/scope2", "/scope3"]
        )
        assert expr.operator == "sum"
        assert len(expr.fields) == 3
        assert expr.is_aggregate()

    def test_null_check_expression(self):
        """Test creating a null check expression."""
        expr = RuleExpression(
            operator="exists",
            field="/optional_field"
        )
        assert expr.operator == "exists"
        assert expr.is_null_check()

    def test_invalid_field_path(self):
        """Test that invalid field paths are rejected."""
        with pytest.raises(ValueError, match="Invalid JSON Pointer"):
            RuleExpression(
                operator="==",
                field="invalid_path",  # Missing leading /
                value="test"
            )

    def test_valid_json_pointer_paths(self):
        """Test various valid JSON Pointer paths."""
        valid_paths = [
            "/simple",
            "/nested/path",
            "/array/0",
            "/field~0name",  # Escaped ~
            "/field~1name",  # Escaped /
        ]
        for path in valid_paths:
            expr = RuleExpression(operator="exists", field=path)
            assert expr.field == path


# =============================================================================
# RULE MODEL TESTS
# =============================================================================


class TestRule:
    """Tests for Rule model."""

    def test_simple_rule(self):
        """Test creating a simple rule."""
        rule = Rule(
            rule_id="test_rule",
            check=RuleExpression(operator="exists", field="/name"),
            message="Name is required"
        )
        assert rule.rule_id == "test_rule"
        assert rule.severity == "error"
        assert rule.message == "Name is required"
        assert not rule.is_conditional()

    def test_conditional_rule(self):
        """Test creating a conditional rule."""
        rule = Rule(
            rule_id="conditional_test",
            when=RuleExpression(operator="==", field="/type", value="special"),
            check=RuleExpression(operator="exists", field="/special_field"),
            message="special_field required when type is special"
        )
        assert rule.is_conditional()
        assert rule.when is not None

    def test_severity_validation(self):
        """Test severity validation."""
        # Valid severities
        for sev in ["error", "warning", "info"]:
            rule = Rule(
                rule_id="test",
                severity=sev,
                check=RuleExpression(operator="exists", field="/x"),
                message="test"
            )
            assert rule.severity == sev

        # Invalid severity
        with pytest.raises(ValueError, match="Invalid severity"):
            Rule(
                rule_id="test",
                severity="critical",  # Invalid
                check=RuleExpression(operator="exists", field="/x"),
                message="test"
            )

    def test_error_code_determination(self):
        """Test error code determination based on rule type."""
        # Conditional required -> E401
        rule1 = Rule(
            rule_id="conditional",
            when=RuleExpression(operator="==", field="/x", value="y"),
            check=RuleExpression(operator="exists", field="/z"),
            message="test"
        )
        assert rule1.get_error_code() == ERROR_CODE_CONDITIONAL_REQUIRED

        # Consistency check -> E402
        rule2 = Rule(
            rule_id="consistency",
            check=RuleExpression(
                operator="==",
                operands=[
                    RuleExpression(operator="sum", fields=["/a", "/b"]),
                    RuleExpression(operator="value", field="/c"),
                ]
            ),
            message="test"
        )
        assert rule2.get_error_code() == ERROR_CODE_CONSISTENCY_ERROR

        # Generic rule -> E400
        rule3 = Rule(
            rule_id="generic",
            check=RuleExpression(operator=">=", field="/x", value=0),
            message="test"
        )
        assert rule3.get_error_code() == ERROR_CODE_RULE_VIOLATION


# =============================================================================
# EXPRESSION EVALUATOR TESTS
# =============================================================================


class TestExpressionEvaluator:
    """Tests for ExpressionEvaluator."""

    # -------------------------------------------------------------------------
    # Comparison Operators
    # -------------------------------------------------------------------------

    def test_equality_comparison(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test == operator."""
        expr = RuleExpression(operator="==", field="/fuel_type", value="gas")
        assert evaluator.evaluate(expr, sample_payload) is True

        expr2 = RuleExpression(operator="==", field="/fuel_type", value="diesel")
        assert evaluator.evaluate(expr2, sample_payload) is False

    def test_inequality_comparison(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test != operator."""
        expr = RuleExpression(operator="!=", field="/fuel_type", value="diesel")
        assert evaluator.evaluate(expr, sample_payload) is True

    def test_greater_than(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test > operator."""
        expr = RuleExpression(operator=">", field="/energy_consumption", value=50)
        assert evaluator.evaluate(expr, sample_payload) is True

        expr2 = RuleExpression(operator=">", field="/energy_consumption", value=100)
        assert evaluator.evaluate(expr2, sample_payload) is False

    def test_greater_equal(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test >= operator."""
        expr = RuleExpression(operator=">=", field="/energy_consumption", value=100)
        assert evaluator.evaluate(expr, sample_payload) is True

    def test_less_than(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test < operator."""
        expr = RuleExpression(operator="<", field="/energy_consumption", value=200)
        assert evaluator.evaluate(expr, sample_payload) is True

    def test_less_equal(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test <= operator."""
        expr = RuleExpression(operator="<=", field="/energy_consumption", value=100)
        assert evaluator.evaluate(expr, sample_payload) is True

    # -------------------------------------------------------------------------
    # Membership Operators
    # -------------------------------------------------------------------------

    def test_in_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'in' operator."""
        expr = RuleExpression(
            operator="in",
            field="/transport_mode",
            value=["car", "truck", "bus"]
        )
        assert evaluator.evaluate(expr, sample_payload) is True

        expr2 = RuleExpression(
            operator="in",
            field="/transport_mode",
            value=["bike", "walk"]
        )
        assert evaluator.evaluate(expr2, sample_payload) is False

    def test_not_in_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'not_in' operator."""
        expr = RuleExpression(
            operator="not_in",
            field="/transport_mode",
            value=["bike", "walk"]
        )
        assert evaluator.evaluate(expr, sample_payload) is True

    def test_contains_operator(self, evaluator: ExpressionEvaluator):
        """Test 'contains' operator."""
        payload = {"name": "hello world"}
        expr = RuleExpression(operator="contains", field="/name", value="world")
        assert evaluator.evaluate(expr, payload) is True

        expr2 = RuleExpression(operator="contains", field="/name", value="foo")
        assert evaluator.evaluate(expr2, payload) is False

    # -------------------------------------------------------------------------
    # String Operators
    # -------------------------------------------------------------------------

    def test_starts_with_operator(self, evaluator: ExpressionEvaluator):
        """Test 'starts_with' operator."""
        payload = {"email": "test@example.com"}
        expr = RuleExpression(operator="starts_with", field="/email", value="test@")
        assert evaluator.evaluate(expr, payload) is True

    def test_ends_with_operator(self, evaluator: ExpressionEvaluator):
        """Test 'ends_with' operator."""
        payload = {"email": "test@example.com"}
        expr = RuleExpression(operator="ends_with", field="/email", value=".com")
        assert evaluator.evaluate(expr, payload) is True

    def test_matches_operator(self, evaluator: ExpressionEvaluator):
        """Test 'matches' (regex) operator."""
        payload = {"code": "ABC-123"}
        expr = RuleExpression(operator="matches", field="/code", value=r"^[A-Z]+-\d+$")
        assert evaluator.evaluate(expr, payload) is True

        expr2 = RuleExpression(operator="matches", field="/code", value=r"^\d+$")
        assert evaluator.evaluate(expr2, payload) is False

    def test_regex_operator(self, evaluator: ExpressionEvaluator):
        """Test 'regex' operator (alias for matches)."""
        payload = {"phone": "555-1234"}
        expr = RuleExpression(operator="regex", field="/phone", value=r"^\d{3}-\d{4}$")
        assert evaluator.evaluate(expr, payload) is True

    # -------------------------------------------------------------------------
    # Null Check Operators
    # -------------------------------------------------------------------------

    def test_is_null_operator(self, evaluator: ExpressionEvaluator):
        """Test 'is_null' operator."""
        payload = {"field": None}
        expr = RuleExpression(operator="is_null", field="/field")
        assert evaluator.evaluate(expr, payload) is True

        payload2 = {"field": "value"}
        assert evaluator.evaluate(expr, payload2) is False

    def test_is_not_null_operator(self, evaluator: ExpressionEvaluator):
        """Test 'is_not_null' operator."""
        payload = {"field": "value"}
        expr = RuleExpression(operator="is_not_null", field="/field")
        assert evaluator.evaluate(expr, payload) is True

    def test_exists_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'exists' operator."""
        expr = RuleExpression(operator="exists", field="/fuel_type")
        assert evaluator.evaluate(expr, sample_payload) is True

        expr2 = RuleExpression(operator="exists", field="/nonexistent")
        assert evaluator.evaluate(expr2, sample_payload) is False

    def test_not_exists_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'not_exists' operator."""
        expr = RuleExpression(operator="not_exists", field="/nonexistent")
        assert evaluator.evaluate(expr, sample_payload) is True

    # -------------------------------------------------------------------------
    # Logical Operators
    # -------------------------------------------------------------------------

    def test_and_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'and' operator."""
        expr = RuleExpression(
            operator="and",
            operands=[
                RuleExpression(operator=">=", field="/temperature", value=0),
                RuleExpression(operator="<=", field="/temperature", value=50),
            ]
        )
        assert evaluator.evaluate(expr, sample_payload) is True

        # One condition fails
        expr2 = RuleExpression(
            operator="and",
            operands=[
                RuleExpression(operator=">=", field="/temperature", value=0),
                RuleExpression(operator="<=", field="/temperature", value=20),  # Fails
            ]
        )
        assert evaluator.evaluate(expr2, sample_payload) is False

    def test_or_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'or' operator."""
        expr = RuleExpression(
            operator="or",
            operands=[
                RuleExpression(operator="==", field="/fuel_type", value="diesel"),
                RuleExpression(operator="==", field="/fuel_type", value="gas"),
            ]
        )
        assert evaluator.evaluate(expr, sample_payload) is True

    def test_not_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'not' operator."""
        expr = RuleExpression(
            operator="not",
            operands=[
                RuleExpression(operator="==", field="/fuel_type", value="diesel"),
            ]
        )
        assert evaluator.evaluate(expr, sample_payload) is True

    def test_empty_and(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test empty 'and' returns True (vacuously true)."""
        expr = RuleExpression(operator="and", operands=[])
        assert evaluator.evaluate(expr, sample_payload) is True

    def test_empty_or(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test empty 'or' returns False (vacuously false)."""
        expr = RuleExpression(operator="or", operands=[])
        assert evaluator.evaluate(expr, sample_payload) is False

    # -------------------------------------------------------------------------
    # Aggregate Operators
    # -------------------------------------------------------------------------

    def test_sum_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'sum' operator."""
        expr = RuleExpression(
            operator="sum",
            fields=["/scope1", "/scope2", "/scope3"]
        )
        result = evaluator.evaluate(expr, sample_payload)
        assert result == 60  # 10 + 20 + 30

    def test_count_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'count' operator."""
        expr = RuleExpression(
            operator="count",
            fields=["/scope1", "/scope2", "/scope3"]
        )
        result = evaluator.evaluate(expr, sample_payload)
        assert result == 3

    def test_min_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'min' operator."""
        expr = RuleExpression(
            operator="min",
            fields=["/scope1", "/scope2", "/scope3"]
        )
        result = evaluator.evaluate(expr, sample_payload)
        assert result == 10

    def test_max_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'max' operator."""
        expr = RuleExpression(
            operator="max",
            fields=["/scope1", "/scope2", "/scope3"]
        )
        result = evaluator.evaluate(expr, sample_payload)
        assert result == 30

    def test_avg_operator(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'avg' operator."""
        expr = RuleExpression(
            operator="avg",
            fields=["/scope1", "/scope2", "/scope3"]
        )
        result = evaluator.evaluate(expr, sample_payload)
        assert result == 20  # (10 + 20 + 30) / 3

    def test_sum_with_array(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test 'sum' operator with array field."""
        expr = RuleExpression(
            operator="sum",
            field="/values"
        )
        result = evaluator.evaluate(expr, sample_payload)
        assert result == 15  # 1 + 2 + 3 + 4 + 5

    # -------------------------------------------------------------------------
    # JSON Pointer Path Resolution
    # -------------------------------------------------------------------------

    def test_nested_path_resolution(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test nested JSON Pointer path resolution."""
        expr = RuleExpression(operator="==", field="/nested/deep/value", value=42)
        assert evaluator.evaluate(expr, sample_payload) is True

    def test_array_index_path(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test array index in JSON Pointer."""
        expr = RuleExpression(operator="==", field="/values/0", value=1)
        assert evaluator.evaluate(expr, sample_payload) is True

        expr2 = RuleExpression(operator="==", field="/values/2", value=3)
        assert evaluator.evaluate(expr2, sample_payload) is True

    def test_nonexistent_path(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test handling of nonexistent paths."""
        expr = RuleExpression(operator="==", field="/nonexistent/path", value=None)
        assert evaluator.evaluate(expr, sample_payload) is True  # None == None

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_none_comparison(self, evaluator: ExpressionEvaluator):
        """Test comparisons with None values."""
        payload = {"value": None}

        # Equality with None
        expr1 = RuleExpression(operator="==", field="/value", value=None)
        assert evaluator.evaluate(expr1, payload) is True

        # Numeric comparison with None returns False
        expr2 = RuleExpression(operator=">", field="/value", value=0)
        assert evaluator.evaluate(expr2, payload) is False

    def test_comparison_with_operands(self, evaluator: ExpressionEvaluator, sample_payload: Dict):
        """Test comparison using operands instead of field/value."""
        # Compare sum of fields with total
        expr = RuleExpression(
            operator="==",
            operands=[
                RuleExpression(operator="sum", fields=["/scope1", "/scope2", "/scope3"]),
                RuleExpression(operator="value", field="/total_emissions"),
            ]
        )
        # This needs special handling - sum returns 60, but we need to extract total_emissions value
        # For now, the second operand with operator="value" should return the field value
        # Actually, let's evaluate this properly
        result = evaluator.evaluate(expr, sample_payload)
        # The expression evaluator should handle this comparison
        # 60 == 60 -> True
        assert result is True

    def test_recursion_depth_limit(self, evaluator: ExpressionEvaluator):
        """Test that deep recursion is prevented."""
        # Create a deeply nested expression
        expr = RuleExpression(operator="exists", field="/x")
        for _ in range(60):  # Exceed max depth of 50
            expr = RuleExpression(operator="not", operands=[expr])

        with pytest.raises(RecursionError):
            evaluator.evaluate(expr, {"x": 1})


# =============================================================================
# RULE VALIDATOR TESTS
# =============================================================================


class TestRuleValidator:
    """Tests for RuleValidator."""

    def test_simple_rule_pass(
        self,
        rule_validator: RuleValidator,
        sample_payload: Dict
    ):
        """Test a simple rule that passes."""
        rule = Rule(
            rule_id="fuel_type_exists",
            check=RuleExpression(operator="exists", field="/fuel_type"),
            message="fuel_type is required"
        )
        findings = rule_validator.validate(sample_payload, [rule])
        assert len(findings) == 0

    def test_simple_rule_fail(
        self,
        rule_validator: RuleValidator,
        sample_payload: Dict
    ):
        """Test a simple rule that fails."""
        rule = Rule(
            rule_id="nonexistent_required",
            check=RuleExpression(operator="exists", field="/nonexistent"),
            message="nonexistent field is required"
        )
        findings = rule_validator.validate(sample_payload, [rule])
        assert len(findings) == 1
        assert findings[0].code == ERROR_CODE_RULE_VIOLATION
        assert "nonexistent" in findings[0].message

    def test_conditional_rule_condition_not_met(
        self,
        rule_validator: RuleValidator,
        sample_payload: Dict
    ):
        """Test conditional rule when condition is not met."""
        rule = Rule(
            rule_id="diesel_specific",
            when=RuleExpression(operator="==", field="/fuel_type", value="diesel"),
            check=RuleExpression(operator="exists", field="/cetane_number"),
            message="cetane_number required for diesel"
        )
        # Payload has fuel_type="gas", so condition not met, rule should pass
        findings = rule_validator.validate(sample_payload, [rule])
        assert len(findings) == 0

    def test_conditional_rule_condition_met_pass(
        self,
        rule_validator: RuleValidator,
        sample_payload: Dict
    ):
        """Test conditional rule when condition is met and check passes."""
        rule = Rule(
            rule_id="gas_methane_slip",
            when=RuleExpression(operator="==", field="/fuel_type", value="gas"),
            check=RuleExpression(operator="exists", field="/methane_slip"),
            message="methane_slip required for gas"
        )
        # Payload has fuel_type="gas" and methane_slip exists
        findings = rule_validator.validate(sample_payload, [rule])
        assert len(findings) == 0

    def test_conditional_rule_condition_met_fail(
        self,
        rule_validator: RuleValidator
    ):
        """Test conditional rule when condition is met but check fails."""
        payload = {"fuel_type": "gas"}  # No methane_slip
        rule = Rule(
            rule_id="gas_methane_slip",
            when=RuleExpression(operator="==", field="/fuel_type", value="gas"),
            check=RuleExpression(operator="exists", field="/methane_slip"),
            message="methane_slip required for gas"
        )
        findings = rule_validator.validate(payload, [rule])
        assert len(findings) == 1
        assert findings[0].code == ERROR_CODE_CONDITIONAL_REQUIRED

    def test_consistency_check_pass(
        self,
        rule_validator: RuleValidator,
        sample_payload: Dict
    ):
        """Test consistency check that passes."""
        rule = create_sum_consistency_rule(
            rule_id="scope_sum",
            component_fields=["/scope1", "/scope2", "/scope3"],
            total_field="/total_emissions"
        )
        # Sample payload has scope1=10, scope2=20, scope3=30, total=60
        findings = rule_validator.validate(sample_payload, [rule])
        assert len(findings) == 0

    def test_consistency_check_fail(
        self,
        rule_validator: RuleValidator
    ):
        """Test consistency check that fails."""
        payload = {
            "scope1": 10,
            "scope2": 20,
            "scope3": 30,
            "total_emissions": 100,  # Wrong! Should be 60
        }
        rule = create_sum_consistency_rule(
            rule_id="scope_sum",
            component_fields=["/scope1", "/scope2", "/scope3"],
            total_field="/total_emissions"
        )
        findings = rule_validator.validate(payload, [rule])
        assert len(findings) == 1
        assert findings[0].code == ERROR_CODE_CONSISTENCY_ERROR

    def test_range_rule_pass(
        self,
        rule_validator: RuleValidator,
        sample_payload: Dict
    ):
        """Test range rule that passes."""
        rule = create_range_rule(
            rule_id="temperature_range",
            field="/temperature",
            minimum=0,
            maximum=100
        )
        # temperature = 25.5, within [0, 100]
        findings = rule_validator.validate(sample_payload, [rule])
        assert len(findings) == 0

    def test_range_rule_fail(
        self,
        rule_validator: RuleValidator
    ):
        """Test range rule that fails."""
        payload = {"temperature": 150}  # Outside range
        rule = create_range_rule(
            rule_id="temperature_range",
            field="/temperature",
            minimum=0,
            maximum=100
        )
        findings = rule_validator.validate(payload, [rule])
        assert len(findings) == 1

    def test_multiple_rules(
        self,
        rule_validator: RuleValidator,
        sample_payload: Dict
    ):
        """Test validation with multiple rules."""
        rules = [
            Rule(
                rule_id="rule1",
                check=RuleExpression(operator="exists", field="/fuel_type"),
                message="fuel_type required"
            ),
            Rule(
                rule_id="rule2",
                check=RuleExpression(operator=">=", field="/energy_consumption", value=0),
                message="energy_consumption must be non-negative"
            ),
            Rule(
                rule_id="rule3",
                check=RuleExpression(operator="exists", field="/nonexistent"),
                message="nonexistent required"
            ),
        ]
        findings = rule_validator.validate(sample_payload, rules)
        # Only rule3 should fail
        assert len(findings) == 1
        assert findings[0].message == "nonexistent required"

    def test_fail_fast_option(
        self,
        schema_ir: SchemaIR
    ):
        """Test fail_fast option stops at first error."""
        options = ValidationOptions(fail_fast=True)
        validator = RuleValidator(schema_ir, options)

        rules = [
            Rule(rule_id="r1", check=RuleExpression(operator="exists", field="/a"), message="a"),
            Rule(rule_id="r2", check=RuleExpression(operator="exists", field="/b"), message="b"),
            Rule(rule_id="r3", check=RuleExpression(operator="exists", field="/c"), message="c"),
        ]
        findings = validator.validate({}, rules)
        # Should stop after first failure
        assert len(findings) == 1

    def test_max_errors_limit(
        self,
        schema_ir: SchemaIR
    ):
        """Test max_errors option limits findings."""
        options = ValidationOptions(max_errors=2)
        validator = RuleValidator(schema_ir, options)

        rules = [
            Rule(rule_id=f"r{i}", check=RuleExpression(operator="exists", field=f"/f{i}"), message=f"m{i}")
            for i in range(5)
        ]
        findings = validator.validate({}, rules)
        # Should stop at 2 errors
        assert len(findings) == 2

    def test_severity_escalation_in_strict_mode(
        self,
        schema_ir: SchemaIR
    ):
        """Test warnings are escalated to errors in strict mode."""
        options = ValidationOptions(profile=ValidationProfile.STRICT)
        validator = RuleValidator(schema_ir, options)

        rule = Rule(
            rule_id="warning_rule",
            severity="warning",
            check=RuleExpression(operator="exists", field="/nonexistent"),
            message="warning message"
        )
        findings = validator.validate({}, [rule])
        assert len(findings) == 1
        assert findings[0].severity == Severity.ERROR  # Escalated from warning

    def test_message_template_formatting(
        self,
        rule_validator: RuleValidator
    ):
        """Test message template with placeholders."""
        rule = Rule(
            rule_id="template_test",
            check=RuleExpression(operator=">", field="/value", value=100),
            message="Default message",
            message_template="Value {{ /value }} is too low (minimum 100)"
        )
        findings = rule_validator.validate({"value": 50}, [rule])
        assert len(findings) == 1
        assert "50" in findings[0].message
        assert "too low" in findings[0].message


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for rule factory functions."""

    def test_create_conditional_required_rule(self):
        """Test create_conditional_required_rule factory."""
        rule = create_conditional_required_rule(
            rule_id="test",
            condition_field="/type",
            condition_value="special",
            required_field="/special_field"
        )
        assert rule.rule_id == "test"
        assert rule.is_conditional()
        assert rule.when.operator == "=="
        assert rule.when.field == "/type"
        assert rule.check.operator == "exists"
        assert rule.check.field == "/special_field"

    def test_create_sum_consistency_rule(self):
        """Test create_sum_consistency_rule factory."""
        rule = create_sum_consistency_rule(
            rule_id="sum_test",
            component_fields=["/a", "/b", "/c"],
            total_field="/total"
        )
        assert rule.rule_id == "sum_test"
        assert rule.check.operator == "=="
        assert len(rule.check.operands) == 2
        assert rule.check.operands[0].operator == "sum"
        assert rule.check.operands[0].fields == ["/a", "/b", "/c"]

    def test_create_range_rule_both_bounds(self):
        """Test create_range_rule with both bounds."""
        rule = create_range_rule(
            rule_id="range_test",
            field="/value",
            minimum=0,
            maximum=100
        )
        assert rule.rule_id == "range_test"
        assert rule.check.operator == "and"
        assert len(rule.check.operands) == 2

    def test_create_range_rule_only_minimum(self):
        """Test create_range_rule with only minimum."""
        rule = create_range_rule(
            rule_id="min_test",
            field="/value",
            minimum=0
        )
        assert rule.check.operator == ">="
        assert rule.check.field == "/value"
        assert rule.check.value == 0

    def test_create_range_rule_exclusive_bounds(self):
        """Test create_range_rule with exclusive bounds."""
        rule = create_range_rule(
            rule_id="exclusive_test",
            field="/value",
            minimum=0,
            maximum=100,
            exclusive_min=True,
            exclusive_max=True
        )
        assert rule.check.operands[0].operator == ">"
        assert rule.check.operands[1].operator == "<"

    def test_create_range_rule_no_bounds_raises(self):
        """Test create_range_rule raises without bounds."""
        with pytest.raises(ValueError, match="At least one"):
            create_range_rule(
                rule_id="no_bounds",
                field="/value"
            )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestRuleValidatorIntegration:
    """Integration tests for RuleValidator with IR rules."""

    def test_validate_with_ir_rules(self):
        """Test validation using rules from schema IR."""
        ir = SchemaIR(
            schema_id="test/integration",
            version="1.0.0",
            schema_hash="a" * 64,
            compiled_at=datetime.now(),
            rule_bindings=[
                RuleBindingIR(
                    rule_id="require_name",
                    severity="error",
                    check={"operator": "exists", "field": "/name"},
                    message="name is required"
                ),
            ]
        )
        options = ValidationOptions()
        validator = RuleValidator(ir, options)

        # Payload missing name
        findings = validator.validate({})
        assert len(findings) == 1
        assert "name" in findings[0].message

        # Payload with name
        findings = validator.validate({"name": "test"})
        assert len(findings) == 0

    def test_validate_with_conditional_ir_rule(self):
        """Test validation with conditional IR rule."""
        ir = SchemaIR(
            schema_id="test/conditional",
            version="1.0.0",
            schema_hash="a" * 64,
            compiled_at=datetime.now(),
            rule_bindings=[
                RuleBindingIR(
                    rule_id="gas_methane",
                    severity="error",
                    when={"operator": "==", "field": "/fuel_type", "value": "gas"},
                    check={"operator": "exists", "field": "/methane_slip"},
                    message="methane_slip required for gas fuel"
                ),
            ]
        )
        options = ValidationOptions()
        validator = RuleValidator(ir, options)

        # Diesel - rule doesn't apply
        findings = validator.validate({"fuel_type": "diesel"})
        assert len(findings) == 0

        # Gas without methane_slip - fails
        findings = validator.validate({"fuel_type": "gas"})
        assert len(findings) == 1

        # Gas with methane_slip - passes
        findings = validator.validate({"fuel_type": "gas", "methane_slip": 0.5})
        assert len(findings) == 0
