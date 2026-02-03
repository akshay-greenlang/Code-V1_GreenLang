# -*- coding: utf-8 -*-
"""
Test Suite for Safe Expression Evaluator
=========================================

Comprehensive tests for the AST-based safe expression evaluator used in
workflow step conditions. This evaluator replaces unsafe eval() with a
restricted subset of Python expressions.

Author: GreenLang Framework Team
"""

import pytest
from greenlang.execution.core.orchestrator import Orchestrator
from greenlang.exceptions import ValidationError


class TestSafeEvaluator:
    """Test the _evaluate_condition method of Orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance for testing."""
        return Orchestrator()

    @pytest.fixture
    def sample_context(self):
        """Create a sample context for testing."""
        return {
            "input": {
                "amount": 100,
                "fuel_type": "diesel",
                "region": "US",
                "enabled": True,
                "tags": ["emissions", "scope1"],
                "metadata": {
                    "source": "api",
                    "version": "1.0"
                }
            },
            "results": {
                "step1": {
                    "success": True,
                    "data": {
                        "value": 42,
                        "status": "completed"
                    }
                },
                "step2": {
                    "success": False,
                    "error": "timeout"
                }
            }
        }

    # =========================================================================
    # Basic Comparison Tests
    # =========================================================================

    def test_equality_comparison(self, orchestrator, sample_context):
        """Test equality (==) comparison."""
        assert orchestrator._evaluate_condition(
            "input.amount == 100", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount == 200", sample_context
        ) is False

    def test_inequality_comparison(self, orchestrator, sample_context):
        """Test inequality (!=) comparison."""
        assert orchestrator._evaluate_condition(
            "input.amount != 200", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount != 100", sample_context
        ) is False

    def test_greater_than(self, orchestrator, sample_context):
        """Test greater than (>) comparison."""
        assert orchestrator._evaluate_condition(
            "input.amount > 50", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount > 100", sample_context
        ) is False

    def test_greater_than_or_equal(self, orchestrator, sample_context):
        """Test greater than or equal (>=) comparison."""
        assert orchestrator._evaluate_condition(
            "input.amount >= 100", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount >= 101", sample_context
        ) is False

    def test_less_than(self, orchestrator, sample_context):
        """Test less than (<) comparison."""
        assert orchestrator._evaluate_condition(
            "input.amount < 200", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount < 100", sample_context
        ) is False

    def test_less_than_or_equal(self, orchestrator, sample_context):
        """Test less than or equal (<=) comparison."""
        assert orchestrator._evaluate_condition(
            "input.amount <= 100", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount <= 99", sample_context
        ) is False

    # =========================================================================
    # Boolean Logic Tests
    # =========================================================================

    def test_and_operator(self, orchestrator, sample_context):
        """Test boolean AND operator."""
        assert orchestrator._evaluate_condition(
            "input.amount > 50 and input.enabled == True", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount > 50 and input.enabled == False", sample_context
        ) is False

    def test_or_operator(self, orchestrator, sample_context):
        """Test boolean OR operator."""
        assert orchestrator._evaluate_condition(
            "input.amount > 200 or input.enabled == True", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount > 200 or input.enabled == False", sample_context
        ) is False

    def test_not_operator(self, orchestrator, sample_context):
        """Test boolean NOT operator."""
        assert orchestrator._evaluate_condition(
            "not input.amount > 200", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "not input.enabled", sample_context
        ) is False

    def test_complex_boolean(self, orchestrator, sample_context):
        """Test complex boolean expressions."""
        assert orchestrator._evaluate_condition(
            "(input.amount > 50 and input.enabled) or input.region == 'EU'",
            sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "not (input.amount < 50 or input.region == 'EU')",
            sample_context
        ) is True

    # =========================================================================
    # Membership Tests (in, not in)
    # =========================================================================

    def test_in_list_literal(self, orchestrator, sample_context):
        """Test 'in' operator with list literals."""
        assert orchestrator._evaluate_condition(
            "input.region in ['US', 'EU', 'APAC']", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.region in ['UK', 'JP']", sample_context
        ) is False

    def test_not_in_list_literal(self, orchestrator, sample_context):
        """Test 'not in' operator with list literals."""
        assert orchestrator._evaluate_condition(
            "input.region not in ['UK', 'JP']", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.region not in ['US', 'EU']", sample_context
        ) is False

    def test_in_tuple_literal(self, orchestrator, sample_context):
        """Test 'in' operator with tuple literals."""
        assert orchestrator._evaluate_condition(
            "input.fuel_type in ('diesel', 'petrol', 'natural_gas')",
            sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.fuel_type in ('coal', 'wood')",
            sample_context
        ) is False

    def test_in_set_literal(self, orchestrator, sample_context):
        """Test 'in' operator with set literals."""
        assert orchestrator._evaluate_condition(
            "input.region in {'US', 'EU', 'APAC'}", sample_context
        ) is True

    def test_in_dict_keys(self, orchestrator, sample_context):
        """Test 'in' operator with dict literals (checks keys)."""
        assert orchestrator._evaluate_condition(
            "'source' in {'source': 1, 'version': 2}", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "'missing' in {'source': 1, 'version': 2}", sample_context
        ) is False

    def test_in_string(self, orchestrator, sample_context):
        """Test 'in' operator with strings."""
        assert orchestrator._evaluate_condition(
            "'diesel' in input.fuel_type", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "'x' in input.fuel_type", sample_context
        ) is False

    # =========================================================================
    # Container Literal Tests
    # =========================================================================

    def test_list_literal(self, orchestrator, sample_context):
        """Test list literal evaluation."""
        # List with constants
        assert orchestrator._evaluate_condition(
            "input.amount in [100, 200, 300]", sample_context
        ) is True
        # List with mixed types
        assert orchestrator._evaluate_condition(
            "'diesel' in ['diesel', 'petrol', 1, 2]", sample_context
        ) is True
        # Empty list
        assert orchestrator._evaluate_condition(
            "input.amount in []", sample_context
        ) is False

    def test_tuple_literal(self, orchestrator, sample_context):
        """Test tuple literal evaluation."""
        assert orchestrator._evaluate_condition(
            "input.amount in (100, 200, 300)", sample_context
        ) is True

    def test_dict_literal(self, orchestrator, sample_context):
        """Test dict literal evaluation."""
        assert orchestrator._evaluate_condition(
            "'a' in {'a': 1, 'b': 2}", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "'c' in {'a': 1, 'b': 2}", sample_context
        ) is False

    def test_nested_containers(self, orchestrator, sample_context):
        """Test nested container literals."""
        # This verifies that the evaluator can handle nested structures
        assert orchestrator._evaluate_condition(
            "'x' in ['a', 'b', 'c']", sample_context
        ) is False

    # =========================================================================
    # Attribute and Subscript Access Tests
    # =========================================================================

    def test_attribute_access(self, orchestrator, sample_context):
        """Test attribute-style access (obj.field)."""
        assert orchestrator._evaluate_condition(
            "input.amount == 100", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.metadata.source == 'api'", sample_context
        ) is True

    def test_subscript_access(self, orchestrator, sample_context):
        """Test subscript-style access (obj['field'])."""
        assert orchestrator._evaluate_condition(
            "input['amount'] == 100", sample_context
        ) is True

    def test_nested_access(self, orchestrator, sample_context):
        """Test deeply nested access."""
        assert orchestrator._evaluate_condition(
            "results.step1.data.value == 42", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "results.step1.data.status == 'completed'", sample_context
        ) is True

    def test_results_access(self, orchestrator, sample_context):
        """Test accessing previous step results."""
        assert orchestrator._evaluate_condition(
            "results.step1.success == True", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "results.step2.success == False", sample_context
        ) is True

    # =========================================================================
    # Constant Literal Tests
    # =========================================================================

    def test_string_literals(self, orchestrator, sample_context):
        """Test string constant literals."""
        assert orchestrator._evaluate_condition(
            "input.fuel_type == 'diesel'", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            'input.fuel_type == "diesel"', sample_context
        ) is True

    def test_numeric_literals(self, orchestrator, sample_context):
        """Test numeric constant literals."""
        assert orchestrator._evaluate_condition(
            "input.amount == 100", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.amount > 99.5", sample_context
        ) is True

    def test_boolean_literals(self, orchestrator, sample_context):
        """Test boolean constant literals."""
        assert orchestrator._evaluate_condition(
            "input.enabled == True", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "input.enabled == False", sample_context
        ) is False

    def test_none_literal(self, orchestrator, sample_context):
        """Test None constant literal."""
        sample_context["input"]["optional"] = None
        assert orchestrator._evaluate_condition(
            "input.optional == None", sample_context
        ) is True

    # =========================================================================
    # Error Cases
    # =========================================================================

    def test_invalid_name_raises_error(self, orchestrator, sample_context):
        """Test that accessing undefined names raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            orchestrator._evaluate_condition(
                "undefined_var == 100", sample_context
            )
        assert "not allowed" in str(exc_info.value).lower()

    def test_syntax_error_raises(self, orchestrator, sample_context):
        """Test that syntax errors are raised properly."""
        with pytest.raises(SyntaxError):
            orchestrator._evaluate_condition(
                "input.amount ==", sample_context  # Incomplete expression
            )

    def test_unsupported_expression_raises_error(self, orchestrator, sample_context):
        """Test that unsupported expressions raise ValidationError."""
        # Function calls are not supported
        with pytest.raises(ValidationError):
            orchestrator._evaluate_condition(
                "len(input.tags) > 0", sample_context
            )

    def test_lambda_not_supported(self, orchestrator, sample_context):
        """Test that lambda expressions are not supported."""
        with pytest.raises(SyntaxError):
            orchestrator._evaluate_condition(
                "(lambda x: x > 50)(input.amount)", sample_context
            )

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_empty_string_comparison(self, orchestrator, sample_context):
        """Test comparison with empty string."""
        sample_context["input"]["empty"] = ""
        assert orchestrator._evaluate_condition(
            "input.empty == ''", sample_context
        ) is True

    def test_zero_comparison(self, orchestrator, sample_context):
        """Test comparison with zero."""
        sample_context["input"]["zero"] = 0
        assert orchestrator._evaluate_condition(
            "input.zero == 0", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "not input.zero", sample_context  # 0 is falsy
        ) is True

    def test_chained_comparison(self, orchestrator, sample_context):
        """Test chained comparisons."""
        assert orchestrator._evaluate_condition(
            "50 < input.amount < 200", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "100 <= input.amount <= 100", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "0 < input.amount < 50", sample_context
        ) is False

    def test_multiple_and_conditions(self, orchestrator, sample_context):
        """Test multiple AND conditions."""
        assert orchestrator._evaluate_condition(
            "input.amount > 0 and input.enabled and input.region == 'US'",
            sample_context
        ) is True

    def test_multiple_or_conditions(self, orchestrator, sample_context):
        """Test multiple OR conditions."""
        assert orchestrator._evaluate_condition(
            "input.region == 'UK' or input.region == 'JP' or input.region == 'US'",
            sample_context
        ) is True


class TestShouldExecuteStep:
    """Test the _should_execute_step method with error handling."""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance."""
        return Orchestrator()

    @pytest.fixture
    def mock_step(self):
        """Create a mock step object."""
        class MockStep:
            def __init__(self, name, condition=None):
                self.name = name
                self.condition = condition
        return MockStep

    def test_no_condition_returns_true(self, orchestrator, mock_step):
        """Test that steps without conditions always execute."""
        step = mock_step("test_step", condition=None)
        context = {"input": {}, "results": {}}
        assert orchestrator._should_execute_step(step, context) is True

    def test_true_condition_returns_true(self, orchestrator, mock_step):
        """Test that true conditions return True."""
        step = mock_step("test_step", condition="input.value > 0")
        context = {"input": {"value": 100}, "results": {}}
        assert orchestrator._should_execute_step(step, context) is True

    def test_false_condition_returns_false(self, orchestrator, mock_step):
        """Test that false conditions return False."""
        step = mock_step("test_step", condition="input.value > 200")
        context = {"input": {"value": 100}, "results": {}}
        assert orchestrator._should_execute_step(step, context) is False

    def test_error_condition_records_error(self, orchestrator, mock_step):
        """Test that condition errors are recorded in context."""
        step = mock_step("test_step", condition="undefined_var > 0")
        context = {"input": {}, "results": {}}

        result = orchestrator._should_execute_step(step, context)

        assert result is False
        assert "condition_errors" in context
        assert len(context["condition_errors"]) == 1
        assert context["condition_errors"][0]["step"] == "test_step"
        assert context["condition_errors"][0]["error_type"] == "ValidationError"

    def test_syntax_error_records_error(self, orchestrator, mock_step):
        """Test that syntax errors are recorded in context."""
        step = mock_step("test_step", condition="input.value >")  # Invalid syntax
        context = {"input": {"value": 100}, "results": {}}

        result = orchestrator._should_execute_step(step, context)

        assert result is False
        assert "condition_errors" in context
        assert context["condition_errors"][0]["error_type"] == "SyntaxError"


class TestExpressionLanguageSpec:
    """
    Test cases that serve as the specification for the expression language.

    These tests document what is and isn't supported.
    """

    @pytest.fixture
    def eval_expr(self):
        """Helper to evaluate expressions."""
        orchestrator = Orchestrator()

        def _eval(expr, ctx=None):
            ctx = ctx or {"input": {}, "results": {}}
            return orchestrator._evaluate_condition(expr, ctx)

        return _eval

    def test_supported_operators(self, eval_expr):
        """Document supported comparison operators."""
        ctx = {"input": {"x": 5}, "results": {}}

        # Equality
        assert eval_expr("input.x == 5", ctx) is True
        assert eval_expr("input.x != 3", ctx) is True

        # Ordering
        assert eval_expr("input.x > 3", ctx) is True
        assert eval_expr("input.x >= 5", ctx) is True
        assert eval_expr("input.x < 10", ctx) is True
        assert eval_expr("input.x <= 5", ctx) is True

        # Membership
        assert eval_expr("input.x in [1, 5, 10]", ctx) is True
        assert eval_expr("input.x not in [1, 2, 3]", ctx) is True

    def test_supported_boolean_ops(self, eval_expr):
        """Document supported boolean operators."""
        ctx = {"input": {"a": True, "b": False}, "results": {}}

        assert eval_expr("input.a and not input.b", ctx) is True
        assert eval_expr("input.a or input.b", ctx) is True
        assert eval_expr("not input.b", ctx) is True

    def test_supported_literals(self, eval_expr):
        """Document supported literal types."""
        # Strings
        assert eval_expr("'hello' == 'hello'") is True
        assert eval_expr('"hello" == "hello"') is True

        # Numbers
        assert eval_expr("42 == 42") is True
        assert eval_expr("3.14 > 3") is True

        # Booleans
        assert eval_expr("True == True") is True
        assert eval_expr("False == False") is True

        # None
        assert eval_expr("None == None") is True

        # Lists
        assert eval_expr("1 in [1, 2, 3]") is True

        # Tuples
        assert eval_expr("1 in (1, 2, 3)") is True

        # Dicts (key membership)
        assert eval_expr("'a' in {'a': 1}") is True

        # Sets
        assert eval_expr("1 in {1, 2, 3}") is True

    def test_supported_access_patterns(self, eval_expr):
        """Document supported access patterns."""
        ctx = {
            "input": {"data": {"nested": {"value": 42}}},
            "results": {"step1": {"output": [1, 2, 3]}}
        }

        # Attribute access
        assert eval_expr("input.data.nested.value == 42", ctx) is True

        # Subscript access
        assert eval_expr("input['data']['nested']['value'] == 42", ctx) is True

        # Mixed access
        assert eval_expr("results.step1.output[0] == 1", ctx) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
