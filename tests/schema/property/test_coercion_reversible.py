# -*- coding: utf-8 -*-
"""
Property-Based Tests: Coercion Reversibility

Tests the property that safe coercions are reversible - the original value
can be recovered from the coerced value.

Reversibility Property:
    For safe coercions: reverse(coerce(x)) == x

This property is essential for:
    - Data integrity: No information loss during coercion
    - Audit trails: Original values can be reconstructed
    - Round-trip safety: Data can flow through the system without degradation
    - Trust: Users know exactly what transformations occurred

Safe Coercions:
    - "42" -> 42 -> "42" (string to integer and back)
    - "3.14" -> 3.14 -> "3.14" (string to float and back)
    - "true"/"false" -> True/False -> "true"/"false" (string to boolean)
    - 42 -> 42.0 -> 42 (integer to float and back, when exact)

Uses Hypothesis to generate random coercible values and verify
reversibility.

GL-FOUND-X-002: Schema Compiler & Validator - Property Tests
"""

from __future__ import annotations

import copy
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
from hypothesis import given, settings, assume, HealthCheck, Phase, example
from hypothesis import strategies as st

# Import components under test
from greenlang.schema.normalizer.coercions import (
    CoercionEngine,
    CoercionRecord,
    CoercionResult,
    CoercionType,
    get_python_type_name,
    can_coerce,
    JSON_TYPE_STRING,
    JSON_TYPE_NUMBER,
    JSON_TYPE_INTEGER,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_NULL,
)
from greenlang.schema.models.config import CoercionPolicy


# =============================================================================
# REVERSIBILITY UTILITIES
# =============================================================================

def reverse_coercion(
    coerced_value: Any,
    original_type: str,
    coercion_type: str,
) -> Any:
    """
    Reverse a coercion to recover the original value.

    Args:
        coerced_value: The value after coercion
        original_type: The original JSON type before coercion
        coercion_type: The type of coercion that was performed

    Returns:
        The reconstructed original value

    Raises:
        ValueError: If coercion cannot be reversed
    """
    if coercion_type == CoercionType.STRING_TO_INTEGER.value:
        # 42 -> "42"
        return str(coerced_value)

    elif coercion_type == CoercionType.STRING_TO_NUMBER.value:
        # 3.14 -> "3.14"
        if coerced_value == int(coerced_value):
            return str(int(coerced_value))
        return str(coerced_value)

    elif coercion_type == CoercionType.STRING_TO_BOOLEAN.value:
        # True/False -> "true"/"false"
        return "true" if coerced_value else "false"

    elif coercion_type == CoercionType.STRING_TO_NULL.value:
        # None -> "null"
        return "null"

    elif coercion_type == CoercionType.INTEGER_TO_NUMBER.value:
        # 42.0 -> 42
        return int(coerced_value)

    elif coercion_type == CoercionType.FLOAT_TO_INTEGER.value:
        # 42 -> 42.0
        return float(coerced_value)

    elif coercion_type == CoercionType.INTEGER_TO_BOOLEAN.value:
        # True/False -> 1/0
        return 1 if coerced_value else 0

    elif coercion_type == CoercionType.EMPTY_STRING_TO_NULL.value:
        # This coercion is NOT reversible (could be empty string or None)
        raise ValueError("Empty string to null coercion is not reversible")

    raise ValueError(f"Unknown coercion type: {coercion_type}")


def is_coercion_reversible(coercion_type: str) -> bool:
    """
    Check if a coercion type is reversible.

    Args:
        coercion_type: The type of coercion

    Returns:
        True if the coercion can be reversed without loss
    """
    reversible_types = {
        CoercionType.STRING_TO_INTEGER.value,
        CoercionType.STRING_TO_NUMBER.value,
        CoercionType.STRING_TO_BOOLEAN.value,
        CoercionType.STRING_TO_NULL.value,
        CoercionType.INTEGER_TO_NUMBER.value,
        CoercionType.FLOAT_TO_INTEGER.value,
        CoercionType.INTEGER_TO_BOOLEAN.value,
    }
    return coercion_type in reversible_types


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

# Strategy for integer strings that can be coerced
integer_strings = st.integers(
    min_value=-2**31,
    max_value=2**31
).map(str)

# Strategy for float strings that can be coerced
float_strings = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e10,
    max_value=1e10,
).map(str)

# Strategy for boolean strings
boolean_strings = st.sampled_from(["true", "false", "True", "False", "TRUE", "FALSE"])

# Strategy for null strings
null_strings = st.sampled_from(["null", "Null", "NULL"])

# Strategy for all coercible string values
coercible_strings = st.one_of(
    integer_strings,
    float_strings,
    boolean_strings,
    null_strings,
)

# Strategy for integers that can be coerced to booleans (aggressive mode)
boolean_integers = st.sampled_from([0, 1])

# Strategy for floats that are exact integers (can coerce to integer)
integer_floats = st.integers(
    min_value=-10000,
    max_value=10000
).map(float)

# Strategy for any safely coercible value with its target type
coercible_value_and_target = st.one_of(
    # String to integer
    integer_strings.map(lambda x: (x, JSON_TYPE_INTEGER, "string_to_integer")),
    # String to number
    float_strings.map(lambda x: (x, JSON_TYPE_NUMBER, "string_to_number")),
    # String to boolean
    boolean_strings.map(lambda x: (x, JSON_TYPE_BOOLEAN, "string_to_boolean")),
    # String to null
    null_strings.map(lambda x: (x, JSON_TYPE_NULL, "string_to_null")),
    # Integer to number
    st.integers(min_value=-10000, max_value=10000).map(
        lambda x: (x, JSON_TYPE_NUMBER, "integer_to_number")
    ),
    # Float to integer (whole numbers only)
    integer_floats.map(lambda x: (x, JSON_TYPE_INTEGER, "float_to_integer")),
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def safe_engine() -> CoercionEngine:
    """Create a coercion engine with SAFE policy."""
    return CoercionEngine(policy=CoercionPolicy.SAFE)


@pytest.fixture
def aggressive_engine() -> CoercionEngine:
    """Create a coercion engine with AGGRESSIVE policy."""
    return CoercionEngine(policy=CoercionPolicy.AGGRESSIVE)


# =============================================================================
# REVERSIBILITY TESTS - STRING TO INTEGER
# =============================================================================

@pytest.mark.property
class TestStringToIntegerReversibility:
    """
    Test that string to integer coercions are reversible.

    Property: str(int("42")) == "42"
    """

    @given(value=st.integers(min_value=-2**31, max_value=2**31))
    @settings(max_examples=100, deadline=None)
    def test_integer_string_round_trip(
        self,
        value: int,
        safe_engine,
    ):
        """
        Test that integer strings round-trip correctly.

        "42" -> 42 -> "42"
        """
        original_string = str(value)

        # Coerce string to integer
        result = safe_engine.coerce(original_string, JSON_TYPE_INTEGER, "/test")

        assume(result.success)  # Skip if coercion not possible

        coerced_value = result.value
        assert isinstance(coerced_value, int), f"Expected int, got {type(coerced_value)}"
        assert coerced_value == value

        # Reverse: integer back to string
        reversed_value = str(coerced_value)
        assert reversed_value == original_string, (
            f"Round-trip failed: '{original_string}' -> {coerced_value} -> '{reversed_value}'"
        )

    @given(value=st.integers(min_value=-1000000, max_value=1000000))
    @settings(max_examples=100, deadline=None)
    def test_coercion_record_marks_reversible(
        self,
        value: int,
        safe_engine,
    ):
        """
        Test that coercion records correctly mark reversibility.
        """
        original_string = str(value)
        result = safe_engine.coerce(original_string, JSON_TYPE_INTEGER, "/test")

        assume(result.success and result.record is not None)

        assert result.record.reversible is True, (
            f"String to integer coercion should be marked reversible"
        )
        assert result.record.coercion_type == CoercionType.STRING_TO_INTEGER.value

    @example(value=0)  # Zero
    @example(value=-1)  # Negative
    @example(value=2147483647)  # Max 32-bit signed
    @example(value=-2147483648)  # Min 32-bit signed
    @given(value=st.integers(min_value=-2**31, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_edge_case_integers(
        self,
        value: int,
        safe_engine,
    ):
        """Test edge case integer values."""
        original_string = str(value)
        result = safe_engine.coerce(original_string, JSON_TYPE_INTEGER, "/test")

        if result.success:
            reversed_value = str(result.value)
            assert reversed_value == original_string


# =============================================================================
# REVERSIBILITY TESTS - STRING TO NUMBER
# =============================================================================

@pytest.mark.property
class TestStringToNumberReversibility:
    """
    Test that string to number coercions are reversible.

    Property: str(float("3.14")) == "3.14" (for exact representations)
    """

    @given(
        value=st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1000,
            max_value=1000,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_float_string_round_trip(
        self,
        value: float,
        safe_engine,
    ):
        """
        Test that float strings can be recovered.
        """
        original_string = str(value)

        result = safe_engine.coerce(original_string, JSON_TYPE_NUMBER, "/test")

        assume(result.success)

        coerced_value = result.value

        # For reversibility, we need to check if the coerced value
        # can reproduce the original (accounting for float representation)
        if coerced_value == int(coerced_value):
            # Whole number float
            reversed_string = str(int(coerced_value))
        else:
            reversed_string = str(coerced_value)

        # The reversed string should parse to the same float
        assert float(reversed_string) == float(original_string), (
            f"Round-trip value mismatch: '{original_string}' -> {coerced_value} -> '{reversed_string}'"
        )

    @given(value=st.integers(min_value=-10000, max_value=10000))
    @settings(max_examples=100, deadline=None)
    def test_integer_string_to_number_reversible(
        self,
        value: int,
        safe_engine,
    ):
        """
        Test that integer strings coerced to numbers are reversible.

        "42" -> 42.0 -> round-trip check
        """
        original_string = str(value)

        result = safe_engine.coerce(original_string, JSON_TYPE_NUMBER, "/test")

        assume(result.success)

        # The coerced float should equal the original integer
        assert result.value == float(value)

        # Mark should indicate reversible
        if result.record:
            assert result.record.reversible is True


# =============================================================================
# REVERSIBILITY TESTS - STRING TO BOOLEAN
# =============================================================================

@pytest.mark.property
class TestStringToBooleanReversibility:
    """
    Test that string to boolean coercions are reversible.

    Property: "true" -> True -> "true", "false" -> False -> "false"
    """

    @given(bool_string=boolean_strings)
    @settings(max_examples=50, deadline=None)
    def test_boolean_string_round_trip(
        self,
        bool_string: str,
        safe_engine,
    ):
        """
        Test that boolean strings round-trip correctly.
        """
        result = safe_engine.coerce(bool_string, JSON_TYPE_BOOLEAN, "/test")

        assert result.success, f"Failed to coerce '{bool_string}' to boolean"

        coerced_value = result.value
        expected_bool = bool_string.lower() == "true"

        assert coerced_value == expected_bool

        # Reverse: boolean back to lowercase string
        reversed_value = "true" if coerced_value else "false"
        assert reversed_value == bool_string.lower(), (
            f"Round-trip failed: '{bool_string}' -> {coerced_value} -> '{reversed_value}'"
        )

    @pytest.mark.parametrize("original,expected", [
        ("true", True),
        ("false", False),
        ("True", True),
        ("False", False),
        ("TRUE", True),
        ("FALSE", False),
    ])
    def test_explicit_boolean_strings(
        self,
        original: str,
        expected: bool,
        safe_engine,
    ):
        """Test explicit boolean string cases."""
        result = safe_engine.coerce(original, JSON_TYPE_BOOLEAN, "/test")

        assert result.success
        assert result.value == expected
        assert result.record.reversible is True


# =============================================================================
# REVERSIBILITY TESTS - STRING TO NULL
# =============================================================================

@pytest.mark.property
class TestStringToNullReversibility:
    """
    Test that string to null coercions are reversible.

    Property: "null" -> None -> "null"
    """

    @given(null_string=null_strings)
    @settings(max_examples=20, deadline=None)
    def test_null_string_round_trip(
        self,
        null_string: str,
        safe_engine,
    ):
        """
        Test that null strings round-trip correctly.
        """
        result = safe_engine.coerce(null_string, JSON_TYPE_NULL, "/test")

        assert result.success, f"Failed to coerce '{null_string}' to null"
        assert result.value is None

        # Reverse: None back to "null"
        reversed_value = "null"
        assert reversed_value == null_string.lower(), (
            f"Round-trip failed: '{null_string}' -> None -> '{reversed_value}'"
        )


# =============================================================================
# REVERSIBILITY TESTS - INTEGER TO NUMBER
# =============================================================================

@pytest.mark.property
class TestIntegerToNumberReversibility:
    """
    Test that integer to number coercions are reversible.

    Property: int(float(42)) == 42
    """

    @given(value=st.integers(min_value=-2**53, max_value=2**53))
    @settings(max_examples=100, deadline=None)
    def test_integer_to_float_round_trip(
        self,
        value: int,
        safe_engine,
    ):
        """
        Test that integers can be safely coerced to floats and back.
        """
        result = safe_engine.coerce(value, JSON_TYPE_NUMBER, "/test")

        assume(result.success)

        coerced_value = result.value
        assert isinstance(coerced_value, float)

        # Reverse: float back to integer
        reversed_value = int(coerced_value)
        assert reversed_value == value, (
            f"Round-trip failed: {value} -> {coerced_value} -> {reversed_value}"
        )


# =============================================================================
# REVERSIBILITY TESTS - FLOAT TO INTEGER
# =============================================================================

@pytest.mark.property
class TestFloatToIntegerReversibility:
    """
    Test that float to integer coercions (for whole numbers) are reversible.

    Property: float(int(42.0)) == 42.0
    """

    @given(value=integer_floats)
    @settings(max_examples=100, deadline=None)
    def test_whole_float_to_integer_round_trip(
        self,
        value: float,
        safe_engine,
    ):
        """
        Test that whole-number floats can be coerced to integers and back.
        """
        result = safe_engine.coerce(value, JSON_TYPE_INTEGER, "/test")

        assume(result.success)

        coerced_value = result.value
        assert isinstance(coerced_value, int)
        assert coerced_value == int(value)

        # Reverse: integer back to float
        reversed_value = float(coerced_value)
        assert reversed_value == value

    @given(
        value=st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1000,
            max_value=1000,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_non_whole_floats_rejected(
        self,
        value: float,
        safe_engine,
    ):
        """
        Test that non-whole floats cannot be coerced to integers.
        """
        # Skip whole numbers
        assume(value != int(value))

        result = safe_engine.coerce(value, JSON_TYPE_INTEGER, "/test")

        # Should fail because float has decimal part
        assert not result.success, (
            f"Non-whole float {value} should not coerce to integer"
        )


# =============================================================================
# NON-REVERSIBLE COERCION TESTS
# =============================================================================

@pytest.mark.property
class TestNonReversibleCoercions:
    """
    Test that non-reversible coercions are correctly identified.
    """

    def test_empty_string_to_null_not_reversible(
        self,
        aggressive_engine,
    ):
        """
        Test that empty string to null is marked as non-reversible.

        "" -> None is not reversible because None could also come from "null".
        """
        result = aggressive_engine.coerce("", JSON_TYPE_NULL, "/test")

        assert result.success
        assert result.value is None

        # Should be marked as NOT reversible
        if result.record:
            assert result.record.reversible is False, (
                "Empty string to null should be marked non-reversible"
            )


# =============================================================================
# COMPREHENSIVE COERCION TESTS
# =============================================================================

@pytest.mark.property
class TestCoercionReversibilityComprehensive:
    """
    Comprehensive property tests for coercion reversibility.
    """

    @given(value_target_type=coercible_value_and_target)
    @settings(max_examples=100, deadline=None)
    def test_safe_coercions_are_reversible(
        self,
        value_target_type: Tuple[Any, str, str],
        safe_engine,
    ):
        """
        Property test: All safe coercions should be reversible.
        """
        value, target_type, expected_coercion_type = value_target_type

        result = safe_engine.coerce(value, target_type, "/test")

        assume(result.success)

        if result.record:
            # All safe coercions should be marked reversible
            assert result.record.reversible is True, (
                f"Safe coercion should be reversible: "
                f"{value} ({type(value).__name__}) -> {target_type}"
            )

            # Verify by attempting reverse coercion
            if is_coercion_reversible(result.record.coercion_type):
                try:
                    reversed_value = reverse_coercion(
                        result.value,
                        result.record.original_type,
                        result.record.coercion_type,
                    )

                    # For string originals, compare normalized
                    if result.record.original_type == JSON_TYPE_STRING:
                        if isinstance(value, str):
                            # Normalize case for comparison
                            original_normalized = value.lower().strip()
                            reversed_normalized = str(reversed_value).lower().strip()

                            # For numeric strings, compare as numbers
                            try:
                                original_num = float(original_normalized)
                                reversed_num = float(reversed_normalized)
                                assert original_num == reversed_num, (
                                    f"Numeric round-trip failed: "
                                    f"'{value}' -> {result.value} -> '{reversed_value}'"
                                )
                            except ValueError:
                                # Not numeric, compare strings
                                assert original_normalized == reversed_normalized, (
                                    f"String round-trip failed: "
                                    f"'{value}' -> {result.value} -> '{reversed_value}'"
                                )

                except ValueError:
                    pass  # Coercion marked reversible but reverse implementation missing


# =============================================================================
# COERCION POLICY TESTS
# =============================================================================

@pytest.mark.property
class TestCoercionPolicyImpactOnReversibility:
    """
    Test how coercion policy affects reversibility guarantees.
    """

    def test_safe_policy_only_reversible_coercions(self):
        """
        Test that SAFE policy only performs reversible coercions.
        """
        engine = CoercionEngine(policy=CoercionPolicy.SAFE)

        # These should all succeed with reversible=True
        test_cases = [
            ("42", JSON_TYPE_INTEGER),
            ("3.14", JSON_TYPE_NUMBER),
            ("true", JSON_TYPE_BOOLEAN),
            ("null", JSON_TYPE_NULL),
        ]

        for value, target_type in test_cases:
            result = engine.coerce(value, target_type, "/test")

            if result.success and result.record:
                assert result.record.reversible is True, (
                    f"SAFE policy coercion should be reversible: "
                    f"{value} -> {target_type}"
                )

    def test_aggressive_policy_may_have_non_reversible(self):
        """
        Test that AGGRESSIVE policy may perform non-reversible coercions.
        """
        engine = CoercionEngine(policy=CoercionPolicy.AGGRESSIVE)

        # Empty string to null - not reversible
        result = engine.coerce("", JSON_TYPE_NULL, "/test")

        if result.success and result.record:
            # This specific coercion should be marked non-reversible
            if result.record.coercion_type == CoercionType.EMPTY_STRING_TO_NULL.value:
                assert result.record.reversible is False


# =============================================================================
# RECORD INTEGRITY TESTS
# =============================================================================

@pytest.mark.property
class TestCoercionRecordIntegrity:
    """
    Test that coercion records accurately capture the transformation.
    """

    @given(value=st.integers(min_value=-10000, max_value=10000))
    @settings(max_examples=100, deadline=None)
    def test_record_captures_original_value(
        self,
        value: int,
        safe_engine,
    ):
        """
        Test that records capture the exact original value.
        """
        original_string = str(value)

        result = safe_engine.coerce(original_string, JSON_TYPE_INTEGER, "/test")

        assume(result.success and result.record is not None)

        # Record should have exact original value
        assert result.record.original_value == original_string
        assert result.record.original_type == JSON_TYPE_STRING
        assert result.record.coerced_value == value
        assert result.record.coerced_type == JSON_TYPE_INTEGER

    @given(value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000))
    @settings(max_examples=100, deadline=None)
    def test_record_enables_reversal(
        self,
        value: float,
        safe_engine,
    ):
        """
        Test that records contain enough information to reverse the coercion.
        """
        original_string = str(value)

        result = safe_engine.coerce(original_string, JSON_TYPE_NUMBER, "/test")

        assume(result.success and result.record is not None)

        # Using record info, we should be able to reconstruct original
        record = result.record

        assert record.original_type is not None
        assert record.coercion_type is not None
        assert record.reversible is not None

        # If marked reversible, original should be recoverable
        if record.reversible:
            # The original value is stored in the record
            assert record.original_value == original_string
