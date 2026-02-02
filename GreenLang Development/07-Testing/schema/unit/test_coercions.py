# -*- coding: utf-8 -*-
"""
Unit Tests for Type Coercion Engine.

Tests for greenlang/schema/normalizer/coercions.py
GL-FOUND-X-002: Schema Compiler & Validator - Task 3.1

Tests cover:
    - Safe coercions (string -> int/float/bool/null)
    - Integer to float coercion
    - Float to integer coercion (whole numbers only)
    - Aggressive mode coercions
    - Policy enforcement (OFF, SAFE, AGGRESSIVE)
    - Coercion record tracking
    - Edge cases and error handling

Author: GreenLang Framework Team
Version: 0.1.0
"""

import pytest
from decimal import Decimal

from greenlang.schema.normalizer.coercions import (
    CoercionEngine,
    CoercionRecord,
    CoercionResult,
    CoercionType,
    JSON_TYPE_ARRAY,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_INTEGER,
    JSON_TYPE_NULL,
    JSON_TYPE_NUMBER,
    JSON_TYPE_OBJECT,
    JSON_TYPE_STRING,
    can_coerce,
    get_python_type_name,
)
from greenlang.schema.models.config import CoercionPolicy


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def safe_engine() -> CoercionEngine:
    """Create CoercionEngine with SAFE policy."""
    return CoercionEngine(policy=CoercionPolicy.SAFE)


@pytest.fixture
def aggressive_engine() -> CoercionEngine:
    """Create CoercionEngine with AGGRESSIVE policy."""
    return CoercionEngine(policy=CoercionPolicy.AGGRESSIVE)


@pytest.fixture
def off_engine() -> CoercionEngine:
    """Create CoercionEngine with OFF policy."""
    return CoercionEngine(policy=CoercionPolicy.OFF)


# =============================================================================
# TEST: get_python_type_name
# =============================================================================


class TestGetPythonTypeName:
    """Tests for get_python_type_name utility function."""

    def test_string(self):
        """Test string type detection."""
        assert get_python_type_name("hello") == JSON_TYPE_STRING
        assert get_python_type_name("") == JSON_TYPE_STRING
        assert get_python_type_name("42") == JSON_TYPE_STRING

    def test_integer(self):
        """Test integer type detection."""
        assert get_python_type_name(42) == JSON_TYPE_INTEGER
        assert get_python_type_name(0) == JSON_TYPE_INTEGER
        assert get_python_type_name(-123) == JSON_TYPE_INTEGER

    def test_float(self):
        """Test float type detection."""
        assert get_python_type_name(3.14) == JSON_TYPE_NUMBER
        assert get_python_type_name(0.0) == JSON_TYPE_NUMBER
        assert get_python_type_name(-1.5) == JSON_TYPE_NUMBER

    def test_boolean(self):
        """Test boolean type detection (before int check)."""
        assert get_python_type_name(True) == JSON_TYPE_BOOLEAN
        assert get_python_type_name(False) == JSON_TYPE_BOOLEAN

    def test_null(self):
        """Test null type detection."""
        assert get_python_type_name(None) == JSON_TYPE_NULL

    def test_array(self):
        """Test array type detection."""
        assert get_python_type_name([]) == JSON_TYPE_ARRAY
        assert get_python_type_name([1, 2, 3]) == JSON_TYPE_ARRAY

    def test_object(self):
        """Test object type detection."""
        assert get_python_type_name({}) == JSON_TYPE_OBJECT
        assert get_python_type_name({"key": "value"}) == JSON_TYPE_OBJECT


# =============================================================================
# TEST: STRING TO INTEGER COERCION
# =============================================================================


class TestStringToIntegerCoercion:
    """Tests for string to integer coercion."""

    def test_basic_positive_integer(self, safe_engine: CoercionEngine):
        """Test coercing positive integer string."""
        result = safe_engine.coerce("42", JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == 42
        assert result.record is not None
        assert result.record.original_value == "42"
        assert result.record.coerced_value == 42
        assert result.record.reversible is True
        assert result.record.coercion_type == CoercionType.STRING_TO_INTEGER.value

    def test_basic_negative_integer(self, safe_engine: CoercionEngine):
        """Test coercing negative integer string."""
        result = safe_engine.coerce("-123", JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == -123
        assert result.record.reversible is True

    def test_zero_integer(self, safe_engine: CoercionEngine):
        """Test coercing zero string."""
        result = safe_engine.coerce("0", JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == 0

    def test_float_string_fails(self, safe_engine: CoercionEngine):
        """Test that float string fails integer coercion."""
        result = safe_engine.coerce("3.14", JSON_TYPE_INTEGER, "/value")

        assert result.success is False
        assert result.error is not None
        assert "not an exact integer" in result.error

    def test_leading_zeros_fail(self, safe_engine: CoercionEngine):
        """Test that leading zeros fail."""
        result = safe_engine.coerce("007", JSON_TYPE_INTEGER, "/value")

        assert result.success is False
        assert result.error is not None

    def test_whitespace_trimmed(self, safe_engine: CoercionEngine):
        """Test that whitespace is handled correctly."""
        result = safe_engine.coerce("  42  ", JSON_TYPE_INTEGER, "/value")

        # Whitespace is stripped before validation, but roundtrip check
        # uses stripped value, so coercion succeeds
        assert result.success is True
        assert result.value == 42
        assert result.record is not None
        # Original value preserved in record
        assert result.record.original_value == "  42  "

    def test_empty_string_fails(self, safe_engine: CoercionEngine):
        """Test that empty string fails."""
        result = safe_engine.coerce("", JSON_TYPE_INTEGER, "/value")

        assert result.success is False

    def test_non_numeric_fails(self, safe_engine: CoercionEngine):
        """Test that non-numeric string fails."""
        result = safe_engine.coerce("hello", JSON_TYPE_INTEGER, "/value")

        assert result.success is False

    def test_scientific_notation_fails(self, safe_engine: CoercionEngine):
        """Test that scientific notation fails for integer."""
        result = safe_engine.coerce("1e10", JSON_TYPE_INTEGER, "/value")

        assert result.success is False

    def test_large_integer(self, safe_engine: CoercionEngine):
        """Test coercing large integer string."""
        result = safe_engine.coerce("999999999999", JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == 999999999999


# =============================================================================
# TEST: STRING TO NUMBER (FLOAT) COERCION
# =============================================================================


class TestStringToNumberCoercion:
    """Tests for string to number coercion."""

    def test_basic_float(self, safe_engine: CoercionEngine):
        """Test coercing float string."""
        result = safe_engine.coerce("3.14", JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 3.14
        assert result.record is not None
        assert result.record.coercion_type == CoercionType.STRING_TO_NUMBER.value

    def test_integer_string_to_float(self, safe_engine: CoercionEngine):
        """Test coercing integer string to float."""
        result = safe_engine.coerce("42", JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 42.0
        assert isinstance(result.value, float)

    def test_negative_float(self, safe_engine: CoercionEngine):
        """Test coercing negative float string."""
        result = safe_engine.coerce("-3.14", JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == -3.14

    def test_scientific_notation(self, safe_engine: CoercionEngine):
        """Test coercing scientific notation."""
        result = safe_engine.coerce("1e10", JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 1e10

    def test_scientific_notation_negative_exponent(self, safe_engine: CoercionEngine):
        """Test coercing scientific notation with negative exponent."""
        result = safe_engine.coerce("1.5e-3", JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 0.0015

    def test_infinity_string_fails(self, safe_engine: CoercionEngine):
        """Test that infinity string fails."""
        result = safe_engine.coerce("inf", JSON_TYPE_NUMBER, "/value")

        assert result.success is False

    def test_negative_infinity_fails(self, safe_engine: CoercionEngine):
        """Test that negative infinity string fails."""
        result = safe_engine.coerce("-inf", JSON_TYPE_NUMBER, "/value")

        assert result.success is False

    def test_nan_fails(self, safe_engine: CoercionEngine):
        """Test that NaN string fails."""
        result = safe_engine.coerce("nan", JSON_TYPE_NUMBER, "/value")

        assert result.success is False

    def test_empty_string_fails(self, safe_engine: CoercionEngine):
        """Test that empty string fails."""
        result = safe_engine.coerce("", JSON_TYPE_NUMBER, "/value")

        assert result.success is False

    def test_non_numeric_fails(self, safe_engine: CoercionEngine):
        """Test that non-numeric string fails."""
        result = safe_engine.coerce("hello", JSON_TYPE_NUMBER, "/value")

        assert result.success is False

    def test_zero_decimal(self, safe_engine: CoercionEngine):
        """Test coercing zero with decimal."""
        result = safe_engine.coerce("0.0", JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 0.0


# =============================================================================
# TEST: STRING TO BOOLEAN COERCION
# =============================================================================


class TestStringToBooleanCoercion:
    """Tests for string to boolean coercion."""

    def test_true_lowercase(self, safe_engine: CoercionEngine):
        """Test coercing 'true' to True."""
        result = safe_engine.coerce("true", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True
        assert result.value is True
        assert result.record is not None
        assert result.record.coercion_type == CoercionType.STRING_TO_BOOLEAN.value

    def test_false_lowercase(self, safe_engine: CoercionEngine):
        """Test coercing 'false' to False."""
        result = safe_engine.coerce("false", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True
        assert result.value is False

    def test_true_uppercase(self, safe_engine: CoercionEngine):
        """Test coercing 'TRUE' to True (case insensitive)."""
        result = safe_engine.coerce("TRUE", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True
        assert result.value is True

    def test_false_uppercase(self, safe_engine: CoercionEngine):
        """Test coercing 'FALSE' to False (case insensitive)."""
        result = safe_engine.coerce("FALSE", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True
        assert result.value is False

    def test_true_mixed_case(self, safe_engine: CoercionEngine):
        """Test coercing 'True' to True."""
        result = safe_engine.coerce("True", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True
        assert result.value is True

    def test_invalid_boolean_string(self, safe_engine: CoercionEngine):
        """Test that invalid boolean strings fail."""
        result = safe_engine.coerce("yes", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is False

    def test_one_string_fails_safe(self, safe_engine: CoercionEngine):
        """Test that '1' fails in safe mode."""
        result = safe_engine.coerce("1", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is False

    def test_zero_string_fails_safe(self, safe_engine: CoercionEngine):
        """Test that '0' fails in safe mode."""
        result = safe_engine.coerce("0", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is False

    def test_empty_string_fails(self, safe_engine: CoercionEngine):
        """Test that empty string fails."""
        result = safe_engine.coerce("", JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is False


# =============================================================================
# TEST: STRING TO NULL COERCION
# =============================================================================


class TestStringToNullCoercion:
    """Tests for string to null coercion."""

    def test_null_lowercase(self, safe_engine: CoercionEngine):
        """Test coercing 'null' to None."""
        result = safe_engine.coerce("null", JSON_TYPE_NULL, "/value")

        assert result.success is True
        assert result.value is None
        assert result.record is not None
        assert result.record.coercion_type == CoercionType.STRING_TO_NULL.value

    def test_null_uppercase(self, safe_engine: CoercionEngine):
        """Test coercing 'NULL' to None (case insensitive)."""
        result = safe_engine.coerce("NULL", JSON_TYPE_NULL, "/value")

        assert result.success is True
        assert result.value is None

    def test_null_mixed_case(self, safe_engine: CoercionEngine):
        """Test coercing 'Null' to None."""
        result = safe_engine.coerce("Null", JSON_TYPE_NULL, "/value")

        assert result.success is True
        assert result.value is None

    def test_empty_string_fails_safe(self, safe_engine: CoercionEngine):
        """Test that empty string fails in safe mode."""
        result = safe_engine.coerce("", JSON_TYPE_NULL, "/value")

        assert result.success is False

    def test_none_string_fails(self, safe_engine: CoercionEngine):
        """Test that 'none' fails (only 'null' accepted)."""
        result = safe_engine.coerce("none", JSON_TYPE_NULL, "/value")

        assert result.success is False

    def test_empty_string_succeeds_aggressive(self, aggressive_engine: CoercionEngine):
        """Test that empty string succeeds in aggressive mode."""
        result = aggressive_engine.coerce("", JSON_TYPE_NULL, "/value")

        assert result.success is True
        assert result.value is None
        assert result.record is not None
        assert result.record.reversible is False  # Not reversible


# =============================================================================
# TEST: INTEGER TO NUMBER COERCION
# =============================================================================


class TestIntegerToNumberCoercion:
    """Tests for integer to number (float) coercion."""

    def test_integer_accepted_as_number(self, safe_engine: CoercionEngine):
        """Test that integer is accepted as number without coercion.

        In JSON Schema, integer is a subtype of number, so an integer
        value is already valid for a number type field.
        """
        result = safe_engine.coerce(42, JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 42
        # Integer is accepted as-is (no coercion needed)
        assert result.record is None

    def test_zero_integer_accepted(self, safe_engine: CoercionEngine):
        """Test that zero integer is accepted as number."""
        result = safe_engine.coerce(0, JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 0

    def test_negative_integer_accepted(self, safe_engine: CoercionEngine):
        """Test that negative integer is accepted as number."""
        result = safe_engine.coerce(-123, JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == -123

    def test_explicit_integer_to_float_via_method(self, safe_engine: CoercionEngine):
        """Test explicit integer to float coercion via direct method."""
        # Call coerce_to_number directly to force coercion
        result = safe_engine.coerce_to_number(42, "/value")

        # When calling method directly, coercion should occur
        assert result.success is True
        # Note: This depends on whether _coerce_integer_to_number is called
        # The main coerce() method treats integer as valid for number


# =============================================================================
# TEST: FLOAT TO INTEGER COERCION
# =============================================================================


class TestFloatToIntegerCoercion:
    """Tests for float to integer coercion."""

    def test_whole_number_float(self, safe_engine: CoercionEngine):
        """Test coercing whole number float to integer."""
        result = safe_engine.coerce(42.0, JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == 42
        assert isinstance(result.value, int)
        assert result.record is not None
        assert result.record.coercion_type == CoercionType.FLOAT_TO_INTEGER.value

    def test_zero_float(self, safe_engine: CoercionEngine):
        """Test coercing 0.0 to integer."""
        result = safe_engine.coerce(0.0, JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == 0

    def test_negative_whole_float(self, safe_engine: CoercionEngine):
        """Test coercing negative whole number float."""
        result = safe_engine.coerce(-5.0, JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == -5

    def test_decimal_float_fails(self, safe_engine: CoercionEngine):
        """Test that float with decimal part fails."""
        result = safe_engine.coerce(3.14, JSON_TYPE_INTEGER, "/value")

        assert result.success is False
        assert "decimal part" in result.error

    def test_tiny_decimal_fails(self, safe_engine: CoercionEngine):
        """Test that even tiny decimal fails."""
        result = safe_engine.coerce(42.00001, JSON_TYPE_INTEGER, "/value")

        assert result.success is False


# =============================================================================
# TEST: INTEGER TO BOOLEAN COERCION (AGGRESSIVE MODE)
# =============================================================================


class TestIntegerToBooleanCoercion:
    """Tests for integer to boolean coercion."""

    def test_one_to_true_aggressive(self, aggressive_engine: CoercionEngine):
        """Test coercing 1 to True in aggressive mode."""
        result = aggressive_engine.coerce(1, JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True
        assert result.value is True
        assert result.record is not None
        assert result.record.coercion_type == CoercionType.INTEGER_TO_BOOLEAN.value

    def test_zero_to_false_aggressive(self, aggressive_engine: CoercionEngine):
        """Test coercing 0 to False in aggressive mode."""
        result = aggressive_engine.coerce(0, JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True
        assert result.value is False

    def test_one_fails_safe(self, safe_engine: CoercionEngine):
        """Test that 1 fails in safe mode."""
        result = safe_engine.coerce(1, JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is False
        assert "aggressive policy" in result.error

    def test_zero_fails_safe(self, safe_engine: CoercionEngine):
        """Test that 0 fails in safe mode."""
        result = safe_engine.coerce(0, JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is False

    def test_two_fails_aggressive(self, aggressive_engine: CoercionEngine):
        """Test that 2 fails even in aggressive mode."""
        result = aggressive_engine.coerce(2, JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is False
        assert "only 0 or 1" in result.error


# =============================================================================
# TEST: COERCION TO STRING
# =============================================================================


class TestCoercionToString:
    """Tests for coercing various types to string."""

    def test_integer_to_string(self, safe_engine: CoercionEngine):
        """Test coercing integer to string."""
        result = safe_engine.coerce(42, JSON_TYPE_STRING, "/value")

        assert result.success is True
        assert result.value == "42"
        assert result.record is not None

    def test_float_to_string(self, safe_engine: CoercionEngine):
        """Test coercing float to string."""
        result = safe_engine.coerce(3.14, JSON_TYPE_STRING, "/value")

        assert result.success is True
        assert result.value == "3.14"

    def test_whole_float_to_string(self, safe_engine: CoercionEngine):
        """Test coercing whole number float to string."""
        result = safe_engine.coerce(42.0, JSON_TYPE_STRING, "/value")

        assert result.success is True
        assert result.value == "42"  # Integer string, not "42.0"

    def test_boolean_true_to_string(self, safe_engine: CoercionEngine):
        """Test coercing True to string."""
        result = safe_engine.coerce(True, JSON_TYPE_STRING, "/flag")

        assert result.success is True
        assert result.value == "true"

    def test_boolean_false_to_string(self, safe_engine: CoercionEngine):
        """Test coercing False to string."""
        result = safe_engine.coerce(False, JSON_TYPE_STRING, "/flag")

        assert result.success is True
        assert result.value == "false"

    def test_null_to_string(self, safe_engine: CoercionEngine):
        """Test coercing None to string."""
        result = safe_engine.coerce(None, JSON_TYPE_STRING, "/value")

        assert result.success is True
        assert result.value == "null"


# =============================================================================
# TEST: POLICY ENFORCEMENT
# =============================================================================


class TestPolicyEnforcement:
    """Tests for coercion policy enforcement."""

    def test_off_policy_rejects_coercion(self, off_engine: CoercionEngine):
        """Test that OFF policy rejects coercion."""
        result = off_engine.coerce("42", JSON_TYPE_INTEGER, "/value")

        assert result.success is False
        assert "coercion disabled" in result.error

    def test_off_policy_accepts_matching_type(self, off_engine: CoercionEngine):
        """Test that OFF policy accepts matching types."""
        result = off_engine.coerce(42, JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == 42
        assert result.record is None  # No coercion needed

    def test_off_policy_accepts_integer_for_number(self, off_engine: CoercionEngine):
        """Test that OFF policy accepts integer for number type."""
        result = off_engine.coerce(42, JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 42  # Integer is valid as number

    def test_safe_policy_rejects_aggressive_coercions(self, safe_engine: CoercionEngine):
        """Test that SAFE policy rejects aggressive coercions."""
        # Integer to boolean is aggressive only
        result = safe_engine.coerce(1, JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is False

    def test_aggressive_policy_allows_aggressive_coercions(self, aggressive_engine: CoercionEngine):
        """Test that AGGRESSIVE policy allows aggressive coercions."""
        result = aggressive_engine.coerce(1, JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True


# =============================================================================
# TEST: NO COERCION NEEDED
# =============================================================================


class TestNoCoercionNeeded:
    """Tests for cases where no coercion is needed."""

    def test_integer_already_integer(self, safe_engine: CoercionEngine):
        """Test that integer doesn't need coercion to integer."""
        result = safe_engine.coerce(42, JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == 42
        assert result.record is None  # No coercion record

    def test_string_already_string(self, safe_engine: CoercionEngine):
        """Test that string doesn't need coercion to string."""
        result = safe_engine.coerce("hello", JSON_TYPE_STRING, "/value")

        assert result.success is True
        assert result.value == "hello"
        assert result.record is None

    def test_boolean_already_boolean(self, safe_engine: CoercionEngine):
        """Test that boolean doesn't need coercion to boolean."""
        result = safe_engine.coerce(True, JSON_TYPE_BOOLEAN, "/flag")

        assert result.success is True
        assert result.value is True
        assert result.record is None

    def test_null_already_null(self, safe_engine: CoercionEngine):
        """Test that null doesn't need coercion to null."""
        result = safe_engine.coerce(None, JSON_TYPE_NULL, "/value")

        assert result.success is True
        assert result.value is None
        assert result.record is None

    def test_float_already_number(self, safe_engine: CoercionEngine):
        """Test that float doesn't need coercion to number."""
        result = safe_engine.coerce(3.14, JSON_TYPE_NUMBER, "/value")

        assert result.success is True
        assert result.value == 3.14
        assert result.record is None

    def test_integer_valid_as_number(self, safe_engine: CoercionEngine):
        """Test that integer is valid as number without coercion."""
        result = safe_engine.coerce(42, JSON_TYPE_NUMBER, "/value")

        # Integer should be coerced to float for number type
        # But first check if types match - integer IS a number
        # So this depends on implementation
        assert result.success is True


# =============================================================================
# TEST: COERCION RECORD TRACKING
# =============================================================================


class TestCoercionRecordTracking:
    """Tests for coercion record tracking."""

    def test_records_stored_in_engine(self, safe_engine: CoercionEngine):
        """Test that records are stored in the engine."""
        safe_engine.coerce("42", JSON_TYPE_INTEGER, "/a")
        safe_engine.coerce("true", JSON_TYPE_BOOLEAN, "/b")

        records = safe_engine.get_records()

        assert len(records) == 2
        assert records[0].path == "/a"
        assert records[1].path == "/b"

    def test_clear_records(self, safe_engine: CoercionEngine):
        """Test clearing coercion records."""
        safe_engine.coerce("42", JSON_TYPE_INTEGER, "/a")
        safe_engine.coerce("true", JSON_TYPE_BOOLEAN, "/b")

        assert len(safe_engine.get_records()) == 2

        safe_engine.clear_records()

        assert len(safe_engine.get_records()) == 0

    def test_record_contains_complete_info(self, safe_engine: CoercionEngine):
        """Test that record contains all required information."""
        result = safe_engine.coerce("42", JSON_TYPE_INTEGER, "/value")

        record = result.record

        assert record.path == "/value"
        assert record.original_value == "42"
        assert record.original_type == JSON_TYPE_STRING
        assert record.coerced_value == 42
        assert record.coerced_type == JSON_TYPE_INTEGER
        assert record.reversible is True
        assert record.coercion_type == CoercionType.STRING_TO_INTEGER.value

    def test_failed_coercion_no_record(self, safe_engine: CoercionEngine):
        """Test that failed coercion doesn't create record."""
        result = safe_engine.coerce("hello", JSON_TYPE_INTEGER, "/value")

        assert result.success is False
        assert result.record is None
        assert len(safe_engine.get_records()) == 0


# =============================================================================
# TEST: can_coerce UTILITY
# =============================================================================


class TestCanCoerce:
    """Tests for can_coerce utility function."""

    def test_can_coerce_string_to_integer(self):
        """Test can_coerce for string to integer."""
        assert can_coerce("42", JSON_TYPE_INTEGER, CoercionPolicy.SAFE) is True
        assert can_coerce("hello", JSON_TYPE_INTEGER, CoercionPolicy.SAFE) is False

    def test_can_coerce_with_off_policy(self):
        """Test can_coerce with OFF policy."""
        assert can_coerce("42", JSON_TYPE_INTEGER, CoercionPolicy.OFF) is False
        assert can_coerce(42, JSON_TYPE_INTEGER, CoercionPolicy.OFF) is True

    def test_can_coerce_integer_to_boolean(self):
        """Test can_coerce for integer to boolean."""
        assert can_coerce(1, JSON_TYPE_BOOLEAN, CoercionPolicy.SAFE) is False
        assert can_coerce(1, JSON_TYPE_BOOLEAN, CoercionPolicy.AGGRESSIVE) is True


# =============================================================================
# TEST: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unsupported_target_type(self, safe_engine: CoercionEngine):
        """Test coercion to unsupported type."""
        result = safe_engine.coerce("hello", JSON_TYPE_OBJECT, "/value")

        assert result.success is False
        assert "not a primitive type" in result.error

    def test_array_target_type_fails(self, safe_engine: CoercionEngine):
        """Test coercion to array type fails."""
        result = safe_engine.coerce("hello", JSON_TYPE_ARRAY, "/value")

        assert result.success is False

    def test_coerce_list_fails(self, safe_engine: CoercionEngine):
        """Test that list cannot be coerced to string."""
        result = safe_engine.coerce([1, 2, 3], JSON_TYPE_STRING, "/value")

        assert result.success is False

    def test_coerce_dict_fails(self, safe_engine: CoercionEngine):
        """Test that dict cannot be coerced to string."""
        result = safe_engine.coerce({"key": "value"}, JSON_TYPE_STRING, "/value")

        assert result.success is False

    def test_very_large_number_string(self, safe_engine: CoercionEngine):
        """Test coercing very large number string."""
        result = safe_engine.coerce("1" + "0" * 100, JSON_TYPE_INTEGER, "/value")

        assert result.success is True
        assert result.value == int("1" + "0" * 100)

    def test_path_preserved_in_record(self, safe_engine: CoercionEngine):
        """Test that path is preserved in coercion record."""
        result = safe_engine.coerce("42", JSON_TYPE_INTEGER, "/nested/deep/path")

        assert result.record.path == "/nested/deep/path"


# =============================================================================
# TEST: COERCION RESULT FACTORY METHODS
# =============================================================================


class TestCoercionResultFactoryMethods:
    """Tests for CoercionResult factory methods."""

    def test_success_result(self):
        """Test success_result factory method."""
        result = CoercionResult.success_result(42, None)

        assert result.success is True
        assert result.value == 42
        assert result.record is None
        assert result.error is None

    def test_failure_result(self):
        """Test failure_result factory method."""
        result = CoercionResult.failure_result("original", "Error message")

        assert result.success is False
        assert result.value == "original"
        assert result.record is None
        assert result.error == "Error message"

    def test_no_coercion_needed(self):
        """Test no_coercion_needed factory method."""
        result = CoercionResult.no_coercion_needed(42)

        assert result.success is True
        assert result.value == 42
        assert result.record is None
        assert result.error is None


# =============================================================================
# TEST: REVERSIBILITY
# =============================================================================


class TestReversibility:
    """Tests for coercion reversibility."""

    def test_string_to_integer_reversible(self, safe_engine: CoercionEngine):
        """Test that string to integer is reversible."""
        result = safe_engine.coerce("42", JSON_TYPE_INTEGER, "/value")

        assert result.record.reversible is True
        # Verify: can convert back to original
        assert str(result.value) == "42"

    def test_string_to_boolean_reversible(self, safe_engine: CoercionEngine):
        """Test that string to boolean is reversible."""
        result = safe_engine.coerce("true", JSON_TYPE_BOOLEAN, "/flag")

        assert result.record.reversible is True

    def test_empty_string_to_null_not_reversible(self, aggressive_engine: CoercionEngine):
        """Test that empty string to null is NOT reversible."""
        result = aggressive_engine.coerce("", JSON_TYPE_NULL, "/value")

        assert result.record.reversible is False
        # Cannot distinguish between "" and "null" when reversing


# =============================================================================
# TEST: TYPE-SPECIFIC COERCION METHODS
# =============================================================================


class TestTypeSpecificMethods:
    """Tests for type-specific coercion methods."""

    def test_coerce_to_integer_method(self, safe_engine: CoercionEngine):
        """Test coerce_to_integer method directly."""
        result = safe_engine.coerce_to_integer("42", "/value")

        assert result.success is True
        assert result.value == 42

    def test_coerce_to_number_method(self, safe_engine: CoercionEngine):
        """Test coerce_to_number method directly."""
        result = safe_engine.coerce_to_number("3.14", "/value")

        assert result.success is True
        assert result.value == 3.14

    def test_coerce_to_boolean_method(self, safe_engine: CoercionEngine):
        """Test coerce_to_boolean method directly."""
        result = safe_engine.coerce_to_boolean("true", "/flag")

        assert result.success is True
        assert result.value is True

    def test_coerce_to_string_method(self, safe_engine: CoercionEngine):
        """Test coerce_to_string method directly."""
        result = safe_engine.coerce_to_string(42, "/value")

        assert result.success is True
        assert result.value == "42"

    def test_coerce_to_null_method(self, safe_engine: CoercionEngine):
        """Test coerce_to_null method directly."""
        result = safe_engine.coerce_to_null("null", "/value")

        assert result.success is True
        assert result.value is None


# =============================================================================
# TEST: COERCION RECORD MODEL
# =============================================================================


class TestCoercionRecordModel:
    """Tests for CoercionRecord model."""

    def test_to_dict(self):
        """Test CoercionRecord to_dict method."""
        record = CoercionRecord(
            path="/value",
            original_value="42",
            original_type="string",
            coerced_value=42,
            coerced_type="integer",
            reversible=True,
            coercion_type="string_to_integer"
        )

        d = record.to_dict()

        assert d["path"] == "/value"
        assert d["original_value"] == "42"
        assert d["original_type"] == "string"
        assert d["coerced_value"] == 42
        assert d["coerced_type"] == "integer"
        assert d["reversible"] is True
        assert d["coercion_type"] == "string_to_integer"

    def test_record_immutable(self):
        """Test that CoercionRecord is immutable."""
        record = CoercionRecord(
            path="/value",
            original_value="42",
            original_type="string",
            coerced_value=42,
            coerced_type="integer",
            reversible=True,
            coercion_type="string_to_integer"
        )

        # Pydantic frozen models raise validation error on assignment
        with pytest.raises(Exception):  # ValidationError or AttributeError
            record.path = "/new_path"
