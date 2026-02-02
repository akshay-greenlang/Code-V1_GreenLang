"""
Tests for GreenLang Financial Decimal Operations

Tests decimal precision, rounding, and financial calculations.
"""

import pytest
from decimal import Decimal

from greenlang.determinism.decimal import (
    FinancialDecimal,
    safe_decimal,
    safe_decimal_multiply,
    safe_decimal_divide,
    safe_decimal_add,
    safe_decimal_sum,
    round_for_reporting,
)


class TestFinancialDecimal:
    """Test FinancialDecimal class."""

    def test_from_float(self):
        """Test conversion from float."""
        result = FinancialDecimal.from_float(3.14159)
        assert isinstance(result, Decimal)
        assert str(result) == "3.14159000"

    def test_from_string(self):
        """Test conversion from string."""
        result = FinancialDecimal.from_string("123.45")
        assert isinstance(result, Decimal)
        assert str(result) == "123.45000000"

    def test_from_string_with_formatting(self):
        """Test parsing string with common formatting."""
        result = FinancialDecimal.from_string("$1,234.56")
        assert str(result) == "1234.56000000"

    def test_from_any_int(self):
        """Test conversion from int."""
        result = FinancialDecimal.from_any(100)
        assert str(result) == "100.00000000"

    def test_from_any_float(self):
        """Test conversion from float."""
        result = FinancialDecimal.from_any(3.14)
        assert str(result) == "3.14000000"

    def test_from_any_string(self):
        """Test conversion from string."""
        result = FinancialDecimal.from_any("2.718")
        assert str(result) == "2.71800000"

    def test_from_any_decimal(self):
        """Test conversion from Decimal."""
        result = FinancialDecimal.from_any(Decimal("1.5"))
        assert str(result) == "1.50000000"

    def test_from_any_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError):
            FinancialDecimal.from_any([1, 2, 3])

    def test_multiply(self):
        """Test multiplication with proper precision."""
        result = FinancialDecimal.multiply(100.5, 2)
        assert str(result) == "201.00000000"

    def test_divide(self):
        """Test division with proper precision."""
        result = FinancialDecimal.divide(100, 3)
        assert str(result) == "33.33333333"

    def test_divide_by_zero(self):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Division by zero"):
            FinancialDecimal.divide(100, 0)

    def test_add(self):
        """Test addition with proper precision."""
        result = FinancialDecimal.add(0.1, 0.2)
        assert str(result) == "0.30000000"

    def test_subtract(self):
        """Test subtraction with proper precision."""
        result = FinancialDecimal.subtract(100, 33.33)
        assert str(result) == "66.67000000"

    def test_sum(self):
        """Test sum of multiple values."""
        result = FinancialDecimal.sum([0.1, 0.2, 0.3, 0.4])
        assert str(result) == "1.00000000"

    def test_round_to_precision(self):
        """Test rounding to specific decimal places."""
        result = FinancialDecimal.round_to_precision(3.14159265, 3)
        assert str(result) == "3.142"

    def test_round_to_precision_zero_places(self):
        """Test rounding to zero decimal places."""
        result = FinancialDecimal.round_to_precision(3.7, 0)
        assert str(result) == "4"

    def test_round_to_precision_invalid_places(self):
        """Test that invalid decimal_places raises ValueError."""
        with pytest.raises(ValueError):
            FinancialDecimal.round_to_precision(3.14, -1)
        with pytest.raises(ValueError):
            FinancialDecimal.round_to_precision(3.14, 9)

    def test_is_positive(self):
        """Test is_positive check."""
        assert FinancialDecimal.is_positive(1) is True
        assert FinancialDecimal.is_positive(0) is False
        assert FinancialDecimal.is_positive(-1) is False

    def test_is_non_negative(self):
        """Test is_non_negative check."""
        assert FinancialDecimal.is_non_negative(1) is True
        assert FinancialDecimal.is_non_negative(0) is True
        assert FinancialDecimal.is_non_negative(-1) is False

    def test_is_zero(self):
        """Test is_zero check."""
        assert FinancialDecimal.is_zero(0) is True
        assert FinancialDecimal.is_zero(0.0) is True
        assert FinancialDecimal.is_zero(0.00001) is False

    def test_is_zero_with_tolerance(self):
        """Test is_zero with tolerance."""
        tolerance = Decimal("0.001")
        assert FinancialDecimal.is_zero(0.0005, tolerance) is True
        assert FinancialDecimal.is_zero(0.002, tolerance) is False


class TestSafeDecimalHelpers:
    """Test safe_decimal helper functions."""

    def test_safe_decimal(self):
        """Test safe_decimal conversion."""
        result = safe_decimal(100)
        assert str(result) == "100.00000000"

    def test_safe_decimal_multiply(self):
        """Test safe_decimal_multiply."""
        result = safe_decimal_multiply(100.5, 2)
        assert str(result) == "201.00000000"

    def test_safe_decimal_divide(self):
        """Test safe_decimal_divide."""
        result = safe_decimal_divide(100, 3)
        assert str(result) == "33.33333333"

    def test_safe_decimal_add(self):
        """Test safe_decimal_add fixes float precision."""
        result = safe_decimal_add(0.1, 0.2)
        assert str(result) == "0.30000000"

    def test_safe_decimal_sum(self):
        """Test safe_decimal_sum."""
        result = safe_decimal_sum([0.1, 0.2, 0.3, 0.4])
        assert str(result) == "1.00000000"

    def test_round_for_reporting(self):
        """Test round_for_reporting with default 3 decimal places."""
        result = round_for_reporting(123.456789)
        assert str(result) == "123.457"

    def test_round_for_reporting_custom_places(self):
        """Test round_for_reporting with custom decimal places."""
        result = round_for_reporting(123.456789, decimal_places=2)
        assert str(result) == "123.46"


class TestDeterminism:
    """Test deterministic behavior across operations."""

    def test_float_precision_issue_fixed(self):
        """Test that float 0.1 + 0.2 issue is fixed."""
        result = safe_decimal_add(0.1, 0.2)
        # In regular Python: 0.1 + 0.2 = 0.30000000000000004
        # With safe_decimal: 0.1 + 0.2 = 0.3
        assert result == Decimal("0.3")

    def test_multiply_deterministic(self):
        """Test multiplication is deterministic."""
        results = [safe_decimal_multiply(100.5, 2) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_divide_deterministic(self):
        """Test division is deterministic."""
        results = [safe_decimal_divide(100, 3) for _ in range(10)]
        assert all(r == results[0] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
