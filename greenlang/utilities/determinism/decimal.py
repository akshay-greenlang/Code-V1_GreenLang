"""
GreenLang Deterministic Decimal - Financial Precision Calculations

This module provides decimal arithmetic with financial precision for compliance.

Features:
- 8 decimal place precision
- ROUND_HALF_UP rounding per regulatory standards
- Safe type conversion from any numeric type
- Zero-hallucination guarantees

Author: GreenLang Team
Date: 2025-11-21
"""

from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Any, Union, List


# Set decimal precision for financial calculations (8 decimal places)
getcontext().prec = 28  # 28 significant digits
getcontext().rounding = ROUND_HALF_UP


class FinancialDecimal:
    """
    Wrapper for Decimal operations with financial precision.

    Ensures all financial calculations use proper decimal arithmetic
    with consistent rounding rules.

    ZERO-HALLUCINATION GUARANTEE:
    - All operations are deterministic (same input -> same output)
    - Uses Decimal throughout to avoid float precision issues
    - Consistent rounding (ROUND_HALF_UP) per regulatory standards
    - Safe type conversion from any numeric type
    """

    PRECISION = Decimal('0.00000001')  # 8 decimal places
    ROUNDING = ROUND_HALF_UP

    @classmethod
    def from_float(cls, value: float) -> Decimal:
        """
        Convert float to Decimal safely.

        IMPORTANT: Converts through string to avoid float precision issues.
        This ensures 0.1 + 0.2 == 0.3 (unlike raw float arithmetic).

        Args:
            value: Float value

        Returns:
            Decimal with proper precision

        Example:
            >>> FinancialDecimal.from_float(0.1)
            Decimal('0.10000000')
        """
        # Convert through string to avoid float precision issues
        return Decimal(str(value)).quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def from_string(cls, value: str) -> Decimal:
        """
        Parse string to Decimal.

        Args:
            value: String representation

        Returns:
            Decimal with proper precision

        Example:
            >>> FinancialDecimal.from_string("1,234.56")
            Decimal('1234.56000000')
        """
        # Remove common formatting
        value = value.replace(',', '').replace('$', '').strip()
        return Decimal(value).quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def from_any(cls, value: Any) -> Decimal:
        """
        Convert any numeric type to Decimal safely.

        This is the primary method for ensuring consistent Decimal usage
        throughout the calculation engine. Always prefer this method when
        accepting user input or external data.

        Args:
            value: Any numeric value (int, float, str, Decimal)

        Returns:
            Decimal with proper precision

        Raises:
            TypeError: If value cannot be converted to Decimal
            ValueError: If string value is invalid

        Example:
            >>> FinancialDecimal.from_any(100)
            Decimal('100.00000000')
            >>> FinancialDecimal.from_any(3.14159)
            Decimal('3.14159000')
            >>> FinancialDecimal.from_any("2.718")
            Decimal('2.71800000')
            >>> FinancialDecimal.from_any(Decimal("1.5"))
            Decimal('1.50000000')
        """
        if isinstance(value, Decimal):
            return value.quantize(cls.PRECISION, rounding=cls.ROUNDING)
        elif isinstance(value, int):
            return Decimal(value).quantize(cls.PRECISION, rounding=cls.ROUNDING)
        elif isinstance(value, float):
            return cls.from_float(value)
        elif isinstance(value, str):
            return cls.from_string(value)
        else:
            raise TypeError(f"Cannot convert {type(value).__name__} to Decimal")

    @classmethod
    def to_float(cls, value: Decimal) -> float:
        """
        Convert Decimal to float (use sparingly, only for external APIs).

        WARNING: This loses precision. Only use when interfacing with
        external systems that require float (e.g., JSON serialization).

        Args:
            value: Decimal value

        Returns:
            Float representation

        Example:
            >>> FinancialDecimal.to_float(Decimal("3.14159265"))
            3.14159265
        """
        return float(value)

    @classmethod
    def multiply(cls, a: Union[Decimal, float, int, str], b: Union[Decimal, float, int, str]) -> Decimal:
        """
        Multiply two values with proper precision.

        Accepts any numeric type and ensures Decimal precision throughout.

        Args:
            a: First operand (any numeric type)
            b: Second operand (any numeric type)

        Returns:
            Product with proper precision

        Example:
            >>> FinancialDecimal.multiply(100.5, "2.0")
            Decimal('201.00000000')
        """
        a_decimal = cls.from_any(a)
        b_decimal = cls.from_any(b)
        result = a_decimal * b_decimal
        return result.quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def divide(cls, a: Union[Decimal, float, int, str], b: Union[Decimal, float, int, str]) -> Decimal:
        """
        Divide two values with proper precision.

        Accepts any numeric type and ensures Decimal precision throughout.

        Args:
            a: Dividend (any numeric type)
            b: Divisor (any numeric type)

        Returns:
            Quotient with proper precision

        Raises:
            ValueError: If divisor is zero

        Example:
            >>> FinancialDecimal.divide(100, 3)
            Decimal('33.33333333')
        """
        a_decimal = cls.from_any(a)
        b_decimal = cls.from_any(b)
        if b_decimal == 0:
            raise ValueError("Division by zero")
        result = a_decimal / b_decimal
        return result.quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def add(cls, a: Union[Decimal, float, int, str], b: Union[Decimal, float, int, str]) -> Decimal:
        """
        Add two values with proper precision.

        Accepts any numeric type and ensures Decimal precision throughout.

        Args:
            a: First operand (any numeric type)
            b: Second operand (any numeric type)

        Returns:
            Sum with proper precision

        Example:
            >>> FinancialDecimal.add(0.1, 0.2)
            Decimal('0.30000000')  # Correctly 0.3, unlike float 0.1 + 0.2
        """
        a_decimal = cls.from_any(a)
        b_decimal = cls.from_any(b)
        result = a_decimal + b_decimal
        return result.quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def subtract(cls, a: Union[Decimal, float, int, str], b: Union[Decimal, float, int, str]) -> Decimal:
        """
        Subtract two values with proper precision.

        Accepts any numeric type and ensures Decimal precision throughout.

        Args:
            a: Minuend (any numeric type)
            b: Subtrahend (any numeric type)

        Returns:
            Difference with proper precision

        Example:
            >>> FinancialDecimal.subtract(100, 33.33)
            Decimal('66.67000000')
        """
        a_decimal = cls.from_any(a)
        b_decimal = cls.from_any(b)
        result = a_decimal - b_decimal
        return result.quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def sum(cls, values: List[Union[Decimal, float, int, str]]) -> Decimal:
        """
        Sum list of values.

        Accepts any numeric types and ensures Decimal precision throughout.

        Args:
            values: List of numeric values

        Returns:
            Sum with proper precision

        Example:
            >>> FinancialDecimal.sum([0.1, 0.2, 0.3, 0.4])
            Decimal('1.00000000')  # Exactly 1.0, unlike float sum
        """
        total = Decimal('0')
        for v in values:
            total += cls.from_any(v)
        return total.quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def round_to_precision(cls, value: Union[Decimal, float, int, str], decimal_places: int) -> Decimal:
        """
        Round value to specific number of decimal places.

        Useful for regulatory reporting which often requires specific precision
        (e.g., 3 decimal places for emissions reporting).

        Args:
            value: Numeric value to round
            decimal_places: Number of decimal places (0-8)

        Returns:
            Rounded Decimal value

        Example:
            >>> FinancialDecimal.round_to_precision(3.14159265, 3)
            Decimal('3.142')
        """
        if decimal_places < 0 or decimal_places > 8:
            raise ValueError(f"decimal_places must be 0-8, got {decimal_places}")

        value_decimal = cls.from_any(value)
        quantize_str = '0.' + '0' * decimal_places if decimal_places > 0 else '1'
        return value_decimal.quantize(Decimal(quantize_str), rounding=cls.ROUNDING)

    @classmethod
    def is_positive(cls, value: Union[Decimal, float, int, str]) -> bool:
        """Check if value is positive (> 0)."""
        return cls.from_any(value) > 0

    @classmethod
    def is_non_negative(cls, value: Union[Decimal, float, int, str]) -> bool:
        """Check if value is non-negative (>= 0)."""
        return cls.from_any(value) >= 0

    @classmethod
    def is_zero(cls, value: Union[Decimal, float, int, str], tolerance: Decimal = None) -> bool:
        """
        Check if value is zero (or within tolerance of zero).

        Args:
            value: Value to check
            tolerance: Optional tolerance for floating point comparison

        Returns:
            True if value is zero (within tolerance)
        """
        value_decimal = cls.from_any(value)
        if tolerance is None:
            return value_decimal == 0
        return abs(value_decimal) <= tolerance


# ==================== SAFE DECIMAL HELPER FUNCTIONS ====================

def safe_decimal(value: Any) -> Decimal:
    """
    Safely convert any numeric value to Decimal.

    This is the primary function for ensuring consistent Decimal usage
    in all GreenLang calculations. Always use this when accepting
    external numeric input.

    ZERO-HALLUCINATION GUARANTEE:
    - Deterministic conversion (same input -> same output)
    - Avoids float precision issues by converting through string
    - Consistent 8 decimal place precision
    - ROUND_HALF_UP rounding for regulatory compliance

    Args:
        value: Any numeric value (int, float, str, Decimal)

    Returns:
        Decimal with 8 decimal place precision

    Raises:
        TypeError: If value cannot be converted to Decimal

    Example:
        >>> safe_decimal(100)
        Decimal('100.00000000')
        >>> safe_decimal(3.14159)
        Decimal('3.14159000')
        >>> safe_decimal("2.718")
        Decimal('2.71800000')
    """
    return FinancialDecimal.from_any(value)


def safe_decimal_multiply(a: Any, b: Any) -> Decimal:
    """
    Safely multiply two values using Decimal arithmetic.

    Args:
        a: First operand (any numeric type)
        b: Second operand (any numeric type)

    Returns:
        Product as Decimal

    Example:
        >>> safe_decimal_multiply(100.5, 2)
        Decimal('201.00000000')
    """
    return FinancialDecimal.multiply(a, b)


def safe_decimal_divide(a: Any, b: Any) -> Decimal:
    """
    Safely divide two values using Decimal arithmetic.

    Args:
        a: Dividend (any numeric type)
        b: Divisor (any numeric type)

    Returns:
        Quotient as Decimal

    Raises:
        ValueError: If divisor is zero

    Example:
        >>> safe_decimal_divide(100, 3)
        Decimal('33.33333333')
    """
    return FinancialDecimal.divide(a, b)


def safe_decimal_add(a: Any, b: Any) -> Decimal:
    """
    Safely add two values using Decimal arithmetic.

    Args:
        a: First operand (any numeric type)
        b: Second operand (any numeric type)

    Returns:
        Sum as Decimal

    Example:
        >>> safe_decimal_add(0.1, 0.2)
        Decimal('0.30000000')  # Correctly 0.3
    """
    return FinancialDecimal.add(a, b)


def safe_decimal_sum(values: List[Any]) -> Decimal:
    """
    Safely sum a list of values using Decimal arithmetic.

    Args:
        values: List of numeric values

    Returns:
        Sum as Decimal

    Example:
        >>> safe_decimal_sum([0.1, 0.2, 0.3, 0.4])
        Decimal('1.00000000')  # Exactly 1.0
    """
    return FinancialDecimal.sum(values)


def round_for_reporting(value: Any, decimal_places: int = 3) -> Decimal:
    """
    Round value for regulatory reporting.

    Most emission reporting requires 3 decimal places (tonnes CO2e).

    Args:
        value: Numeric value to round
        decimal_places: Number of decimal places (default: 3)

    Returns:
        Rounded Decimal value

    Example:
        >>> round_for_reporting(123.456789)
        Decimal('123.457')
    """
    return FinancialDecimal.round_to_precision(value, decimal_places)


__all__ = [
    'FinancialDecimal',
    'safe_decimal',
    'safe_decimal_multiply',
    'safe_decimal_divide',
    'safe_decimal_add',
    'safe_decimal_sum',
    'round_for_reporting',
]
