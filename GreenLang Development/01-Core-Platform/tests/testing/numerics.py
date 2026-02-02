"""
Numeric Testing Utilities
=========================

Utilities for testing numeric calculations and validations.

Author: Testing Team
Created: 2025-11-21
"""

from typing import Union, Optional
import math
import logging

logger = logging.getLogger(__name__)


def assert_numeric_equal(
    actual: Union[int, float],
    expected: Union[int, float],
    tolerance: float = 1e-6,
    message: Optional[str] = None
) -> bool:
    """
    Assert two numbers are equal within tolerance.

    Args:
        actual: Actual value
        expected: Expected value
        tolerance: Absolute tolerance for comparison
        message: Optional custom message

    Returns:
        True if equal within tolerance

    Raises:
        AssertionError: If values differ by more than tolerance
    """
    diff = abs(actual - expected)
    if diff > tolerance:
        msg = message or f"Expected {expected}, got {actual} (diff: {diff}, tolerance: {tolerance})"
        raise AssertionError(msg)
    return True


def assert_numeric_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    message: Optional[str] = None
) -> bool:
    """
    Assert value is within range.

    Args:
        value: Value to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        message: Optional custom message

    Returns:
        True if in range

    Raises:
        AssertionError: If value is outside range
    """
    if min_val is not None and value < min_val:
        msg = message or f"Value {value} is below minimum {min_val}"
        raise AssertionError(msg)

    if max_val is not None and value > max_val:
        msg = message or f"Value {value} exceeds maximum {max_val}"
        raise AssertionError(msg)

    return True


def assert_positive(
    value: Union[int, float],
    message: Optional[str] = None
) -> bool:
    """
    Assert value is positive (> 0).

    Args:
        value: Value to check
        message: Optional custom message

    Returns:
        True if positive

    Raises:
        AssertionError: If value is not positive
    """
    if value <= 0:
        msg = message or f"Value {value} is not positive"
        raise AssertionError(msg)
    return True


def assert_non_negative(
    value: Union[int, float],
    message: Optional[str] = None
) -> bool:
    """
    Assert value is non-negative (>= 0).

    Args:
        value: Value to check
        message: Optional custom message

    Returns:
        True if non-negative

    Raises:
        AssertionError: If value is negative
    """
    if value < 0:
        msg = message or f"Value {value} is negative"
        raise AssertionError(msg)
    return True


def calculate_relative_error(
    actual: Union[int, float],
    expected: Union[int, float]
) -> float:
    """
    Calculate relative error between actual and expected.

    Args:
        actual: Actual value
        expected: Expected value

    Returns:
        Relative error as percentage
    """
    if expected == 0:
        return float('inf') if actual != 0 else 0

    return abs((actual - expected) / expected) * 100


class NumericValidator:
    """
    Validator for numeric calculations and results.

    Provides comprehensive validation for numeric outputs
    in carbon accounting and sustainability calculations.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize numeric validator.

        Args:
            tolerance: Default tolerance for comparisons
        """
        self.tolerance = tolerance
        self.validation_count = 0
        self.failure_count = 0

    def validate_emission_factor(self, factor: float) -> bool:
        """
        Validate emission factor value.

        Args:
            factor: Emission factor to validate

        Returns:
            True if valid

        Raises:
            AssertionError: If invalid
        """
        self.validation_count += 1

        try:
            # Emission factors should be non-negative
            assert_non_negative(factor, "Emission factor cannot be negative")

            # Check reasonable range (0 to 1000 kg CO2e per unit)
            assert_numeric_range(
                factor, 0, 1000,
                "Emission factor outside reasonable range (0-1000)"
            )

            return True
        except AssertionError:
            self.failure_count += 1
            raise

    def validate_carbon_footprint(self, footprint: float) -> bool:
        """
        Validate carbon footprint calculation.

        Args:
            footprint: Carbon footprint value

        Returns:
            True if valid

        Raises:
            AssertionError: If invalid
        """
        self.validation_count += 1

        try:
            # Carbon footprint should be non-negative
            assert_non_negative(footprint, "Carbon footprint cannot be negative")

            # Check for NaN or infinity
            if math.isnan(footprint):
                raise AssertionError("Carbon footprint is NaN")
            if math.isinf(footprint):
                raise AssertionError("Carbon footprint is infinite")

            return True
        except AssertionError:
            self.failure_count += 1
            raise

    def validate_percentage(self, value: float, name: str = "Percentage") -> bool:
        """
        Validate percentage value (0-100).

        Args:
            value: Percentage value
            name: Name for error messages

        Returns:
            True if valid

        Raises:
            AssertionError: If invalid
        """
        self.validation_count += 1

        try:
            assert_numeric_range(
                value, 0, 100,
                f"{name} must be between 0 and 100"
            )
            return True
        except AssertionError:
            self.failure_count += 1
            raise

    def validate_calculation(
        self,
        result: float,
        expected: float,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        Validate calculation result against expected.

        Args:
            result: Calculation result
            expected: Expected value
            tolerance: Tolerance for comparison

        Returns:
            True if valid

        Raises:
            AssertionError: If invalid
        """
        self.validation_count += 1
        tolerance = tolerance or self.tolerance

        try:
            assert_numeric_equal(result, expected, tolerance)
            return True
        except AssertionError:
            self.failure_count += 1
            raise

    def get_stats(self) -> dict:
        """Get validation statistics."""
        return {
            "total_validations": self.validation_count,
            "failures": self.failure_count,
            "success_rate": (
                (self.validation_count - self.failure_count) / self.validation_count
                if self.validation_count > 0 else 0
            )
        }
