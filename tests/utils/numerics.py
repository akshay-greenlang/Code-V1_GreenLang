"""Numeric utilities for testing."""

import math
from typing import Union, Optional


def assert_close(
    actual: Union[float, int],
    expected: Union[float, int],
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
    message: Optional[str] = None
) -> None:
    """
    Assert that two numbers are close within tolerance.
    
    Args:
        actual: The actual value
        expected: The expected value
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
        message: Optional error message
    
    Raises:
        AssertionError: If values are not close within tolerance
    """
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        msg = message or f"Values not close: {actual} != {expected} (rel_tol={rel_tol}, abs_tol={abs_tol})"
        raise AssertionError(msg)


def assert_percentage_sum(
    percentages: list[float],
    expected_sum: float = 100.0,
    tolerance: float = 0.01,
    message: Optional[str] = None
) -> None:
    """
    Assert that percentages sum to expected value within tolerance.
    
    Args:
        percentages: List of percentage values
        expected_sum: Expected sum (default 100.0)
        tolerance: Tolerance for sum
        message: Optional error message
    """
    actual_sum = sum(percentages)
    if abs(actual_sum - expected_sum) > tolerance:
        msg = message or f"Percentages sum to {actual_sum}, expected {expected_sum} ± {tolerance}"
        raise AssertionError(msg)


def round_for_display(value: float, decimals: int = 2) -> float:
    """
    Round a value for display purposes only.
    
    Args:
        value: The value to round
        decimals: Number of decimal places
    
    Returns:
        Rounded value
    """
    return round(value, decimals)


def is_valid_emission(value: Union[float, int]) -> bool:
    """
    Check if a value is a valid emission (non-negative, finite).
    
    Args:
        value: The value to check
    
    Returns:
        True if valid emission value
    """
    if isinstance(value, (int, float)):
        return value >= 0 and math.isfinite(value)
    return False


def normalize_factor(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Normalize emission factors between units.
    
    Args:
        value: The value to normalize
        from_unit: Source unit
        to_unit: Target unit
    
    Returns:
        Normalized value
    """
    conversions = {
        ("kWh", "MWh"): 0.001,
        ("MWh", "kWh"): 1000,
        ("therms", "MMBtu"): 0.1,
        ("MMBtu", "therms"): 10,
        ("m3", "ft3"): 35.3147,
        ("ft3", "m3"): 0.0283168,
        ("sqft", "sqm"): 0.092903,
        ("sqm", "sqft"): 10.7639,
    }
    
    key = (from_unit, to_unit)
    if key in conversions:
        return value * conversions[key]
    elif from_unit == to_unit:
        return value
    else:
        raise ValueError(f"Unknown conversion: {from_unit} to {to_unit}")