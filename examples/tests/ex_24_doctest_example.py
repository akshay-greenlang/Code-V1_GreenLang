"""Example 24: Doctest/xdoctest example linking from docs.

This module demonstrates how to embed testable examples in docstrings
that can be run with doctest or xdoctest.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def calculate_emissions(consumption_kwh, emission_factor):
    """Calculate CO2 emissions from electricity consumption.
    
    This function demonstrates doctest examples that can be
    executed to verify documentation accuracy.
    
    Args:
        consumption_kwh: Electricity consumption in kWh
        emission_factor: Emission factor in kg CO2/kWh
    
    Returns:
        Total emissions in kg CO2
    
    Examples:
        >>> calculate_emissions(1000, 0.5)
        500.0
        
        >>> calculate_emissions(0, 0.5)
        0.0
        
        >>> calculate_emissions(2500, 0.42)
        1050.0
        
        >>> # Edge case: negative values should raise error
        >>> calculate_emissions(-100, 0.5)
        Traceback (most recent call last):
            ...
        ValueError: Consumption cannot be negative
    """
    if consumption_kwh < 0:
        raise ValueError("Consumption cannot be negative")
    return consumption_kwh * emission_factor


def convert_units(value, from_unit, to_unit):
    """Convert between common energy units.
    
    Examples:
        >>> convert_units(1000, "kWh", "MWh")
        1.0
        
        >>> convert_units(1, "MWh", "kWh")
        1000.0
        
        >>> convert_units(100, "therms", "therms")
        100.0
        
        >>> convert_units(0, "kWh", "MWh")
        0.0
    """
    conversions = {
        ("kWh", "MWh"): 0.001,
        ("MWh", "kWh"): 1000.0,
        ("therms", "therms"): 1.0,
    }
    
    factor = conversions.get((from_unit, to_unit))
    if factor is None:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")
    
    return value * factor


class EmissionsCalculator:
    """Calculator for building emissions with embedded examples.
    
    Examples:
        >>> calc = EmissionsCalculator()
        >>> calc.set_emission_factor(0.5)
        >>> calc.calculate(1000)
        500.0
        
        >>> # Chain multiple calculations
        >>> calc = EmissionsCalculator()
        >>> calc.set_emission_factor(0.42)
        >>> results = [calc.calculate(kwh) for kwh in [100, 200, 300]]
        >>> results
        [42.0, 84.0, 126.0]
        
        >>> # Verify total
        >>> sum(results)
        252.0
    """
    
    def __init__(self):
        """Initialize calculator.
        
        Examples:
            >>> calc = EmissionsCalculator()
            >>> calc.emission_factor
            0.0
        """
        self.emission_factor = 0.0
    
    def set_emission_factor(self, factor):
        """Set the emission factor.
        
        Examples:
            >>> calc = EmissionsCalculator()
            >>> calc.set_emission_factor(0.71)
            >>> calc.emission_factor
            0.71
        """
        self.emission_factor = factor
    
    def calculate(self, consumption_kwh):
        """Calculate emissions.
        
        Examples:
            >>> calc = EmissionsCalculator()
            >>> calc.set_emission_factor(0.5)
            >>> calc.calculate(2000)
            1000.0
        """
        return consumption_kwh * self.emission_factor


if __name__ == "__main__":
    import doctest
    import pytest
    
    # Run doctests
    results = doctest.testmod(verbose=True)
    
    # Also create a pytest test that runs doctests
    @pytest.mark.example
    def test_doctests():
        """Run all doctests in this module."""
        failure_count, test_count = doctest.testmod(
            sys.modules[__name__],
            verbose=False
        )
        assert failure_count == 0, f"{failure_count} doctests failed"