# -*- coding: utf-8 -*-
"""
Example 03: Calculator Agent

This example demonstrates mathematical calculations using BaseCalculator.
You'll learn:
- How to perform high-precision calculations
- How to add calculation steps for transparency
- How to use caching for performance
- How to handle division by zero and edge cases
"""

from greenlang.agents import BaseCalculator, CalculatorConfig
from typing import Dict, Any


class CarbonEmissionsCalculator(BaseCalculator):
    """
    Calculate carbon emissions from energy consumption.

    This agent demonstrates:
    - High-precision decimal arithmetic
    - Calculation step tracking
    - Result caching
    - Input validation
    """

    def __init__(self):
        config = CalculatorConfig(
            name="CarbonEmissionsCalculator",
            description="Calculate CO2 emissions from electricity usage",
            precision=4,  # 4 decimal places
            enable_caching=True,
            cache_size=100,
            validate_inputs=True,
            deterministic=True
        )
        super().__init__(config)

    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate emissions from electricity consumption.

        Args:
            inputs: Must contain:
                - electricity_kwh: float (electricity consumption)
                - emission_factor: float (optional, default 0.5 kg CO2/kWh)

        Returns:
            Dictionary with emissions in kg and tons
        """
        # Extract inputs
        electricity_kwh = inputs['electricity_kwh']
        emission_factor = inputs.get('emission_factor', 0.5)  # Default factor

        # Step 1: Calculate emissions in kg
        emissions_kg = electricity_kwh * emission_factor

        self.add_calculation_step(
            step_name="Calculate Emissions (kg)",
            formula="electricity_kwh × emission_factor",
            inputs={
                "electricity_kwh": electricity_kwh,
                "emission_factor": emission_factor
            },
            result=emissions_kg,
            units="kg CO2"
        )

        # Step 2: Convert to tons
        emissions_tons = emissions_kg / 1000

        self.add_calculation_step(
            step_name="Convert to Tons",
            formula="emissions_kg ÷ 1000",
            inputs={"emissions_kg": emissions_kg},
            result=emissions_tons,
            units="tons CO2"
        )

        # Step 3: Calculate monthly savings potential (if reduced by 20%)
        reduction_percentage = 0.20
        potential_reduction = emissions_tons * reduction_percentage

        self.add_calculation_step(
            step_name="Calculate Reduction Potential",
            formula="emissions_tons × reduction_percentage",
            inputs={
                "emissions_tons": emissions_tons,
                "reduction_percentage": reduction_percentage
            },
            result=potential_reduction,
            units="tons CO2"
        )

        return {
            'emissions_kg': emissions_kg,
            'emissions_tons': emissions_tons,
            'reduction_potential_tons': potential_reduction
        }

    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate calculation inputs.

        Args:
            inputs: Inputs to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required field
        if 'electricity_kwh' not in inputs:
            self.logger.error("Missing required input: electricity_kwh")
            return False

        # Check value is numeric and non-negative
        kwh = inputs['electricity_kwh']
        if not isinstance(kwh, (int, float)):
            self.logger.error("electricity_kwh must be numeric")
            return False

        if kwh < 0:
            self.logger.error("electricity_kwh cannot be negative")
            return False

        # Validate emission factor if provided
        if 'emission_factor' in inputs:
            factor = inputs['emission_factor']
            if not isinstance(factor, (int, float)):
                self.logger.error("emission_factor must be numeric")
                return False

            if factor < 0 or factor > 10:
                self.logger.error("emission_factor must be between 0 and 10")
                return False

        return True


def main():
    """Run the example."""
    print("=" * 60)
    print("Example 03: Calculator Agent")
    print("=" * 60)
    print()

    calculator = CarbonEmissionsCalculator()

    # Example 1: Basic calculation
    print("Test 1: Basic Calculation")
    print("-" * 40)

    result = calculator.run({
        "inputs": {
            "electricity_kwh": 1000,
            "emission_factor": 0.45
        }
    })

    if result.success:
        print(f"✓ Calculation successful")
        print(f"  Emissions (kg): {result.result_value['emissions_kg']:.4f}")
        print(f"  Emissions (tons): {result.result_value['emissions_tons']:.4f}")
        print(f"  Reduction potential: {result.result_value['reduction_potential_tons']:.4f} tons")
        print(f"  Cached: {result.cached}")
        print(f"  Execution time: {result.metrics.execution_time_ms:.2f}ms")

        print("\n  Calculation Steps:")
        for i, step in enumerate(result.calculation_steps, 1):
            print(f"    {i}. {step.step_name}")
            print(f"       Formula: {step.formula}")
            print(f"       Result: {step.result} {step.units}")
    else:
        print(f"✗ Calculation failed: {result.error}")
    print()

    # Example 2: Same calculation (should be cached)
    print("Test 2: Repeated Calculation (Cache Hit)")
    print("-" * 40)

    result2 = calculator.run({
        "inputs": {
            "electricity_kwh": 1000,
            "emission_factor": 0.45
        }
    })

    if result2.success:
        print(f"✓ Calculation successful")
        print(f"  Emissions (tons): {result2.result_value['emissions_tons']:.4f}")
        print(f"  Cached: {result2.cached} (should be True)")
        print(f"  Execution time: {result2.metrics.execution_time_ms:.2f}ms (should be faster)")
    print()

    # Example 3: Different input (cache miss)
    print("Test 3: Different Input (Cache Miss)")
    print("-" * 40)

    result3 = calculator.run({
        "inputs": {
            "electricity_kwh": 2000,  # Different value
            "emission_factor": 0.45
        }
    })

    if result3.success:
        print(f"✓ Calculation successful")
        print(f"  Emissions (tons): {result3.result_value['emissions_tons']:.4f}")
        print(f"  Cached: {result3.cached} (should be False)")
    print()

    # Example 4: Invalid input (should fail validation)
    print("Test 4: Invalid Input")
    print("-" * 40)

    result4 = calculator.run({
        "inputs": {
            "electricity_kwh": -100  # Negative value
        }
    })

    if not result4.success:
        print(f"✓ Validation correctly rejected invalid input")
        print(f"  Error: {result4.error}")
    else:
        print(f"✗ Should have failed validation")
    print()

    # Example 5: Check statistics
    print("Calculator Statistics:")
    print("-" * 40)
    stats = calculator.get_stats()
    print(f"  Total executions: {stats['executions']}")
    print(f"  Success rate: {stats['success_rate']}%")
    print(f"  Average time: {stats['avg_time_ms']:.2f}ms")
    print(f"  Cache hits: {stats['custom_counters'].get('cache_hits', 0)}")
    print(f"  Cache misses: {stats['custom_counters'].get('cache_misses', 0)}")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
