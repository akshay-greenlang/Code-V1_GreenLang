#!/usr/bin/env python3
"""
Example 1: Simple Agent - Basic Data Processing
================================================

This example demonstrates how to create a simple agent that processes data
using GreenLang's SDK. The agent calculates building emissions from energy data.

Run: python examples/01_simple_agent.py
"""

import json
from pathlib import Path
from typing import Dict, Any
from greenlang.sdk.base import Agent, Result, Metadata


class EmissionsCalculatorAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Simple agent that calculates emissions from energy consumption.

    Input: Building energy data (electricity, gas)
    Output: Total emissions in metric tons CO2e
    """

    def __init__(self):
        metadata = Metadata(
            id="emissions_calculator",
            name="Emissions Calculator Agent",
            version="1.0.0",
            description="Calculates carbon emissions from energy consumption",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        # Load emission factors from data file
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "emission_factors.json") as f:
            self.factors = json.load(f)["factors"]

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        required = ["electricity_kwh", "gas_therms"]
        has_fields = all(field in input_data for field in required)

        if not has_fields:
            self.logger.error(f"Missing required fields: {required}")
            return False

        # Check for positive values
        if input_data["electricity_kwh"] < 0 or input_data["gas_therms"] < 0:
            self.logger.error("Energy consumption values must be positive")
            return False

        return True

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate emissions from energy consumption"""
        # Get country-specific factors (default to US)
        country = input_data.get("country", "US")

        # Calculate electricity emissions (kgCO2e)
        electricity_kwh = input_data["electricity_kwh"]
        elec_factor = self.factors["electricity"][country]["value"]
        elec_emissions_kg = electricity_kwh * elec_factor

        # Calculate gas emissions (kgCO2e)
        gas_therms = input_data["gas_therms"]
        gas_factor = self.factors["natural_gas"][country]["value"]
        gas_emissions_kg = gas_therms * gas_factor

        # Total emissions in metric tons
        total_emissions_tons = (elec_emissions_kg + gas_emissions_kg) / 1000

        self.logger.info(f"Calculated {total_emissions_tons:.2f} tCO2e for {country}")

        return {
            "total_emissions_tons": round(total_emissions_tons, 2),
            "electricity_emissions_tons": round(elec_emissions_kg / 1000, 2),
            "gas_emissions_tons": round(gas_emissions_kg / 1000, 2),
            "breakdown": {
                "electricity": {
                    "consumption_kwh": electricity_kwh,
                    "factor_kgco2e": elec_factor,
                    "emissions_tons": round(elec_emissions_kg / 1000, 2)
                },
                "gas": {
                    "consumption_therms": gas_therms,
                    "factor_kgco2e": gas_factor,
                    "emissions_tons": round(gas_emissions_kg / 1000, 2)
                }
            },
            "country": country
        }


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 1: Simple Agent - Basic Data Processing")
    print("="*70 + "\n")

    # Create agent instance
    agent = EmissionsCalculatorAgent()

    # Example 1: US office building
    print("Example 1: US Office Building")
    print("-" * 70)
    input_data = {
        "electricity_kwh": 50000,
        "gas_therms": 1000,
        "country": "US"
    }

    result = agent.run(input_data)

    if result.success:
        print(f"Input: {input_data}")
        print(f"\nResults:")
        print(f"  Total Emissions: {result.data['total_emissions_tons']:.2f} metric tons CO2e")
        print(f"  - Electricity: {result.data['electricity_emissions_tons']:.2f} tons")
        print(f"  - Natural Gas: {result.data['gas_emissions_tons']:.2f} tons")
    else:
        print(f"Error: {result.error}")

    # Example 2: UK office building (lower grid intensity)
    print("\n\nExample 2: UK Office Building (Lower Grid Intensity)")
    print("-" * 70)
    input_data_uk = {
        "electricity_kwh": 50000,
        "gas_therms": 1000,
        "country": "UK"
    }

    result_uk = agent.run(input_data_uk)

    if result_uk.success:
        print(f"Input: {input_data_uk}")
        print(f"\nResults:")
        print(f"  Total Emissions: {result_uk.data['total_emissions_tons']:.2f} metric tons CO2e")
        print(f"  - Electricity: {result_uk.data['electricity_emissions_tons']:.2f} tons")
        print(f"  - Natural Gas: {result_uk.data['gas_emissions_tons']:.2f} tons")

        # Show comparison
        reduction = result.data['total_emissions_tons'] - result_uk.data['total_emissions_tons']
        print(f"\nComparison: UK grid is {reduction:.2f} tons lower than US grid")

    # Example 3: Error handling - negative values
    print("\n\nExample 3: Error Handling - Invalid Input")
    print("-" * 70)
    bad_input = {
        "electricity_kwh": -100,
        "gas_therms": 1000,
        "country": "US"
    }

    result_bad = agent.run(bad_input)
    print(f"Input: {bad_input}")
    print(f"Success: {result_bad.success}")
    print(f"Error: {result_bad.error}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
