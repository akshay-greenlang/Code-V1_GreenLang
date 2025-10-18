#!/usr/bin/env python3
"""
Example 7: Unit Conversion Calculator
======================================

This example demonstrates unit conversion for emissions calculations:
- Energy unit conversions (kWh, MWh, GJ, BTU)
- Area unit conversions (sqm, sqft)
- Volume conversions for fuels
- Automatic unit detection and conversion

Run: python examples/07_unit_converter.py
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
from greenlang.sdk.base import Agent, Result, Metadata, Transform


class UnitConverter(Transform[Tuple[float, str, str], float]):
    """
    Unit converter for common climate calculations.
    """

    def __init__(self):
        # Energy conversions (to kWh)
        self.energy_conversions = {
            "kWh": 1.0,
            "MWh": 1000.0,
            "GJ": 277.78,
            "BTU": 0.000293071,
            "kBTU": 0.293071,
            "therm": 29.3001
        }

        # Area conversions (to sqm)
        self.area_conversions = {
            "sqm": 1.0,
            "sqft": 0.092903,
            "sqkm": 1000000.0,
            "acre": 4046.86,
            "hectare": 10000.0
        }

        # Mass conversions (to kg)
        self.mass_conversions = {
            "kg": 1.0,
            "g": 0.001,
            "ton": 1000.0,
            "tonne": 1000.0,
            "lb": 0.453592,
            "oz": 0.0283495
        }

    def apply(self, data: Tuple[float, str, str]) -> float:
        """
        Convert value from one unit to another.

        Args:
            data: (value, from_unit, to_unit)

        Returns:
            Converted value
        """
        value, from_unit, to_unit = data

        # Determine conversion type
        if from_unit in self.energy_conversions and to_unit in self.energy_conversions:
            # Convert to base unit (kWh) then to target
            base_value = value * self.energy_conversions[from_unit]
            return base_value / self.energy_conversions[to_unit]

        elif from_unit in self.area_conversions and to_unit in self.area_conversions:
            # Convert to base unit (sqm) then to target
            base_value = value * self.area_conversions[from_unit]
            return base_value / self.area_conversions[to_unit]

        elif from_unit in self.mass_conversions and to_unit in self.mass_conversions:
            # Convert to base unit (kg) then to target
            base_value = value * self.mass_conversions[from_unit]
            return base_value / self.mass_conversions[to_unit]

        else:
            raise ValueError(f"Cannot convert {from_unit} to {to_unit}")


class UniversalEmissionsCalculator(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Emissions calculator that accepts any common unit.

    Automatically converts input units to standard units before calculation.
    """

    def __init__(self):
        metadata = Metadata(
            id="universal_calculator",
            name="Universal Emissions Calculator",
            version="1.0.0",
            description="Calculator with automatic unit conversion",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        self.converter = UnitConverter()

        # Load emission factors
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "emission_factors.json") as f:
            self.factors = json.load(f)["factors"]

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return "energy_data" in input_data

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate emissions with automatic unit conversion"""
        energy_data = input_data["energy_data"]
        country = input_data.get("country", "US")

        results = []
        total_emissions = 0.0
        conversions_applied = []

        for entry in energy_data:
            fuel_type = entry["fuel_type"]
            consumption = entry["consumption"]
            unit = entry["unit"]

            # Normalize units
            normalized_consumption, normalized_unit = self._normalize_energy_unit(
                fuel_type, consumption, unit
            )

            # Track conversion
            if unit != normalized_unit:
                conversions_applied.append({
                    "original": f"{consumption} {unit}",
                    "converted": f"{normalized_consumption:.2f} {normalized_unit}",
                    "fuel_type": fuel_type
                })

            # Get emission factor
            factor = self.factors[fuel_type][country]["value"]

            # Calculate emissions
            emissions_kg = normalized_consumption * factor
            emissions_tons = emissions_kg / 1000
            total_emissions += emissions_tons

            results.append({
                "fuel_type": fuel_type,
                "original_value": consumption,
                "original_unit": unit,
                "normalized_value": round(normalized_consumption, 2),
                "normalized_unit": normalized_unit,
                "factor": factor,
                "emissions_tons": round(emissions_tons, 4)
            })

        return {
            "total_emissions_tons": round(total_emissions, 4),
            "breakdown": results,
            "conversions_applied": conversions_applied,
            "country": country
        }

    def _normalize_energy_unit(self, fuel_type: str, value: float, unit: str) -> Tuple[float, str]:
        """
        Normalize energy units to standard units.

        For electricity: kWh
        For gas: therms
        """
        if fuel_type == "electricity":
            # Convert to kWh
            if unit in self.converter.energy_conversions:
                normalized = self.converter.apply((value, unit, "kWh"))
                return normalized, "kWh"

        elif fuel_type == "natural_gas":
            # Already in therms or convert
            if unit == "therms":
                return value, "therms"
            elif unit in ["m3", "cubic_meters"]:
                # 1 m3 ≈ 0.37 therms
                return value * 0.37, "therms"

        # Return as-is if no conversion needed
        return value, unit


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 7: Unit Conversion Calculator")
    print("="*70 + "\n")

    # Test 1: Energy unit conversions
    print("Test 1: Energy Unit Conversions")
    print("-" * 70)

    converter = UnitConverter()

    energy_tests = [
        (100, "kWh", "MWh"),
        (1, "MWh", "kWh"),
        (1, "GJ", "kWh"),
        (1000, "BTU", "kWh"),
        (100, "therm", "kWh")
    ]

    for value, from_unit, to_unit in energy_tests:
        result = converter.apply((value, from_unit, to_unit))
        print(f"{value} {from_unit} = {result:.4f} {to_unit}")

    # Test 2: Area unit conversions
    print("\n\nTest 2: Area Unit Conversions")
    print("-" * 70)

    area_tests = [
        (1000, "sqm", "sqft"),
        (10000, "sqft", "sqm"),
        (1, "acre", "sqm"),
        (1, "hectare", "sqm")
    ]

    for value, from_unit, to_unit in area_tests:
        result = converter.apply((value, from_unit, to_unit))
        print(f"{value} {from_unit} = {result:.2f} {to_unit}")

    # Test 3: Universal calculator with mixed units
    print("\n\nTest 3: Universal Calculator with Mixed Units")
    print("-" * 70)

    calculator = UniversalEmissionsCalculator()

    mixed_units_input = {
        "energy_data": [
            {"fuel_type": "electricity", "consumption": 50, "unit": "MWh"},
            {"fuel_type": "electricity", "consumption": 100, "unit": "GJ"},
            {"fuel_type": "natural_gas", "consumption": 1000, "unit": "therms"}
        ],
        "country": "US"
    }

    result = calculator.run(mixed_units_input)

    if result.success:
        print(f"Total Emissions: {result.data['total_emissions_tons']:.4f} tCO2e")
        print(f"\nConversions Applied:")
        for conv in result.data['conversions_applied']:
            print(f"  {conv['fuel_type']}: {conv['original']} → {conv['converted']}")

        print(f"\nDetailed Breakdown:")
        for item in result.data['breakdown']:
            print(f"  {item['fuel_type']}:")
            print(f"    Original: {item['original_value']} {item['original_unit']}")
            print(f"    Normalized: {item['normalized_value']} {item['normalized_unit']}")
            print(f"    Emissions: {item['emissions_tons']:.4f} tons")

    # Test 4: Different input units, same result
    print("\n\nTest 4: Verification - Different Units, Same Result")
    print("-" * 70)

    # 50 MWh = 50,000 kWh
    input_mwh = {
        "energy_data": [
            {"fuel_type": "electricity", "consumption": 50, "unit": "MWh"}
        ],
        "country": "US"
    }

    input_kwh = {
        "energy_data": [
            {"fuel_type": "electricity", "consumption": 50000, "unit": "kWh"}
        ],
        "country": "US"
    }

    result_mwh = calculator.run(input_mwh)
    result_kwh = calculator.run(input_kwh)

    print(f"50 MWh emissions: {result_mwh.data['total_emissions_tons']:.4f} tCO2e")
    print(f"50,000 kWh emissions: {result_kwh.data['total_emissions_tons']:.4f} tCO2e")
    print(f"Results match: {abs(result_mwh.data['total_emissions_tons'] - result_kwh.data['total_emissions_tons']) < 0.0001}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
