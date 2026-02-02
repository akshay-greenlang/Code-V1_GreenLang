#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test BoilerAgent functionality"""

from greenlang.sdk import GreenLangClient

def test_boiler_emissions():
    """Test boiler emissions calculation"""
    print("Testing Boiler Emissions Calculation")
    print("="*50)
    
    client = GreenLangClient(region="US")
    
    # Test natural gas boiler
    result = client.calculate_boiler_emissions(
        fuel_type="natural_gas",
        thermal_output=1000,
        output_unit="kWh",
        efficiency=0.85,
        boiler_type="condensing"
    )
    
    if result["success"]:
        data = result["data"]
        print(f"[OK] Natural Gas Boiler:")
        print(f"  Thermal Output: 1000 kWh")
        print(f"  Efficiency: 85%")
        print(f"  Fuel Consumed: {data.get('fuel_consumption_value', 'N/A')} {data.get('fuel_consumption_unit', '')}")
        print(f"  CO2e Emissions: {data.get('co2e_emissions_kg', 0):.2f} kg")
        print(f"  CO2e Emissions: {data.get('co2e_emissions_kg', 0)/1000:.3f} tons")
    else:
        print(f"[FAIL] Error: {result.get('error')}")
    
    print("\n" + "="*50)
    
    # Test diesel boiler
    result = client.calculate_boiler_emissions(
        fuel_type="diesel",
        thermal_output=500,
        output_unit="kWh",
        efficiency=0.75,
        boiler_type="standard"
    )
    
    if result["success"]:
        data = result["data"]
        print(f"[OK] Diesel Boiler:")
        print(f"  Thermal Output: 500 kWh")
        print(f"  Efficiency: 75%")
        print(f"  Fuel Consumed: {data.get('fuel_consumption_value', 'N/A')} {data.get('fuel_consumption_unit', '')}")
        print(f"  CO2e Emissions: {data.get('co2e_emissions_kg', 0):.2f} kg")
        print(f"  CO2e Emissions: {data.get('co2e_emissions_kg', 0)/1000:.3f} tons")
    else:
        print(f"[FAIL] Error: {result.get('error')}")

if __name__ == "__main__":
    test_boiler_emissions()