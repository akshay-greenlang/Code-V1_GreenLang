#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate emissions using GreenLang SDK"""

from greenlang.sdk import GreenLangClient

# Initialize client
client = GreenLangClient(region="US")

print("\n" + "="*50)
print("GREENLANG SDK - EMISSIONS CALCULATOR")
print("="*50)

# Calculate emissions for different fuel types
electricity = client.calculate_emissions("electricity", 5000, "kWh")
natural_gas = client.calculate_emissions("natural_gas", 200, "therms")
diesel = client.calculate_emissions("diesel", 50, "gallons")

# Display results
print("\nEMISSIONS CALCULATION RESULTS:")
print("-"*40)
print(f"Electricity (5000 kWh):")
print(f"  CO2e: {electricity['data']['co2e_emissions_kg']:.2f} kg")
print(f"  Factor: {electricity['data']['emission_factor']} kgCO2e/kWh")

print(f"\nNatural Gas (200 therms):")
print(f"  CO2e: {natural_gas['data']['co2e_emissions_kg']:.2f} kg")
print(f"  Factor: {natural_gas['data']['emission_factor']} kgCO2e/therm")

print(f"\nDiesel (50 gallons):")
print(f"  CO2e: {diesel['data']['co2e_emissions_kg']:.2f} kg")
print(f"  Factor: {diesel['data']['emission_factor']} kgCO2e/gallon")

# Aggregate total
emissions_list = [
    {"fuel_type": "electricity", "co2e_emissions_kg": electricity['data']['co2e_emissions_kg']},
    {"fuel_type": "natural_gas", "co2e_emissions_kg": natural_gas['data']['co2e_emissions_kg']},
    {"fuel_type": "diesel", "co2e_emissions_kg": diesel['data']['co2e_emissions_kg']}
]
total = client.aggregate_emissions(emissions_list)

print("\n" + "="*40)
print(f"TOTAL EMISSIONS: {total['data']['total_co2e_tons']:.3f} metric tons CO2e")
print(f"                 {total['data']['total_co2e_kg']:.0f} kg CO2e")
print("="*40)