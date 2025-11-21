# -*- coding: utf-8 -*-
# Simple GreenLang Test
# This script calculates carbon footprint for electricity usage

# Import emission factors data
import json
import os

# Simple calculation without SDK
def calculate_emissions(kwh, country="India"):
    # Emission factors (kg CO2 per kWh)
    emission_factors = {
        "India": 0.82,
        "USA": 0.42,
        "UK": 0.23,
        "China": 0.57,
        "default": 0.5
    }
    
    factor = emission_factors.get(country, emission_factors["default"])
    emissions_kg = kwh * factor
    emissions_tons = emissions_kg / 1000
    
    return emissions_tons

# Test the function
if __name__ == "__main__":
    print("\n" + "="*50)
    print("GREENLANG - CARBON FOOTPRINT CALCULATOR")
    print("="*50)
    
    # Calculate for 1000 kWh in India
    kwh = 1000
    country = "India"
    
    emissions = calculate_emissions(kwh, country)
    
    print(f"\nElectricity Usage: {kwh} kWh")
    print(f"Location: {country}")
    print(f"Emission Factor: 0.82 kg CO2/kWh")
    print(f"\nResults:")
    print(f"  Monthly Emissions: {emissions:.4f} tons CO2e")
    print(f"  Annual Emissions: {emissions * 12:.4f} tons CO2e")
    print(f"  Daily Average: {emissions / 30:.4f} tons CO2e")
    
    print("\nComparison with other countries:")
    for country_name in ["USA", "UK", "China"]:
        country_emissions = calculate_emissions(kwh, country_name)
        print(f"  {country_name}: {country_emissions:.4f} tons CO2e")
    
    print("\n" + "="*50)
    print("Calculation Complete!")
    print("="*50)