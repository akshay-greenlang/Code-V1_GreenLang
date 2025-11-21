#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze building emissions using SDK"""

from greenlang.sdk import GreenLangClient
import sys

# Get country from command line (default: US)
country = sys.argv[1] if len(sys.argv) > 1 else "US"

# Initialize client
client = GreenLangClient(region=country)

print("\n" + "="*50)
print(f"BUILDING ANALYSIS - {country}")
print("="*50)

# Define building
building = {
    "metadata": {
        "building_type": "hospital",
        "area": 100000,
        "area_unit": "sqft",
        "location": {"country": country},
        "occupancy": 500,
        "floor_count": 5,
        "building_age": 10
    },
    "energy_consumption": {
        "electricity": {"value": 3500000, "unit": "kWh"},
        "natural_gas": {"value": 10000, "unit": "therms"},
        "diesel": {"value": 5000, "unit": "gallons"}
    }
}

print("\nBuilding Details:")
print(f"  Type: Hospital")
print(f"  Area: 100,000 sqft")
print(f"  Occupancy: 500 people")
print(f"  Location: {country}")

print("\nEnergy Consumption:")
print(f"  Electricity: 3,500,000 kWh/year")
print(f"  Natural Gas: 10,000 therms/year")
print(f"  Diesel: 5,000 gallons/year")

# Analyze
print("\nAnalyzing...")
result = client.analyze_building(building)

if result["success"]:
    data = result["data"]
    
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    
    # Emissions
    print(f"\nTotal Annual Emissions:")
    print(f"  {data['emissions']['total_co2e_tons']:.2f} metric tons CO2e")
    print(f"  {data['emissions']['total_co2e_kg']:,.0f} kg CO2e")
    
    # Intensity
    print(f"\nIntensity Metrics:")
    print(f"  Per sqft: {data['intensity']['intensities']['per_sqft_year']:.2f} kgCO2e/sqft/year")
    print(f"  Per person: {data['intensity']['intensities']['per_person_year']:.0f} kgCO2e/person/year")
    
    # Rating
    print(f"\nPerformance:")
    print(f"  Intensity Rating: {data['intensity'].get('performance_rating', 'N/A')}")
    print(f"  Benchmark Rating: {data['benchmark']['rating']}")
    
    # Breakdown
    print(f"\nEmissions Breakdown:")
    for item in data['emissions']['emissions_breakdown']:
        print(f"  {item['source']:12} : {item['co2e_tons']:7.2f} tons ({item['percentage']:5}%)")
    
    # Recommendations
    if 'recommendations' in data and 'quick_wins' in data['recommendations']:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(data['recommendations']['quick_wins'][:3], 1):
            print(f"  {i}. {rec['action']}")
            print(f"     Impact: {rec['impact']}, Payback: {rec['payback']}")
else:
    print(f"\nError: {result.get('error', 'Analysis failed')}")

print("\n" + "="*50)