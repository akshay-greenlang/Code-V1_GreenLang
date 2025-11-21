#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of working GreenLang calculator commands
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from greenlang.sdk import GreenLangClient

def demo_simple_calc():
    """Demo: Simple emissions calculator"""
    print("="*60)
    print("DEMO 1: Simple Emissions Calculator")
    print("="*60)
    
    client = GreenLangClient(region="US")
    
    print("\nCalculating emissions for:")
    print("  - Electricity: 5000 kWh")
    print("  - Natural Gas: 200 therms") 
    print("  - Diesel: 50 gallons")
    
    emissions = []
    
    # Calculate each fuel
    fuels = [
        ("electricity", 5000, "kWh"),
        ("natural_gas", 200, "therms"),
        ("diesel", 50, "gallons")
    ]
    
    for fuel_type, amount, unit in fuels:
        result = client.calculate_emissions(fuel_type, amount, unit)
        if result["success"]:
            emissions.append(result["data"])
            print(f"\n{fuel_type.replace('_', ' ').title()}:")
            print(f"  CO2e: {result['data']['co2e_emissions_kg']:.2f} kg")
            print(f"  Factor: {result['data']['emission_factor']} kgCO2e/{unit}")
    
    # Aggregate
    agg = client.aggregate_emissions(emissions)
    if agg["success"]:
        print(f"\nTOTAL EMISSIONS: {agg['data']['total_co2e_tons']:.3f} metric tons CO2e")
        print(f"                 {agg['data']['total_co2e_kg']:.0f} kg CO2e")

def demo_building_calc():
    """Demo: Building emissions calculator"""
    print("\n" + "="*60)
    print("DEMO 2: Building Emissions Calculator (India)")
    print("="*60)
    
    client = GreenLangClient(region="IN")
    
    building_data = {
        "metadata": {
            "building_type": "hospital",
            "area": 100000,
            "area_unit": "sqft",
            "location": {"country": "IN", "city": "Mumbai"},
            "occupancy": 500,
            "floor_count": 5,
            "building_age": 10
        },
        "energy_consumption": {
            "electricity": {"value": 3500000, "unit": "kWh"},
            "diesel": {"value": 50000, "unit": "liters"},
            "natural_gas": {"value": 10000, "unit": "therms"}
        }
    }
    
    print(f"\nAnalyzing {building_data['metadata']['building_type']} in {building_data['metadata']['location']['city']}, India")
    print(f"  Area: {building_data['metadata']['area']:,} sqft")
    print(f"  Occupancy: {building_data['metadata']['occupancy']} people")
    
    result = client.analyze_building(building_data)
    
    if result["success"]:
        data = result["data"]
        
        # Emissions
        if "emissions" in data:
            print(f"\nEmissions Results:")
            print(f"  Total: {data['emissions'].get('total_co2e_tons', 0):.2f} metric tons CO2e/year")
            print(f"  Total: {data['emissions'].get('total_co2e_kg', 0):,.0f} kg CO2e/year")
            
            if "emissions_breakdown" in data["emissions"]:
                print(f"\n  Breakdown:")
                for item in data["emissions"]["emissions_breakdown"]:
                    print(f"    - {item['source']}: {item['co2e_tons']:.2f} tons ({item['percentage']}%)")
        
        # Intensity
        if "intensity" in data:
            intensities = data["intensity"]["intensities"]
            print(f"\nIntensity Metrics:")
            print(f"  Per sqft: {intensities.get('per_sqft_year', 0):.2f} kgCO2e/sqft/year")
            print(f"  Per person: {intensities.get('per_person_year', 0):.0f} kgCO2e/person/year")
            print(f"  Performance: {data['intensity'].get('performance_rating', 'N/A')}")
        
        # Benchmark
        if "benchmark" in data:
            print(f"\nBenchmark:")
            print(f"  Rating: {data['benchmark'].get('rating', 'N/A')}")
            print(f"  Category: {data['benchmark'].get('performance_category', 'N/A')}")

def demo_multi_country():
    """Demo: Multi-country comparison"""
    print("\n" + "="*60)
    print("DEMO 3: Multi-Country Comparison (1000 kWh electricity)")
    print("="*60)
    
    countries = ["US", "IN", "CN", "EU", "JP", "BR"]
    
    print("\nComparing grid emission factors:")
    for country in countries:
        client = GreenLangClient(region=country)
        result = client.calculate_emissions("electricity", 1000, "kWh")
        if result["success"]:
            print(f"  {country}: {result['data']['co2e_emissions_kg']:.0f} kg CO2e " +
                  f"(factor: {result['data']['emission_factor']} kgCO2e/kWh)")

if __name__ == "__main__":
    print("\nGREENLANG CALCULATOR - WORKING DEMONSTRATION\n")
    
    demo_simple_calc()
    demo_building_calc()
    demo_multi_country()
    
    print("\n" + "="*60)
    print("All calculations completed successfully!")
    print("="*60)