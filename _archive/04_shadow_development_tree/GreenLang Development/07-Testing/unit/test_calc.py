#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test the calculator functionality directly"""

from greenlang.sdk import GreenLangClient

def test_simple_calc():
    """Test simple emissions calculation"""
    print("Testing Simple Calculator...")
    
    client = GreenLangClient()
    
    # Test electricity
    result = client.calculate_emissions("electricity", 5000, "kWh")
    if result["success"]:
        print(f"[OK] Electricity: {result['data']['co2e_emissions_kg']:.2f} kg CO2e")
    else:
        print(f"[FAIL] Electricity failed: {result.get('error')}")
    
    # Test natural gas
    result = client.calculate_emissions("natural_gas", 200, "therms")
    if result["success"]:
        print(f"[OK] Natural Gas: {result['data']['co2e_emissions_kg']:.2f} kg CO2e")
    else:
        print(f"[FAIL] Natural Gas failed: {result.get('error')}")
    
    # Test diesel
    result = client.calculate_emissions("diesel", 50, "gallons")
    if result["success"]:
        print(f"[OK] Diesel: {result['data']['co2e_emissions_kg']:.2f} kg CO2e")
    else:
        print(f"[FAIL] Diesel failed: {result.get('error')}")
    
    # Test aggregation
    emissions_list = [
        {"fuel_type": "electricity", "co2e_emissions_kg": 1925.0},
        {"fuel_type": "natural_gas", "co2e_emissions_kg": 1060.0},
        {"fuel_type": "diesel", "co2e_emissions_kg": 510.5}
    ]
    
    agg_result = client.aggregate_emissions(emissions_list)
    if agg_result["success"]:
        print(f"\n[OK] Total Emissions: {agg_result['data']['total_co2e_tons']:.3f} metric tons")
    else:
        print(f"[FAIL] Aggregation failed: {agg_result.get('error')}")

def test_building_calc():
    """Test building calculator"""
    print("\n\nTesting Building Calculator...")

    client = GreenLangClient()

    # Calculate emissions for electricity
    elec_result = client.calculate_emissions("electricity", 3500000, "kWh")
    diesel_result = client.calculate_emissions("diesel", 50000, "liters")

    if elec_result["success"] and diesel_result["success"]:
        # Aggregate emissions
        emissions_list = [
            elec_result["data"],
            diesel_result["data"]
        ]

        agg_result = client.aggregate_emissions(emissions_list)

        if agg_result["success"]:
            total_kg = agg_result["data"]["total_co2e_kg"]

            # Benchmark the building
            bench_result = client.benchmark_emissions(
                total_emissions_kg=total_kg,
                building_area=100000,
                building_type="hospital",
                period_months=12
            )

            if bench_result["success"]:
                print(f"[OK] Building Analysis Complete")
                print(f"  Total: {agg_result['data']['total_co2e_tons']:.2f} metric tons CO2e")
                if "emissions_intensity_kg_per_sqft" in bench_result["data"]:
                    print(f"  Intensity: {bench_result['data']['emissions_intensity_kg_per_sqft']:.2f} kgCO2e/sqft/year")
                if "rating" in bench_result["data"]:
                    print(f"  Rating: {bench_result['data']['rating']}")
            else:
                print(f"[FAIL] Benchmark failed: {bench_result.get('error')}")
        else:
            print(f"[FAIL] Aggregation failed: {agg_result.get('error')}")
    else:
        print(f"[FAIL] Emissions calculation failed")

if __name__ == "__main__":
    test_simple_calc()
    test_building_calc()
    print("\nAll tests completed!")