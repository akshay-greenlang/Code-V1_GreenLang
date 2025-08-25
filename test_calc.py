#!/usr/bin/env python
"""Test the calculator functionality directly"""

from greenlang.sdk import GreenLangClient

def test_simple_calc():
    """Test simple emissions calculation"""
    print("Testing Simple Calculator...")
    
    client = GreenLangClient(region="US")
    
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
    
    client = GreenLangClient(region="IN")
    
    building_data = {
        "metadata": {
            "building_type": "hospital",
            "area": 100000,
            "area_unit": "sqft",
            "location": {"country": "IN"},
            "occupancy": 500,
            "floor_count": 5,
            "building_age": 10
        },
        "energy_consumption": {
            "electricity": {"value": 3500000, "unit": "kWh"},
            "diesel": {"value": 50000, "unit": "liters"}
        }
    }
    
    result = client.analyze_building(building_data)
    if result["success"]:
        data = result["data"]
        print(f"[OK] Building Analysis Complete")
        if "emissions" in data:
            print(f"  Total: {data['emissions'].get('total_co2e_tons', 0):.2f} metric tons CO2e")
        if "intensity" in data:
            print(f"  Intensity: {data['intensity']['intensities'].get('per_sqft_year', 0):.2f} kgCO2e/sqft/year")
        if "benchmark" in data:
            print(f"  Rating: {data['benchmark'].get('rating', 'N/A')}")
    else:
        print(f"[FAIL] Building analysis failed: {result.get('error')}")

if __name__ == "__main__":
    test_simple_calc()
    test_building_calc()
    print("\nAll tests completed!")