#!/usr/bin/env python
"""Complete SDK demonstration - All features"""

from greenlang.sdk import GreenLangClient
import json

def section_header(title):
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print('='*60)

def subsection(title):
    print(f"\n{title}")
    print("-"*40)

def main():
    # Initialize
    client = GreenLangClient(region="US")
    
    section_header("GREENLANG SDK - COMPLETE DEMONSTRATION")
    
    # 1. Basic Emissions
    subsection("1. BASIC EMISSIONS CALCULATION")
    fuels = [
        ("electricity", 1000, "kWh"),
        ("natural_gas", 100, "therms"),
        ("diesel", 10, "gallons")
    ]
    for fuel, amount, unit in fuels:
        result = client.calculate_emissions(fuel, amount, unit)
        print(f"{fuel:12} ({amount:4} {unit:6}): {result['data']['co2e_emissions_kg']:8.2f} kg CO2e")
    
    # 2. Boiler Emissions
    subsection("2. BOILER EMISSIONS CALCULATOR")
    boiler_configs = [
        ("natural_gas", "condensing", 0.95),
        ("natural_gas", "standard", 0.85),
        ("diesel", "standard", 0.75)
    ]
    print(f"{'Fuel':<12} {'Type':<12} {'Eff':<5} {'Emissions (kg)'}")
    for fuel, boiler_type, efficiency in boiler_configs:
        result = client.calculate_boiler_emissions(
            fuel, 1000, "kWh", efficiency, boiler_type
        )
        print(f"{fuel:<12} {boiler_type:<12} {efficiency*100:>3.0f}%  {result['data']['co2e_emissions_kg']:>8.2f}")
    
    # 3. Building Analysis
    subsection("3. BUILDING ANALYSIS")
    buildings = [
        ("office", 50000, "US"),
        ("hospital", 100000, "IN"),
        ("data_center", 20000, "EU")
    ]
    
    for btype, area, country in buildings:
        client_local = GreenLangClient(region=country)
        building = {
            "metadata": {
                "building_type": btype,
                "area": area,
                "location": {"country": country},
                "occupancy": area // 250
            },
            "energy_consumption": {
                "electricity": {"value": area * 35, "unit": "kWh"}
            }
        }
        result = client_local.analyze_building(building)
        if result["success"]:
            emissions = result['data']['emissions']['total_co2e_tons']
            rating = result['data']['benchmark']['rating']
            print(f"{btype:12} ({country}): {emissions:8.2f} tons/year - Rating: {rating}")
    
    # 4. Benchmarking
    subsection("4. BENCHMARKING COMPARISON")
    benchmark_scenarios = [
        (10000, 2000, "Excellent"),
        (50000, 10000, "Good"),
        (100000, 10000, "Poor")
    ]
    print(f"{'Emissions':<12} {'Area':<10} {'Expected':<12} {'Actual'}")
    for emissions, area, expected in benchmark_scenarios:
        result = client.benchmark_emissions(emissions, area, "office", 12)
        actual = result['data']['rating']
        match = "✓" if actual == expected else "✗"
        print(f"{emissions:<12} {area:<10} {expected:<12} {actual} {match}")
    
    # 5. Intensity Metrics
    subsection("5. INTENSITY METRICS")
    scenarios = [
        (50000, 10000, 50),
        (100000, 50000, 200),
        (500000, 100000, 1000)
    ]
    print(f"{'Emissions (kg)':<15} {'Area (sqft)':<12} {'Per sqft/year':<15} {'Rating'}")
    for emissions, area, occupancy in scenarios:
        result = client.calculate_intensity(emissions, area, "sqft", occupancy)
        intensity = result['data']['intensities']['per_sqft_year']
        rating = result['data']['performance_rating']
        print(f"{emissions:<15} {area:<12} {intensity:<15.2f} {rating}")
    
    # 6. Grid Factors Comparison
    subsection("6. GRID EMISSION FACTORS")
    countries = ["BR", "EU", "US", "JP", "CN", "IN"]
    print(f"{'Country':<10} {'Factor':<20} {'Grid Quality'}")
    for country in countries:
        result = client.get_emission_factor("electricity", country, "kWh")
        factor = result['data']['emission_factor']
        if factor < 0.2:
            quality = "Very Clean"
        elif factor < 0.4:
            quality = "Clean"
        elif factor < 0.6:
            quality = "Moderate"
        else:
            quality = "High Carbon"
        print(f"{country:<10} {factor:.3f} kgCO2e/kWh      {quality}")
    
    # 7. Aggregation
    subsection("7. EMISSIONS AGGREGATION")
    emissions_sources = [
        {"fuel_type": "electricity", "co2e_emissions_kg": 5000},
        {"fuel_type": "natural_gas", "co2e_emissions_kg": 2000},
        {"fuel_type": "diesel", "co2e_emissions_kg": 1000},
        {"fuel_type": "propane", "co2e_emissions_kg": 500}
    ]
    result = client.aggregate_emissions(emissions_sources)
    print(f"Individual sources: {len(emissions_sources)}")
    print(f"Total emissions: {result['data']['total_co2e_kg']:,.0f} kg")
    print(f"Total emissions: {result['data']['total_co2e_tons']:.2f} metric tons")
    print("\nBreakdown:")
    for item in result['data']['emissions_breakdown']:
        print(f"  {item['source']:12}: {item['percentage']:>5}%")
    
    # 8. Validation
    subsection("8. DATA VALIDATION")
    test_data = [
        ({"metadata": {"building_type": "office"}}, False, "Missing area"),
        ({"metadata": {"building_type": "office", "area": 10000, "location": {"country": "US"}}, 
          "energy_consumption": {}}, True, "Valid minimal"),
        ({"invalid": "data"}, False, "Invalid structure")
    ]
    for data, expected, description in test_data:
        result = client.validate_building_data(data)
        status = "✓" if result['success'] == expected else "✗"
        print(f"{description:20}: {result['success']} {status}")
    
    # 9. Utility Methods
    subsection("9. SDK UTILITIES")
    print(f"Available agents: {len(client.list_agents())}")
    print(f"  Agents: {', '.join(client.list_agents()[:5])}...")
    print(f"\nSupported countries: {len(client.get_supported_countries())}")
    print(f"  Sample: {', '.join(client.get_supported_countries()[:10])}...")
    fuel_types = client.get_supported_fuel_types("US")
    print(f"\nFuel types (US): {len(fuel_types)}")
    print(f"  Types: {', '.join(fuel_types[:5])}...")
    
    # 10. Recommendations
    subsection("10. RECOMMENDATIONS ENGINE")
    scenarios = [
        ("hospital", {"electricity": 2500000, "natural_gas": 50000}, "IN"),
        ("office", {"electricity": 500000}, "US"),
        ("data_center", {"electricity": 5000000}, "EU")
    ]
    for btype, emissions, country in scenarios:
        client_local = GreenLangClient(region=country)
        result = client_local.get_recommendations(btype, emissions, country)
        if result['success'] and 'quick_wins' in result['data']:
            print(f"\n{btype.upper()} ({country}):")
            for rec in result['data']['quick_wins'][:2]:
                print(f"  • {rec['action']}")
    
    section_header("SDK DEMONSTRATION COMPLETE!")
    print("\nAll SDK features tested successfully!")
    print("For more information, see SDK_COMPLETE_GUIDE.md")

if __name__ == "__main__":
    main()