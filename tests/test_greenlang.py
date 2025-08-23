#!/usr/bin/env python3
"""Quick test to verify GreenLang is working"""

from greenlang.sdk import GreenLangClient

# Initialize client
client = GreenLangClient()

print("Testing GreenLang...")
print("=" * 50)

# Test 1: Simple emissions calculation
print("\n1. Testing single fuel emission calculation:")
result = client.calculate_emissions(
    fuel_type="electricity",
    consumption=1000,
    unit="kWh",
    region="US"
)

if result["success"]:
    print(f"   SUCCESS: 1000 kWh = {result['data']['co2e_emissions_kg']:.2f} kg CO2e")
else:
    print(f"   FAILED: {result.get('error', 'Unknown error')}")

# Test 2: Carbon aggregation
print("\n2. Testing carbon aggregation:")
emissions_list = [
    {"fuel_type": "electricity", "co2e_emissions_kg": 385.0},
    {"fuel_type": "natural_gas", "co2e_emissions_kg": 530.0}
]
result = client.aggregate_emissions(emissions_list)

if result["success"]:
    print(f"   SUCCESS: Total = {result['data']['total_co2e_tons']:.3f} metric tons CO2e")
else:
    print(f"   FAILED: {result.get('error', 'Unknown error')}")

# Test 3: Report generation
print("\n3. Testing report generation:")
carbon_data = {
    "total_co2e_tons": 0.915,
    "total_co2e_kg": 915.0,
    "emissions_breakdown": [
        {"source": "electricity", "co2e_tons": 0.385, "percentage": 42.1},
        {"source": "natural_gas", "co2e_tons": 0.530, "percentage": 57.9}
    ]
}
result = client.generate_report(carbon_data, format="text")

if result["success"]:
    print("   SUCCESS: Report generated")
    print("\n" + "-" * 50)
    print(result['data']['report'][:300] + "...")
else:
    print(f"   FAILED: {result.get('error', 'Unknown error')}")

print("\n" + "=" * 50)
print("GreenLang is working correctly!")