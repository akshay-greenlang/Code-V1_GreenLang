#!/usr/bin/env python
"""Compare emissions across countries"""

from greenlang.sdk import GreenLangClient

countries = ["US", "IN", "CN", "EU", "JP", "BR", "KR", "UK", "CA", "AU"]
fuel_type = "electricity"
amount = 1000
unit = "kWh"

print("\n" + "="*60)
print(f"COMPARING {amount} {unit} OF {fuel_type.upper()} ACROSS COUNTRIES")
print("="*60)

print(f"\nEmissions for {amount} {unit} of electricity:")
print("-"*50)
print(f"{'Country':<10} {'Emissions':<15} {'Factor':<20} {'Note'}")
print("-"*50)

results = []
for country in countries:
    client = GreenLangClient(region=country)
    result = client.calculate_emissions(fuel_type, amount, unit)
    if result["success"]:
        emissions = result['data']['co2e_emissions_kg']
        factor = result['data']['emission_factor']
        results.append((country, emissions, factor))

# Sort by emissions (cleanest to dirtiest)
results.sort(key=lambda x: x[1])

# Determine notes
cleanest = results[0][1]
dirtiest = results[-1][1]

for country, emissions, factor in results:
    note = ""
    if emissions == cleanest:
        note = "<- Cleanest grid"
    elif emissions == dirtiest:
        note = "<- Highest emissions"
    elif emissions < 200:
        note = "Clean"
    elif emissions < 400:
        note = "Moderate"
    else:
        note = "High carbon"
    
    print(f"{country:<10} {emissions:>8.1f} kg CO2e   {factor:>6.3f} kgCO2e/{unit}   {note}")

print("-"*50)
print(f"\nKey Insights:")
print(f"  * Cleanest: {results[0][0]} ({results[0][1]:.0f} kg CO2e)")
print(f"  * Dirtiest: {results[-1][0]} ({results[-1][1]:.0f} kg CO2e)")
print(f"  * Variation: {results[-1][1]/results[0][1]:.1f}x difference")

# Compare specific scenarios
print(f"\nPractical Example - Annual Usage (10,000 kWh):")
print("-"*50)
for country, emissions, factor in results[:5]:  # Top 5
    annual = emissions * 10
    print(f"  {country}: {annual/1000:.2f} metric tons CO2e/year")

print("\n" + "="*60)