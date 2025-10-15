#!/usr/bin/env python3
"""
Example: Emission Calculation Using Registry

This script demonstrates how to use the Emission Factors Registry
for carbon footprint calculations with full provenance tracking.

Author: GreenLang Team
"""

from query_emission_factors import EmissionFactorRegistry
import json


def example_building_emissions():
    """
    Example: Calculate emissions for a commercial building.
    """
    print("=" * 70)
    print("EXAMPLE 1: Commercial Building Emissions Calculation")
    print("=" * 70)

    # Initialize registry
    registry = EmissionFactorRegistry()

    # Building data
    building_data = {
        "location": "California",
        "electricity_kwh": 50000,  # Monthly consumption
        "natural_gas_kwh": 15000,  # Monthly heating
        "diesel_liters": 200,      # Backup generator
    }

    print(f"\nBuilding Energy Consumption:")
    print(f"  Location: {building_data['location']}")
    print(f"  Electricity: {building_data['electricity_kwh']:,} kWh")
    print(f"  Natural Gas: {building_data['natural_gas_kwh']:,} kWh")
    print(f"  Diesel: {building_data['diesel_liters']} liters")

    # Calculate Scope 2 - Electricity
    print(f"\n{'-' * 70}")
    print("SCOPE 2: Purchased Electricity")
    print(f"{'-' * 70}")

    grid_factor = registry.get_grid_factor("US_WECC_CA")
    electricity_emissions = (
        building_data['electricity_kwh'] * grid_factor['emission_factor_kwh']
    )

    print(f"Grid: {grid_factor['name']}")
    print(f"Emission Factor: {grid_factor['emission_factor_kwh']} kg CO2e/kWh")
    print(f"Source: {grid_factor['source']}")
    print(f"URI: {grid_factor['uri']}")
    print(f"Renewable Share: {grid_factor['renewable_share'] * 100}%")
    print(f"\nEmissions: {electricity_emissions:,.2f} kg CO2e")
    print(f"           {electricity_emissions / 1000:.2f} metric tons CO2e")

    # Calculate Scope 1 - Natural Gas
    print(f"\n{'-' * 70}")
    print("SCOPE 1: Natural Gas Combustion")
    print(f"{'-' * 70}")

    gas_factor = registry.get_fuel_factor("natural_gas", unit="kwh")
    gas_emissions = building_data['natural_gas_kwh'] * gas_factor['emission_factor']

    print(f"Fuel: {gas_factor['name']}")
    print(f"Emission Factor: {gas_factor['emission_factor']} kg CO2e/kWh")
    print(f"Source: {gas_factor['source']}")
    print(f"URI: {gas_factor['uri']}")
    print(f"Data Quality: {gas_factor['data_quality']}")
    print(f"Uncertainty: {gas_factor['uncertainty']}")
    print(f"\nEmissions: {gas_emissions:,.2f} kg CO2e")
    print(f"           {gas_emissions / 1000:.2f} metric tons CO2e")

    # Calculate Scope 1 - Diesel
    print(f"\n{'-' * 70}")
    print("SCOPE 1: Diesel Backup Generator")
    print(f"{'-' * 70}")

    diesel_factor = registry.get_fuel_factor("diesel", unit="liter")
    diesel_emissions = building_data['diesel_liters'] * diesel_factor['emission_factor']

    print(f"Fuel: {diesel_factor['name']}")
    print(f"Emission Factor: {diesel_factor['emission_factor']} kg CO2e/liter")
    print(f"Source: {diesel_factor['source']}")
    print(f"\nEmissions: {diesel_emissions:,.2f} kg CO2e")
    print(f"           {diesel_emissions / 1000:.2f} metric tons CO2e")

    # Total emissions
    total_emissions = electricity_emissions + gas_emissions + diesel_emissions
    total_scope1 = gas_emissions + diesel_emissions
    total_scope2 = electricity_emissions

    print(f"\n{'=' * 70}")
    print("TOTAL EMISSIONS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Scope 1 (Direct): {total_scope1:,.2f} kg CO2e ({total_scope1/1000:.2f} tons)")
    print(f"Scope 2 (Electricity): {total_scope2:,.2f} kg CO2e ({total_scope2/1000:.2f} tons)")
    print(f"Total: {total_emissions:,.2f} kg CO2e ({total_emissions/1000:.2f} tons)")

    # Breakdown by source
    print(f"\n{'-' * 70}")
    print("Breakdown by Source:")
    print(f"  Electricity: {(electricity_emissions/total_emissions)*100:.1f}%")
    print(f"  Natural Gas: {(gas_emissions/total_emissions)*100:.1f}%")
    print(f"  Diesel: {(diesel_emissions/total_emissions)*100:.1f}%")

    return {
        'total_emissions_kg': total_emissions,
        'scope1_kg': total_scope1,
        'scope2_kg': total_scope2,
        'sources': {
            'electricity': {
                'emissions_kg': electricity_emissions,
                'source': grid_factor['source'],
                'uri': grid_factor['uri']
            },
            'natural_gas': {
                'emissions_kg': gas_emissions,
                'source': gas_factor['source'],
                'uri': gas_factor['uri']
            },
            'diesel': {
                'emissions_kg': diesel_emissions,
                'source': diesel_factor['source'],
                'uri': diesel_factor['uri']
            }
        }
    }


def example_industrial_process():
    """
    Example: Calculate emissions for dairy processing.
    """
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Industrial Process - Dairy Pasteurization")
    print("=" * 70)

    registry = EmissionFactorRegistry()

    # Process data
    milk_kg = 10000  # Daily milk processing
    steam_source = "natural_gas"

    print(f"\nProcess Parameters:")
    print(f"  Milk quantity: {milk_kg:,} kg/day")
    print(f"  Steam source: {steam_source}")

    # Get process energy intensity
    process = registry.get_process_factor("pasteurization")

    print(f"\n{'-' * 70}")
    print("Process: " + process['name'])
    print(f"{'-' * 70}")
    print(f"Temperature: {process['data']['typical_temperature_c']}°C")
    print(f"Duration: {process['data']['duration_seconds']} seconds")
    print(f"Energy Intensity: {process['data']['energy_intensity_kwh_per_kg']} kWh/kg")
    print(f"Source: {process['source']}")
    print(f"URI: {process['uri']}")

    # Calculate energy requirement
    energy_kwh = milk_kg * process['data']['energy_intensity_kwh_per_kg']
    print(f"\nEnergy Required: {energy_kwh:,.2f} kWh/day")

    # Get fuel emission factor
    fuel_factor = registry.get_fuel_factor(steam_source, unit="kwh")

    print(f"\n{'-' * 70}")
    print("Fuel: " + fuel_factor['name'])
    print(f"{'-' * 70}")
    print(f"Emission Factor: {fuel_factor['emission_factor']} kg CO2e/kWh")

    # Calculate emissions
    emissions = energy_kwh * fuel_factor['emission_factor']
    emissions_per_kg = emissions / milk_kg

    print(f"\n{'=' * 70}")
    print("EMISSIONS RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Daily: {emissions:,.2f} kg CO2e")
    print(f"Per kg milk: {emissions_per_kg:.4f} kg CO2e/kg")
    print(f"Annual: {(emissions * 365):,.2f} kg CO2e ({(emissions * 365)/1000:.2f} tons)")

    return {
        'process': 'pasteurization',
        'emissions_per_kg_product': emissions_per_kg,
        'daily_emissions_kg': emissions,
        'annual_emissions_kg': emissions * 365,
        'source_uri': process['uri']
    }


def example_transportation():
    """
    Example: Calculate emissions for freight transportation.
    """
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Transportation Emissions - Multi-Modal Freight")
    print("=" * 70)

    registry = EmissionFactorRegistry()

    # Shipment data
    cargo_tons = 20  # Cargo weight
    truck_km = 500   # Truck distance
    ocean_km = 8000  # Ocean distance

    print(f"\nShipment Details:")
    print(f"  Cargo weight: {cargo_tons} metric tons")
    print(f"  Truck transport: {truck_km:,} km")
    print(f"  Ocean freight: {ocean_km:,} km")

    # Truck freight
    truck = registry.get_process_factor("freight_truck_diesel")
    truck_emissions = (
        cargo_tons * truck_km * truck['data']['emission_factor_kg_co2e_per_ton_km']
    )

    print(f"\n{'-' * 70}")
    print("Truck Freight: " + truck['name'])
    print(f"{'-' * 70}")
    print(f"Emission Factor: {truck['data']['emission_factor_kg_co2e_per_ton_km']} kg CO2e/ton-km")
    print(f"Source: {truck['source']}")
    print(f"Emissions: {truck_emissions:,.2f} kg CO2e")

    # Ocean freight
    ocean = registry.get_process_factor("ocean_freight_container")
    ocean_emissions = (
        cargo_tons * ocean_km * ocean['data']['emission_factor_kg_co2e_per_ton_km']
    )

    print(f"\n{'-' * 70}")
    print("Ocean Freight: " + ocean['name'])
    print(f"{'-' * 70}")
    print(f"Emission Factor: {ocean['data']['emission_factor_kg_co2e_per_ton_km']} kg CO2e/ton-km")
    print(f"Source: {ocean['source']}")
    print(f"Emissions: {ocean_emissions:,.2f} kg CO2e")

    # Total
    total = truck_emissions + ocean_emissions

    print(f"\n{'=' * 70}")
    print("TOTAL TRANSPORTATION EMISSIONS")
    print(f"{'=' * 70}")
    print(f"Truck: {truck_emissions:,.2f} kg CO2e ({(truck_emissions/total)*100:.1f}%)")
    print(f"Ocean: {ocean_emissions:,.2f} kg CO2e ({(ocean_emissions/total)*100:.1f}%)")
    print(f"Total: {total:,.2f} kg CO2e ({total/1000:.2f} tons)")
    print(f"Per ton cargo: {total/cargo_tons:.2f} kg CO2e/ton")

    return {
        'total_emissions_kg': total,
        'truck_emissions_kg': truck_emissions,
        'ocean_emissions_kg': ocean_emissions,
        'emissions_per_ton': total / cargo_tons
    }


def example_comparison():
    """
    Example: Compare emission factors across regions.
    """
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Grid Factor Comparison Across Regions")
    print("=" * 70)

    registry = EmissionFactorRegistry()

    # Same building, different locations
    kwh = 50000  # Monthly consumption

    regions = [
        "US_WECC_CA",    # California
        "US_NATIONAL",   # US Average
        "UK",            # United Kingdom
        "FR",            # France
        "CN",            # China
        "IN"             # India
    ]

    print(f"\nElectricity consumption: {kwh:,} kWh/month\n")
    print(f"{'Region':<30} {'Factor':<12} {'Emissions':<15} {'Renewable'}")
    print(f"{'-' * 30} {'-' * 12} {'-' * 15} {'-' * 10}")

    results = []
    for region in regions:
        factor = registry.get_grid_factor(region)
        emissions = kwh * factor['emission_factor_kwh']
        renewable = factor.get('renewable_share', 0) * 100

        print(f"{factor['name']:<30} "
              f"{factor['emission_factor_kwh']:<12.3f} "
              f"{emissions:>10,.0f} kg  "
              f"{renewable:>6.1f}%")

        results.append({
            'region': region,
            'name': factor['name'],
            'factor': factor['emission_factor_kwh'],
            'emissions': emissions,
            'renewable_share': renewable
        })

    # Find min and max
    min_region = min(results, key=lambda x: x['emissions'])
    max_region = max(results, key=lambda x: x['emissions'])

    print(f"\n{'-' * 70}")
    print(f"Lowest emissions: {min_region['name']} "
          f"({min_region['emissions']:,.0f} kg CO2e)")
    print(f"Highest emissions: {max_region['name']} "
          f"({max_region['emissions']:,.0f} kg CO2e)")
    print(f"Difference: {max_region['emissions'] - min_region['emissions']:,.0f} kg CO2e "
          f"({((max_region['emissions']/min_region['emissions'])-1)*100:.0f}% higher)")

    return results


def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print("  Emission Factors Registry - Usage Examples".center(70))
    print("=" * 70)

    # Run examples
    building_result = example_building_emissions()
    process_result = example_industrial_process()
    transport_result = example_transportation()
    comparison_result = example_comparison()

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nAll calculations include:")
    print("  - Source URIs for audit compliance")
    print("  - Data quality indicators")
    print("  - Standard compliance (GHG Protocol, ISO 14064, IPCC)")
    print("  - Update dates and uncertainty ranges")
    print("\nFor full documentation, see:")
    print("  docs/EMISSION_FACTORS_SOURCES.md")
    print("\n")


if __name__ == "__main__":
    main()
