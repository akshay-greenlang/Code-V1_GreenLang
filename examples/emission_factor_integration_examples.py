"""
Emission Factor Database Integration Examples

This file demonstrates how to use the EmissionFactorClient SDK and FactorBrokerAdapter
in your applications.

Examples included:
1. Basic factor lookup
2. Emission calculations
3. Geographic fallback
4. Unit conversions
5. Grid factors
6. Batch calculations
7. Integration with existing FactorBroker code
8. Complete audit trail
9. Performance optimization

Author: GreenLang Backend Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.sdk.emission_factor_client import (
    EmissionFactorClient,
    EmissionFactorNotFoundError,
    UnitNotAvailableError
)
from greenlang.adapters.factor_broker_adapter import create_factor_broker
from greenlang.models.emission_factor import FactorSearchCriteria


def example_1_basic_lookup():
    """Example 1: Basic emission factor lookup."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Emission Factor Lookup")
    print("="*80)

    with EmissionFactorClient() as client:
        # Get diesel fuel factor
        factor = client.get_factor("fuels_diesel")

        print(f"\nFactor ID: {factor.factor_id}")
        print(f"Name: {factor.name}")
        print(f"Emission Factor: {factor.emission_factor_kg_co2e} kg CO2e/{factor.unit}")
        print(f"Source: {factor.source.source_org}")
        print(f"Source URI: {factor.source.source_uri}")
        print(f"Last Updated: {factor.last_updated}")
        print(f"Data Quality Tier: {factor.data_quality.tier}")

        # Get additional units
        if factor.additional_units:
            print(f"\nAdditional Units:")
            for unit in factor.additional_units:
                print(f"  - {unit.emission_factor_value} kg CO2e/{unit.unit_name}")


def example_2_emissions_calculation():
    """Example 2: Calculate emissions from activity data."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Emissions Calculation with Audit Trail")
    print("="*80)

    with EmissionFactorClient() as client:
        # Calculate emissions from diesel consumption
        result = client.calculate_emissions(
            factor_id="fuels_diesel",
            activity_amount=100.0,
            activity_unit="gallon"
        )

        print(f"\nActivity: {result.activity_amount} {result.activity_unit}")
        print(f"Emission Factor Used: {result.factor_value_applied} kg CO2e/{result.activity_unit}")
        print(f"Total Emissions: {result.emissions_kg_co2e:.2f} kg CO2e")
        print(f"Total Emissions: {result.emissions_metric_tons_co2e:.4f} metric tons CO2e")
        print(f"\nCalculation Timestamp: {result.calculation_timestamp}")
        print(f"Audit Trail Hash: {result.audit_trail}")
        print(f"\nFactor Source: {result.factor_used.source.source_uri}")

        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")


def example_3_geographic_fallback():
    """Example 3: Geographic-specific factors with fallback."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Geographic Fallback Logic")
    print("="*80)

    with EmissionFactorClient() as client:
        # Get grid factor for California
        ca_factor = client.get_grid_factor("California")
        print(f"\nCalifornia Grid: {ca_factor.emission_factor_kg_co2e} kg CO2e/kWh")
        print(f"Renewable Share: {ca_factor.renewable_share*100:.1f}%")

        # Get national average
        us_factor = client.get_grid_factor("US")
        print(f"\nUS National Average: {us_factor.emission_factor_kg_co2e} kg CO2e/kWh")
        print(f"Renewable Share: {us_factor.renewable_share*100:.1f}%")

        # Calculate emissions from both
        electricity_kwh = 1000.0

        ca_emissions = electricity_kwh * ca_factor.emission_factor_kg_co2e
        us_emissions = electricity_kwh * us_factor.emission_factor_kg_co2e

        print(f"\nEmissions from {electricity_kwh} kWh:")
        print(f"  California: {ca_emissions:.2f} kg CO2e")
        print(f"  US Average: {us_emissions:.2f} kg CO2e")
        print(f"  Savings from CA grid: {us_emissions - ca_emissions:.2f} kg CO2e ({((us_emissions - ca_emissions)/us_emissions*100):.1f}%)")


def example_4_unit_conversions():
    """Example 4: Working with multiple units."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Unit Conversions")
    print("="*80)

    with EmissionFactorClient() as client:
        # Get natural gas factor
        factor = client.get_factor("fuels_natural_gas")

        print(f"\nNatural Gas Emission Factors:")
        print(f"Base unit ({factor.unit}): {factor.emission_factor_kg_co2e} kg CO2e")

        # Get all available units
        units_to_try = ["kwh", "m3", "mmbtu", "therm", "scf"]

        for unit in units_to_try:
            try:
                ef_value = factor.get_factor_for_unit(unit)
                print(f"Per {unit}: {ef_value} kg CO2e")
            except ValueError:
                print(f"Per {unit}: Not available")


def example_5_search_and_filter():
    """Example 5: Search and filter emission factors."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Search and Filter Factors")
    print("="*80)

    with EmissionFactorClient() as client:
        # Search by name
        diesel_factors = client.get_factor_by_name("diesel")
        print(f"\nFactors matching 'diesel': {len(diesel_factors)}")
        for f in diesel_factors[:3]:  # Show first 3
            print(f"  - {f.factor_id}: {f.name}")

        # Get all fuels
        fuel_factors = client.get_by_category("fuels")
        print(f"\nFuel factors: {len(fuel_factors)}")

        # Get all Scope 1 factors
        scope1_factors = client.get_by_scope("Scope 1")
        print(f"Scope 1 factors: {len(scope1_factors)}")

        # Advanced search with criteria
        criteria = FactorSearchCriteria(
            category="grids",
            geographic_scope="United States"
        )
        grid_factors = client.search_factors(criteria)
        print(f"\nUS Grid factors: {len(grid_factors)}")
        for f in grid_factors[:5]:
            print(f"  - {f.name}: {f.emission_factor_kg_co2e} kg CO2e/kWh")


def example_6_batch_calculations():
    """Example 6: Batch emission calculations."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Batch Calculations")
    print("="*80)

    with EmissionFactorClient() as client:
        # Define activities
        activities = [
            {"factor_id": "fuels_diesel", "amount": 100, "unit": "gallon"},
            {"factor_id": "fuels_gasoline_motor", "amount": 50, "unit": "gallon"},
            {"factor_id": "fuels_natural_gas", "amount": 1000, "unit": "kwh"},
            {"factor_id": "grids_us_wecc_ca", "amount": 5000, "unit": "kwh"},
        ]

        print("\nCalculating emissions for multiple activities:")
        total_emissions = 0.0

        for activity in activities:
            result = client.calculate_emissions(
                factor_id=activity["factor_id"],
                activity_amount=activity["amount"],
                activity_unit=activity["unit"]
            )

            print(f"\n{result.factor_used.name}:")
            print(f"  Activity: {result.activity_amount} {result.activity_unit}")
            print(f"  Emissions: {result.emissions_kg_co2e:.2f} kg CO2e")

            total_emissions += result.emissions_kg_co2e

        print(f"\n{'='*40}")
        print(f"Total Emissions: {total_emissions:.2f} kg CO2e")
        print(f"Total Emissions: {total_emissions/1000:.4f} metric tons CO2e")


def example_7_factor_broker_adapter():
    """Example 7: Using FactorBrokerAdapter for backward compatibility."""
    print("\n" + "="*80)
    print("EXAMPLE 7: FactorBrokerAdapter (Backward Compatibility)")
    print("="*80)

    # Create broker using adapter
    with create_factor_broker() as broker:
        # Old API: get_factor returns just the numeric value
        diesel_ef = broker.get_factor("diesel", unit="gallons")
        print(f"\nDiesel EF (gallons): {diesel_ef} kg CO2e/gallon")

        # Old API: calculate returns just the numeric value
        emissions = broker.calculate("diesel", 100.0, "gallons")
        print(f"Emissions from 100 gallons diesel: {emissions:.2f} kg CO2e")

        # New API: calculate_detailed returns full result
        result = broker.calculate_detailed("diesel", 100.0, "gallons")
        print(f"\nDetailed result includes:")
        print(f"  - Emissions: {result.emissions_kg_co2e} kg CO2e")
        print(f"  - Audit trail: {result.audit_trail}")
        print(f"  - Source URI: {result.factor_used.source.source_uri}")

        # Grid factors
        ca_grid = broker.get_grid_factor("California", unit="kwh")
        print(f"\nCalifornia grid: {ca_grid} kg CO2e/kWh")

        # Transportation factors
        truck_ef = broker.get_transport_factor("truck", unit="ton_km")
        print(f"Truck transport: {truck_ef} kg CO2e/ton-km")


def example_8_audit_trail_and_provenance():
    """Example 8: Complete audit trail and provenance."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Audit Trail and Provenance Tracking")
    print("="*80)

    with EmissionFactorClient() as client:
        result = client.calculate_emissions(
            factor_id="fuels_diesel",
            activity_amount=500.0,
            activity_unit="gallon"
        )

        print("\nComplete Audit Trail:")
        print(f"Calculation ID: {result.audit_trail[:16]}")
        print(f"Timestamp: {result.calculation_timestamp}")
        print(f"Full Audit Hash: {result.audit_trail}")

        print("\nActivity Data:")
        print(f"  Amount: {result.activity_amount} {result.activity_unit}")

        print("\nEmission Factor Used:")
        factor = result.factor_used
        print(f"  Factor ID: {factor.factor_id}")
        print(f"  Name: {factor.name}")
        print(f"  Value: {result.factor_value_applied} kg CO2e/{result.activity_unit}")
        print(f"  Last Updated: {factor.last_updated}")

        print("\nSource Provenance:")
        print(f"  Organization: {factor.source.source_org}")
        print(f"  Publication: {factor.source.source_publication or 'N/A'}")
        print(f"  URI: {factor.source.source_uri}")
        print(f"  Standard: {factor.source.standard}")

        print("\nData Quality:")
        print(f"  Tier: {factor.data_quality.tier}")
        print(f"  Uncertainty: {factor.data_quality.uncertainty_percent}%")

        print("\nResults:")
        print(f"  Emissions (kg CO2e): {result.emissions_kg_co2e:.2f}")
        print(f"  Emissions (metric tons): {result.emissions_metric_tons_co2e:.4f}")


def example_9_performance_and_caching():
    """Example 9: Performance optimization with caching."""
    print("\n" + "="*80)
    print("EXAMPLE 9: Performance Optimization")
    print("="*80)

    import time

    with EmissionFactorClient(enable_cache=True, cache_size=10000) as client:
        # First lookup (database query)
        start = time.time()
        factor1 = client.get_factor("fuels_diesel")
        time1 = (time.time() - start) * 1000
        print(f"\nFirst lookup (database): {time1:.3f} ms")

        # Second lookup (cached)
        start = time.time()
        factor2 = client.get_factor("fuels_diesel")
        time2 = (time.time() - start) * 1000
        print(f"Second lookup (cached): {time2:.3f} ms")

        print(f"\nSpeedup: {time1/time2:.1f}x faster")

        # Batch lookups
        factor_ids = [
            "fuels_diesel",
            "fuels_gasoline_motor",
            "fuels_natural_gas",
            "grids_us_national",
            "grids_us_wecc_ca"
        ]

        start = time.time()
        for fid in factor_ids:
            client.get_factor(fid)
        elapsed = (time.time() - start) * 1000

        print(f"\nBatch lookup of {len(factor_ids)} factors: {elapsed:.3f} ms")
        print(f"Average per factor: {elapsed/len(factor_ids):.3f} ms")


def example_10_database_statistics():
    """Example 10: Database statistics and metadata."""
    print("\n" + "="*80)
    print("EXAMPLE 10: Database Statistics")
    print("="*80)

    with EmissionFactorClient() as client:
        stats = client.get_statistics()

        print(f"\nDatabase Statistics:")
        print(f"  Total Factors: {stats['total_factors']}")
        print(f"  Stale Factors (>3 years): {stats['stale_factors']}")
        print(f"  Total Calculations Logged: {stats['total_calculations']}")

        print(f"\nFactors by Category:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {category}: {count}")

        print(f"\nFactors by Scope:")
        for scope, count in sorted(stats['by_scope'].items(), key=lambda x: -x[1]):
            print(f"  {scope}: {count}")

        print(f"\nFactors by Source:")
        for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {source}: {count}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*80)
    print("EMISSION FACTOR DATABASE INTEGRATION EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate the GreenLang EmissionFactorClient SDK")
    print("and its integration with existing applications.")
    print("\nFeatures:")
    print("  - 500+ verified emission factors")
    print("  - Zero-hallucination calculations")
    print("  - Complete audit trails")
    print("  - Geographic fallback logic")
    print("  - Unit conversions")
    print("  - High-performance caching")
    print("="*80)

    try:
        example_1_basic_lookup()
        example_2_emissions_calculation()
        example_3_geographic_fallback()
        example_4_unit_conversions()
        example_5_search_and_filter()
        example_6_batch_calculations()
        example_7_factor_broker_adapter()
        example_8_audit_trail_and_provenance()
        example_9_performance_and_caching()
        example_10_database_statistics()

        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
