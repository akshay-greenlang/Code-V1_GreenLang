"""Test EmissionFactorClient SDK functionality"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from greenlang.sdk.emission_factor_client import EmissionFactorClient

    print("=" * 60)
    print("TESTING EMISSION FACTOR CLIENT SDK")
    print("=" * 60)

    # Initialize client
    client = EmissionFactorClient(
        db_path="greenlang/data/emission_factors.db"
    )

    # Test 1: Get statistics
    print("\nTest 1: Database Statistics")
    print("-" * 60)
    stats = client.get_statistics()
    print(f"Total factors: {stats.get('total_factors', 'N/A')}")
    print(f"Categories: {stats.get('total_categories', 'N/A')}")

    # Test 2: Get a specific factor
    print("\nTest 2: Get Specific Factor (natural_gas)")
    print("-" * 60)
    factor = client.get_factor("natural_gas")
    if factor:
        print(f"Factor ID: {factor.factor_id}")
        print(f"Name: {factor.name}")
        print(f"Value: {factor.emission_factor_kg_co2e} {factor.unit}")
        print(f"Scope: {factor.scope}")
        print(f"Source: {factor.source.source_org}")
        print(f"URI: {factor.source.source_uri}")
    else:
        print("Factor not found")

    # Test 3: Get another factor
    print("\nTest 3: Get Another Factor (diesel)")
    print("-" * 60)
    diesel = client.get_factor("diesel")
    if diesel:
        print(f"Factor ID: {diesel.factor_id}")
        print(f"Name: {diesel.name}")
        print(f"Value: {diesel.emission_factor_kg_co2e} kg CO2e/{diesel.unit}")
        print(f"Category: {diesel.category}")

    # Test 4: Get grid factor
    print("\nTest 4: Get Grid Factor (grid_us_national)")
    print("-" * 60)
    grid = client.get_factor("grid_us_national")
    if grid:
        print(f"Factor ID: {grid.factor_id}")
        print(f"Name: {grid.name}")
        print(f"Value: {grid.emission_factor_kg_co2e} kg CO2e/{grid.unit}")
        print(f"Source: {grid.source.source_org}")
    else:
        print("Grid factor not found - trying alternative IDs")
        # Try some alternative IDs
        for alt_id in ["grid_usa", "electricity_us", "us_grid"]:
            alt_grid = client.get_factor(alt_id)
            if alt_grid:
                print(f"Found: {alt_id} - {alt_grid.name}")
                break

    print("\n" + "=" * 60)
    print("SDK TESTS COMPLETE")
    print("=" * 60)

except Exception as e:
    print(f"Error during SDK test: {e}")
    import traceback
    traceback.print_exc()
