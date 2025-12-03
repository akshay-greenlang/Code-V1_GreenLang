"""
Test Emission Factor Database

Verify that the emission factor database is correctly loaded and
returns accurate factors for climate calculations.
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'core'))

from greenlang.validation import EmissionFactorDB, EmissionCategory, DataSource


def test_database_initialization():
    """Test database loads correctly."""
    print("\n" + "="*70)
    print("Testing Emission Factor Database Initialization")
    print("="*70)

    db = EmissionFactorDB()
    stats = db.get_database_stats()

    print(f"\nDatabase Statistics:")
    print(f"  Total factors: {stats['total_factors']}")
    print(f"  Fuel types: {stats['fuel_types']}")
    print(f"  Regions: {stats['regions']}")
    print(f"  Sources: {stats['sources']}")
    print(f"  Scope 1 factors: {stats['scope1_factors']}")
    print(f"  Scope 2 factors: {stats['scope2_factors']}")
    print(f"  Scope 3 factors: {stats['scope3_factors']}")

    # Verify we have a reasonable number of factors
    assert stats['total_factors'] >= 25, "Should have at least 25 emission factors"
    assert stats['fuel_types'] >= 15, "Should have at least 15 fuel types"

    print("\n[PASS] Database initialization PASSED\n")


def test_defra_factors():
    """Test DEFRA 2024 emission factors."""
    print("\n" + "="*70)
    print("Testing DEFRA 2024 Factors")
    print("="*70)

    db = EmissionFactorDB()

    # Test natural gas
    factor = db.get_factor('natural_gas', region='UK')
    assert factor is not None, "Natural gas factor should exist"
    assert factor.factor_value == 0.18385, f"Natural gas factor should be 0.18385, got {factor.factor_value}"
    assert factor.source == DataSource.DEFRA_2024
    print(f"[PASS] Natural Gas: {factor}")

    # Test diesel
    factor = db.get_factor('diesel', region='UK')
    assert factor is not None
    assert factor.factor_value == 2.687
    print(f"[PASS] Diesel: {factor}")

    # Test petrol
    factor = db.get_factor('petrol', region='UK')
    assert factor is not None
    assert factor.factor_value == 2.296
    print(f"[PASS] Petrol: {factor}")

    # Test coal
    factor = db.get_factor('coal', region='UK')
    assert factor is not None
    assert factor.factor_value == 2.269
    print(f"[PASS] Coal: {factor}")

    # Test LPG
    factor = db.get_factor('lpg', region='UK')
    assert factor is not None
    assert factor.factor_value == 1.508
    print(f"[PASS] LPG: {factor}")

    # Test UK electricity
    factor = db.get_factor('electricity', region='UK')
    assert factor is not None
    assert factor.factor_value == 0.193
    print(f"[PASS] UK Electricity: {factor}")

    print("\n[PASS] All DEFRA factors PASSED\n")


def test_epa_factors():
    """Test EPA eGRID 2023 factors."""
    print("\n" + "="*70)
    print("Testing EPA eGRID 2023 Factors")
    print("="*70)

    db = EmissionFactorDB()

    # Test US national average
    factor = db.get_factor('electricity', region='US')
    assert factor is not None
    assert factor.factor_value == 0.417
    assert factor.source == DataSource.EPA_EGRID_2023
    print(f"[PASS] US National Average: {factor}")

    # Test California (CAMX)
    factor = db.get_factor('electricity', region='US_CAMX')
    assert factor is not None
    assert factor.factor_value == 0.197
    print(f"[PASS] California (CAMX): {factor}")

    # Test Texas (ERCOT)
    factor = db.get_factor('electricity', region='US_ERCT')
    assert factor is not None
    assert factor.factor_value == 0.390
    print(f"[PASS] Texas (ERCOT): {factor}")

    print("\n[PASS] All EPA factors PASSED\n")


def test_transport_factors():
    """Test transport emission factors."""
    print("\n" + "="*70)
    print("Testing Transport Factors")
    print("="*70)

    db = EmissionFactorDB()

    # Test diesel car
    factor = db.get_factor('diesel_car', region='UK')
    assert factor is not None
    assert factor.factor_value == 0.171
    assert factor.category == EmissionCategory.SCOPE1
    print(f"[PASS] Diesel Car: {factor}")

    # Test petrol car
    factor = db.get_factor('petrol_car', region='UK')
    assert factor is not None
    assert factor.factor_value == 0.188
    print(f"[PASS] Petrol Car: {factor}")

    # Test HGV
    factor = db.get_factor('hgv_diesel', region='UK')
    assert factor is not None
    assert factor.factor_value == 0.953
    assert factor.category == EmissionCategory.SCOPE3
    print(f"[PASS] HGV: {factor}")

    # Test air freight
    factor = db.get_factor('air_freight', region='GLOBAL')
    assert factor is not None
    assert factor.factor_value == 1.234
    print(f"[PASS] Air Freight: {factor}")

    # Test sea freight
    factor = db.get_factor('sea_freight', region='GLOBAL')
    assert factor is not None
    assert factor.factor_value == 0.0113
    print(f"[PASS] Sea Freight: {factor}")

    print("\n[PASS] All transport factors PASSED\n")


def test_material_factors():
    """Test material production factors."""
    print("\n" + "="*70)
    print("Testing Material Production Factors")
    print("="*70)

    db = EmissionFactorDB()

    # Test steel
    factor = db.get_factor('steel', region='GLOBAL')
    assert factor is not None
    assert factor.factor_value == 2.1
    assert factor.category == EmissionCategory.SCOPE3
    print(f"[PASS] Steel: {factor}")

    # Test aluminium
    factor = db.get_factor('aluminium', region='GLOBAL')
    assert factor is not None
    assert factor.factor_value == 8.5
    print(f"[PASS] Aluminium: {factor}")

    # Test concrete
    factor = db.get_factor('concrete', region='GLOBAL')
    assert factor is not None
    assert factor.factor_value == 0.145
    print(f"[PASS] Concrete: {factor}")

    # Test cement
    factor = db.get_factor('cement', region='UK')
    assert factor is not None
    assert factor.factor_value == 0.876
    print(f"[PASS] Cement: {factor}")

    print("\n[PASS] All material factors PASSED\n")


def test_factor_lookup_by_id():
    """Test looking up factors by exact ID."""
    print("\n" + "="*70)
    print("Testing Factor Lookup by ID")
    print("="*70)

    db = EmissionFactorDB()

    # Test exact ID lookup
    factor = db.get_factor_by_id('defra_2024_natural_gas_gross_cv')
    assert factor is not None
    assert factor.factor_value == 0.18385
    print(f"[PASS] Lookup by ID: {factor}")

    # Test non-existent ID
    factor = db.get_factor_by_id('non_existent_id')
    assert factor is None
    print(f"[PASS] Non-existent ID returns None")

    print("\n[PASS] Factor lookup by ID PASSED\n")


def test_factor_search():
    """Test searching for factors."""
    print("\n" + "="*70)
    print("Testing Factor Search")
    print("="*70)

    db = EmissionFactorDB()

    # Search for all UK factors
    uk_factors = db.search_factors(region='UK')
    assert len(uk_factors) >= 5, "Should have multiple UK factors"
    print(f"[PASS] Found {len(uk_factors)} UK factors")

    # Search for all electricity factors
    electricity_factors = db.search_factors(fuel_type='electricity')
    assert len(electricity_factors) >= 5, "Should have multiple electricity factors"
    print(f"[PASS] Found {len(electricity_factors)} electricity factors")

    # Search for Scope 1 factors
    scope1_factors = db.search_factors(category=EmissionCategory.SCOPE1)
    assert len(scope1_factors) >= 5
    print(f"[PASS] Found {len(scope1_factors)} Scope 1 factors")

    print("\n[PASS] Factor search PASSED\n")


def test_fallback_to_global():
    """Test fallback to global factors when regional not found."""
    print("\n" + "="*70)
    print("Testing Fallback to Global Factors")
    print("="*70)

    db = EmissionFactorDB()

    # Request steel for non-existent region, should fall back to GLOBAL
    factor = db.get_factor('steel', region='NONEXISTENT')
    assert factor is not None
    assert factor.region == 'GLOBAL'
    print(f"[PASS] Fallback to global: {factor}")

    print("\n[PASS] Fallback mechanism PASSED\n")


def test_list_functions():
    """Test list functions."""
    print("\n" + "="*70)
    print("Testing List Functions")
    print("="*70)

    db = EmissionFactorDB()

    # List fuel types
    fuel_types = db.list_fuel_types()
    assert len(fuel_types) >= 15
    print(f"[PASS] Fuel types ({len(fuel_types)}): {', '.join(fuel_types[:5])}...")

    # List regions
    regions = db.list_regions()
    assert 'UK' in regions
    assert 'US' in regions
    assert 'GLOBAL' in regions
    print(f"[PASS] Regions ({len(regions)}): {', '.join(regions)}")

    print("\n[PASS] List functions PASSED\n")


def main():
    """Run all emission factor database tests."""
    print("\n" + "="*70)
    print("EMISSION FACTOR DATABASE TEST SUITE")
    print("="*70)

    try:
        test_database_initialization()
        test_defra_factors()
        test_epa_factors()
        test_transport_factors()
        test_material_factors()
        test_factor_lookup_by_id()
        test_factor_search()
        test_fallback_to_global()
        test_list_functions()

        print("\n" + "="*70)
        print("ALL EMISSION FACTOR DATABASE TESTS PASSED [PASS]")
        print("="*70 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
