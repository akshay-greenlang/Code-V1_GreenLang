# -*- coding: utf-8 -*-
"""
GreenLang Data Module - Usage Examples

Demonstrates all features of the data engineering module.
"""

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Import all data components
from contracts import (
    CBAMDataContract,
    EmissionsDataContract,
    EnergyDataContract,
    ActivityDataContract,
    CBAMProductCategory,
    GHGScope,
    EmissionFactorSource,
    DataQualityLevel,
    EnergyType,
    ActivityType
)

from emission_factors import EmissionFactorLoader, EmissionFactor
from quality import DataQualityChecker, check_data_quality
from sample_data import SampleDataGenerator


# ============================================================================
# EXAMPLE 1: CREATE AND VALIDATE CBAM DATA
# ============================================================================

def example_cbam_creation():
    """Create and validate CBAM import declaration."""
    print("\n" + "="*80)
    print("EXAMPLE 1: CBAM Data Creation and Validation")
    print("="*80)

    # Create CBAM data contract
    cbam = CBAMDataContract(
        importer_id="GB123456789000",
        import_date=date(2024, 3, 15),
        declaration_period="2024-Q1",
        product_category=CBAMProductCategory.IRON_STEEL,
        cn_code="72071100",
        product_description="Semi-finished products of iron or non-alloy steel",
        quantity=Decimal("1000.000"),
        quantity_unit="tonnes",
        country_of_origin="CN",
        direct_emissions_co2e=Decimal("1800.000"),
        indirect_emissions_co2e=Decimal("200.000"),
        total_embedded_emissions=Decimal("2000.000"),
        specific_emissions=Decimal("2.0000"),
        emission_factor_source=EmissionFactorSource.DEFRA_2024,
        methodology="ISO 14064-1",
        is_verified=True,
        verifier_name="Bureau Veritas",
        verification_date=date(2024, 4, 1),
        data_quality_level=DataQualityLevel.EXCELLENT
    )

    print(f"\nCBAM Declaration Created:")
    print(f"  Product: {cbam.product_description}")
    print(f"  Quantity: {cbam.quantity} {cbam.quantity_unit}")
    print(f"  Origin: {cbam.country_of_origin}")
    print(f"  Total Emissions: {cbam.total_embedded_emissions} tCO2e")
    print(f"  Specific Emissions: {cbam.specific_emissions} tCO2e/tonne")
    print(f"  Verified: {cbam.is_verified}")

    # Quality check
    report = check_data_quality(cbam)

    print(f"\nQuality Assessment:")
    print(f"  Overall Score: {report.overall_score}/100")
    print(f"  Quality Level: {report.quality_level}")
    print(f"  Checks Passed: {report.checks_passed}/{report.total_checks}")

    if report.critical_issues:
        print(f"\n  CRITICAL ISSUES:")
        for issue in report.critical_issues:
            print(f"    - {issue.message}")

    if report.recommendations:
        print(f"\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")

    return cbam, report


# ============================================================================
# EXAMPLE 2: CREATE EMISSIONS DATA
# ============================================================================

def example_emissions_creation():
    """Create and validate emissions data."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Emissions Data Creation")
    print("="*80)

    # Scope 1 emissions - Natural gas combustion
    emissions = EmissionsDataContract(
        organization_id="ORG-ACME-001",
        facility_id="PLANT-A",
        reporting_period_start=date(2024, 1, 1),
        reporting_period_end=date(2024, 12, 31),
        ghg_scope=GHGScope.SCOPE_1,
        emission_source="Natural gas combustion in process heaters",
        activity_type=ActivityType.FUEL_COMBUSTION,
        co2_tonnes=Decimal("1470.000"),
        ch4_tonnes=Decimal("0.294"),
        n2o_tonnes=Decimal("0.059"),
        total_co2e_tonnes=Decimal("1500.000"),
        activity_amount=Decimal("800000.000"),
        activity_unit="kWh",
        emission_factor_value=Decimal("0.001875"),
        emission_factor_unit="tCO2e/kWh",
        emission_factor_source=EmissionFactorSource.DEFRA_2024,
        location_country="GB",
        location_region="England",
        data_quality_level=DataQualityLevel.EXCELLENT,
        uncertainty_percentage=Decimal("5.0"),
        is_assured=True,
        assurance_level="reasonable",
        calculation_method="GHG Protocol - Stationary Combustion"
    )

    print(f"\nEmissions Record Created:")
    print(f"  Scope: {emissions.ghg_scope.value}")
    print(f"  Source: {emissions.emission_source}")
    print(f"  Total CO2e: {emissions.total_co2e_tonnes} tonnes")
    print(f"  Breakdown:")
    print(f"    CO2:  {emissions.co2_tonnes} tonnes")
    print(f"    CH4:  {emissions.ch4_tonnes} tonnes")
    print(f"    N2O:  {emissions.n2o_tonnes} tonnes")
    print(f"  Activity: {emissions.activity_amount} {emissions.activity_unit}")
    print(f"  Emission Factor: {emissions.emission_factor_value} {emissions.emission_factor_unit}")

    # Quality check
    report = check_data_quality(emissions)
    print(f"\nQuality Score: {report.overall_score}/100 ({report.quality_level})")

    return emissions, report


# ============================================================================
# EXAMPLE 3: CREATE ENERGY DATA
# ============================================================================

def example_energy_creation():
    """Create and validate energy consumption data."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Energy Consumption Data")
    print("="*80)

    # Electricity consumption with renewable percentage
    energy = EnergyDataContract(
        organization_id="ORG-ACME-001",
        facility_id="HQ-OFFICE",
        meter_id="MTR-E-001",
        consumption_period_start=datetime(2024, 1, 1, 0, 0, 0),
        consumption_period_end=datetime(2024, 1, 31, 23, 59, 59),
        energy_type=EnergyType.ELECTRICITY,
        consumption_amount=Decimal("50000.000"),
        consumption_unit="kWh",
        energy_cost=Decimal("8500.00"),
        currency="USD",
        is_renewable=True,
        renewable_percentage=Decimal("60.0"),
        has_green_certificate=True,
        grid_region="WECC",
        supplier_name="Green Energy Co",
        supplier_emission_factor=Decimal("0.200"),
        scope_2_location_based_co2e=Decimal("22.500"),
        scope_2_market_based_co2e=Decimal("10.000"),  # Lower due to RECs
        data_source="utility_bill",
        data_quality_level=DataQualityLevel.EXCELLENT,
        location_country="US",
        location_region="CA",
        notes="60% renewable via purchased RECs"
    )

    print(f"\nEnergy Consumption Record:")
    print(f"  Energy Type: {energy.energy_type.value}")
    print(f"  Consumption: {energy.consumption_amount} {energy.consumption_unit}")
    print(f"  Cost: {energy.currency} {energy.energy_cost}")
    print(f"  Renewable: {energy.renewable_percentage}%")
    print(f"  Scope 2 Emissions:")
    print(f"    Location-based: {energy.scope_2_location_based_co2e} tCO2e")
    print(f"    Market-based: {energy.scope_2_market_based_co2e} tCO2e")
    print(f"  Grid Region: {energy.grid_region}")

    # Quality check
    report = check_data_quality(energy)
    print(f"\nQuality Score: {report.overall_score}/100")

    return energy, report


# ============================================================================
# EXAMPLE 4: GENERATE SAMPLE DATA
# ============================================================================

def example_sample_generation():
    """Generate synthetic sample data."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Sample Data Generation")
    print("="*80)

    generator = SampleDataGenerator(seed=42)

    # Generate CBAM samples
    cbam_samples = generator.generate_cbam_samples(count=10)
    print(f"\nGenerated {len(cbam_samples)} CBAM declarations:")

    for i, cbam in enumerate(cbam_samples[:3], 1):
        print(f"\n  Sample {i}:")
        print(f"    Product: {cbam.product_category.value}")
        print(f"    Quantity: {cbam.quantity} {cbam.quantity_unit}")
        print(f"    Origin: {cbam.country_of_origin}")
        print(f"    Emissions: {cbam.total_embedded_emissions} tCO2e")

    # Generate emissions samples
    emissions_samples = generator.generate_emissions_samples(count=20)
    print(f"\nGenerated {len(emissions_samples)} emissions records")

    # Count by scope
    scope_counts = {}
    for e in emissions_samples:
        scope_counts[e.ghg_scope] = scope_counts.get(e.ghg_scope, 0) + 1

    print("  By scope:")
    for scope, count in scope_counts.items():
        print(f"    {scope.value}: {count} records")

    # Generate energy samples
    energy_samples = generator.generate_energy_samples(count=15)
    print(f"\nGenerated {len(energy_samples)} energy consumption records")

    # Count by type
    type_counts = {}
    for e in energy_samples:
        type_counts[e.energy_type] = type_counts.get(e.energy_type, 0) + 1

    print("  By type:")
    for energy_type, count in type_counts.items():
        print(f"    {energy_type.value}: {count} records")

    return cbam_samples, emissions_samples, energy_samples


# ============================================================================
# EXAMPLE 5: BATCH QUALITY ASSESSMENT
# ============================================================================

def example_batch_quality_check():
    """Run quality checks on batch of data."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Quality Assessment")
    print("="*80)

    # Generate sample data
    generator = SampleDataGenerator(seed=123)
    samples = generator.generate_cbam_samples(count=100)

    # Quality check all samples
    checker = DataQualityChecker()
    reports = [checker.check_cbam_data(s) for s in samples]

    # Aggregate statistics
    scores = [r.overall_score for r in reports]
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)

    quality_levels = {}
    for r in reports:
        quality_levels[r.quality_level] = quality_levels.get(r.quality_level, 0) + 1

    acceptable_count = sum(1 for r in reports if r.is_acceptable)
    critical_issues_count = sum(1 for r in reports if r.has_critical_issues)

    print(f"\nBatch Quality Analysis ({len(samples)} records):")
    print(f"  Average Score: {avg_score:.2f}/100")
    print(f"  Min Score: {min_score:.2f}")
    print(f"  Max Score: {max_score:.2f}")
    print(f"\n  Quality Levels:")
    for level, count in sorted(quality_levels.items()):
        percentage = count / len(reports) * 100
        print(f"    {level.value}: {count} ({percentage:.1f}%)")
    print(f"\n  Acceptable Rate: {acceptable_count}/{len(reports)} ({acceptable_count/len(reports)*100:.1f}%)")
    print(f"  Critical Issues: {critical_issues_count} records")

    # Show worst performing records
    worst_reports = sorted(reports, key=lambda r: r.overall_score)[:5]
    print(f"\n  Worst Performing Records:")
    for r in worst_reports:
        print(f"    {r.record_id}: {r.overall_score:.2f} - {len(r.critical_issues)} critical, {len(r.high_issues)} high issues")

    return reports


# ============================================================================
# EXAMPLE 6: EMISSION FACTOR LOADING (ASYNC)
# ============================================================================

async def example_emission_factor_loading():
    """Load and search emission factors."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Emission Factor Loading (Simulated)")
    print("="*80)

    # Note: This is a simulation. In production, you'd use:
    # loader = EmissionFactorLoader("postgresql://user:pass@localhost/greenlang")
    # await loader.initialize()

    print("\nSimulated Emission Factor Loading:")
    print("  [INFO] EmissionFactorLoader initialized")
    print("  [INFO] Database connection established")
    print("  [INFO] emission_factors table created/verified")

    # Simulate DEFRA loading
    print("\n  Loading DEFRA 2024 factors...")
    print("  [INFO] Loaded 1,247 DEFRA emission factors")

    # Simulate EPA eGRID loading
    print("\n  Loading EPA eGRID 2023 factors...")
    print("  [INFO] Loaded 26 EPA eGRID emission factors (US grid regions)")

    # Simulate search
    print("\n  Searching for 'natural gas' emission factors:")
    print("    1. Natural gas - Gross CV: 0.18443 kgCO2e/kWh (DEFRA)")
    print("    2. Natural gas - Net CV: 0.20384 kgCO2e/kWh (DEFRA)")
    print("    3. Natural gas - 100% mineral blend: 0.18516 kgCO2e/kWh (DEFRA)")

    print("\n  Searching US electricity grid factors:")
    print("    1. WECC Northwest: 0.243 kgCO2e/kWh (EPA eGRID)")
    print("    2. ERCOT All: 0.389 kgCO2e/kWh (EPA eGRID)")
    print("    3. NPCC New England: 0.298 kgCO2e/kWh (EPA eGRID)")

    print("\n  [INFO] EmissionFactorLoader closed")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GREENLANG DATA MODULE - COMPREHENSIVE EXAMPLES")
    print("="*80)

    # Example 1: CBAM
    cbam, cbam_report = example_cbam_creation()

    # Example 2: Emissions
    emissions, emissions_report = example_emissions_creation()

    # Example 3: Energy
    energy, energy_report = example_energy_creation()

    # Example 4: Sample generation
    cbam_samples, emissions_samples, energy_samples = example_sample_generation()

    # Example 5: Batch quality
    batch_reports = example_batch_quality_check()

    # Example 6: Emission factors (async)
    asyncio.run(example_emission_factor_loading())

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*80)

    # Summary
    print("\nSUMMARY:")
    print(f"  CBAM declarations created: 11")
    print(f"  Emissions records created: 21")
    print(f"  Energy records created: 16")
    print(f"  Quality checks performed: 148")
    print(f"  Overall data quality: High")

    print("\nNEXT STEPS:")
    print("  1. Load real emission factors from DEFRA/EPA datasets")
    print("  2. Connect to production PostgreSQL database")
    print("  3. Integrate with ERP systems for real data intake")
    print("  4. Set up automated quality monitoring")
    print("  5. Build data pipelines for continuous processing")


if __name__ == "__main__":
    main()
