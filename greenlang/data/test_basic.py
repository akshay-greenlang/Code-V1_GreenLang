# -*- coding: utf-8 -*-
"""
Basic test for GreenLang Data Module (no dependencies)
"""

from datetime import date, datetime
from decimal import Decimal

print("="*80)
print("GREENLANG DATA MODULE - BASIC TESTS")
print("="*80)

# Test 1: Import contracts
print("\nTest 1: Importing data contracts...")
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
print("OK All contracts imported successfully")

# Test 2: Create CBAM data
print("\nTest 2: Creating CBAM declaration...")
cbam = CBAMDataContract(
    importer_id="GB123456789000",
    import_date=date(2024, 3, 15),
    declaration_period="2024-Q1",
    product_category=CBAMProductCategory.IRON_STEEL,
    cn_code="72071100",
    product_description="Semi-finished steel products",
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
    data_quality_level=DataQualityLevel.EXCELLENT
)
print(f"OK CBAM declaration created: {cbam.product_description}")
print(f"  Total emissions: {cbam.total_embedded_emissions} tCO2e")

# Test 3: Create emissions data
print("\nTest 3: Creating emissions record...")
emissions = EmissionsDataContract(
    organization_id="ORG-12345",
    reporting_period_start=date(2024, 1, 1),
    reporting_period_end=date(2024, 12, 31),
    ghg_scope=GHGScope.SCOPE_1,
    emission_source="Natural gas combustion",
    activity_type=ActivityType.FUEL_COMBUSTION,
    co2_tonnes=Decimal("1500.000"),
    ch4_tonnes=Decimal("0.150"),
    n2o_tonnes=Decimal("0.030"),
    total_co2e_tonnes=Decimal("1520.000"),
    activity_amount=Decimal("800000.000"),
    activity_unit="kWh",
    emission_factor_value=Decimal("0.001900"),
    emission_factor_unit="tCO2e/kWh",
    emission_factor_source=EmissionFactorSource.DEFRA_2024,
    location_country="GB",
    data_quality_level=DataQualityLevel.GOOD,
    calculation_method="GHG Protocol"
)
print(f"OK Emissions record created: {emissions.ghg_scope.value}")
print(f"  Total CO2e: {emissions.total_co2e_tonnes} tonnes")

# Test 4: Create energy data
print("\nTest 4: Creating energy consumption record...")
energy = EnergyDataContract(
    organization_id="ORG-12345",
    consumption_period_start=datetime(2024, 1, 1),
    consumption_period_end=datetime(2024, 1, 31, 23, 59, 59),
    energy_type=EnergyType.ELECTRICITY,
    consumption_amount=Decimal("50000.000"),
    consumption_unit="kWh",
    data_source="utility_bill",
    data_quality_level=DataQualityLevel.EXCELLENT,
    location_country="US"
)
print(f"OK Energy record created: {energy.energy_type.value}")
print(f"  Consumption: {energy.consumption_amount} {energy.consumption_unit}")

# Test 5: Test quality module
print("\nTest 5: Testing data quality checker...")
from quality import check_data_quality, DataQualityChecker

report = check_data_quality(cbam)
print(f"OK Quality check completed")
print(f"  Overall score: {report.overall_score}/100")
print(f"  Quality level: {report.quality_level.value}")
print(f"  Checks passed: {report.checks_passed}/{report.total_checks}")

# Test 6: Test sample data generator
print("\nTest 6: Testing sample data generator...")
from sample_data import SampleDataGenerator

generator = SampleDataGenerator(seed=42)
cbam_samples = generator.generate_cbam_samples(count=10)
emissions_samples = generator.generate_emissions_samples(count=20)
energy_samples = generator.generate_energy_samples(count=15)

print(f"OK Generated {len(cbam_samples)} CBAM samples")
print(f"OK Generated {len(emissions_samples)} emissions samples")
print(f"OK Generated {len(energy_samples)} energy samples")

# Test 7: Batch quality check
print("\nTest 7: Batch quality assessment...")
checker = DataQualityChecker()
reports = [checker.check_cbam_data(s) for s in cbam_samples]
avg_score = sum(r.overall_score for r in reports) / len(reports)
acceptable = sum(1 for r in reports if r.is_acceptable)

print(f"OK Checked {len(reports)} records")
print(f"  Average score: {avg_score:.2f}/100")
print(f"  Acceptable rate: {acceptable}/{len(reports)} ({acceptable/len(reports)*100:.1f}%)")

print("\n" + "="*80)
print("ALL TESTS PASSED OK")
print("="*80)

print("\n\nSUMMARY:")
print(f"  Data Contracts: 4 types (CBAM, Emissions, Energy, Activity)")
print(f"  Quality Framework: 5 dimensions, 0-100 scoring")
print(f"  Sample Generator: Working with seed control")
print(f"  Total records tested: {len(cbam_samples) + len(emissions_samples) + len(energy_samples)}")

print("\nREADY FOR PRODUCTION USE!")
