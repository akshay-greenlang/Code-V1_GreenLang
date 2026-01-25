# -*- coding: utf-8 -*-
"""
EPA Part 98 Subpart C Integration Examples

Complete working examples demonstrating:
- Single source calculations
- Multi-source facility reporting
- Tier 1, 2, and 3 methodologies
- Annual report generation
- Batch processing
- Error handling

Run any example:
    python -m greenlang.compliance.epa.part98_examples
"""

import json
from greenlang.compliance.epa.part98_ghg import (
    Part98Reporter,
    Part98Config,
    FuelCombustionData,
    FuelType,
    TierLevel,
)


def example_1_single_source_tier1():
    """
    Example 1: Single natural gas source using Tier 1 (default factors)

    This is the most common reporting scenario for smaller facilities
    or when fuel-specific data is unavailable.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Source - Tier 1 (Default Factors)")
    print("="*70)

    # Configure reporter
    config = Part98Config(
        facility_id="FAC-001-NG",
        epa_ghgrp_id="111222333",
        facility_name="Simple Gas Boiler Plant",
    )
    reporter = Part98Reporter(config)

    # Define single fuel source
    fuel_data = FuelCombustionData(
        fuel_type=FuelType.NATURAL_GAS,
        heat_input_mmbtu=50000.0,  # Annual heat input
        facility_id="FAC-001-NG",
        process_id="BOILER-001",
        equipment_type="Natural Gas Steam Boiler",
        reporting_year=2024,
    )

    # Calculate
    result = reporter.calculate_subpart_c(fuel_data, tier=TierLevel.TIER1)

    # Display results
    print(f"\nFacility: {config.facility_name}")
    print(f"Process: {fuel_data.process_id} ({fuel_data.equipment_type})")
    print(f"\nEmissions Calculation:")
    print(f"  Fuel Type: {fuel_data.fuel_type.value}")
    print(f"  Heat Input: {fuel_data.heat_input_mmbtu:,.0f} MMBtu")
    print(f"  CO2 Factor: {result.co2_calculation.emission_factor_used:.2f} kg CO2/MMBtu")
    print(f"\nGHG Emissions:")
    print(f"  CO2: {result.total_co2_metric_tons:,.2f} MT")
    print(f"  CH4: {result.total_ch4_metric_tons:.6f} MT")
    print(f"  N2O: {result.total_n2o_metric_tons:.6f} MT")
    print(f"  CO2e (AR5): {result.total_co2e_metric_tons:,.2f} MT")
    print(f"\nReporting Status:")
    print(f"  Exceeds 25,000 MT CO2e: {result.exceeds_threshold}")
    print(f"  GHGRP Reporting Required: {result.requires_reporting}")
    print(f"  Validation: {result.validation_status}")


def example_2_multi_source_facility():
    """
    Example 2: Multi-source facility with different fuel types

    Industrial facility with multiple combustion equipment types.
    Most realistic reporting scenario.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Source Facility (Natural Gas + Coal + Oil)")
    print("="*70)

    config = Part98Config(
        facility_id="FAC-002-MULTI",
        epa_ghgrp_id="444555666",
        facility_name="Industrial Manufacturing Complex",
        facility_address="100 Factory Road, Industrial City, USA",
    )
    reporter = Part98Reporter(config)

    # Define multiple fuel sources
    sources = [
        FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=80000.0,
            facility_id="FAC-002-MULTI",
            process_id="BOILER-001",
            equipment_type="Natural Gas Steam Boiler",
            reporting_year=2024,
        ),
        FuelCombustionData(
            fuel_type=FuelType.COAL_BITUMINOUS,
            heat_input_mmbtu=120000.0,
            facility_id="FAC-002-MULTI",
            process_id="COAL-GEN-001",
            equipment_type="Coal-fired Power Generation",
            reporting_year=2024,
        ),
        FuelCombustionData(
            fuel_type=FuelType.FUEL_OIL_NO2,
            heat_input_mmbtu=30000.0,
            facility_id="FAC-002-MULTI",
            process_id="FURNACE-001",
            equipment_type="Oil-fired Furnace (Backup)",
            reporting_year=2024,
        ),
    ]

    # Calculate each source
    results = [reporter.calculate_subpart_c(fuel) for fuel in sources]

    # Generate annual report
    annual_report = reporter.generate_annual_report(results)

    # Display summary
    print(f"\nFacility: {config.facility_name}")
    print(f"Reporting Year: {annual_report['reporting_year']}")
    print(f"Number of Sources: {annual_report['total_records']}")

    print(f"\nSource Categories:")
    for source in annual_report["source_categories"]:
        print(f"  {source['process_id']} ({source['fuel_type']})")
        print(f"    CO2: {source['co2_metric_tons']:,.2f} MT")
        print(f"    Tier: {source['calculation_tier']}")

    print(f"\nFacility-Level Emissions Summary:")
    summary = annual_report["emissions_summary"]
    print(f"  Total CO2: {summary['total_co2_metric_tons']:,.2f} MT")
    print(f"  Total CH4: {summary['total_ch4_metric_tons']:.6f} MT")
    print(f"  Total N2O: {summary['total_n2o_metric_tons']:.6f} MT")
    print(f"  Total CO2e: {summary['total_co2e_metric_tons']:,.2f} MT")

    print(f"\nReporting Status:")
    print(f"  Exceeds Threshold: {annual_report['exceeds_threshold']}")
    print(f"  GHGRP Required: {annual_report['requires_reporting']}")
    print(f"  Status: {annual_report['reporting_status']}")


def example_3_tier2_calculation():
    """
    Example 3: Tier 2 calculation using fuel-specific data

    More precise calculation when higher heating value and carbon
    content are measured or provided by fuel supplier.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Tier 2 Calculation (Fuel-Specific Data)")
    print("="*70)

    config = Part98Config(
        facility_id="FAC-003-COAL",
        epa_ghgrp_id="777888999",
        facility_name="Coal Utility Power Plant",
    )
    reporter = Part98Reporter(config)

    # Coal with measured properties
    fuel_data = FuelCombustionData(
        fuel_type=FuelType.COAL_BITUMINOUS,
        heat_input_mmbtu=500000.0,  # Large facility
        fuel_quantity=50000000.0,  # 50M kg coal
        higher_heating_value=12500.0,  # BTU/kg
        carbon_content=76.5,  # % by weight
        facility_id="FAC-003-COAL",
        process_id="COAL-GEN-001",
        equipment_type="Coal-fired Power Plant",
        reporting_year=2024,
    )

    # Calculate using Tier 2
    result = reporter.calculate_subpart_c(fuel_data, tier=TierLevel.TIER2)

    print(f"\nFacility: {config.facility_name}")
    print(f"\nFuel Properties (Tier 2):")
    print(f"  Fuel Type: {fuel_data.fuel_type.value}")
    print(f"  Quantity: {fuel_data.fuel_quantity:,.0f} kg")
    print(f"  Higher Heating Value: {fuel_data.higher_heating_value:,.0f} BTU/kg")
    print(f"  Carbon Content: {fuel_data.carbon_content:.1f}%")
    print(f"  Calculated Heat Input: {result.co2_calculation.heat_input_mmbtu:,.0f} MMBtu")

    print(f"\nEmissions (Tier 2):")
    print(f"  CO2: {result.total_co2_metric_tons:,.2f} MT")
    print(f"  CH4: {result.total_ch4_metric_tons:.6f} MT")
    print(f"  N2O: {result.total_n2o_metric_tons:.6f} MT")
    print(f"  CO2e: {result.total_co2e_metric_tons:,.2f} MT")

    print(f"\nReporting Status:")
    print(f"  Exceeds Threshold: {result.exceeds_threshold}")
    print(f"  GHGRP Reporting Required: {result.requires_reporting}")


def example_4_batch_processing():
    """
    Example 4: Batch processing of multiple facilities

    Processing data for a company with multiple locations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing (Multiple Facilities)")
    print("="*70)

    # Company with 3 locations
    facilities = [
        ("FAC-NYC-001", "New York Plant"),
        ("FAC-CHI-002", "Chicago Plant"),
        ("FAC-LAX-003", "Los Angeles Plant"),
    ]

    all_reports = []

    for fac_id, fac_name in facilities:
        config = Part98Config(
            facility_id=fac_id,
            facility_name=fac_name,
        )
        reporter = Part98Reporter(config)

        # Each facility has different heat input
        heat_inputs = {"New York": 60000, "Chicago": 80000, "Los Angeles": 45000}
        heat_input = heat_inputs.get(fac_name.split()[0], 50000)

        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=float(heat_input),
            facility_id=fac_id,
            process_id="BOILER-001",
            reporting_year=2024,
        )

        result = reporter.calculate_subpart_c(fuel_data)
        report = {
            "facility_id": fac_id,
            "facility_name": fac_name,
            "co2_mt": round(result.total_co2_metric_tons, 2),
            "co2e_mt": round(result.total_co2e_metric_tons, 2),
            "requires_reporting": result.requires_reporting,
        }
        all_reports.append(report)

    # Summary
    total_co2 = sum(r["co2_mt"] for r in all_reports)
    total_co2e = sum(r["co2e_mt"] for r in all_reports)

    print(f"\nCompany Emissions Summary (3 Locations):")
    print(f"\n{'-'*50}")
    for report in all_reports:
        status = "REPORT" if report["requires_reporting"] else "NO REPORT"
        print(f"{report['facility_name']:20} CO2: {report['co2_mt']:>8,.0f} MT [{status}]")
    print(f"{'-'*50}")
    print(f"{'TOTAL':20} CO2e: {total_co2e:>8,.0f} MT")


def example_5_co2e_analysis():
    """
    Example 5: Detailed CO2e analysis with GWP breakdown

    Shows how CH4 and N2O contribute to total emissions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: CO2e Analysis with GWP Breakdown")
    print("="*70)

    config = Part98Config(facility_id="FAC-005")
    reporter = Part98Reporter(config)

    fuel_data = FuelCombustionData(
        fuel_type=FuelType.NATURAL_GAS,
        heat_input_mmbtu=100000.0,
        facility_id="FAC-005",
        reporting_year=2024,
    )

    result = reporter.calculate_subpart_c(fuel_data)

    # Calculate CO2e contributions
    gwp_ch4 = 28
    gwp_n2o = 265

    co2_contribution = result.total_co2_metric_tons
    ch4_contribution = result.total_ch4_metric_tons * gwp_ch4
    n2o_contribution = result.total_n2o_metric_tons * gwp_n2o
    total_co2e = result.total_co2e_metric_tons

    # Percentages
    co2_pct = (co2_contribution / total_co2e * 100) if total_co2e > 0 else 0
    ch4_pct = (ch4_contribution / total_co2e * 100) if total_co2e > 0 else 0
    n2o_pct = (n2o_contribution / total_co2e * 100) if total_co2e > 0 else 0

    print(f"\nGas Emissions (Absolute):")
    print(f"  CO2:  {result.total_co2_metric_tons:>10,.2f} MT")
    print(f"  CH4:  {result.total_ch4_metric_tons:>10,.6f} MT")
    print(f"  N2O:  {result.total_n2o_metric_tons:>10,.6f} MT")

    print(f"\nCO2e Contributions (with AR5 GWP):")
    print(f"  CO2:  {co2_contribution:>10,.2f} MT CO2e ({co2_pct:>5.1f}%)")
    print(f"  CH4:  {ch4_contribution:>10,.2f} MT CO2e ({ch4_pct:>5.1f}%) [GWP={gwp_ch4}]")
    print(f"  N2O:  {n2o_contribution:>10,.2f} MT CO2e ({n2o_pct:>5.1f}%) [GWP={gwp_n2o}]")
    print(f"  {'-'*40}")
    print(f"  TOTAL: {total_co2e:>10,.2f} MT CO2e")


def example_6_validation_and_errors():
    """
    Example 6: Input validation and error handling

    Demonstrates proper error handling for invalid inputs.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Validation and Error Handling")
    print("="*70)

    config = Part98Config(facility_id="FAC-006")
    reporter = Part98Reporter(config)

    # Test Case 1: Valid input
    print("\nTest 1: Valid Input")
    try:
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=5000.0,
            facility_id="FAC-006",
            reporting_year=2024,
        )
        result = reporter.calculate_subpart_c(fuel_data)
        print(f"  Status: PASS")
        print(f"  Result: {result.total_co2_metric_tons:.2f} MT CO2")
    except Exception as e:
        print(f"  Status: FAIL - {e}")

    # Test Case 2: Invalid heat input (negative)
    print("\nTest 2: Negative Heat Input")
    try:
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=-1000.0,  # Invalid
            facility_id="FAC-006",
            reporting_year=2024,
        )
        result = reporter.calculate_subpart_c(fuel_data)
        print(f"  Status: PASS (unexpected)")
    except ValueError as e:
        print(f"  Status: Expected error caught")
        print(f"  Error: {e}")

    # Test Case 3: Invalid carbon content
    print("\nTest 3: Invalid Carbon Content (>100%)")
    try:
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.COAL_BITUMINOUS,
            heat_input_mmbtu=5000.0,
            carbon_content=150.0,  # Invalid
            facility_id="FAC-006",
            reporting_year=2024,
        )
        print(f"  Status: PASS (unexpected)")
    except ValueError as e:
        print(f"  Status: Expected error caught")
        print(f"  Error: {e}")

    # Test Case 4: Tier 2 without required data
    print("\nTest 4: Tier 2 Missing Carbon Content")
    fuel_data = FuelCombustionData(
        fuel_type=FuelType.COAL_BITUMINOUS,
        heat_input_mmbtu=5000.0,
        higher_heating_value=12000.0,
        # Missing carbon_content
        facility_id="FAC-006",
        reporting_year=2024,
    )
    result = reporter.calculate_subpart_c(fuel_data, tier=TierLevel.TIER2)
    print(f"  Status: {result.validation_status}")
    if result.validation_errors:
        print(f"  Errors: {result.validation_errors}")


if __name__ == "__main__":
    """Run all examples"""
    example_1_single_source_tier1()
    example_2_multi_source_facility()
    example_3_tier2_calculation()
    example_4_batch_processing()
    example_5_co2e_analysis()
    example_6_validation_and_errors()

    print("\n" + "="*70)
    print("All Examples Completed Successfully")
    print("="*70 + "\n")
