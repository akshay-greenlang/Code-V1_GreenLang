# -*- coding: utf-8 -*-
"""
GreenLang Calculation Engine Demonstration

This script demonstrates all major features of the zero-hallucination
calculation engine.

Run with: python examples/calculation_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.calculation import (
    EmissionCalculator,
    CalculationRequest,
    Scope1Calculator,
    Scope2Calculator,
    Scope3Calculator,
    BatchCalculator,
    AuditTrailGenerator,
    CalculationValidator,
    UncertaintyCalculator,
)


def demo_basic_calculation():
    """Demo 1: Basic emission calculation"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Emission Calculation")
    print("="*60)

    calc = EmissionCalculator()

    request = CalculationRequest(
        factor_id='diesel',
        activity_amount=100,
        activity_unit='gallons'
    )

    result = calc.calculate(request)

    print(f"Input: {request.activity_amount} {request.activity_unit} of {request.factor_id}")
    print(f"Emissions: {result.emissions_kg_co2e} kg CO2e")
    print(f"Emissions: {result.emissions_kg_co2e / 1000:.3f} tonnes CO2e")
    print(f"Status: {result.status.value}")
    print(f"Calculation Duration: {result.calculation_duration_ms:.2f} ms")
    print(f"Provenance Hash: {result.provenance_hash}")
    print(f"Provenance Valid: {result.verify_provenance()}")


def demo_scope1_calculations():
    """Demo 2: Scope 1 calculations (direct emissions)"""
    print("\n" + "="*60)
    print("DEMO 2: Scope 1 Direct Emissions")
    print("="*60)

    calc = Scope1Calculator()

    # Stationary combustion
    print("\nStationary Combustion (Natural Gas Boiler):")
    result = calc.calculate_fuel_combustion(
        fuel_type='natural_gas',
        amount=500,
        unit='therms',
        combustion_type='stationary'
    )
    print(f"  500 therms natural gas → {result.calculation_result.emissions_kg_co2e:,.0f} kg CO2e")

    # Mobile combustion
    print("\nMobile Combustion (Company Vehicles):")
    result = calc.calculate_mobile_combustion(
        fuel_type='gasoline',
        amount=200,
        unit='gallons',
        vehicle_type='sedan'
    )
    print(f"  200 gallons gasoline → {result.calculation_result.emissions_kg_co2e:,.0f} kg CO2e")

    # Fugitive emissions
    print("\nFugitive Emissions (Refrigerant Leak):")
    result = calc.calculate_fugitive_emissions(
        refrigerant_type='HFC-134a',
        charge_kg=10,
        annual_leakage_rate=0.15  # 15% annual leakage
    )
    print(f"  1.5 kg HFC-134a leaked → {result.calculation_result.emissions_kg_co2e:,.0f} kg CO2e")
    print(f"  (GWP of HFC-134a: 1,430)")


def demo_scope2_calculations():
    """Demo 3: Scope 2 calculations (indirect energy)"""
    print("\n" + "="*60)
    print("DEMO 3: Scope 2 Indirect Energy Emissions")
    print("="*60)

    calc = Scope2Calculator()

    # Location-based
    print("\nLocation-Based Method (Grid Average):")
    result = calc.calculate_location_based(
        electricity_kwh=10000,
        grid_region='US_WECC_CA',  # California grid
        year=2023
    )
    print(f"  10,000 kWh (California grid) → {result.calculation_result.emissions_kg_co2e:,.0f} kg CO2e")
    print(f"  Grid intensity: {result.calculation_result.emissions_kg_co2e / 10000:.3f} kg CO2e/kWh")

    # Market-based with RECs
    print("\nMarket-Based Method (with 50% Renewable Energy Certificates):")
    result = calc.calculate_market_based(
        electricity_kwh=10000,
        supplier_factor_kg_co2e_per_kwh=0.385,  # US average
        rec_certificates_kwh=5000  # 50% renewable
    )
    print(f"  10,000 kWh with 5,000 kWh RECs → {result.calculation_result.emissions_kg_co2e:,.0f} kg CO2e")
    print(f"  (50% reduction from renewable energy)")


def demo_scope3_calculations():
    """Demo 4: Scope 3 calculations (value chain)"""
    print("\n" + "="*60)
    print("DEMO 4: Scope 3 Value Chain Emissions")
    print("="*60)

    calc = Scope3Calculator()

    # Category 4: Upstream transportation
    print("\nCategory 4: Upstream Transportation")
    result = calc.calculate_category_4_upstream_transport(
        mode='freight_truck_diesel',
        distance_km=500,
        weight_tonnes=10
    )
    print(f"  10 tonnes × 500 km (truck) → {result.calculation_result.emissions_kg_co2e:,.0f} kg CO2e")

    # Category 6: Business travel
    print("\nCategory 6: Business Travel")
    result = calc.calculate_category_6_business_travel(
        mode='air_long_haul',
        distance_km=5000,
        passengers=2,
        cabin_class='economy'
    )
    print(f"  2 passengers × 5,000 km (air) → {result.calculation_result.emissions_kg_co2e:,.0f} kg CO2e")

    # Category 7: Employee commuting
    print("\nCategory 7: Employee Commuting")
    result = calc.calculate_category_7_employee_commuting(
        mode='car_gasoline',
        distance_km_per_day=20,
        employees=100,
        working_days_per_year=220
    )
    print(f"  100 employees × 20 km/day × 220 days → {result.calculation_result.emissions_kg_co2e:,.0f} kg CO2e")


def demo_gas_decomposition():
    """Demo 5: Multi-gas decomposition"""
    print("\n" + "="*60)
    print("DEMO 5: Multi-Gas Decomposition (CO2e → CO2/CH4/N2O)")
    print("="*60)

    from greenlang.calculation.gas_decomposition import MultiGasCalculator

    calc = MultiGasCalculator()

    breakdown = calc.decompose(
        total_co2e_kg=1000,
        fuel_type='natural_gas'
    )

    print(f"\nTotal CO2e: {breakdown.total_co2e_kg} kg")
    print("\nIndividual Gas Contributions:")
    for gas, amount in breakdown.gas_amounts_kg.items():
        co2e_contribution = breakdown.gas_co2e_contributions_kg[gas]
        gwp = breakdown.gwp_values[gas]
        print(f"  {gas}:")
        print(f"    Amount: {amount:.3f} kg")
        print(f"    GWP: {gwp}")
        print(f"    CO2e: {co2e_contribution:.1f} kg ({co2e_contribution/breakdown.total_co2e_kg*100:.1f}%)")


def demo_batch_processing():
    """Demo 6: Batch processing for high performance"""
    print("\n" + "="*60)
    print("DEMO 6: Batch Processing (100 Calculations)")
    print("="*60)

    batch_calc = BatchCalculator()

    # Create 100 requests
    requests = [
        CalculationRequest(
            factor_id=['diesel', 'natural_gas', 'gasoline_motor'][i % 3],
            activity_amount=100 + i,
            activity_unit=['gallons', 'therms', 'gallons'][i % 3]
        )
        for i in range(100)
    ]

    print(f"Calculating {len(requests)} emissions...")

    result = batch_calc.calculate_batch(requests)

    print(f"\nResults:")
    print(f"  Total Emissions: {result.total_emissions_kg_co2e:,.0f} kg CO2e")
    print(f"  Total Emissions: {result.total_emissions_kg_co2e/1000:,.1f} tonnes CO2e")
    print(f"  Successful: {result.successful_count}/{len(requests)}")
    print(f"  Failed: {result.failed_count}/{len(requests)}")
    print(f"  Duration: {result.batch_duration_seconds:.3f} seconds")
    print(f"  Throughput: {len(requests)/result.batch_duration_seconds:.1f} calculations/second")
    print(f"  Average per calculation: {result.average_duration_ms:.2f} ms")


def demo_uncertainty_analysis():
    """Demo 7: Uncertainty quantification"""
    print("\n" + "="*60)
    print("DEMO 7: Uncertainty Quantification (Monte Carlo)")
    print("="*60)

    calc = UncertaintyCalculator()

    result = calc.propagate_uncertainty(
        activity_data=100,              # 100 gallons
        activity_uncertainty_pct=5,     # ±5% measurement uncertainty
        emission_factor=10.21,          # kg CO2e/gallon
        factor_uncertainty_pct=10,      # ±10% factor uncertainty
        n_simulations=10000
    )

    print(f"Point Estimate: {100 * 10.21:.1f} kg CO2e")
    print(f"\nMonte Carlo Analysis (10,000 simulations):")
    print(f"  Mean: {result.mean_kg_co2e:.1f} kg CO2e")
    print(f"  Std Dev: {result.std_kg_co2e:.1f} kg CO2e")
    print(f"  Relative Uncertainty: ±{result.relative_uncertainty_pct:.1f}%")
    print(f"\nConfidence Intervals:")
    print(f"  95% CI: [{result.confidence_interval_95[0]:.1f}, {result.confidence_interval_95[1]:.1f}] kg CO2e")
    print(f"  90% CI: [{result.confidence_interval_90[0]:.1f}, {result.confidence_interval_90[1]:.1f}] kg CO2e")


def demo_audit_trail():
    """Demo 8: Audit trail generation"""
    print("\n" + "="*60)
    print("DEMO 8: Audit Trail Generation")
    print("="*60)

    calc = EmissionCalculator()

    request = CalculationRequest(
        factor_id='diesel',
        activity_amount=100,
        activity_unit='gallons'
    )

    result = calc.calculate(request)

    # Generate audit trail
    trail_gen = AuditTrailGenerator()
    audit_trail = trail_gen.generate(result)

    print(f"Calculation ID: {audit_trail.calculation_id}")
    print(f"Trail Hash: {audit_trail.trail_hash}")
    print(f"Integrity Valid: {audit_trail.verify_integrity()}")
    print(f"\nAudit Trail Components:")
    print(f"  - Input Summary: {len(audit_trail.input_summary)} fields")
    print(f"  - Factor Summary: {len(audit_trail.factor_summary)} fields")
    print(f"  - Calculation Steps: {len(audit_trail.steps)} steps")
    print(f"  - Output Summary: {len(audit_trail.output_summary)} metrics")

    # Export to markdown
    markdown = audit_trail.to_markdown()
    print(f"\nMarkdown Report Length: {len(markdown)} characters")
    print(f"First 300 characters:\n{markdown[:300]}...")


def demo_validation():
    """Demo 9: Input/output validation"""
    print("\n" + "="*60)
    print("DEMO 9: Calculation Validation")
    print("="*60)

    validator = CalculationValidator()

    # Valid request
    print("\nValidating Valid Request:")
    request = CalculationRequest(
        factor_id='diesel',
        activity_amount=100,
        activity_unit='gallons'
    )

    validation = validator.validate_request(request)
    print(f"  Valid: {validation.is_valid}")
    print(f"  Errors: {len(validation.errors)}")
    print(f"  Warnings: {len(validation.warnings)}")
    print(f"  Checks Performed: {len(validation.checks_performed)}")

    # Invalid request (negative value)
    print("\nValidating Invalid Request (negative amount):")
    try:
        request_invalid = CalculationRequest(
            factor_id='diesel',
            activity_amount=-100,
            activity_unit='gallons'
        )
    except ValueError as e:
        print(f"  Caught error: {str(e)}")


def run_all_demos():
    """Run all demonstration examples"""
    print("\n" + "="*70)
    print(" GREENLANG CALCULATION ENGINE DEMONSTRATION")
    print(" Zero-Hallucination, Regulatory-Grade Emission Calculations")
    print("="*70)

    demos = [
        demo_basic_calculation,
        demo_scope1_calculations,
        demo_scope2_calculations,
        demo_scope3_calculations,
        demo_gas_decomposition,
        demo_batch_processing,
        demo_uncertainty_analysis,
        demo_audit_trail,
        demo_validation,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n⚠️  Demo failed: {str(e)}")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nFor full documentation, see: docs/CALCULATION_ENGINE.md")
    print("For performance benchmarks, run: python -m benchmarks.calculation_performance")
    print("For tests, run: pytest tests/calculation/ -v")


if __name__ == '__main__':
    run_all_demos()
