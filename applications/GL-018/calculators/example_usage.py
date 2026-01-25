"""
GL-018 FLUEFLOW - Example Usage

Demonstrates zero-hallucination combustion analysis calculators.

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import with relative path handling
import importlib.util

# Load modules dynamically
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = os.path.dirname(os.path.abspath(__file__))

# Load all calculator modules
combustion_module = load_module("combustion_analyzer", os.path.join(base_path, "combustion_analyzer.py"))
efficiency_module = load_module("efficiency_calculator", os.path.join(base_path, "efficiency_calculator.py"))
afr_module = load_module("air_fuel_ratio_calculator", os.path.join(base_path, "air_fuel_ratio_calculator.py"))
emissions_module = load_module("emissions_calculator", os.path.join(base_path, "emissions_calculator.py"))
provenance_module = load_module("provenance", os.path.join(base_path, "provenance.py"))

# Import classes and functions
CombustionAnalyzer = combustion_module.CombustionAnalyzer
CombustionInput = combustion_module.CombustionInput
FuelType = combustion_module.FuelType
calculate_excess_air_from_O2 = combustion_module.calculate_excess_air_from_O2

EfficiencyCalculator = efficiency_module.EfficiencyCalculator
EfficiencyInput = efficiency_module.EfficiencyInput
calculate_stack_loss_siegert = efficiency_module.calculate_stack_loss_siegert

AirFuelRatioCalculator = afr_module.AirFuelRatioCalculator
AirFuelRatioInput = afr_module.AirFuelRatioInput
calculate_lambda_from_O2 = afr_module.calculate_lambda_from_O2

EmissionsCalculator = emissions_module.EmissionsCalculator
EmissionsInput = emissions_module.EmissionsInput
convert_ppm_to_mg_nm3 = emissions_module.convert_ppm_to_mg_nm3

verify_provenance = provenance_module.verify_provenance
format_provenance_report = provenance_module.format_provenance_report


def example_1_combustion_analysis():
    """Example 1: Complete combustion analysis for natural gas."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Combustion Analysis - Natural Gas")
    print("=" * 70)

    # Initialize calculator
    analyzer = CombustionAnalyzer()

    # Create input
    inputs = CombustionInput(
        O2_pct=3.5,
        CO2_pct=12.0,
        CO_ppm=50.0,
        NOx_ppm=150.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        fuel_type=FuelType.NATURAL_GAS.value
    )

    # Calculate
    result, provenance = analyzer.calculate(inputs)

    # Display results
    print(f"\nExcess Air: {result.excess_air_pct:.1f}%")
    print(f"Lambda (λ): {result.stoichiometric_ratio:.3f}")
    print(f"O2 (dry): {result.O2_dry_pct:.2f}%")
    print(f"CO2 (dry): {result.CO2_dry_pct:.2f}%")
    print(f"CO2 Max (theoretical): {result.CO2_max_pct:.2f}%")
    print(f"Flue Gas Volume: {result.flue_gas_volume_nm3_kg:.3f} Nm³/kg fuel")
    print(f"Combustion Quality Index: {result.combustion_quality_index:.1f}/100")
    print(f"Quality Rating: {result.combustion_quality_rating}")
    print(f"Complete Combustion: {result.is_complete_combustion}")

    # Verify provenance
    is_valid = verify_provenance(provenance)
    print(f"\nProvenance Verified: {is_valid}")
    print(f"Provenance Hash: {provenance.provenance_hash[:32]}...")
    print(f"Calculation ID: {provenance.calculation_id}")


def example_2_efficiency_calculation():
    """Example 2: Combustion efficiency calculation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Combustion Efficiency Analysis")
    print("=" * 70)

    calculator = EfficiencyCalculator()

    inputs = EfficiencyInput(
        fuel_type="Natural Gas",
        fuel_flow_rate_kg_hr=1000.0,
        O2_pct_dry=3.5,
        CO2_pct_dry=12.0,
        CO_ppm=50.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        excess_air_pct=20.0,
        heat_input_mw=10.0,
        heat_output_mw=8.5
    )

    result, provenance = calculator.calculate(inputs)

    print(f"\nCombustion Efficiency: {result.combustion_efficiency_pct:.1f}%")
    print(f"Thermal Efficiency: {result.thermal_efficiency_pct:.1f}%")
    print(f"Stack Loss: {result.stack_loss_pct:.2f}%")
    print(f"Moisture Loss: {result.moisture_loss_pct:.2f}%")
    print(f"Incomplete Combustion Loss: {result.incomplete_combustion_loss_pct:.2f}%")
    print(f"Radiation Loss: {result.radiation_loss_pct:.2f}%")
    print(f"Total Losses: {result.total_losses_pct:.2f}%")
    print(f"Available Heat: {result.available_heat_pct:.2f}%")
    print(f"Efficiency Rating: {result.efficiency_rating}")
    print(f"Heat Loss: {result.heat_loss_mw:.3f} MW")

    print(f"\nProvenance Verified: {verify_provenance(provenance)}")


def example_3_air_fuel_ratio():
    """Example 3: Air-fuel ratio analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Air-Fuel Ratio Analysis")
    print("=" * 70)

    calculator = AirFuelRatioCalculator()

    inputs = AirFuelRatioInput(
        fuel_type="Natural Gas",
        O2_measured_pct=3.5
    )

    result, provenance = calculator.calculate(inputs)

    print(f"\nTheoretical Air: {result.theoretical_air_kg_kg:.2f} kg air/kg fuel")
    print(f"Theoretical Air: {result.theoretical_air_nm3_kg:.2f} Nm³ air/kg fuel")
    print(f"Actual Air: {result.actual_air_kg_kg:.2f} kg air/kg fuel")
    print(f"Excess Air: {result.excess_air_kg_kg:.2f} kg air/kg fuel")
    print(f"Excess Air: {result.excess_air_pct:.1f}%")
    print(f"Lambda (λ): {result.lambda_ratio:.3f}")
    print(f"Fuel/Air Ratio: {result.fuel_air_ratio:.4f}")
    print(f"O2 Actual: {result.O2_actual_pct:.2f}%")
    print(f"Air Requirement Rating: {result.air_requirement_rating}")

    print(f"\nProvenance Verified: {verify_provenance(provenance)}")


def example_4_emissions_analysis():
    """Example 4: Emissions analysis and compliance."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Emissions Analysis and Compliance")
    print("=" * 70)

    calculator = EmissionsCalculator()

    inputs = EmissionsInput(
        NOx_ppm=150.0,
        CO_ppm=50.0,
        SO2_ppm=100.0,
        CO2_pct=12.0,
        O2_pct=3.5,
        flue_gas_temp_c=180.0,
        flue_gas_flow_nm3_hr=50000.0,
        fuel_type="Natural Gas",
        reference_O2_pct=3.0
    )

    result, provenance = calculator.calculate(inputs)

    print(f"\nNOx: {result.NOx_mg_nm3:.1f} mg/Nm³ @ measured O2")
    print(f"NOx: {result.NOx_mg_nm3_corrected:.1f} mg/Nm³ @ 3% O2")
    print(f"NOx Mass Flow: {result.NOx_kg_hr:.3f} kg/hr")
    print(f"NOx Compliance: {result.NOx_compliance_status}")

    print(f"\nCO: {result.CO_mg_nm3:.1f} mg/Nm³ @ measured O2")
    print(f"CO: {result.CO_mg_nm3_corrected:.1f} mg/Nm³ @ 3% O2")
    print(f"CO Mass Flow: {result.CO_kg_hr:.3f} kg/hr")
    print(f"CO Compliance: {result.CO_compliance_status}")

    print(f"\nSO2: {result.SO2_mg_nm3:.1f} mg/Nm³ @ measured O2")
    print(f"SO2: {result.SO2_mg_nm3_corrected:.1f} mg/Nm³ @ 3% O2")
    print(f"SO2 Mass Flow: {result.SO2_kg_hr:.3f} kg/hr")
    print(f"SO2 Compliance: {result.SO2_compliance_status}")

    print(f"\nCO/CO2 Ratio: {result.CO_CO2_ratio:.4f}")
    print(f"NOx Emission Factor: {result.emission_factor_NOx_g_GJ:.1f} g/GJ")

    print(f"\nProvenance Verified: {verify_provenance(provenance)}")


def example_5_standalone_functions():
    """Example 5: Using standalone functions for quick calculations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Standalone Functions (No Provenance Tracking)")
    print("=" * 70)

    # Quick calculations without creating calculator objects
    excess_air = calculate_excess_air_from_O2(3.5)
    print(f"\nExcess Air from 3.5% O2: {excess_air:.1f}%")

    lambda_val = calculate_lambda_from_O2(3.5)
    print(f"Lambda from 3.5% O2: {lambda_val:.3f}")

    stack_loss = calculate_stack_loss_siegert(180.0, 25.0, 12.0)
    print(f"Stack Loss (180°C, 25°C, 12% CO2): {stack_loss:.2f}%")

    nox_mg = convert_ppm_to_mg_nm3(100.0, 46.0)
    print(f"100 ppm NOx = {nox_mg:.1f} mg/Nm³")


def example_6_integration_workflow():
    """Example 6: Complete workflow using all calculators."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Complete Combustion Analysis Workflow")
    print("=" * 70)

    # Step 1: Combustion Analysis
    print("\n--- Step 1: Combustion Analysis ---")
    combustion_calc = CombustionAnalyzer()
    combustion_input = CombustionInput(
        O2_pct=3.5,
        CO2_pct=12.0,
        CO_ppm=50.0,
        NOx_ppm=150.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        fuel_type="Natural Gas"
    )
    combustion_result, _ = combustion_calc.calculate(combustion_input)
    print(f"Excess Air: {combustion_result.excess_air_pct:.1f}%")
    print(f"Quality: {combustion_result.combustion_quality_rating}")

    # Step 2: Efficiency Analysis
    print("\n--- Step 2: Efficiency Analysis ---")
    efficiency_calc = EfficiencyCalculator()
    efficiency_input = EfficiencyInput(
        fuel_type="Natural Gas",
        fuel_flow_rate_kg_hr=1000.0,
        O2_pct_dry=combustion_result.O2_dry_pct,
        CO2_pct_dry=combustion_result.CO2_dry_pct,
        CO_ppm=50.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        excess_air_pct=combustion_result.excess_air_pct,
        heat_input_mw=10.0,
        heat_output_mw=8.5
    )
    efficiency_result, _ = efficiency_calc.calculate(efficiency_input)
    print(f"Combustion Efficiency: {efficiency_result.combustion_efficiency_pct:.1f}%")
    print(f"Stack Loss: {efficiency_result.stack_loss_pct:.2f}%")

    # Step 3: Air-Fuel Ratio
    print("\n--- Step 3: Air-Fuel Ratio Analysis ---")
    afr_calc = AirFuelRatioCalculator()
    afr_input = AirFuelRatioInput(
        fuel_type="Natural Gas",
        O2_measured_pct=combustion_result.O2_dry_pct
    )
    afr_result, _ = afr_calc.calculate(afr_input)
    print(f"Lambda: {afr_result.lambda_ratio:.3f}")
    print(f"Theoretical Air: {afr_result.theoretical_air_kg_kg:.2f} kg/kg")

    # Step 4: Emissions
    print("\n--- Step 4: Emissions Analysis ---")
    emissions_calc = EmissionsCalculator()
    emissions_input = EmissionsInput(
        NOx_ppm=150.0,
        CO_ppm=50.0,
        SO2_ppm=100.0,
        CO2_pct=combustion_result.CO2_dry_pct,
        O2_pct=combustion_result.O2_dry_pct,
        flue_gas_temp_c=180.0,
        flue_gas_flow_nm3_hr=50000.0,
        fuel_type="Natural Gas"
    )
    emissions_result, _ = emissions_calc.calculate(emissions_input)
    print(f"NOx @ 3% O2: {emissions_result.NOx_mg_nm3_corrected:.1f} mg/Nm³")
    print(f"NOx Compliance: {emissions_result.NOx_compliance_status}")

    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print(f"Combustion Quality: {combustion_result.combustion_quality_rating}")
    print(f"Efficiency: {efficiency_result.combustion_efficiency_pct:.1f}%")
    print(f"Air Requirement: {afr_result.air_requirement_rating}")
    print(f"Emissions Compliance: {emissions_result.NOx_compliance_status}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GL-018 FLUEFLOW - Zero-Hallucination Combustion Analysis")
    print("Deterministic, Bit-Perfect, Auditable Calculations")
    print("=" * 70)

    # Run all examples
    example_1_combustion_analysis()
    example_2_efficiency_calculation()
    example_3_air_fuel_ratio()
    example_4_emissions_analysis()
    example_5_standalone_functions()
    example_6_integration_workflow()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("Zero-Hallucination Guarantee: Every calculation is 100% deterministic")
    print("=" * 70 + "\n")
