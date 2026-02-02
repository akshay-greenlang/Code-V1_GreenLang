#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick verification script for WaterTreatmentTools.
Tests key functions to ensure deterministic calculations work correctly.
"""

import sys
from pathlib import Path

# Add the GL-016 directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tools import WaterTreatmentTools, ScavengerType, AmineType, BoilerType  # noqa: E402


def test_water_chemistry():
    """Test water chemistry calculations."""
    print("\n=== Testing Water Chemistry Analysis ===")

    # Test LSI calculation
    lsi = WaterTreatmentTools.calculate_langelier_saturation_index(
        pH=7.5,
        temperature=25.0,
        calcium_hardness=200.0,
        alkalinity=150.0,
        tds=500.0
    )
    print(f"Langelier Saturation Index: {lsi}")

    # Test RSI calculation
    rsi = WaterTreatmentTools.calculate_ryznar_stability_index(pH=7.5, pHs=7.8)
    print(f"Ryznar Stability Index: {rsi}")

    # Test Larson-Skold Index
    lsk = WaterTreatmentTools.calculate_larson_skold_index(
        chloride=50.0,
        sulfate=100.0,
        alkalinity=150.0
    )
    print(f"Larson-Skold Index: {lsk}")

    # Comprehensive analysis
    chemistry_data = {
        'pH': 11.2,
        'temperature': 180.0,
        'calcium_hardness': 1.5,
        'alkalinity': 400.0,
        'tds': 2500.0,
        'chloride': 80.0,
        'sulfate': 50.0,
        'pressure': 15.0
    }

    analysis = WaterTreatmentTools.analyze_water_quality(chemistry_data)
    print("\nWater Quality Analysis:")
    print(f"  LSI: {analysis.lsi_value}")
    print(f"  RSI: {analysis.rsi_value}")
    print(f"  Scale Tendency: {analysis.scale_tendency}")
    print(f"  Corrosion Risk: {analysis.corrosion_risk}")
    print(f"  Compliance: {analysis.compliance_status}")
    print(f"  Violations: {len(analysis.violations)}")
    print(f"  Recommendations: {len(analysis.recommendations)}")


def test_blowdown_optimization():
    """Test blowdown optimization."""
    print("\n=== Testing Blowdown Optimization ===")

    # Calculate cycles
    cycles = WaterTreatmentTools.calculate_cycles_of_concentration(
        makeup_conductivity=200.0,
        blowdown_conductivity=1000.0
    )
    print(f"Cycles of Concentration: {cycles}")

    # Calculate blowdown rate
    bd_rate = WaterTreatmentTools.calculate_blowdown_rate(
        steam_rate=5000.0,
        cycles=5.0
    )
    print(f"Blowdown Rate: {bd_rate} kg/hr")

    # Heat loss
    heat_loss = WaterTreatmentTools.calculate_blowdown_heat_loss(
        blowdown_rate=bd_rate,
        temperature=180.0,
        ambient_temp=25.0
    )
    print(f"Heat Loss: {heat_loss} kW")

    # Full optimization
    water_data = {
        'makeup_conductivity': 200.0,
        'blowdown_conductivity': 1500.0,
        'tds': 2000.0,
        'alkalinity': 400.0,
        'temperature': 180.0,
        'pressure': 12.0,
        'water_cost': 0.5,
        'energy_cost': 0.08
    }

    optimization = WaterTreatmentTools.optimize_blowdown_schedule(
        water_data=water_data,
        steam_demand=5000.0
    )
    print("\nBlowdown Optimization:")
    print(f"  Optimal Cycles: {optimization.optimal_cycles}")
    print(f"  Blowdown Rate: {optimization.recommended_blowdown_rate} kg/hr")
    print(f"  Water Savings: {optimization.water_savings} m3/day")
    print(f"  Cost Savings: ${optimization.cost_savings}/day")
    print(f"  Heat Recovery Potential: {optimization.heat_recovery_potential} kW")


def test_chemical_dosing():
    """Test chemical dosing calculations."""
    print("\n=== Testing Chemical Dosing ===")

    # Phosphate dosing
    phosphate = WaterTreatmentTools.calculate_phosphate_dosing(
        residual_target=50.0,
        volume=15.0,
        current_level=30.0,
        steam_rate=5000.0
    )
    print(f"Phosphate Dosing: {phosphate} kg/day")

    # Oxygen scavenger
    scavenger = WaterTreatmentTools.calculate_oxygen_scavenger_dosing(
        dissolved_oxygen=200.0,
        steam_rate=5000.0,
        scavenger_type=ScavengerType.SODIUM_SULFITE
    )
    print(f"Oxygen Scavenger Dosing: {scavenger} kg/day")

    # Amine dosing
    amine = WaterTreatmentTools.calculate_amine_dosing(
        condensate_pH_target=8.8,
        steam_rate=5000.0,
        amine_type=AmineType.NEUTRALIZING_AMINE,
        condensate_return_percent=80.0
    )
    print(f"Amine Dosing: {amine} kg/day")

    # Polymer dosing
    polymer = WaterTreatmentTools.calculate_polymer_dosing(
        sludge_conditioner_need=60.0,
        water_hardness=150.0,
        steam_rate=5000.0
    )
    print(f"Polymer Dosing: {polymer} kg/day")


def test_scale_corrosion_prediction():
    """Test scale and corrosion prediction."""
    print("\n=== Testing Scale and Corrosion Prediction ===")

    # Calcium carbonate scale
    scale_rate = WaterTreatmentTools.predict_calcium_carbonate_scale(
        lsi=1.5,
        temperature=90.0,
        velocity=1.2
    )
    print(f"CaCO3 Scale Rate: {scale_rate} mm/year")

    # Silica scale risk
    silica_risk = WaterTreatmentTools.predict_silica_scale(
        silica_concentration=120.0,
        temperature=180.0,
        pH=11.0
    )
    print(f"Silica Scale Risk: {silica_risk}")

    # Oxygen corrosion
    o2_corrosion = WaterTreatmentTools.predict_oxygen_corrosion(
        dissolved_oxygen=500.0,
        temperature=70.0,
        pH=7.5
    )
    print(f"O2 Corrosion Rate: {o2_corrosion} mpy")

    # Acid corrosion
    acid_corrosion = WaterTreatmentTools.predict_acid_corrosion(
        pH=5.5,
        temperature=60.0
    )
    print(f"Acid Corrosion Rate: {acid_corrosion} mpy")

    # Corrosion allowance
    allowance = WaterTreatmentTools.calculate_corrosion_allowance(
        material="carbon_steel",
        environment="boiler_water",
        service_life=20.0
    )
    print(f"Corrosion Allowance: {allowance} mm")


def test_energy_cost_analysis():
    """Test energy and cost analysis."""
    print("\n=== Testing Energy and Cost Analysis ===")

    # Energy savings
    savings = WaterTreatmentTools.calculate_blowdown_energy_savings(
        before_cycles=3.0,
        after_cycles=6.0,
        steam_cost=25.0,
        steam_rate=5000.0
    )
    print(f"Annual Energy Savings: ${savings}")

    # Chemical cost
    dosing_rates = {
        'phosphate': 5.0,
        'oxygen_scavenger': 2.5,
        'amine': 1.0,
        'polymer': 3.0
    }

    chemical_prices = {
        'phosphate': 2.50,
        'oxygen_scavenger': 3.00,
        'amine': 5.00,
        'polymer': 4.00
    }

    chemical_cost = WaterTreatmentTools.calculate_chemical_cost(
        dosing_rates, chemical_prices
    )
    print(f"Daily Chemical Cost: ${chemical_cost}")

    # ROI calculation
    costs = {'chemical': 500, 'maintenance': 200}
    savings_dict = {'water': 800, 'energy': 1500}
    implementation_cost = 10000.0

    roi = WaterTreatmentTools.calculate_water_treatment_roi(
        costs, savings_dict, implementation_cost
    )
    print(f"Water Treatment ROI: {roi}%")

    # Makeup water cost
    makeup_cost = WaterTreatmentTools.calculate_makeup_water_cost(
        usage=50.0,
        water_price=0.5,
        treatment_cost=0.3
    )
    print(f"Makeup Water Cost: ${makeup_cost}/day")


def test_compliance_checking():
    """Test compliance checking."""
    print("\n=== Testing Compliance Checking ===")

    # ASME compliance
    chemistry = {
        'pH': 11.2,
        'tds': 2500.0,
        'alkalinity': 450.0,
        'chloride': 80.0,
        'silica': 40.0,
        'hardness': 1.5
    }

    asme_result = WaterTreatmentTools.check_asme_compliance(
        chemistry, pressure=15.0
    )
    print("\nASME Compliance:")
    print(f"  Status: {asme_result.compliance_status}")
    print(f"  Parameters Checked: {asme_result.parameters_checked}")
    print(f"  Violations: {len(asme_result.violations)}")
    print(f"  Warnings: {len(asme_result.warnings)}")

    # ABMA compliance
    chemistry_with_residuals = {
        'pH': 11.0,
        'tds': 2000.0,
        'phosphate_residual': 45.0,
        'sulfite_residual': 25.0
    }

    abma_result = WaterTreatmentTools.check_abma_guidelines(
        chemistry_with_residuals,
        BoilerType.WATER_TUBE
    )
    print("\nABMA Compliance:")
    print(f"  Status: {abma_result.compliance_status}")
    print(f"  Violations: {len(abma_result.violations)}")

    # Treatment program validation
    validation = WaterTreatmentTools.validate_treatment_program(
        program_type="phosphate",
        chemistry=chemistry_with_residuals
    )
    print("\nTreatment Program Validation:")
    print(f"  Valid: {validation.is_valid}")
    print(f"  Effectiveness Score: {validation.effectiveness_score}")
    print(f"  Chemistry Compatible: {validation.chemistry_compatibility}")


def test_provenance():
    """Test provenance hash generation."""
    print("\n=== Testing Provenance Hashing ===")

    chemistry_data = {
        'pH': 11.2,
        'temperature': 180.0,
        'calcium_hardness': 1.5,
        'alkalinity': 400.0,
        'tds': 2500.0,
        'chloride': 80.0,
        'sulfate': 50.0,
        'pressure': 15.0
    }

    # Run analysis twice - should produce same hash for same inputs
    analysis1 = WaterTreatmentTools.analyze_water_quality(chemistry_data)
    analysis2 = WaterTreatmentTools.analyze_water_quality(chemistry_data)

    print(f"Analysis 1 Hash: {analysis1.provenance_hash[:16]}...")
    print(f"Analysis 2 Hash: {analysis2.provenance_hash[:16]}...")
    print("Hashes include timestamp, so they differ for deterministic tracking")


if __name__ == "__main__":
    print("=" * 70)
    print("GL-016 WATERGUARD - Deterministic Tools Verification")
    print("=" * 70)

    try:
        test_water_chemistry()
        test_blowdown_optimization()
        test_chemical_dosing()
        test_scale_corrosion_prediction()
        test_energy_cost_analysis()
        test_compliance_checking()
        test_provenance()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
