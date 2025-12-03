#!/usr/bin/env python
"""Run manual tests for the new calculator modules."""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from calculators package
from calculators.economizer_fouling_calculator import (
    FoulingSide,
    FoulingMechanism,
    CleaningMethod,
    TrendModel,
    FoulingMeasurement,
    calculate_fouling_factor_from_u_values,
    calculate_cleanliness_trend,
    predict_fouling_rate,
    calculate_heat_loss_from_fouling,
    calculate_fuel_penalty,
    calculate_carbon_penalty,
    compare_cleaning_effectiveness,
    optimize_cleaning_interval,
)
from calculators.advanced_soot_blower_optimizer import (
    BlowerType,
    BlowingMedium,
    EconomizerZone,
    CleaningPriority,
    WearSeverity,
    SootBlowerConfiguration,
    ZoneFoulingState,
    calculate_optimal_blowing_interval,
    prioritize_cleaning_zones,
    track_media_consumption,
    measure_cleaning_effectiveness,
    analyze_cleaning_roi,
    monitor_erosion_wear,
    optimize_blowing_sequence,
    calculate_soot_blowing_energy_balance,
)

from datetime import datetime, timezone, timedelta
from decimal import Decimal


def test_fouling_calculator():
    """Test the economizer fouling calculator."""
    print("=" * 60)
    print("TESTING: Economizer Fouling Calculator")
    print("=" * 60)

    # Test 1: Basic fouling factor calculation
    print("\n1. Basic Fouling Factor Calculation")
    result = calculate_fouling_factor_from_u_values(
        u_clean=10.0,
        u_current=8.0,
        gas_side_fraction=0.7
    )
    print(f"   Total fouling factor: {result.rf_total}")
    print(f"   Gas side Rf: {result.rf_gas_side}")
    print(f"   Water side Rf: {result.rf_water_side}")
    print(f"   Cleanliness factor: {result.cleanliness_factor}%")
    print(f"   Severity: {result.severity_level}")
    print(f"   Provenance hash: {result.provenance_hash[:32]}...")

    # Test 2: Fuel penalty calculation
    print("\n2. Fuel Penalty Calculation")
    fuel_result = calculate_fuel_penalty(
        heat_loss_mmbtu_hr=2.5,
        boiler_efficiency=0.85,
        fuel_cost_per_mmbtu=4.50,
        operating_hours_per_year=8000
    )
    print(f"   Fuel penalty: {fuel_result.fuel_penalty_mmbtu_hr} MMBtu/hr")
    print(f"   Hourly cost: ${fuel_result.cost_per_hour}/hr")
    print(f"   Annual cost: ${fuel_result.cost_per_year}/year")

    # Test 3: Carbon penalty calculation
    print("\n3. Carbon Penalty Calculation")
    carbon_result = calculate_carbon_penalty(
        fuel_penalty_mmbtu_hr=2.5,
        fuel_type="natural_gas",
        carbon_price_per_tonne=50.00
    )
    print(f"   CO2 emissions: {carbon_result.co2_penalty_kg_hr} kg/hr")
    print(f"   Annual CO2: {carbon_result.co2_penalty_tonnes_yr} tonnes/year")
    print(f"   Annual carbon cost: ${carbon_result.carbon_cost_per_year}/year")

    # Test 4: Heat loss calculation
    print("\n4. Heat Loss Calculation")
    heat_loss = calculate_heat_loss_from_fouling(
        u_current=8.0,
        u_clean=10.0,
        heat_transfer_area_ft2=5000.0,
        lmtd_f=150.0,
        design_duty_mmbtu_hr=75.0
    )
    print(f"   Heat loss: {heat_loss.heat_loss_mmbtu_hr} MMBtu/hr")
    print(f"   Heat loss: {heat_loss.heat_loss_percent}%")
    print(f"   Temperature penalty: {heat_loss.temperature_penalty_f} F")

    print("\n   [FOULING CALCULATOR TESTS PASSED]")


def test_soot_blower_optimizer():
    """Test the advanced soot blower optimizer."""
    print("\n" + "=" * 60)
    print("TESTING: Advanced Soot Blower Optimizer")
    print("=" * 60)

    # Test 1: Optimal blowing interval
    print("\n1. Optimal Blowing Interval Calculation")
    config = SootBlowerConfiguration(
        blower_id="SB-001",
        blower_type=BlowerType.RETRACTABLE_LANCE,
        zone=EconomizerZone.GAS_INLET,
        medium=BlowingMedium.SATURATED_STEAM,
        steam_pressure_psia=300.0,
        steam_flow_lbm_per_cycle=500.0,
        cycle_duration_seconds=300.0,
    )
    interval_result = calculate_optimal_blowing_interval(
        fouling_rate=0.0001,
        cleaning_cost=25.00,
        fuel_cost_per_mmbtu=4.50,
        boiler_efficiency=0.85,
        boiler_heat_input_mmbtu_hr=100.0
    )
    print(f"   Optimal interval: {interval_result.optimal_interval_hours} hours")
    print(f"   Economic penalty: ${interval_result.economic_penalty_per_hour}/hr")
    print(f"   Confidence: {interval_result.confidence_level}")

    # Test 2: Zone prioritization
    print("\n2. Zone Prioritization")
    zones = [
        ZoneFoulingState(
            zone=EconomizerZone.GAS_INLET,
            fouling_factor=0.003,
            u_value_current=8.0,
            u_value_clean=10.0,
            gas_temp_inlet_f=500.0,
            gas_temp_outlet_f=350.0,
            last_cleaning_timestamp=datetime.now(timezone.utc) - timedelta(hours=48)
        ),
        ZoneFoulingState(
            zone=EconomizerZone.MIDDLE,
            fouling_factor=0.002,
            u_value_current=9.0,
            u_value_clean=10.0,
            gas_temp_inlet_f=400.0,
            gas_temp_outlet_f=320.0,
            last_cleaning_timestamp=datetime.now(timezone.utc) - timedelta(hours=24)
        ),
        ZoneFoulingState(
            zone=EconomizerZone.GAS_OUTLET,
            fouling_factor=0.004,
            u_value_current=7.5,
            u_value_clean=10.0,
            gas_temp_inlet_f=350.0,
            gas_temp_outlet_f=280.0,
            last_cleaning_timestamp=datetime.now(timezone.utc) - timedelta(hours=72)
        ),
    ]
    blower_configs = {
        EconomizerZone.GAS_INLET: config,
        EconomizerZone.MIDDLE: config,
        EconomizerZone.GAS_OUTLET: config,
    }
    priority_result = prioritize_cleaning_zones(
        zone_states=zones,
        blower_configs=blower_configs,
        fuel_cost_per_mmbtu=4.50,
        boiler_efficiency=0.85
    )
    print(f"   Highest priority zone: {priority_result[0].zone.value}")
    print(f"   Priority scores: {[round(float(z.priority_score), 2) for z in priority_result]}")

    # Test 3: Erosion monitoring
    print("\n3. Erosion Monitoring")
    erosion_result = monitor_erosion_wear(
        blower_config=config,
        cumulative_cycles=50000,
        initial_tube_thickness_mils=120.0,
        wear_limit_mils=40.0
    )
    print(f"   Estimated wear: {erosion_result.estimated_wear_mils} mils")
    print(f"   Wear severity: {erosion_result.wear_severity.value}")
    print(f"   Cycles to limit: {erosion_result.cycles_to_limit}")
    print(f"   Remaining life: {erosion_result.remaining_life_percent}%")

    print("\n   [SOOT BLOWER OPTIMIZER TESTS PASSED]")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" GL-020 ECONOPULSE: Advanced Calculator Module Tests")
    print("=" * 70)

    try:
        test_fouling_calculator()
        test_soot_blower_optimizer()

        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 70 + "\n")
        return 0
    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
