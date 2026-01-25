# -*- coding: utf-8 -*-
"""
GL-020 ECONOPULSE - Example Usage

This module demonstrates how to use the EconomizerPerformanceAgent
to monitor economizer performance, detect fouling, and optimize soot blowing.

Examples include:
    1. Basic heat transfer calculations
    2. Fouling analysis and trending
    3. Cleaning alert generation
    4. Soot blower optimization
    5. Efficiency loss quantification

Author: GreenLang Team
Date: December 2025
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import math

# Example 1: Basic Heat Transfer Calculations (ASME PTC 4.3)
# ============================================================================


def example_lmtd_calculation():
    """
    Demonstrate Log Mean Temperature Difference (LMTD) calculation.

    ASME PTC 4.3 compliant counter-flow economizer.
    """
    print("=" * 60)
    print("GL-020 ECONOPULSE - LMTD Calculation Example")
    print("=" * 60)

    # Operating conditions
    flue_gas_inlet_temp_f = 650.0  # °F
    flue_gas_outlet_temp_f = 350.0  # °F
    feedwater_inlet_temp_f = 250.0  # °F
    feedwater_outlet_temp_f = 320.0  # °F

    # Counter-flow configuration
    # dT1 = hot side in - cold side out
    # dT2 = hot side out - cold side in
    delta_t1 = flue_gas_inlet_temp_f - feedwater_outlet_temp_f  # 650 - 320 = 330°F
    delta_t2 = flue_gas_outlet_temp_f - feedwater_inlet_temp_f   # 350 - 250 = 100°F

    # LMTD Formula: LMTD = (dT1 - dT2) / ln(dT1/dT2)
    if abs(delta_t1 - delta_t2) < 0.1:
        # Special case: nearly equal temperature differences
        lmtd = (delta_t1 + delta_t2) / 2.0
    else:
        lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

    print(f"\nOperating Conditions:")
    print(f"  Flue Gas Inlet:     {flue_gas_inlet_temp_f}°F")
    print(f"  Flue Gas Outlet:    {flue_gas_outlet_temp_f}°F")
    print(f"  Feedwater Inlet:    {feedwater_inlet_temp_f}°F")
    print(f"  Feedwater Outlet:   {feedwater_outlet_temp_f}°F")

    print(f"\nTemperature Differences:")
    print(f"  ΔT1 (gas in - water out): {delta_t1}°F")
    print(f"  ΔT2 (gas out - water in): {delta_t2}°F")

    print(f"\nLMTD = (ΔT1 - ΔT2) / ln(ΔT1/ΔT2)")
    print(f"LMTD = ({delta_t1} - {delta_t2}) / ln({delta_t1}/{delta_t2})")
    print(f"LMTD = {lmtd:.2f}°F")

    # Approach temperature (important metric for economizer)
    approach_temp = flue_gas_outlet_temp_f - feedwater_inlet_temp_f
    print(f"\nApproach Temperature: {approach_temp}°F")
    print(f"  (Typical range: 50-150°F, lower is better)")

    return lmtd


def example_u_value_calculation():
    """
    Demonstrate Overall Heat Transfer Coefficient (U-value) calculation.

    U = Q / (A × LMTD)
    """
    print("\n" + "=" * 60)
    print("GL-020 ECONOPULSE - U-Value Calculation Example")
    print("=" * 60)

    # Known values
    heat_duty_mmbtu_hr = 25.0  # Heat transfer rate
    surface_area_ft2 = 5000.0  # Heat transfer surface
    lmtd_f = 193.05  # From previous calculation

    # Convert MMBtu/hr to Btu/hr
    heat_duty_btu_hr = heat_duty_mmbtu_hr * 1_000_000

    # U-value calculation: U = Q / (A × LMTD)
    u_value = heat_duty_btu_hr / (surface_area_ft2 * lmtd_f)

    print(f"\nInputs:")
    print(f"  Heat Duty (Q):      {heat_duty_mmbtu_hr} MMBtu/hr = {heat_duty_btu_hr:,.0f} Btu/hr")
    print(f"  Surface Area (A):   {surface_area_ft2:,.0f} ft²")
    print(f"  LMTD:               {lmtd_f:.2f}°F")

    print(f"\nCalculation:")
    print(f"  U = Q / (A × LMTD)")
    print(f"  U = {heat_duty_btu_hr:,.0f} / ({surface_area_ft2:,.0f} × {lmtd_f:.2f})")
    print(f"  U = {u_value:.2f} Btu/(hr·ft²·°F)")

    # Compare to typical values
    print(f"\nTypical U-Values (ASME PTC 4.3):")
    print(f"  Bare tube economizer:     8-15 Btu/(hr·ft²·°F)")
    print(f"  Finned tube economizer:   3-8 Btu/(hr·ft²·°F)")
    print(f"  Extended surface:         5-12 Btu/(hr·ft²·°F)")

    return u_value


# Example 2: Fouling Analysis
# ============================================================================


def example_fouling_calculation():
    """
    Demonstrate fouling factor (Rf) calculation.

    TEMA method: Rf = (1/U_fouled) - (1/U_clean)
    """
    print("\n" + "=" * 60)
    print("GL-020 ECONOPULSE - Fouling Analysis Example")
    print("=" * 60)

    # U-values
    u_clean = 12.5  # Design/clean U-value (Btu/hr·ft²·°F)
    u_fouled = 9.8  # Current U-value after fouling

    # Fouling factor calculation
    rf = (1.0 / u_fouled) - (1.0 / u_clean)

    # Cleanliness factor
    cleanliness_factor = (u_fouled / u_clean) * 100

    print(f"\nU-Values:")
    print(f"  Clean (design):    {u_clean} Btu/(hr·ft²·°F)")
    print(f"  Current (fouled):  {u_fouled} Btu/(hr·ft²·°F)")

    print(f"\nFouling Factor Calculation:")
    print(f"  Rf = (1/U_fouled) - (1/U_clean)")
    print(f"  Rf = (1/{u_fouled}) - (1/{u_clean})")
    print(f"  Rf = {1/u_fouled:.6f} - {1/u_clean:.6f}")
    print(f"  Rf = {rf:.6f} hr·ft²·°F/Btu")
    print(f"  Rf = {rf * 1000:.4f} × 10⁻³ hr·ft²·°F/Btu")

    print(f"\nCleanliness Factor:")
    print(f"  CF = (U_fouled / U_clean) × 100%")
    print(f"  CF = ({u_fouled} / {u_clean}) × 100%")
    print(f"  CF = {cleanliness_factor:.1f}%")

    # Fouling severity classification
    if rf < 0.0005:
        severity = "CLEAN"
    elif rf < 0.001:
        severity = "LIGHT"
    elif rf < 0.002:
        severity = "MODERATE"
    elif rf < 0.003:
        severity = "HEAVY"
    else:
        severity = "SEVERE"

    print(f"\nFouling Severity: {severity}")
    print(f"\nTEMA Maximum Fouling Factors:")
    print(f"  Flue gas (clean fuel):   0.001 hr·ft²·°F/Btu")
    print(f"  Flue gas (ash-bearing):  0.003 hr·ft²·°F/Btu")
    print(f"  Boiler feedwater:        0.0005 hr·ft²·°F/Btu")

    return rf, cleanliness_factor


def example_efficiency_loss():
    """
    Demonstrate efficiency loss quantification from fouling.
    """
    print("\n" + "=" * 60)
    print("GL-020 ECONOPULSE - Efficiency Loss Example")
    print("=" * 60)

    # Parameters
    rf = 0.0022  # Fouling factor (hr·ft²·°F/Btu)
    u_clean = 12.5  # Clean U-value
    heat_duty_design_mmbtu_hr = 30.0  # Design heat duty
    boiler_efficiency = 0.85  # Boiler efficiency
    fuel_price_per_mmbtu = 4.50  # Natural gas price ($/MMBtu)
    operating_hours_per_year = 8000

    # Efficiency loss formula: η_loss = (Rf × U_clean²) / (1 + Rf × U_clean) × 100
    numerator = rf * (u_clean ** 2)
    denominator = 1 + (rf * u_clean)
    efficiency_loss_pct = (numerator / denominator) * 100

    # Heat loss calculation
    heat_loss_mmbtu_hr = heat_duty_design_mmbtu_hr * (efficiency_loss_pct / 100)

    # Fuel penalty (additional fuel needed)
    fuel_penalty_mmbtu_hr = heat_loss_mmbtu_hr / boiler_efficiency
    fuel_penalty_usd_hr = fuel_penalty_mmbtu_hr * fuel_price_per_mmbtu

    # Annual impact
    annual_fuel_penalty_usd = fuel_penalty_usd_hr * operating_hours_per_year

    print(f"\nInputs:")
    print(f"  Fouling Factor (Rf):      {rf} hr·ft²·°F/Btu")
    print(f"  Clean U-Value:            {u_clean} Btu/(hr·ft²·°F)")
    print(f"  Design Heat Duty:         {heat_duty_design_mmbtu_hr} MMBtu/hr")
    print(f"  Boiler Efficiency:        {boiler_efficiency * 100}%")
    print(f"  Fuel Price:               ${fuel_price_per_mmbtu}/MMBtu")
    print(f"  Operating Hours/Year:     {operating_hours_per_year}")

    print(f"\nEfficiency Loss Calculation:")
    print(f"  η_loss = (Rf × U_clean²) / (1 + Rf × U_clean) × 100%")
    print(f"  η_loss = ({rf} × {u_clean}²) / (1 + {rf} × {u_clean}) × 100%")
    print(f"  η_loss = {efficiency_loss_pct:.2f}%")

    print(f"\nEconomic Impact:")
    print(f"  Heat Loss:                {heat_loss_mmbtu_hr:.2f} MMBtu/hr")
    print(f"  Fuel Penalty:             {fuel_penalty_mmbtu_hr:.2f} MMBtu/hr")
    print(f"  Cost Penalty:             ${fuel_penalty_usd_hr:.2f}/hr")
    print(f"  Annual Cost Impact:       ${annual_fuel_penalty_usd:,.0f}/year")

    return efficiency_loss_pct, annual_fuel_penalty_usd


# Example 3: Soot Blower Optimization
# ============================================================================


def example_soot_blower_optimization():
    """
    Demonstrate optimal soot blowing interval calculation.

    t_optimal = √(2 × C_cleaning / (dRf/dt × C_fuel × k))
    """
    print("\n" + "=" * 60)
    print("GL-020 ECONOPULSE - Soot Blower Optimization Example")
    print("=" * 60)

    # Parameters
    cleaning_cost_usd = 50.0  # Cost per cleaning cycle (steam + wear)
    fouling_rate_per_day = 0.0001  # dRf/dt (hr·ft²·°F/Btu per day)
    fuel_cost_per_mmbtu = 4.50
    heat_duty_mmbtu_hr = 30.0
    efficiency_sensitivity = 2.5  # Efficiency loss % per 0.001 Rf
    operating_hours_per_day = 24

    # Convert efficiency sensitivity to dollar impact
    # k = sensitivity factor relating Rf to fuel cost
    k = (efficiency_sensitivity / 0.001) * heat_duty_mmbtu_hr * fuel_cost_per_mmbtu / 100

    # Optimal cleaning interval (days)
    # t_optimal = √(2 × C_cleaning / (dRf/dt × k))
    t_optimal_days = math.sqrt(
        (2 * cleaning_cost_usd) / (fouling_rate_per_day * k)
    )

    # Convert to hours
    t_optimal_hours = t_optimal_days * 24

    print(f"\nInputs:")
    print(f"  Cleaning Cost:            ${cleaning_cost_usd}/cycle")
    print(f"  Fouling Rate:             {fouling_rate_per_day} hr·ft²·°F/Btu per day")
    print(f"  Fuel Cost:                ${fuel_cost_per_mmbtu}/MMBtu")
    print(f"  Heat Duty:                {heat_duty_mmbtu_hr} MMBtu/hr")

    print(f"\nOptimal Cleaning Interval Calculation:")
    print(f"  t_optimal = √(2 × C_cleaning / (dRf/dt × k))")
    print(f"  t_optimal = √(2 × {cleaning_cost_usd} / ({fouling_rate_per_day} × {k:.2f}))")
    print(f"  t_optimal = {t_optimal_days:.1f} days = {t_optimal_hours:.0f} hours")

    # Calculate ROI for cleaning
    # Cost of not cleaning for t_optimal days
    fouling_after_interval = fouling_rate_per_day * t_optimal_days
    avg_fouling = fouling_after_interval / 2  # Average over interval
    efficiency_loss_avg = (avg_fouling / 0.001) * efficiency_sensitivity
    fuel_penalty_per_day = (efficiency_loss_avg / 100) * heat_duty_mmbtu_hr * 24 * fuel_cost_per_mmbtu
    total_penalty_over_interval = fuel_penalty_per_day * t_optimal_days / 2

    print(f"\nCleaning ROI Analysis:")
    print(f"  Cleaning Cost:            ${cleaning_cost_usd}")
    print(f"  Fouling Penalty Avoided:  ${total_penalty_over_interval:.2f}")
    print(f"  Net Benefit per Cycle:    ${total_penalty_over_interval - cleaning_cost_usd:.2f}")

    return t_optimal_hours


# Example 4: Complete Performance Analysis
# ============================================================================


async def example_complete_analysis():
    """
    Demonstrate a complete economizer performance analysis workflow.
    """
    print("\n" + "=" * 60)
    print("GL-020 ECONOPULSE - Complete Performance Analysis")
    print("=" * 60)

    # Simulated sensor readings
    readings = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "economizer_id": "ECON-001",
        "flue_gas_inlet_temp_f": 650.0,
        "flue_gas_outlet_temp_f": 355.0,
        "feedwater_inlet_temp_f": 252.0,
        "feedwater_outlet_temp_f": 318.0,
        "feedwater_flow_lb_hr": 250000.0,
        "flue_gas_flow_lb_hr": 300000.0,
    }

    # Economizer specs
    specs = {
        "surface_area_ft2": 5000.0,
        "design_u_value": 12.5,
        "design_effectiveness": 0.72,
    }

    print(f"\nSensor Readings:")
    for key, value in readings.items():
        print(f"  {key}: {value}")

    # Step 1: Calculate LMTD
    delta_t1 = readings["flue_gas_inlet_temp_f"] - readings["feedwater_outlet_temp_f"]
    delta_t2 = readings["flue_gas_outlet_temp_f"] - readings["feedwater_inlet_temp_f"]
    lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

    # Step 2: Calculate heat duty (water side)
    cp_water = 1.0  # Btu/lb·°F (approximate)
    delta_t_water = readings["feedwater_outlet_temp_f"] - readings["feedwater_inlet_temp_f"]
    heat_duty_btu_hr = readings["feedwater_flow_lb_hr"] * cp_water * delta_t_water
    heat_duty_mmbtu_hr = heat_duty_btu_hr / 1_000_000

    # Step 3: Calculate U-value
    u_value = heat_duty_btu_hr / (specs["surface_area_ft2"] * lmtd)

    # Step 4: Calculate fouling factor
    rf = (1.0 / u_value) - (1.0 / specs["design_u_value"])
    cleanliness_factor = (u_value / specs["design_u_value"]) * 100

    # Step 5: Calculate effectiveness
    t_gas_in = readings["flue_gas_inlet_temp_f"]
    t_water_in = readings["feedwater_inlet_temp_f"]
    t_water_out = readings["feedwater_outlet_temp_f"]
    effectiveness = (t_water_out - t_water_in) / (t_gas_in - t_water_in)

    # Step 6: Calculate approach temperature
    approach_temp = readings["flue_gas_outlet_temp_f"] - readings["feedwater_inlet_temp_f"]

    print(f"\n--- Performance Metrics ---")
    print(f"  LMTD:                {lmtd:.2f}°F")
    print(f"  Heat Duty:           {heat_duty_mmbtu_hr:.2f} MMBtu/hr")
    print(f"  U-Value:             {u_value:.2f} Btu/(hr·ft²·°F)")
    print(f"  Effectiveness:       {effectiveness:.3f} ({effectiveness*100:.1f}%)")
    print(f"  Approach Temp:       {approach_temp:.1f}°F")

    print(f"\n--- Fouling Status ---")
    print(f"  Fouling Factor:      {rf:.6f} hr·ft²·°F/Btu")
    print(f"  Cleanliness Factor:  {cleanliness_factor:.1f}%")

    # Determine fouling severity
    if rf < 0.0005:
        severity = "CLEAN"
        recommendation = "No action needed"
    elif rf < 0.001:
        severity = "LIGHT"
        recommendation = "Monitor closely"
    elif rf < 0.002:
        severity = "MODERATE"
        recommendation = "Schedule cleaning within 7 days"
    elif rf < 0.003:
        severity = "HEAVY"
        recommendation = "Schedule cleaning within 48 hours"
    else:
        severity = "SEVERE"
        recommendation = "IMMEDIATE CLEANING REQUIRED"

    print(f"  Severity:            {severity}")
    print(f"  Recommendation:      {recommendation}")

    # Generate provenance hash (simplified)
    import hashlib
    provenance_data = f"{readings}{lmtd}{u_value}{rf}"
    provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()
    print(f"\n  Provenance Hash:     {provenance_hash[:16]}...")

    return {
        "lmtd": lmtd,
        "heat_duty_mmbtu_hr": heat_duty_mmbtu_hr,
        "u_value": u_value,
        "fouling_factor": rf,
        "cleanliness_factor": cleanliness_factor,
        "effectiveness": effectiveness,
        "approach_temp": approach_temp,
        "severity": severity,
        "recommendation": recommendation,
        "provenance_hash": provenance_hash,
    }


# Main Entry Point
# ============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GL-020 ECONOPULSE - Usage Examples")
    print("Economizer Performance Monitoring Agent")
    print("=" * 60 + "\n")

    # Run examples
    example_lmtd_calculation()
    example_u_value_calculation()
    example_fouling_calculation()
    example_efficiency_loss()
    example_soot_blower_optimization()
    await example_complete_analysis()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
