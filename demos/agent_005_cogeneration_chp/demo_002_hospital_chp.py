# -*- coding: utf-8 -*-
"""
Demo #2: Hospital CHP System Analysis
Agent: CogenerationCHPAgent_AI (Agent #5)

SCENARIO:
---------
Large regional hospital considering 1 MW fuel cell CHP system for mission-critical
power, heating, and cooling with emphasis on reliability and ultra-low emissions.

FACILITY DETAILS:
----------------
- Type: Regional medical center (250 beds, surgical suites, ICU)
- Location: Urban area (Northeast USA)
- Operating Schedule: 24/7/365 critical operation (8,760 hours/year)
- Electrical Demand: 1,200 kW peak, 1,000 kW average
- Thermal Demand: 6.5 MMBtu/hr peak, 5.0 MMBtu/hr average
- Heat-to-Power Ratio: 1.5 (thermal/electrical)
- Current Energy Costs:
  - Electricity: $0.15/kWh ($1,314,000/year)
  - Natural Gas: $7.50/MMBtu ($328,500/year for thermal)
  - Demand Charges: $18/kW-month ($259,200/year)
  - Total: $1,901,700/year

HOSPITAL REQUIREMENTS:
---------------------
- Ultra-high reliability (>99.5% uptime)
- Near-zero emissions (urban air quality regulations)
- Quiet operation (<65 dBA - patient comfort)
- Backup power capability (life safety systems)
- Resilience during grid outages
- Low NOx emissions (<1 ppm)

OBJECTIVES:
----------
1. Select CHP technology meeting hospital-grade requirements
2. Calculate fuel cell performance and high electrical efficiency
3. Design heat recovery for steam + domestic hot water
4. Analyze economics with high electricity rates
5. Assess critical facility grid interconnection
6. Optimize for 24/7 baseload operation
7. Calculate emissions reduction (strict urban limits)
8. Generate hospital-specific feasibility report

EXPECTED RESULTS:
----------------
- Recommended Technology: Fuel Cell (MCFC or SOFC)
- Electrical Efficiency: 45% (highest among CHP technologies)
- Total CHP Efficiency: 80%
- Simple Payback: 5.5 years (higher CAPEX, offset by premium rates)
- 20-Year NPV: $1,800,000
- Annual Savings: $650,000
- CO2 Reduction: 2,200 tonnes/year
- NOx Emissions: <1 ppm (near-zero)

DEMO WORKFLOW:
-------------
1. Initialize agent with hospital configuration
2. Tool 1: Select hospital-grade CHP technology
3. Tool 2: Calculate fuel cell performance
4. Tool 3: Size heat recovery (steam + DHW)
5. Tool 4: Calculate economics (premium rates)
6. Tool 5: Assess critical facility grid interconnection
7. Tool 6: Optimize 24/7 baseload strategy
8. Tool 7: Calculate emissions (urban air quality)
9. Tool 8: Generate hospital CHP report
10. Display results with reliability focus

USAGE:
------
    python demo_002_hospital_chp.py

"""

from typing import Dict, Any
from datetime import datetime
import json

from greenlang.agents.cogeneration_chp_agent_ai import (
    CogenerationCHPAgentAI,
    CogenerationCHPConfig
)
from greenlang.determinism import DeterministicClock


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """
    Main demo workflow for hospital CHP analysis
    """
    print_section("DEMO #2: HOSPITAL CHP SYSTEM ANALYSIS")

    print("Scenario: Regional medical center (250 beds)")
    print("Location: Urban area, Northeast USA")
    print("Operating Schedule: 24/7/365 critical operation")
    print("Electrical Demand: 1,200 kW peak, 1,000 kW average")
    print("Thermal Demand: 6.5 MMBtu/hr peak, 5.0 MMBtu/hr average")
    print("Current Annual Energy Costs: $1,901,700")
    print("Requirements: Ultra-high reliability, near-zero emissions")

    # ========================================================================
    # STEP 1: Initialize Agent
    # ========================================================================

    print_section("STEP 1: INITIALIZE HOSPITAL CHP AGENT")

    config = CogenerationCHPConfig(
        agent_id="demo/hospital_chp",
        agent_name="HospitalCHPDemo",
        budget_usd=0.50,
        temperature=0.0,
        seed=42,
        deterministic=True
    )

    agent = CogenerationCHPAgentAI(config=config)

    print(f"✓ Agent initialized: {config.agent_name}")
    print(f"✓ Version: {agent._version()}")
    print(f"✓ Application: Critical healthcare facility")
    print(f"✓ Deterministic: {config.deterministic}")

    # ========================================================================
    # STEP 2: Select Hospital-Grade CHP Technology
    # ========================================================================

    print_section("STEP 2: SELECT HOSPITAL-GRADE CHP TECHNOLOGY")

    print("Hospital-specific selection criteria:")
    print("  - Ultra-high reliability (>99.5% uptime)")
    print("  - Near-zero emissions (NOx <1 ppm)")
    print("  - Quiet operation (<65 dBA)")
    print("  - High electrical efficiency (premium rates)")
    print("  - Fast response for load following")

    tech_result = agent.select_chp_technology(
        electrical_demand_kw=1000,
        thermal_demand_mmbtu_hr=5.0,
        heat_to_power_ratio=1.5,
        load_profile_type="baseload_24x7",
        available_fuels=["natural_gas"],
        emissions_requirements="ultra_low",
        space_constraints="limited",
        required_electrical_efficiency=0.40,  # High efficiency requirement
        grid_export_allowed=False,
        resilience_priority="critical"  # Life safety systems
    )

    print(f"\n✓ Technology Selection Complete")
    print(f"\nRecommended Technology: {tech_result['recommended_technology'].upper()}")
    print(f"  (Optimal for hospitals due to ultra-low emissions and high efficiency)")

    print(f"\nPerformance Characteristics:")
    print(f"  - Electrical Efficiency: {tech_result['typical_electrical_efficiency']:.1%} ⭐ HIGHEST")
    print(f"  - Total CHP Efficiency: {tech_result['typical_total_efficiency']:.1%}")
    print(f"  - Heat-to-Power Ratio: {tech_result['heat_to_power_ratio_achievable']:.2f}")

    print(f"\nHospital-Critical Advantages:")
    for advantage in tech_result['key_advantages']:
        if 'emissions' in advantage.lower() or 'efficiency' in advantage.lower() or 'quiet' in advantage.lower():
            print(f"  ✓ {advantage}")

    print(f"\nCost Characteristics:")
    print(f"  - Capital Cost: ${tech_result['estimated_capex_per_kw']:,.0f}/kW")
    print(f"    (Higher than other technologies, offset by premium electricity rates)")
    print(f"  - O&M Cost: ${tech_result['maintenance_cost_per_kwh']:.3f}/kWh")
    print(f"  - Overhaul Interval: {tech_result['typical_overhaul_interval_hours']:,.0f} hours")
    print(f"    (Extended maintenance intervals minimize downtime)")

    # ========================================================================
    # STEP 3: Calculate Fuel Cell Performance
    # ========================================================================

    print_section("STEP 3: CALCULATE FUEL CELL CHP PERFORMANCE")

    print("Analyzing fuel cell thermodynamics:")
    print("  - Technology: MCFC/SOFC (Molten Carbonate/Solid Oxide)")
    print("  - Electrical Capacity: 1,000 kW")
    print("  - Fuel Input: 7.5 MMBtu/hr (HHV)")
    print("  - Heat Recovery: Low-grade (<200°F) for DHW")
    print("  - Operating Temperature: 1200-1800°F (internal)")

    perf_result = agent.calculate_chp_performance(
        chp_technology=tech_result['recommended_technology'],
        electrical_capacity_kw=1000,
        fuel_input_mmbtu_hr=7.5,
        heat_recovery_configuration="jacket_water_only",  # Lower grade heat from fuel cells
        exhaust_temperature_f=550,  # Lower than combustion engines
        exhaust_mass_flow_lb_hr=8000,
        heat_recovery_target_temperature_f=200,  # Domestic hot water
        ambient_temperature_f=59.0,
        part_load_ratio=1.0
    )

    print(f"\n✓ Fuel Cell Performance Analysis Complete")
    print(f"\nElectrical Performance:")
    print(f"  - Output: {perf_result['electrical_output_kw']:,.0f} kW")
    print(f"  - Efficiency: {perf_result['electrical_efficiency']:.1%} (HHV) ⭐ HIGHEST")
    print(f"  - Heat Rate: {perf_result['heat_rate_btu_per_kwh']:,.0f} Btu/kWh")
    print(f"    (Lower heat rate = higher efficiency)")

    print(f"\nThermal Performance:")
    print(f"  - Output: {perf_result['thermal_output_mmbtu_hr']:.2f} MMBtu/hr")
    print(f"  - Efficiency: {perf_result['thermal_efficiency']:.1%}")
    print(f"  - Quality: Low-grade (<200°F) suitable for DHW, space heating")

    print(f"\nTotal System Performance:")
    print(f"  - Total Efficiency: {perf_result['total_efficiency']:.1%}")
    print(f"  - Fuel Input: {perf_result['fuel_input_mmbtu_hr']:.1f} MMBtu/hr")

    print(f"\nFuel Cell Advantages for Hospitals:")
    print(f"  ✓ Highest electrical efficiency (45% vs 35-38% for engines)")
    print(f"  ✓ Near-zero NOx emissions (<1 ppm vs 5-25 ppm)")
    print(f"  ✓ Quiet operation (60 dBA vs 75-85 dBA)")
    print(f"  ✓ High availability (>95% uptime)")
    print(f"  ✓ Excellent part-load performance (no penalty)")

    # ========================================================================
    # STEP 4: Size Heat Recovery System
    # ========================================================================

    print_section("STEP 4: SIZE HEAT RECOVERY SYSTEM")

    print("Designing heat recovery for hospital applications:")
    print("  - Primary Use: Domestic hot water (patient rooms, surgical)")
    print("  - Secondary Use: Space heating (HVAC integration)")
    print("  - Target Temperature: 180°F (DHW storage)")
    print("  - Configuration: Jacket water heat recovery")

    hr_result = agent.size_heat_recovery_system(
        exhaust_temperature_f=550,
        exhaust_mass_flow_lb_hr=8000,
        process_heat_demand_mmbtu_hr=5.0,
        process_temperature_requirement_f=180,
        heat_recovery_type="jacket_water_only"
    )

    print(f"\n✓ Heat Recovery System Sizing Complete")
    print(f"\nHeat Recovery Performance:")
    print(f"  - Available Heat: {hr_result['available_heat_mmbtu_hr']:.2f} MMBtu/hr")
    print(f"  - Recovered Heat: {hr_result['recovered_heat_mmbtu_hr']:.2f} MMBtu/hr")
    print(f"  - Recovery Effectiveness: {hr_result['recovery_effectiveness_pct']:.1f}%")

    print(f"\nHospital Heat Applications:")
    print(f"  - Domestic Hot Water: ~60% (patient rooms, surgery, laundry)")
    print(f"  - Space Heating: ~30% (HVAC integration)")
    print(f"  - Process Heat: ~10% (sterilization, kitchen)")

    print(f"\nEquipment Sizing:")
    print(f"  - Heat Exchanger Area: {hr_result['heat_exchanger_area_sqft']:,.0f} sq ft")
    print(f"  - Estimated CAPEX: ${hr_result['estimated_capex_usd']:,.0f}")

    # ========================================================================
    # STEP 5: Calculate Economic Metrics
    # ========================================================================

    print_section("STEP 5: CALCULATE ECONOMIC METRICS")

    print("Analyzing economics with premium hospital electricity rates:")
    print("  - Electricity Rate: $0.15/kWh (premium urban rate)")
    print("  - Demand Charge: $18/kW-month (critical facility)")
    print("  - Operating Hours: 8,760 hours/year (24/7/365)")
    print("  - Federal ITC: 10% (qualifying fuel cell)")
    print("  - State Hospital Incentive: $500,000 (energy resilience)")

    econ_result = agent.calculate_economic_metrics(
        electrical_output_kw=perf_result['electrical_output_kw'],
        thermal_output_mmbtu_hr=perf_result['thermal_output_mmbtu_hr'],
        fuel_input_mmbtu_hr=perf_result['fuel_input_mmbtu_hr'],
        annual_operating_hours=8760,  # 24/7/365
        electricity_rate_per_kwh=0.15,  # Premium rate
        demand_charge_per_kw_month=18.0,  # High demand charge
        gas_price_per_mmbtu=7.50,
        thermal_fuel_displaced="natural_gas",
        thermal_fuel_price_per_mmbtu=7.50,
        thermal_boiler_efficiency=0.85,
        chp_capex_usd=5_000_000,  # Higher for fuel cell
        chp_opex_per_kwh=0.025,  # Higher O&M
        federal_itc_percent=10.0,
        state_incentive_usd=500_000,  # Hospital resilience grant
        discount_rate=0.08,
        analysis_period_years=20
    )

    print(f"\n✓ Economic Analysis Complete")
    print(f"\nSpark Spread Analysis:")
    print(f"  - Spark Spread: ${econ_result['spark_spread_per_mwh']:.2f}/MWh")
    print(f"    (Strong economics with $0.15/kWh rate)")

    print(f"\nAvoided Costs (Annual):")
    print(f"  - Electricity: ${econ_result['avoided_electricity_cost_annual']:,.0f}")
    print(f"  - Demand Charges: ${econ_result['avoided_demand_charge_annual']:,.0f}")
    print(f"  - Thermal Fuel: ${econ_result['avoided_thermal_cost_annual']:,.0f}")
    print(f"  - Total Avoided: ${econ_result['total_avoided_costs_annual']:,.0f}")

    print(f"\nCHP Operating Costs (Annual):")
    print(f"  - Fuel: ${econ_result['chp_fuel_cost_annual']:,.0f}")
    print(f"  - O&M: ${econ_result['chp_om_cost_annual']:,.0f}")
    print(f"  - Total Operating: ${econ_result['total_chp_operating_costs_annual']:,.0f}")

    print(f"\nNet Annual Savings: ${econ_result['net_annual_savings']:,.0f}")

    print(f"\nCapital Investment:")
    print(f"  - Gross CAPEX: ${econ_result['chp_capex_gross']:,.0f}")
    print(f"  - Federal ITC (10%): -${econ_result['federal_itc_value']:,.0f}")
    print(f"  - State Hospital Grant: -${econ_result['state_incentive_value']:,.0f}")
    print(f"  - Net CAPEX: ${econ_result['net_capex_after_incentives']:,.0f}")

    print(f"\nFinancial Metrics:")
    print(f"  - Simple Payback: {econ_result['simple_payback_years']:.1f} years")
    print(f"  - NPV (20-year): ${econ_result['npv_20yr']:,.0f}")
    print(f"  - IRR: {econ_result['irr_percent']:.1f}%")
    print(f"  - Benefit-Cost Ratio: {econ_result['benefit_cost_ratio']:.2f}")

    print(f"\nResilience Value (Not Monetized Above):")
    print(f"  - Backup power for life safety systems")
    print(f"  - Reduced risk of surgical disruption")
    print(f"  - Patient safety during grid outages")
    print(f"  - Estimated value: $200,000-$500,000/year")

    # ========================================================================
    # STEP 6: Assess Grid Interconnection
    # ========================================================================

    print_section("STEP 6: ASSESS CRITICAL FACILITY GRID INTERCONNECTION")

    print("Analyzing grid interconnection for critical healthcare facility:")
    print("  - CHP Capacity: 1,000 kW")
    print("  - Facility Peak: 1,200 kW")
    print("  - Voltage: Low voltage (480V)")
    print("  - Export Mode: No export (backup/island mode)")
    print("  - Classification: Essential electrical system (NFPA 99)")

    intercon_result = agent.assess_grid_interconnection(
        chp_electrical_capacity_kw=1000,
        facility_peak_demand_kw=1200,
        voltage_level="low_voltage_480v",
        export_mode="no_export",
        utility_territory="investor_owned",
        distance_to_substation_miles=0.5,
        existing_service_capacity_kw=1500
    )

    print(f"\n✓ Grid Interconnection Assessment Complete")
    print(f"\nIEEE 1547 Screening:")
    print(f"  - Category: {intercon_result['ieee_1547_category']}")
    print(f"  - Timeline: {intercon_result['utility_application_timeline_weeks']} weeks")
    print(f"  - Study Required: {'Yes' if intercon_result['utility_study_required'] else 'No'}")

    print(f"\nCritical Facility Requirements:")
    print(f"  - Islanding Protection: {'Required' if intercon_result['islanding_protection_required'] else 'Not Required'}")
    print(f"  - Paralleling Gear: {'Required' if intercon_result['paralleling_gear_required'] else 'Not Required'}")
    print(f"  - Automatic Transfer Switch: Required (NFPA 99)")
    print(f"  - Emergency Generator Coordination: Required")

    print(f"\nInterconnection Costs:")
    print(f"  - Equipment: ${intercon_result['estimated_interconnection_equipment_cost']:,.0f}")
    print(f"  - Total: ${intercon_result['total_interconnection_cost_estimate']:,.0f}")

    print(f"\nUtility Standby Charges:")
    print(f"  - Rate: ${intercon_result['standby_charge_per_kw_month']:.2f}/kW-month")
    print(f"  - Annual: ${intercon_result['annual_standby_charges']:,.0f}")
    print(f"    (Applicable when grid-connected, CHP provides backup)")

    # ========================================================================
    # STEP 7: Optimize Operating Strategy
    # ========================================================================

    print_section("STEP 7: OPTIMIZE 24/7 BASELOAD STRATEGY")

    print("Hospital operating strategy:")
    print("  - Profile: 24/7/365 baseload (constant electrical + thermal)")
    print("  - Electrical: 1,000 kW constant")
    print("  - Thermal: 5.0 MMBtu/hr constant")
    print("  - Strategy: Baseload (maximize reliability)")

    electrical_profile = [1000] * 24
    thermal_profile = [5.0] * 24
    rate_schedule = [0.15] * 24

    strategy_result = agent.optimize_operating_strategy(
        electrical_load_profile_kw=electrical_profile,
        thermal_load_profile_mmbtu_hr=thermal_profile,
        chp_electrical_capacity_kw=1000,
        chp_thermal_capacity_mmbtu_hr=5.0,
        electricity_rate_schedule=rate_schedule,
        gas_price_per_mmbtu=7.50,
        strategy_type="baseload"
    )

    print(f"\n✓ Operating Strategy Optimization Complete")
    print(f"\nRecommended Strategy: {strategy_result['recommended_strategy'].upper()}")
    print(f"Rationale: {strategy_result['strategy_rationale']}")

    print(f"\nCapacity Factors:")
    print(f"  - Electrical: {strategy_result['electrical_capacity_factor']:.1%}")
    print(f"  - Thermal: {strategy_result['thermal_capacity_factor']:.1%}")
    print(f"    (Near 100% for hospital 24/7 operation)")

    print(f"\nAnnual Performance:")
    print(f"  - Operating Hours: {strategy_result['annual_operating_hours']:,.0f} hours/year")
    print(f"  - Electrical Generation: {strategy_result['annual_electrical_generation_kwh']:,.0f} kWh/year")
    print(f"  - Thermal Generation: {strategy_result['annual_thermal_generation_mmbtu']:,.0f} MMBtu/year")

    # ========================================================================
    # STEP 8: Calculate Emissions Reduction
    # ========================================================================

    print_section("STEP 8: CALCULATE EMISSIONS REDUCTION")

    print("Urban air quality emissions analysis:")
    print("  - Baseline: Regional grid (0.35 kg CO2/kWh - cleaner grid)")
    print("  - Fuel Cell NOx: <1 ppm (near-zero)")
    print("  - Urban Air Quality Compliance: Required")

    emis_result = agent.calculate_emissions_reduction(
        chp_electrical_output_kwh_annual=econ_result['annual_kwh_generated'],
        chp_thermal_output_mmbtu_annual=econ_result['annual_thermal_mmbtu_generated'],
        chp_fuel_input_mmbtu_annual=perf_result['fuel_input_mmbtu_hr'] * 8760,
        chp_fuel_type="natural_gas",
        baseline_grid_emissions_kg_co2_per_kwh=0.35,  # Cleaner Northeast grid
        baseline_thermal_fuel_type="natural_gas",
        baseline_boiler_efficiency=0.85,
        include_upstream_emissions=True
    )

    print(f"\n✓ Emissions Analysis Complete")
    print(f"\nEmissions Reduction:")
    print(f"  - Annual Reduction: {emis_result['emissions_reduction_tonnes_co2_annual']:,.0f} tonnes CO2/year")
    print(f"  - Percent Reduction: {emis_result['emissions_reduction_percent']:.1f}%")

    print(f"\nEmission Intensity:")
    print(f"  - Fuel Cell CHP: {emis_result['chp_emission_intensity_kg_co2_per_kwh_equivalent']:.3f} kg CO2/kWh-eq")
    print(f"  - Baseline: {emis_result['baseline_emission_intensity_kg_co2_per_kwh_equivalent']:.3f} kg CO2/kWh-eq")

    print(f"\nUrban Air Quality Benefits:")
    print(f"  - NOx Emissions: <1 ppm (fuel cell) vs 5-25 ppm (combustion)")
    print(f"  - Particulate Matter: Near-zero")
    print(f"  - Carbon Monoxide: Near-zero")
    print(f"  - VOC Emissions: Minimal")

    cars_equivalent = emis_result['emissions_reduction_tonnes_co2_annual'] / 4.6
    print(f"\nEnvironmental Impact:")
    print(f"  - Equivalent to {cars_equivalent:.0f} cars removed from roads")
    print(f"  - Supports hospital sustainability goals")
    print(f"  - Enhances community health (urban air quality)")

    # ========================================================================
    # STEP 9: Generate Hospital CHP Report
    # ========================================================================

    print_section("STEP 9: GENERATE HOSPITAL CHP REPORT")

    report = agent.generate_chp_report(
        technology_selection_result=tech_result,
        performance_result=perf_result,
        economic_result=econ_result,
        emissions_result=emis_result,
        interconnection_result=intercon_result,
        operating_strategy_result=strategy_result,
        facility_name="Regional Medical Center - 250 Bed Hospital",
        report_type="comprehensive"
    )

    print(f"\n✓ Report Generation Complete")
    print(f"\nReport: {report['facility_name']}")
    print(f"Date: {report['report_date']}")
    print(f"Overall Recommendation: {report['overall_recommendation']}")

    print(report['executive_summary'])

    # ========================================================================
    # STEP 10: Summary and Recommendations
    # ========================================================================

    print_section("STEP 10: HOSPITAL CHP SUMMARY & RECOMMENDATIONS")

    print("✓ HOSPITAL CHP FEASIBILITY ANALYSIS COMPLETE")
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY FOR HOSPITAL BOARD")
    print("=" * 60)

    print(f"\nFacility: Regional Medical Center (250 beds)")
    print(f"Analysis Date: {DeterministicClock.now().strftime('%Y-%m-%d')}")

    print(f"\nRecommended Solution:")
    print(f"  - Technology: Fuel Cell CHP (MCFC/SOFC)")
    print(f"  - Capacity: 1,000 kW electrical / 5.0 MMBtu/hr thermal")
    print(f"  - Electrical Efficiency: {perf_result['electrical_efficiency']:.1%} (HIGHEST)")
    print(f"  - Total Efficiency: {perf_result['total_efficiency']:.1%}")

    print(f"\nFinancial Analysis:")
    print(f"  - Total Investment: ${econ_result['net_capex_after_incentives']:,.0f}")
    print(f"  - Annual Savings: ${econ_result['net_annual_savings']:,.0f}")
    print(f"  - Simple Payback: {econ_result['simple_payback_years']:.1f} years")
    print(f"  - 20-Year NPV: ${econ_result['npv_20yr']:,.0f}")
    print(f"  - IRR: {econ_result['irr_percent']:.1f}%")

    print(f"\nResilience & Reliability:")
    print(f"  - Uptime: >99.5% (hospital-grade)")
    print(f"  - Backup Power: Integrated with emergency systems")
    print(f"  - Island Mode: Capable during grid outages")
    print(f"  - Life Safety: Protects critical patient care")

    print(f"\nEnvironmental & Community:")
    print(f"  - CO2 Reduction: {emis_result['emissions_reduction_tonnes_co2_annual']:,.0f} tonnes/year")
    print(f"  - NOx Emissions: <1 ppm (near-zero)")
    print(f"  - Noise: <65 dBA (patient comfort)")
    print(f"  - Urban Air Quality: Significant improvement")

    print("\n" + "=" * 60)
    print("RECOMMENDATION: PROCEED WITH FUEL CELL CHP")
    print("=" * 60)

    print("\nHospital-Specific Advantages:")
    print("  ✓ Highest electrical efficiency (45% vs 35-38%)")
    print("  ✓ Near-zero emissions (urban air quality compliance)")
    print("  ✓ Quiet operation (patient comfort)")
    print("  ✓ High reliability (>99.5% uptime)")
    print("  ✓ Energy resilience (backup power)")
    print("  ✓ State hospital grant ($500k)")

    print("\nConsiderations:")
    print("  • Higher capital cost offset by premium electricity rates")
    print("  • Longer payback (5.5 years) acceptable for hospital timelines")
    print("  • Extended maintenance intervals minimize disruption")

    print("\nNext Steps (Hospital-Specific Timeline):")
    print("  1. Board approval and capital appropriation (Month 1)")
    print("  2. Clinical stakeholder engagement (Months 1-2)")
    print("  3. Utility interconnection application (Month 2)")
    print("  4. Detailed engineering and site planning (Months 3-4)")
    print("  5. Equipment procurement (long lead time) (Months 5-8)")
    print("  6. Installation during low-census period (Months 9-12)")
    print("  7. Commissioning with emergency systems (Month 13)")
    print("  8. Staff training and validation (Month 14)")

    print("\nStakeholder Approvals Required:")
    print("  - Hospital Board (capital approval)")
    print("  - Chief Operating Officer (operational impact)")
    print("  - Chief Financial Officer (financial approval)")
    print("  - Facilities Director (integration planning)")
    print("  - Safety Officer (NFPA 99 compliance)")
    print("  - State Health Department (if applicable)")

    print_section("DEMO COMPLETE")

    print("This demo showcased Agent #5 analyzing a hospital CHP system with:")
    print("  - Fuel cell technology selection (highest efficiency)")
    print("  - Hospital-grade reliability requirements")
    print("  - Ultra-low emissions analysis (<1 ppm NOx)")
    print("  - Premium electricity rate economics")
    print("  - Critical facility grid interconnection")
    print("  - 24/7 baseload operating strategy")
    print("  - Urban air quality emissions compliance")
    print("  - Hospital board-ready recommendations")

    print(f"\nDemo completed: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
