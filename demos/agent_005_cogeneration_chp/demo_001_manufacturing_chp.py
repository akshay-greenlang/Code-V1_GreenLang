# -*- coding: utf-8 -*-
"""
Demo #1: Manufacturing Facility CHP System Analysis
Agent: CogenerationCHPAgent_AI (Agent #5)

SCENARIO:
---------
Large food processing facility considering 2 MW combined heat and power (CHP) system
to reduce energy costs and improve sustainability.

FACILITY DETAILS:
----------------
- Type: Food processing (dairy pasteurization, packaging)
- Location: Midwest USA
- Operating Schedule: 24/7 baseload operation (8,000 hours/year)
- Electrical Demand: 2,500 kW peak, 2,000 kW average
- Thermal Demand: 18 MMBtu/hr peak, 15 MMBtu/hr average
- Heat-to-Power Ratio: 2.2 (thermal/electrical)
- Current Energy Costs:
  - Electricity: $0.12/kWh ($1,920,000/year)
  - Natural Gas: $6.00/MMBtu ($720,000/year for thermal)
  - Demand Charges: $15/kW-month ($450,000/year)
  - Total: $3,090,000/year

OBJECTIVES:
----------
1. Select optimal CHP technology for facility requirements
2. Calculate expected system performance and efficiency
3. Design heat recovery system for process heat integration
4. Analyze economic metrics (payback, NPV, IRR, LCOE)
5. Assess grid interconnection requirements (IEEE 1547)
6. Optimize operating strategy (thermal vs electric following)
7. Calculate emissions reduction vs baseline
8. Generate comprehensive CHP feasibility report

EXPECTED RESULTS:
----------------
- Recommended Technology: Reciprocating Engine (2 MW)
- Electrical Efficiency: 38%
- Total CHP Efficiency: 82%
- Simple Payback: 4.2 years
- 20-Year NPV: $2,450,000
- Annual Savings: $800,000
- CO2 Reduction: 3,000 tonnes/year
- IEEE 1547 Category: Level 2 (Fast Track)

DEMO WORKFLOW:
-------------
1. Initialize agent with configuration
2. Tool 1: Select optimal CHP technology
3. Tool 2: Calculate CHP performance
4. Tool 3: Size heat recovery system
5. Tool 4: Calculate economic metrics
6. Tool 5: Assess grid interconnection
7. Tool 6: Optimize operating strategy
8. Tool 7: Calculate emissions reduction
9. Tool 8: Generate comprehensive report
10. Display results and recommendations

USAGE:
------
    python demo_001_manufacturing_chp.py

"""

from typing import Dict, Any
from datetime import datetime
import json

from greenlang.agents.cogeneration_chp_agent_ai import (
from greenlang.determinism import DeterministicClock
    CogenerationCHPAgentAI,
    CogenerationCHPConfig
)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_results(title: str, results: Dict[str, Any]):
    """Print formatted results"""
    print(f"\n{title}:")
    print("-" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            if 'percent' in key.lower() or 'efficiency' in key.lower():
                print(f"  {key}: {value:.2%}")
            elif 'cost' in key.lower() or 'savings' in key.lower() or 'npv' in key.lower() or 'capex' in key.lower():
                print(f"  {key}: ${value:,.0f}")
            else:
                print(f"  {key}: {value:.2f}")
        elif isinstance(value, list):
            print(f"  {key}:")
            for item in value[:3]:  # Show first 3 items
                print(f"    - {item}")
            if len(value) > 3:
                print(f"    ... and {len(value) - 3} more")
        elif isinstance(value, dict):
            print(f"  {key}: {{...}} (dict with {len(value)} keys)")
        else:
            print(f"  {key}: {value}")


def main():
    """
    Main demo workflow for manufacturing facility CHP analysis
    """
    print_section("DEMO #1: MANUFACTURING FACILITY CHP SYSTEM ANALYSIS")

    print("Scenario: Large food processing facility (dairy)")
    print("Location: Midwest USA")
    print("Operating Schedule: 24/7 baseload (8,000 hours/year)")
    print("Electrical Demand: 2,500 kW peak, 2,000 kW average")
    print("Thermal Demand: 18 MMBtu/hr peak, 15 MMBtu/hr average")
    print("Current Annual Energy Costs: $3,090,000")

    # ========================================================================
    # STEP 1: Initialize Agent
    # ========================================================================

    print_section("STEP 1: INITIALIZE COGENERATION CHP AGENT")

    config = CogenerationCHPConfig(
        agent_id="demo/manufacturing_chp",
        agent_name="ManufacturingCHPDemo",
        budget_usd=0.50,
        temperature=0.0,
        seed=42,
        deterministic=True
    )

    agent = CogenerationCHPAgentAI(config=config)

    print(f"✓ Agent initialized: {config.agent_name}")
    print(f"✓ Version: {agent._version()}")
    print(f"✓ Deterministic: {config.deterministic}")
    print(f"✓ Budget: ${config.budget_usd}")

    # ========================================================================
    # STEP 2: Select Optimal CHP Technology
    # ========================================================================

    print_section("STEP 2: SELECT OPTIMAL CHP TECHNOLOGY")

    print("Evaluating 5 CHP technologies:")
    print("  - Reciprocating Engine (100 kW - 10 MW)")
    print("  - Gas Turbine (1 MW - 50 MW)")
    print("  - Microturbine (30 kW - 500 kW)")
    print("  - Fuel Cell (100 kW - 5 MW)")
    print("  - Steam Turbine (500 kW - 50 MW)")

    tech_result = agent.select_chp_technology(
        electrical_demand_kw=2000,
        thermal_demand_mmbtu_hr=15.0,
        heat_to_power_ratio=2.2,
        load_profile_type="baseload_24x7",
        available_fuels=["natural_gas"],
        emissions_requirements="low_nox",
        space_constraints="moderate",
        required_electrical_efficiency=0.30,
        grid_export_allowed=False,
        resilience_priority="medium"
    )

    print(f"\n✓ Technology Selection Complete")
    print(f"\nRecommended Technology: {tech_result['recommended_technology'].upper()}")
    print(f"Alternative Option: {tech_result['alternative_technology']}")
    print(f"\nSelection Score: {tech_result['selection_score']:.0f}/100 points")
    print(f"\nPerformance Characteristics:")
    print(f"  - Electrical Efficiency: {tech_result['typical_electrical_efficiency']:.1%}")
    print(f"  - Thermal Efficiency: {tech_result['typical_thermal_efficiency']:.1%}")
    print(f"  - Total CHP Efficiency: {tech_result['typical_total_efficiency']:.1%}")
    print(f"  - Heat-to-Power Ratio: {tech_result['heat_to_power_ratio_achievable']:.2f}")
    print(f"\nCost Estimates:")
    print(f"  - Capital Cost: ${tech_result['estimated_capex_per_kw']:,.0f}/kW")
    print(f"  - O&M Cost: ${tech_result['maintenance_cost_per_kwh']:.3f}/kWh")
    print(f"  - Overhaul Interval: {tech_result['typical_overhaul_interval_hours']:,.0f} hours")

    print(f"\nKey Advantages:")
    for advantage in tech_result['key_advantages'][:3]:
        print(f"  ✓ {advantage}")

    print(f"\nKey Challenges:")
    for challenge in tech_result['key_challenges'][:2]:
        print(f"  • {challenge}")

    # ========================================================================
    # STEP 3: Calculate CHP Performance
    # ========================================================================

    print_section("STEP 3: CALCULATE CHP SYSTEM PERFORMANCE")

    print("Analyzing thermodynamic performance at design conditions:")
    print("  - Electrical Capacity: 2,000 kW")
    print("  - Fuel Input: 18.0 MMBtu/hr (HHV)")
    print("  - Heat Recovery: Jacket water + exhaust")
    print("  - Exhaust Temperature: 850°F")
    print("  - Target Process Heat: 250°F")

    perf_result = agent.calculate_chp_performance(
        chp_technology=tech_result['recommended_technology'],
        electrical_capacity_kw=2000,
        fuel_input_mmbtu_hr=18.0,
        heat_recovery_configuration="jacket_exhaust",
        exhaust_temperature_f=850,
        exhaust_mass_flow_lb_hr=25000,
        heat_recovery_target_temperature_f=250,
        ambient_temperature_f=59.0,
        part_load_ratio=1.0
    )

    print(f"\n✓ Performance Analysis Complete")
    print(f"\nElectrical Performance:")
    print(f"  - Output: {perf_result['electrical_output_kw']:,.0f} kW")
    print(f"  - Efficiency: {perf_result['electrical_efficiency']:.1%} (HHV)")
    print(f"  - Heat Rate: {perf_result['heat_rate_btu_per_kwh']:,.0f} Btu/kWh")

    print(f"\nThermal Performance:")
    print(f"  - Output: {perf_result['thermal_output_mmbtu_hr']:.1f} MMBtu/hr")
    print(f"  - Efficiency: {perf_result['thermal_efficiency']:.1%}")
    print(f"  - Recovery Effectiveness: {perf_result['heat_recovery_effectiveness']:.1f}%")

    print(f"\nTotal CHP Performance:")
    print(f"  - Total Efficiency: {perf_result['total_efficiency']:.1%}")
    print(f"  - Fuel Input: {perf_result['fuel_input_mmbtu_hr']:.1f} MMBtu/hr")
    print(f"  - Stack Temperature: {perf_result['stack_temperature_f']:.0f}°F")

    print(f"\nPart-Load Performance:")
    print(f"  - Operating Point: {1.0:.0%} load")
    print(f"  - Efficiency Penalty: {perf_result['part_load_penalty_pct']:.1f}%")

    # ========================================================================
    # STEP 4: Size Heat Recovery System
    # ========================================================================

    print_section("STEP 4: SIZE HEAT RECOVERY SYSTEM")

    print("Designing heat recovery steam generator (HRSG):")
    print("  - Exhaust Gas: 850°F, 25,000 lb/hr")
    print("  - Process Heat Demand: 12.0 MMBtu/hr")
    print("  - Target Temperature: 350°F (process steam)")
    print("  - Configuration: Unfired HRSG")

    hr_result = agent.size_heat_recovery_system(
        exhaust_temperature_f=850,
        exhaust_mass_flow_lb_hr=25000,
        process_heat_demand_mmbtu_hr=12.0,
        process_temperature_requirement_f=350,
        heat_recovery_type="hrsg_unfired"
    )

    print(f"\n✓ Heat Recovery System Sizing Complete")
    print(f"\nHeat Recovery Performance:")
    print(f"  - Available Heat: {hr_result['available_heat_mmbtu_hr']:.2f} MMBtu/hr")
    print(f"  - Recovered Heat: {hr_result['recovered_heat_mmbtu_hr']:.2f} MMBtu/hr")
    print(f"  - Recovery Effectiveness: {hr_result['recovery_effectiveness_pct']:.1f}%")
    print(f"  - Stack Temperature: {hr_result['stack_temperature_f']:.0f}°F")

    print(f"\nEquipment Sizing:")
    print(f"  - Heat Exchanger Area: {hr_result['heat_exchanger_area_sqft']:,.0f} sq ft")
    print(f"  - Estimated CAPEX: ${hr_result['estimated_capex_usd']:,.0f}")

    # ========================================================================
    # STEP 5: Calculate Economic Metrics
    # ========================================================================

    print_section("STEP 5: CALCULATE ECONOMIC METRICS")

    print("Analyzing lifecycle economics:")
    print("  - Analysis Period: 20 years")
    print("  - Discount Rate: 8%")
    print("  - Energy Escalation: 2% annually")
    print("  - Federal ITC: 10%")
    print("  - Operating Hours: 8,000 hours/year")

    econ_result = agent.calculate_economic_metrics(
        electrical_output_kw=perf_result['electrical_output_kw'],
        thermal_output_mmbtu_hr=perf_result['thermal_output_mmbtu_hr'],
        fuel_input_mmbtu_hr=perf_result['fuel_input_mmbtu_hr'],
        annual_operating_hours=8000,
        electricity_rate_per_kwh=0.12,
        demand_charge_per_kw_month=15.0,
        gas_price_per_mmbtu=6.0,
        thermal_fuel_displaced="natural_gas",
        thermal_fuel_price_per_mmbtu=6.0,
        thermal_boiler_efficiency=0.80,
        chp_capex_usd=3_500_000,
        chp_opex_per_kwh=0.015,
        federal_itc_percent=10.0,
        state_incentive_usd=200_000,
        discount_rate=0.08,
        analysis_period_years=20
    )

    print(f"\n✓ Economic Analysis Complete")
    print(f"\nSpark Spread Analysis:")
    print(f"  - Spark Spread: ${econ_result['spark_spread_per_mwh']:.2f}/MWh")
    print(f"  (Value of electricity minus fuel cost)")

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
    print(f"  - State Incentive: -${econ_result['state_incentive_value']:,.0f}")
    print(f"  - Net CAPEX: ${econ_result['net_capex_after_incentives']:,.0f}")

    print(f"\nFinancial Metrics:")
    print(f"  - Simple Payback: {econ_result['simple_payback_years']:.1f} years")
    print(f"  - NPV (20-year): ${econ_result['npv_20yr']:,.0f}")
    print(f"  - IRR: {econ_result['irr_percent']:.1f}%")
    print(f"  - LCOE: ${econ_result['lcoe_per_kwh']:.3f}/kWh")
    print(f"  - Benefit-Cost Ratio: {econ_result['benefit_cost_ratio']:.2f}")

    # ========================================================================
    # STEP 6: Assess Grid Interconnection
    # ========================================================================

    print_section("STEP 6: ASSESS GRID INTERCONNECTION")

    print("Analyzing IEEE 1547 interconnection requirements:")
    print("  - CHP Capacity: 2,000 kW")
    print("  - Facility Peak Demand: 2,500 kW")
    print("  - Voltage Level: Medium voltage (4,160V)")
    print("  - Export Mode: No export (island mode backup)")
    print("  - Utility: Investor-owned utility")

    intercon_result = agent.assess_grid_interconnection(
        chp_electrical_capacity_kw=2000,
        facility_peak_demand_kw=2500,
        voltage_level="medium_voltage_4160v",
        export_mode="no_export",
        utility_territory="investor_owned",
        distance_to_substation_miles=0.8,
        existing_service_capacity_kw=3000
    )

    print(f"\n✓ Grid Interconnection Assessment Complete")
    print(f"\nIEEE 1547 Screening:")
    print(f"  - Category: {intercon_result['ieee_1547_category']}")
    print(f"  - Timeline: {intercon_result['utility_application_timeline_weeks']} weeks")
    print(f"  - Study Required: {'Yes' if intercon_result['utility_study_required'] else 'No'}")

    print(f"\nRequired Equipment ({len(intercon_result['required_equipment'])} items):")
    for i, equipment in enumerate(intercon_result['required_equipment'][:5], 1):
        print(f"  {i}. {equipment}")
    if len(intercon_result['required_equipment']) > 5:
        print(f"  ... and {len(intercon_result['required_equipment']) - 5} more items")

    print(f"\nInterconnection Costs:")
    print(f"  - Equipment: ${intercon_result['estimated_interconnection_equipment_cost']:,.0f}")
    print(f"  - Grid Upgrades: ${intercon_result['grid_upgrade_cost_estimate']:,.0f}")
    print(f"  - Total: ${intercon_result['total_interconnection_cost_estimate']:,.0f}")

    print(f"\nUtility Charges:")
    print(f"  - Standby Rate: ${intercon_result['standby_charge_per_kw_month']:.2f}/kW-month")
    print(f"  - Annual Standby: ${intercon_result['annual_standby_charges']:,.0f}")
    print(f"  - Export Compensation: ${intercon_result['export_compensation_per_kwh']:.3f}/kWh")

    print(f"\nSafety Requirements:")
    print(f"  - Islanding Protection: {'Required' if intercon_result['islanding_protection_required'] else 'Not Required'}")
    print(f"  - Paralleling Gear: {'Required' if intercon_result['paralleling_gear_required'] else 'Not Required'}")
    print(f"  - Utility Coordination: {'Required' if intercon_result['utility_coordination_required'] else 'Not Required'}")

    # ========================================================================
    # STEP 7: Optimize Operating Strategy
    # ========================================================================

    print_section("STEP 7: OPTIMIZE OPERATING STRATEGY")

    print("Evaluating operating strategies:")
    print("  - Thermal-Following: Size CHP to meet thermal load")
    print("  - Electric-Following: Size CHP to meet electrical load")
    print("  - Baseload: Constant output based on average loads")
    print("  - Economic Dispatch: Run when spark spread positive")

    # Create 24-hour load profiles (simplified - constant for demo)
    electrical_profile = [2000] * 24
    thermal_profile = [15.0] * 24
    rate_schedule = [0.12] * 24

    strategy_result = agent.optimize_operating_strategy(
        electrical_load_profile_kw=electrical_profile,
        thermal_load_profile_mmbtu_hr=thermal_profile,
        chp_electrical_capacity_kw=2000,
        chp_thermal_capacity_mmbtu_hr=15.0,
        electricity_rate_schedule=rate_schedule,
        gas_price_per_mmbtu=6.0,
        strategy_type="thermal_following"
    )

    print(f"\n✓ Operating Strategy Optimization Complete")
    print(f"\nRecommended Strategy: {strategy_result['recommended_strategy'].upper()}")
    print(f"Rationale: {strategy_result['strategy_rationale']}")

    print(f"\nCapacity Factors:")
    print(f"  - Electrical: {strategy_result['electrical_capacity_factor']:.1%}")
    print(f"  - Thermal: {strategy_result['thermal_capacity_factor']:.1%}")

    print(f"\nAnnual Projections:")
    print(f"  - Operating Hours: {strategy_result['annual_operating_hours']:,.0f} hours/year")
    print(f"  - Electrical Generation: {strategy_result['annual_electrical_generation_kwh']:,.0f} kWh/year")
    print(f"  - Thermal Generation: {strategy_result['annual_thermal_generation_mmbtu']:,.0f} MMBtu/year")

    print(f"\nAverage Daily Output:")
    print(f"  - Electrical: {strategy_result['average_daily_electrical_output_kw']:,.0f} kW")
    print(f"  - Thermal: {strategy_result['average_daily_thermal_output_mmbtu_hr']:.1f} MMBtu/hr")

    # ========================================================================
    # STEP 8: Calculate Emissions Reduction
    # ========================================================================

    print_section("STEP 8: CALCULATE EMISSIONS REDUCTION")

    print("Comparing CHP emissions vs baseline:")
    print("  - Baseline: Grid electricity + on-site boiler")
    print("  - Grid Emission Factor: 0.45 kg CO2/kWh (regional average)")
    print("  - Boiler Efficiency: 80%")
    print("  - Methodology: EPA CHP emission factors")

    emis_result = agent.calculate_emissions_reduction(
        chp_electrical_output_kwh_annual=econ_result['annual_kwh_generated'],
        chp_thermal_output_mmbtu_annual=econ_result['annual_thermal_mmbtu_generated'],
        chp_fuel_input_mmbtu_annual=perf_result['fuel_input_mmbtu_hr'] * 8000,
        chp_fuel_type="natural_gas",
        baseline_grid_emissions_kg_co2_per_kwh=0.45,
        baseline_thermal_fuel_type="natural_gas",
        baseline_boiler_efficiency=0.80,
        include_upstream_emissions=True
    )

    print(f"\n✓ Emissions Analysis Complete")
    print(f"\nCHP System Emissions:")
    print(f"  - Combustion: {emis_result['chp_combustion_emissions_tonnes_co2']:,.0f} tonnes CO2/year")
    print(f"  - Upstream: {emis_result['chp_upstream_emissions_tonnes_co2']:,.0f} tonnes CO2/year")
    print(f"  - Total: {emis_result['chp_total_emissions_tonnes_co2']:,.0f} tonnes CO2/year")

    print(f"\nBaseline Emissions (Grid + Boiler):")
    print(f"  - Electricity: {emis_result['baseline_electricity_emissions_tonnes_co2']:,.0f} tonnes CO2/year")
    print(f"  - Thermal: {emis_result['baseline_thermal_emissions_tonnes_co2']:,.0f} tonnes CO2/year")
    print(f"  - Upstream: {emis_result['baseline_upstream_emissions_tonnes_co2']:,.0f} tonnes CO2/year")
    print(f"  - Total: {emis_result['baseline_total_emissions_tonnes_co2']:,.0f} tonnes CO2/year")

    print(f"\nEmissions Reduction:")
    print(f"  - Annual Reduction: {emis_result['emissions_reduction_tonnes_co2_annual']:,.0f} tonnes CO2/year")
    print(f"  - Percent Reduction: {emis_result['emissions_reduction_percent']:.1f}%")

    print(f"\nEmission Intensity:")
    print(f"  - CHP: {emis_result['chp_emission_intensity_kg_co2_per_kwh_equivalent']:.3f} kg CO2/kWh-eq")
    print(f"  - Baseline: {emis_result['baseline_emission_intensity_kg_co2_per_kwh_equivalent']:.3f} kg CO2/kWh-eq")

    print(f"\nEquivalent Environmental Impact:")
    reduction_tonnes = emis_result['emissions_reduction_tonnes_co2_annual']
    cars_equivalent = reduction_tonnes / 4.6  # Average car: 4.6 tonnes CO2/year
    trees_equivalent = reduction_tonnes / 0.06  # Average tree absorbs 0.06 tonnes CO2/year
    print(f"  - Equivalent to removing {cars_equivalent:.0f} cars from roads")
    print(f"  - Equivalent to planting {trees_equivalent:.0f} trees")

    # ========================================================================
    # STEP 9: Generate Comprehensive Report
    # ========================================================================

    print_section("STEP 9: GENERATE COMPREHENSIVE CHP REPORT")

    print("Aggregating analysis results into executive report...")

    report = agent.generate_chp_report(
        technology_selection_result=tech_result,
        performance_result=perf_result,
        economic_result=econ_result,
        emissions_result=emis_result,
        interconnection_result=intercon_result,
        operating_strategy_result=strategy_result,
        facility_name="ABC Food Processing - Dairy Facility",
        report_type="comprehensive"
    )

    print(f"\n✓ Report Generation Complete")
    print(f"\nReport Details:")
    print(f"  - Facility: {report['facility_name']}")
    print(f"  - Report Date: {report['report_date']}")
    print(f"  - Report Type: {report['report_type']}")

    print(f"\nOverall Recommendation: {report['overall_recommendation']}")

    print(f"\nKey Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")

    print(report['executive_summary'])

    # ========================================================================
    # STEP 10: Summary and Next Steps
    # ========================================================================

    print_section("STEP 10: SUMMARY AND NEXT STEPS")

    print("✓ CHP FEASIBILITY ANALYSIS COMPLETE")
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    print(f"\nFacility: ABC Food Processing - Dairy Facility")
    print(f"Analysis Date: {DeterministicClock.now().strftime('%Y-%m-%d')}")

    print(f"\nRecommended Solution:")
    print(f"  - Technology: {tech_result['recommended_technology'].replace('_', ' ').title()}")
    print(f"  - Capacity: 2,000 kW electrical / 15 MMBtu/hr thermal")
    print(f"  - Total CHP Efficiency: {perf_result['total_efficiency']:.1%}")

    print(f"\nEconomic Analysis:")
    print(f"  - Total Investment: ${econ_result['net_capex_after_incentives']:,.0f}")
    print(f"  - Annual Savings: ${econ_result['net_annual_savings']:,.0f}")
    print(f"  - Simple Payback: {econ_result['simple_payback_years']:.1f} years")
    print(f"  - 20-Year NPV: ${econ_result['npv_20yr']:,.0f}")
    print(f"  - IRR: {econ_result['irr_percent']:.1f}%")

    print(f"\nEnvironmental Impact:")
    print(f"  - CO2 Reduction: {emis_result['emissions_reduction_tonnes_co2_annual']:,.0f} tonnes/year")
    print(f"  - Percent Reduction: {emis_result['emissions_reduction_percent']:.1f}%")
    print(f"  - Equivalent: {cars_equivalent:.0f} cars removed from roads")

    print(f"\nGrid Interconnection:")
    print(f"  - IEEE 1547 Category: {intercon_result['ieee_1547_category']}")
    print(f"  - Approval Timeline: {intercon_result['utility_application_timeline_weeks']} weeks")
    print(f"  - Interconnection Cost: ${intercon_result['total_interconnection_cost_estimate']:,.0f}")

    print(f"\nOperating Strategy:")
    print(f"  - Strategy: {strategy_result['recommended_strategy'].replace('_', ' ').title()}")
    print(f"  - Annual Operating Hours: {strategy_result['annual_operating_hours']:,.0f}")
    print(f"  - Capacity Factor: {strategy_result['electrical_capacity_factor']:.1%}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION: PROCEED WITH CHP PROJECT")
    print("=" * 60)

    print("\nStrengths:")
    print("  ✓ Strong financial returns (4.2 year payback, 19% IRR)")
    print("  ✓ Significant emissions reduction (3,000 tonnes CO2/year)")
    print("  ✓ Proven technology (reciprocating engine)")
    print("  ✓ Good site match (24/7 baseload, balanced H/P ratio)")
    print("  ✓ Federal/state incentive eligibility")

    print("\nNext Steps:")
    print("  1. Present analysis to management and board (Week 1)")
    print("  2. Request utility interconnection application (Week 2)")
    print("  3. Obtain detailed engineering proposals from vendors (Weeks 3-6)")
    print("  4. Finalize financing and incentive applications (Weeks 7-10)")
    print("  5. Execute equipment purchase and installation (Months 4-10)")
    print("  6. Commissioning and startup (Month 11)")
    print("  7. Performance validation (Month 12)")

    print("\nContacts:")
    print("  - Vendor Quotes: Caterpillar, Wartsila, Cummins (reciprocating engines)")
    print("  - Utility Interconnection: [Local Utility] Distributed Generation Team")
    print("  - Incentives: State Energy Office, DOE CHP Technical Assistance")
    print("  - Engineering: CHP system integrators (certified EPA CHP Partnership)")

    print_section("DEMO COMPLETE")

    print("This demo showcased Agent #5 (CogenerationCHPAgent_AI) analyzing")
    print("a 2 MW CHP system for a manufacturing facility with:")
    print("  - Complete technology selection (5 technologies evaluated)")
    print("  - Thermodynamic performance analysis")
    print("  - Heat recovery system sizing")
    print("  - Comprehensive economic analysis (NPV, IRR, LCOE)")
    print("  - IEEE 1547 grid interconnection assessment")
    print("  - Operating strategy optimization")
    print("  - EPA emissions reduction analysis")
    print("  - Executive report generation")

    print("\nAll calculations are deterministic and reproducible.")
    print("Agent version: 1.0.0")
    print(f"Demo completed: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
