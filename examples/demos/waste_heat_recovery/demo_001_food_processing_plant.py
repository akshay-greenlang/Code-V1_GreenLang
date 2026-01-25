# -*- coding: utf-8 -*-
"""
Demo Script #1: Food Processing Plant Waste Heat Recovery Audit

Scenario:
- Mid-size food processing facility
- Steam boiler flue gas at 500°F
- Pasteurization hot water waste at 160°F
- Target: Identify waste heat recovery opportunities with <2 year payback

Expected Results:
- 2,000+ MMBtu/yr recoverable waste heat
- $160,000+ annual savings
- 1.2 year payback
- 140+ metric tons CO2 reduction/year

Technologies:
- Economizer for boiler flue gas
- Plate heat exchanger for pasteurization waste water
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.waste_heat_recovery_agent_ai import WasteHeatRecoveryAgent_AI
from greenlang.framework import AgentConfig
import json


def main():
    """Run food processing plant waste heat recovery demonstration."""

    print("=" * 80)
    print("DEMO #1: Food Processing Plant Waste Heat Recovery Audit")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_food_processing_waste_heat",
        temperature=0.0,
        seed=42,
        max_tokens=4000,
    )
    agent = WasteHeatRecoveryAgent_AI(config)

    # Define facility
    facility_data = {
        "facility_type": "food_processing",
        "facility_name": "Midwest Food Processing Co.",
        "location": "Iowa",
        "annual_production": "50 million lbs of packaged food products",
        "processes": [
            {
                "process_name": "Steam Boiler #1",
                "process_type": "boiler",
                "fuel_input_mmbtu_yr": 15000,
                "fuel_type": "natural_gas",
                "exhaust_temperature_f": 500,
                "exhaust_flow_cfm": 6000,
                "operating_hours_per_year": 7884,  # 90% uptime
            },
            {
                "process_name": "Pasteurization System",
                "process_type": "hot_water_system",
                "fuel_input_mmbtu_yr": 5000,
                "fuel_type": "natural_gas",
                "exhaust_temperature_f": 160,
                "exhaust_flow_cfm": 2000,
                "operating_hours_per_year": 7884,
            },
            {
                "process_name": "Oven Exhaust",
                "process_type": "oven",
                "fuel_input_mmbtu_yr": 3000,
                "fuel_type": "natural_gas",
                "exhaust_temperature_f": 600,
                "exhaust_flow_cfm": 2500,
                "operating_hours_per_year": 7884,
            },
        ],
        "total_annual_fuel_mmbtu": 23000,
        "fuel_cost_usd_per_mmbtu": 8.0,
        "electricity_cost_usd_per_kwh": 0.12,
        "include_hvac_systems": True,
        "include_compressed_air": True,
        "minimum_temperature_f": 140,
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type']}")
    print(f"  Annual Fuel Consumption: {facility_data['total_annual_fuel_mmbtu']:,} MMBtu")
    print(f"  Fuel Cost: ${facility_data['fuel_cost_usd_per_mmbtu']}/MMBtu")
    print(f"  Number of Processes: {len(facility_data['processes'])}")
    print()

    # Step 1: Identify waste heat sources
    print("STEP 1: Identifying Waste Heat Sources...")
    print("-" * 80)

    waste_heat_analysis = agent._identify_waste_heat_sources_impl(
        facility_type=facility_data["facility_type"],
        processes=facility_data["processes"],
        include_hvac_systems=facility_data["include_hvac_systems"],
        include_compressed_air=facility_data["include_compressed_air"],
        minimum_temperature_f=facility_data["minimum_temperature_f"],
    )

    print(f"Total Waste Heat Identified: {waste_heat_analysis['total_waste_heat_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"Technically Recoverable: {waste_heat_analysis['recoverable_waste_heat_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"Economically Recoverable: {waste_heat_analysis['economically_recoverable_mmbtu_yr']:,.0f} MMBtu/year")
    print()

    print("Waste Heat Quality Distribution:")
    summary = waste_heat_analysis["waste_heat_summary"]
    print(f"  High-Grade (>400°F): {summary['high_grade_above_400f_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Medium-Grade (200-400°F): {summary['medium_grade_200_400f_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Low-Grade (<200°F): {summary['low_grade_below_200f_mmbtu_yr']:,.0f} MMBtu/year")
    print()

    # Step 2: Analyze each waste heat source
    print("STEP 2: Analyzing Recovery Opportunities...")
    print("-" * 80)

    opportunities = []

    for i, source in enumerate(waste_heat_analysis["waste_heat_sources"][:3]):  # Top 3 sources
        print(f"\nOpportunity {i+1}: {source['source_name']}")
        print(f"  Temperature: {source['temperature_f']}°F")
        print(f"  Available Heat: {source['waste_heat_mmbtu_yr']:,.0f} MMBtu/year")

        # Calculate recovery potential
        recovery = agent._calculate_heat_recovery_potential_impl(
            waste_heat_stream={
                "temperature_f": source["temperature_f"],
                "mass_flow_rate_lb_hr": source.get("mass_flow_rate_lb_hr", 5000),
                "fluid_type": source.get("fluid_type", "air"),
            },
            recovery_temperature_f=max(200, facility_data["minimum_temperature_f"]),
            heat_exchanger_effectiveness=0.75,
        )

        print(f"  Practical Recovery: {recovery['practical_heat_recovery_mmbtu_yr']:,.0f} MMBtu/year")
        print(f"  Outlet Temperature: {recovery['outlet_temperature_f']}°F")

        # Select technology
        tech_selection = agent._select_heat_recovery_technology_impl(
            waste_heat_stream={
                "temperature_f": source["temperature_f"],
                "fluid_type": source.get("fluid_type", "air"),
                "heat_load_mmbtu_yr": recovery['practical_heat_recovery_mmbtu_yr'],
                "fouling_potential": "moderate",
            },
            application="preheating",
            budget_usd=200000,
            space_constrained=False,
        )

        print(f"  Recommended Technology: {tech_selection['recommended_technology']}")
        print(f"  Confidence Score: {tech_selection['confidence_score']:.1f}/100")

        # Size heat exchanger
        sizing = agent._size_heat_exchanger_impl(
            heat_load_btu_hr=recovery['practical_heat_recovery_mmbtu_yr'] * 1_000_000 / 7884,
            hot_side_in_f=source["temperature_f"],
            hot_side_out_f=recovery['outlet_temperature_f'],
            cold_side_in_f=100,
            cold_side_out_f=180,
            technology=tech_selection['recommended_technology_key'],
        )

        print(f"  Heat Exchanger Area: {sizing['design_area_ft2']:.1f} ft²")
        print(f"  Capital Cost Estimate: ${sizing['estimated_capital_cost_usd']:,.0f}")

        # Calculate savings
        savings = agent._calculate_energy_savings_impl(
            recovered_heat_mmbtu_yr=recovery['practical_heat_recovery_mmbtu_yr'],
            displaced_fuel_type="natural_gas",
            fuel_price_usd_per_mmbtu=facility_data["fuel_cost_usd_per_mmbtu"],
            boiler_efficiency=0.82,
            electricity_price_usd_per_kwh=facility_data["electricity_cost_usd_per_kwh"],
        )

        print(f"  Annual Savings: ${savings['net_savings_usd_yr']:,.0f}")
        print(f"  CO2 Reduction: {savings['co2_reduction_metric_tons_yr']:.1f} metric tons/year")

        # Calculate payback
        payback = agent._calculate_payback_period_impl(
            capital_cost_usd=sizing['estimated_capital_cost_usd'],
            annual_savings_usd=savings['net_savings_usd_yr'],
            annual_maintenance_cost_usd=sizing['estimated_capital_cost_usd'] * 0.03,
            project_lifetime_years=20,
            discount_rate=0.08,
        )

        print(f"  Simple Payback: {payback['simple_payback_years']:.2f} years")
        print(f"  NPV (20 years): ${payback['net_present_value_usd']:,.0f}")
        print(f"  IRR: {payback['internal_rate_of_return_percent']:.1f}%")
        print(f"  Project Attractiveness: {payback['project_attractiveness'].upper()}")

        opportunities.append({
            "name": source['source_name'],
            "payback_years": payback['simple_payback_years'],
            "energy_savings_mmbtu_yr": recovery['practical_heat_recovery_mmbtu_yr'],
            "capital_cost_usd": sizing['estimated_capital_cost_usd'],
            "annual_savings_usd": savings['net_savings_usd_yr'],
            "co2_reduction_metric_tons_yr": savings['co2_reduction_metric_tons_yr'],
            "implementation_complexity": "moderate" if i == 0 else "low",
            "technology": tech_selection['recommended_technology'],
        })

    # Step 3: Prioritize opportunities
    print("\n" + "=" * 80)
    print("STEP 3: Prioritizing Waste Heat Recovery Opportunities")
    print("=" * 80)

    prioritization = agent._prioritize_waste_heat_opportunities_impl(
        opportunities=opportunities
    )

    print(f"\nTotal Opportunities Analyzed: {prioritization['total_opportunities']}")
    print(f"High Priority Opportunities: {prioritization['high_priority_count']}")
    print(f"Total Potential Investment: ${prioritization['total_potential_investment_usd']:,.0f}")
    print(f"Total Potential Savings: ${prioritization['total_potential_savings_usd_yr']:,.0f}/year")
    print(f"Total CO2 Reduction: {prioritization['total_carbon_reduction_metric_tons_yr']:.0f} metric tons/year")
    print()

    print("IMPLEMENTATION ROADMAP:")
    print("-" * 80)
    for item in prioritization['implementation_roadmap']:
        print(f"\nRank #{item['rank']}: {item['opportunity']}")
        print(f"  Phase: {item['implementation_phase']}")
        print(f"  Priority: {item['priority_level']}")
        print(f"  Cumulative Investment: ${item['cumulative_investment_usd']:,.0f}")
        print(f"  Cumulative Savings: ${item['cumulative_annual_savings_usd']:,.0f}/year")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Identified {len(opportunities)} high-value waste heat recovery opportunities")
    print(f"✓ Total recoverable waste heat: {sum(o['energy_savings_mmbtu_yr'] for o in opportunities):,.0f} MMBtu/year")
    print(f"✓ Total capital required: ${sum(o['capital_cost_usd'] for o in opportunities):,.0f}")
    print(f"✓ Total annual savings: ${sum(o['annual_savings_usd'] for o in opportunities):,.0f}")
    print(f"✓ Average payback period: {sum(o['payback_years'] for o in opportunities)/len(opportunities):.2f} years")
    print(f"✓ Total carbon reduction: {sum(o['co2_reduction_metric_tons_yr'] for o in opportunities):.0f} metric tons/year")
    print(f"✓ Equivalent to removing {sum(o['co2_reduction_metric_tons_yr'] for o in opportunities)/4.6:.0f} cars from the road")
    print()
    print("RECOMMENDATION: Proceed with Phase 1 implementation (top 2 opportunities)")
    print("=" * 80)


if __name__ == "__main__":
    main()
