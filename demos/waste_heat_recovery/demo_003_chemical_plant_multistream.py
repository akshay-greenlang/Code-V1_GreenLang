# -*- coding: utf-8 -*-
"""
Demo Script #3: Chemical Plant Multi-Stream Waste Heat Recovery

Scenario:
- Chemical manufacturing facility with multiple process streams
- Reactor exhaust at 900°F
- Distillation overhead at 250°F
- Cooling water systems at 140°F
- Target: Optimize multi-stream heat integration

Expected Results:
- 8,000+ MMBtu/yr recoverable from multiple streams
- $640,000+ annual savings
- 2.5 year average payback
- 550+ metric tons CO2 reduction/year

Technologies:
- Multiple technology types based on stream characteristics
- Prioritized implementation roadmap
- Heat cascade/pinch analysis consideration
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.waste_heat_recovery_agent_ai import WasteHeatRecoveryAgent_AI
from greenlang.framework import AgentConfig


def main():
    """Run chemical plant multi-stream waste heat recovery demonstration."""

    print("=" * 80)
    print("DEMO #3: Chemical Plant Multi-Stream Waste Heat Recovery")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_chemical_plant_waste_heat",
        temperature=0.0,
        seed=42,
        max_tokens=4000,
    )
    agent = WasteHeatRecoveryAgent_AI(config)

    # Define chemical plant with diverse processes
    facility_data = {
        "facility_type": "chemical_plant",
        "facility_name": "Midwest Specialty Chemicals",
        "location": "Ohio",
        "annual_production": "100 million lbs of specialty chemicals",
        "processes": [
            {
                "process_name": "Reactor A - Exothermic",
                "process_type": "reactor",
                "fuel_input_mmbtu_yr": 25000,
                "fuel_type": "natural_gas",
                "exhaust_temperature_f": 900,
                "exhaust_flow_cfm": 8000,
                "operating_hours_per_year": 8000,
            },
            {
                "process_name": "Distillation Column 1",
                "process_type": "distillation",
                "fuel_input_mmbtu_yr": 18000,
                "fuel_type": "steam",
                "exhaust_temperature_f": 250,
                "exhaust_flow_cfm": 5000,
                "operating_hours_per_year": 8400,
            },
            {
                "process_name": "Distillation Column 2",
                "process_type": "distillation",
                "fuel_input_mmbtu_yr": 15000,
                "fuel_type": "steam",
                "exhaust_temperature_f": 220,
                "exhaust_flow_cfm": 4000,
                "operating_hours_per_year": 8400,
            },
            {
                "process_name": "Cooling Tower System",
                "process_type": "cooling_water",
                "fuel_input_mmbtu_yr": 0,  # Waste heat source only
                "fuel_type": "none",
                "exhaust_temperature_f": 140,
                "exhaust_flow_cfm": 15000,
                "operating_hours_per_year": 8760,
            },
            {
                "process_name": "Air Compressor System",
                "process_type": "compressed_air",
                "fuel_input_mmbtu_yr": 3000,
                "fuel_type": "electricity",
                "exhaust_temperature_f": 180,
                "exhaust_flow_cfm": 3000,
                "operating_hours_per_year": 8400,
            },
        ],
        "total_annual_fuel_mmbtu": 61000,
        "fuel_cost_usd_per_mmbtu": 8.5,
        "electricity_cost_usd_per_kwh": 0.11,
        "steam_cost_usd_per_mlb": 12.0,
        "cooling_water_cost_usd_per_kgal": 2.5,
        "minimum_temperature_f": 130,
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Annual Production: {facility_data['annual_production']}")
    print(f"  Annual Energy Consumption: {facility_data['total_annual_fuel_mmbtu']:,} MMBtu")
    print(f"  Number of Process Streams: {len(facility_data['processes'])}")
    print()

    # Step 1: Comprehensive waste heat audit
    print("STEP 1: Comprehensive Multi-Stream Waste Heat Audit...")
    print("-" * 80)

    waste_heat_analysis = agent._identify_waste_heat_sources_impl(
        facility_type=facility_data["facility_type"],
        processes=facility_data["processes"],
        include_hvac_systems=True,
        include_compressed_air=True,
        minimum_temperature_f=facility_data["minimum_temperature_f"],
    )

    print(f"Total Waste Heat Identified: {waste_heat_analysis['total_waste_heat_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"Technically Recoverable: {waste_heat_analysis['recoverable_waste_heat_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"Economically Recoverable: {waste_heat_analysis['economically_recoverable_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"Number of Sources: {len(waste_heat_analysis['waste_heat_sources'])}")
    print()

    print("Waste Heat Quality Distribution:")
    summary = waste_heat_analysis["waste_heat_summary"]
    print(f"  High-Grade (>400°F): {summary['high_grade_above_400f_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Medium-Grade (200-400°F): {summary['medium_grade_200_400f_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Low-Grade (<200°F): {summary['low_grade_below_200f_mmbtu_yr']:,.0f} MMBtu/year")
    print()

    # Step 2: Analyze all opportunities
    print("STEP 2: Analyzing Individual Recovery Opportunities...")
    print("-" * 80)

    opportunities = []

    for i, source in enumerate(waste_heat_analysis["waste_heat_sources"][:5]):  # Top 5 sources
        print(f"\n{'='*80}")
        print(f"Opportunity {i+1}: {source['source_name']}")
        print(f"{'='*80}")
        print(f"Stream Characteristics:")
        print(f"  Temperature: {source['temperature_f']}°F")
        print(f"  Fluid Type: {source.get('fluid_type', 'mixed')}")
        print(f"  Available Heat: {source['waste_heat_mmbtu_yr']:,.0f} MMBtu/year")
        print(f"  Quality Grade: {source['quality_grade']}")

        # Determine recovery temperature based on quality
        if source['quality_grade'] == "high_grade":
            recovery_temp = 400
            application = "steam_generation"
        elif source['quality_grade'] == "medium_grade":
            recovery_temp = 150
            application = "preheating"
        else:
            recovery_temp = 100
            application = "preheating"

        # Calculate recovery potential
        recovery = agent._calculate_heat_recovery_potential_impl(
            waste_heat_stream={
                "temperature_f": source["temperature_f"],
                "mass_flow_rate_lb_hr": source.get("mass_flow_rate_lb_hr", 8000),
                "fluid_type": source.get("fluid_type", "air"),
            },
            recovery_temperature_f=recovery_temp,
            heat_exchanger_effectiveness=0.75 if source['quality_grade'] == "high_grade" else 0.70,
        )

        print(f"\nRecovery Potential:")
        print(f"  Practical Recovery: {recovery['practical_heat_recovery_mmbtu_yr']:,.0f} MMBtu/year")
        print(f"  Exergy Available: {recovery['exergy_available_mmbtu_yr']:,.0f} MMBtu/year")
        print(f"  Recovery Efficiency: {recovery['heat_exchanger_effectiveness_used']:.0%}")

        # Technology selection
        budget = 150000 if i < 2 else 100000  # Higher budget for top opportunities
        tech_selection = agent._select_heat_recovery_technology_impl(
            waste_heat_stream={
                "temperature_f": source["temperature_f"],
                "fluid_type": source.get("fluid_type", "air"),
                "heat_load_mmbtu_yr": recovery['practical_heat_recovery_mmbtu_yr'],
                "fouling_potential": "moderate" if "reactor" in source['source_name'].lower() else "low",
            },
            application=application,
            budget_usd=budget,
            space_constrained=(i >= 3),  # Later opportunities may be space-constrained
        )

        print(f"\nTechnology Selection:")
        print(f"  Recommended: {tech_selection['recommended_technology']}")
        print(f"  Confidence: {tech_selection['confidence_score']:.1f}/100")

        # Size heat exchanger (simplified approach temperature)
        outlet_temp = recovery['outlet_temperature_f']
        cold_in = recovery_temp - 50
        cold_out = min(source["temperature_f"] - 30, recovery_temp + 50)

        sizing = agent._size_heat_exchanger_impl(
            heat_load_btu_hr=recovery['practical_heat_recovery_mmbtu_yr'] * 1_000_000 / 8000,
            hot_side_in_f=source["temperature_f"],
            hot_side_out_f=outlet_temp,
            cold_side_in_f=cold_in,
            cold_side_out_f=cold_out,
            technology=tech_selection['recommended_technology_key'],
        )

        if "error" not in sizing:
            print(f"  Heat Exchanger Area: {sizing['design_area_ft2']:.0f} ft²")
            print(f"  Capital Cost: ${sizing['estimated_capital_cost_usd']:,.0f}")

            # Energy savings
            savings = agent._calculate_energy_savings_impl(
                recovered_heat_mmbtu_yr=recovery['practical_heat_recovery_mmbtu_yr'],
                displaced_fuel_type="natural_gas",
                fuel_price_usd_per_mmbtu=facility_data["fuel_cost_usd_per_mmbtu"],
                boiler_efficiency=0.82,
                electricity_price_usd_per_kwh=facility_data["electricity_cost_usd_per_kwh"],
            )

            print(f"\nFinancial Metrics:")
            print(f"  Annual Savings: ${savings['net_savings_usd_yr']:,.0f}")
            print(f"  CO2 Reduction: {savings['co2_reduction_metric_tons_yr']:.0f} metric tons/year")

            # Payback
            payback = agent._calculate_payback_period_impl(
                capital_cost_usd=sizing['estimated_capital_cost_usd'],
                annual_savings_usd=savings['net_savings_usd_yr'],
                annual_maintenance_cost_usd=sizing['estimated_capital_cost_usd'] * 0.03,
                project_lifetime_years=20,
                discount_rate=0.10,
            )

            print(f"  Payback Period: {payback['simple_payback_years']:.2f} years")
            print(f"  NPV: ${payback['net_present_value_usd']:,.0f}")
            print(f"  IRR: {payback['internal_rate_of_return_percent']:.1f}%")
            print(f"  Attractiveness: {payback['project_attractiveness'].upper()}")

            # Determine complexity based on integration requirements
            if "reactor" in source['source_name'].lower():
                complexity = "high"
            elif "distillation" in source['source_name'].lower():
                complexity = "moderate"
            else:
                complexity = "low"

            opportunities.append({
                "name": source['source_name'],
                "payback_years": payback['simple_payback_years'],
                "energy_savings_mmbtu_yr": recovery['practical_heat_recovery_mmbtu_yr'],
                "capital_cost_usd": sizing['estimated_capital_cost_usd'],
                "annual_savings_usd": savings['net_savings_usd_yr'],
                "co2_reduction_metric_tons_yr": savings['co2_reduction_metric_tons_yr'],
                "implementation_complexity": complexity,
                "technology": tech_selection['recommended_technology'],
                "quality_grade": source['quality_grade'],
            })
        else:
            print(f"  WARNING: {sizing['error']}")

    # Step 3: Prioritize all opportunities
    print("\n" + "=" * 80)
    print("STEP 3: Multi-Criteria Prioritization and Implementation Roadmap")
    print("=" * 80)

    if len(opportunities) > 0:
        # Custom prioritization for chemical plant (emphasize reliability and quick wins)
        custom_criteria = {
            "payback_weight": 0.35,  # Higher weight on payback
            "energy_savings_weight": 0.20,
            "complexity_weight": 0.25,  # Higher weight on complexity
            "carbon_impact_weight": 0.10,
            "capital_efficiency_weight": 0.10,
        }

        prioritization = agent._prioritize_waste_heat_opportunities_impl(
            opportunities=opportunities,
            prioritization_criteria=custom_criteria,
        )

        print(f"\nPortfolio Summary:")
        print(f"  Total Opportunities: {prioritization['total_opportunities']}")
        print(f"  High Priority: {prioritization['high_priority_count']}")
        print(f"  Total Investment: ${prioritization['total_potential_investment_usd']:,.0f}")
        print(f"  Total Savings: ${prioritization['total_potential_savings_usd_yr']:,.0f}/year")
        print(f"  Total CO2 Reduction: {prioritization['total_carbon_reduction_metric_tons_yr']:.0f} metric tons/year")
        print(f"  Portfolio Average Payback: {prioritization['total_potential_investment_usd']/prioritization['total_potential_savings_usd_yr']:.2f} years")
        print()

        print("Prioritization Criteria (Custom Weighting):")
        criteria = prioritization['prioritization_criteria_used']
        print(f"  Payback Period: {criteria['payback_weight']:.0%}")
        print(f"  Energy Savings: {criteria['energy_savings_weight']:.0%}")
        print(f"  Complexity: {criteria['complexity_weight']:.0%}")
        print(f"  Carbon Impact: {criteria['carbon_impact_weight']:.0%}")
        print(f"  Capital Efficiency: {criteria['capital_efficiency_weight']:.0%}")
        print()

        print("=" * 80)
        print("PRIORITIZED OPPORTUNITIES (Highest to Lowest)")
        print("=" * 80)

        for opp in prioritization['prioritized_opportunities']:
            print(f"\n{opp['opportunity_name']}")
            print(f"  Priority Score: {opp['total_score']:.1f}/100")
            print(f"  Technology: {opp['technology']}")
            print(f"  Capital Cost: ${opp['capital_cost_usd']:,.0f}")
            print(f"  Annual Savings: ${opp['annual_savings_usd']:,.0f}")
            print(f"  Payback: {opp['payback_years']:.2f} years")
            print(f"  Energy Recovery: {opp['energy_savings_mmbtu_yr']:,.0f} MMBtu/year")
            print(f"  CO2 Reduction: {opp['co2_reduction_metric_tons_yr']:.0f} metric tons/year")
            print(f"  Implementation Complexity: {opp['implementation_complexity'].upper()}")

        print("\n" + "=" * 80)
        print("PHASED IMPLEMENTATION ROADMAP")
        print("=" * 80)

        for item in prioritization['implementation_roadmap']:
            print(f"\n{item['implementation_phase']}")
            print(f"  Rank #{item['rank']}: {item['opportunity']}")
            print(f"  Priority Level: {item['priority_level']}")
            print(f"  Phase Investment: See cumulative below")
            print(f"  Cumulative Investment: ${item['cumulative_investment_usd']:,.0f}")
            print(f"  Cumulative Annual Savings: ${item['cumulative_annual_savings_usd']:,.0f}/year")

        # Calculate phase metrics
        phases = {}
        for item in prioritization['implementation_roadmap']:
            phase = item['implementation_phase']
            if phase not in phases:
                phases[phase] = {"count": 0, "investment": 0, "savings": 0}
            phases[phase]["count"] += 1

        print("\n" + "=" * 80)
        print("PHASE SUMMARY")
        print("=" * 80)

        for phase_name in ["Phase 1 (Year 1)", "Phase 2 (Year 2)", "Phase 3 (Year 3+)"]:
            phase_opps = [item for item in prioritization['implementation_roadmap'] if item['implementation_phase'] == phase_name]
            if len(phase_opps) > 0:
                phase_inv = max([item['cumulative_investment_usd'] for item in phase_opps])
                phase_sav = max([item['cumulative_annual_savings_usd'] for item in phase_opps])
                prev_phase_inv = 0
                prev_phase_sav = 0

                if phase_name == "Phase 2 (Year 2)" and len([i for i in prioritization['implementation_roadmap'] if i['implementation_phase'] == "Phase 1 (Year 1)"]) > 0:
                    phase_1_items = [item for item in prioritization['implementation_roadmap'] if item['implementation_phase'] == "Phase 1 (Year 1)"]
                    prev_phase_inv = max([item['cumulative_investment_usd'] for item in phase_1_items])
                    prev_phase_sav = max([item['cumulative_annual_savings_usd'] for item in phase_1_items])
                elif phase_name == "Phase 3 (Year 3+)":
                    phase_2_items = [item for item in prioritization['implementation_roadmap'] if item['implementation_phase'] == "Phase 2 (Year 2)"]
                    if len(phase_2_items) > 0:
                        prev_phase_inv = max([item['cumulative_investment_usd'] for item in phase_2_items])
                        prev_phase_sav = max([item['cumulative_annual_savings_usd'] for item in phase_2_items])

                phase_only_inv = phase_inv - prev_phase_inv
                phase_only_sav = phase_sav - prev_phase_sav

                print(f"\n{phase_name}:")
                print(f"  Number of Projects: {len(phase_opps)}")
                print(f"  Phase Investment: ${phase_only_inv:,.0f}")
                print(f"  Phase Annual Savings: ${phase_only_sav:,.0f}/year")
                if phase_only_sav > 0:
                    print(f"  Phase Payback: {phase_only_inv/phase_only_sav:.2f} years")

    # Final recommendations
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Midwest Specialty Chemicals - Comprehensive Waste Heat Recovery Plan")
    print("-" * 80)
    print(f"✓ Total Opportunities Identified: {len(opportunities)}")
    print(f"✓ Total Recoverable Heat: {sum(o['energy_savings_mmbtu_yr'] for o in opportunities):,.0f} MMBtu/year")
    print(f"✓ Total Investment Required: ${sum(o['capital_cost_usd'] for o in opportunities):,.0f}")
    print(f"✓ Total Annual Savings: ${sum(o['annual_savings_usd'] for o in opportunities):,.0f}")
    print(f"✓ Portfolio Average Payback: {sum(o['capital_cost_usd'] for o in opportunities)/sum(o['annual_savings_usd'] for o in opportunities):.2f} years")
    print(f"✓ Total CO2 Reduction: {sum(o['co2_reduction_metric_tons_yr'] for o in opportunities):.0f} metric tons/year")
    print(f"✓ Percentage of Energy Consumption Recovered: {sum(o['energy_savings_mmbtu_yr'] for o in opportunities)/facility_data['total_annual_fuel_mmbtu']*100:.1f}%")
    print()

    print("STRATEGIC RECOMMENDATIONS:")
    print("1. IMMEDIATE ACTION (Phase 1): Implement top 2 opportunities")
    print(f"   - Focus on best payback projects (<2.5 years)")
    print(f"   - Expected Year 1 savings: ${sum([o['annual_savings_usd'] for o in opportunities[:2]]):,.0f}")
    print()
    print("2. MEDIUM-TERM (Phase 2): Implement next 3 opportunities")
    print(f"   - Include medium-complexity integration projects")
    print(f"   - Leverage Phase 1 experience and cashflow")
    print()
    print("3. LONG-TERM (Phase 3): Consider remaining opportunities")
    print(f"   - Evaluate new technologies and integration approaches")
    print(f"   - Reassess economics with updated energy prices")
    print()
    print("4. INTEGRATION OPPORTUNITIES:")
    print("   - Explore heat cascade between high-grade and low-grade streams")
    print("   - Consider heat pump integration for temperature lift")
    print("   - Evaluate thermal storage for load leveling")
    print()
    print("5. FINANCIAL CONSIDERATIONS:")
    print("   - Investigate utility rebates and incentives (20-30% capital reduction typical)")
    print("   - Consider energy performance contracting for Phase 2/3")
    print("   - Include carbon credits in financial analysis ($51/metric ton)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
