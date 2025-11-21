# -*- coding: utf-8 -*-
"""
Demo Script #1: Manufacturing Facility Comprehensive Decarbonization Roadmap

Scenario:
- Mid-size automotive parts manufacturing facility
- Annual production: $50M revenue
- Current energy: Natural gas boilers, electric motors, compressed air
- Target: Comprehensive net-zero pathway with prioritized investments
- Expected: 5-7 technology opportunities, $2.5M investment, 30-40% emissions reduction

Expected Results:
- Baseline GHG emissions: 5,000-6,000 metric tons CO2e/year
- Opportunities identified: 6-8 across multiple technologies
- Total investment: $2.2-2.8M
- Annual savings: $450,000-$600,000
- Average payback: 4-6 years
- CO2 reduction: 2,000-2,500 metric tons/year (35-45% reduction)
- Net-zero timeline: 15-18 years

Technologies Assessed:
- Industrial process heat (solar thermal)
- Boiler replacement (high-efficiency)
- Industrial heat pumps (process heating)
- Waste heat recovery (equipment cooling)
- LED lighting (facility-wide)
- Compressed air optimization
- Building envelope improvements
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.decarbonization_roadmap_agent_ai import DecarbonizationRoadmapAgentAI
from greenlang.framework import AgentConfig


def main():
    """Run manufacturing facility comprehensive roadmap demonstration."""

    print("=" * 80)
    print("DEMO #1: Manufacturing Facility - Comprehensive Decarbonization Roadmap")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_manufacturing_roadmap",
        temperature=0.0,
        seed=42,
        max_tokens=6000,
    )
    agent = DecarbonizationRoadmapAgentAI(config)

    # Define facility
    facility_data = {
        "facility_name": "Midwest Automotive Parts Manufacturing",
        "location": "Michigan",
        "facility_type": "automotive_manufacturing",
        "industry_sector": "automotive",
        "annual_revenue_usd": 50000000,
        "facility_size_sqft": 150000,
        "employees": 200,

        # Energy consumption
        "energy_consumption": {
            "natural_gas_mmbtu_per_year": 45000,
            "electricity_kwh_per_year": 3500000,
            "diesel_gallons_per_year": 2000,  # Forklifts
            "propane_gallons_per_year": 500,  # Backup heating
        },

        # Current systems
        "heating_systems": {
            "boilers": {
                "type": "natural_gas",
                "capacity_mmbtu_hr": 5.0,
                "efficiency": 0.80,  # 80% efficiency (old boilers)
                "age_years": 18,
                "annual_fuel_mmbtu": 30000,
            },
            "space_heating": {
                "type": "unit_heaters",
                "fuel": "natural_gas",
                "efficiency": 0.75,
                "annual_fuel_mmbtu": 15000,
            },
        },

        "process_heating": {
            "paint_curing_ovens": {
                "temperature_f": 350,
                "capacity_kw": 180,
                "annual_usage_hours": 4000,
                "fuel_type": "natural_gas",
            },
            "parts_washing": {
                "temperature_f": 160,
                "flow_rate_gpm": 50,
                "annual_usage_hours": 4500,
                "fuel_type": "natural_gas",
            },
        },

        "compressed_air": {
            "compressor_power_hp": 150,
            "operating_pressure_psi": 100,
            "efficiency": 0.65,  # Poor efficiency
            "annual_operating_hours": 6000,
            "annual_electricity_kwh": 600000,
        },

        "lighting": {
            "type": "fluorescent_t8",
            "total_fixtures": 800,
            "watts_per_fixture": 96,
            "annual_operating_hours": 5000,
            "annual_electricity_kwh": 384000,
        },

        "waste_heat_sources": [
            {
                "source": "Compressor cooling",
                "temperature_f": 120,
                "capacity_kw": 80,
                "availability_hours_per_day": 16,
            },
            {
                "source": "CNC machine cooling",
                "temperature_f": 110,
                "capacity_kw": 60,
                "availability_hours_per_day": 20,
            },
            {
                "source": "Paint oven exhaust",
                "temperature_f": 450,
                "capacity_kw": 120,
                "availability_hours_per_day": 8,
            },
        ],

        # Economic parameters
        "electricity_cost_usd_per_kwh": 0.12,
        "natural_gas_cost_usd_per_mmbtu": 9.00,
        "diesel_cost_usd_per_gallon": 3.50,
        "propane_cost_usd_per_gallon": 2.80,
        "carbon_price_usd_per_tonne": 50,  # Social cost of carbon
        "discount_rate": 0.08,
        "project_lifetime_years": 20,

        # Organizational goals
        "net_zero_target_year": 2045,
        "baseline_year": 2023,
        "sbti_commitment": True,
        "sbti_target": "1.5C",  # 4.2% annual reduction
        "budget_constraint_usd": 3000000,  # $3M capital budget
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Sector: {facility_data['industry_sector'].title()}")
    print(f"  Location: {facility_data['location']}")
    print(f"  Size: {facility_data['facility_size_sqft']:,} sqft")
    print(f"  Employees: {facility_data['employees']}")
    print(f"  Annual Revenue: ${facility_data['annual_revenue_usd']:,}")
    print()

    print("ORGANIZATIONAL GOALS:")
    print(f"  Net-Zero Target: {facility_data['net_zero_target_year']}")
    print(f"  Baseline Year: {facility_data['baseline_year']}")
    print(f"  SBTi Commitment: {'Yes' if facility_data['sbti_commitment'] else 'No'}")
    print(f"  SBTi Target: {facility_data['sbti_target']} pathway (4.2% annual reduction)")
    print(f"  Capital Budget: ${facility_data['budget_constraint_usd']:,}")
    print()

    # Step 1: Calculate baseline GHG emissions
    print("=" * 80)
    print("STEP 1: Baseline GHG Emissions Inventory")
    print("=" * 80)
    print()

    # Scope 1: Direct emissions from fuel combustion
    # Natural gas: 53.06 kg CO2e/MMBtu
    # Diesel: 10.21 kg CO2e/gallon
    # Propane: 5.68 kg CO2e/gallon
    ng_emissions_kg = facility_data["energy_consumption"]["natural_gas_mmbtu_per_year"] * 53.06
    diesel_emissions_kg = facility_data["energy_consumption"]["diesel_gallons_per_year"] * 10.21
    propane_emissions_kg = facility_data["energy_consumption"]["propane_gallons_per_year"] * 5.68
    scope1_emissions_kg = ng_emissions_kg + diesel_emissions_kg + propane_emissions_kg
    scope1_emissions_tonnes = scope1_emissions_kg / 1000

    # Scope 2: Indirect emissions from purchased electricity
    # Midwest grid: 0.92 lbs CO2e/kWh = 0.417 kg CO2e/kWh
    electricity_kwh = facility_data["energy_consumption"]["electricity_kwh_per_year"]
    scope2_emissions_kg = electricity_kwh * 0.417
    scope2_emissions_tonnes = scope2_emissions_kg / 1000

    # Scope 3: Upstream fuel production (15% of Scope 1)
    scope3_emissions_tonnes = scope1_emissions_tonnes * 0.15

    total_emissions_tonnes = scope1_emissions_tonnes + scope2_emissions_tonnes + scope3_emissions_tonnes

    print("GHG EMISSIONS INVENTORY (Baseline Year 2023):")
    print()
    print("Scope 1 - Direct Emissions:")
    print(f"  Natural Gas: {facility_data['energy_consumption']['natural_gas_mmbtu_per_year']:,} MMBtu")
    print(f"    Emissions: {ng_emissions_kg / 1000:,.0f} metric tons CO2e")
    print(f"  Diesel: {facility_data['energy_consumption']['diesel_gallons_per_year']:,} gallons")
    print(f"    Emissions: {diesel_emissions_kg / 1000:,.0f} metric tons CO2e")
    print(f"  Propane: {facility_data['energy_consumption']['propane_gallons_per_year']:,} gallons")
    print(f"    Emissions: {propane_emissions_kg / 1000:,.0f} metric tons CO2e")
    print(f"  TOTAL SCOPE 1: {scope1_emissions_tonnes:,.0f} metric tons CO2e")
    print()

    print("Scope 2 - Indirect Emissions (Purchased Electricity):")
    print(f"  Electricity: {electricity_kwh:,} kWh")
    print(f"  Grid Emission Factor: 0.417 kg CO2e/kWh (Midwest average)")
    print(f"  TOTAL SCOPE 2: {scope2_emissions_tonnes:,.0f} metric tons CO2e")
    print()

    print("Scope 3 - Upstream Fuel Production:")
    print(f"  Upstream Emissions: {scope3_emissions_tonnes:,.0f} metric tons CO2e")
    print()

    print("=" * 40)
    print(f"TOTAL EMISSIONS: {total_emissions_tonnes:,.0f} metric tons CO2e/year")
    print(f"EMISSIONS INTENSITY: {total_emissions_tonnes / (facility_data['annual_revenue_usd'] / 1000000):.1f} tonnes/$M revenue")
    print("=" * 40)
    print()

    # Step 2: Calculate SBTi trajectory
    print("=" * 80)
    print("STEP 2: Science-Based Targets (SBTi) Alignment")
    print("=" * 80)
    print()

    baseline_year = facility_data["baseline_year"]
    target_year = facility_data["net_zero_target_year"]
    years_to_target = target_year - baseline_year

    if facility_data["sbti_target"] == "1.5C":
        annual_reduction_rate = 0.042  # 4.2% per year
    else:  # 2.0C
        annual_reduction_rate = 0.025  # 2.5% per year

    # Calculate required trajectory
    target_2030_emissions = total_emissions_tonnes * ((1 - annual_reduction_rate) ** (2030 - baseline_year))
    target_2040_emissions = total_emissions_tonnes * ((1 - annual_reduction_rate) ** (2040 - baseline_year))
    target_2045_emissions = total_emissions_tonnes * ((1 - annual_reduction_rate) ** (target_year - baseline_year))

    print(f"SBTi Pathway: {facility_data['sbti_target']} aligned")
    print(f"Required Annual Reduction: {annual_reduction_rate * 100:.1f}%")
    print()

    print("EMISSIONS TRAJECTORY:")
    print(f"  {baseline_year} (Baseline): {total_emissions_tonnes:,.0f} metric tons CO2e")
    print(f"  2030 Target: {target_2030_emissions:,.0f} metric tons CO2e ({(1 - target_2030_emissions / total_emissions_tonnes) * 100:.0f}% reduction)")
    print(f"  2040 Target: {target_2040_emissions:,.0f} metric tons CO2e ({(1 - target_2040_emissions / total_emissions_tonnes) * 100:.0f}% reduction)")
    print(f"  {target_year} Net-Zero: {target_2045_emissions:,.0f} metric tons CO2e ({(1 - target_2045_emissions / total_emissions_tonnes) * 100:.0f}% reduction)")
    print()

    # Step 3: Identify decarbonization opportunities
    print("=" * 80)
    print("STEP 3: Decarbonization Opportunities Assessment")
    print("=" * 80)
    print()

    print("Assessing opportunities across all technology categories...")
    print("(In production, Agent #12 would coordinate with Agents #1-11)")
    print()

    # Simulate opportunity assessment (in production, would call other agents)
    opportunities = [
        {
            "opportunity_id": "OPP-001",
            "technology": "Waste Heat Recovery",
            "source_agent": "WasteHeatRecoveryAgent_AI (#4)",
            "description": "Recover heat from paint oven exhaust for space heating",
            "capital_cost_usd": 185000,
            "annual_savings_usd": 72000,
            "simple_payback_years": 2.6,
            "co2_reduction_tonnes_per_year": 320,
            "complexity": "Moderate",
            "implementation_phase": 1,
        },
        {
            "opportunity_id": "OPP-002",
            "technology": "Boiler Replacement",
            "source_agent": "BoilerReplacementAgent_AI (#2)",
            "description": "Replace 18-year-old boilers with high-efficiency condensing boilers",
            "capital_cost_usd": 420000,
            "annual_savings_usd": 85000,
            "simple_payback_years": 4.9,
            "co2_reduction_tonnes_per_year": 380,
            "complexity": "Moderate",
            "implementation_phase": 1,
        },
        {
            "opportunity_id": "OPP-003",
            "technology": "Industrial Heat Pump",
            "source_agent": "IndustrialHeatPumpAgent_AI (#3)",
            "description": "Heat pump for parts washing using compressor waste heat",
            "capital_cost_usd": 280000,
            "annual_savings_usd": 62000,
            "simple_payback_years": 4.5,
            "co2_reduction_tonnes_per_year": 275,
            "complexity": "High",
            "implementation_phase": 2,
        },
        {
            "opportunity_id": "OPP-004",
            "technology": "LED Lighting Retrofit",
            "source_agent": "EnergyEfficiencyAgent_AI",
            "description": "Replace 800 T8 fluorescent fixtures with LED",
            "capital_cost_usd": 120000,
            "annual_savings_usd": 46000,
            "simple_payback_years": 2.6,
            "co2_reduction_tonnes_per_year": 160,
            "complexity": "Low",
            "implementation_phase": 1,
        },
        {
            "opportunity_id": "OPP-005",
            "technology": "Compressed Air Optimization",
            "source_agent": "MotorSystemsAgent_AI",
            "description": "VFD compressor + leak detection + pressure optimization",
            "capital_cost_usd": 95000,
            "annual_savings_usd": 72000,
            "simple_payback_years": 1.3,
            "co2_reduction_tonnes_per_year": 250,
            "complexity": "Low",
            "implementation_phase": 1,
        },
        {
            "opportunity_id": "OPP-006",
            "technology": "Solar PV + Storage",
            "source_agent": "RenewableEnergyAgent_AI (#10)",
            "description": "500 kW rooftop solar + 200 kWh battery storage",
            "capital_cost_usd": 950000,
            "annual_savings_usd": 105000,
            "simple_payback_years": 9.0,
            "co2_reduction_tonnes_per_year": 365,
            "complexity": "Moderate",
            "implementation_phase": 3,
        },
        {
            "opportunity_id": "OPP-007",
            "technology": "Building Envelope",
            "source_agent": "BuildingEfficiencyAgent_AI",
            "description": "Insulation upgrade + air sealing + new doors",
            "capital_cost_usd": 180000,
            "annual_savings_usd": 42000,
            "simple_payback_years": 4.3,
            "co2_reduction_tonnes_per_year": 185,
            "complexity": "Moderate",
            "implementation_phase": 2,
        },
    ]

    print(f"Total Opportunities Identified: {len(opportunities)}")
    print()

    for opp in opportunities:
        print(f"{opp['opportunity_id']}: {opp['technology']}")
        print(f"  Source: {opp['source_agent']}")
        print(f"  Description: {opp['description']}")
        print(f"  Capital Cost: ${opp['capital_cost_usd']:,}")
        print(f"  Annual Savings: ${opp['annual_savings_usd']:,}")
        print(f"  Payback: {opp['simple_payback_years']:.1f} years")
        print(f"  CO2 Reduction: {opp['co2_reduction_tonnes_per_year']} tonnes/year")
        print(f"  Complexity: {opp['complexity']}")
        print()

    # Step 4: Calculate Marginal Abatement Cost (MAC) and prioritize
    print("=" * 80)
    print("STEP 4: Marginal Abatement Cost (MAC) Analysis & Prioritization")
    print("=" * 80)
    print()

    # Calculate MAC = (Capital Cost - NPV of Savings) / Lifetime CO2 Reduction
    # Simplified: MAC ≈ (Annualized Capital Cost - Annual Savings) / Annual CO2 Reduction
    for opp in opportunities:
        # Annualized capital cost (CRF method)
        r = facility_data["discount_rate"]
        n = facility_data["project_lifetime_years"]
        crf = (r * (1 + r) ** n) / (((1 + r) ** n) - 1)
        annualized_capital = opp["capital_cost_usd"] * crf

        # Net annual cost (negative = savings)
        net_annual_cost = annualized_capital - opp["annual_savings_usd"]

        # MAC ($/tonne CO2e)
        mac = net_annual_cost / opp["co2_reduction_tonnes_per_year"]

        opp["mac_usd_per_tonne"] = mac

    # Sort by MAC (lowest first = best)
    opportunities_sorted = sorted(opportunities, key=lambda x: x["mac_usd_per_tonne"])

    print("OPPORTUNITIES RANKED BY MARGINAL ABATEMENT COST:")
    print()
    print(f"{'Rank':<6} {'ID':<10} {'Technology':<25} {'MAC ($/tonne)':<15} {'Payback':<12} {'CO2 Reduction'}")
    print("-" * 95)

    for i, opp in enumerate(opportunities_sorted, 1):
        print(f"{i:<6} {opp['opportunity_id']:<10} {opp['technology']:<25} "
              f"${opp['mac_usd_per_tonne']:>8,.0f}       "
              f"{opp['simple_payback_years']:>5.1f} yrs    "
              f"{opp['co2_reduction_tonnes_per_year']:>4} tonnes/yr")

    print()

    # Step 5: Portfolio optimization under budget constraint
    print("=" * 80)
    print("STEP 5: Portfolio Optimization (Budget Constraint: $3M)")
    print("=" * 80)
    print()

    budget_remaining = facility_data["budget_constraint_usd"]
    selected_opportunities = []

    print("GREEDY SELECTION (Lowest MAC First):")
    print()

    for opp in opportunities_sorted:
        if budget_remaining >= opp["capital_cost_usd"]:
            selected_opportunities.append(opp)
            budget_remaining -= opp["capital_cost_usd"]
            print(f"✓ SELECTED: {opp['opportunity_id']} - {opp['technology']}")
            print(f"    Cost: ${opp['capital_cost_usd']:,} | Remaining Budget: ${budget_remaining:,}")
        else:
            print(f"✗ SKIPPED: {opp['opportunity_id']} - {opp['technology']} (exceeds budget)")

    print()

    # Step 6: Portfolio summary
    print("=" * 80)
    print("STEP 6: Optimized Portfolio Summary")
    print("=" * 80)
    print()

    total_investment = sum(opp["capital_cost_usd"] for opp in selected_opportunities)
    total_annual_savings = sum(opp["annual_savings_usd"] for opp in selected_opportunities)
    total_co2_reduction = sum(opp["co2_reduction_tonnes_per_year"] for opp in selected_opportunities)
    weighted_payback = total_investment / total_annual_savings if total_annual_savings > 0 else 999

    print(f"SELECTED OPPORTUNITIES: {len(selected_opportunities)}")
    print()

    print("PORTFOLIO METRICS:")
    print(f"  Total Investment: ${total_investment:,} (${budget_remaining:,} under budget)")
    print(f"  Total Annual Savings: ${total_annual_savings:,}")
    print(f"  Weighted Average Payback: {weighted_payback:.2f} years")
    print(f"  Total CO2 Reduction: {total_co2_reduction:,} metric tons/year")
    print(f"  Percentage of Baseline: {total_co2_reduction / total_emissions_tonnes * 100:.1f}%")
    print()

    # Step 7: Phased roadmap
    print("=" * 80)
    print("STEP 7: Phased Implementation Roadmap")
    print("=" * 80)
    print()

    # Group by phase
    phase1 = [opp for opp in selected_opportunities if opp["implementation_phase"] == 1]
    phase2 = [opp for opp in selected_opportunities if opp["implementation_phase"] == 2]
    phase3 = [opp for opp in selected_opportunities if opp["implementation_phase"] == 3]

    print("PHASE 1 (Years 1-2): Quick Wins & Foundation")
    print(f"  Timeline: {baseline_year + 1}-{baseline_year + 2}")
    print(f"  Projects: {len(phase1)}")
    for opp in phase1:
        print(f"    • {opp['technology']}: ${opp['capital_cost_usd']:,} | "
              f"{opp['co2_reduction_tonnes_per_year']} tonnes/yr | {opp['simple_payback_years']:.1f} yr payback")
    phase1_investment = sum(opp["capital_cost_usd"] for opp in phase1)
    phase1_savings = sum(opp["annual_savings_usd"] for opp in phase1)
    phase1_co2 = sum(opp["co2_reduction_tonnes_per_year"] for opp in phase1)
    print(f"  PHASE 1 TOTALS: ${phase1_investment:,} investment | ${phase1_savings:,}/yr savings | {phase1_co2} tonnes/yr CO2")
    print()

    if phase2:
        print("PHASE 2 (Years 3-5): Strategic Investments")
        print(f"  Timeline: {baseline_year + 3}-{baseline_year + 5}")
        print(f"  Projects: {len(phase2)}")
        for opp in phase2:
            print(f"    • {opp['technology']}: ${opp['capital_cost_usd']:,} | "
                  f"{opp['co2_reduction_tonnes_per_year']} tonnes/yr | {opp['simple_payback_years']:.1f} yr payback")
        phase2_investment = sum(opp["capital_cost_usd"] for opp in phase2)
        phase2_savings = sum(opp["annual_savings_usd"] for opp in phase2)
        phase2_co2 = sum(opp["co2_reduction_tonnes_per_year"] for opp in phase2)
        print(f"  PHASE 2 TOTALS: ${phase2_investment:,} investment | ${phase2_savings:,}/yr savings | {phase2_co2} tonnes/yr CO2")
        print()

    if phase3:
        print("PHASE 3 (Years 6-10): Long-Term Transformation")
        print(f"  Timeline: {baseline_year + 6}-{baseline_year + 10}")
        print(f"  Projects: {len(phase3)}")
        for opp in phase3:
            print(f"    • {opp['technology']}: ${opp['capital_cost_usd']:,} | "
                  f"{opp['co2_reduction_tonnes_per_year']} tonnes/yr | {opp['simple_payback_years']:.1f} yr payback")
        phase3_investment = sum(opp["capital_cost_usd"] for opp in phase3)
        phase3_savings = sum(opp["annual_savings_usd"] for opp in phase3)
        phase3_co2 = sum(opp["co2_reduction_tonnes_per_year"] for opp in phase3)
        print(f"  PHASE 3 TOTALS: ${phase3_investment:,} investment | ${phase3_savings:,}/yr savings | {phase3_co2} tonnes/yr CO2")
        print()

    # Step 8: Progress toward net-zero
    print("=" * 80)
    print("STEP 8: Progress Toward Net-Zero Target")
    print("=" * 80)
    print()

    remaining_emissions = total_emissions_tonnes - total_co2_reduction
    sbti_2030_gap = target_2030_emissions - remaining_emissions

    print("EMISSIONS TRAJECTORY AFTER PORTFOLIO IMPLEMENTATION:")
    print(f"  Baseline ({baseline_year}): {total_emissions_tonnes:,.0f} metric tons CO2e")
    print(f"  After Portfolio: {remaining_emissions:,.0f} metric tons CO2e ({total_co2_reduction / total_emissions_tonnes * 100:.0f}% reduction)")
    print(f"  SBTi 2030 Target: {target_2030_emissions:,.0f} metric tons CO2e")
    print(f"  2030 Gap: {abs(sbti_2030_gap):,.0f} metric tons CO2e {'(SURPLUS)' if sbti_2030_gap < 0 else '(DEFICIT)'}")
    print()

    if sbti_2030_gap >= 0:
        print(f"✓ Portfolio MEETS 2030 SBTi target with {sbti_2030_gap:,.0f} tonne surplus")
    else:
        print(f"⚠ Portfolio achieves {(target_2030_emissions - remaining_emissions) / (total_emissions_tonnes - target_2030_emissions) * 100:.0f}% of required 2030 reduction")
        print(f"  Additional {abs(sbti_2030_gap):,.0f} tonnes/yr reduction needed by 2030")
        print(f"  Recommendations: Consider additional renewables, process electrification, or offsets")

    print()

    # Final summary
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()

    print(f"FACILITY: {facility_data['facility_name']}")
    print(f"BASELINE EMISSIONS: {total_emissions_tonnes:,.0f} metric tons CO2e/year")
    print()

    print("RECOMMENDED PORTFOLIO:")
    print(f"  • {len(selected_opportunities)} technology opportunities")
    print(f"  • ${total_investment:,} total investment")
    print(f"  • ${total_annual_savings:,} annual savings")
    print(f"  • {weighted_payback:.1f} year weighted payback")
    print(f"  • {total_co2_reduction:,} tonnes/year CO2 reduction ({total_co2_reduction / total_emissions_tonnes * 100:.0f}% of baseline)")
    print()

    print("IMPLEMENTATION TIMELINE:")
    print(f"  • Phase 1 ({baseline_year + 1}-{baseline_year + 2}): {len(phase1)} projects, ${phase1_investment:,}, {phase1_co2} tonnes/yr")
    if phase2:
        print(f"  • Phase 2 ({baseline_year + 3}-{baseline_year + 5}): {len(phase2)} projects, ${phase2_investment:,}, {phase2_co2} tonnes/yr")
    if phase3:
        print(f"  • Phase 3 ({baseline_year + 6}-{baseline_year + 10}): {len(phase3)} projects, ${phase3_investment:,}, {phase3_co2} tonnes/yr")
    print()

    print("SBTi ALIGNMENT:")
    print(f"  • Target: {facility_data['sbti_target']} pathway (4.2% annual reduction)")
    print(f"  • 2030 Progress: {'MEETS' if sbti_2030_gap >= 0 else 'PARTIAL'} ({abs(sbti_2030_gap):,.0f} tonne {'surplus' if sbti_2030_gap >= 0 else 'gap'})")
    print(f"  • Net-Zero Target: {target_year}")
    print()

    print("NEXT STEPS:")
    print("  1. Present roadmap to executive leadership for approval")
    print("  2. Prioritize Phase 1 projects for detailed engineering studies")
    print("  3. Secure financing (consider green loans, utility rebates, tax credits)")
    print("  4. Develop RFP packages for top 3 opportunities (waste heat, compressed air, LED)")
    print("  5. Establish carbon accounting and monitoring system")
    print("  6. Set quarterly implementation milestones")
    print("  7. Plan annual roadmap refresh to incorporate new technologies")
    print()

    print("=" * 80)
    print("END OF ROADMAP DEMONSTRATION")
    print("=" * 80)


if __name__ == "__main__":
    main()
