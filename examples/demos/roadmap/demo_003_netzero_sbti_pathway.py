# -*- coding: utf-8 -*-
"""
Demo Script #3: Net-Zero Pathway with Science-Based Targets (SBTi) Alignment

Scenario:
- Large chemical manufacturing facility
- Committed to SBTi 1.5°C pathway (4.2% annual reduction)
- Target: Net-zero by 2050
- Comprehensive technology roadmap with interim milestones
- Expected: Multi-decade transformation, 95%+ emissions reduction, residual offsets

Expected Results:
- Baseline emissions: 25,000 metric tons CO2e/year (2025)
- 2030 Target: 19,500 tonnes (22% reduction) - SBTi aligned
- 2040 Target: 12,000 tonnes (52% reduction) - On trajectory
- 2050 Net-Zero: <1,250 tonnes residual (95% reduction + offsets)
- Total investment: $25-30M over 25 years
- Technology phases: 4 phases (2025-2030, 2030-2040, 2040-2050, residual offsets)

Technologies:
- Phase 1: Efficiency, waste heat recovery, heat pumps
- Phase 2: Electrification, renewable energy, storage
- Phase 3: Process transformation, green hydrogen, CCUS
- Phase 4: Hard-to-abate offsets
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.decarbonization_roadmap_agent_ai import DecarbonizationRoadmapAgentAI
from greenlang.framework import AgentConfig
import math


def main():
    """Run net-zero SBTi pathway demonstration."""

    print("=" * 80)
    print("DEMO #3: Net-Zero Pathway with Science-Based Targets (SBTi) Alignment")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_netzero_sbti_pathway",
        temperature=0.0,
        seed=42,
        max_tokens=6000,
    )
    agent = DecarbonizationRoadmapAgentAI(config)

    # Define facility
    facility_data = {
        "facility_name": "Global Chemical Manufacturing - Texas Plant",
        "location": "Texas",
        "facility_type": "chemical_manufacturing",
        "industry_sector": "chemicals",
        "annual_revenue_usd": 250000000,
        "facility_size_sqft": 500000,
        "employees": 450,

        # GHG baseline (2025)
        "baseline_year": 2025,
        "baseline_emissions_tonnes": 25000,
        "scope1_emissions_tonnes": 18000,  # Direct fuel combustion, process emissions
        "scope2_emissions_tonnes": 6500,   # Purchased electricity
        "scope3_emissions_tonnes": 500,    # Upstream fuel production

        # Energy consumption
        "natural_gas_mmbtu_per_year": 180000,
        "electricity_kwh_per_year": 15000000,
        "steam_demand_klbs_per_hour": 80,
        "process_heat_mw_thermal": 25,

        # SBTi commitment
        "sbti_commitment": True,
        "sbti_target": "1.5C",  # 4.2% annual reduction
        "net_zero_target_year": 2050,
        "interim_targets": {
            2030: 0.22,  # 22% reduction from baseline
            2040: 0.52,  # 52% reduction from baseline
            2050: 0.95,  # 95% reduction (net-zero with offsets)
        },

        # Economic parameters
        "electricity_cost_usd_per_kwh": 0.10,
        "natural_gas_cost_usd_per_mmbtu": 7.50,
        "carbon_price_usd_per_tonne_2025": 50,
        "carbon_price_escalation_rate": 0.05,  # 5% per year
        "discount_rate": 0.08,
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Location: {facility_data['location']}")
    print(f"  Size: {facility_data['facility_size_sqft']:,} sqft")
    print(f"  Employees: {facility_data['employees']}")
    print(f"  Annual Revenue: ${facility_data['annual_revenue_usd']:,}")
    print()

    print("SBTi COMMITMENT:")
    print(f"  Target: {facility_data['sbti_target']} pathway (4.2% annual reduction)")
    print(f"  Baseline Year: {facility_data['baseline_year']}")
    print(f"  Net-Zero Target: {facility_data['net_zero_target_year']}")
    print(f"  Baseline Emissions: {facility_data['baseline_emissions_tonnes']:,} metric tons CO2e/year")
    print()

    # Step 1: SBTi trajectory calculation
    print("=" * 80)
    print("STEP 1: Science-Based Targets (SBTi) 1.5°C Trajectory")
    print("=" * 80)
    print()

    baseline_year = facility_data["baseline_year"]
    net_zero_year = facility_data["net_zero_target_year"]
    baseline_emissions = facility_data["baseline_emissions_tonnes"]
    annual_reduction_rate = 0.042  # 4.2% for 1.5°C

    print("SBTi 1.5°C PATHWAY REQUIREMENTS:")
    print(f"  • Annual Reduction Rate: {annual_reduction_rate * 100:.1f}%")
    print(f"  • Reduction by 2030: {(1 - (1 - annual_reduction_rate) ** (2030 - baseline_year)) * 100:.1f}%")
    print(f"  • Reduction by 2040: {(1 - (1 - annual_reduction_rate) ** (2040 - baseline_year)) * 100:.1f}%")
    print(f"  • Reduction by 2050: {(1 - (1 - annual_reduction_rate) ** (2050 - baseline_year)) * 100:.1f}%")
    print()

    # Calculate trajectory
    print("EMISSIONS TRAJECTORY (SBTi 1.5°C):")
    print()
    print(f"{'Year':<8} {'Emissions (tonnes)':<20} {'Cumulative Reduction':<25} {'Status'}")
    print("-" * 75)

    milestones = [2025, 2030, 2035, 2040, 2045, 2050]
    for year in milestones:
        years_from_baseline = year - baseline_year
        target_emissions = baseline_emissions * ((1 - annual_reduction_rate) ** years_from_baseline)
        reduction_pct = (1 - target_emissions / baseline_emissions) * 100

        if year == baseline_year:
            status = "Baseline"
        elif year == 2030:
            status = "Interim Target"
        elif year == 2040:
            status = "Interim Target"
        elif year == net_zero_year:
            status = "NET-ZERO"
        else:
            status = "Milestone"

        print(f"{year:<8} {target_emissions:>8,.0f}           {reduction_pct:>6.1f}%                  {status}")

    print()

    # Step 2: Technology roadmap by phase
    print("=" * 80)
    print("STEP 2: Multi-Phase Technology Roadmap (2025-2050)")
    print("=" * 80)
    print()

    # Define technology phases
    technology_phases = [
        {
            "phase": 1,
            "years": "2025-2030",
            "focus": "Efficiency & Waste Heat Recovery",
            "technologies": [
                {"name": "Waste Heat Recovery", "capex_m": 2.5, "co2_reduction": 2800, "payback": 2.8},
                {"name": "Boiler Replacement", "capex_m": 1.8, "co2_reduction": 1500, "payback": 4.2},
                {"name": "Industrial Heat Pumps", "capex_m": 2.2, "co2_reduction": 1800, "payback": 4.5},
                {"name": "Process Optimization", "capex_m": 1.2, "co2_reduction": 900, "payback": 3.1},
                {"name": "Compressed Air & Motors", "capex_m": 0.8, "co2_reduction": 600, "payback": 2.5},
            ],
            "total_capex_m": 8.5,
            "total_co2_reduction": 7600,
            "cumulative_reduction_pct": 30.4,
        },
        {
            "phase": 2,
            "years": "2030-2040",
            "focus": "Electrification & Renewable Energy",
            "technologies": [
                {"name": "Process Electrification", "capex_m": 5.5, "co2_reduction": 4200, "payback": 6.8},
                {"name": "Solar PV + Storage", "capex_m": 8.0, "co2_reduction": 2800, "payback": 8.5},
                {"name": "Electric Boilers", "capex_m": 2.8, "co2_reduction": 1600, "payback": 7.2},
                {"name": "Heat Pump Scale-Up", "capex_m": 1.5, "co2_reduction": 800, "payback": 5.5},
            ],
            "total_capex_m": 17.8,
            "total_co2_reduction": 9400,
            "cumulative_reduction_pct": 68.0,
        },
        {
            "phase": 3,
            "years": "2040-2050",
            "focus": "Process Transformation & Hard-to-Abate",
            "technologies": [
                {"name": "Green Hydrogen (Steam Reformer)", "capex_m": 12.0, "co2_reduction": 3500, "payback": 15.0},
                {"name": "Carbon Capture (CCUS)", "capex_m": 8.5, "co2_reduction": 2200, "payback": 20.0},
                {"name": "Advanced Heat Pumps (High Temp)", "capex_m": 2.5, "co2_reduction": 900, "payback": 8.0},
                {"name": "Renewable Energy Expansion", "capex_m": 3.0, "co2_reduction": 600, "payback": 9.5},
            ],
            "total_capex_m": 26.0,
            "total_co2_reduction": 7200,
            "cumulative_reduction_pct": 96.8,
        },
    ]

    for phase_data in technology_phases:
        print(f"PHASE {phase_data['phase']}: {phase_data['years']} - {phase_data['focus']}")
        print()

        for tech in phase_data["technologies"]:
            print(f"  • {tech['name']}:")
            print(f"      Capital: ${tech['capex_m']:.1f}M | CO2 Reduction: {tech['co2_reduction']:,} tonnes/yr | Payback: {tech['payback']:.1f} yrs")

        print()
        print(f"  PHASE {phase_data['phase']} TOTALS:")
        print(f"    Total Capital: ${phase_data['total_capex_m']:.1f}M")
        print(f"    Total CO2 Reduction: {phase_data['total_co2_reduction']:,} tonnes/year")
        print(f"    Cumulative Reduction: {phase_data['cumulative_reduction_pct']:.1f}% from baseline")
        print()

    # Step 3: Cumulative impact analysis
    print("=" * 80)
    print("STEP 3: Cumulative Impact Analysis (2025-2050)")
    print("=" * 80)
    print()

    cumulative_capex = 0
    cumulative_co2_reduction = 0
    remaining_emissions = baseline_emissions

    print(f"{'End of Phase':<15} {'Capital ($M)':<15} {'CO2 Reduction':<20} {'Remaining Emissions':<22} {'SBTi Compliance'}")
    print("-" * 90)

    for phase_data in technology_phases:
        cumulative_capex += phase_data["total_capex_m"]
        cumulative_co2_reduction += phase_data["total_co2_reduction"]
        remaining_emissions = baseline_emissions - cumulative_co2_reduction

        # Check SBTi compliance for phase end year
        end_year = int(phase_data["years"].split("-")[1])
        target_emissions = baseline_emissions * ((1 - annual_reduction_rate) ** (end_year - baseline_year))

        if remaining_emissions <= target_emissions:
            compliance = "✓ MEETS"
        else:
            compliance = f"✗ GAP: {remaining_emissions - target_emissions:,.0f}t"

        print(f"{phase_data['years']:<15} ${cumulative_capex:<14.1f} {cumulative_co2_reduction:>8,} tonnes/yr   "
              f"{remaining_emissions:>8,.0f} tonnes/yr   {compliance}")

    print()

    # Step 4: Residual emissions & offsets
    print("=" * 80)
    print("STEP 4: Residual Emissions & Offset Strategy")
    print("=" * 80)
    print()

    final_remaining_emissions = baseline_emissions - cumulative_co2_reduction
    residual_pct = (final_remaining_emissions / baseline_emissions) * 100

    print("HARD-TO-ABATE EMISSIONS (Residual after Phase 3):")
    print(f"  • Remaining Emissions: {final_remaining_emissions:,.0f} metric tons CO2e/year")
    print(f"  • Percentage of Baseline: {residual_pct:.1f}%")
    print()

    print("RESIDUAL EMISSIONS SOURCES:")
    print("  1. Process Emissions: Chemical reactions (non-energy)")
    print("  2. Fugitive Emissions: Refrigerants, leaks")
    print("  3. Scope 3: Supply chain emissions (partial)")
    print("  4. Grid Emissions: Residual fossil fuel in electricity")
    print()

    # Offset strategy
    offset_cost_per_tonne = 75  # Assume $75/tonne for high-quality offsets in 2050
    annual_offset_cost = final_remaining_emissions * offset_cost_per_tonne

    print("OFFSET STRATEGY (Net-Zero Completion):")
    print(f"  • Annual Offsets Required: {final_remaining_emissions:,.0f} metric tons CO2e")
    print(f"  • Offset Cost (2050): ${offset_cost_per_tonne}/tonne (high-quality, permanent)")
    print(f"  • Annual Offset Budget: ${annual_offset_cost:,.0f}/year")
    print()

    print("PREFERRED OFFSET TYPES:")
    print("  1. Direct Air Capture (DAC) - Permanent removal")
    print("  2. Enhanced Weathering - Long-term carbon storage")
    print("  3. Biochar - Durable carbon sequestration")
    print("  4. Reforestation - Nature-based (supplementary)")
    print()

    # Step 5: Financial analysis
    print("=" * 80)
    print("STEP 5: Financial Analysis (25-Year Program)")
    print("=" * 80)
    print()

    total_capex = sum(phase["total_capex_m"] for phase in technology_phases)
    total_co2_reduction = sum(phase["total_co2_reduction"] for phase in technology_phases)

    # Estimate savings (simplified: energy cost reduction proportional to CO2 reduction)
    energy_cost_baseline = (facility_data["natural_gas_mmbtu_per_year"] * facility_data["natural_gas_cost_usd_per_mmbtu"] +
                            facility_data["electricity_kwh_per_year"] * facility_data["electricity_cost_usd_per_kwh"])

    annual_savings_phase1 = (technology_phases[0]["total_co2_reduction"] / baseline_emissions) * energy_cost_baseline * 0.6  # 60% savings
    annual_savings_phase2 = (technology_phases[1]["total_co2_reduction"] / baseline_emissions) * energy_cost_baseline * 0.5
    annual_savings_phase3 = (technology_phases[2]["total_co2_reduction"] / baseline_emissions) * energy_cost_baseline * 0.4

    total_annual_savings = annual_savings_phase1 + annual_savings_phase2 + annual_savings_phase3

    # Carbon pricing benefit
    carbon_price_2050 = facility_data["carbon_price_usd_per_tonne_2025"] * ((1 + facility_data["carbon_price_escalation_rate"]) ** (2050 - 2025))
    annual_carbon_benefit = total_co2_reduction * carbon_price_2050 * 0.5  # Assume 50% benefit realization

    print("PROGRAM INVESTMENT:")
    print(f"  • Phase 1 (2025-2030): ${technology_phases[0]['total_capex_m']:.1f}M")
    print(f"  • Phase 2 (2030-2040): ${technology_phases[1]['total_capex_m']:.1f}M")
    print(f"  • Phase 3 (2040-2050): ${technology_phases[2]['total_capex_m']:.1f}M")
    print(f"  • TOTAL CAPITAL: ${total_capex:.1f}M over 25 years")
    print()

    print("ANNUAL BENEFITS (Stabilized by 2050):")
    print(f"  • Energy Cost Savings: ${total_annual_savings:,.0f}/year")
    print(f"  • Carbon Pricing Benefit: ${annual_carbon_benefit:,.0f}/year (avoided costs)")
    print(f"  • Total Annual Benefit: ${total_annual_savings + annual_carbon_benefit:,.0f}/year")
    print()

    print("ONGOING COSTS (Post-2050):")
    print(f"  • Annual Offset Costs: ${annual_offset_cost:,.0f}/year")
    print(f"  • Net Annual Benefit: ${total_annual_savings + annual_carbon_benefit - annual_offset_cost:,.0f}/year")
    print()

    # Simple payback on cumulative basis
    effective_payback = (total_capex * 1000000) / (total_annual_savings + annual_carbon_benefit - annual_offset_cost)

    print("FINANCIAL METRICS:")
    print(f"  • Total Program Investment: ${total_capex:.1f}M")
    print(f"  • Effective Payback Period: {effective_payback:.1f} years (from 2050 stabilization)")
    print(f"  • Cost per Tonne Avoided: ${(total_capex * 1000000) / (total_co2_reduction * 25):,.0f}/tonne (25-year average)")
    print(f"  • NPV (8% discount, 25-year program): (Complex calculation - requires annual cashflows)")
    print()

    # Step 6: Risk & enablers
    print("=" * 80)
    print("STEP 6: Risks, Enablers & Success Factors")
    print("=" * 80)
    print()

    print("KEY RISKS:")
    print("  1. Technology Maturity: Green hydrogen & CCUS not yet cost-competitive (2025)")
    print("  2. Grid Decarbonization: Phase 2/3 assume cleaner grid (uncertain timeline)")
    print("  3. Regulatory Changes: Carbon pricing assumptions may not materialize")
    print("  4. Capital Availability: $52M total investment over 25 years")
    print("  5. Offset Supply: High-quality permanent offsets may be scarce/expensive by 2050")
    print()

    print("CRITICAL ENABLERS:")
    print("  1. Government Incentives: IRA/45Q credits for CCUS, hydrogen")
    print("  2. Technology Cost Reduction: 50-70% cost decline for hydrogen & CCUS by 2040")
    print("  3. Grid Decarbonization: Regional grid reaching 80%+ renewables by 2040")
    print("  4. Carbon Pricing: Federal carbon price starting $50/tonne, escalating 5%/year")
    print("  5. Utility Partnerships: Renewable energy PPAs, grid services")
    print()

    print("SUCCESS FACTORS:")
    print("  ✓ Phased approach aligns with technology maturity and capital availability")
    print("  ✓ Early wins (Phase 1) fund later phases through savings")
    print("  ✓ SBTi alignment ensures credibility with stakeholders and investors")
    print("  ✓ Portfolio approach diversifies risk across 14 technology categories")
    print("  ✓ Net-zero by 2050 positions company as climate leader in chemicals sector")
    print()

    # Step 7: Milestones & governance
    print("=" * 80)
    print("STEP 7: Governance, Milestones & Monitoring")
    print("=" * 80)
    print()

    print("GOVERNANCE STRUCTURE:")
    print("  • Executive Sponsor: Chief Sustainability Officer (CSO)")
    print("  • Program Management: Decarbonization Program Manager (dedicated role)")
    print("  • Site Leadership: Plant Manager + Facilities Director")
    print("  • Technical Leads: Engineering team (energy, process, mechanical)")
    print("  • Finance: Capital planning, incentive management")
    print("  • External: Consultants for specialized technologies (H2, CCUS)")
    print()

    print("ANNUAL MILESTONES & REPORTING:")
    print()
    print("2026: Phase 1 Launch")
    print("  • Waste heat recovery & heat pump installation begins")
    print("  • Baseline verification & measurement system deployment")
    print()
    print("2030: Interim Target (22% reduction)")
    print("  • Verify 7,600 tonnes/year reduction (SBTi target: 5,500 tonnes)")
    print("  • Phase 2 planning complete, capital approval secured")
    print("  • Report to SBTi and CDP")
    print()
    print("2035: Phase 2 Midpoint")
    print("  • Process electrification 50% complete")
    print("  • Solar PV + storage operational")
    print("  • 12,000 tonnes/year cumulative reduction")
    print()
    print("2040: Interim Target (52% reduction)")
    print("  • Verify 17,000 tonnes/year reduction (SBTi target: 13,000 tonnes)")
    print("  • Phase 3 technology pilots (H2, CCUS)")
    print("  • Offset strategy finalization")
    print()
    print("2045: Phase 3 Deployment")
    print("  • Green hydrogen & CCUS operational")
    print("  • 22,000 tonnes/year cumulative reduction")
    print("  • Offset contracts secured")
    print()
    print("2050: NET-ZERO ACHIEVEMENT")
    print("  • 95% emissions reduction (24,200 tonnes/year)")
    print("  • 800 tonnes/year residual + offsets = NET-ZERO")
    print("  • SBTi validation & public announcement")
    print()

    # Final summary
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()

    print(f"FACILITY: {facility_data['facility_name']}")
    print(f"COMMITMENT: SBTi 1.5°C pathway, Net-Zero by {net_zero_year}")
    print(f"BASELINE EMISSIONS (2025): {baseline_emissions:,} metric tons CO2e/year")
    print()

    print("25-YEAR TRANSFORMATION:")
    print(f"  • Phase 1 (2025-2030): Efficiency & waste heat - {technology_phases[0]['total_co2_reduction']:,} tonnes/yr, ${technology_phases[0]['total_capex_m']:.1f}M")
    print(f"  • Phase 2 (2030-2040): Electrification & renewables - {technology_phases[1]['total_co2_reduction']:,} tonnes/yr, ${technology_phases[1]['total_capex_m']:.1f}M")
    print(f"  • Phase 3 (2040-2050): Process transformation - {technology_phases[2]['total_co2_reduction']:,} tonnes/yr, ${technology_phases[2]['total_capex_m']:.1f}M")
    print(f"  • Total Investment: ${total_capex:.1f}M")
    print(f"  • Total Reduction: {total_co2_reduction:,} tonnes/year (96.8% of baseline)")
    print(f"  • Residual Offsets: {final_remaining_emissions:,.0f} tonnes/year")
    print()

    print("SBTi MILESTONE COMPLIANCE:")
    print("  ✓ 2030: 30.4% reduction (target: 22%) - EXCEEDS")
    print("  ✓ 2040: 68.0% reduction (target: 52%) - EXCEEDS")
    print("  ✓ 2050: 96.8% reduction + offsets = NET-ZERO - ACHIEVED")
    print()

    print("FINANCIAL OUTLOOK:")
    print(f"  • Total Capital: ${total_capex:.1f}M over 25 years")
    print(f"  • Annual Savings: ${total_annual_savings:,.0f} (energy cost reduction)")
    print(f"  • Annual Offset Cost: ${annual_offset_cost:,.0f} (ongoing post-2050)")
    print(f"  • Net Annual Benefit: ${total_annual_savings + annual_carbon_benefit - annual_offset_cost:,.0f}/year")
    print()

    print("NEXT ACTIONS:")
    print("  1. Board approval for 25-year net-zero commitment and Phase 1 funding")
    print("  2. Submit SBTi targets for validation (near-term: 2030, net-zero: 2050)")
    print("  3. Establish dedicated program management office (PMO)")
    print("  4. Launch Phase 1 detailed engineering studies (waste heat, heat pumps)")
    print("  5. Develop annual carbon accounting and reporting system")
    print("  6. Engage suppliers on Scope 3 reduction strategies")
    print("  7. Join industry coalitions (First Movers Coalition, Mission Possible)")
    print("  8. Annual roadmap review and update based on technology progress")
    print()

    print("=" * 80)
    print("PATHWAY TO NET-ZERO - DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
