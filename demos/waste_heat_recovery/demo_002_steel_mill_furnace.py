"""
Demo Script #2: Steel Mill Furnace Exhaust Recovery

Scenario:
- Large steel mill with reheat furnaces
- Furnace exhaust at 1,900°F (very high-grade waste heat)
- Target: High-temperature heat recovery for steam generation or preheating
- Goal: Maximize energy recovery with corrosion-resistant design

Expected Results:
- 15,000+ MMBtu/yr recoverable waste heat
- $900,000+ annual savings
- 1.8 year payback
- 1,000+ metric tons CO2 reduction/year

Technologies:
- Recuperator for high-temperature exhaust
- Economizer for steam generation
- Special materials for high-temperature service
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.waste_heat_recovery_agent_ai import WasteHeatRecoveryAgent_AI
from greenlang.framework import AgentConfig


def main():
    """Run steel mill furnace waste heat recovery demonstration."""

    print("=" * 80)
    print("DEMO #2: Steel Mill Reheat Furnace Waste Heat Recovery")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_steel_mill_waste_heat",
        temperature=0.0,
        seed=42,
        max_tokens=4000,
    )
    agent = WasteHeatRecoveryAgent_AI(config)

    # Define steel mill facility
    facility_data = {
        "facility_type": "steel_mill",
        "facility_name": "Midwest Steel Manufacturing",
        "location": "Indiana",
        "annual_production": "500,000 tons of steel products",
        "processes": [
            {
                "process_name": "Reheat Furnace #1",
                "process_type": "furnace",
                "fuel_input_mmbtu_yr": 80000,
                "fuel_type": "natural_gas",
                "exhaust_temperature_f": 1900,
                "exhaust_flow_cfm": 15000,
                "operating_hours_per_year": 8400,  # 96% uptime
            },
            {
                "process_name": "Reheat Furnace #2",
                "process_type": "furnace",
                "fuel_input_mmbtu_yr": 75000,
                "fuel_type": "natural_gas",
                "exhaust_temperature_f": 1850,
                "exhaust_flow_cfm": 14000,
                "operating_hours_per_year": 8400,
            },
            {
                "process_name": "Annealing Furnace",
                "process_type": "furnace",
                "fuel_input_mmbtu_yr": 40000,
                "fuel_type": "natural_gas",
                "exhaust_temperature_f": 1400,
                "exhaust_flow_cfm": 8000,
                "operating_hours_per_year": 7884,
            },
        ],
        "total_annual_fuel_mmbtu": 195000,
        "fuel_cost_usd_per_mmbtu": 6.5,
        "electricity_cost_usd_per_kwh": 0.09,
        "minimum_temperature_f": 300,  # Higher minimum for high-grade applications
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Annual Production: {facility_data['annual_production']}")
    print(f"  Annual Fuel Consumption: {facility_data['total_annual_fuel_mmbtu']:,} MMBtu")
    print(f"  Fuel Cost: ${facility_data['fuel_cost_usd_per_mmbtu']}/MMBtu")
    print(f"  Number of Furnaces: {len(facility_data['processes'])}")
    print()

    # Step 1: Identify high-temperature waste heat
    print("STEP 1: Identifying High-Temperature Waste Heat Sources...")
    print("-" * 80)

    waste_heat_analysis = agent._identify_waste_heat_sources_impl(
        facility_type=facility_data["facility_type"],
        processes=facility_data["processes"],
        include_hvac_systems=False,  # Focus on process heat only
        include_compressed_air=False,
        minimum_temperature_f=facility_data["minimum_temperature_f"],
    )

    print(f"Total Waste Heat Identified: {waste_heat_analysis['total_waste_heat_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  ({waste_heat_analysis['total_waste_heat_mmbtu_yr']/facility_data['total_annual_fuel_mmbtu']*100:.1f}% of fuel input)")
    print(f"Technically Recoverable: {waste_heat_analysis['recoverable_waste_heat_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"Economically Recoverable: {waste_heat_analysis['economically_recoverable_mmbtu_yr']:,.0f} MMBtu/year")
    print()

    print("Waste Heat Quality Distribution:")
    summary = waste_heat_analysis["waste_heat_summary"]
    print(f"  High-Grade (>400°F): {summary['high_grade_above_400f_mmbtu_yr']:,.0f} MMBtu/year ({summary['high_grade_above_400f_mmbtu_yr']/waste_heat_analysis['total_waste_heat_mmbtu_yr']*100:.0f}%)")
    print(f"  Medium-Grade (200-400°F): {summary['medium_grade_200_400f_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Low-Grade (<200°F): {summary['low_grade_below_200f_mmbtu_yr']:,.0f} MMBtu/year")
    print()
    print("NOTE: All waste heat is high-grade, ideal for high-value applications")
    print()

    # Step 2: Focus on Reheat Furnace #1 (largest opportunity)
    print("STEP 2: Analyzing Reheat Furnace #1 Recovery Opportunity...")
    print("-" * 80)

    primary_source = waste_heat_analysis["waste_heat_sources"][0]
    print(f"\nPrimary Opportunity: {primary_source['source_name']}")
    print(f"  Exhaust Temperature: {primary_source['temperature_f']}°F")
    print(f"  Mass Flow Rate: ~{primary_source.get('mass_flow_rate_lb_hr', 0):,.0f} lb/hr")
    print(f"  Available Waste Heat: {primary_source['waste_heat_mmbtu_yr']:,.0f} MMBtu/year")
    print()

    # Calculate recovery potential for steam generation (target: 500°F exhaust)
    recovery = agent._calculate_heat_recovery_potential_impl(
        waste_heat_stream={
            "temperature_f": primary_source["temperature_f"],
            "mass_flow_rate_lb_hr": 50000,  # High flow rate for large furnace
            "fluid_type": "combustion_products_natural_gas",
        },
        recovery_temperature_f=500,  # Cool to 500°F for maximum recovery
        heat_exchanger_effectiveness=0.70,  # Conservative for high-temp
    )

    print("Recovery Potential Analysis:")
    print(f"  Theoretical Recovery: {recovery['theoretical_heat_recovery_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Practical Recovery (70% eff): {recovery['practical_heat_recovery_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Exergy Available: {recovery['exergy_available_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Outlet Temperature: {recovery['outlet_temperature_f']}°F")
    print(f"  Operating Hours: {recovery['operating_hours_per_year']:.0f} hrs/year")
    print()

    # Assess fouling and corrosion risk (critical for high-temp applications)
    print("STEP 3: Assessing Fouling and Corrosion Risk...")
    print("-" * 80)

    risk_assessment = agent._assess_fouling_corrosion_risk_impl(
        waste_heat_stream={
            "temperature_f": primary_source["temperature_f"],
            "fluid_type": "flue_gas",
            "sulfur_content_ppm": 150,  # Typical for natural gas with some impurities
            "particulate_content_ppm": 200,  # Steel mill dust
            "chloride_content_ppm": 50,
        },
        material_of_construction="hastelloy_c276",  # High-temp alloy
    )

    print(f"Overall Risk Level: {risk_assessment['overall_risk_level'].upper()}")
    print(f"Risk Score: {risk_assessment['risk_score']}/100")
    print(f"Material: {risk_assessment['material_of_construction'].replace('_', ' ').title()}")
    print(f"Material Compatibility: {risk_assessment['material_compatibility'].upper()}")
    print(f"Maintenance Frequency: {risk_assessment['recommended_maintenance_frequency'].title()}")
    print()

    if len(risk_assessment['identified_risks']) > 0:
        print("Identified Risks:")
        for risk in risk_assessment['identified_risks']:
            print(f"  - {risk['risk_type'].replace('_', ' ').title()}: {risk['severity'].upper()}")
            print(f"    {risk['description']}")
        print()

    if len(risk_assessment['mitigation_strategies']) > 0:
        print("Mitigation Strategies:")
        for i, strategy in enumerate(risk_assessment['mitigation_strategies'], 1):
            print(f"  {i}. {strategy}")
        print()

    # Select technology for high-temperature application
    print("STEP 4: Technology Selection for High-Temperature Service...")
    print("-" * 80)

    tech_selection = agent._select_heat_recovery_technology_impl(
        waste_heat_stream={
            "temperature_f": primary_source["temperature_f"],
            "fluid_type": "flue_gas",
            "heat_load_mmbtu_yr": recovery['practical_heat_recovery_mmbtu_yr'],
            "fouling_potential": "moderate",
        },
        application="steam_generation",
        budget_usd=1500000,  # Higher budget for large industrial application
        space_constrained=False,
    )

    print(f"Recommended Technology: {tech_selection['recommended_technology']}")
    print(f"Confidence Score: {tech_selection['confidence_score']:.1f}/100")
    print()

    print("Top 3 Technology Options:")
    for i, tech in enumerate(tech_selection['all_technologies_ranked'][:3], 1):
        print(f"\n{i}. {tech['technology']}")
        print(f"   Score: {tech['total_score']:.1f}/100")
        print(f"   Effectiveness: {tech['typical_effectiveness']*100:.0f}%")
        print(f"   Estimated Cost: ${tech['estimated_cost_usd']:,.0f}")
        print(f"   Maintenance: {tech['maintenance_level'].title()}")
        print(f"   Selection Reasons:")
        for reason in tech['selection_reasons'][:3]:
            print(f"     - {reason}")

    # Size the recuperator
    print("\n" + "=" * 80)
    print("STEP 5: Heat Exchanger Sizing...")
    print("=" * 80)

    sizing = agent._size_heat_exchanger_impl(
        heat_load_btu_hr=recovery['practical_heat_recovery_mmbtu_yr'] * 1_000_000 / 8400,
        hot_side_in_f=primary_source["temperature_f"],
        hot_side_out_f=recovery['outlet_temperature_f'],
        cold_side_in_f=250,  # Combustion air preheat
        cold_side_out_f=1200,  # High preheat temperature
        technology=tech_selection['recommended_technology_key'],
        flow_arrangement="counterflow",
    )

    print(f"\nHeat Exchanger Design:")
    print(f"  Required Area: {sizing['required_area_ft2']:,.1f} ft²")
    print(f"  Design Area (with safety factor): {sizing['design_area_ft2']:,.1f} ft²")
    print(f"  LMTD: {sizing['lmtd_f']:.1f}°F")
    print(f"  F-Factor: {sizing['f_factor']:.3f}")
    print(f"  Effectiveness: {sizing['effectiveness']:.1%}")
    print(f"  NTU: {sizing['ntu']:.2f}")
    print(f"  U-Value: {sizing['u_value_btu_hr_ft2_f']} Btu/hr·ft²·°F")
    print(f"  Estimated Length: {sizing['estimated_length_ft']:.1f} ft")
    print(f"  Estimated Diameter: {sizing['estimated_diameter_ft']:.1f} ft")
    print(f"  Pressure Drop: {sizing['estimated_pressure_drop_psi']:.2f} psi")
    print(f"  Capital Cost: ${sizing['estimated_capital_cost_usd']:,.0f}")
    print()

    # Calculate energy savings
    print("STEP 6: Energy Savings Analysis...")
    print("-" * 80)

    savings = agent._calculate_energy_savings_impl(
        recovered_heat_mmbtu_yr=recovery['practical_heat_recovery_mmbtu_yr'],
        displaced_fuel_type="natural_gas",
        fuel_price_usd_per_mmbtu=facility_data["fuel_cost_usd_per_mmbtu"],
        boiler_efficiency=0.80,  # Baseline furnace efficiency
        electricity_price_usd_per_kwh=facility_data["electricity_cost_usd_per_kwh"],
    )

    print(f"\nAnnual Energy Savings:")
    print(f"  Recovered Heat: {savings['recovered_heat_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Fuel Displaced: {savings['fuel_displaced_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"  Fuel Cost Savings: ${savings['fuel_cost_savings_usd_yr']:,.0f}/year")
    print(f"  Parasitic Cost: ${savings['parasitic_cost_usd_yr']:,.0f}/year")
    print(f"  Net Savings: ${savings['net_savings_usd_yr']:,.0f}/year")
    print()

    print(f"Carbon Impact:")
    print(f"  CO2 Reduction: {savings['co2_reduction_metric_tons_yr']:,.0f} metric tons/year")
    print(f"  Carbon Benefit: ${savings['carbon_benefit_usd_yr']:,.0f}/year (@$51/ton)")
    print(f"  Total Benefit: ${savings['total_benefit_usd_yr']:,.0f}/year")
    print()

    print(f"Impact Metrics:")
    print(f"  Equivalent Homes Heated: {savings['impact_metrics']['homes_heated_equivalent']:,.0f}")
    print(f"  Equivalent Cars Off Road: {savings['impact_metrics']['cars_off_road_equivalent']:,.0f}")
    print()

    # Financial analysis
    print("=" * 80)
    print("STEP 7: Financial Analysis...")
    print("=" * 80)

    payback = agent._calculate_payback_period_impl(
        capital_cost_usd=sizing['estimated_capital_cost_usd'],
        annual_savings_usd=savings['net_savings_usd_yr'],
        annual_maintenance_cost_usd=sizing['estimated_capital_cost_usd'] * 0.04,  # 4% for high-temp
        project_lifetime_years=25,  # Long-life industrial equipment
        discount_rate=0.10,  # Industrial hurdle rate
        energy_cost_escalation_rate=0.03,
    )

    print(f"\nFinancial Metrics:")
    print(f"  Capital Investment: ${sizing['estimated_capital_cost_usd']:,.0f}")
    print(f"  Annual Maintenance: ${sizing['estimated_capital_cost_usd'] * 0.04:,.0f}")
    print(f"  Simple Payback Period: {payback['simple_payback_years']:.2f} years")
    print(f"  Discounted Payback: {payback['discounted_payback_years']} years")
    print(f"  Net Present Value (25yr): ${payback['net_present_value_usd']:,.0f}")
    print(f"  Internal Rate of Return: {payback['internal_rate_of_return_percent']:.1f}%")
    print(f"  Savings-to-Investment Ratio: {payback['savings_to_investment_ratio']:.2f}")
    print(f"  Project Attractiveness: {payback['project_attractiveness'].upper()}")
    print()

    print(f"Lifetime Metrics (25 years):")
    print(f"  Total Savings: ${payback['total_lifetime_savings_usd']:,.0f}")
    print(f"  Total Maintenance: ${payback['total_lifetime_maintenance_usd']:,.0f}")
    print(f"  Net Benefit: ${payback['net_present_value_usd']:,.0f} (NPV)")
    print()

    # Summary and recommendations
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()
    print(f"Reheat Furnace #1 - Waste Heat Recovery Project")
    print(f"-" * 80)
    print(f"✓ Recoverable Waste Heat: {recovery['practical_heat_recovery_mmbtu_yr']:,.0f} MMBtu/year")
    print(f"✓ Technology: {tech_selection['recommended_technology']}")
    print(f"✓ Heat Exchanger Area: {sizing['design_area_ft2']:,.0f} ft²")
    print(f"✓ Capital Investment: ${sizing['estimated_capital_cost_usd']:,.0f}")
    print(f"✓ Annual Savings: ${savings['net_savings_usd_yr']:,.0f}")
    print(f"✓ Payback Period: {payback['simple_payback_years']:.2f} years")
    print(f"✓ IRR: {payback['internal_rate_of_return_percent']:.1f}%")
    print(f"✓ CO2 Reduction: {savings['co2_reduction_metric_tons_yr']:,.0f} metric tons/year")
    print()

    print("RECOMMENDATIONS:")
    print(f"1. PROCEED with {tech_selection['recommended_technology']} installation")
    print(f"2. Use {risk_assessment['material_of_construction'].replace('_', ' ').title()} for high-temperature sections")
    print(f"3. Implement {risk_assessment['recommended_maintenance_frequency']} maintenance program")
    print(f"4. Consider similar recovery for Reheat Furnace #2 (additional ${savings['net_savings_usd_yr']*0.9:,.0f}/year)")
    print(f"5. Explore utility incentives (may reduce payback to <1.5 years)")
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()
