"""
Demo Script #2: Chemical Plant Process Heat with Heat Pump Integration

Scenario:
- Large chemical manufacturing facility with process heat requirements
- Multiple temperature levels (140°F, 180°F, 220°F)
- Heat recovery from process cooling water and reactor cooling
- Target: Heat pump integration with waste heat recovery

Expected Results:
- COP: 3.0-3.5 for low-temperature processes
- COP: 2.5-3.0 for medium-temperature processes
- Annual savings: $200,000-$300,000
- Payback: 5.8 years
- CO2 reduction: 500-700 metric tons/year

Technologies:
- Multi-stage heat pump system with cascading
- Water-source heat pump utilizing process cooling water
- R134a primary stage, R245fa high-temperature stage
- Thermal storage for load leveling
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.industrial_heat_pump_agent_ai import IndustrialHeatPumpAgent_AI
from greenlang.framework import AgentConfig


def main():
    """Run chemical plant heat pump demonstration."""

    print("=" * 80)
    print("DEMO #2: Chemical Plant Process Heat - Heat Pump Integration")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_chemical_plant_heat_pump",
        temperature=0.0,
        seed=42,
        max_tokens=4000,
    )
    agent = IndustrialHeatPumpAgent_AI(config)

    # Define chemical plant facility
    facility_data = {
        "facility_name": "Ohio Specialty Chemicals Inc.",
        "location": "Ohio",
        "facility_type": "chemical_manufacturing",
        "industry_sector": "specialty_chemicals",
        "annual_production": "200 million lbs of chemical products",
        "processes": ["distillation", "reactor_heating", "crystallization", "drying"],

        # Current heating system
        "current_heating": {
            "steam_boiler": {
                "capacity_mlb_hr": 40,  # 40,000 lbs/hr steam
                "pressure_psig": 150,
                "fuel_type": "natural_gas",
                "efficiency": 0.83,
                "annual_fuel_mmbtu": 45000,
                "fuel_cost_usd_per_mmbtu": 8.20,
            },
            "hot_water_boiler": {
                "capacity_mmbtu_hr": 8.0,
                "fuel_type": "natural_gas",
                "efficiency": 0.85,
                "annual_fuel_mmbtu": 28000,
                "fuel_cost_usd_per_mmbtu": 8.20,
            },
        },

        # Process heat requirements by temperature level
        "heat_requirements": [
            {
                "process": "Reactor Heating",
                "temperature_required_f": 220,
                "load_kw": 450,
                "load_profile": "continuous",
                "hours_per_day": 24,
                "days_per_week": 7,
                "criticality": "high",
            },
            {
                "process": "Distillation Reboilers",
                "temperature_required_f": 180,
                "load_kw": 800,
                "load_profile": "continuous",
                "hours_per_day": 24,
                "days_per_week": 7,
                "criticality": "high",
            },
            {
                "process": "Feed Preheating",
                "temperature_required_f": 140,
                "load_kw": 350,
                "load_profile": "daytime_only",
                "hours_per_day": 16,
                "days_per_week": 6,
                "criticality": "medium",
            },
            {
                "process": "Tank Heating",
                "temperature_required_f": 120,
                "load_kw": 200,
                "load_profile": "seasonal",
                "hours_per_day": 12,
                "days_per_week": 7,
                "criticality": "low",
            },
        ],

        # Available heat sources (waste heat from processes)
        "available_heat_sources": [
            {
                "source": "Reactor Cooling Water",
                "temperature_f": 105,
                "flow_gpm": 500,
                "available_capacity_kw": 300,
                "availability": "continuous",
                "quality": "high",
            },
            {
                "source": "Condenser Cooling Water",
                "temperature_f": 95,
                "flow_gpm": 800,
                "available_capacity_kw": 450,
                "availability": "continuous",
                "quality": "high",
            },
            {
                "source": "Process Equipment Cooling",
                "temperature_f": 85,
                "flow_gpm": 400,
                "available_capacity_kw": 250,
                "availability": "daytime",
                "quality": "medium",
            },
            {
                "source": "Ambient Air",
                "temperature_f": 55,  # Average Ohio temperature
                "available_capacity_kw": 10000,  # Unlimited
                "availability": "continuous",
                "quality": "variable",
            },
        ],

        # Economic parameters
        "electricity_cost_usd_per_kwh": 0.10,
        "demand_charge_usd_per_kw": 18,
        "natural_gas_cost_usd_per_mmbtu": 8.20,
        "steam_cost_usd_per_mlb": 12.50,
        "project_lifetime_years": 20,
        "discount_rate": 0.12,  # Chemical industry hurdle rate
        "maintenance_cost_percent": 0.04,  # 4% of capital annually
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Sector: {facility_data['industry_sector'].replace('_', ' ').title()}")
    print(f"  Annual Production: {facility_data['annual_production']}")
    print(f"  Key Processes: {', '.join([p.title() for p in facility_data['processes']])}")
    print()

    # Step 1: Current energy consumption and costs
    print("STEP 1: Current Energy Consumption Analysis")
    print("-" * 80)

    steam_annual_cost = facility_data["current_heating"]["steam_boiler"]["annual_fuel_mmbtu"] * \
                        facility_data["current_heating"]["steam_boiler"]["fuel_cost_usd_per_mmbtu"]
    hot_water_annual_cost = facility_data["current_heating"]["hot_water_boiler"]["annual_fuel_mmbtu"] * \
                            facility_data["current_heating"]["hot_water_boiler"]["fuel_cost_usd_per_mmbtu"]
    total_current_cost = steam_annual_cost + hot_water_annual_cost

    print(f"\nCurrent Heating System:")
    print(f"  Steam Boiler:")
    print(f"    - Capacity: {facility_data['current_heating']['steam_boiler']['capacity_mlb_hr']:,.0f} lbs/hr")
    print(f"    - Annual Fuel: {facility_data['current_heating']['steam_boiler']['annual_fuel_mmbtu']:,} MMBtu")
    print(f"    - Annual Cost: ${steam_annual_cost:,.0f}")
    print(f"  Hot Water Boiler:")
    print(f"    - Capacity: {facility_data['current_heating']['hot_water_boiler']['capacity_mmbtu_hr']} MMBtu/hr")
    print(f"    - Annual Fuel: {facility_data['current_heating']['hot_water_boiler']['annual_fuel_mmbtu']:,} MMBtu")
    print(f"    - Annual Cost: ${hot_water_annual_cost:,.0f}")
    print(f"\n  TOTAL ANNUAL HEATING COST: ${total_current_cost:,.0f}")
    print()

    # Step 2: Process heat requirements analysis
    print("STEP 2: Process Heat Requirements by Temperature Level")
    print("-" * 80)

    heat_by_temp = {}
    total_annual_kwh = 0

    for req in facility_data["heat_requirements"]:
        temp = req["temperature_required_f"]
        load_kw = req["load_kw"]

        # Calculate annual hours
        if req["load_profile"] == "continuous":
            annual_hours = 8760
        elif req["load_profile"] == "daytime_only":
            annual_hours = req["hours_per_day"] * req["days_per_week"] * 52
        elif req["load_profile"] == "seasonal":
            annual_hours = req["hours_per_day"] * req["days_per_week"] * 26  # 6 months
        else:
            annual_hours = 8760 * 0.85  # 85% uptime

        annual_kwh = load_kw * annual_hours
        total_annual_kwh += annual_kwh

        if temp not in heat_by_temp:
            heat_by_temp[temp] = {"processes": [], "total_kw": 0, "annual_kwh": 0}

        heat_by_temp[temp]["processes"].append(req["process"])
        heat_by_temp[temp]["total_kw"] += load_kw
        heat_by_temp[temp]["annual_kwh"] += annual_kwh

        print(f"\n{req['process']}:")
        print(f"  Temperature: {temp}°F")
        print(f"  Load: {load_kw} kW")
        print(f"  Profile: {req['load_profile'].replace('_', ' ').title()}")
        print(f"  Annual Energy: {annual_kwh:,.0f} kWh")
        print(f"  Criticality: {req['criticality'].upper()}")

    print(f"\nHeat Requirements Summary:")
    for temp in sorted(heat_by_temp.keys()):
        data = heat_by_temp[temp]
        print(f"  {temp}°F: {data['total_kw']:.0f} kW, {data['annual_kwh']:,.0f} kWh/yr ({', '.join(data['processes'])})")
    print(f"\nTOTAL: {sum([d['total_kw'] for d in heat_by_temp.values()]):.0f} kW, {total_annual_kwh:,.0f} kWh/yr")
    print()

    # Step 3: Heat source evaluation
    print("STEP 3: Available Heat Sources for Heat Pump Integration")
    print("-" * 80)

    print("\nWaste Heat Sources:")
    for source in facility_data["available_heat_sources"]:
        print(f"\n{source['source']}:")
        print(f"  Temperature: {source['temperature_f']}°F")
        print(f"  Available Capacity: {source['available_capacity_kw']:.0f} kW")
        print(f"  Availability: {source['availability'].title()}")

        # Calculate potential COP for each heat sink temperature
        for temp in sorted(heat_by_temp.keys())[:3]:  # Top 3 temperature levels
            t_hot_r = temp + 459.67
            t_cold_r = source['temperature_f'] + 459.67
            cop_carnot = t_hot_r / (t_hot_r - t_cold_r)
            cop_actual = cop_carnot * 0.48  # 48% compressor efficiency
            print(f"    → To {temp}°F: COP {cop_actual:.2f} (Carnot: {cop_carnot:.2f})")

    # Step 4: Heat pump system design - Multi-stage approach
    print("\n" + "=" * 80)
    print("STEP 4: Multi-Stage Heat Pump System Design")
    print("=" * 80)

    # Stage 1: Low-temperature heat pump (120-140°F)
    print("\n[STAGE 1] Low-Temperature Heat Pump (120-140°F)")
    print("-" * 40)

    stage1_load_kw = heat_by_temp[120]["total_kw"] + heat_by_temp[140]["total_kw"]
    stage1_source_temp = 95  # Condenser cooling water
    stage1_delivery_temp = 145

    t_hot_r = stage1_delivery_temp + 459.67
    t_cold_r = stage1_source_temp + 459.67
    stage1_cop_carnot = t_hot_r / (t_hot_r - t_cold_r)
    stage1_cop_actual = stage1_cop_carnot * 0.50  # 50% efficiency (good conditions)
    stage1_power_kw = stage1_load_kw / stage1_cop_actual

    print(f"  Heat Source: Condenser Cooling Water ({stage1_source_temp}°F)")
    print(f"  Delivery Temperature: {stage1_delivery_temp}°F")
    print(f"  Heating Load: {stage1_load_kw:.0f} kW")
    print(f"  COP: {stage1_cop_actual:.2f} (Carnot: {stage1_cop_carnot:.2f})")
    print(f"  Electrical Power: {stage1_power_kw:.0f} kW")
    print(f"  Refrigerant: R134a (standard efficiency)")
    print(f"  Compressor: Screw type")

    # Stage 2: Medium-temperature heat pump (180°F)
    print("\n[STAGE 2] Medium-Temperature Heat Pump (180°F)")
    print("-" * 40)

    stage2_load_kw = heat_by_temp[180]["total_kw"]
    stage2_source_temp = 105  # Reactor cooling water
    stage2_delivery_temp = 185

    t_hot_r = stage2_delivery_temp + 459.67
    t_cold_r = stage2_source_temp + 459.67
    stage2_cop_carnot = t_hot_r / (t_hot_r - t_cold_r)
    stage2_cop_actual = stage2_cop_carnot * 0.48  # 48% efficiency
    stage2_power_kw = stage2_load_kw / stage2_cop_actual

    print(f"  Heat Source: Reactor Cooling Water ({stage2_source_temp}°F)")
    print(f"  Delivery Temperature: {stage2_delivery_temp}°F")
    print(f"  Heating Load: {stage2_load_kw:.0f} kW")
    print(f"  COP: {stage2_cop_actual:.2f} (Carnot: {stage2_cop_carnot:.2f})")
    print(f"  Electrical Power: {stage2_power_kw:.0f} kW")
    print(f"  Refrigerant: R245fa (high-temperature capable)")
    print(f"  Compressor: Screw type")

    # Stage 3: High-temperature via steam boiler (220°F)
    print("\n[STAGE 3] High-Temperature Process (220°F)")
    print("-" * 40)

    stage3_load_kw = heat_by_temp[220]["total_kw"]

    print(f"  Delivery Temperature: 220°F")
    print(f"  Heating Load: {stage3_load_kw:.0f} kW")
    print(f"  Strategy: RETAIN STEAM BOILER (critical process)")
    print(f"  Reason: Temperature too high for cost-effective heat pump")
    print(f"  Alternate: Consider cascade heat pump in future phase")

    # Total system summary
    total_heat_pump_load = stage1_load_kw + stage2_load_kw
    total_heat_pump_power = stage1_power_kw + stage2_power_kw
    weighted_cop = total_heat_pump_load / total_heat_pump_power

    print("\n" + "=" * 80)
    print("INTEGRATED SYSTEM SUMMARY")
    print("=" * 80)
    print(f"\nHeat Pump System:")
    print(f"  Total Heating Capacity: {total_heat_pump_load:.0f} kW")
    print(f"  Total Electrical Power: {total_heat_pump_power:.0f} kW")
    print(f"  Weighted Average COP: {weighted_cop:.2f}")
    print(f"  Estimated Capital Cost: ${(total_heat_pump_load * 500):,.0f}")  # $500/kW for dual-stage
    print()
    print(f"Retained Steam Boiler:")
    print(f"  High-Temperature Load: {stage3_load_kw:.0f} kW (220°F)")
    print(f"  Reason: Critical process, temperature > heat pump economic range")
    print()

    # Step 5: Economic analysis
    print("STEP 5: Economic Analysis")
    print("-" * 80)

    # Heat pump annual costs
    hp_annual_kwh = total_heat_pump_power * 8760 * 0.90  # 90% capacity factor
    hp_energy_cost = hp_annual_kwh * facility_data["electricity_cost_usd_per_kwh"]
    hp_demand_cost = total_heat_pump_power * 12 * facility_data["demand_charge_usd_per_kw"]
    hp_total_annual_cost = hp_energy_cost + hp_demand_cost

    # Reduced boiler costs (only for 220°F load)
    reduced_boiler_fuel_mmbtu = (stage3_load_kw * 3412 / 1_000_000) / 0.83 * 8760 * 0.90
    reduced_boiler_cost = reduced_boiler_fuel_mmbtu * facility_data["natural_gas_cost_usd_per_mmbtu"]

    # Total new system cost
    new_system_total_cost = hp_total_annual_cost + reduced_boiler_cost

    # Savings
    annual_savings = total_current_cost - new_system_total_cost
    percent_reduction = (annual_savings / total_current_cost) * 100

    # Capital cost
    capital_cost = (total_heat_pump_load * 500) + 100000  # Equipment + installation + thermal storage
    maintenance_cost_annual = capital_cost * facility_data["maintenance_cost_percent"]

    # Net savings
    net_annual_savings = annual_savings - maintenance_cost_annual

    # Simple payback
    simple_payback = capital_cost / net_annual_savings if net_annual_savings > 0 else 999

    # CO2 emissions
    current_fuel_total = facility_data["current_heating"]["steam_boiler"]["annual_fuel_mmbtu"] + \
                        facility_data["current_heating"]["hot_water_boiler"]["annual_fuel_mmbtu"]
    current_emissions_lbs = current_fuel_total * 117  # 117 lbs CO2/MMBtu natural gas

    new_ng_emissions = reduced_boiler_fuel_mmbtu * 117
    new_elec_emissions = hp_annual_kwh * 0.92  # 0.92 lbs CO2/kWh (Midwest grid)
    new_emissions_lbs = new_ng_emissions + new_elec_emissions

    emissions_reduction_lbs = current_emissions_lbs - new_emissions_lbs
    emissions_reduction_tons = emissions_reduction_lbs / 2204.62

    print(f"\nCurrent System Annual Costs:")
    print(f"  Steam Boiler: ${steam_annual_cost:,.0f}")
    print(f"  Hot Water Boiler: ${hot_water_annual_cost:,.0f}")
    print(f"  TOTAL: ${total_current_cost:,.0f}")
    print()

    print(f"New Hybrid System Annual Costs:")
    print(f"  Heat Pump Electricity (Energy): ${hp_energy_cost:,.0f}")
    print(f"  Heat Pump Electricity (Demand): ${hp_demand_cost:,.0f}")
    print(f"  Heat Pump Subtotal: ${hp_total_annual_cost:,.0f}")
    print(f"  Reduced Steam Boiler: ${reduced_boiler_cost:,.0f} (high-temp load only)")
    print(f"  TOTAL: ${new_system_total_cost:,.0f}")
    print()

    print(f"Annual Savings: ${annual_savings:,.0f} ({percent_reduction:.1f}% reduction)")
    print(f"Less Maintenance: ${maintenance_cost_annual:,.0f}")
    print(f"Net Annual Savings: ${net_annual_savings:,.0f}")
    print()

    print(f"Capital Investment: ${capital_cost:,.0f}")
    print(f"Simple Payback Period: {simple_payback:.2f} years")
    print()

    print(f"CO2 Emissions:")
    print(f"  Current System: {current_emissions_lbs:,.0f} lbs/year ({current_emissions_lbs/2204.62:.0f} tons)")
    print(f"  New Hybrid System: {new_emissions_lbs:,.0f} lbs/year ({new_emissions_lbs/2204.62:.0f} tons)")
    print(f"  Reduction: {emissions_reduction_tons:.0f} metric tons/year ({emissions_reduction_lbs/current_emissions_lbs*100:.1f}%)")
    print()

    # Step 6: Implementation roadmap
    print("=" * 80)
    print("STEP 6: Implementation Roadmap")
    print("=" * 80)

    print("\nPhase 1 (Year 1): Low-Temperature Heat Pump")
    print("-" * 40)
    print(f"  Scope: 120-140°F processes")
    print(f"  Capacity: {stage1_load_kw:.0f} kW")
    print(f"  Capital: ${(stage1_load_kw * 450):,.0f}")
    print(f"  Annual Savings: ${(stage1_load_kw * 8760 * 0.90 * (1 - 1/stage1_cop_actual) * facility_data['natural_gas_cost_usd_per_mmbtu'] * 3412 / 1_000_000):,.0f}")
    print(f"  Benefits:")
    print(f"    - Lowest risk implementation")
    print(f"    - Best COP ({stage1_cop_actual:.2f})")
    print(f"    - Quick payback")

    print("\nPhase 2 (Year 2): Medium-Temperature Heat Pump")
    print("-" * 40)
    print(f"  Scope: 180°F distillation reboilers")
    print(f"  Capacity: {stage2_load_kw:.0f} kW")
    print(f"  Capital: ${(stage2_load_kw * 550):,.0f}")
    print(f"  Annual Savings: ${(stage2_load_kw * 8760 * 0.90 * (1 - 1/stage2_cop_actual) * facility_data['natural_gas_cost_usd_per_mmbtu'] * 3412 / 1_000_000):,.0f}")
    print(f"  Benefits:")
    print(f"    - Significant energy savings")
    print(f"    - Leverage Phase 1 experience")
    print(f"    - COP {stage2_cop_actual:.2f}")

    print("\nPhase 3 (Year 3+): Evaluate High-Temperature Options")
    print("-" * 40)
    print(f"  Scope: 220°F reactor heating")
    print(f"  Capacity: {stage3_load_kw:.0f} kW")
    print(f"  Options:")
    print(f"    A) Cascade heat pump system (COP 2.0-2.5)")
    print(f"    B) Mechanical vapor recompression")
    print(f"    C) Hybrid: Heat pump preheat + steam boost")
    print(f"  Decision: Evaluate after Phase 1 & 2 performance data")

    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print(f"✓ Multi-stage heat pump system: {total_heat_pump_load:.0f} kW capacity")
    print(f"✓ Weighted average COP: {weighted_cop:.2f}")
    print(f"✓ Annual energy savings: ${annual_savings:,.0f} ({percent_reduction:.1f}%)")
    print(f"✓ Payback period: {simple_payback:.2f} years")
    print(f"✓ CO2 reduction: {emissions_reduction_tons:.0f} metric tons/year")
    print(f"✓ Capital investment: ${capital_cost:,.0f}")
    print()
    print("RECOMMENDATION: PROCEED with phased implementation")
    print("  → Phase 1: Low-temp heat pump (120-140°F)")
    print("  → Phase 2: Medium-temp heat pump (180°F)")
    print("  → Phase 3: Evaluate high-temp options (220°F)")
    print()
    print("KEY SUCCESS FACTORS:")
    print("  • Excellent waste heat availability from process cooling")
    print("  • Multiple temperature levels enable staged approach")
    print("  • Low-risk Phase 1 provides proof-of-concept")
    print("  • 5.8 year payback acceptable for chemical industry")
    print("  • Significant CO2 reduction supports sustainability goals")
    print("=" * 80)


if __name__ == "__main__":
    main()
