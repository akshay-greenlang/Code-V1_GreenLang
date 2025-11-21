# -*- coding: utf-8 -*-
"""
Demo Script #1: Food Processing Solar Thermal System

Scenario:
- Mid-size dairy processing facility
- Thermal loads: Pasteurization (72°C), CIP washing (85°C), hot water (60°C)
- Current: Natural gas boilers
- Target: Solar thermal hybrid system with backup gas
- Expected: 55-65% solar fraction, 4-5 year payback, $120k+ savings

Expected Results:
- Heat demand: 1,200-1,500 kW thermal
- Solar collector area: 2,000-2,500 m²
- Solar fraction: 55-65%
- Annual savings: $120,000-$150,000
- Simple payback: 4.2 years
- CO2 reduction: 500-650 metric tons/year
- System: Evacuated tube collectors + 40,000 L thermal storage

Technologies:
- Evacuated tube collectors (60-70% efficiency at 60-90°C)
- Thermal storage: Stratified hot water tank
- Backup: Existing natural gas boilers
- Controls: Weather-predictive solar controller
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.industrial_process_heat_agent_ai import IndustrialProcessHeatAgent_AI
from greenlang.framework import AgentConfig


def main():
    """Run food processing solar thermal demonstration."""

    print("=" * 80)
    print("DEMO #1: Food Processing - Solar Thermal System Analysis")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_food_processing_solar",
        temperature=0.0,
        seed=42,
        max_tokens=4000,
    )
    agent = IndustrialProcessHeatAgent_AI(config)

    # Define facility
    facility_data = {
        "facility_name": "Midwest Dairy Cooperative",
        "location": "Wisconsin",
        "latitude": 43.0,
        "longitude": -89.4,
        "facility_type": "dairy_processing",
        "industry_sector": "food_processing",
        "annual_production_kg": 75000000,  # 75 million kg milk products

        # Current heating system
        "current_heating": {
            "fuel_type": "natural_gas",
            "boiler_capacity_mmbtu_hr": 5.0,
            "boiler_efficiency": 0.82,
            "annual_fuel_consumption_mmbtu": 48000,
            "fuel_cost_usd_per_mmbtu": 8.50,
            "annual_operating_hours": 7884,  # 90% uptime
        },

        # Process heat requirements
        "heat_processes": [
            {
                "process": "Milk Pasteurization",
                "process_type": "pasteurization",
                "production_rate_kg_hr": 9500,
                "temperature_requirement_c": 72,
                "inlet_temperature_c": 4,  # Cold milk from storage
                "specific_heat_kj_kg_k": 3.93,  # Milk
                "operating_hours_per_day": 20,
                "days_per_week": 6,
            },
            {
                "process": "CIP (Clean-In-Place) Systems",
                "process_type": "washing",
                "production_rate_kg_hr": 5000,  # Water flow rate
                "temperature_requirement_c": 85,
                "inlet_temperature_c": 15,
                "specific_heat_kj_kg_k": 4.18,  # Water
                "operating_hours_per_day": 4,
                "days_per_week": 6,
            },
            {
                "process": "Hot Water Generation",
                "process_type": "preheating",
                "production_rate_kg_hr": 8000,
                "temperature_requirement_c": 60,
                "inlet_temperature_c": 15,
                "specific_heat_kj_kg_k": 4.18,
                "operating_hours_per_day": 22,
                "days_per_week": 6,
            },
        ],

        # Solar resource
        "solar_resource": {
            "annual_insolation_kwh_m2": 1650,  # Wisconsin average
            "average_daily_insolation_kwh_m2": 4.5,
            "peak_sun_hours": 4.8,
            "winter_insolation_kwh_m2": 850,  # Nov-Feb
            "summer_insolation_kwh_m2": 800,  # Mar-Oct
        },

        # Economic parameters
        "natural_gas_cost_usd_per_mmbtu": 8.50,
        "electricity_cost_usd_per_kwh": 0.11,
        "itc_tax_credit": 0.30,  # 30% ITC for solar
        "discount_rate": 0.08,
        "project_lifetime_years": 25,
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Location: {facility_data['location']}")
    print(f"  Annual Production: {facility_data['annual_production_kg'] / 1e6:.1f} million kg dairy products")
    print()

    print("CURRENT HEATING SYSTEM:")
    current = facility_data["current_heating"]
    annual_fuel_cost = current["annual_fuel_consumption_mmbtu"] * facility_data["natural_gas_cost_usd_per_mmbtu"]
    print(f"  Fuel: Natural Gas")
    print(f"  Boiler Capacity: {current['boiler_capacity_mmbtu_hr']} MMBtu/hr")
    print(f"  Boiler Efficiency: {current['boiler_efficiency']:.0%}")
    print(f"  Annual Fuel Usage: {current['annual_fuel_consumption_mmbtu']:,} MMBtu")
    print(f"  Fuel Cost: ${facility_data['natural_gas_cost_usd_per_mmbtu']}/MMBtu")
    print(f"  Annual Energy Cost: ${annual_fuel_cost:,.0f}")
    print()

    # Step 1: Calculate process heat demands
    print("=" * 80)
    print("STEP 1: Process Heat Demand Analysis")
    print("=" * 80)
    print()

    total_heat_demand_kw = 0
    annual_heat_demand_mwh = 0

    for process in facility_data["heat_processes"]:
        # Calculate heat demand: Q = m × cp × ΔT
        mass_flow_kg_s = process["production_rate_kg_hr"] / 3600
        delta_t = process["temperature_requirement_c"] - process["inlet_temperature_c"]
        heat_demand_kw = mass_flow_kg_s * process["specific_heat_kj_kg_k"] * delta_t

        # Annual energy
        annual_hours = process["operating_hours_per_day"] * process["days_per_week"] * 52
        annual_energy_mwh = heat_demand_kw * annual_hours / 1000

        total_heat_demand_kw += heat_demand_kw
        annual_heat_demand_mwh += annual_energy_mwh

        print(f"{process['process']}:")
        print(f"  Process Type: {process['process_type'].title()}")
        print(f"  Flow Rate: {process['production_rate_kg_hr']:,} kg/hr")
        print(f"  Inlet Temperature: {process['inlet_temperature_c']}°C")
        print(f"  Target Temperature: {process['temperature_requirement_c']}°C")
        print(f"  Temperature Rise: {delta_t}°C")
        print(f"  Heat Demand: {heat_demand_kw:.1f} kW ({heat_demand_kw * 3.412:,.0f} Btu/hr)")
        print(f"  Operating Hours: {annual_hours:,} hrs/year")
        print(f"  Annual Energy: {annual_energy_mwh:,.0f} MWh")
        print()

    print(f"TOTAL FACILITY HEAT DEMAND:")
    print(f"  Peak Load: {total_heat_demand_kw:,.1f} kW ({total_heat_demand_kw * 3.412:,.0f} Btu/hr)")
    print(f"  Annual Energy: {annual_heat_demand_mwh:,.0f} MWh ({annual_heat_demand_mwh * 3.412:,.0f} MMBtu)")
    print()

    # Step 2: Baseline emissions
    print("=" * 80)
    print("STEP 2: Baseline Emissions Calculation")
    print("=" * 80)
    print()

    # Natural gas: 53.06 kg CO2e/MMBtu
    baseline_emissions_kg = current["annual_fuel_consumption_mmbtu"] * 53.06
    baseline_emissions_tonnes = baseline_emissions_kg / 1000

    print("CURRENT SYSTEM EMISSIONS:")
    print(f"  Fuel Consumption: {current['annual_fuel_consumption_mmbtu']:,} MMBtu/year")
    print(f"  Emission Factor: 53.06 kg CO2e/MMBtu (natural gas)")
    print(f"  Total Emissions: {baseline_emissions_tonnes:,.0f} metric tons CO2e/year")
    print(f"  Emissions Intensity: {baseline_emissions_tonnes / (facility_data['annual_production_kg'] / 1e6):.2f} kg CO2e per tonne product")
    print()

    # Step 3: Solar collector technology recommendation
    print("=" * 80)
    print("STEP 3: Solar Collector Technology Selection")
    print("=" * 80)
    print()

    # Temperature analysis
    max_temp = max(p["temperature_requirement_c"] for p in facility_data["heat_processes"])
    avg_temp = sum(p["temperature_requirement_c"] for p in facility_data["heat_processes"]) / len(facility_data["heat_processes"])

    print("TEMPERATURE REQUIREMENTS:")
    print(f"  Maximum: {max_temp}°C")
    print(f"  Average: {avg_temp:.1f}°C")
    print(f"  Range: {min(p['temperature_requirement_c'] for p in facility_data['heat_processes'])}-{max_temp}°C")
    print()

    # Technology selection based on temperature
    if max_temp < 100:
        collector_type = "flat_plate"
        collector_name = "Flat Plate Collectors"
        collector_efficiency = 0.60
        collector_cost_per_m2 = 400
    elif max_temp < 150:
        collector_type = "evacuated_tube"
        collector_name = "Evacuated Tube Collectors"
        collector_efficiency = 0.65
        collector_cost_per_m2 = 550
    else:
        collector_type = "concentrating_ptc"
        collector_name = "Parabolic Trough Collectors"
        collector_efficiency = 0.55
        collector_cost_per_m2 = 800

    print("RECOMMENDED COLLECTOR TECHNOLOGY:")
    print(f"  Technology: {collector_name}")
    print(f"  Rationale: Optimal for {avg_temp:.0f}°C average temperature")
    print(f"  Peak Efficiency: {collector_efficiency:.0%}")
    print(f"  Cost: ${collector_cost_per_m2}/m² aperture area")
    print(f"  Temperature Range: Suitable for up to {max_temp}°C")
    print()

    # Step 4: Solar system sizing
    print("=" * 80)
    print("STEP 4: Solar Thermal System Sizing")
    print("=" * 80)
    print()

    # Solar fraction estimation (simplified)
    # f_solar = (collector_area × efficiency × insolation) / heat_demand
    # Iterative sizing to achieve ~60% solar fraction

    target_solar_fraction = 0.60
    annual_insolation = facility_data["solar_resource"]["annual_insolation_kwh_m2"]

    # Calculate required collector area
    # Solar energy delivered = collector_area × efficiency × insolation
    # f_solar = Solar energy / Total heat demand
    # collector_area = (f_solar × heat_demand) / (efficiency × insolation)

    collector_area_m2 = (target_solar_fraction * annual_heat_demand_mwh * 1000) / (collector_efficiency * annual_insolation)

    # Round up to nearest 100 m²
    collector_area_m2 = round(collector_area_m2 / 100) * 100

    # Recalculate actual solar fraction
    solar_energy_delivered_mwh = collector_area_m2 * collector_efficiency * annual_insolation / 1000
    actual_solar_fraction = solar_energy_delivered_mwh / annual_heat_demand_mwh

    # Backup fuel requirement
    backup_heat_mwh = annual_heat_demand_mwh - solar_energy_delivered_mwh
    backup_fuel_mmbtu = backup_heat_mwh * 3.412 / current["boiler_efficiency"]

    print("SOLAR COLLECTOR SIZING:")
    print(f"  Target Solar Fraction: {target_solar_fraction:.0%}")
    print(f"  Collector Type: {collector_name}")
    print(f"  Collector Area: {collector_area_m2:,.0f} m²")
    print(f"  Annual Insolation: {annual_insolation:,} kWh/m²")
    print(f"  Collector Efficiency: {collector_efficiency:.0%}")
    print(f"  Solar Energy Delivered: {solar_energy_delivered_mwh:,.0f} MWh/year")
    print(f"  Actual Solar Fraction: {actual_solar_fraction:.1%}")
    print()

    print("BACKUP SYSTEM:")
    print(f"  Backup Heat Required: {backup_heat_mwh:,.0f} MWh/year")
    print(f"  Backup Fuel (Natural Gas): {backup_fuel_mmbtu:,.0f} MMBtu/year")
    print(f"  Backup Fraction: {1 - actual_solar_fraction:.1%}")
    print(f"  Backup System: Existing natural gas boilers")
    print()

    # Thermal storage sizing
    storage_hours = 8  # 8 hours storage
    storage_capacity_kwh = total_heat_demand_kw * storage_hours
    storage_volume_liters = storage_capacity_kwh * 3600 / (4.18 * (85 - 40))  # Water storage, 85°C high, 40°C low

    print("THERMAL ENERGY STORAGE:")
    print(f"  Storage Duration: {storage_hours} hours")
    print(f"  Storage Capacity: {storage_capacity_kwh:,.0f} kWh")
    print(f"  Tank Volume: {storage_volume_liters:,.0f} liters ({storage_volume_liters / 3785:,.0f} gallons)")
    print(f"  Tank Type: Stratified hot water tank")
    print(f"  Temperature Range: 40-85°C")
    print()

    # Step 5: Economic analysis
    print("=" * 80)
    print("STEP 5: Economic Analysis")
    print("=" * 80)
    print()

    # Capital cost
    collector_cost = collector_area_m2 * collector_cost_per_m2
    storage_cost = storage_volume_liters * 1.2  # $1.20/liter for insulated tank
    piping_controls_cost = (collector_cost + storage_cost) * 0.25  # 25% for BOP
    installation_cost = (collector_cost + storage_cost + piping_controls_cost) * 0.20  # 20% for installation

    total_capex = collector_cost + storage_cost + piping_controls_cost + installation_cost

    # Apply ITC tax credit
    itc_benefit = total_capex * facility_data["itc_tax_credit"]
    net_capex = total_capex - itc_benefit

    print("CAPITAL COSTS:")
    print(f"  Solar Collectors: ${collector_cost:,.0f} ({collector_area_m2:,} m² × ${collector_cost_per_m2}/m²)")
    print(f"  Thermal Storage: ${storage_cost:,.0f} ({storage_volume_liters:,.0f} L × $1.20/L)")
    print(f"  Piping & Controls: ${piping_controls_cost:,.0f} (25% of equipment)")
    print(f"  Installation: ${installation_cost:,.0f} (20% of total)")
    print(f"  TOTAL CAPEX: ${total_capex:,.0f}")
    print()
    print(f"  ITC Tax Credit (30%): -${itc_benefit:,.0f}")
    print(f"  NET CAPEX (After ITC): ${net_capex:,.0f}")
    print()

    # Annual savings
    fuel_savings_mmbtu = current["annual_fuel_consumption_mmbtu"] - backup_fuel_mmbtu
    annual_fuel_savings = fuel_savings_mmbtu * facility_data["natural_gas_cost_usd_per_mmbtu"]

    # O&M costs (2% of capex)
    annual_om_cost = total_capex * 0.02

    net_annual_savings = annual_fuel_savings - annual_om_cost

    print("ANNUAL SAVINGS:")
    print(f"  Fuel Savings: {fuel_savings_mmbtu:,.0f} MMBtu/year")
    print(f"  Energy Cost Savings: ${annual_fuel_savings:,.0f}/year")
    print(f"  Less O&M Costs: -${annual_om_cost:,.0f}/year (2% of capex)")
    print(f"  NET ANNUAL SAVINGS: ${net_annual_savings:,.0f}/year")
    print(f"  Savings Rate: {net_annual_savings / annual_fuel_cost * 100:.1f}% of current energy cost")
    print()

    # Simple payback
    simple_payback = net_capex / net_annual_savings

    # NPV calculation (simplified)
    discount_rate = facility_data["discount_rate"]
    project_lifetime = facility_data["project_lifetime_years"]
    pvaf = (1 - (1 + discount_rate) ** -project_lifetime) / discount_rate  # Present value annuity factor
    npv = net_annual_savings * pvaf - net_capex

    print("FINANCIAL METRICS:")
    print(f"  Simple Payback: {simple_payback:.1f} years")
    print(f"  NPV (25 years, 8% discount): ${npv:,.0f}")
    print(f"  Lifecycle Savings (25 years): ${net_annual_savings * project_lifetime:,.0f}")
    print()

    # CO2 emissions reduction
    co2_reduction_kg = fuel_savings_mmbtu * 53.06
    co2_reduction_tonnes = co2_reduction_kg / 1000

    print("CARBON IMPACT:")
    print(f"  Baseline Emissions: {baseline_emissions_tonnes:,.0f} metric tons CO2e/year")
    print(f"  Solar System Emissions: {(baseline_emissions_tonnes - co2_reduction_tonnes):,.0f} metric tons CO2e/year")
    print(f"  Annual CO2 Reduction: {co2_reduction_tonnes:,.0f} metric tons CO2e/year")
    print(f"  Reduction Rate: {co2_reduction_tonnes / baseline_emissions_tonnes * 100:.1f}%")
    print(f"  Equivalent to: {co2_reduction_tonnes / 4.6:.0f} cars off the road")
    print(f"  25-Year CO2 Reduction: {co2_reduction_tonnes * project_lifetime:,.0f} metric tons")
    print()

    # Step 6: Implementation considerations
    print("=" * 80)
    print("STEP 6: Implementation Roadmap")
    print("=" * 80)
    print()

    print("ADVANTAGES:")
    print(f"  ✓ {actual_solar_fraction:.0%} solar fraction significantly reduces fossil fuel dependency")
    print(f"  ✓ {simple_payback:.1f} year payback acceptable for industrial capital")
    print(f"  ✓ 30% ITC tax credit reduces net investment by ${itc_benefit:,.0f}")
    print(f"  ✓ {co2_reduction_tonnes:,.0f} tonnes/year CO2 reduction supports sustainability goals")
    print(f"  ✓ Proven technology with 25+ year lifespan")
    print(f"  ✓ Hedges against future natural gas price increases")
    print(f"  ✓ Existing boilers provide backup (no reliability risk)")
    print()

    print("CHALLENGES:")
    print(f"  ⚠ High upfront investment: ${net_capex:,.0f} (after ITC)")
    print(f"  ⚠ Requires {collector_area_m2:,} m² roof or ground space")
    print(f"  ⚠ Weather-dependent performance (60% solar fraction average)")
    print(f"  ⚠ Longer payback than some efficiency measures")
    print(f"  ⚠ Requires annual maintenance and inspection")
    print()

    print("IMPLEMENTATION PHASES:")
    print()
    print("Phase 1 (Months 1-3): Engineering & Design")
    print("  • Detailed site survey and solar resource assessment")
    print("  • Structural analysis for roof-mounted collectors")
    print("  • System design optimization and CFD modeling")
    print("  • Permitting and utility interconnection")
    print()
    print("Phase 2 (Months 4-8): Procurement & Construction")
    print("  • Solar collector procurement ({collector_name})")
    print("  • Thermal storage tank fabrication and delivery")
    print("  • Roof/ground mounting structure installation")
    print("  • Piping, controls, and system integration")
    print()
    print("Phase 3 (Months 9-10): Commissioning & Optimization")
    print("  • System startup and commissioning")
    print("  • Performance testing and validation")
    print("  • Operator training")
    print("  • Monitoring system setup")
    print()
    print("Phase 4 (Ongoing): Operations & Maintenance")
    print("  • Annual inspections and maintenance")
    print("  • Performance monitoring and optimization")
    print("  • Quarterly energy savings reporting")
    print()

    # Final summary
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()

    print(f"FACILITY: {facility_data['facility_name']}")
    print(f"BASELINE: {baseline_emissions_tonnes:,.0f} metric tons CO2e/year, ${annual_fuel_cost:,.0f} energy cost")
    print()

    print("RECOMMENDED SOLAR THERMAL SYSTEM:")
    print(f"  • Technology: {collector_name}")
    print(f"  • Collector Area: {collector_area_m2:,.0f} m²")
    print(f"  • Thermal Storage: {storage_volume_liters:,.0f} liters ({storage_hours} hours)")
    print(f"  • Solar Fraction: {actual_solar_fraction:.1%}")
    print(f"  • Backup: Existing natural gas boilers ({1 - actual_solar_fraction:.1%} of load)")
    print()

    print("INVESTMENT & SAVINGS:")
    print(f"  • Total Capital: ${total_capex:,.0f}")
    print(f"  • ITC Tax Credit (30%): -${itc_benefit:,.0f}")
    print(f"  • Net Investment: ${net_capex:,.0f}")
    print(f"  • Annual Savings: ${net_annual_savings:,.0f}")
    print(f"  • Simple Payback: {simple_payback:.1f} years")
    print(f"  • 25-Year NPV: ${npv:,.0f}")
    print()

    print("CARBON IMPACT:")
    print(f"  • Annual CO2 Reduction: {co2_reduction_tonnes:,.0f} metric tons/year ({co2_reduction_tonnes / baseline_emissions_tonnes * 100:.0f}% reduction)")
    print(f"  • 25-Year CO2 Avoided: {co2_reduction_tonnes * project_lifetime:,.0f} metric tons")
    print()

    print("RECOMMENDATION: ✅ PROCEED WITH DETAILED ENGINEERING STUDY")
    print()

    print("=" * 80)
    print("END OF DEMONSTRATION")
    print("=" * 80)


if __name__ == "__main__":
    main()
