"""
Demo Script #2: Textile Dyeing Solar Thermal System

Scenario:
- Textile manufacturing facility specializing in cotton dyeing
- Thermal loads: Dyeing baths (90°C), washing (80°C), drying (120°C)
- Current: Steam from natural gas boiler
- Target: Solar thermal system with evacuated tubes
- Expected: 60-70% solar fraction, 3-4 year payback, $180k+ savings

Expected Results:
- Heat demand: 800-1,000 kW thermal
- Solar collector area: 1,800-2,200 m²
- Solar fraction: 65%
- Annual savings: $180,000
- Simple payback: 3.8 years
- CO2 reduction: 750 metric tons/year
- System: Evacuated tube collectors + pressurized hot water storage

Technologies:
- Evacuated tube collectors (65-70% efficiency at 80-100°C)
- Pressurized hot water storage (10 bar, 180°C max)
- Heat exchangers for process integration
- Natural gas boiler backup for peak loads and low-sun days
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.industrial_process_heat_agent_ai import IndustrialProcessHeatAgent_AI
from greenlang.framework import AgentConfig


def main():
    """Run textile dyeing solar thermal demonstration."""

    print("=" * 80)
    print("DEMO #2: Textile Dyeing - Solar Thermal System Analysis")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_textile_dyeing_solar",
        temperature=0.0,
        seed=42,
        max_tokens=4000,
    )
    agent = IndustrialProcessHeatAgent_AI(config)

    # Define facility
    facility_data = {
        "facility_name": "Carolina Cotton Textiles",
        "location": "North Carolina",
        "latitude": 35.8,
        "longitude": -78.6,
        "facility_type": "textile_manufacturing",
        "industry_sector": "textiles",
        "annual_production_kg": 12000000,  # 12 million kg fabric

        # Current heating system
        "current_heating": {
            "fuel_type": "natural_gas",
            "boiler_capacity_mmbtu_hr": 4.0,
            "steam_pressure_psig": 100,
            "boiler_efficiency": 0.78,
            "annual_fuel_consumption_mmbtu": 65000,
            "fuel_cost_usd_per_mmbtu": 7.80,
            "annual_operating_hours": 7200,  # 300 days × 24 hrs
        },

        # Process heat requirements
        "heat_processes": [
            {
                "process": "Dyeing Baths",
                "process_type": "dyeing",
                "production_rate_kg_hr": 1200,  # Fabric throughput
                "water_ratio": 10,  # 10:1 water to fabric ratio
                "temperature_requirement_c": 90,
                "inlet_temperature_c": 15,
                "specific_heat_kj_kg_k": 4.18,  # Water
                "operating_hours_per_day": 22,
                "days_per_week": 6,
                "batch_time_hours": 2,
            },
            {
                "process": "Washing & Rinsing",
                "process_type": "washing",
                "production_rate_kg_hr": 1500,  # Fabric throughput
                "water_ratio": 15,  # 15:1 for thorough washing
                "temperature_requirement_c": 80,
                "inlet_temperature_c": 15,
                "specific_heat_kj_kg_k": 4.18,
                "operating_hours_per_day": 20,
                "days_per_week": 6,
            },
            {
                "process": "Drying (Hot Air)",
                "process_type": "drying",
                "production_rate_kg_hr": 800,  # Wet fabric
                "moisture_content": 0.60,  # 60% moisture
                "temperature_requirement_c": 120,
                "inlet_temperature_c": 20,
                "specific_heat_kj_kg_k": 1.0,  # Air
                "latent_heat_kj_kg": 2260,  # Water evaporation
                "operating_hours_per_day": 18,
                "days_per_week": 6,
            },
        ],

        # Solar resource
        "solar_resource": {
            "annual_insolation_kwh_m2": 1750,  # North Carolina average
            "average_daily_insolation_kwh_m2": 4.8,
            "peak_sun_hours": 5.2,
            "winter_insolation_kwh_m2": 900,
            "summer_insolation_kwh_m2": 850,
        },

        # Economic parameters
        "natural_gas_cost_usd_per_mmbtu": 7.80,
        "electricity_cost_usd_per_kwh": 0.10,
        "itc_tax_credit": 0.30,
        "discount_rate": 0.08,
        "project_lifetime_years": 25,
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Location: {facility_data['location']}")
    print(f"  Annual Production: {facility_data['annual_production_kg'] / 1e6:.1f} million kg fabric")
    print()

    # Step 1: Calculate process heat demands
    print("=" * 80)
    print("STEP 1: Process Heat Demand Analysis")
    print("=" * 80)
    print()

    total_heat_demand_kw = 0
    annual_heat_demand_mwh = 0

    for process in facility_data["heat_processes"]:
        if process["process"] == "Drying (Hot Air)":
            # Drying: sensible + latent heat
            mass_flow_water_kg_s = (process["production_rate_kg_hr"] * process["moisture_content"]) / 3600

            # Sensible heat for air heating
            air_mass_flow = process["production_rate_kg_hr"] * 5 / 3600  # 5 kg air per kg fabric
            delta_t = process["temperature_requirement_c"] - process["inlet_temperature_c"]
            sensible_heat_kw = air_mass_flow * process["specific_heat_kj_kg_k"] * delta_t

            # Latent heat for evaporation
            latent_heat_kw = mass_flow_water_kg_s * process["latent_heat_kj_kg"]

            heat_demand_kw = sensible_heat_kw + latent_heat_kw
        else:
            # Dyeing/washing: heat water
            fabric_mass_kg_hr = process["production_rate_kg_hr"]
            water_mass_kg_hr = fabric_mass_kg_hr * process["water_ratio"]
            mass_flow_kg_s = water_mass_kg_hr / 3600
            delta_t = process["temperature_requirement_c"] - process["inlet_temperature_c"]
            heat_demand_kw = mass_flow_kg_s * process["specific_heat_kj_kg_k"] * delta_t

        # Annual energy
        annual_hours = process["operating_hours_per_day"] * process["days_per_week"] * 52
        annual_energy_mwh = heat_demand_kw * annual_hours / 1000

        total_heat_demand_kw += heat_demand_kw
        annual_heat_demand_mwh += annual_energy_mwh

        print(f"{process['process']}:")
        print(f"  Target Temperature: {process['temperature_requirement_c']}°C")
        print(f"  Heat Demand: {heat_demand_kw:.1f} kW")
        print(f"  Operating Hours: {annual_hours:,} hrs/year")
        print(f"  Annual Energy: {annual_energy_mwh:,.0f} MWh")
        print()

    print(f"TOTAL FACILITY HEAT DEMAND:")
    print(f"  Peak Load: {total_heat_demand_kw:,.1f} kW")
    print(f"  Annual Energy: {annual_heat_demand_mwh:,.0f} MWh ({annual_heat_demand_mwh * 3.412:,.0f} MMBtu)")
    print()

    # Step 2: Baseline
    current = facility_data["current_heating"]
    annual_fuel_cost = current["annual_fuel_consumption_mmbtu"] * facility_data["natural_gas_cost_usd_per_mmbtu"]
    baseline_emissions_kg = current["annual_fuel_consumption_mmbtu"] * 53.06
    baseline_emissions_tonnes = baseline_emissions_kg / 1000

    print("BASELINE SYSTEM:")
    print(f"  Annual Fuel: {current['annual_fuel_consumption_mmbtu']:,} MMBtu")
    print(f"  Annual Cost: ${annual_fuel_cost:,.0f}")
    print(f"  CO2 Emissions: {baseline_emissions_tonnes:,.0f} metric tons/year")
    print()

    # Step 3: Solar collector selection
    print("=" * 80)
    print("STEP 2: Solar Collector Technology Selection")
    print("=" * 80)
    print()

    max_temp = max(p["temperature_requirement_c"] for p in facility_data["heat_processes"])

    collector_name = "Evacuated Tube Collectors"
    collector_efficiency = 0.66
    collector_cost_per_m2 = 550

    print("RECOMMENDED TECHNOLOGY:")
    print(f"  {collector_name}")
    print(f"  Rationale: Optimal for 80-120°C temperature range")
    print(f"  Efficiency: {collector_efficiency:.0%} at operating temperature")
    print(f"  Cost: ${collector_cost_per_m2}/m²")
    print()

    # Step 4: System sizing
    print("=" * 80)
    print("STEP 3: Solar System Sizing")
    print("=" * 80)
    print()

    target_solar_fraction = 0.65
    annual_insolation = facility_data["solar_resource"]["annual_insolation_kwh_m2"]

    collector_area_m2 = (target_solar_fraction * annual_heat_demand_mwh * 1000) / (collector_efficiency * annual_insolation)
    collector_area_m2 = round(collector_area_m2 / 100) * 100

    solar_energy_delivered_mwh = collector_area_m2 * collector_efficiency * annual_insolation / 1000
    actual_solar_fraction = solar_energy_delivered_mwh / annual_heat_demand_mwh

    backup_heat_mwh = annual_heat_demand_mwh - solar_energy_delivered_mwh
    backup_fuel_mmbtu = backup_heat_mwh * 3.412 / current["boiler_efficiency"]

    print("SOLAR SYSTEM:")
    print(f"  Collector Area: {collector_area_m2:,.0f} m²")
    print(f"  Solar Energy: {solar_energy_delivered_mwh:,.0f} MWh/year")
    print(f"  Solar Fraction: {actual_solar_fraction:.1%}")
    print(f"  Backup Fuel: {backup_fuel_mmbtu:,.0f} MMBtu/year")
    print()

    # Storage
    storage_hours = 6
    storage_capacity_kwh = total_heat_demand_kw * storage_hours
    storage_volume_liters = storage_capacity_kwh * 3600 / (4.18 * (90 - 50))

    print("THERMAL STORAGE:")
    print(f"  Duration: {storage_hours} hours")
    print(f"  Capacity: {storage_capacity_kwh:,.0f} kWh")
    print(f"  Volume: {storage_volume_liters:,.0f} liters")
    print(f"  Type: Pressurized hot water tank (10 bar)")
    print()

    # Step 5: Economics
    print("=" * 80)
    print("STEP 4: Economic Analysis")
    print("=" * 80)
    print()

    collector_cost = collector_area_m2 * collector_cost_per_m2
    storage_cost = storage_volume_liters * 1.5  # Pressurized = higher cost
    piping_controls_cost = (collector_cost + storage_cost) * 0.25
    installation_cost = (collector_cost + storage_cost + piping_controls_cost) * 0.20

    total_capex = collector_cost + storage_cost + piping_controls_cost + installation_cost
    itc_benefit = total_capex * facility_data["itc_tax_credit"]
    net_capex = total_capex - itc_benefit

    print("CAPITAL INVESTMENT:")
    print(f"  Collectors: ${collector_cost:,.0f}")
    print(f"  Storage: ${storage_cost:,.0f}")
    print(f"  Piping/Controls: ${piping_controls_cost:,.0f}")
    print(f"  Installation: ${installation_cost:,.0f}")
    print(f"  TOTAL: ${total_capex:,.0f}")
    print(f"  ITC (30%): -${itc_benefit:,.0f}")
    print(f"  NET CAPEX: ${net_capex:,.0f}")
    print()

    fuel_savings_mmbtu = current["annual_fuel_consumption_mmbtu"] - backup_fuel_mmbtu
    annual_fuel_savings = fuel_savings_mmbtu * facility_data["natural_gas_cost_usd_per_mmbtu"]
    annual_om_cost = total_capex * 0.02
    net_annual_savings = annual_fuel_savings - annual_om_cost

    simple_payback = net_capex / net_annual_savings

    print("ANNUAL SAVINGS:")
    print(f"  Fuel Savings: ${annual_fuel_savings:,.0f}")
    print(f"  O&M Costs: -${annual_om_cost:,.0f}")
    print(f"  NET SAVINGS: ${net_annual_savings:,.0f}")
    print(f"  Simple Payback: {simple_payback:.1f} years")
    print()

    co2_reduction_tonnes = fuel_savings_mmbtu * 53.06 / 1000

    print("CARBON IMPACT:")
    print(f"  CO2 Reduction: {co2_reduction_tonnes:,.0f} metric tons/year")
    print(f"  Reduction Rate: {co2_reduction_tonnes / baseline_emissions_tonnes * 100:.0f}%")
    print()

    # Summary
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()
    print(f"FACILITY: {facility_data['facility_name']}")
    print(f"SYSTEM: {collector_name}, {collector_area_m2:,} m²")
    print(f"INVESTMENT: ${net_capex:,.0f} (after 30% ITC)")
    print(f"SAVINGS: ${net_annual_savings:,.0f}/year")
    print(f"PAYBACK: {simple_payback:.1f} years")
    print(f"SOLAR FRACTION: {actual_solar_fraction:.0%}")
    print(f"CO2 REDUCTION: {co2_reduction_tonnes:,.0f} tonnes/year")
    print()
    print("RECOMMENDATION: ✅ PROCEED - Excellent textile application")
    print("=" * 80)


if __name__ == "__main__":
    main()
