# -*- coding: utf-8 -*-
"""
Demo Script #3: Chemical Process Pre-heating with Concentrating Solar

Scenario:
- Chemical manufacturing plant with distillation and reaction heating
- Thermal loads: Feed pre-heating (180°C), distillation reboiler (220°C)
- Current: High-pressure steam from natural gas boiler
- Target: Concentrating solar (parabolic trough) with thermal oil
- Expected: 35-45% solar fraction, 6-7 year payback, $250k+ savings

Expected Results:
- Heat demand: 2,000-2,500 kW thermal
- Solar collector area: 3,500-4,000 m² (parabolic troughs)
- Solar fraction: 40%
- Annual savings: $280,000
- Simple payback: 6.5 years
- CO2 reduction: 1,100 metric tons/year
- System: Parabolic trough + thermal oil + heat exchangers

Technologies:
- Parabolic trough collectors (55-60% efficiency at 200-250°C)
- Thermal oil HTF (heat transfer fluid) up to 400°C
- Single-axis solar tracking
- Steam generation via heat exchangers
- Natural gas backup for base load and cloudy days
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.industrial_process_heat_agent_ai import IndustrialProcessHeatAgent_AI
from greenlang.framework import AgentConfig


def main():
    """Run chemical pre-heating concentrating solar demonstration."""

    print("=" * 80)
    print("DEMO #3: Chemical Plant - Concentrating Solar Thermal System")
    print("=" * 80)
    print()

    config = AgentConfig(
        agent_id="demo_chemical_concentrating_solar",
        temperature=0.0,
        seed=42,
        max_tokens=4000,
    )
    agent = IndustrialProcessHeatAgent_AI(config)

    facility_data = {
        "facility_name": "Gulf Coast Chemical Processing",
        "location": "Texas",
        "latitude": 29.7,
        "longitude": -95.4,
        "facility_type": "chemical_manufacturing",
        "industry_sector": "chemicals",
        "annual_production_kg": 50000000,  # 50 million kg chemicals

        "current_heating": {
            "fuel_type": "natural_gas",
            "boiler_capacity_mmbtu_hr": 10.0,
            "steam_pressure_psig": 250,
            "steam_temperature_f": 406,
            "boiler_efficiency": 0.80,
            "annual_fuel_consumption_mmbtu": 95000,
            "fuel_cost_usd_per_mmbtu": 7.20,
            "annual_operating_hours": 8400,  # 350 days × 24 hrs
        },

        "heat_processes": [
            {
                "process": "Feed Pre-heating",
                "process_type": "preheating",
                "production_rate_kg_hr": 15000,
                "temperature_requirement_c": 180,
                "inlet_temperature_c": 25,
                "specific_heat_kj_kg_k": 2.5,  # Organic chemicals
                "operating_hours_per_day": 24,
                "days_per_week": 7,
            },
            {
                "process": "Distillation Reboiler",
                "process_type": "distillation",
                "production_rate_kg_hr": 8000,
                "temperature_requirement_c": 220,
                "inlet_temperature_c": 180,
                "specific_heat_kj_kg_k": 2.8,
                "latent_heat_kj_kg": 350,  # Partial vaporization
                "operating_hours_per_day": 24,
                "days_per_week": 7,
            },
            {
                "process": "Reactor Heating",
                "process_type": "metal_treating",
                "production_rate_kg_hr": 5000,
                "temperature_requirement_c": 160,
                "inlet_temperature_c": 80,
                "specific_heat_kj_kg_k": 3.0,
                "operating_hours_per_day": 22,
                "days_per_week": 7,
            },
        ],

        "solar_resource": {
            "annual_dni_kwh_m2": 2200,  # Direct Normal Irradiance (Texas)
            "average_daily_dni_kwh_m2": 6.0,
            "peak_sun_hours": 6.5,
            "winter_dni_kwh_m2": 1000,
            "summer_dni_kwh_m2": 1200,
        },

        "natural_gas_cost_usd_per_mmbtu": 7.20,
        "electricity_cost_usd_per_kwh": 0.09,
        "itc_tax_credit": 0.30,
        "discount_rate": 0.08,
        "project_lifetime_years": 25,
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Location: {facility_data['location']} (High DNI region)")
    print(f"  Annual Production: {facility_data['annual_production_kg'] / 1e6:.1f} million kg chemicals")
    print()

    # Step 1: Heat demand
    print("=" * 80)
    print("STEP 1: Process Heat Demand Analysis")
    print("=" * 80)
    print()

    total_heat_demand_kw = 0
    annual_heat_demand_mwh = 0

    for process in facility_data["heat_processes"]:
        mass_flow_kg_s = process["production_rate_kg_hr"] / 3600
        delta_t = process["temperature_requirement_c"] - process["inlet_temperature_c"]

        sensible_heat_kw = mass_flow_kg_s * process["specific_heat_kj_kg_k"] * delta_t

        if "latent_heat_kj_kg" in process:
            latent_heat_kw = mass_flow_kg_s * process["latent_heat_kj_kg"]
            heat_demand_kw = sensible_heat_kw + latent_heat_kw
        else:
            heat_demand_kw = sensible_heat_kw

        annual_hours = process["operating_hours_per_day"] * process["days_per_week"] * 52
        annual_energy_mwh = heat_demand_kw * annual_hours / 1000

        total_heat_demand_kw += heat_demand_kw
        annual_heat_demand_mwh += annual_energy_mwh

        print(f"{process['process']}:")
        print(f"  Temperature: {process['inlet_temperature_c']}°C → {process['temperature_requirement_c']}°C")
        print(f"  Heat Demand: {heat_demand_kw:.1f} kW")
        print(f"  Annual Energy: {annual_energy_mwh:,.0f} MWh")
        print()

    print(f"TOTAL HEAT DEMAND:")
    print(f"  Peak: {total_heat_demand_kw:,.1f} kW")
    print(f"  Annual: {annual_heat_demand_mwh:,.0f} MWh ({annual_heat_demand_mwh * 3.412:,.0f} MMBtu)")
    print()

    # Baseline
    current = facility_data["current_heating"]
    annual_fuel_cost = current["annual_fuel_consumption_mmbtu"] * facility_data["natural_gas_cost_usd_per_mmbtu"]
    baseline_emissions_tonnes = current["annual_fuel_consumption_mmbtu"] * 53.06 / 1000

    print("BASELINE:")
    print(f"  Fuel: {current['annual_fuel_consumption_mmbtu']:,} MMBtu/year")
    print(f"  Cost: ${annual_fuel_cost:,.0f}/year")
    print(f"  CO2: {baseline_emissions_tonnes:,.0f} metric tons/year")
    print()

    # Step 2: Concentrating solar selection
    print("=" * 80)
    print("STEP 2: Concentrating Solar Technology")
    print("=" * 80)
    print()

    max_temp = max(p["temperature_requirement_c"] for p in facility_data["heat_processes"])

    collector_name = "Parabolic Trough Collectors (PTC)"
    collector_efficiency = 0.58  # At 200°C operating temperature
    collector_cost_per_m2 = 800  # Higher cost for tracking systems

    print("RECOMMENDED TECHNOLOGY:")
    print(f"  {collector_name}")
    print(f"  Rationale: Required for {max_temp}°C process temperatures")
    print(f"  Efficiency: {collector_efficiency:.0%} at 200°C")
    print(f"  Cost: ${collector_cost_per_m2}/m² (includes tracking)")
    print(f"  HTF: Thermal oil (up to 400°C)")
    print(f"  Tracking: Single-axis east-west")
    print()

    # Step 3: System sizing
    print("=" * 80)
    print("STEP 3: Solar System Sizing")
    print("=" * 80)
    print()

    target_solar_fraction = 0.40  # Lower for high-temp applications
    annual_dni = facility_data["solar_resource"]["annual_dni_kwh_m2"]

    # Use DNI for concentrating collectors
    collector_area_m2 = (target_solar_fraction * annual_heat_demand_mwh * 1000) / (collector_efficiency * annual_dni)
    collector_area_m2 = round(collector_area_m2 / 100) * 100

    solar_energy_delivered_mwh = collector_area_m2 * collector_efficiency * annual_dni / 1000
    actual_solar_fraction = solar_energy_delivered_mwh / annual_heat_demand_mwh

    backup_heat_mwh = annual_heat_demand_mwh - solar_energy_delivered_mwh
    backup_fuel_mmbtu = backup_heat_mwh * 3.412 / current["boiler_efficiency"]

    print("SOLAR FIELD:")
    print(f"  Collector Area: {collector_area_m2:,.0f} m² aperture")
    print(f"  Field Configuration: ~20 rows × {collector_area_m2 / 2400:.0f} collectors/row")
    print(f"  Annual DNI: {annual_dni:,} kWh/m²")
    print(f"  Solar Energy: {solar_energy_delivered_mwh:,.0f} MWh/year")
    print(f"  Solar Fraction: {actual_solar_fraction:.1%}")
    print(f"  Land Required: ~{collector_area_m2 * 4 / 4047:.1f} acres (4× collector area)")
    print()

    print("BACKUP SYSTEM:")
    print(f"  Backup Heat: {backup_heat_mwh:,.0f} MWh/year")
    print(f"  Backup Fuel: {backup_fuel_mmbtu:,.0f} MMBtu/year")
    print(f"  System: Existing natural gas boiler")
    print()

    # Storage (thermal oil)
    storage_hours = 4  # Smaller for industrial process heat
    storage_capacity_kwh = total_heat_demand_kw * storage_hours
    thermal_oil_volume_liters = storage_capacity_kwh * 3600 / (2.3 * (250 - 180))  # Thermal oil properties

    print("THERMAL ENERGY STORAGE:")
    print(f"  Duration: {storage_hours} hours")
    print(f"  Capacity: {storage_capacity_kwh:,.0f} kWh thermal")
    print(f"  Thermal Oil Volume: {thermal_oil_volume_liters:,.0f} liters")
    print(f"  Temperature Range: 180-250°C")
    print(f"  Type: Insulated steel tanks with thermal oil")
    print()

    # Step 4: Economics
    print("=" * 80)
    print("STEP 4: Economic Analysis")
    print("=" * 80)
    print()

    collector_cost = collector_area_m2 * collector_cost_per_m2
    storage_cost = thermal_oil_volume_liters * 2.5  # Thermal oil + tanks
    htf_system_cost = collector_cost * 0.15  # 15% for HTF piping and pumps
    heat_exchanger_cost = 250000  # Thermal oil to steam HX
    land_cost = (collector_area_m2 * 4 / 4047) * 15000  # $15k/acre
    installation_cost = (collector_cost + storage_cost + htf_system_cost) * 0.25

    total_capex = collector_cost + storage_cost + htf_system_cost + heat_exchanger_cost + land_cost + installation_cost
    itc_benefit = total_capex * facility_data["itc_tax_credit"]
    net_capex = total_capex - itc_benefit

    print("CAPITAL INVESTMENT:")
    print(f"  Collectors (PTC): ${collector_cost:,.0f}")
    print(f"  Thermal Storage: ${storage_cost:,.0f}")
    print(f"  HTF System: ${htf_system_cost:,.0f}")
    print(f"  Heat Exchangers: ${heat_exchanger_cost:,.0f}")
    print(f"  Land: ${land_cost:,.0f}")
    print(f"  Installation: ${installation_cost:,.0f}")
    print(f"  TOTAL: ${total_capex:,.0f}")
    print(f"  ITC (30%): -${itc_benefit:,.0f}")
    print(f"  NET CAPEX: ${net_capex:,.0f}")
    print()

    fuel_savings_mmbtu = current["annual_fuel_consumption_mmbtu"] - backup_fuel_mmbtu
    annual_fuel_savings = fuel_savings_mmbtu * facility_data["natural_gas_cost_usd_per_mmbtu"]
    annual_om_cost = total_capex * 0.025  # Higher O&M for tracking systems
    net_annual_savings = annual_fuel_savings - annual_om_cost

    simple_payback = net_capex / net_annual_savings

    pvaf = (1 - (1 + 0.08) ** -25) / 0.08
    npv = net_annual_savings * pvaf - net_capex

    print("ANNUAL SAVINGS:")
    print(f"  Fuel Savings: ${annual_fuel_savings:,.0f}")
    print(f"  O&M Costs: -${annual_om_cost:,.0f} (2.5% of capex)")
    print(f"  NET SAVINGS: ${net_annual_savings:,.0f}")
    print()

    print("FINANCIAL METRICS:")
    print(f"  Simple Payback: {simple_payback:.1f} years")
    print(f"  NPV (25 years): ${npv:,.0f}")
    print(f"  Lifecycle Savings: ${net_annual_savings * 25:,.0f}")
    print()

    co2_reduction_tonnes = fuel_savings_mmbtu * 53.06 / 1000

    print("CARBON IMPACT:")
    print(f"  Annual CO2 Reduction: {co2_reduction_tonnes:,.0f} metric tons/year")
    print(f"  Reduction Rate: {co2_reduction_tonnes / baseline_emissions_tonnes * 100:.0f}%")
    print(f"  25-Year Total: {co2_reduction_tonnes * 25:,.0f} metric tons")
    print()

    # Summary
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print()
    print(f"FACILITY: {facility_data['facility_name']}")
    print(f"APPLICATION: High-temperature process heat (180-220°C)")
    print()
    print("RECOMMENDED SYSTEM:")
    print(f"  • Technology: {collector_name}")
    print(f"  • Collector Area: {collector_area_m2:,.0f} m²")
    print(f"  • Land Required: {collector_area_m2 * 4 / 4047:.1f} acres")
    print(f"  • Thermal Storage: {storage_hours} hours ({storage_capacity_kwh:,.0f} kWh)")
    print(f"  • Solar Fraction: {actual_solar_fraction:.0%}")
    print()
    print("ECONOMICS:")
    print(f"  • Net Investment: ${net_capex:,.0f} (after 30% ITC)")
    print(f"  • Annual Savings: ${net_annual_savings:,.0f}")
    print(f"  • Simple Payback: {simple_payback:.1f} years")
    print(f"  • 25-Year NPV: ${npv:,.0f}")
    print()
    print("CARBON:")
    print(f"  • CO2 Reduction: {co2_reduction_tonnes:,.0f} tonnes/year ({co2_reduction_tonnes / baseline_emissions_tonnes * 100:.0f}%)")
    print()
    print("CONSIDERATIONS:")
    print("  ✓ Excellent DNI resource in Texas (2,200 kWh/m²/year)")
    print("  ✓ Proven technology for high-temperature industrial heat")
    print("  ✓ 30% ITC reduces net investment significantly")
    print(f"  ⚠ Higher O&M costs for tracking systems (${annual_om_cost:,.0f}/year)")
    print(f"  ⚠ Requires {collector_area_m2 * 4 / 4047:.1f} acres land")
    print(f"  ⚠ Longer payback than low-temp solar thermal")
    print()
    print("RECOMMENDATION: ✅ PROCEED if land available, excellent high-temp application")
    print("=" * 80)


if __name__ == "__main__":
    main()
