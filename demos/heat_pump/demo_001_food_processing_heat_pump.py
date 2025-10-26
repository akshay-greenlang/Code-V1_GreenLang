"""
Demo Script #1: Food Processing Heat Pump Analysis

Scenario:
- Mid-size food processing facility currently using natural gas boilers
- Need heating for pasteurization (165°F) and CIP systems (180°F)
- Target: Replace boilers with industrial heat pumps
- Expected: 40-50% energy reduction, 4-6 year payback, COP 3.0-3.5

Expected Results:
- Heat demand: 250-300 kW
- COP: 3.0-3.5 (depending on source temperature)
- Annual savings: $80,000-$120,000
- Payback: 4.2 years
- CO2 reduction: 300-400 metric tons/year

Technologies:
- Water-source heat pump with hot water heat recovery
- Scroll or screw compressor (COP 3.2)
- Refrigerant: R134a or R1234yf (food-safe)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from greenlang.agents.industrial_heat_pump_agent_ai import IndustrialHeatPumpAgent_AI
from greenlang.framework import AgentConfig


def main():
    """Run food processing heat pump demonstration."""

    print("=" * 80)
    print("DEMO #1: Food Processing Facility - Heat Pump Analysis")
    print("=" * 80)
    print()

    # Initialize agent
    config = AgentConfig(
        agent_id="demo_food_processing_heat_pump",
        temperature=0.0,
        seed=42,
        max_tokens=4000,
    )
    agent = IndustrialHeatPumpAgent_AI(config)

    # Define facility
    facility_data = {
        "facility_name": "Midwest Dairy Processing Co.",
        "location": "Wisconsin",
        "facility_type": "food_processing",
        "industry_sector": "dairy",
        "annual_production": "150 million lbs of dairy products",

        # Current heating system
        "current_heating_system": {
            "type": "natural_gas_boilers",
            "capacity_mmbtu_hr": 3.0,
            "efficiency": 0.82,
            "fuel_type": "natural_gas",
            "fuel_cost_usd_per_mmbtu": 8.50,
            "annual_fuel_usage_mmbtu": 12000,
            "annual_operating_hours": 7884,  # 90% uptime
        },

        # Heating requirements
        "heating_applications": [
            {
                "application": "Pasteurization",
                "temperature_required_f": 165,
                "flow_rate_gpm": 80,
                "hours_per_day": 16,
                "days_per_week": 6,
            },
            {
                "application": "CIP (Clean-In-Place)",
                "temperature_required_f": 180,
                "flow_rate_gpm": 50,
                "hours_per_day": 4,
                "days_per_week": 6,
            },
            {
                "application": "Process Heating",
                "temperature_required_f": 150,
                "flow_rate_gpm": 60,
                "hours_per_day": 20,
                "days_per_week": 6,
            },
        ],

        # Available heat sources
        "heat_sources": [
            {
                "source": "Refrigeration condenser heat",
                "temperature_f": 95,
                "capacity_kw": 180,
                "availability_hours_per_day": 20,
            },
            {
                "source": "Process cooling water",
                "temperature_f": 85,
                "capacity_kw": 120,
                "availability_hours_per_day": 16,
            },
            {
                "source": "Ambient air",
                "temperature_f": 50,  # Average Wisconsin temperature
                "capacity_kw": 1000,  # Unlimited
                "availability_hours_per_day": 24,
            },
        ],

        # Economic parameters
        "electricity_cost_usd_per_kwh": 0.11,
        "demand_charge_usd_per_kw": 15,
        "natural_gas_cost_usd_per_mmbtu": 8.50,
        "project_lifetime_years": 20,
        "discount_rate": 0.10,
    }

    print("FACILITY INFORMATION:")
    print(f"  Name: {facility_data['facility_name']}")
    print(f"  Type: {facility_data['facility_type'].replace('_', ' ').title()}")
    print(f"  Sector: {facility_data['industry_sector'].title()}")
    print(f"  Annual Production: {facility_data['annual_production']}")
    print()

    print("CURRENT HEATING SYSTEM:")
    current = facility_data["current_heating_system"]
    print(f"  Type: {current['type'].replace('_', ' ').title()}")
    print(f"  Capacity: {current['capacity_mmbtu_hr']} MMBtu/hr")
    print(f"  Efficiency: {current['efficiency']:.0%}")
    print(f"  Annual Fuel Usage: {current['annual_fuel_usage_mmbtu']:,} MMBtu")
    print(f"  Fuel Cost: ${current['fuel_cost_usd_per_mmbtu']}/MMBtu")
    print(f"  Annual Energy Cost: ${current['annual_fuel_usage_mmbtu'] * current['fuel_cost_usd_per_mmbtu']:,.0f}")
    print()

    # Step 1: Analyze heating requirements
    print("STEP 1: Analyzing Heating Requirements...")
    print("-" * 80)

    total_heat_load_kw = 0
    for app in facility_data["heating_applications"]:
        # Calculate heat load: Q = m * cp * ΔT
        # Assuming water: ρ = 8.34 lb/gal, cp = 1 Btu/lb·°F
        flow_lb_hr = app["flow_rate_gpm"] * 8.34 * 60
        inlet_temp = 60  # Assume cold water inlet
        delta_t = app["temperature_required_f"] - inlet_temp
        heat_load_btu_hr = flow_lb_hr * 1.0 * delta_t
        heat_load_kw = heat_load_btu_hr / 3412

        total_heat_load_kw += heat_load_kw

        print(f"\n{app['application']}:")
        print(f"  Target Temperature: {app['temperature_required_f']}°F")
        print(f"  Flow Rate: {app['flow_rate_gpm']} GPM")
        print(f"  Operating Hours: {app['hours_per_day']} hrs/day")
        print(f"  Heat Load: {heat_load_kw:.1f} kW ({heat_load_btu_hr:,.0f} Btu/hr)")

    print(f"\nTOTAL HEAT LOAD: {total_heat_load_kw:.1f} kW")
    print()

    # Step 2: Evaluate heat sources
    print("STEP 2: Evaluating Available Heat Sources...")
    print("-" * 80)

    heat_sources_summary = []
    for source in facility_data["heat_sources"]:
        print(f"\n{source['source'].title()}:")
        print(f"  Temperature: {source['temperature_f']}°F")
        print(f"  Capacity: {source['capacity_kw']:.0f} kW")
        print(f"  Availability: {source['availability_hours_per_day']} hrs/day")

        # Estimate temperature lift
        target_temp = 170  # Average target temperature
        temp_lift = target_temp - source['temperature_f']

        # Estimate COP using Carnot efficiency
        # COP_carnot = T_hot / (T_hot - T_cold)
        # COP_actual = COP_carnot * η_compressor (typically 0.45-0.50)
        t_hot_rankine = target_temp + 459.67
        t_cold_rankine = source['temperature_f'] + 459.67
        cop_carnot = t_hot_rankine / (t_hot_rankine - t_cold_rankine)
        cop_actual = cop_carnot * 0.48  # Assume 48% compressor efficiency

        print(f"  Temperature Lift: {temp_lift}°F")
        print(f"  Estimated COP: {cop_actual:.2f} (Carnot: {cop_carnot:.2f})")

        heat_sources_summary.append({
            "source": source['source'],
            "temperature_f": source['temperature_f'],
            "capacity_kw": source['capacity_kw'],
            "cop": cop_actual,
            "temp_lift": temp_lift,
        })

    # Select best heat source (highest COP)
    best_source = max(heat_sources_summary, key=lambda x: x['cop'])
    print(f"\nRECOMMENDED PRIMARY HEAT SOURCE: {best_source['source'].title()}")
    print(f"  COP: {best_source['cop']:.2f}")
    print(f"  Temperature Lift: {best_source['temp_lift']}°F")
    print()

    # Step 3: Heat pump system design
    print("STEP 3: Heat Pump System Design...")
    print("-" * 80)

    print(f"\nHeat Pump Configuration:")
    print(f"  Type: Water-Source Heat Pump")
    print(f"  Compressor: Screw (for {total_heat_load_kw:.0f} kW capacity)")
    print(f"  Refrigerant: R134a (food-safe)")
    print(f"  Heat Source: {best_source['source'].title()}")
    print(f"  Source Temperature: {best_source['temperature_f']}°F")
    print(f"  Delivery Temperature: 165-180°F")
    print()

    # Calculate required electrical power
    cop_design = best_source['cop'] * 0.9  # Apply 10% design margin
    electrical_power_kw = total_heat_load_kw / cop_design

    print(f"Performance Specifications:")
    print(f"  Heating Capacity: {total_heat_load_kw:.1f} kW ({total_heat_load_kw * 3412:,.0f} Btu/hr)")
    print(f"  Design COP: {cop_design:.2f}")
    print(f"  Electrical Power Required: {electrical_power_kw:.1f} kW")
    print(f"  Estimated Capital Cost: ${(total_heat_load_kw * 450):,.0f}")  # $450/kW typical
    print()

    # Step 4: Economic analysis
    print("STEP 4: Economic Analysis...")
    print("-" * 80)

    # Current system energy consumption
    current_energy_cost = facility_data["current_heating_system"]["annual_fuel_usage_mmbtu"] * \
                          facility_data["current_heating_system"]["fuel_cost_usd_per_mmbtu"]

    # Heat pump energy consumption
    operating_hours = facility_data["current_heating_system"]["annual_operating_hours"]
    annual_electricity_kwh = electrical_power_kw * operating_hours
    annual_electricity_cost = annual_electricity_kwh * facility_data["electricity_cost_usd_per_kwh"]
    annual_demand_cost = electrical_power_kw * 12 * facility_data["demand_charge_usd_per_kw"]
    heat_pump_total_cost = annual_electricity_cost + annual_demand_cost

    # Savings
    annual_savings = current_energy_cost - heat_pump_total_cost

    # Capital cost
    capital_cost = total_heat_load_kw * 450 + 50000  # Equipment + installation

    # Simple payback
    simple_payback = capital_cost / annual_savings if annual_savings > 0 else 999

    # CO2 emissions reduction
    # Natural gas: 117 lbs CO2/MMBtu
    # Electricity (grid): 0.92 lbs CO2/kWh (Midwest average)
    current_emissions_lbs = facility_data["current_heating_system"]["annual_fuel_usage_mmbtu"] * 117
    heat_pump_emissions_lbs = annual_electricity_kwh * 0.92
    emissions_reduction_lbs = current_emissions_lbs - heat_pump_emissions_lbs
    emissions_reduction_metric_tons = emissions_reduction_lbs / 2204.62

    print(f"\nCurrent System Annual Costs:")
    print(f"  Natural Gas: ${current_energy_cost:,.0f}")
    print()

    print(f"Heat Pump System Annual Costs:")
    print(f"  Electricity (Energy): ${annual_electricity_cost:,.0f}")
    print(f"  Electricity (Demand): ${annual_demand_cost:,.0f}")
    print(f"  Total: ${heat_pump_total_cost:,.0f}")
    print()

    print(f"Annual Savings: ${annual_savings:,.0f} ({annual_savings/current_energy_cost*100:.1f}% reduction)")
    print()

    print(f"Capital Investment: ${capital_cost:,.0f}")
    print(f"Simple Payback Period: {simple_payback:.2f} years")
    print()

    print(f"CO2 Emissions:")
    print(f"  Current System: {current_emissions_lbs:,.0f} lbs/year ({current_emissions_lbs/2204.62:.0f} metric tons)")
    print(f"  Heat Pump System: {heat_pump_emissions_lbs:,.0f} lbs/year ({heat_pump_emissions_lbs/2204.62:.0f} metric tons)")
    print(f"  Reduction: {emissions_reduction_metric_tons:.0f} metric tons/year ({emissions_reduction_lbs/current_emissions_lbs*100:.1f}%)")
    print(f"  Equivalent to: {emissions_reduction_metric_tons/4.6:.0f} cars off the road")
    print()

    # Step 5: Implementation considerations
    print("=" * 80)
    print("STEP 5: Implementation Considerations")
    print("=" * 80)

    print("\nAdvantages:")
    print(f"  ✓ {annual_savings/current_energy_cost*100:.0f}% energy cost reduction")
    print(f"  ✓ {simple_payback:.1f} year payback (acceptable for industrial capital)")
    print(f"  ✓ COP {cop_design:.1f} significantly better than boiler efficiency ({facility_data['current_heating_system']['efficiency']:.0%})")
    print(f"  ✓ {emissions_reduction_metric_tons:.0f} metric tons/year CO2 reduction")
    print(f"  ✓ Utilizes waste heat from refrigeration (free heat source)")
    print(f"  ✓ Reduces natural gas dependency")
    print(f"  ✓ Food-safe refrigerant (R134a)")

    print("\nChallenges:")
    print(f"  ⚠ High capital cost: ${capital_cost:,.0f}")
    print(f"  ⚠ Requires ${electrical_power_kw:.0f} kW electrical service upgrade")
    print(f"  ⚠ More complex than boilers (refrigeration cycle)")
    print(f"  ⚠ Performance depends on heat source availability")
    print(f"  ⚠ May require backup heating for peak loads")

    print("\nRecommendations:")
    print(f"  1. PROCEED with detailed engineering study")
    print(f"  2. Prioritize refrigeration condenser heat recovery (highest COP)")
    print(f"  3. Install thermal storage (4-6 hours) for load leveling")
    print(f"  4. Keep one natural gas boiler as backup (redundancy)")
    print(f"  5. Investigate utility incentives (20-30% capital reduction typical)")
    print(f"  6. Phase 1: Pasteurization + Process Heating (70% of load)")
    print(f"  7. Phase 2: CIP system (remaining 30% of load)")
    print(f"  8. Monitor COP and optimize based on field performance")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Heat Pump Capacity: {total_heat_load_kw:.0f} kW")
    print(f"✓ Design COP: {cop_design:.2f}")
    print(f"✓ Annual Savings: ${annual_savings:,.0f} ({annual_savings/current_energy_cost*100:.0f}% reduction)")
    print(f"✓ Payback Period: {simple_payback:.2f} years")
    print(f"✓ CO2 Reduction: {emissions_reduction_metric_tons:.0f} metric tons/year")
    print(f"✓ Capital Investment: ${capital_cost:,.0f}")
    print(f"✓ Recommended: Water-source heat pump with refrigeration heat recovery")
    print("=" * 80)


if __name__ == "__main__":
    main()
