"""
demos/wtt_boundary_demo.py

Well-to-Tank (WTT) Boundary Support Demo

Demonstrates:
- WTT (Well-to-Tank): Upstream emissions only
- WTW (Well-to-Wheel): Full lifecycle = WTT + combustion
- Comparison of emission boundaries

Author: GreenLang Framework Team
Date: October 2025
"""

from greenlang.agents.fuel_agent_ai_v2 import FuelAgentAI_v2
from greenlang.data import wtt_emission_factors
import json


def demo_wtt_factors():
    """Demo: Retrieve WTT factors from database."""
    print("=" * 80)
    print("DEMO 1: WTT Factor Retrieval")
    print("=" * 80)

    fuels = [
        ("diesel", "gallons", "US"),
        ("gasoline", "gallons", "US"),
        ("natural_gas", "therms", "US"),
        ("electricity", "kWh", "US"),
        ("coal", "tons", "US"),
    ]

    for fuel_type, unit, country in fuels:
        wtt_factor, source = wtt_emission_factors.get_wtt_factor(
            fuel_type, unit, country
        )

        print(f"\n{fuel_type.upper()} ({unit}, {country}):")
        print(f"  WTT Factor: {wtt_factor:.3f} kgCO2e/{unit}")
        print(f"  Source: {source}")


def demo_wtt_ratios():
    """Demo: Typical WTT ratios by fuel type."""
    print("\n" + "=" * 80)
    print("DEMO 2: Typical WTT Ratios")
    print("=" * 80)

    print("\nWTT as percentage of combustion emissions:")
    for fuel_type, ratio in wtt_emission_factors.TYPICAL_WTT_RATIOS.items():
        print(f"  {fuel_type:20s} {ratio*100:>5.1f}%")


def demo_boundary_comparison():
    """Demo: Compare combustion, WTT, and WTW boundaries."""
    print("\n" + "=" * 80)
    print("DEMO 3: Boundary Comparison (Diesel, 1000 gallons)")
    print("=" * 80)

    agent = FuelAgentAI_v2(enable_fast_path=True)

    base_payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
        "country": "US",
        "response_format": "compact",
    }

    boundaries = ["combustion", "WTT", "WTW"]
    results = {}

    for boundary in boundaries:
        payload = {**base_payload, "boundary": boundary}
        result = agent.run(payload)

        if result["success"]:
            results[boundary] = result["data"]["co2e_emissions_kg"]
        else:
            print(f"\n❌ {boundary} failed: {result['error']}")

    # Display results
    print("\nEmissions by boundary:")
    print(f"  Combustion (direct): {results.get('combustion', 0):>10.2f} kgCO2e")
    print(f"  WTT (upstream):      {results.get('WTT', 0):>10.2f} kgCO2e")
    print(f"  WTW (full lifecycle):{results.get('WTW', 0):>10.2f} kgCO2e")

    # Calculate percentages
    if "combustion" in results and "WTW" in results:
        wtt_pct = ((results["WTW"] - results["combustion"]) / results["WTW"]) * 100
        print(f"\nWTT represents {wtt_pct:.1f}% of total lifecycle emissions")


def demo_wtt_calculation():
    """Demo: WTT boundary calculation with FuelAgentAI v2."""
    print("\n" + "=" * 80)
    print("DEMO 4: WTT Boundary Calculation")
    print("=" * 80)

    agent = FuelAgentAI_v2(enable_fast_path=True)

    payload = {
        "fuel_type": "diesel",
        "amount": 500,
        "unit": "gallons",
        "country": "US",
        "boundary": "WTT",
        "response_format": "enhanced",
    }

    print("\n📋 Input:")
    print(json.dumps(payload, indent=2))

    result = agent.run(payload)

    print("\n✅ Result:")
    if result["success"]:
        data = result["data"]
        print(f"  Total WTT Emissions: {data['co2e_emissions_kg']:.2f} kgCO2e")
        print(f"  Boundary: {data['boundary']}")
        print(f"  Multi-gas breakdown:")
        print(f"    CO2: {data['ghg_breakdown']['CO2_kg']:.2f} kg")
        print(f"    CH4: {data['ghg_breakdown']['CH4_kg']:.6f} kg")
        print(f"    N2O: {data['ghg_breakdown']['N2O_kg']:.6f} kg")
        print(f"\n  Provenance:")
        print(f"    Source: {data['provenance']['source_org']}")
        print(f"    Citation: {data['provenance']['citation'][:80]}...")
    else:
        print(f"  ❌ Error: {result['error']}")


def demo_wtw_calculation():
    """Demo: WTW boundary calculation with FuelAgentAI v2."""
    print("\n" + "=" * 80)
    print("DEMO 5: WTW (Full Lifecycle) Boundary Calculation")
    print("=" * 80)

    agent = FuelAgentAI_v2(enable_fast_path=True)

    payload = {
        "fuel_type": "natural_gas",
        "amount": 10000,
        "unit": "therms",
        "country": "US",
        "boundary": "WTW",
        "response_format": "enhanced",
    }

    print("\n📋 Input:")
    print(json.dumps(payload, indent=2))

    result = agent.run(payload)

    print("\n✅ Result:")
    if result["success"]:
        data = result["data"]
        print(f"  Total WTW Emissions: {data['co2e_emissions_kg']:.2f} kgCO2e")
        print(f"  Boundary: {data['boundary']}")
        print(f"  Multi-gas breakdown:")
        print(f"    CO2: {data['ghg_breakdown']['CO2_kg']:.2f} kg")
        print(f"    CH4: {data['ghg_breakdown']['CH4_kg']:.3f} kg")
        print(f"    N2O: {data['ghg_breakdown']['N2O_kg']:.3f} kg")
        print(f"\n  Data Quality Score: {data['dqs']['overall_score']:.2f}/5.0 ({data['dqs']['rating']})")
        print(f"  Uncertainty (±95% CI): {data['uncertainty_95ci_pct']:.1f}%")
    else:
        print(f"  ❌ Error: {result['error']}")


def demo_electricity_wtw():
    """Demo: Electricity WTW (generation + transmission losses)."""
    print("\n" + "=" * 80)
    print("DEMO 6: Electricity WTW (Generation + T&D Losses)")
    print("=" * 80)

    agent = FuelAgentAI_v2(enable_fast_path=True)

    base_payload = {
        "fuel_type": "electricity",
        "amount": 10000,
        "unit": "kWh",
        "country": "US",
        "response_format": "compact",
    }

    # Combustion (generation only)
    combustion_result = agent.run({**base_payload, "boundary": "combustion"})

    # WTT (T&D losses)
    wtt_result = agent.run({**base_payload, "boundary": "WTT"})

    # WTW (generation + T&D)
    wtw_result = agent.run({**base_payload, "boundary": "WTW"})

    print("\n10,000 kWh Electricity Consumption:")
    print(f"  Generation (Scope 2):     {combustion_result['data']['co2e_emissions_kg']:>8.2f} kgCO2e")
    print(f"  T&D Losses (WTT):         {wtt_result['data']['co2e_emissions_kg']:>8.2f} kgCO2e")
    print(f"  Total (WTW):              {wtw_result['data']['co2e_emissions_kg']:>8.2f} kgCO2e")

    td_loss_pct = (wtt_result['data']['co2e_emissions_kg'] / wtw_result['data']['co2e_emissions_kg']) * 100
    print(f"\n  T&D losses represent {td_loss_pct:.1f}% of total electricity emissions")


def demo_wtw_with_renewable_offset():
    """Demo: WTW calculation with renewable offset."""
    print("\n" + "=" * 80)
    print("DEMO 7: WTW with Renewable Offset")
    print("=" * 80)

    agent = FuelAgentAI_v2(enable_fast_path=True)

    base_payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
        "country": "US",
        "boundary": "WTW",
        "response_format": "compact",
    }

    renewable_percentages = [0, 25, 50, 100]

    print("\nWTW Emissions with Renewable Offsets (1000 gallons diesel):")
    print(f"  {'Renewable %':<15} {'WTW Emissions (kgCO2e)':<25} {'Reduction':<15}")
    print("  " + "-" * 60)

    baseline_emissions = None

    for renewable_pct in renewable_percentages:
        payload = {**base_payload, "renewable_percentage": renewable_pct}
        result = agent.run(payload)

        emissions = result["data"]["co2e_emissions_kg"]

        if baseline_emissions is None:
            baseline_emissions = emissions
            reduction_str = "baseline"
        else:
            reduction_pct = ((baseline_emissions - emissions) / baseline_emissions) * 100
            reduction_str = f"-{reduction_pct:.1f}%"

        print(f"  {renewable_pct}%{'':<12} {emissions:>10.2f}{'':<15} {reduction_str}")


def demo_wtw_multigas_breakdown():
    """Demo: Multi-gas breakdown for WTW."""
    print("\n" + "=" * 80)
    print("DEMO 8: WTW Multi-Gas Breakdown (Natural Gas)")
    print("=" * 80)

    agent = FuelAgentAI_v2(enable_fast_path=True)

    payload = {
        "fuel_type": "natural_gas",
        "amount": 5000,
        "unit": "therms",
        "country": "US",
        "boundary": "WTW",
        "response_format": "enhanced",
    }

    result = agent.run(payload)

    if result["success"]:
        data = result["data"]

        print(f"\n5000 therms natural gas (WTW):")
        print(f"\n  GHG Breakdown:")
        print(f"    CO2:  {data['ghg_breakdown']['CO2_kg']:>10.2f} kg")
        print(f"    CH4:  {data['ghg_breakdown']['CH4_kg']:>10.3f} kg  (includes upstream methane leakage)")
        print(f"    N2O:  {data['ghg_breakdown']['N2O_kg']:>10.3f} kg")

        print(f"\n  Total CO2e (IPCC AR6 100-year GWP):")
        print(f"    {data['co2e_emissions_kg']:>10.2f} kgCO2e")

        # Calculate CH4 contribution
        ch4_co2e = data['ghg_breakdown']['CH4_kg'] * 28  # GWP of CH4
        ch4_pct = (ch4_co2e / data['co2e_emissions_kg']) * 100

        print(f"\n  CH4 contribution to CO2e: {ch4_pct:.1f}% (important for natural gas!)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WTT BOUNDARY SUPPORT DEMO")
    print("FuelAgentAI v2 - Well-to-Tank & Well-to-Wheel Analysis")
    print("=" * 80)

    demo_wtt_factors()
    demo_wtt_ratios()
    demo_boundary_comparison()
    demo_wtt_calculation()
    demo_wtw_calculation()
    demo_electricity_wtw()
    demo_wtw_with_renewable_offset()
    demo_wtw_multigas_breakdown()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • WTT = upstream emissions (extraction, refining, transport)")
    print("  • WTW = full lifecycle (combustion + WTT)")
    print("  • Diesel/gasoline WTT ~18-20% of combustion")
    print("  • Natural gas WTT ~18% (includes methane leakage)")
    print("  • Coal WTT ~8% (lower upstream footprint)")
    print("  • Electricity WTT = T&D losses (~8%)")
    print("  • WTW provides complete lifecycle emissions picture")
    print()
