"""Demo: AI-Powered Grid Carbon Intensity Lookup

This script demonstrates the GridFactorAgentAI which enhances the original GridFactorAgent
with AI orchestration while preserving all deterministic lookups.

Key Features Demonstrated:
1. Tool-first lookups (all data from database)
2. Natural language explanations of grid intensity
3. Deterministic results (same input -> same output)
4. Temporal analysis (hourly interpolation)
5. Weighted averages for mixed sources
6. Intelligent recommendations for cleaner energy
7. Backward compatibility with GridFactorAgent
8. Performance tracking

Usage:
    python examples/grid_factor_agent_ai_demo.py

Author: GreenLang Framework Team
Date: October 2025
"""

from greenlang.agents.grid_factor_agent_ai import GridFactorAgentAI
from greenlang.agents.grid_factor_agent import GridFactorAgent
import json


def demo_basic_lookup():
    """Demo 1: Basic grid intensity lookup with AI explanation."""
    print("=" * 80)
    print("DEMO 1: Basic Grid Intensity Lookup with AI")
    print("=" * 80)

    agent = GridFactorAgentAI(budget_usd=0.50, enable_explanations=True)

    payload = {
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
    }

    print(f"\nInput: {json.dumps(payload, indent=2)}")
    print("\nLooking up grid intensity...")

    result = agent.run(payload)

    if result["success"]:
        data = result["data"]
        print(f"\nResults:")
        print(f"  Emission Factor: {data['emission_factor']} {data['unit']}")
        print(f"  Country: {data['country']}")
        print(f"  Fuel Type: {data['fuel_type']}")
        print(f"  Source: {data['source']}")
        print(f"  Last Updated: {data['last_updated']}")

        if "grid_mix" in data:
            grid_mix = data["grid_mix"]
            print(f"\nGrid Mix:")
            print(f"  Renewable: {grid_mix.get('renewable', 0)*100:.1f}%")
            print(f"  Fossil: {grid_mix.get('fossil', 0)*100:.1f}%")

        if "explanation" in data:
            print(f"\nAI Explanation:")
            print(f"  {data['explanation']}")

        print(f"\nMetadata:")
        metadata = result.get("metadata", {})
        print(f"  Provider: {metadata.get('provider', 'N/A')}")
        print(f"  Model: {metadata.get('model', 'N/A')}")
        print(f"  Tool calls: {metadata.get('tool_calls', 0)}")
        print(f"  Cost: ${metadata.get('cost_usd', 0):.4f}")
    else:
        print(f"\nError: {result['error']['message']}")

    print()


def demo_country_comparison():
    """Demo 2: Compare grid intensity across countries."""
    print("=" * 80)
    print("DEMO 2: Country Grid Intensity Comparison")
    print("=" * 80)

    agent = GridFactorAgentAI()

    countries = [
        ("US", "United States"),
        ("IN", "India"),
        ("EU", "European Union"),
        ("BR", "Brazil"),
        ("CN", "China"),
        ("AU", "Australia"),
    ]

    print("\nGrid Carbon Intensity Comparison (gCO2/kWh):\n")
    print(f"{'Country':<20} {'Code':<6} {'Intensity':<12} {'Renewable %':<15}")
    print("-" * 60)

    results = []
    for code, name in countries:
        try:
            result = agent._lookup_grid_intensity_impl(
                country=code,
                fuel_type="electricity",
                unit="kWh",
            )
            intensity = result["emission_factor"] * 1000  # Convert to gCO2/kWh
            renewable = result.get("grid_mix", {}).get("renewable", 0) * 100

            results.append({
                "name": name,
                "code": code,
                "intensity": intensity,
                "renewable": renewable,
            })

            print(f"{name:<20} {code:<6} {intensity:<12.0f} {renewable:<15.1f}%")
        except Exception as e:
            print(f"{name:<20} {code:<6} ERROR: {str(e)}")

    # Find cleanest and dirtiest
    if results:
        cleanest = min(results, key=lambda x: x["intensity"])
        dirtiest = max(results, key=lambda x: x["intensity"])

        print(f"\nCleanest Grid: {cleanest['name']} ({cleanest['intensity']:.0f} gCO2/kWh)")
        print(f"Dirtiest Grid: {dirtiest['name']} ({dirtiest['intensity']:.0f} gCO2/kWh)")
        print(f"Difference: {dirtiest['intensity'] - cleanest['intensity']:.0f} gCO2/kWh ({(dirtiest['intensity'] / cleanest['intensity']):.1f}x)")

    print()


def demo_determinism():
    """Demo 3: Deterministic lookups."""
    print("=" * 80)
    print("DEMO 3: Determinism Test")
    print("=" * 80)

    agent = GridFactorAgentAI()

    payload = {
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
    }

    print(f"\nRunning same lookup 3 times...")
    print(f"Input: {payload['country']} {payload['fuel_type']} ({payload['unit']})")

    results = []
    for i in range(3):
        # Use tool directly to show determinism
        result = agent._lookup_grid_intensity_impl(
            country=payload["country"],
            fuel_type=payload["fuel_type"],
            unit=payload["unit"],
        )
        results.append(result["emission_factor"])
        print(f"  Run {i+1}: {result['emission_factor']} {result['unit']}")

    all_same = len(set(results)) == 1
    print(f"\nDeterministic: {all_same} [PASS]" if all_same else f"\nDeterministic: {all_same} [FAIL]")
    print()


def demo_backward_compatibility():
    """Demo 4: Backward compatibility with GridFactorAgent."""
    print("=" * 80)
    print("DEMO 4: Backward Compatibility")
    print("=" * 80)

    original = GridFactorAgent()
    ai_enhanced = GridFactorAgentAI()

    payload = {
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
    }

    print(f"\nInput: {payload['country']} {payload['fuel_type']} ({payload['unit']})")

    # Original agent
    result_orig = original.run(payload)
    factor_orig = result_orig["data"]["emission_factor"] if result_orig["success"] else 0

    # AI agent tool (bypassing AI for exact comparison)
    result_ai = ai_enhanced._lookup_grid_intensity_impl(
        country=payload["country"],
        fuel_type=payload["fuel_type"],
        unit=payload["unit"],
    )
    factor_ai = result_ai["emission_factor"]

    print(f"\nOriginal GridFactorAgent:  {factor_orig} kgCO2e/{payload['unit']}")
    print(f"AI GridFactorAgent Tool:   {factor_ai} kgCO2e/{payload['unit']}")
    print(f"Match: {factor_orig == factor_ai} [PASS]" if factor_orig == factor_ai else f"Match: {factor_orig == factor_ai} [FAIL]")
    print()


def demo_hourly_interpolation():
    """Demo 5: Hourly grid intensity interpolation."""
    print("=" * 80)
    print("DEMO 5: Hourly Grid Intensity Interpolation")
    print("=" * 80)

    agent = GridFactorAgentAI()

    # Get base intensity
    base_result = agent._lookup_grid_intensity_impl(
        country="US",
        fuel_type="electricity",
        unit="kWh",
    )
    base_intensity = base_result["emission_factor"] * 1000  # Convert to gCO2/kWh
    renewable_share = base_result.get("grid_mix", {}).get("renewable", 0)

    print(f"\nUS Grid Base Intensity: {base_intensity:.0f} gCO2/kWh")
    print(f"Renewable Share: {renewable_share*100:.1f}%\n")

    print(f"{'Hour':<8} {'Period':<20} {'Intensity (gCO2/kWh)':<25} {'vs. Average':<15}")
    print("-" * 70)

    # Sample key hours
    sample_hours = [2, 8, 13, 18, 22]

    for hour in sample_hours:
        result = agent._interpolate_hourly_data_impl(
            base_intensity=base_intensity,
            hour=hour,
            renewable_share=renewable_share,
        )

        interpolated = result["interpolated_intensity"]
        period = result["period"].replace("_", " ").title()
        diff_pct = ((interpolated - base_intensity) / base_intensity) * 100

        print(f"{hour:02d}:00   {period:<20} {interpolated:<25.1f} {diff_pct:+.1f}%")

    print("\nNote: Peak hours have higher intensity due to increased fossil fuel generation.")
    print("      Midday has lower intensity due to solar generation (varies by renewable share).")
    print()


def demo_weighted_average():
    """Demo 6: Weighted average for mixed energy sources."""
    print("=" * 80)
    print("DEMO 6: Weighted Average for Mixed Energy Sources")
    print("=" * 80)

    agent = GridFactorAgentAI()

    # Scenario: Facility with mixed energy sources
    print("\nScenario: Manufacturing facility with mixed energy portfolio")
    print("  - 60% grid electricity")
    print("  - 30% on-site solar")
    print("  - 10% backup diesel generator\n")

    # Get intensities
    grid_intensity = 385.0  # gCO2/kWh (US grid)
    solar_intensity = 0.0   # gCO2/kWh (zero emissions)
    diesel_intensity = 700.0  # gCO2/kWh (diesel generator)

    intensities = [grid_intensity, solar_intensity, diesel_intensity]
    weights = [0.6, 0.3, 0.1]

    result = agent._calculate_weighted_average_impl(
        intensities=intensities,
        weights=weights,
    )

    print(f"Individual Intensities:")
    print(f"  Grid electricity:     {grid_intensity:.0f} gCO2/kWh (60%)")
    print(f"  On-site solar:        {solar_intensity:.0f} gCO2/kWh (30%)")
    print(f"  Diesel generator:     {diesel_intensity:.0f} gCO2/kWh (10%)")

    print(f"\nWeighted Average Intensity: {result['weighted_average']:.1f} gCO2/kWh")

    # Calculate savings vs pure grid
    savings = ((grid_intensity - result['weighted_average']) / grid_intensity) * 100
    print(f"Reduction vs. Pure Grid Power: {savings:.1f}%")

    print()


def demo_recommendations():
    """Demo 7: Intelligent recommendations for cleaner energy."""
    print("=" * 80)
    print("DEMO 7: Recommendations for Cleaner Energy")
    print("=" * 80)

    agent = GridFactorAgentAI(enable_recommendations=True)

    # Test different scenarios
    scenarios = [
        {
            "country": "IN",
            "intensity": 710.0,
            "renewable_share": 0.23,
            "description": "India (Coal-Heavy Grid)",
        },
        {
            "country": "US",
            "intensity": 385.0,
            "renewable_share": 0.21,
            "description": "United States (Mixed Grid)",
        },
        {
            "country": "BR",
            "intensity": 120.0,
            "renewable_share": 0.83,
            "description": "Brazil (Hydro-Heavy Grid)",
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['description']}")
        print(f"Current Intensity: {scenario['intensity']:.0f} gCO2/kWh")
        print(f"Renewable Share: {scenario['renewable_share']*100:.0f}%")

        result = agent._generate_recommendations_impl(
            country=scenario["country"],
            current_intensity=scenario["intensity"],
            renewable_share=scenario["renewable_share"],
        )

        print(f"\nTop Recommendations:")

        for i, rec in enumerate(result["recommendations"][:3], 1):
            print(f"\n  {i}. [{rec['priority'].upper()}] {rec['action']}")
            print(f"     Impact: {rec['impact']}")
            print(f"     Potential Reduction: {rec['potential_reduction_gco2_kwh']:.0f} gCO2/kWh")
            print(f"     Payback: {rec['estimated_payback']}")

        print("\n" + "-" * 80)

    print()


def demo_available_data():
    """Demo 8: Show available countries and fuel types."""
    print("=" * 80)
    print("DEMO 8: Available Data")
    print("=" * 80)

    agent = GridFactorAgentAI()

    # Get available countries
    countries = agent.get_available_countries()
    print(f"\nAvailable Countries ({len(countries)}):")
    print(f"  {', '.join(sorted(countries))}")

    # Get fuel types for sample countries
    sample_countries = ["US", "IN", "EU"]

    for country in sample_countries:
        fuel_types = agent.get_available_fuel_types(country)
        print(f"\n{country} Fuel Types ({len(fuel_types)}):")
        print(f"  {', '.join(sorted(fuel_types))}")

    print()


def demo_performance_metrics():
    """Demo 9: Performance tracking."""
    print("=" * 80)
    print("DEMO 9: Performance Metrics")
    print("=" * 80)

    agent = GridFactorAgentAI()

    # Make several tool calls
    lookup_configs = [
        ("US", "electricity", "kWh"),
        ("IN", "electricity", "kWh"),
        ("EU", "natural_gas", "m3"),
        ("CN", "electricity", "MWh"),
    ]

    for country, fuel_type, unit in lookup_configs:
        agent._lookup_grid_intensity_impl(
            country=country,
            fuel_type=fuel_type,
            unit=unit,
        )

    summary = agent.get_performance_summary()

    print(f"\nPerformance Summary:")
    print(f"  AI calls: {summary['ai_metrics']['ai_call_count']}")
    print(f"  Tool calls: {summary['ai_metrics']['tool_call_count']}")
    print(f"  Total cost: ${summary['ai_metrics']['total_cost_usd']:.4f}")
    print(f"  Avg cost per lookup: ${summary['ai_metrics']['avg_cost_per_lookup']:.4f}")

    print(f"\nBase Agent:")
    print(f"  Agent ID: {summary['base_agent_metrics']['agent_id']}")
    print(f"  Name: {summary['base_agent_metrics']['name']}")
    print(f"  Version: {summary['base_agent_metrics']['version']}")

    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("AI-Powered GridFactorAgent Demonstration")
    print("=" * 80)
    print()

    # Note about demo mode
    print("NOTE: Running in demo mode (no API keys required)")
    print("      For production use with real AI, set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    print()

    try:
        demo_basic_lookup()
        demo_country_comparison()
        demo_determinism()
        demo_backward_compatibility()
        demo_hourly_interpolation()
        demo_weighted_average()
        demo_recommendations()
        demo_available_data()
        demo_performance_metrics()

        print("=" * 80)
        print("All demos completed successfully! [PASS]")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\nError running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
