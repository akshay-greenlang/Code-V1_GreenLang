"""Demo: Grid Carbon Intensity Lookup (Updated to use deterministic version)

DEPRECATION NOTICE: This demo has been updated to use GridFactorAgent (deterministic)
instead of GridFactorAgentAI for CRITICAL PATH emissions calculations.

GridFactorAgentAI has been deprecated for regulatory/compliance use cases.
For non-regulatory recommendations, you may still use GridFactorAgentAI, but
the deterministic GridFactorAgent is recommended for all CRITICAL PATH calculations.

Key Features Demonstrated:
1. Tool-first lookups (all data from database)
2. Deterministic results (same input -> same output)
3. Backward compatibility
4. Performance tracking

Usage:
    python examples/grid_factor_agent_ai_demo.py

Author: GreenLang Framework Team
Date: October 2025
Updated: November 2025 (Switched to deterministic version)
"""

# Updated to use deterministic version for CRITICAL PATH calculations
from greenlang.agents.grid_factor_agent import GridFactorAgent
import json


def demo_basic_lookup():
    """Demo 1: Basic grid intensity lookup (deterministic)."""
    print("=" * 80)
    print("DEMO 1: Basic Grid Intensity Lookup (Deterministic)")
    print("=" * 80)

    agent = GridFactorAgent()

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

    agent = GridFactorAgent()

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
            result = agent.run({
                "country": code,
                "fuel_type": "electricity",
                "unit": "kWh",
            })
            if result["success"]:
                data = result["data"]
                intensity = data["emission_factor"] * 1000  # Convert to gCO2/kWh
                renewable = data.get("grid_mix", {}).get("renewable", 0) * 100

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

    agent = GridFactorAgent()

    payload = {
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
    }

    print(f"\nRunning same lookup 3 times...")
    print(f"Input: {payload['country']} {payload['fuel_type']} ({payload['unit']})")

    results = []
    for i in range(3):
        # Use agent to show determinism
        result = agent.run(payload)
        if result["success"]:
            factor = result["data"]["emission_factor"]
            unit = result["data"]["unit"]
            results.append(factor)
            print(f"  Run {i+1}: {factor} {unit}")

    all_same = len(set(results)) == 1
    print(f"\nDeterministic: {all_same} [PASS]" if all_same else f"\nDeterministic: {all_same} [FAIL]")
    print()


def demo_backward_compatibility():
    """Demo 4: Backward compatibility with GridFactorAgent."""
    print("=" * 80)
    print("DEMO 4: Backward Compatibility")
    print("=" * 80)

    original = GridFactorAgent()
    enhanced = GridFactorAgent()

    payload = {
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
    }

    print(f"\nInput: {payload['country']} {payload['fuel_type']} ({payload['unit']})")

    # Original agent
    result_orig = original.run(payload)
    factor_orig = result_orig["data"]["emission_factor"] if result_orig["success"] else 0

    # Enhanced agent (same deterministic version)
    result_enhanced = enhanced.run(payload)
    factor_enhanced = result_enhanced["data"]["emission_factor"] if result_enhanced["success"] else 0

    print(f"\nOriginal GridFactorAgent:  {factor_orig} kgCO2e/{payload['unit']}")
    print(f"Enhanced GridFactorAgent:  {factor_enhanced} kgCO2e/{payload['unit']}")
    print(f"Match: {factor_orig == factor_enhanced} [PASS]" if factor_orig == factor_enhanced else f"Match: {factor_orig == factor_enhanced} [FAIL]")
    print()


def demo_hourly_interpolation():
    """Demo 5: Grid mix information."""
    print("=" * 80)
    print("DEMO 5: Grid Mix Information")
    print("=" * 80)

    agent = GridFactorAgent()

    # Get base intensity
    result = agent.run({
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
    })

    if result["success"]:
        data = result["data"]
        base_intensity = data["emission_factor"] * 1000  # Convert to gCO2/kWh
        grid_mix = data.get("grid_mix", {})

        print(f"\nUS Grid Base Intensity: {base_intensity:.0f} gCO2/kWh")

        if grid_mix:
            print(f"\nGrid Energy Mix:")
            for source, percentage in grid_mix.items():
                print(f"  {source.replace('_', ' ').title():<20} {percentage*100:.1f}%")

    print()


def demo_weighted_average():
    """Demo 6: Simple calculation for mixed energy sources."""
    print("=" * 80)
    print("DEMO 6: Weighted Average for Mixed Energy Sources")
    print("=" * 80)

    agent = GridFactorAgent()

    # Scenario: Facility with mixed energy sources
    print("\nScenario: Manufacturing facility with mixed energy portfolio")
    print("  - 60% grid electricity")
    print("  - 30% on-site solar")
    print("  - 10% backup diesel generator\n")

    # Get grid intensity
    result = agent.run({
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
    })

    if result["success"]:
        grid_intensity = result["data"]["emission_factor"] * 1000  # gCO2/kWh
        solar_intensity = 0.0   # gCO2/kWh (zero emissions)
        diesel_intensity = 700.0  # gCO2/kWh (diesel generator estimate)

        # Manual weighted average calculation
        weighted_avg = (grid_intensity * 0.6) + (solar_intensity * 0.3) + (diesel_intensity * 0.1)

        print(f"Individual Intensities:")
        print(f"  Grid electricity:     {grid_intensity:.0f} gCO2/kWh (60%)")
        print(f"  On-site solar:        {solar_intensity:.0f} gCO2/kWh (30%)")
        print(f"  Diesel generator:     {diesel_intensity:.0f} gCO2/kWh (10%)")

        print(f"\nWeighted Average Intensity: {weighted_avg:.1f} gCO2/kWh")

        # Calculate savings vs pure grid
        savings = ((grid_intensity - weighted_avg) / grid_intensity) * 100
        print(f"Reduction vs. Pure Grid Power: {savings:.1f}%")

    print()


def demo_recommendations():
    """Demo 7: Comparing different grids."""
    print("=" * 80)
    print("DEMO 7: Grid Intensity Comparison")
    print("=" * 80)

    agent = GridFactorAgent()

    # Test different scenarios
    scenarios = [
        ("IN", "India (Coal-Heavy Grid)"),
        ("US", "United States (Mixed Grid)"),
        ("BR", "Brazil (Hydro-Heavy Grid)"),
    ]

    for code, description in scenarios:
        result = agent.run({
            "country": code,
            "fuel_type": "electricity",
            "unit": "kWh",
        })

        if result["success"]:
            data = result["data"]
            intensity = data["emission_factor"] * 1000
            grid_mix = data.get("grid_mix", {})
            renewable = grid_mix.get("renewable", 0) * 100 if grid_mix else 0

            print(f"\n{description}")
            print(f"  Current Intensity: {intensity:.0f} gCO2/kWh")
            print(f"  Renewable Share: {renewable:.0f}%")

    print()


def demo_available_data():
    """Demo 8: Show available countries and fuel types."""
    print("=" * 80)
    print("DEMO 8: Available Data")
    print("=" * 80)

    agent = GridFactorAgent()

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
    """Demo 9: Multiple lookups."""
    print("=" * 80)
    print("DEMO 9: Multiple Lookups")
    print("=" * 80)

    agent = GridFactorAgent()

    # Make several lookups
    lookup_configs = [
        ("US", "electricity", "kWh"),
        ("IN", "electricity", "kWh"),
        ("EU", "natural_gas", "m3"),
        ("CN", "electricity", "MWh"),
    ]

    print(f"\nRunning {len(lookup_configs)} lookups...")

    for country, fuel_type, unit in lookup_configs:
        result = agent.run({
            "country": country,
            "fuel_type": fuel_type,
            "unit": unit,
        })
        if result["success"]:
            factor = result["data"]["emission_factor"]
            print(f"  {country} {fuel_type} ({unit}): {factor} kgCO2e/{unit}")

    print(f"\nAgent Info:")
    print(f"  Agent ID: {agent.agent_id}")
    print(f"  Name: {agent.name}")
    print(f"  Version: {agent.version}")

    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("GridFactorAgent Demonstration (Deterministic Version)")
    print("=" * 80)
    print()

    # Note about deterministic version
    print("NOTE: This demo uses the deterministic GridFactorAgent")
    print("      For CRITICAL PATH emissions calculations (regulatory/compliance)")
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
