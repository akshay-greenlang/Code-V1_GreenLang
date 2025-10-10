"""Demo: AI-Powered Carbon Footprint Aggregation

This script demonstrates the CarbonAgentAI which enhances the original CarbonAgent
with AI orchestration while preserving all deterministic calculations.

Key Features Demonstrated:
1. Tool-first numerics (all calculations via tools)
2. Natural language summaries and insights
3. Intelligent recommendations based on breakdown
4. Deterministic results (same input -> same output)
5. Carbon intensity metrics (per sqft, per person)
6. Backward compatibility with CarbonAgent

Usage:
    python examples/carbon_agent_ai_demo.py

Author: GreenLang Framework Team
Date: October 2025
"""

from greenlang.agents.carbon_agent_ai import CarbonAgentAI
from greenlang.agents.carbon_agent import CarbonAgent
import json


def demo_basic_aggregation():
    """Demo 1: Basic carbon footprint aggregation with AI insights."""
    print("=" * 80)
    print("DEMO 1: Basic Aggregation with AI Insights")
    print("=" * 80)

    agent = CarbonAgentAI(budget_usd=0.50, enable_ai_summary=True)

    # Sample emissions data from multiple sources
    payload = {
        "emissions": [
            {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 8500},
            {"fuel_type": "diesel", "co2e_emissions_kg": 3200},
        ]
    }

    print(f"\nInput: {len(payload['emissions'])} emission sources")
    for emission in payload["emissions"]:
        print(f"  - {emission['fuel_type']}: {emission['co2e_emissions_kg']:,.0f} kg CO2e")

    print("\nAggregating emissions...")

    result = agent.execute(payload)

    if result.success:
        data = result.data
        print(f"\nResults:")
        print(f"  Total Emissions: {data['total_co2e_kg']:,.2f} kg CO2e ({data['total_co2e_tons']:.3f} metric tons)")

        print(f"\nBreakdown by Source:")
        for item in data["emissions_breakdown"]:
            print(f"  - {item['source']}: {item['co2e_kg']:,.2f} kg ({item['percentage']:.2f}%)")

        if "ai_summary" in data:
            print(f"\nAI Summary:")
            print(f"  {data['ai_summary']}")

        print(f"\nMetadata:")
        print(f"  Agent: {result.metadata.get('agent', 'N/A')}")
        print(f"  Sources analyzed: {result.metadata.get('num_sources', 0)}")
        print(f"  Tool calls: {result.metadata.get('tool_calls', 0)}")
        print(f"  Calculation time: {result.metadata.get('calculation_time_ms', 0):.2f} ms")
    else:
        print(f"\nError: {result.error}")

    print()


def demo_with_building_metadata():
    """Demo 2: Aggregation with building area and occupancy."""
    print("=" * 80)
    print("DEMO 2: Carbon Intensity Metrics")
    print("=" * 80)

    agent = CarbonAgentAI()

    payload = {
        "emissions": [
            {"fuel_type": "electricity", "co2e_emissions_kg": 25000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 15000},
        ],
        "building_area": 50000,  # Square feet
        "occupancy": 200,  # People
    }

    print(f"\nBuilding Profile:")
    print(f"  Area: {payload['building_area']:,} sqft")
    print(f"  Occupancy: {payload['occupancy']} people")
    print(f"  Emission sources: {len(payload['emissions'])}")

    result = agent.execute(payload)

    if result.success:
        data = result.data
        print(f"\nTotal Emissions: {data['total_co2e_tons']:.3f} metric tons CO2e")

        if "carbon_intensity" in data and data["carbon_intensity"]:
            intensity = data["carbon_intensity"]
            print(f"\nCarbon Intensity:")
            if "per_sqft" in intensity:
                print(f"  Per square foot: {intensity['per_sqft']:.4f} kg CO2e/sqft")
            if "per_person" in intensity:
                print(f"  Per person: {intensity['per_person']:.2f} kg CO2e/person")

        print(f"\nTraditional Summary:")
        print(f"{data['summary']}")

    print()


def demo_intelligent_recommendations():
    """Demo 3: AI-powered reduction recommendations."""
    print("=" * 80)
    print("DEMO 3: Intelligent Recommendations")
    print("=" * 80)

    agent = CarbonAgentAI(enable_recommendations=True)

    # Electricity-heavy scenario
    payload = {
        "emissions": [
            {"fuel_type": "electricity", "co2e_emissions_kg": 50000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 12000},
            {"fuel_type": "diesel", "co2e_emissions_kg": 3000},
        ]
    }

    print(f"\nScenario: High electricity consumption building")
    for emission in payload["emissions"]:
        print(f"  {emission['fuel_type']}: {emission['co2e_emissions_kg']:,.0f} kg CO2e")

    result = agent.execute(payload)

    if result.success:
        data = result.data

        print(f"\nTotal Footprint: {data['total_co2e_tons']:.2f} metric tons CO2e")

        if "recommendations" in data and data["recommendations"]:
            print(f"\nReduction Recommendations:")
            for i, rec in enumerate(data["recommendations"], 1):
                print(f"\n  {i}. [{rec['priority'].upper()}] {rec['source']}")
                print(f"     Impact: {rec['impact']}")
                print(f"     Action: {rec['action']}")
                print(f"     Potential reduction: {rec['potential_reduction']}")
                if "estimated_payback" in rec:
                    print(f"     Est. payback: {rec['estimated_payback']}")

    print()


def demo_determinism():
    """Demo 4: Deterministic calculations."""
    print("=" * 80)
    print("DEMO 4: Determinism Test")
    print("=" * 80)

    agent = CarbonAgentAI()

    payload = {
        "emissions": [
            {"fuel_type": "electricity", "co2e_emissions_kg": 10000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 5000},
        ]
    }

    print(f"\nRunning same aggregation 3 times...")
    print(f"Input: {len(payload['emissions'])} sources (10,000 kg + 5,000 kg)")

    results = []
    for i in range(3):
        # Use tool directly to show determinism
        result = agent._aggregate_emissions_impl(payload["emissions"])
        results.append(result["total_kg"])
        print(f"  Run {i+1}: {result['total_kg']:,.2f} kg CO2e ({result['total_tons']:.3f} tons)")

    all_same = len(set(results)) == 1
    print(f"\nDeterministic: {all_same}")
    if all_same:
        print("  [PASS] All runs produced identical results")
    else:
        print("  [FAIL] Results varied across runs")

    print()


def demo_backward_compatibility():
    """Demo 5: Backward compatibility with CarbonAgent."""
    print("=" * 80)
    print("DEMO 5: Backward Compatibility")
    print("=" * 80)

    original = CarbonAgent()
    ai_enhanced = CarbonAgentAI()

    payload = {
        "emissions": [
            {"fuel_type": "electricity", "co2e_emissions_kg": 8000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 4000},
        ]
    }

    print(f"\nInput: 2 emission sources (8,000 kg + 4,000 kg)")

    # Original agent
    result_orig = original.execute(payload)
    total_orig = result_orig.data["total_co2e_kg"] if result_orig.success else 0

    # AI agent tool (bypassing AI for exact comparison)
    result_ai = ai_enhanced._aggregate_emissions_impl(payload["emissions"])
    total_ai = result_ai["total_kg"]

    print(f"\nOriginal CarbonAgent:    {total_orig:,.2f} kg CO2e")
    print(f"AI CarbonAgent Tool:     {total_ai:,.2f} kg CO2e")

    match = total_orig == total_ai
    print(f"Match: {match}")
    if match:
        print("  [PASS] Both agents produce identical numeric results")
    else:
        print("  [FAIL] Results differ")

    print()


def demo_breakdown_analysis():
    """Demo 6: Detailed breakdown analysis."""
    print("=" * 80)
    print("DEMO 6: Breakdown Analysis")
    print("=" * 80)

    agent = CarbonAgentAI()

    payload = {
        "emissions": [
            {"fuel_type": "electricity", "co2e_emissions_kg": 20000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 15000},
            {"fuel_type": "diesel", "co2e_emissions_kg": 8000},
            {"fuel_type": "propane", "co2e_emissions_kg": 2000},
        ]
    }

    print(f"\nAnalyzing {len(payload['emissions'])} emission sources...")

    # Use breakdown tool directly
    total_result = agent._aggregate_emissions_impl(payload["emissions"])
    total_kg = total_result["total_kg"]

    breakdown_result = agent._calculate_breakdown_impl(payload["emissions"], total_kg)

    print(f"\nTotal: {total_kg:,.2f} kg CO2e")
    print(f"\nDetailed Breakdown (sorted by impact):")

    for i, item in enumerate(breakdown_result["breakdown"], 1):
        bar_length = int(item["percentage"] / 2)  # Scale to 50 chars max
        bar = "#" * bar_length  # Use # instead of Unicode block for Windows compatibility
        print(f"  {i}. {item['source']:15} {item['co2e_kg']:>10,.0f} kg ({item['percentage']:>5.2f}%) {bar}")

    print()


def demo_empty_emissions():
    """Demo 7: Handling empty emissions list."""
    print("=" * 80)
    print("DEMO 7: Edge Case - Empty Emissions")
    print("=" * 80)

    agent = CarbonAgentAI()

    payload = {"emissions": []}

    print(f"\nInput: Empty emissions list")

    result = agent.execute(payload)

    if result.success:
        data = result.data
        print(f"\nResults:")
        print(f"  Total: {data['total_co2e_kg']} kg CO2e")
        print(f"  Breakdown: {len(data['emissions_breakdown'])} items")
        print(f"  Summary: {data['summary']}")
        print("\n  [PASS] Empty emissions handled gracefully")
    else:
        print(f"\nError: {result.error}")

    print()


def demo_realistic_building():
    """Demo 8: Realistic office building scenario."""
    print("=" * 80)
    print("DEMO 8: Realistic Office Building Scenario")
    print("=" * 80)

    agent = CarbonAgentAI(
        budget_usd=0.50,
        enable_ai_summary=True,
        enable_recommendations=True,
    )

    # Typical mid-size office building annual emissions
    payload = {
        "emissions": [
            {"fuel_type": "electricity", "co2e_emissions_kg": 125000},  # ~250,000 kWh
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 68000},   # Heating
            {"fuel_type": "diesel", "co2e_emissions_kg": 12000},        # Backup generators
        ],
        "building_area": 100000,  # 100,000 sqft
        "occupancy": 500,         # 500 employees
    }

    print(f"\nBuilding Profile:")
    print(f"  Type: Commercial office building")
    print(f"  Area: {payload['building_area']:,} sqft")
    print(f"  Occupancy: {payload['occupancy']} employees")
    print(f"  Analysis period: Annual")

    print(f"\nEmission Sources:")
    for emission in payload["emissions"]:
        print(f"  - {emission['fuel_type']:15} {emission['co2e_emissions_kg']:>10,.0f} kg CO2e")

    result = agent.execute(payload)

    if result.success:
        data = result.data

        print(f"\n{'='*60}")
        print(f"CARBON FOOTPRINT ANALYSIS")
        print(f"{'='*60}")

        print(f"\nTotal Annual Emissions: {data['total_co2e_tons']:.2f} metric tons CO2e")
        print(f"                        {data['total_co2e_kg']:,.0f} kg CO2e")

        print(f"\nEmissions Breakdown:")
        for item in data["emissions_breakdown"]:
            print(f"  {item['source']:15} {item['co2e_tons']:>8.3f} tons ({item['percentage']:>5.2f}%)")

        if "carbon_intensity" in data:
            intensity = data["carbon_intensity"]
            print(f"\nCarbon Intensity Benchmarks:")
            if "per_sqft" in intensity:
                print(f"  Per sqft:  {intensity['per_sqft']:.4f} kg CO2e/sqft/year")
                # Industry benchmark comparison
                if intensity['per_sqft'] < 5.0:
                    benchmark = "Excellent (below average)"
                elif intensity['per_sqft'] < 8.0:
                    benchmark = "Good (average)"
                else:
                    benchmark = "Needs improvement (above average)"
                print(f"             Rating: {benchmark}")

            if "per_person" in intensity:
                print(f"  Per person: {intensity['per_person']:.2f} kg CO2e/person/year")

        if "recommendations" in data and data["recommendations"]:
            print(f"\n{'='*60}")
            print(f"REDUCTION RECOMMENDATIONS")
            print(f"{'='*60}")

            for i, rec in enumerate(data["recommendations"], 1):
                print(f"\n{i}. {rec['source'].upper()} - {rec['priority'].upper()} PRIORITY")
                print(f"   Current Impact: {rec['impact']}")
                print(f"   Action: {rec['action']}")
                print(f"   Potential Reduction: {rec['potential_reduction']}")
                if "estimated_payback" in rec:
                    print(f"   Est. Payback: {rec['estimated_payback']}")

        print(f"\n{'='*60}")
        print(f"PERFORMANCE METRICS")
        print(f"{'='*60}")
        metadata = result.metadata
        print(f"  Provider: {metadata.get('provider', 'N/A')}")
        print(f"  Model: {metadata.get('model', 'N/A')}")
        print(f"  Tool calls: {metadata.get('tool_calls', 0)}")
        print(f"  Calculation time: {metadata.get('calculation_time_ms', 0):.2f} ms")
        print(f"  Cost: ${metadata.get('cost_usd', 0):.4f}")
        print(f"  Deterministic: {metadata.get('deterministic', False)}")

    print()


def demo_performance_tracking():
    """Demo 9: Performance metrics tracking."""
    print("=" * 80)
    print("DEMO 9: Performance Metrics")
    print("=" * 80)

    agent = CarbonAgentAI()

    # Make several tool calls
    scenarios = [
        [{"fuel_type": "electricity", "co2e_emissions_kg": 10000}],
        [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 5000},
            {"fuel_type": "diesel", "co2e_emissions_kg": 2000},
        ],
        [
            {"fuel_type": "electricity", "co2e_emissions_kg": 8000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 4000},
            {"fuel_type": "propane", "co2e_emissions_kg": 1000},
        ],
    ]

    for i, emissions in enumerate(scenarios, 1):
        agent._aggregate_emissions_impl(emissions)

    summary = agent.get_performance_summary()

    print(f"\nPerformance Summary:")
    print(f"  Agent: {summary['agent']}")
    print(f"  AI calls: {summary['ai_metrics']['ai_call_count']}")
    print(f"  Tool calls: {summary['ai_metrics']['tool_call_count']}")
    print(f"  Total cost: ${summary['ai_metrics']['total_cost_usd']:.4f}")
    print(f"  Avg cost per aggregation: ${summary['ai_metrics']['avg_cost_per_aggregation']:.4f}")

    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("AI-Powered CarbonAgent Demonstration")
    print("=" * 80)
    print()

    # Note about demo mode
    print("NOTE: Running in demo mode (no API keys required)")
    print("      For production use with real AI, set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    print()

    try:
        demo_basic_aggregation()
        demo_with_building_metadata()
        demo_intelligent_recommendations()
        demo_determinism()
        demo_backward_compatibility()
        demo_breakdown_analysis()
        demo_empty_emissions()
        demo_realistic_building()
        demo_performance_tracking()

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
