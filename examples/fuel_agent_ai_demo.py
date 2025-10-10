"""Demo: AI-Powered Fuel Emissions Calculator

This script demonstrates the FuelAgentAI which enhances the original FuelAgent
with AI orchestration while preserving all deterministic calculations.

Key Features Demonstrated:
1. Tool-first numerics (all calculations via tools)
2. Natural language explanations
3. Deterministic results (same input -> same output)
4. Backward compatibility with FuelAgent
5. Performance tracking

Usage:
    python examples/fuel_agent_ai_demo.py

Author: GreenLang Framework Team
Date: October 2025
"""

from greenlang.agents.fuel_agent_ai import FuelAgentAI
from greenlang.agents.fuel_agent import FuelAgent
import json


def demo_basic_calculation():
    """Demo 1: Basic calculation with AI explanation."""
    print("=" * 80)
    print("DEMO 1: Basic Calculation with AI")
    print("=" * 80)

    agent = FuelAgentAI(budget_usd=0.50, enable_explanations=True)

    payload = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US",
    }

    print(f"\nInput: {json.dumps(payload, indent=2)}")
    print("\nCalculating...")

    result = agent.run(payload)

    if result["success"]:
        data = result["data"]
        print(f"\nResults:")
        print(f"  Emissions: {data['co2e_emissions_kg']:,.1f} kg CO2e")
        print(f"  Emission Factor: {data['emission_factor']} {data['emission_factor_unit']}")
        print(f"  Scope: {data['scope']}")
        print(f"  Energy Content: {data['energy_content_mmbtu']:.2f} MMBtu")

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


def demo_determinism():
    """Demo 2: Deterministic calculations."""
    print("=" * 80)
    print("DEMO 2: Determinism Test")
    print("=" * 80)

    agent = FuelAgentAI()

    payload = {
        "fuel_type": "diesel",
        "amount": 500,
        "unit": "gallons",
        "country": "US",
    }

    print(f"\nRunning same calculation 3 times...")
    print(f"Input: {payload['amount']} {payload['unit']} of {payload['fuel_type']}")

    results = []
    for i in range(3):
        # Use tool directly to show determinism
        result = agent._calculate_emissions_impl(
            fuel_type=payload["fuel_type"],
            amount=payload["amount"],
            unit=payload["unit"],
            country=payload["country"],
        )
        results.append(result["emissions_kg_co2e"])
        print(f"  Run {i+1}: {result['emissions_kg_co2e']:,.2f} kg CO2e")

    all_same = len(set(results)) == 1
    print(f"\nDeterministic: {all_same} [PASS]" if all_same else f"\nDeterministic: {all_same} [FAIL]")
    print()


def demo_backward_compatibility():
    """Demo 3: Backward compatibility with FuelAgent."""
    print("=" * 80)
    print("DEMO 3: Backward Compatibility")
    print("=" * 80)

    original = FuelAgent()
    ai_enhanced = FuelAgentAI()

    payload = {
        "fuel_type": "natural_gas",
        "amount": 100,
        "unit": "therms",
        "country": "US",
    }

    print(f"\nInput: {payload['amount']} {payload['unit']} of {payload['fuel_type']}")

    # Original agent
    result_orig = original.run(payload)
    emissions_orig = result_orig["data"]["co2e_emissions_kg"] if result_orig["success"] else 0

    # AI agent tool (bypassing AI for exact comparison)
    result_ai = ai_enhanced._calculate_emissions_impl(
        fuel_type=payload["fuel_type"],
        amount=payload["amount"],
        unit=payload["unit"],
        country=payload["country"],
    )
    emissions_ai = result_ai["emissions_kg_co2e"]

    print(f"\nOriginal FuelAgent:  {emissions_orig:,.2f} kg CO2e")
    print(f"AI FuelAgent Tool:   {emissions_ai:,.2f} kg CO2e")
    print(f"Match: {emissions_orig == emissions_ai} [PASS]" if emissions_orig == emissions_ai else f"Match: {emissions_orig == emissions_ai} [FAIL]")
    print()


def demo_renewable_offset():
    """Demo 4: Renewable energy offset."""
    print("=" * 80)
    print("DEMO 4: Renewable Energy Offset")
    print("=" * 80)

    agent = FuelAgentAI()

    base_payload = {
        "fuel_type": "electricity",
        "amount": 10000,
        "unit": "kWh",
        "country": "US",
    }

    print(f"\nInput: {base_payload['amount']} {base_payload['unit']} of {base_payload['fuel_type']}")

    # No renewable
    result_0 = agent._calculate_emissions_impl(**base_payload, renewable_percentage=0)
    print(f"\nNo renewable offset:    {result_0['emissions_kg_co2e']:,.2f} kg CO2e")

    # 25% renewable
    result_25 = agent._calculate_emissions_impl(**base_payload, renewable_percentage=25)
    print(f"25% renewable offset:   {result_25['emissions_kg_co2e']:,.2f} kg CO2e")

    # 50% renewable
    result_50 = agent._calculate_emissions_impl(**base_payload, renewable_percentage=50)
    print(f"50% renewable offset:   {result_50['emissions_kg_co2e']:,.2f} kg CO2e")

    # 100% renewable
    result_100 = agent._calculate_emissions_impl(**base_payload, renewable_percentage=100)
    print(f"100% renewable offset:  {result_100['emissions_kg_co2e']:,.2f} kg CO2e")

    print()


def demo_recommendations():
    """Demo 5: Fuel switching recommendations."""
    print("=" * 80)
    print("DEMO 5: Recommendations")
    print("=" * 80)

    agent = FuelAgentAI(enable_recommendations=True)

    # High-emission fuel
    result = agent._generate_recommendations_impl(
        fuel_type="coal",
        emissions_kg=50000,
        country="US",
    )

    print(f"\nFuel type: coal")
    print(f"Emissions: 50,000 kg CO2e")
    print(f"\nRecommendations ({result['count']} total):")

    for i, rec in enumerate(result["recommendations"][:3], 1):
        print(f"\n  {i}. [{rec['priority'].upper()}] {rec['action']}")
        print(f"     Impact: {rec['impact']}")
        print(f"     Feasibility: {rec['feasibility']}")

    print()


def demo_performance_metrics():
    """Demo 6: Performance tracking."""
    print("=" * 80)
    print("DEMO 6: Performance Metrics")
    print("=" * 80)

    agent = FuelAgentAI()

    # Make several tool calls
    fuel_configs = [
        ("natural_gas", 100, "therms"),
        ("electricity", 1000, "kWh"),
        ("diesel", 50, "gallons"),
    ]

    for fuel_type, amount, unit in fuel_configs:
        agent._calculate_emissions_impl(
            fuel_type=fuel_type,
            amount=amount,
            unit=unit,
            country="US",
        )

    summary = agent.get_performance_summary()

    print(f"\nPerformance Summary:")
    print(f"  AI calls: {summary['ai_metrics']['ai_call_count']}")
    print(f"  Tool calls: {summary['ai_metrics']['tool_call_count']}")
    print(f"  Total cost: ${summary['ai_metrics']['total_cost_usd']:.4f}")
    print(f"  Avg cost per calc: ${summary['ai_metrics']['avg_cost_per_calculation']:.4f}")

    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("AI-Powered FuelAgent Demonstration")
    print("=" * 80)
    print()

    # Note about demo mode
    print("NOTE: Running in demo mode (no API keys required)")
    print("      For production use with real AI, set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    print()

    try:
        demo_basic_calculation()
        demo_determinism()
        demo_backward_compatibility()
        demo_renewable_offset()
        demo_recommendations()
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
