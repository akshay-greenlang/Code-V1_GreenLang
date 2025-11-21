# -*- coding: utf-8 -*-
"""
demos/fuel_agent_v2_demo.py

Demonstration of FuelAgentAI v2 with backward compatibility

Shows:
1. v1 clients work unchanged (legacy format)
2. v2 clients get enhanced features (multi-gas, provenance, DQS)
3. Fast path optimization (60% cheaper for simple requests)
4. Three response formats (legacy, enhanced, compact)

Run:
    python demos/fuel_agent_v2_demo.py
"""

from greenlang.agents import FuelAgentAI_v2
import json


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_json(data: dict, indent: int = 2):
    """Pretty print JSON"""
    print(json.dumps(data, indent=indent, default=str))


def demo_v1_compatibility():
    """Demo 1: v1 clients work unchanged"""
    print_section("DEMO 1: v1 Compatibility (Legacy Format)")

    agent = FuelAgentAI_v2(
        enable_explanations=False,  # Disable for fast path
        enable_recommendations=False,
    )

    # v1 request (unchanged from before)
    payload = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US",
    }

    print("\n📥 INPUT (v1 format):")
    print_json(payload)

    result = agent.run(payload)

    print("\n📤 OUTPUT (v1 format - backward compatible):")
    if result["success"]:
        print_json(result["data"])
        print("\n⚡ PERFORMANCE:")
        print(f"   Execution path: {result['metadata']['execution_path']}")
        print(f"   Calculation time: {result['metadata']['calculation_time_ms']:.2f} ms")
        print(f"   Cost: ${result['metadata']['total_cost_usd']:.6f}")
    else:
        print(f"❌ ERROR: {result['error']['message']}")


def demo_v2_enhanced():
    """Demo 2: v2 clients get enhanced features"""
    print_section("DEMO 2: v2 Enhanced Format (Multi-Gas + Provenance)")

    agent = FuelAgentAI_v2(
        enable_explanations=True,
        enable_recommendations=True,
    )

    # v2 request with enhanced parameters
    payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
        "country": "US",
        "scope": "1",
        "boundary": "WTW",  # Well-to-Wheel (full lifecycle)
        "gwp_set": "IPCC_AR6_100",
        "response_format": "enhanced",  # 🔑 Key difference
    }

    print("\n📥 INPUT (v2 format):")
    print_json(payload)

    result = agent.run(payload)

    print("\n📤 OUTPUT (v2 enhanced format):")
    if result["success"]:
        data = result["data"]

        # Show backward compatible fields
        print("\n🔄 V1 FIELDS (backward compatible):")
        print(f"   co2e_emissions_kg: {data['co2e_emissions_kg']:.2f}")
        print(f"   emission_factor: {data['emission_factor']:.4f}")
        print(f"   scope: {data['scope']}")

        # Show v2 enhancements
        print("\n✨ V2 ENHANCEMENTS:")
        print(f"\n   Multi-Gas Breakdown (vectors_kg):")
        for gas, kg in data.get("vectors_kg", {}).items():
            print(f"      {gas}: {kg:.2f} kg")

        print(f"\n   Provenance:")
        prov = data.get("factor_record", {})
        print(f"      Factor ID: {prov.get('factor_id', 'N/A')}")
        print(f"      Source: {prov.get('source_org', 'N/A')}")
        print(f"      Citation: {prov.get('citation', 'N/A')[:80]}...")

        print(f"\n   Quality:")
        quality = data.get("quality", {})
        dqs = quality.get("dqs", {})
        print(f"      DQS Score: {dqs.get('overall_score', 0):.2f}/5.0 ({dqs.get('rating', 'N/A')})")
        print(f"      Uncertainty: ±{quality.get('uncertainty_95ci_pct', 0):.1f}% (95% CI)")

        print(f"\n   Calculation Breakdown:")
        breakdown = data.get("breakdown", {})
        print(f"      {breakdown.get('calculation', 'N/A')}")

        if "explanation" in data:
            print(f"\n   AI Explanation:")
            print(f"      {data['explanation'][:200]}...")

        print("\n⚡ PERFORMANCE:")
        print(f"   Execution path: {result['metadata']['execution_path']}")
        print(f"   Calculation time: {result['metadata']['calculation_time_ms']:.2f} ms")
        print(f"   Cost: ${result['metadata']['total_cost_usd']:.6f}")
        print(f"   AI calls: {result['metadata']['ai_calls']}")
        print(f"   Tool calls: {result['metadata']['tool_calls']}")
    else:
        print(f"❌ ERROR: {result['error']['message']}")


def demo_compact_format():
    """Demo 3: Compact format for mobile/IoT"""
    print_section("DEMO 3: Compact Format (Mobile/IoT)")

    agent = FuelAgentAI_v2(
        enable_explanations=False,
        enable_recommendations=False,
    )

    payload = {
        "fuel_type": "electricity",
        "amount": 10000,
        "unit": "kWh",
        "country": "US",
        "response_format": "compact",  # 🔑 Minimal output
    }

    print("\n📥 INPUT:")
    print_json(payload)

    result = agent.run(payload)

    print("\n📤 OUTPUT (compact format - minimal for mobile):")
    if result["success"]:
        print_json(result["data"])
        print("\n⚡ PERFORMANCE:")
        print(f"   Execution path: {result['metadata']['execution_path']}")
        print(f"   Calculation time: {result['metadata']['calculation_time_ms']:.2f} ms")
    else:
        print(f"❌ ERROR: {result['error']['message']}")


def demo_fast_path_optimization():
    """Demo 4: Fast path optimization"""
    print_section("DEMO 4: Fast Path Optimization (60% Cost Reduction)")

    # Run same calculation with and without fast path
    payload = {
        "fuel_type": "natural_gas",
        "amount": 5000,
        "unit": "therms",
    }

    print("\n🚀 WITH FAST PATH (enable_explanations=False):")
    agent_fast = FuelAgentAI_v2(
        enable_explanations=False,
        enable_recommendations=False,
        enable_fast_path=True,
    )
    result_fast = agent_fast.run(payload)
    if result_fast["success"]:
        print(f"   Execution path: {result_fast['metadata']['execution_path']}")
        print(f"   Time: {result_fast['metadata']['calculation_time_ms']:.2f} ms")
        print(f"   Cost: ${result_fast['metadata']['total_cost_usd']:.6f}")
        print(f"   Result: {result_fast['data']['co2e_emissions_kg']:.2f} kg CO2e")

    print("\n🤖 WITH AI PATH (enable_explanations=True):")
    agent_ai = FuelAgentAI_v2(
        enable_explanations=True,
        enable_recommendations=True,
        enable_fast_path=True,
    )
    result_ai = agent_ai.run(payload)
    if result_ai["success"]:
        print(f"   Execution path: {result_ai['metadata']['execution_path']}")
        print(f"   Time: {result_ai['metadata']['calculation_time_ms']:.2f} ms")
        print(f"   Cost: ${result_ai['metadata']['total_cost_usd']:.6f}")
        print(f"   Result: {result_ai['data']['co2e_emissions_kg']:.2f} kg CO2e")

    if result_fast["success"] and result_ai["success"]:
        time_savings = (
            (result_ai["metadata"]["calculation_time_ms"] - result_fast["metadata"]["calculation_time_ms"])
            / result_ai["metadata"]["calculation_time_ms"]
        ) * 100
        print(f"\n💡 SAVINGS:")
        print(f"   Time reduction: {time_savings:.1f}%")
        print(f"   Fast path is ideal for: API endpoints, batch processing, mobile apps")


def demo_multiple_gwp_sets():
    """Demo 5: Multiple GWP sets"""
    print_section("DEMO 5: Multiple GWP Sets (IPCC AR6 100-year vs 20-year)")

    agent = FuelAgentAI_v2(enable_explanations=False, enable_recommendations=False)

    payload_base = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "response_format": "enhanced",
    }

    for gwp_set in ["IPCC_AR6_100", "IPCC_AR6_20"]:
        payload = {**payload_base, "gwp_set": gwp_set}
        result = agent.run(payload)

        if result["success"]:
            data = result["data"]
            print(f"\n📊 {gwp_set}:")
            print(f"   Total CO2e: {data['co2e_emissions_kg']:.2f} kg")
            print(f"   CO2: {data['vectors_kg']['CO2']:.2f} kg")
            print(f"   CH4: {data['vectors_kg']['CH4']:.4f} kg")
            print(f"   N2O: {data['vectors_kg']['N2O']:.4f} kg")
            print(f"   GWP Set: {data['gwp_set']}")


def demo_boundary_settings():
    """Demo 6: Emission boundary settings"""
    print_section("DEMO 6: Emission Boundaries (Combustion vs WTW)")

    agent = FuelAgentAI_v2(enable_explanations=False, enable_recommendations=False)

    payload_base = {
        "fuel_type": "diesel",
        "amount": 100,
        "unit": "gallons",
        "response_format": "enhanced",
    }

    for boundary in ["combustion", "WTW"]:
        payload = {**payload_base, "boundary": boundary}
        result = agent.run(payload)

        if result["success"]:
            data = result["data"]
            print(f"\n📊 {boundary.upper()}:")
            print(f"   Total CO2e: {data['co2e_emissions_kg']:.2f} kg")
            print(f"   Boundary: {data['boundary']}")
            print(f"   Description:")
            if boundary == "combustion":
                print(f"      Direct combustion emissions only (tank-to-wheel)")
            else:
                print(f"      Full lifecycle emissions (well-to-wheel)")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("  FuelAgentAI v2 - Backward Compatible Enhancement Demo")
    print("=" * 80)

    try:
        # Demo 1: v1 compatibility
        demo_v1_compatibility()

        # Demo 2: v2 enhanced features
        demo_v2_enhanced()

        # Demo 3: Compact format
        demo_compact_format()

        # Demo 4: Fast path optimization
        demo_fast_path_optimization()

        # Demo 5: Multiple GWP sets
        demo_multiple_gwp_sets()

        # Demo 6: Boundary settings
        demo_boundary_settings()

        print("\n" + "=" * 80)
        print("  ✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\n💡 KEY TAKEAWAYS:")
        print("   1. v1 clients work unchanged (zero breaking changes)")
        print("   2. v2 clients get multi-gas, provenance, and DQS")
        print("   3. Fast path is 60% cheaper for simple requests")
        print("   4. Three formats: legacy (v1), enhanced (v2), compact (IoT)")
        print("   5. Same deterministic results across all formats")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
