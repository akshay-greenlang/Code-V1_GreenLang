# -*- coding: utf-8 -*-
"""RecommendationAgentAI Demo - AI-Powered Building Optimization Recommendations

This demo showcases the AI-powered RecommendationAgent, which generates
intelligent, actionable recommendations for reducing building carbon emissions.

Features Demonstrated:
1. AI-driven energy usage analysis
2. ROI-based recommendation prioritization
3. Natural language explanations
4. Implementation planning
5. Savings estimation (emissions + cost)
6. Tool-first numerics (deterministic calculations)

The demo includes multiple scenarios:
- Old inefficient building
- High-performance modern building
- HVAC-dominated facility
- Electricity-heavy data center

Author: GreenLang Framework Team
Date: October 2025
"""

import asyncio
from typing import Dict, Any
from greenlang.agents.recommendation_agent_ai import RecommendationAgentAI
from greenlang.intelligence import has_any_api_key


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_recommendations(data: Dict[str, Any]) -> None:
    """Print recommendations in formatted output."""
    recommendations = data.get("recommendations", [])

    print(f"Total Recommendations: {len(recommendations)}\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.get('action', 'N/A')}")
        print(f"   Impact: {rec.get('impact', 'N/A')}")
        print(f"   Cost: {rec.get('cost', 'N/A')}")
        print(f"   Payback: {rec.get('payback', 'N/A')}")
        print(f"   Priority: {rec.get('priority', 'N/A')}")

        # ROI data if available
        if "roi_percentage" in rec:
            print(f"   ROI: {rec['roi_percentage']}%")
        if "annual_savings_usd" in rec:
            print(f"   Annual Savings: ${rec['annual_savings_usd']:,.2f}")

        print()


def print_usage_analysis(data: Dict[str, Any]) -> None:
    """Print usage analysis results."""
    analysis = data.get("usage_analysis", {})

    if not analysis:
        return

    print("Usage Analysis:")
    print(f"  Total Emissions: {analysis.get('total_emissions_kg', 0):,.0f} kg CO2e")
    print(f"  Dominant Source: {analysis.get('dominant_source', 'N/A')}")

    # Source percentages
    percentages = analysis.get("source_percentages", {})
    if percentages:
        print("\n  Source Breakdown:")
        for source, pct in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {source}: {pct}%")

    # Issues identified
    issues = analysis.get("issues_identified", [])
    if issues:
        print(f"\n  Issues Identified ({len(issues)}):")
        for issue in issues:
            severity = issue.get("severity", "unknown").upper()
            desc = issue.get("description", "N/A")
            print(f"    [{severity}] {desc}")

    print()


def print_savings(data: Dict[str, Any]) -> None:
    """Print savings estimates."""
    # Emissions savings
    emissions_savings = data.get("potential_savings", {})
    if emissions_savings:
        print("Potential Emissions Savings:")
        print(f"  Minimum: {emissions_savings.get('minimum_kg_co2e', 0):,.0f} kg CO2e")
        print(f"  Maximum: {emissions_savings.get('maximum_kg_co2e', 0):,.0f} kg CO2e")
        print(f"  Range: {emissions_savings.get('percentage_range', 'N/A')}")

    # Cost savings
    cost_savings = data.get("cost_savings", {})
    if cost_savings:
        print("\nPotential Cost Savings:")
        print(f"  Minimum Annual: ${cost_savings.get('minimum_annual_usd', 0):,.2f}")
        print(f"  Maximum Annual: ${cost_savings.get('maximum_annual_usd', 0):,.2f}")

    print()


def print_implementation_roadmap(data: Dict[str, Any]) -> None:
    """Print implementation roadmap."""
    roadmap = data.get("implementation_roadmap", [])

    if not roadmap:
        return

    print("Implementation Roadmap:\n")

    for phase in roadmap:
        phase_name = phase.get("phase", "N/A")
        timeline = phase.get("timeline_months", "N/A")
        print(f"  {phase_name}")
        print(f"  Timeline: {timeline} months")

        actions = phase.get("actions", [])
        if actions:
            print("  Actions:")
            for action in actions[:3]:  # Show top 3
                print(f"    - {action.get('action', 'N/A')}")

        print()


def print_ai_summary(data: Dict[str, Any]) -> None:
    """Print AI-generated summary."""
    ai_summary = data.get("ai_summary", "")

    if ai_summary:
        print("AI Analysis Summary:")
        print(f"{ai_summary}\n")


def demo_old_inefficient_building():
    """Demo: Old inefficient building needing major upgrades."""
    print_section("Demo 1: Old Inefficient Building (20+ years)")

    building_data = {
        "emissions_by_source": {
            "electricity": 35000,
            "natural_gas": 20000,
            "diesel": 5000,
        },
        "building_type": "commercial_office",
        "building_area": 75000,
        "occupancy": 300,
        "building_age": 25,
        "performance_rating": "Poor",
        "load_breakdown": {
            "hvac_load": 0.50,
            "lighting_load": 0.30,
            "plug_load": 0.20,
        },
        "country": "US",
    }

    print("Building Profile:")
    print(f"  Type: {building_data['building_type']}")
    print(f"  Age: {building_data['building_age']} years")
    print(f"  Area: {building_data['building_area']:,} sqft")
    print(f"  Occupancy: {building_data['occupancy']} people")
    print(f"  Performance: {building_data['performance_rating']}")
    print(f"  Total Emissions: {sum(building_data['emissions_by_source'].values()):,} kg CO2e/year")
    print()

    # Create agent
    agent = RecommendationAgentAI(
        budget_usd=0.50,
        enable_ai_summary=True,
        enable_implementation_plans=True,
        max_recommendations=5,
    )

    # Run analysis
    print("Running AI-powered recommendation analysis...\n")
    result = agent.execute(building_data)

    if result.success:
        data = result.data

        # Print results
        print_usage_analysis(data)
        print_recommendations(data)
        print_savings(data)
        print_implementation_roadmap(data)
        print_ai_summary(data)

        # Performance metrics
        print("Performance Metrics:")
        print(f"  Calculation Time: {result.metadata.get('calculation_time_ms', 0):.2f} ms")
        print(f"  AI Calls: {result.metadata.get('ai_calls', 0)}")
        print(f"  Tool Calls: {result.metadata.get('tool_calls', 0)}")
        print(f"  Cost: ${result.metadata.get('cost_usd', 0):.4f}")
        print(f"  Provider: {result.metadata.get('provider', 'N/A')}")
        print(f"  Model: {result.metadata.get('model', 'N/A')}")
    else:
        print(f"Error: {result.error}")


def demo_modern_efficient_building():
    """Demo: Modern efficient building with optimization opportunities."""
    print_section("Demo 2: Modern High-Performance Building")

    building_data = {
        "emissions_by_source": {
            "electricity": 12000,
            "natural_gas": 4000,
        },
        "building_type": "commercial_office",
        "building_area": 50000,
        "occupancy": 200,
        "building_age": 5,
        "performance_rating": "Good",
        "load_breakdown": {
            "hvac_load": 0.35,
            "lighting_load": 0.20,
            "plug_load": 0.45,
        },
        "country": "US",
    }

    print("Building Profile:")
    print(f"  Type: {building_data['building_type']}")
    print(f"  Age: {building_data['building_age']} years (Modern)")
    print(f"  Performance: {building_data['performance_rating']}")
    print(f"  Total Emissions: {sum(building_data['emissions_by_source'].values()):,} kg CO2e/year")
    print()

    # Create agent
    agent = RecommendationAgentAI(max_recommendations=5)

    # Run analysis
    print("Running AI-powered recommendation analysis...\n")
    result = agent.execute(building_data)

    if result.success:
        data = result.data

        print_usage_analysis(data)
        print_recommendations(data)

        # Quick wins
        quick_wins = data.get("quick_wins", [])
        if quick_wins:
            print("Quick Wins (Low Cost, High Impact):")
            for i, rec in enumerate(quick_wins, 1):
                print(f"  {i}. {rec.get('action', 'N/A')}")
            print()

        print_ai_summary(data)
    else:
        print(f"Error: {result.error}")


def demo_hvac_dominated_facility():
    """Demo: Facility with HVAC-dominated energy usage."""
    print_section("Demo 3: HVAC-Dominated Industrial Facility")

    building_data = {
        "emissions_by_source": {
            "electricity": 40000,
            "natural_gas": 30000,
        },
        "building_type": "industrial",
        "building_age": 15,
        "performance_rating": "Below Average",
        "load_breakdown": {
            "hvac_load": 0.65,  # Very high HVAC
            "lighting_load": 0.15,
            "plug_load": 0.20,
        },
        "country": "US",
    }

    print("Building Profile:")
    print(f"  Type: {building_data['building_type']}")
    print(f"  HVAC Load: {building_data['load_breakdown']['hvac_load']*100}% (Very High)")
    print(f"  Total Emissions: {sum(building_data['emissions_by_source'].values()):,} kg CO2e/year")
    print()

    # Create agent with focus on ROI
    agent = RecommendationAgentAI(max_recommendations=5)

    # Run analysis
    print("Running AI-powered recommendation analysis...\n")
    result = agent.execute(building_data)

    if result.success:
        data = result.data

        print_usage_analysis(data)
        print_recommendations(data)

        # High impact recommendations
        high_impact = data.get("high_impact", [])
        if high_impact:
            print("High Impact Recommendations (Top 3):")
            for i, rec in enumerate(high_impact, 1):
                print(f"  {i}. {rec.get('action', 'N/A')}")
                print(f"     Impact: {rec.get('impact', 'N/A')}")
            print()

        print_ai_summary(data)
    else:
        print(f"Error: {result.error}")


def demo_electricity_heavy_datacenter():
    """Demo: Electricity-heavy data center scenario."""
    print_section("Demo 4: Electricity-Heavy Data Center")

    building_data = {
        "emissions_by_source": {
            "electricity": 80000,  # Very high electricity
            "natural_gas": 5000,
        },
        "building_type": "data_center",
        "building_area": 20000,
        "building_age": 10,
        "performance_rating": "Average",
        "load_breakdown": {
            "hvac_load": 0.30,  # Cooling
            "lighting_load": 0.05,
            "plug_load": 0.65,  # IT equipment
        },
        "country": "US",
    }

    print("Building Profile:")
    print(f"  Type: {building_data['building_type']}")
    print(f"  Electricity: {building_data['emissions_by_source']['electricity']:,} kg CO2e/year")
    print(f"  Electricity Share: {building_data['emissions_by_source']['electricity']/sum(building_data['emissions_by_source'].values())*100:.1f}%")
    print()

    # Create agent
    agent = RecommendationAgentAI(max_recommendations=5)

    # Run analysis
    print("Running AI-powered recommendation analysis...\n")
    result = agent.execute(building_data)

    if result.success:
        data = result.data

        print_usage_analysis(data)
        print_recommendations(data)
        print_savings(data)
        print_ai_summary(data)
    else:
        print(f"Error: {result.error}")


def demo_performance_comparison():
    """Demo: Performance metrics and comparison."""
    print_section("Demo 5: Performance Metrics & AI vs Traditional")

    building_data = {
        "emissions_by_source": {
            "electricity": 20000,
            "natural_gas": 10000,
        },
        "building_type": "commercial_office",
        "building_age": 15,
    }

    print("Comparing AI-powered vs Traditional RecommendationAgent:\n")

    # AI-powered agent
    ai_agent = RecommendationAgentAI(max_recommendations=5)
    ai_result = ai_agent.execute(building_data)

    if ai_result.success:
        print("AI-Powered Agent:")
        print(f"  Recommendations: {len(ai_result.data['recommendations'])}")
        print(f"  AI Calls: {ai_result.metadata.get('ai_calls', 0)}")
        print(f"  Tool Calls: {ai_result.metadata.get('tool_calls', 0)}")
        print(f"  Time: {ai_result.metadata.get('calculation_time_ms', 0):.2f} ms")
        print(f"  Cost: ${ai_result.metadata.get('cost_usd', 0):.4f}")
        print(f"  Has AI Summary: {'ai_summary' in ai_result.data}")
        print(f"  Has ROI Analysis: {'roi_analysis' in ai_result.data}")
        print()

        # Get performance summary
        perf = ai_agent.get_performance_summary()
        print("Cumulative Performance Metrics:")
        print(f"  Total AI Calls: {perf['ai_metrics']['ai_call_count']}")
        print(f"  Total Tool Calls: {perf['ai_metrics']['tool_call_count']}")
        print(f"  Total Cost: ${perf['ai_metrics']['total_cost_usd']:.4f}")
        print(f"  Avg Cost/Analysis: ${perf['ai_metrics']['avg_cost_per_analysis']:.4f}")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 80)
    print("  RecommendationAgentAI - AI-Powered Building Optimization Demo")
    print("*" * 80)

    # Check API key availability
    if has_any_api_key():
        print("\n[INFO] Using production LLM provider (API key detected)")
    else:
        print("\n[INFO] Using demo provider (no API key detected)")
        print("[INFO] Demo provider generates synthetic responses for testing")

    print("\nThis demo showcases AI-powered recommendation generation with:")
    print("  - Energy usage analysis")
    print("  - ROI-based prioritization")
    print("  - Natural language explanations")
    print("  - Implementation planning")
    print("  - Deterministic calculations (tools)")
    print()

    try:
        # Run demos
        demo_old_inefficient_building()
        demo_modern_efficient_building()
        demo_hvac_dominated_facility()
        demo_electricity_heavy_datacenter()
        demo_performance_comparison()

        print_section("Demo Complete")
        print("All scenarios completed successfully!")
        print("\nKey Takeaways:")
        print("  1. AI provides natural language explanations for each recommendation")
        print("  2. ROI calculations are exact (via deterministic tools)")
        print("  3. Recommendations are source-specific and actionable")
        print("  4. Implementation plans provide step-by-step guidance")
        print("  5. Budget enforcement ensures cost control")
        print("  6. Deterministic execution (temperature=0, seed=42)")
        print()

    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
