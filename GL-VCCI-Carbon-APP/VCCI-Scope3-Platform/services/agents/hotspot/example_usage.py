"""
HotspotAnalysisAgent - Usage Examples
GL-VCCI Scope 3 Platform

Demonstrates common usage patterns for the HotspotAnalysisAgent.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import json
from pathlib import Path

from services.agents.hotspot import HotspotAnalysisAgent
from services.agents.hotspot.models import (
    Initiative,
    SupplierSwitchScenario,
    ModalShiftScenario,
    ProductSubstitutionScenario
)
from services.agents.hotspot.config import (
    HotspotAnalysisConfig,
    HotspotCriteria,
    AnalysisDimension,
    ROIConfig
)


def example_1_comprehensive_analysis():
    """Example 1: Run comprehensive analysis on emissions data."""
    print("=" * 80)
    print("EXAMPLE 1: Comprehensive Analysis")
    print("=" * 80)

    # Sample emissions data
    emissions_data = [
        {
            "record_id": "REC-001",
            "emissions_tco2e": 45000,
            "emissions_kgco2e": 45000000,
            "supplier_name": "Acme Steel Corp",
            "scope3_category": 1,
            "product_name": "Steel Sheets",
            "region": "US",
            "dqi_score": 85.0,
            "tier": 1,
            "spend_usd": 5000000
        },
        {
            "record_id": "REC-002",
            "emissions_tco2e": 22000,
            "emissions_kgco2e": 22000000,
            "supplier_name": "Beta Aluminum Ltd",
            "scope3_category": 1,
            "product_name": "Aluminum Ingots",
            "region": "EU",
            "dqi_score": 75.0,
            "tier": 2,
            "spend_usd": 3000000
        },
        {
            "record_id": "REC-003",
            "emissions_tco2e": 18000,
            "emissions_kgco2e": 18000000,
            "supplier_name": "Gamma Plastics Inc",
            "scope3_category": 1,
            "product_name": "Plastic Resins",
            "region": "US",
            "dqi_score": 45.0,
            "tier": 3,
            "spend_usd": 2000000
        }
    ]

    # Initialize agent
    agent = HotspotAnalysisAgent()

    # Run comprehensive analysis
    results = agent.analyze_comprehensive(emissions_data)

    # Display results
    print(f"\n📊 ANALYSIS SUMMARY")
    print(f"Total Emissions: {results['summary']['total_emissions_tco2e']:,.0f} tCO2e")
    print(f"Total Records: {results['summary']['total_records']}")
    print(f"Hotspots Identified: {results['summary']['n_hotspots']}")
    print(f"Insights Generated: {results['summary']['n_insights']}")
    print(f"Processing Time: {results['summary']['processing_time_seconds']:.2f}s")

    # Display Pareto analysis
    if results.get('pareto'):
        pareto = results['pareto']
        print(f"\n📈 PARETO ANALYSIS (80/20 Rule)")
        print(f"Pareto Achieved: {'Yes' if pareto.pareto_achieved else 'No'}")
        print(f"Top 20% Efficiency: {pareto.pareto_efficiency * 100:.1f}%")
        print(f"\nTop Contributors:")
        for item in pareto.top_20_percent[:3]:
            print(f"  {item.rank}. {item.entity_name}: "
                  f"{item.emissions_tco2e:,.0f} tCO2e ({item.percent_of_total:.1f}%)")

    # Display hotspots
    if results.get('hotspots'):
        hotspots = results['hotspots']
        print(f"\n🔥 HOTSPOTS DETECTED")
        print(f"Critical: {len(hotspots.critical_hotspots)}")
        print(f"High: {len(hotspots.high_hotspots)}")
        if hotspots.critical_hotspots:
            print(f"\nCritical Hotspot:")
            h = hotspots.critical_hotspots[0]
            print(f"  Entity: {h.entity_name}")
            print(f"  Emissions: {h.emissions_tco2e:,.0f} tCO2e ({h.percent_of_total:.1f}%)")
            print(f"  Rules Triggered: {', '.join(h.triggered_rules[:2])}")

    # Display insights
    if results.get('insights'):
        insights = results['insights']
        print(f"\n💡 TOP RECOMMENDATIONS")
        for i, rec in enumerate(insights.top_recommendations[:3], 1):
            print(f"  {i}. {rec}")

    print("\n" + "=" * 80 + "\n")


def example_2_pareto_and_segmentation():
    """Example 2: Detailed Pareto and segmentation analysis."""
    print("=" * 80)
    print("EXAMPLE 2: Pareto & Segmentation Analysis")
    print("=" * 80)

    # Sample data
    emissions_data = [
        {"emissions_tco2e": 50000, "supplier_name": "Supplier A", "scope3_category": 1},
        {"emissions_tco2e": 30000, "supplier_name": "Supplier B", "scope3_category": 1},
        {"emissions_tco2e": 15000, "supplier_name": "Supplier C", "scope3_category": 4},
        {"emissions_tco2e": 10000, "supplier_name": "Supplier D", "scope3_category": 1},
        {"emissions_tco2e": 8000, "supplier_name": "Supplier E", "scope3_category": 6},
    ]

    agent = HotspotAnalysisAgent()

    # Pareto analysis by supplier
    print("\n📊 PARETO ANALYSIS BY SUPPLIER")
    pareto = agent.analyze_pareto(emissions_data, "supplier_name")
    for item in pareto.top_20_percent:
        print(f"  {item.rank}. {item.entity_name}: "
              f"{item.emissions_tco2e:,.0f} tCO2e "
              f"(Cumulative: {item.cumulative_percent:.1f}%)")

    # Multi-dimensional segmentation
    print("\n📊 MULTI-DIMENSIONAL SEGMENTATION")
    segments = agent.analyze_segmentation(
        emissions_data,
        dimensions=[AnalysisDimension.SUPPLIER, AnalysisDimension.CATEGORY]
    )

    for dimension, analysis in segments.items():
        print(f"\n  {dimension.value.upper()}:")
        print(f"    Total Segments: {analysis.n_segments}")
        print(f"    Top 3 Concentration: {analysis.top_3_concentration:.1f}%")
        print(f"    Top Segment: {analysis.top_10_segments[0].segment_name} "
              f"({analysis.top_10_segments[0].percent_of_total:.1f}%)")

    print("\n" + "=" * 80 + "\n")


def example_3_roi_and_abatement_curve():
    """Example 3: ROI analysis and abatement curve generation."""
    print("=" * 80)
    print("EXAMPLE 3: ROI Analysis & Abatement Curve")
    print("=" * 80)

    agent = HotspotAnalysisAgent()

    # Define initiatives
    initiatives = [
        Initiative(
            name="Switch to Renewable Energy",
            description="Install solar panels at manufacturing facilities",
            reduction_potential_tco2e=10000,
            implementation_cost_usd=200000,
            annual_operating_cost_usd=5000,
            annual_savings_usd=50000
        ),
        Initiative(
            name="Supplier Engagement Program",
            description="Collect primary data from top 20 suppliers",
            reduction_potential_tco2e=5000,
            implementation_cost_usd=75000,
            annual_operating_cost_usd=10000,
            annual_savings_usd=0
        ),
        Initiative(
            name="Modal Shift to Rail",
            description="Shift 50% of road freight to rail",
            reduction_potential_tco2e=3000,
            implementation_cost_usd=-10000,  # Savings
            annual_operating_cost_usd=0,
            annual_savings_usd=15000
        )
    ]

    # Calculate ROI for each
    print("\n💰 ROI ANALYSIS")
    for initiative in initiatives:
        roi = agent.calculate_roi(initiative)
        print(f"\n  {initiative.name}:")
        print(f"    Reduction: {initiative.reduction_potential_tco2e:,.0f} tCO2e")
        print(f"    Cost per tCO2e: ${roi.roi_usd_per_tco2e:.2f}")
        if roi.payback_period_years:
            print(f"    Payback: {roi.payback_period_years:.1f} years")
        print(f"    10-Year NPV: ${roi.npv_10y_usd:,.0f}")

    # Generate abatement curve
    print("\n📉 MARGINAL ABATEMENT COST CURVE")
    macc = agent.generate_abatement_curve(initiatives)
    print(f"  Total Reduction Potential: {macc.total_reduction_potential_tco2e:,.0f} tCO2e")
    print(f"  Total Cost: ${macc.total_cost_usd:,.0f}")
    print(f"  Weighted Avg Cost: ${macc.weighted_average_cost_per_tco2e:.2f}/tCO2e")
    print(f"  Initiatives with Savings: {macc.n_negative_cost}")
    print(f"\n  Sorted by Cost-Effectiveness:")
    for point in macc.initiatives:
        status = "💰 SAVINGS" if point.cost_per_tco2e < 0 else ""
        print(f"    {point.initiative_name}: ${point.cost_per_tco2e:.2f}/tCO2e "
              f"({point.reduction_tco2e:,.0f} tCO2e) {status}")

    print("\n" + "=" * 80 + "\n")


def example_4_scenario_modeling():
    """Example 4: Scenario modeling (framework stubs)."""
    print("=" * 80)
    print("EXAMPLE 4: Scenario Modeling (Framework v1.0)")
    print("=" * 80)

    agent = HotspotAnalysisAgent()

    # Supplier switching scenario
    print("\n🔄 SUPPLIER SWITCHING SCENARIO")
    supplier_scenario = SupplierSwitchScenario(
        name="Switch to Low Carbon Steel Supplier",
        from_supplier="High Carbon Steel Co",
        to_supplier="Green Steel Inc",
        products=["steel_sheets", "steel_bars"],
        current_emissions_tco2e=45000,
        new_emissions_tco2e=30000,
        estimated_reduction_tco2e=15000,
        estimated_cost_usd=100000
    )

    result = agent.model_scenario(supplier_scenario)
    print(f"  Baseline: {result.baseline_emissions_tco2e:,.0f} tCO2e")
    print(f"  Projected: {result.projected_emissions_tco2e:,.0f} tCO2e")
    print(f"  Reduction: {result.reduction_tco2e:,.0f} tCO2e ({result.reduction_percent:.1f}%)")
    print(f"  Cost: ${result.implementation_cost_usd:,.0f}")
    print(f"  ROI: ${result.roi_usd_per_tco2e:.2f}/tCO2e")

    # Modal shift scenario
    print("\n🚢 MODAL SHIFT SCENARIO")
    modal_scenario = ModalShiftScenario(
        name="Shift Air to Sea Freight",
        from_mode="air",
        to_mode="sea",
        routes=["US-EU"],
        volume_pct=50,
        estimated_reduction_tco2e=2000,
        estimated_cost_usd=-10000  # Savings
    )

    result = agent.model_scenario(modal_scenario)
    print(f"  Reduction: {result.reduction_tco2e:,.0f} tCO2e")
    print(f"  Annual Savings: ${result.annual_savings_usd:,.0f}")
    print(f"  ROI: ${result.roi_usd_per_tco2e:.2f}/tCO2e (negative = savings)")

    # Compare scenarios
    print("\n📊 SCENARIO COMPARISON")
    comparison = agent.compare_scenarios([supplier_scenario, modal_scenario])
    print(f"  Total Reduction: {comparison['total_reduction_potential_tco2e']:,.0f} tCO2e")
    print(f"  Total Cost: ${comparison['total_implementation_cost_usd']:,.0f}")
    print(f"\n  Ranked by Cost-Effectiveness:")
    for scenario in comparison['ranked_by_roi']:
        print(f"    {scenario['name']}: ${scenario['roi_usd_per_tco2e']:.2f}/tCO2e")

    print("\n⚠️  NOTE: Scenario modeling is framework v1.0 with stubs.")
    print("   Full optimization coming in Week 27+")

    print("\n" + "=" * 80 + "\n")


def example_5_custom_configuration():
    """Example 5: Using custom configuration."""
    print("=" * 80)
    print("EXAMPLE 5: Custom Configuration")
    print("=" * 80)

    # Create custom configuration
    custom_config = HotspotAnalysisConfig(
        hotspot_criteria=HotspotCriteria(
            emission_threshold_tco2e=5000.0,  # Lower threshold
            percent_threshold=10.0,            # Higher percentage
            dqi_threshold=60.0,                # Higher DQI requirement
            tier_threshold=2,                  # Flag Tier 2 and 3
            concentration_threshold=25.0       # Lower concentration risk
        ),
        roi_config=ROIConfig(
            discount_rate=0.10,               # 10% discount rate
            analysis_period_years=15,          # 15-year analysis
            carbon_price_usd_per_tco2e=75.0   # $75/tCO2e carbon price
        )
    )

    # Initialize agent with custom config
    agent = HotspotAnalysisAgent(config=custom_config)

    print("\n⚙️  CUSTOM CONFIGURATION")
    print(f"  Emission Threshold: {custom_config.hotspot_criteria.emission_threshold_tco2e:,.0f} tCO2e")
    print(f"  DQI Threshold: {custom_config.hotspot_criteria.dqi_threshold:.0f}")
    print(f"  Discount Rate: {custom_config.roi_config.discount_rate * 100:.0f}%")
    print(f"  Carbon Price: ${custom_config.roi_config.carbon_price_usd_per_tco2e:.0f}/tCO2e")

    # Use agent with custom config
    emissions_data = [
        {"emissions_tco2e": 6000, "supplier_name": "Supplier A", "dqi_score": 55.0},
        {"emissions_tco2e": 4000, "supplier_name": "Supplier B", "dqi_score": 70.0},
        {"emissions_tco2e": 3000, "supplier_name": "Supplier C", "dqi_score": 65.0},
    ]

    hotspots = agent.identify_hotspots(emissions_data)
    print(f"\n  Hotspots Found: {hotspots.n_hotspots}")
    print(f"  (With custom thresholds)")

    print("\n" + "=" * 80 + "\n")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "HotspotAnalysisAgent v1.0 - Usage Examples" + " " * 19 + "║")
    print("║" + " " * 20 + "GL-VCCI Scope 3 Platform" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Run examples
    example_1_comprehensive_analysis()
    example_2_pareto_and_segmentation()
    example_3_roi_and_abatement_curve()
    example_4_scenario_modeling()
    example_5_custom_configuration()

    print("\n✅ All examples completed successfully!")
    print("\nFor more information, see:")
    print("  - README.md: Complete user guide")
    print("  - IMPLEMENTATION_SUMMARY.md: Technical details")
    print("  - Source code: Detailed API documentation")
    print("\n")


if __name__ == "__main__":
    main()
