"""ReportAgentAI Demo - AI-Powered Emissions Report Generation

This demo showcases the AI-powered ReportAgent, which generates
comprehensive emissions reports compliant with international frameworks.

Features Demonstrated:
1. Multi-framework support (TCFD, CDP, GRI, SASB, SEC, ISO14064)
2. AI-generated narratives for report sections
3. Executive summaries for leadership
4. Compliance verification
5. Trend analysis and YoY comparisons
6. Chart and visualization data generation
7. Tool-first numerics (deterministic calculations)

The demo includes multiple scenarios:
- TCFD report for commercial office
- CDP disclosure for manufacturing facility
- GRI sustainability report
- SASB sector-specific report
- Quarterly trend analysis
- Multi-year comparison report

Author: GreenLang Framework Team
Date: October 2025
"""

import asyncio
from typing import Dict, Any
from greenlang.agents.report_agent_ai import ReportAgentAI
from greenlang.intelligence import has_any_api_key


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_report_summary(data: Dict[str, Any]) -> None:
    """Print report summary."""
    print("Report Summary:")
    print(f"  Framework: {data.get('framework', 'N/A')}")
    print(f"  Format: {data.get('format', 'N/A')}")
    print(f"  Generated: {data.get('generated_at', 'N/A')}")
    print(f"  Total Emissions: {data.get('total_co2e_tons', 0):.2f} metric tons CO2e")
    print(f"                   {data.get('total_co2e_kg', 0):,.0f} kg CO2e")
    print()


def print_emissions_breakdown(data: Dict[str, Any]) -> None:
    """Print emissions breakdown."""
    breakdown = data.get("emissions_breakdown", [])

    if not breakdown:
        return

    print("Emissions Breakdown:")
    for item in breakdown:
        source = item.get("source", "Unknown")
        tons = item.get("co2e_tons", 0)
        pct = item.get("percentage", 0)
        print(f"  {source:20} {tons:8.2f} tons ({pct:5.2f}%)")
    print()


def print_carbon_intensity(data: Dict[str, Any]) -> None:
    """Print carbon intensity metrics."""
    intensity = data.get("carbon_intensity", {})

    if not intensity:
        return

    print("Carbon Intensity:")
    if "per_sqft" in intensity:
        print(f"  Per Square Foot: {intensity['per_sqft']:.4f} kg CO2e/sqft")
    if "per_person" in intensity:
        print(f"  Per Person: {intensity['per_person']:.2f} kg CO2e/person")
    print()


def print_trends(data: Dict[str, Any]) -> None:
    """Print trend analysis."""
    trends = data.get("trends", {})

    if not trends:
        return

    print("Trend Analysis:")
    print(f"  Current Period: {trends.get('current_emissions_tons', 0):.2f} tons")

    if "previous_emissions_tons" in trends:
        print(f"  Previous Period: {trends['previous_emissions_tons']:.2f} tons")
        print(f"  YoY Change: {trends.get('yoy_change_tons', 0):+.2f} tons ({trends.get('yoy_change_percentage', 0):+.2f}%)")
        print(f"  Direction: {trends.get('direction', 'N/A').title()}")

    if "baseline_emissions_tons" in trends:
        print(f"  Baseline (2020): {trends['baseline_emissions_tons']:.2f} tons")
        print(f"  Change from Baseline: {trends.get('baseline_change_tons', 0):+.2f} tons ({trends.get('baseline_change_percentage', 0):+.2f}%)")

    print()


def print_compliance_status(data: Dict[str, Any]) -> None:
    """Print compliance status."""
    status = data.get("compliance_status", "Unknown")
    checks = data.get("compliance_checks", [])

    print(f"Compliance Status: {status}")

    if checks:
        print(f"  Total Checks: {len(checks)}")
        passed = sum(1 for c in checks if c.get("status") == "pass")
        print(f"  Passed: {passed}/{len(checks)}")

        print("\n  Compliance Checks:")
        for check in checks[:5]:  # Show top 5
            status_icon = "✓" if check.get("status") == "pass" else "✗"
            requirement = check.get("requirement", "N/A")
            print(f"    {status_icon} {requirement}")

    print()


def print_executive_summary(data: Dict[str, Any]) -> None:
    """Print executive summary."""
    summary = data.get("executive_summary", "")

    if summary:
        print("Executive Summary:")
        print(f"{summary}\n")


def print_ai_narrative(data: Dict[str, Any]) -> None:
    """Print AI-generated narrative."""
    narrative = data.get("ai_narrative", "")

    if narrative:
        print("AI-Generated Report Narrative:")
        # Print first 500 characters
        if len(narrative) > 500:
            print(f"{narrative[:500]}...\n")
        else:
            print(f"{narrative}\n")


def print_charts(data: Dict[str, Any]) -> None:
    """Print chart information."""
    charts = data.get("charts", {})

    if not charts:
        return

    print(f"Visualization Charts Generated: {len(charts)}")
    for chart_name, chart_data in charts.items():
        chart_type = chart_data.get("type", "N/A")
        title = chart_data.get("title", "N/A")
        data_points = len(chart_data.get("data", []))
        print(f"  - {title} ({chart_type}, {data_points} data points)")
    print()


def print_performance_metrics(metadata: Dict[str, Any]) -> None:
    """Print performance metrics."""
    print("Performance Metrics:")
    print(f"  Calculation Time: {metadata.get('calculation_time_ms', 0):.2f} ms")
    print(f"  AI Calls: {metadata.get('ai_calls', 0)}")
    print(f"  Tool Calls: {metadata.get('tool_calls', 0)}")
    print(f"  Cost: ${metadata.get('cost_usd', 0):.4f}")
    print(f"  Provider: {metadata.get('provider', 'N/A')}")
    print(f"  Model: {metadata.get('model', 'N/A')}")
    print(f"  Deterministic: {metadata.get('deterministic', False)}")
    print()


def demo_tcfd_commercial_office():
    """Demo: TCFD report for commercial office building."""
    print_section("Demo 1: TCFD Report - Commercial Office Building")

    report_data = {
        "framework": "TCFD",
        "format": "markdown",
        "carbon_data": {
            "total_co2e_tons": 125.5,
            "total_co2e_kg": 125500,
            "emissions_breakdown": [
                {"source": "electricity", "co2e_tons": 75.0, "percentage": 59.76},
                {"source": "natural_gas", "co2e_tons": 35.0, "percentage": 27.89},
                {"source": "diesel", "co2e_tons": 10.5, "percentage": 8.37},
                {"source": "waste", "co2e_tons": 5.0, "percentage": 3.98},
            ],
            "carbon_intensity": {
                "per_sqft": 0.628,
                "per_person": 251.0,
            },
        },
        "building_info": {
            "type": "commercial_office",
            "area": 200000,
            "occupancy": 500,
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "duration": 12,
            "duration_unit": "months",
        },
    }

    print("Building Profile:")
    print(f"  Type: {report_data['building_info']['type']}")
    print(f"  Area: {report_data['building_info']['area']:,} sqft")
    print(f"  Occupancy: {report_data['building_info']['occupancy']} people")
    print(f"  Period: {report_data['period']['start_date']} to {report_data['period']['end_date']}")
    print()

    # Create agent
    agent = ReportAgentAI(
        budget_usd=1.0,
        enable_ai_narrative=True,
        enable_executive_summary=True,
        enable_compliance_check=True,
    )

    # Generate report
    print("Generating TCFD-compliant report...\n")
    result = agent.execute(report_data)

    if result.success:
        data = result.data

        # Print results
        print_report_summary(data)
        print_emissions_breakdown(data)
        print_carbon_intensity(data)
        print_compliance_status(data)
        print_charts(data)
        print_executive_summary(data)
        print_ai_narrative(data)
        print_performance_metrics(result.metadata)
    else:
        print(f"Error: {result.error}")


def demo_cdp_manufacturing_facility():
    """Demo: CDP disclosure for manufacturing facility."""
    print_section("Demo 2: CDP Disclosure - Manufacturing Facility")

    report_data = {
        "framework": "CDP",
        "format": "markdown",
        "carbon_data": {
            "total_co2e_tons": 450.0,
            "emissions_breakdown": [
                {"source": "natural_gas", "co2e_tons": 250.0, "percentage": 55.56},
                {"source": "electricity", "co2e_tons": 150.0, "percentage": 33.33},
                {"source": "diesel", "co2e_tons": 40.0, "percentage": 8.89},
                {"source": "propane", "co2e_tons": 10.0, "percentage": 2.22},
            ],
            "carbon_intensity": {
                "per_sqft": 1.125,
            },
        },
        "building_info": {
            "type": "industrial",
            "area": 400000,
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    }

    print("Facility Profile:")
    print(f"  Type: {report_data['building_info']['type']}")
    print(f"  Area: {report_data['building_info']['area']:,} sqft")
    print(f"  Total Emissions: {report_data['carbon_data']['total_co2e_tons']} tons CO2e/year")
    print()

    # Create agent
    agent = ReportAgentAI(budget_usd=1.0)

    # Generate report
    print("Generating CDP disclosure...\n")
    result = agent.execute(report_data)

    if result.success:
        data = result.data

        print_report_summary(data)
        print_emissions_breakdown(data)
        print_compliance_status(data)
        print_executive_summary(data)
        print_performance_metrics(result.metadata)
    else:
        print(f"Error: {result.error}")


def demo_gri_sustainability_report():
    """Demo: GRI sustainability report."""
    print_section("Demo 3: GRI Sustainability Report - Corporate Campus")

    report_data = {
        "framework": "GRI",
        "format": "markdown",
        "carbon_data": {
            "total_co2e_tons": 280.0,
            "emissions_breakdown": [
                {"source": "electricity", "co2e_tons": 180.0, "percentage": 64.29},
                {"source": "natural_gas", "co2e_tons": 70.0, "percentage": 25.00},
                {"source": "fleet_diesel", "co2e_tons": 25.0, "percentage": 8.93},
                {"source": "refrigerants", "co2e_tons": 5.0, "percentage": 1.79},
            ],
            "carbon_intensity": {
                "per_sqft": 0.933,
                "per_person": 350.0,
            },
        },
        "building_info": {
            "type": "corporate_campus",
            "area": 300000,
            "occupancy": 800,
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    }

    print("Campus Profile:")
    print(f"  Type: {report_data['building_info']['type']}")
    print(f"  Area: {report_data['building_info']['area']:,} sqft")
    print(f"  Occupancy: {report_data['building_info']['occupancy']} people")
    print()

    # Create agent
    agent = ReportAgentAI(budget_usd=1.0)

    # Generate report
    print("Generating GRI 305: Emissions report...\n")
    result = agent.execute(report_data)

    if result.success:
        data = result.data

        print_report_summary(data)
        print_emissions_breakdown(data)
        print_carbon_intensity(data)
        print_executive_summary(data)
        print_performance_metrics(result.metadata)
    else:
        print(f"Error: {result.error}")


def demo_sasb_sector_specific():
    """Demo: SASB sector-specific report."""
    print_section("Demo 4: SASB Sector-Specific Report - Data Center")

    report_data = {
        "framework": "SASB",
        "format": "markdown",
        "carbon_data": {
            "total_co2e_tons": 950.0,
            "emissions_breakdown": [
                {"source": "electricity_it", "co2e_tons": 750.0, "percentage": 78.95},
                {"source": "electricity_cooling", "co2e_tons": 150.0, "percentage": 15.79},
                {"source": "backup_generators", "co2e_tons": 40.0, "percentage": 4.21},
                {"source": "facility", "co2e_tons": 10.0, "percentage": 1.05},
            ],
            "carbon_intensity": {
                "per_sqft": 19.0,
            },
        },
        "building_info": {
            "type": "data_center",
            "area": 50000,
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    }

    print("Data Center Profile:")
    print(f"  Type: {report_data['building_info']['type']}")
    print(f"  Area: {report_data['building_info']['area']:,} sqft")
    print(f"  Electricity Dominance: {report_data['carbon_data']['emissions_breakdown'][0]['percentage']:.1f}%")
    print()

    # Create agent
    agent = ReportAgentAI(budget_usd=1.0)

    # Generate report
    print("Generating SASB sector-specific report...\n")
    result = agent.execute(report_data)

    if result.success:
        data = result.data

        print_report_summary(data)
        print_emissions_breakdown(data)
        print_compliance_status(data)
        print_executive_summary(data)
        print_performance_metrics(result.metadata)
    else:
        print(f"Error: {result.error}")


def demo_quarterly_trend_analysis():
    """Demo: Quarterly report with trend analysis."""
    print_section("Demo 5: Quarterly Trend Analysis - Q1 2025 vs Q1 2024")

    report_data = {
        "framework": "TCFD",
        "format": "markdown",
        "carbon_data": {
            "total_co2e_tons": 28.5,
            "emissions_breakdown": [
                {"source": "electricity", "co2e_tons": 18.0, "percentage": 63.16},
                {"source": "natural_gas", "co2e_tons": 8.5, "percentage": 29.82},
                {"source": "diesel", "co2e_tons": 2.0, "percentage": 7.02},
            ],
            "carbon_intensity": {
                "per_sqft": 0.570,
                "per_person": 142.5,
            },
        },
        "building_info": {
            "type": "commercial_office",
            "area": 50000,
            "occupancy": 200,
        },
        "period": {
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "duration": 3,
            "duration_unit": "months",
        },
        "previous_period_data": {
            "total_co2e_tons": 32.0,
        },
        "baseline_data": {
            "total_co2e_tons": 35.0,
            "year": 2020,
        },
    }

    print("Reporting Period:")
    print(f"  Current: Q1 2025 ({report_data['carbon_data']['total_co2e_tons']} tons)")
    print(f"  Previous: Q1 2024 ({report_data['previous_period_data']['total_co2e_tons']} tons)")
    print(f"  Baseline: 2020 ({report_data['baseline_data']['total_co2e_tons']} tons)")
    print()

    # Create agent
    agent = ReportAgentAI(budget_usd=1.0)

    # Generate report
    print("Generating quarterly trend report...\n")
    result = agent.execute(report_data)

    if result.success:
        data = result.data

        print_report_summary(data)
        print_emissions_breakdown(data)
        print_trends(data)
        print_executive_summary(data)
        print_performance_metrics(result.metadata)
    else:
        print(f"Error: {result.error}")


def demo_multi_format_comparison():
    """Demo: Generate same report in multiple formats."""
    print_section("Demo 6: Multi-Format Report Generation")

    base_data = {
        "carbon_data": {
            "total_co2e_tons": 75.0,
            "emissions_breakdown": [
                {"source": "electricity", "co2e_tons": 50.0, "percentage": 66.67},
                {"source": "natural_gas", "co2e_tons": 25.0, "percentage": 33.33},
            ],
        },
        "building_info": {
            "type": "commercial_office",
            "area": 100000,
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    }

    # Create agent
    agent = ReportAgentAI(budget_usd=0.50)

    formats = ["markdown", "text", "json"]

    for fmt in formats:
        print(f"\nGenerating {fmt.upper()} format...")
        report_data = {**base_data, "format": fmt, "framework": "TCFD"}

        result = agent.execute(report_data)

        if result.success:
            print(f"  ✓ {fmt.upper()} report generated successfully")
            print(f"  Size: {len(str(result.data['report']))} characters")
        else:
            print(f"  ✗ Failed: {result.error}")

    # Performance summary
    print("\nCumulative Performance:")
    perf = agent.get_performance_summary()
    print(f"  Total Reports: {perf['ai_metrics']['ai_call_count']}")
    print(f"  Total Cost: ${perf['ai_metrics']['total_cost_usd']:.4f}")
    print(f"  Avg Cost/Report: ${perf['ai_metrics']['avg_cost_per_report']:.4f}")


def demo_framework_comparison():
    """Demo: Compare different frameworks for same building."""
    print_section("Demo 7: Framework Comparison - Same Building, Multiple Standards")

    base_data = {
        "carbon_data": {
            "total_co2e_tons": 100.0,
            "emissions_breakdown": [
                {"source": "electricity", "co2e_tons": 60.0, "percentage": 60.0},
                {"source": "natural_gas", "co2e_tons": 30.0, "percentage": 30.0},
                {"source": "diesel", "co2e_tons": 10.0, "percentage": 10.0},
            ],
            "carbon_intensity": {
                "per_sqft": 0.667,
            },
        },
        "building_info": {
            "type": "commercial_office",
            "area": 150000,
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    }

    print("Building: 150,000 sqft Commercial Office")
    print("Total Emissions: 100.0 tons CO2e\n")

    # Create agent
    agent = ReportAgentAI(budget_usd=0.30)

    frameworks = ["TCFD", "CDP", "GRI", "SASB"]

    print("Generating reports for multiple frameworks:\n")

    for framework in frameworks:
        report_data = {**base_data, "framework": framework}
        result = agent.execute(report_data)

        if result.success:
            data = result.data
            status = data.get("compliance_status", "Unknown")
            checks = len(data.get("compliance_checks", []))
            print(f"  {framework:10} ✓ Generated | Compliance: {status} | Checks: {checks}")
        else:
            print(f"  {framework:10} ✗ Failed: {result.error}")

    print("\nFramework Comparison Complete!")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 80)
    print("  ReportAgentAI - AI-Powered Emissions Report Generation Demo")
    print("*" * 80)

    # Check API key availability
    if has_any_api_key():
        print("\n[INFO] Using production LLM provider (API key detected)")
    else:
        print("\n[INFO] Using demo provider (no API key detected)")
        print("[INFO] Demo provider generates synthetic responses for testing")

    print("\nThis demo showcases AI-powered report generation with:")
    print("  - Multi-framework support (TCFD, CDP, GRI, SASB, SEC, ISO14064)")
    print("  - AI-generated narratives and summaries")
    print("  - Compliance verification")
    print("  - Trend analysis")
    print("  - Chart generation")
    print("  - Deterministic calculations (tools)")
    print()

    try:
        # Run demos
        demo_tcfd_commercial_office()
        demo_cdp_manufacturing_facility()
        demo_gri_sustainability_report()
        demo_sasb_sector_specific()
        demo_quarterly_trend_analysis()
        demo_multi_format_comparison()
        demo_framework_comparison()

        print_section("Demo Complete")
        print("All scenarios completed successfully!")
        print("\nKey Takeaways:")
        print("  1. AI generates professional narratives for each framework")
        print("  2. All numeric calculations are exact (via deterministic tools)")
        print("  3. Reports are framework-compliant and audit-ready")
        print("  4. Executive summaries are tailored for leadership")
        print("  5. Trend analysis provides YoY and baseline comparisons")
        print("  6. Budget enforcement ensures cost control")
        print("  7. Deterministic execution (temperature=0, seed=42)")
        print()

    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
