# -*- coding: utf-8 -*-
"""
Example Usage: Report Narrative Agent V2 (InsightAgent Pattern)
Demonstrates the hybrid architecture: deterministic calculations + AI narratives

This example shows:
1. Deterministic report data calculation (reproducible, auditable)
2. AI-powered narrative generation with RAG (compelling, framework-compliant)
3. Stakeholder-specific customization
4. Framework compliance verification
5. Performance metrics and audit trail

Author: GreenLang Framework Team
Date: November 2025
"""

import asyncio
from datetime import datetime
from greenlang.agents.report_narrative_agent_ai_v2 import ReportNarrativeAgentAI_V2
from greenlang.intelligence import create_provider, ChatSession
from greenlang.intelligence.rag.engine import RAGEngine


async def example_basic_tcfd_report():
    """
    Example 1: Basic TCFD Report Generation

    Demonstrates:
    - Simple deterministic calculation
    - Basic AI narrative generation
    - Executive-level reporting
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic TCFD Report")
    print("=" * 80)

    # Initialize agent
    agent = ReportNarrativeAgentAI_V2(enable_audit_trail=True)

    # Prepare emissions data
    carbon_data = {
        "total_co2e_tons": 45.5,
        "total_co2e_kg": 45500,
        "emissions_breakdown": [
            {"source": "Electricity", "co2e_tons": 25.0, "percentage": 54.9},
            {"source": "Natural Gas", "co2e_tons": 15.0, "percentage": 33.0},
            {"source": "Transportation", "co2e_tons": 5.5, "percentage": 12.1}
        ],
        "carbon_intensity": {
            "kg_per_sqft": 9.1,
            "kg_per_kwh": 0.5
        }
    }

    building_info = {
        "type": "commercial_office",
        "area": 5000,
        "location": "San Francisco, CA"
    }

    period = {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }

    # STEP 1: Calculate deterministic report data
    print("\nStep 1: Calculating deterministic report data...")

    calculation_inputs = {
        "framework": "TCFD",
        "carbon_data": carbon_data,
        "building_info": building_info,
        "period": period,
        "report_format": "markdown"
    }

    report_data = agent.calculate(calculation_inputs)

    print(f"✓ Report data calculated")
    print(f"  - Total Emissions: {report_data['total_co2e_tons']} tons CO2e")
    print(f"  - Compliance Status: {report_data['compliance_status']}")
    print(f"  - Charts Generated: {len(report_data['charts'])}")
    print(f"  - Compliance Checks Passed: {len([c for c in report_data['compliance_checks'] if c['status'] == 'pass'])}/{len(report_data['compliance_checks'])}")

    # STEP 2: Generate AI narrative (requires live infrastructure)
    print("\nStep 2: Generating AI narrative...")
    print("⚠ Requires ChatSession and RAGEngine (skipped in this example)")

    # In production, you would do:
    # provider = create_provider()
    # session = ChatSession(provider)
    # rag_engine = RAGEngine()
    #
    # narrative = await agent.explain(
    #     calculation_result=report_data,
    #     context={
    #         "stakeholder_level": "executive",
    #         "industry": "Technology",
    #         "narrative_focus": "strategy"
    #     },
    #     session=session,
    #     rag_engine=rag_engine
    # )
    #
    # print("✓ Narrative generated")
    # print(narrative)

    # Show audit trail
    if agent.enable_audit_trail:
        print(f"\n✓ Audit trail captured: {len(agent.audit_trail)} entries")
        latest = agent.audit_trail[-1]
        print(f"  - Timestamp: {latest.timestamp}")
        print(f"  - Operation: {latest.operation}")
        print(f"  - Calculation steps: {len(latest.calculation_trace)}")

    return report_data


async def example_trend_analysis_report():
    """
    Example 2: Report with Year-over-Year Trend Analysis

    Demonstrates:
    - Historical comparison (YoY trends)
    - Baseline comparison
    - Trend visualization recommendations
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Report with Trend Analysis")
    print("=" * 80)

    agent = ReportNarrativeAgentAI_V2(enable_audit_trail=True)

    # Current year data
    current_data = {
        "total_co2e_tons": 42.0,
        "emissions_breakdown": [
            {"source": "Electricity", "co2e_tons": 22.0, "percentage": 52.4},
            {"source": "Natural Gas", "co2e_tons": 14.0, "percentage": 33.3},
            {"source": "Transportation", "co2e_tons": 6.0, "percentage": 14.3}
        ]
    }

    # Previous year data for YoY comparison
    previous_data = {
        "total_co2e_tons": 50.0
    }

    # Baseline data for long-term comparison
    baseline_data = {
        "total_co2e_tons": 60.0
    }

    print("\nStep 1: Calculating with trend analysis...")

    calculation_inputs = {
        "framework": "CDP",
        "carbon_data": current_data,
        "previous_period_data": previous_data,
        "baseline_data": baseline_data,
        "building_info": {
            "type": "warehouse",
            "area": 10000,
            "location": "Seattle, WA"
        },
        "period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
    }

    report_data = agent.calculate(calculation_inputs)

    print(f"✓ Report data with trends calculated")
    print(f"  - Current Emissions: {report_data['total_co2e_tons']} tons CO2e")

    if report_data.get('trends'):
        trends = report_data['trends']
        print(f"\nTrend Analysis:")
        print(f"  - YoY Change: {trends.get('yoy_change_percentage', 0):.1f}% ({trends.get('direction', 'stable')})")
        print(f"  - Change from Baseline: {trends.get('baseline_change_percentage', 0):.1f}%")
        print(f"  - Previous Year: {trends.get('previous_emissions_tons', 0):.1f} tons CO2e")
        print(f"  - Baseline: {trends.get('baseline_emissions_tons', 0):.1f} tons CO2e")

    # In production, narrative would emphasize trends:
    print("\nNarrative Context (for AI generation):")
    print("  - Stakeholder: Board of Directors")
    print("  - Focus: Strategic progress towards targets")
    print("  - Visualization: Line chart showing historical trajectory")

    return report_data


async def example_multi_framework_comparison():
    """
    Example 3: Multi-Framework Report Comparison

    Demonstrates:
    - Same data, different frameworks (TCFD vs GRI vs SASB)
    - Framework-specific compliance checks
    - Framework-specific narrative approaches
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multi-Framework Comparison")
    print("=" * 80)

    agent = ReportNarrativeAgentAI_V2(enable_audit_trail=False)

    # Same emissions data
    carbon_data = {
        "total_co2e_tons": 75.0,
        "emissions_breakdown": [
            {"source": "Scope 1 - Direct", "co2e_tons": 30.0, "percentage": 40.0},
            {"source": "Scope 2 - Indirect", "co2e_tons": 35.0, "percentage": 46.7},
            {"source": "Scope 3 - Value Chain", "co2e_tons": 10.0, "percentage": 13.3}
        ]
    }

    building_info = {
        "type": "manufacturing",
        "area": 15000,
        "location": "Austin, TX"
    }

    frameworks = ["TCFD", "GRI", "SASB"]

    print("\nCalculating reports for multiple frameworks...")

    results = {}
    for framework in frameworks:
        print(f"\n  Framework: {framework}")

        result = agent.calculate({
            "framework": framework,
            "carbon_data": carbon_data,
            "building_info": building_info
        })

        results[framework] = result

        print(f"    - Compliance Status: {result['compliance_status']}")
        print(f"    - Compliance Checks: {len(result['compliance_checks'])}")
        print(f"    - Framework Sections: {', '.join(result['framework_metadata'].get('sections', []))}")

    print("\n✓ All frameworks calculated with same data")
    print("  - Each framework has specific compliance requirements")
    print("  - Each framework has different reporting sections")
    print("  - AI narratives would adapt to framework requirements")

    return results


async def example_stakeholder_customization():
    """
    Example 4: Stakeholder-Specific Customization

    Demonstrates:
    - Executive vs Technical vs Regulatory narratives
    - Stakeholder-appropriate language and focus
    - Visualization recommendations by audience
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Stakeholder Customization")
    print("=" * 80)

    agent = ReportNarrativeAgentAI_V2()

    # Calculate once
    carbon_data = {
        "total_co2e_tons": 120.5,
        "emissions_breakdown": [
            {"source": "Electricity", "co2e_tons": 60.0, "percentage": 49.8},
            {"source": "Natural Gas", "co2e_tons": 40.0, "percentage": 33.2},
            {"source": "Fleet Vehicles", "co2e_tons": 15.5, "percentage": 12.9},
            {"source": "Refrigerants", "co2e_tons": 5.0, "percentage": 4.1}
        ]
    }

    report_data = agent.calculate({
        "framework": "TCFD",
        "carbon_data": carbon_data,
        "building_info": {
            "type": "data_center",
            "area": 20000,
            "location": "Virginia"
        }
    })

    print(f"✓ Base report data calculated: {report_data['total_co2e_tons']} tons CO2e")

    # Different stakeholder contexts
    stakeholder_contexts = {
        "Executive": {
            "stakeholder_level": "executive",
            "narrative_focus": "strategy",
            "reporting_goals": "Board presentation and investor relations"
        },
        "Technical Team": {
            "stakeholder_level": "technical",
            "narrative_focus": "metrics",
            "reporting_goals": "Detailed methodology and data quality assessment"
        },
        "Regulatory": {
            "stakeholder_level": "regulatory",
            "narrative_focus": "comprehensive",
            "reporting_goals": "SEC climate disclosure compliance"
        },
        "Board": {
            "stakeholder_level": "board",
            "narrative_focus": "risk",
            "reporting_goals": "Governance oversight and risk management"
        }
    }

    print("\nStakeholder-Specific Narrative Approaches:")
    for stakeholder, context in stakeholder_contexts.items():
        print(f"\n  {stakeholder}:")
        print(f"    - Focus: {context['narrative_focus']}")
        print(f"    - Goal: {context['reporting_goals']}")

        # In production, each would generate different narrative:
        # narrative = await agent.explain(
        #     calculation_result=report_data,
        #     context=context,
        #     session=session,
        #     rag_engine=rag_engine
        # )

        if context['stakeholder_level'] == 'executive':
            print("    - Narrative: Strategic implications, business impact, high-level insights")
            print("    - Visuals: Executive dashboard, trend arrows, key metrics")
        elif context['stakeholder_level'] == 'technical':
            print("    - Narrative: Methodology details, data quality, calculation steps")
            print("    - Visuals: Detailed breakdowns, uncertainty ranges, technical specs")
        elif context['stakeholder_level'] == 'regulatory':
            print("    - Narrative: Compliance verification, disclosure completeness, audit trail")
            print("    - Visuals: Compliance checklists, framework alignment tables")
        elif context['stakeholder_level'] == 'board':
            print("    - Narrative: Governance oversight, risk management, fiduciary duties")
            print("    - Visuals: Risk matrices, governance structures, trend comparisons")

    return report_data


async def example_visualization_recommendations():
    """
    Example 5: Data Visualization Recommendations

    Demonstrates:
    - Chart type recommendations based on data
    - Stakeholder-appropriate visualizations
    - Data storytelling guidance
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Visualization Recommendations")
    print("=" * 80)

    agent = ReportNarrativeAgentAI_V2()

    # Rich emissions data suitable for multiple visualizations
    carbon_data = {
        "total_co2e_tons": 250.0,
        "emissions_breakdown": [
            {"source": "Building Energy", "co2e_tons": 100.0, "percentage": 40.0},
            {"source": "Process Emissions", "co2e_tons": 70.0, "percentage": 28.0},
            {"source": "Transportation", "co2e_tons": 40.0, "percentage": 16.0},
            {"source": "Waste", "co2e_tons": 25.0, "percentage": 10.0},
            {"source": "Water", "co2e_tons": 15.0, "percentage": 6.0}
        ]
    }

    report_data = agent.calculate({
        "framework": "GRI",
        "carbon_data": carbon_data,
        "previous_period_data": {"total_co2e_tons": 280.0},
        "baseline_data": {"total_co2e_tons": 350.0}
    })

    print(f"✓ Report with rich visualization data calculated")
    print(f"  - Total Emissions: {report_data['total_co2e_tons']} tons CO2e")
    print(f"  - Emission Sources: {len(report_data['emissions_breakdown'])}")
    print(f"  - Trend Data: Available (YoY and baseline)")

    print("\nVisualization Opportunities:")
    print("  1. Pie Chart - Emission source breakdown (5 sources)")
    print("  2. Horizontal Bar Chart - Comparative magnitudes by source")
    print("  3. Line Chart - Historical trend (current vs previous vs baseline)")
    print("  4. Waterfall Chart - Cumulative contribution to total")
    print("  5. Donut Chart - Scope 1/2/3 categorization")

    print("\nData Storytelling Approach:")
    print("  - Lead: 'Building energy accounts for 40% of emissions'")
    print("  - Progress: '10.7% reduction from previous year'")
    print("  - Achievement: '28.6% reduction from baseline'")
    print("  - Next: 'Focus on process emissions for further reduction'")

    # In production, data_visualization_tool would provide these recommendations:
    print("\nAI Tool Output (data_visualization_tool):")
    print("  - Recommended Charts: 4 primary + 2 supplementary")
    print("  - Visual Priority: Total → Breakdown → Trend → Comparisons")
    print("  - Color Scheme: Professional blues/grays with green accent for progress")
    print("  - Annotations: Mark 40% reduction target, highlight process emissions")

    return report_data


async def example_performance_metrics():
    """
    Example 6: Performance Metrics and Audit Trail

    Demonstrates:
    - Performance tracking
    - Cost monitoring
    - Audit trail export
    - Reproducibility verification
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Performance Metrics & Audit Trail")
    print("=" * 80)

    agent = ReportNarrativeAgentAI_V2(enable_audit_trail=True)

    # Generate multiple reports
    print("\nGenerating 3 reports for performance tracking...")

    frameworks = ["TCFD", "CDP", "GRI"]
    for i, framework in enumerate(frameworks, 1):
        print(f"\n  Report {i}/{len(frameworks)}: {framework}")

        report_data = agent.calculate({
            "framework": framework,
            "carbon_data": {
                "total_co2e_tons": 50.0 + (i * 10),
                "emissions_breakdown": [
                    {"source": "Source A", "co2e_tons": 30.0, "percentage": 60.0},
                    {"source": "Source B", "co2e_tons": 20.0, "percentage": 40.0}
                ]
            }
        })

        print(f"    ✓ Calculated: {report_data['total_co2e_tons']} tons CO2e")

    # Performance summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)

    perf = agent.get_performance_summary()
    print(f"\nAgent: {perf['agent_id']}")
    print(f"Category: {perf['category']}")
    print(f"Reports Generated: {perf['total_reports']}")
    print(f"Narratives Generated: {perf['total_narratives']}")
    print(f"Total Cost: ${perf['total_cost_usd']:.4f}")
    print(f"Avg Cost per Narrative: ${perf['avg_cost_per_narrative']:.4f}")

    # Audit trail
    print("\n" + "=" * 80)
    print("Audit Trail")
    print("=" * 80)

    if agent.enable_audit_trail:
        print(f"\nTotal Audit Entries: {len(agent.audit_trail)}")

        for i, entry in enumerate(agent.audit_trail, 1):
            print(f"\nEntry {i}:")
            print(f"  - Timestamp: {entry.timestamp}")
            print(f"  - Operation: {entry.operation}")
            print(f"  - Input Hash: {entry.input_hash[:16]}...")
            print(f"  - Output Hash: {entry.output_hash[:16]}...")
            print(f"  - Calculation Steps: {len(entry.calculation_trace)}")

        # Export audit trail
        print("\n✓ Audit trail can be exported for regulatory compliance")
        print("  agent.audit_trail[0].to_dict() for JSON export")

    # Reproducibility test
    print("\n" + "=" * 80)
    print("Reproducibility Verification")
    print("=" * 80)

    test_input = {
        "framework": "TCFD",
        "carbon_data": {
            "total_co2e_tons": 100.0,
            "emissions_breakdown": []
        }
    }

    result1 = agent.calculate(test_input)
    result2 = agent.calculate(test_input)

    is_reproducible = (
        result1['total_co2e_tons'] == result2['total_co2e_tons'] and
        result1['compliance_status'] == result2['compliance_status'] and
        len(result1['compliance_checks']) == len(result2['compliance_checks'])
    )

    print(f"\n✓ Reproducibility Test: {'PASSED' if is_reproducible else 'FAILED'}")
    print(f"  - Same inputs → Same outputs: {is_reproducible}")
    print(f"  - Result 1 Hash: {hash(str(result1))}")
    print(f"  - Result 2 Hash: {hash(str(result2))}")

    return perf


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("REPORT NARRATIVE AGENT V2 - COMPREHENSIVE EXAMPLES")
    print("InsightAgent Pattern: Deterministic Calculations + AI Narratives")
    print("=" * 80)

    # Run examples
    await example_basic_tcfd_report()
    await example_trend_analysis_report()
    await example_multi_framework_comparison()
    await example_stakeholder_customization()
    await example_visualization_recommendations()
    await example_performance_metrics()

    # Summary
    print("\n" + "=" * 80)
    print("TRANSFORMATION SUMMARY")
    print("=" * 80)

    print("\nV1 → V2 Transformation:")
    print("  ✓ Pattern: ChatSession orchestration → InsightAgent (hybrid)")
    print("  ✓ Temperature: 0.0 (deterministic) → 0.6 (narrative consistency)")
    print("  ✓ Architecture: Monolithic → Separated concerns (calculate + explain)")
    print("  ✓ Data Collection: 6 deterministic tools (preserved)")
    print("  ✓ Narrative Generation: 2 new AI tools (added)")
    print("  ✓ RAG Integration: None → 4 collections for best practices")

    print("\nKey Features:")
    print("  ✓ Deterministic report data (reproducible, auditable)")
    print("  ✓ AI-powered narratives (compelling, framework-compliant)")
    print("  ✓ Stakeholder customization (executive, board, technical, regulatory)")
    print("  ✓ Multi-framework support (TCFD, CDP, GRI, SASB, SEC, ISO14064)")
    print("  ✓ Visualization recommendations (data storytelling)")
    print("  ✓ Full audit trail (regulatory compliance)")
    print("  ✓ RAG-enhanced best practices (industry peer insights)")

    print("\nUse Cases:")
    print("  ✓ Annual climate disclosures (TCFD, CDP)")
    print("  ✓ Investor relations reporting")
    print("  ✓ Board presentations")
    print("  ✓ Regulatory submissions (SEC)")
    print("  ✓ Sustainability reports (GRI)")
    print("  ✓ Industry-specific reporting (SASB)")

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
