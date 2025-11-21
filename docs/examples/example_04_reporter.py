# -*- coding: utf-8 -*-
"""
Example 04: Reporter Agent

This example demonstrates report generation using BaseReporter.
You'll learn:
- How to aggregate data for reporting
- How to build report sections
- How to render reports in multiple formats (Markdown, HTML, JSON)
- How to create professional-looking reports
"""

from greenlang.agents import BaseReporter, ReporterConfig, ReportSection
from typing import Dict, Any, List
from datetime import datetime


class EnergyConsumptionReporter(BaseReporter):
    """
    Generate monthly energy consumption reports.

    This agent demonstrates:
    - Data aggregation
    - Multi-section reports
    - Multiple output formats
    - Summary generation
    """

    def __init__(self, output_format='markdown'):
        config = ReporterConfig(
            name="Monthly Energy Report",
            description="Comprehensive monthly energy consumption analysis",
            output_format=output_format,
            include_summary=True,
            include_details=True
        )
        super().__init__(config)

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate energy consumption data.

        Args:
            input_data: Must contain 'readings' list with daily consumption data

        Returns:
            Aggregated statistics dictionary
        """
        readings = input_data['readings']

        # Calculate statistics
        total_kwh = sum(r['kwh'] for r in readings)
        avg_kwh = total_kwh / len(readings) if readings else 0
        max_reading = max(readings, key=lambda r: r['kwh']) if readings else None
        min_reading = min(readings, key=lambda r: r['kwh']) if readings else None

        # Calculate cost (assuming $0.12 per kWh)
        rate = 0.12
        total_cost = total_kwh * rate
        avg_daily_cost = total_cost / len(readings) if readings else 0

        # Calculate emissions (assuming 0.5 kg CO2 per kWh)
        emission_factor = 0.5
        total_emissions_kg = total_kwh * emission_factor
        total_emissions_tons = total_emissions_kg / 1000

        return {
            'month': input_data.get('month', 'Unknown'),
            'num_days': len(readings),
            'total_consumption_kwh': round(total_kwh, 2),
            'average_daily_kwh': round(avg_kwh, 2),
            'peak_day': max_reading['date'] if max_reading else 'N/A',
            'peak_consumption_kwh': max_reading['kwh'] if max_reading else 0,
            'lowest_day': min_reading['date'] if min_reading else 'N/A',
            'lowest_consumption_kwh': min_reading['kwh'] if min_reading else 0,
            'total_cost_usd': round(total_cost, 2),
            'average_daily_cost_usd': round(avg_daily_cost, 2),
            'total_emissions_tons': round(total_emissions_tons, 4),
            'rate_per_kwh': rate,
            'emission_factor': emission_factor
        }

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        """
        Build report sections from aggregated data.

        Args:
            aggregated_data: Aggregated statistics

        Returns:
            List of ReportSection objects
        """
        sections = []

        # Section 1: Consumption Statistics
        consumption_table = [
            {'Metric': 'Total Consumption', 'Value': f"{aggregated_data['total_consumption_kwh']:,.2f} kWh"},
            {'Metric': 'Average Daily', 'Value': f"{aggregated_data['average_daily_kwh']:,.2f} kWh"},
            {'Metric': 'Peak Day', 'Value': f"{aggregated_data['peak_day']} ({aggregated_data['peak_consumption_kwh']:.2f} kWh)"},
            {'Metric': 'Lowest Day', 'Value': f"{aggregated_data['lowest_day']} ({aggregated_data['lowest_consumption_kwh']:.2f} kWh)"},
        ]

        sections.append(ReportSection(
            title="Consumption Statistics",
            content=consumption_table,
            level=2,
            section_type="table"
        ))

        # Section 2: Cost Analysis
        cost_table = [
            {'Metric': 'Rate', 'Value': f"${aggregated_data['rate_per_kwh']:.2f} per kWh"},
            {'Metric': 'Total Cost', 'Value': f"${aggregated_data['total_cost_usd']:,.2f}"},
            {'Metric': 'Average Daily Cost', 'Value': f"${aggregated_data['average_daily_cost_usd']:.2f}"},
        ]

        sections.append(ReportSection(
            title="Cost Analysis",
            content=cost_table,
            level=2,
            section_type="table"
        ))

        # Section 3: Environmental Impact
        emissions_content = (
            f"Based on an emission factor of {aggregated_data['emission_factor']} kg CO2 per kWh, "
            f"the total carbon footprint for {aggregated_data['month']} is "
            f"**{aggregated_data['total_emissions_tons']:.4f} metric tons CO2e**.\n\n"
            f"This is equivalent to:\n"
            f"- Driving approximately {aggregated_data['total_emissions_tons'] * 2204:.0f} miles in an average car\n"
            f"- The annual CO2 absorption of {aggregated_data['total_emissions_tons'] * 50:.1f} tree seedlings"
        )

        sections.append(ReportSection(
            title="Environmental Impact",
            content=emissions_content,
            level=2,
            section_type="text"
        ))

        # Section 4: Recommendations
        avg_kwh = aggregated_data['average_daily_kwh']

        if avg_kwh > 100:
            recommendations = [
                "**High Usage Detected**: Consider energy audit to identify inefficiencies",
                "Upgrade to LED lighting throughout the facility (20-30% reduction)",
                "Install smart thermostats for HVAC optimization (15-25% reduction)",
                "Schedule HVAC maintenance to ensure optimal efficiency",
                "Review equipment runtime schedules to avoid unnecessary consumption"
            ]
        elif avg_kwh > 50:
            recommendations = [
                "**Moderate Usage**: Implement energy-saving best practices",
                "Ensure proper insulation to reduce heating/cooling losses",
                "Use occupancy sensors in common areas",
                "Consider time-of-use rates for cost optimization"
            ]
        else:
            recommendations = [
                "**Excellent Energy Management**: Usage is within optimal range",
                "Continue current energy conservation practices",
                "Monitor for any unusual spikes in consumption"
            ]

        sections.append(ReportSection(
            title="Recommendations",
            content=recommendations,
            level=2,
            section_type="list"
        ))

        return sections


def main():
    """Run the example."""
    print("=" * 60)
    print("Example 04: Reporter Agent")
    print("=" * 60)
    print()

    # Sample data: daily energy readings for January
    monthly_data = {
        'month': 'January 2025',
        'readings': [
            {'date': '2025-01-01', 'kwh': 95.3},
            {'date': '2025-01-02', 'kwh': 102.7},
            {'date': '2025-01-03', 'kwh': 88.4},
            {'date': '2025-01-04', 'kwh': 125.8},
            {'date': '2025-01-05', 'kwh': 91.2},
            {'date': '2025-01-06', 'kwh': 87.6},
            {'date': '2025-01-07', 'kwh': 98.9},
        ]
    }

    # Example 1: Markdown Report
    print("Test 1: Generate Markdown Report")
    print("-" * 40)

    reporter = EnergyConsumptionReporter(output_format='markdown')
    result = reporter.run(monthly_data)

    if result.success:
        print(f"✓ Report generated successfully")
        print(f"  Format: {result.data['format']}")
        print(f"  Sections: {result.data['sections_count']}")
        print(f"  Execution time: {result.metrics.execution_time_ms:.2f}ms")
        print()
        print("Report Content:")
        print("=" * 60)
        print(result.data['report'])
        print("=" * 60)
    else:
        print(f"✗ Report generation failed: {result.error}")
    print()

    # Example 2: HTML Report
    print("Test 2: Generate HTML Report")
    print("-" * 40)

    reporter_html = EnergyConsumptionReporter(output_format='html')
    result_html = reporter_html.run(monthly_data)

    if result_html.success:
        print(f"✓ HTML report generated successfully")
        print(f"  Report length: {len(result_html.data['report'])} characters")

        # Save to file
        with open('energy_report.html', 'w', encoding='utf-8') as f:
            f.write(result_html.data['report'])
        print(f"  Saved to: energy_report.html")
    else:
        print(f"✗ HTML report generation failed: {result_html.error}")
    print()

    # Example 3: JSON Report
    print("Test 3: Generate JSON Report")
    print("-" * 40)

    reporter_json = EnergyConsumptionReporter(output_format='json')
    result_json = reporter_json.run(monthly_data)

    if result_json.success:
        print(f"✓ JSON report generated successfully")

        # Parse and display
        import json
        report_data = json.loads(result_json.data['report'])
        print(f"  Report name: {report_data['report_name']}")
        print(f"  Generated at: {report_data['generated_at']}")
        print(f"  Number of sections: {len(report_data['sections'])}")
    else:
        print(f"✗ JSON report generation failed: {result_json.error}")
    print()

    # Example 4: Check statistics
    print("Reporter Statistics:")
    print("-" * 40)
    stats = reporter.get_stats()
    print(f"  Total executions: {stats['executions']}")
    print(f"  Success rate: {stats['success_rate']}%")
    print(f"  Average time: {stats['avg_time_ms']:.2f}ms")
    print()

    print("=" * 60)
    print("Example complete!")
    print("Check 'energy_report.html' for the HTML output")
    print("=" * 60)


if __name__ == "__main__":
    main()
