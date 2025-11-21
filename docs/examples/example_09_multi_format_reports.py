# -*- coding: utf-8 -*-
"""
Example 09: Multi-Format Report Generation

This example demonstrates generating reports in multiple formats.
You'll learn:
- How to generate Markdown reports
- How to generate HTML reports
- How to generate JSON reports
- How to customize report styling and layout
"""

from greenlang.agents import BaseReporter, ReporterConfig, ReportSection
from typing import Dict, Any, List
from pathlib import Path


class ComprehensiveEmissionsReporter(BaseReporter):
    """
    Generate comprehensive emissions reports in multiple formats.

    This agent demonstrates:
    - Multi-format output (Markdown, HTML, JSON)
    - Custom styling
    - Rich visualizations
    - Professional report layouts
    """

    def __init__(self, output_format='markdown'):
        config = ReporterConfig(
            name="Comprehensive Emissions Report",
            description="Detailed carbon emissions analysis and reporting",
            output_format=output_format,
            include_summary=True,
            include_details=True
        )
        super().__init__(config)

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate emissions data across multiple buildings.

        Args:
            input_data: Must contain 'buildings' list

        Returns:
            Aggregated statistics
        """
        buildings = input_data['buildings']

        # Calculate totals
        total_emissions = sum(b['emissions_tons'] for b in buildings)
        total_area = sum(b['area_sqft'] for b in buildings)

        # Calculate averages
        avg_emissions = total_emissions / len(buildings) if buildings else 0
        avg_intensity = (total_emissions * 1000) / total_area if total_area > 0 else 0

        # Find extremes
        highest_emitter = max(buildings, key=lambda b: b['emissions_tons']) if buildings else None
        lowest_emitter = min(buildings, key=lambda b: b['emissions_tons']) if buildings else None

        # Calculate by fuel type
        fuel_breakdown = {}
        for building in buildings:
            for fuel_type, emissions in building.get('breakdown', {}).items():
                fuel_breakdown[fuel_type] = fuel_breakdown.get(fuel_type, 0) + emissions

        return {
            'num_buildings': len(buildings),
            'total_emissions_tons': round(total_emissions, 2),
            'total_area_sqft': total_area,
            'avg_emissions_tons': round(avg_emissions, 2),
            'avg_intensity_kg_per_sqft': round(avg_intensity, 4),
            'highest_emitter': highest_emitter['name'] if highest_emitter else 'N/A',
            'highest_emissions': highest_emitter['emissions_tons'] if highest_emitter else 0,
            'lowest_emitter': lowest_emitter['name'] if lowest_emitter else 'N/A',
            'lowest_emissions': lowest_emitter['emissions_tons'] if lowest_emitter else 0,
            'fuel_breakdown': fuel_breakdown,
            'report_date': input_data.get('report_date', '2025-01-15')
        }

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        """
        Build comprehensive report sections.

        Args:
            aggregated_data: Aggregated data

        Returns:
            List of report sections
        """
        sections = []

        # Section 1: Executive Summary
        exec_summary = f"""
This report analyzes carbon emissions across **{aggregated_data['num_buildings']} buildings**
covering a total area of **{aggregated_data['total_area_sqft']:,} square feet**.

**Key Findings:**
- Total emissions: **{aggregated_data['total_emissions_tons']:,.2f} metric tons CO2e**
- Average intensity: **{aggregated_data['avg_intensity_kg_per_sqft']:.4f} kg CO2e per sqft**
- Highest emitter: **{aggregated_data['highest_emitter']}**
  ({aggregated_data['highest_emissions']:.2f} tons)
- Lowest emitter: **{aggregated_data['lowest_emitter']}**
  ({aggregated_data['lowest_emissions']:.2f} tons)
        """.strip()

        sections.append(ReportSection(
            title="Executive Summary",
            content=exec_summary,
            level=2,
            section_type="text"
        ))

        # Section 2: Emissions Overview (Table)
        overview_table = [
            {'Metric': 'Total Buildings', 'Value': f"{aggregated_data['num_buildings']:,}"},
            {'Metric': 'Total Floor Area', 'Value': f"{aggregated_data['total_area_sqft']:,} sqft"},
            {'Metric': 'Total Emissions', 'Value': f"{aggregated_data['total_emissions_tons']:,.2f} tons CO2e"},
            {'Metric': 'Average Emissions', 'Value': f"{aggregated_data['avg_emissions_tons']:.2f} tons CO2e per building"},
            {'Metric': 'Carbon Intensity', 'Value': f"{aggregated_data['avg_intensity_kg_per_sqft']:.4f} kg CO2e per sqft"},
        ]

        sections.append(ReportSection(
            title="Emissions Overview",
            content=overview_table,
            level=2,
            section_type="table"
        ))

        # Section 3: Fuel Type Breakdown
        if aggregated_data['fuel_breakdown']:
            fuel_table = []
            total = sum(aggregated_data['fuel_breakdown'].values())

            for fuel_type, emissions in sorted(
                aggregated_data['fuel_breakdown'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (emissions / total * 100) if total > 0 else 0
                fuel_table.append({
                    'Fuel Type': fuel_type.replace('_', ' ').title(),
                    'Emissions (tons CO2e)': f"{emissions:,.2f}",
                    'Percentage': f"{percentage:.1f}%"
                })

            sections.append(ReportSection(
                title="Emissions by Fuel Type",
                content=fuel_table,
                level=2,
                section_type="table"
            ))

        # Section 4: Performance Benchmarks
        benchmark_content = f"""
**Performance Against Industry Benchmarks:**

Portfolio Average: **{aggregated_data['avg_intensity_kg_per_sqft']:.4f} kg CO2e/sqft**

Industry Benchmarks:
- Office Buildings (Best Practice): 0.0050 kg CO2e/sqft
- Office Buildings (Average): 0.0100 kg CO2e/sqft
- Office Buildings (Poor): 0.0150 kg CO2e/sqft

**Assessment:** {'Excellent' if aggregated_data['avg_intensity_kg_per_sqft'] < 0.0050 else
                  'Good' if aggregated_data['avg_intensity_kg_per_sqft'] < 0.0100 else
                  'Needs Improvement'}
        """.strip()

        sections.append(ReportSection(
            title="Performance Benchmarking",
            content=benchmark_content,
            level=2,
            section_type="text"
        ))

        # Section 5: Recommendations
        intensity = aggregated_data['avg_intensity_kg_per_sqft']

        if intensity > 0.0150:
            recommendations = [
                "**Critical**: Immediate energy audit required for all buildings",
                "Implement building-wide LED retrofit program (estimated 25-30% reduction)",
                "Upgrade HVAC systems to high-efficiency models",
                "Install building automation systems for optimal control",
                "Consider renewable energy sources (solar PV, wind)",
                "Implement employee engagement program for energy conservation"
            ]
        elif intensity > 0.0100:
            recommendations = [
                "**Moderate Priority**: Targeted improvements needed",
                "Focus on highest-emitting buildings first",
                "Implement smart thermostat controls",
                "Optimize HVAC schedules based on occupancy",
                "Improve building envelope insulation",
                "Consider renewable energy for offset"
            ]
        else:
            recommendations = [
                "**Excellent Performance**: Continue current practices",
                "Monitor for performance degradation",
                "Share best practices across portfolio",
                "Consider carbon-neutral certification",
                "Investigate remaining reduction opportunities"
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
    print("Example 09: Multi-Format Report Generation")
    print("=" * 60)
    print()

    # Sample data: emissions from multiple buildings
    buildings_data = {
        'report_date': '2025-01-15',
        'buildings': [
            {
                'name': 'Main Office',
                'area_sqft': 25000,
                'emissions_tons': 125.5,
                'breakdown': {'electricity': 85.5, 'natural_gas': 40.0}
            },
            {
                'name': 'Warehouse A',
                'area_sqft': 50000,
                'emissions_tons': 285.2,
                'breakdown': {'electricity': 250.2, 'natural_gas': 35.0}
            },
            {
                'name': 'Retail Store',
                'area_sqft': 15000,
                'emissions_tons': 68.8,
                'breakdown': {'electricity': 58.8, 'natural_gas': 10.0}
            },
            {
                'name': 'Distribution Center',
                'area_sqft': 75000,
                'emissions_tons': 420.5,
                'breakdown': {'electricity': 350.5, 'diesel': 70.0}
            },
        ]
    }

    # Create output directory
    output_dir = Path("reports_output")
    output_dir.mkdir(exist_ok=True)

    # Example 1: Markdown Report
    print("Test 1: Generate Markdown Report")
    print("-" * 40)

    reporter_md = ComprehensiveEmissionsReporter(output_format='markdown')
    result_md = reporter_md.run(buildings_data)

    if result_md.success:
        print(f"✓ Markdown report generated")
        print(f"  Sections: {result_md.data['sections_count']}")

        # Save to file
        output_file = output_dir / "emissions_report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_md.data['report'])

        print(f"  Saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")

        # Show preview
        print(f"\n  Preview (first 500 characters):")
        print("  " + "-" * 56)
        preview = result_md.data['report'][:500].replace('\n', '\n  ')
        print(f"  {preview}...")
        print("  " + "-" * 56)
    else:
        print(f"✗ Failed: {result_md.error}")
    print()

    # Example 2: HTML Report
    print("Test 2: Generate HTML Report")
    print("-" * 40)

    reporter_html = ComprehensiveEmissionsReporter(output_format='html')
    result_html = reporter_html.run(buildings_data)

    if result_html.success:
        print(f"✓ HTML report generated")

        # Save to file
        output_file = output_dir / "emissions_report.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_html.data['report'])

        print(f"  Saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")
        print(f"  Open in browser to view formatted report")
    else:
        print(f"✗ Failed: {result_html.error}")
    print()

    # Example 3: JSON Report
    print("Test 3: Generate JSON Report")
    print("-" * 40)

    reporter_json = ComprehensiveEmissionsReporter(output_format='json')
    result_json = reporter_json.run(buildings_data)

    if result_json.success:
        print(f"✓ JSON report generated")

        # Save to file
        output_file = output_dir / "emissions_report.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_json.data['report'])

        print(f"  Saved to: {output_file}")
        print(f"  File size: {output_file.stat().st_size} bytes")

        # Parse and show structure
        import json
        report_data = json.loads(result_json.data['report'])
        print(f"\n  Report Structure:")
        print(f"    Report name: {report_data['report_name']}")
        print(f"    Sections: {len(report_data['sections'])}")
        print(f"    Generated: {report_data['generated_at']}")

        print(f"\n  Section Titles:")
        for section in report_data['sections']:
            print(f"    • {section['title']}")
    else:
        print(f"✗ Failed: {result_json.error}")
    print()

    # Example 4: Format Comparison
    print("Test 4: Format Comparison")
    print("-" * 40)

    formats = ['markdown', 'html', 'json']
    file_sizes = {}

    for fmt in formats:
        file_path = output_dir / f"emissions_report.{fmt if fmt != 'markdown' else 'md'}"
        if file_path.exists():
            file_sizes[fmt] = file_path.stat().st_size

    print(f"  File Sizes:")
    for fmt, size in file_sizes.items():
        print(f"    {fmt.upper():<12} {size:>8,} bytes")

    print(f"\n  Format Recommendations:")
    print(f"    Markdown:  Best for version control, documentation, GitHub")
    print(f"    HTML:      Best for sharing, presentations, stakeholders")
    print(f"    JSON:      Best for APIs, data processing, archival")
    print()

    # Statistics
    print("Reporter Statistics:")
    print("-" * 40)
    stats = reporter_md.get_stats()
    print(f"  Total reports generated: {stats['executions']}")
    print(f"  Success rate: {stats['success_rate']}%")
    print(f"  Average generation time: {stats['avg_time_ms']:.2f}ms")
    print()

    print("=" * 60)
    print("Example complete!")
    print(f"Check '{output_dir}' directory for generated reports:")
    print(f"  • emissions_report.md   (Markdown)")
    print(f"  • emissions_report.html (HTML - open in browser)")
    print(f"  • emissions_report.json (JSON)")
    print("=" * 60)


if __name__ == "__main__":
    main()
