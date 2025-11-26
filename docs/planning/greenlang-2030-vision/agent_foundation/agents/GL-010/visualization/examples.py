"""
GL-010 EMISSIONWATCH - Visualization Examples

Comprehensive usage examples for the visualization engine.
Demonstrates all major features and chart types.

Author: GreenLang Team
Version: 1.0.0
"""

import json
from datetime import datetime, timedelta
from pathlib import Path


def example_compliance_dashboard():
    """
    Example: Generate a complete compliance dashboard.

    This example demonstrates:
    - Creating pollutant status data
    - Building a compliance dashboard
    - Generating multiple chart types
    - Exporting to HTML and JSON
    """
    print("=" * 60)
    print("Example 1: Compliance Dashboard")
    print("=" * 60)

    from compliance_dashboard import (
        ComplianceDashboard,
        ComplianceDashboardData,
        PollutantStatus,
        Violation,
        ComplianceStatus,
        ViolationType
    )

    # Define pollutant statuses
    pollutants = {
        "NOx": PollutantStatus(
            pollutant_id="NOx",
            pollutant_name="Nitrogen Oxides",
            current_value=145.5,
            unit="lb/hr",
            permit_limit=200.0,
            averaging_period="1-hour rolling",
            status=ComplianceStatus.COMPLIANT,
            margin_percent=27.25,
            trend="stable",
            last_updated="2024-01-15T14:30:00Z",
            data_quality=98.5
        ),
        "SO2": PollutantStatus(
            pollutant_id="SO2",
            pollutant_name="Sulfur Dioxide",
            current_value=92.5,
            unit="lb/hr",
            permit_limit=100.0,
            averaging_period="1-hour rolling",
            status=ComplianceStatus.WARNING,
            margin_percent=7.5,
            trend="increasing",
            last_updated="2024-01-15T14:30:00Z",
            data_quality=97.2
        ),
        "PM": PollutantStatus(
            pollutant_id="PM",
            pollutant_name="Particulate Matter",
            current_value=12.3,
            unit="lb/hr",
            permit_limit=25.0,
            averaging_period="6-hour rolling",
            status=ComplianceStatus.COMPLIANT,
            margin_percent=50.8,
            trend="decreasing",
            last_updated="2024-01-15T14:30:00Z",
            data_quality=95.0
        ),
        "CO": PollutantStatus(
            pollutant_id="CO",
            pollutant_name="Carbon Monoxide",
            current_value=350.0,
            unit="ppm",
            permit_limit=500.0,
            averaging_period="8-hour rolling",
            status=ComplianceStatus.COMPLIANT,
            margin_percent=30.0,
            trend="stable",
            last_updated="2024-01-15T14:30:00Z",
            data_quality=99.1
        )
    }

    # Define active violations
    violations = [
        Violation(
            violation_id="VIO-2024-001",
            violation_type=ViolationType.EMISSION_EXCEEDANCE,
            pollutant="SO2",
            start_time="2024-01-10T08:15:00Z",
            end_time="2024-01-10T09:45:00Z",
            duration_minutes=90,
            exceedance_value=115.3,
            permit_limit=100.0,
            exceedance_percent=15.3,
            severity="moderate",
            status="under_review",
            regulatory_action=None,
            root_cause="Fuel quality variation",
            corrective_action="Fuel supplier notified"
        )
    ]

    # Create dashboard data
    data = ComplianceDashboardData(
        timestamp="2024-01-15T14:30:00Z",
        facility_id="FAC-001",
        facility_name="GreenPower Plant Alpha",
        jurisdiction="California - SCAQMD",
        permit_number="SCAQMD-12345",
        pollutants=pollutants,
        overall_status=ComplianceStatus.WARNING,
        active_violations=violations,
        margin_to_limits={"NOx": 27.25, "SO2": 7.5, "PM": 50.8, "CO": 30.0},
        reporting_period="Q1 2024",
        data_completeness=97.5,
        next_report_due="2024-04-15",
        notes=["Fuel switch planned for Q2", "Stack test scheduled March 2024"]
    )

    # Create dashboard
    dashboard = ComplianceDashboard(data, color_blind_safe=False)

    # Generate individual charts
    print("\n1. Status Matrix Chart:")
    status_matrix = dashboard.generate_status_matrix()
    print(f"   Generated chart with {len(status_matrix['data'])} trace(s)")

    print("\n2. Gauge Charts:")
    for pollutant_id in pollutants.keys():
        gauge = dashboard.generate_gauge_chart(pollutant_id)
        print(f"   {pollutant_id}: Generated gauge chart")

    print("\n3. Violation Summary:")
    violation_summary = dashboard.generate_violation_summary()
    print(f"   Generated summary with {len(violation_summary['data'])} trace(s)")

    print("\n4. Margin Chart:")
    margin_chart = dashboard.generate_margin_chart()
    print(f"   Generated margin chart")

    # Export to JSON
    print("\n5. Exporting to Plotly JSON:")
    json_output = dashboard.to_plotly_json()
    print(f"   JSON output size: {len(json_output)} characters")

    # Export to HTML
    print("\n6. Exporting to HTML:")
    html_output = dashboard.to_html()
    print(f"   HTML output size: {len(html_output)} characters")

    print("\nDashboard generation complete!")
    return dashboard


def example_emissions_trend():
    """
    Example: Generate emissions trend visualizations.

    This example demonstrates:
    - Creating trend configuration
    - Setting up emission data points
    - Building hourly/daily/monthly trends
    - Anomaly detection and forecasting
    """
    print("\n" + "=" * 60)
    print("Example 2: Emissions Trend Analysis")
    print("=" * 60)

    from emissions_trends import (
        EmissionsTrendChart,
        EmissionsTrendDashboard,
        EmissionDataPoint,
        TrendConfig,
        TimeResolution,
        create_sample_trend_data
    )

    # Configure trend chart
    config = TrendConfig(
        pollutant="NOx",
        pollutant_name="Nitrogen Oxides",
        unit="lb/hr",
        permit_limit=200.0,
        warning_threshold=180.0,
        resolution=TimeResolution.HOURLY,
        show_rolling_average=True,
        rolling_window=24,
        show_forecast=True,
        forecast_periods=24,
        show_confidence_bands=True,
        confidence_level=0.95,
        highlight_anomalies=True,
        anomaly_threshold=2.0
    )

    # Create chart
    chart = EmissionsTrendChart(config)

    # Generate sample data (7 days of hourly data)
    sample_data = create_sample_trend_data(168)
    chart.set_data(sample_data)

    print(f"\n1. Data loaded: {len(sample_data)} data points")

    # Get statistics
    stats = chart.get_statistics()
    if stats:
        print(f"\n2. Statistics:")
        print(f"   Mean: {stats.mean_value:.2f} lb/hr")
        print(f"   Max: {stats.max_value:.2f} lb/hr")
        print(f"   Min: {stats.min_value:.2f} lb/hr")
        print(f"   95th Percentile: {stats.percentile_95:.2f} lb/hr")
        print(f"   Exceedances: {stats.exceedance_count}")
        print(f"   Trend: {stats.trend_direction.value}")

    # Get anomalies
    anomalies = chart.get_anomalies()
    print(f"\n3. Anomalies detected: {len(anomalies)}")
    for anom in anomalies[:3]:  # Show first 3
        print(f"   - {anom['timestamp']}: {anom['value']:.2f} (z-score: {anom['z_score']:.2f})")

    # Get forecast
    forecast = chart.get_forecast()
    if forecast and "values" in forecast:
        print(f"\n4. Forecast generated:")
        print(f"   Periods: {len(forecast['values'])}")
        print(f"   Model: {forecast['model']}")
        print(f"   Slope: {forecast['slope']:.4f}")

    # Build charts
    print("\n5. Building charts:")
    hourly_chart = chart.build_hourly_trend()
    print(f"   Hourly trend: {len(hourly_chart['data'])} traces")

    stats_panel = chart.build_statistics_panel()
    print(f"   Statistics panel: {len(stats_panel['data'])} indicators")

    # Create multi-pollutant dashboard
    print("\n6. Multi-pollutant dashboard:")
    pollutant_configs = [
        TrendConfig(
            pollutant="NOx",
            pollutant_name="Nitrogen Oxides",
            unit="lb/hr",
            permit_limit=200.0,
            warning_threshold=180.0,
            resolution=TimeResolution.HOURLY
        ),
        TrendConfig(
            pollutant="SO2",
            pollutant_name="Sulfur Dioxide",
            unit="lb/hr",
            permit_limit=100.0,
            warning_threshold=90.0,
            resolution=TimeResolution.HOURLY
        )
    ]

    trend_dashboard = EmissionsTrendDashboard(
        facility_name="GreenPower Plant Alpha",
        pollutants=pollutant_configs
    )

    # Set data for each pollutant
    for config in pollutant_configs:
        data = create_sample_trend_data(168)
        trend_dashboard.set_pollutant_data(config.pollutant, data)

    all_trends = trend_dashboard.generate_all_trends()
    print(f"   Generated {len(all_trends)} individual trend charts")

    combined = trend_dashboard.generate_combined_trend()
    print(f"   Combined chart: {len(combined['data'])} traces")

    print("\nTrend analysis complete!")
    return chart


def example_violation_timeline():
    """
    Example: Generate violation timeline visualizations.

    This example demonstrates:
    - Creating violation records
    - Building timeline charts
    - Status and severity analysis
    - Regulatory response tracking
    """
    print("\n" + "=" * 60)
    print("Example 3: Violation Timeline")
    print("=" * 60)

    from violation_timeline import (
        ViolationTimelineChart,
        ViolationRecord,
        ViolationSeverity,
        ViolationStatus,
        ViolationType,
        RegulatoryResponse,
        TimelineConfig,
        create_sample_violations
    )

    # Create sample violations
    violations = create_sample_violations(25)
    print(f"\n1. Created {len(violations)} sample violations")

    # Configure timeline
    config = TimelineConfig(
        title="Facility Violation History",
        show_duration_bars=True,
        show_severity_legend=True,
        group_by_pollutant=False,
        color_blind_safe=False
    )

    # Create timeline
    timeline = ViolationTimelineChart(violations, config)

    # Get summary statistics
    stats = timeline.get_summary_statistics()
    print(f"\n2. Summary Statistics:")
    print(f"   Total violations: {stats['total_violations']}")
    print(f"   Active violations: {stats['active_violations']}")
    print(f"   Total duration: {stats['total_duration_hours']:.1f} hours")
    print(f"   Average exceedance: {stats['avg_exceedance_percent']:.1f}%")

    print(f"\n   By severity:")
    for sev, count in stats['by_severity'].items():
        print(f"     {sev.title()}: {count}")

    print(f"\n   By status:")
    for status, count in stats['by_status'].items():
        print(f"     {status.replace('_', ' ').title()}: {count}")

    # Build charts
    print("\n3. Building charts:")
    gantt = timeline.build_gantt_timeline()
    print(f"   Gantt timeline: {len(gantt['data'])} traces")

    scatter = timeline.build_scatter_timeline()
    print(f"   Scatter timeline: {len(scatter['data'])} traces")

    status_breakdown = timeline.build_status_breakdown()
    print(f"   Status breakdown: {len(status_breakdown['data'])} traces")

    severity_dist = timeline.build_severity_distribution()
    print(f"   Severity distribution: {len(severity_dist['data'])} traces")

    monthly = timeline.build_monthly_trend()
    print(f"   Monthly trend: {len(monthly['data'])} traces")

    duration_analysis = timeline.build_duration_analysis()
    print(f"   Duration analysis: {len(duration_analysis['data'])} traces")

    regulatory = timeline.build_regulatory_timeline()
    print(f"   Regulatory timeline: {len(regulatory['data'])} traces")

    # Export for report
    print("\n4. Exporting for report:")
    report_data = timeline.export_for_report()
    print(f"   Report data keys: {list(report_data.keys())}")

    print("\nViolation timeline complete!")
    return timeline


def example_source_breakdown():
    """
    Example: Generate source breakdown visualizations.

    This example demonstrates:
    - Creating emission source data
    - Building pie/bar/treemap charts
    - Sankey flow diagrams
    - Fuel type analysis
    """
    print("\n" + "=" * 60)
    print("Example 4: Source Breakdown")
    print("=" * 60)

    from source_breakdown import (
        SourceBreakdownChart,
        EmissionSource,
        SourceType,
        FuelType,
        SourceBreakdownConfig,
        create_sample_sources
    )

    # Create sample sources
    sources = create_sample_sources(20)
    print(f"\n1. Created {len(sources)} emission sources")

    # Configure breakdown
    config = SourceBreakdownConfig(
        title="Facility Emissions by Source",
        show_percentages=True,
        min_percent_label=5.0,
        sort_by_value=True
    )

    # Create breakdown chart
    breakdown = SourceBreakdownChart(sources, config)

    # Get summary statistics
    stats = breakdown.get_summary_statistics()
    print(f"\n2. Summary Statistics:")
    print(f"   Total sources: {stats['total_sources']}")
    print(f"   Total emissions: {stats['total_emissions']:,.0f}")

    print(f"\n   By source type:")
    for st, data in stats['by_type'].items():
        print(f"     {st.replace('_', ' ').title()}: {data['count']} sources, {data['emissions']:,.0f} tons")

    print(f"\n   Top 5 sources:")
    for src in stats['top_sources'][:5]:
        print(f"     {src['name']}: {src['emissions']:,.0f} tons")

    # Build charts
    print("\n3. Building charts:")
    pie_chart = breakdown.build_pie_chart(group_by="source")
    print(f"   Pie chart (by source): {len(pie_chart['data'])} traces")

    pie_type = breakdown.build_pie_chart(group_by="type")
    print(f"   Pie chart (by type): {len(pie_type['data'])} traces")

    bar_chart = breakdown.build_bar_chart(horizontal=True)
    print(f"   Bar chart: {len(bar_chart['data'])} traces")

    stacked = breakdown.build_stacked_bar_chart()
    print(f"   Stacked bar: {len(stacked['data'])} traces")

    treemap = breakdown.build_treemap(hierarchy=["unit", "type", "source"])
    print(f"   Treemap: {len(treemap['data'])} traces")

    sankey = breakdown.build_sankey_diagram()
    print(f"   Sankey diagram: {len(sankey['data'])} traces")

    fuel = breakdown.build_fuel_breakdown()
    print(f"   Fuel breakdown: {len(fuel['data'])} traces")

    units = breakdown.build_process_unit_comparison()
    print(f"   Unit comparison: {len(units['data'])} traces")

    profile = breakdown.build_pollutant_profile()
    print(f"   Pollutant profile: {len(profile['data'])} traces")

    print("\nSource breakdown complete!")
    return breakdown


def example_regulatory_heatmap():
    """
    Example: Generate regulatory compliance heatmap.

    This example demonstrates:
    - Creating jurisdiction and compliance data
    - Building heatmaps with drill-down
    - Animated time-series heatmaps
    - Geographic visualization
    """
    print("\n" + "=" * 60)
    print("Example 5: Regulatory Heatmap")
    print("=" * 60)

    from regulatory_heatmap import (
        RegulatoryHeatmap,
        JurisdictionInfo,
        ComplianceCell,
        ComplianceLevel,
        HeatmapConfig,
        create_sample_heatmap_data
    )

    # Create sample data
    jurisdictions, pollutants, cells = create_sample_heatmap_data(
        num_jurisdictions=8,
        num_pollutants=6
    )

    print(f"\n1. Created data:")
    print(f"   Jurisdictions: {len(jurisdictions)}")
    print(f"   Pollutants: {len(pollutants)}")
    print(f"   Compliance cells: {len(cells)}")

    # Configure heatmap
    config = HeatmapConfig(
        title="Multi-Jurisdiction Compliance Status",
        color_blind_safe=False,
        show_values=True,
        show_margins=True
    )

    # Create heatmap
    heatmap = RegulatoryHeatmap(jurisdictions, pollutants, config)
    heatmap.set_compliance_data(cells)

    # Get summary statistics
    stats = heatmap.get_summary_statistics()
    print(f"\n2. Summary Statistics:")
    print(f"   Total cells: {stats['total_cells']}")
    print(f"   Overall compliance rate: {stats['overall_compliance_rate']:.1f}%")

    print(f"\n   By compliance level:")
    for level, count in stats['by_level'].items():
        print(f"     {level.title()}: {count}")

    print(f"\n   Worst cases:")
    for case in stats['worst_cases'][:3]:
        print(f"     {case['jurisdiction_id']} / {case['pollutant']}: {case['margin_percent']:.1f}%")

    # Build charts
    print("\n3. Building charts:")
    main_heatmap = heatmap.build_heatmap()
    print(f"   Main heatmap: {len(main_heatmap['data'])} traces")

    summary = heatmap.build_summary_indicators()
    print(f"   Summary indicators: {len(summary['data'])} indicators")

    # Jurisdiction detail
    first_jurisdiction = jurisdictions[0].jurisdiction_id
    detail = heatmap.build_jurisdiction_detail(first_jurisdiction)
    print(f"   Jurisdiction detail ({first_jurisdiction}): {len(detail['data'])} traces")

    # Pollutant comparison
    first_pollutant = pollutants[0]
    comparison = heatmap.build_pollutant_comparison(first_pollutant)
    print(f"   Pollutant comparison ({first_pollutant}): {len(comparison['data'])} traces")

    print("\nRegulatory heatmap complete!")
    return heatmap


def example_report_export():
    """
    Example: Export reports in multiple formats.

    This example demonstrates:
    - Creating a report exporter
    - Adding charts and tables
    - Exporting to PDF, HTML, JSON, Excel, and XML
    """
    print("\n" + "=" * 60)
    print("Example 6: Report Export")
    print("=" * 60)

    from export import (
        ReportExporter,
        ExportConfig,
        ExportFormat,
        TableData,
        create_sample_report
    )

    # Create configuration
    config = ExportConfig(
        title="Quarterly Emissions Compliance Report",
        subtitle="Q1 2024 - GreenPower Plant Alpha",
        facility_name="GreenPower Plant Alpha",
        facility_id="FAC-001",
        permit_number="SCAQMD-12345",
        reporting_period="January 1 - March 31, 2024",
        prepared_by="Environmental Compliance Team",
        prepared_date=datetime.now().strftime("%Y-%m-%d")
    )

    print(f"\n1. Report Configuration:")
    print(f"   Title: {config.title}")
    print(f"   Facility: {config.facility_name}")
    print(f"   Period: {config.reporting_period}")

    # Create exporter
    exporter = ReportExporter(config)

    # Add summary data
    exporter.set_summary({
        "total_emissions_tons": 15234.5,
        "active_violations": 2,
        "compliance_rate": 98.5,
        "data_availability": 99.2,
        "exceedance_hours": 4.5
    })

    # Add tables
    emissions_table = TableData(
        title="Emissions Summary by Pollutant",
        headers=["Pollutant", "Emissions (tons)", "Limit (tons)", "Margin (%)", "Status"],
        rows=[
            ["NOx", "1,234.5", "2,000.0", "38.3%", "Compliant"],
            ["SO2", "567.8", "1,000.0", "43.2%", "Compliant"],
            ["PM", "123.4", "250.0", "50.6%", "Compliant"],
            ["CO", "2,345.6", "3,000.0", "21.8%", "Compliant"],
            ["VOC", "456.7", "800.0", "42.9%", "Compliant"]
        ],
        footnotes=[
            "All values are tons per quarter",
            "Data availability: 99.2%",
            "Reporting period: January - March 2024"
        ]
    )
    exporter.add_table(emissions_table)

    violations_table = TableData(
        title="Violation Summary",
        headers=["ID", "Pollutant", "Date", "Duration", "Severity", "Status"],
        rows=[
            ["VIO-2024-001", "SO2", "2024-01-10", "90 min", "Moderate", "Resolved"],
            ["VIO-2024-002", "NOx", "2024-02-15", "45 min", "Minor", "Closed"]
        ]
    )
    exporter.add_table(violations_table)

    print(f"\n2. Content Added:")
    print(f"   Tables: 2")
    print(f"   Summary items: 5")

    # Add emissions data for XML export
    exporter.add_emissions_data([
        {
            "pollutant": "NOx",
            "value": 1234.5,
            "unit": "tons",
            "averaging_period": "quarterly",
            "permit_limit": 2000.0,
            "timestamp": "2024-03-31T23:59:59Z",
            "data_quality": 99.2
        },
        {
            "pollutant": "SO2",
            "value": 567.8,
            "unit": "tons",
            "averaging_period": "quarterly",
            "permit_limit": 1000.0,
            "timestamp": "2024-03-31T23:59:59Z",
            "data_quality": 98.5
        }
    ])

    exporter.add_violations_data([
        {
            "violation_id": "VIO-2024-001",
            "violation_type": "emission_exceedance",
            "pollutant": "SO2",
            "start_time": "2024-01-10T08:15:00Z",
            "end_time": "2024-01-10T09:45:00Z",
            "duration_minutes": 90,
            "exceedance_value": 115.3,
            "permit_limit": 100.0,
            "severity": "moderate",
            "root_cause": "Fuel quality variation",
            "corrective_action": "Fuel supplier notified"
        }
    ])

    # Export to different formats
    print("\n3. Export Formats:")

    # JSON export
    json_bytes = exporter.export_bytes(ExportFormat.JSON)
    print(f"   JSON: {len(json_bytes):,} bytes")

    # HTML export
    html_bytes = exporter.export_bytes(ExportFormat.HTML)
    print(f"   HTML: {len(html_bytes):,} bytes")

    # PDF-ready HTML
    pdf_bytes = exporter.export_bytes(ExportFormat.PDF)
    print(f"   PDF (HTML): {len(pdf_bytes):,} bytes")

    # Excel (JSON structure)
    excel_bytes = exporter.export_bytes(ExportFormat.EXCEL)
    print(f"   Excel (JSON): {len(excel_bytes):,} bytes")

    # EPA CEDRI XML
    xml_bytes = exporter.export_bytes(ExportFormat.XML)
    print(f"   XML (CEDRI): {len(xml_bytes):,} bytes")

    # Preview XML output
    print("\n4. XML Preview (first 500 chars):")
    print(xml_bytes.decode('utf-8')[:500])

    print("\nReport export complete!")
    return exporter


def example_complete_workflow():
    """
    Example: Complete visualization workflow.

    This example demonstrates:
    - Loading data from multiple sources
    - Creating all visualization types
    - Generating a comprehensive report
    """
    print("\n" + "=" * 60)
    print("Example 7: Complete Workflow")
    print("=" * 60)

    from compliance_dashboard import (
        ComplianceDashboard,
        create_sample_dashboard_data
    )
    from emissions_trends import (
        EmissionsTrendChart,
        TrendConfig,
        TimeResolution,
        create_sample_trend_data
    )
    from violation_timeline import (
        ViolationTimelineChart,
        TimelineConfig,
        create_sample_violations
    )
    from source_breakdown import (
        SourceBreakdownChart,
        SourceBreakdownConfig,
        create_sample_sources
    )
    from regulatory_heatmap import (
        RegulatoryHeatmap,
        HeatmapConfig,
        create_sample_heatmap_data
    )
    from export import (
        ReportExporter,
        ExportConfig,
        ExportFormat,
        TableData
    )

    print("\n1. Creating visualizations...")

    # Dashboard
    dashboard_data = create_sample_dashboard_data()
    dashboard = ComplianceDashboard(dashboard_data)
    print("   - Compliance dashboard created")

    # Trend charts
    trend_config = TrendConfig(
        pollutant="NOx",
        pollutant_name="Nitrogen Oxides",
        unit="lb/hr",
        permit_limit=200.0,
        warning_threshold=180.0,
        resolution=TimeResolution.HOURLY,
        show_rolling_average=True,
        show_forecast=True
    )
    trend_chart = EmissionsTrendChart(trend_config)
    trend_chart.set_data(create_sample_trend_data(168))
    print("   - Emissions trends created")

    # Violation timeline
    violations = create_sample_violations(20)
    timeline = ViolationTimelineChart(
        violations,
        TimelineConfig(title="Violation History")
    )
    print("   - Violation timeline created")

    # Source breakdown
    sources = create_sample_sources(15)
    breakdown = SourceBreakdownChart(
        sources,
        SourceBreakdownConfig(title="Emissions Sources")
    )
    print("   - Source breakdown created")

    # Regulatory heatmap
    jurisdictions, pollutants, cells = create_sample_heatmap_data(6, 5)
    heatmap = RegulatoryHeatmap(
        jurisdictions,
        pollutants,
        HeatmapConfig(title="Compliance Heatmap")
    )
    heatmap.set_compliance_data(cells)
    print("   - Regulatory heatmap created")

    # Create comprehensive report
    print("\n2. Building comprehensive report...")

    report_config = ExportConfig(
        title="Comprehensive Emissions Compliance Report",
        facility_name="GreenPower Plant Alpha",
        facility_id="FAC-001",
        permit_number="SCAQMD-12345",
        reporting_period="Q1 2024"
    )

    exporter = ReportExporter(report_config)

    # Add all charts
    exporter.add_chart(dashboard.generate_status_matrix(), "Compliance Status Matrix")
    exporter.add_chart(trend_chart.build_hourly_trend(), "NOx Emissions Trend")
    exporter.add_chart(timeline.build_gantt_timeline(), "Violation Timeline")
    exporter.add_chart(breakdown.build_sankey_diagram(), "Emissions Flow")
    exporter.add_chart(heatmap.build_heatmap(), "Multi-Jurisdiction Compliance")

    # Add summary from dashboard
    exporter.set_summary({
        "facility": dashboard_data.facility_name,
        "status": dashboard_data.overall_status.value,
        "violations": len(dashboard_data.active_violations),
        "data_completeness": dashboard_data.data_completeness
    })

    # Add statistics table
    stats = trend_chart.get_statistics()
    if stats:
        exporter.add_table(TableData(
            title="Emissions Statistics",
            headers=["Metric", "Value"],
            rows=[
                ["Mean", f"{stats.mean_value:.2f} lb/hr"],
                ["Maximum", f"{stats.max_value:.2f} lb/hr"],
                ["95th Percentile", f"{stats.percentile_95:.2f} lb/hr"],
                ["Exceedances", str(stats.exceedance_count)],
                ["Trend", stats.trend_direction.value.title()]
            ]
        ))

    print(f"   - Added {len(exporter._charts)} charts")
    print(f"   - Added {len(exporter._tables)} tables")

    # Export
    print("\n3. Exporting report...")
    json_output = exporter.export_bytes(ExportFormat.JSON)
    print(f"   JSON size: {len(json_output):,} bytes")

    html_output = exporter.export_bytes(ExportFormat.HTML)
    print(f"   HTML size: {len(html_output):,} bytes")

    print("\nComplete workflow finished!")
    return exporter


def run_all_examples():
    """Run all visualization examples."""
    print("\n" + "#" * 60)
    print("# GL-010 EMISSIONWATCH Visualization Examples")
    print("#" * 60)

    examples = [
        ("Compliance Dashboard", example_compliance_dashboard),
        ("Emissions Trend", example_emissions_trend),
        ("Violation Timeline", example_violation_timeline),
        ("Source Breakdown", example_source_breakdown),
        ("Regulatory Heatmap", example_regulatory_heatmap),
        ("Report Export", example_report_export),
        ("Complete Workflow", example_complete_workflow)
    ]

    results = {}

    for name, func in examples:
        try:
            result = func()
            results[name] = "SUCCESS"
        except Exception as e:
            results[name] = f"ERROR: {str(e)}"
            print(f"\nError in {name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("EXAMPLE RESULTS SUMMARY")
    print("=" * 60)

    for name, status in results.items():
        status_icon = "[OK]" if status == "SUCCESS" else "[FAIL]"
        print(f"  {status_icon} {name}: {status}")

    success_count = sum(1 for s in results.values() if s == "SUCCESS")
    print(f"\nCompleted: {success_count}/{len(examples)} examples successful")


if __name__ == "__main__":
    run_all_examples()
