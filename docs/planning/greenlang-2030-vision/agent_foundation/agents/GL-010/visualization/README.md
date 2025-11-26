# GL-010 EMISSIONWATCH Visualization Engine

Comprehensive visualization module for emissions compliance monitoring as part of the EmissionsComplianceAgent (GL-010).

## Overview

The visualization engine provides interactive dashboards, trend analysis, violation tracking, source breakdowns, regulatory heatmaps, and export functionality for emissions compliance monitoring.

## Installation

```bash
# Install required dependencies
pip install plotly pandas numpy

# Optional dependencies for export functionality
pip install weasyprint  # PDF generation
pip install openpyxl    # Excel export
pip install kaleido     # Static image export
```

## Quick Start

### Basic Usage

```python
from visualization import (
    ComplianceDashboard,
    create_sample_dashboard_data
)

# Create sample data
data = create_sample_dashboard_data()

# Generate dashboard
dashboard = ComplianceDashboard(data)

# Export to HTML
html_content = dashboard.to_html()
with open("dashboard.html", "w") as f:
    f.write(html_content)

# Export to Plotly JSON
json_content = dashboard.to_plotly_json()
```

### Convenience Functions

```python
from visualization import (
    create_compliance_dashboard,
    create_trend_chart,
    create_violation_timeline,
    create_source_breakdown,
    create_regulatory_heatmap,
    create_report_exporter
)

# Quick dashboard creation
dashboard = create_compliance_dashboard(data, color_blind_safe=True)

# Quick trend chart
chart = create_trend_chart(
    pollutant="NOx",
    pollutant_name="Nitrogen Oxides",
    unit="lb/hr",
    permit_limit=200.0,
    show_forecast=True
)
```

## Chart Types

### 1. Compliance Dashboard

Main compliance status visualization with multiple chart types.

```python
from visualization import (
    ComplianceDashboard,
    ComplianceDashboardData,
    PollutantStatus,
    ComplianceStatus
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
    )
}

# Create dashboard data
data = ComplianceDashboardData(
    timestamp="2024-01-15T14:30:00Z",
    facility_id="FAC-001",
    facility_name="GreenPower Plant",
    jurisdiction="California - SCAQMD",
    permit_number="SCAQMD-12345",
    pollutants=pollutants,
    overall_status=ComplianceStatus.COMPLIANT,
    active_violations=[],
    margin_to_limits={"NOx": 27.25},
    reporting_period="Q1 2024",
    data_completeness=97.5
)

# Generate dashboard
dashboard = ComplianceDashboard(data, color_blind_safe=False)

# Get individual charts
status_matrix = dashboard.generate_status_matrix()
gauge_nox = dashboard.generate_gauge_chart("NOx")
violation_summary = dashboard.generate_violation_summary()
margin_chart = dashboard.generate_margin_chart()
```

### 2. Emissions Trends

Time-series visualization with forecasting and anomaly detection.

```python
from visualization import (
    EmissionsTrendChart,
    EmissionDataPoint,
    TrendConfig,
    TimeResolution
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
    highlight_anomalies=True
)

# Create chart
chart = EmissionsTrendChart(config)

# Add data
data_points = [
    EmissionDataPoint(
        timestamp="2024-01-15T00:00:00Z",
        value=145.5,
        unit="lb/hr",
        data_quality=98.5
    ),
    # ... more data points
]
chart.set_data(data_points)

# Generate charts
hourly_trend = chart.build_hourly_trend()
statistics = chart.get_statistics()
anomalies = chart.get_anomalies()
forecast = chart.get_forecast()
```

### 3. Violation Timeline

Historical violation tracking and analysis.

```python
from visualization import (
    ViolationTimelineChart,
    ViolationRecord,
    ViolationSeverity,
    ViolationStatus,
    TimelineConfig
)
from visualization.violation_timeline import ViolationType

# Create violation records
violations = [
    ViolationRecord(
        violation_id="VIO-2024-001",
        violation_type=ViolationType.EMISSION_EXCEEDANCE,
        pollutant="SO2",
        pollutant_name="Sulfur Dioxide",
        start_time="2024-01-10T08:15:00Z",
        end_time="2024-01-10T09:45:00Z",
        duration_minutes=90,
        exceedance_value=115.3,
        permit_limit=100.0,
        unit="lb/hr",
        exceedance_percent=15.3,
        severity=ViolationSeverity.MODERATE,
        status=ViolationStatus.UNDER_REVIEW,
        source_unit="Boiler-1",
        affected_limit="1-hour average"
    )
]

# Create timeline
config = TimelineConfig(
    title="Facility Violation Timeline",
    show_duration_bars=True,
    group_by_pollutant=False
)
timeline = ViolationTimelineChart(violations, config)

# Generate charts
gantt_timeline = timeline.build_gantt_timeline()
scatter_timeline = timeline.build_scatter_timeline()
status_breakdown = timeline.build_status_breakdown()
monthly_trend = timeline.build_monthly_trend()
```

### 4. Source Breakdown

Emissions source analysis with multiple visualization types.

```python
from visualization import (
    SourceBreakdownChart,
    EmissionSource,
    SourceType,
    FuelType,
    SourceBreakdownConfig
)

# Define emission sources
sources = [
    EmissionSource(
        source_id="SRC-001",
        source_name="Main Boiler",
        source_type=SourceType.COMBUSTION,
        unit_id="UNIT-1",
        unit_name="Boiler Unit 1",
        fuel_type=FuelType.NATURAL_GAS,
        emissions_by_pollutant={"NOx": 500, "SO2": 200, "CO": 150},
        total_emissions=850,
        unit="tons/year",
        operating_hours=8000
    )
]

# Create breakdown chart
config = SourceBreakdownConfig(
    title="Facility Emissions Breakdown",
    show_percentages=True
)
breakdown = SourceBreakdownChart(sources, config)

# Generate charts
pie_chart = breakdown.build_pie_chart(group_by="source")
bar_chart = breakdown.build_bar_chart(horizontal=True)
stacked_chart = breakdown.build_stacked_bar_chart()
treemap = breakdown.build_treemap()
sankey = breakdown.build_sankey_diagram()
```

### 5. Regulatory Heatmap

Multi-jurisdiction compliance visualization.

```python
from visualization import (
    RegulatoryHeatmap,
    JurisdictionInfo,
    ComplianceCell,
    ComplianceLevel,
    HeatmapConfig
)

# Define jurisdictions
jurisdictions = [
    JurisdictionInfo(
        jurisdiction_id="CA_SCAQMD",
        name="South Coast AQMD",
        abbreviation="SCAQMD",
        country="USA",
        region="California"
    )
]

# Define compliance data
cells = [
    ComplianceCell(
        jurisdiction_id="CA_SCAQMD",
        pollutant="NOx",
        current_value=145.5,
        limit_value=200.0,
        unit="lb/hr",
        margin_percent=27.25,
        compliance_level=ComplianceLevel.GOOD,
        timestamp="2024-01-15T14:30:00Z",
        data_quality=98.5
    )
]

# Create heatmap
config = HeatmapConfig(
    title="Multi-State Compliance",
    animate=False
)
heatmap = RegulatoryHeatmap(
    jurisdictions=jurisdictions,
    pollutants=["NOx", "SO2", "PM"],
    config=config
)
heatmap.set_compliance_data(cells)

# Generate charts
main_heatmap = heatmap.build_heatmap()
summary = heatmap.build_summary_indicators()
jurisdiction_detail = heatmap.build_jurisdiction_detail("CA_SCAQMD")
```

## Export Functionality

### PDF Export

```python
from visualization import (
    ReportExporter,
    ExportConfig,
    ExportFormat,
    TableData
)

# Configure export
config = ExportConfig(
    title="Emissions Compliance Report",
    subtitle="Q1 2024 Quarterly Report",
    facility_name="GreenPower Plant Alpha",
    facility_id="FAC-001",
    permit_number="SCAQMD-12345",
    reporting_period="January - March 2024"
)

# Create exporter
exporter = ReportExporter(config)

# Add content
exporter.add_chart(dashboard.generate_status_matrix(), "Compliance Status")
exporter.add_table(TableData(
    title="Emissions Summary",
    headers=["Pollutant", "Emissions", "Limit", "Status"],
    rows=[["NOx", "145.5 lb/hr", "200 lb/hr", "Compliant"]]
))
exporter.set_summary({
    "total_emissions": 15234.5,
    "violations": 2,
    "compliance_rate": 98.5
})

# Export to PDF (generates print-ready HTML)
exporter.export("report.pdf", ExportFormat.PDF)
```

### Excel Export

```python
# Export to Excel
exporter.export("report.xlsx", ExportFormat.EXCEL)
```

### EPA CEDRI XML Export

```python
# Add emissions data
exporter.add_emissions_data([
    {
        "pollutant": "NOx",
        "value": 1234.5,
        "unit": "tons",
        "permit_limit": 2000.0,
        "timestamp": "2024-03-31T23:59:59Z"
    }
])

# Export to EPA CEDRI XML format
exporter.export("submission.xml", ExportFormat.XML)
```

### Multiple Formats

```python
# Export to all formats
results = exporter.export_all(
    "report",
    formats=[
        ExportFormat.PDF,
        ExportFormat.HTML,
        ExportFormat.JSON,
        ExportFormat.EXCEL,
        ExportFormat.XML
    ]
)
```

## Customization

### Color Schemes

All visualizations support color-blind safe palettes:

```python
# Enable color-blind safe mode
dashboard = ComplianceDashboard(data, color_blind_safe=True)
chart = EmissionsTrendChart(config)
config.color_blind_safe = True
```

### Standard Colors

| Status | Standard | Color-Blind Safe |
|--------|----------|------------------|
| Compliant | #2ECC71 | #009E73 |
| Warning | #F39C12 | #E69F00 |
| Violation | #E74C3C | #D55E00 |
| Unknown | #95A5A6 | #999999 |

### Responsive Design

All charts are configured for responsive layouts:

```python
# Charts automatically resize
# HTML dashboards include responsive CSS
# Plotly charts use responsive: true
```

## Integration Examples

### Web Application (React)

```typescript
import Plot from 'react-plotly.js';

const ComplianceDashboard: React.FC = () => {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    // Fetch from API
    fetch('/api/compliance/dashboard')
      .then(res => res.json())
      .then(data => setChartData(data));
  }, []);

  if (!chartData) return <Loading />;

  return (
    <Plot
      data={chartData.data}
      layout={chartData.layout}
      config={chartData.config}
    />
  );
};
```

### API Endpoint (FastAPI)

```python
from fastapi import FastAPI
from visualization import ComplianceDashboard

app = FastAPI()

@app.get("/api/compliance/dashboard")
async def get_dashboard():
    data = get_compliance_data()  # Your data source
    dashboard = ComplianceDashboard(data)
    return dashboard.build_status_matrix()
```

### Jupyter Notebook

```python
import plotly.io as pio
from visualization import create_sample_dashboard_data, ComplianceDashboard

# Create dashboard
data = create_sample_dashboard_data()
dashboard = ComplianceDashboard(data)

# Display inline
chart = dashboard.generate_status_matrix()
pio.show(chart)
```

## API Reference

### ComplianceDashboard

| Method | Returns | Description |
|--------|---------|-------------|
| `generate_status_matrix()` | Dict | Multi-pollutant status matrix |
| `generate_gauge_chart(pollutant)` | Dict | Gauge chart for pollutant |
| `generate_all_gauges()` | List[Dict] | All pollutant gauges |
| `generate_trend_chart(pollutant, data)` | Dict | Trend chart with data |
| `generate_violation_summary()` | Dict | Violation summary chart |
| `generate_margin_chart()` | Dict | Margin to limits chart |
| `to_plotly_json()` | str | JSON export |
| `to_html()` | str | Standalone HTML |
| `to_d3_json()` | str | D3.js compatible JSON |

### EmissionsTrendChart

| Method | Returns | Description |
|--------|---------|-------------|
| `set_data(data)` | None | Set emission data |
| `build_hourly_trend()` | Dict | Hourly trend chart |
| `build_daily_summary(data)` | Dict | Daily summary chart |
| `build_monthly_comparison(data)` | Dict | Monthly comparison |
| `build_annual_summary(data)` | Dict | Annual summary |
| `build_statistics_panel()` | Dict | Statistics indicators |
| `get_statistics()` | TrendStatistics | Calculated statistics |
| `get_anomalies()` | List[Dict] | Detected anomalies |
| `get_forecast()` | Dict | Forecast data |

### ViolationTimelineChart

| Method | Returns | Description |
|--------|---------|-------------|
| `build_gantt_timeline()` | Dict | Gantt-style timeline |
| `build_scatter_timeline()` | Dict | Scatter plot timeline |
| `build_status_breakdown()` | Dict | Status pie chart |
| `build_severity_distribution()` | Dict | Severity bar chart |
| `build_monthly_trend()` | Dict | Monthly trend |
| `get_summary_statistics()` | Dict | Summary statistics |
| `export_for_report()` | Dict | Report-ready data |

### SourceBreakdownChart

| Method | Returns | Description |
|--------|---------|-------------|
| `build_pie_chart(group_by)` | Dict | Pie/donut chart |
| `build_bar_chart(horizontal)` | Dict | Bar chart |
| `build_stacked_bar_chart()` | Dict | Stacked bar chart |
| `build_treemap(hierarchy)` | Dict | Hierarchical treemap |
| `build_sankey_diagram()` | Dict | Sankey flow diagram |
| `build_fuel_breakdown()` | Dict | Fuel type breakdown |
| `build_pollutant_profile()` | Dict | Radar/spider chart |

### RegulatoryHeatmap

| Method | Returns | Description |
|--------|---------|-------------|
| `set_compliance_data(data)` | None | Set compliance cells |
| `add_time_frame(ts, data)` | None | Add animation frame |
| `build_heatmap()` | Dict | Main heatmap |
| `build_animated_heatmap()` | Dict | Animated heatmap |
| `build_jurisdiction_detail(id)` | Dict | Jurisdiction detail |
| `build_pollutant_comparison(p)` | Dict | Pollutant comparison |
| `build_summary_indicators()` | Dict | Summary indicators |
| `build_geographic_map(coords)` | Dict | Geographic map |

### ReportExporter

| Method | Returns | Description |
|--------|---------|-------------|
| `add_chart(data, title)` | self | Add chart |
| `add_table(table)` | self | Add table |
| `set_summary(summary)` | self | Set summary |
| `add_emissions_data(data)` | self | Add emissions (XML) |
| `export(path, format)` | str | Export to file |
| `export_bytes(format)` | bytes | Export to bytes |
| `export_all(base, formats)` | Dict | Export all formats |

## Best Practices

1. **Performance**: Use `limit` parameters for large datasets
2. **Accessibility**: Enable `color_blind_safe` for public dashboards
3. **Responsiveness**: All charts auto-resize; use percentage-based containers
4. **Caching**: Cache generated charts when data doesn't change frequently
5. **Error Handling**: Always check for empty data before chart generation

## Support

For issues and feature requests, please contact the GreenLang development team or create an issue in the repository.

## License

MIT License - See LICENSE file for details.
