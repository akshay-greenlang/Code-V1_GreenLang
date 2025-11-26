# GL-009 THERMALIQ Visualization Module

Comprehensive energy flow visualization toolkit for thermal efficiency analysis. Generates interactive Plotly-compatible charts for Sankey diagrams, waterfall charts, efficiency trends, and loss breakdowns.

## Features

### 1. Sankey Diagrams (`sankey_engine.py`)
Interactive energy flow visualizations showing inputs, processes, outputs, and losses.

**Capabilities:**
- Single-stage and multi-stage process flows
- Color-coded by energy type, efficiency, or temperature
- Automatic node positioning with manual override
- Provenance hashing for data lineage tracking
- Support for complex industrial processes

**Example:**
```python
from visualization import SankeyEngine, ColorScheme

engine = SankeyEngine(color_scheme=ColorScheme.ENERGY_TYPE)
diagram = engine.generate_from_efficiency_result(
    energy_inputs={"natural_gas": 5000.0, "electricity": 150.0},
    useful_outputs={"steam": 4200.0, "hot_water": 300.0},
    losses={"flue_gas": 350.0, "radiation": 120.0, "blowdown": 100.0},
    title="Industrial Boiler Energy Flow"
)

# Export to Plotly JSON
plotly_fig = diagram.to_plotly_json()
```

### 2. Waterfall Charts (`waterfall_chart.py`)
Step-by-step heat balance breakdown from input through losses to output.

**Capabilities:**
- Sequential energy transformation visualization
- Color-coded gains and losses
- Cumulative energy tracking
- Detailed process and distribution stage breakdown

**Example:**
```python
from visualization import WaterfallChart

chart = WaterfallChart()
waterfall = chart.generate_from_heat_balance(
    input_energy={"fuel_input": 5150.0},
    losses={
        "flue_gas": 350.0,
        "radiation": 120.0,
        "convection": 80.0
    },
    useful_output={"steam_output": 4450.0},
    title="Boiler Heat Balance"
)

plotly_fig = waterfall.to_plotly_json()
```

### 3. Efficiency Trends (`efficiency_trends.py`)
Time-series analysis for efficiency monitoring and benchmarking.

**Capabilities:**
- Multi-metric trend visualization
- Moving averages and smoothing
- Benchmark comparisons
- Baseline vs current performance analysis
- Statistical summaries (avg, min, max, std dev)

**Example:**
```python
from visualization import EfficiencyTrends
from datetime import datetime, timedelta

trends = EfficiencyTrends()

# Generate sample data
efficiency_data = [
    (datetime(2024, 1, 1) + timedelta(days=i), 87.0 + i * 0.1)
    for i in range(30)
]

trend = trends.generate_efficiency_trend(
    efficiency_data=efficiency_data,
    title="30-Day Thermal Efficiency Trend",
    benchmark_efficiency=88.0,
    moving_average_days=7
)

plotly_fig = trend.to_plotly_json()
```

### 4. Loss Breakdown (`loss_breakdown.py`)
Pie charts, donut charts, and bar charts for heat loss distribution.

**Capabilities:**
- Pie charts with percentage labels
- Donut charts with center text
- Horizontal and vertical bar charts
- Baseline vs current comparison charts
- Color-coded by loss type

**Example:**
```python
from visualization import LossBreakdown, ChartType

breakdown = LossBreakdown()

# Pie chart
pie = breakdown.generate_pie_chart(
    losses={
        "flue_gas": 350.0,
        "radiation": 120.0,
        "convection": 80.0
    },
    title="Heat Loss Distribution",
    total_input=5150.0
)

# Donut chart
donut = breakdown.generate_donut_chart(
    losses={...},
    title="Loss Breakdown"
)

# Bar chart
bar = breakdown.generate_bar_chart(
    losses={...},
    horizontal=True
)

plotly_fig = pie.to_plotly_json()
```

### 5. Export Utilities (`export.py`)
Multi-format export for all visualization types.

**Supported Formats:**
- HTML (interactive with embedded Plotly.js)
- PNG (static image via HTML download)
- SVG (vector graphics via HTML download)
- JSON (raw data with metadata)
- Dashboard (multi-chart HTML with grid/tabs layout)

**Example:**
```python
from visualization import (
    export_to_html,
    export_to_json,
    export_dashboard,
    ExportConfig
)

# Configure export
config = ExportConfig(width=1200, height=800, dpi=300)

# Export single chart
export_to_html(
    plotly_fig,
    "energy_flow.html",
    title="My Energy Flow Analysis",
    config=config
)

# Export to JSON
export_to_json(plotly_fig, "energy_flow.json", config=config)

# Export dashboard
export_dashboard(
    figures=[fig1, fig2, fig3],
    output_path="dashboard.html",
    title="Energy Analysis Dashboard",
    layout="grid"  # or "tabs", "vertical"
)
```

## Installation

### Required Dependencies
```bash
pip install plotly>=5.18.0
```

### Optional Dependencies (for static image export)
```bash
pip install kaleido>=0.2.1  # For programmatic PNG/SVG export
```

## Quick Start

```python
from visualization import (
    SankeyEngine,
    WaterfallChart,
    EfficiencyTrends,
    LossBreakdown,
    export_to_html
)

# 1. Generate Sankey diagram
engine = SankeyEngine()
sankey = engine.generate_from_efficiency_result(
    energy_inputs={"fuel": 5000},
    useful_outputs={"steam": 4200},
    losses={"flue_gas": 350, "radiation": 120}
)

# 2. Export to HTML
export_to_html(
    sankey.to_plotly_json(),
    "energy_flow.html",
    title="Energy Flow Analysis"
)
```

## Examples

Run the included examples file to generate sample visualizations:

```bash
python examples.py
```

This will create:
- 10 individual visualization examples
- 2 comprehensive dashboards (grid + tabs layout)
- Output in `examples_output/` directory

## API Reference

### SankeyEngine

```python
class SankeyEngine:
    def __init__(self, color_scheme: ColorScheme = ColorScheme.EFFICIENCY)

    def generate_from_efficiency_result(
        energy_inputs: Dict[str, float],
        useful_outputs: Dict[str, float],
        losses: Dict[str, float],
        title: str = "Thermal Energy Flow",
        process_name: str = "Process",
        metadata: Optional[Dict[str, Any]] = None
    ) -> SankeyDiagram

    def generate_multi_stage(
        stages: List[Dict[str, Any]],
        title: str = "Multi-Stage Energy Flow"
    ) -> SankeyDiagram
```

### WaterfallChart

```python
class WaterfallChart:
    def generate_from_heat_balance(
        input_energy: Dict[str, float],
        losses: Dict[str, float],
        useful_output: Dict[str, float],
        title: str = "Heat Balance Waterfall",
        include_subtotals: bool = True
    ) -> WaterfallData

    def generate_detailed_breakdown(
        input_energy: Dict[str, float],
        process_losses: Dict[str, float],
        distribution_losses: Dict[str, float],
        useful_output: Dict[str, float],
        title: str = "Detailed Heat Balance"
    ) -> WaterfallData
```

### EfficiencyTrends

```python
class EfficiencyTrends:
    def generate_efficiency_trend(
        efficiency_data: List[Tuple[datetime, float]],
        title: str = "Thermal Efficiency Trend",
        benchmark_efficiency: Optional[float] = None,
        moving_average_days: int = 7
    ) -> TrendData

    def generate_loss_trend(
        loss_data: List[Tuple[datetime, Dict[str, float]]],
        title: str = "Heat Loss Trends",
        loss_categories: Optional[List[str]] = None
    ) -> Dict[str, TrendData]

    def generate_comparison_chart(
        baseline_data: List[Tuple[datetime, float]],
        current_data: List[Tuple[datetime, float]],
        title: str = "Baseline vs Current Performance"
    ) -> Dict
```

### LossBreakdown

```python
class LossBreakdown:
    def generate_pie_chart(
        losses: Dict[str, float],
        title: str = "Heat Loss Breakdown",
        total_input: Optional[float] = None
    ) -> BreakdownChart

    def generate_donut_chart(
        losses: Dict[str, float],
        title: str = "Heat Loss Distribution",
        total_input: Optional[float] = None
    ) -> BreakdownChart

    def generate_bar_chart(
        losses: Dict[str, float],
        title: str = "Heat Loss Comparison",
        horizontal: bool = False
    ) -> BreakdownChart

    def generate_comparison_chart(
        baseline_losses: Dict[str, float],
        current_losses: Dict[str, float],
        title: str = "Loss Comparison: Baseline vs Current"
    ) -> Dict
```

## Color Schemes

### Energy Type Colors
- **Inputs:** Fuel (red), Electricity (teal), Steam (blue)
- **Outputs:** Steam (green), Hot Water (yellow), Process Heat (purple)
- **Losses:** Radiation (red), Flue Gas (gray), Convection (orange)
- **Processes:** Boiler (blue), Furnace (red), Heat Exchanger (green)

### Color Customization
All visualizations support custom colors through the `color` parameter in node/category definitions.

## Integration with GL-009 THERMALIQ

The visualization module integrates seamlessly with GL-009 calculators:

```python
from calculators import BoilerEfficiencyCalculator
from visualization import SankeyEngine, export_to_html

# Calculate efficiency
calculator = BoilerEfficiencyCalculator()
result = calculator.calculate({
    "fuel_input_kw": 5000.0,
    "steam_output_kw": 4200.0,
    "feedwater_temp_c": 25.0,
    "steam_temp_c": 180.0
})

# Visualize results
engine = SankeyEngine()
diagram = engine.generate_from_efficiency_result(
    energy_inputs=result["energy_inputs"],
    useful_outputs=result["useful_outputs"],
    losses=result["losses"]
)

export_to_html(diagram.to_plotly_json(), "boiler_analysis.html")
```

## Performance Considerations

- **Large datasets:** Use `head_limit` and pagination for trend data
- **Dashboard size:** Limit to 4-6 charts per dashboard for optimal performance
- **Export time:** HTML export is instant; PNG/SVG requires browser rendering
- **File size:** HTML files are 100-500 KB; include Plotly.js CDN to reduce size

## Troubleshooting

### Common Issues

**Q: Charts not displaying in HTML?**
A: Ensure internet connection for Plotly.js CDN, or set `include_plotlyjs=True`

**Q: Static image export not working?**
A: Install `kaleido` package: `pip install kaleido`

**Q: Dashboard layout issues?**
A: Adjust `ExportConfig` width/height or use different layout mode

**Q: Performance slow with large datasets?**
A: Use data sampling or aggregation for trend charts

## Contributing

For bugs, feature requests, or contributions, contact the GL-009 THERMALIQ team.

## License

Copyright (c) 2024 GreenLang. All rights reserved.

## Version History

- **1.0.0** (2024-11-26): Initial release
  - Sankey diagram engine
  - Waterfall charts
  - Efficiency trends
  - Loss breakdown charts
  - Multi-format export
  - Dashboard generation
