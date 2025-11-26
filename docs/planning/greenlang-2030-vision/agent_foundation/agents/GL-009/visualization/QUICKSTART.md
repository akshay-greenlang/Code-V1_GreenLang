# GL-009 THERMALIQ Visualization Quick Start

Get started with THERMALIQ visualizations in 5 minutes.

## Installation

```bash
pip install plotly>=5.18.0
```

## 1. Basic Sankey Diagram

```python
from visualization import SankeyEngine, export_to_html

# Create engine
engine = SankeyEngine()

# Generate diagram
diagram = engine.generate_from_efficiency_result(
    energy_inputs={"natural_gas": 5000.0},
    useful_outputs={"steam": 4200.0},
    losses={"flue_gas": 350.0, "radiation": 120.0}
)

# Export to HTML
export_to_html(
    diagram.to_plotly_json(),
    "energy_flow.html",
    title="My Boiler Energy Flow"
)
```

**Output:** Interactive Sankey diagram showing energy flow from inputs through losses to outputs.

## 2. Heat Balance Waterfall

```python
from visualization import WaterfallChart, export_to_html

# Create chart
chart = WaterfallChart()

# Generate waterfall
waterfall = chart.generate_from_heat_balance(
    input_energy={"fuel": 5000.0},
    losses={"flue_gas": 350, "radiation": 120, "convection": 80},
    useful_output={"steam": 4450.0}
)

# Export
export_to_html(
    waterfall.to_plotly_json(),
    "heat_balance.html"
)
```

**Output:** Waterfall chart showing step-by-step energy transformation.

## 3. Efficiency Trends

```python
from visualization import EfficiencyTrends, export_to_html
from datetime import datetime, timedelta

# Create trends analyzer
trends = EfficiencyTrends()

# Sample data (30 days)
efficiency_data = [
    (datetime(2024, 1, 1) + timedelta(days=i), 87.0 + i * 0.1)
    for i in range(30)
]

# Generate trend
trend = trends.generate_efficiency_trend(
    efficiency_data=efficiency_data,
    benchmark_efficiency=88.0,
    moving_average_days=7
)

# Export
export_to_html(
    trend.to_plotly_json(),
    "efficiency_trend.html"
)
```

**Output:** Time-series chart with moving average and benchmark line.

## 4. Loss Breakdown

```python
from visualization import LossBreakdown, export_to_html

# Create breakdown
breakdown = LossBreakdown()

# Generate pie chart
chart = breakdown.generate_pie_chart(
    losses={
        "flue_gas": 350.0,
        "radiation": 120.0,
        "convection": 80.0,
        "blowdown": 100.0
    },
    total_input=5000.0
)

# Export
export_to_html(
    chart.to_plotly_json(),
    "loss_breakdown.html"
)
```

**Output:** Pie chart with percentages and color-coded loss categories.

## 5. Complete Dashboard

```python
from visualization import (
    SankeyEngine,
    WaterfallChart,
    EfficiencyTrends,
    LossBreakdown,
    export_dashboard
)

# Generate all charts
engine = SankeyEngine()
sankey = engine.generate_from_efficiency_result(...)

chart = WaterfallChart()
waterfall = chart.generate_from_heat_balance(...)

trends = EfficiencyTrends()
trend = trends.generate_efficiency_trend(...)

breakdown = LossBreakdown()
pie = breakdown.generate_pie_chart(...)

# Combine into dashboard
export_dashboard(
    figures=[
        sankey.to_plotly_json(),
        waterfall.to_plotly_json(),
        trend.to_plotly_json(),
        pie.to_plotly_json()
    ],
    output_path="dashboard.html",
    title="THERMALIQ Energy Analysis Dashboard",
    layout="grid"  # or "tabs"
)
```

**Output:** Multi-chart dashboard with all visualizations.

## Run Examples

```bash
cd visualization
python examples.py
```

This generates 10+ example visualizations in `examples_output/` directory.

## Run Tests

```bash
cd visualization
python test_visualization.py
```

This runs comprehensive unit tests for all components.

## Next Steps

1. Read full documentation: `README.md`
2. Explore examples: `examples.py`
3. Integrate with GL-009 calculators
4. Customize colors and styling
5. Export to multiple formats (HTML, JSON, PNG, SVG)

## Common Patterns

### Integration with Calculator Results

```python
from calculators import BoilerEfficiencyCalculator
from visualization import SankeyEngine, export_to_html

# Calculate
calculator = BoilerEfficiencyCalculator()
result = calculator.calculate({
    "fuel_input_kw": 5000.0,
    "steam_output_kw": 4200.0,
    # ... other parameters
})

# Visualize
engine = SankeyEngine()
diagram = engine.generate_from_efficiency_result(
    energy_inputs=result["energy_inputs"],
    useful_outputs=result["useful_outputs"],
    losses=result["losses"]
)

export_to_html(diagram.to_plotly_json(), "boiler_analysis.html")
```

### Custom Colors

```python
from visualization import SankeyEngine, SankeyNode

# Manual node creation with custom colors
nodes = [
    SankeyNode(
        id="custom_input",
        label="Custom Input",
        node_type=NodeType.INPUT,
        value_kw=1000.0,
        color="#FF0000"  # Custom red
    )
]
```

### Export Formats

```python
from visualization import (
    export_to_html,
    export_to_json,
    export_to_png,
    ExportConfig
)

# Configure
config = ExportConfig(width=1600, height=1000, dpi=300)

# Export
export_to_html(figure, "chart.html", config=config)
export_to_json(figure, "chart.json", config=config)
export_to_png(figure, "chart.png", config=config)  # Creates HTML with download
```

## Troubleshooting

**Charts not showing?**
- Check internet connection (Plotly.js CDN)
- Open HTML in modern browser (Chrome, Firefox, Edge)

**Performance issues?**
- Reduce data points for trends
- Use sampling for large datasets
- Limit dashboard to 4-6 charts

**Need static images?**
- Install: `pip install kaleido`
- Or use browser "Save as PNG" from interactive chart

## Support

For issues, contact the GL-009 THERMALIQ team or refer to:
- Full documentation: `README.md`
- Example code: `examples.py`
- Unit tests: `test_visualization.py`
