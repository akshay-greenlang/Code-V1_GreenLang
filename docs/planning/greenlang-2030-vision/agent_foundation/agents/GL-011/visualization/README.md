# GL-011 FUELCRAFT Visualization Suite

## Overview

The GL-011 FUELCRAFT Visualization Suite provides a comprehensive set of interactive visualization tools designed specifically for fuel management analytics. This module enables organizations to visualize complex fuel procurement, blending, cost analysis, and carbon footprint data through intuitive, accessible, and high-performance charts and dashboards.

**Version:** 1.0.0
**Author:** GreenLang Team
**Standards Compliance:** WCAG 2.1 Level AA, ISO 12647-2, GHG Protocol

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Modules](#modules)
   - [Configuration](#configuration)
   - [Waterfall Charts](#waterfall-charts)
   - [Sankey Diagrams](#sankey-diagrams)
   - [Carbon Footprint](#carbon-footprint)
   - [Price Trends](#price-trends)
   - [Procurement Dashboard](#procurement-dashboard)
   - [Export Engine](#export-engine)
5. [API Reference](#api-reference)
6. [Theming and Styling](#theming-and-styling)
7. [Accessibility](#accessibility)
8. [Performance Optimization](#performance-optimization)
9. [Integration Guide](#integration-guide)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [License](#license)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 16+ (for React integration)
- Required Python packages:
  - plotly >= 5.0.0
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - kaleido >= 0.2.0 (for image export)
  - reportlab >= 3.6.0 (for PDF export)
  - openpyxl >= 3.0.0 (for Excel export)

### Python Installation

```bash
# Install via pip
pip install greenlang-visualization

# Or install from source
git clone https://github.com/greenlang/gl-011-visualization.git
cd gl-011-visualization
pip install -e .
```

### React/TypeScript Installation

```bash
# Install npm package
npm install @greenlang/visualization

# Or with yarn
yarn add @greenlang/visualization
```

---

## Quick Start

### Python Quick Start

```python
from visualization import (
    WaterfallChartEngine,
    FuelBlendSankeyEngine,
    CarbonFootprintEngine,
    PriceTrendEngine,
    ProcurementDashboardEngine,
    ExportEngine,
    ConfigFactory,
    create_sample_cost_breakdown,
    create_sample_blend_flow,
    create_sample_carbon_data,
    create_sample_price_data,
    create_sample_dashboard_data,
)

# Initialize configuration
config = ConfigFactory.get_instance()

# Create a waterfall chart
waterfall_engine = WaterfallChartEngine()
cost_data = create_sample_cost_breakdown()
waterfall_spec = waterfall_engine.generate(cost_data)

# Create a Sankey diagram
sankey_engine = FuelBlendSankeyEngine()
blend_data = create_sample_blend_flow()
sankey_spec = sankey_engine.generate(blend_data)

# Create carbon footprint visualization
carbon_engine = CarbonFootprintEngine()
carbon_data = create_sample_carbon_data()
carbon_spec = carbon_engine.generate(carbon_data)

# Export to PNG
export_engine = ExportEngine()
export_engine.export_to_png(waterfall_spec, "/output/waterfall.png")
```

### React/TypeScript Quick Start

```typescript
import React from 'react';
import Plot from 'react-plotly.js';
import {
  WaterfallChartEngine,
  FuelBlendSankeyEngine,
  createSampleCostBreakdown,
  createSampleBlendFlow,
} from '@greenlang/visualization';

// Initialize engines
const waterfallEngine = new WaterfallChartEngine();
const sankeyEngine = new FuelBlendSankeyEngine();

// Generate chart specifications
const costData = createSampleCostBreakdown();
const waterfallSpec = waterfallEngine.generate(costData);

const blendData = createSampleBlendFlow();
const sankeySpec = sankeyEngine.generate(blendData);

// Render in React component
const FuelDashboard: React.FC = () => {
  return (
    <div className="dashboard">
      <h2>Cost Breakdown</h2>
      <Plot
        data={waterfallSpec.data}
        layout={waterfallSpec.layout}
        config={waterfallSpec.config}
      />

      <h2>Fuel Blend Flow</h2>
      <Plot
        data={sankeySpec.data}
        layout={sankeySpec.layout}
        config={sankeySpec.config}
      />
    </div>
  );
};

export default FuelDashboard;
```

---

## Architecture

### Module Structure

```
visualization/
├── __init__.py                    # Module exports and public API
├── config.py                      # Configuration and theming
├── fuel_cost_waterfall.py         # Waterfall chart engine
├── fuel_blend_sankey.py           # Sankey diagram engine
├── carbon_footprint_breakdown.py  # Carbon visualization
├── price_trends.py                # Time-series price charts
├── procurement_dashboard.py       # Multi-panel dashboard
├── export_engine.py               # Universal export functionality
└── README.md                      # This documentation
```

### Design Principles

1. **Plotly-Native Output**: All engines generate Plotly-compatible JSON specifications
2. **Type Safety**: Full TypeScript/dataclass type annotations throughout
3. **Immutable Data**: Data classes use frozen=True for thread safety
4. **Factory Pattern**: ConfigFactory provides global configuration management
5. **Builder Pattern**: BlendFlowDataBuilder enables fluent data construction
6. **Caching**: Built-in caching with provenance hash keys for performance

### Data Flow

```
Raw Data → Data Classes → Engine → Plotly Spec → Renderer/Export
              ↑
         Validation
              ↑
       Configuration
```

---

## Modules

### Configuration

The configuration module provides centralized management of visualization settings, color schemes, and themes.

#### Color Schemes

```python
from visualization import (
    FuelTypeColors,
    CostCategoryColors,
    EmissionColors,
    StatusColors,
    get_fuel_color,
    get_cost_color,
)

# Get fuel type color
diesel_color = get_fuel_color("diesel")  # Returns "#2E5A88"
biodiesel_color = get_fuel_color("biodiesel")  # Returns "#4CAF50"

# Get cost category color
base_cost_color = get_cost_color("base_cost")  # Returns "#2196F3"
carbon_tax_color = get_cost_color("carbon_tax")  # Returns "#FF5722"
```

#### Theme Management

```python
from visualization import (
    ThemeMode,
    ThemeConfig,
    ConfigFactory,
)

# Get configuration factory
config = ConfigFactory.get_instance()

# Set theme mode
config.set_theme(ThemeMode.DARK)
config.set_theme(ThemeMode.LIGHT)
config.set_theme(ThemeMode.HIGH_CONTRAST)  # For accessibility

# Get current theme configuration
theme = config.theme_config
print(f"Background: {theme.background_color}")
print(f"Text color: {theme.text_color}")
```

#### Custom Configuration

```python
from visualization import (
    VisualizationConfig,
    FontConfig,
    MarginConfig,
    AnimationConfig,
)

# Create custom configuration
custom_config = VisualizationConfig(
    font=FontConfig(
        family="Arial, sans-serif",
        size=14,
        color="#333333"
    ),
    margin=MarginConfig(
        left=80,
        right=40,
        top=60,
        bottom=80
    ),
    animation=AnimationConfig(
        enabled=True,
        duration=500,
        easing="cubic-in-out"
    ),
    width=1200,
    height=800
)
```

---

### Waterfall Charts

The waterfall chart module provides visualization of cost breakdowns, showing how individual components contribute to a total value.

#### Basic Usage

```python
from visualization import (
    WaterfallChartEngine,
    CostItem,
    CostBreakdown,
    CostCategory,
    WaterfallChartOptions,
)

# Create cost items
items = [
    CostItem(
        id="base_fuel",
        category=CostCategory.BASE_FUEL_COST,
        label="Base Fuel Cost",
        value=1000000.0,
        unit="USD"
    ),
    CostItem(
        id="blending",
        category=CostCategory.BLENDING_COST,
        label="Blending Operations",
        value=150000.0,
        unit="USD"
    ),
    CostItem(
        id="carbon_tax",
        category=CostCategory.CARBON_TAX,
        label="Carbon Tax",
        value=75000.0,
        unit="USD"
    ),
    CostItem(
        id="transport",
        category=CostCategory.LOGISTICS_COST,
        label="Transportation",
        value=50000.0,
        unit="USD"
    ),
]

# Create cost breakdown
breakdown = CostBreakdown(
    id="q4_2024",
    name="Q4 2024 Fuel Costs",
    items=items,
    total=1275000.0,
    currency="USD",
    period_start="2024-10-01",
    period_end="2024-12-31"
)

# Configure options
options = WaterfallChartOptions(
    show_connectors=True,
    show_totals=True,
    show_percentages=True,
    enable_drill_down=True,
    color_positive="#4CAF50",
    color_negative="#F44336"
)

# Generate chart
engine = WaterfallChartEngine(options=options)
spec = engine.generate(breakdown)
```

#### Fuel Switching Waterfall

Compare costs before and after fuel switching:

```python
from visualization import FuelSwitchingWaterfall

# Create fuel switching analysis
engine = FuelSwitchingWaterfall()
spec = engine.generate(
    before_data=diesel_breakdown,
    after_data=biodiesel_breakdown,
    title="Diesel to Biodiesel Transition Cost Impact"
)
```

#### Blend Optimization Waterfall

Visualize cost impact of blend optimization:

```python
from visualization import BlendOptimizationWaterfall

engine = BlendOptimizationWaterfall()
spec = engine.generate(
    baseline_data=current_blend,
    optimized_data=optimized_blend,
    optimization_targets=["cost", "emissions"]
)
```

#### Carbon Cost Waterfall

Analyze carbon cost components:

```python
from visualization import CarbonCostWaterfall

engine = CarbonCostWaterfall()
spec = engine.generate(
    carbon_data=carbon_cost_breakdown,
    show_regulatory_limits=True,
    include_offset_credits=True
)
```

---

### Sankey Diagrams

The Sankey diagram module visualizes flow relationships between fuel sources, blending processes, and energy outputs.

#### Basic Usage

```python
from visualization import (
    FuelBlendSankeyEngine,
    BlendFlowData,
    FuelSource,
    BlendRecipe,
    EnergyOutput,
    SankeyChartOptions,
)

# Create fuel sources
sources = [
    FuelSource(
        id="diesel",
        name="Ultra Low Sulfur Diesel",
        fuel_type="diesel",
        volume=10000.0,
        unit="liters",
        carbon_intensity=2.68
    ),
    FuelSource(
        id="biodiesel",
        name="B100 Biodiesel",
        fuel_type="biodiesel",
        volume=2000.0,
        unit="liters",
        carbon_intensity=0.45
    ),
]

# Create blend recipe
recipes = [
    BlendRecipe(
        id="b20",
        name="B20 Blend",
        components={
            "diesel": 0.80,
            "biodiesel": 0.20
        },
        output_volume=12000.0,
        target_carbon_intensity=2.24
    ),
]

# Create energy outputs
outputs = [
    EnergyOutput(
        id="fleet",
        name="Fleet Operations",
        fuel_type="b20",
        consumption=12000.0,
        unit="liters",
        efficiency=0.92
    ),
]

# Create blend flow data
blend_data = BlendFlowData(
    id="q4_blend",
    name="Q4 2024 Fuel Blending",
    sources=sources,
    recipes=recipes,
    outputs=outputs
)

# Generate Sankey diagram
engine = FuelBlendSankeyEngine()
spec = engine.generate(blend_data)
```

#### Using the Builder Pattern

```python
from visualization import BlendFlowDataBuilder

# Build blend flow data fluently
builder = BlendFlowDataBuilder()
blend_data = (
    builder
    .set_id("production_blend")
    .set_name("Production Fuel Blend")
    .add_source(
        id="diesel",
        name="ULSD",
        fuel_type="diesel",
        volume=50000.0,
        carbon_intensity=2.68
    )
    .add_source(
        id="hvo",
        name="HVO Renewable Diesel",
        fuel_type="hvo",
        volume=10000.0,
        carbon_intensity=0.38
    )
    .add_recipe(
        id="hvo20",
        name="HVO20 Blend",
        components={"diesel": 0.80, "hvo": 0.20},
        output_volume=60000.0
    )
    .add_output(
        id="trucking",
        name="Trucking Fleet",
        fuel_type="hvo20",
        consumption=60000.0
    )
    .build()
)
```

#### Multi-Stage Sankey

Visualize multi-stage blending processes:

```python
from visualization import MultiStageBlendSankey

engine = MultiStageBlendSankey()
spec = engine.generate(
    stages=[stage1_data, stage2_data, stage3_data],
    show_intermediate_storage=True
)
```

#### Supply Chain Sankey

Visualize fuel supply chain flows:

```python
from visualization import SupplyChainSankey

engine = SupplyChainSankey()
spec = engine.generate(
    suppliers=supplier_list,
    terminals=terminal_list,
    distribution_points=distribution_list,
    end_users=end_user_list
)
```

---

### Carbon Footprint

The carbon footprint module provides visualization of emissions data following the GHG Protocol.

#### Basic Usage

```python
from visualization import (
    CarbonFootprintEngine,
    CarbonFootprintData,
    EmissionSource,
    EmissionScope,
    EmissionTarget,
    RegulatoryLimit,
    ChartMode,
    CarbonChartOptions,
)

# Create emission sources
sources = [
    EmissionSource(
        id="fleet_diesel",
        name="Fleet Diesel Combustion",
        scope=EmissionScope.SCOPE_1,
        emissions=5000.0,  # tCO2e
        unit="tCO2e",
        fuel_type="diesel",
        activity_data={"volume": 1870.0, "unit": "kL"}
    ),
    EmissionSource(
        id="electricity",
        name="Facility Electricity",
        scope=EmissionScope.SCOPE_2,
        emissions=1200.0,
        unit="tCO2e",
        activity_data={"consumption": 3000.0, "unit": "MWh"}
    ),
    EmissionSource(
        id="supply_chain",
        name="Supply Chain Transport",
        scope=EmissionScope.SCOPE_3,
        emissions=3500.0,
        unit="tCO2e"
    ),
]

# Create targets
targets = [
    EmissionTarget(
        id="2025_target",
        name="2025 Reduction Target",
        target_value=8000.0,
        target_year=2025,
        baseline_year=2020,
        baseline_value=12000.0
    ),
]

# Create regulatory limits
limits = [
    RegulatoryLimit(
        id="eu_ets",
        name="EU ETS Allocation",
        limit_value=7500.0,
        regulation="EU ETS",
        penalty_rate=100.0  # EUR per tCO2e
    ),
]

# Create carbon footprint data
carbon_data = CarbonFootprintData(
    id="2024_footprint",
    name="2024 Carbon Footprint",
    sources=sources,
    targets=targets,
    regulatory_limits=limits,
    reporting_period="2024"
)

# Generate visualization
options = CarbonChartOptions(
    chart_mode=ChartMode.DONUT,
    show_scope_breakdown=True,
    show_targets=True,
    show_regulatory_limits=True
)

engine = CarbonFootprintEngine(options=options)
spec = engine.generate(carbon_data)
```

#### Scope Breakdown

Generate scope-specific breakdown:

```python
from visualization import ScopeBreakdownGenerator

engine = ScopeBreakdownGenerator()
spec = engine.generate(
    carbon_data,
    scope=EmissionScope.SCOPE_1,
    breakdown_by="fuel_type"
)
```

#### Regulatory Compliance

Generate compliance dashboard:

```python
from visualization import RegulatoryComplianceGenerator

engine = RegulatoryComplianceGenerator()
spec = engine.generate(
    carbon_data,
    regulations=["EU_ETS", "CORSIA", "UK_ETS"],
    show_gap_analysis=True
)
```

---

### Price Trends

The price trends module provides time-series visualization of fuel prices with technical analysis.

#### Basic Usage

```python
from visualization import (
    PriceTrendEngine,
    PriceTrendData,
    FuelPriceSeries,
    PriceDataPoint,
    MovingAverageConfig,
    ForecastConfig,
    MarketEvent,
    PriceTrendOptions,
    TimeRange,
    ChartStyle,
)
from datetime import datetime, timedelta

# Create price data points
base_date = datetime(2024, 1, 1)
diesel_prices = [
    PriceDataPoint(
        timestamp=base_date + timedelta(days=i),
        price=1.50 + (i * 0.002) + (0.05 * (i % 7 - 3)),
        volume=1000000.0,
        currency="USD",
        unit="gallon"
    )
    for i in range(365)
]

# Create price series
diesel_series = FuelPriceSeries(
    id="diesel_2024",
    fuel_type="diesel",
    name="ULSD Gulf Coast",
    data_points=diesel_prices,
    source="Platts"
)

# Create market events
events = [
    MarketEvent(
        id="refinery_outage",
        timestamp=datetime(2024, 3, 15),
        event_type="supply_disruption",
        title="Gulf Coast Refinery Outage",
        description="Major refinery maintenance",
        impact_level="medium"
    ),
]

# Create trend data
trend_data = PriceTrendData(
    id="fuel_trends_2024",
    name="2024 Fuel Price Trends",
    series=[diesel_series],
    events=events
)

# Configure options
options = PriceTrendOptions(
    time_range=TimeRange.YEAR_TO_DATE,
    chart_style=ChartStyle.CANDLESTICK,
    moving_averages=[
        MovingAverageConfig(type="SMA", period=20),
        MovingAverageConfig(type="EMA", period=50),
    ],
    show_volume=True,
    show_events=True
)

# Generate chart
engine = PriceTrendEngine(options=options)
spec = engine.generate(trend_data)
```

#### Spot Price Tracker

Real-time spot price monitoring:

```python
from visualization import SpotPriceTracker

engine = SpotPriceTracker()
spec = engine.generate(
    fuel_types=["diesel", "biodiesel", "hvo"],
    markets=["gulf_coast", "nwe", "singapore"],
    refresh_interval=60  # seconds
)
```

#### Forward Curve Analyzer

Analyze forward price curves:

```python
from visualization import ForwardCurveAnalyzer

engine = ForwardCurveAnalyzer()
spec = engine.generate(
    spot_prices=current_spot,
    forward_contracts=forward_contracts,
    show_contango_backwardation=True,
    forecast_horizon=12  # months
)
```

---

### Procurement Dashboard

The procurement dashboard module provides multi-panel KPI dashboards for procurement management.

#### Basic Usage

```python
from visualization import (
    ProcurementDashboardEngine,
    ProcurementDashboardData,
    InventoryLevel,
    Contract,
    SupplierMetrics,
    DeliveryRecord,
    ProcurementAlert,
    KPIMetric,
    ProcurementDashboardOptions,
)
from datetime import datetime

# Create inventory data
inventory = [
    InventoryLevel(
        id="tank_1",
        location="Terminal A",
        fuel_type="diesel",
        current_level=850000.0,
        max_capacity=1000000.0,
        min_threshold=200000.0,
        unit="liters",
        last_updated=datetime.now()
    ),
]

# Create contracts
contracts = [
    Contract(
        id="contract_001",
        supplier="Shell",
        fuel_type="diesel",
        volume=5000000.0,
        price=1.45,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        status="active",
        fulfilled_volume=3500000.0
    ),
]

# Create supplier metrics
suppliers = [
    SupplierMetrics(
        id="shell",
        name="Shell",
        on_time_delivery_rate=0.98,
        quality_score=0.96,
        price_competitiveness=0.92,
        total_volume=5000000.0,
        total_spend=7250000.0
    ),
]

# Create alerts
alerts = [
    ProcurementAlert(
        id="alert_001",
        type="LOW_INVENTORY",
        severity="WARNING",
        title="Low Diesel Inventory at Terminal B",
        message="Diesel inventory below minimum threshold",
        timestamp=datetime.now(),
        acknowledged=False
    ),
]

# Create KPIs
kpis = [
    KPIMetric(
        id="cost_savings",
        name="Cost Savings YTD",
        value=1250000.0,
        target=1500000.0,
        unit="USD",
        trend="up",
        trend_value=0.15
    ),
]

# Create dashboard data
dashboard_data = ProcurementDashboardData(
    id="procurement_q4",
    name="Q4 2024 Procurement Dashboard",
    inventory=inventory,
    contracts=contracts,
    suppliers=suppliers,
    alerts=alerts,
    kpis=kpis,
    period_start=datetime(2024, 10, 1),
    period_end=datetime(2024, 12, 31)
)

# Configure options
options = ProcurementDashboardOptions(
    show_inventory_panel=True,
    show_contracts_panel=True,
    show_suppliers_panel=True,
    show_alerts_panel=True,
    show_kpi_panel=True,
    refresh_interval=300  # 5 minutes
)

# Generate dashboard
engine = ProcurementDashboardEngine(options=options)
specs = engine.generate(dashboard_data)
```

---

### Export Engine

The export engine provides universal export functionality for all visualization types.

#### Basic Usage

```python
from visualization import (
    ExportEngine,
    ExportOptions,
    ExportQuality,
    PageSize,
    PageOrientation,
    export_to_png,
    export_to_pdf,
    export_to_json,
    export_data_to_csv,
    export_data_to_excel,
)

# Initialize export engine
engine = ExportEngine()

# Configure export options
options = ExportOptions(
    quality=ExportQuality.HIGH,
    width=1920,
    height=1080,
    scale=2.0,  # Retina display
    page_size=PageSize.A4,
    page_orientation=PageOrientation.LANDSCAPE
)

# Export to various formats
engine.export_to_png(chart_spec, "/output/chart.png", options=options)
engine.export_to_svg(chart_spec, "/output/chart.svg", options=options)
engine.export_to_pdf(chart_spec, "/output/chart.pdf", options=options)
engine.export_to_json(chart_spec, "/output/chart.json")
engine.export_to_html(chart_spec, "/output/chart.html", options=options)
```

#### Convenience Functions

```python
# Quick exports using convenience functions
export_to_png(chart_spec, "/output/chart.png")
export_to_pdf(chart_spec, "/output/report.pdf")
export_to_json(chart_spec, "/output/data.json")

# Export data to tabular formats
export_data_to_csv(data, "/output/data.csv")
export_data_to_excel(data, "/output/data.xlsx")
```

#### Batch Export

```python
from visualization import BatchExportJob

# Create batch export job
job = BatchExportJob(
    id="monthly_reports",
    charts=[chart1, chart2, chart3],
    formats=["png", "pdf"],
    output_directory="/output/monthly",
    naming_pattern="{chart_name}_{date}_{format}"
)

# Execute batch export
results = engine.batch_export(job)
```

#### Template-Based Reports

```python
from visualization import ReportTemplate

# Register template
template = ReportTemplate(
    id="executive_report",
    name="Executive Summary Report",
    layout=[
        {"type": "header", "content": "Monthly Fuel Analytics Report"},
        {"type": "chart", "chart_id": "cost_waterfall", "position": "full"},
        {"type": "chart", "chart_id": "carbon_breakdown", "position": "half"},
        {"type": "chart", "chart_id": "price_trends", "position": "half"},
        {"type": "footer", "content": "Generated by GL-011 FUELCRAFT"}
    ],
    page_size=PageSize.A4,
    orientation=PageOrientation.PORTRAIT
)

engine.register_template(template)

# Generate report from template
engine.generate_report(
    template_id="executive_report",
    charts={
        "cost_waterfall": waterfall_spec,
        "carbon_breakdown": carbon_spec,
        "price_trends": price_spec
    },
    output_path="/output/executive_report.pdf"
)
```

#### Scheduled Exports

```python
from visualization import ScheduleConfig, ScheduleFrequency

# Configure scheduled export
schedule = ScheduleConfig(
    frequency=ScheduleFrequency.DAILY,
    time="06:00",
    timezone="UTC",
    email_recipients=["reports@company.com"],
    enabled=True
)

engine.schedule_export(
    template_id="daily_dashboard",
    schedule=schedule
)
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `WaterfallChartEngine` | Base waterfall chart generator |
| `FuelSwitchingWaterfall` | Fuel switching cost analysis |
| `BlendOptimizationWaterfall` | Blend optimization impact |
| `CarbonCostWaterfall` | Carbon cost breakdown |
| `FuelBlendSankeyEngine` | Base Sankey diagram generator |
| `MultiStageBlendSankey` | Multi-stage blending flows |
| `SupplyChainSankey` | Supply chain visualization |
| `CarbonFootprintEngine` | Carbon emissions charts |
| `ScopeBreakdownGenerator` | Scope-specific breakdowns |
| `RegulatoryComplianceGenerator` | Compliance dashboards |
| `PriceTrendEngine` | Time-series price charts |
| `SpotPriceTracker` | Real-time spot prices |
| `ForwardCurveAnalyzer` | Forward curve analysis |
| `ProcurementDashboardEngine` | Multi-panel KPI dashboard |
| `ExportEngine` | Universal export functionality |
| `ConfigFactory` | Global configuration management |

### Data Classes

| Class | Description |
|-------|-------------|
| `CostItem` | Individual cost component |
| `CostBreakdown` | Complete cost structure |
| `SankeyNode` | Sankey diagram node |
| `SankeyLink` | Sankey diagram link |
| `FuelSource` | Fuel source definition |
| `BlendRecipe` | Blending recipe |
| `EnergyOutput` | Energy consumption endpoint |
| `BlendFlowData` | Complete blend flow data |
| `EmissionSource` | Emission source definition |
| `EmissionTarget` | Reduction target |
| `RegulatoryLimit` | Regulatory emission limit |
| `CarbonFootprintData` | Complete carbon data |
| `PriceDataPoint` | Single price observation |
| `FuelPriceSeries` | Price time series |
| `MarketEvent` | Market event annotation |
| `PriceTrendData` | Complete price trend data |
| `InventoryLevel` | Inventory status |
| `Contract` | Procurement contract |
| `SupplierMetrics` | Supplier performance |
| `ProcurementAlert` | Procurement alert |
| `KPIMetric` | KPI metric value |
| `ExportResult` | Export operation result |

### Enumerations

| Enum | Values |
|------|--------|
| `ThemeMode` | LIGHT, DARK, HIGH_CONTRAST, PRINT, PRESENTATION |
| `ChartType` | WATERFALL, SANKEY, PIE, DONUT, BAR, LINE, AREA |
| `ExportFormat` | PNG, SVG, PDF, JSON, CSV, EXCEL, HTML |
| `EmissionScope` | SCOPE_1, SCOPE_2, SCOPE_3 |
| `CostCategory` | BASE_FUEL_COST, BLENDING_COST, CARBON_TAX, etc. |
| `TimeRange` | DAY, WEEK, MONTH, QUARTER, YEAR, YTD, CUSTOM |
| `AlertSeverity` | INFO, WARNING, CRITICAL, EMERGENCY |

---

## Theming and Styling

### Built-in Themes

```python
from visualization import ThemeMode, ConfigFactory

config = ConfigFactory.get_instance()

# Light theme (default)
config.set_theme(ThemeMode.LIGHT)

# Dark theme
config.set_theme(ThemeMode.DARK)

# High contrast for accessibility
config.set_theme(ThemeMode.HIGH_CONTRAST)

# Print-optimized
config.set_theme(ThemeMode.PRINT)

# Presentation mode
config.set_theme(ThemeMode.PRESENTATION)
```

### Custom Themes

```python
from visualization import ThemeConfig

custom_theme = ThemeConfig(
    background_color="#FAFAFA",
    paper_color="#FFFFFF",
    text_color="#212121",
    grid_color="#E0E0E0",
    primary_color="#1976D2",
    secondary_color="#424242",
    accent_color="#FF5722",
    success_color="#4CAF50",
    warning_color="#FF9800",
    error_color="#F44336"
)

config.set_custom_theme(custom_theme)
```

### Color Palettes

```python
from visualization import (
    FuelTypeColors,
    CostCategoryColors,
    EmissionColors,
    generate_color_palette,
)

# Use predefined colors
diesel_color = FuelTypeColors.DIESEL
biofuel_color = FuelTypeColors.BIODIESEL

# Generate custom palette
custom_palette = generate_color_palette(
    base_color="#1976D2",
    count=10,
    mode="analogous"  # or "complementary", "triadic"
)
```

---

## Accessibility

The visualization suite is designed to meet WCAG 2.1 Level AA standards.

### Color Contrast

All color combinations maintain minimum 4.5:1 contrast ratio for normal text and 3:1 for large text.

```python
from visualization import AccessibilityConfig

accessibility = AccessibilityConfig(
    high_contrast_mode=True,
    pattern_fills=True,  # Use patterns in addition to colors
    aria_labels=True,
    keyboard_navigation=True,
    screen_reader_support=True
)
```

### Color-Blind Safe Mode

```python
# Enable color-blind safe palette
config.enable_color_blind_safe_mode()

# Or use specific palette
from visualization import ColorPalette
config.set_color_palette(ColorPalette.COLOR_BLIND_SAFE)
```

### Screen Reader Support

All charts include ARIA labels and descriptions:

```python
options = WaterfallChartOptions(
    aria_label="Cost breakdown waterfall chart showing Q4 2024 fuel costs",
    aria_description="Interactive chart with 6 cost components totaling $1.275M"
)
```

### Keyboard Navigation

Charts support full keyboard navigation:

- **Tab**: Move between chart elements
- **Enter/Space**: Activate/select element
- **Arrow keys**: Navigate within chart
- **Escape**: Close tooltips/menus

---

## Performance Optimization

### Caching

The visualization suite includes built-in caching:

```python
from visualization import CachingConfig

caching = CachingConfig(
    enabled=True,
    max_size=100,  # Maximum cached items
    ttl=3600,  # Time-to-live in seconds
    use_provenance_hash=True  # Hash-based cache keys
)
```

### Large Datasets

For datasets with >10,000 points:

```python
options = PriceTrendOptions(
    downsample=True,
    downsample_threshold=10000,
    downsample_method="lttb",  # Largest-Triangle-Three-Buckets
    virtual_scroll=True
)
```

### Lazy Loading

Enable lazy loading for dashboards:

```python
options = ProcurementDashboardOptions(
    lazy_load_panels=True,
    load_priority=["kpi", "alerts", "inventory", "contracts", "suppliers"]
)
```

### Bundle Optimization

For React applications:

```typescript
// Tree-shakeable imports
import { WaterfallChartEngine } from '@greenlang/visualization/waterfall';
import { FuelBlendSankeyEngine } from '@greenlang/visualization/sankey';

// Instead of
import { WaterfallChartEngine, FuelBlendSankeyEngine } from '@greenlang/visualization';
```

---

## Integration Guide

### React Integration

```typescript
import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { useQuery } from '@tanstack/react-query';
import {
  WaterfallChartEngine,
  WaterfallChartOptions
} from '@greenlang/visualization';

interface FuelCostChartProps {
  jobId: string;
}

const FuelCostChart: React.FC<FuelCostChartProps> = ({ jobId }) => {
  const [chartSpec, setChartSpec] = useState<any>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['fuelCosts', jobId],
    queryFn: () => fetch(`/api/jobs/${jobId}/costs`).then(r => r.json()),
  });

  useEffect(() => {
    if (data) {
      const engine = new WaterfallChartEngine();
      const spec = engine.generate(data);
      setChartSpec(spec);
    }
  }, [data]);

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error loading chart</div>;
  if (!chartSpec) return null;

  return (
    <Plot
      data={chartSpec.data}
      layout={chartSpec.layout}
      config={chartSpec.config}
      style={{ width: '100%', height: '500px' }}
    />
  );
};

export default FuelCostChart;
```

### Flask/FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from visualization import (
    WaterfallChartEngine,
    export_to_json,
    export_to_png,
)
import io
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/api/charts/waterfall/{job_id}")
async def get_waterfall_chart(job_id: str):
    """Return waterfall chart specification as JSON."""
    data = await get_cost_data(job_id)
    engine = WaterfallChartEngine()
    spec = engine.generate(data)
    return spec

@app.get("/api/charts/waterfall/{job_id}/png")
async def get_waterfall_png(job_id: str):
    """Return waterfall chart as PNG image."""
    data = await get_cost_data(job_id)
    engine = WaterfallChartEngine()
    spec = engine.generate(data)

    buffer = io.BytesIO()
    export_to_png(spec, buffer)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=waterfall_{job_id}.png"}
    )
```

### Django Integration

```python
from django.http import JsonResponse, HttpResponse
from django.views import View
from visualization import WaterfallChartEngine, export_to_pdf

class WaterfallChartView(View):
    def get(self, request, job_id):
        data = CostBreakdown.objects.get(job_id=job_id)
        engine = WaterfallChartEngine()
        spec = engine.generate(data.to_dict())
        return JsonResponse(spec)

class WaterfallPDFView(View):
    def get(self, request, job_id):
        data = CostBreakdown.objects.get(job_id=job_id)
        engine = WaterfallChartEngine()
        spec = engine.generate(data.to_dict())

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="waterfall_{job_id}.pdf"'
        export_to_pdf(spec, response)
        return response
```

---

## Examples

### Complete Dashboard Example

```python
from visualization import (
    WaterfallChartEngine,
    FuelBlendSankeyEngine,
    CarbonFootprintEngine,
    PriceTrendEngine,
    ProcurementDashboardEngine,
    ExportEngine,
    create_sample_cost_breakdown,
    create_sample_blend_flow,
    create_sample_carbon_data,
    create_sample_price_data,
    create_sample_dashboard_data,
)

# Create all visualization engines
waterfall_engine = WaterfallChartEngine()
sankey_engine = FuelBlendSankeyEngine()
carbon_engine = CarbonFootprintEngine()
price_engine = PriceTrendEngine()
dashboard_engine = ProcurementDashboardEngine()
export_engine = ExportEngine()

# Generate sample data
cost_data = create_sample_cost_breakdown()
blend_data = create_sample_blend_flow()
carbon_data = create_sample_carbon_data()
price_data = create_sample_price_data()
dashboard_data = create_sample_dashboard_data()

# Generate all charts
waterfall_spec = waterfall_engine.generate(cost_data)
sankey_spec = sankey_engine.generate(blend_data)
carbon_spec = carbon_engine.generate(carbon_data)
price_spec = price_engine.generate(price_data)
dashboard_specs = dashboard_engine.generate(dashboard_data)

# Export all charts
export_engine.export_to_png(waterfall_spec, "output/waterfall.png")
export_engine.export_to_png(sankey_spec, "output/sankey.png")
export_engine.export_to_png(carbon_spec, "output/carbon.png")
export_engine.export_to_png(price_spec, "output/price_trends.png")

# Export dashboard panels
for panel_name, panel_spec in dashboard_specs.items():
    export_engine.export_to_png(panel_spec, f"output/dashboard_{panel_name}.png")

print("All charts exported successfully!")
```

### Interactive React Dashboard

```typescript
import React from 'react';
import { Grid, Card, CardContent, Typography } from '@mui/material';
import Plot from 'react-plotly.js';
import { useQuery } from '@tanstack/react-query';
import {
  WaterfallChartEngine,
  FuelBlendSankeyEngine,
  CarbonFootprintEngine,
  PriceTrendEngine,
} from '@greenlang/visualization';

const FuelAnalyticsDashboard: React.FC<{ jobId: string }> = ({ jobId }) => {
  const { data: analyticsData } = useQuery({
    queryKey: ['analytics', jobId],
    queryFn: () => fetch(`/api/jobs/${jobId}/analytics`).then(r => r.json()),
  });

  if (!analyticsData) return <div>Loading...</div>;

  // Generate chart specifications
  const waterfallEngine = new WaterfallChartEngine();
  const sankeyEngine = new FuelBlendSankeyEngine();
  const carbonEngine = new CarbonFootprintEngine();
  const priceEngine = new PriceTrendEngine();

  const waterfallSpec = waterfallEngine.generate(analyticsData.costs);
  const sankeySpec = sankeyEngine.generate(analyticsData.blends);
  const carbonSpec = carbonEngine.generate(analyticsData.carbon);
  const priceSpec = priceEngine.generate(analyticsData.prices);

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Cost Breakdown</Typography>
            <Plot
              data={waterfallSpec.data}
              layout={{ ...waterfallSpec.layout, autosize: true }}
              config={waterfallSpec.config}
              style={{ width: '100%', height: '400px' }}
            />
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Fuel Blend Flow</Typography>
            <Plot
              data={sankeySpec.data}
              layout={{ ...sankeySpec.layout, autosize: true }}
              config={sankeySpec.config}
              style={{ width: '100%', height: '400px' }}
            />
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Carbon Footprint</Typography>
            <Plot
              data={carbonSpec.data}
              layout={{ ...carbonSpec.layout, autosize: true }}
              config={carbonSpec.config}
              style={{ width: '100%', height: '400px' }}
            />
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Price Trends</Typography>
            <Plot
              data={priceSpec.data}
              layout={{ ...priceSpec.layout, autosize: true }}
              config={priceSpec.config}
              style={{ width: '100%', height: '400px' }}
            />
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default FuelAnalyticsDashboard;
```

---

## Troubleshooting

### Common Issues

#### Chart Not Rendering

```python
# Check if Plotly is properly installed
import plotly
print(plotly.__version__)  # Should be >= 5.0.0

# Verify chart specification
import json
print(json.dumps(chart_spec, indent=2))
```

#### Export Failing

```python
# For PNG/PDF export, ensure kaleido is installed
pip install kaleido

# Check kaleido version
import kaleido
print(kaleido.__version__)
```

#### Memory Issues with Large Datasets

```python
# Enable downsampling
options = PriceTrendOptions(
    downsample=True,
    downsample_threshold=5000
)

# Use streaming for exports
engine.export_to_pdf(spec, output_path, streaming=True)
```

#### Color Issues in Print

```python
# Use print-optimized theme
config.set_theme(ThemeMode.PRINT)

# Or enable grayscale
options.export_options.grayscale = True
```

### Getting Help

- **Documentation**: [https://docs.greenlang.io/gl-011/visualization](https://docs.greenlang.io/gl-011/visualization)
- **GitHub Issues**: [https://github.com/greenlang/gl-011/issues](https://github.com/greenlang/gl-011/issues)
- **Support Email**: support@greenlang.io
- **Community Slack**: [https://greenlang.slack.com](https://greenlang.slack.com)

---

## Contributing

We welcome contributions to the GL-011 FUELCRAFT Visualization Suite!

### Development Setup

```bash
# Clone repository
git clone https://github.com/greenlang/gl-011-visualization.git
cd gl-011-visualization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 visualization/
mypy visualization/
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type annotations throughout
- Write comprehensive docstrings
- Include unit tests for new features
- Maintain WCAG 2.1 Level AA accessibility

---

## License

Copyright (c) 2024 GreenLang Team. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

For licensing inquiries, contact: licensing@greenlang.io

---

## Changelog

### Version 1.0.0 (2024-12-01)

**Initial Release**

- Waterfall chart engine with fuel switching, blend optimization, and carbon cost analysis
- Sankey diagram engine for fuel blend flows with multi-stage and supply chain support
- Carbon footprint visualization with GHG Protocol scope tracking
- Price trend analysis with technical indicators and forecasting
- Procurement dashboard with multi-panel KPI views
- Universal export engine supporting PNG, SVG, PDF, JSON, CSV, Excel, HTML
- Comprehensive theming with light, dark, high-contrast, print, and presentation modes
- WCAG 2.1 Level AA accessibility compliance
- Built-in caching with provenance hash keys
- React/TypeScript integration support

---

*Generated by GL-011 FUELCRAFT Visualization Suite v1.0.0*
