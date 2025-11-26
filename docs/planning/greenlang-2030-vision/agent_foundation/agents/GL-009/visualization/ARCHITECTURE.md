# GL-009 THERMALIQ Visualization Module - Architecture

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GL-009 THERMALIQ Visualization                   │
│                         visualization/                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌────────────────┐  ┌────────────┐  ┌──────────────┐
        │ Sankey Engine  │  │ Waterfall  │  │  Efficiency  │
        │                │  │   Chart    │  │    Trends    │
        └────────────────┘  └────────────┘  └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌──────────────┐  ┌──────────┐  ┌──────────────┐
            │     Loss     │  │  Export  │  │   Examples   │
            │  Breakdown   │  │ Utilities│  │  & Testing   │
            └──────────────┘  └──────────┘  └──────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              ┌──────────┐    ┌──────────┐   ┌──────────┐
              │   HTML   │    │   JSON   │   │Dashboard │
              └──────────┘    └──────────┘   └──────────┘
```

## Component Hierarchy

### 1. Data Layer
```
Input Data (Dict[str, float])
    │
    ├── Energy Inputs (fuel, electricity, etc.)
    ├── Useful Outputs (steam, heat, power, etc.)
    └── Losses (flue gas, radiation, etc.)
```

### 2. Processing Layer
```
Generator Classes
    │
    ├── SankeyEngine
    │   ├── generate_from_efficiency_result()
    │   └── generate_multi_stage()
    │
    ├── WaterfallChart
    │   ├── generate_from_heat_balance()
    │   └── generate_detailed_breakdown()
    │
    ├── EfficiencyTrends
    │   ├── generate_efficiency_trend()
    │   ├── generate_loss_trend()
    │   └── generate_comparison_chart()
    │
    └── LossBreakdown
        ├── generate_pie_chart()
        ├── generate_donut_chart()
        └── generate_bar_chart()
```

### 3. Data Model Layer
```
Data Classes
    │
    ├── SankeyDiagram
    │   ├── nodes: List[SankeyNode]
    │   ├── links: List[SankeyLink]
    │   └── metadata: Dict
    │
    ├── WaterfallData
    │   ├── bars: List[WaterfallBar]
    │   └── statistics: Dict
    │
    ├── TrendData
    │   ├── points: List[TrendPoint]
    │   └── statistics: Dict
    │
    └── BreakdownChart
        ├── categories: List[LossCategory]
        └── metadata: Dict
```

### 4. Export Layer
```
Export Utilities
    │
    ├── export_to_html() → Standalone HTML
    ├── export_to_json() → JSON with metadata
    ├── export_to_png()  → PNG (via browser)
    ├── export_to_svg()  → SVG (via browser)
    └── export_dashboard() → Multi-chart HTML
```

## Data Flow

### Sankey Diagram Generation Flow
```
1. Input Data
   ├── energy_inputs: {"natural_gas": 5000}
   ├── useful_outputs: {"steam": 4200}
   └── losses: {"flue_gas": 350}
        │
        ▼
2. SankeyEngine.generate_from_efficiency_result()
   ├── Create nodes (inputs, process, outputs, losses)
   ├── Create links (flows between nodes)
   ├── Calculate efficiency
   └── Generate provenance hash
        │
        ▼
3. SankeyDiagram object
   ├── nodes: List[SankeyNode]
   ├── links: List[SankeyLink]
   ├── efficiency_percent: 87.4
   └── provenance_hash: "a1b2c3d4..."
        │
        ▼
4. to_plotly_json()
   ├── Map nodes to Plotly format
   ├── Map links to Plotly format
   └── Build layout and annotations
        │
        ▼
5. Plotly JSON
   {
     "data": [{...}],
     "layout": {...}
   }
        │
        ▼
6. export_to_html()
   └── Standalone HTML file
```

### Waterfall Chart Generation Flow
```
Input → WaterfallChart → WaterfallData → to_plotly_json() → Export
```

### Trend Analysis Flow
```
Time Series Data → EfficiencyTrends → TrendData → to_plotly_json() → Export
```

### Loss Breakdown Flow
```
Loss Dict → LossBreakdown → BreakdownChart → to_plotly_json() → Export
```

## Class Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                      SankeyEngine                           │
├─────────────────────────────────────────────────────────────┤
│ - color_scheme: ColorScheme                                 │
│ - COLORS: Dict[str, Dict[str, str]]                         │
├─────────────────────────────────────────────────────────────┤
│ + generate_from_efficiency_result() → SankeyDiagram         │
│ + generate_multi_stage() → SankeyDiagram                    │
└─────────────────────────────────────────────────────────────┘
                         │ creates
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     SankeyDiagram                           │
├─────────────────────────────────────────────────────────────┤
│ - nodes: List[SankeyNode]                                   │
│ - links: List[SankeyLink]                                   │
│ - efficiency_percent: float                                 │
│ - provenance_hash: str                                      │
├─────────────────────────────────────────────────────────────┤
│ + to_plotly_json() → Dict                                   │
│ + to_dict() → Dict                                          │
└─────────────────────────────────────────────────────────────┘
                         │ contains
            ┌────────────┴────────────┐
            ▼                         ▼
    ┌─────────────┐          ┌─────────────┐
    │ SankeyNode  │          │ SankeyLink  │
    ├─────────────┤          ├─────────────┤
    │ - id        │          │ - source_id │
    │ - label     │          │ - target_id │
    │ - value_kw  │          │ - value_kw  │
    │ - color     │          │ - color     │
    │ - node_type │          └─────────────┘
    └─────────────┘
```

## Export Architecture

```
┌──────────────────────────────────────────────────────────┐
│              VisualizationExporter                       │
├──────────────────────────────────────────────────────────┤
│ - config: ExportConfig                                   │
├──────────────────────────────────────────────────────────┤
│ + export(figure, path, format) → Path                    │
│ + export_dashboard(figures, path) → Path                 │
└──────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
  ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │   HTML   │    │   JSON   │    │  Dashboard   │
  │          │    │          │    │              │
  │ Plotly.js│    │ Metadata │    │ Multi-chart  │
  │ Embedded │    │ Included │    │ Grid/Tabs    │
  └──────────┘    └──────────┘    └──────────────┘
```

## Color Palette Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Color System                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Inputs (7 colors)                                      │
│  ├── Fuel       → #FF6B6B (Red)                         │
│  ├── Natural Gas→ #FF8C42 (Orange)                      │
│  ├── Electricity→ #4ECDC4 (Teal)                        │
│  └── ...                                                │
│                                                         │
│  Outputs (6 colors)                                     │
│  ├── Steam      → #96CEB4 (Green)                       │
│  ├── Hot Water  → #FFEAA7 (Yellow)                      │
│  └── ...                                                │
│                                                         │
│  Losses (9 colors)                                      │
│  ├── Flue Gas   → #95A5A6 (Gray)                        │
│  ├── Radiation  → #E74C3C (Red)                         │
│  └── ...                                                │
│                                                         │
│  Processes (6 colors)                                   │
│  ├── Boiler     → #3498DB (Blue)                        │
│  ├── Furnace    → #E74C3C (Red)                         │
│  └── ...                                                │
└─────────────────────────────────────────────────────────┘
```

## Integration Points

```
┌─────────────────────────────────────────────────────────┐
│           GL-009 THERMALIQ System Integration           │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Calculators  │  │   Runbooks   │  │   API        │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ Boiler       │→ │ Efficiency   │→ │ GET /viz     │
│ Furnace      │  │ Optimization │  │ POST /report │
│ Heat Exchg.  │  │ Analysis     │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
              ┌──────────────────┐
              │  Visualization   │
              │      Module      │
              └──────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Web UI       │  │ Reports      │  │ Dashboard    │
│ (React)      │  │ (PDF/Excel)  │  │ (Plotly)     │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Typical Usage Pattern

```
User Request
     │
     ├─→ GL-009 Calculator
     │        │
     │        ├─→ Calculate Efficiency
     │        └─→ Return Results
     │                 │
     ├─→ Visualization Module
     │        │
     │        ├─→ Generate Sankey
     │        ├─→ Generate Waterfall
     │        ├─→ Generate Trends
     │        └─→ Generate Loss Breakdown
     │                 │
     ├─→ Export Utilities
     │        │
     │        ├─→ Create Dashboard HTML
     │        ├─→ Export JSON
     │        └─→ Return File Paths
     │                 │
     └─→ Deliver to User
              │
              ├─→ Interactive HTML
              ├─→ JSON Data
              └─→ Reports
```

## Performance Considerations

```
┌─────────────────────────────────────────────────────────┐
│              Performance Architecture                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Data Processing (Python)                               │
│  ├── Dataclasses (fast)                                 │
│  ├── No heavy dependencies                              │
│  └── Efficient algorithms                               │
│                                                         │
│  Rendering (Client-side)                                │
│  ├── Plotly.js (WebGL accelerated)                      │
│  ├── SVG/Canvas rendering                               │
│  └── Browser caching                                    │
│                                                         │
│  Export (Hybrid)                                        │
│  ├── HTML: Instant                                      │
│  ├── JSON: Instant                                      │
│  └── Images: Browser-based                              │
│                                                         │
│  Optimization Strategies                                │
│  ├── Data sampling for large datasets                   │
│  ├── Lazy loading for dashboards                        │
│  ├── CDN for Plotly.js                                  │
│  └── Configurable chart sizes                           │
└─────────────────────────────────────────────────────────┘
```

## Testing Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Test Coverage                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Unit Tests (30+ methods)                               │
│  ├── TestSankeyEngine                                   │
│  ├── TestWaterfallChart                                 │
│  ├── TestEfficiencyTrends                               │
│  ├── TestLossBreakdown                                  │
│  └── TestExport                                         │
│                                                         │
│  Integration Tests                                      │
│  └── TestIntegration                                    │
│      ├── Complete workflow                              │
│      ├── Multi-chart generation                         │
│      └── Export pipeline                                │
│                                                         │
│  Example Tests                                          │
│  └── examples.py                                        │
│      ├── 10 example visualizations                      │
│      └── Dashboard generation                           │
└─────────────────────────────────────────────────────────┘
```

## Security & Provenance

```
┌─────────────────────────────────────────────────────────┐
│           Data Lineage & Provenance                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input Data                                             │
│      │                                                  │
│      ├─→ SHA-256 Hash Generation                        │
│      │       │                                          │
│      │       └─→ Provenance Hash (16 chars)             │
│      │                                                  │
│      ├─→ Timestamp Recording                            │
│      │       │                                          │
│      │       └─→ ISO 8601 format                        │
│      │                                                  │
│      └─→ Metadata Preservation                          │
│              │                                          │
│              └─→ Facility, Equipment, Parameters        │
│                                                         │
│  Stored in Output                                       │
│      ├─→ JSON exports                                   │
│      ├─→ Diagram objects                                │
│      └─→ HTML metadata                                  │
└─────────────────────────────────────────────────────────┘
```

This architecture ensures:
- Clean separation of concerns
- Easy extensibility
- High performance
- Full traceability
- Production-ready reliability
