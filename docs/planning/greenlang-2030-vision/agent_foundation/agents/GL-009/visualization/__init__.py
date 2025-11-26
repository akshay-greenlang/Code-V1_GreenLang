"""GL-009 THERMALIQ Visualization Module.

Comprehensive energy flow visualization toolkit for thermal efficiency analysis.
Generates interactive Sankey diagrams, waterfall charts, and efficiency trends.

Key Components:
- SankeyEngine: Interactive energy flow Sankey diagrams
- WaterfallChart: Heat balance breakdown visualization
- EfficiencyTrends: Time-series efficiency analysis
- LossBreakdown: Loss distribution pie/bar charts
- Export: Multi-format export utilities (HTML, PNG, SVG, JSON)

All visualizations are Plotly-compatible for web rendering.
"""

from .sankey_engine import (
    SankeyEngine,
    SankeyDiagram,
    SankeyNode,
    SankeyLink,
    NodeType,
    ColorScheme
)

from .waterfall_chart import (
    WaterfallChart,
    WaterfallData,
    WaterfallBar,
    BarType
)

from .efficiency_trends import (
    EfficiencyTrends,
    TrendData,
    TrendPoint,
    TrendType
)

from .loss_breakdown import (
    LossBreakdown,
    LossCategory,
    BreakdownChart,
    ChartType
)

from .export import (
    VisualizationExporter,
    ExportFormat,
    export_to_html,
    export_to_png,
    export_to_svg,
    export_to_json,
    export_dashboard
)

__all__ = [
    # Sankey Engine
    "SankeyEngine",
    "SankeyDiagram",
    "SankeyNode",
    "SankeyLink",
    "NodeType",
    "ColorScheme",

    # Waterfall Chart
    "WaterfallChart",
    "WaterfallData",
    "WaterfallBar",
    "BarType",

    # Efficiency Trends
    "EfficiencyTrends",
    "TrendData",
    "TrendPoint",
    "TrendType",

    # Loss Breakdown
    "LossBreakdown",
    "LossCategory",
    "BreakdownChart",
    "ChartType",

    # Export Utilities
    "VisualizationExporter",
    "ExportFormat",
    "export_to_html",
    "export_to_png",
    "export_to_svg",
    "export_to_json",
    "export_dashboard"
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-009 THERMALIQ Team"
