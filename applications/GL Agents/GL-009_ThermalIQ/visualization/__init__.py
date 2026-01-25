"""
ThermalIQ Visualization Module

Provides interactive visualizations for thermal system analyses
including Sankey diagrams, property plots, and efficiency dashboards.
"""

from .sankey_generator import (
    SankeyDiagramGenerator,
    SankeyDiagram,
    SankeyNode,
    SankeyLink,
    ColorScheme
)
from .property_plots import (
    FluidPropertyPlotter,
    PropertyType,
    ComparisonChart
)
from .efficiency_dashboard import (
    EfficiencyDashboard,
    GaugeConfig,
    WaterfallConfig
)

__all__ = [
    # Sankey Diagrams
    "SankeyDiagramGenerator",
    "SankeyDiagram",
    "SankeyNode",
    "SankeyLink",
    "ColorScheme",
    # Property Plots
    "FluidPropertyPlotter",
    "PropertyType",
    "ComparisonChart",
    # Efficiency Dashboard
    "EfficiencyDashboard",
    "GaugeConfig",
    "WaterfallConfig",
]

__version__ = "1.0.0"
