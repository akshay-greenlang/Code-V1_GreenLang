# -*- coding: utf-8 -*-
"""
GreenLang Visualization Module
==============================

Enterprise-grade visualization tools for energy flows, emissions, and process heat systems.

This module provides:
- Sankey diagram generation for energy balance visualization
- Process heat system flow visualization
- Energy loss tracking and visualization
- Interactive Plotly-based charts
- Zero-hallucination energy balance validation

Example:
    >>> from greenlang.visualization import ProcessHeatSankeyGenerator
    >>> generator = ProcessHeatSankeyGenerator("Boiler Energy Balance", unit="BTU/hr")
    >>> # Add nodes and flows
    >>> figure = generator.generate_figure()
    >>> generator.export_html("boiler_sankey.html")
"""

from .sankey_generator import (
    # Data Classes
    SankeyNode,
    SankeyFlow,
    SankeyDiagramConfig,
    EnergyBalanceResult,
    # Main Generator
    ProcessHeatSankeyGenerator,
    # Factory Functions
    create_boiler_sankey,
    create_furnace_sankey,
    create_heat_recovery_sankey,
    create_steam_system_sankey,
    # Color Scheme
    PROCESS_HEAT_COLORS,
)

__all__ = [
    # Data Classes
    "SankeyNode",
    "SankeyFlow",
    "SankeyDiagramConfig",
    "EnergyBalanceResult",
    # Main Generator
    "ProcessHeatSankeyGenerator",
    # Factory Functions
    "create_boiler_sankey",
    "create_furnace_sankey",
    "create_heat_recovery_sankey",
    "create_steam_system_sankey",
    # Color Scheme
    "PROCESS_HEAT_COLORS",
]

__version__ = "1.0.0"
