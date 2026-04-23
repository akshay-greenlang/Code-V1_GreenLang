# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Insulation Scanning & Thermal Assessment Agent

An AI agent for comprehensive thermal insulation assessment in process heat
applications. INSULSCAN provides:

- **Heat Loss Analysis**: Calculate heat loss from degraded insulation using
  ASTM C680 standard formulas
- **Hot Spot Detection**: Identify insulation failures from thermal imaging
- **Condition Assessment**: Score insulation condition (0-100) with severity
  classification
- **ROI Calculations**: Payback period, NPV, and lifecycle cost analysis for
  insulation repairs
- **Repair Recommendations**: Prioritized maintenance recommendations with
  cost-benefit analysis

Key Features:
- Zero-hallucination calculations using deterministic physics formulas
- Full provenance tracking with SHA-256 hashing
- ISO 50001 energy management compliance
- Integration with thermal cameras, OPC-UA, and CMMS systems

Example:
    >>> from gl_015_insulscan import InsulscanOrchestrator, InsulscanSettings
    >>> from gl_015_insulscan.core.schemas import InsulationAsset, ThermalMeasurement
    >>>
    >>> # Initialize orchestrator
    >>> settings = InsulscanSettings()
    >>> orchestrator = InsulscanOrchestrator(settings)
    >>>
    >>> # Analyze insulation condition
    >>> asset = InsulationAsset(
    ...     asset_id="PIPE-001",
    ...     surface_type="pipe",
    ...     insulation_type="mineral_wool",
    ...     thickness_mm=50.0,
    ...     operating_temp_c=180.0,
    ...     ambient_temp_c=25.0,
    ...     surface_area_m2=12.5,
    ... )
    >>> result = await orchestrator.analyze_insulation(asset, measurements)
    >>> print(f"Heat Loss: {result.heat_loss.heat_loss_w:.1f} W")
    >>> print(f"Condition Score: {result.condition.condition_score}")

Author: GreenLang AI Agent Workforce
Version: 1.0.0
Agent ID: GL-015
Agent Name: INSULSCAN
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-015"
__agent_name__ = "INSULSCAN"
__author__ = "GreenLang AI Agent Workforce"

# Lazy imports to avoid circular dependencies and improve startup time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.config import InsulscanSettings
    from .core.orchestrator import InsulscanOrchestrator
    from .core.schemas import (
        AnalysisResult,
        HeatLossResult,
        HotSpotDetection,
        InsulationAsset,
        InsulationCondition,
        RepairRecommendation,
        ThermalMeasurement,
    )


def __getattr__(name: str):
    """Lazy import for top-level exports."""
    if name == "InsulscanSettings":
        from .core.config import InsulscanSettings
        return InsulscanSettings
    elif name == "InsulscanOrchestrator":
        from .core.orchestrator import InsulscanOrchestrator
        return InsulscanOrchestrator
    elif name in (
        "AnalysisResult",
        "HeatLossResult",
        "HotSpotDetection",
        "InsulationAsset",
        "InsulationCondition",
        "RepairRecommendation",
        "ThermalMeasurement",
    ):
        from . import core
        return getattr(core.schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # Core classes
    "InsulscanSettings",
    "InsulscanOrchestrator",
    # Schemas
    "InsulationAsset",
    "ThermalMeasurement",
    "HotSpotDetection",
    "InsulationCondition",
    "HeatLossResult",
    "RepairRecommendation",
    "AnalysisResult",
]
