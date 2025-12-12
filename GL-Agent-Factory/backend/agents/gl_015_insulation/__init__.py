"""
GL-015 INSULSCAN - Insulation Analysis Agent

This module provides the InsulationAnalysisAgent for comprehensive industrial
insulation analysis including heat loss quantification, economic thickness
optimization, thermal imaging integration, and ROI analysis.

The agent implements zero-hallucination calculations using physics-based
thermal engineering formulas following industry standards:
- ASTM C680 Standard Practice for Estimate of Heat Gain/Loss
- ASTM C585 Standard Practice for Inner and Outer Diameters
- ISO 12241 Thermal insulation for building equipment
- NAIMA 3E Plus methodology

Features:
- 50+ insulation material database
- SHAP/LIME explainability for recommendations
- Economic thickness calculations
- Thermal imaging (IR camera) integration
- Zero-hallucination heat loss calculations
- ROI optimization
- SHA-256 provenance tracking

Agent ID: GL-015
Agent Name: INSULSCAN
Version: 1.0.0
Market Size: $3B

Example:
    >>> from gl_015_insulation import InsulationAnalysisAgent
    >>> from gl_015_insulation.schemas import (
    ...     InsulationAnalysisInput, SurfaceGeometry, TemperatureConditions
    ... )
    >>>
    >>> agent = InsulationAnalysisAgent()
    >>> input_data = InsulationAnalysisInput(
    ...     analysis_id="INS-001",
    ...     geometry=SurfaceGeometry(
    ...         surface_type="pipe",
    ...         outer_diameter_m=0.1,
    ...         length_m=100
    ...     ),
    ...     temperature=TemperatureConditions(
    ...         process_temp_c=180,
    ...         ambient_temp_c=25
    ...     )
    ... )
    >>> result = agent.run(input_data)
    >>> print(f"Heat loss: {result.current_heat_loss.heat_loss_w:.0f} W")
    >>> print(f"Economic thickness: {result.economic_thickness.economic_thickness_mm:.0f} mm")
"""

from .agent import InsulationAnalysisAgent

from .schemas import (
    # Input/Output models
    InsulationAnalysisInput,
    InsulationAnalysisOutput,
    AgentConfig,
    # Geometry models
    SurfaceType,
    SurfaceGeometry,
    # Temperature models
    TemperatureConditions,
    # IR Camera models
    IRCameraData,
    # Material models
    InsulationMaterialSpec,
    InsulationCondition,
    # Economic models
    EconomicParameters,
    EconomicThicknessResult,
    # Environment models
    EnvironmentalConditions,
    # Result models
    HeatLossQuantification,
    InsulationRecommendation,
    MaintenancePriority,
    MaterialComparison,
    ThermalMapPoint,
    # Explainability
    ExplainabilityReport,
    ExplainabilityFactor,
)

# Agent metadata
AGENT_ID = "GL-015"
AGENT_NAME = "INSULSCAN"
VERSION = "1.0.0"
DESCRIPTION = "Insulation Analysis Agent for heat loss quantification and economic optimization"
CATEGORY = "Energy Conservation"
AGENT_TYPE = "Monitor"
PRIORITY = "P2"
MARKET_SIZE = "$3B"

__all__ = [
    # Agent class
    "InsulationAnalysisAgent",
    # Input/Output models
    "InsulationAnalysisInput",
    "InsulationAnalysisOutput",
    "AgentConfig",
    # Geometry models
    "SurfaceType",
    "SurfaceGeometry",
    # Temperature models
    "TemperatureConditions",
    # IR Camera models
    "IRCameraData",
    # Material models
    "InsulationMaterialSpec",
    "InsulationCondition",
    # Economic models
    "EconomicParameters",
    "EconomicThicknessResult",
    # Environment models
    "EnvironmentalConditions",
    # Result models
    "HeatLossQuantification",
    "InsulationRecommendation",
    "MaintenancePriority",
    "MaterialComparison",
    "ThermalMapPoint",
    # Explainability
    "ExplainabilityReport",
    "ExplainabilityFactor",
    # Metadata
    "AGENT_ID",
    "AGENT_NAME",
    "VERSION",
    "DESCRIPTION",
    "CATEGORY",
    "AGENT_TYPE",
    "PRIORITY",
    "MARKET_SIZE",
]
