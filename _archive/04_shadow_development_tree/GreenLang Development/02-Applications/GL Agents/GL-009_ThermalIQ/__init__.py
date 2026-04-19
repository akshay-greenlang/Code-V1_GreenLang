"""
GL-009 THERMALIQ - Thermal Fluid Analyzer Agent

Enterprise-grade thermal fluid analysis agent for industrial heat systems.
Provides thermal efficiency calculations, exergy analysis, Sankey diagram
generation, fluid property lookups, and explainable recommendations.

Agent ID: GL-009
Unique Name: THERMALIQ
Agent Name: ThermalFluidAnalyzer
Type: Calculator (shared calculation library)
Priority: P1
Status: Production

Key Features:
    - Thermal efficiency calculations (deterministic, zero-hallucination)
    - Exergy destruction analysis with improvement recommendations
    - Sankey diagram data generation for energy flow visualization
    - Thermal fluid property library with phase detection
    - SHAP/LIME explainability integration
    - Full SHA-256 provenance tracking
    - Audit logging for regulatory compliance

Standards Compliance:
    - Zero-hallucination calculations (no LLM for numeric values)
    - Deterministic, reproducible outputs
    - SHA-256 provenance tracking
    - ASME PTC 4.1 (Process Heat Performance)
    - ISO 50001:2018 (Energy Management)
    - ASHRAE thermodynamic standards

Reference Equations:
    - Thermal Efficiency: eta = Q_out / Q_in * 100
    - Exergy Input: Ex_in = Q * (1 - T0/T_source)
    - Exergy Destruction: Ex_d = Ex_in - Ex_out
    - Carnot Factor: 1 - T0/T

Copyright (c) 2025 GreenLang. All rights reserved.
"""

__version__ = "1.0.0"
__agent_id__ = "GL-009"
__unique_name__ = "THERMALIQ"
__agent_name__ = "ThermalFluidAnalyzer"
__agent_type__ = "Calculator"
__priority__ = "P1"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.orchestrator import ThermalIQOrchestrator
    from .core.config import ThermalIQConfig, CalculationMode, FluidConfig
    from .core.schemas import (
        ThermalAnalysisInput,
        ThermalAnalysisOutput,
        FluidProperties,
        ExergyResult,
        SankeyData,
        ExplainabilityReport,
    )

__all__ = [
    # Main orchestrator
    "ThermalIQOrchestrator",
    # Configuration
    "ThermalIQConfig",
    "CalculationMode",
    "FluidConfig",
    # Schemas
    "ThermalAnalysisInput",
    "ThermalAnalysisOutput",
    "FluidProperties",
    "ExergyResult",
    "SankeyData",
    "ExplainabilityReport",
    # Metadata
    "__version__",
    "__agent_id__",
    "__unique_name__",
    "__agent_name__",
    "__agent_type__",
    "__priority__",
]
