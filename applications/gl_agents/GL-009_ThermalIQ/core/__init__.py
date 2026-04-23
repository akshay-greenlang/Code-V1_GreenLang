"""
GL-009 THERMALIQ - Core Module

Core components for the Thermal Fluid Analyzer agent including
configuration, schemas, handlers, and orchestration.

This module provides the foundational classes and functions for:
    - Thermal efficiency calculations
    - Exergy analysis
    - Fluid property management
    - Sankey diagram generation
    - Explainability integration
"""

from .config import (
    ThermalIQConfig,
    CalculationMode,
    FluidConfig,
    SafetyConfig,
    ExplainabilityConfig,
    FluidPhase,
    FluidLibraryType,
)
from .schemas import (
    ThermalAnalysisInput,
    ThermalAnalysisOutput,
    FluidProperties,
    ExergyResult,
    SankeyData,
    SankeyNode,
    SankeyLink,
    ExplainabilityReport,
    FeatureImportance,
    LIMEExplanation,
    Recommendation,
    ProvenanceRecord,
    OperatingConditions,
    CalculationEvent,
    AgentStatus,
    HealthCheckResponse,
)
from .orchestrator import ThermalIQOrchestrator
from .handlers import (
    AnalysisHandler,
    FluidPropertyHandler,
    SankeyHandler,
    ExplainabilityHandler,
)

__all__ = [
    # Configuration
    "ThermalIQConfig",
    "CalculationMode",
    "FluidConfig",
    "SafetyConfig",
    "ExplainabilityConfig",
    "FluidPhase",
    "FluidLibraryType",
    # Schemas - Input/Output
    "ThermalAnalysisInput",
    "ThermalAnalysisOutput",
    "FluidProperties",
    "ExergyResult",
    "OperatingConditions",
    # Schemas - Sankey
    "SankeyData",
    "SankeyNode",
    "SankeyLink",
    # Schemas - Explainability
    "ExplainabilityReport",
    "FeatureImportance",
    "LIMEExplanation",
    "Recommendation",
    # Schemas - Audit/Status
    "ProvenanceRecord",
    "CalculationEvent",
    "AgentStatus",
    "HealthCheckResponse",
    # Orchestrator
    "ThermalIQOrchestrator",
    # Handlers
    "AnalysisHandler",
    "FluidPropertyHandler",
    "SankeyHandler",
    "ExplainabilityHandler",
]
