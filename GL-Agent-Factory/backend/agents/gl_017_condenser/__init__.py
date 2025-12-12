"""
GL-017 CONDENSYNC - Condenser Optimization Agent

This module provides the CondenserOptimizationAgent for optimizing steam
surface condenser performance following HEI Standards.

The agent implements HEI-compliant heat transfer calculations, cleanliness
factor tracking, vacuum optimization, and fouling prediction with
SHAP/LIME-style explainability - all using zero-hallucination principles
with deterministic calculations only.

Agent ID: GL-017
Agent Name: CONDENSYNC
Category: Steam Systems
Type: Optimizer
Priority: P2
Market Size: $4B

Features:
- HEI Standards compliance for condenser performance
- Cleanliness factor tracking and trending
- SHAP/LIME explainability for optimization decisions
- Vacuum system optimization
- Fouling prediction and cleaning schedule optimization
- Zero-hallucination heat transfer calculations
- SHA-256 provenance tracking

Example:
    >>> from gl_017_condenser import CondenserOptimizationAgent
    >>> agent = CondenserOptimizationAgent(config)
    >>> result = agent.run(input_data)
"""

from .agent import CondenserOptimizationAgent

from .schemas import (
    # Input models
    CondenserInput,
    CondenserDesignData,
    CoolingWaterData,
    VacuumData,
    AirLeakageData,
    TubeFoulingData,
    # Output models
    CondenserOutput,
    HeatTransferAnalysis,
    CleanlinessAnalysis,
    VacuumAnalysis,
    FoulingAnalysis,
    AirLeakageAnalysis,
    OptimizationRecommendation,
    CleaningSchedule,
    ExplainabilityReport,
    # Config
    AgentConfig,
    # Enums
    TubeMaterial,
    CoolingWaterSource,
    FoulingMechanism,
    CleaningMethod,
    OptimizationPriority,
)

# Agent metadata
AGENT_ID = "GL-017"
AGENT_NAME = "CONDENSYNC"
VERSION = "1.0.0"
DESCRIPTION = "Condenser Optimization Agent using HEI Standards"
CATEGORY = "Steam Systems"
AGENT_TYPE = "Optimizer"
PRIORITY = "P2"
MARKET_SIZE = "$4B"
STANDARDS = ["HEI", "ASME PTC 12.2", "TEMA"]

__all__ = [
    # Agent class
    "CondenserOptimizationAgent",
    # Input models
    "CondenserInput",
    "CondenserDesignData",
    "CoolingWaterData",
    "VacuumData",
    "AirLeakageData",
    "TubeFoulingData",
    # Output models
    "CondenserOutput",
    "HeatTransferAnalysis",
    "CleanlinessAnalysis",
    "VacuumAnalysis",
    "FoulingAnalysis",
    "AirLeakageAnalysis",
    "OptimizationRecommendation",
    "CleaningSchedule",
    "ExplainabilityReport",
    # Config
    "AgentConfig",
    # Enums
    "TubeMaterial",
    "CoolingWaterSource",
    "FoulingMechanism",
    "CleaningMethod",
    "OptimizationPriority",
    # Metadata
    "AGENT_ID",
    "AGENT_NAME",
    "VERSION",
    "DESCRIPTION",
    "CATEGORY",
    "AGENT_TYPE",
    "PRIORITY",
    "MARKET_SIZE",
    "STANDARDS",
]
