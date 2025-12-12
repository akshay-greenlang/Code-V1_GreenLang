"""
GL-014 EXCHANGERPRO - Heat Exchanger Optimizer Agent

This module provides the HeatExchangerOptimizerAgent for TEMA-compliant
heat exchanger optimization including thermal analysis, fouling prediction,
and cleaning schedule optimization.

The agent implements epsilon-NTU and LMTD methods for heat transfer analysis,
deterministic fouling models, and cost-optimized cleaning schedules following
zero-hallucination principles with SHA-256 provenance tracking.

Agent ID: GL-014
Agent Name: EXCHANGERPRO
Version: 1.0.0
Priority: P1
Market Size: $6B

Features:
    - TEMA-compliant calculations
    - Epsilon-NTU method for effectiveness
    - LMTD analysis with correction factors
    - Fouling prediction with deterministic models
    - Cleaning schedule optimization
    - UA degradation analysis
    - SHAP/LIME explainability
    - SHA-256 provenance tracking

Example:
    >>> from agents.gl_014_heat_exchanger import HeatExchangerOptimizerAgent
    >>> agent = HeatExchangerOptimizerAgent()
    >>> result = agent.run(input_data)
"""

from .agent import HeatExchangerOptimizerAgent

from .schemas import (
    # Enums
    FlowArrangement,
    ExchangerType,
    FluidCategory,
    FoulingMechanism,
    MaintenanceUrgency,
    FoulingStatus,
    # Input models
    HeatExchangerInput,
    StreamData,
    FluidProperties,
    ExchangerGeometry,
    CleaningHistoryEntry,
    # Output models
    HeatExchangerOutput,
    LMTDAnalysis,
    EffectivenessAnalysis,
    UADegradationAnalysis,
    FoulingPrediction,
    CleaningScheduleRecommendation,
    EfficiencyGains,
    ExplainabilityReport,
    OptimizationRecommendation,
    # Config
    AgentConfig,
)

# Agent metadata
AGENT_ID = "GL-014"
AGENT_NAME = "EXCHANGERPRO"
VERSION = "1.0.0"
DESCRIPTION = "Heat Exchanger Optimizer Agent with TEMA-compliant analysis and fouling prediction"
CATEGORY = "Heat Exchangers"
AGENT_TYPE = "Optimizer"
PRIORITY = "P1"
MARKET_SIZE = "$6B"

__all__ = [
    # Agent class
    "HeatExchangerOptimizerAgent",
    # Enums
    "FlowArrangement",
    "ExchangerType",
    "FluidCategory",
    "FoulingMechanism",
    "MaintenanceUrgency",
    "FoulingStatus",
    # Input models
    "HeatExchangerInput",
    "StreamData",
    "FluidProperties",
    "ExchangerGeometry",
    "CleaningHistoryEntry",
    # Output models
    "HeatExchangerOutput",
    "LMTDAnalysis",
    "EffectivenessAnalysis",
    "UADegradationAnalysis",
    "FoulingPrediction",
    "CleaningScheduleRecommendation",
    "EfficiencyGains",
    "ExplainabilityReport",
    "OptimizationRecommendation",
    # Config
    "AgentConfig",
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
