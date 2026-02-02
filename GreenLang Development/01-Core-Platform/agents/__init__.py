# -*- coding: utf-8 -*-
"""
GreenLang Agents Module
========================

This module provides the agent framework and all available agents.

Base Classes:
- BaseAgent: Original base agent class (legacy - do not use for new agents)
- IntelligentAgentBase: NEW - AI-native base class with LLM intelligence
- AgentSpecV2Base: Standardized base class for AgentSpec v2 compliant agents

Intelligence Framework (NEW - Solves Intelligence Paradox):
- IntelligentAgentBase: Base class with built-in LLM intelligence
- IntelligenceMixin: Add intelligence to existing agents (retrofit)
- IntelligenceInterface: Mandatory contract for AI-native agents

Category Mixins (AgentSpec v2 Pattern):
- DeterministicMixin: Zero-hallucination calculation agents
- ReasoningMixin: AI-powered reasoning agents
- InsightMixin: Hybrid calculation + AI agents

Migration Guide:
See greenlang/agents/MIGRATION_TO_AGENTSPECV2.md for migration instructions.

NEW AGENTS: Must extend IntelligentAgentBase, not BaseAgent.
EXISTING AGENTS: Add IntelligenceMixin to retrofit with LLM intelligence.
"""

# Lazy imports to avoid requiring analytics dependencies at import time
# Import base agent directly as it has no heavy dependencies
from greenlang.agents.base import BaseAgent

# =============================================================================
# NEW: Intelligence Framework (Solves the Intelligence Paradox)
# =============================================================================
# IntelligentAgentBase - MANDATORY for all new agents
from greenlang.agents.intelligent_base import (
    IntelligentAgentBase,
    IntelligentAgentConfig,
    IntelligenceLevel,
    IntelligenceMetrics,
    Recommendation,
    Anomaly,
    create_intelligent_agent_config,
)

# IntelligenceMixin - For retrofitting existing agents
from greenlang.agents.intelligence_mixin import (
    IntelligenceMixin,
    IntelligenceConfig,
    retrofit_agent_class,
    create_intelligent_wrapper,
    retrofit_all_agents_in_module,
)

# Intelligence Interface - Mandatory contract for all agents
from greenlang.agents.intelligence_interface import (
    IntelligentAgent,
    IntelligenceCapabilities,
    AgentIntelligenceValidator,
    ValidationResult,
    require_intelligence,
    is_intelligent_agent,
    get_agent_intelligence_level,
)

# Import AgentSpec v2 base classes and mixins
from greenlang.agents.agentspec_v2_base import AgentSpecV2Base, AgentExecutionContext
from greenlang.agents.mixins import (
    DeterministicMixin,
    ReasoningMixin,
    InsightMixin,
    get_category_mixin,
    validate_mixin_usage,
)


# Define lazy imports for agents that require pandas/numpy
def __getattr__(name):
    """Lazy import agents to avoid dependency issues"""
    if name == "FuelAgent":
        from greenlang.agents.fuel_agent import FuelAgent

        return FuelAgent
    elif name == "BoilerAgent":
        from greenlang.agents.boiler_agent import BoilerAgent

        return BoilerAgent
    elif name == "CarbonAgent":
        from greenlang.agents.carbon_agent import CarbonAgent

        return CarbonAgent
    elif name == "InputValidatorAgent":
        from greenlang.agents.validator_agent import InputValidatorAgent

        return InputValidatorAgent
    elif name == "ReportAgent":
        from greenlang.agents.report_agent import ReportAgent

        return ReportAgent
    elif name == "BenchmarkAgent":
        from greenlang.agents.benchmark_agent import BenchmarkAgent

        return BenchmarkAgent
    elif name == "GridFactorAgent":
        from greenlang.agents.grid_factor_agent import GridFactorAgent

        return GridFactorAgent
    elif name == "BuildingProfileAgent":
        from greenlang.agents.building_profile_agent import BuildingProfileAgent

        return BuildingProfileAgent
    elif name == "IntensityAgent":
        from greenlang.agents.intensity_agent import IntensityAgent

        return IntensityAgent
    elif name == "RecommendationAgent":
        from greenlang.agents.recommendation_agent import RecommendationAgent

        return RecommendationAgent
    elif name == "SiteInputAgent":
        from greenlang.agents.site_input_agent import SiteInputAgent

        return SiteInputAgent
    elif name == "SolarResourceAgent":
        from greenlang.agents.solar_resource_agent import SolarResourceAgent

        return SolarResourceAgent
    elif name == "LoadProfileAgent":
        from greenlang.agents.load_profile_agent import LoadProfileAgent

        return LoadProfileAgent
    elif name == "FieldLayoutAgent":
        from greenlang.agents.field_layout_agent import FieldLayoutAgent

        return FieldLayoutAgent
    elif name == "EnergyBalanceAgent":
        from greenlang.agents.energy_balance_agent import EnergyBalanceAgent

        return EnergyBalanceAgent
    elif name == "IndustrialProcessHeatAgent_AI":
        from greenlang.agents.industrial_process_heat_agent_ai import IndustrialProcessHeatAgent_AI

        return IndustrialProcessHeatAgent_AI
    elif name == "BoilerReplacementAgent_AI":
        from greenlang.agents.boiler_replacement_agent_ai import BoilerReplacementAgent_AI

        return BoilerReplacementAgent_AI
    elif name == "IndustrialHeatPumpAgent_AI":
        from greenlang.agents.industrial_heat_pump_agent_ai import IndustrialHeatPumpAgent_AI

        return IndustrialHeatPumpAgent_AI
    # Phase 6 V4 agents with shared tool library
    elif name == "IndustrialHeatPumpAgentAI_V4":
        from greenlang.agents.industrial_heat_pump_agent_ai_v4 import IndustrialHeatPumpAgentAI_V4

        return IndustrialHeatPumpAgentAI_V4
    elif name == "BoilerReplacementAgentAI_V4":
        from greenlang.agents.boiler_replacement_agent_ai_v4 import BoilerReplacementAgentAI_V4

        return BoilerReplacementAgentAI_V4
    elif name == "DecarbonizationRoadmapAgentAI":
        from greenlang.agents.decarbonization_roadmap_agent_ai import DecarbonizationRoadmapAgentAI

        return DecarbonizationRoadmapAgentAI
    elif name == "ThermalStorageAgent_AI":
        from greenlang.agents.thermal_storage_agent_ai import ThermalStorageAgent_AI

        return ThermalStorageAgent_AI
    # DEPRECATED: AI versions of CRITICAL PATH agents
    # These agents have been deprecated for regulatory/compliance calculations
    # Use deterministic versions instead (e.g., FuelAgent, GridFactorAgent)
    elif name == "FuelAgentAI":
        import warnings
        warnings.warn(
            "FuelAgentAI is deprecated for CRITICAL PATH calculations. "
            "Use FuelAgent instead for regulatory/compliance work.",
            DeprecationWarning,
            stacklevel=2
        )
        from greenlang.agents.fuel_agent_ai import FuelAgentAI
        return FuelAgentAI
    elif name == "FuelAgentAI_v2":
        import warnings
        warnings.warn(
            "FuelAgentAI_v2 is deprecated for CRITICAL PATH calculations. "
            "Use FuelAgent instead for regulatory/compliance work.",
            DeprecationWarning,
            stacklevel=2
        )
        from greenlang.agents.fuel_agent_ai_v2 import FuelAgentAI_v2
        return FuelAgentAI_v2
    # ==========================================================================
    # INTELLIGENT AGENTS (AI-Native - Solves Intelligence Paradox)
    # ==========================================================================
    elif name == "IntelligentCarbonAgent":
        from greenlang.agents.carbon_agent_intelligent import IntelligentCarbonAgent
        return IntelligentCarbonAgent
    elif name == "IntelligentFuelAgent":
        from greenlang.agents.fuel_agent_intelligent import IntelligentFuelAgent
        return IntelligentFuelAgent
    elif name == "IntelligentGridFactorAgent":
        from greenlang.agents.grid_factor_agent_intelligent import IntelligentGridFactorAgent
        return IntelligentGridFactorAgent
    elif name == "IntelligentRecommendationAgent":
        from greenlang.agents.recommendation_agent_intelligent import IntelligentRecommendationAgent
        return IntelligentRecommendationAgent
    # Factory functions for intelligent agents
    elif name == "create_intelligent_carbon_agent":
        from greenlang.agents.carbon_agent_intelligent import create_intelligent_carbon_agent
        return create_intelligent_carbon_agent
    elif name == "create_intelligent_fuel_agent":
        from greenlang.agents.fuel_agent_intelligent import create_intelligent_fuel_agent
        return create_intelligent_fuel_agent
    elif name == "create_intelligent_grid_factor_agent":
        from greenlang.agents.grid_factor_agent_intelligent import create_intelligent_grid_factor_agent
        return create_intelligent_grid_factor_agent
    elif name == "create_intelligent_recommendation_agent":
        from greenlang.agents.recommendation_agent_intelligent import create_intelligent_recommendation_agent
        return create_intelligent_recommendation_agent
    raise AttributeError(f"module 'greenlang.agents' has no attribute '{name}'")


__all__ = [
    # Base classes (legacy - do not use for new agents)
    "BaseAgent",

    # ==========================================================================
    # NEW: Intelligence Framework (Solves Intelligence Paradox)
    # ==========================================================================
    # IntelligentAgentBase - MANDATORY for all new agents
    "IntelligentAgentBase",
    "IntelligentAgentConfig",
    "IntelligenceLevel",
    "IntelligenceMetrics",
    "Recommendation",
    "Anomaly",
    "create_intelligent_agent_config",

    # IntelligenceMixin - For retrofitting existing agents
    "IntelligenceMixin",
    "IntelligenceConfig",
    "retrofit_agent_class",
    "create_intelligent_wrapper",
    "retrofit_all_agents_in_module",

    # Intelligence Interface - Mandatory contract
    "IntelligentAgent",
    "IntelligenceCapabilities",
    "AgentIntelligenceValidator",
    "ValidationResult",
    "require_intelligence",
    "is_intelligent_agent",
    "get_agent_intelligence_level",

    # AgentSpec v2 Base Classes
    "AgentSpecV2Base",
    "AgentExecutionContext",

    # Category Mixins (AgentSpec v2 Pattern)
    "DeterministicMixin",
    "ReasoningMixin",
    "InsightMixin",
    "get_category_mixin",
    "validate_mixin_usage",

    # CRITICAL PATH agents (deterministic for regulatory/compliance)
    "FuelAgent",
    "GridFactorAgent",
    "BoilerAgent",
    "CarbonAgent",
    "InputValidatorAgent",
    "ReportAgent",
    "BenchmarkAgent",
    "BuildingProfileAgent",
    "IntensityAgent",
    "RecommendationAgent",
    "SiteInputAgent",
    "SolarResourceAgent",
    "LoadProfileAgent",
    "FieldLayoutAgent",
    "EnergyBalanceAgent",

    # RECOMMENDATION PATH agents (AI for decision support)
    "IndustrialProcessHeatAgent_AI",
    "BoilerReplacementAgent_AI",
    "IndustrialHeatPumpAgent_AI",
    "DecarbonizationRoadmapAgentAI",
    "ThermalStorageAgent_AI",

    # Phase 6 V4 agents with shared tool library
    "IndustrialHeatPumpAgentAI_V4",
    "BoilerReplacementAgentAI_V4",

    # DEPRECATED: AI versions of CRITICAL PATH agents (backward compatibility only)
    # Use deterministic versions (FuelAgent, GridFactorAgent) for regulatory work
    "FuelAgentAI",  # DEPRECATED - use FuelAgent
    "FuelAgentAI_v2",  # DEPRECATED - use FuelAgent

    # ==========================================================================
    # INTELLIGENT AGENTS (AI-Native - Solves Intelligence Paradox)
    # ==========================================================================
    # These are the retrofitted versions with full LLM intelligence
    "IntelligentCarbonAgent",
    "IntelligentFuelAgent",
    "IntelligentGridFactorAgent",
    "IntelligentRecommendationAgent",

    # Factory functions for intelligent agents
    "create_intelligent_carbon_agent",
    "create_intelligent_fuel_agent",
    "create_intelligent_grid_factor_agent",
    "create_intelligent_recommendation_agent",
]
