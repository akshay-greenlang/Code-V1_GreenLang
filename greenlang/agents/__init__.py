# Lazy imports to avoid requiring analytics dependencies at import time
# Import base agent directly as it has no heavy dependencies
from greenlang.agents.base import BaseAgent


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
    raise AttributeError(f"module 'greenlang.agents' has no attribute '{name}'")


__all__ = [
    "BaseAgent",
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
]
