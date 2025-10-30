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
    elif name == "DecarbonizationRoadmapAgentAI":
        from greenlang.agents.decarbonization_roadmap_agent_ai import DecarbonizationRoadmapAgentAI

        return DecarbonizationRoadmapAgentAI
    elif name == "ThermalStorageAgent_AI":
        from greenlang.agents.thermal_storage_agent_ai import ThermalStorageAgent_AI

        return ThermalStorageAgent_AI
    elif name == "FuelAgentAI":
        from greenlang.agents.fuel_agent_ai import FuelAgentAI

        return FuelAgentAI
    elif name == "FuelAgentAI_v2":
        from greenlang.agents.fuel_agent_ai_v2 import FuelAgentAI_v2

        return FuelAgentAI_v2
    raise AttributeError(f"module 'greenlang.agents' has no attribute '{name}'")


__all__ = [
    "BaseAgent",
    "FuelAgent",
    "FuelAgentAI",
    "FuelAgentAI_v2",
    "BoilerAgent",
    "CarbonAgent",
    "InputValidatorAgent",
    "ReportAgent",
    "BenchmarkAgent",
    "GridFactorAgent",
    "BuildingProfileAgent",
    "IntensityAgent",
    "RecommendationAgent",
    "SiteInputAgent",
    "SolarResourceAgent",
    "LoadProfileAgent",
    "FieldLayoutAgent",
    "EnergyBalanceAgent",
    "IndustrialProcessHeatAgent_AI",
    "BoilerReplacementAgent_AI",
    "IndustrialHeatPumpAgent_AI",
    "DecarbonizationRoadmapAgentAI",
    "ThermalStorageAgent_AI",
]
