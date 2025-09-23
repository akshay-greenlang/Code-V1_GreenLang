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
    raise AttributeError(f"module 'greenlang.agents' has no attribute '{name}'")

__all__ = [
    "BaseAgent",
    "FuelAgent",
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
]
