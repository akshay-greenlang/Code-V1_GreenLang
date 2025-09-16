"""
GreenLang v0.1: Infrastructure for Climate Intelligence
========================================================

GreenLang is now pure infrastructure. Domain logic lives in packs.
Platform = SDK/CLI/Runtime + Hub + Policy/Provenance

Transitional compatibility layer for v0.0.1 code.
"""

from ._version import __version__

__author__ = "GreenLang Team"
__email__ = "team@greenlang.io"
__license__ = "MIT"

# Import v0.1 infrastructure (from core/greenlang)
try:
    from core.greenlang.sdk.base import Agent, Pipeline, Connector, Dataset, Report
    from core.greenlang.packs.registry import PackRegistry
    from core.greenlang.packs.loader import PackLoader
    from core.greenlang.runtime.executor import Executor
    from core.greenlang.policy.enforcer import PolicyEnforcer
    NEW_ARCHITECTURE = True
except ImportError:
    NEW_ARCHITECTURE = False
    # Fallback - keep old imports for compatibility
    pass

# Transitional: Keep old agent imports for backward compatibility
# These will be moved to packs in v0.1+ 
from greenlang.agents import (
    FuelAgent,
    CarbonAgent,
    InputValidatorAgent,
    ReportAgent,
    BenchmarkAgent,
    BoilerAgent,
    GridFactorAgent,
    BuildingProfileAgent,
    IntensityAgent,
    RecommendationAgent,
    # Climatenza AI Agents
    SiteInputAgent,
    SolarResourceAgent,
    LoadProfileAgent,
    FieldLayoutAgent,
    EnergyBalanceAgent
)
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow

__all__ = [
    # v0.1 Core Infrastructure (when available)
    "Agent" if NEW_ARCHITECTURE else "FuelAgent",
    "Pipeline" if NEW_ARCHITECTURE else "Workflow",
    "PackRegistry" if NEW_ARCHITECTURE else "Orchestrator",
    "PackLoader" if NEW_ARCHITECTURE else "CarbonAgent",
    
    # Legacy v0.0.1 agents (for compatibility)
    "FuelAgent",
    "CarbonAgent", 
    "InputValidatorAgent",
    "ReportAgent",
    "BenchmarkAgent",
    "BoilerAgent",
    "GridFactorAgent",
    "BuildingProfileAgent",
    "IntensityAgent",
    "RecommendationAgent",
    # Climatenza AI Agents
    "SiteInputAgent",
    "SolarResourceAgent",
    "LoadProfileAgent",
    "FieldLayoutAgent",
    "EnergyBalanceAgent",
    # Core Components
    "Orchestrator",
    "Workflow",
]

# Display migration notice
import warnings
if not NEW_ARCHITECTURE:
    warnings.warn(
        "GreenLang v0.1 introduces a new architecture. "
        "Domain agents will move to packs. "
        "See https://greenlang.io/migration for details.",
        DeprecationWarning,
        stacklevel=2
    )