# -*- coding: utf-8 -*-
"""
GreenLang Decarbonization Energy Sector Agents
===============================================

This package contains agents for energy sector decarbonization planning,
strategy development, and transition management.

Agents:
    GL-DECARB-ENE-001: GridDecarbonizationPlannerAgent - Grid decarbonization
    GL-DECARB-ENE-002: RenewableIntegrationAgent - Renewable energy planning
    GL-DECARB-ENE-003: StorageOptimizationAgent - Energy storage optimization
    GL-DECARB-ENE-004: DemandFlexibilityAgent - Demand-side management
    GL-DECARB-ENE-005: HydrogenStrategyAgent - Hydrogen transition planning
    GL-DECARB-ENE-006: NuclearAssessmentAgent - Nuclear energy assessment
    GL-DECARB-ENE-007: CCUSPowerAgent - Carbon capture for power
    GL-DECARB-ENE-008: DistributedGenerationAgent - DER planning
    GL-DECARB-ENE-009: GridModernizationAgent - Grid modernization
    GL-DECARB-ENE-010: JustTransitionPlannerAgent - Workforce transition

All agents follow the RECOMMENDATION PATH pattern with:
- AI-powered strategic analysis
- Deterministic financial calculations
- RAG-enhanced knowledge retrieval
"""

from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent

# Schemas
from greenlang.agents.decarbonization.energy.schemas import (
    DecarbonizationPathway,
    TimeHorizon,
    TechnologyReadinessLevel,
    RenewableTechnology,
    StorageApplication,
    HydrogenApplication,
    NuclearTechnology,
    CCUSTechnology,
    DecarbonizationBaseInput,
    DecarbonizationBaseOutput,
    GridDecarbonizationInput,
    GridDecarbonizationOutput,
    RenewableIntegrationInput,
    RenewableIntegrationOutput,
    StorageOptimizationInput,
    StorageOptimizationOutput,
    DemandFlexibilityInput,
    DemandFlexibilityOutput,
    HydrogenStrategyInput,
    HydrogenStrategyOutput,
    NuclearAssessmentInput,
    NuclearAssessmentOutput,
    CCUSPowerInput,
    CCUSPowerOutput,
    DistributedGenerationInput,
    DistributedGenerationOutput,
    GridModernizationInput,
    GridModernizationOutput,
    JustTransitionInput,
    JustTransitionOutput,
)

# Agents
from greenlang.agents.decarbonization.energy.grid_decarbonization_planner import (
    GridDecarbonizationPlannerAgent,
)
from greenlang.agents.decarbonization.energy.renewable_integration import (
    RenewableIntegrationAgent,
)
from greenlang.agents.decarbonization.energy.storage_optimization import (
    StorageOptimizationAgent,
)
from greenlang.agents.decarbonization.energy.demand_flexibility import (
    DemandFlexibilityAgent,
)
from greenlang.agents.decarbonization.energy.hydrogen_strategy import (
    HydrogenStrategyAgent,
)
from greenlang.agents.decarbonization.energy.nuclear_assessment import (
    NuclearAssessmentAgent,
)
from greenlang.agents.decarbonization.energy.ccus_power import (
    CCUSPowerAgent,
)
from greenlang.agents.decarbonization.energy.distributed_generation import (
    DistributedGenerationAgent,
)
from greenlang.agents.decarbonization.energy.grid_modernization import (
    GridModernizationAgent,
)
from greenlang.agents.decarbonization.energy.just_transition_planner import (
    JustTransitionPlannerAgent,
)

__version__ = "1.0.0"

__all__ = [
    # Base
    "DecarbonizationEnergyBaseAgent",
    # Enums
    "DecarbonizationPathway",
    "TimeHorizon",
    "TechnologyReadinessLevel",
    "RenewableTechnology",
    "StorageApplication",
    "HydrogenApplication",
    "NuclearTechnology",
    "CCUSTechnology",
    # Base models
    "DecarbonizationBaseInput",
    "DecarbonizationBaseOutput",
    # Grid Decarbonization (GL-DECARB-ENE-001)
    "GridDecarbonizationPlannerAgent",
    "GridDecarbonizationInput",
    "GridDecarbonizationOutput",
    # Renewable Integration (GL-DECARB-ENE-002)
    "RenewableIntegrationAgent",
    "RenewableIntegrationInput",
    "RenewableIntegrationOutput",
    # Storage Optimization (GL-DECARB-ENE-003)
    "StorageOptimizationAgent",
    "StorageOptimizationInput",
    "StorageOptimizationOutput",
    # Demand Flexibility (GL-DECARB-ENE-004)
    "DemandFlexibilityAgent",
    "DemandFlexibilityInput",
    "DemandFlexibilityOutput",
    # Hydrogen Strategy (GL-DECARB-ENE-005)
    "HydrogenStrategyAgent",
    "HydrogenStrategyInput",
    "HydrogenStrategyOutput",
    # Nuclear Assessment (GL-DECARB-ENE-006)
    "NuclearAssessmentAgent",
    "NuclearAssessmentInput",
    "NuclearAssessmentOutput",
    # CCUS Power (GL-DECARB-ENE-007)
    "CCUSPowerAgent",
    "CCUSPowerInput",
    "CCUSPowerOutput",
    # Distributed Generation (GL-DECARB-ENE-008)
    "DistributedGenerationAgent",
    "DistributedGenerationInput",
    "DistributedGenerationOutput",
    # Grid Modernization (GL-DECARB-ENE-009)
    "GridModernizationAgent",
    "GridModernizationInput",
    "GridModernizationOutput",
    # Just Transition (GL-DECARB-ENE-010)
    "JustTransitionPlannerAgent",
    "JustTransitionInput",
    "JustTransitionOutput",
]

# Agent registry
AGENT_REGISTRY = {
    "GL-DECARB-ENE-001": GridDecarbonizationPlannerAgent,
    "GL-DECARB-ENE-002": RenewableIntegrationAgent,
    "GL-DECARB-ENE-003": StorageOptimizationAgent,
    "GL-DECARB-ENE-004": DemandFlexibilityAgent,
    "GL-DECARB-ENE-005": HydrogenStrategyAgent,
    "GL-DECARB-ENE-006": NuclearAssessmentAgent,
    "GL-DECARB-ENE-007": CCUSPowerAgent,
    "GL-DECARB-ENE-008": DistributedGenerationAgent,
    "GL-DECARB-ENE-009": GridModernizationAgent,
    "GL-DECARB-ENE-010": JustTransitionPlannerAgent,
}


def get_agent(agent_id: str):
    """Get a Decarbonization Energy agent by ID."""
    if agent_id not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent: {agent_id}")
    return AGENT_REGISTRY[agent_id]
