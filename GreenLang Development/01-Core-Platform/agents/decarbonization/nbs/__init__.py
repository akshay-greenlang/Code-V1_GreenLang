# -*- coding: utf-8 -*-
"""
GreenLang Decarbonization NBS (Nature-Based Solutions) Agents
=============================================================

This package contains agents for planning and implementing nature-based
solutions for carbon removal and emissions reduction.

Agents:
    GL-DECARB-NBS-001: Afforestation Planner - Tree planting planning
    GL-DECARB-NBS-002: Reforestation Planner - Forest restoration planning
    GL-DECARB-NBS-003: Soil Carbon Enhancement - Soil carbon strategies
    GL-DECARB-NBS-004: Wetland Restoration - Wetland restoration planning
    GL-DECARB-NBS-005: Blue Carbon Projects - Coastal carbon projects
    GL-DECARB-NBS-006: Avoided Deforestation (REDD+) - REDD+ project planning
    GL-DECARB-NBS-007: Urban Green Infrastructure - Urban greening strategies
    GL-DECARB-NBS-008: Regenerative Agriculture - Regenerative practices planning

All agents follow the RECOMMENDATION PATH pattern with AI-assisted
planning capabilities while maintaining deterministic cost and carbon
calculations.

Author: GreenLang Team
Version: 1.0.0
"""

from greenlang.agents.decarbonization.nbs.afforestation_planner import (
    AfforestationPlannerAgent,
    AfforestationInput,
    AfforestationOutput,
    AfforestationPlan,
)

from greenlang.agents.decarbonization.nbs.reforestation_planner import (
    ReforestationPlannerAgent,
    ReforestationInput,
    ReforestationOutput,
    ReforestationPlan,
)

from greenlang.agents.decarbonization.nbs.soil_carbon_enhancement import (
    SoilCarbonEnhancementAgent,
    SoilEnhancementInput,
    SoilEnhancementOutput,
    SoilEnhancementStrategy,
)

from greenlang.agents.decarbonization.nbs.wetland_restoration import (
    WetlandRestorationAgent,
    WetlandRestorationInput,
    WetlandRestorationOutput,
    WetlandRestorationPlan,
)

from greenlang.agents.decarbonization.nbs.blue_carbon_projects import (
    BlueCarbonProjectsAgent,
    BlueCarbonProjectInput,
    BlueCarbonProjectOutput,
    BlueCarbonProjectPlan,
)

from greenlang.agents.decarbonization.nbs.avoided_deforestation import (
    AvoidedDeforestationAgent,
    REDDPlusInput,
    REDDPlusOutput,
    REDDPlusPlan,
)

from greenlang.agents.decarbonization.nbs.urban_green_infrastructure import (
    UrbanGreenInfrastructureAgent,
    UrbanGreenInput,
    UrbanGreenOutput,
    UrbanGreenPlan,
)

from greenlang.agents.decarbonization.nbs.regenerative_agriculture import (
    RegenerativeAgricultureAgent,
    RegenerativeAgInput,
    RegenerativeAgOutput,
    RegenerativeAgPlan,
)

__all__ = [
    # Afforestation Planner (GL-DECARB-NBS-001)
    "AfforestationPlannerAgent",
    "AfforestationInput",
    "AfforestationOutput",
    "AfforestationPlan",
    # Reforestation Planner (GL-DECARB-NBS-002)
    "ReforestationPlannerAgent",
    "ReforestationInput",
    "ReforestationOutput",
    "ReforestationPlan",
    # Soil Carbon Enhancement (GL-DECARB-NBS-003)
    "SoilCarbonEnhancementAgent",
    "SoilEnhancementInput",
    "SoilEnhancementOutput",
    "SoilEnhancementStrategy",
    # Wetland Restoration (GL-DECARB-NBS-004)
    "WetlandRestorationAgent",
    "WetlandRestorationInput",
    "WetlandRestorationOutput",
    "WetlandRestorationPlan",
    # Blue Carbon Projects (GL-DECARB-NBS-005)
    "BlueCarbonProjectsAgent",
    "BlueCarbonProjectInput",
    "BlueCarbonProjectOutput",
    "BlueCarbonProjectPlan",
    # Avoided Deforestation (GL-DECARB-NBS-006)
    "AvoidedDeforestationAgent",
    "REDDPlusInput",
    "REDDPlusOutput",
    "REDDPlusPlan",
    # Urban Green Infrastructure (GL-DECARB-NBS-007)
    "UrbanGreenInfrastructureAgent",
    "UrbanGreenInput",
    "UrbanGreenOutput",
    "UrbanGreenPlan",
    # Regenerative Agriculture (GL-DECARB-NBS-008)
    "RegenerativeAgricultureAgent",
    "RegenerativeAgInput",
    "RegenerativeAgOutput",
    "RegenerativeAgPlan",
]

__version__ = "1.0.0"
