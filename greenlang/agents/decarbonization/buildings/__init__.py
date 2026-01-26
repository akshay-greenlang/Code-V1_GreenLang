# -*- coding: utf-8 -*-
"""
GreenLang Buildings Sector Decarbonization Agents
===================================================

Decarbonization agents for building sector emission reduction planning.
These agents analyze buildings and recommend strategies to achieve
carbon neutrality through efficiency, electrification, and renewables.

Agents:
    GL-DECARB-BLD-001: Building Energy Efficiency - Retrofit planning
    GL-DECARB-BLD-002: HVAC Decarbonization - Heat pump transition
    GL-DECARB-BLD-003: Building Electrification - All-electric buildings
    GL-DECARB-BLD-004: Embodied Carbon Reduction - Low-carbon materials
    GL-DECARB-BLD-005: Net Zero Building Planner - NZC building planning
    GL-DECARB-BLD-006: Passive Design Optimizer - Passive design strategies
    GL-DECARB-BLD-007: On-site Renewables Planner - Solar/storage planning
    GL-DECARB-BLD-008: District Energy Planner - District heating/cooling
    GL-DECARB-BLD-009: Building Automation Optimizer - BMS optimization
    GL-DECARB-BLD-010: Tenant Engagement Agent - Tenant programs
    GL-DECARB-BLD-011: Portfolio Decarbonization - Multi-building strategy
    GL-DECARB-BLD-012: Green Lease Agent - Green lease provisions

Design Principles:
    - Recommendation pathway: AI-enhanced with deterministic savings
    - Standards-aligned: ASHRAE, LEED, passive house
    - Financial analysis: NPV, payback, ROI calculations
"""

from greenlang.agents.decarbonization.buildings.base import (
    # Base classes
    BuildingDecarbonizationBaseAgent,
    DecarbonizationInput,
    DecarbonizationOutput,
    # Data models
    TechnologySpec,
    FinancialMetrics,
    DecarbonizationMeasure,
    DecarbonizationPathway,
    BuildingBaseline,
    DecarbonizationTarget,
    # Enums
    TechnologyCategory,
    RecommendationPriority,
    ImplementationPhase,
    RiskLevel,
    # Constants
    HEAT_PUMP_COP,
    LED_SAVINGS_PERCENT,
    ENVELOPE_SAVINGS,
    SOLAR_CAPACITY_FACTOR,
)

from greenlang.agents.decarbonization.buildings.energy_efficiency import (
    BuildingEnergyEfficiencyAgent,
    EnergyEfficiencyInput,
    EnergyEfficiencyOutput,
)

from greenlang.agents.decarbonization.buildings.hvac_decarbonization import (
    HVACDecarbonizationAgent,
    HVACDecarbonizationInput,
    HVACDecarbonizationOutput,
)

from greenlang.agents.decarbonization.buildings.additional_agents import (
    # GL-DECARB-BLD-003
    BuildingElectrificationAgent,
    BuildingElectrificationInput,
    BuildingElectrificationOutput,
    # GL-DECARB-BLD-004
    EmbodiedCarbonReductionAgent,
    EmbodiedCarbonInput,
    EmbodiedCarbonOutput,
    # GL-DECARB-BLD-005
    NetZeroBuildingPlannerAgent,
    NetZeroPlannerInput,
    NetZeroPlannerOutput,
    # GL-DECARB-BLD-006
    PassiveDesignOptimizerAgent,
    PassiveDesignInput,
    PassiveDesignOutput,
    # GL-DECARB-BLD-007
    OnsiteRenewablesPlannerAgent,
    RenewablesPlannerInput,
    RenewablesPlannerOutput,
    # GL-DECARB-BLD-008
    DistrictEnergyPlannerAgent,
    DistrictEnergyInput,
    DistrictEnergyOutput,
    # GL-DECARB-BLD-009
    BuildingAutomationOptimizerAgent,
    BuildingAutomationInput,
    BuildingAutomationOutput,
    # GL-DECARB-BLD-010
    TenantEngagementAgent,
    TenantEngagementInput,
    TenantEngagementOutput,
    # GL-DECARB-BLD-011
    PortfolioDecarbonizationAgent,
    PortfolioDecarbonizationInput,
    PortfolioDecarbonizationOutput,
    PortfolioBuilding,
    # GL-DECARB-BLD-012
    GreenLeaseAgent,
    GreenLeaseInput,
    GreenLeaseOutput,
)

__all__ = [
    # Base classes
    "BuildingDecarbonizationBaseAgent",
    "DecarbonizationInput",
    "DecarbonizationOutput",
    # Data models
    "TechnologySpec",
    "FinancialMetrics",
    "DecarbonizationMeasure",
    "DecarbonizationPathway",
    "BuildingBaseline",
    "DecarbonizationTarget",
    # Enums
    "TechnologyCategory",
    "RecommendationPriority",
    "ImplementationPhase",
    "RiskLevel",
    # Constants
    "HEAT_PUMP_COP",
    "LED_SAVINGS_PERCENT",
    "ENVELOPE_SAVINGS",
    "SOLAR_CAPACITY_FACTOR",
    # GL-DECARB-BLD-001: Building Energy Efficiency
    "BuildingEnergyEfficiencyAgent",
    "EnergyEfficiencyInput",
    "EnergyEfficiencyOutput",
    # GL-DECARB-BLD-002: HVAC Decarbonization
    "HVACDecarbonizationAgent",
    "HVACDecarbonizationInput",
    "HVACDecarbonizationOutput",
    # GL-DECARB-BLD-003: Building Electrification
    "BuildingElectrificationAgent",
    "BuildingElectrificationInput",
    "BuildingElectrificationOutput",
    # GL-DECARB-BLD-004: Embodied Carbon Reduction
    "EmbodiedCarbonReductionAgent",
    "EmbodiedCarbonInput",
    "EmbodiedCarbonOutput",
    # GL-DECARB-BLD-005: Net Zero Building Planner
    "NetZeroBuildingPlannerAgent",
    "NetZeroPlannerInput",
    "NetZeroPlannerOutput",
    # GL-DECARB-BLD-006: Passive Design Optimizer
    "PassiveDesignOptimizerAgent",
    "PassiveDesignInput",
    "PassiveDesignOutput",
    # GL-DECARB-BLD-007: On-site Renewables Planner
    "OnsiteRenewablesPlannerAgent",
    "RenewablesPlannerInput",
    "RenewablesPlannerOutput",
    # GL-DECARB-BLD-008: District Energy Planner
    "DistrictEnergyPlannerAgent",
    "DistrictEnergyInput",
    "DistrictEnergyOutput",
    # GL-DECARB-BLD-009: Building Automation Optimizer
    "BuildingAutomationOptimizerAgent",
    "BuildingAutomationInput",
    "BuildingAutomationOutput",
    # GL-DECARB-BLD-010: Tenant Engagement Agent
    "TenantEngagementAgent",
    "TenantEngagementInput",
    "TenantEngagementOutput",
    # GL-DECARB-BLD-011: Portfolio Decarbonization
    "PortfolioDecarbonizationAgent",
    "PortfolioDecarbonizationInput",
    "PortfolioDecarbonizationOutput",
    "PortfolioBuilding",
    # GL-DECARB-BLD-012: Green Lease Agent
    "GreenLeaseAgent",
    "GreenLeaseInput",
    "GreenLeaseOutput",
]
