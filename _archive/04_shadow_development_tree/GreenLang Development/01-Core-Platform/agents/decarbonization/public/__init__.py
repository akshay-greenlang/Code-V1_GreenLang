# -*- coding: utf-8 -*-
"""
GreenLang Public Sector Decarbonization Agents
==============================================

Specialized agents for public sector decarbonization initiatives including
municipal climate action planning, fleet electrification, building efficiency,
street lighting optimization, and sustainable procurement.

Agents:
    GL-DECARB-PUB-001: Municipal Climate Action - City climate plans
    GL-DECARB-PUB-002: Public Fleet Electrification - Government fleet EVs
    GL-DECARB-PUB-003: Public Building Efficiency - Government building efficiency
    GL-DECARB-PUB-004: Street Lighting Optimization - Municipal lighting
    GL-DECARB-PUB-005: Public Procurement Greening - Sustainable public procurement
"""

from greenlang.agents.decarbonization.public.municipal_climate_action import (
    MunicipalClimateActionAgent,
    MunicipalClimateActionInput,
    MunicipalClimateActionOutput,
    ClimateTarget,
    TargetType,
    SectorEmissions,
    ClimateAction,
    ActionCategory,
    ActionStatus,
    ClimateActionPlan,
)

from greenlang.agents.decarbonization.public.fleet_electrification import (
    PublicFleetElectrificationAgent,
    FleetElectrificationInput,
    FleetElectrificationOutput,
    VehicleCategory,
    FuelType,
    FleetVehicle,
    ElectrificationScenario,
    ChargingInfrastructure,
    ElectrificationPlan,
)

from greenlang.agents.decarbonization.public.building_efficiency import (
    PublicBuildingEfficiencyAgent,
    BuildingEfficiencyInput,
    BuildingEfficiencyOutput,
    BuildingType,
    EnergySource,
    PublicBuilding,
    EfficiencyMeasure,
    MeasureCategory,
    BuildingEfficiencyPlan,
)

from greenlang.agents.decarbonization.public.street_lighting import (
    StreetLightingOptimizationAgent,
    StreetLightingInput,
    StreetLightingOutput,
    LightingTechnology,
    ControlStrategy,
    StreetLight,
    LightingZone,
    LightingOptimizationPlan,
)

from greenlang.agents.decarbonization.public.procurement_greening import (
    PublicProcurementGreeningAgent,
    ProcurementGreeningInput,
    ProcurementGreeningOutput,
    ProcurementCategory,
    SustainabilityCriteria,
    ProcurementItem,
    GreenProcurementPolicy,
    SupplierAssessment,
)

__all__ = [
    # Municipal Climate Action
    "MunicipalClimateActionAgent",
    "MunicipalClimateActionInput",
    "MunicipalClimateActionOutput",
    "ClimateTarget",
    "TargetType",
    "SectorEmissions",
    "ClimateAction",
    "ActionCategory",
    "ActionStatus",
    "ClimateActionPlan",
    # Fleet Electrification
    "PublicFleetElectrificationAgent",
    "FleetElectrificationInput",
    "FleetElectrificationOutput",
    "VehicleCategory",
    "FuelType",
    "FleetVehicle",
    "ElectrificationScenario",
    "ChargingInfrastructure",
    "ElectrificationPlan",
    # Building Efficiency
    "PublicBuildingEfficiencyAgent",
    "BuildingEfficiencyInput",
    "BuildingEfficiencyOutput",
    "BuildingType",
    "EnergySource",
    "PublicBuilding",
    "EfficiencyMeasure",
    "MeasureCategory",
    "BuildingEfficiencyPlan",
    # Street Lighting
    "StreetLightingOptimizationAgent",
    "StreetLightingInput",
    "StreetLightingOutput",
    "LightingTechnology",
    "ControlStrategy",
    "StreetLight",
    "LightingZone",
    "LightingOptimizationPlan",
    # Procurement Greening
    "PublicProcurementGreeningAgent",
    "ProcurementGreeningInput",
    "ProcurementGreeningOutput",
    "ProcurementCategory",
    "SustainabilityCriteria",
    "ProcurementItem",
    "GreenProcurementPolicy",
    "SupplierAssessment",
]
