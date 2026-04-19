# -*- coding: utf-8 -*-
"""
GreenLang Decarbonization Transport Sector Agents
==================================================

Transport sector decarbonization agents for fleet electrification,
fuel switching, and logistics optimization.

Agents:
    GL-DECARB-TRN-001 to GL-DECARB-TRN-013
"""

from greenlang.agents.decarbonization.transport.agents import (
    RoadFleetElectrificationAgent,
    AviationDecarbonizationAgent,
    MaritimeDecarbonizationAgent,
    RailDecarbonizationAgent,
    LastMileDeliveryAgent,
    EVChargingInfrastructureAgent,
    LogisticsOptimizationAgent,
    BusinessTravelDecarbonizationAgent,
    HydrogenMobilityAgent,
    SustainableFuelAgent,
    ModeShiftPlannerAgent,
    FleetTransitionAgent,
    SupplyChainDecarbonizationAgent,
)

__all__ = [
    "RoadFleetElectrificationAgent",
    "AviationDecarbonizationAgent",
    "MaritimeDecarbonizationAgent",
    "RailDecarbonizationAgent",
    "LastMileDeliveryAgent",
    "EVChargingInfrastructureAgent",
    "LogisticsOptimizationAgent",
    "BusinessTravelDecarbonizationAgent",
    "HydrogenMobilityAgent",
    "SustainableFuelAgent",
    "ModeShiftPlannerAgent",
    "FleetTransitionAgent",
    "SupplyChainDecarbonizationAgent",
]

AGENT_REGISTRY = {
    "GL-DECARB-TRN-001": RoadFleetElectrificationAgent,
    "GL-DECARB-TRN-002": AviationDecarbonizationAgent,
    "GL-DECARB-TRN-003": MaritimeDecarbonizationAgent,
    "GL-DECARB-TRN-004": RailDecarbonizationAgent,
    "GL-DECARB-TRN-005": LastMileDeliveryAgent,
    "GL-DECARB-TRN-006": EVChargingInfrastructureAgent,
    "GL-DECARB-TRN-007": LogisticsOptimizationAgent,
    "GL-DECARB-TRN-008": BusinessTravelDecarbonizationAgent,
    "GL-DECARB-TRN-009": HydrogenMobilityAgent,
    "GL-DECARB-TRN-010": SustainableFuelAgent,
    "GL-DECARB-TRN-011": ModeShiftPlannerAgent,
    "GL-DECARB-TRN-012": FleetTransitionAgent,
    "GL-DECARB-TRN-013": SupplyChainDecarbonizationAgent,
}
