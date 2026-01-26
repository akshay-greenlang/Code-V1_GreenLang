# -*- coding: utf-8 -*-
"""
GreenLang Adaptation Transport Sector Agents
=============================================

Transport infrastructure climate adaptation agents.

Agents:
    GL-ADAPT-TRN-001 to GL-ADAPT-TRN-007
"""

from greenlang.agents.adaptation.transport.agents import (
    RoadInfrastructureResilienceAgent,
    AirportResilienceAgent,
    PortResilienceAgent,
    RailInfrastructureResilienceAgent,
    FleetClimateResilienceAgent,
    SupplyChainResilienceAgent,
    TransportNetworkResilienceAgent,
)

__all__ = [
    "RoadInfrastructureResilienceAgent",
    "AirportResilienceAgent",
    "PortResilienceAgent",
    "RailInfrastructureResilienceAgent",
    "FleetClimateResilienceAgent",
    "SupplyChainResilienceAgent",
    "TransportNetworkResilienceAgent",
]

AGENT_REGISTRY = {
    "GL-ADAPT-TRN-001": RoadInfrastructureResilienceAgent,
    "GL-ADAPT-TRN-002": AirportResilienceAgent,
    "GL-ADAPT-TRN-003": PortResilienceAgent,
    "GL-ADAPT-TRN-004": RailInfrastructureResilienceAgent,
    "GL-ADAPT-TRN-005": FleetClimateResilienceAgent,
    "GL-ADAPT-TRN-006": SupplyChainResilienceAgent,
    "GL-ADAPT-TRN-007": TransportNetworkResilienceAgent,
}
