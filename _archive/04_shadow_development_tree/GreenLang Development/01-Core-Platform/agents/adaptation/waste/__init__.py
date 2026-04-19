# -*- coding: utf-8 -*-
"""
GreenLang Adaptation Waste Sector Agents
=========================================

Waste infrastructure climate adaptation and resilience agents.

Agents:
    GL-ADAPT-WST-001 to GL-ADAPT-WST-005
"""

from greenlang.agents.adaptation.waste.agents import (
    LandfillClimateResilienceAgent,
    WasteCollectionResilienceAgent,
    RecyclingFacilityResilienceAgent,
    WastewaterTreatmentResilienceAgent,
    WasteInfrastructureNetworkResilienceAgent,
)

__all__ = [
    "LandfillClimateResilienceAgent",
    "WasteCollectionResilienceAgent",
    "RecyclingFacilityResilienceAgent",
    "WastewaterTreatmentResilienceAgent",
    "WasteInfrastructureNetworkResilienceAgent",
]

AGENT_REGISTRY = {
    "GL-ADAPT-WST-001": LandfillClimateResilienceAgent,
    "GL-ADAPT-WST-002": WasteCollectionResilienceAgent,
    "GL-ADAPT-WST-003": RecyclingFacilityResilienceAgent,
    "GL-ADAPT-WST-004": WastewaterTreatmentResilienceAgent,
    "GL-ADAPT-WST-005": WasteInfrastructureNetworkResilienceAgent,
}
