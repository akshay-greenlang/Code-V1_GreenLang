# -*- coding: utf-8 -*-
"""
GreenLang Public Sector Adaptation Agents
=========================================

Specialized agents for public sector climate adaptation including
urban heat action, flood response, infrastructure protection,
public health, emergency services, and community resilience.

Agents:
    GL-ADAPT-PUB-001: Urban Heat Action - Heat wave preparedness
    GL-ADAPT-PUB-002: Flood Response Planning - Flood emergency planning
    GL-ADAPT-PUB-003: Critical Infrastructure Protection - Infrastructure resilience
    GL-ADAPT-PUB-004: Public Health & Climate - Health adaptation
    GL-ADAPT-PUB-005: Emergency Services Adaptation - Emergency response
    GL-ADAPT-PUB-006: Community Resilience Planner - Community adaptation
"""

from greenlang.agents.adaptation.public.urban_heat import (
    UrbanHeatActionAgent,
    UrbanHeatActionInput,
    UrbanHeatActionOutput,
    HeatVulnerabilityZone,
    CoolingCenter,
    HeatActionPlan,
)

from greenlang.agents.adaptation.public.flood_response import (
    FloodResponsePlanningAgent,
    FloodResponseInput,
    FloodResponseOutput,
    FloodZone,
    FloodRiskLevel,
    EvacuationRoute,
    FloodResponsePlan,
)

from greenlang.agents.adaptation.public.infrastructure_protection import (
    CriticalInfrastructureProtectionAgent,
    InfrastructureProtectionInput,
    InfrastructureProtectionOutput,
    CriticalAsset,
    AssetCategory,
    ClimateHazard,
    ResilienceAssessment,
)

from greenlang.agents.adaptation.public.public_health import (
    PublicHealthClimateAgent,
    PublicHealthInput,
    PublicHealthOutput,
    HealthOutcome,
    VulnerablePopulation,
    HealthAdaptationPlan,
)

from greenlang.agents.adaptation.public.emergency_services import (
    EmergencyServicesAdaptationAgent,
    EmergencyServicesInput,
    EmergencyServicesOutput,
    EmergencyResource,
    ResponseCapability,
    EmergencyServicesPlan,
)

from greenlang.agents.adaptation.public.community_resilience import (
    CommunityResiliencePlannerAgent,
    CommunityResilienceInput,
    CommunityResilienceOutput,
    CommunityAsset,
    ResilienceIndicator,
    CommunityResiliencePlan,
)

__all__ = [
    # Urban Heat Action
    "UrbanHeatActionAgent",
    "UrbanHeatActionInput",
    "UrbanHeatActionOutput",
    "HeatVulnerabilityZone",
    "CoolingCenter",
    "HeatActionPlan",
    # Flood Response
    "FloodResponsePlanningAgent",
    "FloodResponseInput",
    "FloodResponseOutput",
    "FloodZone",
    "FloodRiskLevel",
    "EvacuationRoute",
    "FloodResponsePlan",
    # Infrastructure Protection
    "CriticalInfrastructureProtectionAgent",
    "InfrastructureProtectionInput",
    "InfrastructureProtectionOutput",
    "CriticalAsset",
    "AssetCategory",
    "ClimateHazard",
    "ResilienceAssessment",
    # Public Health
    "PublicHealthClimateAgent",
    "PublicHealthInput",
    "PublicHealthOutput",
    "HealthOutcome",
    "VulnerablePopulation",
    "HealthAdaptationPlan",
    # Emergency Services
    "EmergencyServicesAdaptationAgent",
    "EmergencyServicesInput",
    "EmergencyServicesOutput",
    "EmergencyResource",
    "ResponseCapability",
    "EmergencyServicesPlan",
    # Community Resilience
    "CommunityResiliencePlannerAgent",
    "CommunityResilienceInput",
    "CommunityResilienceOutput",
    "CommunityAsset",
    "ResilienceIndicator",
    "CommunityResiliencePlan",
]
