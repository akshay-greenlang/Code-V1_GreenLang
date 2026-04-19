# -*- coding: utf-8 -*-
"""
GreenLang Water Adaptation Agents
=================================

This package provides adaptation agents for water sector climate resilience,
including drought risk, flood risk, and water security planning.

Agents:
    GL-ADAPT-WAT-001: WaterScarcityRiskAgent - Drought risk assessment
    GL-ADAPT-WAT-002: FloodRiskAgent - Flood risk assessment
    GL-ADAPT-WAT-003: WaterInfrastructureResilienceAgent - Infrastructure adaptation
    GL-ADAPT-WAT-004: WaterSecurityPlannerAgent - Water security planning
    GL-ADAPT-WAT-005: GroundwaterManagementAgent - Aquifer management

All agents follow the GreenLang standard patterns with:
    - Zero-hallucination calculations
    - Complete provenance tracking
    - Deterministic outputs
"""

from greenlang.agents.adaptation.water.water_scarcity import (
    WaterScarcityRiskAgent,
    WaterScarcityInput,
    WaterScarcityOutput,
    DroughtIndicator,
    ScarcityRiskLevel,
)
from greenlang.agents.adaptation.water.flood_risk import (
    FloodRiskAgent,
    FloodRiskInput,
    FloodRiskOutput,
    FloodHazard,
    FloodVulnerability,
)
from greenlang.agents.adaptation.water.infrastructure_resilience import (
    WaterInfrastructureResilienceAgent,
    InfraResilienceInput,
    InfraResilienceOutput,
    ResilienceAssessment,
    AdaptationMeasure,
)
from greenlang.agents.adaptation.water.water_security import (
    WaterSecurityPlannerAgent,
    WaterSecurityInput,
    WaterSecurityOutput,
    SecurityDimension,
    SecurityScorecard,
)
from greenlang.agents.adaptation.water.groundwater import (
    GroundwaterManagementAgent,
    GroundwaterInput,
    GroundwaterOutput,
    AquiferStatus,
    RechargeAssessment,
)

__all__ = [
    # Water Scarcity (GL-ADAPT-WAT-001)
    "WaterScarcityRiskAgent",
    "WaterScarcityInput",
    "WaterScarcityOutput",
    "DroughtIndicator",
    "ScarcityRiskLevel",
    # Flood Risk (GL-ADAPT-WAT-002)
    "FloodRiskAgent",
    "FloodRiskInput",
    "FloodRiskOutput",
    "FloodHazard",
    "FloodVulnerability",
    # Infrastructure Resilience (GL-ADAPT-WAT-003)
    "WaterInfrastructureResilienceAgent",
    "InfraResilienceInput",
    "InfraResilienceOutput",
    "ResilienceAssessment",
    "AdaptationMeasure",
    # Water Security (GL-ADAPT-WAT-004)
    "WaterSecurityPlannerAgent",
    "WaterSecurityInput",
    "WaterSecurityOutput",
    "SecurityDimension",
    "SecurityScorecard",
    # Groundwater (GL-ADAPT-WAT-005)
    "GroundwaterManagementAgent",
    "GroundwaterInput",
    "GroundwaterOutput",
    "AquiferStatus",
    "RechargeAssessment",
]
