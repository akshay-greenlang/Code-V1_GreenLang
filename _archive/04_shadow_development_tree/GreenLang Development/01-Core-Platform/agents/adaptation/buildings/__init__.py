# -*- coding: utf-8 -*-
"""
GreenLang Buildings Sector Adaptation Agents
=============================================

Climate adaptation agents for building sector resilience planning.
These agents assess climate risks and recommend adaptation strategies
to protect buildings from physical climate hazards.

Agents:
    GL-ADAPT-BLD-001: Heat Resilience Agent - Extreme heat adaptation
    GL-ADAPT-BLD-002: Flood Resilience Agent - Flooding adaptation
    GL-ADAPT-BLD-003: Wind Resilience Agent - High wind/storm adaptation
    GL-ADAPT-BLD-004: Wildfire Resilience Agent - Wildfire protection
    GL-ADAPT-BLD-005: Sea Level Rise Agent - Coastal flooding
    GL-ADAPT-BLD-006: Drought Resilience Agent - Water scarcity
    GL-ADAPT-BLD-007: Building Envelope Resilience - Weatherization
    GL-ADAPT-BLD-008: Grid Resilience Agent - Power outage protection

Design Principles:
    - Risk-based approach: Quantitative risk assessment
    - Multi-hazard analysis: Combined risk evaluation
    - Financial analysis: Cost-benefit for adaptation measures
    - Standards-aligned: TCFD, CRREM physical risk

Author: GreenLang Framework Team
Version: 1.0.0
"""

from greenlang.agents.adaptation.buildings.adaptation_agents import (
    # Base
    BuildingAdaptationBaseAgent,
    AdaptationInput,
    AdaptationOutput,
    ClimateHazard,
    RiskLevel,
    AdaptationMeasure,
    RiskAssessment,
    # GL-ADAPT-BLD-001
    HeatResilienceAgent,
    HeatResilienceInput,
    HeatResilienceOutput,
    # GL-ADAPT-BLD-002
    FloodResilienceAgent,
    FloodResilienceInput,
    FloodResilienceOutput,
    # GL-ADAPT-BLD-003
    WindResilienceAgent,
    WindResilienceInput,
    WindResilienceOutput,
    # GL-ADAPT-BLD-004
    WildfireResilienceAgent,
    WildfireResilienceInput,
    WildfireResilienceOutput,
    # GL-ADAPT-BLD-005
    SeaLevelRiseAgent,
    SeaLevelRiseInput,
    SeaLevelRiseOutput,
    # GL-ADAPT-BLD-006
    DroughtResilienceAgent,
    DroughtResilienceInput,
    DroughtResilienceOutput,
    # GL-ADAPT-BLD-007
    EnvelopeResilienceAgent,
    EnvelopeResilienceInput,
    EnvelopeResilienceOutput,
    # GL-ADAPT-BLD-008
    GridResilienceAgent,
    GridResilienceInput,
    GridResilienceOutput,
)

__all__ = [
    # Base
    "BuildingAdaptationBaseAgent",
    "AdaptationInput",
    "AdaptationOutput",
    "ClimateHazard",
    "RiskLevel",
    "AdaptationMeasure",
    "RiskAssessment",
    # Agents
    "HeatResilienceAgent",
    "HeatResilienceInput",
    "HeatResilienceOutput",
    "FloodResilienceAgent",
    "FloodResilienceInput",
    "FloodResilienceOutput",
    "WindResilienceAgent",
    "WindResilienceInput",
    "WindResilienceOutput",
    "WildfireResilienceAgent",
    "WildfireResilienceInput",
    "WildfireResilienceOutput",
    "SeaLevelRiseAgent",
    "SeaLevelRiseInput",
    "SeaLevelRiseOutput",
    "DroughtResilienceAgent",
    "DroughtResilienceInput",
    "DroughtResilienceOutput",
    "EnvelopeResilienceAgent",
    "EnvelopeResilienceInput",
    "EnvelopeResilienceOutput",
    "GridResilienceAgent",
    "GridResilienceInput",
    "GridResilienceOutput",
]
