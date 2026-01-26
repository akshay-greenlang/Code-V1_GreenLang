# -*- coding: utf-8 -*-
"""
GreenLang Industrial Climate Adaptation Agents
===============================================

Adaptation agents for industrial climate risk management:
    - GL-ADAPT-IND-001 to IND-012: Sector-specific adaptation agents

Features:
    - Physical climate risk assessment
    - Supply chain vulnerability analysis
    - Adaptation measure evaluation
    - Resilience planning

Author: GreenLang Framework Team
Version: 1.0.0
"""

from .sector_agents import (
    IndustrialAdaptationBaseAgent,
    AdaptationInput,
    AdaptationOutput,
    SteelAdaptationAgent,
    CementAdaptationAgent,
    ChemicalsAdaptationAgent,
    AluminumAdaptationAgent,
    PulpPaperAdaptationAgent,
    GlassAdaptationAgent,
    FoodProcessingAdaptationAgent,
    PharmaceuticalAdaptationAgent,
    ElectronicsAdaptationAgent,
    AutomotiveAdaptationAgent,
    TextilesAdaptationAgent,
    MiningAdaptationAgent,
)

__all__ = [
    "IndustrialAdaptationBaseAgent",
    "AdaptationInput",
    "AdaptationOutput",
    "SteelAdaptationAgent",
    "CementAdaptationAgent",
    "ChemicalsAdaptationAgent",
    "AluminumAdaptationAgent",
    "PulpPaperAdaptationAgent",
    "GlassAdaptationAgent",
    "FoodProcessingAdaptationAgent",
    "PharmaceuticalAdaptationAgent",
    "ElectronicsAdaptationAgent",
    "AutomotiveAdaptationAgent",
    "TextilesAdaptationAgent",
    "MiningAdaptationAgent",
]

__version__ = "1.0.0"
