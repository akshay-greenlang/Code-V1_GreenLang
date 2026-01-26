# -*- coding: utf-8 -*-
"""
GreenLang Industrial Decarbonization Agents
============================================

This package contains decarbonization planning agents for industrial sectors:
    - GL-DECARB-IND-001: Steel Decarbonization
    - GL-DECARB-IND-002: Cement Decarbonization
    - GL-DECARB-IND-003: Chemicals Decarbonization
    - GL-DECARB-IND-004: Aluminum Decarbonization
    - GL-DECARB-IND-005: Pulp & Paper Decarbonization
    - GL-DECARB-IND-006: Glass Decarbonization
    - GL-DECARB-IND-007: Food Processing Decarbonization
    - GL-DECARB-IND-008: Pharmaceutical Decarbonization
    - GL-DECARB-IND-009: Electronics Decarbonization
    - GL-DECARB-IND-010: Automotive Decarbonization
    - GL-DECARB-IND-011: Textiles Decarbonization
    - GL-DECARB-IND-012: Mining Decarbonization

Author: GreenLang Framework Team
Version: 1.0.0
"""

from typing import List

from .base import (
    IndustrialDecarbonizationBaseAgent,
    DecarbonizationInput,
    DecarbonizationOutput,
    DecarbonizationPathway,
    Technology,
    TechnologyReadiness,
    DecarbonizationLever,
    Milestone,
    TimeHorizon,
)

from .sector_agents import (
    SteelDecarbonizationAgent,
    SteelDecarbInput,
    SteelDecarbOutput,
    CementDecarbonizationAgent,
    ChemicalsDecarbonizationAgent,
    AluminumDecarbonizationAgent,
    PulpPaperDecarbonizationAgent,
    GlassDecarbonizationAgent,
    FoodProcessingDecarbonizationAgent,
    PharmaceuticalDecarbonizationAgent,
    ElectronicsDecarbonizationAgent,
    AutomotiveDecarbonizationAgent,
    TextilesDecarbonizationAgent,
    MiningDecarbonizationAgent,
)

__all__: List[str] = [
    # Base
    "IndustrialDecarbonizationBaseAgent",
    "DecarbonizationInput",
    "DecarbonizationOutput",
    "DecarbonizationPathway",
    "Technology",
    "TechnologyReadiness",
    "DecarbonizationLever",
    "Milestone",
    "TimeHorizon",
    # Sector agents
    "SteelDecarbonizationAgent",
    "SteelDecarbInput",
    "SteelDecarbOutput",
    "CementDecarbonizationAgent",
    "ChemicalsDecarbonizationAgent",
    "AluminumDecarbonizationAgent",
    "PulpPaperDecarbonizationAgent",
    "GlassDecarbonizationAgent",
    "FoodProcessingDecarbonizationAgent",
    "PharmaceuticalDecarbonizationAgent",
    "ElectronicsDecarbonizationAgent",
    "AutomotiveDecarbonizationAgent",
    "TextilesDecarbonizationAgent",
    "MiningDecarbonizationAgent",
]

__version__ = "1.0.0"
