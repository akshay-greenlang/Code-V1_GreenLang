# -*- coding: utf-8 -*-
"""
GreenLang Industrial Sustainable Procurement Agents
====================================================

Procurement agents for sustainable industrial supply chains:
    - GL-PROC-IND-001 to IND-015: Sector-specific procurement agents

Features:
    - Supplier carbon footprint assessment
    - Low-carbon material sourcing
    - Scope 3 emissions optimization
    - Circular procurement strategies

Author: GreenLang Framework Team
Version: 1.0.0
"""

from .sector_agents import (
    IndustrialProcurementBaseAgent,
    ProcurementInput,
    ProcurementOutput,
    SteelProcurementAgent,
    CementProcurementAgent,
    ChemicalsProcurementAgent,
    AluminumProcurementAgent,
    PulpPaperProcurementAgent,
    GlassProcurementAgent,
    FoodProcessingProcurementAgent,
    PharmaceuticalProcurementAgent,
    ElectronicsProcurementAgent,
    AutomotiveProcurementAgent,
    TextilesProcurementAgent,
    MiningProcurementAgent,
    PlasticsProcurementAgent,
    PackagingProcurementAgent,
    ConstructionProcurementAgent,
)

__all__ = [
    "IndustrialProcurementBaseAgent",
    "ProcurementInput",
    "ProcurementOutput",
    "SteelProcurementAgent",
    "CementProcurementAgent",
    "ChemicalsProcurementAgent",
    "AluminumProcurementAgent",
    "PulpPaperProcurementAgent",
    "GlassProcurementAgent",
    "FoodProcessingProcurementAgent",
    "PharmaceuticalProcurementAgent",
    "ElectronicsProcurementAgent",
    "AutomotiveProcurementAgent",
    "TextilesProcurementAgent",
    "MiningProcurementAgent",
    "PlasticsProcurementAgent",
    "PackagingProcurementAgent",
    "ConstructionProcurementAgent",
]

__version__ = "1.0.0"
