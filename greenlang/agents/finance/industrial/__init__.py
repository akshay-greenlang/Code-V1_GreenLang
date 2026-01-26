# -*- coding: utf-8 -*-
"""
GreenLang Industrial Finance Agents
====================================

Finance agents for industrial climate investments and green financing:
    - GL-FIN-IND-001 to IND-012: Sector-specific finance agents

Features:
    - Carbon pricing impact analysis
    - Green bond eligibility assessment
    - CAPEX/OPEX modeling for decarbonization
    - ROI and NPV calculations
    - Carbon credit valuation

Author: GreenLang Framework Team
Version: 1.0.0
"""

from .sector_agents import (
    IndustrialFinanceBaseAgent,
    FinanceInput,
    FinanceOutput,
    SteelFinanceAgent,
    CementFinanceAgent,
    ChemicalsFinanceAgent,
    AluminumFinanceAgent,
    PulpPaperFinanceAgent,
    GlassFinanceAgent,
    FoodProcessingFinanceAgent,
    PharmaceuticalFinanceAgent,
    ElectronicsFinanceAgent,
    AutomotiveFinanceAgent,
    TextilesFinanceAgent,
    MiningFinanceAgent,
)

__all__ = [
    "IndustrialFinanceBaseAgent",
    "FinanceInput",
    "FinanceOutput",
    "SteelFinanceAgent",
    "CementFinanceAgent",
    "ChemicalsFinanceAgent",
    "AluminumFinanceAgent",
    "PulpPaperFinanceAgent",
    "GlassFinanceAgent",
    "FoodProcessingFinanceAgent",
    "PharmaceuticalFinanceAgent",
    "ElectronicsFinanceAgent",
    "AutomotiveFinanceAgent",
    "TextilesFinanceAgent",
    "MiningFinanceAgent",
]

__version__ = "1.0.0"
