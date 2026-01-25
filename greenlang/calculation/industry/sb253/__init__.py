# -*- coding: utf-8 -*-
"""
California SB 253 Calculators Package

California Senate Bill 253 - Climate Corporate Data Accountability Act
Requires companies with >$1B annual revenue to disclose Scope 1, 2, and 3 emissions.

This package provides GHG Protocol-compliant calculators for:
- Scope 1: Direct emissions (combustion, fugitives, process)
- Scope 2: Indirect energy emissions (purchased electricity, steam, heat, cooling)
- Scope 3: Value chain emissions (15 categories)

Key Features:
- EPA EEIO factors for spend-based calculations
- GHG Protocol Technical Guidance alignment
- CARB (California Air Resources Board) methodology compliance
- Full audit trail with SHA-256 hashing

Version: 1.0.0
"""

from greenlang.calculators.sb253.scope3 import (
    Scope3CategoryCalculator,
    Category01PurchasedGoodsCalculator,
    Category02CapitalGoodsCalculator,
    Category03FuelEnergyCalculator,
    Category04UpstreamTransportCalculator,
    Category05WasteCalculator,
    Category06BusinessTravelCalculator,
    Category07CommutingCalculator,
    Category08UpstreamLeasedAssetsCalculator,
    Category09DownstreamTransportCalculator,
    Category10ProcessingCalculator,
    Category11ProductUseCalculator,
    Category12EndOfLifeCalculator,
    Category13DownstreamLeasedCalculator,
    Category14FranchisesCalculator,
    Category15InvestmentsCalculator,
)

__all__ = [
    "Scope3CategoryCalculator",
    "Category01PurchasedGoodsCalculator",
    "Category02CapitalGoodsCalculator",
    "Category03FuelEnergyCalculator",
    "Category04UpstreamTransportCalculator",
    "Category05WasteCalculator",
    "Category06BusinessTravelCalculator",
    "Category07CommutingCalculator",
    "Category08UpstreamLeasedAssetsCalculator",
    "Category09DownstreamTransportCalculator",
    "Category10ProcessingCalculator",
    "Category11ProductUseCalculator",
    "Category12EndOfLifeCalculator",
    "Category13DownstreamLeasedCalculator",
    "Category14FranchisesCalculator",
    "Category15InvestmentsCalculator",
]

__version__ = "1.0.0"
