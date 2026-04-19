# -*- coding: utf-8 -*-
"""
SB 253 Scope 3 Calculators Package

Comprehensive calculators for all 15 GHG Protocol Scope 3 categories
required under California SB 253 Climate Corporate Data Accountability Act.

Upstream Categories (1-8):
- Category 1: Purchased Goods and Services
- Category 2: Capital Goods
- Category 3: Fuel and Energy Related Activities
- Category 4: Upstream Transportation and Distribution
- Category 5: Waste Generated in Operations
- Category 6: Business Travel
- Category 7: Employee Commuting
- Category 8: Upstream Leased Assets

Downstream Categories (9-15):
- Category 9: Downstream Transportation and Distribution
- Category 10: Processing of Sold Products
- Category 11: Use of Sold Products
- Category 12: End-of-Life Treatment of Sold Products
- Category 13: Downstream Leased Assets
- Category 14: Franchises
- Category 15: Investments

All calculators implement:
- Zero-Hallucination calculation paths (no LLM)
- EPA EEIO factors for spend-based methods
- GHG Protocol Technical Guidance formulas
- SHA-256 audit trail generation
- Pydantic models for type safety

Version: 1.0.0
"""

from .base import (
    Scope3CategoryCalculator,
    Scope3CalculationResult,
    Scope3CalculationInput,
    CalculationMethod,
    EmissionFactorSource,
)

from .category01_purchased_goods import (
    Category01PurchasedGoodsCalculator,
)

from .category02_capital_goods import (
    Category02CapitalGoodsCalculator,
)

from .category03_fuel_energy import (
    Category03FuelEnergyCalculator,
)

from .category04_upstream_transport import (
    Category04UpstreamTransportCalculator,
)

from .category05_waste import (
    Category05WasteCalculator,
)

from .category06_business_travel import (
    Category06BusinessTravelCalculator,
)

from .category07_commuting import (
    Category07CommutingCalculator,
)

from .category08_leased_assets import (
    Category08UpstreamLeasedAssetsCalculator,
)

from .category09_downstream_transport import (
    Category09DownstreamTransportCalculator,
)

from .category10_processing import (
    Category10ProcessingCalculator,
)

from .category11_product_use import (
    Category11ProductUseCalculator,
)

from .category12_end_of_life import (
    Category12EndOfLifeCalculator,
)

from .category13_downstream_leased import (
    Category13DownstreamLeasedCalculator,
)

from .category14_franchises import (
    Category14FranchisesCalculator,
)

from .category15_investments import (
    Category15InvestmentsCalculator,
)

__all__ = [
    # Base classes
    "Scope3CategoryCalculator",
    "Scope3CalculationResult",
    "Scope3CalculationInput",
    "CalculationMethod",
    "EmissionFactorSource",
    # Category calculators
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
