# -*- coding: utf-8 -*-
"""
SB 253 Scope 2 Energy Indirect Emission Calculators
====================================================

Scope 2 emissions are indirect GHG emissions from purchased energy:
- Purchased electricity
- Purchased steam
- Purchased heating
- Purchased cooling

Methods (GHG Protocol Scope 2 Guidance):
    - Location-based: Grid average emission factors (MANDATORY)
    - Market-based: Contractual instruments (RECOMMENDED)

Emission Factors:
    - EPA eGRID 2023 (subregional grid factors)
    - California utility-specific factors

Author: GreenLang Framework Team
Version: 1.0.0
"""

from .location_based import LocationBasedCalculator
from .market_based import MarketBasedCalculator
from .dual_reporter import Scope2DualReporter

__all__ = [
    "LocationBasedCalculator",
    "MarketBasedCalculator",
    "Scope2DualReporter",
]
