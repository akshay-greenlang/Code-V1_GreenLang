# -*- coding: utf-8 -*-
"""
SB 253 Scope 1 Direct Emission Calculators
==========================================

Scope 1 emissions are direct GHG emissions from sources owned or
controlled by the reporting company.

Categories:
    - Stationary Combustion: Boilers, furnaces, heaters, generators
    - Mobile Combustion: Fleet vehicles
    - Fugitive Emissions: Refrigerant leakage
    - Process Emissions: Industrial processes

Emission Factors:
    - EPA GHG Emission Factors Hub 2024
    - IPCC AR6 GWP-100 values

Author: GreenLang Framework Team
Version: 1.0.0
"""

from .stationary_combustion import StationaryCombustionCalculator
from .mobile_combustion import MobileCombustionCalculator
from .fugitive_emissions import FugitiveEmissionsCalculator
from .process_emissions import ProcessEmissionsCalculator
from .aggregator import Scope1Aggregator

__all__ = [
    "StationaryCombustionCalculator",
    "MobileCombustionCalculator",
    "FugitiveEmissionsCalculator",
    "ProcessEmissionsCalculator",
    "Scope1Aggregator",
]
