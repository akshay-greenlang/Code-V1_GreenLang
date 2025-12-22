"""
GL-002 FLAMEGUARD - Calculators Module

Zero-hallucination calculation engines for:
- Boiler efficiency (ASME PTC 4.1)
- Emissions (EPA factors)
- Fuel blending
- Heat balance
"""

from .efficiency_calculator import EfficiencyCalculator
from .emissions_calculator import EmissionsCalculator
from .fuel_blending_calculator import FuelBlendingCalculator
from .heat_balance_calculator import HeatBalanceCalculator

__all__ = [
    "EfficiencyCalculator",
    "EmissionsCalculator",
    "FuelBlendingCalculator",
    "HeatBalanceCalculator",
]
