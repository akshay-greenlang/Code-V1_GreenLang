"""
CBAM Carbon Intensity Calculator - Calculates carbon intensity for CBAM-regulated goods (steel, cement, aluminum, fertilizers) and determines CBAM certificate requirements for EU imports.


Version: 1.0.0
License: Apache-2.0
"""

from carbon_intensity_v1.agent import CbamCarbonIntensityCalculatorAgent
from carbon_intensity_v1.tools import *

__all__ = [
    "CbamCarbonIntensityCalculatorAgent",
]

__version__ = "1.0.0"
