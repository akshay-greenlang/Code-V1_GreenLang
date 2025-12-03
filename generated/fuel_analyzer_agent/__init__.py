"""
Fuel Emissions Analyzer - Calculates greenhouse gas emissions from fuel combustion using IPCC emission factors. Supports multiple fuel types (natural gas, diesel, gasoline, LPG) and provides complete provenance tracking for regulatory compliance.


Version: 1.0.0
License: Apache-2.0
"""

from fuel_analyzer_v1.agent import FuelEmissionsAnalyzerAgent
from fuel_analyzer_v1.tools import *

__all__ = [
    "FuelEmissionsAnalyzerAgent",
]

__version__ = "1.0.0"
