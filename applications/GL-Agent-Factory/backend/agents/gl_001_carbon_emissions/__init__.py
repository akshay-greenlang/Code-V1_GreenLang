"""
GL-001: Carbon Emissions Calculator Agent

GHG emissions calculator with zero-hallucination deterministic calculations.
"""

from .agent import (
    CarbonEmissionsAgent,
    CarbonEmissionsInput,
    CarbonEmissionsOutput,
    FuelType,
    Scope,
    EmissionFactor,
)

__all__ = [
    "CarbonEmissionsAgent",
    "CarbonEmissionsInput",
    "CarbonEmissionsOutput",
    "FuelType",
    "Scope",
    "EmissionFactor",
]
