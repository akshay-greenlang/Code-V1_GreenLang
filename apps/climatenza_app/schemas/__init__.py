"""
Climatenza AI Schemas - Data validation and contracts
"""

from .feasibility import (
    Site,
    ProcessDemand,
    Boiler,
    SolarConfig,
    FinanceInputs,
    Assumptions,
    FeasibilityInput
)

__all__ = [
    "Site",
    "ProcessDemand",
    "Boiler",
    "SolarConfig",
    "FinanceInputs",
    "Assumptions",
    "FeasibilityInput"
]