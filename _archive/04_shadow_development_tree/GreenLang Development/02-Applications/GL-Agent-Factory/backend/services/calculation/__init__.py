"""
Calculation Engine Service Module

Provides zero-hallucination calculation capabilities:
- Deterministic formula execution
- Emission factor database access
- Unit conversion
- Provenance tracking
"""

from services.calculation.calculation_engine_service import (
    CalculationEngineService,
    CalculationResult,
    EmissionFactor,
)
from services.calculation.unit_converter import UnitConverter

__all__ = [
    "CalculationEngineService",
    "CalculationResult",
    "EmissionFactor",
    "UnitConverter",
]
