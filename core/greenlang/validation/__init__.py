"""
GreenLang Validation Framework

Zero-hallucination validation hooks for climate calculations.
"""

from .hooks import (
    EmissionFactorValidator,
    UnitValidator,
    ThermodynamicValidator,
    GWPValidator,
    ValidationResult,
    ValidationError,
    ValidationLevel
)
from .emission_factors import EmissionFactorDB, EmissionCategory, DataSource

__all__ = [
    'EmissionFactorValidator',
    'UnitValidator',
    'ThermodynamicValidator',
    'GWPValidator',
    'ValidationResult',
    'ValidationError',
    'ValidationLevel',
    'EmissionFactorDB',
    'EmissionCategory',
    'DataSource'
]
