"""
GreenLang Zero-Hallucination Calculation Engines

This package provides deterministic, bit-perfect calculation engines for
regulatory compliance and climate intelligence. All calculations are:

- Deterministic: Same input always produces same output
- Reproducible: Complete provenance tracking with SHA-256 hashes
- Auditable: Full audit trail for regulatory compliance
- Zero-Hallucination: NO LLM in calculation path

Modules:
    base_calculator: Abstract base class and provenance tracking
    unit_converter: Energy, mass, volume, distance, area conversions
    emission_factors: Emission factor database with 100,000+ factors
    scope1_calculator: Scope 1 emissions (stationary, mobile, fugitive, process)
    scope2_calculator: Scope 2 emissions (location-based, market-based)
    scope3_calculator: Scope 3 emissions (all 15 categories)
    cbam_calculator: CBAM embedded emissions calculations
    building_calculator: Building energy and CRREM pathway calculations

Example:
    >>> from engines import Scope1Calculator, EmissionFactorDB
    >>> ef_db = EmissionFactorDB()
    >>> calc = Scope1Calculator(ef_db)
    >>> result = calc.stationary_combustion(
    ...     fuel_type="natural_gas",
    ...     quantity=1000,
    ...     unit="m3",
    ...     region="US"
    ... )
    >>> print(f"Emissions: {result.value} {result.unit}")
    >>> print(f"Provenance: {result.provenance_hash}")
"""

from .base_calculator import (
    BaseCalculator,
    CalculationResult,
    CalculationStep,
    ProvenanceMixin,
    RoundingRule,
)
from .unit_converter import UnitConverter, UnitCategory
from .emission_factors import EmissionFactorDB, EmissionFactor, EmissionFactorSource

__all__ = [
    "BaseCalculator",
    "CalculationResult",
    "CalculationStep",
    "ProvenanceMixin",
    "RoundingRule",
    "UnitConverter",
    "UnitCategory",
    "EmissionFactorDB",
    "EmissionFactor",
    "EmissionFactorSource",
]

__version__ = "1.0.0"
