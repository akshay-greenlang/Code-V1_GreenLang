"""
GreenLang Emission Factor Data Module

This module provides comprehensive emission factor data and services
for GHG calculations. It includes:

- 1000+ emission factors from authoritative sources
- EPA, DEFRA, IPCC, IEA factor databases
- Version-pinned factors for reproducibility
- Unit conversion utilities
- Factor lookup services with fallback hierarchies

Usage:
    from backend.data import get_emission_factor_service, get_unit_converter

    # Get emission factor
    ef_service = get_emission_factor_service()
    result = ef_service.get_factor("natural_gas", region="US")

    # Convert units
    converter = get_unit_converter()
    kwh = converter.mmbtu_to_kwh(100)
"""

from .models import (
    EmissionFactor,
    EmissionFactorSource,
    GWPSet,
    EmissionScope,
    EmissionCategory,
    UncertaintyRange,
    GWPValue,
    GridEmissionFactor,
    MaterialEmissionFactor,
    TransportEmissionFactor,
    EmissionFactorVersion,
    STANDARD_GWP_AR6_100YR,
    STANDARD_GWP_AR5_100YR,
)

from .ef_service import (
    EmissionFactorService,
    FactorLookupResult,
    get_emission_factor_service,
)

from .unit_converter import (
    UnitConverter,
    UnitCategory,
    ConversionResult,
    get_unit_converter,
)

__all__ = [
    # Models
    "EmissionFactor",
    "EmissionFactorSource",
    "GWPSet",
    "EmissionScope",
    "EmissionCategory",
    "UncertaintyRange",
    "GWPValue",
    "GridEmissionFactor",
    "MaterialEmissionFactor",
    "TransportEmissionFactor",
    "EmissionFactorVersion",
    "STANDARD_GWP_AR6_100YR",
    "STANDARD_GWP_AR5_100YR",
    # Services
    "EmissionFactorService",
    "FactorLookupResult",
    "get_emission_factor_service",
    # Unit Converter
    "UnitConverter",
    "UnitCategory",
    "ConversionResult",
    "get_unit_converter",
]

__version__ = "1.0.0"
