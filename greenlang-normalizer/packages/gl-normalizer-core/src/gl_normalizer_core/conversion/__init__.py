"""
Unit Conversion Engine for GL-FOUND-X-003 GreenLang Normalizer.

This module provides a production-grade unit conversion engine for sustainability
reporting, with support for:

- GL Canonical Units: MJ, kg, kgCO2e, m3, kPa_abs, degC, K
- GWP version handling: AR4, AR5, AR6 for CO2e conversions
- Basis handling: LHV/HHV for energy, wet/dry for mass
- Reference conditions: Temperature and pressure for gas volumes
- Pressure mode: Gauge to absolute conversion with atmospheric reference

Key Components:
    - ConversionEngine: Main conversion class with all conversion methods
    - ConversionContext: Immutable context for conversion parameters
    - ConversionFactorRegistry: Registry of conversion factors with versioning
    - ValidationResult: Result type for validation operations

All functions are pure and deterministic with no side effects or I/O operations.

Example:
    >>> from gl_normalizer_core.conversion import ConversionEngine, ConversionContext
    >>> engine = ConversionEngine()
    >>> context = ConversionContext(gwp_version="AR6", basis="LHV")
    >>> result = engine.convert(100.0, "kWh", "MJ", context)
    >>> print(result.converted_value)
    360.0

    >>> # Convert greenhouse gas to CO2e
    >>> result = engine.convert_ghg_to_co2e(1.0, "CH4", "kg", "AR6")
    >>> print(result.converted_value)
    27.9

    >>> # Convert with basis (LHV to HHV)
    >>> result = engine.convert_with_basis(
    ...     100.0, "MJ", "MJ", "LHV", "HHV", "natural_gas"
    ... )
    >>> print(result.converted_value)
    110.9
"""

# Core engine
from gl_normalizer_core.conversion.engine import (
    ConversionEngine,
    ConversionResult,
    ConversionStep,
    GL_CANONICAL_UNITS,
    UNIT_DIMENSIONS,
)

# Conversion context
from gl_normalizer_core.conversion.contexts import (
    ConversionContext,
    GWPVersion,
    EnergyBasis,
    MassBasis,
    PressureMode,
    STP_TEMPERATURE_C,
    STP_PRESSURE_KPA,
    NTP_TEMPERATURE_C,
    NTP_PRESSURE_KPA,
    ISO_TEMPERATURE_C,
    ISO_PRESSURE_KPA,
    DEFAULT_ATMOSPHERIC_PRESSURE_KPA,
    gauge_to_absolute,
    absolute_to_gauge,
    celsius_to_kelvin,
    kelvin_to_celsius,
)

# Conversion factors
from gl_normalizer_core.conversion.factors import (
    ConversionFactor,
    ConversionFactorRegistry,
    ConversionType,
    FactorStatus,
    GWPFactor,
    create_inverse_factor,
    ENERGY_FACTORS,
    MASS_FACTORS,
    VOLUME_FACTORS,
    TEMPERATURE_FACTORS,
    PRESSURE_FACTORS,
    GWP_FACTORS,
    LHV_HHV_RATIOS,
)

# Validators
from gl_normalizer_core.conversion.validators import (
    ValidationResult,
    validate_gwp_version,
    validate_basis,
    validate_reference_conditions,
    validate_conversion_path,
    validate_context_for_conversion,
    validate_dimension_compatibility,
    validate_numeric_value,
    VALID_GWP_VERSIONS,
    VALID_ENERGY_BASES,
    VALID_MASS_BASES,
    VALID_PRESSURE_MODES,
    UNITS_REQUIRING_REFERENCE_CONDITIONS,
    UNITS_REQUIRING_GWP,
    UNITS_REQUIRING_BASIS,
)


__all__ = [
    # Engine
    "ConversionEngine",
    "ConversionResult",
    "ConversionStep",
    "GL_CANONICAL_UNITS",
    "UNIT_DIMENSIONS",
    # Contexts
    "ConversionContext",
    "GWPVersion",
    "EnergyBasis",
    "MassBasis",
    "PressureMode",
    "STP_TEMPERATURE_C",
    "STP_PRESSURE_KPA",
    "NTP_TEMPERATURE_C",
    "NTP_PRESSURE_KPA",
    "ISO_TEMPERATURE_C",
    "ISO_PRESSURE_KPA",
    "DEFAULT_ATMOSPHERIC_PRESSURE_KPA",
    "gauge_to_absolute",
    "absolute_to_gauge",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    # Factors
    "ConversionFactor",
    "ConversionFactorRegistry",
    "ConversionType",
    "FactorStatus",
    "GWPFactor",
    "create_inverse_factor",
    "ENERGY_FACTORS",
    "MASS_FACTORS",
    "VOLUME_FACTORS",
    "TEMPERATURE_FACTORS",
    "PRESSURE_FACTORS",
    "GWP_FACTORS",
    "LHV_HHV_RATIOS",
    # Validators
    "ValidationResult",
    "validate_gwp_version",
    "validate_basis",
    "validate_reference_conditions",
    "validate_conversion_path",
    "validate_context_for_conversion",
    "validate_dimension_compatibility",
    "validate_numeric_value",
    "VALID_GWP_VERSIONS",
    "VALID_ENERGY_BASES",
    "VALID_MASS_BASES",
    "VALID_PRESSURE_MODES",
    "UNITS_REQUIRING_REFERENCE_CONDITIONS",
    "UNITS_REQUIRING_GWP",
    "UNITS_REQUIRING_BASIS",
]
