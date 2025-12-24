"""
GL-001 ThermalCommand - Calculators Module

High-precision calculators for thermal energy management with
regulatory compliance and full provenance tracking.
"""

from .precision_utils import (
    PrecisionCalculator,
    PrecisionUnitConverter,
    PrecisionEmissionsCalculator,
    PrecisionEfficiencyCalculator,
    PrecisionError,
    PRECISION_CONTEXTS,
    UNIT_CONVERSIONS,
    EPA_EMISSION_FACTORS,
    GWP_AR5,
    UnitConversionResult,
    EmissionsCalculationResult,
    EfficiencyCalculationResult,
    validate_precision_result,
    format_decimal_for_report,
)

__all__ = [
    "PrecisionCalculator",
    "PrecisionUnitConverter",
    "PrecisionEmissionsCalculator",
    "PrecisionEfficiencyCalculator",
    "PrecisionError",
    "PRECISION_CONTEXTS",
    "UNIT_CONVERSIONS",
    "EPA_EMISSION_FACTORS",
    "GWP_AR5",
    "UnitConversionResult",
    "EmissionsCalculationResult",
    "EfficiencyCalculationResult",
    "validate_precision_result",
    "format_decimal_for_report",
]
