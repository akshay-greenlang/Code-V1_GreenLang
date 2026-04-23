"""
GL-001 ThermalCommand - Compliance Module

Regulatory compliance mappings and validation for EPA, ASME, and other
standards applicable to thermal energy management systems.
"""

from .epa_mapping import (
    EPARegulation,
    PollutantType,
    FuelCategory,
    EPAEmissionFactor,
    ComplianceRequirement,
    ComplianceValidationResult,
    EPAComplianceMapper,
    CalculationMethodMapping,
    EPA_CO2_FACTORS,
    EPA_CH4_N2O_FACTORS,
    EPA_GWP_VALUES,
    EPA_COMPLIANCE_REQUIREMENTS,
    CALCULATION_METHOD_MAPPINGS,
    get_calculation_method,
    get_all_calculation_methods,
)

__all__ = [
    "EPARegulation",
    "PollutantType",
    "FuelCategory",
    "EPAEmissionFactor",
    "ComplianceRequirement",
    "ComplianceValidationResult",
    "EPAComplianceMapper",
    "CalculationMethodMapping",
    "EPA_CO2_FACTORS",
    "EPA_CH4_N2O_FACTORS",
    "EPA_GWP_VALUES",
    "EPA_COMPLIANCE_REQUIREMENTS",
    "CALCULATION_METHOD_MAPPINGS",
    "get_calculation_method",
    "get_all_calculation_methods",
]
