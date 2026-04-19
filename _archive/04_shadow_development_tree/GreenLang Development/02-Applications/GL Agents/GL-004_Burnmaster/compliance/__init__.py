"""
Compliance Module for GL-004 BURNMASTER
=========================================

Provides permit management, compliance tracking, and regulatory reporting
functionality for air quality compliance. Includes comprehensive support
for EPA 40 CFR Part 60 NSPS (New Source Performance Standards).

Submodules:
    permit_limits: Facility-specific permit limit management
    epa_40cfr_part60: EPA 40 CFR Part 60 NSPS compliance validation
        - Subpart D: Fossil-fuel-fired steam generators (>250 MMBtu/hr)
        - Subpart Da: Electric utility steam generating units
        - Subpart Db: Industrial-commercial-institutional units (>100 MMBtu/hr)
        - Subpart Dc: Small ICI units (10-100 MMBtu/hr)

Exports:
    - PermitLimit: Permit limit data model
    - PermitLimitsManager: Main permit management class
    - EmissionUnitConverter: Unit conversion utilities
    - AveragingCalculator: Averaging period calculations
    - ComplianceStatus: Compliance status enumeration
    - Pollutant: Pollutant type enumeration
    - DeviationType: Deviation type enumeration
    - NSPSComplianceValidator: EPA 40 CFR Part 60 NSPS validator
    - check_nsps_compliance: Quick NSPS compliance check function
    - get_emission_limit: NSPS emission limit lookup

Usage:
    >>> from compliance import check_nsps_compliance, get_emission_limit
    >>>
    >>> # Quick NSPS compliance check
    >>> result = check_nsps_compliance(
    ...     unit_capacity_mmbtu_hr=150.0,
    ...     fuel_type="natural_gas",
    ...     measured_nox_lb_mmbtu=0.08,
    ...     measured_so2_lb_mmbtu=0.0,
    ...     measured_pm_lb_mmbtu=0.0,
    ... )
    >>> print(f"Status: {result.compliance_status.value}")

Author: GL-RegulatoryIntelligence
Version: 1.1.0
"""

# Permit limits module
from .permit_limits import (
    # Data classes
    PermitLimit,
    EmissionMeasurement,
    ExemptPeriod,
    Deviation,
    ComplianceCheckResult,
    # Pydantic models
    PermitLimitInput,
    ComplianceCheckInput,
    DeviationReportOutput,
    # Enumerations
    Pollutant,
    LimitUnits,
    AveragingPeriod,
    RegulatoryBasis,
    PermitType,
    ComplianceStatus,
    DeviationType,
    RootCauseCategory,
    CorrectiveActionStatus,
    # Core classes
    EmissionUnitConverter,
    AveragingCalculator,
    PermitLimitsManager,
)

# EPA 40 CFR Part 60 NSPS module
from .epa_40cfr_part60 import (
    # Main validator class
    NSPSComplianceValidator,
    # Enums (with NSPS prefix to avoid conflicts)
    NSPSSubpart,
    FuelCategory,
    PollutantType as NSPSPollutantType,
    ComplianceStatus as NSPSComplianceStatus,
    MonitoringMethod,
    ExemptionType,
    ReportingPeriod,
    # Data classes
    UnitCharacteristics,
    EmissionMeasurement as NSPSEmissionMeasurement,
    ExemptionRecord,
    DeviationEvent,
    ComplianceResult as NSPSComplianceResult,
    NSPSComplianceReport,
    CEMSRequirement,
    FuelAnalysisRequirement,
    RecordkeepingRequirement,
    ReportingRequirement,
    # Convenience functions
    check_nsps_compliance,
    get_emission_limit,
    determine_applicable_subpart,
    # Emission limit constants
    NSPS_LIMITS_SUBPART_D,
    NSPS_LIMITS_SUBPART_Da,
    NSPS_LIMITS_SUBPART_Db,
    NSPS_LIMITS_SUBPART_Dc,
    # Requirements constants
    CEMS_REQUIREMENTS_BY_SUBPART,
    FUEL_ANALYSIS_REQUIREMENTS,
    RECORDKEEPING_REQUIREMENTS,
    REPORTING_REQUIREMENTS,
    SUBPART_APPLICABILITY_DATES,
)

__all__ = [
    # Permit Limits Module
    "PermitLimit",
    "EmissionMeasurement",
    "ExemptPeriod",
    "Deviation",
    "ComplianceCheckResult",
    "PermitLimitInput",
    "ComplianceCheckInput",
    "DeviationReportOutput",
    "Pollutant",
    "LimitUnits",
    "AveragingPeriod",
    "RegulatoryBasis",
    "PermitType",
    "ComplianceStatus",
    "DeviationType",
    "RootCauseCategory",
    "CorrectiveActionStatus",
    "EmissionUnitConverter",
    "AveragingCalculator",
    "PermitLimitsManager",
    # EPA 40 CFR Part 60 NSPS Module
    "NSPSComplianceValidator",
    "NSPSSubpart",
    "FuelCategory",
    "NSPSPollutantType",
    "NSPSComplianceStatus",
    "MonitoringMethod",
    "ExemptionType",
    "ReportingPeriod",
    "UnitCharacteristics",
    "NSPSEmissionMeasurement",
    "ExemptionRecord",
    "DeviationEvent",
    "NSPSComplianceResult",
    "NSPSComplianceReport",
    "CEMSRequirement",
    "FuelAnalysisRequirement",
    "RecordkeepingRequirement",
    "ReportingRequirement",
    "check_nsps_compliance",
    "get_emission_limit",
    "determine_applicable_subpart",
    "NSPS_LIMITS_SUBPART_D",
    "NSPS_LIMITS_SUBPART_Da",
    "NSPS_LIMITS_SUBPART_Db",
    "NSPS_LIMITS_SUBPART_Dc",
    "CEMS_REQUIREMENTS_BY_SUBPART",
    "FUEL_ANALYSIS_REQUIREMENTS",
    "RECORDKEEPING_REQUIREMENTS",
    "REPORTING_REQUIREMENTS",
    "SUBPART_APPLICABILITY_DATES",
]

__version__ = "1.1.0"
__author__ = "GL-RegulatoryIntelligence"
