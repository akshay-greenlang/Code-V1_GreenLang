"""
GL-012 SteamQual - Compliance Module

Comprehensive regulatory compliance framework for steam quality monitoring
and control systems including quality standards, measurement governance,
carbon accounting, and regulatory tracking.

This module provides:
1. Steam quality standards compliance (ASME PTC 19.11)
2. Measurement governance and data quality scoring
3. Carbon and energy accounting with meter reconciliation
4. Regulatory compliance tracking and alerting

All calculations follow zero-hallucination principles using deterministic
formulas with complete provenance tracking for audit trails.

Author: GL-BackendDeveloper
Version: 1.0.0
"""

# =============================================================================
# QUALITY STANDARDS
# =============================================================================

from .quality_standards import (
    # Enumerations
    SteamQualityStandard,
    SteamPressureClass,
    SteamApplication,
    QualityParameter,
    ComplianceStatus as QualityComplianceStatus,

    # Data classes
    QualityLimit,
    BestPracticeThreshold,
    SiteQualityRequirement,
    QualityValidationResult,

    # Main validator
    SteamQualityStandardsValidator,

    # Factory functions
    create_quality_validator,
    get_asme_limits_for_pressure_class,
    get_quality_limits_for_parameter,

    # Reference data
    ASME_QUALITY_LIMITS,
    INDUSTRY_BEST_PRACTICES,
)

# =============================================================================
# MEASUREMENT GOVERNANCE
# =============================================================================

from .measurement_governance import (
    # Enumerations
    SensorType,
    CalibrationStatus,
    AccuracyClass,
    DataQualityGrade,
    MeasurementApplication,

    # Data classes
    SensorRequirement,
    RegisteredSensor,
    CalibrationRecord,
    DataQualityScore,
    UncertaintyBudget,

    # Main classes
    DataQualityScorer,
    UncertaintyCalculator,
    MeasurementGovernanceManager,

    # Factory functions
    create_governance_manager,
    create_data_quality_scorer,
    create_uncertainty_calculator,
    get_sensor_requirements,

    # Reference data
    SENSOR_REQUIREMENTS,
)

# =============================================================================
# CARBON ACCOUNTING
# =============================================================================

from .carbon_accounting import (
    # Enumerations
    FuelType,
    EmissionScope,
    QualityImprovementType,
    EmissionFactorSource,

    # Data classes
    EmissionFactor,
    EnergySavingsResult,
    EmissionsImpactResult,
    MeterReconciliationResult,

    # Main classes
    EnergySavingsCalculator,
    EmissionsImpactCalculator,
    MeterReconciler,
    CarbonAccountingManager,

    # Factory functions
    create_carbon_accounting_manager,
    get_emission_factor,
    get_all_emission_factors,

    # Reference data
    EPA_EMISSION_FACTORS,
    GWP_VALUES,
)

# =============================================================================
# REGULATORY TRACKER
# =============================================================================

from .regulatory_tracker import (
    # Enumerations
    RegulatoryAgency,
    RegulationType,
    ComplianceStatus as RegulatoryComplianceStatus,
    PermitType,
    AlertPriority,

    # Data classes
    RegulatoryRequirement,
    EnvironmentalPermit,
    ComplianceRecord,
    ComplianceAlert,
    EfficiencyStandard,

    # Main class
    RegulatoryComplianceTracker,

    # Factory functions
    create_regulatory_tracker,
    get_requirements_by_agency,
    get_efficiency_standards,
    get_all_requirements,

    # Reference data
    STEAM_SYSTEM_REQUIREMENTS,
    EFFICIENCY_STANDARDS,
)

# =============================================================================
# MODULE VERSION
# =============================================================================

__version__ = "1.0.0"
__author__ = "GL-BackendDeveloper"

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Quality Standards - Enumerations
    "SteamQualityStandard",
    "SteamPressureClass",
    "SteamApplication",
    "QualityParameter",
    "QualityComplianceStatus",

    # Quality Standards - Data classes
    "QualityLimit",
    "BestPracticeThreshold",
    "SiteQualityRequirement",
    "QualityValidationResult",

    # Quality Standards - Main class
    "SteamQualityStandardsValidator",

    # Quality Standards - Factory functions
    "create_quality_validator",
    "get_asme_limits_for_pressure_class",
    "get_quality_limits_for_parameter",

    # Quality Standards - Reference data
    "ASME_QUALITY_LIMITS",
    "INDUSTRY_BEST_PRACTICES",

    # Measurement Governance - Enumerations
    "SensorType",
    "CalibrationStatus",
    "AccuracyClass",
    "DataQualityGrade",
    "MeasurementApplication",

    # Measurement Governance - Data classes
    "SensorRequirement",
    "RegisteredSensor",
    "CalibrationRecord",
    "DataQualityScore",
    "UncertaintyBudget",

    # Measurement Governance - Main classes
    "DataQualityScorer",
    "UncertaintyCalculator",
    "MeasurementGovernanceManager",

    # Measurement Governance - Factory functions
    "create_governance_manager",
    "create_data_quality_scorer",
    "create_uncertainty_calculator",
    "get_sensor_requirements",

    # Measurement Governance - Reference data
    "SENSOR_REQUIREMENTS",

    # Carbon Accounting - Enumerations
    "FuelType",
    "EmissionScope",
    "QualityImprovementType",
    "EmissionFactorSource",

    # Carbon Accounting - Data classes
    "EmissionFactor",
    "EnergySavingsResult",
    "EmissionsImpactResult",
    "MeterReconciliationResult",

    # Carbon Accounting - Main classes
    "EnergySavingsCalculator",
    "EmissionsImpactCalculator",
    "MeterReconciler",
    "CarbonAccountingManager",

    # Carbon Accounting - Factory functions
    "create_carbon_accounting_manager",
    "get_emission_factor",
    "get_all_emission_factors",

    # Carbon Accounting - Reference data
    "EPA_EMISSION_FACTORS",
    "GWP_VALUES",

    # Regulatory Tracker - Enumerations
    "RegulatoryAgency",
    "RegulationType",
    "RegulatoryComplianceStatus",
    "PermitType",
    "AlertPriority",

    # Regulatory Tracker - Data classes
    "RegulatoryRequirement",
    "EnvironmentalPermit",
    "ComplianceRecord",
    "ComplianceAlert",
    "EfficiencyStandard",

    # Regulatory Tracker - Main class
    "RegulatoryComplianceTracker",

    # Regulatory Tracker - Factory functions
    "create_regulatory_tracker",
    "get_requirements_by_agency",
    "get_efficiency_standards",
    "get_all_requirements",

    # Regulatory Tracker - Reference data
    "STEAM_SYSTEM_REQUIREMENTS",
    "EFFICIENCY_STANDARDS",
]
