"""
GL-010 EMISSIONWATCH Calculator Modules.

This package provides zero-hallucination calculator modules for emissions
compliance and air quality monitoring. All calculations are deterministic,
physics-based, and include full provenance tracking.

Modules:
    nox_calculator: NOx emissions calculations (Zeldovich, EPA Method 19)
    sox_calculator: SOx emissions calculations (mass balance, scrubber modeling)
    co2_calculator: CO2 emissions calculations (GHG Protocol, CFR Part 98)
    particulate_calculator: PM10/PM2.5 calculations (EPA Method 5)
    emission_factors: EPA AP-42 emission factors database
    fuel_analyzer: Fuel composition and property analysis
    combustion_stoichiometry: Combustion chemistry calculations
    compliance_checker: Multi-jurisdiction regulatory compliance
    violation_detector: Real-time exceedance detection
    report_generator: EPA CEDRI and EU E-PRTR report generation
    dispersion_model: Gaussian plume dispersion modeling
    provenance: SHA-256 hashing and audit trails

Zero-Hallucination Guarantee:
    - All calculations are deterministic (same input -> same output)
    - No LLM involvement in calculation path
    - Full provenance tracking with SHA-256 hashes
    - Complete audit trails for regulatory compliance

Example:
    >>> from calculators import NOxCalculator, calculate_nox_from_fuel
    >>> result = calculate_nox_from_fuel(
    ...     fuel_type="natural_gas",
    ...     heat_input_mmbtu_hr=100,
    ...     excess_air_percent=15
    ... )
    >>> print(f"NOx: {result.total_nox} {result.unit}")
"""

from .constants import (
    # Physical constants
    MW, F_FACTORS, GWP_100,
    # Standard conditions
    EPA_STD_TEMP_K, EPA_STD_PRESSURE_KPA,
    NORMAL_TEMP_K, NORMAL_PRESSURE_KPA,
    # Conversion factors
    LB_TO_KG, KG_TO_LB, MMBTU_TO_GJ,
    BTU_TO_JOULE, GAL_TO_LITER, FT3_TO_M3,
    # Reference O2 levels
    O2_REFERENCE,
)

from .units import (
    UnitConverter,
    GasLawCalculator,
    UnitConversionResult,
    MassUnit, VolumeUnit, EnergyUnit, TemperatureUnit,
    PressureUnit, ConcentrationUnit, EmissionRateUnit,
)

from .nox_calculator import (
    NOxCalculator,
    NOxCalculationResult,
    NOxSpecies, CombustionType, NOxControlDevice,
    CombustionInput, CEMSDataInput,
    calculate_nox_from_fuel,
    calculate_nox_from_cems,
)

from .sox_calculator import (
    SOxCalculator,
    SOxCalculationResult,
    SOxSpecies, SOxControlDevice,
    FuelSulfurInput, ScrubberInput, CEMSSOxInput,
    calculate_sox_from_fuel,
    calculate_sox_from_cems,
)

from .co2_calculator import (
    CO2Calculator,
    CO2CalculationResult,
    CarbonSource, EmissionScope, CalculationMethod,
    CarbonCaptureType, CarbonCaptureInput,
    FuelCarbonInput, DefaultFactorInput,
    calculate_co2_from_fuel,
    calculate_co2e,
)

from .particulate_calculator import (
    PMCalculator,
    PMCalculationResult,
    PMSize, PMType, PMControlDevice, CombustionSource,
    FuelPMInput, StackTestInput, ControlDeviceInput,
    calculate_pm_from_fuel,
)

from .emission_factors import (
    EmissionFactorDatabase,
    EmissionFactor,
    EmissionFactorMetadata,
    EmissionFactorSource,
    PollutantType, FuelCategory,
    EmissionFactorQuery,
    get_emission_factor_database,
    lookup_emission_factor,
)

from .fuel_analyzer import (
    FuelAnalyzer,
    FuelProperties,
    UltimateAnalysis,
    ProximateAnalysis,
    NaturalGasComposition,
    FuelState, AnalysisBasis,
    analyze_coal,
    get_fuel_properties,
)

from .combustion_stoichiometry import (
    CombustionStoichiometry,
    CombustionResult,
    FuelComposition,
    CombustionConditions,
    CombustionMode,
    calculate_combustion_products,
    get_excess_air_from_o2,
)

from .compliance_checker import (
    ComplianceChecker,
    ComplianceReport,
    ComplianceCheckResult,
    EmissionLimit,
    Jurisdiction,
    RegulatoryProgram,
    AveragingPeriod,
    ComplianceStatus,
    SourceCategory,
    EmissionLimitQuery,
    MeasuredEmission,
    check_nox_compliance,
    get_compliance_checker,
)

from .violation_detector import (
    ViolationDetector,
    ViolationAlert,
    AlertSeverity,
    AlertType,
    DataQuality,
    TrendAnalysis,
    EmissionDataPoint,
    DetectionConfig,
    create_detector,
    check_for_violations,
)

from .report_generator import (
    ReportGenerator,
    EmissionReport,
    ReportMetadata,
    FacilityInfo,
    EmissionUnit,
    EmissionData,
    ExcessEmissionEvent,
    ReportFormat,
    ReportType,
    ReportingPeriod,
    create_quarterly_report,
    create_cedri_report,
)

from .dispersion_model import (
    GaussianPlumeModel,
    DispersionResult,
    MaxConcentrationResult,
    StackParameters,
    MeteorologicalConditions,
    ReceptorLocation,
    StabilityClass,
    TerrainType,
    calculate_ground_level_concentration,
    find_max_impact,
)

from .provenance import (
    ProvenanceTracker,
    CalculationProvenance,
    AuditTrail,
    ProvenanceEvent,
    ProvenanceEventType,
    DataQualityLevel,
    ProvenanceInput,
    get_provenance_tracker,
    calculate_hash,
    create_provenance,
    verify_provenance,
)


__all__ = [
    # Constants
    "MW", "F_FACTORS", "GWP_100",
    "EPA_STD_TEMP_K", "EPA_STD_PRESSURE_KPA",
    "NORMAL_TEMP_K", "NORMAL_PRESSURE_KPA",
    "LB_TO_KG", "KG_TO_LB", "MMBTU_TO_GJ",
    "O2_REFERENCE",

    # Units
    "UnitConverter", "GasLawCalculator", "UnitConversionResult",
    "MassUnit", "VolumeUnit", "EnergyUnit", "TemperatureUnit",
    "PressureUnit", "ConcentrationUnit", "EmissionRateUnit",

    # NOx Calculator
    "NOxCalculator", "NOxCalculationResult",
    "NOxSpecies", "CombustionType", "NOxControlDevice",
    "CombustionInput", "CEMSDataInput",
    "calculate_nox_from_fuel", "calculate_nox_from_cems",

    # SOx Calculator
    "SOxCalculator", "SOxCalculationResult",
    "SOxSpecies", "SOxControlDevice",
    "FuelSulfurInput", "ScrubberInput", "CEMSSOxInput",
    "calculate_sox_from_fuel", "calculate_sox_from_cems",

    # CO2 Calculator
    "CO2Calculator", "CO2CalculationResult",
    "CarbonSource", "EmissionScope", "CalculationMethod",
    "CarbonCaptureType", "CarbonCaptureInput",
    "FuelCarbonInput", "DefaultFactorInput",
    "calculate_co2_from_fuel", "calculate_co2e",

    # Particulate Calculator
    "PMCalculator", "PMCalculationResult",
    "PMSize", "PMType", "PMControlDevice", "CombustionSource",
    "FuelPMInput", "StackTestInput", "ControlDeviceInput",
    "calculate_pm_from_fuel",

    # Emission Factors
    "EmissionFactorDatabase", "EmissionFactor",
    "EmissionFactorMetadata", "EmissionFactorSource",
    "PollutantType", "FuelCategory", "EmissionFactorQuery",
    "get_emission_factor_database", "lookup_emission_factor",

    # Fuel Analyzer
    "FuelAnalyzer", "FuelProperties",
    "UltimateAnalysis", "ProximateAnalysis", "NaturalGasComposition",
    "FuelState", "AnalysisBasis",
    "analyze_coal", "get_fuel_properties",

    # Combustion Stoichiometry
    "CombustionStoichiometry", "CombustionResult",
    "FuelComposition", "CombustionConditions", "CombustionMode",
    "calculate_combustion_products", "get_excess_air_from_o2",

    # Compliance Checker
    "ComplianceChecker", "ComplianceReport", "ComplianceCheckResult",
    "EmissionLimit", "Jurisdiction", "RegulatoryProgram",
    "AveragingPeriod", "ComplianceStatus", "SourceCategory",
    "EmissionLimitQuery", "MeasuredEmission",
    "check_nox_compliance", "get_compliance_checker",

    # Violation Detector
    "ViolationDetector", "ViolationAlert",
    "AlertSeverity", "AlertType", "DataQuality",
    "TrendAnalysis", "EmissionDataPoint", "DetectionConfig",
    "create_detector", "check_for_violations",

    # Report Generator
    "ReportGenerator", "EmissionReport",
    "ReportMetadata", "FacilityInfo", "EmissionUnit",
    "EmissionData", "ExcessEmissionEvent",
    "ReportFormat", "ReportType", "ReportingPeriod",
    "create_quarterly_report", "create_cedri_report",

    # Dispersion Model
    "GaussianPlumeModel", "DispersionResult", "MaxConcentrationResult",
    "StackParameters", "MeteorologicalConditions", "ReceptorLocation",
    "StabilityClass", "TerrainType",
    "calculate_ground_level_concentration", "find_max_impact",

    # Provenance
    "ProvenanceTracker", "CalculationProvenance",
    "AuditTrail", "ProvenanceEvent",
    "ProvenanceEventType", "DataQualityLevel", "ProvenanceInput",
    "get_provenance_tracker", "calculate_hash",
    "create_provenance", "verify_provenance",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EMISSIONWATCH Team"
