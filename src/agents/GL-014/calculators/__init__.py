# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Calculators Module.

Zero-hallucination calculation engines for heat exchanger analysis including:
- Economic impact assessment
- Energy loss calculations
- Production impact analysis
- Maintenance cost modeling
- Total Cost of Ownership (TCO)
- ROI and payback analysis
- Sensitivity analysis with Monte Carlo support
- Carbon cost integration
- Cleaning schedule optimization
- Comprehensive constants library (TEMA fouling, correlations, materials)
- Unit conversion utilities (SI/Imperial bidirectional)
- Provenance tracking for audit trails

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from .economic_calculator import (
    # Main Calculator
    EconomicCalculator,

    # Enumerations
    FuelType,
    CleaningMethod,
    MaintenanceType,
    DepreciationMethod,
    EmissionScope,

    # Input Data Classes
    EnergyLossInput,
    ProductionImpactInput,
    MaintenanceCostInput,
    TCOInput,
    ROIInput,
    SensitivityInput,
    CarbonImpactInput,

    # Output Data Classes
    CalculationStep,
    EnergyLossResult,
    ProductionImpactResult,
    MaintenanceCostResult,
    TCOResult,
    ROIResult,
    SensitivityResult,
    MonteCarloResult,
    CarbonImpactResult,
    EconomicReport,

    # Constants
    GWP_AR6,
    CO2_EMISSION_FACTORS,
    MACRS_5_YEAR,
    MACRS_7_YEAR,
    MACRS_10_YEAR,
    MACRS_15_YEAR,
)

from .cleaning_optimizer import (
    # Cleaning Optimizer
    CleaningOptimizer,

    # Cleaning Enumerations
    CleaningMethod as CleaningMethodOptimizer,
    FoulingType,
    RiskCategory,
    SchedulePriority,

    # Cleaning Parameter Classes
    CleaningCostParameters,
    EnergyParameters,
    FoulingParameters,
    ExchangerSpecification,
    CleaningMethodCharacteristics,

    # Cleaning Result Classes
    OptimalIntervalResult,
    CostBenefitResult,
    CleaningMethodSelection,
    FleetScheduleEntry,
    FleetScheduleResult,
    RiskAssessmentResult,
    CleaningScheduleResult,
    ROIResult as CleaningROIResult,

    # Utility Functions
    get_cleaning_method_characteristics,
    validate_calculation_result,
    verify_reproducibility,
)

from .performance_tracker import (
    # Performance Tracker
    PerformanceTracker,

    # Performance Enumerations
    DegradationModel,
    HealthStatus,
    TrendDirection,
    SeasonalPattern,

    # Performance Constants
    HEALTH_THRESHOLDS,
    DEFAULT_HEALTH_WEIGHTS,
    MIN_ACCEPTABLE_EFFICIENCY,
    MIN_ACCEPTABLE_U_VALUE_RATIO,
    MAX_ACCEPTABLE_PRESSURE_DROP_RATIO,

    # Performance Result Classes
    ThermalPerformanceResult,
    HydraulicPerformanceResult,
    DegradationTrendResult,
    PerformancePatternResult,
    BenchmarkResult,
    HealthIndexResult,
    RemainingPerformanceLifeResult,
    PerformanceReportResult,
    CalculationStep as PerformanceCalculationStep,
)

from .fouling_calculator import (
    # Main Fouling Calculator
    FoulingCalculator,
    create_fouling_calculator,

    # Fouling Enumerations
    ExchangerType,
    FluidType,
    FoulingMechanism,
    ScalingType,
    FoulingSeverity,

    # Fouling Input Models
    FoulingResistanceInput,
    FoulingRateInput,
    KernSeatonInput,
    EbertPanchalInput,
    FoulingClassificationInput,
    FoulingSeverityInput,
    FoulingPredictionInput,
    TimeToCleaningInput,

    # Fouling Result Models (Immutable)
    FoulingResistanceResult,
    FoulingRateResult,
    KernSeatonResult,
    EbertPanchalResult,
    FoulingClassificationResult,
    FoulingSeverityResult,
    FoulingPredictionResult,
    TimeToCleaningResult,

    # Fouling Constants
    TEMA_FOULING_FACTORS,
    ACTIVATION_ENERGIES,
    GAS_CONSTANT_R,
)

# Constants Library - Comprehensive lookup tables
from .constants import (
    # Fouling Categories and Data
    FoulingCategory,
    FoulingFactorData,
    FoulingFactors,

    # Heat Transfer Correlations
    CorrelationCoefficients,
    HeatTransferCorrelations,

    # Material Properties
    MaterialThermalData,
    MaterialProperties,

    # Fluid Property Tables
    FluidPropertyPoint,
    FluidPropertyTables,

    # TEMA Shell Types
    ShellTypeData,
    TEMAShellTypes,

    # Baffle Configurations
    BaffleConfiguration,
    BaffleConfigurations,

    # Tube Patterns
    TubePatternData,
    TubePatterns,

    # Cleaning Parameters
    CleaningMethodData,
    CleaningParameters,

    # Standard Tube Dimensions
    TubeDimension,
    StandardTubeDimensions,

    # Physical Constants
    PhysicalConstants,
)

# Unit Conversion Library
from .units import (
    # Unit Enumerations
    TemperatureUnit,
    PressureUnit,
    FlowRateUnit,
    HeatTransferCoefficientUnit,
    ThermalConductivityUnit,
    FoulingResistanceUnit,
    HeatDutyUnit,
    AreaUnit,
    LengthUnit,
    ViscosityUnit,
    DensityUnit,
    SpecificHeatUnit,
    VelocityUnit,

    # Conversion Classes
    ConversionFactor,
    UnitConverter,

    # Convenience Function
    convert,
)

# Provenance Tracking Library
from .provenance import (
    # Status Enumerations
    CalculationStatus,
    ProvenanceLevel,

    # Data Classes
    CalculationStep as ProvenanceCalculationStep,
    ProvenanceRecord,

    # Builder Classes
    ProvenanceBuilder,
    StepBuilder,

    # Utility Functions
    create_calculation_hash,
    verify_provenance_record,
    load_provenance_from_json,
    save_provenance_to_file,
    load_provenance_from_file,
)

from .heat_transfer_calculator import (
    # Heat Transfer Calculator
    HeatTransferCalculator,

    # Heat Transfer Enumerations
    FlowArrangement,
    CorrelationType,
    FluidPhase,
    TubeLayout,

    # Heat Transfer Lookup Tables
    TEMA_FOULING_FACTORS as TEMA_FOULING_FACTORS_HT,
    TUBE_MATERIAL_CONDUCTIVITY,
    STANDARD_TUBE_DIMENSIONS,

    # Heat Transfer Data Classes
    TubeDimensions,
    CalculationStep as HeatTransferCalculationStep,
    ProvenanceRecord as HeatTransferProvenanceRecord,
    ProvenanceBuilder as HeatTransferProvenanceBuilder,
    OverallCoefficientResult,
    LMTDResult,
    EffectivenessNTUResult,
    HeatDutyResult,
    ThermalResistanceResult,
    FilmCoefficientResult,
    EnergyBalanceResult,
)

# Predictive Fouling Engine - Advanced Analytics
from .predictive_fouling_engine import (
    # Predictive Fouling Engine
    PredictiveFoulingEngine,

    # Fouling Engine Constants
    DEFAULT_DECIMAL_PRECISION as FOULING_DECIMAL_PRECISION,
    Z_SCORES as FOULING_Z_SCORES,
    FOULING_THRESHOLDS,

    # Fouling Engine Enumerations
    FoulingRegime,
    AnomalyType as FoulingAnomalyType,
    AnomalySeverity as FoulingAnomalySeverity,
    TrendDirection as FoulingTrendDirection,
    SeasonalPeriod,

    # Fouling Engine Result Classes
    TrendAnalysisResult,
    SeasonalDecompositionResult,
    ForecastResult,
    AnomalyDetectionResult as FoulingAnomalyDetectionResult,
    PatternRecognitionResult,
    ThresholdPredictionResult,
    CorrelationResult,
    ModelValidationResult,
    PredictionReportResult,
    CalculationStep as FoulingCalculationStep,
)

# TEMA Design Calculator - Shell-and-Tube Design per TEMA Standards
from .tema_design_calculator import (
    # TEMA Design Calculator
    TEMADesignCalculator,
    create_tema_calculator,

    # TEMA Enumerations
    TEMAFrontEnd,
    TEMAShellType,
    TEMARearEnd,
    TEMAClass,
    TubeLayout as TEMATubeLayout,
    BaffleType as TEMABaffleType,
    FlowRegime as TEMAFlowRegime,

    # TEMA Input Data Classes
    FluidProperties as TEMAFluidProperties,
    TubeSpecification,
    ShellSpecification,
    BaffleSpecification,
    TEMADesignInput,

    # TEMA Result Data Classes
    CalculationStep as TEMACalculationStep,
    TubeCountResult,
    BaffleDesignResult,
    ShellSidePressureDropResult,
    TubeSidePressureDropResult,
    TEMAClearanceResult,
    TEMADesignResult,

    # TEMA Constants
    TEMAClearances,
    TEMATubeDimensions,
    STANDARD_TUBES,
    TUBE_COUNT_CONSTANTS,
    TUBE_COUNT_PASS_FACTORS,
)

# NTU-Effectiveness Calculator - Epsilon-NTU Method
from .ntu_effectiveness_calculator import (
    # NTU Calculator
    NTUEffectivenessCalculator,
    create_ntu_calculator,

    # NTU Enumerations
    FlowConfiguration,
    CalculationMode,
    FluidSide,

    # NTU Input Data Classes
    FluidStream,
    HeatExchangerGeometry,
    NTUEffectivenessInput,

    # NTU Result Data Classes
    CalculationStep as NTUCalculationStep,
    EffectivenessResult,
    NTUFromEffectivenessResult,
    ConfigurationComparisonResult,
    CapacityRatioSensitivityResult,

    # NTU Constants
    DECIMAL_PRECISION as NTU_DECIMAL_PRECISION,
    MAX_ITERATIONS as NTU_MAX_ITERATIONS,
    CONVERGENCE_TOLERANCE as NTU_CONVERGENCE_TOLERANCE,
)

__all__ = [
    # Main Economic Calculator
    "EconomicCalculator",

    # Economic Enumerations
    "FuelType",
    "CleaningMethod",
    "MaintenanceType",
    "DepreciationMethod",
    "EmissionScope",

    # Economic Input Data Classes
    "EnergyLossInput",
    "ProductionImpactInput",
    "MaintenanceCostInput",
    "TCOInput",
    "ROIInput",
    "SensitivityInput",
    "CarbonImpactInput",

    # Economic Output Data Classes
    "CalculationStep",
    "EnergyLossResult",
    "ProductionImpactResult",
    "MaintenanceCostResult",
    "TCOResult",
    "ROIResult",
    "SensitivityResult",
    "MonteCarloResult",
    "CarbonImpactResult",
    "EconomicReport",

    # Economic Constants
    "GWP_AR6",
    "CO2_EMISSION_FACTORS",
    "MACRS_5_YEAR",
    "MACRS_7_YEAR",
    "MACRS_10_YEAR",
    "MACRS_15_YEAR",

    # Cleaning Optimizer
    "CleaningOptimizer",

    # Cleaning Enumerations
    "CleaningMethodOptimizer",
    "FoulingType",
    "RiskCategory",
    "SchedulePriority",

    # Cleaning Parameter Classes
    "CleaningCostParameters",
    "EnergyParameters",
    "FoulingParameters",
    "ExchangerSpecification",
    "CleaningMethodCharacteristics",

    # Cleaning Result Classes
    "OptimalIntervalResult",
    "CostBenefitResult",
    "CleaningMethodSelection",
    "FleetScheduleEntry",
    "FleetScheduleResult",
    "RiskAssessmentResult",
    "CleaningScheduleResult",
    "CleaningROIResult",

    # Utility Functions
    "get_cleaning_method_characteristics",
    "validate_calculation_result",
    "verify_reproducibility",

    # Performance Tracker
    "PerformanceTracker",

    # Performance Enumerations
    "DegradationModel",
    "HealthStatus",
    "TrendDirection",
    "SeasonalPattern",

    # Performance Constants
    "HEALTH_THRESHOLDS",
    "DEFAULT_HEALTH_WEIGHTS",
    "MIN_ACCEPTABLE_EFFICIENCY",
    "MIN_ACCEPTABLE_U_VALUE_RATIO",
    "MAX_ACCEPTABLE_PRESSURE_DROP_RATIO",

    # Performance Result Classes
    "ThermalPerformanceResult",
    "HydraulicPerformanceResult",
    "DegradationTrendResult",
    "PerformancePatternResult",
    "BenchmarkResult",
    "HealthIndexResult",
    "RemainingPerformanceLifeResult",
    "PerformanceReportResult",
    "PerformanceCalculationStep",

    # Fouling Calculator
    "FoulingCalculator",
    "create_fouling_calculator",

    # Fouling Enumerations
    "ExchangerType",
    "FluidType",
    "FoulingMechanism",
    "ScalingType",
    "FoulingSeverity",

    # Fouling Input Models
    "FoulingResistanceInput",
    "FoulingRateInput",
    "KernSeatonInput",
    "EbertPanchalInput",
    "FoulingClassificationInput",
    "FoulingSeverityInput",
    "FoulingPredictionInput",
    "TimeToCleaningInput",

    # Fouling Result Models (Immutable)
    "FoulingResistanceResult",
    "FoulingRateResult",
    "KernSeatonResult",
    "EbertPanchalResult",
    "FoulingClassificationResult",
    "FoulingSeverityResult",
    "FoulingPredictionResult",
    "TimeToCleaningResult",

    # Fouling Constants
    "TEMA_FOULING_FACTORS",
    "ACTIVATION_ENERGIES",
    "GAS_CONSTANT_R",

    # =========================================================================
    # Constants Library
    # =========================================================================

    # Fouling Categories and Data
    "FoulingCategory",
    "FoulingFactorData",
    "FoulingFactors",

    # Heat Transfer Correlations
    "CorrelationCoefficients",
    "HeatTransferCorrelations",

    # Material Properties
    "MaterialThermalData",
    "MaterialProperties",

    # Fluid Property Tables
    "FluidPropertyPoint",
    "FluidPropertyTables",

    # TEMA Shell Types
    "ShellTypeData",
    "TEMAShellTypes",

    # Baffle Configurations
    "BaffleConfiguration",
    "BaffleConfigurations",

    # Tube Patterns
    "TubePatternData",
    "TubePatterns",

    # Cleaning Parameters (from constants)
    "CleaningMethodData",
    "CleaningParameters",

    # Standard Tube Dimensions
    "TubeDimension",
    "StandardTubeDimensions",

    # Physical Constants
    "PhysicalConstants",

    # =========================================================================
    # Unit Conversion Library
    # =========================================================================

    # Unit Enumerations
    "TemperatureUnit",
    "PressureUnit",
    "FlowRateUnit",
    "HeatTransferCoefficientUnit",
    "ThermalConductivityUnit",
    "FoulingResistanceUnit",
    "HeatDutyUnit",
    "AreaUnit",
    "LengthUnit",
    "ViscosityUnit",
    "DensityUnit",
    "SpecificHeatUnit",
    "VelocityUnit",

    # Conversion Classes and Functions
    "ConversionFactor",
    "UnitConverter",
    "convert",

    # =========================================================================
    # Provenance Tracking Library
    # =========================================================================

    # Status Enumerations
    "CalculationStatus",
    "ProvenanceLevel",

    # Data Classes
    "ProvenanceCalculationStep",
    "ProvenanceRecord",

    # Builder Classes
    "ProvenanceBuilder",
    "StepBuilder",

    # Utility Functions
    "create_calculation_hash",
    "verify_provenance_record",
    "load_provenance_from_json",
    "save_provenance_to_file",
    "load_provenance_from_file",

    # =========================================================================
    # Predictive Fouling Engine
    # =========================================================================

    # Main Engine
    "PredictiveFoulingEngine",

    # Fouling Engine Constants
    "FOULING_DECIMAL_PRECISION",
    "FOULING_Z_SCORES",
    "FOULING_THRESHOLDS",

    # Fouling Engine Enumerations
    "FoulingRegime",
    "FoulingAnomalyType",
    "FoulingAnomalySeverity",
    "FoulingTrendDirection",
    "SeasonalPeriod",

    # Fouling Engine Result Classes
    "TrendAnalysisResult",
    "SeasonalDecompositionResult",
    "ForecastResult",
    "FoulingAnomalyDetectionResult",
    "PatternRecognitionResult",
    "ThresholdPredictionResult",
    "CorrelationResult",
    "ModelValidationResult",
    "PredictionReportResult",
    "FoulingCalculationStep",

    # =========================================================================
    # TEMA Design Calculator
    # =========================================================================

    # TEMA Calculator
    "TEMADesignCalculator",
    "create_tema_calculator",

    # TEMA Enumerations
    "TEMAFrontEnd",
    "TEMAShellType",
    "TEMARearEnd",
    "TEMAClass",
    "TEMATubeLayout",
    "TEMABaffleType",
    "TEMAFlowRegime",

    # TEMA Input Data Classes
    "TEMAFluidProperties",
    "TubeSpecification",
    "ShellSpecification",
    "BaffleSpecification",
    "TEMADesignInput",

    # TEMA Result Data Classes
    "TEMACalculationStep",
    "TubeCountResult",
    "BaffleDesignResult",
    "ShellSidePressureDropResult",
    "TubeSidePressureDropResult",
    "TEMAClearanceResult",
    "TEMADesignResult",

    # TEMA Constants
    "TEMAClearances",
    "TEMATubeDimensions",
    "STANDARD_TUBES",
    "TUBE_COUNT_CONSTANTS",
    "TUBE_COUNT_PASS_FACTORS",

    # =========================================================================
    # NTU-Effectiveness Calculator
    # =========================================================================

    # NTU Calculator
    "NTUEffectivenessCalculator",
    "create_ntu_calculator",

    # NTU Enumerations
    "FlowConfiguration",
    "CalculationMode",
    "FluidSide",

    # NTU Input Data Classes
    "FluidStream",
    "HeatExchangerGeometry",
    "NTUEffectivenessInput",

    # NTU Result Data Classes
    "NTUCalculationStep",
    "EffectivenessResult",
    "NTUFromEffectivenessResult",
    "ConfigurationComparisonResult",
    "CapacityRatioSensitivityResult",

    # NTU Constants
    "NTU_DECIMAL_PRECISION",
    "NTU_MAX_ITERATIONS",
    "NTU_CONVERGENCE_TOLERANCE",
]
