"""
GL-013 PREDICTMAINT - Calculator Modules Package

This package provides comprehensive calculation capabilities for
predictive maintenance with zero-hallucination guarantee.

All calculations are:
- Deterministic (bit-perfect reproducibility)
- Provenance-tracked (SHA-256 hashed)
- Standards-compliant (ISO, IEEE, IEC)
- Fully documented

Modules:
---------
constants
    Physical constants, equipment parameters, and standards data.

units
    Unit conversion utilities with dimensional analysis.

provenance
    Audit trail, Merkle trees, and calculation replay.

rul_calculator
    Remaining Useful Life estimation using Weibull, Exponential,
    and Log-Normal reliability models.

failure_probability_calculator
    Failure probability and hazard rate calculations with
    multi-failure-mode aggregation and Bayesian updating.

vibration_analyzer
    ISO 10816 compliant vibration analysis, bearing fault
    frequency calculation, and spectrum analysis.

thermal_degradation_calculator
    Arrhenius-based thermal life estimation, hot spot calculation,
    and thermal cycling fatigue analysis.

maintenance_scheduler
    Optimal maintenance scheduling with cost optimization,
    work order prioritization, and resource scheduling.

spare_parts_calculator
    Inventory optimization with EOQ, safety stock, Poisson
    demand modeling, and critical spares analysis.

anomaly_detector
    Statistical anomaly detection including CUSUM, SPC,
    Mahalanobis distance, and Isolation Forest.

weibull_analysis_calculator
    Comprehensive Weibull analysis with MLE parameter estimation,
    Median Rank Regression, B-life calculations, and confidence intervals.

vibration_analysis_calculator
    Advanced vibration analysis with FFT spectrum analysis,
    order analysis, bearing defect frequencies, envelope analysis,
    and ISO 10816 severity assessment.

Example Usage:
--------------
>>> from calculators import RULCalculator, VibrationAnalyzer
>>> from calculators.constants import MachineClass
>>> from decimal import Decimal

>>> # Calculate RUL for a pump
>>> rul_calc = RULCalculator()
>>> result = rul_calc.calculate_weibull_rul(
...     equipment_type="pump_centrifugal",
...     operating_hours=Decimal("30000"),
...     target_reliability="0.5"
... )
>>> print(f"RUL: {result.rul_hours} hours")

>>> # Analyze vibration severity
>>> vib_analyzer = VibrationAnalyzer()
>>> severity = vib_analyzer.assess_severity(
...     velocity_rms=Decimal("4.5"),
...     machine_class=MachineClass.CLASS_II
... )
>>> print(f"Zone: {severity.zone.name}")

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

# =============================================================================
# VERSION INFORMATION
# =============================================================================

__version__ = "1.0.0"
__author__ = "GL-CalculatorEngineer"
__license__ = "Proprietary - GreenLang"

# =============================================================================
# IMPORTS - Constants
# =============================================================================

from .constants import (
    # Physical constants
    BOLTZMANN_CONSTANT,
    BOLTZMANN_CONSTANT_EV,
    KELVIN_OFFSET,
    STANDARD_TEMP_K,
    STANDARD_GRAVITY,
    PI,
    E,

    # Weibull parameters
    WeibullParameters,
    WEIBULL_PARAMETERS,

    # Failure rates
    FAILURE_RATES_FPMH,

    # Arrhenius parameters
    ArrheniusParameters,
    ARRHENIUS_PARAMETERS,

    # Vibration standards
    MachineClass,
    VibrationZone,
    VibrationLimits,
    ISO_10816_VIBRATION_LIMITS,

    # Bearing data
    BearingGeometry,
    BEARING_GEOMETRIES,

    # Maintenance costs
    MaintenanceCostParameters,
    MAINTENANCE_COST_RATIOS,

    # Statistical constants
    Z_SCORES,
    CHI_SQUARE_CRITICAL,

    # Conversions
    TIME_TO_HOURS,
    VIBRATION_CONVERSIONS,
    celsius_to_kelvin,
    fahrenheit_to_kelvin,

    # Defaults
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_DECIMAL_PRECISION,
    DEFAULT_MONTE_CARLO_ITERATIONS,
    DEFAULT_PREDICTION_HORIZON_HOURS,
    MIN_PROBABILITY_THRESHOLD,
    MAX_PROBABILITY,
)

# =============================================================================
# IMPORTS - Units
# =============================================================================

from .units import (
    # Enums
    UnitCategory,

    # Data classes
    UnitDefinition,
    TemperatureUnit,
    ConversionResult,

    # Converter class
    UnitConverter,

    # Functions
    convert,
    convert_with_provenance,
    hours_to_years,
    years_to_hours,
    celsius_to_kelvin,
    kelvin_to_celsius,
    rpm_to_hz,
    hz_to_rpm,
    mm_s_to_in_s,
    in_s_to_mm_s,
    g_to_m_s2,
    m_s2_to_g,
    kw_to_hp,
    hp_to_kw,
)

# =============================================================================
# IMPORTS - Provenance
# =============================================================================

from .provenance import (
    # Enums
    CalculationType,
    ProvenanceStatus,

    # Data classes
    CalculationStep,
    ProvenanceRecord,
    MerkleNode,

    # Classes
    MerkleTree,
    ProvenanceBuilder,
    ProvenanceStore,
    InMemoryProvenanceStore,
    CalculationReplayer,

    # Functions
    calculate_hash,
    verify_hash,
    create_provenance_record,
    get_global_store,
    set_global_store,
    store_provenance,
    retrieve_provenance,
    verify_provenance,
)

# =============================================================================
# IMPORTS - RUL Calculator
# =============================================================================

from .rul_calculator import (
    # Enums
    ReliabilityModel,
    HealthState,

    # Constants
    HEALTH_ADJUSTMENT_FACTORS,

    # Data classes
    RULResult,
    ReliabilityProfile,
    ConfidenceInterval,

    # Main class
    RULCalculator,
)

# =============================================================================
# IMPORTS - Failure Probability Calculator
# =============================================================================

from .failure_probability_calculator import (
    # Enums
    DistributionType,
    FailureMode,

    # Data classes
    FailureProbabilityResult,
    HazardRateResult,
    MultiModeFailureResult,
    BayesianUpdateResult,

    # Main class
    FailureProbabilityCalculator,
)

# =============================================================================
# IMPORTS - Vibration Analyzer
# =============================================================================

from .vibration_analyzer import (
    # Enums
    VibrationMeasureType,
    FaultType,
    AlarmLevel,

    # Data classes
    VibrationSeverityResult,
    BearingFaultFrequencies,
    SpectrumPeak,
    SpectrumAnalysisResult,
    TrendAnalysisResult,
    EnvelopeAnalysisResult,

    # Main class
    VibrationAnalyzer,
)

# =============================================================================
# IMPORTS - Thermal Degradation Calculator
# =============================================================================

from .thermal_degradation_calculator import (
    # Enums
    InsulationClass,
    LoadingType,
    EquipmentCategory,

    # Constants
    INSULATION_RATINGS,
    IEEE_REFERENCE_LIFE_HOURS,

    # Data classes
    ThermalLifeResult,
    AgingAccelerationResult,
    HotSpotResult,
    ThermalCycleResult,
    OverloadAssessmentResult,

    # Main class
    ThermalDegradationCalculator,
)

# =============================================================================
# IMPORTS - Maintenance Scheduler
# =============================================================================

from .maintenance_scheduler import (
    # Enums
    MaintenancePolicy,
    PriorityLevel,
    MaintenanceType,
    ResourceType,

    # Data classes
    OptimalIntervalResult,
    CostOptimizationResult,
    WorkOrderPriority,
    ScheduleResult,
    CBMTrigger,

    # Main class
    MaintenanceScheduler,
)

# =============================================================================
# IMPORTS - Spare Parts Calculator
# =============================================================================

from .spare_parts_calculator import (
    # Enums
    PartCriticality,
    DemandPattern,
    ReplenishmentPolicy,

    # Constants
    SERVICE_LEVEL_Z,

    # Data classes
    InventoryOptimizationResult,
    EOQResult,
    SafetyStockResult,
    CriticalSpareResult,
    PoissonDemandResult,

    # Main class
    SparePartsCalculator,
)

# =============================================================================
# IMPORTS - Anomaly Detector
# =============================================================================

from .anomaly_detector import (
    # Enums
    AnomalyType,
    AnomalySeverity,
    ControlChartRule,

    # Data classes
    AnomalyResult,
    CUSUMResult,
    SPCResult,
    MahalanobisResult,
    IsolationForestResult,

    # Main class
    AnomalyDetector,
)

# =============================================================================
# IMPORTS - Weibull Analysis Calculator
# =============================================================================

from .weibull_analysis_calculator import (
    # Enums
    EstimationMethod,
    DataType,
    WeibullType,

    # Data classes
    WeibullParameterEstimate,
    BLifeResult,
    ReliabilityResult,
    HazardRateProfileResult,
    ConfidenceIntervalResult,
    MLEIterationResult,

    # Main class
    WeibullAnalysisCalculator,
)

# =============================================================================
# IMPORTS - Vibration Analysis Calculator
# =============================================================================

from .vibration_analysis_calculator import (
    # Enums
    VibrationParameter,
    AnalysisType,
    FaultCategory,
    SeverityLevel,
    TrendStatus,

    # Constants
    ISO_10816_EXTENDED_LIMITS,

    # Data classes
    FFTSpectrumResult,
    OrderAnalysisResult,
    BearingDefectFrequencies as AdvancedBearingDefectFrequencies,
    EnvelopeAnalysisResult as AdvancedEnvelopeAnalysisResult,
    OverallVibrationResult,
    TrendAnalysisResult as AdvancedTrendAnalysisResult,
    FaultDiagnosisResult,

    # Main class
    VibrationAnalysisCalculator,
)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",

    # ==========================================================================
    # CONSTANTS
    # ==========================================================================
    # Physical constants
    "BOLTZMANN_CONSTANT",
    "BOLTZMANN_CONSTANT_EV",
    "KELVIN_OFFSET",
    "STANDARD_TEMP_K",
    "STANDARD_GRAVITY",
    "PI",
    "E",

    # Weibull
    "WeibullParameters",
    "WEIBULL_PARAMETERS",
    "FAILURE_RATES_FPMH",

    # Arrhenius
    "ArrheniusParameters",
    "ARRHENIUS_PARAMETERS",

    # Vibration
    "MachineClass",
    "VibrationZone",
    "VibrationLimits",
    "ISO_10816_VIBRATION_LIMITS",

    # Bearings
    "BearingGeometry",
    "BEARING_GEOMETRIES",

    # Maintenance
    "MaintenanceCostParameters",
    "MAINTENANCE_COST_RATIOS",

    # Statistics
    "Z_SCORES",
    "CHI_SQUARE_CRITICAL",
    "SERVICE_LEVEL_Z",

    # Conversions
    "TIME_TO_HOURS",
    "VIBRATION_CONVERSIONS",

    # Defaults
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_DECIMAL_PRECISION",

    # ==========================================================================
    # UNITS
    # ==========================================================================
    "UnitCategory",
    "UnitDefinition",
    "TemperatureUnit",
    "ConversionResult",
    "UnitConverter",
    "convert",
    "convert_with_provenance",
    "hours_to_years",
    "years_to_hours",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "fahrenheit_to_kelvin",
    "rpm_to_hz",
    "hz_to_rpm",
    "mm_s_to_in_s",
    "in_s_to_mm_s",
    "g_to_m_s2",
    "m_s2_to_g",
    "kw_to_hp",
    "hp_to_kw",

    # ==========================================================================
    # PROVENANCE
    # ==========================================================================
    "CalculationType",
    "ProvenanceStatus",
    "CalculationStep",
    "ProvenanceRecord",
    "MerkleNode",
    "MerkleTree",
    "ProvenanceBuilder",
    "ProvenanceStore",
    "InMemoryProvenanceStore",
    "CalculationReplayer",
    "calculate_hash",
    "verify_hash",
    "create_provenance_record",
    "get_global_store",
    "set_global_store",
    "store_provenance",
    "retrieve_provenance",
    "verify_provenance",

    # ==========================================================================
    # RUL CALCULATOR
    # ==========================================================================
    "ReliabilityModel",
    "HealthState",
    "HEALTH_ADJUSTMENT_FACTORS",
    "RULResult",
    "ReliabilityProfile",
    "ConfidenceInterval",
    "RULCalculator",

    # ==========================================================================
    # FAILURE PROBABILITY
    # ==========================================================================
    "DistributionType",
    "FailureMode",
    "FailureProbabilityResult",
    "HazardRateResult",
    "MultiModeFailureResult",
    "BayesianUpdateResult",
    "FailureProbabilityCalculator",

    # ==========================================================================
    # VIBRATION ANALYZER
    # ==========================================================================
    "VibrationMeasureType",
    "FaultType",
    "AlarmLevel",
    "VibrationSeverityResult",
    "BearingFaultFrequencies",
    "SpectrumPeak",
    "SpectrumAnalysisResult",
    "TrendAnalysisResult",
    "EnvelopeAnalysisResult",
    "VibrationAnalyzer",

    # ==========================================================================
    # THERMAL DEGRADATION
    # ==========================================================================
    "InsulationClass",
    "LoadingType",
    "EquipmentCategory",
    "INSULATION_RATINGS",
    "IEEE_REFERENCE_LIFE_HOURS",
    "ThermalLifeResult",
    "AgingAccelerationResult",
    "HotSpotResult",
    "ThermalCycleResult",
    "OverloadAssessmentResult",
    "ThermalDegradationCalculator",

    # ==========================================================================
    # MAINTENANCE SCHEDULER
    # ==========================================================================
    "MaintenancePolicy",
    "PriorityLevel",
    "MaintenanceType",
    "ResourceType",
    "OptimalIntervalResult",
    "CostOptimizationResult",
    "WorkOrderPriority",
    "ScheduleResult",
    "CBMTrigger",
    "MaintenanceScheduler",

    # ==========================================================================
    # SPARE PARTS
    # ==========================================================================
    "PartCriticality",
    "DemandPattern",
    "ReplenishmentPolicy",
    "InventoryOptimizationResult",
    "EOQResult",
    "SafetyStockResult",
    "CriticalSpareResult",
    "PoissonDemandResult",
    "SparePartsCalculator",

    # ==========================================================================
    # ANOMALY DETECTOR
    # ==========================================================================
    "AnomalyType",
    "AnomalySeverity",
    "ControlChartRule",
    "AnomalyResult",
    "CUSUMResult",
    "SPCResult",
    "MahalanobisResult",
    "IsolationForestResult",
    "AnomalyDetector",

    # ==========================================================================
    # WEIBULL ANALYSIS CALCULATOR
    # ==========================================================================
    "EstimationMethod",
    "DataType",
    "WeibullType",
    "WeibullParameterEstimate",
    "BLifeResult",
    "ReliabilityResult",
    "HazardRateProfileResult",
    "ConfidenceIntervalResult",
    "MLEIterationResult",
    "WeibullAnalysisCalculator",

    # ==========================================================================
    # VIBRATION ANALYSIS CALCULATOR (Advanced)
    # ==========================================================================
    "VibrationParameter",
    "AnalysisType",
    "FaultCategory",
    "SeverityLevel",
    "TrendStatus",
    "ISO_10816_EXTENDED_LIMITS",
    "FFTSpectrumResult",
    "OrderAnalysisResult",
    "AdvancedBearingDefectFrequencies",
    "AdvancedEnvelopeAnalysisResult",
    "OverallVibrationResult",
    "AdvancedTrendAnalysisResult",
    "FaultDiagnosisResult",
    "VibrationAnalysisCalculator",
]


# =============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# =============================================================================

def create_rul_calculator(**kwargs) -> RULCalculator:
    """Create a configured RUL Calculator."""
    return RULCalculator(**kwargs)


def create_failure_calculator(**kwargs) -> FailureProbabilityCalculator:
    """Create a configured Failure Probability Calculator."""
    return FailureProbabilityCalculator(**kwargs)


def create_vibration_analyzer(**kwargs) -> VibrationAnalyzer:
    """Create a configured Vibration Analyzer."""
    return VibrationAnalyzer(**kwargs)


def create_thermal_calculator(**kwargs) -> ThermalDegradationCalculator:
    """Create a configured Thermal Degradation Calculator."""
    return ThermalDegradationCalculator(**kwargs)


def create_maintenance_scheduler(**kwargs) -> MaintenanceScheduler:
    """Create a configured Maintenance Scheduler."""
    return MaintenanceScheduler(**kwargs)


def create_spare_parts_calculator(**kwargs) -> SparePartsCalculator:
    """Create a configured Spare Parts Calculator."""
    return SparePartsCalculator(**kwargs)


def create_anomaly_detector(**kwargs) -> AnomalyDetector:
    """Create a configured Anomaly Detector."""
    return AnomalyDetector(**kwargs)


def create_weibull_calculator(**kwargs) -> WeibullAnalysisCalculator:
    """Create a configured Weibull Analysis Calculator."""
    return WeibullAnalysisCalculator(**kwargs)


def create_vibration_analysis_calculator(**kwargs) -> VibrationAnalysisCalculator:
    """Create a configured Vibration Analysis Calculator."""
    return VibrationAnalysisCalculator(**kwargs)


# Add factory functions to exports
__all__.extend([
    "create_rul_calculator",
    "create_failure_calculator",
    "create_vibration_analyzer",
    "create_thermal_calculator",
    "create_maintenance_scheduler",
    "create_spare_parts_calculator",
    "create_anomaly_detector",
    "create_weibull_calculator",
    "create_vibration_analysis_calculator",
])
