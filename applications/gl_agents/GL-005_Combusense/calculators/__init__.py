# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE Calculator Modules

Comprehensive calculation engines for combustion diagnostics with
100% determinism guarantee. All calculations are pure Python arithmetic
with no AI/ML in the calculation path.

Modules:
    - combustion_stability_calculator: Stability index and oscillation detection
    - fuel_air_optimizer: Optimal fuel-air ratio for target heat output
    - heat_output_calculator: Heat output and thermal efficiency calculations
    - pid_controller: PID control with anti-windup and auto-tuning
    - safety_validator: Safety interlock validation and limit checking
    - emissions_calculator: Emission calculations with regulatory compliance
    - feedforward_controller: Feedforward control for heat demand anticipation
    - stability_analyzer: Time-series stability analysis with FFT
    - air_fuel_ratio_calculator: Stoichiometric AFR and lambda calculations
    - combustion_performance_calculator: ASME PTC 4.1 performance analysis
    - combustion_diagnostics: Advanced flame pattern analysis, burner health,
      soot prediction, flashback/blowoff risk, maintenance recommendations
    - cqi_calculator: Combustion Quality Index (CQI) scoring with 5 sub-scores
    - explainability: SHAP/LIME feature attributions and attention visualization
    - narrative_generator: Natural-language explanation generation
    - anomaly_detection: Real-time anomaly detection with standardized taxonomy

Zero-Hallucination Guarantee:
    - Deterministic: Same inputs -> Same outputs (bit-perfect)
    - Reproducible: Complete provenance tracking
    - Auditable: SHA-256 hashes for all calculations
    - No LLM: Pure mathematical calculations only
    - Validated: Tested against known reference values

Reference Standards:
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces
    - ASME PTC 4.1: Fired Steam Generators Performance Test Codes
    - EPA 40 CFR Part 60: Standards of Performance
    - ISO 16001: Energy Management Systems
    - IEC 61508: Functional Safety Standards
    - IEC 61511: SIL 2 Safety Boundary Compliance
    - API 556: Instrumentation for Gas Fired Heaters
    - EN 746-2: Industrial Thermoprocessing Equipment Safety
"""

from .combustion_stability_calculator import (
    CombustionStabilityCalculator,
    StabilityInput,
    StabilityResult,
    StabilityLevel,
    OscillationPattern
)

from .fuel_air_optimizer import (
    FuelAirOptimizer,
    OptimizerInput,
    OptimizerResult,
    FuelType,
    OptimizationObjective,
    EmissionConstraints
)

from .heat_output_calculator import (
    HeatOutputCalculator,
    HeatOutputInput,
    HeatOutputResult,
    HeatLossCategory
)

from .pid_controller import (
    PIDController,
    PIDInput,
    PIDOutput,
    AutoTuneInput,
    AutoTuneOutput,
    ControlMode,
    TuningMethod,
    AntiWindupMethod,
    PIDGains
)

from .safety_validator import (
    SafetyValidator,
    SafetyValidatorInput,
    SafetyValidatorOutput,
    SafetyLevel,
    SafetyLimits,
    InterlockStatus,
    AlarmPriority
)

from .emissions_calculator import (
    EmissionsCalculator,
    EmissionsInput,
    EmissionsResult,
    EmissionType,
    RegulatoryStandard,
    ComplianceStatus
)

from .feedforward_controller import (
    FeedforwardController,
    FeedforwardInput,
    FeedforwardOutput,
    DisturbanceType,
    CompensationType,
    FeedforwardGains
)

from .stability_analyzer import (
    StabilityAnalyzer,
    StabilityInput as StabilityAnalysisInput,
    StabilityResult as StabilityAnalysisResult,
    StabilityLevel as StabilityLevelAnalysis,
    OscillationType
)

from .air_fuel_ratio_calculator import (
    AirFuelRatioCalculator,
    AirFuelRatioInput,
    AirFuelRatioOutput,
    FuelType as AFRFuelType,
    FuelComposition,
    StoichiometricProperties
)

from .combustion_performance_calculator import (
    CombustionPerformanceCalculator,
    CombustionPerformanceInput,
    CombustionPerformanceOutput,
    PerformanceLevel,
    LossCategory,
    HeatLoss
)

from .combustion_diagnostics import (
    # Main Calculator
    AdvancedCombustionDiagnosticsCalculator,
    CombustionDiagnosticsCalculator,  # Alias for backward compatibility

    # Input/Output Models
    AdvancedDiagnosticInput,
    AdvancedDiagnosticOutput,
    ZoneInput,
    DiagnosticInput,  # Alias
    DiagnosticOutput,  # Alias

    # Summary and Result Types
    AdvancedDiagnosticSummary,
    DiagnosticSummary,  # Alias

    # Enumerations
    FaultType,
    FaultSeverity,
    FlamePattern,
    CombustionMode,
    SensorType,
    TrendDirection,
    MaintenancePriority,
    BurnerHealthCategory,

    # Frozen Dataclass Results
    FaultDetectionResult,
    FlamePatternMetrics,
    IncompleteCombustionMetrics,
    EfficiencyDegradationTrend,
    BurnerHealthScore,
    FuelQualityVariation,
    ZoneAirDistribution,
    AirDistributionAnalysis,
    SootFormationPrediction,
    FlashbackBlowoffRisk,
    MaintenanceRecommendation,
    SensorDriftCompensation,
    CrossLimitParameters,
    TrimControlParameters,
    CombustionInstabilityIndicators,
    DiagnosticTrend,

    # Constants
    STOICHIOMETRIC_AFR,
    ADIABATIC_FLAME_TEMP,
    LAMINAR_FLAME_SPEED,
)

__all__ = [
    # Combustion Stability
    'CombustionStabilityCalculator',
    'StabilityInput',
    'StabilityResult',
    'StabilityLevel',
    'OscillationPattern',

    # Fuel-Air Optimization
    'FuelAirOptimizer',
    'OptimizerInput',
    'OptimizerResult',
    'FuelType',
    'OptimizationObjective',
    'EmissionConstraints',

    # Heat Output
    'HeatOutputCalculator',
    'HeatOutputInput',
    'HeatOutputResult',
    'HeatLossCategory',

    # PID Control
    'PIDController',
    'PIDInput',
    'PIDOutput',
    'AutoTuneInput',
    'AutoTuneOutput',
    'ControlMode',
    'TuningMethod',
    'AntiWindupMethod',
    'PIDGains',

    # Safety
    'SafetyValidator',
    'SafetyValidatorInput',
    'SafetyValidatorOutput',
    'SafetyLevel',
    'SafetyLimits',
    'InterlockStatus',
    'AlarmPriority',

    # Emissions
    'EmissionsCalculator',
    'EmissionsInput',
    'EmissionsResult',
    'EmissionType',
    'RegulatoryStandard',
    'ComplianceStatus',

    # Feedforward Control
    'FeedforwardController',
    'FeedforwardInput',
    'FeedforwardOutput',
    'DisturbanceType',
    'CompensationType',
    'FeedforwardGains',

    # Stability Analysis
    'StabilityAnalyzer',
    'StabilityAnalysisInput',
    'StabilityAnalysisResult',
    'StabilityLevelAnalysis',
    'OscillationType',

    # Air-Fuel Ratio
    'AirFuelRatioCalculator',
    'AirFuelRatioInput',
    'AirFuelRatioOutput',
    'AFRFuelType',
    'FuelComposition',
    'StoichiometricProperties',

    # Combustion Performance
    'CombustionPerformanceCalculator',
    'CombustionPerformanceInput',
    'CombustionPerformanceOutput',
    'PerformanceLevel',
    'LossCategory',
    'HeatLoss',

    # Advanced Combustion Diagnostics
    'AdvancedCombustionDiagnosticsCalculator',
    'CombustionDiagnosticsCalculator',
    'AdvancedDiagnosticInput',
    'AdvancedDiagnosticOutput',
    'ZoneInput',
    'DiagnosticInput',
    'DiagnosticOutput',
    'AdvancedDiagnosticSummary',
    'DiagnosticSummary',

    # Diagnostics Enumerations
    'FaultType',
    'FaultSeverity',
    'FlamePattern',
    'CombustionMode',
    'SensorType',
    'TrendDirection',
    'MaintenancePriority',
    'BurnerHealthCategory',

    # Diagnostics Result Types
    'FaultDetectionResult',
    'FlamePatternMetrics',
    'IncompleteCombustionMetrics',
    'EfficiencyDegradationTrend',
    'BurnerHealthScore',
    'FuelQualityVariation',
    'ZoneAirDistribution',
    'AirDistributionAnalysis',
    'SootFormationPrediction',
    'FlashbackBlowoffRisk',
    'MaintenanceRecommendation',
    'SensorDriftCompensation',
    'CrossLimitParameters',
    'TrimControlParameters',
    'CombustionInstabilityIndicators',
    'DiagnosticTrend',

    # Combustion Physics Constants
    'STOICHIOMETRIC_AFR',
    'ADIABATIC_FLAME_TEMP',
    'LAMINAR_FLAME_SPEED',

    # CQI Calculator (Playbook Section 8)
    'CQICalculator',
    'CQIConfiguration',
    'CQIResult',
    'CQISubScores',
    'CombustionSignals',
    'CQIGrade',
    'OperatingMode',
    'SafetyStatus',
    'DataQualityStatus',

    # Explainability (Playbook Section 10)
    'ExplainabilityEngine',
    'SHAPStyleExplainer',
    'LIMEStyleExplainer',
    'AttentionVisualizer',
    'FeatureAttribution',
    'ExplanationResult',
    'IncidentExplanation',
    'AttentionMap',

    # Narrative Generator (Playbook Section 10.4)
    'NarrativeGenerator',
    'NarrativeResult',
    'NarrativeInput',
    'EvidenceBundle',

    # Anomaly Detection (Playbook Section 9)
    'AnomalyDetector',
    'AnomalyEvent',
    'AnomalyType',
    'Severity',
    'DetectionLayer',
    'AnomalyStatus',
    'CombustionState',
    'DetectionConfig',
    'ANOMALY_TAXONOMY',
]

# CQI Calculator imports
from .cqi_calculator import (
    CQICalculator,
    CQIConfiguration,
    CQIResult,
    CQISubScores,
    CombustionSignals,
    CQIGrade,
    OperatingMode,
    SafetyStatus,
    DataQualityStatus,
    create_default_calculator as create_cqi_calculator,
    calculate_cqi_quick,
)

# Explainability imports
from .explainability import (
    ExplainabilityEngine,
    SHAPStyleExplainer,
    LIMEStyleExplainer,
    AttentionVisualizer,
    FeatureAttribution,
    ExplanationResult,
    IncidentExplanation,
    AttentionMap,
    ExplanationType,
    SignalSnapshot,
    SignalTimeSeries,
    create_default_engine as create_explainability_engine,
)

# Narrative Generator imports
from .narrative_generator import (
    NarrativeGenerator,
    NarrativeResult,
    NarrativeInput,
    EvidenceBundle,
    NarrativeStyle,
    NarrativeSeverity,
    create_default_generator as create_narrative_generator,
)

# Anomaly Detection imports
from .anomaly_detection import (
    AnomalyDetector,
    AnomalyEvent,
    AnomalyType,
    Severity,
    DetectionLayer,
    AnomalyStatus,
    CombustionState,
    DetectionConfig,
    RuleBasedDetector,
    StatisticalDetector,
    ANOMALY_TAXONOMY,
    create_default_detector as create_anomaly_detector,
)

__version__ = '2.1.0'
__author__ = 'GreenLang GL-005 Team'
__description__ = 'Zero-hallucination combustion diagnostics calculators with CQI, explainability, and anomaly detection'
