"""
GL-016 Waterguard Calculators

This package provides deterministic calculation engines for industrial water
chemistry monitoring and boiler blowdown optimization. All calculators use
deterministic methods (NO generative AI) for numeric calculations, ensuring
zero-hallucination compliance.

Modules:
    Anomaly Detection:
    - anomaly_detector: Multi-variate anomaly detection with Isolation Forest
    - drift_detector: Sensor drift detection (slow bias, step changes, flatline)
    - corrosion_detector: Corrosion event detection (iron/copper spikes)
    - o2_excursion_detector: Dissolved O2 monitoring for deaerator issues
    - conductivity_divergence: CoC model divergence detection
    - calibration_monitor: Analyzer calibration health monitoring

    Thermal Calculations:
    - provenance: SHA-256 cryptographic provenance tracking
    - thermal_calculator: IAPWS-IF97 blowdown energy loss calculations
    - savings_calculator: Water, energy, emissions savings with uncertainty
    - thermal_constraints: Thermal safety constraints and limits
    - heat_recovery: Flash tank and heat exchanger modeling

Author: GreenLang Waterguard Team
Version: 1.2.0
"""

__version__ = "1.2.0"

# =============================================================================
# Provenance Tracking
# =============================================================================
from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    ProvenanceValidator,
    CalculationStep,
    create_calculation_hash,
)

# =============================================================================
# Thermal Calculations (IAPWS-IF97)
# =============================================================================
from .thermal_calculator import (
    BlowdownThermalCalculator,
    BlowdownEnergyResult,
    IAPWS97Calculator,
    UnitConverter,
    ThermalResult,
)

# =============================================================================
# Savings Calculations
# =============================================================================
from .savings_calculator import (
    SavingsCalculator,
    SavingsResult,
    BaselineMethodology,
    EmissionFactorDatabase,
    FuelType,
    TimePeriod,
    UncertaintyBand,
)

# =============================================================================
# Thermal Safety Constraints
# =============================================================================
from .thermal_constraints import (
    BlowdownRampRateLimit,
    MinimumBlowdownInterval,
    ThermalShockProtection,
    ScalingRiskAssessment,
    ConstraintViolation,
    RiskLevel,
    ConstraintValidationResult,
    ThermalShockAssessment,
)

# =============================================================================
# Heat Recovery Systems
# =============================================================================
from .heat_recovery import (
    BlowdownHeatRecoverySystem,
    FlashTankModel,
    HeatExchangerModel,
    FlashSteamResult,
    HeatExchangerResult,
    RecoveredEnergyResult,
)

# =============================================================================
# Anomaly Detection
# =============================================================================
from .anomaly_detector import (
    WaterguardAnomalyDetector,
    AnomalyResult,
    AnomalyType,
    AnomalySeverity,
    AnomalyDetectorConfig,
    SensorData,
)

from .drift_detector import (
    SensorDriftDetector,
    DriftResult,
    DriftType,
    DriftDetectorConfig,
)

from .corrosion_detector import (
    CorrosionDetector,
    CorrosionEvent,
    CorrosionSeverity,
    AffectedSystem,
    CorrosionType,
    CorrosionDetectorConfig,
)

from .o2_excursion_detector import (
    DissolvedO2Monitor,
    O2ExcursionEvent,
    O2CauseHypothesis,
    O2Severity,
    O2Location,
    O2MonitorConfig,
)

from .conductivity_divergence import (
    ConductivityDivergenceDetector,
    DivergenceDetectorConfig,
    ConductivityReading,
    ContextualData,
    DivergenceEvent,
    DivergenceTrend,
    DivergenceCause,
    DivergenceSeverity,
    DivergenceDirection,
    create_divergence_detector,
)

from .calibration_monitor import (
    CalibrationMonitor,
    CalibrationMonitorConfig,
    AnalyzerSpec,
    CalibrationRecord,
    CalibrationStatusResult,
    DriftAnalysisResult,
    CMMSWorkOrder,
    AnalyzerType,
    CalibrationStatus,
    DriftSeverity as CalibrationDriftSeverity,
    WorkOrderPriority,
    WorkOrderType,
    create_calibration_monitor,
    create_analyzer_spec,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceRecord",
    "ProvenanceValidator",
    "CalculationStep",
    "create_calculation_hash",
    # Thermal Calculator
    "BlowdownThermalCalculator",
    "BlowdownEnergyResult",
    "IAPWS97Calculator",
    "UnitConverter",
    "ThermalResult",
    # Savings Calculator
    "SavingsCalculator",
    "SavingsResult",
    "BaselineMethodology",
    "EmissionFactorDatabase",
    "FuelType",
    "TimePeriod",
    "UncertaintyBand",
    # Thermal Constraints
    "BlowdownRampRateLimit",
    "MinimumBlowdownInterval",
    "ThermalShockProtection",
    "ScalingRiskAssessment",
    "ConstraintViolation",
    "RiskLevel",
    "ConstraintValidationResult",
    "ThermalShockAssessment",
    # Heat Recovery
    "BlowdownHeatRecoverySystem",
    "FlashTankModel",
    "HeatExchangerModel",
    "FlashSteamResult",
    "HeatExchangerResult",
    "RecoveredEnergyResult",
    # Anomaly Detection
    "WaterguardAnomalyDetector",
    "AnomalyResult",
    "AnomalyType",
    "AnomalySeverity",
    "AnomalyDetectorConfig",
    "SensorData",
    # Drift Detection
    "SensorDriftDetector",
    "DriftResult",
    "DriftType",
    "DriftDetectorConfig",
    # Corrosion Detection
    "CorrosionDetector",
    "CorrosionEvent",
    "CorrosionSeverity",
    "AffectedSystem",
    "CorrosionType",
    "CorrosionDetectorConfig",
    # O2 Excursion Detection
    "DissolvedO2Monitor",
    "O2ExcursionEvent",
    "O2CauseHypothesis",
    "O2Severity",
    "O2Location",
    "O2MonitorConfig",
    # Conductivity Divergence
    "ConductivityDivergenceDetector",
    "DivergenceDetectorConfig",
    "ConductivityReading",
    "ContextualData",
    "DivergenceEvent",
    "DivergenceTrend",
    "DivergenceCause",
    "DivergenceSeverity",
    "DivergenceDirection",
    "create_divergence_detector",
    # Calibration Monitoring
    "CalibrationMonitor",
    "CalibrationMonitorConfig",
    "AnalyzerSpec",
    "CalibrationRecord",
    "CalibrationStatusResult",
    "DriftAnalysisResult",
    "CMMSWorkOrder",
    "AnalyzerType",
    "CalibrationStatus",
    "CalibrationDriftSeverity",
    "WorkOrderPriority",
    "WorkOrderType",
    "create_calibration_monitor",
    "create_analyzer_spec",
]
