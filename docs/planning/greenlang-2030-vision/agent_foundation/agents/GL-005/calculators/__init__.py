# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent Calculator Modules

Comprehensive calculation engines for real-time combustion control with
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

Zero-Hallucination Guarantee:
    ✓ Deterministic: Same inputs → Same outputs (bit-perfect)
    ✓ Reproducible: Complete provenance tracking
    ✓ Auditable: SHA-256 hashes for all calculations
    ✓ No LLM: Pure mathematical calculations only
    ✓ Validated: Tested against known reference values

Reference Standards:
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces
    - ASME PTC 4.1: Fired Steam Generators Performance Test Codes
    - EPA 40 CFR Part 60: Standards of Performance
    - ISO 16001: Energy Management Systems
    - IEC 61508: Functional Safety Standards
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
]

__version__ = '1.0.0'
__author__ = 'GreenLang GL-005 Team'
__description__ = 'Zero-hallucination combustion control calculators'
