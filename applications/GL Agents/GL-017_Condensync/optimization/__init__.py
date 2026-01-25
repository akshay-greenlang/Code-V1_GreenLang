# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC Optimization Module

Condenser vacuum system optimization components including:
- Vacuum Optimizer: SLSQP-based optimization for CW flow, pump staging, fan staging
- Cleaning Scheduler: Predictive cleaning schedule optimization with ROI analysis
- Constraint Manager: Safety limits, equipment constraints, data quality gating
- What-If Analyzer: Scenario analysis and sensitivity studies

Zero-Hallucination Guarantee:
    All optimization modules use deterministic algorithms and physics-based models.
    No AI/ML inference in any calculation path.
    Complete audit trails with SHA-256 provenance hashing.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

# Vacuum Optimizer
from .vacuum_optimizer import (
    # Main class
    VacuumOptimizer,
    # Data classes
    PumpCurve,
    FanCurve,
    CondenserState,
    EquipmentInventory,
    OperatingLimits,
    CostParameters,
    OptimizerConfig,
    OptimizationRecommendation,
    OptimizationResult,
    # Enums
    OptimizationStatus,
    PumpStatus,
    FanStatus,
    OperatingMode,
    # Factory
    create_default_optimizer,
)

# Cleaning Scheduler
from .cleaning_scheduler import (
    # Main classes
    CleaningScheduler,
    CFDegradationModel,
    EnergyLossCalculator,
    # Data classes
    CleaningCosts,
    DowntimeCosts,
    CleaningParameters,
    DegradationParameters,
    EnergyParameters,
    SchedulerConfig,
    CFPrediction,
    EnergyLossEstimate,
    CleaningROI,
    CleaningRecommendation,
    # Enums
    DegradationModel,
    CleaningMethod,
    RecommendationUrgency,
    # Factory
    create_default_scheduler,
)

# Constraint Manager
from .constraint_manager import (
    # Main classes
    ConstraintManager,
    ConstraintLibrary,
    DataQualityChecker,
    # Data classes
    ConstraintDefinition,
    ConstraintEvaluation,
    DataQualityCheck,
    DataQualityGate,
    ConstraintManagerConfig,
    ConstraintValidationResult,
    # Enums
    ConstraintType,
    ConstraintSeverity,
    ConstraintStatus,
    DataQualityFlag,
    # Factory
    create_default_constraint_manager,
)

# What-If Analyzer
from .what_if_analyzer import (
    # Main class
    WhatIfAnalyzer,
    # Supporting classes
    CondenserPhysicsModel,
    ScenarioLibrary,
    # Data classes
    BaselineState,
    ScenarioDefinition,
    ScenarioResult,
    SensitivityResult,
    CounterfactualRecommendation,
    WhatIfAnalyzerConfig,
    WhatIfAnalysisResult,
    # Enums
    ScenarioType,
    ParameterType,
    SensitivityDirection,
    # Factory
    create_default_analyzer,
    create_baseline_from_dict,
)


__all__ = [
    # =========================================================================
    # Vacuum Optimizer
    # =========================================================================
    "VacuumOptimizer",
    "PumpCurve",
    "FanCurve",
    "CondenserState",
    "EquipmentInventory",
    "OperatingLimits",
    "CostParameters",
    "OptimizerConfig",
    "OptimizationRecommendation",
    "OptimizationResult",
    "OptimizationStatus",
    "PumpStatus",
    "FanStatus",
    "OperatingMode",
    "create_default_optimizer",

    # =========================================================================
    # Cleaning Scheduler
    # =========================================================================
    "CleaningScheduler",
    "CFDegradationModel",
    "EnergyLossCalculator",
    "CleaningCosts",
    "DowntimeCosts",
    "CleaningParameters",
    "DegradationParameters",
    "EnergyParameters",
    "SchedulerConfig",
    "CFPrediction",
    "EnergyLossEstimate",
    "CleaningROI",
    "CleaningRecommendation",
    "DegradationModel",
    "CleaningMethod",
    "RecommendationUrgency",
    "create_default_scheduler",

    # =========================================================================
    # Constraint Manager
    # =========================================================================
    "ConstraintManager",
    "ConstraintLibrary",
    "DataQualityChecker",
    "ConstraintDefinition",
    "ConstraintEvaluation",
    "DataQualityCheck",
    "DataQualityGate",
    "ConstraintManagerConfig",
    "ConstraintValidationResult",
    "ConstraintType",
    "ConstraintSeverity",
    "ConstraintStatus",
    "DataQualityFlag",
    "create_default_constraint_manager",

    # =========================================================================
    # What-If Analyzer
    # =========================================================================
    "WhatIfAnalyzer",
    "CondenserPhysicsModel",
    "ScenarioLibrary",
    "BaselineState",
    "ScenarioDefinition",
    "ScenarioResult",
    "SensitivityResult",
    "CounterfactualRecommendation",
    "WhatIfAnalyzerConfig",
    "WhatIfAnalysisResult",
    "ScenarioType",
    "ParameterType",
    "SensitivityDirection",
    "create_default_analyzer",
    "create_baseline_from_dict",
]


# Module version
__version__ = "1.0.0"
