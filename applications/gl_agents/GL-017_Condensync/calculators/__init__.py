# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC Calculators Package

This package provides deterministic calculators for steam condenser performance
monitoring, optimization, and predictive maintenance following GreenLang's
zero-hallucination principles.

Modules:
    hei_condenser_calculator: HEI Standards compliant condenser performance
    vacuum_optimization_calculator: Vacuum/backpressure optimization
    fouling_prediction_calculator: Fouling prediction and cleaning schedule
    subcooling_calculator: Condensate subcooling analysis

Standards Compliance:
    - HEI-2629: Standards for Steam Surface Condensers (12th Edition)
    - ASME PTC 12.2: Steam Surface Condensers Performance Test Code
    - ASME PTC 6: Steam Turbines Performance Test Code
    - IAPWS-IF97: Industrial Formulation for Water and Steam Properties
    - ANSI/HI 9.6.1: Rotodynamic Pumps - NPSH Margin
    - EPRI Guidelines for Condenser Performance Assessment

Zero-Hallucination Guarantee:
    All calculations use deterministic engineering formulas from published standards.
    No LLM or AI inference in any calculation path.
    Same inputs always produce identical outputs with bit-perfect reproducibility.
    Complete provenance tracking with SHA-256 hashes for audit trails.

Example:
    >>> from calculators import HEICondenserCalculator, TubeMaterial
    >>> from decimal import Decimal
    >>>
    >>> calculator = HEICondenserCalculator()
    >>> result = calculator.calculate_performance(
    ...     condenser_id="COND-001",
    ...     steam_flow_kg_s=Decimal("150.0"),
    ...     cw_inlet_temp_c=Decimal("20.0"),
    ...     cw_outlet_temp_c=Decimal("30.0"),
    ...     cw_flow_m3_s=Decimal("15.0"),
    ...     backpressure_kpa=Decimal("5.0"),
    ...     tube_material=TubeMaterial.TITANIUM
    ... )
    >>> print(f"Cleanliness Factor: {result.cleanliness.cleanliness_factor}")

Author: GL-CalculatorEngineer
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

# HEI Condenser Performance Calculator
from .hei_condenser_calculator import (
    # Main calculator
    HEICondenserCalculator,
    # Configuration
    HEICondenserConfig,
    # Enums
    TubeMaterial,
    CondenserType,
    PerformanceStatus,
    FoulingType,
    AlertSeverity,
    # Data classes
    CondenserSpecifications,
    CoolingWaterConditions,
    SteamConditions,
    HeatTransferResult,
    CleanlinessFactorResult,
    HEICorrectionFactors,
    PerformanceAlert,
    CondenserPerformanceResult,
    # Reference data
    IAPWS_SATURATION_TABLE,
    HEI_MATERIAL_FACTORS,
    TUBE_THERMAL_CONDUCTIVITY,
    HEI_VELOCITY_FACTORS,
    WATER_PROPERTIES_TABLE,
    # Utility functions
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    kpa_to_inhg,
    inhg_to_kpa,
    mw_to_mmbtu_hr,
    m3_s_to_gpm,
)

# Vacuum Optimization Calculator
from .vacuum_optimization_calculator import (
    # Main calculator
    VacuumOptimizationCalculator,
    # Configuration
    VacuumOptimizationConfig,
    # Enums
    OptimizationMode,
    ConstraintType,
    PenaltyType,
    AirLeakageSeverity,
    RecommendationPriority as VacuumRecommendationPriority,
    # Data classes
    TurbineCharacteristics,
    OperatingConditions,
    BackpressurePenalty,
    AirInLeakageImpact,
    CWFlowOptimization,
    OptimalVacuumSetpoint,
    EconomicAnalysis,
    OptimizationRecommendation,
    VacuumOptimizationResult,
    # Reference data
    BACKPRESSURE_CORRECTION_TABLE,
    HEAT_RATE_CORRECTION_FACTORS,
    AIR_LEAKAGE_IMPACT_TABLE,
    VACUUM_SATURATION_TABLE,
)

# Fouling Prediction Calculator
from .fouling_prediction_calculator import (
    # Main calculator
    FoulingPredictionCalculator,
    # Configuration
    FoulingPredictionConfig,
    # Enums
    FoulingType as PredictionFoulingType,
    TrendDirection,
    CleaningUrgency,
    MaintenanceStrategy,
    SeasonalPattern,
    # Data classes
    CFReading,
    DegradationAnalysis,
    TimeToThreshold,
    CleaningROI,
    OptimalCleaningWindow,
    WeibullReliability,
    SeasonalAnalysis,
    FoulingPredictionResult,
    # Reference data
    SEASONAL_FOULING_FACTORS,
    WATER_TYPE_FACTORS,
)

# Subcooling Calculator
from .subcooling_calculator import (
    # Main calculator
    SubcoolingCalculator,
    # Configuration
    SubcoolingConfig,
    # Enums
    SubcoolingStatus,
    AirBindingStatus,
    NPSHStatus,
    DissolvedOxygenRisk,
    RecommendationPriority as SubcoolingRecommendationPriority,
    # Data classes
    HotwellConditions,
    PumpCharacteristics,
    SubcoolingAnalysisResult,
    AirBindingAnalysis,
    NPSHAnalysis,
    DissolvedOxygenAnalysis,
    SubcoolingRecommendation,
    SubcoolingResult,
    # Reference data
    SATURATION_TABLE,
    DO_SOLUBILITY_TABLE,
)

# Shared Provenance Classes (from HEI calculator)
from .hei_condenser_calculator import (
    ProvenanceTracker,
    ProvenanceStep,
)

__version__ = "1.0.0"
__author__ = "GL-CalculatorEngineer"

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # =========================================================================
    # HEI CONDENSER CALCULATOR
    # =========================================================================
    "HEICondenserCalculator",
    "HEICondenserConfig",
    # Enums
    "TubeMaterial",
    "CondenserType",
    "PerformanceStatus",
    "FoulingType",
    "AlertSeverity",
    # Data classes
    "CondenserSpecifications",
    "CoolingWaterConditions",
    "SteamConditions",
    "HeatTransferResult",
    "CleanlinessFactorResult",
    "HEICorrectionFactors",
    "PerformanceAlert",
    "CondenserPerformanceResult",
    # Reference data
    "IAPWS_SATURATION_TABLE",
    "HEI_MATERIAL_FACTORS",
    "TUBE_THERMAL_CONDUCTIVITY",
    "HEI_VELOCITY_FACTORS",
    "WATER_PROPERTIES_TABLE",
    # Utility functions
    "celsius_to_fahrenheit",
    "fahrenheit_to_celsius",
    "kpa_to_inhg",
    "inhg_to_kpa",
    "mw_to_mmbtu_hr",
    "m3_s_to_gpm",

    # =========================================================================
    # VACUUM OPTIMIZATION CALCULATOR
    # =========================================================================
    "VacuumOptimizationCalculator",
    "VacuumOptimizationConfig",
    # Enums
    "OptimizationMode",
    "ConstraintType",
    "PenaltyType",
    "AirLeakageSeverity",
    "VacuumRecommendationPriority",
    # Data classes
    "TurbineCharacteristics",
    "OperatingConditions",
    "BackpressurePenalty",
    "AirInLeakageImpact",
    "CWFlowOptimization",
    "OptimalVacuumSetpoint",
    "EconomicAnalysis",
    "OptimizationRecommendation",
    "VacuumOptimizationResult",
    # Reference data
    "BACKPRESSURE_CORRECTION_TABLE",
    "HEAT_RATE_CORRECTION_FACTORS",
    "AIR_LEAKAGE_IMPACT_TABLE",
    "VACUUM_SATURATION_TABLE",

    # =========================================================================
    # FOULING PREDICTION CALCULATOR
    # =========================================================================
    "FoulingPredictionCalculator",
    "FoulingPredictionConfig",
    # Enums
    "PredictionFoulingType",
    "TrendDirection",
    "CleaningUrgency",
    "MaintenanceStrategy",
    "SeasonalPattern",
    # Data classes
    "CFReading",
    "DegradationAnalysis",
    "TimeToThreshold",
    "CleaningROI",
    "OptimalCleaningWindow",
    "WeibullReliability",
    "SeasonalAnalysis",
    "FoulingPredictionResult",
    # Reference data
    "SEASONAL_FOULING_FACTORS",
    "WATER_TYPE_FACTORS",

    # =========================================================================
    # SUBCOOLING CALCULATOR
    # =========================================================================
    "SubcoolingCalculator",
    "SubcoolingConfig",
    # Enums
    "SubcoolingStatus",
    "AirBindingStatus",
    "NPSHStatus",
    "DissolvedOxygenRisk",
    "SubcoolingRecommendationPriority",
    # Data classes
    "HotwellConditions",
    "PumpCharacteristics",
    "SubcoolingAnalysisResult",
    "AirBindingAnalysis",
    "NPSHAnalysis",
    "DissolvedOxygenAnalysis",
    "SubcoolingRecommendation",
    "SubcoolingResult",
    # Reference data
    "SATURATION_TABLE",
    "DO_SOLUBILITY_TABLE",

    # =========================================================================
    # SHARED PROVENANCE
    # =========================================================================
    "ProvenanceTracker",
    "ProvenanceStep",
]
