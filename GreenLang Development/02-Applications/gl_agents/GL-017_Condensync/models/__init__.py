# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Models Package

Comprehensive Pydantic and dataclass models for condenser performance
optimization, fouling prediction, and maintenance recommendation.

This package provides:
- Input models: Sensor data, configuration, climate, constraints
- Output models: Performance metrics, optimization recommendations, explanations
- Domain models: Enumerations, physical properties, reference data

Standards Reference:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- EPRI Condenser Performance Guidelines

Zero-Hallucination Guarantee:
All models enforce strict validation at boundaries.
Deterministic types ensure reproducible calculations.
Complete provenance tracking for audit compliance.

Example Usage:
    >>> from models import CondenserDiagnosticInput, CondenserPerformanceOutput
    >>> from models import TubeMaterial, FailureMode, CleaningMethod
    >>>
    >>> # Create diagnostic input
    >>> input_data = CondenserDiagnosticInput(
    ...     condenser_id="COND-001",
    ...     cw_inlet_temp_c=Decimal("25.0"),
    ...     cw_outlet_temp_c=Decimal("35.0"),
    ...     cw_flow_rate_kg_s=Decimal("15000.0"),
    ...     condenser_pressure_kpa_abs=Decimal("5.0"),
    ...     steam_flow_rate_kg_s=Decimal("150.0"),
    ... )
    >>>
    >>> # Access tube material properties
    >>> tube = TubeMaterial.TITANIUM_GRADE_2
    >>> print(f"Thermal conductivity: {tube.thermal_conductivity_w_m_k} W/m-K")

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

# ============================================================================
# DOMAIN MODELS
# ============================================================================

from .domain import (
    # Material enumerations
    TubeMaterial,
    TubeSupport,
    TubeEndConnection,
    # Failure mode enumerations
    FailureMode,
    FailureSeverity,
    # Cleaning enumerations
    CleaningMethod,
    # Status enumerations
    AlertLevel,
    OperatingMode,
    WaterSource,
    VacuumControlMode,
    # Physical property dataclasses
    SteamProperties,
    CoolingWaterProperties,
    AirInLeakage,
    # Metric dataclasses
    CleanlinessFactorReading,
    VacuumReading,
    TemperatureDifferential,
    # CMMS integration
    CMMSWorkOrder,
    # Reference data
    HEIStandardConditions,
)

# ============================================================================
# INPUT MODELS
# ============================================================================

from .inputs import (
    # Primary input model
    CondenserDiagnosticInput,
    # Configuration model
    CondenserConfiguration,
    # Climate/environment model
    ClimateInput,
    # Constraint model
    OperatingConstraints,
    # Historical data model
    HistoricalDataInput,
    # Optimization request model
    OptimizationRequest,
)

# ============================================================================
# OUTPUT MODELS
# ============================================================================

from .outputs import (
    # Base output model
    BaseOutputModel,
    # Performance output
    CondenserPerformanceOutput,
    # Optimization output
    VacuumOptimizationOutput,
    # Fouling prediction output
    FoulingPredictionOutput,
    # Maintenance recommendation output
    MaintenanceRecommendation,
    # Explainability components
    FeatureContribution,
    Counterfactual,
    ExplainabilityOutput,
    # Comprehensive report
    CondenserDiagnosticReport,
)

# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "1.0.0"
__agent_id__ = "GL-017"
__agent_name__ = "CONDENSYNC"

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # Domain - Materials
    "TubeMaterial",
    "TubeSupport",
    "TubeEndConnection",
    # Domain - Failure
    "FailureMode",
    "FailureSeverity",
    # Domain - Cleaning
    "CleaningMethod",
    # Domain - Status
    "AlertLevel",
    "OperatingMode",
    "WaterSource",
    "VacuumControlMode",
    # Domain - Properties
    "SteamProperties",
    "CoolingWaterProperties",
    "AirInLeakage",
    # Domain - Metrics
    "CleanlinessFactorReading",
    "VacuumReading",
    "TemperatureDifferential",
    # Domain - CMMS
    "CMMSWorkOrder",
    # Domain - Reference
    "HEIStandardConditions",
    # Inputs
    "CondenserDiagnosticInput",
    "CondenserConfiguration",
    "ClimateInput",
    "OperatingConstraints",
    "HistoricalDataInput",
    "OptimizationRequest",
    # Outputs
    "BaseOutputModel",
    "CondenserPerformanceOutput",
    "VacuumOptimizationOutput",
    "FoulingPredictionOutput",
    "MaintenanceRecommendation",
    "FeatureContribution",
    "Counterfactual",
    "ExplainabilityOutput",
    "CondenserDiagnosticReport",
]


def get_model_info() -> dict:
    """
    Get information about the models package.

    Returns:
        Dictionary with package metadata.
    """
    return {
        "agent_id": __agent_id__,
        "agent_name": __agent_name__,
        "version": __version__,
        "input_models": [
            "CondenserDiagnosticInput",
            "CondenserConfiguration",
            "ClimateInput",
            "OperatingConstraints",
            "HistoricalDataInput",
            "OptimizationRequest",
        ],
        "output_models": [
            "CondenserPerformanceOutput",
            "VacuumOptimizationOutput",
            "FoulingPredictionOutput",
            "MaintenanceRecommendation",
            "ExplainabilityOutput",
            "CondenserDiagnosticReport",
        ],
        "domain_enums": [
            "TubeMaterial",
            "FailureMode",
            "CleaningMethod",
            "AlertLevel",
            "OperatingMode",
            "WaterSource",
        ],
        "standards": [
            "HEI Steam Surface Condensers (12th Ed)",
            "ASME PTC 12.2",
            "EPRI Condenser Guidelines",
        ],
    }
