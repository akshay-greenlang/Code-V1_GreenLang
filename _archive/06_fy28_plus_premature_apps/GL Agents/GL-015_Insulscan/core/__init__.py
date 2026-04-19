# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Core Module

Core components for the Insulation Scanning & Thermal Assessment agent including
configuration, data models, orchestration, and seed management.

This module provides:
- InsulscanOrchestrator: Main orchestration class for thermal analysis,
  hot spot detection, heat loss calculations, and repair recommendations
- Configuration management via pydantic-settings
- Pydantic v2 data models for insulation assessment operations
- Deterministic seed management for reproducible ML predictions

All components follow GreenLang zero-hallucination principles with
SHA-256 provenance tracking and strict type safety.

Example:
    >>> from core import InsulscanOrchestrator, InsulscanSettings
    >>> settings = InsulscanSettings()
    >>> orchestrator = InsulscanOrchestrator(settings)
    >>> result = await orchestrator.analyze_insulation(asset, measurements)

Version: 1.0.0
Author: GreenLang GL-015 INSULSCAN
"""

from .config import (
    # Enums
    InsulationType,
    SurfaceType,
    HotSpotSeverity,
    ConditionSeverity,
    RepairPriority,
    RepairType,
    ThermalImagingMode,
    DataQuality,
    # Thermal properties
    DEFAULT_THERMAL_CONDUCTIVITY,
    DEFAULT_EMISSIVITY,
    # Settings classes
    ThermalAnalysisSettings,
    InsulationAssessmentSettings,
    EconomicSettings,
    MLServiceSettings,
    ThermalImagingSettings,
    APISettings,
    LoggingSettings,
    ProvenanceSettings,
    FeatureFlags,
    InsulscanSettings,
    # Functions
    get_settings,
    reload_settings,
)
from .schemas import (
    # Enums
    AnalysisStatus,
    MeasurementType,
    DamageType,
    # Location schemas
    AssetLocation,
    # Asset schemas
    InsulationAsset,
    # Measurement schemas
    ThermalMeasurement,
    # Hot spot schemas
    HotSpotDetection,
    # Condition schemas
    InsulationCondition,
    # Heat loss schemas
    HeatLossResult,
    # Recommendation schemas
    RepairRecommendation,
    # Result schemas
    AnalysisResult,
)
from .orchestrator import (
    InsulscanOrchestrator,
    CalculationEvent,
    OrchestratorStatus,
    run_analysis_sync,
)
from .seed_manager import (
    SeedManager,
    SeedRecord,
    ReproducibilityContext,
    SeedDomain,
    get_seed_manager,
    set_global_seed,
    reset_seeds,
    get_reproducibility_context,
    seed_context,
    deterministic,
    NUMPY_AVAILABLE,
)

__version__ = "1.0.0"
__agent_id__ = "GL-015"
__agent_name__ = "INSULSCAN"

__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # Configuration - Enums
    "InsulationType",
    "SurfaceType",
    "HotSpotSeverity",
    "ConditionSeverity",
    "RepairPriority",
    "RepairType",
    "ThermalImagingMode",
    "DataQuality",
    # Configuration - Thermal properties
    "DEFAULT_THERMAL_CONDUCTIVITY",
    "DEFAULT_EMISSIVITY",
    # Configuration - Settings classes
    "ThermalAnalysisSettings",
    "InsulationAssessmentSettings",
    "EconomicSettings",
    "MLServiceSettings",
    "ThermalImagingSettings",
    "APISettings",
    "LoggingSettings",
    "ProvenanceSettings",
    "FeatureFlags",
    "InsulscanSettings",
    # Configuration - Functions
    "get_settings",
    "reload_settings",
    # Schemas - Enums
    "AnalysisStatus",
    "MeasurementType",
    "DamageType",
    # Schemas - Location
    "AssetLocation",
    # Schemas - Asset
    "InsulationAsset",
    # Schemas - Measurement
    "ThermalMeasurement",
    # Schemas - Hot spot
    "HotSpotDetection",
    # Schemas - Condition
    "InsulationCondition",
    # Schemas - Heat loss
    "HeatLossResult",
    # Schemas - Recommendation
    "RepairRecommendation",
    # Schemas - Result
    "AnalysisResult",
    # Orchestrator
    "InsulscanOrchestrator",
    "CalculationEvent",
    "OrchestratorStatus",
    "run_analysis_sync",
    # Seed Management
    "SeedManager",
    "SeedRecord",
    "ReproducibilityContext",
    "SeedDomain",
    "get_seed_manager",
    "set_global_seed",
    "reset_seeds",
    "get_reproducibility_context",
    "seed_context",
    "deterministic",
    "NUMPY_AVAILABLE",
]
