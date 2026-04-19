# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Core Module

Core components for the Heat Exchanger Optimizer agent including
configuration, data models, orchestration, and seed management.

This module provides:
- ExchangerProOrchestrator: Main orchestration class for thermal calculations,
  fouling predictions, and cleaning optimization
- Configuration management via pydantic-settings
- Pydantic v2 data models for heat exchanger operations
- Deterministic seed management for reproducible ML predictions

All components follow GreenLang zero-hallucination principles with
SHA-256 provenance tracking and strict type safety.

Example:
    >>> from core import ExchangerProOrchestrator, ExchangerProSettings
    >>> settings = ExchangerProSettings()
    >>> orchestrator = ExchangerProOrchestrator(settings)
    >>> result = await orchestrator.analyze_exchanger(operating_state)

Version: 1.0.0
Author: GreenLang GL-014 EXCHANGERPRO
"""

from .config import (
    ExchangerProSettings,
    ThermalEngineSettings,
    MLServiceSettings,
    OptimizerSettings,
    APISettings,
    LoggingSettings,
    ProvenanceSettings,
    FeatureFlags,
    TEMAType,
    FlowArrangement,
    ShellType,
    TubePattern,
    MaterialGrade,
)
from .schemas import (
    ExchangerConfig,
    TubeGeometry,
    ShellGeometry,
    BaffleConfig,
    MaterialProperties,
    OperatingState,
    TemperatureProfile,
    FlowProfile,
    PressureProfile,
    ThermalKPIs,
    HeatBalance,
    EffectivenessMetrics,
    FoulingState,
    FoulingTrend,
    CleaningRecommendation,
    CleaningSchedule,
    AnalysisResult,
    OptimizationResult,
)
from .orchestrator import ExchangerProOrchestrator
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
__agent_id__ = "GL-014"
__agent_name__ = "EXCHANGERPRO"

__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # Configuration
    "ExchangerProSettings",
    "ThermalEngineSettings",
    "MLServiceSettings",
    "OptimizerSettings",
    "APISettings",
    "LoggingSettings",
    "ProvenanceSettings",
    "FeatureFlags",
    # Config Enums
    "TEMAType",
    "FlowArrangement",
    "ShellType",
    "TubePattern",
    "MaterialGrade",
    # Schemas - Configuration
    "ExchangerConfig",
    "TubeGeometry",
    "ShellGeometry",
    "BaffleConfig",
    "MaterialProperties",
    # Schemas - Operating State
    "OperatingState",
    "TemperatureProfile",
    "FlowProfile",
    "PressureProfile",
    # Schemas - KPIs
    "ThermalKPIs",
    "HeatBalance",
    "EffectivenessMetrics",
    # Schemas - Fouling
    "FoulingState",
    "FoulingTrend",
    "CleaningRecommendation",
    "CleaningSchedule",
    # Schemas - Results
    "AnalysisResult",
    "OptimizationResult",
    # Orchestrator
    "ExchangerProOrchestrator",
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
