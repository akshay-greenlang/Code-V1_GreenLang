"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Core Module

This module provides the core functionality for the UNIFIEDSTEAM
SteamSystemOptimizer agent, including configuration, schemas, handlers,
and orchestration for comprehensive steam system optimization.

The UNIFIEDSTEAM agent implements:
    - IAPWS-IF97 compliant thermodynamic calculations
    - Steam trap diagnostics and failure prediction
    - Desuperheater optimization
    - Condensate recovery optimization
    - Enthalpy balance calculations
    - Causal Root Cause Analysis (RCA)
    - Uncertainty quantification with confidence bounds
    - Full explainability (SHAP/LIME integration)

Standards Compliance:
    - IAPWS-IF97 (International Association for Properties of Water and Steam)
    - ASME PTC 19.11 (Steam and Water in Industrial Systems)
    - ISO 50001 (Energy Management Systems)
    - GHG Protocol (Scope 1 emissions reporting)

Business Value: $14B annual savings potential
Target: Q1 2026

Example:
    >>> from gl_003_unifiedsteam.core import SteamSystemConfig, SteamSystemOrchestrator
    >>> config = SteamSystemConfig(system_id="STEAM-001")
    >>> orchestrator = SteamSystemOrchestrator(config)
    >>> await orchestrator.start()
    >>> result = await orchestrator.optimize_desuperheater(process_data)
"""

from .config import (
    # Enums
    OperatingState,
    SteamQuality,
    OptimizationType,
    DeploymentMode,
    SafetyIntegrityLevel,
    SensorType,
    TrapFailureMode,
    MaintenancePriority,
    ConfidenceLevel,
    # Sensor Configurations
    PressureSensorConfig,
    TemperatureSensorConfig,
    FlowSensorConfig,
    QualitySensorConfig,
    AcousticSensorConfig,
    # Safety and Thresholds
    SafetyLimitsConfig,
    ThresholdConfig,
    # IAPWS Configuration
    IAPWSIF97Config,
    # Sub-configurations
    SensorArrayConfig,
    OptimizationConfig,
    IntegrationConfig,
    MetricsConfig,
    ExplainabilityConfig,
    UncertaintyConfig,
    # Main Configuration
    SteamSystemConfig,
)

from .schemas import (
    # Process Data
    SteamProcessData,
    WaterChemistry,
    TrapAcousticsData,
    CondenserData,
    # Thermodynamic Results
    SteamProperties,
    EnthalpyBalanceResult,
    HeatLossBreakdown,
    # Optimization Results
    DesuperheaterRecommendation,
    SprayWaterSetpoint,
    CondensateRecoveryResult,
    FlashLossAnalysis,
    TrapDiagnosticsResult,
    TrapHealthAssessment,
    # Analysis Results
    CausalAnalysisResult,
    CausalFactor,
    Counterfactual,
    InterventionRecommendation,
    # Explainability
    ExplainabilityPayload,
    PhysicsTrace,
    ModelTrace,
    FeatureContribution,
    # Uncertainty
    UncertaintyBounds,
    ConfidenceInterval,
    # Combined Results
    OptimizationResult,
    SystemOptimizationSummary,
    # Status
    SteamSystemStatus,
    AgentStatus,
    # Events
    SteamSystemEvent,
    OptimizationEvent,
    AlarmEvent,
)

from .handlers import (
    # Base Handler
    EventHandler,
    # Specific Handlers
    SteamSafetyEventHandler,
    TrapDiagnosticsEventHandler,
    OptimizationEventHandler,
    ThermodynamicsEventHandler,
    CondensateEventHandler,
    AuditEventHandler,
    MetricsEventHandler,
)

from .orchestrator import (
    SteamSystemOrchestrator,
)

__all__ = [
    # Config Enums
    "OperatingState",
    "SteamQuality",
    "OptimizationType",
    "DeploymentMode",
    "SafetyIntegrityLevel",
    "SensorType",
    "TrapFailureMode",
    "MaintenancePriority",
    "ConfidenceLevel",
    # Config Classes
    "PressureSensorConfig",
    "TemperatureSensorConfig",
    "FlowSensorConfig",
    "QualitySensorConfig",
    "AcousticSensorConfig",
    "SafetyLimitsConfig",
    "ThresholdConfig",
    "IAPWSIF97Config",
    "SensorArrayConfig",
    "OptimizationConfig",
    "IntegrationConfig",
    "MetricsConfig",
    "ExplainabilityConfig",
    "UncertaintyConfig",
    "SteamSystemConfig",
    # Schema Classes
    "SteamProcessData",
    "WaterChemistry",
    "TrapAcousticsData",
    "CondenserData",
    "SteamProperties",
    "EnthalpyBalanceResult",
    "HeatLossBreakdown",
    "DesuperheaterRecommendation",
    "SprayWaterSetpoint",
    "CondensateRecoveryResult",
    "FlashLossAnalysis",
    "TrapDiagnosticsResult",
    "TrapHealthAssessment",
    "CausalAnalysisResult",
    "CausalFactor",
    "Counterfactual",
    "InterventionRecommendation",
    "ExplainabilityPayload",
    "PhysicsTrace",
    "ModelTrace",
    "FeatureContribution",
    "UncertaintyBounds",
    "ConfidenceInterval",
    "OptimizationResult",
    "SystemOptimizationSummary",
    "SteamSystemStatus",
    "AgentStatus",
    "SteamSystemEvent",
    "OptimizationEvent",
    "AlarmEvent",
    # Handlers
    "EventHandler",
    "SteamSafetyEventHandler",
    "TrapDiagnosticsEventHandler",
    "OptimizationEventHandler",
    "ThermodynamicsEventHandler",
    "CondensateEventHandler",
    "AuditEventHandler",
    "MetricsEventHandler",
    # Orchestrator
    "SteamSystemOrchestrator",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-003"
__codename__ = "UNIFIEDSTEAM"
