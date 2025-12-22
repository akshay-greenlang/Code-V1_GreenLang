"""
GL-004 BURNMASTER Core Module

Core components including configuration, schemas, and orchestration.

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

from .config import (
    BurnmasterConfig,
    OperatingMode,
    FuelType,
    BurnerType,
    SafetyLimits,
    OptimizationConfig,
    MLModelConfig,
    IntegrationConfig,
    MonitoringConfig,
    AuditConfig,
    RuntimeContext,
)

from .schemas import (
    SensorQuality,
    RecommendationPriority,
    RecommendationCategory,
    UncertainValue,
    Provenance,
    BurnerSensorData,
    ProcessState,
    Setpoint,
    SetpointRecommendation,
    OptimizationResult,
    CausalFactor,
    RootCauseAnalysis,
    HealthStatus,
    Alert,
    EmissionsReport,
    BurnerDiagnostics,
)

from .orchestrator import (
    BurnerOrchestrator,
    OrchestrationMetrics,
)

# Aliases for compatibility with main __init__.py
BurnerConfig = BurnmasterConfig
BurnerProcessData = ProcessState
CombustionProperties = BurnerSensorData
BurnerSystemOrchestrator = BurnerOrchestrator

__all__ = [
    # Config
    "BurnmasterConfig",
    "BurnerConfig",
    "OperatingMode",
    "FuelType",
    "BurnerType",
    "SafetyLimits",
    "OptimizationConfig",
    "MLModelConfig",
    "IntegrationConfig",
    "MonitoringConfig",
    "AuditConfig",
    "RuntimeContext",
    # Schemas
    "SensorQuality",
    "RecommendationPriority",
    "RecommendationCategory",
    "UncertainValue",
    "Provenance",
    "BurnerSensorData",
    "BurnerProcessData",
    "ProcessState",
    "CombustionProperties",
    "Setpoint",
    "SetpointRecommendation",
    "OptimizationResult",
    "CausalFactor",
    "RootCauseAnalysis",
    "HealthStatus",
    "Alert",
    "EmissionsReport",
    "BurnerDiagnostics",
    # Orchestrator
    "BurnerOrchestrator",
    "BurnerSystemOrchestrator",
    "OrchestrationMetrics",
]
