"""
GL-002 FLAMEGUARD BoilerEfficiencyOptimizer - Core Module

This module exports all core components for the FLAMEGUARD agent.
"""

from .config import (
    # Enums
    BoilerType,
    FuelType,
    CombustionMode,
    SafetyIntegrityLevel,
    OperatingState,
    OptimizationObjective,
    EmissionStandard,
    ControlMode,
    # Configuration Classes
    FuelProperties,
    FuelConfig,
    O2TrimConfig,
    ExcessAirConfig,
    COMonitoringConfig,
    CombustionConfig,
    FlameDetectionConfig,
    PurgeConfig,
    SafetyInterlockConfig,
    SafetyConfig,
    EfficiencyCalculationConfig,
    HeatBalanceConfig,
    EmissionsConfig,
    AIOptimizationConfig,
    SetpointOptimizationConfig,
    OptimizationConfig,
    SCADAConfig,
    IntegrationConfig,
    BoilerSpecifications,
    MetricsConfig,
    AlertingConfig,
    APIConfig,
    FlameguardConfig,
)

from .schemas import (
    # Enums
    OptimizationStatus,
    CalculationType,
    SeverityLevel,
    AlarmState,
    TripType,
    # Process Data
    BoilerProcessData,
    CombustionAnalysis,
    EfficiencyCalculation,
    EmissionsCalculation,
    FuelBlendOptimization,
    # Optimization
    OptimizationRequest,
    OptimizationResult,
    SetpointRecommendation,
    # Safety
    SafetyStatus,
    SafetyEvent,
    # Status
    BoilerStatus,
    AgentStatus,
    # Events
    FlameguardEvent,
    CalculationEvent,
    # API
    APIResponse,
    HealthCheckResponse,
)

from .handlers import (
    EventHandler,
    BoilerSafetyEventHandler,
    CombustionEventHandler,
    OptimizationEventHandler,
    EfficiencyEventHandler,
    EmissionsEventHandler,
    AuditEventHandler,
    MetricsEventHandler,
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitOpenError,
    CircuitHalfOpenError,
    circuit_breaker,
)

__all__ = [
    # Config Enums
    "BoilerType",
    "FuelType",
    "CombustionMode",
    "SafetyIntegrityLevel",
    "OperatingState",
    "OptimizationObjective",
    "EmissionStandard",
    "ControlMode",
    # Configuration Classes
    "FuelProperties",
    "FuelConfig",
    "O2TrimConfig",
    "ExcessAirConfig",
    "COMonitoringConfig",
    "CombustionConfig",
    "FlameDetectionConfig",
    "PurgeConfig",
    "SafetyInterlockConfig",
    "SafetyConfig",
    "EfficiencyCalculationConfig",
    "HeatBalanceConfig",
    "EmissionsConfig",
    "AIOptimizationConfig",
    "SetpointOptimizationConfig",
    "OptimizationConfig",
    "SCADAConfig",
    "IntegrationConfig",
    "BoilerSpecifications",
    "MetricsConfig",
    "AlertingConfig",
    "APIConfig",
    "FlameguardConfig",
    # Schema Enums
    "OptimizationStatus",
    "CalculationType",
    "SeverityLevel",
    "AlarmState",
    "TripType",
    # Process Data Schemas
    "BoilerProcessData",
    "CombustionAnalysis",
    "EfficiencyCalculation",
    "EmissionsCalculation",
    "FuelBlendOptimization",
    # Optimization Schemas
    "OptimizationRequest",
    "OptimizationResult",
    "SetpointRecommendation",
    # Safety Schemas
    "SafetyStatus",
    "SafetyEvent",
    # Status Schemas
    "BoilerStatus",
    "AgentStatus",
    # Event Schemas
    "FlameguardEvent",
    "CalculationEvent",
    # API Schemas
    "APIResponse",
    "HealthCheckResponse",
    # Handlers
    "EventHandler",
    "BoilerSafetyEventHandler",
    "CombustionEventHandler",
    "OptimizationEventHandler",
    "EfficiencyEventHandler",
    "EmissionsEventHandler",
    "AuditEventHandler",
    "MetricsEventHandler",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerRegistry",
    "CircuitState",
    "CircuitOpenError",
    "CircuitHalfOpenError",
    "circuit_breaker",
]
