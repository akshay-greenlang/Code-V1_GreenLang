# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Integration Connectors Package.

This package provides enterprise-grade integration connectors for steam quality
monitoring and control. All connectors implement common patterns for connection
pooling, retry logic, circuit breaker protection, health monitoring, caching,
and audit logging.

Connectors:
    - SteamQualityMeterConnector: Steam quality measurement device integration
    - ControlValveConnector: Valve actuator communication
    - DesuperheaterConnector: Desuperheater system control
    - SCADAConnector: SCADA system integration

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from .base_connector import (
    # Base classes
    BaseConnector,
    BaseConnectorConfig,

    # Connection pooling
    ConnectionPool,

    # Circuit breaker
    CircuitBreaker,
    CircuitState,

    # Cache
    LRUCache,
    CacheEntry,

    # Retry
    with_retry,

    # Metrics
    MetricsCollector,
    MetricsSnapshot,

    # Audit
    AuditLogger,
    AuditLogEntry,

    # Enums
    ConnectionState,
    ConnectorType,
    ProtocolType,
    HealthStatus,

    # Models
    ConnectionInfo,
    HealthCheckResult,

    # Exceptions
    ConnectorError,
    ConnectionError,
    AuthenticationError,
    TimeoutError,
    ValidationError,
    CircuitOpenError,
    RetryExhaustedError,
    ConfigurationError,
    DataQualityError,
    SafetyInterlockError,
    CalibrationError,
    CommunicationError,

    # Context manager
    ConnectorContextManager,
)

from .steam_quality_meter_connector import (
    # Main connector
    SteamQualityMeterConnector,
    SteamQualityMeterConfig,

    # Data models
    SteamQualityReading,
    PressureReading,
    TemperatureReading,
    FlowReading,
    MeterStatusReading,
    CalibrationData,
    CalibrationResult,

    # Enums
    MeterType,
    MeterVendor,
    MeterStatus,
    CalibrationStatus,
    QualityFlag,

    # Supporting classes
    ModbusRegisterMap,
    OPCUANodeMap,

    # Factory
    create_steam_quality_meter_connector,
)

from .control_valve_connector import (
    # Main connector
    ControlValveConnector,
    ControlValveConfig,

    # Data models
    ValveStatusReading,
    ValveDiagnostics,
    SafetyInterlockState,

    # Enums
    ValveType,
    ActuatorType,
    ValveStatus,
    ActuatorStatus,
    FailSafeAction,
    InterlockType,

    # Factory
    create_control_valve_connector,
)

from .desuperheater_connector import (
    # Main connector
    DesuperheaterConnector,
    DesuperheaterConfig,

    # Data models
    DesuperheaterStatusReading,
    InjectionRateResult,
    DesuperheaterDiagnostics,

    # Enums
    DesuperheaterType,
    SprayValveType,
    DesuperheaterStatus,
    WaterSupplyStatus,

    # Factory
    create_desuperheater_connector,
)

from .scada_connector import (
    # Main connector
    SCADAConnector,
    SCADAConnectorConfig,

    # Data models
    TagValue,
    TagConfig,
    Subscription,
    AlarmData,
    AlarmSendResult,
    HistoricalDataRequest,

    # Enums
    SCADAVendor,
    TagDataType,
    TagQuality,
    AlarmPriority,
    AlarmState,
    SubscriptionMode,

    # Factory
    create_scada_connector,
)


__all__ = [
    # ==========================================================================
    # Base Connector
    # ==========================================================================
    "BaseConnector",
    "BaseConnectorConfig",
    "ConnectionPool",
    "CircuitBreaker",
    "CircuitState",
    "LRUCache",
    "CacheEntry",
    "with_retry",
    "MetricsCollector",
    "MetricsSnapshot",
    "AuditLogger",
    "AuditLogEntry",
    "ConnectionState",
    "ConnectorType",
    "ProtocolType",
    "HealthStatus",
    "ConnectionInfo",
    "HealthCheckResult",
    "ConnectorError",
    "ConnectionError",
    "AuthenticationError",
    "TimeoutError",
    "ValidationError",
    "CircuitOpenError",
    "RetryExhaustedError",
    "ConfigurationError",
    "DataQualityError",
    "SafetyInterlockError",
    "CalibrationError",
    "CommunicationError",
    "ConnectorContextManager",

    # ==========================================================================
    # Steam Quality Meter Connector
    # ==========================================================================
    "SteamQualityMeterConnector",
    "SteamQualityMeterConfig",
    "SteamQualityReading",
    "PressureReading",
    "TemperatureReading",
    "FlowReading",
    "MeterStatusReading",
    "CalibrationData",
    "CalibrationResult",
    "MeterType",
    "MeterVendor",
    "MeterStatus",
    "CalibrationStatus",
    "QualityFlag",
    "ModbusRegisterMap",
    "OPCUANodeMap",
    "create_steam_quality_meter_connector",

    # ==========================================================================
    # Control Valve Connector
    # ==========================================================================
    "ControlValveConnector",
    "ControlValveConfig",
    "ValveStatusReading",
    "ValveDiagnostics",
    "SafetyInterlockState",
    "ValveType",
    "ActuatorType",
    "ValveStatus",
    "ActuatorStatus",
    "FailSafeAction",
    "InterlockType",
    "create_control_valve_connector",

    # ==========================================================================
    # Desuperheater Connector
    # ==========================================================================
    "DesuperheaterConnector",
    "DesuperheaterConfig",
    "DesuperheaterStatusReading",
    "InjectionRateResult",
    "DesuperheaterDiagnostics",
    "DesuperheaterType",
    "SprayValveType",
    "DesuperheaterStatus",
    "WaterSupplyStatus",
    "create_desuperheater_connector",

    # ==========================================================================
    # SCADA Connector
    # ==========================================================================
    "SCADAConnector",
    "SCADAConnectorConfig",
    "TagValue",
    "TagConfig",
    "Subscription",
    "AlarmData",
    "AlarmSendResult",
    "HistoricalDataRequest",
    "SCADAVendor",
    "TagDataType",
    "TagQuality",
    "AlarmPriority",
    "AlarmState",
    "SubscriptionMode",
    "create_scada_connector",
]

__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"
