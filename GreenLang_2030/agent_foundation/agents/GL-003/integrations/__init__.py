"""
GL-003 SteamSystemAnalyzer Integration Modules

Comprehensive integration connectors for steam system monitoring and control.

Modules:
- base_connector: Abstract base class with retry logic, circuit breaker, health checks
- steam_meter_connector: Multi-protocol steam flow measurement (Modbus, HART, 4-20mA)
- pressure_sensor_connector: Multi-point pressure monitoring with drift detection
- temperature_sensor_connector: RTD and thermocouple support with smoothing
- scada_connector: OPC UA and Modbus integration for DCS/SCADA systems
- condensate_meter_connector: Condensate return monitoring and flash steam calculation
- agent_coordinator: Multi-agent communication and coordination
- data_transformers: Data normalization, quality scoring, and validation
"""

from .base_connector import (
    BaseConnector,
    ConnectionConfig,
    ConnectionState,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
    HealthStatus
)

from .steam_meter_connector import (
    SteamMeterConnector,
    SteamMeterConfig,
    MeterProtocol,
    FlowMeasurementType,
    FlowReading,
    TotalizerState
)

from .pressure_sensor_connector import (
    PressureSensorConnector,
    PressureSensorConfig,
    PressureType,
    SensorType as PressureSensorType,
    PressureReading
)

from .temperature_sensor_connector import (
    TemperatureSensorConnector,
    TemperatureSensorConfig,
    SensorType as TemperatureSensorType,
    TemperatureReading
)

from .scada_connector import (
    SCADAConnector,
    SCADAConnectionConfig,
    SCADAProtocol,
    SCADATag,
    DataQuality
)

from .condensate_meter_connector import (
    CondensateMeterConnector,
    CondensateMeterConfig,
    CondensateReading
)

from .agent_coordinator import (
    AgentCoordinator,
    MessageBus,
    TaskScheduler,
    StateManager,
    CollaborativeOptimizer,
    AgentMessage,
    AgentTask,
    AgentProfile,
    AgentCapability,
    MessageType,
    MessagePriority,
    AgentRole,
    TaskStatus
)

from .data_transformers import (
    DataTransformationPipeline,
    UnitConverter,
    DataValidator,
    OutlierDetector,
    DataImputer,
    TimeSeriesAligner,
    SensorFusion,
    DataPoint,
    SensorConfig,
    UnitSystem,
    DataQualityIssue
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"

# Define public API
__all__ = [
    # Base Connector
    "BaseConnector",
    "ConnectionConfig",
    "ConnectionState",
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    "HealthStatus",

    # Steam Meter
    "SteamMeterConnector",
    "SteamMeterConfig",
    "MeterProtocol",
    "FlowMeasurementType",
    "FlowReading",
    "TotalizerState",

    # Pressure Sensor
    "PressureSensorConnector",
    "PressureSensorConfig",
    "PressureType",
    "PressureSensorType",
    "PressureReading",

    # Temperature Sensor
    "TemperatureSensorConnector",
    "TemperatureSensorConfig",
    "TemperatureSensorType",
    "TemperatureReading",

    # SCADA
    "SCADAConnector",
    "SCADAConnectionConfig",
    "SCADAProtocol",
    "SCADATag",
    "DataQuality",

    # Condensate Meter
    "CondensateMeterConnector",
    "CondensateMeterConfig",
    "CondensateReading",

    # Agent Coordination
    "AgentCoordinator",
    "MessageBus",
    "TaskScheduler",
    "StateManager",
    "CollaborativeOptimizer",
    "AgentMessage",
    "AgentTask",
    "AgentProfile",
    "AgentCapability",
    "MessageType",
    "MessagePriority",
    "AgentRole",
    "TaskStatus",

    # Data Transformation
    "DataTransformationPipeline",
    "UnitConverter",
    "DataValidator",
    "OutlierDetector",
    "DataImputer",
    "TimeSeriesAligner",
    "SensorFusion",
    "DataPoint",
    "SensorConfig",
    "UnitSystem",
    "DataQualityIssue"
]
