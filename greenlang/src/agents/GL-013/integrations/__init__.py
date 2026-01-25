"""
GL-013 PREDICTMAINT Integrations Package.

Enterprise-grade data integrations for the Predictive Maintenance Agent.
Provides connectors for CMMS, condition monitoring systems, IoT sensors,
multi-agent coordination, and data transformation utilities.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

# Base Connector
from .base_connector import (
    # Enumerations
    ConnectionState,
    CircuitState,
    HealthStatus,
    ConnectorType,
    DataQualityLevel,
    RateLimitStrategy,
    # Configuration
    BaseConnectorConfig,
    ConnectionInfo,
    HealthCheckResult,
    DataQualityResult,
    MetricsSnapshot,
    AuditLogEntry,
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
    RateLimitError,
    ProtocolError,
    # Components
    LRUCache,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    CircuitBreaker,
    ConnectionPool,
    MetricsCollector,
    AuditLogger,
    DataValidator,
    # Base Class
    BaseConnector,
    ConnectorContextManager,
    # Decorators
    with_retry,
    # Factory
    create_connector_config,
)

# CMMS Connector
from .cmms_connector import (
    # Enumerations
    CMSProvider,
    WorkOrderStatus,
    WorkOrderPriority,
    WorkOrderType,
    EquipmentStatus,
    EquipmentCriticality,
    MaintenanceType,
    AuthenticationType,
    # Configuration
    OAuth2Config,
    APIKeyConfig,
    BasicAuthConfig,
    CMSSConnectorConfig,
    # Data Models
    Equipment,
    WorkOrder,
    MaintenanceHistory,
    SparePart,
    Notification,
    CostRecord,
    # Request/Response Models
    WorkOrderCreateRequest,
    WorkOrderUpdateRequest,
    EquipmentQueryParams,
    MaintenanceHistoryQueryParams,
    # Connector
    CMSSConnector,
    # Factory
    create_cmms_connector,
)

# Condition Monitoring Connector
from .condition_monitoring_connector import (
    # Enumerations
    ConditionMonitoringProvider,
    CommunicationProtocol,
    VibrationUnit,
    MeasurementType,
    AlarmSeverity,
    AlarmState,
    MeasurementAxis,
    MachineState,
    TrendDirection,
    # Configuration
    OPCUAConfig,
    ModbusConfig,
    ConditionMonitoringConnectorConfig,
    # Data Models
    MeasurementPoint,
    VibrationReading,
    SpectrumData,
    WaveformData,
    Alarm,
    TrendData,
    RouteData,
    # Connector
    ConditionMonitoringConnector,
    # Factory
    create_condition_monitoring_connector,
)

# IoT Sensor Connector
from .iot_sensor_connector import (
    # Enumerations
    SensorType,
    SensorProtocol,
    MQTTQoS,
    SensorStatus,
    DataFormat,
    TimeSyncMode,
    # Configuration
    MQTTConfig,
    SensorGatewayConfig,
    IoTSensorConnectorConfig,
    # Data Models
    IoTSensor,
    SensorReading,
    SensorBatch,
    GatewayStatus,
    # Connector
    IoTSensorConnector,
    # Factory
    create_iot_sensor_connector,
)

# Agent Coordinator
from .agent_coordinator import (
    # Enumerations
    AgentID,
    MessageType,
    MessagePriority,
    TaskStatus,
    ConsensusState,
    RoutingStrategy,
    LoadBalanceStrategy,
    # Configuration
    MessageBusConfig,
    AgentCoordinatorConfig,
    # Data Models
    AgentMessage,
    AgentResponse,
    AgentStatus,
    DistributedTask,
    ConsensusProposal,
    # Coordinator
    AgentCoordinator,
    # Factory
    create_agent_coordinator,
)

# Data Transformers
from .data_transformers import (
    # Enumerations
    UnitCategory,
    TimeZoneHandling,
    MissingDataStrategy,
    OutlierMethod,
    DataQualityDimension,
    # Components
    UnitConverter,
    TimestampNormalizer,
    SchemaMapper,
    DataQualityScorer,
    MissingDataHandler,
    OutlierDetector,
    DataTransformer,
    # Data Classes
    UnitDefinition,
    FieldMapping,
    QualityMetric,
)


__all__ = [
    # Base Connector - Enumerations
    "ConnectionState",
    "CircuitState",
    "HealthStatus",
    "ConnectorType",
    "DataQualityLevel",
    "RateLimitStrategy",
    # Base Connector - Configuration
    "BaseConnectorConfig",
    "ConnectionInfo",
    "HealthCheckResult",
    "DataQualityResult",
    "MetricsSnapshot",
    "AuditLogEntry",
    # Base Connector - Exceptions
    "ConnectorError",
    "ConnectionError",
    "AuthenticationError",
    "TimeoutError",
    "ValidationError",
    "CircuitOpenError",
    "RetryExhaustedError",
    "ConfigurationError",
    "DataQualityError",
    "RateLimitError",
    "ProtocolError",
    # Base Connector - Components
    "LRUCache",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "CircuitBreaker",
    "ConnectionPool",
    "MetricsCollector",
    "AuditLogger",
    "DataValidator",
    # Base Connector - Base Class
    "BaseConnector",
    "ConnectorContextManager",
    # Base Connector - Decorators
    "with_retry",
    # Base Connector - Factory
    "create_connector_config",

    # CMMS Connector - Enumerations
    "CMSProvider",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "WorkOrderType",
    "EquipmentStatus",
    "EquipmentCriticality",
    "MaintenanceType",
    "AuthenticationType",
    # CMMS Connector - Configuration
    "OAuth2Config",
    "APIKeyConfig",
    "BasicAuthConfig",
    "CMSSConnectorConfig",
    # CMMS Connector - Data Models
    "Equipment",
    "WorkOrder",
    "MaintenanceHistory",
    "SparePart",
    "Notification",
    "CostRecord",
    # CMMS Connector - Request/Response
    "WorkOrderCreateRequest",
    "WorkOrderUpdateRequest",
    "EquipmentQueryParams",
    "MaintenanceHistoryQueryParams",
    # CMMS Connector - Connector
    "CMSSConnector",
    # CMMS Connector - Factory
    "create_cmms_connector",

    # Condition Monitoring - Enumerations
    "ConditionMonitoringProvider",
    "CommunicationProtocol",
    "VibrationUnit",
    "MeasurementType",
    "AlarmSeverity",
    "AlarmState",
    "MeasurementAxis",
    "MachineState",
    "TrendDirection",
    # Condition Monitoring - Configuration
    "OPCUAConfig",
    "ModbusConfig",
    "ConditionMonitoringConnectorConfig",
    # Condition Monitoring - Data Models
    "MeasurementPoint",
    "VibrationReading",
    "SpectrumData",
    "WaveformData",
    "Alarm",
    "TrendData",
    "RouteData",
    # Condition Monitoring - Connector
    "ConditionMonitoringConnector",
    # Condition Monitoring - Factory
    "create_condition_monitoring_connector",

    # IoT Sensor - Enumerations
    "SensorType",
    "SensorProtocol",
    "MQTTQoS",
    "SensorStatus",
    "DataFormat",
    "TimeSyncMode",
    # IoT Sensor - Configuration
    "MQTTConfig",
    "SensorGatewayConfig",
    "IoTSensorConnectorConfig",
    # IoT Sensor - Data Models
    "IoTSensor",
    "SensorReading",
    "SensorBatch",
    "GatewayStatus",
    # IoT Sensor - Connector
    "IoTSensorConnector",
    # IoT Sensor - Factory
    "create_iot_sensor_connector",

    # Agent Coordinator - Enumerations
    "AgentID",
    "MessageType",
    "MessagePriority",
    "TaskStatus",
    "ConsensusState",
    "RoutingStrategy",
    "LoadBalanceStrategy",
    # Agent Coordinator - Configuration
    "MessageBusConfig",
    "AgentCoordinatorConfig",
    # Agent Coordinator - Data Models
    "AgentMessage",
    "AgentResponse",
    "AgentStatus",
    "DistributedTask",
    "ConsensusProposal",
    # Agent Coordinator - Coordinator
    "AgentCoordinator",
    # Agent Coordinator - Factory
    "create_agent_coordinator",

    # Data Transformers - Enumerations
    "UnitCategory",
    "TimeZoneHandling",
    "MissingDataStrategy",
    "OutlierMethod",
    "DataQualityDimension",
    # Data Transformers - Components
    "UnitConverter",
    "TimestampNormalizer",
    "SchemaMapper",
    "DataQualityScorer",
    "MissingDataHandler",
    "OutlierDetector",
    "DataTransformer",
    # Data Transformers - Data Classes
    "UnitDefinition",
    "FieldMapping",
    "QualityMetric",
]


__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"
