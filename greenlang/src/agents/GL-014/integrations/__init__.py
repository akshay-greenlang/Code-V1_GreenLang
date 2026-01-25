"""
GL-014 EXCHANGER-PRO Integrations Package.

Enterprise-grade data integrations for the Heat Exchanger Optimization Agent.
Provides connectors for process historians, CMMS, DCS/SCADA systems,
multi-agent coordination, and data transformation utilities.

This module enables GL-014 EXCHANGER-PRO to integrate with:
- Process Historians: OSIsoft PI, Honeywell PHD, AspenTech IP.21, OPC-UA
- CMMS Systems: SAP PM, IBM Maximo, Oracle EAM
- DCS/SCADA: Emerson DeltaV, Honeywell Experion, Yokogawa CENTUM
- GreenLang Agents: GL-001 THERMOSYNC, GL-006 HEATRECLAIM, GL-013 PREDICTMAINT

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
    AuthenticationType,
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

# Process Historian Connector
from .process_historian_connector import (
    # Enumerations
    HistorianProvider,
    DataRetrievalMode,
    TagQuality,
    TagDataType,
    TagType,
    # Configuration
    PIWebAPIConfig,
    PHDConfig,
    IP21Config,
    OPCUAConfig,
    ProcessHistorianConnectorConfig,
    # Data Models
    TagDefinition,
    TagValue,
    TimeSeriesData,
    BulkTimeSeriesRequest,
    BulkTimeSeriesResponse,
    TagSearchRequest,
    TagSearchResponse,
    # Heat Exchanger Models
    HeatExchangerTagSet,
    HeatExchangerSnapshot,
    # Connector
    ProcessHistorianConnector,
    # Factory
    create_process_historian_connector,
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
    CleaningMethod,
    # Configuration
    OAuth2Config,
    APIKeyConfig,
    BasicAuthConfig,
    CMSSConnectorConfig,
    # Data Models
    HeatExchangerEquipment,
    CleaningWorkOrder,
    MaintenanceHistory,
    CleaningSchedule,
    Notification,
    # Request/Response Models
    CleaningWorkOrderCreateRequest,
    WorkOrderUpdateRequest,
    EquipmentQueryParams,
    # Connector
    CMSSConnector,
    # Factory
    create_cmms_connector,
)

# DCS/SCADA Connector
from .dcs_scada_connector import (
    # Enumerations
    DCSProvider,
    TagQuality as DCSTagQuality,
    AlarmPriority,
    AlarmState,
    AlarmType,
    ControlMode,
    ModuleStatus,
    SubscriptionType,
    # Configuration
    DeltaVConfig,
    ExperionConfig,
    CentumConfig,
    DCSConnectorConfig,
    # Data Models
    RealtimeTagValue,
    TagSubscription,
    DCSAlarm,
    ControllerStatus,
    HeatExchangerControlTags,
    HeatExchangerRealtimeData,
    SetpointChangeRequest,
    SetpointChangeResponse,
    # Connector
    DCSConnector,
    # Factory
    create_dcs_connector,
)

# Agent Coordinator
from .agent_coordinator import (
    # Enumerations
    AgentID,
    MessageType,
    MessagePriority,
    TaskStatus,
    RoutingStrategy,
    CommunicationProtocol,
    # Configuration
    AgentEndpointConfig,
    MessageBusConfig,
    AgentCoordinatorConfig,
    # Data Models
    AgentMessage,
    AgentResponse,
    AgentStatus,
    # Heat Exchanger Data Sharing Models
    HeatExchangerPerformanceData,
    HeatRecoveryOpportunity,
    MaintenancePrediction,
    ThermalEfficiencyData,
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
    # Unit Conversion
    UnitDefinition,
    UNIT_DEFINITIONS,
    UnitConverter,
    # Timestamp Handling
    TimestampNormalizer,
    # Schema Mapping
    FieldMapping,
    SchemaMapper,
    # Data Quality
    QualityMetric,
    DataQualityScorer,
    # Missing Data
    MissingDataHandler,
    # Outlier Detection
    OutlierDetector,
    # Comprehensive Transformer
    DataTransformer,
)

# Test Utilities
from .test_integrations import (
    # Data Generator
    HeatExchangerDataGenerator,
    # Mock Connectors
    MockProcessHistorianConnector,
    MockCMSSConnector,
    MockDCSConnector,
    MockAgentCoordinator,
    # Response Simulator
    ResponseSimulator,
    # Test Fixtures
    IntegrationTestFixture,
    # Test Utilities
    assert_health_check_passed,
    assert_data_quality_passed,
    run_integration_test,
)


__all__ = [
    # Base Connector - Enumerations
    "ConnectionState",
    "CircuitState",
    "HealthStatus",
    "ConnectorType",
    "DataQualityLevel",
    "RateLimitStrategy",
    "AuthenticationType",
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

    # Process Historian - Enumerations
    "HistorianProvider",
    "DataRetrievalMode",
    "TagQuality",
    "TagDataType",
    "TagType",
    # Process Historian - Configuration
    "PIWebAPIConfig",
    "PHDConfig",
    "IP21Config",
    "OPCUAConfig",
    "ProcessHistorianConnectorConfig",
    # Process Historian - Data Models
    "TagDefinition",
    "TagValue",
    "TimeSeriesData",
    "BulkTimeSeriesRequest",
    "BulkTimeSeriesResponse",
    "TagSearchRequest",
    "TagSearchResponse",
    "HeatExchangerTagSet",
    "HeatExchangerSnapshot",
    # Process Historian - Connector
    "ProcessHistorianConnector",
    "create_process_historian_connector",

    # CMMS - Enumerations
    "CMSProvider",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "WorkOrderType",
    "EquipmentStatus",
    "EquipmentCriticality",
    "MaintenanceType",
    "CleaningMethod",
    # CMMS - Configuration
    "OAuth2Config",
    "APIKeyConfig",
    "BasicAuthConfig",
    "CMSSConnectorConfig",
    # CMMS - Data Models
    "HeatExchangerEquipment",
    "CleaningWorkOrder",
    "MaintenanceHistory",
    "CleaningSchedule",
    "Notification",
    # CMMS - Request/Response
    "CleaningWorkOrderCreateRequest",
    "WorkOrderUpdateRequest",
    "EquipmentQueryParams",
    # CMMS - Connector
    "CMSSConnector",
    "create_cmms_connector",

    # DCS/SCADA - Enumerations
    "DCSProvider",
    "DCSTagQuality",
    "AlarmPriority",
    "AlarmState",
    "AlarmType",
    "ControlMode",
    "ModuleStatus",
    "SubscriptionType",
    # DCS/SCADA - Configuration
    "DeltaVConfig",
    "ExperionConfig",
    "CentumConfig",
    "DCSConnectorConfig",
    # DCS/SCADA - Data Models
    "RealtimeTagValue",
    "TagSubscription",
    "DCSAlarm",
    "ControllerStatus",
    "HeatExchangerControlTags",
    "HeatExchangerRealtimeData",
    "SetpointChangeRequest",
    "SetpointChangeResponse",
    # DCS/SCADA - Connector
    "DCSConnector",
    "create_dcs_connector",

    # Agent Coordinator - Enumerations
    "AgentID",
    "MessageType",
    "MessagePriority",
    "TaskStatus",
    "RoutingStrategy",
    "CommunicationProtocol",
    # Agent Coordinator - Configuration
    "AgentEndpointConfig",
    "MessageBusConfig",
    "AgentCoordinatorConfig",
    # Agent Coordinator - Data Models
    "AgentMessage",
    "AgentResponse",
    "AgentStatus",
    "HeatExchangerPerformanceData",
    "HeatRecoveryOpportunity",
    "MaintenancePrediction",
    "ThermalEfficiencyData",
    # Agent Coordinator - Coordinator
    "AgentCoordinator",
    "create_agent_coordinator",

    # Data Transformers - Enumerations
    "UnitCategory",
    "TimeZoneHandling",
    "MissingDataStrategy",
    "OutlierMethod",
    "DataQualityDimension",
    # Data Transformers - Unit Conversion
    "UnitDefinition",
    "UNIT_DEFINITIONS",
    "UnitConverter",
    # Data Transformers - Timestamp
    "TimestampNormalizer",
    # Data Transformers - Schema Mapping
    "FieldMapping",
    "SchemaMapper",
    # Data Transformers - Quality
    "QualityMetric",
    "DataQualityScorer",
    # Data Transformers - Missing Data
    "MissingDataHandler",
    # Data Transformers - Outliers
    "OutlierDetector",
    # Data Transformers - Comprehensive
    "DataTransformer",

    # Test Utilities
    "HeatExchangerDataGenerator",
    "MockProcessHistorianConnector",
    "MockCMSSConnector",
    "MockDCSConnector",
    "MockAgentCoordinator",
    "ResponseSimulator",
    "IntegrationTestFixture",
    "assert_health_check_passed",
    "assert_data_quality_passed",
    "run_integration_test",
]


__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"
