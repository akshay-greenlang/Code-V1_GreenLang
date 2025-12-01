"""
GL-015 INSULSCAN Integrations Package.

Enterprise-grade data integrations for the Insulation Inspection Agent.
Provides connectors for thermal cameras, CMMS systems, asset management,
weather services, multi-agent coordination, and data transformation utilities.

This module enables GL-015 INSULSCAN to integrate with:
- Thermal Cameras: FLIR Systems, Fluke Ti-series, Testo, Optris PI, InfraTec ImageIR, ONVIF
- CMMS Systems: SAP PM, IBM Maximo, Oracle EAM
- Asset Management: Equipment registry, insulation inventory, location hierarchy
- Weather Services: OpenWeatherMap, NOAA
- GreenLang Agents: GL-001 THERMOSYNC, GL-006 HEATRECLAIM, GL-014 EXCHANGER-PRO

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
    ImageFormat,
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
    CameraError,
    StreamingError,
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

# Thermal Camera Connector
from .thermal_camera_connector import (
    # Enumerations
    ThermalCameraProvider,
    StreamingMode,
    ImageOutputFormat,
    TemperatureUnit,
    PaletteType,
    # Configuration
    FLIRConfig,
    FlukeTiConfig,
    TestoConfig,
    OptrisPIConfig,
    InfraTecConfig,
    ONVIFConfig,
    StreamingConfig,
    ThermalCameraConnectorConfig,
    # Data Models
    CameraInfo,
    CalibrationData,
    RadiometricParameters,
    TemperatureMatrix,
    ThermalImage,
    ThermalFrame,
    RegionOfInterest,
    SpotMeasurement,
    ThermalImageCapture,
    ThermalStreamFrame,
    # Connector
    ThermalCameraConnector,
    # Factory
    create_thermal_camera_connector,
)

# CMMS Connector
from .cmms_connector import (
    # Enumerations
    CMSSProvider,
    WorkOrderStatus,
    WorkOrderPriority,
    WorkOrderType,
    InsulationCondition,
    RepairType,
    MaterialType,
    # Configuration
    OAuth2Config,
    APIKeyConfig,
    BasicAuthConfig,
    CMSSConnectorConfig,
    # Data Models
    InsulatedEquipment,
    InsulationRepairWorkOrder,
    MaterialRequisition,
    InspectionSchedule,
    MaintenanceHistory,
    InsulationDefectRecord,
    # Request/Response Models
    RepairWorkOrderCreateRequest,
    MaterialRequisitionRequest,
    InspectionScheduleRequest,
    WorkOrderUpdateRequest,
    EquipmentQueryParams,
    # Connector
    CMSSConnector,
    # Factory
    create_cmms_connector,
)

# Asset Management Connector
from .asset_management_connector import (
    # Enumerations
    LocationLevel,
    AssetStatus,
    InsulationType,
    CladdingMaterial,
    # Configuration
    AssetManagementConnectorConfig,
    # Data Models
    LocationNode,
    LocationHierarchy,
    InsulatedEquipmentAsset,
    InsulationSpecification,
    InsulationMaterialStock,
    TagMapping,
    AssetInspectionHistory,
    InsulationConditionAssessment,
    # Connector
    AssetManagementConnector,
    # Factory
    create_asset_management_connector,
)

# Weather Connector
from .weather_connector import (
    # Enumerations
    WeatherProvider,
    WeatherCondition,
    WindDirection,
    PrecipitationType,
    InspectionSuitability,
    # Configuration
    OpenWeatherMapConfig,
    NOAAConfig,
    WeatherConnectorConfig,
    # Data Models
    CurrentWeather,
    HourlyForecast,
    DailyForecast,
    WeatherAlert,
    InspectionWindow,
    InspectionPlanningReport,
    HistoricalWeather,
    # Connector
    WeatherConnector,
    # Factory
    create_weather_connector,
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
    # Insulation-specific Data Sharing Models
    ThermalEfficiencyContext,
    HeatRecoveryOpportunity,
    HeatExchangerInsulationContext,
    InsulationDefectData,
    InsulationRecommendation,
    AgentCollaborationRequest,
    AgentCollaborationResponse,
    # Coordinator
    AgentCoordinator,
    # Factory
    create_agent_coordinator,
)

# Data Transformers
from .data_transformers import (
    # Enumerations
    TemperatureUnitType,
    ThermalUnitType,
    InsulationUnitType,
    NormalizationMethod,
    InterpolationMethod,
    NoiseReductionMethod,
    # Unit Conversion
    UnitConverter,
    # Temperature Matrix Processing
    TemperatureMatrixNormalizer,
    # Schema Mapping
    FieldMapping,
    SchemaMapper,
    # Data Quality
    ThermalDataQualityScorer,
    # Image Processing
    ThermalImageProcessor,
    # Comprehensive Transformer
    InsulationDataTransformer,
)

# Test Utilities
from .test_integrations import (
    # Data Generator
    InsulationDataGenerator,
    # Mock Connectors
    MockThermalCameraConnector,
    MockCMSSConnector,
    MockAssetManagementConnector,
    MockWeatherConnector,
    MockAgentCoordinator,
    # Response Simulator
    ResponseSimulator,
    # Test Fixtures
    IntegrationTestFixture,
    # Test Utilities
    assert_health_check_passed,
    assert_data_quality_passed,
    assert_temperature_in_range,
    assert_defect_valid,
    run_integration_test,
    run_connector_health_check_suite,
    # Sample Tests
    sample_thermal_image_capture_test,
    sample_work_order_creation_test,
    sample_agent_coordination_test,
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
    "ImageFormat",
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
    "CameraError",
    "StreamingError",
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

    # Thermal Camera - Enumerations
    "ThermalCameraProvider",
    "StreamingMode",
    "ImageOutputFormat",
    "TemperatureUnit",
    "PaletteType",
    # Thermal Camera - Configuration
    "FLIRConfig",
    "FlukeTiConfig",
    "TestoConfig",
    "OptrisPIConfig",
    "InfraTecConfig",
    "ONVIFConfig",
    "StreamingConfig",
    "ThermalCameraConnectorConfig",
    # Thermal Camera - Data Models
    "CameraInfo",
    "CalibrationData",
    "RadiometricParameters",
    "TemperatureMatrix",
    "ThermalImage",
    "ThermalFrame",
    "RegionOfInterest",
    "SpotMeasurement",
    "ThermalImageCapture",
    "ThermalStreamFrame",
    # Thermal Camera - Connector
    "ThermalCameraConnector",
    "create_thermal_camera_connector",

    # CMMS - Enumerations
    "CMSSProvider",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "WorkOrderType",
    "InsulationCondition",
    "RepairType",
    "MaterialType",
    # CMMS - Configuration
    "OAuth2Config",
    "APIKeyConfig",
    "BasicAuthConfig",
    "CMSSConnectorConfig",
    # CMMS - Data Models
    "InsulatedEquipment",
    "InsulationRepairWorkOrder",
    "MaterialRequisition",
    "InspectionSchedule",
    "MaintenanceHistory",
    "InsulationDefectRecord",
    # CMMS - Request/Response
    "RepairWorkOrderCreateRequest",
    "MaterialRequisitionRequest",
    "InspectionScheduleRequest",
    "WorkOrderUpdateRequest",
    "EquipmentQueryParams",
    # CMMS - Connector
    "CMSSConnector",
    "create_cmms_connector",

    # Asset Management - Enumerations
    "LocationLevel",
    "AssetStatus",
    "InsulationType",
    "CladdingMaterial",
    # Asset Management - Configuration
    "AssetManagementConnectorConfig",
    # Asset Management - Data Models
    "LocationNode",
    "LocationHierarchy",
    "InsulatedEquipmentAsset",
    "InsulationSpecification",
    "InsulationMaterialStock",
    "TagMapping",
    "AssetInspectionHistory",
    "InsulationConditionAssessment",
    # Asset Management - Connector
    "AssetManagementConnector",
    "create_asset_management_connector",

    # Weather - Enumerations
    "WeatherProvider",
    "WeatherCondition",
    "WindDirection",
    "PrecipitationType",
    "InspectionSuitability",
    # Weather - Configuration
    "OpenWeatherMapConfig",
    "NOAAConfig",
    "WeatherConnectorConfig",
    # Weather - Data Models
    "CurrentWeather",
    "HourlyForecast",
    "DailyForecast",
    "WeatherAlert",
    "InspectionWindow",
    "InspectionPlanningReport",
    "HistoricalWeather",
    # Weather - Connector
    "WeatherConnector",
    "create_weather_connector",

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
    "ThermalEfficiencyContext",
    "HeatRecoveryOpportunity",
    "HeatExchangerInsulationContext",
    "InsulationDefectData",
    "InsulationRecommendation",
    "AgentCollaborationRequest",
    "AgentCollaborationResponse",
    # Agent Coordinator - Coordinator
    "AgentCoordinator",
    "create_agent_coordinator",

    # Data Transformers - Enumerations
    "TemperatureUnitType",
    "ThermalUnitType",
    "InsulationUnitType",
    "NormalizationMethod",
    "InterpolationMethod",
    "NoiseReductionMethod",
    # Data Transformers - Unit Conversion
    "UnitConverter",
    # Data Transformers - Matrix Processing
    "TemperatureMatrixNormalizer",
    # Data Transformers - Schema Mapping
    "FieldMapping",
    "SchemaMapper",
    # Data Transformers - Quality
    "ThermalDataQualityScorer",
    # Data Transformers - Image Processing
    "ThermalImageProcessor",
    # Data Transformers - Comprehensive
    "InsulationDataTransformer",

    # Test Utilities - Data Generator
    "InsulationDataGenerator",
    # Test Utilities - Mock Connectors
    "MockThermalCameraConnector",
    "MockCMSSConnector",
    "MockAssetManagementConnector",
    "MockWeatherConnector",
    "MockAgentCoordinator",
    # Test Utilities - Response Simulator
    "ResponseSimulator",
    # Test Utilities - Fixtures
    "IntegrationTestFixture",
    # Test Utilities - Assertions
    "assert_health_check_passed",
    "assert_data_quality_passed",
    "assert_temperature_in_range",
    "assert_defect_valid",
    "run_integration_test",
    "run_connector_health_check_suite",
    # Test Utilities - Sample Tests
    "sample_thermal_image_capture_test",
    "sample_work_order_creation_test",
    "sample_agent_coordination_test",
]


__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"
