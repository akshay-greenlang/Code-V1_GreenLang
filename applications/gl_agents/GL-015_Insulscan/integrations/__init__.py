"""
Integrations Module for GL-015 INSULSCAN (Insulation Inspection Agent).

This module provides enterprise-grade data integration capabilities for:
- OPC-UA connectivity to plant automation systems
- Thermal camera integration (FLIR SDK)
- CMMS integration (SAP PM, IBM Maximo)
- Kafka streaming for real-time data
- Process historian connectivity (OSIsoft PI, AVEVA)
- Tag mapping and configuration management

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

# =============================================================================
# OPC-UA Connector
# =============================================================================

from .opcua_connector import (
    # Configuration
    OPCUAConnectorConfig,
    OPCUAServerConfig,
    SubscriptionConfig,
    # Enums
    OPCUASecurityPolicy,
    OPCUASecurityMode,
    OPCUAAuthenticationType,
    NodeClass,
    DataQuality as OPCUADataQuality,
    SubscriptionState,
    # Models
    OPCUATag,
    TagGroup,
    OPCUAValue,
    TemperatureReading,
    DataChangeEvent,
    BrowseResult,
    ServerInfo,
    HealthCheckResult as OPCUAHealthCheckResult,
    # Connector
    OPCUAConnector,
    # Factory functions
    create_opcua_connector,
    create_tag_from_node,
    # Exceptions
    OPCUAError,
    OPCUAConnectionError,
    OPCUASubscriptionError,
    OPCUAReadError,
    OPCUAWriteError,
    OPCUABrowseError,
    OPCUASecurityError,
)

# =============================================================================
# Thermal Camera Connector
# =============================================================================

from .thermal_camera_connector import (
    # Configuration
    ThermalCameraConnectorConfig,
    CameraConfig,
    CaptureSettings,
    HotSpotDetectionConfig,
    # Enums
    CameraManufacturer,
    CameraModel,
    CameraConnectionType,
    ImageFormat,
    TemperatureUnit,
    ColorPalette,
    CalibrationStatus,
    # Models
    CameraSpecs,
    CameraCalibration,
    ThermalPixel,
    TemperatureStatistics,
    HotSpot,
    RegionOfInterest,
    RadiometricMetadata,
    ThermalImage,
    ThermalImageCapture,
    # Connector
    ThermalCameraConnector,
    # Factory functions
    create_flir_camera_connector,
    create_network_thermal_camera,
    # Exceptions
    ThermalCameraError,
    CameraConnectionError,
    CameraNotFoundError,
    ImageCaptureError,
    RadiometricDataError,
    CalibrationError,
)

# =============================================================================
# CMMS Connector
# =============================================================================

from .cmms_connector import (
    # Configuration
    CMSSConnectorConfig,
    OAuth2Config,
    BasicAuthConfig,
    APIKeyConfig,
    SAPPMConfig,
    MaximoConfig,
    # Enums
    CMSSProvider,
    AuthenticationType,
    WorkOrderStatus,
    WorkOrderPriority,
    WorkOrderType,
    InsulationRepairType,
    MaterialType,
    EquipmentType,
    InspectionStatus,
    # Models
    InsulatedEquipment,
    WorkOrderTask,
    MaterialRequisition,
    InsulationRepairWorkOrder,
    WorkOrderCreateRequest,
    InspectionSchedule,
    # Connector
    CMSSConnector,
    # Factory functions
    create_sap_pm_connector,
    create_maximo_connector,
    # Exceptions
    CMSSError,
    CMSSConnectionError,
    CMSSAuthenticationError,
    CMSSValidationError,
    CMSSWorkOrderError,
)

# =============================================================================
# Kafka Streaming
# =============================================================================

from .kafka_streaming import (
    # Configuration
    KafkaStreamingConfig,
    KafkaSecurityConfig,
    SchemaRegistryConfig,
    ProducerConfig,
    ConsumerConfig,
    # Enums
    SerializationType,
    CompressionType,
    AcksMode,
    OffsetResetPolicy,
    IsolationLevel,
    DeliverySemantics,
    ProducerState,
    ConsumerState,
    # Models
    MessageHeaders,
    KafkaMessage,
    ConsumedMessage,
    ProducerAck,
    TemperatureDataMessage,
    ThermalAnalysisResult,
    InsulationDefectEvent,
    WorkOrderCreatedEvent,
    # Components
    AvroSchemaRegistry,
    KafkaProducer,
    KafkaConsumer,
    KafkaStreamingManager,
    # Factory functions
    create_kafka_streaming_config,
    create_kafka_streaming_manager,
    # Exceptions
    KafkaError,
    KafkaConnectionError,
    KafkaProducerError,
    KafkaConsumerError,
    KafkaSerializationError,
    KafkaTransactionError,
    SchemaRegistryError,
)

# =============================================================================
# Historian Connector
# =============================================================================

from .historian_connector import (
    # Configuration
    HistorianConnectorConfig,
    PIWebAPIConfig,
    AVEVAHistorianConfig,
    # Enums
    HistorianProvider,
    PIConnectionType,
    InterpolationMethod,
    AggregationMethod,
    DataQuality as HistorianDataQuality,
    # Models
    HistorianTag,
    HistorianValue,
    TimeSeries,
    TrendData,
    SnapshotValue,
    TimeRange,
    HistorianQuery,
    TrendQuery,
    # Connector
    HistorianConnector,
    # Factory functions
    create_pi_connector,
    create_aveva_connector,
    # Exceptions
    HistorianError,
    HistorianConnectionError,
    HistorianQueryError,
    HistorianTagNotFoundError,
    HistorianAuthenticationError,
)

# =============================================================================
# Tag Mapping
# =============================================================================

from .tag_mapping import (
    # Enums
    TagSource,
    MeasurementType,
    EngineeringUnit,
    TransformationType,
    ValidationRule,
    MappingStatus,
    # Transformation Models
    ScaleTransformation,
    PolynomialTransformation,
    UnitConversion,
    LookupTableEntry,
    LookupTableTransformation,
    ClampTransformation,
    DeadbandTransformation,
    TransformationChain,
    # Validation Models
    RangeValidation,
    RateOfChangeValidation,
    StaleDataValidation,
    ValidationConfig,
    # Mapping Models
    TagMapping,
    TagMappingGroup,
    # Manager
    TagMappingManager,
    # Factory functions
    create_temperature_mapping,
    create_equipment_mapping_group,
    create_tag_mapping_manager,
)

# =============================================================================
# Module Version
# =============================================================================

__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"

# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # OPC-UA
    "OPCUAConnector",
    "OPCUAConnectorConfig",
    "OPCUAServerConfig",
    "SubscriptionConfig",
    "OPCUATag",
    "TagGroup",
    "OPCUAValue",
    "TemperatureReading",
    "DataChangeEvent",
    "BrowseResult",
    "create_opcua_connector",
    "create_tag_from_node",

    # Thermal Camera
    "ThermalCameraConnector",
    "ThermalCameraConnectorConfig",
    "CameraConfig",
    "CaptureSettings",
    "HotSpotDetectionConfig",
    "ThermalImage",
    "ThermalImageCapture",
    "HotSpot",
    "RadiometricMetadata",
    "create_flir_camera_connector",
    "create_network_thermal_camera",

    # CMMS
    "CMSSConnector",
    "CMSSConnectorConfig",
    "SAPPMConfig",
    "MaximoConfig",
    "InsulatedEquipment",
    "InsulationRepairWorkOrder",
    "WorkOrderCreateRequest",
    "WorkOrderTask",
    "MaterialRequisition",
    "InspectionSchedule",
    "create_sap_pm_connector",
    "create_maximo_connector",

    # Kafka
    "KafkaStreamingManager",
    "KafkaProducer",
    "KafkaConsumer",
    "KafkaStreamingConfig",
    "KafkaMessage",
    "ConsumedMessage",
    "ThermalAnalysisResult",
    "InsulationDefectEvent",
    "create_kafka_streaming_manager",

    # Historian
    "HistorianConnector",
    "HistorianConnectorConfig",
    "PIWebAPIConfig",
    "AVEVAHistorianConfig",
    "HistorianTag",
    "TimeSeries",
    "TrendData",
    "HistorianQuery",
    "TrendQuery",
    "create_pi_connector",
    "create_aveva_connector",

    # Tag Mapping
    "TagMappingManager",
    "TagMapping",
    "TagMappingGroup",
    "TransformationChain",
    "ValidationConfig",
    "create_temperature_mapping",
    "create_equipment_mapping_group",
    "create_tag_mapping_manager",

    # Enums (commonly used)
    "CMSSProvider",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "WorkOrderType",
    "InsulationRepairType",
    "MaterialType",
    "EquipmentType",
    "CameraManufacturer",
    "CameraModel",
    "HistorianProvider",
    "InterpolationMethod",
    "AggregationMethod",
    "TagSource",
    "MeasurementType",
    "EngineeringUnit",
]
