# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Integration Module

Provides connectors and integrations for condenser optimization:
- Condenser sensor integration (temperatures, pressures, flows)
- Cooling water system connectors (pumps, fans, VFDs)
- OPC-UA integration for industrial systems
- Historian connectors (PI, Wonderware, InfluxDB, TimescaleDB)
- CMMS integration (SAP PM, Maximo, Infor)
- Kafka event streaming
- Climate/weather data integration

All connectors implement:
- Async operations with connection pooling
- Circuit breaker pattern for fault tolerance
- Comprehensive error handling and retry logic
- Real-time subscriptions where applicable
- Data quality scoring and validation

Author: GL-DataIntegrationEngineer
Date: December 2025
Version: 1.0.0
"""

from .condenser_sensor_connector import (
    CondenserSensorConnector,
    CondenserSensorConfig,
    SensorReading,
    CondenserDataBundle,
    TagMapping,
    ValidationResult,
    SensorType,
    DataQuality as CondenserDataQuality,
    ConnectionState as CondenserConnectionState,
    ProtocolType as CondenserProtocolType,
    CondenserVendor,
    AlarmSeverity,
    create_condenser_sensor_connector,
)

from .cooling_system_connector import (
    CoolingSystemConnector,
    CoolingSystemConfig,
    PumpStatus,
    FanStatus,
    BasinStatus,
    VFDSetpoint,
    FanStaging,
    CoolingSystemBundle,
    EquipmentType,
    EquipmentStatus,
    FanSpeedMode,
    DataQuality as CoolingDataQuality,
    ConnectionState as CoolingConnectionState,
    ProtocolType as CoolingProtocolType,
    AlarmSeverity as CoolingAlarmSeverity,
    create_cooling_system_connector,
)

from .opc_ua_connector import (
    OPCUAConnector,
    OPCUAConfig,
    CertificateConfig,
    NodeId,
    DataValue,
    BrowseResult,
    MonitoredItem,
    Subscription,
    HistoryReadResult,
    ConnectionState as OPCConnectionState,
    SecurityMode,
    SecurityPolicy,
    AuthenticationType,
    OPCDataQuality,
    NodeClass,
    AttributeId,
    create_opc_ua_connector,
)

from .historian_connector import (
    HistorianConnector,
    HistorianConfig,
    TimeSeriesPoint,
    TimeSeriesData,
    AggregatedValue,
    TrendQuery,
    BackfillResult,
    TagInfo,
    HistorianType,
    InterpolationMode,
    AggregationFunction,
    DataQuality as HistorianDataQuality,
    BackfillStatus,
    ConnectionState as HistorianConnectionState,
    create_historian_connector,
)

from .cmms_connector import (
    CMMSConnector,
    CMMSConfig,
    WorkOrder,
    Asset,
    MaintenanceNotification,
    MaintenanceHistory,
    CMMSType,
    WorkOrderPriority,
    WorkOrderStatus,
    WorkOrderType,
    AssetStatus,
    AssetCriticality,
    NotificationType,
    ConnectionState as CMMSConnectionState,
    create_cmms_connector,
)

from .kafka_connector import (
    KafkaConnector,
    KafkaConfig,
    KafkaMessage,
    ConsumedMessage,
    ProduceResult,
    AvroSchema,
    TopicConfig,
    ConsumerGroup,
    ConnectionState as KafkaConnectionState,
    CompressionType,
    AcksMode,
    AutoOffsetReset,
    IsolationLevel,
    SerializationType,
    TopicType,
    MessagePriority,
    CONDENSER_DATA_SCHEMA,
    OPTIMIZATION_EVENT_SCHEMA,
    create_kafka_connector,
)

from .climate_connector import (
    ClimateConnector,
    ClimateConfig,
    AmbientConditions,
    WeatherForecast,
    HistoricalClimateData,
    PsychrometricCalculator,
    DataSourceType,
    DataQuality as ClimateDataQuality,
    ForecastType,
    ConnectionState as ClimateConnectionState,
    create_climate_connector,
)


__all__ = [
    # Condenser Sensor Connector
    "CondenserSensorConnector",
    "CondenserSensorConfig",
    "SensorReading",
    "CondenserDataBundle",
    "TagMapping",
    "ValidationResult",
    "SensorType",
    "CondenserDataQuality",
    "CondenserConnectionState",
    "CondenserProtocolType",
    "CondenserVendor",
    "AlarmSeverity",
    "create_condenser_sensor_connector",
    # Cooling System Connector
    "CoolingSystemConnector",
    "CoolingSystemConfig",
    "PumpStatus",
    "FanStatus",
    "BasinStatus",
    "VFDSetpoint",
    "FanStaging",
    "CoolingSystemBundle",
    "EquipmentType",
    "EquipmentStatus",
    "FanSpeedMode",
    "CoolingDataQuality",
    "CoolingConnectionState",
    "CoolingProtocolType",
    "CoolingAlarmSeverity",
    "create_cooling_system_connector",
    # OPC-UA Connector
    "OPCUAConnector",
    "OPCUAConfig",
    "CertificateConfig",
    "NodeId",
    "DataValue",
    "BrowseResult",
    "MonitoredItem",
    "Subscription",
    "HistoryReadResult",
    "OPCConnectionState",
    "SecurityMode",
    "SecurityPolicy",
    "AuthenticationType",
    "OPCDataQuality",
    "NodeClass",
    "AttributeId",
    "create_opc_ua_connector",
    # Historian Connector
    "HistorianConnector",
    "HistorianConfig",
    "TimeSeriesPoint",
    "TimeSeriesData",
    "AggregatedValue",
    "TrendQuery",
    "BackfillResult",
    "TagInfo",
    "HistorianType",
    "InterpolationMode",
    "AggregationFunction",
    "HistorianDataQuality",
    "BackfillStatus",
    "HistorianConnectionState",
    "create_historian_connector",
    # CMMS Connector
    "CMMSConnector",
    "CMMSConfig",
    "WorkOrder",
    "Asset",
    "MaintenanceNotification",
    "MaintenanceHistory",
    "CMMSType",
    "WorkOrderPriority",
    "WorkOrderStatus",
    "WorkOrderType",
    "AssetStatus",
    "AssetCriticality",
    "NotificationType",
    "CMMSConnectionState",
    "create_cmms_connector",
    # Kafka Connector
    "KafkaConnector",
    "KafkaConfig",
    "KafkaMessage",
    "ConsumedMessage",
    "ProduceResult",
    "AvroSchema",
    "TopicConfig",
    "ConsumerGroup",
    "KafkaConnectionState",
    "CompressionType",
    "AcksMode",
    "AutoOffsetReset",
    "IsolationLevel",
    "SerializationType",
    "TopicType",
    "MessagePriority",
    "CONDENSER_DATA_SCHEMA",
    "OPTIMIZATION_EVENT_SCHEMA",
    "create_kafka_connector",
    # Climate Connector
    "ClimateConnector",
    "ClimateConfig",
    "AmbientConditions",
    "WeatherForecast",
    "HistoricalClimateData",
    "PsychrometricCalculator",
    "DataSourceType",
    "ClimateDataQuality",
    "ForecastType",
    "ClimateConnectionState",
    "create_climate_connector",
]

__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"
