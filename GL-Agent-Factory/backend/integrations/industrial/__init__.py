"""
Industrial Protocol Connectors for GreenLang.

This package provides comprehensive industrial protocol integrations
for connecting to PLCs, DCS systems, historians, and IIoT devices.

Supported Protocols:
    - OPC-UA: OPC Unified Architecture for industrial automation
    - Modbus: Modbus TCP/RTU for PLC communication
    - MQTT: IIoT messaging with Sparkplug B support
    - Historians: PI, Aveva, InfluxDB time-series databases
    - DCS: Honeywell Experion, Emerson DeltaV, ABB 800xA

Example:
    >>> from integrations.industrial import (
    ...     OPCUAConnector, OPCUAConfig,
    ...     ModbusConnector, ModbusTCPConfig,
    ...     MQTTConnector, MQTTConfig,
    ... )
    >>>
    >>> # OPC-UA connection
    >>> opcua_config = OPCUAConfig(
    ...     endpoint_url="opc.tcp://localhost:4840"
    ... )
    >>> opcua = OPCUAConnector(opcua_config)
    >>>
    >>> # Modbus TCP connection
    >>> modbus_config = ModbusTCPConfig(
    ...     host="192.168.1.100",
    ...     port=502,
    ...     unit_id=1
    ... )
    >>> modbus = ModbusConnector(modbus_config)
    >>>
    >>> # MQTT IIoT connection
    >>> mqtt_config = MQTTConfig(
    ...     host="mqtt.factory.local",
    ...     port=8883
    ... )
    >>> mqtt = MQTTConnector(mqtt_config)
    >>>
    >>> # Usage with async context manager
    >>> async with opcua:
    ...     values = await opcua.read_tags(["ns=2;s=Temperature"])
    ...     print(values)
"""

# Data Models
from .data_models import (
    # Quality codes
    DataQuality,
    AlarmSeverity,
    AlarmState,
    DataType,
    AggregationType,
    # Core models
    TagValue,
    TagMetadata,
    AlarmEvent,
    # Query models
    HistoricalQuery,
    HistoricalResult,
    # Batch models
    BatchReadRequest,
    BatchReadResponse,
    BatchWriteRequest,
    BatchWriteResponse,
    # Subscription models
    SubscriptionConfig,
    SubscriptionStatus,
    # Connection models
    ConnectionState,
    ConnectionMetrics,
    # Unit conversion
    UnitConversion,
    UNIT_CONVERSIONS,
    convert_units,
)

# Base Connector
from .base import (
    SecurityMode,
    AuthenticationType,
    TLSConfig,
    RateLimitConfig,
    ReconnectConfig,
    BaseConnectorConfig,
    TokenBucketRateLimiter,
    BaseIndustrialConnector,
)

# OPC-UA Connector
from .opcua_connector import (
    OPCUAConfig,
    OPCUASecurityPolicy,
    OPCUAMessageSecurityMode,
    OPCUANodeId,
    OPCUAConnector,
    OPCUASubscriptionHandler,
)

# Modbus Connector
from .modbus_connector import (
    ModbusTCPConfig,
    ModbusRTUConfig,
    ModbusTagConfig,
    ModbusProtocol,
    ModbusFunctionCode,
    ModbusRegisterType,
    ModbusDataType,
    ByteOrder,
    ModbusExceptionCode,
    ModbusConnector,
    ModbusDataConverter,
    ModbusConnection,
    ModbusConnectionPool,
    ModbusException,
)

# MQTT Connector
from .mqtt_connector import (
    MQTTConfig,
    TopicSubscription,
    MQTTVersion,
    QoSLevel,
    MessageFormat,
    SparkplugMessageType,
    SparkplugDataType,
    MQTTMessage,
    SparkplugMetric,
    SparkplugPayload,
    SparkplugEncoder,
    MQTTMessageParser,
    MQTTConnector,
)

# Historian Connectors
from .historian_connector import (
    HistorianType,
    CompressionType,
    InfluxDBVersion,
    BaseHistorianConfig,
    PIConfig,
    AvevaConfig,
    InfluxDBConfig,
    PIConnector,
    AvevaConnector,
    InfluxDBConnector,
    get_historian_connector,
)

# DCS Connectors
from .dcs_connector import (
    DCSVendor,
    ControlModuleType,
    ControlMode,
    TagType,
    DCSTagMapping,
    TagMappingRegistry,
    BaseDCSConfig,
    ExperionConfig,
    DeltaVConfig,
    ABB800xAConfig,
    BaseDCSConnector,
    ExperionConnector,
    DeltaVConnector,
    ABB800xAConnector,
    get_dcs_connector,
)


__all__ = [
    # === Data Models ===
    # Quality
    "DataQuality",
    "AlarmSeverity",
    "AlarmState",
    "DataType",
    "AggregationType",
    # Core
    "TagValue",
    "TagMetadata",
    "AlarmEvent",
    # Query
    "HistoricalQuery",
    "HistoricalResult",
    # Batch
    "BatchReadRequest",
    "BatchReadResponse",
    "BatchWriteRequest",
    "BatchWriteResponse",
    # Subscription
    "SubscriptionConfig",
    "SubscriptionStatus",
    # Connection
    "ConnectionState",
    "ConnectionMetrics",
    # Units
    "UnitConversion",
    "UNIT_CONVERSIONS",
    "convert_units",
    # === Base ===
    "SecurityMode",
    "AuthenticationType",
    "TLSConfig",
    "RateLimitConfig",
    "ReconnectConfig",
    "BaseConnectorConfig",
    "TokenBucketRateLimiter",
    "BaseIndustrialConnector",
    # === OPC-UA ===
    "OPCUAConfig",
    "OPCUASecurityPolicy",
    "OPCUAMessageSecurityMode",
    "OPCUANodeId",
    "OPCUAConnector",
    "OPCUASubscriptionHandler",
    # === Modbus ===
    "ModbusTCPConfig",
    "ModbusRTUConfig",
    "ModbusTagConfig",
    "ModbusProtocol",
    "ModbusFunctionCode",
    "ModbusRegisterType",
    "ModbusDataType",
    "ByteOrder",
    "ModbusExceptionCode",
    "ModbusConnector",
    "ModbusDataConverter",
    "ModbusConnection",
    "ModbusConnectionPool",
    "ModbusException",
    # === MQTT ===
    "MQTTConfig",
    "TopicSubscription",
    "MQTTVersion",
    "QoSLevel",
    "MessageFormat",
    "SparkplugMessageType",
    "SparkplugDataType",
    "MQTTMessage",
    "SparkplugMetric",
    "SparkplugPayload",
    "SparkplugEncoder",
    "MQTTMessageParser",
    "MQTTConnector",
    # === Historians ===
    "HistorianType",
    "CompressionType",
    "InfluxDBVersion",
    "BaseHistorianConfig",
    "PIConfig",
    "AvevaConfig",
    "InfluxDBConfig",
    "PIConnector",
    "AvevaConnector",
    "InfluxDBConnector",
    "get_historian_connector",
    # === DCS ===
    "DCSVendor",
    "ControlModuleType",
    "ControlMode",
    "TagType",
    "DCSTagMapping",
    "TagMappingRegistry",
    "BaseDCSConfig",
    "ExperionConfig",
    "DeltaVConfig",
    "ABB800xAConfig",
    "BaseDCSConnector",
    "ExperionConnector",
    "DeltaVConnector",
    "ABB800xAConnector",
    "get_dcs_connector",
]
