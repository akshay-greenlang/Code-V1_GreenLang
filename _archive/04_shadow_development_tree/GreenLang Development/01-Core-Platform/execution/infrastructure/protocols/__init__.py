"""
GreenLang Infrastructure - Protocol Implementations

This module provides industrial protocol implementations for GreenLang agents,
including OPC-UA, MQTT, Kafka, and Modbus gateways.

All protocols follow:
- Async-first design
- Connection pooling
- Automatic reconnection
- Comprehensive error handling
- Provenance tracking

Enterprise Protocol Framework:
- ProtocolManager: Unified interface for all protocols
- ProtocolType: Enum for protocol identification
- Health monitoring and automatic failover
"""

# Individual protocol implementations
from greenlang.infrastructure.protocols.opcua_server import OPCUAServer
from greenlang.infrastructure.protocols.opcua_client import OPCUAClient
from greenlang.infrastructure.protocols.mqtt_client import MQTTClient
from greenlang.infrastructure.protocols.kafka_producer import KafkaAvroProducer
from greenlang.infrastructure.protocols.kafka_consumer import KafkaExactlyOnceConsumer
from greenlang.infrastructure.protocols.modbus_gateway import ModbusGateway

# Enterprise Protocol Framework - Unified Interface
from greenlang.infrastructure.protocol_framework import (
    # Enums
    ProtocolType,
    ProtocolState,
    HealthStatus,
    SecurityMode,
    SecurityPolicy,
    QoS,
    CompressionType,
    Acks,
    PartitionStrategy,
    AutoOffsetReset,
    ProcessingStatus,
    ModbusProtocol,
    ModbusDataType,
    ByteOrder,
    # Base
    BaseProtocolClient,
    # Configs
    OPCUAServerConfig,
    OPCUAClientConfig,
    MQTTClientConfig,
    KafkaProducerConfig,
    KafkaConsumerConfig,
    ModbusGatewayConfig,
    ProtocolManagerConfig,
    # Models
    AgentNode,
    SubscriptionInfo,
    DataChangeNotification,
    MQTTMessage,
    MQTTSubscription,
    ProducerRecord,
    ProducerResult,
    ConsumerRecord,
    ProcessingResult,
    AvroSchemaRegistry,
    RegisterMapping,
    ModbusValue,
    ProtocolHealth,
    # Manager
    ProtocolManager,
)

__all__ = [
    # Individual implementations
    "OPCUAServer",
    "OPCUAClient",
    "MQTTClient",
    "KafkaAvroProducer",
    "KafkaExactlyOnceConsumer",
    "ModbusGateway",
    # Enums
    "ProtocolType",
    "ProtocolState",
    "HealthStatus",
    "SecurityMode",
    "SecurityPolicy",
    "QoS",
    "CompressionType",
    "Acks",
    "PartitionStrategy",
    "AutoOffsetReset",
    "ProcessingStatus",
    "ModbusProtocol",
    "ModbusDataType",
    "ByteOrder",
    # Base
    "BaseProtocolClient",
    # Configs
    "OPCUAServerConfig",
    "OPCUAClientConfig",
    "MQTTClientConfig",
    "KafkaProducerConfig",
    "KafkaConsumerConfig",
    "ModbusGatewayConfig",
    "ProtocolManagerConfig",
    # Models
    "AgentNode",
    "SubscriptionInfo",
    "DataChangeNotification",
    "MQTTMessage",
    "MQTTSubscription",
    "ProducerRecord",
    "ProducerResult",
    "ConsumerRecord",
    "ProcessingResult",
    "AvroSchemaRegistry",
    "RegisterMapping",
    "ModbusValue",
    "ProtocolHealth",
    # Manager
    "ProtocolManager",
]
