"""
GL-002 FLAMEGUARD - Integration Module

SCADA/DCS connectivity and sensor data transformation with circuit breaker
protection for fault-tolerant industrial control system integration.
"""

from .scada_connector import (
    SCADAConnector,
    SCADAProtocol,
    SCADAConnectionConfig,
    TagMapping,
    TagValue,
    TagQuality,
    DataType,
)
from .sensor_transformer import (
    SensorDataTransformer,
    SensorReading,
    TransformationRule,
    DataQuality,
)
from .protected_scada_connector import (
    ProtectedSCADAConnector,
    ProtectedModbusClient,
    ProtectedOPCUAClient,
    FallbackConfig,
    FallbackStrategy,
    DegradedModeLevel,
    CachedValue,
)

__all__ = [
    # SCADA Connector
    "SCADAConnector",
    "SCADAProtocol",
    "SCADAConnectionConfig",
    "TagMapping",
    "TagValue",
    "TagQuality",
    "DataType",
    # Sensor Transformer
    "SensorDataTransformer",
    "SensorReading",
    "TransformationRule",
    "DataQuality",
    # Protected Connectors (with circuit breakers)
    "ProtectedSCADAConnector",
    "ProtectedModbusClient",
    "ProtectedOPCUAClient",
    "FallbackConfig",
    "FallbackStrategy",
    "DegradedModeLevel",
    "CachedValue",
]
