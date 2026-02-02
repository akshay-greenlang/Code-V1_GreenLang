"""
FurnacePulse Integration Module

This module provides enterprise-grade integration components for the FurnacePulse
predictive maintenance system, enabling connectivity with:

- OPC-UA servers for real-time furnace telemetry
- Apache Kafka for event streaming
- CMMS systems for work order management
- IR cameras for thermal imaging
- Historian databases for time-series data

All integrations implement:
- Async/await patterns for high performance
- Comprehensive error handling and retry logic
- Security best practices (certificate auth, credential management)
- Data quality validation and scoring
"""

from .opcua_client import OPCUAClient, OPCUAConfig, TagRegistry, SignalQuality
from .kafka_producer import KafkaEventProducer, KafkaProducerConfig, TelemetryMessage, EventMessage
from .kafka_consumer import KafkaEventConsumer, KafkaConsumerConfig, MessageHandler
from .cmms_integration import CMMSIntegrator, CMMSConfig, WorkOrderRequest, AssetHierarchy
from .ir_camera_client import IRCameraClient, IRCameraConfig, ThermalFrame, HotspotMap
from .historian_client import HistorianClient, HistorianConfig, TimeSeriesQuery

__all__ = [
    # OPC-UA
    "OPCUAClient",
    "OPCUAConfig",
    "TagRegistry",
    "SignalQuality",
    # Kafka Producer
    "KafkaEventProducer",
    "KafkaProducerConfig",
    "TelemetryMessage",
    "EventMessage",
    # Kafka Consumer
    "KafkaEventConsumer",
    "KafkaConsumerConfig",
    "MessageHandler",
    # CMMS
    "CMMSIntegrator",
    "CMMSConfig",
    "WorkOrderRequest",
    "AssetHierarchy",
    # IR Camera
    "IRCameraClient",
    "IRCameraConfig",
    "ThermalFrame",
    "HotspotMap",
    # Historian
    "HistorianClient",
    "HistorianConfig",
    "TimeSeriesQuery",
]

__version__ = "1.0.0"
