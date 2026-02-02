"""
GL-003 UNIFIEDSTEAM - Streaming Module

Real-time data streaming infrastructure using Apache Kafka for
steam system optimization.

Topic Hierarchy:
    gl003.<site>.<area>.raw          - Raw sensor signals
    gl003.<site>.<area>.validated    - Validated/transformed signals
    gl003.<site>.<area>.features     - Extracted features
    gl003.<site>.<area>.computed     - Computed properties (enthalpy, efficiency)
    gl003.<site>.<area>.recommendations - Optimization recommendations
    gl003.<site>.<area>.events       - System events and alarms

Components:
- Kafka producer for publishing sensor data and events
- Kafka consumer for processing data streams
- Stream processor for validation, feature extraction, and anomaly detection
- Event publisher for alarms, recommendations, and maintenance events
"""

from .kafka_producer import (
    SteamKafkaProducer,
    KafkaProducerConfig,
    SignalData,
    ValidatedSignalData,
    ComputedPropertyData,
    RecommendationData,
    EventData,
)
from .kafka_consumer import (
    SteamKafkaConsumer,
    KafkaConsumerConfig,
    ConsumedMessage,
    MessageHandler,
    ConsumerGroup,
)
from .stream_processor import (
    StreamProcessor,
    ValidatedSignal,
    FeatureSet,
    Anomaly,
    ProcessingConfig,
    WindowConfig,
)
from .event_publisher import (
    EventPublisher,
    Alarm,
    AlarmSeverity,
    Recommendation,
    RecommendationType,
    MaintenanceEvent,
    SetpointChange,
)

__all__ = [
    # Kafka Producer
    "SteamKafkaProducer",
    "KafkaProducerConfig",
    "SignalData",
    "ValidatedSignalData",
    "ComputedPropertyData",
    "RecommendationData",
    "EventData",
    # Kafka Consumer
    "SteamKafkaConsumer",
    "KafkaConsumerConfig",
    "ConsumedMessage",
    "MessageHandler",
    "ConsumerGroup",
    # Stream Processor
    "StreamProcessor",
    "ValidatedSignal",
    "FeatureSet",
    "Anomaly",
    "ProcessingConfig",
    "WindowConfig",
    # Event Publisher
    "EventPublisher",
    "Alarm",
    "AlarmSeverity",
    "Recommendation",
    "RecommendationType",
    "MaintenanceEvent",
    "SetpointChange",
]
