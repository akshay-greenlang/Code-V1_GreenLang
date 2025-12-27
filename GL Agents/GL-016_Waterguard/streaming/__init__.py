"""
GL-016 Waterguard Streaming Package

Kafka-based streaming infrastructure for boiler water chemistry monitoring.
Provides producers, consumers, stream processors, and event envelope handling.
"""

from streaming.kafka_streaming import (
    WaterguardKafkaProducer,
    WaterguardKafkaConsumer,
    KafkaConfig,
)
from streaming.kafka_schemas import (
    RawChemistryMessage,
    CleanedChemistryMessage,
    RecommendationMessage,
    CommandMessage,
    AckMessage,
    AlertMessage,
    AuditMessage,
)
from streaming.stream_processor import (
    DataCleaningProcessor,
    FeatureEngineeringProcessor,
    QualityFlag,
)
from streaming.event_envelope import (
    EventEnvelope,
    EventMetadata,
)

__all__ = [
    # Kafka streaming
    "WaterguardKafkaProducer",
    "WaterguardKafkaConsumer",
    "KafkaConfig",
    # Schemas
    "RawChemistryMessage",
    "CleanedChemistryMessage",
    "RecommendationMessage",
    "CommandMessage",
    "AckMessage",
    "AlertMessage",
    "AuditMessage",
    # Processors
    "DataCleaningProcessor",
    "FeatureEngineeringProcessor",
    "QualityFlag",
    # Event envelope
    "EventEnvelope",
    "EventMetadata",
]

__version__ = "1.0.0"
