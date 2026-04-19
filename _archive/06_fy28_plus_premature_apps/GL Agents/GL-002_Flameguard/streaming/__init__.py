"""
GL-002 FLAMEGUARD - Streaming Module

Kafka-based event streaming for real-time boiler data.
"""

from .kafka_producer import (
    FlameguardKafkaProducer,
    KafkaConfig,
)
from .kafka_consumer import (
    FlameguardKafkaConsumer,
    ConsumerConfig,
)
from .event_schemas import (
    ProcessDataEvent,
    OptimizationEvent,
    SafetyEvent,
    EfficiencyEvent,
    EmissionsEvent,
    AlarmEvent,
)

__all__ = [
    "FlameguardKafkaProducer",
    "KafkaConfig",
    "FlameguardKafkaConsumer",
    "ConsumerConfig",
    "ProcessDataEvent",
    "OptimizationEvent",
    "SafetyEvent",
    "EfficiencyEvent",
    "EmissionsEvent",
    "AlarmEvent",
]
