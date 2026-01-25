"""
GL-009 ThermalIQ - Streaming Module

Kafka integration for thermal fluid analysis event streaming.

Topics:
- thermaliq.analysis.requests - Incoming analysis requests
- thermaliq.analysis.results - Analysis results and computations
- thermaliq.fluids.updates - Fluid property updates
- thermaliq.exergy.results - Exergy analysis results
- thermaliq.sankey.generated - Sankey diagram generation events

Provides:
- ThermalIQKafkaProducer: Event publishing
- ThermalIQKafkaConsumer: Event consumption
- Event schemas for Avro serialization
"""

from .kafka_producer import ThermalIQKafkaProducer, KafkaProducerConfig
from .kafka_consumer import ThermalIQKafkaConsumer, KafkaConsumerConfig
from .event_schemas import (
    EventType,
    MessageHeader,
    AnalysisRequestedEvent,
    AnalysisCompletedEvent,
    FluidPropertyUpdatedEvent,
    ExergyCalculatedEvent,
    SankeyGeneratedEvent,
    AlertEvent,
    get_avro_schema,
    AVRO_SCHEMAS,
)

__all__ = [
    # Producer
    "ThermalIQKafkaProducer",
    "KafkaProducerConfig",
    # Consumer
    "ThermalIQKafkaConsumer",
    "KafkaConsumerConfig",
    # Event types
    "EventType",
    "MessageHeader",
    "AnalysisRequestedEvent",
    "AnalysisCompletedEvent",
    "FluidPropertyUpdatedEvent",
    "ExergyCalculatedEvent",
    "SankeyGeneratedEvent",
    "AlertEvent",
    # Schema utilities
    "get_avro_schema",
    "AVRO_SCHEMAS",
]
