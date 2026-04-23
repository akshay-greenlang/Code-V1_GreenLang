"""
GL-006 HEATRECLAIM Streaming Module

Kafka integration for heat exchanger network optimization and pinch analysis.
"""

from .kafka_producer import HeatReclaimKafkaProducer, KafkaProducerConfig
from .kafka_consumer import HeatReclaimKafkaConsumer, KafkaConsumerConfig

__all__ = [
    "HeatReclaimKafkaProducer",
    "HeatReclaimKafkaConsumer",
    "KafkaProducerConfig",
    "KafkaConsumerConfig",
]
