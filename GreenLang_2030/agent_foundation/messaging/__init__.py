"""
GreenLang Agent Foundation - Messaging System

Production-ready message broker for distributed agent communication.
Supports Redis Streams (MVP) and Kafka (scale).

Features:
    - AsyncIO for high concurrency (10K+ msg/s)
    - Consumer groups for parallel processing
    - Dead letter queue (DLQ) for failed messages
    - At-least-once delivery guarantee
    - Request-reply, pub-sub, work queue patterns
    - Circuit breaker and saga patterns
    - Comprehensive monitoring and metrics

Quick Start:
    >>> from messaging import RedisStreamsBroker, MessagingConfig
    >>> config = MessagingConfig.from_env()
    >>> broker = RedisStreamsBroker(**config.redis.to_dict())
    >>> await broker.connect()
    >>> await broker.publish("agent.tasks", {"task": "analyze_esg"})

Performance Targets:
    Redis Streams:
        - Throughput: 10K msg/s
        - Latency P95: < 10ms
        - Max Consumers: 100
        - Message Size: 1MB
        - Retention: 7 days

    Kafka:
        - Throughput: 100K msg/s
        - Latency P95: < 50ms
        - Max Consumers: 1000+
        - Message Size: 10MB
        - Retention: 30 days
"""

# Core message models
from .message import (
    Message,
    MessageBatch,
    MessageAck,
    DeadLetterMessage,
    MessagePriority,
    MessageStatus,
)

# Broker interfaces
from .broker_interface import (
    MessageBrokerInterface,
    BrokerMetrics,
)

# Broker implementations
from .redis_streams_broker import RedisStreamsBroker

# Configuration
from .config import (
    MessagingConfig,
    RedisConfig,
    KafkaConfig,
    load_config,
)

# Messaging patterns
from .patterns import (
    RequestReplyPattern,
    PubSubPattern,
    WorkQueuePattern,
    EventSourcingPattern,
    SagaPattern,
    CircuitBreakerPattern,
    PatternType,
)

# Consumer group management
from .consumer_group import (
    ConsumerGroupManager,
    ConsumerInfo,
    ConsumerGroupStats,
    ConsumerState,
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"

__all__ = [
    # Message models
    "Message",
    "MessageBatch",
    "MessageAck",
    "DeadLetterMessage",
    "MessagePriority",
    "MessageStatus",
    # Broker interface
    "MessageBrokerInterface",
    "BrokerMetrics",
    # Broker implementations
    "RedisStreamsBroker",
    # Configuration
    "MessagingConfig",
    "RedisConfig",
    "KafkaConfig",
    "load_config",
    # Patterns
    "RequestReplyPattern",
    "PubSubPattern",
    "WorkQueuePattern",
    "EventSourcingPattern",
    "SagaPattern",
    "CircuitBreakerPattern",
    "PatternType",
    # Consumer groups
    "ConsumerGroupManager",
    "ConsumerInfo",
    "ConsumerGroupStats",
    "ConsumerState",
]


def get_version() -> str:
    """Get messaging system version."""
    return __version__


def get_broker(broker_type: str = "redis", **config) -> MessageBrokerInterface:
    """
    Factory function to create message broker.

    Args:
        broker_type: "redis" or "kafka"
        **config: Broker configuration parameters

    Returns:
        Configured message broker instance

    Example:
        >>> broker = get_broker("redis", redis_url="redis://localhost:6379")
        >>> await broker.connect()
    """
    if broker_type == "redis":
        return RedisStreamsBroker(**config)
    elif broker_type == "kafka":
        # Import Kafka broker when needed
        try:
            from .kafka_broker import KafkaBroker
            return KafkaBroker(**config)
        except ImportError:
            raise ImportError(
                "Kafka broker not available. Implement kafka_broker.py or use Redis."
            )
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
