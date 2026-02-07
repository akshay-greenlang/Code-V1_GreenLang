# -*- coding: utf-8 -*-
"""
Streaming PII Scanner Package - SEC-011

Real-time PII scanning for streaming platforms (Kafka and Kinesis).
This package provides scanners that consume messages from streams,
detect PII using the enforcement engine, and route messages based on
the detection result.

Components:
    - BaseStreamProcessor: Abstract base class for stream processors
    - KafkaPIIScanner: PII scanner for Apache Kafka streams
    - KinesisPIIScanner: PII scanner for AWS Kinesis Data Streams
    - StreamingConfig: Configuration for streaming scanners
    - StreamingPIIMetrics: Prometheus metrics for observability

Architecture:
    +-----------------+     +------------------+     +----------------+
    |  Input Stream   | --> | PII Scanner      | --> | Output Stream  |
    | (Kafka/Kinesis) |     | (detect/redact)  |     | (clean msgs)   |
    +-----------------+     +------------------+     +----------------+
                                    |
                                    v
                            +----------------+
                            | DLQ Stream     |
                            | (blocked msgs) |
                            +----------------+

Message Flow:
    1. Scanner consumes messages from input topics/streams
    2. Each message is decoded and metadata extracted
    3. Content is scanned using PIIEnforcementEngine
    4. Based on enforcement result:
       - ALLOW: Forward unchanged to output
       - REDACT: Forward with PII redacted
       - BLOCK: Send to dead letter queue
    5. Metrics recorded for all operations

Example Usage - Kafka:
    >>> from greenlang.infrastructure.pii_service.streaming import (
    ...     KafkaPIIScanner,
    ...     StreamingConfig,
    ...     KafkaConfig,
    ... )
    >>> config = StreamingConfig(
    ...     backend="kafka",
    ...     kafka=KafkaConfig(
    ...         bootstrap_servers=["kafka:9092"],
    ...         input_topics=["events.raw"],
    ...         output_topic="events.clean",
    ...         dlq_topic="events.blocked",
    ...         consumer_group="pii-scanner",
    ...     ),
    ...     enforcement_mode="redact",
    ... )
    >>> scanner = KafkaPIIScanner(enforcement_engine, config)
    >>> await scanner.start()

Example Usage - Kinesis:
    >>> from greenlang.infrastructure.pii_service.streaming import (
    ...     KinesisPIIScanner,
    ...     StreamingConfig,
    ...     KinesisConfig,
    ... )
    >>> config = StreamingConfig(
    ...     backend="kinesis",
    ...     kinesis=KinesisConfig(
    ...         stream_name="greenlang-events",
    ...         output_stream_name="greenlang-events-scanned",
    ...         dlq_stream_name="greenlang-events-blocked",
    ...         region="us-west-2",
    ...     ),
    ...     enforcement_mode="redact",
    ... )
    >>> scanner = KinesisPIIScanner(enforcement_engine, config)
    >>> await scanner.start()

Configuration via Environment Variables:
    PII_STREAMING_ENABLED=true
    PII_STREAMING_BACKEND=kafka
    PII_KAFKA_BOOTSTRAP_SERVERS=kafka1:9092,kafka2:9092
    PII_KAFKA_INPUT_TOPICS=events.raw,audit.events
    PII_KAFKA_OUTPUT_TOPIC=events.scanned
    PII_KAFKA_DLQ_TOPIC=events.pii-blocked
    PII_KAFKA_CONSUMER_GROUP=pii-scanner-prod
    PII_KAFKA_SECURITY_PROTOCOL=SASL_SSL
    PII_KAFKA_SASL_MECHANISM=SCRAM-SHA-256
    PII_KAFKA_SASL_USERNAME=scanner
    PII_KAFKA_SASL_PASSWORD=secret

Performance Targets (from PRD):
    - Stream processing: >1,000 msg/sec
    - Processing latency: <10ms P99
    - Detection accuracy: >99%

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from greenlang.infrastructure.pii_service.streaming.config import (
    EnforcementMode,
    KafkaConfig,
    KafkaSASLMechanism,
    KafkaSecurityProtocol,
    KinesisConfig,
    KinesisShardIteratorType,
    StreamingBackend,
    StreamingConfig,
    configure_streaming,
    get_streaming_config,
)
from greenlang.infrastructure.pii_service.streaming.metrics import (
    StreamingPIIMetrics,
    get_streaming_metrics,
    reset_streaming_metrics,
)
from greenlang.infrastructure.pii_service.streaming.stream_processor import (
    BaseStreamProcessor,
    EnforcementContext,
    EnforcementResult,
    PIIDetection,
    PIIEnforcementEngine,
    ProcessingResult,
    ProcessingStats,
)

# Conditionally import scanners to avoid hard dependency on aiokafka/boto3
try:
    from greenlang.infrastructure.pii_service.streaming.kafka_scanner import (
        AIOKAFKA_AVAILABLE,
        KafkaPIIScanner,
        create_kafka_scanner,
    )
except ImportError:
    AIOKAFKA_AVAILABLE = False
    KafkaPIIScanner = None  # type: ignore
    create_kafka_scanner = None  # type: ignore

try:
    from greenlang.infrastructure.pii_service.streaming.kinesis_scanner import (
        BOTO3_AVAILABLE,
        KinesisPIIScanner,
        create_kinesis_scanner,
    )
except ImportError:
    BOTO3_AVAILABLE = False
    KinesisPIIScanner = None  # type: ignore
    create_kinesis_scanner = None  # type: ignore


__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Availability flags
    "AIOKAFKA_AVAILABLE",
    "BOTO3_AVAILABLE",
    # Configuration enums
    "StreamingBackend",
    "EnforcementMode",
    "KafkaSecurityProtocol",
    "KafkaSASLMechanism",
    "KinesisShardIteratorType",
    # Configuration classes
    "KafkaConfig",
    "KinesisConfig",
    "StreamingConfig",
    # Configuration functions
    "get_streaming_config",
    "configure_streaming",
    # Base processor
    "BaseStreamProcessor",
    "PIIEnforcementEngine",
    "EnforcementContext",
    "EnforcementResult",
    # Result models
    "PIIDetection",
    "ProcessingResult",
    "ProcessingStats",
    # Kafka scanner
    "KafkaPIIScanner",
    "create_kafka_scanner",
    # Kinesis scanner
    "KinesisPIIScanner",
    "create_kinesis_scanner",
    # Metrics
    "StreamingPIIMetrics",
    "get_streaming_metrics",
    "reset_streaming_metrics",
]
