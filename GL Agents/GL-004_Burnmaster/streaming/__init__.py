"""
GL-004 BURNMASTER Streaming Module

This module provides real-time data streaming and feature computation capabilities
for the BURNMASTER combustion optimization system. It implements edge buffering
with exactly-once semantics and time-aligned feature computation.

Key Components:
    - CombustionDataProducer: Kafka producer with exactly-once semantics
    - CombustionDataConsumer: Kafka consumer with rebalance handling
    - RealTimeFeaturePipeline: Time-aligned feature computation
    - EdgeBuffer: Edge buffer with data integrity guarantees
    - CombustionStreamProcessor: Real-time stream processing
    - TimeSynchronizer: Multi-source timestamp synchronization
    - BackpressureHandler: Backpressure detection and load shedding

Features:
    - Exactly-once delivery semantics via idempotent producers
    - Time-aligned feature computation for ML models
    - Edge buffering for network partition resilience
    - Real-time anomaly detection in combustion data
    - Multi-source clock drift detection and correction
    - Automatic backpressure handling with load shedding

Example:
    >>> from gl004_streaming import CombustionDataProducer, EdgeBuffer
    >>> producer = CombustionDataProducer(config)
    >>> buffer = EdgeBuffer(config)
    >>> await producer.connect(kafka_config)
    >>> result = await producer.publish_combustion_data(data)

Author: GreenLang Combustion Optimization Team
Version: 1.0.0
"""

from __future__ import annotations

# Kafka Producer
from .kafka_producer import (
    CombustionDataProducer,
    KafkaConfig,
    ProducerConfig,
    ConnectionResult,
    PublishResult,
    DeliveryGuarantee,
    PublishFailure,
    RecoveryAction,
    CombustionData,
    CombustionEvent,
    Recommendation,
    Message,
)

# Kafka Consumer
from .kafka_consumer import (
    CombustionDataConsumer,
    ConsumerConfig,
    SubscriptionResult,
    RebalanceResult,
    CommitResult,
    ConsumedMessage,
)

# Feature Pipeline
from .feature_pipeline import (
    RealTimeFeaturePipeline,
    FeatureSpec,
    FeatureVector,
    AlignedFeatures,
    FeatureValidationResult,
    RollingFeatureConfig,
    LagFeatureConfig,
)

# Edge Buffer
from .edge_buffer import (
    EdgeBuffer,
    EdgeBufferConfig,
    BufferResult,
    FlushResult,
    OverflowResult,
    IntegrityCheck,
    BufferOverflowStrategy,
)

# Stream Processor
from .stream_processor import (
    CombustionStreamProcessor,
    StreamProcessorConfig,
    DataStream,
    ProcessedStream,
    TransformedData,
    Anomaly,
    AlertResult,
    AggregatedMetrics,
)

# Time Synchronizer
from .time_synchronizer import (
    TimeSynchronizer,
    TimeSyncConfig,
    SyncResult,
    DriftDetection,
    OrderedEvents,
    SyncIssue,
    Event,
)

# Backpressure Handler
from .backpressure_handler import (
    BackpressureHandler,
    BackpressureConfig,
    BackpressureStatus,
    BackpressureResult,
    LoadShedResult,
    ResumeResult,
    BackpressureStrategy,
)

__all__ = [
    # Kafka Producer
    "CombustionDataProducer",
    "KafkaConfig",
    "ProducerConfig",
    "ConnectionResult",
    "PublishResult",
    "DeliveryGuarantee",
    "PublishFailure",
    "RecoveryAction",
    "CombustionData",
    "CombustionEvent",
    "Recommendation",
    "Message",
    # Kafka Consumer
    "CombustionDataConsumer",
    "ConsumerConfig",
    "SubscriptionResult",
    "RebalanceResult",
    "CommitResult",
    "ConsumedMessage",
    # Feature Pipeline
    "RealTimeFeaturePipeline",
    "FeatureSpec",
    "FeatureVector",
    "AlignedFeatures",
    "FeatureValidationResult",
    "RollingFeatureConfig",
    "LagFeatureConfig",
    # Edge Buffer
    "EdgeBuffer",
    "EdgeBufferConfig",
    "BufferResult",
    "FlushResult",
    "OverflowResult",
    "IntegrityCheck",
    "BufferOverflowStrategy",
    # Stream Processor
    "CombustionStreamProcessor",
    "StreamProcessorConfig",
    "DataStream",
    "ProcessedStream",
    "TransformedData",
    "Anomaly",
    "AlertResult",
    "AggregatedMetrics",
    # Time Synchronizer
    "TimeSynchronizer",
    "TimeSyncConfig",
    "SyncResult",
    "DriftDetection",
    "OrderedEvents",
    "SyncIssue",
    "Event",
    # Backpressure Handler
    "BackpressureHandler",
    "BackpressureConfig",
    "BackpressureStatus",
    "BackpressureResult",
    "LoadShedResult",
    "ResumeResult",
    "BackpressureStrategy",
]

__version__ = "1.0.0"
__author__ = "GreenLang Combustion Optimization Team"
