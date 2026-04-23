"""
GL-001 ThermalCommand Streaming Module

Kafka streaming integration for real-time process heat orchestration.
Provides event-driven architecture with exactly-once semantics,
schema versioning, and comprehensive audit trails.

Topics:
    - gl001.telemetry.normalized: Normalized time-series points with units and quality
    - gl001.plan.dispatch: Dispatch plan, allocations, solver status, expected impact
    - gl001.actions.recommendations: Setpoint recommendations with bounds and rationale
    - gl001.safety.events: Boundary violations, blocked writes, SIS-permissive changes
    - gl001.maintenance.triggers: Work order recommendations with evidence
    - gl001.explainability.reports: SHAP/LIME summaries, feature contributions, uncertainty
    - gl001.audit.log: Append-only audit events with correlation IDs

Key Components:
    - EventEnvelope: Standard message wrapper with correlation tracking
    - KafkaSchemas: Pydantic models for all topic schemas
    - KafkaStreaming: Producer/Consumer with batching and offset management
    - StreamProcessor: Real-time aggregation with windowing operations

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .event_envelope import (
        EventEnvelope,
        EnvelopeMetadata,
        SchemaVersion,
    )
    from .kafka_schemas import (
        TelemetryNormalizedEvent,
        DispatchPlanEvent,
        ActionRecommendationEvent,
        SafetyEvent,
        MaintenanceTriggerEvent,
        ExplainabilityReportEvent,
        AuditLogEvent,
    )
    from .kafka_streaming import (
        ThermalCommandProducer,
        ThermalCommandConsumer,
        KafkaConfig,
        TopicConfig,
    )
    from .stream_processor import (
        StreamProcessor,
        WindowConfig,
        AggregationResult,
    )

__version__ = "1.0.0"
__all__ = [
    # Event Envelope
    "EventEnvelope",
    "EnvelopeMetadata",
    "SchemaVersion",
    # Kafka Schemas
    "TelemetryNormalizedEvent",
    "DispatchPlanEvent",
    "ActionRecommendationEvent",
    "SafetyEvent",
    "MaintenanceTriggerEvent",
    "ExplainabilityReportEvent",
    "AuditLogEvent",
    # Kafka Streaming
    "ThermalCommandProducer",
    "ThermalCommandConsumer",
    "KafkaConfig",
    "TopicConfig",
    # Stream Processor
    "StreamProcessor",
    "WindowConfig",
    "AggregationResult",
]
