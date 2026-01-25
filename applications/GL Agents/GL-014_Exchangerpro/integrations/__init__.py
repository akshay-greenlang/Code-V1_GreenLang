# -*- coding: utf-8 -*-
"""
GL-014 ExchangerPro - Integrations Package

Enterprise-grade integration modules for heat exchanger monitoring and optimization.
All integrations are READ-ONLY by default for OT safety compliance.

Modules:
    - opcua_connector: OPC-UA connectivity with exchanger tag manifest
    - kafka_streaming: Event streaming for real-time data and predictions
    - cmms_connector: CMMS integration for work orders and maintenance
    - historian_connector: Process historian integration for backfills
    - tag_mapping: Canonical tag mapping and unit normalization

Security:
    - IEC 62443 OT cybersecurity compliance
    - Certificate-based authentication
    - Network segmentation support
    - Read-only by default (no control actions)

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

__version__ = "1.0.0"

# OPC-UA Connector
from integrations.opcua_connector import (
    OPCUAConnector,
    OPCUASubscriptionManager,
    OPCUAConnectionPool,
    OPCUAConfig,
    OPCUASecurityConfig,
    OPCUADataPoint,
    OPCUAQualityCode,
    ConnectionState,
    CircuitBreaker,
    DataBuffer,
    StoreAndForwardBuffer,
    ExchangerTagManifest,
    TagOnboardingResult,
    SecurityPolicy,
    SecurityMode,
)

# Kafka Streaming
from integrations.kafka_streaming import (
    KafkaStreamingIntegration,
    KafkaProducerWrapper,
    KafkaConsumerWrapper,
    KafkaStreamConfig,
    GL014Topics,
    ExchangerEventSchema,
    TemperatureEventSchema,
    FlowEventSchema,
    PressureEventSchema,
    KPIEventSchema,
    FoulingPredictionSchema,
    CleaningRecommendationSchema,
    CleaningEventSchema,
    SchemaRegistry,
    ExactlyOnceProducer,
    DeadLetterHandler,
    ConsumerGroup,
)

# CMMS Connector
from integrations.cmms_connector import (
    CMMSConnector,
    SAPPMConnector,
    MaximoConnector,
    CMMSManager,
    WorkOrder,
    WorkOrderPriority,
    WorkOrderType,
    WorkOrderStatus,
    WorkOrderMode,
    MaintenanceEvent,
    MaintenanceCost,
    CleaningRecommendation,
    RecommendationStatus,
    ApprovalWorkflow,
    ComputationLinkage,
)

# Historian Connector
from integrations.historian_connector import (
    HistorianConnector,
    OSIsoftPIConnector,
    HoneywellPHDConnector,
    AspenIP21Connector,
    HistorianConfig,
    TimeRangeQuery,
    BatchIngestionResult,
    DataAggregation,
    AggregationType,
    BackfillRequest,
    BackfillStatus,
    HistorianHealthStatus,
)

# Tag Mapping
from integrations.tag_mapping import (
    TagMapper,
    ExchangerTagSchema,
    CanonicalTagName,
    SiteTagTranslation,
    TagMappingEntry,
    TagMappingConfig,
    UnitNormalizer,
    TagValidator,
    TagValidationResult,
    MeasurementType,
    EngineeringUnit,
)

__all__ = [
    # Version
    "__version__",

    # OPC-UA Connector
    "OPCUAConnector",
    "OPCUASubscriptionManager",
    "OPCUAConnectionPool",
    "OPCUAConfig",
    "OPCUASecurityConfig",
    "OPCUADataPoint",
    "OPCUAQualityCode",
    "ConnectionState",
    "CircuitBreaker",
    "DataBuffer",
    "StoreAndForwardBuffer",
    "ExchangerTagManifest",
    "TagOnboardingResult",
    "SecurityPolicy",
    "SecurityMode",

    # Kafka Streaming
    "KafkaStreamingIntegration",
    "KafkaProducerWrapper",
    "KafkaConsumerWrapper",
    "KafkaStreamConfig",
    "GL014Topics",
    "ExchangerEventSchema",
    "TemperatureEventSchema",
    "FlowEventSchema",
    "PressureEventSchema",
    "KPIEventSchema",
    "FoulingPredictionSchema",
    "CleaningRecommendationSchema",
    "CleaningEventSchema",
    "SchemaRegistry",
    "ExactlyOnceProducer",
    "DeadLetterHandler",
    "ConsumerGroup",

    # CMMS Connector
    "CMMSConnector",
    "SAPPMConnector",
    "MaximoConnector",
    "CMMSManager",
    "WorkOrder",
    "WorkOrderPriority",
    "WorkOrderType",
    "WorkOrderStatus",
    "WorkOrderMode",
    "MaintenanceEvent",
    "MaintenanceCost",
    "CleaningRecommendation",
    "RecommendationStatus",
    "ApprovalWorkflow",
    "ComputationLinkage",

    # Historian Connector
    "HistorianConnector",
    "OSIsoftPIConnector",
    "HoneywellPHDConnector",
    "AspenIP21Connector",
    "HistorianConfig",
    "TimeRangeQuery",
    "BatchIngestionResult",
    "DataAggregation",
    "AggregationType",
    "BackfillRequest",
    "BackfillStatus",
    "HistorianHealthStatus",

    # Tag Mapping
    "TagMapper",
    "ExchangerTagSchema",
    "CanonicalTagName",
    "SiteTagTranslation",
    "TagMappingEntry",
    "TagMappingConfig",
    "UnitNormalizer",
    "TagValidator",
    "TagValidationResult",
    "MeasurementType",
    "EngineeringUnit",
]
