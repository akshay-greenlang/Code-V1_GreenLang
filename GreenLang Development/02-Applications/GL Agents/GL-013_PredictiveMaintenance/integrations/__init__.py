# -*- coding: utf-8 -*-
"""
GL-013 PredictiveMaintenance - Integrations Package

This package provides enterprise-grade integration modules for the
Predictive Maintenance agent.

Author: GreenLang Predictive Maintenance Team
Version: 1.0.0
"""

__version__ = "1.0.0"

from integrations.opcua_connector import (
    OPCUAConnector,
    OPCUASubscriptionManager,
    OPCUAConnectionPool,
    OPCUAConfig,
    OPCUATagConfig,
    OPCUADataPoint,
    OPCUAQualityCode,
    ConnectionState,
    CircuitBreaker,
    DataBuffer,
)

from integrations.kafka_streaming import (
    KafkaProducerWrapper,
    KafkaConsumerWrapper,
    KafkaStreamConfig,
    KafkaTopics,
    MessageSchema,
    DeadLetterHandler,
    SchemaRegistry,
    ExactlyOnceProducer,
)

from integrations.graphql_api import (
    GraphQLService,
    AssetType,
    PredictionType,
    ExplanationType,
    WorkOrderMutation,
    AuthorizationMiddleware,
    AuditLogger,
    QueryResolver,
    MutationResolver,
)

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
    EvidenceAttachment,
    FeedbackIngestion,
)

from integrations.tag_mapping import (
    TagMappingRegistry,
    TagMapping,
    TagAlias,
    ChangeControlEntry,
    SiteConfig,
    VendorAlias,
    TagResolutionResult,
)

__all__ = [
    "__version__",
    "OPCUAConnector", "OPCUASubscriptionManager", "OPCUAConnectionPool",
    "OPCUAConfig", "OPCUATagConfig", "OPCUADataPoint", "OPCUAQualityCode",
    "ConnectionState", "CircuitBreaker", "DataBuffer",
    "KafkaProducerWrapper", "KafkaConsumerWrapper", "KafkaStreamConfig",
    "KafkaTopics", "MessageSchema", "DeadLetterHandler",
    "SchemaRegistry", "ExactlyOnceProducer",
    "GraphQLService", "AssetType", "PredictionType", "ExplanationType",
    "WorkOrderMutation", "AuthorizationMiddleware", "AuditLogger",
    "QueryResolver", "MutationResolver",
    "CMMSConnector", "SAPPMConnector", "MaximoConnector", "CMMSManager",
    "WorkOrder", "WorkOrderPriority", "WorkOrderType", "WorkOrderStatus",
    "WorkOrderMode", "EvidenceAttachment", "FeedbackIngestion",
    "TagMappingRegistry", "TagMapping", "TagAlias", "ChangeControlEntry",
    "SiteConfig", "VendorAlias", "TagResolutionResult",
]
