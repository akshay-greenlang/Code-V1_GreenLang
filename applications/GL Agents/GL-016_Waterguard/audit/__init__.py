"""
GL-016 Waterguard Audit Package

This package provides comprehensive audit logging, provenance tracking,
and evidence generation for the Waterguard boiler water chemistry
optimization agent.

Modules:
    - audit_events: Strongly-typed audit event models
    - audit_logger: Immutable append-only audit logging with hash chain
    - audit_query: Query service for audit log retrieval
    - provenance_enhanced: SHA-256 provenance tracking
    - evidence_pack: Regulatory evidence package models
    - evidence_generator: Compliance evidence generation

Key Features:
    - 7-year retention for regulatory compliance
    - Tamper-resistant hash chain (SHA-256)
    - ASME/ABMA alignment documentation
    - Water chemistry decision tracking
    - Operator action audit trails

Example:
    >>> from audit import WaterguardAuditLogger, ChemistryCalculationEvent
    >>> logger = WaterguardAuditLogger(storage_path="/audit/logs")
    >>> logger.log_recommendation(rec_id, inputs, outputs, explanation)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from .audit_events import (
    EventType,
    ChemistryCalculationEvent,
    RecommendationGeneratedEvent,
    CommandExecutedEvent,
    ConstraintViolationEvent,
    ConfigChangeEvent,
    OperatorActionEvent,
    WaterguardAuditEvent,
    create_event_from_dict,
)

from .audit_logger import (
    WaterguardAuditLogger,
    HashChainEntry,
    RetentionPolicy,
    StorageBackend,
    FileStorageBackend,
    InMemoryStorageBackend,
    get_correlation_id,
    set_correlation_id,
    correlation_context,
    generate_correlation_id,
)

from .audit_query import (
    AuditQueryService,
    QueryFilter,
    QueryResult,
    AggregationResult,
    ExportFormat,
)

from .provenance_enhanced import (
    ProvenanceTracker,
    ProvenanceNode,
    ProvenanceEdge,
    LineageGraph,
    ProvenanceNodeType,
    HashAlgorithm,
)

from .evidence_pack import (
    EvidencePack,
    EvidencePackStatus,
    ChemistryCalculationSummary,
    ConstraintComplianceSummary,
    OperatorDecisionRecord,
)

from .evidence_generator import (
    ComplianceEvidenceGenerator,
    WeeklyComplianceReport,
    MonthlyComplianceReport,
    ASMEAlignmentReport,
)

__all__ = [
    # Audit Events
    "EventType",
    "ChemistryCalculationEvent",
    "RecommendationGeneratedEvent",
    "CommandExecutedEvent",
    "ConstraintViolationEvent",
    "ConfigChangeEvent",
    "OperatorActionEvent",
    "WaterguardAuditEvent",
    "create_event_from_dict",
    # Audit Logger
    "WaterguardAuditLogger",
    "HashChainEntry",
    "RetentionPolicy",
    "StorageBackend",
    "FileStorageBackend",
    "InMemoryStorageBackend",
    "get_correlation_id",
    "set_correlation_id",
    "correlation_context",
    "generate_correlation_id",
    # Audit Query
    "AuditQueryService",
    "QueryFilter",
    "QueryResult",
    "AggregationResult",
    "ExportFormat",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceNode",
    "ProvenanceEdge",
    "LineageGraph",
    "ProvenanceNodeType",
    "HashAlgorithm",
    # Evidence Pack
    "EvidencePack",
    "EvidencePackStatus",
    "ChemistryCalculationSummary",
    "ConstraintComplianceSummary",
    "OperatorDecisionRecord",
    # Evidence Generator
    "ComplianceEvidenceGenerator",
    "WeeklyComplianceReport",
    "MonthlyComplianceReport",
    "ASMEAlignmentReport",
]

__version__ = "1.0.0"
