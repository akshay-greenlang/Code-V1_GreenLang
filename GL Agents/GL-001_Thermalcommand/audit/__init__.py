"""
GL-001 ThermalCommand Audit Module

This module provides comprehensive audit, provenance, and compliance
capabilities for the ThermalCommand ProcessHeatOrchestrator as specified
in Appendix B of the specification.

Key Components:
    - Audit Events: Strongly-typed audit event models
    - Enhanced Provenance: SHA-256 chaining and lineage tracking
    - Evidence Packs: Per-decision evidence pack generation
    - Audit Logger: Append-only logging with hash chaining
    - Query API: Flexible querying and export capabilities

Features:
    - Correlation ID propagation across services
    - Append-only storage with hash chain integrity
    - Multi-index queryability (asset, time, operator, event type, boundary)
    - Retention policy enforcement (7+ years)
    - Export to JSON, CSV, and PDF formats
    - Evidence pack generation for regulatory compliance

Reference Standards:
    - EPA 40 CFR 98 Subpart C (GHG Reporting)
    - ISO 50001:2018 (Energy Management)
    - IEC 61511 (Safety Instrumented Systems)
    - SOC 2 Type II (Audit Requirements)

Example:
    >>> from audit import (
    ...     EnhancedAuditLogger,
    ...     EvidencePackGenerator,
    ...     AuditQueryAPI,
    ...     DecisionAuditEvent,
    ...     correlation_context,
    ... )
    >>>
    >>> # Initialize components
    >>> logger = EnhancedAuditLogger(retention_years=7)
    >>> evidence_gen = EvidencePackGenerator()
    >>> query_api = AuditQueryAPI(logger, evidence_gen)
    >>>
    >>> # Log events with correlation
    >>> with correlation_context("corr-12345"):
    ...     logger.log_decision(decision_event)
    ...     logger.log_action(action_event)
    >>>
    >>> # Query and export
    >>> results = query_api.query(QueryFilter(asset_id="boiler-001"))
    >>> query_api.export(results, ExportFormat.CSV, "audit.csv")

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__all__ = [
    # Audit Events
    "AuditEvent",
    "BaseAuditEvent",
    "DecisionAuditEvent",
    "ActionAuditEvent",
    "SafetyAuditEvent",
    "ComplianceAuditEvent",
    "SystemAuditEvent",
    "OverrideAuditEvent",
    "EventType",
    "SolverStatus",
    "ActionStatus",
    "SafetyLevel",
    "ComplianceStatus",
    "OperatorType",
    "ModelVersionInfo",
    "InputDatasetReference",
    "ConstraintInfo",
    "RecommendedAction",
    "ExpectedImpact",
    "ExplainabilityArtifact",
    "UncertaintyQuantification",
    "create_event_from_dict",
    # Provenance
    "EnhancedProvenanceTracker",
    "ProvenanceNode",
    "ProvenanceEdge",
    "ProvenanceNodeType",
    "LineageGraph",
    "ModelVersionRecord",
    "HashAlgorithm",
    # Evidence Packs
    "EvidencePack",
    "EvidencePackGenerator",
    "EvidencePackFormat",
    "EvidencePackStatus",
    "TimestampRecord",
    "DatasetSummary",
    "ModelVersionSummary",
    "ConstraintSummary",
    "SolverSummary",
    "ExplainabilitySummary",
    "ActionSummary",
    "ImpactSummary",
    "OperatorActionRecord",
    # Audit Logger
    "EnhancedAuditLogger",
    "StorageBackend",
    "FileStorageBackend",
    "InMemoryStorageBackend",
    "HashChainEntry",
    "RetentionPolicy",
    "correlation_context",
    "get_correlation_id",
    "set_correlation_id",
    "generate_correlation_id",
    # Query API
    "AuditQueryAPI",
    "QueryFilter",
    "QueryResult",
    "QuerySortField",
    "QuerySortOrder",
    "ExportFormat",
    "AggregationResult",
    "ComplianceReport",
]

# Import all public components
from .audit_events import (
    AuditEvent,
    BaseAuditEvent,
    DecisionAuditEvent,
    ActionAuditEvent,
    SafetyAuditEvent,
    ComplianceAuditEvent,
    SystemAuditEvent,
    OverrideAuditEvent,
    EventType,
    SolverStatus,
    ActionStatus,
    SafetyLevel,
    ComplianceStatus,
    OperatorType,
    ModelVersionInfo,
    InputDatasetReference,
    ConstraintInfo,
    RecommendedAction,
    ExpectedImpact,
    ExplainabilityArtifact,
    UncertaintyQuantification,
    create_event_from_dict,
)

from .provenance_enhanced import (
    EnhancedProvenanceTracker,
    ProvenanceNode,
    ProvenanceEdge,
    ProvenanceNodeType,
    LineageGraph,
    ModelVersionRecord,
    HashAlgorithm,
)

from .evidence_pack import (
    EvidencePack,
    EvidencePackGenerator,
    EvidencePackFormat,
    EvidencePackStatus,
    TimestampRecord,
    DatasetSummary,
    ModelVersionSummary,
    ConstraintSummary,
    SolverSummary,
    ExplainabilitySummary,
    ActionSummary,
    ImpactSummary,
    OperatorActionRecord,
)

from .audit_logger import (
    EnhancedAuditLogger,
    StorageBackend,
    FileStorageBackend,
    InMemoryStorageBackend,
    HashChainEntry,
    RetentionPolicy,
    correlation_context,
    get_correlation_id,
    set_correlation_id,
    generate_correlation_id,
)

from .audit_query import (
    AuditQueryAPI,
    QueryFilter,
    QueryResult,
    QuerySortField,
    QuerySortOrder,
    ExportFormat,
    AggregationResult,
    ComplianceReport,
)
