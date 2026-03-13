# -*- coding: utf-8 -*-
"""
Chain of Custody Agent - AGENT-EUDR-009

Production-grade chain of custody tracking, batch lifecycle management,
mass balance accounting, transformation processing, document management,
chain verification, and compliance reporting for the EU Deforestation
Regulation (EUDR).

This package provides a comprehensive custody chain engine for EUDR
supply chain traceability supporting four CoC models per ISO 22095:2020:

    CoC Models:
        - Identity Preserved (IP): Full physical separation with
          plot-to-product traceability throughout the chain.
        - Segregated (SG): Compliant material kept separate from
          non-compliant but may mix with other compliant sources.
        - Mass Balance (MB): Administrative tracking with credit-based
          reconciliation (3/12-month periods).
        - Controlled Blending (CB): Tracked blending of compliant and
          non-compliant material with ratio-based claims.

    Foundational modules:
        - config: ChainOfCustodyConfig with GL_EUDR_COC_ env var support
        - models: Pydantic v2 data models with 11 enumerations, 10 core
          models, 12 request models, and 8 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          10 entity types and 12 actions
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_coc_)

    Engine modules:
        - custody_event_tracker: Records and validates custody events
        - batch_lifecycle_manager: Manages batch creation, splits, merges
        - coc_model_enforcer: Enforces IP/SG/MB/CB model rules
        - mass_balance_engine: Maintains input/output mass balance ledgers
        - transformation_tracker: Tracks commodity transformations
        - document_chain_verifier: Verifies document completeness
        - chain_integrity_verifier: Validates end-to-end chain integrity
        - compliance_reporter: Generates Article 9/14 compliance reports

PRD: PRD-AGENT-EUDR-009
Agent ID: GL-EUDR-COC-009
Regulation: EU 2023/1115 (EUDR) Articles 4, 10(2)(f), 14, 31
Standard: ISO 22095:2020 Chain of Custody
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.chain_of_custody import (
    ...     CustodyEvent,
    ...     CustodyEventType,
    ...     Batch,
    ...     BatchStatus,
    ...     CoCModelType,
    ...     ChainOfCustodyConfig,
    ...     get_config,
    ... )
    >>> event = CustodyEvent(
    ...     batch_id="batch-001",
    ...     event_type=CustodyEventType.TRANSFER,
    ...     operator_id="op-001",
    ...     operator_name="Amazonia Cocoa Cooperative",
    ...     country_code="BR",
    ...     commodity="cocoa",
    ...     quantity_kg=5000,
    ...     coc_model=CoCModelType.SEGREGATED,
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-COC-009"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.chain_of_custody.config import (
        ChainOfCustodyConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    ChainOfCustodyConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.chain_of_custody.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_BATCH_SIZE,
        MAX_SPLIT_PARTS,
        MAX_MERGE_BATCHES,
        DEFAULT_GAP_THRESHOLD_HOURS,
        DEFAULT_MAX_AMENDMENT_DEPTH,
        EUDR_RETENTION_YEARS,
        MB_SHORT_CREDIT_MONTHS,
        MB_LONG_CREDIT_MONTHS,
        MB_OVERDRAFT_THRESHOLD_PCT,
        PRIMARY_COMMODITIES,
        DERIVED_TO_PRIMARY,
        DEFAULT_YIELD_RATIOS,
        # Enumerations
        CustodyEventType,
        BatchStatus,
        CoCModelType,
        DocumentType,
        ProcessType,
        OriginAllocationType,
        ReportFormat,
        BatchOperationType,
        GapSeverity,
        VerificationStatus,
        LedgerEntryType,
        # Core Models
        CustodyEvent,
        Batch,
        BatchOrigin,
        BatchOperation,
        CoCModelAssignment,
        MassBalanceEntry,
        TransformationRecord,
        CustodyDocument,
        ChainVerificationResult,
        OriginAllocation,
        # Request Models
        RecordEventRequest,
        CreateBatchRequest,
        SplitBatchRequest,
        MergeBatchRequest,
        BlendBatchRequest,
        AssignModelRequest,
        RecordInputRequest,
        RecordOutputRequest,
        RecordTransformRequest,
        LinkDocumentRequest,
        VerifyChainRequest,
        GenerateReportRequest,
        # Response Models
        RecordEventResponse,
        CreateBatchResponse,
        BatchOperationResponse,
        BalanceResponse,
        TransformResponse,
        VerificationResponse,
        ReportResponse,
        BatchResult,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.chain_of_custody.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_provenance_tracker,
        set_provenance_tracker,
        reset_provenance_tracker,
    )
except ImportError:
    ProvenanceRecord = None  # type: ignore[assignment,misc]
    ProvenanceTracker = None  # type: ignore[assignment,misc]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_provenance_tracker = None  # type: ignore[assignment]
    set_provenance_tracker = None  # type: ignore[assignment]
    reset_provenance_tracker = None  # type: ignore[assignment]

# ---- Foundational: metrics ----
try:
    from greenlang.agents.eudr.chain_of_custody.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        coc_events_recorded_total,
        coc_batches_created_total,
        coc_batch_operations_total,
        coc_mass_balance_entries_total,
        coc_transformations_total,
        coc_documents_linked_total,
        coc_verifications_total,
        coc_verification_failures_total,
        coc_reports_generated_total,
        coc_mass_balance_overdrafts_total,
        coc_custody_gaps_total,
        coc_batch_jobs_total,
        coc_api_errors_total,
        coc_event_recording_duration_seconds,
        coc_verification_duration_seconds,
        coc_mass_balance_duration_seconds,
        coc_active_batches,
        coc_chain_completeness_avg,
        # Helper functions
        record_event_recorded,
        record_batch_created,
        record_batch_operation,
        record_mass_balance_entry,
        record_transformation,
        record_document_linked,
        record_verification,
        record_verification_failure,
        record_report_generated,
        record_mass_balance_overdraft,
        record_custody_gap,
        record_batch_job,
        record_api_error,
        observe_event_recording_duration,
        observe_verification_duration,
        observe_mass_balance_duration,
        set_active_batches,
        set_chain_completeness_avg,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

# ---- Engine 5: Transformation Tracker ----
try:
    from greenlang.agents.eudr.chain_of_custody.transformation_tracker import (
        TransformationTracker,
    )
except ImportError:
    TransformationTracker = None  # type: ignore[assignment,misc]

# ---- Engine 6: Document Chain Verifier ----
try:
    from greenlang.agents.eudr.chain_of_custody.document_chain_verifier import (
        DocumentChainVerifier,
    )
except ImportError:
    DocumentChainVerifier = None  # type: ignore[assignment,misc]

# ---- Engine 7: Chain Integrity Verifier ----
try:
    from greenlang.agents.eudr.chain_of_custody.chain_integrity_verifier import (
        ChainIntegrityVerifier,
    )
except ImportError:
    ChainIntegrityVerifier = None  # type: ignore[assignment,misc]

# ---- Engine 8: Compliance Reporter ----
try:
    from greenlang.agents.eudr.chain_of_custody.compliance_reporter import (
        ComplianceReporter,
    )
except ImportError:
    ComplianceReporter = None  # type: ignore[assignment,misc]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "ChainOfCustodyConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "MAX_SPLIT_PARTS",
    "MAX_MERGE_BATCHES",
    "DEFAULT_GAP_THRESHOLD_HOURS",
    "DEFAULT_MAX_AMENDMENT_DEPTH",
    "EUDR_RETENTION_YEARS",
    "MB_SHORT_CREDIT_MONTHS",
    "MB_LONG_CREDIT_MONTHS",
    "MB_OVERDRAFT_THRESHOLD_PCT",
    "PRIMARY_COMMODITIES",
    "DERIVED_TO_PRIMARY",
    "DEFAULT_YIELD_RATIOS",
    # -- Enumerations --
    "CustodyEventType",
    "BatchStatus",
    "CoCModelType",
    "DocumentType",
    "ProcessType",
    "OriginAllocationType",
    "ReportFormat",
    "BatchOperationType",
    "GapSeverity",
    "VerificationStatus",
    "LedgerEntryType",
    # -- Core Models --
    "CustodyEvent",
    "Batch",
    "BatchOrigin",
    "BatchOperation",
    "CoCModelAssignment",
    "MassBalanceEntry",
    "TransformationRecord",
    "CustodyDocument",
    "ChainVerificationResult",
    "OriginAllocation",
    # -- Request Models --
    "RecordEventRequest",
    "CreateBatchRequest",
    "SplitBatchRequest",
    "MergeBatchRequest",
    "BlendBatchRequest",
    "AssignModelRequest",
    "RecordInputRequest",
    "RecordOutputRequest",
    "RecordTransformRequest",
    "LinkDocumentRequest",
    "VerifyChainRequest",
    "GenerateReportRequest",
    # -- Response Models --
    "RecordEventResponse",
    "CreateBatchResponse",
    "BatchOperationResponse",
    "BalanceResponse",
    "TransformResponse",
    "VerificationResponse",
    "ReportResponse",
    "BatchResult",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "coc_events_recorded_total",
    "coc_batches_created_total",
    "coc_batch_operations_total",
    "coc_mass_balance_entries_total",
    "coc_transformations_total",
    "coc_documents_linked_total",
    "coc_verifications_total",
    "coc_verification_failures_total",
    "coc_reports_generated_total",
    "coc_mass_balance_overdrafts_total",
    "coc_custody_gaps_total",
    "coc_batch_jobs_total",
    "coc_api_errors_total",
    "coc_event_recording_duration_seconds",
    "coc_verification_duration_seconds",
    "coc_mass_balance_duration_seconds",
    "coc_active_batches",
    "coc_chain_completeness_avg",
    "record_event_recorded",
    "record_batch_created",
    "record_batch_operation",
    "record_mass_balance_entry",
    "record_transformation",
    "record_document_linked",
    "record_verification",
    "record_verification_failure",
    "record_report_generated",
    "record_mass_balance_overdraft",
    "record_custody_gap",
    "record_batch_job",
    "record_api_error",
    "observe_event_recording_duration",
    "observe_verification_duration",
    "observe_mass_balance_duration",
    "set_active_batches",
    "set_chain_completeness_avg",
    # -- Engine 5: Transformation Tracker --
    "TransformationTracker",
    # -- Engine 6: Document Chain Verifier --
    "DocumentChainVerifier",
    # -- Engine 7: Chain Integrity Verifier --
    "ChainIntegrityVerifier",
    # -- Engine 8: Compliance Reporter --
    "ComplianceReporter",
]
