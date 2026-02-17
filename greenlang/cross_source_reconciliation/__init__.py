# -*- coding: utf-8 -*-
"""
GL-DATA-X-018: GreenLang Cross-Source Reconciliation Agent SDK
==============================================================

This package provides cross-source data reconciliation for GreenLang
sustainability datasets. It supports:

- Source registration with metadata (type, schema, priority, credibility,
  refresh cadence), schema mapping rules, column aliases, unit/currency/date
  format definitions, and source health tracking
- Cross-source record matching using composite keys (entity+period+metric),
  fuzzy key matching (Jaro-Winkler, n-gram), temporal alignment
  (daily<->monthly<->quarterly<->annual), blocking for scalability, and
  match confidence scoring
- Field-by-field comparison of matched records with configurable tolerances
  (absolute, relative, percentage), unit-aware comparison, currency-aware
  comparison, null handling, and aggregation-level comparison
- Discrepancy detection and classification: value_mismatch, missing_in_source,
  extra_in_source, timing_difference, unit_difference, aggregation_mismatch,
  with severity scoring (CRITICAL/HIGH/MEDIUM/LOW/INFO) and systematic bias
  pattern detection
- Configurable conflict resolution per field: priority_wins, most_recent,
  weighted_average, most_complete, consensus, manual_review, with golden
  record assembly and per-field lineage tracking
- Complete audit trail: match decisions, comparison results, discrepancy
  classification, resolution choices, golden record field selection with
  source attribution, regulatory attestation, and compliance report assembly
- End-to-end pipeline orchestration: register sources -> align schemas ->
  match records -> compare fields -> detect discrepancies -> resolve conflicts
  -> assemble golden records -> generate audit trail, with batch processing
  and checkpoint/resume
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- 20 REST API endpoints

Key Components:
    - config: CrossSourceReconciliationConfig with GL_CSR_ env prefix
    - source_registry: Source registration and schema mapping engine
    - matching_engine: Cross-source record matching engine
    - comparison_engine: Field-level tolerance-aware comparison engine
    - discrepancy_detector: Discrepancy detection and classification engine
    - resolution_engine: Conflict resolution and golden record assembly engine
    - audit_trail: Audit trail and compliance report engine
    - reconciliation_pipeline: End-to-end pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - setup: Service facade and FastAPI integration

Example:
    >>> from greenlang.cross_source_reconciliation import CrossSourceReconciliationService
    >>> service = CrossSourceReconciliationService()
    >>> service.startup()
    >>> source = service.register_source("ERP System", source_type="erp")
    >>> result = service.run_pipeline(
    ...     records_a=[{"entity_id": "E1", "period": "2025-Q1", "emissions": 100}],
    ...     records_b=[{"entity_id": "E1", "period": "2025-Q1", "emissions": 105}],
    ... )
    >>> print(result["golden_record_count"])

Agent ID: GL-DATA-X-018
Agent Name: Cross-Source Reconciliation Agent
Internal Label: AGENT-DATA-015

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation
Status: Production Ready
"""

# ---------------------------------------------------------------------------
# Agent metadata constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-DATA-X-018"
AGENT_NAME = "Cross-Source Reconciliation Agent"
AGENT_VERSION = "1.0.0"
AGENT_CATEGORY = "Layer 2 - Data Quality Agents"
AGENT_LABEL = "AGENT-DATA-015"

__version__ = AGENT_VERSION
__agent_id__ = AGENT_ID
__agent_name__ = AGENT_NAME

# SDK availability flag
CROSS_SOURCE_RECONCILIATION_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Provenance (2 items)
# ---------------------------------------------------------------------------
from greenlang.cross_source_reconciliation.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
)

# ---------------------------------------------------------------------------
# Metrics (25 items)
# ---------------------------------------------------------------------------
from greenlang.cross_source_reconciliation.metrics import (
    PROMETHEUS_AVAILABLE,
    csr_jobs_processed_total,
    csr_records_matched_total,
    csr_comparisons_total,
    csr_discrepancies_detected_total,
    csr_resolutions_applied_total,
    csr_golden_records_created_total,
    csr_processing_errors_total,
    csr_match_confidence,
    csr_processing_duration_seconds,
    csr_discrepancy_magnitude,
    csr_active_jobs,
    csr_pending_reviews,
    inc_jobs_processed,
    inc_records_matched,
    inc_comparisons,
    inc_discrepancies,
    inc_resolutions,
    inc_golden_records,
    observe_confidence,
    observe_duration,
    observe_magnitude,
    set_active_jobs,
    set_pending_reviews,
    inc_errors,
)

# ---------------------------------------------------------------------------
# Configuration (optional, graceful fallback)
# ---------------------------------------------------------------------------
try:
    from greenlang.cross_source_reconciliation.config import (
        CrossSourceReconciliationConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    CrossSourceReconciliationConfig = None  # type: ignore[assignment, misc]
    get_config = None  # type: ignore[assignment, misc]
    set_config = None  # type: ignore[assignment, misc]
    reset_config = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Models (optional, graceful fallback)
# ---------------------------------------------------------------------------
try:
    from greenlang.cross_source_reconciliation.models import (
        ReconciliationJob,
        DataSource,
        RecordMatch,
        FieldComparison,
        Discrepancy,
        Resolution,
        GoldenRecord,
    )
except ImportError:
    ReconciliationJob = None  # type: ignore[assignment, misc]
    DataSource = None  # type: ignore[assignment, misc]
    RecordMatch = None  # type: ignore[assignment, misc]
    FieldComparison = None  # type: ignore[assignment, misc]
    Discrepancy = None  # type: ignore[assignment, misc]
    Resolution = None  # type: ignore[assignment, misc]
    GoldenRecord = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback (7)
# ---------------------------------------------------------------------------
try:
    from greenlang.cross_source_reconciliation.source_registry import (
        SourceRegistryEngine,
    )
except ImportError:
    SourceRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.matching_engine import (
        MatchingEngine,
    )
except ImportError:
    MatchingEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.comparison_engine import (
        ComparisonEngine,
    )
except ImportError:
    ComparisonEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.discrepancy_detector import (
        DiscrepancyDetectorEngine,
    )
except ImportError:
    DiscrepancyDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.resolution_engine import (
        ResolutionEngine,
    )
except ImportError:
    ResolutionEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.audit_trail import (
        AuditTrailEngine,
    )
except ImportError:
    AuditTrailEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.cross_source_reconciliation.reconciliation_pipeline import (
        ReconciliationPipelineEngine,
    )
except ImportError:
    ReconciliationPipelineEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from data_quality_profiler.consistency_analyzer (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_quality_profiler.consistency_analyzer import (
        ConsistencyAnalyzer as L1ConsistencyAnalyzer,
    )
    ConsistencyAnalyzer = L1ConsistencyAnalyzer
except ImportError:
    ConsistencyAnalyzer = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from duplicate_detector.similarity_scorer (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.duplicate_detector.similarity_scorer import (
        SimilarityScorer as L1SimilarityScorer,
    )
    SimilarityScorer = L1SimilarityScorer
except ImportError:
    SimilarityScorer = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from duplicate_detector.match_classifier (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.duplicate_detector.match_classifier import (
        MatchClassifier as L1MatchClassifier,
    )
    MatchClassifier = L1MatchClassifier
except ImportError:
    MatchClassifier = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from data_engineering.reconciliation (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.data.data_engineering.reconciliation.factor_reconciliation import (
        FactorReconciler as L1FactorReconciler,
    )
    FactorReconciler = L1FactorReconciler
except ImportError:
    FactorReconciler = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade (4 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.cross_source_reconciliation.setup import (
        CrossSourceReconciliationService,
        configure_reconciliation,
        get_reconciliation,
        get_router,
    )
except ImportError:
    CrossSourceReconciliationService = None  # type: ignore[assignment, misc]
    configure_reconciliation = None  # type: ignore[assignment, misc]
    get_reconciliation = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Router (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.cross_source_reconciliation.api.router import router
except ImportError:
    router = None  # type: ignore[assignment]


__all__ = [
    # -------------------------------------------------------------------------
    # Agent metadata (5)
    # -------------------------------------------------------------------------
    "AGENT_ID",
    "AGENT_NAME",
    "AGENT_VERSION",
    "AGENT_CATEGORY",
    "AGENT_LABEL",
    # -------------------------------------------------------------------------
    # Version and identity (3)
    # -------------------------------------------------------------------------
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # -------------------------------------------------------------------------
    # SDK flag (1)
    # -------------------------------------------------------------------------
    "CROSS_SOURCE_RECONCILIATION_SDK_AVAILABLE",
    # -------------------------------------------------------------------------
    # Configuration (4)
    # -------------------------------------------------------------------------
    "CrossSourceReconciliationConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -------------------------------------------------------------------------
    # Provenance (2)
    # -------------------------------------------------------------------------
    "ProvenanceTracker",
    "ProvenanceEntry",
    # -------------------------------------------------------------------------
    # Metrics flag (1)
    # -------------------------------------------------------------------------
    "PROMETHEUS_AVAILABLE",
    # -------------------------------------------------------------------------
    # Metric objects (12)
    # -------------------------------------------------------------------------
    "csr_jobs_processed_total",
    "csr_records_matched_total",
    "csr_comparisons_total",
    "csr_discrepancies_detected_total",
    "csr_resolutions_applied_total",
    "csr_golden_records_created_total",
    "csr_processing_errors_total",
    "csr_match_confidence",
    "csr_processing_duration_seconds",
    "csr_discrepancy_magnitude",
    "csr_active_jobs",
    "csr_pending_reviews",
    # -------------------------------------------------------------------------
    # Metric helper functions (12)
    # -------------------------------------------------------------------------
    "inc_jobs_processed",
    "inc_records_matched",
    "inc_comparisons",
    "inc_discrepancies",
    "inc_resolutions",
    "inc_golden_records",
    "observe_confidence",
    "observe_duration",
    "observe_magnitude",
    "set_active_jobs",
    "set_pending_reviews",
    "inc_errors",
    # -------------------------------------------------------------------------
    # Models (7)
    # -------------------------------------------------------------------------
    "ReconciliationJob",
    "DataSource",
    "RecordMatch",
    "FieldComparison",
    "Discrepancy",
    "Resolution",
    "GoldenRecord",
    # -------------------------------------------------------------------------
    # Core engines (Layer 2) (7)
    # -------------------------------------------------------------------------
    "SourceRegistryEngine",
    "MatchingEngine",
    "ComparisonEngine",
    "DiscrepancyDetectorEngine",
    "ResolutionEngine",
    "AuditTrailEngine",
    "ReconciliationPipelineEngine",
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (4)
    # -------------------------------------------------------------------------
    "ConsistencyAnalyzer",
    "SimilarityScorer",
    "MatchClassifier",
    "FactorReconciler",
    # -------------------------------------------------------------------------
    # Service setup facade (4)
    # -------------------------------------------------------------------------
    "CrossSourceReconciliationService",
    "configure_reconciliation",
    "get_reconciliation",
    "get_router",
    # -------------------------------------------------------------------------
    # Router (1)
    # -------------------------------------------------------------------------
    "router",
]
