# -*- coding: utf-8 -*-
"""
GL-DATA-X-019: GreenLang Data Freshness Monitor Agent SDK
==========================================================

This package provides dataset freshness monitoring for GreenLang
sustainability datasets. It supports:

- Dataset registration with metadata (name, source, owner, refresh cadence,
  SLA config, priority, tags), dataset grouping, bulk registration, health
  status tracking
- SLA definition and management per dataset/group: warning threshold (hours),
  critical threshold (hours), breach severity classification
  (INFO/LOW/MEDIUM/HIGH/CRITICAL), escalation policies, SLA templates
- Freshness checking: compute age since last update, apply freshness scoring
  (5-tier: excellent/good/fair/poor/stale), evaluate SLA compliance, batch
  checking across all registered datasets, incremental checks
- Staleness pattern detection: historical trend analysis, recurring staleness
  identification, seasonal pattern detection, source reliability scoring,
  systematic failure detection, refresh drift monitoring
- Refresh prediction: estimate next expected refresh time based on historical
  patterns, detect anomalous delays, compute refresh regularity score,
  identify degrading refresh patterns
- Alert generation and management: multi-severity alerts (warning/critical/
  emergency), alert deduplication and throttling, escalation chain execution,
  alert lifecycle (open->acknowledged->resolved), notification formatting
- End-to-end pipeline orchestration: register datasets -> check freshness ->
  detect staleness -> predict refreshes -> evaluate SLAs -> generate alerts
  -> produce reports, with batch processing and checkpoint/resume
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- 20 REST API endpoints

Key Components:
    - config: DataFreshnessMonitorConfig with GL_DFM_ env prefix
    - dataset_registry: Dataset registration and metadata management engine
    - sla_definition: SLA rule creation and management engine
    - freshness_checker: Freshness scoring and SLA compliance engine
    - staleness_detector: Staleness pattern detection and source reliability engine
    - refresh_predictor: Refresh prediction and anomaly detection engine
    - alert_manager: Alert generation, deduplication, and escalation engine
    - freshness_pipeline: End-to-end monitoring pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - setup: Service facade and FastAPI integration

Example:
    >>> from greenlang.data_freshness_monitor import DataFreshnessMonitorService
    >>> service = DataFreshnessMonitorService()
    >>> service.startup()
    >>> dataset = service.register_dataset("ERP Revenue", source="erp")
    >>> result = service.run_check(dataset_id=dataset["dataset_id"])
    >>> print(result["freshness_level"], result["sla_status"])

Agent ID: GL-DATA-X-019
Agent Name: Data Freshness Monitor Agent
Internal Label: AGENT-DATA-016

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor
Status: Production Ready
"""

# ---------------------------------------------------------------------------
# Agent metadata constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-DATA-X-019"
AGENT_NAME = "Data Freshness Monitor Agent"
AGENT_VERSION = "1.0.0"
AGENT_CATEGORY = "Layer 2 - Data Quality Agents"
AGENT_LABEL = "AGENT-DATA-016"

__version__ = AGENT_VERSION
__agent_id__ = AGENT_ID
__agent_name__ = AGENT_NAME

# SDK availability flag
DATA_FRESHNESS_MONITOR_SDK_AVAILABLE = True

VERSION = AGENT_VERSION

# ---------------------------------------------------------------------------
# Provenance (3 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_freshness_monitor.provenance import (
        ProvenanceTracker,
        ProvenanceEntry,
        get_provenance_tracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]
    ProvenanceEntry = None  # type: ignore[assignment, misc]
    get_provenance_tracker = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Metrics (24 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_freshness_monitor.metrics import (
        PROMETHEUS_AVAILABLE,
        dfm_checks_performed_total,
        dfm_sla_breaches_total,
        dfm_alerts_sent_total,
        dfm_datasets_registered_total,
        dfm_refresh_events_total,
        dfm_predictions_made_total,
        dfm_freshness_score,
        dfm_data_age_hours,
        dfm_processing_duration_seconds,
        dfm_active_breaches,
        dfm_monitored_datasets,
        dfm_processing_errors_total,
        record_check,
        record_breach,
        record_alert,
        record_dataset_registered,
        record_refresh_event,
        record_prediction,
        observe_freshness_score,
        observe_data_age,
        observe_duration,
        set_active_breaches,
        set_monitored_datasets,
        record_error,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    dfm_checks_performed_total = None  # type: ignore[assignment]
    dfm_sla_breaches_total = None  # type: ignore[assignment]
    dfm_alerts_sent_total = None  # type: ignore[assignment]
    dfm_datasets_registered_total = None  # type: ignore[assignment]
    dfm_refresh_events_total = None  # type: ignore[assignment]
    dfm_predictions_made_total = None  # type: ignore[assignment]
    dfm_freshness_score = None  # type: ignore[assignment]
    dfm_data_age_hours = None  # type: ignore[assignment]
    dfm_processing_duration_seconds = None  # type: ignore[assignment]
    dfm_active_breaches = None  # type: ignore[assignment]
    dfm_monitored_datasets = None  # type: ignore[assignment]
    dfm_processing_errors_total = None  # type: ignore[assignment]
    record_check = None  # type: ignore[assignment]
    record_breach = None  # type: ignore[assignment]
    record_alert = None  # type: ignore[assignment]
    record_dataset_registered = None  # type: ignore[assignment]
    record_refresh_event = None  # type: ignore[assignment]
    record_prediction = None  # type: ignore[assignment]
    observe_freshness_score = None  # type: ignore[assignment]
    observe_data_age = None  # type: ignore[assignment]
    observe_duration = None  # type: ignore[assignment]
    set_active_breaches = None  # type: ignore[assignment]
    set_monitored_datasets = None  # type: ignore[assignment]
    record_error = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration (4 items, optional graceful fallback)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_freshness_monitor.config import (
        DataFreshnessMonitorConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    DataFreshnessMonitorConfig = None  # type: ignore[assignment, misc]
    get_config = None  # type: ignore[assignment, misc]
    set_config = None  # type: ignore[assignment, misc]
    reset_config = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Models (20 items, optional graceful fallback)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_freshness_monitor.models import (
        FreshnessTier,
        CheckStatus,
        AlertChannel,
        AlertStatus,
        AlertSeverity,
        BreachStatus,
        BreachSeverity,
        SLAStatus,
        PredictionConfidence,
        PatternType,
        EscalationLevel,
        EscalationPolicy,
        DatasetRegistration,
        FreshnessCheck,
        SLADefinition,
        SLABreach,
        FreshnessAlert,
        StalenessPrediction,
        FreshnessPattern,
        DataFreshnessMonitorStatistics,
    )
except ImportError:
    FreshnessTier = None  # type: ignore[assignment, misc]
    CheckStatus = None  # type: ignore[assignment, misc]
    AlertChannel = None  # type: ignore[assignment, misc]
    AlertStatus = None  # type: ignore[assignment, misc]
    AlertSeverity = None  # type: ignore[assignment, misc]
    BreachStatus = None  # type: ignore[assignment, misc]
    BreachSeverity = None  # type: ignore[assignment, misc]
    SLAStatus = None  # type: ignore[assignment, misc]
    PredictionConfidence = None  # type: ignore[assignment, misc]
    PatternType = None  # type: ignore[assignment, misc]
    EscalationLevel = None  # type: ignore[assignment, misc]
    EscalationPolicy = None  # type: ignore[assignment, misc]
    DatasetRegistration = None  # type: ignore[assignment, misc]
    FreshnessCheck = None  # type: ignore[assignment, misc]
    SLADefinition = None  # type: ignore[assignment, misc]
    SLABreach = None  # type: ignore[assignment, misc]
    FreshnessAlert = None  # type: ignore[assignment, misc]
    StalenessPrediction = None  # type: ignore[assignment, misc]
    FreshnessPattern = None  # type: ignore[assignment, misc]
    DataFreshnessMonitorStatistics = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback (7)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_freshness_monitor.dataset_registry import (
        DatasetRegistryEngine,
    )
except ImportError:
    DatasetRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.sla_definition import (
        SLADefinitionEngine,
    )
except ImportError:
    SLADefinitionEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.freshness_checker import (
        FreshnessCheckerEngine,
    )
except ImportError:
    FreshnessCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.staleness_detector import (
        StalenessDetectorEngine,
    )
except ImportError:
    StalenessDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.refresh_predictor import (
        RefreshPredictorEngine,
    )
except ImportError:
    RefreshPredictorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.alert_manager import (
        AlertManagerEngine,
    )
except ImportError:
    AlertManagerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_freshness_monitor.freshness_pipeline import (
        FreshnessMonitorPipelineEngine,
    )
except ImportError:
    FreshnessMonitorPipelineEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from data_quality_profiler.timeliness_tracker (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_quality_profiler.timeliness_tracker import (
        TimelinessTracker as L1TimelinessTracker,
    )
    TimelinessTracker = L1TimelinessTracker
except ImportError:
    TimelinessTracker = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from cross_source_reconciliation.source_registry (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.cross_source_reconciliation.source_registry import (
        SourceRegistryEngine as L1SourceRegistryEngine,
    )
    L1SourceRegistry = L1SourceRegistryEngine
except ImportError:
    L1SourceRegistry = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade (7 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_freshness_monitor.setup import (
        DataFreshnessMonitorService,
        configure_freshness_monitor,
        get_freshness_monitor,
        get_router,
        get_service,
        set_service,
        reset_service,
    )
except ImportError:
    DataFreshnessMonitorService = None  # type: ignore[assignment, misc]
    configure_freshness_monitor = None  # type: ignore[assignment, misc]
    get_freshness_monitor = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]
    get_service = None  # type: ignore[assignment, misc]
    set_service = None  # type: ignore[assignment, misc]
    reset_service = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Router (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_freshness_monitor.api.router import router
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
    # Version and identity (4)
    # -------------------------------------------------------------------------
    "__version__",
    "__agent_id__",
    "__agent_name__",
    "VERSION",
    # -------------------------------------------------------------------------
    # SDK flag (1)
    # -------------------------------------------------------------------------
    "DATA_FRESHNESS_MONITOR_SDK_AVAILABLE",
    # -------------------------------------------------------------------------
    # Configuration (4)
    # -------------------------------------------------------------------------
    "DataFreshnessMonitorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -------------------------------------------------------------------------
    # Provenance (3)
    # -------------------------------------------------------------------------
    "ProvenanceTracker",
    "ProvenanceEntry",
    "get_provenance_tracker",
    # -------------------------------------------------------------------------
    # Metrics flag (1)
    # -------------------------------------------------------------------------
    "PROMETHEUS_AVAILABLE",
    # -------------------------------------------------------------------------
    # Metric objects (12)
    # -------------------------------------------------------------------------
    "dfm_checks_performed_total",
    "dfm_sla_breaches_total",
    "dfm_alerts_sent_total",
    "dfm_datasets_registered_total",
    "dfm_refresh_events_total",
    "dfm_predictions_made_total",
    "dfm_freshness_score",
    "dfm_data_age_hours",
    "dfm_processing_duration_seconds",
    "dfm_active_breaches",
    "dfm_monitored_datasets",
    "dfm_processing_errors_total",
    # -------------------------------------------------------------------------
    # Metric helper functions (12)
    # -------------------------------------------------------------------------
    "record_check",
    "record_breach",
    "record_alert",
    "record_dataset_registered",
    "record_refresh_event",
    "record_prediction",
    "observe_freshness_score",
    "observe_data_age",
    "observe_duration",
    "set_active_breaches",
    "set_monitored_datasets",
    "record_error",
    # -------------------------------------------------------------------------
    # Models - Enumerations (10)
    # -------------------------------------------------------------------------
    "FreshnessTier",
    "CheckStatus",
    "AlertChannel",
    "AlertStatus",
    "AlertSeverity",
    "BreachStatus",
    "BreachSeverity",
    "SLAStatus",
    "PredictionConfidence",
    "PatternType",
    # -------------------------------------------------------------------------
    # Models - Data classes (10)
    # -------------------------------------------------------------------------
    "EscalationLevel",
    "EscalationPolicy",
    "DatasetRegistration",
    "FreshnessCheck",
    "SLADefinition",
    "SLABreach",
    "FreshnessAlert",
    "StalenessPrediction",
    "FreshnessPattern",
    "DataFreshnessMonitorStatistics",
    # -------------------------------------------------------------------------
    # Core engines (Layer 2) (7)
    # -------------------------------------------------------------------------
    "DatasetRegistryEngine",
    "SLADefinitionEngine",
    "FreshnessCheckerEngine",
    "StalenessDetectorEngine",
    "RefreshPredictorEngine",
    "AlertManagerEngine",
    "FreshnessMonitorPipelineEngine",
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (2)
    # -------------------------------------------------------------------------
    "TimelinessTracker",
    "L1SourceRegistry",
    # -------------------------------------------------------------------------
    # Service setup facade (7)
    # -------------------------------------------------------------------------
    "DataFreshnessMonitorService",
    "configure_freshness_monitor",
    "get_freshness_monitor",
    "get_router",
    "get_service",
    "set_service",
    "reset_service",
    # -------------------------------------------------------------------------
    # Router (1)
    # -------------------------------------------------------------------------
    "router",
]
