# -*- coding: utf-8 -*-
"""
Deforestation Alert System Agent - AGENT-EUDR-020

Production-grade real-time satellite monitoring and alerting system for
deforestation events affecting EUDR supply chain plots. Integrates five
satellite data sources -- Sentinel-2 (10m resolution, 5-day revisit),
Landsat 8/9 (30m resolution, 8-day revisit), GLAD alerts (University of
Maryland weekly Landsat-based deforestation alerts), Hansen Global Forest
Change (annual tree cover loss from Landsat time series), and RADD
(Radar Alerts for Detecting Deforestation from Sentinel-1 SAR) -- to
provide multi-source change detection with configurable confidence
thresholds and spatial resolution fusion.

The agent performs continuous monitoring of deforestation activity near
EUDR-regulated commodity production plots through spectral change
detection (NDVI, EVI, NBR, NDMI, SAVI indices), spatial buffer zone
monitoring with configurable radii (1-50 km), EUDR cutoff date
verification (31 December 2020 per Article 2(1)), historical baseline
comparison (2018-2020 reference period), automated alert workflow
management with SLA tracking, and compliance impact assessment mapping
deforestation events to supply chain disruption and market restrictions.

This package provides a complete deforestation alert system for EUDR
regulatory compliance per EU 2023/1115 Articles 2, 9, 10, 11, and 31:

    Capabilities:
        - Multi-source satellite change detection combining Sentinel-2,
          Landsat, GLAD, Hansen GFC, and RADD alert sources with
          configurable confidence thresholds and cloud cover filtering
        - Alert generation with deduplication, batch processing, and
          real-time streaming for near-real-time deforestation event
          notification with configurable retention (5 years per Art. 31)
        - Five-tier severity classification using weighted scoring across
          area (0.25), deforestation rate (0.20), proximity (0.25),
          protected area overlay (0.15), and post-cutoff timing (0.15)
          with severity levels CRITICAL/HIGH/MEDIUM/LOW/INFORMATIONAL
        - Spatial buffer monitoring with circular, polygon, and adaptive
          buffer geometries supporting 1-50 km radii at 64-point
          resolution for proximity detection to supply chain plots
        - EUDR cutoff date verification (31 December 2020) with multi-
          source temporal evidence, pre/post-cutoff classification,
          90-day grace period handling, and 0.85 confidence threshold
        - Historical baseline comparison using 2018-2020 reference
          period canopy cover and forest area measurements with minimum
          3-sample requirement and 10% canopy cover threshold
        - Alert workflow management with auto-triage, investigation,
          resolution, and escalation states; configurable SLA deadlines
          (triage 4h, investigation 48h, resolution 168h); and up to
          3 escalation levels
        - Compliance impact assessment mapping deforestation alerts to
          affected suppliers, products, market restrictions, remediation
          actions, and estimated financial impact with auto-assessment
          and market restriction thresholds

    Foundational modules:
        - config: DeforestationAlertSystemConfig with GL_EUDR_DAS_
          env var support (60+ settings covering satellite sources,
          change detection, alert generation, severity weights, spatial
          buffers, cutoff dates, baselines, workflow SLAs, compliance,
          and rate limiting)
        - models: Pydantic v2 data models with 12 enumerations,
          12+ core models, 8 request models, and 8 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 12 actions for full traceability
        - metrics: 20 Prometheus self-monitoring metrics (gl_eudr_das_)
          covering satellite detection, alert generation, severity
          classification, buffer monitoring, cutoff verification,
          workflow transitions, and compliance assessments

PRD: PRD-AGENT-EUDR-020
Agent ID: GL-EUDR-DAS-020
Regulation: EU 2023/1115 (EUDR) Articles 2, 9, 10, 11, 31
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.deforestation_alert_system import (
    ...     SatelliteSource,
    ...     AlertSeverity,
    ...     ChangeType,
    ...     DeforestationAlertSystemConfig,
    ...     get_config,
    ... )
    >>> from decimal import Decimal
    >>> cfg = get_config()
    >>> print(cfg.ndvi_change_threshold, cfg.cutoff_date)
    -0.15 2020-12-31

    >>> from greenlang.agents.eudr.deforestation_alert_system import (
    ...     SatelliteDetection,
    ...     DeforestationAlert,
    ...     SeverityScore,
    ... )
    >>> detection = SatelliteDetection(
    ...     detection_id="det-001",
    ...     source=SatelliteSource.SENTINEL2,
    ...     latitude=Decimal("-3.1234"),
    ...     longitude=Decimal("28.5678"),
    ...     area_ha=Decimal("12.5"),
    ...     change_type=ChangeType.DEFORESTATION,
    ...     confidence=Decimal("0.92"),
    ... )
    >>> print(detection.source, detection.change_type)
    SatelliteSource.SENTINEL2 ChangeType.DEFORESTATION

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-DAS-020"

# ---------------------------------------------------------------------------
# Foundational imports: config
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.config import (
        DeforestationAlertSystemConfig,
        get_config,
        reset_config,
        set_config,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import config module: {e}")
    DeforestationAlertSystemConfig = None  # type: ignore[misc,assignment]
    get_config = None  # type: ignore[misc,assignment]
    reset_config = None  # type: ignore[misc,assignment]
    set_config = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: models
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.models import (
        # Enumerations (12)
        SatelliteSource,
        ChangeType,
        AlertSeverity,
        AlertStatus,
        BufferType,
        CutoffResult,
        ComplianceOutcome,
        WorkflowAction,
        EUDRCommodity,
        SpectralIndex,
        EvidenceQuality,
        RemediationAction,
        # Core Models (12)
        SatelliteDetection,
        DeforestationAlert,
        SeverityScore,
        SpatialBuffer,
        BufferViolation,
        CutoffVerification,
        HistoricalBaseline,
        BaselineComparison,
        WorkflowState,
        WorkflowTransition,
        ComplianceImpact,
        AuditLogEntry,
        # Request Models (8)
        DetectChangesRequest,
        GenerateAlertsRequest,
        ClassifySeverityRequest,
        CheckBufferRequest,
        VerifyCutoffRequest,
        CompareBaselineRequest,
        TransitionWorkflowRequest,
        AssessComplianceRequest,
        # Response Models (8)
        DetectChangesResponse,
        GenerateAlertsResponse,
        ClassifySeverityResponse,
        CheckBufferResponse,
        VerifyCutoffResponse,
        CompareBaselineResponse,
        TransitionWorkflowResponse,
        AssessComplianceResponse,
        # Constants
        VERSION,
        EUDR_CUTOFF_DATE,
        MAX_BUFFER_RADIUS_KM,
        MIN_BUFFER_RADIUS_KM,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        SUPPORTED_SATELLITE_SOURCES,
        SUPPORTED_SPECTRAL_INDICES,
        SUPPORTED_COMMODITIES,
        DEFAULT_BUFFER_RESOLUTION,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import models module: {e}")
    # Enumerations (12)
    SatelliteSource = None  # type: ignore[misc,assignment]
    ChangeType = None  # type: ignore[misc,assignment]
    AlertSeverity = None  # type: ignore[misc,assignment]
    AlertStatus = None  # type: ignore[misc,assignment]
    BufferType = None  # type: ignore[misc,assignment]
    CutoffResult = None  # type: ignore[misc,assignment]
    ComplianceOutcome = None  # type: ignore[misc,assignment]
    WorkflowAction = None  # type: ignore[misc,assignment]
    EUDRCommodity = None  # type: ignore[misc,assignment]
    SpectralIndex = None  # type: ignore[misc,assignment]
    EvidenceQuality = None  # type: ignore[misc,assignment]
    RemediationAction = None  # type: ignore[misc,assignment]
    # Core Models (12)
    SatelliteDetection = None  # type: ignore[misc,assignment]
    DeforestationAlert = None  # type: ignore[misc,assignment]
    SeverityScore = None  # type: ignore[misc,assignment]
    SpatialBuffer = None  # type: ignore[misc,assignment]
    BufferViolation = None  # type: ignore[misc,assignment]
    CutoffVerification = None  # type: ignore[misc,assignment]
    HistoricalBaseline = None  # type: ignore[misc,assignment]
    BaselineComparison = None  # type: ignore[misc,assignment]
    WorkflowState = None  # type: ignore[misc,assignment]
    WorkflowTransition = None  # type: ignore[misc,assignment]
    ComplianceImpact = None  # type: ignore[misc,assignment]
    AuditLogEntry = None  # type: ignore[misc,assignment]
    # Request Models (8)
    DetectChangesRequest = None  # type: ignore[misc,assignment]
    GenerateAlertsRequest = None  # type: ignore[misc,assignment]
    ClassifySeverityRequest = None  # type: ignore[misc,assignment]
    CheckBufferRequest = None  # type: ignore[misc,assignment]
    VerifyCutoffRequest = None  # type: ignore[misc,assignment]
    CompareBaselineRequest = None  # type: ignore[misc,assignment]
    TransitionWorkflowRequest = None  # type: ignore[misc,assignment]
    AssessComplianceRequest = None  # type: ignore[misc,assignment]
    # Response Models (8)
    DetectChangesResponse = None  # type: ignore[misc,assignment]
    GenerateAlertsResponse = None  # type: ignore[misc,assignment]
    ClassifySeverityResponse = None  # type: ignore[misc,assignment]
    CheckBufferResponse = None  # type: ignore[misc,assignment]
    VerifyCutoffResponse = None  # type: ignore[misc,assignment]
    CompareBaselineResponse = None  # type: ignore[misc,assignment]
    TransitionWorkflowResponse = None  # type: ignore[misc,assignment]
    AssessComplianceResponse = None  # type: ignore[misc,assignment]
    # Constants
    VERSION = None  # type: ignore[misc,assignment]
    EUDR_CUTOFF_DATE = None  # type: ignore[misc,assignment]
    MAX_BUFFER_RADIUS_KM = None  # type: ignore[misc,assignment]
    MIN_BUFFER_RADIUS_KM = None  # type: ignore[misc,assignment]
    MAX_BATCH_SIZE = None  # type: ignore[misc,assignment]
    EUDR_RETENTION_YEARS = None  # type: ignore[misc,assignment]
    SUPPORTED_SATELLITE_SOURCES = None  # type: ignore[misc,assignment]
    SUPPORTED_SPECTRAL_INDICES = None  # type: ignore[misc,assignment]
    SUPPORTED_COMMODITIES = None  # type: ignore[misc,assignment]
    DEFAULT_BUFFER_RESOLUTION = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_tracker,
        reset_tracker,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import provenance module: {e}")
    ProvenanceRecord = None  # type: ignore[misc,assignment]
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_tracker = None  # type: ignore[misc,assignment]
    reset_tracker = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.metrics import (
        PROMETHEUS_AVAILABLE,
        # Counter helpers (10)
        record_satellite_detection,
        record_alert_generated,
        record_severity_classification,
        record_buffer_check,
        record_cutoff_verification,
        record_baseline_comparison,
        record_workflow_transition,
        record_compliance_assessment,
        record_false_positive,
        record_api_error,
        # Histogram helpers (4)
        observe_detection_latency,
        observe_alert_generation_duration,
        observe_severity_scoring_duration,
        observe_compliance_assessment_duration,
        # Gauge helpers (6)
        set_active_alerts,
        set_monitored_plots,
        set_active_buffers,
        set_pending_investigations,
        set_sla_breaches,
        set_detection_backlog,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import metrics module: {e}")
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    # Counter helpers (10)
    record_satellite_detection = None  # type: ignore[misc,assignment]
    record_alert_generated = None  # type: ignore[misc,assignment]
    record_severity_classification = None  # type: ignore[misc,assignment]
    record_buffer_check = None  # type: ignore[misc,assignment]
    record_cutoff_verification = None  # type: ignore[misc,assignment]
    record_baseline_comparison = None  # type: ignore[misc,assignment]
    record_workflow_transition = None  # type: ignore[misc,assignment]
    record_compliance_assessment = None  # type: ignore[misc,assignment]
    record_false_positive = None  # type: ignore[misc,assignment]
    record_api_error = None  # type: ignore[misc,assignment]
    # Histogram helpers (4)
    observe_detection_latency = None  # type: ignore[misc,assignment]
    observe_alert_generation_duration = None  # type: ignore[misc,assignment]
    observe_severity_scoring_duration = None  # type: ignore[misc,assignment]
    observe_compliance_assessment_duration = None  # type: ignore[misc,assignment]
    # Gauge helpers (6)
    set_active_alerts = None  # type: ignore[misc,assignment]
    set_monitored_plots = None  # type: ignore[misc,assignment]
    set_active_buffers = None  # type: ignore[misc,assignment]
    set_pending_investigations = None  # type: ignore[misc,assignment]
    set_sla_breaches = None  # type: ignore[misc,assignment]
    set_detection_backlog = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional - engines may not exist yet)
# ---------------------------------------------------------------------------

# ---- Engine 1: Satellite Change Detector ----
try:
    from greenlang.agents.eudr.deforestation_alert_system.satellite_change_detector import (
        SatelliteChangeDetector,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.engines.satellite_change_detector import (
            SatelliteChangeDetector,
        )
    except ImportError:
        SatelliteChangeDetector = None  # type: ignore[misc,assignment]

# ---- Engine 2: Alert Generator ----
try:
    from greenlang.agents.eudr.deforestation_alert_system.alert_generator import (
        AlertGenerator,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.engines.alert_generator import (
            AlertGenerator,
        )
    except ImportError:
        AlertGenerator = None  # type: ignore[misc,assignment]

# ---- Engine 3: Severity Classifier ----
try:
    from greenlang.agents.eudr.deforestation_alert_system.severity_classifier import (
        SeverityClassifier,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.engines.severity_classifier import (
            SeverityClassifier,
        )
    except ImportError:
        SeverityClassifier = None  # type: ignore[misc,assignment]

# ---- Engine 4: Spatial Buffer Monitor ----
try:
    from greenlang.agents.eudr.deforestation_alert_system.spatial_buffer_monitor import (
        SpatialBufferMonitor,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.engines.spatial_buffer_monitor import (
            SpatialBufferMonitor,
        )
    except ImportError:
        SpatialBufferMonitor = None  # type: ignore[misc,assignment]

# ---- Engine 5: Cutoff Date Verifier ----
try:
    from greenlang.agents.eudr.deforestation_alert_system.cutoff_date_verifier import (
        CutoffDateVerifier,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.engines.cutoff_date_verifier import (
            CutoffDateVerifier,
        )
    except ImportError:
        CutoffDateVerifier = None  # type: ignore[misc,assignment]

# ---- Engine 6: Historical Baseline Engine ----
try:
    from greenlang.agents.eudr.deforestation_alert_system.historical_baseline_engine import (
        HistoricalBaselineEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.engines.historical_baseline_engine import (
            HistoricalBaselineEngine,
        )
    except ImportError:
        HistoricalBaselineEngine = None  # type: ignore[misc,assignment]

# ---- Engine 7: Alert Workflow Engine ----
try:
    from greenlang.agents.eudr.deforestation_alert_system.alert_workflow_engine import (
        AlertWorkflowEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.engines.alert_workflow_engine import (
            AlertWorkflowEngine,
        )
    except ImportError:
        AlertWorkflowEngine = None  # type: ignore[misc,assignment]

# ---- Engine 8: Compliance Impact Assessor ----
try:
    from greenlang.agents.eudr.deforestation_alert_system.compliance_impact_assessor import (
        ComplianceImpactAssessor,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.engines.compliance_impact_assessor import (
            ComplianceImpactAssessor,
        )
    except ImportError:
        ComplianceImpactAssessor = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Service facade import (conditional - service may not exist yet)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.setup import (
        DeforestationAlertSystemSetup,
    )
except ImportError:
    DeforestationAlertSystemSetup = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Config --
    "DeforestationAlertSystemConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Enumerations (12) --
    "SatelliteSource",
    "ChangeType",
    "AlertSeverity",
    "AlertStatus",
    "BufferType",
    "CutoffResult",
    "ComplianceOutcome",
    "WorkflowAction",
    "EUDRCommodity",
    "SpectralIndex",
    "EvidenceQuality",
    "RemediationAction",
    # -- Core Models (12) --
    "SatelliteDetection",
    "DeforestationAlert",
    "SeverityScore",
    "SpatialBuffer",
    "BufferViolation",
    "CutoffVerification",
    "HistoricalBaseline",
    "BaselineComparison",
    "WorkflowState",
    "WorkflowTransition",
    "ComplianceImpact",
    "AuditLogEntry",
    # -- Request Models (8) --
    "DetectChangesRequest",
    "GenerateAlertsRequest",
    "ClassifySeverityRequest",
    "CheckBufferRequest",
    "VerifyCutoffRequest",
    "CompareBaselineRequest",
    "TransitionWorkflowRequest",
    "AssessComplianceRequest",
    # -- Response Models (8) --
    "DetectChangesResponse",
    "GenerateAlertsResponse",
    "ClassifySeverityResponse",
    "CheckBufferResponse",
    "VerifyCutoffResponse",
    "CompareBaselineResponse",
    "TransitionWorkflowResponse",
    "AssessComplianceResponse",
    # -- Constants --
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BUFFER_RADIUS_KM",
    "MIN_BUFFER_RADIUS_KM",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "SUPPORTED_SATELLITE_SOURCES",
    "SUPPORTED_SPECTRAL_INDICES",
    "SUPPORTED_COMMODITIES",
    "DEFAULT_BUFFER_RESOLUTION",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_tracker",
    "reset_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_satellite_detection",
    "record_alert_generated",
    "record_severity_classification",
    "record_buffer_check",
    "record_cutoff_verification",
    "record_baseline_comparison",
    "record_workflow_transition",
    "record_compliance_assessment",
    "record_false_positive",
    "record_api_error",
    "observe_detection_latency",
    "observe_alert_generation_duration",
    "observe_severity_scoring_duration",
    "observe_compliance_assessment_duration",
    "set_active_alerts",
    "set_monitored_plots",
    "set_active_buffers",
    "set_pending_investigations",
    "set_sla_breaches",
    "set_detection_backlog",
    # -- Engines (8) --
    "SatelliteChangeDetector",
    "AlertGenerator",
    "SeverityClassifier",
    "SpatialBufferMonitor",
    "CutoffDateVerifier",
    "HistoricalBaselineEngine",
    "AlertWorkflowEngine",
    "ComplianceImpactAssessor",
    # -- Setup Facade --
    "DeforestationAlertSystemSetup",
]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string.

    Returns:
        Version string in semver format (e.g. "1.0.0").

    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata.

    Returns:
        Dictionary with agent_id, version, regulation references,
        engine listing, satellite sources, and model counts for
        the Deforestation Alert System agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-DAS-020'
        >>> info["engine_count"]
        8
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Deforestation Alert System",
        "prd": "PRD-AGENT-EUDR-020",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["2", "9", "10", "11", "31"],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "satellite_sources": [
            "Sentinel-2 (10m, 5-day revisit)",
            "Landsat 8/9 (30m, 8-day revisit)",
            "GLAD (University of Maryland weekly alerts)",
            "Hansen GFC (annual tree cover loss)",
            "RADD (Sentinel-1 SAR radar alerts)",
        ],
        "spectral_indices": [
            "NDVI (Normalized Difference Vegetation Index)",
            "EVI (Enhanced Vegetation Index)",
            "NBR (Normalized Burn Ratio)",
            "NDMI (Normalized Difference Moisture Index)",
            "SAVI (Soil-Adjusted Vegetation Index)",
        ],
        "eudr_commodities": [
            "cattle",
            "cocoa",
            "coffee",
            "palm_oil",
            "rubber",
            "soya",
            "wood",
        ],
        "cutoff_date": "2020-12-31",
        "engines": [
            "SatelliteChangeDetector",
            "AlertGenerator",
            "SeverityClassifier",
            "SpatialBufferMonitor",
            "CutoffDateVerifier",
            "HistoricalBaselineEngine",
            "AlertWorkflowEngine",
            "ComplianceImpactAssessor",
        ],
        "engine_count": 8,
        "enum_count": 12,
        "core_model_count": 12,
        "request_model_count": 8,
        "response_model_count": 8,
        "metrics_count": 20,
        "db_prefix": "gl_eudr_das_",
        "metrics_prefix": "gl_eudr_das_",
        "env_prefix": "GL_EUDR_DAS_",
    }
