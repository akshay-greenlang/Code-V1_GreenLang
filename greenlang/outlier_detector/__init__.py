# -*- coding: utf-8 -*-
"""
GL-DATA-X-016: GreenLang Outlier Detection Agent SDK
=====================================================

This package provides intelligent outlier detection for GreenLang
sustainability datasets. It supports:

- Statistical detection (IQR, z-score, modified z-score, MAD, Grubbs,
  Tukey fences, percentile-based)
- Contextual detection (group-based, peer comparison, conditional)
- Temporal detection (CUSUM, trend break, seasonal residual, moving
  window, EWMA control chart)
- Multivariate detection (Mahalanobis distance, Isolation Forest, LOF,
  DBSCAN)
- Ensemble combination (weighted average, majority vote, max score,
  mean score)
- Outlier classification (error, genuine extreme, data entry, regime
  change, sensor fault)
- Treatment strategies (cap, winsorize, flag, remove, replace,
  investigate) with undo support
- End-to-end pipeline orchestration (detect, classify, treat, validate,
  document)
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability

Key Components:
    - config: OutlierDetectorConfig with GL_OD_ env prefix
    - statistical_detector: Statistical outlier detection engine
    - contextual_detector: Context-aware group-based detection engine
    - temporal_detector: Time-series anomaly detection engine
    - multivariate_detector: Multivariate detection engine
    - outlier_classifier: Outlier classification engine
    - treatment_engine: Outlier treatment engine
    - outlier_pipeline: End-to-end pipeline engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - setup: Service facade and FastAPI integration

Example:
    >>> from greenlang.outlier_detector import OutlierDetectorService
    >>> service = OutlierDetectorService()
    >>> result = service.detect_outliers(
    ...     records=[{"val": 1}, {"val": 2}, {"val": 3}, {"val": 100}],
    ...     column="val",
    ... )
    >>> print(result.outliers_found)

Agent ID: GL-DATA-X-016
Agent Name: Outlier Detection Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-016"
__agent_name__ = "Outlier Detection Agent"

# SDK availability flag
OUTLIER_DETECTOR_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration (4 items)
# ---------------------------------------------------------------------------
from greenlang.outlier_detector.config import (
    OutlierDetectorConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance (1 item)
# ---------------------------------------------------------------------------
from greenlang.outlier_detector.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics (25 items)
# ---------------------------------------------------------------------------
from greenlang.outlier_detector.metrics import (
    PROMETHEUS_AVAILABLE,
    od_jobs_processed_total,
    od_outliers_detected_total,
    od_outliers_classified_total,
    od_treatments_applied_total,
    od_thresholds_evaluated_total,
    od_feedback_received_total,
    od_processing_errors_total,
    od_ensemble_score,
    od_processing_duration_seconds,
    od_detection_confidence,
    od_active_jobs,
    od_total_outliers_flagged,
    inc_jobs,
    inc_outliers_detected,
    inc_outliers_classified,
    inc_treatments,
    inc_thresholds,
    inc_feedback,
    inc_errors,
    observe_ensemble_score,
    observe_duration,
    observe_confidence,
    set_active_jobs,
    set_total_outliers_flagged,
)

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback (7)
# ---------------------------------------------------------------------------
try:
    from greenlang.outlier_detector.statistical_detector import (
        StatisticalDetectorEngine,
    )
except ImportError:
    StatisticalDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.outlier_detector.contextual_detector import (
        ContextualDetectorEngine,
    )
except ImportError:
    ContextualDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.outlier_detector.temporal_detector import (
        TemporalDetectorEngine,
    )
except ImportError:
    TemporalDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.outlier_detector.multivariate_detector import (
        MultivariateDetectorEngine,
    )
except ImportError:
    MultivariateDetectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.outlier_detector.outlier_classifier import (
        OutlierClassifierEngine,
    )
except ImportError:
    OutlierClassifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.outlier_detector.treatment_engine import TreatmentEngine
except ImportError:
    TreatmentEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.outlier_detector.outlier_pipeline import (
        OutlierPipelineEngine,
    )
except ImportError:
    OutlierPipelineEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from data_quality_profiler (3 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.data_quality_profiler.models import (
        QualityDimension as L1QualityDimension,
    )
    QualityDimension = L1QualityDimension
except ImportError:
    QualityDimension = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.anomaly_detector import (
        AnomalyDetector as L1AnomalyDetector,
        AnomalyResult as L1AnomalyResult,
    )
    AnomalyDetector = L1AnomalyDetector
    AnomalyResult = L1AnomalyResult
except ImportError:
    AnomalyDetector = None  # type: ignore[assignment, misc]
    AnomalyResult = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Enumerations (13 items)
# ---------------------------------------------------------------------------
from greenlang.outlier_detector.models import (
    DetectionMethod,
    OutlierClass,
    TreatmentStrategy,
    OutlierStatus,
    EnsembleMethod,
    ContextType,
    TemporalMethod,
    SeverityLevel,
    ReportFormat,
    PipelineStage,
    ThresholdSource,
    FeedbackType,
    DataColumnType,
)

# ---------------------------------------------------------------------------
# SDK data models (20 items)
# ---------------------------------------------------------------------------
from greenlang.outlier_detector.models import (
    OutlierScore,
    DetectionResult,
    ContextualResult,
    TemporalResult,
    MultivariateResult,
    EnsembleResult,
    OutlierClassification,
    TreatmentResult,
    TreatmentRecord,
    DomainThreshold,
    FeedbackEntry,
    ImpactAnalysis,
    OutlierReport,
    PipelineConfig,
    PipelineResult,
    DetectionJobConfig,
    OutlierStatistics,
    ThresholdConfig,
    ColumnOutlierSummary,
    BatchDetectionResult,
)

# ---------------------------------------------------------------------------
# Request models (8 items)
# ---------------------------------------------------------------------------
from greenlang.outlier_detector.models import (
    CreateDetectionJobRequest,
    DetectOutliersRequest,
    ClassifyOutliersRequest,
    TreatOutliersRequest,
    SubmitFeedbackRequest,
    RunPipelineRequest,
    BatchDetectRequest,
    ConfigureThresholdsRequest,
)

# ---------------------------------------------------------------------------
# Service setup facade (4 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.outlier_detector.setup import (
        OutlierDetectorService,
        configure_outlier_detector,
        get_outlier_detector,
        get_router,
    )
except ImportError:
    OutlierDetectorService = None  # type: ignore[assignment, misc]
    configure_outlier_detector = None  # type: ignore[assignment, misc]
    get_outlier_detector = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Response models (8 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.outlier_detector.setup import (
        DetectionResponse,
        BatchDetectionResponse,
        ClassificationResponse,
        TreatmentResponse,
        ThresholdResponse,
        FeedbackResponse,
        PipelineResponse,
        StatsResponse,
    )
except ImportError:
    DetectionResponse = None  # type: ignore[assignment, misc]
    BatchDetectionResponse = None  # type: ignore[assignment, misc]
    ClassificationResponse = None  # type: ignore[assignment, misc]
    TreatmentResponse = None  # type: ignore[assignment, misc]
    ThresholdResponse = None  # type: ignore[assignment, misc]
    FeedbackResponse = None  # type: ignore[assignment, misc]
    PipelineResponse = None  # type: ignore[assignment, misc]
    StatsResponse = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Router (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.outlier_detector.api.router import router
except ImportError:
    router = None  # type: ignore[assignment]


__all__ = [
    # -------------------------------------------------------------------------
    # Version and identity (3)
    # -------------------------------------------------------------------------
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # -------------------------------------------------------------------------
    # SDK flag (1)
    # -------------------------------------------------------------------------
    "OUTLIER_DETECTOR_SDK_AVAILABLE",
    # -------------------------------------------------------------------------
    # Configuration (4)
    # -------------------------------------------------------------------------
    "OutlierDetectorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -------------------------------------------------------------------------
    # Provenance (1)
    # -------------------------------------------------------------------------
    "ProvenanceTracker",
    # -------------------------------------------------------------------------
    # Metrics flag (1)
    # -------------------------------------------------------------------------
    "PROMETHEUS_AVAILABLE",
    # -------------------------------------------------------------------------
    # Metric objects (12)
    # -------------------------------------------------------------------------
    "od_jobs_processed_total",
    "od_outliers_detected_total",
    "od_outliers_classified_total",
    "od_treatments_applied_total",
    "od_thresholds_evaluated_total",
    "od_feedback_received_total",
    "od_processing_errors_total",
    "od_ensemble_score",
    "od_processing_duration_seconds",
    "od_detection_confidence",
    "od_active_jobs",
    "od_total_outliers_flagged",
    # -------------------------------------------------------------------------
    # Metric helper functions (12)
    # -------------------------------------------------------------------------
    "inc_jobs",
    "inc_outliers_detected",
    "inc_outliers_classified",
    "inc_treatments",
    "inc_thresholds",
    "inc_feedback",
    "inc_errors",
    "observe_ensemble_score",
    "observe_duration",
    "observe_confidence",
    "set_active_jobs",
    "set_total_outliers_flagged",
    # -------------------------------------------------------------------------
    # Core engines (Layer 2) (7)
    # -------------------------------------------------------------------------
    "StatisticalDetectorEngine",
    "ContextualDetectorEngine",
    "TemporalDetectorEngine",
    "MultivariateDetectorEngine",
    "OutlierClassifierEngine",
    "TreatmentEngine",
    "OutlierPipelineEngine",
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (3)
    # -------------------------------------------------------------------------
    "QualityDimension",
    "AnomalyDetector",
    "AnomalyResult",
    # -------------------------------------------------------------------------
    # Enumerations (13)
    # -------------------------------------------------------------------------
    "DetectionMethod",
    "OutlierClass",
    "TreatmentStrategy",
    "OutlierStatus",
    "EnsembleMethod",
    "ContextType",
    "TemporalMethod",
    "SeverityLevel",
    "ReportFormat",
    "PipelineStage",
    "ThresholdSource",
    "FeedbackType",
    "DataColumnType",
    # -------------------------------------------------------------------------
    # SDK data models (20)
    # -------------------------------------------------------------------------
    "OutlierScore",
    "DetectionResult",
    "ContextualResult",
    "TemporalResult",
    "MultivariateResult",
    "EnsembleResult",
    "OutlierClassification",
    "TreatmentResult",
    "TreatmentRecord",
    "DomainThreshold",
    "FeedbackEntry",
    "ImpactAnalysis",
    "OutlierReport",
    "PipelineConfig",
    "PipelineResult",
    "DetectionJobConfig",
    "OutlierStatistics",
    "ThresholdConfig",
    "ColumnOutlierSummary",
    "BatchDetectionResult",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "CreateDetectionJobRequest",
    "DetectOutliersRequest",
    "ClassifyOutliersRequest",
    "TreatOutliersRequest",
    "SubmitFeedbackRequest",
    "RunPipelineRequest",
    "BatchDetectRequest",
    "ConfigureThresholdsRequest",
    # -------------------------------------------------------------------------
    # Service setup facade (4)
    # -------------------------------------------------------------------------
    "OutlierDetectorService",
    "configure_outlier_detector",
    "get_outlier_detector",
    "get_router",
    # -------------------------------------------------------------------------
    # Response models (8)
    # -------------------------------------------------------------------------
    "DetectionResponse",
    "BatchDetectionResponse",
    "ClassificationResponse",
    "TreatmentResponse",
    "ThresholdResponse",
    "FeedbackResponse",
    "PipelineResponse",
    "StatsResponse",
    # -------------------------------------------------------------------------
    # Router (1)
    # -------------------------------------------------------------------------
    "router",
]
