# -*- coding: utf-8 -*-
"""
Outlier Detection Agent Service Data Models - AGENT-DATA-013

Pydantic v2 data models for the Outlier Detection SDK. Re-exports the
Layer 1 models from the Data Quality Profiler's AnomalyDetector,
and defines additional SDK models for outlier detection, classification,
treatment, ensemble scoring, pipeline orchestration, and reporting.

Re-exported Layer 1 sources:
    - greenlang.data_quality_profiler.models: QualityDimension
        (as L1QualityDimension)
    - greenlang.data_quality_profiler.anomaly_detector:
        AnomalyDetector (as L1AnomalyDetector),
        AnomalyResult (as L1AnomalyResult)

New enumerations (13):
    - DetectionMethod, OutlierClass, TreatmentStrategy,
      OutlierStatus, EnsembleMethod, ContextType,
      TemporalMethod, SeverityLevel, ReportFormat,
      PipelineStage, ThresholdSource, FeedbackType,
      DataColumnType

New SDK models (20):
    - OutlierScore, DetectionResult, ContextualResult,
      TemporalResult, MultivariateResult, EnsembleResult,
      OutlierClassification, TreatmentResult, TreatmentRecord,
      DomainThreshold, FeedbackEntry, ImpactAnalysis,
      OutlierReport, PipelineConfig, PipelineResult,
      DetectionJobConfig, OutlierStatistics, ThresholdConfig,
      ColumnOutlierSummary, BatchDetectionResult

Request models (8):
    - CreateDetectionJobRequest, DetectOutliersRequest,
      ClassifyOutliersRequest, TreatOutliersRequest,
      SubmitFeedbackRequest, RunPipelineRequest,
      BatchDetectRequest, ConfigureThresholdsRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from data_quality_profiler
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
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default detection method weights for ensemble scoring.
DEFAULT_METHOD_WEIGHTS: Dict[str, float] = {
    "iqr": 1.0,
    "zscore": 1.0,
    "modified_zscore": 1.2,
    "mad": 1.1,
    "grubbs": 0.8,
    "tukey": 1.0,
    "percentile": 0.7,
    "lof": 1.3,
    "isolation_forest": 1.3,
    "mahalanobis": 1.2,
    "dbscan": 0.9,
}

#: Default severity thresholds by outlier score.
SEVERITY_THRESHOLDS: Dict[str, float] = {
    "critical": 0.95,
    "high": 0.80,
    "medium": 0.60,
    "low": 0.40,
    "info": 0.0,
}

#: Classification confidence thresholds.
CLASSIFICATION_CONFIDENCE: Dict[str, float] = {
    "high": 0.85,
    "medium": 0.65,
    "low": 0.40,
}

#: Pipeline stage execution order.
PIPELINE_STAGE_ORDER: tuple = (
    "detect", "classify", "treat", "validate", "document",
)

#: All supported detection methods.
DETECTION_METHODS: tuple = (
    "iqr", "zscore", "modified_zscore", "mad", "grubbs",
    "tukey", "percentile", "lof", "isolation_forest",
    "mahalanobis", "dbscan", "contextual", "temporal",
)

#: All supported treatment strategies.
TREATMENT_STRATEGIES: tuple = (
    "cap", "winsorize", "flag", "remove", "replace", "investigate",
)

#: Report format options.
REPORT_FORMAT_OPTIONS: tuple = ("json", "csv", "markdown", "html")

#: All supported temporal methods.
TEMPORAL_METHODS: tuple = (
    "cusum", "trend_break", "seasonal_residual",
    "moving_window", "ewma",
)


# =============================================================================
# Enumerations (13)
# =============================================================================


class DetectionMethod(str, Enum):
    """Statistical or algorithmic method for outlier detection.

    IQR: Interquartile Range fences (Q1 - k*IQR, Q3 + k*IQR).
    ZSCORE: Standard z-score threshold.
    MODIFIED_ZSCORE: MAD-based modified z-score (robust to outliers).
    MAD: Median Absolute Deviation threshold.
    GRUBBS: Grubbs test for single maximum or minimum outlier.
    TUKEY: Tukey box-plot fences (inner and outer).
    PERCENTILE: Percentile-based lower/upper bounds.
    LOF: Local Outlier Factor (density-based).
    ISOLATION_FOREST: Isolation Forest (tree-based anomaly detection).
    MAHALANOBIS: Mahalanobis distance from multivariate centroid.
    DBSCAN: Density-Based Spatial Clustering noise points.
    CONTEXTUAL: Context-aware group-based detection.
    TEMPORAL: Time-series anomaly detection.
    """

    IQR = "iqr"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    MAD = "mad"
    GRUBBS = "grubbs"
    TUKEY = "tukey"
    PERCENTILE = "percentile"
    LOF = "lof"
    ISOLATION_FOREST = "isolation_forest"
    MAHALANOBIS = "mahalanobis"
    DBSCAN = "dbscan"
    CONTEXTUAL = "contextual"
    TEMPORAL = "temporal"


class OutlierClass(str, Enum):
    """Classification of an outlier's root cause.

    ERROR: Data processing or transmission error (corrupt value).
    GENUINE_EXTREME: Legitimate extreme value from real phenomenon.
    DATA_ENTRY: Human data entry mistake (transposition, unit error).
    REGIME_CHANGE: Structural change in the underlying process.
    SENSOR_FAULT: Sensor malfunction (stuck, drift, spike).
    """

    ERROR = "error"
    GENUINE_EXTREME = "genuine_extreme"
    DATA_ENTRY = "data_entry"
    REGIME_CHANGE = "regime_change"
    SENSOR_FAULT = "sensor_fault"


class TreatmentStrategy(str, Enum):
    """Strategy for treating detected outliers.

    CAP: Cap values at a percentile threshold.
    WINSORIZE: Replace with the nearest non-outlier percentile value.
    FLAG: Flag for review without modifying data.
    REMOVE: Mark for exclusion from analysis.
    REPLACE: Replace with an imputed value (mean, median, etc.).
    INVESTIGATE: Route to human investigation queue.
    """

    CAP = "cap"
    WINSORIZE = "winsorize"
    FLAG = "flag"
    REMOVE = "remove"
    REPLACE = "replace"
    INVESTIGATE = "investigate"


class OutlierStatus(str, Enum):
    """Lifecycle status of an outlier detection job.

    Tracks the current execution state of a detection pipeline
    from submission through completion, failure, or cancellation.
    """

    PENDING = "pending"
    DETECTING = "detecting"
    CLASSIFYING = "classifying"
    TREATING = "treating"
    COMPLETED = "completed"
    FAILED = "failed"


class EnsembleMethod(str, Enum):
    """Method for combining scores from multiple detectors.

    WEIGHTED_AVERAGE: Weighted average of normalised scores.
    MAJORITY_VOTE: Binary vote from each detector, majority wins.
    MAX_SCORE: Take the maximum score across all detectors.
    MEAN_SCORE: Simple arithmetic mean of all scores.
    """

    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    MAX_SCORE = "max_score"
    MEAN_SCORE = "mean_score"


class ContextType(str, Enum):
    """Type of context for contextual outlier detection.

    FACILITY: Group by facility or site identifier.
    REGION: Group by geographic region.
    SECTOR: Group by industry sector.
    TIME_PERIOD: Group by time period (month, quarter, year).
    PEER_GROUP: Group by peer comparison set.
    CUSTOM: User-defined grouping.
    """

    FACILITY = "facility"
    REGION = "region"
    SECTOR = "sector"
    TIME_PERIOD = "time_period"
    PEER_GROUP = "peer_group"
    CUSTOM = "custom"


class TemporalMethod(str, Enum):
    """Temporal anomaly detection method.

    CUSUM: Cumulative Sum control chart change-point detection.
    TREND_BREAK: Detect sudden breaks in trend direction.
    SEASONAL_RESIDUAL: Outliers in residuals after seasonal decomposition.
    MOVING_WINDOW: Moving window-based anomaly detection.
    EWMA: Exponentially Weighted Moving Average control chart.
    """

    CUSUM = "cusum"
    TREND_BREAK = "trend_break"
    SEASONAL_RESIDUAL = "seasonal_residual"
    MOVING_WINDOW = "moving_window"
    EWMA = "ewma"


class SeverityLevel(str, Enum):
    """Severity classification for a detected outlier.

    CRITICAL: Extreme outlier requiring immediate attention.
    HIGH: Significant outlier that should be investigated.
    MEDIUM: Moderate outlier that may need review.
    LOW: Minor outlier, likely noise.
    INFO: Marginal value, informational only.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReportFormat(str, Enum):
    """Output format for outlier detection reports.

    Defines the serialization format for generated reports
    including detection results, classifications, and treatments.
    """

    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"


class PipelineStage(str, Enum):
    """Stage in the outlier detection pipeline.

    The pipeline executes stages in order: detect, classify,
    treat, validate, document. Each stage produces output
    consumed by the next stage.
    """

    DETECT = "detect"
    CLASSIFY = "classify"
    TREAT = "treat"
    VALIDATE = "validate"
    DOCUMENT = "document"


class ThresholdSource(str, Enum):
    """Source of a detection threshold value.

    DOMAIN: Domain expert-defined threshold.
    STATISTICAL: Statistically derived threshold.
    REGULATORY: Regulatory or compliance threshold.
    CUSTOM: User-defined custom threshold.
    LEARNED: Threshold learned from historical feedback.
    """

    DOMAIN = "domain"
    STATISTICAL = "statistical"
    REGULATORY = "regulatory"
    CUSTOM = "custom"
    LEARNED = "learned"


class FeedbackType(str, Enum):
    """Type of human feedback on an outlier classification.

    CONFIRMED_OUTLIER: Human confirms the point is an outlier.
    FALSE_POSITIVE: Human overrides, the point is not an outlier.
    RECLASSIFIED: Human reclassifies the outlier to a different class.
    UNKNOWN: Human is uncertain and cannot determine classification.
    """

    CONFIRMED_OUTLIER = "confirmed_outlier"
    FALSE_POSITIVE = "false_positive"
    RECLASSIFIED = "reclassified"
    UNKNOWN = "unknown"


class DataColumnType(str, Enum):
    """Data type classification for dataset columns.

    Determines which detection methods are applicable and
    how treatment is performed after detection.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"


# =============================================================================
# SDK Data Models (20)
# =============================================================================


class OutlierScore(BaseModel):
    """Score for a single data point from one detection method.

    Represents the outlier score assigned to a specific value by
    a single detection method, including the normalised score,
    whether it was flagged, and method-specific details.

    Attributes:
        record_index: Row index of the scored record.
        column_name: Column name containing the scored value.
        value: The actual data value that was scored.
        method: Detection method that produced this score.
        score: Normalised outlier score (0.0 = inlier, 1.0 = extreme outlier).
        is_outlier: Whether this point was flagged as an outlier.
        threshold: Threshold used for flagging.
        severity: Severity classification based on the score.
        details: Method-specific details (e.g. z-score value, IQR fences).
        confidence: Confidence in the outlier determination (0.0-1.0).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_index: int = Field(
        ..., ge=0,
        description="Row index of the scored record",
    )
    column_name: str = Field(
        default="",
        description="Column name containing the scored value",
    )
    value: Optional[Any] = Field(
        None, description="The actual data value that was scored",
    )
    method: DetectionMethod = Field(
        ..., description="Detection method that produced this score",
    )
    score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Normalised outlier score (0.0-1.0)",
    )
    is_outlier: bool = Field(
        default=False,
        description="Whether this point was flagged as an outlier",
    )
    threshold: float = Field(
        default=0.0,
        description="Threshold used for flagging",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.INFO,
        description="Severity classification based on the score",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific details",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the outlier determination (0.0-1.0)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class DetectionResult(BaseModel):
    """Result of outlier detection for a single column.

    Contains outlier scores for all data points in one column,
    summary statistics, and the detection method used.

    Attributes:
        result_id: Unique identifier for this detection result.
        column_name: Name of the analyzed column.
        method: Detection method used.
        total_points: Total number of data points analyzed.
        outliers_found: Number of outliers detected.
        outlier_pct: Fraction of points flagged as outliers.
        scores: List of per-point outlier scores.
        lower_fence: Lower outlier fence (if applicable).
        upper_fence: Upper outlier fence (if applicable).
        processing_time_ms: Detection processing time in milliseconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this detection result",
    )
    column_name: str = Field(
        ..., description="Name of the analyzed column",
    )
    method: DetectionMethod = Field(
        ..., description="Detection method used",
    )
    total_points: int = Field(
        default=0, ge=0,
        description="Total number of data points analyzed",
    )
    outliers_found: int = Field(
        default=0, ge=0,
        description="Number of outliers detected",
    )
    outlier_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of points flagged as outliers",
    )
    scores: List[OutlierScore] = Field(
        default_factory=list,
        description="List of per-point outlier scores",
    )
    lower_fence: Optional[float] = Field(
        None, description="Lower outlier fence (if applicable)",
    )
    upper_fence: Optional[float] = Field(
        None, description="Upper outlier fence (if applicable)",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Detection processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("column_name")
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        """Validate column_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("column_name must be non-empty")
        return v


class ContextualResult(BaseModel):
    """Result of contextual (group-based) outlier detection.

    Contains detection results within a specific context group,
    including group statistics and per-point scores.

    Attributes:
        result_id: Unique identifier for this contextual result.
        context_type: Type of context grouping.
        group_key: Group identifier value.
        column_name: Column analyzed for outliers.
        group_size: Number of records in this group.
        group_mean: Mean of the group values.
        group_std: Standard deviation of the group values.
        outliers_found: Number of outliers in this group.
        scores: Per-point outlier scores within the group.
        confidence: Confidence in the contextual detection.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this contextual result",
    )
    context_type: ContextType = Field(
        default=ContextType.CUSTOM,
        description="Type of context grouping",
    )
    group_key: str = Field(
        default="",
        description="Group identifier value",
    )
    column_name: str = Field(
        default="",
        description="Column analyzed for outliers",
    )
    group_size: int = Field(
        default=0, ge=0,
        description="Number of records in this group",
    )
    group_mean: float = Field(
        default=0.0,
        description="Mean of the group values",
    )
    group_std: float = Field(
        default=0.0, ge=0.0,
        description="Standard deviation of the group values",
    )
    outliers_found: int = Field(
        default=0, ge=0,
        description="Number of outliers in this group",
    )
    scores: List[OutlierScore] = Field(
        default_factory=list,
        description="Per-point outlier scores within the group",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the contextual detection",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class TemporalResult(BaseModel):
    """Result of temporal (time-series) outlier detection.

    Contains detection results from a time-series method, including
    detected anomalies and change points.

    Attributes:
        result_id: Unique identifier for this temporal result.
        method: Temporal detection method used.
        column_name: Column analyzed for temporal anomalies.
        series_length: Number of time points in the series.
        anomalies_found: Number of temporal anomalies detected.
        change_points: List of detected change point indices.
        scores: Per-point temporal outlier scores.
        baseline_mean: Mean of the baseline period.
        baseline_std: Standard deviation of the baseline period.
        confidence: Confidence in the temporal detection.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this temporal result",
    )
    method: TemporalMethod = Field(
        ..., description="Temporal detection method used",
    )
    column_name: str = Field(
        default="",
        description="Column analyzed for temporal anomalies",
    )
    series_length: int = Field(
        default=0, ge=0,
        description="Number of time points in the series",
    )
    anomalies_found: int = Field(
        default=0, ge=0,
        description="Number of temporal anomalies detected",
    )
    change_points: List[int] = Field(
        default_factory=list,
        description="List of detected change point indices",
    )
    scores: List[OutlierScore] = Field(
        default_factory=list,
        description="Per-point temporal outlier scores",
    )
    baseline_mean: float = Field(
        default=0.0,
        description="Mean of the baseline period",
    )
    baseline_std: float = Field(
        default=0.0, ge=0.0,
        description="Standard deviation of the baseline period",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the temporal detection",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class MultivariateResult(BaseModel):
    """Result of multivariate outlier detection.

    Contains detection results from multivariate methods operating
    on multiple columns simultaneously.

    Attributes:
        result_id: Unique identifier for this multivariate result.
        method: Multivariate detection method used.
        columns: List of columns used in detection.
        total_points: Total number of data points analyzed.
        outliers_found: Number of multivariate outliers detected.
        scores: Per-point multivariate outlier scores.
        distance_threshold: Distance threshold used for flagging.
        mean_distance: Mean distance of all points.
        max_distance: Maximum distance observed.
        confidence: Confidence in the multivariate detection.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this multivariate result",
    )
    method: DetectionMethod = Field(
        ..., description="Multivariate detection method used",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="List of columns used in detection",
    )
    total_points: int = Field(
        default=0, ge=0,
        description="Total number of data points analyzed",
    )
    outliers_found: int = Field(
        default=0, ge=0,
        description="Number of multivariate outliers detected",
    )
    scores: List[OutlierScore] = Field(
        default_factory=list,
        description="Per-point multivariate outlier scores",
    )
    distance_threshold: float = Field(
        default=0.0, ge=0.0,
        description="Distance threshold used for flagging",
    )
    mean_distance: float = Field(
        default=0.0, ge=0.0,
        description="Mean distance of all points",
    )
    max_distance: float = Field(
        default=0.0, ge=0.0,
        description="Maximum distance observed",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the multivariate detection",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class EnsembleResult(BaseModel):
    """Combined outlier score from multiple detection methods.

    Aggregates scores from multiple detectors into a single
    consensus score using the configured ensemble method.

    Attributes:
        record_index: Row index of the scored record.
        column_name: Column name.
        value: The actual data value.
        ensemble_score: Combined normalised outlier score.
        is_outlier: Whether the ensemble consensus flagged this point.
        method_scores: Per-method scores contributing to the ensemble.
        methods_flagged: Number of methods that flagged this point.
        total_methods: Total number of methods in the ensemble.
        ensemble_method: Ensemble combination method used.
        severity: Severity classification.
        confidence: Confidence in the ensemble result.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_index: int = Field(
        ..., ge=0,
        description="Row index of the scored record",
    )
    column_name: str = Field(
        default="",
        description="Column name",
    )
    value: Optional[Any] = Field(
        None, description="The actual data value",
    )
    ensemble_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Combined normalised outlier score",
    )
    is_outlier: bool = Field(
        default=False,
        description="Whether the ensemble consensus flagged this point",
    )
    method_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-method scores contributing to the ensemble",
    )
    methods_flagged: int = Field(
        default=0, ge=0,
        description="Number of methods that flagged this point",
    )
    total_methods: int = Field(
        default=0, ge=0,
        description="Total number of methods in the ensemble",
    )
    ensemble_method: EnsembleMethod = Field(
        default=EnsembleMethod.WEIGHTED_AVERAGE,
        description="Ensemble combination method used",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.INFO,
        description="Severity classification",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the ensemble result",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class OutlierClassification(BaseModel):
    """Classification of a detected outlier by root cause.

    Assigns a classification (error, genuine extreme, data entry,
    regime change, sensor fault) with confidence scores.

    Attributes:
        classification_id: Unique identifier.
        record_index: Row index of the classified record.
        column_name: Column name.
        value: The outlier value.
        outlier_class: Assigned classification.
        confidence: Confidence in the classification (0.0-1.0).
        class_scores: Per-class confidence scores.
        evidence: Evidence supporting the classification.
        severity: Severity level.
        recommended_treatment: Suggested treatment strategy.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    classification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    record_index: int = Field(
        ..., ge=0,
        description="Row index of the classified record",
    )
    column_name: str = Field(
        default="",
        description="Column name",
    )
    value: Optional[Any] = Field(
        None, description="The outlier value",
    )
    outlier_class: OutlierClass = Field(
        ..., description="Assigned classification",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the classification (0.0-1.0)",
    )
    class_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-class confidence scores",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting the classification",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Severity level",
    )
    recommended_treatment: TreatmentStrategy = Field(
        default=TreatmentStrategy.FLAG,
        description="Suggested treatment strategy",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class TreatmentResult(BaseModel):
    """Result of applying a treatment to an outlier.

    Tracks original and treated values with full provenance.

    Attributes:
        treatment_id: Unique identifier.
        record_index: Row index of the treated record.
        column_name: Column name.
        original_value: Value before treatment.
        treated_value: Value after treatment.
        strategy: Treatment strategy applied.
        reason: Reason for applying this treatment.
        reversible: Whether this treatment can be undone.
        confidence: Confidence in the treatment appropriateness.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    treatment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    record_index: int = Field(
        ..., ge=0,
        description="Row index of the treated record",
    )
    column_name: str = Field(
        default="",
        description="Column name",
    )
    original_value: Optional[Any] = Field(
        None, description="Value before treatment",
    )
    treated_value: Optional[Any] = Field(
        None, description="Value after treatment",
    )
    strategy: TreatmentStrategy = Field(
        ..., description="Treatment strategy applied",
    )
    reason: str = Field(
        default="",
        description="Reason for applying this treatment",
    )
    reversible: bool = Field(
        default=True,
        description="Whether this treatment can be undone",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the treatment appropriateness",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class TreatmentRecord(BaseModel):
    """Persistent record of a treatment for undo/audit.

    Attributes:
        record_id: Unique persistent record identifier.
        treatment_id: Reference to the treatment result.
        job_id: Parent detection job identifier.
        column_name: Column treated.
        record_index: Row index.
        original_value: Original value before treatment.
        treated_value: Value after treatment.
        strategy: Treatment strategy used.
        applied_at: Timestamp of treatment application.
        undone: Whether this treatment has been undone.
        undone_at: Timestamp when undone (if applicable).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique persistent record identifier",
    )
    treatment_id: str = Field(
        default="",
        description="Reference to the treatment result",
    )
    job_id: str = Field(
        default="",
        description="Parent detection job identifier",
    )
    column_name: str = Field(
        default="",
        description="Column treated",
    )
    record_index: int = Field(
        default=0, ge=0,
        description="Row index",
    )
    original_value: Optional[Any] = Field(
        None, description="Original value before treatment",
    )
    treated_value: Optional[Any] = Field(
        None, description="Value after treatment",
    )
    strategy: TreatmentStrategy = Field(
        default=TreatmentStrategy.FLAG,
        description="Treatment strategy used",
    )
    applied_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of treatment application",
    )
    undone: bool = Field(
        default=False,
        description="Whether this treatment has been undone",
    )
    undone_at: Optional[datetime] = Field(
        None, description="Timestamp when undone (if applicable)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class DomainThreshold(BaseModel):
    """Domain-specific threshold for outlier detection.

    Provides domain knowledge bounds for a specific column,
    overriding statistical thresholds.

    Attributes:
        threshold_id: Unique identifier.
        column_name: Column this threshold applies to.
        lower_bound: Lower acceptable bound (values below are outliers).
        upper_bound: Upper acceptable bound (values above are outliers).
        source: Source of this threshold definition.
        description: Human-readable description.
        active: Whether this threshold is currently active.
        created_at: When the threshold was created.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    threshold_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    column_name: str = Field(
        ..., description="Column this threshold applies to",
    )
    lower_bound: Optional[float] = Field(
        None, description="Lower acceptable bound",
    )
    upper_bound: Optional[float] = Field(
        None, description="Upper acceptable bound",
    )
    source: ThresholdSource = Field(
        default=ThresholdSource.DOMAIN,
        description="Source of this threshold definition",
    )
    description: str = Field(
        default="",
        description="Human-readable description",
    )
    active: bool = Field(
        default=True,
        description="Whether this threshold is currently active",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the threshold was created",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("column_name")
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        """Validate column_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("column_name must be non-empty")
        return v


class FeedbackEntry(BaseModel):
    """Human feedback on an outlier detection result.

    Attributes:
        feedback_id: Unique identifier.
        record_index: Row index of the feedback target.
        column_name: Column name.
        feedback_type: Type of feedback provided.
        original_class: Original classification.
        corrected_class: Corrected classification (if reclassified).
        comment: Human comment or justification.
        user_id: User who provided the feedback.
        created_at: When the feedback was submitted.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    feedback_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    record_index: int = Field(
        ..., ge=0,
        description="Row index of the feedback target",
    )
    column_name: str = Field(
        default="",
        description="Column name",
    )
    feedback_type: FeedbackType = Field(
        ..., description="Type of feedback provided",
    )
    original_class: Optional[OutlierClass] = Field(
        None, description="Original classification",
    )
    corrected_class: Optional[OutlierClass] = Field(
        None, description="Corrected classification (if reclassified)",
    )
    comment: str = Field(
        default="",
        description="Human comment or justification",
    )
    user_id: str = Field(
        default="system",
        description="User who provided the feedback",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the feedback was submitted",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class ImpactAnalysis(BaseModel):
    """Impact analysis comparing original and treated datasets.

    Quantifies the statistical effect of outlier treatment on the
    dataset, including changes to mean, standard deviation, and
    distribution shape.

    Attributes:
        analysis_id: Unique identifier.
        column_name: Column analyzed.
        records_affected: Number of records affected by treatment.
        original_mean: Mean before treatment.
        treated_mean: Mean after treatment.
        original_std: Std dev before treatment.
        treated_std: Std dev after treatment.
        original_median: Median before treatment.
        treated_median: Median after treatment.
        mean_change_pct: Percentage change in mean.
        std_change_pct: Percentage change in std dev.
        distribution_shift: Measure of distribution change.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    column_name: str = Field(
        default="",
        description="Column analyzed",
    )
    records_affected: int = Field(
        default=0, ge=0,
        description="Number of records affected by treatment",
    )
    original_mean: float = Field(
        default=0.0, description="Mean before treatment",
    )
    treated_mean: float = Field(
        default=0.0, description="Mean after treatment",
    )
    original_std: float = Field(
        default=0.0, ge=0.0, description="Std dev before treatment",
    )
    treated_std: float = Field(
        default=0.0, ge=0.0, description="Std dev after treatment",
    )
    original_median: float = Field(
        default=0.0, description="Median before treatment",
    )
    treated_median: float = Field(
        default=0.0, description="Median after treatment",
    )
    mean_change_pct: float = Field(
        default=0.0,
        description="Percentage change in mean",
    )
    std_change_pct: float = Field(
        default=0.0,
        description="Percentage change in std dev",
    )
    distribution_shift: float = Field(
        default=0.0, ge=0.0,
        description="Measure of distribution change",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class OutlierReport(BaseModel):
    """Complete outlier detection report for a dataset.

    Aggregates detection, classification, and treatment results into
    a comprehensive report for review and audit.

    Attributes:
        report_id: Unique identifier.
        job_id: Parent detection job identifier.
        dataset_id: Identifier of the analyzed dataset.
        total_records: Total records in the dataset.
        total_columns_analyzed: Number of columns analyzed.
        total_outliers: Total outliers detected.
        outlier_pct: Overall fraction of outliers.
        by_method: Outlier counts per detection method.
        by_class: Outlier counts per classification.
        by_treatment: Treatment counts per strategy.
        column_summaries: Per-column summaries.
        impact: Impact analysis of treatments.
        generated_at: When the report was generated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier",
    )
    job_id: str = Field(
        default="", description="Parent detection job identifier",
    )
    dataset_id: str = Field(
        default="",
        description="Identifier of the analyzed dataset",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total records in the dataset",
    )
    total_columns_analyzed: int = Field(
        default=0, ge=0,
        description="Number of columns analyzed",
    )
    total_outliers: int = Field(
        default=0, ge=0,
        description="Total outliers detected",
    )
    outlier_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall fraction of outliers",
    )
    by_method: Dict[str, int] = Field(
        default_factory=dict,
        description="Outlier counts per detection method",
    )
    by_class: Dict[str, int] = Field(
        default_factory=dict,
        description="Outlier counts per classification",
    )
    by_treatment: Dict[str, int] = Field(
        default_factory=dict,
        description="Treatment counts per strategy",
    )
    column_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-column summaries",
    )
    impact: Optional[ImpactAnalysis] = Field(
        None, description="Impact analysis of treatments",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="When the report was generated",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class PipelineConfig(BaseModel):
    """Configuration for the full outlier detection pipeline.

    Defines pipeline-level settings including detection methods,
    ensemble method, treatment strategy, and output preferences.

    Attributes:
        methods: List of detection methods to run.
        ensemble_method: Method for combining detector scores.
        min_consensus: Minimum methods flagging for consensus.
        treatment_strategy: Default treatment for outliers.
        enable_classification: Whether to classify outlier types.
        enable_contextual: Whether to run contextual detection.
        enable_temporal: Whether to run temporal detection.
        enable_multivariate: Whether to run multivariate detection.
        columns: Specific columns to analyze (empty = all numeric).
        group_columns: Columns for contextual grouping.
        time_column: Column for temporal ordering.
        report_format: Output report format.
        confidence_threshold: Minimum confidence for flagging.
    """

    methods: List[DetectionMethod] = Field(
        default_factory=lambda: [
            DetectionMethod.IQR,
            DetectionMethod.ZSCORE,
            DetectionMethod.MODIFIED_ZSCORE,
        ],
        description="List of detection methods to run",
    )
    ensemble_method: EnsembleMethod = Field(
        default=EnsembleMethod.WEIGHTED_AVERAGE,
        description="Method for combining detector scores",
    )
    min_consensus: int = Field(
        default=2, ge=1,
        description="Minimum methods flagging for consensus",
    )
    treatment_strategy: TreatmentStrategy = Field(
        default=TreatmentStrategy.FLAG,
        description="Default treatment for outliers",
    )
    enable_classification: bool = Field(
        default=True,
        description="Whether to classify outlier types",
    )
    enable_contextual: bool = Field(
        default=False,
        description="Whether to run contextual detection",
    )
    enable_temporal: bool = Field(
        default=False,
        description="Whether to run temporal detection",
    )
    enable_multivariate: bool = Field(
        default=False,
        description="Whether to run multivariate detection",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Specific columns to analyze (empty = all numeric)",
    )
    group_columns: List[str] = Field(
        default_factory=list,
        description="Columns for contextual grouping",
    )
    time_column: Optional[str] = Field(
        None, description="Column for temporal ordering",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output report format",
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for flagging",
    )

    model_config = {"extra": "forbid"}


class PipelineResult(BaseModel):
    """Complete result of an outlier detection pipeline run.

    Aggregates detection, classification, treatment, and validation
    results into a single pipeline output with full provenance.

    Attributes:
        pipeline_id: Unique identifier for this pipeline run.
        job_id: Identifier of the parent detection job.
        stage: Final pipeline stage reached.
        status: Final pipeline status.
        detection_results: Per-column detection results.
        ensemble_results: Ensemble combined results.
        classifications: Outlier classifications.
        treatments: Treatment results.
        validation_summary: Validation stage summary.
        report: Generated outlier report.
        total_processing_time_ms: Total pipeline processing time.
        created_at: When the pipeline completed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this pipeline run",
    )
    job_id: str = Field(
        default="", description="Identifier of the parent detection job",
    )
    stage: PipelineStage = Field(
        default=PipelineStage.DOCUMENT,
        description="Final pipeline stage reached",
    )
    status: OutlierStatus = Field(
        default=OutlierStatus.COMPLETED,
        description="Final pipeline status",
    )
    detection_results: List[DetectionResult] = Field(
        default_factory=list,
        description="Per-column detection results",
    )
    ensemble_results: List[EnsembleResult] = Field(
        default_factory=list,
        description="Ensemble combined results",
    )
    classifications: List[OutlierClassification] = Field(
        default_factory=list,
        description="Outlier classifications",
    )
    treatments: List[TreatmentResult] = Field(
        default_factory=list,
        description="Treatment results",
    )
    validation_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation stage summary",
    )
    report: Optional[OutlierReport] = Field(
        None, description="Generated outlier report",
    )
    total_processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Total pipeline processing time in milliseconds",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the pipeline completed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class DetectionJobConfig(BaseModel):
    """Configuration and tracking for an outlier detection job.

    Represents a single end-to-end detection run with progress
    counters, timing, error tracking, and provenance for audit trail.

    Attributes:
        job_id: Unique identifier for this detection job.
        dataset_id: Identifier of the dataset being analyzed.
        status: Current job execution status.
        stage: Current pipeline stage being executed.
        total_records: Total number of input records.
        total_columns: Total number of columns in the dataset.
        outliers_detected: Total outliers detected so far.
        outliers_classified: Total outliers classified so far.
        treatments_applied: Total treatments applied so far.
        pipeline_config: Pipeline configuration for this job.
        error_message: Error message if the job failed.
        started_at: Timestamp when the job started.
        completed_at: Timestamp when the job completed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this detection job",
    )
    dataset_id: str = Field(
        default="",
        description="Identifier of the dataset being analyzed",
    )
    status: OutlierStatus = Field(
        default=OutlierStatus.PENDING,
        description="Current job execution status",
    )
    stage: PipelineStage = Field(
        default=PipelineStage.DETECT,
        description="Current pipeline stage being executed",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total number of input records",
    )
    total_columns: int = Field(
        default=0, ge=0,
        description="Total number of columns in the dataset",
    )
    outliers_detected: int = Field(
        default=0, ge=0,
        description="Total outliers detected so far",
    )
    outliers_classified: int = Field(
        default=0, ge=0,
        description="Total outliers classified so far",
    )
    treatments_applied: int = Field(
        default=0, ge=0,
        description="Total treatments applied so far",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None, description="Pipeline configuration for this job",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if the job failed",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the job started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the job completed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @property
    def is_active(self) -> bool:
        """Return True if the job is currently executing."""
        return self.status in (
            OutlierStatus.PENDING,
            OutlierStatus.DETECTING,
            OutlierStatus.CLASSIFYING,
            OutlierStatus.TREATING,
        )

    @property
    def progress_pct(self) -> float:
        """Return pipeline progress as a percentage (0.0 to 100.0)."""
        stage_progress = {
            PipelineStage.DETECT: 20.0,
            PipelineStage.CLASSIFY: 40.0,
            PipelineStage.TREAT: 60.0,
            PipelineStage.VALIDATE: 80.0,
            PipelineStage.DOCUMENT: 100.0,
        }
        if self.status == OutlierStatus.COMPLETED:
            return 100.0
        if self.status == OutlierStatus.FAILED:
            return 0.0
        return stage_progress.get(self.stage, 0.0)


class OutlierStatistics(BaseModel):
    """Aggregated operational statistics for the detection service.

    Provides high-level metrics for monitoring the overall
    health, throughput, and effectiveness of the detection
    pipeline.

    Attributes:
        total_jobs: Total number of detection jobs executed.
        total_records: Total number of records processed across all jobs.
        total_outliers_detected: Total outliers detected.
        total_treatments_applied: Total treatments applied.
        avg_outlier_pct: Average outlier percentage across jobs.
        by_status: Count of jobs per status.
        by_method: Count of outliers per detection method.
        by_class: Count of outliers per classification.
        by_treatment: Count of treatments per strategy.
        timestamp: Timestamp when statistics were computed.
    """

    total_jobs: int = Field(
        default=0, ge=0,
        description="Total number of detection jobs executed",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total number of records processed across all jobs",
    )
    total_outliers_detected: int = Field(
        default=0, ge=0,
        description="Total outliers detected",
    )
    total_treatments_applied: int = Field(
        default=0, ge=0,
        description="Total treatments applied",
    )
    avg_outlier_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average outlier percentage across jobs",
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of jobs per status",
    )
    by_method: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of outliers per detection method",
    )
    by_class: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of outliers per classification",
    )
    by_treatment: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of treatments per strategy",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when statistics were computed",
    )

    model_config = {"extra": "forbid"}


class ThresholdConfig(BaseModel):
    """Configuration for a detection threshold.

    Attributes:
        column_name: Column this threshold applies to.
        method: Detection method this threshold is for.
        value: Threshold value.
        source: Source of the threshold.
        description: Human-readable description.
    """

    column_name: str = Field(
        ..., description="Column this threshold applies to",
    )
    method: DetectionMethod = Field(
        ..., description="Detection method this threshold is for",
    )
    value: float = Field(
        ..., description="Threshold value",
    )
    source: ThresholdSource = Field(
        default=ThresholdSource.STATISTICAL,
        description="Source of the threshold",
    )
    description: str = Field(
        default="",
        description="Human-readable description",
    )

    model_config = {"extra": "forbid"}

    @field_validator("column_name")
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        """Validate column_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("column_name must be non-empty")
        return v


class ColumnOutlierSummary(BaseModel):
    """Summary of outlier detection results for a single column.

    Attributes:
        column_name: Name of the column.
        total_points: Total data points.
        outliers_detected: Number of outliers detected.
        outlier_pct: Fraction flagged as outliers.
        mean_score: Mean outlier score.
        max_score: Maximum outlier score.
        methods_used: Detection methods applied.
        dominant_class: Most frequent outlier classification.
        treatment_applied: Treatment strategy used.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    column_name: str = Field(
        ..., description="Name of the column",
    )
    total_points: int = Field(
        default=0, ge=0,
        description="Total data points",
    )
    outliers_detected: int = Field(
        default=0, ge=0,
        description="Number of outliers detected",
    )
    outlier_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction flagged as outliers",
    )
    mean_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean outlier score",
    )
    max_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Maximum outlier score",
    )
    methods_used: List[str] = Field(
        default_factory=list,
        description="Detection methods applied",
    )
    dominant_class: Optional[str] = Field(
        None, description="Most frequent outlier classification",
    )
    treatment_applied: Optional[str] = Field(
        None, description="Treatment strategy used",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("column_name")
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        """Validate column_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("column_name must be non-empty")
        return v


class BatchDetectionResult(BaseModel):
    """Batch result aggregating detection across multiple columns.

    Attributes:
        batch_id: Unique identifier for this batch.
        job_id: Parent detection job identifier.
        results: Per-column detection results.
        ensemble_results: Ensemble combined results.
        total_outliers: Total outliers across all columns.
        columns_analyzed: Number of columns analyzed.
        processing_time_ms: Total processing time.
        created_at: When the batch was processed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this batch",
    )
    job_id: str = Field(
        default="",
        description="Parent detection job identifier",
    )
    results: List[DetectionResult] = Field(
        default_factory=list,
        description="Per-column detection results",
    )
    ensemble_results: List[EnsembleResult] = Field(
        default_factory=list,
        description="Ensemble combined results",
    )
    total_outliers: int = Field(
        default=0, ge=0,
        description="Total outliers across all columns",
    )
    columns_analyzed: int = Field(
        default=0, ge=0,
        description="Number of columns analyzed",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Total processing time in milliseconds",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the batch was processed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models (8)
# =============================================================================


class CreateDetectionJobRequest(BaseModel):
    """Request body for creating a new outlier detection job.

    Attributes:
        dataset_id: Identifier of the dataset to analyze.
        records: List of record dictionaries to analyze.
        pipeline_config: Optional pipeline configuration.
    """

    dataset_id: str = Field(
        default="", description="Identifier of the dataset to analyze",
    )
    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to analyze",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None, description="Optional pipeline configuration",
    )

    model_config = {"extra": "forbid"}


class DetectOutliersRequest(BaseModel):
    """Request body for detecting outliers in a dataset.

    Attributes:
        records: List of record dictionaries to analyze.
        columns: Specific columns to analyze (empty = auto-detect numeric).
        methods: Detection methods to use.
        ensemble_method: Ensemble combination method.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to analyze",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Specific columns to analyze",
    )
    methods: List[DetectionMethod] = Field(
        default_factory=lambda: [DetectionMethod.IQR, DetectionMethod.ZSCORE],
        description="Detection methods to use",
    )
    ensemble_method: EnsembleMethod = Field(
        default=EnsembleMethod.WEIGHTED_AVERAGE,
        description="Ensemble combination method",
    )

    model_config = {"extra": "forbid"}


class ClassifyOutliersRequest(BaseModel):
    """Request body for classifying detected outliers.

    Attributes:
        records: Original record dictionaries.
        detections: List of outlier score dictionaries.
        context: Optional context information for classification.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="Original record dictionaries",
    )
    detections: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of outlier score dictionaries",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context information",
    )

    model_config = {"extra": "forbid"}


class TreatOutliersRequest(BaseModel):
    """Request body for treating detected outliers.

    Attributes:
        records: Original record dictionaries.
        detections: List of outlier score dictionaries.
        strategy: Treatment strategy to apply.
        options: Optional treatment options.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="Original record dictionaries",
    )
    detections: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of outlier score dictionaries",
    )
    strategy: TreatmentStrategy = Field(
        default=TreatmentStrategy.FLAG,
        description="Treatment strategy to apply",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional treatment options",
    )

    model_config = {"extra": "forbid"}


class SubmitFeedbackRequest(BaseModel):
    """Request body for submitting feedback on an outlier.

    Attributes:
        record_index: Row index of the outlier.
        column_name: Column name.
        feedback_type: Type of feedback.
        corrected_class: Corrected class (for reclassification).
        comment: Human comment.
    """

    record_index: int = Field(
        ..., ge=0,
        description="Row index of the outlier",
    )
    column_name: str = Field(
        default="",
        description="Column name",
    )
    feedback_type: FeedbackType = Field(
        ..., description="Type of feedback",
    )
    corrected_class: Optional[OutlierClass] = Field(
        None, description="Corrected class (for reclassification)",
    )
    comment: str = Field(
        default="",
        description="Human comment",
    )

    model_config = {"extra": "forbid"}


class RunPipelineRequest(BaseModel):
    """Request body for running the full outlier detection pipeline.

    Attributes:
        records: List of record dictionaries to analyze.
        dataset_id: Optional dataset identifier.
        pipeline_config: Optional pipeline configuration.
        options: Optional overrides.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to analyze",
    )
    dataset_id: str = Field(
        default="", description="Optional dataset identifier",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None, description="Optional pipeline configuration",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional overrides",
    )

    model_config = {"extra": "forbid"}


class BatchDetectRequest(BaseModel):
    """Request body for batch detection across multiple datasets.

    Attributes:
        datasets: List of dataset records (each a list of dicts).
        dataset_ids: List of dataset identifiers.
        pipeline_config: Shared pipeline configuration.
    """

    datasets: List[List[Dict[str, Any]]] = Field(
        ..., min_length=1,
        description="List of dataset records (each a list of dicts)",
    )
    dataset_ids: List[str] = Field(
        default_factory=list,
        description="List of dataset identifiers",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None, description="Shared pipeline configuration",
    )

    model_config = {"extra": "forbid"}


class ConfigureThresholdsRequest(BaseModel):
    """Request body for configuring detection thresholds.

    Attributes:
        thresholds: List of threshold configurations.
        domain_thresholds: List of domain-specific thresholds.
    """

    thresholds: List[ThresholdConfig] = Field(
        default_factory=list,
        description="List of threshold configurations",
    )
    domain_thresholds: List[DomainThreshold] = Field(
        default_factory=list,
        description="List of domain-specific thresholds",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports
    # -------------------------------------------------------------------------
    "AnomalyDetector",
    "AnomalyResult",
    "QualityDimension",
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "DEFAULT_METHOD_WEIGHTS",
    "SEVERITY_THRESHOLDS",
    "CLASSIFICATION_CONFIDENCE",
    "PIPELINE_STAGE_ORDER",
    "DETECTION_METHODS",
    "TREATMENT_STRATEGIES",
    "REPORT_FORMAT_OPTIONS",
    "TEMPORAL_METHODS",
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
]
