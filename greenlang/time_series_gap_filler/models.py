# -*- coding: utf-8 -*-
"""
Time Series Gap Filler Agent Service Data Models - AGENT-DATA-014

Pydantic v2 data models for the Time Series Gap Filler SDK. Re-exports the
Layer 1 models from the Missing Value Imputer's TimeSeriesImputerEngine,
and defines additional SDK models for gap detection, frequency analysis,
gap characterisation, fill strategy selection, fill execution, validation,
calendar-aware filling, cross-series filling, seasonal decomposition,
pipeline orchestration, statistics, and reporting.

Re-exported Layer 1 sources:
    - greenlang.missing_value_imputer.time_series_imputer:
        TimeSeriesImputerEngine (as L1TimeSeriesImputerEngine)
    - greenlang.missing_value_imputer.models:
        ImputedValue (as L1ImputedValue),
        ConfidenceLevel (as L1ConfidenceLevel)

New enumerations (13):
    - FrequencyLevel, GapType, GapSeverity, FillStrategy,
      FillStatus, ValidationResult, CalendarType,
      SeasonalPattern, TrendType, PipelineStage, ReportFormat,
      ConfidenceLevel, DataResolution

New SDK models (20):
    - GapRecord, GapDetectionResult, FrequencyResult,
      FillValue, FillResult, ValidationReport,
      CalendarDefinition, ReferenceSeries, GapFillStrategy,
      ImpactAssessment, GapFillReport, PipelineConfig,
      PipelineResult, GapFillerJobConfig, GapFillerStatistics,
      SeriesMetadata, BatchDetectionResult, CrossSeriesResult,
      SeasonalDecomposition, GapCharacterization

Request models (8):
    - CreateJobRequest, DetectGapsRequest,
      BatchDetectRequest, AnalyzeFrequencyRequest,
      FillGapsRequest, ValidateFillsRequest,
      CreateCalendarRequest, RunPipelineRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from missing_value_imputer
# ---------------------------------------------------------------------------

try:
    from greenlang.missing_value_imputer.time_series_imputer import (
        TimeSeriesImputerEngine as L1TimeSeriesImputerEngine,
    )
    TimeSeriesImputerEngine = L1TimeSeriesImputerEngine
except ImportError:
    TimeSeriesImputerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.models import (
        ImputedValue as L1ImputedValue,
    )
    ImputedValue_L1 = L1ImputedValue
except ImportError:
    ImputedValue_L1 = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.models import (
        ConfidenceLevel as L1ConfidenceLevel,
    )
    ConfidenceLevel_L1 = L1ConfidenceLevel
except ImportError:
    ConfidenceLevel_L1 = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default fill strategy weights for auto-selection scoring.
DEFAULT_STRATEGY_WEIGHTS: Dict[str, float] = {
    "linear": 1.0,
    "cubic_spline": 1.2,
    "polynomial": 0.9,
    "akima": 1.1,
    "nearest": 0.6,
    "seasonal": 1.4,
    "trend": 1.0,
    "cross_series": 1.3,
    "calendar": 0.8,
    "auto": 0.0,
}

#: Default gap severity thresholds by gap length (number of missing points).
GAP_SEVERITY_THRESHOLDS: Dict[str, int] = {
    "critical": 50,
    "high": 20,
    "medium": 5,
    "low": 1,
}

#: Fill confidence thresholds for quality assessment.
FILL_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "very_high": 0.95,
    "high": 0.85,
    "medium": 0.65,
    "low": 0.40,
    "very_low": 0.0,
}

#: Pipeline stage execution order.
PIPELINE_STAGE_ORDER: tuple = (
    "detect", "characterize", "select_strategy", "fill", "validate", "document",
)

#: All supported fill strategies.
FILL_STRATEGIES: tuple = (
    "linear", "cubic_spline", "polynomial", "akima", "nearest",
    "seasonal", "trend", "cross_series", "calendar", "auto",
)

#: All supported gap types.
GAP_TYPES: tuple = (
    "short_gap", "long_gap", "periodic_gap", "random_gap", "systematic_gap",
)

#: Report format options.
REPORT_FORMAT_OPTIONS: tuple = ("json", "text", "markdown", "html")

#: Standard time-series frequency strings.
FREQUENCY_STRINGS: tuple = (
    "sub_hourly", "hourly", "daily", "weekly", "biweekly",
    "monthly", "quarterly", "semi_annual", "annual",
)

#: Calendar type options.
CALENDAR_TYPES: tuple = ("business_days", "fiscal_year", "custom")

#: Data resolution levels.
RESOLUTION_LEVELS: tuple = (
    "raw", "hourly", "daily", "weekly", "monthly", "quarterly", "annual",
)


# =============================================================================
# Enumerations (13)
# =============================================================================


class FrequencyLevel(str, Enum):
    """Detected or expected time-series frequency.

    SUB_HOURLY: Data arrives more frequently than once per hour.
    HOURLY: Data arrives approximately once per hour.
    DAILY: Data arrives approximately once per day.
    WEEKLY: Data arrives approximately once per week.
    BIWEEKLY: Data arrives approximately once every two weeks.
    MONTHLY: Data arrives approximately once per month.
    QUARTERLY: Data arrives approximately once per quarter.
    SEMI_ANNUAL: Data arrives approximately once every six months.
    ANNUAL: Data arrives approximately once per year.
    """

    SUB_HOURLY = "sub_hourly"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


class GapType(str, Enum):
    """Classification of a gap pattern in a time series.

    SHORT_GAP: A gap spanning a small number of consecutive missing points.
    LONG_GAP: A gap spanning a large number of consecutive missing points.
    PERIODIC_GAP: A gap that recurs at a regular interval.
    RANDOM_GAP: A gap with no discernible pattern in its occurrence.
    SYSTEMATIC_GAP: A gap caused by a systematic data collection issue.
    """

    SHORT_GAP = "short_gap"
    LONG_GAP = "long_gap"
    PERIODIC_GAP = "periodic_gap"
    RANDOM_GAP = "random_gap"
    SYSTEMATIC_GAP = "systematic_gap"


class GapSeverity(str, Enum):
    """Severity classification for a detected gap.

    LOW: Minor gap with minimal impact on analysis.
    MEDIUM: Moderate gap that may affect trend estimation.
    HIGH: Significant gap that degrades analysis quality.
    CRITICAL: Severe gap that makes analysis unreliable.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FillStrategy(str, Enum):
    """Strategy for filling detected gaps in a time series.

    LINEAR: Linear interpolation between boundary points.
    CUBIC_SPLINE: Cubic spline interpolation for smooth fills.
    POLYNOMIAL: Polynomial interpolation of configurable degree.
    AKIMA: Akima sub-spline interpolation (local, avoids overshoot).
    NEAREST: Nearest-neighbour value propagation.
    SEASONAL: Seasonal decomposition-based fill using periodic patterns.
    TREND: Trend extrapolation-based fill using detected trend.
    CROSS_SERIES: Fill using correlated donor series values.
    CALENDAR: Calendar-aware fill using business day or fiscal rules.
    AUTO: Automatic strategy selection based on gap characteristics.
    """

    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    POLYNOMIAL = "polynomial"
    AKIMA = "akima"
    NEAREST = "nearest"
    SEASONAL = "seasonal"
    TREND = "trend"
    CROSS_SERIES = "cross_series"
    CALENDAR = "calendar"
    AUTO = "auto"


class FillStatus(str, Enum):
    """Lifecycle status of a gap-fill job.

    Tracks the current execution state of a fill pipeline
    from submission through completion, failure, or cancellation.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationResult(str, Enum):
    """Outcome of a fill validation check.

    PASSED: The fill values passed all validation checks.
    FAILED: The fill values failed one or more validation checks.
    WARNING: The fill values triggered non-critical warnings.
    """

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class CalendarType(str, Enum):
    """Type of calendar used for calendar-aware gap filling.

    BUSINESS_DAYS: Standard business-day calendar (Mon-Fri).
    FISCAL_YEAR: Fiscal year calendar with configurable start month.
    CUSTOM: User-defined calendar with explicit holidays and rules.
    """

    BUSINESS_DAYS = "business_days"
    FISCAL_YEAR = "fiscal_year"
    CUSTOM = "custom"


class SeasonalPattern(str, Enum):
    """Detected or specified seasonal pattern period.

    DAILY: Pattern repeats every day (e.g. intra-day load profile).
    WEEKLY: Pattern repeats every week (e.g. weekday vs weekend).
    MONTHLY: Pattern repeats every month (e.g. billing cycles).
    QUARTERLY: Pattern repeats every quarter (e.g. reporting cycles).
    ANNUAL: Pattern repeats every year (e.g. heating season).
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class TrendType(str, Enum):
    """Type of trend detected in a time series.

    LINEAR: Constant rate of change over time.
    MODERATE_LINEAR: Moderate linear trend (lower R-squared).
    EXPONENTIAL: Rate of change proportional to current level.
    POLYNOMIAL: Higher-order polynomial trend.
    STATIONARY: No significant trend (flat / random walk).
    NONE: No significant trend detected.
    UNKNOWN: Insufficient data to determine trend.
    """

    LINEAR = "linear"
    MODERATE_LINEAR = "moderate_linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    STATIONARY = "stationary"
    NONE = "none"
    UNKNOWN = "unknown"


class PipelineStage(str, Enum):
    """Stage in the gap-fill pipeline.

    The pipeline executes stages in order: detect, characterize,
    select_strategy, fill, validate, document. Each stage produces
    output consumed by the next stage.
    """

    DETECT = "detect"
    CHARACTERIZE = "characterize"
    SELECT_STRATEGY = "select_strategy"
    FILL = "fill"
    VALIDATE = "validate"
    DOCUMENT = "document"


class ReportFormat(str, Enum):
    """Output format for gap-fill reports.

    Defines the serialization format for generated reports
    including detection results, fill details, and validations.
    """

    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


class ConfidenceLevel(str, Enum):
    """Confidence classification for a fill value.

    VERY_LOW: Very low confidence in the filled value (< 0.40).
    LOW: Low confidence in the filled value (0.40 - 0.65).
    MEDIUM: Medium confidence (0.65 - 0.85).
    HIGH: High confidence (0.85 - 0.95).
    VERY_HIGH: Very high confidence (>= 0.95).
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DataResolution(str, Enum):
    """Temporal resolution of the data series.

    Describes the level of aggregation applied to raw data
    before gap detection and filling operations.
    """

    RAW = "raw"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


# =============================================================================
# SDK Data Models (20)
# =============================================================================


class GapRecord(BaseModel):
    """A single detected gap in a time series.

    Represents one contiguous block of missing data, including its
    location, length, classification, and severity.

    Attributes:
        gap_id: Unique identifier for this gap record.
        start_index: Index of the first missing point in the gap.
        end_index: Index of the last missing point in the gap (inclusive).
        gap_length: Number of consecutive missing points in this gap.
        gap_type: Classification of the gap pattern.
        severity: Severity classification of the gap.
        expected_count: Expected number of data points in this interval.
        start_timestamp: Timestamp of the first missing point (if available).
        end_timestamp: Timestamp of the last missing point (if available).
        context_before: Value immediately preceding the gap (if available).
        context_after: Value immediately following the gap (if available).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    gap_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this gap record",
    )
    start_index: int = Field(
        ..., ge=0,
        description="Index of the first missing point in the gap",
    )
    end_index: int = Field(
        ..., ge=0,
        description="Index of the last missing point in the gap (inclusive)",
    )
    gap_length: int = Field(
        default=1, ge=1,
        description="Number of consecutive missing points in this gap",
    )
    gap_type: GapType = Field(
        default=GapType.RANDOM_GAP,
        description="Classification of the gap pattern",
    )
    severity: GapSeverity = Field(
        default=GapSeverity.LOW,
        description="Severity classification of the gap",
    )
    expected_count: int = Field(
        default=0, ge=0,
        description="Expected number of data points in this interval",
    )
    start_timestamp: Optional[str] = Field(
        None,
        description="Timestamp of the first missing point (if available)",
    )
    end_timestamp: Optional[str] = Field(
        None,
        description="Timestamp of the last missing point (if available)",
    )
    context_before: Optional[float] = Field(
        None,
        description="Value immediately preceding the gap (if available)",
    )
    context_after: Optional[float] = Field(
        None,
        description="Value immediately following the gap (if available)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("end_index")
    @classmethod
    def validate_end_index(cls, v: int, info: Any) -> int:
        """Validate end_index is greater than or equal to start_index."""
        start = info.data.get("start_index")
        if start is not None and v < start:
            raise ValueError(
                f"end_index ({v}) must be >= start_index ({start})"
            )
        return v


class GapDetectionResult(BaseModel):
    """Result of gap detection for a single time series.

    Contains all detected gaps, summary statistics, and the
    overall missingness profile for the series.

    Attributes:
        result_id: Unique identifier for this detection result.
        series_id: Identifier of the analyzed time series.
        total_points: Total number of data points in the series.
        missing_count: Total number of missing values detected.
        gap_count: Number of distinct contiguous gaps detected.
        gaps: List of detected gap records.
        gap_pct: Fraction of total points that are missing (0.0-1.0).
        longest_gap: Length of the longest contiguous gap.
        mean_gap_length: Mean length of all detected gaps.
        processing_time_ms: Detection processing time in milliseconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this detection result",
    )
    series_id: str = Field(
        ..., description="Identifier of the analyzed time series",
    )
    total_points: int = Field(
        default=0, ge=0,
        description="Total number of data points in the series",
    )
    missing_count: int = Field(
        default=0, ge=0,
        description="Total number of missing values detected",
    )
    gap_count: int = Field(
        default=0, ge=0,
        description="Number of distinct contiguous gaps detected",
    )
    gaps: List[GapRecord] = Field(
        default_factory=list,
        description="List of detected gap records",
    )
    gap_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of total points that are missing (0.0-1.0)",
    )
    longest_gap: int = Field(
        default=0, ge=0,
        description="Length of the longest contiguous gap",
    )
    mean_gap_length: float = Field(
        default=0.0, ge=0.0,
        description="Mean length of all detected gaps",
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

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, v: str) -> str:
        """Validate series_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("series_id must be non-empty")
        return v


class FrequencyResult(BaseModel):
    """Result of frequency analysis for a time series.

    Reports the detected dominant frequency, regularity score,
    and confidence in the frequency determination.

    Attributes:
        result_id: Unique identifier for this frequency result.
        series_id: Identifier of the analyzed time series.
        detected_frequency: Detected dominant frequency level.
        regularity_score: Score (0.0-1.0) indicating how regular the spacing is.
        confidence: Confidence in the frequency determination (0.0-1.0).
        sample_count: Number of inter-observation intervals sampled.
        dominant_period: Dominant period in seconds.
        median_interval_seconds: Median interval between observations in seconds.
        std_interval_seconds: Standard deviation of intervals in seconds.
        detected_patterns: List of secondary frequency patterns detected.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this frequency result",
    )
    series_id: str = Field(
        ..., description="Identifier of the analyzed time series",
    )
    detected_frequency: FrequencyLevel = Field(
        ..., description="Detected dominant frequency level",
    )
    regularity_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Score indicating how regular the spacing is",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the frequency determination (0.0-1.0)",
    )
    sample_count: int = Field(
        default=0, ge=0,
        description="Number of inter-observation intervals sampled",
    )
    dominant_period: float = Field(
        default=0.0, ge=0.0,
        description="Dominant period in seconds",
    )
    median_interval_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Median interval between observations in seconds",
    )
    std_interval_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Standard deviation of intervals in seconds",
    )
    detected_patterns: List[str] = Field(
        default_factory=list,
        description="List of secondary frequency patterns detected",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, v: str) -> str:
        """Validate series_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("series_id must be non-empty")
        return v


class FillValue(BaseModel):
    """A single filled value for a missing data point.

    Represents the result of filling one gap position, including
    the filled value, method used, confidence, and provenance.

    Attributes:
        fill_id: Unique identifier for this fill record.
        index: Position index in the original series.
        timestamp: Timestamp of the filled position (if available).
        filled_value: The value produced by the fill operation.
        original_state: Original state of the data point ('missing' or 'null').
        method: Fill strategy that produced this value.
        confidence: Confidence in the filled value (0.0-1.0).
        confidence_level: Categorical confidence classification.
        lower_bound: Lower bound of the fill confidence interval.
        upper_bound: Upper bound of the fill confidence interval.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    fill_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this fill record",
    )
    index: int = Field(
        ..., ge=0,
        description="Position index in the original series",
    )
    timestamp: Optional[str] = Field(
        None,
        description="Timestamp of the filled position (if available)",
    )
    filled_value: float = Field(
        ..., description="The value produced by the fill operation",
    )
    original_state: str = Field(
        default="missing",
        description="Original state of the data point",
    )
    method: FillStrategy = Field(
        ..., description="Fill strategy that produced this value",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the filled value (0.0-1.0)",
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Categorical confidence classification",
    )
    lower_bound: Optional[float] = Field(
        None,
        description="Lower bound of the fill confidence interval",
    )
    upper_bound: Optional[float] = Field(
        None,
        description="Upper bound of the fill confidence interval",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class FillResult(BaseModel):
    """Result of a fill operation for a single time series.

    Aggregates all filled values produced by one fill execution,
    including method, confidence statistics, and timing.

    Attributes:
        result_id: Unique identifier for this fill result.
        series_id: Identifier of the target time series.
        fills_count: Number of values filled.
        method: Primary fill strategy used.
        avg_confidence: Mean confidence across all filled values.
        min_confidence: Minimum confidence among filled values.
        max_confidence: Maximum confidence among filled values.
        fill_values: List of individual fill records.
        gaps_addressed: Number of gaps addressed by this fill.
        duration_ms: Fill operation duration in milliseconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this fill result",
    )
    series_id: str = Field(
        ..., description="Identifier of the target time series",
    )
    fills_count: int = Field(
        default=0, ge=0,
        description="Number of values filled",
    )
    method: FillStrategy = Field(
        ..., description="Primary fill strategy used",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean confidence across all filled values",
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum confidence among filled values",
    )
    max_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Maximum confidence among filled values",
    )
    fill_values: List[FillValue] = Field(
        default_factory=list,
        description="List of individual fill records",
    )
    gaps_addressed: int = Field(
        default=0, ge=0,
        description="Number of gaps addressed by this fill",
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0,
        description="Fill operation duration in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, v: str) -> str:
        """Validate series_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("series_id must be non-empty")
        return v


class ValidationReport(BaseModel):
    """Result of fill validation for a single time series.

    Evaluates the quality of filled values by comparing the
    post-fill distribution to the original distribution and
    checking plausibility of each filled point.

    Attributes:
        validation_id: Unique identifier for this validation report.
        series_id: Identifier of the validated time series.
        result: Overall validation outcome.
        distribution_preserved: Whether the original distribution was preserved.
        plausibility_passed: Whether all filled values passed plausibility checks.
        ks_statistic: Kolmogorov-Smirnov statistic comparing distributions.
        p_value: P-value from the KS test (> 0.05 suggests preservation).
        rmse: Root mean square error (if ground truth available).
        mae: Mean absolute error (if ground truth available).
        fill_count_validated: Number of filled values validated.
        outlier_count: Number of fills flagged as potential outliers.
        warnings: List of validation warning messages.
        details: Additional validation metrics and per-fill results.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    validation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this validation report",
    )
    series_id: str = Field(
        ..., description="Identifier of the validated time series",
    )
    result: ValidationResult = Field(
        default=ValidationResult.PASSED,
        description="Overall validation outcome",
    )
    distribution_preserved: bool = Field(
        default=True,
        description="Whether the original distribution was preserved",
    )
    plausibility_passed: bool = Field(
        default=True,
        description="Whether all filled values passed plausibility checks",
    )
    ks_statistic: float = Field(
        default=0.0, ge=0.0,
        description="Kolmogorov-Smirnov statistic comparing distributions",
    )
    p_value: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="P-value from the KS test",
    )
    rmse: Optional[float] = Field(
        None,
        description="Root mean square error (if ground truth available)",
    )
    mae: Optional[float] = Field(
        None,
        description="Mean absolute error (if ground truth available)",
    )
    fill_count_validated: int = Field(
        default=0, ge=0,
        description="Number of filled values validated",
    )
    outlier_count: int = Field(
        default=0, ge=0,
        description="Number of fills flagged as potential outliers",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warning messages",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional validation metrics and per-fill results",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, v: str) -> str:
        """Validate series_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("series_id must be non-empty")
        return v


class CalendarDefinition(BaseModel):
    """Calendar configuration for calendar-aware gap filling.

    Defines business days, holidays, and fiscal period boundaries
    used to distinguish expected gaps (weekends, holidays) from
    unexpected gaps in a time series.

    Attributes:
        calendar_id: Unique identifier for this calendar.
        name: Human-readable calendar name.
        calendar_type: Type of calendar.
        business_days: List of ISO weekday numbers (1=Mon ... 7=Sun) that are business days.
        holidays: List of holiday date strings (YYYY-MM-DD format).
        fiscal_start_month: Month number (1-12) for fiscal year start.
        timezone: IANA timezone string for the calendar.
        description: Human-readable description of the calendar.
        active: Whether this calendar is currently active.
        created_at: When the calendar was created.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    calendar_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this calendar",
    )
    name: str = Field(
        ..., description="Human-readable calendar name",
    )
    calendar_type: CalendarType = Field(
        default=CalendarType.BUSINESS_DAYS,
        description="Type of calendar",
    )
    business_days: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5],
        description="List of ISO weekday numbers that are business days",
    )
    holidays: List[str] = Field(
        default_factory=list,
        description="List of holiday date strings (YYYY-MM-DD format)",
    )
    fiscal_start_month: int = Field(
        default=1, ge=1, le=12,
        description="Month number (1-12) for fiscal year start",
    )
    timezone: str = Field(
        default="UTC",
        description="IANA timezone string for the calendar",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the calendar",
    )
    active: bool = Field(
        default=True,
        description="Whether this calendar is currently active",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the calendar was created",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("business_days")
    @classmethod
    def validate_business_days(cls, v: List[int]) -> List[int]:
        """Validate business_days contains valid ISO weekday numbers."""
        for day in v:
            if day < 1 or day > 7:
                raise ValueError(
                    f"business_days must contain values 1-7, got {day}"
                )
        return v


class ReferenceSeries(BaseModel):
    """A donor time series registered for cross-series gap filling.

    Contains the donor series data, metadata, and pre-computed
    correlation with the target series for ranked selection.

    Attributes:
        reference_id: Unique identifier for this reference series.
        series_id: Identifier of the donor series.
        name: Human-readable name of the donor series.
        values: List of numeric values in the donor series.
        timestamps: List of timestamp strings for the donor series.
        frequency: Detected frequency of the donor series.
        correlation: Pearson correlation with the target series.
        lag: Optimal lag offset (in periods) relative to the target.
        coverage_pct: Fraction of target timestamps covered by the donor.
        active: Whether this reference is currently active.
        created_at: When the reference was registered.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    reference_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this reference series",
    )
    series_id: str = Field(
        ..., description="Identifier of the donor series",
    )
    name: str = Field(
        default="",
        description="Human-readable name of the donor series",
    )
    values: List[Optional[float]] = Field(
        default_factory=list,
        description="List of numeric values in the donor series",
    )
    timestamps: List[str] = Field(
        default_factory=list,
        description="List of timestamp strings for the donor series",
    )
    frequency: Optional[FrequencyLevel] = Field(
        None,
        description="Detected frequency of the donor series",
    )
    correlation: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Pearson correlation with the target series",
    )
    lag: int = Field(
        default=0,
        description="Optimal lag offset (in periods) relative to the target",
    )
    coverage_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of target timestamps covered by the donor",
    )
    active: bool = Field(
        default=True,
        description="Whether this reference is currently active",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the reference was registered",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, v: str) -> str:
        """Validate series_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("series_id must be non-empty")
        return v


class GapFillStrategy(BaseModel):
    """Selected fill strategy for a specific gap.

    Records which strategy was chosen (manually or via auto-selection),
    its fallback chain, method-specific parameters, and selection metadata.

    Attributes:
        strategy_id: Unique identifier for this strategy selection.
        gap_id: Identifier of the gap this strategy is assigned to.
        strategy: Selected fill strategy.
        fallback_chain: Ordered list of fallback strategies if the primary fails.
        parameters: Method-specific parameters (e.g. polynomial degree, window size).
        auto_selected: Whether this strategy was chosen by the auto-selector.
        selection_score: Score from the auto-selector (0.0-1.0) if auto_selected.
        selection_reason: Reason for auto-selection (e.g. 'short gap, linear suitable').
        created_at: When the strategy was selected.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    strategy_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this strategy selection",
    )
    gap_id: str = Field(
        ..., description="Identifier of the gap this strategy is assigned to",
    )
    strategy: FillStrategy = Field(
        ..., description="Selected fill strategy",
    )
    fallback_chain: List[FillStrategy] = Field(
        default_factory=list,
        description="Ordered list of fallback strategies",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific parameters",
    )
    auto_selected: bool = Field(
        default=False,
        description="Whether this strategy was chosen by the auto-selector",
    )
    selection_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Score from the auto-selector",
    )
    selection_reason: str = Field(
        default="",
        description="Reason for auto-selection",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the strategy was selected",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("gap_id")
    @classmethod
    def validate_gap_id(cls, v: str) -> str:
        """Validate gap_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("gap_id must be non-empty")
        return v


class ImpactAssessment(BaseModel):
    """Assessment of the impact of gaps on downstream analysis.

    Quantifies how gaps affect trend estimation, year-over-year
    comparisons, and compliance reporting accuracy.

    Attributes:
        assessment_id: Unique identifier for this assessment.
        series_id: Identifier of the assessed time series.
        total_impact_pct: Overall data quality impact percentage (0.0-100.0).
        trend_impact: Impact on trend estimation (0.0 = none, 1.0 = severe).
        yoy_impact: Impact on year-over-year comparison (0.0-1.0).
        compliance_impact: Impact on regulatory compliance calculations (0.0-1.0).
        affected_periods: Number of reporting periods affected by gaps.
        data_completeness: Post-fill data completeness ratio (0.0-1.0).
        pre_fill_completeness: Pre-fill data completeness ratio (0.0-1.0).
        recommendations: List of recommendations for improving data quality.
        risk_level: Overall risk classification.
        created_at: When the assessment was performed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    assessment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this assessment",
    )
    series_id: str = Field(
        ..., description="Identifier of the assessed time series",
    )
    total_impact_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Overall data quality impact percentage (0.0-100.0)",
    )
    trend_impact: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Impact on trend estimation",
    )
    yoy_impact: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Impact on year-over-year comparison",
    )
    compliance_impact: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Impact on regulatory compliance calculations",
    )
    affected_periods: int = Field(
        default=0, ge=0,
        description="Number of reporting periods affected by gaps",
    )
    data_completeness: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Post-fill data completeness ratio",
    )
    pre_fill_completeness: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Pre-fill data completeness ratio",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of recommendations for improving data quality",
    )
    risk_level: GapSeverity = Field(
        default=GapSeverity.LOW,
        description="Overall risk classification",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the assessment was performed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, v: str) -> str:
        """Validate series_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("series_id must be non-empty")
        return v


class GapFillReport(BaseModel):
    """Compliance-grade report summarising gap detection and filling.

    Provides a complete audit trail for regulatory submissions,
    including all gaps found, methods used, and validation results.

    Attributes:
        report_id: Unique identifier for this report.
        series_id: Identifier of the reported time series.
        format: Report output format.
        gaps_detected: Total number of gaps detected.
        gaps_filled: Total number of gaps successfully filled.
        gaps_failed: Total number of gaps that failed to fill.
        methods_used: List of fill strategies applied.
        fill_pct: Fraction of detected gaps that were successfully filled.
        avg_fill_confidence: Mean confidence across all fills.
        validation_passed: Whether post-fill validation passed.
        regulatory_compliance: Whether the fill meets regulatory requirements.
        framework_references: Regulatory frameworks applicable to this report.
        impact_assessment: Associated impact assessment.
        generated_at: When the report was generated.
        content: Rendered report content (JSON, text, markdown, or HTML).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    series_id: str = Field(
        ..., description="Identifier of the reported time series",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Report output format",
    )
    gaps_detected: int = Field(
        default=0, ge=0,
        description="Total number of gaps detected",
    )
    gaps_filled: int = Field(
        default=0, ge=0,
        description="Total number of gaps successfully filled",
    )
    gaps_failed: int = Field(
        default=0, ge=0,
        description="Total number of gaps that failed to fill",
    )
    methods_used: List[str] = Field(
        default_factory=list,
        description="List of fill strategies applied",
    )
    fill_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of detected gaps successfully filled",
    )
    avg_fill_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean confidence across all fills",
    )
    validation_passed: bool = Field(
        default=False,
        description="Whether post-fill validation passed",
    )
    regulatory_compliance: bool = Field(
        default=False,
        description="Whether the fill meets regulatory requirements",
    )
    framework_references: List[str] = Field(
        default_factory=list,
        description="Regulatory frameworks applicable to this report",
    )
    impact_assessment: Optional[ImpactAssessment] = Field(
        None,
        description="Associated impact assessment",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="When the report was generated",
    )
    content: str = Field(
        default="",
        description="Rendered report content",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, v: str) -> str:
        """Validate series_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("series_id must be non-empty")
        return v


class PipelineConfig(BaseModel):
    """Configuration for the full gap-fill pipeline.

    Defines pipeline-level settings including fill strategy,
    validation, calendar, cross-series, and fallback preferences.

    Attributes:
        series_id: Identifier of the target time series.
        strategy: Default fill strategy for the pipeline.
        enable_validation: Whether to run post-fill validation.
        enable_calendar: Whether to apply calendar-aware logic.
        enable_cross_series: Whether to use cross-series filling.
        fallback_chain: Ordered list of fallback strategies.
        min_confidence: Minimum confidence for accepting a fill.
        max_gap_length: Maximum gap length to attempt filling.
        calendar_id: Identifier of the calendar to use.
        reference_series_ids: List of donor series identifiers.
        report_format: Output report format.
        auto_detect_frequency: Whether to auto-detect series frequency.
        seasonal_period: Override seasonal period (in data points).
        polynomial_degree: Degree for polynomial interpolation.
        validation_alpha: Significance level for KS validation test.
    """

    series_id: str = Field(
        default="",
        description="Identifier of the target time series",
    )
    strategy: FillStrategy = Field(
        default=FillStrategy.AUTO,
        description="Default fill strategy for the pipeline",
    )
    enable_validation: bool = Field(
        default=True,
        description="Whether to run post-fill validation",
    )
    enable_calendar: bool = Field(
        default=False,
        description="Whether to apply calendar-aware logic",
    )
    enable_cross_series: bool = Field(
        default=False,
        description="Whether to use cross-series filling",
    )
    fallback_chain: List[FillStrategy] = Field(
        default_factory=lambda: [FillStrategy.LINEAR, FillStrategy.NEAREST],
        description="Ordered list of fallback strategies",
    )
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for accepting a fill",
    )
    max_gap_length: int = Field(
        default=100, ge=1,
        description="Maximum gap length to attempt filling",
    )
    calendar_id: Optional[str] = Field(
        None,
        description="Identifier of the calendar to use",
    )
    reference_series_ids: List[str] = Field(
        default_factory=list,
        description="List of donor series identifiers",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output report format",
    )
    auto_detect_frequency: bool = Field(
        default=True,
        description="Whether to auto-detect series frequency",
    )
    seasonal_period: Optional[int] = Field(
        None, ge=2,
        description="Override seasonal period (in data points)",
    )
    polynomial_degree: int = Field(
        default=3, ge=1, le=10,
        description="Degree for polynomial interpolation",
    )
    validation_alpha: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Significance level for KS validation test",
    )

    model_config = {"extra": "forbid"}


class PipelineResult(BaseModel):
    """Complete result of a gap-fill pipeline run.

    Aggregates detection, characterisation, strategy selection,
    fill execution, and validation into a single pipeline output.

    Attributes:
        pipeline_id: Unique identifier for this pipeline run.
        job_id: Identifier of the parent gap-fill job.
        series_id: Identifier of the target time series.
        stage: Final pipeline stage reached.
        status: Final pipeline status.
        gaps_detected: Number of gaps detected.
        gaps_filled: Number of gaps filled.
        gaps_failed: Number of gaps that failed to fill.
        detection_result: Gap detection output.
        fill_results: List of fill results.
        validations: List of validation reports.
        strategies: List of selected strategies per gap.
        report: Generated gap-fill report.
        duration_ms: Total pipeline processing time in milliseconds.
        created_at: When the pipeline completed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this pipeline run",
    )
    job_id: str = Field(
        default="",
        description="Identifier of the parent gap-fill job",
    )
    series_id: str = Field(
        default="",
        description="Identifier of the target time series",
    )
    stage: PipelineStage = Field(
        default=PipelineStage.DOCUMENT,
        description="Final pipeline stage reached",
    )
    status: FillStatus = Field(
        default=FillStatus.COMPLETED,
        description="Final pipeline status",
    )
    gaps_detected: int = Field(
        default=0, ge=0,
        description="Number of gaps detected",
    )
    gaps_filled: int = Field(
        default=0, ge=0,
        description="Number of gaps filled",
    )
    gaps_failed: int = Field(
        default=0, ge=0,
        description="Number of gaps that failed to fill",
    )
    detection_result: Optional[GapDetectionResult] = Field(
        None,
        description="Gap detection output",
    )
    fill_results: List[FillResult] = Field(
        default_factory=list,
        description="List of fill results",
    )
    validations: List[ValidationReport] = Field(
        default_factory=list,
        description="List of validation reports",
    )
    strategies: List[GapFillStrategy] = Field(
        default_factory=list,
        description="List of selected strategies per gap",
    )
    report: Optional[GapFillReport] = Field(
        None,
        description="Generated gap-fill report",
    )
    duration_ms: float = Field(
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


class GapFillerJobConfig(BaseModel):
    """Configuration and tracking for a gap-fill job.

    Represents a single end-to-end gap-fill run with progress
    counters, timing, error tracking, and provenance for audit trail.

    Attributes:
        job_id: Unique identifier for this gap-fill job.
        series_name: Human-readable name for the target series.
        description: Description of the job purpose.
        status: Current job execution status.
        stage: Current pipeline stage being executed.
        strategy: Default fill strategy for this job.
        auto_detect_frequency: Whether to auto-detect series frequency.
        pipeline_config: Pipeline configuration for this job.
        total_points: Total number of data points in the series.
        gaps_detected: Total gaps detected so far.
        gaps_filled: Total gaps filled so far.
        gaps_failed: Total gaps that failed to fill so far.
        error_message: Error message if the job failed.
        started_at: Timestamp when the job started.
        completed_at: Timestamp when the job completed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this gap-fill job",
    )
    series_name: str = Field(
        default="",
        description="Human-readable name for the target series",
    )
    description: str = Field(
        default="",
        description="Description of the job purpose",
    )
    status: FillStatus = Field(
        default=FillStatus.PENDING,
        description="Current job execution status",
    )
    stage: PipelineStage = Field(
        default=PipelineStage.DETECT,
        description="Current pipeline stage being executed",
    )
    strategy: FillStrategy = Field(
        default=FillStrategy.AUTO,
        description="Default fill strategy for this job",
    )
    auto_detect_frequency: bool = Field(
        default=True,
        description="Whether to auto-detect series frequency",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None,
        description="Pipeline configuration for this job",
    )
    total_points: int = Field(
        default=0, ge=0,
        description="Total number of data points in the series",
    )
    gaps_detected: int = Field(
        default=0, ge=0,
        description="Total gaps detected so far",
    )
    gaps_filled: int = Field(
        default=0, ge=0,
        description="Total gaps filled so far",
    )
    gaps_failed: int = Field(
        default=0, ge=0,
        description="Total gaps that failed to fill so far",
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
            FillStatus.PENDING,
            FillStatus.IN_PROGRESS,
        )

    @property
    def progress_pct(self) -> float:
        """Return pipeline progress as a percentage (0.0 to 100.0)."""
        stage_progress = {
            PipelineStage.DETECT: 16.7,
            PipelineStage.CHARACTERIZE: 33.3,
            PipelineStage.SELECT_STRATEGY: 50.0,
            PipelineStage.FILL: 66.7,
            PipelineStage.VALIDATE: 83.3,
            PipelineStage.DOCUMENT: 100.0,
        }
        if self.status == FillStatus.COMPLETED:
            return 100.0
        if self.status in (FillStatus.FAILED, FillStatus.CANCELLED):
            return 0.0
        return stage_progress.get(self.stage, 0.0)


class GapFillerStatistics(BaseModel):
    """Aggregated operational statistics for the gap-fill service.

    Provides high-level metrics for monitoring the overall
    health, throughput, and effectiveness of the gap-fill pipeline.

    Attributes:
        total_jobs: Total number of gap-fill jobs executed.
        total_gaps_detected: Total gaps detected across all jobs.
        total_gaps_filled: Total gaps successfully filled.
        total_gaps_failed: Total gaps that failed to fill.
        avg_confidence: Average fill confidence across all jobs.
        active_jobs: Number of currently active jobs.
        by_status: Count of jobs per status.
        by_strategy: Count of fills per strategy.
        by_gap_type: Count of gaps per gap type.
        by_severity: Count of gaps per severity level.
        avg_fill_time_ms: Average fill processing time in milliseconds.
        timestamp: Timestamp when statistics were computed.
    """

    total_jobs: int = Field(
        default=0, ge=0,
        description="Total number of gap-fill jobs executed",
    )
    total_gaps_detected: int = Field(
        default=0, ge=0,
        description="Total gaps detected across all jobs",
    )
    total_gaps_filled: int = Field(
        default=0, ge=0,
        description="Total gaps successfully filled",
    )
    total_gaps_failed: int = Field(
        default=0, ge=0,
        description="Total gaps that failed to fill",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average fill confidence across all jobs",
    )
    active_jobs: int = Field(
        default=0, ge=0,
        description="Number of currently active jobs",
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of jobs per status",
    )
    by_strategy: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of fills per strategy",
    )
    by_gap_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of gaps per gap type",
    )
    by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of gaps per severity level",
    )
    avg_fill_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Average fill processing time in milliseconds",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when statistics were computed",
    )

    model_config = {"extra": "forbid"}


class SeriesMetadata(BaseModel):
    """Metadata for a registered time series.

    Contains descriptive information about a series, including
    length, frequency, regularity, gap profile, and temporal bounds.

    Attributes:
        metadata_id: Unique identifier for this metadata record.
        series_id: Identifier of the time series.
        name: Human-readable series name.
        length: Total number of data points in the series.
        frequency: Detected or declared frequency.
        resolution: Data resolution level.
        regularity: Regularity score (0.0-1.0).
        gap_pct: Fraction of points that are missing (0.0-1.0).
        gap_count: Number of distinct contiguous gaps.
        first_timestamp: Timestamp of the first data point.
        last_timestamp: Timestamp of the last data point.
        unit: Unit of measurement for the series values.
        source: Data source identifier.
        tags: Key-value tags for categorisation.
        created_at: When the metadata was registered.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    metadata_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this metadata record",
    )
    series_id: str = Field(
        ..., description="Identifier of the time series",
    )
    name: str = Field(
        default="",
        description="Human-readable series name",
    )
    length: int = Field(
        default=0, ge=0,
        description="Total number of data points in the series",
    )
    frequency: Optional[FrequencyLevel] = Field(
        None,
        description="Detected or declared frequency",
    )
    resolution: DataResolution = Field(
        default=DataResolution.RAW,
        description="Data resolution level",
    )
    regularity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Regularity score (0.0-1.0)",
    )
    gap_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of points that are missing (0.0-1.0)",
    )
    gap_count: int = Field(
        default=0, ge=0,
        description="Number of distinct contiguous gaps",
    )
    first_timestamp: Optional[str] = Field(
        None,
        description="Timestamp of the first data point",
    )
    last_timestamp: Optional[str] = Field(
        None,
        description="Timestamp of the last data point",
    )
    unit: str = Field(
        default="",
        description="Unit of measurement for the series values",
    )
    source: str = Field(
        default="",
        description="Data source identifier",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Key-value tags for categorisation",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the metadata was registered",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, v: str) -> str:
        """Validate series_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("series_id must be non-empty")
        return v


class BatchDetectionResult(BaseModel):
    """Batch result aggregating gap detection across multiple series.

    Attributes:
        batch_id: Unique identifier for this batch.
        series_count: Number of series analyzed in this batch.
        total_gaps: Total gaps detected across all series.
        total_missing: Total missing values across all series.
        per_series_results: Per-series gap detection results.
        processing_time_ms: Total batch processing time in milliseconds.
        created_at: When the batch was processed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this batch",
    )
    series_count: int = Field(
        default=0, ge=0,
        description="Number of series analyzed in this batch",
    )
    total_gaps: int = Field(
        default=0, ge=0,
        description="Total gaps detected across all series",
    )
    total_missing: int = Field(
        default=0, ge=0,
        description="Total missing values across all series",
    )
    per_series_results: List[GapDetectionResult] = Field(
        default_factory=list,
        description="Per-series gap detection results",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Total batch processing time in milliseconds",
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


class CrossSeriesResult(BaseModel):
    """Result of cross-series gap filling.

    Records the outcome of filling gaps in a target series
    using values from a correlated donor series.

    Attributes:
        result_id: Unique identifier for this cross-series result.
        target_id: Identifier of the target series with gaps.
        donor_id: Identifier of the donor series used for filling.
        correlation: Pearson correlation between target and donor.
        fills_count: Number of values filled from the donor.
        avg_confidence: Mean confidence across cross-series fills.
        min_confidence: Minimum confidence among cross-series fills.
        lag_applied: Lag offset applied to the donor series.
        scaling_factor: Scaling factor applied to donor values.
        fill_values: List of individual cross-series fill records.
        duration_ms: Cross-series fill processing time in milliseconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this cross-series result",
    )
    target_id: str = Field(
        ..., description="Identifier of the target series with gaps",
    )
    donor_id: str = Field(
        ..., description="Identifier of the donor series used for filling",
    )
    correlation: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Pearson correlation between target and donor",
    )
    fills_count: int = Field(
        default=0, ge=0,
        description="Number of values filled from the donor",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean confidence across cross-series fills",
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum confidence among cross-series fills",
    )
    lag_applied: int = Field(
        default=0,
        description="Lag offset applied to the donor series",
    )
    scaling_factor: float = Field(
        default=1.0,
        description="Scaling factor applied to donor values",
    )
    fill_values: List[FillValue] = Field(
        default_factory=list,
        description="List of individual cross-series fill records",
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0,
        description="Cross-series fill processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, v: str) -> str:
        """Validate target_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_id must be non-empty")
        return v

    @field_validator("donor_id")
    @classmethod
    def validate_donor_id(cls, v: str) -> str:
        """Validate donor_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("donor_id must be non-empty")
        return v


class SeasonalDecomposition(BaseModel):
    """Result of seasonal decomposition of a time series.

    Contains the decomposed components: trend, seasonal, and
    residual, along with the detected period.

    Attributes:
        decomposition_id: Unique identifier for this decomposition.
        series_id: Identifier of the decomposed time series.
        trend: Trend component values.
        seasonal: Seasonal component values.
        residual: Residual component values.
        period: Detected or specified seasonal period (in data points).
        seasonal_pattern: Classified seasonal pattern.
        trend_type: Detected trend type.
        trend_slope: Slope of the linear trend component.
        seasonal_strength: Strength of the seasonal component (0.0-1.0).
        trend_strength: Strength of the trend component (0.0-1.0).
        processing_time_ms: Decomposition processing time in milliseconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    decomposition_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this decomposition",
    )
    series_id: str = Field(
        default="",
        description="Identifier of the decomposed time series",
    )
    trend: List[Optional[float]] = Field(
        default_factory=list,
        description="Trend component values",
    )
    seasonal: List[float] = Field(
        default_factory=list,
        description="Seasonal component values",
    )
    residual: List[Optional[float]] = Field(
        default_factory=list,
        description="Residual component values",
    )
    period: int = Field(
        default=1, ge=1,
        description="Detected or specified seasonal period (in data points)",
    )
    seasonal_pattern: Optional[SeasonalPattern] = Field(
        None,
        description="Classified seasonal pattern",
    )
    trend_type: TrendType = Field(
        default=TrendType.NONE,
        description="Detected trend type",
    )
    trend_slope: float = Field(
        default=0.0,
        description="Slope of the linear trend component",
    )
    seasonal_strength: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Strength of the seasonal component (0.0-1.0)",
    )
    trend_strength: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Strength of the trend component (0.0-1.0)",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Decomposition processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class GapCharacterization(BaseModel):
    """Detailed characterisation of a single gap.

    Classifies the gap by type, periodicity, and severity, and
    provides contextual features used for strategy auto-selection.

    Attributes:
        characterization_id: Unique identifier for this characterisation.
        gap_id: Identifier of the characterised gap.
        gap_type: Classification of the gap pattern.
        periodicity_score: Score (0.0-1.0) indicating periodicity.
        severity: Severity classification.
        is_boundary: Whether the gap is at the start or end of the series.
        has_seasonal_context: Whether seasonal context is available around the gap.
        has_trend_context: Whether trend context is available around the gap.
        neighbor_variance: Variance of values immediately around the gap.
        neighbor_trend: Local trend direction around the gap (-1, 0, +1).
        recommended_strategy: Recommended fill strategy based on characteristics.
        features: Additional characterisation features for auto-selection.
        created_at: When the characterisation was computed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    characterization_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this characterisation",
    )
    gap_id: str = Field(
        ..., description="Identifier of the characterised gap",
    )
    gap_type: GapType = Field(
        default=GapType.RANDOM_GAP,
        description="Classification of the gap pattern",
    )
    periodicity_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Score indicating periodicity (0.0 = random, 1.0 = periodic)",
    )
    severity: GapSeverity = Field(
        default=GapSeverity.LOW,
        description="Severity classification",
    )
    is_boundary: bool = Field(
        default=False,
        description="Whether the gap is at the start or end of the series",
    )
    has_seasonal_context: bool = Field(
        default=False,
        description="Whether seasonal context is available around the gap",
    )
    has_trend_context: bool = Field(
        default=False,
        description="Whether trend context is available around the gap",
    )
    neighbor_variance: float = Field(
        default=0.0, ge=0.0,
        description="Variance of values immediately around the gap",
    )
    neighbor_trend: int = Field(
        default=0,
        description="Local trend direction around the gap (-1, 0, +1)",
    )
    recommended_strategy: FillStrategy = Field(
        default=FillStrategy.LINEAR,
        description="Recommended fill strategy based on characteristics",
    )
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional characterisation features for auto-selection",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the characterisation was computed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("gap_id")
    @classmethod
    def validate_gap_id(cls, v: str) -> str:
        """Validate gap_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("gap_id must be non-empty")
        return v

    @field_validator("neighbor_trend")
    @classmethod
    def validate_neighbor_trend(cls, v: int) -> int:
        """Validate neighbor_trend is -1, 0, or +1."""
        if v not in (-1, 0, 1):
            raise ValueError(
                f"neighbor_trend must be -1, 0, or 1, got {v}"
            )
        return v


# =============================================================================
# Request Models (8)
# =============================================================================


class CreateJobRequest(BaseModel):
    """Request body for creating a new gap-fill job.

    Attributes:
        series_name: Human-readable name for the target series.
        description: Description of the job purpose.
        values: List of numeric values (None for missing).
        timestamps: List of timestamp strings corresponding to values.
        strategy: Default fill strategy for the job.
        auto_detect_frequency: Whether to auto-detect series frequency.
        pipeline_config: Optional pipeline configuration.
    """

    series_name: str = Field(
        default="",
        description="Human-readable name for the target series",
    )
    description: str = Field(
        default="",
        description="Description of the job purpose",
    )
    values: List[Optional[float]] = Field(
        ..., min_length=2,
        description="List of numeric values (None for missing)",
    )
    timestamps: List[str] = Field(
        default_factory=list,
        description="List of timestamp strings corresponding to values",
    )
    strategy: FillStrategy = Field(
        default=FillStrategy.AUTO,
        description="Default fill strategy for the job",
    )
    auto_detect_frequency: bool = Field(
        default=True,
        description="Whether to auto-detect series frequency",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None,
        description="Optional pipeline configuration",
    )

    model_config = {"extra": "forbid"}


class DetectGapsRequest(BaseModel):
    """Request body for detecting gaps in a single time series.

    Attributes:
        series_id: Identifier for the time series.
        values: List of numeric values (None for missing).
        timestamps: List of timestamp strings corresponding to values.
        expected_frequency: Expected frequency (if known).
        min_gap_length: Minimum consecutive missing points to count as a gap.
    """

    series_id: str = Field(
        default="",
        description="Identifier for the time series",
    )
    values: List[Optional[float]] = Field(
        ..., min_length=1,
        description="List of numeric values (None for missing)",
    )
    timestamps: List[str] = Field(
        default_factory=list,
        description="List of timestamp strings corresponding to values",
    )
    expected_frequency: Optional[FrequencyLevel] = Field(
        None,
        description="Expected frequency (if known)",
    )
    min_gap_length: int = Field(
        default=1, ge=1,
        description="Minimum consecutive missing points to count as a gap",
    )

    model_config = {"extra": "forbid"}


class BatchDetectRequest(BaseModel):
    """Request body for batch gap detection across multiple series.

    Attributes:
        series_list: List of per-series value lists (None for missing).
        series_ids: List of series identifiers.
        shared_frequency: Shared expected frequency for all series.
    """

    series_list: List[List[Optional[float]]] = Field(
        ..., min_length=1,
        description="List of per-series value lists (None for missing)",
    )
    series_ids: List[str] = Field(
        default_factory=list,
        description="List of series identifiers",
    )
    shared_frequency: Optional[FrequencyLevel] = Field(
        None,
        description="Shared expected frequency for all series",
    )

    model_config = {"extra": "forbid"}


class AnalyzeFrequencyRequest(BaseModel):
    """Request body for analyzing the frequency of a time series.

    Attributes:
        series_id: Identifier for the time series.
        values: List of numeric values (None for missing).
        timestamps: List of timestamp strings corresponding to values.
        max_sample_size: Maximum number of intervals to sample.
    """

    series_id: str = Field(
        default="",
        description="Identifier for the time series",
    )
    values: List[Optional[float]] = Field(
        default_factory=list,
        description="List of numeric values (None for missing)",
    )
    timestamps: List[str] = Field(
        ..., min_length=2,
        description="List of timestamp strings corresponding to values",
    )
    max_sample_size: int = Field(
        default=1000, ge=2,
        description="Maximum number of intervals to sample",
    )

    model_config = {"extra": "forbid"}


class FillGapsRequest(BaseModel):
    """Request body for filling gaps in a time series.

    Attributes:
        series_id: Identifier for the time series.
        values: List of numeric values (None for missing).
        timestamps: List of timestamp strings corresponding to values.
        strategy: Fill strategy to use.
        fallback_chain: Ordered fallback strategies.
        min_confidence: Minimum confidence for accepting a fill.
        max_gap_length: Maximum gap length to attempt filling.
        seasonal_period: Override seasonal period (if applicable).
        polynomial_degree: Polynomial degree (if applicable).
        reference_series: List of donor series for cross-series fill.
        calendar_id: Calendar identifier for calendar-aware fill.
    """

    series_id: str = Field(
        default="",
        description="Identifier for the time series",
    )
    values: List[Optional[float]] = Field(
        ..., min_length=2,
        description="List of numeric values (None for missing)",
    )
    timestamps: List[str] = Field(
        default_factory=list,
        description="List of timestamp strings corresponding to values",
    )
    strategy: FillStrategy = Field(
        default=FillStrategy.AUTO,
        description="Fill strategy to use",
    )
    fallback_chain: List[FillStrategy] = Field(
        default_factory=list,
        description="Ordered fallback strategies",
    )
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence for accepting a fill",
    )
    max_gap_length: int = Field(
        default=100, ge=1,
        description="Maximum gap length to attempt filling",
    )
    seasonal_period: Optional[int] = Field(
        None, ge=2,
        description="Override seasonal period (if applicable)",
    )
    polynomial_degree: int = Field(
        default=3, ge=1, le=10,
        description="Polynomial degree (if applicable)",
    )
    reference_series: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of donor series for cross-series fill",
    )
    calendar_id: Optional[str] = Field(
        None,
        description="Calendar identifier for calendar-aware fill",
    )

    model_config = {"extra": "forbid"}


class ValidateFillsRequest(BaseModel):
    """Request body for validating filled values.

    Attributes:
        series_id: Identifier for the time series.
        original_values: Original values (with missing as None).
        filled_values: Filled values (no missing).
        timestamps: List of timestamp strings corresponding to values.
        fill_indices: List of indices that were filled.
        ground_truth: Ground truth values for accuracy assessment (if available).
        alpha: Significance level for the KS test.
    """

    series_id: str = Field(
        default="",
        description="Identifier for the time series",
    )
    original_values: List[Optional[float]] = Field(
        ..., min_length=1,
        description="Original values (with missing as None)",
    )
    filled_values: List[float] = Field(
        ..., min_length=1,
        description="Filled values (no missing)",
    )
    timestamps: List[str] = Field(
        default_factory=list,
        description="List of timestamp strings corresponding to values",
    )
    fill_indices: List[int] = Field(
        default_factory=list,
        description="List of indices that were filled",
    )
    ground_truth: List[Optional[float]] = Field(
        default_factory=list,
        description="Ground truth values for accuracy assessment (if available)",
    )
    alpha: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Significance level for the KS test",
    )

    model_config = {"extra": "forbid"}


class CreateCalendarRequest(BaseModel):
    """Request body for creating a calendar definition.

    Attributes:
        name: Human-readable calendar name.
        calendar_type: Type of calendar.
        business_days: List of ISO weekday numbers that are business days.
        holidays: List of holiday date strings (YYYY-MM-DD format).
        fiscal_start_month: Month number (1-12) for fiscal year start.
        timezone: IANA timezone string.
        description: Human-readable description of the calendar.
    """

    name: str = Field(
        ..., description="Human-readable calendar name",
    )
    calendar_type: CalendarType = Field(
        default=CalendarType.BUSINESS_DAYS,
        description="Type of calendar",
    )
    business_days: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5],
        description="List of ISO weekday numbers that are business days",
    )
    holidays: List[str] = Field(
        default_factory=list,
        description="List of holiday date strings (YYYY-MM-DD format)",
    )
    fiscal_start_month: int = Field(
        default=1, ge=1, le=12,
        description="Month number (1-12) for fiscal year start",
    )
    timezone: str = Field(
        default="UTC",
        description="IANA timezone string",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the calendar",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("business_days")
    @classmethod
    def validate_business_days(cls, v: List[int]) -> List[int]:
        """Validate business_days contains valid ISO weekday numbers."""
        for day in v:
            if day < 1 or day > 7:
                raise ValueError(
                    f"business_days must contain values 1-7, got {day}"
                )
        return v


class RunPipelineRequest(BaseModel):
    """Request body for running the full gap-fill pipeline.

    Attributes:
        series_id: Identifier for the time series.
        values: List of numeric values (None for missing).
        timestamps: List of timestamp strings corresponding to values.
        pipeline_config: Optional pipeline configuration.
        reference_series: List of donor series for cross-series fill.
        calendar_id: Calendar identifier for calendar-aware fill.
        options: Optional overrides.
    """

    series_id: str = Field(
        default="",
        description="Identifier for the time series",
    )
    values: List[Optional[float]] = Field(
        ..., min_length=2,
        description="List of numeric values (None for missing)",
    )
    timestamps: List[str] = Field(
        default_factory=list,
        description="List of timestamp strings corresponding to values",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None,
        description="Optional pipeline configuration",
    )
    reference_series: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of donor series for cross-series fill",
    )
    calendar_id: Optional[str] = Field(
        None,
        description="Calendar identifier for calendar-aware fill",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional overrides",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Engine-internal models (used by InterpolationEngine & TrendExtrapolatorEngine)
# =============================================================================


class FillMethod(str, Enum):
    """Specific algorithm used by an engine to fill gaps.

    Unlike FillStrategy (which is the user-facing strategy selection),
    FillMethod identifies the concrete algorithm applied by the engine
    (e.g. LINEAR_TREND, EXPONENTIAL_SMOOTHING, HOLT_WINTERS).
    """

    LINEAR = "linear"
    LINEAR_TREND = "linear_trend"
    CUBIC_SPLINE = "cubic_spline"
    POLYNOMIAL = "polynomial"
    AKIMA = "akima"
    PCHIP = "pchip"
    NEAREST = "nearest"
    TIME_WEIGHTED = "time_weighted"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    DOUBLE_EXPONENTIAL = "double_exponential"
    HOLT_WINTERS = "holt_winters"
    MOVING_AVERAGE = "moving_average"
    SEASONAL = "seasonal"
    CROSS_SERIES = "cross_series"
    CALENDAR = "calendar"


class FillPoint(BaseModel):
    """A single point produced during engine-level gap filling.

    Attributes:
        index: Position index in the original series.
        original_value: Original value at this position (None if missing).
        filled_value: Value produced by the fill (may equal original if not missing).
        was_missing: Whether this point was a missing value that was filled.
        method: The concrete FillMethod applied.
        confidence: Confidence in the filled value (0.0-1.0).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    index: int = Field(..., ge=0, description="Position index in the series")
    original_value: Optional[float] = Field(
        None, description="Original value (None if missing)",
    )
    filled_value: float = Field(
        ..., description="Value produced by the fill operation",
    )
    was_missing: bool = Field(
        default=True, description="Whether this point was missing",
    )
    method: FillMethod = Field(
        ..., description="Concrete fill algorithm applied",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the filled value (0.0-1.0)",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}


class TrendAnalysis(BaseModel):
    """Result of full trend analysis on a time series.

    Returned by TrendExtrapolatorEngine.analyze_trend(), providing
    OLS regression parameters and the classified trend type.

    Attributes:
        trend_type: Classified trend type.
        slope: OLS regression slope coefficient.
        intercept: OLS regression intercept.
        r_squared: Coefficient of determination (R-squared).
        confidence: Overall confidence in the analysis (0.0-1.0).
        series_length: Total number of data points in the series.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    trend_type: TrendType = Field(
        default=TrendType.UNKNOWN,
        description="Classified trend type",
    )
    slope: float = Field(default=0.0, description="OLS regression slope")
    intercept: float = Field(default=0.0, description="OLS regression intercept")
    r_squared: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Coefficient of determination",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall confidence (0.0-1.0)",
    )
    series_length: int = Field(
        default=0, ge=0,
        description="Total number of data points in the series",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports
    # -------------------------------------------------------------------------
    "TimeSeriesImputerEngine",
    "ImputedValue_L1",
    "ConfidenceLevel_L1",
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "DEFAULT_STRATEGY_WEIGHTS",
    "GAP_SEVERITY_THRESHOLDS",
    "FILL_CONFIDENCE_THRESHOLDS",
    "PIPELINE_STAGE_ORDER",
    "FILL_STRATEGIES",
    "GAP_TYPES",
    "REPORT_FORMAT_OPTIONS",
    "FREQUENCY_STRINGS",
    "CALENDAR_TYPES",
    "RESOLUTION_LEVELS",
    # -------------------------------------------------------------------------
    # Enumerations (13)
    # -------------------------------------------------------------------------
    "FrequencyLevel",
    "GapType",
    "GapSeverity",
    "FillStrategy",
    "FillStatus",
    "ValidationResult",
    "CalendarType",
    "SeasonalPattern",
    "TrendType",
    "PipelineStage",
    "ReportFormat",
    "ConfidenceLevel",
    "DataResolution",
    # -------------------------------------------------------------------------
    # SDK data models (20)
    # -------------------------------------------------------------------------
    "GapRecord",
    "GapDetectionResult",
    "FrequencyResult",
    "FillValue",
    "FillResult",
    "ValidationReport",
    "CalendarDefinition",
    "ReferenceSeries",
    "GapFillStrategy",
    "ImpactAssessment",
    "GapFillReport",
    "PipelineConfig",
    "PipelineResult",
    "GapFillerJobConfig",
    "GapFillerStatistics",
    "SeriesMetadata",
    "BatchDetectionResult",
    "CrossSeriesResult",
    "SeasonalDecomposition",
    "GapCharacterization",
    # -------------------------------------------------------------------------
    # Engine-internal models (3)
    # -------------------------------------------------------------------------
    "FillMethod",
    "FillPoint",
    "TrendAnalysis",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "CreateJobRequest",
    "DetectGapsRequest",
    "BatchDetectRequest",
    "AnalyzeFrequencyRequest",
    "FillGapsRequest",
    "ValidateFillsRequest",
    "CreateCalendarRequest",
    "RunPipelineRequest",
]
