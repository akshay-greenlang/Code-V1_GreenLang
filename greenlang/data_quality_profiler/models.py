# -*- coding: utf-8 -*-
"""
Data Quality Profiler Service Data Models - AGENT-DATA-010

Pydantic v2 data models for the Data Quality Profiler SDK. Re-exports the
Layer 1 enumerations and models from the Excel/CSV Normalizer's
DataQualityScorer (QualityLevel, DataQualityReport, DataQualityScorer),
and defines additional SDK models for dataset profiling, quality assessment,
anomaly detection, freshness checking, rule evaluation, quality gates,
trend tracking, scorecards, reporting, and batch management.

Re-exported Layer 1 sources:
    - greenlang.excel_normalizer.data_quality_scorer: QualityLevel
        (as L1QualityLevel), DataQualityReport (as L1DataQualityReport),
        DataQualityScorer (as L1DataQualityScorer)

New enumerations (13):
    - QualityDimension, DataType, ProfileStatus, AssessmentStatus,
      AnomalyMethod, AnomalySeverity, RuleType, RuleOperator,
      GateOutcome, IssueSeverity, MissingPattern, ReportFormat,
      TrendDirection

New SDK models (15):
    - ColumnProfile, DatasetProfile, DimensionScore, QualityAssessment,
      QualityIssue, AnomalyResult, FreshnessResult, QualityRule,
      RuleEvaluation, QualityGate, QualityTrend, QualityScorecardRow,
      QualityScorecard, DataQualityProfilerStatistics, ProfileSummary

Request models (7):
    - ProfileDatasetRequest, AssessQualityRequest, ValidateDatasetRequest,
      DetectAnomaliesRequest, CheckFreshnessRequest, CreateRuleRequest,
      GenerateReportRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from excel_normalizer.data_quality_scorer
# ---------------------------------------------------------------------------

from greenlang.excel_normalizer.data_quality_scorer import (
    QualityLevel as L1QualityLevel,
    DataQualityReport as L1DataQualityReport,
    DataQualityScorer as L1DataQualityScorer,
)

# Canonical re-exports for downstream usage
QualityLevel = L1QualityLevel
DataQualityReport = L1DataQualityReport
DataQualityScorer = L1DataQualityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default weights for six quality dimensions (sum = 1.0).
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "completeness": 0.20,
    "validity": 0.20,
    "consistency": 0.20,
    "timeliness": 0.15,
    "uniqueness": 0.15,
    "accuracy": 0.10,
}

#: Score boundaries for quality level classification.
QUALITY_LEVEL_THRESHOLDS: Dict[str, float] = {
    "excellent": 0.95,
    "good": 0.80,
    "fair": 0.60,
    "poor": 0.40,
    "critical": 0.00,
}

#: Default anomaly detection thresholds per method.
DEFAULT_ANOMALY_THRESHOLDS: Dict[str, float] = {
    "iqr": 1.5,
    "zscore": 3.0,
    "mad": 3.5,
    "grubbs": 0.05,
    "modified_zscore": 3.5,
}

#: Freshness level boundaries in hours.
FRESHNESS_BOUNDARIES_HOURS: Dict[str, int] = {
    "excellent": 24,
    "good": 72,
    "fair": 168,
    "poor": 720,
}

#: All six quality dimensions as a tuple for iteration.
ALL_QUALITY_DIMENSIONS: tuple = (
    "completeness",
    "validity",
    "consistency",
    "timeliness",
    "uniqueness",
    "accuracy",
)

#: Supported data types for column profiling.
SUPPORTED_DATA_TYPES: tuple = (
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "datetime",
    "email",
    "url",
    "phone",
    "ip_address",
    "uuid",
    "json",
    "unknown",
)

#: Available anomaly detection method names.
ANOMALY_METHOD_NAMES: tuple = ("iqr", "zscore", "mad", "grubbs", "modified_zscore")

#: Available report format options.
REPORT_FORMAT_OPTIONS: tuple = ("json", "markdown", "html", "text", "csv")

#: Available gate outcome values.
GATE_OUTCOME_VALUES: tuple = ("pass", "warn", "fail")

#: Issue severity levels ordered by increasing severity.
ISSUE_SEVERITY_ORDER: tuple = ("info", "warning", "error", "critical")


# =============================================================================
# New Enumerations (13)
# =============================================================================


class QualityDimension(str, Enum):
    """Quality dimensions for data quality assessment.

    Follows the six-dimension model used by DAMA-DMBOK and ISO 8000
    for systematic data quality measurement.
    """

    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    ACCURACY = "accuracy"


class DataType(str, Enum):
    """Detected or declared data types for column profiling.

    Covers primitive types, structured formats, and special types
    used in climate and sustainability datasets.
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    UUID = "uuid"
    JSON = "json"
    UNKNOWN = "unknown"


class ProfileStatus(str, Enum):
    """Lifecycle status of a dataset profiling operation.

    Tracks the current state of a profiling job from submission
    through completion or failure.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AssessmentStatus(str, Enum):
    """Lifecycle status of a quality assessment operation.

    Extends ProfileStatus with PASSED, WARNING, and FAILED outcomes
    reflecting assessment results rather than just execution state.
    """

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class AnomalyMethod(str, Enum):
    """Statistical methods for outlier and anomaly detection.

    Each method has different sensitivity characteristics and
    assumptions about the underlying data distribution.
    """

    IQR = "iqr"
    ZSCORE = "zscore"
    MAD = "mad"
    GRUBBS = "grubbs"
    MODIFIED_ZSCORE = "modified_zscore"


class AnomalySeverity(str, Enum):
    """Severity classification for detected anomalies.

    Determines the urgency and impact of anomalous values found
    during data quality analysis.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleType(str, Enum):
    """Types of data quality validation rules.

    Categorizes rules by the quality dimension or constraint
    they enforce on dataset columns or rows.
    """

    COMPLETENESS = "completeness"
    RANGE = "range"
    FORMAT = "format"
    UNIQUENESS = "uniqueness"
    CUSTOM = "custom"
    FRESHNESS = "freshness"


class RuleOperator(str, Enum):
    """Comparison operators for quality rule evaluation.

    Defines how rule thresholds are compared against actual
    data values during validation.
    """

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    BETWEEN = "between"
    MATCHES = "matches"
    CONTAINS = "contains"
    IN_SET = "in_set"


class GateOutcome(str, Enum):
    """Outcome of a quality gate evaluation.

    Quality gates act as go/no-go checkpoints in data pipelines,
    determining whether data meets minimum quality thresholds.
    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class IssueSeverity(str, Enum):
    """Severity classification for quality issues.

    Determines the visibility and required action for each
    quality issue detected during assessment.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MissingPattern(str, Enum):
    """Classification of missing data patterns.

    Based on Rubin's taxonomy of missing data mechanisms, used
    to inform imputation strategy recommendations.

    MCAR: Missing Completely At Random - no systematic pattern.
    MAR: Missing At Random - related to observed data.
    MNAR: Missing Not At Random - related to unobserved data.
    UNKNOWN: Pattern has not been determined.
    """

    MCAR = "mcar"
    MAR = "mar"
    MNAR = "mnar"
    UNKNOWN = "unknown"


class ReportFormat(str, Enum):
    """Output format for quality profiler reports.

    Defines the serialization format for generated reports
    including profiling summaries and assessment outputs.
    """

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"


class TrendDirection(str, Enum):
    """Direction of quality score trend over time.

    Classifies whether data quality is improving, stable,
    or degrading across successive profiling runs.
    """

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


# =============================================================================
# SDK Data Models (15)
# =============================================================================


class ColumnProfile(BaseModel):
    """Statistical profile of a single column in a dataset.

    Contains descriptive statistics, type information, null analysis,
    cardinality metrics, and distribution summaries for a column.

    Attributes:
        name: Column name or header label.
        data_type: Detected or declared data type for the column.
        total: Total number of values (rows) in the column.
        non_null: Count of non-null, non-empty values.
        null_count: Count of null or missing values.
        null_pct: Percentage of null values (0.0 to 100.0).
        unique_count: Number of distinct non-null values.
        cardinality: Cardinality ratio (unique / non_null, 0.0 to 1.0).
        min_val: Minimum value (as string for universal representation).
        max_val: Maximum value (as string for universal representation).
        mean: Arithmetic mean for numeric columns.
        median: Median value for numeric columns.
        stddev: Standard deviation for numeric columns.
        p25: 25th percentile for numeric columns.
        p50: 50th percentile (same as median) for numeric columns.
        p75: 75th percentile for numeric columns.
        p95: 95th percentile for numeric columns.
        p99: 99th percentile for numeric columns.
        most_common: List of (value, count) pairs for most frequent values.
        pattern: Detected regex or format pattern for string columns.
        missing_pattern: Classification of missing data mechanism.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    name: str = Field(
        ..., description="Column name or header label",
    )
    data_type: DataType = Field(
        default=DataType.UNKNOWN,
        description="Detected or declared data type for the column",
    )
    total: int = Field(
        default=0, ge=0,
        description="Total number of values (rows) in the column",
    )
    non_null: int = Field(
        default=0, ge=0,
        description="Count of non-null, non-empty values",
    )
    null_count: int = Field(
        default=0, ge=0,
        description="Count of null or missing values",
    )
    null_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of null values (0.0 to 100.0)",
    )
    unique_count: int = Field(
        default=0, ge=0,
        description="Number of distinct non-null values",
    )
    cardinality: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Cardinality ratio (unique / non_null, 0.0 to 1.0)",
    )
    min_val: Optional[str] = Field(
        None,
        description="Minimum value (as string for universal representation)",
    )
    max_val: Optional[str] = Field(
        None,
        description="Maximum value (as string for universal representation)",
    )
    mean: Optional[float] = Field(
        None, description="Arithmetic mean for numeric columns",
    )
    median: Optional[float] = Field(
        None, description="Median value for numeric columns",
    )
    stddev: Optional[float] = Field(
        None, ge=0.0,
        description="Standard deviation for numeric columns",
    )
    p25: Optional[float] = Field(
        None, description="25th percentile for numeric columns",
    )
    p50: Optional[float] = Field(
        None, description="50th percentile for numeric columns",
    )
    p75: Optional[float] = Field(
        None, description="75th percentile for numeric columns",
    )
    p95: Optional[float] = Field(
        None, description="95th percentile for numeric columns",
    )
    p99: Optional[float] = Field(
        None, description="99th percentile for numeric columns",
    )
    most_common: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {value, count} dicts for most frequent values",
    )
    pattern: Optional[str] = Field(
        None,
        description="Detected regex or format pattern for string columns",
    )
    missing_pattern: MissingPattern = Field(
        default=MissingPattern.UNKNOWN,
        description="Classification of missing data mechanism",
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


class DatasetProfile(BaseModel):
    """Complete statistical profile of a dataset.

    Aggregates per-column profiles with dataset-level metadata
    including row/column counts, schema hash, memory estimate,
    and profiling timestamps for reproducibility.

    Attributes:
        profile_id: Unique identifier for this profiling run.
        dataset_name: Name or identifier of the profiled dataset.
        row_count: Total number of rows in the dataset.
        column_count: Total number of columns in the dataset.
        columns: List of per-column statistical profiles.
        schema_hash: SHA-256 hash of the column names and types.
        memory_estimate_bytes: Estimated memory footprint in bytes.
        created_at: Timestamp when profiling was initiated.
        profiled_at: Timestamp when profiling completed.
        status: Current profiling job status.
        profiling_duration_ms: Profiling execution time in milliseconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    profile_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this profiling run",
    )
    dataset_name: str = Field(
        ..., description="Name or identifier of the profiled dataset",
    )
    row_count: int = Field(
        default=0, ge=0,
        description="Total number of rows in the dataset",
    )
    column_count: int = Field(
        default=0, ge=0,
        description="Total number of columns in the dataset",
    )
    columns: List[ColumnProfile] = Field(
        default_factory=list,
        description="List of per-column statistical profiles",
    )
    schema_hash: str = Field(
        default="",
        description="SHA-256 hash of the column names and types",
    )
    memory_estimate_bytes: int = Field(
        default=0, ge=0,
        description="Estimated memory footprint in bytes",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when profiling was initiated",
    )
    profiled_at: Optional[datetime] = Field(
        None,
        description="Timestamp when profiling completed",
    )
    status: ProfileStatus = Field(
        default=ProfileStatus.PENDING,
        description="Current profiling job status",
    )
    profiling_duration_ms: float = Field(
        default=0.0, ge=0.0,
        description="Profiling execution time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


class QualityIssue(BaseModel):
    """A single data quality issue detected during assessment.

    Represents a specific quality problem found in a dataset column
    or row, with severity classification, description, and optional
    remediation guidance.

    Attributes:
        issue_id: Unique identifier for this quality issue.
        severity: Issue severity classification.
        dimension: Quality dimension this issue relates to.
        column: Column name where the issue was detected.
        row_index: Row index where the issue was detected (if applicable).
        description: Human-readable description of the issue.
        suggested_fix: Recommended remediation action.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    issue_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this quality issue",
    )
    severity: IssueSeverity = Field(
        default=IssueSeverity.WARNING,
        description="Issue severity classification",
    )
    dimension: QualityDimension = Field(
        ..., description="Quality dimension this issue relates to",
    )
    column: Optional[str] = Field(
        None, description="Column name where the issue was detected",
    )
    row_index: Optional[int] = Field(
        None, ge=0,
        description="Row index where the issue was detected (if applicable)",
    )
    description: str = Field(
        ..., description="Human-readable description of the issue",
    )
    suggested_fix: Optional[str] = Field(
        None, description="Recommended remediation action",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is non-empty."""
        if not v or not v.strip():
            raise ValueError("description must be non-empty")
        return v


class DimensionScore(BaseModel):
    """Score for a single quality dimension within an assessment.

    Contains the raw score, weight, weighted contribution, detail
    metadata, and issue count for one quality dimension.

    Attributes:
        dimension: Quality dimension being scored.
        score: Raw dimension score (0.0 to 1.0).
        weight: Weight applied to this dimension in overall score.
        weighted_score: Weighted contribution (score * weight).
        details: Dimension-specific detail metadata.
        issues_count: Number of issues found for this dimension.
    """

    dimension: QualityDimension = Field(
        ..., description="Quality dimension being scored",
    )
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw dimension score (0.0 to 1.0)",
    )
    weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weight applied to this dimension in overall score",
    )
    weighted_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted contribution (score * weight)",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dimension-specific detail metadata",
    )
    issues_count: int = Field(
        default=0, ge=0,
        description="Number of issues found for this dimension",
    )

    model_config = {"extra": "forbid"}


class QualityAssessment(BaseModel):
    """Complete quality assessment result for a dataset.

    Aggregates dimension scores, overall score, quality level
    classification, and all detected issues for a single
    assessment run.

    Attributes:
        assessment_id: Unique identifier for this assessment.
        dataset_name: Name or identifier of the assessed dataset.
        overall_score: Weighted overall quality score (0.0 to 1.0).
        quality_level: Quality level classification.
        dimensions: List of per-dimension score breakdowns.
        total_issues: Total number of quality issues detected.
        issues: List of quality issues found during assessment.
        status: Current assessment status.
        created_at: Timestamp when assessment was initiated.
        completed_at: Timestamp when assessment completed.
        assessment_duration_ms: Assessment execution time in milliseconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    assessment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this assessment",
    )
    dataset_name: str = Field(
        ..., description="Name or identifier of the assessed dataset",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted overall quality score (0.0 to 1.0)",
    )
    quality_level: QualityLevel = Field(
        default=QualityLevel.CRITICAL,
        description="Quality level classification",
    )
    dimensions: List[DimensionScore] = Field(
        default_factory=list,
        description="List of per-dimension score breakdowns",
    )
    total_issues: int = Field(
        default=0, ge=0,
        description="Total number of quality issues detected",
    )
    issues: List[QualityIssue] = Field(
        default_factory=list,
        description="List of quality issues found during assessment",
    )
    status: AssessmentStatus = Field(
        default=AssessmentStatus.PENDING,
        description="Current assessment status",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when assessment was initiated",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when assessment completed",
    )
    assessment_duration_ms: float = Field(
        default=0.0, ge=0.0,
        description="Assessment execution time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


class AnomalyResult(BaseModel):
    """Result of anomaly detection for a single column value or set.

    Contains the anomalous value, expected range, detection method,
    statistical scores, severity, and affected row indices.

    Attributes:
        anomaly_id: Unique identifier for this anomaly.
        column: Column name where the anomaly was detected.
        method: Statistical method used for detection.
        value: The anomalous value detected.
        expected_range: Expected value range as [lower, upper] string.
        z_score: Z-score of the anomalous value (if applicable).
        severity: Anomaly severity classification.
        description: Human-readable description of the anomaly.
        row_indices: List of row indices where the anomaly occurs.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    anomaly_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this anomaly",
    )
    column: str = Field(
        ..., description="Column name where the anomaly was detected",
    )
    method: AnomalyMethod = Field(
        ..., description="Statistical method used for detection",
    )
    value: Optional[float] = Field(
        None, description="The anomalous value detected",
    )
    expected_range: str = Field(
        default="",
        description="Expected value range as '[lower, upper]' string",
    )
    z_score: Optional[float] = Field(
        None, description="Z-score of the anomalous value (if applicable)",
    )
    severity: AnomalySeverity = Field(
        default=AnomalySeverity.MEDIUM,
        description="Anomaly severity classification",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the anomaly",
    )
    row_indices: List[int] = Field(
        default_factory=list,
        description="List of row indices where the anomaly occurs",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("column")
    @classmethod
    def validate_column(cls, v: str) -> str:
        """Validate column is non-empty."""
        if not v or not v.strip():
            raise ValueError("column must be non-empty")
        return v


class FreshnessResult(BaseModel):
    """Result of a dataset freshness / timeliness check.

    Evaluates how recently a dataset was updated relative to
    configured freshness thresholds and SLA requirements.

    Attributes:
        dataset_name: Name or identifier of the checked dataset.
        last_updated: ISO-8601 timestamp of the last dataset update.
        age_hours: Age of the dataset in hours since last update.
        freshness_level: Quality level based on freshness thresholds.
        sla_hours: SLA deadline in hours for this dataset.
        sla_compliant: Whether the dataset meets the SLA deadline.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    dataset_name: str = Field(
        ..., description="Name or identifier of the checked dataset",
    )
    last_updated: str = Field(
        ...,
        description="ISO-8601 timestamp of the last dataset update",
    )
    age_hours: float = Field(
        ..., ge=0.0,
        description="Age of the dataset in hours since last update",
    )
    freshness_level: QualityLevel = Field(
        ..., description="Quality level based on freshness thresholds",
    )
    sla_hours: int = Field(
        ..., ge=0,
        description="SLA deadline in hours for this dataset",
    )
    sla_compliant: bool = Field(
        ..., description="Whether the dataset meets the SLA deadline",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v

    @field_validator("last_updated")
    @classmethod
    def validate_last_updated(cls, v: str) -> str:
        """Validate last_updated is non-empty."""
        if not v or not v.strip():
            raise ValueError("last_updated must be non-empty")
        return v


class QualityRule(BaseModel):
    """A data quality validation rule for automated checks.

    Defines a constraint or expectation on a dataset column
    that is evaluated during quality validation. Rules can
    enforce completeness thresholds, value ranges, format
    patterns, uniqueness constraints, and custom conditions.

    Attributes:
        rule_id: Unique identifier for this rule.
        name: Human-readable rule name.
        description: Detailed description of the rule purpose.
        rule_type: Type of quality check this rule performs.
        column: Target column name (None for dataset-level rules).
        operator: Comparison operator for threshold evaluation.
        threshold: Numeric threshold for the rule evaluation.
        parameters: Additional rule parameters as key-value pairs.
        active: Whether this rule is currently active.
        priority: Evaluation priority (lower = higher priority).
        created_at: Timestamp when the rule was created.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    rule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this rule",
    )
    name: str = Field(
        ..., description="Human-readable rule name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the rule purpose",
    )
    rule_type: RuleType = Field(
        ..., description="Type of quality check this rule performs",
    )
    column: Optional[str] = Field(
        None,
        description="Target column name (None for dataset-level rules)",
    )
    operator: RuleOperator = Field(
        default=RuleOperator.GREATER_THAN,
        description="Comparison operator for threshold evaluation",
    )
    threshold: Optional[float] = Field(
        None,
        description="Numeric threshold for the rule evaluation",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional rule parameters as key-value pairs",
    )
    active: bool = Field(
        default=True,
        description="Whether this rule is currently active",
    )
    priority: int = Field(
        default=100, ge=0,
        description="Evaluation priority (lower = higher priority)",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the rule was created",
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


class RuleEvaluation(BaseModel):
    """Result of evaluating a single quality rule against a dataset.

    Contains the pass/fail outcome, actual measured value,
    threshold comparison, and explanatory message.

    Attributes:
        evaluation_id: Unique identifier for this evaluation.
        rule_id: Identifier of the rule that was evaluated.
        rule_name: Human-readable name of the evaluated rule.
        passed: Whether the rule evaluation passed.
        actual_value: Actual measured value from the dataset.
        threshold: Threshold the actual value was compared against.
        message: Human-readable evaluation result message.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    evaluation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this evaluation",
    )
    rule_id: str = Field(
        ..., description="Identifier of the rule that was evaluated",
    )
    rule_name: str = Field(
        default="",
        description="Human-readable name of the evaluated rule",
    )
    passed: bool = Field(
        ..., description="Whether the rule evaluation passed",
    )
    actual_value: Optional[float] = Field(
        None, description="Actual measured value from the dataset",
    )
    threshold: Optional[float] = Field(
        None,
        description="Threshold the actual value was compared against",
    )
    message: str = Field(
        default="",
        description="Human-readable evaluation result message",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, v: str) -> str:
        """Validate rule_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_id must be non-empty")
        return v


class QualityGate(BaseModel):
    """A quality gate checkpoint for pipeline data governance.

    Quality gates aggregate multiple conditions (dimension thresholds)
    into a single go/no-go decision point. Data must meet the gate
    threshold to proceed through the pipeline.

    Attributes:
        gate_id: Unique identifier for this quality gate.
        name: Human-readable gate name.
        description: Detailed description of the gate purpose.
        conditions: List of gate condition specifications.
        threshold: Overall pass threshold (0.0 to 1.0).
        outcome: Gate evaluation outcome.
        overall_score: Overall quality score at time of evaluation.
        dimension_scores: Per-dimension scores at time of evaluation.
        evaluations: Rule evaluation results contributing to this gate.
        evaluated_at: Timestamp when the gate was evaluated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    gate_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this quality gate",
    )
    name: str = Field(
        ..., description="Human-readable gate name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the gate purpose",
    )
    conditions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of gate condition specifications",
    )
    threshold: float = Field(
        default=0.70, ge=0.0, le=1.0,
        description="Overall pass threshold (0.0 to 1.0)",
    )
    outcome: GateOutcome = Field(
        default=GateOutcome.FAIL,
        description="Gate evaluation outcome",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall quality score at time of evaluation",
    )
    dimension_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension scores at time of evaluation",
    )
    evaluations: List[RuleEvaluation] = Field(
        default_factory=list,
        description="Rule evaluation results contributing to this gate",
    )
    evaluated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the gate was evaluated",
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


class QualityTrend(BaseModel):
    """Quality score trend for a dataset over time.

    Tracks historical quality scores across multiple profiling
    runs with direction classification and percentage change.

    Attributes:
        dataset_name: Name or identifier of the tracked dataset.
        period: Time period label for the trend window.
        scores: List of historical quality scores (chronological).
        direction: Overall trend direction classification.
        change_pct: Percentage change from first to last score.
        min_score: Minimum score in the trend window.
        max_score: Maximum score in the trend window.
        avg_score: Average score in the trend window.
        data_points: Number of profiling runs in the trend window.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    dataset_name: str = Field(
        ..., description="Name or identifier of the tracked dataset",
    )
    period: str = Field(
        default="",
        description="Time period label for the trend window",
    )
    scores: List[float] = Field(
        default_factory=list,
        description="List of historical quality scores (chronological)",
    )
    direction: TrendDirection = Field(
        default=TrendDirection.UNKNOWN,
        description="Overall trend direction classification",
    )
    change_pct: float = Field(
        default=0.0,
        description="Percentage change from first to last score",
    )
    min_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum score in the trend window",
    )
    max_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Maximum score in the trend window",
    )
    avg_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average score in the trend window",
    )
    data_points: int = Field(
        default=0, ge=0,
        description="Number of profiling runs in the trend window",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


class QualityScorecardRow(BaseModel):
    """A single row in a quality scorecard.

    Represents one quality dimension with its score, weight,
    weighted contribution, issue count, and trend indicator.

    Attributes:
        dimension: Quality dimension name.
        score: Raw dimension score (0.0 to 1.0).
        weight: Weight applied to this dimension.
        weighted_score: Weighted contribution (score * weight).
        issues_count: Number of issues for this dimension.
        trend: Trend direction for this dimension.
    """

    dimension: QualityDimension = Field(
        ..., description="Quality dimension name",
    )
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw dimension score (0.0 to 1.0)",
    )
    weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weight applied to this dimension",
    )
    weighted_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted contribution (score * weight)",
    )
    issues_count: int = Field(
        default=0, ge=0,
        description="Number of issues for this dimension",
    )
    trend: TrendDirection = Field(
        default=TrendDirection.UNKNOWN,
        description="Trend direction for this dimension",
    )

    model_config = {"extra": "forbid"}


class QualityScorecard(BaseModel):
    """Complete quality scorecard for a dataset.

    Presents a tabular summary of all quality dimensions with
    overall score, quality level, and trend information for
    executive dashboards and compliance reports.

    Attributes:
        scorecard_id: Unique identifier for this scorecard.
        dataset_name: Name or identifier of the scored dataset.
        overall_score: Weighted overall quality score (0.0 to 1.0).
        quality_level: Quality level classification.
        rows: Per-dimension scorecard rows.
        total_issues: Total issues across all dimensions.
        created_at: Timestamp when the scorecard was generated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    scorecard_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this scorecard",
    )
    dataset_name: str = Field(
        ..., description="Name or identifier of the scored dataset",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted overall quality score (0.0 to 1.0)",
    )
    quality_level: QualityLevel = Field(
        default=QualityLevel.CRITICAL,
        description="Quality level classification",
    )
    rows: List[QualityScorecardRow] = Field(
        default_factory=list,
        description="Per-dimension scorecard rows",
    )
    total_issues: int = Field(
        default=0, ge=0,
        description="Total issues across all dimensions",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the scorecard was generated",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


class DataQualityProfilerStatistics(BaseModel):
    """Aggregated operational statistics for the profiler service.

    Provides high-level metrics for monitoring the overall
    health, throughput, and effectiveness of the profiler.

    Attributes:
        datasets_profiled: Total number of datasets profiled.
        assessments_completed: Total number of assessments completed.
        rules_evaluated: Total number of rule evaluations performed.
        anomalies_detected: Total number of anomalies detected.
        gates_evaluated: Total number of quality gates evaluated.
        avg_quality_score: Average quality score across all assessments.
        by_dimension: Average scores broken down by quality dimension.
        by_quality_level: Count of assessments per quality level.
        timestamp: Timestamp when statistics were computed.
    """

    datasets_profiled: int = Field(
        default=0, ge=0,
        description="Total number of datasets profiled",
    )
    assessments_completed: int = Field(
        default=0, ge=0,
        description="Total number of assessments completed",
    )
    rules_evaluated: int = Field(
        default=0, ge=0,
        description="Total number of rule evaluations performed",
    )
    anomalies_detected: int = Field(
        default=0, ge=0,
        description="Total number of anomalies detected",
    )
    gates_evaluated: int = Field(
        default=0, ge=0,
        description="Total number of quality gates evaluated",
    )
    avg_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average quality score across all assessments",
    )
    by_dimension: Dict[str, float] = Field(
        default_factory=dict,
        description="Average scores broken down by quality dimension",
    )
    by_quality_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of assessments per quality level",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when statistics were computed",
    )

    model_config = {"extra": "forbid"}


class ProfileSummary(BaseModel):
    """Lightweight summary of a completed dataset profile.

    Provides key metadata for listing and searching profiles
    without loading the full column-level detail.

    Attributes:
        profile_id: Unique profile identifier.
        dataset_name: Name or identifier of the profiled dataset.
        row_count: Total number of rows in the dataset.
        column_count: Total number of columns in the dataset.
        overall_quality: Overall quality score from latest assessment.
        status: Profile completion status.
        created_at: Timestamp when profiling was initiated.
    """

    profile_id: str = Field(
        ..., description="Unique profile identifier",
    )
    dataset_name: str = Field(
        ..., description="Name or identifier of the profiled dataset",
    )
    row_count: int = Field(
        default=0, ge=0,
        description="Total number of rows in the dataset",
    )
    column_count: int = Field(
        default=0, ge=0,
        description="Total number of columns in the dataset",
    )
    overall_quality: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall quality score from latest assessment",
    )
    status: ProfileStatus = Field(
        default=ProfileStatus.COMPLETED,
        description="Profile completion status",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when profiling was initiated",
    )

    model_config = {"extra": "forbid"}

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, v: str) -> str:
        """Validate profile_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("profile_id must be non-empty")
        return v

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


# =============================================================================
# Request Models (7)
# =============================================================================


class ProfileDatasetRequest(BaseModel):
    """Request body for profiling a dataset.

    Attributes:
        dataset_name: Name or identifier for the dataset.
        data: List of row dictionaries to profile.
        columns: Optional subset of columns to profile.
        enable_schema_inference: Override schema inference setting.
        enable_cardinality_analysis: Override cardinality analysis setting.
        sample_size: Override sample size for statistics.
    """

    dataset_name: str = Field(
        ..., description="Name or identifier for the dataset",
    )
    data: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of row dictionaries to profile",
    )
    columns: Optional[List[str]] = Field(
        None,
        description="Optional subset of columns to profile",
    )
    enable_schema_inference: Optional[bool] = Field(
        None,
        description="Override schema inference setting",
    )
    enable_cardinality_analysis: Optional[bool] = Field(
        None,
        description="Override cardinality analysis setting",
    )
    sample_size: Optional[int] = Field(
        None, ge=1,
        description="Override sample size for statistics",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


class AssessQualityRequest(BaseModel):
    """Request body for assessing data quality of a dataset.

    Attributes:
        dataset_name: Name or identifier for the dataset.
        data: List of row dictionaries to assess.
        dimensions: Optional subset of quality dimensions to evaluate.
        include_issues: Whether to include detailed issue list.
        quality_level_threshold: Minimum quality level to pass.
    """

    dataset_name: str = Field(
        ..., description="Name or identifier for the dataset",
    )
    data: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of row dictionaries to assess",
    )
    dimensions: Optional[List[QualityDimension]] = Field(
        None,
        description="Optional subset of quality dimensions to evaluate",
    )
    include_issues: bool = Field(
        default=True,
        description="Whether to include detailed issue list",
    )
    quality_level_threshold: Optional[str] = Field(
        None,
        description="Minimum quality level to pass (e.g. 'good')",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


class ValidateDatasetRequest(BaseModel):
    """Request body for validating a dataset against quality rules.

    Attributes:
        dataset_name: Name or identifier for the dataset.
        data: List of row dictionaries to validate.
        rule_ids: Optional list of specific rule IDs to evaluate.
        fail_fast: Whether to stop on first rule failure.
    """

    dataset_name: str = Field(
        ..., description="Name or identifier for the dataset",
    )
    data: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of row dictionaries to validate",
    )
    rule_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of specific rule IDs to evaluate",
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to stop on first rule failure",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


class DetectAnomaliesRequest(BaseModel):
    """Request body for detecting anomalies in a dataset.

    Attributes:
        dataset_name: Name or identifier for the dataset.
        data: List of row dictionaries to analyze.
        columns: Optional subset of columns to check for anomalies.
        method: Override anomaly detection method.
        severity_threshold: Minimum severity level to report.
    """

    dataset_name: str = Field(
        ..., description="Name or identifier for the dataset",
    )
    data: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of row dictionaries to analyze",
    )
    columns: Optional[List[str]] = Field(
        None,
        description="Optional subset of columns to check for anomalies",
    )
    method: Optional[AnomalyMethod] = Field(
        None,
        description="Override anomaly detection method",
    )
    severity_threshold: Optional[AnomalySeverity] = Field(
        None,
        description="Minimum severity level to report",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v


class CheckFreshnessRequest(BaseModel):
    """Request body for checking dataset freshness.

    Attributes:
        dataset_name: Name or identifier for the dataset.
        last_updated: ISO-8601 timestamp of the last dataset update.
        sla_hours: Override SLA deadline in hours.
    """

    dataset_name: str = Field(
        ..., description="Name or identifier for the dataset",
    )
    last_updated: str = Field(
        ...,
        description="ISO-8601 timestamp of the last dataset update",
    )
    sla_hours: Optional[int] = Field(
        None, ge=1,
        description="Override SLA deadline in hours",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v

    @field_validator("last_updated")
    @classmethod
    def validate_last_updated(cls, v: str) -> str:
        """Validate last_updated is non-empty."""
        if not v or not v.strip():
            raise ValueError("last_updated must be non-empty")
        return v


class CreateRuleRequest(BaseModel):
    """Request body for creating a new quality rule.

    Attributes:
        name: Human-readable rule name.
        description: Detailed description of the rule purpose.
        rule_type: Type of quality check this rule performs.
        column: Target column name (None for dataset-level rules).
        operator: Comparison operator for threshold evaluation.
        threshold: Numeric threshold for the rule evaluation.
        parameters: Additional rule parameters as key-value pairs.
        priority: Evaluation priority (lower = higher priority).
    """

    name: str = Field(
        ..., description="Human-readable rule name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the rule purpose",
    )
    rule_type: RuleType = Field(
        ..., description="Type of quality check this rule performs",
    )
    column: Optional[str] = Field(
        None,
        description="Target column name (None for dataset-level rules)",
    )
    operator: RuleOperator = Field(
        default=RuleOperator.GREATER_THAN,
        description="Comparison operator for threshold evaluation",
    )
    threshold: Optional[float] = Field(
        None,
        description="Numeric threshold for the rule evaluation",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional rule parameters as key-value pairs",
    )
    priority: int = Field(
        default=100, ge=0,
        description="Evaluation priority (lower = higher priority)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class GenerateReportRequest(BaseModel):
    """Request body for generating a quality profiler report.

    Attributes:
        dataset_name: Name or identifier for the dataset.
        report_type: Type of report to generate.
        format: Desired output format.
        include_profile: Whether to include profiling details.
        include_assessment: Whether to include assessment details.
        include_anomalies: Whether to include anomaly details.
        include_trends: Whether to include trend analysis.
    """

    dataset_name: str = Field(
        ..., description="Name or identifier for the dataset",
    )
    report_type: str = Field(
        default="comprehensive",
        description="Type of report (comprehensive, summary, executive)",
    )
    format: Optional[ReportFormat] = Field(
        None,
        description="Desired output format",
    )
    include_profile: bool = Field(
        default=True,
        description="Whether to include profiling details",
    )
    include_assessment: bool = Field(
        default=True,
        description="Whether to include assessment details",
    )
    include_anomalies: bool = Field(
        default=True,
        description="Whether to include anomaly details",
    )
    include_trends: bool = Field(
        default=True,
        description="Whether to include trend analysis",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        """Validate dataset_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_name must be non-empty")
        return v

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v: str) -> str:
        """Validate report_type is a known type."""
        allowed = {"comprehensive", "summary", "executive", "compliance"}
        if v not in allowed:
            raise ValueError(
                f"report_type must be one of {allowed}, got '{v}'"
            )
        return v


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Re-exported Layer 1 models (excel_normalizer.data_quality_scorer)
    # -------------------------------------------------------------------------
    "L1QualityLevel",
    "L1DataQualityReport",
    "L1DataQualityScorer",
    "QualityLevel",
    "DataQualityReport",
    "DataQualityScorer",
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "DEFAULT_DIMENSION_WEIGHTS",
    "QUALITY_LEVEL_THRESHOLDS",
    "DEFAULT_ANOMALY_THRESHOLDS",
    "FRESHNESS_BOUNDARIES_HOURS",
    "ALL_QUALITY_DIMENSIONS",
    "SUPPORTED_DATA_TYPES",
    "ANOMALY_METHOD_NAMES",
    "REPORT_FORMAT_OPTIONS",
    "GATE_OUTCOME_VALUES",
    "ISSUE_SEVERITY_ORDER",
    # -------------------------------------------------------------------------
    # New enumerations (13)
    # -------------------------------------------------------------------------
    "QualityDimension",
    "DataType",
    "ProfileStatus",
    "AssessmentStatus",
    "AnomalyMethod",
    "AnomalySeverity",
    "RuleType",
    "RuleOperator",
    "GateOutcome",
    "IssueSeverity",
    "MissingPattern",
    "ReportFormat",
    "TrendDirection",
    # -------------------------------------------------------------------------
    # SDK data models (15)
    # -------------------------------------------------------------------------
    "ColumnProfile",
    "DatasetProfile",
    "DimensionScore",
    "QualityAssessment",
    "QualityIssue",
    "AnomalyResult",
    "FreshnessResult",
    "QualityRule",
    "RuleEvaluation",
    "QualityGate",
    "QualityTrend",
    "QualityScorecardRow",
    "QualityScorecard",
    "DataQualityProfilerStatistics",
    "ProfileSummary",
    # -------------------------------------------------------------------------
    # Request models (7)
    # -------------------------------------------------------------------------
    "ProfileDatasetRequest",
    "AssessQualityRequest",
    "ValidateDatasetRequest",
    "DetectAnomaliesRequest",
    "CheckFreshnessRequest",
    "CreateRuleRequest",
    "GenerateReportRequest",
]
