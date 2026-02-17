# -*- coding: utf-8 -*-
"""
Missing Value Imputer Agent Service Data Models - AGENT-DATA-012

Pydantic v2 data models for the Missing Value Imputer SDK. Re-exports the
Layer 1 models from the Data Quality Profiler's CompletenessAnalyzer,
and defines additional SDK models for missingness analysis, strategy
selection, imputation execution, validation, rule-based imputation,
template management, pipeline orchestration, statistics, and reporting.

Re-exported Layer 1 sources:
    - greenlang.data_quality_profiler.models: QualityDimension
        (as L1QualityDimension), DataType (as L1DataType)
    - greenlang.data_quality_profiler.completeness_analyzer:
        CompletenessAnalyzer (as L1CompletenessAnalyzer),
        CompletenessReport (as L1CompletenessReport)

New enumerations (12):
    - MissingnessType, ImputationStrategy, ImputationStatus,
      ConfidenceLevel, DataColumnType, ValidationMethod,
      RuleConditionType, RulePriority, ReportFormat, PipelineStage,
      PatternType, TimeSeriesFrequency

New SDK models (20):
    - MissingnessPattern, ColumnAnalysis, MissingnessReport,
      ImputationRule, RuleCondition, LookupTable, LookupEntry,
      ImputedValue, ImputationResult, ImputationBatch,
      ValidationResult, ValidationReport, ImputationTemplate,
      PipelineConfig, PipelineResult, ImputationJobConfig,
      ImputationStatistics, TimeSeriesConfig, MLModelConfig,
      StrategySelection

Request models (8):
    - CreateJobRequest, AnalyzeMissingnessRequest,
      ImputeValuesRequest, BatchImputeRequest, ValidateRequest,
      CreateRuleRequest, CreateTemplateRequest, RunPipelineRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
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
        DataType as L1DataType,
    )
    QualityDimension = L1QualityDimension
    DataType = L1DataType
except ImportError:
    QualityDimension = None  # type: ignore[assignment, misc]
    DataType = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.completeness_analyzer import (
        CompletenessAnalyzer as L1CompletenessAnalyzer,
        CompletenessReport as L1CompletenessReport,
    )
    CompletenessAnalyzer = L1CompletenessAnalyzer
    CompletenessReport = L1CompletenessReport
except ImportError:
    CompletenessAnalyzer = None  # type: ignore[assignment, misc]
    CompletenessReport = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default confidence thresholds for imputation acceptance.
DEFAULT_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "high": 0.85,
    "medium": 0.70,
    "low": 0.50,
    "very_low": 0.30,
}

#: Default strategy recommendations by data column type.
STRATEGY_BY_COLUMN_TYPE: Dict[str, str] = {
    "numeric": "mean",
    "categorical": "mode",
    "datetime": "linear_interpolation",
    "boolean": "mode",
    "text": "mode",
}

#: Default KNN neighbor counts by dataset size ranges.
KNN_NEIGHBORS_BY_SIZE: Dict[str, int] = {
    "small": 3,       # < 1000 records
    "medium": 5,      # 1000-10000 records
    "large": 7,       # 10000-50000 records
    "very_large": 11,  # > 50000 records
}

#: Maximum missing percentage before column is excluded from imputation.
MAX_MISSING_PCT_DEFAULT: float = 0.80

#: Minimum records required for ML-based imputation.
MIN_RECORDS_ML: int = 100

#: Maximum MICE iterations before convergence warning.
MAX_MICE_ITERATIONS: int = 50

#: All supported imputation strategies.
IMPUTATION_STRATEGIES: tuple = (
    "mean", "median", "mode", "knn", "regression",
    "hot_deck", "locf", "nocb", "random_forest",
    "gradient_boosting", "mice", "matrix_factorization",
    "linear_interpolation", "spline_interpolation",
    "seasonal_decomposition", "rule_based", "lookup_table",
    "regulatory_default", "custom",
)

#: All supported validation methods.
VALIDATION_METHODS: tuple = (
    "ks_test", "chi_square", "plausibility_range",
    "distribution_preservation", "cross_validation",
)

#: Pipeline stage execution order.
PIPELINE_STAGE_ORDER: tuple = (
    "analyze", "strategize", "impute", "validate", "document",
)

#: All supported time-series frequencies.
TIME_SERIES_FREQUENCIES: tuple = (
    "hourly", "daily", "weekly", "monthly", "quarterly", "yearly",
)

#: All supported interpolation methods.
INTERPOLATION_METHODS: tuple = (
    "linear", "quadratic", "cubic", "spline", "polynomial",
)

#: Report format options.
REPORT_FORMAT_OPTIONS: tuple = ("json", "csv", "markdown", "html", "pdf")


# =============================================================================
# Enumerations (12)
# =============================================================================


class MissingnessType(str, Enum):
    """Classification of the missingness mechanism.

    MCAR: Missing Completely At Random - missingness is independent of
        both observed and unobserved data. Safe for most imputation methods.
    MAR: Missing At Random - missingness depends on observed data but
        not on the missing values themselves. Requires conditional methods.
    MNAR: Missing Not At Random - missingness depends on the missing
        values themselves. Requires model-based or domain-knowledge methods.
    UNKNOWN: Missingness mechanism has not been determined.
    """

    MCAR = "mcar"
    MAR = "mar"
    MNAR = "mnar"
    UNKNOWN = "unknown"


class ImputationStrategy(str, Enum):
    """Strategy for imputing missing values.

    Covers statistical, ML-based, time-series, and rule-based approaches.

    MEAN: Replace with column mean (numeric only).
    MEDIAN: Replace with column median (numeric only).
    MODE: Replace with most frequent value (any type).
    KNN: K-nearest neighbors imputation using similar records.
    REGRESSION: Linear/logistic regression-based imputation.
    HOT_DECK: Random draw from similar complete records.
    LOCF: Last Observation Carried Forward (time-series).
    NOCB: Next Observation Carried Backward (time-series).
    RANDOM_FOREST: Random forest model imputation.
    GRADIENT_BOOSTING: Gradient boosting model imputation.
    MICE: Multiple Imputation by Chained Equations.
    MATRIX_FACTORIZATION: Low-rank matrix factorization imputation.
    LINEAR_INTERPOLATION: Linear interpolation (time-series).
    SPLINE_INTERPOLATION: Spline interpolation (time-series).
    SEASONAL_DECOMPOSITION: Seasonal decomposition imputation (time-series).
    RULE_BASED: Domain-specific rule-based imputation.
    LOOKUP_TABLE: Lookup table-based imputation from reference data.
    REGULATORY_DEFAULT: Regulatory or standard default value.
    CUSTOM: User-defined custom imputation function.
    """

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    KNN = "knn"
    REGRESSION = "regression"
    HOT_DECK = "hot_deck"
    LOCF = "locf"
    NOCB = "nocb"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    MICE = "mice"
    MATRIX_FACTORIZATION = "matrix_factorization"
    LINEAR_INTERPOLATION = "linear_interpolation"
    SPLINE_INTERPOLATION = "spline_interpolation"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    RULE_BASED = "rule_based"
    LOOKUP_TABLE = "lookup_table"
    REGULATORY_DEFAULT = "regulatory_default"
    CUSTOM = "custom"


class ImputationStatus(str, Enum):
    """Lifecycle status of an imputation job.

    Tracks the current execution state of an imputation pipeline
    from submission through completion, failure, or cancellation.
    """

    PENDING = "pending"
    ANALYZING = "analyzing"
    IMPUTING = "imputing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Confidence level classification for imputed values.

    HIGH: Confidence score >= 0.85 - strong imputation evidence.
    MEDIUM: Confidence score >= 0.70 - acceptable imputation evidence.
    LOW: Confidence score >= 0.50 - weak imputation evidence.
    VERY_LOW: Confidence score < 0.50 - imputation is unreliable.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class DataColumnType(str, Enum):
    """Data type classification for dataset columns.

    Determines which imputation strategies are applicable and
    how validation is performed after imputation.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"


class ValidationMethod(str, Enum):
    """Statistical method for validating imputation quality.

    KS_TEST: Kolmogorov-Smirnov test for distribution similarity.
    CHI_SQUARE: Chi-square test for categorical distributions.
    PLAUSIBILITY_RANGE: Check imputed values fall within plausible range.
    DISTRIBUTION_PRESERVATION: Compare pre/post imputation distributions.
    CROSS_VALIDATION: Hold-out cross-validation on known values.
    """

    KS_TEST = "ks_test"
    CHI_SQUARE = "chi_square"
    PLAUSIBILITY_RANGE = "plausibility_range"
    DISTRIBUTION_PRESERVATION = "distribution_preservation"
    CROSS_VALIDATION = "cross_validation"


class RuleConditionType(str, Enum):
    """Condition type for rule-based imputation rules.

    Defines how a rule condition is evaluated against record data
    to determine whether the rule applies.
    """

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN_LIST = "in_list"
    REGEX = "regex"
    IS_NULL = "is_null"


class RulePriority(str, Enum):
    """Priority level for imputation rules.

    Higher-priority rules are evaluated first. When multiple rules
    match, the highest-priority rule determines the imputed value.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFAULT = "default"


class ReportFormat(str, Enum):
    """Output format for imputation reports.

    Defines the serialization format for generated reports
    including analysis summaries, imputation details, and validation
    results.
    """

    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


class PipelineStage(str, Enum):
    """Stage in the imputation pipeline.

    The pipeline executes stages in order: analyze, strategize,
    impute, validate, document. Each stage produces output
    consumed by the next stage.
    """

    ANALYZE = "analyze"
    STRATEGIZE = "strategize"
    IMPUTE = "impute"
    VALIDATE = "validate"
    DOCUMENT = "document"


class PatternType(str, Enum):
    """Classification of missing data patterns.

    UNIVARIATE: Missing values in a single column only.
    MONOTONE: Columns can be ordered such that if a value is missing
        in column j, all values in columns j+1, j+2, ... are also missing.
    ARBITRARY: No discernible pattern in the missing data layout.
    PLANNED: Missing by design (e.g. skip patterns in surveys).
    """

    UNIVARIATE = "univariate"
    MONOTONE = "monotone"
    ARBITRARY = "arbitrary"
    PLANNED = "planned"


class TimeSeriesFrequency(str, Enum):
    """Frequency classification for time-series data.

    Determines the expected temporal spacing between observations
    and influences interpolation and decomposition parameters.
    """

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# =============================================================================
# SDK Data Models (20)
# =============================================================================


class MissingnessPattern(BaseModel):
    """Pattern analysis result for missing data in a dataset.

    Describes the spatial and statistical pattern of missing values
    across columns, including the missingness mechanism classification
    and pattern type.

    Attributes:
        pattern_id: Unique identifier for this pattern analysis.
        dataset_id: Identifier of the dataset analyzed.
        pattern_type: Classification of the missing data pattern.
        missingness_type: Mechanism classification (MCAR/MAR/MNAR).
        affected_columns: List of columns with missing values.
        correlation_matrix: Pairwise missingness correlations between columns.
        total_missing: Total count of missing values across all columns.
        total_cells: Total count of cells in the dataset.
        overall_missing_pct: Overall fraction of missing values.
        created_at: Timestamp when the pattern was analyzed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    pattern_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this pattern analysis",
    )
    dataset_id: str = Field(
        default="", description="Identifier of the dataset analyzed",
    )
    pattern_type: PatternType = Field(
        default=PatternType.ARBITRARY,
        description="Classification of the missing data pattern",
    )
    missingness_type: MissingnessType = Field(
        default=MissingnessType.UNKNOWN,
        description="Mechanism classification (MCAR/MAR/MNAR)",
    )
    affected_columns: List[str] = Field(
        default_factory=list,
        description="List of columns with missing values",
    )
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Pairwise missingness correlations between columns",
    )
    total_missing: int = Field(
        default=0, ge=0,
        description="Total count of missing values across all columns",
    )
    total_cells: int = Field(
        default=0, ge=0,
        description="Total count of cells in the dataset",
    )
    overall_missing_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall fraction of missing values",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the pattern was analyzed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class ColumnAnalysis(BaseModel):
    """Missing value analysis for a single dataset column.

    Provides detailed statistics about missing values in one column
    including count, percentage, distribution characteristics, and
    recommended imputation strategy.

    Attributes:
        column_name: Name of the analyzed column.
        column_type: Data type classification of the column.
        total_values: Total number of values (including missing).
        missing_count: Count of missing values.
        missing_pct: Fraction of values that are missing (0.0-1.0).
        missingness_type: Inferred missingness mechanism for this column.
        unique_values: Number of unique non-missing values.
        mean_value: Mean of non-missing values (numeric columns only).
        median_value: Median of non-missing values (numeric columns only).
        mode_value: Most frequent non-missing value.
        std_dev: Standard deviation of non-missing values (numeric only).
        min_value: Minimum non-missing value (numeric/datetime only).
        max_value: Maximum non-missing value (numeric/datetime only).
        recommended_strategy: Auto-recommended imputation strategy.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    column_name: str = Field(
        ..., description="Name of the analyzed column",
    )
    column_type: DataColumnType = Field(
        default=DataColumnType.TEXT,
        description="Data type classification of the column",
    )
    total_values: int = Field(
        default=0, ge=0,
        description="Total number of values (including missing)",
    )
    missing_count: int = Field(
        default=0, ge=0,
        description="Count of missing values",
    )
    missing_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of values that are missing (0.0-1.0)",
    )
    missingness_type: MissingnessType = Field(
        default=MissingnessType.UNKNOWN,
        description="Inferred missingness mechanism for this column",
    )
    unique_values: int = Field(
        default=0, ge=0,
        description="Number of unique non-missing values",
    )
    mean_value: Optional[float] = Field(
        None, description="Mean of non-missing values (numeric columns only)",
    )
    median_value: Optional[float] = Field(
        None, description="Median of non-missing values (numeric columns only)",
    )
    mode_value: Optional[Any] = Field(
        None, description="Most frequent non-missing value",
    )
    std_dev: Optional[float] = Field(
        None, description="Standard deviation of non-missing values (numeric only)",
    )
    min_value: Optional[Any] = Field(
        None, description="Minimum non-missing value (numeric/datetime only)",
    )
    max_value: Optional[Any] = Field(
        None, description="Maximum non-missing value (numeric/datetime only)",
    )
    recommended_strategy: ImputationStrategy = Field(
        default=ImputationStrategy.MEAN,
        description="Auto-recommended imputation strategy",
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


class MissingnessReport(BaseModel):
    """Complete missingness analysis report for a dataset.

    Aggregates pattern analysis, per-column analysis, and overall
    statistics into a comprehensive report for strategy selection.

    Attributes:
        report_id: Unique identifier for this report.
        dataset_id: Identifier of the analyzed dataset.
        pattern: Missingness pattern analysis result.
        columns: Per-column analysis results.
        total_records: Total number of records in the dataset.
        total_columns: Total number of columns in the dataset.
        columns_with_missing: Number of columns that have missing values.
        complete_records: Number of records with no missing values.
        complete_record_pct: Fraction of records that are complete.
        generated_at: Timestamp when the report was generated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    dataset_id: str = Field(
        default="", description="Identifier of the analyzed dataset",
    )
    pattern: Optional[MissingnessPattern] = Field(
        None, description="Missingness pattern analysis result",
    )
    columns: List[ColumnAnalysis] = Field(
        default_factory=list,
        description="Per-column analysis results",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total number of records in the dataset",
    )
    total_columns: int = Field(
        default=0, ge=0,
        description="Total number of columns in the dataset",
    )
    columns_with_missing: int = Field(
        default=0, ge=0,
        description="Number of columns that have missing values",
    )
    complete_records: int = Field(
        default=0, ge=0,
        description="Number of records with no missing values",
    )
    complete_record_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of records that are complete",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the report was generated",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class RuleCondition(BaseModel):
    """A single condition within an imputation rule.

    Defines a predicate that is evaluated against record fields
    to determine whether the parent rule applies.

    Attributes:
        field_name: Name of the field to evaluate.
        condition_type: Type of condition (equals, contains, etc.).
        value: Expected value or pattern for the condition.
        case_sensitive: Whether string comparison is case-sensitive.
    """

    field_name: str = Field(
        ..., description="Name of the field to evaluate",
    )
    condition_type: RuleConditionType = Field(
        ..., description="Type of condition (equals, contains, etc.)",
    )
    value: Optional[Any] = Field(
        None, description="Expected value or pattern for the condition",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether string comparison is case-sensitive",
    )

    model_config = {"extra": "forbid"}

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v


class ImputationRule(BaseModel):
    """A rule defining conditional imputation logic.

    Encapsulates conditions that must be met and the imputation
    value or strategy to apply when conditions are satisfied.
    Rules are evaluated in priority order.

    Attributes:
        rule_id: Unique identifier for this rule.
        name: Human-readable rule name.
        description: Detailed description of the rule purpose.
        target_column: Column whose missing values this rule imputes.
        conditions: List of conditions that must all be met.
        impute_value: Static value to impute when conditions are met.
        impute_strategy: Strategy to use when conditions are met.
        priority: Rule evaluation priority.
        active: Whether this rule is currently active.
        created_at: Timestamp when the rule was created.
        updated_at: Timestamp when the rule was last updated.
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
    target_column: str = Field(
        ..., description="Column whose missing values this rule imputes",
    )
    conditions: List[RuleCondition] = Field(
        default_factory=list,
        description="List of conditions that must all be met",
    )
    impute_value: Optional[Any] = Field(
        None, description="Static value to impute when conditions are met",
    )
    impute_strategy: Optional[ImputationStrategy] = Field(
        None, description="Strategy to use when conditions are met",
    )
    priority: RulePriority = Field(
        default=RulePriority.MEDIUM,
        description="Rule evaluation priority",
    )
    active: bool = Field(
        default=True,
        description="Whether this rule is currently active",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the rule was created",
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the rule was last updated",
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

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v: str) -> str:
        """Validate target_column is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_column must be non-empty")
        return v


class LookupEntry(BaseModel):
    """A single entry in a lookup table for imputation.

    Attributes:
        key: Lookup key value (matching value from source column).
        value: Value to impute when the key matches.
        source: Source of this lookup entry (e.g. regulatory body).
    """

    key: str = Field(
        ..., description="Lookup key value",
    )
    value: Any = Field(
        ..., description="Value to impute when the key matches",
    )
    source: str = Field(
        default="",
        description="Source of this lookup entry",
    )

    model_config = {"extra": "forbid"}

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Validate key is non-empty."""
        if not v or not v.strip():
            raise ValueError("key must be non-empty")
        return v


class LookupTable(BaseModel):
    """Reference lookup table for lookup-based imputation.

    Provides a mapping from key column values to imputation values,
    used by the LOOKUP_TABLE imputation strategy.

    Attributes:
        table_id: Unique identifier for this lookup table.
        name: Human-readable table name.
        description: Detailed description of the table purpose.
        key_column: Name of the column used for key matching.
        target_column: Name of the column to impute.
        entries: List of lookup entries.
        default_value: Value to use when no key matches.
        created_at: Timestamp when the table was created.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    table_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this lookup table",
    )
    name: str = Field(
        ..., description="Human-readable table name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the table purpose",
    )
    key_column: str = Field(
        ..., description="Name of the column used for key matching",
    )
    target_column: str = Field(
        ..., description="Name of the column to impute",
    )
    entries: List[LookupEntry] = Field(
        default_factory=list,
        description="List of lookup entries",
    )
    default_value: Optional[Any] = Field(
        None, description="Value to use when no key matches",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the table was created",
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

    @field_validator("key_column")
    @classmethod
    def validate_key_column(cls, v: str) -> str:
        """Validate key_column is non-empty."""
        if not v or not v.strip():
            raise ValueError("key_column must be non-empty")
        return v

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v: str) -> str:
        """Validate target_column is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_column must be non-empty")
        return v


class ImputedValue(BaseModel):
    """A single imputed value with provenance and confidence.

    Represents one missing value that has been filled by an imputation
    strategy, including the original record context, confidence score,
    and complete audit trail.

    Attributes:
        record_index: Row index of the imputed record in the dataset.
        column_name: Column name where the value was imputed.
        imputed_value: The imputed value.
        original_value: The original value (typically None/NaN).
        strategy: Imputation strategy that produced this value.
        confidence: Confidence score for the imputed value (0.0-1.0).
        confidence_level: Classified confidence level.
        contributing_records: Number of records used to compute the value.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_index: int = Field(
        ..., ge=0,
        description="Row index of the imputed record in the dataset",
    )
    column_name: str = Field(
        ..., description="Column name where the value was imputed",
    )
    imputed_value: Any = Field(
        ..., description="The imputed value",
    )
    original_value: Optional[Any] = Field(
        None, description="The original value (typically None/NaN)",
    )
    strategy: ImputationStrategy = Field(
        ..., description="Imputation strategy that produced this value",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence score for the imputed value (0.0-1.0)",
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Classified confidence level",
    )
    contributing_records: int = Field(
        default=0, ge=0,
        description="Number of records used to compute the value",
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


class ImputationResult(BaseModel):
    """Result of imputing missing values for a single column.

    Contains the list of imputed values, strategy used, overall
    confidence, and completeness improvement statistics.

    Attributes:
        result_id: Unique identifier for this result.
        column_name: Name of the column that was imputed.
        strategy: Imputation strategy applied.
        values_imputed: Number of values imputed.
        imputed_values: List of individual imputed value records.
        avg_confidence: Average confidence across all imputed values.
        min_confidence: Minimum confidence among imputed values.
        completeness_before: Column completeness before imputation.
        completeness_after: Column completeness after imputation.
        processing_time_ms: Time in milliseconds to impute this column.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this result",
    )
    column_name: str = Field(
        ..., description="Name of the column that was imputed",
    )
    strategy: ImputationStrategy = Field(
        ..., description="Imputation strategy applied",
    )
    values_imputed: int = Field(
        default=0, ge=0,
        description="Number of values imputed",
    )
    imputed_values: List[ImputedValue] = Field(
        default_factory=list,
        description="List of individual imputed value records",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average confidence across all imputed values",
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum confidence among imputed values",
    )
    completeness_before: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Column completeness before imputation",
    )
    completeness_after: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Column completeness after imputation",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Time in milliseconds to impute this column",
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


class ImputationBatch(BaseModel):
    """Batch of imputation results across multiple columns.

    Groups per-column imputation results into a single batch for
    job tracking and reporting.

    Attributes:
        batch_id: Unique identifier for this batch.
        job_id: Identifier of the parent imputation job.
        results: List of per-column imputation results.
        total_values_imputed: Total values imputed across all columns.
        avg_confidence: Average confidence across all columns.
        processing_time_ms: Total processing time for the batch.
        created_at: Timestamp when the batch was processed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this batch",
    )
    job_id: str = Field(
        default="", description="Identifier of the parent imputation job",
    )
    results: List[ImputationResult] = Field(
        default_factory=list,
        description="List of per-column imputation results",
    )
    total_values_imputed: int = Field(
        default=0, ge=0,
        description="Total values imputed across all columns",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average confidence across all columns",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Total processing time for the batch",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the batch was processed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class ValidationResult(BaseModel):
    """Result of validating an imputed column using a statistical test.

    Attributes:
        column_name: Name of the validated column.
        method: Validation method used.
        passed: Whether the validation passed.
        test_statistic: Test statistic value (e.g. KS statistic, chi2).
        p_value: P-value from the statistical test.
        threshold: Significance threshold used for the test.
        details: Additional validation details.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    column_name: str = Field(
        ..., description="Name of the validated column",
    )
    method: ValidationMethod = Field(
        ..., description="Validation method used",
    )
    passed: bool = Field(
        default=False,
        description="Whether the validation passed",
    )
    test_statistic: Optional[float] = Field(
        None, description="Test statistic value",
    )
    p_value: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="P-value from the statistical test",
    )
    threshold: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Significance threshold used for the test",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional validation details",
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


class ValidationReport(BaseModel):
    """Complete validation report for an imputation batch.

    Aggregates per-column validation results and provides an
    overall pass/fail assessment of imputation quality.

    Attributes:
        report_id: Unique identifier for this validation report.
        batch_id: Identifier of the imputation batch validated.
        results: List of per-column validation results.
        overall_passed: Whether all validations passed.
        columns_passed: Number of columns that passed validation.
        columns_failed: Number of columns that failed validation.
        generated_at: Timestamp when the report was generated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this validation report",
    )
    batch_id: str = Field(
        default="", description="Identifier of the imputation batch validated",
    )
    results: List[ValidationResult] = Field(
        default_factory=list,
        description="List of per-column validation results",
    )
    overall_passed: bool = Field(
        default=False,
        description="Whether all validations passed",
    )
    columns_passed: int = Field(
        default=0, ge=0,
        description="Number of columns that passed validation",
    )
    columns_failed: int = Field(
        default=0, ge=0,
        description="Number of columns that failed validation",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the report was generated",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class ImputationTemplate(BaseModel):
    """Reusable imputation template for consistent processing.

    Defines a named set of column-level strategy assignments,
    rules, and configuration that can be applied to multiple
    datasets for consistent imputation behavior.

    Attributes:
        template_id: Unique identifier for this template.
        name: Human-readable template name.
        description: Detailed description of the template purpose.
        column_strategies: Mapping of column names to strategies.
        rules: List of imputation rules included in this template.
        lookup_tables: List of lookup tables included in this template.
        default_strategy: Fallback strategy for unmapped columns.
        confidence_threshold: Minimum confidence for acceptance.
        active: Whether this template is currently active.
        created_at: Timestamp when the template was created.
        updated_at: Timestamp when the template was last updated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this template",
    )
    name: str = Field(
        ..., description="Human-readable template name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the template purpose",
    )
    column_strategies: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column names to imputation strategies",
    )
    rules: List[ImputationRule] = Field(
        default_factory=list,
        description="List of imputation rules included in this template",
    )
    lookup_tables: List[LookupTable] = Field(
        default_factory=list,
        description="List of lookup tables included in this template",
    )
    default_strategy: ImputationStrategy = Field(
        default=ImputationStrategy.MEAN,
        description="Fallback strategy for unmapped columns",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum confidence for acceptance",
    )
    active: bool = Field(
        default=True,
        description="Whether this template is currently active",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the template was created",
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the template was last updated",
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


class TimeSeriesConfig(BaseModel):
    """Configuration for time-series imputation methods.

    Defines the temporal parameters needed for time-series aware
    imputation strategies such as interpolation and seasonal
    decomposition.

    Attributes:
        time_column: Name of the datetime column for ordering.
        frequency: Expected time-series frequency.
        interpolation_method: Interpolation method to use.
        seasonal_period: Number of periods in one seasonal cycle.
        trend_window: Window size for trend estimation.
        fill_leading: Whether to fill missing values at the start.
        fill_trailing: Whether to fill missing values at the end.
    """

    time_column: str = Field(
        ..., description="Name of the datetime column for ordering",
    )
    frequency: TimeSeriesFrequency = Field(
        default=TimeSeriesFrequency.MONTHLY,
        description="Expected time-series frequency",
    )
    interpolation_method: str = Field(
        default="linear",
        description="Interpolation method to use",
    )
    seasonal_period: int = Field(
        default=12, ge=2,
        description="Number of periods in one seasonal cycle",
    )
    trend_window: int = Field(
        default=6, ge=2,
        description="Window size for trend estimation",
    )
    fill_leading: bool = Field(
        default=False,
        description="Whether to fill missing values at the start",
    )
    fill_trailing: bool = Field(
        default=False,
        description="Whether to fill missing values at the end",
    )

    model_config = {"extra": "forbid"}

    @field_validator("time_column")
    @classmethod
    def validate_time_column(cls, v: str) -> str:
        """Validate time_column is non-empty."""
        if not v or not v.strip():
            raise ValueError("time_column must be non-empty")
        return v


class MLModelConfig(BaseModel):
    """Configuration for ML-based imputation models.

    Defines hyperparameters for random forest and gradient boosting
    imputation models.

    Attributes:
        model_type: Type of ML model (random_forest, gradient_boosting).
        n_estimators: Number of trees/estimators.
        max_depth: Maximum tree depth (None for unlimited).
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf node.
        random_state: Random seed for reproducibility.
        feature_columns: List of columns to use as features.
        cross_validation_folds: Number of CV folds for evaluation.
    """

    model_type: ImputationStrategy = Field(
        default=ImputationStrategy.RANDOM_FOREST,
        description="Type of ML model (random_forest, gradient_boosting)",
    )
    n_estimators: int = Field(
        default=100, ge=1, le=1000,
        description="Number of trees/estimators",
    )
    max_depth: Optional[int] = Field(
        None, ge=1,
        description="Maximum tree depth (None for unlimited)",
    )
    min_samples_split: int = Field(
        default=2, ge=2,
        description="Minimum samples to split a node",
    )
    min_samples_leaf: int = Field(
        default=1, ge=1,
        description="Minimum samples in a leaf node",
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    feature_columns: List[str] = Field(
        default_factory=list,
        description="List of columns to use as features",
    )
    cross_validation_folds: int = Field(
        default=5, ge=2, le=20,
        description="Number of CV folds for evaluation",
    )

    model_config = {"extra": "forbid"}


class StrategySelection(BaseModel):
    """Recommended strategy selection for a column.

    Output of the auto-strategy selection engine that recommends
    the best imputation strategy based on column characteristics,
    missingness pattern, and available data.

    Attributes:
        column_name: Name of the column.
        recommended_strategy: Best recommended strategy.
        alternative_strategies: Ranked list of alternative strategies.
        rationale: Explanation of why this strategy was chosen.
        estimated_confidence: Expected confidence for the recommendation.
        column_type: Detected column data type.
        missing_pct: Fraction of values missing in this column.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    column_name: str = Field(
        ..., description="Name of the column",
    )
    recommended_strategy: ImputationStrategy = Field(
        ..., description="Best recommended strategy",
    )
    alternative_strategies: List[ImputationStrategy] = Field(
        default_factory=list,
        description="Ranked list of alternative strategies",
    )
    rationale: str = Field(
        default="",
        description="Explanation of why this strategy was chosen",
    )
    estimated_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Expected confidence for the recommendation",
    )
    column_type: DataColumnType = Field(
        default=DataColumnType.TEXT,
        description="Detected column data type",
    )
    missing_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of values missing in this column",
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


class PipelineConfig(BaseModel):
    """Configuration for the full imputation pipeline.

    Defines pipeline-level settings including strategy overrides,
    template reference, validation methods, and output preferences.

    Attributes:
        template_id: Optional template to apply for strategy assignment.
        column_strategies: Manual column-to-strategy overrides.
        validation_methods: List of validation methods to apply.
        confidence_threshold: Minimum confidence for value acceptance.
        max_missing_pct: Maximum missing percentage before exclusion.
        enable_ml: Whether ML strategies are allowed.
        enable_timeseries: Whether time-series strategies are allowed.
        timeseries_config: Optional time-series configuration.
        ml_config: Optional ML model configuration.
        report_format: Output report format.
    """

    template_id: Optional[str] = Field(
        None, description="Optional template to apply for strategy assignment",
    )
    column_strategies: Dict[str, str] = Field(
        default_factory=dict,
        description="Manual column-to-strategy overrides",
    )
    validation_methods: List[ValidationMethod] = Field(
        default_factory=lambda: [ValidationMethod.PLAUSIBILITY_RANGE],
        description="List of validation methods to apply",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum confidence for value acceptance",
    )
    max_missing_pct: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Maximum missing percentage before exclusion",
    )
    enable_ml: bool = Field(
        default=True,
        description="Whether ML strategies are allowed",
    )
    enable_timeseries: bool = Field(
        default=True,
        description="Whether time-series strategies are allowed",
    )
    timeseries_config: Optional[TimeSeriesConfig] = Field(
        None, description="Optional time-series configuration",
    )
    ml_config: Optional[MLModelConfig] = Field(
        None, description="Optional ML model configuration",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output report format",
    )

    model_config = {"extra": "forbid"}


class PipelineResult(BaseModel):
    """Complete result of an imputation pipeline run.

    Aggregates analysis, imputation, and validation results into
    a single pipeline output with full provenance.

    Attributes:
        pipeline_id: Unique identifier for this pipeline run.
        job_id: Identifier of the parent imputation job.
        stage: Final pipeline stage reached.
        status: Final pipeline status.
        analysis_report: Missingness analysis report.
        strategy_selections: Per-column strategy selections.
        imputation_batch: Imputation results batch.
        validation_report: Validation report.
        total_processing_time_ms: Total pipeline processing time.
        created_at: Timestamp when the pipeline completed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this pipeline run",
    )
    job_id: str = Field(
        default="", description="Identifier of the parent imputation job",
    )
    stage: PipelineStage = Field(
        default=PipelineStage.DOCUMENT,
        description="Final pipeline stage reached",
    )
    status: ImputationStatus = Field(
        default=ImputationStatus.COMPLETED,
        description="Final pipeline status",
    )
    analysis_report: Optional[MissingnessReport] = Field(
        None, description="Missingness analysis report",
    )
    strategy_selections: List[StrategySelection] = Field(
        default_factory=list,
        description="Per-column strategy selections",
    )
    imputation_batch: Optional[ImputationBatch] = Field(
        None, description="Imputation results batch",
    )
    validation_report: Optional[ValidationReport] = Field(
        None, description="Validation report",
    )
    total_processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Total pipeline processing time in milliseconds",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the pipeline completed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class ImputationJobConfig(BaseModel):
    """Configuration and tracking for an imputation job.

    Represents a single end-to-end imputation run with progress
    counters, timing, error tracking, and provenance for audit trail.

    Attributes:
        job_id: Unique identifier for this imputation job.
        dataset_id: Identifier of the dataset being imputed.
        status: Current job execution status.
        stage: Current pipeline stage being executed.
        total_records: Total number of input records.
        total_columns: Total number of columns in the dataset.
        columns_imputed: Number of columns imputed so far.
        values_imputed: Total number of values imputed.
        avg_confidence: Average confidence across all imputations.
        pipeline_config: Pipeline configuration for this job.
        error_message: Error message if the job failed.
        started_at: Timestamp when the job started.
        completed_at: Timestamp when the job completed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this imputation job",
    )
    dataset_id: str = Field(
        default="",
        description="Identifier of the dataset being imputed",
    )
    status: ImputationStatus = Field(
        default=ImputationStatus.PENDING,
        description="Current job execution status",
    )
    stage: PipelineStage = Field(
        default=PipelineStage.ANALYZE,
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
    columns_imputed: int = Field(
        default=0, ge=0,
        description="Number of columns imputed so far",
    )
    values_imputed: int = Field(
        default=0, ge=0,
        description="Total number of values imputed",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average confidence across all imputations",
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
            ImputationStatus.PENDING,
            ImputationStatus.ANALYZING,
            ImputationStatus.IMPUTING,
            ImputationStatus.VALIDATING,
        )

    @property
    def progress_pct(self) -> float:
        """Return pipeline progress as a percentage (0.0 to 100.0).

        Maps the current stage to a progress value based on
        pipeline stage order.
        """
        stage_progress = {
            PipelineStage.ANALYZE: 15.0,
            PipelineStage.STRATEGIZE: 30.0,
            PipelineStage.IMPUTE: 60.0,
            PipelineStage.VALIDATE: 85.0,
            PipelineStage.DOCUMENT: 100.0,
        }
        if self.status == ImputationStatus.COMPLETED:
            return 100.0
        if self.status == ImputationStatus.FAILED:
            return 0.0
        return stage_progress.get(self.stage, 0.0)


class ImputationStatistics(BaseModel):
    """Aggregated operational statistics for the imputation service.

    Provides high-level metrics for monitoring the overall
    health, throughput, and effectiveness of the imputation
    pipeline.

    Attributes:
        total_jobs: Total number of imputation jobs executed.
        total_records: Total number of records processed across all jobs.
        total_values_imputed: Total number of values imputed.
        total_validations: Total number of validation runs.
        avg_confidence: Average confidence score across all imputations.
        avg_completeness_improvement: Average completeness improvement.
        by_status: Count of jobs per status.
        by_strategy: Count of values imputed per strategy.
        by_missingness: Count of columns per missingness type.
        timestamp: Timestamp when statistics were computed.
    """

    total_jobs: int = Field(
        default=0, ge=0,
        description="Total number of imputation jobs executed",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total number of records processed across all jobs",
    )
    total_values_imputed: int = Field(
        default=0, ge=0,
        description="Total number of values imputed",
    )
    total_validations: int = Field(
        default=0, ge=0,
        description="Total number of validation runs",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average confidence score across all imputations",
    )
    avg_completeness_improvement: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average completeness improvement",
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of jobs per status",
    )
    by_strategy: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of values imputed per strategy",
    )
    by_missingness: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of columns per missingness type",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when statistics were computed",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models (8)
# =============================================================================


class CreateJobRequest(BaseModel):
    """Request body for creating a new imputation job.

    Attributes:
        dataset_id: Identifier of the dataset to impute.
        records: List of record dictionaries to impute.
        pipeline_config: Optional pipeline configuration.
        template_id: Optional template to apply.
    """

    dataset_id: str = Field(
        default="", description="Identifier of the dataset to impute",
    )
    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to impute",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None, description="Optional pipeline configuration",
    )
    template_id: Optional[str] = Field(
        None, description="Optional template to apply",
    )

    model_config = {"extra": "forbid"}


class AnalyzeMissingnessRequest(BaseModel):
    """Request body for analyzing missingness patterns.

    Attributes:
        records: List of record dictionaries to analyze.
        columns: Optional list of columns to analyze (all if empty).
        dataset_id: Optional dataset identifier.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to analyze",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Optional list of columns to analyze (all if empty)",
    )
    dataset_id: str = Field(
        default="", description="Optional dataset identifier",
    )

    model_config = {"extra": "forbid"}


class ImputeValuesRequest(BaseModel):
    """Request body for imputing missing values in a dataset.

    Attributes:
        records: List of record dictionaries to impute.
        column_strategies: Mapping of column names to strategies.
        confidence_threshold: Minimum confidence for acceptance.
        timeseries_config: Optional time-series configuration.
        ml_config: Optional ML model configuration.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to impute",
    )
    column_strategies: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column names to imputation strategies",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum confidence for acceptance",
    )
    timeseries_config: Optional[TimeSeriesConfig] = Field(
        None, description="Optional time-series configuration",
    )
    ml_config: Optional[MLModelConfig] = Field(
        None, description="Optional ML model configuration",
    )

    model_config = {"extra": "forbid"}


class BatchImputeRequest(BaseModel):
    """Request body for batch imputation across multiple datasets.

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


class ValidateRequest(BaseModel):
    """Request body for validating imputation results.

    Attributes:
        original_records: Original records before imputation.
        imputed_records: Records after imputation.
        columns: List of columns to validate.
        methods: List of validation methods to apply.
    """

    original_records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="Original records before imputation",
    )
    imputed_records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="Records after imputation",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="List of columns to validate",
    )
    methods: List[ValidationMethod] = Field(
        default_factory=lambda: [ValidationMethod.PLAUSIBILITY_RANGE],
        description="List of validation methods to apply",
    )

    model_config = {"extra": "forbid"}


class CreateRuleRequest(BaseModel):
    """Request body for creating a new imputation rule.

    Attributes:
        name: Human-readable rule name.
        description: Detailed description of the rule purpose.
        target_column: Column whose missing values this rule imputes.
        conditions: List of rule conditions.
        impute_value: Static value to impute when conditions are met.
        impute_strategy: Strategy to use when conditions are met.
        priority: Rule evaluation priority.
    """

    name: str = Field(
        ..., description="Human-readable rule name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the rule purpose",
    )
    target_column: str = Field(
        ..., description="Column whose missing values this rule imputes",
    )
    conditions: List[RuleCondition] = Field(
        default_factory=list,
        description="List of rule conditions",
    )
    impute_value: Optional[Any] = Field(
        None, description="Static value to impute when conditions are met",
    )
    impute_strategy: Optional[ImputationStrategy] = Field(
        None, description="Strategy to use when conditions are met",
    )
    priority: RulePriority = Field(
        default=RulePriority.MEDIUM,
        description="Rule evaluation priority",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v: str) -> str:
        """Validate target_column is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_column must be non-empty")
        return v


class CreateTemplateRequest(BaseModel):
    """Request body for creating a new imputation template.

    Attributes:
        name: Human-readable template name.
        description: Detailed description of the template purpose.
        column_strategies: Mapping of column names to strategies.
        rule_ids: List of rule IDs to include in the template.
        lookup_table_ids: List of lookup table IDs to include.
        default_strategy: Fallback strategy for unmapped columns.
        confidence_threshold: Minimum confidence for acceptance.
    """

    name: str = Field(
        ..., description="Human-readable template name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the template purpose",
    )
    column_strategies: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column names to imputation strategies",
    )
    rule_ids: List[str] = Field(
        default_factory=list,
        description="List of rule IDs to include in the template",
    )
    lookup_table_ids: List[str] = Field(
        default_factory=list,
        description="List of lookup table IDs to include",
    )
    default_strategy: ImputationStrategy = Field(
        default=ImputationStrategy.MEAN,
        description="Fallback strategy for unmapped columns",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum confidence for acceptance",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class RunPipelineRequest(BaseModel):
    """Request body for running the full imputation pipeline.

    Encapsulates input records, optional pipeline configuration,
    and template reference for end-to-end imputation.

    Attributes:
        records: List of record dictionaries to impute.
        dataset_id: Optional dataset identifier.
        pipeline_config: Optional pipeline configuration.
        template_id: Optional template to apply.
        options: Optional pipeline configuration overrides.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to impute",
    )
    dataset_id: str = Field(
        default="", description="Optional dataset identifier",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None, description="Optional pipeline configuration",
    )
    template_id: Optional[str] = Field(
        None, description="Optional template to apply",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional pipeline configuration overrides",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports
    # -------------------------------------------------------------------------
    "CompletenessAnalyzer",
    "CompletenessReport",
    "QualityDimension",
    "DataType",
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "DEFAULT_CONFIDENCE_THRESHOLDS",
    "STRATEGY_BY_COLUMN_TYPE",
    "KNN_NEIGHBORS_BY_SIZE",
    "MAX_MISSING_PCT_DEFAULT",
    "MIN_RECORDS_ML",
    "MAX_MICE_ITERATIONS",
    "IMPUTATION_STRATEGIES",
    "VALIDATION_METHODS",
    "PIPELINE_STAGE_ORDER",
    "TIME_SERIES_FREQUENCIES",
    "INTERPOLATION_METHODS",
    "REPORT_FORMAT_OPTIONS",
    # -------------------------------------------------------------------------
    # Enumerations (12)
    # -------------------------------------------------------------------------
    "MissingnessType",
    "ImputationStrategy",
    "ImputationStatus",
    "ConfidenceLevel",
    "DataColumnType",
    "ValidationMethod",
    "RuleConditionType",
    "RulePriority",
    "ReportFormat",
    "PipelineStage",
    "PatternType",
    "TimeSeriesFrequency",
    # -------------------------------------------------------------------------
    # SDK data models (20)
    # -------------------------------------------------------------------------
    "MissingnessPattern",
    "ColumnAnalysis",
    "MissingnessReport",
    "ImputationRule",
    "RuleCondition",
    "LookupTable",
    "LookupEntry",
    "ImputedValue",
    "ImputationResult",
    "ImputationBatch",
    "ValidationResult",
    "ValidationReport",
    "ImputationTemplate",
    "PipelineConfig",
    "PipelineResult",
    "ImputationJobConfig",
    "ImputationStatistics",
    "TimeSeriesConfig",
    "MLModelConfig",
    "StrategySelection",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "CreateJobRequest",
    "AnalyzeMissingnessRequest",
    "ImputeValuesRequest",
    "BatchImputeRequest",
    "ValidateRequest",
    "CreateRuleRequest",
    "CreateTemplateRequest",
    "RunPipelineRequest",
]
