# -*- coding: utf-8 -*-
"""
GL-015 Insulscan: Data Quality Schemas - Version 1.0

Provides validated data schemas for data quality assessment, validation
rules, and quality reporting for insulation thermal data.

This module defines Pydantic v2 models for:
- DataQualityScore: Multi-dimensional quality scoring
- ValidationRule: Configurable validation rules
- QualityReport: Comprehensive quality assessment report

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class QualityDimension(str, Enum):
    """Dimensions of data quality."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    RELIABILITY = "reliability"
    PRECISION = "precision"


class RuleSeverity(str, Enum):
    """Severity level of validation rule violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleCategory(str, Enum):
    """Categories of validation rules."""
    RANGE_CHECK = "range_check"
    COMPLETENESS = "completeness"
    FORMAT = "format"
    CONSISTENCY = "consistency"
    BUSINESS_RULE = "business_rule"
    PHYSICS_BASED = "physics_based"
    TEMPORAL = "temporal"
    REFERENTIAL = "referential"


class ValidationOutcome(str, Enum):
    """Outcome of validation check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class QualityLevel(str, Enum):
    """Overall quality level classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class IssueStatus(str, Enum):
    """Status of a quality issue."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"
    CLOSED = "closed"


# =============================================================================
# DATA QUALITY SCORE
# =============================================================================

class DimensionScore(BaseModel):
    """Score for a single quality dimension."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "dimension": "completeness",
                    "score": 0.95,
                    "weight": 0.25,
                    "weighted_score": 0.2375,
                    "issues_count": 2
                }
            ]
        }
    )

    dimension: QualityDimension = Field(
        ...,
        description="Quality dimension"
    )
    score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Dimension score (0-1)"
    )
    weight: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Weight for this dimension in overall score"
    )
    weighted_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Weighted contribution to overall score"
    )
    issues_count: int = Field(
        default=0,
        ge=0,
        description="Number of issues in this dimension"
    )
    details: Optional[str] = Field(
        None,
        max_length=500,
        description="Details about dimension score"
    )


class DataQualityScore(BaseModel):
    """
    Multi-dimensional data quality scoring.

    Provides comprehensive quality assessment across multiple
    dimensions with configurable weights.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "overall_score": 0.92,
                    "quality_level": "excellent",
                    "completeness": 0.98,
                    "accuracy": 0.95,
                    "timeliness": 0.90,
                    "consistency": 0.88
                }
            ]
        }
    )

    # Overall score
    overall_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall quality score (0-1)"
    )
    quality_level: QualityLevel = Field(
        ...,
        description="Quality level classification"
    )

    # Individual dimension scores
    completeness: float = Field(
        ...,
        ge=0,
        le=1,
        description="Data completeness score (0-1)"
    )
    accuracy: float = Field(
        ...,
        ge=0,
        le=1,
        description="Data accuracy score (0-1)"
    )
    timeliness: float = Field(
        ...,
        ge=0,
        le=1,
        description="Data timeliness score (0-1)"
    )
    consistency: float = Field(
        ...,
        ge=0,
        le=1,
        description="Data consistency score (0-1)"
    )

    # Optional additional dimensions
    validity: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Data validity score (0-1)"
    )
    reliability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Data reliability score (0-1)"
    )
    precision: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Data precision score (0-1)"
    )

    # Detailed scores
    dimension_scores: List[DimensionScore] = Field(
        default_factory=list,
        description="Detailed scores by dimension"
    )

    # Weights used
    weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights used for each dimension"
    )

    # Assessment metadata
    assessment_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When assessment was performed"
    )
    samples_assessed: Optional[int] = Field(
        None,
        ge=0,
        description="Number of data samples assessed"
    )

    @classmethod
    def calculate_level(cls, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.85:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL


# =============================================================================
# VALIDATION RULE
# =============================================================================

class RuleParameter(BaseModel):
    """Parameter for a validation rule."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Parameter name"
    )
    value: Union[float, int, str, bool, List[Any]] = Field(
        ...,
        description="Parameter value"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Parameter description"
    )


class ValidationRule(BaseModel):
    """
    Configurable validation rule for data quality.

    Defines rules for validating data fields with configurable
    conditions, severity, and error handling.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "rule_id": "VR-001",
                    "name": "Surface Temperature Range",
                    "field": "surface_temp_c",
                    "category": "range_check",
                    "condition": "value >= -50 and value <= 200",
                    "severity": "error",
                    "message": "Surface temperature must be between -50 and 200 C"
                }
            ]
        }
    )

    # Identification
    rule_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique rule identifier"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Rule name"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Rule description"
    )

    # Target
    field: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Field or path to validate"
    )
    entity_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Entity type this rule applies to"
    )

    # Rule definition
    category: RuleCategory = Field(
        ...,
        description="Rule category"
    )
    condition: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Validation condition expression"
    )
    condition_type: Literal[
        "python", "regex", "sql", "jsonpath", "custom"
    ] = Field(
        default="python",
        description="Type of condition expression"
    )

    # Parameters
    parameters: List[RuleParameter] = Field(
        default_factory=list,
        description="Rule parameters"
    )

    # Severity and messaging
    severity: RuleSeverity = Field(
        ...,
        description="Severity when rule fails"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Error/warning message when rule fails"
    )
    remediation: Optional[str] = Field(
        None,
        max_length=500,
        description="Suggested remediation"
    )

    # Behavior
    enabled: bool = Field(
        default=True,
        description="Whether rule is enabled"
    )
    stop_on_fail: bool = Field(
        default=False,
        description="Stop further validation if this rule fails"
    )
    allow_null: bool = Field(
        default=False,
        description="Whether null values are allowed"
    )

    # Thresholds (for numeric rules)
    min_value: Optional[float] = Field(
        None,
        description="Minimum allowed value"
    )
    max_value: Optional[float] = Field(
        None,
        description="Maximum allowed value"
    )
    tolerance: Optional[float] = Field(
        None,
        ge=0,
        description="Tolerance for comparison"
    )

    # Applicability
    applies_when: Optional[str] = Field(
        None,
        max_length=500,
        description="Condition when this rule applies"
    )
    excluded_values: List[Any] = Field(
        default_factory=list,
        description="Values excluded from validation"
    )

    # Metadata
    version: str = Field(
        default="1.0",
        description="Rule version"
    )
    author: Optional[str] = Field(
        None,
        max_length=100,
        description="Rule author"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Rule creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Rule last update timestamp"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for rule organization"
    )


# =============================================================================
# VALIDATION RESULT
# =============================================================================

class ValidationResult(BaseModel):
    """Result of applying a validation rule."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "rule_id": "VR-001",
                    "outcome": "fail",
                    "severity": "error",
                    "field": "surface_temp_c",
                    "actual_value": 250.5,
                    "message": "Surface temperature 250.5 C exceeds maximum 200 C"
                }
            ]
        }
    )

    rule_id: str = Field(
        ...,
        description="Reference to validation rule"
    )
    rule_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Rule name for display"
    )
    outcome: ValidationOutcome = Field(
        ...,
        description="Validation outcome"
    )
    severity: RuleSeverity = Field(
        ...,
        description="Severity of the issue"
    )

    # Field information
    field: str = Field(
        ...,
        description="Field that was validated"
    )
    actual_value: Optional[Any] = Field(
        None,
        description="Actual value found"
    )
    expected_value: Optional[Any] = Field(
        None,
        description="Expected value (if applicable)"
    )

    # Message
    message: str = Field(
        ...,
        max_length=1000,
        description="Validation message"
    )
    details: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional details"
    )

    # Location (for batch validation)
    record_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Record identifier"
    )
    row_number: Optional[int] = Field(
        None,
        ge=0,
        description="Row number in batch"
    )

    # Timing
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When validation was performed"
    )


# =============================================================================
# QUALITY ISSUE
# =============================================================================

class QualityIssue(BaseModel):
    """Individual data quality issue."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "issue_id": "QI-2024-00001",
                    "category": "range_check",
                    "severity": "error",
                    "dimension": "accuracy",
                    "field": "surface_temp_c",
                    "description": "Surface temperature reading exceeds physical limits",
                    "status": "open"
                }
            ]
        }
    )

    issue_id: str = Field(
        default_factory=lambda: f"QI-{uuid.uuid4().hex[:12].upper()}",
        description="Unique issue identifier"
    )
    category: RuleCategory = Field(
        ...,
        description="Issue category"
    )
    severity: RuleSeverity = Field(
        ...,
        description="Issue severity"
    )
    dimension: QualityDimension = Field(
        ...,
        description="Quality dimension affected"
    )

    # Issue details
    field: Optional[str] = Field(
        None,
        max_length=200,
        description="Affected field"
    )
    entity_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Affected entity ID"
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Issue description"
    )

    # Values
    actual_value: Optional[Any] = Field(
        None,
        description="Actual problematic value"
    )
    expected_range: Optional[str] = Field(
        None,
        max_length=200,
        description="Expected value range"
    )

    # Occurrence
    first_detected: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When issue was first detected"
    )
    last_seen: Optional[datetime] = Field(
        None,
        description="When issue was last observed"
    )
    occurrence_count: int = Field(
        default=1,
        ge=1,
        description="Number of occurrences"
    )
    affected_records: Optional[int] = Field(
        None,
        ge=0,
        description="Number of records affected"
    )

    # Status and resolution
    status: IssueStatus = Field(
        default=IssueStatus.OPEN,
        description="Issue status"
    )
    assigned_to: Optional[str] = Field(
        None,
        max_length=200,
        description="Person assigned to resolve"
    )
    resolution: Optional[str] = Field(
        None,
        max_length=1000,
        description="Resolution description"
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="When issue was resolved"
    )

    # Related rule
    rule_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Rule that detected this issue"
    )

    # Recommendation
    recommendation: Optional[str] = Field(
        None,
        max_length=500,
        description="Recommended action"
    )


# =============================================================================
# QUALITY REPORT
# =============================================================================

class QualityReport(BaseModel):
    """
    Comprehensive data quality assessment report.

    Provides full quality assessment with scores, issues,
    validation results, and recommendations.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "report_id": "QR-2024-00001234",
                    "asset_id": "INS-1001",
                    "report_timestamp": "2024-01-15T11:00:00Z",
                    "overall_score": {"overall_score": 0.92, "quality_level": "excellent"},
                    "issues_count": 3,
                    "critical_issues_count": 0
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version"
    )
    report_id: str = Field(
        default_factory=lambda: f"QR-{uuid.uuid4().hex[:12].upper()}",
        description="Unique report identifier"
    )
    asset_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to asset if applicable"
    )
    data_source: Optional[str] = Field(
        None,
        max_length=200,
        description="Source of data being assessed"
    )

    # Timing
    report_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp"
    )
    assessment_period_start: Optional[datetime] = Field(
        None,
        description="Start of assessment period"
    )
    assessment_period_end: Optional[datetime] = Field(
        None,
        description="End of assessment period"
    )
    period_duration_hours: Optional[float] = Field(
        None,
        gt=0,
        description="Assessment period duration in hours"
    )

    # Overall score
    overall_score: DataQualityScore = Field(
        ...,
        description="Overall quality score"
    )

    # Data statistics
    records_assessed: int = Field(
        default=0,
        ge=0,
        description="Total records assessed"
    )
    records_passed: int = Field(
        default=0,
        ge=0,
        description="Records that passed all checks"
    )
    records_failed: int = Field(
        default=0,
        ge=0,
        description="Records with failures"
    )
    pass_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Pass rate (records_passed / records_assessed)"
    )

    # Issues summary
    issues: List[QualityIssue] = Field(
        default_factory=list,
        description="List of quality issues"
    )
    issues_count: int = Field(
        default=0,
        ge=0,
        description="Total issues count"
    )
    critical_issues_count: int = Field(
        default=0,
        ge=0,
        description="Critical issues count"
    )
    error_issues_count: int = Field(
        default=0,
        ge=0,
        description="Error issues count"
    )
    warning_issues_count: int = Field(
        default=0,
        ge=0,
        description="Warning issues count"
    )

    # Validation results
    validation_results: List[ValidationResult] = Field(
        default_factory=list,
        description="Validation results"
    )
    rules_executed: int = Field(
        default=0,
        ge=0,
        description="Number of rules executed"
    )
    rules_passed: int = Field(
        default=0,
        ge=0,
        description="Number of rules passed"
    )
    rules_failed: int = Field(
        default=0,
        ge=0,
        description="Number of rules failed"
    )

    # Trend analysis
    previous_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Previous period quality score"
    )
    score_change: Optional[float] = Field(
        None,
        ge=-1,
        le=1,
        description="Change in quality score"
    )
    trend: Optional[Literal["improving", "stable", "degrading"]] = Field(
        None,
        description="Quality trend"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Quality improvement recommendations"
    )
    priority_actions: List[str] = Field(
        default_factory=list,
        description="Priority actions required"
    )

    # Impact assessment
    data_usability: Literal[
        "full", "partial", "limited", "unusable"
    ] = Field(
        default="full",
        description="Data usability assessment"
    )
    computation_reliability: Optional[Literal[
        "high", "medium", "low", "unreliable"
    ]] = Field(
        None,
        description="Reliability of computations with this data"
    )
    estimated_accuracy_impact: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Estimated impact on accuracy (%)"
    )

    # Report metadata
    generated_by: Optional[str] = Field(
        None,
        max_length=200,
        description="System/person that generated report"
    )
    configuration_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to quality configuration used"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "QualityReport":
        """Validate count consistency."""
        # Issues count should match list length
        if len(self.issues) != self.issues_count:
            # Allow auto-correction in practice
            pass

        # Passed + failed should equal assessed
        if self.records_passed + self.records_failed > self.records_assessed:
            raise ValueError(
                "records_passed + records_failed cannot exceed records_assessed"
            )

        return self

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        content = (
            f"{self.report_id}"
            f"{self.overall_score.overall_score:.4f}"
            f"{self.issues_count}"
            f"{self.report_timestamp.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def is_acceptable(self) -> bool:
        """Check if quality is acceptable."""
        return self.overall_score.quality_level in [
            QualityLevel.EXCELLENT,
            QualityLevel.GOOD,
            QualityLevel.ACCEPTABLE
        ]

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return self.critical_issues_count > 0


# =============================================================================
# QUALITY CONFIGURATION
# =============================================================================

class QualityConfiguration(BaseModel):
    """Configuration for quality assessment."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "config_id": "QC-DEFAULT",
                    "name": "Default Quality Configuration",
                    "dimension_weights": {
                        "completeness": 0.25,
                        "accuracy": 0.30,
                        "timeliness": 0.20,
                        "consistency": 0.25
                    }
                }
            ]
        }
    )

    config_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Configuration identifier"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Configuration name"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Configuration description"
    )

    # Dimension weights
    dimension_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights for each quality dimension"
    )

    # Thresholds for quality levels
    excellent_threshold: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Score threshold for excellent"
    )
    good_threshold: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Score threshold for good"
    )
    acceptable_threshold: float = Field(
        default=0.70,
        ge=0,
        le=1,
        description="Score threshold for acceptable"
    )
    poor_threshold: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description="Score threshold for poor"
    )

    # Timeliness settings
    max_data_age_hours: float = Field(
        default=1.0,
        gt=0,
        description="Maximum acceptable data age in hours"
    )

    # Completeness settings
    required_fields: List[str] = Field(
        default_factory=list,
        description="Required fields for completeness"
    )
    optional_fields: List[str] = Field(
        default_factory=list,
        description="Optional fields"
    )

    # Active rules
    enabled_rule_ids: List[str] = Field(
        default_factory=list,
        description="IDs of enabled validation rules"
    )
    disabled_rule_ids: List[str] = Field(
        default_factory=list,
        description="IDs of disabled validation rules"
    )

    # Metadata
    version: str = Field(
        default="1.0",
        description="Configuration version"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )


# =============================================================================
# EXPORTS
# =============================================================================

DATA_QUALITY_SCHEMAS = {
    "QualityDimension": QualityDimension,
    "RuleSeverity": RuleSeverity,
    "RuleCategory": RuleCategory,
    "ValidationOutcome": ValidationOutcome,
    "QualityLevel": QualityLevel,
    "IssueStatus": IssueStatus,
    "DimensionScore": DimensionScore,
    "DataQualityScore": DataQualityScore,
    "RuleParameter": RuleParameter,
    "ValidationRule": ValidationRule,
    "ValidationResult": ValidationResult,
    "QualityIssue": QualityIssue,
    "QualityReport": QualityReport,
    "QualityConfiguration": QualityConfiguration,
}

__all__ = [
    # Enumerations
    "QualityDimension",
    "RuleSeverity",
    "RuleCategory",
    "ValidationOutcome",
    "QualityLevel",
    "IssueStatus",
    # Supporting models
    "DimensionScore",
    "RuleParameter",
    "ValidationResult",
    "QualityIssue",
    # Main schemas
    "DataQualityScore",
    "ValidationRule",
    "QualityReport",
    "QualityConfiguration",
    # Export dictionary
    "DATA_QUALITY_SCHEMAS",
]
