# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro: Data Quality Schemas - Version 1.0

Provides validated data schemas for data quality assessment, sensor validation,
and quality reporting for heat exchanger monitoring data.

This module defines Pydantic v2 models for:
- DataQualityReport: Comprehensive quality assessment with metrics
- SensorValidation: Individual sensor validation status
- Quality assessment tools for exchanger-specific validation

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class QualitySeverity(str, Enum):
    """Severity levels for quality issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityDimension(str, Enum):
    """Dimensions of data quality."""
    COMPLETENESS = "completeness"  # All required data present
    ACCURACY = "accuracy"  # Values within expected ranges
    TIMELINESS = "timeliness"  # Data is recent enough
    CONSISTENCY = "consistency"  # Values are internally consistent
    VALIDITY = "validity"  # Values pass validation rules
    UNIQUENESS = "uniqueness"  # No duplicate records
    RELIABILITY = "reliability"  # Sensor/source reliability


class SensorStatus(str, Enum):
    """Sensor operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTY = "faulty"
    OFFLINE = "offline"
    CALIBRATION_DUE = "calibration_due"
    SUSPECT = "suspect"
    UNKNOWN = "unknown"


class DriftType(str, Enum):
    """Types of sensor drift detected."""
    NO_DRIFT = "no_drift"
    OFFSET_DRIFT = "offset_drift"  # Constant offset change
    GAIN_DRIFT = "gain_drift"  # Scale factor change
    NONLINEAR_DRIFT = "nonlinear_drift"
    STEP_CHANGE = "step_change"  # Sudden shift
    NOISE_INCREASE = "noise_increase"


class IssueCategory(str, Enum):
    """Categories of data quality issues."""
    MISSING_DATA = "missing_data"
    OUT_OF_RANGE = "out_of_range"
    STUCK_SENSOR = "stuck_sensor"
    SENSOR_DRIFT = "sensor_drift"
    ENERGY_BALANCE = "energy_balance"
    IMPOSSIBLE_VALUE = "impossible_value"
    TIMESTAMP_ISSUE = "timestamp_issue"
    DUPLICATE_DATA = "duplicate_data"
    INCONSISTENT_DATA = "inconsistent_data"
    COMMUNICATION_FAILURE = "communication_failure"


# =============================================================================
# QUALITY ISSUE
# =============================================================================

class QualityIssue(BaseModel):
    """Represents a single data quality issue."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "issue_id": "QI-2024-00001",
                    "category": "stuck_sensor",
                    "dimension": "reliability",
                    "severity": "warning",
                    "sensor_id": "TI-1001",
                    "description": "Temperature sensor stuck at 185.0 C for 45 minutes",
                    "first_detected": "2024-01-15T10:00:00Z",
                    "sample_count": 9
                }
            ]
        }
    )

    issue_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique issue identifier"
    )
    category: IssueCategory = Field(
        ...,
        description="Category of quality issue"
    )
    dimension: QualityDimension = Field(
        ...,
        description="Quality dimension affected"
    )
    severity: QualitySeverity = Field(
        ...,
        description="Issue severity level"
    )

    # Context
    sensor_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Affected sensor ID"
    )
    tag_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Affected tag name"
    )
    exchanger_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Affected exchanger ID"
    )

    # Issue details
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Detailed issue description"
    )
    first_detected: datetime = Field(
        ...,
        description="When issue was first detected"
    )
    last_seen: Optional[datetime] = Field(
        None,
        description="When issue was last observed"
    )
    sample_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of samples affected"
    )

    # Values
    expected_value: Optional[float] = Field(
        None,
        description="Expected value (if applicable)"
    )
    actual_value: Optional[float] = Field(
        None,
        description="Actual observed value"
    )
    deviation: Optional[float] = Field(
        None,
        description="Deviation from expected"
    )
    threshold: Optional[float] = Field(
        None,
        description="Threshold that was violated"
    )

    # Resolution
    auto_corrected: bool = Field(
        default=False,
        description="Whether issue was auto-corrected"
    )
    correction_applied: Optional[str] = Field(
        None,
        max_length=500,
        description="Description of correction applied"
    )
    recommendation: Optional[str] = Field(
        None,
        max_length=500,
        description="Recommended action"
    )


# =============================================================================
# SENSOR VALIDATION
# =============================================================================

class CalibrationRecord(BaseModel):
    """Sensor calibration record."""

    model_config = ConfigDict(frozen=True)

    calibration_id: str = Field(
        ...,
        description="Unique calibration identifier"
    )
    calibration_date: datetime = Field(
        ...,
        description="Date of calibration"
    )
    calibration_due: Optional[datetime] = Field(
        None,
        description="Next calibration due date"
    )
    calibration_interval_days: Optional[int] = Field(
        None,
        ge=1,
        description="Calibration interval in days"
    )

    # Calibration results
    as_found_error: Optional[float] = Field(
        None,
        description="As-found error before calibration"
    )
    as_left_error: Optional[float] = Field(
        None,
        description="As-left error after calibration"
    )
    tolerance: Optional[float] = Field(
        None,
        ge=0,
        description="Acceptable tolerance"
    )
    passed: bool = Field(
        default=True,
        description="Whether calibration passed"
    )

    # Reference
    reference_standard: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference standard used"
    )
    technician: Optional[str] = Field(
        None,
        max_length=100,
        description="Calibration technician"
    )
    certificate_ref: Optional[str] = Field(
        None,
        max_length=100,
        description="Calibration certificate reference"
    )


class DriftAnalysis(BaseModel):
    """Sensor drift analysis results."""

    model_config = ConfigDict(frozen=True)

    drift_type: DriftType = Field(
        ...,
        description="Type of drift detected"
    )
    drift_detected: bool = Field(
        ...,
        description="Whether significant drift was detected"
    )
    drift_rate: Optional[float] = Field(
        None,
        description="Drift rate (units per day)"
    )
    drift_magnitude: Optional[float] = Field(
        None,
        description="Total drift magnitude"
    )
    detection_confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Confidence in drift detection"
    )
    analysis_period_days: Optional[int] = Field(
        None,
        ge=1,
        description="Analysis period in days"
    )
    samples_analyzed: Optional[int] = Field(
        None,
        ge=1,
        description="Number of samples analyzed"
    )
    trend_slope: Optional[float] = Field(
        None,
        description="Linear trend slope"
    )
    trend_r_squared: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="R-squared of trend fit"
    )


class SensorValidation(BaseModel):
    """
    Validation status for individual sensor.

    Tracks sensor health, calibration status, drift detection,
    and historical performance for reliability assessment.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "sensor_id": "TI-1001",
                    "tag_name": "TI-1001.PV",
                    "sensor_type": "thermocouple",
                    "status": "healthy",
                    "reliability_score": 0.95,
                    "last_reading_timestamp": "2024-01-15T10:30:00Z",
                    "last_reading_value": 185.5,
                    "in_range": True,
                    "calibration_valid": True
                }
            ]
        }
    )

    # Identifiers
    sensor_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique sensor identifier"
    )
    tag_name: Optional[str] = Field(
        None,
        max_length=200,
        description="SCADA/Historian tag name"
    )
    exchanger_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Associated exchanger ID"
    )
    location: Optional[str] = Field(
        None,
        max_length=200,
        description="Sensor location description"
    )

    # Sensor metadata
    sensor_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Sensor type (e.g., thermocouple, RTD, pressure transmitter)"
    )
    manufacturer: Optional[str] = Field(
        None,
        max_length=100,
        description="Sensor manufacturer"
    )
    model: Optional[str] = Field(
        None,
        max_length=100,
        description="Sensor model"
    )
    range_min: Optional[float] = Field(
        None,
        description="Sensor minimum range"
    )
    range_max: Optional[float] = Field(
        None,
        description="Sensor maximum range"
    )
    unit: Optional[str] = Field(
        None,
        max_length=50,
        description="Engineering unit"
    )

    # Current status
    status: SensorStatus = Field(
        ...,
        description="Current sensor status"
    )
    status_reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Reason for current status"
    )
    reliability_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Reliability score (0-1)"
    )

    # Latest reading
    last_reading_timestamp: Optional[datetime] = Field(
        None,
        description="Timestamp of last reading"
    )
    last_reading_value: Optional[float] = Field(
        None,
        description="Value of last reading"
    )
    last_reading_quality: Optional[str] = Field(
        None,
        max_length=50,
        description="Quality code of last reading"
    )

    # Validation checks
    in_range: bool = Field(
        default=True,
        description="Whether latest reading is within valid range"
    )
    is_stuck: bool = Field(
        default=False,
        description="Whether sensor appears stuck"
    )
    stuck_duration_minutes: Optional[float] = Field(
        None,
        ge=0,
        description="Duration sensor has been stuck"
    )
    noise_level: Optional[float] = Field(
        None,
        ge=0,
        description="Current noise level (std dev)"
    )
    expected_noise_level: Optional[float] = Field(
        None,
        ge=0,
        description="Expected normal noise level"
    )

    # Calibration
    calibration_valid: bool = Field(
        default=True,
        description="Whether calibration is valid"
    )
    last_calibration: Optional[CalibrationRecord] = Field(
        None,
        description="Last calibration record"
    )
    calibration_overdue_days: Optional[int] = Field(
        None,
        ge=0,
        description="Days calibration is overdue"
    )

    # Drift detection
    drift_detected: bool = Field(
        default=False,
        description="Whether drift has been detected"
    )
    drift_analysis: Optional[DriftAnalysis] = Field(
        None,
        description="Drift analysis results"
    )

    # Historical metrics
    availability_30d: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="30-day data availability"
    )
    quality_good_30d: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Percentage of good quality readings (30-day)"
    )
    out_of_range_count_30d: Optional[int] = Field(
        None,
        ge=0,
        description="Count of out-of-range readings (30-day)"
    )

    # Issues
    active_issues: List[str] = Field(
        default_factory=list,
        description="List of active issue IDs for this sensor"
    )

    # Timestamps
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When validation was performed"
    )

    @property
    def is_healthy(self) -> bool:
        """Check if sensor is in healthy state."""
        return self.status == SensorStatus.HEALTHY

    @property
    def needs_attention(self) -> bool:
        """Check if sensor needs attention."""
        return self.status in [
            SensorStatus.DEGRADED,
            SensorStatus.CALIBRATION_DUE,
            SensorStatus.SUSPECT,
        ]


# =============================================================================
# DATA QUALITY METRICS
# =============================================================================

class CompletenessMetrics(BaseModel):
    """Completeness metrics for data quality."""

    model_config = ConfigDict(frozen=True)

    total_expected: int = Field(
        ...,
        ge=0,
        description="Total expected data points"
    )
    total_present: int = Field(
        ...,
        ge=0,
        description="Total present data points"
    )
    total_missing: int = Field(
        ...,
        ge=0,
        description="Total missing data points"
    )
    completeness_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Completeness score (0-1)"
    )
    missing_by_tag: Dict[str, int] = Field(
        default_factory=dict,
        description="Missing count by tag name"
    )


class RangeViolationMetrics(BaseModel):
    """Range violation metrics for data quality."""

    model_config = ConfigDict(frozen=True)

    total_samples: int = Field(
        ...,
        ge=0,
        description="Total samples checked"
    )
    violations_count: int = Field(
        ...,
        ge=0,
        description="Number of range violations"
    )
    violation_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Violation rate (0-1)"
    )
    violations_by_tag: Dict[str, int] = Field(
        default_factory=dict,
        description="Violation count by tag name"
    )
    worst_violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Details of worst violations"
    )


class StuckSensorMetrics(BaseModel):
    """Stuck sensor detection metrics."""

    model_config = ConfigDict(frozen=True)

    sensors_checked: int = Field(
        ...,
        ge=0,
        description="Number of sensors checked"
    )
    stuck_sensors_count: int = Field(
        ...,
        ge=0,
        description="Number of stuck sensors detected"
    )
    stuck_sensor_ids: List[str] = Field(
        default_factory=list,
        description="IDs of stuck sensors"
    )
    detection_threshold_minutes: float = Field(
        ...,
        gt=0,
        description="Threshold for stuck detection (minutes)"
    )
    detection_tolerance: float = Field(
        ...,
        ge=0,
        description="Value tolerance for stuck detection"
    )


class EnergyBalanceMetrics(BaseModel):
    """Energy balance check metrics for heat exchangers."""

    model_config = ConfigDict(frozen=True)

    q_hot_kw: float = Field(
        ...,
        description="Heat duty from hot side (kW)"
    )
    q_cold_kw: float = Field(
        ...,
        description="Heat duty from cold side (kW)"
    )
    imbalance_kw: float = Field(
        ...,
        description="Absolute imbalance (kW)"
    )
    imbalance_percent: float = Field(
        ...,
        description="Imbalance as percentage of average duty"
    )
    balance_acceptable: bool = Field(
        ...,
        description="Whether balance is within tolerance"
    )
    tolerance_percent: float = Field(
        ...,
        ge=0,
        description="Tolerance threshold (%)"
    )
    possible_causes: List[str] = Field(
        default_factory=list,
        description="Possible causes of imbalance"
    )


# =============================================================================
# DATA QUALITY REPORT
# =============================================================================

class DataQualityReport(BaseModel):
    """
    Comprehensive data quality report for heat exchanger data.

    Provides detailed quality assessment across all dimensions
    with metrics, issues, and recommendations.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "report_id": "DQR-2024-00001234",
                    "exchanger_id": "HX-1001",
                    "report_timestamp": "2024-01-15T11:00:00Z",
                    "period_start": "2024-01-15T10:00:00Z",
                    "period_end": "2024-01-15T11:00:00Z",
                    "overall_quality_score": 0.92,
                    "quality_status": "good"
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    report_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique report identifier"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to exchanger asset"
    )

    # Time period
    report_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When report was generated"
    )
    period_start: datetime = Field(
        ...,
        description="Start of assessment period"
    )
    period_end: datetime = Field(
        ...,
        description="End of assessment period"
    )
    period_duration_hours: Optional[float] = Field(
        None,
        gt=0,
        description="Duration of assessment period in hours"
    )

    # Overall scores
    overall_quality_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall quality score (0-1)"
    )
    quality_status: Literal[
        "excellent", "good", "acceptable", "poor", "critical"
    ] = Field(
        ...,
        description="Quality status classification"
    )

    # Dimension scores
    completeness_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Completeness dimension score"
    )
    accuracy_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Accuracy dimension score"
    )
    timeliness_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Timeliness dimension score"
    )
    consistency_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Consistency dimension score"
    )
    reliability_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Reliability dimension score"
    )

    # Detailed metrics
    missingness: CompletenessMetrics = Field(
        ...,
        description="Completeness/missingness metrics"
    )
    range_violations: RangeViolationMetrics = Field(
        ...,
        description="Range violation metrics"
    )
    stuck_sensors: StuckSensorMetrics = Field(
        ...,
        description="Stuck sensor detection metrics"
    )
    energy_balance_deviation: Optional[EnergyBalanceMetrics] = Field(
        None,
        description="Energy balance check metrics"
    )

    # Sensor validation
    sensor_validations: List[SensorValidation] = Field(
        default_factory=list,
        description="Validation results for each sensor"
    )
    healthy_sensors: int = Field(
        default=0,
        ge=0,
        description="Count of healthy sensors"
    )
    degraded_sensors: int = Field(
        default=0,
        ge=0,
        description="Count of degraded sensors"
    )
    faulty_sensors: int = Field(
        default=0,
        ge=0,
        description="Count of faulty sensors"
    )

    # Issues
    issues: List[QualityIssue] = Field(
        default_factory=list,
        description="List of quality issues found"
    )
    critical_issues_count: int = Field(
        default=0,
        ge=0,
        description="Count of critical issues"
    )
    warning_issues_count: int = Field(
        default=0,
        ge=0,
        description="Count of warning issues"
    )

    # Impact assessment
    predictions_affected: int = Field(
        default=0,
        ge=0,
        description="Number of predictions affected by quality issues"
    )
    predictions_blocked: int = Field(
        default=0,
        ge=0,
        description="Number of predictions blocked due to quality"
    )
    computation_reliability: Optional[Literal[
        "high", "medium", "low", "unreliable"
    ]] = Field(
        None,
        description="Reliability of computations with this data"
    )

    # Trend analysis
    quality_trend: Optional[Literal[
        "improving", "stable", "degrading"
    ]] = Field(
        None,
        description="Quality trend over recent period"
    )
    previous_quality_score: Optional[float] = Field(
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

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for quality improvement"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )

    @model_validator(mode="after")
    def validate_period(self) -> "DataQualityReport":
        """Validate time period."""
        if self.period_end < self.period_start:
            raise ValueError("Period end must be after period start")
        return self

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for report provenance."""
        content = (
            f"{self.report_id}"
            f"{self.exchanger_id}"
            f"{self.overall_quality_score:.4f}"
            f"{self.report_timestamp.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def is_acceptable(self) -> bool:
        """Check if quality is acceptable for analysis."""
        return self.quality_status in ["excellent", "good", "acceptable"]

    @property
    def total_sensors(self) -> int:
        """Get total sensor count."""
        return self.healthy_sensors + self.degraded_sensors + self.faulty_sensors


# =============================================================================
# DATA QUALITY ASSESSOR
# =============================================================================

class QualityThresholds(BaseModel):
    """Configurable thresholds for quality assessment."""

    model_config = ConfigDict(frozen=True)

    min_completeness: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Minimum completeness threshold"
    )
    max_staleness_minutes: float = Field(
        default=10.0,
        gt=0,
        description="Maximum data staleness in minutes"
    )
    max_range_violation_rate: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description="Maximum allowed range violation rate"
    )
    stuck_sensor_threshold_minutes: float = Field(
        default=30.0,
        gt=0,
        description="Minutes of constant value to flag as stuck"
    )
    stuck_sensor_tolerance: float = Field(
        default=0.1,
        ge=0,
        description="Value tolerance for stuck detection (% of range)"
    )
    max_energy_imbalance_percent: float = Field(
        default=5.0,
        ge=0,
        le=100,
        description="Maximum acceptable energy imbalance (%)"
    )
    sensor_reliability_threshold: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description="Minimum sensor reliability score"
    )

    # Score thresholds for quality classification
    excellent_threshold: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Threshold for excellent quality"
    )
    good_threshold: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Threshold for good quality"
    )
    acceptable_threshold: float = Field(
        default=0.70,
        ge=0,
        le=1,
        description="Threshold for acceptable quality"
    )
    poor_threshold: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description="Threshold for poor quality (below is critical)"
    )


# =============================================================================
# EXPORTS
# =============================================================================

DATA_QUALITY_SCHEMAS = {
    "QualitySeverity": QualitySeverity,
    "QualityDimension": QualityDimension,
    "SensorStatus": SensorStatus,
    "DriftType": DriftType,
    "IssueCategory": IssueCategory,
    "QualityIssue": QualityIssue,
    "CalibrationRecord": CalibrationRecord,
    "DriftAnalysis": DriftAnalysis,
    "SensorValidation": SensorValidation,
    "CompletenessMetrics": CompletenessMetrics,
    "RangeViolationMetrics": RangeViolationMetrics,
    "StuckSensorMetrics": StuckSensorMetrics,
    "EnergyBalanceMetrics": EnergyBalanceMetrics,
    "DataQualityReport": DataQualityReport,
    "QualityThresholds": QualityThresholds,
}

__all__ = [
    # Enumerations
    "QualitySeverity",
    "QualityDimension",
    "SensorStatus",
    "DriftType",
    "IssueCategory",
    # Quality issue model
    "QualityIssue",
    # Sensor validation
    "CalibrationRecord",
    "DriftAnalysis",
    "SensorValidation",
    # Metrics models
    "CompletenessMetrics",
    "RangeViolationMetrics",
    "StuckSensorMetrics",
    "EnergyBalanceMetrics",
    # Main schemas
    "DataQualityReport",
    "QualityThresholds",
    # Export dictionary
    "DATA_QUALITY_SCHEMAS",
]
