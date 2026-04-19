# -*- coding: utf-8 -*-
"""
Data Quality Module for GL-013 PredictiveMaintenance Agent.

Provides data quality assessment, validation, and reporting
to ensure reliable predictions with zero-hallucination guarantees.

Features:
- Completeness checking
- Value range validation
- Outlier detection
- Staleness detection
- Consistency checking
- Quality scoring and reporting

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


class QualitySeverity(str, Enum):
    """Severity levels for quality issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityDimension(str, Enum):
    """Dimensions of data quality."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


class QualityIssue(BaseModel):
    """Represents a data quality issue."""
    issue_id: str = Field(..., description="Unique issue identifier")
    dimension: QualityDimension
    severity: QualitySeverity
    sensor_id: Optional[str] = None
    feature_name: Optional[str] = None
    description: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Issue details
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    deviation: Optional[float] = None

    # Resolution
    auto_corrected: bool = False
    correction_applied: Optional[str] = None
    recommendation: Optional[str] = None


class DataQualityMetrics(BaseModel):
    """Metrics for data quality assessment."""
    # Completeness
    total_fields: int = Field(..., ge=0)
    present_fields: int = Field(..., ge=0)
    completeness_score: float = Field(..., ge=0, le=1)

    # Timeliness
    most_recent_timestamp: Optional[datetime] = None
    staleness_seconds: float = Field(0, ge=0)
    timeliness_score: float = Field(..., ge=0, le=1)

    # Validity
    valid_values: int = Field(..., ge=0)
    invalid_values: int = Field(..., ge=0)
    validity_score: float = Field(..., ge=0, le=1)

    # Consistency
    consistency_checks_passed: int = Field(..., ge=0)
    consistency_checks_failed: int = Field(..., ge=0)
    consistency_score: float = Field(..., ge=0, le=1)

    # Accuracy (based on outlier detection)
    outliers_detected: int = Field(..., ge=0)
    accuracy_score: float = Field(..., ge=0, le=1)

    # Overall
    overall_quality_score: float = Field(..., ge=0, le=1)

    # Issues
    issues: List[QualityIssue] = Field(default_factory=list)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def critical_issues(self) -> int:
        return sum(1 for i in self.issues if i.severity == QualitySeverity.CRITICAL)


class QualityReport(BaseModel):
    """Comprehensive data quality report."""
    report_id: str = Field(..., description="Unique report identifier")
    asset_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    report_period_start: datetime
    report_period_end: datetime

    # Metrics
    metrics: DataQualityMetrics

    # Trend analysis
    quality_trend: str = Field("stable", description="improving, stable, degrading")
    previous_quality_score: Optional[float] = None
    score_change: Optional[float] = None

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    # Affected predictions
    predictions_affected: int = Field(0, ge=0)
    predictions_blocked: int = Field(0, ge=0)

    # Provenance
    provenance_hash: str = ""

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for report provenance."""
        content = (
            f"{self.report_id}{self.asset_id}"
            f"{self.metrics.overall_quality_score:.4f}"
            f"{self.timestamp}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ValidationRule:
    """Defines a validation rule for a data field."""
    field_name: str
    rule_type: str  # range, regex, enum, custom
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: QualitySeverity = QualitySeverity.WARNING
    description: str = ""
    validator_fn: Optional[Callable] = None


class DataValidator:
    """Validates data against defined rules."""

    def __init__(self, rules: Optional[List[ValidationRule]] = None):
        """
        Initialize validator with rules.

        Args:
            rules: List of validation rules
        """
        self.rules: Dict[str, List[ValidationRule]] = {}
        if rules:
            for rule in rules:
                if rule.field_name not in self.rules:
                    self.rules[rule.field_name] = []
                self.rules[rule.field_name].append(rule)

        # Default rules for common fields
        self._add_default_rules()

    def _add_default_rules(self):
        """Add default validation rules for common sensor data."""
        # Temperature rules
        self._add_rule(ValidationRule(
            field_name="temperature",
            rule_type="range",
            parameters={"min": -50, "max": 500},
            severity=QualitySeverity.ERROR,
            description="Temperature must be within reasonable range"
        ))

        # Vibration rules
        self._add_rule(ValidationRule(
            field_name="vibration_rms",
            rule_type="range",
            parameters={"min": 0, "max": 100},
            severity=QualitySeverity.WARNING,
            description="Vibration RMS must be non-negative and reasonable"
        ))

        # Pressure rules
        self._add_rule(ValidationRule(
            field_name="pressure",
            rule_type="range",
            parameters={"min": 0, "max": 1000},
            severity=QualitySeverity.ERROR,
            description="Pressure must be within reasonable range"
        ))

        # Current rules
        self._add_rule(ValidationRule(
            field_name="current_rms",
            rule_type="range",
            parameters={"min": 0, "max": 10000},
            severity=QualitySeverity.WARNING,
            description="Current must be non-negative"
        ))

    def _add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        if rule.field_name not in self.rules:
            self.rules[rule.field_name] = []
        self.rules[rule.field_name].append(rule)

    def add_rule(
        self,
        field_name: str,
        rule_type: str,
        parameters: Dict[str, Any],
        severity: QualitySeverity = QualitySeverity.WARNING,
        description: str = ""
    ) -> None:
        """Add a custom validation rule."""
        self._add_rule(ValidationRule(
            field_name=field_name,
            rule_type=rule_type,
            parameters=parameters,
            severity=severity,
            description=description
        ))

    def validate(
        self,
        data: Dict[str, Any],
        strict: bool = False
    ) -> Tuple[bool, List[QualityIssue]]:
        """
        Validate data against rules.

        Args:
            data: Dictionary of field names to values
            strict: If True, fail on any issue

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        issue_counter = 0

        for field_name, value in data.items():
            if field_name not in self.rules:
                continue

            for rule in self.rules[field_name]:
                is_valid, issue = self._apply_rule(rule, value, issue_counter)
                if not is_valid:
                    issues.append(issue)
                    issue_counter += 1

        # Check for critical issues
        has_critical = any(i.severity == QualitySeverity.CRITICAL for i in issues)

        if strict:
            return len(issues) == 0, issues
        else:
            return not has_critical, issues

    def _apply_rule(
        self,
        rule: ValidationRule,
        value: Any,
        issue_id: int
    ) -> Tuple[bool, Optional[QualityIssue]]:
        """Apply a single validation rule."""

        if value is None:
            return False, QualityIssue(
                issue_id=f"val_{issue_id:04d}",
                dimension=QualityDimension.COMPLETENESS,
                severity=rule.severity,
                feature_name=rule.field_name,
                description=f"Missing value for {rule.field_name}"
            )

        if rule.rule_type == "range":
            min_val = rule.parameters.get("min")
            max_val = rule.parameters.get("max")

            if min_val is not None and value < min_val:
                return False, QualityIssue(
                    issue_id=f"val_{issue_id:04d}",
                    dimension=QualityDimension.VALIDITY,
                    severity=rule.severity,
                    feature_name=rule.field_name,
                    description=f"{rule.field_name} value {value} below minimum {min_val}",
                    expected_value=min_val,
                    actual_value=value,
                    deviation=min_val - value
                )

            if max_val is not None and value > max_val:
                return False, QualityIssue(
                    issue_id=f"val_{issue_id:04d}",
                    dimension=QualityDimension.VALIDITY,
                    severity=rule.severity,
                    feature_name=rule.field_name,
                    description=f"{rule.field_name} value {value} above maximum {max_val}",
                    expected_value=max_val,
                    actual_value=value,
                    deviation=value - max_val
                )

        elif rule.rule_type == "enum":
            allowed_values = rule.parameters.get("values", [])
            if value not in allowed_values:
                return False, QualityIssue(
                    issue_id=f"val_{issue_id:04d}",
                    dimension=QualityDimension.VALIDITY,
                    severity=rule.severity,
                    feature_name=rule.field_name,
                    description=f"{rule.field_name} value '{value}' not in allowed values"
                )

        elif rule.rule_type == "custom" and rule.validator_fn:
            try:
                if not rule.validator_fn(value):
                    return False, QualityIssue(
                        issue_id=f"val_{issue_id:04d}",
                        dimension=QualityDimension.VALIDITY,
                        severity=rule.severity,
                        feature_name=rule.field_name,
                        description=rule.description or f"Custom validation failed for {rule.field_name}"
                    )
            except Exception as e:
                return False, QualityIssue(
                    issue_id=f"val_{issue_id:04d}",
                    dimension=QualityDimension.VALIDITY,
                    severity=QualitySeverity.ERROR,
                    feature_name=rule.field_name,
                    description=f"Validation error: {str(e)}"
                )

        return True, None


class DataQualityAssessor:
    """Assesses data quality for predictive maintenance inputs."""

    def __init__(
        self,
        max_staleness_seconds: float = 600,
        min_completeness: float = 0.95,
        outlier_sigma: float = 4.0,
        historical_data: Optional[Dict[str, List[float]]] = None
    ):
        """
        Initialize quality assessor.

        Args:
            max_staleness_seconds: Maximum allowed data age
            min_completeness: Minimum required data completeness
            outlier_sigma: Standard deviations for outlier detection
            historical_data: Historical values for statistical analysis
        """
        self.max_staleness_seconds = max_staleness_seconds
        self.min_completeness = min_completeness
        self.outlier_sigma = outlier_sigma
        self.historical_data = historical_data or {}

        # Initialize validator
        self.validator = DataValidator()

        # Statistics for each feature
        self._feature_stats: Dict[str, Dict[str, float]] = {}

    def assess(
        self,
        data: Dict[str, Any],
        timestamp: datetime,
        required_fields: Optional[List[str]] = None,
        asset_id: str = ""
    ) -> DataQualityMetrics:
        """
        Assess data quality.

        Args:
            data: Dictionary of field names to values
            timestamp: Data timestamp
            required_fields: List of required field names
            asset_id: Asset identifier for context

        Returns:
            DataQualityMetrics with comprehensive quality assessment
        """
        issues = []
        issue_counter = 0

        # 1. Completeness check
        if required_fields:
            total_fields = len(required_fields)
            present_fields = sum(1 for f in required_fields if f in data and data[f] is not None)
        else:
            total_fields = len(data)
            present_fields = sum(1 for v in data.values() if v is not None)

        completeness_score = present_fields / max(1, total_fields)

        if completeness_score < self.min_completeness:
            missing = [f for f in (required_fields or data.keys()) if f not in data or data[f] is None]
            issues.append(QualityIssue(
                issue_id=f"qual_{issue_counter:04d}",
                dimension=QualityDimension.COMPLETENESS,
                severity=QualitySeverity.WARNING if completeness_score > 0.8 else QualitySeverity.ERROR,
                description=f"Data completeness {completeness_score:.1%} below threshold. Missing: {missing[:5]}",
                recommendation="Check sensor connectivity and data pipeline"
            ))
            issue_counter += 1

        # 2. Timeliness check
        now = datetime.utcnow()
        staleness_seconds = (now - timestamp).total_seconds()
        timeliness_score = max(0, 1 - (staleness_seconds / self.max_staleness_seconds))

        if staleness_seconds > self.max_staleness_seconds:
            issues.append(QualityIssue(
                issue_id=f"qual_{issue_counter:04d}",
                dimension=QualityDimension.TIMELINESS,
                severity=QualitySeverity.WARNING,
                description=f"Data is {staleness_seconds:.0f}s old (max: {self.max_staleness_seconds}s)",
                recommendation="Check data streaming latency"
            ))
            issue_counter += 1

        # 3. Validity check (using validator)
        is_valid, validation_issues = self.validator.validate(data)
        issues.extend(validation_issues)
        issue_counter += len(validation_issues)

        valid_values = sum(1 for v in data.values() if v is not None)
        invalid_values = len(validation_issues)
        validity_score = max(0, 1 - (invalid_values / max(1, valid_values)))

        # 4. Outlier detection
        outliers_detected = 0
        for field_name, value in data.items():
            if value is None or not isinstance(value, (int, float)):
                continue

            if field_name in self._feature_stats:
                stats = self._feature_stats[field_name]
                mean = stats.get("mean", value)
                std = stats.get("std", 1)

                if std > 0:
                    z_score = abs(value - mean) / std
                    if z_score > self.outlier_sigma:
                        outliers_detected += 1
                        issues.append(QualityIssue(
                            issue_id=f"qual_{issue_counter:04d}",
                            dimension=QualityDimension.ACCURACY,
                            severity=QualitySeverity.WARNING,
                            feature_name=field_name,
                            description=f"Outlier detected: {field_name}={value} (z-score: {z_score:.2f})",
                            expected_value=mean,
                            actual_value=value,
                            deviation=z_score
                        ))
                        issue_counter += 1

            # Update statistics
            self._update_feature_stats(field_name, value)

        accuracy_score = max(0, 1 - (outliers_detected / max(1, present_fields)))

        # 5. Consistency checks (simple cross-field checks)
        consistency_passed = 0
        consistency_failed = 0

        # Example: temperature should be correlated with thermal features
        if "temperature" in data and "thermal_stress" in data:
            temp = data["temperature"]
            stress = data["thermal_stress"]
            if temp is not None and stress is not None:
                if temp > 100 and stress < 0.1:
                    consistency_failed += 1
                    issues.append(QualityIssue(
                        issue_id=f"qual_{issue_counter:04d}",
                        dimension=QualityDimension.CONSISTENCY,
                        severity=QualitySeverity.WARNING,
                        description="High temperature but low thermal stress - possible sensor issue"
                    ))
                    issue_counter += 1
                else:
                    consistency_passed += 1

        consistency_total = consistency_passed + consistency_failed
        consistency_score = consistency_passed / max(1, consistency_total) if consistency_total > 0 else 1.0

        # 6. Calculate overall quality score (weighted average)
        weights = {
            "completeness": 0.25,
            "timeliness": 0.20,
            "validity": 0.25,
            "accuracy": 0.20,
            "consistency": 0.10
        }

        overall_score = (
            weights["completeness"] * completeness_score +
            weights["timeliness"] * timeliness_score +
            weights["validity"] * validity_score +
            weights["accuracy"] * accuracy_score +
            weights["consistency"] * consistency_score
        )

        return DataQualityMetrics(
            total_fields=total_fields,
            present_fields=present_fields,
            completeness_score=completeness_score,
            most_recent_timestamp=timestamp,
            staleness_seconds=staleness_seconds,
            timeliness_score=timeliness_score,
            valid_values=valid_values,
            invalid_values=invalid_values,
            validity_score=validity_score,
            consistency_checks_passed=consistency_passed,
            consistency_checks_failed=consistency_failed,
            consistency_score=consistency_score,
            outliers_detected=outliers_detected,
            accuracy_score=accuracy_score,
            overall_quality_score=overall_score,
            issues=issues
        )

    def _update_feature_stats(self, field_name: str, value: float) -> None:
        """Update running statistics for a feature."""
        if field_name not in self._feature_stats:
            self._feature_stats[field_name] = {
                "count": 0,
                "mean": 0,
                "M2": 0,  # For Welford's algorithm
                "min": float("inf"),
                "max": float("-inf")
            }

        stats = self._feature_stats[field_name]
        stats["count"] += 1
        n = stats["count"]

        # Welford's online algorithm for mean and variance
        delta = value - stats["mean"]
        stats["mean"] += delta / n
        delta2 = value - stats["mean"]
        stats["M2"] += delta * delta2

        # Update min/max
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)

        # Calculate std
        if n > 1:
            stats["std"] = np.sqrt(stats["M2"] / (n - 1))
        else:
            stats["std"] = 0

    def generate_report(
        self,
        metrics_history: List[DataQualityMetrics],
        asset_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> QualityReport:
        """
        Generate a comprehensive quality report.

        Args:
            metrics_history: List of historical quality metrics
            asset_id: Asset identifier
            period_start: Report period start
            period_end: Report period end

        Returns:
            QualityReport with trend analysis and recommendations
        """
        if not metrics_history:
            raise ValueError("No metrics history provided")

        # Aggregate metrics
        latest = metrics_history[-1]

        # Calculate trend
        if len(metrics_history) >= 2:
            scores = [m.overall_quality_score for m in metrics_history]
            trend_slope = (scores[-1] - scores[0]) / len(scores)

            if trend_slope > 0.01:
                trend = "improving"
            elif trend_slope < -0.01:
                trend = "degrading"
            else:
                trend = "stable"

            previous_score = metrics_history[-2].overall_quality_score
            score_change = latest.overall_quality_score - previous_score
        else:
            trend = "stable"
            previous_score = None
            score_change = None

        # Generate recommendations
        recommendations = []

        if latest.completeness_score < 0.95:
            recommendations.append(
                "Improve data completeness by checking sensor connectivity"
            )

        if latest.timeliness_score < 0.9:
            recommendations.append(
                "Reduce data latency by optimizing data pipeline"
            )

        if latest.outliers_detected > 0:
            recommendations.append(
                f"Investigate {latest.outliers_detected} outliers detected"
            )

        if latest.validity_score < 0.95:
            recommendations.append(
                "Review sensor calibration for invalid readings"
            )

        # Count affected predictions
        predictions_affected = sum(
            1 for m in metrics_history
            if m.overall_quality_score < 0.8
        )

        predictions_blocked = sum(
            1 for m in metrics_history
            if m.overall_quality_score < 0.6
        )

        report = QualityReport(
            report_id=f"qr_{hashlib.sha256(f'{asset_id}{period_end}'.encode()).hexdigest()[:12]}",
            asset_id=asset_id,
            report_period_start=period_start,
            report_period_end=period_end,
            metrics=latest,
            quality_trend=trend,
            previous_quality_score=previous_score,
            score_change=score_change,
            recommendations=recommendations,
            predictions_affected=predictions_affected,
            predictions_blocked=predictions_blocked
        )

        report.provenance_hash = report.compute_provenance_hash()

        return report

    def is_prediction_ready(
        self,
        metrics: DataQualityMetrics,
        min_quality: float = 0.7
    ) -> Tuple[bool, str]:
        """
        Check if data quality is sufficient for prediction.

        Args:
            metrics: Quality metrics to check
            min_quality: Minimum required quality score

        Returns:
            Tuple of (is_ready, reason if not ready)
        """
        if metrics.overall_quality_score < min_quality:
            return False, f"Quality score {metrics.overall_quality_score:.2f} below threshold {min_quality}"

        if metrics.critical_issues > 0:
            return False, f"{metrics.critical_issues} critical issues detected"

        if metrics.completeness_score < 0.8:
            return False, f"Completeness {metrics.completeness_score:.2%} too low"

        if metrics.staleness_seconds > self.max_staleness_seconds:
            return False, f"Data too stale ({metrics.staleness_seconds:.0f}s)"

        return True, "Data quality sufficient"
