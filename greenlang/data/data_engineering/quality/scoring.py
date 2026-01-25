"""
Data Quality Scoring Framework
==============================

Comprehensive data quality scoring across 6 dimensions:
Completeness, Validity, Accuracy, Consistency, Uniqueness, Timeliness.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from datetime import datetime, date, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
import logging
import hashlib
import statistics

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QualityDimension(str, Enum):
    """Data quality dimensions per ISO 8000 / DAMA."""
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""
    dimension: QualityDimension
    score: float  # 0-100
    weight: float  # Weight in overall score
    records_assessed: int
    records_passed: int
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class QualityScore(BaseModel):
    """Overall quality score with breakdown."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    grade: str = Field(default="", description="Letter grade (A-F)")
    total_records: int = Field(default=0)
    records_passed: int = Field(default=0)
    records_failed: int = Field(default=0)
    critical_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    assessment_time: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)

    def to_grade(self) -> str:
        """Convert score to letter grade."""
        if self.overall_score >= 95:
            return "A+"
        elif self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 85:
            return "B+"
        elif self.overall_score >= 80:
            return "B"
        elif self.overall_score >= 75:
            return "C+"
        elif self.overall_score >= 70:
            return "C"
        elif self.overall_score >= 65:
            return "D+"
        elif self.overall_score >= 60:
            return "D"
        else:
            return "F"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class DataQualityScorer:
    """
    Comprehensive data quality scorer.

    Scores data across 6 dimensions:
    1. Completeness (25%): Are required fields populated?
    2. Validity (25%): Do values conform to expected formats?
    3. Accuracy (20%): Are values within expected ranges?
    4. Consistency (15%): Are related values logically consistent?
    5. Uniqueness (10%): Are records unique (no duplicates)?
    6. Timeliness (5%): Is data current and fresh?

    Supports:
    - Configurable dimension weights
    - Custom validation rules
    - Threshold-based alerting
    - Detailed issue reporting
    """

    # Default dimension weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        QualityDimension.COMPLETENESS: 0.25,
        QualityDimension.VALIDITY: 0.25,
        QualityDimension.ACCURACY: 0.20,
        QualityDimension.CONSISTENCY: 0.15,
        QualityDimension.UNIQUENESS: 0.10,
        QualityDimension.TIMELINESS: 0.05,
    }

    def __init__(
        self,
        weights: Optional[Dict[QualityDimension, float]] = None,
        required_fields: Optional[List[str]] = None,
        valid_value_rules: Optional[Dict[str, Callable]] = None,
        range_rules: Optional[Dict[str, tuple]] = None,
        consistency_rules: Optional[List[Callable]] = None,
        unique_fields: Optional[List[str]] = None,
        freshness_field: Optional[str] = None,
        freshness_threshold_days: int = 365,
    ):
        """
        Initialize quality scorer.

        Args:
            weights: Custom dimension weights
            required_fields: Fields that must be populated
            valid_value_rules: Field -> validation function
            range_rules: Field -> (min, max) tuple
            consistency_rules: Cross-field validation functions
            unique_fields: Fields that should be unique
            freshness_field: Field containing date for timeliness
            freshness_threshold_days: Max age in days for timeliness
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.required_fields = required_fields or ['factor_id', 'factor_value', 'factor_unit']
        self.valid_value_rules = valid_value_rules or {}
        self.range_rules = range_rules or {}
        self.consistency_rules = consistency_rules or []
        self.unique_fields = unique_fields or ['factor_hash']
        self.freshness_field = freshness_field or 'reference_year'
        self.freshness_threshold_days = freshness_threshold_days

        # Tracking for uniqueness
        self._seen_values: Dict[str, Set] = {}

    def score_dataset(self, records: List[Dict[str, Any]]) -> QualityScore:
        """
        Score a dataset on all quality dimensions.

        Args:
            records: List of data records

        Returns:
            QualityScore with overall and dimension breakdowns
        """
        if not records:
            return QualityScore(
                overall_score=0.0,
                grade="F",
                total_records=0,
                critical_issues=["No records to assess"],
            )

        # Reset tracking
        self._seen_values.clear()

        # Score each dimension
        dimension_results = {}

        dimension_results[QualityDimension.COMPLETENESS] = self._score_completeness(records)
        dimension_results[QualityDimension.VALIDITY] = self._score_validity(records)
        dimension_results[QualityDimension.ACCURACY] = self._score_accuracy(records)
        dimension_results[QualityDimension.CONSISTENCY] = self._score_consistency(records)
        dimension_results[QualityDimension.UNIQUENESS] = self._score_uniqueness(records)
        dimension_results[QualityDimension.TIMELINESS] = self._score_timeliness(records)

        # Calculate weighted overall score
        overall_score = 0.0
        dimension_scores = {}
        all_issues = []
        all_warnings = []

        for dimension, result in dimension_results.items():
            weighted_score = result.score * self.weights[dimension]
            overall_score += weighted_score
            dimension_scores[dimension.value] = round(result.score, 2)

            # Collect issues
            if result.score < 70:
                all_issues.extend(result.issues)
            else:
                all_warnings.extend(result.issues)

        # Count passed/failed records
        records_passed = sum(1 for r in records if self._record_passes(r, dimension_results))
        records_failed = len(records) - records_passed

        quality_score = QualityScore(
            overall_score=round(overall_score, 2),
            dimension_scores=dimension_scores,
            total_records=len(records),
            records_passed=records_passed,
            records_failed=records_failed,
            critical_issues=all_issues[:10],  # Top 10 issues
            warnings=all_warnings[:10],
            details={
                "dimension_details": {
                    d.value: {
                        "score": r.score,
                        "weight": self.weights[d],
                        "assessed": r.records_assessed,
                        "passed": r.records_passed,
                        "issues_count": len(r.issues),
                    }
                    for d, r in dimension_results.items()
                }
            },
        )

        quality_score.grade = quality_score.to_grade()

        return quality_score

    def _score_completeness(self, records: List[Dict[str, Any]]) -> DimensionScore:
        """
        Score completeness dimension.

        Checks if required fields are populated (non-null, non-empty).
        """
        total_checks = 0
        passed_checks = 0
        issues = []
        field_completeness = {}

        for record in records:
            for field in self.required_fields:
                total_checks += 1
                value = record.get(field)

                if value is not None and value != '':
                    passed_checks += 1
                    field_completeness[field] = field_completeness.get(field, 0) + 1
                else:
                    issues.append(f"Missing required field: {field}")

        # Calculate score
        score = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        # Report fields with low completeness
        for field in self.required_fields:
            completeness = (field_completeness.get(field, 0) / len(records) * 100) if records else 0
            if completeness < 95:
                issues.append(f"Field '{field}' completeness: {completeness:.1f}%")

        return DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            weight=self.weights[QualityDimension.COMPLETENESS],
            records_assessed=len(records),
            records_passed=passed_checks // len(self.required_fields) if self.required_fields else len(records),
            issues=issues[:10],
            details={"field_completeness": field_completeness},
        )

    def _score_validity(self, records: List[Dict[str, Any]]) -> DimensionScore:
        """
        Score validity dimension.

        Checks if values conform to expected formats/types.
        """
        total_checks = 0
        passed_checks = 0
        issues = []

        # Default validity rules
        default_rules = {
            'factor_value': lambda v: v is not None and float(v) >= 0,
            'reference_year': lambda v: v is not None and 1990 <= int(v) <= 2050,
            'country_code': lambda v: v is None or (isinstance(v, str) and len(v) in [2, 3]),
        }

        all_rules = {**default_rules, **self.valid_value_rules}

        for record in records:
            for field, validator in all_rules.items():
                if field in record:
                    total_checks += 1
                    try:
                        if validator(record.get(field)):
                            passed_checks += 1
                        else:
                            issues.append(f"Invalid value for {field}: {record.get(field)}")
                    except Exception as e:
                        issues.append(f"Validation error for {field}: {str(e)}")

        score = (passed_checks / total_checks * 100) if total_checks > 0 else 100

        return DimensionScore(
            dimension=QualityDimension.VALIDITY,
            score=score,
            weight=self.weights[QualityDimension.VALIDITY],
            records_assessed=len(records),
            records_passed=passed_checks,
            issues=issues[:10],
        )

    def _score_accuracy(self, records: List[Dict[str, Any]]) -> DimensionScore:
        """
        Score accuracy dimension.

        Checks if values fall within expected ranges.
        """
        total_checks = 0
        passed_checks = 0
        issues = []
        outliers = []

        # Default range rules for emission factors
        default_ranges = {
            'factor_value': (0, 1000),  # kgCO2e typically 0-1000
            'aggregate_dqi': (0, 100),
        }

        all_ranges = {**default_ranges, **self.range_rules}

        # Also check for statistical outliers
        numeric_values = {}

        for record in records:
            for field, (min_val, max_val) in all_ranges.items():
                if field in record and record.get(field) is not None:
                    total_checks += 1
                    try:
                        value = float(record.get(field))

                        # Collect for outlier detection
                        if field not in numeric_values:
                            numeric_values[field] = []
                        numeric_values[field].append(value)

                        if min_val <= value <= max_val:
                            passed_checks += 1
                        else:
                            issues.append(f"Out of range {field}: {value} (expected {min_val}-{max_val})")
                    except (ValueError, TypeError):
                        issues.append(f"Non-numeric value for {field}")

        # Detect statistical outliers (> 3 standard deviations)
        for field, values in numeric_values.items():
            if len(values) > 10:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values)
                if stdev > 0:
                    outlier_count = sum(1 for v in values if abs(v - mean) > 3 * stdev)
                    if outlier_count > 0:
                        issues.append(f"Field '{field}' has {outlier_count} statistical outliers")

        score = (passed_checks / total_checks * 100) if total_checks > 0 else 100

        return DimensionScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            weight=self.weights[QualityDimension.ACCURACY],
            records_assessed=len(records),
            records_passed=passed_checks,
            issues=issues[:10],
        )

    def _score_consistency(self, records: List[Dict[str, Any]]) -> DimensionScore:
        """
        Score consistency dimension.

        Checks cross-field logical consistency.
        """
        total_checks = 0
        passed_checks = 0
        issues = []

        # Default consistency rules
        default_rules = [
            # valid_from should be before valid_to
            lambda r: (
                r.get('valid_from') is None or
                r.get('valid_to') is None or
                str(r.get('valid_from')) <= str(r.get('valid_to'))
            ),
            # reference_year should match valid_from year
            lambda r: (
                r.get('reference_year') is None or
                r.get('valid_from') is None or
                str(r.get('valid_from')).startswith(str(r.get('reference_year')))
            ),
        ]

        all_rules = default_rules + self.consistency_rules

        for record in records:
            for rule in all_rules:
                total_checks += 1
                try:
                    if rule(record):
                        passed_checks += 1
                    else:
                        issues.append(f"Consistency rule failed for record {record.get('factor_id', 'unknown')}")
                except Exception:
                    # Rule evaluation error
                    pass

        score = (passed_checks / total_checks * 100) if total_checks > 0 else 100

        return DimensionScore(
            dimension=QualityDimension.CONSISTENCY,
            score=score,
            weight=self.weights[QualityDimension.CONSISTENCY],
            records_assessed=len(records),
            records_passed=passed_checks,
            issues=issues[:10],
        )

    def _score_uniqueness(self, records: List[Dict[str, Any]]) -> DimensionScore:
        """
        Score uniqueness dimension.

        Checks for duplicate records based on unique fields.
        """
        total_records = len(records)
        unique_records = 0
        duplicate_count = 0
        issues = []

        for field in self.unique_fields:
            if field not in self._seen_values:
                self._seen_values[field] = set()

        for record in records:
            is_unique = True
            for field in self.unique_fields:
                value = record.get(field)
                if value is not None:
                    if value in self._seen_values[field]:
                        is_unique = False
                        duplicate_count += 1
                        issues.append(f"Duplicate {field}: {value}")
                        break
                    self._seen_values[field].add(value)

            if is_unique:
                unique_records += 1

        score = (unique_records / total_records * 100) if total_records > 0 else 100

        if duplicate_count > 0:
            issues.insert(0, f"Found {duplicate_count} duplicate records")

        return DimensionScore(
            dimension=QualityDimension.UNIQUENESS,
            score=score,
            weight=self.weights[QualityDimension.UNIQUENESS],
            records_assessed=total_records,
            records_passed=unique_records,
            issues=issues[:10],
            details={"duplicates": duplicate_count},
        )

    def _score_timeliness(self, records: List[Dict[str, Any]]) -> DimensionScore:
        """
        Score timeliness dimension.

        Checks if data is current (within threshold).
        """
        total_records = len(records)
        current_records = 0
        issues = []
        current_year = datetime.now().year

        for record in records:
            freshness_value = record.get(self.freshness_field)

            if freshness_value is None:
                current_records += 1  # No date, assume current
                continue

            try:
                if isinstance(freshness_value, int):
                    # Year field
                    age_days = (current_year - freshness_value) * 365
                elif isinstance(freshness_value, str):
                    # Try to parse as date
                    if len(freshness_value) == 4:  # Year only
                        age_days = (current_year - int(freshness_value)) * 365
                    else:
                        record_date = datetime.fromisoformat(freshness_value.replace('Z', '+00:00'))
                        age_days = (datetime.now() - record_date.replace(tzinfo=None)).days
                elif isinstance(freshness_value, (date, datetime)):
                    if isinstance(freshness_value, datetime):
                        age_days = (datetime.now() - freshness_value.replace(tzinfo=None)).days
                    else:
                        age_days = (date.today() - freshness_value).days
                else:
                    current_records += 1
                    continue

                if age_days <= self.freshness_threshold_days:
                    current_records += 1
                else:
                    years_old = age_days // 365
                    issues.append(f"Outdated data ({years_old} years old): {record.get('factor_id', 'unknown')}")

            except Exception:
                current_records += 1  # Parsing error, assume current

        score = (current_records / total_records * 100) if total_records > 0 else 100

        outdated_count = total_records - current_records
        if outdated_count > 0:
            issues.insert(0, f"{outdated_count} records exceed {self.freshness_threshold_days} day freshness threshold")

        return DimensionScore(
            dimension=QualityDimension.TIMELINESS,
            score=score,
            weight=self.weights[QualityDimension.TIMELINESS],
            records_assessed=total_records,
            records_passed=current_records,
            issues=issues[:10],
            details={"outdated_records": outdated_count},
        )

    def _record_passes(self, record: Dict[str, Any], dimension_results: Dict) -> bool:
        """Check if a record passes quality checks."""
        # A record passes if it has required fields and valid values
        for field in self.required_fields:
            if record.get(field) is None or record.get(field) == '':
                return False

        # Check validity
        for field, validator in self.valid_value_rules.items():
            if field in record:
                try:
                    if not validator(record.get(field)):
                        return False
                except:
                    return False

        return True

    def score_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single record.

        Returns simplified quality assessment for one record.
        """
        issues = []
        score = 100.0
        penalty_per_issue = 10.0

        # Completeness check
        for field in self.required_fields:
            if record.get(field) is None or record.get(field) == '':
                issues.append(f"Missing: {field}")
                score -= penalty_per_issue

        # Validity check
        for field, validator in self.valid_value_rules.items():
            if field in record:
                try:
                    if not validator(record.get(field)):
                        issues.append(f"Invalid: {field}")
                        score -= penalty_per_issue
                except:
                    issues.append(f"Error validating: {field}")
                    score -= penalty_per_issue / 2

        # Range check
        for field, (min_val, max_val) in self.range_rules.items():
            if field in record and record.get(field) is not None:
                try:
                    value = float(record.get(field))
                    if not (min_val <= value <= max_val):
                        issues.append(f"Out of range: {field}")
                        score -= penalty_per_issue
                except:
                    pass

        score = max(0, score)

        return {
            "score": round(score, 2),
            "grade": self._score_to_grade(score),
            "issues": issues,
            "passed": len(issues) == 0,
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 65:
            return "D+"
        elif score >= 60:
            return "D"
        else:
            return "F"


# Pre-configured scorers for common use cases
def create_emission_factor_scorer() -> DataQualityScorer:
    """Create scorer configured for emission factor data."""
    return DataQualityScorer(
        required_fields=[
            'factor_id', 'factor_value', 'factor_unit', 'industry',
            'region', 'scope_type', 'reference_year',
        ],
        valid_value_rules={
            'factor_value': lambda v: v is not None and float(v) >= 0,
            'reference_year': lambda v: v is not None and 1990 <= int(v) <= 2050,
            'scope_type': lambda v: v in ['scope_1', 'scope_2_location', 'scope_2_market', 'scope_3', 'well_to_tank'],
        },
        range_rules={
            'factor_value': (0, 10000),
            'aggregate_dqi': (0, 100),
        },
        unique_fields=['factor_hash'],
        freshness_field='reference_year',
        freshness_threshold_days=365 * 5,  # 5 years
    )


def create_cbam_scorer() -> DataQualityScorer:
    """Create scorer configured for CBAM data."""
    return DataQualityScorer(
        required_fields=[
            'factor_id', 'factor_value', 'factor_unit', 'product_code',
            'country_code', 'production_route',
        ],
        valid_value_rules={
            'product_code': lambda v: v is not None and len(str(v)) == 8,
            'country_code': lambda v: v is not None and len(str(v)) in [2, 3],
        },
        range_rules={
            'factor_value': (0, 50),  # CBAM factors typically lower range
        },
        freshness_threshold_days=365,  # Must be current year
    )
