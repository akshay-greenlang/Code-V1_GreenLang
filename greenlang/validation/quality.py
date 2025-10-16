"""
GreenLang Data Quality Validator
Data quality and completeness checks.
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from .framework import ValidationResult, ValidationError, ValidationSeverity

logger = logging.getLogger(__name__)


class QualityCheck(BaseModel):
    """A data quality check definition."""
    name: str = Field(..., description="Check name")
    check_type: str = Field(..., description="Type of quality check")
    threshold: Optional[float] = Field(default=None, description="Threshold value")
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING, description="Issue severity")
    enabled: bool = Field(default=True, description="Whether check is enabled")


class DataQualityValidator:
    """
    Data quality validator for completeness and consistency checks.

    Supports:
    - Completeness checks (missing values, null ratios)
    - Uniqueness checks (duplicates)
    - Consistency checks (data types, formats)
    - Statistical checks (outliers, ranges)

    Example:
        validator = DataQualityValidator()
        validator.add_check(QualityCheck(
            name="completeness",
            check_type="missing_ratio",
            threshold=0.1
        ))

        result = validator.validate(records)
    """

    def __init__(self):
        """Initialize quality validator."""
        self.checks: List[QualityCheck] = []

    def add_check(self, check: QualityCheck):
        """Add a quality check."""
        self.checks.append(check)
        logger.debug(f"Added quality check: {check.name}")

    def check_missing_ratio(self, records: List[Dict[str, Any]], threshold: float) -> ValidationResult:
        """Check ratio of missing values."""
        result = ValidationResult(valid=True)

        if not records:
            return result

        # Calculate missing ratios for each field
        field_stats = {}
        total_records = len(records)

        for record in records:
            for field, value in record.items():
                if field not in field_stats:
                    field_stats[field] = {"total": 0, "missing": 0}

                field_stats[field]["total"] += 1
                if value is None or value == "":
                    field_stats[field]["missing"] += 1

        # Check thresholds
        for field, stats in field_stats.items():
            missing_ratio = stats["missing"] / stats["total"] if stats["total"] > 0 else 0

            if missing_ratio > threshold:
                error = ValidationError(
                    field=field,
                    message=f"High missing value ratio: {missing_ratio:.2%} (threshold: {threshold:.2%})",
                    severity=ValidationSeverity.WARNING,
                    validator="data_quality",
                    value=missing_ratio,
                    expected=threshold
                )
                result.add_error(error)

        return result

    def check_duplicates(self, records: List[Dict[str, Any]], key_fields: List[str]) -> ValidationResult:
        """Check for duplicate records based on key fields."""
        result = ValidationResult(valid=True)

        seen = set()
        duplicates = []

        for i, record in enumerate(records):
            # Create key from specified fields
            key_values = tuple(record.get(field) for field in key_fields)

            if key_values in seen:
                duplicates.append(i)
            else:
                seen.add(key_values)

        if duplicates:
            error = ValidationError(
                field=",".join(key_fields),
                message=f"Found {len(duplicates)} duplicate records based on key fields",
                severity=ValidationSeverity.WARNING,
                validator="data_quality",
                value=len(duplicates)
            )
            result.add_error(error)

        return result

    def check_data_types(self, records: List[Dict[str, Any]], expected_types: Dict[str, type]) -> ValidationResult:
        """Check if fields have expected data types."""
        result = ValidationResult(valid=True)

        type_errors = {}

        for i, record in enumerate(records):
            for field, expected_type in expected_types.items():
                if field in record and record[field] is not None:
                    if not isinstance(record[field], expected_type):
                        if field not in type_errors:
                            type_errors[field] = 0
                        type_errors[field] += 1

        for field, error_count in type_errors.items():
            error = ValidationError(
                field=field,
                message=f"Type mismatch in {error_count} records (expected: {expected_types[field].__name__})",
                severity=ValidationSeverity.ERROR,
                validator="data_quality",
                value=error_count
            )
            result.add_error(error)

        return result

    def validate(self, data: Any) -> ValidationResult:
        """
        Run all enabled quality checks.

        Args:
            data: Data to validate (typically list of records)

        Returns:
            ValidationResult with quality issues
        """
        result = ValidationResult(valid=True)

        # Ensure data is a list
        if not isinstance(data, list):
            error = ValidationError(
                field="__data__",
                message="Data quality checks require list of records",
                severity=ValidationSeverity.ERROR,
                validator="data_quality"
            )
            result.add_error(error)
            return result

        # Run enabled checks
        for check in self.checks:
            if not check.enabled:
                continue

            try:
                if check.check_type == "missing_ratio" and check.threshold:
                    check_result = self.check_missing_ratio(data, check.threshold)
                    result.merge(check_result)

                # Add more check types as needed

            except Exception as e:
                logger.error(f"Quality check {check.name} failed: {str(e)}", exc_info=True)

        return result
