# -*- coding: utf-8 -*-
"""
Live data validator for supplier portal.

Provides real-time validation feedback during data entry.
"""
import logging
from typing import Dict, List, Any
import re

from ..models import ValidationResult


logger = logging.getLogger(__name__)


class LiveValidator:
    """
    Real-time data validation for supplier submissions.

    Features:
    - Field-level validation
    - Data quality scoring
    - Completeness checking
    - Format validation
    """

    def __init__(self):
        """Initialize live validator."""
        self.required_fields = [
            "supplier_id", "product_id", "emission_factor", "unit"
        ]
        self.optional_fields = [
            "activity_data", "uncertainty", "data_quality", "source"
        ]
        logger.info("LiveValidator initialized")

    def validate_record(self, record: Dict[str, Any]) -> ValidationResult:
        """
        Validate individual data record.

        Args:
            record: Data record to validate

        Returns:
            Validation result with errors, warnings, and DQI score
        """
        errors = []
        warnings = []
        field_validations = {}

        # Check required fields
        for field in self.required_fields:
            if field not in record or not record[field]:
                errors.append(f"Missing required field: {field}")
                field_validations[field] = False
            else:
                field_validations[field] = True

        # Validate field formats
        if "emission_factor" in record and record["emission_factor"]:
            if not self._validate_numeric(record["emission_factor"]):
                errors.append("emission_factor must be numeric")
                field_validations["emission_factor"] = False

        if "unit" in record and record["unit"]:
            valid_units = ["kg", "tonnes", "t", "mt", "kg CO2e", "t CO2e"]
            if record["unit"] not in valid_units:
                warnings.append(f"Unit '{record['unit']}' not in standard list: {valid_units}")

        if "uncertainty" in record and record["uncertainty"]:
            if not self._validate_percentage(record["uncertainty"]):
                warnings.append("uncertainty should be a percentage (0-100)")

        # Calculate data quality score
        dqi_score = self._calculate_dqi(record, errors, warnings)

        # Calculate completeness
        total_fields = len(self.required_fields) + len(self.optional_fields)
        filled_fields = sum(1 for f in self.required_fields + self.optional_fields if record.get(f))
        completeness = (filled_fields / total_fields) * 100

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            data_quality_score=dqi_score,
            completeness_percentage=completeness,
            field_validations=field_validations
        )

    def validate_batch(
        self,
        records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate batch of records.

        Args:
            records: List of data records

        Returns:
            Batch validation results
        """
        results = []
        total_errors = 0
        total_warnings = 0

        for i, record in enumerate(records):
            result = self.validate_record(record)
            results.append({
                "record_index": i,
                "validation": result.model_dump()
            })
            total_errors += len(result.errors)
            total_warnings += len(result.warnings)

        # Calculate batch metrics
        valid_count = sum(1 for r in results if r["validation"]["is_valid"])
        avg_dqi = sum(r["validation"]["data_quality_score"] for r in results) / len(results)
        avg_completeness = sum(r["validation"]["completeness_percentage"] for r in results) / len(results)

        return {
            "total_records": len(records),
            "valid_records": valid_count,
            "invalid_records": len(records) - valid_count,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "average_dqi": avg_dqi,
            "average_completeness": avg_completeness,
            "results": results
        }

    def _validate_numeric(self, value: Any) -> bool:
        """Validate that value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _validate_percentage(self, value: Any) -> bool:
        """Validate that value is a valid percentage."""
        try:
            num = float(value)
            return 0 <= num <= 100
        except (ValueError, TypeError):
            return False

    def _calculate_dqi(
        self,
        record: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ) -> float:
        """
        Calculate data quality index (0-1).

        Args:
            record: Data record
            errors: Validation errors
            warnings: Validation warnings

        Returns:
            DQI score (0.0-1.0)
        """
        # Start with perfect score
        score = 1.0

        # Deduct for errors (0.2 per error)
        score -= len(errors) * 0.2

        # Deduct for warnings (0.05 per warning)
        score -= len(warnings) * 0.05

        # Bonus for optional fields
        optional_filled = sum(
            1 for field in self.optional_fields
            if record.get(field)
        )
        score += optional_filled * 0.05

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
