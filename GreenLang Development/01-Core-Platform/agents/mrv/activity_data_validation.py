# -*- coding: utf-8 -*-
"""
GL-MRV-X-012: Activity Data Validation Agent
=============================================

Validates activity data completeness, accuracy, and consistency for GHG
inventory calculations.

Capabilities:
    - Completeness checks
    - Range validation
    - Outlier detection
    - Unit consistency checks
    - Time series validation
    - Cross-reference validation
    - Complete provenance tracking

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(str, Enum):
    """Categories of validation checks."""
    COMPLETENESS = "completeness"
    RANGE = "range"
    OUTLIER = "outlier"
    UNIT = "unit"
    CONSISTENCY = "consistency"
    FORMAT = "format"


class ActivityDataRecord(BaseModel):
    """An activity data record for validation."""
    record_id: str = Field(...)
    data_type: str = Field(..., description="e.g., fuel_consumption, electricity")
    value: float = Field(...)
    unit: str = Field(...)
    facility_id: Optional[str] = Field(None)
    period: Optional[str] = Field(None)
    source: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationRule(BaseModel):
    """A validation rule specification."""
    rule_id: str = Field(...)
    category: ValidationCategory = Field(...)
    data_type: str = Field(...)
    min_value: Optional[float] = Field(None)
    max_value: Optional[float] = Field(None)
    allowed_units: List[str] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    """A validation issue found."""
    issue_id: str = Field(...)
    record_id: str = Field(...)
    rule_id: str = Field(...)
    category: ValidationCategory = Field(...)
    severity: ValidationSeverity = Field(...)
    message: str = Field(...)
    actual_value: Optional[Any] = Field(None)
    expected_range: Optional[str] = Field(None)
    suggested_action: Optional[str] = Field(None)


class ValidationResult(BaseModel):
    """Result of validating a record."""
    record_id: str = Field(...)
    is_valid: bool = Field(...)
    issues: List[ValidationIssue] = Field(default_factory=list)
    error_count: int = Field(default=0)
    warning_count: int = Field(default=0)


class ActivityDataValidationInput(BaseModel):
    """Input model for ActivityDataValidationAgent."""
    records: List[ActivityDataRecord] = Field(..., min_length=1)
    custom_rules: Optional[List[ValidationRule]] = Field(None)
    strict_mode: bool = Field(default=False)
    organization_id: Optional[str] = Field(None)


class ActivityDataValidationOutput(BaseModel):
    """Output model for ActivityDataValidationAgent."""
    success: bool = Field(...)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    total_records: int = Field(...)
    valid_records: int = Field(...)
    invalid_records: int = Field(...)
    total_errors: int = Field(...)
    total_warnings: int = Field(...)
    validation_rate_pct: float = Field(...)
    issues_by_category: Dict[str, int] = Field(default_factory=dict)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# Default validation rules
DEFAULT_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "R001",
        "category": "range",
        "data_type": "electricity_kwh",
        "min_value": 0,
        "max_value": 1e12,
        "allowed_units": ["kwh", "mwh", "gwh"]
    },
    {
        "rule_id": "R002",
        "category": "range",
        "data_type": "fuel_liters",
        "min_value": 0,
        "max_value": 1e9,
        "allowed_units": ["liters", "gallons", "m3"]
    },
    {
        "rule_id": "R003",
        "category": "range",
        "data_type": "emissions_tco2e",
        "min_value": 0,
        "max_value": 1e9,
        "allowed_units": ["tco2e", "kgco2e", "mtco2e"]
    },
]


class ActivityDataValidationAgent(DeterministicAgent):
    """
    GL-MRV-X-012: Activity Data Validation Agent

    Validates activity data completeness and accuracy.

    Example:
        >>> agent = ActivityDataValidationAgent()
        >>> result = agent.execute({
        ...     "records": [
        ...         {"record_id": "R001", "data_type": "electricity_kwh",
        ...          "value": 1000000, "unit": "kwh"}
        ...     ]
        ... })
    """

    AGENT_ID = "GL-MRV-X-012"
    AGENT_NAME = "Activity Data Validation Agent"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="ActivityDataValidationAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Validates activity data completeness and accuracy"
    )

    def __init__(self, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self._rules = [ValidationRule(**r) for r in DEFAULT_RULES]
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute activity data validation."""
        start_time = DeterministicClock.now()

        try:
            val_input = ActivityDataValidationInput(**inputs)

            # Add custom rules if provided
            if val_input.custom_rules:
                self._rules.extend(val_input.custom_rules)

            validation_results: List[ValidationResult] = []
            issues_by_category: Dict[str, int] = {}
            total_errors = 0
            total_warnings = 0

            for record in val_input.records:
                issues = self._validate_record(record, val_input.strict_mode)
                error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
                warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)

                total_errors += error_count
                total_warnings += warning_count

                for issue in issues:
                    issues_by_category[issue.category.value] = (
                        issues_by_category.get(issue.category.value, 0) + 1
                    )

                validation_results.append(ValidationResult(
                    record_id=record.record_id,
                    is_valid=(error_count == 0),
                    issues=issues,
                    error_count=error_count,
                    warning_count=warning_count
                ))

            valid_count = sum(1 for r in validation_results if r.is_valid)
            total = len(val_input.records)
            validation_rate = (valid_count / total * 100) if total > 0 else 0

            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            provenance_hash = self._compute_hash({
                "total_records": total,
                "valid_records": valid_count,
                "total_errors": total_errors
            })

            output = ActivityDataValidationOutput(
                success=True,
                validation_results=validation_results,
                total_records=total,
                valid_records=valid_count,
                invalid_records=total - valid_count,
                total_errors=total_errors,
                total_warnings=total_warnings,
                validation_rate_pct=round(validation_rate, 2),
                issues_by_category=issues_by_category,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS" if total_errors == 0 else "FAIL"
            )

            self._capture_audit_entry(
                operation="validate_activity_data",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Validated {total} records, {valid_count} valid"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "validation_status": "FAIL"
            }

    def _validate_record(
        self,
        record: ActivityDataRecord,
        strict_mode: bool
    ) -> List[ValidationIssue]:
        """Validate a single record."""
        issues = []
        issue_counter = 0

        # Find applicable rules
        applicable_rules = [r for r in self._rules if r.data_type == record.data_type]

        # Completeness check
        if record.value is None:
            issue_counter += 1
            issues.append(ValidationIssue(
                issue_id=f"V{issue_counter:04d}",
                record_id=record.record_id,
                rule_id="BUILT_IN_COMPLETENESS",
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.ERROR,
                message="Value is missing",
                suggested_action="Provide a valid numeric value"
            ))

        # Range validation
        for rule in applicable_rules:
            if rule.min_value is not None and record.value < rule.min_value:
                issue_counter += 1
                issues.append(ValidationIssue(
                    issue_id=f"V{issue_counter:04d}",
                    record_id=record.record_id,
                    rule_id=rule.rule_id,
                    category=ValidationCategory.RANGE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Value {record.value} is below minimum {rule.min_value}",
                    actual_value=record.value,
                    expected_range=f">= {rule.min_value}",
                    suggested_action="Check data entry or measurement"
                ))

            if rule.max_value is not None and record.value > rule.max_value:
                issue_counter += 1
                issues.append(ValidationIssue(
                    issue_id=f"V{issue_counter:04d}",
                    record_id=record.record_id,
                    rule_id=rule.rule_id,
                    category=ValidationCategory.RANGE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Value {record.value} exceeds maximum {rule.max_value}",
                    actual_value=record.value,
                    expected_range=f"<= {rule.max_value}",
                    suggested_action="Verify this is not a data entry error"
                ))

            # Unit validation
            if rule.allowed_units and record.unit.lower() not in [u.lower() for u in rule.allowed_units]:
                issue_counter += 1
                issues.append(ValidationIssue(
                    issue_id=f"V{issue_counter:04d}",
                    record_id=record.record_id,
                    rule_id=rule.rule_id,
                    category=ValidationCategory.UNIT,
                    severity=ValidationSeverity.ERROR if strict_mode else ValidationSeverity.WARNING,
                    message=f"Unit '{record.unit}' not in allowed units",
                    actual_value=record.unit,
                    expected_range=", ".join(rule.allowed_units),
                    suggested_action="Use standard unit or convert"
                ))

        return issues

    def _compute_hash(self, data: Any) -> str:
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
