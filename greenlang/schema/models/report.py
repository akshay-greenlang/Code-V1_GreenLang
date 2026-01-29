# -*- coding: utf-8 -*-
"""
Validation Report Models
========================

Pydantic models for validation reports and summaries.

This module provides:
- ValidationSummary: Counts of errors, warnings, and info findings
- TimingInfo: Performance timing for each validation phase
- ValidationReport: Complete report for a single validation
- BatchSummary: Summary for batch validation
- ItemResult: Result for a single item in batch validation
- BatchValidationReport: Complete report for batch validation

Example:
    >>> report = ValidationReport(
    ...     valid=False,
    ...     schema_ref=SchemaRef(schema_id="test", version="1.0.0"),
    ...     schema_hash="abc123...",
    ...     summary=ValidationSummary(valid=False, error_count=2, warning_count=1),
    ...     findings=[...],
    ...     timings=TimingInfo(validate_ms=15.5, total_ms=25.0)
    ... )
    >>> print(report.valid)
    False

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.models.finding import Finding
from greenlang.schema.models.patch import FixSuggestion


class ValidationSummary(BaseModel):
    """
    Summary of validation findings.

    Provides a quick overview of validation results without
    the full details of each finding.

    Attributes:
        valid: Whether the payload passed validation (no errors).
        error_count: Number of error-level findings.
        warning_count: Number of warning-level findings.
        info_count: Number of info-level findings.

    Example:
        >>> summary = ValidationSummary(
        ...     valid=False,
        ...     error_count=3,
        ...     warning_count=2,
        ...     info_count=1
        ... )
        >>> print(summary.total_findings())
        6
    """

    valid: bool = Field(
        ...,
        description="Whether validation passed (no errors)"
    )

    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of error-level findings"
    )

    warning_count: int = Field(
        default=0,
        ge=0,
        description="Number of warning-level findings"
    )

    info_count: int = Field(
        default=0,
        ge=0,
        description="Number of info-level findings"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "valid": True,
                    "error_count": 0,
                    "warning_count": 1,
                    "info_count": 0
                },
                {
                    "valid": False,
                    "error_count": 3,
                    "warning_count": 2,
                    "info_count": 1
                }
            ]
        }
    }

    @model_validator(mode="after")
    def validate_consistency(self) -> "ValidationSummary":
        """
        Ensure valid flag is consistent with error_count.

        Returns:
            The validated summary.

        Raises:
            ValueError: If valid=True but error_count > 0.
        """
        if self.valid and self.error_count > 0:
            raise ValueError(
                f"Inconsistent summary: valid=True but error_count={self.error_count}"
            )
        if not self.valid and self.error_count == 0:
            raise ValueError(
                "Inconsistent summary: valid=False but error_count=0"
            )
        return self

    def total_findings(self) -> int:
        """
        Get total count of all findings.

        Returns:
            Sum of error, warning, and info counts.
        """
        return self.error_count + self.warning_count + self.info_count

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return self.warning_count > 0

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return self.error_count > 0

    @classmethod
    def from_findings(cls, findings: List[Finding]) -> "ValidationSummary":
        """
        Create summary from a list of findings.

        Args:
            findings: List of validation findings.

        Returns:
            ValidationSummary computed from findings.

        Example:
            >>> findings = [
            ...     Finding(code="GLSCHEMA-E100", severity=Severity.ERROR, ...),
            ...     Finding(code="GLSCHEMA-W700", severity=Severity.WARNING, ...)
            ... ]
            >>> summary = ValidationSummary.from_findings(findings)
            >>> print(summary.error_count)
            1
        """
        error_count = sum(1 for f in findings if f.is_error())
        warning_count = sum(1 for f in findings if f.is_warning())
        info_count = sum(1 for f in findings if f.is_info())

        return cls(
            valid=(error_count == 0),
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
        )


class TimingInfo(BaseModel):
    """
    Performance timing information for validation phases.

    Records the time spent in each phase of validation for
    performance monitoring and optimization.

    Attributes:
        parse_ms: Time to parse the payload (YAML/JSON).
        compile_ms: Time to compile the schema (if not cached).
        validate_ms: Time for validation checks.
        normalize_ms: Time for normalization (if enabled).
        suggest_ms: Time to generate fix suggestions (if enabled).
        total_ms: Total wall-clock time.

    Example:
        >>> timing = TimingInfo(
        ...     parse_ms=2.5,
        ...     validate_ms=15.3,
        ...     total_ms=20.0
        ... )
    """

    parse_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time to parse payload in milliseconds"
    )

    compile_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time to compile schema in milliseconds"
    )

    validate_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time for validation in milliseconds"
    )

    normalize_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time for normalization in milliseconds"
    )

    suggest_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time to generate suggestions in milliseconds"
    )

    total_ms: float = Field(
        ...,
        ge=0,
        description="Total wall-clock time in milliseconds"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "parse_ms": 2.5,
                    "compile_ms": 0.0,
                    "validate_ms": 15.3,
                    "normalize_ms": 3.2,
                    "suggest_ms": 1.5,
                    "total_ms": 22.5
                }
            ]
        }
    }

    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary (excluding None values).

        Returns:
            Dictionary of timing values.
        """
        result = {"total_ms": self.total_ms}
        if self.parse_ms is not None:
            result["parse_ms"] = self.parse_ms
        if self.compile_ms is not None:
            result["compile_ms"] = self.compile_ms
        if self.validate_ms is not None:
            result["validate_ms"] = self.validate_ms
        if self.normalize_ms is not None:
            result["normalize_ms"] = self.normalize_ms
        if self.suggest_ms is not None:
            result["suggest_ms"] = self.suggest_ms
        return result

    def overhead_ms(self) -> float:
        """
        Calculate overhead time (total minus tracked phases).

        Returns:
            Overhead time in milliseconds.
        """
        tracked = sum(
            t for t in [
                self.parse_ms,
                self.compile_ms,
                self.validate_ms,
                self.normalize_ms,
                self.suggest_ms,
            ]
            if t is not None
        )
        return max(0.0, self.total_ms - tracked)


class ValidationReport(BaseModel):
    """
    Complete validation report for a single payload.

    Contains all information about a validation run including
    findings, normalized payload, fix suggestions, and timing.

    Attributes:
        valid: Whether the payload passed validation.
        schema_ref: Reference to the schema used.
        schema_hash: SHA-256 hash of the compiled schema.
        summary: Summary counts of findings.
        findings: List of all validation findings.
        normalized_payload: Normalized payload (if normalization enabled).
        fix_suggestions: List of fix suggestions (if enabled).
        timings: Performance timing information.

    Example:
        >>> report = ValidationReport(
        ...     valid=False,
        ...     schema_ref=SchemaRef(schema_id="test", version="1.0.0"),
        ...     schema_hash="abc123...",
        ...     summary=ValidationSummary(valid=False, error_count=1),
        ...     findings=[Finding(...)],
        ...     timings=TimingInfo(total_ms=25.0)
        ... )
    """

    valid: bool = Field(
        ...,
        description="Whether validation passed (no errors)"
    )

    schema_ref: SchemaRef = Field(
        ...,
        description="Reference to the schema used for validation"
    )

    schema_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the compiled schema"
    )

    summary: ValidationSummary = Field(
        ...,
        description="Summary of validation findings"
    )

    findings: List[Finding] = Field(
        default_factory=list,
        description="List of validation findings"
    )

    normalized_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Normalized payload (if normalization enabled)"
    )

    fix_suggestions: Optional[List[FixSuggestion]] = Field(
        default=None,
        description="List of fix suggestions (if enabled)"
    )

    timings: TimingInfo = Field(
        ...,
        description="Performance timing information"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "valid": True,
                    "schema_ref": {
                        "schema_id": "emissions/activity",
                        "version": "1.3.0"
                    },
                    "schema_hash": "a" * 64,
                    "summary": {
                        "valid": True,
                        "error_count": 0,
                        "warning_count": 0,
                        "info_count": 0
                    },
                    "findings": [],
                    "normalized_payload": {"energy": {"value": 100, "unit": "kWh"}},
                    "timings": {"total_ms": 15.5}
                }
            ]
        }
    }

    @field_validator("schema_hash")
    @classmethod
    def validate_schema_hash(cls, v: str) -> str:
        """
        Validate schema_hash is a valid SHA-256 hex string.

        Args:
            v: The hash string.

        Returns:
            The validated hash string (lowercase).

        Raises:
            ValueError: If not a valid SHA-256 hex string.
        """
        import re
        if not re.match(r"^[a-fA-F0-9]{64}$", v):
            raise ValueError(
                f"Invalid schema_hash '{v}'. Must be a 64-character hex string (SHA-256)."
            )
        return v.lower()

    @model_validator(mode="after")
    def validate_consistency(self) -> "ValidationReport":
        """
        Ensure report is internally consistent.

        Returns:
            The validated report.

        Raises:
            ValueError: If valid flag doesn't match summary or findings.
        """
        # Check valid flag matches summary
        if self.valid != self.summary.valid:
            raise ValueError(
                f"Inconsistent report: valid={self.valid} but summary.valid={self.summary.valid}"
            )

        # Check findings count matches summary
        error_count = sum(1 for f in self.findings if f.is_error())
        if error_count != self.summary.error_count:
            raise ValueError(
                f"Inconsistent report: {error_count} errors in findings but "
                f"summary.error_count={self.summary.error_count}"
            )

        return self

    def errors_only(self) -> List[Finding]:
        """
        Get only error-level findings.

        Returns:
            List of error findings.
        """
        return [f for f in self.findings if f.is_error()]

    def warnings_only(self) -> List[Finding]:
        """
        Get only warning-level findings.

        Returns:
            List of warning findings.
        """
        return [f for f in self.findings if f.is_warning()]

    def findings_by_path(self, path: str) -> List[Finding]:
        """
        Get findings at a specific path.

        Args:
            path: JSON Pointer path.

        Returns:
            List of findings at that path.
        """
        return [f for f in self.findings if f.path == path]

    def findings_by_code(self, code: str) -> List[Finding]:
        """
        Get findings with a specific error code.

        Args:
            code: Error code (e.g., "GLSCHEMA-E100").

        Returns:
            List of findings with that code.
        """
        return [f for f in self.findings if f.code == code]

    def has_safe_fixes(self) -> bool:
        """
        Check if there are any safe fix suggestions.

        Returns:
            True if there are safe fix suggestions available.
        """
        if not self.fix_suggestions:
            return False
        return any(s.is_safe() for s in self.fix_suggestions)

    def format_summary(self) -> str:
        """
        Format a human-readable summary.

        Returns:
            Summary string.
        """
        status = "VALID" if self.valid else "INVALID"
        counts = []
        if self.summary.error_count:
            counts.append(f"{self.summary.error_count} error(s)")
        if self.summary.warning_count:
            counts.append(f"{self.summary.warning_count} warning(s)")
        if self.summary.info_count:
            counts.append(f"{self.summary.info_count} info(s)")

        count_str = ", ".join(counts) if counts else "no issues"
        return f"{status}: {self.schema_ref} ({count_str}) [{self.timings.total_ms:.1f}ms]"


class BatchSummary(BaseModel):
    """
    Summary for batch validation results.

    Provides aggregate statistics for validating multiple payloads.

    Attributes:
        total_items: Total number of items in the batch.
        valid_count: Number of items that passed validation.
        error_count: Number of items with at least one error.
        warning_count: Number of items with warnings (but no errors).

    Example:
        >>> summary = BatchSummary(
        ...     total_items=100,
        ...     valid_count=95,
        ...     error_count=5,
        ...     warning_count=10
        ... )
    """

    total_items: int = Field(
        ...,
        ge=0,
        description="Total number of items in the batch"
    )

    valid_count: int = Field(
        default=0,
        ge=0,
        description="Number of items that passed validation"
    )

    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of items with errors"
    )

    warning_count: int = Field(
        default=0,
        ge=0,
        description="Number of items with warnings"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def validate_counts(self) -> "BatchSummary":
        """
        Ensure counts are consistent.

        Returns:
            The validated summary.

        Raises:
            ValueError: If counts don't add up.
        """
        if self.valid_count + self.error_count > self.total_items:
            raise ValueError(
                f"Invalid counts: valid_count ({self.valid_count}) + "
                f"error_count ({self.error_count}) > total_items ({self.total_items})"
            )
        return self

    def success_rate(self) -> float:
        """
        Calculate the success rate as a percentage.

        Returns:
            Success rate (0.0 to 100.0).
        """
        if self.total_items == 0:
            return 100.0
        return (self.valid_count / self.total_items) * 100.0


class ItemResult(BaseModel):
    """
    Validation result for a single item in batch validation.

    Contains the validation outcome for one payload in a batch,
    including its index and optional identifier.

    Attributes:
        index: Zero-based index in the batch.
        id: Optional identifier for the item.
        valid: Whether this item passed validation.
        findings: List of findings for this item.
        normalized_payload: Normalized payload (if normalization enabled).
        fix_suggestions: Fix suggestions for this item (if enabled).

    Example:
        >>> result = ItemResult(
        ...     index=5,
        ...     id="record-123",
        ...     valid=False,
        ...     findings=[Finding(...)]
        ... )
    """

    index: int = Field(
        ...,
        ge=0,
        description="Zero-based index in the batch"
    )

    id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional identifier for the item"
    )

    valid: bool = Field(
        ...,
        description="Whether this item passed validation"
    )

    findings: List[Finding] = Field(
        default_factory=list,
        description="List of findings for this item"
    )

    normalized_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Normalized payload (if normalization enabled)"
    )

    fix_suggestions: Optional[List[FixSuggestion]] = Field(
        default=None,
        description="Fix suggestions for this item (if enabled)"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    def error_count(self) -> int:
        """Get count of errors for this item."""
        return sum(1 for f in self.findings if f.is_error())

    def warning_count(self) -> int:
        """Get count of warnings for this item."""
        return sum(1 for f in self.findings if f.is_warning())


class BatchValidationReport(BaseModel):
    """
    Complete validation report for batch validation.

    Contains results for validating multiple payloads against
    the same schema in a single batch operation.

    Attributes:
        schema_ref: Reference to the schema used.
        schema_hash: SHA-256 hash of the compiled schema.
        summary: Aggregate summary of all items.
        results: Individual results for each item.

    Example:
        >>> batch_report = BatchValidationReport(
        ...     schema_ref=SchemaRef(schema_id="test", version="1.0.0"),
        ...     schema_hash="abc123...",
        ...     summary=BatchSummary(total_items=10, valid_count=8, error_count=2),
        ...     results=[ItemResult(...), ...]
        ... )
    """

    schema_ref: SchemaRef = Field(
        ...,
        description="Reference to the schema used for validation"
    )

    schema_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the compiled schema"
    )

    summary: BatchSummary = Field(
        ...,
        description="Aggregate summary of all items"
    )

    results: List[ItemResult] = Field(
        default_factory=list,
        description="Individual results for each item"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    @field_validator("schema_hash")
    @classmethod
    def validate_schema_hash(cls, v: str) -> str:
        """Validate schema_hash is a valid SHA-256 hex string."""
        import re
        if not re.match(r"^[a-fA-F0-9]{64}$", v):
            raise ValueError(
                f"Invalid schema_hash '{v}'. Must be a 64-character hex string (SHA-256)."
            )
        return v.lower()

    @model_validator(mode="after")
    def validate_consistency(self) -> "BatchValidationReport":
        """
        Ensure report is internally consistent.

        Returns:
            The validated report.

        Raises:
            ValueError: If counts don't match results.
        """
        if len(self.results) != self.summary.total_items:
            raise ValueError(
                f"Inconsistent report: {len(self.results)} results but "
                f"summary.total_items={self.summary.total_items}"
            )

        valid_count = sum(1 for r in self.results if r.valid)
        if valid_count != self.summary.valid_count:
            raise ValueError(
                f"Inconsistent report: {valid_count} valid results but "
                f"summary.valid_count={self.summary.valid_count}"
            )

        return self

    def failed_items(self) -> List[ItemResult]:
        """
        Get only the items that failed validation.

        Returns:
            List of failed item results.
        """
        return [r for r in self.results if not r.valid]

    def valid_items(self) -> List[ItemResult]:
        """
        Get only the items that passed validation.

        Returns:
            List of valid item results.
        """
        return [r for r in self.results if r.valid]

    def items_with_warnings(self) -> List[ItemResult]:
        """
        Get items that have warnings.

        Returns:
            List of items with warnings.
        """
        return [r for r in self.results if r.warning_count() > 0]

    def total_findings(self) -> int:
        """
        Get total count of all findings across all items.

        Returns:
            Total finding count.
        """
        return sum(len(r.findings) for r in self.results)
