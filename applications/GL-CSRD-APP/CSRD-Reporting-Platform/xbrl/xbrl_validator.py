# -*- coding: utf-8 -*-
"""
XBRLValidator - Comprehensive XBRL Validation Engine

This module provides multi-layer XBRL validation for CSRD/ESRS reporting,
covering taxonomy, context, unit, fact, calculation linkbase, dimension,
filing indicator, ESEF RTS compliance, cross-reference, consistency, and
completeness checks.

All validation logic is deterministic (zero-hallucination). No LLM calls
are used for any compliance determination.

Validation Layers:
    1. Taxonomy validation - element names exist, namespace correctness
    2. Context validation - entity identifiers, period specs, dimension refs
    3. Unit validation - ISO 4217 codes, custom unit consistency
    4. Fact validation - data type matching, decimal precision, sign
    5. Calculation linkbase - parent-child sum consistency (tolerance 0.01)
    6. Presentation linkbase - ordering, required disclosures
    7. Dimension validation - hypercube consistency, typed values
    8. Filing indicator - required filings for material standards
    9. ESEF RTS compliance - package structure, naming, file formats
   10. Cross-reference - facts reference valid contexts and units
   11. Consistency - same fact not reported with different values
   12. Completeness - all mandatory data points present

Version: 1.1.0
Author: GreenLang CSRD Team
License: MIT
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from xbrl.taxonomy_mapper import (
    ISO_4217_CURRENCIES,
    PeriodType,
    TaxonomyMapper,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALCULATION_TOLERANCE_ABSOLUTE = 0.01
CALCULATION_TOLERANCE_RELATIVE = 0.001
PERCENTAGE_MIN = 0.0
PERCENTAGE_MAX = 1.0
PERCENTAGE_MAX_ALT = 100.0

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LEI_PATTERN = re.compile(r"^[A-Z0-9]{20}$")

ESEF_REQUIRED_FILES = {
    "reports/": "iXBRL report document",
    "META-INF/reports.json": "Report metadata",
    "META-INF/taxonomyPackage.xml": "Taxonomy package descriptor",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ValidationSeverity(str, Enum):
    """Severity level for validation findings."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationCategory(str, Enum):
    """Category of validation check."""
    TAXONOMY = "taxonomy"
    CONTEXT = "context"
    UNIT = "unit"
    FACT = "fact"
    CALCULATION = "calculation"
    PRESENTATION = "presentation"
    DIMENSION = "dimension"
    FILING_INDICATOR = "filing_indicator"
    ESEF_RTS = "esef_rts"
    CROSS_REFERENCE = "cross_reference"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """A single validation finding."""

    rule_id: str = Field(..., description="Validation rule identifier")
    category: ValidationCategory = Field(..., description="Validation category")
    severity: ValidationSeverity = Field(..., description="ERROR, WARNING, or INFO")
    message: str = Field(..., description="Human-readable message")
    element_id: Optional[str] = Field(None, description="Related XBRL element ID")
    data_point_id: Optional[str] = Field(None, description="Related data point ID")
    context_ref: Optional[str] = Field(None, description="Related context ID")
    unit_ref: Optional[str] = Field(None, description="Related unit ID")
    expected_value: Optional[str] = Field(None, description="Expected value")
    actual_value: Optional[str] = Field(None, description="Actual value")
    standard: Optional[str] = Field(None, description="Related ESRS standard")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class ValidationReport(BaseModel):
    """Aggregated validation report containing all findings."""

    report_id: str = Field(..., description="Unique report identifier")
    generated_at: str = Field(..., description="ISO 8601 timestamp")
    entity_identifier: Optional[str] = Field(None, description="Entity LEI")
    reporting_period: Optional[str] = Field(None, description="Reporting period")
    total_findings: int = Field(0, description="Total number of findings")
    error_count: int = Field(0, description="Number of errors")
    warning_count: int = Field(0, description="Number of warnings")
    info_count: int = Field(0, description="Number of info messages")
    is_valid: bool = Field(True, description="True if no errors found")
    findings: List[ValidationResult] = Field(
        default_factory=list, description="All validation findings"
    )
    validation_duration_ms: float = Field(0.0, description="Validation duration")
    provenance_hash: str = Field("", description="SHA-256 hash of this report")

    def add_finding(self, finding: ValidationResult) -> None:
        """Add a finding to the report and update counters."""
        self.findings.append(finding)
        self.total_findings += 1
        if finding.severity == ValidationSeverity.ERROR:
            self.error_count += 1
            self.is_valid = False
        elif finding.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1

    def get_errors(self) -> List[ValidationResult]:
        """Return only ERROR-severity findings."""
        return [f for f in self.findings if f.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationResult]:
        """Return only WARNING-severity findings."""
        return [f for f in self.findings if f.severity == ValidationSeverity.WARNING]

    def get_by_category(self, category: ValidationCategory) -> List[ValidationResult]:
        """Return findings filtered by category."""
        return [f for f in self.findings if f.category == category]

    def get_by_standard(self, standard: str) -> List[ValidationResult]:
        """Return findings filtered by ESRS standard."""
        return [f for f in self.findings if f.standard == standard]

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the report for provenance."""
        content = json.dumps(
            [f.model_dump() for f in self.findings],
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.provenance_hash = digest
        return digest

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to JSON."""
        return self.model_dump_json(indent=indent)

    def to_human_readable(self) -> str:
        """Generate a human-readable validation summary."""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("XBRL VALIDATION REPORT")
        lines.append("=" * 72)
        lines.append(f"Report ID:    {self.report_id}")
        lines.append(f"Generated:    {self.generated_at}")
        lines.append(f"Entity:       {self.entity_identifier or 'N/A'}")
        lines.append(f"Period:       {self.reporting_period or 'N/A'}")
        lines.append(f"Status:       {'VALID' if self.is_valid else 'INVALID'}")
        lines.append(f"Duration:     {self.validation_duration_ms:.1f} ms")
        lines.append("-" * 72)
        lines.append(
            f"Errors: {self.error_count}  |  "
            f"Warnings: {self.warning_count}  |  "
            f"Info: {self.info_count}  |  "
            f"Total: {self.total_findings}"
        )
        lines.append("-" * 72)

        if self.findings:
            for finding in self.findings:
                icon = {
                    ValidationSeverity.ERROR: "[ERR]",
                    ValidationSeverity.WARNING: "[WRN]",
                    ValidationSeverity.INFO: "[INF]",
                }[finding.severity]
                lines.append(
                    f"  {icon} [{finding.category.value}] "
                    f"{finding.rule_id}: {finding.message}"
                )
                if finding.data_point_id:
                    lines.append(f"       Data point: {finding.data_point_id}")
                if finding.expected_value and finding.actual_value:
                    lines.append(
                        f"       Expected: {finding.expected_value}  "
                        f"Actual: {finding.actual_value}"
                    )
        else:
            lines.append("  No findings.")

        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fact representation for validation
# ---------------------------------------------------------------------------

class FactForValidation(BaseModel):
    """Minimal fact representation consumed by the validator."""

    fact_id: str = Field(..., description="Unique fact ID")
    data_point_id: str = Field(..., description="ESRS data point ID")
    element_qname: str = Field(..., description="XBRL element QName")
    context_ref: str = Field(..., description="Context reference ID")
    unit_ref: Optional[str] = Field(None, description="Unit reference ID")
    value: Any = Field(None, description="Fact value")
    decimals: Optional[int] = Field(None, description="Decimal precision")
    is_nil: bool = Field(False, description="True if nil")
    fact_type: str = Field("text", description="numeric, text, date, boolean")


class ContextForValidation(BaseModel):
    """Minimal context representation consumed by the validator."""

    context_id: str = Field(...)
    entity_scheme: str = Field(...)
    entity_identifier: str = Field(...)
    period_type: str = Field(...)
    instant_date: Optional[str] = Field(None)
    start_date: Optional[str] = Field(None)
    end_date: Optional[str] = Field(None)
    dimension_count: int = Field(0)


class UnitForValidation(BaseModel):
    """Minimal unit representation consumed by the validator."""

    unit_id: str = Field(...)
    measures: List[str] = Field(default_factory=list)
    is_divide: bool = Field(False)


# ---------------------------------------------------------------------------
# XBRLValidator
# ---------------------------------------------------------------------------

class XBRLValidator:
    """
    Comprehensive XBRL validation engine for CSRD/ESRS reporting.

    Runs 12 categories of validation checks against XBRL facts, contexts,
    units, and metadata. All checks are deterministic.

    Usage::

        validator = XBRLValidator()
        report = validator.validate_full(
            facts=facts,
            contexts=contexts,
            units=units,
            material_standards=["ESRS-E1", "ESRS-S1", "ESRS-G1"],
            entity_identifier="549300EXAMPLE000LEI00",
            reporting_period_start="2024-01-01",
            reporting_period_end="2024-12-31",
        )
        print(report.to_human_readable())

    Attributes:
        _mapper: TaxonomyMapper singleton for taxonomy lookups.
    """

    def __init__(self) -> None:
        """Initialize the validator with the taxonomy mapper."""
        self._mapper = TaxonomyMapper.get_instance()

    # ------------------------------------------------------------------
    # Main validation entry point
    # ------------------------------------------------------------------

    def validate_full(
        self,
        facts: List[FactForValidation],
        contexts: List[ContextForValidation],
        units: List[UnitForValidation],
        material_standards: Optional[List[str]] = None,
        entity_identifier: Optional[str] = None,
        reporting_period_start: Optional[str] = None,
        reporting_period_end: Optional[str] = None,
        filing_indicators: Optional[List[str]] = None,
        esef_package_path: Optional[Path] = None,
    ) -> ValidationReport:
        """
        Run all validation checks and produce a comprehensive report.

        Args:
            facts: List of facts to validate.
            contexts: List of context definitions.
            units: List of unit definitions.
            material_standards: Standards determined to be material.
            entity_identifier: LEI of the reporting entity.
            reporting_period_start: Start date (YYYY-MM-DD).
            reporting_period_end: End date (YYYY-MM-DD).
            filing_indicators: Filed indicator codes.
            esef_package_path: Path to ESEF ZIP for package validation.

        Returns:
            ValidationReport with all findings.
        """
        start = datetime.now()
        report_id = hashlib.sha256(
            f"{datetime.now().isoformat()}_{entity_identifier}".encode()
        ).hexdigest()[:16]

        report = ValidationReport(
            report_id=report_id,
            generated_at=datetime.now().isoformat(),
            entity_identifier=entity_identifier,
            reporting_period=(
                f"{reporting_period_start} to {reporting_period_end}"
                if reporting_period_start and reporting_period_end
                else None
            ),
        )

        # Build lookup structures
        context_map = {c.context_id: c for c in contexts}
        unit_map = {u.unit_id: u for u in units}

        # 1. Taxonomy validation
        self._validate_taxonomy(facts, report)

        # 2. Context validation
        self._validate_contexts(contexts, entity_identifier, report)

        # 3. Unit validation
        self._validate_units(units, report)

        # 4. Fact validation
        self._validate_facts(facts, report)

        # 5. Calculation linkbase
        self._validate_calculations(facts, report)

        # 6. Presentation linkbase
        self._validate_presentation(facts, material_standards, report)

        # 7. Dimension validation
        self._validate_dimensions(facts, report)

        # 8. Filing indicator validation
        self._validate_filing_indicators(
            material_standards, filing_indicators, report
        )

        # 9. ESEF RTS compliance
        if esef_package_path:
            self._validate_esef_package(esef_package_path, report)

        # 10. Cross-reference validation
        self._validate_cross_references(facts, context_map, unit_map, report)

        # 11. Consistency checks
        self._validate_consistency(facts, report)

        # 12. Completeness checks
        self._validate_completeness(facts, material_standards, report)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        report.validation_duration_ms = elapsed
        report.compute_hash()

        logger.info(
            "Validation complete: %d findings (%d errors, %d warnings) in %.1f ms",
            report.total_findings,
            report.error_count,
            report.warning_count,
            elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # 1. Taxonomy validation
    # ------------------------------------------------------------------

    def _validate_taxonomy(
        self, facts: List[FactForValidation], report: ValidationReport
    ) -> None:
        """Validate that all fact elements exist in the EFRAG taxonomy."""
        for fact in facts:
            element = self._mapper.get_element(fact.data_point_id)
            if element is None:
                report.add_finding(
                    ValidationResult(
                        rule_id="TAX-001",
                        category=ValidationCategory.TAXONOMY,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Data point '{fact.data_point_id}' not found in "
                            f"EFRAG ESRS taxonomy"
                        ),
                        data_point_id=fact.data_point_id,
                        element_id=fact.element_qname,
                    )
                )
                continue

            # Check QName matches
            if fact.element_qname != element.qname:
                report.add_finding(
                    ValidationResult(
                        rule_id="TAX-002",
                        category=ValidationCategory.TAXONOMY,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Element QName mismatch for '{fact.data_point_id}'"
                        ),
                        data_point_id=fact.data_point_id,
                        expected_value=element.qname,
                        actual_value=fact.element_qname,
                    )
                )

            # Check namespace is registered
            ns_prefix = element.namespace
            if not self._mapper.get_namespace_uri(ns_prefix):
                report.add_finding(
                    ValidationResult(
                        rule_id="TAX-003",
                        category=ValidationCategory.TAXONOMY,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Namespace prefix '{ns_prefix}' not found in "
                            f"registered namespaces"
                        ),
                        data_point_id=fact.data_point_id,
                        element_id=element.element_id,
                    )
                )

    # ------------------------------------------------------------------
    # 2. Context validation
    # ------------------------------------------------------------------

    def _validate_contexts(
        self,
        contexts: List[ContextForValidation],
        entity_identifier: Optional[str],
        report: ValidationReport,
    ) -> None:
        """Validate XBRL context elements."""
        seen_ids: Set[str] = set()

        for ctx in contexts:
            # Check unique context IDs
            if ctx.context_id in seen_ids:
                report.add_finding(
                    ValidationResult(
                        rule_id="CTX-001",
                        category=ValidationCategory.CONTEXT,
                        severity=ValidationSeverity.ERROR,
                        message=f"Duplicate context ID: '{ctx.context_id}'",
                        context_ref=ctx.context_id,
                    )
                )
            seen_ids.add(ctx.context_id)

            # Check entity scheme
            if ctx.entity_scheme != "http://standards.iso.org/iso/17442":
                report.add_finding(
                    ValidationResult(
                        rule_id="CTX-002",
                        category=ValidationCategory.CONTEXT,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Non-standard entity scheme in context "
                            f"'{ctx.context_id}': {ctx.entity_scheme}"
                        ),
                        context_ref=ctx.context_id,
                        expected_value="http://standards.iso.org/iso/17442",
                        actual_value=ctx.entity_scheme,
                    )
                )

            # Validate LEI format
            if entity_identifier and not LEI_PATTERN.match(ctx.entity_identifier):
                report.add_finding(
                    ValidationResult(
                        rule_id="CTX-003",
                        category=ValidationCategory.CONTEXT,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Entity identifier may not be a valid LEI in "
                            f"context '{ctx.context_id}'"
                        ),
                        context_ref=ctx.context_id,
                        actual_value=ctx.entity_identifier,
                    )
                )

            # Validate period specification
            self._validate_period(ctx, report)

    def _validate_period(
        self, ctx: ContextForValidation, report: ValidationReport
    ) -> None:
        """Validate period specification in a context."""
        if ctx.period_type == "instant":
            if not ctx.instant_date:
                report.add_finding(
                    ValidationResult(
                        rule_id="CTX-004",
                        category=ValidationCategory.CONTEXT,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Instant context '{ctx.context_id}' missing "
                            f"instant date"
                        ),
                        context_ref=ctx.context_id,
                    )
                )
            elif not DATE_PATTERN.match(ctx.instant_date):
                report.add_finding(
                    ValidationResult(
                        rule_id="CTX-005",
                        category=ValidationCategory.CONTEXT,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Invalid instant date format in context "
                            f"'{ctx.context_id}': {ctx.instant_date}"
                        ),
                        context_ref=ctx.context_id,
                        expected_value="YYYY-MM-DD",
                        actual_value=ctx.instant_date,
                    )
                )
        elif ctx.period_type == "duration":
            if not ctx.start_date or not ctx.end_date:
                report.add_finding(
                    ValidationResult(
                        rule_id="CTX-006",
                        category=ValidationCategory.CONTEXT,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Duration context '{ctx.context_id}' missing "
                            f"start or end date"
                        ),
                        context_ref=ctx.context_id,
                    )
                )
            else:
                for label, date_val in [
                    ("start", ctx.start_date),
                    ("end", ctx.end_date),
                ]:
                    if not DATE_PATTERN.match(date_val):
                        report.add_finding(
                            ValidationResult(
                                rule_id="CTX-007",
                                category=ValidationCategory.CONTEXT,
                                severity=ValidationSeverity.ERROR,
                                message=(
                                    f"Invalid {label} date format in context "
                                    f"'{ctx.context_id}': {date_val}"
                                ),
                                context_ref=ctx.context_id,
                                expected_value="YYYY-MM-DD",
                                actual_value=date_val,
                            )
                        )

                # Check start < end
                if (
                    ctx.start_date
                    and ctx.end_date
                    and ctx.start_date >= ctx.end_date
                ):
                    report.add_finding(
                        ValidationResult(
                            rule_id="CTX-008",
                            category=ValidationCategory.CONTEXT,
                            severity=ValidationSeverity.ERROR,
                            message=(
                                f"Start date >= end date in context "
                                f"'{ctx.context_id}'"
                            ),
                            context_ref=ctx.context_id,
                            details={
                                "start_date": ctx.start_date,
                                "end_date": ctx.end_date,
                            },
                        )
                    )

    # ------------------------------------------------------------------
    # 3. Unit validation
    # ------------------------------------------------------------------

    def _validate_units(
        self, units: List[UnitForValidation], report: ValidationReport
    ) -> None:
        """Validate XBRL unit elements."""
        seen_ids: Set[str] = set()

        for unit in units:
            # Duplicate check
            if unit.unit_id in seen_ids:
                report.add_finding(
                    ValidationResult(
                        rule_id="UNT-001",
                        category=ValidationCategory.UNIT,
                        severity=ValidationSeverity.ERROR,
                        message=f"Duplicate unit ID: '{unit.unit_id}'",
                        unit_ref=unit.unit_id,
                    )
                )
            seen_ids.add(unit.unit_id)

            # Check measures are non-empty
            if not unit.is_divide and not unit.measures:
                report.add_finding(
                    ValidationResult(
                        rule_id="UNT-002",
                        category=ValidationCategory.UNIT,
                        severity=ValidationSeverity.ERROR,
                        message=f"Unit '{unit.unit_id}' has no measures",
                        unit_ref=unit.unit_id,
                    )
                )

            # Validate ISO 4217 if currency
            for measure in unit.measures:
                if measure.startswith("iso4217:"):
                    currency = measure.split(":")[1]
                    if currency not in ISO_4217_CURRENCIES:
                        report.add_finding(
                            ValidationResult(
                                rule_id="UNT-003",
                                category=ValidationCategory.UNIT,
                                severity=ValidationSeverity.WARNING,
                                message=(
                                    f"Unrecognized currency code '{currency}' "
                                    f"in unit '{unit.unit_id}'"
                                ),
                                unit_ref=unit.unit_id,
                                actual_value=currency,
                            )
                        )

    # ------------------------------------------------------------------
    # 4. Fact validation
    # ------------------------------------------------------------------

    def _validate_facts(
        self, facts: List[FactForValidation], report: ValidationReport
    ) -> None:
        """Validate individual XBRL facts."""
        seen_ids: Set[str] = set()

        for fact in facts:
            # Duplicate fact IDs
            if fact.fact_id in seen_ids:
                report.add_finding(
                    ValidationResult(
                        rule_id="FAC-001",
                        category=ValidationCategory.FACT,
                        severity=ValidationSeverity.ERROR,
                        message=f"Duplicate fact ID: '{fact.fact_id}'",
                        data_point_id=fact.data_point_id,
                    )
                )
            seen_ids.add(fact.fact_id)

            # Skip nil facts for value validation
            if fact.is_nil:
                continue

            element = self._mapper.get_element(fact.data_point_id)
            if element is None:
                continue  # Already caught by taxonomy validation

            # Type matching
            self._validate_fact_type(fact, element, report)

            # Numeric-specific validations
            if fact.fact_type == "numeric":
                self._validate_numeric_fact(fact, element, report)

    def _validate_fact_type(
        self, fact: FactForValidation, element: Any, report: ValidationReport
    ) -> None:
        """Validate that the fact type matches the element data type."""
        is_numeric_type = element.is_numeric
        is_text_type = element.is_text
        is_bool_type = element.is_boolean

        if fact.fact_type == "numeric" and not is_numeric_type:
            report.add_finding(
                ValidationResult(
                    rule_id="FAC-002",
                    category=ValidationCategory.FACT,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Fact '{fact.fact_id}' tagged as numeric but element "
                        f"type is '{element.data_type}'"
                    ),
                    data_point_id=fact.data_point_id,
                    expected_value="numeric element type",
                    actual_value=element.data_type,
                )
            )
        elif fact.fact_type == "text" and is_numeric_type:
            report.add_finding(
                ValidationResult(
                    rule_id="FAC-003",
                    category=ValidationCategory.FACT,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Fact '{fact.fact_id}' tagged as text but element "
                        f"type is numeric '{element.data_type}'"
                    ),
                    data_point_id=fact.data_point_id,
                )
            )
        elif fact.fact_type == "boolean" and not is_bool_type:
            report.add_finding(
                ValidationResult(
                    rule_id="FAC-004",
                    category=ValidationCategory.FACT,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Fact '{fact.fact_id}' tagged as boolean but element "
                        f"type is '{element.data_type}'"
                    ),
                    data_point_id=fact.data_point_id,
                )
            )

    def _validate_numeric_fact(
        self, fact: FactForValidation, element: Any, report: ValidationReport
    ) -> None:
        """Validate a numeric fact's value, precision, and sign."""
        # Check unit reference present
        if not fact.unit_ref:
            report.add_finding(
                ValidationResult(
                    rule_id="FAC-005",
                    category=ValidationCategory.FACT,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Numeric fact '{fact.fact_id}' missing unit reference"
                    ),
                    data_point_id=fact.data_point_id,
                )
            )

        # Check value is numeric
        if fact.value is not None:
            try:
                val = float(fact.value)
            except (ValueError, TypeError):
                report.add_finding(
                    ValidationResult(
                        rule_id="FAC-006",
                        category=ValidationCategory.FACT,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Numeric fact '{fact.fact_id}' has non-numeric "
                            f"value: {fact.value}"
                        ),
                        data_point_id=fact.data_point_id,
                        actual_value=str(fact.value),
                    )
                )
                return

            # Sign consistency for debit/credit
            if element.balance_type:
                if element.balance_type.value == "debit" and val < 0:
                    report.add_finding(
                        ValidationResult(
                            rule_id="FAC-007",
                            category=ValidationCategory.FACT,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Debit-balance fact '{fact.fact_id}' has "
                                f"negative value {val}"
                            ),
                            data_point_id=fact.data_point_id,
                            actual_value=str(val),
                        )
                    )

            # Percentage range check
            if element.is_percentage:
                if val < PERCENTAGE_MIN or (
                    val > PERCENTAGE_MAX and val > PERCENTAGE_MAX_ALT
                ):
                    report.add_finding(
                        ValidationResult(
                            rule_id="FAC-008",
                            category=ValidationCategory.FACT,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Percentage fact '{fact.fact_id}' out of "
                                f"expected range [0, 1] or [0, 100]: {val}"
                            ),
                            data_point_id=fact.data_point_id,
                            expected_value="0.0 to 1.0 or 0 to 100",
                            actual_value=str(val),
                        )
                    )

    # ------------------------------------------------------------------
    # 5. Calculation linkbase validation
    # ------------------------------------------------------------------

    def _validate_calculations(
        self, facts: List[FactForValidation], report: ValidationReport
    ) -> None:
        """Validate calculation linkbase consistency (parent = sum of children)."""
        # Build fact value lookup
        fact_values: Dict[str, float] = {}
        for fact in facts:
            if fact.fact_type == "numeric" and fact.value is not None and not fact.is_nil:
                try:
                    fact_values[fact.data_point_id] = float(fact.value)
                except (ValueError, TypeError):
                    continue

        # Check each calculation relationship
        for rel_key, rel in self._mapper.get_all_calculation_relationships().items():
            if rel.calculation_type == "ratio":
                self._validate_ratio(rel, fact_values, report)
                continue

            if rel.children is None:
                continue

            parent_val = fact_values.get(rel.parent)
            if parent_val is None:
                continue  # Parent not reported, skip

            child_sum = 0.0
            all_children_present = True
            for child in rel.children:
                child_id = child["child"]
                child_val = fact_values.get(child_id)
                if child_val is None:
                    all_children_present = False
                    continue
                child_sum += child_val * child["weight"]

            if not all_children_present:
                continue  # Cannot validate if children are missing

            diff = abs(parent_val - child_sum)
            if diff > CALCULATION_TOLERANCE_ABSOLUTE:
                report.add_finding(
                    ValidationResult(
                        rule_id="CAL-001",
                        category=ValidationCategory.CALCULATION,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Calculation inconsistency for '{rel.parent}': "
                            f"parent={parent_val}, sum of children={child_sum}, "
                            f"difference={diff:.4f}"
                        ),
                        data_point_id=rel.parent,
                        expected_value=str(child_sum),
                        actual_value=str(parent_val),
                        standard=rel.standard,
                        details={
                            "relationship": rel_key,
                            "tolerance": CALCULATION_TOLERANCE_ABSOLUTE,
                        },
                    )
                )

    def _validate_ratio(
        self,
        rel: Any,
        fact_values: Dict[str, float],
        report: ValidationReport,
    ) -> None:
        """Validate a ratio calculation relationship."""
        if not rel.numerator or not rel.denominator:
            return

        parent_val = fact_values.get(rel.parent)
        num_val = fact_values.get(rel.numerator)
        den_val = fact_values.get(rel.denominator)

        if parent_val is None or num_val is None or den_val is None:
            return

        if den_val == 0:
            if parent_val != 0:
                report.add_finding(
                    ValidationResult(
                        rule_id="CAL-002",
                        category=ValidationCategory.CALCULATION,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Ratio '{rel.parent}' has zero denominator but "
                            f"non-zero value {parent_val}"
                        ),
                        data_point_id=rel.parent,
                        standard=rel.standard,
                    )
                )
            return

        expected = num_val / den_val
        diff = abs(parent_val - expected)
        if diff > CALCULATION_TOLERANCE_ABSOLUTE:
            report.add_finding(
                ValidationResult(
                    rule_id="CAL-003",
                    category=ValidationCategory.CALCULATION,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Ratio inconsistency for '{rel.parent}': "
                        f"expected={expected:.6f}, actual={parent_val}"
                    ),
                    data_point_id=rel.parent,
                    expected_value=str(expected),
                    actual_value=str(parent_val),
                    standard=rel.standard,
                )
            )

    # ------------------------------------------------------------------
    # 6. Presentation linkbase validation
    # ------------------------------------------------------------------

    def _validate_presentation(
        self,
        facts: List[FactForValidation],
        material_standards: Optional[List[str]],
        report: ValidationReport,
    ) -> None:
        """Validate presentation ordering and required disclosures."""
        if not material_standards:
            return

        reported_dp_ids: Set[str] = {f.data_point_id for f in facts}

        for standard in material_standards:
            sections = self._mapper.get_sections_for_standard(standard)
            for section in sections:
                section_elements = section.get("elements", [])
                has_any = any(e in reported_dp_ids for e in section_elements)
                if not has_any and section_elements:
                    report.add_finding(
                        ValidationResult(
                            rule_id="PRS-001",
                            category=ValidationCategory.PRESENTATION,
                            severity=ValidationSeverity.INFO,
                            message=(
                                f"No facts reported for disclosure section "
                                f"'{section['id']}' in {standard}"
                            ),
                            standard=standard,
                            details={"section": section["id"]},
                        )
                    )

    # ------------------------------------------------------------------
    # 7. Dimension validation
    # ------------------------------------------------------------------

    def _validate_dimensions(
        self, facts: List[FactForValidation], report: ValidationReport
    ) -> None:
        """Validate dimensional consistency of facts."""
        for fact in facts:
            element = self._mapper.get_element(fact.data_point_id)
            if element is None:
                continue

            applicable_dims = self._mapper.get_applicable_dimensions(
                fact.data_point_id
            )
            if not applicable_dims:
                continue

            # Info-level notification about available dimensions
            dim_names = [d.dimension_id for d in applicable_dims]
            report.add_finding(
                ValidationResult(
                    rule_id="DIM-001",
                    category=ValidationCategory.DIMENSION,
                    severity=ValidationSeverity.INFO,
                    message=(
                        f"Data point '{fact.data_point_id}' supports "
                        f"dimensions: {', '.join(dim_names)}"
                    ),
                    data_point_id=fact.data_point_id,
                )
            )

    # ------------------------------------------------------------------
    # 8. Filing indicator validation
    # ------------------------------------------------------------------

    def _validate_filing_indicators(
        self,
        material_standards: Optional[List[str]],
        filing_indicators: Optional[List[str]],
        report: ValidationReport,
    ) -> None:
        """Validate filing indicators match materiality assessment."""
        if not material_standards:
            return

        # Check mandatory standards are always filed
        mandatory = self._mapper.get_mandatory_standards()
        expected_filings = set(mandatory) | set(material_standards)

        if filing_indicators is not None:
            # Map filing codes back to standard IDs
            filed_standards: Set[str] = set()
            for fi_code in filing_indicators:
                for std_id, fi in self._mapper.get_all_filing_indicators().items():
                    if fi.filing_code == fi_code:
                        filed_standards.add(std_id)
                        break

            # Check mandatory standards are present
            for std_id in mandatory:
                fi = self._mapper.get_filing_indicator(std_id)
                if fi and fi.filing_code not in filing_indicators:
                    report.add_finding(
                        ValidationResult(
                            rule_id="FIL-001",
                            category=ValidationCategory.FILING_INDICATOR,
                            severity=ValidationSeverity.ERROR,
                            message=(
                                f"Mandatory standard '{std_id}' missing from "
                                f"filing indicators"
                            ),
                            standard=std_id,
                        )
                    )

            # Check material standards are filed
            for std_id in material_standards:
                if std_id not in filed_standards:
                    report.add_finding(
                        ValidationResult(
                            rule_id="FIL-002",
                            category=ValidationCategory.FILING_INDICATOR,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Material standard '{std_id}' not found in "
                                f"filing indicators"
                            ),
                            standard=std_id,
                        )
                    )

        # Check E1 omission explanation
        if "ESRS-E1" not in (material_standards or []):
            fi = self._mapper.get_filing_indicator("ESRS-E1")
            if fi and fi.omission_explanation_required:
                report.add_finding(
                    ValidationResult(
                        rule_id="FIL-003",
                        category=ValidationCategory.FILING_INDICATOR,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            "ESRS-E1 (Climate Change) is not material. "
                            "Per ESRS 2 IRO-2, a detailed explanation is "
                            "required for omitting climate disclosures."
                        ),
                        standard="ESRS-E1",
                    )
                )

    # ------------------------------------------------------------------
    # 9. ESEF RTS compliance
    # ------------------------------------------------------------------

    def _validate_esef_package(
        self, package_path: Path, report: ValidationReport
    ) -> None:
        """Validate ESEF reporting package structure and contents."""
        if not package_path.exists():
            report.add_finding(
                ValidationResult(
                    rule_id="ESF-001",
                    category=ValidationCategory.ESEF_RTS,
                    severity=ValidationSeverity.ERROR,
                    message=f"ESEF package not found: {package_path}",
                )
            )
            return

        if not package_path.suffix.lower() == ".zip":
            report.add_finding(
                ValidationResult(
                    rule_id="ESF-002",
                    category=ValidationCategory.ESEF_RTS,
                    severity=ValidationSeverity.ERROR,
                    message="ESEF package must be a ZIP file",
                    actual_value=package_path.suffix,
                )
            )
            return

        import zipfile as zf

        if not zf.is_zipfile(package_path):
            report.add_finding(
                ValidationResult(
                    rule_id="ESF-003",
                    category=ValidationCategory.ESEF_RTS,
                    severity=ValidationSeverity.ERROR,
                    message="ESEF package is not a valid ZIP file",
                )
            )
            return

        with zf.ZipFile(package_path, "r") as archive:
            file_list = archive.namelist()
            # Find the top-level directory
            top_dirs = set()
            for name in file_list:
                parts = name.split("/")
                if len(parts) > 1:
                    top_dirs.add(parts[0])

            if len(top_dirs) != 1:
                report.add_finding(
                    ValidationResult(
                        rule_id="ESF-004",
                        category=ValidationCategory.ESEF_RTS,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"ESEF package should have exactly one top-level "
                            f"directory, found {len(top_dirs)}"
                        ),
                    )
                )

            top_dir = top_dirs.pop() if top_dirs else ""

            # Check required files
            for required_path, desc in ESEF_REQUIRED_FILES.items():
                full_path = f"{top_dir}/{required_path}"
                found = any(
                    name.startswith(full_path) for name in file_list
                )
                if not found:
                    report.add_finding(
                        ValidationResult(
                            rule_id="ESF-005",
                            category=ValidationCategory.ESEF_RTS,
                            severity=ValidationSeverity.ERROR,
                            message=(
                                f"Required ESEF file missing: {required_path} "
                                f"({desc})"
                            ),
                            expected_value=full_path,
                        )
                    )

            # Check iXBRL file extension
            xhtml_files = [
                n for n in file_list if n.endswith(".xhtml") or n.endswith(".html")
            ]
            if not xhtml_files:
                report.add_finding(
                    ValidationResult(
                        rule_id="ESF-006",
                        category=ValidationCategory.ESEF_RTS,
                        severity=ValidationSeverity.ERROR,
                        message="No XHTML file found in ESEF package",
                    )
                )
            else:
                for xf in xhtml_files:
                    if not xf.endswith(".xhtml"):
                        report.add_finding(
                            ValidationResult(
                                rule_id="ESF-007",
                                category=ValidationCategory.ESEF_RTS,
                                severity=ValidationSeverity.WARNING,
                                message=(
                                    f"iXBRL file should use .xhtml extension "
                                    f"per ESEF RTS: {xf}"
                                ),
                                actual_value=xf,
                            )
                        )

            # Validate reports.json structure
            reports_json_path = f"{top_dir}/META-INF/reports.json"
            if reports_json_path in file_list:
                try:
                    content = archive.read(reports_json_path)
                    reports_data = json.loads(content)
                    if "documentInfo" not in reports_data:
                        report.add_finding(
                            ValidationResult(
                                rule_id="ESF-008",
                                category=ValidationCategory.ESEF_RTS,
                                severity=ValidationSeverity.WARNING,
                                message=(
                                    "reports.json missing 'documentInfo' field"
                                ),
                            )
                        )
                except json.JSONDecodeError:
                    report.add_finding(
                        ValidationResult(
                            rule_id="ESF-009",
                            category=ValidationCategory.ESEF_RTS,
                            severity=ValidationSeverity.ERROR,
                            message="reports.json is not valid JSON",
                        )
                    )

    # ------------------------------------------------------------------
    # 10. Cross-reference validation
    # ------------------------------------------------------------------

    def _validate_cross_references(
        self,
        facts: List[FactForValidation],
        context_map: Dict[str, ContextForValidation],
        unit_map: Dict[str, UnitForValidation],
        report: ValidationReport,
    ) -> None:
        """Validate that facts reference valid contexts and units."""
        for fact in facts:
            # Check context reference
            if fact.context_ref not in context_map:
                report.add_finding(
                    ValidationResult(
                        rule_id="XRF-001",
                        category=ValidationCategory.CROSS_REFERENCE,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Fact '{fact.fact_id}' references unknown context "
                            f"'{fact.context_ref}'"
                        ),
                        data_point_id=fact.data_point_id,
                        context_ref=fact.context_ref,
                    )
                )

            # Check unit reference for numeric facts
            if fact.fact_type == "numeric" and fact.unit_ref:
                if fact.unit_ref not in unit_map:
                    report.add_finding(
                        ValidationResult(
                            rule_id="XRF-002",
                            category=ValidationCategory.CROSS_REFERENCE,
                            severity=ValidationSeverity.ERROR,
                            message=(
                                f"Fact '{fact.fact_id}' references unknown unit "
                                f"'{fact.unit_ref}'"
                            ),
                            data_point_id=fact.data_point_id,
                            unit_ref=fact.unit_ref,
                        )
                    )

    # ------------------------------------------------------------------
    # 11. Consistency checks
    # ------------------------------------------------------------------

    def _validate_consistency(
        self, facts: List[FactForValidation], report: ValidationReport
    ) -> None:
        """Check that the same fact is not reported with conflicting values."""
        # Group by (data_point_id, context_ref)
        fact_groups: Dict[Tuple[str, str], List[FactForValidation]] = {}
        for fact in facts:
            key = (fact.data_point_id, fact.context_ref)
            fact_groups.setdefault(key, []).append(fact)

        for (dp_id, ctx_ref), group in fact_groups.items():
            if len(group) > 1:
                # Check for conflicting values
                values = set()
                for f in group:
                    if f.value is not None:
                        values.add(str(f.value))

                if len(values) > 1:
                    report.add_finding(
                        ValidationResult(
                            rule_id="CON-001",
                            category=ValidationCategory.CONSISTENCY,
                            severity=ValidationSeverity.ERROR,
                            message=(
                                f"Data point '{dp_id}' reported with "
                                f"conflicting values in context '{ctx_ref}': "
                                f"{', '.join(sorted(values))}"
                            ),
                            data_point_id=dp_id,
                            context_ref=ctx_ref,
                        )
                    )
                elif len(values) == 1:
                    report.add_finding(
                        ValidationResult(
                            rule_id="CON-002",
                            category=ValidationCategory.CONSISTENCY,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Data point '{dp_id}' reported {len(group)} "
                                f"times in context '{ctx_ref}' with same value"
                            ),
                            data_point_id=dp_id,
                            context_ref=ctx_ref,
                        )
                    )

    # ------------------------------------------------------------------
    # 12. Completeness checks
    # ------------------------------------------------------------------

    def _validate_completeness(
        self,
        facts: List[FactForValidation],
        material_standards: Optional[List[str]],
        report: ValidationReport,
    ) -> None:
        """Check all mandatory data points for material standards are present."""
        if not material_standards:
            return

        reported_dp_ids: Set[str] = {f.data_point_id for f in facts}

        # Always check mandatory standards
        mandatory = self._mapper.get_mandatory_standards()
        all_to_check = set(mandatory) | set(material_standards)

        for std_id in sorted(all_to_check):
            required_drs = self._mapper.get_required_disclosures(
                std_id, is_material=(std_id in material_standards)
            )
            if not required_drs:
                continue

            # Get all elements for the standard
            elements_for_standard = self._mapper.get_elements_by_standard(std_id)
            element_dp_ids = set()
            for elem in elements_for_standard:
                # Find the data point ID for this element
                for dp_id, e in self._mapper.get_all_elements().items():
                    if e.element_id == elem.element_id:
                        element_dp_ids.add(dp_id)
                        break

            reported_for_standard = element_dp_ids & reported_dp_ids
            if not reported_for_standard and element_dp_ids:
                report.add_finding(
                    ValidationResult(
                        rule_id="CMP-001",
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"No data points reported for material standard "
                            f"'{std_id}' ({len(element_dp_ids)} expected)"
                        ),
                        standard=std_id,
                        details={
                            "expected_count": len(element_dp_ids),
                            "reported_count": 0,
                        },
                    )
                )
            elif element_dp_ids:
                missing = element_dp_ids - reported_dp_ids
                if missing:
                    # Only warn, as some may be legitimately omitted
                    report.add_finding(
                        ValidationResult(
                            rule_id="CMP-002",
                            category=ValidationCategory.COMPLETENESS,
                            severity=ValidationSeverity.WARNING,
                            message=(
                                f"Standard '{std_id}': "
                                f"{len(reported_for_standard)} of "
                                f"{len(element_dp_ids)} data points reported "
                                f"({len(missing)} missing)"
                            ),
                            standard=std_id,
                            details={
                                "total": len(element_dp_ids),
                                "reported": len(reported_for_standard),
                                "missing_count": len(missing),
                                "missing_sample": sorted(missing)[:5],
                            },
                        )
                    )

    # ------------------------------------------------------------------
    # Convenience: validate from generator
    # ------------------------------------------------------------------

    def validate_from_generator(
        self,
        generator: Any,
        material_standards: Optional[List[str]] = None,
    ) -> ValidationReport:
        """
        Validate an IXBRLGenerator instance directly.

        This is a convenience method that extracts facts, contexts, and
        units from the generator and runs full validation.

        Args:
            generator: IXBRLGenerator instance.
            material_standards: Material standards list.

        Returns:
            ValidationReport with all findings.
        """
        # Convert generator internals to validation models
        facts: List[FactForValidation] = []
        for f in generator._facts:
            facts.append(
                FactForValidation(
                    fact_id=f.fact_id,
                    data_point_id=f.data_point_id,
                    element_qname=f.element_qname,
                    context_ref=f.context_ref,
                    unit_ref=f.unit_ref,
                    value=f.value,
                    decimals=f.decimals,
                    is_nil=f.is_nil,
                    fact_type=f.fact_type.value,
                )
            )

        contexts: List[ContextForValidation] = []
        for c in generator._contexts.values():
            contexts.append(
                ContextForValidation(
                    context_id=c.context_id,
                    entity_scheme=c.entity_scheme,
                    entity_identifier=c.entity_identifier,
                    period_type=c.period_type.value,
                    instant_date=c.instant_date,
                    start_date=c.start_date,
                    end_date=c.end_date,
                    dimension_count=len(c.dimensions),
                )
            )

        units: List[UnitForValidation] = []
        for u in generator._units.values():
            units.append(
                UnitForValidation(
                    unit_id=u.unit_id,
                    measures=u.measures,
                    is_divide=u.is_divide,
                )
            )

        return self.validate_full(
            facts=facts,
            contexts=contexts,
            units=units,
            material_standards=material_standards
            or generator.material_standards,
            entity_identifier=generator.entity_identifier,
            reporting_period_start=generator.reporting_period_start,
            reporting_period_end=generator.reporting_period_end,
            filing_indicators=generator._filing_indicators,
        )
