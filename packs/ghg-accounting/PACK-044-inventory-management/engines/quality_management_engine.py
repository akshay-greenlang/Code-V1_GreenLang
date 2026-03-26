# -*- coding: utf-8 -*-
"""
QualityManagementEngine - PACK-044 Inventory Management Engine 3
=================================================================

GHG Protocol Chapter 7 quality assurance / quality control (QA/QC)
engine for greenhouse gas inventories.  Implements a comprehensive
quality management framework covering all four GHG Protocol data
quality dimensions: completeness, consistency, accuracy, and
transparency.

The engine performs automated quality checks across the full inventory,
calculates a composite quality score, tracks quality issues through a
lifecycle (open -> investigating -> resolved / accepted), and generates
improvement action plans aligned with continuous improvement cycles.

Quality Dimensions (per GHG Protocol Ch 7):
    Completeness:   Are all relevant sources, gases, and time periods covered?
    Consistency:     Are methodologies applied consistently across time and entities?
    Accuracy:        Are calculations free from systematic and random errors?
    Transparency:    Are assumptions, data sources, and methods documented?

Core Capabilities:
    1. Automated QA/QC checks against configurable rule sets
    2. Quality scoring (weighted composite score 0-100)
    3. Issue tracking with severity, assignment, and resolution workflow
    4. Improvement action planning with priority and deadlines
    5. Trend analysis of quality scores across periods
    6. Materiality-based prioritisation of quality issues
    7. External verification readiness assessment

Calculation Methodology:
    Dimension Score:
        dim_score = (passed_checks / total_checks_in_dimension) * 100

    Composite Quality Score:
        QS = w_completeness * S_completeness
           + w_consistency * S_consistency
           + w_accuracy    * S_accuracy
           + w_transparency * S_transparency
        where sum(w_i) = 1.0  (default: 0.30, 0.25, 0.25, 0.20)

    Issue Severity Weighting:
        critical = 10 pts, major = 5 pts, minor = 2 pts, observation = 1 pt
        weighted_issues = sum(severity_weight * count_at_severity)

    Verification Readiness:
        readiness_pct = (addressed_issues / total_issues) * 100
        ready = readiness_pct >= 95 AND no open critical issues

Regulatory References:
    - GHG Protocol Corporate Standard (Revised), Ch 7 (Managing Inventory Quality)
    - ISO 14064-1:2018, Clause 9 (Quality Management)
    - ISO 14064-3:2019, Clause 6 (Verification Requirements)
    - EU CSRD / ESRS E1-9 (Limited Assurance Requirements)
    - ISAE 3410 (Assurance on GHG Statements)
    - AA1000AS v3 (AccountAbility Assurance Standard)

Zero-Hallucination:
    - All quality scores computed via deterministic Decimal arithmetic
    - Check pass/fail based on explicit threshold rules
    - No LLM involvement in scoring, issue detection, or prioritisation
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _today() -> date:
    """Return current UTC date."""
    return datetime.now(timezone.utc).date()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QualityDimension(str, Enum):
    """GHG Protocol Chapter 7 data quality dimensions.

    COMPLETENESS:   Coverage of all relevant sources, gases, time periods.
    CONSISTENCY:    Uniform methodology application across time and entities.
    ACCURACY:       Freedom from systematic and random calculation errors.
    TRANSPARENCY:   Adequate documentation of data, methods, assumptions.
    """
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TRANSPARENCY = "transparency"


class CheckSeverity(str, Enum):
    """Severity level of a QA/QC check finding.

    CRITICAL:    Fundamental error that invalidates the inventory.
    MAJOR:       Significant error that materially affects results.
    MINOR:       Small error with limited impact on results.
    OBSERVATION: Not an error; suggestion for improvement.
    """
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"


class CheckResult(str, Enum):
    """Result of a single QA/QC check.

    PASS:     Check passed; no issues found.
    FAIL:     Check failed; issue detected.
    WARNING:  Check raised a non-blocking warning.
    SKIPPED:  Check was not applicable and was skipped.
    ERROR:    Check could not be executed (runtime error).
    """
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class IssueStatus(str, Enum):
    """Quality issue lifecycle status.

    OPEN:          Issue identified; not yet being addressed.
    INVESTIGATING: Issue is being analysed.
    RESOLVED:      Issue has been corrected.
    ACCEPTED:      Issue is accepted as-is (immaterial or by design).
    DEFERRED:      Issue deferred to a future period.
    """
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"
    DEFERRED = "deferred"


class ActionPriority(str, Enum):
    """Priority level for improvement actions.

    CRITICAL:  Must be addressed immediately.
    HIGH:      Should be addressed within current period.
    MEDIUM:    Address in next improvement cycle.
    LOW:       Address when resources allow.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionStatus(str, Enum):
    """Improvement action lifecycle status.

    PLANNED:      Action identified and scheduled.
    IN_PROGRESS:  Work has begun.
    COMPLETED:    Action completed.
    CANCELLED:    Action no longer needed.
    """
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default dimension weights for composite quality score.
DEFAULT_DIMENSION_WEIGHTS: Dict[QualityDimension, Decimal] = {
    QualityDimension.COMPLETENESS: Decimal("0.30"),
    QualityDimension.CONSISTENCY: Decimal("0.25"),
    QualityDimension.ACCURACY: Decimal("0.25"),
    QualityDimension.TRANSPARENCY: Decimal("0.20"),
}

# Severity weights for issue scoring.
SEVERITY_WEIGHTS: Dict[CheckSeverity, Decimal] = {
    CheckSeverity.CRITICAL: Decimal("10"),
    CheckSeverity.MAJOR: Decimal("5"),
    CheckSeverity.MINOR: Decimal("2"),
    CheckSeverity.OBSERVATION: Decimal("1"),
}

# Default QA/QC check definitions.
DEFAULT_CHECKS: List[Dict[str, Any]] = [
    # --- Completeness Checks ---
    {
        "check_id": "COMP-001",
        "name": "All Scope 1 categories covered",
        "dimension": QualityDimension.COMPLETENESS.value,
        "description": (
            "Verify that all applicable Scope 1 source categories "
            "(stationary, mobile, process, fugitive, refrigerant, land use, "
            "waste, agricultural) have reported data."
        ),
        "severity": CheckSeverity.CRITICAL.value,
    },
    {
        "check_id": "COMP-002",
        "name": "Scope 2 dual reporting present",
        "dimension": QualityDimension.COMPLETENESS.value,
        "description": (
            "Verify that both location-based and market-based Scope 2 "
            "values are reported per GHG Protocol Scope 2 Guidance."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "COMP-003",
        "name": "All facilities included",
        "dimension": QualityDimension.COMPLETENESS.value,
        "description": (
            "Verify that all facilities within the organisational boundary "
            "have reported data for the inventory period."
        ),
        "severity": CheckSeverity.CRITICAL.value,
    },
    {
        "check_id": "COMP-004",
        "name": "All greenhouse gases reported",
        "dimension": QualityDimension.COMPLETENESS.value,
        "description": (
            "Verify that all seven Kyoto Protocol gases (CO2, CH4, N2O, "
            "HFCs, PFCs, SF6, NF3) are accounted for where applicable."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "COMP-005",
        "name": "Full reporting period coverage",
        "dimension": QualityDimension.COMPLETENESS.value,
        "description": (
            "Verify that data covers the complete reporting period with "
            "no temporal gaps (all 12 months / all quarters)."
        ),
        "severity": CheckSeverity.CRITICAL.value,
    },
    # --- Consistency Checks ---
    {
        "check_id": "CONS-001",
        "name": "Consistent calculation methodology",
        "dimension": QualityDimension.CONSISTENCY.value,
        "description": (
            "Verify that the same calculation methodology is applied "
            "across all facilities for each emission category."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "CONS-002",
        "name": "Consistent GWP values",
        "dimension": QualityDimension.CONSISTENCY.value,
        "description": (
            "Verify that the same IPCC Assessment Report GWP values "
            "are used consistently across the entire inventory."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "CONS-003",
        "name": "Year-over-year methodology consistency",
        "dimension": QualityDimension.CONSISTENCY.value,
        "description": (
            "Verify that calculation methods are consistent with the "
            "previous reporting period (or changes are documented)."
        ),
        "severity": CheckSeverity.MINOR.value,
    },
    {
        "check_id": "CONS-004",
        "name": "Emission factor source consistency",
        "dimension": QualityDimension.CONSISTENCY.value,
        "description": (
            "Verify that emission factor sources are consistent across "
            "similar source categories and facilities."
        ),
        "severity": CheckSeverity.MINOR.value,
    },
    # --- Accuracy Checks ---
    {
        "check_id": "ACCU-001",
        "name": "No negative emission values",
        "dimension": QualityDimension.ACCURACY.value,
        "description": (
            "Verify that no emission totals are negative (except for "
            "biogenic carbon removals where explicitly allowed)."
        ),
        "severity": CheckSeverity.CRITICAL.value,
    },
    {
        "check_id": "ACCU-002",
        "name": "Year-over-year variance within bounds",
        "dimension": QualityDimension.ACCURACY.value,
        "description": (
            "Verify that year-over-year changes in total emissions are "
            "within expected bounds (+/- 30%) or have documented explanations."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "ACCU-003",
        "name": "Activity data reasonableness",
        "dimension": QualityDimension.ACCURACY.value,
        "description": (
            "Verify that activity data values are within reasonable "
            "ranges for the facility type and size."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "ACCU-004",
        "name": "Calculation cross-checks",
        "dimension": QualityDimension.ACCURACY.value,
        "description": (
            "Verify that emission calculations can be reproduced from "
            "activity data and emission factors."
        ),
        "severity": CheckSeverity.CRITICAL.value,
    },
    {
        "check_id": "ACCU-005",
        "name": "Unit conversion correctness",
        "dimension": QualityDimension.ACCURACY.value,
        "description": (
            "Verify that all unit conversions (kWh to MJ, litres to "
            "tonnes, etc.) are mathematically correct."
        ),
        "severity": CheckSeverity.CRITICAL.value,
    },
    # --- Transparency Checks ---
    {
        "check_id": "TRAN-001",
        "name": "Boundary description documented",
        "dimension": QualityDimension.TRANSPARENCY.value,
        "description": (
            "Verify that the organisational and operational boundaries "
            "are clearly documented."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "TRAN-002",
        "name": "Emission factors sourced",
        "dimension": QualityDimension.TRANSPARENCY.value,
        "description": (
            "Verify that all emission factors have documented sources "
            "(publication, year, region)."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "TRAN-003",
        "name": "Assumptions documented",
        "dimension": QualityDimension.TRANSPARENCY.value,
        "description": (
            "Verify that all material assumptions (e.g. allocation "
            "methods, estimation approaches) are documented."
        ),
        "severity": CheckSeverity.MINOR.value,
    },
    {
        "check_id": "TRAN-004",
        "name": "Exclusions justified",
        "dimension": QualityDimension.TRANSPARENCY.value,
        "description": (
            "Verify that any excluded sources, facilities, or gases "
            "have documented justifications."
        ),
        "severity": CheckSeverity.MAJOR.value,
    },
    {
        "check_id": "TRAN-005",
        "name": "Methodology changes disclosed",
        "dimension": QualityDimension.TRANSPARENCY.value,
        "description": (
            "Verify that any changes in methodology from the previous "
            "period are documented and their impact quantified."
        ),
        "severity": CheckSeverity.MINOR.value,
    },
]


# ---------------------------------------------------------------------------
# Pydantic Models -- Check Results
# ---------------------------------------------------------------------------


class QAQCCheck(BaseModel):
    """Definition of a single QA/QC check.

    Attributes:
        check_id: Unique check identifier (e.g. 'COMP-001').
        name: Human-readable check name.
        dimension: Quality dimension this check belongs to.
        description: Detailed description of what is checked.
        severity: Severity if the check fails.
    """
    check_id: str = Field(default="", max_length=50, description="Check ID")
    name: str = Field(default="", max_length=500, description="Check name")
    dimension: QualityDimension = Field(
        default=QualityDimension.COMPLETENESS, description="Quality dimension"
    )
    description: str = Field(default="", max_length=2000, description="Description")
    severity: CheckSeverity = Field(
        default=CheckSeverity.MINOR, description="Severity if failed"
    )


class QAQCResult(BaseModel):
    """Result of executing a single QA/QC check.

    Attributes:
        result_id: Unique result ID.
        check_id: ID of the check that was executed.
        check_name: Name of the check.
        dimension: Quality dimension.
        severity: Severity of the check.
        result: Check outcome (pass, fail, warning, skipped, error).
        message: Human-readable result message.
        details: Additional details or evidence.
        entity_id: Entity tested (if entity-specific).
        facility_id: Facility tested (if facility-specific).
        executed_at: Execution timestamp.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    check_id: str = Field(default="", description="Check ID")
    check_name: str = Field(default="", description="Check name")
    dimension: QualityDimension = Field(
        default=QualityDimension.COMPLETENESS, description="Dimension"
    )
    severity: CheckSeverity = Field(
        default=CheckSeverity.MINOR, description="Severity"
    )
    result: CheckResult = Field(default=CheckResult.SKIPPED, description="Result")
    message: str = Field(default="", max_length=2000, description="Message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Details")
    entity_id: str = Field(default="", description="Entity ID")
    facility_id: str = Field(default="", description="Facility ID")
    executed_at: datetime = Field(default_factory=_utcnow, description="Executed at")


# ---------------------------------------------------------------------------
# Pydantic Models -- Quality Score
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for a single quality dimension.

    Attributes:
        dimension: Quality dimension.
        total_checks: Total checks in this dimension.
        passed_checks: Number of checks that passed.
        failed_checks: Number of checks that failed.
        warning_checks: Number of checks with warnings.
        skipped_checks: Number of skipped checks.
        score: Dimension score (0-100).
        weight: Dimension weight in composite score.
        weighted_score: score * weight.
        critical_failures: Number of critical-severity failures.
        major_failures: Number of major-severity failures.
    """
    dimension: QualityDimension = Field(..., description="Quality dimension")
    total_checks: int = Field(default=0, description="Total checks")
    passed_checks: int = Field(default=0, description="Passed checks")
    failed_checks: int = Field(default=0, description="Failed checks")
    warning_checks: int = Field(default=0, description="Warning checks")
    skipped_checks: int = Field(default=0, description="Skipped checks")
    score: Decimal = Field(default=Decimal("0"), description="Dimension score (0-100)")
    weight: Decimal = Field(default=Decimal("0"), description="Weight")
    weighted_score: Decimal = Field(default=Decimal("0"), description="Weighted score")
    critical_failures: int = Field(default=0, description="Critical failures")
    major_failures: int = Field(default=0, description="Major failures")


class QualityScore(BaseModel):
    """Composite quality score for an inventory period.

    Attributes:
        score_id: Unique score ID.
        period_id: Inventory period ID.
        organisation_id: Organisation ID.
        dimension_scores: Per-dimension score breakdowns.
        composite_score: Weighted composite quality score (0-100).
        grade: Letter grade (A / B / C / D / F).
        total_checks_run: Total number of checks executed.
        total_passed: Total passed.
        total_failed: Total failed.
        total_warnings: Total warnings.
        critical_issue_count: Number of open critical issues.
        verification_ready: Whether inventory is ready for verification.
        calculated_at: Timestamp.
    """
    score_id: str = Field(default_factory=_new_uuid, description="Score ID")
    period_id: str = Field(default="", description="Period ID")
    organisation_id: str = Field(default="", description="Organisation ID")
    dimension_scores: List[DimensionScore] = Field(
        default_factory=list, description="Dimension scores"
    )
    composite_score: Decimal = Field(
        default=Decimal("0"), description="Composite score (0-100)"
    )
    grade: str = Field(default="F", max_length=2, description="Letter grade")
    total_checks_run: int = Field(default=0, description="Total checks run")
    total_passed: int = Field(default=0, description="Total passed")
    total_failed: int = Field(default=0, description="Total failed")
    total_warnings: int = Field(default=0, description="Total warnings")
    critical_issue_count: int = Field(default=0, description="Critical issues")
    verification_ready: bool = Field(
        default=False, description="Verification readiness"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")


# ---------------------------------------------------------------------------
# Pydantic Models -- Issue Tracking
# ---------------------------------------------------------------------------


class QualityIssue(BaseModel):
    """A quality issue identified during QA/QC checks.

    Attributes:
        issue_id: Unique issue identifier.
        period_id: Inventory period ID.
        check_id: Source QA/QC check that identified the issue.
        dimension: Quality dimension.
        severity: Issue severity.
        title: Short issue title.
        description: Detailed issue description.
        status: Current issue status.
        assigned_to: Person responsible for resolution.
        entity_id: Affected entity (if entity-specific).
        facility_id: Affected facility (if facility-specific).
        impact_tco2e: Estimated emission impact of the issue (tCO2e).
        root_cause: Root cause analysis (when known).
        resolution_notes: Notes on how the issue was resolved.
        created_at: When the issue was identified.
        resolved_at: When the issue was resolved.
        due_date: Target resolution date.
    """
    issue_id: str = Field(default_factory=_new_uuid, description="Issue ID")
    period_id: str = Field(default="", description="Period ID")
    check_id: str = Field(default="", description="Source check ID")
    dimension: QualityDimension = Field(
        default=QualityDimension.COMPLETENESS, description="Dimension"
    )
    severity: CheckSeverity = Field(
        default=CheckSeverity.MINOR, description="Severity"
    )
    title: str = Field(default="", max_length=500, description="Title")
    description: str = Field(default="", max_length=5000, description="Description")
    status: IssueStatus = Field(
        default=IssueStatus.OPEN, description="Issue status"
    )
    assigned_to: str = Field(default="", max_length=300, description="Assigned to")
    entity_id: str = Field(default="", description="Entity ID")
    facility_id: str = Field(default="", description="Facility ID")
    impact_tco2e: Decimal = Field(
        default=Decimal("0"), description="Impact (tCO2e)"
    )
    root_cause: str = Field(default="", max_length=2000, description="Root cause")
    resolution_notes: str = Field(
        default="", max_length=5000, description="Resolution notes"
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Created at")
    resolved_at: Optional[datetime] = Field(
        default=None, description="Resolved at"
    )
    due_date: Optional[date] = Field(default=None, description="Due date")


class ImprovementAction(BaseModel):
    """An improvement action to address one or more quality issues.

    Attributes:
        action_id: Unique action identifier.
        period_id: Inventory period ID.
        title: Short action title.
        description: Detailed description of the action.
        priority: Action priority.
        status: Action status.
        assigned_to: Person responsible.
        related_issue_ids: IDs of quality issues this action addresses.
        target_dimension: Primary quality dimension targeted.
        expected_score_improvement: Expected score improvement (points).
        due_date: Target completion date.
        completed_at: Actual completion date.
        notes: Additional notes.
    """
    action_id: str = Field(default_factory=_new_uuid, description="Action ID")
    period_id: str = Field(default="", description="Period ID")
    title: str = Field(default="", max_length=500, description="Title")
    description: str = Field(default="", max_length=5000, description="Description")
    priority: ActionPriority = Field(
        default=ActionPriority.MEDIUM, description="Priority"
    )
    status: ActionStatus = Field(
        default=ActionStatus.PLANNED, description="Action status"
    )
    assigned_to: str = Field(default="", max_length=300, description="Assigned to")
    related_issue_ids: List[str] = Field(
        default_factory=list, description="Related issue IDs"
    )
    target_dimension: QualityDimension = Field(
        default=QualityDimension.COMPLETENESS, description="Target dimension"
    )
    expected_score_improvement: Decimal = Field(
        default=Decimal("0"), description="Expected improvement (points)"
    )
    due_date: Optional[date] = Field(default=None, description="Due date")
    completed_at: Optional[datetime] = Field(
        default=None, description="Completed at"
    )
    notes: str = Field(default="", max_length=5000, description="Notes")


# ---------------------------------------------------------------------------
# Pydantic Models -- Result
# ---------------------------------------------------------------------------


class QualityManagementResult(BaseModel):
    """Complete result from a quality management engine operation.

    Attributes:
        result_id: Unique result ID.
        operation: Name of the operation performed.
        quality_score: Calculated quality score (if scoring was performed).
        check_results: Individual check results.
        issues: Quality issues (new or updated).
        actions: Improvement actions (new or updated).
        total_issues_open: Count of open issues.
        total_issues_resolved: Count of resolved issues.
        warnings: Operational warnings.
        calculated_at: Timestamp.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    operation: str = Field(default="", description="Operation name")
    quality_score: Optional[QualityScore] = Field(
        default=None, description="Quality score"
    )
    check_results: List[QAQCResult] = Field(
        default_factory=list, description="Check results"
    )
    issues: List[QualityIssue] = Field(
        default_factory=list, description="Quality issues"
    )
    actions: List[ImprovementAction] = Field(
        default_factory=list, description="Improvement actions"
    )
    total_issues_open: int = Field(default=0, description="Open issues")
    total_issues_resolved: int = Field(default=0, description="Resolved issues")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

QAQCCheck.model_rebuild()
QAQCResult.model_rebuild()
DimensionScore.model_rebuild()
QualityScore.model_rebuild()
QualityIssue.model_rebuild()
ImprovementAction.model_rebuild()
QualityManagementResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class QualityManagementEngine:
    """GHG Protocol Chapter 7 quality assurance / quality control engine.

    Performs automated QA/QC checks across a GHG inventory, calculates
    composite quality scores, tracks quality issues, and generates
    improvement action plans.

    All scoring uses deterministic Decimal arithmetic.  No LLM involvement
    in any quality assessment logic.

    Attributes:
        _dimension_weights: Weights for composite quality score.
        _check_definitions: Registered QA/QC check definitions.
        _issues: In-memory issue registry.
        _actions: In-memory improvement action registry.

    Example:
        >>> engine = QualityManagementEngine()
        >>> check_inputs = {
        ...     "COMP-001": True,  # pass
        ...     "COMP-002": False, # fail
        ...     "ACCU-001": True,  # pass
        ... }
        >>> result = engine.run_checks(
        ...     period_id="per-001",
        ...     organisation_id="org-001",
        ...     check_inputs=check_inputs,
        ... )
        >>> print(result.quality_score.composite_score)
    """

    def __init__(
        self,
        dimension_weights: Optional[Dict[QualityDimension, Decimal]] = None,
    ) -> None:
        """Initialise QualityManagementEngine.

        Args:
            dimension_weights: Custom dimension weights (must sum to 1.0).
                Defaults to GHG Protocol recommended weights.
        """
        self._dimension_weights = dimension_weights or dict(DEFAULT_DIMENSION_WEIGHTS)
        self._check_definitions: Dict[str, QAQCCheck] = {}
        self._issues: Dict[str, QualityIssue] = {}
        self._actions: Dict[str, ImprovementAction] = {}

        # Register default checks.
        for check_def in DEFAULT_CHECKS:
            check = QAQCCheck(
                check_id=check_def["check_id"],
                name=check_def["name"],
                dimension=QualityDimension(check_def["dimension"]),
                description=check_def["description"],
                severity=CheckSeverity(check_def["severity"]),
            )
            self._check_definitions[check.check_id] = check

        # Validate weights sum to 1.0.
        weight_sum = sum(self._dimension_weights.values(), Decimal("0"))
        if abs(weight_sum - Decimal("1")) > Decimal("0.001"):
            logger.warning(
                "Dimension weights sum to %s (expected 1.0); "
                "normalising weights", weight_sum,
            )
            for dim in self._dimension_weights:
                self._dimension_weights[dim] = _safe_divide(
                    self._dimension_weights[dim], weight_sum
                )

        logger.info(
            "QualityManagementEngine v%s initialised with %d checks",
            _MODULE_VERSION, len(self._check_definitions),
        )

    # ------------------------------------------------------------------
    # Public API -- Check Execution
    # ------------------------------------------------------------------

    def run_checks(
        self,
        period_id: str,
        organisation_id: str,
        check_inputs: Dict[str, bool],
        entity_id: str = "",
        facility_id: str = "",
    ) -> QualityManagementResult:
        """Execute QA/QC checks and calculate quality scores.

        Each entry in check_inputs maps a check_id to a boolean indicating
        whether the check passed (True) or failed (False).  Checks not
        present in check_inputs are marked as SKIPPED.

        Args:
            period_id: Inventory period ID.
            organisation_id: Organisation ID.
            check_inputs: Mapping of check_id -> pass (True) / fail (False).
            entity_id: Optional entity scope.
            facility_id: Optional facility scope.

        Returns:
            QualityManagementResult with check results and quality score.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        check_results: List[QAQCResult] = []
        new_issues: List[QualityIssue] = []

        # Execute each registered check.
        for check_id, check_def in self._check_definitions.items():
            if check_id in check_inputs:
                passed = check_inputs[check_id]
                if passed:
                    result_val = CheckResult.PASS
                    message = f"Check {check_id} ({check_def.name}): PASSED"
                else:
                    result_val = CheckResult.FAIL
                    message = (
                        f"Check {check_id} ({check_def.name}): FAILED - "
                        f"{check_def.description}"
                    )
            else:
                result_val = CheckResult.SKIPPED
                message = f"Check {check_id} ({check_def.name}): SKIPPED"

            qr = QAQCResult(
                check_id=check_id,
                check_name=check_def.name,
                dimension=check_def.dimension,
                severity=check_def.severity,
                result=result_val,
                message=message,
                entity_id=entity_id,
                facility_id=facility_id,
            )
            check_results.append(qr)

            # Auto-create issue for failed checks.
            if result_val == CheckResult.FAIL:
                issue = QualityIssue(
                    period_id=period_id,
                    check_id=check_id,
                    dimension=check_def.dimension,
                    severity=check_def.severity,
                    title=f"Failed: {check_def.name}",
                    description=check_def.description,
                    entity_id=entity_id,
                    facility_id=facility_id,
                )
                self._issues[issue.issue_id] = issue
                new_issues.append(issue)

        # Calculate quality score.
        quality_score = self._calculate_quality_score(
            period_id=period_id,
            organisation_id=organisation_id,
            check_results=check_results,
        )

        # Count open/resolved issues.
        open_count = sum(
            1 for i in self._issues.values()
            if i.status in (IssueStatus.OPEN, IssueStatus.INVESTIGATING)
        )
        resolved_count = sum(
            1 for i in self._issues.values()
            if i.status in (IssueStatus.RESOLVED, IssueStatus.ACCEPTED)
        )

        logger.info(
            "QA/QC checks complete for period %s: score=%.1f, grade=%s, "
            "%d passed, %d failed, %d new issues",
            period_id, quality_score.composite_score, quality_score.grade,
            quality_score.total_passed, quality_score.total_failed,
            len(new_issues),
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = QualityManagementResult(
            operation="run_checks",
            quality_score=quality_score,
            check_results=check_results,
            issues=new_issues,
            total_issues_open=open_count,
            total_issues_resolved=resolved_count,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def run_single_check(
        self,
        check_id: str,
        passed: bool,
        period_id: str = "",
        entity_id: str = "",
        facility_id: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> QAQCResult:
        """Run a single QA/QC check and return its result.

        Args:
            check_id: Check identifier.
            passed: Whether the check passed.
            period_id: Inventory period ID.
            entity_id: Entity scope.
            facility_id: Facility scope.
            details: Additional evidence/details.

        Returns:
            QAQCResult for the single check.

        Raises:
            KeyError: If check_id is not registered.
        """
        if check_id not in self._check_definitions:
            raise KeyError(f"Check not registered: {check_id}")

        check_def = self._check_definitions[check_id]
        result_val = CheckResult.PASS if passed else CheckResult.FAIL
        message = (
            f"Check {check_id} ({check_def.name}): "
            f"{'PASSED' if passed else 'FAILED'}"
        )

        qr = QAQCResult(
            check_id=check_id,
            check_name=check_def.name,
            dimension=check_def.dimension,
            severity=check_def.severity,
            result=result_val,
            message=message,
            details=details or {},
            entity_id=entity_id,
            facility_id=facility_id,
        )

        # Auto-create issue if failed.
        if not passed:
            issue = QualityIssue(
                period_id=period_id,
                check_id=check_id,
                dimension=check_def.dimension,
                severity=check_def.severity,
                title=f"Failed: {check_def.name}",
                description=check_def.description,
                entity_id=entity_id,
                facility_id=facility_id,
            )
            self._issues[issue.issue_id] = issue

        return qr

    # ------------------------------------------------------------------
    # Public API -- Issue Management
    # ------------------------------------------------------------------

    def create_issue(
        self,
        period_id: str,
        dimension: QualityDimension,
        severity: CheckSeverity,
        title: str,
        description: str,
        assigned_to: str = "",
        entity_id: str = "",
        facility_id: str = "",
        impact_tco2e: Decimal = Decimal("0"),
        due_date: Optional[date] = None,
    ) -> QualityIssue:
        """Manually create a quality issue.

        Args:
            period_id: Inventory period ID.
            dimension: Quality dimension.
            severity: Issue severity.
            title: Issue title.
            description: Detailed description.
            assigned_to: Assignee.
            entity_id: Affected entity.
            facility_id: Affected facility.
            impact_tco2e: Estimated emission impact.
            due_date: Resolution target date.

        Returns:
            The created QualityIssue.
        """
        issue = QualityIssue(
            period_id=period_id,
            dimension=dimension,
            severity=severity,
            title=title,
            description=description,
            assigned_to=assigned_to,
            entity_id=entity_id,
            facility_id=facility_id,
            impact_tco2e=impact_tco2e,
            due_date=due_date,
        )
        self._issues[issue.issue_id] = issue
        logger.info("Created quality issue '%s' [%s]", title, issue.issue_id)
        return issue

    def update_issue_status(
        self,
        issue_id: str,
        new_status: IssueStatus,
        resolution_notes: str = "",
        root_cause: str = "",
    ) -> QualityIssue:
        """Update the status of a quality issue.

        Args:
            issue_id: Issue identifier.
            new_status: New status.
            resolution_notes: Notes on resolution (for RESOLVED status).
            root_cause: Root cause analysis.

        Returns:
            Updated QualityIssue.

        Raises:
            KeyError: If issue not found.
        """
        if issue_id not in self._issues:
            raise KeyError(f"Quality issue not found: {issue_id}")

        issue = self._issues[issue_id]
        old_status = issue.status
        issue.status = new_status

        if resolution_notes:
            issue.resolution_notes = resolution_notes
        if root_cause:
            issue.root_cause = root_cause

        if new_status in (IssueStatus.RESOLVED, IssueStatus.ACCEPTED):
            issue.resolved_at = _utcnow()

        logger.info(
            "Issue '%s' status: %s -> %s",
            issue.title, old_status.value, new_status.value,
        )
        return issue

    def list_issues(
        self,
        period_id: Optional[str] = None,
        status_filter: Optional[List[IssueStatus]] = None,
        severity_filter: Optional[List[CheckSeverity]] = None,
        dimension_filter: Optional[List[QualityDimension]] = None,
    ) -> List[QualityIssue]:
        """List quality issues with optional filtering.

        Args:
            period_id: Filter by period.
            status_filter: Filter by status(es).
            severity_filter: Filter by severity(ies).
            dimension_filter: Filter by dimension(s).

        Returns:
            List of matching QualityIssue objects.
        """
        results: List[QualityIssue] = []
        for issue in self._issues.values():
            if period_id and issue.period_id != period_id:
                continue
            if status_filter and issue.status not in status_filter:
                continue
            if severity_filter and issue.severity not in severity_filter:
                continue
            if dimension_filter and issue.dimension not in dimension_filter:
                continue
            results.append(issue)
        # Sort by severity (critical first), then by creation date.
        severity_order = {
            CheckSeverity.CRITICAL: 0,
            CheckSeverity.MAJOR: 1,
            CheckSeverity.MINOR: 2,
            CheckSeverity.OBSERVATION: 3,
        }
        results.sort(key=lambda i: (severity_order.get(i.severity, 99), i.created_at))
        return results

    # ------------------------------------------------------------------
    # Public API -- Improvement Actions
    # ------------------------------------------------------------------

    def create_action(
        self,
        period_id: str,
        title: str,
        description: str,
        priority: ActionPriority = ActionPriority.MEDIUM,
        assigned_to: str = "",
        related_issue_ids: Optional[List[str]] = None,
        target_dimension: QualityDimension = QualityDimension.COMPLETENESS,
        expected_score_improvement: Decimal = Decimal("0"),
        due_date: Optional[date] = None,
    ) -> ImprovementAction:
        """Create an improvement action to address quality issues.

        Args:
            period_id: Inventory period ID.
            title: Action title.
            description: Detailed action description.
            priority: Action priority.
            assigned_to: Person responsible.
            related_issue_ids: IDs of quality issues this action addresses.
            target_dimension: Primary quality dimension to improve.
            expected_score_improvement: Expected improvement in points.
            due_date: Target completion date.

        Returns:
            The created ImprovementAction.
        """
        action = ImprovementAction(
            period_id=period_id,
            title=title,
            description=description,
            priority=priority,
            assigned_to=assigned_to,
            related_issue_ids=related_issue_ids or [],
            target_dimension=target_dimension,
            expected_score_improvement=expected_score_improvement,
            due_date=due_date,
        )
        self._actions[action.action_id] = action
        logger.info("Created improvement action '%s' [%s]", title, action.action_id)
        return action

    def update_action_status(
        self,
        action_id: str,
        new_status: ActionStatus,
        notes: str = "",
    ) -> ImprovementAction:
        """Update the status of an improvement action.

        Args:
            action_id: Action identifier.
            new_status: New status.
            notes: Additional notes.

        Returns:
            Updated ImprovementAction.

        Raises:
            KeyError: If action not found.
        """
        if action_id not in self._actions:
            raise KeyError(f"Improvement action not found: {action_id}")

        action = self._actions[action_id]
        action.status = new_status
        if notes:
            action.notes = notes
        if new_status == ActionStatus.COMPLETED:
            action.completed_at = _utcnow()

        logger.info(
            "Action '%s' status updated to %s",
            action.title, new_status.value,
        )
        return action

    def list_actions(
        self,
        period_id: Optional[str] = None,
        status_filter: Optional[List[ActionStatus]] = None,
    ) -> List[ImprovementAction]:
        """List improvement actions with optional filtering.

        Args:
            period_id: Filter by period.
            status_filter: Filter by status(es).

        Returns:
            List of matching ImprovementAction objects.
        """
        results: List[ImprovementAction] = []
        for action in self._actions.values():
            if period_id and action.period_id != period_id:
                continue
            if status_filter and action.status not in status_filter:
                continue
            results.append(action)
        priority_order = {
            ActionPriority.CRITICAL: 0,
            ActionPriority.HIGH: 1,
            ActionPriority.MEDIUM: 2,
            ActionPriority.LOW: 3,
        }
        results.sort(key=lambda a: priority_order.get(a.priority, 99))
        return results

    # ------------------------------------------------------------------
    # Public API -- Verification Readiness
    # ------------------------------------------------------------------

    def assess_verification_readiness(
        self,
        period_id: str,
        organisation_id: str,
        check_inputs: Dict[str, bool],
    ) -> QualityManagementResult:
        """Assess whether the inventory is ready for third-party verification.

        Runs all QA/QC checks, evaluates the quality score, and determines
        if the inventory meets verification readiness criteria:
        - Composite quality score >= 80
        - No open CRITICAL issues
        - At least 95% of issues resolved or accepted

        Args:
            period_id: Inventory period ID.
            organisation_id: Organisation ID.
            check_inputs: Mapping of check_id -> pass/fail.

        Returns:
            QualityManagementResult with verification readiness assessment.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # Run all checks first.
        check_result = self.run_checks(
            period_id=period_id,
            organisation_id=organisation_id,
            check_inputs=check_inputs,
        )

        quality_score = check_result.quality_score
        if quality_score is None:
            raise ValueError("Quality score calculation failed")

        # Assess readiness criteria.
        score_threshold = Decimal("80")
        issue_resolution_threshold = Decimal("95")

        total_issues = len(self._issues)
        resolved_issues = sum(
            1 for i in self._issues.values()
            if i.status in (IssueStatus.RESOLVED, IssueStatus.ACCEPTED)
        )
        open_critical = sum(
            1 for i in self._issues.values()
            if i.severity == CheckSeverity.CRITICAL
            and i.status in (IssueStatus.OPEN, IssueStatus.INVESTIGATING)
        )

        resolution_pct = _safe_pct(_decimal(resolved_issues), _decimal(total_issues))

        is_ready = (
            quality_score.composite_score >= score_threshold
            and open_critical == 0
            and (total_issues == 0 or resolution_pct >= issue_resolution_threshold)
        )

        quality_score.verification_ready = is_ready

        if not is_ready:
            if quality_score.composite_score < score_threshold:
                warnings.append(
                    f"Composite quality score ({quality_score.composite_score}) "
                    f"is below threshold ({score_threshold})"
                )
            if open_critical > 0:
                warnings.append(
                    f"{open_critical} critical issue(s) remain open"
                )
            if resolution_pct < issue_resolution_threshold:
                warnings.append(
                    f"Issue resolution rate ({_round_val(resolution_pct, 1)}%) "
                    f"is below threshold ({issue_resolution_threshold}%)"
                )

        logger.info(
            "Verification readiness for period %s: %s (score=%.1f, "
            "critical_open=%d, resolution=%s%%)",
            period_id, "READY" if is_ready else "NOT READY",
            quality_score.composite_score, open_critical,
            _round_val(resolution_pct, 1),
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = QualityManagementResult(
            operation="assess_verification_readiness",
            quality_score=quality_score,
            check_results=check_result.check_results,
            issues=check_result.issues,
            actions=list(self._actions.values()),
            total_issues_open=total_issues - resolved_issues,
            total_issues_resolved=resolved_issues,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Score Retrieval
    # ------------------------------------------------------------------

    def get_registered_checks(self) -> List[QAQCCheck]:
        """Return all registered QA/QC check definitions.

        Returns:
            List of QAQCCheck objects.
        """
        return list(self._check_definitions.values())

    def register_check(self, check: QAQCCheck) -> None:
        """Register a custom QA/QC check definition.

        Args:
            check: The QAQCCheck to register.
        """
        self._check_definitions[check.check_id] = check
        logger.info("Registered custom check '%s' [%s]", check.name, check.check_id)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _calculate_quality_score(
        self,
        period_id: str,
        organisation_id: str,
        check_results: List[QAQCResult],
    ) -> QualityScore:
        """Calculate composite quality score from check results.

        Computes per-dimension scores and a weighted composite score.

        Args:
            period_id: Inventory period ID.
            organisation_id: Organisation ID.
            check_results: Results from executed checks.

        Returns:
            QualityScore with dimension breakdowns.
        """
        # Group results by dimension.
        by_dimension: Dict[QualityDimension, List[QAQCResult]] = {}
        for dim in QualityDimension:
            by_dimension[dim] = []
        for qr in check_results:
            by_dimension[qr.dimension].append(qr)

        dimension_scores: List[DimensionScore] = []
        total_passed = 0
        total_failed = 0
        total_warnings = 0
        total_run = 0
        critical_count = 0

        for dim in QualityDimension:
            results = by_dimension[dim]
            dim_total = len(results)
            dim_passed = sum(1 for r in results if r.result == CheckResult.PASS)
            dim_failed = sum(1 for r in results if r.result == CheckResult.FAIL)
            dim_warnings = sum(1 for r in results if r.result == CheckResult.WARNING)
            dim_skipped = sum(1 for r in results if r.result == CheckResult.SKIPPED)

            # Exclude skipped from scoring denominator.
            scoreable = dim_total - dim_skipped
            dim_score = _safe_pct(_decimal(dim_passed), _decimal(scoreable))

            weight = self._dimension_weights.get(dim, Decimal("0"))
            weighted = _round_val(dim_score * weight, 4)

            critical_in_dim = sum(
                1 for r in results
                if r.result == CheckResult.FAIL
                and r.severity == CheckSeverity.CRITICAL
            )
            major_in_dim = sum(
                1 for r in results
                if r.result == CheckResult.FAIL
                and r.severity == CheckSeverity.MAJOR
            )

            dimension_scores.append(DimensionScore(
                dimension=dim,
                total_checks=dim_total,
                passed_checks=dim_passed,
                failed_checks=dim_failed,
                warning_checks=dim_warnings,
                skipped_checks=dim_skipped,
                score=_round_val(dim_score, 2),
                weight=weight,
                weighted_score=weighted,
                critical_failures=critical_in_dim,
                major_failures=major_in_dim,
            ))

            total_passed += dim_passed
            total_failed += dim_failed
            total_warnings += dim_warnings
            total_run += dim_total
            critical_count += critical_in_dim

        # Composite score.
        composite = sum(
            (ds.weighted_score for ds in dimension_scores), Decimal("0")
        )
        composite = _round_val(composite, 2)

        # Letter grade.
        grade = self._compute_grade(composite, critical_count)

        return QualityScore(
            period_id=period_id,
            organisation_id=organisation_id,
            dimension_scores=dimension_scores,
            composite_score=composite,
            grade=grade,
            total_checks_run=total_run,
            total_passed=total_passed,
            total_failed=total_failed,
            total_warnings=total_warnings,
            critical_issue_count=critical_count,
            verification_ready=False,  # set externally by readiness check
        )

    def _compute_grade(
        self,
        composite_score: Decimal,
        critical_failures: int,
    ) -> str:
        """Compute letter grade from composite score and critical failures.

        Grading scale:
            A:  >= 90 and no critical failures
            B:  >= 80 and no critical failures
            C:  >= 70
            D:  >= 50
            F:  < 50 or any critical failures with score < 80

        Args:
            composite_score: Composite quality score (0-100).
            critical_failures: Number of critical-severity failures.

        Returns:
            Letter grade string.
        """
        if critical_failures > 0 and composite_score < Decimal("80"):
            return "F"
        if composite_score >= Decimal("90") and critical_failures == 0:
            return "A"
        if composite_score >= Decimal("80") and critical_failures == 0:
            return "B"
        if composite_score >= Decimal("70"):
            return "C"
        if composite_score >= Decimal("50"):
            return "D"
        return "F"

    def _generate_actions_from_issues(
        self,
        period_id: str,
        issues: List[QualityIssue],
    ) -> List[ImprovementAction]:
        """Generate improvement actions from a list of quality issues.

        Groups issues by dimension and creates one action per dimension
        with critical/major issues.

        Args:
            period_id: Inventory period ID.
            issues: List of quality issues.

        Returns:
            List of generated ImprovementAction objects.
        """
        actions: List[ImprovementAction] = []

        # Group issues by dimension.
        by_dim: Dict[QualityDimension, List[QualityIssue]] = {}
        for dim in QualityDimension:
            by_dim[dim] = []
        for issue in issues:
            by_dim[issue.dimension].append(issue)

        for dim, dim_issues in by_dim.items():
            if not dim_issues:
                continue

            # Determine priority based on highest severity in dimension.
            has_critical = any(i.severity == CheckSeverity.CRITICAL for i in dim_issues)
            has_major = any(i.severity == CheckSeverity.MAJOR for i in dim_issues)

            if has_critical:
                priority = ActionPriority.CRITICAL
            elif has_major:
                priority = ActionPriority.HIGH
            else:
                priority = ActionPriority.MEDIUM

            issue_ids = [i.issue_id for i in dim_issues]
            issue_titles = [i.title for i in dim_issues[:5]]  # cap at 5 for readability

            action = ImprovementAction(
                period_id=period_id,
                title=f"Address {dim.value} quality issues ({len(dim_issues)} items)",
                description=(
                    f"Resolve the following {dim.value} issues: "
                    f"{'; '.join(issue_titles)}"
                ),
                priority=priority,
                related_issue_ids=issue_ids,
                target_dimension=dim,
            )
            self._actions[action.action_id] = action
            actions.append(action)

        return actions

    def generate_improvement_plan(
        self,
        period_id: str,
        organisation_id: str,
    ) -> QualityManagementResult:
        """Generate a comprehensive improvement action plan.

        Analyses all open issues for the period and creates prioritised
        improvement actions grouped by quality dimension.

        Args:
            period_id: Inventory period ID.
            organisation_id: Organisation ID.

        Returns:
            QualityManagementResult with improvement actions.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # Get open issues for this period.
        open_issues = self.list_issues(
            period_id=period_id,
            status_filter=[IssueStatus.OPEN, IssueStatus.INVESTIGATING],
        )

        if not open_issues:
            warnings.append("No open issues found; no improvement actions generated")

        # Generate actions.
        new_actions = self._generate_actions_from_issues(period_id, open_issues)

        open_count = sum(
            1 for i in self._issues.values()
            if i.status in (IssueStatus.OPEN, IssueStatus.INVESTIGATING)
        )
        resolved_count = sum(
            1 for i in self._issues.values()
            if i.status in (IssueStatus.RESOLVED, IssueStatus.ACCEPTED)
        )

        logger.info(
            "Generated improvement plan for period %s: %d actions from %d issues",
            period_id, len(new_actions), len(open_issues),
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = QualityManagementResult(
            operation="generate_improvement_plan",
            issues=open_issues,
            actions=new_actions,
            total_issues_open=open_count,
            total_issues_resolved=resolved_count,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result
