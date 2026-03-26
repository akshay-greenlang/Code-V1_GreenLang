# -*- coding: utf-8 -*-
"""
Annual Review Workflow
==========================

4-phase workflow for annual base year policy compliance review
within PACK-045 Base Year Management Pack.

Phases:
    1. PolicyReview            -- Review the organization's base year policy
                                  for currency, completeness, and alignment
                                  with current regulatory requirements.
    2. TriggerScan             -- Scan the reporting year for events that
                                  could constitute recalculation triggers,
                                  comparing against the trigger registry.
    3. ConsistencyCheck        -- Verify time series consistency between
                                  base year and current reporting, check
                                  methodology alignment, scope coverage.
    4. ReportGeneration        -- Generate the annual review report with
                                  compliance status, findings, and
                                  recommendations for the following year.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Standard Chapter 5 (Tracking Emissions Over Time)
    ISO 14064-1:2018 Clause 9 (Base year and recalculation)
    ESRS E1 (Annual disclosure requirements)

Schedule: Annually, after reporting period close
Estimated duration: 1-2 weeks

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ReviewPhase(str, Enum):
    """Annual review workflow phases."""

    POLICY_REVIEW = "policy_review"
    TRIGGER_SCAN = "trigger_scan"
    CONSISTENCY_CHECK = "consistency_check"
    REPORT_GENERATION = "report_generation"


class PolicyStatus(str, Enum):
    """Status of policy review findings."""

    CURRENT = "current"
    NEEDS_UPDATE = "needs_update"
    EXPIRED = "expired"
    NOT_DOCUMENTED = "not_documented"


class ConsistencyStatus(str, Enum):
    """Status of time series consistency checks."""

    CONSISTENT = "consistent"
    MINOR_INCONSISTENCY = "minor_inconsistency"
    MAJOR_INCONSISTENCY = "major_inconsistency"
    BROKEN = "broken"


class FindingSeverity(str, Enum):
    """Severity classification for review findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class TriggerScanStatus(str, Enum):
    """Outcome of trigger scanning."""

    NO_TRIGGERS = "no_triggers"
    TRIGGERS_FOUND = "triggers_found"
    REVIEW_RECOMMENDED = "review_recommended"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class BaseYearPolicy(BaseModel):
    """Organization's base year management policy."""

    policy_id: str = Field(default="")
    version: str = Field(default="v1.0")
    last_reviewed: str = Field(default="", description="ISO date of last review")
    significance_threshold_pct: float = Field(default=5.0, ge=0.1, le=50.0)
    mandatory_structural: bool = Field(default=True)
    mandatory_methodology: bool = Field(default=True)
    review_frequency_months: int = Field(default=12, ge=1, le=60)
    approved_by: str = Field(default="")
    scope3_included: bool = Field(default=False)


class YearData(BaseModel):
    """Data for the reporting year under review."""

    year: int = Field(..., ge=2010, le=2050)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    facilities_count: int = Field(default=0, ge=0)
    methodology_version: str = Field(default="")
    consolidation_approach: str = Field(default="operational_control")
    structural_changes: List[str] = Field(default_factory=list)
    methodology_changes: List[str] = Field(default_factory=list)
    data_corrections: List[str] = Field(default_factory=list)


class PolicyFinding(BaseModel):
    """Finding from policy review phase."""

    finding_id: str = Field(default_factory=lambda: f"fnd-{uuid.uuid4().hex[:8]}")
    severity: FindingSeverity = Field(default=FindingSeverity.MEDIUM)
    category: str = Field(default="")
    description: str = Field(default="")
    recommendation: str = Field(default="")
    regulatory_reference: str = Field(default="")


class ScanResult(BaseModel):
    """Result from trigger scanning."""

    scan_id: str = Field(default_factory=lambda: f"scn-{uuid.uuid4().hex[:8]}")
    trigger_type: str = Field(default="")
    description: str = Field(default="")
    impact_estimate_pct: float = Field(default=0.0)
    requires_assessment: bool = Field(default=False)
    source: str = Field(default="")


class ConsistencyCheckResult(BaseModel):
    """Result from a single consistency check."""

    check_id: str = Field(default_factory=lambda: f"chk-{uuid.uuid4().hex[:8]}")
    check_name: str = Field(default="")
    status: ConsistencyStatus = Field(default=ConsistencyStatus.CONSISTENT)
    description: str = Field(default="")
    base_year_value: float = Field(default=0.0)
    current_year_value: float = Field(default=0.0)
    variance_pct: float = Field(default=0.0)


class ReviewReport(BaseModel):
    """Generated annual review report."""

    report_id: str = Field(default_factory=lambda: f"rpt-{uuid.uuid4().hex[:8]}")
    title: str = Field(default="")
    generated_at: str = Field(default="")
    policy_compliant: bool = Field(default=True)
    summary: str = Field(default="")
    findings_count: int = Field(default=0, ge=0)
    recommendations_count: int = Field(default=0, ge=0)
    next_review_date: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class AnnualReviewInput(BaseModel):
    """Input data model for AnnualReviewWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organization identifier")
    base_year: int = Field(..., ge=2010, le=2050, description="Current base year")
    reporting_year: int = Field(..., ge=2010, le=2050, description="Year under review")
    policy: BaseYearPolicy = Field(
        default_factory=BaseYearPolicy,
        description="Current base year management policy",
    )
    year_data: YearData = Field(
        ..., description="Reporting year emissions and change data",
    )
    base_year_total_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_scope1_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_scope2_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_methodology: str = Field(default="")
    base_year_consolidation: str = Field(default="operational_control")
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class AnnualReviewResult(BaseModel):
    """Complete result from annual review workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="annual_review")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    base_year: int = Field(default=0)
    reporting_year: int = Field(default=0)
    policy_compliant: bool = Field(default=True)
    triggers_found: List[ScanResult] = Field(default_factory=list)
    consistency_status: ConsistencyStatus = Field(default=ConsistencyStatus.CONSISTENT)
    consistency_checks: List[ConsistencyCheckResult] = Field(default_factory=list)
    findings: List[PolicyFinding] = Field(default_factory=list)
    review_report: Optional[ReviewReport] = Field(default=None)
    provenance_hash: str = Field(default="")


# =============================================================================
# POLICY COMPLIANCE RULES (Zero-Hallucination)
# =============================================================================

POLICY_CHECKS: List[Dict[str, str]] = [
    {
        "name": "significance_threshold_defined",
        "requirement": "Policy must define a significance threshold",
        "reference": "GHG Protocol Ch.5",
    },
    {
        "name": "structural_change_policy",
        "requirement": "Policy must address structural changes (M&A, divestitures)",
        "reference": "GHG Protocol Ch.5 S.5.4",
    },
    {
        "name": "methodology_change_policy",
        "requirement": "Policy must address methodology changes",
        "reference": "GHG Protocol Ch.5 S.5.5",
    },
    {
        "name": "data_error_policy",
        "requirement": "Policy must address significant data error corrections",
        "reference": "GHG Protocol Ch.5 S.5.6",
    },
    {
        "name": "review_frequency",
        "requirement": "Policy must specify annual review requirement",
        "reference": "ISO 14064-1:2018 Clause 9",
    },
    {
        "name": "approval_authority",
        "requirement": "Policy must identify approval authority",
        "reference": "ISO 14064-1:2018 Clause 8",
    },
    {
        "name": "scope3_coverage",
        "requirement": "Policy should address Scope 3 base year if applicable",
        "reference": "GHG Protocol Scope 3 Standard Ch.5",
    },
]

# Year-over-year variance thresholds for consistency checks
CONSISTENCY_THRESHOLDS: Dict[str, float] = {
    "total_variance_warning_pct": 15.0,
    "total_variance_critical_pct": 30.0,
    "scope_variance_warning_pct": 20.0,
    "scope_variance_critical_pct": 40.0,
    "facility_count_change_warning_pct": 10.0,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class AnnualReviewWorkflow:
    """
    4-phase annual base year policy compliance review workflow.

    Reviews policy currency, scans for recalculation triggers, checks
    time series consistency, and generates the annual review report.

    Zero-hallucination: all variance calculations and compliance checks
    use deterministic formulas, no LLM calls in numeric paths.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _findings: Policy and consistency findings.
        _triggers: Detected potential triggers.
        _consistency_checks: Consistency check results.
        _policy_compliant: Overall compliance status.

    Example:
        >>> wf = AnnualReviewWorkflow()
        >>> year_data = YearData(year=2025, total_tco2e=48000.0)
        >>> inp = AnnualReviewInput(
        ...     organization_id="org-001", base_year=2022,
        ...     reporting_year=2025, year_data=year_data,
        ...     base_year_total_tco2e=50000.0,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[ReviewPhase] = [
        ReviewPhase.POLICY_REVIEW,
        ReviewPhase.TRIGGER_SCAN,
        ReviewPhase.CONSISTENCY_CHECK,
        ReviewPhase.REPORT_GENERATION,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize AnnualReviewWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._findings: List[PolicyFinding] = []
        self._triggers: List[ScanResult] = []
        self._consistency_checks: List[ConsistencyCheckResult] = []
        self._consistency_status: ConsistencyStatus = ConsistencyStatus.CONSISTENT
        self._policy_compliant: bool = True
        self._report: Optional[ReviewReport] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: AnnualReviewInput) -> AnnualReviewResult:
        """
        Execute the 4-phase annual review workflow.

        Args:
            input_data: Base year context, policy, and reporting year data.

        Returns:
            AnnualReviewResult with compliance status, findings, and report.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting annual review %s org=%s base_year=%d reporting_year=%d",
            self.workflow_id, input_data.organization_id,
            input_data.base_year, input_data.reporting_year,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_policy_review,
            self._phase_trigger_scan,
            self._phase_consistency_check,
            self._phase_report_generation,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._execute_with_retry(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Annual review failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = AnnualReviewResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            base_year=input_data.base_year,
            reporting_year=input_data.reporting_year,
            policy_compliant=self._policy_compliant,
            triggers_found=self._triggers,
            consistency_status=self._consistency_status,
            consistency_checks=self._consistency_checks,
            findings=self._findings,
            review_report=self._report,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Annual review %s completed in %.2fs status=%s compliant=%s",
            self.workflow_id, elapsed, overall_status.value, self._policy_compliant,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: AnnualReviewInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Policy Review
    # -------------------------------------------------------------------------

    async def _phase_policy_review(
        self, input_data: AnnualReviewInput,
    ) -> PhaseResult:
        """Review base year policy for currency and completeness."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        policy = input_data.policy
        policy_status = PolicyStatus.CURRENT

        # Check policy age
        if policy.last_reviewed:
            try:
                last_reviewed = datetime.fromisoformat(policy.last_reviewed)
                months_since = (datetime.utcnow() - last_reviewed).days / 30.0
                if months_since > policy.review_frequency_months * 1.5:
                    policy_status = PolicyStatus.EXPIRED
                    self._findings.append(PolicyFinding(
                        severity=FindingSeverity.HIGH,
                        category="policy_currency",
                        description=(
                            f"Policy last reviewed {months_since:.0f} months ago; "
                            f"exceeds {policy.review_frequency_months}-month review cycle"
                        ),
                        recommendation="Update and re-approve base year policy immediately",
                        regulatory_reference="ISO 14064-1:2018 Clause 9",
                    ))
                elif months_since > policy.review_frequency_months:
                    policy_status = PolicyStatus.NEEDS_UPDATE
                    self._findings.append(PolicyFinding(
                        severity=FindingSeverity.MEDIUM,
                        category="policy_currency",
                        description=f"Policy review overdue by {months_since - policy.review_frequency_months:.0f} months",
                        recommendation="Schedule policy review within 30 days",
                        regulatory_reference="ISO 14064-1:2018 Clause 9",
                    ))
            except (ValueError, TypeError):
                policy_status = PolicyStatus.NEEDS_UPDATE
                warnings.append("Could not parse policy last_reviewed date")
        else:
            policy_status = PolicyStatus.NOT_DOCUMENTED
            self._findings.append(PolicyFinding(
                severity=FindingSeverity.CRITICAL,
                category="policy_documentation",
                description="No last_reviewed date on base year policy",
                recommendation="Document policy review date and approval",
                regulatory_reference="GHG Protocol Ch.5",
            ))

        # Check mandatory policy elements
        for check in POLICY_CHECKS:
            passed = self._evaluate_policy_check(check["name"], policy)
            if not passed:
                self._findings.append(PolicyFinding(
                    severity=FindingSeverity.MEDIUM,
                    category="policy_completeness",
                    description=f"Missing: {check['requirement']}",
                    recommendation=f"Address per {check['reference']}",
                    regulatory_reference=check["reference"],
                ))

        # Determine overall compliance
        critical_findings = sum(
            1 for f in self._findings if f.severity == FindingSeverity.CRITICAL
        )
        high_findings = sum(
            1 for f in self._findings if f.severity == FindingSeverity.HIGH
        )
        if critical_findings > 0:
            self._policy_compliant = False

        outputs["policy_status"] = policy_status.value
        outputs["policy_version"] = policy.version
        outputs["findings_count"] = len(self._findings)
        outputs["critical_findings"] = critical_findings
        outputs["high_findings"] = high_findings
        outputs["policy_compliant"] = self._policy_compliant

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 PolicyReview: status=%s findings=%d compliant=%s",
            policy_status.value, len(self._findings), self._policy_compliant,
        )
        return PhaseResult(
            phase_name="policy_review", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _evaluate_policy_check(self, check_name: str, policy: BaseYearPolicy) -> bool:
        """Evaluate a single policy check."""
        checks: Dict[str, bool] = {
            "significance_threshold_defined": policy.significance_threshold_pct > 0,
            "structural_change_policy": policy.mandatory_structural is not None,
            "methodology_change_policy": policy.mandatory_methodology is not None,
            "data_error_policy": policy.significance_threshold_pct > 0,
            "review_frequency": policy.review_frequency_months > 0,
            "approval_authority": bool(policy.approved_by),
            "scope3_coverage": not policy.scope3_included or True,
        }
        return checks.get(check_name, True)

    # -------------------------------------------------------------------------
    # Phase 2: Trigger Scan
    # -------------------------------------------------------------------------

    async def _phase_trigger_scan(
        self, input_data: AnnualReviewInput,
    ) -> PhaseResult:
        """Scan reporting year for potential recalculation triggers."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._triggers = []
        year_data = input_data.year_data
        threshold = input_data.policy.significance_threshold_pct
        base_total = input_data.base_year_total_tco2e

        # Scan structural changes
        for change in year_data.structural_changes:
            self._triggers.append(ScanResult(
                trigger_type="structural",
                description=change,
                impact_estimate_pct=0.0,  # Requires detailed assessment
                requires_assessment=True,
                source="year_data_structural_changes",
            ))

        # Scan methodology changes
        for change in year_data.methodology_changes:
            self._triggers.append(ScanResult(
                trigger_type="methodological",
                description=change,
                impact_estimate_pct=0.0,
                requires_assessment=True,
                source="year_data_methodology_changes",
            ))

        # Scan data corrections
        for correction in year_data.data_corrections:
            self._triggers.append(ScanResult(
                trigger_type="data_error",
                description=correction,
                impact_estimate_pct=0.0,
                requires_assessment=True,
                source="year_data_corrections",
            ))

        # Check for methodology version mismatch
        if (
            input_data.base_year_methodology
            and year_data.methodology_version
            and input_data.base_year_methodology != year_data.methodology_version
        ):
            self._triggers.append(ScanResult(
                trigger_type="methodological",
                description=(
                    f"Methodology version changed: {input_data.base_year_methodology} -> "
                    f"{year_data.methodology_version}"
                ),
                impact_estimate_pct=0.0,
                requires_assessment=True,
                source="methodology_version_comparison",
            ))

        # Check for consolidation approach change
        if (
            input_data.base_year_consolidation
            and year_data.consolidation_approach
            and input_data.base_year_consolidation != year_data.consolidation_approach
        ):
            self._triggers.append(ScanResult(
                trigger_type="structural",
                description=(
                    f"Consolidation approach changed: "
                    f"{input_data.base_year_consolidation} -> "
                    f"{year_data.consolidation_approach}"
                ),
                impact_estimate_pct=0.0,
                requires_assessment=True,
                source="consolidation_comparison",
            ))

        # Check for significant total emissions variance
        if base_total > 0 and year_data.total_tco2e > 0:
            total_var_pct = abs(
                (year_data.total_tco2e - base_total) / base_total
            ) * 100.0
            if total_var_pct > CONSISTENCY_THRESHOLDS["total_variance_critical_pct"]:
                self._triggers.append(ScanResult(
                    trigger_type="variance",
                    description=(
                        f"Total emissions variance from base year: {total_var_pct:.1f}%"
                    ),
                    impact_estimate_pct=round(total_var_pct, 2),
                    requires_assessment=total_var_pct > threshold,
                    source="emissions_variance_check",
                ))

        scan_status = (
            TriggerScanStatus.TRIGGERS_FOUND
            if self._triggers
            else TriggerScanStatus.NO_TRIGGERS
        )
        requires_assessment = sum(1 for t in self._triggers if t.requires_assessment)

        outputs["scan_status"] = scan_status.value
        outputs["triggers_found"] = len(self._triggers)
        outputs["requires_assessment"] = requires_assessment
        outputs["trigger_types"] = list(set(t.trigger_type for t in self._triggers))

        if requires_assessment > 0:
            warnings.append(
                f"{requires_assessment} trigger(s) require formal assessment"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 TriggerScan: %d triggers found, %d require assessment",
            len(self._triggers), requires_assessment,
        )
        return PhaseResult(
            phase_name="trigger_scan", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Consistency Check
    # -------------------------------------------------------------------------

    async def _phase_consistency_check(
        self, input_data: AnnualReviewInput,
    ) -> PhaseResult:
        """Verify time series consistency between base year and current year."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._consistency_checks = []
        year_data = input_data.year_data
        self._consistency_status = ConsistencyStatus.CONSISTENT

        # Check 1: Total emissions variance
        total_var = self._check_variance(
            "total_emissions",
            input_data.base_year_total_tco2e,
            year_data.total_tco2e,
            CONSISTENCY_THRESHOLDS["total_variance_warning_pct"],
            CONSISTENCY_THRESHOLDS["total_variance_critical_pct"],
        )
        self._consistency_checks.append(total_var)

        # Check 2: Scope 1 consistency
        scope1_var = self._check_variance(
            "scope1_emissions",
            input_data.base_year_scope1_tco2e,
            year_data.scope1_tco2e,
            CONSISTENCY_THRESHOLDS["scope_variance_warning_pct"],
            CONSISTENCY_THRESHOLDS["scope_variance_critical_pct"],
        )
        self._consistency_checks.append(scope1_var)

        # Check 3: Scope 2 consistency
        scope2_var = self._check_variance(
            "scope2_emissions",
            input_data.base_year_scope2_tco2e,
            year_data.scope2_tco2e,
            CONSISTENCY_THRESHOLDS["scope_variance_warning_pct"],
            CONSISTENCY_THRESHOLDS["scope_variance_critical_pct"],
        )
        self._consistency_checks.append(scope2_var)

        # Check 4: Scope 3 consistency (if applicable)
        if input_data.base_year_scope3_tco2e > 0 or year_data.scope3_tco2e > 0:
            scope3_var = self._check_variance(
                "scope3_emissions",
                input_data.base_year_scope3_tco2e,
                year_data.scope3_tco2e,
                CONSISTENCY_THRESHOLDS["scope_variance_warning_pct"],
                CONSISTENCY_THRESHOLDS["scope_variance_critical_pct"],
            )
            self._consistency_checks.append(scope3_var)

        # Check 5: Methodology alignment
        method_consistent = (
            not input_data.base_year_methodology
            or not year_data.methodology_version
            or input_data.base_year_methodology == year_data.methodology_version
        )
        self._consistency_checks.append(ConsistencyCheckResult(
            check_name="methodology_alignment",
            status=(
                ConsistencyStatus.CONSISTENT
                if method_consistent
                else ConsistencyStatus.MINOR_INCONSISTENCY
            ),
            description=(
                "Methodology versions aligned"
                if method_consistent
                else f"Methodology changed: {input_data.base_year_methodology} -> {year_data.methodology_version}"
            ),
        ))

        # Check 6: Consolidation approach consistency
        consol_consistent = (
            not input_data.base_year_consolidation
            or not year_data.consolidation_approach
            or input_data.base_year_consolidation == year_data.consolidation_approach
        )
        self._consistency_checks.append(ConsistencyCheckResult(
            check_name="consolidation_approach",
            status=(
                ConsistencyStatus.CONSISTENT
                if consol_consistent
                else ConsistencyStatus.MAJOR_INCONSISTENCY
            ),
            description=(
                "Consolidation approach consistent"
                if consol_consistent
                else "Consolidation approach changed; base year recalculation may be required"
            ),
        ))

        # Determine overall consistency status
        statuses = [c.status for c in self._consistency_checks]
        if ConsistencyStatus.BROKEN in statuses:
            self._consistency_status = ConsistencyStatus.BROKEN
        elif ConsistencyStatus.MAJOR_INCONSISTENCY in statuses:
            self._consistency_status = ConsistencyStatus.MAJOR_INCONSISTENCY
        elif ConsistencyStatus.MINOR_INCONSISTENCY in statuses:
            self._consistency_status = ConsistencyStatus.MINOR_INCONSISTENCY

        if self._consistency_status in (
            ConsistencyStatus.BROKEN, ConsistencyStatus.MAJOR_INCONSISTENCY,
        ):
            self._policy_compliant = False
            self._findings.append(PolicyFinding(
                severity=FindingSeverity.HIGH,
                category="consistency",
                description=f"Time series consistency: {self._consistency_status.value}",
                recommendation="Investigate inconsistencies and determine if recalculation is needed",
                regulatory_reference="GHG Protocol Ch.5",
            ))

        outputs["checks_performed"] = len(self._consistency_checks)
        outputs["consistent"] = sum(
            1 for c in self._consistency_checks
            if c.status == ConsistencyStatus.CONSISTENT
        )
        outputs["inconsistent"] = sum(
            1 for c in self._consistency_checks
            if c.status != ConsistencyStatus.CONSISTENT
        )
        outputs["overall_status"] = self._consistency_status.value

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ConsistencyCheck: %d checks, overall=%s",
            len(self._consistency_checks), self._consistency_status.value,
        )
        return PhaseResult(
            phase_name="consistency_check", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _check_variance(
        self,
        name: str,
        base_value: float,
        current_value: float,
        warning_pct: float,
        critical_pct: float,
    ) -> ConsistencyCheckResult:
        """Check variance between base year and current year values."""
        if base_value <= 0:
            return ConsistencyCheckResult(
                check_name=name,
                status=ConsistencyStatus.CONSISTENT,
                description=f"No base year value for {name}",
                base_year_value=base_value,
                current_year_value=current_value,
                variance_pct=0.0,
            )

        variance_pct = ((current_value - base_value) / base_value) * 100.0
        abs_var = abs(variance_pct)

        if abs_var >= critical_pct:
            status = ConsistencyStatus.MAJOR_INCONSISTENCY
        elif abs_var >= warning_pct:
            status = ConsistencyStatus.MINOR_INCONSISTENCY
        else:
            status = ConsistencyStatus.CONSISTENT

        return ConsistencyCheckResult(
            check_name=name,
            status=status,
            description=f"{name}: {variance_pct:+.2f}% variance from base year",
            base_year_value=round(base_value, 4),
            current_year_value=round(current_value, 4),
            variance_pct=round(variance_pct, 4),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: AnnualReviewInput,
    ) -> PhaseResult:
        """Generate the annual review report."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        now_iso = datetime.utcnow().isoformat()

        # Calculate next review date
        review_months = input_data.policy.review_frequency_months
        next_year = input_data.reporting_year + 1
        next_review = f"{next_year}-01-15"

        # Build summary
        trigger_summary = (
            f"{len(self._triggers)} trigger(s) detected"
            if self._triggers
            else "No triggers detected"
        )
        consistency_summary = f"Time series consistency: {self._consistency_status.value}"
        findings_summary = (
            f"{len(self._findings)} finding(s): "
            f"{sum(1 for f in self._findings if f.severity == FindingSeverity.CRITICAL)} critical, "
            f"{sum(1 for f in self._findings if f.severity == FindingSeverity.HIGH)} high, "
            f"{sum(1 for f in self._findings if f.severity == FindingSeverity.MEDIUM)} medium"
        )

        report_summary = (
            f"Annual Base Year Review for {input_data.organization_id}, "
            f"reporting year {input_data.reporting_year}. "
            f"Base year: {input_data.base_year}. "
            f"Policy compliant: {self._policy_compliant}. "
            f"{trigger_summary}. {consistency_summary}. {findings_summary}."
        )

        report_hash = hashlib.sha256(report_summary.encode("utf-8")).hexdigest()

        self._report = ReviewReport(
            title=f"Annual Base Year Review - {input_data.reporting_year}",
            generated_at=now_iso,
            policy_compliant=self._policy_compliant,
            summary=report_summary,
            findings_count=len(self._findings),
            recommendations_count=sum(
                1 for f in self._findings if f.recommendation
            ),
            next_review_date=next_review,
            provenance_hash=report_hash,
        )

        outputs["report_generated"] = True
        outputs["policy_compliant"] = self._policy_compliant
        outputs["findings_count"] = len(self._findings)
        outputs["triggers_count"] = len(self._triggers)
        outputs["consistency_status"] = self._consistency_status.value
        outputs["next_review_date"] = next_review

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ReportGeneration: compliant=%s findings=%d next_review=%s",
            self._policy_compliant, len(self._findings), next_review,
        )
        return PhaseResult(
            phase_name="report_generation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._findings = []
        self._triggers = []
        self._consistency_checks = []
        self._consistency_status = ConsistencyStatus.CONSISTENT
        self._policy_compliant = True
        self._report = None

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: AnnualReviewResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.organization_id}|{result.reporting_year}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
