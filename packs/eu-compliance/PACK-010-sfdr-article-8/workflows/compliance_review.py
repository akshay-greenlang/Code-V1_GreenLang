# -*- coding: utf-8 -*-
"""
Compliance Review Workflow
=============================

Four-phase workflow for reviewing SFDR Article 8 disclosure completeness
and compliance. Orchestrates disclosure completeness checking, data quality
assessment, commitment adherence verification, and action item generation
into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 8 products must maintain three active disclosures:
      * Annex II: Pre-contractual disclosure
      * Annex III: Website disclosure
      * Annex IV: Periodic reporting
    - All disclosures must be current, consistent, and published.
    - PAI data coverage must meet minimum quality standards.
    - Binding elements (exclusions, minimums) must be continuously satisfied.
    - NCAs may request evidence at any time; readiness is essential.

    Compliance Review Scope:
    - Publication timeliness and freshness of all three annexes
    - Internal consistency across disclosures
    - Data quality thresholds for PAI indicators
    - Binding element adherence (exclusions active, minimums maintained)
    - Portfolio screening currency
    - Taxonomy alignment vs commitment

Phases:
    1. DisclosureCompleteness - Check all mandatory disclosures are current,
       verify publication dates, identify missing/outdated content
    2. DataQuality - Assess PAI data coverage ratios, estimation rates,
       data age, identify quality improvements needed
    3. CommitmentAdherence - Check all binding elements are satisfied,
       verify minimum proportions maintained, flag breaches
    4. ActionItems - Generate prioritized remediation tasks, compliance
       calendar updates, escalation items

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# UTILITIES
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class IssueSeverity(str, Enum):
    """Compliance issue severity."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class DisclosureType(str, Enum):
    """Type of SFDR disclosure."""
    ANNEX_II = "ANNEX_II"
    ANNEX_III = "ANNEX_III"
    ANNEX_IV = "ANNEX_IV"
    PAI_STATEMENT = "PAI_STATEMENT"

class ActionPriority(str, Enum):
    """Action item priority."""
    URGENT = "URGENT"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

# =============================================================================
# DATA MODELS - SHARED
# =============================================================================

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)

class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

# =============================================================================
# DATA MODELS - COMPLIANCE REVIEW
# =============================================================================

class DisclosureStatus(BaseModel):
    """Status of a specific disclosure document."""
    disclosure_type: DisclosureType = Field(...)
    exists: bool = Field(default=False)
    last_updated: Optional[str] = Field(None, description="YYYY-MM-DD")
    version: Optional[str] = Field(None)
    published: bool = Field(default=False)
    publication_url: Optional[str] = Field(None)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)

class BindingElement(BaseModel):
    """A binding element commitment to verify."""
    element_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Binding element name")
    element_type: str = Field(
        default="exclusion",
        description="Type: exclusion, minimum_proportion, threshold"
    )
    committed_value: float = Field(default=0.0)
    actual_value: Optional[float] = Field(None)
    unit: str = Field(default="%")
    is_satisfied: Optional[bool] = Field(None)

class PAIDataQuality(BaseModel):
    """PAI data quality metrics."""
    indicator_id: str = Field(...)
    indicator_name: str = Field(default="")
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    estimation_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_age_months: Optional[int] = Field(None, ge=0)
    data_source: str = Field(default="")

class ComplianceReviewInput(BaseModel):
    """Input configuration for the compliance review workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    review_date: str = Field(..., description="Review date YYYY-MM-DD")
    disclosure_statuses: List[DisclosureStatus] = Field(
        default_factory=list, description="Current disclosure statuses"
    )
    binding_elements: List[BindingElement] = Field(
        default_factory=list, description="Binding elements to verify"
    )
    pai_data_quality: List[PAIDataQuality] = Field(
        default_factory=list, description="PAI data quality metrics"
    )
    portfolio_screening_date: Optional[str] = Field(
        None, description="Date of last portfolio screening"
    )
    taxonomy_alignment_actual_pct: float = Field(
        default=0.0, ge=0.0, le=100.0
    )
    taxonomy_alignment_commitment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0
    )
    sustainable_investment_actual_pct: float = Field(
        default=0.0, ge=0.0, le=100.0
    )
    sustainable_investment_commitment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0
    )
    minimum_pai_coverage_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Minimum acceptable PAI coverage"
    )
    maximum_estimation_rate_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Maximum acceptable estimation rate"
    )
    maximum_data_age_months: int = Field(
        default=18, ge=1, description="Maximum acceptable data age"
    )
    disclosure_freshness_months: int = Field(
        default=12, ge=1, description="Maximum age for current disclosure"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate review date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("review_date must be YYYY-MM-DD format")
        return v

class ComplianceReviewResult(WorkflowResult):
    """Complete result from the compliance review workflow."""
    product_name: str = Field(default="")
    overall_compliance_score: float = Field(default=0.0)
    disclosure_completeness_score: float = Field(default=0.0)
    data_quality_score: float = Field(default=0.0)
    commitment_adherence_score: float = Field(default=0.0)
    critical_issues: int = Field(default=0)
    high_issues: int = Field(default=0)
    total_action_items: int = Field(default=0)
    urgent_action_items: int = Field(default=0)
    is_fully_compliant: bool = Field(default=False)

# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

class DisclosureCompletenessPhase:
    """
    Phase 1: Disclosure Completeness.

    Checks all mandatory disclosures are current, verifies publication
    dates, and identifies missing or outdated content.
    """

    PHASE_NAME = "disclosure_completeness"

    REQUIRED_DISCLOSURES = [
        DisclosureType.ANNEX_II.value,
        DisclosureType.ANNEX_III.value,
        DisclosureType.ANNEX_IV.value,
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute disclosure completeness phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            disclosures = config.get("disclosure_statuses", [])
            review_date = config.get("review_date", "")
            freshness_months = config.get("disclosure_freshness_months", 12)

            review_dt = datetime.strptime(review_date, "%Y-%m-%d")

            issues: List[Dict[str, Any]] = []
            disclosure_checks: List[Dict[str, Any]] = []
            completeness_score = 0.0
            max_score = len(self.REQUIRED_DISCLOSURES) * 100.0

            disclosed_types = {
                d.get("disclosure_type") for d in disclosures
            }

            for req_type in self.REQUIRED_DISCLOSURES:
                matching = [
                    d for d in disclosures
                    if d.get("disclosure_type") == req_type
                ]

                if not matching:
                    issues.append({
                        "severity": IssueSeverity.CRITICAL.value,
                        "category": "missing_disclosure",
                        "description": (
                            f"Required disclosure {req_type} is missing"
                        ),
                        "disclosure_type": req_type,
                    })
                    disclosure_checks.append({
                        "disclosure_type": req_type,
                        "exists": False,
                        "current": False,
                        "published": False,
                        "score": 0.0,
                    })
                    continue

                disc = matching[0]
                score = 0.0

                # Check existence
                if disc.get("exists", False):
                    score += 25.0

                # Check completeness
                completeness = disc.get("completeness_pct", 0.0)
                score += completeness * 0.25

                # Check freshness
                last_updated = disc.get("last_updated", "")
                is_current = False
                if last_updated:
                    try:
                        update_dt = datetime.strptime(
                            last_updated, "%Y-%m-%d"
                        )
                        age_days = (review_dt - update_dt).days
                        age_months = age_days / 30.0
                        is_current = age_months <= freshness_months
                        if is_current:
                            score += 25.0
                        else:
                            issues.append({
                                "severity": IssueSeverity.HIGH.value,
                                "category": "outdated_disclosure",
                                "description": (
                                    f"{req_type} is {age_months:.0f} months "
                                    f"old (max: {freshness_months})"
                                ),
                                "disclosure_type": req_type,
                            })
                    except ValueError:
                        pass

                # Check publication
                if disc.get("published", False):
                    score += 25.0
                else:
                    issues.append({
                        "severity": IssueSeverity.MEDIUM.value,
                        "category": "unpublished_disclosure",
                        "description": f"{req_type} exists but is not published",
                        "disclosure_type": req_type,
                    })

                completeness_score += score
                disclosure_checks.append({
                    "disclosure_type": req_type,
                    "exists": disc.get("exists", False),
                    "current": is_current,
                    "published": disc.get("published", False),
                    "completeness_pct": completeness,
                    "last_updated": last_updated,
                    "score": round(score, 1),
                })

            final_score = round(
                completeness_score / max(max_score, 1) * 100, 1
            )
            outputs["disclosure_checks"] = disclosure_checks
            outputs["issues"] = issues
            outputs["completeness_score"] = final_score
            outputs["disclosures_complete"] = len(issues) == 0

            if issues:
                warnings.append(
                    f"{len(issues)} disclosure issue(s) identified"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "DisclosureCompleteness failed: %s", exc, exc_info=True
            )
            errors.append(f"Disclosure completeness check failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class DataQualityPhase:
    """
    Phase 2: Data Quality.

    Assesses PAI data coverage ratios, estimation rates, and data age.
    Identifies quality improvements needed.
    """

    PHASE_NAME = "data_quality"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute data quality phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            pai_quality = config.get("pai_data_quality", [])
            min_coverage = config.get("minimum_pai_coverage_pct", 50.0)
            max_estimation = config.get("maximum_estimation_rate_pct", 50.0)
            max_age = config.get("maximum_data_age_months", 18)

            issues: List[Dict[str, Any]] = []
            quality_checks: List[Dict[str, Any]] = []
            total_coverage = 0.0
            total_estimation = 0.0

            for pai in pai_quality:
                indicator_id = pai.get("indicator_id", "")
                coverage = pai.get("coverage_pct", 0.0)
                estimation = pai.get("estimation_rate_pct", 0.0)
                age = pai.get("data_age_months")

                total_coverage += coverage
                total_estimation += estimation

                indicator_issues: List[str] = []

                if coverage < min_coverage:
                    indicator_issues.append(
                        f"Coverage {coverage:.1f}% below minimum "
                        f"{min_coverage:.1f}%"
                    )
                    issues.append({
                        "severity": IssueSeverity.HIGH.value,
                        "category": "low_coverage",
                        "description": (
                            f"PAI {indicator_id}: coverage {coverage:.1f}% "
                            f"below minimum {min_coverage:.1f}%"
                        ),
                        "indicator_id": indicator_id,
                    })

                if estimation > max_estimation:
                    indicator_issues.append(
                        f"Estimation rate {estimation:.1f}% exceeds "
                        f"maximum {max_estimation:.1f}%"
                    )
                    issues.append({
                        "severity": IssueSeverity.MEDIUM.value,
                        "category": "high_estimation",
                        "description": (
                            f"PAI {indicator_id}: estimation rate "
                            f"{estimation:.1f}% exceeds {max_estimation:.1f}%"
                        ),
                        "indicator_id": indicator_id,
                    })

                if age is not None and age > max_age:
                    indicator_issues.append(
                        f"Data age {age} months exceeds maximum {max_age}"
                    )
                    issues.append({
                        "severity": IssueSeverity.MEDIUM.value,
                        "category": "stale_data",
                        "description": (
                            f"PAI {indicator_id}: data is {age} months old "
                            f"(max: {max_age})"
                        ),
                        "indicator_id": indicator_id,
                    })

                quality_checks.append({
                    "indicator_id": indicator_id,
                    "indicator_name": pai.get("indicator_name", ""),
                    "coverage_pct": coverage,
                    "estimation_rate_pct": estimation,
                    "data_age_months": age,
                    "issues": indicator_issues,
                    "meets_standards": len(indicator_issues) == 0,
                })

            count = max(len(pai_quality), 1)
            avg_coverage = total_coverage / count
            avg_estimation = total_estimation / count

            outputs["quality_checks"] = quality_checks
            outputs["issues"] = issues
            outputs["average_coverage_pct"] = round(avg_coverage, 1)
            outputs["average_estimation_rate_pct"] = round(avg_estimation, 1)

            # Compute data quality score
            coverage_score = min(avg_coverage / min_coverage * 50, 50)
            estimation_score = max(
                50 - (avg_estimation / max(max_estimation, 1) * 50), 0
            )
            quality_score = round(coverage_score + estimation_score, 1)
            outputs["data_quality_score"] = quality_score

            indicators_passing = sum(
                1 for q in quality_checks if q.get("meets_standards", False)
            )
            outputs["indicators_meeting_standards"] = indicators_passing
            outputs["indicators_total"] = len(pai_quality)

            if issues:
                warnings.append(
                    f"{len(issues)} PAI data quality issue(s) identified"
                )

            status = PhaseStatus.COMPLETED
            records = len(pai_quality)

        except Exception as exc:
            logger.error("DataQuality failed: %s", exc, exc_info=True)
            errors.append(f"Data quality assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

class CommitmentAdherencePhase:
    """
    Phase 3: Commitment Adherence.

    Checks all binding elements are satisfied, verifies minimum
    proportions are maintained, and flags any breaches.
    """

    PHASE_NAME = "commitment_adherence"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute commitment adherence phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            binding_elements = config.get("binding_elements", [])

            issues: List[Dict[str, Any]] = []
            element_checks: List[Dict[str, Any]] = []
            elements_met = 0

            for element in binding_elements:
                name = element.get("name", "")
                el_type = element.get("element_type", "")
                committed = element.get("committed_value", 0.0)
                actual = element.get("actual_value")
                is_satisfied = element.get("is_satisfied")

                # Auto-determine satisfaction if not provided
                if is_satisfied is None and actual is not None:
                    if el_type in ("minimum_proportion", "threshold"):
                        is_satisfied = actual >= committed
                    elif el_type == "exclusion":
                        is_satisfied = actual == 0.0 or actual <= committed
                    else:
                        is_satisfied = actual >= committed

                if is_satisfied:
                    elements_met += 1

                element_check = {
                    "element_id": element.get("element_id", ""),
                    "name": name,
                    "type": el_type,
                    "committed_value": committed,
                    "actual_value": actual,
                    "unit": element.get("unit", "%"),
                    "is_satisfied": is_satisfied,
                    "variance": (
                        round(actual - committed, 2)
                        if actual is not None else None
                    ),
                }
                element_checks.append(element_check)

                if is_satisfied is False:
                    severity = (
                        IssueSeverity.CRITICAL.value
                        if el_type == "exclusion"
                        else IssueSeverity.HIGH.value
                    )
                    issues.append({
                        "severity": severity,
                        "category": "binding_element_breach",
                        "description": (
                            f"Binding element '{name}' not satisfied: "
                            f"committed {committed}, actual {actual}"
                        ),
                        "element_name": name,
                    })

            # Taxonomy alignment check
            tax_actual = config.get("taxonomy_alignment_actual_pct", 0.0)
            tax_committed = config.get(
                "taxonomy_alignment_commitment_pct", 0.0
            )
            if tax_committed > 0:
                tax_met = tax_actual >= tax_committed
                element_checks.append({
                    "name": "Taxonomy Alignment Minimum",
                    "type": "minimum_proportion",
                    "committed_value": tax_committed,
                    "actual_value": tax_actual,
                    "unit": "%",
                    "is_satisfied": tax_met,
                    "variance": round(tax_actual - tax_committed, 2),
                })
                if tax_met:
                    elements_met += 1
                else:
                    issues.append({
                        "severity": IssueSeverity.HIGH.value,
                        "category": "taxonomy_breach",
                        "description": (
                            f"Taxonomy alignment {tax_actual:.2f}% below "
                            f"commitment {tax_committed:.2f}%"
                        ),
                    })

            # Sustainable investment check
            si_actual = config.get(
                "sustainable_investment_actual_pct", 0.0
            )
            si_committed = config.get(
                "sustainable_investment_commitment_pct", 0.0
            )
            if si_committed > 0:
                si_met = si_actual >= si_committed
                element_checks.append({
                    "name": "Sustainable Investment Minimum",
                    "type": "minimum_proportion",
                    "committed_value": si_committed,
                    "actual_value": si_actual,
                    "unit": "%",
                    "is_satisfied": si_met,
                    "variance": round(si_actual - si_committed, 2),
                })
                if si_met:
                    elements_met += 1
                else:
                    issues.append({
                        "severity": IssueSeverity.HIGH.value,
                        "category": "sustainable_investment_breach",
                        "description": (
                            f"Sustainable investment {si_actual:.2f}% below "
                            f"commitment {si_committed:.2f}%"
                        ),
                    })

            total_elements = len(element_checks)
            adherence_score = round(
                elements_met / max(total_elements, 1) * 100, 1
            )

            outputs["element_checks"] = element_checks
            outputs["issues"] = issues
            outputs["elements_met"] = elements_met
            outputs["elements_total"] = total_elements
            outputs["adherence_score"] = adherence_score
            outputs["all_elements_met"] = elements_met == total_elements

            # Portfolio screening freshness
            screening_date = config.get("portfolio_screening_date")
            if screening_date:
                try:
                    review_dt = datetime.strptime(
                        config.get("review_date", ""), "%Y-%m-%d"
                    )
                    screen_dt = datetime.strptime(
                        screening_date, "%Y-%m-%d"
                    )
                    screening_age_days = (review_dt - screen_dt).days
                    outputs["screening_age_days"] = screening_age_days
                    if screening_age_days > 90:
                        issues.append({
                            "severity": IssueSeverity.MEDIUM.value,
                            "category": "stale_screening",
                            "description": (
                                f"Portfolio screening is {screening_age_days} "
                                f"days old (recommend <90 days)"
                            ),
                        })
                except ValueError:
                    pass

            if issues:
                warnings.append(
                    f"{len(issues)} commitment adherence issue(s) identified"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "CommitmentAdherence failed: %s", exc, exc_info=True
            )
            errors.append(f"Commitment adherence check failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class ActionItemsPhase:
    """
    Phase 4: Action Items.

    Generates prioritized remediation tasks, compliance calendar
    updates, and escalation items.
    """

    PHASE_NAME = "action_items"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute action items phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            disclosure_issues = context.get_phase_output(
                "disclosure_completeness"
            ).get("issues", [])
            quality_issues = context.get_phase_output(
                "data_quality"
            ).get("issues", [])
            adherence_issues = context.get_phase_output(
                "commitment_adherence"
            ).get("issues", [])

            all_issues = disclosure_issues + quality_issues + adherence_issues

            # Generate action items from issues
            action_items: List[Dict[str, Any]] = []

            for issue in all_issues:
                severity = issue.get("severity", IssueSeverity.MEDIUM.value)
                category = issue.get("category", "")

                # Map severity to priority and deadline
                if severity == IssueSeverity.CRITICAL.value:
                    priority = ActionPriority.URGENT.value
                    deadline_days = 7
                elif severity == IssueSeverity.HIGH.value:
                    priority = ActionPriority.HIGH.value
                    deadline_days = 30
                elif severity == IssueSeverity.MEDIUM.value:
                    priority = ActionPriority.MEDIUM.value
                    deadline_days = 60
                else:
                    priority = ActionPriority.LOW.value
                    deadline_days = 90

                action_items.append({
                    "action_id": str(uuid.uuid4()),
                    "priority": priority,
                    "category": category,
                    "description": issue.get("description", ""),
                    "remediation": self._generate_remediation(
                        category, issue
                    ),
                    "deadline_days": deadline_days,
                    "owner": self._assign_owner(category),
                    "status": "open",
                })

            # Sort by priority
            priority_order = {
                ActionPriority.URGENT.value: 0,
                ActionPriority.HIGH.value: 1,
                ActionPriority.MEDIUM.value: 2,
                ActionPriority.LOW.value: 3,
            }
            action_items.sort(
                key=lambda x: priority_order.get(x.get("priority", ""), 99)
            )

            outputs["action_items"] = action_items
            outputs["total_action_items"] = len(action_items)
            outputs["urgent_items"] = sum(
                1 for a in action_items
                if a.get("priority") == ActionPriority.URGENT.value
            )
            outputs["high_items"] = sum(
                1 for a in action_items
                if a.get("priority") == ActionPriority.HIGH.value
            )

            # Overall compliance scores
            disc_score = context.get_phase_output(
                "disclosure_completeness"
            ).get("completeness_score", 0.0)
            dq_score = context.get_phase_output(
                "data_quality"
            ).get("data_quality_score", 0.0)
            adh_score = context.get_phase_output(
                "commitment_adherence"
            ).get("adherence_score", 0.0)

            overall_score = round(
                (disc_score + dq_score + adh_score) / 3, 1
            )
            outputs["overall_compliance_score"] = overall_score
            outputs["disclosure_completeness_score"] = disc_score
            outputs["data_quality_score"] = dq_score
            outputs["commitment_adherence_score"] = adh_score
            outputs["is_fully_compliant"] = (
                overall_score >= 90.0
                and outputs["urgent_items"] == 0
            )

            # Issue summary by severity
            severity_counts = {s.value: 0 for s in IssueSeverity}
            for issue in all_issues:
                sev = issue.get("severity", IssueSeverity.INFO.value)
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            outputs["issue_severity_distribution"] = severity_counts

            # Compliance calendar
            outputs["compliance_calendar"] = {
                "next_review_date": config.get("review_date", ""),
                "annex_ii_update_due": (
                    "On material change" if disc_score >= 80 else "Immediate"
                ),
                "annex_iii_update_due": (
                    "Quarterly review" if disc_score >= 80 else "Immediate"
                ),
                "annex_iv_due": "Within 12 months of reporting period end",
                "pai_data_refresh": "Quarterly",
                "portfolio_screening": "Continuous / at minimum quarterly",
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ActionItems failed: %s", exc, exc_info=True)
            errors.append(f"Action items generation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _generate_remediation(
        self, category: str, issue: Dict[str, Any]
    ) -> str:
        """Generate remediation guidance for an issue."""
        remediations = {
            "missing_disclosure": (
                "Create and publish the missing disclosure document. "
                "Use the relevant workflow to generate content."
            ),
            "outdated_disclosure": (
                "Update the disclosure with current data and republish."
            ),
            "unpublished_disclosure": (
                "Review the disclosure and publish to the website."
            ),
            "low_coverage": (
                "Improve data sourcing for this PAI indicator. "
                "Consider additional data providers or direct engagement."
            ),
            "high_estimation": (
                "Reduce estimation dependency by sourcing reported data "
                "from investee companies."
            ),
            "stale_data": (
                "Refresh data from current reporting period."
            ),
            "binding_element_breach": (
                "Immediate portfolio adjustment required to restore "
                "compliance with binding element."
            ),
            "taxonomy_breach": (
                "Review portfolio composition and increase taxonomy-aligned "
                "holdings to meet commitment."
            ),
            "sustainable_investment_breach": (
                "Increase sustainable investment proportion or review "
                "pre-contractual commitment level."
            ),
            "stale_screening": (
                "Run portfolio screening workflow to refresh compliance status."
            ),
        }
        return remediations.get(
            category, "Review and address the identified issue."
        )

    def _assign_owner(self, category: str) -> str:
        """Assign default owner for an action item category."""
        owner_map = {
            "missing_disclosure": "compliance_officer",
            "outdated_disclosure": "compliance_officer",
            "unpublished_disclosure": "compliance_officer",
            "low_coverage": "data_team",
            "high_estimation": "data_team",
            "stale_data": "data_team",
            "binding_element_breach": "portfolio_manager",
            "taxonomy_breach": "portfolio_manager",
            "sustainable_investment_breach": "portfolio_manager",
            "stale_screening": "compliance_officer",
        }
        return owner_map.get(category, "compliance_officer")

# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================

class ComplianceReviewWorkflow:
    """
    Four-phase compliance review workflow for SFDR Article 8.

    Orchestrates disclosure completeness checking, data quality assessment,
    commitment adherence verification, and action item generation.

    Example:
        >>> wf = ComplianceReviewWorkflow()
        >>> input_data = ComplianceReviewInput(
        ...     organization_id="org-123",
        ...     product_name="Green Bond Fund",
        ...     review_date="2026-03-15",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "compliance_review"

    PHASE_ORDER = [
        "disclosure_completeness",
        "data_quality",
        "commitment_adherence",
        "action_items",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize the compliance review workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "disclosure_completeness": DisclosureCompletenessPhase(),
            "data_quality": DataQualityPhase(),
            "commitment_adherence": CommitmentAdherencePhase(),
            "action_items": ActionItemsPhase(),
        }

    async def run(
        self, input_data: ComplianceReviewInput
    ) -> ComplianceReviewResult:
        """Execute the complete 4-phase compliance review workflow."""
        started_at = utcnow()
        logger.info(
            "Starting compliance review workflow %s for org=%s product=%s",
            self.workflow_id, input_data.organization_id,
            input_data.product_name,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised unhandled exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )

        return ComplianceReviewResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            product_name=summary.get("product_name", ""),
            overall_compliance_score=summary.get(
                "overall_compliance_score", 0.0
            ),
            disclosure_completeness_score=summary.get(
                "disclosure_completeness_score", 0.0
            ),
            data_quality_score=summary.get("data_quality_score", 0.0),
            commitment_adherence_score=summary.get(
                "commitment_adherence_score", 0.0
            ),
            critical_issues=summary.get("critical_issues", 0),
            high_issues=summary.get("high_issues", 0),
            total_action_items=summary.get("total_action_items", 0),
            urgent_action_items=summary.get("urgent_action_items", 0),
            is_fully_compliant=summary.get("is_fully_compliant", False),
        )

    def _build_config(
        self, input_data: ComplianceReviewInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        if input_data.disclosure_statuses:
            config["disclosure_statuses"] = [
                d.model_dump() for d in input_data.disclosure_statuses
            ]
            for d in config["disclosure_statuses"]:
                d["disclosure_type"] = d["disclosure_type"].value if isinstance(
                    d["disclosure_type"], DisclosureType
                ) else d["disclosure_type"]
        if input_data.binding_elements:
            config["binding_elements"] = [
                b.model_dump() for b in input_data.binding_elements
            ]
        if input_data.pai_data_quality:
            config["pai_data_quality"] = [
                p.model_dump() for p in input_data.pai_data_quality
            ]
        return config

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        config = context.config
        action_out = context.get_phase_output("action_items")

        issue_dist = action_out.get("issue_severity_distribution", {})

        return {
            "product_name": config.get("product_name", ""),
            "overall_compliance_score": action_out.get(
                "overall_compliance_score", 0.0
            ),
            "disclosure_completeness_score": action_out.get(
                "disclosure_completeness_score", 0.0
            ),
            "data_quality_score": action_out.get(
                "data_quality_score", 0.0
            ),
            "commitment_adherence_score": action_out.get(
                "commitment_adherence_score", 0.0
            ),
            "critical_issues": issue_dist.get(
                IssueSeverity.CRITICAL.value, 0
            ),
            "high_issues": issue_dist.get(IssueSeverity.HIGH.value, 0),
            "total_action_items": action_out.get(
                "total_action_items", 0
            ),
            "urgent_action_items": action_out.get("urgent_items", 0),
            "is_fully_compliant": action_out.get(
                "is_fully_compliant", False
            ),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
