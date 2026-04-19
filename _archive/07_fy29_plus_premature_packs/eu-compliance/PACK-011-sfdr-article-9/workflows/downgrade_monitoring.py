# -*- coding: utf-8 -*-
"""
Downgrade Monitoring Workflow
================================================

Four-phase workflow for monitoring Article 9 to Article 8 downgrade risk
under SFDR. Orchestrates compliance checking, threshold monitoring, risk
scoring, and alert generation into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 9 products must maintain sustainable investment as their
      objective at all times. Failure to meet Article 9 requirements may
      trigger a mandatory reclassification (downgrade) to Article 8 or
      Article 6.
    - Post-2023 SFDR review, regulators have increased scrutiny on Article 9
      products, with several major fund downgrades observed across the EU.
    - Key downgrade triggers include: loss of 100% sustainable investment
      commitment, failure to meet taxonomy alignment minimum, DNSH
      assessment gaps, benchmark misalignment, and PAI non-compliance.
    - Proactive monitoring prevents regulatory action and reputational damage.
    - National Competent Authorities (NCAs) may require immediate disclosure
      updates upon reclassification.

Phases:
    1. ComplianceCheck - Verify ongoing Article 9 compliance requirements
    2. ThresholdMonitoring - Monitor key thresholds against regulatory limits
    3. RiskScoring - Calculate composite downgrade risk score
    4. AlertGeneration - Generate alerts and escalation recommendations

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity

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

class DowngradeRiskLevel(str, Enum):
    """Downgrade risk classification level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class DowngradeTrigger(str, Enum):
    """Types of events that can trigger a downgrade."""
    SUSTAINABLE_COMMITMENT_BREACH = "SUSTAINABLE_COMMITMENT_BREACH"
    TAXONOMY_ALIGNMENT_SHORTFALL = "TAXONOMY_ALIGNMENT_SHORTFALL"
    DNSH_ASSESSMENT_GAP = "DNSH_ASSESSMENT_GAP"
    BENCHMARK_MISALIGNMENT = "BENCHMARK_MISALIGNMENT"
    PAI_NON_COMPLIANCE = "PAI_NON_COMPLIANCE"
    EXCLUSION_VIOLATION = "EXCLUSION_VIOLATION"
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    REGULATORY_CHANGE = "REGULATORY_CHANGE"
    GOOD_GOVERNANCE_FAILURE = "GOOD_GOVERNANCE_FAILURE"

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
# DATA MODELS - DOWNGRADE MONITORING
# =============================================================================

class DowngradeMonitoringInput(BaseModel):
    """Input configuration for the downgrade monitoring workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None, description="ISIN if applicable")
    assessment_date: str = Field(
        ..., description="Assessment date YYYY-MM-DD"
    )
    current_classification: str = Field(
        default="ARTICLE_9",
        description="Current SFDR classification"
    )
    sustainable_investment_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Current sustainable investment percentage"
    )
    taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Current taxonomy alignment percentage"
    )
    minimum_taxonomy_commitment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Committed minimum taxonomy alignment"
    )
    dnsh_assessment_complete: bool = Field(
        default=True,
        description="Whether DNSH assessment is current and complete"
    )
    dnsh_objectives_covered: int = Field(
        default=6, ge=0, le=6,
        description="Number of environmental objectives with DNSH assessment"
    )
    benchmark_aligned: bool = Field(
        default=True,
        description="Whether portfolio is aligned with designated benchmark"
    )
    benchmark_deviation_pct: float = Field(
        default=0.0, ge=0.0,
        description="Percentage deviation from benchmark alignment"
    )
    pai_indicators_compliant: int = Field(
        default=14, ge=0, le=14,
        description="Number of compliant mandatory PAI indicators"
    )
    exclusion_violations: int = Field(
        default=0, ge=0,
        description="Number of active exclusion criteria violations"
    )
    good_governance_issues: int = Field(
        default=0, ge=0,
        description="Number of investees failing good governance checks"
    )
    data_quality_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Overall data quality score (0-1)"
    )
    engagement_rate_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Percentage of investees engaged on sustainability"
    )
    prior_risk_score: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Previous assessment risk score for trend analysis"
    )
    alert_recipients: List[str] = Field(
        default_factory=list,
        description="Email addresses for alert notifications"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("assessment_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate assessment date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("assessment_date must be YYYY-MM-DD format")
        return v

class DowngradeMonitoringResult(WorkflowResult):
    """Complete result from the downgrade monitoring workflow."""
    product_name: str = Field(default="")
    current_classification: str = Field(default="ARTICLE_9")
    risk_level: str = Field(default=DowngradeRiskLevel.LOW.value)
    risk_score: float = Field(default=0.0)
    compliance_checks_passed: int = Field(default=0)
    compliance_checks_total: int = Field(default=0)
    thresholds_breached: int = Field(default=0)
    triggers_active: int = Field(default=0)
    alerts_generated: int = Field(default=0)
    critical_alerts: int = Field(default=0)
    immediate_action_required: bool = Field(default=False)

# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

class ComplianceCheckPhase:
    """
    Phase 1: Compliance Check.

    Verifies ongoing compliance with all Article 9 requirements including
    sustainable investment commitment, taxonomy alignment, DNSH coverage,
    benchmark alignment, PAI compliance, and good governance.
    """

    PHASE_NAME = "compliance_check"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute compliance check phase.

        Args:
            context: Workflow context with product status data.

        Returns:
            PhaseResult with compliance check results.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            product_name = config.get("product_name", "")
            outputs["product_name"] = product_name
            outputs["current_classification"] = config.get(
                "current_classification", "ARTICLE_9"
            )

            compliance_checks = []

            # Check 1: 100% sustainable investment commitment
            sustainable_pct = config.get(
                "sustainable_investment_pct", 100.0
            )
            check_passed = sustainable_pct >= 99.0
            compliance_checks.append({
                "check_id": "CC-01",
                "name": "Sustainable Investment Commitment",
                "passed": check_passed,
                "required": ">=99%",
                "actual": f"{sustainable_pct:.1f}%",
                "trigger": (
                    DowngradeTrigger.SUSTAINABLE_COMMITMENT_BREACH.value
                    if not check_passed else None
                ),
            })

            # Check 2: Taxonomy alignment minimum
            taxonomy_pct = config.get("taxonomy_aligned_pct", 0.0)
            min_taxonomy = config.get(
                "minimum_taxonomy_commitment_pct", 0.0
            )
            check_passed = taxonomy_pct >= min_taxonomy
            compliance_checks.append({
                "check_id": "CC-02",
                "name": "Taxonomy Alignment Minimum",
                "passed": check_passed,
                "required": f">={min_taxonomy:.1f}%",
                "actual": f"{taxonomy_pct:.1f}%",
                "trigger": (
                    DowngradeTrigger.TAXONOMY_ALIGNMENT_SHORTFALL.value
                    if not check_passed else None
                ),
            })

            # Check 3: DNSH assessment completeness
            dnsh_complete = config.get("dnsh_assessment_complete", True)
            dnsh_objectives = config.get("dnsh_objectives_covered", 6)
            check_passed = dnsh_complete and dnsh_objectives == 6
            compliance_checks.append({
                "check_id": "CC-03",
                "name": "DNSH Assessment Completeness",
                "passed": check_passed,
                "required": "Complete (6/6 objectives)",
                "actual": (
                    f"{'Complete' if dnsh_complete else 'Incomplete'} "
                    f"({dnsh_objectives}/6 objectives)"
                ),
                "trigger": (
                    DowngradeTrigger.DNSH_ASSESSMENT_GAP.value
                    if not check_passed else None
                ),
            })

            # Check 4: Benchmark alignment
            benchmark_aligned = config.get("benchmark_aligned", True)
            benchmark_deviation = config.get(
                "benchmark_deviation_pct", 0.0
            )
            check_passed = benchmark_aligned and benchmark_deviation < 5.0
            compliance_checks.append({
                "check_id": "CC-04",
                "name": "Benchmark Alignment",
                "passed": check_passed,
                "required": "Aligned (<5% deviation)",
                "actual": (
                    f"{'Aligned' if benchmark_aligned else 'Misaligned'} "
                    f"({benchmark_deviation:.1f}% deviation)"
                ),
                "trigger": (
                    DowngradeTrigger.BENCHMARK_MISALIGNMENT.value
                    if not check_passed else None
                ),
            })

            # Check 5: PAI compliance
            pai_compliant = config.get("pai_indicators_compliant", 14)
            check_passed = pai_compliant >= 14
            compliance_checks.append({
                "check_id": "CC-05",
                "name": "Mandatory PAI Compliance",
                "passed": check_passed,
                "required": "14/14 indicators",
                "actual": f"{pai_compliant}/14 indicators",
                "trigger": (
                    DowngradeTrigger.PAI_NON_COMPLIANCE.value
                    if not check_passed else None
                ),
            })

            # Check 6: Exclusion criteria
            exclusion_violations = config.get(
                "exclusion_violations", 0
            )
            check_passed = exclusion_violations == 0
            compliance_checks.append({
                "check_id": "CC-06",
                "name": "Exclusion Criteria Compliance",
                "passed": check_passed,
                "required": "0 violations",
                "actual": f"{exclusion_violations} violation(s)",
                "trigger": (
                    DowngradeTrigger.EXCLUSION_VIOLATION.value
                    if not check_passed else None
                ),
            })

            # Check 7: Good governance
            governance_issues = config.get(
                "good_governance_issues", 0
            )
            check_passed = governance_issues == 0
            compliance_checks.append({
                "check_id": "CC-07",
                "name": "Good Governance",
                "passed": check_passed,
                "required": "0 issues",
                "actual": f"{governance_issues} issue(s)",
                "trigger": (
                    DowngradeTrigger.GOOD_GOVERNANCE_FAILURE.value
                    if not check_passed else None
                ),
            })

            # Check 8: Data quality
            data_quality = config.get("data_quality_score", 1.0)
            check_passed = data_quality >= 0.5
            compliance_checks.append({
                "check_id": "CC-08",
                "name": "Data Quality Threshold",
                "passed": check_passed,
                "required": ">=0.50",
                "actual": f"{data_quality:.2f}",
                "trigger": (
                    DowngradeTrigger.DATA_QUALITY_FAILURE.value
                    if not check_passed else None
                ),
            })

            passed_count = sum(
                1 for c in compliance_checks if c["passed"]
            )
            total_count = len(compliance_checks)

            # Collect active triggers
            active_triggers = [
                c["trigger"] for c in compliance_checks
                if c.get("trigger") is not None
            ]

            outputs["compliance_checks"] = compliance_checks
            outputs["checks_passed"] = passed_count
            outputs["checks_total"] = total_count
            outputs["all_compliant"] = passed_count == total_count
            outputs["active_triggers"] = active_triggers
            outputs["triggers_count"] = len(active_triggers)

            if active_triggers:
                warnings.append(
                    f"{len(active_triggers)} compliance check(s) "
                    f"failed: {', '.join(active_triggers)}"
                )

            status = PhaseStatus.COMPLETED
            records = total_count

        except Exception as exc:
            logger.error(
                "ComplianceCheck failed: %s", exc, exc_info=True
            )
            errors.append(f"Compliance check failed: {str(exc)}")
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

class ThresholdMonitoringPhase:
    """
    Phase 2: Threshold Monitoring.

    Monitors key metrics against regulatory and internal thresholds,
    calculates proximity to breach levels, and identifies metrics
    trending toward breaches.
    """

    PHASE_NAME = "threshold_monitoring"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute threshold monitoring phase.

        Args:
            context: Workflow context with compliance check results.

        Returns:
            PhaseResult with threshold monitoring results.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config

            thresholds: List[Dict[str, Any]] = []

            # Threshold 1: Sustainable investment percentage
            sustainable_pct = config.get(
                "sustainable_investment_pct", 100.0
            )
            breach_level = 99.0
            warning_level = 99.5
            thresholds.append({
                "metric": "sustainable_investment_pct",
                "current_value": sustainable_pct,
                "breach_level": breach_level,
                "warning_level": warning_level,
                "breached": sustainable_pct < breach_level,
                "in_warning_zone": (
                    breach_level <= sustainable_pct < warning_level
                ),
                "headroom_pct": round(
                    sustainable_pct - breach_level, 2
                ),
            })

            # Threshold 2: Taxonomy alignment
            taxonomy_pct = config.get("taxonomy_aligned_pct", 0.0)
            min_taxonomy = config.get(
                "minimum_taxonomy_commitment_pct", 0.0
            )
            warning_buffer = min_taxonomy * 1.1 if min_taxonomy > 0 else 5.0
            thresholds.append({
                "metric": "taxonomy_aligned_pct",
                "current_value": taxonomy_pct,
                "breach_level": min_taxonomy,
                "warning_level": warning_buffer,
                "breached": taxonomy_pct < min_taxonomy,
                "in_warning_zone": (
                    min_taxonomy <= taxonomy_pct < warning_buffer
                ),
                "headroom_pct": round(
                    taxonomy_pct - min_taxonomy, 2
                ),
            })

            # Threshold 3: DNSH coverage
            dnsh_objectives = config.get("dnsh_objectives_covered", 6)
            thresholds.append({
                "metric": "dnsh_objectives_covered",
                "current_value": dnsh_objectives,
                "breach_level": 6,
                "warning_level": 6,
                "breached": dnsh_objectives < 6,
                "in_warning_zone": False,
                "headroom_pct": dnsh_objectives - 6,
            })

            # Threshold 4: PAI indicators
            pai_compliant = config.get("pai_indicators_compliant", 14)
            thresholds.append({
                "metric": "pai_indicators_compliant",
                "current_value": pai_compliant,
                "breach_level": 14,
                "warning_level": 14,
                "breached": pai_compliant < 14,
                "in_warning_zone": False,
                "headroom_pct": pai_compliant - 14,
            })

            # Threshold 5: Benchmark deviation
            benchmark_deviation = config.get(
                "benchmark_deviation_pct", 0.0
            )
            thresholds.append({
                "metric": "benchmark_deviation_pct",
                "current_value": benchmark_deviation,
                "breach_level": 5.0,
                "warning_level": 3.0,
                "breached": benchmark_deviation >= 5.0,
                "in_warning_zone": (
                    3.0 <= benchmark_deviation < 5.0
                ),
                "headroom_pct": round(
                    5.0 - benchmark_deviation, 2
                ),
            })

            # Threshold 6: Data quality
            data_quality = config.get("data_quality_score", 1.0)
            thresholds.append({
                "metric": "data_quality_score",
                "current_value": data_quality,
                "breach_level": 0.5,
                "warning_level": 0.65,
                "breached": data_quality < 0.5,
                "in_warning_zone": 0.5 <= data_quality < 0.65,
                "headroom_pct": round(data_quality - 0.5, 3),
            })

            breached_count = sum(
                1 for t in thresholds if t["breached"]
            )
            warning_count = sum(
                1 for t in thresholds if t["in_warning_zone"]
            )

            outputs["thresholds"] = thresholds
            outputs["thresholds_breached"] = breached_count
            outputs["thresholds_in_warning"] = warning_count
            outputs["thresholds_total"] = len(thresholds)

            if breached_count > 0:
                breached_names = [
                    t["metric"] for t in thresholds if t["breached"]
                ]
                warnings.append(
                    f"{breached_count} threshold(s) breached: "
                    f"{', '.join(breached_names)}"
                )

            if warning_count > 0:
                warning_names = [
                    t["metric"] for t in thresholds
                    if t["in_warning_zone"]
                ]
                warnings.append(
                    f"{warning_count} metric(s) in warning zone: "
                    f"{', '.join(warning_names)}"
                )

            status = PhaseStatus.COMPLETED
            records = len(thresholds)

        except Exception as exc:
            logger.error(
                "ThresholdMonitoring failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Threshold monitoring failed: {str(exc)}"
            )
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

class RiskScoringPhase:
    """
    Phase 3: Risk Scoring.

    Calculates a composite downgrade risk score (0-100) based on
    compliance check results, threshold proximity, and trend analysis.
    Uses deterministic weighted scoring formula.
    """

    PHASE_NAME = "risk_scoring"

    # Weight allocation for risk score components
    RISK_WEIGHTS = {
        "compliance_failures": 0.40,
        "threshold_proximity": 0.25,
        "threshold_breaches": 0.25,
        "trend": 0.10,
    }

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute risk scoring phase.

        Args:
            context: Workflow context with compliance and threshold data.

        Returns:
            PhaseResult with composite risk score and classification.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            compliance_output = context.get_phase_output(
                "compliance_check"
            )
            threshold_output = context.get_phase_output(
                "threshold_monitoring"
            )

            checks_passed = compliance_output.get("checks_passed", 0)
            checks_total = compliance_output.get("checks_total", 1)
            triggers_count = compliance_output.get("triggers_count", 0)
            thresholds_breached = threshold_output.get(
                "thresholds_breached", 0
            )
            thresholds_total = threshold_output.get(
                "thresholds_total", 1
            )
            thresholds_warning = threshold_output.get(
                "thresholds_in_warning", 0
            )

            # Component 1: Compliance failure score (0-100)
            compliance_failure_rate = (
                (checks_total - checks_passed) / checks_total * 100.0
                if checks_total > 0 else 0.0
            )

            # Component 2: Threshold proximity score (0-100)
            # Combines warning zone metrics
            proximity_score = (
                thresholds_warning / thresholds_total * 50.0
                if thresholds_total > 0 else 0.0
            )

            # Component 3: Threshold breach score (0-100)
            breach_score = (
                thresholds_breached / thresholds_total * 100.0
                if thresholds_total > 0 else 0.0
            )

            # Component 4: Trend score (0-100)
            prior_risk = config.get("prior_risk_score")
            if prior_risk is not None:
                # If risk is increasing, trend contributes to score
                trend_score = max(0.0, min(100.0, prior_risk * 0.5))
            else:
                trend_score = 0.0

            # Composite weighted score
            composite = (
                compliance_failure_rate * self.RISK_WEIGHTS["compliance_failures"]
                + proximity_score * self.RISK_WEIGHTS["threshold_proximity"]
                + breach_score * self.RISK_WEIGHTS["threshold_breaches"]
                + trend_score * self.RISK_WEIGHTS["trend"]
            )
            composite = round(min(composite, 100.0), 1)

            # Risk level classification
            if composite >= 75.0:
                risk_level = DowngradeRiskLevel.CRITICAL.value
            elif composite >= 50.0:
                risk_level = DowngradeRiskLevel.HIGH.value
            elif composite >= 25.0:
                risk_level = DowngradeRiskLevel.MEDIUM.value
            else:
                risk_level = DowngradeRiskLevel.LOW.value

            outputs["risk_score"] = composite
            outputs["risk_level"] = risk_level
            outputs["score_components"] = {
                "compliance_failure_rate": round(
                    compliance_failure_rate, 1
                ),
                "proximity_score": round(proximity_score, 1),
                "breach_score": round(breach_score, 1),
                "trend_score": round(trend_score, 1),
            }
            outputs["risk_weights"] = self.RISK_WEIGHTS
            outputs["prior_risk_score"] = prior_risk
            outputs["risk_trend"] = (
                "increasing" if prior_risk and composite > prior_risk
                else "decreasing" if prior_risk and composite < prior_risk
                else "stable" if prior_risk
                else "no_prior_data"
            )
            outputs["triggers_active"] = triggers_count
            outputs["immediate_action_required"] = (
                risk_level == DowngradeRiskLevel.CRITICAL.value
            )

            if risk_level == DowngradeRiskLevel.CRITICAL.value:
                warnings.append(
                    f"CRITICAL downgrade risk (score: {composite:.1f}). "
                    f"Immediate action required to prevent "
                    f"Article 9 reclassification."
                )
            elif risk_level == DowngradeRiskLevel.HIGH.value:
                warnings.append(
                    f"HIGH downgrade risk (score: {composite:.1f}). "
                    f"Remediation actions should be prioritized."
                )

            status = PhaseStatus.COMPLETED
            records = 1

        except Exception as exc:
            logger.error(
                "RiskScoring failed: %s", exc, exc_info=True
            )
            errors.append(f"Risk scoring failed: {str(exc)}")
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

class AlertGenerationPhase:
    """
    Phase 4: Alert Generation.

    Generates structured alerts based on risk scoring results, compliance
    failures, and threshold breaches. Includes escalation recommendations
    and remediation timelines.
    """

    PHASE_NAME = "alert_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute alert generation phase.

        Args:
            context: Workflow context with risk scoring results.

        Returns:
            PhaseResult with generated alerts and escalation plan.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            compliance_output = context.get_phase_output(
                "compliance_check"
            )
            threshold_output = context.get_phase_output(
                "threshold_monitoring"
            )
            risk_output = context.get_phase_output("risk_scoring")

            risk_level = risk_output.get(
                "risk_level", DowngradeRiskLevel.LOW.value
            )
            risk_score = risk_output.get("risk_score", 0.0)
            compliance_checks = compliance_output.get(
                "compliance_checks", []
            )
            thresholds = threshold_output.get("thresholds", [])
            alert_recipients = config.get("alert_recipients", [])

            alerts: List[Dict[str, Any]] = []

            # Generate alerts for compliance failures
            for check in compliance_checks:
                if not check.get("passed", True):
                    severity = (
                        AlertSeverity.CRITICAL.value
                        if check.get("trigger") in (
                            DowngradeTrigger.SUSTAINABLE_COMMITMENT_BREACH.value,
                            DowngradeTrigger.EXCLUSION_VIOLATION.value,
                        )
                        else AlertSeverity.HIGH.value
                    )
                    alerts.append({
                        "alert_id": str(uuid.uuid4()),
                        "severity": severity,
                        "category": "compliance_failure",
                        "title": (
                            f"Compliance Check Failed: {check['name']}"
                        ),
                        "description": (
                            f"{check['name']}: required {check['required']}, "
                            f"actual {check['actual']}"
                        ),
                        "trigger": check.get("trigger"),
                        "recommended_action": (
                            f"Address {check['name']} failure immediately"
                        ),
                        "timeline": (
                            "Immediate" if severity == AlertSeverity.CRITICAL.value
                            else "Within 30 days"
                        ),
                        "recipients": alert_recipients,
                        "generated_at": utcnow().isoformat(),
                    })

            # Generate alerts for threshold breaches
            for threshold in thresholds:
                if threshold.get("breached"):
                    alerts.append({
                        "alert_id": str(uuid.uuid4()),
                        "severity": AlertSeverity.HIGH.value,
                        "category": "threshold_breach",
                        "title": (
                            f"Threshold Breached: {threshold['metric']}"
                        ),
                        "description": (
                            f"{threshold['metric']}: current "
                            f"{threshold['current_value']}, "
                            f"breach level {threshold['breach_level']}"
                        ),
                        "trigger": None,
                        "recommended_action": (
                            f"Restore {threshold['metric']} above "
                            f"breach level ({threshold['breach_level']})"
                        ),
                        "timeline": "Within 30 days",
                        "recipients": alert_recipients,
                        "generated_at": utcnow().isoformat(),
                    })
                elif threshold.get("in_warning_zone"):
                    alerts.append({
                        "alert_id": str(uuid.uuid4()),
                        "severity": AlertSeverity.WARNING.value,
                        "category": "threshold_warning",
                        "title": (
                            f"Warning Zone: {threshold['metric']}"
                        ),
                        "description": (
                            f"{threshold['metric']} approaching breach "
                            f"level (headroom: "
                            f"{threshold['headroom_pct']})"
                        ),
                        "trigger": None,
                        "recommended_action": (
                            f"Monitor {threshold['metric']} closely "
                            f"and prepare contingency plan"
                        ),
                        "timeline": "Within 60 days",
                        "recipients": alert_recipients,
                        "generated_at": utcnow().isoformat(),
                    })

            # Generate overall risk alert
            if risk_level in (
                DowngradeRiskLevel.HIGH.value,
                DowngradeRiskLevel.CRITICAL.value,
            ):
                alerts.append({
                    "alert_id": str(uuid.uuid4()),
                    "severity": (
                        AlertSeverity.CRITICAL.value
                        if risk_level == DowngradeRiskLevel.CRITICAL.value
                        else AlertSeverity.HIGH.value
                    ),
                    "category": "overall_risk",
                    "title": (
                        f"Downgrade Risk: {risk_level} "
                        f"(Score: {risk_score:.1f})"
                    ),
                    "description": (
                        f"Product '{config.get('product_name', '')}' has "
                        f"{risk_level} downgrade risk with composite "
                        f"score of {risk_score:.1f}/100"
                    ),
                    "trigger": None,
                    "recommended_action": (
                        "Convene risk committee to review Article 9 status"
                        if risk_level == DowngradeRiskLevel.CRITICAL.value
                        else "Schedule review of compliance gaps"
                    ),
                    "timeline": (
                        "Within 48 hours"
                        if risk_level == DowngradeRiskLevel.CRITICAL.value
                        else "Within 2 weeks"
                    ),
                    "recipients": alert_recipients,
                    "generated_at": utcnow().isoformat(),
                })

            # Escalation plan
            escalation = {
                "risk_level": risk_level,
                "escalation_required": risk_level in (
                    DowngradeRiskLevel.HIGH.value,
                    DowngradeRiskLevel.CRITICAL.value,
                ),
                "escalation_chain": [],
            }

            if risk_level == DowngradeRiskLevel.CRITICAL.value:
                escalation["escalation_chain"] = [
                    {"role": "Compliance Officer", "timeline": "Immediate"},
                    {"role": "Chief Risk Officer", "timeline": "Within 24h"},
                    {"role": "Board Risk Committee", "timeline": "Within 48h"},
                    {"role": "NCA Notification (if required)", "timeline": "Within 5 days"},
                ]
            elif risk_level == DowngradeRiskLevel.HIGH.value:
                escalation["escalation_chain"] = [
                    {"role": "Compliance Officer", "timeline": "Within 24h"},
                    {"role": "Head of Product", "timeline": "Within 1 week"},
                ]

            critical_count = sum(
                1 for a in alerts
                if a["severity"] == AlertSeverity.CRITICAL.value
            )

            outputs["alerts"] = alerts
            outputs["alerts_count"] = len(alerts)
            outputs["critical_alerts"] = critical_count
            outputs["escalation_plan"] = escalation
            outputs["alert_recipients"] = alert_recipients
            outputs["generated_at"] = utcnow().isoformat()

            status = PhaseStatus.COMPLETED
            records = len(alerts)

        except Exception as exc:
            logger.error(
                "AlertGeneration failed: %s", exc, exc_info=True
            )
            errors.append(f"Alert generation failed: {str(exc)}")
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

# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================

class DowngradeMonitoringWorkflow:
    """
    Four-phase downgrade risk monitoring workflow for Article 9.

    Orchestrates the complete downgrade monitoring pipeline from compliance
    checking through threshold monitoring, risk scoring, and alert
    generation. Supports checkpoint/resume and phase skipping.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = DowngradeMonitoringWorkflow()
        >>> input_data = DowngradeMonitoringInput(
        ...     organization_id="org-123",
        ...     product_name="Climate Solutions Fund",
        ...     assessment_date="2026-03-01",
        ...     sustainable_investment_pct=98.5,
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    """

    WORKFLOW_NAME = "downgrade_monitoring"

    PHASE_ORDER = [
        "compliance_check",
        "threshold_monitoring",
        "risk_scoring",
        "alert_generation",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the downgrade monitoring workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "compliance_check": ComplianceCheckPhase(),
            "threshold_monitoring": ThresholdMonitoringPhase(),
            "risk_scoring": RiskScoringPhase(),
            "alert_generation": AlertGenerationPhase(),
        }

    async def run(
        self, input_data: DowngradeMonitoringInput
    ) -> DowngradeMonitoringResult:
        """
        Execute the complete 4-phase downgrade monitoring workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            DowngradeMonitoringResult with per-phase details and summary.
        """
        started_at = utcnow()
        logger.info(
            "Starting downgrade monitoring workflow %s for org=%s product=%s",
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
                logger.info(
                    "Phase '%s' already completed, skipping",
                    phase_name,
                )
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(
                phase_name, f"Starting: {phase_name}", pct
            )
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(
                        phase_name, phase_result.outputs
                    )
                    context.mark_phase(
                        phase_name, PhaseStatus.COMPLETED
                    )
                else:
                    context.mark_phase(
                        phase_name, phase_result.status
                    )
                    if phase_name == "compliance_check":
                        overall_status = WorkflowStatus.FAILED
                        logger.error(
                            "Critical phase '%s' failed, aborting",
                            phase_name,
                        )
                        break

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
                WorkflowStatus.COMPLETED if all_ok
                else WorkflowStatus.PARTIAL
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
        logger.info(
            "Downgrade monitoring workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return DowngradeMonitoringResult(
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
            current_classification=summary.get(
                "current_classification", "ARTICLE_9"
            ),
            risk_level=summary.get(
                "risk_level", DowngradeRiskLevel.LOW.value
            ),
            risk_score=summary.get("risk_score", 0.0),
            compliance_checks_passed=summary.get(
                "compliance_checks_passed", 0
            ),
            compliance_checks_total=summary.get(
                "compliance_checks_total", 0
            ),
            thresholds_breached=summary.get("thresholds_breached", 0),
            triggers_active=summary.get("triggers_active", 0),
            alerts_generated=summary.get("alerts_generated", 0),
            critical_alerts=summary.get("critical_alerts", 0),
            immediate_action_required=summary.get(
                "immediate_action_required", False
            ),
        )

    def _build_config(
        self, input_data: DowngradeMonitoringInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return input_data.model_dump()

    def _build_summary(
        self, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        compliance = context.get_phase_output("compliance_check")
        threshold = context.get_phase_output("threshold_monitoring")
        risk = context.get_phase_output("risk_scoring")
        alert = context.get_phase_output("alert_generation")

        return {
            "product_name": compliance.get("product_name", ""),
            "current_classification": compliance.get(
                "current_classification", "ARTICLE_9"
            ),
            "risk_level": risk.get(
                "risk_level", DowngradeRiskLevel.LOW.value
            ),
            "risk_score": risk.get("risk_score", 0.0),
            "compliance_checks_passed": compliance.get(
                "checks_passed", 0
            ),
            "compliance_checks_total": compliance.get(
                "checks_total", 0
            ),
            "thresholds_breached": threshold.get(
                "thresholds_breached", 0
            ),
            "triggers_active": risk.get("triggers_active", 0),
            "alerts_generated": alert.get("alerts_count", 0),
            "critical_alerts": alert.get("critical_alerts", 0),
            "immediate_action_required": risk.get(
                "immediate_action_required", False
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
                logger.debug(
                    "Progress callback failed for phase=%s", phase
                )
