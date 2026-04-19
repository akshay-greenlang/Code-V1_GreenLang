# -*- coding: utf-8 -*-
"""
Annual Compliance Review Workflow
======================================

Five-phase workflow that performs a comprehensive year-end compliance review
across all four constituent regulation packs (CSRD, CBAM, EU Taxonomy, EUDR),
analyzes year-over-year trends, generates a board-ready summary report,
and proposes an action plan for the next period.

Phases:
    1. YearEndCollection - Collect year-end results from all 4 packs
    2. MultiRegulationReview - Review compliance status across regulations
    3. TrendAnalysis - Analyze year-over-year trends
    4. BoardReport - Generate board-ready summary
    5. ActionPlan - Propose action plan for next period

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class RegulationPack(str, Enum):
    """Constituent regulation packs in the bundle."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EU_TAXONOMY = "EU_TAXONOMY"
    EUDR = "EUDR"


class ComplianceLevel(str, Enum):
    """Compliance level for a regulation."""
    FULLY_COMPLIANT = "FULLY_COMPLIANT"
    SUBSTANTIALLY_COMPLIANT = "SUBSTANTIALLY_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_ASSESSED = "NOT_ASSESSED"


class TrendDirection(str, Enum):
    """Direction of a year-over-year trend."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    NEW = "NEW"


class ActionPriority(str, Enum):
    """Priority for action plan items."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ActionCategory(str, Enum):
    """Category of action plan item."""
    REGULATORY_CHANGE = "REGULATORY_CHANGE"
    DATA_IMPROVEMENT = "DATA_IMPROVEMENT"
    PROCESS_ENHANCEMENT = "PROCESS_ENHANCEMENT"
    TECHNOLOGY_UPGRADE = "TECHNOLOGY_UPGRADE"
    TRAINING = "TRAINING"
    GOVERNANCE = "GOVERNANCE"
    REPORTING = "REPORTING"
    REMEDIATION = "REMEDIATION"


# =============================================================================
# REVIEW CONFIGURATION
# =============================================================================


PACK_REVIEW_METRICS: Dict[str, List[Dict[str, Any]]] = {
    RegulationPack.CSRD.value: [
        {"metric_id": "CSRD-M01", "name": "ESRS disclosure completeness", "unit": "%", "target": 100.0, "weight": 0.20},
        {"metric_id": "CSRD-M02", "name": "GHG emissions measurement coverage", "unit": "%", "target": 95.0, "weight": 0.15},
        {"metric_id": "CSRD-M03", "name": "Data quality score", "unit": "score", "target": 90.0, "weight": 0.15},
        {"metric_id": "CSRD-M04", "name": "Assurance readiness", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "CSRD-M05", "name": "Transition plan progress", "unit": "%", "target": 80.0, "weight": 0.10},
        {"metric_id": "CSRD-M06", "name": "Value chain data coverage", "unit": "%", "target": 70.0, "weight": 0.10},
        {"metric_id": "CSRD-M07", "name": "Filing timeliness", "unit": "days_early", "target": 14.0, "weight": 0.10},
        {"metric_id": "CSRD-M08", "name": "Stakeholder engagement score", "unit": "score", "target": 75.0, "weight": 0.05},
    ],
    RegulationPack.CBAM.value: [
        {"metric_id": "CBAM-M01", "name": "Quarterly report timeliness", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "CBAM-M02", "name": "Embedded emissions accuracy", "unit": "%", "target": 95.0, "weight": 0.20},
        {"metric_id": "CBAM-M03", "name": "Supplier data completeness", "unit": "%", "target": 90.0, "weight": 0.15},
        {"metric_id": "CBAM-M04", "name": "Certificate management efficiency", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "CBAM-M05", "name": "Verification coverage", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "CBAM-M06", "name": "CN code classification accuracy", "unit": "%", "target": 99.0, "weight": 0.10},
        {"metric_id": "CBAM-M07", "name": "Declaration accuracy", "unit": "%", "target": 100.0, "weight": 0.10},
    ],
    RegulationPack.EU_TAXONOMY.value: [
        {"metric_id": "TAX-M01", "name": "Eligibility screening coverage", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "TAX-M02", "name": "Alignment assessment completeness", "unit": "%", "target": 100.0, "weight": 0.20},
        {"metric_id": "TAX-M03", "name": "KPI calculation accuracy", "unit": "%", "target": 98.0, "weight": 0.15},
        {"metric_id": "TAX-M04", "name": "DNSH assessment coverage", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "TAX-M05", "name": "Minimum safeguards compliance", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "TAX-M06", "name": "Taxonomy-aligned revenue growth", "unit": "%", "target": 5.0, "weight": 0.10},
        {"metric_id": "TAX-M07", "name": "Third-party review completion", "unit": "%", "target": 100.0, "weight": 0.10},
    ],
    RegulationPack.EUDR.value: [
        {"metric_id": "EUDR-M01", "name": "Geolocation data completeness", "unit": "%", "target": 100.0, "weight": 0.20},
        {"metric_id": "EUDR-M02", "name": "Supply chain traceability coverage", "unit": "%", "target": 95.0, "weight": 0.20},
        {"metric_id": "EUDR-M03", "name": "Risk assessment currency", "unit": "days_old", "target": 30.0, "weight": 0.15},
        {"metric_id": "EUDR-M04", "name": "Due diligence statement completion", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "EUDR-M05", "name": "Deforestation-free verification", "unit": "%", "target": 100.0, "weight": 0.15},
        {"metric_id": "EUDR-M06", "name": "Monitoring system effectiveness", "unit": "%", "target": 90.0, "weight": 0.10},
        {"metric_id": "EUDR-M07", "name": "Staff training completion", "unit": "%", "target": 100.0, "weight": 0.05},
    ],
}

UPCOMING_REGULATORY_CHANGES: List[Dict[str, Any]] = [
    {"regulation": "CSRD", "change": "ESRS sector-specific standards adoption", "expected_year": 2027, "impact": "HIGH", "description": "Sector-specific ESRS standards will require additional disclosures"},
    {"regulation": "CSRD", "change": "Reasonable assurance requirement", "expected_year": 2028, "impact": "HIGH", "description": "Shift from limited to reasonable assurance"},
    {"regulation": "CBAM", "change": "Full CBAM regime activation", "expected_year": 2026, "impact": "CRITICAL", "description": "End of transitional period, full financial obligations"},
    {"regulation": "CBAM", "change": "Scope expansion to indirect emissions", "expected_year": 2027, "impact": "MEDIUM", "description": "Potential inclusion of more Scope 2/3 emissions"},
    {"regulation": "EU_TAXONOMY", "change": "Remaining environmental objectives", "expected_year": 2027, "impact": "MEDIUM", "description": "Full adoption of all 6 environmental objectives"},
    {"regulation": "EUDR", "change": "Full enforcement for SMEs", "expected_year": 2027, "impact": "HIGH", "description": "Smaller operators required to comply"},
    {"regulation": "EUDR", "change": "Country benchmarking system operational", "expected_year": 2026, "impact": "MEDIUM", "description": "Official country risk categorization deployed"},
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
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
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class WorkflowConfig(BaseModel):
    """Configuration for annual compliance review workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2024, le=2050)
    target_packs: List[RegulationPack] = Field(
        default_factory=lambda: list(RegulationPack)
    )
    current_year_data: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-pack metric values for current year"
    )
    prior_year_data: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-pack metric values for prior year"
    )
    board_presentation_date: Optional[str] = Field(
        None,
        description="ISO date for board presentation"
    )
    next_year_budget_eur: Optional[float] = Field(
        None, ge=0,
        description="Budget for next year's compliance activities"
    )
    skip_phases: List[str] = Field(default_factory=list)


class AnnualComplianceReviewResult(WorkflowResult):
    """Result from annual compliance review workflow."""
    packs_reviewed: int = Field(default=0)
    overall_compliance_score: float = Field(default=0.0)
    trends_analyzed: int = Field(default=0)
    action_items: int = Field(default=0)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class AnnualComplianceReviewWorkflow:
    """
    Five-phase annual compliance review workflow.

    Collects year-end results, reviews compliance status, analyzes
    trends, generates board report, and proposes action plan.

    Example:
        >>> wf = AnnualComplianceReviewWorkflow()
        >>> config = WorkflowConfig(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ... )
        >>> result = wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "annual_compliance_review"

    PHASE_ORDER = [
        "year_end_collection",
        "multi_regulation_review",
        "trend_analysis",
        "board_report",
        "action_plan",
    ]

    def __init__(self) -> None:
        """Initialize the annual compliance review workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._phase_outputs: Dict[str, Dict[str, Any]] = {}

    def execute(self, config: WorkflowConfig) -> AnnualComplianceReviewResult:
        """
        Execute the five-phase annual compliance review workflow.

        Args:
            config: Validated workflow configuration.

        Returns:
            AnnualComplianceReviewResult with review outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting annual compliance review %s for org=%s year=%d",
            self.workflow_id, config.organization_id, config.reporting_year,
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        phase_methods = {
            "year_end_collection": self._phase_year_end_collection,
            "multi_regulation_review": self._phase_multi_regulation_review,
            "trend_analysis": self._phase_trend_analysis,
            "board_report": self._phase_board_report,
            "action_plan": self._phase_action_plan,
        }

        for phase_name in self.PHASE_ORDER:
            if phase_name in config.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                continue

            try:
                phase_result = phase_methods[phase_name](config)
                completed_phases.append(phase_result)
                if phase_result.status == PhaseStatus.COMPLETED:
                    self._phase_outputs[phase_name] = phase_result.outputs
                elif phase_result.status == PhaseStatus.FAILED:
                    overall_status = WorkflowStatus.FAILED
                    break
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = datetime.utcnow()
        summary = self._build_summary()
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return AnnualComplianceReviewResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            packs_reviewed=summary.get("packs_reviewed", 0),
            overall_compliance_score=summary.get("overall_compliance_score", 0.0),
            trends_analyzed=summary.get("trends_analyzed", 0),
            action_items=summary.get("action_items", 0),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Year-End Collection
    # -------------------------------------------------------------------------

    def _phase_year_end_collection(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 1: Collect year-end results from all 4 packs.

        Gathers metric values for each pack, validates completeness,
        and prepares data for review and trend analysis.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            pack_collections: Dict[str, Dict[str, Any]] = {}
            total_metrics = 0
            metrics_with_data = 0

            for pack in config.target_packs:
                pack_name = pack.value
                metrics = PACK_REVIEW_METRICS.get(pack_name, [])
                pack_data = config.current_year_data.get(pack_name, {})
                prior_data = config.prior_year_data.get(pack_name, {})

                metric_results: List[Dict[str, Any]] = []
                for metric in metrics:
                    metric_id = metric["metric_id"]
                    current_value = pack_data.get(metric_id)
                    prior_value = prior_data.get(metric_id)
                    has_data = current_value is not None

                    metric_result = {
                        "metric_id": metric_id,
                        "name": metric["name"],
                        "unit": metric["unit"],
                        "target": metric["target"],
                        "weight": metric["weight"],
                        "current_value": current_value,
                        "prior_value": prior_value,
                        "has_data": has_data,
                        "meets_target": (
                            self._check_meets_target(current_value, metric)
                            if has_data else False
                        ),
                    }
                    metric_results.append(metric_result)
                    total_metrics += 1
                    if has_data:
                        metrics_with_data += 1

                pack_completeness = (
                    sum(1 for m in metric_results if m["has_data"]) /
                    max(len(metric_results), 1) * 100
                )

                pack_collections[pack_name] = {
                    "pack": pack_name,
                    "metrics": metric_results,
                    "total_metrics": len(metric_results),
                    "metrics_with_data": sum(1 for m in metric_results if m["has_data"]),
                    "metrics_meeting_target": sum(1 for m in metric_results if m.get("meets_target")),
                    "completeness_pct": round(pack_completeness, 2),
                    "collected_at": datetime.utcnow().isoformat(),
                }

                if pack_completeness < 100:
                    missing = [m["name"] for m in metric_results if not m["has_data"]]
                    warnings.append(
                        f"{pack_name}: {len(missing)} metrics missing data"
                    )

            outputs["pack_collections"] = pack_collections
            outputs["total_metrics"] = total_metrics
            outputs["metrics_with_data"] = metrics_with_data
            outputs["overall_completeness_pct"] = round(
                (metrics_with_data / max(total_metrics, 1)) * 100, 2
            )

            logger.info(
                "Year-end collection complete: %d metrics, %d with data (%.1f%%)",
                total_metrics, metrics_with_data,
                outputs["overall_completeness_pct"],
            )

            status = PhaseStatus.COMPLETED
            records = total_metrics

        except Exception as exc:
            logger.error("Year-end collection failed: %s", exc, exc_info=True)
            errors.append(f"Year-end collection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="year_end_collection",
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

    def _check_meets_target(self, value: Any, metric: Dict[str, Any]) -> bool:
        """Check if a metric value meets its target."""
        if value is None:
            return False
        target = metric["target"]
        unit = metric["unit"]

        if not isinstance(value, (int, float)):
            return False

        if unit == "days_old":
            return value <= target
        elif unit == "days_early":
            return value >= target
        else:
            return value >= target

    # -------------------------------------------------------------------------
    # Phase 2: Multi-Regulation Review
    # -------------------------------------------------------------------------

    def _phase_multi_regulation_review(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 2: Review compliance status across all regulations.

        Computes weighted compliance scores per pack and determines
        overall compliance level.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            collection_out = self._phase_outputs.get("year_end_collection", {})
            pack_collections = collection_out.get("pack_collections", {})

            pack_reviews: Dict[str, Dict[str, Any]] = {}
            pack_scores: List[float] = []

            for pack_name, collection in pack_collections.items():
                metrics = collection.get("metrics", [])

                weighted_sum = 0.0
                weight_sum = 0.0
                for m in metrics:
                    if m["has_data"] and m["current_value"] is not None:
                        target = m["target"]
                        value = m["current_value"]
                        weight = m["weight"]

                        if isinstance(value, (int, float)) and isinstance(target, (int, float)):
                            if m["unit"] == "days_old":
                                ratio = max(0, min(1.0, target / max(value, 0.001)))
                            elif m["unit"] == "days_early":
                                ratio = min(1.0, value / max(target, 0.001))
                            else:
                                ratio = min(1.0, value / max(target, 0.001))
                        else:
                            ratio = 0.5

                        weighted_sum += ratio * weight
                        weight_sum += weight

                pack_score = (weighted_sum / max(weight_sum, 0.001)) * 100
                pack_score = round(min(pack_score, 100.0), 2)

                compliance_level = self._score_to_compliance(pack_score)

                pack_reviews[pack_name] = {
                    "pack": pack_name,
                    "compliance_score": pack_score,
                    "compliance_level": compliance_level,
                    "metrics_assessed": len(metrics),
                    "metrics_meeting_target": collection.get("metrics_meeting_target", 0),
                    "completeness_pct": collection.get("completeness_pct", 0),
                    "key_findings": self._generate_findings(metrics, pack_name),
                    "reviewed_at": datetime.utcnow().isoformat(),
                }
                pack_scores.append(pack_score)

            overall_score = sum(pack_scores) / max(len(pack_scores), 1)
            overall_score = round(overall_score, 2)
            overall_level = self._score_to_compliance(overall_score)

            outputs["pack_reviews"] = pack_reviews
            outputs["packs_reviewed"] = len(pack_reviews)
            outputs["overall_compliance_score"] = overall_score
            outputs["overall_compliance_level"] = overall_level
            outputs["pack_scores"] = {
                p: r["compliance_score"] for p, r in pack_reviews.items()
            }
            outputs["weakest_pack"] = min(
                pack_reviews.items(),
                key=lambda x: x[1]["compliance_score"]
            )[0] if pack_reviews else None
            outputs["strongest_pack"] = max(
                pack_reviews.items(),
                key=lambda x: x[1]["compliance_score"]
            )[0] if pack_reviews else None

            logger.info(
                "Multi-regulation review complete: overall_score=%.2f level=%s",
                overall_score, overall_level,
            )

            status = PhaseStatus.COMPLETED
            records = len(pack_reviews)

        except Exception as exc:
            logger.error("Multi-regulation review failed: %s", exc, exc_info=True)
            errors.append(f"Multi-regulation review failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="multi_regulation_review",
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

    def _score_to_compliance(self, score: float) -> str:
        """Convert numeric score to compliance level."""
        if score >= 90:
            return ComplianceLevel.FULLY_COMPLIANT.value
        elif score >= 70:
            return ComplianceLevel.SUBSTANTIALLY_COMPLIANT.value
        elif score >= 50:
            return ComplianceLevel.PARTIALLY_COMPLIANT.value
        else:
            return ComplianceLevel.NON_COMPLIANT.value

    def _generate_findings(
        self,
        metrics: List[Dict[str, Any]],
        pack_name: str,
    ) -> List[Dict[str, Any]]:
        """Generate key findings from metric results."""
        findings: List[Dict[str, Any]] = []

        below_target = [
            m for m in metrics
            if m["has_data"] and not m.get("meets_target", False)
        ]
        for m in below_target:
            findings.append({
                "type": "below_target",
                "metric": m["name"],
                "current": m["current_value"],
                "target": m["target"],
                "message": (
                    f"{m['name']} at {m['current_value']} {m['unit']} "
                    f"vs target {m['target']} {m['unit']}"
                ),
            })

        above_target = [
            m for m in metrics
            if m["has_data"] and m.get("meets_target", False)
        ]
        if above_target:
            findings.append({
                "type": "achievement",
                "metric": "multiple",
                "message": (
                    f"{len(above_target)}/{len(metrics)} metrics meeting or exceeding targets"
                ),
            })

        missing = [m for m in metrics if not m["has_data"]]
        if missing:
            findings.append({
                "type": "data_gap",
                "metric": "multiple",
                "message": (
                    f"{len(missing)} metrics missing data: "
                    + ", ".join(m["name"] for m in missing[:3])
                    + ("..." if len(missing) > 3 else "")
                ),
            })

        return findings

    # -------------------------------------------------------------------------
    # Phase 3: Trend Analysis
    # -------------------------------------------------------------------------

    def _phase_trend_analysis(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 3: Analyze year-over-year trends.

        Compares current year metrics with prior year to identify
        improving, stable, and declining areas.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            collection_out = self._phase_outputs.get("year_end_collection", {})
            pack_collections = collection_out.get("pack_collections", {})

            trend_results: Dict[str, List[Dict[str, Any]]] = {}
            total_trends = 0
            improving_count = 0
            stable_count = 0
            declining_count = 0
            new_count = 0

            for pack_name, collection in pack_collections.items():
                metrics = collection.get("metrics", [])
                pack_trends: List[Dict[str, Any]] = []

                for m in metrics:
                    current = m.get("current_value")
                    prior = m.get("prior_value")

                    if current is None:
                        continue

                    if prior is None:
                        direction = TrendDirection.NEW.value
                        change_pct = 0.0
                        new_count += 1
                    elif isinstance(current, (int, float)) and isinstance(prior, (int, float)):
                        if prior != 0:
                            change_pct = ((current - prior) / abs(prior)) * 100
                        else:
                            change_pct = 100.0 if current > 0 else 0.0

                        if m["unit"] == "days_old":
                            if change_pct < -5:
                                direction = TrendDirection.IMPROVING.value
                            elif change_pct > 5:
                                direction = TrendDirection.DECLINING.value
                            else:
                                direction = TrendDirection.STABLE.value
                        else:
                            if change_pct > 5:
                                direction = TrendDirection.IMPROVING.value
                            elif change_pct < -5:
                                direction = TrendDirection.DECLINING.value
                            else:
                                direction = TrendDirection.STABLE.value

                        change_pct = round(change_pct, 2)
                    else:
                        if str(current) == str(prior):
                            direction = TrendDirection.STABLE.value
                        else:
                            direction = TrendDirection.DECLINING.value
                        change_pct = 0.0

                    if direction == TrendDirection.IMPROVING.value:
                        improving_count += 1
                    elif direction == TrendDirection.STABLE.value:
                        stable_count += 1
                    elif direction == TrendDirection.DECLINING.value:
                        declining_count += 1

                    trend = {
                        "metric_id": m["metric_id"],
                        "name": m["name"],
                        "unit": m["unit"],
                        "current_value": current,
                        "prior_value": prior,
                        "change_pct": change_pct,
                        "direction": direction,
                        "target": m["target"],
                        "meets_target_current": m.get("meets_target", False),
                    }
                    pack_trends.append(trend)
                    total_trends += 1

                trend_results[pack_name] = pack_trends

            outputs["trend_results"] = trend_results
            outputs["total_trends"] = total_trends
            outputs["improving_count"] = improving_count
            outputs["stable_count"] = stable_count
            outputs["declining_count"] = declining_count
            outputs["new_count"] = new_count

            outputs["trend_summary"] = {
                "overall_direction": (
                    TrendDirection.IMPROVING.value if improving_count > declining_count
                    else TrendDirection.DECLINING.value if declining_count > improving_count
                    else TrendDirection.STABLE.value
                ),
                "improvement_rate": round(
                    (improving_count / max(total_trends, 1)) * 100, 2
                ),
                "decline_rate": round(
                    (declining_count / max(total_trends, 1)) * 100, 2
                ),
            }

            declining_items: List[Dict[str, Any]] = []
            for pack_name, trends in trend_results.items():
                for t in trends:
                    if t["direction"] == TrendDirection.DECLINING.value:
                        declining_items.append({
                            "pack": pack_name,
                            "metric": t["name"],
                            "change_pct": t["change_pct"],
                        })
            outputs["declining_metrics"] = declining_items

            if declining_items:
                warnings.append(
                    f"{len(declining_items)} metrics showing declining trends"
                )

            logger.info(
                "Trend analysis complete: %d trends, %d improving, %d stable, %d declining",
                total_trends, improving_count, stable_count, declining_count,
            )

            status = PhaseStatus.COMPLETED
            records = total_trends

        except Exception as exc:
            logger.error("Trend analysis failed: %s", exc, exc_info=True)
            errors.append(f"Trend analysis failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="trend_analysis",
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

    # -------------------------------------------------------------------------
    # Phase 4: Board Report
    # -------------------------------------------------------------------------

    def _phase_board_report(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 4: Generate board-ready summary report.

        Creates a structured executive summary suitable for
        board-level presentation.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            review_out = self._phase_outputs.get("multi_regulation_review", {})
            trend_out = self._phase_outputs.get("trend_analysis", {})
            pack_reviews = review_out.get("pack_reviews", {})
            trend_summary = trend_out.get("trend_summary", {})
            declining_metrics = trend_out.get("declining_metrics", [])

            report_id = str(uuid.uuid4())

            executive_summary = {
                "reporting_year": config.reporting_year,
                "organization_id": config.organization_id,
                "overall_compliance_score": review_out.get("overall_compliance_score", 0.0),
                "overall_compliance_level": review_out.get("overall_compliance_level", "NOT_ASSESSED"),
                "regulations_covered": len(pack_reviews),
                "overall_trend": trend_summary.get("overall_direction", "STABLE"),
                "improvement_rate_pct": trend_summary.get("improvement_rate", 0.0),
                "key_risk_areas": len(declining_metrics),
            }

            regulation_scorecards: List[Dict[str, Any]] = []
            for pack_name, review in pack_reviews.items():
                scorecard = {
                    "regulation": pack_name,
                    "score": review["compliance_score"],
                    "level": review["compliance_level"],
                    "metrics_assessed": review["metrics_assessed"],
                    "targets_met": review["metrics_meeting_target"],
                    "completeness": review["completeness_pct"],
                    "key_findings_count": len(review.get("key_findings", [])),
                }
                regulation_scorecards.append(scorecard)

            risk_highlights: List[Dict[str, Any]] = []
            for dec in declining_metrics[:5]:
                risk_highlights.append({
                    "regulation": dec["pack"],
                    "metric": dec["metric"],
                    "change": f"{dec['change_pct']:+.1f}%",
                    "severity": "HIGH" if abs(dec["change_pct"]) > 20 else "MEDIUM",
                })

            relevant_changes = [
                ch for ch in UPCOMING_REGULATORY_CHANGES
                if ch["expected_year"] <= config.reporting_year + 2
                and ch["regulation"] in [p.value for p in config.target_packs]
            ]

            board_report = {
                "report_id": report_id,
                "report_type": "annual_compliance_review",
                "executive_summary": executive_summary,
                "regulation_scorecards": regulation_scorecards,
                "risk_highlights": risk_highlights,
                "upcoming_regulatory_changes": relevant_changes,
                "prepared_for": config.board_presentation_date or "TBD",
                "generated_at": datetime.utcnow().isoformat(),
                "confidentiality": "BOARD CONFIDENTIAL",
            }

            outputs["board_report"] = board_report
            outputs["report_id"] = report_id
            outputs["scorecard_count"] = len(regulation_scorecards)
            outputs["risk_count"] = len(risk_highlights)
            outputs["regulatory_changes"] = len(relevant_changes)

            logger.info(
                "Board report generated: report_id=%s scorecards=%d risks=%d",
                report_id, len(regulation_scorecards), len(risk_highlights),
            )

            status = PhaseStatus.COMPLETED
            records = 1

        except Exception as exc:
            logger.error("Board report failed: %s", exc, exc_info=True)
            errors.append(f"Board report failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="board_report",
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

    # -------------------------------------------------------------------------
    # Phase 5: Action Plan
    # -------------------------------------------------------------------------

    def _phase_action_plan(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 5: Propose action plan for next period.

        Creates prioritized action items based on gaps, declining
        trends, and upcoming regulatory changes.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            review_out = self._phase_outputs.get("multi_regulation_review", {})
            trend_out = self._phase_outputs.get("trend_analysis", {})
            board_out = self._phase_outputs.get("board_report", {})
            pack_reviews = review_out.get("pack_reviews", {})
            trend_results = trend_out.get("trend_results", {})
            board_report = board_out.get("board_report", {})
            regulatory_changes = board_report.get("upcoming_regulatory_changes", [])

            action_items: List[Dict[str, Any]] = []
            next_year = config.reporting_year + 1

            for pack_name, review in pack_reviews.items():
                if review["compliance_level"] in (
                    ComplianceLevel.NON_COMPLIANT.value,
                    ComplianceLevel.PARTIALLY_COMPLIANT.value,
                ):
                    action_items.append({
                        "action_id": str(uuid.uuid4()),
                        "pack": pack_name,
                        "category": ActionCategory.REMEDIATION.value,
                        "priority": ActionPriority.CRITICAL.value if review["compliance_score"] < 50 else ActionPriority.HIGH.value,
                        "title": f"Improve {pack_name} compliance to substantial level",
                        "description": (
                            f"Current score: {review['compliance_score']:.1f}%. "
                            f"Target: 70%+ for substantial compliance."
                        ),
                        "target_date": f"{next_year}-Q2",
                        "estimated_effort_days": 60,
                        "estimated_cost_eur": 48000.0,
                    })

                findings = review.get("key_findings", [])
                for finding in findings:
                    if finding.get("type") == "data_gap":
                        action_items.append({
                            "action_id": str(uuid.uuid4()),
                            "pack": pack_name,
                            "category": ActionCategory.DATA_IMPROVEMENT.value,
                            "priority": ActionPriority.HIGH.value,
                            "title": f"Close data gaps in {pack_name}",
                            "description": finding["message"],
                            "target_date": f"{next_year}-Q1",
                            "estimated_effort_days": 20,
                            "estimated_cost_eur": 16000.0,
                        })

            for pack_name, trends in trend_results.items():
                declining = [t for t in trends if t["direction"] == TrendDirection.DECLINING.value]
                for trend in declining:
                    action_items.append({
                        "action_id": str(uuid.uuid4()),
                        "pack": pack_name,
                        "category": ActionCategory.PROCESS_ENHANCEMENT.value,
                        "priority": (
                            ActionPriority.HIGH.value
                            if abs(trend.get("change_pct", 0)) > 20
                            else ActionPriority.MEDIUM.value
                        ),
                        "title": f"Address declining trend: {trend['name']}",
                        "description": (
                            f"{trend['name']} declined {trend['change_pct']:+.1f}% YoY. "
                            f"Current: {trend['current_value']}, Target: {trend['target']}"
                        ),
                        "target_date": f"{next_year}-Q2",
                        "estimated_effort_days": 15,
                        "estimated_cost_eur": 12000.0,
                    })

            for change in regulatory_changes:
                if change["impact"] in ("CRITICAL", "HIGH"):
                    action_items.append({
                        "action_id": str(uuid.uuid4()),
                        "pack": change["regulation"],
                        "category": ActionCategory.REGULATORY_CHANGE.value,
                        "priority": (
                            ActionPriority.CRITICAL.value
                            if change["impact"] == "CRITICAL"
                            else ActionPriority.HIGH.value
                        ),
                        "title": f"Prepare for: {change['change']}",
                        "description": (
                            f"{change['description']}. "
                            f"Expected: {change['expected_year']}."
                        ),
                        "target_date": f"{change['expected_year']}-Q1",
                        "estimated_effort_days": 30,
                        "estimated_cost_eur": 24000.0,
                    })

            action_items.append({
                "action_id": str(uuid.uuid4()),
                "pack": "ALL",
                "category": ActionCategory.TRAINING.value,
                "priority": ActionPriority.MEDIUM.value,
                "title": "Annual compliance training refresh",
                "description": "Refresh training on CSRD, CBAM, EU Taxonomy, and EUDR requirements",
                "target_date": f"{next_year}-Q1",
                "estimated_effort_days": 5,
                "estimated_cost_eur": 4000.0,
            })

            action_items.append({
                "action_id": str(uuid.uuid4()),
                "pack": "ALL",
                "category": ActionCategory.GOVERNANCE.value,
                "priority": ActionPriority.MEDIUM.value,
                "title": "Review and update compliance governance framework",
                "description": "Annual review of roles, responsibilities, and escalation procedures",
                "target_date": f"{next_year}-Q1",
                "estimated_effort_days": 10,
                "estimated_cost_eur": 8000.0,
            })

            priority_order = {
                ActionPriority.CRITICAL.value: 0,
                ActionPriority.HIGH.value: 1,
                ActionPriority.MEDIUM.value: 2,
                ActionPriority.LOW.value: 3,
            }
            action_items.sort(key=lambda a: priority_order.get(a["priority"], 99))

            by_priority: Dict[str, int] = {}
            for a in action_items:
                p = a["priority"]
                by_priority[p] = by_priority.get(p, 0) + 1

            by_category: Dict[str, int] = {}
            for a in action_items:
                c = a["category"]
                by_category[c] = by_category.get(c, 0) + 1

            total_effort = sum(a["estimated_effort_days"] for a in action_items)
            total_cost = sum(a["estimated_cost_eur"] for a in action_items)

            outputs["action_items"] = action_items
            outputs["total_actions"] = len(action_items)
            outputs["by_priority"] = by_priority
            outputs["by_category"] = by_category
            outputs["total_effort_days"] = total_effort
            outputs["total_estimated_cost_eur"] = round(total_cost, 2)

            if config.next_year_budget_eur is not None:
                outputs["budget_eur"] = config.next_year_budget_eur
                outputs["budget_sufficient"] = total_cost <= config.next_year_budget_eur
                outputs["budget_gap_eur"] = round(
                    max(0, total_cost - config.next_year_budget_eur), 2
                )

            outputs["plan_summary"] = {
                "next_year": next_year,
                "total_actions": len(action_items),
                "critical_actions": by_priority.get(ActionPriority.CRITICAL.value, 0),
                "total_effort_days": total_effort,
                "total_cost_eur": round(total_cost, 2),
            }

            logger.info(
                "Action plan complete: %d actions, %d days effort, EUR %.2f",
                len(action_items), total_effort, total_cost,
            )

            status = PhaseStatus.COMPLETED
            records = len(action_items)

        except Exception as exc:
            logger.error("Action plan failed: %s", exc, exc_info=True)
            errors.append(f"Action plan failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="action_plan",
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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build workflow summary from all phase outputs."""
        collection = self._phase_outputs.get("year_end_collection", {})
        review = self._phase_outputs.get("multi_regulation_review", {})
        trends = self._phase_outputs.get("trend_analysis", {})
        board = self._phase_outputs.get("board_report", {})
        actions = self._phase_outputs.get("action_plan", {})

        return {
            "packs_reviewed": review.get("packs_reviewed", 0),
            "overall_compliance_score": review.get("overall_compliance_score", 0.0),
            "overall_compliance_level": review.get("overall_compliance_level", "NOT_ASSESSED"),
            "metrics_collected": collection.get("total_metrics", 0),
            "completeness_pct": collection.get("overall_completeness_pct", 0.0),
            "trends_analyzed": trends.get("total_trends", 0),
            "improving_trends": trends.get("improving_count", 0),
            "declining_trends": trends.get("declining_count", 0),
            "action_items": actions.get("total_actions", 0),
            "total_effort_days": actions.get("total_effort_days", 0),
            "total_cost_eur": actions.get("total_estimated_cost_eur", 0.0),
        }


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
