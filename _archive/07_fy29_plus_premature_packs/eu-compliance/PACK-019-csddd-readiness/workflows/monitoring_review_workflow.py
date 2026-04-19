# -*- coding: utf-8 -*-
"""
CSDDD Monitoring and Review Workflow
===============================================

4-phase workflow for monitoring due diligence effectiveness and conducting
periodic reviews under the EU Corporate Sustainability Due Diligence Directive
(CSDDD / CS3D). Defines KPIs, collects monitoring data, analyses performance,
and produces annual review recommendations.

Phases:
    1. KPIDefinition           -- Define and validate monitoring KPIs
    2. DataCollection          -- Collect and validate monitoring data
    3. PerformanceAnalysis     -- Analyse trends and performance against targets
    4. AnnualReview            -- Produce annual review findings and recommendations

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Art. 12: Monitoring
    - Art. 12(1): Periodic assessment of own operations and subsidiaries
    - Art. 12(2): Assessment of business relationships
    - Art. 12(3): Updating due diligence measures
    - Art. 13: Communicating
    - Art. 14: Reporting under CSRD

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class WorkflowPhase(str, Enum):
    """Phases of the monitoring review workflow."""
    KPI_DEFINITION = "kpi_definition"
    DATA_COLLECTION = "data_collection"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    ANNUAL_REVIEW = "annual_review"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class KPICategory(str, Enum):
    """Categories of monitoring KPIs."""
    IMPACT_REDUCTION = "impact_reduction"
    PREVENTION_EFFECTIVENESS = "prevention_effectiveness"
    GRIEVANCE_PERFORMANCE = "grievance_performance"
    SUPPLIER_COMPLIANCE = "supplier_compliance"
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"
    POLICY_IMPLEMENTATION = "policy_implementation"
    CLIMATE_TRANSITION = "climate_transition"

class KPIStatus(str, Enum):
    """KPI target achievement status."""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    ACHIEVED = "achieved"
    NOT_STARTED = "not_started"

class TrendDirection(str, Enum):
    """Trend direction for performance analysis."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    INSUFFICIENT_DATA = "insufficient_data"

class ReviewOutcome(str, Enum):
    """Outcome of the annual review."""
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    UNSATISFACTORY = "unsatisfactory"
    MAJOR_REVISION_REQUIRED = "major_revision_required"

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class KPIDefinition(BaseModel):
    """Definition of a monitoring KPI."""
    kpi_id: str = Field(default_factory=lambda: f"kpi-{_new_uuid()[:8]}")
    kpi_name: str = Field(default="", description="KPI label")
    description: str = Field(default="", description="What this KPI measures")
    category: KPICategory = Field(default=KPICategory.IMPACT_REDUCTION)
    unit: str = Field(default="", description="Measurement unit (%, count, days, etc.)")
    target_value: float = Field(default=0.0, description="Target value")
    baseline_value: float = Field(default=0.0, description="Baseline value")
    direction: str = Field(default="decrease", description="decrease or increase is better")
    frequency: str = Field(default="quarterly", description="Measurement frequency")
    csddd_article: str = Field(default="art_12", description="Relevant CSDDD article")
    responsible: str = Field(default="", description="Responsible department/person")

class MonitoringDataPoint(BaseModel):
    """Single data point for KPI monitoring."""
    data_point_id: str = Field(default_factory=lambda: f"dp-{_new_uuid()[:8]}")
    kpi_id: str = Field(default="", description="Associated KPI ID")
    period: str = Field(default="", description="Reporting period (e.g., 2026-Q1)")
    value: float = Field(default=0.0, description="Measured value")
    source: str = Field(default="", description="Data source")
    verified: bool = Field(default=False, description="Data verified by independent party")
    notes: str = Field(default="")

class PreviousReview(BaseModel):
    """Summary of a previous annual review for comparison."""
    review_year: int = Field(default=0, ge=0)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    kpis_on_track: int = Field(default=0, ge=0)
    kpis_total: int = Field(default=0, ge=0)
    key_findings: List[str] = Field(default_factory=list)
    outstanding_actions: List[str] = Field(default_factory=list)

class MonitoringReviewInput(BaseModel):
    """Input data model for MonitoringReviewWorkflow."""
    entity_id: str = Field(default="", description="Reporting entity ID")
    entity_name: str = Field(default="", description="Reporting entity name")
    reporting_year: int = Field(default=2026, ge=2024, le=2050)
    kpi_definitions: List[KPIDefinition] = Field(
        default_factory=list, description="KPI definitions to monitor"
    )
    monitoring_data: List[MonitoringDataPoint] = Field(
        default_factory=list, description="Collected monitoring data points"
    )
    previous_review: Optional[PreviousReview] = Field(
        default=None, description="Previous year review for comparison"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class KPIResult(BaseModel):
    """Analysis result for a single KPI."""
    kpi_id: str = Field(...)
    kpi_name: str = Field(default="")
    category: str = Field(default="")
    target_value: float = Field(default=0.0)
    current_value: float = Field(default=0.0)
    baseline_value: float = Field(default=0.0)
    achievement_pct: float = Field(default=0.0, ge=0.0)
    status: KPIStatus = Field(default=KPIStatus.NOT_STARTED)
    trend: TrendDirection = Field(default=TrendDirection.INSUFFICIENT_DATA)
    data_points_count: int = Field(default=0, ge=0)
    data_verified_pct: float = Field(default=0.0, ge=0.0, le=100.0)

class MonitoringReviewResult(BaseModel):
    """Complete result from monitoring review workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="monitoring_review")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    # KPI results
    kpi_results: List[KPIResult] = Field(default_factory=list)
    total_kpis: int = Field(default=0, ge=0)
    kpis_on_track: int = Field(default=0, ge=0)
    kpis_at_risk: int = Field(default=0, ge=0)
    kpis_off_track: int = Field(default=0, ge=0)
    kpis_achieved: int = Field(default=0, ge=0)
    # Performance
    overall_performance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_achievement_pct: float = Field(default=0.0, ge=0.0)
    performance_trends: Dict[str, str] = Field(default_factory=dict)
    # Review
    review_outcome: str = Field(default="needs_improvement")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    year_over_year_change: float = Field(default=0.0)
    reporting_year: int = Field(default=2026)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class MonitoringReviewWorkflow:
    """
    4-phase CSDDD monitoring and review workflow.

    Defines KPIs, collects monitoring data, analyses performance trends, and
    produces annual review findings per Art. 12.

    Zero-hallucination: all performance scores use deterministic arithmetic.
    No LLM in numeric calculation paths.

    Example:
        >>> wf = MonitoringReviewWorkflow()
        >>> inp = MonitoringReviewInput(kpi_definitions=[...], monitoring_data=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_kpis >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MonitoringReviewWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._kpi_results: List[KPIResult] = []
        self._overall_score: float = 0.0
        self._recommendations: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.KPI_DEFINITION.value, "description": "Define and validate KPIs"},
            {"name": WorkflowPhase.DATA_COLLECTION.value, "description": "Collect and validate monitoring data"},
            {"name": WorkflowPhase.PERFORMANCE_ANALYSIS.value, "description": "Analyse performance trends"},
            {"name": WorkflowPhase.ANNUAL_REVIEW.value, "description": "Produce annual review findings"},
        ]

    def validate_inputs(self, input_data: MonitoringReviewInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.kpi_definitions:
            issues.append("No KPI definitions provided")
        if not input_data.monitoring_data:
            issues.append("No monitoring data points provided")
        return issues

    async def execute(
        self,
        input_data: Optional[MonitoringReviewInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> MonitoringReviewResult:
        """
        Execute the 4-phase monitoring review workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            MonitoringReviewResult with KPI results and review recommendations.
        """
        if input_data is None:
            input_data = MonitoringReviewInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting monitoring review workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_kpi_definition(input_data))
            phase_results.append(await self._phase_data_collection(input_data))
            phase_results.append(await self._phase_performance_analysis(input_data))
            phase_results.append(await self._phase_annual_review(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Monitoring review failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        on_track = sum(1 for kr in self._kpi_results if kr.status == KPIStatus.ON_TRACK)
        at_risk = sum(1 for kr in self._kpi_results if kr.status == KPIStatus.AT_RISK)
        off_track = sum(1 for kr in self._kpi_results if kr.status == KPIStatus.OFF_TRACK)
        achieved = sum(1 for kr in self._kpi_results if kr.status == KPIStatus.ACHIEVED)

        avg_achievement = round(
            sum(kr.achievement_pct for kr in self._kpi_results) / len(self._kpi_results), 1
        ) if self._kpi_results else 0.0

        trends = {kr.kpi_id: kr.trend.value for kr in self._kpi_results}

        yoy_change = 0.0
        if input_data.previous_review:
            yoy_change = round(self._overall_score - input_data.previous_review.overall_score, 1)

        review_outcome = self._determine_review_outcome()

        result = MonitoringReviewResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            kpi_results=self._kpi_results,
            total_kpis=len(self._kpi_results),
            kpis_on_track=on_track,
            kpis_at_risk=at_risk,
            kpis_off_track=off_track,
            kpis_achieved=achieved,
            overall_performance_score=self._overall_score,
            avg_achievement_pct=avg_achievement,
            performance_trends=trends,
            review_outcome=review_outcome,
            recommendations=self._recommendations,
            year_over_year_change=yoy_change,
            reporting_year=input_data.reporting_year,
            executed_at=utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Monitoring review %s completed in %.2fs: score=%.1f%%, %d KPIs",
            self.workflow_id, elapsed, self._overall_score, len(self._kpi_results),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: KPI Definition
    # -------------------------------------------------------------------------

    async def _phase_kpi_definition(
        self, input_data: MonitoringReviewInput,
    ) -> PhaseResult:
        """Define and validate monitoring KPIs."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        kpis = input_data.kpi_definitions

        # Validate KPI quality
        valid_kpis = 0
        quality_issues: List[str] = []
        by_category: Dict[str, int] = {}

        for kpi in kpis:
            by_category[kpi.category.value] = by_category.get(kpi.category.value, 0) + 1
            issues = []
            if not kpi.kpi_name:
                issues.append(f"{kpi.kpi_id}: missing name")
            if kpi.target_value == 0 and kpi.baseline_value == 0:
                issues.append(f"{kpi.kpi_id}: target and baseline both zero")
            if not kpi.unit:
                issues.append(f"{kpi.kpi_id}: missing unit")
            if not kpi.responsible:
                issues.append(f"{kpi.kpi_id}: no responsible party assigned")
            if issues:
                quality_issues.extend(issues)
            else:
                valid_kpis += 1

        # Check category coverage
        expected_categories = {c.value for c in KPICategory}
        present_categories = set(by_category.keys())
        missing_categories = expected_categories - present_categories

        outputs["total_kpis"] = len(kpis)
        outputs["valid_kpis"] = valid_kpis
        outputs["quality_issues"] = len(quality_issues)
        outputs["by_category"] = by_category
        outputs["category_coverage_pct"] = round(
            (len(present_categories) / len(expected_categories)) * 100, 1
        ) if expected_categories else 0.0
        outputs["missing_categories"] = list(missing_categories)

        if quality_issues:
            warnings.append(f"{len(quality_issues)} KPI quality issues found")
        if missing_categories:
            warnings.append(f"Missing KPI categories: {', '.join(sorted(missing_categories))}")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 KPIDefinition: %d KPIs, %d valid, %d categories",
            len(kpis), valid_kpis, len(present_categories),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.KPI_DEFINITION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(
        self, input_data: MonitoringReviewInput,
    ) -> PhaseResult:
        """Collect and validate monitoring data points."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data_points = input_data.monitoring_data
        kpi_ids = {kpi.kpi_id for kpi in input_data.kpi_definitions}

        # Validate data points
        valid_points = 0
        orphan_points = 0
        verified_count = 0

        data_by_kpi: Dict[str, List[MonitoringDataPoint]] = {}
        for dp in data_points:
            if dp.kpi_id in kpi_ids:
                data_by_kpi.setdefault(dp.kpi_id, []).append(dp)
                valid_points += 1
            else:
                orphan_points += 1
            if dp.verified:
                verified_count += 1

        # Coverage: which KPIs have data
        kpis_with_data = len(data_by_kpi)
        kpis_without_data = len(kpi_ids) - kpis_with_data

        outputs["total_data_points"] = len(data_points)
        outputs["valid_data_points"] = valid_points
        outputs["orphan_data_points"] = orphan_points
        outputs["verified_data_points"] = verified_count
        outputs["verification_rate_pct"] = round(
            (verified_count / len(data_points)) * 100, 1
        ) if data_points else 0.0
        outputs["kpis_with_data"] = kpis_with_data
        outputs["kpis_without_data"] = kpis_without_data
        outputs["data_coverage_pct"] = round(
            (kpis_with_data / len(kpi_ids)) * 100, 1
        ) if kpi_ids else 0.0
        outputs["avg_points_per_kpi"] = round(
            valid_points / kpis_with_data, 1
        ) if kpis_with_data > 0 else 0.0

        if orphan_points > 0:
            warnings.append(f"{orphan_points} data points reference undefined KPIs")
        if kpis_without_data > 0:
            warnings.append(f"{kpis_without_data} KPIs have no monitoring data")
        if verified_count < len(data_points) * 0.5 and data_points:
            warnings.append("Less than 50% of data points are independently verified")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DataCollection: %d points, %d valid, %.1f%% coverage",
            len(data_points), valid_points, outputs["data_coverage_pct"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.DATA_COLLECTION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Performance Analysis
    # -------------------------------------------------------------------------

    async def _phase_performance_analysis(
        self, input_data: MonitoringReviewInput,
    ) -> PhaseResult:
        """Analyse performance trends and achievement against targets."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._kpi_results = []

        # Build data lookup
        data_by_kpi: Dict[str, List[MonitoringDataPoint]] = {}
        for dp in input_data.monitoring_data:
            data_by_kpi.setdefault(dp.kpi_id, []).append(dp)

        for kpi in input_data.kpi_definitions:
            points = data_by_kpi.get(kpi.kpi_id, [])

            # Sort by period
            points.sort(key=lambda p: p.period)
            values = [p.value for p in points]

            # Current value = latest data point
            current_value = values[-1] if values else kpi.baseline_value

            # Achievement calculation
            if kpi.direction == "decrease":
                # Target is to decrease: achievement = how much reduced vs target reduction
                target_delta = kpi.baseline_value - kpi.target_value
                actual_delta = kpi.baseline_value - current_value
                achievement = round(
                    (actual_delta / target_delta) * 100, 1
                ) if target_delta != 0 else (100.0 if actual_delta >= 0 else 0.0)
            else:
                # Target is to increase
                target_delta = kpi.target_value - kpi.baseline_value
                actual_delta = current_value - kpi.baseline_value
                achievement = round(
                    (actual_delta / target_delta) * 100, 1
                ) if target_delta != 0 else (100.0 if actual_delta >= 0 else 0.0)

            achievement = max(0.0, min(200.0, achievement))  # Cap at 200%

            # Determine status
            if achievement >= 100:
                kpi_status = KPIStatus.ACHIEVED
            elif achievement >= 70:
                kpi_status = KPIStatus.ON_TRACK
            elif achievement >= 40:
                kpi_status = KPIStatus.AT_RISK
            elif len(values) == 0:
                kpi_status = KPIStatus.NOT_STARTED
            else:
                kpi_status = KPIStatus.OFF_TRACK

            # Trend analysis
            trend = self._determine_trend(values, kpi.direction)

            # Verification rate
            verified = sum(1 for p in points if p.verified)
            verified_pct = round(
                (verified / len(points)) * 100, 1
            ) if points else 0.0

            self._kpi_results.append(KPIResult(
                kpi_id=kpi.kpi_id,
                kpi_name=kpi.kpi_name,
                category=kpi.category.value,
                target_value=kpi.target_value,
                current_value=current_value,
                baseline_value=kpi.baseline_value,
                achievement_pct=achievement,
                status=kpi_status,
                trend=trend,
                data_points_count=len(points),
                data_verified_pct=verified_pct,
            ))

        # Overall performance score
        achievements = [kr.achievement_pct for kr in self._kpi_results]
        self._overall_score = round(
            sum(min(a, 100.0) for a in achievements) / len(achievements), 1
        ) if achievements else 0.0

        status_dist = {
            s.value: sum(1 for kr in self._kpi_results if kr.status == s)
            for s in KPIStatus
        }

        outputs["kpis_analysed"] = len(self._kpi_results)
        outputs["overall_performance_score"] = self._overall_score
        outputs["avg_achievement_pct"] = round(
            sum(achievements) / len(achievements), 1
        ) if achievements else 0.0
        outputs["status_distribution"] = status_dist
        outputs["improving_kpis"] = sum(
            1 for kr in self._kpi_results if kr.trend == TrendDirection.IMPROVING
        )
        outputs["declining_kpis"] = sum(
            1 for kr in self._kpi_results if kr.trend == TrendDirection.DECLINING
        )

        if status_dist.get("off_track", 0) > 0:
            warnings.append(
                f"{status_dist['off_track']} KPIs are off track -- corrective action needed"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 PerformanceAnalysis: score=%.1f%%, %d KPIs",
            self._overall_score, len(self._kpi_results),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.PERFORMANCE_ANALYSIS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Annual Review
    # -------------------------------------------------------------------------

    async def _phase_annual_review(
        self, input_data: MonitoringReviewInput,
    ) -> PhaseResult:
        """Produce annual review findings and recommendations."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._recommendations = []

        review_outcome = self._determine_review_outcome()
        prev = input_data.previous_review

        # Year-over-year comparison
        yoy_change = 0.0
        if prev:
            yoy_change = round(self._overall_score - prev.overall_score, 1)

        # Generate recommendations based on KPI results
        for kr in self._kpi_results:
            if kr.status == KPIStatus.OFF_TRACK:
                self._recommendations.append({
                    "kpi_id": kr.kpi_id,
                    "kpi_name": kr.kpi_name,
                    "priority": "high",
                    "recommendation": f"Revise approach for {kr.kpi_name} -- currently at {kr.achievement_pct}% achievement",
                    "suggested_action": "Conduct root cause analysis and update prevention/mitigation measures",
                })
            elif kr.status == KPIStatus.AT_RISK:
                self._recommendations.append({
                    "kpi_id": kr.kpi_id,
                    "kpi_name": kr.kpi_name,
                    "priority": "medium",
                    "recommendation": f"Monitor closely: {kr.kpi_name} at {kr.achievement_pct}% achievement",
                    "suggested_action": "Increase monitoring frequency and allocate additional resources",
                })
            if kr.trend == TrendDirection.DECLINING:
                self._recommendations.append({
                    "kpi_id": kr.kpi_id,
                    "kpi_name": kr.kpi_name,
                    "priority": "high",
                    "recommendation": f"Declining trend detected for {kr.kpi_name}",
                    "suggested_action": "Investigate cause of decline and implement corrective measures per Art. 12(3)",
                })

        # Check for outstanding actions from previous review
        if prev and prev.outstanding_actions:
            for action in prev.outstanding_actions:
                self._recommendations.append({
                    "kpi_id": "n/a",
                    "kpi_name": "Previous review follow-up",
                    "priority": "high",
                    "recommendation": f"Outstanding from {prev.review_year}: {action}",
                    "suggested_action": "Complete outstanding action before next review cycle",
                })

        outputs["review_outcome"] = review_outcome
        outputs["overall_performance_score"] = self._overall_score
        outputs["year_over_year_change"] = yoy_change
        outputs["total_recommendations"] = len(self._recommendations)
        outputs["high_priority_recommendations"] = sum(
            1 for r in self._recommendations if r["priority"] == "high"
        )
        outputs["kpis_requiring_revision"] = sum(
            1 for kr in self._kpi_results if kr.status == KPIStatus.OFF_TRACK
        )
        outputs["next_review_due"] = f"{input_data.reporting_year + 1}-12-31"
        outputs["art_12_compliance"] = self._overall_score >= 50

        if review_outcome in ("unsatisfactory", "major_revision_required"):
            warnings.append(f"Annual review outcome: {review_outcome} -- immediate action required")
        if yoy_change < -10:
            warnings.append(f"Performance declined by {abs(yoy_change)}% year-over-year")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 AnnualReview: outcome=%s, %d recommendations",
            review_outcome, len(self._recommendations),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.ANNUAL_REVIEW.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _determine_trend(values: List[float], direction: str) -> TrendDirection:
        """Determine trend direction from a series of values."""
        if len(values) < 2:
            return TrendDirection.INSUFFICIENT_DATA

        # Simple linear trend: compare first half average to second half average
        mid = len(values) // 2
        first_half = sum(values[:mid]) / mid if mid > 0 else 0
        second_half = sum(values[mid:]) / (len(values) - mid) if (len(values) - mid) > 0 else 0

        delta = second_half - first_half
        threshold = abs(first_half) * 0.05 if first_half != 0 else 0.01

        if direction == "decrease":
            # Decreasing is good
            if delta < -threshold:
                return TrendDirection.IMPROVING
            elif delta > threshold:
                return TrendDirection.DECLINING
            return TrendDirection.STABLE
        else:
            # Increasing is good
            if delta > threshold:
                return TrendDirection.IMPROVING
            elif delta < -threshold:
                return TrendDirection.DECLINING
            return TrendDirection.STABLE

    def _determine_review_outcome(self) -> str:
        """Determine overall review outcome based on performance."""
        if self._overall_score >= 80:
            return ReviewOutcome.SATISFACTORY.value
        elif self._overall_score >= 50:
            return ReviewOutcome.NEEDS_IMPROVEMENT.value
        elif self._overall_score >= 25:
            return ReviewOutcome.UNSATISFACTORY.value
        return ReviewOutcome.MAJOR_REVISION_REQUIRED.value

    def _compute_provenance(self, result: MonitoringReviewResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
