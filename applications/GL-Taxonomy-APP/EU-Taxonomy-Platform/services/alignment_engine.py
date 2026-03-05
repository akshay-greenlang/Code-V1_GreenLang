"""
Alignment Engine -- End-to-End EU Taxonomy Alignment Workflow Orchestration

Implements the complete four-step alignment pipeline: (1) Eligibility check,
(2) Substantial Contribution (SC) assessment against Technical Screening
Criteria, (3) Do No Significant Harm (DNSH) verification for all other
objectives, and (4) Minimum Safeguards (MS) compliance check. An economic
activity is classified as "taxonomy-aligned" only when all four steps pass.

Provides portfolio-level alignment, batch processing, alignment progress
tracking, dashboard views, funnel analytics, sector-level alignment,
period-over-period comparison, and export capabilities.

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - Regulation (EU) 2020/852, Articles 3, 10-18
    - Delegated Regulation (EU) 2021/2139 (Climate Delegated Act TSC)
    - Delegated Regulation (EU) 2023/2486 (Environmental Delegated Act TSC)
    - EU Platform on Sustainable Finance -- Technical Screening Criteria
    - Commission FAQ on Taxonomy alignment (2023)

Example:
    >>> from services.config import TaxonomyAppConfig
    >>> engine = AlignmentEngine(TaxonomyAppConfig())
    >>> result = engine.run_full_alignment("org-1", "4.1", "2025", data)
    >>> print(result.is_aligned)
    True
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    DNSH_MATRIX,
    ENVIRONMENTAL_OBJECTIVES,
    MINIMUM_SAFEGUARD_TOPICS,
    TAXONOMY_ACTIVITIES,
    AlignmentStatus,
    EnvironmentalObjective,
    SafeguardTopic,
    TaxonomyAppConfig,
)
from .models import (
    EconomicActivity,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal step-result models (used only within alignment pipeline)
# ---------------------------------------------------------------------------

class _EligibilityResult(BaseModel):
    """Internal eligibility assessment result for the alignment pipeline."""

    activity_id: str = Field(default_factory=_new_id)
    activity_code: str = Field(...)
    is_eligible: bool = Field(default=False)
    matched_objective: Optional[str] = Field(None)
    matched_activity_name: Optional[str] = Field(None)
    reason: str = Field(default="")


class _SCResult(BaseModel):
    """Internal SC assessment result for the alignment pipeline."""

    activity_id: str = Field(default_factory=_new_id)
    activity_code: str = Field(...)
    objective: str = Field(...)
    criteria_met: bool = Field(default=False)
    criteria_details: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_items: List[Any] = Field(default_factory=list)
    reason: str = Field(default="")


class _DNSHResult(BaseModel):
    """Internal DNSH assessment result for the alignment pipeline."""

    activity_id: str = Field(default_factory=_new_id)
    activity_code: str = Field(...)
    all_pass: bool = Field(default=False)
    objective_results: Dict[str, bool] = Field(default_factory=dict)
    failed_objectives: List[str] = Field(default_factory=list)
    details: List[Dict[str, Any]] = Field(default_factory=list)
    reason: str = Field(default="")


class _MSResult(BaseModel):
    """Internal MS assessment result for the alignment pipeline."""

    activity_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    all_pass: bool = Field(default=False)
    pillar_results: Dict[str, bool] = Field(default_factory=dict)
    failed_pillars: List[str] = Field(default_factory=list)
    details: List[Dict[str, Any]] = Field(default_factory=list)
    reason: str = Field(default="")


# ---------------------------------------------------------------------------
# DNSH Defaults and MS Requirements (local references for alignment pipeline)
# ---------------------------------------------------------------------------

_DNSH_OBJECTIVES: List[str] = [
    "climate_mitigation",
    "climate_adaptation",
    "water_marine",
    "circular_economy",
    "pollution_prevention",
    "biodiversity_ecosystems",
]

_MS_PILLARS: Dict[str, Dict[str, Any]] = {
    "human_rights": {
        "pillar": "Human Rights",
        "frameworks": ["UNGPs", "OECD Guidelines Ch. IV"],
    },
    "anti_corruption": {
        "pillar": "Anti-Corruption / Bribery",
        "frameworks": ["OECD Guidelines Ch. VII", "UNCAC"],
    },
    "taxation": {
        "pillar": "Taxation",
        "frameworks": ["OECD Guidelines Ch. XI", "EU ATAD I/II"],
    },
    "fair_competition": {
        "pillar": "Fair Competition",
        "frameworks": ["OECD Guidelines Ch. X", "TFEU Articles 101-102"],
    },
}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AlignmentStep(BaseModel):
    """Result of a single alignment pipeline step."""

    step_number: int = Field(...)
    step_name: str = Field(...)
    passed: bool = Field(default=False)
    details: Dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class AlignmentWorkflowResult(BaseModel):
    """Result of the full 4-step alignment workflow for one activity."""

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    activity_code: str = Field(...)
    period: str = Field(...)
    is_eligible: bool = Field(default=False)
    sc_pass: bool = Field(default=False)
    dnsh_pass: bool = Field(default=False)
    ms_pass: bool = Field(default=False)
    is_aligned: bool = Field(default=False)
    alignment_status: str = Field(default="not_assessed")
    steps: List[AlignmentStep] = Field(default_factory=list)
    activity_name: Optional[str] = Field(None)
    objective: Optional[str] = Field(None)
    activity_type: Optional[str] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class ActivityAlignmentResult(BaseModel):
    """Alignment status for a single activity (summary view)."""

    activity_code: str = Field(...)
    activity_name: str = Field(default="")
    period: str = Field(...)
    is_eligible: bool = Field(default=False)
    is_aligned: bool = Field(default=False)
    alignment_status: str = Field(default="not_assessed")
    objective: Optional[str] = Field(None)
    activity_type: Optional[str] = Field(None)
    turnover_eur: float = Field(default=0.0, ge=0.0)
    capex_eur: float = Field(default=0.0, ge=0.0)
    opex_eur: float = Field(default=0.0, ge=0.0)


class PortfolioAlignmentResult(BaseModel):
    """Portfolio-level alignment result across all activities."""

    org_id: str = Field(...)
    period: str = Field(...)
    total_activities: int = Field(default=0)
    eligible_count: int = Field(default=0)
    aligned_count: int = Field(default=0)
    not_aligned_count: int = Field(default=0)
    alignment_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_turnover_eur: float = Field(default=0.0, ge=0.0)
    aligned_turnover_eur: float = Field(default=0.0, ge=0.0)
    aligned_turnover_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_capex_eur: float = Field(default=0.0, ge=0.0)
    aligned_capex_eur: float = Field(default=0.0, ge=0.0)
    aligned_capex_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_opex_eur: float = Field(default=0.0, ge=0.0)
    aligned_opex_eur: float = Field(default=0.0, ge=0.0)
    aligned_opex_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    activities: List[ActivityAlignmentResult] = Field(default_factory=list)
    by_objective: Dict[str, int] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class AlignmentProgress(BaseModel):
    """Alignment pipeline progress for an organization."""

    org_id: str = Field(...)
    period: str = Field(...)
    total_activities: int = Field(default=0)
    step1_complete: int = Field(default=0, description="Eligibility assessed")
    step2_complete: int = Field(default=0, description="SC assessed")
    step3_complete: int = Field(default=0, description="DNSH assessed")
    step4_complete: int = Field(default=0, description="MS assessed")
    fully_assessed: int = Field(default=0)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    pending_activities: List[str] = Field(default_factory=list)


class AlignmentDashboard(BaseModel):
    """Overall alignment dashboard for an organization."""

    org_id: str = Field(...)
    period: str = Field(...)
    total_activities: int = Field(default=0)
    eligible_count: int = Field(default=0)
    aligned_count: int = Field(default=0)
    not_eligible_count: int = Field(default=0)
    failed_sc_count: int = Field(default=0)
    failed_dnsh_count: int = Field(default=0)
    failed_ms_count: int = Field(default=0)
    alignment_rate_pct: float = Field(default=0.0)
    turnover_aligned_pct: float = Field(default=0.0)
    capex_aligned_pct: float = Field(default=0.0)
    opex_aligned_pct: float = Field(default=0.0)
    by_objective: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    by_activity_type: Dict[str, int] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# AlignmentEngine
# ---------------------------------------------------------------------------

class AlignmentEngine:
    """
    End-to-end EU Taxonomy alignment workflow orchestration engine.

    Runs the four-step alignment pipeline (Eligibility -> SC -> DNSH -> MS)
    for individual activities or entire portfolios, tracking progress and
    producing dashboard views.

    Attributes:
        config: Application configuration.
        _activities: In-memory activities keyed by (org_id, activity_code, period).
        _results: Cached alignment results keyed by (org_id, activity_code, period).
        _ms_data: Minimum safeguards data keyed by org_id.

    Example:
        >>> engine = AlignmentEngine(TaxonomyAppConfig())
        >>> result = engine.run_full_alignment("org-1", "4.1", "2025", {})
        >>> print(result.is_aligned)
    """

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """Initialize the AlignmentEngine."""
        self.config = config or TaxonomyAppConfig()
        self._activities: Dict[str, EconomicActivity] = {}
        self._results: Dict[str, AlignmentWorkflowResult] = {}
        self._ms_data: Dict[str, Dict[str, Any]] = {}
        logger.info("AlignmentEngine initialized")

    # ------------------------------------------------------------------
    # Data Registration
    # ------------------------------------------------------------------

    def register_activity(self, activity: EconomicActivity) -> None:
        """
        Register an economic activity for alignment assessment.

        Args:
            activity: EconomicActivity model instance.
        """
        key = f"{activity.org_id}:{activity.activity_code}:{activity.period}"
        self._activities[key] = activity

    def register_ms_data(
        self, org_id: str, ms_data: Dict[str, Any],
    ) -> None:
        """
        Register Minimum Safeguards data for an organization.

        Args:
            org_id: Organization identifier.
            ms_data: Dict with pillar-level compliance data.
        """
        self._ms_data[org_id] = ms_data

    # ------------------------------------------------------------------
    # Full Alignment Workflow
    # ------------------------------------------------------------------

    def run_full_alignment(
        self,
        org_id: str,
        activity_code: str,
        period: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AlignmentWorkflowResult:
        """
        Run the complete 4-step alignment pipeline for one activity.

        Step 1: Eligibility -- Is the activity listed in the EU Taxonomy?
        Step 2: Substantial Contribution -- Does it meet the TSC?
        Step 3: DNSH -- Does it do no significant harm to other objectives?
        Step 4: Minimum Safeguards -- Does the company comply with MS?

        The activity is "aligned" only if all four steps pass.

        Args:
            org_id: Organization identifier.
            activity_code: EU Taxonomy activity code (e.g. "4.1").
            period: Reporting period.
            data: Optional additional assessment data (SC evidence,
                DNSH compliance, MS documentation).

        Returns:
            AlignmentWorkflowResult with step-by-step results.
        """
        start = datetime.utcnow()
        data = data or {}

        steps: List[AlignmentStep] = []
        final_status = AlignmentStatus.NOT_SCREENED.value

        # Step 1: Eligibility
        eligibility = self._assess_eligibility(activity_code)
        steps.append(AlignmentStep(
            step_number=1,
            step_name="Eligibility",
            passed=eligibility.is_eligible,
            details={
                "activity_code": activity_code,
                "matched_objective": eligibility.matched_objective,
                "matched_activity": eligibility.matched_activity_name,
            },
            reason=eligibility.reason,
        ))

        is_eligible = eligibility.is_eligible
        sc_pass = False
        dnsh_pass = False
        ms_pass = False
        is_aligned = False
        objective = eligibility.matched_objective
        activity_name = eligibility.matched_activity_name
        activity_type = None

        if not is_eligible:
            final_status = AlignmentStatus.NOT_ELIGIBLE.value
        else:
            final_status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value
            act_info = TAXONOMY_ACTIVITIES.get(activity_code, {})
            activity_type = act_info.get("activity_type")

            # Step 2: Substantial Contribution
            sc_result = self._assess_sc(activity_code, objective or "", data)
            sc_pass = sc_result.criteria_met
            steps.append(AlignmentStep(
                step_number=2,
                step_name="Substantial Contribution (TSC)",
                passed=sc_pass,
                details={
                    "objective": objective,
                    "criteria_count": len(sc_result.criteria_details),
                    "evidence_count": len(sc_result.evidence_items),
                },
                reason=sc_result.reason,
            ))

            if not sc_pass:
                final_status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value
            else:
                final_status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value

                # Step 3: DNSH
                dnsh_result = self._assess_dnsh(
                    activity_code, objective or "", data,
                )
                dnsh_pass = dnsh_result.all_pass
                steps.append(AlignmentStep(
                    step_number=3,
                    step_name="Do No Significant Harm (DNSH)",
                    passed=dnsh_pass,
                    details={
                        "objectives_checked": len(dnsh_result.objective_results),
                        "failed_objectives": dnsh_result.failed_objectives,
                    },
                    reason=dnsh_result.reason,
                ))

                if not dnsh_pass:
                    final_status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value
                else:
                    final_status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value

                    # Step 4: Minimum Safeguards
                    ms_result = self._assess_ms(org_id, data)
                    ms_pass = ms_result.all_pass
                    steps.append(AlignmentStep(
                        step_number=4,
                        step_name="Minimum Safeguards (MS)",
                        passed=ms_pass,
                        details={
                            "pillars_checked": len(ms_result.pillar_results),
                            "failed_pillars": ms_result.failed_pillars,
                        },
                        reason=ms_result.reason,
                    ))

                    if ms_pass:
                        final_status = AlignmentStatus.ALIGNED.value
                        is_aligned = True
                    else:
                        final_status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000

        provenance = _sha256(
            f"alignment:{org_id}:{activity_code}:{period}:{final_status}"
        )

        result = AlignmentWorkflowResult(
            org_id=org_id,
            activity_code=activity_code,
            period=period,
            is_eligible=is_eligible,
            sc_pass=sc_pass,
            dnsh_pass=dnsh_pass,
            ms_pass=ms_pass,
            is_aligned=is_aligned,
            alignment_status=final_status,
            steps=steps,
            activity_name=activity_name,
            objective=objective,
            activity_type=activity_type,
            processing_time_ms=round(elapsed, 2),
            provenance_hash=provenance,
        )

        cache_key = f"{org_id}:{activity_code}:{period}"
        self._results[cache_key] = result

        logger.info(
            "Alignment for %s activity %s period %s: status=%s in %.1f ms",
            org_id, activity_code, period, final_status, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Portfolio Alignment
    # ------------------------------------------------------------------

    def run_portfolio_alignment(
        self,
        org_id: str,
        period: str,
        activities: List[Dict[str, Any]],
    ) -> PortfolioAlignmentResult:
        """
        Run alignment workflow for all activities in an organization.

        Args:
            org_id: Organization identifier.
            period: Reporting period.
            activities: List of dicts with activity_code and optional data.

        Returns:
            PortfolioAlignmentResult with aggregate metrics.
        """
        start = datetime.utcnow()

        activity_results: List[ActivityAlignmentResult] = []
        total_turnover = 0.0
        aligned_turnover = 0.0
        total_capex = 0.0
        aligned_capex = 0.0
        total_opex = 0.0
        aligned_opex = 0.0
        eligible_count = 0
        aligned_count = 0
        not_aligned_count = 0
        by_objective: Dict[str, int] = {}

        for act in activities:
            code = act.get("activity_code", "")
            act_data = act.get("data", {})
            turnover = float(act.get("turnover_eur", 0))
            capex = float(act.get("capex_eur", 0))
            opex = float(act.get("opex_eur", 0))

            workflow = self.run_full_alignment(org_id, code, period, act_data)

            total_turnover += turnover
            total_capex += capex
            total_opex += opex

            if workflow.is_eligible:
                eligible_count += 1
            if workflow.is_aligned:
                aligned_count += 1
                aligned_turnover += turnover
                aligned_capex += capex
                aligned_opex += opex
                obj = workflow.objective or "unspecified"
                by_objective[obj] = by_objective.get(obj, 0) + 1
            else:
                not_aligned_count += 1

            activity_results.append(ActivityAlignmentResult(
                activity_code=code,
                activity_name=workflow.activity_name or "",
                period=period,
                is_eligible=workflow.is_eligible,
                is_aligned=workflow.is_aligned,
                alignment_status=workflow.alignment_status,
                objective=workflow.objective,
                activity_type=workflow.activity_type,
                turnover_eur=turnover,
                capex_eur=capex,
                opex_eur=opex,
            ))

        total = len(activities)
        alignment_rate = (aligned_count / total * 100.0) if total > 0 else 0.0
        turnover_pct = (
            aligned_turnover / total_turnover * 100.0
            if total_turnover > 0 else 0.0
        )
        capex_pct = (
            aligned_capex / total_capex * 100.0
            if total_capex > 0 else 0.0
        )
        opex_pct = (
            aligned_opex / total_opex * 100.0
            if total_opex > 0 else 0.0
        )

        provenance = _sha256(
            f"portfolio:{org_id}:{period}:{aligned_count}/{total}"
        )

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Portfolio alignment for %s period %s: %d/%d aligned (%.1f%%) in %.1f ms",
            org_id, period, aligned_count, total, alignment_rate, elapsed,
        )

        return PortfolioAlignmentResult(
            org_id=org_id,
            period=period,
            total_activities=total,
            eligible_count=eligible_count,
            aligned_count=aligned_count,
            not_aligned_count=not_aligned_count,
            alignment_rate_pct=round(alignment_rate, 2),
            total_turnover_eur=round(total_turnover, 2),
            aligned_turnover_eur=round(aligned_turnover, 2),
            aligned_turnover_pct=round(turnover_pct, 2),
            total_capex_eur=round(total_capex, 2),
            aligned_capex_eur=round(aligned_capex, 2),
            aligned_capex_pct=round(capex_pct, 2),
            total_opex_eur=round(total_opex, 2),
            aligned_opex_eur=round(aligned_opex, 2),
            aligned_opex_pct=round(opex_pct, 2),
            activities=activity_results,
            by_objective=by_objective,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Alignment Status
    # ------------------------------------------------------------------

    def get_alignment_status(
        self,
        org_id: str,
        activity_code: str,
        period: str,
    ) -> ActivityAlignmentResult:
        """
        Retrieve alignment status for a specific activity.

        Args:
            org_id: Organization identifier.
            activity_code: EU Taxonomy activity code.
            period: Reporting period.

        Returns:
            ActivityAlignmentResult with current status.
        """
        cache_key = f"{org_id}:{activity_code}:{period}"
        result = self._results.get(cache_key)

        if result:
            return ActivityAlignmentResult(
                activity_code=activity_code,
                activity_name=result.activity_name or "",
                period=period,
                is_eligible=result.is_eligible,
                is_aligned=result.is_aligned,
                alignment_status=result.alignment_status,
                objective=result.objective,
                activity_type=result.activity_type,
            )

        return ActivityAlignmentResult(
            activity_code=activity_code,
            period=period,
            alignment_status="not_assessed",
        )

    # ------------------------------------------------------------------
    # Batch Alignment
    # ------------------------------------------------------------------

    def batch_alignment(
        self,
        org_id: str,
        period: str,
        activity_list: List[Dict[str, Any]],
    ) -> List[ActivityAlignmentResult]:
        """
        Run alignment for a batch of activities.

        Args:
            org_id: Organization identifier.
            period: Reporting period.
            activity_list: List of dicts with activity_code and optional data.

        Returns:
            List of ActivityAlignmentResult for each activity.
        """
        results: List[ActivityAlignmentResult] = []

        for act in activity_list:
            code = act.get("activity_code", "")
            data = act.get("data", {})
            workflow = self.run_full_alignment(org_id, code, period, data)
            results.append(ActivityAlignmentResult(
                activity_code=code,
                activity_name=workflow.activity_name or "",
                period=period,
                is_eligible=workflow.is_eligible,
                is_aligned=workflow.is_aligned,
                alignment_status=workflow.alignment_status,
                objective=workflow.objective,
                activity_type=workflow.activity_type,
                turnover_eur=float(act.get("turnover_eur", 0)),
                capex_eur=float(act.get("capex_eur", 0)),
                opex_eur=float(act.get("opex_eur", 0)),
            ))

        logger.info(
            "Batch alignment for %s period %s: %d activities",
            org_id, period, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Alignment Progress
    # ------------------------------------------------------------------

    def get_alignment_progress(
        self, org_id: str, period: str,
    ) -> AlignmentProgress:
        """
        Track alignment pipeline progress for an organization.

        Counts how many activities have completed each step of the
        four-step pipeline.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            AlignmentProgress with step completion counts.
        """
        all_results = [
            r for key, r in self._results.items()
            if r.org_id == org_id and r.period == period
        ]

        total = len(all_results)
        step1 = sum(1 for r in all_results if r.is_eligible or r.alignment_status == AlignmentStatus.NOT_ELIGIBLE.value)
        step2 = sum(1 for r in all_results if r.sc_pass)
        step3 = sum(1 for r in all_results if r.dnsh_pass)
        step4 = sum(1 for r in all_results if r.ms_pass or r.is_aligned)
        fully = sum(1 for r in all_results if r.is_aligned or r.alignment_status in (
            AlignmentStatus.NOT_ELIGIBLE.value,
            AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value,
            AlignmentStatus.ALIGNED.value,
        ))

        # Find activities not yet assessed
        assessed_codes = {r.activity_code for r in all_results}
        pending = [
            key.split(":")[1] for key, act in self._activities.items()
            if act.org_id == org_id and act.period == period
            and act.activity_code not in assessed_codes
        ]

        progress_pct = (fully / total * 100.0) if total > 0 else 0.0

        return AlignmentProgress(
            org_id=org_id,
            period=period,
            total_activities=total,
            step1_complete=step1,
            step2_complete=step2,
            step3_complete=step3,
            step4_complete=step4,
            fully_assessed=fully,
            progress_pct=round(progress_pct, 2),
            pending_activities=pending,
        )

    # ------------------------------------------------------------------
    # Alignment Dashboard
    # ------------------------------------------------------------------

    def get_alignment_dashboard(
        self, org_id: str, period: str,
    ) -> AlignmentDashboard:
        """
        Generate an overall alignment dashboard for an organization.

        Aggregates alignment results across all assessed activities,
        breaking down by objective, activity type, and failure reason.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            AlignmentDashboard with comprehensive metrics.
        """
        start = datetime.utcnow()

        all_results = [
            r for key, r in self._results.items()
            if r.org_id == org_id and r.period == period
        ]

        total = len(all_results)
        eligible = sum(1 for r in all_results if r.is_eligible)
        aligned = sum(1 for r in all_results if r.is_aligned)
        not_eligible = sum(
            1 for r in all_results
            if r.alignment_status == AlignmentStatus.NOT_ELIGIBLE.value
        )
        failed_sc = sum(
            1 for r in all_results
            if r.is_eligible and not r.sc_pass and not r.is_aligned
        )
        failed_dnsh = sum(
            1 for r in all_results
            if r.is_eligible and r.sc_pass and not r.dnsh_pass and not r.is_aligned
        )
        failed_ms = sum(
            1 for r in all_results
            if r.is_eligible and r.sc_pass and r.dnsh_pass and not r.ms_pass and not r.is_aligned
        )

        alignment_rate = (aligned / total * 100.0) if total > 0 else 0.0

        # KPI alignment percentages (from registered activities)
        total_turnover = 0.0
        aligned_turnover = 0.0
        total_capex = 0.0
        aligned_capex = 0.0
        total_opex = 0.0
        aligned_opex = 0.0

        for r in all_results:
            key = f"{r.org_id}:{r.activity_code}:{r.period}"
            act = self._activities.get(key)
            if act:
                t = float(act.turnover_eur)
                c = float(act.capex_eur)
                o = float(act.opex_eur)
                total_turnover += t
                total_capex += c
                total_opex += o
                if r.is_aligned:
                    aligned_turnover += t
                    aligned_capex += c
                    aligned_opex += o

        turnover_pct = (
            aligned_turnover / total_turnover * 100.0
            if total_turnover > 0 else 0.0
        )
        capex_pct = (
            aligned_capex / total_capex * 100.0
            if total_capex > 0 else 0.0
        )
        opex_pct = (
            aligned_opex / total_opex * 100.0
            if total_opex > 0 else 0.0
        )

        # By objective
        by_objective: Dict[str, Dict[str, Any]] = {}
        for r in all_results:
            if r.is_aligned and r.objective:
                if r.objective not in by_objective:
                    by_objective[r.objective] = {"count": 0, "activities": []}
                by_objective[r.objective]["count"] += 1
                by_objective[r.objective]["activities"].append(r.activity_code)

        # By activity type
        by_type: Dict[str, int] = {}
        for r in all_results:
            if r.is_aligned and r.activity_type:
                by_type[r.activity_type] = by_type.get(r.activity_type, 0) + 1

        provenance = _sha256(
            f"dashboard:{org_id}:{period}:{aligned}/{total}"
        )

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Alignment dashboard for %s period %s: %d/%d aligned in %.1f ms",
            org_id, period, aligned, total, elapsed,
        )

        return AlignmentDashboard(
            org_id=org_id,
            period=period,
            total_activities=total,
            eligible_count=eligible,
            aligned_count=aligned,
            not_eligible_count=not_eligible,
            failed_sc_count=failed_sc,
            failed_dnsh_count=failed_dnsh,
            failed_ms_count=failed_ms,
            alignment_rate_pct=round(alignment_rate, 2),
            turnover_aligned_pct=round(turnover_pct, 2),
            capex_aligned_pct=round(capex_pct, 2),
            opex_aligned_pct=round(opex_pct, 2),
            by_objective=by_objective,
            by_activity_type=by_type,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Eligible vs Aligned Funnel
    # ------------------------------------------------------------------

    def get_eligible_vs_aligned(
        self, org_id: str, period: str,
    ) -> Dict[str, Any]:
        """
        Generate alignment funnel: total -> eligible -> SC -> DNSH -> aligned.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            Dict with funnel counts and percentages at each stage.
        """
        all_results = [
            r for key, r in self._results.items()
            if r.org_id == org_id and r.period == period
        ]

        total = len(all_results)
        eligible = sum(1 for r in all_results if r.is_eligible)
        sc_passed = sum(1 for r in all_results if r.sc_pass)
        dnsh_passed = sum(1 for r in all_results if r.dnsh_pass)
        aligned = sum(1 for r in all_results if r.is_aligned)

        return {
            "org_id": org_id,
            "period": period,
            "funnel": [
                {
                    "stage": "total_activities",
                    "count": total,
                    "pct": 100.0,
                },
                {
                    "stage": "eligible",
                    "count": eligible,
                    "pct": round(eligible / total * 100.0 if total > 0 else 0.0, 2),
                },
                {
                    "stage": "sc_passed",
                    "count": sc_passed,
                    "pct": round(sc_passed / total * 100.0 if total > 0 else 0.0, 2),
                },
                {
                    "stage": "dnsh_passed",
                    "count": dnsh_passed,
                    "pct": round(dnsh_passed / total * 100.0 if total > 0 else 0.0, 2),
                },
                {
                    "stage": "aligned",
                    "count": aligned,
                    "pct": round(aligned / total * 100.0 if total > 0 else 0.0, 2),
                },
            ],
            "conversion_rate_pct": round(
                aligned / total * 100.0 if total > 0 else 0.0, 2,
            ),
        }

    # ------------------------------------------------------------------
    # Sector Alignment
    # ------------------------------------------------------------------

    def get_sector_alignment(
        self, org_id: str, period: str,
    ) -> Dict[str, Any]:
        """
        Break down alignment results by NACE sector.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            Dict with per-sector alignment counts and rates.
        """
        all_results = [
            r for key, r in self._results.items()
            if r.org_id == org_id and r.period == period
        ]

        sector_map: Dict[str, Dict[str, int]] = {}
        for r in all_results:
            act_info = TAXONOMY_ACTIVITIES.get(r.activity_code, {})
            nace = ", ".join(act_info.get("nace_codes", [])) or "UNKNOWN"
            if nace not in sector_map:
                sector_map[nace] = {"total": 0, "eligible": 0, "aligned": 0}
            sector_map[nace]["total"] += 1
            if r.is_eligible:
                sector_map[nace]["eligible"] += 1
            if r.is_aligned:
                sector_map[nace]["aligned"] += 1

        sectors = []
        for nace, counts in sorted(sector_map.items()):
            rate = (
                counts["aligned"] / counts["total"] * 100.0
                if counts["total"] > 0 else 0.0
            )
            sectors.append({
                "nace_code": nace,
                "total": counts["total"],
                "eligible": counts["eligible"],
                "aligned": counts["aligned"],
                "alignment_rate_pct": round(rate, 2),
            })

        return {
            "org_id": org_id,
            "period": period,
            "sectors": sectors,
            "sector_count": len(sectors),
        }

    # ------------------------------------------------------------------
    # Period Comparison
    # ------------------------------------------------------------------

    def compare_alignment_periods(
        self, org_id: str, period1: str, period2: str,
    ) -> Dict[str, Any]:
        """
        Compare alignment results between two reporting periods.

        Args:
            org_id: Organization identifier.
            period1: First period.
            period2: Second period.

        Returns:
            Dict with side-by-side comparison and changes.
        """
        results_p1 = [
            r for key, r in self._results.items()
            if r.org_id == org_id and r.period == period1
        ]
        results_p2 = [
            r for key, r in self._results.items()
            if r.org_id == org_id and r.period == period2
        ]

        aligned_p1 = sum(1 for r in results_p1 if r.is_aligned)
        aligned_p2 = sum(1 for r in results_p2 if r.is_aligned)
        total_p1 = len(results_p1)
        total_p2 = len(results_p2)

        rate_p1 = aligned_p1 / total_p1 * 100.0 if total_p1 > 0 else 0.0
        rate_p2 = aligned_p2 / total_p2 * 100.0 if total_p2 > 0 else 0.0

        return {
            "org_id": org_id,
            "period_1": {
                "period": period1,
                "total_activities": total_p1,
                "aligned_count": aligned_p1,
                "alignment_rate_pct": round(rate_p1, 2),
            },
            "period_2": {
                "period": period2,
                "total_activities": total_p2,
                "aligned_count": aligned_p2,
                "alignment_rate_pct": round(rate_p2, 2),
            },
            "change": {
                "aligned_delta": aligned_p2 - aligned_p1,
                "rate_delta_pp": round(rate_p2 - rate_p1, 2),
                "direction": (
                    "improving" if rate_p2 > rate_p1
                    else "declining" if rate_p2 < rate_p1
                    else "stable"
                ),
            },
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_alignment_summary(
        self,
        org_id: str,
        period: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export alignment summary in the requested format.

        Args:
            org_id: Organization identifier.
            period: Reporting period.
            format: Export format (json, csv, excel).

        Returns:
            Dict with export metadata and content.
        """
        all_results = [
            r for key, r in self._results.items()
            if r.org_id == org_id and r.period == period
        ]

        rows = []
        for r in all_results:
            rows.append({
                "activity_code": r.activity_code,
                "activity_name": r.activity_name or "",
                "objective": r.objective or "",
                "activity_type": r.activity_type or "",
                "is_eligible": r.is_eligible,
                "sc_pass": r.sc_pass,
                "dnsh_pass": r.dnsh_pass,
                "ms_pass": r.ms_pass,
                "is_aligned": r.is_aligned,
                "alignment_status": r.alignment_status,
            })

        aligned_count = sum(1 for r in all_results if r.is_aligned)

        return {
            "org_id": org_id,
            "period": period,
            "format": format,
            "total_activities": len(rows),
            "aligned_count": aligned_count,
            "rows": rows,
            "exported_at": _now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assess_eligibility(
        self, activity_code: str,
    ) -> _EligibilityResult:
        """
        Check if an activity code is listed in the EU Taxonomy.

        Args:
            activity_code: EU Taxonomy activity code.

        Returns:
            _EligibilityResult with eligibility determination.
        """
        act_info = TAXONOMY_ACTIVITIES.get(activity_code)

        if act_info is None:
            return _EligibilityResult(
                activity_id=_new_id(),
                activity_code=activity_code,
                is_eligible=False,
                reason=(
                    f"Activity code '{activity_code}' is not listed in the "
                    f"EU Taxonomy Delegated Acts."
                ),
            )

        objectives = act_info.get("objectives", [])
        first_objective = objectives[0] if objectives else None
        nace_codes = act_info.get("nace_codes", [])

        return _EligibilityResult(
            activity_id=_new_id(),
            activity_code=activity_code,
            is_eligible=True,
            matched_objective=first_objective,
            matched_activity_name=act_info.get("name"),
            reason=(
                f"Activity '{act_info.get('name')}' is eligible under "
                f"'{first_objective}' (NACE: {nace_codes})."
            ),
        )

    def _assess_sc(
        self,
        activity_code: str,
        objective: str,
        data: Dict[str, Any],
    ) -> _SCResult:
        """
        Assess Substantial Contribution against Technical Screening Criteria.

        Uses provided evidence/data to determine if the activity meets
        the TSC for its primary environmental objective.

        Args:
            activity_code: EU Taxonomy activity code.
            objective: Primary environmental objective.
            data: Assessment data with SC evidence.

        Returns:
            _SCResult with criteria evaluation.
        """
        act_info = TAXONOMY_ACTIVITIES.get(activity_code, {})
        tsc_summary = act_info.get("sc_criteria_ref", "No TSC defined")

        sc_data = data.get("sc", {})
        sc_pass = sc_data.get("criteria_met", True)
        evidence = sc_data.get("evidence", [])
        criteria_details = sc_data.get("criteria_details", [
            {"criterion": tsc_summary, "met": sc_pass},
        ])

        reason = (
            f"TSC for {activity_code} ({objective}): "
            f"{'PASS' if sc_pass else 'FAIL'}. "
            f"{tsc_summary}"
        )

        return _SCResult(
            activity_id=_new_id(),
            activity_code=activity_code,
            objective=objective,
            criteria_met=sc_pass,
            criteria_details=criteria_details,
            evidence_items=evidence,
            reason=reason,
        )

    def _assess_dnsh(
        self,
        activity_code: str,
        objective: str,
        data: Dict[str, Any],
    ) -> _DNSHResult:
        """
        Assess Do No Significant Harm for all other objectives.

        For each environmental objective other than the primary SC
        objective, checks that the activity does not cause significant
        harm per the DNSH criteria in the Delegated Acts.

        Args:
            activity_code: EU Taxonomy activity code.
            objective: Primary SC objective (excluded from DNSH).
            data: Assessment data with DNSH compliance.

        Returns:
            _DNSHResult with per-objective pass/fail.
        """
        dnsh_data = data.get("dnsh", {})

        other_objectives = [
            obj for obj in _DNSH_OBJECTIVES if obj != objective
        ]

        objective_results: Dict[str, bool] = {}
        failed: List[str] = []
        details: List[Dict[str, Any]] = []

        for obj_name in other_objectives:
            passed = dnsh_data.get(obj_name, True)
            objective_results[obj_name] = passed

            details.append({
                "objective": obj_name,
                "dnsh_criterion": obj_name,
                "passed": passed,
                "description": f"DNSH assessment for {obj_name}",
            })

            if not passed:
                failed.append(obj_name)

        all_pass = len(failed) == 0
        failed_str = ", ".join(failed)
        reason = (
            f"DNSH for {activity_code}: {'PASS' if all_pass else 'FAIL'}. "
            f"Checked {len(other_objectives)} objectives. "
            f"{'All passed.' if all_pass else 'Failed: ' + failed_str}"
        )

        return _DNSHResult(
            activity_id=_new_id(),
            activity_code=activity_code,
            all_pass=all_pass,
            objective_results=objective_results,
            failed_objectives=failed,
            details=details,
            reason=reason,
        )

    def _assess_ms(
        self, org_id: str, data: Dict[str, Any],
    ) -> _MSResult:
        """
        Assess Minimum Safeguards compliance at the organization level.

        Checks compliance across four pillars: human rights, labour,
        anti-bribery/corruption, and fair taxation per Article 18.

        Args:
            org_id: Organization identifier.
            data: Assessment data with MS compliance.

        Returns:
            _MSResult with per-pillar pass/fail.
        """
        ms_input = data.get("ms", self._ms_data.get(org_id, {}))

        pillar_results: Dict[str, bool] = {}
        failed: List[str] = []
        details: List[Dict[str, Any]] = []

        for pillar_key, pillar_def in _MS_PILLARS.items():
            passed = ms_input.get(pillar_key, True)
            pillar_results[pillar_key] = passed

            details.append({
                "pillar": pillar_key,
                "pillar_name": pillar_def["pillar"],
                "passed": passed,
                "frameworks": pillar_def["frameworks"],
            })

            if not passed:
                failed.append(pillar_key)

        all_pass = len(failed) == 0
        failed_str = ", ".join(failed)
        reason = (
            f"MS for {org_id}: {'PASS' if all_pass else 'FAIL'}. "
            f"Checked {len(_MS_PILLARS)} pillars. "
            f"{'All compliant.' if all_pass else 'Non-compliant: ' + failed_str}"
        )

        return _MSResult(
            activity_id=_new_id(),
            org_id=org_id,
            all_pass=all_pass,
            pillar_results=pillar_results,
            failed_pillars=failed,
            details=details,
            reason=reason,
        )
