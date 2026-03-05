"""
Gap Analysis Engine -- Disclosure maturity assessment and improvement planning.

This module implements the ``GapAnalysisEngine`` for GL-TCFD-APP v1.0.
It assesses organizational maturity across all four TCFD pillars using
10 dimensions per pillar (40 total dimensions), identifies requirement-level
gaps with severity classification, benchmarks against sector peers,
generates prioritized action plans, estimates effort and timelines,
and tracks progress across assessment cycles.

Maturity levels (1-5):
    1 - Initial:    Ad hoc, no formal processes
    2 - Developing: Some processes in place, inconsistent
    3 - Defined:    Standardized processes, partial implementation
    4 - Managed:    Quantified processes, regularly reviewed
    5 - Optimized:  Continuous improvement, best-in-class

Reference:
    - TCFD Final Report (June 2017)
    - TCFD Status Reports (2018-2023)
    - IFRS S2 Comparison to TCFD (October 2023)

Example:
    >>> from services.config import TCFDAppConfig
    >>> engine = GapAnalysisEngine(TCFDAppConfig())
    >>> assessment = engine.assess_maturity("org-1")
    >>> print(assessment.overall_maturity)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    GovernanceMaturityLevel,
    MATURITY_SCORES,
    SectorType,
    TCFDAppConfig,
    TCFDPillar,
    TCFD_DISCLOSURES,
)
from .models import (
    GapAssessment,
    MaturityScore as DomainMaturityScore,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Maturity dimensions per pillar (10 dimensions each)
# ---------------------------------------------------------------------------

MATURITY_DIMENSIONS: Dict[str, List[Dict[str, str]]] = {
    "governance": [
        {"id": "gov_d01", "name": "Board Climate Oversight", "description": "Frequency and depth of board-level climate risk reviews"},
        {"id": "gov_d02", "name": "Dedicated Committee", "description": "Existence of sustainability/climate committee with clear mandate"},
        {"id": "gov_d03", "name": "Management Accountability", "description": "Explicit climate accountability in management roles"},
        {"id": "gov_d04", "name": "Climate Competency", "description": "Board and management climate science/policy expertise"},
        {"id": "gov_d05", "name": "Remuneration Linkage", "description": "Climate KPIs linked to executive compensation"},
        {"id": "gov_d06", "name": "Reporting Quality", "description": "Quality and frequency of climate reporting to board"},
        {"id": "gov_d07", "name": "Strategy Integration", "description": "Climate integrated into strategic planning process"},
        {"id": "gov_d08", "name": "Decision Making", "description": "Climate factors in investment and M&A decisions"},
        {"id": "gov_d09", "name": "External Engagement", "description": "Board engagement with external climate frameworks/initiatives"},
        {"id": "gov_d10", "name": "Training Programs", "description": "Formal climate training for board and senior management"},
    ],
    "strategy": [
        {"id": "str_d01", "name": "Risk Identification", "description": "Comprehensiveness of physical and transition risk identification"},
        {"id": "str_d02", "name": "Opportunity Mapping", "description": "Systematic identification and sizing of climate opportunities"},
        {"id": "str_d03", "name": "Time Horizon Coverage", "description": "Analysis across short, medium, and long-term horizons"},
        {"id": "str_d04", "name": "Financial Quantification", "description": "Quantified financial impacts of identified risks/opportunities"},
        {"id": "str_d05", "name": "Scenario Analysis", "description": "Quality and depth of climate scenario analysis (incl. 2C)"},
        {"id": "str_d06", "name": "Business Model Impact", "description": "Understanding of climate impact on business model"},
        {"id": "str_d07", "name": "Value Chain Assessment", "description": "Assessment of climate risks across the value chain"},
        {"id": "str_d08", "name": "Transition Planning", "description": "Existence and quality of climate transition plan"},
        {"id": "str_d09", "name": "Resilience Assessment", "description": "Assessment of strategic resilience under climate scenarios"},
        {"id": "str_d10", "name": "Capital Allocation", "description": "Integration of climate into capital allocation decisions"},
    ],
    "risk_management": [
        {"id": "rm_d01", "name": "Identification Process", "description": "Formal process for identifying climate-related risks"},
        {"id": "rm_d02", "name": "Assessment Methodology", "description": "Standardized methodology for risk assessment"},
        {"id": "rm_d03", "name": "Materiality Determination", "description": "Clear criteria for climate risk materiality"},
        {"id": "rm_d04", "name": "Risk Prioritization", "description": "Systematic prioritization of climate risks"},
        {"id": "rm_d05", "name": "Response Planning", "description": "Defined response strategies for each climate risk"},
        {"id": "rm_d06", "name": "ERM Integration", "description": "Climate risks integrated into enterprise risk management"},
        {"id": "rm_d07", "name": "Monitoring & Review", "description": "Regular monitoring and review of climate risks"},
        {"id": "rm_d08", "name": "Escalation Process", "description": "Clear escalation paths for climate risk events"},
        {"id": "rm_d09", "name": "Data Quality", "description": "Quality of data used for climate risk assessment"},
        {"id": "rm_d10", "name": "External Validation", "description": "Third-party review or validation of risk processes"},
    ],
    "metrics_targets": [
        {"id": "mt_d01", "name": "Scope 1 Emissions", "description": "Completeness and accuracy of Scope 1 GHG reporting"},
        {"id": "mt_d02", "name": "Scope 2 Emissions", "description": "Location-based and market-based Scope 2 reporting"},
        {"id": "mt_d03", "name": "Scope 3 Emissions", "description": "Coverage and quality of Scope 3 reporting (15 categories)"},
        {"id": "mt_d04", "name": "Cross-Industry Metrics", "description": "Reporting of ISSB 7 cross-industry climate metrics"},
        {"id": "mt_d05", "name": "Industry Metrics", "description": "Reporting of SASB-derived industry-specific metrics"},
        {"id": "mt_d06", "name": "Target Setting", "description": "Quality and ambition of climate targets"},
        {"id": "mt_d07", "name": "SBTi Alignment", "description": "Targets validated by Science Based Targets initiative"},
        {"id": "mt_d08", "name": "Performance Tracking", "description": "Regular tracking and reporting of progress against targets"},
        {"id": "mt_d09", "name": "Data Assurance", "description": "Third-party assurance of climate metrics"},
        {"id": "mt_d10", "name": "Internal Carbon Price", "description": "Use and disclosure of internal carbon price"},
    ],
}


# ---------------------------------------------------------------------------
# Peer benchmarks by sector (sample industry averages, score 0-100)
# ---------------------------------------------------------------------------

PEER_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "energy": {
        "governance": 72.0, "strategy": 68.0, "risk_management": 65.0,
        "metrics_targets": 75.0, "overall": 70.0, "sample_size": 85,
        "top_quartile": 85.0, "median": 68.0, "bottom_quartile": 52.0,
    },
    "transport": {
        "governance": 62.0, "strategy": 58.0, "risk_management": 55.0,
        "metrics_targets": 60.0, "overall": 59.0, "sample_size": 60,
        "top_quartile": 75.0, "median": 58.0, "bottom_quartile": 42.0,
    },
    "materials": {
        "governance": 65.0, "strategy": 60.0, "risk_management": 58.0,
        "metrics_targets": 62.0, "overall": 61.0, "sample_size": 70,
        "top_quartile": 78.0, "median": 60.0, "bottom_quartile": 45.0,
    },
    "agriculture": {
        "governance": 48.0, "strategy": 45.0, "risk_management": 42.0,
        "metrics_targets": 40.0, "overall": 44.0, "sample_size": 40,
        "top_quartile": 62.0, "median": 43.0, "bottom_quartile": 30.0,
    },
    "buildings": {
        "governance": 58.0, "strategy": 55.0, "risk_management": 52.0,
        "metrics_targets": 58.0, "overall": 56.0, "sample_size": 55,
        "top_quartile": 72.0, "median": 55.0, "bottom_quartile": 40.0,
    },
    "banking": {
        "governance": 75.0, "strategy": 70.0, "risk_management": 72.0,
        "metrics_targets": 68.0, "overall": 71.0, "sample_size": 90,
        "top_quartile": 88.0, "median": 70.0, "bottom_quartile": 55.0,
    },
    "insurance": {
        "governance": 72.0, "strategy": 68.0, "risk_management": 70.0,
        "metrics_targets": 65.0, "overall": 69.0, "sample_size": 50,
        "top_quartile": 85.0, "median": 68.0, "bottom_quartile": 52.0,
    },
    "asset_management": {
        "governance": 70.0, "strategy": 65.0, "risk_management": 62.0,
        "metrics_targets": 68.0, "overall": 66.0, "sample_size": 45,
        "top_quartile": 82.0, "median": 65.0, "bottom_quartile": 50.0,
    },
    "consumer_goods": {
        "governance": 55.0, "strategy": 50.0, "risk_management": 48.0,
        "metrics_targets": 52.0, "overall": 51.0, "sample_size": 65,
        "top_quartile": 68.0, "median": 50.0, "bottom_quartile": 35.0,
    },
    "technology": {
        "governance": 60.0, "strategy": 58.0, "risk_management": 55.0,
        "metrics_targets": 62.0, "overall": 59.0, "sample_size": 80,
        "top_quartile": 75.0, "median": 58.0, "bottom_quartile": 42.0,
    },
    "healthcare": {
        "governance": 50.0, "strategy": 45.0, "risk_management": 42.0,
        "metrics_targets": 45.0, "overall": 46.0, "sample_size": 35,
        "top_quartile": 65.0, "median": 45.0, "bottom_quartile": 32.0,
    },
}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class MaturityScore(BaseModel):
    """Per-pillar maturity score with dimension breakdown."""
    pillar: str = Field(...)
    pillar_score: float = Field(default=0.0, ge=0.0, le=100.0)
    maturity_level: str = Field(default="initial")
    dimensions: List[Dict[str, Any]] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class GapItem(BaseModel):
    """A single gap identified in the assessment."""
    id: str = Field(default_factory=_new_id)
    pillar: str = Field(...)
    dimension: str = Field(...)
    disclosure_code: str = Field(default="")
    severity: str = Field(default="medium")
    current_score: int = Field(default=1, ge=1, le=5)
    target_score: int = Field(default=3, ge=1, le=5)
    gap_size: int = Field(default=0)
    description: str = Field(default="")
    recommendation: str = Field(default="")


class PeerBenchmark(BaseModel):
    """Peer benchmarking result."""
    org_id: str = Field(...)
    sector: str = Field(...)
    org_score: float = Field(default=0.0)
    peer_average: float = Field(default=0.0)
    peer_median: float = Field(default=0.0)
    top_quartile: float = Field(default=0.0)
    bottom_quartile: float = Field(default=0.0)
    percentile_rank: int = Field(default=0, ge=0, le=100)
    sample_size: int = Field(default=0)
    pillar_comparison: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_now)


class ActionItem(BaseModel):
    """A single action in the improvement plan."""
    id: str = Field(default_factory=_new_id)
    title: str = Field(...)
    pillar: str = Field(default="")
    priority: int = Field(default=3, ge=1, le=5)
    effort_days: int = Field(default=5, ge=1)
    impact_score: float = Field(default=0.0)
    description: str = Field(default="")
    responsible: str = Field(default="")
    deadline: Optional[str] = Field(None)


class ActionPlan(BaseModel):
    """Prioritized action plan for gap remediation."""
    org_id: str = Field(...)
    total_actions: int = Field(default=0)
    total_effort_days: int = Field(default=0)
    actions: List[ActionItem] = Field(default_factory=list)
    timeline_months: int = Field(default=0)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class GapSummary(BaseModel):
    """Comprehensive gap analysis summary."""
    org_id: str = Field(...)
    overall_maturity_score: float = Field(default=0.0)
    overall_maturity_level: str = Field(default="initial")
    pillar_scores: Dict[str, float] = Field(default_factory=dict)
    total_gaps: int = Field(default=0)
    gaps_by_severity: Dict[str, int] = Field(default_factory=dict)
    gaps_by_pillar: Dict[str, int] = Field(default_factory=dict)
    top_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    improvement_potential: float = Field(default=0.0)
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# GapAnalysisEngine
# ---------------------------------------------------------------------------

class GapAnalysisEngine:
    """
    Gap Analysis Engine for TCFD disclosure maturity assessment.

    Evaluates organizational maturity across 40 dimensions (10 per pillar),
    identifies gaps, benchmarks against peers, and generates action plans.

    Attributes:
        config: Application configuration.
        _assessments: Historical assessments per organization.
        _org_scores: Cached dimension scores per organization.

    Example:
        >>> engine = GapAnalysisEngine(TCFDAppConfig())
        >>> assessment = engine.assess_maturity("org-1")
        >>> gaps = engine.identify_gaps("org-1")
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """
        Initialize the GapAnalysisEngine.

        Args:
            config: Optional application configuration.
        """
        self.config = config or TCFDAppConfig()
        self._assessments: Dict[str, List[GapAssessment]] = {}
        self._org_scores: Dict[str, Dict[str, int]] = {}
        logger.info("GapAnalysisEngine initialized")

    def assess_maturity(
        self,
        org_id: str,
        dimension_scores: Optional[Dict[str, int]] = None,
        tenant_id: str = "default",
    ) -> GapAssessment:
        """
        Assess organizational maturity across all four TCFD pillars.

        If dimension_scores are provided, they are used directly.
        Otherwise, default scores of 1 (initial) are assigned to all
        dimensions, and the caller should update via score_pillar_maturity.

        Args:
            org_id: Organization ID.
            dimension_scores: Optional dict mapping dimension_id -> score (1-5).
            tenant_id: Tenant identifier for multi-tenant isolation.

        Returns:
            GapAssessment with per-pillar scores and overall maturity.
        """
        scores = dimension_scores or self._org_scores.get(org_id, {})

        pillar_scores_raw: Dict[str, float] = {}
        for pillar, dimensions in MATURITY_DIMENSIONS.items():
            dim_scores = [
                max(1, min(5, scores.get(d["id"], 1)))
                for d in dimensions
            ]
            avg = sum(dim_scores) / max(len(dim_scores), 1)
            pillar_scores_raw[pillar] = round(avg, 1)

        overall_avg = sum(pillar_scores_raw.values()) / max(len(pillar_scores_raw), 1)
        overall_level = self._score_to_maturity(overall_avg)

        # Build domain MaturityScore list for the GapAssessment model
        pillar_key_to_enum = {
            "governance": TCFDPillar.GOVERNANCE,
            "strategy": TCFDPillar.STRATEGY,
            "risk_management": TCFDPillar.RISK_MANAGEMENT,
            "metrics_targets": TCFDPillar.METRICS_TARGETS,
        }
        domain_pillar_scores: List[DomainMaturityScore] = []
        for pillar_key, avg_score in pillar_scores_raw.items():
            level_str = self._score_to_maturity(avg_score)
            domain_pillar_scores.append(DomainMaturityScore(
                pillar=pillar_key_to_enum.get(pillar_key, TCFDPillar.GOVERNANCE),
                score=max(1, min(5, round(avg_score))),
                level=GovernanceMaturityLevel(level_str),
            ))

        overall_score_pct = round(overall_avg / 5.0 * 100)

        # Identify gaps
        gap_list: List[Dict[str, str]] = []
        critical_count = 0
        for pillar, dimensions in MATURITY_DIMENSIONS.items():
            for dim in dimensions:
                dim_score = scores.get(dim["id"], 1)
                if dim_score < 3:
                    severity = "critical" if dim_score == 1 else "high"
                    gap_list.append({
                        "pillar": pillar,
                        "dimension": dim["name"],
                        "current_score": str(dim_score),
                        "target_score": "3",
                        "severity": severity,
                    })
                    if severity == "critical":
                        critical_count += 1

        assessment = GapAssessment(
            tenant_id=tenant_id,
            org_id=org_id,
            pillar_scores=domain_pillar_scores,
            overall_score=overall_score_pct,
            overall_maturity=GovernanceMaturityLevel(overall_level),
            gaps=gap_list,
            total_gaps=len(gap_list),
            critical_gaps=critical_count,
        )

        # Store for historical tracking
        self._assessments.setdefault(org_id, []).append(assessment)
        if dimension_scores:
            self._org_scores[org_id] = dimension_scores

        logger.info(
            "Maturity assessment for org %s: overall=%.1f (%s), %d gaps",
            org_id, overall_avg, overall_level, len(gap_list),
        )
        return assessment

    def score_pillar_maturity(
        self,
        org_id: str,
        pillar: str,
        dimension_scores: Optional[Dict[str, int]] = None,
    ) -> MaturityScore:
        """
        Score maturity for a specific TCFD pillar with 10 dimensions.

        Args:
            org_id: Organization ID.
            pillar: Pillar key (governance, strategy, risk_management, metrics_targets).
            dimension_scores: Optional dict mapping dimension_id -> score (1-5).

        Returns:
            MaturityScore with dimension-level detail.

        Raises:
            ValueError: If pillar is not a valid TCFD pillar.
        """
        if pillar not in MATURITY_DIMENSIONS:
            raise ValueError(
                f"Invalid pillar '{pillar}'. "
                f"Valid: {list(MATURITY_DIMENSIONS.keys())}"
            )

        dimensions = MATURITY_DIMENSIONS[pillar]
        org_scores = self._org_scores.get(org_id, {})
        if dimension_scores:
            org_scores.update(dimension_scores)
            self._org_scores[org_id] = org_scores

        dim_results: List[Dict[str, Any]] = []
        total_score = 0.0
        strengths: List[str] = []
        weaknesses: List[str] = []

        for dim in dimensions:
            score = max(1, min(5, org_scores.get(dim["id"], 1)))
            total_score += score
            level = self._score_to_maturity(score)
            dim_results.append({
                "id": dim["id"],
                "name": dim["name"],
                "description": dim["description"],
                "score": score,
                "maturity_level": level,
            })
            if score >= 4:
                strengths.append(dim["name"])
            elif score <= 2:
                weaknesses.append(dim["name"])

        avg_score = total_score / max(len(dimensions), 1)
        pillar_pct = round(avg_score / 5.0 * 100, 1)

        result = MaturityScore(
            pillar=pillar,
            pillar_score=pillar_pct,
            maturity_level=self._score_to_maturity(avg_score),
            dimensions=dim_results,
            strengths=strengths,
            weaknesses=weaknesses,
        )

        logger.info(
            "Pillar maturity for org %s / %s: %.1f%% (%s)",
            org_id, pillar, pillar_pct, result.maturity_level,
        )
        return result

    def identify_gaps(
        self,
        org_id: str,
        target_level: int = 3,
    ) -> List[GapItem]:
        """
        Identify requirement-level gaps with severity classification.

        Args:
            org_id: Organization ID.
            target_level: Target maturity level (1-5, default 3 = Defined).

        Returns:
            List of GapItem objects with severity and recommendations.
        """
        org_scores = self._org_scores.get(org_id, {})
        gaps: List[GapItem] = []

        for pillar, dimensions in MATURITY_DIMENSIONS.items():
            for dim in dimensions:
                current = max(1, min(5, org_scores.get(dim["id"], 1)))
                if current < target_level:
                    gap_size = target_level - current
                    severity = (
                        "critical" if gap_size >= 3
                        else "high" if gap_size == 2
                        else "medium"
                    )
                    gaps.append(GapItem(
                        pillar=pillar,
                        dimension=dim["name"],
                        severity=severity,
                        current_score=current,
                        target_score=target_level,
                        gap_size=gap_size,
                        description=(
                            f"{dim['name']} is at level {current} ({self._score_to_maturity(current)}). "
                            f"Target is level {target_level} ({self._score_to_maturity(target_level)})."
                        ),
                        recommendation=self._generate_gap_recommendation(pillar, dim, current, target_level),
                    ))

        # Sort by severity then gap size
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: (severity_order.get(g.severity, 3), -g.gap_size))

        logger.info(
            "Identified %d gaps for org %s (target level %d)",
            len(gaps), org_id, target_level,
        )
        return gaps

    def benchmark_against_peers(
        self,
        org_id: str,
        sector: str,
    ) -> PeerBenchmark:
        """
        Benchmark organization against sector peers.

        Args:
            org_id: Organization ID.
            sector: Sector key (e.g. "energy", "banking").

        Returns:
            PeerBenchmark with percentile ranking and pillar comparison.
        """
        sector_key = sector.lower().replace(" ", "_")
        peer_data = PEER_BENCHMARKS.get(sector_key, PEER_BENCHMARKS.get("technology", {}))

        # Calculate org overall score
        org_scores = self._org_scores.get(org_id, {})
        all_scores = [max(1, min(5, org_scores.get(d["id"], 1)))
                      for dims in MATURITY_DIMENSIONS.values() for d in dims]
        org_avg = sum(all_scores) / max(len(all_scores), 1)
        org_overall = round(org_avg / 5.0 * 100, 1)

        peer_avg = peer_data.get("overall", 50.0)
        peer_median = peer_data.get("median", 50.0)
        top_q = peer_data.get("top_quartile", 75.0)
        bottom_q = peer_data.get("bottom_quartile", 25.0)
        sample = peer_data.get("sample_size", 50)

        # Compute percentile (approximate)
        if org_overall >= top_q:
            percentile = min(95, 75 + int((org_overall - top_q) / max(100 - top_q, 1) * 25))
        elif org_overall >= peer_median:
            percentile = 50 + int((org_overall - peer_median) / max(top_q - peer_median, 1) * 25)
        elif org_overall >= bottom_q:
            percentile = 25 + int((org_overall - bottom_q) / max(peer_median - bottom_q, 1) * 25)
        else:
            percentile = max(5, int(org_overall / max(bottom_q, 1) * 25))

        # Pillar comparison
        pillar_comparison: Dict[str, Dict[str, float]] = {}
        for pillar in MATURITY_DIMENSIONS:
            dims = MATURITY_DIMENSIONS[pillar]
            p_scores = [max(1, min(5, org_scores.get(d["id"], 1))) for d in dims]
            p_avg = sum(p_scores) / max(len(p_scores), 1)
            p_pct = round(p_avg / 5.0 * 100, 1)
            peer_pillar = peer_data.get(pillar, 50.0)
            pillar_comparison[pillar] = {
                "org_score": p_pct,
                "peer_average": peer_pillar,
                "delta": round(p_pct - peer_pillar, 1),
            }

        result = PeerBenchmark(
            org_id=org_id,
            sector=sector_key,
            org_score=org_overall,
            peer_average=peer_avg,
            peer_median=peer_median,
            top_quartile=top_q,
            bottom_quartile=bottom_q,
            percentile_rank=percentile,
            sample_size=sample,
            pillar_comparison=pillar_comparison,
        )

        logger.info(
            "Benchmark for org %s vs %s: score=%.1f%%, percentile=%d, peer_avg=%.1f%%",
            org_id, sector_key, org_overall, percentile, peer_avg,
        )
        return result

    def generate_action_plan(
        self,
        org_id: str,
        gaps: Optional[List[GapItem]] = None,
    ) -> ActionPlan:
        """
        Generate a prioritized action plan from identified gaps.

        Args:
            org_id: Organization ID.
            gaps: Optional pre-computed gap list. If None, identify_gaps is called.

        Returns:
            ActionPlan with prioritized actions and timeline.
        """
        if gaps is None:
            gaps = self.identify_gaps(org_id)

        actions: List[ActionItem] = []
        for gap in gaps:
            effort = self.estimate_effort_for_gap(gap)
            priority = {"critical": 1, "high": 2, "medium": 3, "low": 4}.get(gap.severity, 3)
            impact = gap.gap_size * 10.0  # 10 points per level improvement

            actions.append(ActionItem(
                title=f"Improve {gap.dimension} ({gap.pillar})",
                pillar=gap.pillar,
                priority=priority,
                effort_days=effort,
                impact_score=impact,
                description=gap.recommendation,
                responsible=self._suggest_responsible(gap.pillar),
            ))

        # Sort by priority then impact/effort ratio
        actions.sort(key=lambda a: (a.priority, -a.impact_score / max(a.effort_days, 1)))

        total_effort = sum(a.effort_days for a in actions)
        timeline_months = max(3, total_effort // 20)

        milestones = self._generate_milestones(actions, timeline_months)
        provenance = _sha256(f"{org_id}:{len(actions)}:{total_effort}")

        plan = ActionPlan(
            org_id=org_id,
            total_actions=len(actions),
            total_effort_days=total_effort,
            actions=actions,
            timeline_months=timeline_months,
            milestones=milestones,
            provenance_hash=provenance,
        )

        logger.info(
            "Generated action plan for org %s: %d actions, %d days, %d months",
            org_id, len(actions), total_effort, timeline_months,
        )
        return plan

    def estimate_effort(self, action: ActionItem) -> int:
        """
        Estimate effort in person-days for an action item.

        Args:
            action: ActionItem to estimate.

        Returns:
            Estimated person-days.
        """
        base_effort = {
            "governance": 8,
            "strategy": 15,
            "risk_management": 12,
            "metrics_targets": 18,
        }.get(action.pillar, 10)

        priority_multiplier = {1: 1.5, 2: 1.2, 3: 1.0, 4: 0.8, 5: 0.6}.get(action.priority, 1.0)

        return max(3, round(base_effort * priority_multiplier))

    def estimate_effort_for_gap(self, gap: GapItem) -> int:
        """
        Estimate effort in person-days for closing a gap.

        Args:
            gap: GapItem to estimate.

        Returns:
            Estimated person-days.
        """
        base_effort = {
            "governance": 8,
            "strategy": 15,
            "risk_management": 12,
            "metrics_targets": 18,
        }.get(gap.pillar, 10)

        gap_multiplier = gap.gap_size * 0.6
        return max(3, round(base_effort * gap_multiplier))

    def estimate_timeline(self, action_plan: ActionPlan) -> Dict[str, Any]:
        """
        Estimate timeline with milestones for an action plan.

        Args:
            action_plan: ActionPlan to estimate.

        Returns:
            Timeline dictionary with months and milestone breakdown.
        """
        total_days = action_plan.total_effort_days
        parallel_factor = 0.6  # Assume 60% of work can be parallelized
        effective_days = round(total_days * parallel_factor)
        months = max(3, effective_days // 20)

        milestones = self._generate_milestones(action_plan.actions, months)

        return {
            "total_effort_days": total_days,
            "effective_calendar_days": effective_days,
            "estimated_months": months,
            "parallel_factor": parallel_factor,
            "milestones": milestones,
            "start_date": date.today().isoformat(),
        }

    def track_progress(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Track progress over assessment cycles.

        Args:
            org_id: Organization ID.

        Returns:
            List of historical assessment summaries.
        """
        history = self._assessments.get(org_id, [])
        results: List[Dict[str, Any]] = []

        for assessment in history:
            pillar_dict = {
                ms.pillar.value: ms.score
                for ms in assessment.pillar_scores
            }
            overall_avg = (
                sum(ms.score for ms in assessment.pillar_scores)
                / max(len(assessment.pillar_scores), 1)
            )
            results.append({
                "assessment_id": assessment.id,
                "assessment_date": assessment.assessment_date.isoformat(),
                "overall_score": round(overall_avg, 1),
                "overall_maturity": assessment.overall_maturity.value,
                "pillar_scores": pillar_dict,
                "gap_count": len(assessment.gaps),
            })

        logger.info(
            "Progress tracking for org %s: %d assessments", org_id, len(results),
        )
        return results

    def get_gap_summary(self, org_id: str) -> GapSummary:
        """
        Get comprehensive gap analysis summary.

        Args:
            org_id: Organization ID.

        Returns:
            GapSummary with overall scores, gap counts, and top priorities.
        """
        gaps = self.identify_gaps(org_id)

        org_scores = self._org_scores.get(org_id, {})
        all_scores = [
            max(1, min(5, org_scores.get(d["id"], 1)))
            for dims in MATURITY_DIMENSIONS.values()
            for d in dims
        ]
        overall_avg = sum(all_scores) / max(len(all_scores), 1)
        overall_pct = round(overall_avg / 5.0 * 100, 1)

        pillar_scores: Dict[str, float] = {}
        for pillar, dimensions in MATURITY_DIMENSIONS.items():
            p_scores = [max(1, min(5, org_scores.get(d["id"], 1))) for d in dimensions]
            p_avg = sum(p_scores) / max(len(p_scores), 1)
            pillar_scores[pillar] = round(p_avg / 5.0 * 100, 1)

        gaps_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        gaps_by_pillar: Dict[str, int] = {}
        for gap in gaps:
            gaps_by_severity[gap.severity] = gaps_by_severity.get(gap.severity, 0) + 1
            gaps_by_pillar[gap.pillar] = gaps_by_pillar.get(gap.pillar, 0) + 1

        top_gaps = [
            {
                "pillar": g.pillar,
                "dimension": g.dimension,
                "severity": g.severity,
                "gap_size": g.gap_size,
                "recommendation": g.recommendation,
            }
            for g in gaps[:5]
        ]

        # Max possible improvement
        max_score = 100.0
        improvement = round(max_score - overall_pct, 1)

        return GapSummary(
            org_id=org_id,
            overall_maturity_score=overall_pct,
            overall_maturity_level=self._score_to_maturity(overall_avg),
            pillar_scores=pillar_scores,
            total_gaps=len(gaps),
            gaps_by_severity=gaps_by_severity,
            gaps_by_pillar=gaps_by_pillar,
            top_gaps=top_gaps,
            improvement_potential=improvement,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_maturity(score: float) -> str:
        """Map a numeric score to GovernanceMaturityLevel value."""
        if score >= 4.5:
            return GovernanceMaturityLevel.OPTIMIZED.value
        elif score >= 3.5:
            return GovernanceMaturityLevel.MANAGED.value
        elif score >= 2.5:
            return GovernanceMaturityLevel.DEFINED.value
        elif score >= 1.5:
            return GovernanceMaturityLevel.DEVELOPING.value
        return GovernanceMaturityLevel.INITIAL.value

    @staticmethod
    def _generate_gap_recommendation(
        pillar: str, dimension: Dict[str, str], current: int, target: int,
    ) -> str:
        """Generate a specific recommendation for closing a gap."""
        level_names = {1: "Initial", 2: "Developing", 3: "Defined", 4: "Managed", 5: "Optimized"}
        return (
            f"Advance {dimension['name']} from {level_names.get(current, 'Initial')} "
            f"to {level_names.get(target, 'Defined')} level. "
            f"Focus on: {dimension['description']}. "
            f"This requires formalizing processes, documenting procedures, "
            f"and establishing regular review cycles."
        )

    @staticmethod
    def _suggest_responsible(pillar: str) -> str:
        """Suggest a responsible role for a pillar improvement."""
        roles = {
            "governance": "Company Secretary / Board Liaison",
            "strategy": "Chief Strategy Officer / CSO",
            "risk_management": "Chief Risk Officer / CRO",
            "metrics_targets": "Head of Sustainability / ESG Lead",
        }
        return roles.get(pillar, "Sustainability Team")

    @staticmethod
    def _generate_milestones(
        actions: List[ActionItem], timeline_months: int,
    ) -> List[Dict[str, Any]]:
        """Generate milestones from action list."""
        if not actions:
            return []

        milestones: List[Dict[str, Any]] = []
        chunk_size = max(1, len(actions) // 3)

        phases = [
            ("Phase 1: Quick Wins", actions[:chunk_size]),
            ("Phase 2: Core Improvements", actions[chunk_size:chunk_size * 2]),
            ("Phase 3: Advanced Maturity", actions[chunk_size * 2:]),
        ]

        months_elapsed = 0
        for phase_name, phase_actions in phases:
            if not phase_actions:
                continue
            phase_effort = sum(a.effort_days for a in phase_actions)
            phase_months = max(1, timeline_months // 3)
            months_elapsed += phase_months
            milestones.append({
                "name": phase_name,
                "month": months_elapsed,
                "actions": len(phase_actions),
                "effort_days": phase_effort,
                "pillars": list(set(a.pillar for a in phase_actions)),
            })

        return milestones
