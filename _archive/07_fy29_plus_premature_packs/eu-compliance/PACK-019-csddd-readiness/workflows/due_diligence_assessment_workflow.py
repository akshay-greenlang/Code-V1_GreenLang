# -*- coding: utf-8 -*-
"""
CSDDD Due Diligence Assessment Workflow
===============================================

5-phase workflow for assessing corporate due diligence readiness under the
EU Corporate Sustainability Due Diligence Directive (CSDDD / CS3D).
Implements scope determination, policy review, gap analysis, risk
prioritization, and readiness scoring with full provenance tracking.

Phases:
    1. ScopeDetermination      -- Determine CSDDD applicability and scope
    2. PolicyReview            -- Review existing DD policies against Art. 5-11
    3. GapAnalysis             -- Identify gaps in current DD framework
    4. RiskPrioritization      -- Prioritize risks by severity and likelihood
    5. ReadinessScoring        -- Calculate overall CSDDD readiness score

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Articles 1-36 covering due diligence obligations
    - Art. 5: Due diligence policy
    - Art. 6: Identifying adverse impacts
    - Art. 7: Preventing potential adverse impacts
    - Art. 8: Bringing actual adverse impacts to an end
    - Art. 9: Remediation
    - Art. 10: Meaningful engagement with stakeholders
    - Art. 11: Complaints procedure (grievance mechanism)

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
    """Phases of the due diligence assessment workflow."""
    SCOPE_DETERMINATION = "scope_determination"
    POLICY_REVIEW = "policy_review"
    GAP_ANALYSIS = "gap_analysis"
    RISK_PRIORITIZATION = "risk_prioritization"
    READINESS_SCORING = "readiness_scoring"

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

class CompanySize(str, Enum):
    """Company size tier for CSDDD scope determination."""
    GROUP_1 = "group_1"        # >1000 employees and >EUR 450M turnover
    GROUP_2 = "group_2"        # >500 employees in high-impact sectors
    NON_EU_GROUP_1 = "non_eu_group_1"  # Non-EU >EUR 450M turnover in EU
    NON_EU_GROUP_2 = "non_eu_group_2"  # Non-EU >EUR 80M turnover in EU
    OUT_OF_SCOPE = "out_of_scope"

class ArticleStatus(str, Enum):
    """Compliance status per CSDDD article."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"

class RiskSeverity(str, Enum):
    """Severity level for identified risks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

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

class CompanyProfile(BaseModel):
    """Company profile for CSDDD scope determination."""
    company_id: str = Field(default="", description="Unique company identifier")
    company_name: str = Field(default="", description="Legal entity name")
    headquarters_country: str = Field(default="", description="ISO 3166-1 alpha-2")
    is_eu_based: bool = Field(default=True, description="Whether HQ is in EU/EEA")
    employee_count: int = Field(default=0, ge=0, description="Total employees")
    net_turnover_eur: float = Field(default=0.0, ge=0.0, description="Net turnover in EUR")
    eu_turnover_eur: float = Field(default=0.0, ge=0.0, description="EU net turnover in EUR")
    sector: str = Field(default="", description="Primary sector of activity")
    is_high_impact_sector: bool = Field(default=False)
    is_regulated_financial: bool = Field(default=False)
    reporting_year: int = Field(default=2026, ge=2024, le=2050)

class PolicyRecord(BaseModel):
    """Record of an existing due diligence policy."""
    policy_id: str = Field(default_factory=lambda: f"pol-{_new_uuid()[:8]}")
    policy_name: str = Field(default="", description="Policy title")
    csddd_article: str = Field(default="", description="Mapped CSDDD article (e.g., art_5)")
    scope: str = Field(default="", description="Policy scope (own ops, supply chain, etc.)")
    last_updated: str = Field(default="", description="ISO date of last update")
    covers_human_rights: bool = Field(default=False)
    covers_environment: bool = Field(default=False)
    has_board_approval: bool = Field(default=False)
    effectiveness_score: float = Field(default=0.0, ge=0.0, le=100.0)

class RiskIndicator(BaseModel):
    """Risk indicator for due diligence assessment."""
    indicator_id: str = Field(default_factory=lambda: f"ri-{_new_uuid()[:8]}")
    indicator_name: str = Field(default="", description="Risk indicator label")
    category: str = Field(default="", description="human_rights or environment")
    severity: RiskSeverity = Field(default=RiskSeverity.MEDIUM)
    likelihood: float = Field(default=0.5, ge=0.0, le=1.0, description="Likelihood 0-1")
    affected_stakeholders: List[str] = Field(default_factory=list)
    geographic_scope: List[str] = Field(default_factory=list)
    existing_mitigation: str = Field(default="", description="Current mitigation measures")

class ArticleAssessment(BaseModel):
    """Assessment result for a specific CSDDD article."""
    article: str = Field(..., description="Article reference (e.g., art_5)")
    article_title: str = Field(default="", description="Article descriptive title")
    status: ArticleStatus = Field(default=ArticleStatus.NON_COMPLIANT)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    gaps: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)

class DueDiligenceAssessmentInput(BaseModel):
    """Input data model for DueDiligenceAssessmentWorkflow."""
    company_profile: CompanyProfile = Field(
        default_factory=CompanyProfile, description="Company profile data"
    )
    existing_policies: List[PolicyRecord] = Field(
        default_factory=list, description="Current DD policies"
    )
    risk_indicators: List[RiskIndicator] = Field(
        default_factory=list, description="Identified risk indicators"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class DueDiligenceAssessmentResult(BaseModel):
    """Complete result from due diligence assessment workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="due_diligence_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    # Scope determination
    in_scope: bool = Field(default=False)
    company_size_tier: str = Field(default="out_of_scope")
    compliance_deadline: str = Field(default="")
    # Article-by-article status
    article_assessments: List[ArticleAssessment] = Field(default_factory=list)
    # Readiness
    overall_readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    readiness_level: str = Field(default="not_ready")
    # Gaps and actions
    total_gaps: int = Field(default=0, ge=0)
    prioritized_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    # Risk summary
    critical_risks: int = Field(default=0, ge=0)
    high_risks: int = Field(default=0, ge=0)
    medium_risks: int = Field(default=0, ge=0)
    low_risks: int = Field(default=0, ge=0)
    reporting_year: int = Field(default=2026)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")

# =============================================================================
# CSDDD ARTICLE DEFINITIONS
# =============================================================================

CSDDD_ARTICLES: List[Dict[str, str]] = [
    {"article": "art_5", "title": "Due diligence policy"},
    {"article": "art_6", "title": "Identifying adverse impacts"},
    {"article": "art_7", "title": "Preventing potential adverse impacts"},
    {"article": "art_8", "title": "Bringing actual adverse impacts to an end"},
    {"article": "art_9", "title": "Remediation"},
    {"article": "art_10", "title": "Meaningful stakeholder engagement"},
    {"article": "art_11", "title": "Complaints procedure"},
    {"article": "art_15", "title": "Climate transition plan"},
    {"article": "art_22", "title": "Civil liability"},
]

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class DueDiligenceAssessmentWorkflow:
    """
    5-phase CSDDD due diligence assessment workflow.

    Orchestrates scope determination, policy review, gap analysis, risk
    prioritization, and readiness scoring for CSDDD compliance. Produces
    article-by-article compliance status, prioritized gaps, and action items.

    Zero-hallucination: all scoring uses deterministic arithmetic.
    No LLM in numeric calculation paths.

    Example:
        >>> wf = DueDiligenceAssessmentWorkflow()
        >>> inp = DueDiligenceAssessmentInput(
        ...     company_profile=CompanyProfile(employee_count=1500, net_turnover_eur=500_000_000)
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.in_scope is True
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize DueDiligenceAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._article_assessments: List[ArticleAssessment] = []
        self._prioritized_gaps: List[Dict[str, Any]] = []
        self._action_items: List[Dict[str, Any]] = []
        self._in_scope: bool = False
        self._company_tier: str = "out_of_scope"
        self._deadline: str = ""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.SCOPE_DETERMINATION.value, "description": "Determine CSDDD applicability"},
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review policies against articles"},
            {"name": WorkflowPhase.GAP_ANALYSIS.value, "description": "Identify compliance gaps"},
            {"name": WorkflowPhase.RISK_PRIORITIZATION.value, "description": "Prioritize risks"},
            {"name": WorkflowPhase.READINESS_SCORING.value, "description": "Calculate readiness score"},
        ]

    def validate_inputs(self, input_data: DueDiligenceAssessmentInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        cp = input_data.company_profile
        if not cp.company_name:
            issues.append("Company name is required")
        if cp.employee_count <= 0:
            issues.append("Employee count must be positive")
        if cp.net_turnover_eur <= 0:
            issues.append("Net turnover must be positive")
        return issues

    async def execute(
        self,
        input_data: Optional[DueDiligenceAssessmentInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> DueDiligenceAssessmentResult:
        """
        Execute the 5-phase due diligence assessment workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            DueDiligenceAssessmentResult with readiness score and gaps.
        """
        if input_data is None:
            input_data = DueDiligenceAssessmentInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting DD assessment workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_scope_determination(input_data))
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_gap_analysis(input_data))
            phase_results.append(await self._phase_risk_prioritization(input_data))
            phase_results.append(await self._phase_readiness_scoring(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("DD assessment workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        # Risk counts
        risk_counts = self._count_risks(input_data.risk_indicators)

        # Calculate overall readiness
        readiness = self._calculate_overall_readiness()

        result = DueDiligenceAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            in_scope=self._in_scope,
            company_size_tier=self._company_tier,
            compliance_deadline=self._deadline,
            article_assessments=self._article_assessments,
            overall_readiness_score=readiness,
            readiness_level=self._readiness_level(readiness),
            total_gaps=sum(len(a.gaps) for a in self._article_assessments),
            prioritized_gaps=self._prioritized_gaps,
            action_items=self._action_items,
            critical_risks=risk_counts.get("critical", 0),
            high_risks=risk_counts.get("high", 0),
            medium_risks=risk_counts.get("medium", 0),
            low_risks=risk_counts.get("low", 0),
            reporting_year=input_data.company_profile.reporting_year,
            executed_at=utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "DD assessment %s completed in %.2fs: readiness=%.1f%%",
            self.workflow_id, elapsed, readiness,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Scope Determination
    # -------------------------------------------------------------------------

    async def _phase_scope_determination(
        self, input_data: DueDiligenceAssessmentInput,
    ) -> PhaseResult:
        """Determine CSDDD applicability based on company profile."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        cp = input_data.company_profile

        # Determine tier based on CSDDD thresholds
        if cp.is_eu_based:
            if cp.employee_count > 1000 and cp.net_turnover_eur > 450_000_000:
                self._company_tier = CompanySize.GROUP_1.value
                self._in_scope = True
                self._deadline = "2027-07-26"
            elif cp.employee_count > 500 and cp.is_high_impact_sector:
                self._company_tier = CompanySize.GROUP_2.value
                self._in_scope = True
                self._deadline = "2028-07-26"
            else:
                self._company_tier = CompanySize.OUT_OF_SCOPE.value
                self._in_scope = False
                self._deadline = ""
        else:
            # Non-EU companies: based on EU turnover
            if cp.eu_turnover_eur > 450_000_000:
                self._company_tier = CompanySize.NON_EU_GROUP_1.value
                self._in_scope = True
                self._deadline = "2029-07-26"
            elif cp.eu_turnover_eur > 80_000_000 and cp.is_high_impact_sector:
                self._company_tier = CompanySize.NON_EU_GROUP_2.value
                self._in_scope = True
                self._deadline = "2029-07-26"
            else:
                self._company_tier = CompanySize.OUT_OF_SCOPE.value
                self._in_scope = False
                self._deadline = ""

        outputs["in_scope"] = self._in_scope
        outputs["company_tier"] = self._company_tier
        outputs["compliance_deadline"] = self._deadline
        outputs["employee_count"] = cp.employee_count
        outputs["net_turnover_eur"] = cp.net_turnover_eur
        outputs["is_eu_based"] = cp.is_eu_based
        outputs["is_high_impact_sector"] = cp.is_high_impact_sector

        if not self._in_scope:
            warnings.append("Company appears out of CSDDD scope based on current thresholds")
        if cp.is_regulated_financial:
            warnings.append("Financial sector entity -- specific provisions apply under Art. 2")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 ScopeDetermination: in_scope=%s, tier=%s",
            self._in_scope, self._company_tier,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SCOPE_DETERMINATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Policy Review
    # -------------------------------------------------------------------------

    async def _phase_policy_review(
        self, input_data: DueDiligenceAssessmentInput,
    ) -> PhaseResult:
        """Review existing policies against CSDDD article requirements."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        policies = input_data.existing_policies

        # Map policies to articles
        article_coverage: Dict[str, List[PolicyRecord]] = {}
        for article_def in CSDDD_ARTICLES:
            art_key = article_def["article"]
            article_coverage[art_key] = [
                p for p in policies if p.csddd_article == art_key
            ]

        outputs["policies_count"] = len(policies)
        outputs["articles_covered"] = sum(
            1 for v in article_coverage.values() if v
        )
        outputs["articles_total"] = len(CSDDD_ARTICLES)
        outputs["coverage_pct"] = round(
            (outputs["articles_covered"] / outputs["articles_total"]) * 100, 1
        ) if outputs["articles_total"] > 0 else 0.0

        # Check policy quality indicators
        hr_coverage = sum(1 for p in policies if p.covers_human_rights)
        env_coverage = sum(1 for p in policies if p.covers_environment)
        board_approved = sum(1 for p in policies if p.has_board_approval)

        outputs["human_rights_policies"] = hr_coverage
        outputs["environment_policies"] = env_coverage
        outputs["board_approved_policies"] = board_approved
        outputs["avg_effectiveness"] = round(
            sum(p.effectiveness_score for p in policies) / len(policies), 1
        ) if policies else 0.0

        if not policies:
            warnings.append("No DD policies found -- Art. 5 compliance at risk")
        if hr_coverage == 0:
            warnings.append("No policies covering human rights")
        if env_coverage == 0:
            warnings.append("No policies covering environmental impacts")
        if board_approved == 0 and policies:
            warnings.append("No policies with board-level approval")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 PolicyReview: %d policies, %d/%d articles covered",
            len(policies), outputs["articles_covered"], outputs["articles_total"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(
        self, input_data: DueDiligenceAssessmentInput,
    ) -> PhaseResult:
        """Identify gaps in current DD framework against each CSDDD article."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._article_assessments = []

        policies = input_data.existing_policies
        policy_map: Dict[str, List[PolicyRecord]] = {}
        for p in policies:
            policy_map.setdefault(p.csddd_article, []).append(p)

        for article_def in CSDDD_ARTICLES:
            art_key = article_def["article"]
            art_title = article_def["title"]
            matched = policy_map.get(art_key, [])

            gaps: List[str] = []
            actions: List[str] = []
            score = 0.0

            if not matched:
                gaps.append(f"No policy mapped to {art_key}: {art_title}")
                actions.append(f"Develop policy for {art_title}")
                score = 0.0
                status = ArticleStatus.NON_COMPLIANT
            else:
                # Evaluate based on quality criteria
                avg_eff = sum(p.effectiveness_score for p in matched) / len(matched)
                has_hr = any(p.covers_human_rights for p in matched)
                has_env = any(p.covers_environment for p in matched)
                has_board = any(p.has_board_approval for p in matched)

                criteria_met = sum([has_hr, has_env, has_board])

                if not has_hr:
                    gaps.append(f"{art_key}: Missing human rights coverage")
                    actions.append(f"Extend {art_key} policy to include human rights")
                if not has_env:
                    gaps.append(f"{art_key}: Missing environmental coverage")
                    actions.append(f"Extend {art_key} policy to include environment")
                if not has_board:
                    gaps.append(f"{art_key}: No board-level approval")
                    actions.append(f"Obtain board approval for {art_key} policy")

                # Score: 40% effectiveness + 60% criteria coverage
                criteria_score = (criteria_met / 3.0) * 100
                score = round(0.4 * avg_eff + 0.6 * criteria_score, 1)

                if score >= 80:
                    status = ArticleStatus.COMPLIANT
                elif score >= 40:
                    status = ArticleStatus.PARTIALLY_COMPLIANT
                else:
                    status = ArticleStatus.NON_COMPLIANT

            self._article_assessments.append(ArticleAssessment(
                article=art_key,
                article_title=art_title,
                status=status,
                score=score,
                gaps=gaps,
                action_items=actions,
            ))

        # Build prioritized gaps list
        all_gaps: List[Dict[str, Any]] = []
        for assessment in self._article_assessments:
            for gap in assessment.gaps:
                priority = "critical" if assessment.score < 20 else (
                    "high" if assessment.score < 50 else "medium"
                )
                all_gaps.append({
                    "article": assessment.article,
                    "gap": gap,
                    "priority": priority,
                    "current_score": assessment.score,
                })

        self._prioritized_gaps = sorted(
            all_gaps, key=lambda g: {"critical": 0, "high": 1, "medium": 2}.get(g["priority"], 3)
        )

        # Build action items
        self._action_items = []
        for assessment in self._article_assessments:
            for action in assessment.action_items:
                self._action_items.append({
                    "article": assessment.article,
                    "action": action,
                    "priority": "critical" if assessment.score < 20 else (
                        "high" if assessment.score < 50 else "medium"
                    ),
                })

        compliant_count = sum(
            1 for a in self._article_assessments if a.status == ArticleStatus.COMPLIANT
        )
        partial_count = sum(
            1 for a in self._article_assessments if a.status == ArticleStatus.PARTIALLY_COMPLIANT
        )

        outputs["articles_assessed"] = len(self._article_assessments)
        outputs["compliant_articles"] = compliant_count
        outputs["partially_compliant_articles"] = partial_count
        outputs["non_compliant_articles"] = len(self._article_assessments) - compliant_count - partial_count
        outputs["total_gaps"] = len(self._prioritized_gaps)
        outputs["total_action_items"] = len(self._action_items)

        if outputs["non_compliant_articles"] > 0:
            warnings.append(
                f"{outputs['non_compliant_articles']} articles are non-compliant"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 GapAnalysis: %d gaps, %d/%d compliant",
            outputs["total_gaps"], compliant_count, outputs["articles_assessed"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.GAP_ANALYSIS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Risk Prioritization
    # -------------------------------------------------------------------------

    async def _phase_risk_prioritization(
        self, input_data: DueDiligenceAssessmentInput,
    ) -> PhaseResult:
        """Prioritize risks by severity and likelihood per Art. 6."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        indicators = input_data.risk_indicators

        # Compute risk score: severity weight * likelihood
        severity_weights = {
            RiskSeverity.CRITICAL: 4.0,
            RiskSeverity.HIGH: 3.0,
            RiskSeverity.MEDIUM: 2.0,
            RiskSeverity.LOW: 1.0,
        }

        scored_risks: List[Dict[str, Any]] = []
        for ind in indicators:
            weight = severity_weights.get(ind.severity, 2.0)
            risk_score = round(weight * ind.likelihood * 25, 1)  # Scale to 0-100
            scored_risks.append({
                "indicator_id": ind.indicator_id,
                "indicator_name": ind.indicator_name,
                "category": ind.category,
                "severity": ind.severity.value,
                "likelihood": ind.likelihood,
                "risk_score": risk_score,
                "affected_stakeholders": ind.affected_stakeholders,
                "geographic_scope": ind.geographic_scope,
                "has_mitigation": bool(ind.existing_mitigation),
            })

        # Sort by risk score descending
        scored_risks.sort(key=lambda r: r["risk_score"], reverse=True)

        # Categorize
        hr_risks = [r for r in scored_risks if r["category"] == "human_rights"]
        env_risks = [r for r in scored_risks if r["category"] == "environment"]

        outputs["total_risks"] = len(scored_risks)
        outputs["human_rights_risks"] = len(hr_risks)
        outputs["environment_risks"] = len(env_risks)
        outputs["top_5_risks"] = scored_risks[:5]
        outputs["risks_with_mitigation"] = sum(1 for r in scored_risks if r["has_mitigation"])
        outputs["risks_without_mitigation"] = sum(1 for r in scored_risks if not r["has_mitigation"])
        outputs["avg_risk_score"] = round(
            sum(r["risk_score"] for r in scored_risks) / len(scored_risks), 1
        ) if scored_risks else 0.0

        if any(r["risk_score"] > 75 and not r["has_mitigation"] for r in scored_risks):
            warnings.append("Critical risks identified without mitigation measures")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 RiskPrioritization: %d risks scored, avg=%.1f",
            len(scored_risks), outputs["avg_risk_score"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.RISK_PRIORITIZATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Readiness Scoring
    # -------------------------------------------------------------------------

    async def _phase_readiness_scoring(
        self, input_data: DueDiligenceAssessmentInput,
    ) -> PhaseResult:
        """Calculate overall CSDDD readiness score."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Component scores (weighted)
        # 1. Article compliance (40%)
        article_scores = [a.score for a in self._article_assessments]
        avg_article_score = (
            sum(article_scores) / len(article_scores) if article_scores else 0.0
        )

        # 2. Policy coverage (20%)
        policies = input_data.existing_policies
        policy_score = min(100.0, (len(policies) / max(len(CSDDD_ARTICLES), 1)) * 100)

        # 3. Risk mitigation (20%)
        indicators = input_data.risk_indicators
        mitigated = sum(1 for i in indicators if i.existing_mitigation)
        mitigation_score = (mitigated / len(indicators) * 100) if indicators else 50.0

        # 4. Governance readiness (20%)
        board_approved = sum(1 for p in policies if p.has_board_approval)
        governance_score = min(100.0, (board_approved / max(len(CSDDD_ARTICLES), 1)) * 100)

        overall = round(
            0.40 * avg_article_score
            + 0.20 * policy_score
            + 0.20 * mitigation_score
            + 0.20 * governance_score,
            1,
        )

        outputs["article_compliance_score"] = round(avg_article_score, 1)
        outputs["policy_coverage_score"] = round(policy_score, 1)
        outputs["risk_mitigation_score"] = round(mitigation_score, 1)
        outputs["governance_readiness_score"] = round(governance_score, 1)
        outputs["overall_readiness_score"] = overall
        outputs["readiness_level"] = self._readiness_level(overall)
        outputs["weight_breakdown"] = {
            "article_compliance": 0.40,
            "policy_coverage": 0.20,
            "risk_mitigation": 0.20,
            "governance_readiness": 0.20,
        }

        if overall < 30:
            warnings.append("CRITICAL: Readiness score below 30% -- significant work required")
        elif overall < 60:
            warnings.append("Readiness score below 60% -- moderate improvements needed")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 ReadinessScoring: overall=%.1f%% (%s)",
            overall, outputs["readiness_level"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.READINESS_SCORING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _count_risks(self, indicators: List[RiskIndicator]) -> Dict[str, int]:
        """Count risk indicators by severity level."""
        counts: Dict[str, int] = {
            "critical": 0, "high": 0, "medium": 0, "low": 0,
        }
        for ind in indicators:
            counts[ind.severity.value] = counts.get(ind.severity.value, 0) + 1
        return counts

    def _calculate_overall_readiness(self) -> float:
        """Calculate overall readiness from article assessments."""
        if not self._article_assessments:
            return 0.0
        return round(
            sum(a.score for a in self._article_assessments) / len(self._article_assessments),
            1,
        )

    @staticmethod
    def _readiness_level(score: float) -> str:
        """Map numeric score to readiness level label."""
        if score >= 80:
            return "ready"
        elif score >= 60:
            return "mostly_ready"
        elif score >= 40:
            return "partially_ready"
        elif score >= 20:
            return "early_stage"
        return "not_ready"

    def _compute_provenance(self, result: DueDiligenceAssessmentResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
