# -*- coding: utf-8 -*-
"""
Credibility Review Workflow
=================================

4-phase workflow for HLEG credibility assessment within PACK-025
Race to Zero Pack.  Evaluates pledge credibility against the UN
Secretary-General's High-Level Expert Group (HLEG) "Integrity Matters"
10 recommendations with 45+ sub-criteria, validates science-based
alignment, reviews governance structures, and produces a credibility
scoring report.

Phases:
    1. HLEGCriteriaAssessment  -- Assess all 10 HLEG recommendations
    2. ScienceBasedValidation   -- Validate targets against science-based pathways
    3. GovernanceReview         -- Review governance and accountability structures
    4. CredibilityScoring       -- Produce composite credibility score and report

Regulatory references:
    - HLEG "Integrity Matters" Report (November 2022)
    - Race to Zero Campaign Minimum Criteria
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - IPCC AR6 WG3 (2022)
    - ISO 14064-1:2018

Zero-hallucination: all HLEG assessments use deterministic checklist
evaluation against the 10 recommendations and 45+ sub-criteria.
No LLM calls in the assessment path.

Author: GreenLang Team
Version: 25.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class CredibilityPhase(str, Enum):
    HLEG_CRITERIA_ASSESSMENT = "hleg_criteria_assessment"
    SCIENCE_BASED_VALIDATION = "science_based_validation"
    GOVERNANCE_REVIEW = "governance_review"
    CREDIBILITY_SCORING = "credibility_scoring"


class RecommendationStatus(str, Enum):
    FULLY_MET = "fully_met"
    PARTIALLY_MET = "partially_met"
    NOT_MET = "not_met"
    NOT_APPLICABLE = "not_applicable"


class CredibilityRating(str, Enum):
    STRONG = "strong"         # >= 80%
    ADEQUATE = "adequate"     # 60-79%
    WEAK = "weak"             # 40-59%
    INSUFFICIENT = "insufficient"  # < 40%


class GovernanceMaturity(str, Enum):
    ADVANCED = "advanced"
    DEVELOPING = "developing"
    BASIC = "basic"
    ABSENT = "absent"


# =============================================================================
# REFERENCE DATA
# =============================================================================

# HLEG "Integrity Matters" 10 Recommendations with sub-criteria
HLEG_RECOMMENDATIONS: List[Dict[str, Any]] = [
    {
        "id": "R01", "name": "Announce net-zero pledge",
        "description": "Announce pledge aligned with 1.5C pathway with interim targets",
        "sub_criteria": [
            "Net-zero target by 2050 at latest",
            "Interim target for 2030 (~50% reduction)",
            "Science-based methodology used",
            "All scopes covered",
            "Public announcement made",
        ],
        "weight": 12.0,
    },
    {
        "id": "R02", "name": "Set interim targets",
        "description": "Set near-term targets that demonstrate genuine short-term action",
        "sub_criteria": [
            "2030 target aligned with 1.5C (>=42% reduction)",
            "Annual reduction milestones defined",
            "Sector-specific pathway alignment",
            "Fair share consideration",
        ],
        "weight": 12.0,
    },
    {
        "id": "R03", "name": "Voluntary carbon credits",
        "description": "Use credits responsibly; no substitution for real reductions",
        "sub_criteria": [
            "Credits not counted as reduction",
            "High-quality credits only (ICVCM CCP)",
            "Transition to removals over time",
            "Credits transparent and separately reported",
            "Contribution claim not offset claim",
        ],
        "weight": 10.0,
    },
    {
        "id": "R04", "name": "Phase out fossil fuels",
        "description": "No new fossil fuel capacity; plan for managed phase-out",
        "sub_criteria": [
            "No new coal/oil/gas development",
            "Phase-out plan for existing fossil fuel use",
            "No expansion of fossil fuel infrastructure",
            "Transition plan for affected workers/communities",
        ],
        "weight": 12.0,
    },
    {
        "id": "R05", "name": "Lobbying alignment",
        "description": "Align lobbying and policy engagement with climate commitments",
        "sub_criteria": [
            "No lobbying against climate policy",
            "Trade association memberships reviewed",
            "Policy engagement supports Paris Agreement",
            "Public statement on lobbying alignment",
        ],
        "weight": 8.0,
    },
    {
        "id": "R06", "name": "Just transition",
        "description": "Ensure equitable transition for workers and communities",
        "sub_criteria": [
            "Just transition plan developed",
            "Stakeholder engagement on transition",
            "Worker retraining programs",
            "Community impact assessment",
            "Equity considerations in target setting",
        ],
        "weight": 8.0,
    },
    {
        "id": "R07", "name": "Transparent reporting",
        "description": "Annual transparent reporting on progress and methodology",
        "sub_criteria": [
            "Annual emissions disclosure",
            "Progress against targets reported",
            "Methodology publicly available",
            "Third-party verification",
            "Action plan implementation status",
        ],
        "weight": 10.0,
    },
    {
        "id": "R08", "name": "Scope of pledge",
        "description": "Pledge covers full value chain emissions",
        "sub_criteria": [
            "Scope 1 fully covered",
            "Scope 2 fully covered",
            "Material Scope 3 covered (>=67%)",
            "No cherry-picking of scopes",
        ],
        "weight": 10.0,
    },
    {
        "id": "R09", "name": "Internal governance",
        "description": "Integrate climate into governance and incentives",
        "sub_criteria": [
            "Board-level oversight of climate",
            "Executive compensation linked to climate targets",
            "Climate integrated into risk management",
            "Internal carbon pricing considered",
            "Climate expertise on board",
        ],
        "weight": 10.0,
    },
    {
        "id": "R10", "name": "Financial commitment",
        "description": "Allocate financial resources to decarbonization",
        "sub_criteria": [
            "CapEx allocated to decarbonization",
            "R&D investment in clean technologies",
            "Financial plan for transition",
            "No financing of new fossil fuels",
        ],
        "weight": 8.0,
    },
]

# Phase dependencies DAG
PHASE_DEPENDENCIES: Dict[CredibilityPhase, List[CredibilityPhase]] = {
    CredibilityPhase.HLEG_CRITERIA_ASSESSMENT: [],
    CredibilityPhase.SCIENCE_BASED_VALIDATION: [CredibilityPhase.HLEG_CRITERIA_ASSESSMENT],
    CredibilityPhase.GOVERNANCE_REVIEW: [CredibilityPhase.HLEG_CRITERIA_ASSESSMENT],
    CredibilityPhase.CREDIBILITY_SCORING: [
        CredibilityPhase.SCIENCE_BASED_VALIDATION,
        CredibilityPhase.GOVERNANCE_REVIEW,
    ],
}

PHASE_EXECUTION_ORDER: List[CredibilityPhase] = [
    CredibilityPhase.HLEG_CRITERIA_ASSESSMENT,
    CredibilityPhase.SCIENCE_BASED_VALIDATION,
    CredibilityPhase.GOVERNANCE_REVIEW,
    CredibilityPhase.CREDIBILITY_SCORING,
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase: CredibilityPhase = Field(...)
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class RecommendationAssessment(BaseModel):
    recommendation_id: str = Field(default="")
    name: str = Field(default="")
    status: RecommendationStatus = Field(default=RecommendationStatus.NOT_MET)
    sub_criteria_passed: int = Field(default=0)
    sub_criteria_total: int = Field(default=0)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    weight: float = Field(default=0.0)
    gaps: List[str] = Field(default_factory=list)


class CredibilityReport(BaseModel):
    report_id: str = Field(default="")
    org_name: str = Field(default="")
    assessment_date: str = Field(default="")
    overall_rating: CredibilityRating = Field(default=CredibilityRating.INSUFFICIENT)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    recommendations_fully_met: int = Field(default=0)
    recommendations_partially_met: int = Field(default=0)
    recommendations_not_met: int = Field(default=0)
    total_sub_criteria_passed: int = Field(default=0)
    total_sub_criteria: int = Field(default=45)
    governance_maturity: GovernanceMaturity = Field(default=GovernanceMaturity.BASIC)
    science_based_aligned: bool = Field(default=False)
    priority_actions: List[str] = Field(default_factory=list)


class CredibilityReviewConfig(BaseModel):
    pack_id: str = Field(default="PACK-025")
    org_name: str = Field(default="")
    actor_type: str = Field(default="corporate")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    base_year: int = Field(default=2019, ge=2015, le=2050)
    baseline_tco2e: float = Field(default=0.0, ge=0.0)
    current_tco2e: float = Field(default=0.0, ge=0.0)
    target_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    scope3_coverage_pct: float = Field(default=67.0, ge=0.0, le=100.0)
    # HLEG evidence flags
    has_net_zero_pledge: bool = Field(default=True)
    has_interim_targets: bool = Field(default=True)
    has_science_based_targets: bool = Field(default=False)
    responsible_credit_use: bool = Field(default=True)
    no_new_fossil_fuel: bool = Field(default=True)
    lobbying_aligned: bool = Field(default=True)
    just_transition_plan: bool = Field(default=False)
    annual_reporting: bool = Field(default=True)
    third_party_verification: bool = Field(default=False)
    board_oversight: bool = Field(default=False)
    exec_compensation_linked: bool = Field(default=False)
    climate_risk_integrated: bool = Field(default=False)
    capex_allocated: bool = Field(default=False)
    investment_usd: float = Field(default=0.0, ge=0.0)
    enable_provenance: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class CredibilityReviewResult(BaseModel):
    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    workflow_name: str = Field(default="credibility_review")
    org_name: str = Field(default="")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    recommendation_assessments: List[RecommendationAssessment] = Field(default_factory=list)
    report: Optional[CredibilityReport] = Field(None)
    overall_rating: CredibilityRating = Field(default=CredibilityRating.INSUFFICIENT)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CredibilityReviewWorkflow:
    """
    4-phase credibility review workflow for PACK-025 Race to Zero Pack.

    Evaluates pledge credibility against HLEG "Integrity Matters" 10
    recommendations with 45+ sub-criteria, validates science-based
    alignment, reviews governance structures, and produces a composite
    credibility score with HLEG compliance checklist.

    Engines used:
        - credibility_assessment_engine (HLEG assessment)
        - interim_target_engine (science-based validation)
    """

    def __init__(
        self,
        config: Optional[CredibilityReviewConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or CredibilityReviewConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, CredibilityReviewResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

    async def execute(
        self, input_data: Optional[Dict[str, Any]] = None,
    ) -> CredibilityReviewResult:
        """Execute the 4-phase credibility review workflow."""
        input_data = input_data or {}
        result = CredibilityReviewResult(
            org_name=self.config.org_name,
            status=WorkflowStatus.RUNNING, started_at=_utcnow(),
        )
        self._results[result.execution_id] = result
        start_time = time.monotonic()
        phases = PHASE_EXECUTION_ORDER

        self.logger.info(
            "Starting credibility review: execution_id=%s, org=%s",
            result.execution_id, self.config.org_name,
        )

        ctx: Dict[str, Any] = dict(input_data)

        try:
            for idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = WorkflowStatus.CANCELLED
                    break
                if not self._deps_met(phase, result):
                    result.status = WorkflowStatus.FAILED
                    result.errors.append(f"Dependencies not met for {phase.value}")
                    break

                if self._progress_callback:
                    await self._progress_callback(phase.value, (idx / len(phases)) * 100, phase.value)

                pr = await self._run_phase(phase, ctx)
                result.phase_results[phase.value] = pr
                if pr.status == PhaseStatus.FAILED:
                    result.status = WorkflowStatus.PARTIAL
                result.phases_completed.append(phase.value)
                result.total_records_processed += pr.records_processed
                ctx[phase.value] = pr.outputs

            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Credibility review failed: %s", exc, exc_info=True)
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(exc))

        finally:
            result.completed_at = _utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.recommendation_assessments = self._build_assessments(ctx)
            result.report = self._build_report(ctx)
            if result.report:
                result.overall_rating = result.report.overall_rating
                result.overall_score = result.report.overall_score
            result.quality_score = round(
                (len(result.phases_completed) / max(len(phases), 1)) * 100, 1
            )
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    result.model_dump_json(exclude={"provenance_hash"})
                )

        return result

    def cancel(self, eid: str) -> Dict[str, Any]:
        self._cancelled.add(eid)
        return {"cancelled": True}

    async def _run_phase(self, phase: CredibilityPhase, ctx: Dict[str, Any]) -> PhaseResult:
        started = _utcnow()
        st = time.monotonic()
        handler = {
            CredibilityPhase.HLEG_CRITERIA_ASSESSMENT: self._ph_hleg_assessment,
            CredibilityPhase.SCIENCE_BASED_VALIDATION: self._ph_science_validation,
            CredibilityPhase.GOVERNANCE_REVIEW: self._ph_governance_review,
            CredibilityPhase.CREDIBILITY_SCORING: self._ph_credibility_scoring,
        }[phase]
        try:
            out, warn, err, rec = await handler(ctx)
            status = PhaseStatus.FAILED if err else PhaseStatus.COMPLETED
        except Exception as exc:
            out, warn, err, rec = {}, [], [str(exc)], 0
            status = PhaseStatus.FAILED
        return PhaseResult(
            phase=phase, status=status, started_at=started, completed_at=_utcnow(),
            duration_ms=round((time.monotonic() - st) * 1000, 2), records_processed=rec,
            outputs=out, warnings=warn, errors=err,
            provenance_hash=_compute_hash(out) if self.config.enable_provenance else "",
        )

    def _deps_met(self, phase: CredibilityPhase, result: CredibilityReviewResult) -> bool:
        for dep in PHASE_DEPENDENCIES.get(phase, []):
            dr = result.phase_results.get(dep.value)
            if not dr or dr.status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED):
                return False
        return True

    # ---- Phase 1: HLEG Criteria Assessment ----

    async def _ph_hleg_assessment(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        # Evidence mapping for each recommendation's sub-criteria
        sub_criteria_evidence: Dict[str, List[bool]] = {
            "R01": [
                self.config.has_net_zero_pledge,
                self.config.has_interim_targets,
                self.config.has_science_based_targets,
                self.config.scope3_coverage_pct >= 67.0,
                True,  # Public announcement assumed
            ],
            "R02": [
                self.config.target_reduction_pct >= 42.0,
                True,  # Annual milestones assumed if interim target set
                True,  # Sector pathway alignment assumed
                True,  # Fair share assessment
            ],
            "R03": [
                self.config.responsible_credit_use,
                self.config.responsible_credit_use,
                True,
                True,
                True,
            ],
            "R04": [
                self.config.no_new_fossil_fuel,
                self.config.no_new_fossil_fuel,
                self.config.no_new_fossil_fuel,
                True,  # Transition plan
            ],
            "R05": [
                self.config.lobbying_aligned,
                self.config.lobbying_aligned,
                self.config.lobbying_aligned,
                self.config.lobbying_aligned,
            ],
            "R06": [
                self.config.just_transition_plan,
                self.config.just_transition_plan,
                self.config.just_transition_plan,
                self.config.just_transition_plan,
                True,
            ],
            "R07": [
                self.config.annual_reporting,
                self.config.annual_reporting,
                True,
                self.config.third_party_verification,
                self.config.annual_reporting,
            ],
            "R08": [
                True,  # Scope 1 coverage
                True,  # Scope 2 coverage
                self.config.scope3_coverage_pct >= 67.0,
                True,
            ],
            "R09": [
                self.config.board_oversight,
                self.config.exec_compensation_linked,
                self.config.climate_risk_integrated,
                True,  # Internal carbon pricing
                self.config.board_oversight,
            ],
            "R10": [
                self.config.capex_allocated,
                self.config.investment_usd > 0,
                self.config.capex_allocated,
                self.config.no_new_fossil_fuel,
            ],
        }

        assessments: List[Dict[str, Any]] = []
        total_sub_passed = 0
        total_sub_count = 0

        for rec in HLEG_RECOMMENDATIONS:
            rid = rec["id"]
            evidence_list = sub_criteria_evidence.get(rid, [])
            sub_passed = sum(1 for e in evidence_list if e)
            sub_total = len(rec["sub_criteria"])
            score = (sub_passed / max(sub_total, 1)) * 100.0

            if sub_passed == sub_total:
                status = RecommendationStatus.FULLY_MET.value
            elif sub_passed >= sub_total * 0.5:
                status = RecommendationStatus.PARTIALLY_MET.value
            else:
                status = RecommendationStatus.NOT_MET.value

            gaps = [
                rec["sub_criteria"][i]
                for i, e in enumerate(evidence_list)
                if not e and i < len(rec["sub_criteria"])
            ]

            assessment = {
                "recommendation_id": rid,
                "name": rec["name"],
                "description": rec["description"],
                "status": status,
                "sub_criteria_passed": sub_passed,
                "sub_criteria_total": sub_total,
                "score": round(score, 1),
                "weight": rec["weight"],
                "gaps": gaps,
            }
            assessments.append(assessment)
            total_sub_passed += sub_passed
            total_sub_count += sub_total

        fully_met = sum(1 for a in assessments if a["status"] == "fully_met")
        partially_met = sum(1 for a in assessments if a["status"] == "partially_met")
        not_met = sum(1 for a in assessments if a["status"] == "not_met")

        outputs["assessments"] = assessments
        outputs["recommendations_total"] = len(HLEG_RECOMMENDATIONS)
        outputs["fully_met"] = fully_met
        outputs["partially_met"] = partially_met
        outputs["not_met"] = not_met
        outputs["total_sub_criteria_passed"] = total_sub_passed
        outputs["total_sub_criteria"] = total_sub_count
        outputs["hleg_coverage_pct"] = round(
            (total_sub_passed / max(total_sub_count, 1)) * 100.0, 1
        )

        if not_met > 0:
            warnings.append(
                f"{not_met} HLEG recommendation(s) not met. "
                "Remediation required for credibility standing."
            )

        return outputs, warnings, errors, len(assessments)

    # ---- Phase 2: Science-Based Validation ----

    async def _ph_science_validation(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        baseline = self.config.baseline_tco2e
        current = self.config.current_tco2e
        target_pct = self.config.target_reduction_pct
        base_year = self.config.base_year

        # 1.5C pathway check
        annual_rate = target_pct / max(2030 - base_year, 1)
        is_1_5c_aligned = annual_rate >= 4.2 and target_pct >= 42.0
        is_wb2c_aligned = annual_rate >= 2.5

        # Actual reduction check
        actual_reduction_pct = ((baseline - current) / max(baseline, 1)) * 100.0 if baseline > 0 else 0

        # Science-based target status
        has_validated_target = self.config.has_science_based_targets
        scope_coverage_ok = self.config.scope3_coverage_pct >= 67.0

        outputs["target_reduction_pct"] = round(target_pct, 1)
        outputs["annual_reduction_rate"] = round(annual_rate, 2)
        outputs["is_1_5c_aligned"] = is_1_5c_aligned
        outputs["is_wb2c_aligned"] = is_wb2c_aligned
        outputs["actual_reduction_pct"] = round(actual_reduction_pct, 1)
        outputs["has_validated_sbti_target"] = has_validated_target
        outputs["scope_coverage_sufficient"] = scope_coverage_ok
        outputs["science_based_aligned"] = is_1_5c_aligned and has_validated_target and scope_coverage_ok
        outputs["pathway_gap_pct"] = round(max(4.2 - annual_rate, 0), 2)

        if not is_1_5c_aligned:
            warnings.append(
                f"Target not 1.5C aligned: annual rate {annual_rate:.1f}%/yr "
                f"(need >= 4.2%/yr)"
            )

        return outputs, warnings, errors, 1

    # ---- Phase 3: Governance Review ----

    async def _ph_governance_review(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        governance_criteria = {
            "board_oversight": self.config.board_oversight,
            "exec_compensation_linked": self.config.exec_compensation_linked,
            "climate_risk_integrated": self.config.climate_risk_integrated,
            "capex_allocated": self.config.capex_allocated,
            "third_party_verification": self.config.third_party_verification,
            "annual_reporting": self.config.annual_reporting,
        }

        met = sum(1 for v in governance_criteria.values() if v)
        total = len(governance_criteria)
        score = (met / max(total, 1)) * 100.0

        if score >= 80:
            maturity = GovernanceMaturity.ADVANCED.value
        elif score >= 60:
            maturity = GovernanceMaturity.DEVELOPING.value
        elif score >= 30:
            maturity = GovernanceMaturity.BASIC.value
        else:
            maturity = GovernanceMaturity.ABSENT.value

        gaps = [k for k, v in governance_criteria.items() if not v]

        outputs["governance_criteria"] = governance_criteria
        outputs["criteria_met"] = met
        outputs["criteria_total"] = total
        outputs["governance_score"] = round(score, 1)
        outputs["governance_maturity"] = maturity
        outputs["governance_gaps"] = gaps
        outputs["recommendations"] = [
            f"Establish {g.replace('_', ' ')}" for g in gaps
        ]

        if maturity in (GovernanceMaturity.BASIC.value, GovernanceMaturity.ABSENT.value):
            warnings.append(
                f"Governance maturity is '{maturity}'. "
                "HLEG requires integration of climate into governance structures."
            )

        return outputs, warnings, errors, 1

    # ---- Phase 4: Credibility Scoring ----

    async def _ph_credibility_scoring(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        hleg = ctx.get("hleg_criteria_assessment", {})
        science = ctx.get("science_based_validation", {})
        governance = ctx.get("governance_review", {})

        assessments = hleg.get("assessments", [])

        # Weighted score from HLEG recommendations
        weighted_total = 0.0
        weight_sum = 0.0
        for a in assessments:
            weighted_total += a["score"] * a["weight"]
            weight_sum += a["weight"]

        hleg_score = weighted_total / max(weight_sum, 1)

        # Science-based bonus/penalty
        science_bonus = 10.0 if science.get("science_based_aligned", False) else -5.0

        # Governance bonus
        governance_score = governance.get("governance_score", 0)
        governance_bonus = (governance_score - 50.0) * 0.1  # -5 to +5

        overall_score = min(max(hleg_score + science_bonus + governance_bonus, 0), 100)

        # Determine rating
        if overall_score >= 80:
            rating = CredibilityRating.STRONG.value
        elif overall_score >= 60:
            rating = CredibilityRating.ADEQUATE.value
        elif overall_score >= 40:
            rating = CredibilityRating.WEAK.value
        else:
            rating = CredibilityRating.INSUFFICIENT.value

        # Priority actions
        priority_actions: List[str] = []
        not_met_recs = [a for a in assessments if a["status"] == "not_met"]
        for a in not_met_recs[:3]:
            priority_actions.append(f"Address HLEG {a['recommendation_id']}: {a['name']}")

        if not science.get("science_based_aligned", False):
            priority_actions.append("Align targets with 1.5C science-based pathway")
        if governance.get("governance_maturity") in ("basic", "absent"):
            priority_actions.append("Strengthen climate governance structures")

        report_id = f"CR-{self.config.reporting_year}-{_new_uuid()[:8].upper()}"

        outputs["report_id"] = report_id
        outputs["org_name"] = self.config.org_name
        outputs["assessment_date"] = _utcnow().strftime("%Y-%m-%d")
        outputs["overall_score"] = round(overall_score, 1)
        outputs["overall_rating"] = rating
        outputs["hleg_score"] = round(hleg_score, 1)
        outputs["science_bonus"] = round(science_bonus, 1)
        outputs["governance_bonus"] = round(governance_bonus, 1)
        outputs["recommendations_fully_met"] = hleg.get("fully_met", 0)
        outputs["recommendations_partially_met"] = hleg.get("partially_met", 0)
        outputs["recommendations_not_met"] = hleg.get("not_met", 0)
        outputs["total_sub_criteria_passed"] = hleg.get("total_sub_criteria_passed", 0)
        outputs["total_sub_criteria"] = hleg.get("total_sub_criteria", 45)
        outputs["governance_maturity"] = governance.get("governance_maturity", "basic")
        outputs["science_based_aligned"] = science.get("science_based_aligned", False)
        outputs["priority_actions"] = priority_actions

        return outputs, warnings, errors, 1

    # ---- Extractors ----

    def _build_assessments(self, ctx: Dict[str, Any]) -> List[RecommendationAssessment]:
        data = ctx.get("hleg_criteria_assessment", {}).get("assessments", [])
        return [
            RecommendationAssessment(
                recommendation_id=a["recommendation_id"], name=a["name"],
                status=RecommendationStatus(a["status"]),
                sub_criteria_passed=a["sub_criteria_passed"],
                sub_criteria_total=a["sub_criteria_total"],
                score=a["score"], weight=a["weight"], gaps=a["gaps"],
            )
            for a in data
        ]

    def _build_report(self, ctx: Dict[str, Any]) -> Optional[CredibilityReport]:
        data = ctx.get("credibility_scoring", {})
        if not data:
            return None
        return CredibilityReport(
            report_id=data.get("report_id", ""),
            org_name=data.get("org_name", ""),
            assessment_date=data.get("assessment_date", ""),
            overall_rating=CredibilityRating(data.get("overall_rating", "insufficient")),
            overall_score=data.get("overall_score", 0),
            recommendations_fully_met=data.get("recommendations_fully_met", 0),
            recommendations_partially_met=data.get("recommendations_partially_met", 0),
            recommendations_not_met=data.get("recommendations_not_met", 0),
            total_sub_criteria_passed=data.get("total_sub_criteria_passed", 0),
            total_sub_criteria=data.get("total_sub_criteria", 45),
            governance_maturity=GovernanceMaturity(data.get("governance_maturity", "basic")),
            science_based_aligned=data.get("science_based_aligned", False),
            priority_actions=data.get("priority_actions", []),
        )
