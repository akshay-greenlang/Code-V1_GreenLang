# -*- coding: utf-8 -*-
"""
ActionPlanEngine - PACK-025 Race to Zero Engine 4
===================================================

Generates and validates climate action plans per Race to Zero
publication requirements. Produces quantified action plans with
10 sections (Emissions Profile, Targets, Reduction Actions,
Timeline, Resource Plan, Sector Alignment, Scope 3 Strategy,
Governance, Just Transition, Monitoring) and validates plan
completeness against HLEG and Interpretation Guide criteria.

Calculation Methodology:
    Plan Completeness Score (0-100):
        Each of 10 sections scored 0-10:
            COMPLETE:   9-10 (all required elements present)
            ADEQUATE:   7-8  (most elements present)
            PARTIAL:    5-6  (some elements, gaps exist)
            INCOMPLETE: 3-4  (significant gaps)
            MISSING:    0-2  (section not present)

        Section weights (sum to 1.0):
            emissions_profile:  0.12
            targets:            0.15  (highest: core credibility)
            reduction_actions:  0.15
            timeline:           0.10
            resource_plan:      0.10
            sector_alignment:   0.08
            scope3_strategy:    0.10
            governance:         0.08
            just_transition:    0.06
            monitoring:         0.06

        completeness = sum(section_score * section_weight) * 10

    Action Prioritization (MACC-based):
        priority_score = (abatement_tco2e / cost_total) * trl_factor
        where trl_factor = trl / 9.0  (TRL 1-9 scale)

    Publication Deadline:
        Must be published within 12 months of joining Race to Zero.
        Engine tracks days remaining and readiness percentage.

Regulatory References:
    - Race to Zero Interpretation Guide (June 2022), SL-A1..SL-A5
    - HLEG "Integrity Matters" (November 2022), Rec 3
    - IPCC AR6 WG3 (2022), Sector mitigation options
    - SBTi Corporate Net-Zero Standard v1.3 (2024)
    - TCFD Recommendations (2017), Transition planning

Zero-Hallucination:
    - All 10 plan sections from Race to Zero Interpretation Guide
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  4 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PlanSection(str, Enum):
    """Climate action plan sections per Interpretation Guide."""
    EMISSIONS_PROFILE = "emissions_profile"
    TARGETS = "targets"
    REDUCTION_ACTIONS = "reduction_actions"
    TIMELINE = "timeline"
    RESOURCE_PLAN = "resource_plan"
    SECTOR_ALIGNMENT = "sector_alignment"
    SCOPE3_STRATEGY = "scope3_strategy"
    GOVERNANCE = "governance"
    JUST_TRANSITION = "just_transition"
    MONITORING = "monitoring"

class SectionRating(str, Enum):
    """Section completeness rating."""
    COMPLETE = "complete"
    ADEQUATE = "adequate"
    PARTIAL = "partial"
    INCOMPLETE = "incomplete"
    MISSING = "missing"

class ActionCategory(str, Enum):
    """Decarbonization action category."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    ELECTRIFICATION = "electrification"
    FUEL_SWITCHING = "fuel_switching"
    PROCESS_CHANGE = "process_change"
    SUPPLY_CHAIN = "supply_chain"
    TRANSPORT = "transport"
    BUILDINGS = "buildings"
    WASTE_REDUCTION = "waste_reduction"
    CARBON_CAPTURE = "carbon_capture"
    NATURE_BASED = "nature_based"
    BEHAVIORAL = "behavioral"
    OTHER = "other"

class PlanQuality(str, Enum):
    """Overall plan quality classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    INADEQUATE = "inadequate"
    MISSING = "missing"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTION_IDS = [s.value for s in PlanSection]

SECTION_WEIGHTS: Dict[str, Decimal] = {
    PlanSection.EMISSIONS_PROFILE.value: Decimal("0.12"),
    PlanSection.TARGETS.value: Decimal("0.15"),
    PlanSection.REDUCTION_ACTIONS.value: Decimal("0.15"),
    PlanSection.TIMELINE.value: Decimal("0.10"),
    PlanSection.RESOURCE_PLAN.value: Decimal("0.10"),
    PlanSection.SECTOR_ALIGNMENT.value: Decimal("0.08"),
    PlanSection.SCOPE3_STRATEGY.value: Decimal("0.10"),
    PlanSection.GOVERNANCE.value: Decimal("0.08"),
    PlanSection.JUST_TRANSITION.value: Decimal("0.06"),
    PlanSection.MONITORING.value: Decimal("0.06"),
}

SECTION_LABELS: Dict[str, str] = {
    PlanSection.EMISSIONS_PROFILE.value: "1. Emissions Profile",
    PlanSection.TARGETS.value: "2. Targets",
    PlanSection.REDUCTION_ACTIONS.value: "3. Reduction Actions",
    PlanSection.TIMELINE.value: "4. Timeline & Milestones",
    PlanSection.RESOURCE_PLAN.value: "5. Resource Plan",
    PlanSection.SECTOR_ALIGNMENT.value: "6. Sector Alignment",
    PlanSection.SCOPE3_STRATEGY.value: "7. Scope 3 Strategy",
    PlanSection.GOVERNANCE.value: "8. Governance",
    PlanSection.JUST_TRANSITION.value: "9. Just Transition",
    PlanSection.MONITORING.value: "10. Monitoring & Reporting",
}

RATING_THRESHOLDS: List[Tuple[Decimal, str]] = [
    (Decimal("9"), SectionRating.COMPLETE.value),
    (Decimal("7"), SectionRating.ADEQUATE.value),
    (Decimal("5"), SectionRating.PARTIAL.value),
    (Decimal("3"), SectionRating.INCOMPLETE.value),
    (Decimal("0"), SectionRating.MISSING.value),
]

QUALITY_THRESHOLDS: List[Tuple[Decimal, str]] = [
    (Decimal("85"), PlanQuality.EXCELLENT.value),
    (Decimal("70"), PlanQuality.GOOD.value),
    (Decimal("50"), PlanQuality.ADEQUATE.value),
    (Decimal("25"), PlanQuality.INADEQUATE.value),
    (Decimal("0"), PlanQuality.MISSING.value),
]

MIN_ACTIONS_COUNT: int = 10
PUBLICATION_DEADLINE_MONTHS: int = 12

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class DecarbonizationActionInput(BaseModel):
    """Input for a single decarbonization action.

    Attributes:
        action_id: Unique action identifier.
        action_name: Action name/description.
        category: Action category.
        scope_impact: Scopes impacted (1/2/3).
        abatement_tco2e: Estimated abatement (tCO2e/year).
        cost_total_usd: Total implementation cost (USD).
        cost_per_tco2e_usd: Cost-effectiveness (USD/tCO2e).
        start_year: Implementation start year.
        end_year: Full implementation year.
        trl: Technology readiness level (1-9).
        responsible_party: Responsible party.
        status: Action status (planned/in_progress/completed).
        milestones: Key milestones.
        notes: Additional notes.
    """
    action_id: str = Field(default_factory=_new_uuid)
    action_name: str = Field(default="", max_length=300)
    category: str = Field(default=ActionCategory.OTHER.value)
    scope_impact: List[int] = Field(default_factory=lambda: [1])
    abatement_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    cost_total_usd: Decimal = Field(default=Decimal("0"), ge=0)
    cost_per_tco2e_usd: Decimal = Field(default=Decimal("0"), ge=0)
    start_year: int = Field(default=2025, ge=2020, le=2060)
    end_year: int = Field(default=2030, ge=2020, le=2060)
    trl: int = Field(default=9, ge=1, le=9)
    responsible_party: str = Field(default="")
    status: str = Field(default="planned")
    milestones: List[str] = Field(default_factory=list)
    notes: str = Field(default="")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid = {c.value for c in ActionCategory}
        if v not in valid:
            raise ValueError(f"Unknown action category '{v}'.")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in ("planned", "in_progress", "completed"):
            raise ValueError(f"Unknown action status '{v}'.")
        return v

class SectionInput(BaseModel):
    """Input for a single plan section assessment.

    Attributes:
        section_id: Section identifier.
        score: Section score (0-10).
        content_summary: Summary of section content.
        evidence_documents: Supporting documents.
        assessor_notes: Assessor comments.
    """
    section_id: str = Field(..., description="Section identifier")
    score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("10"))
    content_summary: str = Field(default="")
    evidence_documents: List[str] = Field(default_factory=list)
    assessor_notes: str = Field(default="")

    @field_validator("section_id")
    @classmethod
    def validate_section(cls, v: str) -> str:
        if v not in SECTION_IDS:
            raise ValueError(f"Unknown section '{v}'. Must be one of: {SECTION_IDS}")
        return v

class ActionPlanInput(BaseModel):
    """Complete input for action plan assessment.

    Attributes:
        entity_name: Entity name.
        actor_type: Actor type.
        sector: Industry sector.
        baseline_year: Baseline year.
        baseline_emissions_tco2e: Baseline total emissions.
        target_year: Interim target year.
        target_emissions_tco2e: Target emissions.
        net_zero_year: Net-zero target year.
        join_date: Race to Zero join date (YYYY-MM-DD).
        plan_published: Whether plan has been published.
        plan_publication_date: Publication date (YYYY-MM-DD).
        actions: Decarbonization action list.
        sections: Per-section assessment data.
        scope1_emissions_tco2e: Current Scope 1 emissions.
        scope2_emissions_tco2e: Current Scope 2 emissions.
        scope3_emissions_tco2e: Current Scope 3 emissions.
        total_budget_usd: Total decarbonization budget.
        fte_allocated: FTE allocated to climate actions.
        has_governance_structure: Whether governance is established.
        has_board_oversight: Whether board provides oversight.
        has_just_transition_plan: Whether just transition is addressed.
        has_monitoring_kpis: Whether monitoring KPIs are defined.
        include_prioritization: Whether to prioritize actions.
    """
    entity_name: str = Field(..., min_length=1, max_length=300)
    actor_type: str = Field(default="corporate")
    sector: str = Field(default="general", max_length=100)
    baseline_year: int = Field(default=2019, ge=2010, le=2060)
    baseline_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    target_year: int = Field(default=2030, ge=2025, le=2040)
    target_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    net_zero_year: int = Field(default=2050, ge=2030, le=2060)
    join_date: Optional[str] = Field(default=None)
    plan_published: bool = Field(default=False)
    plan_publication_date: Optional[str] = Field(default=None)
    actions: List[DecarbonizationActionInput] = Field(default_factory=list)
    sections: List[SectionInput] = Field(default_factory=list)
    scope1_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    total_budget_usd: Decimal = Field(default=Decimal("0"), ge=0)
    fte_allocated: Decimal = Field(default=Decimal("0"), ge=0)
    has_governance_structure: bool = Field(default=False)
    has_board_oversight: bool = Field(default=False)
    has_just_transition_plan: bool = Field(default=False)
    has_monitoring_kpis: bool = Field(default=False)
    include_prioritization: bool = Field(default=True)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ActionResult(BaseModel):
    """Assessment result for a single action.

    Attributes:
        action_id: Action identifier.
        action_name: Action name.
        category: Action category.
        abatement_tco2e: Estimated abatement.
        cost_total_usd: Total cost.
        cost_effectiveness_usd: Cost per tCO2e.
        priority_score: MACC-based priority score.
        priority_rank: Priority rank (1 = highest).
        trl: Technology readiness level.
        scope_impact: Scopes impacted.
        status: Implementation status.
        pct_of_total_abatement: % of total plan abatement.
    """
    action_id: str = Field(default="")
    action_name: str = Field(default="")
    category: str = Field(default="")
    abatement_tco2e: Decimal = Field(default=Decimal("0"))
    cost_total_usd: Decimal = Field(default=Decimal("0"))
    cost_effectiveness_usd: Decimal = Field(default=Decimal("0"))
    priority_score: Decimal = Field(default=Decimal("0"))
    priority_rank: int = Field(default=0)
    trl: int = Field(default=9)
    scope_impact: List[int] = Field(default_factory=list)
    status: str = Field(default="planned")
    pct_of_total_abatement: Decimal = Field(default=Decimal("0"))

class SectionResult(BaseModel):
    """Assessment result for a single plan section.

    Attributes:
        section_id: Section identifier.
        section_name: Display name.
        score: Section score (0-10).
        weight: Section weight.
        weighted_score: score * weight.
        rating: Section rating.
        gap_description: Description of gaps.
        recommendations: Section recommendations.
    """
    section_id: str = Field(default="")
    section_name: str = Field(default="")
    score: Decimal = Field(default=Decimal("0"))
    weight: Decimal = Field(default=Decimal("0"))
    weighted_score: Decimal = Field(default=Decimal("0"))
    rating: str = Field(default=SectionRating.MISSING.value)
    gap_description: str = Field(default="")
    recommendations: List[str] = Field(default_factory=list)

class AbatementSummary(BaseModel):
    """Summary of abatement from all actions.

    Attributes:
        total_abatement_tco2e: Total planned abatement.
        total_cost_usd: Total planned cost.
        avg_cost_per_tco2e: Average cost-effectiveness.
        abatement_by_scope: Abatement breakdown by scope.
        abatement_by_category: Abatement by action category.
        target_gap_tco2e: Gap between abatement and target reduction.
        target_coverage_pct: % of required reduction covered by actions.
        actions_by_status: Count by status (planned/in_progress/completed).
    """
    total_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    total_cost_usd: Decimal = Field(default=Decimal("0"))
    avg_cost_per_tco2e: Decimal = Field(default=Decimal("0"))
    abatement_by_scope: Dict[str, Decimal] = Field(default_factory=dict)
    abatement_by_category: Dict[str, Decimal] = Field(default_factory=dict)
    target_gap_tco2e: Decimal = Field(default=Decimal("0"))
    target_coverage_pct: Decimal = Field(default=Decimal("0"))
    actions_by_status: Dict[str, int] = Field(default_factory=dict)

class ActionPlanResult(BaseModel):
    """Complete action plan assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        completeness_score: Plan completeness score (0-100).
        plan_quality: Overall quality classification.
        section_results: Per-section assessment results.
        action_results: Per-action assessment results.
        abatement_summary: Abatement summary.
        total_actions: Total number of actions.
        meets_min_actions: Whether minimum action count is met.
        plan_published: Whether plan is published.
        days_until_deadline: Days until publication deadline.
        publication_on_time: Whether published within deadline.
        hleg_rec3_aligned: Whether aligned with HLEG Recommendation 3.
        gaps: Identified gaps.
        recommendations: Improvement recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    completeness_score: Decimal = Field(default=Decimal("0"))
    plan_quality: str = Field(default=PlanQuality.MISSING.value)
    section_results: List[SectionResult] = Field(default_factory=list)
    action_results: List[ActionResult] = Field(default_factory=list)
    abatement_summary: Optional[AbatementSummary] = Field(default=None)
    total_actions: int = Field(default=0)
    meets_min_actions: bool = Field(default=False)
    plan_published: bool = Field(default=False)
    days_until_deadline: Optional[int] = Field(default=None)
    publication_on_time: Optional[bool] = Field(default=None)
    hleg_rec3_aligned: bool = Field(default=False)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ActionPlanEngine:
    """Race to Zero climate action plan generation and validation engine.

    Validates 10-section action plans, prioritizes decarbonization
    actions using MACC-based scoring, assesses plan completeness,
    and checks compliance with HLEG Recommendation 3.

    Usage::

        engine = ActionPlanEngine()
        result = engine.assess(input_data)
        print(f"Completeness: {result.completeness_score}/100")
        print(f"Quality: {result.plan_quality}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise ActionPlanEngine.

        Args:
            config: Optional overrides.
        """
        self.config = config or {}
        self._weights = dict(SECTION_WEIGHTS)
        self._min_actions = int(self.config.get("min_actions", MIN_ACTIONS_COUNT))
        logger.info("ActionPlanEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def assess(
        self, data: ActionPlanInput,
    ) -> ActionPlanResult:
        """Perform complete action plan assessment.

        Args:
            data: Validated action plan input.

        Returns:
            ActionPlanResult with full assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Action plan assessment: entity=%s, actions=%d",
            data.entity_name, len(data.actions),
        )

        warnings: List[str] = []
        errors: List[str] = []
        gaps: List[str] = []

        # Step 1: Assess sections
        section_map: Dict[str, SectionInput] = {s.section_id: s for s in data.sections}
        section_results = self._assess_sections(data, section_map, gaps)

        # Step 2: Calculate completeness
        completeness = Decimal("0")
        for sr in section_results:
            completeness += sr.weighted_score
        completeness = _round_val(completeness * Decimal("10"), 2)

        # Step 3: Plan quality
        plan_quality = self._determine_quality(completeness)

        # Step 4: Assess and prioritize actions
        action_results = self._assess_actions(data.actions, data)

        # Step 5: Abatement summary
        abatement_summary = self._build_abatement_summary(data.actions, data)

        # Step 6: Minimum actions check
        meets_min = len(data.actions) >= self._min_actions
        if not meets_min:
            gaps.append(
                f"Only {len(data.actions)} actions specified. "
                f"Minimum {self._min_actions} required."
            )

        # Step 7: Publication deadline
        days_until: Optional[int] = None
        on_time: Optional[bool] = None
        if data.join_date:
            try:
                join_dt = datetime.strptime(data.join_date, "%Y-%m-%d")
                from datetime import timedelta

                deadline = join_dt + timedelta(days=PUBLICATION_DEADLINE_MONTHS * 30)
                now = datetime.now()
                days_until = (deadline - now).days
                if data.plan_published and data.plan_publication_date:
                    pub_dt = datetime.strptime(data.plan_publication_date, "%Y-%m-%d")
                    on_time = pub_dt <= deadline
                elif data.plan_published:
                    on_time = True
            except ValueError:
                warnings.append("Invalid date format in join_date or publication_date.")

        # Step 8: HLEG Rec 3 alignment
        hleg_aligned = (
            completeness >= Decimal("70")
            and meets_min
            and data.has_governance_structure
            and data.has_monitoring_kpis
        )

        # Step 9: Recommendations
        recommendations = self._generate_recommendations(
            section_results, action_results, completeness,
            meets_min, data
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ActionPlanResult(
            entity_name=data.entity_name,
            completeness_score=completeness,
            plan_quality=plan_quality,
            section_results=section_results,
            action_results=action_results,
            abatement_summary=abatement_summary,
            total_actions=len(data.actions),
            meets_min_actions=meets_min,
            plan_published=data.plan_published,
            days_until_deadline=days_until,
            publication_on_time=on_time,
            hleg_rec3_aligned=hleg_aligned,
            gaps=gaps,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Action plan assessment complete: completeness=%.1f, quality=%s, "
            "actions=%d, hleg=%s, hash=%s",
            float(completeness), plan_quality, len(data.actions),
            hleg_aligned, result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _assess_sections(
        self,
        data: ActionPlanInput,
        section_map: Dict[str, SectionInput],
        gaps: List[str],
    ) -> List[SectionResult]:
        """Assess all 10 plan sections.

        Args:
            data: Input data.
            section_map: Section input data.
            gaps: Gap list.

        Returns:
            List of SectionResult.
        """
        results: List[SectionResult] = []

        for sec in PlanSection:
            sec_id = sec.value
            weight = self._weights.get(sec_id, Decimal("0.05"))
            label = SECTION_LABELS.get(sec_id, sec_id)

            if sec_id in section_map:
                score = section_map[sec_id].score
            else:
                score = self._auto_score_section(sec_id, data)

            weighted = score * weight
            rating = self._section_rating(score)

            gap_desc = ""
            recs: List[str] = []

            if score < Decimal("7"):
                gap_desc, rec = self._get_section_gap(sec_id, score)
                gaps.append(f"{label}: {gap_desc}")
                recs.append(rec)

            results.append(SectionResult(
                section_id=sec_id,
                section_name=label,
                score=score,
                weight=weight,
                weighted_score=_round_val(weighted, 4),
                rating=rating,
                gap_description=gap_desc,
                recommendations=recs,
            ))

        return results

    def _auto_score_section(
        self, sec_id: str, data: ActionPlanInput,
    ) -> Decimal:
        """Auto-score a section from input data fields.

        Args:
            sec_id: Section identifier.
            data: Input data.

        Returns:
            Score (0-10).
        """
        if sec_id == PlanSection.EMISSIONS_PROFILE.value:
            score = Decimal("0")
            if data.scope1_emissions_tco2e > 0:
                score += Decimal("3")
            if data.scope2_emissions_tco2e > 0:
                score += Decimal("3")
            if data.scope3_emissions_tco2e > 0:
                score += Decimal("4")
            return min(score, Decimal("10"))

        if sec_id == PlanSection.TARGETS.value:
            score = Decimal("0")
            if data.target_emissions_tco2e > 0:
                score += Decimal("5")
            if data.net_zero_year > 0:
                score += Decimal("3")
            if data.baseline_emissions_tco2e > 0:
                score += Decimal("2")
            return min(score, Decimal("10"))

        if sec_id == PlanSection.REDUCTION_ACTIONS.value:
            n = len(data.actions)
            if n >= 15:
                return Decimal("10")
            elif n >= 10:
                return Decimal("8")
            elif n >= 5:
                return Decimal("6")
            elif n >= 1:
                return Decimal("3")
            return Decimal("0")

        if sec_id == PlanSection.TIMELINE.value:
            has_timeline = any(a.milestones for a in data.actions)
            if has_timeline and len(data.actions) >= 5:
                return Decimal("8")
            elif has_timeline:
                return Decimal("5")
            return Decimal("2")

        if sec_id == PlanSection.RESOURCE_PLAN.value:
            score = Decimal("0")
            if data.total_budget_usd > 0:
                score += Decimal("5")
            if data.fte_allocated > 0:
                score += Decimal("3")
            costed = sum(1 for a in data.actions if a.cost_total_usd > 0)
            if costed >= 5:
                score += Decimal("2")
            return min(score, Decimal("10"))

        if sec_id == PlanSection.SECTOR_ALIGNMENT.value:
            return Decimal("5") if data.sector != "general" else Decimal("2")

        if sec_id == PlanSection.SCOPE3_STRATEGY.value:
            s3_actions = sum(1 for a in data.actions if 3 in a.scope_impact)
            if s3_actions >= 5:
                return Decimal("8")
            elif s3_actions >= 2:
                return Decimal("5")
            elif s3_actions >= 1:
                return Decimal("3")
            return Decimal("0")

        if sec_id == PlanSection.GOVERNANCE.value:
            score = Decimal("0")
            if data.has_governance_structure:
                score += Decimal("5")
            if data.has_board_oversight:
                score += Decimal("5")
            return min(score, Decimal("10"))

        if sec_id == PlanSection.JUST_TRANSITION.value:
            return Decimal("8") if data.has_just_transition_plan else Decimal("2")

        if sec_id == PlanSection.MONITORING.value:
            return Decimal("8") if data.has_monitoring_kpis else Decimal("2")

        return Decimal("5")

    def _assess_actions(
        self,
        actions: List[DecarbonizationActionInput],
        data: ActionPlanInput,
    ) -> List[ActionResult]:
        """Assess and prioritize actions using MACC-based scoring.

        Args:
            actions: List of actions.
            data: Input data.

        Returns:
            List of ActionResult sorted by priority.
        """
        total_abatement = sum(
            (a.abatement_tco2e for a in actions), Decimal("0")
        )

        scored: List[ActionResult] = []
        for a in actions:
            cost_eff = a.cost_per_tco2e_usd
            if cost_eff <= Decimal("0") and a.cost_total_usd > Decimal("0") and a.abatement_tco2e > Decimal("0"):
                cost_eff = _round_val(a.cost_total_usd / a.abatement_tco2e, 2)

            trl_factor = _decimal(a.trl) / Decimal("9")
            if cost_eff > Decimal("0"):
                priority_score = _round_val(
                    (a.abatement_tco2e / cost_eff) * trl_factor, 4
                )
            else:
                priority_score = a.abatement_tco2e * trl_factor

            pct = _round_val(_safe_pct(a.abatement_tco2e, total_abatement), 2)

            scored.append(ActionResult(
                action_id=a.action_id,
                action_name=a.action_name,
                category=a.category,
                abatement_tco2e=_round_val(a.abatement_tco2e),
                cost_total_usd=_round_val(a.cost_total_usd),
                cost_effectiveness_usd=_round_val(cost_eff, 2),
                priority_score=priority_score,
                priority_rank=0,
                trl=a.trl,
                scope_impact=a.scope_impact,
                status=a.status,
                pct_of_total_abatement=pct,
            ))

        scored.sort(key=lambda x: x.priority_score, reverse=True)
        for i, ar in enumerate(scored, 1):
            ar.priority_rank = i

        return scored

    def _build_abatement_summary(
        self,
        actions: List[DecarbonizationActionInput],
        data: ActionPlanInput,
    ) -> AbatementSummary:
        """Build abatement summary across all actions.

        Args:
            actions: List of actions.
            data: Input data.

        Returns:
            AbatementSummary.
        """
        total_abatement = sum((a.abatement_tco2e for a in actions), Decimal("0"))
        total_cost = sum((a.cost_total_usd for a in actions), Decimal("0"))
        avg_cost = _safe_divide(total_cost, total_abatement)

        by_scope: Dict[str, Decimal] = {}
        by_category: Dict[str, Decimal] = {}
        by_status: Dict[str, int] = {"planned": 0, "in_progress": 0, "completed": 0}

        for a in actions:
            for s in a.scope_impact:
                key = f"scope_{s}"
                by_scope[key] = by_scope.get(key, Decimal("0")) + a.abatement_tco2e
            by_category[a.category] = by_category.get(a.category, Decimal("0")) + a.abatement_tco2e
            by_status[a.status] = by_status.get(a.status, 0) + 1

        required_reduction = data.baseline_emissions_tco2e - data.target_emissions_tco2e
        target_gap = max(Decimal("0"), required_reduction - total_abatement)
        coverage = _safe_pct(total_abatement, required_reduction) if required_reduction > Decimal("0") else Decimal("0")

        return AbatementSummary(
            total_abatement_tco2e=_round_val(total_abatement),
            total_cost_usd=_round_val(total_cost),
            avg_cost_per_tco2e=_round_val(avg_cost, 2),
            abatement_by_scope={k: _round_val(v) for k, v in by_scope.items()},
            abatement_by_category={k: _round_val(v) for k, v in by_category.items()},
            target_gap_tco2e=_round_val(target_gap),
            target_coverage_pct=_round_val(min(coverage, Decimal("100")), 2),
            actions_by_status=by_status,
        )

    def _section_rating(self, score: Decimal) -> str:
        """Determine section rating from score."""
        for threshold, rating in RATING_THRESHOLDS:
            if score >= threshold:
                return rating
        return SectionRating.MISSING.value

    def _determine_quality(self, completeness: Decimal) -> str:
        """Determine plan quality from completeness score."""
        for threshold, quality in QUALITY_THRESHOLDS:
            if completeness >= threshold:
                return quality
        return PlanQuality.MISSING.value

    def _get_section_gap(
        self, sec_id: str, score: Decimal,
    ) -> Tuple[str, str]:
        """Get gap description and recommendation for a section.

        Args:
            sec_id: Section identifier.
            score: Current section score.

        Returns:
            Tuple of (gap_description, recommendation).
        """
        gap_map: Dict[str, Tuple[str, str]] = {
            PlanSection.EMISSIONS_PROFILE.value: (
                "Emissions profile incomplete or missing scope coverage.",
                "Complete Scope 1, 2, and material Scope 3 emissions inventory."
            ),
            PlanSection.TARGETS.value: (
                "Targets section missing interim or long-term targets.",
                "Document both 2030 interim and 2050 net-zero targets with methodology."
            ),
            PlanSection.REDUCTION_ACTIONS.value: (
                "Insufficient quantified reduction actions.",
                f"Add at least {self._min_actions} specific actions with tCO2e impact."
            ),
            PlanSection.TIMELINE.value: (
                "Timeline and milestones not defined for actions.",
                "Create implementation timeline with dated milestones for each action."
            ),
            PlanSection.RESOURCE_PLAN.value: (
                "Resource allocation (budget, FTE) not specified.",
                "Allocate budget and FTE resources to each action item."
            ),
            PlanSection.SECTOR_ALIGNMENT.value: (
                "Sector pathway alignment not assessed.",
                "Map actions to relevant sector pathway benchmarks (IEA/MPP/CRREM)."
            ),
            PlanSection.SCOPE3_STRATEGY.value: (
                "Scope 3 value chain strategy insufficient.",
                "Develop specific Scope 3 reduction strategies with supplier engagement."
            ),
            PlanSection.GOVERNANCE.value: (
                "Climate governance structure not established.",
                "Establish board oversight and executive responsibility for climate."
            ),
            PlanSection.JUST_TRANSITION.value: (
                "Just transition considerations not addressed.",
                "Integrate just transition principles, workforce planning, and community engagement."
            ),
            PlanSection.MONITORING.value: (
                "Monitoring KPIs and reporting plan not defined.",
                "Define KPIs, reporting frequency, and corrective action triggers."
            ),
        }
        return gap_map.get(sec_id, ("Section incomplete.", "Review and complete section."))

    def _generate_recommendations(
        self,
        sections: List[SectionResult],
        actions: List[ActionResult],
        completeness: Decimal,
        meets_min: bool,
        data: ActionPlanInput,
    ) -> List[str]:
        """Generate improvement recommendations.

        Args:
            sections: Section results.
            actions: Action results.
            completeness: Completeness score.
            meets_min: Whether min actions met.
            data: Input data.

        Returns:
            List of recommendations.
        """
        recs: List[str] = []

        if completeness < Decimal("50"):
            recs.append(
                "CRITICAL: Action plan completeness below 50%. "
                "Multiple sections require significant development."
            )

        if not meets_min:
            recs.append(
                f"Add at least {self._min_actions - len(data.actions)} more "
                f"quantified decarbonization actions."
            )

        if not data.plan_published:
            recs.append(
                "Publish the action plan to meet Race to Zero 12-month requirement."
            )

        if not data.has_governance_structure:
            recs.append(
                "Establish climate governance structure with board oversight."
            )

        if not data.has_just_transition_plan:
            recs.append(
                "Integrate just transition considerations per HLEG Recommendation 7."
            )

        # Recommend improving lowest-scored sections
        for sr in sorted(sections, key=lambda s: s.score):
            if sr.score < Decimal("5"):
                for rec in sr.recommendations:
                    recs.append(f"[{sr.section_name}] {rec}")

        return recs
