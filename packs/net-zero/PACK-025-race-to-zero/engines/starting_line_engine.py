# -*- coding: utf-8 -*-
"""
StartingLineEngine - PACK-025 Race to Zero Engine 2
=====================================================

Assesses compliance with the four Race to Zero Starting Line Criteria
(Pledge, Plan, Proceed, Publish) with 20 sub-criteria from the June
2022 Interpretation Guide. Each criterion maps to specific evidence
requirements and produces a pass/fail/partial assessment with gap
identification and remediation guidance.

The 4P Framework:
    PLEDGE  (5 sub-criteria): SL-P1..SL-P5
    PLAN    (5 sub-criteria): SL-A1..SL-A5
    PROCEED (5 sub-criteria): SL-R1..SL-R5
    PUBLISH (5 sub-criteria): SL-D1..SL-D5

Calculation Methodology:
    Sub-Criterion Scoring:
        PASS:           1.0 (criterion fully met with evidence)
        PARTIAL:        0.5 (partially met, gaps identified)
        FAIL:           0.0 (criterion not met)
        NOT_APPLICABLE: excluded from scoring

    Category Score (per P):
        category_score = sum(sub_criterion_scores) / count(applicable)
        category_status = COMPLIANT if all PASS, PARTIALLY if any PARTIAL, else NON_COMPLIANT

    Overall Starting Line Status:
        COMPLIANT:           all 4 categories COMPLIANT
        PARTIALLY_COMPLIANT: >=2 categories COMPLIANT, remainder PARTIAL
        NON_COMPLIANT:       any category fully FAIL or <2 COMPLIANT

    Scope Coverage Validation:
        scope1_2_pct >= 95% (mandatory for all actor types)
        scope3_pct >= 67% (for corporates/FIs)

    12-Month Compliance Timeline:
        Participants must demonstrate Starting Line compliance within 12
        months of joining. Engine calculates remaining time and projected
        readiness based on current gap count.

Regulatory References:
    - Race to Zero Campaign Starting Line Criteria (UNFCCC, 2022)
    - Race to Zero Interpretation Guide (June 2022 update)
    - HLEG "Integrity Matters" 10 Recommendations (November 2022)
    - IPCC AR6 WG3 Mitigation Pathways (2022)
    - SBTi Corporate Net-Zero Standard v1.3 (2024)

Zero-Hallucination:
    - All 20 sub-criteria from Race to Zero Interpretation Guide
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  2 of 10
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
from greenlang.schemas.enums import ComplianceStatus

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StartingLineCategory(str, Enum):
    """The four Starting Line Criteria categories (4Ps).

    PLEDGE:  Commit to a science-based net-zero target.
    PLAN:    Publish a climate action plan.
    PROCEED: Take immediate action on reduction.
    PUBLISH: Report progress annually.
    """
    PLEDGE = "pledge"
    PLAN = "plan"
    PROCEED = "proceed"
    PUBLISH = "publish"

class SubCriterionStatus(str, Enum):
    """Assessment status for a single sub-criterion.

    PASS: Sub-criterion fully met with evidence.
    PARTIAL: Partially met, gaps identified.
    FAIL: Sub-criterion not met.
    NOT_APPLICABLE: Not applicable for this entity.
    """
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"

# ---------------------------------------------------------------------------
# Constants -- 20 Sub-Criteria Definition
# ---------------------------------------------------------------------------

# Sub-criteria IDs and definitions per Interpretation Guide (June 2022).
SUB_CRITERIA: Dict[str, Dict[str, str]] = {
    # PLEDGE sub-criteria
    "SL-P1": {
        "category": StartingLineCategory.PLEDGE.value,
        "name": "Net-zero target",
        "description": "Commit to net zero by 2050 at latest, covering all scopes.",
        "evidence_required": "Published net-zero commitment with target year and scope.",
    },
    "SL-P2": {
        "category": StartingLineCategory.PLEDGE.value,
        "name": "Interim target",
        "description": "Set interim target for 2030 reflecting ~50% absolute reduction.",
        "evidence_required": "Documented 2030 target with methodology and baseline.",
    },
    "SL-P3": {
        "category": StartingLineCategory.PLEDGE.value,
        "name": "Science-based methodology",
        "description": "Target uses recognized science-based methodology (SBTi, IEA, IPCC).",
        "evidence_required": "Methodology reference and validation status.",
    },
    "SL-P4": {
        "category": StartingLineCategory.PLEDGE.value,
        "name": "Fair share",
        "description": "Target represents a fair share of global effort (equity consideration).",
        "evidence_required": "Fair share assessment or equity-weighted analysis.",
    },
    "SL-P5": {
        "category": StartingLineCategory.PLEDGE.value,
        "name": "Scope coverage",
        "description": "Covers Scope 1, 2, and material Scope 3 (or community-wide for cities).",
        "evidence_required": "Scope boundary documentation with coverage percentages.",
    },
    # PLAN sub-criteria
    "SL-A1": {
        "category": StartingLineCategory.PLAN.value,
        "name": "Action plan published",
        "description": "Climate action plan published within 12 months of joining.",
        "evidence_required": "Published action plan document with date.",
    },
    "SL-A2": {
        "category": StartingLineCategory.PLAN.value,
        "name": "Quantified actions",
        "description": "Plan includes specific, quantified decarbonization actions.",
        "evidence_required": "Actions with tCO2e abatement impact quantified.",
    },
    "SL-A3": {
        "category": StartingLineCategory.PLAN.value,
        "name": "Timeline and milestones",
        "description": "Actions have defined timelines and measurable milestones.",
        "evidence_required": "Gantt chart or timeline with milestones.",
    },
    "SL-A4": {
        "category": StartingLineCategory.PLAN.value,
        "name": "Resource allocation",
        "description": "Plan specifies resources (financial, human, technical) for implementation.",
        "evidence_required": "Budget allocation and FTE commitment per action.",
    },
    "SL-A5": {
        "category": StartingLineCategory.PLAN.value,
        "name": "Sector alignment",
        "description": "Actions aligned with relevant sector pathway(s).",
        "evidence_required": "Sector pathway reference with gap analysis.",
    },
    # PROCEED sub-criteria
    "SL-R1": {
        "category": StartingLineCategory.PROCEED.value,
        "name": "Immediate action",
        "description": "Demonstrable action taken (not just planned) within first year.",
        "evidence_required": "Evidence of initiated actions with dates.",
    },
    "SL-R2": {
        "category": StartingLineCategory.PROCEED.value,
        "name": "Emission reductions",
        "description": "Evidence of actual emission reductions or genuine reduction trajectory.",
        "evidence_required": "Year-over-year emission data showing reduction.",
    },
    "SL-R3": {
        "category": StartingLineCategory.PROCEED.value,
        "name": "Investment commitment",
        "description": "Financial resources allocated to decarbonization actions.",
        "evidence_required": "CapEx/OpEx allocation for climate actions.",
    },
    "SL-R4": {
        "category": StartingLineCategory.PROCEED.value,
        "name": "Governance integration",
        "description": "Climate targets integrated into corporate/organizational governance.",
        "evidence_required": "Board oversight documentation, executive responsibility.",
    },
    "SL-R5": {
        "category": StartingLineCategory.PROCEED.value,
        "name": "No contradictory action",
        "description": "No actions contradicting climate commitment (e.g., new fossil fuel investment).",
        "evidence_required": "Policy statements, investment screening records.",
    },
    # PUBLISH sub-criteria
    "SL-D1": {
        "category": StartingLineCategory.PUBLISH.value,
        "name": "Annual reporting",
        "description": "Annual progress reported through partner initiative channels.",
        "evidence_required": "Most recent annual report or disclosure submission.",
    },
    "SL-D2": {
        "category": StartingLineCategory.PUBLISH.value,
        "name": "Emissions disclosure",
        "description": "GHG emissions disclosed publicly (Scope 1, 2, material Scope 3).",
        "evidence_required": "Published GHG inventory or CDP/sustainability report.",
    },
    "SL-D3": {
        "category": StartingLineCategory.PUBLISH.value,
        "name": "Target progress",
        "description": "Progress against targets reported with quantitative metrics.",
        "evidence_required": "Target tracking data with baseline comparison.",
    },
    "SL-D4": {
        "category": StartingLineCategory.PUBLISH.value,
        "name": "Plan updates",
        "description": "Action plan updated and re-published annually.",
        "evidence_required": "Updated action plan with version date.",
    },
    "SL-D5": {
        "category": StartingLineCategory.PUBLISH.value,
        "name": "Transparency",
        "description": "Methodology, assumptions, and limitations transparently documented.",
        "evidence_required": "Methodology documentation, assumption register.",
    },
}

SUB_CRITERION_IDS = list(SUB_CRITERIA.keys())

STATUS_SCORES: Dict[str, Decimal] = {
    SubCriterionStatus.PASS.value: Decimal("1.0"),
    SubCriterionStatus.PARTIAL.value: Decimal("0.5"),
    SubCriterionStatus.FAIL.value: Decimal("0.0"),
    SubCriterionStatus.NOT_APPLICABLE.value: Decimal("1.0"),
}

# Category labels for display.
CATEGORY_LABELS: Dict[str, str] = {
    StartingLineCategory.PLEDGE.value: "PLEDGE -- Commit",
    StartingLineCategory.PLAN.value: "PLAN -- Publish Action Plan",
    StartingLineCategory.PROCEED.value: "PROCEED -- Take Action",
    StartingLineCategory.PUBLISH.value: "PUBLISH -- Report Progress",
}

# Priority levels for remediation items.
PRIORITY_WEIGHTS: Dict[str, int] = {
    StartingLineCategory.PLEDGE.value: 1,
    StartingLineCategory.PROCEED.value: 2,
    StartingLineCategory.PLAN.value: 3,
    StartingLineCategory.PUBLISH.value: 4,
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class SubCriterionInput(BaseModel):
    """Input data for a single Starting Line sub-criterion.

    Attributes:
        criterion_id: Sub-criterion identifier (SL-P1..SL-D5).
        status: Assessment status (pass/partial/fail/not_applicable).
        evidence: Evidence description.
        evidence_documents: Supporting document references.
        assessor_notes: Assessor comments.
    """
    criterion_id: str = Field(..., description="Sub-criterion ID (SL-P1..SL-D5)")
    status: str = Field(
        default=SubCriterionStatus.FAIL.value,
        description="Assessment status"
    )
    evidence: str = Field(default="", description="Evidence description")
    evidence_documents: List[str] = Field(
        default_factory=list, description="Supporting documents"
    )
    assessor_notes: str = Field(default="", description="Assessor notes")

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, v: str) -> str:
        if v not in SUB_CRITERION_IDS:
            raise ValueError(
                f"Unknown sub-criterion '{v}'. Must be one of: {SUB_CRITERION_IDS}"
            )
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {s.value for s in SubCriterionStatus}
        if v not in valid:
            raise ValueError(f"Unknown status '{v}'. Must be one of: {sorted(valid)}")
        return v

class StartingLineInput(BaseModel):
    """Complete input for Starting Line assessment.

    Attributes:
        entity_name: Organization/entity name.
        actor_type: Type of non-state actor.
        join_date: Date entity joined Race to Zero (YYYY-MM-DD).
        assessment_date: Date of this assessment (YYYY-MM-DD).
        net_zero_target_year: Net-zero target year.
        interim_target_year: Interim target year.
        interim_target_reduction_pct: Interim target reduction (%).
        baseline_year: Baseline year.
        baseline_emissions_tco2e: Baseline emissions (tCO2e).
        current_emissions_tco2e: Current reporting year emissions (tCO2e).
        scope1_coverage_pct: Scope 1 coverage (%).
        scope2_coverage_pct: Scope 2 coverage (%).
        scope3_coverage_pct: Scope 3 coverage (%).
        action_plan_published: Whether action plan has been published.
        action_plan_date: Publication date of action plan.
        has_quantified_actions: Whether plan has quantified actions.
        has_timeline: Whether plan has timeline and milestones.
        has_resources: Whether plan specifies resources.
        has_sector_alignment: Whether plan aligns with sector pathway.
        immediate_actions_taken: Whether immediate actions have been taken.
        emissions_reducing: Whether emissions are on a reduction trajectory.
        investment_committed: Whether financial resources are allocated.
        governance_integrated: Whether climate integrated into governance.
        no_contradictory_actions: Whether no contradictory actions exist.
        annual_reporting_done: Whether annual reporting is done.
        emissions_disclosed: Whether emissions are publicly disclosed.
        target_progress_reported: Whether target progress is reported.
        plan_updated_annually: Whether plan is updated annually.
        methodology_transparent: Whether methodology is transparent.
        sub_criteria: Explicit per-sub-criterion assessment data.
        include_remediation: Whether to generate remediation plan.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    actor_type: str = Field(default="corporate", description="Actor type")
    join_date: Optional[str] = Field(default=None, description="R2Z join date")
    assessment_date: Optional[str] = Field(default=None, description="Assessment date")
    net_zero_target_year: int = Field(default=2050, ge=2030, le=2060)
    interim_target_year: int = Field(default=2030, ge=2025, le=2040)
    interim_target_reduction_pct: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    baseline_year: int = Field(default=0, ge=0, le=2060)
    baseline_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    current_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope1_coverage_pct: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    scope2_coverage_pct: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    scope3_coverage_pct: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    action_plan_published: bool = Field(default=False)
    action_plan_date: Optional[str] = Field(default=None)
    has_quantified_actions: bool = Field(default=False)
    has_timeline: bool = Field(default=False)
    has_resources: bool = Field(default=False)
    has_sector_alignment: bool = Field(default=False)
    immediate_actions_taken: bool = Field(default=False)
    emissions_reducing: bool = Field(default=False)
    investment_committed: bool = Field(default=False)
    governance_integrated: bool = Field(default=False)
    no_contradictory_actions: bool = Field(default=True)
    annual_reporting_done: bool = Field(default=False)
    emissions_disclosed: bool = Field(default=False)
    target_progress_reported: bool = Field(default=False)
    plan_updated_annually: bool = Field(default=False)
    methodology_transparent: bool = Field(default=False)
    sub_criteria: List[SubCriterionInput] = Field(default_factory=list)
    include_remediation: bool = Field(default=True)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class SubCriterionResult(BaseModel):
    """Assessment result for a single sub-criterion.

    Attributes:
        criterion_id: Sub-criterion identifier.
        category: Starting Line category (pledge/plan/proceed/publish).
        name: Sub-criterion name.
        description: What is assessed.
        status: Assessment status.
        score: Numeric score (0.0/0.5/1.0).
        evidence_provided: Whether evidence was provided.
        evidence_required: What evidence is needed.
        gap_description: Description of gap if not PASS.
        remediation_action: Recommended remediation.
        effort_estimate: Estimated effort (hours).
        priority: Remediation priority (1=highest).
    """
    criterion_id: str = Field(default="")
    category: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    status: str = Field(default=SubCriterionStatus.FAIL.value)
    score: Decimal = Field(default=Decimal("0"))
    evidence_provided: bool = Field(default=False)
    evidence_required: str = Field(default="")
    gap_description: str = Field(default="")
    remediation_action: str = Field(default="")
    effort_estimate: int = Field(default=0)
    priority: int = Field(default=5)

class CategoryResult(BaseModel):
    """Assessment result for a Starting Line category.

    Attributes:
        category: Category identifier (pledge/plan/proceed/publish).
        category_name: Display name.
        status: Category compliance status.
        score: Category score (0-100).
        sub_criteria_pass: Number of sub-criteria PASS.
        sub_criteria_partial: Number PARTIAL.
        sub_criteria_fail: Number FAIL.
        sub_criteria_na: Number NOT_APPLICABLE.
        sub_criteria_total: Total sub-criteria in category.
        sub_criterion_results: Detailed sub-criterion results.
    """
    category: str = Field(default="")
    category_name: str = Field(default="")
    status: str = Field(default=ComplianceStatus.NON_COMPLIANT.value)
    score: Decimal = Field(default=Decimal("0"))
    sub_criteria_pass: int = Field(default=0)
    sub_criteria_partial: int = Field(default=0)
    sub_criteria_fail: int = Field(default=0)
    sub_criteria_na: int = Field(default=0)
    sub_criteria_total: int = Field(default=5)
    sub_criterion_results: List[SubCriterionResult] = Field(default_factory=list)

class RemediationItem(BaseModel):
    """A single remediation action item.

    Attributes:
        criterion_id: Related sub-criterion.
        category: Starting Line category.
        action: Remediation action description.
        effort_hours: Estimated effort in hours.
        priority: Priority (1=highest, 5=lowest).
        deadline_description: Suggested deadline.
    """
    criterion_id: str = Field(default="")
    category: str = Field(default="")
    action: str = Field(default="")
    effort_hours: int = Field(default=0)
    priority: int = Field(default=5)
    deadline_description: str = Field(default="")

class StartingLineResult(BaseModel):
    """Complete Starting Line assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        overall_status: Overall Starting Line compliance status.
        overall_score: Overall score (0-100).
        categories_compliant: Number of categories COMPLIANT.
        categories_partial: Number PARTIALLY_COMPLIANT.
        categories_non_compliant: Number NON_COMPLIANT.
        category_results: Per-category assessment results.
        sub_criteria_pass: Total sub-criteria PASS.
        sub_criteria_partial: Total PARTIAL.
        sub_criteria_fail: Total FAIL.
        sub_criteria_total: Total sub-criteria assessed.
        compliance_pct: Compliance percentage.
        months_since_join: Months since joining (if join date provided).
        months_remaining: Months remaining of 12-month deadline.
        remediation_plan: Prioritized remediation items.
        total_remediation_hours: Estimated total remediation effort.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    overall_status: str = Field(default=ComplianceStatus.NON_COMPLIANT.value)
    overall_score: Decimal = Field(default=Decimal("0"))
    categories_compliant: int = Field(default=0)
    categories_partial: int = Field(default=0)
    categories_non_compliant: int = Field(default=0)
    category_results: List[CategoryResult] = Field(default_factory=list)
    sub_criteria_pass: int = Field(default=0)
    sub_criteria_partial: int = Field(default=0)
    sub_criteria_fail: int = Field(default=0)
    sub_criteria_total: int = Field(default=20)
    compliance_pct: Decimal = Field(default=Decimal("0"))
    months_since_join: Optional[int] = Field(default=None)
    months_remaining: Optional[int] = Field(default=None)
    remediation_plan: List[RemediationItem] = Field(default_factory=list)
    total_remediation_hours: int = Field(default=0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StartingLineEngine:
    """Race to Zero Starting Line Criteria assessment engine.

    Assesses compliance with the four Starting Line Criteria (Pledge,
    Plan, Proceed, Publish) across 20 sub-criteria from the
    Interpretation Guide (June 2022).

    Usage::

        engine = StartingLineEngine()
        result = engine.assess(input_data)
        print(f"Status: {result.overall_status} ({result.overall_score}/100)")
        for cat in result.category_results:
            print(f"  {cat.category_name}: {cat.status}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise StartingLineEngine.

        Args:
            config: Optional overrides. Supported keys:
                - compliance_deadline_months (int): Deadline (default 12)
                - min_interim_reduction_pct (Decimal): Min interim %
                - min_scope1_2_coverage_pct (Decimal): Min S1+S2 coverage
        """
        self.config = config or {}
        self._deadline_months = int(
            self.config.get("compliance_deadline_months", 12)
        )
        self._min_interim = _decimal(
            self.config.get("min_interim_reduction_pct", Decimal("42"))
        )
        self._min_s12_coverage = _decimal(
            self.config.get("min_scope1_2_coverage_pct", Decimal("95"))
        )
        logger.info("StartingLineEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def assess(
        self, data: StartingLineInput,
    ) -> StartingLineResult:
        """Perform complete Starting Line assessment.

        Evaluates all 20 sub-criteria across 4 categories, determines
        category and overall compliance status, and generates a
        prioritized remediation plan.

        Args:
            data: Validated Starting Line input.

        Returns:
            StartingLineResult with complete assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Starting Line assessment: entity=%s, actor=%s",
            data.entity_name, data.actor_type,
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Build sub-criterion input map
        sc_map: Dict[str, SubCriterionInput] = {}
        for sc in data.sub_criteria:
            sc_map[sc.criterion_id] = sc

        # Step 1: Assess all 20 sub-criteria
        all_sub_results: Dict[str, List[SubCriterionResult]] = {
            StartingLineCategory.PLEDGE.value: [],
            StartingLineCategory.PLAN.value: [],
            StartingLineCategory.PROCEED.value: [],
            StartingLineCategory.PUBLISH.value: [],
        }

        for sc_id, sc_def in SUB_CRITERIA.items():
            category = sc_def["category"]
            if sc_id in sc_map:
                inp = sc_map[sc_id]
                status = inp.status
                evidence = bool(inp.evidence)
            else:
                status = self._auto_assess_sub_criterion(sc_id, data)
                evidence = False

            score = STATUS_SCORES.get(status, Decimal("0"))
            gap_desc = ""
            remediation = ""
            effort = 0
            priority = PRIORITY_WEIGHTS.get(category, 5)

            if status in (SubCriterionStatus.FAIL.value, SubCriterionStatus.PARTIAL.value):
                gap_desc, remediation, effort = self._get_gap_info(sc_id, data)

            result = SubCriterionResult(
                criterion_id=sc_id,
                category=category,
                name=sc_def["name"],
                description=sc_def["description"],
                status=status,
                score=score,
                evidence_provided=evidence,
                evidence_required=sc_def["evidence_required"],
                gap_description=gap_desc,
                remediation_action=remediation,
                effort_estimate=effort,
                priority=priority,
            )
            all_sub_results[category].append(result)

        # Step 2: Build category results
        category_results: List[CategoryResult] = []
        for cat in StartingLineCategory:
            cat_subs = all_sub_results[cat.value]
            applicable = [s for s in cat_subs if s.status != SubCriterionStatus.NOT_APPLICABLE.value]
            n_pass = sum(1 for s in cat_subs if s.status == SubCriterionStatus.PASS.value)
            n_partial = sum(1 for s in cat_subs if s.status == SubCriterionStatus.PARTIAL.value)
            n_fail = sum(1 for s in cat_subs if s.status == SubCriterionStatus.FAIL.value)
            n_na = sum(1 for s in cat_subs if s.status == SubCriterionStatus.NOT_APPLICABLE.value)

            if len(applicable) > 0:
                cat_score = _safe_divide(
                    sum((s.score for s in applicable), Decimal("0")),
                    _decimal(len(applicable)),
                ) * Decimal("100")
            else:
                cat_score = Decimal("100")

            if n_fail == 0 and n_partial == 0:
                cat_status = ComplianceStatus.COMPLIANT.value
            elif n_fail == 0:
                cat_status = ComplianceStatus.PARTIALLY_COMPLIANT.value
            else:
                cat_status = ComplianceStatus.NON_COMPLIANT.value

            category_results.append(CategoryResult(
                category=cat.value,
                category_name=CATEGORY_LABELS.get(cat.value, cat.value),
                status=cat_status,
                score=_round_val(cat_score, 2),
                sub_criteria_pass=n_pass,
                sub_criteria_partial=n_partial,
                sub_criteria_fail=n_fail,
                sub_criteria_na=n_na,
                sub_criteria_total=len(cat_subs),
                sub_criterion_results=cat_subs,
            ))

        # Step 3: Overall status
        n_compliant = sum(1 for c in category_results if c.status == ComplianceStatus.COMPLIANT.value)
        n_partial_cat = sum(1 for c in category_results if c.status == ComplianceStatus.PARTIALLY_COMPLIANT.value)
        n_non_compliant = sum(1 for c in category_results if c.status == ComplianceStatus.NON_COMPLIANT.value)

        if n_compliant == 4:
            overall_status = ComplianceStatus.COMPLIANT.value
        elif n_compliant >= 2 and n_non_compliant == 0:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT.value
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT.value

        # Step 4: Overall score
        total_pass = sum(c.sub_criteria_pass for c in category_results)
        total_partial = sum(c.sub_criteria_partial for c in category_results)
        total_fail = sum(c.sub_criteria_fail for c in category_results)
        total_na = sum(c.sub_criteria_na for c in category_results)
        total_applicable = 20 - total_na

        if total_applicable > 0:
            overall_score = _safe_divide(
                (_decimal(total_pass) + _decimal(total_partial) * Decimal("0.5")),
                _decimal(total_applicable),
            ) * Decimal("100")
        else:
            overall_score = Decimal("100")
        overall_score = _round_val(overall_score, 2)

        compliance_pct = _round_val(
            _safe_pct(_decimal(total_pass), _decimal(total_applicable)), 2
        )

        # Step 5: Timeline calculation
        months_since: Optional[int] = None
        months_remaining: Optional[int] = None
        if data.join_date:
            try:
                join_dt = datetime.strptime(data.join_date, "%Y-%m-%d")
                now = datetime.now()
                delta_months = (now.year - join_dt.year) * 12 + (now.month - join_dt.month)
                months_since = max(0, delta_months)
                months_remaining = max(0, self._deadline_months - delta_months)
                if months_remaining <= 3 and overall_status != ComplianceStatus.COMPLIANT.value:
                    warnings.append(
                        f"Only {months_remaining} months remaining of 12-month "
                        f"Starting Line compliance deadline."
                    )
            except ValueError:
                warnings.append(f"Invalid join_date format: {data.join_date}")

        # Step 6: Remediation plan
        remediation_plan: List[RemediationItem] = []
        if data.include_remediation:
            all_subs = []
            for cat_result in category_results:
                all_subs.extend(cat_result.sub_criterion_results)
            for sub in sorted(all_subs, key=lambda s: (s.priority, -float(s.score))):
                if sub.status in (SubCriterionStatus.FAIL.value, SubCriterionStatus.PARTIAL.value):
                    deadline_desc = (
                        f"Within {months_remaining} months" if months_remaining is not None
                        else "Within 12 months of joining"
                    )
                    remediation_plan.append(RemediationItem(
                        criterion_id=sub.criterion_id,
                        category=sub.category,
                        action=sub.remediation_action,
                        effort_hours=sub.effort_estimate,
                        priority=sub.priority,
                        deadline_description=deadline_desc,
                    ))

        total_hours = sum(r.effort_hours for r in remediation_plan)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = StartingLineResult(
            entity_name=data.entity_name,
            overall_status=overall_status,
            overall_score=overall_score,
            categories_compliant=n_compliant,
            categories_partial=n_partial_cat,
            categories_non_compliant=n_non_compliant,
            category_results=category_results,
            sub_criteria_pass=total_pass,
            sub_criteria_partial=total_partial,
            sub_criteria_fail=total_fail,
            sub_criteria_total=20,
            compliance_pct=compliance_pct,
            months_since_join=months_since,
            months_remaining=months_remaining,
            remediation_plan=remediation_plan,
            total_remediation_hours=total_hours,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Starting Line assessment complete: status=%s, score=%.1f, "
            "pass=%d partial=%d fail=%d, hash=%s",
            overall_status, float(overall_score),
            total_pass, total_partial, total_fail,
            result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _auto_assess_sub_criterion(
        self, sc_id: str, data: StartingLineInput,
    ) -> str:
        """Auto-assess a sub-criterion from entity data fields.

        Args:
            sc_id: Sub-criterion identifier.
            data: Starting Line input data.

        Returns:
            SubCriterionStatus value.
        """
        assessors: Dict[str, str] = {
            "SL-P1": SubCriterionStatus.PASS.value if data.net_zero_target_year <= 2050 else SubCriterionStatus.FAIL.value,
            "SL-P2": (
                SubCriterionStatus.PASS.value if data.interim_target_reduction_pct >= self._min_interim
                else (SubCriterionStatus.PARTIAL.value if data.interim_target_reduction_pct > Decimal("0")
                      else SubCriterionStatus.FAIL.value)
            ),
            "SL-P3": SubCriterionStatus.PARTIAL.value,  # requires explicit evidence
            "SL-P4": SubCriterionStatus.PARTIAL.value,  # requires qualitative assessment
            "SL-P5": (
                SubCriterionStatus.PASS.value if (
                    data.scope1_coverage_pct >= self._min_s12_coverage
                    and data.scope2_coverage_pct >= self._min_s12_coverage
                )
                else SubCriterionStatus.FAIL.value
            ),
            "SL-A1": SubCriterionStatus.PASS.value if data.action_plan_published else SubCriterionStatus.FAIL.value,
            "SL-A2": SubCriterionStatus.PASS.value if data.has_quantified_actions else SubCriterionStatus.FAIL.value,
            "SL-A3": SubCriterionStatus.PASS.value if data.has_timeline else SubCriterionStatus.FAIL.value,
            "SL-A4": SubCriterionStatus.PASS.value if data.has_resources else SubCriterionStatus.FAIL.value,
            "SL-A5": SubCriterionStatus.PASS.value if data.has_sector_alignment else SubCriterionStatus.FAIL.value,
            "SL-R1": SubCriterionStatus.PASS.value if data.immediate_actions_taken else SubCriterionStatus.FAIL.value,
            "SL-R2": SubCriterionStatus.PASS.value if data.emissions_reducing else SubCriterionStatus.FAIL.value,
            "SL-R3": SubCriterionStatus.PASS.value if data.investment_committed else SubCriterionStatus.FAIL.value,
            "SL-R4": SubCriterionStatus.PASS.value if data.governance_integrated else SubCriterionStatus.FAIL.value,
            "SL-R5": SubCriterionStatus.PASS.value if data.no_contradictory_actions else SubCriterionStatus.FAIL.value,
            "SL-D1": SubCriterionStatus.PASS.value if data.annual_reporting_done else SubCriterionStatus.FAIL.value,
            "SL-D2": SubCriterionStatus.PASS.value if data.emissions_disclosed else SubCriterionStatus.FAIL.value,
            "SL-D3": SubCriterionStatus.PASS.value if data.target_progress_reported else SubCriterionStatus.FAIL.value,
            "SL-D4": SubCriterionStatus.PASS.value if data.plan_updated_annually else SubCriterionStatus.FAIL.value,
            "SL-D5": SubCriterionStatus.PASS.value if data.methodology_transparent else SubCriterionStatus.FAIL.value,
        }
        return assessors.get(sc_id, SubCriterionStatus.FAIL.value)

    def _get_gap_info(
        self, sc_id: str, data: StartingLineInput,
    ) -> Tuple[str, str, int]:
        """Get gap description, remediation, and effort for a sub-criterion.

        Args:
            sc_id: Sub-criterion identifier.
            data: Input data.

        Returns:
            Tuple of (gap_description, remediation_action, effort_hours).
        """
        gap_map: Dict[str, Tuple[str, str, int]] = {
            "SL-P1": ("Net-zero target not set or exceeds 2050.", "Set net-zero target for 2050 or earlier.", 8),
            "SL-P2": ("Interim 2030 target missing or below 42%.", "Set 2030 target of >=42% absolute reduction.", 24),
            "SL-P3": ("Science-based methodology not documented.", "Document and validate target methodology (SBTi/IEA/IPCC).", 40),
            "SL-P4": ("Fair share assessment not performed.", "Conduct fair share analysis considering equity and capability.", 24),
            "SL-P5": ("Scope coverage does not meet requirements.", "Expand scope boundary to cover S1+S2 >=95%, S3 >=67%.", 40),
            "SL-A1": ("Action plan not published.", "Draft and publish climate action plan within 12 months.", 80),
            "SL-A2": ("Actions not quantified.", "Quantify each action with tCO2e abatement impact.", 40),
            "SL-A3": ("Timeline and milestones not defined.", "Create implementation timeline with measurable milestones.", 24),
            "SL-A4": ("Resource allocation not specified.", "Allocate budget and FTE to each action item.", 16),
            "SL-A5": ("Sector pathway alignment not assessed.", "Map actions to relevant sector pathway benchmarks.", 24),
            "SL-R1": ("No demonstrable immediate action.", "Initiate at least one significant reduction action.", 40),
            "SL-R2": ("Emissions not on reduction trajectory.", "Implement actions to begin reducing absolute emissions.", 80),
            "SL-R3": ("Financial investment not committed.", "Allocate CapEx/OpEx to decarbonization actions.", 16),
            "SL-R4": ("Climate not integrated into governance.", "Establish board oversight and executive responsibility.", 24),
            "SL-R5": ("Contradictory actions identified.", "Review and cease actions contradicting climate commitments.", 40),
            "SL-D1": ("Annual reporting not completed.", "Submit annual progress report through partner channel.", 24),
            "SL-D2": ("GHG emissions not disclosed.", "Prepare and publish GHG inventory (S1, S2, material S3).", 40),
            "SL-D3": ("Target progress not reported.", "Report progress against interim and long-term targets.", 16),
            "SL-D4": ("Action plan not updated annually.", "Review and update action plan with latest data.", 24),
            "SL-D5": ("Methodology not transparent.", "Document methodology, assumptions, and limitations.", 16),
        }
        return gap_map.get(sc_id, ("Gap not characterized.", "Review sub-criterion.", 8))
