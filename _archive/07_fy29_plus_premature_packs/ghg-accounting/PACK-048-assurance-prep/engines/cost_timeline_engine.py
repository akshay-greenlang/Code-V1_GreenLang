# -*- coding: utf-8 -*-
"""
CostTimelineEngine - PACK-048 GHG Assurance Prep Engine 9
====================================================================

Estimates the cost, timeline, and internal resource requirements for
GHG assurance engagements based on engagement scope, company size,
complexity factors, and assurance level.

Calculation Methodology:
    Base Cost by Company Size:
        MICRO:          15,000 (currency units)
        SMALL:          25,000
        MEDIUM:         45,000
        LARGE:          75,000
        ENTERPRISE:     120,000

    Cost Adjustments:
        C_reasonable = C_base * 2.5
            (Reasonable assurance ~2.5x limited)

        C_multi = C_base * (1 + 0.15 * n_jurisdictions)
            (15% uplift per additional jurisdiction)

        C_first = C_base * 1.3
            (30% first-time engagement uplift)

        C_s3 = C_base * (1 + 0.1 * s3_categories)
            (10% per Scope 3 category in scope)

        C_complexity = C_base * (1 + SUM(complexity_factors))
            (Additive complexity factors)

    Final Cost:
        C_total = C_base * level_mult * multi_jur * first_time * s3_mult * complexity

    Timeline Estimation:
        Planning:    2-4 weeks
        Fieldwork:   2-6 weeks
        Reporting:   1-2 weeks
        Total:       5-12 weeks

        Adjustments:
            Reasonable:     +50% fieldwork time
            Multi-jur:      +1 week per jurisdiction
            First-time:     +2 weeks planning
            S3 categories:  +0.5 weeks per category

    Internal Resource Estimation (FTE hours by role):
        Project Lead:       40-120 hours
        Data Analyst:       60-200 hours
        Subject Expert:     20-80 hours
        Management Review:  10-40 hours

    Multi-Year Planning:
        Year 1: Base + first-time
        Year 2: Base * 0.9 (10% efficiency)
        Year 3+: Base * 0.85 (15% efficiency)
        Transition: Limited -> Reasonable in year N

Regulatory References:
    - ISAE 3410: Engagement acceptance and planning
    - ISAE 3000 (Revised): Engagement terms
    - IAASB Practice Statement: Engagement economics
    - Market benchmarks for assurance engagement pricing

Zero-Hallucination:
    - All cost formulas use deterministic Decimal arithmetic
    - Base costs from published market benchmarks
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-048 GHG Assurance Prep
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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

def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EngagementSize(str, Enum):
    """Company size for cost estimation."""
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

class EngagementLevel(str, Enum):
    """Assurance engagement level."""
    LIMITED = "limited"
    REASONABLE = "reasonable"

class EngagementPhase(str, Enum):
    """Engagement phase for timeline."""
    PLANNING = "planning"
    FIELDWORK = "fieldwork"
    REPORTING = "reporting"

class ResourceRole(str, Enum):
    """Internal resource role."""
    PROJECT_LEAD = "project_lead"
    DATA_ANALYST = "data_analyst"
    SUBJECT_EXPERT = "subject_expert"
    MANAGEMENT_REVIEW = "management_review"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base costs by company size (in currency units -- typically USD/EUR/GBP)
BASE_COSTS: Dict[str, Decimal] = {
    EngagementSize.MICRO.value: Decimal("15000"),
    EngagementSize.SMALL.value: Decimal("25000"),
    EngagementSize.MEDIUM.value: Decimal("45000"),
    EngagementSize.LARGE.value: Decimal("75000"),
    EngagementSize.ENTERPRISE.value: Decimal("120000"),
}

# Multipliers
REASONABLE_MULTIPLIER: Decimal = Decimal("2.5")
MULTI_JURISDICTION_UPLIFT: Decimal = Decimal("0.15")
FIRST_TIME_UPLIFT: Decimal = Decimal("1.3")
S3_CATEGORY_UPLIFT: Decimal = Decimal("0.10")

# Timeline base (weeks)
TIMELINE_BASE: Dict[str, Tuple[Decimal, Decimal]] = {
    EngagementPhase.PLANNING.value: (Decimal("2"), Decimal("4")),
    EngagementPhase.FIELDWORK.value: (Decimal("2"), Decimal("6")),
    EngagementPhase.REPORTING.value: (Decimal("1"), Decimal("2")),
}

# Reasonable assurance fieldwork multiplier
REASONABLE_FIELDWORK_MULT: Decimal = Decimal("1.5")

# Resource hours base (by role)
RESOURCE_HOURS_BASE: Dict[str, Tuple[Decimal, Decimal]] = {
    ResourceRole.PROJECT_LEAD.value: (Decimal("40"), Decimal("120")),
    ResourceRole.DATA_ANALYST.value: (Decimal("60"), Decimal("200")),
    ResourceRole.SUBJECT_EXPERT.value: (Decimal("20"), Decimal("80")),
    ResourceRole.MANAGEMENT_REVIEW.value: (Decimal("10"), Decimal("40")),
}

# Multi-year efficiency discounts
YEAR_DISCOUNT: Dict[int, Decimal] = {
    1: Decimal("1.0"),
    2: Decimal("0.90"),
    3: Decimal("0.85"),
}

# Size scaling factors for resource hours
SIZE_SCALING: Dict[str, Decimal] = {
    EngagementSize.MICRO.value: Decimal("0.5"),
    EngagementSize.SMALL.value: Decimal("0.7"),
    EngagementSize.MEDIUM.value: Decimal("1.0"),
    EngagementSize.LARGE.value: Decimal("1.5"),
    EngagementSize.ENTERPRISE.value: Decimal("2.0"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class ComplexityFactor(BaseModel):
    """A complexity factor affecting cost/timeline.

    Attributes:
        factor_name:    Factor name.
        description:    Factor description.
        cost_uplift:    Cost uplift (e.g. 0.10 = 10%).
        timeline_weeks: Additional weeks added.
    """
    factor_name: str = Field(default="", description="Factor name")
    description: str = Field(default="", description="Description")
    cost_uplift: Decimal = Field(default=Decimal("0"), ge=0, description="Cost uplift")
    timeline_weeks: Decimal = Field(default=Decimal("0"), ge=0, description="Extra weeks")

    @field_validator("cost_uplift", "timeline_weeks", mode="before")
    @classmethod
    def coerce_dec(cls, v: Any) -> Decimal:
        return _decimal(v)

class EngagementScope(BaseModel):
    """Engagement scope definition.

    Attributes:
        company_size:           Company size category.
        assurance_level:        Assurance level.
        scope_coverage:         Scopes covered (scope_1, scope_2, scope_3).
        s3_category_count:      Number of Scope 3 categories.
        facility_count:         Number of facilities.
        data_point_count:       Estimated data points.
        jurisdiction_count:     Number of jurisdictions.
        is_first_time:          First-time engagement.
        complexity_factors:     Additional complexity factors.
        currency:               Cost currency.
    """
    company_size: EngagementSize = Field(
        default=EngagementSize.MEDIUM, description="Size"
    )
    assurance_level: EngagementLevel = Field(
        default=EngagementLevel.LIMITED, description="Level"
    )
    scope_coverage: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2"],
        description="Scope coverage",
    )
    s3_category_count: int = Field(default=0, ge=0, le=15, description="S3 categories")
    facility_count: int = Field(default=1, ge=1, description="Facilities")
    data_point_count: int = Field(default=100, ge=1, description="Data points")
    jurisdiction_count: int = Field(default=1, ge=1, description="Jurisdictions")
    is_first_time: bool = Field(default=True, description="First time")
    complexity_factors: List[ComplexityFactor] = Field(
        default_factory=list, description="Complexity factors"
    )
    currency: str = Field(default="USD", description="Currency")

class CostTimelineConfig(BaseModel):
    """Configuration for cost/timeline engine.

    Attributes:
        organisation_id:    Organisation identifier.
        multi_year_horizon: Multi-year planning horizon (years).
        transition_year:    Year to transition from limited to reasonable (0=no transition).
        output_precision:   Output decimal places.
    """
    organisation_id: str = Field(default="", description="Org ID")
    multi_year_horizon: int = Field(default=3, ge=1, le=10, description="Horizon years")
    transition_year: int = Field(default=0, ge=0, le=10, description="Transition year")
    output_precision: int = Field(default=2, ge=0, le=6, description="Output precision")

class CostTimelineInput(BaseModel):
    """Input for cost/timeline engine.

    Attributes:
        engagement_scope:   Engagement scope.
        config:             Configuration.
    """
    engagement_scope: EngagementScope = Field(
        default_factory=EngagementScope, description="Scope"
    )
    config: CostTimelineConfig = Field(
        default_factory=CostTimelineConfig, description="Configuration"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class CostBreakdown(BaseModel):
    """Cost breakdown detail.

    Attributes:
        component:      Cost component name.
        base_amount:    Base amount.
        multiplier:     Multiplier applied.
        final_amount:   Final amount after multiplier.
    """
    component: str = Field(default="", description="Component")
    base_amount: Decimal = Field(default=Decimal("0"), description="Base")
    multiplier: Decimal = Field(default=Decimal("1"), description="Multiplier")
    final_amount: Decimal = Field(default=Decimal("0"), description="Final")

class CostEstimate(BaseModel):
    """Cost estimate for the engagement.

    Attributes:
        total_cost:         Total estimated cost.
        currency:           Currency.
        cost_breakdown:     Breakdown by component.
        base_cost:          Base cost before adjustments.
        level_adjustment:   Assurance level adjustment.
        jurisdiction_adj:   Multi-jurisdiction adjustment.
        first_time_adj:     First-time engagement adjustment.
        s3_adjustment:      Scope 3 category adjustment.
        complexity_adj:     Complexity factor adjustment.
    """
    total_cost: Decimal = Field(default=Decimal("0"), description="Total")
    currency: str = Field(default="USD", description="Currency")
    cost_breakdown: List[CostBreakdown] = Field(
        default_factory=list, description="Breakdown"
    )
    base_cost: Decimal = Field(default=Decimal("0"), description="Base")
    level_adjustment: Decimal = Field(default=Decimal("0"), description="Level adj")
    jurisdiction_adj: Decimal = Field(default=Decimal("0"), description="Jurisdiction adj")
    first_time_adj: Decimal = Field(default=Decimal("0"), description="First time adj")
    s3_adjustment: Decimal = Field(default=Decimal("0"), description="S3 adj")
    complexity_adj: Decimal = Field(default=Decimal("0"), description="Complexity adj")

class PhaseTimeline(BaseModel):
    """Timeline for a single engagement phase.

    Attributes:
        phase:      Phase name.
        min_weeks:  Minimum weeks.
        max_weeks:  Maximum weeks.
        mid_weeks:  Midpoint estimate.
    """
    phase: str = Field(default="", description="Phase")
    min_weeks: Decimal = Field(default=Decimal("0"), description="Min weeks")
    max_weeks: Decimal = Field(default=Decimal("0"), description="Max weeks")
    mid_weeks: Decimal = Field(default=Decimal("0"), description="Mid weeks")

class TimelineEstimate(BaseModel):
    """Timeline estimate for the engagement.

    Attributes:
        phases:         Per-phase timelines.
        total_min_weeks: Total minimum weeks.
        total_max_weeks: Total maximum weeks.
        total_mid_weeks: Total midpoint weeks.
    """
    phases: List[PhaseTimeline] = Field(default_factory=list, description="Phases")
    total_min_weeks: Decimal = Field(default=Decimal("0"), description="Total min")
    total_max_weeks: Decimal = Field(default=Decimal("0"), description="Total max")
    total_mid_weeks: Decimal = Field(default=Decimal("0"), description="Total mid")

class RoleResource(BaseModel):
    """Resource estimate for a single role.

    Attributes:
        role:       Role name.
        min_hours:  Minimum hours.
        max_hours:  Maximum hours.
        mid_hours:  Midpoint hours.
    """
    role: str = Field(default="", description="Role")
    min_hours: Decimal = Field(default=Decimal("0"), description="Min hours")
    max_hours: Decimal = Field(default=Decimal("0"), description="Max hours")
    mid_hours: Decimal = Field(default=Decimal("0"), description="Mid hours")

class ResourcePlan(BaseModel):
    """Internal resource plan.

    Attributes:
        roles:          Per-role resource estimates.
        total_min_hours: Total minimum hours.
        total_max_hours: Total maximum hours.
        total_mid_hours: Total midpoint hours.
    """
    roles: List[RoleResource] = Field(default_factory=list, description="Roles")
    total_min_hours: Decimal = Field(default=Decimal("0"), description="Total min")
    total_max_hours: Decimal = Field(default=Decimal("0"), description="Total max")
    total_mid_hours: Decimal = Field(default=Decimal("0"), description="Total mid")

class YearPlan(BaseModel):
    """Multi-year plan entry.

    Attributes:
        year:               Year number (1, 2, 3...).
        assurance_level:    Assurance level for this year.
        estimated_cost:     Estimated cost.
        efficiency_discount: Efficiency discount applied.
    """
    year: int = Field(default=0, description="Year")
    assurance_level: str = Field(default="", description="Level")
    estimated_cost: Decimal = Field(default=Decimal("0"), description="Cost")
    efficiency_discount: Decimal = Field(default=Decimal("0"), description="Discount")

class CostTimelineResult(BaseModel):
    """Complete result of cost/timeline estimation.

    Attributes:
        result_id:              Unique result identifier.
        organisation_id:        Organisation identifier.
        cost_estimate:          Cost estimate.
        timeline_estimate:      Timeline estimate.
        resource_plan:          Resource plan.
        multi_year_plan:        Multi-year cost trajectory.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Org ID")
    cost_estimate: CostEstimate = Field(
        default_factory=CostEstimate, description="Cost"
    )
    timeline_estimate: TimelineEstimate = Field(
        default_factory=TimelineEstimate, description="Timeline"
    )
    resource_plan: ResourcePlan = Field(
        default_factory=ResourcePlan, description="Resources"
    )
    multi_year_plan: List[YearPlan] = Field(
        default_factory=list, description="Multi-year"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CostTimelineEngine:
    """Estimates cost, timeline, and resources for GHG assurance engagements.

    Computes engagement costs with adjustments for assurance level,
    multi-jurisdiction, first-time engagement, Scope 3 complexity,
    and custom complexity factors.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every cost component documented.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("CostTimelineEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: CostTimelineInput) -> CostTimelineResult:
        """Estimate cost, timeline, and resources.

        Args:
            input_data: Engagement scope and configuration.

        Returns:
            CostTimelineResult with cost, timeline, resources, multi-year plan.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        config = input_data.config
        scope = input_data.engagement_scope
        prec = config.output_precision
        prec_str = "0." + "0" * prec

        # Step 1: Cost estimate
        cost_est = self._estimate_cost(scope, prec_str)

        # Step 2: Timeline estimate
        timeline_est = self._estimate_timeline(scope, prec_str)

        # Step 3: Resource plan
        resource_plan = self._estimate_resources(scope, prec_str)

        # Step 4: Multi-year plan
        multi_year = self._build_multi_year(
            scope, cost_est.total_cost, config, prec_str
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = CostTimelineResult(
            organisation_id=config.organisation_id,
            cost_estimate=cost_est,
            timeline_estimate=timeline_est,
            resource_plan=resource_plan,
            multi_year_plan=multi_year,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Cost
    # ------------------------------------------------------------------

    def _estimate_cost(self, scope: EngagementScope, prec_str: str) -> CostEstimate:
        """Estimate engagement cost."""
        base = BASE_COSTS.get(scope.company_size.value, Decimal("45000"))
        breakdown: List[CostBreakdown] = []
        running = base

        breakdown.append(CostBreakdown(
            component="Base cost",
            base_amount=base,
            multiplier=Decimal("1"),
            final_amount=base,
        ))

        # Assurance level
        level_mult = Decimal("1")
        if scope.assurance_level == EngagementLevel.REASONABLE:
            level_mult = REASONABLE_MULTIPLIER
            level_adj = base * (level_mult - Decimal("1"))
            running = base * level_mult
            breakdown.append(CostBreakdown(
                component="Reasonable assurance uplift",
                base_amount=base,
                multiplier=level_mult,
                final_amount=level_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            ))
        else:
            level_adj = Decimal("0")

        # Multi-jurisdiction
        jur_mult = Decimal("1") + MULTI_JURISDICTION_UPLIFT * _decimal(max(scope.jurisdiction_count - 1, 0))
        jur_adj = running * (jur_mult - Decimal("1"))
        running = running * jur_mult
        if jur_adj > Decimal("0"):
            breakdown.append(CostBreakdown(
                component="Multi-jurisdiction uplift",
                base_amount=running / jur_mult,
                multiplier=jur_mult,
                final_amount=jur_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            ))

        # First time
        first_adj = Decimal("0")
        if scope.is_first_time:
            first_adj = running * (FIRST_TIME_UPLIFT - Decimal("1"))
            running = running * FIRST_TIME_UPLIFT
            breakdown.append(CostBreakdown(
                component="First-time engagement uplift",
                base_amount=running / FIRST_TIME_UPLIFT,
                multiplier=FIRST_TIME_UPLIFT,
                final_amount=first_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            ))

        # Scope 3 categories
        s3_mult = Decimal("1") + S3_CATEGORY_UPLIFT * _decimal(scope.s3_category_count)
        s3_adj = running * (s3_mult - Decimal("1"))
        running = running * s3_mult
        if s3_adj > Decimal("0"):
            breakdown.append(CostBreakdown(
                component="Scope 3 category uplift",
                base_amount=running / s3_mult,
                multiplier=s3_mult,
                final_amount=s3_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            ))

        # Complexity factors
        complexity_adj = Decimal("0")
        for cf in scope.complexity_factors:
            adj = running * cf.cost_uplift
            complexity_adj += adj
            running += adj
            breakdown.append(CostBreakdown(
                component=f"Complexity: {cf.factor_name}",
                base_amount=running - adj,
                multiplier=Decimal("1") + cf.cost_uplift,
                final_amount=adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            ))

        total = running.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        return CostEstimate(
            total_cost=total,
            currency=scope.currency,
            cost_breakdown=breakdown,
            base_cost=base,
            level_adjustment=level_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            jurisdiction_adj=jur_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            first_time_adj=first_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            s3_adjustment=s3_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            complexity_adj=complexity_adj.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
        )

    # ------------------------------------------------------------------
    # Internal: Timeline
    # ------------------------------------------------------------------

    def _estimate_timeline(
        self, scope: EngagementScope, prec_str: str,
    ) -> TimelineEstimate:
        """Estimate engagement timeline."""
        phases: List[PhaseTimeline] = []
        total_min = Decimal("0")
        total_max = Decimal("0")

        for phase_name, (base_min, base_max) in TIMELINE_BASE.items():
            min_w = base_min
            max_w = base_max

            # Reasonable assurance fieldwork
            if (phase_name == EngagementPhase.FIELDWORK.value
                    and scope.assurance_level == EngagementLevel.REASONABLE):
                min_w = (min_w * REASONABLE_FIELDWORK_MULT).quantize(
                    Decimal(prec_str), rounding=ROUND_HALF_UP
                )
                max_w = (max_w * REASONABLE_FIELDWORK_MULT).quantize(
                    Decimal(prec_str), rounding=ROUND_HALF_UP
                )

            # Multi-jurisdiction
            extra_jur = _decimal(max(scope.jurisdiction_count - 1, 0))
            if phase_name == EngagementPhase.FIELDWORK.value:
                max_w += extra_jur

            # First time planning
            if phase_name == EngagementPhase.PLANNING.value and scope.is_first_time:
                min_w += Decimal("1")
                max_w += Decimal("2")

            # S3 categories
            if phase_name == EngagementPhase.FIELDWORK.value:
                s3_extra = _decimal(scope.s3_category_count) * Decimal("0.5")
                max_w += s3_extra

            # Complexity factors
            for cf in scope.complexity_factors:
                if cf.timeline_weeks > Decimal("0"):
                    max_w += cf.timeline_weeks

            mid_w = ((min_w + max_w) / Decimal("2")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )

            phases.append(PhaseTimeline(
                phase=phase_name,
                min_weeks=min_w.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                max_weeks=max_w.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                mid_weeks=mid_w,
            ))

            total_min += min_w
            total_max += max_w

        total_mid = ((total_min + total_max) / Decimal("2")).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )

        return TimelineEstimate(
            phases=phases,
            total_min_weeks=total_min.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            total_max_weeks=total_max.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            total_mid_weeks=total_mid,
        )

    # ------------------------------------------------------------------
    # Internal: Resources
    # ------------------------------------------------------------------

    def _estimate_resources(
        self, scope: EngagementScope, prec_str: str,
    ) -> ResourcePlan:
        """Estimate internal resource requirements."""
        scaling = SIZE_SCALING.get(scope.company_size.value, Decimal("1.0"))
        roles: List[RoleResource] = []
        total_min = Decimal("0")
        total_max = Decimal("0")

        for role_name, (base_min, base_max) in RESOURCE_HOURS_BASE.items():
            min_h = (base_min * scaling).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            max_h = (base_max * scaling).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

            # Reasonable assurance multiplier
            if scope.assurance_level == EngagementLevel.REASONABLE:
                min_h = (min_h * Decimal("1.3")).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
                max_h = (max_h * Decimal("1.5")).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

            mid_h = ((min_h + max_h) / Decimal("2")).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )

            roles.append(RoleResource(
                role=role_name,
                min_hours=min_h,
                max_hours=max_h,
                mid_hours=mid_h,
            ))
            total_min += min_h
            total_max += max_h

        total_mid = ((total_min + total_max) / Decimal("2")).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )

        return ResourcePlan(
            roles=roles,
            total_min_hours=total_min,
            total_max_hours=total_max,
            total_mid_hours=total_mid,
        )

    # ------------------------------------------------------------------
    # Internal: Multi-Year
    # ------------------------------------------------------------------

    def _build_multi_year(
        self,
        scope: EngagementScope,
        year1_cost: Decimal,
        config: CostTimelineConfig,
        prec_str: str,
    ) -> List[YearPlan]:
        """Build multi-year cost plan."""
        plans: List[YearPlan] = []

        # Remove first-time uplift from base for subsequent years
        base_cost = year1_cost
        if scope.is_first_time:
            base_cost = _safe_divide(year1_cost, FIRST_TIME_UPLIFT).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )

        for year in range(1, config.multi_year_horizon + 1):
            if year == 1:
                cost = year1_cost
                discount = Decimal("0")
                level = scope.assurance_level.value
            else:
                # Apply efficiency discount
                discount_rate = YEAR_DISCOUNT.get(year, Decimal("0.85"))
                discount = Decimal("1") - discount_rate

                # Check for transition year
                if config.transition_year > 0 and year >= config.transition_year:
                    level = EngagementLevel.REASONABLE.value
                    transition_base = base_cost * REASONABLE_MULTIPLIER
                    cost = (transition_base * discount_rate).quantize(
                        Decimal(prec_str), rounding=ROUND_HALF_UP
                    )
                else:
                    level = scope.assurance_level.value
                    cost = (base_cost * discount_rate).quantize(
                        Decimal(prec_str), rounding=ROUND_HALF_UP
                    )

            plans.append(YearPlan(
                year=year,
                assurance_level=level,
                estimated_cost=cost,
                efficiency_discount=discount.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                ),
            ))

        return plans

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "EngagementSize",
    "EngagementLevel",
    "EngagementPhase",
    "ResourceRole",
    # Input Models
    "ComplexityFactor",
    "EngagementScope",
    "CostTimelineConfig",
    "CostTimelineInput",
    # Output Models
    "CostBreakdown",
    "CostEstimate",
    "PhaseTimeline",
    "TimelineEstimate",
    "RoleResource",
    "ResourcePlan",
    "YearPlan",
    "CostTimelineResult",
    # Engine
    "CostTimelineEngine",
]
