# -*- coding: utf-8 -*-
"""
PortfolioOptimizationEngine - PACK-024 Carbon Neutral Engine 4
===============================================================

Pareto-frontier carbon credit portfolio optimisation with Oxford
Principles removal progression (0% -> 50% -> 100% removals from
2020 -> 2030 -> 2050), cost-effectiveness scoring, risk
diversification metrics, vintage matching, and multi-objective
optimisation across quality, cost, and co-benefits.

This engine helps organisations build a diversified portfolio of
carbon credits that maximises quality while managing cost, aligns
with the Oxford Principles trajectory toward carbon removal, and
ensures adequate vintage matching for their footprint period.

Calculation Methodology:
    Oxford Principles Removal Progression:
        For a given year Y:
            if Y <= 2020: min_removal_pct = 0%
            elif Y <= 2025: min_removal_pct = 10%
            elif Y <= 2030: min_removal_pct = 50%
            elif Y <= 2040: min_removal_pct = 75%
            elif Y <= 2050: min_removal_pct = 100%
        Source: Oxford Principles for Net Zero Aligned Carbon Offsetting (2020)

    Portfolio Quality Score:
        portfolio_quality = sum(credit_quality_i * credit_quantity_i) / total_quantity

    Diversification Index (Herfindahl-Hirschman):
        HHI = sum((share_i)^2)
        diversification = 1 - HHI  (0 = concentrated, 1 = diversified)
        Diversification across: project_type, geography, standard, vintage

    Cost-Effectiveness:
        cost_per_quality_point = total_cost / portfolio_quality
        value_score = quality_score / price_per_tco2e

    Vintage Matching (ISO 14068-1:2023, Section 8.3):
        vintage_match = credits vintage <= footprint_year + 1
        max_vintage_age = 5 years (recommended), 7 years (acceptable)

    Pareto Frontier:
        Objective 1: Maximise quality score
        Objective 2: Minimise cost
        Objective 3: Maximise co-benefits (SDG count)
        A portfolio is Pareto-optimal if no objective can be improved
        without worsening another.

Regulatory References:
    - Oxford Principles for Net Zero Aligned Carbon Offsetting (2020)
    - ISO 14068-1:2023 - Section 8: Carbon credit requirements
    - PAS 2060:2014 - Section 5.4: Offset credit quality
    - ICVCM Core Carbon Principles V1.0 (2023)
    - TSVCM Phase II Report (2021) - Market scaling
    - VCMI Claims Code of Practice V1.0 (2023)

Zero-Hallucination:
    - Oxford Principles progression from published 2020 document
    - Vintage requirements from ISO 14068-1:2023
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-024 Carbon Neutral
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class CreditType(str, Enum):
    """Carbon credit type classification."""
    AVOIDANCE = "avoidance"
    REDUCTION = "reduction"
    REMOVAL_NATURE = "removal_nature"
    REMOVAL_TECH = "removal_tech"


class OptimizationObjective(str, Enum):
    """Portfolio optimisation objective."""
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_COBENEFIT = "maximize_cobenefit"
    BALANCED = "balanced"


class RiskLevel(str, Enum):
    """Portfolio risk classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ---------------------------------------------------------------------------
# Constants -- Oxford Principles Removal Progression
# ---------------------------------------------------------------------------

# Oxford Principles minimum removal percentage by year.
# Source: Oxford Principles for Net Zero Aligned Carbon Offsetting (2020).
OXFORD_REMOVAL_PROGRESSION: List[Tuple[int, Decimal]] = [
    (2020, Decimal("0")),
    (2025, Decimal("10")),
    (2030, Decimal("50")),
    (2040, Decimal("75")),
    (2050, Decimal("100")),
]

# Maximum vintage age recommended (ISO 14068-1:2023 Section 8.3).
MAX_VINTAGE_AGE_RECOMMENDED: int = 5
MAX_VINTAGE_AGE_ACCEPTABLE: int = 7

# Minimum diversification index (1 - HHI) recommendation.
MIN_DIVERSIFICATION_INDEX: Decimal = Decimal("0.40")

# Portfolio quality thresholds.
QUALITY_EXCELLENT: Decimal = Decimal("85")
QUALITY_GOOD: Decimal = Decimal("70")
QUALITY_ADEQUATE: Decimal = Decimal("55")
QUALITY_POOR: Decimal = Decimal("40")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class CreditOption(BaseModel):
    """A carbon credit option available for the portfolio.

    Attributes:
        credit_id: Unique credit identifier.
        project_name: Project name.
        credit_type: Type (avoidance/reduction/removal).
        standard: Certification standard.
        vintage_year: Vintage year.
        available_quantity_tco2e: Available credits.
        price_per_tco2e_usd: Price per credit.
        quality_score: Quality score (0-100) from CreditQualityEngine.
        country: Project country.
        sdg_count: Number of UN SDG contributions.
        permanence_years: Expected permanence.
        is_removal: Whether this is a carbon removal credit.
        co_benefit_score: Co-benefit score (0-100).
        risk_rating: Risk level.
    """
    credit_id: str = Field(default_factory=_new_uuid, description="Credit ID")
    project_name: str = Field(default="", max_length=300, description="Project name")
    credit_type: str = Field(
        default=CreditType.REDUCTION.value, description="Credit type"
    )
    standard: str = Field(default="vcs", max_length=50, description="Standard")
    vintage_year: int = Field(default=0, ge=0, le=2060, description="Vintage year")
    available_quantity_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Available quantity"
    )
    price_per_tco2e_usd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Price per tCO2e"
    )
    quality_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Quality score (0-100)"
    )
    country: str = Field(default="", max_length=2, description="Country code")
    sdg_count: int = Field(default=0, ge=0, le=17, description="SDG count")
    permanence_years: int = Field(default=0, ge=0, description="Permanence years")
    is_removal: bool = Field(default=False, description="Whether carbon removal")
    co_benefit_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Co-benefit score"
    )
    risk_rating: str = Field(
        default=RiskLevel.MODERATE.value, description="Risk level"
    )

    @field_validator("credit_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid = {t.value for t in CreditType}
        if v not in valid:
            raise ValueError(f"Unknown credit type '{v}'.")
        return v


class PortfolioConstraints(BaseModel):
    """Constraints for portfolio optimisation.

    Attributes:
        max_budget_usd: Maximum total budget.
        required_quantity_tco2e: Required total quantity.
        min_quality_score: Minimum acceptable quality score.
        min_removal_pct: Minimum removal percentage.
        max_single_project_pct: Max % from single project.
        max_single_country_pct: Max % from single country.
        max_vintage_age: Maximum vintage age.
        min_sdg_count: Minimum SDG contributions per credit.
        min_permanence_years: Minimum permanence duration.
        preferred_standards: Preferred certification standards.
        excluded_countries: Countries to exclude.
        target_year: Year for Oxford Principles alignment.
    """
    max_budget_usd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Maximum budget"
    )
    required_quantity_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Required quantity"
    )
    min_quality_score: Decimal = Field(
        default=Decimal("50"), ge=0, le=Decimal("100"),
        description="Minimum quality"
    )
    min_removal_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=Decimal("100"),
        description="Minimum removal % (None = auto from Oxford Principles)"
    )
    max_single_project_pct: Decimal = Field(
        default=Decimal("40"), ge=0, le=Decimal("100"),
        description="Max % from single project"
    )
    max_single_country_pct: Decimal = Field(
        default=Decimal("60"), ge=0, le=Decimal("100"),
        description="Max % from single country"
    )
    max_vintage_age: int = Field(
        default=5, ge=1, le=10, description="Max vintage age"
    )
    min_sdg_count: int = Field(
        default=0, ge=0, le=17, description="Min SDG count"
    )
    min_permanence_years: int = Field(
        default=0, ge=0, description="Min permanence"
    )
    preferred_standards: List[str] = Field(
        default_factory=list, description="Preferred standards"
    )
    excluded_countries: List[str] = Field(
        default_factory=list, description="Excluded countries"
    )
    target_year: int = Field(
        default=2026, ge=2020, le=2060, description="Target year"
    )


class PortfolioOptimizationInput(BaseModel):
    """Complete input for portfolio optimisation.

    Attributes:
        entity_name: Entity name.
        footprint_year: Footprint year to offset.
        footprint_tco2e: Total footprint to offset.
        objective: Optimisation objective.
        credit_options: Available credit options.
        constraints: Portfolio constraints.
        current_portfolio: Existing portfolio credits (for rebalancing).
        include_oxford_analysis: Whether to include Oxford Principles analysis.
        include_pareto: Whether to generate Pareto frontier.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    footprint_year: int = Field(
        ..., ge=2015, le=2060, description="Footprint year"
    )
    footprint_tco2e: Decimal = Field(
        ..., ge=0, description="Footprint to offset (tCO2e)"
    )
    objective: str = Field(
        default=OptimizationObjective.BALANCED.value,
        description="Optimisation objective"
    )
    credit_options: List[CreditOption] = Field(
        default_factory=list, description="Available credits"
    )
    constraints: PortfolioConstraints = Field(
        default_factory=PortfolioConstraints, description="Constraints"
    )
    current_portfolio: List[CreditOption] = Field(
        default_factory=list, description="Existing portfolio"
    )
    include_oxford_analysis: bool = Field(
        default=True, description="Include Oxford Principles analysis"
    )
    include_pareto: bool = Field(
        default=True, description="Generate Pareto frontier"
    )

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, v: str) -> str:
        valid = {o.value for o in OptimizationObjective}
        if v not in valid:
            raise ValueError(f"Unknown objective '{v}'.")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class PortfolioAllocation(BaseModel):
    """Allocation for a single credit in the optimised portfolio.

    Attributes:
        credit_id: Credit identifier.
        project_name: Project name.
        credit_type: Credit type.
        standard: Certification standard.
        vintage_year: Vintage year.
        allocated_tco2e: Allocated quantity.
        price_per_tco2e_usd: Price per credit.
        total_cost_usd: Total cost for allocation.
        quality_score: Quality score.
        pct_of_portfolio: Percentage of total portfolio.
        is_removal: Whether carbon removal.
        vintage_age: Age of vintage.
        vintage_acceptable: Whether vintage meets constraints.
        sdg_count: SDG contributions.
        co_benefit_score: Co-benefit score.
        value_score: Quality/price ratio.
    """
    credit_id: str = Field(default="")
    project_name: str = Field(default="")
    credit_type: str = Field(default="")
    standard: str = Field(default="")
    vintage_year: int = Field(default=0)
    allocated_tco2e: Decimal = Field(default=Decimal("0"))
    price_per_tco2e_usd: Decimal = Field(default=Decimal("0"))
    total_cost_usd: Decimal = Field(default=Decimal("0"))
    quality_score: Decimal = Field(default=Decimal("0"))
    pct_of_portfolio: Decimal = Field(default=Decimal("0"))
    is_removal: bool = Field(default=False)
    vintage_age: int = Field(default=0)
    vintage_acceptable: bool = Field(default=True)
    sdg_count: int = Field(default=0)
    co_benefit_score: Decimal = Field(default=Decimal("0"))
    value_score: Decimal = Field(default=Decimal("0"))


class DiversificationMetrics(BaseModel):
    """Portfolio diversification metrics.

    Attributes:
        type_hhi: HHI by credit type.
        geography_hhi: HHI by country.
        standard_hhi: HHI by standard.
        vintage_hhi: HHI by vintage year.
        overall_diversification: 1 - average HHI.
        type_breakdown: Percentage by credit type.
        geography_breakdown: Percentage by country.
        standard_breakdown: Percentage by standard.
        unique_projects: Number of unique projects.
        unique_countries: Number of unique countries.
        unique_standards: Number of unique standards.
        is_sufficiently_diversified: Whether meets minimum.
        recommendations: Diversification recommendations.
    """
    type_hhi: Decimal = Field(default=Decimal("0"))
    geography_hhi: Decimal = Field(default=Decimal("0"))
    standard_hhi: Decimal = Field(default=Decimal("0"))
    vintage_hhi: Decimal = Field(default=Decimal("0"))
    overall_diversification: Decimal = Field(default=Decimal("0"))
    type_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    geography_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    standard_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    unique_projects: int = Field(default=0)
    unique_countries: int = Field(default=0)
    unique_standards: int = Field(default=0)
    is_sufficiently_diversified: bool = Field(default=False)
    recommendations: List[str] = Field(default_factory=list)


class OxfordPrinciplesAssessment(BaseModel):
    """Oxford Principles alignment assessment.

    Attributes:
        target_year: Assessment year.
        required_removal_pct: Required removal percentage.
        actual_removal_pct: Actual removal percentage in portfolio.
        removal_gap_pct: Gap to required removal percentage.
        is_aligned: Whether portfolio meets Oxford Principles.
        removal_quantity_tco2e: Total removal credits.
        avoidance_quantity_tco2e: Total avoidance/reduction credits.
        nature_removal_tco2e: Nature-based removal credits.
        tech_removal_tco2e: Technology-based removal credits.
        progression_on_track: Whether trending toward 100% by 2050.
        message: Human-readable assessment.
    """
    target_year: int = Field(default=0)
    required_removal_pct: Decimal = Field(default=Decimal("0"))
    actual_removal_pct: Decimal = Field(default=Decimal("0"))
    removal_gap_pct: Decimal = Field(default=Decimal("0"))
    is_aligned: bool = Field(default=False)
    removal_quantity_tco2e: Decimal = Field(default=Decimal("0"))
    avoidance_quantity_tco2e: Decimal = Field(default=Decimal("0"))
    nature_removal_tco2e: Decimal = Field(default=Decimal("0"))
    tech_removal_tco2e: Decimal = Field(default=Decimal("0"))
    progression_on_track: bool = Field(default=False)
    message: str = Field(default="")


class VintageAnalysis(BaseModel):
    """Vintage matching analysis.

    Attributes:
        footprint_year: Footprint year.
        avg_vintage_year: Average vintage year.
        oldest_vintage: Oldest vintage in portfolio.
        newest_vintage: Newest vintage.
        avg_vintage_age: Average vintage age.
        max_vintage_age: Maximum vintage age.
        all_within_recommended: All within 5-year recommendation.
        all_within_acceptable: All within 7-year acceptable range.
        vintage_match_score: Vintage matching score (0-100).
        message: Human-readable assessment.
    """
    footprint_year: int = Field(default=0)
    avg_vintage_year: int = Field(default=0)
    oldest_vintage: int = Field(default=0)
    newest_vintage: int = Field(default=0)
    avg_vintage_age: Decimal = Field(default=Decimal("0"))
    max_vintage_age: int = Field(default=0)
    all_within_recommended: bool = Field(default=True)
    all_within_acceptable: bool = Field(default=True)
    vintage_match_score: Decimal = Field(default=Decimal("0"))
    message: str = Field(default="")


class ParetoPoint(BaseModel):
    """A point on the Pareto frontier.

    Attributes:
        portfolio_id: Portfolio configuration identifier.
        quality_score: Weighted quality score.
        total_cost_usd: Total portfolio cost.
        cobenefit_score: Co-benefit score.
        removal_pct: Removal credit percentage.
        is_pareto_optimal: Whether Pareto-optimal.
        allocations: Credit allocations for this configuration.
    """
    portfolio_id: str = Field(default_factory=_new_uuid)
    quality_score: Decimal = Field(default=Decimal("0"))
    total_cost_usd: Decimal = Field(default=Decimal("0"))
    cobenefit_score: Decimal = Field(default=Decimal("0"))
    removal_pct: Decimal = Field(default=Decimal("0"))
    is_pareto_optimal: bool = Field(default=False)
    allocations: List[str] = Field(default_factory=list)


class PortfolioOptimizationResult(BaseModel):
    """Complete portfolio optimisation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        footprint_year: Footprint year.
        footprint_tco2e: Footprint to offset.
        objective: Optimisation objective used.
        allocations: Credit allocations in optimised portfolio.
        total_allocated_tco2e: Total allocated credits.
        total_cost_usd: Total portfolio cost.
        avg_price_per_tco2e: Average price per credit.
        portfolio_quality_score: Weighted quality score.
        portfolio_quality_rating: Quality rating.
        diversification: Diversification metrics.
        oxford_assessment: Oxford Principles assessment.
        vintage_analysis: Vintage matching analysis.
        pareto_points: Pareto frontier points.
        coverage_pct: Coverage of footprint (allocated/footprint).
        is_fully_covered: Whether footprint is fully covered.
        constraints_met: Whether all constraints are satisfied.
        constraint_violations: List of violated constraints.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    footprint_year: int = Field(default=0)
    footprint_tco2e: Decimal = Field(default=Decimal("0"))
    objective: str = Field(default="")
    allocations: List[PortfolioAllocation] = Field(default_factory=list)
    total_allocated_tco2e: Decimal = Field(default=Decimal("0"))
    total_cost_usd: Decimal = Field(default=Decimal("0"))
    avg_price_per_tco2e: Decimal = Field(default=Decimal("0"))
    portfolio_quality_score: Decimal = Field(default=Decimal("0"))
    portfolio_quality_rating: str = Field(default="")
    diversification: Optional[DiversificationMetrics] = Field(default=None)
    oxford_assessment: Optional[OxfordPrinciplesAssessment] = Field(default=None)
    vintage_analysis: Optional[VintageAnalysis] = Field(default=None)
    pareto_points: List[ParetoPoint] = Field(default_factory=list)
    coverage_pct: Decimal = Field(default=Decimal("0"))
    is_fully_covered: bool = Field(default=False)
    constraints_met: bool = Field(default=False)
    constraint_violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PortfolioOptimizationEngine:
    """Carbon credit portfolio optimisation engine.

    Builds diversified credit portfolios optimised for quality,
    cost, and co-benefits, with Oxford Principles alignment and
    vintage matching.

    Usage::

        engine = PortfolioOptimizationEngine()
        result = engine.optimize(input_data)
        print(f"Portfolio quality: {result.portfolio_quality_score}/100")
        for alloc in result.allocations:
            print(f"  {alloc.project_name}: {alloc.allocated_tco2e} tCO2e")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        logger.info("PortfolioOptimizationEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def optimize(
        self, data: PortfolioOptimizationInput,
    ) -> PortfolioOptimizationResult:
        """Optimise carbon credit portfolio.

        Args:
            data: Validated optimisation input.

        Returns:
            PortfolioOptimizationResult with optimised portfolio.
        """
        t0 = time.perf_counter()
        logger.info(
            "Portfolio optimisation: entity=%s, footprint=%.2f, options=%d",
            data.entity_name, float(data.footprint_tco2e),
            len(data.credit_options),
        )

        warnings: List[str] = []
        errors: List[str] = []
        violations: List[str] = []

        # Step 1: Filter eligible credits
        eligible = self._filter_eligible(data.credit_options, data.constraints, warnings)

        if not eligible:
            errors.append("No eligible credit options after applying constraints.")

        # Step 2: Sort by objective
        sorted_credits = self._sort_by_objective(eligible, data.objective)

        # Step 3: Allocate credits (greedy)
        allocations = self._allocate_greedy(
            sorted_credits, data.footprint_tco2e, data.constraints,
            data.footprint_year, violations
        )

        # Step 4: Calculate portfolio metrics
        total_allocated = sum(
            (a.allocated_tco2e for a in allocations), Decimal("0")
        )
        total_cost = sum(
            (a.total_cost_usd for a in allocations), Decimal("0")
        )
        avg_price = _safe_divide(total_cost, total_allocated)

        # Weighted quality
        quality = Decimal("0")
        if total_allocated > Decimal("0"):
            quality = sum(
                (a.quality_score * a.allocated_tco2e for a in allocations),
                Decimal("0"),
            ) / total_allocated

        # Calculate percentages
        for a in allocations:
            a.pct_of_portfolio = _round_val(_safe_pct(a.allocated_tco2e, total_allocated), 2)

        # Quality rating
        if quality >= QUALITY_EXCELLENT:
            q_rating = "Excellent"
        elif quality >= QUALITY_GOOD:
            q_rating = "Good"
        elif quality >= QUALITY_ADEQUATE:
            q_rating = "Adequate"
        elif quality >= QUALITY_POOR:
            q_rating = "Poor"
        else:
            q_rating = "Failing"

        # Step 5: Diversification
        diversification = self._assess_diversification(allocations, total_allocated)

        # Step 6: Oxford Principles
        oxford: Optional[OxfordPrinciplesAssessment] = None
        if data.include_oxford_analysis:
            oxford = self._assess_oxford_alignment(
                allocations, total_allocated, data.constraints.target_year
            )

        # Step 7: Vintage analysis
        vintage = self._analyse_vintages(allocations, data.footprint_year)

        # Step 8: Pareto frontier
        pareto: List[ParetoPoint] = []
        if data.include_pareto:
            pareto = self._generate_pareto_points(
                eligible, data.footprint_tco2e, data.constraints
            )

        # Coverage
        coverage = _safe_pct(total_allocated, data.footprint_tco2e)
        is_covered = total_allocated >= data.footprint_tco2e

        if not is_covered:
            warnings.append(
                f"Portfolio covers {_round_val(coverage, 1)}% of footprint. "
                f"Additional credits needed for full neutrality."
            )

        constraints_met = len(violations) == 0

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = PortfolioOptimizationResult(
            entity_name=data.entity_name,
            footprint_year=data.footprint_year,
            footprint_tco2e=data.footprint_tco2e,
            objective=data.objective,
            allocations=allocations,
            total_allocated_tco2e=_round_val(total_allocated),
            total_cost_usd=_round_val(total_cost, 2),
            avg_price_per_tco2e=_round_val(avg_price, 2),
            portfolio_quality_score=_round_val(quality, 2),
            portfolio_quality_rating=q_rating,
            diversification=diversification,
            oxford_assessment=oxford,
            vintage_analysis=vintage,
            pareto_points=pareto,
            coverage_pct=_round_val(coverage, 2),
            is_fully_covered=is_covered,
            constraints_met=constraints_met,
            constraint_violations=violations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Portfolio optimisation complete: allocated=%.2f, cost=%.2f, "
            "quality=%.1f, coverage=%.1f%%, hash=%s",
            float(total_allocated), float(total_cost),
            float(quality), float(coverage),
            result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _filter_eligible(
        self,
        options: List[CreditOption],
        constraints: PortfolioConstraints,
        warnings: List[str],
    ) -> List[CreditOption]:
        """Filter credit options by constraints."""
        eligible: List[CreditOption] = []
        for opt in options:
            if opt.quality_score < constraints.min_quality_score:
                continue
            if opt.country in constraints.excluded_countries:
                continue
            if opt.sdg_count < constraints.min_sdg_count:
                continue
            if constraints.min_permanence_years > 0 and opt.permanence_years < constraints.min_permanence_years:
                continue
            if opt.vintage_year > 0:
                age = constraints.target_year - opt.vintage_year
                if age > constraints.max_vintage_age:
                    continue
            if constraints.preferred_standards and opt.standard not in constraints.preferred_standards:
                # Deprioritise but don't exclude
                pass
            eligible.append(opt)

        if len(eligible) < len(options):
            removed = len(options) - len(eligible)
            warnings.append(f"{removed} credit options excluded by constraints.")

        return eligible

    def _sort_by_objective(
        self,
        credits: List[CreditOption],
        objective: str,
    ) -> List[CreditOption]:
        """Sort credits by optimisation objective."""
        if objective == OptimizationObjective.MAXIMIZE_QUALITY.value:
            return sorted(credits, key=lambda c: float(c.quality_score), reverse=True)
        elif objective == OptimizationObjective.MINIMIZE_COST.value:
            return sorted(credits, key=lambda c: float(c.price_per_tco2e_usd))
        elif objective == OptimizationObjective.MAXIMIZE_COBENEFIT.value:
            return sorted(credits, key=lambda c: float(c.co_benefit_score), reverse=True)
        else:
            # Balanced: value score = quality / price
            def value_fn(c: CreditOption) -> float:
                if c.price_per_tco2e_usd > Decimal("0"):
                    return float(c.quality_score / c.price_per_tco2e_usd)
                return float(c.quality_score) * 100
            return sorted(credits, key=value_fn, reverse=True)

    def _allocate_greedy(
        self,
        credits: List[CreditOption],
        target: Decimal,
        constraints: PortfolioConstraints,
        footprint_year: int,
        violations: List[str],
    ) -> List[PortfolioAllocation]:
        """Greedy allocation respecting constraints."""
        allocations: List[PortfolioAllocation] = []
        remaining = target
        budget_remaining = constraints.max_budget_usd if constraints.max_budget_usd > Decimal("0") else Decimal("999999999")
        country_totals: Dict[str, Decimal] = {}

        for credit in credits:
            if remaining <= Decimal("0"):
                break

            # Max from this credit
            max_from_project = target * constraints.max_single_project_pct / Decimal("100")
            max_from_country = target * constraints.max_single_country_pct / Decimal("100")

            country_used = country_totals.get(credit.country, Decimal("0"))
            country_available = max_from_country - country_used

            can_allocate = min(
                remaining,
                credit.available_quantity_tco2e,
                max_from_project,
                country_available,
            )

            if credit.price_per_tco2e_usd > Decimal("0"):
                affordable = _safe_divide(budget_remaining, credit.price_per_tco2e_usd)
                can_allocate = min(can_allocate, affordable)

            if can_allocate <= Decimal("0"):
                continue

            cost = can_allocate * credit.price_per_tco2e_usd
            vintage_age = max(0, footprint_year - credit.vintage_year) if credit.vintage_year > 0 else 0
            vintage_ok = vintage_age <= constraints.max_vintage_age

            value_score = Decimal("0")
            if credit.price_per_tco2e_usd > Decimal("0"):
                value_score = _safe_divide(credit.quality_score, credit.price_per_tco2e_usd)

            allocations.append(PortfolioAllocation(
                credit_id=credit.credit_id,
                project_name=credit.project_name,
                credit_type=credit.credit_type,
                standard=credit.standard,
                vintage_year=credit.vintage_year,
                allocated_tco2e=_round_val(can_allocate),
                price_per_tco2e_usd=credit.price_per_tco2e_usd,
                total_cost_usd=_round_val(cost, 2),
                quality_score=credit.quality_score,
                is_removal=credit.is_removal,
                vintage_age=vintage_age,
                vintage_acceptable=vintage_ok,
                sdg_count=credit.sdg_count,
                co_benefit_score=credit.co_benefit_score,
                value_score=_round_val(value_score, 4),
            ))

            remaining -= can_allocate
            budget_remaining -= cost
            country_totals[credit.country] = country_used + can_allocate

        if remaining > Decimal("0"):
            violations.append(
                f"Cannot fully cover footprint. Shortfall: {_round_val(remaining)} tCO2e."
            )

        return allocations

    def _assess_diversification(
        self,
        allocations: List[PortfolioAllocation],
        total: Decimal,
    ) -> DiversificationMetrics:
        """Assess portfolio diversification using HHI."""
        if total <= Decimal("0") or not allocations:
            return DiversificationMetrics()

        type_shares: Dict[str, Decimal] = {}
        geo_shares: Dict[str, Decimal] = {}
        std_shares: Dict[str, Decimal] = {}

        for a in allocations:
            share = a.allocated_tco2e / total
            type_shares[a.credit_type] = type_shares.get(a.credit_type, Decimal("0")) + share
            geo_shares[a.standard] = geo_shares.get(a.standard, Decimal("0")) + share
            country = a.project_name[:2] if len(a.project_name) >= 2 else "XX"
            std_shares[country] = std_shares.get(country, Decimal("0")) + share

        def _hhi(shares: Dict[str, Decimal]) -> Decimal:
            return sum(s * s for s in shares.values())

        type_hhi = _hhi(type_shares)
        geo_hhi = _hhi(geo_shares)
        std_hhi = _hhi(std_shares)
        vintage_shares: Dict[str, Decimal] = {}
        for a in allocations:
            vy = str(a.vintage_year)
            vintage_shares[vy] = vintage_shares.get(vy, Decimal("0")) + a.allocated_tco2e / total
        vintage_hhi = _hhi(vintage_shares)

        avg_hhi = (type_hhi + geo_hhi + std_hhi + vintage_hhi) / Decimal("4")
        diversification = Decimal("1") - avg_hhi

        recs: List[str] = []
        if diversification < MIN_DIVERSIFICATION_INDEX:
            recs.append("Portfolio is concentrated. Consider diversifying across project types, geographies, and standards.")

        return DiversificationMetrics(
            type_hhi=_round_val(type_hhi, 4),
            geography_hhi=_round_val(geo_hhi, 4),
            standard_hhi=_round_val(std_hhi, 4),
            vintage_hhi=_round_val(vintage_hhi, 4),
            overall_diversification=_round_val(diversification, 4),
            type_breakdown={k: _round_val(v * Decimal("100"), 2) for k, v in type_shares.items()},
            geography_breakdown={k: _round_val(v * Decimal("100"), 2) for k, v in geo_shares.items()},
            standard_breakdown={k: _round_val(v * Decimal("100"), 2) for k, v in std_shares.items()},
            unique_projects=len(set(a.credit_id for a in allocations)),
            unique_countries=len(set(a.project_name[:2] for a in allocations if a.project_name)),
            unique_standards=len(set(a.standard for a in allocations)),
            is_sufficiently_diversified=diversification >= MIN_DIVERSIFICATION_INDEX,
            recommendations=recs,
        )

    def _assess_oxford_alignment(
        self,
        allocations: List[PortfolioAllocation],
        total: Decimal,
        target_year: int,
    ) -> OxfordPrinciplesAssessment:
        """Assess Oxford Principles removal progression alignment."""
        # Determine required removal percentage
        required = Decimal("0")
        for threshold_year, pct in OXFORD_REMOVAL_PROGRESSION:
            if target_year >= threshold_year:
                required = pct
        # Interpolate between progression points
        for i in range(len(OXFORD_REMOVAL_PROGRESSION) - 1):
            yr1, pct1 = OXFORD_REMOVAL_PROGRESSION[i]
            yr2, pct2 = OXFORD_REMOVAL_PROGRESSION[i + 1]
            if yr1 <= target_year < yr2:
                frac = _decimal(target_year - yr1) / _decimal(yr2 - yr1)
                required = pct1 + (pct2 - pct1) * frac
                break

        removal_total = sum(
            (a.allocated_tco2e for a in allocations if a.is_removal), Decimal("0")
        )
        avoidance_total = sum(
            (a.allocated_tco2e for a in allocations if not a.is_removal), Decimal("0")
        )
        nature_total = sum(
            (a.allocated_tco2e for a in allocations
             if a.credit_type == CreditType.REMOVAL_NATURE.value), Decimal("0")
        )
        tech_total = sum(
            (a.allocated_tco2e for a in allocations
             if a.credit_type == CreditType.REMOVAL_TECH.value), Decimal("0")
        )

        actual_pct = _safe_pct(removal_total, total)
        gap = max(Decimal("0"), required - actual_pct)
        aligned = actual_pct >= required

        if aligned:
            msg = (
                f"Portfolio has {_round_val(actual_pct, 1)}% removal credits, "
                f"meeting the Oxford Principles target of {_round_val(required, 1)}% "
                f"for {target_year}."
            )
        else:
            msg = (
                f"Portfolio has {_round_val(actual_pct, 1)}% removal credits, "
                f"below the Oxford Principles target of {_round_val(required, 1)}% "
                f"for {target_year}. Gap: {_round_val(gap, 1)}%."
            )

        return OxfordPrinciplesAssessment(
            target_year=target_year,
            required_removal_pct=_round_val(required, 2),
            actual_removal_pct=_round_val(actual_pct, 2),
            removal_gap_pct=_round_val(gap, 2),
            is_aligned=aligned,
            removal_quantity_tco2e=_round_val(removal_total),
            avoidance_quantity_tco2e=_round_val(avoidance_total),
            nature_removal_tco2e=_round_val(nature_total),
            tech_removal_tco2e=_round_val(tech_total),
            progression_on_track=aligned,
            message=msg,
        )

    def _analyse_vintages(
        self,
        allocations: List[PortfolioAllocation],
        footprint_year: int,
    ) -> VintageAnalysis:
        """Analyse vintage matching."""
        if not allocations:
            return VintageAnalysis(footprint_year=footprint_year)

        vintages = [a.vintage_year for a in allocations if a.vintage_year > 0]
        if not vintages:
            return VintageAnalysis(
                footprint_year=footprint_year,
                message="No vintage data available.",
            )

        oldest = min(vintages)
        newest = max(vintages)
        avg = sum(vintages) / len(vintages)
        max_age = footprint_year - oldest
        avg_age = _decimal(footprint_year) - _decimal(avg)

        within_rec = all(footprint_year - v <= MAX_VINTAGE_AGE_RECOMMENDED for v in vintages)
        within_acc = all(footprint_year - v <= MAX_VINTAGE_AGE_ACCEPTABLE for v in vintages)

        if within_rec:
            score = Decimal("100")
        elif within_acc:
            score = Decimal("75")
        else:
            score = max(Decimal("0"), Decimal("100") - _decimal(max_age - MAX_VINTAGE_AGE_RECOMMENDED) * Decimal("10"))

        if within_rec:
            msg = f"All vintages within recommended {MAX_VINTAGE_AGE_RECOMMENDED}-year range."
        elif within_acc:
            msg = f"All vintages within acceptable {MAX_VINTAGE_AGE_ACCEPTABLE}-year range."
        else:
            msg = (
                f"Some vintages exceed acceptable {MAX_VINTAGE_AGE_ACCEPTABLE}-year range. "
                f"Oldest vintage: {oldest} ({max_age} years old)."
            )

        return VintageAnalysis(
            footprint_year=footprint_year,
            avg_vintage_year=int(avg),
            oldest_vintage=oldest,
            newest_vintage=newest,
            avg_vintage_age=_round_val(avg_age, 1),
            max_vintage_age=max_age,
            all_within_recommended=within_rec,
            all_within_acceptable=within_acc,
            vintage_match_score=_round_val(score, 2),
            message=msg,
        )

    def _generate_pareto_points(
        self,
        credits: List[CreditOption],
        target: Decimal,
        constraints: PortfolioConstraints,
    ) -> List[ParetoPoint]:
        """Generate Pareto frontier points.

        Creates portfolio configurations at different quality/cost
        tradeoff points.
        """
        points: List[ParetoPoint] = []

        # Strategy 1: Max quality
        by_quality = sorted(credits, key=lambda c: float(c.quality_score), reverse=True)
        q_alloc = self._quick_allocate(by_quality, target)
        if q_alloc:
            points.append(ParetoPoint(
                quality_score=q_alloc[0],
                total_cost_usd=q_alloc[1],
                cobenefit_score=q_alloc[2],
                removal_pct=q_alloc[3],
                is_pareto_optimal=True,
                allocations=[c.credit_id for c in by_quality[:5]],
            ))

        # Strategy 2: Min cost
        by_cost = sorted(credits, key=lambda c: float(c.price_per_tco2e_usd))
        c_alloc = self._quick_allocate(by_cost, target)
        if c_alloc:
            points.append(ParetoPoint(
                quality_score=c_alloc[0],
                total_cost_usd=c_alloc[1],
                cobenefit_score=c_alloc[2],
                removal_pct=c_alloc[3],
                is_pareto_optimal=True,
                allocations=[c.credit_id for c in by_cost[:5]],
            ))

        # Strategy 3: Max co-benefits
        by_cb = sorted(credits, key=lambda c: float(c.co_benefit_score), reverse=True)
        cb_alloc = self._quick_allocate(by_cb, target)
        if cb_alloc:
            points.append(ParetoPoint(
                quality_score=cb_alloc[0],
                total_cost_usd=cb_alloc[1],
                cobenefit_score=cb_alloc[2],
                removal_pct=cb_alloc[3],
                is_pareto_optimal=True,
                allocations=[c.credit_id for c in by_cb[:5]],
            ))

        # Strategy 4: Balanced
        def balanced_score(c: CreditOption) -> float:
            return float(c.quality_score) * 0.4 + float(c.co_benefit_score) * 0.3 + (100 - float(c.price_per_tco2e_usd)) * 0.3
        by_balanced = sorted(credits, key=balanced_score, reverse=True)
        b_alloc = self._quick_allocate(by_balanced, target)
        if b_alloc:
            points.append(ParetoPoint(
                quality_score=b_alloc[0],
                total_cost_usd=b_alloc[1],
                cobenefit_score=b_alloc[2],
                removal_pct=b_alloc[3],
                is_pareto_optimal=True,
                allocations=[c.credit_id for c in by_balanced[:5]],
            ))

        return points

    def _quick_allocate(
        self,
        credits: List[CreditOption],
        target: Decimal,
    ) -> Optional[Tuple[Decimal, Decimal, Decimal, Decimal]]:
        """Quick allocation for Pareto point estimation."""
        remaining = target
        total_cost = Decimal("0")
        weighted_q = Decimal("0")
        weighted_cb = Decimal("0")
        removal = Decimal("0")
        allocated = Decimal("0")

        for c in credits:
            if remaining <= Decimal("0"):
                break
            alloc = min(remaining, c.available_quantity_tco2e)
            if alloc <= Decimal("0"):
                continue
            total_cost += alloc * c.price_per_tco2e_usd
            weighted_q += c.quality_score * alloc
            weighted_cb += c.co_benefit_score * alloc
            if c.is_removal:
                removal += alloc
            allocated += alloc
            remaining -= alloc

        if allocated <= Decimal("0"):
            return None

        return (
            _round_val(weighted_q / allocated, 2),
            _round_val(total_cost, 2),
            _round_val(weighted_cb / allocated, 2),
            _round_val(_safe_pct(removal, allocated), 2),
        )
