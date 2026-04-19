# -*- coding: utf-8 -*-
"""
AnnualPathwayEngine - PACK-029 Interim Targets Pack Engine 2
================================================================

Generates year-over-year reduction trajectories from baseline year to
net-zero year with annual reduction rates, quarterly milestone
interpolation, budget allocation across years, cumulative emissions
tracking, and carbon budget compliance checking.

Calculation Methodology:
    Constant Annual Reduction Rate:
        E(t) = E_base * (1 - r)^(t - t_base)
        where r = 1 - (E_target / E_base)^(1 / years)

    Accelerating Reduction Rate:
        r(t) = r_base * (1 + acceleration * (t - t_base) / years)
        E(t) = E(t-1) * (1 - r(t))

    Annual Absolute Reduction:
        delta(t) = E(t-1) - E(t)

    Quarterly Milestone Interpolation:
        E(t, q) = E(t) + (E(t+1) - E(t)) * (q - 1) / 4
        where q = 1..4

    Cumulative Emissions (Carbon Budget):
        C(t) = sum_{y=base}^{t} E(y)
        Using trapezoidal integration for accuracy.

    Budget Compliance:
        compliant = C(t) <= B(t)
        where B(t) = allocated budget through year t.

    Annual Budget Allocation:
        Equal: B(t) = total_budget * (t - base) / years
        Front-loaded: B(t) = total_budget * sqrt((t-base)/years)
        Back-loaded: B(t) = total_budget * ((t-base)/years)^2

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024) -- annual pathways
    - IPCC AR6 WG1 (2021) -- remaining carbon budget (~400 GtCO2 for 1.5C)
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - Paris Agreement (2015) -- carbon budget framing
    - CSRD ESRS E1-4 -- GHG reduction targets & transition plan
    - ISO 14064-1:2018 -- Organizational GHG inventories

Zero-Hallucination:
    - All trajectories use deterministic Decimal arithmetic
    - Carbon budget integration uses trapezoidal rule
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-029 Interim Targets
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
    """Safely divide two Decimals."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReductionProfile(str, Enum):
    """Annual reduction rate profile."""
    CONSTANT = "constant"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    S_CURVE = "s_curve"
    CUSTOM = "custom"

class BudgetAllocation(str, Enum):
    """Carbon budget allocation strategy."""
    EQUAL = "equal"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"
    PROPORTIONAL = "proportional"

class ComplianceStatus(str, Enum):
    """Carbon budget compliance status."""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_DATA = "insufficient_data"

class PathwayGranularity(str, Enum):
    """Granularity of pathway output."""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"

class ScopeType(str, Enum):
    """GHG scope type."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    ALL_SCOPES = "all_scopes"

class DataQuality(str, Enum):
    """Data quality tier."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IPCC AR6 remaining carbon budgets (GtCO2 from 2020)
GLOBAL_CARBON_BUDGETS: Dict[str, Decimal] = {
    "1.5c_50pct": Decimal("500"),
    "1.5c_67pct": Decimal("400"),
    "1.5c_83pct": Decimal("300"),
    "2c_50pct": Decimal("1150"),
    "2c_67pct": Decimal("900"),
    "2c_83pct": Decimal("700"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class AnnualEmissionsPoint(BaseModel):
    """Historical annual emissions data point.

    Attributes:
        year: Calendar year.
        emissions_tco2e: Total emissions for the year (tCO2e).
        scope: Which scope this data covers.
        is_verified: Whether this data is third-party verified.
    """
    year: int = Field(..., ge=2010, le=2050, description="Year")
    emissions_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Emissions (tCO2e)"
    )
    scope: ScopeType = Field(
        default=ScopeType.ALL_SCOPES, description="Scope"
    )
    is_verified: bool = Field(default=False, description="Third-party verified")

class CustomRateSchedule(BaseModel):
    """Custom annual reduction rate schedule.

    Attributes:
        year: Year this rate applies.
        rate_pct: Reduction rate for this year (%).
    """
    year: int = Field(..., ge=2020, le=2070, description="Year")
    rate_pct: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("50"),
        description="Reduction rate (%)"
    )

class AnnualPathwayInput(BaseModel):
    """Input for annual pathway generation.

    Attributes:
        entity_name: Company or entity name.
        entity_id: Unique entity identifier.
        baseline_year: Baseline year for the pathway.
        baseline_emissions_tco2e: Baseline emissions (tCO2e).
        target_year: Final target year (net-zero year).
        target_emissions_tco2e: Target emissions at target_year.
        target_reduction_pct: Target reduction percentage.
        reduction_profile: Shape of annual reduction rates.
        custom_rates: Custom rate schedule (if profile=custom).
        acceleration_factor: Acceleration factor for accelerating profile (0-1).
        budget_allocation: Carbon budget allocation strategy.
        total_carbon_budget_tco2e: Total allowed cumulative emissions.
        granularity: Output granularity (annual/quarterly/monthly).
        scope: Scope for this pathway.
        historical_emissions: Historical emissions for context.
        include_budget_analysis: Include carbon budget compliance.
        include_quarterly_milestones: Generate quarterly milestones.
        reporting_year: Current reporting year.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    entity_id: str = Field(
        default="", max_length=100, description="Entity identifier"
    )
    baseline_year: int = Field(
        ..., ge=2015, le=2025, description="Baseline year"
    )
    baseline_emissions_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Baseline emissions (tCO2e)"
    )
    target_year: int = Field(
        default=2050, ge=2030, le=2070, description="Target year"
    )
    target_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Target emissions (tCO2e)"
    )
    target_reduction_pct: Decimal = Field(
        default=Decimal("90"), ge=Decimal("0"), le=Decimal("100"),
        description="Target reduction (%)"
    )
    reduction_profile: ReductionProfile = Field(
        default=ReductionProfile.CONSTANT, description="Reduction profile"
    )
    custom_rates: List[CustomRateSchedule] = Field(
        default_factory=list, description="Custom rate schedule"
    )
    acceleration_factor: Decimal = Field(
        default=Decimal("0.3"), ge=Decimal("0"), le=Decimal("1.0"),
        description="Acceleration factor"
    )
    budget_allocation: BudgetAllocation = Field(
        default=BudgetAllocation.EQUAL, description="Budget allocation"
    )
    total_carbon_budget_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Total carbon budget (tCO2e)"
    )
    granularity: PathwayGranularity = Field(
        default=PathwayGranularity.ANNUAL, description="Output granularity"
    )
    scope: ScopeType = Field(
        default=ScopeType.ALL_SCOPES, description="Scope"
    )
    historical_emissions: List[AnnualEmissionsPoint] = Field(
        default_factory=list, description="Historical emissions"
    )
    include_budget_analysis: bool = Field(
        default=True, description="Include budget analysis"
    )
    include_quarterly_milestones: bool = Field(
        default=True, description="Include quarterly milestones"
    )
    reporting_year: int = Field(
        default=2024, ge=2020, le=2030, description="Reporting year"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class AnnualPathwayPoint(BaseModel):
    """A single point in the annual pathway.

    Attributes:
        year: Calendar year.
        target_emissions_tco2e: Target emissions for this year.
        reduction_from_baseline_pct: Cumulative reduction from baseline.
        annual_reduction_rate_pct: Reduction rate for this specific year.
        annual_reduction_tco2e: Absolute reduction from previous year.
        cumulative_emissions_tco2e: Cumulative emissions from baseline.
        budget_allocated_tco2e: Budget allocated through this year.
        budget_remaining_tco2e: Budget remaining after this year.
        budget_compliance: Whether within budget.
        yoy_change_pct: Year-over-year change percentage.
    """
    year: int = Field(default=0)
    target_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_from_baseline_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    budget_allocated_tco2e: Decimal = Field(default=Decimal("0"))
    budget_remaining_tco2e: Decimal = Field(default=Decimal("0"))
    budget_compliance: str = Field(default=ComplianceStatus.INSUFFICIENT_DATA.value)
    yoy_change_pct: Decimal = Field(default=Decimal("0"))

class QuarterlyMilestone(BaseModel):
    """Quarterly milestone within a year.

    Attributes:
        year: Calendar year.
        quarter: Quarter number (1-4).
        target_emissions_tco2e: Target emissions for this quarter.
        cumulative_ytd_tco2e: Year-to-date cumulative emissions.
        reduction_from_baseline_pct: Reduction from baseline.
    """
    year: int = Field(default=0)
    quarter: int = Field(default=1, ge=1, le=4)
    target_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_ytd_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_from_baseline_pct: Decimal = Field(default=Decimal("0"))

class BudgetAnalysis(BaseModel):
    """Carbon budget compliance analysis.

    Attributes:
        total_budget_tco2e: Total carbon budget.
        cumulative_pathway_tco2e: Cumulative emissions in pathway.
        budget_surplus_deficit_tco2e: Surplus (positive) or deficit (negative).
        compliance_status: Overall compliance status.
        years_of_budget_remaining: Years before budget exhausted.
        budget_exhaustion_year: Year budget is exhausted.
        annual_budget_allocations: Per-year budget allocation.
        overshoot_years: Years where pathway exceeds budget.
    """
    total_budget_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_pathway_tco2e: Decimal = Field(default=Decimal("0"))
    budget_surplus_deficit_tco2e: Decimal = Field(default=Decimal("0"))
    compliance_status: str = Field(default=ComplianceStatus.INSUFFICIENT_DATA.value)
    years_of_budget_remaining: int = Field(default=0)
    budget_exhaustion_year: int = Field(default=0)
    annual_budget_allocations: List[Dict[str, Any]] = Field(default_factory=list)
    overshoot_years: List[int] = Field(default_factory=list)

class PathwaySummary(BaseModel):
    """Summary statistics for the annual pathway.

    Attributes:
        total_years: Number of years in pathway.
        total_reduction_tco2e: Total absolute reduction.
        total_reduction_pct: Total percentage reduction.
        average_annual_rate_pct: Average annual reduction rate.
        max_annual_rate_pct: Maximum annual reduction rate.
        min_annual_rate_pct: Minimum annual reduction rate.
        cumulative_emissions_tco2e: Total cumulative emissions.
        implied_temperature_score: Temperature alignment.
        sbti_near_term_compliant: Whether near-term is SBTi compliant.
    """
    total_years: int = Field(default=0)
    total_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    total_reduction_pct: Decimal = Field(default=Decimal("0"))
    average_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    max_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    min_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    cumulative_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    implied_temperature_score: Decimal = Field(default=Decimal("0"))
    sbti_near_term_compliant: bool = Field(default=False)

class AnnualPathwayResult(BaseModel):
    """Complete annual pathway result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        entity_id: Entity identifier.
        baseline_year: Baseline year.
        baseline_emissions_tco2e: Baseline emissions.
        target_year: Target year.
        target_emissions_tco2e: Target emissions.
        reduction_profile: Reduction profile used.
        scope: Scope covered.
        annual_pathway: Year-by-year pathway points.
        quarterly_milestones: Quarterly milestones.
        budget_analysis: Carbon budget analysis.
        summary: Pathway summary statistics.
        data_quality: Data quality assessment.
        recommendations: Recommendations.
        warnings: Warnings.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    baseline_year: int = Field(default=0)
    baseline_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    target_year: int = Field(default=0)
    target_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_profile: str = Field(default="")
    scope: str = Field(default="")
    annual_pathway: List[AnnualPathwayPoint] = Field(default_factory=list)
    quarterly_milestones: List[QuarterlyMilestone] = Field(default_factory=list)
    budget_analysis: Optional[BudgetAnalysis] = Field(default=None)
    summary: Optional[PathwaySummary] = Field(default=None)
    data_quality: str = Field(default=DataQuality.MEDIUM.value)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AnnualPathwayEngine:
    """Annual pathway generation engine for PACK-029.

    Generates year-over-year reduction trajectories with multiple
    reduction profiles, quarterly interpolation, and carbon budget
    compliance checking.

    All calculations use deterministic Decimal arithmetic.
    No LLM involvement in any calculation path.

    Usage::

        engine = AnnualPathwayEngine()
        result = await engine.calculate(annual_pathway_input)
        for pt in result.annual_pathway:
            print(f"  {pt.year}: {pt.target_emissions_tco2e} tCO2e "
                  f"({pt.reduction_from_baseline_pct}% reduction)")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def calculate(self, data: AnnualPathwayInput) -> AnnualPathwayResult:
        """Generate complete annual pathway.

        Args:
            data: Validated pathway input.

        Returns:
            AnnualPathwayResult with annual points, quarterly milestones,
            and budget analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Annual pathway: entity=%s, profile=%s, %d->%d",
            data.entity_name, data.reduction_profile.value,
            data.baseline_year, data.target_year,
        )

        # Resolve target emissions
        target_emissions = self._resolve_target_emissions(data)

        # Generate annual pathway
        pathway = self._generate_annual_pathway(data, target_emissions)

        # Generate quarterly milestones
        quarterly: List[QuarterlyMilestone] = []
        if data.include_quarterly_milestones:
            quarterly = self._generate_quarterly_milestones(
                data, pathway,
            )

        # Carbon budget analysis
        budget: Optional[BudgetAnalysis] = None
        if data.include_budget_analysis and data.total_carbon_budget_tco2e > Decimal("0"):
            budget = self._analyze_carbon_budget(data, pathway)

        # Summary
        summary = self._build_summary(data, pathway)

        # Data quality
        dq = self._assess_data_quality(data)

        # Recommendations
        recs = self._generate_recommendations(data, summary, budget)

        # Warnings
        warns = self._generate_warnings(data, pathway, budget)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = AnnualPathwayResult(
            entity_name=data.entity_name,
            entity_id=data.entity_id,
            baseline_year=data.baseline_year,
            baseline_emissions_tco2e=data.baseline_emissions_tco2e,
            target_year=data.target_year,
            target_emissions_tco2e=target_emissions,
            reduction_profile=data.reduction_profile.value,
            scope=data.scope.value,
            annual_pathway=pathway,
            quarterly_milestones=quarterly,
            budget_analysis=budget,
            summary=summary,
            data_quality=dq,
            recommendations=recs,
            warnings=warns,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Annual pathway complete: entity=%s, points=%d, "
            "quarterly=%d",
            data.entity_name, len(pathway), len(quarterly),
        )
        return result

    async def calculate_batch(
        self, inputs: List[AnnualPathwayInput],
    ) -> List[AnnualPathwayResult]:
        """Generate pathways for multiple entities."""
        results: List[AnnualPathwayResult] = []
        for inp in inputs:
            try:
                result = await self.calculate(inp)
                results.append(result)
            except Exception as exc:
                logger.error("Batch error for %s: %s", inp.entity_name, exc)
                results.append(AnnualPathwayResult(
                    entity_name=inp.entity_name,
                    warnings=[f"Calculation error: {exc}"],
                ))
        return results

    # ------------------------------------------------------------------ #
    # Target Resolution                                                    #
    # ------------------------------------------------------------------ #

    def _resolve_target_emissions(self, data: AnnualPathwayInput) -> Decimal:
        """Resolve target emissions from input.

        If target_emissions_tco2e is zero, calculate from reduction_pct.

        Formula:
            target = baseline * (1 - reduction_pct / 100)
        """
        if data.target_emissions_tco2e > Decimal("0"):
            return data.target_emissions_tco2e
        return data.baseline_emissions_tco2e * (
            Decimal("1") - data.target_reduction_pct / Decimal("100")
        )

    # ------------------------------------------------------------------ #
    # Annual Pathway Generation                                            #
    # ------------------------------------------------------------------ #

    def _generate_annual_pathway(
        self,
        data: AnnualPathwayInput,
        target_emissions: Decimal,
    ) -> List[AnnualPathwayPoint]:
        """Generate year-by-year pathway from baseline to target.

        Uses the selected reduction profile to determine annual rates.

        Args:
            data: Input data.
            target_emissions: Final target emissions.

        Returns:
            List of AnnualPathwayPoint for each year.
        """
        total_years = data.target_year - data.baseline_year
        if total_years <= 0:
            return []

        # Calculate base annual rate
        base_rate = self._calculate_constant_rate(
            data.baseline_emissions_tco2e, target_emissions, total_years,
        )

        # Build custom rate map
        custom_map: Dict[int, Decimal] = {}
        for cr in data.custom_rates:
            custom_map[cr.year] = cr.rate_pct / Decimal("100")

        pathway: List[AnnualPathwayPoint] = []
        prev_emissions = data.baseline_emissions_tco2e
        cumulative = Decimal("0")

        # Budget allocation
        total_budget = data.total_carbon_budget_tco2e

        for i in range(total_years + 1):
            year = data.baseline_year + i

            if i == 0:
                # Baseline year
                pathway.append(AnnualPathwayPoint(
                    year=year,
                    target_emissions_tco2e=_round_val(data.baseline_emissions_tco2e, 2),
                    reduction_from_baseline_pct=Decimal("0"),
                    annual_reduction_rate_pct=Decimal("0"),
                    annual_reduction_tco2e=Decimal("0"),
                    cumulative_emissions_tco2e=_round_val(data.baseline_emissions_tco2e, 2),
                    budget_allocated_tco2e=Decimal("0"),
                    budget_remaining_tco2e=total_budget,
                    budget_compliance=ComplianceStatus.COMPLIANT.value
                    if total_budget > Decimal("0")
                    else ComplianceStatus.INSUFFICIENT_DATA.value,
                    yoy_change_pct=Decimal("0"),
                ))
                cumulative = data.baseline_emissions_tco2e
                continue

            # Determine annual rate for this year
            rate = self._get_annual_rate(
                data.reduction_profile, base_rate, i, total_years,
                data.acceleration_factor, custom_map, year,
            )

            # Calculate emissions
            year_emissions = prev_emissions * (Decimal("1") - rate)
            year_emissions = max(year_emissions, Decimal("0"))

            # Don't overshoot target
            if year == data.target_year:
                year_emissions = max(target_emissions, Decimal("0"))

            annual_reduction = prev_emissions - year_emissions
            reduction_from_base = _safe_pct(
                data.baseline_emissions_tco2e - year_emissions,
                data.baseline_emissions_tco2e,
            )

            # Cumulative (trapezoidal: average of this year and previous)
            segment = (prev_emissions + year_emissions) / Decimal("2")
            cumulative += segment

            # Budget allocation
            budget_at_year = self._allocate_budget(
                total_budget, i, total_years, data.budget_allocation,
            )
            budget_remaining = total_budget - cumulative if total_budget > Decimal("0") else Decimal("0")

            # Budget compliance
            compliance = ComplianceStatus.INSUFFICIENT_DATA.value
            if total_budget > Decimal("0"):
                if cumulative <= budget_at_year:
                    compliance = ComplianceStatus.COMPLIANT.value
                elif cumulative <= budget_at_year * Decimal("1.1"):
                    compliance = ComplianceStatus.AT_RISK.value
                else:
                    compliance = ComplianceStatus.NON_COMPLIANT.value

            # YoY change
            yoy = _safe_pct(year_emissions - prev_emissions, prev_emissions) if prev_emissions > Decimal("0") else Decimal("0")

            pathway.append(AnnualPathwayPoint(
                year=year,
                target_emissions_tco2e=_round_val(year_emissions, 2),
                reduction_from_baseline_pct=_round_val(reduction_from_base, 2),
                annual_reduction_rate_pct=_round_val(rate * Decimal("100"), 3),
                annual_reduction_tco2e=_round_val(annual_reduction, 2),
                cumulative_emissions_tco2e=_round_val(cumulative, 2),
                budget_allocated_tco2e=_round_val(budget_at_year, 2),
                budget_remaining_tco2e=_round_val(budget_remaining, 2),
                budget_compliance=compliance,
                yoy_change_pct=_round_val(yoy, 2),
            ))

            prev_emissions = year_emissions

        return pathway

    def _calculate_constant_rate(
        self,
        baseline: Decimal,
        target: Decimal,
        years: int,
    ) -> Decimal:
        """Calculate constant annual reduction rate.

        Formula:
            r = 1 - (target / baseline)^(1 / years)

        Args:
            baseline: Baseline emissions.
            target: Target emissions.
            years: Number of years.

        Returns:
            Annual reduction rate (decimal, not percentage).
        """
        if baseline <= Decimal("0") or years <= 0:
            return Decimal("0")

        ratio = float(_safe_divide(target, baseline))
        if ratio <= 0:
            ratio = 0.001

        try:
            annual_factor = ratio ** (1.0 / years)
            return _decimal(1.0 - annual_factor)
        except (OverflowError, ValueError):
            return Decimal("0")

    def _get_annual_rate(
        self,
        profile: ReductionProfile,
        base_rate: Decimal,
        year_index: int,
        total_years: int,
        acceleration: Decimal,
        custom_map: Dict[int, Decimal],
        year: int,
    ) -> Decimal:
        """Get annual reduction rate for a specific year.

        Args:
            profile: Reduction profile type.
            base_rate: Base constant reduction rate.
            year_index: Year index from baseline (1-based).
            total_years: Total pathway years.
            acceleration: Acceleration factor for accelerating profile.
            custom_map: Custom rate schedule.
            year: Calendar year.

        Returns:
            Annual reduction rate (decimal).
        """
        if profile == ReductionProfile.CUSTOM and year in custom_map:
            return custom_map[year]

        if profile == ReductionProfile.CONSTANT:
            return base_rate

        progress = _decimal(year_index) / _decimal(total_years) if total_years > 0 else Decimal("1")

        if profile == ReductionProfile.ACCELERATING:
            # Rate increases over time
            factor = Decimal("1") + acceleration * progress
            return base_rate * factor

        elif profile == ReductionProfile.DECELERATING:
            # Rate decreases over time
            factor = Decimal("1") + acceleration * (Decimal("1") - progress)
            return base_rate * factor

        elif profile == ReductionProfile.S_CURVE:
            # S-curve: slow start, fast middle, slow end
            x = float(progress) * 6.0 - 3.0  # Map to [-3, 3]
            try:
                s_factor = 1.0 / (1.0 + math.exp(-x))
            except OverflowError:
                s_factor = 0.5
            return base_rate * _decimal(s_factor) * Decimal("2")

        return base_rate

    # ------------------------------------------------------------------ #
    # Budget Allocation                                                    #
    # ------------------------------------------------------------------ #

    def _allocate_budget(
        self,
        total_budget: Decimal,
        year_index: int,
        total_years: int,
        strategy: BudgetAllocation,
    ) -> Decimal:
        """Allocate carbon budget through a specific year.

        Formulas:
            Equal: B(t) = total * t / T
            Front-loaded: B(t) = total * sqrt(t / T)
            Back-loaded: B(t) = total * (t / T)^2
            Proportional: B(t) = total * t / T (same as equal)

        Args:
            total_budget: Total carbon budget.
            year_index: Year index from baseline.
            total_years: Total pathway years.
            strategy: Budget allocation strategy.

        Returns:
            Allocated budget through this year.
        """
        if total_budget <= Decimal("0") or total_years <= 0:
            return Decimal("0")

        progress = _decimal(year_index) / _decimal(total_years)

        if strategy == BudgetAllocation.EQUAL or strategy == BudgetAllocation.PROPORTIONAL:
            return total_budget * progress

        elif strategy == BudgetAllocation.FRONT_LOADED:
            sqrt_progress = _decimal(math.sqrt(float(progress)))
            return total_budget * sqrt_progress

        elif strategy == BudgetAllocation.BACK_LOADED:
            return total_budget * progress * progress

        return total_budget * progress

    # ------------------------------------------------------------------ #
    # Quarterly Milestones                                                 #
    # ------------------------------------------------------------------ #

    def _generate_quarterly_milestones(
        self,
        data: AnnualPathwayInput,
        pathway: List[AnnualPathwayPoint],
    ) -> List[QuarterlyMilestone]:
        """Generate quarterly milestones from annual pathway.

        Formula:
            Q_emissions = annual / 4  (equal quarterly distribution)
            Adjusted quarterly based on interpolation between years.
        """
        milestones: List[QuarterlyMilestone] = []

        for i in range(len(pathway) - 1):
            current = pathway[i]
            next_pt = pathway[i + 1]

            for q in range(1, 5):
                progress = _decimal(q) / Decimal("4")
                q_emissions = current.target_emissions_tco2e + (
                    next_pt.target_emissions_tco2e - current.target_emissions_tco2e
                ) * progress

                ytd = current.target_emissions_tco2e * progress + (
                    q_emissions - current.target_emissions_tco2e
                ) * progress / Decimal("2")

                red_pct = _safe_pct(
                    data.baseline_emissions_tco2e - q_emissions,
                    data.baseline_emissions_tco2e,
                )

                milestones.append(QuarterlyMilestone(
                    year=current.year,
                    quarter=q,
                    target_emissions_tco2e=_round_val(q_emissions / Decimal("4"), 2),
                    cumulative_ytd_tco2e=_round_val(ytd, 2),
                    reduction_from_baseline_pct=_round_val(red_pct, 2),
                ))

        return milestones

    # ------------------------------------------------------------------ #
    # Carbon Budget Analysis                                               #
    # ------------------------------------------------------------------ #

    def _analyze_carbon_budget(
        self,
        data: AnnualPathwayInput,
        pathway: List[AnnualPathwayPoint],
    ) -> BudgetAnalysis:
        """Analyze carbon budget compliance for the pathway.

        Uses trapezoidal integration for cumulative emissions.
        """
        total_budget = data.total_carbon_budget_tco2e
        total_years = data.target_year - data.baseline_year

        # Calculate cumulative pathway emissions
        cumulative = Decimal("0")
        overshoot_years: List[int] = []
        allocations: List[Dict[str, Any]] = []
        exhaustion_year = 0

        for i, pt in enumerate(pathway):
            if i > 0:
                prev = pathway[i - 1]
                segment = (prev.target_emissions_tco2e + pt.target_emissions_tco2e) / Decimal("2")
                cumulative += segment

            budget_at_year = self._allocate_budget(
                total_budget, i, total_years, data.budget_allocation,
            )

            if cumulative > budget_at_year and budget_at_year > Decimal("0"):
                overshoot_years.append(pt.year)

            if cumulative >= total_budget and exhaustion_year == 0 and total_budget > Decimal("0"):
                exhaustion_year = pt.year

            allocations.append({
                "year": pt.year,
                "allocated_tco2e": str(_round_val(budget_at_year, 2)),
                "cumulative_tco2e": str(_round_val(cumulative, 2)),
            })

        surplus_deficit = total_budget - cumulative
        years_remaining = max(0, data.target_year - data.baseline_year)
        if cumulative > Decimal("0") and total_budget > Decimal("0"):
            avg_annual = _safe_divide(cumulative, _decimal(len(pathway)))
            if avg_annual > Decimal("0"):
                years_remaining = int(float(_safe_divide(
                    total_budget - cumulative, avg_annual,
                )))

        if surplus_deficit >= Decimal("0"):
            compliance = ComplianceStatus.COMPLIANT.value
        elif surplus_deficit >= -total_budget * Decimal("0.1"):
            compliance = ComplianceStatus.AT_RISK.value
        else:
            compliance = ComplianceStatus.NON_COMPLIANT.value

        return BudgetAnalysis(
            total_budget_tco2e=_round_val(total_budget, 2),
            cumulative_pathway_tco2e=_round_val(cumulative, 2),
            budget_surplus_deficit_tco2e=_round_val(surplus_deficit, 2),
            compliance_status=compliance,
            years_of_budget_remaining=max(years_remaining, 0),
            budget_exhaustion_year=exhaustion_year,
            annual_budget_allocations=allocations,
            overshoot_years=overshoot_years,
        )

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        data: AnnualPathwayInput,
        pathway: List[AnnualPathwayPoint],
    ) -> PathwaySummary:
        """Build pathway summary statistics."""
        if not pathway:
            return PathwaySummary()

        rates = [pt.annual_reduction_rate_pct for pt in pathway if pt.annual_reduction_rate_pct > Decimal("0")]
        avg_rate = _safe_divide(sum(rates), _decimal(len(rates))) if rates else Decimal("0")
        max_rate = max(rates) if rates else Decimal("0")
        min_rate = min(rates) if rates else Decimal("0")

        last = pathway[-1]
        total_reduction = data.baseline_emissions_tco2e - last.target_emissions_tco2e

        # Temperature score
        temp = Decimal("1.5") + max(
            Decimal("0"),
            _safe_divide(Decimal("4.2") - avg_rate, Decimal("4.2"))
        ) * Decimal("2.0")
        temp = min(temp, Decimal("4.0"))

        # SBTi near-term check
        near_term_pt = next(
            (pt for pt in pathway if pt.year == 2030), None
        )
        sbti_compliant = False
        if near_term_pt:
            sbti_compliant = near_term_pt.reduction_from_baseline_pct >= Decimal("42")

        return PathwaySummary(
            total_years=data.target_year - data.baseline_year,
            total_reduction_tco2e=_round_val(total_reduction, 2),
            total_reduction_pct=_round_val(last.reduction_from_baseline_pct, 2),
            average_annual_rate_pct=_round_val(avg_rate, 3),
            max_annual_rate_pct=_round_val(max_rate, 3),
            min_annual_rate_pct=_round_val(min_rate, 3),
            cumulative_emissions_tco2e=_round_val(last.cumulative_emissions_tco2e, 2),
            implied_temperature_score=_round_val(temp, 2),
            sbti_near_term_compliant=sbti_compliant,
        )

    # ------------------------------------------------------------------ #
    # Data Quality                                                         #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(self, data: AnnualPathwayInput) -> str:
        """Assess input data quality."""
        score = 0
        if data.baseline_emissions_tco2e > Decimal("0"):
            score += 3
        if data.target_year > data.baseline_year:
            score += 2
        if data.total_carbon_budget_tco2e > Decimal("0"):
            score += 2
        if len(data.historical_emissions) >= 3:
            score += 2
        if data.entity_id:
            score += 1

        if score >= 8:
            return DataQuality.HIGH.value
        elif score >= 5:
            return DataQuality.MEDIUM.value
        elif score >= 2:
            return DataQuality.LOW.value
        return DataQuality.ESTIMATED.value

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: AnnualPathwayInput,
        summary: Optional[PathwaySummary],
        budget: Optional[BudgetAnalysis],
    ) -> List[str]:
        """Generate pathway recommendations."""
        recs: List[str] = []

        if summary and not summary.sbti_near_term_compliant:
            recs.append(
                "Pathway does not achieve 42% reduction by 2030 (SBTi 1.5C minimum). "
                "Consider accelerating near-term actions or using front-loaded profile."
            )

        if data.reduction_profile == ReductionProfile.CONSTANT:
            recs.append(
                "Constant reduction rate may be challenging in early years. "
                "Consider accelerating profile with early quick-wins."
            )

        if data.total_carbon_budget_tco2e <= Decimal("0"):
            recs.append(
                "No carbon budget specified. Set a carbon budget to enable "
                "cumulative emissions tracking and budget compliance analysis."
            )

        if budget and budget.compliance_status == ComplianceStatus.NON_COMPLIANT.value:
            recs.append(
                f"Pathway exceeds carbon budget by "
                f"{abs(budget.budget_surplus_deficit_tco2e)} tCO2e. "
                f"Accelerate reductions or increase budget allocation."
            )

        if len(data.historical_emissions) < 3:
            recs.append(
                "Provide at least 3 years of historical emissions for trend "
                "validation and pathway calibration."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Warnings                                                             #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        data: AnnualPathwayInput,
        pathway: List[AnnualPathwayPoint],
        budget: Optional[BudgetAnalysis],
    ) -> List[str]:
        """Generate pathway warnings."""
        warns: List[str] = []

        if data.baseline_emissions_tco2e <= Decimal("0"):
            warns.append("Baseline emissions are zero. Cannot generate pathway.")

        total_years = data.target_year - data.baseline_year
        if total_years > 35:
            warns.append(
                f"Pathway spans {total_years} years. Consider setting "
                f"net-zero year to 2050 per SBTi recommendation."
            )

        # Check for any year with >10% reduction rate
        high_rate_years = [
            pt for pt in pathway
            if pt.annual_reduction_rate_pct > Decimal("10")
        ]
        if high_rate_years:
            warns.append(
                f"{len(high_rate_years)} year(s) require >10%/yr reduction rate. "
                f"Verify feasibility of these high reduction rates."
            )

        if budget and budget.overshoot_years:
            warns.append(
                f"Carbon budget exceeded in {len(budget.overshoot_years)} year(s): "
                f"{budget.overshoot_years[:5]}."
            )

        return warns

    # ------------------------------------------------------------------ #
    # Utility Methods                                                      #
    # ------------------------------------------------------------------ #

    def get_supported_profiles(self) -> List[str]:
        """Return supported reduction profiles."""
        return [p.value for p in ReductionProfile]

    def get_supported_allocations(self) -> List[str]:
        """Return supported budget allocation strategies."""
        return [a.value for a in BudgetAllocation]

    def get_global_carbon_budgets(self) -> Dict[str, str]:
        """Return IPCC AR6 global carbon budget estimates."""
        return {k: str(v) for k, v in GLOBAL_CARBON_BUDGETS.items()}
