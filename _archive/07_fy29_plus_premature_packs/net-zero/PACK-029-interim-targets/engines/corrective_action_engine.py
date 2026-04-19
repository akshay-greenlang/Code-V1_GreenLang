# -*- coding: utf-8 -*-
"""
CorrectiveActionEngine - PACK-029 Interim Targets Pack Engine 6
=================================================================

Quantifies the gap between current trajectory and target trajectory,
generates corrective action portfolios with initiative optimization,
accelerated reduction scenarios, investment requirements, catch-up
timeline modeling, and risk-adjusted action plans.

Calculation Methodology:
    Gap-to-Target Quantification:
        gap_tco2e = projected_at_target_year - target_at_target_year
        gap_annual = gap_tco2e / remaining_years
        gap_rate = required_rate - actual_rate

    Initiative Portfolio Optimization:
        Sort initiatives by cost-effectiveness (cost_per_tco2e ascending).
        Greedy fill until gap is covered or budget exhausted.
        Total cost = sum(initiative_cost * deployment_fraction)

    Accelerated Reduction Scenarios:
        Required rate = 1 - ((target - available_reduction) / current)^(1/remaining)
        Feasibility check: required_rate <= max_feasible_rate

    Catch-Up Timeline:
        At accelerated rate: years = ln(target/current) / ln(1 - accel_rate)
        Total catch-up = deviation_years + years_to_close_gap

    Investment Requirement:
        Total investment = sum(initiative_capex + initiative_opex * years)
        ROI from carbon savings and energy cost reduction.

    Risk-Adjusted Plan:
        risk_adjusted_reduction = reduction * (1 - risk_factor)
        Buffer = target_gap * risk_buffer_pct

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024) -- corrective measures
    - SBTi Target Tracking Protocol v2.0
    - CSRD ESRS E1-3 -- Actions to manage climate impacts
    - TCFD Recommendations -- Transition planning
    - McKinsey MACC methodology

Zero-Hallucination:
    - All gap calculations use deterministic Decimal arithmetic
    - Initiative ranking uses exact cost-per-tCO2e sorting
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-029 Interim Targets
Engine:  6 of 10
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

def _safe_divide(n: Decimal, d: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    if d == Decimal("0"):
        return default
    return n / d

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    q = "0." + "0" * places
    return value.quantize(Decimal(q), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InitiativeCategory(str, Enum):
    """Abatement initiative category."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    ELECTRIFICATION = "electrification"
    PROCESS_OPTIMIZATION = "process_optimization"
    SUPPLY_CHAIN = "supply_chain"
    BEHAVIORAL = "behavioral"
    TECHNOLOGY = "technology"
    CARBON_REMOVAL = "carbon_removal"

class FeasibilityLevel(str, Enum):
    """Feasibility assessment."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class UrgencyLevel(str, Enum):
    """Urgency of corrective action."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RiskLevel(str, Enum):
    """Risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class DataQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class AvailableInitiative(BaseModel):
    """An available abatement initiative.

    Attributes:
        initiative_id: Unique identifier.
        name: Initiative name.
        category: Initiative category.
        annual_reduction_tco2e: Annual reduction potential (tCO2e).
        cost_per_tco2e: Marginal abatement cost (currency/tCO2e).
        capex: Capital expenditure required.
        annual_opex: Annual operating cost.
        annual_savings: Annual cost savings.
        implementation_years: Years to full deployment.
        risk_factor: Risk factor (0-1, probability of underperformance).
        trl: Technology readiness level (1-9).
        max_deployment_pct: Maximum deployment fraction (0-100%).
    """
    initiative_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(default="", max_length=300)
    category: InitiativeCategory = Field(default=InitiativeCategory.ENERGY_EFFICIENCY)
    annual_reduction_tco2e: Decimal = Field(..., ge=Decimal("0"))
    cost_per_tco2e: Decimal = Field(default=Decimal("0"))
    capex: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    annual_opex: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    annual_savings: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    implementation_years: int = Field(default=1, ge=1, le=10)
    risk_factor: Decimal = Field(default=Decimal("0.1"), ge=Decimal("0"), le=Decimal("1"))
    trl: int = Field(default=9, ge=1, le=9)
    max_deployment_pct: Decimal = Field(default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"))

class CorrectiveActionInput(BaseModel):
    """Input for corrective action planning.

    Attributes:
        entity_name: Company name.
        entity_id: Entity identifier.
        current_emissions_tco2e: Current annual emissions.
        target_emissions_tco2e: Target annual emissions.
        target_year: Year to achieve target.
        current_year: Current year.
        actual_annual_rate_pct: Current annual reduction rate (%).
        required_annual_rate_pct: Required rate to hit target (%).
        projected_emissions_at_target_tco2e: BAU projection at target year.
        available_initiatives: Portfolio of available initiatives.
        budget_constraint: Maximum total investment budget.
        risk_buffer_pct: Risk buffer percentage (default 10%).
        max_feasible_annual_rate_pct: Maximum feasible annual reduction (%).
        include_risk_adjustment: Include risk-adjusted plan.
        include_catch_up: Include catch-up timeline.
        include_investment: Include investment analysis.
    """
    entity_name: str = Field(..., min_length=1, max_length=300)
    entity_id: str = Field(default="", max_length=100)
    current_emissions_tco2e: Decimal = Field(..., ge=Decimal("0"))
    target_emissions_tco2e: Decimal = Field(..., ge=Decimal("0"))
    target_year: int = Field(..., ge=2025, le=2070)
    current_year: int = Field(default=2024, ge=2020, le=2035)
    actual_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    required_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    projected_emissions_at_target_tco2e: Decimal = Field(default=Decimal("0"))
    available_initiatives: List[AvailableInitiative] = Field(default_factory=list)
    budget_constraint: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    risk_buffer_pct: Decimal = Field(default=Decimal("10"), ge=Decimal("0"), le=Decimal("50"))
    max_feasible_annual_rate_pct: Decimal = Field(default=Decimal("15"), ge=Decimal("0"))
    include_risk_adjustment: bool = Field(default=True)
    include_catch_up: bool = Field(default=True)
    include_investment: bool = Field(default=True)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class GapQuantification(BaseModel):
    """Gap-to-target quantification.

    Attributes:
        gap_tco2e: Total gap (projected - target).
        gap_pct: Gap as percentage of target.
        gap_annual_tco2e: Annual gap (spread over remaining years).
        rate_gap_pct: Gap in annual reduction rate.
        urgency: Urgency level based on gap size.
        remaining_years: Years remaining to target.
    """
    gap_tco2e: Decimal = Field(default=Decimal("0"))
    gap_pct: Decimal = Field(default=Decimal("0"))
    gap_annual_tco2e: Decimal = Field(default=Decimal("0"))
    rate_gap_pct: Decimal = Field(default=Decimal("0"))
    urgency: str = Field(default=UrgencyLevel.MEDIUM.value)
    remaining_years: int = Field(default=0)

class SelectedInitiative(BaseModel):
    """An initiative selected for the corrective action plan.

    Attributes:
        initiative_id: Initiative ID.
        name: Initiative name.
        category: Category.
        deployment_pct: Deployment fraction selected (%).
        annual_reduction_tco2e: Reduction at selected deployment.
        risk_adjusted_reduction_tco2e: After risk adjustment.
        total_cost: Total cost (capex + opex over period).
        cost_per_tco2e: Marginal cost.
        cumulative_gap_covered_pct: Running % of gap covered.
        implementation_start_year: Recommended start year.
        priority_rank: Priority ranking.
    """
    initiative_id: str = Field(default="")
    name: str = Field(default="")
    category: str = Field(default="")
    deployment_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    risk_adjusted_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    total_cost: Decimal = Field(default=Decimal("0"))
    cost_per_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_gap_covered_pct: Decimal = Field(default=Decimal("0"))
    implementation_start_year: int = Field(default=0)
    priority_rank: int = Field(default=0)

class AcceleratedScenario(BaseModel):
    """Accelerated reduction scenario.

    Attributes:
        scenario_name: Scenario name.
        required_rate_pct: Required annual reduction rate.
        is_feasible: Whether the rate is achievable.
        feasibility_level: Feasibility assessment.
        years_to_close_gap: Years to return to target pathway.
        additional_annual_reduction_tco2e: Extra annual reduction needed.
        additional_investment: Extra investment needed.
    """
    scenario_name: str = Field(default="")
    required_rate_pct: Decimal = Field(default=Decimal("0"))
    is_feasible: bool = Field(default=False)
    feasibility_level: str = Field(default=FeasibilityLevel.MEDIUM.value)
    years_to_close_gap: int = Field(default=0)
    additional_annual_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    additional_investment: Decimal = Field(default=Decimal("0"))

class CatchUpTimeline(BaseModel):
    """Catch-up timeline modeling.

    Attributes:
        current_deviation_years: Years behind schedule.
        catch_up_rate_pct: Required catch-up rate.
        catch_up_years: Years to return to pathway.
        return_to_pathway_year: Year when pathway is rejoined.
        is_achievable: Whether catch-up is achievable.
        annual_pathway: Year-by-year catch-up pathway.
    """
    current_deviation_years: int = Field(default=0)
    catch_up_rate_pct: Decimal = Field(default=Decimal("0"))
    catch_up_years: int = Field(default=0)
    return_to_pathway_year: int = Field(default=0)
    is_achievable: bool = Field(default=False)
    annual_pathway: List[Dict[str, Any]] = Field(default_factory=list)

class InvestmentAnalysis(BaseModel):
    """Investment requirement analysis.

    Attributes:
        total_capex: Total capital expenditure.
        total_annual_opex: Total annual operating cost.
        total_annual_savings: Total annual savings.
        net_annual_cost: Net annual cost (opex - savings).
        simple_payback_years: Simple payback period.
        total_investment: Total investment over period.
        cost_per_tco2e_abated: Average cost per tCO2e abated.
        roi_pct: Return on investment percentage.
    """
    total_capex: Decimal = Field(default=Decimal("0"))
    total_annual_opex: Decimal = Field(default=Decimal("0"))
    total_annual_savings: Decimal = Field(default=Decimal("0"))
    net_annual_cost: Decimal = Field(default=Decimal("0"))
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    total_investment: Decimal = Field(default=Decimal("0"))
    cost_per_tco2e_abated: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))

class CorrectiveActionResult(BaseModel):
    """Complete corrective action result."""
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    gap: Optional[GapQuantification] = Field(default=None)
    selected_initiatives: List[SelectedInitiative] = Field(default_factory=list)
    total_portfolio_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    gap_coverage_pct: Decimal = Field(default=Decimal("0"))
    accelerated_scenarios: List[AcceleratedScenario] = Field(default_factory=list)
    catch_up_timeline: Optional[CatchUpTimeline] = Field(default=None)
    investment_analysis: Optional[InvestmentAnalysis] = Field(default=None)
    urgency_level: str = Field(default=UrgencyLevel.MEDIUM.value)
    data_quality: str = Field(default=DataQuality.MEDIUM.value)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CorrectiveActionEngine:
    """Corrective action engine for PACK-029.

    Quantifies gap-to-target and builds optimized initiative portfolios
    for returning to target pathway.

    Usage::

        engine = CorrectiveActionEngine()
        result = await engine.calculate(corrective_input)
        print(f"Gap: {result.gap.gap_tco2e} tCO2e")
        for init in result.selected_initiatives:
            print(f"  {init.name}: {init.annual_reduction_tco2e} tCO2e")
    """

    engine_version: str = _MODULE_VERSION

    async def calculate(self, data: CorrectiveActionInput) -> CorrectiveActionResult:
        """Run complete corrective action analysis."""
        t0 = time.perf_counter()
        logger.info(
            "Corrective action: entity=%s, gap=%s->%s, year=%d",
            data.entity_name,
            str(data.current_emissions_tco2e),
            str(data.target_emissions_tco2e),
            data.target_year,
        )

        # Gap quantification
        gap = self._quantify_gap(data)

        # Initiative portfolio optimization
        selected, total_reduction, coverage = self._optimize_portfolio(data, gap)

        # Accelerated scenarios
        accel = self._build_accelerated_scenarios(data, gap)

        # Catch-up timeline
        catch_up: Optional[CatchUpTimeline] = None
        if data.include_catch_up:
            catch_up = self._model_catch_up(data, gap)

        # Investment analysis
        investment: Optional[InvestmentAnalysis] = None
        if data.include_investment and selected:
            investment = self._analyze_investment(data, selected)

        dq = self._assess_data_quality(data)
        recs = self._generate_recommendations(data, gap, coverage, investment)
        warns = self._generate_warnings(data, gap, selected)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = CorrectiveActionResult(
            entity_name=data.entity_name,
            entity_id=data.entity_id,
            gap=gap,
            selected_initiatives=selected,
            total_portfolio_reduction_tco2e=_round_val(total_reduction, 2),
            gap_coverage_pct=_round_val(coverage, 2),
            accelerated_scenarios=accel,
            catch_up_timeline=catch_up,
            investment_analysis=investment,
            urgency_level=gap.urgency if gap else UrgencyLevel.MEDIUM.value,
            data_quality=dq,
            recommendations=recs,
            warnings=warns,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    async def calculate_batch(
        self, inputs: List[CorrectiveActionInput],
    ) -> List[CorrectiveActionResult]:
        results: List[CorrectiveActionResult] = []
        for inp in inputs:
            try:
                results.append(await self.calculate(inp))
            except Exception as exc:
                logger.error("Batch error for %s: %s", inp.entity_name, exc)
                results.append(CorrectiveActionResult(
                    entity_name=inp.entity_name,
                    warnings=[f"Calculation error: {exc}"],
                ))
        return results

    # ------------------------------------------------------------------ #
    # Gap Quantification                                                   #
    # ------------------------------------------------------------------ #

    def _quantify_gap(self, data: CorrectiveActionInput) -> GapQuantification:
        """Quantify the gap between current trajectory and target."""
        remaining = max(data.target_year - data.current_year, 1)

        # Use projected emissions or extrapolate from current rate
        projected = data.projected_emissions_at_target_tco2e
        if projected <= Decimal("0"):
            # Project at current rate
            if data.actual_annual_rate_pct > Decimal("0"):
                try:
                    factor = (Decimal("1") - data.actual_annual_rate_pct / Decimal("100")) ** remaining
                    projected = data.current_emissions_tco2e * factor
                except (OverflowError, ValueError):
                    projected = data.current_emissions_tco2e
            else:
                projected = data.current_emissions_tco2e

        gap = projected - data.target_emissions_tco2e
        gap = max(gap, Decimal("0"))
        gap_pct = _safe_pct(gap, data.target_emissions_tco2e)
        gap_annual = _safe_divide(gap, _decimal(remaining))
        rate_gap = data.required_annual_rate_pct - data.actual_annual_rate_pct

        # Urgency
        if gap_pct > Decimal("50"):
            urgency = UrgencyLevel.CRITICAL.value
        elif gap_pct > Decimal("25"):
            urgency = UrgencyLevel.HIGH.value
        elif gap_pct > Decimal("10"):
            urgency = UrgencyLevel.MEDIUM.value
        else:
            urgency = UrgencyLevel.LOW.value

        return GapQuantification(
            gap_tco2e=_round_val(gap, 2),
            gap_pct=_round_val(gap_pct, 2),
            gap_annual_tco2e=_round_val(gap_annual, 2),
            rate_gap_pct=_round_val(rate_gap, 3),
            urgency=urgency,
            remaining_years=remaining,
        )

    # ------------------------------------------------------------------ #
    # Portfolio Optimization                                               #
    # ------------------------------------------------------------------ #

    def _optimize_portfolio(
        self, data: CorrectiveActionInput, gap: GapQuantification,
    ) -> Tuple[List[SelectedInitiative], Decimal, Decimal]:
        """Optimize initiative portfolio to close the gap.

        Uses greedy selection sorted by cost-effectiveness.
        """
        if gap.gap_tco2e <= Decimal("0") or not data.available_initiatives:
            return [], Decimal("0"), Decimal("100")

        # Sort by cost per tCO2e (MACC ordering)
        sorted_inits = sorted(
            data.available_initiatives,
            key=lambda i: float(i.cost_per_tco2e),
        )

        selected: List[SelectedInitiative] = []
        remaining_gap = gap.gap_tco2e
        if data.include_risk_adjustment:
            remaining_gap = remaining_gap * (Decimal("1") + data.risk_buffer_pct / Decimal("100"))

        total_reduction = Decimal("0")
        total_cost = Decimal("0")
        budget = data.budget_constraint if data.budget_constraint > Decimal("0") else Decimal("999999999")
        rank = 0

        for init in sorted_inits:
            if remaining_gap <= Decimal("0"):
                break

            max_deploy = init.max_deployment_pct / Decimal("100")
            max_reduction = init.annual_reduction_tco2e * max_deploy

            # Determine deployment fraction
            if max_reduction <= remaining_gap:
                deploy_frac = max_deploy
            else:
                deploy_frac = max_deploy * _safe_divide(remaining_gap, max_reduction)

            actual_reduction = init.annual_reduction_tco2e * deploy_frac
            risk_adjusted = actual_reduction * (Decimal("1") - init.risk_factor)

            remaining_years = max(data.target_year - data.current_year, 1)
            init_cost = init.capex * deploy_frac + init.annual_opex * deploy_frac * _decimal(remaining_years)

            if init_cost > budget - total_cost and total_cost > Decimal("0"):
                continue

            rank += 1
            total_reduction += risk_adjusted if data.include_risk_adjustment else actual_reduction
            total_cost += init_cost
            remaining_gap -= risk_adjusted if data.include_risk_adjustment else actual_reduction

            coverage_so_far = _safe_pct(total_reduction, gap.gap_tco2e)

            selected.append(SelectedInitiative(
                initiative_id=init.initiative_id,
                name=init.name,
                category=init.category.value,
                deployment_pct=_round_val(deploy_frac * Decimal("100"), 1),
                annual_reduction_tco2e=_round_val(actual_reduction, 2),
                risk_adjusted_reduction_tco2e=_round_val(risk_adjusted, 2),
                total_cost=_round_val(init_cost, 2),
                cost_per_tco2e=_round_val(init.cost_per_tco2e, 2),
                cumulative_gap_covered_pct=_round_val(coverage_so_far, 1),
                implementation_start_year=data.current_year + min(rank - 1, 2),
                priority_rank=rank,
            ))

        gap_coverage = _safe_pct(total_reduction, gap.gap_tco2e)
        return selected, total_reduction, gap_coverage

    # ------------------------------------------------------------------ #
    # Accelerated Scenarios                                                #
    # ------------------------------------------------------------------ #

    def _build_accelerated_scenarios(
        self, data: CorrectiveActionInput, gap: GapQuantification,
    ) -> List[AcceleratedScenario]:
        """Build accelerated reduction scenarios."""
        scenarios: List[AcceleratedScenario] = []
        remaining = max(data.target_year - data.current_year, 1)

        for name, multiplier in [("moderate", Decimal("1.5")), ("aggressive", Decimal("2.0")), ("maximum", Decimal("3.0"))]:
            accel_rate = data.actual_annual_rate_pct * multiplier
            additional = (accel_rate - data.actual_annual_rate_pct) / Decimal("100") * data.current_emissions_tco2e

            is_feasible = accel_rate <= data.max_feasible_annual_rate_pct
            if accel_rate <= Decimal("5"):
                feasibility = FeasibilityLevel.HIGH.value
            elif accel_rate <= Decimal("10"):
                feasibility = FeasibilityLevel.MEDIUM.value
            elif accel_rate <= Decimal("15"):
                feasibility = FeasibilityLevel.LOW.value
            else:
                feasibility = FeasibilityLevel.VERY_LOW.value

            # Years to close gap at accelerated rate
            if additional > Decimal("0") and gap.gap_tco2e > Decimal("0"):
                years_close = min(int(float(_safe_divide(gap.gap_tco2e, additional))), remaining)
            else:
                years_close = remaining

            # Rough investment estimate
            avg_cost = Decimal("50")  # Default cost per tCO2e
            if data.available_initiatives:
                costs = [i.cost_per_tco2e for i in data.available_initiatives if i.cost_per_tco2e > Decimal("0")]
                if costs:
                    avg_cost = sum(costs) / _decimal(len(costs))
            add_investment = additional * avg_cost * _decimal(years_close)

            scenarios.append(AcceleratedScenario(
                scenario_name=f"{name}_acceleration",
                required_rate_pct=_round_val(accel_rate, 3),
                is_feasible=is_feasible,
                feasibility_level=feasibility,
                years_to_close_gap=years_close,
                additional_annual_reduction_tco2e=_round_val(additional, 2),
                additional_investment=_round_val(add_investment, 2),
            ))

        return scenarios

    # ------------------------------------------------------------------ #
    # Catch-Up Timeline                                                    #
    # ------------------------------------------------------------------ #

    def _model_catch_up(
        self, data: CorrectiveActionInput, gap: GapQuantification,
    ) -> CatchUpTimeline:
        """Model timeline to catch up to target pathway."""
        remaining = max(data.target_year - data.current_year, 1)

        # Calculate required catch-up rate
        if data.current_emissions_tco2e > Decimal("0") and data.target_emissions_tco2e > Decimal("0"):
            ratio = float(_safe_divide(data.target_emissions_tco2e, data.current_emissions_tco2e))
            if ratio > 0 and remaining > 0:
                try:
                    catch_up_rate = _decimal((1.0 - ratio ** (1.0 / remaining)) * 100)
                except (OverflowError, ValueError):
                    catch_up_rate = Decimal("0")
            else:
                catch_up_rate = Decimal("0")
        else:
            catch_up_rate = Decimal("0")

        deviation_years = 0
        if data.actual_annual_rate_pct > Decimal("0"):
            if gap.rate_gap_pct > Decimal("0"):
                deviation_years = min(
                    int(float(_safe_divide(gap.gap_pct, gap.rate_gap_pct * Decimal("10")))),
                    remaining,
                )

        catch_up_years = min(
            int(float(_safe_divide(gap.gap_pct, catch_up_rate))) if catch_up_rate > Decimal("0") else remaining,
            remaining,
        )

        return_year = data.current_year + catch_up_years
        is_achievable = catch_up_rate <= data.max_feasible_annual_rate_pct

        # Annual catch-up pathway
        annual_path: List[Dict[str, Any]] = []
        current_e = data.current_emissions_tco2e
        for y in range(data.current_year, data.target_year + 1):
            current_e = current_e * (Decimal("1") - catch_up_rate / Decimal("100"))
            current_e = max(current_e, Decimal("0"))
            annual_path.append({
                "year": y,
                "emissions_tco2e": str(_round_val(current_e, 2)),
                "catch_up_rate_pct": str(_round_val(catch_up_rate, 3)),
            })

        return CatchUpTimeline(
            current_deviation_years=deviation_years,
            catch_up_rate_pct=_round_val(catch_up_rate, 3),
            catch_up_years=catch_up_years,
            return_to_pathway_year=return_year,
            is_achievable=is_achievable,
            annual_pathway=annual_path,
        )

    # ------------------------------------------------------------------ #
    # Investment Analysis                                                  #
    # ------------------------------------------------------------------ #

    def _analyze_investment(
        self, data: CorrectiveActionInput,
        selected: List[SelectedInitiative],
    ) -> InvestmentAnalysis:
        """Analyze investment requirements for selected initiatives."""
        remaining = max(data.target_year - data.current_year, 1)

        total_capex = Decimal("0")
        total_opex = Decimal("0")
        total_savings = Decimal("0")
        total_reduction = Decimal("0")

        for sel in selected:
            init = next(
                (i for i in data.available_initiatives if i.initiative_id == sel.initiative_id),
                None,
            )
            if init:
                deploy = sel.deployment_pct / Decimal("100")
                total_capex += init.capex * deploy
                total_opex += init.annual_opex * deploy
                total_savings += init.annual_savings * deploy
                total_reduction += sel.annual_reduction_tco2e

        net_annual = total_opex - total_savings
        total_investment = total_capex + net_annual * _decimal(remaining)

        payback = _safe_divide(total_capex, total_savings - total_opex) if total_savings > total_opex else Decimal("0")
        cost_per_tco2e = _safe_divide(total_investment, total_reduction * _decimal(remaining)) if total_reduction > Decimal("0") else Decimal("0")

        roi = Decimal("0")
        if total_investment > Decimal("0"):
            total_savings_lifetime = total_savings * _decimal(remaining)
            roi = _safe_pct(total_savings_lifetime - total_investment, total_investment)

        return InvestmentAnalysis(
            total_capex=_round_val(total_capex, 2),
            total_annual_opex=_round_val(total_opex, 2),
            total_annual_savings=_round_val(total_savings, 2),
            net_annual_cost=_round_val(net_annual, 2),
            simple_payback_years=_round_val(max(payback, Decimal("0")), 1),
            total_investment=_round_val(total_investment, 2),
            cost_per_tco2e_abated=_round_val(cost_per_tco2e, 2),
            roi_pct=_round_val(roi, 1),
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(self, data: CorrectiveActionInput) -> str:
        score = 0
        if data.current_emissions_tco2e > Decimal("0"):
            score += 2
        if data.target_emissions_tco2e > Decimal("0"):
            score += 2
        if len(data.available_initiatives) >= 3:
            score += 2
        if data.actual_annual_rate_pct > Decimal("0"):
            score += 1
        if data.projected_emissions_at_target_tco2e > Decimal("0"):
            score += 1
        if data.budget_constraint > Decimal("0"):
            score += 1
        if data.entity_id:
            score += 1
        if score >= 8:
            return DataQuality.HIGH.value
        elif score >= 5:
            return DataQuality.MEDIUM.value
        elif score >= 2:
            return DataQuality.LOW.value
        return DataQuality.ESTIMATED.value

    def _generate_recommendations(
        self, data: CorrectiveActionInput,
        gap: GapQuantification, coverage: Decimal,
        investment: Optional[InvestmentAnalysis],
    ) -> List[str]:
        recs: List[str] = []
        if gap.urgency == UrgencyLevel.CRITICAL.value:
            recs.append(
                f"CRITICAL: Gap of {gap.gap_tco2e} tCO2e requires immediate "
                f"action. Deploy highest-impact initiatives within 6 months."
            )
        if coverage < Decimal("100"):
            recs.append(
                f"Available initiatives cover only {coverage}% of the gap. "
                f"Identify additional reduction opportunities or adjust targets."
            )
        if investment and investment.roi_pct > Decimal("0"):
            recs.append(
                f"Portfolio ROI of {investment.roi_pct}% indicates net-positive "
                f"business case. Payback period: {investment.simple_payback_years} years."
            )
        if not data.available_initiatives:
            recs.append(
                "No initiatives provided. Conduct abatement opportunity "
                "assessment (MACC analysis) to identify options."
            )
        return recs

    def _generate_warnings(
        self, data: CorrectiveActionInput,
        gap: GapQuantification, selected: List[SelectedInitiative],
    ) -> List[str]:
        warns: List[str] = []
        if gap.gap_tco2e <= Decimal("0"):
            warns.append("No gap detected. Entity is on track or ahead of target.")
        if gap.rate_gap_pct > Decimal("5"):
            warns.append(
                f"Rate gap of {gap.rate_gap_pct}%/yr is very large. "
                f"May require fundamental business model changes."
            )
        high_risk = [s for s in selected if s.risk_adjusted_reduction_tco2e < s.annual_reduction_tco2e * Decimal("0.7")]
        if high_risk:
            warns.append(
                f"{len(high_risk)} initiative(s) have >30% risk adjustment. "
                f"Consider pilot testing before full deployment."
            )
        return warns

    def get_initiative_categories(self) -> List[str]:
        """Return supported initiative categories."""
        return [c.value for c in InitiativeCategory]
