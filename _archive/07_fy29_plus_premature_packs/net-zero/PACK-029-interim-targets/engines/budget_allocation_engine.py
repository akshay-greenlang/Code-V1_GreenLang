# -*- coding: utf-8 -*-
"""
BudgetAllocationEngine - PACK-029 Interim Targets Pack Engine 9
=================================================================

Calculates organizational carbon budgets and allocates annual emission
allowances with equal, front-loaded, and back-loaded strategies.
Tracks budget drawdown, overshoot/undershoot analysis, rebalancing
recommendations, and internal carbon pricing integration.

Calculation Methodology:
    Carbon Budget Calculation (Cumulative Emissions):
        total_budget = sum_{y=base}^{target} pathway(y)
        Approximated using trapezoidal integration of the pathway.

    Annual Budget Allocation:
        Equal: B(y) = total_budget / years
        Front-loaded: B(y) = total_budget * (2*(years-i+1)) / (years*(years+1))
        Back-loaded: B(y) = total_budget * (2*i) / (years*(years+1))
        Proportional: B(y) = pathway(y)  (match the target pathway)

    Budget Drawdown:
        remaining(y) = total_budget - cumulative_actual(y)
        drawdown_rate = cumulative_actual(y) / total_budget

    Overshoot/Undershoot:
        overshoot(y) = actual(y) - budget(y)
        cumulative_overshoot = sum(max(0, overshoot(y)))

    Rebalancing:
        If overshoot in year y: spread deficit over remaining years
        new_budget(y+i) = old_budget(y+i) - deficit / remaining_years

    Internal Carbon Price:
        internal_price = total_cost_of_reduction / total_reduction_tco2e
        budget_cost(y) = budget(y) * internal_price
        overshoot_penalty = overshoot(y) * penalty_rate

Regulatory References:
    - IPCC AR6 WG1 (2021) -- Remaining carbon budgets
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - CSRD ESRS E1-4 -- GHG reduction targets
    - TCFD Recommendations -- Carbon pricing disclosure
    - World Bank State of Carbon Pricing (2024)
    - High-Level Commission on Carbon Prices (Stern-Stiglitz, 2017)

Zero-Hallucination:
    - All budget calculations use trapezoidal integration
    - Allocation formulas use exact arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-029 Interim Targets
Engine:  9 of 10
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

from pydantic import BaseModel, Field

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
        serializable = {k: v for k, v in serializable.items() if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
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

class AllocationStrategy(str, Enum):
    EQUAL = "equal"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"
    PROPORTIONAL = "proportional"

class BudgetStatus(str, Enum):
    WITHIN_BUDGET = "within_budget"
    AT_RISK = "at_risk"
    OVERSHOOT = "overshoot"
    EXHAUSTED = "exhausted"

class DataQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Internal carbon price benchmarks (USD/tCO2e, 2024)
CARBON_PRICE_BENCHMARKS: Dict[str, Decimal] = {
    "low": Decimal("25"),
    "medium": Decimal("75"),
    "high": Decimal("150"),
    "stern_stiglitz_2030": Decimal("100"),
    "eu_ets_2024": Decimal("85"),
    "iea_nze_2030": Decimal("140"),
    "iea_nze_2050": Decimal("250"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class PathwayPoint(BaseModel):
    """A point on the target pathway."""
    year: int = Field(..., ge=2015, le=2070)
    target_tco2e: Decimal = Field(..., ge=Decimal("0"))

class ActualEmission(BaseModel):
    """Actual emissions for a year."""
    year: int = Field(..., ge=2015, le=2050)
    actual_tco2e: Decimal = Field(..., ge=Decimal("0"))

class BudgetAllocationInput(BaseModel):
    """Input for budget allocation."""
    entity_name: str = Field(..., min_length=1, max_length=300)
    entity_id: str = Field(default="", max_length=100)
    baseline_year: int = Field(..., ge=2015, le=2025)
    baseline_tco2e: Decimal = Field(..., ge=Decimal("0"))
    target_year: int = Field(default=2050, ge=2030, le=2070)
    target_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    target_reduction_pct: Decimal = Field(default=Decimal("90"))
    pathway_points: List[PathwayPoint] = Field(default_factory=list)
    actual_emissions: List[ActualEmission] = Field(default_factory=list)
    allocation_strategy: AllocationStrategy = Field(default=AllocationStrategy.PROPORTIONAL)
    total_carbon_budget_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    internal_carbon_price: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    overshoot_penalty_rate: Decimal = Field(default=Decimal("2.0"))
    include_rebalancing: bool = Field(default=True)
    include_carbon_pricing: bool = Field(default=True)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class AnnualBudget(BaseModel):
    """Annual budget allocation and tracking."""
    year: int = Field(default=0)
    budget_tco2e: Decimal = Field(default=Decimal("0"))
    actual_tco2e: Decimal = Field(default=Decimal("0"))
    variance_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_budget_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_actual_tco2e: Decimal = Field(default=Decimal("0"))
    remaining_budget_tco2e: Decimal = Field(default=Decimal("0"))
    drawdown_pct: Decimal = Field(default=Decimal("0"))
    status: str = Field(default=BudgetStatus.WITHIN_BUDGET.value)
    carbon_cost: Decimal = Field(default=Decimal("0"))
    overshoot_penalty: Decimal = Field(default=Decimal("0"))

class RebalancingRecommendation(BaseModel):
    """Budget rebalancing recommendation."""
    trigger_year: int = Field(default=0)
    overshoot_tco2e: Decimal = Field(default=Decimal("0"))
    remaining_years: int = Field(default=0)
    adjusted_annual_budget_tco2e: Decimal = Field(default=Decimal("0"))
    adjustment_pct: Decimal = Field(default=Decimal("0"))
    description: str = Field(default="")

class CarbonPricingAnalysis(BaseModel):
    """Internal carbon pricing analysis."""
    internal_price_per_tco2e: Decimal = Field(default=Decimal("0"))
    total_budget_value: Decimal = Field(default=Decimal("0"))
    total_overshoot_penalty: Decimal = Field(default=Decimal("0"))
    annual_carbon_costs: List[Dict[str, Any]] = Field(default_factory=list)
    price_vs_benchmarks: Dict[str, str] = Field(default_factory=dict)

class BudgetSummary(BaseModel):
    """Budget summary statistics."""
    total_budget_tco2e: Decimal = Field(default=Decimal("0"))
    total_allocated_tco2e: Decimal = Field(default=Decimal("0"))
    total_actual_tco2e: Decimal = Field(default=Decimal("0"))
    total_overshoot_tco2e: Decimal = Field(default=Decimal("0"))
    total_undershoot_tco2e: Decimal = Field(default=Decimal("0"))
    budget_utilization_pct: Decimal = Field(default=Decimal("0"))
    years_in_overshoot: int = Field(default=0)
    years_in_undershoot: int = Field(default=0)
    exhaustion_year: int = Field(default=0)
    years_remaining_at_current_rate: int = Field(default=0)

class BudgetAllocationResult(BaseModel):
    """Complete budget allocation result."""
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    allocation_strategy: str = Field(default="")
    annual_budgets: List[AnnualBudget] = Field(default_factory=list)
    rebalancing: List[RebalancingRecommendation] = Field(default_factory=list)
    carbon_pricing: Optional[CarbonPricingAnalysis] = Field(default=None)
    summary: Optional[BudgetSummary] = Field(default=None)
    data_quality: str = Field(default=DataQuality.MEDIUM.value)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BudgetAllocationEngine:
    """Carbon budget allocation engine for PACK-029.

    Calculates and allocates carbon budgets with multiple strategies,
    tracks drawdown, and integrates internal carbon pricing.

    Usage::

        engine = BudgetAllocationEngine()
        result = await engine.calculate(budget_input)
        for ab in result.annual_budgets:
            print(f"  {ab.year}: budget={ab.budget_tco2e}, status={ab.status}")
    """

    engine_version: str = _MODULE_VERSION

    async def calculate(self, data: BudgetAllocationInput) -> BudgetAllocationResult:
        """Run complete budget allocation analysis."""
        t0 = time.perf_counter()
        logger.info("Budget allocation: entity=%s, strategy=%s", data.entity_name, data.allocation_strategy.value)

        # Calculate total carbon budget
        total_budget = self._calculate_total_budget(data)

        # Allocate annual budgets
        annual_budgets = self._allocate_annual(data, total_budget)

        # Track actuals
        annual_budgets = self._track_actuals(data, annual_budgets, total_budget)

        # Rebalancing
        rebalancing: List[RebalancingRecommendation] = []
        if data.include_rebalancing:
            rebalancing = self._generate_rebalancing(data, annual_budgets, total_budget)

        # Carbon pricing
        pricing: Optional[CarbonPricingAnalysis] = None
        if data.include_carbon_pricing and data.internal_carbon_price > Decimal("0"):
            pricing = self._analyze_carbon_pricing(data, annual_budgets)

        # Summary
        summary = self._build_summary(data, annual_budgets, total_budget)

        dq = self._assess_data_quality(data)
        recs = self._generate_recommendations(data, summary, rebalancing)
        warns = self._generate_warnings(data, summary)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = BudgetAllocationResult(
            entity_name=data.entity_name, entity_id=data.entity_id,
            allocation_strategy=data.allocation_strategy.value,
            annual_budgets=annual_budgets, rebalancing=rebalancing,
            carbon_pricing=pricing, summary=summary,
            data_quality=dq, recommendations=recs, warnings=warns,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    async def calculate_batch(self, inputs: List[BudgetAllocationInput]) -> List[BudgetAllocationResult]:
        results = []
        for inp in inputs:
            try:
                results.append(await self.calculate(inp))
            except Exception as exc:
                logger.error("Batch error: %s", exc)
                results.append(BudgetAllocationResult(entity_name=inp.entity_name, warnings=[f"Error: {exc}"]))
        return results

    def _calculate_total_budget(self, data: BudgetAllocationInput) -> Decimal:
        """Calculate total carbon budget using trapezoidal integration."""
        if data.total_carbon_budget_tco2e > Decimal("0"):
            return data.total_carbon_budget_tco2e

        target = data.target_tco2e
        if target <= Decimal("0"):
            target = data.baseline_tco2e * (Decimal("1") - data.target_reduction_pct / Decimal("100"))

        years = data.target_year - data.baseline_year
        if years <= 0:
            return data.baseline_tco2e

        # If pathway points provided, use trapezoidal integration
        if data.pathway_points:
            points = sorted(data.pathway_points, key=lambda p: p.year)
            budget = Decimal("0")
            for i in range(len(points) - 1):
                dt = _decimal(points[i + 1].year - points[i].year)
                avg = (points[i].target_tco2e + points[i + 1].target_tco2e) / Decimal("2")
                budget += avg * dt
            return budget

        # Linear interpolation
        return (data.baseline_tco2e + target) / Decimal("2") * _decimal(years)

    def _allocate_annual(
        self, data: BudgetAllocationInput, total_budget: Decimal,
    ) -> List[AnnualBudget]:
        """Allocate budget across years."""
        years = data.target_year - data.baseline_year
        if years <= 0:
            return []

        target = data.target_tco2e
        if target <= Decimal("0"):
            target = data.baseline_tco2e * (Decimal("1") - data.target_reduction_pct / Decimal("100"))

        # Build pathway lookup from pathway_points
        pathway_map: Dict[int, Decimal] = {}
        if data.pathway_points:
            for pp in data.pathway_points:
                pathway_map[pp.year] = pp.target_tco2e

        budgets: List[AnnualBudget] = []
        cumulative = Decimal("0")

        for i in range(years + 1):
            year = data.baseline_year + i
            n = years

            if data.allocation_strategy == AllocationStrategy.EQUAL:
                annual = _safe_divide(total_budget, _decimal(n))

            elif data.allocation_strategy == AllocationStrategy.FRONT_LOADED:
                # Higher budget in early years: weight = 2*(n-i+1) / (n*(n+1))
                weight = Decimal("2") * _decimal(n - i + 1) / (_decimal(n) * _decimal(n + 1))
                annual = total_budget * weight

            elif data.allocation_strategy == AllocationStrategy.BACK_LOADED:
                # Lower budget in early years: weight = 2*(i+1) / (n*(n+1))
                weight = Decimal("2") * _decimal(i + 1) / (_decimal(n) * _decimal(n + 1))
                annual = total_budget * weight

            elif data.allocation_strategy == AllocationStrategy.PROPORTIONAL:
                # Match the target pathway
                if year in pathway_map:
                    annual = pathway_map[year]
                else:
                    # Linear interpolation
                    progress = _decimal(i) / _decimal(n) if n > 0 else Decimal("1")
                    annual = data.baseline_tco2e + (target - data.baseline_tco2e) * progress
            else:
                annual = _safe_divide(total_budget, _decimal(n))

            cumulative += annual

            budgets.append(AnnualBudget(
                year=year,
                budget_tco2e=_round_val(annual, 2),
                cumulative_budget_tco2e=_round_val(cumulative, 2),
                remaining_budget_tco2e=_round_val(total_budget - cumulative, 2),
                drawdown_pct=_round_val(_safe_pct(cumulative, total_budget), 1),
                status=BudgetStatus.WITHIN_BUDGET.value,
            ))

        return budgets

    def _track_actuals(
        self, data: BudgetAllocationInput,
        budgets: List[AnnualBudget], total_budget: Decimal,
    ) -> List[AnnualBudget]:
        """Track actual emissions against budgets."""
        actual_map = {a.year: a.actual_tco2e for a in data.actual_emissions}
        cumulative_actual = Decimal("0")

        for ab in budgets:
            actual = actual_map.get(ab.year)
            if actual is not None:
                ab.actual_tco2e = actual
                ab.variance_tco2e = _round_val(actual - ab.budget_tco2e, 2)
                cumulative_actual += actual
                ab.cumulative_actual_tco2e = _round_val(cumulative_actual, 2)

                remaining = total_budget - cumulative_actual
                ab.remaining_budget_tco2e = _round_val(remaining, 2)
                ab.drawdown_pct = _round_val(_safe_pct(cumulative_actual, total_budget), 1)

                if actual <= ab.budget_tco2e:
                    ab.status = BudgetStatus.WITHIN_BUDGET.value
                elif actual <= ab.budget_tco2e * Decimal("1.1"):
                    ab.status = BudgetStatus.AT_RISK.value
                else:
                    ab.status = BudgetStatus.OVERSHOOT.value

                if remaining <= Decimal("0"):
                    ab.status = BudgetStatus.EXHAUSTED.value

                # Carbon pricing
                if data.internal_carbon_price > Decimal("0"):
                    ab.carbon_cost = _round_val(actual * data.internal_carbon_price, 2)
                    if ab.variance_tco2e > Decimal("0"):
                        ab.overshoot_penalty = _round_val(
                            ab.variance_tco2e * data.internal_carbon_price * data.overshoot_penalty_rate, 2
                        )

        return budgets

    def _generate_rebalancing(
        self, data: BudgetAllocationInput,
        budgets: List[AnnualBudget], total_budget: Decimal,
    ) -> List[RebalancingRecommendation]:
        """Generate rebalancing recommendations for overshoot years."""
        recs: List[RebalancingRecommendation] = []

        for i, ab in enumerate(budgets):
            if ab.variance_tco2e > Decimal("0") and ab.actual_tco2e > Decimal("0"):
                remaining_years = len(budgets) - i - 1
                if remaining_years <= 0:
                    continue

                adjustment = _safe_divide(ab.variance_tco2e, _decimal(remaining_years))
                future_avg = Decimal("0")
                if i + 1 < len(budgets):
                    future_budgets = [b.budget_tco2e for b in budgets[i + 1:]]
                    future_avg = sum(future_budgets) / _decimal(len(future_budgets))

                new_annual = future_avg - adjustment
                adj_pct = _safe_pct(adjustment, future_avg) if future_avg > Decimal("0") else Decimal("0")

                recs.append(RebalancingRecommendation(
                    trigger_year=ab.year,
                    overshoot_tco2e=_round_val(ab.variance_tco2e, 2),
                    remaining_years=remaining_years,
                    adjusted_annual_budget_tco2e=_round_val(new_annual, 2),
                    adjustment_pct=_round_val(adj_pct, 1),
                    description=(
                        f"Overshoot of {_round_val(ab.variance_tco2e, 0)} tCO2e in {ab.year}. "
                        f"Reduce remaining annual budgets by {_round_val(adjustment, 0)} tCO2e/yr "
                        f"({_round_val(adj_pct, 1)}%) across {remaining_years} years."
                    ),
                ))

        return recs

    def _analyze_carbon_pricing(
        self, data: BudgetAllocationInput,
        budgets: List[AnnualBudget],
    ) -> CarbonPricingAnalysis:
        """Analyze internal carbon pricing impact."""
        price = data.internal_carbon_price
        total_value = Decimal("0")
        total_penalty = Decimal("0")
        annual_costs: List[Dict[str, Any]] = []

        for ab in budgets:
            cost = ab.budget_tco2e * price
            penalty = max(ab.variance_tco2e, Decimal("0")) * price * data.overshoot_penalty_rate if ab.actual_tco2e > Decimal("0") else Decimal("0")
            total_value += cost
            total_penalty += penalty
            annual_costs.append({
                "year": ab.year,
                "budget_cost": str(_round_val(cost, 2)),
                "overshoot_penalty": str(_round_val(penalty, 2)),
                "total_cost": str(_round_val(cost + penalty, 2)),
            })

        benchmarks = {}
        for name, bench_price in CARBON_PRICE_BENCHMARKS.items():
            if price >= bench_price:
                benchmarks[name] = f"Above ({price} vs {bench_price})"
            else:
                benchmarks[name] = f"Below ({price} vs {bench_price})"

        return CarbonPricingAnalysis(
            internal_price_per_tco2e=_round_val(price, 2),
            total_budget_value=_round_val(total_value, 2),
            total_overshoot_penalty=_round_val(total_penalty, 2),
            annual_carbon_costs=annual_costs,
            price_vs_benchmarks=benchmarks,
        )

    def _build_summary(
        self, data: BudgetAllocationInput,
        budgets: List[AnnualBudget], total_budget: Decimal,
    ) -> BudgetSummary:
        """Build budget summary statistics."""
        total_actual = sum(ab.actual_tco2e for ab in budgets)
        total_allocated = sum(ab.budget_tco2e for ab in budgets)
        overshoot = sum(max(ab.variance_tco2e, Decimal("0")) for ab in budgets if ab.actual_tco2e > Decimal("0"))
        undershoot = sum(abs(min(ab.variance_tco2e, Decimal("0"))) for ab in budgets if ab.actual_tco2e > Decimal("0"))
        years_over = sum(1 for ab in budgets if ab.status == BudgetStatus.OVERSHOOT.value)
        years_under = sum(1 for ab in budgets if ab.status == BudgetStatus.WITHIN_BUDGET.value and ab.actual_tco2e > Decimal("0"))

        exhaustion = 0
        for ab in budgets:
            if ab.remaining_budget_tco2e <= Decimal("0") and ab.actual_tco2e > Decimal("0"):
                exhaustion = ab.year
                break

        # Years remaining at current rate
        actuals_with_data = [ab for ab in budgets if ab.actual_tco2e > Decimal("0")]
        years_remaining = 0
        if actuals_with_data and len(actuals_with_data) >= 2:
            recent = actuals_with_data[-1]
            if recent.actual_tco2e > Decimal("0") and recent.remaining_budget_tco2e > Decimal("0"):
                years_remaining = int(float(_safe_divide(recent.remaining_budget_tco2e, recent.actual_tco2e)))

        return BudgetSummary(
            total_budget_tco2e=_round_val(total_budget, 2),
            total_allocated_tco2e=_round_val(total_allocated, 2),
            total_actual_tco2e=_round_val(total_actual, 2),
            total_overshoot_tco2e=_round_val(overshoot, 2),
            total_undershoot_tco2e=_round_val(undershoot, 2),
            budget_utilization_pct=_round_val(_safe_pct(total_actual, total_budget), 1),
            years_in_overshoot=years_over,
            years_in_undershoot=years_under,
            exhaustion_year=exhaustion,
            years_remaining_at_current_rate=years_remaining,
        )

    def _assess_data_quality(self, data: BudgetAllocationInput) -> str:
        score = 0
        if data.baseline_tco2e > Decimal("0"):
            score += 2
        if len(data.pathway_points) >= 3:
            score += 2
        if len(data.actual_emissions) >= 2:
            score += 2
        if data.internal_carbon_price > Decimal("0"):
            score += 1
        if data.entity_id:
            score += 1
        if data.total_carbon_budget_tco2e > Decimal("0"):
            score += 1
        if score >= 7:
            return DataQuality.HIGH.value
        elif score >= 4:
            return DataQuality.MEDIUM.value
        elif score >= 2:
            return DataQuality.LOW.value
        return DataQuality.ESTIMATED.value

    def _generate_recommendations(self, data, summary, rebalancing) -> List[str]:
        recs: List[str] = []
        if summary and summary.years_in_overshoot > 0:
            recs.append(f"{summary.years_in_overshoot} year(s) in budget overshoot. Implement corrective actions.")
        if summary and summary.years_remaining_at_current_rate > 0 and summary.years_remaining_at_current_rate < (data.target_year - data.baseline_year) // 2:
            recs.append(f"At current emission rate, budget exhausts in {summary.years_remaining_at_current_rate} years. Accelerate reductions.")
        if data.internal_carbon_price <= Decimal("0"):
            recs.append("Set an internal carbon price to create financial incentive for emissions reduction.")
        if rebalancing:
            recs.append(f"{len(rebalancing)} rebalancing recommendation(s) generated. Review and implement.")
        return recs

    def _generate_warnings(self, data, summary) -> List[str]:
        warns: List[str] = []
        if summary and summary.exhaustion_year > 0:
            warns.append(f"Carbon budget exhausted in {summary.exhaustion_year}.")
        if data.baseline_tco2e <= Decimal("0"):
            warns.append("Baseline emissions are zero. Cannot calculate budget.")
        return warns

    def get_carbon_price_benchmarks(self) -> Dict[str, str]:
        return {k: str(v) for k, v in CARBON_PRICE_BENCHMARKS.items()}

    def get_allocation_strategies(self) -> List[str]:
        return [s.value for s in AllocationStrategy]
