# -*- coding: utf-8 -*-
"""
BudgetSystemBridge - Internal Carbon Budget System Integration for PACK-029
=============================================================================

Enterprise bridge for integrating with internal carbon pricing systems,
carbon budget allocation by year/scope/category, shadow carbon price
application, carbon fee/levy calculation, financial impact analysis
(P&L, NPV), and budget rebalancing triggers.

Integration Points:
    - Carbon Budget Allocation: Year/scope/category carbon budgets
    - Shadow Carbon Price: Internal carbon price application
    - Carbon Fee/Levy: Carbon fee calculation for budget enforcement
    - Financial Impact: P&L impact and NPV analysis
    - Budget Rebalancing: Triggers for budget reallocation
    - Interim Target Alignment: Budget aligned to interim milestones

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


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
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BudgetScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    ALL_SCOPES = "all_scopes"


class CarbonPriceType(str, Enum):
    SHADOW = "shadow"           # Internal shadow price
    FEE = "fee"                 # Internal carbon fee
    LEVY = "levy"               # Budget-based levy
    MARKET = "market"           # External market price (EU ETS, etc.)
    BLENDED = "blended"         # Blend of internal + external


class BudgetStatus(str, Enum):
    UNDER_BUDGET = "under_budget"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OVER_BUDGET = "over_budget"
    CRITICAL = "critical"


class RebalanceTrigger(str, Enum):
    QUARTERLY_OVERRUN = "quarterly_overrun"
    ANNUAL_TRAJECTORY = "annual_trajectory"
    INITIATIVE_DELAY = "initiative_delay"
    SCOPE_SHIFT = "scope_shift"
    STRUCTURAL_CHANGE = "structural_change"
    POLICY_CHANGE = "policy_change"


# ---------------------------------------------------------------------------
# Carbon Price Schedules
# ---------------------------------------------------------------------------

DEFAULT_SHADOW_PRICE_SCHEDULE: Dict[int, float] = {
    2023: 50, 2024: 60, 2025: 75, 2026: 90, 2027: 110,
    2028: 130, 2029: 150, 2030: 175, 2035: 250, 2040: 300, 2050: 250,
}

EU_ETS_PRICE_ASSUMPTIONS: Dict[int, float] = {
    2023: 85, 2024: 70, 2025: 80, 2026: 90, 2027: 100,
    2028: 110, 2029: 120, 2030: 140, 2035: 200, 2040: 250, 2050: 200,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class BudgetSystemConfig(BaseModel):
    """Configuration for the budget system bridge."""
    pack_id: str = Field(default="PACK-029")
    organization_id: str = Field(default="")
    currency: str = Field(default="USD")
    base_year: int = Field(default=2023, ge=2015, le=2025)
    budget_start_year: int = Field(default=2025, ge=2020, le=2035)
    budget_end_year: int = Field(default=2030, ge=2025, le=2060)
    carbon_price_type: CarbonPriceType = Field(default=CarbonPriceType.SHADOW)
    shadow_price_schedule: Dict[int, float] = Field(default_factory=lambda: dict(DEFAULT_SHADOW_PRICE_SCHEDULE))
    discount_rate_pct: float = Field(default=8.0, ge=0.0, le=20.0)
    rebalance_threshold_pct: float = Field(default=10.0, ge=1.0, le=50.0)
    enable_provenance: bool = Field(default=True)


class AnnualCarbonBudget(BaseModel):
    """Carbon budget allocation for a single year."""
    budget_id: str = Field(default_factory=_new_uuid)
    year: int = Field(default=2025)
    scope1_budget_tco2e: float = Field(default=0.0)
    scope2_budget_tco2e: float = Field(default=0.0)
    scope3_budget_tco2e: float = Field(default=0.0)
    total_budget_tco2e: float = Field(default=0.0)
    scope1_actual_tco2e: float = Field(default=0.0)
    scope2_actual_tco2e: float = Field(default=0.0)
    scope3_actual_tco2e: float = Field(default=0.0)
    total_actual_tco2e: float = Field(default=0.0)
    variance_tco2e: float = Field(default=0.0)
    variance_pct: float = Field(default=0.0)
    budget_status: BudgetStatus = Field(default=BudgetStatus.ON_TRACK)
    carbon_price_usd: float = Field(default=0.0)
    carbon_cost_usd: float = Field(default=0.0)
    scope3_by_category: Dict[int, float] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class CarbonPriceResult(BaseModel):
    """Carbon price application result."""
    price_id: str = Field(default_factory=_new_uuid)
    year: int = Field(default=2025)
    price_type: CarbonPriceType = Field(default=CarbonPriceType.SHADOW)
    price_per_tco2e_usd: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    total_carbon_cost_usd: float = Field(default=0.0)
    by_scope: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    revenue_impact_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class FinancialImpact(BaseModel):
    """Financial impact analysis result."""
    impact_id: str = Field(default_factory=_new_uuid)
    analysis_period_start: int = Field(default=2025)
    analysis_period_end: int = Field(default=2030)
    total_carbon_cost_usd: float = Field(default=0.0)
    total_abatement_cost_usd: float = Field(default=0.0)
    total_savings_usd: float = Field(default=0.0)
    net_cost_usd: float = Field(default=0.0)
    npv_usd: float = Field(default=0.0)
    irr_pct: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)
    annual_pl_impact: Dict[int, float] = Field(default_factory=dict)
    carbon_price_trajectory: Dict[int, float] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class RebalanceRecommendation(BaseModel):
    """Budget rebalance recommendation."""
    recommendation_id: str = Field(default_factory=_new_uuid)
    trigger: RebalanceTrigger = Field(default=RebalanceTrigger.QUARTERLY_OVERRUN)
    trigger_description: str = Field(default="")
    severity: str = Field(default="medium")
    affected_scope: str = Field(default="")
    current_budget_tco2e: float = Field(default=0.0)
    recommended_budget_tco2e: float = Field(default=0.0)
    adjustment_tco2e: float = Field(default=0.0)
    rationale: str = Field(default="")
    financial_impact_usd: float = Field(default=0.0)


class BudgetSystemResult(BaseModel):
    """Complete budget system result."""
    result_id: str = Field(default_factory=_new_uuid)
    annual_budgets: List[AnnualCarbonBudget] = Field(default_factory=list)
    carbon_price: Optional[CarbonPriceResult] = Field(None)
    financial_impact: Optional[FinancialImpact] = Field(None)
    rebalance_recommendations: List[RebalanceRecommendation] = Field(default_factory=list)
    total_budget_tco2e: float = Field(default=0.0)
    total_actual_tco2e: float = Field(default=0.0)
    overall_status: BudgetStatus = Field(default=BudgetStatus.ON_TRACK)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# BudgetSystemBridge
# ---------------------------------------------------------------------------


class BudgetSystemBridge:
    """Internal carbon budget system bridge for PACK-029.

    Manages carbon budget allocation, shadow carbon pricing, carbon
    fee/levy calculations, financial impact analysis, and budget
    rebalancing aligned to interim targets.

    Example:
        >>> bridge = BudgetSystemBridge(BudgetSystemConfig(
        ...     budget_start_year=2025, budget_end_year=2030,
        ... ))
        >>> budgets = await bridge.allocate_annual_budgets(baseline, pathway_rate)
        >>> price = await bridge.apply_carbon_price(2025, emissions)
        >>> impact = await bridge.analyze_financial_impact(budgets)
    """

    def __init__(self, config: Optional[BudgetSystemConfig] = None) -> None:
        self.config = config or BudgetSystemConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._budget_cache: List[AnnualCarbonBudget] = []

        self.logger.info(
            "BudgetSystemBridge (PACK-029) initialized: %d-%d, price=%s",
            self.config.budget_start_year, self.config.budget_end_year,
            self.config.carbon_price_type.value,
        )

    async def allocate_annual_budgets(
        self,
        base_year_emissions: Dict[str, float],
        annual_reduction_rate_pct: float = 4.2,
        scope3_rate_pct: float = 2.5,
    ) -> List[AnnualCarbonBudget]:
        """Allocate annual carbon budgets aligned to interim targets."""
        s1_base = base_year_emissions.get("scope1_tco2e", 50000.0)
        s2_base = base_year_emissions.get("scope2_tco2e", 25000.0)
        s3_base = base_year_emissions.get("scope3_tco2e", 120000.0)

        budgets: List[AnnualCarbonBudget] = []

        for year in range(self.config.budget_start_year, self.config.budget_end_year + 1):
            years_from_base = year - self.config.base_year
            s12_factor = max(0.0, 1.0 - (annual_reduction_rate_pct / 100.0) * years_from_base)
            s3_factor = max(0.0, 1.0 - (scope3_rate_pct / 100.0) * years_from_base)

            s1_budget = round(s1_base * s12_factor, 2)
            s2_budget = round(s2_base * s12_factor, 2)
            s3_budget = round(s3_base * s3_factor, 2)
            total_budget = s1_budget + s2_budget + s3_budget

            carbon_price = self._get_carbon_price(year)
            carbon_cost = total_budget * carbon_price

            budget = AnnualCarbonBudget(
                year=year,
                scope1_budget_tco2e=s1_budget,
                scope2_budget_tco2e=s2_budget,
                scope3_budget_tco2e=s3_budget,
                total_budget_tco2e=round(total_budget, 2),
                carbon_price_usd=carbon_price,
                carbon_cost_usd=round(carbon_cost, 2),
                budget_status=BudgetStatus.ON_TRACK,
            )

            if self.config.enable_provenance:
                budget.provenance_hash = _compute_hash(budget)
            budgets.append(budget)

        self._budget_cache = budgets
        self.logger.info(
            "Annual budgets allocated: %d years, total=%.0f tCO2e, "
            "S12 rate=%.1f%%/yr, S3 rate=%.1f%%/yr",
            len(budgets), sum(b.total_budget_tco2e for b in budgets),
            annual_reduction_rate_pct, scope3_rate_pct,
        )
        return budgets

    async def update_actuals(
        self,
        year: int,
        actual_emissions: Dict[str, float],
    ) -> AnnualCarbonBudget:
        """Update actual emissions for a budget year."""
        budget = next((b for b in self._budget_cache if b.year == year), None)
        if not budget:
            budget = AnnualCarbonBudget(year=year)
            self._budget_cache.append(budget)

        budget.scope1_actual_tco2e = actual_emissions.get("scope1_tco2e", 0.0)
        budget.scope2_actual_tco2e = actual_emissions.get("scope2_tco2e", 0.0)
        budget.scope3_actual_tco2e = actual_emissions.get("scope3_tco2e", 0.0)
        budget.total_actual_tco2e = (
            budget.scope1_actual_tco2e + budget.scope2_actual_tco2e + budget.scope3_actual_tco2e
        )

        budget.variance_tco2e = round(budget.total_actual_tco2e - budget.total_budget_tco2e, 2)
        budget.variance_pct = round(
            (budget.variance_tco2e / max(budget.total_budget_tco2e, 1.0)) * 100.0, 2
        )

        # Update status
        if budget.variance_pct <= -5.0:
            budget.budget_status = BudgetStatus.UNDER_BUDGET
        elif budget.variance_pct <= 5.0:
            budget.budget_status = BudgetStatus.ON_TRACK
        elif budget.variance_pct <= self.config.rebalance_threshold_pct:
            budget.budget_status = BudgetStatus.AT_RISK
        elif budget.variance_pct <= 25.0:
            budget.budget_status = BudgetStatus.OVER_BUDGET
        else:
            budget.budget_status = BudgetStatus.CRITICAL

        budget.carbon_cost_usd = round(
            budget.total_actual_tco2e * budget.carbon_price_usd, 2
        )

        if self.config.enable_provenance:
            budget.provenance_hash = _compute_hash(budget)

        self.logger.info(
            "Actuals updated: year=%d, budget=%.0f, actual=%.0f, "
            "variance=%.1f%%, status=%s",
            year, budget.total_budget_tco2e, budget.total_actual_tco2e,
            budget.variance_pct, budget.budget_status.value,
        )
        return budget

    async def apply_carbon_price(
        self,
        year: int,
        emissions: Dict[str, float],
        revenue_usd: float = 0.0,
    ) -> CarbonPriceResult:
        """Apply carbon price to emissions."""
        price = self._get_carbon_price(year)
        s1 = emissions.get("scope1_tco2e", 0.0)
        s2 = emissions.get("scope2_tco2e", 0.0)
        s3 = emissions.get("scope3_tco2e", 0.0)
        total = s1 + s2 + s3

        total_cost = total * price
        revenue_impact = (total_cost / max(revenue_usd, 1.0)) * 100.0 if revenue_usd > 0 else 0.0

        result = CarbonPriceResult(
            year=year,
            price_type=self.config.carbon_price_type,
            price_per_tco2e_usd=price,
            total_emissions_tco2e=round(total, 2),
            total_carbon_cost_usd=round(total_cost, 2),
            by_scope={
                "scope_1": {"emissions_tco2e": round(s1, 2), "cost_usd": round(s1 * price, 2)},
                "scope_2": {"emissions_tco2e": round(s2, 2), "cost_usd": round(s2 * price, 2)},
                "scope_3": {"emissions_tco2e": round(s3, 2), "cost_usd": round(s3 * price, 2)},
            },
            revenue_impact_pct=round(revenue_impact, 4),
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Carbon price applied: year=%d, price=$%.0f/tCO2e, "
            "total=$%.0f, revenue_impact=%.2f%%",
            year, price, total_cost, revenue_impact,
        )
        return result

    async def analyze_financial_impact(
        self,
        budgets: Optional[List[AnnualCarbonBudget]] = None,
        abatement_costs: Optional[Dict[int, float]] = None,
        savings: Optional[Dict[int, float]] = None,
    ) -> FinancialImpact:
        """Analyze financial impact of carbon budgets and abatement."""
        budget_list = budgets or self._budget_cache
        if not budget_list:
            return FinancialImpact()

        abatement = abatement_costs or {}
        save = savings or {}
        discount = self.config.discount_rate_pct / 100.0

        total_carbon_cost = 0.0
        total_abatement = 0.0
        total_savings = 0.0
        annual_pl: Dict[int, float] = {}
        price_trajectory: Dict[int, float] = {}
        npv = 0.0

        for budget in budget_list:
            year = budget.year
            carbon_cost = budget.carbon_cost_usd
            abate_cost = abatement.get(year, carbon_cost * 0.15)
            year_savings = save.get(year, abate_cost * 0.3)

            net = carbon_cost + abate_cost - year_savings
            annual_pl[year] = round(net, 2)
            price_trajectory[year] = budget.carbon_price_usd

            total_carbon_cost += carbon_cost
            total_abatement += abate_cost
            total_savings += year_savings

            # NPV calculation
            years_from_start = year - self.config.budget_start_year
            if years_from_start >= 0:
                npv += net / ((1 + discount) ** years_from_start)

        net_cost = total_carbon_cost + total_abatement - total_savings

        # Simple payback
        avg_annual_savings = total_savings / max(len(budget_list), 1)
        avg_annual_abate = total_abatement / max(len(budget_list), 1)
        payback = avg_annual_abate / max(avg_annual_savings, 1.0) if avg_annual_savings > 0 else 0

        impact = FinancialImpact(
            analysis_period_start=self.config.budget_start_year,
            analysis_period_end=self.config.budget_end_year,
            total_carbon_cost_usd=round(total_carbon_cost, 2),
            total_abatement_cost_usd=round(total_abatement, 2),
            total_savings_usd=round(total_savings, 2),
            net_cost_usd=round(net_cost, 2),
            npv_usd=round(npv, 2),
            payback_years=round(payback, 2),
            annual_pl_impact=annual_pl,
            carbon_price_trajectory=price_trajectory,
        )

        if self.config.enable_provenance:
            impact.provenance_hash = _compute_hash(impact)

        self.logger.info(
            "Financial impact: carbon=$%.0f, abatement=$%.0f, savings=$%.0f, "
            "net=$%.0f, NPV=$%.0f",
            total_carbon_cost, total_abatement, total_savings, net_cost, npv,
        )
        return impact

    async def check_rebalance_triggers(self) -> List[RebalanceRecommendation]:
        """Check for budget rebalance triggers."""
        recommendations: List[RebalanceRecommendation] = []

        for budget in self._budget_cache:
            if budget.total_actual_tco2e == 0:
                continue

            # Quarterly overrun
            if budget.variance_pct > self.config.rebalance_threshold_pct:
                adjustment = budget.variance_tco2e
                recommendations.append(RebalanceRecommendation(
                    trigger=RebalanceTrigger.QUARTERLY_OVERRUN,
                    trigger_description=f"Year {budget.year} over budget by {budget.variance_pct:.1f}%",
                    severity="high" if budget.variance_pct > 20 else "medium",
                    affected_scope="all_scopes",
                    current_budget_tco2e=budget.total_budget_tco2e,
                    recommended_budget_tco2e=round(budget.total_actual_tco2e * 0.95, 2),
                    adjustment_tco2e=round(adjustment, 2),
                    rationale="Accelerate reduction initiatives to return to budget trajectory",
                    financial_impact_usd=round(adjustment * budget.carbon_price_usd, 2),
                ))

            # Scope shift
            s1_var = abs(budget.scope1_actual_tco2e - budget.scope1_budget_tco2e)
            s2_var = abs(budget.scope2_actual_tco2e - budget.scope2_budget_tco2e)
            if s1_var > budget.scope1_budget_tco2e * 0.15 or s2_var > budget.scope2_budget_tco2e * 0.15:
                recommendations.append(RebalanceRecommendation(
                    trigger=RebalanceTrigger.SCOPE_SHIFT,
                    trigger_description=f"Year {budget.year}: scope-level variance >15%",
                    severity="medium",
                    affected_scope="scope_1_2",
                    current_budget_tco2e=budget.scope1_budget_tco2e + budget.scope2_budget_tco2e,
                    recommended_budget_tco2e=round(budget.scope1_actual_tco2e + budget.scope2_actual_tco2e, 2),
                    adjustment_tco2e=round(s1_var + s2_var, 2),
                    rationale="Rebalance between Scope 1 and Scope 2 budgets",
                ))

        self.logger.info("Rebalance check: %d triggers found", len(recommendations))
        return recommendations

    async def get_budget_dashboard(self) -> Dict[str, Any]:
        """Get complete budget dashboard data."""
        budgets = self._budget_cache

        total_budget = sum(b.total_budget_tco2e for b in budgets)
        total_actual = sum(b.total_actual_tco2e for b in budgets if b.total_actual_tco2e > 0)
        total_cost = sum(b.carbon_cost_usd for b in budgets)

        by_year = [
            {
                "year": b.year,
                "budget_tco2e": b.total_budget_tco2e,
                "actual_tco2e": b.total_actual_tco2e,
                "variance_pct": b.variance_pct,
                "status": b.budget_status.value,
                "carbon_price": b.carbon_price_usd,
                "carbon_cost": b.carbon_cost_usd,
            }
            for b in budgets
        ]

        statuses = [b.budget_status.value for b in budgets if b.total_actual_tco2e > 0]
        if "critical" in statuses:
            overall = BudgetStatus.CRITICAL
        elif "over_budget" in statuses:
            overall = BudgetStatus.OVER_BUDGET
        elif "at_risk" in statuses:
            overall = BudgetStatus.AT_RISK
        elif "under_budget" in statuses or "on_track" in statuses:
            overall = BudgetStatus.ON_TRACK
        else:
            overall = BudgetStatus.ON_TRACK

        return {
            "total_budget_tco2e": round(total_budget, 2),
            "total_actual_tco2e": round(total_actual, 2),
            "total_carbon_cost_usd": round(total_cost, 2),
            "years": len(budgets),
            "by_year": by_year,
            "overall_status": overall.value,
        }

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "budget_period": f"{self.config.budget_start_year}-{self.config.budget_end_year}",
            "carbon_price_type": self.config.carbon_price_type.value,
            "budgets_allocated": len(self._budget_cache),
            "currency": self.config.currency,
        }

    def _get_carbon_price(self, year: int) -> float:
        """Get carbon price for a specific year (interpolated)."""
        schedule = self.config.shadow_price_schedule
        if not schedule:
            return 75.0
        if year in schedule:
            return schedule[year]
        years = sorted(schedule.keys())
        if year <= years[0]:
            return schedule[years[0]]
        if year >= years[-1]:
            return schedule[years[-1]]
        for i in range(len(years) - 1):
            if years[i] <= year <= years[i + 1]:
                frac = (year - years[i]) / (years[i + 1] - years[i])
                return schedule[years[i]] + frac * (schedule[years[i + 1]] - schedule[years[i]])
        return 75.0
