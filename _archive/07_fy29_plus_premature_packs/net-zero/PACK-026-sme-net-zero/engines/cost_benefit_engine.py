# -*- coding: utf-8 -*-
"""
CostBenefitEngine - PACK-026 SME Net Zero Pack Engine 6
=========================================================

Financial analysis engine for SME decarbonization actions with NPV,
IRR, simple/discounted payback, grant/incentive adjustment, and
conservative/optimistic scenario analysis.

Calculation Methodology:
    NPV (configurable horizon):
        npv = -net_capex + sum(annual_net_benefit / (1+r)^t)

    IRR (bisection method):
        Find r where NPV(r) = 0

    Simple payback:
        payback = net_capex / annual_net_benefit

    Discounted payback:
        First year t where cumulative discounted benefits >= net_capex

    Grant adjustment:
        net_capex = capex * (1 - grant_pct / 100)

    Carbon savings:
        carbon_value = annual_tco2e * carbon_price_usd

    Scenario analysis:
        conservative: +20% cost, -20% savings
        optimistic:   -20% cost, +20% savings

Regulatory References:
    - UK Green Finance Institute guidance for SMEs
    - EU Taxonomy delegated acts on CapEx
    - SBTi transition plan requirements
    - TCFD recommendations on transition planning

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Standard financial formulas (DCF, IRR bisection)
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-026 SME Net Zero Pack
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import RiskLevel

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
    numerator: Decimal, denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

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

class CostCategory(str, Enum):
    """Category of cost/investment."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    FLEET_ELECTRIFICATION = "fleet_electrification"
    BUILDING_UPGRADE = "building_upgrade"
    PROCESS_IMPROVEMENT = "process_improvement"
    BEHAVIORAL_CHANGE = "behavioral_change"
    PROCUREMENT_CHANGE = "procurement_change"
    WASTE_REDUCTION = "waste_reduction"
    OTHER = "other"

class ScenarioType(str, Enum):
    """Financial scenario type."""
    BASE = "base"
    CONSERVATIVE = "conservative"
    OPTIMISTIC = "optimistic"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DISCOUNT_RATE: Decimal = Decimal("0.08")
DEFAULT_HORIZON_YEARS: int = 10
SENSITIVITY_FACTOR: Decimal = Decimal("0.20")  # +/-20%

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class CostBenefitItem(BaseModel):
    """A single action for cost-benefit analysis.

    Attributes:
        item_id: Unique identifier.
        name: Action name.
        category: Cost category.
        capex_usd: Upfront capital expenditure.
        annual_opex_savings_usd: Annual operating cost savings.
        annual_revenue_increase_usd: Annual revenue increase (if any).
        annual_tco2e_reduction: Annual emissions reduction.
        grant_pct: Grant/incentive coverage (%).
        grant_name: Name of the grant program.
        useful_life_years: Expected useful life.
        implementation_months: Time to implement.
        maintenance_cost_annual_usd: Annual maintenance cost.
        carbon_price_usd_per_tco2e: Internal carbon price.
        residual_value_pct: Residual value at end of life (% of capex).
        notes: Optional notes.
    """
    item_id: str = Field(default_factory=_new_uuid)
    name: str = Field(..., min_length=1, max_length=300)
    category: CostCategory = Field(default=CostCategory.ENERGY_EFFICIENCY)
    capex_usd: Decimal = Field(..., ge=Decimal("0"))
    annual_opex_savings_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    annual_revenue_increase_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    annual_tco2e_reduction: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    grant_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100")
    )
    grant_name: str = Field(default="")
    useful_life_years: int = Field(default=10, ge=1, le=30)
    implementation_months: int = Field(default=3, ge=1, le=36)
    maintenance_cost_annual_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    carbon_price_usd_per_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    residual_value_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100")
    )
    notes: str = Field(default="", max_length=500)

    @field_validator("capex_usd")
    @classmethod
    def validate_capex(cls, v: Decimal) -> Decimal:
        if v > Decimal("10000000"):
            raise ValueError("CapEx exceeds SME reasonable range")
        return v

class CostBenefitInput(BaseModel):
    """Complete input for cost-benefit analysis.

    Attributes:
        entity_name: Company name.
        items: Actions to analyze.
        discount_rate: Discount rate (default 8%).
        analysis_horizon_years: Analysis horizon (default 10).
        inflation_rate: Annual inflation rate (default 2%).
    """
    entity_name: str = Field(..., min_length=1, max_length=300)
    items: List[CostBenefitItem] = Field(..., min_length=1, max_length=20)
    discount_rate: Decimal = Field(
        default=DEFAULT_DISCOUNT_RATE, ge=Decimal("0"), le=Decimal("0.30")
    )
    analysis_horizon_years: int = Field(
        default=DEFAULT_HORIZON_YEARS, ge=1, le=20
    )
    inflation_rate: Decimal = Field(
        default=Decimal("0.02"), ge=Decimal("0"), le=Decimal("0.20")
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class YearCashFlow(BaseModel):
    """Cash flow for a single year.

    Attributes:
        year: Year number (0 = investment).
        capex: Capital expenditure (year 0 only).
        opex_savings: Operating savings.
        revenue_increase: Revenue increase.
        maintenance: Maintenance cost.
        carbon_value: Carbon savings value.
        residual_value: Residual value (final year only).
        net_cash_flow: Net cash flow.
        discounted_cf: Discounted cash flow.
        cumulative_cf: Cumulative net cash flow.
        cumulative_dcf: Cumulative discounted cash flow.
    """
    year: int = Field(default=0)
    capex: Decimal = Field(default=Decimal("0"))
    opex_savings: Decimal = Field(default=Decimal("0"))
    revenue_increase: Decimal = Field(default=Decimal("0"))
    maintenance: Decimal = Field(default=Decimal("0"))
    carbon_value: Decimal = Field(default=Decimal("0"))
    residual_value: Decimal = Field(default=Decimal("0"))
    net_cash_flow: Decimal = Field(default=Decimal("0"))
    discounted_cf: Decimal = Field(default=Decimal("0"))
    cumulative_cf: Decimal = Field(default=Decimal("0"))
    cumulative_dcf: Decimal = Field(default=Decimal("0"))

class ScenarioAnalysis(BaseModel):
    """Financial analysis under a specific scenario.

    Attributes:
        scenario: Scenario type.
        npv_usd: Net present value.
        irr_pct: Internal rate of return.
        simple_payback_years: Simple payback.
        discounted_payback_years: Discounted payback.
        total_benefit_usd: Total nominal benefits over horizon.
        benefit_cost_ratio: Benefit to cost ratio.
    """
    scenario: str = Field(default="base")
    npv_usd: Decimal = Field(default=Decimal("0"))
    irr_pct: Decimal = Field(default=Decimal("0"))
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    discounted_payback_years: Decimal = Field(default=Decimal("0"))
    total_benefit_usd: Decimal = Field(default=Decimal("0"))
    benefit_cost_ratio: Decimal = Field(default=Decimal("0"))

class ItemAnalysis(BaseModel):
    """Complete cost-benefit analysis for a single item.

    Attributes:
        item_id: Item identifier.
        name: Action name.
        category: Cost category.
        net_capex_usd: CapEx after grants.
        grant_savings_usd: Amount saved through grants.
        annual_net_benefit_usd: Annual net benefit.
        scenarios: Analysis under different scenarios.
        cash_flows: Year-by-year cash flows.
        risk_level: Risk assessment.
        risk_factors: Identified risk factors.
        recommendation: Go/No-go recommendation.
    """
    item_id: str = Field(default="")
    name: str = Field(default="")
    category: str = Field(default="")
    net_capex_usd: Decimal = Field(default=Decimal("0"))
    grant_savings_usd: Decimal = Field(default=Decimal("0"))
    annual_net_benefit_usd: Decimal = Field(default=Decimal("0"))
    scenarios: List[ScenarioAnalysis] = Field(default_factory=list)
    cash_flows: List[YearCashFlow] = Field(default_factory=list)
    risk_level: str = Field(default="medium")
    risk_factors: List[str] = Field(default_factory=list)
    recommendation: str = Field(default="")

class PortfolioSummary(BaseModel):
    """Portfolio-level cost-benefit summary.

    Attributes:
        total_items: Number of items analyzed.
        total_capex_usd: Total gross CapEx.
        total_net_capex_usd: Total CapEx after grants.
        total_grant_savings_usd: Total grant savings.
        portfolio_npv_usd: Portfolio NPV (base scenario).
        portfolio_irr_pct: Weighted average IRR.
        total_annual_savings_usd: Combined annual savings.
        total_annual_tco2e_reduction: Combined annual reduction.
        avg_payback_years: Average payback period.
        go_count: Number of recommended items.
        no_go_count: Number of not recommended items.
    """
    total_items: int = Field(default=0)
    total_capex_usd: Decimal = Field(default=Decimal("0"))
    total_net_capex_usd: Decimal = Field(default=Decimal("0"))
    total_grant_savings_usd: Decimal = Field(default=Decimal("0"))
    portfolio_npv_usd: Decimal = Field(default=Decimal("0"))
    portfolio_irr_pct: Decimal = Field(default=Decimal("0"))
    total_annual_savings_usd: Decimal = Field(default=Decimal("0"))
    total_annual_tco2e_reduction: Decimal = Field(default=Decimal("0"))
    avg_payback_years: Decimal = Field(default=Decimal("0"))
    go_count: int = Field(default=0)
    no_go_count: int = Field(default=0)

class CostBenefitResult(BaseModel):
    """Complete cost-benefit analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        entity_name: Company name.
        items: Individual item analyses.
        portfolio: Portfolio summary.
        discount_rate_used: Discount rate used.
        horizon_years_used: Analysis horizon used.
        processing_time_ms: Calculation time (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")

    items: List[ItemAnalysis] = Field(default_factory=list)
    portfolio: PortfolioSummary = Field(default_factory=PortfolioSummary)
    discount_rate_used: Decimal = Field(default=DEFAULT_DISCOUNT_RATE)
    horizon_years_used: int = Field(default=DEFAULT_HORIZON_YEARS)

    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CostBenefitEngine:
    """Financial cost-benefit analysis engine for SME decarbonization.

    Computes NPV, IRR, payback, and scenario analysis for each action.
    Supports grant/incentive adjustments, carbon pricing, and inflation.

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = CostBenefitEngine()
        result = engine.calculate(cost_benefit_input)
        for item in result.items:
            print(f"{item.name}: NPV=${item.scenarios[0].npv_usd}, rec={item.recommendation}")
    """

    engine_version: str = _MODULE_VERSION

    def calculate(self, data: CostBenefitInput) -> CostBenefitResult:
        """Run cost-benefit analysis for all items.

        Args:
            data: Validated cost-benefit input.

        Returns:
            CostBenefitResult with item-level and portfolio analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Cost-Benefit: entity=%s, items=%d, rate=%.1f%%, horizon=%d yrs",
            data.entity_name, len(data.items),
            float(data.discount_rate * Decimal("100")),
            data.analysis_horizon_years,
        )

        item_analyses: List[ItemAnalysis] = []
        total_capex = Decimal("0")
        total_net_capex = Decimal("0")
        total_grant_savings = Decimal("0")
        total_annual_savings = Decimal("0")
        total_tco2e = Decimal("0")
        total_npv = Decimal("0")
        go_count = 0
        payback_sum = Decimal("0")
        payback_count = 0

        for item in data.items:
            analysis = self._analyze_item(
                item, data.discount_rate,
                data.analysis_horizon_years, data.inflation_rate,
            )
            item_analyses.append(analysis)

            total_capex += item.capex_usd
            total_net_capex += analysis.net_capex_usd
            total_grant_savings += analysis.grant_savings_usd
            total_annual_savings += analysis.annual_net_benefit_usd
            total_tco2e += item.annual_tco2e_reduction

            # Get base scenario NPV
            base_scenario = next(
                (s for s in analysis.scenarios if s.scenario == "base"), None
            )
            if base_scenario:
                total_npv += base_scenario.npv_usd
                if base_scenario.simple_payback_years > Decimal("0"):
                    payback_sum += base_scenario.simple_payback_years
                    payback_count += 1

            if analysis.recommendation.startswith("GO"):
                go_count += 1

        # Portfolio IRR (weighted by capex)
        portfolio_irr = Decimal("0")
        if total_net_capex > Decimal("0") and total_annual_savings > Decimal("0"):
            portfolio_irr = self._compute_irr(
                total_net_capex, total_annual_savings, data.analysis_horizon_years
            )

        avg_payback = _safe_divide(
            payback_sum, _decimal(payback_count) if payback_count > 0 else Decimal("1")
        )

        portfolio = PortfolioSummary(
            total_items=len(data.items),
            total_capex_usd=_round_val(total_capex, 2),
            total_net_capex_usd=_round_val(total_net_capex, 2),
            total_grant_savings_usd=_round_val(total_grant_savings, 2),
            portfolio_npv_usd=_round_val(total_npv, 2),
            portfolio_irr_pct=portfolio_irr,
            total_annual_savings_usd=_round_val(total_annual_savings, 2),
            total_annual_tco2e_reduction=_round_val(total_tco2e),
            avg_payback_years=_round_val(avg_payback, 1),
            go_count=go_count,
            no_go_count=len(data.items) - go_count,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = CostBenefitResult(
            entity_name=data.entity_name,
            items=item_analyses,
            portfolio=portfolio,
            discount_rate_used=data.discount_rate,
            horizon_years_used=data.analysis_horizon_years,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Cost-Benefit complete: %d items, portfolio NPV=$%.0f, hash=%s",
            len(data.items), float(total_npv), result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Item Analysis                                                        #
    # ------------------------------------------------------------------ #

    def _analyze_item(
        self,
        item: CostBenefitItem,
        discount_rate: Decimal,
        horizon: int,
        inflation_rate: Decimal,
    ) -> ItemAnalysis:
        """Analyze a single cost-benefit item.

        Args:
            item: Action to analyze.
            discount_rate: Discount rate.
            horizon: Analysis horizon.
            inflation_rate: Annual inflation rate.

        Returns:
            ItemAnalysis with full financial breakdown.
        """
        net_capex = _round_val(
            item.capex_usd * (Decimal("100") - item.grant_pct) / Decimal("100"), 2
        )
        grant_savings = _round_val(item.capex_usd - net_capex, 2)

        # Annual net benefit
        carbon_value = item.annual_tco2e_reduction * item.carbon_price_usd_per_tco2e
        annual_net = (
            item.annual_opex_savings_usd
            + item.annual_revenue_increase_usd
            + carbon_value
            - item.maintenance_cost_annual_usd
        )

        # Cash flows
        cash_flows = self._build_cash_flows(
            net_capex, item.annual_opex_savings_usd,
            item.annual_revenue_increase_usd,
            item.maintenance_cost_annual_usd, carbon_value,
            item.residual_value_pct, item.capex_usd,
            discount_rate, horizon, inflation_rate,
        )

        # Scenario analysis
        scenarios = []
        for scenario_type, cost_mult, savings_mult in [
            (ScenarioType.BASE, Decimal("1.00"), Decimal("1.00")),
            (ScenarioType.CONSERVATIVE, Decimal("1.20"), Decimal("0.80")),
            (ScenarioType.OPTIMISTIC, Decimal("0.80"), Decimal("1.20")),
        ]:
            adj_capex = net_capex * cost_mult
            adj_annual = annual_net * savings_mult

            npv = self._compute_npv(adj_capex, adj_annual, discount_rate, horizon)
            irr = self._compute_irr(adj_capex, adj_annual, horizon)
            simple_pb = Decimal("0")
            if adj_annual > Decimal("0") and adj_capex > Decimal("0"):
                simple_pb = _round_val(_safe_divide(adj_capex, adj_annual), 2)
            disc_pb = self._discounted_payback(adj_capex, adj_annual, discount_rate, horizon)

            total_benefit = adj_annual * _decimal(horizon)
            bcr = _safe_divide(total_benefit, adj_capex) if adj_capex > Decimal("0") else Decimal("999")

            scenarios.append(ScenarioAnalysis(
                scenario=scenario_type.value,
                npv_usd=_round_val(npv, 2),
                irr_pct=irr,
                simple_payback_years=simple_pb,
                discounted_payback_years=disc_pb,
                total_benefit_usd=_round_val(total_benefit, 2),
                benefit_cost_ratio=_round_val(bcr, 2),
            ))

        # Risk assessment
        risk_level, risk_factors = self._assess_risk(item, scenarios)

        # Recommendation
        base = scenarios[0]
        conservative = scenarios[1]
        if base.npv_usd > Decimal("0") and conservative.npv_usd > Decimal("0"):
            recommendation = "GO - Positive NPV in all scenarios"
        elif base.npv_usd > Decimal("0"):
            recommendation = "GO (conditional) - Positive NPV in base case, negative in conservative"
        elif net_capex == Decimal("0"):
            recommendation = "GO - Zero cost action"
        else:
            recommendation = "NO-GO - Negative NPV in base case"

        return ItemAnalysis(
            item_id=item.item_id,
            name=item.name,
            category=item.category.value,
            net_capex_usd=net_capex,
            grant_savings_usd=grant_savings,
            annual_net_benefit_usd=_round_val(annual_net, 2),
            scenarios=scenarios,
            cash_flows=cash_flows,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------ #
    # Financial Calculations                                               #
    # ------------------------------------------------------------------ #

    def _build_cash_flows(
        self,
        net_capex: Decimal,
        annual_savings: Decimal,
        annual_revenue: Decimal,
        annual_maintenance: Decimal,
        annual_carbon: Decimal,
        residual_pct: Decimal,
        gross_capex: Decimal,
        rate: Decimal,
        horizon: int,
        inflation: Decimal,
    ) -> List[YearCashFlow]:
        """Build year-by-year cash flow table.

        Args:
            net_capex: Net capital expenditure.
            annual_savings: Annual opex savings.
            annual_revenue: Annual revenue increase.
            annual_maintenance: Annual maintenance cost.
            annual_carbon: Annual carbon value.
            residual_pct: Residual value percentage.
            gross_capex: Gross capex for residual calculation.
            rate: Discount rate.
            horizon: Analysis horizon.
            inflation: Annual inflation rate.

        Returns:
            List of YearCashFlow entries.
        """
        flows: List[YearCashFlow] = []
        cumulative_cf = Decimal("0")
        cumulative_dcf = Decimal("0")

        # Year 0: investment
        net_cf_y0 = -net_capex
        cumulative_cf = net_cf_y0
        cumulative_dcf = net_cf_y0

        flows.append(YearCashFlow(
            year=0,
            capex=net_capex,
            net_cash_flow=net_cf_y0,
            discounted_cf=net_cf_y0,
            cumulative_cf=cumulative_cf,
            cumulative_dcf=cumulative_dcf,
        ))

        for t in range(1, horizon + 1):
            # Apply inflation
            inflation_mult = (Decimal("1") + inflation) ** (t - 1)
            savings = _round_val(annual_savings * inflation_mult, 2)
            revenue = _round_val(annual_revenue * inflation_mult, 2)
            maintenance = _round_val(annual_maintenance * inflation_mult, 2)
            carbon = _round_val(annual_carbon, 2)  # carbon value not inflated

            residual = Decimal("0")
            if t == horizon and residual_pct > Decimal("0"):
                residual = _round_val(
                    gross_capex * residual_pct / Decimal("100"), 2
                )

            net_cf = savings + revenue + carbon - maintenance + residual
            disc_factor = (Decimal("1") + rate) ** t
            disc_cf = _round_val(_safe_divide(net_cf, disc_factor), 2)

            cumulative_cf += net_cf
            cumulative_dcf += disc_cf

            flows.append(YearCashFlow(
                year=t,
                opex_savings=savings,
                revenue_increase=revenue,
                maintenance=maintenance,
                carbon_value=carbon,
                residual_value=residual,
                net_cash_flow=_round_val(net_cf, 2),
                discounted_cf=disc_cf,
                cumulative_cf=_round_val(cumulative_cf, 2),
                cumulative_dcf=_round_val(cumulative_dcf, 2),
            ))

        return flows

    def _compute_npv(
        self, capex: Decimal, annual_cf: Decimal,
        rate: Decimal, years: int,
    ) -> Decimal:
        npv = -capex
        for t in range(1, years + 1):
            npv += _safe_divide(annual_cf, (Decimal("1") + rate) ** t)
        return npv

    def _compute_irr(
        self, capex: Decimal, annual_cf: Decimal, years: int,
    ) -> Decimal:
        if capex == Decimal("0"):
            return Decimal("999") if annual_cf > Decimal("0") else Decimal("0")
        if annual_cf <= Decimal("0"):
            return Decimal("0")

        low, high = Decimal("-0.50"), Decimal("5.00")
        for _ in range(100):
            mid = (low + high) / Decimal("2")
            npv = self._compute_npv(capex, annual_cf, mid, years)
            if abs(npv) < Decimal("0.01"):
                return _round_val(mid * Decimal("100"), 2)
            elif npv > Decimal("0"):
                low = mid
            else:
                high = mid
        return _round_val(((low + high) / Decimal("2")) * Decimal("100"), 2)

    def _discounted_payback(
        self, capex: Decimal, annual_cf: Decimal,
        rate: Decimal, max_years: int,
    ) -> Decimal:
        if capex == Decimal("0"):
            return Decimal("0")
        if annual_cf <= Decimal("0"):
            return Decimal("0")
        cumulative = Decimal("0")
        for t in range(1, max_years + 1):
            cumulative += _safe_divide(annual_cf, (Decimal("1") + rate) ** t)
            if cumulative >= capex:
                return _round_val(_decimal(t), 1)
        return Decimal("0")

    def _assess_risk(
        self,
        item: CostBenefitItem,
        scenarios: List[ScenarioAnalysis],
    ) -> tuple[str, List[str]]:
        """Assess risk level and identify risk factors.

        Args:
            item: The action being analyzed.
            scenarios: Scenario analysis results.

        Returns:
            Tuple of (risk_level, risk_factors).
        """
        factors: List[str] = []

        # Check if conservative scenario is negative NPV
        conservative = next(
            (s for s in scenarios if s.scenario == "conservative"), None
        )
        if conservative and conservative.npv_usd < Decimal("0"):
            factors.append("Negative NPV under conservative scenario")

        # High capex relative to savings
        base = next((s for s in scenarios if s.scenario == "base"), None)
        if base and base.simple_payback_years > Decimal("5"):
            factors.append("Payback period exceeds 5 years")

        # Technology risk
        if item.implementation_months > 12:
            factors.append("Implementation timeline exceeds 12 months")

        # Grant dependency
        if item.grant_pct > Decimal("50"):
            factors.append(f"High grant dependency ({float(item.grant_pct):.0f}%)")

        level = RiskLevel.LOW.value
        if len(factors) >= 3:
            level = RiskLevel.HIGH.value
        elif len(factors) >= 1:
            level = RiskLevel.MEDIUM.value

        return level, factors
