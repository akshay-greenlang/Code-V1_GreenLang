# -*- coding: utf-8 -*-
"""
ActionPrioritizationEngine - PACK-026 SME Net Zero Pack Engine 5
=================================================================

MACC-lite (no Monte Carlo) action prioritization with NPV, IRR,
payback analysis, and phased roadmap generation for SMEs.

Ranks up to 10 decarbonization actions by cost-effectiveness and
generates a 3-year phased implementation roadmap optimized for
SME budget constraints.

Calculation Methodology:
    NPV (5-year horizon):
        npv = -capex + sum(annual_net_savings / (1+r)^t) for t=1..5

    IRR (iterative):
        Find r where NPV(r) = 0 using bisection method.

    Simple Payback:
        payback = capex / annual_net_savings

    Discounted Payback:
        Earliest year t where cumulative discounted savings >= capex.

    MACC-lite Score:
        score = abatement_cost_per_tco2e * -1  (negative = net benefit)
        abatement_cost = (annualized_capex - annual_savings) / annual_tco2e

    Sensitivity: +/-20% on cost and savings parameters.

Regulatory References:
    - SBTi Net-Zero Standard v1.2 (2024) - Transition plan requirements
    - TCFD (2017) - Transition plan disclosure
    - EU CSRD / ESRS E1-3 - Actions to manage climate impacts
    - McKinsey MACC methodology

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - NPV/IRR use standard financial formulas
    - No Monte Carlo or stochastic modeling (SME simplicity)
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

class ActionScope(str, Enum):
    """Emission scope affected by the action."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"

class ActionEase(str, Enum):
    """Ease of implementation on 1-5 scale.

    1 = Very easy (no disruption, policy change only).
    2 = Easy (minor installation, minimal disruption).
    3 = Moderate (project management needed, some disruption).
    4 = Hard (significant planning, temporary disruption).
    5 = Very hard (major project, extended disruption).
    """
    VERY_EASY = "1"
    EASY = "2"
    MODERATE = "3"
    HARD = "4"
    VERY_HARD = "5"

class RoadmapPhase(str, Enum):
    """Phase in the implementation roadmap."""
    YEAR_1 = "year_1"
    YEAR_2 = "year_2"
    YEAR_3 = "year_3"

class SensitivityScenario(str, Enum):
    """Sensitivity analysis scenario."""
    BASE = "base"
    OPTIMISTIC = "optimistic"       # -20% cost, +20% savings
    CONSERVATIVE = "conservative"   # +20% cost, -20% savings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default financial parameters.
DEFAULT_DISCOUNT_RATE: Decimal = Decimal("0.08")  # 8% for SMEs
NPV_HORIZON_YEARS: int = 5
TARGET_IRR: Decimal = Decimal("15.0")  # 15% target IRR
TARGET_PAYBACK_YEARS: Decimal = Decimal("3.0")
SENSITIVITY_PCT: Decimal = Decimal("20.0")  # +/-20%

# Ease of implementation weights for scoring.
EASE_WEIGHTS: Dict[str, Decimal] = {
    "1": Decimal("1.5"),
    "2": Decimal("1.3"),
    "3": Decimal("1.0"),
    "4": Decimal("0.8"),
    "5": Decimal("0.6"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class ActionInput(BaseModel):
    """A single decarbonization action to be prioritized.

    Attributes:
        action_id: Unique action identifier.
        name: Action name.
        description: Brief description.
        scope: Emission scope affected.
        capex_usd: Upfront capital expenditure.
        annual_opex_change_usd: Annual operating cost change (negative = savings).
        annual_savings_usd: Annual cost savings from the action.
        annual_tco2e_reduction: Expected annual emissions reduction.
        ease_of_implementation: Ease score (1-5).
        implementation_months: Time to implement (months).
        useful_life_years: Expected useful life of the investment.
        grant_pct: Percentage of capex covered by grants/incentives.
        notes: Optional notes.
    """
    action_id: str = Field(
        default_factory=_new_uuid, description="Action ID"
    )
    name: str = Field(
        ..., min_length=1, max_length=300, description="Action name"
    )
    description: str = Field(default="", max_length=1000)
    scope: ActionScope = Field(
        default=ActionScope.SCOPE_1_2, description="Scope affected"
    )
    capex_usd: Decimal = Field(
        ..., ge=Decimal("0"), description="Upfront cost (USD)"
    )
    annual_opex_change_usd: Decimal = Field(
        default=Decimal("0"),
        description="Annual opex change (negative = savings)",
    )
    annual_savings_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Annual savings (USD)",
    )
    annual_tco2e_reduction: Decimal = Field(
        ..., ge=Decimal("0"), description="Annual reduction (tCO2e)"
    )
    ease_of_implementation: ActionEase = Field(
        default=ActionEase.MODERATE, description="Ease (1-5)"
    )
    implementation_months: int = Field(
        default=3, ge=1, le=36, description="Implementation time (months)"
    )
    useful_life_years: int = Field(
        default=10, ge=1, le=30, description="Useful life (years)"
    )
    grant_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Grant coverage (%)",
    )
    notes: str = Field(default="", max_length=500)

    @field_validator("capex_usd")
    @classmethod
    def validate_capex(cls, v: Decimal) -> Decimal:
        if v > Decimal("10000000"):
            raise ValueError("CapEx exceeds SME reasonable range ($10M)")
        return v

class PrioritizationInput(BaseModel):
    """Complete input for action prioritization.

    Attributes:
        entity_name: Company name.
        actions: List of actions to prioritize (max 10).
        total_emissions_tco2e: Current total emissions.
        annual_budget_usd: Available annual budget for decarbonization.
        discount_rate: Discount rate for NPV (default 8%).
        npv_horizon_years: NPV calculation horizon (default 5).
        carbon_price_usd_per_tco2e: Internal carbon price (optional).
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Company name"
    )
    actions: List[ActionInput] = Field(
        ..., min_length=1, max_length=10,
        description="Actions to prioritize (max 10)",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Current total emissions",
    )
    annual_budget_usd: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Annual budget for decarbonization"
    )
    discount_rate: Decimal = Field(
        default=DEFAULT_DISCOUNT_RATE, ge=Decimal("0"), le=Decimal("0.30"),
        description="Discount rate for NPV",
    )
    npv_horizon_years: int = Field(
        default=NPV_HORIZON_YEARS, ge=1, le=20,
        description="NPV horizon (years)",
    )
    carbon_price_usd_per_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Internal carbon price (USD/tCO2e)",
    )

    @field_validator("actions")
    @classmethod
    def validate_actions_count(cls, v: List[ActionInput]) -> List[ActionInput]:
        if len(v) > 10:
            raise ValueError("Maximum 10 actions for SME prioritization")
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class FinancialMetrics(BaseModel):
    """Financial analysis for a single action.

    Attributes:
        npv_usd: Net present value.
        irr_pct: Internal rate of return (%).
        simple_payback_years: Simple payback period.
        discounted_payback_years: Discounted payback period.
        abatement_cost_per_tco2e: Marginal abatement cost (USD/tCO2e).
        net_capex_after_grants: CapEx after grant deduction.
        total_5yr_savings_usd: Cumulative 5-year savings.
        roi_pct: 5-year return on investment.
    """
    npv_usd: Decimal = Field(default=Decimal("0"))
    irr_pct: Decimal = Field(default=Decimal("0"))
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    discounted_payback_years: Decimal = Field(default=Decimal("0"))
    abatement_cost_per_tco2e: Decimal = Field(default=Decimal("0"))
    net_capex_after_grants: Decimal = Field(default=Decimal("0"))
    total_5yr_savings_usd: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))

class SensitivityResult(BaseModel):
    """Sensitivity analysis result for one scenario.

    Attributes:
        scenario: Scenario name.
        npv_usd: NPV under this scenario.
        irr_pct: IRR under this scenario.
        payback_years: Payback under this scenario.
        still_viable: Whether action remains viable.
    """
    scenario: str = Field(default="base")
    npv_usd: Decimal = Field(default=Decimal("0"))
    irr_pct: Decimal = Field(default=Decimal("0"))
    payback_years: Decimal = Field(default=Decimal("0"))
    still_viable: bool = Field(default=True)

class PrioritizedAction(BaseModel):
    """A single prioritized action with financial analysis.

    Attributes:
        rank: Priority rank (1 = highest).
        action_id: Action identifier.
        name: Action name.
        description: Action description.
        scope: Scope affected.
        financials: Financial metrics.
        sensitivity: Sensitivity analysis (base/optimistic/conservative).
        roadmap_phase: Recommended implementation phase.
        composite_score: Overall priority score.
        reduction_tco2e: Annual emissions reduction.
        reduction_pct: Reduction as % of total emissions.
        ease: Ease of implementation.
        implementation_months: Time to implement.
        passes_irr_target: Whether IRR >= 15%.
        passes_payback_target: Whether payback <= 3 years.
    """
    rank: int = Field(default=0)
    action_id: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    scope: str = Field(default="")
    financials: FinancialMetrics = Field(default_factory=FinancialMetrics)
    sensitivity: List[SensitivityResult] = Field(default_factory=list)
    roadmap_phase: str = Field(default="year_1")
    composite_score: Decimal = Field(default=Decimal("0"))
    reduction_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_pct: Decimal = Field(default=Decimal("0"))
    ease: str = Field(default="3")
    implementation_months: int = Field(default=3)
    passes_irr_target: bool = Field(default=False)
    passes_payback_target: bool = Field(default=False)

class RoadmapSummary(BaseModel):
    """Summary of the phased implementation roadmap.

    Attributes:
        year_1_actions: Actions for Year 1.
        year_2_actions: Actions for Year 2.
        year_3_actions: Actions for Year 3.
        year_1_capex_usd: Year 1 total CapEx.
        year_2_capex_usd: Year 2 total CapEx.
        year_3_capex_usd: Year 3 total CapEx.
        cumulative_reduction_tco2e: Total reduction across all phases.
        cumulative_reduction_pct: Total reduction as % of baseline.
        total_capex_usd: Total investment required.
        total_5yr_savings_usd: Total 5-year savings.
        total_npv_usd: Portfolio NPV.
    """
    year_1_actions: List[str] = Field(default_factory=list)
    year_2_actions: List[str] = Field(default_factory=list)
    year_3_actions: List[str] = Field(default_factory=list)
    year_1_capex_usd: Decimal = Field(default=Decimal("0"))
    year_2_capex_usd: Decimal = Field(default=Decimal("0"))
    year_3_capex_usd: Decimal = Field(default=Decimal("0"))
    cumulative_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_reduction_pct: Decimal = Field(default=Decimal("0"))
    total_capex_usd: Decimal = Field(default=Decimal("0"))
    total_5yr_savings_usd: Decimal = Field(default=Decimal("0"))
    total_npv_usd: Decimal = Field(default=Decimal("0"))

class PrioritizationResult(BaseModel):
    """Complete action prioritization result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        entity_name: Company name.
        actions: Ranked list of prioritized actions.
        roadmap: Phased roadmap summary.
        total_actions: Number of actions analyzed.
        discount_rate_used: Discount rate used.
        npv_horizon_used: NPV horizon used.
        processing_time_ms: Calculation time (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")

    actions: List[PrioritizedAction] = Field(default_factory=list)
    roadmap: RoadmapSummary = Field(default_factory=RoadmapSummary)
    total_actions: int = Field(default=0)
    discount_rate_used: Decimal = Field(default=DEFAULT_DISCOUNT_RATE)
    npv_horizon_used: int = Field(default=NPV_HORIZON_YEARS)

    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ActionPrioritizationEngine:
    """MACC-lite action prioritization engine for SMEs.

    Ranks up to 10 decarbonization actions by NPV, IRR, payback,
    emissions reduction, and ease of implementation.  Generates a
    3-year phased roadmap.

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    No LLM or Monte Carlo is used in any calculation path.

    Usage::

        engine = ActionPrioritizationEngine()
        result = engine.calculate(prioritization_input)
        for action in result.actions:
            print(f"#{action.rank} {action.name}: NPV=${action.financials.npv_usd}")
    """

    engine_version: str = _MODULE_VERSION

    def calculate(self, data: PrioritizationInput) -> PrioritizationResult:
        """Prioritize actions and generate roadmap.

        Args:
            data: Validated prioritization input.

        Returns:
            PrioritizationResult with ranked actions and roadmap.
        """
        t0 = time.perf_counter()
        logger.info(
            "Action Prioritization: entity=%s, actions=%d, budget=%s",
            data.entity_name, len(data.actions),
            str(data.annual_budget_usd) if data.annual_budget_usd else "unlimited",
        )

        prioritized: List[PrioritizedAction] = []

        for action in data.actions:
            # Net capex after grants
            net_capex = _round_val(
                action.capex_usd * (Decimal("100") - action.grant_pct) / Decimal("100")
            )

            # Annual net savings (savings - opex change)
            annual_net = action.annual_savings_usd + action.annual_opex_change_usd
            # Carbon price benefit
            if data.carbon_price_usd_per_tco2e:
                annual_net += action.annual_tco2e_reduction * data.carbon_price_usd_per_tco2e

            # Financial metrics
            financials = self._compute_financials(
                net_capex, annual_net, action.annual_tco2e_reduction,
                data.discount_rate, data.npv_horizon_years,
            )

            # Sensitivity analysis
            sensitivity = self._compute_sensitivity(
                action.capex_usd, action.grant_pct,
                annual_net, action.annual_tco2e_reduction,
                data.discount_rate, data.npv_horizon_years,
            )

            # Reduction percentage
            reduction_pct = Decimal("0")
            if data.total_emissions_tco2e > Decimal("0"):
                reduction_pct = _round_val(
                    action.annual_tco2e_reduction * Decimal("100")
                    / data.total_emissions_tco2e, 2
                )

            # Composite score
            # Weight: NPV(30%) + Abatement(25%) + Payback(20%) + Ease(15%) + Reduction(10%)
            score = self._compute_composite_score(
                financials, action.ease_of_implementation,
                action.annual_tco2e_reduction, reduction_pct,
            )

            prioritized.append(PrioritizedAction(
                action_id=action.action_id,
                name=action.name,
                description=action.description,
                scope=action.scope.value,
                financials=financials,
                sensitivity=sensitivity,
                composite_score=score,
                reduction_tco2e=action.annual_tco2e_reduction,
                reduction_pct=reduction_pct,
                ease=action.ease_of_implementation.value,
                implementation_months=action.implementation_months,
                passes_irr_target=financials.irr_pct >= TARGET_IRR,
                passes_payback_target=(
                    financials.simple_payback_years <= TARGET_PAYBACK_YEARS
                    or financials.simple_payback_years == Decimal("0")
                ),
            ))

        # Sort by composite score descending
        prioritized.sort(key=lambda x: x.composite_score, reverse=True)

        # Assign ranks and phases
        for i, action in enumerate(prioritized):
            action.rank = i + 1
            if action.implementation_months <= 6 or i < 3:
                action.roadmap_phase = RoadmapPhase.YEAR_1.value
            elif action.implementation_months <= 18 or i < 6:
                action.roadmap_phase = RoadmapPhase.YEAR_2.value
            else:
                action.roadmap_phase = RoadmapPhase.YEAR_3.value

        # Budget constraint: if budget provided, adjust phasing
        if data.annual_budget_usd is not None:
            self._apply_budget_constraint(prioritized, data.annual_budget_usd)

        # Roadmap summary
        roadmap = self._build_roadmap(prioritized, data.total_emissions_tco2e)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = PrioritizationResult(
            entity_name=data.entity_name,
            actions=prioritized,
            roadmap=roadmap,
            total_actions=len(data.actions),
            discount_rate_used=data.discount_rate,
            npv_horizon_used=data.npv_horizon_years,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Prioritization complete: %d actions ranked, total NPV=$%.0f, hash=%s",
            len(prioritized), float(roadmap.total_npv_usd),
            result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Financial Calculations                                               #
    # ------------------------------------------------------------------ #

    def _compute_financials(
        self,
        net_capex: Decimal,
        annual_net_savings: Decimal,
        annual_tco2e: Decimal,
        discount_rate: Decimal,
        horizon: int,
    ) -> FinancialMetrics:
        """Compute NPV, IRR, payback, and abatement cost.

        Args:
            net_capex: Net CapEx after grants.
            annual_net_savings: Annual net cash benefit.
            annual_tco2e: Annual emissions reduction.
            discount_rate: Discount rate.
            horizon: NPV horizon in years.

        Returns:
            FinancialMetrics with all computed values.
        """
        # NPV
        npv = self._compute_npv(net_capex, annual_net_savings, discount_rate, horizon)

        # IRR
        irr = self._compute_irr(net_capex, annual_net_savings, horizon)

        # Simple payback
        simple_payback = Decimal("0")
        if annual_net_savings > Decimal("0") and net_capex > Decimal("0"):
            simple_payback = _round_val(
                _safe_divide(net_capex, annual_net_savings), 2
            )

        # Discounted payback
        discounted_payback = self._compute_discounted_payback(
            net_capex, annual_net_savings, discount_rate, horizon
        )

        # Abatement cost (USD per tCO2e)
        abatement_cost = Decimal("0")
        if annual_tco2e > Decimal("0"):
            # Annualized capex over useful life (simplified to horizon)
            annualized_capex = _safe_divide(net_capex, _decimal(horizon))
            net_annual_cost = annualized_capex - annual_net_savings
            abatement_cost = _round_val(
                _safe_divide(net_annual_cost, annual_tco2e), 2
            )

        # 5-year total savings
        total_5yr = _round_val(annual_net_savings * Decimal("5") - net_capex)

        # ROI
        roi = Decimal("0")
        if net_capex > Decimal("0"):
            roi = _round_val(total_5yr * Decimal("100") / net_capex, 2)

        return FinancialMetrics(
            npv_usd=_round_val(npv, 2),
            irr_pct=irr,
            simple_payback_years=simple_payback,
            discounted_payback_years=discounted_payback,
            abatement_cost_per_tco2e=abatement_cost,
            net_capex_after_grants=net_capex,
            total_5yr_savings_usd=total_5yr,
            roi_pct=roi,
        )

    def _compute_npv(
        self,
        capex: Decimal,
        annual_cf: Decimal,
        rate: Decimal,
        years: int,
    ) -> Decimal:
        """Compute Net Present Value.

        NPV = -capex + sum(annual_cf / (1+r)^t) for t=1..years

        Args:
            capex: Initial investment.
            annual_cf: Annual cash flow (positive = inflow).
            rate: Discount rate.
            years: Number of years.

        Returns:
            NPV as Decimal.
        """
        npv = -capex
        for t in range(1, years + 1):
            discount_factor = (Decimal("1") + rate) ** t
            npv += _safe_divide(annual_cf, discount_factor)
        return npv

    def _compute_irr(
        self,
        capex: Decimal,
        annual_cf: Decimal,
        years: int,
    ) -> Decimal:
        """Compute Internal Rate of Return using bisection.

        Args:
            capex: Initial investment.
            annual_cf: Annual cash flow.
            years: Number of years.

        Returns:
            IRR as percentage (e.g., 15.0 for 15%).
        """
        if capex == Decimal("0"):
            return Decimal("999") if annual_cf > Decimal("0") else Decimal("0")
        if annual_cf <= Decimal("0"):
            return Decimal("0")

        low = Decimal("-0.50")
        high = Decimal("5.00")

        for _ in range(100):  # max iterations
            mid = (low + high) / Decimal("2")
            npv_mid = self._compute_npv(capex, annual_cf, mid, years)

            if abs(npv_mid) < Decimal("0.01"):
                return _round_val(mid * Decimal("100"), 2)
            elif npv_mid > Decimal("0"):
                low = mid
            else:
                high = mid

        mid = (low + high) / Decimal("2")
        return _round_val(mid * Decimal("100"), 2)

    def _compute_discounted_payback(
        self,
        capex: Decimal,
        annual_cf: Decimal,
        rate: Decimal,
        max_years: int,
    ) -> Decimal:
        """Compute discounted payback period.

        Args:
            capex: Initial investment.
            annual_cf: Annual cash flow.
            rate: Discount rate.
            max_years: Maximum years to check.

        Returns:
            Discounted payback in years (0 if not recoverable).
        """
        if capex == Decimal("0"):
            return Decimal("0")
        if annual_cf <= Decimal("0"):
            return Decimal("0")

        cumulative = Decimal("0")
        for t in range(1, max_years + 1):
            discount_factor = (Decimal("1") + rate) ** t
            cumulative += _safe_divide(annual_cf, discount_factor)
            if cumulative >= capex:
                return _round_val(_decimal(t), 1)

        return Decimal("0")  # Not recovered within horizon

    # ------------------------------------------------------------------ #
    # Sensitivity Analysis                                                 #
    # ------------------------------------------------------------------ #

    def _compute_sensitivity(
        self,
        capex: Decimal,
        grant_pct: Decimal,
        annual_net: Decimal,
        annual_tco2e: Decimal,
        rate: Decimal,
        horizon: int,
    ) -> List[SensitivityResult]:
        """Run +/-20% sensitivity on cost and savings.

        Args:
            capex: Original CapEx.
            grant_pct: Grant percentage.
            annual_net: Annual net savings.
            annual_tco2e: Annual emissions reduction.
            rate: Discount rate.
            horizon: NPV horizon.

        Returns:
            List of SensitivityResult for base/optimistic/conservative.
        """
        scenarios = [
            (SensitivityScenario.BASE, Decimal("1.00"), Decimal("1.00")),
            (SensitivityScenario.OPTIMISTIC, Decimal("0.80"), Decimal("1.20")),
            (SensitivityScenario.CONSERVATIVE, Decimal("1.20"), Decimal("0.80")),
        ]

        results: List[SensitivityResult] = []
        for scenario, cost_mult, savings_mult in scenarios:
            adj_capex = capex * cost_mult
            adj_net_capex = adj_capex * (Decimal("100") - grant_pct) / Decimal("100")
            adj_savings = annual_net * savings_mult

            npv = self._compute_npv(adj_net_capex, adj_savings, rate, horizon)
            irr = self._compute_irr(adj_net_capex, adj_savings, horizon)
            payback = Decimal("0")
            if adj_savings > Decimal("0") and adj_net_capex > Decimal("0"):
                payback = _round_val(
                    _safe_divide(adj_net_capex, adj_savings), 2
                )

            results.append(SensitivityResult(
                scenario=scenario.value,
                npv_usd=_round_val(npv, 2),
                irr_pct=irr,
                payback_years=payback,
                still_viable=npv > Decimal("0"),
            ))

        return results

    # ------------------------------------------------------------------ #
    # Scoring & Roadmap                                                    #
    # ------------------------------------------------------------------ #

    def _compute_composite_score(
        self,
        financials: FinancialMetrics,
        ease: ActionEase,
        reduction_tco2e: Decimal,
        reduction_pct: Decimal,
    ) -> Decimal:
        """Compute composite priority score.

        Weights: NPV(30%) + Abatement_cost(25%) + Payback(20%) + Ease(15%) + Reduction(10%)

        Args:
            financials: Computed financial metrics.
            ease: Ease of implementation.
            reduction_tco2e: Annual emissions reduction.
            reduction_pct: Reduction as % of total.

        Returns:
            Composite score (higher = better).
        """
        # NPV score: normalize to 0-100 range
        npv_score = Decimal("50")
        if financials.npv_usd > Decimal("0"):
            npv_score = min(Decimal("100"), Decimal("50") + financials.npv_usd / Decimal("100"))
        elif financials.npv_usd < Decimal("0"):
            npv_score = max(Decimal("0"), Decimal("50") + financials.npv_usd / Decimal("100"))

        # Abatement cost score: lower = better (negative = net benefit)
        abatement_score = Decimal("50")
        if financials.abatement_cost_per_tco2e < Decimal("0"):
            abatement_score = min(
                Decimal("100"),
                Decimal("50") + abs(financials.abatement_cost_per_tco2e) / Decimal("10")
            )
        elif financials.abatement_cost_per_tco2e > Decimal("0"):
            abatement_score = max(
                Decimal("0"),
                Decimal("50") - financials.abatement_cost_per_tco2e / Decimal("10")
            )

        # Payback score: shorter = better
        payback_score = Decimal("50")
        if financials.simple_payback_years == Decimal("0"):
            payback_score = Decimal("100")
        elif financials.simple_payback_years <= Decimal("1"):
            payback_score = Decimal("90")
        elif financials.simple_payback_years <= Decimal("2"):
            payback_score = Decimal("70")
        elif financials.simple_payback_years <= Decimal("3"):
            payback_score = Decimal("50")
        elif financials.simple_payback_years <= Decimal("5"):
            payback_score = Decimal("30")
        else:
            payback_score = Decimal("10")

        # Ease score
        ease_score = EASE_WEIGHTS.get(ease.value, Decimal("1.0")) * Decimal("66.67")

        # Reduction score
        reduction_score = min(Decimal("100"), reduction_pct * Decimal("10"))

        composite = (
            npv_score * Decimal("0.30")
            + abatement_score * Decimal("0.25")
            + payback_score * Decimal("0.20")
            + ease_score * Decimal("0.15")
            + reduction_score * Decimal("0.10")
        )

        return _round_val(composite, 2)

    def _apply_budget_constraint(
        self,
        actions: List[PrioritizedAction],
        annual_budget: Decimal,
    ) -> None:
        """Adjust roadmap phasing based on budget constraints.

        Moves actions to later phases if annual budget is exceeded.
        Mutates actions in place.

        Args:
            actions: List of prioritized actions (sorted by score).
            annual_budget: Available annual budget.
        """
        phase_budgets = {
            RoadmapPhase.YEAR_1.value: Decimal("0"),
            RoadmapPhase.YEAR_2.value: Decimal("0"),
            RoadmapPhase.YEAR_3.value: Decimal("0"),
        }

        for action in actions:
            capex = action.financials.net_capex_after_grants
            # Try to fit in the assigned phase
            phase = action.roadmap_phase
            if phase_budgets[phase] + capex > annual_budget:
                # Move to next phase
                if phase == RoadmapPhase.YEAR_1.value:
                    phase = RoadmapPhase.YEAR_2.value
                elif phase == RoadmapPhase.YEAR_2.value:
                    phase = RoadmapPhase.YEAR_3.value

            if phase_budgets.get(phase, Decimal("0")) + capex <= annual_budget:
                action.roadmap_phase = phase
                phase_budgets[phase] = phase_budgets.get(
                    phase, Decimal("0")
                ) + capex
            else:
                action.roadmap_phase = RoadmapPhase.YEAR_3.value

    def _build_roadmap(
        self,
        actions: List[PrioritizedAction],
        total_emissions: Decimal,
    ) -> RoadmapSummary:
        """Build the phased roadmap summary.

        Args:
            actions: Prioritized actions with assigned phases.
            total_emissions: Total current emissions.

        Returns:
            RoadmapSummary with phase details.
        """
        y1_actions: List[str] = []
        y2_actions: List[str] = []
        y3_actions: List[str] = []
        y1_capex = Decimal("0")
        y2_capex = Decimal("0")
        y3_capex = Decimal("0")
        total_reduction = Decimal("0")
        total_savings = Decimal("0")
        total_npv = Decimal("0")
        total_capex = Decimal("0")

        for a in actions:
            capex = a.financials.net_capex_after_grants
            total_capex += capex
            total_reduction += a.reduction_tco2e
            total_savings += a.financials.total_5yr_savings_usd
            total_npv += a.financials.npv_usd

            if a.roadmap_phase == RoadmapPhase.YEAR_1.value:
                y1_actions.append(a.name)
                y1_capex += capex
            elif a.roadmap_phase == RoadmapPhase.YEAR_2.value:
                y2_actions.append(a.name)
                y2_capex += capex
            else:
                y3_actions.append(a.name)
                y3_capex += capex

        reduction_pct = Decimal("0")
        if total_emissions > Decimal("0"):
            reduction_pct = _round_val(
                total_reduction * Decimal("100") / total_emissions, 2
            )

        return RoadmapSummary(
            year_1_actions=y1_actions,
            year_2_actions=y2_actions,
            year_3_actions=y3_actions,
            year_1_capex_usd=_round_val(y1_capex, 2),
            year_2_capex_usd=_round_val(y2_capex, 2),
            year_3_capex_usd=_round_val(y3_capex, 2),
            cumulative_reduction_tco2e=_round_val(total_reduction),
            cumulative_reduction_pct=reduction_pct,
            total_capex_usd=_round_val(total_capex, 2),
            total_5yr_savings_usd=_round_val(total_savings, 2),
            total_npv_usd=_round_val(total_npv, 2),
        )
