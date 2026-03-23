# -*- coding: utf-8 -*-
"""
RevenueOptimizerEngine - PACK-037 Demand Response Engine 8
============================================================

Revenue stream optimisation engine for demand response programmes.
Calculates revenue from capacity payments, energy payments, ancillary
service payments, demand charge savings, and price arbitrage.  Computes
net revenue after penalties, enabling costs, and operational costs.
Provides ROI/payback analysis, annual forecasting, what-if scenario
analysis, and portfolio optimisation across multiple DR programmes.

Calculation Methodology:
    Revenue Streams:
        capacity_revenue   = nominated_kw * capacity_price_per_kw_month * months
        energy_revenue     = curtailed_kwh * energy_price_per_kwh
        ancillary_revenue  = regulation_mw * ancillary_price_per_mw_h * hours
        demand_savings     = avoided_peak_kw * demand_charge_per_kw
        arbitrage_revenue  = discharged_kwh * (peak_price - off_peak_price)

    Net Revenue:
        gross_revenue      = sum(all revenue streams)
        net_revenue        = gross_revenue - penalties - enabling_costs - opex
        enabling_costs     = metering + controls + communication + integration

    ROI and Payback:
        total_investment   = capex + enabling_costs
        annual_net         = net_revenue - annual_opex
        simple_payback     = total_investment / annual_net  (years)
        roi_pct            = (annual_net * project_life - total_investment)
                             / total_investment * 100
        npv                = -investment + sum(net_t / (1+r)^t for t in 1..n)
        irr                = rate where NPV = 0 (bisection method)

    What-If Scenarios:
        Vary event frequency, capacity price, compliance rate, or
        enabling costs and recalculate net revenue and ROI.

Regulatory References:
    - FERC Order 745 - Demand Response Compensation
    - FERC Order 2222 - DER Aggregation in Wholesale Markets
    - PJM Reliability Pricing Model (RPM) Capacity Market
    - ISO-NE Forward Capacity Market Rules
    - NYISO ICAP / SCR Payments
    - CAISO Resource Adequacy Requirements
    - ERCOT Emergency Response Service (ERS)

Zero-Hallucination:
    - Revenue formulas from published ISO/RTO tariff schedules
    - NPV/IRR use standard engineering economics (no LLM)
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-037 Demand Response
Engine:  8 of 10
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RevenueStreamType(str, Enum):
    """Revenue stream type classification.

    CAPACITY_PAYMENT:     Fixed payment for available curtailment capacity.
    ENERGY_PAYMENT:       Variable payment for actual energy curtailment.
    ANCILLARY_PAYMENT:    Payment for ancillary services (regulation, reserves).
    DEMAND_CHARGE_SAVINGS: Avoided demand charges from peak reduction.
    PRICE_ARBITRAGE:      Revenue from energy price differentials.
    """
    CAPACITY_PAYMENT = "capacity_payment"
    ENERGY_PAYMENT = "energy_payment"
    ANCILLARY_PAYMENT = "ancillary_payment"
    DEMAND_CHARGE_SAVINGS = "demand_charge_savings"
    PRICE_ARBITRAGE = "price_arbitrage"


class CostCategory(str, Enum):
    """Cost category classification.

    METERING:        Interval metering equipment and installation.
    CONTROLS:        EMCS, BAS, or load control hardware/software.
    COMMUNICATION:   Telemetry and communication infrastructure.
    INTEGRATION:     System integration and configuration.
    AGGREGATOR_FEE:  Third-party aggregator service fees.
    ANNUAL_OPEX:     Annual operational expenditure.
    MAINTENANCE:     Equipment maintenance costs.
    """
    METERING = "metering"
    CONTROLS = "controls"
    COMMUNICATION = "communication"
    INTEGRATION = "integration"
    AGGREGATOR_FEE = "aggregator_fee"
    ANNUAL_OPEX = "annual_opex"
    MAINTENANCE = "maintenance"


class ScenarioParameter(str, Enum):
    """Parameter to vary in what-if analysis.

    EVENT_FREQUENCY:   Number of DR events per year.
    CAPACITY_PRICE:    Capacity payment rate.
    ENERGY_PRICE:      Energy payment rate.
    COMPLIANCE_RATE:   Assumed compliance rate.
    ENABLING_COST:     Total enabling cost.
    CURTAILMENT_KW:    Nominated curtailment capacity.
    PENALTY_RATE:      Penalty rate for non-compliance.
    """
    EVENT_FREQUENCY = "event_frequency"
    CAPACITY_PRICE = "capacity_price"
    ENERGY_PRICE = "energy_price"
    COMPLIANCE_RATE = "compliance_rate"
    ENABLING_COST = "enabling_cost"
    CURTAILMENT_KW = "curtailment_kw"
    PENALTY_RATE = "penalty_rate"


class OptimisationObjective(str, Enum):
    """Portfolio optimisation objective.

    MAXIMIZE_NET_REVENUE: Maximise total net revenue.
    MAXIMIZE_ROI:         Maximise return on investment.
    MINIMIZE_PAYBACK:     Minimise simple payback period.
    MAXIMIZE_NPV:         Maximise net present value.
    """
    MAXIMIZE_NET_REVENUE = "maximize_net_revenue"
    MAXIMIZE_ROI = "maximize_roi"
    MINIMIZE_PAYBACK = "minimize_payback"
    MAXIMIZE_NPV = "maximize_npv"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DISCOUNT_RATE: Decimal = Decimal("0.08")
DEFAULT_PROJECT_LIFE_YEARS: int = 10
DEFAULT_REVENUE_ESCALATION: Decimal = Decimal("0.02")
DEFAULT_COST_ESCALATION: Decimal = Decimal("0.025")
MAX_IRR_ITERATIONS: int = 100
IRR_TOLERANCE: Decimal = Decimal("0.0001")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class RevenueStream(BaseModel):
    """Single revenue stream definition.

    Attributes:
        stream_type: Type of revenue stream.
        label: Human-readable label.
        annual_amount: Expected annual revenue (USD).
        unit_rate: Rate per unit (USD/kW-month, USD/kWh, etc.).
        unit_quantity: Quantity (kW, kWh, MW-h, etc.).
        months_active: Months per year this stream is active.
        compliance_factor: Expected compliance rate (0-1).
        notes: Additional notes.
    """
    stream_type: RevenueStreamType = Field(
        ..., description="Revenue stream type"
    )
    label: str = Field(
        default="", max_length=500, description="Label"
    )
    annual_amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual amount (USD)"
    )
    unit_rate: Decimal = Field(
        default=Decimal("0"), ge=0, description="Unit rate (USD/unit)"
    )
    unit_quantity: Decimal = Field(
        default=Decimal("0"), ge=0, description="Unit quantity"
    )
    months_active: int = Field(
        default=12, ge=1, le=12, description="Months active per year"
    )
    compliance_factor: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=Decimal("1.0"),
        description="Compliance factor (0-1)"
    )
    notes: str = Field(
        default="", max_length=1000, description="Notes"
    )

    @field_validator("stream_type", mode="before")
    @classmethod
    def validate_stream_type(cls, v: Any) -> Any:
        """Accept string values for RevenueStreamType."""
        if isinstance(v, str):
            valid = {t.value for t in RevenueStreamType}
            if v not in valid:
                raise ValueError(
                    f"Unknown stream type '{v}'. Must be one of: {sorted(valid)}"
                )
        return v


class CostItem(BaseModel):
    """Single cost item.

    Attributes:
        category: Cost category.
        label: Description.
        amount: Cost amount (USD).
        is_recurring: True if annual recurring cost.
        escalation_rate: Annual escalation rate for recurring costs.
    """
    category: CostCategory = Field(
        ..., description="Cost category"
    )
    label: str = Field(
        default="", max_length=500, description="Description"
    )
    amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Amount (USD)"
    )
    is_recurring: bool = Field(
        default=False, description="Recurring annual cost"
    )
    escalation_rate: Decimal = Field(
        default=DEFAULT_COST_ESCALATION, ge=0, le=Decimal("0.15"),
        description="Annual escalation rate"
    )


class ProgrammeFinancials(BaseModel):
    """Financial parameters for a DR programme.

    Attributes:
        programme_id: Programme identifier.
        programme_name: Human-readable name.
        revenue_streams: List of revenue streams.
        costs: List of cost items.
        capex: Capital expenditure for DR participation.
        annual_penalties: Expected annual penalties (USD).
        discount_rate: Discount rate for NPV/IRR.
        project_life_years: Project evaluation period.
        revenue_escalation_rate: Annual revenue escalation.
        nominated_kw: Total nominated curtailment (kW).
        events_per_year: Expected events per year.
        avg_event_duration_hours: Average event duration (hours).
    """
    programme_id: str = Field(
        default_factory=_new_uuid, description="Programme ID"
    )
    programme_name: str = Field(
        default="", max_length=500, description="Programme name"
    )
    revenue_streams: List[RevenueStream] = Field(
        default_factory=list, description="Revenue streams"
    )
    costs: List[CostItem] = Field(
        default_factory=list, description="Cost items"
    )
    capex: Decimal = Field(
        default=Decimal("0"), ge=0, description="Capital expenditure (USD)"
    )
    annual_penalties: Decimal = Field(
        default=Decimal("0"), ge=0, description="Expected annual penalties (USD)"
    )
    discount_rate: Decimal = Field(
        default=DEFAULT_DISCOUNT_RATE, ge=0, le=Decimal("0.50"),
        description="Discount rate"
    )
    project_life_years: int = Field(
        default=DEFAULT_PROJECT_LIFE_YEARS, ge=1, le=30,
        description="Project life (years)"
    )
    revenue_escalation_rate: Decimal = Field(
        default=DEFAULT_REVENUE_ESCALATION, ge=Decimal("-0.05"), le=Decimal("0.15"),
        description="Revenue escalation rate"
    )
    nominated_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Nominated curtailment (kW)"
    )
    events_per_year: int = Field(
        default=10, ge=0, le=365, description="Events per year"
    )
    avg_event_duration_hours: Decimal = Field(
        default=Decimal("4"), ge=Decimal("0.25"), le=Decimal("24"),
        description="Avg event duration (hours)"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class AnnualCashFlow(BaseModel):
    """Year-by-year cash flow projection.

    Attributes:
        year: Year number (1-based).
        gross_revenue: Gross revenue (USD).
        penalties: Penalties (USD).
        enabling_costs: One-time or recurring enabling costs (USD).
        operating_costs: Annual operating costs (USD).
        net_revenue: Net revenue (USD).
        cumulative_net: Cumulative net revenue (USD).
        discounted_net: Discounted net revenue (USD).
        cumulative_discounted: Cumulative discounted net (USD).
    """
    year: int = Field(default=0, ge=0, description="Year number")
    gross_revenue: Decimal = Field(default=Decimal("0"))
    penalties: Decimal = Field(default=Decimal("0"))
    enabling_costs: Decimal = Field(default=Decimal("0"))
    operating_costs: Decimal = Field(default=Decimal("0"))
    net_revenue: Decimal = Field(default=Decimal("0"))
    cumulative_net: Decimal = Field(default=Decimal("0"))
    discounted_net: Decimal = Field(default=Decimal("0"))
    cumulative_discounted: Decimal = Field(default=Decimal("0"))


class RevenueForecast(BaseModel):
    """Annual revenue forecast.

    Attributes:
        programme_id: Programme identifier.
        total_gross_revenue: Total gross revenue (USD).
        total_net_revenue: Total net revenue (USD).
        total_enabling_costs: Total enabling costs (USD).
        total_operating_costs: Total operating costs (USD).
        total_penalties: Total penalties (USD).
        npv: Net present value (USD).
        irr: Internal rate of return (%).
        simple_payback_years: Simple payback period (years).
        roi_pct: Return on investment (%).
        cash_flows: Year-by-year projections.
        revenue_by_stream: Revenue breakdown by stream type.
        is_profitable: True if NPV > 0.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    programme_id: str = Field(default="", description="Programme ID")
    total_gross_revenue: Decimal = Field(default=Decimal("0"))
    total_net_revenue: Decimal = Field(default=Decimal("0"))
    total_enabling_costs: Decimal = Field(default=Decimal("0"))
    total_operating_costs: Decimal = Field(default=Decimal("0"))
    total_penalties: Decimal = Field(default=Decimal("0"))
    npv: Decimal = Field(default=Decimal("0"))
    irr: Decimal = Field(default=Decimal("0"))
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))
    cash_flows: List[AnnualCashFlow] = Field(default_factory=list)
    revenue_by_stream: Dict[str, Decimal] = Field(default_factory=dict)
    is_profitable: bool = Field(default=False)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class WhatIfScenario(BaseModel):
    """What-if scenario analysis result.

    Attributes:
        scenario_id: Scenario identifier.
        parameter: Parameter varied.
        base_value: Base case value.
        test_value: Tested value.
        base_net_revenue: Base case net revenue (USD).
        scenario_net_revenue: Scenario net revenue (USD).
        revenue_delta: Change in net revenue (USD).
        revenue_delta_pct: Percentage change.
        base_roi_pct: Base case ROI (%).
        scenario_roi_pct: Scenario ROI (%).
        base_payback_years: Base case payback (years).
        scenario_payback_years: Scenario payback (years).
        provenance_hash: SHA-256 audit hash.
    """
    scenario_id: str = Field(default_factory=_new_uuid)
    parameter: ScenarioParameter = Field(...)
    base_value: Decimal = Field(default=Decimal("0"))
    test_value: Decimal = Field(default=Decimal("0"))
    base_net_revenue: Decimal = Field(default=Decimal("0"))
    scenario_net_revenue: Decimal = Field(default=Decimal("0"))
    revenue_delta: Decimal = Field(default=Decimal("0"))
    revenue_delta_pct: Decimal = Field(default=Decimal("0"))
    base_roi_pct: Decimal = Field(default=Decimal("0"))
    scenario_roi_pct: Decimal = Field(default=Decimal("0"))
    base_payback_years: Decimal = Field(default=Decimal("0"))
    scenario_payback_years: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class RevenueOptimization(BaseModel):
    """Portfolio revenue optimisation result.

    Attributes:
        optimization_id: Optimisation run identifier.
        objective: Optimisation objective.
        programmes: Individual programme forecasts.
        total_investment: Total investment across portfolio (USD).
        total_annual_net_revenue: Total annual net revenue (USD).
        portfolio_npv: Portfolio NPV (USD).
        portfolio_roi_pct: Portfolio ROI (%).
        portfolio_payback_years: Portfolio payback (years).
        recommendations: Optimisation recommendations.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    optimization_id: str = Field(default_factory=_new_uuid)
    objective: OptimisationObjective = Field(
        default=OptimisationObjective.MAXIMIZE_NET_REVENUE
    )
    programmes: List[RevenueForecast] = Field(default_factory=list)
    total_investment: Decimal = Field(default=Decimal("0"))
    total_annual_net_revenue: Decimal = Field(default=Decimal("0"))
    portfolio_npv: Decimal = Field(default=Decimal("0"))
    portfolio_roi_pct: Decimal = Field(default=Decimal("0"))
    portfolio_payback_years: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RevenueOptimizerEngine:
    """Revenue optimisation engine for demand response programmes.

    Calculates revenue from multiple streams, projects cash flows,
    computes ROI/payback/NPV/IRR, runs what-if scenarios, and
    optimises across a portfolio of programmes.

    Usage::

        engine = RevenueOptimizerEngine()
        forecast = engine.forecast_annual(programme_financials)
        print(f"NPV: {forecast.npv}, ROI: {forecast.roi_pct}%")

        scenario = engine.run_what_if(
            programme_financials,
            ScenarioParameter.CAPACITY_PRICE,
            Decimal("15.00"),
        )

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise RevenueOptimizerEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - discount_rate (Decimal): default discount rate
                - project_life_years (int): default project life
                - revenue_escalation (Decimal): default escalation
                - cost_escalation (Decimal): default cost escalation
        """
        self.config = config or {}
        self._discount_rate = _decimal(
            self.config.get("discount_rate", DEFAULT_DISCOUNT_RATE)
        )
        self._project_life = int(
            self.config.get("project_life_years", DEFAULT_PROJECT_LIFE_YEARS)
        )
        self._rev_escalation = _decimal(
            self.config.get("revenue_escalation", DEFAULT_REVENUE_ESCALATION)
        )
        self._cost_escalation = _decimal(
            self.config.get("cost_escalation", DEFAULT_COST_ESCALATION)
        )
        logger.info(
            "RevenueOptimizerEngine v%s initialised (discount=%.2f, life=%d yr)",
            self.engine_version, float(self._discount_rate), self._project_life,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate_revenue(
        self,
        streams: List[RevenueStream],
    ) -> Dict[str, Decimal]:
        """Calculate annual revenue from multiple streams.

        Sums across all revenue streams, applying compliance factors
        and active month adjustments.

        Args:
            streams: List of revenue stream definitions.

        Returns:
            Dictionary with per-stream and total revenue.
        """
        t0 = time.perf_counter()
        logger.info("Calculating revenue: %d streams", len(streams))

        result: Dict[str, Decimal] = {}
        total = Decimal("0")

        for stream in streams:
            if stream.annual_amount > Decimal("0"):
                amount = stream.annual_amount * stream.compliance_factor
            else:
                amount = (
                    stream.unit_rate
                    * stream.unit_quantity
                    * _decimal(stream.months_active)
                    * stream.compliance_factor
                )
            amount = _round_val(amount, 2)
            result[stream.stream_type.value] = (
                result.get(stream.stream_type.value, Decimal("0")) + amount
            )
            total += amount

        result["total"] = _round_val(total, 2)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Revenue calculated: total=%.2f, %d streams (%.1f ms)",
            float(total), len(streams), elapsed,
        )
        return result

    def forecast_annual(
        self,
        financials: ProgrammeFinancials,
    ) -> RevenueForecast:
        """Forecast annual revenue and compute financial metrics.

        Builds year-by-year cash flows with escalation, computes
        NPV, IRR, simple payback, and ROI over the project life.

        Args:
            financials: Programme financial parameters.

        Returns:
            RevenueForecast with projections and financial metrics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Forecasting: programme=%s, life=%d yr, capex=%s",
            financials.programme_name, financials.project_life_years,
            str(financials.capex),
        )

        # Step 1: Calculate base year revenue
        revenue_breakdown = self.calculate_revenue(financials.revenue_streams)
        base_gross = revenue_breakdown.get("total", Decimal("0"))

        # Step 2: Calculate costs
        one_time_costs = sum(
            (c.amount for c in financials.costs if not c.is_recurring),
            Decimal("0"),
        )
        recurring_costs = sum(
            (c.amount for c in financials.costs if c.is_recurring),
            Decimal("0"),
        )
        total_investment = financials.capex + one_time_costs

        # Step 3: Build cash flows
        cash_flows = self._build_cash_flows(
            base_gross=base_gross,
            annual_penalties=financials.annual_penalties,
            one_time_costs=one_time_costs,
            recurring_costs=recurring_costs,
            rev_escalation=financials.revenue_escalation_rate,
            discount_rate=financials.discount_rate,
            project_life=financials.project_life_years,
        )

        # Step 4: NPV
        npv = self._calculate_npv(cash_flows, total_investment, financials.discount_rate)

        # Step 5: IRR
        irr = self._calculate_irr(cash_flows, total_investment)

        # Step 6: Simple payback
        year1_net = cash_flows[0].net_revenue if cash_flows else Decimal("0")
        simple_payback = _safe_divide(total_investment, year1_net, Decimal("99"))
        simple_payback = max(simple_payback, Decimal("0"))

        # Step 7: ROI
        total_net = sum((cf.net_revenue for cf in cash_flows), Decimal("0"))
        roi_pct = _safe_pct(total_net - total_investment, total_investment)

        # Totals
        total_gross = sum((cf.gross_revenue for cf in cash_flows), Decimal("0"))
        total_penalties = sum((cf.penalties for cf in cash_flows), Decimal("0"))
        total_enabling = one_time_costs
        total_opex = sum((cf.operating_costs for cf in cash_flows), Decimal("0"))

        forecast = RevenueForecast(
            programme_id=financials.programme_id,
            total_gross_revenue=_round_val(total_gross, 2),
            total_net_revenue=_round_val(total_net, 2),
            total_enabling_costs=_round_val(total_enabling, 2),
            total_operating_costs=_round_val(total_opex, 2),
            total_penalties=_round_val(total_penalties, 2),
            npv=_round_val(npv, 2),
            irr=_round_val(irr, 2),
            simple_payback_years=_round_val(simple_payback, 2),
            roi_pct=_round_val(roi_pct, 2),
            cash_flows=cash_flows,
            revenue_by_stream={
                k: _round_val(v, 2) for k, v in revenue_breakdown.items()
                if k != "total"
            },
            is_profitable=npv > Decimal("0"),
        )
        forecast.provenance_hash = _compute_hash(forecast)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Forecast complete: programme=%s, NPV=%.2f, ROI=%.1f%%, "
            "payback=%.1f yr, profitable=%s, hash=%s (%.1f ms)",
            financials.programme_name, float(npv), float(roi_pct),
            float(simple_payback), forecast.is_profitable,
            forecast.provenance_hash[:16], elapsed,
        )
        return forecast

    def run_what_if(
        self,
        financials: ProgrammeFinancials,
        parameter: ScenarioParameter,
        test_value: Decimal,
    ) -> WhatIfScenario:
        """Run a what-if scenario varying a single parameter.

        Compares base case to scenario case, computing deltas
        in net revenue, ROI, and payback.

        Args:
            financials: Base case programme financials.
            parameter: Parameter to vary.
            test_value: Value to test.

        Returns:
            WhatIfScenario with base vs scenario comparison.
        """
        t0 = time.perf_counter()
        logger.info(
            "What-if scenario: param=%s, test_value=%s",
            parameter.value, str(test_value),
        )

        # Base case
        base_forecast = self.forecast_annual(financials)

        # Adjust parameter
        adjusted = self._apply_scenario(financials, parameter, test_value)
        base_value = self._get_base_value(financials, parameter)

        # Scenario case
        scenario_forecast = self.forecast_annual(adjusted)

        # Deltas
        revenue_delta = scenario_forecast.total_net_revenue - base_forecast.total_net_revenue
        delta_pct = _safe_pct(revenue_delta, base_forecast.total_net_revenue)

        scenario = WhatIfScenario(
            parameter=parameter,
            base_value=_round_val(base_value, 4),
            test_value=_round_val(test_value, 4),
            base_net_revenue=base_forecast.total_net_revenue,
            scenario_net_revenue=scenario_forecast.total_net_revenue,
            revenue_delta=_round_val(revenue_delta, 2),
            revenue_delta_pct=_round_val(delta_pct, 2),
            base_roi_pct=base_forecast.roi_pct,
            scenario_roi_pct=scenario_forecast.roi_pct,
            base_payback_years=base_forecast.simple_payback_years,
            scenario_payback_years=scenario_forecast.simple_payback_years,
        )
        scenario.provenance_hash = _compute_hash(scenario)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "What-if complete: param=%s, delta=%.2f (%.1f%%), hash=%s (%.1f ms)",
            parameter.value, float(revenue_delta), float(delta_pct),
            scenario.provenance_hash[:16], elapsed,
        )
        return scenario

    def optimize_portfolio(
        self,
        programmes: List[ProgrammeFinancials],
        objective: OptimisationObjective = OptimisationObjective.MAXIMIZE_NET_REVENUE,
    ) -> RevenueOptimization:
        """Optimise revenue across a portfolio of DR programmes.

        Forecasts each programme, aggregates portfolio-level metrics,
        ranks programmes by objective, and generates recommendations.

        Args:
            programmes: List of programme financials.
            objective: Optimisation objective.

        Returns:
            RevenueOptimization with ranked programmes and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Optimising portfolio: %d programmes, objective=%s",
            len(programmes), objective.value,
        )

        # Forecast each programme
        forecasts: List[RevenueForecast] = []
        for prog in programmes:
            forecast = self.forecast_annual(prog)
            forecasts.append(forecast)

        # Sort by objective
        sorted_forecasts = self._sort_by_objective(forecasts, objective)

        # Aggregate
        total_investment = sum(
            (p.capex + sum((c.amount for c in p.costs if not c.is_recurring), Decimal("0"))
             for p in programmes),
            Decimal("0"),
        )
        total_annual_net = sum(
            (f.total_net_revenue / _decimal(
                next((p.project_life_years for p in programmes
                      if p.programme_id == f.programme_id), self._project_life)
            ) for f in forecasts),
            Decimal("0"),
        )
        portfolio_npv = sum((f.npv for f in forecasts), Decimal("0"))
        portfolio_roi = _safe_pct(
            portfolio_npv, total_investment,
        )
        portfolio_payback = _safe_divide(
            total_investment, total_annual_net, Decimal("99"),
        )

        # Recommendations
        recommendations = self._generate_recommendations(
            forecasts, programmes, objective,
        )

        result = RevenueOptimization(
            objective=objective,
            programmes=sorted_forecasts,
            total_investment=_round_val(total_investment, 2),
            total_annual_net_revenue=_round_val(total_annual_net, 2),
            portfolio_npv=_round_val(portfolio_npv, 2),
            portfolio_roi_pct=_round_val(portfolio_roi, 2),
            portfolio_payback_years=_round_val(portfolio_payback, 2),
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Portfolio optimisation complete: %d programmes, NPV=%.2f, "
            "ROI=%.1f%%, payback=%.1f yr, hash=%s (%.1f ms)",
            len(programmes), float(portfolio_npv), float(portfolio_roi),
            float(portfolio_payback), result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_roi(
        self,
        total_investment: Decimal,
        annual_net_revenue: Decimal,
        project_life_years: int = 10,
        discount_rate: Optional[Decimal] = None,
    ) -> Dict[str, Decimal]:
        """Calculate standalone ROI, payback, and NPV.

        Convenience method for quick financial assessment.

        Args:
            total_investment: Total investment (USD).
            annual_net_revenue: Annual net revenue (USD).
            project_life_years: Project life (years).
            discount_rate: Discount rate (optional).

        Returns:
            Dictionary with roi_pct, simple_payback_years, npv.
        """
        rate = discount_rate or self._discount_rate
        simple_payback = _safe_divide(
            total_investment, annual_net_revenue, Decimal("99"),
        )

        total_net = annual_net_revenue * _decimal(project_life_years)
        roi_pct = _safe_pct(total_net - total_investment, total_investment)

        # NPV
        npv = -total_investment
        for t in range(1, project_life_years + 1):
            discount_factor = (Decimal("1") + rate) ** _decimal(t)
            npv += _safe_divide(annual_net_revenue, discount_factor)

        return {
            "roi_pct": _round_val(roi_pct, 2),
            "simple_payback_years": _round_val(simple_payback, 2),
            "npv": _round_val(npv, 2),
            "total_net_revenue": _round_val(total_net, 2),
            "provenance_hash": _compute_hash({
                "investment": str(total_investment),
                "annual_net": str(annual_net_revenue),
                "life": project_life_years,
                "rate": str(rate),
            }),
        }

    # ------------------------------------------------------------------ #
    # Internal: Cash Flow Construction                                    #
    # ------------------------------------------------------------------ #

    def _build_cash_flows(
        self,
        base_gross: Decimal,
        annual_penalties: Decimal,
        one_time_costs: Decimal,
        recurring_costs: Decimal,
        rev_escalation: Decimal,
        discount_rate: Decimal,
        project_life: int,
    ) -> List[AnnualCashFlow]:
        """Build year-by-year cash flow projections.

        Args:
            base_gross: Year 1 gross revenue.
            annual_penalties: Annual penalties.
            one_time_costs: One-time enabling costs (year 0).
            recurring_costs: Annual recurring costs.
            rev_escalation: Revenue escalation rate.
            discount_rate: Discount rate.
            project_life: Number of years.

        Returns:
            List of AnnualCashFlow objects.
        """
        flows: List[AnnualCashFlow] = []
        cumulative_net = Decimal("0")
        cumulative_disc = Decimal("0")

        for t in range(1, project_life + 1):
            # Revenue with escalation
            esc_factor = (Decimal("1") + rev_escalation) ** _decimal(t - 1)
            gross = base_gross * esc_factor

            # Penalties (constant or escalated)
            penalties = annual_penalties

            # Enabling costs: one-time in year 1 only
            enabling = one_time_costs if t == 1 else Decimal("0")

            # Operating costs with cost escalation
            cost_esc = (Decimal("1") + self._cost_escalation) ** _decimal(t - 1)
            opex = recurring_costs * cost_esc

            # Net
            net = gross - penalties - enabling - opex
            cumulative_net += net

            # Discounted
            disc_factor = (Decimal("1") + discount_rate) ** _decimal(t)
            disc_net = _safe_divide(net, disc_factor)
            cumulative_disc += disc_net

            flows.append(AnnualCashFlow(
                year=t,
                gross_revenue=_round_val(gross, 2),
                penalties=_round_val(penalties, 2),
                enabling_costs=_round_val(enabling, 2),
                operating_costs=_round_val(opex, 2),
                net_revenue=_round_val(net, 2),
                cumulative_net=_round_val(cumulative_net, 2),
                discounted_net=_round_val(disc_net, 2),
                cumulative_discounted=_round_val(cumulative_disc, 2),
            ))

        return flows

    # ------------------------------------------------------------------ #
    # Internal: NPV / IRR                                                 #
    # ------------------------------------------------------------------ #

    def _calculate_npv(
        self,
        cash_flows: List[AnnualCashFlow],
        investment: Decimal,
        discount_rate: Decimal,
    ) -> Decimal:
        """Calculate net present value.

        NPV = -investment + sum(discounted_net_t)

        Args:
            cash_flows: Year-by-year projections.
            investment: Total initial investment.
            discount_rate: Discount rate.

        Returns:
            NPV value.
        """
        pv_savings = sum(
            (cf.discounted_net for cf in cash_flows), Decimal("0")
        )
        return pv_savings - investment

    def _calculate_irr(
        self,
        cash_flows: List[AnnualCashFlow],
        investment: Decimal,
    ) -> Decimal:
        """Calculate IRR via bisection method.

        Args:
            cash_flows: Year-by-year net revenue (undiscounted).
            investment: Total investment.

        Returns:
            IRR as percentage.
        """
        if investment <= Decimal("0"):
            return Decimal("0")

        yearly_net = [cf.net_revenue for cf in cash_flows]
        if not yearly_net or all(n <= Decimal("0") for n in yearly_net):
            return Decimal("0")

        lo = Decimal("-0.50")
        hi = Decimal("5.00")

        for _ in range(MAX_IRR_ITERATIONS):
            mid = (lo + hi) / Decimal("2")
            npv_mid = -investment
            for t, net in enumerate(yearly_net, start=1):
                denom = (Decimal("1") + mid) ** _decimal(t)
                if denom != Decimal("0"):
                    npv_mid += _safe_divide(net, denom)

            if abs(npv_mid) < Decimal("1"):
                return mid * Decimal("100")
            elif npv_mid > Decimal("0"):
                lo = mid
            else:
                hi = mid

            if abs(hi - lo) < IRR_TOLERANCE:
                break

        return ((lo + hi) / Decimal("2")) * Decimal("100")

    # ------------------------------------------------------------------ #
    # Internal: Scenario Application                                      #
    # ------------------------------------------------------------------ #

    def _apply_scenario(
        self,
        financials: ProgrammeFinancials,
        parameter: ScenarioParameter,
        value: Decimal,
    ) -> ProgrammeFinancials:
        """Create adjusted financials for a what-if scenario.

        Args:
            financials: Base case financials.
            parameter: Parameter to vary.
            value: New value.

        Returns:
            Adjusted ProgrammeFinancials.
        """
        data = financials.model_dump()

        if parameter == ScenarioParameter.EVENT_FREQUENCY:
            data["events_per_year"] = int(value)

        elif parameter == ScenarioParameter.CAPACITY_PRICE:
            for stream in data.get("revenue_streams", []):
                if stream["stream_type"] == RevenueStreamType.CAPACITY_PAYMENT.value:
                    stream["unit_rate"] = value

        elif parameter == ScenarioParameter.ENERGY_PRICE:
            for stream in data.get("revenue_streams", []):
                if stream["stream_type"] == RevenueStreamType.ENERGY_PAYMENT.value:
                    stream["unit_rate"] = value

        elif parameter == ScenarioParameter.COMPLIANCE_RATE:
            for stream in data.get("revenue_streams", []):
                stream["compliance_factor"] = value

        elif parameter == ScenarioParameter.ENABLING_COST:
            data["capex"] = value

        elif parameter == ScenarioParameter.CURTAILMENT_KW:
            data["nominated_kw"] = value
            for stream in data.get("revenue_streams", []):
                if stream["stream_type"] == RevenueStreamType.CAPACITY_PAYMENT.value:
                    stream["unit_quantity"] = value

        elif parameter == ScenarioParameter.PENALTY_RATE:
            data["annual_penalties"] = value

        return ProgrammeFinancials(**data)

    def _get_base_value(
        self,
        financials: ProgrammeFinancials,
        parameter: ScenarioParameter,
    ) -> Decimal:
        """Extract the current value of a scenario parameter.

        Args:
            financials: Programme financials.
            parameter: Parameter to extract.

        Returns:
            Current value.
        """
        if parameter == ScenarioParameter.EVENT_FREQUENCY:
            return _decimal(financials.events_per_year)
        if parameter == ScenarioParameter.COMPLIANCE_RATE:
            if financials.revenue_streams:
                return financials.revenue_streams[0].compliance_factor
            return Decimal("1.0")
        if parameter == ScenarioParameter.ENABLING_COST:
            return financials.capex
        if parameter == ScenarioParameter.CURTAILMENT_KW:
            return financials.nominated_kw
        if parameter == ScenarioParameter.PENALTY_RATE:
            return financials.annual_penalties
        if parameter == ScenarioParameter.CAPACITY_PRICE:
            for s in financials.revenue_streams:
                if s.stream_type == RevenueStreamType.CAPACITY_PAYMENT:
                    return s.unit_rate
        if parameter == ScenarioParameter.ENERGY_PRICE:
            for s in financials.revenue_streams:
                if s.stream_type == RevenueStreamType.ENERGY_PAYMENT:
                    return s.unit_rate
        return Decimal("0")

    # ------------------------------------------------------------------ #
    # Internal: Optimisation Helpers                                      #
    # ------------------------------------------------------------------ #

    def _sort_by_objective(
        self,
        forecasts: List[RevenueForecast],
        objective: OptimisationObjective,
    ) -> List[RevenueForecast]:
        """Sort forecasts by optimisation objective.

        Args:
            forecasts: List of programme forecasts.
            objective: Optimisation objective.

        Returns:
            Sorted list (best first).
        """
        if objective == OptimisationObjective.MAXIMIZE_NET_REVENUE:
            return sorted(forecasts, key=lambda f: f.total_net_revenue, reverse=True)
        if objective == OptimisationObjective.MAXIMIZE_ROI:
            return sorted(forecasts, key=lambda f: f.roi_pct, reverse=True)
        if objective == OptimisationObjective.MINIMIZE_PAYBACK:
            return sorted(forecasts, key=lambda f: f.simple_payback_years)
        if objective == OptimisationObjective.MAXIMIZE_NPV:
            return sorted(forecasts, key=lambda f: f.npv, reverse=True)
        return forecasts

    def _generate_recommendations(
        self,
        forecasts: List[RevenueForecast],
        programmes: List[ProgrammeFinancials],
        objective: OptimisationObjective,
    ) -> List[str]:
        """Generate portfolio optimisation recommendations.

        Args:
            forecasts: Programme forecasts.
            programmes: Programme definitions.
            objective: Optimisation objective.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        profitable = [f for f in forecasts if f.is_profitable]
        unprofitable = [f for f in forecasts if not f.is_profitable]

        if unprofitable:
            recs.append(
                f"{len(unprofitable)} of {len(forecasts)} programmes have "
                f"negative NPV. Consider renegotiating terms or exiting."
            )

        if profitable:
            best = max(profitable, key=lambda f: f.roi_pct)
            recs.append(
                f"Highest ROI programme ({best.programme_id}): "
                f"{float(best.roi_pct):.1f}%. Consider increasing "
                f"nominated capacity in this programme."
            )

        short_payback = [f for f in forecasts if f.simple_payback_years < Decimal("3")]
        if short_payback:
            recs.append(
                f"{len(short_payback)} programmes have <3-year payback. "
                f"Prioritise capacity allocation to these programmes."
            )

        total_penalties = sum(
            (f.total_penalties for f in forecasts), Decimal("0")
        )
        if total_penalties > Decimal("0"):
            recs.append(
                f"Portfolio-wide penalties total ${float(total_penalties):,.2f}. "
                f"Invest in reliability improvements to reduce non-compliance."
            )

        if not recs:
            recs.append(
                "Portfolio is well-optimised. Monitor market conditions "
                "for additional programme opportunities."
            )

        return recs
