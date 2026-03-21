# -*- coding: utf-8 -*-
"""
PaybackCalculatorEngine - PACK-033 Quick Wins Identifier Engine 2
=================================================================

Comprehensive financial analysis engine for quick-win energy efficiency
measures.  Computes simple payback, discounted payback, NPV, IRR, ROI,
LCOE, SIR (savings-to-investment ratio), and full year-by-year cash-flow
projections with energy-price escalation, tax depreciation benefits, and
incentive offsets.

Supports batch analysis across a portfolio of measures with aggregate
metrics (total NPV, portfolio IRR, portfolio payback) and sensitivity
analysis over five key parameters (discount rate, energy price,
implementation cost, savings estimate, incentive amount).

Calculation Methodology:
    NPV  = -net_cost + sum( net_savings_t / (1 + r)^t  for t in 1..n )
    IRR  = r  such that NPV(r) = 0       (bisection, 100 iterations max)
    Simple Payback         = net_cost / annual_net_savings
    Discounted Payback     = first year t where cumulative_discounted >= net_cost
    LCOE = annualized_cost / annual_savings_kwh
           where annualized_cost = net_cost * CRF
    CRF  = r * (1+r)^n / ((1+r)^n - 1)   (Capital Recovery Factor)
    SIR  = PV(savings) / PV(costs)         (Savings-to-Investment Ratio)
    ROI  = (total_net_savings - net_cost) / net_cost * 100

    Tax Depreciation:
        SECTION_179      - full deduction in year 1
        MACRS_5Y         - 5-year schedule  (20/32/19.2/11.52/11.52/5.76)
        MACRS_7Y         - 7-year schedule  (14.29/24.49/17.49/12.49/8.93/8.92/8.93/4.46)
        MACRS_15Y        - 15-year schedule (5/9.5/8.55/7.70/6.93/6.23/5.90/5.90/5.91/
                                             5.90/5.91/5.90/5.91/5.90/5.91/2.95)
        BONUS_DEPRECIATION - 100 % deduction in year 1

    Incentives:
        Fixed-amount or percentage-of-cost rebates / credits / grants
        applied to reduce net implementation cost.

Regulatory References:
    - ASHRAE Standard 90.1 - Energy Standard for Buildings
    - ISO 50001:2018 - Energy management systems
    - IRS Publication 946 - How to Depreciate Property (MACRS tables)
    - DOE Better Buildings - Financial Analysis Methodologies
    - EU EED Article 8 - Energy audit financial assessment

Zero-Hallucination:
    - All formulas are standard engineering economics (no LLM in calc path)
    - MACRS schedules sourced from IRS Publication 946
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-033 Quick Wins Identifier
Engine:  2 of 8
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


class AnalysisPeriod(str, Enum):
    """Time horizon for financial analysis.

    SHORT_TERM_3Y:  3-year analysis window.
    MEDIUM_TERM_5Y: 5-year analysis window.
    LONG_TERM_10Y:  10-year analysis window (default).
    EXTENDED_15Y:   15-year analysis window.
    CUSTOM:         User-defined period.
    """
    SHORT_TERM_3Y = "short_term_3y"
    MEDIUM_TERM_5Y = "medium_term_5y"
    LONG_TERM_10Y = "long_term_10y"
    EXTENDED_15Y = "extended_15y"
    CUSTOM = "custom"


class FinancialMetric(str, Enum):
    """Supported financial metrics.

    SIMPLE_PAYBACK:    Net cost / annual net savings.
    DISCOUNTED_PAYBACK: Year where cumulative discounted savings >= net cost.
    NPV:               Net present value of all cash flows.
    IRR:               Internal rate of return (bisection method).
    ROI:               Return on investment percentage.
    LCOE:              Levelised cost of energy saved.
    SIR:               Savings-to-investment ratio (PV savings / PV costs).
    ANNUAL_CASH_FLOW:  Year-by-year cash flow projection.
    """
    SIMPLE_PAYBACK = "simple_payback"
    DISCOUNTED_PAYBACK = "discounted_payback"
    NPV = "npv"
    IRR = "irr"
    ROI = "roi"
    LCOE = "lcoe"
    SIR = "sir"
    ANNUAL_CASH_FLOW = "annual_cash_flow"


class TaxTreatment(str, Enum):
    """Tax depreciation treatment for capital expenditure.

    NONE:                No depreciation benefit.
    SECTION_179:         Full deduction in year 1 (IRS Section 179).
    MACRS_5Y:            5-year MACRS schedule.
    MACRS_7Y:            7-year MACRS schedule.
    MACRS_15Y:           15-year MACRS schedule.
    BONUS_DEPRECIATION:  100 % bonus depreciation in year 1.
    CUSTOM:              User-defined depreciation schedule.
    """
    NONE = "none"
    SECTION_179 = "section_179"
    MACRS_5Y = "macrs_5y"
    MACRS_7Y = "macrs_7y"
    MACRS_15Y = "macrs_15y"
    BONUS_DEPRECIATION = "bonus_depreciation"
    CUSTOM = "custom"


class IncentiveType(str, Enum):
    """Types of financial incentives.

    UTILITY_REBATE:        Cash rebate from utility company.
    TAX_CREDIT:            Direct tax credit (reduces tax liability).
    TAX_DEDUCTION:         Tax deduction (reduces taxable income).
    GRANT:                 Direct government or institutional grant.
    LOAN_SUBSIDY:          Subsidised / below-market-rate financing.
    PERFORMANCE_INCENTIVE: Performance-based incentive payment.
    """
    UTILITY_REBATE = "utility_rebate"
    TAX_CREDIT = "tax_credit"
    TAX_DEDUCTION = "tax_deduction"
    GRANT = "grant"
    LOAN_SUBSIDY = "loan_subsidy"
    PERFORMANCE_INCENTIVE = "performance_incentive"


class SensitivityParameter(str, Enum):
    """Parameters available for sensitivity analysis.

    DISCOUNT_RATE:        Vary the discount rate.
    ENERGY_PRICE:         Vary energy price / escalation.
    IMPLEMENTATION_COST:  Vary the implementation cost.
    SAVINGS_ESTIMATE:     Vary the annual savings estimate.
    INCENTIVE_AMOUNT:     Vary the incentive amount.
    """
    DISCOUNT_RATE = "discount_rate"
    ENERGY_PRICE = "energy_price"
    IMPLEMENTATION_COST = "implementation_cost"
    SAVINGS_ESTIMATE = "savings_estimate"
    INCENTIVE_AMOUNT = "incentive_amount"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DISCOUNT_RATE: Decimal = Decimal("0.08")
DEFAULT_INFLATION_RATE: Decimal = Decimal("0.025")
DEFAULT_ELEC_ESCALATION: Decimal = Decimal("0.03")
DEFAULT_GAS_ESCALATION: Decimal = Decimal("0.025")
DEFAULT_TAX_RATE: Decimal = Decimal("0.21")
DEFAULT_ANALYSIS_YEARS: int = 10
MAX_IRR_ITERATIONS: int = 100
IRR_TOLERANCE: Decimal = Decimal("0.0001")

# MACRS depreciation schedules (IRS Publication 946).
MACRS_5Y_SCHEDULE: List[Decimal] = [
    Decimal("0.2000"), Decimal("0.3200"), Decimal("0.1920"),
    Decimal("0.1152"), Decimal("0.1152"), Decimal("0.0576"),
]

MACRS_7Y_SCHEDULE: List[Decimal] = [
    Decimal("0.1429"), Decimal("0.2449"), Decimal("0.1749"),
    Decimal("0.1249"), Decimal("0.0893"), Decimal("0.0892"),
    Decimal("0.0893"), Decimal("0.0446"),
]

MACRS_15Y_SCHEDULE: List[Decimal] = [
    Decimal("0.0500"), Decimal("0.0950"), Decimal("0.0855"),
    Decimal("0.0770"), Decimal("0.0693"), Decimal("0.0623"),
    Decimal("0.0590"), Decimal("0.0590"), Decimal("0.0591"),
    Decimal("0.0590"), Decimal("0.0591"), Decimal("0.0590"),
    Decimal("0.0591"), Decimal("0.0590"), Decimal("0.0591"),
    Decimal("0.0295"),
]

# Map AnalysisPeriod enum to years.
PERIOD_YEARS: Dict[str, int] = {
    AnalysisPeriod.SHORT_TERM_3Y.value: 3,
    AnalysisPeriod.MEDIUM_TERM_5Y.value: 5,
    AnalysisPeriod.LONG_TERM_10Y.value: 10,
    AnalysisPeriod.EXTENDED_15Y.value: 15,
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class MeasureFinancials(BaseModel):
    """Financial data for an energy efficiency measure.

    Attributes:
        measure_id: Unique measure identifier.
        name: Human-readable measure name.
        implementation_cost: Total implementation cost (currency units).
        annual_savings_kwh: Annual electricity savings (kWh).
        annual_savings_therms: Annual natural gas savings (therms).
        annual_savings_cost: Annual cost savings (currency units).
        annual_maintenance_cost: Annual incremental maintenance cost.
        measure_life_years: Economic useful life of the measure (years).
        salvage_value: Residual / salvage value at end of life.
    """
    measure_id: str = Field(
        default_factory=_new_uuid, description="Unique measure identifier"
    )
    name: str = Field(
        default="", max_length=500, description="Measure name"
    )
    implementation_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total implementation cost"
    )
    annual_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual electricity savings (kWh)"
    )
    annual_savings_therms: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual gas savings (therms)"
    )
    annual_savings_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual cost savings"
    )
    annual_maintenance_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual maintenance cost"
    )
    measure_life_years: int = Field(
        default=10, ge=1, le=40, description="Measure economic life (years)"
    )
    salvage_value: Decimal = Field(
        default=Decimal("0"), ge=0, description="Salvage value at end of life"
    )

    @field_validator("implementation_cost", "annual_savings_cost")
    @classmethod
    def validate_positive_cost(cls, v: Decimal) -> Decimal:
        """Ensure cost values are non-negative."""
        if v < Decimal("0"):
            raise ValueError("Cost values must be >= 0")
        return v


class FinancialParameters(BaseModel):
    """Parameters governing the financial analysis.

    Attributes:
        discount_rate: Nominal discount rate (e.g. 0.08 = 8 %).
        inflation_rate: General inflation rate.
        electricity_escalation_rate: Annual electricity price escalation.
        gas_escalation_rate: Annual gas price escalation.
        analysis_period_years: Number of years to analyse.
        tax_rate: Marginal tax rate for depreciation benefits.
        tax_treatment: Depreciation method to apply.
    """
    discount_rate: Decimal = Field(
        default=DEFAULT_DISCOUNT_RATE, ge=0, le=Decimal("0.50"),
        description="Discount rate"
    )
    inflation_rate: Decimal = Field(
        default=DEFAULT_INFLATION_RATE, ge=0, le=Decimal("0.20"),
        description="Inflation rate"
    )
    electricity_escalation_rate: Decimal = Field(
        default=DEFAULT_ELEC_ESCALATION, ge=Decimal("-0.05"), le=Decimal("0.15"),
        description="Electricity price escalation rate"
    )
    gas_escalation_rate: Decimal = Field(
        default=DEFAULT_GAS_ESCALATION, ge=Decimal("-0.05"), le=Decimal("0.15"),
        description="Gas price escalation rate"
    )
    analysis_period_years: int = Field(
        default=DEFAULT_ANALYSIS_YEARS, ge=1, le=40,
        description="Analysis period (years)"
    )
    tax_rate: Decimal = Field(
        default=DEFAULT_TAX_RATE, ge=0, le=Decimal("0.50"),
        description="Marginal tax rate"
    )
    tax_treatment: TaxTreatment = Field(
        default=TaxTreatment.NONE, description="Tax depreciation treatment"
    )

    @field_validator("tax_treatment", mode="before")
    @classmethod
    def validate_tax_treatment(cls, v: Any) -> Any:
        """Accept string values for TaxTreatment."""
        if isinstance(v, str):
            valid = {t.value for t in TaxTreatment}
            if v not in valid:
                raise ValueError(
                    f"Unknown tax treatment '{v}'. Must be one of: {sorted(valid)}"
                )
        return v


class Incentive(BaseModel):
    """Financial incentive that reduces effective project cost.

    Attributes:
        incentive_type: Type of incentive.
        amount: Incentive amount (currency units or percentage).
        is_percentage: If True, *amount* is a percentage of cost (0-100).
        max_amount: Cap on incentive value when using percentage.
        year_received: Year in which the incentive is received (0 = upfront).
    """
    incentive_type: IncentiveType = Field(
        ..., description="Type of incentive"
    )
    amount: Decimal = Field(
        default=Decimal("0"), ge=0, description="Incentive amount"
    )
    is_percentage: bool = Field(
        default=False, description="True if amount is a percentage of cost"
    )
    max_amount: Optional[Decimal] = Field(
        default=None, ge=0, description="Maximum incentive cap"
    )
    year_received: int = Field(
        default=0, ge=0, le=40, description="Year incentive is received"
    )

    @field_validator("incentive_type", mode="before")
    @classmethod
    def validate_incentive_type(cls, v: Any) -> Any:
        """Accept string values for IncentiveType."""
        if isinstance(v, str):
            valid = {t.value for t in IncentiveType}
            if v not in valid:
                raise ValueError(
                    f"Unknown incentive type '{v}'. Must be one of: {sorted(valid)}"
                )
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class CashFlow(BaseModel):
    """Year-by-year cash-flow projection for a measure.

    Attributes:
        year: Year number (1-based).
        gross_savings: Annual savings before escalation in year 1 terms.
        escalated_savings: Savings after applying energy-price escalation.
        maintenance_cost: Annual maintenance cost.
        net_savings: Escalated savings minus maintenance.
        cumulative_net_savings: Running sum of net_savings.
        discounted_net_savings: PV of this year's net savings.
        cumulative_discounted: Running sum of discounted net savings.
        depreciation_benefit: Tax shield from depreciation in this year.
        tax_benefit: Total tax benefit in this year.
    """
    year: int = Field(default=0, ge=0, description="Year number")
    gross_savings: Decimal = Field(default=Decimal("0"))
    escalated_savings: Decimal = Field(default=Decimal("0"))
    maintenance_cost: Decimal = Field(default=Decimal("0"))
    net_savings: Decimal = Field(default=Decimal("0"))
    cumulative_net_savings: Decimal = Field(default=Decimal("0"))
    discounted_net_savings: Decimal = Field(default=Decimal("0"))
    cumulative_discounted: Decimal = Field(default=Decimal("0"))
    depreciation_benefit: Decimal = Field(default=Decimal("0"))
    tax_benefit: Decimal = Field(default=Decimal("0"))


class PaybackResult(BaseModel):
    """Complete payback analysis result for a single measure.

    Attributes:
        measure_id: Measure identifier.
        simple_payback_years: Simple payback period (years).
        discounted_payback_years: Discounted payback period (years).
        npv: Net present value.
        irr: Internal rate of return (percentage).
        roi_pct: Return on investment (percentage).
        lcoe: Levelised cost of energy saved (cost / kWh).
        sir: Savings-to-investment ratio.
        net_implementation_cost: Cost after subtracting incentives.
        total_savings: Total undiscounted net savings over analysis period.
        total_discounted_savings: Total PV of net savings.
        cash_flows: Year-by-year cash flow projections.
        is_cost_effective: True when NPV > 0 and SIR >= 1.0.
        calculated_at: Timestamp of calculation.
        provenance_hash: SHA-256 audit hash.
    """
    measure_id: str = Field(default="")
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    discounted_payback_years: Decimal = Field(default=Decimal("0"))
    npv: Decimal = Field(default=Decimal("0"))
    irr: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))
    lcoe: Decimal = Field(default=Decimal("0"))
    sir: Decimal = Field(default=Decimal("0"))
    net_implementation_cost: Decimal = Field(default=Decimal("0"))
    total_savings: Decimal = Field(default=Decimal("0"))
    total_discounted_savings: Decimal = Field(default=Decimal("0"))
    cash_flows: List[CashFlow] = Field(default_factory=list)
    is_cost_effective: bool = Field(default=False)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class BatchPaybackResult(BaseModel):
    """Aggregated payback analysis for a portfolio of measures.

    Attributes:
        results: Individual PaybackResult per measure.
        total_investment: Sum of net implementation costs.
        total_npv: Sum of individual NPVs.
        portfolio_irr: Weighted-average IRR across portfolio.
        portfolio_simple_payback: Portfolio-level simple payback.
        calculated_at: Timestamp of calculation.
        provenance_hash: SHA-256 audit hash.
    """
    results: List[PaybackResult] = Field(default_factory=list)
    total_investment: Decimal = Field(default=Decimal("0"))
    total_npv: Decimal = Field(default=Decimal("0"))
    portfolio_irr: Decimal = Field(default=Decimal("0"))
    portfolio_simple_payback: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class SensitivityResult(BaseModel):
    """Sensitivity analysis result for a single parameter.

    Attributes:
        parameter: The parameter being varied.
        values: List of parameter values tested.
        npvs: Corresponding NPV at each value.
        irrs: Corresponding IRR at each value.
        paybacks: Corresponding simple payback at each value.
        breakeven_value: Parameter value where NPV = 0, if found.
        provenance_hash: SHA-256 audit hash.
    """
    parameter: SensitivityParameter = Field(...)
    values: List[Decimal] = Field(default_factory=list)
    npvs: List[Decimal] = Field(default_factory=list)
    irrs: List[Decimal] = Field(default_factory=list)
    paybacks: List[Decimal] = Field(default_factory=list)
    breakeven_value: Optional[Decimal] = Field(default=None)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PaybackCalculatorEngine:
    """Financial analysis engine for quick-win energy efficiency measures.

    Computes simple / discounted payback, NPV, IRR, ROI, LCOE, SIR for
    individual measures and portfolios.  Supports year-by-year cash-flow
    projections with energy-price escalation, MACRS / Section 179
    depreciation, and incentive offsets.

    Usage::

        engine = PaybackCalculatorEngine()
        result = engine.calculate_payback(measure, params)
        print(f"Simple payback: {result.simple_payback_years} years")
        print(f"NPV: {result.npv}")

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise PaybackCalculatorEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - discount_rate (Decimal): default discount rate
                - tax_rate (Decimal): default marginal tax rate
                - analysis_years (int): default analysis period
        """
        self.config = config or {}
        self._default_discount = _decimal(
            self.config.get("discount_rate", DEFAULT_DISCOUNT_RATE)
        )
        self._default_tax_rate = _decimal(
            self.config.get("tax_rate", DEFAULT_TAX_RATE)
        )
        self._default_years = int(
            self.config.get("analysis_years", DEFAULT_ANALYSIS_YEARS)
        )
        logger.info(
            "PaybackCalculatorEngine v%s initialised (discount=%.2f, tax=%.2f)",
            self.engine_version,
            float(self._default_discount),
            float(self._default_tax_rate),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate_payback(
        self,
        measure: MeasureFinancials,
        params: FinancialParameters,
        incentives: Optional[List[Incentive]] = None,
    ) -> PaybackResult:
        """Compute full payback analysis for a single measure.

        Workflow:
            1. Apply incentives to reduce net implementation cost.
            2. Build year-by-year cash flows with escalation and depreciation.
            3. Compute simple payback, discounted payback, NPV.
            4. Compute IRR via bisection.
            5. Compute ROI, LCOE, SIR.
            6. Generate SHA-256 provenance hash.

        Args:
            measure: Financial data for the measure.
            params: Financial parameters governing the analysis.
            incentives: Optional list of applicable incentives.

        Returns:
            PaybackResult with all metrics and cash flows.
        """
        t0 = time.perf_counter()
        logger.info(
            "Payback calculation: measure=%s, cost=%s, savings=%s",
            measure.measure_id, measure.implementation_cost,
            measure.annual_savings_cost,
        )

        # Step 1 -- Net cost after incentives
        net_cost = self._apply_incentives(
            measure.implementation_cost, incentives or []
        )
        net_cost = max(net_cost, Decimal("0"))

        # Step 2 -- Build cash flows
        analysis_years = min(params.analysis_period_years, measure.measure_life_years)
        cash_flows = self._build_cash_flows(measure, params, net_cost, analysis_years)

        # Step 3a -- Simple payback
        annual_net = measure.annual_savings_cost - measure.annual_maintenance_cost
        simple_payback = _safe_divide(net_cost, annual_net, Decimal("99"))
        simple_payback = max(simple_payback, Decimal("0"))

        # Step 3b -- NPV
        npv = self._calculate_npv(cash_flows, params, net_cost)

        # Step 3c -- Discounted payback (from cash flow data)
        discounted_payback = self._find_discounted_payback(
            cash_flows, net_cost, analysis_years
        )

        # Step 3d -- Total savings
        total_savings = sum(
            (cf.net_savings for cf in cash_flows), Decimal("0")
        )
        total_discounted = sum(
            (cf.discounted_net_savings for cf in cash_flows), Decimal("0")
        )

        # Step 4 -- IRR
        irr = self._calculate_irr(cash_flows, net_cost)

        # Step 5a -- ROI
        roi_pct = _safe_pct(total_savings - net_cost, net_cost)

        # Step 5b -- LCOE
        lcoe = self._calculate_lcoe(
            net_cost, measure, params, analysis_years
        )

        # Step 5c -- SIR
        sir = _safe_divide(total_discounted, net_cost, Decimal("0"))

        # Step 6 -- Cost-effective flag
        is_cost_effective = (npv > Decimal("0")) and (sir >= Decimal("1"))

        # Round outputs
        result = PaybackResult(
            measure_id=measure.measure_id,
            simple_payback_years=_round_val(simple_payback, 2),
            discounted_payback_years=_round_val(discounted_payback, 2),
            npv=_round_val(npv, 2),
            irr=_round_val(irr, 2),
            roi_pct=_round_val(roi_pct, 2),
            lcoe=_round_val(lcoe, 4),
            sir=_round_val(sir, 4),
            net_implementation_cost=_round_val(net_cost, 2),
            total_savings=_round_val(total_savings, 2),
            total_discounted_savings=_round_val(total_discounted, 2),
            cash_flows=cash_flows,
            is_cost_effective=is_cost_effective,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Payback complete: measure=%s, payback=%.2f yr, NPV=%.2f, "
            "IRR=%.2f%%, cost_effective=%s, hash=%s (%.1f ms)",
            measure.measure_id,
            float(result.simple_payback_years),
            float(result.npv),
            float(result.irr),
            result.is_cost_effective,
            result.provenance_hash[:16],
            elapsed,
        )
        return result

    def calculate_batch(
        self,
        measures: List[MeasureFinancials],
        params: FinancialParameters,
        incentives: Optional[Dict[str, List[Incentive]]] = None,
    ) -> BatchPaybackResult:
        """Compute payback analysis for a portfolio of measures.

        Aggregates individual results into portfolio-level metrics:
        total investment, total NPV, weighted-average IRR, and
        portfolio simple payback.

        Args:
            measures: List of measures to analyse.
            params: Shared financial parameters.
            incentives: Map of measure_id -> list of incentives.

        Returns:
            BatchPaybackResult with individual and aggregate metrics.
        """
        t0 = time.perf_counter()
        incentives = incentives or {}
        logger.info(
            "Batch payback calculation: %d measures", len(measures)
        )

        results: List[PaybackResult] = []
        for m in measures:
            m_incentives = incentives.get(m.measure_id, [])
            pr = self.calculate_payback(m, params, m_incentives)
            results.append(pr)

        # Aggregate metrics
        total_investment = sum(
            (r.net_implementation_cost for r in results), Decimal("0")
        )
        total_npv = sum((r.npv for r in results), Decimal("0"))

        # Portfolio simple payback
        total_annual = sum(
            (m.annual_savings_cost - m.annual_maintenance_cost for m in measures),
            Decimal("0"),
        )
        portfolio_payback = _safe_divide(
            total_investment, total_annual, Decimal("99")
        )

        # Portfolio IRR (cost-weighted average)
        portfolio_irr = Decimal("0")
        if total_investment > Decimal("0"):
            for r in results:
                weight = _safe_divide(r.net_implementation_cost, total_investment)
                portfolio_irr += r.irr * weight

        batch = BatchPaybackResult(
            results=results,
            total_investment=_round_val(total_investment, 2),
            total_npv=_round_val(total_npv, 2),
            portfolio_irr=_round_val(portfolio_irr, 2),
            portfolio_simple_payback=_round_val(portfolio_payback, 2),
        )
        batch.provenance_hash = _compute_hash(batch)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Batch payback complete: %d measures, total_invest=%.2f, "
            "total_npv=%.2f, portfolio_payback=%.2f yr, hash=%s (%.1f ms)",
            len(results),
            float(total_investment),
            float(total_npv),
            float(portfolio_payback),
            batch.provenance_hash[:16],
            elapsed,
        )
        return batch

    def run_sensitivity(
        self,
        measure: MeasureFinancials,
        params: FinancialParameters,
        parameter: SensitivityParameter,
        values: List[Decimal],
    ) -> SensitivityResult:
        """Run sensitivity analysis varying a single parameter.

        Re-computes the payback analysis at each supplied value of the
        chosen parameter, recording NPV, IRR, and simple payback.  Also
        attempts to locate the break-even value (NPV = 0) via linear
        interpolation between adjacent positive/negative NPV results.

        Args:
            measure: Base measure financials.
            params: Base financial parameters.
            parameter: Which parameter to vary.
            values: List of values to evaluate.

        Returns:
            SensitivityResult with arrays of NPV, IRR, payback per value.
        """
        t0 = time.perf_counter()
        logger.info(
            "Sensitivity analysis: measure=%s, param=%s, %d values",
            measure.measure_id, parameter.value, len(values),
        )

        npvs: List[Decimal] = []
        irrs: List[Decimal] = []
        paybacks: List[Decimal] = []
        breakeven: Optional[Decimal] = None

        for val in values:
            adjusted_measure, adjusted_params = self._apply_sensitivity(
                measure, params, parameter, val
            )
            pr = self.calculate_payback(adjusted_measure, adjusted_params)
            npvs.append(pr.npv)
            irrs.append(pr.irr)
            paybacks.append(pr.simple_payback_years)

        # Find break-even (NPV crosses zero)
        breakeven = self._find_breakeven(values, npvs)

        result = SensitivityResult(
            parameter=parameter,
            values=[_round_val(v, 6) for v in values],
            npvs=[_round_val(n, 2) for n in npvs],
            irrs=[_round_val(i, 2) for i in irrs],
            paybacks=[_round_val(p, 2) for p in paybacks],
            breakeven_value=(
                _round_val(breakeven, 4) if breakeven is not None else None
            ),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Sensitivity complete: param=%s, breakeven=%s, hash=%s (%.1f ms)",
            parameter.value,
            str(breakeven) if breakeven is not None else "N/A",
            result.provenance_hash[:16],
            elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Cash Flow Construction                                              #
    # ------------------------------------------------------------------ #

    def _build_cash_flows(
        self,
        measure: MeasureFinancials,
        params: FinancialParameters,
        net_cost: Decimal,
        analysis_years: int,
    ) -> List[CashFlow]:
        """Build year-by-year cash-flow projections.

        Applies energy-price escalation to savings, subtracts maintenance,
        computes depreciation tax benefits, and tracks cumulative sums.

        Args:
            measure: Measure financials.
            params: Financial parameters.
            net_cost: Net implementation cost after incentives.
            analysis_years: Number of years to project.

        Returns:
            Ordered list of CashFlow objects (year 1 .. analysis_years).
        """
        flows: List[CashFlow] = []
        cumulative_net = Decimal("0")
        cumulative_disc = Decimal("0")

        gross_savings_yr1 = measure.annual_savings_cost
        maintenance = measure.annual_maintenance_cost

        # Blended escalation: weight by savings mix
        escalation = self._blended_escalation(measure, params)

        for t in range(1, analysis_years + 1):
            # Escalated savings
            esc_factor = (Decimal("1") + escalation) ** _decimal(t - 1)
            escalated = gross_savings_yr1 * esc_factor

            # Maintenance (escalated with inflation)
            maint_factor = (Decimal("1") + params.inflation_rate) ** _decimal(t - 1)
            maint = maintenance * maint_factor

            # Net savings
            net = escalated - maint

            # Depreciation benefit
            dep_benefit = self._calculate_depreciation(
                net_cost, t, params.tax_treatment
            )
            tax_benefit = dep_benefit * params.tax_rate

            # Cumulative
            cumulative_net += net
            # Discounted
            discount_factor = (Decimal("1") + params.discount_rate) ** _decimal(t)
            disc_net = _safe_divide(net + tax_benefit, discount_factor)
            cumulative_disc += disc_net

            flows.append(CashFlow(
                year=t,
                gross_savings=_round_val(gross_savings_yr1, 2),
                escalated_savings=_round_val(escalated, 2),
                maintenance_cost=_round_val(maint, 2),
                net_savings=_round_val(net, 2),
                cumulative_net_savings=_round_val(cumulative_net, 2),
                discounted_net_savings=_round_val(disc_net, 2),
                cumulative_discounted=_round_val(cumulative_disc, 2),
                depreciation_benefit=_round_val(dep_benefit, 2),
                tax_benefit=_round_val(tax_benefit, 2),
            ))

        # Add salvage value in final year (discounted)
        if measure.salvage_value > Decimal("0") and flows:
            last = flows[-1]
            t_final = analysis_years
            disc_factor = (Decimal("1") + params.discount_rate) ** _decimal(t_final)
            disc_salvage = _safe_divide(measure.salvage_value, disc_factor)
            last.net_savings += _round_val(measure.salvage_value, 2)
            last.cumulative_net_savings += _round_val(measure.salvage_value, 2)
            last.discounted_net_savings += _round_val(disc_salvage, 2)
            last.cumulative_discounted += _round_val(disc_salvage, 2)

        return flows

    def _blended_escalation(
        self,
        measure: MeasureFinancials,
        params: FinancialParameters,
    ) -> Decimal:
        """Compute a blended escalation rate based on savings mix.

        If a measure saves both electricity and gas, the escalation rate
        is the weighted average of the two rates by savings magnitude.

        Args:
            measure: Measure financials.
            params: Financial parameters.

        Returns:
            Blended escalation rate.
        """
        elec = measure.annual_savings_kwh
        gas = measure.annual_savings_therms
        total = elec + gas
        if total <= Decimal("0"):
            return params.electricity_escalation_rate

        elec_weight = _safe_divide(elec, total)
        gas_weight = _safe_divide(gas, total)
        blended = (
            elec_weight * params.electricity_escalation_rate
            + gas_weight * params.gas_escalation_rate
        )
        return blended

    # ------------------------------------------------------------------ #
    # NPV Calculation                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_npv(
        self,
        cash_flows: List[CashFlow],
        params: FinancialParameters,
        net_cost: Decimal,
    ) -> Decimal:
        """Calculate net present value from cash flows.

        NPV = -net_cost + sum( discounted_net_savings_t  for all t )

        Args:
            cash_flows: Year-by-year projections.
            params: Financial parameters.
            net_cost: Net implementation cost.

        Returns:
            NPV value.
        """
        pv_savings = sum(
            (cf.discounted_net_savings for cf in cash_flows), Decimal("0")
        )
        return pv_savings - net_cost

    # ------------------------------------------------------------------ #
    # IRR Calculation (Bisection)                                         #
    # ------------------------------------------------------------------ #

    def _calculate_irr(
        self,
        cash_flows: List[CashFlow],
        net_cost: Decimal,
    ) -> Decimal:
        """Calculate IRR via bisection method.

        Finds rate r where NPV(r) = 0 by testing midpoints between
        lo = -0.50 and hi = 5.00, up to 100 iterations.

        Args:
            cash_flows: Year-by-year net savings (undiscounted).
            net_cost: Net implementation cost.

        Returns:
            IRR as a percentage (e.g. 25.00 for 25 %).
        """
        if net_cost <= Decimal("0"):
            return Decimal("0")

        # Extract undiscounted net savings per year
        yearly_savings: List[Decimal] = [cf.net_savings for cf in cash_flows]
        if not yearly_savings or all(s <= Decimal("0") for s in yearly_savings):
            return Decimal("0")

        lo = Decimal("-0.50")
        hi = Decimal("5.00")

        for _ in range(MAX_IRR_ITERATIONS):
            mid = (lo + hi) / Decimal("2")
            npv_mid = -net_cost
            for t, savings in enumerate(yearly_savings, start=1):
                denom = (Decimal("1") + mid) ** _decimal(t)
                if denom != Decimal("0"):
                    npv_mid += _safe_divide(savings, denom)

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
    # Discounted Payback                                                  #
    # ------------------------------------------------------------------ #

    def _find_discounted_payback(
        self,
        cash_flows: List[CashFlow],
        net_cost: Decimal,
        analysis_years: int,
    ) -> Decimal:
        """Find the discounted payback period.

        The discounted payback is the year at which cumulative discounted
        savings first equal or exceed the net cost.  Linear interpolation
        is used within the crossover year.

        Args:
            cash_flows: Year-by-year projections.
            net_cost: Net implementation cost.
            analysis_years: Total analysis horizon.

        Returns:
            Discounted payback in years.
        """
        if net_cost <= Decimal("0"):
            return Decimal("0")

        cumulative = Decimal("0")
        prev_cumulative = Decimal("0")

        for cf in cash_flows:
            prev_cumulative = cumulative
            cumulative += cf.discounted_net_savings

            if cumulative >= net_cost:
                # Interpolate within this year
                remaining = net_cost - prev_cumulative
                fraction = _safe_divide(
                    remaining, cf.discounted_net_savings, Decimal("1")
                )
                return _decimal(cf.year - 1) + fraction

        # Never paid back within analysis period
        return _decimal(analysis_years)

    # ------------------------------------------------------------------ #
    # LCOE (Levelised Cost of Energy Saved)                               #
    # ------------------------------------------------------------------ #

    def _calculate_lcoe(
        self,
        net_cost: Decimal,
        measure: MeasureFinancials,
        params: FinancialParameters,
        analysis_years: int,
    ) -> Decimal:
        """Compute levelised cost of energy saved.

        LCOE = annualized_cost / annual_savings_kwh
        where annualized_cost = net_cost * CRF
              CRF = r*(1+r)^n / ((1+r)^n - 1)

        Args:
            net_cost: Net implementation cost.
            measure: Measure financials.
            params: Financial parameters.
            analysis_years: Years for annualisation.

        Returns:
            LCOE in cost per kWh saved.
        """
        if measure.annual_savings_kwh <= Decimal("0"):
            return Decimal("0")

        r = params.discount_rate
        n = _decimal(analysis_years)

        # CRF = r*(1+r)^n / ((1+r)^n - 1)
        if r <= Decimal("0"):
            # No discounting: annualized = cost / n
            annualized = _safe_divide(net_cost, n)
        else:
            compound = (Decimal("1") + r) ** n
            crf_num = r * compound
            crf_den = compound - Decimal("1")
            crf = _safe_divide(crf_num, crf_den, Decimal("0"))
            annualized = net_cost * crf

        # Add annual maintenance
        annualized += measure.annual_maintenance_cost

        return _safe_divide(annualized, measure.annual_savings_kwh)

    # ------------------------------------------------------------------ #
    # Depreciation Calculation                                            #
    # ------------------------------------------------------------------ #

    def _calculate_depreciation(
        self,
        cost: Decimal,
        year: int,
        treatment: TaxTreatment,
    ) -> Decimal:
        """Calculate depreciation deduction for a given year.

        Supports MACRS (5/7/15-year), Section 179 (full year-1 deduction),
        bonus depreciation (100 % year-1), and no depreciation.

        Args:
            cost: Depreciable cost basis.
            year: Year number (1-based).
            treatment: Tax treatment to apply.

        Returns:
            Depreciation amount for the year.
        """
        if treatment == TaxTreatment.NONE:
            return Decimal("0")

        if treatment == TaxTreatment.SECTION_179:
            return cost if year == 1 else Decimal("0")

        if treatment == TaxTreatment.BONUS_DEPRECIATION:
            return cost if year == 1 else Decimal("0")

        if treatment == TaxTreatment.MACRS_5Y:
            schedule = MACRS_5Y_SCHEDULE
        elif treatment == TaxTreatment.MACRS_7Y:
            schedule = MACRS_7Y_SCHEDULE
        elif treatment == TaxTreatment.MACRS_15Y:
            schedule = MACRS_15Y_SCHEDULE
        else:
            # CUSTOM or unknown: no depreciation
            return Decimal("0")

        idx = year - 1
        if idx < 0 or idx >= len(schedule):
            return Decimal("0")

        return cost * schedule[idx]

    # ------------------------------------------------------------------ #
    # Incentive Application                                               #
    # ------------------------------------------------------------------ #

    def _apply_incentives(
        self,
        cost: Decimal,
        incentives: List[Incentive],
    ) -> Decimal:
        """Apply incentives to reduce the implementation cost.

        Processes upfront incentives (year_received == 0) as direct cost
        reductions.  Percentage-based incentives are computed against the
        original cost and capped at max_amount if specified.

        Args:
            cost: Original implementation cost.
            incentives: List of applicable incentives.

        Returns:
            Net cost after applying upfront incentives.
        """
        net = cost

        for inc in incentives:
            # Only apply upfront incentives to net cost
            if inc.year_received != 0:
                continue

            if inc.is_percentage:
                reduction = cost * inc.amount / Decimal("100")
                if inc.max_amount is not None:
                    reduction = min(reduction, inc.max_amount)
            else:
                reduction = inc.amount

            net -= reduction
            logger.debug(
                "Applied incentive %s: -%.2f (type=%s)",
                inc.incentive_type.value, float(reduction),
                "pct" if inc.is_percentage else "fixed",
            )

        return net

    # ------------------------------------------------------------------ #
    # Sensitivity Helpers                                                 #
    # ------------------------------------------------------------------ #

    def _apply_sensitivity(
        self,
        measure: MeasureFinancials,
        params: FinancialParameters,
        parameter: SensitivityParameter,
        value: Decimal,
    ) -> Tuple[MeasureFinancials, FinancialParameters]:
        """Create adjusted copies of measure/params for sensitivity run.

        Args:
            measure: Original measure.
            params: Original parameters.
            parameter: Which parameter to vary.
            value: New value for the parameter.

        Returns:
            Tuple of (adjusted_measure, adjusted_params).
        """
        m_data = measure.model_dump()
        p_data = params.model_dump()

        if parameter == SensitivityParameter.DISCOUNT_RATE:
            p_data["discount_rate"] = value

        elif parameter == SensitivityParameter.ENERGY_PRICE:
            p_data["electricity_escalation_rate"] = value

        elif parameter == SensitivityParameter.IMPLEMENTATION_COST:
            m_data["implementation_cost"] = value

        elif parameter == SensitivityParameter.SAVINGS_ESTIMATE:
            m_data["annual_savings_cost"] = value

        elif parameter == SensitivityParameter.INCENTIVE_AMOUNT:
            # Adjust implementation cost downward by incentive amount
            original_cost = _decimal(m_data["implementation_cost"])
            m_data["implementation_cost"] = max(
                original_cost - value, Decimal("0")
            )

        return (
            MeasureFinancials(**m_data),
            FinancialParameters(**p_data),
        )

    def _find_breakeven(
        self,
        values: List[Decimal],
        npvs: List[Decimal],
    ) -> Optional[Decimal]:
        """Find the break-even parameter value where NPV crosses zero.

        Uses linear interpolation between adjacent positive/negative NPV
        data points.

        Args:
            values: Parameter values tested.
            npvs: Corresponding NPV at each value.

        Returns:
            Interpolated break-even value, or None if no crossover.
        """
        if len(values) < 2:
            return None

        for i in range(len(npvs) - 1):
            npv_a = npvs[i]
            npv_b = npvs[i + 1]

            # Check for sign change
            if (npv_a >= Decimal("0") and npv_b < Decimal("0")) or \
               (npv_a < Decimal("0") and npv_b >= Decimal("0")):
                # Linear interpolation
                val_a = values[i]
                val_b = values[i + 1]
                denom = npv_a - npv_b
                if denom == Decimal("0"):
                    continue
                fraction = npv_a / denom
                breakeven = val_a + fraction * (val_b - val_a)
                return breakeven

        return None
