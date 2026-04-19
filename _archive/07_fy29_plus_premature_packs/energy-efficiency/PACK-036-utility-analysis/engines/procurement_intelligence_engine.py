# -*- coding: utf-8 -*-
"""
ProcurementIntelligenceEngine - PACK-036 Utility Analysis Engine 6
===================================================================

Provides energy procurement analytics including contract comparison,
load-weighted pricing, price-risk assessment, procurement planning,
green product evaluation, supplier scoring, market condition analysis,
hedge-value calculation, and forward-curve cost projection.  Designed
for commercial and industrial energy buyers who need deterministic,
audit-ready procurement decision support.

Calculation Methodology:
    Load-Weighted Price:
        LWP = Sum(Interval_kWh * Interval_Price) / Sum(Interval_kWh)

    Profile Premium:
        profile_premium_pct = (LWP - Baseload_Price) / Baseload_Price * 100

    VaR (Parametric, 95%):
        VaR_95 = mu + 1.645 * sigma
        VaR_99 = mu + 2.326 * sigma

    CVaR (Conditional VaR):
        CVaR_alpha = mu + sigma * phi(z_alpha) / (1 - alpha)
        where phi is the standard normal PDF, z_alpha is the quantile.

    Hedge Value:
        hedge_value = Sum((Fixed_Price - Index_Price) * Volume_per_period)
        Positive = hedge saved money vs. spot exposure.

    Block & Index Cost:
        cost = Block_Vol * Block_Price + Index_Vol * Index_Price
               + Adder * Total_Volume

    Contract Cost (Fixed):
        cost = Volume_MWh * Fixed_Price_per_MWh

    Contract Cost (Index):
        cost = Sum(Interval_Vol * Index_Price) + Adder * Total_Volume

    Supplier Overall Score:
        score = Sum(criteria_weight_i * normalised_score_i)

Regulatory / Standard References:
    - EFET European Framework Agreement for Electricity (2024)
    - ISDA Master Agreement for Energy Derivatives
    - EU Directive 2019/944 Internal Electricity Market
    - EU Guarantees of Origin (Directive 2018/2001 RED II)
    - RE100 Technical Criteria (2024)
    - GHG Protocol Scope 2 Guidance (Market-Based Method)
    - ISO 50001:2018 Energy Management Systems (Procurement clause)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - VaR/CVaR computed from parametric formulas (no Monte Carlo)
    - Contract costs calculated from explicit price * volume
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-036 Utility Analysis
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
from datetime import datetime, date, timezone
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

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* decimal digits and return float."""
    return float(value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))

def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def _round6(value: Any) -> float:
    """Round to 6 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

def _std_normal_pdf(x: float) -> float:
    """Standard normal probability density function phi(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _std_normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContractType(str, Enum):
    """Energy supply contract structure.

    FIXED:              Flat price for entire volume over term.
    INDEX:              100% indexed to a market reference price.
    BLOCK_AND_INDEX:    Portion at fixed block price, remainder indexed.
    HEAT_RATE:          Price tied to fuel input heat-rate formula.
    TOLLING:            Buyer provides fuel, pays conversion fee.
    SHAPED:             Price shaped to buyer's load profile.
    FULL_REQUIREMENTS:  Supplier covers full load with all-in price.
    SLEEVE:             Third party wraps PPA for credit/balancing.
    """
    FIXED = "fixed"
    INDEX = "index"
    BLOCK_AND_INDEX = "block_and_index"
    HEAT_RATE = "heat_rate"
    TOLLING = "tolling"
    SHAPED = "shaped"
    FULL_REQUIREMENTS = "full_requirements"
    SLEEVE = "sleeve"

class MarketIndex(str, Enum):
    """Market price indices for energy commodities.

    DAY_AHEAD:          Day-ahead spot market price.
    REAL_TIME:          Real-time / balancing market price.
    MONTHLY_FORWARD:    Monthly forward / futures contract.
    QUARTERLY_FORWARD:  Quarterly forward / futures contract.
    ANNUAL_FORWARD:     Calendar-year forward / futures.
    ICE_TTF:            ICE TTF natural gas (EUR/MWh).
    HENRY_HUB:          Henry Hub natural gas (USD/MMBtu).
    NBP:                National Balancing Point gas (GBP/therm).
    AECO:               AECO-C Alberta gas hub (CAD/GJ).
    JKM:                Japan Korea Marker LNG (USD/MMBtu).
    """
    DAY_AHEAD = "day_ahead"
    REAL_TIME = "real_time"
    MONTHLY_FORWARD = "monthly_forward"
    QUARTERLY_FORWARD = "quarterly_forward"
    ANNUAL_FORWARD = "annual_forward"
    ICE_TTF = "ice_ttf"
    HENRY_HUB = "henry_hub"
    NBP = "nbp"
    AECO = "aeco"
    JKM = "jkm"

class ProcurementStrategy(str, Enum):
    """Procurement hedging strategies.

    FULL_FIXED:         Lock in 100% of volume at a fixed price.
    FULL_INDEX:         Leave 100% of volume floating on index.
    LAYERED_HEDGE:      Fix volume in tranches over time.
    PORTFOLIO:          Mix of fixed, index, and financial hedges.
    BLOCK_AND_INDEX:    Base-load block fixed, peaks indexed.
    PROGRESSIVE_FIXED:  Progressively increase fixed percentage.
    """
    FULL_FIXED = "full_fixed"
    FULL_INDEX = "full_index"
    LAYERED_HEDGE = "layered_hedge"
    PORTFOLIO = "portfolio"
    BLOCK_AND_INDEX = "block_and_index"
    PROGRESSIVE_FIXED = "progressive_fixed"

class RiskMetric(str, Enum):
    """Quantitative risk measures for procurement cost exposure.

    VAR_95:         Value at Risk at 95% confidence.
    VAR_99:         Value at Risk at 99% confidence.
    CVAR_95:        Conditional VaR (Expected Shortfall) at 95%.
    CVAR_99:        Conditional VaR (Expected Shortfall) at 99%.
    STD_DEV:        Standard deviation of cost distribution.
    MAX_DRAWDOWN:   Maximum cost increase from trough to peak.
    SHARPE_RATIO:   Risk-adjusted cost savings ratio.
    """
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    CVAR_95 = "cvar_95"
    CVAR_99 = "cvar_99"
    STD_DEV = "std_dev"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"

class GreenProduct(str, Enum):
    """Green energy procurement products.

    PPA_PHYSICAL:       Physical Power Purchase Agreement.
    PPA_VIRTUAL:        Virtual / Financial PPA (contract for differences).
    REC:                Renewable Energy Certificate (US / global).
    GO:                 Guarantee of Origin (EU).
    GREEN_TARIFF:       Utility green tariff programme.
    BUNDLED_RENEWABLE:  Bundled renewable supply contract.
    """
    PPA_PHYSICAL = "ppa_physical"
    PPA_VIRTUAL = "ppa_virtual"
    REC = "rec"
    GO = "go"
    GREEN_TARIFF = "green_tariff"
    BUNDLED_RENEWABLE = "bundled_renewable"

class SupplierRating(str, Enum):
    """Supplier credit / quality rating scale.

    AAA through B represent investment-grade to speculative-grade.
    UNRATED indicates the supplier has not been evaluated.
    """
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    UNRATED = "unrated"

class MarketCondition(str, Enum):
    """Market forward curve shape classification.

    CONTANGO:       Forward prices above spot (premium for future delivery).
    BACKWARDATION:  Forward prices below spot (discount for future).
    FLAT:           Forward curve roughly flat within threshold.
    VOLATILE:       High price dispersion / unstable curve shape.
    """
    CONTANGO = "contango"
    BACKWARDATION = "backwardation"
    FLAT = "flat"
    VOLATILE = "volatile"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Parametric VaR z-scores.
_Z_95: float = 1.6449
_Z_99: float = 2.3263

# Credit rating numeric scores (higher = better).
_CREDIT_SCORE_MAP: Dict[str, float] = {
    SupplierRating.AAA.value: 100.0,
    SupplierRating.AA.value: 90.0,
    SupplierRating.A.value: 80.0,
    SupplierRating.BBB.value: 65.0,
    SupplierRating.BB.value: 50.0,
    SupplierRating.B.value: 35.0,
    SupplierRating.UNRATED.value: 20.0,
}

# Flat curve tolerance percentage.
_FLAT_CURVE_TOLERANCE_PCT: float = 2.0

# Volatility threshold (coefficient of variation) for VOLATILE classification.
_VOLATILITY_CV_THRESHOLD: float = 0.15

# Default emission factor for grid electricity (tCO2/MWh).
# Source: EU average 2023 (EEA).
_DEFAULT_GRID_EMISSION_FACTOR: Decimal = Decimal("0.230")

# Strategy budget certainty ranges (percentage of cost that is fixed/known).
_STRATEGY_CERTAINTY: Dict[str, float] = {
    ProcurementStrategy.FULL_FIXED.value: 100.0,
    ProcurementStrategy.FULL_INDEX.value: 0.0,
    ProcurementStrategy.LAYERED_HEDGE.value: 70.0,
    ProcurementStrategy.PORTFOLIO.value: 60.0,
    ProcurementStrategy.BLOCK_AND_INDEX.value: 50.0,
    ProcurementStrategy.PROGRESSIVE_FIXED.value: 55.0,
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class MarketPrice(BaseModel):
    """A single market price observation.

    Attributes:
        index: Market index reference.
        date: Observation date.
        price_per_mwh: Price in currency per MWh.
        currency: ISO currency code.
        source: Data source identifier.
    """
    index: MarketIndex = Field(default=MarketIndex.DAY_AHEAD, description="Market index")
    date: date = Field(description="Observation date")
    price_per_mwh: float = Field(ge=0.0, description="Price per MWh")
    currency: str = Field(default="EUR", max_length=3, description="ISO currency code")
    source: str = Field(default="", description="Data source identifier")

class ForwardCurve(BaseModel):
    """Forward price curve for a market index.

    Attributes:
        index: Market index reference.
        prices: Monthly forward prices ordered chronologically.
        as_of_date: Valuation date of the curve.
        source: Data source identifier.
    """
    index: MarketIndex = Field(default=MarketIndex.MONTHLY_FORWARD, description="Index")
    prices: List[float] = Field(default_factory=list, description="Monthly prices EUR/MWh")
    as_of_date: date = Field(default_factory=lambda: date.today(), description="Curve date")
    source: str = Field(default="", description="Data source")

class ContractTerms(BaseModel):
    """Terms of an energy supply contract.

    Attributes:
        contract_id: Unique contract identifier.
        supplier: Supplier company name.
        contract_type: Contract pricing structure.
        start_date: Contract start date.
        end_date: Contract end date.
        volume_mwh: Contracted annual volume (MWh).
        fixed_price_per_mwh: Fixed price component (EUR/MWh).
        index_reference: Index for floating price component.
        adder_per_mwh: Adder / margin on top of index (EUR/MWh).
        swing_pct: Allowed volume flexibility percentage.
        minimum_volume: Minimum take-or-pay volume (MWh).
        maximum_volume: Maximum deliverable volume (MWh).
        green_percentage: Percentage of volume from renewable sources.
    """
    contract_id: str = Field(default_factory=_new_uuid, description="Contract ID")
    supplier: str = Field(default="", description="Supplier name")
    contract_type: ContractType = Field(default=ContractType.FIXED, description="Contract type")
    start_date: date = Field(default_factory=lambda: date.today(), description="Start date")
    end_date: date = Field(default_factory=lambda: date.today(), description="End date")
    volume_mwh: float = Field(default=0.0, ge=0.0, description="Annual volume MWh")
    fixed_price_per_mwh: Optional[float] = Field(
        default=None, ge=0.0, description="Fixed price EUR/MWh"
    )
    index_reference: Optional[MarketIndex] = Field(default=None, description="Index reference")
    adder_per_mwh: float = Field(default=0.0, description="Adder EUR/MWh")
    swing_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Swing %")
    minimum_volume: Optional[float] = Field(default=None, ge=0.0, description="Min volume MWh")
    maximum_volume: Optional[float] = Field(default=None, ge=0.0, description="Max volume MWh")
    green_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Green %")

class LoadInterval(BaseModel):
    """A single load interval (e.g. hourly, half-hourly, 15-min).

    Attributes:
        timestamp: Interval start timestamp.
        kwh: Energy consumed in the interval (kWh).
        price_per_kwh: Market price for the interval (EUR/kWh).
    """
    timestamp: datetime = Field(description="Interval start")
    kwh: float = Field(default=0.0, ge=0.0, description="Interval consumption kWh")
    price_per_kwh: Optional[float] = Field(default=None, ge=0.0, description="Price EUR/kWh")

class GreenProductOffering(BaseModel):
    """A green energy product available for procurement.

    Attributes:
        product_type: Type of green product.
        price_premium_per_mwh: Premium above conventional price (EUR/MWh).
        available_volume_mwh: Maximum volume available (MWh).
        additionality_score: Additionality rating 0-100 (RE100 criteria).
        certificate_vintage: Certificate vintage year.
        technology: Generation technology (wind, solar, hydro, etc.).
        location: Generation facility location / country.
        emission_factor_tco2_per_mwh: Residual emission factor (tCO2/MWh).
    """
    product_type: GreenProduct = Field(default=GreenProduct.GO, description="Product type")
    price_premium_per_mwh: float = Field(default=0.0, ge=0.0, description="Premium EUR/MWh")
    available_volume_mwh: float = Field(default=0.0, ge=0.0, description="Volume MWh")
    additionality_score: float = Field(
        default=50.0, ge=0.0, le=100.0, description="Additionality 0-100"
    )
    certificate_vintage: int = Field(default=2025, ge=2010, le=2040, description="Vintage")
    technology: str = Field(default="", description="Generation technology")
    location: str = Field(default="", description="Generation location")
    emission_factor_tco2_per_mwh: float = Field(
        default=0.0, ge=0.0, description="Residual emissions tCO2/MWh"
    )

class SupplierProfile(BaseModel):
    """Profile of an energy supplier for evaluation.

    Attributes:
        supplier_name: Supplier company name.
        credit_rating: Supplier credit rating.
        price_competitiveness_score: Price competitiveness 0-100.
        service_quality_score: Service quality rating 0-100.
        green_offering_score: Green product offering breadth 0-100.
        market_share_pct: Market share percentage.
        years_in_market: Years of operational history.
        renewable_percentage: Percentage of portfolio from renewables.
    """
    supplier_name: str = Field(default="", description="Supplier name")
    credit_rating: SupplierRating = Field(
        default=SupplierRating.UNRATED, description="Credit rating"
    )
    price_competitiveness_score: float = Field(
        default=50.0, ge=0.0, le=100.0, description="Price score"
    )
    service_quality_score: float = Field(
        default=50.0, ge=0.0, le=100.0, description="Service score"
    )
    green_offering_score: float = Field(
        default=50.0, ge=0.0, le=100.0, description="Green score"
    )
    market_share_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Market share %")
    years_in_market: int = Field(default=0, ge=0, description="Years in market")
    renewable_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Renewable %"
    )

class HedgeLayer(BaseModel):
    """A single layer / tranche in a layered hedging strategy.

    Attributes:
        layer_id: Layer identifier.
        volume_mwh: Volume hedged in this layer (MWh).
        price_per_mwh: Executed or target price (EUR/MWh).
        execution_date: Date layer was or will be executed.
        hedge_type: FIXED or INDEX for this layer.
        percentage_of_total: Layer as percentage of total volume.
    """
    layer_id: str = Field(default_factory=_new_uuid, description="Layer ID")
    volume_mwh: float = Field(default=0.0, ge=0.0, description="Layer volume MWh")
    price_per_mwh: float = Field(default=0.0, ge=0.0, description="Layer price EUR/MWh")
    execution_date: Optional[date] = Field(default=None, description="Execution date")
    hedge_type: str = Field(default="fixed", description="fixed or index")
    percentage_of_total: float = Field(
        default=0.0, ge=0.0, le=100.0, description="% of total volume"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ContractComparison(BaseModel):
    """Result of comparing multiple contract offers.

    Attributes:
        contracts: List of evaluated contract IDs.
        annual_cost_by_contract: Projected annual cost per contract (EUR).
        cheapest_contract_id: ID of lowest-cost contract.
        savings_vs_default: Savings of cheapest vs. most expensive (EUR).
        load_weighted_prices: LWP per contract (EUR/MWh).
        risk_metrics: Risk metric values per contract.
        methodology_notes: Calculation methodology notes.
        processing_time_ms: Processing duration (ms).
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    contracts: List[str] = Field(default_factory=list)
    annual_cost_by_contract: Dict[str, float] = Field(default_factory=dict)
    cheapest_contract_id: str = Field(default="")
    savings_vs_default: float = Field(default=0.0)
    load_weighted_prices: Dict[str, float] = Field(default_factory=dict)
    risk_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class LoadWeightedPrice(BaseModel):
    """Load-weighted price calculation result.

    Attributes:
        period: Description of the analysis period.
        total_kwh: Total energy in the period (kWh).
        total_cost: Total cost in the period (EUR).
        lwp_per_kwh: Load-weighted price (EUR/kWh).
        lwp_per_mwh: Load-weighted price (EUR/MWh).
        baseload_price_per_mwh: Flat baseload average price (EUR/MWh).
        profile_premium_pct: Profile premium vs. baseload (%).
        interval_count: Number of intervals analysed.
        peak_price_per_kwh: Maximum interval price (EUR/kWh).
        off_peak_price_per_kwh: Minimum interval price (EUR/kWh).
        processing_time_ms: Processing duration (ms).
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    period: str = Field(default="")
    total_kwh: float = Field(default=0.0)
    total_cost: float = Field(default=0.0)
    lwp_per_kwh: float = Field(default=0.0)
    lwp_per_mwh: float = Field(default=0.0)
    baseload_price_per_mwh: float = Field(default=0.0)
    profile_premium_pct: float = Field(default=0.0)
    interval_count: int = Field(default=0, ge=0)
    peak_price_per_kwh: float = Field(default=0.0)
    off_peak_price_per_kwh: float = Field(default=0.0)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class PriceRiskAssessment(BaseModel):
    """Price risk assessment result.

    Attributes:
        var_95_eur: Value at Risk at 95% confidence (EUR).
        var_99_eur: Value at Risk at 99% confidence (EUR).
        cvar_95_eur: Conditional VaR at 95% (EUR).
        cvar_99_eur: Conditional VaR at 99% (EUR).
        expected_cost_eur: Expected (mean) annual cost (EUR).
        best_case_eur: Best-case cost (5th percentile).
        worst_case_eur: Worst-case cost (99th percentile).
        std_dev_eur: Standard deviation of cost (EUR).
        coefficient_of_variation: CV of cost distribution.
        strategy: Procurement strategy evaluated.
        methodology_notes: Calculation methodology notes.
        processing_time_ms: Processing duration (ms).
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    var_95_eur: float = Field(default=0.0)
    var_99_eur: float = Field(default=0.0)
    cvar_95_eur: float = Field(default=0.0)
    cvar_99_eur: float = Field(default=0.0)
    expected_cost_eur: float = Field(default=0.0)
    best_case_eur: float = Field(default=0.0)
    worst_case_eur: float = Field(default=0.0)
    std_dev_eur: float = Field(default=0.0)
    coefficient_of_variation: float = Field(default=0.0)
    strategy: str = Field(default="")
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class ProcurementPlan(BaseModel):
    """Recommended procurement plan.

    Attributes:
        strategy: Recommended procurement strategy.
        layers: List of hedge layers / tranches.
        total_volume_mwh: Total volume covered (MWh).
        weighted_avg_price: Weighted average price across layers (EUR/MWh).
        budget_certainty_pct: Percentage of budget that is price-certain.
        fixed_volume_pct: Percentage of volume at fixed price.
        index_volume_pct: Percentage of volume at floating index.
        estimated_annual_cost_eur: Estimated total annual cost (EUR).
        execution_timeline: Recommended execution schedule.
        methodology_notes: Calculation methodology notes.
        processing_time_ms: Processing duration (ms).
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    strategy: str = Field(default="")
    layers: List[HedgeLayer] = Field(default_factory=list)
    total_volume_mwh: float = Field(default=0.0)
    weighted_avg_price: float = Field(default=0.0)
    budget_certainty_pct: float = Field(default=0.0)
    fixed_volume_pct: float = Field(default=0.0)
    index_volume_pct: float = Field(default=0.0)
    estimated_annual_cost_eur: float = Field(default=0.0)
    execution_timeline: List[str] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class GreenProcurement(BaseModel):
    """Green procurement option evaluation result.

    Attributes:
        product_type: Type of green product.
        volume_mwh: Volume to be procured (MWh).
        price_premium_per_mwh: Premium vs. conventional (EUR/MWh).
        total_premium_eur: Total annual premium cost (EUR).
        additionality_score: Additionality rating 0-100.
        certificates_needed: Number of certificates required.
        annual_co2_avoided_tonnes: Annual CO2 emissions avoided (tCO2).
        cost_per_tonne_avoided: Cost per tonne CO2 avoided (EUR/tCO2).
        re100_eligible: Whether product meets RE100 criteria.
    """
    product_type: str = Field(default="")
    volume_mwh: float = Field(default=0.0)
    price_premium_per_mwh: float = Field(default=0.0)
    total_premium_eur: float = Field(default=0.0)
    additionality_score: float = Field(default=0.0)
    certificates_needed: int = Field(default=0, ge=0)
    annual_co2_avoided_tonnes: float = Field(default=0.0)
    cost_per_tonne_avoided: float = Field(default=0.0)
    re100_eligible: bool = Field(default=False)

class SupplierEvaluation(BaseModel):
    """Supplier evaluation result.

    Attributes:
        supplier_name: Supplier company name.
        credit_rating: Supplier credit rating.
        credit_score: Numeric credit score 0-100.
        price_competitiveness_score: Price score 0-100.
        service_quality_score: Service quality score 0-100.
        green_offering_score: Green offering score 0-100.
        overall_score: Weighted overall score 0-100.
        rank: Rank among evaluated suppliers (1 = best).
        recommendation: Textual recommendation.
    """
    supplier_name: str = Field(default="")
    credit_rating: str = Field(default="")
    credit_score: float = Field(default=0.0)
    price_competitiveness_score: float = Field(default=0.0)
    service_quality_score: float = Field(default=0.0)
    green_offering_score: float = Field(default=0.0)
    overall_score: float = Field(default=0.0)
    rank: int = Field(default=0, ge=0)
    recommendation: str = Field(default="")

class MonthlyProjection(BaseModel):
    """Monthly cost projection from a procurement plan.

    Attributes:
        month: Month number (1-12) or sequential month index.
        month_label: Human-readable month label.
        volume_mwh: Projected volume for the month (MWh).
        fixed_cost_eur: Cost from fixed-price layers (EUR).
        index_cost_eur: Cost from index-priced layers (EUR).
        total_cost_eur: Total projected cost (EUR).
        effective_price_per_mwh: Blended effective price (EUR/MWh).
    """
    month: int = Field(default=1, ge=1)
    month_label: str = Field(default="")
    volume_mwh: float = Field(default=0.0)
    fixed_cost_eur: float = Field(default=0.0)
    index_cost_eur: float = Field(default=0.0)
    total_cost_eur: float = Field(default=0.0)
    effective_price_per_mwh: float = Field(default=0.0)

class ProcurementResult(BaseModel):
    """Complete procurement intelligence result.

    Attributes:
        result_id: Unique result identifier.
        current_contract_cost: Current annual contract cost (EUR).
        optimal_strategy: Recommended procurement strategy name.
        projected_savings: Projected annual savings vs. current (EUR).
        savings_pct: Projected savings as percentage of current cost.
        risk_assessment: Price risk assessment result.
        green_options: Evaluated green procurement options.
        supplier_rankings: Ranked supplier evaluations.
        execution_plan: Recommended procurement plan.
        market_condition: Current market condition assessment.
        methodology_notes: Methodology and source notes.
        processing_time_ms: Processing duration (ms).
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    current_contract_cost: float = Field(default=0.0)
    optimal_strategy: str = Field(default="")
    projected_savings: float = Field(default=0.0)
    savings_pct: float = Field(default=0.0)
    risk_assessment: Optional[PriceRiskAssessment] = Field(default=None)
    green_options: List[GreenProcurement] = Field(default_factory=list)
    supplier_rankings: List[SupplierEvaluation] = Field(default_factory=list)
    execution_plan: Optional[ProcurementPlan] = Field(default=None)
    market_condition: str = Field(default="")
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ProcurementIntelligenceEngine:
    """Zero-hallucination energy procurement intelligence engine.

    Provides deterministic analytics for energy procurement decisions
    including contract comparison, load-weighted pricing, price-risk
    quantification, procurement planning, green product evaluation,
    supplier scoring, market analysis, hedge valuation, and forward-
    curve cost projection.

    Guarantees:
        - Deterministic: same inputs produce identical outputs.
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full methodology notes and intermediate values.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = ProcurementIntelligenceEngine()
        comparison = engine.compare_contracts(load_profile, contracts, market_prices)
        risk = engine.assess_price_risk(strategy, market_data, volume_mwh)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the procurement intelligence engine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - currency (str): default currency code (default EUR)
                - grid_emission_factor (float): tCO2/MWh for grid
                - risk_confidence (float): default VaR confidence level
                - default_strategy (str): default procurement strategy
        """
        self._config = config or {}
        self._currency = self._config.get("currency", "EUR")
        self._grid_ef = _decimal(
            self._config.get("grid_emission_factor", _DEFAULT_GRID_EMISSION_FACTOR)
        )
        self._risk_confidence = float(self._config.get("risk_confidence", 0.95))
        self._default_strategy = ProcurementStrategy(
            self._config.get("default_strategy", ProcurementStrategy.LAYERED_HEDGE.value)
        )
        self._notes: List[str] = []
        logger.info("ProcurementIntelligenceEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API -- Contract Comparison
    # --------------------------------------------------------------------- #

    def compare_contracts(
        self,
        load_profile: List[LoadInterval],
        contracts: List[ContractTerms],
        market_prices: List[MarketPrice],
    ) -> ContractComparison:
        """Compare multiple contract offers against a load profile.

        Calculates the annual cost of each contract given the buyer's
        actual load shape and prevailing market prices for any indexed
        components.

        Args:
            load_profile: Interval-level load data with timestamps and kWh.
            contracts: List of contract offers to compare.
            market_prices: Historical or forecast market prices for indices.

        Returns:
            ContractComparison with cost per contract, cheapest option,
            savings, load-weighted prices, and risk metrics.

        Raises:
            ValueError: If load_profile is empty or no contracts provided.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]

        if not load_profile:
            raise ValueError("Load profile must contain at least one interval.")
        if not contracts:
            raise ValueError("At least one contract must be provided.")

        total_load_kwh = sum(_decimal(iv.kwh) for iv in load_profile)
        total_load_mwh = _safe_divide(total_load_kwh, Decimal("1000"))
        price_lookup = self._build_price_lookup(market_prices)

        annual_costs: Dict[str, float] = {}
        lwp_map: Dict[str, float] = {}
        risk_map: Dict[str, Dict[str, float]] = {}

        for ct in contracts:
            cost = self._calculate_contract_cost(ct, load_profile, price_lookup, total_load_mwh)
            annual_costs[ct.contract_id] = _round2(float(cost))

            # Load-weighted price for this contract (EUR/MWh).
            lwp = _safe_divide(cost, total_load_mwh)
            lwp_map[ct.contract_id] = _round4(float(lwp))

            # Simplified risk: std dev of index prices as proxy.
            risk_vals = self._contract_risk_metrics(ct, price_lookup, total_load_mwh)
            risk_map[ct.contract_id] = risk_vals

        # Identify cheapest.
        cheapest_id = min(annual_costs, key=annual_costs.get)  # type: ignore[arg-type]
        most_expensive_cost = max(annual_costs.values())
        cheapest_cost = annual_costs[cheapest_id]
        savings = most_expensive_cost - cheapest_cost

        self._notes.append(
            f"Compared {len(contracts)} contracts against "
            f"{len(load_profile)} load intervals ({_round2(float(total_load_mwh))} MWh)."
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ContractComparison(
            contracts=[c.contract_id for c in contracts],
            annual_cost_by_contract=annual_costs,
            cheapest_contract_id=cheapest_id,
            savings_vs_default=_round2(savings),
            load_weighted_prices=lwp_map,
            risk_metrics=risk_map,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Contract comparison complete: cheapest=%s, cost=%.2f %s, hash=%s (%.1f ms)",
            cheapest_id[:12],
            cheapest_cost,
            self._currency,
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # --------------------------------------------------------------------- #
    # Public API -- Load-Weighted Price
    # --------------------------------------------------------------------- #

    def calculate_load_weighted_price(
        self,
        intervals: List[LoadInterval],
        market_prices: Optional[List[MarketPrice]] = None,
    ) -> LoadWeightedPrice:
        """Calculate load-weighted price from interval data.

        LWP = Sum(Interval_kWh * Interval_Price) / Sum(Interval_kWh)

        If intervals already have price_per_kwh, those are used directly.
        Otherwise, market_prices are matched by date.

        Args:
            intervals: Interval-level load data.
            market_prices: Optional market prices to fill missing interval prices.

        Returns:
            LoadWeightedPrice with LWP, profile premium, and statistics.

        Raises:
            ValueError: If no intervals provided or no prices available.
        """
        t0 = time.perf_counter()

        if not intervals:
            raise ValueError("At least one load interval is required.")

        price_lookup = self._build_price_lookup(market_prices or [])

        total_kwh = Decimal("0")
        total_cost = Decimal("0")
        all_prices: List[Decimal] = []
        priced_count = 0

        for iv in intervals:
            kwh = _decimal(iv.kwh)
            price = self._resolve_interval_price(iv, price_lookup)
            if price is not None and price > Decimal("0"):
                total_kwh += kwh
                total_cost += kwh * price
                all_prices.append(price)
                priced_count += 1

        if total_kwh == Decimal("0"):
            raise ValueError("Total load is zero; cannot compute LWP.")

        lwp_per_kwh = _safe_divide(total_cost, total_kwh)
        lwp_per_mwh = lwp_per_kwh * Decimal("1000")

        # Baseload price = simple average of all interval prices.
        baseload_per_kwh = _safe_divide(
            sum(all_prices), _decimal(len(all_prices))
        ) if all_prices else Decimal("0")
        baseload_per_mwh = baseload_per_kwh * Decimal("1000")

        profile_premium = _safe_pct(
            lwp_per_mwh - baseload_per_mwh, baseload_per_mwh
        )

        peak_price = max(all_prices) if all_prices else Decimal("0")
        off_peak_price = min(all_prices) if all_prices else Decimal("0")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = LoadWeightedPrice(
            period=f"{intervals[0].timestamp.date()} to {intervals[-1].timestamp.date()}",
            total_kwh=_round2(float(total_kwh)),
            total_cost=_round2(float(total_cost)),
            lwp_per_kwh=_round6(float(lwp_per_kwh)),
            lwp_per_mwh=_round4(float(lwp_per_mwh)),
            baseload_price_per_mwh=_round4(float(baseload_per_mwh)),
            profile_premium_pct=_round2(float(profile_premium)),
            interval_count=priced_count,
            peak_price_per_kwh=_round6(float(peak_price)),
            off_peak_price_per_kwh=_round6(float(off_peak_price)),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "LWP calculated: %.4f %s/MWh, profile premium %.2f%%, hash=%s (%.1f ms)",
            float(lwp_per_mwh),
            self._currency,
            float(profile_premium),
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # --------------------------------------------------------------------- #
    # Public API -- Price Risk Assessment
    # --------------------------------------------------------------------- #

    def assess_price_risk(
        self,
        strategy: ProcurementStrategy,
        market_data: List[MarketPrice],
        volume_mwh: float,
    ) -> PriceRiskAssessment:
        """Assess price risk of a procurement strategy.

        Uses parametric VaR/CVaR based on historical price distribution.
        VaR_95 = mu + 1.645 * sigma (cost is a loss, so upper tail).
        CVaR_95 = mu + sigma * phi(z_alpha) / (1 - alpha).

        Args:
            strategy: Procurement strategy to evaluate.
            market_data: Historical market prices for distribution fitting.
            volume_mwh: Annual volume in MWh to apply risk to.

        Returns:
            PriceRiskAssessment with VaR, CVaR, expected cost, and ranges.

        Raises:
            ValueError: If market_data has fewer than 2 observations.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]

        if len(market_data) < 2:
            raise ValueError("At least 2 market price observations required for risk assessment.")

        d_volume = _decimal(volume_mwh)
        prices = [_decimal(mp.price_per_mwh) for mp in market_data]

        mu, sigma = self._compute_price_stats(prices)
        budget_certainty = _decimal(
            _STRATEGY_CERTAINTY.get(strategy.value, 50.0)
        ) / Decimal("100")

        # Adjust sigma for strategy: fixed portions have zero volatility.
        adjusted_sigma = sigma * (Decimal("1") - budget_certainty)

        # Expected cost.
        expected_cost = mu * d_volume

        # VaR: upper tail of cost distribution (parametric normal).
        var_95 = (mu + _decimal(_Z_95) * adjusted_sigma) * d_volume
        var_99 = (mu + _decimal(_Z_99) * adjusted_sigma) * d_volume

        # CVaR: expected shortfall beyond VaR.
        cvar_95 = self._compute_cvar(mu, adjusted_sigma, d_volume, 0.95)
        cvar_99 = self._compute_cvar(mu, adjusted_sigma, d_volume, 0.99)

        # Best/worst case.
        best_case = (mu - _decimal(_Z_95) * adjusted_sigma) * d_volume
        worst_case = var_99  # 99th percentile cost.

        std_dev_cost = adjusted_sigma * d_volume
        cv = _safe_divide(adjusted_sigma, mu)

        self._notes.append(
            f"Strategy {strategy.value}: budget certainty {float(budget_certainty * 100):.0f}%, "
            f"adjusted sigma {_round4(float(adjusted_sigma))} {self._currency}/MWh."
        )
        self._notes.append(
            f"Parametric VaR: z_95={_Z_95}, z_99={_Z_99}."
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = PriceRiskAssessment(
            var_95_eur=_round2(float(var_95)),
            var_99_eur=_round2(float(var_99)),
            cvar_95_eur=_round2(float(cvar_95)),
            cvar_99_eur=_round2(float(cvar_99)),
            expected_cost_eur=_round2(float(expected_cost)),
            best_case_eur=_round2(float(best_case)),
            worst_case_eur=_round2(float(worst_case)),
            std_dev_eur=_round2(float(std_dev_cost)),
            coefficient_of_variation=_round4(float(cv)),
            strategy=strategy.value,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Risk assessment for %s: VaR95=%.2f, CVaR95=%.2f, hash=%s (%.1f ms)",
            strategy.value,
            float(var_95),
            float(cvar_95),
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # --------------------------------------------------------------------- #
    # Public API -- Procurement Plan
    # --------------------------------------------------------------------- #

    def develop_procurement_plan(
        self,
        total_volume_mwh: float,
        market_prices: List[MarketPrice],
        risk_tolerance: float = 0.5,
        strategy: Optional[ProcurementStrategy] = None,
    ) -> ProcurementPlan:
        """Develop a procurement plan with hedge layers.

        Decomposes total volume into fixed and index layers based on
        the chosen strategy and the buyer's risk tolerance (0 = risk
        averse / all fixed, 1 = risk seeking / all index).

        Args:
            total_volume_mwh: Total annual volume to procure (MWh).
            market_prices: Market prices for forward pricing.
            risk_tolerance: Risk tolerance 0.0 (averse) to 1.0 (seeking).
            strategy: Strategy override; otherwise chosen from risk tolerance.

        Returns:
            ProcurementPlan with layers, pricing, and execution timeline.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]

        d_volume = _decimal(total_volume_mwh)
        risk_tolerance = max(0.0, min(1.0, risk_tolerance))

        chosen_strategy = strategy or self._select_strategy(risk_tolerance)
        layers = self._build_layers(chosen_strategy, d_volume, market_prices, risk_tolerance)

        # Compute weighted average price across layers.
        wap = self._weighted_avg_price(layers, d_volume)

        # Fixed vs. index volume percentages.
        fixed_vol = sum(
            _decimal(ly.volume_mwh) for ly in layers if ly.hedge_type == "fixed"
        )
        index_vol = d_volume - fixed_vol
        fixed_pct = _safe_pct(fixed_vol, d_volume)
        index_pct = _safe_pct(index_vol, d_volume)

        budget_certainty = _decimal(
            _STRATEGY_CERTAINTY.get(chosen_strategy.value, 50.0)
        )

        estimated_cost = wap * d_volume

        # Execution timeline.
        timeline = self._build_execution_timeline(chosen_strategy, layers)

        self._notes.append(
            f"Strategy {chosen_strategy.value}: {len(layers)} layers, "
            f"WAP {_round4(float(wap))} {self._currency}/MWh."
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ProcurementPlan(
            strategy=chosen_strategy.value,
            layers=layers,
            total_volume_mwh=_round2(float(d_volume)),
            weighted_avg_price=_round4(float(wap)),
            budget_certainty_pct=_round2(float(budget_certainty)),
            fixed_volume_pct=_round2(float(fixed_pct)),
            index_volume_pct=_round2(float(index_pct)),
            estimated_annual_cost_eur=_round2(float(estimated_cost)),
            execution_timeline=timeline,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Procurement plan: strategy=%s, WAP=%.4f, cost=%.2f, hash=%s (%.1f ms)",
            chosen_strategy.value,
            float(wap),
            float(estimated_cost),
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # --------------------------------------------------------------------- #
    # Public API -- Green Options Evaluation
    # --------------------------------------------------------------------- #

    def evaluate_green_options(
        self,
        total_volume_mwh: float,
        green_products: List[GreenProductOffering],
    ) -> List[GreenProcurement]:
        """Evaluate green energy procurement options.

        For each green product offering, calculates the total premium
        cost, CO2 avoided, cost per tonne avoided, and RE100 eligibility.

        Args:
            total_volume_mwh: Total annual load to cover (MWh).
            green_products: Available green product offerings.

        Returns:
            List of GreenProcurement evaluations sorted by cost per tonne.
        """
        self._notes = [f"Engine version: {self.engine_version}"]
        results: List[GreenProcurement] = []

        d_volume = _decimal(total_volume_mwh)

        for gp in green_products:
            volume = min(d_volume, _decimal(gp.available_volume_mwh))
            if volume <= Decimal("0"):
                continue

            premium = _decimal(gp.price_premium_per_mwh)
            total_premium = premium * volume

            # CO2 avoided = volume * (grid_EF - product_EF).
            product_ef = _decimal(gp.emission_factor_tco2_per_mwh)
            avoided = volume * (self._grid_ef - product_ef)
            avoided = max(avoided, Decimal("0"))

            cost_per_tonne = _safe_divide(total_premium, avoided)

            # Certificates needed: 1 certificate per MWh is the EU GO standard.
            certificates = int(volume.to_integral_value(rounding=ROUND_HALF_UP))

            # RE100 eligibility: physical PPA, bundled, or GO with additionality >= 70.
            re100 = self._check_re100_eligibility(gp)

            results.append(GreenProcurement(
                product_type=gp.product_type.value,
                volume_mwh=_round2(float(volume)),
                price_premium_per_mwh=_round4(float(premium)),
                total_premium_eur=_round2(float(total_premium)),
                additionality_score=_round2(gp.additionality_score),
                certificates_needed=certificates,
                annual_co2_avoided_tonnes=_round2(float(avoided)),
                cost_per_tonne_avoided=_round2(float(cost_per_tonne)),
                re100_eligible=re100,
            ))

        # Sort by cost per tonne avoided ascending (cheapest abatement first).
        results.sort(key=lambda r: r.cost_per_tonne_avoided)

        self._notes.append(
            f"Evaluated {len(results)} green products for {_round2(float(d_volume))} MWh."
        )
        logger.info("Green options evaluated: %d products.", len(results))
        return results

    # --------------------------------------------------------------------- #
    # Public API -- Supplier Evaluation
    # --------------------------------------------------------------------- #

    def evaluate_suppliers(
        self,
        suppliers: List[SupplierProfile],
        criteria_weights: Optional[Dict[str, float]] = None,
    ) -> List[SupplierEvaluation]:
        """Evaluate and rank energy suppliers.

        Overall score = Sum(criteria_weight_i * normalised_score_i).
        Default weights: credit 25%, price 30%, service 20%, green 25%.

        Args:
            suppliers: Supplier profiles to evaluate.
            criteria_weights: Custom weights {credit, price, service, green}.

        Returns:
            List of SupplierEvaluation results sorted by overall score desc.
        """
        self._notes = [f"Engine version: {self.engine_version}"]

        weights = criteria_weights or {
            "credit": 0.25,
            "price": 0.30,
            "service": 0.20,
            "green": 0.25,
        }

        # Normalise weights to sum to 1.0.
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        results: List[SupplierEvaluation] = []

        for sp in suppliers:
            credit_score = _CREDIT_SCORE_MAP.get(sp.credit_rating.value, 20.0)

            overall = (
                weights.get("credit", 0.25) * credit_score
                + weights.get("price", 0.30) * sp.price_competitiveness_score
                + weights.get("service", 0.20) * sp.service_quality_score
                + weights.get("green", 0.25) * sp.green_offering_score
            )

            recommendation = self._generate_supplier_recommendation(
                sp.supplier_name, overall, credit_score
            )

            results.append(SupplierEvaluation(
                supplier_name=sp.supplier_name,
                credit_rating=sp.credit_rating.value,
                credit_score=_round2(credit_score),
                price_competitiveness_score=_round2(sp.price_competitiveness_score),
                service_quality_score=_round2(sp.service_quality_score),
                green_offering_score=_round2(sp.green_offering_score),
                overall_score=_round2(overall),
                recommendation=recommendation,
            ))

        # Sort by overall score descending.
        results.sort(key=lambda r: r.overall_score, reverse=True)

        # Assign ranks.
        for idx, ev in enumerate(results):
            ev.rank = idx + 1

        self._notes.append(f"Evaluated {len(results)} suppliers.")
        logger.info("Supplier evaluation: %d suppliers ranked.", len(results))
        return results

    # --------------------------------------------------------------------- #
    # Public API -- Market Condition Analysis
    # --------------------------------------------------------------------- #

    def analyze_market_conditions(
        self,
        prices: List[MarketPrice],
    ) -> MarketCondition:
        """Analyse market conditions from price observations.

        Classifies the forward curve / price trend as CONTANGO,
        BACKWARDATION, FLAT, or VOLATILE based on the slope of
        prices over time and the coefficient of variation.

        Args:
            prices: Time-ordered market price observations.

        Returns:
            MarketCondition enum value.

        Raises:
            ValueError: If fewer than 2 prices provided.
        """
        if len(prices) < 2:
            raise ValueError("At least 2 price observations required.")

        sorted_prices = sorted(prices, key=lambda p: p.date)
        price_values = [_decimal(p.price_per_mwh) for p in sorted_prices]

        mu, sigma = self._compute_price_stats(price_values)

        # Coefficient of variation.
        cv = float(_safe_divide(sigma, mu)) if mu > Decimal("0") else 0.0

        # Check volatility first.
        if cv > _VOLATILITY_CV_THRESHOLD:
            logger.info("Market condition: VOLATILE (CV=%.4f).", cv)
            return MarketCondition.VOLATILE

        # Slope: compare first-third average to last-third average.
        n = len(price_values)
        third = max(1, n // 3)
        early_avg = _safe_divide(
            sum(price_values[:third]), _decimal(third)
        )
        late_avg = _safe_divide(
            sum(price_values[-third:]), _decimal(third)
        )

        if early_avg == Decimal("0"):
            logger.info("Market condition: FLAT (early avg zero).")
            return MarketCondition.FLAT

        change_pct = float(_safe_pct(late_avg - early_avg, early_avg))

        if change_pct > _FLAT_CURVE_TOLERANCE_PCT:
            logger.info("Market condition: CONTANGO (change=%.2f%%).", change_pct)
            return MarketCondition.CONTANGO
        elif change_pct < -_FLAT_CURVE_TOLERANCE_PCT:
            logger.info("Market condition: BACKWARDATION (change=%.2f%%).", change_pct)
            return MarketCondition.BACKWARDATION
        else:
            logger.info("Market condition: FLAT (change=%.2f%%).", change_pct)
            return MarketCondition.FLAT

    # --------------------------------------------------------------------- #
    # Public API -- Hedge Value Calculation
    # --------------------------------------------------------------------- #

    def calculate_hedge_value(
        self,
        fixed_price_per_mwh: float,
        actual_index_prices: List[MarketPrice],
        volume_mwh_per_period: float,
    ) -> Decimal:
        """Calculate the value of a fixed-price hedge vs. actual index.

        hedge_value = Sum((Fixed_Price - Index_Price) * Volume_per_period)
        Positive value means the hedge saved money (index was higher).
        Negative value means the hedge cost more than spot.

        Args:
            fixed_price_per_mwh: Locked-in fixed price (EUR/MWh).
            actual_index_prices: Actual observed index prices per period.
            volume_mwh_per_period: Volume exposed per period (MWh).

        Returns:
            Total hedge value in EUR (positive = savings).
        """
        d_fixed = _decimal(fixed_price_per_mwh)
        d_volume = _decimal(volume_mwh_per_period)

        total_value = Decimal("0")
        for mp in actual_index_prices:
            d_index = _decimal(mp.price_per_mwh)
            period_value = (d_fixed - d_index) * d_volume
            total_value += period_value

        # Positive = hedge was cheaper than index (we saved money).
        # Note the sign: fixed - index > 0 means fixed was higher,
        # but that means the hedge "cost" more.  Hedge value from
        # buyer's perspective is (Index - Fixed) * Volume.
        # Correcting: hedge saved money when index > fixed.
        hedge_value = -total_value  # Flip: positive = index > fixed = savings

        logger.info(
            "Hedge value: %.2f %s (fixed=%.2f, %d periods, %.1f MWh/period).",
            float(hedge_value),
            self._currency,
            float(d_fixed),
            len(actual_index_prices),
            float(d_volume),
        )
        return hedge_value

    # --------------------------------------------------------------------- #
    # Public API -- Procurement Cost Projection
    # --------------------------------------------------------------------- #

    def project_procurement_cost(
        self,
        plan: ProcurementPlan,
        forward_curve: ForwardCurve,
    ) -> List[MonthlyProjection]:
        """Project monthly procurement costs using forward curve prices.

        For each month in the forward curve, calculates the cost of
        fixed layers (at their locked price) and index layers (at the
        forward curve price), producing a month-by-month cost projection.

        Args:
            plan: Procurement plan with hedge layers.
            forward_curve: Forward price curve for index pricing.

        Returns:
            List of MonthlyProjection with cost breakdown per month.
        """
        self._notes = [f"Engine version: {self.engine_version}"]
        month_labels = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        n_months = len(forward_curve.prices)
        if n_months == 0:
            logger.warning("Forward curve has no prices; returning empty projection.")
            return []

        d_total = _decimal(plan.total_volume_mwh)

        # Split layers into fixed and index.
        fixed_layers = [ly for ly in plan.layers if ly.hedge_type == "fixed"]
        index_layers = [ly for ly in plan.layers if ly.hedge_type != "fixed"]

        total_fixed_vol = sum(_decimal(ly.volume_mwh) for ly in fixed_layers)
        total_index_vol = d_total - total_fixed_vol

        # Monthly volume allocation (even split across months).
        monthly_fixed_vol = _safe_divide(total_fixed_vol, _decimal(n_months))
        monthly_index_vol = _safe_divide(total_index_vol, _decimal(n_months))

        # Weighted fixed price across fixed layers.
        if total_fixed_vol > Decimal("0"):
            wap_fixed = _safe_divide(
                sum(_decimal(ly.volume_mwh) * _decimal(ly.price_per_mwh) for ly in fixed_layers),
                total_fixed_vol,
            )
        else:
            wap_fixed = Decimal("0")

        projections: List[MonthlyProjection] = []

        for m_idx, fwd_price in enumerate(forward_curve.prices):
            month_num = m_idx + 1
            label = month_labels[m_idx % 12] if m_idx < 12 else f"M{month_num}"

            d_fwd = _decimal(fwd_price)
            fixed_cost = monthly_fixed_vol * wap_fixed
            index_cost = monthly_index_vol * d_fwd
            total_cost = fixed_cost + index_cost

            monthly_vol = monthly_fixed_vol + monthly_index_vol
            effective_price = _safe_divide(total_cost, monthly_vol)

            projections.append(MonthlyProjection(
                month=month_num,
                month_label=label,
                volume_mwh=_round2(float(monthly_vol)),
                fixed_cost_eur=_round2(float(fixed_cost)),
                index_cost_eur=_round2(float(index_cost)),
                total_cost_eur=_round2(float(total_cost)),
                effective_price_per_mwh=_round4(float(effective_price)),
            ))

        total_projected = sum(_decimal(p.total_cost_eur) for p in projections)
        self._notes.append(
            f"Projected {n_months} months, total cost {_round2(float(total_projected))} "
            f"{self._currency}."
        )
        logger.info(
            "Cost projection: %d months, total=%.2f %s.",
            n_months,
            float(total_projected),
            self._currency,
        )
        return projections

    # --------------------------------------------------------------------- #
    # Public API -- Full Procurement Analysis
    # --------------------------------------------------------------------- #

    def run_full_analysis(
        self,
        load_profile: List[LoadInterval],
        contracts: List[ContractTerms],
        market_prices: List[MarketPrice],
        green_products: Optional[List[GreenProductOffering]] = None,
        suppliers: Optional[List[SupplierProfile]] = None,
        risk_tolerance: float = 0.5,
    ) -> ProcurementResult:
        """Run a comprehensive procurement intelligence analysis.

        Combines contract comparison, risk assessment, procurement
        planning, green option evaluation, and supplier ranking into
        a single audit-ready result.

        Args:
            load_profile: Interval-level load data.
            contracts: Contract offers to compare.
            market_prices: Market price observations.
            green_products: Optional green product offerings.
            suppliers: Optional supplier profiles.
            risk_tolerance: Risk tolerance 0.0 to 1.0.

        Returns:
            ProcurementResult with full analysis and provenance hash.
        """
        t0 = time.perf_counter()
        notes: List[str] = [f"Engine version: {self.engine_version}"]

        # 1. Contract comparison.
        comparison = self.compare_contracts(load_profile, contracts, market_prices)
        current_cost = max(comparison.annual_cost_by_contract.values()) if comparison.annual_cost_by_contract else 0.0
        best_cost = comparison.annual_cost_by_contract.get(comparison.cheapest_contract_id, 0.0)

        # 2. Market condition.
        market_condition = self.analyze_market_conditions(market_prices)

        # 3. Risk assessment on recommended strategy.
        total_mwh = sum(_decimal(iv.kwh) for iv in load_profile) / Decimal("1000")
        chosen_strategy = self._select_strategy(risk_tolerance)
        risk = self.assess_price_risk(chosen_strategy, market_prices, float(total_mwh))

        # 4. Procurement plan.
        plan = self.develop_procurement_plan(
            float(total_mwh), market_prices, risk_tolerance, chosen_strategy
        )

        # 5. Green options.
        green_results: List[GreenProcurement] = []
        if green_products:
            green_results = self.evaluate_green_options(float(total_mwh), green_products)

        # 6. Supplier rankings.
        supplier_results: List[SupplierEvaluation] = []
        if suppliers:
            supplier_results = self.evaluate_suppliers(suppliers)

        # Projected savings.
        projected_savings = current_cost - best_cost
        savings_pct = float(_safe_pct(_decimal(projected_savings), _decimal(current_cost)))

        notes.extend(comparison.methodology_notes)
        notes.append(f"Market condition: {market_condition.value}.")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ProcurementResult(
            current_contract_cost=_round2(current_cost),
            optimal_strategy=chosen_strategy.value,
            projected_savings=_round2(projected_savings),
            savings_pct=_round2(savings_pct),
            risk_assessment=risk,
            green_options=green_results,
            supplier_rankings=supplier_results,
            execution_plan=plan,
            market_condition=market_condition.value,
            methodology_notes=notes,
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full procurement analysis complete: savings=%.2f %s (%.1f%%), hash=%s (%.1f ms)",
            projected_savings,
            self._currency,
            savings_pct,
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # ===================================================================== #
    # Private Helpers
    # ===================================================================== #

    def _build_price_lookup(
        self,
        market_prices: List[MarketPrice],
    ) -> Dict[date, Decimal]:
        """Build a date-keyed lookup of market prices.

        If multiple prices exist for the same date, the average is used.

        Args:
            market_prices: Market price observations.

        Returns:
            Dictionary mapping date to average price per MWh.
        """
        date_prices: Dict[date, List[Decimal]] = {}
        for mp in market_prices:
            date_prices.setdefault(mp.date, []).append(_decimal(mp.price_per_mwh))

        lookup: Dict[date, Decimal] = {}
        for d, prices in date_prices.items():
            lookup[d] = _safe_divide(sum(prices), _decimal(len(prices)))
        return lookup

    def _resolve_interval_price(
        self,
        interval: LoadInterval,
        price_lookup: Dict[date, Decimal],
    ) -> Optional[Decimal]:
        """Resolve the price for a load interval.

        Uses interval.price_per_kwh if available, otherwise falls back
        to the market price lookup by date (converting EUR/MWh to EUR/kWh).

        Args:
            interval: Load interval.
            price_lookup: Date-keyed market prices in EUR/MWh.

        Returns:
            Price in EUR/kWh, or None if no price available.
        """
        if interval.price_per_kwh is not None:
            return _decimal(interval.price_per_kwh)

        interval_date = interval.timestamp.date()
        mwh_price = price_lookup.get(interval_date)
        if mwh_price is not None:
            return mwh_price / Decimal("1000")  # EUR/MWh -> EUR/kWh

        return None

    def _calculate_contract_cost(
        self,
        contract: ContractTerms,
        load_profile: List[LoadInterval],
        price_lookup: Dict[date, Decimal],
        total_volume_mwh: Decimal,
    ) -> Decimal:
        """Calculate the annual cost of a contract given load and prices.

        Handles FIXED, INDEX, BLOCK_AND_INDEX, and other contract types.

        Args:
            contract: Contract terms.
            load_profile: Load intervals.
            price_lookup: Market prices by date.
            total_volume_mwh: Total load in MWh.

        Returns:
            Projected annual cost in EUR.
        """
        ct = contract.contract_type
        d_volume = _decimal(contract.volume_mwh)

        # Use the smaller of contracted volume and actual load.
        effective_volume = min(d_volume, total_volume_mwh) if d_volume > 0 else total_volume_mwh

        if ct == ContractType.FIXED:
            return self._cost_fixed(contract, effective_volume)
        elif ct == ContractType.INDEX:
            return self._cost_index(contract, load_profile, price_lookup, effective_volume)
        elif ct == ContractType.BLOCK_AND_INDEX:
            return self._cost_block_and_index(
                contract, load_profile, price_lookup, effective_volume
            )
        elif ct == ContractType.SHAPED:
            return self._cost_shaped(contract, load_profile, price_lookup, effective_volume)
        elif ct == ContractType.FULL_REQUIREMENTS:
            return self._cost_full_requirements(
                contract, load_profile, price_lookup, effective_volume
            )
        else:
            # Default: treat as fixed if fixed_price available, else index.
            if contract.fixed_price_per_mwh is not None:
                return self._cost_fixed(contract, effective_volume)
            return self._cost_index(contract, load_profile, price_lookup, effective_volume)

    def _cost_fixed(self, contract: ContractTerms, volume_mwh: Decimal) -> Decimal:
        """Calculate cost for a FIXED contract: Volume * Fixed_Price."""
        price = _decimal(contract.fixed_price_per_mwh or 0)
        return volume_mwh * price

    def _cost_index(
        self,
        contract: ContractTerms,
        load_profile: List[LoadInterval],
        price_lookup: Dict[date, Decimal],
        volume_mwh: Decimal,
    ) -> Decimal:
        """Calculate cost for an INDEX contract using interval prices.

        cost = Sum(Interval_Vol * Index_Price) + Adder * Total_Volume
        """
        adder = _decimal(contract.adder_per_mwh)

        # Calculate index cost from load intervals.
        index_cost = Decimal("0")
        total_kwh = Decimal("0")

        for iv in load_profile:
            kwh = _decimal(iv.kwh)
            price_per_kwh = self._resolve_interval_price(iv, price_lookup)
            if price_per_kwh is not None:
                index_cost += kwh * price_per_kwh
                total_kwh += kwh

        # Add the adder on total volume.
        adder_cost = adder * volume_mwh

        return index_cost + adder_cost

    def _cost_block_and_index(
        self,
        contract: ContractTerms,
        load_profile: List[LoadInterval],
        price_lookup: Dict[date, Decimal],
        volume_mwh: Decimal,
    ) -> Decimal:
        """Calculate Block & Index cost.

        Block portion at fixed price, remainder at index + adder.
        Block volume is the minimum_volume if set, otherwise 50% of total.
        """
        block_vol = _decimal(contract.minimum_volume or 0)
        if block_vol == Decimal("0"):
            block_vol = volume_mwh * Decimal("0.5")

        block_vol = min(block_vol, volume_mwh)
        index_vol = volume_mwh - block_vol

        block_price = _decimal(contract.fixed_price_per_mwh or 0)
        block_cost = block_vol * block_price

        adder = _decimal(contract.adder_per_mwh)

        # Index portion: use average market price.
        if price_lookup:
            avg_price = _safe_divide(
                sum(price_lookup.values()),
                _decimal(len(price_lookup)),
            )
        else:
            avg_price = Decimal("0")

        index_cost = index_vol * avg_price
        adder_cost = adder * volume_mwh

        self._notes.append(
            f"Block&Index {contract.contract_id[:8]}: block={_round2(float(block_vol))} MWh "
            f"@ {_round4(float(block_price))}, index={_round2(float(index_vol))} MWh "
            f"@ avg {_round4(float(avg_price))}."
        )
        return block_cost + index_cost + adder_cost

    def _cost_shaped(
        self,
        contract: ContractTerms,
        load_profile: List[LoadInterval],
        price_lookup: Dict[date, Decimal],
        volume_mwh: Decimal,
    ) -> Decimal:
        """Calculate SHAPED contract cost.

        Shaped contracts follow the buyer's load profile with a shaping
        premium applied via the adder field.
        """
        base_cost = self._cost_index(contract, load_profile, price_lookup, volume_mwh)
        # Shaping premium is captured in the adder_per_mwh field.
        return base_cost

    def _cost_full_requirements(
        self,
        contract: ContractTerms,
        load_profile: List[LoadInterval],
        price_lookup: Dict[date, Decimal],
        volume_mwh: Decimal,
    ) -> Decimal:
        """Calculate FULL_REQUIREMENTS cost.

        All-in price covering full load, balancing, and imbalance risk.
        Uses fixed_price_per_mwh as the all-in rate.
        """
        if contract.fixed_price_per_mwh is not None:
            return volume_mwh * _decimal(contract.fixed_price_per_mwh)
        # Fallback to index-based if no fixed price.
        return self._cost_index(contract, load_profile, price_lookup, volume_mwh)

    def _contract_risk_metrics(
        self,
        contract: ContractTerms,
        price_lookup: Dict[date, Decimal],
        volume_mwh: Decimal,
    ) -> Dict[str, float]:
        """Compute simplified risk metrics for a contract.

        Returns a dictionary of risk metric values for comparison.

        Args:
            contract: Contract terms.
            price_lookup: Market price lookup.
            volume_mwh: Total volume in MWh.

        Returns:
            Dictionary with risk metric name -> value.
        """
        metrics: Dict[str, float] = {}

        if contract.contract_type == ContractType.FIXED:
            # Fixed contracts have zero price risk.
            metrics[RiskMetric.STD_DEV.value] = 0.0
            metrics[RiskMetric.VAR_95.value] = 0.0
            metrics[RiskMetric.VAR_99.value] = 0.0
            return metrics

        if not price_lookup:
            return metrics

        prices = list(price_lookup.values())
        mu, sigma = self._compute_price_stats(prices)

        # Scale to cost risk.
        cost_mu = float(mu * volume_mwh)
        cost_sigma = float(sigma * volume_mwh)

        # For block & index, reduce risk by the fixed portion.
        if contract.contract_type == ContractType.BLOCK_AND_INDEX:
            block_frac = Decimal("0.5")
            if contract.minimum_volume is not None and volume_mwh > Decimal("0"):
                block_frac = _safe_divide(_decimal(contract.minimum_volume), volume_mwh)
                block_frac = min(block_frac, Decimal("1"))
            cost_sigma *= float(Decimal("1") - block_frac)

        metrics[RiskMetric.STD_DEV.value] = _round2(cost_sigma)
        metrics[RiskMetric.VAR_95.value] = _round2(cost_mu + _Z_95 * cost_sigma)
        metrics[RiskMetric.VAR_99.value] = _round2(cost_mu + _Z_99 * cost_sigma)

        return metrics

    def _compute_price_stats(
        self,
        prices: List[Decimal],
    ) -> Tuple[Decimal, Decimal]:
        """Compute mean and standard deviation of prices.

        Args:
            prices: List of price observations.

        Returns:
            Tuple of (mean, standard_deviation).
        """
        n = len(prices)
        if n == 0:
            return Decimal("0"), Decimal("0")

        d_n = _decimal(n)
        mu = _safe_divide(sum(prices), d_n)

        if n < 2:
            return mu, Decimal("0")

        variance = _safe_divide(
            sum((p - mu) ** 2 for p in prices),
            _decimal(n - 1),  # Sample variance (Bessel's correction).
        )

        # Decimal does not have sqrt; use float math then convert back.
        sigma = _decimal(math.sqrt(float(variance)))

        return mu, sigma

    def _compute_cvar(
        self,
        mu: Decimal,
        sigma: Decimal,
        volume: Decimal,
        alpha: float,
    ) -> Decimal:
        """Compute Conditional VaR (Expected Shortfall) at confidence alpha.

        CVaR_alpha = mu_cost + sigma_cost * phi(z_alpha) / (1 - alpha)
        where phi is the standard normal PDF.

        Args:
            mu: Mean price per MWh.
            sigma: Adjusted standard deviation per MWh.
            volume: Volume in MWh.
            alpha: Confidence level (e.g. 0.95).

        Returns:
            CVaR in EUR.
        """
        if alpha >= 1.0:
            alpha = 0.99

        z_alpha = _Z_95 if abs(alpha - 0.95) < 0.001 else _Z_99
        phi_z = _std_normal_pdf(z_alpha)
        tail_prob = 1.0 - alpha

        if tail_prob <= 0:
            tail_prob = 0.01

        cvar_price = mu + sigma * _decimal(phi_z / tail_prob)
        return cvar_price * volume

    def _select_strategy(self, risk_tolerance: float) -> ProcurementStrategy:
        """Select procurement strategy based on risk tolerance.

        0.0 = fully risk averse -> FULL_FIXED
        1.0 = fully risk seeking -> FULL_INDEX
        Intermediate values map to layered strategies.

        Args:
            risk_tolerance: Risk tolerance 0.0 to 1.0.

        Returns:
            Recommended ProcurementStrategy.
        """
        if risk_tolerance <= 0.15:
            return ProcurementStrategy.FULL_FIXED
        elif risk_tolerance <= 0.35:
            return ProcurementStrategy.PROGRESSIVE_FIXED
        elif risk_tolerance <= 0.55:
            return ProcurementStrategy.LAYERED_HEDGE
        elif risk_tolerance <= 0.75:
            return ProcurementStrategy.BLOCK_AND_INDEX
        elif risk_tolerance <= 0.90:
            return ProcurementStrategy.PORTFOLIO
        else:
            return ProcurementStrategy.FULL_INDEX

    def _build_layers(
        self,
        strategy: ProcurementStrategy,
        total_volume: Decimal,
        market_prices: List[MarketPrice],
        risk_tolerance: float,
    ) -> List[HedgeLayer]:
        """Build hedge layers for a procurement strategy.

        Args:
            strategy: Chosen procurement strategy.
            total_volume: Total volume to procure (MWh).
            market_prices: Market prices for indicative pricing.
            risk_tolerance: Risk tolerance for layer sizing.

        Returns:
            List of HedgeLayer defining the procurement plan.
        """
        if not market_prices:
            # Single fixed layer at zero price (no market data).
            return [HedgeLayer(
                volume_mwh=_round2(float(total_volume)),
                price_per_mwh=0.0,
                hedge_type="fixed",
                percentage_of_total=100.0,
            )]

        avg_price = _safe_divide(
            sum(_decimal(mp.price_per_mwh) for mp in market_prices),
            _decimal(len(market_prices)),
        )

        layers: List[HedgeLayer] = []

        if strategy == ProcurementStrategy.FULL_FIXED:
            layers.append(HedgeLayer(
                volume_mwh=_round2(float(total_volume)),
                price_per_mwh=_round4(float(avg_price)),
                hedge_type="fixed",
                percentage_of_total=100.0,
            ))

        elif strategy == ProcurementStrategy.FULL_INDEX:
            layers.append(HedgeLayer(
                volume_mwh=_round2(float(total_volume)),
                price_per_mwh=_round4(float(avg_price)),
                hedge_type="index",
                percentage_of_total=100.0,
            ))

        elif strategy == ProcurementStrategy.LAYERED_HEDGE:
            layers = self._layered_hedge_layers(total_volume, avg_price)

        elif strategy == ProcurementStrategy.BLOCK_AND_INDEX:
            block_pct = Decimal("0.5")
            block_vol = total_volume * block_pct
            index_vol = total_volume - block_vol
            layers.append(HedgeLayer(
                volume_mwh=_round2(float(block_vol)),
                price_per_mwh=_round4(float(avg_price)),
                hedge_type="fixed",
                percentage_of_total=50.0,
            ))
            layers.append(HedgeLayer(
                volume_mwh=_round2(float(index_vol)),
                price_per_mwh=_round4(float(avg_price)),
                hedge_type="index",
                percentage_of_total=50.0,
            ))

        elif strategy == ProcurementStrategy.PROGRESSIVE_FIXED:
            layers = self._progressive_fixed_layers(total_volume, avg_price)

        elif strategy == ProcurementStrategy.PORTFOLIO:
            layers = self._portfolio_layers(total_volume, avg_price)

        else:
            # Fallback: single layer.
            layers.append(HedgeLayer(
                volume_mwh=_round2(float(total_volume)),
                price_per_mwh=_round4(float(avg_price)),
                hedge_type="fixed",
                percentage_of_total=100.0,
            ))

        return layers

    def _layered_hedge_layers(
        self,
        total_volume: Decimal,
        avg_price: Decimal,
    ) -> List[HedgeLayer]:
        """Build 3-tranche layered hedge (40/30/30 split).

        Layer 1: 40% fixed (earliest execution, best certainty).
        Layer 2: 30% fixed (mid-term execution).
        Layer 3: 30% index (remaining floating exposure).

        Args:
            total_volume: Total volume (MWh).
            avg_price: Average market price (EUR/MWh).

        Returns:
            List of 3 HedgeLayer objects.
        """
        vol_1 = total_volume * Decimal("0.40")
        vol_2 = total_volume * Decimal("0.30")
        vol_3 = total_volume - vol_1 - vol_2

        # Slight price discount for early execution, premium for later.
        price_1 = avg_price * Decimal("0.98")   # 2% early-lock discount.
        price_2 = avg_price                       # At market.
        price_3 = avg_price                       # Index-based.

        return [
            HedgeLayer(
                volume_mwh=_round2(float(vol_1)),
                price_per_mwh=_round4(float(price_1)),
                hedge_type="fixed",
                percentage_of_total=40.0,
            ),
            HedgeLayer(
                volume_mwh=_round2(float(vol_2)),
                price_per_mwh=_round4(float(price_2)),
                hedge_type="fixed",
                percentage_of_total=30.0,
            ),
            HedgeLayer(
                volume_mwh=_round2(float(vol_3)),
                price_per_mwh=_round4(float(price_3)),
                hedge_type="index",
                percentage_of_total=30.0,
            ),
        ]

    def _progressive_fixed_layers(
        self,
        total_volume: Decimal,
        avg_price: Decimal,
    ) -> List[HedgeLayer]:
        """Build progressive fixed layers (25/25/25/25 quarterly).

        Each quarter fixes 25% more of remaining exposure.

        Args:
            total_volume: Total volume (MWh).
            avg_price: Average market price (EUR/MWh).

        Returns:
            List of 4 HedgeLayer objects.
        """
        layers: List[HedgeLayer] = []
        quarter_vol = total_volume * Decimal("0.25")

        for q in range(4):
            # Progressive price: later quarters may be at slightly different prices.
            price_adj = Decimal("1") + _decimal(q) * Decimal("0.005")
            layer_price = avg_price * price_adj

            layers.append(HedgeLayer(
                volume_mwh=_round2(float(quarter_vol)),
                price_per_mwh=_round4(float(layer_price)),
                hedge_type="fixed",
                percentage_of_total=25.0,
            ))

        return layers

    def _portfolio_layers(
        self,
        total_volume: Decimal,
        avg_price: Decimal,
    ) -> List[HedgeLayer]:
        """Build portfolio-style layers (30% fixed / 30% index / 40% mixed).

        Diversified approach balancing price certainty and market exposure.

        Args:
            total_volume: Total volume (MWh).
            avg_price: Average market price (EUR/MWh).

        Returns:
            List of 3 HedgeLayer objects.
        """
        fixed_vol = total_volume * Decimal("0.30")
        index_vol = total_volume * Decimal("0.30")
        mixed_vol = total_volume - fixed_vol - index_vol

        return [
            HedgeLayer(
                volume_mwh=_round2(float(fixed_vol)),
                price_per_mwh=_round4(float(avg_price * Decimal("0.99"))),
                hedge_type="fixed",
                percentage_of_total=30.0,
            ),
            HedgeLayer(
                volume_mwh=_round2(float(index_vol)),
                price_per_mwh=_round4(float(avg_price)),
                hedge_type="index",
                percentage_of_total=30.0,
            ),
            HedgeLayer(
                volume_mwh=_round2(float(mixed_vol)),
                price_per_mwh=_round4(float(avg_price * Decimal("1.01"))),
                hedge_type="fixed",
                percentage_of_total=40.0,
            ),
        ]

    def _weighted_avg_price(
        self,
        layers: List[HedgeLayer],
        total_volume: Decimal,
    ) -> Decimal:
        """Compute volume-weighted average price across layers.

        WAP = Sum(layer_volume * layer_price) / total_volume.

        Args:
            layers: Hedge layers.
            total_volume: Total volume (MWh).

        Returns:
            Weighted average price (EUR/MWh).
        """
        numerator = sum(
            _decimal(ly.volume_mwh) * _decimal(ly.price_per_mwh)
            for ly in layers
        )
        return _safe_divide(numerator, total_volume)

    def _build_execution_timeline(
        self,
        strategy: ProcurementStrategy,
        layers: List[HedgeLayer],
    ) -> List[str]:
        """Build a recommended execution timeline for the strategy.

        Args:
            strategy: Chosen strategy.
            layers: Hedge layers.

        Returns:
            List of timeline step descriptions.
        """
        timeline: List[str] = []

        if strategy == ProcurementStrategy.FULL_FIXED:
            timeline.append("Execute full fixed-price contract immediately.")
            timeline.append("Lock price for full term; no further market action needed.")

        elif strategy == ProcurementStrategy.FULL_INDEX:
            timeline.append("Sign index-based supply contract.")
            timeline.append("Monitor market daily; set price alerts for budget thresholds.")
            timeline.append("Consider opportunistic hedges if prices drop significantly.")

        elif strategy == ProcurementStrategy.LAYERED_HEDGE:
            timeline.append(
                f"Layer 1 ({layers[0].percentage_of_total:.0f}%): "
                "Execute immediately at current market levels."
            )
            if len(layers) > 1:
                timeline.append(
                    f"Layer 2 ({layers[1].percentage_of_total:.0f}%): "
                    "Execute within 60 days or on price target."
                )
            if len(layers) > 2:
                timeline.append(
                    f"Layer 3 ({layers[2].percentage_of_total:.0f}%): "
                    "Maintain as index exposure; hedge opportunistically."
                )

        elif strategy == ProcurementStrategy.PROGRESSIVE_FIXED:
            for idx, ly in enumerate(layers):
                quarter = idx + 1
                timeline.append(
                    f"Q{quarter}: Fix {ly.percentage_of_total:.0f}% "
                    f"({ly.volume_mwh:.0f} MWh) at target price."
                )

        elif strategy == ProcurementStrategy.BLOCK_AND_INDEX:
            timeline.append("Execute base-load block at fixed price.")
            timeline.append("Sign index supply for peak / variable volume.")
            timeline.append("Review block size quarterly against actual load shape.")

        elif strategy == ProcurementStrategy.PORTFOLIO:
            timeline.append("Execute fixed tranche immediately.")
            timeline.append("Sign index supply agreement for floating tranche.")
            timeline.append("Execute mixed tranche based on market signals.")
            timeline.append("Rebalance portfolio quarterly.")

        else:
            timeline.append("Execute procurement based on selected strategy.")

        return timeline

    def _check_re100_eligibility(self, product: GreenProductOffering) -> bool:
        """Check if a green product meets RE100 technical criteria.

        RE100 requires:
        - Energy attribute certificates from renewable sources.
        - Certificates from the same market/country as consumption.
        - Additionality encouraged (score >= 70 for full compliance).
        - Physical PPAs and bundled renewables are preferred.

        Source: RE100 Technical Criteria (2024 edition).

        Args:
            product: Green product offering.

        Returns:
            True if the product is RE100-eligible.
        """
        eligible_types = {
            GreenProduct.PPA_PHYSICAL,
            GreenProduct.PPA_VIRTUAL,
            GreenProduct.BUNDLED_RENEWABLE,
            GreenProduct.GREEN_TARIFF,
            GreenProduct.GO,
            GreenProduct.REC,
        }

        if product.product_type not in eligible_types:
            return False

        # All listed types are eligible under RE100 criteria.
        # Higher additionality scores are preferred but not disqualifying.
        # Vintage must be within 1 year of reporting year (simplified check).
        current_year = datetime.now(timezone.utc).year
        if product.certificate_vintage < current_year - 1:
            return False

        return True

    def _generate_supplier_recommendation(
        self,
        name: str,
        overall_score: float,
        credit_score: float,
    ) -> str:
        """Generate a deterministic recommendation string for a supplier.

        Args:
            name: Supplier name.
            overall_score: Weighted overall score (0-100).
            credit_score: Credit score (0-100).

        Returns:
            Recommendation text.
        """
        if overall_score >= 85:
            tier = "Strongly Recommended"
            detail = "Top-tier supplier across all criteria."
        elif overall_score >= 70:
            tier = "Recommended"
            detail = "Strong overall performance with minor gaps."
        elif overall_score >= 55:
            tier = "Conditionally Recommended"
            detail = "Acceptable but consider negotiating improved terms."
        elif overall_score >= 40:
            tier = "Caution"
            detail = "Below-average scores; require enhanced due diligence."
        else:
            tier = "Not Recommended"
            detail = "Significant weaknesses across multiple criteria."

        if credit_score < 50:
            detail += " Credit risk elevated; require financial guarantees."

        return f"{tier}: {detail}"
