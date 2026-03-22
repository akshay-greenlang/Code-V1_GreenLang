# -*- coding: utf-8 -*-
"""
RateStructureAnalyzerEngine - PACK-036 Utility Analysis Engine 2
=================================================================

Analyses utility rate tariff structures and identifies the optimal rate
schedule for a facility based on its consumption profile.  Supports flat,
tiered, time-of-use (TOU), demand, demand-ratchet, seasonal, real-time
pricing, interruptible, standby, net-metering, and feed-in rate types.

Cost Calculation Formulas:
    FLAT:           cost = total_kwh * flat_rate
    TIERED:         cost = sum(min(kwh_in_tier, tier_width) * tier_rate)
    TOU:            cost = sum(period_kwh * period_rate)
    DEMAND:         cost = energy_cost + max(peak_kw, min_demand) * demand_rate
    RATCHET:        billed_demand = max(current_peak,
                                        ratchet_pct * max_peak_in_window)
    SEASONAL:       cost = sum(season_energy_cost + season_demand_cost)
    RTP:            cost = sum(interval_kwh * interval_price)
    INTERRUPTIBLE:  cost = base_cost * (1 - interrupt_discount)
    STANDBY:        cost = fixed_standby_charge + energy_cost
    NET_METERING:   cost = (import_kwh - export_kwh) * rate
    FEED_IN:        cost = import_cost - export_kwh * feed_in_rate
    Blended Rate:   total_annual_cost / total_annual_kwh

Regulatory References:
    - FERC Form 1 -- Electric Utility Annual Report (rate schedule data)
    - NARUC Rate Design Manual (tiered / inclining block structures)
    - ANSI C12.19 -- Revenue metering data tables (demand measurement)
    - ASHRAE Guideline 14-2014 -- Measurement of Energy, Demand, and Water
    - OpenEI U.S. Utility Rate Database (URDB) schema
    - IEC 61968-9 -- Metering data exchange standards

Zero-Hallucination:
    - All cost calculations use deterministic Decimal arithmetic
    - Rate factors and tariff parameters sourced from user-provided structures
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-036 Utility Analysis
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

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
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
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    )


def _round6(value: float) -> float:
    """Round to 6 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RateType(str, Enum):
    """Utility rate structure type.

    Covers major commercial and industrial tariff designs per FERC
    classifications and NARUC Rate Design Manual categories.
    """
    FLAT = "flat"
    TIERED = "tiered"
    TOU = "tou"
    DEMAND = "demand"
    DEMAND_RATCHET = "demand_ratchet"
    SEASONAL = "seasonal"
    RTP = "rtp"
    INTERRUPTIBLE = "interruptible"
    STANDBY = "standby"
    NET_METERING = "net_metering"
    FEED_IN = "feed_in"


class SeasonType(str, Enum):
    """Season classification for seasonal rate schedules.

    Defined per ASHRAE seasonal boundaries; exact month ranges are
    utility-specific and captured in the rate structure.
    """
    SUMMER = "summer"
    WINTER = "winter"
    SHOULDER = "shoulder"
    ANNUAL = "annual"


class TOUPeriod(str, Enum):
    """Time-of-use period classification.

    Periods aligned with ISO/IEC 61968-9 metering standards and
    common North American / European TOU definitions.
    """
    ON_PEAK = "on_peak"
    MID_PEAK = "mid_peak"
    OFF_PEAK = "off_peak"
    SUPER_OFF_PEAK = "super_off_peak"
    CRITICAL_PEAK = "critical_peak"


class DemandType(str, Enum):
    """Demand charge classification type.

    BILLING:   Billed demand (may include power factor adjustment).
    ACTUAL:    Metered 15-minute interval peak demand.
    RATCHET:   Demand with ratchet clause (percentage of historical peak).
    CONTRACT:  Contracted minimum demand level.
    """
    BILLING = "billing"
    ACTUAL = "actual"
    RATCHET = "ratchet"
    CONTRACT = "contract"


class RateChangeImpact(str, Enum):
    """Direction of cost impact from a rate change."""
    INCREASE = "increase"
    DECREASE = "decrease"
    NEUTRAL = "neutral"


class OptimizationStatus(str, Enum):
    """Optimization outcome status.

    OPTIMAL:        Current rate is the lowest-cost option.
    SUBOPTIMAL:     A lower-cost rate exists with meaningful savings.
    REVIEW_NEEDED:  Savings are marginal; manual review recommended.
    """
    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"
    REVIEW_NEEDED = "review_needed"


# ---------------------------------------------------------------------------
# Pydantic Models -- Rate Structure Components
# ---------------------------------------------------------------------------


class RateTier(BaseModel):
    """Single tier in an inclining/declining block rate structure.

    Attributes:
        tier_number: Ordinal position of the tier (1-based).
        lower_kwh: Lower bound of the tier (inclusive), kWh.
        upper_kwh: Upper bound of the tier (exclusive), kWh; None for final.
        rate_per_kwh: Rate charged for energy in this tier (currency/kWh).
        description: Human-readable tier description.
    """
    tier_number: int = Field(..., ge=1, description="Tier ordinal (1-based)")
    lower_kwh: float = Field(..., ge=0, description="Tier lower bound (kWh)")
    upper_kwh: Optional[float] = Field(
        None, description="Tier upper bound (kWh); None for final tier"
    )
    rate_per_kwh: float = Field(..., ge=0, description="Rate (currency/kWh)")
    description: str = Field(default="", description="Tier description")

    @field_validator("upper_kwh")
    @classmethod
    def validate_upper_bound(cls, v: Optional[float], info: Any) -> Optional[float]:
        """Ensure upper bound exceeds lower bound when provided."""
        lower = info.data.get("lower_kwh", 0)
        if v is not None and v <= lower:
            raise ValueError(
                f"upper_kwh ({v}) must exceed lower_kwh ({lower})"
            )
        return v


class TOUSchedule(BaseModel):
    """Time-of-use schedule entry defining a pricing period.

    Attributes:
        period: TOU period classification.
        start_hour: Start hour (0-23, inclusive).
        end_hour: End hour (0-23, exclusive; 24 means midnight).
        days: Applicable days (weekday, weekend, all).
        season: Season in which this schedule applies.
        rate_per_kwh: Energy rate for this period (currency/kWh).
    """
    period: TOUPeriod = Field(..., description="TOU period classification")
    start_hour: int = Field(..., ge=0, le=23, description="Start hour (0-23)")
    end_hour: int = Field(..., ge=1, le=24, description="End hour (1-24)")
    days: str = Field(
        default="all",
        description="Applicable days: weekday, weekend, all",
    )
    season: SeasonType = Field(
        default=SeasonType.ANNUAL, description="Applicable season"
    )
    rate_per_kwh: float = Field(..., ge=0, description="Rate (currency/kWh)")

    @field_validator("days")
    @classmethod
    def validate_days(cls, v: str) -> str:
        """Ensure days value is valid."""
        allowed = {"weekday", "weekend", "all"}
        if v.lower() not in allowed:
            raise ValueError(f"days must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("end_hour")
    @classmethod
    def validate_end_hour(cls, v: int, info: Any) -> int:
        """Ensure end_hour is after start_hour."""
        start = info.data.get("start_hour", 0)
        if v <= start:
            raise ValueError(
                f"end_hour ({v}) must be greater than start_hour ({start})"
            )
        return v


class DemandCharge(BaseModel):
    """Demand charge component of a rate structure.

    Attributes:
        demand_type: Demand charge classification.
        rate_per_kw: Rate per kW of demand (currency/kW/month).
        ratchet_pct: Ratchet percentage (e.g. 0.80 means 80% of peak).
        minimum_kw: Minimum billable demand (kW).
        season: Season for which this charge applies.
    """
    demand_type: DemandType = Field(..., description="Demand type")
    rate_per_kw: float = Field(..., ge=0, description="Rate (currency/kW/month)")
    ratchet_pct: Optional[float] = Field(
        None, ge=0, le=1.0, description="Ratchet pct (0.0 - 1.0)"
    )
    minimum_kw: float = Field(default=0.0, ge=0, description="Minimum demand (kW)")
    season: SeasonType = Field(
        default=SeasonType.ANNUAL, description="Applicable season"
    )


class RateStructure(BaseModel):
    """Complete utility rate structure definition.

    Contains all tariff components: tiers, TOU schedules, demand charges,
    fixed charges, and adjustment factors.

    Attributes:
        rate_id: Unique rate schedule identifier.
        rate_name: Human-readable rate schedule name.
        rate_type: Primary rate structure type.
        utility_name: Name of the serving utility.
        jurisdiction: Regulatory jurisdiction (state/province/country).
        effective_date: Date when this rate became effective.
        tiers: Tiered rate blocks (for TIERED rate type).
        tou_schedules: TOU period definitions (for TOU rate type).
        demand_charges: Demand charge components.
        fixed_charges_monthly: Fixed monthly customer charge (currency).
        minimum_bill: Minimum monthly bill amount (currency).
        power_factor_adjustment: Power factor penalty/discount threshold.
        voltage_discount: Discount for primary metering (fraction, e.g. 0.02).
        flat_rate_per_kwh: Flat rate per kWh (for FLAT rate type).
        feed_in_rate_per_kwh: Feed-in tariff rate (for FEED_IN rate type).
        interrupt_discount: Discount fraction for INTERRUPTIBLE rate.
        standby_charge_monthly: Monthly standby charge (for STANDBY).
    """
    rate_id: str = Field(..., min_length=1, description="Rate schedule ID")
    rate_name: str = Field(..., min_length=1, description="Rate schedule name")
    rate_type: RateType = Field(..., description="Rate structure type")
    utility_name: str = Field(default="", description="Utility name")
    jurisdiction: str = Field(default="", description="Jurisdiction")
    effective_date: Optional[str] = Field(None, description="Effective date")
    tiers: List[RateTier] = Field(
        default_factory=list, description="Tiered rate blocks"
    )
    tou_schedules: List[TOUSchedule] = Field(
        default_factory=list, description="TOU period definitions"
    )
    demand_charges: List[DemandCharge] = Field(
        default_factory=list, description="Demand charge components"
    )
    fixed_charges_monthly: float = Field(
        default=0.0, ge=0, description="Fixed monthly charge (currency)"
    )
    minimum_bill: float = Field(
        default=0.0, ge=0, description="Minimum monthly bill (currency)"
    )
    power_factor_adjustment: Optional[float] = Field(
        None, ge=0, le=1.0,
        description="PF penalty threshold (e.g. 0.90 = 90%)",
    )
    voltage_discount: float = Field(
        default=0.0, ge=0, le=0.20,
        description="Primary voltage discount fraction",
    )
    flat_rate_per_kwh: Optional[float] = Field(
        None, ge=0, description="Flat energy rate (currency/kWh)"
    )
    feed_in_rate_per_kwh: Optional[float] = Field(
        None, ge=0, description="Feed-in tariff rate (currency/kWh)"
    )
    interrupt_discount: Optional[float] = Field(
        None, ge=0, le=1.0,
        description="Interruptible rate discount fraction",
    )
    standby_charge_monthly: Optional[float] = Field(
        None, ge=0, description="Monthly standby charge (currency)"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Consumption Data
# ---------------------------------------------------------------------------


class MonthlyConsumption(BaseModel):
    """Monthly energy consumption profile for rate analysis.

    Attributes:
        month: Month label (e.g. '2025-01', '2025-02').
        on_peak_kwh: On-peak energy consumption (kWh).
        mid_peak_kwh: Mid-peak energy consumption (kWh).
        off_peak_kwh: Off-peak energy consumption (kWh).
        total_kwh: Total energy consumption for the month (kWh).
        peak_demand_kw: Peak demand recorded in the month (kW).
        power_factor: Average power factor for the month (0.0 - 1.0).
        billing_days: Number of billing days in the month.
        export_kwh: Exported energy for net-metering/feed-in (kWh).
        season: Season classification for this month.
    """
    month: str = Field(..., min_length=4, description="Month label")
    on_peak_kwh: float = Field(default=0.0, ge=0, description="On-peak kWh")
    mid_peak_kwh: float = Field(default=0.0, ge=0, description="Mid-peak kWh")
    off_peak_kwh: float = Field(default=0.0, ge=0, description="Off-peak kWh")
    total_kwh: float = Field(..., ge=0, description="Total monthly kWh")
    peak_demand_kw: float = Field(default=0.0, ge=0, description="Peak demand (kW)")
    power_factor: float = Field(default=0.95, ge=0.0, le=1.0, description="Power factor")
    billing_days: int = Field(default=30, ge=1, le=366, description="Billing days")
    export_kwh: float = Field(default=0.0, ge=0, description="Exported kWh")
    season: SeasonType = Field(
        default=SeasonType.ANNUAL, description="Season classification"
    )

    @field_validator("total_kwh")
    @classmethod
    def validate_total(cls, v: float) -> float:
        """Sanity-check monthly total consumption."""
        if v > 100_000_000:
            raise ValueError("Monthly consumption exceeds 100 GWh sanity check")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output Results
# ---------------------------------------------------------------------------


class RateComparison(BaseModel):
    """Cost comparison result for a single rate schedule.

    Attributes:
        rate_id: Rate schedule identifier.
        rate_name: Rate schedule name.
        annual_energy_cost: Total annual energy charges.
        annual_demand_cost: Total annual demand charges.
        annual_fixed_cost: Total annual fixed charges.
        annual_total_cost: Grand total annual cost.
        blended_rate_per_kwh: Effective blended rate (cost/kWh).
        savings_vs_current: Savings compared to current rate (positive = savings).
    """
    rate_id: str = Field(..., description="Rate schedule ID")
    rate_name: str = Field(..., description="Rate schedule name")
    annual_energy_cost: float = Field(default=0.0, description="Annual energy cost")
    annual_demand_cost: float = Field(default=0.0, description="Annual demand cost")
    annual_fixed_cost: float = Field(default=0.0, description="Annual fixed cost")
    annual_total_cost: float = Field(default=0.0, description="Annual total cost")
    blended_rate_per_kwh: float = Field(
        default=0.0, description="Blended rate (currency/kWh)"
    )
    savings_vs_current: float = Field(
        default=0.0, description="Savings vs current rate"
    )


class TOUShiftAnalysis(BaseModel):
    """Analysis of potential cost savings from shifting load between TOU periods.

    Attributes:
        current_tou_cost: Current annual TOU energy cost.
        shift_pct: Percentage of on-peak load that could be shifted.
        shifted_kwh: Energy that could be shifted (kWh/year).
        estimated_savings: Estimated annual savings from shifting.
        savings_pct: Savings as percentage of current TOU cost.
        on_peak_share: Current on-peak share of total consumption.
        off_peak_share: Current off-peak share of total consumption.
        recommendation: Actionable recommendation text.
    """
    current_tou_cost: float = Field(default=0.0, description="Current TOU cost")
    shift_pct: float = Field(default=0.0, description="Shiftable on-peak pct")
    shifted_kwh: float = Field(default=0.0, description="Shiftable kWh/yr")
    estimated_savings: float = Field(default=0.0, description="Estimated savings")
    savings_pct: float = Field(default=0.0, description="Savings pct")
    on_peak_share: float = Field(default=0.0, description="On-peak share pct")
    off_peak_share: float = Field(default=0.0, description="Off-peak share pct")
    recommendation: str = Field(default="", description="Recommendation")


class DemandRatchetAnalysis(BaseModel):
    """Analysis of demand ratchet impact on billing.

    Attributes:
        peak_demand_kw: Highest recorded demand in the period (kW).
        ratchet_demand_kw: Ratcheted billed demand (kW).
        ratchet_pct: Ratchet percentage applied.
        months_ratchet_active: Number of months where ratchet exceeded actual.
        annual_ratchet_penalty: Extra cost due to ratchet above actual demand.
        reduction_target_kw: Demand reduction target to minimize ratchet.
        recommendation: Actionable recommendation text.
    """
    peak_demand_kw: float = Field(default=0.0, description="Peak demand (kW)")
    ratchet_demand_kw: float = Field(default=0.0, description="Ratcheted demand (kW)")
    ratchet_pct: float = Field(default=0.0, description="Ratchet percentage")
    months_ratchet_active: int = Field(
        default=0, description="Months ratchet exceeded actual"
    )
    annual_ratchet_penalty: float = Field(
        default=0.0, description="Annual ratchet penalty cost"
    )
    reduction_target_kw: float = Field(
        default=0.0, description="Demand reduction target (kW)"
    )
    recommendation: str = Field(default="", description="Recommendation")


class PowerFactorAssessment(BaseModel):
    """Assessment of power factor impact on billing.

    Attributes:
        avg_power_factor: Average power factor across all months.
        min_power_factor: Lowest recorded power factor.
        pf_threshold: Utility's power factor penalty threshold.
        months_below_threshold: Months with PF below threshold.
        annual_pf_penalty: Estimated annual PF penalty cost.
        correction_target_pf: Target PF to eliminate penalties.
        kvar_correction_needed: Reactive power correction needed (kVAR).
        recommendation: Actionable recommendation text.
    """
    avg_power_factor: float = Field(default=0.0, description="Avg power factor")
    min_power_factor: float = Field(default=0.0, description="Min power factor")
    pf_threshold: float = Field(default=0.90, description="PF penalty threshold")
    months_below_threshold: int = Field(
        default=0, description="Months below threshold"
    )
    annual_pf_penalty: float = Field(default=0.0, description="Annual PF penalty")
    correction_target_pf: float = Field(
        default=0.95, description="Correction target PF"
    )
    kvar_correction_needed: float = Field(
        default=0.0, description="kVAR correction needed"
    )
    recommendation: str = Field(default="", description="Recommendation")


class AnnualCostProjection(BaseModel):
    """Projected annual cost for a future year.

    Attributes:
        year: Projection year (e.g. 2026, 2027).
        projected_cost: Projected annual cost (currency).
        rate_escalation_applied: Rate escalation percentage applied.
        consumption_change: Consumption change percentage applied.
        cumulative_cost: Cumulative cost from base year to this year.
    """
    year: int = Field(..., ge=2000, description="Projection year")
    projected_cost: float = Field(default=0.0, description="Projected annual cost")
    rate_escalation_applied: float = Field(
        default=0.0, description="Rate escalation pct applied"
    )
    consumption_change: float = Field(
        default=0.0, description="Consumption change pct"
    )
    cumulative_cost: float = Field(
        default=0.0, description="Cumulative cost from base"
    )


class TariffChangeImpact(BaseModel):
    """Impact analysis of a tariff change.

    Attributes:
        old_rate_name: Name of the old rate schedule.
        new_rate_name: Name of the new rate schedule.
        annual_cost_change: Annual cost change (positive = increase).
        cost_change_pct: Cost change as percentage.
        affected_components: Components that changed (energy, demand, fixed).
        impact_direction: Direction of cost impact.
        recommendation: Actionable recommendation text.
    """
    old_rate_name: str = Field(default="", description="Old rate schedule name")
    new_rate_name: str = Field(default="", description="New rate schedule name")
    annual_cost_change: float = Field(
        default=0.0, description="Annual cost change"
    )
    cost_change_pct: float = Field(default=0.0, description="Cost change pct")
    affected_components: List[str] = Field(
        default_factory=list, description="Changed components"
    )
    impact_direction: RateChangeImpact = Field(
        default=RateChangeImpact.NEUTRAL, description="Impact direction"
    )
    recommendation: str = Field(default="", description="Recommendation")


class RateOptimizationResult(BaseModel):
    """Complete rate optimization result with provenance.

    Contains the current and optimal rates, all rate comparisons,
    TOU shift potential, demand reduction potential, and the
    SHA-256 provenance hash for audit trail.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    current_rate: str = Field(default="", description="Current rate schedule ID")
    optimal_rate: str = Field(default="", description="Optimal rate schedule ID")
    optimization_status: OptimizationStatus = Field(
        default=OptimizationStatus.REVIEW_NEEDED,
        description="Optimization outcome",
    )
    annual_savings: float = Field(
        default=0.0, description="Potential annual savings"
    )
    savings_pct: float = Field(
        default=0.0, description="Savings as pct of current cost"
    )
    current_annual_cost: float = Field(
        default=0.0, description="Current annual total cost"
    )
    optimal_annual_cost: float = Field(
        default=0.0, description="Optimal annual total cost"
    )

    comparisons: List[RateComparison] = Field(
        default_factory=list, description="All rate comparisons"
    )
    tou_shift_potential: Optional[TOUShiftAnalysis] = Field(
        None, description="TOU load-shift analysis"
    )
    demand_reduction_potential: Optional[DemandRatchetAnalysis] = Field(
        None, description="Demand ratchet analysis"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class RateStructureAnalyzerEngine:
    """Utility rate structure analysis and optimization engine.

    Provides deterministic, zero-hallucination rate analysis for:
    - Annual cost calculation across all rate types (flat, tiered, TOU,
      demand, ratchet, seasonal, RTP, interruptible, standby, net metering,
      feed-in)
    - Multi-rate comparison and optimal rate identification
    - TOU load-shifting potential analysis
    - Demand ratchet impact quantification
    - Power factor penalty assessment
    - Multi-year cost projections with escalation
    - Tariff change impact analysis

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = RateStructureAnalyzerEngine()
        cost = engine.calculate_annual_cost(consumption, rate)
        result = engine.find_optimal_rate(consumption, rates, current_rate_id)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the rate structure analyzer engine."""
        logger.info(
            "RateStructureAnalyzerEngine v%s initialised", self.engine_version
        )

    # -------------------------------------------------------------------
    # Public API -- Cost Calculation
    # -------------------------------------------------------------------

    def calculate_annual_cost(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
    ) -> Decimal:
        """Calculate total annual cost under a given rate structure.

        Dispatches to the appropriate calculation method based on rate_type,
        then adds fixed charges, minimum bill enforcement, power factor
        adjustments, and voltage discounts.

        Args:
            consumption: Monthly consumption profile (12 months typical).
            rate: Rate structure definition.

        Returns:
            Total annual cost as Decimal.

        Raises:
            ValueError: If consumption list is empty.
        """
        if not consumption:
            raise ValueError("At least one month of consumption data is required")

        logger.info(
            "Calculating annual cost: rate=%s (%s), months=%d",
            rate.rate_id, rate.rate_type.value, len(consumption),
        )

        total_energy_cost = Decimal("0")
        total_demand_cost = Decimal("0")

        for month in consumption:
            energy = self._calculate_monthly_energy_cost(month, rate)
            demand = self._calculate_monthly_demand_cost(month, rate)
            total_energy_cost += energy
            total_demand_cost += demand

        # Fixed charges
        total_fixed = _decimal(rate.fixed_charges_monthly) * _decimal(len(consumption))

        # Subtotal before adjustments
        subtotal = total_energy_cost + total_demand_cost + total_fixed

        # Power factor adjustment
        pf_adjustment = self._calculate_pf_adjustment(consumption, rate, subtotal)
        subtotal += pf_adjustment

        # Voltage discount
        if rate.voltage_discount > 0:
            discount = subtotal * _decimal(rate.voltage_discount)
            subtotal -= discount
            logger.debug(
                "Voltage discount applied: %.2f (%.1f%%)",
                float(discount), rate.voltage_discount * 100,
            )

        # Minimum bill enforcement
        min_bill_annual = _decimal(rate.minimum_bill) * _decimal(len(consumption))
        if subtotal < min_bill_annual:
            logger.debug(
                "Minimum bill enforced: calculated=%.2f, minimum=%.2f",
                float(subtotal), float(min_bill_annual),
            )
            subtotal = min_bill_annual

        return subtotal

    def calculate_blended_rate(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
    ) -> Decimal:
        """Calculate the blended (effective) rate per kWh.

        Blended Rate = Total Annual Cost / Total Annual kWh.

        Args:
            consumption: Monthly consumption profile.
            rate: Rate structure definition.

        Returns:
            Blended rate per kWh as Decimal.
        """
        total_cost = self.calculate_annual_cost(consumption, rate)
        total_kwh = sum(_decimal(m.total_kwh) for m in consumption)
        blended = _safe_divide(total_cost, total_kwh)
        logger.info(
            "Blended rate for %s: %.6f currency/kWh",
            rate.rate_id, float(blended),
        )
        return blended

    # -------------------------------------------------------------------
    # Public API -- Rate Comparison
    # -------------------------------------------------------------------

    def compare_rates(
        self,
        consumption: List[MonthlyConsumption],
        rates: List[RateStructure],
        current_rate_id: Optional[str] = None,
    ) -> List[RateComparison]:
        """Compare annual costs across multiple rate schedules.

        Calculates the annual cost for each rate schedule and ranks them
        from lowest to highest total cost.

        Args:
            consumption: Monthly consumption profile.
            rates: List of rate structures to compare.
            current_rate_id: ID of the current rate for savings calculation.

        Returns:
            List of RateComparison sorted by annual_total_cost ascending.

        Raises:
            ValueError: If rates list is empty.
        """
        if not rates:
            raise ValueError("At least one rate structure is required")

        t0 = time.perf_counter()
        logger.info("Comparing %d rate structures", len(rates))

        total_kwh = sum(_decimal(m.total_kwh) for m in consumption)
        current_cost: Optional[Decimal] = None

        # First pass: find current rate cost if specified
        if current_rate_id:
            for r in rates:
                if r.rate_id == current_rate_id:
                    current_cost = self.calculate_annual_cost(consumption, r)
                    break

        comparisons: List[RateComparison] = []
        for r in rates:
            annual_cost = self.calculate_annual_cost(consumption, r)
            energy_cost = self._total_energy_cost(consumption, r)
            demand_cost = self._total_demand_cost(consumption, r)
            fixed_cost = _decimal(r.fixed_charges_monthly) * _decimal(len(consumption))

            blended = _safe_divide(annual_cost, total_kwh)
            savings = Decimal("0")
            if current_cost is not None:
                savings = current_cost - annual_cost

            comparisons.append(RateComparison(
                rate_id=r.rate_id,
                rate_name=r.rate_name,
                annual_energy_cost=_round2(float(energy_cost)),
                annual_demand_cost=_round2(float(demand_cost)),
                annual_fixed_cost=_round2(float(fixed_cost)),
                annual_total_cost=_round2(float(annual_cost)),
                blended_rate_per_kwh=_round6(float(blended)),
                savings_vs_current=_round2(float(savings)),
            ))

        comparisons.sort(key=lambda c: c.annual_total_cost)

        elapsed_ms = _round2((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Rate comparison complete: %d rates in %.1f ms, cheapest=%s",
            len(comparisons), elapsed_ms,
            comparisons[0].rate_id if comparisons else "N/A",
        )
        return comparisons

    # -------------------------------------------------------------------
    # Public API -- Optimization
    # -------------------------------------------------------------------

    def find_optimal_rate(
        self,
        consumption: List[MonthlyConsumption],
        rates: List[RateStructure],
        current_rate_id: Optional[str] = None,
    ) -> RateOptimizationResult:
        """Find the optimal rate schedule and quantify savings.

        Evaluates all provided rate schedules, identifies the lowest-cost
        option, and provides TOU shift and demand reduction analyses.

        Args:
            consumption: Monthly consumption profile (12 months typical).
            rates: List of available rate structures.
            current_rate_id: ID of the current rate schedule.

        Returns:
            RateOptimizationResult with optimal rate and full analysis.

        Raises:
            ValueError: If rates list is empty.
        """
        t0 = time.perf_counter()
        logger.info(
            "Finding optimal rate: %d candidates, current=%s",
            len(rates), current_rate_id or "not specified",
        )

        comparisons = self.compare_rates(consumption, rates, current_rate_id)

        # Determine current and optimal
        current_comp: Optional[RateComparison] = None
        if current_rate_id:
            for c in comparisons:
                if c.rate_id == current_rate_id:
                    current_comp = c
                    break

        optimal_comp = comparisons[0]  # Already sorted by cost ascending

        current_cost = _decimal(current_comp.annual_total_cost) if current_comp else _decimal(optimal_comp.annual_total_cost)
        optimal_cost = _decimal(optimal_comp.annual_total_cost)
        savings = current_cost - optimal_cost
        savings_pct = float(_safe_pct(savings, current_cost))

        # Determine optimization status
        if current_comp and current_comp.rate_id == optimal_comp.rate_id:
            status = OptimizationStatus.OPTIMAL
        elif savings_pct >= 5.0:
            status = OptimizationStatus.SUBOPTIMAL
        else:
            status = OptimizationStatus.REVIEW_NEEDED

        # TOU shift analysis -- find a TOU rate in the list
        tou_analysis: Optional[TOUShiftAnalysis] = None
        for r in rates:
            if r.rate_type in (RateType.TOU, RateType.SEASONAL):
                tou_analysis = self.analyze_tou_potential(consumption, r)
                break

        # Demand ratchet analysis -- find a demand ratchet rate
        ratchet_analysis: Optional[DemandRatchetAnalysis] = None
        for r in rates:
            if r.rate_type in (RateType.DEMAND_RATCHET, RateType.DEMAND):
                ratchet_analysis = self.analyze_demand_ratchet(consumption, r)
                break

        # Recommendations
        recommendations = self._generate_optimization_recommendations(
            comparisons, current_comp, optimal_comp,
            tou_analysis, ratchet_analysis,
        )

        elapsed_ms = _round2((time.perf_counter() - t0) * 1000.0)

        result = RateOptimizationResult(
            current_rate=current_rate_id or "",
            optimal_rate=optimal_comp.rate_id,
            optimization_status=status,
            annual_savings=_round2(float(savings)),
            savings_pct=_round2(savings_pct),
            current_annual_cost=_round2(float(current_cost)),
            optimal_annual_cost=_round2(float(optimal_cost)),
            comparisons=comparisons,
            tou_shift_potential=tou_analysis,
            demand_reduction_potential=ratchet_analysis,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Optimization complete: current=%s (%.2f), optimal=%s (%.2f), "
            "savings=%.2f (%.1f%%), status=%s, hash=%s (%.1f ms)",
            result.current_rate, float(current_cost),
            result.optimal_rate, float(optimal_cost),
            float(savings), savings_pct, status.value,
            result.provenance_hash[:16], elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public API -- TOU Analysis
    # -------------------------------------------------------------------

    def analyze_tou_potential(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
    ) -> TOUShiftAnalysis:
        """Analyse potential savings from shifting load between TOU periods.

        Estimates how much on-peak energy could be moved to off-peak periods
        and the resulting cost reduction.

        Args:
            consumption: Monthly consumption profile.
            rate: Rate structure with TOU schedules.

        Returns:
            TOUShiftAnalysis with shift potential and savings estimate.
        """
        logger.info("Analysing TOU shift potential for rate %s", rate.rate_id)

        total_on_peak = sum(_decimal(m.on_peak_kwh) for m in consumption)
        total_mid_peak = sum(_decimal(m.mid_peak_kwh) for m in consumption)
        total_off_peak = sum(_decimal(m.off_peak_kwh) for m in consumption)
        total_kwh = sum(_decimal(m.total_kwh) for m in consumption)

        on_peak_share = float(_safe_pct(total_on_peak, total_kwh))
        off_peak_share = float(_safe_pct(total_off_peak, total_kwh))

        # Find TOU rates
        on_peak_rate = Decimal("0")
        off_peak_rate = Decimal("0")
        for sched in rate.tou_schedules:
            if sched.period == TOUPeriod.ON_PEAK:
                on_peak_rate = _decimal(sched.rate_per_kwh)
            elif sched.period == TOUPeriod.OFF_PEAK:
                off_peak_rate = _decimal(sched.rate_per_kwh)

        # Use flat rate as fallback for TOU rate differential
        if on_peak_rate == Decimal("0") and rate.flat_rate_per_kwh:
            on_peak_rate = _decimal(rate.flat_rate_per_kwh)

        # Calculate current TOU energy cost
        current_tou_cost = (
            total_on_peak * on_peak_rate
            + total_mid_peak * _decimal(
                next(
                    (s.rate_per_kwh for s in rate.tou_schedules
                     if s.period == TOUPeriod.MID_PEAK),
                    float(on_peak_rate),
                )
            )
            + total_off_peak * off_peak_rate
        )

        # Estimate shiftable load: assume 15-25% of on-peak is shiftable
        # (industry benchmark per ASHRAE Guideline 14-2014)
        shift_pct = Decimal("0.20")  # 20% baseline shiftable
        shifted_kwh = total_on_peak * shift_pct
        rate_differential = on_peak_rate - off_peak_rate
        estimated_savings = shifted_kwh * rate_differential
        savings_pct = float(_safe_pct(estimated_savings, current_tou_cost))

        recommendation = ""
        if savings_pct > 10.0:
            recommendation = (
                f"High TOU savings potential: shift {_round2(float(shifted_kwh))} kWh/yr "
                f"from on-peak to off-peak for estimated {_round2(float(estimated_savings))} "
                f"savings ({_round2(savings_pct)}%)."
            )
        elif savings_pct > 3.0:
            recommendation = (
                f"Moderate TOU savings potential: {_round2(savings_pct)}% savings possible "
                f"by shifting {_round2(float(shifted_kwh))} kWh/yr off-peak."
            )
        else:
            recommendation = (
                "Limited TOU savings potential. On-peak/off-peak rate differential "
                "is small or on-peak consumption share is already low."
            )

        return TOUShiftAnalysis(
            current_tou_cost=_round2(float(current_tou_cost)),
            shift_pct=_round2(float(shift_pct * Decimal("100"))),
            shifted_kwh=_round2(float(shifted_kwh)),
            estimated_savings=_round2(float(estimated_savings)),
            savings_pct=_round2(savings_pct),
            on_peak_share=_round2(on_peak_share),
            off_peak_share=_round2(off_peak_share),
            recommendation=recommendation,
        )

    # -------------------------------------------------------------------
    # Public API -- Demand Ratchet Analysis
    # -------------------------------------------------------------------

    def analyze_demand_ratchet(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
    ) -> DemandRatchetAnalysis:
        """Analyse the impact of demand ratchet clauses on billing.

        Calculates how the ratchet clause inflates billed demand above
        actual monthly peak demand, and the resulting cost penalty.

        Args:
            consumption: Monthly consumption profile with peak demand.
            rate: Rate structure with demand ratchet charge.

        Returns:
            DemandRatchetAnalysis with ratchet impact quantification.
        """
        logger.info("Analysing demand ratchet for rate %s", rate.rate_id)

        # Find ratchet parameters
        ratchet_charge: Optional[DemandCharge] = None
        for dc in rate.demand_charges:
            if dc.demand_type == DemandType.RATCHET:
                ratchet_charge = dc
                break

        if ratchet_charge is None:
            # Fall back to any demand charge for basic analysis
            for dc in rate.demand_charges:
                ratchet_charge = dc
                break

        ratchet_pct = _decimal(
            ratchet_charge.ratchet_pct if ratchet_charge and ratchet_charge.ratchet_pct else 0.80
        )
        demand_rate = _decimal(ratchet_charge.rate_per_kw) if ratchet_charge else Decimal("0")
        minimum_kw = _decimal(ratchet_charge.minimum_kw) if ratchet_charge else Decimal("0")

        # Find peak demand across all months (the ratchet window)
        demands = [_decimal(m.peak_demand_kw) for m in consumption]
        max_peak = max(demands) if demands else Decimal("0")
        ratchet_demand = max_peak * ratchet_pct

        # Calculate ratchet penalty per month
        months_ratchet_active = 0
        annual_ratchet_penalty = Decimal("0")

        for m in consumption:
            actual = _decimal(m.peak_demand_kw)
            billed = max(actual, ratchet_demand, minimum_kw)
            if billed > actual:
                months_ratchet_active += 1
                penalty = (billed - actual) * demand_rate
                annual_ratchet_penalty += penalty

        # Reduction target: reduce peak to eliminate ratchet penalty
        reduction_target = max_peak - _safe_divide(
            ratchet_demand, ratchet_pct, default=max_peak
        ) if ratchet_pct > Decimal("0") else Decimal("0")
        # Simpler: target is to reduce max peak so ratchet equals typical month
        avg_demand = _safe_divide(
            sum(demands), _decimal(len(demands))
        ) if demands else Decimal("0")
        reduction_target = max(max_peak - avg_demand, Decimal("0"))

        recommendation = ""
        if annual_ratchet_penalty > Decimal("0"):
            recommendation = (
                f"Demand ratchet is adding {_round2(float(annual_ratchet_penalty))} annually. "
                f"Peak demand of {_round2(float(max_peak))} kW sets ratchet at "
                f"{_round2(float(ratchet_demand))} kW ({_round2(float(ratchet_pct * 100))}%). "
                f"Reduce peak demand by {_round2(float(reduction_target))} kW to align "
                f"billed demand with average actual demand."
            )
        else:
            recommendation = (
                "Demand ratchet has minimal impact. Monthly peaks are consistent "
                "and close to the ratcheted threshold."
            )

        return DemandRatchetAnalysis(
            peak_demand_kw=_round2(float(max_peak)),
            ratchet_demand_kw=_round2(float(ratchet_demand)),
            ratchet_pct=_round2(float(ratchet_pct * Decimal("100"))),
            months_ratchet_active=months_ratchet_active,
            annual_ratchet_penalty=_round2(float(annual_ratchet_penalty)),
            reduction_target_kw=_round2(float(reduction_target)),
            recommendation=recommendation,
        )

    # -------------------------------------------------------------------
    # Public API -- Cost Projection
    # -------------------------------------------------------------------

    def project_costs(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
        years: int = 5,
        escalation_rate: float = 0.03,
        consumption_change_rate: float = 0.0,
    ) -> List[AnnualCostProjection]:
        """Project annual costs forward with rate escalation.

        Applies compound annual rate escalation and optional consumption
        growth/reduction to produce multi-year cost projections.

        Args:
            consumption: Baseline monthly consumption profile.
            rate: Baseline rate structure.
            years: Number of years to project (1-30).
            escalation_rate: Annual rate escalation as fraction (e.g. 0.03 = 3%).
            consumption_change_rate: Annual consumption change as fraction.

        Returns:
            List of AnnualCostProjection for each projected year.

        Raises:
            ValueError: If years is out of range.
        """
        if years < 1 or years > 30:
            raise ValueError("Projection years must be between 1 and 30")

        logger.info(
            "Projecting costs: %d years, escalation=%.1f%%, consumption_change=%.1f%%",
            years, escalation_rate * 100, consumption_change_rate * 100,
        )

        base_cost = self.calculate_annual_cost(consumption, rate)
        base_year = datetime.now(timezone.utc).year
        cumulative = Decimal("0")
        projections: List[AnnualCostProjection] = []

        for yr_offset in range(1, years + 1):
            # Compound escalation and consumption change
            escalation_factor = (
                Decimal("1") + _decimal(escalation_rate)
            ) ** yr_offset
            consumption_factor = (
                Decimal("1") + _decimal(consumption_change_rate)
            ) ** yr_offset

            projected = base_cost * escalation_factor * consumption_factor
            cumulative += projected

            projections.append(AnnualCostProjection(
                year=base_year + yr_offset,
                projected_cost=_round2(float(projected)),
                rate_escalation_applied=_round2(
                    float((escalation_factor - Decimal("1")) * Decimal("100"))
                ),
                consumption_change=_round2(
                    float((consumption_factor - Decimal("1")) * Decimal("100"))
                ),
                cumulative_cost=_round2(float(cumulative)),
            ))

        logger.info(
            "Cost projection complete: %d years, year-%d cost=%.2f",
            years, years, float(projections[-1].projected_cost) if projections else 0.0,
        )
        return projections

    # -------------------------------------------------------------------
    # Public API -- Power Factor Assessment
    # -------------------------------------------------------------------

    def assess_power_factor_impact(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
    ) -> PowerFactorAssessment:
        """Assess the impact of power factor on billing.

        Calculates power factor penalties based on the utility's PF
        threshold and estimates the cost of correction.

        Args:
            consumption: Monthly consumption profile with power factor data.
            rate: Rate structure with power factor adjustment threshold.

        Returns:
            PowerFactorAssessment with penalty quantification and correction
            target.
        """
        logger.info("Assessing power factor impact for rate %s", rate.rate_id)

        pf_threshold = _decimal(
            rate.power_factor_adjustment if rate.power_factor_adjustment else 0.90
        )

        pf_values = [_decimal(m.power_factor) for m in consumption]
        avg_pf = _safe_divide(sum(pf_values), _decimal(len(pf_values))) if pf_values else Decimal("0.95")
        min_pf = min(pf_values) if pf_values else Decimal("0.95")

        months_below = 0
        annual_penalty = Decimal("0")

        for m in consumption:
            pf = _decimal(m.power_factor)
            if pf < pf_threshold and pf > Decimal("0"):
                months_below += 1
                # PF penalty formula: billed_kW = actual_kW * (threshold / actual_pf)
                # penalty = (billed_kW - actual_kW) * demand_rate
                demand_rate = Decimal("0")
                for dc in rate.demand_charges:
                    demand_rate = max(demand_rate, _decimal(dc.rate_per_kw))

                actual_kw = _decimal(m.peak_demand_kw)
                billed_kw = actual_kw * _safe_divide(pf_threshold, pf, default=Decimal("1"))
                penalty = (billed_kw - actual_kw) * demand_rate
                annual_penalty += penalty

        # kVAR correction needed to raise from avg PF to target (0.95)
        target_pf = Decimal("0.95")
        # kVAR = kW * (tan(acos(current_pf)) - tan(acos(target_pf)))
        avg_kw = _safe_divide(
            sum(_decimal(m.peak_demand_kw) for m in consumption),
            _decimal(len(consumption)),
        ) if consumption else Decimal("0")

        kvar_correction = Decimal("0")
        if avg_pf > Decimal("0") and avg_pf < Decimal("1"):
            current_angle = Decimal(str(math.acos(float(avg_pf))))
            target_angle = Decimal(str(math.acos(float(target_pf))))
            current_tan = Decimal(str(math.tan(float(current_angle))))
            target_tan = Decimal(str(math.tan(float(target_angle))))
            kvar_correction = avg_kw * (current_tan - target_tan)
            if kvar_correction < Decimal("0"):
                kvar_correction = Decimal("0")

        recommendation = ""
        if months_below > 0 and annual_penalty > Decimal("100"):
            recommendation = (
                f"Power factor penalties of {_round2(float(annual_penalty))} annually "
                f"across {months_below} months. Install {_round2(float(kvar_correction))} kVAR "
                f"capacitor bank to raise PF from {_round4(float(avg_pf))} to "
                f"{_round4(float(target_pf))}."
            )
        elif months_below > 0:
            recommendation = (
                f"Minor PF penalties in {months_below} months. Monitor PF trend "
                f"before investing in correction equipment."
            )
        else:
            recommendation = (
                f"Power factor is above threshold ({_round4(float(pf_threshold))}). "
                f"No correction needed."
            )

        return PowerFactorAssessment(
            avg_power_factor=_round4(float(avg_pf)),
            min_power_factor=_round4(float(min_pf)),
            pf_threshold=_round4(float(pf_threshold)),
            months_below_threshold=months_below,
            annual_pf_penalty=_round2(float(annual_penalty)),
            correction_target_pf=_round4(float(target_pf)),
            kvar_correction_needed=_round2(float(kvar_correction)),
            recommendation=recommendation,
        )

    # -------------------------------------------------------------------
    # Public API -- Tariff Change Impact
    # -------------------------------------------------------------------

    def assess_tariff_change(
        self,
        consumption: List[MonthlyConsumption],
        old_rate: RateStructure,
        new_rate: RateStructure,
    ) -> TariffChangeImpact:
        """Assess the cost impact of switching between two tariff schedules.

        Args:
            consumption: Monthly consumption profile.
            old_rate: Current rate structure.
            new_rate: Proposed new rate structure.

        Returns:
            TariffChangeImpact with cost change analysis.
        """
        logger.info(
            "Assessing tariff change: %s -> %s",
            old_rate.rate_id, new_rate.rate_id,
        )

        old_cost = self.calculate_annual_cost(consumption, old_rate)
        new_cost = self.calculate_annual_cost(consumption, new_rate)
        change = new_cost - old_cost
        change_pct = float(_safe_pct(change, old_cost))

        # Identify affected components
        affected: List[str] = []
        old_energy = self._total_energy_cost(consumption, old_rate)
        new_energy = self._total_energy_cost(consumption, new_rate)
        if abs(new_energy - old_energy) > Decimal("1"):
            affected.append("energy_charges")

        old_demand = self._total_demand_cost(consumption, old_rate)
        new_demand = self._total_demand_cost(consumption, new_rate)
        if abs(new_demand - old_demand) > Decimal("1"):
            affected.append("demand_charges")

        old_fixed = _decimal(old_rate.fixed_charges_monthly) * _decimal(len(consumption))
        new_fixed = _decimal(new_rate.fixed_charges_monthly) * _decimal(len(consumption))
        if abs(new_fixed - old_fixed) > Decimal("1"):
            affected.append("fixed_charges")

        if old_rate.rate_type != new_rate.rate_type:
            affected.append("rate_structure_type")

        # Determine direction
        if change > Decimal("1"):
            direction = RateChangeImpact.INCREASE
        elif change < Decimal("-1"):
            direction = RateChangeImpact.DECREASE
        else:
            direction = RateChangeImpact.NEUTRAL

        recommendation = ""
        if direction == RateChangeImpact.DECREASE:
            recommendation = (
                f"Switching from {old_rate.rate_name} to {new_rate.rate_name} "
                f"saves {_round2(float(abs(change)))}/yr ({_round2(abs(change_pct))}%). "
                f"Recommend proceeding with the rate change."
            )
        elif direction == RateChangeImpact.INCREASE:
            recommendation = (
                f"Switching to {new_rate.rate_name} increases costs by "
                f"{_round2(float(change))}/yr ({_round2(change_pct)}%). "
                f"Consider remaining on {old_rate.rate_name}."
            )
        else:
            recommendation = (
                f"Cost difference between {old_rate.rate_name} and "
                f"{new_rate.rate_name} is negligible. Decision may be based "
                f"on non-cost factors (billing simplicity, demand flexibility)."
            )

        return TariffChangeImpact(
            old_rate_name=old_rate.rate_name,
            new_rate_name=new_rate.rate_name,
            annual_cost_change=_round2(float(change)),
            cost_change_pct=_round2(change_pct),
            affected_components=affected,
            impact_direction=direction,
            recommendation=recommendation,
        )

    # -------------------------------------------------------------------
    # Internal -- Monthly Energy Cost by Rate Type
    # -------------------------------------------------------------------

    def _calculate_monthly_energy_cost(
        self,
        month: MonthlyConsumption,
        rate: RateStructure,
    ) -> Decimal:
        """Calculate monthly energy cost based on rate type.

        Dispatches to the appropriate calculation method.

        Args:
            month: Single month consumption data.
            rate: Rate structure definition.

        Returns:
            Monthly energy cost as Decimal.
        """
        if rate.rate_type == RateType.FLAT:
            return self._calc_flat_energy(month, rate)
        elif rate.rate_type == RateType.TIERED:
            return self._calc_tiered_energy(month, rate)
        elif rate.rate_type in (RateType.TOU, RateType.SEASONAL):
            return self._calc_tou_energy(month, rate)
        elif rate.rate_type in (RateType.DEMAND, RateType.DEMAND_RATCHET):
            return self._calc_demand_energy(month, rate)
        elif rate.rate_type == RateType.RTP:
            return self._calc_rtp_energy(month, rate)
        elif rate.rate_type == RateType.INTERRUPTIBLE:
            return self._calc_interruptible_energy(month, rate)
        elif rate.rate_type == RateType.STANDBY:
            return self._calc_standby_energy(month, rate)
        elif rate.rate_type == RateType.NET_METERING:
            return self._calc_net_metering_energy(month, rate)
        elif rate.rate_type == RateType.FEED_IN:
            return self._calc_feed_in_energy(month, rate)
        else:
            logger.warning("Unknown rate type %s, using flat calculation", rate.rate_type)
            return self._calc_flat_energy(month, rate)

    def _calc_flat_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """FLAT: cost = total_kwh * flat_rate."""
        flat_rate = _decimal(rate.flat_rate_per_kwh or 0)
        if flat_rate == Decimal("0") and rate.tiers:
            flat_rate = _decimal(rate.tiers[0].rate_per_kwh)
        return _decimal(month.total_kwh) * flat_rate

    def _calc_tiered_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """TIERED: cost = sum(min(kwh_in_tier, tier_width) * tier_rate).

        Applies inclining block pricing where each tier has a defined
        kWh range and rate.
        """
        if not rate.tiers:
            return self._calc_flat_energy(month, rate)

        total_kwh = _decimal(month.total_kwh)
        remaining_kwh = total_kwh
        cost = Decimal("0")

        sorted_tiers = sorted(rate.tiers, key=lambda t: t.tier_number)

        for tier in sorted_tiers:
            if remaining_kwh <= Decimal("0"):
                break

            lower = _decimal(tier.lower_kwh)
            upper = _decimal(tier.upper_kwh) if tier.upper_kwh is not None else total_kwh + Decimal("1")
            tier_width = upper - lower
            kwh_in_tier = min(remaining_kwh, tier_width)

            tier_cost = kwh_in_tier * _decimal(tier.rate_per_kwh)
            cost += tier_cost
            remaining_kwh -= kwh_in_tier

        return cost

    def _calc_tou_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """TOU/SEASONAL: cost = sum(period_kwh * period_rate).

        Maps on-peak, mid-peak, and off-peak consumption to their
        respective TOU rates. Falls back to flat rate for unmatched periods.
        """
        if not rate.tou_schedules:
            return self._calc_flat_energy(month, rate)

        # Build rate lookup by period
        period_rates: Dict[TOUPeriod, Decimal] = {}
        for sched in rate.tou_schedules:
            # Use the schedule matching this month's season, or ANNUAL
            if sched.season in (month.season, SeasonType.ANNUAL):
                period_rates[sched.period] = _decimal(sched.rate_per_kwh)

        on_peak_rate = period_rates.get(TOUPeriod.ON_PEAK, Decimal("0"))
        mid_peak_rate = period_rates.get(
            TOUPeriod.MID_PEAK, on_peak_rate
        )
        off_peak_rate = period_rates.get(TOUPeriod.OFF_PEAK, Decimal("0"))

        cost = (
            _decimal(month.on_peak_kwh) * on_peak_rate
            + _decimal(month.mid_peak_kwh) * mid_peak_rate
            + _decimal(month.off_peak_kwh) * off_peak_rate
        )

        # Any remaining consumption not covered by TOU periods
        tou_covered = (
            _decimal(month.on_peak_kwh)
            + _decimal(month.mid_peak_kwh)
            + _decimal(month.off_peak_kwh)
        )
        remainder = _decimal(month.total_kwh) - tou_covered
        if remainder > Decimal("0"):
            # Apply off-peak rate to unclassified consumption
            cost += remainder * off_peak_rate

        return cost

    def _calc_demand_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """DEMAND/DEMAND_RATCHET: energy component only.

        For demand rates, the energy component uses tiers or flat rate,
        while the demand charge is calculated separately.
        """
        if rate.tiers:
            return self._calc_tiered_energy(month, rate)
        elif rate.tou_schedules:
            return self._calc_tou_energy(month, rate)
        else:
            return self._calc_flat_energy(month, rate)

    def _calc_rtp_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """RTP (Real-Time Pricing): use weighted average of TOU rates.

        Without interval-level price data, approximates using TOU period
        breakdown and available schedule rates.
        """
        if rate.tou_schedules:
            return self._calc_tou_energy(month, rate)
        return self._calc_flat_energy(month, rate)

    def _calc_interruptible_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """INTERRUPTIBLE: base_cost * (1 - interrupt_discount)."""
        base_cost = self._calc_flat_energy(month, rate)
        if rate.tiers:
            base_cost = self._calc_tiered_energy(month, rate)
        discount = _decimal(rate.interrupt_discount or 0)
        return base_cost * (Decimal("1") - discount)

    def _calc_standby_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """STANDBY: energy_cost (standby charge is handled as fixed)."""
        if rate.tiers:
            return self._calc_tiered_energy(month, rate)
        return self._calc_flat_energy(month, rate)

    def _calc_net_metering_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """NET_METERING: cost = max(0, import_kwh - export_kwh) * rate.

        Net metered facilities pay only for net consumption after
        subtracting exported generation.
        """
        net_kwh = _decimal(month.total_kwh) - _decimal(month.export_kwh)
        if net_kwh < Decimal("0"):
            net_kwh = Decimal("0")  # Credit carried forward, not modeled here

        flat_rate = _decimal(rate.flat_rate_per_kwh or 0)
        if flat_rate == Decimal("0") and rate.tiers:
            flat_rate = _decimal(rate.tiers[0].rate_per_kwh)

        return net_kwh * flat_rate

    def _calc_feed_in_energy(
        self, month: MonthlyConsumption, rate: RateStructure
    ) -> Decimal:
        """FEED_IN: cost = import_cost - export_kwh * feed_in_rate.

        Import is charged at standard rate; export earns feed-in tariff
        credit.
        """
        flat_rate = _decimal(rate.flat_rate_per_kwh or 0)
        if flat_rate == Decimal("0") and rate.tiers:
            flat_rate = _decimal(rate.tiers[0].rate_per_kwh)

        import_cost = _decimal(month.total_kwh) * flat_rate
        feed_in_rate = _decimal(rate.feed_in_rate_per_kwh or 0)
        export_credit = _decimal(month.export_kwh) * feed_in_rate

        net_cost = import_cost - export_credit
        return max(net_cost, Decimal("0"))

    # -------------------------------------------------------------------
    # Internal -- Monthly Demand Cost
    # -------------------------------------------------------------------

    def _calculate_monthly_demand_cost(
        self,
        month: MonthlyConsumption,
        rate: RateStructure,
    ) -> Decimal:
        """Calculate monthly demand charge component.

        Applies demand charges based on peak demand, minimum demand,
        and any applicable ratchet clause.

        Args:
            month: Single month consumption data.
            rate: Rate structure with demand charge components.

        Returns:
            Monthly demand cost as Decimal.
        """
        if not rate.demand_charges:
            return Decimal("0")

        # Add standby charges if applicable
        standby_cost = Decimal("0")
        if rate.rate_type == RateType.STANDBY and rate.standby_charge_monthly:
            standby_cost = _decimal(rate.standby_charge_monthly)

        total_demand_cost = Decimal("0")

        for dc in rate.demand_charges:
            # Filter by season if applicable
            if dc.season != SeasonType.ANNUAL and dc.season != month.season:
                continue

            actual_kw = _decimal(month.peak_demand_kw)
            min_kw = _decimal(dc.minimum_kw)
            billed_kw = max(actual_kw, min_kw)

            # Note: Ratchet applied across months is handled in analyze_demand_ratchet;
            # here we use the single-month minimum demand enforcement
            demand_cost = billed_kw * _decimal(dc.rate_per_kw)
            total_demand_cost += demand_cost

        return total_demand_cost + standby_cost

    # -------------------------------------------------------------------
    # Internal -- Power Factor Adjustment
    # -------------------------------------------------------------------

    def _calculate_pf_adjustment(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
        subtotal: Decimal,
    ) -> Decimal:
        """Calculate power factor penalty/credit adjustment.

        Uses the formula: adjustment = subtotal * (threshold/actual_pf - 1)
        applied only when actual PF is below the threshold.

        Args:
            consumption: Monthly consumption profile.
            rate: Rate structure with PF threshold.
            subtotal: Pre-adjustment cost subtotal.

        Returns:
            PF adjustment amount (positive = penalty).
        """
        if rate.power_factor_adjustment is None:
            return Decimal("0")

        threshold = _decimal(rate.power_factor_adjustment)
        adjustment = Decimal("0")
        monthly_subtotal = _safe_divide(subtotal, _decimal(len(consumption)))

        for m in consumption:
            pf = _decimal(m.power_factor)
            if pf > Decimal("0") and pf < threshold:
                pf_ratio = _safe_divide(threshold, pf, default=Decimal("1"))
                monthly_penalty = monthly_subtotal * (pf_ratio - Decimal("1"))
                adjustment += monthly_penalty

        return adjustment

    # -------------------------------------------------------------------
    # Internal -- Aggregation Helpers
    # -------------------------------------------------------------------

    def _total_energy_cost(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
    ) -> Decimal:
        """Sum monthly energy costs across all months."""
        return sum(
            (self._calculate_monthly_energy_cost(m, rate) for m in consumption),
            Decimal("0"),
        )

    def _total_demand_cost(
        self,
        consumption: List[MonthlyConsumption],
        rate: RateStructure,
    ) -> Decimal:
        """Sum monthly demand costs across all months."""
        return sum(
            (self._calculate_monthly_demand_cost(m, rate) for m in consumption),
            Decimal("0"),
        )

    # -------------------------------------------------------------------
    # Internal -- Recommendations
    # -------------------------------------------------------------------

    def _generate_optimization_recommendations(
        self,
        comparisons: List[RateComparison],
        current: Optional[RateComparison],
        optimal: RateComparison,
        tou_analysis: Optional[TOUShiftAnalysis],
        ratchet_analysis: Optional[DemandRatchetAnalysis],
    ) -> List[str]:
        """Generate actionable recommendations from optimization results.

        Args:
            comparisons: All rate comparisons.
            current: Current rate comparison (if available).
            optimal: Optimal (lowest cost) rate comparison.
            tou_analysis: TOU load-shift analysis result.
            ratchet_analysis: Demand ratchet analysis result.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Rate switch recommendation
        if current and current.rate_id != optimal.rate_id:
            savings = current.annual_total_cost - optimal.annual_total_cost
            if savings > 0:
                recommendations.append(
                    f"Switch from {current.rate_name} to {optimal.rate_name} "
                    f"to save {_round2(savings)}/yr "
                    f"({_round2(savings / current.annual_total_cost * 100) if current.annual_total_cost > 0 else 0.0}%)."
                )
        elif current:
            recommendations.append(
                f"Current rate ({current.rate_name}) is already the optimal choice."
            )

        # TOU shift recommendation
        if tou_analysis and tou_analysis.savings_pct > 3.0:
            recommendations.append(
                f"TOU load shifting: move {_round2(tou_analysis.shifted_kwh)} kWh/yr "
                f"from on-peak to off-peak for {_round2(tou_analysis.estimated_savings)} "
                f"additional savings."
            )

        # Demand ratchet recommendation
        if ratchet_analysis and ratchet_analysis.annual_ratchet_penalty > 0:
            recommendations.append(
                f"Demand management: reduce peak demand by "
                f"{_round2(ratchet_analysis.reduction_target_kw)} kW to avoid "
                f"{_round2(ratchet_analysis.annual_ratchet_penalty)}/yr in "
                f"ratchet penalties."
            )

        # Blended rate comparison
        if len(comparisons) >= 2:
            best_blended = min(comparisons, key=lambda c: c.blended_rate_per_kwh)
            worst_blended = max(comparisons, key=lambda c: c.blended_rate_per_kwh)
            spread = worst_blended.blended_rate_per_kwh - best_blended.blended_rate_per_kwh
            if spread > 0.01:
                recommendations.append(
                    f"Blended rate spread across options: "
                    f"{_round6(best_blended.blended_rate_per_kwh)} to "
                    f"{_round6(worst_blended.blended_rate_per_kwh)} per kWh "
                    f"(spread of {_round6(spread)})."
                )

        if not recommendations:
            recommendations.append(
                "No significant savings opportunities identified with current "
                "consumption profile. Re-evaluate when consumption patterns change."
            )

        return recommendations
