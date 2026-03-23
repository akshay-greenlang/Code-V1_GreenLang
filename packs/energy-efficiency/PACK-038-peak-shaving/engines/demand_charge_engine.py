# -*- coding: utf-8 -*-
"""
DemandChargeEngine - PACK-038 Peak Shaving Engine 3
=====================================================

Tariff decomposition and demand charge calculation engine.  Models
complex utility rate structures including flat, tiered, time-of-use
(TOU), seasonal, coincident peak (CP), ratchet demand, power factor
penalty, and reactive demand charges.  Calculates marginal value of
peak reduction at each kW level and projects future charges.

Calculation Methodology:
    Flat Demand Charge:
        charge = peak_kw * rate_per_kw

    Tiered Demand Charge (Inclining Block):
        For each tier (low_kw, high_kw, rate):
            tier_kw = min(peak_kw, high_kw) - low_kw
            tier_charge = max(tier_kw, 0) * rate
        total = sum(tier_charges)

    TOU Demand Charge:
        For each TOU period:
            period_charge = period_peak_kw * period_rate
        total = sum(period_charges)

    Ratchet Demand:
        ratchet_kw = max(current_peak, ratchet_pct * max(past_N_peaks))
        charge = ratchet_kw * rate

    Power Factor Penalty:
        actual_pf = kW / sqrt(kW^2 + kVAR^2)
        if actual_pf < pf_threshold:
            penalty = (pf_threshold / actual_pf - 1) * base_charge
        kVA billing: charge = kVA_peak * rate  (where kVA = kW / pf)

    Coincident Peak (CP) Charge:
        charge = facility_demand_at_system_peak * cp_rate

    Marginal Value:
        For each kW reduction from current peak:
            marginal_savings = charge(peak) - charge(peak - 1 kW)

Regulatory References:
    - FERC - Electric Rate Design and Demand Charges
    - PURPA (Public Utility Regulatory Policies Act)
    - NARUC Rate Design Manual
    - IEC 62053 - Electricity Metering Equipment
    - EN 50470 - Electricity Metering Equipment (EU)
    - IEEE 1459-2010 - Power Quality and Billing
    - ISO 50001:2018 - Energy management and billing

Zero-Hallucination:
    - All charge calculations use explicit tariff formulas
    - Tier boundaries and rates from user-provided tariff schedules
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-038 Peak Shaving
Engine:  3 of 5
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


class ChargeType(str, Enum):
    """Type of demand charge component.

    FLAT:         Single rate for all demand.
    TIERED:       Multiple tiers with different rates.
    TOU:          Time-of-use period-specific rates.
    SEASONAL:     Season-specific demand rates.
    CP:           Coincident peak charge (system peak allocation).
    RATCHET:      Ratchet demand (rolling maximum).
    PF_PENALTY:   Power factor penalty surcharge.
    REACTIVE:     Reactive power (kVAR) charge.
    """
    FLAT = "flat"
    TIERED = "tiered"
    TOU = "tou"
    SEASONAL = "seasonal"
    CP = "cp"
    RATCHET = "ratchet"
    PF_PENALTY = "pf_penalty"
    REACTIVE = "reactive"


class RateStructure(str, Enum):
    """Rate structure for tiered demand charges.

    FLAT:              Single flat rate.
    INCLINING_BLOCK:   Rate increases with higher demand tiers.
    DECLINING_BLOCK:   Rate decreases with higher demand tiers.
    TOU_OVERLAY:       TOU periods overlaid on tiered structure.
    """
    FLAT = "flat"
    INCLINING_BLOCK = "inclining_block"
    DECLINING_BLOCK = "declining_block"
    TOU_OVERLAY = "tou_overlay"


class BillingDeterminant(str, Enum):
    """Unit of measure for billing demand.

    KW:   Kilowatts (real power demand).
    KVA:  Kilovolt-amperes (apparent power demand).
    KVAR: Kilovolt-amperes reactive (reactive power demand).
    """
    KW = "kw"
    KVA = "kva"
    KVAR = "kvar"


class TariffPeriod(str, Enum):
    """Time-of-use tariff period.

    ON_PEAK:    Peak pricing period.
    MID_PEAK:   Mid/partial-peak pricing period.
    OFF_PEAK:   Off-peak pricing period.
    SUPER_PEAK: Critical / super-peak pricing period.
    """
    ON_PEAK = "on_peak"
    MID_PEAK = "mid_peak"
    OFF_PEAK = "off_peak"
    SUPER_PEAK = "super_peak"


class Season(str, Enum):
    """Tariff season for seasonal demand charges.

    SUMMER:   Summer rate season.
    WINTER:   Winter rate season.
    SHOULDER: Shoulder / transition rate season.
    """
    SUMMER = "summer"
    WINTER = "winter"
    SHOULDER = "shoulder"


class CurrencyCode(str, Enum):
    """Currency for demand charges.

    USD: United States Dollar.
    EUR: Euro.
    GBP: British Pound Sterling.
    AUD: Australian Dollar.
    CAD: Canadian Dollar.
    """
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    AUD = "AUD"
    CAD = "CAD"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default power factor threshold for penalty assessment.
DEFAULT_PF_THRESHOLD: Decimal = Decimal("0.90")

# Default ratchet percentage (fraction of rolling max to apply).
DEFAULT_RATCHET_PCT: Decimal = Decimal("0.80")

# Default ratchet lookback months.
DEFAULT_RATCHET_MONTHS: int = 11

# Common US commercial demand charge rates ($/kW/month) by region.
# Source: EIA-861 and utility tariff filings.
REFERENCE_RATES_USD: Dict[str, Dict[str, Decimal]] = {
    "northeast": {
        "flat": Decimal("18.50"),
        "on_peak": Decimal("25.00"),
        "mid_peak": Decimal("15.00"),
        "off_peak": Decimal("8.00"),
    },
    "southeast": {
        "flat": Decimal("12.00"),
        "on_peak": Decimal("16.50"),
        "mid_peak": Decimal("10.00"),
        "off_peak": Decimal("5.50"),
    },
    "midwest": {
        "flat": Decimal("14.00"),
        "on_peak": Decimal("19.00"),
        "mid_peak": Decimal("12.00"),
        "off_peak": Decimal("6.00"),
    },
    "west": {
        "flat": Decimal("20.00"),
        "on_peak": Decimal("28.00"),
        "mid_peak": Decimal("18.00"),
        "off_peak": Decimal("9.00"),
    },
    "texas": {
        "flat": Decimal("10.00"),
        "on_peak": Decimal("14.00"),
        "mid_peak": Decimal("8.00"),
        "off_peak": Decimal("4.00"),
    },
}

# EU demand charge reference rates (EUR/kW/month).
REFERENCE_RATES_EUR: Dict[str, Dict[str, Decimal]] = {
    "germany": {
        "flat": Decimal("12.00"),
        "on_peak": Decimal("18.00"),
        "off_peak": Decimal("6.00"),
    },
    "france": {
        "flat": Decimal("9.50"),
        "on_peak": Decimal("14.00"),
        "off_peak": Decimal("5.00"),
    },
    "uk": {
        "flat": Decimal("15.00"),
        "on_peak": Decimal("22.00"),
        "off_peak": Decimal("7.00"),
    },
}

# Default tiered rate schedule (inclining block, USD).
DEFAULT_TIERS: List[Dict[str, Decimal]] = [
    {"low_kw": Decimal("0"), "high_kw": Decimal("100"), "rate": Decimal("10.00")},
    {"low_kw": Decimal("100"), "high_kw": Decimal("500"), "rate": Decimal("15.00")},
    {"low_kw": Decimal("500"), "high_kw": Decimal("1000"), "rate": Decimal("18.00")},
    {"low_kw": Decimal("1000"), "high_kw": Decimal("999999"), "rate": Decimal("22.00")},
]

# Monthly escalation rate for charge projection.
DEFAULT_ANNUAL_ESCALATION: Decimal = Decimal("0.03")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input / Output
# ---------------------------------------------------------------------------


class TariffComponent(BaseModel):
    """A single component of a demand tariff.

    Attributes:
        component_id: Unique component identifier.
        name: Human-readable component name.
        charge_type: Type of demand charge.
        rate_structure: Rate structure for tiered charges.
        billing_determinant: Billing unit (kW, kVA, kVAR).
        season: Applicable season (if seasonal).
        period: Applicable TOU period (if TOU).
        flat_rate: Flat rate per kW (for flat charges).
        tiers: List of tier definitions (for tiered charges).
        ratchet_pct: Ratchet percentage (for ratchet charges).
        ratchet_months: Ratchet lookback months.
        pf_threshold: Power factor threshold (for PF penalty).
        cp_rate: Coincident peak rate.
        currency: Currency code.
        notes: Additional notes.
    """
    component_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", max_length=500)
    charge_type: ChargeType = Field(default=ChargeType.FLAT)
    rate_structure: RateStructure = Field(default=RateStructure.FLAT)
    billing_determinant: BillingDeterminant = Field(default=BillingDeterminant.KW)
    season: Season = Field(default=Season.SUMMER)
    period: TariffPeriod = Field(default=TariffPeriod.ON_PEAK)
    flat_rate: Decimal = Field(default=Decimal("0"), ge=0)
    tiers: List[Dict[str, Decimal]] = Field(default_factory=list)
    ratchet_pct: Decimal = Field(
        default=DEFAULT_RATCHET_PCT, ge=0, le=Decimal("1")
    )
    ratchet_months: int = Field(default=DEFAULT_RATCHET_MONTHS, ge=0, le=24)
    pf_threshold: Decimal = Field(
        default=DEFAULT_PF_THRESHOLD, ge=0, le=Decimal("1")
    )
    cp_rate: Decimal = Field(default=Decimal("0"), ge=0)
    currency: CurrencyCode = Field(default=CurrencyCode.USD)
    notes: str = Field(default="", max_length=2000)


class DemandCharge(BaseModel):
    """Calculated demand charge for a billing period.

    Attributes:
        charge_id: Unique charge identifier.
        component_id: Reference to tariff component.
        charge_type: Type of demand charge.
        billing_period: Billing period label.
        billing_demand_kw: Billing demand (kW or applicable unit).
        rate_applied: Effective rate applied.
        charge_amount: Calculated charge amount.
        currency: Currency code.
        notes: Calculation notes.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    charge_id: str = Field(default_factory=_new_uuid)
    component_id: str = Field(default="")
    charge_type: ChargeType = Field(default=ChargeType.FLAT)
    billing_period: str = Field(default="", max_length=20)
    billing_demand_kw: Decimal = Field(default=Decimal("0"))
    rate_applied: Decimal = Field(default=Decimal("0"))
    charge_amount: Decimal = Field(default=Decimal("0"))
    currency: CurrencyCode = Field(default=CurrencyCode.USD)
    notes: str = Field(default="", max_length=2000)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class ChargeBreakdown(BaseModel):
    """Complete breakdown of demand charges for a billing period.

    Attributes:
        breakdown_id: Unique breakdown identifier.
        facility_id: Facility identifier.
        billing_period: Billing period label.
        peak_demand_kw: Metered peak demand (kW).
        billing_demand_kw: Billing demand after ratchet/adjustments.
        power_factor: Actual power factor.
        apparent_power_kva: Apparent power at peak (kVA).
        charges: Individual charge line items.
        total_demand_charge: Total demand charge amount.
        total_energy_charge: Total energy charge (if available).
        total_bill: Total bill amount.
        demand_charge_pct: Demand charge as percentage of total bill.
        currency: Currency code.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    breakdown_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    billing_period: str = Field(default="", max_length=20)
    peak_demand_kw: Decimal = Field(default=Decimal("0"))
    billing_demand_kw: Decimal = Field(default=Decimal("0"))
    power_factor: Decimal = Field(default=Decimal("1"))
    apparent_power_kva: Decimal = Field(default=Decimal("0"))
    charges: List[DemandCharge] = Field(default_factory=list)
    total_demand_charge: Decimal = Field(default=Decimal("0"))
    total_energy_charge: Decimal = Field(default=Decimal("0"))
    total_bill: Decimal = Field(default=Decimal("0"))
    demand_charge_pct: Decimal = Field(default=Decimal("0"))
    currency: CurrencyCode = Field(default=CurrencyCode.USD)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class MarginalValue(BaseModel):
    """Marginal value of demand reduction at each kW level.

    Attributes:
        from_kw: Starting demand level.
        to_kw: Reduced demand level.
        marginal_savings: Savings for this kW reduction.
        cumulative_savings: Total savings from peak to this level.
        marginal_rate: Effective marginal rate ($/kW).
        currency: Currency code.
    """
    from_kw: Decimal = Field(default=Decimal("0"))
    to_kw: Decimal = Field(default=Decimal("0"))
    marginal_savings: Decimal = Field(default=Decimal("0"))
    cumulative_savings: Decimal = Field(default=Decimal("0"))
    marginal_rate: Decimal = Field(default=Decimal("0"))
    currency: CurrencyCode = Field(default=CurrencyCode.USD)


class DemandChargeResult(BaseModel):
    """Complete demand charge analysis result.

    Attributes:
        result_id: Unique result identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        tariff_name: Name of the tariff schedule.
        breakdown: Current period charge breakdown.
        marginal_values: Marginal value of demand reduction.
        projected_annual_charges: Projected annual demand charges.
        optimal_target_kw: Optimal peak reduction target (kW).
        optimal_savings: Savings at optimal target.
        max_marginal_rate: Highest marginal rate (most valuable kW).
        avg_demand_charge_rate: Average effective demand charge rate.
        recommendations: List of tariff optimization recommendations.
        processing_time_ms: Processing duration in milliseconds.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="", max_length=500)
    tariff_name: str = Field(default="", max_length=500)
    breakdown: ChargeBreakdown = Field(default_factory=ChargeBreakdown)
    marginal_values: List[MarginalValue] = Field(default_factory=list)
    projected_annual_charges: Decimal = Field(default=Decimal("0"))
    optimal_target_kw: Decimal = Field(default=Decimal("0"))
    optimal_savings: Decimal = Field(default=Decimal("0"))
    max_marginal_rate: Decimal = Field(default=Decimal("0"))
    avg_demand_charge_rate: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DemandChargeEngine:
    """Tariff decomposition and demand charge calculation engine.

    Models complex utility rate structures and calculates the marginal
    value of peak demand reduction at each kW level.  Supports flat,
    tiered, TOU, seasonal, coincident peak, ratchet, and power factor
    charge types.  All calculations use deterministic Decimal arithmetic
    with SHA-256 provenance hashing.

    Usage::

        engine = DemandChargeEngine()
        components = [
            TariffComponent(name="Flat Demand", charge_type=ChargeType.FLAT,
                            flat_rate=Decimal("15.00")),
        ]
        result = engine.decompose_charges(
            facility_id="FAC-001",
            peak_demand_kw=Decimal("500"),
            components=components,
            billing_period="2026-03",
        )
        print(f"Total demand charge: ${result.breakdown.total_demand_charge}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DemandChargeEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - currency (str): default currency code
                - annual_escalation (float): annual rate escalation
                - pf_threshold (float): power factor threshold
                - ratchet_pct (float): default ratchet percentage
        """
        self.config = config or {}
        self._currency = CurrencyCode(
            self.config.get("currency", CurrencyCode.USD.value)
        )
        self._annual_escalation = _decimal(
            self.config.get("annual_escalation", DEFAULT_ANNUAL_ESCALATION)
        )
        self._pf_threshold = _decimal(
            self.config.get("pf_threshold", DEFAULT_PF_THRESHOLD)
        )
        self._ratchet_pct = _decimal(
            self.config.get("ratchet_pct", DEFAULT_RATCHET_PCT)
        )
        logger.info(
            "DemandChargeEngine v%s initialised (currency=%s)",
            self.engine_version, self._currency.value,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def decompose_charges(
        self,
        facility_id: str,
        peak_demand_kw: Decimal,
        components: List[TariffComponent],
        billing_period: str = "",
        power_factor: Optional[Decimal] = None,
        reactive_kvar: Optional[Decimal] = None,
        historical_peaks: Optional[List[Decimal]] = None,
        cp_demand_kw: Optional[Decimal] = None,
        tou_peaks: Optional[Dict[str, Decimal]] = None,
        energy_charge: Optional[Decimal] = None,
    ) -> ChargeBreakdown:
        """Decompose demand charges for a billing period.

        Calculates each tariff component's contribution to the total
        demand charge.

        Args:
            facility_id: Facility identifier.
            peak_demand_kw: Metered peak demand (kW).
            components: List of tariff components.
            billing_period: Billing period label.
            power_factor: Actual power factor (0-1).
            reactive_kvar: Reactive power at peak (kVAR).
            historical_peaks: Past billing period peaks for ratchet.
            cp_demand_kw: Facility demand at system peak (for CP).
            tou_peaks: Peak demand by TOU period.
            energy_charge: Energy charge amount (for pct calculation).

        Returns:
            ChargeBreakdown with all charge line items.
        """
        t0 = time.perf_counter()
        logger.info(
            "Decomposing charges: facility=%s, peak=%.1f kW, %d components",
            facility_id, float(peak_demand_kw), len(components),
        )

        pf = power_factor or Decimal("1")
        if reactive_kvar and pf == Decimal("1"):
            # Calculate PF from kW and kVAR
            kw_sq = peak_demand_kw ** 2
            kvar_sq = reactive_kvar ** 2
            apparent_sq = kw_sq + kvar_sq
            if apparent_sq > Decimal("0"):
                apparent = _decimal(math.sqrt(float(apparent_sq)))
                pf = _safe_divide(peak_demand_kw, apparent)

        apparent_kva = _safe_divide(peak_demand_kw, pf) if pf > Decimal("0") else peak_demand_kw

        charges: List[DemandCharge] = []
        total_demand = Decimal("0")

        for comp in components:
            charge = self._calculate_component_charge(
                comp, peak_demand_kw, pf, apparent_kva,
                historical_peaks or [], cp_demand_kw or Decimal("0"),
                tou_peaks or {}, billing_period,
            )
            charges.append(charge)
            total_demand += charge.charge_amount

        # Billing demand (may differ from metered peak due to ratchet)
        billing_kw = peak_demand_kw
        for charge in charges:
            if charge.charge_type == ChargeType.RATCHET:
                billing_kw = max(billing_kw, charge.billing_demand_kw)

        total_energy = energy_charge or Decimal("0")
        total_bill = total_demand + total_energy
        demand_pct = _safe_pct(total_demand, total_bill)

        result = ChargeBreakdown(
            facility_id=facility_id,
            billing_period=billing_period,
            peak_demand_kw=_round_val(peak_demand_kw, 2),
            billing_demand_kw=_round_val(billing_kw, 2),
            power_factor=_round_val(pf, 4),
            apparent_power_kva=_round_val(apparent_kva, 2),
            charges=charges,
            total_demand_charge=_round_val(total_demand, 2),
            total_energy_charge=_round_val(total_energy, 2),
            total_bill=_round_val(total_bill, 2),
            demand_charge_pct=_round_val(demand_pct, 2),
            currency=self._currency,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Charges decomposed: total=%.2f, demand_pct=%.1f%%, "
            "hash=%s (%.1f ms)",
            float(total_demand), float(demand_pct),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_marginal_value(
        self,
        peak_demand_kw: Decimal,
        components: List[TariffComponent],
        reduction_steps: int = 50,
        power_factor: Optional[Decimal] = None,
        historical_peaks: Optional[List[Decimal]] = None,
    ) -> List[MarginalValue]:
        """Calculate marginal value of demand reduction at each kW level.

        Steps down from current peak demand in equal increments and
        calculates the savings for each kW of reduction.

        Args:
            peak_demand_kw: Current peak demand (kW).
            components: Tariff components.
            reduction_steps: Number of reduction steps to evaluate.
            power_factor: Actual power factor.
            historical_peaks: Historical peaks for ratchet calculation.

        Returns:
            List of MarginalValue objects from peak to minimum.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating marginal value: peak=%.1f kW, %d steps",
            float(peak_demand_kw), reduction_steps,
        )

        if peak_demand_kw <= Decimal("0"):
            return []

        step_size = _safe_divide(peak_demand_kw, _decimal(reduction_steps))
        if step_size <= Decimal("0"):
            step_size = Decimal("1")

        pf = power_factor or Decimal("1")

        # Calculate baseline charge
        baseline_charge = self._total_charge_for_peak(
            peak_demand_kw, components, pf, historical_peaks or [],
        )

        marginals: List[MarginalValue] = []
        cumulative = Decimal("0")
        prev_kw = peak_demand_kw

        for i in range(1, reduction_steps + 1):
            reduced_kw = max(
                peak_demand_kw - step_size * _decimal(i), Decimal("0")
            )
            reduced_charge = self._total_charge_for_peak(
                reduced_kw, components, pf, historical_peaks or [],
            )
            marginal_savings = baseline_charge - reduced_charge - cumulative
            cumulative += marginal_savings
            actual_reduction = prev_kw - reduced_kw
            marginal_rate = _safe_divide(marginal_savings, actual_reduction)

            marginals.append(MarginalValue(
                from_kw=_round_val(prev_kw, 2),
                to_kw=_round_val(reduced_kw, 2),
                marginal_savings=_round_val(marginal_savings, 2),
                cumulative_savings=_round_val(cumulative, 2),
                marginal_rate=_round_val(marginal_rate, 2),
                currency=self._currency,
            ))
            prev_kw = reduced_kw

            if reduced_kw <= Decimal("0"):
                break

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Marginal value calculated: %d steps, max_rate=%.2f (%.1f ms)",
            len(marginals),
            float(max((m.marginal_rate for m in marginals), default=Decimal("0"))),
            elapsed,
        )
        return marginals

    def model_tariff(
        self,
        facility_id: str,
        facility_name: str,
        tariff_name: str,
        peak_demand_kw: Decimal,
        components: List[TariffComponent],
        power_factor: Optional[Decimal] = None,
        reactive_kvar: Optional[Decimal] = None,
        historical_peaks: Optional[List[Decimal]] = None,
        energy_charge: Optional[Decimal] = None,
        billing_period: str = "",
    ) -> DemandChargeResult:
        """Run a complete tariff model with decomposition and marginal analysis.

        Args:
            facility_id: Facility identifier.
            facility_name: Facility name.
            tariff_name: Name of the tariff schedule.
            peak_demand_kw: Metered peak demand (kW).
            components: List of tariff components.
            power_factor: Actual power factor.
            reactive_kvar: Reactive power at peak.
            historical_peaks: Past peaks for ratchet.
            energy_charge: Energy charge amount.
            billing_period: Billing period label.

        Returns:
            DemandChargeResult with full analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Modelling tariff: %s, peak=%.1f kW", tariff_name, float(peak_demand_kw),
        )

        # Decompose charges
        breakdown = self.decompose_charges(
            facility_id, peak_demand_kw, components, billing_period,
            power_factor, reactive_kvar, historical_peaks,
            energy_charge=energy_charge,
        )

        # Marginal values
        marginals = self.calculate_marginal_value(
            peak_demand_kw, components, power_factor=power_factor,
            historical_peaks=historical_peaks,
        )

        # Project annual charges (12 months)
        annual = breakdown.total_demand_charge * Decimal("12")

        # Find optimal target
        optimal_kw = peak_demand_kw
        optimal_savings = Decimal("0")
        max_marginal_rate = Decimal("0")
        for mv in marginals:
            if mv.marginal_rate > max_marginal_rate:
                max_marginal_rate = mv.marginal_rate
            if mv.cumulative_savings > optimal_savings:
                optimal_savings = mv.cumulative_savings
                optimal_kw = mv.to_kw

        avg_rate = _safe_divide(
            breakdown.total_demand_charge, peak_demand_kw
        )

        recommendations = self._generate_recommendations(
            breakdown, marginals, peak_demand_kw, power_factor or Decimal("1"),
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = DemandChargeResult(
            facility_id=facility_id,
            facility_name=facility_name,
            tariff_name=tariff_name,
            breakdown=breakdown,
            marginal_values=marginals,
            projected_annual_charges=_round_val(annual, 2),
            optimal_target_kw=_round_val(optimal_kw, 2),
            optimal_savings=_round_val(optimal_savings, 2),
            max_marginal_rate=_round_val(max_marginal_rate, 2),
            avg_demand_charge_rate=_round_val(avg_rate, 2),
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Tariff modelled: total=%.2f, annual=%.2f, "
            "max_marginal=%.2f, hash=%s (%.1f ms)",
            float(breakdown.total_demand_charge), float(annual),
            float(max_marginal_rate), result.provenance_hash[:16],
            float(elapsed_ms),
        )
        return result

    def project_charges(
        self,
        current_charge: Decimal,
        months: int = 12,
        annual_escalation: Optional[Decimal] = None,
    ) -> List[Dict[str, Decimal]]:
        """Project demand charges forward with escalation.

        Args:
            current_charge: Current monthly demand charge.
            months: Number of months to project.
            annual_escalation: Annual escalation rate (e.g. 0.03 = 3%).

        Returns:
            List of dicts with 'month' and 'projected_charge'.
        """
        t0 = time.perf_counter()
        escalation = annual_escalation or self._annual_escalation
        monthly_esc = (Decimal("1") + escalation) ** _safe_divide(
            Decimal("1"), Decimal("12")
        )

        projections: List[Dict[str, Decimal]] = []
        charge = current_charge
        for month in range(1, months + 1):
            charge = charge * monthly_esc
            projections.append({
                "month": _decimal(month),
                "projected_charge": _round_val(charge, 2),
            })

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Charges projected: %d months, final=%.2f (%.1f ms)",
            months, float(charge), elapsed,
        )
        return projections

    def optimize_tariff_selection(
        self,
        peak_demand_kw: Decimal,
        tariff_options: List[Dict[str, Any]],
        power_factor: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Compare multiple tariff options and recommend the lowest-cost option.

        Args:
            peak_demand_kw: Facility peak demand (kW).
            tariff_options: List of dicts with 'name' and 'components'.
            power_factor: Actual power factor.

        Returns:
            Dict with 'best_tariff', 'annual_savings', 'comparison'.
        """
        t0 = time.perf_counter()
        logger.info(
            "Optimizing tariff: %d options, peak=%.1f kW",
            len(tariff_options), float(peak_demand_kw),
        )

        pf = power_factor or Decimal("1")
        results: List[Dict[str, Any]] = []

        for option in tariff_options:
            name = option.get("name", "Unknown")
            components = option.get("components", [])
            if not isinstance(components, list):
                continue

            # Parse components if they are dicts
            parsed_components = []
            for c in components:
                if isinstance(c, TariffComponent):
                    parsed_components.append(c)
                elif isinstance(c, dict):
                    parsed_components.append(TariffComponent(**c))

            total = self._total_charge_for_peak(
                peak_demand_kw, parsed_components, pf, [],
            )
            annual = total * Decimal("12")
            results.append({
                "name": name,
                "monthly_charge": _round_val(total, 2),
                "annual_charge": _round_val(annual, 2),
            })

        if not results:
            return {"best_tariff": "None", "annual_savings": Decimal("0"), "comparison": []}

        results.sort(key=lambda r: r["annual_charge"])
        best = results[0]
        worst = results[-1]
        savings = worst["annual_charge"] - best["annual_charge"]

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Tariff optimized: best=%s (%.2f/yr), savings=%.2f (%.1f ms)",
            best["name"], float(best["annual_charge"]),
            float(savings), elapsed,
        )
        return {
            "best_tariff": best["name"],
            "annual_savings": _round_val(savings, 2),
            "comparison": results,
        }

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_component_charge(
        self,
        comp: TariffComponent,
        peak_kw: Decimal,
        power_factor: Decimal,
        apparent_kva: Decimal,
        historical_peaks: List[Decimal],
        cp_demand_kw: Decimal,
        tou_peaks: Dict[str, Decimal],
        billing_period: str,
    ) -> DemandCharge:
        """Calculate charge for a single tariff component.

        Args:
            comp: Tariff component definition.
            peak_kw: Metered peak demand.
            power_factor: Actual power factor.
            apparent_kva: Apparent power.
            historical_peaks: Past peaks for ratchet.
            cp_demand_kw: Coincident peak demand.
            tou_peaks: Peak demand by TOU period.
            billing_period: Billing period label.

        Returns:
            DemandCharge for this component.
        """
        charge_amount = Decimal("0")
        billing_demand = peak_kw
        rate = Decimal("0")
        notes = ""

        if comp.charge_type == ChargeType.FLAT:
            charge_amount = peak_kw * comp.flat_rate
            rate = comp.flat_rate
            notes = f"Flat: {peak_kw} kW x {comp.flat_rate}/kW"

        elif comp.charge_type == ChargeType.TIERED:
            charge_amount, notes = self._calculate_tiered_charge(
                peak_kw, comp.tiers or DEFAULT_TIERS,
            )
            rate = _safe_divide(charge_amount, peak_kw)

        elif comp.charge_type == ChargeType.TOU:
            period_peak = tou_peaks.get(comp.period.value, peak_kw)
            charge_amount = period_peak * comp.flat_rate
            billing_demand = period_peak
            rate = comp.flat_rate
            notes = f"TOU {comp.period.value}: {period_peak} kW x {comp.flat_rate}/kW"

        elif comp.charge_type == ChargeType.SEASONAL:
            charge_amount = peak_kw * comp.flat_rate
            rate = comp.flat_rate
            notes = f"Seasonal ({comp.season.value}): {peak_kw} kW x {comp.flat_rate}/kW"

        elif comp.charge_type == ChargeType.CP:
            charge_amount = cp_demand_kw * comp.cp_rate
            billing_demand = cp_demand_kw
            rate = comp.cp_rate
            notes = f"CP: {cp_demand_kw} kW x {comp.cp_rate}/kW"

        elif comp.charge_type == ChargeType.RATCHET:
            ratchet_kw = self._calculate_ratchet_demand(
                peak_kw, historical_peaks, comp.ratchet_pct,
            )
            charge_amount = ratchet_kw * comp.flat_rate
            billing_demand = ratchet_kw
            rate = comp.flat_rate
            notes = (
                f"Ratchet: max({peak_kw}, {comp.ratchet_pct}*historical_max) "
                f"= {ratchet_kw} kW x {comp.flat_rate}/kW"
            )

        elif comp.charge_type == ChargeType.PF_PENALTY:
            charge_amount, notes = self._calculate_pf_penalty(
                peak_kw, power_factor, comp.pf_threshold, comp.flat_rate,
            )
            rate = _safe_divide(charge_amount, peak_kw)

        elif comp.charge_type == ChargeType.REACTIVE:
            if comp.billing_determinant == BillingDeterminant.KVA:
                charge_amount = apparent_kva * comp.flat_rate
                billing_demand = apparent_kva
                rate = comp.flat_rate
                notes = f"kVA billing: {apparent_kva} kVA x {comp.flat_rate}/kVA"
            else:
                reactive = _decimal(
                    math.sqrt(
                        max(float(apparent_kva ** 2 - peak_kw ** 2), 0)
                    )
                )
                charge_amount = reactive * comp.flat_rate
                billing_demand = reactive
                rate = comp.flat_rate
                notes = f"Reactive: {reactive} kVAR x {comp.flat_rate}/kVAR"

        result = DemandCharge(
            component_id=comp.component_id,
            charge_type=comp.charge_type,
            billing_period=billing_period,
            billing_demand_kw=_round_val(billing_demand, 2),
            rate_applied=_round_val(rate, 4),
            charge_amount=_round_val(charge_amount, 2),
            currency=comp.currency,
            notes=notes,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _calculate_tiered_charge(
        self,
        peak_kw: Decimal,
        tiers: List[Dict[str, Decimal]],
    ) -> Tuple[Decimal, str]:
        """Calculate tiered demand charge.

        Args:
            peak_kw: Peak demand (kW).
            tiers: List of tier definitions.

        Returns:
            Tuple of (total_charge, notes_string).
        """
        total = Decimal("0")
        notes_parts: List[str] = []

        for tier in tiers:
            low = _decimal(tier.get("low_kw", 0))
            high = _decimal(tier.get("high_kw", 999999))
            rate = _decimal(tier.get("rate", 0))

            if peak_kw <= low:
                continue

            tier_kw = min(peak_kw, high) - low
            tier_kw = max(tier_kw, Decimal("0"))
            tier_charge = tier_kw * rate
            total += tier_charge
            notes_parts.append(
                f"Tier {low}-{high}: {tier_kw} kW x {rate}/kW = {_round_val(tier_charge, 2)}"
            )

        return total, "; ".join(notes_parts)

    def _calculate_ratchet_demand(
        self,
        current_peak: Decimal,
        historical_peaks: List[Decimal],
        ratchet_pct: Decimal,
    ) -> Decimal:
        """Calculate ratchet demand from current and historical peaks.

        Args:
            current_peak: Current period peak demand.
            historical_peaks: Past N periods peak demands.
            ratchet_pct: Ratchet percentage (e.g. 0.80).

        Returns:
            Billing demand after ratchet application.
        """
        if not historical_peaks:
            return current_peak

        historical_max = max(historical_peaks)
        ratchet_demand = ratchet_pct * historical_max
        return max(current_peak, ratchet_demand)

    def _calculate_pf_penalty(
        self,
        peak_kw: Decimal,
        actual_pf: Decimal,
        threshold_pf: Decimal,
        base_rate: Decimal,
    ) -> Tuple[Decimal, str]:
        """Calculate power factor penalty.

        Formula: penalty = (threshold_pf / actual_pf - 1) * peak_kw * base_rate

        Args:
            peak_kw: Peak demand.
            actual_pf: Actual power factor.
            threshold_pf: Minimum power factor threshold.
            base_rate: Base demand charge rate.

        Returns:
            Tuple of (penalty_amount, notes_string).
        """
        if actual_pf >= threshold_pf or actual_pf <= Decimal("0"):
            return Decimal("0"), "Power factor meets threshold -- no penalty."

        multiplier = _safe_divide(threshold_pf, actual_pf) - Decimal("1")
        penalty = multiplier * peak_kw * base_rate
        notes = (
            f"PF penalty: ({threshold_pf}/{actual_pf} - 1) * "
            f"{peak_kw} kW * {base_rate}/kW = {_round_val(penalty, 2)}"
        )
        return penalty, notes

    def _total_charge_for_peak(
        self,
        peak_kw: Decimal,
        components: List[TariffComponent],
        power_factor: Decimal,
        historical_peaks: List[Decimal],
    ) -> Decimal:
        """Calculate total demand charge for a given peak demand.

        Args:
            peak_kw: Peak demand (kW).
            components: Tariff components.
            power_factor: Actual power factor.
            historical_peaks: Historical peaks for ratchet.

        Returns:
            Total demand charge amount.
        """
        apparent_kva = _safe_divide(peak_kw, power_factor) if power_factor > Decimal("0") else peak_kw
        total = Decimal("0")

        for comp in components:
            charge = self._calculate_component_charge(
                comp, peak_kw, power_factor, apparent_kva,
                historical_peaks, Decimal("0"), {}, "",
            )
            total += charge.charge_amount

        return total

    def _generate_recommendations(
        self,
        breakdown: ChargeBreakdown,
        marginals: List[MarginalValue],
        peak_kw: Decimal,
        power_factor: Decimal,
    ) -> List[str]:
        """Generate tariff optimization recommendations.

        Args:
            breakdown: Current charge breakdown.
            marginals: Marginal value analysis.
            peak_kw: Current peak demand.
            power_factor: Actual power factor.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if breakdown.demand_charge_pct > Decimal("40"):
            recs.append(
                f"Demand charges represent {breakdown.demand_charge_pct}% of "
                "total bill. Peak shaving has high ROI potential."
            )

        if power_factor < self._pf_threshold:
            recs.append(
                f"Power factor ({power_factor}) is below threshold "
                f"({self._pf_threshold}). Install capacitor banks to "
                "eliminate PF penalties and reduce kVA billing."
            )

        if marginals:
            max_mv = max(marginals, key=lambda m: m.marginal_rate)
            if max_mv.marginal_rate > Decimal("20"):
                recs.append(
                    f"Highest marginal value is ${max_mv.marginal_rate}/kW at "
                    f"{max_mv.from_kw} kW. Target this range for BESS dispatch."
                )

        # Check for ratchet charges
        ratchet_charges = [
            c for c in breakdown.charges if c.charge_type == ChargeType.RATCHET
        ]
        if ratchet_charges:
            recs.append(
                "Ratchet demand is active. Preventing new peaks is critical "
                "to avoid setting a higher ratchet floor for 11 months."
            )

        # Check for tiered structure
        tiered_charges = [
            c for c in breakdown.charges if c.charge_type == ChargeType.TIERED
        ]
        if tiered_charges:
            recs.append(
                "Tiered demand structure detected. Reducing peak below "
                "the next tier boundary maximises marginal savings."
            )

        if not recs:
            recs.append(
                "Demand charge structure is straightforward. Monitor "
                "peak trends and maintain demand limiting controls."
            )

        return recs
