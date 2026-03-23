# -*- coding: utf-8 -*-
"""
RatchetAnalysisEngine - PACK-038 Peak Shaving Engine 7
=======================================================

Ratchet demand analysis engine for billing peak persistence.  Analyses
how historical demand spikes lock in elevated demand charges through
ratchet clauses, quantifies the financial impact of ratchet persistence,
identifies spike root causes, plans prevention strategies, and projects
ratchet decay timelines.

Calculation Methodology:
    Ratchet Demand:
        ratchet_kw = max(peak_demands_in_window) * ratchet_pct / 100
        billed_demand = max(actual_demand, ratchet_kw)
        excess_charge = (ratchet_kw - actual_demand) * demand_rate
                        if ratchet_kw > actual_demand else 0

    12-Month Rolling Ratchet:
        ratchet_kw = max(peak_demand for last 12 months) * ratchet_pct
        Resets when all peaks in window fall below threshold.

    Spike Root Cause Analysis:
        weather_correlation = corr(temp, demand) over spike period
        equipment_flag = demand_delta > 2 * std_dev in < 15 min
        startup_flag = spike occurs within 60 min of operating start

    Prevention ROI:
        annual_excess_charges = sum(monthly excess ratchet charges)
        prevention_cost = equipment + installation + maintenance
        payback_months = prevention_cost / monthly_savings
        roi_pct = annual_savings / prevention_cost * 100

    Ratchet Decay Timeline:
        months_remaining = months until oldest spike exits window
        projected_ratchet[m] = max(peaks in rolling window at month m)

Regulatory References:
    - NARUC Electric Utility Rate Design Manual
    - FERC Uniform System of Accounts (18 CFR Part 101)
    - Standard commercial/industrial tariff structures (all US utilities)
    - UK DCUSA - Distribution Connection and Use of System Agreement
    - EU Directive 2019/944 - Common rules for internal electricity market
    - ASHRAE Guideline 14-2014 - Measurement of Energy Savings

Zero-Hallucination:
    - Ratchet calculations use deterministic rolling-window max functions
    - Demand rate data from published utility tariff schedules
    - No LLM involvement in any financial or ratchet calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-038 Peak Shaving
Engine:  7 of 10
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


class RatchetType(str, Enum):
    """Ratchet clause type.

    ANNUAL:       12-month rolling ratchet (most common).
    SEASONAL:     Seasonal ratchet (e.g. summer only).
    ROLLING:      Configurable rolling window (N months).
    FIXED_PERIOD: Fixed period ratchet (e.g. contract term).
    """
    ANNUAL = "annual"
    SEASONAL = "seasonal"
    ROLLING = "rolling"
    FIXED_PERIOD = "fixed_period"


class RatchetPercentage(str, Enum):
    """Ratchet percentage threshold.

    Common ratchet clause percentages that determine the minimum
    billed demand as a fraction of peak demand in the window.

    PCT_75:  75% of peak demand.
    PCT_80:  80% of peak demand (most common).
    PCT_85:  85% of peak demand.
    PCT_90:  90% of peak demand.
    PCT_100: 100% of peak demand (full ratchet).
    """
    PCT_75 = "75"
    PCT_80 = "80"
    PCT_85 = "85"
    PCT_90 = "90"
    PCT_100 = "100"


class SpikeCause(str, Enum):
    """Root cause classification for demand spikes.

    WEATHER:            Weather-driven demand increase (HVAC).
    EQUIPMENT_FAILURE:  Equipment failure causing abnormal load.
    STARTUP_SEQUENCE:   Simultaneous equipment startup.
    PRODUCTION_SURGE:   Production demand spike.
    UTILITY_ERROR:      Utility metering or billing error.
    UNKNOWN:            Root cause not determined.
    """
    WEATHER = "weather"
    EQUIPMENT_FAILURE = "equipment_failure"
    STARTUP_SEQUENCE = "startup_sequence"
    PRODUCTION_SURGE = "production_surge"
    UTILITY_ERROR = "utility_error"
    UNKNOWN = "unknown"


class PreventionStrategy(str, Enum):
    """Demand spike prevention strategy.

    BESS:               Battery energy storage for peak limiting.
    LOAD_LIMIT:         Demand limiting controller.
    STARTUP_SEQUENCE:   Sequenced equipment startup.
    DEMAND_CONTROLLER:  Automated demand controller.
    COMBINED:           Multiple strategies combined.
    """
    BESS = "bess"
    LOAD_LIMIT = "load_limit"
    STARTUP_SEQUENCE = "startup_sequence"
    DEMAND_CONTROLLER = "demand_controller"
    COMBINED = "combined"


class RiskLevel(str, Enum):
    """Ratchet risk level.

    LOW:       Demand well below ratchet threshold.
    MODERATE:  Demand approaching ratchet threshold.
    HIGH:      Demand near or at ratchet threshold.
    CRITICAL:  Demand spike likely to set new ratchet.
    """
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default ratchet parameters by type.
DEFAULT_RATCHET_PARAMS: Dict[str, Dict[str, Any]] = {
    RatchetType.ANNUAL.value: {
        "window_months": 12,
        "default_pct": Decimal("80"),
        "reset_eligible": True,
        "description": "12-month rolling ratchet at 80% of peak",
    },
    RatchetType.SEASONAL.value: {
        "window_months": 4,
        "default_pct": Decimal("85"),
        "reset_eligible": True,
        "description": "Seasonal ratchet (summer 4-month window)",
    },
    RatchetType.ROLLING.value: {
        "window_months": 12,
        "default_pct": Decimal("80"),
        "reset_eligible": True,
        "description": "Configurable rolling window ratchet",
    },
    RatchetType.FIXED_PERIOD.value: {
        "window_months": 36,
        "default_pct": Decimal("90"),
        "reset_eligible": False,
        "description": "Fixed-period contract ratchet (36 months)",
    },
}

# Prevention strategy costs (typical ranges, USD).
PREVENTION_COSTS: Dict[str, Dict[str, Decimal]] = {
    PreventionStrategy.BESS.value: {
        "cost_per_kw": Decimal("400"),
        "annual_maintenance_pct": Decimal("2.0"),
        "lifespan_years": Decimal("15"),
        "description": "Battery energy storage for peak limiting",
    },
    PreventionStrategy.LOAD_LIMIT.value: {
        "cost_per_kw": Decimal("50"),
        "annual_maintenance_pct": Decimal("5.0"),
        "lifespan_years": Decimal("10"),
        "description": "Demand limiting relay/controller",
    },
    PreventionStrategy.STARTUP_SEQUENCE.value: {
        "cost_per_kw": Decimal("15"),
        "annual_maintenance_pct": Decimal("3.0"),
        "lifespan_years": Decimal("20"),
        "description": "Sequenced startup controller",
    },
    PreventionStrategy.DEMAND_CONTROLLER.value: {
        "cost_per_kw": Decimal("75"),
        "annual_maintenance_pct": Decimal("4.0"),
        "lifespan_years": Decimal("12"),
        "description": "Automated demand response controller",
    },
    PreventionStrategy.COMBINED.value: {
        "cost_per_kw": Decimal("200"),
        "annual_maintenance_pct": Decimal("3.0"),
        "lifespan_years": Decimal("12"),
        "description": "Combined prevention strategy",
    },
}

# Spike detection thresholds.
SPIKE_THRESHOLD_STD_DEVS: Decimal = Decimal("2.0")
STARTUP_WINDOW_MINUTES: int = 60
MAX_MONTHLY_RECORDS: int = 120


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class RatchetDemand(BaseModel):
    """Monthly demand record for ratchet analysis.

    Attributes:
        record_id: Record identifier.
        month: Month (1-12).
        year: Year.
        peak_demand_kw: Monthly peak demand (kW).
        avg_demand_kw: Monthly average demand (kW).
        billed_demand_kw: Billed demand including ratchet (kW).
        demand_rate_per_kw: Demand charge rate (USD/kW).
        demand_charge_usd: Total demand charge (USD).
        peak_timestamp: Timestamp of peak demand.
        temperature_high_f: Monthly high temperature (F).
        notes: Additional notes.
    """
    record_id: str = Field(
        default_factory=_new_uuid, description="Record ID"
    )
    month: int = Field(
        ..., ge=1, le=12, description="Month (1-12)"
    )
    year: int = Field(
        ..., ge=2000, le=2050, description="Year"
    )
    peak_demand_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Peak demand (kW)"
    )
    avg_demand_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Average demand (kW)"
    )
    billed_demand_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Billed demand (kW)"
    )
    demand_rate_per_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Demand rate (USD/kW)"
    )
    demand_charge_usd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Demand charge (USD)"
    )
    peak_timestamp: Optional[datetime] = Field(
        default=None, description="Peak timestamp"
    )
    temperature_high_f: Decimal = Field(
        default=Decimal("85"), description="Monthly high temp (F)"
    )
    notes: str = Field(
        default="", max_length=1000, description="Notes"
    )

    @field_validator("peak_demand_kw")
    @classmethod
    def validate_peak(cls, v: Decimal) -> Decimal:
        """Ensure peak demand is non-negative."""
        if v < Decimal("0"):
            raise ValueError("Peak demand cannot be negative.")
        return v


class SpikeAnalysis(BaseModel):
    """Demand spike data for root cause analysis.

    Attributes:
        spike_id: Spike identifier.
        spike_date: Date/time of the spike.
        spike_demand_kw: Peak demand during spike (kW).
        pre_spike_demand_kw: Demand before spike (kW).
        demand_delta_kw: Change in demand (kW).
        ramp_rate_kw_per_min: Ramp rate during spike (kW/min).
        duration_minutes: Duration of spike (minutes).
        temperature_f: Temperature at time of spike (F).
        is_startup_related: Whether spike is within startup window.
        suspected_cause: Suspected root cause.
        notes: Additional notes.
    """
    spike_id: str = Field(
        default_factory=_new_uuid, description="Spike ID"
    )
    spike_date: datetime = Field(
        default_factory=_utcnow, description="Spike date/time"
    )
    spike_demand_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Spike demand (kW)"
    )
    pre_spike_demand_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Pre-spike demand (kW)"
    )
    demand_delta_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Demand delta (kW)"
    )
    ramp_rate_kw_per_min: Decimal = Field(
        default=Decimal("0"), ge=0, description="Ramp rate (kW/min)"
    )
    duration_minutes: int = Field(
        default=15, ge=1, description="Spike duration (min)"
    )
    temperature_f: Decimal = Field(
        default=Decimal("85"), description="Temperature (F)"
    )
    is_startup_related: bool = Field(
        default=False, description="Startup-related flag"
    )
    suspected_cause: SpikeCause = Field(
        default=SpikeCause.UNKNOWN, description="Suspected cause"
    )
    notes: str = Field(
        default="", max_length=1000, description="Notes"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class RatchetImpact(BaseModel):
    """Ratchet clause financial impact analysis.

    Attributes:
        impact_id: Impact analysis identifier.
        ratchet_type: Ratchet clause type.
        ratchet_pct: Ratchet percentage applied.
        window_months: Ratchet window (months).
        peak_demand_in_window_kw: Highest peak in ratchet window (kW).
        ratchet_threshold_kw: Ratchet threshold demand (kW).
        current_actual_demand_kw: Current month actual demand (kW).
        excess_demand_kw: Demand billed above actual (kW).
        monthly_excess_charge_usd: Monthly excess ratchet charge (USD).
        annual_excess_charge_usd: Annualised excess charge (USD).
        months_remaining: Months until ratchet resets.
        peak_month: Month that set the ratchet.
        peak_year: Year that set the ratchet.
        risk_level: Current ratchet risk level.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    impact_id: str = Field(
        default_factory=_new_uuid, description="Impact ID"
    )
    ratchet_type: RatchetType = Field(
        default=RatchetType.ANNUAL, description="Ratchet type"
    )
    ratchet_pct: Decimal = Field(
        default=Decimal("80"), description="Ratchet percentage"
    )
    window_months: int = Field(
        default=12, ge=1, description="Window months"
    )
    peak_demand_in_window_kw: Decimal = Field(
        default=Decimal("0"), description="Peak demand in window (kW)"
    )
    ratchet_threshold_kw: Decimal = Field(
        default=Decimal("0"), description="Ratchet threshold (kW)"
    )
    current_actual_demand_kw: Decimal = Field(
        default=Decimal("0"), description="Current actual demand (kW)"
    )
    excess_demand_kw: Decimal = Field(
        default=Decimal("0"), description="Excess demand (kW)"
    )
    monthly_excess_charge_usd: Decimal = Field(
        default=Decimal("0"), description="Monthly excess charge (USD)"
    )
    annual_excess_charge_usd: Decimal = Field(
        default=Decimal("0"), description="Annual excess charge (USD)"
    )
    months_remaining: int = Field(
        default=0, ge=0, description="Months until ratchet resets"
    )
    peak_month: int = Field(
        default=1, ge=1, le=12, description="Peak month"
    )
    peak_year: int = Field(
        default=2026, description="Peak year"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW, description="Risk level"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class PreventionPlan(BaseModel):
    """Ratchet spike prevention plan.

    Attributes:
        plan_id: Plan identifier.
        strategy: Prevention strategy.
        target_peak_reduction_kw: Target peak reduction (kW).
        equipment_cost_usd: Equipment cost (USD).
        installation_cost_usd: Installation cost (USD).
        annual_maintenance_usd: Annual maintenance cost (USD).
        total_investment_usd: Total investment (USD).
        annual_savings_usd: Annual savings from prevention (USD).
        simple_payback_months: Simple payback period (months).
        roi_pct: Return on investment (%).
        npv_10yr_usd: 10-year net present value (USD).
        lifespan_years: Equipment lifespan (years).
        break_even_date: Estimated break-even date.
        recommendation: Strategy recommendation.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    plan_id: str = Field(
        default_factory=_new_uuid, description="Plan ID"
    )
    strategy: PreventionStrategy = Field(
        default=PreventionStrategy.DEMAND_CONTROLLER, description="Strategy"
    )
    target_peak_reduction_kw: Decimal = Field(
        default=Decimal("0"), description="Target reduction (kW)"
    )
    equipment_cost_usd: Decimal = Field(
        default=Decimal("0"), description="Equipment cost (USD)"
    )
    installation_cost_usd: Decimal = Field(
        default=Decimal("0"), description="Installation cost (USD)"
    )
    annual_maintenance_usd: Decimal = Field(
        default=Decimal("0"), description="Annual maintenance (USD)"
    )
    total_investment_usd: Decimal = Field(
        default=Decimal("0"), description="Total investment (USD)"
    )
    annual_savings_usd: Decimal = Field(
        default=Decimal("0"), description="Annual savings (USD)"
    )
    simple_payback_months: Decimal = Field(
        default=Decimal("0"), description="Payback (months)"
    )
    roi_pct: Decimal = Field(
        default=Decimal("0"), description="ROI (%)"
    )
    npv_10yr_usd: Decimal = Field(
        default=Decimal("0"), description="10-year NPV (USD)"
    )
    lifespan_years: Decimal = Field(
        default=Decimal("10"), description="Lifespan (years)"
    )
    break_even_date: Optional[str] = Field(
        default=None, description="Break-even date"
    )
    recommendation: str = Field(
        default="", max_length=2000, description="Recommendation"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class RatchetResult(BaseModel):
    """Comprehensive ratchet analysis result.

    Attributes:
        result_id: Result identifier.
        ratchet_type: Ratchet type analysed.
        ratchet_pct: Ratchet percentage.
        impact: Financial impact analysis.
        spike_causes: Spike root cause breakdown.
        prevention_plans: Prevention strategy plans.
        decay_timeline: Month-by-month ratchet decay projection.
        total_excess_charges_usd: Total excess charges in analysis period.
        total_preventable_savings_usd: Total preventable savings.
        months_analysed: Number of months analysed.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID"
    )
    ratchet_type: RatchetType = Field(
        default=RatchetType.ANNUAL, description="Ratchet type"
    )
    ratchet_pct: Decimal = Field(
        default=Decimal("80"), description="Ratchet percentage"
    )
    impact: Optional[RatchetImpact] = Field(
        default=None, description="Impact analysis"
    )
    spike_causes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Spike cause breakdown"
    )
    prevention_plans: List[PreventionPlan] = Field(
        default_factory=list, description="Prevention plans"
    )
    decay_timeline: List[Dict[str, Any]] = Field(
        default_factory=list, description="Ratchet decay timeline"
    )
    total_excess_charges_usd: Decimal = Field(
        default=Decimal("0"), description="Total excess charges (USD)"
    )
    total_preventable_savings_usd: Decimal = Field(
        default=Decimal("0"), description="Total preventable savings (USD)"
    )
    months_analysed: int = Field(
        default=0, ge=0, description="Months analysed"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RatchetAnalysisEngine:
    """Ratchet demand analysis engine for billing peak persistence.

    Analyses how historical demand spikes lock in elevated demand
    charges through ratchet clauses, quantifies financial impact,
    identifies spike root causes, plans prevention strategies, and
    projects ratchet decay timelines.

    Usage::

        engine = RatchetAnalysisEngine()
        impact = engine.analyze_ratchet(monthly_records)
        quantified = engine.quantify_impact(monthly_records, demand_rate)
        spikes = engine.identify_spikes(spike_data)
        plan = engine.plan_prevention(target_kw, annual_excess)
        decay = engine.project_decay(monthly_records, ratchet_pct)

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise RatchetAnalysisEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - ratchet_type (str): default ratchet type
                - ratchet_pct (Decimal): default ratchet percentage
                - window_months (int): default window months
                - discount_rate (Decimal): NPV discount rate
        """
        self.config = config or {}
        self._ratchet_type = RatchetType(
            self.config.get("ratchet_type", RatchetType.ANNUAL.value)
        )
        self._ratchet_pct = _decimal(
            self.config.get("ratchet_pct", Decimal("80"))
        )
        self._window_months = int(
            self.config.get("window_months", 12)
        )
        self._discount_rate = _decimal(
            self.config.get("discount_rate", Decimal("0.08"))
        )
        logger.info(
            "RatchetAnalysisEngine v%s initialised (type=%s, pct=%.0f%%, window=%d mo)",
            self.engine_version,
            self._ratchet_type.value,
            float(self._ratchet_pct),
            self._window_months,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze_ratchet(
        self,
        records: List[RatchetDemand],
        ratchet_type: Optional[RatchetType] = None,
        ratchet_pct: Optional[Decimal] = None,
        window_months: Optional[int] = None,
    ) -> RatchetImpact:
        """Analyse ratchet demand from monthly billing records.

        Identifies the peak demand within the ratchet window, calculates
        the ratchet threshold, and determines excess charges due to the
        ratchet clause.

        Args:
            records: Monthly demand records (sorted by date ascending).
            ratchet_type: Override ratchet type.
            ratchet_pct: Override ratchet percentage.
            window_months: Override window months.

        Returns:
            RatchetImpact with financial analysis.

        Raises:
            ValueError: If no records provided.
        """
        t0 = time.perf_counter()
        r_type = ratchet_type or self._ratchet_type
        r_pct = ratchet_pct or self._ratchet_pct
        w_months = window_months or self._window_months

        logger.info(
            "Analysing ratchet: %d records, type=%s, pct=%.0f%%, window=%d mo",
            len(records), r_type.value, float(r_pct), w_months,
        )

        if not records:
            raise ValueError("No monthly demand records provided.")

        # Sort by year/month
        sorted_records = sorted(records, key=lambda r: (r.year, r.month))

        # Get the ratchet window (last N months)
        window_records = sorted_records[-w_months:] if len(sorted_records) >= w_months else sorted_records

        # Find peak in window
        peak_record = max(window_records, key=lambda r: r.peak_demand_kw)
        peak_kw = peak_record.peak_demand_kw

        # Ratchet threshold
        ratchet_threshold = peak_kw * r_pct / Decimal("100")

        # Current month (last record)
        current = sorted_records[-1]
        current_actual = current.peak_demand_kw

        # Excess demand
        excess_kw = max(ratchet_threshold - current_actual, Decimal("0"))

        # Demand rate
        demand_rate = current.demand_rate_per_kw
        if demand_rate <= Decimal("0"):
            # Estimate from billed charge
            demand_rate = _safe_divide(
                current.demand_charge_usd, current.billed_demand_kw
            )

        monthly_excess = excess_kw * demand_rate
        annual_excess = monthly_excess * Decimal("12")

        # Months remaining until ratchet-setting peak exits window
        peak_idx = 0
        for idx, rec in enumerate(sorted_records):
            if rec.record_id == peak_record.record_id:
                peak_idx = idx
                break
        total_records = len(sorted_records)
        months_from_peak_to_end = total_records - 1 - peak_idx
        months_remaining = max(w_months - months_from_peak_to_end, 0)

        # Risk level
        risk = self._assess_risk(current_actual, ratchet_threshold, peak_kw)

        impact = RatchetImpact(
            ratchet_type=r_type,
            ratchet_pct=_round_val(r_pct, 0),
            window_months=w_months,
            peak_demand_in_window_kw=_round_val(peak_kw, 2),
            ratchet_threshold_kw=_round_val(ratchet_threshold, 2),
            current_actual_demand_kw=_round_val(current_actual, 2),
            excess_demand_kw=_round_val(excess_kw, 2),
            monthly_excess_charge_usd=_round_val(monthly_excess, 2),
            annual_excess_charge_usd=_round_val(annual_excess, 2),
            months_remaining=months_remaining,
            peak_month=peak_record.month,
            peak_year=peak_record.year,
            risk_level=risk,
        )
        impact.provenance_hash = _compute_hash(impact)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Ratchet analysis: peak=%.0f kW, threshold=%.0f kW, "
            "excess=%.0f kW ($%.2f/mo), %d mo remaining, risk=%s, "
            "hash=%s (%.1f ms)",
            float(peak_kw), float(ratchet_threshold), float(excess_kw),
            float(monthly_excess), months_remaining, risk.value,
            impact.provenance_hash[:16], elapsed,
        )
        return impact

    def quantify_impact(
        self,
        records: List[RatchetDemand],
        demand_rate: Optional[Decimal] = None,
        ratchet_pct: Optional[Decimal] = None,
        window_months: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Quantify the total financial impact of ratchet clauses.

        Calculates month-by-month excess charges due to ratchet,
        total charges over the analysis period, and what the charges
        would have been without the ratchet clause.

        Args:
            records: Monthly demand records.
            demand_rate: Override demand rate (USD/kW).
            ratchet_pct: Override ratchet percentage.
            window_months: Override window months.

        Returns:
            Dictionary with detailed financial impact analysis.
        """
        t0 = time.perf_counter()
        r_pct = ratchet_pct or self._ratchet_pct
        w_months = window_months or self._window_months

        logger.info(
            "Quantifying ratchet impact: %d records, pct=%.0f%%, window=%d mo",
            len(records), float(r_pct), w_months,
        )

        if not records:
            raise ValueError("No monthly demand records provided.")

        sorted_records = sorted(records, key=lambda r: (r.year, r.month))
        monthly_details: List[Dict[str, Any]] = []
        total_actual_charge = Decimal("0")
        total_ratchet_charge = Decimal("0")
        total_excess = Decimal("0")

        for idx, record in enumerate(sorted_records):
            # Determine rolling window
            start_idx = max(0, idx - w_months + 1)
            window = sorted_records[start_idx:idx + 1]

            # Peak in window
            peak_in_window = max(r.peak_demand_kw for r in window)
            ratchet_threshold = peak_in_window * r_pct / Decimal("100")

            # Billed demand
            billed = max(record.peak_demand_kw, ratchet_threshold)

            # Demand rate
            rate = demand_rate or record.demand_rate_per_kw
            if rate <= Decimal("0"):
                rate = _safe_divide(record.demand_charge_usd, billed)

            # Charges
            actual_charge = record.peak_demand_kw * rate
            ratchet_charge = billed * rate
            excess_charge = max(ratchet_charge - actual_charge, Decimal("0"))

            total_actual_charge += actual_charge
            total_ratchet_charge += ratchet_charge
            total_excess += excess_charge

            monthly_details.append({
                "month": record.month,
                "year": record.year,
                "peak_demand_kw": str(_round_val(record.peak_demand_kw, 2)),
                "ratchet_threshold_kw": str(_round_val(ratchet_threshold, 2)),
                "billed_demand_kw": str(_round_val(billed, 2)),
                "actual_charge_usd": str(_round_val(actual_charge, 2)),
                "ratchet_charge_usd": str(_round_val(ratchet_charge, 2)),
                "excess_charge_usd": str(_round_val(excess_charge, 2)),
                "demand_rate_per_kw": str(_round_val(rate, 4)),
            })

        excess_pct = _safe_pct(total_excess, total_ratchet_charge)

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "months_analysed": len(sorted_records),
            "ratchet_pct": str(_round_val(r_pct, 0)),
            "window_months": w_months,
            "total_actual_charge_usd": str(_round_val(total_actual_charge, 2)),
            "total_ratchet_charge_usd": str(_round_val(total_ratchet_charge, 2)),
            "total_excess_charge_usd": str(_round_val(total_excess, 2)),
            "excess_pct_of_total": str(_round_val(excess_pct, 2)),
            "avg_monthly_excess_usd": str(_round_val(
                _safe_divide(total_excess, _decimal(len(sorted_records))), 2
            )),
            "monthly_details": monthly_details,
            "calculated_at": _utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Ratchet impact: %d months, excess=$%.2f (%.1f%% of total), "
            "hash=%s (%.1f ms)",
            len(sorted_records), float(total_excess), float(excess_pct),
            result["provenance_hash"][:16], elapsed,
        )
        return result

    def identify_spikes(
        self,
        spikes: List[SpikeAnalysis],
        demand_std_dev: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Identify and classify demand spikes by root cause.

        Analyses each spike's characteristics to determine likely root
        cause (weather, equipment failure, startup sequence, production
        surge, or utility error).

        Args:
            spikes: List of demand spike records.
            demand_std_dev: Standard deviation of normal demand (kW).

        Returns:
            Dictionary with spike classifications and breakdown.
        """
        t0 = time.perf_counter()
        logger.info("Identifying spike causes: %d spikes", len(spikes))

        if not spikes:
            empty: Dict[str, Any] = {
                "total_spikes": 0,
                "classifications": [],
                "calculated_at": _utcnow().isoformat(),
            }
            empty["provenance_hash"] = _compute_hash(empty)
            return empty

        # Compute std_dev if not provided
        if demand_std_dev is None or demand_std_dev <= Decimal("0"):
            deltas = [s.demand_delta_kw for s in spikes if s.demand_delta_kw > Decimal("0")]
            if deltas:
                mean_delta = sum(deltas, Decimal("0")) / _decimal(len(deltas))
                variance = sum(
                    ((d - mean_delta) ** 2 for d in deltas), Decimal("0")
                ) / _decimal(len(deltas))
                demand_std_dev = variance.sqrt() if variance > Decimal("0") else Decimal("100")
            else:
                demand_std_dev = Decimal("100")

        classifications: List[Dict[str, Any]] = []
        cause_counts: Dict[str, int] = {}
        cause_total_kw: Dict[str, Decimal] = {}

        for spike in spikes:
            cause = self._classify_spike(spike, demand_std_dev)

            cause_name = cause.value
            cause_counts[cause_name] = cause_counts.get(cause_name, 0) + 1
            cause_total_kw[cause_name] = (
                cause_total_kw.get(cause_name, Decimal("0")) + spike.demand_delta_kw
            )

            classifications.append({
                "spike_id": spike.spike_id,
                "spike_date": spike.spike_date.isoformat(),
                "demand_delta_kw": str(_round_val(spike.demand_delta_kw, 2)),
                "ramp_rate_kw_per_min": str(_round_val(spike.ramp_rate_kw_per_min, 2)),
                "classified_cause": cause_name,
                "confidence": self._classify_confidence(spike, cause),
                "preventable": cause_name in (
                    SpikeCause.STARTUP_SEQUENCE.value,
                    SpikeCause.EQUIPMENT_FAILURE.value,
                    SpikeCause.PRODUCTION_SURGE.value,
                ),
            })

        # Breakdown
        breakdown: List[Dict[str, Any]] = []
        total_spikes = len(spikes)
        for cause_name, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
            breakdown.append({
                "cause": cause_name,
                "count": count,
                "pct_of_total": str(_round_val(
                    _safe_pct(_decimal(count), _decimal(total_spikes)), 1
                )),
                "total_delta_kw": str(_round_val(
                    cause_total_kw.get(cause_name, Decimal("0")), 2
                )),
            })

        preventable_count = sum(
            1 for c in classifications if c.get("preventable", False)
        )
        preventable_pct = _safe_pct(_decimal(preventable_count), _decimal(total_spikes))

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "total_spikes": total_spikes,
            "preventable_spikes": preventable_count,
            "preventable_pct": str(_round_val(preventable_pct, 1)),
            "cause_breakdown": breakdown,
            "classifications": classifications,
            "demand_std_dev_kw": str(_round_val(demand_std_dev, 2)),
            "calculated_at": _utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Spike analysis: %d spikes, %d preventable (%.1f%%), hash=%s (%.1f ms)",
            total_spikes, preventable_count, float(preventable_pct),
            result["provenance_hash"][:16], elapsed,
        )
        return result

    def plan_prevention(
        self,
        target_reduction_kw: Decimal,
        annual_excess_charge_usd: Decimal,
        demand_rate_per_kw: Decimal = Decimal("15"),
        strategies: Optional[List[PreventionStrategy]] = None,
    ) -> List[PreventionPlan]:
        """Generate prevention plans for ratchet spike avoidance.

        Evaluates candidate prevention strategies and calculates ROI,
        payback period, and NPV for each.

        Args:
            target_reduction_kw: Target peak demand reduction (kW).
            annual_excess_charge_usd: Annual excess ratchet charges (USD).
            demand_rate_per_kw: Demand charge rate (USD/kW).
            strategies: Specific strategies to evaluate (None = all).

        Returns:
            List of PreventionPlan objects sorted by ROI.
        """
        t0 = time.perf_counter()
        strats = strategies or list(PreventionStrategy)
        logger.info(
            "Planning prevention: target=%.0f kW, annual_excess=$%.2f, %d strategies",
            float(target_reduction_kw), float(annual_excess_charge_usd), len(strats),
        )

        plans: List[PreventionPlan] = []

        for strategy in strats:
            cost_data = PREVENTION_COSTS.get(strategy.value)
            if cost_data is None:
                continue

            # Equipment cost
            cost_per_kw = cost_data["cost_per_kw"]
            equipment_cost = target_reduction_kw * cost_per_kw

            # Installation (20% of equipment)
            installation_cost = equipment_cost * Decimal("0.20")

            # Total investment
            total_investment = equipment_cost + installation_cost

            # Annual maintenance
            maint_pct = cost_data["annual_maintenance_pct"] / Decimal("100")
            annual_maintenance = equipment_cost * maint_pct

            # Annual savings = excess charge eliminated
            # Assume strategy eliminates the full excess if it covers the target
            annual_savings = annual_excess_charge_usd

            # Simple payback (months)
            monthly_net_savings = _safe_divide(
                annual_savings - annual_maintenance, Decimal("12")
            )
            payback_months = _safe_divide(
                total_investment, monthly_net_savings
            )

            # ROI
            net_annual = annual_savings - annual_maintenance
            roi = _safe_pct(net_annual, total_investment)

            # NPV (10 years)
            lifespan = cost_data["lifespan_years"]
            npv = self._calculate_npv(
                total_investment, net_annual, min(Decimal("10"), lifespan),
            )

            # Break-even date
            if monthly_net_savings > Decimal("0"):
                months_int = int(_round_val(payback_months, 0))
                be_year = _utcnow().year + months_int // 12
                be_month = _utcnow().month + months_int % 12
                if be_month > 12:
                    be_year += 1
                    be_month -= 12
                break_even = f"{be_year}-{be_month:02d}"
            else:
                break_even = None

            # Recommendation
            if payback_months <= Decimal("24"):
                rec = f"Strongly recommended: {strategy.value} with {_round_val(payback_months, 0)}-month payback"
            elif payback_months <= Decimal("48"):
                rec = f"Recommended: {strategy.value} with reasonable payback"
            elif payback_months <= Decimal("72"):
                rec = f"Consider: {strategy.value} with extended payback"
            else:
                rec = f"Marginal: {strategy.value} may not justify investment"

            plan = PreventionPlan(
                strategy=strategy,
                target_peak_reduction_kw=_round_val(target_reduction_kw, 2),
                equipment_cost_usd=_round_val(equipment_cost, 2),
                installation_cost_usd=_round_val(installation_cost, 2),
                annual_maintenance_usd=_round_val(annual_maintenance, 2),
                total_investment_usd=_round_val(total_investment, 2),
                annual_savings_usd=_round_val(annual_savings, 2),
                simple_payback_months=_round_val(payback_months, 1),
                roi_pct=_round_val(roi, 2),
                npv_10yr_usd=_round_val(npv, 2),
                lifespan_years=lifespan,
                break_even_date=break_even,
                recommendation=rec,
            )
            plan.provenance_hash = _compute_hash(plan)
            plans.append(plan)

        # Sort by ROI descending
        plans.sort(key=lambda p: p.roi_pct, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Prevention plans: %d strategies evaluated, best ROI=%.1f%%, "
            "hash=%s (%.1f ms)",
            len(plans),
            float(plans[0].roi_pct) if plans else 0.0,
            plans[0].provenance_hash[:16] if plans else "N/A",
            elapsed,
        )
        return plans

    def project_decay(
        self,
        records: List[RatchetDemand],
        ratchet_pct: Optional[Decimal] = None,
        window_months: Optional[int] = None,
        projection_months: int = 24,
        expected_peak_kw: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Project ratchet decay timeline.

        Projects how the ratchet threshold will change month by month
        as older peak demands exit the rolling window.

        Args:
            records: Monthly demand records.
            ratchet_pct: Override ratchet percentage.
            window_months: Override window months.
            projection_months: Months to project forward.
            expected_peak_kw: Expected future monthly peak demand.

        Returns:
            Dictionary with month-by-month decay projection.
        """
        t0 = time.perf_counter()
        r_pct = ratchet_pct or self._ratchet_pct
        w_months = window_months or self._window_months

        logger.info(
            "Projecting ratchet decay: %d records, pct=%.0f%%, "
            "window=%d, projection=%d months",
            len(records), float(r_pct), w_months, projection_months,
        )

        if not records:
            raise ValueError("No monthly demand records provided.")

        sorted_records = sorted(records, key=lambda r: (r.year, r.month))

        # Current ratchet threshold
        current_window = sorted_records[-w_months:]
        current_peak = max(r.peak_demand_kw for r in current_window)
        current_ratchet = current_peak * r_pct / Decimal("100")

        # Expected future peak (average of last 3 months if not provided)
        if expected_peak_kw is None or expected_peak_kw <= Decimal("0"):
            recent = sorted_records[-3:] if len(sorted_records) >= 3 else sorted_records
            expected_peak_kw = sum(
                (r.peak_demand_kw for r in recent), Decimal("0")
            ) / _decimal(len(recent))

        # Build demand series: existing + projected
        demand_series: List[Decimal] = [r.peak_demand_kw for r in sorted_records]

        # Project forward
        for _ in range(projection_months):
            demand_series.append(expected_peak_kw)

        # Calculate ratchet for each projected month
        timeline: List[Dict[str, Any]] = []
        base_idx = len(sorted_records)

        for m in range(projection_months):
            proj_idx = base_idx + m

            # Window for this projected month
            start_idx = max(0, proj_idx - w_months + 1)
            window = demand_series[start_idx:proj_idx + 1]

            if not window:
                continue

            peak_in_window = max(window)
            ratchet_threshold = peak_in_window * r_pct / Decimal("100")
            billed = max(expected_peak_kw, ratchet_threshold)
            excess = max(ratchet_threshold - expected_peak_kw, Decimal("0"))

            # Determine if original spike still in window
            original_spike_in_window = current_peak in window

            timeline.append({
                "month_offset": m + 1,
                "peak_in_window_kw": str(_round_val(peak_in_window, 2)),
                "ratchet_threshold_kw": str(_round_val(ratchet_threshold, 2)),
                "expected_demand_kw": str(_round_val(expected_peak_kw, 2)),
                "billed_demand_kw": str(_round_val(billed, 2)),
                "excess_demand_kw": str(_round_val(excess, 2)),
                "original_spike_in_window": original_spike_in_window,
                "ratchet_reset": not original_spike_in_window and excess <= Decimal("0"),
            })

        # Find reset month
        reset_month = None
        for entry in timeline:
            if entry.get("ratchet_reset", False):
                reset_month = entry["month_offset"]
                break

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "current_peak_kw": str(_round_val(current_peak, 2)),
            "current_ratchet_kw": str(_round_val(current_ratchet, 2)),
            "expected_demand_kw": str(_round_val(expected_peak_kw, 2)),
            "ratchet_pct": str(_round_val(r_pct, 0)),
            "window_months": w_months,
            "projection_months": projection_months,
            "months_to_reset": reset_month,
            "timeline": timeline,
            "calculated_at": _utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Ratchet decay: current=%.0f kW, reset in %s months, "
            "hash=%s (%.1f ms)",
            float(current_ratchet),
            str(reset_month) if reset_month else "N/A",
            result["provenance_hash"][:16], elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal: Spike Classification                                      #
    # ------------------------------------------------------------------ #

    def _classify_spike(
        self,
        spike: SpikeAnalysis,
        demand_std_dev: Decimal,
    ) -> SpikeCause:
        """Classify a demand spike by root cause.

        Uses deterministic heuristics based on ramp rate, duration,
        temperature correlation, and startup timing.

        Args:
            spike: Spike analysis data.
            demand_std_dev: Standard deviation of normal demand.

        Returns:
            Classified SpikeCause.
        """
        # If explicitly flagged as startup-related
        if spike.is_startup_related:
            return SpikeCause.STARTUP_SEQUENCE

        # If pre-assigned cause is not UNKNOWN, honour it
        if spike.suspected_cause != SpikeCause.UNKNOWN:
            return spike.suspected_cause

        # Equipment failure: very fast ramp, short duration
        threshold_kw = demand_std_dev * SPIKE_THRESHOLD_STD_DEVS
        if (spike.ramp_rate_kw_per_min > Decimal("0")
                and spike.demand_delta_kw > threshold_kw
                and spike.duration_minutes <= 15):
            return SpikeCause.EQUIPMENT_FAILURE

        # Startup sequence: occurs during startup window
        if spike.duration_minutes <= STARTUP_WINDOW_MINUTES:
            hour = spike.spike_date.hour
            if 5 <= hour <= 9:
                return SpikeCause.STARTUP_SEQUENCE

        # Weather: high temperature correlation
        if spike.temperature_f >= Decimal("95"):
            return SpikeCause.WEATHER

        # Production surge: moderate ramp, longer duration
        if spike.duration_minutes >= 30 and spike.demand_delta_kw > threshold_kw:
            return SpikeCause.PRODUCTION_SURGE

        return SpikeCause.UNKNOWN

    def _classify_confidence(
        self,
        spike: SpikeAnalysis,
        cause: SpikeCause,
    ) -> str:
        """Determine classification confidence level.

        Args:
            spike: Spike data.
            cause: Classified cause.

        Returns:
            Confidence string: 'high', 'medium', or 'low'.
        """
        if spike.suspected_cause != SpikeCause.UNKNOWN:
            return "high"
        if spike.is_startup_related and cause == SpikeCause.STARTUP_SEQUENCE:
            return "high"
        if cause == SpikeCause.WEATHER and spike.temperature_f >= Decimal("100"):
            return "high"
        if cause == SpikeCause.EQUIPMENT_FAILURE and spike.duration_minutes <= 5:
            return "high"
        if cause == SpikeCause.UNKNOWN:
            return "low"
        return "medium"

    # ------------------------------------------------------------------ #
    # Internal: Risk and Financial Calculations                           #
    # ------------------------------------------------------------------ #

    def _assess_risk(
        self,
        actual_kw: Decimal,
        ratchet_kw: Decimal,
        peak_kw: Decimal,
    ) -> RiskLevel:
        """Assess ratchet risk level.

        Args:
            actual_kw: Current actual demand (kW).
            ratchet_kw: Ratchet threshold (kW).
            peak_kw: Peak demand that set the ratchet (kW).

        Returns:
            RiskLevel classification.
        """
        if actual_kw >= peak_kw * Decimal("0.95"):
            return RiskLevel.CRITICAL
        elif actual_kw >= ratchet_kw:
            return RiskLevel.HIGH
        elif actual_kw >= ratchet_kw * Decimal("0.80"):
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _calculate_npv(
        self,
        investment: Decimal,
        annual_net_savings: Decimal,
        years: Decimal,
    ) -> Decimal:
        """Calculate net present value of prevention investment.

        NPV = -investment + sum(annual_net / (1+r)^t for t in 1..years)

        Args:
            investment: Initial investment (USD).
            annual_net_savings: Annual net savings (USD).
            years: Analysis period (years).

        Returns:
            NPV in USD.
        """
        npv = -investment
        for yr in range(1, int(years) + 1):
            discount_factor = (Decimal("1") + self._discount_rate) ** _decimal(yr)
            npv += _safe_divide(annual_net_savings, discount_factor)
        return npv
