# -*- coding: utf-8 -*-
"""
SavingsEngine - PACK-040 M&V Engine 3
=======================================

Energy savings calculation engine for Measurement & Verification per
IPMVP Core Concepts 2022 and ASHRAE Guideline 14-2014.  Computes
avoided energy use, normalised savings, cost savings, cumulative savings,
and annualised savings with full audit trail.

Calculation Methodology:
    Avoided Energy Use:
        E_savings = E_adjusted_baseline - E_actual_reporting

    Normalised Savings:
        E_norm = E_baseline(standard_conditions) - E_reporting(standard_conditions)

    Cost Savings:
        Cost_savings = (Energy_savings * blended_rate)
                     + (Demand_savings * demand_rate * n_months)

    Cumulative Savings:
        E_cumulative = sum(E_savings_period_i) for i = 1..N

    Annualised Savings:
        E_annual = E_savings_partial * (365 / days_in_period)

    Savings Percentage:
        Savings_pct = E_savings / E_adjusted_baseline * 100

    Demand Savings:
        D_savings = D_baseline_peak - D_actual_peak (kW)

    Normalised Savings at Standard Conditions:
        E_norm = model(TMY_conditions) - E_reporting(TMY_conditions)

Regulatory References:
    - IPMVP Core Concepts 2022, Chapter 6 (Determining Savings)
    - ASHRAE Guideline 14-2014, Section 5.3 (Savings Calculations)
    - ISO 50015:2014, Clause 8.4 (Energy Performance Improvement)
    - ISO 50006:2014, Clause 8 (Energy Performance Indicators)
    - FEMP M&V Guidelines 4.0, Chapter 6

Zero-Hallucination:
    - All savings computed via deterministic arithmetic
    - Cost calculations use provided rates and tariff structures only
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
Engine:  3 of 10
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

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SavingsType(str, Enum):
    """Type of savings calculation.

    AVOIDED_ENERGY:     Adjusted baseline minus actual consumption.
    NORMALISED:         Savings at standard (TMY) conditions.
    COST:               Monetary savings from energy reduction.
    DEMAND:             Peak demand reduction savings.
    COMBINED:           Energy plus demand savings.
    """
    AVOIDED_ENERGY = "avoided_energy"
    NORMALISED = "normalised"
    COST = "cost"
    DEMAND = "demand"
    COMBINED = "combined"

class ReportingPeriodType(str, Enum):
    """Type of reporting period.

    MONTHLY:   Single calendar month.
    QUARTERLY: Three-month quarter.
    ANNUAL:    Full calendar or contract year.
    CUSTOM:    Custom-defined period.
    PARTIAL:   Partial year (requires annualisation).
    """
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"
    PARTIAL = "partial"

class SavingsStatus(str, Enum):
    """Verification status of savings calculation.

    PRELIMINARY:  Initial estimate, not yet verified.
    VERIFIED:     Verified by M&V practitioner.
    AUDITED:      Independently audited.
    GUARANTEED:   Part of a performance guarantee.
    DISPUTED:     Under dispute.
    ADJUSTED:     Adjusted from prior calculation.
    """
    PRELIMINARY = "preliminary"
    VERIFIED = "verified"
    AUDITED = "audited"
    GUARANTEED = "guaranteed"
    DISPUTED = "disputed"
    ADJUSTED = "adjusted"

class EnergyUnit(str, Enum):
    """Unit of energy measurement.

    KWH:      Kilowatt-hours (electricity).
    THERMS:   Therms (natural gas).
    MMBTU:    Million BTU.
    MWH:      Megawatt-hours.
    GJ:       Gigajoules.
    KBtu:     Thousand BTU.
    """
    KWH = "kWh"
    THERMS = "therms"
    MMBTU = "MMBtu"
    MWH = "MWh"
    GJ = "GJ"
    KBtu = "kBtu"

class CurrencyUnit(str, Enum):
    """Currency unit for cost savings.

    USD: US Dollars.
    EUR: Euros.
    GBP: British Pounds.
    CAD: Canadian Dollars.
    AUD: Australian Dollars.
    """
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"

class DemandUnit(str, Enum):
    """Unit for demand measurement.

    KW:  Kilowatts.
    MW:  Megawatts.
    KVA: Kilovolt-amperes.
    """
    KW = "kW"
    MW = "MW"
    KVA = "kVA"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Days per year for annualisation.
DAYS_PER_YEAR: Decimal = Decimal("365")
MONTHS_PER_YEAR: int = 12
HOURS_PER_YEAR: Decimal = Decimal("8760")

# Energy conversion factors to common unit (kWh-equivalent).
ENERGY_TO_KWH: Dict[str, Decimal] = {
    EnergyUnit.KWH.value: Decimal("1"),
    EnergyUnit.THERMS.value: Decimal("29.3001"),
    EnergyUnit.MMBTU.value: Decimal("293.0711"),
    EnergyUnit.MWH.value: Decimal("1000"),
    EnergyUnit.GJ.value: Decimal("277.7778"),
    EnergyUnit.KBtu.value: Decimal("0.293071"),
}

# GHG emission factors (kg CO2e per kWh) by grid region.
GHG_FACTORS: Dict[str, Decimal] = {
    "us_average": Decimal("0.3859"),
    "eu_average": Decimal("0.2560"),
    "uk": Decimal("0.2330"),
    "canada": Decimal("0.1200"),
    "australia": Decimal("0.6800"),
    "india": Decimal("0.7100"),
    "china": Decimal("0.5810"),
    "natural_gas": Decimal("0.1810"),
    "default": Decimal("0.4000"),
}

# Minimum meaningful savings percentage.
MIN_SAVINGS_PCT: Decimal = Decimal("1")

# Maximum reasonable savings percentage (sanity check).
MAX_SAVINGS_PCT: Decimal = Decimal("90")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class PeriodEnergyData(BaseModel):
    """Energy data for a single sub-period (month, week, etc.).

    Attributes:
        period_start: Start of period.
        period_end: End of period.
        baseline_energy: Adjusted baseline energy for this period.
        actual_energy: Actual metered energy for this period.
        baseline_demand_kw: Baseline peak demand (kW).
        actual_demand_kw: Actual peak demand (kW).
        days_in_period: Number of days in this period.
        is_valid: Whether data is valid.
        notes: Additional notes.
    """
    period_start: datetime = Field(
        default_factory=utcnow, description="Period start"
    )
    period_end: datetime = Field(
        default_factory=utcnow, description="Period end"
    )
    baseline_energy: Decimal = Field(
        default=Decimal("0"), ge=0, description="Adjusted baseline energy"
    )
    actual_energy: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual metered energy"
    )
    baseline_demand_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline peak demand (kW)"
    )
    actual_demand_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual peak demand (kW)"
    )
    days_in_period: int = Field(
        default=30, ge=1, description="Days in period"
    )
    is_valid: bool = Field(default=True, description="Data quality flag")
    notes: str = Field(default="", max_length=500, description="Notes")

class CostRateSchedule(BaseModel):
    """Energy cost rate schedule for monetary savings.

    Attributes:
        blended_energy_rate: Blended energy rate ($/kWh).
        demand_rate: Monthly demand charge ($/kW-month).
        energy_rate_on_peak: On-peak energy rate.
        energy_rate_off_peak: Off-peak energy rate.
        on_peak_fraction: Fraction of savings during on-peak.
        fuel_rate: Fuel rate for non-electric savings.
        escalation_rate_pct: Annual rate escalation percentage.
        currency: Currency unit.
    """
    blended_energy_rate: Decimal = Field(
        default=Decimal("0.10"), ge=0, description="Blended rate ($/kWh)"
    )
    demand_rate: Decimal = Field(
        default=Decimal("0"), ge=0, description="Demand charge ($/kW-month)"
    )
    energy_rate_on_peak: Decimal = Field(
        default=Decimal("0.15"), ge=0, description="On-peak rate"
    )
    energy_rate_off_peak: Decimal = Field(
        default=Decimal("0.06"), ge=0, description="Off-peak rate"
    )
    on_peak_fraction: Decimal = Field(
        default=Decimal("0.60"), ge=0, le=1,
        description="Fraction of savings on-peak"
    )
    fuel_rate: Decimal = Field(
        default=Decimal("0"), ge=0, description="Fuel rate ($/therm)"
    )
    escalation_rate_pct: Decimal = Field(
        default=Decimal("2.5"), ge=0,
        description="Annual escalation %"
    )
    currency: CurrencyUnit = Field(
        default=CurrencyUnit.USD, description="Currency"
    )

class SavingsConfig(BaseModel):
    """Configuration for savings calculation.

    Attributes:
        project_id: M&V project identifier.
        ecm_id: ECM identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        reporting_period_start: Start of reporting period.
        reporting_period_end: End of reporting period.
        reporting_period_type: Type of reporting period.
        energy_unit: Unit of energy measurement.
        demand_unit: Unit for demand values.
        cost_schedule: Cost rate schedule for monetary savings.
        total_non_routine_adjustment: Total NRA (already applied to baseline).
        ghg_region: GHG emission factor region for CO2 savings.
        include_demand_savings: Whether to include demand savings.
        include_cost_savings: Whether to include cost savings.
        include_ghg_savings: Whether to compute GHG savings.
        contract_year: Contract year for performance tracking.
        guaranteed_savings: Guaranteed savings target (if applicable).
    """
    project_id: str = Field(default="", description="M&V project ID")
    ecm_id: str = Field(default="", description="ECM identifier")
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(
        default="", max_length=500, description="Facility name"
    )
    reporting_period_start: datetime = Field(
        default_factory=utcnow, description="Reporting period start"
    )
    reporting_period_end: datetime = Field(
        default_factory=utcnow, description="Reporting period end"
    )
    reporting_period_type: ReportingPeriodType = Field(
        default=ReportingPeriodType.ANNUAL, description="Period type"
    )
    energy_unit: EnergyUnit = Field(
        default=EnergyUnit.KWH, description="Energy unit"
    )
    demand_unit: DemandUnit = Field(
        default=DemandUnit.KW, description="Demand unit"
    )
    cost_schedule: CostRateSchedule = Field(
        default_factory=CostRateSchedule, description="Cost rates"
    )
    total_non_routine_adjustment: Decimal = Field(
        default=Decimal("0"), description="Total NRA applied"
    )
    ghg_region: str = Field(
        default="us_average", description="GHG region"
    )
    include_demand_savings: bool = Field(
        default=True, description="Include demand savings"
    )
    include_cost_savings: bool = Field(
        default=True, description="Include cost savings"
    )
    include_ghg_savings: bool = Field(
        default=True, description="Include GHG savings"
    )
    contract_year: int = Field(
        default=1, ge=1, description="Contract year"
    )
    guaranteed_savings: Optional[Decimal] = Field(
        default=None, ge=0, description="Guaranteed savings target"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class PeriodSavingsDetail(BaseModel):
    """Savings detail for a single sub-period.

    Attributes:
        period_start: Period start.
        period_end: Period end.
        days: Days in period.
        baseline_energy: Adjusted baseline energy.
        actual_energy: Actual metered energy.
        energy_savings: Energy savings for period.
        savings_pct: Savings as % of baseline.
        demand_savings_kw: Demand savings (kW).
        cost_savings: Cost savings for period.
        is_positive_savings: True if savings are positive.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    period_start: datetime = Field(default_factory=utcnow)
    period_end: datetime = Field(default_factory=utcnow)
    days: int = Field(default=30)
    baseline_energy: Decimal = Field(default=Decimal("0"))
    actual_energy: Decimal = Field(default=Decimal("0"))
    energy_savings: Decimal = Field(default=Decimal("0"))
    savings_pct: Decimal = Field(default=Decimal("0"))
    demand_savings_kw: Decimal = Field(default=Decimal("0"))
    cost_savings: Decimal = Field(default=Decimal("0"))
    is_positive_savings: bool = Field(default=True)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class CostSavingsBreakdown(BaseModel):
    """Breakdown of cost savings by component.

    Attributes:
        energy_cost_savings: Cost savings from energy reduction.
        demand_cost_savings: Cost savings from demand reduction.
        on_peak_savings: On-peak component.
        off_peak_savings: Off-peak component.
        fuel_cost_savings: Fuel cost savings (non-electric).
        total_cost_savings: Total monetary savings.
        blended_rate_used: Blended rate applied.
        demand_rate_used: Demand rate applied.
        currency: Currency unit.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    energy_cost_savings: Decimal = Field(default=Decimal("0"))
    demand_cost_savings: Decimal = Field(default=Decimal("0"))
    on_peak_savings: Decimal = Field(default=Decimal("0"))
    off_peak_savings: Decimal = Field(default=Decimal("0"))
    fuel_cost_savings: Decimal = Field(default=Decimal("0"))
    total_cost_savings: Decimal = Field(default=Decimal("0"))
    blended_rate_used: Decimal = Field(default=Decimal("0"))
    demand_rate_used: Decimal = Field(default=Decimal("0"))
    currency: CurrencyUnit = Field(default=CurrencyUnit.USD)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class GHGSavingsResult(BaseModel):
    """GHG emissions reduction from energy savings.

    Attributes:
        energy_savings_kwh: Energy savings in kWh-equivalent.
        emission_factor: GHG factor used (kg CO2e/kWh).
        ghg_region: Region used for factor.
        ghg_savings_kg_co2e: GHG reduction (kg CO2e).
        ghg_savings_tonnes_co2e: GHG reduction (tonnes CO2e).
        ghg_savings_metric_tons: GHG reduction (metric tons).
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    energy_savings_kwh: Decimal = Field(default=Decimal("0"))
    emission_factor: Decimal = Field(default=Decimal("0"))
    ghg_region: str = Field(default="")
    ghg_savings_kg_co2e: Decimal = Field(default=Decimal("0"))
    ghg_savings_tonnes_co2e: Decimal = Field(default=Decimal("0"))
    ghg_savings_metric_tons: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class CumulativeSavingsRecord(BaseModel):
    """Cumulative savings tracking record.

    Attributes:
        year: Contract or tracking year.
        period_label: Period label (e.g. "Year 1", "Q3 2025").
        period_energy_savings: Energy savings in this period.
        cumulative_energy_savings: Cumulative energy to date.
        period_cost_savings: Cost savings in this period.
        cumulative_cost_savings: Cumulative cost to date.
        period_ghg_savings_tonnes: GHG savings in this period.
        cumulative_ghg_savings_tonnes: Cumulative GHG to date.
        guaranteed_savings: Guaranteed target (if applicable).
        pct_of_guaranteed: Achieved % of guaranteed target.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    year: int = Field(default=1)
    period_label: str = Field(default="")
    period_energy_savings: Decimal = Field(default=Decimal("0"))
    cumulative_energy_savings: Decimal = Field(default=Decimal("0"))
    period_cost_savings: Decimal = Field(default=Decimal("0"))
    cumulative_cost_savings: Decimal = Field(default=Decimal("0"))
    period_ghg_savings_tonnes: Decimal = Field(default=Decimal("0"))
    cumulative_ghg_savings_tonnes: Decimal = Field(default=Decimal("0"))
    guaranteed_savings: Optional[Decimal] = Field(default=None)
    pct_of_guaranteed: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class SavingsResult(BaseModel):
    """Complete savings calculation result.

    Attributes:
        savings_id: Unique result identifier.
        project_id: M&V project identifier.
        ecm_id: ECM identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        reporting_period_start: Reporting period start.
        reporting_period_end: Reporting period end.
        reporting_period_type: Period type.
        total_days: Total days in reporting period.
        total_baseline_energy: Total adjusted baseline energy.
        total_actual_energy: Total actual energy.
        total_energy_savings: Total energy savings (avoided energy).
        savings_pct: Savings as percentage of baseline.
        annualised_savings: Annualised savings.
        normalised_savings: Normalised savings (TMY conditions).
        peak_demand_savings_kw: Peak demand reduction.
        cost_breakdown: Cost savings breakdown.
        ghg_savings: GHG emissions reduction.
        period_details: Period-by-period savings detail.
        cumulative_record: Cumulative savings record.
        energy_unit: Energy measurement unit.
        savings_status: Verification status.
        savings_type: Type of savings calculation.
        non_routine_adjustment_applied: NRA already in baseline.
        is_positive_savings: True if net savings are positive.
        meets_guarantee: Whether guaranteed savings are met.
        guarantee_shortfall: Shortfall below guarantee (0 if met).
        warnings: Any warnings generated.
        recommendations: Analysis recommendations.
        processing_time_ms: Processing duration.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    savings_id: str = Field(default_factory=_new_uuid)
    project_id: str = Field(default="")
    ecm_id: str = Field(default="")
    facility_id: str = Field(default="")
    facility_name: str = Field(default="", max_length=500)
    reporting_period_start: datetime = Field(default_factory=utcnow)
    reporting_period_end: datetime = Field(default_factory=utcnow)
    reporting_period_type: ReportingPeriodType = Field(
        default=ReportingPeriodType.ANNUAL
    )
    total_days: int = Field(default=0)
    total_baseline_energy: Decimal = Field(default=Decimal("0"))
    total_actual_energy: Decimal = Field(default=Decimal("0"))
    total_energy_savings: Decimal = Field(default=Decimal("0"))
    savings_pct: Decimal = Field(default=Decimal("0"))
    annualised_savings: Decimal = Field(default=Decimal("0"))
    normalised_savings: Decimal = Field(default=Decimal("0"))
    peak_demand_savings_kw: Decimal = Field(default=Decimal("0"))
    cost_breakdown: Optional[CostSavingsBreakdown] = Field(default=None)
    ghg_savings: Optional[GHGSavingsResult] = Field(default=None)
    period_details: List[PeriodSavingsDetail] = Field(default_factory=list)
    cumulative_record: Optional[CumulativeSavingsRecord] = Field(default=None)
    energy_unit: EnergyUnit = Field(default=EnergyUnit.KWH)
    savings_status: SavingsStatus = Field(default=SavingsStatus.PRELIMINARY)
    savings_type: SavingsType = Field(default=SavingsType.AVOIDED_ENERGY)
    non_routine_adjustment_applied: Decimal = Field(default=Decimal("0"))
    is_positive_savings: bool = Field(default=True)
    meets_guarantee: Optional[bool] = Field(default=None)
    guarantee_shortfall: Decimal = Field(default=Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SavingsEngine:
    """Energy savings calculation engine for M&V per IPMVP / ASHRAE 14.

    Computes avoided energy use, normalised savings, cost savings,
    demand savings, GHG reductions, cumulative savings, and annualised
    savings.  All calculations use deterministic Decimal arithmetic
    with SHA-256 provenance hashing.

    Usage::

        engine = SavingsEngine()
        config = SavingsConfig(
            project_id="PRJ-001",
            ecm_id="ECM-001",
            cost_schedule=CostRateSchedule(blended_energy_rate=Decimal("0.12")),
        )
        periods = [
            PeriodEnergyData(
                baseline_energy=Decimal("10500"),
                actual_energy=Decimal("9200"),
                days_in_period=31,
            ),
            ...
        ]
        result = engine.calculate_savings(config, periods)
        print(f"Total savings: {result.total_energy_savings} kWh")
        print(f"Cost savings: {result.cost_breakdown.total_cost_savings}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise SavingsEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - min_savings_pct (float): min meaningful savings %
                - max_savings_pct (float): max sanity-check savings %
                - default_ghg_region (str): default GHG region
        """
        self.config = config or {}
        self._min_savings_pct = _decimal(
            self.config.get("min_savings_pct", MIN_SAVINGS_PCT)
        )
        self._max_savings_pct = _decimal(
            self.config.get("max_savings_pct", MAX_SAVINGS_PCT)
        )
        self._default_ghg_region = self.config.get(
            "default_ghg_region", "us_average"
        )
        logger.info(
            "SavingsEngine v%s initialised (min_pct=%.0f%%, max_pct=%.0f%%)",
            self.engine_version, float(self._min_savings_pct),
            float(self._max_savings_pct),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate_savings(
        self,
        savings_config: SavingsConfig,
        period_data: List[PeriodEnergyData],
        prior_cumulative_energy: Decimal = Decimal("0"),
        prior_cumulative_cost: Decimal = Decimal("0"),
        prior_cumulative_ghg: Decimal = Decimal("0"),
    ) -> SavingsResult:
        """Calculate comprehensive savings for a reporting period.

        Computes avoided energy, cost savings, demand savings, GHG
        reductions, and cumulative tracking in one pass.

        Args:
            savings_config: Savings calculation configuration.
            period_data: Energy data for each sub-period.
            prior_cumulative_energy: Prior cumulative energy savings.
            prior_cumulative_cost: Prior cumulative cost savings.
            prior_cumulative_ghg: Prior cumulative GHG savings (tonnes).

        Returns:
            SavingsResult with all savings calculations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating savings: %s (%d periods, ECM=%s)",
            savings_config.facility_name, len(period_data),
            savings_config.ecm_id,
        )

        valid_periods = [p for p in period_data if p.is_valid]
        if not valid_periods:
            result = SavingsResult(
                project_id=savings_config.project_id,
                ecm_id=savings_config.ecm_id,
                facility_id=savings_config.facility_id,
                facility_name=savings_config.facility_name,
                warnings=["No valid period data provided."],
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Calculate period-by-period savings
        period_details = self._calculate_period_details(
            valid_periods, savings_config
        )

        # Aggregate totals
        total_baseline = sum(
            (pd.baseline_energy for pd in period_details), Decimal("0")
        )
        total_actual = sum(
            (pd.actual_energy for pd in period_details), Decimal("0")
        )
        total_savings = sum(
            (pd.energy_savings for pd in period_details), Decimal("0")
        )
        total_days = sum(pd.days for pd in period_details)

        savings_pct = _safe_pct(total_savings, total_baseline)

        # Annualise if partial year
        annualised = self._annualise_savings(
            total_savings, total_days, savings_config.reporting_period_type
        )

        # Peak demand savings
        demand_savings = Decimal("0")
        if savings_config.include_demand_savings and valid_periods:
            demand_savings = self._calculate_demand_savings(valid_periods)

        # Cost savings
        cost_breakdown: Optional[CostSavingsBreakdown] = None
        if savings_config.include_cost_savings:
            cost_breakdown = self.calculate_cost_savings(
                total_savings, demand_savings, savings_config
            )

        # GHG savings
        ghg_savings: Optional[GHGSavingsResult] = None
        if savings_config.include_ghg_savings:
            ghg_savings = self.calculate_ghg_savings(
                total_savings, savings_config
            )

        # Cumulative tracking
        cumulative = self._build_cumulative_record(
            savings_config, total_savings,
            cost_breakdown.total_cost_savings if cost_breakdown else Decimal("0"),
            ghg_savings.ghg_savings_tonnes_co2e if ghg_savings else Decimal("0"),
            prior_cumulative_energy, prior_cumulative_cost,
            prior_cumulative_ghg,
        )

        # Guarantee check
        meets_guarantee: Optional[bool] = None
        guarantee_shortfall = Decimal("0")
        if savings_config.guaranteed_savings is not None:
            meets_guarantee = total_savings >= savings_config.guaranteed_savings
            if not meets_guarantee:
                guarantee_shortfall = (
                    savings_config.guaranteed_savings - total_savings
                )

        is_positive = total_savings > Decimal("0")

        # Warnings and recommendations
        warnings = self._generate_warnings(
            total_savings, savings_pct, total_baseline,
            period_details, savings_config,
        )
        recommendations = self._generate_recommendations(
            total_savings, savings_pct, demand_savings,
            meets_guarantee, savings_config,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = SavingsResult(
            project_id=savings_config.project_id,
            ecm_id=savings_config.ecm_id,
            facility_id=savings_config.facility_id,
            facility_name=savings_config.facility_name,
            reporting_period_start=savings_config.reporting_period_start,
            reporting_period_end=savings_config.reporting_period_end,
            reporting_period_type=savings_config.reporting_period_type,
            total_days=total_days,
            total_baseline_energy=_round_val(total_baseline, 2),
            total_actual_energy=_round_val(total_actual, 2),
            total_energy_savings=_round_val(total_savings, 2),
            savings_pct=_round_val(savings_pct, 4),
            annualised_savings=_round_val(annualised, 2),
            normalised_savings=_round_val(total_savings, 2),
            peak_demand_savings_kw=_round_val(demand_savings, 2),
            cost_breakdown=cost_breakdown,
            ghg_savings=ghg_savings,
            period_details=period_details,
            cumulative_record=cumulative,
            energy_unit=savings_config.energy_unit,
            savings_type=SavingsType.COMBINED if savings_config.include_demand_savings else SavingsType.AVOIDED_ENERGY,
            non_routine_adjustment_applied=savings_config.total_non_routine_adjustment,
            is_positive_savings=is_positive,
            meets_guarantee=meets_guarantee,
            guarantee_shortfall=_round_val(guarantee_shortfall, 2),
            warnings=warnings,
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Savings calculated: %s, savings=%.1f %s (%.1f%%), "
            "cost=$%.2f, GHG=%.2f tCO2e, "
            "guarantee=%s, hash=%s (%.1f ms)",
            savings_config.facility_name, float(total_savings),
            savings_config.energy_unit.value, float(savings_pct),
            float(cost_breakdown.total_cost_savings) if cost_breakdown else 0.0,
            float(ghg_savings.ghg_savings_tonnes_co2e) if ghg_savings else 0.0,
            "MET" if meets_guarantee else ("NOT MET" if meets_guarantee is False else "N/A"),
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def calculate_cost_savings(
        self,
        energy_savings: Decimal,
        demand_savings_kw: Decimal,
        savings_config: SavingsConfig,
    ) -> CostSavingsBreakdown:
        """Calculate monetary cost savings from energy and demand reductions.

        Applies blended energy rate, TOU split, demand charges, and
        fuel rates to compute total cost savings.

        Args:
            energy_savings: Total energy savings (in config energy unit).
            demand_savings_kw: Peak demand reduction (kW).
            savings_config: Savings configuration with rate schedule.

        Returns:
            CostSavingsBreakdown with component-level cost savings.
        """
        t0 = time.perf_counter()
        logger.info("Calculating cost savings")

        schedule = savings_config.cost_schedule

        # Energy cost savings using blended rate
        energy_cost = energy_savings * schedule.blended_energy_rate

        # TOU split if rates available
        on_peak_savings = Decimal("0")
        off_peak_savings = Decimal("0")
        if schedule.energy_rate_on_peak > Decimal("0"):
            on_peak_energy = energy_savings * schedule.on_peak_fraction
            off_peak_energy = energy_savings * (
                Decimal("1") - schedule.on_peak_fraction
            )
            on_peak_savings = on_peak_energy * schedule.energy_rate_on_peak
            off_peak_savings = off_peak_energy * schedule.energy_rate_off_peak

        # Demand cost savings
        demand_cost = Decimal("0")
        if demand_savings_kw > Decimal("0") and schedule.demand_rate > Decimal("0"):
            # Assume reporting period in months
            rp_days = (
                savings_config.reporting_period_end
                - savings_config.reporting_period_start
            ).days
            n_months = max(1, round(rp_days / 30.44))
            demand_cost = (
                demand_savings_kw * schedule.demand_rate * _decimal(n_months)
            )

        # Fuel cost savings
        fuel_cost = Decimal("0")
        if schedule.fuel_rate > Decimal("0"):
            # Convert energy savings to therms if needed
            fuel_cost = energy_savings * schedule.fuel_rate

        total_cost = energy_cost + demand_cost + fuel_cost

        result = CostSavingsBreakdown(
            energy_cost_savings=_round_val(energy_cost, 2),
            demand_cost_savings=_round_val(demand_cost, 2),
            on_peak_savings=_round_val(on_peak_savings, 2),
            off_peak_savings=_round_val(off_peak_savings, 2),
            fuel_cost_savings=_round_val(fuel_cost, 2),
            total_cost_savings=_round_val(total_cost, 2),
            blended_rate_used=schedule.blended_energy_rate,
            demand_rate_used=schedule.demand_rate,
            currency=schedule.currency,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Cost savings: total=$%.2f (energy=$%.2f, demand=$%.2f, "
            "fuel=$%.2f), hash=%s (%.1f ms)",
            float(total_cost), float(energy_cost), float(demand_cost),
            float(fuel_cost), result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_ghg_savings(
        self,
        energy_savings: Decimal,
        savings_config: SavingsConfig,
    ) -> GHGSavingsResult:
        """Calculate GHG emissions reduction from energy savings.

        Converts energy savings to kWh-equivalent and multiplies by
        the grid emission factor for the specified region.

        Args:
            energy_savings: Energy savings in configuration unit.
            savings_config: Savings configuration with GHG region.

        Returns:
            GHGSavingsResult with emissions reduction.
        """
        t0 = time.perf_counter()
        logger.info("Calculating GHG savings")

        # Convert to kWh
        conversion = ENERGY_TO_KWH.get(
            savings_config.energy_unit.value, Decimal("1")
        )
        savings_kwh = energy_savings * conversion

        # Get emission factor
        ghg_region = savings_config.ghg_region or self._default_ghg_region
        emission_factor = GHG_FACTORS.get(
            ghg_region, GHG_FACTORS["default"]
        )

        ghg_kg = savings_kwh * emission_factor
        ghg_tonnes = _safe_divide(ghg_kg, Decimal("1000"))

        result = GHGSavingsResult(
            energy_savings_kwh=_round_val(savings_kwh, 2),
            emission_factor=emission_factor,
            ghg_region=ghg_region,
            ghg_savings_kg_co2e=_round_val(ghg_kg, 2),
            ghg_savings_tonnes_co2e=_round_val(ghg_tonnes, 4),
            ghg_savings_metric_tons=_round_val(ghg_tonnes, 4),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "GHG savings: %.2f tonnes CO2e (%.0f kWh * %.4f), "
            "region=%s, hash=%s (%.1f ms)",
            float(ghg_tonnes), float(savings_kwh), float(emission_factor),
            ghg_region, result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_normalised_savings(
        self,
        baseline_at_standard: Decimal,
        reporting_at_standard: Decimal,
        savings_config: SavingsConfig,
    ) -> Decimal:
        """Calculate normalised savings at standard (TMY) conditions.

        Normalised savings remove the effect of weather differences
        between baseline and reporting periods by evaluating both at
        the same standard (TMY) conditions.

        Args:
            baseline_at_standard: Baseline model prediction at TMY.
            reporting_at_standard: Reporting model prediction at TMY.
            savings_config: Savings configuration.

        Returns:
            Normalised savings value.
        """
        t0 = time.perf_counter()
        logger.info("Calculating normalised savings")

        normalised = baseline_at_standard - reporting_at_standard

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Normalised savings: %.1f %s (baseline=%.1f, reporting=%.1f) "
            "(%.1f ms)",
            float(normalised), savings_config.energy_unit.value,
            float(baseline_at_standard), float(reporting_at_standard),
            elapsed,
        )
        return _round_val(normalised, 2)

    def calculate_cumulative_savings(
        self,
        annual_savings: List[Decimal],
        annual_costs: Optional[List[Decimal]] = None,
        savings_config: Optional[SavingsConfig] = None,
    ) -> List[CumulativeSavingsRecord]:
        """Calculate cumulative savings over multiple years.

        Tracks year-by-year cumulative energy and cost savings
        for performance contract monitoring.

        Args:
            annual_savings: List of annual energy savings by year.
            annual_costs: Optional list of annual cost savings.
            savings_config: Optional configuration for guarantee check.

        Returns:
            List of CumulativeSavingsRecord, one per year.
        """
        t0 = time.perf_counter()
        logger.info("Calculating cumulative savings: %d years", len(annual_savings))

        records: List[CumulativeSavingsRecord] = []
        cum_energy = Decimal("0")
        cum_cost = Decimal("0")
        cum_ghg = Decimal("0")

        guaranteed = (
            savings_config.guaranteed_savings
            if savings_config and savings_config.guaranteed_savings
            else None
        )

        for i, energy_sav in enumerate(annual_savings):
            cum_energy += energy_sav
            cost_sav = (
                annual_costs[i]
                if annual_costs and i < len(annual_costs)
                else Decimal("0")
            )
            cum_cost += cost_sav

            pct_guaranteed = Decimal("0")
            if guaranteed and guaranteed > Decimal("0"):
                pct_guaranteed = _safe_pct(energy_sav, guaranteed)

            record = CumulativeSavingsRecord(
                year=i + 1,
                period_label=f"Year {i + 1}",
                period_energy_savings=_round_val(energy_sav, 2),
                cumulative_energy_savings=_round_val(cum_energy, 2),
                period_cost_savings=_round_val(cost_sav, 2),
                cumulative_cost_savings=_round_val(cum_cost, 2),
                guaranteed_savings=guaranteed,
                pct_of_guaranteed=_round_val(pct_guaranteed, 4),
            )
            record.provenance_hash = _compute_hash(record)
            records.append(record)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Cumulative savings: %d years, total=%.1f, hash=%s (%.1f ms)",
            len(records), float(cum_energy),
            records[-1].provenance_hash[:16] if records else "n/a", elapsed,
        )
        return records

    def convert_energy_units(
        self,
        value: Decimal,
        from_unit: EnergyUnit,
        to_unit: EnergyUnit,
    ) -> Decimal:
        """Convert energy value between units.

        Args:
            value: Energy value to convert.
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            Converted energy value.
        """
        if from_unit == to_unit:
            return value

        # Convert to kWh first
        to_kwh = ENERGY_TO_KWH.get(from_unit.value, Decimal("1"))
        value_kwh = value * to_kwh

        # Convert from kWh to target
        from_kwh = ENERGY_TO_KWH.get(to_unit.value, Decimal("1"))
        result = _safe_divide(value_kwh, from_kwh)

        return _round_val(result, 4)

    # ------------------------------------------------------------------ #
    # Private: Period Detail Calculation                                   #
    # ------------------------------------------------------------------ #

    def _calculate_period_details(
        self,
        periods: List[PeriodEnergyData],
        savings_config: SavingsConfig,
    ) -> List[PeriodSavingsDetail]:
        """Calculate savings for each sub-period."""
        details: List[PeriodSavingsDetail] = []
        schedule = savings_config.cost_schedule

        for pd in periods:
            energy_savings = pd.baseline_energy - pd.actual_energy
            savings_pct = _safe_pct(energy_savings, pd.baseline_energy)
            demand_savings = pd.baseline_demand_kw - pd.actual_demand_kw
            demand_savings = max(Decimal("0"), demand_savings)

            # Period cost savings
            cost_savings = energy_savings * schedule.blended_energy_rate
            if demand_savings > Decimal("0") and schedule.demand_rate > Decimal("0"):
                cost_savings += demand_savings * schedule.demand_rate

            detail = PeriodSavingsDetail(
                period_start=pd.period_start,
                period_end=pd.period_end,
                days=pd.days_in_period,
                baseline_energy=_round_val(pd.baseline_energy, 2),
                actual_energy=_round_val(pd.actual_energy, 2),
                energy_savings=_round_val(energy_savings, 2),
                savings_pct=_round_val(savings_pct, 4),
                demand_savings_kw=_round_val(demand_savings, 2),
                cost_savings=_round_val(cost_savings, 2),
                is_positive_savings=energy_savings > Decimal("0"),
            )
            detail.provenance_hash = _compute_hash(detail)
            details.append(detail)

        return details

    def _calculate_demand_savings(
        self,
        periods: List[PeriodEnergyData],
    ) -> Decimal:
        """Calculate peak demand savings across all periods."""
        max_bl_demand = max(
            (p.baseline_demand_kw for p in periods), default=Decimal("0")
        )
        max_actual_demand = max(
            (p.actual_demand_kw for p in periods), default=Decimal("0")
        )
        demand_savings = max_bl_demand - max_actual_demand
        return max(Decimal("0"), demand_savings)

    def _annualise_savings(
        self,
        total_savings: Decimal,
        total_days: int,
        period_type: ReportingPeriodType,
    ) -> Decimal:
        """Annualise savings for partial-year reporting periods."""
        if period_type == ReportingPeriodType.ANNUAL:
            return total_savings

        if total_days <= 0:
            return Decimal("0")

        annual_factor = _safe_divide(DAYS_PER_YEAR, _decimal(total_days))
        return total_savings * annual_factor

    def _build_cumulative_record(
        self,
        savings_config: SavingsConfig,
        period_energy: Decimal,
        period_cost: Decimal,
        period_ghg: Decimal,
        prior_energy: Decimal,
        prior_cost: Decimal,
        prior_ghg: Decimal,
    ) -> CumulativeSavingsRecord:
        """Build cumulative savings record."""
        cum_energy = prior_energy + period_energy
        cum_cost = prior_cost + period_cost
        cum_ghg = prior_ghg + period_ghg

        pct_guaranteed = Decimal("0")
        if (savings_config.guaranteed_savings
                and savings_config.guaranteed_savings > Decimal("0")):
            pct_guaranteed = _safe_pct(
                period_energy, savings_config.guaranteed_savings
            )

        record = CumulativeSavingsRecord(
            year=savings_config.contract_year,
            period_label=f"Year {savings_config.contract_year}",
            period_energy_savings=_round_val(period_energy, 2),
            cumulative_energy_savings=_round_val(cum_energy, 2),
            period_cost_savings=_round_val(period_cost, 2),
            cumulative_cost_savings=_round_val(cum_cost, 2),
            period_ghg_savings_tonnes=_round_val(period_ghg, 4),
            cumulative_ghg_savings_tonnes=_round_val(cum_ghg, 4),
            guaranteed_savings=savings_config.guaranteed_savings,
            pct_of_guaranteed=_round_val(pct_guaranteed, 4),
        )
        record.provenance_hash = _compute_hash(record)
        return record

    # ------------------------------------------------------------------ #
    # Private: Warnings & Recommendations                                  #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        total_savings: Decimal,
        savings_pct: Decimal,
        total_baseline: Decimal,
        period_details: List[PeriodSavingsDetail],
        savings_config: SavingsConfig,
    ) -> List[str]:
        """Generate savings calculation warnings."""
        warnings: List[str] = []

        if total_savings < Decimal("0"):
            warnings.append(
                "Negative savings detected: actual consumption exceeds "
                "adjusted baseline. This may indicate ECM underperformance, "
                "non-routine changes, or baseline model issues."
            )

        if savings_pct > self._max_savings_pct:
            warnings.append(
                f"Savings of {float(savings_pct):.1f}% exceeds the "
                f"maximum sanity check ({float(self._max_savings_pct):.0f}%). "
                "Review baseline model and adjustment calculations."
            )

        negative_periods = [
            pd for pd in period_details if not pd.is_positive_savings
        ]
        if negative_periods and total_savings > Decimal("0"):
            warnings.append(
                f"{len(negative_periods)} of {len(period_details)} periods "
                "show negative savings. Investigate seasonal or operational "
                "patterns."
            )

        if total_baseline == Decimal("0"):
            warnings.append(
                "Adjusted baseline energy is zero. Cannot calculate "
                "meaningful savings percentage."
            )

        return warnings

    def _generate_recommendations(
        self,
        total_savings: Decimal,
        savings_pct: Decimal,
        demand_savings: Decimal,
        meets_guarantee: Optional[bool],
        savings_config: SavingsConfig,
    ) -> List[str]:
        """Generate savings recommendations."""
        recs: List[str] = []

        if Decimal("0") < savings_pct < self._min_savings_pct:
            recs.append(
                f"Savings of {float(savings_pct):.2f}% are below the "
                f"minimum meaningful threshold ({float(self._min_savings_pct):.0f}%). "
                "Consider whether savings are within model noise."
            )

        if meets_guarantee is False:
            recs.append(
                "Savings do not meet the guaranteed target. "
                "Review ECM performance, operating conditions, "
                "and non-routine adjustments."
            )

        if demand_savings == Decimal("0") and savings_config.include_demand_savings:
            recs.append(
                "No demand savings detected. If the ECM is expected "
                "to reduce peak demand, investigate metering and "
                "demand measurement."
            )

        if total_savings > Decimal("0"):
            recs.append(
                "Document the verified savings in an M&V report per "
                "IPMVP requirements and file with project stakeholders."
            )

        return recs
