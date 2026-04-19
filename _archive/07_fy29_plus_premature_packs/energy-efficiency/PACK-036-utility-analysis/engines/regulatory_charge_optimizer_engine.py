# -*- coding: utf-8 -*-
"""
RegulatoryChargeOptimizerEngine - PACK-036 Utility Analysis Engine 8
======================================================================

Analyses and optimises non-commodity charges on electricity and gas utility
bills.  Non-commodity charges (transmission, distribution, levies, taxes,
capacity, reactive power penalties) typically represent 40-65 % of the total
electricity bill in European and North American markets.  This engine
decomposes a bill into its regulatory components, identifies exemption
eligibility, optimises capacity and power-factor charges, evaluates voltage-
level migration opportunities, projects future charge trajectories, and
quantifies the impact of on-site self-generation.

Decomposition Methodology:
    Non-commodity share = (total_bill - commodity_cost) / total_bill * 100
    Each charge is classified by ChargeType and ChargeMethodology, with
    the annual amount and share-of-bill computed deterministically.

Capacity Optimisation:
    optimal_kW = actual_max_demand_kW * (1 + headroom_pct)
    savings     = (current_kW - optimal_kW) * rate_per_kW_month * 12

Power Factor Correction:
    Required kVAR = kW * (tan(arccos(current_pf)) - tan(arccos(target_pf)))
    penalty_saved = current_penalty - corrected_penalty (0 when pf >= threshold)

Voltage Level Migration:
    savings = annual_kWh * (current_rate - proposed_rate)
    payback = transformation_capex / savings

Regulatory References:
    - German EnWG (Energiewirtschaftsgesetz) Netzentgelte
    - German StromNEV (Stromnetzentgeltverordnung) sections 17-19
    - UK Ofgem DUoS/TNUoS Charging Methodologies (DCUSA, CUSC)
    - UK BSUoS (Balancing Services Use of System)
    - UK Climate Change Levy (CCL) Finance Act 2000
    - US FERC Order 2222, ISO/RTO tariff schedules
    - EU Directive 2019/944 (Electricity Market Directive)
    - EU Regulation 2019/943 (Electricity Market Regulation)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Charge rates from published tariff schedules / regulatory databases
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-036 Utility Analysis
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

from pydantic import BaseModel, Field, field_validator, model_validator

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

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChargeType(str, Enum):
    """Non-commodity charge types found on electricity and gas bills.

    Covers the major categories across EU (DE, FR, UK, NL) and US ISO markets.
    Source: Ofgem, BNetzA, FERC tariff classifications.
    """
    TRANSMISSION = "transmission"
    DISTRIBUTION = "distribution"
    SYSTEM = "system"
    RENEWABLE_LEVY = "renewable_levy"
    CAPACITY = "capacity"
    REACTIVE_POWER = "reactive_power"
    TAX = "tax"
    CLIMATE_LEVY = "climate_levy"
    BALANCING = "balancing"
    ANCILLARY = "ancillary"
    CONNECTION = "connection"
    METER = "meter"
    STRANDED_COST = "stranded_cost"
    PUBLIC_PURPOSE = "public_purpose"

class Jurisdiction(str, Enum):
    """Electricity market jurisdictions with distinct regulatory charge regimes.

    EU member states use country-level codes; US jurisdictions use ISO/RTO
    identifiers because charge structures vary by independent system operator.
    """
    EU_DE = "eu_de"
    EU_FR = "eu_fr"
    EU_NL = "eu_nl"
    EU_IT = "eu_it"
    EU_ES = "eu_es"
    EU_AT = "eu_at"
    EU_BE = "eu_be"
    EU_PL = "eu_pl"
    UK_GB = "uk_gb"
    US_PJM = "us_pjm"
    US_ERCOT = "us_ercot"
    US_CAISO = "us_caiso"
    US_NYISO = "us_nyiso"
    US_MISO = "us_miso"
    US_ISO_NE = "us_iso_ne"
    US_SPP = "us_spp"

class ExemptionType(str, Enum):
    """Regulatory charge exemption or reduction categories.

    These exemptions are codified in national legislation (e.g. German
    StromNEV section 19(2), UK CCL Climate Change Agreement scheme).
    """
    ENERGY_INTENSIVE = "energy_intensive"
    SELF_SUPPLY = "self_supply"
    CHP = "chp"
    RENEWABLE_GENERATOR = "renewable_generator"
    STORAGE = "storage"
    ETS_COMPENSATION = "ets_compensation"
    REDUCED_NETWORK = "reduced_network"
    INTERRUPTIBLE_LOAD = "interruptible_load"

class VoltageLevel(str, Enum):
    """Grid connection voltage levels.

    Higher voltage connections generally attract lower per-kWh network
    charges because fewer transformation steps are required.
    """
    LOW_VOLTAGE = "low_voltage"
    MEDIUM_VOLTAGE = "medium_voltage"
    HIGH_VOLTAGE = "high_voltage"
    EXTRA_HIGH_VOLTAGE = "extra_high_voltage"

class OptimizationAction(str, Enum):
    """Categories of actions to reduce non-commodity charges."""
    VOLTAGE_UPGRADE = "voltage_upgrade"
    CAPACITY_REDUCTION = "capacity_reduction"
    PF_CORRECTION = "pf_correction"
    EXEMPTION_APPLICATION = "exemption_application"
    SELF_GENERATION = "self_generation"
    DEMAND_SHIFTING = "demand_shifting"
    STORAGE_ARBITRAGE = "storage_arbitrage"
    CONTRACT_RENEGOTIATION = "contract_renegotiation"

class ChargeMethodology(str, Enum):
    """Methodology by which a regulatory charge is computed.

    VOLUMETRIC:       EUR/kWh (or ct/kWh) based on total consumption.
    CAPACITY_BASED:   EUR/kW/month based on contracted capacity.
    PEAK_BASED:       EUR/kW based on measured peak demand.
    FIXED:            EUR/month flat charge.
    COINCIDENT_PEAK:  EUR/kW based on demand at system coincident peak.
    POSTAGE_STAMP:    Flat volumetric rate regardless of location.
    """
    VOLUMETRIC = "volumetric"
    CAPACITY_BASED = "capacity_based"
    PEAK_BASED = "peak_based"
    FIXED = "fixed"
    COINCIDENT_PEAK = "coincident_peak"
    POSTAGE_STAMP = "postage_stamp"

# ---------------------------------------------------------------------------
# Reference Data Constants
# ---------------------------------------------------------------------------

# German electricity non-commodity charge components (ct/kWh unless noted).
# Source: BNetzA Monitoringbericht 2025, BDEW Strompreisanalyse.
# Applicable to commercial/industrial consumers > 100 MWh/year.
DE_CHARGE_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "netzentgelte": {
        "type": ChargeType.DISTRIBUTION,
        "methodology": ChargeMethodology.CAPACITY_BASED,
        "rate_ct_kwh_lv": 7.50,
        "rate_ct_kwh_mv": 4.20,
        "rate_ct_kwh_hv": 2.10,
        "rate_ct_kwh_ehv": 1.00,
        "source": "StromNEV, BNetzA 2025",
    },
    "kwk_umlage": {
        "type": ChargeType.RENEWABLE_LEVY,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_ct_kwh": 0.275,
        "source": "KWKG 2023, section 26",
    },
    "offshore_netzumlage": {
        "type": ChargeType.SYSTEM,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_ct_kwh": 0.591,
        "source": "EnWG section 17f",
    },
    "stromnev_umlage": {
        "type": ChargeType.DISTRIBUTION,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_ct_kwh": 0.643,
        "source": "StromNEV section 19(2)",
    },
    "konzessionsabgabe": {
        "type": ChargeType.PUBLIC_PURPOSE,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_ct_kwh": 1.32,
        "source": "KAV section 2",
    },
    "stromsteuer": {
        "type": ChargeType.TAX,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_ct_kwh": 2.05,
        "source": "StromStG section 3",
    },
    "mehrwertsteuer": {
        "type": ChargeType.TAX,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_pct": 19.0,
        "source": "UStG section 12",
    },
}

# UK electricity non-commodity charge components.
# Source: Ofgem, DCUSA, CUSC, HMRC, published charging methodologies 2025/26.
UK_CHARGE_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "duos": {
        "type": ChargeType.DISTRIBUTION,
        "methodology": ChargeMethodology.CAPACITY_BASED,
        "rate_p_kwh_lv": 3.50,
        "rate_p_kwh_hv": 1.80,
        "rate_p_kwh_ehv": 0.60,
        "source": "DCUSA DUoS Methodology 2025/26",
    },
    "tnuos": {
        "type": ChargeType.TRANSMISSION,
        "methodology": ChargeMethodology.COINCIDENT_PEAK,
        "rate_gbp_kw": 55.0,
        "triad_months": ["nov", "dec", "jan", "feb"],
        "source": "CUSC TNUoS Tariff 2025/26",
    },
    "bsuos": {
        "type": ChargeType.BALANCING,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_p_kwh": 0.55,
        "source": "ESO BSUoS Forecast Q1 2026",
    },
    "ccl": {
        "type": ChargeType.CLIMATE_LEVY,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_p_kwh": 0.775,
        "cca_discount_pct": 92.0,
        "source": "HMRC Climate Change Levy 2025/26",
    },
    "ro": {
        "type": ChargeType.RENEWABLE_LEVY,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_p_kwh": 0.65,
        "source": "Ofgem Renewables Obligation 2025/26",
    },
    "cfd": {
        "type": ChargeType.RENEWABLE_LEVY,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_p_kwh": 0.80,
        "source": "LCCC CFD Interim Levy Rate 2025/26",
    },
    "cm": {
        "type": ChargeType.CAPACITY,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_p_kwh": 0.30,
        "source": "EMR Capacity Market Settlement 2025/26",
    },
    "vat": {
        "type": ChargeType.TAX,
        "methodology": ChargeMethodology.VOLUMETRIC,
        "rate_pct": 20.0,
        "source": "HMRC VAT Act 1994",
    },
}

# US typical non-commodity charge components by ISO/RTO (USD/kWh or $/kW-month).
# Source: ISO/RTO published tariff schedules, FERC filings 2025.
US_CHARGE_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "us_pjm": {
        "transmission_usd_kw_month": 6.50,
        "ancillary_usd_mwh": 3.20,
        "capacity_usd_kw_month": 8.50,
        "rps_usd_mwh": 2.80,
        "source": "PJM OATT Schedule 1, 7, 8 (2025)",
    },
    "us_ercot": {
        "transmission_usd_kw_month": 4.80,
        "ancillary_usd_mwh": 4.50,
        "capacity_usd_kw_month": 0.0,
        "rps_usd_mwh": 1.20,
        "source": "ERCOT Protocols, PUCT tariffs (2025)",
    },
    "us_caiso": {
        "transmission_usd_kw_month": 7.20,
        "ancillary_usd_mwh": 5.10,
        "capacity_usd_kw_month": 9.80,
        "rps_usd_mwh": 4.50,
        "source": "CAISO OATT, CPUC tariffs (2025)",
    },
    "us_nyiso": {
        "transmission_usd_kw_month": 8.00,
        "ancillary_usd_mwh": 3.80,
        "capacity_usd_kw_month": 11.20,
        "rps_usd_mwh": 3.60,
        "source": "NYISO OATT, NYPSC tariffs (2025)",
    },
    "us_miso": {
        "transmission_usd_kw_month": 5.50,
        "ancillary_usd_mwh": 2.90,
        "capacity_usd_kw_month": 6.00,
        "rps_usd_mwh": 2.10,
        "source": "MISO OATT Schedule 26 (2025)",
    },
    "us_iso_ne": {
        "transmission_usd_kw_month": 9.50,
        "ancillary_usd_mwh": 4.20,
        "capacity_usd_kw_month": 10.50,
        "rps_usd_mwh": 5.80,
        "source": "ISO-NE OATT, state RPS schedules (2025)",
    },
    "us_spp": {
        "transmission_usd_kw_month": 4.20,
        "ancillary_usd_mwh": 2.50,
        "capacity_usd_kw_month": 3.50,
        "rps_usd_mwh": 1.50,
        "source": "SPP OATT Schedule 9, 11 (2025)",
    },
}

# Non-commodity share of total electricity bill by jurisdiction (%).
# Source: ACER Market Monitoring Report 2025, EIA Electric Power Annual 2024.
NON_COMMODITY_SHARE_PCT: Dict[str, float] = {
    Jurisdiction.EU_DE: 52.0,
    Jurisdiction.EU_FR: 45.0,
    Jurisdiction.EU_NL: 48.0,
    Jurisdiction.EU_IT: 50.0,
    Jurisdiction.EU_ES: 46.0,
    Jurisdiction.EU_AT: 47.0,
    Jurisdiction.EU_BE: 49.0,
    Jurisdiction.EU_PL: 42.0,
    Jurisdiction.UK_GB: 55.0,
    Jurisdiction.US_PJM: 40.0,
    Jurisdiction.US_ERCOT: 35.0,
    Jurisdiction.US_CAISO: 48.0,
    Jurisdiction.US_NYISO: 45.0,
    Jurisdiction.US_MISO: 38.0,
    Jurisdiction.US_ISO_NE: 47.0,
    Jurisdiction.US_SPP: 36.0,
}

# Network charge rate differentials by voltage level (index, LV = 1.00).
# Lower voltage = higher per-kWh charge due to more transformation steps.
# Source: BNetzA Netzentgelte Strukturvergleich 2025, Ofgem CDCM.
VOLTAGE_LEVEL_FACTORS: Dict[VoltageLevel, float] = {
    VoltageLevel.LOW_VOLTAGE: 1.00,
    VoltageLevel.MEDIUM_VOLTAGE: 0.56,
    VoltageLevel.HIGH_VOLTAGE: 0.28,
    VoltageLevel.EXTRA_HIGH_VOLTAGE: 0.13,
}

# Typical annual charge growth rates by jurisdiction (% per annum).
# Source: ACER, Ofgem forward cost estimates, ISO/RTO tariff trends.
CHARGE_GROWTH_RATES: Dict[str, Dict[str, float]] = {
    Jurisdiction.EU_DE: {
        ChargeType.DISTRIBUTION: 2.5,
        ChargeType.TRANSMISSION: 3.0,
        ChargeType.RENEWABLE_LEVY: -1.5,
        ChargeType.TAX: 1.0,
        ChargeType.CLIMATE_LEVY: 4.0,
        ChargeType.CAPACITY: 3.5,
        "default": 2.0,
    },
    Jurisdiction.UK_GB: {
        ChargeType.DISTRIBUTION: 3.0,
        ChargeType.TRANSMISSION: 4.5,
        ChargeType.RENEWABLE_LEVY: 2.0,
        ChargeType.TAX: 1.5,
        ChargeType.CLIMATE_LEVY: 3.0,
        ChargeType.BALANCING: 5.0,
        ChargeType.CAPACITY: 4.0,
        "default": 2.5,
    },
    "default": {
        ChargeType.DISTRIBUTION: 2.5,
        ChargeType.TRANSMISSION: 3.0,
        ChargeType.RENEWABLE_LEVY: 1.0,
        ChargeType.TAX: 1.5,
        ChargeType.CAPACITY: 3.0,
        "default": 2.0,
    },
}

# Power-factor penalty thresholds by jurisdiction.
# Most European networks penalise below cos(phi) = 0.90.
PF_PENALTY_THRESHOLDS: Dict[str, float] = {
    Jurisdiction.EU_DE: 0.90,
    Jurisdiction.EU_FR: 0.93,
    Jurisdiction.EU_NL: 0.85,
    Jurisdiction.EU_IT: 0.90,
    Jurisdiction.EU_ES: 0.90,
    Jurisdiction.UK_GB: 0.95,
    "default": 0.90,
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input / Output
# ---------------------------------------------------------------------------

class RegulatoryCharge(BaseModel):
    """A single non-commodity charge line item on a utility bill."""

    charge_type: ChargeType = Field(..., description="Category of charge")
    name: str = Field(..., description="Human-readable charge name")
    methodology: ChargeMethodology = Field(
        ..., description="How the charge is calculated"
    )
    rate: float = Field(0.0, description="Unit rate for the charge")
    unit: str = Field(
        "per_kwh",
        description="Rate unit: per_kwh, per_kw, per_month, pct",
    )
    annual_amount_eur: float = Field(
        0.0, ge=0, description="Total annual charge amount in EUR"
    )
    share_of_total_bill_pct: float = Field(
        0.0, ge=0, le=100, description="Charge as percentage of total bill"
    )
    jurisdiction: Optional[str] = Field(
        None, description="Applicable jurisdiction"
    )
    description: str = Field("", description="Explanatory note")
    optimizable: bool = Field(
        True, description="Whether this charge can be reduced via action"
    )

class ChargeBreakdown(BaseModel):
    """Full decomposition of a utility bill into commodity and non-commodity."""

    total_bill_eur: float = Field(..., ge=0, description="Total bill amount")
    commodity_cost_eur: float = Field(
        ..., ge=0, description="Wholesale energy cost"
    )
    commodity_pct: float = Field(
        ..., ge=0, le=100, description="Commodity share of total bill"
    )
    non_commodity_cost_eur: float = Field(
        ..., ge=0, description="Total non-commodity charges"
    )
    non_commodity_pct: float = Field(
        ..., ge=0, le=100, description="Non-commodity share of total bill"
    )
    charges: List[RegulatoryCharge] = Field(
        default_factory=list, description="Itemised charge list"
    )
    taxes_eur: float = Field(0.0, ge=0, description="Total tax amount")
    taxes_pct: float = Field(
        0.0, ge=0, le=100, description="Tax share of total bill"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

class ExemptionAssessment(BaseModel):
    """Assessment of a facility's eligibility for a regulatory charge exemption."""

    exemption_type: ExemptionType = Field(
        ..., description="Type of exemption"
    )
    jurisdiction: str = Field(..., description="Applicable jurisdiction")
    eligible: bool = Field(
        False, description="Whether the facility qualifies"
    )
    eligibility_criteria: List[str] = Field(
        default_factory=list, description="Criteria that must be met"
    )
    current_charge_eur: float = Field(
        0.0, ge=0, description="Current annual charge subject to exemption"
    )
    exempted_amount_eur: float = Field(
        0.0, ge=0, description="Amount that would be exempted"
    )
    savings_eur: float = Field(
        0.0, ge=0, description="Net annual savings from exemption"
    )
    application_requirements: List[str] = Field(
        default_factory=list, description="Documents / steps to apply"
    )
    deadline: Optional[str] = Field(
        None, description="Application deadline if applicable"
    )

class CapacityOptimization(BaseModel):
    """Result of contracted capacity vs actual demand analysis."""

    current_capacity_kw: float = Field(
        ..., ge=0, description="Currently contracted capacity in kW"
    )
    actual_max_demand_kw: float = Field(
        ..., ge=0, description="Measured maximum demand in kW"
    )
    utilization_pct: float = Field(
        ..., ge=0, le=100, description="Capacity utilization percentage"
    )
    optimal_capacity_kw: float = Field(
        ..., ge=0, description="Recommended contracted capacity"
    )
    annual_savings_eur: float = Field(
        ..., ge=0, description="Annual savings from capacity reduction"
    )
    penalty_risk: str = Field(
        "low", description="Risk level for exceeding capacity: low/medium/high"
    )
    recommendation: str = Field("", description="Action recommendation")

class PowerFactorOptimization(BaseModel):
    """Power factor correction analysis and capacitor sizing."""

    current_pf: float = Field(
        ..., ge=0, le=1.0, description="Current power factor (cos phi)"
    )
    penalty_threshold: float = Field(
        ..., ge=0, le=1.0, description="Jurisdictional penalty threshold"
    )
    current_penalty_eur: float = Field(
        ..., ge=0, description="Current annual reactive power penalty"
    )
    target_pf: float = Field(
        ..., ge=0, le=1.0, description="Target power factor"
    )
    capacitor_kvar: float = Field(
        ..., ge=0, description="Required capacitor bank size in kVAR"
    )
    capacitor_cost_eur: float = Field(
        ..., ge=0, description="Estimated capacitor bank cost"
    )
    annual_savings_eur: float = Field(
        ..., ge=0, description="Annual penalty savings"
    )
    payback_months: float = Field(
        ..., ge=0, description="Simple payback in months"
    )

class VoltageLevelAnalysis(BaseModel):
    """Analysis of voltage level migration economics."""

    current_voltage: VoltageLevel = Field(
        ..., description="Current connection voltage level"
    )
    proposed_voltage: VoltageLevel = Field(
        ..., description="Proposed higher voltage level"
    )
    current_rate_per_kwh: float = Field(
        ..., ge=0, description="Current network charge rate EUR/kWh"
    )
    proposed_rate_per_kwh: float = Field(
        ..., ge=0, description="Proposed rate at higher voltage EUR/kWh"
    )
    transformation_cost_eur: float = Field(
        ..., ge=0, description="Capital cost of transformation equipment"
    )
    annual_savings_eur: float = Field(
        ..., ge=0, description="Annual charge savings"
    )
    payback_years: float = Field(
        ..., ge=0, description="Simple payback in years"
    )
    feasibility: str = Field(
        "feasible",
        description="Feasibility rating: feasible / marginal / infeasible",
    )

class GridChargeProjection(BaseModel):
    """Projected non-commodity charges for a future year."""

    year: int = Field(..., ge=2024, le=2050, description="Projection year")
    projected_charges_by_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Projected annual charge by ChargeType",
    )
    total_projected_eur: float = Field(
        ..., ge=0, description="Total projected non-commodity charges"
    )
    change_vs_current_pct: float = Field(
        ..., description="Percentage change vs current year"
    )
    regulatory_drivers: List[str] = Field(
        default_factory=list,
        description="Key regulatory drivers affecting charges",
    )

class SelfGenerationImpact(BaseModel):
    """Impact of on-site generation on non-commodity charges."""

    generation_kwh: float = Field(
        ..., ge=0, description="Annual on-site generation in kWh"
    )
    avoided_grid_charges_eur: float = Field(
        ..., ge=0, description="Grid charges avoided by self-generation"
    )
    remaining_grid_charges_eur: float = Field(
        ..., ge=0, description="Residual grid charges on imported electricity"
    )
    net_savings_eur: float = Field(
        ..., description="Net savings after standby / backup charges"
    )
    standby_charges_eur: float = Field(
        ..., ge=0, description="Standby / backup supply charges"
    )
    backup_charges_eur: float = Field(
        ..., ge=0, description="Additional backup capacity charges"
    )

class ChargeOptimizationResult(BaseModel):
    """Comprehensive result of non-commodity charge optimisation for a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    current_non_commodity_eur: float = Field(
        ..., ge=0, description="Current annual non-commodity charges"
    )
    optimized_non_commodity_eur: float = Field(
        ..., ge=0, description="Projected optimised annual charges"
    )
    total_savings_eur: float = Field(
        ..., ge=0, description="Total annual savings achievable"
    )
    savings_pct: float = Field(
        ..., ge=0, le=100, description="Savings as percentage of current"
    )
    actions: List[str] = Field(
        default_factory=list,
        description="Recommended optimisation actions",
    )
    exemptions: List[ExemptionAssessment] = Field(
        default_factory=list, description="Exemption assessments"
    )
    capacity_optimization: Optional[CapacityOptimization] = Field(
        None, description="Capacity optimisation result"
    )
    pf_optimization: Optional[PowerFactorOptimization] = Field(
        None, description="Power factor correction result"
    )
    voltage_analysis: Optional[VoltageLevelAnalysis] = Field(
        None, description="Voltage level migration analysis"
    )
    projections: List[GridChargeProjection] = Field(
        default_factory=list, description="Multi-year charge projections"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(
        0.0, ge=0, description="Engine processing time in ms"
    )
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Model Rebuild (required with `from __future__ import annotations`)
# ---------------------------------------------------------------------------

RegulatoryCharge.model_rebuild()
ChargeBreakdown.model_rebuild()
ExemptionAssessment.model_rebuild()
CapacityOptimization.model_rebuild()
PowerFactorOptimization.model_rebuild()
VoltageLevelAnalysis.model_rebuild()
GridChargeProjection.model_rebuild()
SelfGenerationImpact.model_rebuild()
ChargeOptimizationResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RegulatoryChargeOptimizerEngine:
    """Engine for analysing and optimising non-commodity utility charges.

    Decomposes utility bills into regulatory charge components, assesses
    exemption eligibility, optimises capacity and power-factor charges,
    evaluates voltage-level migration, projects future charges, and
    quantifies self-generation impact.

    All numeric computations use deterministic Decimal arithmetic with no
    LLM involvement (zero-hallucination principle).  Every result carries
    a SHA-256 provenance hash for full audit trail.

    Attributes:
        version: Engine version string.
        notes: Accumulated processing notes for audit trail.

    Example:
        >>> engine = RegulatoryChargeOptimizerEngine()
        >>> breakdown = engine.decompose_bill(bill_data, Jurisdiction.EU_DE)
        >>> assert breakdown.provenance_hash != ""
    """

    def __init__(self) -> None:
        """Initialise RegulatoryChargeOptimizerEngine."""
        self.version: str = _MODULE_VERSION
        self._notes: List[str] = []
        logger.info(
            "RegulatoryChargeOptimizerEngine v%s initialised.", self.version
        )

    # ------------------------------------------------------------------ #
    # Public -- Bill Decomposition
    # ------------------------------------------------------------------ #

    def decompose_bill(
        self,
        bill_data: Dict[str, Any],
        jurisdiction: Jurisdiction,
    ) -> ChargeBreakdown:
        """Decompose a utility bill into commodity and non-commodity components.

        Takes raw bill data containing total amount, commodity cost, and
        consumption, then itemises all non-commodity charges applicable to
        the given jurisdiction.

        Args:
            bill_data: Dict with keys: total_bill_eur, commodity_cost_eur,
                annual_consumption_kwh, contracted_capacity_kw,
                voltage_level (optional).
            jurisdiction: Regulatory jurisdiction for charge lookup.

        Returns:
            ChargeBreakdown with itemised charges and percentages.
        """
        t0 = time.perf_counter()
        self._notes = []

        d_total = _decimal(bill_data.get("total_bill_eur", 0))
        d_commodity = _decimal(bill_data.get("commodity_cost_eur", 0))
        d_consumption = _decimal(bill_data.get("annual_consumption_kwh", 0))
        d_capacity = _decimal(bill_data.get("contracted_capacity_kw", 0))
        voltage = bill_data.get("voltage_level", VoltageLevel.LOW_VOLTAGE)
        if isinstance(voltage, str):
            try:
                voltage = VoltageLevel(voltage)
            except ValueError:
                voltage = VoltageLevel.LOW_VOLTAGE

        d_non_commodity = d_total - d_commodity
        commodity_pct = _safe_pct(d_commodity, d_total)
        non_commodity_pct = _safe_pct(d_non_commodity, d_total)

        charges = self._build_charge_list(
            jurisdiction, d_consumption, d_capacity, d_non_commodity, voltage
        )

        d_taxes = sum(
            _decimal(c.annual_amount_eur)
            for c in charges
            if c.charge_type == ChargeType.TAX
        )
        taxes_pct = _safe_pct(d_taxes, d_total)

        # Recompute share-of-bill for each charge.
        for charge in charges:
            charge.share_of_total_bill_pct = _round2(
                float(_safe_pct(_decimal(charge.annual_amount_eur), d_total))
            )

        self._notes.append(
            f"Bill decomposition: total={float(d_total):.2f}, "
            f"commodity={float(d_commodity):.2f} ({_round2(float(commodity_pct))}%), "
            f"non-commodity={float(d_non_commodity):.2f} ({_round2(float(non_commodity_pct))}%), "
            f"jurisdiction={jurisdiction.value}."
        )

        elapsed = (time.perf_counter() - t0) * 1000

        result = ChargeBreakdown(
            total_bill_eur=_round2(float(d_total)),
            commodity_cost_eur=_round2(float(d_commodity)),
            commodity_pct=_round2(float(commodity_pct)),
            non_commodity_cost_eur=_round2(float(d_non_commodity)),
            non_commodity_pct=_round2(float(non_commodity_pct)),
            charges=charges,
            taxes_eur=_round2(float(d_taxes)),
            taxes_pct=_round2(float(taxes_pct)),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "decompose_bill completed in %.2f ms, %d charges identified.",
            elapsed, len(charges),
        )
        return result

    # ------------------------------------------------------------------ #
    # Public -- Exemption Assessment
    # ------------------------------------------------------------------ #

    def assess_exemptions(
        self,
        facility: Dict[str, Any],
        jurisdiction: Jurisdiction,
    ) -> List[ExemptionAssessment]:
        """Assess facility eligibility for regulatory charge exemptions.

        Checks all applicable exemption types for the given jurisdiction
        against facility characteristics (energy intensity, self-supply
        ratio, CHP status, etc.).

        Args:
            facility: Dict with keys: annual_consumption_kwh,
                annual_production_value_eur, electricity_cost_share_pct,
                has_chp (bool), has_storage (bool), is_interruptible (bool),
                self_supply_pct, current_non_commodity_eur.
            jurisdiction: Regulatory jurisdiction.

        Returns:
            List of ExemptionAssessment for all relevant exemption types.
        """
        t0 = time.perf_counter()
        self._notes = []
        assessments: List[ExemptionAssessment] = []

        d_consumption = _decimal(facility.get("annual_consumption_kwh", 0))
        d_prod_value = _decimal(facility.get("annual_production_value_eur", 0))
        elec_cost_share = _decimal(facility.get("electricity_cost_share_pct", 0))
        has_chp = facility.get("has_chp", False)
        has_storage = facility.get("has_storage", False)
        is_interruptible = facility.get("is_interruptible", False)
        self_supply_pct = _decimal(facility.get("self_supply_pct", 0))
        current_nc = _decimal(facility.get("current_non_commodity_eur", 0))

        # Energy-Intensive Industry exemption (DE: StromNEV s19(2), BesAR).
        ei_threshold = Decimal("14") if jurisdiction == Jurisdiction.EU_DE else Decimal("20")
        ei_eligible = elec_cost_share >= ei_threshold and d_consumption >= Decimal("1000000")
        ei_savings = current_nc * Decimal("0.80") if ei_eligible else Decimal("0")
        assessments.append(ExemptionAssessment(
            exemption_type=ExemptionType.ENERGY_INTENSIVE,
            jurisdiction=jurisdiction.value,
            eligible=ei_eligible,
            eligibility_criteria=[
                f"Electricity cost share >= {float(ei_threshold)}% of production value",
                "Annual consumption >= 1 GWh",
                "Application to regulatory authority required",
            ],
            current_charge_eur=_round2(float(current_nc)),
            exempted_amount_eur=_round2(float(ei_savings)),
            savings_eur=_round2(float(ei_savings)),
            application_requirements=[
                "Certified energy audit (ISO 50001 or EN 16247)",
                "Financial statements showing electricity cost share",
                "Production volume documentation",
            ],
            deadline="June 30 of preceding year",
        ))

        # Self-supply exemption.
        ss_eligible = self_supply_pct >= Decimal("50")
        ss_reduction = Decimal("0.40") if ss_eligible else Decimal("0")
        ss_savings = current_nc * ss_reduction
        assessments.append(ExemptionAssessment(
            exemption_type=ExemptionType.SELF_SUPPLY,
            jurisdiction=jurisdiction.value,
            eligible=ss_eligible,
            eligibility_criteria=[
                "Self-supply ratio >= 50%",
                "Generation and consumption at same site",
                "Metered separation of self-supply and grid import",
            ],
            current_charge_eur=_round2(float(current_nc)),
            exempted_amount_eur=_round2(float(ss_savings)),
            savings_eur=_round2(float(ss_savings)),
            application_requirements=[
                "Metering concept approved by DSO",
                "Self-supply declaration with annual reconciliation",
            ],
        ))

        # CHP exemption (KWKG).
        chp_savings = current_nc * Decimal("0.30") if has_chp else Decimal("0")
        assessments.append(ExemptionAssessment(
            exemption_type=ExemptionType.CHP,
            jurisdiction=jurisdiction.value,
            eligible=has_chp,
            eligibility_criteria=[
                "Registered CHP plant at site",
                "CHP efficiency >= 70% (per EU CHP Directive)",
                "Annual reporting to regulatory authority",
            ],
            current_charge_eur=_round2(float(current_nc)),
            exempted_amount_eur=_round2(float(chp_savings)),
            savings_eur=_round2(float(chp_savings)),
            application_requirements=[
                "CHP plant registration certificate",
                "Efficiency test report from accredited body",
            ],
        ))

        # Storage exemption (loss-adjusted, EU Directive 2019/944 Art. 15).
        storage_savings = (
            current_nc * Decimal("0.20") if has_storage else Decimal("0")
        )
        assessments.append(ExemptionAssessment(
            exemption_type=ExemptionType.STORAGE,
            jurisdiction=jurisdiction.value,
            eligible=has_storage,
            eligibility_criteria=[
                "Registered electricity storage facility",
                "Storage not co-located with generation > 20 MW",
                "Double-charging avoidance under EU Directive 2019/944",
            ],
            current_charge_eur=_round2(float(current_nc)),
            exempted_amount_eur=_round2(float(storage_savings)),
            savings_eur=_round2(float(storage_savings)),
            application_requirements=[
                "Storage facility registration with TSO/DSO",
                "Metering arrangements for charge/discharge cycles",
            ],
        ))

        # Interruptible load (DE: AbLaV, UK: STOR/DSR).
        int_savings = (
            current_nc * Decimal("0.15") if is_interruptible else Decimal("0")
        )
        assessments.append(ExemptionAssessment(
            exemption_type=ExemptionType.INTERRUPTIBLE_LOAD,
            jurisdiction=jurisdiction.value,
            eligible=is_interruptible,
            eligibility_criteria=[
                "Minimum interruptible capacity >= 5 MW",
                "Response time <= 15 minutes (sofort) or <= 200 ms (sofort+)",
                "Availability >= 95% of contracted hours",
            ],
            current_charge_eur=_round2(float(current_nc)),
            exempted_amount_eur=_round2(float(int_savings)),
            savings_eur=_round2(float(int_savings)),
            application_requirements=[
                "Pre-qualification with TSO",
                "Automatic frequency response equipment installed",
                "Annual availability declaration",
            ],
            deadline="Quarterly auction schedule (DE AbLaV)",
        ))

        elapsed = (time.perf_counter() - t0) * 1000
        eligible_count = sum(1 for a in assessments if a.eligible)
        self._notes.append(
            f"Exemption assessment: {eligible_count}/{len(assessments)} eligible "
            f"in {jurisdiction.value}, completed in {elapsed:.2f} ms."
        )
        logger.info(
            "assess_exemptions completed in %.2f ms, %d/%d eligible.",
            elapsed, eligible_count, len(assessments),
        )
        return assessments

    # ------------------------------------------------------------------ #
    # Public -- Capacity Optimisation
    # ------------------------------------------------------------------ #

    def optimize_capacity(
        self,
        facility: Dict[str, Any],
        demand_profile: List[float],
        connection_agreement: Dict[str, Any],
    ) -> CapacityOptimization:
        """Optimise contracted capacity based on actual demand profile.

        Compares contracted capacity against measured peak demand with a
        configurable headroom buffer.  Recommends capacity reduction where
        significant over-contracting exists.

        Args:
            facility: Dict with keys: facility_id, contracted_capacity_kw.
            demand_profile: List of demand readings in kW (e.g. 8760 hourly).
            connection_agreement: Dict with keys: capacity_rate_eur_kw_month,
                penalty_rate_pct (penalty for exceeding contracted capacity),
                min_contract_kw (minimum contractable capacity),
                headroom_pct (safety margin, default 10%).

        Returns:
            CapacityOptimization result with savings and recommendation.
        """
        t0 = time.perf_counter()

        d_contracted = _decimal(facility.get("contracted_capacity_kw", 0))
        d_rate = _decimal(connection_agreement.get("capacity_rate_eur_kw_month", 0))
        d_penalty_rate = _decimal(connection_agreement.get("penalty_rate_pct", 150))
        d_min_contract = _decimal(connection_agreement.get("min_contract_kw", 0))
        headroom_pct = _decimal(connection_agreement.get("headroom_pct", 10))

        # Find actual max demand from profile.
        if demand_profile:
            actual_max = _decimal(max(demand_profile))
        else:
            actual_max = d_contracted
            logger.warning("Empty demand profile; using contracted capacity.")

        utilization = _safe_pct(actual_max, d_contracted)

        # Optimal capacity = max demand * (1 + headroom / 100).
        headroom_factor = Decimal("1") + headroom_pct / Decimal("100")
        optimal_raw = actual_max * headroom_factor
        # Do not go below minimum contract.
        optimal = max(optimal_raw, d_min_contract)

        # Annual savings.
        reduction = max(d_contracted - optimal, Decimal("0"))
        annual_savings = reduction * d_rate * Decimal("12")

        # Penalty risk assessment.
        buffer_pct = _safe_pct(optimal - actual_max, actual_max)
        if float(buffer_pct) >= 15:
            risk = "low"
        elif float(buffer_pct) >= 5:
            risk = "medium"
        else:
            risk = "high"

        # Recommendation text.
        if float(reduction) > 0:
            rec = (
                f"Reduce contracted capacity from {_round2(float(d_contracted))} kW "
                f"to {_round2(float(optimal))} kW (headroom {_round2(float(headroom_pct))}%). "
                f"Annual savings: EUR {_round2(float(annual_savings))}. "
                f"Penalty risk: {risk}."
            )
        else:
            rec = (
                "Current contracted capacity is at or below optimal level. "
                "No capacity reduction recommended."
            )

        elapsed = (time.perf_counter() - t0) * 1000
        self._notes.append(
            f"Capacity optimisation: contracted={float(d_contracted):.0f} kW, "
            f"max_demand={float(actual_max):.0f} kW, "
            f"optimal={float(optimal):.0f} kW, savings={float(annual_savings):.2f} EUR."
        )
        logger.info("optimize_capacity completed in %.2f ms.", elapsed)

        return CapacityOptimization(
            current_capacity_kw=_round2(float(d_contracted)),
            actual_max_demand_kw=_round2(float(actual_max)),
            utilization_pct=_round2(float(utilization)),
            optimal_capacity_kw=_round2(float(optimal)),
            annual_savings_eur=_round2(float(annual_savings)),
            penalty_risk=risk,
            recommendation=rec,
        )

    # ------------------------------------------------------------------ #
    # Public -- Power Factor Optimisation
    # ------------------------------------------------------------------ #

    def optimize_power_factor(
        self,
        demand_profile: List[float],
        current_pf: float,
        tariff: Dict[str, Any],
    ) -> PowerFactorOptimization:
        """Analyse power factor and size capacitor bank for correction.

        Calculates reactive power penalty under the applicable tariff,
        determines the target power factor, sizes the required capacitor
        bank, and computes payback.

        Args:
            demand_profile: List of active power demand readings in kW.
            current_pf: Current measured power factor (cos phi), 0-1.
            tariff: Dict with keys: jurisdiction (Jurisdiction value string),
                penalty_rate_eur_kvar_month (reactive power penalty rate),
                capacitor_cost_eur_per_kvar (installed cost per kVAR).

        Returns:
            PowerFactorOptimization with capacitor sizing and payback.
        """
        t0 = time.perf_counter()

        jurisdiction_str = tariff.get("jurisdiction", "default")
        threshold = _decimal(
            PF_PENALTY_THRESHOLDS.get(jurisdiction_str, PF_PENALTY_THRESHOLDS["default"])
        )
        d_pf = _decimal(current_pf)
        d_target = max(threshold + Decimal("0.03"), Decimal("0.95"))
        d_penalty_rate = _decimal(tariff.get("penalty_rate_eur_kvar_month", 1.50))
        d_cap_cost = _decimal(tariff.get("capacitor_cost_eur_per_kvar", 25.0))

        # Average active power from profile.
        if demand_profile:
            avg_kw = _decimal(sum(demand_profile) / len(demand_profile))
        else:
            avg_kw = Decimal("0")

        # Current reactive power: Q = P * tan(arccos(pf)).
        pf_float = float(d_pf)
        if pf_float <= 0 or pf_float > 1:
            pf_float = 0.85
            d_pf = Decimal("0.85")

        current_tan = _decimal(math.tan(math.acos(pf_float)))
        target_float = float(d_target)
        target_tan = _decimal(math.tan(math.acos(min(target_float, 0.9999))))

        current_kvar = avg_kw * current_tan
        target_kvar = avg_kw * target_tan

        # Excess kVAR above threshold.
        threshold_float = float(threshold)
        threshold_tan = _decimal(math.tan(math.acos(min(threshold_float, 0.9999))))
        excess_kvar = max(avg_kw * current_tan - avg_kw * threshold_tan, Decimal("0"))

        # Current annual penalty.
        current_penalty = excess_kvar * d_penalty_rate * Decimal("12")

        # Required capacitor bank: difference between current and target Q.
        required_kvar = max(current_kvar - target_kvar, Decimal("0"))
        capacitor_cost = required_kvar * d_cap_cost

        # Annual savings = full penalty elimination (target is above threshold).
        annual_savings = current_penalty

        # Payback in months.
        if annual_savings > Decimal("0"):
            payback_months = _safe_divide(
                capacitor_cost * Decimal("12"), annual_savings
            )
        else:
            payback_months = Decimal("0")

        elapsed = (time.perf_counter() - t0) * 1000
        self._notes.append(
            f"PF optimisation: current={pf_float:.3f}, threshold={threshold_float:.2f}, "
            f"target={target_float:.2f}, capacitor={float(required_kvar):.1f} kVAR, "
            f"savings={float(annual_savings):.2f} EUR/yr."
        )
        logger.info("optimize_power_factor completed in %.2f ms.", elapsed)

        return PowerFactorOptimization(
            current_pf=_round3(pf_float),
            penalty_threshold=_round3(threshold_float),
            current_penalty_eur=_round2(float(current_penalty)),
            target_pf=_round3(target_float),
            capacitor_kvar=_round2(float(required_kvar)),
            capacitor_cost_eur=_round2(float(capacitor_cost)),
            annual_savings_eur=_round2(float(annual_savings)),
            payback_months=_round2(float(payback_months)),
        )

    # ------------------------------------------------------------------ #
    # Public -- Voltage Level Analysis
    # ------------------------------------------------------------------ #

    def analyze_voltage_level(
        self,
        facility: Dict[str, Any],
        current_voltage: VoltageLevel,
    ) -> VoltageLevelAnalysis:
        """Analyse economics of migrating to a higher voltage connection.

        Higher voltage connections attract lower per-kWh network charges.
        This method quantifies the savings against the capital cost of
        installing customer-owned transformation equipment.

        Args:
            facility: Dict with keys: annual_consumption_kwh,
                base_network_rate_eur_kwh (LV rate).
            current_voltage: Current connection voltage level.

        Returns:
            VoltageLevelAnalysis with savings and payback.
        """
        t0 = time.perf_counter()

        d_consumption = _decimal(facility.get("annual_consumption_kwh", 0))
        d_base_rate = _decimal(facility.get("base_network_rate_eur_kwh", 0.075))

        # Current effective rate.
        current_factor = _decimal(VOLTAGE_LEVEL_FACTORS[current_voltage])
        current_rate = d_base_rate * current_factor

        # Determine next higher voltage level.
        voltage_order = [
            VoltageLevel.LOW_VOLTAGE,
            VoltageLevel.MEDIUM_VOLTAGE,
            VoltageLevel.HIGH_VOLTAGE,
            VoltageLevel.EXTRA_HIGH_VOLTAGE,
        ]
        current_idx = voltage_order.index(current_voltage)
        if current_idx >= len(voltage_order) - 1:
            # Already at highest level.
            logger.info("Facility already at EXTRA_HIGH_VOLTAGE; no upgrade possible.")
            return VoltageLevelAnalysis(
                current_voltage=current_voltage,
                proposed_voltage=current_voltage,
                current_rate_per_kwh=_round4(float(current_rate)),
                proposed_rate_per_kwh=_round4(float(current_rate)),
                transformation_cost_eur=0.0,
                annual_savings_eur=0.0,
                payback_years=0.0,
                feasibility="infeasible",
            )

        proposed_voltage = voltage_order[current_idx + 1]
        proposed_factor = _decimal(VOLTAGE_LEVEL_FACTORS[proposed_voltage])
        proposed_rate = d_base_rate * proposed_factor

        # Annual savings.
        savings = (current_rate - proposed_rate) * d_consumption

        # Transformation capital cost (reference costs, EUR).
        transformation_costs: Dict[str, Decimal] = {
            "low_voltage->medium_voltage": Decimal("150000"),
            "medium_voltage->high_voltage": Decimal("800000"),
            "high_voltage->extra_high_voltage": Decimal("3500000"),
        }
        cost_key = f"{current_voltage.value}->{proposed_voltage.value}"
        capex = transformation_costs.get(cost_key, Decimal("250000"))

        # Payback.
        payback_years = _safe_divide(capex, savings)

        # Feasibility assessment.
        payback_float = float(payback_years)
        if payback_float <= 5:
            feasibility = "feasible"
        elif payback_float <= 10:
            feasibility = "marginal"
        else:
            feasibility = "infeasible"

        elapsed = (time.perf_counter() - t0) * 1000
        self._notes.append(
            f"Voltage analysis: {current_voltage.value} -> {proposed_voltage.value}, "
            f"savings={float(savings):.2f} EUR/yr, payback={payback_float:.1f} yr, "
            f"feasibility={feasibility}."
        )
        logger.info("analyze_voltage_level completed in %.2f ms.", elapsed)

        return VoltageLevelAnalysis(
            current_voltage=current_voltage,
            proposed_voltage=proposed_voltage,
            current_rate_per_kwh=_round4(float(current_rate)),
            proposed_rate_per_kwh=_round4(float(proposed_rate)),
            transformation_cost_eur=_round2(float(capex)),
            annual_savings_eur=_round2(float(savings)),
            payback_years=_round2(payback_float),
            feasibility=feasibility,
        )

    # ------------------------------------------------------------------ #
    # Public -- Self-Generation Impact
    # ------------------------------------------------------------------ #

    def analyze_self_generation_impact(
        self,
        facility: Dict[str, Any],
        generation_profile: List[float],
        tariff: Dict[str, Any],
    ) -> SelfGenerationImpact:
        """Quantify impact of on-site generation on non-commodity charges.

        Self-generation reduces grid import and therefore avoids volumetric
        network charges, levies, and taxes.  However, standby and backup
        charges may apply for the retained grid connection.

        Args:
            facility: Dict with keys: annual_consumption_kwh,
                current_non_commodity_eur, contracted_capacity_kw.
            generation_profile: Hourly (or sub-hourly) generation in kWh.
            tariff: Dict with keys: non_commodity_rate_eur_kwh
                (blended non-commodity rate), standby_rate_eur_kw_month,
                backup_pct (fraction of contracted capacity charged for backup).

        Returns:
            SelfGenerationImpact with avoided and residual charges.
        """
        t0 = time.perf_counter()

        d_consumption = _decimal(facility.get("annual_consumption_kwh", 0))
        d_nc_total = _decimal(facility.get("current_non_commodity_eur", 0))
        d_capacity = _decimal(facility.get("contracted_capacity_kw", 0))

        d_nc_rate = _decimal(tariff.get("non_commodity_rate_eur_kwh", 0))
        d_standby_rate = _decimal(tariff.get("standby_rate_eur_kw_month", 2.50))
        d_backup_pct = _decimal(tariff.get("backup_pct", 0.50))

        # Total generation.
        d_generation = _decimal(sum(generation_profile)) if generation_profile else Decimal("0")

        # If no explicit NC rate, derive from total NC / consumption.
        if d_nc_rate == Decimal("0") and d_consumption > Decimal("0"):
            d_nc_rate = _safe_divide(d_nc_total, d_consumption)

        # Avoided grid charges = generation * blended NC rate.
        # Capped at actual grid import reduction (cannot exceed consumption).
        effective_gen = min(d_generation, d_consumption)
        avoided = effective_gen * d_nc_rate

        # Remaining grid charges on imported electricity.
        remaining_import = max(d_consumption - effective_gen, Decimal("0"))
        remaining_charges = remaining_import * d_nc_rate

        # Standby charges: apply to full contracted capacity.
        standby = d_capacity * d_standby_rate * Decimal("12")

        # Backup charges: fraction of capacity.
        backup = d_capacity * d_backup_pct * d_standby_rate * Decimal("12")

        # Net savings.
        net_savings = avoided - standby - backup

        elapsed = (time.perf_counter() - t0) * 1000
        self._notes.append(
            f"Self-generation impact: generation={float(d_generation):.0f} kWh, "
            f"avoided={float(avoided):.2f} EUR, standby={float(standby):.2f} EUR, "
            f"net={float(net_savings):.2f} EUR."
        )
        logger.info(
            "analyze_self_generation_impact completed in %.2f ms.", elapsed
        )

        return SelfGenerationImpact(
            generation_kwh=_round2(float(d_generation)),
            avoided_grid_charges_eur=_round2(float(avoided)),
            remaining_grid_charges_eur=_round2(float(remaining_charges)),
            net_savings_eur=_round2(float(net_savings)),
            standby_charges_eur=_round2(float(standby)),
            backup_charges_eur=_round2(float(backup)),
        )

    # ------------------------------------------------------------------ #
    # Public -- Charge Projections
    # ------------------------------------------------------------------ #

    def project_charges(
        self,
        current_charges: List[RegulatoryCharge],
        jurisdiction: Jurisdiction,
        years: int = 5,
    ) -> List[GridChargeProjection]:
        """Project non-commodity charges over multiple years.

        Applies jurisdiction-specific annual growth rates by charge type
        to produce forward cost estimates for budgeting and strategy.

        Args:
            current_charges: Current year's charge breakdown.
            jurisdiction: Regulatory jurisdiction for growth rate lookup.
            years: Number of years to project forward.

        Returns:
            List of GridChargeProjection, one per year.
        """
        t0 = time.perf_counter()
        projections: List[GridChargeProjection] = []

        # Look up growth rates.
        growth_map = CHARGE_GROWTH_RATES.get(
            jurisdiction, CHARGE_GROWTH_RATES["default"]
        )
        default_growth = _decimal(growth_map.get("default", 2.0))

        # Current year total for change calculation.
        current_total = sum(
            _decimal(c.annual_amount_eur) for c in current_charges
        )
        current_year = utcnow().year

        for y_offset in range(1, years + 1):
            proj_year = current_year + y_offset
            projected_by_type: Dict[str, float] = {}
            total_projected = Decimal("0")
            drivers: List[str] = []

            for charge in current_charges:
                # Get type-specific growth rate.
                growth_rate = _decimal(
                    growth_map.get(charge.charge_type, float(default_growth))
                )
                # Compound growth: amount * (1 + rate/100)^years.
                growth_factor = (
                    Decimal("1") + growth_rate / Decimal("100")
                ) ** y_offset
                projected_amount = _decimal(charge.annual_amount_eur) * growth_factor

                type_key = charge.charge_type.value
                if type_key in projected_by_type:
                    projected_by_type[type_key] += _round2(float(projected_amount))
                else:
                    projected_by_type[type_key] = _round2(float(projected_amount))
                total_projected += projected_amount

                # Record high-growth drivers.
                if float(growth_rate) >= 3.0 and charge.name not in drivers:
                    drivers.append(
                        f"{charge.name}: +{_round2(float(growth_rate))}%/yr"
                    )

            change_pct = _safe_pct(
                total_projected - current_total, current_total
            )

            projections.append(GridChargeProjection(
                year=proj_year,
                projected_charges_by_type=projected_by_type,
                total_projected_eur=_round2(float(total_projected)),
                change_vs_current_pct=_round2(float(change_pct)),
                regulatory_drivers=drivers,
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        self._notes.append(
            f"Charge projection: {years} years from {current_year}, "
            f"jurisdiction={jurisdiction.value}, completed in {elapsed:.2f} ms."
        )
        logger.info(
            "project_charges completed in %.2f ms, %d years projected.",
            elapsed, years,
        )
        return projections

    # ------------------------------------------------------------------ #
    # Public -- Full Optimisation
    # ------------------------------------------------------------------ #

    def full_optimization(
        self,
        facility: Dict[str, Any],
        bills: List[Dict[str, Any]],
        demand_profile: List[float],
        jurisdiction: Jurisdiction,
    ) -> ChargeOptimizationResult:
        """Run comprehensive non-commodity charge optimisation.

        Orchestrates bill decomposition, exemption assessment, capacity
        optimisation, power-factor correction, voltage-level analysis,
        and multi-year projections into a single result.

        Args:
            facility: Facility data dict (see sub-method docs for keys).
            bills: List of bill data dicts for decomposition.
            demand_profile: Hourly demand readings in kW.
            jurisdiction: Regulatory jurisdiction.

        Returns:
            ChargeOptimizationResult with all optimisation findings.
        """
        t0 = time.perf_counter()
        self._notes = []
        actions: List[str] = []
        total_savings = Decimal("0")

        facility_id = facility.get("facility_id", _new_uuid())

        # Step 1: Decompose the most recent bill.
        latest_bill = bills[-1] if bills else {}
        breakdown = self.decompose_bill(latest_bill, jurisdiction)
        d_current_nc = _decimal(breakdown.non_commodity_cost_eur)

        # Step 2: Assess exemptions.
        facility_for_exemptions = {**facility, "current_non_commodity_eur": float(d_current_nc)}
        exemptions = self.assess_exemptions(facility_for_exemptions, jurisdiction)
        eligible_exemptions = [e for e in exemptions if e.eligible]
        exemption_savings = sum(_decimal(e.savings_eur) for e in eligible_exemptions)
        if float(exemption_savings) > 0:
            actions.append(OptimizationAction.EXEMPTION_APPLICATION.value)
            total_savings += exemption_savings

        # Step 3: Capacity optimisation.
        connection_agreement = facility.get("connection_agreement", {
            "capacity_rate_eur_kw_month": 8.0,
            "penalty_rate_pct": 150,
            "min_contract_kw": 0,
            "headroom_pct": 10,
        })
        cap_opt = self.optimize_capacity(facility, demand_profile, connection_agreement)
        d_cap_savings = _decimal(cap_opt.annual_savings_eur)
        if float(d_cap_savings) > 0:
            actions.append(OptimizationAction.CAPACITY_REDUCTION.value)
            total_savings += d_cap_savings

        # Step 4: Power factor optimisation.
        current_pf = facility.get("current_pf", 0.85)
        pf_tariff = facility.get("pf_tariff", {
            "jurisdiction": jurisdiction.value,
            "penalty_rate_eur_kvar_month": 1.50,
            "capacitor_cost_eur_per_kvar": 25.0,
        })
        pf_opt = self.optimize_power_factor(demand_profile, current_pf, pf_tariff)
        d_pf_savings = _decimal(pf_opt.annual_savings_eur)
        if float(d_pf_savings) > 0:
            actions.append(OptimizationAction.PF_CORRECTION.value)
            total_savings += d_pf_savings

        # Step 5: Voltage level analysis.
        current_voltage = facility.get("voltage_level", VoltageLevel.LOW_VOLTAGE)
        if isinstance(current_voltage, str):
            try:
                current_voltage = VoltageLevel(current_voltage)
            except ValueError:
                current_voltage = VoltageLevel.LOW_VOLTAGE
        voltage_analysis = self.analyze_voltage_level(facility, current_voltage)
        if voltage_analysis.feasibility == "feasible":
            d_v_savings = _decimal(voltage_analysis.annual_savings_eur)
            if float(d_v_savings) > 0:
                actions.append(OptimizationAction.VOLTAGE_UPGRADE.value)
                total_savings += d_v_savings

        # Step 6: Project charges forward.
        projections = self.project_charges(breakdown.charges, jurisdiction, years=5)

        # Step 7: Compile result.
        optimised_nc = max(d_current_nc - total_savings, Decimal("0"))
        savings_pct = _safe_pct(total_savings, d_current_nc)

        elapsed = (time.perf_counter() - t0) * 1000

        result = ChargeOptimizationResult(
            facility_id=facility_id,
            current_non_commodity_eur=_round2(float(d_current_nc)),
            optimized_non_commodity_eur=_round2(float(optimised_nc)),
            total_savings_eur=_round2(float(total_savings)),
            savings_pct=_round2(float(savings_pct)),
            actions=actions,
            exemptions=exemptions,
            capacity_optimization=cap_opt,
            pf_optimization=pf_opt,
            voltage_analysis=voltage_analysis,
            projections=projections,
            processing_time_ms=_round2(elapsed),
        )
        result.provenance_hash = _compute_hash(result)

        self._notes.append(
            f"Full optimisation: current NC={float(d_current_nc):.2f} EUR, "
            f"savings={float(total_savings):.2f} EUR ({float(savings_pct):.1f}%), "
            f"{len(actions)} actions recommended."
        )
        logger.info(
            "full_optimization completed in %.2f ms. "
            "Current NC: %.2f EUR, savings: %.2f EUR (%.1f%%).",
            elapsed, float(d_current_nc), float(total_savings),
            float(savings_pct),
        )
        return result

    # ------------------------------------------------------------------ #
    # Public -- Jurisdiction Comparison
    # ------------------------------------------------------------------ #

    def compare_jurisdictions(
        self,
        facility_profile: Dict[str, Any],
        jurisdictions: List[Jurisdiction],
    ) -> List[ChargeBreakdown]:
        """Compare non-commodity charge structures across jurisdictions.

        For a given facility consumption profile, computes the indicative
        non-commodity charges in each jurisdiction using reference charge
        components and rates.

        Args:
            facility_profile: Dict with keys: annual_consumption_kwh,
                contracted_capacity_kw, total_bill_eur (estimated),
                voltage_level.
            jurisdictions: List of jurisdictions to compare.

        Returns:
            List of ChargeBreakdown, one per jurisdiction, sorted by
            non-commodity cost ascending.
        """
        t0 = time.perf_counter()
        results: List[ChargeBreakdown] = []

        for jur in jurisdictions:
            # Estimate commodity cost from typical NC share.
            nc_share = _decimal(
                NON_COMMODITY_SHARE_PCT.get(jur, 45.0)
            ) / Decimal("100")

            d_total = _decimal(facility_profile.get("total_bill_eur", 0))
            if d_total > Decimal("0"):
                d_commodity = d_total * (Decimal("1") - nc_share)
            else:
                # Estimate total from consumption and average all-in price.
                d_consumption = _decimal(
                    facility_profile.get("annual_consumption_kwh", 0)
                )
                # Assume EUR 0.20/kWh all-in for estimation.
                d_total = d_consumption * Decimal("0.20")
                d_commodity = d_total * (Decimal("1") - nc_share)

            bill_data = {
                "total_bill_eur": float(d_total),
                "commodity_cost_eur": float(d_commodity),
                "annual_consumption_kwh": facility_profile.get(
                    "annual_consumption_kwh", 0
                ),
                "contracted_capacity_kw": facility_profile.get(
                    "contracted_capacity_kw", 0
                ),
                "voltage_level": facility_profile.get(
                    "voltage_level", VoltageLevel.LOW_VOLTAGE.value
                ),
            }
            breakdown = self.decompose_bill(bill_data, jur)
            results.append(breakdown)

        # Sort by non-commodity cost ascending.
        results.sort(key=lambda b: b.non_commodity_cost_eur)

        elapsed = (time.perf_counter() - t0) * 1000
        self._notes.append(
            f"Jurisdiction comparison: {len(jurisdictions)} jurisdictions "
            f"compared in {elapsed:.2f} ms."
        )
        logger.info(
            "compare_jurisdictions completed in %.2f ms, %d jurisdictions.",
            elapsed, len(jurisdictions),
        )
        return results

    # ------------------------------------------------------------------ #
    # Private -- Build Charge List
    # ------------------------------------------------------------------ #

    def _build_charge_list(
        self,
        jurisdiction: Jurisdiction,
        consumption: Decimal,
        capacity: Decimal,
        non_commodity_total: Decimal,
        voltage: VoltageLevel,
    ) -> List[RegulatoryCharge]:
        """Build itemised charge list for a jurisdiction.

        Routes to jurisdiction-specific builders for DE, UK, and US ISOs.
        For other jurisdictions, produces a generic charge split based
        on typical proportions.

        Args:
            jurisdiction: Target jurisdiction.
            consumption: Annual consumption in kWh.
            capacity: Contracted capacity in kW.
            non_commodity_total: Total non-commodity amount in EUR.
            voltage: Connection voltage level.

        Returns:
            List of RegulatoryCharge items.
        """
        if jurisdiction == Jurisdiction.EU_DE:
            return self._build_de_charges(consumption, capacity, voltage)
        elif jurisdiction == Jurisdiction.UK_GB:
            return self._build_uk_charges(consumption, capacity, voltage)
        elif jurisdiction.value.startswith("us_"):
            return self._build_us_charges(jurisdiction, consumption, capacity)
        else:
            return self._build_generic_charges(
                jurisdiction, consumption, non_commodity_total
            )

    def _build_de_charges(
        self,
        consumption: Decimal,
        capacity: Decimal,
        voltage: VoltageLevel,
    ) -> List[RegulatoryCharge]:
        """Build German electricity non-commodity charge items.

        Source: BNetzA Strompreisanalyse, StromNEV, KWKG, EnWG.
        """
        charges: List[RegulatoryCharge] = []
        ct_to_eur = Decimal("0.01")  # ct/kWh -> EUR/kWh.

        # Netzentgelte (network charges) - voltage dependent.
        netz = DE_CHARGE_COMPONENTS["netzentgelte"]
        rate_key = {
            VoltageLevel.LOW_VOLTAGE: "rate_ct_kwh_lv",
            VoltageLevel.MEDIUM_VOLTAGE: "rate_ct_kwh_mv",
            VoltageLevel.HIGH_VOLTAGE: "rate_ct_kwh_hv",
            VoltageLevel.EXTRA_HIGH_VOLTAGE: "rate_ct_kwh_ehv",
        }
        netz_rate = _decimal(netz[rate_key[voltage]]) * ct_to_eur
        netz_amount = netz_rate * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.DISTRIBUTION,
            name="Netzentgelte",
            methodology=ChargeMethodology.CAPACITY_BASED,
            rate=_round4(float(netz_rate)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(netz_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.EU_DE.value,
            description=f"Network charges ({voltage.value}), {netz['source']}",
            optimizable=True,
        ))

        # KWK-Umlage.
        kwk = DE_CHARGE_COMPONENTS["kwk_umlage"]
        kwk_rate = _decimal(kwk["rate_ct_kwh"]) * ct_to_eur
        kwk_amount = kwk_rate * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.RENEWABLE_LEVY,
            name="KWK-Umlage",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(kwk_rate)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(kwk_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.EU_DE.value,
            description=kwk["source"],
            optimizable=True,
        ))

        # Offshore-Netzumlage.
        off = DE_CHARGE_COMPONENTS["offshore_netzumlage"]
        off_rate = _decimal(off["rate_ct_kwh"]) * ct_to_eur
        off_amount = off_rate * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.SYSTEM,
            name="Offshore-Netzumlage",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(off_rate)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(off_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.EU_DE.value,
            description=off["source"],
            optimizable=False,
        ))

        # StromNEV-Umlage.
        snev = DE_CHARGE_COMPONENTS["stromnev_umlage"]
        snev_rate = _decimal(snev["rate_ct_kwh"]) * ct_to_eur
        snev_amount = snev_rate * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.DISTRIBUTION,
            name="StromNEV-Umlage",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(snev_rate)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(snev_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.EU_DE.value,
            description=snev["source"],
            optimizable=True,
        ))

        # Konzessionsabgabe.
        konz = DE_CHARGE_COMPONENTS["konzessionsabgabe"]
        konz_rate = _decimal(konz["rate_ct_kwh"]) * ct_to_eur
        konz_amount = konz_rate * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.PUBLIC_PURPOSE,
            name="Konzessionsabgabe",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(konz_rate)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(konz_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.EU_DE.value,
            description=konz["source"],
            optimizable=False,
        ))

        # Stromsteuer.
        strom = DE_CHARGE_COMPONENTS["stromsteuer"]
        strom_rate = _decimal(strom["rate_ct_kwh"]) * ct_to_eur
        strom_amount = strom_rate * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.TAX,
            name="Stromsteuer",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(strom_rate)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(strom_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.EU_DE.value,
            description=strom["source"],
            optimizable=True,
        ))

        return charges

    def _build_uk_charges(
        self,
        consumption: Decimal,
        capacity: Decimal,
        voltage: VoltageLevel,
    ) -> List[RegulatoryCharge]:
        """Build UK electricity non-commodity charge items.

        Source: Ofgem DCUSA, CUSC, HMRC, LCCC published tariffs.
        Amounts converted to EUR at GBP/EUR = 1.16 (ECB reference rate).
        """
        charges: List[RegulatoryCharge] = []
        gbp_eur = Decimal("1.16")
        p_to_gbp = Decimal("0.01")  # p/kWh -> GBP/kWh.

        # DUoS - Distribution Use of System.
        duos = UK_CHARGE_COMPONENTS["duos"]
        rate_key = {
            VoltageLevel.LOW_VOLTAGE: "rate_p_kwh_lv",
            VoltageLevel.MEDIUM_VOLTAGE: "rate_p_kwh_lv",
            VoltageLevel.HIGH_VOLTAGE: "rate_p_kwh_hv",
            VoltageLevel.EXTRA_HIGH_VOLTAGE: "rate_p_kwh_ehv",
        }
        duos_rate_gbp = _decimal(duos[rate_key[voltage]]) * p_to_gbp
        duos_rate_eur = duos_rate_gbp * gbp_eur
        duos_amount = duos_rate_eur * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.DISTRIBUTION,
            name="DUoS",
            methodology=ChargeMethodology.CAPACITY_BASED,
            rate=_round4(float(duos_rate_eur)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(duos_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.UK_GB.value,
            description=f"Distribution Use of System ({voltage.value}), {duos['source']}",
            optimizable=True,
        ))

        # TNUoS - Transmission Network Use of System.
        tnuos = UK_CHARGE_COMPONENTS["tnuos"]
        tnuos_rate_gbp = _decimal(tnuos["rate_gbp_kw"])
        tnuos_annual_gbp = tnuos_rate_gbp * capacity
        tnuos_annual_eur = tnuos_annual_gbp * gbp_eur
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.TRANSMISSION,
            name="TNUoS",
            methodology=ChargeMethodology.COINCIDENT_PEAK,
            rate=_round2(float(tnuos_rate_gbp * gbp_eur)),
            unit="per_kw",
            annual_amount_eur=_round2(float(tnuos_annual_eur)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.UK_GB.value,
            description=f"Transmission Network Use of System (Triad), {tnuos['source']}",
            optimizable=True,
        ))

        # BSUoS - Balancing Services Use of System.
        bsuos = UK_CHARGE_COMPONENTS["bsuos"]
        bsuos_rate_eur = _decimal(bsuos["rate_p_kwh"]) * p_to_gbp * gbp_eur
        bsuos_amount = bsuos_rate_eur * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.BALANCING,
            name="BSUoS",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(bsuos_rate_eur)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(bsuos_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.UK_GB.value,
            description=f"Balancing Services, {bsuos['source']}",
            optimizable=False,
        ))

        # CCL - Climate Change Levy.
        ccl = UK_CHARGE_COMPONENTS["ccl"]
        ccl_rate_eur = _decimal(ccl["rate_p_kwh"]) * p_to_gbp * gbp_eur
        ccl_amount = ccl_rate_eur * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.CLIMATE_LEVY,
            name="CCL",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(ccl_rate_eur)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(ccl_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.UK_GB.value,
            description=f"Climate Change Levy, {ccl['source']} (CCA discount {ccl['cca_discount_pct']}%)",
            optimizable=True,
        ))

        # RO - Renewables Obligation.
        ro = UK_CHARGE_COMPONENTS["ro"]
        ro_rate_eur = _decimal(ro["rate_p_kwh"]) * p_to_gbp * gbp_eur
        ro_amount = ro_rate_eur * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.RENEWABLE_LEVY,
            name="RO",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(ro_rate_eur)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(ro_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.UK_GB.value,
            description=f"Renewables Obligation, {ro['source']}",
            optimizable=False,
        ))

        # CFD - Contracts for Difference.
        cfd = UK_CHARGE_COMPONENTS["cfd"]
        cfd_rate_eur = _decimal(cfd["rate_p_kwh"]) * p_to_gbp * gbp_eur
        cfd_amount = cfd_rate_eur * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.RENEWABLE_LEVY,
            name="CFD",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(cfd_rate_eur)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(cfd_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.UK_GB.value,
            description=f"Contracts for Difference, {cfd['source']}",
            optimizable=False,
        ))

        # CM - Capacity Market.
        cm = UK_CHARGE_COMPONENTS["cm"]
        cm_rate_eur = _decimal(cm["rate_p_kwh"]) * p_to_gbp * gbp_eur
        cm_amount = cm_rate_eur * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.CAPACITY,
            name="CM",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(cm_rate_eur)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(cm_amount)),
            share_of_total_bill_pct=0.0,
            jurisdiction=Jurisdiction.UK_GB.value,
            description=f"Capacity Market, {cm['source']}",
            optimizable=False,
        ))

        return charges

    def _build_us_charges(
        self,
        jurisdiction: Jurisdiction,
        consumption: Decimal,
        capacity: Decimal,
    ) -> List[RegulatoryCharge]:
        """Build US ISO/RTO non-commodity charge items.

        Source: ISO/RTO OATT tariff schedules, FERC filings.
        """
        charges: List[RegulatoryCharge] = []
        iso_key = jurisdiction.value
        components = US_CHARGE_COMPONENTS.get(iso_key, US_CHARGE_COMPONENTS.get("us_pjm", {}))

        # Transmission charge ($/kW-month).
        trans_rate = _decimal(components.get("transmission_usd_kw_month", 6.0))
        trans_annual = trans_rate * capacity * Decimal("12")
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.TRANSMISSION,
            name="Transmission",
            methodology=ChargeMethodology.CAPACITY_BASED,
            rate=_round2(float(trans_rate)),
            unit="per_kw",
            annual_amount_eur=_round2(float(trans_annual)),
            share_of_total_bill_pct=0.0,
            jurisdiction=jurisdiction.value,
            description=f"Transmission charge, {components.get('source', 'ISO tariff')}",
            optimizable=True,
        ))

        # Ancillary services ($/MWh -> $/kWh).
        anc_rate_mwh = _decimal(components.get("ancillary_usd_mwh", 3.0))
        anc_rate_kwh = anc_rate_mwh / Decimal("1000")
        anc_annual = anc_rate_kwh * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.ANCILLARY,
            name="Ancillary Services",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(anc_rate_kwh)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(anc_annual)),
            share_of_total_bill_pct=0.0,
            jurisdiction=jurisdiction.value,
            description=f"Ancillary services, {components.get('source', 'ISO tariff')}",
            optimizable=False,
        ))

        # Capacity charge ($/kW-month).
        cap_rate = _decimal(components.get("capacity_usd_kw_month", 5.0))
        cap_annual = cap_rate * capacity * Decimal("12")
        if float(cap_rate) > 0:
            charges.append(RegulatoryCharge(
                charge_type=ChargeType.CAPACITY,
                name="Capacity",
                methodology=ChargeMethodology.CAPACITY_BASED,
                rate=_round2(float(cap_rate)),
                unit="per_kw",
                annual_amount_eur=_round2(float(cap_annual)),
                share_of_total_bill_pct=0.0,
                jurisdiction=jurisdiction.value,
                description=f"Capacity charge, {components.get('source', 'ISO tariff')}",
                optimizable=True,
            ))

        # RPS / renewable surcharge ($/MWh -> $/kWh).
        rps_rate_mwh = _decimal(components.get("rps_usd_mwh", 2.0))
        rps_rate_kwh = rps_rate_mwh / Decimal("1000")
        rps_annual = rps_rate_kwh * consumption
        charges.append(RegulatoryCharge(
            charge_type=ChargeType.RENEWABLE_LEVY,
            name="RPS Surcharge",
            methodology=ChargeMethodology.VOLUMETRIC,
            rate=_round4(float(rps_rate_kwh)),
            unit="per_kwh",
            annual_amount_eur=_round2(float(rps_annual)),
            share_of_total_bill_pct=0.0,
            jurisdiction=jurisdiction.value,
            description=f"Renewable Portfolio Standard, {components.get('source', 'state tariff')}",
            optimizable=False,
        ))

        return charges

    def _build_generic_charges(
        self,
        jurisdiction: Jurisdiction,
        consumption: Decimal,
        non_commodity_total: Decimal,
    ) -> List[RegulatoryCharge]:
        """Build generic charge split for jurisdictions without specific data.

        Allocates the non-commodity total across typical charge categories
        using representative proportions from ACER market monitoring data.

        Args:
            jurisdiction: Target jurisdiction.
            consumption: Annual consumption in kWh.
            non_commodity_total: Total non-commodity amount in EUR.

        Returns:
            List of RegulatoryCharge items with proportional allocation.
        """
        # Typical proportions (from ACER Market Monitoring Report 2025).
        proportions: List[Tuple[ChargeType, str, float, bool]] = [
            (ChargeType.DISTRIBUTION, "Distribution Charges", 0.35, True),
            (ChargeType.TRANSMISSION, "Transmission Charges", 0.15, True),
            (ChargeType.RENEWABLE_LEVY, "Renewable Surcharge", 0.15, False),
            (ChargeType.TAX, "Electricity Tax", 0.12, True),
            (ChargeType.CAPACITY, "Capacity Charge", 0.08, True),
            (ChargeType.CLIMATE_LEVY, "Climate Levy", 0.05, False),
            (ChargeType.SYSTEM, "System Charges", 0.05, False),
            (ChargeType.METER, "Metering Charges", 0.03, False),
            (ChargeType.PUBLIC_PURPOSE, "Public Purpose Charge", 0.02, False),
        ]

        charges: List[RegulatoryCharge] = []
        for ctype, name, proportion, optimizable in proportions:
            amount = non_commodity_total * _decimal(proportion)
            rate = _safe_divide(amount, consumption)
            charges.append(RegulatoryCharge(
                charge_type=ctype,
                name=name,
                methodology=ChargeMethodology.VOLUMETRIC,
                rate=_round4(float(rate)),
                unit="per_kwh",
                annual_amount_eur=_round2(float(amount)),
                share_of_total_bill_pct=0.0,
                jurisdiction=jurisdiction.value,
                description=f"Estimated {name.lower()}, {jurisdiction.value}",
                optimizable=optimizable,
            ))

        return charges

    # ------------------------------------------------------------------ #
    # Public -- Audit Trail
    # ------------------------------------------------------------------ #

    def get_processing_notes(self) -> List[str]:
        """Return accumulated processing notes for audit trail.

        Returns:
            List of human-readable processing notes.
        """
        return list(self._notes)
