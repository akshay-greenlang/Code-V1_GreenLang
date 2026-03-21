# -*- coding: utf-8 -*-
"""
EUICalculatorEngine - PACK-035 Energy Benchmark Engine 1
=========================================================

Calculates Energy Use Intensity (EUI) for buildings and facilities in
kWh/m2/yr across multiple accounting boundaries (site energy, source
energy, primary energy, cost-normalised).  Supports rolling 12-month
calculations, multi-carrier aggregation, floor-area type conversions,
occupancy normalisation, and activity-based normalisation.

EUI Calculation:
    EUI = Total_Annual_Energy_kWh / Floor_Area_m2
    Source EUI = sum( carrier_kWh * source_factor ) / Floor_Area_m2
    Primary EUI = sum( carrier_kWh * primary_factor ) / Floor_Area_m2
    Cost-Normalised EUI = EUI * (occupancy_adj) * (activity_adj)

Source Energy Factors:
    Electricity: 2.80 (U.S. avg site-to-source, ENERGY STAR Technical Reference 2023)
    Natural Gas: 1.047 (site-to-source, ENERGY STAR Technical Reference)
    Fuel Oil:    1.01 (ENERGY STAR Technical Reference)
    District Heating: 1.20 (EU avg, EN 15603:2008)
    District Cooling:  1.30 (EU avg, EN 15603:2008)

Primary Energy Factors:
    Per EN 15603:2008, Annex A, and national adaptations.
    Electricity: 2.50 (non-renewable primary, EU weighted avg)
    Natural Gas: 1.10 (non-renewable primary)
    Fuel Oil:    1.10 (non-renewable primary)
    District Heating: 0.80 (assumes CHP)
    Biomass:     0.20 (renewable primary)

Floor Area Types:
    GIA (Gross Internal Area)  - total enclosed floor area
    NIA (Net Internal Area)    - GIA minus structural elements
    GLA (Gross Lettable Area)  - retail/commercial lettable area
    TFA (Treated Floor Area)   - area within thermal envelope

Regulatory References:
    - ENERGY STAR Portfolio Manager Technical Reference (2023)
    - EN 15603:2008 Energy performance of buildings
    - ISO 52000-1:2017 Energy performance of buildings
    - CIBSE TM46:2008 Energy benchmarks
    - ASHRAE Standard 100-2018 Energy Efficiency in Buildings
    - EU Directive 2010/31/EU (EPBD recast)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Source/primary energy factors from published references
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-035 Energy Benchmark
Engine:  1 of 10
Status:  Production Ready
"""

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


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EnergyCarrier(str, Enum):
    """Energy carrier types for EUI calculations.

    Covers all major energy carriers per ENERGY STAR Portfolio Manager
    and EN 15603 energy performance of buildings.
    """
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    LPG = "lpg"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    BIOMASS = "biomass"
    SOLAR_THERMAL = "solar_thermal"


class FloorAreaType(str, Enum):
    """Floor area measurement types per RICS/ISO 9836.

    GIA: Gross Internal Area - total enclosed floor area measured to the
         internal face of external walls (RICS Code of Measuring Practice).
    NIA: Net Internal Area - GIA minus structural elements, service risers,
         and common areas (RICS IPMS 3).
    GLA: Gross Lettable Area - total area available for exclusive tenant
         occupation (retail/commercial).
    TFA: Treated Floor Area - area within the thermal envelope, used for
         energy certificate calculations (EN ISO 13789).
    """
    GIA = "gia"
    NIA = "nia"
    GLA = "gla"
    TFA = "tfa"


class EUIAccountingBoundary(str, Enum):
    """EUI accounting boundary methods.

    SITE_ENERGY:     Energy consumed at the building meter boundary (kWh).
    SOURCE_ENERGY:   Total primary energy including generation/transmission
                     losses, per ENERGY STAR Technical Reference.
    PRIMARY_ENERGY:  Non-renewable primary energy per EN 15603.
    COST_NORMALISED: Site EUI adjusted by occupancy and activity factors.
    """
    SITE_ENERGY = "site_energy"
    SOURCE_ENERGY = "source_energy"
    PRIMARY_ENERGY = "primary_energy"
    COST_NORMALISED = "cost_normalised"


class CalculationPeriod(str, Enum):
    """Time period basis for EUI calculation.

    ROLLING_12_MONTH: Most recent 12 months of data.
    CALENDAR_YEAR:    January to December.
    FISCAL_YEAR:      User-defined fiscal year.
    CUSTOM:           User-defined start and end dates.
    """
    ROLLING_12_MONTH = "rolling_12_month"
    CALENDAR_YEAR = "calendar_year"
    FISCAL_YEAR = "fiscal_year"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Constants -- Source Energy Factors
# ---------------------------------------------------------------------------

# Source energy factors (site-to-source multipliers).
# Source: ENERGY STAR Portfolio Manager Technical Reference, August 2023.
# https://www.energystar.gov/buildings/benchmark/understand_metrics/source_energy
# U.S. national averages; factor includes generation and transmission losses.
SOURCE_ENERGY_FACTORS: Dict[str, Dict[str, Any]] = {
    EnergyCarrier.ELECTRICITY: {
        "factor": 2.80,
        "source": "ENERGY STAR Technical Reference 2023, U.S. national avg grid",
    },
    EnergyCarrier.NATURAL_GAS: {
        "factor": 1.047,
        "source": "ENERGY STAR Technical Reference 2023, pipeline natural gas",
    },
    EnergyCarrier.FUEL_OIL: {
        "factor": 1.01,
        "source": "ENERGY STAR Technical Reference 2023, fuel oil No. 2",
    },
    EnergyCarrier.LPG: {
        "factor": 1.01,
        "source": "ENERGY STAR Technical Reference 2023, propane/LPG",
    },
    EnergyCarrier.DISTRICT_HEATING: {
        "factor": 1.20,
        "source": "ENERGY STAR Technical Reference 2023, district steam/hot water",
    },
    EnergyCarrier.DISTRICT_COOLING: {
        "factor": 1.30,
        "source": "ENERGY STAR Technical Reference 2023, district chilled water",
    },
    EnergyCarrier.BIOMASS: {
        "factor": 1.00,
        "source": "ENERGY STAR: wood/biomass treated as 1.0 site-to-source",
    },
    EnergyCarrier.SOLAR_THERMAL: {
        "factor": 1.00,
        "source": "On-site renewable generation, site-to-source = 1.0",
    },
}
"""Site-to-source energy multipliers from ENERGY STAR Technical Reference."""


# Primary energy factors (non-renewable primary energy per kWh delivered).
# Source: EN 15603:2008 Annex A, and CEN/TR 15615.
# These are EU-average values; national values may differ.
PRIMARY_ENERGY_FACTORS: Dict[str, Dict[str, Any]] = {
    EnergyCarrier.ELECTRICITY: {
        "factor": 2.50,
        "source": "EN 15603:2008 Annex A, EU weighted avg non-renewable primary",
    },
    EnergyCarrier.NATURAL_GAS: {
        "factor": 1.10,
        "source": "EN 15603:2008 Annex A, natural gas non-renewable primary",
    },
    EnergyCarrier.FUEL_OIL: {
        "factor": 1.10,
        "source": "EN 15603:2008 Annex A, fuel oil non-renewable primary",
    },
    EnergyCarrier.LPG: {
        "factor": 1.10,
        "source": "EN 15603:2008 Annex A, LPG non-renewable primary",
    },
    EnergyCarrier.DISTRICT_HEATING: {
        "factor": 0.80,
        "source": "EN 15603:2008, CHP-based district heating (EU avg)",
    },
    EnergyCarrier.DISTRICT_COOLING: {
        "factor": 0.75,
        "source": "EN 15603:2008, district cooling with free-cooling component",
    },
    EnergyCarrier.BIOMASS: {
        "factor": 0.20,
        "source": "EN 15603:2008 Annex A, biomass renewable primary (non-ren share)",
    },
    EnergyCarrier.SOLAR_THERMAL: {
        "factor": 0.00,
        "source": "EN 15603:2008, solar thermal zero non-renewable primary",
    },
}
"""Non-renewable primary energy factors per EN 15603:2008."""


# Floor area conversion factors (from measured type to GIA).
# Source: RICS Code of Measuring Practice 6th ed, BS EN 15221-6:2011.
# The ratio NIA/GIA depends on building type; here we use typical commercial.
FLOOR_AREA_CONVERSION: Dict[str, Dict[str, float]] = {
    FloorAreaType.GIA: {
        "to_gia": 1.00,
        "source": "Identity; GIA is the reference basis",
    },
    FloorAreaType.NIA: {
        "to_gia": 1.20,
        "source": "RICS typical commercial: NIA ~= 0.83 * GIA, inverse ~= 1.20",
    },
    FloorAreaType.GLA: {
        "to_gia": 1.15,
        "source": "RICS typical retail: GLA ~= 0.87 * GIA, inverse ~= 1.15",
    },
    FloorAreaType.TFA: {
        "to_gia": 1.10,
        "source": "EN ISO 13789 typical: TFA ~= 0.91 * GIA, inverse ~= 1.10",
    },
}
"""Conversion multipliers from measured floor area type to GIA."""


# Occupancy adjustment reference values (hours/week).
# Source: ASHRAE Standard 100-2018, Table 7-1; ENERGY STAR Portfolio Manager.
STANDARD_OCCUPANCY_HOURS: Dict[str, float] = {
    "office": 50.0,
    "retail": 65.0,
    "warehouse": 50.0,
    "school": 40.0,
    "hospital": 168.0,
    "hotel": 168.0,
    "restaurant": 80.0,
    "supermarket": 80.0,
    "data_center": 168.0,
    "DEFAULT": 50.0,
}
"""Standard weekly operating hours by building type, for occupancy normalisation."""


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class FacilityProfile(BaseModel):
    """Building/facility profile for EUI calculation.

    Attributes:
        facility_id: Unique facility identifier.
        name: Human-readable facility name.
        floor_area: Floor area in square metres.
        floor_area_type: Which floor area measurement standard is used.
        building_type: Building use type (office, retail, etc.).
        country_code: ISO 3166-1 alpha-2 country code.
        climate_zone: ASHRAE climate zone (e.g. '4A') or blank.
        year_built: Construction year (optional).
        occupant_count: Number of regular occupants (optional).
        weekly_operating_hours: Actual weekly operating hours (optional).
    """
    facility_id: str = Field(..., min_length=1, description="Facility identifier")
    name: str = Field(..., min_length=1, description="Facility name")
    floor_area: float = Field(..., gt=0, description="Floor area (m2)")
    floor_area_type: FloorAreaType = Field(
        default=FloorAreaType.GIA, description="Floor area measurement type"
    )
    building_type: str = Field(
        default="office", min_length=1, description="Building use type"
    )
    country_code: str = Field(
        default="US", min_length=2, max_length=3, description="Country code"
    )
    climate_zone: str = Field(
        default="", max_length=10, description="ASHRAE climate zone"
    )
    year_built: Optional[int] = Field(
        None, ge=1800, le=2030, description="Year of construction"
    )
    occupant_count: Optional[int] = Field(
        None, ge=0, description="Number of regular occupants"
    )
    weekly_operating_hours: Optional[float] = Field(
        None, ge=0, le=168.0, description="Weekly operating hours"
    )

    @field_validator("floor_area")
    @classmethod
    def validate_floor_area(cls, v: float) -> float:
        """Ensure floor area is within plausible bounds."""
        if v > 5_000_000:
            raise ValueError("Floor area exceeds 5 million m2 sanity check")
        return v


class EnergyMeterData(BaseModel):
    """Energy consumption data for a single carrier in a single period.

    Attributes:
        meter_id: Meter or sub-meter identifier.
        period: Time period label (e.g. '2024-01', '2024-Q1').
        carrier: Energy carrier type.
        consumption_kwh: Energy consumed in kWh for this period.
        cost: Energy cost for the period (optional).
        cost_currency: Currency code (default EUR).
    """
    meter_id: str = Field(default="main", min_length=1, description="Meter ID")
    period: str = Field(..., min_length=4, description="Time period label")
    carrier: EnergyCarrier = Field(..., description="Energy carrier type")
    consumption_kwh: float = Field(..., ge=0, description="Consumption (kWh)")
    cost: Optional[float] = Field(None, ge=0, description="Cost for period")
    cost_currency: str = Field(default="EUR", description="Currency code")

    @field_validator("consumption_kwh")
    @classmethod
    def validate_consumption(cls, v: float) -> float:
        """Ensure consumption is within plausible bounds."""
        if v > 500_000_000:
            raise ValueError("Consumption exceeds 500 GWh per period sanity check")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class EUIResult(BaseModel):
    """Energy Use Intensity result for a single accounting boundary.

    Attributes:
        accounting_boundary: EUI boundary (site, source, primary, cost-normalised).
        eui_kwh_per_m2_yr: Calculated EUI value.
        total_energy_kwh: Total annual energy consumed.
        floor_area_m2: Floor area used for denominator.
        floor_area_type: Floor area type used.
        energy_by_carrier: Breakdown of energy by carrier (kWh).
        carrier_shares_pct: Percentage share by carrier.
    """
    accounting_boundary: str = Field(..., description="Accounting boundary")
    eui_kwh_per_m2_yr: float = Field(default=0.0, description="EUI (kWh/m2/yr)")
    total_energy_kwh: float = Field(default=0.0, description="Total energy (kWh)")
    floor_area_m2: float = Field(default=0.0, description="Floor area (m2)")
    floor_area_type: str = Field(default="", description="Floor area type")
    energy_by_carrier: Dict[str, float] = Field(
        default_factory=dict, description="Energy by carrier (kWh)"
    )
    carrier_shares_pct: Dict[str, float] = Field(
        default_factory=dict, description="Carrier share (%)"
    )


class NormalisedEUI(BaseModel):
    """Normalised EUI after adjusting for occupancy and/or activity.

    Attributes:
        base_eui: Unadjusted EUI.
        normalised_eui: EUI after normalisation adjustments.
        occupancy_adjustment_factor: Adjustment factor for occupancy.
        activity_adjustment_factor: Adjustment factor for activity level.
        normalisation_method: Description of method applied.
    """
    base_eui: float = Field(default=0.0, description="Base EUI (kWh/m2/yr)")
    normalised_eui: float = Field(default=0.0, description="Normalised EUI")
    occupancy_adjustment_factor: float = Field(
        default=1.0, description="Occupancy adjustment factor"
    )
    activity_adjustment_factor: float = Field(
        default=1.0, description="Activity adjustment factor"
    )
    normalisation_method: str = Field(
        default="", description="Normalisation method description"
    )


class RollingEUIPoint(BaseModel):
    """Single data point in a rolling 12-month EUI time series.

    Attributes:
        period_end: End month of the 12-month window.
        eui_kwh_per_m2_yr: EUI for this 12-month window.
        total_energy_kwh: Total energy in this window.
        months_included: Number of months with data in the window.
    """
    period_end: str = Field(..., description="End month of rolling window")
    eui_kwh_per_m2_yr: float = Field(default=0.0, description="Rolling EUI")
    total_energy_kwh: float = Field(default=0.0, description="Total energy (kWh)")
    months_included: int = Field(default=0, description="Months with data")


class EUICalculationResult(BaseModel):
    """Complete EUI calculation result with full provenance.

    Contains site EUI, source EUI, primary EUI, per-carrier breakdown,
    rolling EUI time series, normalised EUI, and recommendations.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    facility_id: str = Field(default="", description="Facility identifier")
    facility_name: str = Field(default="", description="Facility name")
    building_type: str = Field(default="", description="Building type")

    calculation_period: str = Field(default="", description="Period type used")
    period_start: str = Field(default="", description="Start period")
    period_end: str = Field(default="", description="End period")
    months_of_data: int = Field(default=0, description="Months of data available")

    site_eui: Optional[EUIResult] = Field(None, description="Site EUI result")
    source_eui: Optional[EUIResult] = Field(None, description="Source EUI result")
    primary_eui: Optional[EUIResult] = Field(None, description="Primary EUI result")

    rolling_eui: List[RollingEUIPoint] = Field(
        default_factory=list, description="Rolling 12-month EUI series"
    )
    normalised_eui: Optional[NormalisedEUI] = Field(
        None, description="Normalised EUI result"
    )

    eui_per_occupant: Optional[float] = Field(
        None, description="EUI per occupant (kWh/person/yr)"
    )
    cost_per_m2: Optional[float] = Field(
        None, description="Energy cost per m2 (currency/m2/yr)"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class EUICalculatorEngine:
    """Energy Use Intensity calculator engine.

    Provides deterministic, zero-hallucination EUI calculations for:
    - Site EUI (energy at the meter boundary / floor area)
    - Source EUI (including generation/transmission losses per ENERGY STAR)
    - Primary EUI (non-renewable primary energy per EN 15603)
    - Rolling 12-month EUI tracking
    - Occupancy normalisation (actual vs standard hours)
    - Activity normalisation (occupant density adjustment)
    - Floor area type conversions (GIA/NIA/GLA/TFA)
    - Multi-carrier energy aggregation

    All calculations are bit-perfect reproducible using Decimal arithmetic.
    No LLM is used in any calculation path.

    Usage::

        engine = EUICalculatorEngine()
        result = engine.calculate_eui(facility, meter_data)
        print(f"Site EUI: {result.site_eui.eui_kwh_per_m2_yr} kWh/m2/yr")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the EUI calculator engine with embedded reference data."""
        self._source_factors = SOURCE_ENERGY_FACTORS
        self._primary_factors = PRIMARY_ENERGY_FACTORS
        self._area_conversions = FLOOR_AREA_CONVERSION
        self._occupancy_hours = STANDARD_OCCUPANCY_HOURS

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def calculate_eui(
        self,
        facility: FacilityProfile,
        meter_data: List[EnergyMeterData],
        period_type: CalculationPeriod = CalculationPeriod.ROLLING_12_MONTH,
        accounting_boundary: EUIAccountingBoundary = EUIAccountingBoundary.SITE_ENERGY,
    ) -> EUICalculationResult:
        """Calculate comprehensive EUI for a facility.

        Computes site, source, and primary EUI in a single pass, along
        with carrier-level breakdowns, rolling time series, and optional
        normalisation adjustments.

        Args:
            facility: Facility profile with floor area and metadata.
            meter_data: Energy meter readings across all carriers.
            period_type: Which time period to use for annualisation.
            accounting_boundary: Primary accounting boundary to report.

        Returns:
            EUICalculationResult with all EUI variants and provenance.

        Raises:
            ValueError: If meter_data is empty or floor area is zero.
        """
        t0 = time.perf_counter()

        if not meter_data:
            raise ValueError("At least one meter reading is required")

        logger.info(
            "EUI calculation for facility %s (%s), %d readings",
            facility.facility_id, facility.building_type, len(meter_data),
        )

        # Step 1: Convert floor area to GIA-equivalent
        gia_area = self._convert_to_gia(facility.floor_area, facility.floor_area_type)

        # Step 2: Aggregate energy by carrier and period
        carrier_totals, period_totals = self._aggregate_energy(meter_data)

        # Step 3: Determine period range
        periods = sorted(period_totals.keys())
        period_start = periods[0] if periods else ""
        period_end = periods[-1] if periods else ""

        # Step 4: Annualise energy
        total_site_energy, annualised_by_carrier = self._annualise_energy(
            carrier_totals, len(periods), period_type,
        )

        # Step 5: Calculate Site EUI
        site_eui_result = self._calculate_site_eui(
            annualised_by_carrier, total_site_energy, gia_area,
            facility.floor_area_type,
        )

        # Step 6: Calculate Source EUI
        source_eui_result = self._calculate_source_eui(
            annualised_by_carrier, gia_area, facility.floor_area_type,
        )

        # Step 7: Calculate Primary EUI
        primary_eui_result = self._calculate_primary_eui(
            annualised_by_carrier, gia_area, facility.floor_area_type,
        )

        # Step 8: Calculate rolling 12-month EUI
        rolling = self._calculate_rolling_eui_series(
            meter_data, gia_area,
        )

        # Step 9: Normalised EUI
        normalised = self._normalise_eui(
            site_eui_result.eui_kwh_per_m2_yr, facility,
        )

        # Step 10: Per-occupant and cost metrics
        eui_per_occupant = None
        if facility.occupant_count and facility.occupant_count > 0:
            eui_per_occupant = _round2(
                float(_safe_divide(
                    _decimal(total_site_energy),
                    _decimal(facility.occupant_count),
                ))
            )

        total_cost = Decimal("0")
        for m in meter_data:
            if m.cost is not None:
                total_cost += _decimal(m.cost)
        cost_annualised = self._annualise_scalar(total_cost, len(periods))
        cost_per_m2 = None
        if gia_area > Decimal("0"):
            cost_per_m2 = _round2(float(_safe_divide(cost_annualised, gia_area)))

        # Step 11: Recommendations
        recommendations = self._generate_recommendations(
            facility, site_eui_result, source_eui_result, normalised,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = EUICalculationResult(
            facility_id=facility.facility_id,
            facility_name=facility.name,
            building_type=facility.building_type,
            calculation_period=period_type.value,
            period_start=period_start,
            period_end=period_end,
            months_of_data=len(periods),
            site_eui=site_eui_result,
            source_eui=source_eui_result,
            primary_eui=primary_eui_result,
            rolling_eui=rolling,
            normalised_eui=normalised,
            eui_per_occupant=eui_per_occupant,
            cost_per_m2=cost_per_m2,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "EUI complete: facility=%s, site_eui=%.1f, source_eui=%.1f, "
            "primary_eui=%.1f kWh/m2/yr, hash=%s (%.1f ms)",
            facility.facility_id,
            site_eui_result.eui_kwh_per_m2_yr,
            source_eui_result.eui_kwh_per_m2_yr,
            primary_eui_result.eui_kwh_per_m2_yr,
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def calculate_rolling_eui(
        self,
        facility: FacilityProfile,
        meter_data: List[EnergyMeterData],
    ) -> List[RollingEUIPoint]:
        """Calculate rolling 12-month EUI time series.

        For each month that has at least one reading, computes EUI
        using the preceding 12 months of data.

        Args:
            facility: Facility profile with floor area.
            meter_data: Energy meter readings.

        Returns:
            List of RollingEUIPoint sorted chronologically.
        """
        gia_area = self._convert_to_gia(facility.floor_area, facility.floor_area_type)
        return self._calculate_rolling_eui_series(meter_data, gia_area)

    def calculate_source_eui(
        self,
        facility: FacilityProfile,
        meter_data: List[EnergyMeterData],
    ) -> EUIResult:
        """Calculate source EUI using ENERGY STAR site-to-source factors.

        Source energy accounts for energy lost in generation, transmission,
        and distribution. This is the metric used by ENERGY STAR Portfolio
        Manager for building comparisons.

        Args:
            facility: Facility profile.
            meter_data: Energy meter readings.

        Returns:
            EUIResult with source EUI.
        """
        gia_area = self._convert_to_gia(facility.floor_area, facility.floor_area_type)
        carrier_totals, period_totals = self._aggregate_energy(meter_data)
        _, annualised = self._annualise_energy(
            carrier_totals, len(period_totals), CalculationPeriod.ROLLING_12_MONTH,
        )
        return self._calculate_source_eui(
            annualised, gia_area, facility.floor_area_type,
        )

    def normalise_by_occupancy(
        self,
        base_eui: float,
        actual_hours_per_week: float,
        building_type: str = "office",
    ) -> NormalisedEUI:
        """Normalise EUI for occupancy using operating hours.

        Adjusts the base EUI by the ratio of standard operating hours
        to actual operating hours, so that buildings with different
        schedules can be compared on an equivalent basis.

        Formula:
            adjustment = standard_hours / actual_hours
            normalised_eui = base_eui * adjustment

        Args:
            base_eui: Unadjusted EUI (kWh/m2/yr).
            actual_hours_per_week: Actual weekly operating hours.
            building_type: Building type for standard hours lookup.

        Returns:
            NormalisedEUI with occupancy adjustment.
        """
        standard = self._occupancy_hours.get(
            building_type.lower(),
            self._occupancy_hours["DEFAULT"],
        )

        if actual_hours_per_week <= 0:
            actual_hours_per_week = standard

        occ_factor = _safe_divide(
            _decimal(standard), _decimal(actual_hours_per_week), Decimal("1"),
        )

        normalised = _decimal(base_eui) * occ_factor

        return NormalisedEUI(
            base_eui=_round2(base_eui),
            normalised_eui=_round2(float(normalised)),
            occupancy_adjustment_factor=_round4(float(occ_factor)),
            activity_adjustment_factor=1.0,
            normalisation_method=(
                f"Occupancy normalisation: standard {standard}h/wk vs "
                f"actual {actual_hours_per_week}h/wk ({building_type})"
            ),
        )

    def normalise_by_activity(
        self,
        base_eui: float,
        actual_occupant_density: float,
        standard_occupant_density: float = 20.0,
    ) -> NormalisedEUI:
        """Normalise EUI for occupant density (activity level).

        Adjusts EUI by the ratio of standard occupant density to actual
        occupant density.  Standard density is typically 20 m2/person
        for offices (ASHRAE 62.1).

        Formula:
            adjustment = standard_density / actual_density
            normalised_eui = base_eui * adjustment

        Args:
            base_eui: Unadjusted EUI (kWh/m2/yr).
            actual_occupant_density: Actual m2 per occupant.
            standard_occupant_density: Standard m2 per occupant (default 20).

        Returns:
            NormalisedEUI with activity adjustment.
        """
        if actual_occupant_density <= 0:
            actual_occupant_density = standard_occupant_density

        act_factor = _safe_divide(
            _decimal(standard_occupant_density),
            _decimal(actual_occupant_density),
            Decimal("1"),
        )

        normalised = _decimal(base_eui) * act_factor

        return NormalisedEUI(
            base_eui=_round2(base_eui),
            normalised_eui=_round2(float(normalised)),
            occupancy_adjustment_factor=1.0,
            activity_adjustment_factor=_round4(float(act_factor)),
            normalisation_method=(
                f"Activity normalisation: standard {standard_occupant_density} m2/occ "
                f"vs actual {actual_occupant_density} m2/occ"
            ),
        )

    # -------------------------------------------------------------------
    # Internal: Floor Area Conversion
    # -------------------------------------------------------------------

    def _convert_to_gia(
        self,
        area: float,
        area_type: FloorAreaType,
    ) -> Decimal:
        """Convert measured floor area to GIA-equivalent.

        Args:
            area: Measured floor area in m2.
            area_type: Floor area measurement type.

        Returns:
            GIA-equivalent floor area as Decimal.
        """
        conversion = self._area_conversions.get(area_type, {})
        factor = _decimal(conversion.get("to_gia", 1.0))
        gia = _decimal(area) * factor
        logger.debug(
            "Floor area conversion: %.1f %s * %.2f = %.1f GIA m2",
            area, area_type.value, float(factor), float(gia),
        )
        return gia

    # -------------------------------------------------------------------
    # Internal: Energy Aggregation
    # -------------------------------------------------------------------

    def _aggregate_energy(
        self,
        meter_data: List[EnergyMeterData],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Decimal]]:
        """Aggregate meter readings by carrier and by period.

        Args:
            meter_data: List of energy meter readings.

        Returns:
            Tuple of (carrier_totals, period_totals) dictionaries.
        """
        carrier_totals: Dict[str, Decimal] = {}
        period_totals: Dict[str, Decimal] = {}

        for m in meter_data:
            carrier_key = m.carrier.value
            kwh = _decimal(m.consumption_kwh)

            carrier_totals[carrier_key] = (
                carrier_totals.get(carrier_key, Decimal("0")) + kwh
            )
            period_totals[m.period] = (
                period_totals.get(m.period, Decimal("0")) + kwh
            )

        return carrier_totals, period_totals

    # -------------------------------------------------------------------
    # Internal: Annualisation
    # -------------------------------------------------------------------

    def _annualise_energy(
        self,
        carrier_totals: Dict[str, Decimal],
        num_periods: int,
        period_type: CalculationPeriod,
    ) -> Tuple[Decimal, Dict[str, Decimal]]:
        """Annualise carrier energy totals based on number of periods.

        If the data covers fewer than 12 months, it is scaled up to
        an annual equivalent.

        Args:
            carrier_totals: Energy by carrier (raw totals).
            num_periods: Number of monthly periods of data.
            period_type: Period type (used for scaling logic).

        Returns:
            Tuple of (total_annual_kwh, annualised_by_carrier).
        """
        if num_periods <= 0:
            return Decimal("0"), {}

        scale_factor = Decimal("1")
        if num_periods < 12:
            scale_factor = _safe_divide(
                Decimal("12"), _decimal(num_periods), Decimal("1"),
            )
        elif num_periods > 12:
            # More than 12 months: average down to 12
            scale_factor = _safe_divide(
                Decimal("12"), _decimal(num_periods), Decimal("1"),
            )

        annualised: Dict[str, Decimal] = {}
        total = Decimal("0")
        for carrier, kwh in carrier_totals.items():
            annual_kwh = kwh * scale_factor
            annualised[carrier] = annual_kwh
            total += annual_kwh

        return total, annualised

    def _annualise_scalar(
        self, total: Decimal, num_periods: int,
    ) -> Decimal:
        """Annualise a single scalar value from period data.

        Args:
            total: Raw total value.
            num_periods: Number of monthly periods.

        Returns:
            Annualised value.
        """
        if num_periods <= 0:
            return Decimal("0")
        if num_periods == 12:
            return total
        return total * _safe_divide(Decimal("12"), _decimal(num_periods), Decimal("1"))

    # -------------------------------------------------------------------
    # Internal: Site EUI
    # -------------------------------------------------------------------

    def _calculate_site_eui(
        self,
        annualised_by_carrier: Dict[str, Decimal],
        total_annual: Decimal,
        gia_area: Decimal,
        area_type: FloorAreaType,
    ) -> EUIResult:
        """Calculate site EUI.

        Site EUI = Total_Annual_Site_Energy / Floor_Area.

        Args:
            annualised_by_carrier: Annualised energy by carrier.
            total_annual: Total annualised energy.
            gia_area: GIA-equivalent floor area.
            area_type: Original floor area type.

        Returns:
            EUIResult for site energy boundary.
        """
        site_eui = _safe_divide(total_annual, gia_area)

        # Carrier shares
        shares: Dict[str, float] = {}
        for carrier, kwh in annualised_by_carrier.items():
            shares[carrier] = _round2(float(_safe_pct(kwh, total_annual)))

        return EUIResult(
            accounting_boundary=EUIAccountingBoundary.SITE_ENERGY.value,
            eui_kwh_per_m2_yr=_round2(float(site_eui)),
            total_energy_kwh=_round2(float(total_annual)),
            floor_area_m2=_round2(float(gia_area)),
            floor_area_type=area_type.value,
            energy_by_carrier={
                k: _round2(float(v)) for k, v in annualised_by_carrier.items()
            },
            carrier_shares_pct=shares,
        )

    # -------------------------------------------------------------------
    # Internal: Source EUI
    # -------------------------------------------------------------------

    def _calculate_source_eui(
        self,
        annualised_by_carrier: Dict[str, Decimal],
        gia_area: Decimal,
        area_type: FloorAreaType,
    ) -> EUIResult:
        """Calculate source EUI using ENERGY STAR site-to-source factors.

        Source_Energy = sum( site_kwh * source_factor ) for each carrier.
        Source_EUI = Source_Energy / Floor_Area.

        Args:
            annualised_by_carrier: Annualised site energy by carrier.
            gia_area: GIA-equivalent floor area.
            area_type: Original floor area type.

        Returns:
            EUIResult for source energy boundary.
        """
        source_by_carrier: Dict[str, Decimal] = {}
        total_source = Decimal("0")

        for carrier, site_kwh in annualised_by_carrier.items():
            factor_info = self._source_factors.get(carrier, {})
            factor = _decimal(factor_info.get("factor", 1.0))
            source_kwh = site_kwh * factor
            source_by_carrier[carrier] = source_kwh
            total_source += source_kwh

        source_eui = _safe_divide(total_source, gia_area)

        shares: Dict[str, float] = {}
        for carrier, kwh in source_by_carrier.items():
            shares[carrier] = _round2(float(_safe_pct(kwh, total_source)))

        return EUIResult(
            accounting_boundary=EUIAccountingBoundary.SOURCE_ENERGY.value,
            eui_kwh_per_m2_yr=_round2(float(source_eui)),
            total_energy_kwh=_round2(float(total_source)),
            floor_area_m2=_round2(float(gia_area)),
            floor_area_type=area_type.value,
            energy_by_carrier={
                k: _round2(float(v)) for k, v in source_by_carrier.items()
            },
            carrier_shares_pct=shares,
        )

    # -------------------------------------------------------------------
    # Internal: Primary EUI
    # -------------------------------------------------------------------

    def _calculate_primary_eui(
        self,
        annualised_by_carrier: Dict[str, Decimal],
        gia_area: Decimal,
        area_type: FloorAreaType,
    ) -> EUIResult:
        """Calculate primary EUI using EN 15603 primary energy factors.

        Primary_Energy = sum( site_kwh * primary_factor ) for each carrier.
        Primary_EUI = Primary_Energy / Floor_Area.

        Args:
            annualised_by_carrier: Annualised site energy by carrier.
            gia_area: GIA-equivalent floor area.
            area_type: Original floor area type.

        Returns:
            EUIResult for primary energy boundary.
        """
        primary_by_carrier: Dict[str, Decimal] = {}
        total_primary = Decimal("0")

        for carrier, site_kwh in annualised_by_carrier.items():
            factor_info = self._primary_factors.get(carrier, {})
            factor = _decimal(factor_info.get("factor", 1.0))
            primary_kwh = site_kwh * factor
            primary_by_carrier[carrier] = primary_kwh
            total_primary += primary_kwh

        primary_eui = _safe_divide(total_primary, gia_area)

        shares: Dict[str, float] = {}
        for carrier, kwh in primary_by_carrier.items():
            shares[carrier] = _round2(float(_safe_pct(kwh, total_primary)))

        return EUIResult(
            accounting_boundary=EUIAccountingBoundary.PRIMARY_ENERGY.value,
            eui_kwh_per_m2_yr=_round2(float(primary_eui)),
            total_energy_kwh=_round2(float(total_primary)),
            floor_area_m2=_round2(float(gia_area)),
            floor_area_type=area_type.value,
            energy_by_carrier={
                k: _round2(float(v)) for k, v in primary_by_carrier.items()
            },
            carrier_shares_pct=shares,
        )

    # -------------------------------------------------------------------
    # Internal: Rolling 12-Month EUI
    # -------------------------------------------------------------------

    def _calculate_rolling_eui_series(
        self,
        meter_data: List[EnergyMeterData],
        gia_area: Decimal,
    ) -> List[RollingEUIPoint]:
        """Calculate rolling 12-month EUI for each available endpoint.

        For each unique period in the data, sums the preceding 12 months
        (inclusive) and computes EUI over that window.

        Args:
            meter_data: Energy meter readings.
            gia_area: GIA-equivalent floor area.

        Returns:
            List of RollingEUIPoint sorted chronologically.
        """
        # Aggregate by period
        period_energy: Dict[str, Decimal] = {}
        for m in meter_data:
            kwh = _decimal(m.consumption_kwh)
            period_energy[m.period] = period_energy.get(m.period, Decimal("0")) + kwh

        sorted_periods = sorted(period_energy.keys())
        if len(sorted_periods) < 2:
            return []

        results: List[RollingEUIPoint] = []

        for i, end_period in enumerate(sorted_periods):
            # Look back up to 12 periods
            start_idx = max(0, i - 11)
            window_periods = sorted_periods[start_idx: i + 1]
            window_months = len(window_periods)

            window_energy = sum(
                period_energy.get(p, Decimal("0")) for p in window_periods
            )

            # Scale to annual if window < 12 months
            if window_months > 0 and window_months < 12:
                annual_energy = window_energy * _safe_divide(
                    Decimal("12"), _decimal(window_months), Decimal("1"),
                )
            else:
                annual_energy = window_energy

            rolling_eui = _safe_divide(annual_energy, gia_area)

            results.append(RollingEUIPoint(
                period_end=end_period,
                eui_kwh_per_m2_yr=_round2(float(rolling_eui)),
                total_energy_kwh=_round2(float(window_energy)),
                months_included=window_months,
            ))

        return results

    # -------------------------------------------------------------------
    # Internal: Normalisation
    # -------------------------------------------------------------------

    def _normalise_eui(
        self,
        base_eui: float,
        facility: FacilityProfile,
    ) -> Optional[NormalisedEUI]:
        """Apply occupancy and activity normalisation if data is available.

        Args:
            base_eui: Unadjusted site EUI.
            facility: Facility profile with occupancy data.

        Returns:
            NormalisedEUI or None if no normalisation data available.
        """
        has_hours = (
            facility.weekly_operating_hours is not None
            and facility.weekly_operating_hours > 0
        )
        has_occupants = (
            facility.occupant_count is not None
            and facility.occupant_count > 0
            and facility.floor_area > 0
        )

        if not has_hours and not has_occupants:
            return None

        occ_factor = Decimal("1")
        act_factor = Decimal("1")
        method_parts: List[str] = []

        # Occupancy hours adjustment
        if has_hours:
            standard = self._occupancy_hours.get(
                facility.building_type.lower(),
                self._occupancy_hours["DEFAULT"],
            )
            occ_factor = _safe_divide(
                _decimal(standard),
                _decimal(facility.weekly_operating_hours),
                Decimal("1"),
            )
            method_parts.append(
                f"Hours: standard={standard}h vs actual={facility.weekly_operating_hours}h"
            )

        # Occupant density adjustment
        if has_occupants:
            actual_density = _safe_divide(
                _decimal(facility.floor_area),
                _decimal(facility.occupant_count),
                Decimal("20"),
            )
            standard_density = Decimal("20")  # 20 m2/person (ASHRAE 62.1 office)
            act_factor = _safe_divide(
                standard_density, actual_density, Decimal("1"),
            )
            method_parts.append(
                f"Density: standard=20 m2/occ vs actual={_round2(float(actual_density))} m2/occ"
            )

        combined_factor = occ_factor * act_factor
        normalised_eui = _decimal(base_eui) * combined_factor

        return NormalisedEUI(
            base_eui=_round2(base_eui),
            normalised_eui=_round2(float(normalised_eui)),
            occupancy_adjustment_factor=_round4(float(occ_factor)),
            activity_adjustment_factor=_round4(float(act_factor)),
            normalisation_method="; ".join(method_parts),
        )

    # -------------------------------------------------------------------
    # Internal: Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        facility: FacilityProfile,
        site_eui: EUIResult,
        source_eui: EUIResult,
        normalised: Optional[NormalisedEUI],
    ) -> List[str]:
        """Generate deterministic recommendations based on EUI results.

        All recommendations are threshold-based comparisons against
        known reference values. No LLM involvement.

        Args:
            facility: Facility profile.
            site_eui: Calculated site EUI.
            source_eui: Calculated source EUI.
            normalised: Normalised EUI (if available).

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: Data completeness
        if not facility.weekly_operating_hours:
            recs.append(
                "Operating hours data is not available. Providing actual weekly "
                "operating hours enables occupancy-normalised EUI comparison "
                "against ENERGY STAR benchmarks."
            )

        # R2: High source-to-site ratio
        if site_eui.eui_kwh_per_m2_yr > 0:
            ratio = source_eui.eui_kwh_per_m2_yr / site_eui.eui_kwh_per_m2_yr
            if ratio > 2.5:
                recs.append(
                    f"Source-to-site energy ratio is {_round2(ratio)}, indicating "
                    f"heavy electricity dependence. On-site renewable generation "
                    f"(solar PV) could significantly reduce source EUI."
                )

        # R3: Single carrier dominance
        for carrier, share in site_eui.carrier_shares_pct.items():
            if share > 85.0 and carrier == EnergyCarrier.ELECTRICITY.value:
                recs.append(
                    f"Electricity accounts for {share}% of site energy. "
                    f"Investigate thermal end-uses (heating, DHW) that could "
                    f"be served by more efficient sources (heat pumps, solar thermal)."
                )

        # R4: Sub-metering recommendation
        if len(site_eui.energy_by_carrier) <= 1:
            recs.append(
                "Only one energy carrier is metered. Sub-metering by end-use "
                "(HVAC, lighting, plug loads, process) is recommended per "
                "ISO 50001 for identifying improvement opportunities."
            )

        # R5: Rolling trend
        return recs
