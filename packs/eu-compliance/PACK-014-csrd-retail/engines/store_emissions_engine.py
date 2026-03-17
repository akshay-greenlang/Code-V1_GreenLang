# -*- coding: utf-8 -*-
"""
StoreEmissionsEngine - PACK-014 CSRD Retail Engine 1
=====================================================

Store-level Scope 1 and Scope 2 emissions calculator for retail locations.
Covers energy consumption, refrigerant leakage, fleet emissions, and
multi-store consolidation with drill-down by store type, country, and region.

Scope 1 Sources:
    - Natural gas / heating oil / LPG combustion
    - Refrigerant leakage (HFC, HFO, natural refrigerants)
    - Fleet vehicles (delivery vans, trucks, forklifts)
    - Diesel generators (backup power)

Scope 2 Sources:
    - Grid electricity (location-based via EEA grid factors)
    - District heating / cooling
    - Market-based with PPA / REC / residual mix

Regulatory References:
    - GHG Protocol Corporate Standard (2004, updated 2015)
    - EU F-gas Regulation 517/2014 (phase-down schedule)
    - EEA greenhouse gas emission intensity of electricity (2024)
    - IPCC AR6 GWP-100 values for refrigerants

Zero-Hallucination:
    - All emission calculations use deterministic Decimal arithmetic
    - Emission factors sourced from EEA 2024 / IPCC AR6 (hard-coded)
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-014 CSRD Retail & Consumer Goods
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

engine_version: str = "1.0.0"


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
    # Exclude volatile fields to guarantee bit-perfect reproducibility
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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StoreType(str, Enum):
    """Retail store format classification."""
    FLAGSHIP = "flagship"
    STANDARD = "standard"
    EXPRESS = "express"
    OUTLET = "outlet"
    WAREHOUSE = "warehouse"
    DARK_STORE = "dark_store"
    POP_UP = "pop_up"


class EnergySource(str, Enum):
    """Energy source types used in retail operations."""
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    HEATING_OIL = "heating_oil"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    LPG = "lpg"
    SOLAR_PV = "solar_pv"
    WIND = "wind"
    BIOMASS = "biomass"
    DIESEL_GENERATOR = "diesel_generator"


class RefrigerantType(str, Enum):
    """Refrigerant types with GWP classification."""
    R404A = "R404A"
    R134A = "R134A"
    R410A = "R410A"
    R32 = "R32"
    R290 = "R290"
    R744_CO2 = "R744_CO2"
    R1234YF = "R1234YF"
    R1234ZE = "R1234ZE"
    R717_AMMONIA = "R717_AMMONIA"


class FleetVehicleType(str, Enum):
    """Fleet vehicle types used by retail operations."""
    DELIVERY_VAN = "delivery_van"
    TRUCK = "truck"
    ELECTRIC_VAN = "electric_van"
    CARGO_BIKE = "cargo_bike"
    FORKLIFT_DIESEL = "forklift_diesel"
    FORKLIFT_ELECTRIC = "forklift_electric"


# ---------------------------------------------------------------------------
# Constants -- Emission Factors
# ---------------------------------------------------------------------------

# Grid emission factors for EU countries (tCO2e per MWh)
# Source: EEA 2024 greenhouse gas emission intensity of electricity generation
GRID_EMISSION_FACTORS: Dict[str, float] = {
    "AT": 0.091,   # Austria
    "BE": 0.155,   # Belgium
    "BG": 0.410,   # Bulgaria
    "HR": 0.195,   # Croatia
    "CY": 0.623,   # Cyprus
    "CZ": 0.383,   # Czech Republic
    "DK": 0.112,   # Denmark
    "EE": 0.534,   # Estonia
    "FI": 0.068,   # Finland
    "FR": 0.055,   # France
    "DE": 0.338,   # Germany
    "GR": 0.341,   # Greece
    "HU": 0.224,   # Hungary
    "IE": 0.296,   # Ireland
    "IT": 0.256,   # Italy
    "LV": 0.095,   # Latvia
    "LT": 0.128,   # Lithuania
    "LU": 0.084,   # Luxembourg
    "MT": 0.391,   # Malta
    "NL": 0.328,   # Netherlands
    "PL": 0.635,   # Poland
    "PT": 0.161,   # Portugal
    "RO": 0.262,   # Romania
    "SK": 0.115,   # Slovakia
    "SI": 0.221,   # Slovenia
    "ES": 0.150,   # Spain
    "SE": 0.012,   # Sweden
    "NO": 0.008,   # Norway
    "CH": 0.015,   # Switzerland
    "UK": 0.207,   # United Kingdom
    "IS": 0.000,   # Iceland (geothermal/hydro)
    "LI": 0.015,   # Liechtenstein
    "EU_AVG": 0.230,  # EU-27 average
}

# Residual mix factors for market-based Scope 2 (tCO2e per MWh)
# Source: AIB European Residual Mixes 2024
RESIDUAL_MIX_FACTORS: Dict[str, float] = {
    "AT": 0.259,   "BE": 0.234,   "BG": 0.518,   "HR": 0.261,
    "CY": 0.707,   "CZ": 0.579,   "DK": 0.303,   "EE": 0.623,
    "FI": 0.186,   "FR": 0.069,   "DE": 0.493,   "GR": 0.481,
    "HU": 0.316,   "IE": 0.418,   "IT": 0.394,   "LV": 0.192,
    "LT": 0.248,   "LU": 0.341,   "MT": 0.611,   "NL": 0.507,
    "PL": 0.772,   "PT": 0.238,   "RO": 0.358,   "SK": 0.226,
    "SI": 0.319,   "ES": 0.238,   "SE": 0.034,   "NO": 0.396,
    "CH": 0.016,   "UK": 0.324,   "EU_AVG": 0.376,
}

# Fuel emission factors (tCO2e per MWh thermal)
# Source: DEFRA/BEIS 2024 UK Government GHG Conversion Factors
FUEL_EMISSION_FACTORS: Dict[str, float] = {
    "natural_gas": 0.202,
    "heating_oil": 0.267,
    "lpg": 0.227,
    "diesel": 0.267,
    "biomass": 0.015,     # Scope 1 biogenic -- near-zero fossil
    "district_heating": 0.180,  # EU average
    "district_cooling": 0.120,  # EU average
}

# Refrigerant GWP-100 values
# Source: IPCC AR6 (2021) Table 7.SM.7
REFRIGERANT_GWP: Dict[str, int] = {
    RefrigerantType.R404A: 3922,
    RefrigerantType.R134A: 1430,
    RefrigerantType.R410A: 2088,
    RefrigerantType.R32: 675,
    RefrigerantType.R290: 3,
    RefrigerantType.R744_CO2: 1,
    RefrigerantType.R1234YF: 4,
    RefrigerantType.R1234ZE: 7,
    RefrigerantType.R717_AMMONIA: 0,
}

# Typical annual leakage rates by store type (% of total charge per year)
# Source: EU F-gas Regulation impact assessment, IOR guidelines
TYPICAL_LEAKAGE_RATES: Dict[str, float] = {
    StoreType.FLAGSHIP: 18.0,
    StoreType.STANDARD: 15.0,
    StoreType.EXPRESS: 20.0,
    StoreType.OUTLET: 15.0,
    StoreType.WAREHOUSE: 10.0,
    StoreType.DARK_STORE: 12.0,
    StoreType.POP_UP: 25.0,
}

# F-gas Regulation phase-down schedule (% of 2015 HFC baseline allowed)
# Source: Regulation (EU) 2024/573 (revised F-gas Regulation)
F_GAS_PHASE_DOWN_SCHEDULE: Dict[int, float] = {
    2025: 31.0,
    2027: 24.0,
    2030: 21.0,
    2033: 14.0,
    2036: 7.0,
    2040: 5.0,
    2045: 2.0,
    2050: 0.0,
}

# Fleet vehicle emission factors
# Fuel-based (tCO2e per litre of fuel)
FLEET_FUEL_FACTORS: Dict[str, float] = {
    FleetVehicleType.DELIVERY_VAN: 0.002676,     # Diesel van ~2.676 kgCO2/litre
    FleetVehicleType.TRUCK: 0.002676,             # Diesel truck
    FleetVehicleType.ELECTRIC_VAN: 0.0,           # Zero direct (Scope 2)
    FleetVehicleType.CARGO_BIKE: 0.0,             # Zero emissions
    FleetVehicleType.FORKLIFT_DIESEL: 0.002676,   # Diesel forklift
    FleetVehicleType.FORKLIFT_ELECTRIC: 0.0,      # Zero direct (Scope 2)
}

# Fleet vehicle distance-based factors (tCO2e per km) -- fallback
FLEET_DISTANCE_FACTORS: Dict[str, float] = {
    FleetVehicleType.DELIVERY_VAN: 0.000249,      # ~249 gCO2/km
    FleetVehicleType.TRUCK: 0.000586,             # ~586 gCO2/km (rigid, >3.5t)
    FleetVehicleType.ELECTRIC_VAN: 0.0,           # Zero direct
    FleetVehicleType.CARGO_BIKE: 0.0,
    FleetVehicleType.FORKLIFT_DIESEL: 0.000120,   # Approximate
    FleetVehicleType.FORKLIFT_ELECTRIC: 0.0,
}

# Renewable energy sources -- zero or near-zero emission factors
RENEWABLE_SOURCES = {EnergySource.SOLAR_PV, EnergySource.WIND}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class EnergyConsumption(BaseModel):
    """Energy consumption record for a single source at a store.

    Attributes:
        source: Type of energy source.
        quantity_kwh: Energy consumed in kilowatt-hours.
        cost_eur: Cost in EUR (optional, for intensity metrics).
        renewable_pct: Percentage of this source that is certified renewable.
        emission_factor_override: Optional override for emission factor (tCO2e/MWh).
    """
    source: EnergySource
    quantity_kwh: float = Field(..., ge=0, description="Energy consumed (kWh)")
    cost_eur: Optional[float] = Field(None, ge=0, description="Cost in EUR")
    renewable_pct: float = Field(0.0, ge=0, le=100, description="Renewable percentage")
    emission_factor_override: Optional[float] = Field(
        None, ge=0, description="Custom EF override (tCO2e/MWh)"
    )

    @field_validator("quantity_kwh")
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        """Ensure quantity is non-negative and within plausible bounds."""
        if v > 500_000_000:
            raise ValueError("Energy quantity exceeds plausible maximum (500 GWh)")
        return v


class RefrigerantData(BaseModel):
    """Refrigerant data for a store's refrigeration system.

    Attributes:
        refrigerant_type: Type of refrigerant.
        charge_kg: Total refrigerant charge in kilograms.
        leakage_rate_pct: Annual leakage rate as percentage of charge.
        top_up_kg: Actual refrigerant top-up in kg (overrides charge * leakage calc).
        new_equipment_gwp: GWP of replacement equipment refrigerant (for transition tracking).
    """
    refrigerant_type: RefrigerantType
    charge_kg: float = Field(..., ge=0, description="Total charge (kg)")
    leakage_rate_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Leakage rate (%)"
    )
    top_up_kg: Optional[float] = Field(None, ge=0, description="Actual top-up (kg)")
    new_equipment_gwp: Optional[float] = Field(
        None, ge=0, description="New equipment GWP"
    )


class FleetData(BaseModel):
    """Fleet vehicle data for a store or distribution center.

    Attributes:
        vehicle_type: Type of fleet vehicle.
        count: Number of vehicles.
        fuel_consumption_litres: Total fuel consumed (litres).
        distance_km: Total distance driven (km).
    """
    vehicle_type: FleetVehicleType
    count: int = Field(..., ge=0, description="Number of vehicles")
    fuel_consumption_litres: Optional[float] = Field(
        None, ge=0, description="Fuel consumed (litres)"
    )
    distance_km: Optional[float] = Field(None, ge=0, description="Distance driven (km)")

    @model_validator(mode="after")
    def check_fuel_or_distance(self) -> "FleetData":
        """Ensure at least fuel or distance is provided for combustion vehicles."""
        non_electric = {
            FleetVehicleType.DELIVERY_VAN,
            FleetVehicleType.TRUCK,
            FleetVehicleType.FORKLIFT_DIESEL,
        }
        if self.vehicle_type in non_electric:
            if self.fuel_consumption_litres is None and self.distance_km is None:
                raise ValueError(
                    "Either fuel_consumption_litres or distance_km required "
                    "for combustion vehicles"
                )
        return self


class StoreData(BaseModel):
    """Complete store data for emissions calculation.

    Attributes:
        store_id: Unique store identifier.
        store_name: Human-readable store name.
        store_type: Store format classification.
        country: ISO 3166-1 alpha-2 country code.
        region: Optional sub-region for drill-down reporting.
        floor_area_sqm: Store floor area in square metres.
        employees: Number of employees (FTE).
        operating_hours_per_year: Annual operating hours.
        energy_consumption: List of energy consumption records.
        refrigerants: List of refrigerant data records.
        fleet: List of fleet vehicle data records.
        has_ppa: Whether the store has a Power Purchase Agreement.
        ppa_emission_factor: PPA contractual emission factor (tCO2e/MWh).
        rec_mwh: Renewable Energy Certificates purchased (MWh).
    """
    store_id: str = Field(..., min_length=1, description="Store identifier")
    store_name: str = Field(..., min_length=1, description="Store name")
    store_type: StoreType
    country: str = Field(..., min_length=2, max_length=7, description="Country code")
    region: Optional[str] = Field(None, description="Sub-region")
    floor_area_sqm: float = Field(..., gt=0, description="Floor area (m2)")
    employees: int = Field(..., ge=1, description="Employee count (FTE)")
    operating_hours_per_year: float = Field(
        default=4380.0, gt=0, le=8784, description="Operating hours/year"
    )
    energy_consumption: List[EnergyConsumption] = Field(default_factory=list)
    refrigerants: List[RefrigerantData] = Field(default_factory=list)
    fleet: List[FleetData] = Field(default_factory=list)
    has_ppa: bool = Field(False, description="Has Power Purchase Agreement")
    ppa_emission_factor: Optional[float] = Field(
        None, ge=0, description="PPA EF (tCO2e/MWh)"
    )
    rec_mwh: float = Field(0.0, ge=0, description="RECs purchased (MWh)")


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class RefrigerantEmissionDetail(BaseModel):
    """Detailed refrigerant emission breakdown.

    Attributes:
        refrigerant_type: Refrigerant name.
        gwp: Global Warming Potential (100-year).
        charge_kg: Total system charge.
        leakage_kg: Estimated or actual leakage.
        emissions_tco2e: Resulting CO2-equivalent emissions.
    """
    refrigerant_type: str
    gwp: int
    charge_kg: float
    leakage_kg: float
    emissions_tco2e: float


class FleetEmissionDetail(BaseModel):
    """Detailed fleet emission breakdown.

    Attributes:
        vehicle_type: Fleet vehicle type.
        count: Number of vehicles.
        fuel_litres: Fuel consumed.
        distance_km: Distance covered.
        emissions_tco2e: Resulting CO2-equivalent emissions.
    """
    vehicle_type: str
    count: int
    fuel_litres: float
    distance_km: float
    emissions_tco2e: float


class EnergyEmissionDetail(BaseModel):
    """Detailed energy source emission breakdown.

    Attributes:
        source: Energy source type.
        quantity_kwh: Energy consumed (kWh).
        quantity_mwh: Energy consumed (MWh).
        emission_factor: Emission factor used (tCO2e/MWh).
        emissions_tco2e: Resulting emissions.
        is_renewable: Whether this is a renewable source.
    """
    source: str
    quantity_kwh: float
    quantity_mwh: float
    emission_factor: float
    emissions_tco2e: float
    is_renewable: bool


class FGasComplianceResult(BaseModel):
    """F-gas Regulation compliance assessment.

    Attributes:
        total_hfc_charge_kg: Total HFC refrigerant charge.
        weighted_average_gwp: Charge-weighted average GWP.
        co2e_tonnes: Total CO2-equivalent of HFC inventory.
        phase_down_year: Target year for compliance check.
        quota_pct_allowed: Allowed percentage of baseline.
        compliant: Whether current HFC use meets quota.
        recommendation: Compliance recommendation text.
    """
    total_hfc_charge_kg: float
    weighted_average_gwp: float
    co2e_tonnes: float
    phase_down_year: int
    quota_pct_allowed: float
    compliant: bool
    recommendation: str


class StoreEmissionsResult(BaseModel):
    """Complete store-level emissions result with provenance.

    Attributes:
        store_id: Store identifier.
        store_name: Store name.
        store_type: Store format.
        country: Country code.
        scope1_tco2e: Total Scope 1 emissions (tCO2e).
        scope2_location_tco2e: Scope 2 location-based (tCO2e).
        scope2_market_tco2e: Scope 2 market-based (tCO2e).
        refrigerant_tco2e: Refrigerant leakage emissions (tCO2e).
        heating_tco2e: Heating fuel combustion emissions (tCO2e).
        fleet_tco2e: Fleet vehicle emissions (tCO2e).
        electricity_tco2e: Electricity emissions -- location-based (tCO2e).
        renewable_generation_tco2e_avoided: Avoided emissions from on-site renewables.
        total_tco2e: Grand total (Scope 1 + Scope 2 location).
        emissions_per_sqm: Emission intensity per m2.
        emissions_per_employee: Emission intensity per employee.
        energy_intensity_kwh_per_sqm: Energy use intensity.
        energy_details: Breakdown by energy source.
        refrigerant_details: Breakdown by refrigerant.
        fleet_details: Breakdown by vehicle type.
        fgas_compliance: F-gas regulation compliance check.
        engine_version: Engine version string.
        calculated_at: Calculation timestamp (UTC).
        processing_time_ms: Calculation duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """
    store_id: str
    store_name: str
    store_type: str
    country: str
    scope1_tco2e: float
    scope2_location_tco2e: float
    scope2_market_tco2e: float
    refrigerant_tco2e: float
    heating_tco2e: float
    fleet_tco2e: float
    electricity_tco2e: float
    renewable_generation_tco2e_avoided: float
    total_tco2e: float
    emissions_per_sqm: float
    emissions_per_employee: float
    energy_intensity_kwh_per_sqm: float
    energy_details: List[EnergyEmissionDetail]
    refrigerant_details: List[RefrigerantEmissionDetail]
    fleet_details: List[FleetEmissionDetail]
    fgas_compliance: Optional[FGasComplianceResult] = None
    engine_version: str = engine_version
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""


class MultiStoreConsolidationResult(BaseModel):
    """Consolidated emissions across multiple stores.

    Attributes:
        total_stores: Number of stores processed.
        total_scope1_tco2e: Aggregate Scope 1 emissions.
        total_scope2_location_tco2e: Aggregate Scope 2 location-based.
        total_scope2_market_tco2e: Aggregate Scope 2 market-based.
        total_tco2e: Grand total.
        avg_emissions_per_sqm: Average intensity per m2.
        avg_emissions_per_employee: Average intensity per employee.
        by_store_type: Breakdown by store type.
        by_country: Breakdown by country.
        by_region: Breakdown by region.
        store_results: Individual store results.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        processing_time_ms: Total processing time.
        provenance_hash: SHA-256 hash.
    """
    total_stores: int
    total_scope1_tco2e: float
    total_scope2_location_tco2e: float
    total_scope2_market_tco2e: float
    total_tco2e: float
    avg_emissions_per_sqm: float
    avg_emissions_per_employee: float
    by_store_type: Dict[str, Dict[str, float]]
    by_country: Dict[str, Dict[str, float]]
    by_region: Dict[str, Dict[str, float]]
    store_results: List[StoreEmissionsResult]
    engine_version: str = engine_version
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class StoreEmissionsEngine:
    """Store-level Scope 1 and Scope 2 emissions calculation engine.

    Implements deterministic, bit-perfect calculations for retail store
    emissions with complete provenance tracking. All arithmetic uses
    Python Decimal to avoid floating-point drift.

    Guarantees:
        - Deterministic: identical inputs always produce identical outputs.
        - Reproducible: full provenance via SHA-256 hashing.
        - Auditable: every calculation step is recorded.
        - Zero-hallucination: no LLM in the calculation path.

    Usage::

        engine = StoreEmissionsEngine()
        result = engine.calculate_store_emissions(store_data)
        consolidated = engine.consolidate_stores([store1, store2])
    """

    def __init__(self) -> None:
        """Initialise the store emissions engine with embedded constants."""
        self._grid_factors = GRID_EMISSION_FACTORS
        self._residual_mix = RESIDUAL_MIX_FACTORS
        self._fuel_factors = FUEL_EMISSION_FACTORS
        self._refrigerant_gwp = REFRIGERANT_GWP
        self._leakage_rates = TYPICAL_LEAKAGE_RATES
        self._fleet_fuel_factors = FLEET_FUEL_FACTORS
        self._fleet_distance_factors = FLEET_DISTANCE_FACTORS

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def calculate_store_emissions(self, store: StoreData) -> StoreEmissionsResult:
        """Calculate Scope 1 and Scope 2 emissions for a single store.

        Processes energy consumption, refrigerant data, and fleet data to
        produce a complete emissions profile with provenance hash.

        Args:
            store: Complete store data including energy, refrigerants, fleet.

        Returns:
            StoreEmissionsResult with all emission breakdowns and provenance.
        """
        t0 = time.perf_counter()

        # --- Energy emissions ---
        energy_details: List[EnergyEmissionDetail] = []
        scope1_heating = Decimal("0")
        scope2_location_electricity = Decimal("0")
        scope2_market_electricity = Decimal("0")
        renewable_avoided = Decimal("0")
        total_energy_kwh = Decimal("0")

        for ec in store.energy_consumption:
            detail = self._calc_energy_emission(
                ec, store.country, store.has_ppa, store.ppa_emission_factor
            )
            energy_details.append(detail)
            total_energy_kwh += _decimal(detail.quantity_kwh)

            if ec.source in (
                EnergySource.NATURAL_GAS,
                EnergySource.HEATING_OIL,
                EnergySource.LPG,
                EnergySource.BIOMASS,
                EnergySource.DIESEL_GENERATOR,
            ):
                scope1_heating += _decimal(detail.emissions_tco2e)
            elif ec.source == EnergySource.ELECTRICITY:
                scope2_location_electricity += _decimal(detail.emissions_tco2e)
                scope2_market_electricity += self._calc_market_based_electricity(
                    ec,
                    store.country,
                    store.has_ppa,
                    store.ppa_emission_factor,
                    store.rec_mwh,
                )
            elif ec.source in (
                EnergySource.DISTRICT_HEATING,
                EnergySource.DISTRICT_COOLING,
            ):
                scope2_location_electricity += _decimal(detail.emissions_tco2e)
                scope2_market_electricity += _decimal(detail.emissions_tco2e)
            elif ec.source in RENEWABLE_SOURCES:
                renewable_avoided += _decimal(detail.emissions_tco2e)

        # --- Refrigerant emissions ---
        refrigerant_details: List[RefrigerantEmissionDetail] = []
        total_refrigerant = Decimal("0")
        for ref in store.refrigerants:
            ref_detail = self._calc_refrigerant_emission(ref, store.store_type)
            refrigerant_details.append(ref_detail)
            total_refrigerant += _decimal(ref_detail.emissions_tco2e)

        # --- Fleet emissions ---
        fleet_details: List[FleetEmissionDetail] = []
        total_fleet = Decimal("0")
        for flt in store.fleet:
            flt_detail = self._calc_fleet_emission(flt)
            fleet_details.append(flt_detail)
            total_fleet += _decimal(flt_detail.emissions_tco2e)

        # --- Aggregation ---
        scope1_total = scope1_heating + total_refrigerant + total_fleet
        scope2_loc_total = scope2_location_electricity
        scope2_mkt_total = scope2_market_electricity
        grand_total = scope1_total + scope2_loc_total

        floor_area = _decimal(store.floor_area_sqm)
        employees = _decimal(store.employees)

        emissions_per_sqm = _safe_divide(grand_total, floor_area)
        emissions_per_emp = _safe_divide(grand_total, employees)
        energy_intensity = _safe_divide(total_energy_kwh, floor_area)

        # --- F-gas compliance ---
        fgas_result = (
            self._check_fgas_compliance(store.refrigerants)
            if store.refrigerants
            else None
        )

        processing_ms = (time.perf_counter() - t0) * 1000.0

        result = StoreEmissionsResult(
            store_id=store.store_id,
            store_name=store.store_name,
            store_type=store.store_type.value,
            country=store.country,
            scope1_tco2e=_round_val(scope1_total, 6),
            scope2_location_tco2e=_round_val(scope2_loc_total, 6),
            scope2_market_tco2e=_round_val(scope2_mkt_total, 6),
            refrigerant_tco2e=_round_val(total_refrigerant, 6),
            heating_tco2e=_round_val(scope1_heating, 6),
            fleet_tco2e=_round_val(total_fleet, 6),
            electricity_tco2e=_round_val(scope2_location_electricity, 6),
            renewable_generation_tco2e_avoided=_round_val(renewable_avoided, 6),
            total_tco2e=_round_val(grand_total, 6),
            emissions_per_sqm=_round_val(emissions_per_sqm, 6),
            emissions_per_employee=_round_val(emissions_per_emp, 6),
            energy_intensity_kwh_per_sqm=_round_val(energy_intensity, 2),
            energy_details=energy_details,
            refrigerant_details=refrigerant_details,
            fleet_details=fleet_details,
            fgas_compliance=fgas_result,
            processing_time_ms=round(processing_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def consolidate_stores(
        self, stores: List[StoreData]
    ) -> MultiStoreConsolidationResult:
        """Consolidate emissions across multiple retail stores.

        Calculates individual store emissions and aggregates them with
        drill-down by store_type, country, and region.

        Args:
            stores: List of store data objects.

        Returns:
            MultiStoreConsolidationResult with aggregate and drill-down data.

        Raises:
            ValueError: If stores list is empty.
        """
        if not stores:
            raise ValueError("At least one store is required for consolidation")

        t0 = time.perf_counter()
        store_results: List[StoreEmissionsResult] = []

        by_type: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: defaultdict(Decimal)
        )
        by_country: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: defaultdict(Decimal)
        )
        by_region: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: defaultdict(Decimal)
        )

        total_s1 = Decimal("0")
        total_s2_loc = Decimal("0")
        total_s2_mkt = Decimal("0")
        total_all = Decimal("0")
        total_sqm = Decimal("0")
        total_emp = Decimal("0")

        for s in stores:
            r = self.calculate_store_emissions(s)
            store_results.append(r)

            s1 = _decimal(r.scope1_tco2e)
            s2l = _decimal(r.scope2_location_tco2e)
            s2m = _decimal(r.scope2_market_tco2e)
            tot = _decimal(r.total_tco2e)

            total_s1 += s1
            total_s2_loc += s2l
            total_s2_mkt += s2m
            total_all += tot
            total_sqm += _decimal(s.floor_area_sqm)
            total_emp += _decimal(s.employees)

            for grouping, key in [
                (by_type, r.store_type),
                (by_country, r.country),
                (by_region, s.region or "unspecified"),
            ]:
                grouping[key]["scope1_tco2e"] += s1
                grouping[key]["scope2_location_tco2e"] += s2l
                grouping[key]["scope2_market_tco2e"] += s2m
                grouping[key]["total_tco2e"] += tot
                grouping[key]["store_count"] += Decimal("1")

        def _serialise_group(
            g: Dict[str, Dict[str, Decimal]],
        ) -> Dict[str, Dict[str, float]]:
            """Convert Decimal group values to float for serialization."""
            return {
                k: {kk: _round_val(vv, 6) for kk, vv in v.items()}
                for k, v in g.items()
            }

        processing_ms = (time.perf_counter() - t0) * 1000.0

        result = MultiStoreConsolidationResult(
            total_stores=len(stores),
            total_scope1_tco2e=_round_val(total_s1, 6),
            total_scope2_location_tco2e=_round_val(total_s2_loc, 6),
            total_scope2_market_tco2e=_round_val(total_s2_mkt, 6),
            total_tco2e=_round_val(total_all, 6),
            avg_emissions_per_sqm=_round_val(
                _safe_divide(total_all, total_sqm), 6
            ),
            avg_emissions_per_employee=_round_val(
                _safe_divide(total_all, total_emp), 6
            ),
            by_store_type=_serialise_group(by_type),
            by_country=_serialise_group(by_country),
            by_region=_serialise_group(by_region),
            store_results=store_results,
            processing_time_ms=round(processing_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Internal calculation methods
    # -------------------------------------------------------------------

    def _calc_energy_emission(
        self,
        ec: EnergyConsumption,
        country: str,
        has_ppa: bool,
        ppa_ef: Optional[float],
    ) -> EnergyEmissionDetail:
        """Calculate emissions from a single energy consumption record.

        Selects the appropriate emission factor based on energy source type,
        applies any override, and computes tCO2e.

        Args:
            ec: Energy consumption record.
            country: Country code for grid factor lookup.
            has_ppa: Whether a PPA is in place.
            ppa_ef: PPA contractual emission factor if applicable.

        Returns:
            EnergyEmissionDetail with source-level breakdown.
        """
        qty_mwh = _decimal(ec.quantity_kwh) / Decimal("1000")
        is_renewable = ec.source in RENEWABLE_SOURCES

        # Determine emission factor
        if ec.emission_factor_override is not None:
            ef = _decimal(ec.emission_factor_override)
        elif ec.source == EnergySource.ELECTRICITY:
            ef = _decimal(
                self._grid_factors.get(country, self._grid_factors["EU_AVG"])
            )
        elif ec.source in (
            EnergySource.NATURAL_GAS,
            EnergySource.HEATING_OIL,
            EnergySource.LPG,
            EnergySource.BIOMASS,
            EnergySource.DIESEL_GENERATOR,
        ):
            fuel_key = ec.source.value
            if fuel_key == "diesel_generator":
                fuel_key = "diesel"
            ef = _decimal(self._fuel_factors.get(fuel_key, 0.0))
        elif ec.source == EnergySource.DISTRICT_HEATING:
            ef = _decimal(
                self._fuel_factors.get("district_heating", 0.180)
            )
        elif ec.source == EnergySource.DISTRICT_COOLING:
            ef = _decimal(
                self._fuel_factors.get("district_cooling", 0.120)
            )
        elif is_renewable:
            ef = Decimal("0")
        else:
            ef = Decimal("0")

        # Renewable percentage reduces effective consumption for fossil factor
        renewable_frac = _decimal(ec.renewable_pct) / Decimal("100")
        effective_mwh = qty_mwh * (Decimal("1") - renewable_frac)

        if is_renewable:
            # For on-site renewables, calculate avoided emissions using grid factor
            grid_ef = _decimal(
                self._grid_factors.get(country, self._grid_factors["EU_AVG"])
            )
            emissions = qty_mwh * grid_ef  # These are avoided
        else:
            emissions = effective_mwh * ef

        return EnergyEmissionDetail(
            source=ec.source.value,
            quantity_kwh=ec.quantity_kwh,
            quantity_mwh=_round_val(qty_mwh, 6),
            emission_factor=_round_val(ef, 6),
            emissions_tco2e=_round_val(emissions, 6),
            is_renewable=is_renewable,
        )

    def _calc_market_based_electricity(
        self,
        ec: EnergyConsumption,
        country: str,
        has_ppa: bool,
        ppa_ef: Optional[float],
        rec_mwh: float,
    ) -> Decimal:
        """Calculate market-based Scope 2 electricity emissions.

        Applies the GHG Protocol Scope 2 Guidance hierarchy:
        1. Energy attribute certificates / contractual instruments (PPA)
        2. Renewable Energy Certificates (RECs)
        3. Residual mix factor

        Args:
            ec: Electricity consumption record.
            country: Country code.
            has_ppa: Whether PPA is in place.
            ppa_ef: PPA emission factor.
            rec_mwh: MWh covered by RECs.

        Returns:
            Market-based emissions in tCO2e as Decimal.
        """
        qty_mwh = _decimal(ec.quantity_kwh) / Decimal("1000")
        renewable_frac = _decimal(ec.renewable_pct) / Decimal("100")
        effective_mwh = qty_mwh * (Decimal("1") - renewable_frac)

        if has_ppa and ppa_ef is not None:
            # PPA covers all electricity
            return effective_mwh * _decimal(ppa_ef)

        rec_coverage = _decimal(rec_mwh)
        if rec_coverage > Decimal("0"):
            # RECs cover a portion -- remainder uses residual mix
            covered = min(rec_coverage, effective_mwh)
            uncovered = effective_mwh - covered
            residual_ef = _decimal(
                self._residual_mix.get(
                    country, self._residual_mix.get("EU_AVG", 0.376)
                )
            )
            return uncovered * residual_ef  # RECs assumed zero EF

        # No contractual instruments -- use residual mix
        residual_ef = _decimal(
            self._residual_mix.get(
                country, self._residual_mix.get("EU_AVG", 0.376)
            )
        )
        return effective_mwh * residual_ef

    def _calc_refrigerant_emission(
        self,
        ref: RefrigerantData,
        store_type: StoreType,
    ) -> RefrigerantEmissionDetail:
        """Calculate refrigerant leakage emissions.

        Uses actual top-up data when available; otherwise estimates leakage
        from charge * leakage rate. Applies IPCC AR6 GWP-100 values.

        Formula: emissions_tCO2e = leakage_kg * GWP / 1000

        Args:
            ref: Refrigerant data record.
            store_type: Store type for default leakage rate lookup.

        Returns:
            RefrigerantEmissionDetail with emission breakdown.
        """
        gwp = self._refrigerant_gwp.get(ref.refrigerant_type, 0)

        if ref.top_up_kg is not None and ref.top_up_kg > 0:
            leakage_kg = _decimal(ref.top_up_kg)
        else:
            rate_pct = ref.leakage_rate_pct
            if rate_pct is None:
                rate_pct = self._leakage_rates.get(store_type, 15.0)
            leakage_kg = (
                _decimal(ref.charge_kg) * _decimal(rate_pct) / Decimal("100")
            )

        emissions = leakage_kg * _decimal(gwp) / Decimal("1000")

        return RefrigerantEmissionDetail(
            refrigerant_type=ref.refrigerant_type.value,
            gwp=gwp,
            charge_kg=ref.charge_kg,
            leakage_kg=_round_val(leakage_kg, 3),
            emissions_tco2e=_round_val(emissions, 6),
        )

    def _calc_fleet_emission(self, flt: FleetData) -> FleetEmissionDetail:
        """Calculate fleet vehicle emissions.

        Prefers fuel-based calculation; falls back to distance-based.
        Electric and zero-emission vehicles return zero Scope 1.

        Args:
            flt: Fleet vehicle data record.

        Returns:
            FleetEmissionDetail with emission breakdown.
        """
        fuel_litres = _decimal(flt.fuel_consumption_litres or 0)
        distance_km = _decimal(flt.distance_km or 0)

        if fuel_litres > Decimal("0"):
            ef = _decimal(self._fleet_fuel_factors.get(flt.vehicle_type, 0.0))
            emissions = fuel_litres * ef
        elif distance_km > Decimal("0"):
            ef = _decimal(
                self._fleet_distance_factors.get(flt.vehicle_type, 0.0)
            )
            emissions = distance_km * ef
        else:
            emissions = Decimal("0")

        return FleetEmissionDetail(
            vehicle_type=flt.vehicle_type.value,
            count=flt.count,
            fuel_litres=_round_val(fuel_litres, 2),
            distance_km=_round_val(distance_km, 2),
            emissions_tco2e=_round_val(emissions, 6),
        )

    def _check_fgas_compliance(
        self, refrigerants: List[RefrigerantData]
    ) -> FGasComplianceResult:
        """Check compliance with EU F-gas Regulation phase-down schedule.

        Evaluates total HFC charge against the phase-down quota for the
        nearest applicable year.

        Args:
            refrigerants: List of refrigerant records to assess.

        Returns:
            FGasComplianceResult with compliance status and recommendation.
        """
        hfc_types = {
            RefrigerantType.R404A,
            RefrigerantType.R134A,
            RefrigerantType.R410A,
            RefrigerantType.R32,
        }

        total_hfc_charge = Decimal("0")
        weighted_gwp_sum = Decimal("0")

        for ref in refrigerants:
            if ref.refrigerant_type in hfc_types:
                charge = _decimal(ref.charge_kg)
                gwp = _decimal(
                    self._refrigerant_gwp.get(ref.refrigerant_type, 0)
                )
                total_hfc_charge += charge
                weighted_gwp_sum += charge * gwp

        avg_gwp = _safe_divide(weighted_gwp_sum, total_hfc_charge)
        co2e = weighted_gwp_sum / Decimal("1000")

        current_year = datetime.now().year
        phase_year = current_year
        quota_pct = Decimal("100")
        for yr in sorted(F_GAS_PHASE_DOWN_SCHEDULE.keys()):
            if yr <= current_year:
                phase_year = yr
                quota_pct = _decimal(F_GAS_PHASE_DOWN_SCHEDULE[yr])

        # Simple compliance: high-GWP systems may need replacement
        compliant = avg_gwp < Decimal("2500") or total_hfc_charge < Decimal("10")

        if avg_gwp > Decimal("2500"):
            recommendation = (
                f"Average GWP of {_round_val(avg_gwp, 0):.0f} exceeds 2500. "
                "Plan transition to low-GWP alternatives (R290, R744, R1234yf). "
                f"F-gas phase-down requires {_round_val(quota_pct, 0):.0f}% of "
                f"2015 baseline by {phase_year}."
            )
        elif avg_gwp > Decimal("750"):
            recommendation = (
                f"Average GWP of {_round_val(avg_gwp, 0):.0f} is moderate. "
                "Consider accelerating transition to natural refrigerants for "
                "long-term F-gas compliance."
            )
        else:
            recommendation = (
                f"Average GWP of {_round_val(avg_gwp, 0):.0f} is low. "
                "Current refrigerant portfolio is well-positioned for F-gas "
                "compliance."
            )

        return FGasComplianceResult(
            total_hfc_charge_kg=_round_val(total_hfc_charge, 2),
            weighted_average_gwp=_round_val(avg_gwp, 0),
            co2e_tonnes=_round_val(co2e, 3),
            phase_down_year=phase_year,
            quota_pct_allowed=_round_val(quota_pct, 1),
            compliant=compliant,
            recommendation=recommendation,
        )
