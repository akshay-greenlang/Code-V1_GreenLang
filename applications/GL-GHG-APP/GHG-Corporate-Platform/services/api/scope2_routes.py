"""
GL-GHG-APP Scope 2 Indirect Emissions API

Manages Scope 2 (energy indirect) GHG emissions per GHG Protocol
Scope 2 Guidance (2015). Implements mandatory dual reporting:
    - Location-based method: grid-average emission factors
    - Market-based method: contractual instruments (RECs, PPAs, green tariffs)

Supports reconciliation waterfall showing the bridge between the two methods.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/scope2", tags=["Scope 2 Emissions"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope2Method(str, Enum):
    """GHG Protocol Scope 2 calculation methods."""
    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"


class EnergyType(str, Enum):
    """Types of purchased energy for Scope 2."""
    ELECTRICITY = "electricity"
    STEAM = "steam"
    HEATING = "heating"
    COOLING = "cooling"


class InstrumentType(str, Enum):
    """Contractual instrument types for market-based method."""
    REC = "rec"
    GO = "guarantee_of_origin"
    PPA = "power_purchase_agreement"
    GREEN_TARIFF = "green_tariff"
    DIRECT_LINE = "direct_line"
    DEFAULT_EF = "residual_mix"
    SUPPLIER_SPECIFIC = "supplier_specific"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class Scope2DataSubmission(BaseModel):
    """Request to submit Scope 2 energy consumption data."""
    energy_type: EnergyType = Field(..., description="Type of purchased energy")
    facility_id: Optional[str] = Field(None, description="Facility entity ID")
    facility_name: Optional[str] = Field(None, description="Facility display name")
    electricity_kwh: float = Field(..., gt=0, description="Energy consumed in kWh")
    grid_region: Optional[str] = Field(None, description="eGRID subregion or country grid code")
    grid_emission_factor: Optional[float] = Field(
        None, ge=0, description="Location-based EF (kg CO2e/kWh)"
    )
    instruments: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Contractual instruments applied. Each dict: "
            "{type, quantity_kwh, ef_kgco2e_per_kwh, certificate_id, vintage_year}"
        ),
    )
    t_and_d_loss_pct: Optional[float] = Field(
        None, ge=0, le=50, description="Transmission & distribution loss percentage"
    )
    period_start: Optional[str] = Field(None, description="Period start (YYYY-MM-DD)")
    period_end: Optional[str] = Field(None, description="Period end (YYYY-MM-DD)")
    notes: Optional[str] = Field(None, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "energy_type": "electricity",
                "facility_name": "East Coast Plant",
                "electricity_kwh": 2500000,
                "grid_region": "RFCW",
                "grid_emission_factor": 0.495,
                "instruments": [
                    {
                        "type": "rec",
                        "quantity_kwh": 1000000,
                        "ef_kgco2e_per_kwh": 0.0,
                        "certificate_id": "REC-2025-00001",
                        "vintage_year": 2025
                    }
                ],
                "t_and_d_loss_pct": 5.3,
                "period_start": "2025-01-01",
                "period_end": "2025-12-31"
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class Scope2Summary(BaseModel):
    """Scope 2 summary with dual reporting totals."""
    inventory_id: str
    location_based_tco2e: float
    market_based_tco2e: float
    difference_tco2e: float
    difference_pct: float
    total_electricity_mwh: float
    total_steam_mwh: float
    total_heating_mwh: float
    total_cooling_mwh: float
    renewable_pct: float
    facility_count: int
    grid_regions_count: int
    data_quality_score: float
    year_over_year_change_pct: Optional[float]


class LocationBasedEntry(BaseModel):
    """Location-based Scope 2 breakdown entry."""
    grid_region: str
    grid_region_name: str
    electricity_mwh: float
    emission_factor_kgco2e_per_kwh: float
    total_tco2e: float
    percentage_of_location_total: float
    t_and_d_loss_tco2e: float
    data_source: str


class MarketBasedEntry(BaseModel):
    """Market-based Scope 2 breakdown entry."""
    instrument_type: str
    instrument_label: str
    electricity_mwh: float
    emission_factor_kgco2e_per_kwh: float
    total_tco2e: float
    percentage_of_market_total: float
    certificate_count: int
    vintage_year: Optional[int]


class ReconciliationWaterfall(BaseModel):
    """Dual reporting reconciliation waterfall."""
    inventory_id: str
    location_based_total_tco2e: float
    market_based_total_tco2e: float
    waterfall_steps: List[Dict[str, Any]]
    net_difference_tco2e: float
    net_difference_pct: float
    reconciliation_status: str
    notes: List[str]


class InstrumentRecord(BaseModel):
    """A contractual instrument record."""
    instrument_id: str
    type: str
    type_label: str
    facility_name: str
    quantity_mwh: float
    emission_factor_kgco2e_per_kwh: float
    avoided_tco2e: float
    certificate_id: Optional[str]
    vintage_year: int
    issuer: Optional[str]
    registry: Optional[str]
    verified: bool
    expiry_date: Optional[str]


class Scope2AggregationResponse(BaseModel):
    """Response for Scope 2 aggregation."""
    inventory_id: str
    status: str
    location_based_tco2e: float
    market_based_tco2e: float
    facilities_aggregated: int
    data_records_processed: int
    instruments_applied: int
    aggregated_at: datetime


class Scope2DataResponse(BaseModel):
    """Response for Scope 2 data submission."""
    record_id: str
    inventory_id: str
    energy_type: str
    facility_name: Optional[str]
    electricity_mwh: float
    location_based_tco2e: float
    market_based_tco2e: float
    instruments_applied: int
    renewable_kwh: float
    status: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# Default grid emission factors (kg CO2e per kWh) -- eGRID 2022 subregions
GRID_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "RFCW": {"name": "RFC West", "ef": 0.495, "source": "EPA eGRID 2022"},
    "SRMW": {"name": "SERC Midwest", "ef": 0.631, "source": "EPA eGRID 2022"},
    "CAMX": {"name": "WECC California", "ef": 0.225, "source": "EPA eGRID 2022"},
    "NWPP": {"name": "WECC Northwest", "ef": 0.287, "source": "EPA eGRID 2022"},
    "NYUP": {"name": "NPCC Upstate NY", "ef": 0.112, "source": "EPA eGRID 2022"},
    "ERCT": {"name": "ERCOT Texas", "ef": 0.388, "source": "EPA eGRID 2022"},
    "SRSO": {"name": "SERC South", "ef": 0.441, "source": "EPA eGRID 2022"},
    "MROE": {"name": "MRO East", "ef": 0.543, "source": "EPA eGRID 2022"},
}


def _simulated_location_data(inventory_id: str) -> List[Dict[str, Any]]:
    """Simulated location-based Scope 2 breakdown by grid region."""
    total_location = 8320.5
    entries = [
        {
            "grid_region": "RFCW",
            "grid_region_name": "RFC West (OH/WV/PA)",
            "electricity_mwh": 10500.0,
            "emission_factor_kgco2e_per_kwh": 0.495,
            "total_tco2e": 5197.5,
            "t_and_d_loss_tco2e": 275.5,
            "data_source": "EPA eGRID 2022",
        },
        {
            "grid_region": "CAMX",
            "grid_region_name": "WECC California",
            "electricity_mwh": 8500.0,
            "emission_factor_kgco2e_per_kwh": 0.225,
            "total_tco2e": 1912.5,
            "t_and_d_loss_tco2e": 101.4,
            "data_source": "EPA eGRID 2022",
        },
        {
            "grid_region": "ERCT",
            "grid_region_name": "ERCOT Texas",
            "electricity_mwh": 3120.0,
            "emission_factor_kgco2e_per_kwh": 0.388,
            "total_tco2e": 1210.5,
            "t_and_d_loss_tco2e": 64.2,
            "data_source": "EPA eGRID 2022",
        },
    ]
    for e in entries:
        e["percentage_of_location_total"] = round(e["total_tco2e"] / total_location * 100, 2)
    return entries


def _simulated_market_data(inventory_id: str) -> List[Dict[str, Any]]:
    """Simulated market-based Scope 2 breakdown by instrument type."""
    total_market = 5890.3
    entries = [
        {
            "instrument_type": "residual_mix",
            "instrument_label": "Grid Residual Mix (unmatched)",
            "electricity_mwh": 12120.0,
            "emission_factor_kgco2e_per_kwh": 0.486,
            "total_tco2e": 5890.3,
            "certificate_count": 0,
            "vintage_year": None,
        },
        {
            "instrument_type": "rec",
            "instrument_label": "Renewable Energy Certificates",
            "electricity_mwh": 6000.0,
            "emission_factor_kgco2e_per_kwh": 0.0,
            "total_tco2e": 0.0,
            "certificate_count": 6000,
            "vintage_year": 2025,
        },
        {
            "instrument_type": "ppa",
            "instrument_label": "Power Purchase Agreement (Solar)",
            "electricity_mwh": 4000.0,
            "emission_factor_kgco2e_per_kwh": 0.0,
            "total_tco2e": 0.0,
            "certificate_count": 1,
            "vintage_year": 2025,
        },
    ]
    for e in entries:
        e["percentage_of_market_total"] = round(
            e["total_tco2e"] / total_market * 100, 2
        ) if total_market > 0 else 0.0
    return entries


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/aggregate/{inventory_id}",
    response_model=Scope2AggregationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Aggregate Scope 2 emissions",
    description=(
        "Trigger aggregation of Scope 2 emissions using both location-based "
        "and market-based methods per the GHG Protocol Scope 2 Guidance."
    ),
)
async def aggregate_scope2(inventory_id: str) -> Scope2AggregationResponse:
    return Scope2AggregationResponse(
        inventory_id=inventory_id,
        status="completed",
        location_based_tco2e=8320.5,
        market_based_tco2e=5890.3,
        facilities_aggregated=3,
        data_records_processed=36,
        instruments_applied=2,
        aggregated_at=_now(),
    )


@router.get(
    "/{inventory_id}/summary",
    response_model=Scope2Summary,
    summary="Scope 2 emissions summary with dual reporting",
    description=(
        "High-level summary showing both location-based and market-based "
        "totals, the difference between methods, renewable energy percentage, "
        "and data quality."
    ),
)
async def get_scope2_summary(inventory_id: str) -> Scope2Summary:
    location = 8320.5
    market = 5890.3
    diff = location - market
    return Scope2Summary(
        inventory_id=inventory_id,
        location_based_tco2e=location,
        market_based_tco2e=market,
        difference_tco2e=round(diff, 2),
        difference_pct=round(diff / location * 100, 2) if location > 0 else 0.0,
        total_electricity_mwh=22120.0,
        total_steam_mwh=0.0,
        total_heating_mwh=0.0,
        total_cooling_mwh=0.0,
        renewable_pct=45.2,
        facility_count=3,
        grid_regions_count=3,
        data_quality_score=91.2,
        year_over_year_change_pct=-5.8,
    )


@router.get(
    "/{inventory_id}/location-based",
    response_model=List[LocationBasedEntry],
    summary="Location-based Scope 2 breakdown",
    description=(
        "Breakdown of location-based Scope 2 emissions by grid region. "
        "Uses EPA eGRID subregion emission factors for US facilities "
        "and IEA country-level factors for international."
    ),
)
async def get_scope2_location_based(inventory_id: str) -> List[LocationBasedEntry]:
    entries = _simulated_location_data(inventory_id)
    return [LocationBasedEntry(**e) for e in entries]


@router.get(
    "/{inventory_id}/market-based",
    response_model=List[MarketBasedEntry],
    summary="Market-based Scope 2 breakdown",
    description=(
        "Breakdown of market-based Scope 2 emissions by contractual "
        "instrument type. Shows the impact of RECs, PPAs, green tariffs, "
        "and residual mix on reported emissions."
    ),
)
async def get_scope2_market_based(inventory_id: str) -> List[MarketBasedEntry]:
    entries = _simulated_market_data(inventory_id)
    return [MarketBasedEntry(**e) for e in entries]


@router.get(
    "/{inventory_id}/reconciliation",
    response_model=ReconciliationWaterfall,
    summary="Dual reporting reconciliation",
    description=(
        "Waterfall analysis showing the bridge from location-based to "
        "market-based Scope 2 totals. Identifies the specific instruments "
        "and factors driving the difference per GHG Protocol Ch. 7."
    ),
)
async def get_scope2_reconciliation(inventory_id: str) -> ReconciliationWaterfall:
    location = 8320.5
    market = 5890.3
    diff = location - market
    return ReconciliationWaterfall(
        inventory_id=inventory_id,
        location_based_total_tco2e=location,
        market_based_total_tco2e=market,
        waterfall_steps=[
            {
                "step": 1,
                "label": "Location-Based Total",
                "value_tco2e": location,
                "cumulative_tco2e": location,
                "type": "start",
            },
            {
                "step": 2,
                "label": "RECs Applied (6,000 MWh wind)",
                "value_tco2e": -1530.2,
                "cumulative_tco2e": location - 1530.2,
                "type": "reduction",
            },
            {
                "step": 3,
                "label": "PPA Applied (4,000 MWh solar)",
                "value_tco2e": -900.0,
                "cumulative_tco2e": location - 1530.2 - 900.0,
                "type": "reduction",
            },
            {
                "step": 4,
                "label": "Market-Based Total",
                "value_tco2e": market,
                "cumulative_tco2e": market,
                "type": "end",
            },
        ],
        net_difference_tco2e=round(diff, 2),
        net_difference_pct=round(diff / location * 100, 2),
        reconciliation_status="reconciled",
        notes=[
            "6,000 RECs from Green-e certified wind projects offset RFCW grid emissions",
            "4,000 MWh PPA with SunPower solar farm in California",
            "Remaining 12,120 MWh uses residual mix emission factor",
        ],
    )


@router.get(
    "/{inventory_id}/instruments",
    response_model=List[InstrumentRecord],
    summary="Contractual instruments list",
    description=(
        "List of all contractual instruments (RECs, PPAs, green tariffs) "
        "applied to the market-based Scope 2 calculation. Includes "
        "certificate IDs, vintage years, and verification status."
    ),
)
async def get_scope2_instruments(inventory_id: str) -> List[InstrumentRecord]:
    return [
        InstrumentRecord(
            instrument_id=_generate_id("inst"),
            type="rec",
            type_label="Renewable Energy Certificate",
            facility_name="East Coast Plant",
            quantity_mwh=4000.0,
            emission_factor_kgco2e_per_kwh=0.0,
            avoided_tco2e=1980.0,
            certificate_id="REC-2025-WIND-00001",
            vintage_year=2025,
            issuer="Green-e Energy",
            registry="M-RETS",
            verified=True,
            expiry_date="2026-12-31",
        ),
        InstrumentRecord(
            instrument_id=_generate_id("inst"),
            type="rec",
            type_label="Renewable Energy Certificate",
            facility_name="West Coast Distribution",
            quantity_mwh=2000.0,
            emission_factor_kgco2e_per_kwh=0.0,
            avoided_tco2e=450.0,
            certificate_id="REC-2025-WIND-00045",
            vintage_year=2025,
            issuer="Green-e Energy",
            registry="WREGIS",
            verified=True,
            expiry_date="2026-12-31",
        ),
        InstrumentRecord(
            instrument_id=_generate_id("inst"),
            type="ppa",
            type_label="Power Purchase Agreement",
            facility_name="ERCOT Operations",
            quantity_mwh=4000.0,
            emission_factor_kgco2e_per_kwh=0.0,
            avoided_tco2e=1552.0,
            certificate_id="PPA-2023-SOLAR-TX-001",
            vintage_year=2025,
            issuer="SunPower Corp",
            registry="ERCOT",
            verified=True,
            expiry_date="2038-06-30",
        ),
    ]


@router.post(
    "/{inventory_id}/data",
    response_model=Scope2DataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Scope 2 energy data",
    description=(
        "Submit purchased energy consumption data for Scope 2 calculation. "
        "Provide electricity kWh, grid region, and any contractual instruments. "
        "Both location-based and market-based emissions are calculated."
    ),
)
async def submit_scope2_data(
    inventory_id: str,
    data: Scope2DataSubmission,
) -> Scope2DataResponse:
    # Determine location-based EF
    grid_ef = data.grid_emission_factor
    if grid_ef is None and data.grid_region:
        region_info = GRID_EMISSION_FACTORS.get(data.grid_region)
        if region_info:
            grid_ef = region_info["ef"]
    if grid_ef is None:
        grid_ef = 0.417  # US national average

    electricity_mwh = data.electricity_kwh / 1000.0
    location_tco2e = round(data.electricity_kwh * grid_ef / 1000.0, 4)

    # Market-based: subtract instrument kWh at zero EF
    renewable_kwh = 0.0
    instruments_applied = 0
    if data.instruments:
        for inst in data.instruments:
            qty = inst.get("quantity_kwh", 0)
            renewable_kwh += qty
            instruments_applied += 1

    uncovered_kwh = max(0, data.electricity_kwh - renewable_kwh)
    residual_ef = grid_ef * 1.05  # Residual mix slightly higher than grid average
    market_tco2e = round(uncovered_kwh * residual_ef / 1000.0, 4)

    record_id = _generate_id("s2d")
    return Scope2DataResponse(
        record_id=record_id,
        inventory_id=inventory_id,
        energy_type=data.energy_type.value,
        facility_name=data.facility_name,
        electricity_mwh=electricity_mwh,
        location_based_tco2e=location_tco2e,
        market_based_tco2e=market_tco2e,
        instruments_applied=instruments_applied,
        renewable_kwh=renewable_kwh,
        status="accepted",
        created_at=_now(),
    )
