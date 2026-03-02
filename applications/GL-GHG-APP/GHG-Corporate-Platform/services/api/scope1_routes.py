"""
GL-GHG-APP Scope 1 Direct Emissions API

Manages Scope 1 (direct) GHG emissions per GHG Protocol Chapter 4.
Covers eight source categories:
    1. Stationary Combustion (boilers, furnaces, turbines)
    2. Mobile Combustion (fleet vehicles, equipment)
    3. Process Emissions (chemical/physical transformations)
    4. Fugitive Emissions (methane leaks, venting)
    5. Refrigerants & F-Gases (HVAC, chillers)
    6. Land Use & Forestry (on-site biomass changes)
    7. Waste Treatment (on-site incineration, treatment)
    8. Agricultural (on-site livestock, soil management)

GHGs: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3 (Kyoto basket).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/scope1", tags=["Scope 1 Emissions"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope1Category(str, Enum):
    """GHG Protocol Scope 1 source categories."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    REFRIGERANTS = "refrigerants"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"


class GHGGas(str, Enum):
    """Kyoto basket greenhouse gases."""
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFC = "HFC"
    PFC = "PFC"
    SF6 = "SF6"
    NF3 = "NF3"


class CalculationTier(str, Enum):
    """IPCC calculation tier methodology."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class Scope1DataSubmission(BaseModel):
    """Request to submit Scope 1 emission activity data."""
    category: Scope1Category = Field(..., description="Scope 1 source category")
    facility_id: Optional[str] = Field(None, description="Facility entity ID")
    facility_name: Optional[str] = Field(None, description="Facility name for display")
    quantity: float = Field(..., gt=0, description="Activity quantity")
    unit: str = Field(..., description="Unit of measure (e.g. therms, gallons, kg)")
    fuel_type: Optional[str] = Field(None, description="Fuel type (natural_gas, diesel, etc.)")
    gas_type: Optional[str] = Field(None, description="Specific gas (HFC-134a, etc.)")
    emission_factor: Optional[float] = Field(None, ge=0, description="Custom emission factor (kg CO2e per unit)")
    emission_factor_source: Optional[str] = Field(None, description="EF source (EPA, IPCC, custom)")
    calculation_tier: CalculationTier = Field(CalculationTier.TIER_1, description="IPCC calculation tier")
    period_start: Optional[str] = Field(None, description="Activity period start (YYYY-MM-DD)")
    period_end: Optional[str] = Field(None, description="Activity period end (YYYY-MM-DD)")
    notes: Optional[str] = Field(None, max_length=1000, description="Data submission notes")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "stationary_combustion",
                "facility_id": "ent_abc123",
                "facility_name": "East Coast Plant",
                "quantity": 150000,
                "unit": "therms",
                "fuel_type": "natural_gas",
                "emission_factor": 5.302,
                "emission_factor_source": "EPA",
                "calculation_tier": "tier_1",
                "period_start": "2025-01-01",
                "period_end": "2025-12-31",
                "notes": "Annual natural gas consumption from utility bills"
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class CategoryBreakdown(BaseModel):
    """Emission breakdown for a single Scope 1 category."""
    category: str
    category_label: str
    total_tco2e: float
    percentage_of_scope1: float
    co2_tonnes: float
    ch4_tonnes_co2e: float
    n2o_tonnes_co2e: float
    other_ghg_tonnes_co2e: float
    facility_count: int
    data_quality_score: float
    calculation_tier: str
    emission_factor_source: str


class FacilityBreakdown(BaseModel):
    """Emission breakdown for a single facility."""
    facility_id: str
    facility_name: str
    country: str
    total_tco2e: float
    percentage_of_scope1: float
    categories: List[str]
    top_source: str
    data_quality_score: float


class GasBreakdown(BaseModel):
    """Emission breakdown for a single greenhouse gas."""
    gas: str
    gas_name: str
    gwp_ar5: int
    total_tonnes: float
    total_tco2e: float
    percentage_of_scope1: float
    primary_sources: List[str]


class Scope1Summary(BaseModel):
    """Scope 1 emissions summary."""
    inventory_id: str
    total_tco2e: float
    total_co2_tonnes: float
    total_ch4_tco2e: float
    total_n2o_tco2e: float
    total_fgas_tco2e: float
    category_count: int
    facility_count: int
    data_quality_score: float
    calculation_completeness_pct: float
    year_over_year_change_pct: Optional[float]
    top_category: str
    top_facility: str


class Scope1AggregationResponse(BaseModel):
    """Response for Scope 1 aggregation trigger."""
    inventory_id: str
    status: str
    total_tco2e: float
    categories_aggregated: int
    facilities_aggregated: int
    data_records_processed: int
    aggregated_at: datetime


class Scope1DataResponse(BaseModel):
    """Response for Scope 1 data submission."""
    record_id: str
    inventory_id: str
    category: str
    facility_name: Optional[str]
    quantity: float
    unit: str
    calculated_tco2e: float
    emission_factor_used: float
    emission_factor_source: str
    calculation_tier: str
    status: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Simulated Data
# ---------------------------------------------------------------------------

SCOPE1_CATEGORY_LABELS = {
    "stationary_combustion": "Stationary Combustion",
    "mobile_combustion": "Mobile Combustion",
    "process_emissions": "Process Emissions",
    "fugitive_emissions": "Fugitive Emissions",
    "refrigerants": "Refrigerants & F-Gases",
    "land_use": "Land Use & Forestry",
    "waste_treatment": "On-site Waste Treatment",
    "agricultural": "Agricultural Emissions",
}

GAS_INFO = {
    "CO2": {"name": "Carbon Dioxide", "gwp_ar5": 1},
    "CH4": {"name": "Methane", "gwp_ar5": 28},
    "N2O": {"name": "Nitrous Oxide", "gwp_ar5": 265},
    "HFC": {"name": "Hydrofluorocarbons", "gwp_ar5": 1430},
    "PFC": {"name": "Perfluorocarbons", "gwp_ar5": 7390},
    "SF6": {"name": "Sulfur Hexafluoride", "gwp_ar5": 23500},
    "NF3": {"name": "Nitrogen Trifluoride", "gwp_ar5": 16100},
}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _simulated_category_data(inventory_id: str) -> List[Dict[str, Any]]:
    """Generate simulated Scope 1 category breakdowns."""
    total_scope1 = 12450.8
    categories = [
        {
            "category": "stationary_combustion",
            "category_label": "Stationary Combustion",
            "total_tco2e": 5820.3,
            "co2_tonnes": 5680.0,
            "ch4_tonnes_co2e": 98.5,
            "n2o_tonnes_co2e": 41.8,
            "other_ghg_tonnes_co2e": 0.0,
            "facility_count": 4,
            "data_quality_score": 92.0,
            "calculation_tier": "tier_1",
            "emission_factor_source": "EPA AP-42 / eGRID",
        },
        {
            "category": "mobile_combustion",
            "category_label": "Mobile Combustion",
            "total_tco2e": 2340.5,
            "co2_tonnes": 2290.0,
            "ch4_tonnes_co2e": 35.2,
            "n2o_tonnes_co2e": 15.3,
            "other_ghg_tonnes_co2e": 0.0,
            "facility_count": 3,
            "data_quality_score": 85.0,
            "calculation_tier": "tier_1",
            "emission_factor_source": "EPA GHG Emission Factors Hub",
        },
        {
            "category": "process_emissions",
            "category_label": "Process Emissions",
            "total_tco2e": 1890.0,
            "co2_tonnes": 1890.0,
            "ch4_tonnes_co2e": 0.0,
            "n2o_tonnes_co2e": 0.0,
            "other_ghg_tonnes_co2e": 0.0,
            "facility_count": 1,
            "data_quality_score": 88.0,
            "calculation_tier": "tier_2",
            "emission_factor_source": "IPCC 2006 Vol. 3",
        },
        {
            "category": "fugitive_emissions",
            "category_label": "Fugitive Emissions",
            "total_tco2e": 1250.0,
            "co2_tonnes": 120.0,
            "ch4_tonnes_co2e": 1120.0,
            "n2o_tonnes_co2e": 10.0,
            "other_ghg_tonnes_co2e": 0.0,
            "facility_count": 2,
            "data_quality_score": 75.0,
            "calculation_tier": "tier_1",
            "emission_factor_source": "EPA Subpart W",
        },
        {
            "category": "refrigerants",
            "category_label": "Refrigerants & F-Gases",
            "total_tco2e": 1150.0,
            "co2_tonnes": 0.0,
            "ch4_tonnes_co2e": 0.0,
            "n2o_tonnes_co2e": 0.0,
            "other_ghg_tonnes_co2e": 1150.0,
            "facility_count": 4,
            "data_quality_score": 90.0,
            "calculation_tier": "tier_1",
            "emission_factor_source": "EPA SNAP / IPCC AR5",
        },
        {
            "category": "land_use",
            "category_label": "Land Use & Forestry",
            "total_tco2e": 0.0,
            "co2_tonnes": 0.0,
            "ch4_tonnes_co2e": 0.0,
            "n2o_tonnes_co2e": 0.0,
            "other_ghg_tonnes_co2e": 0.0,
            "facility_count": 0,
            "data_quality_score": 0.0,
            "calculation_tier": "tier_1",
            "emission_factor_source": "N/A",
        },
        {
            "category": "waste_treatment",
            "category_label": "On-site Waste Treatment",
            "total_tco2e": 0.0,
            "co2_tonnes": 0.0,
            "ch4_tonnes_co2e": 0.0,
            "n2o_tonnes_co2e": 0.0,
            "other_ghg_tonnes_co2e": 0.0,
            "facility_count": 0,
            "data_quality_score": 0.0,
            "calculation_tier": "tier_1",
            "emission_factor_source": "N/A",
        },
        {
            "category": "agricultural",
            "category_label": "Agricultural Emissions",
            "total_tco2e": 0.0,
            "co2_tonnes": 0.0,
            "ch4_tonnes_co2e": 0.0,
            "n2o_tonnes_co2e": 0.0,
            "other_ghg_tonnes_co2e": 0.0,
            "facility_count": 0,
            "data_quality_score": 0.0,
            "calculation_tier": "tier_1",
            "emission_factor_source": "N/A",
        },
    ]
    for cat in categories:
        cat["percentage_of_scope1"] = round(cat["total_tco2e"] / total_scope1 * 100, 2) if total_scope1 > 0 else 0.0
    return categories


def _simulated_facility_data() -> List[Dict[str, Any]]:
    """Generate simulated Scope 1 facility breakdowns."""
    total = 12450.8
    facilities = [
        {
            "facility_id": "ent_east_plant",
            "facility_name": "East Coast Manufacturing Plant",
            "country": "US",
            "total_tco2e": 5230.5,
            "categories": ["stationary_combustion", "mobile_combustion", "refrigerants"],
            "top_source": "Stationary Combustion (natural gas boilers)",
            "data_quality_score": 91.0,
        },
        {
            "facility_id": "ent_west_plant",
            "facility_name": "West Coast Distribution Center",
            "country": "US",
            "total_tco2e": 3120.3,
            "categories": ["stationary_combustion", "mobile_combustion", "fugitive_emissions"],
            "top_source": "Mobile Combustion (delivery fleet)",
            "data_quality_score": 87.0,
        },
        {
            "facility_id": "ent_cement_ops",
            "facility_name": "Cement Processing Facility",
            "country": "US",
            "total_tco2e": 2800.0,
            "categories": ["process_emissions", "stationary_combustion", "fugitive_emissions"],
            "top_source": "Process Emissions (clinker production)",
            "data_quality_score": 88.0,
        },
        {
            "facility_id": "ent_hq_office",
            "facility_name": "Corporate Headquarters",
            "country": "US",
            "total_tco2e": 1300.0,
            "categories": ["stationary_combustion", "refrigerants"],
            "top_source": "Refrigerants (HVAC system HFC-134a)",
            "data_quality_score": 93.0,
        },
    ]
    for fac in facilities:
        fac["percentage_of_scope1"] = round(fac["total_tco2e"] / total * 100, 2)
    return facilities


def _simulated_gas_data() -> List[Dict[str, Any]]:
    """Generate simulated Scope 1 gas breakdowns."""
    total = 12450.8
    gases = [
        {
            "gas": "CO2",
            "gas_name": "Carbon Dioxide",
            "gwp_ar5": 1,
            "total_tonnes": 9980.0,
            "total_tco2e": 9980.0,
            "primary_sources": ["stationary_combustion", "mobile_combustion", "process_emissions"],
        },
        {
            "gas": "CH4",
            "gas_name": "Methane",
            "gwp_ar5": 28,
            "total_tonnes": 44.78,
            "total_tco2e": 1253.7,
            "primary_sources": ["fugitive_emissions", "stationary_combustion"],
        },
        {
            "gas": "N2O",
            "gas_name": "Nitrous Oxide",
            "gwp_ar5": 265,
            "total_tonnes": 0.253,
            "total_tco2e": 67.1,
            "primary_sources": ["stationary_combustion", "mobile_combustion"],
        },
        {
            "gas": "HFC",
            "gas_name": "Hydrofluorocarbons (HFC-134a)",
            "gwp_ar5": 1430,
            "total_tonnes": 0.804,
            "total_tco2e": 1150.0,
            "primary_sources": ["refrigerants"],
        },
        {
            "gas": "PFC",
            "gas_name": "Perfluorocarbons",
            "gwp_ar5": 7390,
            "total_tonnes": 0.0,
            "total_tco2e": 0.0,
            "primary_sources": [],
        },
        {
            "gas": "SF6",
            "gas_name": "Sulfur Hexafluoride",
            "gwp_ar5": 23500,
            "total_tonnes": 0.0,
            "total_tco2e": 0.0,
            "primary_sources": [],
        },
        {
            "gas": "NF3",
            "gas_name": "Nitrogen Trifluoride",
            "gwp_ar5": 16100,
            "total_tonnes": 0.0,
            "total_tco2e": 0.0,
            "primary_sources": [],
        },
    ]
    for gas in gases:
        gas["percentage_of_scope1"] = round(gas["total_tco2e"] / total * 100, 2) if total > 0 else 0.0
    return gases


# Default emission factors (kg CO2e per unit) for demo calculations
DEFAULT_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "stationary_combustion": {
        "natural_gas": 5.302,       # kg CO2e per therm
        "diesel": 10.21,            # kg CO2e per gallon
        "fuel_oil_no2": 10.16,      # kg CO2e per gallon
        "propane": 5.72,            # kg CO2e per gallon
        "coal_bituminous": 2328.0,  # kg CO2e per short ton
    },
    "mobile_combustion": {
        "gasoline": 8.887,          # kg CO2e per gallon
        "diesel": 10.21,            # kg CO2e per gallon
        "cng": 0.0545,             # kg CO2e per scf
        "lng": 4.46,               # kg CO2e per gallon
    },
    "refrigerants": {
        "HFC-134a": 1430.0,        # kg CO2e per kg (GWP)
        "R-410A": 2088.0,
        "R-407C": 1774.0,
        "R-404A": 3922.0,
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/aggregate/{inventory_id}",
    response_model=Scope1AggregationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Aggregate Scope 1 emissions",
    description=(
        "Trigger aggregation of all Scope 1 emission source categories. "
        "Consolidates data from stationary combustion, mobile combustion, "
        "process emissions, fugitive emissions, refrigerants, land use, "
        "waste treatment, and agricultural sources."
    ),
)
async def aggregate_scope1(inventory_id: str) -> Scope1AggregationResponse:
    return Scope1AggregationResponse(
        inventory_id=inventory_id,
        status="completed",
        total_tco2e=12450.8,
        categories_aggregated=5,
        facilities_aggregated=4,
        data_records_processed=47,
        aggregated_at=_now(),
    )


@router.get(
    "/{inventory_id}/summary",
    response_model=Scope1Summary,
    summary="Scope 1 emissions summary",
    description=(
        "Retrieve a high-level summary of Scope 1 direct emissions including "
        "totals by gas, category count, facility count, data quality score, "
        "and year-over-year change."
    ),
)
async def get_scope1_summary(inventory_id: str) -> Scope1Summary:
    return Scope1Summary(
        inventory_id=inventory_id,
        total_tco2e=12450.8,
        total_co2_tonnes=9980.0,
        total_ch4_tco2e=1253.7,
        total_n2o_tco2e=67.1,
        total_fgas_tco2e=1150.0,
        category_count=5,
        facility_count=4,
        data_quality_score=88.5,
        calculation_completeness_pct=94.0,
        year_over_year_change_pct=-3.2,
        top_category="Stationary Combustion",
        top_facility="East Coast Manufacturing Plant",
    )


@router.get(
    "/{inventory_id}/categories",
    response_model=List[CategoryBreakdown],
    summary="Scope 1 by category",
    description=(
        "Breakdown of Scope 1 emissions by the eight source categories. "
        "Includes per-category gas splits, facility counts, data quality, "
        "and methodology details."
    ),
)
async def get_scope1_categories(
    inventory_id: str,
    include_zero: bool = Query(False, description="Include categories with zero emissions"),
) -> List[CategoryBreakdown]:
    categories = _simulated_category_data(inventory_id)
    if not include_zero:
        categories = [c for c in categories if c["total_tco2e"] > 0]
    return [CategoryBreakdown(**c) for c in categories]


@router.get(
    "/{inventory_id}/facilities",
    response_model=List[FacilityBreakdown],
    summary="Scope 1 by facility",
    description=(
        "Breakdown of Scope 1 emissions by reporting facility. "
        "Shows each facility's contribution, top emission source, "
        "and data quality score."
    ),
)
async def get_scope1_facilities(inventory_id: str) -> List[FacilityBreakdown]:
    facilities = _simulated_facility_data()
    return [FacilityBreakdown(**f) for f in facilities]


@router.get(
    "/{inventory_id}/gases",
    response_model=List[GasBreakdown],
    summary="Scope 1 by greenhouse gas",
    description=(
        "Breakdown of Scope 1 emissions by individual greenhouse gas "
        "from the Kyoto basket: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3. "
        "Uses AR5 GWP values."
    ),
)
async def get_scope1_gases(
    inventory_id: str,
    include_zero: bool = Query(False, description="Include gases with zero emissions"),
) -> List[GasBreakdown]:
    gases = _simulated_gas_data()
    if not include_zero:
        gases = [g for g in gases if g["total_tco2e"] > 0]
    return [GasBreakdown(**g) for g in gases]


@router.post(
    "/{inventory_id}/data",
    response_model=Scope1DataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Scope 1 emission data",
    description=(
        "Submit activity data for a Scope 1 emission source. "
        "Provide the category, facility, quantity, unit, and optionally "
        "a custom emission factor. The system calculates tCO2e using "
        "the appropriate emission factor and GWP."
    ),
)
async def submit_scope1_data(
    inventory_id: str,
    data: Scope1DataSubmission,
) -> Scope1DataResponse:
    # Determine emission factor
    ef = data.emission_factor
    ef_source = data.emission_factor_source or "user_provided"
    if ef is None:
        category_factors = DEFAULT_EMISSION_FACTORS.get(data.category.value, {})
        lookup_key = data.fuel_type or data.gas_type
        if lookup_key and lookup_key in category_factors:
            ef = category_factors[lookup_key]
            ef_source = "EPA GHG Emission Factors Hub"
        else:
            ef = 0.0
            ef_source = "unknown"

    # Calculate tCO2e
    calculated_tco2e = round(data.quantity * ef / 1000.0, 4)

    record_id = _generate_id("s1d")
    return Scope1DataResponse(
        record_id=record_id,
        inventory_id=inventory_id,
        category=data.category.value,
        facility_name=data.facility_name,
        quantity=data.quantity,
        unit=data.unit,
        calculated_tco2e=calculated_tco2e,
        emission_factor_used=ef,
        emission_factor_source=ef_source,
        calculation_tier=data.calculation_tier.value,
        status="accepted",
        created_at=_now(),
    )
