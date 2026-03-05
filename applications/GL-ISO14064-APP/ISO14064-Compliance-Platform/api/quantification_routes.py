"""
GL-ISO14064-APP Quantification API

Manages emission sources within ISO 14064-1 inventories.
Supports CRUD operations for emission sources and triggers deterministic
calculation of tCO2e using activity data, emission factors, and GWP values.

Quantification methods per ISO 14064-1 Clause 5.2.4:
    - Calculation-based (activity data x emission factor x GWP)
    - Direct measurement (CEMS)
    - Mass balance

GHGs (Kyoto basket): CO2, CH4, N2O, HFCs, PFCs, SF6, NF3
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import hashlib

router = APIRouter(prefix="/api/v1/iso14064/quantification", tags=["Quantification"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ISOCategory(str, Enum):
    """ISO 14064-1 emission categories."""
    CATEGORY_1_DIRECT = "category_1_direct"
    CATEGORY_2_ENERGY = "category_2_energy"
    CATEGORY_3_TRANSPORT = "category_3_transport"
    CATEGORY_4_PRODUCTS_USED = "category_4_products_used"
    CATEGORY_5_PRODUCTS_FROM_ORG = "category_5_products_from_org"
    CATEGORY_6_OTHER = "category_6_other"


class GHGGas(str, Enum):
    """Seven GHGs per ISO 14064-1:2018."""
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFCS = "HFCs"
    PFCS = "PFCs"
    SF6 = "SF6"
    NF3 = "NF3"


class QuantificationMethod(str, Enum):
    """Quantification methods per ISO 14064-1 Clause 5.2.4."""
    CALCULATION_BASED = "calculation_based"
    DIRECT_MEASUREMENT = "direct_measurement"
    MASS_BALANCE = "mass_balance"


class DataQualityTier(str, Enum):
    """Data quality tiers."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"


# ---------------------------------------------------------------------------
# GWP Lookup
# ---------------------------------------------------------------------------

GWP_AR5: Dict[str, float] = {
    "CO2": 1, "CH4": 28, "N2O": 265,
    "HFCs": 1430, "PFCs": 6630, "SF6": 23500, "NF3": 16100,
}


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class AddEmissionSourceRequest(BaseModel):
    """Request to add an emission source to an inventory."""
    category: ISOCategory = Field(..., description="ISO 14064-1 category")
    source_name: str = Field(..., min_length=1, max_length=500, description="Source description")
    facility_id: Optional[str] = Field(None, description="Facility/entity ID")
    gas: GHGGas = Field(GHGGas.CO2, description="Greenhouse gas")
    method: QuantificationMethod = Field(
        QuantificationMethod.CALCULATION_BASED, description="Quantification method"
    )
    activity_data: float = Field(0.0, ge=0, description="Activity data quantity")
    activity_unit: str = Field("", description="Activity data unit (e.g. MWh, litres, kg)")
    emission_factor: float = Field(0.0, ge=0, description="Emission factor value")
    ef_unit: str = Field("", description="Emission factor unit")
    ef_source: str = Field("", description="Emission factor source (e.g. IPCC, EPA)")
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_1, description="Data quality tier"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "category": "category_1_direct",
                "source_name": "Natural gas boiler - Building A",
                "facility_id": "ent_abc123",
                "gas": "CO2",
                "method": "calculation_based",
                "activity_data": 150000.0,
                "activity_unit": "therms",
                "emission_factor": 5.302,
                "ef_unit": "kg CO2e per therm",
                "ef_source": "EPA GHG Emission Factors Hub",
                "data_quality_tier": "tier_2",
            }
        }


class UpdateEmissionSourceRequest(BaseModel):
    """Request to update an emission source."""
    source_name: Optional[str] = Field(None, min_length=1, max_length=500)
    activity_data: Optional[float] = Field(None, ge=0)
    activity_unit: Optional[str] = None
    emission_factor: Optional[float] = Field(None, ge=0)
    ef_unit: Optional[str] = None
    ef_source: Optional[str] = None
    data_quality_tier: Optional[DataQualityTier] = None


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class EmissionSourceResponse(BaseModel):
    """An emission source within an inventory."""
    source_id: str
    inventory_id: str
    category: str
    source_name: str
    facility_id: Optional[str]
    gas: str
    method: str
    activity_data: float
    activity_unit: str
    emission_factor: float
    ef_unit: str
    ef_source: str
    gwp: float
    raw_emissions_tonnes: float
    tco2e: float
    biogenic_co2: float
    data_quality_tier: str
    provenance_hash: str
    created_at: datetime
    updated_at: datetime


class CalculationResponse(BaseModel):
    """Result from triggering emission calculation."""
    inventory_id: str
    sources_calculated: int
    total_tco2e: float
    by_category: Dict[str, float]
    by_gas: Dict[str, float]
    calculated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_sources: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _sha256(payload: str) -> str:
    """SHA-256 hex digest for provenance tracking."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _calculate_emissions(source: Dict[str, Any]) -> None:
    """Deterministic calculation: activity_data * emission_factor / 1000 * GWP."""
    gas = source["gas"]
    gwp = GWP_AR5.get(gas, 1.0)
    source["gwp"] = gwp
    raw_tonnes = source["activity_data"] * source["emission_factor"] / 1000.0
    source["raw_emissions_tonnes"] = round(raw_tonnes, 6)
    source["tco2e"] = round(raw_tonnes * gwp, 6)
    # Provenance
    payload = (
        f"{source['source_id']}:{source['category']}:{source['gas']}:"
        f"{source['activity_data']}:{source['emission_factor']}:{source['tco2e']}"
    )
    source["provenance_hash"] = _sha256(payload)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{inventory_id}/sources",
    response_model=EmissionSourceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add emission source",
    description=(
        "Add an emission source to an ISO 14064-1 inventory.  Provide activity "
        "data, emission factor, and gas type.  The system calculates tCO2e "
        "deterministically using the formula: activity_data * EF / 1000 * GWP."
    ),
)
async def add_emission_source(
    inventory_id: str,
    request: AddEmissionSourceRequest,
) -> EmissionSourceResponse:
    """Add an emission source and calculate tCO2e."""
    source_id = _generate_id("src")
    now = _now()
    source = {
        "source_id": source_id,
        "inventory_id": inventory_id,
        "category": request.category.value,
        "source_name": request.source_name,
        "facility_id": request.facility_id,
        "gas": request.gas.value,
        "method": request.method.value,
        "activity_data": request.activity_data,
        "activity_unit": request.activity_unit,
        "emission_factor": request.emission_factor,
        "ef_unit": request.ef_unit,
        "ef_source": request.ef_source,
        "gwp": 1.0,
        "raw_emissions_tonnes": 0.0,
        "tco2e": 0.0,
        "biogenic_co2": 0.0,
        "data_quality_tier": request.data_quality_tier.value,
        "provenance_hash": "",
        "created_at": now,
        "updated_at": now,
    }
    _calculate_emissions(source)
    _sources[source_id] = source
    return EmissionSourceResponse(**source)


@router.get(
    "/{inventory_id}/sources",
    response_model=List[EmissionSourceResponse],
    summary="List emission sources",
    description="Retrieve all emission sources for an inventory, optionally filtered by category or gas.",
)
async def list_emission_sources(
    inventory_id: str,
    category: Optional[str] = Query(None, description="Filter by ISO category"),
    gas: Optional[str] = Query(None, description="Filter by GHG gas"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[EmissionSourceResponse]:
    """List emission sources for an inventory."""
    sources = [s for s in _sources.values() if s["inventory_id"] == inventory_id]
    if category:
        sources = [s for s in sources if s["category"] == category]
    if gas:
        sources = [s for s in sources if s["gas"] == gas]
    sources.sort(key=lambda s: s["tco2e"], reverse=True)
    return [EmissionSourceResponse(**s) for s in sources[:limit]]


@router.get(
    "/{inventory_id}/sources/{source_id}",
    response_model=EmissionSourceResponse,
    summary="Get emission source",
    description="Retrieve a single emission source by ID.",
)
async def get_emission_source(
    inventory_id: str,
    source_id: str,
) -> EmissionSourceResponse:
    """Retrieve a specific emission source."""
    source = _sources.get(source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emission source {source_id} not found",
        )
    if source["inventory_id"] != inventory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Source {source_id} does not belong to inventory {inventory_id}",
        )
    return EmissionSourceResponse(**source)


@router.put(
    "/{inventory_id}/sources/{source_id}",
    response_model=EmissionSourceResponse,
    summary="Update emission source",
    description="Update an emission source and recalculate tCO2e.",
)
async def update_emission_source(
    inventory_id: str,
    source_id: str,
    request: UpdateEmissionSourceRequest,
) -> EmissionSourceResponse:
    """Update an emission source and recalculate."""
    source = _sources.get(source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emission source {source_id} not found",
        )
    if source["inventory_id"] != inventory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Source {source_id} does not belong to inventory {inventory_id}",
        )
    updates = request.model_dump(exclude_unset=True)
    if "data_quality_tier" in updates:
        updates["data_quality_tier"] = updates["data_quality_tier"].value if hasattr(updates["data_quality_tier"], "value") else updates["data_quality_tier"]
    source.update(updates)
    _calculate_emissions(source)
    source["updated_at"] = _now()
    return EmissionSourceResponse(**source)


@router.delete(
    "/{inventory_id}/sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete emission source",
    description="Remove an emission source from the inventory.",
)
async def delete_emission_source(
    inventory_id: str,
    source_id: str,
) -> None:
    """Delete an emission source."""
    source = _sources.get(source_id)
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Emission source {source_id} not found",
        )
    if source["inventory_id"] != inventory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Source {source_id} does not belong to inventory {inventory_id}",
        )
    del _sources[source_id]
    return None


@router.post(
    "/{inventory_id}/calculate",
    response_model=CalculationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger emission calculation",
    description=(
        "Trigger recalculation of all emission sources for an inventory. "
        "Uses deterministic formula: activity_data * emission_factor / 1000 * GWP. "
        "Returns totals by category and by gas."
    ),
)
async def trigger_calculation(inventory_id: str) -> CalculationResponse:
    """Recalculate all emission sources for an inventory."""
    sources = [s for s in _sources.values() if s["inventory_id"] == inventory_id]
    by_category: Dict[str, float] = {}
    by_gas: Dict[str, float] = {}
    for source in sources:
        _calculate_emissions(source)
        source["updated_at"] = _now()
        cat = source["category"]
        gas = source["gas"]
        by_category[cat] = round(by_category.get(cat, 0.0) + source["tco2e"], 6)
        by_gas[gas] = round(by_gas.get(gas, 0.0) + source["tco2e"], 6)
    total_tco2e = round(sum(by_category.values()), 6)
    return CalculationResponse(
        inventory_id=inventory_id,
        sources_calculated=len(sources),
        total_tco2e=total_tco2e,
        by_category=by_category,
        by_gas=by_gas,
        calculated_at=_now(),
    )
