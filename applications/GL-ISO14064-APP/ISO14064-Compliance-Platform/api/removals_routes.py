"""
GL-ISO14064-APP Removals API

Manages GHG removal sources within ISO 14064-1 inventories per Clause 6.2.
Removals represent GHG absorbed from the atmosphere through activities such
as forestry, soil carbon sequestration, CCS, direct air capture (DAC), and
wetland restoration.

Each removal includes:
    - Gross removal quantity (tCO2e)
    - Permanence classification and discount factor
    - Credited (permanence-adjusted) removal
    - Monitoring plan reference

Permanence levels: permanent (>1000yr), long-term (100-1000yr),
                   medium-term (25-100yr), short-term (5-25yr),
                   reversible (<5yr)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import hashlib

router = APIRouter(prefix="/api/v1/iso14064/removals", tags=["Removals"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RemovalType(str, Enum):
    """Types of GHG removals per ISO 14064-1:2018 Clause 5.2.3."""
    FORESTRY = "forestry"
    SOIL_CARBON = "soil_carbon"
    CCS = "ccs"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BECCS = "beccs"
    WETLAND_RESTORATION = "wetland_restoration"
    OCEAN_BASED = "ocean_based"
    OTHER = "other"


class PermanenceLevel(str, Enum):
    """Permanence classification for GHG removals."""
    PERMANENT = "permanent"
    LONG_TERM = "long_term"
    MEDIUM_TERM = "medium_term"
    SHORT_TERM = "short_term"
    REVERSIBLE = "reversible"


class DataQualityTier(str, Enum):
    """Data quality tiers."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"


# ---------------------------------------------------------------------------
# Permanence Discount Factors
# ---------------------------------------------------------------------------

PERMANENCE_DISCOUNT: Dict[str, float] = {
    "permanent": 1.0,
    "long_term": 0.95,
    "medium_term": 0.80,
    "short_term": 0.50,
    "reversible": 0.20,
}


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class AddRemovalSourceRequest(BaseModel):
    """Request to add a removal source to an inventory."""
    removal_type: RemovalType = Field(..., description="Type of removal activity")
    source_name: str = Field(..., min_length=1, max_length=500, description="Removal source description")
    facility_id: Optional[str] = Field(None, description="Facility/entity ID")
    gross_removals_tco2e: float = Field(..., ge=0, description="Gross removals before permanence adjustment")
    permanence_level: PermanenceLevel = Field(
        PermanenceLevel.LONG_TERM, description="Permanence classification"
    )
    monitoring_plan: Optional[str] = Field(None, max_length=1000, description="Monitoring plan reference")
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_2, description="Data quality tier"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "removal_type": "forestry",
                "source_name": "Reforestation project - Northern parcels",
                "facility_id": "ent_abc123",
                "gross_removals_tco2e": 1250.0,
                "permanence_level": "long_term",
                "monitoring_plan": "MP-FOR-2025-001",
                "data_quality_tier": "tier_2",
            }
        }


class UpdateRemovalSourceRequest(BaseModel):
    """Request to update a removal source."""
    source_name: Optional[str] = Field(None, min_length=1, max_length=500)
    gross_removals_tco2e: Optional[float] = Field(None, ge=0)
    permanence_level: Optional[PermanenceLevel] = None
    monitoring_plan: Optional[str] = Field(None, max_length=1000)
    data_quality_tier: Optional[DataQualityTier] = None


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class RemovalSourceResponse(BaseModel):
    """A GHG removal source within an inventory."""
    removal_id: str
    inventory_id: str
    removal_type: str
    source_name: str
    facility_id: Optional[str]
    gross_removals_tco2e: float
    permanence_level: str
    permanence_discount_factor: float
    credited_removals_tco2e: float
    monitoring_plan: Optional[str]
    data_quality_tier: str
    provenance_hash: str
    created_at: datetime
    updated_at: datetime


class RemovalSummaryResponse(BaseModel):
    """Summary of all removals for an inventory."""
    inventory_id: str
    total_gross_removals_tco2e: float
    total_credited_removals_tco2e: float
    removal_count: int
    by_type: Dict[str, float]
    by_permanence: Dict[str, float]
    calculated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_removals: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _sha256(payload: str) -> str:
    """SHA-256 hex digest for provenance tracking."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_credited_removal(removal: Dict[str, Any]) -> None:
    """Apply permanence discount to gross removal."""
    perm_level = removal["permanence_level"]
    discount = PERMANENCE_DISCOUNT.get(perm_level, 1.0)
    removal["permanence_discount_factor"] = discount
    removal["credited_removals_tco2e"] = round(
        removal["gross_removals_tco2e"] * discount, 6
    )
    payload = (
        f"{removal['removal_id']}:{removal['removal_type']}:"
        f"{removal['gross_removals_tco2e']}:{removal['permanence_level']}"
    )
    removal["provenance_hash"] = _sha256(payload)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{inventory_id}",
    response_model=RemovalSourceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add removal source",
    description=(
        "Add a GHG removal source to an ISO 14064-1 inventory. "
        "The system applies the permanence discount factor to compute "
        "the credited removal amount."
    ),
)
async def add_removal_source(
    inventory_id: str,
    request: AddRemovalSourceRequest,
) -> RemovalSourceResponse:
    """Add a removal source and compute credited removals."""
    removal_id = _generate_id("rmv")
    now = _now()
    removal = {
        "removal_id": removal_id,
        "inventory_id": inventory_id,
        "removal_type": request.removal_type.value,
        "source_name": request.source_name,
        "facility_id": request.facility_id,
        "gross_removals_tco2e": request.gross_removals_tco2e,
        "permanence_level": request.permanence_level.value,
        "permanence_discount_factor": 1.0,
        "credited_removals_tco2e": 0.0,
        "monitoring_plan": request.monitoring_plan,
        "data_quality_tier": request.data_quality_tier.value,
        "provenance_hash": "",
        "created_at": now,
        "updated_at": now,
    }
    _compute_credited_removal(removal)
    _removals[removal_id] = removal
    return RemovalSourceResponse(**removal)


@router.get(
    "/{inventory_id}",
    response_model=List[RemovalSourceResponse],
    summary="List removal sources",
    description="Retrieve all removal sources for an inventory.",
)
async def list_removal_sources(
    inventory_id: str,
    removal_type: Optional[str] = Query(None, description="Filter by removal type"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[RemovalSourceResponse]:
    """List removal sources for an inventory."""
    removals = [r for r in _removals.values() if r["inventory_id"] == inventory_id]
    if removal_type:
        removals = [r for r in removals if r["removal_type"] == removal_type]
    removals.sort(key=lambda r: r["credited_removals_tco2e"], reverse=True)
    return [RemovalSourceResponse(**r) for r in removals[:limit]]


@router.get(
    "/{inventory_id}/{removal_id}",
    response_model=RemovalSourceResponse,
    summary="Get removal source",
    description="Retrieve a single removal source by ID.",
)
async def get_removal_source(
    inventory_id: str,
    removal_id: str,
) -> RemovalSourceResponse:
    """Retrieve a specific removal source."""
    removal = _removals.get(removal_id)
    if not removal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Removal source {removal_id} not found",
        )
    if removal["inventory_id"] != inventory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Removal {removal_id} does not belong to inventory {inventory_id}",
        )
    return RemovalSourceResponse(**removal)


@router.put(
    "/{inventory_id}/{removal_id}",
    response_model=RemovalSourceResponse,
    summary="Update removal source",
    description="Update a removal source and recalculate credited removals.",
)
async def update_removal_source(
    inventory_id: str,
    removal_id: str,
    request: UpdateRemovalSourceRequest,
) -> RemovalSourceResponse:
    """Update a removal source and recompute credited removals."""
    removal = _removals.get(removal_id)
    if not removal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Removal source {removal_id} not found",
        )
    if removal["inventory_id"] != inventory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Removal {removal_id} does not belong to inventory {inventory_id}",
        )
    updates = request.model_dump(exclude_unset=True)
    if "permanence_level" in updates:
        updates["permanence_level"] = updates["permanence_level"].value if hasattr(updates["permanence_level"], "value") else updates["permanence_level"]
    if "data_quality_tier" in updates:
        updates["data_quality_tier"] = updates["data_quality_tier"].value if hasattr(updates["data_quality_tier"], "value") else updates["data_quality_tier"]
    removal.update(updates)
    _compute_credited_removal(removal)
    removal["updated_at"] = _now()
    return RemovalSourceResponse(**removal)


@router.delete(
    "/{inventory_id}/{removal_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete removal source",
    description="Remove a GHG removal source from the inventory.",
)
async def delete_removal_source(
    inventory_id: str,
    removal_id: str,
) -> None:
    """Delete a removal source."""
    removal = _removals.get(removal_id)
    if not removal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Removal source {removal_id} not found",
        )
    if removal["inventory_id"] != inventory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Removal {removal_id} does not belong to inventory {inventory_id}",
        )
    del _removals[removal_id]
    return None


@router.get(
    "/{inventory_id}/summary",
    response_model=RemovalSummaryResponse,
    summary="Removal summary",
    description="Aggregated summary of all removals for an inventory by type and permanence level.",
)
async def get_removal_summary(inventory_id: str) -> RemovalSummaryResponse:
    """Get aggregated removal summary for an inventory."""
    removals = [r for r in _removals.values() if r["inventory_id"] == inventory_id]
    total_gross = 0.0
    total_credited = 0.0
    by_type: Dict[str, float] = {}
    by_permanence: Dict[str, float] = {}
    for r in removals:
        total_gross += r["gross_removals_tco2e"]
        total_credited += r["credited_removals_tco2e"]
        rtype = r["removal_type"]
        perm = r["permanence_level"]
        by_type[rtype] = round(by_type.get(rtype, 0.0) + r["credited_removals_tco2e"], 6)
        by_permanence[perm] = round(by_permanence.get(perm, 0.0) + r["credited_removals_tco2e"], 6)
    return RemovalSummaryResponse(
        inventory_id=inventory_id,
        total_gross_removals_tco2e=round(total_gross, 6),
        total_credited_removals_tco2e=round(total_credited, 6),
        removal_count=len(removals),
        by_type=by_type,
        by_permanence=by_permanence,
        calculated_at=_now(),
    )
