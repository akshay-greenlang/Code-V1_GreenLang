"""
GL-ISO14064-APP Inventory Lifecycle API

Manages ISO 14064-1 GHG inventories: creation, retrieval, updates,
and status transitions through the inventory lifecycle.

Inventory statuses: draft -> in_review -> approved -> verified -> published

Each inventory represents one organization-year GHG accounting period
with a chosen consolidation approach and GWP source.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/iso14064/inventories", tags=["Inventories"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InventoryStatus(str, Enum):
    """Lifecycle status of an ISO 14064-1 GHG inventory."""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    VERIFIED = "verified"
    PUBLISHED = "published"


class ConsolidationApproach(str, Enum):
    """Consolidation approach."""
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class GWPSource(str, Enum):
    """GWP assessment report source."""
    AR5 = "ar5"
    AR6 = "ar6"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateInventoryRequest(BaseModel):
    """Request to create an ISO 14064-1 inventory."""
    org_id: str = Field(..., description="Organization ID")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL, description="Consolidation approach"
    )
    gwp_source: GWPSource = Field(GWPSource.AR5, description="GWP assessment report")

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "reporting_year": 2025,
                "consolidation_approach": "operational_control",
                "gwp_source": "ar5",
            }
        }


class UpdateInventoryRequest(BaseModel):
    """Request to update an inventory."""
    consolidation_approach: Optional[ConsolidationApproach] = None
    gwp_source: Optional[GWPSource] = None


class TransitionStatusRequest(BaseModel):
    """Request to transition inventory status."""
    target_status: InventoryStatus = Field(
        ..., description="Target status for the inventory"
    )
    notes: Optional[str] = Field(None, max_length=2000, description="Transition notes")

    class Config:
        json_schema_extra = {
            "example": {
                "target_status": "in_review",
                "notes": "Ready for internal review",
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class InventoryResponse(BaseModel):
    """ISO 14064-1 inventory."""
    inventory_id: str
    org_id: str
    reporting_year: int
    consolidation_approach: str
    gwp_source: str
    status: str
    total_emissions_tco2e: float
    total_removals_tco2e: float
    net_emissions_tco2e: float
    source_count: int
    removal_count: int
    created_at: datetime
    updated_at: datetime


class InventoryListEntry(BaseModel):
    """Summary entry in an inventory list."""
    inventory_id: str
    org_id: str
    reporting_year: int
    status: str
    total_emissions_tco2e: float
    net_emissions_tco2e: float
    created_at: datetime


# ---------------------------------------------------------------------------
# Valid Status Transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Dict[str, List[str]] = {
    "draft": ["in_review"],
    "in_review": ["draft", "approved"],
    "approved": ["in_review", "verified"],
    "verified": ["approved", "published"],
    "published": ["verified"],
}


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_inventories: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=InventoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create ISO 14064-1 inventory",
    description=(
        "Create a new GHG inventory for the given organization and reporting year. "
        "Initializes with zero emissions.  Populate via quantification and removal endpoints."
    ),
)
async def create_inventory(request: CreateInventoryRequest) -> InventoryResponse:
    """Create a new ISO 14064-1 GHG inventory."""
    existing = [
        inv for inv in _inventories.values()
        if inv["org_id"] == request.org_id and inv["reporting_year"] == request.reporting_year
    ]
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Inventory for year {request.reporting_year} already exists: "
                f"{existing[0]['inventory_id']}"
            ),
        )
    inventory_id = _generate_id("inv")
    now = _now()
    inventory = {
        "inventory_id": inventory_id,
        "org_id": request.org_id,
        "reporting_year": request.reporting_year,
        "consolidation_approach": request.consolidation_approach.value,
        "gwp_source": request.gwp_source.value,
        "status": InventoryStatus.DRAFT.value,
        "total_emissions_tco2e": 0.0,
        "total_removals_tco2e": 0.0,
        "net_emissions_tco2e": 0.0,
        "source_count": 0,
        "removal_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    _inventories[inventory_id] = inventory
    return InventoryResponse(**inventory)


@router.get(
    "/{inventory_id}",
    response_model=InventoryResponse,
    summary="Get inventory",
    description="Retrieve the full ISO 14064-1 inventory including totals and status.",
)
async def get_inventory(inventory_id: str) -> InventoryResponse:
    """Retrieve an inventory by ID."""
    inventory = _inventories.get(inventory_id)
    if not inventory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory {inventory_id} not found",
        )
    return InventoryResponse(**inventory)


@router.get(
    "",
    response_model=List[InventoryListEntry],
    summary="List inventories for organization",
    description="Retrieve all inventories for an organization, optionally filtered by year or status.",
)
async def list_inventories(
    org_id: str = Query(..., description="Organization ID"),
    reporting_year: Optional[int] = Query(None, ge=1990, le=2100, description="Filter by year"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> List[InventoryListEntry]:
    """List inventories for an organization."""
    inventories = [inv for inv in _inventories.values() if inv["org_id"] == org_id]
    if reporting_year is not None:
        inventories = [inv for inv in inventories if inv["reporting_year"] == reporting_year]
    if status_filter is not None:
        inventories = [inv for inv in inventories if inv["status"] == status_filter]
    inventories.sort(key=lambda i: i["reporting_year"], reverse=True)
    return [
        InventoryListEntry(
            inventory_id=inv["inventory_id"],
            org_id=inv["org_id"],
            reporting_year=inv["reporting_year"],
            status=inv["status"],
            total_emissions_tco2e=inv["total_emissions_tco2e"],
            net_emissions_tco2e=inv["net_emissions_tco2e"],
            created_at=inv["created_at"],
        )
        for inv in inventories[:limit]
    ]


@router.put(
    "/{inventory_id}",
    response_model=InventoryResponse,
    summary="Update inventory settings",
    description="Update consolidation approach or GWP source.  Only allowed in draft status.",
)
async def update_inventory(
    inventory_id: str,
    request: UpdateInventoryRequest,
) -> InventoryResponse:
    """Update inventory settings."""
    inventory = _inventories.get(inventory_id)
    if not inventory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory {inventory_id} not found",
        )
    if inventory["status"] != InventoryStatus.DRAFT.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inventory settings can only be changed in draft status.",
        )
    updates = request.model_dump(exclude_unset=True)
    if "consolidation_approach" in updates:
        inventory["consolidation_approach"] = updates["consolidation_approach"]
    if "gwp_source" in updates:
        inventory["gwp_source"] = updates["gwp_source"]
    inventory["updated_at"] = _now()
    return InventoryResponse(**inventory)


@router.post(
    "/{inventory_id}/transition",
    response_model=InventoryResponse,
    summary="Transition inventory status",
    description=(
        "Transition the inventory to a new lifecycle status. "
        "Valid transitions: draft->in_review, in_review->draft|approved, "
        "approved->in_review|verified, verified->approved|published, "
        "published->verified."
    ),
)
async def transition_status(
    inventory_id: str,
    request: TransitionStatusRequest,
) -> InventoryResponse:
    """Transition inventory status through the lifecycle."""
    inventory = _inventories.get(inventory_id)
    if not inventory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory {inventory_id} not found",
        )
    current_status = inventory["status"]
    target = request.target_status.value
    allowed = VALID_TRANSITIONS.get(current_status, [])
    if target not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot transition from '{current_status}' to '{target}'. "
                f"Allowed transitions: {allowed}"
            ),
        )
    inventory["status"] = target
    inventory["updated_at"] = _now()
    return InventoryResponse(**inventory)
