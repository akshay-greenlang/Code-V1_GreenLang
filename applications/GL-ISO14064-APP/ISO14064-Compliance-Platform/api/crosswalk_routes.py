"""
GL-ISO14064-APP Crosswalk API

Generates crosswalk mappings between ISO 14064-1:2018 six-category model
and the GHG Protocol three-scope model.  This enables organizations that
report under both standards to reconcile their inventories.

Mapping:
    ISO Category 1 (Direct)         ->  GHG Protocol Scope 1
    ISO Category 2 (Energy)         ->  GHG Protocol Scope 2
    ISO Category 3 (Transport)      ->  GHG Protocol Scope 3 (Cat 4, 6, 7, 9)
    ISO Category 4 (Products used)  ->  GHG Protocol Scope 3 (Cat 1, 2, 3, 5, 8)
    ISO Category 5 (Products from)  ->  GHG Protocol Scope 3 (Cat 10, 11, 12, 13, 14)
    ISO Category 6 (Other)          ->  GHG Protocol Scope 3 (Cat 15)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/iso14064/crosswalk", tags=["Crosswalk"])


# ---------------------------------------------------------------------------
# ISO -> GHG Protocol Mapping Definition
# ---------------------------------------------------------------------------

CROSSWALK_DEFINITION = [
    {
        "iso_category": "category_1_direct",
        "iso_category_name": "Category 1 - Direct GHG emissions and removals",
        "ghg_scope": "scope_1",
        "ghg_categories": [
            "Stationary combustion", "Mobile combustion", "Process emissions",
            "Fugitive emissions", "Refrigerants", "Land use", "Waste treatment",
            "Agricultural",
        ],
        "notes": "Direct one-to-one mapping between ISO Category 1 and GHG Protocol Scope 1",
    },
    {
        "iso_category": "category_2_energy",
        "iso_category_name": "Category 2 - Indirect GHG emissions from imported energy",
        "ghg_scope": "scope_2",
        "ghg_categories": [
            "Purchased electricity (location-based)", "Purchased electricity (market-based)",
            "Purchased steam/heat", "Purchased cooling",
        ],
        "notes": "Direct one-to-one mapping between ISO Category 2 and GHG Protocol Scope 2",
    },
    {
        "iso_category": "category_3_transport",
        "iso_category_name": "Category 3 - Indirect GHG emissions from transportation",
        "ghg_scope": "scope_3",
        "ghg_categories": [
            "Cat 4: Upstream transportation", "Cat 6: Business travel",
            "Cat 7: Employee commuting", "Cat 9: Downstream transportation",
        ],
        "notes": "Maps to GHG Protocol Scope 3 transportation-related categories",
    },
    {
        "iso_category": "category_4_products_used",
        "iso_category_name": "Category 4 - Indirect GHG emissions from products used by the organization",
        "ghg_scope": "scope_3",
        "ghg_categories": [
            "Cat 1: Purchased goods & services", "Cat 2: Capital goods",
            "Cat 3: Fuel & energy activities", "Cat 5: Waste generated in operations",
            "Cat 8: Upstream leased assets",
        ],
        "notes": "Maps to GHG Protocol Scope 3 upstream product-related categories",
    },
    {
        "iso_category": "category_5_products_from_org",
        "iso_category_name": "Category 5 - Indirect GHG emissions from the use of products from the organization",
        "ghg_scope": "scope_3",
        "ghg_categories": [
            "Cat 10: Processing of sold products", "Cat 11: Use of sold products",
            "Cat 12: End-of-life treatment", "Cat 13: Downstream leased assets",
            "Cat 14: Franchises",
        ],
        "notes": "Maps to GHG Protocol Scope 3 downstream product-related categories",
    },
    {
        "iso_category": "category_6_other",
        "iso_category_name": "Category 6 - Indirect GHG emissions from other sources",
        "ghg_scope": "scope_3",
        "ghg_categories": ["Cat 15: Investments"],
        "notes": "Maps to GHG Protocol Scope 3 Category 15 (Investments)",
    },
]


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class GenerateCrosswalkRequest(BaseModel):
    """Request to generate a crosswalk between ISO 14064-1 and GHG Protocol."""
    inventory_id: str = Field(..., description="Inventory to generate crosswalk for")

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_id": "inv_abc123",
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class CrosswalkMappingResponse(BaseModel):
    """A single mapping between ISO 14064-1 and GHG Protocol."""
    iso_category: str
    iso_category_name: str
    ghg_scope: str
    ghg_categories: List[str]
    iso_tco2e: float
    ghg_tco2e: float
    difference_tco2e: float
    notes: str


class CrosswalkResultResponse(BaseModel):
    """Full crosswalk result."""
    crosswalk_id: str
    inventory_id: str
    mappings: List[CrosswalkMappingResponse]
    iso_total_tco2e: float
    ghg_protocol_total_tco2e: float
    reconciliation_difference_tco2e: float
    reconciliation_pct: float
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_crosswalks: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Simulated Emissions by Category (for demo)
# ---------------------------------------------------------------------------

SIMULATED_EMISSIONS = {
    "category_1_direct": 12450.8,
    "category_2_energy": 8320.5,
    "category_3_transport": 9840.0,
    "category_4_products_used": 26180.2,
    "category_5_products_from_org": 8450.0,
    "category_6_other": 760.0,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=CrosswalkResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate crosswalk",
    description=(
        "Generate a crosswalk mapping between the ISO 14064-1:2018 six-category "
        "model and the GHG Protocol three-scope model.  The result shows how "
        "each ISO category maps to GHG Protocol scopes and Scope 3 categories, "
        "with emissions totals for reconciliation."
    ),
)
async def generate_crosswalk(
    request: GenerateCrosswalkRequest,
) -> CrosswalkResultResponse:
    """Generate a crosswalk between ISO 14064-1 and GHG Protocol."""
    crosswalk_id = _generate_id("xwalk")
    now = _now()
    mappings = []
    iso_total = 0.0
    ghg_total = 0.0
    for defn in CROSSWALK_DEFINITION:
        iso_cat = defn["iso_category"]
        iso_emissions = SIMULATED_EMISSIONS.get(iso_cat, 0.0)
        # GHG Protocol side uses same values (reconciled)
        ghg_emissions = iso_emissions
        iso_total += iso_emissions
        ghg_total += ghg_emissions
        mappings.append({
            "iso_category": iso_cat,
            "iso_category_name": defn["iso_category_name"],
            "ghg_scope": defn["ghg_scope"],
            "ghg_categories": defn["ghg_categories"],
            "iso_tco2e": iso_emissions,
            "ghg_tco2e": ghg_emissions,
            "difference_tco2e": round(iso_emissions - ghg_emissions, 6),
            "notes": defn["notes"],
        })
    reconciliation_diff = round(iso_total - ghg_total, 6)
    reconciliation_pct = round(
        abs(reconciliation_diff) / iso_total * 100, 2
    ) if iso_total > 0 else 0.0
    crosswalk = {
        "crosswalk_id": crosswalk_id,
        "inventory_id": request.inventory_id,
        "mappings": mappings,
        "iso_total_tco2e": round(iso_total, 2),
        "ghg_protocol_total_tco2e": round(ghg_total, 2),
        "reconciliation_difference_tco2e": reconciliation_diff,
        "reconciliation_pct": reconciliation_pct,
        "generated_at": now,
    }
    _crosswalks[crosswalk_id] = crosswalk
    return CrosswalkResultResponse(**crosswalk)


@router.get(
    "/{crosswalk_id}",
    response_model=CrosswalkResultResponse,
    summary="Get crosswalk result",
    description="Retrieve a previously generated crosswalk result.",
)
async def get_crosswalk(crosswalk_id: str) -> CrosswalkResultResponse:
    """Retrieve a crosswalk result by ID."""
    crosswalk = _crosswalks.get(crosswalk_id)
    if not crosswalk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crosswalk {crosswalk_id} not found",
        )
    return CrosswalkResultResponse(**crosswalk)


@router.get(
    "/history/{inventory_id}",
    response_model=List[CrosswalkResultResponse],
    summary="Crosswalk history",
    description="List all crosswalk results generated for an inventory.",
)
async def get_crosswalk_history(
    inventory_id: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
) -> List[CrosswalkResultResponse]:
    """List crosswalk history for an inventory."""
    crosswalks = [
        c for c in _crosswalks.values() if c["inventory_id"] == inventory_id
    ]
    crosswalks.sort(key=lambda c: c["generated_at"], reverse=True)
    return [CrosswalkResultResponse(**c) for c in crosswalks[:limit]]
