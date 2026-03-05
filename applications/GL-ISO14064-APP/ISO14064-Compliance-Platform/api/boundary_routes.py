"""
GL-ISO14064-APP Boundary Management API

Manages organizational boundaries (consolidation approach, entity inclusion)
and operational boundaries (category inclusion, significance assessments)
per ISO 14064-1:2018 Clauses 5.1 and 5.2.

Consolidation approaches:
    - Operational Control
    - Financial Control
    - Equity Share

ISO Categories 1-6:
    1. Direct GHG emissions and removals
    2. Indirect GHG emissions from imported energy
    3. Indirect GHG emissions from transportation
    4. Indirect GHG emissions from products used by the organization
    5. Indirect GHG emissions from the use of products from the organization
    6. Indirect GHG emissions from other sources
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/iso14064/boundaries", tags=["Boundaries"])


# ---------------------------------------------------------------------------
# Enums (local for route type safety)
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    """ISO 14064-1:2018 Clause 5.1 organizational boundary approaches."""
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class ISOCategory(str, Enum):
    """ISO 14064-1:2018 Clause 5.2.2 six emission/removal categories."""
    CATEGORY_1_DIRECT = "category_1_direct"
    CATEGORY_2_ENERGY = "category_2_energy"
    CATEGORY_3_TRANSPORT = "category_3_transport"
    CATEGORY_4_PRODUCTS_USED = "category_4_products_used"
    CATEGORY_5_PRODUCTS_FROM_ORG = "category_5_products_from_org"
    CATEGORY_6_OTHER = "category_6_other"


class SignificanceLevel(str, Enum):
    """Significance assessment outcome."""
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    UNDER_REVIEW = "under_review"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class SetOrganizationalBoundaryRequest(BaseModel):
    """Request to set the organizational boundary per ISO 14064-1 Clause 5.1."""
    consolidation_approach: ConsolidationApproach = Field(
        ..., description="Consolidation approach for boundary setting"
    )
    entity_ids: List[str] = Field(
        default_factory=list, description="Entity IDs to include in the boundary"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "consolidation_approach": "operational_control",
                "entity_ids": ["ent_abc123", "ent_def456"],
            }
        }


class UpdateOrganizationalBoundaryRequest(BaseModel):
    """Request to update the organizational boundary."""
    consolidation_approach: Optional[ConsolidationApproach] = None
    entity_ids: Optional[List[str]] = None


class CategoryInclusionRequest(BaseModel):
    """Inclusion decision for a single ISO category."""
    category: ISOCategory = Field(..., description="ISO 14064-1 category")
    included: bool = Field(True, description="Whether category is included")
    significance: SignificanceLevel = Field(
        SignificanceLevel.SIGNIFICANT, description="Significance assessment result"
    )
    justification: Optional[str] = Field(
        None, max_length=2000, description="Reason if excluded or not significant"
    )


class SetOperationalBoundaryRequest(BaseModel):
    """Request to set the operational boundary per ISO 14064-1 Clause 5.2."""
    categories: List[CategoryInclusionRequest] = Field(
        ..., min_length=1, description="Category inclusion decisions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "categories": [
                    {"category": "category_1_direct", "included": True, "significance": "significant"},
                    {"category": "category_2_energy", "included": True, "significance": "significant"},
                    {"category": "category_3_transport", "included": True, "significance": "significant"},
                    {"category": "category_4_products_used", "included": True, "significance": "significant"},
                    {"category": "category_5_products_from_org", "included": False, "significance": "not_significant", "justification": "Below 1% significance threshold"},
                    {"category": "category_6_other", "included": True, "significance": "significant"},
                ]
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class OrganizationalBoundaryResponse(BaseModel):
    """Organizational boundary configuration."""
    boundary_id: str
    org_id: str
    consolidation_approach: str
    entity_ids: List[str]
    entity_count: int
    created_at: datetime
    updated_at: datetime


class CategoryInclusionResponse(BaseModel):
    """Inclusion status of a single ISO category."""
    category: str
    category_name: str
    included: bool
    significance: str
    justification: Optional[str]


class OperationalBoundaryResponse(BaseModel):
    """Operational boundary configuration."""
    boundary_id: str
    org_id: str
    categories: List[CategoryInclusionResponse]
    included_count: int
    excluded_count: int
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_org_boundaries: Dict[str, Dict[str, Any]] = {}
_op_boundaries: Dict[str, Dict[str, Any]] = {}

ISO_CATEGORY_NAMES = {
    "category_1_direct": "Category 1 - Direct GHG emissions and removals",
    "category_2_energy": "Category 2 - Indirect GHG emissions from imported energy",
    "category_3_transport": "Category 3 - Indirect GHG emissions from transportation",
    "category_4_products_used": "Category 4 - Indirect GHG emissions from products used by the organization",
    "category_5_products_from_org": "Category 5 - Indirect GHG emissions from the use of products from the organization",
    "category_6_other": "Category 6 - Indirect GHG emissions from other sources",
}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints -- Organizational Boundary
# ---------------------------------------------------------------------------

@router.post(
    "/organizational/{org_id}",
    response_model=OrganizationalBoundaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Set organizational boundary",
    description=(
        "Set the organizational boundary per ISO 14064-1 Clause 5.1.  "
        "Define the consolidation approach and which entities are included "
        "in the GHG inventory boundary."
    ),
)
async def set_organizational_boundary(
    org_id: str,
    request: SetOrganizationalBoundaryRequest,
) -> OrganizationalBoundaryResponse:
    """Set the organizational boundary for an organization."""
    boundary_id = _generate_id("obnd")
    now = _now()
    boundary = {
        "boundary_id": boundary_id,
        "org_id": org_id,
        "consolidation_approach": request.consolidation_approach.value,
        "entity_ids": request.entity_ids,
        "entity_count": len(request.entity_ids),
        "created_at": now,
        "updated_at": now,
    }
    _org_boundaries[org_id] = boundary
    return OrganizationalBoundaryResponse(**boundary)


@router.get(
    "/organizational/{org_id}",
    response_model=OrganizationalBoundaryResponse,
    summary="Get organizational boundary",
    description="Retrieve the organizational boundary for an organization.",
)
async def get_organizational_boundary(org_id: str) -> OrganizationalBoundaryResponse:
    """Retrieve the organizational boundary."""
    boundary = _org_boundaries.get(org_id)
    if not boundary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No organizational boundary defined for organization {org_id}. Set one first via POST.",
        )
    return OrganizationalBoundaryResponse(**boundary)


@router.put(
    "/organizational/{org_id}",
    response_model=OrganizationalBoundaryResponse,
    summary="Update organizational boundary",
    description="Update the organizational boundary consolidation approach or entity inclusion.",
)
async def update_organizational_boundary(
    org_id: str,
    request: UpdateOrganizationalBoundaryRequest,
) -> OrganizationalBoundaryResponse:
    """Update the organizational boundary."""
    boundary = _org_boundaries.get(org_id)
    if not boundary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No organizational boundary defined for organization {org_id}",
        )
    updates = request.model_dump(exclude_unset=True)
    if "consolidation_approach" in updates:
        boundary["consolidation_approach"] = updates["consolidation_approach"]
    if "entity_ids" in updates:
        boundary["entity_ids"] = updates["entity_ids"]
        boundary["entity_count"] = len(updates["entity_ids"])
    boundary["updated_at"] = _now()
    return OrganizationalBoundaryResponse(**boundary)


# ---------------------------------------------------------------------------
# Endpoints -- Operational Boundary
# ---------------------------------------------------------------------------

@router.post(
    "/operational/{org_id}",
    response_model=OperationalBoundaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Set operational boundary",
    description=(
        "Set the operational boundary per ISO 14064-1 Clause 5.2.  "
        "Specify which of the 6 ISO categories are included with their "
        "significance assessment outcomes.  Categories 1 and 2 are mandatory; "
        "categories 3-6 require significance assessment."
    ),
)
async def set_operational_boundary(
    org_id: str,
    request: SetOperationalBoundaryRequest,
) -> OperationalBoundaryResponse:
    """Set the operational boundary for an organization."""
    boundary_id = _generate_id("opbnd")
    now = _now()
    categories = []
    for cat_req in request.categories:
        categories.append({
            "category": cat_req.category.value,
            "category_name": ISO_CATEGORY_NAMES.get(cat_req.category.value, ""),
            "included": cat_req.included,
            "significance": cat_req.significance.value,
            "justification": cat_req.justification,
        })
    included_count = sum(1 for c in categories if c["included"])
    excluded_count = len(categories) - included_count
    boundary = {
        "boundary_id": boundary_id,
        "org_id": org_id,
        "categories": categories,
        "included_count": included_count,
        "excluded_count": excluded_count,
        "created_at": now,
        "updated_at": now,
    }
    _op_boundaries[org_id] = boundary
    return OperationalBoundaryResponse(**boundary)


@router.get(
    "/operational/{org_id}",
    response_model=OperationalBoundaryResponse,
    summary="Get operational boundary",
    description="Retrieve the operational boundary and category inclusion settings.",
)
async def get_operational_boundary(org_id: str) -> OperationalBoundaryResponse:
    """Retrieve the operational boundary."""
    boundary = _op_boundaries.get(org_id)
    if not boundary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No operational boundary defined for organization {org_id}. Set one first via POST.",
        )
    return OperationalBoundaryResponse(**boundary)


@router.put(
    "/operational/{org_id}",
    response_model=OperationalBoundaryResponse,
    summary="Update operational boundary",
    description="Update category inclusion settings in the operational boundary.",
)
async def update_operational_boundary(
    org_id: str,
    request: SetOperationalBoundaryRequest,
) -> OperationalBoundaryResponse:
    """Update the operational boundary."""
    boundary = _op_boundaries.get(org_id)
    if not boundary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No operational boundary defined for organization {org_id}",
        )
    categories = []
    for cat_req in request.categories:
        categories.append({
            "category": cat_req.category.value,
            "category_name": ISO_CATEGORY_NAMES.get(cat_req.category.value, ""),
            "included": cat_req.included,
            "significance": cat_req.significance.value,
            "justification": cat_req.justification,
        })
    boundary["categories"] = categories
    boundary["included_count"] = sum(1 for c in categories if c["included"])
    boundary["excluded_count"] = len(categories) - boundary["included_count"]
    boundary["updated_at"] = _now()
    return OperationalBoundaryResponse(**boundary)
