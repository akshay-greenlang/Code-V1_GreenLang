"""
GL-ISO14064-APP Organization & Entity API

CRUD operations for reporting organizations and their entity hierarchies.
Organizations are the top-level legal entities performing ISO 14064-1 GHG
accounting.  Entities (subsidiaries, facilities, operations, joint ventures)
form a hierarchy under each organization and are subject to the chosen
consolidation approach per ISO 14064-1 Clause 5.1.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
import uuid

router = APIRouter(prefix="/api/v1/iso14064/organizations", tags=["Organizations"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateOrganizationRequest(BaseModel):
    """Request to create a new reporting organization."""
    name: str = Field(..., min_length=1, max_length=500, description="Legal entity name")
    industry: str = Field(..., min_length=1, max_length=100, description="Industry sector")
    country: str = Field(..., min_length=2, max_length=3, description="ISO 3166-1 country code")
    description: Optional[str] = Field(None, max_length=2000, description="Organization description")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Acme Manufacturing GmbH",
                "industry": "manufacturing",
                "country": "DE",
                "description": "European industrial manufacturer",
            }
        }


class UpdateOrganizationRequest(BaseModel):
    """Request to update an existing organization."""
    name: Optional[str] = Field(None, min_length=1, max_length=500)
    industry: Optional[str] = Field(None, min_length=1, max_length=100)
    country: Optional[str] = Field(None, min_length=2, max_length=3)
    description: Optional[str] = Field(None, max_length=2000)


class AddEntityRequest(BaseModel):
    """Request to add an entity to an organization."""
    name: str = Field(..., min_length=1, max_length=255, description="Entity name")
    entity_type: str = Field(..., description="subsidiary, facility, operation, joint_venture")
    parent_id: Optional[str] = Field(None, description="Parent entity ID for hierarchy")
    ownership_pct: float = Field(100.0, ge=0, le=100, description="Equity share percentage")
    country: str = Field(..., min_length=2, max_length=3, description="ISO country code")
    employees: Optional[int] = Field(None, ge=0, description="Full-time equivalents")
    revenue: Optional[float] = Field(None, ge=0, description="Annual revenue (USD)")
    floor_area_m2: Optional[float] = Field(None, ge=0, description="Floor area in m2")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Berlin Production Facility",
                "entity_type": "facility",
                "parent_id": None,
                "ownership_pct": 100.0,
                "country": "DE",
                "employees": 450,
                "revenue": 85000000.0,
                "floor_area_m2": 12000.0,
            }
        }


class UpdateEntityRequest(BaseModel):
    """Request to update an existing entity."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    ownership_pct: Optional[float] = Field(None, ge=0, le=100)
    employees: Optional[int] = Field(None, ge=0)
    revenue: Optional[float] = Field(None, ge=0)
    floor_area_m2: Optional[float] = Field(None, ge=0)
    active: Optional[bool] = None


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class EntityResponse(BaseModel):
    """An entity in the organizational hierarchy."""
    entity_id: str
    name: str
    entity_type: str
    parent_id: Optional[str]
    ownership_pct: float
    country: str
    employees: Optional[int]
    revenue: Optional[float]
    floor_area_m2: Optional[float]
    active: bool
    children: Optional[List["EntityResponse"]] = None
    created_at: datetime
    updated_at: datetime


class OrganizationResponse(BaseModel):
    """Organization with entity hierarchy."""
    org_id: str
    name: str
    industry: str
    country: str
    description: Optional[str]
    entity_count: int
    entities: Optional[List[EntityResponse]] = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_organizations: Dict[str, Dict[str, Any]] = {}
_entities: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _build_entity_tree(org_id: str) -> List[Dict[str, Any]]:
    """Build hierarchical entity tree for an organization."""
    org_entities = [e for e in _entities.values() if e["org_id"] == org_id]
    entity_map: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for ent in org_entities:
        parent = ent.get("parent_id")
        entity_map.setdefault(parent, []).append(ent)

    def _attach_children(parent_id: Optional[str]) -> List[Dict[str, Any]]:
        children = entity_map.get(parent_id, [])
        result = []
        for child in children:
            child_copy = {k: v for k, v in child.items() if k != "org_id"}
            child_copy["children"] = _attach_children(child["entity_id"])
            result.append(child_copy)
        return result

    return _attach_children(None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create reporting organization",
    description=(
        "Create a new reporting organization for ISO 14064-1 GHG accounting. "
        "This is the top-level entity owning the entity hierarchy, boundaries, "
        "and inventories."
    ),
)
async def create_organization(request: CreateOrganizationRequest) -> OrganizationResponse:
    """Create a new organization for ISO 14064-1 compliance."""
    org_id = _generate_id("org")
    now = _now()
    org = {
        "org_id": org_id,
        "name": request.name,
        "industry": request.industry,
        "country": request.country,
        "description": request.description,
        "entity_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    _organizations[org_id] = org
    return OrganizationResponse(**org, entities=[])


@router.get(
    "/{org_id}",
    response_model=OrganizationResponse,
    summary="Get organization with entity hierarchy",
    description=(
        "Retrieve organization details including the full entity hierarchy "
        "tree.  Entities are nested by parent-child relationships."
    ),
)
async def get_organization(
    org_id: str,
    include_entities: bool = Query(True, description="Include entity hierarchy tree"),
) -> OrganizationResponse:
    """Retrieve an organization by ID."""
    org = _organizations.get(org_id)
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found",
        )
    entities = _build_entity_tree(org_id) if include_entities else None
    org_entities = [e for e in _entities.values() if e["org_id"] == org_id]
    return OrganizationResponse(
        **{**org, "entity_count": len(org_entities)},
        entities=entities,
    )


@router.put(
    "/{org_id}",
    response_model=OrganizationResponse,
    summary="Update organization",
    description="Update properties of an existing organization.",
)
async def update_organization(
    org_id: str,
    request: UpdateOrganizationRequest,
) -> OrganizationResponse:
    """Update an existing organization."""
    org = _organizations.get(org_id)
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found",
        )
    updates = request.model_dump(exclude_unset=True)
    org.update(updates)
    org["updated_at"] = _now()
    org_entities = [e for e in _entities.values() if e["org_id"] == org_id]
    return OrganizationResponse(
        **{**org, "entity_count": len(org_entities)},
        entities=None,
    )


@router.post(
    "/{org_id}/entities",
    response_model=EntityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add entity to organization",
    description=(
        "Add a subsidiary, facility, operation, or joint venture to the "
        "organization's entity hierarchy.  Entities form a tree used for "
        "consolidation under the chosen boundary approach per ISO 14064-1 Clause 5.1."
    ),
)
async def create_entity(org_id: str, request: AddEntityRequest) -> EntityResponse:
    """Add an entity to the organizational hierarchy."""
    if org_id not in _organizations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found",
        )
    if request.parent_id and request.parent_id not in _entities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Parent entity {request.parent_id} not found",
        )
    entity_id = _generate_id("ent")
    now = _now()
    entity = {
        "entity_id": entity_id,
        "org_id": org_id,
        "name": request.name,
        "entity_type": request.entity_type,
        "parent_id": request.parent_id,
        "ownership_pct": request.ownership_pct,
        "country": request.country,
        "employees": request.employees,
        "revenue": request.revenue,
        "floor_area_m2": request.floor_area_m2,
        "active": True,
        "created_at": now,
        "updated_at": now,
    }
    _entities[entity_id] = entity
    return EntityResponse(**{k: v for k, v in entity.items() if k != "org_id"}, children=[])


@router.get(
    "/{org_id}/entities",
    response_model=List[EntityResponse],
    summary="List entities for organization",
    description="Retrieve all entities belonging to an organization.",
)
async def list_entities(
    org_id: str,
    active_only: bool = Query(True, description="Return only active entities"),
) -> List[EntityResponse]:
    """List all entities for an organization."""
    if org_id not in _organizations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found",
        )
    org_entities = [e for e in _entities.values() if e["org_id"] == org_id]
    if active_only:
        org_entities = [e for e in org_entities if e.get("active", True)]
    return [
        EntityResponse(**{k: v for k, v in e.items() if k != "org_id"}, children=[])
        for e in org_entities
    ]


@router.put(
    "/{org_id}/entities/{entity_id}",
    response_model=EntityResponse,
    summary="Update entity",
    description="Update properties of an existing entity in the hierarchy.",
)
async def update_entity(
    org_id: str,
    entity_id: str,
    request: UpdateEntityRequest,
) -> EntityResponse:
    """Update an existing entity."""
    if org_id not in _organizations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found",
        )
    entity = _entities.get(entity_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found",
        )
    if entity["org_id"] != org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Entity {entity_id} does not belong to organization {org_id}",
        )
    updates = request.model_dump(exclude_unset=True)
    entity.update(updates)
    entity["updated_at"] = _now()
    return EntityResponse(**{k: v for k, v in entity.items() if k != "org_id"}, children=[])
