"""
GL-GHG-APP Inventory Management API

Manages organizations, entity hierarchies, inventory boundaries,
and GHG inventories per the GHG Protocol Corporate Standard.

Consolidation approaches: Operational Control, Financial Control, Equity Share.
Scopes: 1 (Direct), 2 (Energy Indirect), 3 (Other Indirect).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/inventory", tags=["Inventory Management"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches for organizational boundary."""
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class EntityType(str, Enum):
    """Types of reporting entities in the organizational hierarchy."""
    CORPORATE = "corporate"
    DIVISION = "division"
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"
    FACILITY = "facility"
    SITE = "site"


class InventoryStatus(str, Enum):
    """Status of a GHG inventory."""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    PUBLISHED = "published"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateOrganizationRequest(BaseModel):
    """Request to create a new reporting organization."""
    name: str = Field(..., min_length=1, max_length=255, description="Organization legal name")
    industry: str = Field(..., min_length=1, max_length=100, description="Industry sector (e.g. Manufacturing, Energy)")
    country: str = Field(..., min_length=2, max_length=3, description="ISO 3166-1 alpha-2 or alpha-3 country code")
    description: Optional[str] = Field(None, max_length=1000, description="Organization description")
    fiscal_year_end_month: int = Field(12, ge=1, le=12, description="Fiscal year end month (1-12)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Acme Manufacturing Inc.",
                "industry": "Manufacturing",
                "country": "US",
                "description": "Global manufacturer of industrial widgets",
                "fiscal_year_end_month": 12
            }
        }


class CreateEntityRequest(BaseModel):
    """Request to add an entity to the organizational hierarchy."""
    name: str = Field(..., min_length=1, max_length=255, description="Entity name")
    type: EntityType = Field(..., description="Entity type in the hierarchy")
    parent_id: Optional[str] = Field(None, description="Parent entity ID (null for top-level)")
    ownership_pct: float = Field(100.0, ge=0, le=100, description="Equity ownership percentage")
    country: str = Field(..., min_length=2, max_length=3, description="ISO country code where entity operates")
    address: Optional[str] = Field(None, max_length=500, description="Physical address")
    naics_code: Optional[str] = Field(None, max_length=10, description="NAICS industry code")
    employees: Optional[int] = Field(None, ge=0, description="Number of employees at entity")
    operational_control: bool = Field(True, description="Whether organization has operational control")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Acme East Coast Plant",
                "type": "facility",
                "parent_id": "ent_abc123",
                "ownership_pct": 100.0,
                "country": "US",
                "address": "123 Industrial Blvd, Newark, NJ 07102",
                "naics_code": "332710",
                "employees": 350,
                "operational_control": True
            }
        }


class UpdateEntityRequest(BaseModel):
    """Request to update an existing entity."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    type: Optional[EntityType] = None
    parent_id: Optional[str] = None
    ownership_pct: Optional[float] = Field(None, ge=0, le=100)
    country: Optional[str] = Field(None, min_length=2, max_length=3)
    address: Optional[str] = Field(None, max_length=500)
    naics_code: Optional[str] = Field(None, max_length=10)
    employees: Optional[int] = Field(None, ge=0)
    operational_control: Optional[bool] = None
    active: Optional[bool] = None


class SetBoundaryRequest(BaseModel):
    """Request to set the GHG inventory boundary per GHG Protocol Ch. 3-4."""
    consolidation_approach: ConsolidationApproach = Field(
        ..., description="Organizational boundary approach"
    )
    scopes: List[int] = Field(
        ..., min_length=1, description="Scopes to report (1, 2, 3)"
    )
    base_year: int = Field(..., ge=1990, le=2100, description="Base year for tracking")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Current reporting year")
    scope3_categories: Optional[List[int]] = Field(
        None, description="Scope 3 categories included (1-15)"
    )
    significance_threshold_pct: float = Field(
        1.0, ge=0, le=100, description="Materiality threshold percentage for Scope 3"
    )
    exclusions: Optional[List[str]] = Field(
        None, description="List of excluded sources with justification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "consolidation_approach": "operational_control",
                "scopes": [1, 2, 3],
                "base_year": 2019,
                "reporting_year": 2025,
                "scope3_categories": [1, 2, 3, 4, 5, 6, 7, 11, 12],
                "significance_threshold_pct": 1.0,
                "exclusions": []
            }
        }


class CreateInventoryRequest(BaseModel):
    """Request to create a GHG inventory for a reporting year."""
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    notes: Optional[str] = Field(None, max_length=2000, description="Inventory notes")

    class Config:
        json_schema_extra = {
            "example": {
                "reporting_year": 2025,
                "notes": "Annual corporate GHG inventory per GHG Protocol"
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class EntityResponse(BaseModel):
    """An entity in the organizational hierarchy."""
    entity_id: str
    name: str
    type: EntityType
    parent_id: Optional[str]
    ownership_pct: float
    country: str
    address: Optional[str]
    naics_code: Optional[str]
    employees: Optional[int]
    operational_control: bool
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
    fiscal_year_end_month: int
    entity_count: int
    entities: Optional[List[EntityResponse]] = None
    created_at: datetime
    updated_at: datetime


class BoundaryResponse(BaseModel):
    """GHG inventory boundary configuration."""
    boundary_id: str
    org_id: str
    consolidation_approach: ConsolidationApproach
    scopes: List[int]
    base_year: int
    reporting_year: int
    scope3_categories: Optional[List[int]]
    significance_threshold_pct: float
    exclusions: List[str]
    entity_count_in_boundary: int
    created_at: datetime
    updated_at: datetime


class ScopeSummary(BaseModel):
    """Summary of emissions for a single scope."""
    scope: int
    total_tco2e: float
    percentage_of_total: float
    category_count: int
    data_quality_score: Optional[float] = None


class IntensityMetrics(BaseModel):
    """Emission intensity metrics."""
    per_revenue: Optional[float] = Field(None, description="tCO2e per million USD revenue")
    per_employee: Optional[float] = Field(None, description="tCO2e per employee")
    per_sqft: Optional[float] = Field(None, description="tCO2e per square foot")
    per_unit_produced: Optional[float] = Field(None, description="tCO2e per unit produced")


class DataQualitySummary(BaseModel):
    """Overall data quality assessment."""
    overall_score: float = Field(..., ge=0, le=100, description="0-100 quality score")
    grade: str = Field(..., description="A through F grade")
    completeness_pct: float
    accuracy_score: float
    timeliness_score: float
    consistency_score: float
    issues: List[str]


class InventoryResponse(BaseModel):
    """Full GHG inventory with all scopes and quality metrics."""
    inventory_id: str
    org_id: str
    reporting_year: int
    status: InventoryStatus
    total_tco2e: float
    scope_summaries: List[ScopeSummary]
    intensity_metrics: IntensityMetrics
    data_quality: DataQualitySummary
    base_year_emissions: Optional[float]
    base_year_change_pct: Optional[float]
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Simulated Data Store
# ---------------------------------------------------------------------------

_organizations: Dict[str, Dict[str, Any]] = {}
_entities: Dict[str, Dict[str, Any]] = {}
_boundaries: Dict[str, Dict[str, Any]] = {}
_inventories: Dict[str, Dict[str, Any]] = {}


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
            child_copy = dict(child)
            child_copy["children"] = _attach_children(child["entity_id"])
            result.append(child_copy)
        return result

    return _attach_children(None)


def _simulate_inventory_data(inventory_id: str, org_id: str, year: int) -> Dict[str, Any]:
    """Generate realistic simulated inventory data for demo purposes."""
    scope1_total = 12450.8
    scope2_total = 8320.5
    scope3_total = 45230.2
    grand_total = scope1_total + scope2_total + scope3_total

    return {
        "inventory_id": inventory_id,
        "org_id": org_id,
        "reporting_year": year,
        "status": InventoryStatus.DRAFT.value,
        "total_tco2e": round(grand_total, 2),
        "scope_summaries": [
            {
                "scope": 1,
                "total_tco2e": scope1_total,
                "percentage_of_total": round(scope1_total / grand_total * 100, 2),
                "category_count": 5,
                "data_quality_score": 88.5,
            },
            {
                "scope": 2,
                "total_tco2e": scope2_total,
                "percentage_of_total": round(scope2_total / grand_total * 100, 2),
                "category_count": 2,
                "data_quality_score": 91.2,
            },
            {
                "scope": 3,
                "total_tco2e": scope3_total,
                "percentage_of_total": round(scope3_total / grand_total * 100, 2),
                "category_count": 9,
                "data_quality_score": 72.1,
            },
        ],
        "intensity_metrics": {
            "per_revenue": 42.3,
            "per_employee": 18.7,
            "per_sqft": 0.032,
            "per_unit_produced": None,
        },
        "data_quality": {
            "overall_score": 82.4,
            "grade": "B",
            "completeness_pct": 89.5,
            "accuracy_score": 85.0,
            "timeliness_score": 78.2,
            "consistency_score": 76.9,
            "issues": [
                "Scope 3 Category 8 missing supplier data for 3 vendors",
                "Mobile combustion fuel records incomplete for Q4",
                "Refrigerant inventory not reconciled with maintenance logs",
            ],
        },
        "base_year_emissions": 70500.0,
        "base_year_change_pct": round((grand_total - 70500.0) / 70500.0 * 100, 2),
        "notes": None,
        "created_at": _now().isoformat(),
        "updated_at": _now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/organizations",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create reporting organization",
    description=(
        "Create a new reporting organization. This is the top-level entity "
        "for GHG Protocol corporate accounting. Each organization defines "
        "its own boundary, entity hierarchy, and inventories."
    ),
)
async def create_organization(request: CreateOrganizationRequest) -> OrganizationResponse:
    org_id = _generate_id("org")
    now = _now()
    org = {
        "org_id": org_id,
        "name": request.name,
        "industry": request.industry,
        "country": request.country,
        "description": request.description,
        "fiscal_year_end_month": request.fiscal_year_end_month,
        "entity_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    _organizations[org_id] = org
    return OrganizationResponse(
        **org,
        entities=[],
    )


@router.get(
    "/organizations/{org_id}",
    response_model=OrganizationResponse,
    summary="Get organization with entity hierarchy",
    description=(
        "Retrieve organization details including the full entity hierarchy "
        "tree. Entities are nested by parent-child relationships."
    ),
)
async def get_organization(
    org_id: str,
    include_entities: bool = Query(True, description="Include entity hierarchy tree"),
) -> OrganizationResponse:
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


@router.post(
    "/organizations/{org_id}/entities",
    response_model=EntityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add entity to organization",
    description=(
        "Add a subsidiary, division, joint venture, facility, or site to "
        "the organization's entity hierarchy. Entities form a tree used "
        "for consolidation under the chosen boundary approach."
    ),
)
async def create_entity(org_id: str, request: CreateEntityRequest) -> EntityResponse:
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
        "type": request.type.value,
        "parent_id": request.parent_id,
        "ownership_pct": request.ownership_pct,
        "country": request.country,
        "address": request.address,
        "naics_code": request.naics_code,
        "employees": request.employees,
        "operational_control": request.operational_control,
        "active": True,
        "created_at": now,
        "updated_at": now,
    }
    _entities[entity_id] = entity
    return EntityResponse(**{k: v for k, v in entity.items() if k != "org_id"}, children=[])


@router.put(
    "/entities/{entity_id}",
    response_model=EntityResponse,
    summary="Update entity",
    description="Update properties of an existing entity in the hierarchy.",
)
async def update_entity(entity_id: str, request: UpdateEntityRequest) -> EntityResponse:
    entity = _entities.get(entity_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found",
        )
    updates = request.model_dump(exclude_unset=True)
    if "parent_id" in updates and updates["parent_id"]:
        if updates["parent_id"] not in _entities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Parent entity {updates['parent_id']} not found",
            )
        if updates["parent_id"] == entity_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entity cannot be its own parent",
            )
    if "type" in updates:
        updates["type"] = updates["type"].value if isinstance(updates["type"], EntityType) else updates["type"]
    entity.update(updates)
    entity["updated_at"] = _now()
    return EntityResponse(**{k: v for k, v in entity.items() if k != "org_id"}, children=[])


@router.post(
    "/organizations/{org_id}/boundary",
    response_model=BoundaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Set inventory boundary",
    description=(
        "Set the organizational and operational boundary for GHG accounting "
        "per GHG Protocol Chapters 3 and 4. Defines the consolidation "
        "approach, included scopes, base year, and Scope 3 category selection."
    ),
)
async def set_boundary(org_id: str, request: SetBoundaryRequest) -> BoundaryResponse:
    if org_id not in _organizations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found",
        )
    for scope in request.scopes:
        if scope not in (1, 2, 3):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scope {scope}. Must be 1, 2, or 3.",
            )
    if request.scope3_categories:
        for cat in request.scope3_categories:
            if cat < 1 or cat > 15:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid Scope 3 category {cat}. Must be 1-15.",
                )
    if request.reporting_year < request.base_year:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reporting year cannot precede base year.",
        )
    boundary_id = _generate_id("bnd")
    now = _now()
    org_entities = [e for e in _entities.values() if e["org_id"] == org_id and e.get("active", True)]
    boundary = {
        "boundary_id": boundary_id,
        "org_id": org_id,
        "consolidation_approach": request.consolidation_approach.value,
        "scopes": request.scopes,
        "base_year": request.base_year,
        "reporting_year": request.reporting_year,
        "scope3_categories": request.scope3_categories or [],
        "significance_threshold_pct": request.significance_threshold_pct,
        "exclusions": request.exclusions or [],
        "entity_count_in_boundary": len(org_entities),
        "created_at": now,
        "updated_at": now,
    }
    _boundaries[org_id] = boundary
    return BoundaryResponse(**boundary)


@router.get(
    "/organizations/{org_id}/boundary",
    response_model=BoundaryResponse,
    summary="Get current boundary",
    description="Retrieve the current inventory boundary for an organization.",
)
async def get_boundary(org_id: str) -> BoundaryResponse:
    if org_id not in _organizations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found",
        )
    boundary = _boundaries.get(org_id)
    if not boundary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No boundary defined for organization {org_id}. Set one first via POST.",
        )
    return BoundaryResponse(**boundary)


@router.post(
    "/organizations/{org_id}/inventories",
    response_model=InventoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create GHG inventory",
    description=(
        "Create a new GHG inventory for the given reporting year. "
        "Initializes with zero emissions; populate via Scope 1/2/3 data endpoints. "
        "Simulated demo data is generated for illustration."
    ),
)
async def create_inventory(org_id: str, request: CreateInventoryRequest) -> InventoryResponse:
    if org_id not in _organizations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {org_id} not found",
        )
    existing = [
        inv for inv in _inventories.values()
        if inv["org_id"] == org_id and inv["reporting_year"] == request.reporting_year
    ]
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Inventory for year {request.reporting_year} already exists: {existing[0]['inventory_id']}",
        )
    inventory_id = _generate_id("inv")
    inventory_data = _simulate_inventory_data(inventory_id, org_id, request.reporting_year)
    if request.notes:
        inventory_data["notes"] = request.notes
    _inventories[inventory_id] = inventory_data
    return InventoryResponse(**inventory_data)


@router.get(
    "/inventories/{inventory_id}",
    response_model=InventoryResponse,
    summary="Get full GHG inventory",
    description=(
        "Retrieve the complete GHG inventory including all scope totals, "
        "intensity metrics, and data quality assessment."
    ),
)
async def get_inventory(inventory_id: str) -> InventoryResponse:
    inventory = _inventories.get(inventory_id)
    if not inventory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inventory {inventory_id} not found",
        )
    return InventoryResponse(**inventory)
