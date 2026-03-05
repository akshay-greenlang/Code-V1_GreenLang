"""
GL-CDP-APP Settings API

Manages organization-level settings for the CDP disclosure platform
including organization profile, team member management, MRV integration
configuration, notification preferences, and questionnaire defaults.

Settings areas:
    - Organization profile (name, sector, region, GICS code)
    - Team management (add/remove members, roles, assignments)
    - MRV integrations (connect to GreenLang MRV agents)
    - Notification preferences (deadlines, reviews, scoring)
    - Questionnaire defaults (year, language, sector routing)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/settings", tags=["Settings"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TeamRole(str, Enum):
    """Team member roles in CDP disclosure."""
    ADMIN = "admin"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"
    APPROVER = "approver"
    SUBMITTER = "submitter"


class IntegrationStatus(str, Enum):
    """MRV integration connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class OrgSettingsResponse(BaseModel):
    """Organization settings."""
    org_id: str
    org_name: str
    sector: str
    sector_name: str
    gics_code: Optional[str]
    region: str
    country: str
    reporting_year: int
    questionnaire_year: str
    default_language: str
    currency: str
    fiscal_year_end: str
    base_year: int
    consolidation_approach: str
    auto_populate_enabled: bool
    auto_save_enabled: bool
    notification_enabled: bool
    updated_at: datetime


class UpdateOrgSettingsRequest(BaseModel):
    """Request to update organization settings."""
    org_name: Optional[str] = Field(None, max_length=300)
    sector: Optional[str] = None
    gics_code: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    reporting_year: Optional[int] = Field(None, ge=2020, le=2100)
    questionnaire_year: Optional[str] = None
    default_language: Optional[str] = None
    currency: Optional[str] = None
    fiscal_year_end: Optional[str] = None
    base_year: Optional[int] = Field(None, ge=2000, le=2100)
    consolidation_approach: Optional[str] = None
    auto_populate_enabled: Optional[bool] = None
    auto_save_enabled: Optional[bool] = None
    notification_enabled: Optional[bool] = None

    class Config:
        json_schema_extra = {
            "example": {
                "sector": "materials",
                "reporting_year": 2025,
                "auto_populate_enabled": True,
            }
        }


class TeamMemberResponse(BaseModel):
    """Team member record."""
    member_id: str
    org_id: str
    user_id: str
    email: str
    name: str
    role: str
    assigned_modules: List[str]
    assigned_question_count: int
    responses_completed: int
    reviews_completed: int
    is_active: bool
    last_active_at: Optional[datetime]
    added_at: datetime


class AddTeamMemberRequest(BaseModel):
    """Request to add a team member."""
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="Team member email")
    name: str = Field(..., min_length=1, max_length=200, description="Display name")
    role: TeamRole = Field(TeamRole.EDITOR, description="Team role")
    assigned_modules: Optional[List[str]] = Field(
        None, description="Modules assigned to this member"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "usr_abc123",
                "email": "j.smith@example.com",
                "name": "Jane Smith",
                "role": "editor",
                "assigned_modules": ["M1", "M7", "M13"],
            }
        }


class IntegrationConfig(BaseModel):
    """MRV integration configuration."""
    integration_id: str
    integration_name: str
    integration_type: str
    status: str
    mrv_agent_ids: List[str]
    description: str
    auto_populate_modules: List[str]
    last_sync_at: Optional[datetime]
    data_freshness_days: Optional[int]
    is_enabled: bool


class UpdateIntegrationsRequest(BaseModel):
    """Request to configure MRV integrations."""
    scope_1_enabled: bool = Field(True, description="Enable Scope 1 MRV integration")
    scope_2_enabled: bool = Field(True, description="Enable Scope 2 MRV integration")
    scope_3_enabled: bool = Field(True, description="Enable Scope 3 MRV integration")
    auto_populate_m7: bool = Field(True, description="Auto-populate M7 Environmental Performance")
    auto_populate_m11: bool = Field(True, description="Auto-populate M11 Additional Metrics")
    sync_frequency: str = Field("daily", description="Sync frequency: manual, daily, weekly")
    data_freshness_threshold_days: int = Field(
        30, ge=1, le=365, description="Alert if data older than N days"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "scope_1_enabled": True,
                "scope_2_enabled": True,
                "scope_3_enabled": True,
                "auto_populate_m7": True,
                "auto_populate_m11": True,
                "sync_frequency": "daily",
                "data_freshness_threshold_days": 30,
            }
        }


class IntegrationUpdateResponse(BaseModel):
    """Response after updating integrations."""
    integrations: List[IntegrationConfig]
    sync_frequency: str
    data_freshness_threshold_days: int
    updated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_org_settings: Dict[str, Dict[str, Any]] = {}
_team_members: Dict[str, Dict[str, Any]] = {}

DEFAULT_SETTINGS = {
    "org_name": "GreenLang Demo Organization",
    "sector": "materials",
    "sector_name": "Materials",
    "gics_code": "1510",
    "region": "europe",
    "country": "GB",
    "reporting_year": 2025,
    "questionnaire_year": "2026",
    "default_language": "en",
    "currency": "USD",
    "fiscal_year_end": "12-31",
    "base_year": 2019,
    "consolidation_approach": "operational_control",
    "auto_populate_enabled": True,
    "auto_save_enabled": True,
    "notification_enabled": True,
    "updated_at": datetime.utcnow(),
}

# Seed team members
_seed_members = [
    {"uid": "usr_001", "email": "j.smith@example.com", "name": "Jane Smith", "role": "admin", "modules": ["M0", "M13"], "responses": 15, "reviews": 8},
    {"uid": "usr_002", "email": "j.doe@example.com", "name": "John Doe", "role": "editor", "modules": ["M1", "M2", "M5"], "responses": 22, "reviews": 0},
    {"uid": "usr_003", "email": "a.chen@example.com", "name": "Dr. Alice Chen", "role": "reviewer", "modules": ["M7", "M10"], "responses": 5, "reviews": 35},
    {"uid": "usr_004", "email": "m.garcia@example.com", "name": "Maria Garcia", "role": "editor", "modules": ["M3", "M4", "M6"], "responses": 18, "reviews": 0},
    {"uid": "usr_005", "email": "ceo@example.com", "name": "Robert Johnson, CEO", "role": "approver", "modules": ["M13"], "responses": 0, "reviews": 12},
]

for sm in _seed_members:
    mid = f"mbr_{sm['uid']}"
    _team_members[mid] = {
        "member_id": mid, "org_id": "default", "user_id": sm["uid"],
        "email": sm["email"], "name": sm["name"], "role": sm["role"],
        "assigned_modules": sm["modules"], "assigned_question_count": len(sm["modules"]) * 15,
        "responses_completed": sm["responses"], "reviews_completed": sm["reviews"],
        "is_active": True, "last_active_at": datetime.utcnow(),
        "added_at": datetime(2025, 1, 10),
    }


MRV_INTEGRATIONS = [
    {
        "id": "int_scope1", "name": "Scope 1 MRV Agents", "type": "scope_1",
        "agents": ["MRV-001", "MRV-002", "MRV-003", "MRV-004", "MRV-005"],
        "desc": "Stationary, refrigerants, mobile, process, fugitive emissions",
        "modules": ["M7"],
    },
    {
        "id": "int_scope1_lu", "name": "Scope 1 Land Use Agents", "type": "scope_1_land_use",
        "agents": ["MRV-006", "MRV-007", "MRV-008"],
        "desc": "Land use, waste treatment, agricultural emissions",
        "modules": ["M7"],
    },
    {
        "id": "int_scope2", "name": "Scope 2 MRV Agents", "type": "scope_2",
        "agents": ["MRV-009", "MRV-010", "MRV-011", "MRV-012", "MRV-013"],
        "desc": "Location-based, market-based, steam/heat, cooling, dual reporting",
        "modules": ["M7", "M11"],
    },
    {
        "id": "int_scope3_up", "name": "Scope 3 Upstream Agents", "type": "scope_3_upstream",
        "agents": ["MRV-014", "MRV-015", "MRV-016", "MRV-017", "MRV-018", "MRV-019", "MRV-020", "MRV-021"],
        "desc": "Categories 1-8 upstream Scope 3",
        "modules": ["M7"],
    },
    {
        "id": "int_scope3_down", "name": "Scope 3 Downstream Agents", "type": "scope_3_downstream",
        "agents": ["MRV-022", "MRV-023", "MRV-024", "MRV-025", "MRV-026", "MRV-027", "MRV-028"],
        "desc": "Categories 9-15 downstream Scope 3",
        "modules": ["M7"],
    },
    {
        "id": "int_crosscut", "name": "Cross-Cutting Agents", "type": "cross_cutting",
        "agents": ["MRV-029", "MRV-030"],
        "desc": "Category mapper, audit trail & lineage",
        "modules": ["M7"],
    },
]


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints -- Organization Settings
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}",
    response_model=OrgSettingsResponse,
    summary="Get organization settings",
    description="Retrieve all organization-level settings for the CDP platform.",
)
async def get_settings(org_id: str) -> OrgSettingsResponse:
    """Get organization settings."""
    settings = _org_settings.get(org_id, {**DEFAULT_SETTINGS, "org_id": org_id})
    if "org_id" not in settings:
        settings["org_id"] = org_id
    return OrgSettingsResponse(**settings)


@router.put(
    "/{org_id}",
    response_model=OrgSettingsResponse,
    summary="Update organization settings",
    description="Update organization-level settings.",
)
async def update_settings(
    org_id: str,
    request: UpdateOrgSettingsRequest,
) -> OrgSettingsResponse:
    """Update organization settings."""
    settings = _org_settings.get(org_id, {**DEFAULT_SETTINGS, "org_id": org_id})
    updates = request.model_dump(exclude_unset=True)

    # Map sector to sector_name
    sector_names = {
        "energy": "Energy", "materials": "Materials", "industrials": "Industrials",
        "consumer_discretionary": "Consumer Discretionary", "consumer_staples": "Consumer Staples",
        "health_care": "Health Care", "financials": "Financials",
        "information_technology": "Information Technology",
        "communication_services": "Communication Services",
        "utilities": "Utilities", "real_estate": "Real Estate",
    }
    if "sector" in updates:
        settings["sector_name"] = sector_names.get(updates["sector"], updates["sector"])

    settings.update(updates)
    settings["updated_at"] = _now()
    settings["org_id"] = org_id
    _org_settings[org_id] = settings
    return OrgSettingsResponse(**settings)


# ---------------------------------------------------------------------------
# Endpoints -- Team Management
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}/team",
    response_model=List[TeamMemberResponse],
    summary="List team members",
    description="Retrieve all team members for the CDP disclosure.",
)
async def list_team(
    org_id: str,
    role: Optional[str] = Query(None, description="Filter by team role"),
    active_only: bool = Query(True, description="Only return active members"),
) -> List[TeamMemberResponse]:
    """List team members."""
    members = [m for m in _team_members.values()]
    if role:
        members = [m for m in members if m["role"] == role]
    if active_only:
        members = [m for m in members if m["is_active"]]
    members.sort(key=lambda m: m["name"])
    return [TeamMemberResponse(**m) for m in members]


@router.post(
    "/{org_id}/team",
    response_model=TeamMemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add team member",
    description="Add a team member to the CDP disclosure team.",
)
async def add_team_member(
    org_id: str,
    request: AddTeamMemberRequest,
) -> TeamMemberResponse:
    """Add a team member."""
    existing = [
        m for m in _team_members.values()
        if m["email"] == request.email
    ]
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Team member with email {request.email} already exists.",
        )

    member_id = _generate_id("mbr")
    now = _now()
    member = {
        "member_id": member_id,
        "org_id": org_id,
        "user_id": request.user_id,
        "email": request.email,
        "name": request.name,
        "role": request.role.value,
        "assigned_modules": request.assigned_modules or [],
        "assigned_question_count": len(request.assigned_modules or []) * 15,
        "responses_completed": 0,
        "reviews_completed": 0,
        "is_active": True,
        "last_active_at": None,
        "added_at": now,
    }
    _team_members[member_id] = member
    return TeamMemberResponse(**member)


# ---------------------------------------------------------------------------
# Endpoints -- MRV Integrations
# ---------------------------------------------------------------------------

@router.put(
    "/{org_id}/integrations",
    response_model=IntegrationUpdateResponse,
    summary="Configure MRV integrations",
    description=(
        "Configure integration with GreenLang MRV agents for auto-population "
        "of emissions data. Controls which scopes are connected, auto-populate "
        "targets, sync frequency, and data freshness thresholds."
    ),
)
async def update_integrations(
    org_id: str,
    request: UpdateIntegrationsRequest,
) -> IntegrationUpdateResponse:
    """Configure MRV integrations."""
    now = _now()

    integrations = []
    for integ in MRV_INTEGRATIONS:
        # Determine if enabled based on scope
        enabled = False
        if integ["type"] == "scope_1" and request.scope_1_enabled:
            enabled = True
        elif integ["type"] == "scope_1_land_use" and request.scope_1_enabled:
            enabled = True
        elif integ["type"] == "scope_2" and request.scope_2_enabled:
            enabled = True
        elif integ["type"] == "scope_3_upstream" and request.scope_3_enabled:
            enabled = True
        elif integ["type"] == "scope_3_downstream" and request.scope_3_enabled:
            enabled = True
        elif integ["type"] == "cross_cutting":
            enabled = request.scope_1_enabled or request.scope_2_enabled or request.scope_3_enabled

        integrations.append(IntegrationConfig(
            integration_id=integ["id"],
            integration_name=integ["name"],
            integration_type=integ["type"],
            status=IntegrationStatus.CONNECTED.value if enabled else IntegrationStatus.DISCONNECTED.value,
            mrv_agent_ids=integ["agents"],
            description=integ["desc"],
            auto_populate_modules=integ["modules"] if enabled else [],
            last_sync_at=now if enabled else None,
            data_freshness_days=request.data_freshness_threshold_days if enabled else None,
            is_enabled=enabled,
        ))

    return IntegrationUpdateResponse(
        integrations=integrations,
        sync_frequency=request.sync_frequency,
        data_freshness_threshold_days=request.data_freshness_threshold_days,
        updated_at=now,
    )
