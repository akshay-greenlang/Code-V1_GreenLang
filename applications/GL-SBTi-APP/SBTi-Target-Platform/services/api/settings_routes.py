"""
GL-SBTi-APP Settings API

Manages organization-level configuration for the SBTi Target Setting &
Validation Platform.  Provides endpoints for retrieving and updating
organization settings, sector classification, framework preferences,
MRV agent connection settings, and notification preferences.  Also
exposes reference-data catalogues for SBTi criteria definitions, sectors,
and FLAG commodities.

Settings Categories:
    - Organization: Name, industry, contact, base year
    - Sector: SBTi sector classification, sub-sector, methodology
    - Frameworks: Framework preferences and alignment settings
    - MRV Connection: Integration with GreenLang MRV agents
    - Notifications: Alert and reporting preferences
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/sbti/settings", tags=["Settings"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    CNY = "CNY"
    SGD = "SGD"


class EmissionUnit(str, Enum):
    TCO2E = "tCO2e"
    KTCO2E = "ktCO2e"
    MTCO2E = "MtCO2e"


class FiscalYearEnd(str, Enum):
    MARCH = "march"
    JUNE = "june"
    SEPTEMBER = "september"
    DECEMBER = "december"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class UpdateSettingsRequest(BaseModel):
    """Update organization settings."""
    organization_name: Optional[str] = Field(None, max_length=300)
    industry: Optional[str] = Field(None, max_length=200)
    contact_email: Optional[str] = Field(None, max_length=300)
    fiscal_year_end: Optional[FiscalYearEnd] = None
    reporting_currency: Optional[Currency] = None
    emission_unit: Optional[EmissionUnit] = None
    base_year: Optional[int] = Field(None, ge=2015, le=2025)
    sbti_commitment_date: Optional[str] = Field(None, description="ISO date of SBTi commitment")
    sbti_validation_date: Optional[str] = Field(None, description="ISO date of SBTi validation")
    sbti_status: Optional[str] = Field(None, description="committed, submitted, validated, none")


class UpdateSectorRequest(BaseModel):
    """Update sector classification."""
    sector_id: str = Field(..., description="SBTi sector identifier")
    sub_sector: Optional[str] = Field(None, max_length=200)
    isic_code: Optional[str] = Field(None, max_length=10)
    nace_code: Optional[str] = Field(None, max_length=10)
    naics_code: Optional[str] = Field(None, max_length=10)
    methodology_preference: Optional[str] = Field(None, description="aca or sda")


class UpdateFrameworksRequest(BaseModel):
    """Update framework preferences."""
    primary_framework: str = Field("sbti", description="Primary reporting framework")
    secondary_frameworks: List[str] = Field(
        default_factory=list, description="Additional frameworks",
    )
    enable_csrd_alignment: bool = Field(True)
    enable_cdp_alignment: bool = Field(True)
    enable_tcfd_alignment: bool = Field(True)
    enable_iso14064_alignment: bool = Field(False)
    enable_sec_alignment: bool = Field(False)


class UpdateMRVRequest(BaseModel):
    """Update MRV agent connection settings."""
    mrv_enabled: bool = Field(True)
    auto_sync_emissions: bool = Field(True)
    sync_frequency: str = Field("daily", description="hourly, daily, weekly, monthly")
    scope1_agents: List[str] = Field(
        default_factory=list, description="Connected Scope 1 MRV agents",
    )
    scope2_agents: List[str] = Field(
        default_factory=list, description="Connected Scope 2 MRV agents",
    )
    scope3_agents: List[str] = Field(
        default_factory=list, description="Connected Scope 3 MRV agents",
    )
    data_quality_threshold: float = Field(3.0, ge=1, le=5)


class UpdateNotificationsRequest(BaseModel):
    """Update notification preferences."""
    email_notifications: bool = Field(True)
    notification_email: Optional[str] = Field(None)
    progress_alerts: bool = Field(True)
    deadline_reminders: bool = Field(True)
    reminder_days_before: int = Field(30, ge=1, le=365)
    off_track_alerts: bool = Field(True)
    recalculation_alerts: bool = Field(True)
    review_reminders: bool = Field(True)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SettingsResponse(BaseModel):
    """Organization settings."""
    org_id: str
    organization_name: str
    industry: str
    contact_email: Optional[str]
    fiscal_year_end: str
    reporting_currency: str
    emission_unit: str
    base_year: int
    sbti_commitment_date: Optional[str]
    sbti_validation_date: Optional[str]
    sbti_status: str
    updated_at: datetime


class SectorSettingsResponse(BaseModel):
    """Sector classification settings."""
    org_id: str
    sector_id: str
    sector_name: str
    sub_sector: Optional[str]
    isic_code: Optional[str]
    nace_code: Optional[str]
    naics_code: Optional[str]
    methodology_preference: str
    sda_available: bool
    intensity_unit: str
    updated_at: datetime


class FrameworkSettingsResponse(BaseModel):
    """Framework preferences."""
    org_id: str
    primary_framework: str
    secondary_frameworks: List[str]
    enable_csrd_alignment: bool
    enable_cdp_alignment: bool
    enable_tcfd_alignment: bool
    enable_iso14064_alignment: bool
    enable_sec_alignment: bool
    updated_at: datetime


class MRVSettingsResponse(BaseModel):
    """MRV agent connection settings."""
    org_id: str
    mrv_enabled: bool
    auto_sync_emissions: bool
    sync_frequency: str
    scope1_agents: List[str]
    scope2_agents: List[str]
    scope3_agents: List[str]
    data_quality_threshold: float
    last_sync_at: Optional[datetime]
    updated_at: datetime


class NotificationSettingsResponse(BaseModel):
    """Notification preferences."""
    org_id: str
    email_notifications: bool
    notification_email: Optional[str]
    progress_alerts: bool
    deadline_reminders: bool
    reminder_days_before: int
    off_track_alerts: bool
    recalculation_alerts: bool
    review_reminders: bool
    updated_at: datetime


class CriterionDefinitionResponse(BaseModel):
    """SBTi criterion definition."""
    criterion_id: str
    name: str
    category: str
    description: str
    applies_to: List[str]
    threshold: Optional[str]
    required: bool
    version: str


class SectorListItem(BaseModel):
    """Sector list item."""
    sector_id: str
    name: str
    sda_available: bool
    intensity_unit: str
    methodology: str
    sub_sectors: List[str]


class FLAGCommodityItem(BaseModel):
    """FLAG commodity list item."""
    commodity_id: str
    name: str
    category: str
    pathway_reduction_rate_pct: float
    deforestation_relevant: bool


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

CRITERIA = [
    {"criterion_id": "C1", "name": "Target Boundary", "category": "boundary", "description": "Target boundary must cover all relevant GHG emissions sources.", "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": True, "version": "v2.1"},
    {"criterion_id": "C2", "name": "Scope 1+2 Coverage (95%)", "category": "boundary", "description": "Scope 1+2 targets must cover at least 95% of company-wide emissions.", "applies_to": ["near_term", "long_term", "net_zero"], "threshold": "95%", "required": True, "version": "v2.1"},
    {"criterion_id": "C3", "name": "Base Year >= 2015", "category": "timeframe", "description": "Base year must be 2015 or more recent.", "applies_to": ["near_term", "long_term", "net_zero"], "threshold": ">=2015", "required": True, "version": "v2.1"},
    {"criterion_id": "C4", "name": "Near-Term 5-10 Years", "category": "timeframe", "description": "Near-term targets must have 5-10 year timeframe.", "applies_to": ["near_term"], "threshold": "5-10 years", "required": True, "version": "v2.1"},
    {"criterion_id": "C5", "name": "Long-Term by 2050", "category": "timeframe", "description": "Long-term targets must be by 2050 or sooner.", "applies_to": ["long_term", "net_zero"], "threshold": "<=2050", "required": True, "version": "v2.1"},
    {"criterion_id": "C6", "name": "1.5C Ambition (S1+2)", "category": "ambition", "description": "Scope 1+2 near-term targets must align with 1.5C (4.2% p.a.).", "applies_to": ["near_term"], "threshold": ">=4.2%/yr", "required": True, "version": "v2.1"},
    {"criterion_id": "C7", "name": "WB2C Ambition (S3)", "category": "ambition", "description": "Scope 3 targets must align with well-below 2C (2.5% p.a.).", "applies_to": ["near_term"], "threshold": ">=2.5%/yr", "required": True, "version": "v2.1"},
    {"criterion_id": "C8", "name": "Long-Term 90% Reduction", "category": "ambition", "description": "Long-term targets must reduce at least 90% of Scope 1+2.", "applies_to": ["long_term", "net_zero"], "threshold": ">=90%", "required": True, "version": "v2.1"},
    {"criterion_id": "C13", "name": "Scope 3 Trigger (40%)", "category": "scope3", "description": "Scope 3 target required if S3 >= 40% of total.", "applies_to": ["near_term"], "threshold": "40%", "required": True, "version": "v2.1"},
    {"criterion_id": "C14", "name": "Scope 3 Coverage (67%)", "category": "scope3", "description": "Scope 3 target must cover >= 67% of Scope 3 emissions.", "applies_to": ["near_term"], "threshold": "67%", "required": True, "version": "v2.1"},
    {"criterion_id": "C19", "name": "Recalculation (5%)", "category": "reporting", "description": "Significant changes (>5%) require base year recalculation.", "applies_to": ["near_term", "long_term", "net_zero"], "threshold": "5%", "required": True, "version": "v2.1"},
    {"criterion_id": "C20", "name": "Net-Zero Companion Target", "category": "net_zero", "description": "Net-zero requires a validated companion near-term target.", "applies_to": ["net_zero"], "threshold": None, "required": True, "version": "v2.1"},
    {"criterion_id": "C21", "name": "Residual Emissions <=10%", "category": "net_zero", "description": "Residual emissions at net-zero year must be <=10%.", "applies_to": ["net_zero"], "threshold": "<=10%", "required": True, "version": "v2.1"},
    {"criterion_id": "C24", "name": "FLAG Trigger (20%)", "category": "flag", "description": "FLAG target required if FLAG >= 20% of total emissions.", "applies_to": ["near_term"], "threshold": "20%", "required": True, "version": "v2.1"},
    {"criterion_id": "C25", "name": "Zero Deforestation 2025", "category": "flag", "description": "FLAG target setters must commit to zero deforestation by 2025.", "applies_to": ["near_term"], "threshold": None, "required": True, "version": "v2.1"},
]

SECTORS = [
    {"sector_id": "power_generation", "name": "Power Generation", "sda_available": True, "intensity_unit": "tCO2e/MWh", "methodology": "SDA", "sub_sectors": ["thermal", "renewable", "nuclear"]},
    {"sector_id": "transport_passenger", "name": "Passenger Transport", "sda_available": True, "intensity_unit": "gCO2e/pkm", "methodology": "SDA", "sub_sectors": ["road", "rail", "air", "maritime"]},
    {"sector_id": "transport_freight", "name": "Freight Transport", "sda_available": True, "intensity_unit": "gCO2e/tkm", "methodology": "SDA", "sub_sectors": ["road", "rail", "air", "maritime"]},
    {"sector_id": "buildings_commercial", "name": "Commercial Buildings", "sda_available": True, "intensity_unit": "kgCO2e/m2", "methodology": "SDA", "sub_sectors": ["office", "retail", "hospitality"]},
    {"sector_id": "buildings_residential", "name": "Residential Buildings", "sda_available": True, "intensity_unit": "kgCO2e/m2", "methodology": "SDA", "sub_sectors": ["single_family", "multi_family"]},
    {"sector_id": "cement", "name": "Cement", "sda_available": True, "intensity_unit": "tCO2e/tonne_clinker", "methodology": "SDA", "sub_sectors": ["portland", "blended"]},
    {"sector_id": "iron_steel", "name": "Iron & Steel", "sda_available": True, "intensity_unit": "tCO2e/tonne_steel", "methodology": "SDA", "sub_sectors": ["blast_furnace", "electric_arc"]},
    {"sector_id": "aluminium", "name": "Aluminium", "sda_available": True, "intensity_unit": "tCO2e/tonne_aluminium", "methodology": "SDA", "sub_sectors": ["smelting", "rolling"]},
    {"sector_id": "pulp_paper", "name": "Pulp & Paper", "sda_available": True, "intensity_unit": "tCO2e/tonne_product", "methodology": "SDA", "sub_sectors": ["pulp", "paper", "packaging"]},
    {"sector_id": "chemicals", "name": "Chemicals", "sda_available": False, "intensity_unit": "tCO2e/tonne_product", "methodology": "ACA", "sub_sectors": ["basic", "specialty", "pharma"]},
    {"sector_id": "oil_gas", "name": "Oil & Gas", "sda_available": False, "intensity_unit": "tCO2e/TJ", "methodology": "ACA", "sub_sectors": ["upstream", "midstream", "downstream"]},
    {"sector_id": "aviation", "name": "Aviation", "sda_available": True, "intensity_unit": "gCO2e/RTK", "methodology": "SDA", "sub_sectors": ["passenger", "cargo"]},
    {"sector_id": "shipping", "name": "Shipping", "sda_available": True, "intensity_unit": "gCO2e/tnm", "methodology": "SDA", "sub_sectors": ["container", "bulk", "tanker"]},
    {"sector_id": "general", "name": "All Other Sectors", "sda_available": False, "intensity_unit": "tCO2e", "methodology": "ACA", "sub_sectors": ["services", "technology", "healthcare", "retail", "finance"]},
]

FLAG_COMMODITIES = [
    {"commodity_id": "cattle_beef", "name": "Cattle (Beef)", "category": "livestock", "pathway_reduction_rate_pct": 3.8, "deforestation_relevant": True},
    {"commodity_id": "cattle_dairy", "name": "Cattle (Dairy)", "category": "livestock", "pathway_reduction_rate_pct": 3.2, "deforestation_relevant": True},
    {"commodity_id": "poultry", "name": "Poultry", "category": "livestock", "pathway_reduction_rate_pct": 2.5, "deforestation_relevant": True},
    {"commodity_id": "pork", "name": "Pork", "category": "livestock", "pathway_reduction_rate_pct": 2.8, "deforestation_relevant": True},
    {"commodity_id": "palm_oil", "name": "Palm Oil", "category": "crops", "pathway_reduction_rate_pct": 5.0, "deforestation_relevant": True},
    {"commodity_id": "soy", "name": "Soy", "category": "crops", "pathway_reduction_rate_pct": 4.5, "deforestation_relevant": True},
    {"commodity_id": "rice", "name": "Rice", "category": "crops", "pathway_reduction_rate_pct": 2.0, "deforestation_relevant": False},
    {"commodity_id": "wheat", "name": "Wheat", "category": "crops", "pathway_reduction_rate_pct": 1.8, "deforestation_relevant": False},
    {"commodity_id": "maize", "name": "Maize (Corn)", "category": "crops", "pathway_reduction_rate_pct": 1.8, "deforestation_relevant": False},
    {"commodity_id": "timber_pulp", "name": "Timber & Pulp", "category": "forestry", "pathway_reduction_rate_pct": 4.2, "deforestation_relevant": True},
    {"commodity_id": "other_crops", "name": "Other Crops", "category": "crops", "pathway_reduction_rate_pct": 2.0, "deforestation_relevant": False},
]


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_org_settings: Dict[str, Dict[str, Any]] = {}
_sector_settings: Dict[str, Dict[str, Any]] = {}
_framework_settings: Dict[str, Dict[str, Any]] = {}
_mrv_settings: Dict[str, Dict[str, Any]] = {}
_notification_settings: Dict[str, Dict[str, Any]] = {}


def _now() -> datetime:
    return datetime.utcnow()


def _default_settings(org_id: str) -> Dict[str, Any]:
    return {
        "org_id": org_id,
        "organization_name": "Default Organization",
        "industry": "General",
        "contact_email": None,
        "fiscal_year_end": FiscalYearEnd.DECEMBER.value,
        "reporting_currency": Currency.USD.value,
        "emission_unit": EmissionUnit.TCO2E.value,
        "base_year": 2020,
        "sbti_commitment_date": None,
        "sbti_validation_date": None,
        "sbti_status": "none",
        "updated_at": _now(),
    }


def _default_sector(org_id: str) -> Dict[str, Any]:
    return {
        "org_id": org_id,
        "sector_id": "general",
        "sector_name": "All Other Sectors",
        "sub_sector": None,
        "isic_code": None,
        "nace_code": None,
        "naics_code": None,
        "methodology_preference": "aca",
        "sda_available": False,
        "intensity_unit": "tCO2e",
        "updated_at": _now(),
    }


def _default_frameworks(org_id: str) -> Dict[str, Any]:
    return {
        "org_id": org_id,
        "primary_framework": "sbti",
        "secondary_frameworks": ["cdp", "tcfd"],
        "enable_csrd_alignment": True,
        "enable_cdp_alignment": True,
        "enable_tcfd_alignment": True,
        "enable_iso14064_alignment": False,
        "enable_sec_alignment": False,
        "updated_at": _now(),
    }


def _default_mrv(org_id: str) -> Dict[str, Any]:
    return {
        "org_id": org_id,
        "mrv_enabled": True,
        "auto_sync_emissions": True,
        "sync_frequency": "daily",
        "scope1_agents": ["gl_sc_agent", "gl_mc_agent", "gl_pe_agent", "gl_fe_agent"],
        "scope2_agents": ["gl_s2l_agent", "gl_s2m_agent"],
        "scope3_agents": ["gl_pgs_agent", "gl_uto_agent", "gl_bt_agent", "gl_ec_agent"],
        "data_quality_threshold": 3.0,
        "last_sync_at": None,
        "updated_at": _now(),
    }


def _default_notifications(org_id: str) -> Dict[str, Any]:
    return {
        "org_id": org_id,
        "email_notifications": True,
        "notification_email": None,
        "progress_alerts": True,
        "deadline_reminders": True,
        "reminder_days_before": 30,
        "off_track_alerts": True,
        "recalculation_alerts": True,
        "review_reminders": True,
        "updated_at": _now(),
    }


# ---------------------------------------------------------------------------
# Endpoints -- Organization Settings
# ---------------------------------------------------------------------------

@router.get(
    "/org/{org_id}",
    response_model=SettingsResponse,
    summary="Get organization settings",
    description="Retrieve organization-level SBTi platform settings.",
)
async def get_settings(org_id: str) -> SettingsResponse:
    """Get organization settings."""
    if org_id not in _org_settings:
        _org_settings[org_id] = _default_settings(org_id)
    return SettingsResponse(**_org_settings[org_id])


@router.put(
    "/org/{org_id}",
    response_model=SettingsResponse,
    summary="Update organization settings",
    description="Update organization-level SBTi platform settings.",
)
async def update_settings(org_id: str, request: UpdateSettingsRequest) -> SettingsResponse:
    """Update organization settings."""
    if org_id not in _org_settings:
        _org_settings[org_id] = _default_settings(org_id)
    settings = _org_settings[org_id]
    updates = request.model_dump(exclude_unset=True)
    for key, value in updates.items():
        if hasattr(value, "value"):
            updates[key] = value.value
    settings.update(updates)
    settings["updated_at"] = _now()
    return SettingsResponse(**settings)


# ---------------------------------------------------------------------------
# Endpoints -- Sector Classification
# ---------------------------------------------------------------------------

@router.get(
    "/org/{org_id}/sector",
    response_model=SectorSettingsResponse,
    summary="Get sector classification",
    description="Retrieve the organization's SBTi sector classification.",
)
async def get_sector(org_id: str) -> SectorSettingsResponse:
    """Get sector classification."""
    if org_id not in _sector_settings:
        _sector_settings[org_id] = _default_sector(org_id)
    return SectorSettingsResponse(**_sector_settings[org_id])


@router.put(
    "/org/{org_id}/sector",
    response_model=SectorSettingsResponse,
    summary="Update sector classification",
    description="Update the organization's SBTi sector classification.",
)
async def update_sector(org_id: str, request: UpdateSectorRequest) -> SectorSettingsResponse:
    """Update sector classification."""
    sec = next((s for s in SECTORS if s["sector_id"] == request.sector_id), None)
    if not sec:
        valid = [s["sector_id"] for s in SECTORS]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sector '{request.sector_id}'. Valid: {valid}",
        )

    data = {
        "org_id": org_id,
        "sector_id": request.sector_id,
        "sector_name": sec["name"],
        "sub_sector": request.sub_sector,
        "isic_code": request.isic_code,
        "nace_code": request.nace_code,
        "naics_code": request.naics_code,
        "methodology_preference": request.methodology_preference or sec["methodology"].lower(),
        "sda_available": sec["sda_available"],
        "intensity_unit": sec["intensity_unit"],
        "updated_at": _now(),
    }
    _sector_settings[org_id] = data
    return SectorSettingsResponse(**data)


# ---------------------------------------------------------------------------
# Endpoints -- Framework Preferences
# ---------------------------------------------------------------------------

@router.get(
    "/org/{org_id}/frameworks",
    response_model=FrameworkSettingsResponse,
    summary="Get framework preferences",
    description="Retrieve framework alignment preferences.",
)
async def get_frameworks(org_id: str) -> FrameworkSettingsResponse:
    """Get framework preferences."""
    if org_id not in _framework_settings:
        _framework_settings[org_id] = _default_frameworks(org_id)
    return FrameworkSettingsResponse(**_framework_settings[org_id])


@router.put(
    "/org/{org_id}/frameworks",
    response_model=FrameworkSettingsResponse,
    summary="Update framework preferences",
    description="Update framework alignment preferences.",
)
async def update_frameworks(org_id: str, request: UpdateFrameworksRequest) -> FrameworkSettingsResponse:
    """Update framework preferences."""
    data = {
        "org_id": org_id,
        **request.model_dump(),
        "updated_at": _now(),
    }
    _framework_settings[org_id] = data
    return FrameworkSettingsResponse(**data)


# ---------------------------------------------------------------------------
# Endpoints -- MRV Connection
# ---------------------------------------------------------------------------

@router.get(
    "/org/{org_id}/mrv-connection",
    response_model=MRVSettingsResponse,
    summary="Get MRV connection settings",
    description="Retrieve MRV agent connection and synchronization settings.",
)
async def get_mrv_connection(org_id: str) -> MRVSettingsResponse:
    """Get MRV connection settings."""
    if org_id not in _mrv_settings:
        _mrv_settings[org_id] = _default_mrv(org_id)
    return MRVSettingsResponse(**_mrv_settings[org_id])


@router.put(
    "/org/{org_id}/mrv-connection",
    response_model=MRVSettingsResponse,
    summary="Update MRV connection settings",
    description="Update MRV agent connection and synchronization settings.",
)
async def update_mrv_connection(org_id: str, request: UpdateMRVRequest) -> MRVSettingsResponse:
    """Update MRV connection settings."""
    if org_id not in _mrv_settings:
        _mrv_settings[org_id] = _default_mrv(org_id)
    settings = _mrv_settings[org_id]
    updates = request.model_dump(exclude_unset=True)
    settings.update(updates)
    settings["updated_at"] = _now()
    return MRVSettingsResponse(**settings)


# ---------------------------------------------------------------------------
# Endpoints -- Notification Preferences
# ---------------------------------------------------------------------------

@router.get(
    "/org/{org_id}/notifications",
    response_model=NotificationSettingsResponse,
    summary="Get notification preferences",
    description="Retrieve notification and alert preferences.",
)
async def get_notifications(org_id: str) -> NotificationSettingsResponse:
    """Get notification preferences."""
    if org_id not in _notification_settings:
        _notification_settings[org_id] = _default_notifications(org_id)
    return NotificationSettingsResponse(**_notification_settings[org_id])


@router.put(
    "/org/{org_id}/notifications",
    response_model=NotificationSettingsResponse,
    summary="Update notification preferences",
    description="Update notification and alert preferences.",
)
async def update_notifications(org_id: str, request: UpdateNotificationsRequest) -> NotificationSettingsResponse:
    """Update notification preferences."""
    data = {
        "org_id": org_id,
        **request.model_dump(),
        "updated_at": _now(),
    }
    _notification_settings[org_id] = data
    return NotificationSettingsResponse(**data)


# ---------------------------------------------------------------------------
# Endpoints -- Reference Data: Criteria Definitions
# ---------------------------------------------------------------------------

@router.get(
    "/criteria/definitions",
    response_model=List[CriterionDefinitionResponse],
    summary="All SBTi criteria definitions",
    description=(
        "List all SBTi criteria definitions with descriptions, thresholds, "
        "applicability, and version information."
    ),
)
async def get_criteria_definitions(
    category: Optional[str] = Query(None, description="Filter by category"),
    applies_to: Optional[str] = Query(None, description="Filter by target type"),
) -> List[CriterionDefinitionResponse]:
    """Get criteria definitions."""
    results = CRITERIA
    if category:
        results = [c for c in results if c["category"] == category]
    if applies_to:
        results = [c for c in results if applies_to in c["applies_to"]]
    return [CriterionDefinitionResponse(**c) for c in results]


# ---------------------------------------------------------------------------
# Endpoints -- Reference Data: Sectors
# ---------------------------------------------------------------------------

@router.get(
    "/sectors/list",
    response_model=List[SectorListItem],
    summary="All sectors with metadata",
    description="List all SBTi sectors with methodology, intensity units, and sub-sectors.",
)
async def list_sectors() -> List[SectorListItem]:
    """List all sectors."""
    return [SectorListItem(**s) for s in SECTORS]


# ---------------------------------------------------------------------------
# Endpoints -- Reference Data: FLAG Commodities
# ---------------------------------------------------------------------------

@router.get(
    "/flag-commodities/list",
    response_model=List[FLAGCommodityItem],
    summary="All FLAG commodities",
    description="List all 11 FLAG commodities with pathway reduction rates.",
)
async def list_flag_commodities() -> List[FLAGCommodityItem]:
    """List all FLAG commodities."""
    return [FLAGCommodityItem(**c) for c in FLAG_COMMODITIES]
