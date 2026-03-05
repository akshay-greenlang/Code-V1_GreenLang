"""
GL-Taxonomy-APP Settings API

Manages organization-level configuration for the EU Taxonomy Alignment
& Green Investment Ratio Platform.  Provides endpoints for organization
settings, reporting periods, custom thresholds, and MRV agent mapping.

Settings Categories:
    - Organization: Name, entity type, size, CSRD scope, contact
    - Reporting Periods: Fiscal year configuration, reporting deadlines
    - Thresholds: Custom de minimis, DQ, CapEx plan thresholds
    - MRV Mapping: Integration with GreenLang MRV agents for emissions data
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/settings", tags=["Settings"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """Entity type for taxonomy reporting."""
    NON_FINANCIAL = "non_financial"
    CREDIT_INSTITUTION = "credit_institution"
    ASSET_MANAGER = "asset_manager"
    INSURANCE = "insurance"
    INVESTMENT_FIRM = "investment_firm"


class ReportingFramework(str, Enum):
    """Reporting framework."""
    CSRD = "csrd"
    NFRD = "nfrd"
    EBA_PILLAR3 = "eba_pillar3"
    VOLUNTARY = "voluntary"


class Currency(str, Enum):
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"
    SEK = "SEK"
    DKK = "DKK"
    NOK = "NOK"
    PLN = "PLN"


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
    entity_type: Optional[EntityType] = None
    employee_count: Optional[int] = Field(None, ge=1)
    annual_turnover_eur: Optional[float] = Field(None, ge=0)
    total_assets_eur: Optional[float] = Field(None, ge=0)
    reporting_framework: Optional[ReportingFramework] = None
    fiscal_year_end: Optional[FiscalYearEnd] = None
    reporting_currency: Optional[Currency] = None
    country_code: Optional[str] = Field(None, max_length=2)
    contact_email: Optional[str] = Field(None, max_length=300)
    csrd_scope: Optional[str] = Field(None, description="in_scope, out_of_scope, voluntary")
    nace_codes: Optional[List[str]] = Field(None, description="Organization primary NACE codes")


class CreateReportingPeriodRequest(BaseModel):
    """Create a reporting period."""
    period_name: str = Field(..., max_length=100)
    start_date: str = Field(..., description="ISO date")
    end_date: str = Field(..., description="ISO date")
    reporting_deadline: str = Field(..., description="ISO date for submission deadline")
    framework: ReportingFramework = Field(ReportingFramework.CSRD)
    notes: Optional[str] = Field(None, max_length=2000)


class UpdateThresholdsRequest(BaseModel):
    """Update custom thresholds."""
    de_minimis_pct: Optional[float] = Field(None, ge=0, le=100, description="De minimis threshold %")
    dq_minimum_score: Optional[float] = Field(None, ge=1, le=5, description="Minimum DQ score for alignment")
    sc_confidence_threshold: Optional[float] = Field(None, ge=0, le=1, description="SC confidence threshold")
    capex_plan_max_years: Optional[int] = Field(None, ge=1, le=10, description="Max CapEx plan duration")
    epc_alignment_grades: Optional[List[str]] = Field(None, description="EPC grades for alignment (e.g. [A, B])")
    transitional_reporting_enabled: Optional[bool] = Field(None)
    enabling_reporting_enabled: Optional[bool] = Field(None)
    omnibus_simplified: Optional[bool] = Field(None, description="Use Omnibus simplified reporting")


class UpdateMRVMappingRequest(BaseModel):
    """Update MRV agent mapping."""
    mrv_enabled: bool = Field(True)
    auto_sync: bool = Field(True)
    sync_frequency: str = Field("daily", description="hourly, daily, weekly")
    scope1_agents: List[str] = Field(default_factory=list)
    scope2_agents: List[str] = Field(default_factory=list)
    scope3_agents: List[str] = Field(default_factory=list)
    emission_factor_source: str = Field("mrv_calculated", description="mrv_calculated, manual, hybrid")
    data_quality_threshold: float = Field(3.0, ge=1, le=5)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SettingsResponse(BaseModel):
    """Organization settings."""
    org_id: str
    organization_name: str
    entity_type: str
    employee_count: Optional[int]
    annual_turnover_eur: Optional[float]
    total_assets_eur: Optional[float]
    reporting_framework: str
    fiscal_year_end: str
    reporting_currency: str
    country_code: str
    contact_email: Optional[str]
    csrd_scope: str
    nace_codes: List[str]
    updated_at: datetime


class ReportingPeriodResponse(BaseModel):
    """Reporting period."""
    period_id: str
    org_id: str
    period_name: str
    start_date: str
    end_date: str
    reporting_deadline: str
    framework: str
    status: str
    notes: Optional[str]
    created_at: datetime


class ReportingPeriodsListResponse(BaseModel):
    """List of reporting periods."""
    org_id: str
    periods: List[ReportingPeriodResponse]
    total_count: int
    current_period: Optional[str]
    generated_at: datetime


class ThresholdsResponse(BaseModel):
    """Custom thresholds."""
    org_id: str
    de_minimis_pct: float
    dq_minimum_score: float
    sc_confidence_threshold: float
    capex_plan_max_years: int
    epc_alignment_grades: List[str]
    transitional_reporting_enabled: bool
    enabling_reporting_enabled: bool
    omnibus_simplified: bool
    updated_at: datetime


class MRVMappingResponse(BaseModel):
    """MRV agent mapping."""
    org_id: str
    mrv_enabled: bool
    auto_sync: bool
    sync_frequency: str
    scope1_agents: List[str]
    scope2_agents: List[str]
    scope3_agents: List[str]
    emission_factor_source: str
    data_quality_threshold: float
    last_sync_at: Optional[datetime]
    agent_status: Dict[str, str]
    updated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_org_settings: Dict[str, Dict[str, Any]] = {}
_reporting_periods: Dict[str, List[Dict[str, Any]]] = {}
_thresholds: Dict[str, Dict[str, Any]] = {}
_mrv_mappings: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _default_settings(org_id: str) -> Dict[str, Any]:
    return {
        "org_id": org_id,
        "organization_name": "Default Organization",
        "entity_type": EntityType.NON_FINANCIAL.value,
        "employee_count": None,
        "annual_turnover_eur": None,
        "total_assets_eur": None,
        "reporting_framework": ReportingFramework.CSRD.value,
        "fiscal_year_end": FiscalYearEnd.DECEMBER.value,
        "reporting_currency": Currency.EUR.value,
        "country_code": "DE",
        "contact_email": None,
        "csrd_scope": "in_scope",
        "nace_codes": [],
        "updated_at": _now(),
    }


def _default_thresholds(org_id: str) -> Dict[str, Any]:
    return {
        "org_id": org_id,
        "de_minimis_pct": 5.0,
        "dq_minimum_score": 3.0,
        "sc_confidence_threshold": 0.8,
        "capex_plan_max_years": 5,
        "epc_alignment_grades": ["A", "B"],
        "transitional_reporting_enabled": True,
        "enabling_reporting_enabled": True,
        "omnibus_simplified": False,
        "updated_at": _now(),
    }


def _default_mrv(org_id: str) -> Dict[str, Any]:
    return {
        "org_id": org_id,
        "mrv_enabled": True,
        "auto_sync": True,
        "sync_frequency": "daily",
        "scope1_agents": ["gl_sc_agent", "gl_mc_agent", "gl_pe_agent", "gl_fe_agent"],
        "scope2_agents": ["gl_s2l_agent", "gl_s2m_agent"],
        "scope3_agents": ["gl_pgs_agent", "gl_uto_agent", "gl_bt_agent", "gl_ec_agent"],
        "emission_factor_source": "mrv_calculated",
        "data_quality_threshold": 3.0,
        "last_sync_at": None,
        "agent_status": {
            "gl_sc_agent": "connected", "gl_mc_agent": "connected",
            "gl_pe_agent": "connected", "gl_fe_agent": "connected",
            "gl_s2l_agent": "connected", "gl_s2m_agent": "connected",
            "gl_pgs_agent": "connected", "gl_uto_agent": "connected",
            "gl_bt_agent": "connected", "gl_ec_agent": "connected",
        },
        "updated_at": _now(),
    }


# ---------------------------------------------------------------------------
# Endpoints -- Organization Settings
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}",
    response_model=SettingsResponse,
    summary="Get org settings",
    description="Retrieve organization-level EU Taxonomy platform settings.",
)
async def get_settings(org_id: str) -> SettingsResponse:
    """Get organization settings."""
    if org_id not in _org_settings:
        _org_settings[org_id] = _default_settings(org_id)
    return SettingsResponse(**_org_settings[org_id])


@router.put(
    "/{org_id}",
    response_model=SettingsResponse,
    summary="Update org settings",
    description="Update organization-level EU Taxonomy platform settings.",
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
# Endpoints -- Reporting Periods
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}/reporting-periods",
    response_model=ReportingPeriodsListResponse,
    summary="Get reporting periods",
    description="Retrieve all reporting periods for an organization.",
)
async def get_reporting_periods(org_id: str) -> ReportingPeriodsListResponse:
    """Get reporting periods."""
    periods = _reporting_periods.get(org_id, [])
    periods.sort(key=lambda p: p["end_date"], reverse=True)

    current = None
    now_str = _now().strftime("%Y-%m-%d")
    for p in periods:
        if p["start_date"] <= now_str <= p["end_date"]:
            current = p["period_id"]
            break

    return ReportingPeriodsListResponse(
        org_id=org_id,
        periods=[ReportingPeriodResponse(**p) for p in periods],
        total_count=len(periods),
        current_period=current,
        generated_at=_now(),
    )


@router.post(
    "/{org_id}/reporting-periods",
    response_model=ReportingPeriodResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create reporting period",
    description="Create a new reporting period.",
)
async def create_reporting_period(
    org_id: str,
    request: CreateReportingPeriodRequest,
) -> ReportingPeriodResponse:
    """Create a reporting period."""
    period_id = _generate_id("rp")
    data = {
        "period_id": period_id,
        "org_id": org_id,
        "period_name": request.period_name,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "reporting_deadline": request.reporting_deadline,
        "framework": request.framework.value,
        "status": "active",
        "notes": request.notes,
        "created_at": _now(),
    }
    if org_id not in _reporting_periods:
        _reporting_periods[org_id] = []
    _reporting_periods[org_id].append(data)
    return ReportingPeriodResponse(**data)


# ---------------------------------------------------------------------------
# Endpoints -- Custom Thresholds
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}/thresholds",
    response_model=ThresholdsResponse,
    summary="Get custom thresholds",
    description="Retrieve custom threshold settings for taxonomy assessment.",
)
async def get_thresholds(org_id: str) -> ThresholdsResponse:
    """Get custom thresholds."""
    if org_id not in _thresholds:
        _thresholds[org_id] = _default_thresholds(org_id)
    return ThresholdsResponse(**_thresholds[org_id])


@router.put(
    "/{org_id}/thresholds",
    response_model=ThresholdsResponse,
    summary="Update thresholds",
    description="Update custom threshold settings.",
)
async def update_thresholds(org_id: str, request: UpdateThresholdsRequest) -> ThresholdsResponse:
    """Update thresholds."""
    if org_id not in _thresholds:
        _thresholds[org_id] = _default_thresholds(org_id)
    thresholds = _thresholds[org_id]
    updates = request.model_dump(exclude_unset=True)
    thresholds.update(updates)
    thresholds["updated_at"] = _now()
    return ThresholdsResponse(**thresholds)


# ---------------------------------------------------------------------------
# Endpoints -- MRV Agent Mapping
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}/mrv-mapping",
    response_model=MRVMappingResponse,
    summary="Get MRV agent mapping",
    description="Retrieve MRV agent integration mapping for emissions data.",
)
async def get_mrv_mapping(org_id: str) -> MRVMappingResponse:
    """Get MRV mapping."""
    if org_id not in _mrv_mappings:
        _mrv_mappings[org_id] = _default_mrv(org_id)
    return MRVMappingResponse(**_mrv_mappings[org_id])


@router.put(
    "/{org_id}/mrv-mapping",
    response_model=MRVMappingResponse,
    summary="Update MRV mapping",
    description="Update MRV agent integration mapping.",
)
async def update_mrv_mapping(org_id: str, request: UpdateMRVMappingRequest) -> MRVMappingResponse:
    """Update MRV mapping."""
    if org_id not in _mrv_mappings:
        _mrv_mappings[org_id] = _default_mrv(org_id)
    mapping = _mrv_mappings[org_id]
    updates = request.model_dump(exclude_unset=True)
    mapping.update(updates)
    mapping["updated_at"] = _now()

    # Update agent status based on new agent lists
    all_agents = set(mapping.get("scope1_agents", []) + mapping.get("scope2_agents", []) + mapping.get("scope3_agents", []))
    mapping["agent_status"] = {agent: "connected" for agent in all_agents}

    return MRVMappingResponse(**mapping)
