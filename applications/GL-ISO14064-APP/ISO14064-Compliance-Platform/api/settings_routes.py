"""
GL-ISO14064-APP Settings API

Manages application-level settings for the ISO 14064-1 compliance platform
including GWP source selection, emission factor database defaults, materiality
thresholds, notification preferences, and export templates.

GWP sources: AR4 (2007), AR5 (2014), AR6 (2021)
EF databases: IPCC 2006, DEFRA 2024, EPA 2024, ecoinvent 3.10, custom
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/iso14064/settings", tags=["Settings"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GWPSource(str, Enum):
    AR4 = "ar4"
    AR5 = "ar5"
    AR6 = "ar6"


class EFDatabase(str, Enum):
    IPCC_2006 = "ipcc_2006"
    DEFRA_2024 = "defra_2024"
    EPA_2024 = "epa_2024"
    ECOINVENT_3_10 = "ecoinvent_3_10"
    CUSTOM = "custom"


class ExportFormat(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class GeneralSettings(BaseModel):
    org_id: str
    default_reporting_year: int
    base_year: int
    consolidation_approach: str
    default_gwp_source: str
    default_ef_database: str
    default_currency: str
    default_unit_system: str
    auto_recalculate: bool
    updated_at: datetime


class UpdateGeneralSettingsRequest(BaseModel):
    default_reporting_year: Optional[int] = Field(None, ge=1990, le=2100)
    base_year: Optional[int] = Field(None, ge=1990, le=2100)
    consolidation_approach: Optional[str] = None
    default_currency: Optional[str] = None
    default_unit_system: Optional[str] = None
    auto_recalculate: Optional[bool] = None

    class Config:
        json_schema_extra = {"example": {"default_reporting_year": 2025, "base_year": 2019, "auto_recalculate": True}}


class GWPSourceInfo(BaseModel):
    source: str
    name: str
    year: int
    description: str
    is_active: bool


class SetGWPSourceRequest(BaseModel):
    source: GWPSource = Field(...)

    class Config:
        json_schema_extra = {"example": {"source": "ar6"}}


class EFDatabaseInfo(BaseModel):
    database: str
    name: str
    version: str
    description: str
    last_updated: str
    is_default: bool


class SetDefaultEFDatabaseRequest(BaseModel):
    database: EFDatabase = Field(...)

    class Config:
        json_schema_extra = {"example": {"database": "defra_2024"}}


class ThresholdSettings(BaseModel):
    significance_threshold_pct: float
    materiality_threshold_tco2e: float
    de_minimis_threshold_pct: float
    data_quality_minimum_score: float
    recalculation_trigger_pct: float
    updated_at: datetime


class UpdateThresholdSettingsRequest(BaseModel):
    significance_threshold_pct: Optional[float] = Field(None, ge=0, le=100)
    materiality_threshold_tco2e: Optional[float] = Field(None, ge=0)
    de_minimis_threshold_pct: Optional[float] = Field(None, ge=0, le=100)
    data_quality_minimum_score: Optional[float] = Field(None, ge=0, le=100)
    recalculation_trigger_pct: Optional[float] = Field(None, ge=0, le=100)


class NotificationPreferences(BaseModel):
    email_enabled: bool
    email_recipients: List[str]
    notify_on_data_import: bool
    notify_on_verification_update: bool
    notify_on_threshold_breach: bool
    notify_on_report_generation: bool
    notify_on_corrective_action: bool
    digest_frequency: str
    updated_at: datetime


class UpdateNotificationPreferencesRequest(BaseModel):
    email_enabled: Optional[bool] = None
    email_recipients: Optional[List[str]] = None
    notify_on_data_import: Optional[bool] = None
    notify_on_verification_update: Optional[bool] = None
    notify_on_threshold_breach: Optional[bool] = None
    notify_on_report_generation: Optional[bool] = None
    notify_on_corrective_action: Optional[bool] = None
    digest_frequency: Optional[str] = None


class ExportTemplate(BaseModel):
    template_id: str
    name: str
    description: str
    format: str
    sections: List[str]
    include_appendices: bool
    include_verification_statement: bool
    created_at: datetime


class CreateExportTemplateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    format: ExportFormat = Field(ExportFormat.PDF)
    sections: List[str] = Field(default_factory=list)
    include_appendices: bool = Field(True)
    include_verification_statement: bool = Field(False)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "ISO 14064-1 Full Report",
                "format": "pdf",
                "sections": ["organization_description", "emissions", "removals", "uncertainty"],
                "include_appendices": True,
                "include_verification_statement": True,
            }
        }


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_general_settings: Dict[str, Any] = {
    "org_id": "default", "default_reporting_year": 2025, "base_year": 2019,
    "consolidation_approach": "operational_control", "default_gwp_source": GWPSource.AR6.value,
    "default_ef_database": EFDatabase.DEFRA_2024.value, "default_currency": "USD",
    "default_unit_system": "metric", "auto_recalculate": True, "updated_at": datetime.utcnow(),
}

_active_gwp_source: str = GWPSource.AR6.value
_default_ef_database: str = EFDatabase.DEFRA_2024.value

_threshold_settings: Dict[str, Any] = {
    "significance_threshold_pct": 5.0, "materiality_threshold_tco2e": 500.0,
    "de_minimis_threshold_pct": 1.0, "data_quality_minimum_score": 70.0,
    "recalculation_trigger_pct": 10.0, "updated_at": datetime.utcnow(),
}

_notification_preferences: Dict[str, Any] = {
    "email_enabled": True, "email_recipients": ["sustainability@example.com"],
    "notify_on_data_import": True, "notify_on_verification_update": True,
    "notify_on_threshold_breach": True, "notify_on_report_generation": False,
    "notify_on_corrective_action": True, "digest_frequency": "daily", "updated_at": datetime.utcnow(),
}

_export_templates: Dict[str, Dict[str, Any]] = {}

GWP_SOURCES = [
    {"source": GWPSource.AR4.value, "name": "IPCC Fourth Assessment Report", "year": 2007,
     "description": "GWP values from IPCC AR4 (2007). 100-year time horizon."},
    {"source": GWPSource.AR5.value, "name": "IPCC Fifth Assessment Report", "year": 2014,
     "description": "GWP values from IPCC AR5 (2014). Required by UNFCCC."},
    {"source": GWPSource.AR6.value, "name": "IPCC Sixth Assessment Report", "year": 2021,
     "description": "GWP values from IPCC AR6 (2021). Recommended for ISO 14064-1."},
]

EF_DATABASES = [
    {"database": EFDatabase.IPCC_2006.value, "name": "IPCC 2006 Guidelines", "version": "2006 (2019 Refinement)",
     "description": "IPCC default Tier 1 factors.", "last_updated": "2019-05-01"},
    {"database": EFDatabase.DEFRA_2024.value, "name": "UK DEFRA/BEIS Conversion Factors", "version": "2024",
     "description": "UK Government GHG conversion factors.", "last_updated": "2024-06-01"},
    {"database": EFDatabase.EPA_2024.value, "name": "US EPA Emission Factors Hub", "version": "2024",
     "description": "US EPA emission factors for GHG inventories.", "last_updated": "2024-04-01"},
    {"database": EFDatabase.ECOINVENT_3_10.value, "name": "ecoinvent", "version": "3.10",
     "description": "LCI database with 18,000+ datasets.", "last_updated": "2023-12-01"},
    {"database": EFDatabase.CUSTOM.value, "name": "Custom Emission Factors", "version": "User-managed",
     "description": "Organization-specific emission factors.", "last_updated": "N/A"},
]


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints -- General Settings
# ---------------------------------------------------------------------------

@router.get("/general", response_model=GeneralSettings, summary="Get general settings",
            description="Retrieve general platform settings.")
async def get_general_settings() -> GeneralSettings:
    return GeneralSettings(**_general_settings)


@router.put("/general", response_model=GeneralSettings, summary="Update general settings",
            description="Update general platform settings.")
async def update_general_settings(request: UpdateGeneralSettingsRequest) -> GeneralSettings:
    updates = request.model_dump(exclude_unset=True)
    _general_settings.update(updates)
    _general_settings["updated_at"] = _now()
    return GeneralSettings(**_general_settings)


# ---------------------------------------------------------------------------
# Endpoints -- GWP Sources
# ---------------------------------------------------------------------------

@router.get("/gwp-sources", response_model=List[GWPSourceInfo], summary="List GWP sources",
            description="List available IPCC AR GWP sources and indicate active source.")
async def list_gwp_sources() -> List[GWPSourceInfo]:
    global _active_gwp_source
    return [GWPSourceInfo(source=s["source"], name=s["name"], year=s["year"],
                          description=s["description"], is_active=(s["source"] == _active_gwp_source)) for s in GWP_SOURCES]


@router.put("/gwp-source", response_model=GWPSourceInfo, summary="Set active GWP source",
            description="Set the active GWP source for emission calculations.")
async def set_gwp_source(request: SetGWPSourceRequest) -> GWPSourceInfo:
    global _active_gwp_source
    _active_gwp_source = request.source.value
    _general_settings["default_gwp_source"] = _active_gwp_source
    _general_settings["updated_at"] = _now()
    info = next(s for s in GWP_SOURCES if s["source"] == _active_gwp_source)
    return GWPSourceInfo(source=info["source"], name=info["name"], year=info["year"],
                         description=info["description"], is_active=True)


# ---------------------------------------------------------------------------
# Endpoints -- Emission Factor Databases
# ---------------------------------------------------------------------------

@router.get("/emission-factors", response_model=List[EFDatabaseInfo], summary="List emission factor databases",
            description="List available emission factor databases.")
async def list_ef_databases() -> List[EFDatabaseInfo]:
    global _default_ef_database
    return [EFDatabaseInfo(database=d["database"], name=d["name"], version=d["version"],
                           description=d["description"], last_updated=d["last_updated"],
                           is_default=(d["database"] == _default_ef_database)) for d in EF_DATABASES]


@router.put("/emission-factors/default", response_model=EFDatabaseInfo, summary="Set default EF database",
            description="Set the default emission factor database.")
async def set_default_ef_database(request: SetDefaultEFDatabaseRequest) -> EFDatabaseInfo:
    global _default_ef_database
    _default_ef_database = request.database.value
    _general_settings["default_ef_database"] = _default_ef_database
    _general_settings["updated_at"] = _now()
    info = next(d for d in EF_DATABASES if d["database"] == _default_ef_database)
    return EFDatabaseInfo(database=info["database"], name=info["name"], version=info["version"],
                          description=info["description"], last_updated=info["last_updated"], is_default=True)


# ---------------------------------------------------------------------------
# Endpoints -- Thresholds
# ---------------------------------------------------------------------------

@router.get("/thresholds", response_model=ThresholdSettings, summary="Get threshold settings",
            description="Retrieve significance, materiality, and data quality thresholds.")
async def get_thresholds() -> ThresholdSettings:
    return ThresholdSettings(**_threshold_settings)


@router.put("/thresholds", response_model=ThresholdSettings, summary="Update threshold settings",
            description="Update significance, materiality, or data quality thresholds.")
async def update_thresholds(request: UpdateThresholdSettingsRequest) -> ThresholdSettings:
    updates = request.model_dump(exclude_unset=True)
    _threshold_settings.update(updates)
    _threshold_settings["updated_at"] = _now()
    return ThresholdSettings(**_threshold_settings)


# ---------------------------------------------------------------------------
# Endpoints -- Notification Preferences
# ---------------------------------------------------------------------------

@router.get("/notification-preferences", response_model=NotificationPreferences,
            summary="Get notification preferences", description="Retrieve notification settings.")
async def get_notification_preferences() -> NotificationPreferences:
    return NotificationPreferences(**_notification_preferences)


@router.put("/notification-preferences", response_model=NotificationPreferences,
            summary="Update notification preferences", description="Update notification settings.")
async def update_notification_preferences(request: UpdateNotificationPreferencesRequest) -> NotificationPreferences:
    updates = request.model_dump(exclude_unset=True)
    _notification_preferences.update(updates)
    _notification_preferences["updated_at"] = _now()
    return NotificationPreferences(**_notification_preferences)


# ---------------------------------------------------------------------------
# Endpoints -- Export Templates
# ---------------------------------------------------------------------------

@router.get("/export-templates", response_model=List[ExportTemplate], summary="List export templates",
            description="List available export templates including built-in and custom.")
async def list_export_templates() -> List[ExportTemplate]:
    builtins = [
        ExportTemplate(template_id="tpl_builtin_full", name="ISO 14064-1 Full Report",
                       description="Complete ISO 14064-1 compliant report.", format="pdf",
                       sections=["organization", "boundary", "emissions", "removals", "uncertainty", "base_year", "management_plan"],
                       include_appendices=True, include_verification_statement=True, created_at=datetime(2025, 1, 1)),
        ExportTemplate(template_id="tpl_builtin_summary", name="Executive Summary",
                       description="High-level emissions summary.", format="pdf",
                       sections=["organization", "emissions", "removals"],
                       include_appendices=False, include_verification_statement=False, created_at=datetime(2025, 1, 1)),
        ExportTemplate(template_id="tpl_builtin_data", name="Raw Data Export",
                       description="Tabular emission source data.", format="excel",
                       sections=["emissions", "removals"],
                       include_appendices=False, include_verification_statement=False, created_at=datetime(2025, 1, 1)),
    ]
    custom = [ExportTemplate(**t) for t in _export_templates.values()]
    return builtins + custom


@router.post("/export-templates", response_model=ExportTemplate, status_code=status.HTTP_201_CREATED,
             summary="Create export template", description="Create a custom export template.")
async def create_export_template(request: CreateExportTemplateRequest) -> ExportTemplate:
    template_id = _generate_id("tpl")
    now = _now()
    template = {
        "template_id": template_id, "name": request.name, "description": request.description,
        "format": request.format.value, "sections": request.sections,
        "include_appendices": request.include_appendices,
        "include_verification_statement": request.include_verification_statement, "created_at": now,
    }
    _export_templates[template_id] = template
    return ExportTemplate(**template)
