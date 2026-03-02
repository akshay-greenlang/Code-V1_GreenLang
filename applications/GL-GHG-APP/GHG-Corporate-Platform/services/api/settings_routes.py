"""
GL-GHG-APP Settings API

Platform-wide configuration for GHG accounting parameters:
    - GWP version (AR4, AR5, AR6)
    - Significance/materiality thresholds
    - Emission factor database preferences
    - Reporting preferences and display options
    - Data quality thresholds
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

router = APIRouter(prefix="/api/v1/settings", tags=["Settings"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GWPVersion(str, Enum):
    """IPCC Assessment Report versions for Global Warming Potentials."""
    AR4 = "AR4"  # 2007
    AR5 = "AR5"  # 2014 -- most commonly used
    AR6 = "AR6"  # 2021


class EFSource(str, Enum):
    """Preferred emission factor database sources."""
    EPA = "epa"
    DEFRA = "defra"
    IPCC = "ipcc"
    ECOINVENT = "ecoinvent"
    EXIOBASE = "exiobase"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GWPValues(BaseModel):
    """GWP values for the selected IPCC Assessment Report."""
    version: str
    co2: int = 1
    ch4: int
    n2o: int
    hfc_134a: int
    sf6: int
    nf3: int
    source: str


class DataQualityThresholds(BaseModel):
    """Thresholds for data quality scoring."""
    minimum_acceptable_score: float = Field(50.0, ge=0, le=100)
    target_score: float = Field(80.0, ge=0, le=100)
    critical_alert_below: float = Field(30.0, ge=0, le=100)
    primary_data_target_pct: float = Field(60.0, ge=0, le=100)


class ReportingPreferences(BaseModel):
    """Display and reporting preferences."""
    currency: str = Field("USD", description="Currency for spend-based calculations")
    unit_system: str = Field("metric", description="metric or imperial")
    decimal_places: int = Field(2, ge=0, le=6)
    show_biogenic_separately: bool = Field(True)
    scope2_primary_method: str = Field("market_based", description="Primary Scope 2 display method")
    fiscal_year_start_month: int = Field(1, ge=1, le=12)


class SettingsResponse(BaseModel):
    """Complete platform settings."""
    gwp_version: str
    gwp_values: GWPValues
    significance_threshold_pct: float
    emission_factor_sources: List[str]
    primary_ef_source: str
    data_quality_thresholds: DataQualityThresholds
    reporting_preferences: ReportingPreferences
    recalculation_policy: Dict[str, Any]
    updated_at: datetime
    updated_by: Optional[str]


class UpdateSettingsRequest(BaseModel):
    """Request to update platform settings."""
    gwp_version: Optional[GWPVersion] = Field(None, description="GWP Assessment Report version")
    significance_threshold_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Materiality threshold for Scope 3 categories"
    )
    primary_ef_source: Optional[EFSource] = Field(None, description="Preferred EF database")
    emission_factor_sources: Optional[List[EFSource]] = Field(
        None, description="Ordered list of EF sources"
    )
    data_quality_thresholds: Optional[DataQualityThresholds] = None
    reporting_preferences: Optional[ReportingPreferences] = None
    recalculation_policy: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Base year recalculation policy: "
            "{trigger_threshold_pct, structural_changes, methodology_changes, cumulative_threshold_pct}"
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "gwp_version": "AR5",
                "significance_threshold_pct": 1.0,
                "primary_ef_source": "epa",
                "emission_factor_sources": ["epa", "defra", "ipcc"],
                "data_quality_thresholds": {
                    "minimum_acceptable_score": 50.0,
                    "target_score": 80.0,
                    "critical_alert_below": 30.0,
                    "primary_data_target_pct": 60.0
                },
                "reporting_preferences": {
                    "currency": "USD",
                    "unit_system": "metric",
                    "decimal_places": 2,
                    "show_biogenic_separately": True,
                    "scope2_primary_method": "market_based",
                    "fiscal_year_start_month": 1
                }
            }
        }


class DefaultSettingsResponse(BaseModel):
    """Default platform settings for reference."""
    gwp_version: str
    gwp_values: GWPValues
    significance_threshold_pct: float
    emission_factor_sources: List[str]
    primary_ef_source: str
    data_quality_thresholds: DataQualityThresholds
    reporting_preferences: ReportingPreferences
    recalculation_policy: Dict[str, Any]
    note: str


# ---------------------------------------------------------------------------
# GWP Data
# ---------------------------------------------------------------------------

GWP_DATA = {
    "AR4": GWPValues(
        version="AR4", ch4=25, n2o=298, hfc_134a=1430, sf6=22800, nf3=17200,
        source="IPCC Fourth Assessment Report (2007)"
    ),
    "AR5": GWPValues(
        version="AR5", ch4=28, n2o=265, hfc_134a=1300, sf6=23500, nf3=16100,
        source="IPCC Fifth Assessment Report (2014)"
    ),
    "AR6": GWPValues(
        version="AR6", ch4=27, n2o=273, hfc_134a=1526, sf6=25200, nf3=17400,
        source="IPCC Sixth Assessment Report (2021)"
    ),
}

# Default recalculation policy per GHG Protocol Ch. 5
DEFAULT_RECALCULATION_POLICY = {
    "trigger_threshold_pct": 5.0,
    "structural_changes": True,
    "methodology_changes": True,
    "cumulative_threshold_pct": 10.0,
    "triggers": [
        "Acquisitions or divestitures",
        "Mergers or restructuring",
        "Changes in calculation methodology",
        "Discovery of significant errors",
        "Changes in emission factors",
    ],
    "description": (
        "Base year emissions are recalculated when structural or methodology "
        "changes result in a cumulative impact exceeding the significance "
        "threshold. Per GHG Protocol Chapter 5."
    ),
}


# ---------------------------------------------------------------------------
# In-Memory Settings Store
# ---------------------------------------------------------------------------

_current_settings: Dict[str, Any] = {
    "gwp_version": "AR5",
    "significance_threshold_pct": 1.0,
    "emission_factor_sources": ["epa", "defra", "ipcc", "ecoinvent"],
    "primary_ef_source": "epa",
    "data_quality_thresholds": {
        "minimum_acceptable_score": 50.0,
        "target_score": 80.0,
        "critical_alert_below": 30.0,
        "primary_data_target_pct": 60.0,
    },
    "reporting_preferences": {
        "currency": "USD",
        "unit_system": "metric",
        "decimal_places": 2,
        "show_biogenic_separately": True,
        "scope2_primary_method": "market_based",
        "fiscal_year_start_month": 1,
    },
    "recalculation_policy": DEFAULT_RECALCULATION_POLICY,
    "updated_at": datetime.utcnow(),
    "updated_by": None,
}


def _now() -> datetime:
    return datetime.utcnow()


def _build_settings_response(settings: Dict[str, Any]) -> SettingsResponse:
    """Construct a SettingsResponse from the settings dict."""
    gwp_version = settings["gwp_version"]
    return SettingsResponse(
        gwp_version=gwp_version,
        gwp_values=GWP_DATA[gwp_version],
        significance_threshold_pct=settings["significance_threshold_pct"],
        emission_factor_sources=settings["emission_factor_sources"],
        primary_ef_source=settings["primary_ef_source"],
        data_quality_thresholds=DataQualityThresholds(**settings["data_quality_thresholds"]),
        reporting_preferences=ReportingPreferences(**settings["reporting_preferences"]),
        recalculation_policy=settings["recalculation_policy"],
        updated_at=settings["updated_at"],
        updated_by=settings.get("updated_by"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=SettingsResponse,
    summary="Get current settings",
    description=(
        "Retrieve current platform settings including GWP version, "
        "significance threshold, emission factor sources, data quality "
        "thresholds, and reporting preferences."
    ),
)
async def get_settings() -> SettingsResponse:
    return _build_settings_response(_current_settings)


@router.put(
    "/",
    response_model=SettingsResponse,
    summary="Update settings",
    description=(
        "Update platform settings. Only provided fields are updated; "
        "omitted fields retain their current values. Changing the GWP version "
        "may trigger base year recalculation if the impact exceeds the "
        "recalculation threshold."
    ),
)
async def update_settings(request: UpdateSettingsRequest) -> SettingsResponse:
    updates = request.model_dump(exclude_unset=True)

    if "gwp_version" in updates:
        version = updates["gwp_version"]
        if version not in GWP_DATA:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid GWP version '{version}'. Must be AR4, AR5, or AR6.",
            )
        _current_settings["gwp_version"] = version

    if "significance_threshold_pct" in updates:
        _current_settings["significance_threshold_pct"] = updates["significance_threshold_pct"]

    if "primary_ef_source" in updates:
        _current_settings["primary_ef_source"] = updates["primary_ef_source"]

    if "emission_factor_sources" in updates:
        _current_settings["emission_factor_sources"] = updates["emission_factor_sources"]

    if "data_quality_thresholds" in updates and updates["data_quality_thresholds"] is not None:
        dq = updates["data_quality_thresholds"]
        if isinstance(dq, dict):
            _current_settings["data_quality_thresholds"].update(dq)
        else:
            _current_settings["data_quality_thresholds"] = dq.model_dump()

    if "reporting_preferences" in updates and updates["reporting_preferences"] is not None:
        rp = updates["reporting_preferences"]
        if isinstance(rp, dict):
            _current_settings["reporting_preferences"].update(rp)
        else:
            _current_settings["reporting_preferences"] = rp.model_dump()

    if "recalculation_policy" in updates and updates["recalculation_policy"] is not None:
        _current_settings["recalculation_policy"] = updates["recalculation_policy"]

    _current_settings["updated_at"] = _now()
    _current_settings["updated_by"] = "api_user"

    return _build_settings_response(_current_settings)


@router.get(
    "/defaults",
    response_model=DefaultSettingsResponse,
    summary="Get default settings",
    description=(
        "Retrieve the platform default settings. Useful as a reference "
        "when resetting or comparing against current configuration."
    ),
)
async def get_default_settings() -> DefaultSettingsResponse:
    return DefaultSettingsResponse(
        gwp_version="AR5",
        gwp_values=GWP_DATA["AR5"],
        significance_threshold_pct=1.0,
        emission_factor_sources=["epa", "defra", "ipcc", "ecoinvent"],
        primary_ef_source="epa",
        data_quality_thresholds=DataQualityThresholds(
            minimum_acceptable_score=50.0,
            target_score=80.0,
            critical_alert_below=30.0,
            primary_data_target_pct=60.0,
        ),
        reporting_preferences=ReportingPreferences(
            currency="USD",
            unit_system="metric",
            decimal_places=2,
            show_biogenic_separately=True,
            scope2_primary_method="market_based",
            fiscal_year_start_month=1,
        ),
        recalculation_policy=DEFAULT_RECALCULATION_POLICY,
        note=(
            "These are the GHG Protocol-recommended defaults. "
            "AR5 GWP values are used as they are the most widely accepted. "
            "EPA emission factors are primary for US-based organizations."
        ),
    )
