# -*- coding: utf-8 -*-
"""
SBTiAlignmentSetupWizard - 6-Step Guided Configuration for PACK-023
=====================================================================

This module implements a 6-step configuration wizard for organisations
setting up the SBTi Alignment Pack. Each step is tailored to the SBTi
Corporate Manual V5.3, Net-Zero Standard V1.3, FLAG Guidance, and
FI Net-Zero (FINZ) V1.0 requirements.

Wizard Steps (6):
    1. organization_profile    -- Name, sector, region, size, FI status,
                                  FLAG exposure, consolidation approach
    2. boundary_selection      -- GHG Protocol boundary, SBTi target boundary
                                  (Scope 1+2+3), equity share rules
    3. scope_configuration     -- Scope 1/2/3 inclusion, Scope 3 materiality
                                  screening (40% trigger), 67%/90% coverage
    4. data_source_setup       -- ERP type, file formats, API connections,
                                  emission factor databases, base year data
    5. target_preferences      -- Pathway (ACA/SDA/FLAG), SDA sector, FLAG
                                  commodities, FI asset classes, ambition
                                  level, base year, target years, net-zero
    6. preset_selection        -- Auto-recommend based on sector, SBTi pathway,
                                  FI status, and FLAG exposure

Sector Presets (10):
    manufacturing_aca, manufacturing_sda, services_aca, technology_aca,
    retail_aca, financial_institution, energy_sda, heavy_industry_sda,
    agriculture_flag, transport_sda

SBTi Requirements Enforced:
    - C1-C2: Commitment letter, public announcement
    - C5-C8: Base year, recalculation policy
    - C9-C11: Scope 1+2 near-term (min -4.2%/yr for 1.5C)
    - C12-C15: Scope 3 materiality + coverage (40% + 67%/90%)
    - C22-C24: Long-term targets (90% S1+S2, 97% total by 2050)
    - NZ-C1 to NZ-C14: Net-zero criteria
    - FLAG: 20% threshold, commodity-level targets
    - FINZ: 8 asset classes, portfolio coverage

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SBTiWizardStep(str, Enum):
    """Names of wizard steps in execution order."""

    ORGANIZATION_PROFILE = "organization_profile"
    BOUNDARY_SELECTION = "boundary_selection"
    SCOPE_CONFIGURATION = "scope_configuration"
    DATA_SOURCE_SETUP = "data_source_setup"
    TARGET_PREFERENCES = "target_preferences"
    PRESET_SELECTION = "preset_selection"

class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches (per SBTi C5)."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

class OrganizationSize(str, Enum):
    """Organization size classification."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

class SBTiPathwayType(str, Enum):
    """SBTi target-setting pathway."""

    ACA = "aca"          # Absolute Contraction Approach (cross-sector)
    SDA = "sda"          # Sectoral Decarbonization Approach (12 sectors)
    FLAG = "flag"        # Forest, Land and Agriculture
    FINZ = "finz"        # Financial Institution Net-Zero

class SBTiAmbitionLevel(str, Enum):
    """SBTi target ambition level."""

    AMBITIOUS_1_5C = "1.5C"
    WELL_BELOW_2C = "well_below_2C"
    TWO_DEGREES = "2C"

class OrganizationType(str, Enum):
    """Organization type for SBTi classification."""

    CORPORATE = "corporate"
    FINANCIAL_INSTITUTION = "financial_institution"
    SME = "sme"

class FLAGExposure(str, Enum):
    """FLAG sector exposure level."""

    NONE = "none"
    BELOW_THRESHOLD = "below_threshold"     # <20% FLAG revenue
    ABOVE_THRESHOLD = "above_threshold"     # >=20% FLAG revenue
    PRIMARILY_FLAG = "primarily_flag"       # >50% FLAG revenue

class EmissionFactorSource(str, Enum):
    """Emission factor database sources."""

    GHG_PROTOCOL = "ghg_protocol"
    IPCC_AR6 = "ipcc_ar6"
    DEFRA = "defra"
    EPA = "epa"
    ECOINVENT = "ecoinvent"
    GLEC = "glec"
    CUSTOM = "custom"

# ---------------------------------------------------------------------------
# SBTi SDA Sectors & FLAG Commodities Reference
# ---------------------------------------------------------------------------

SDA_SECTORS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "intensity_metric": "tCO2e/MWh",
        "2050_benchmark": 0.0,
        "description": "Electricity generation",
        "activity_unit": "MWh",
    },
    "cement": {
        "intensity_metric": "tCO2e/t_cementitious",
        "2050_benchmark": 0.143,
        "description": "Cement and clinker production",
        "activity_unit": "tonnes cementitious",
    },
    "steel": {
        "intensity_metric": "tCO2e/t_steel",
        "2050_benchmark": 0.186,
        "description": "Iron and steel production",
        "activity_unit": "tonnes steel",
    },
    "aluminium": {
        "intensity_metric": "tCO2e/t_aluminium",
        "2050_benchmark": 1.22,
        "description": "Primary aluminium production",
        "activity_unit": "tonnes aluminium",
    },
    "pulp_paper": {
        "intensity_metric": "tCO2e/t_product",
        "2050_benchmark": 0.18,
        "description": "Pulp and paper production",
        "activity_unit": "tonnes product",
    },
    "transport": {
        "intensity_metric": "gCO2e/pkm or gCO2e/tkm",
        "2050_benchmark": 0.0,
        "description": "Passenger and freight transport",
        "activity_unit": "passenger-km or tonne-km",
    },
    "buildings_residential": {
        "intensity_metric": "kgCO2e/m2",
        "2050_benchmark": 0.0,
        "description": "Residential buildings",
        "activity_unit": "m2 floor area",
    },
    "buildings_services": {
        "intensity_metric": "kgCO2e/m2",
        "2050_benchmark": 0.0,
        "description": "Service/commercial buildings",
        "activity_unit": "m2 floor area",
    },
    "chemicals": {
        "intensity_metric": "tCO2e/t_product",
        "2050_benchmark": 0.24,
        "description": "Chemical production",
        "activity_unit": "tonnes product",
    },
    "glass": {
        "intensity_metric": "tCO2e/t_glass",
        "2050_benchmark": 0.25,
        "description": "Glass production",
        "activity_unit": "tonnes glass",
    },
    "textiles": {
        "intensity_metric": "tCO2e/t_textile",
        "2050_benchmark": 0.8,
        "description": "Textile manufacturing",
        "activity_unit": "tonnes textile",
    },
    "food_beverage": {
        "intensity_metric": "tCO2e/t_product",
        "2050_benchmark": 0.3,
        "description": "Food and beverage processing",
        "activity_unit": "tonnes product",
    },
}

FLAG_COMMODITIES: Dict[str, Dict[str, Any]] = {
    "cattle_beef": {
        "ghg_type": "CH4_N2O_CO2",
        "key_sources": ["enteric_fermentation", "manure", "feed_production"],
        "convergence_year": 2050,
    },
    "cattle_dairy": {
        "ghg_type": "CH4_N2O_CO2",
        "key_sources": ["enteric_fermentation", "manure", "feed_production"],
        "convergence_year": 2050,
    },
    "poultry": {
        "ghg_type": "N2O_CO2",
        "key_sources": ["manure", "feed_production"],
        "convergence_year": 2050,
    },
    "pork": {
        "ghg_type": "CH4_N2O_CO2",
        "key_sources": ["enteric_fermentation", "manure", "feed_production"],
        "convergence_year": 2050,
    },
    "rice": {
        "ghg_type": "CH4_N2O",
        "key_sources": ["paddy_cultivation", "fertilizer"],
        "convergence_year": 2050,
    },
    "palm_oil": {
        "ghg_type": "CO2_N2O",
        "key_sources": ["land_use_change", "peatland", "fertilizer"],
        "convergence_year": 2050,
    },
    "soy": {
        "ghg_type": "CO2_N2O",
        "key_sources": ["land_use_change", "fertilizer"],
        "convergence_year": 2050,
    },
    "timber_forestry": {
        "ghg_type": "CO2",
        "key_sources": ["harvesting", "land_use_change", "sequestration"],
        "convergence_year": 2050,
    },
    "cocoa": {
        "ghg_type": "CO2_N2O",
        "key_sources": ["land_use_change", "fertilizer"],
        "convergence_year": 2050,
    },
    "coffee": {
        "ghg_type": "CO2_N2O",
        "key_sources": ["land_use_change", "fertilizer", "processing"],
        "convergence_year": 2050,
    },
    "rubber": {
        "ghg_type": "CO2_N2O",
        "key_sources": ["land_use_change", "fertilizer"],
        "convergence_year": 2050,
    },
}

FINZ_ASSET_CLASSES: Dict[str, Dict[str, Any]] = {
    "listed_equity": {
        "methods": ["portfolio_coverage", "temperature_rating", "sectoral"],
        "min_coverage_pct": 67.0,
        "description": "Publicly listed equity investments",
    },
    "corporate_bonds": {
        "methods": ["portfolio_coverage", "temperature_rating", "sectoral"],
        "min_coverage_pct": 67.0,
        "description": "Corporate bond holdings",
    },
    "project_finance": {
        "methods": ["sectoral", "absolute"],
        "min_coverage_pct": 67.0,
        "description": "Project finance and infrastructure lending",
    },
    "commercial_real_estate": {
        "methods": ["sectoral", "crrem"],
        "min_coverage_pct": 67.0,
        "description": "Commercial real estate portfolio",
    },
    "mortgages": {
        "methods": ["sectoral", "crrem"],
        "min_coverage_pct": 67.0,
        "description": "Residential mortgage portfolio",
    },
    "sovereign_debt": {
        "methods": ["engagement", "sectoral"],
        "min_coverage_pct": 50.0,
        "description": "Sovereign and public sector debt",
    },
    "private_equity": {
        "methods": ["portfolio_coverage", "engagement"],
        "min_coverage_pct": 50.0,
        "description": "Private equity and venture capital",
    },
    "sme_lending": {
        "methods": ["engagement", "portfolio_coverage"],
        "min_coverage_pct": 50.0,
        "description": "SME and small business lending",
    },
}

# ---------------------------------------------------------------------------
# Step Data Models
# ---------------------------------------------------------------------------

class OrganizationProfile(BaseModel):
    """Organization profile from step 1 -- SBTi-specific fields."""

    organization_name: str = Field(..., min_length=1, max_length=255)
    sector: str = Field(default="general")
    sub_sector: str = Field(default="")
    region: str = Field(default="EU")
    country: str = Field(default="DE")
    employee_count: int = Field(default=500, ge=1)
    annual_revenue_eur: float = Field(default=100_000_000.0, ge=0)
    size: OrganizationSize = Field(default=OrganizationSize.MEDIUM)
    is_listed: bool = Field(default=False)
    nace_code: str = Field(default="")
    fiscal_year_end: str = Field(default="12-31")

    # SBTi-specific fields
    organization_type: OrganizationType = Field(
        default=OrganizationType.CORPORATE,
        description="SBTi classification: corporate, FI, or SME",
    )
    is_financial_institution: bool = Field(
        default=False,
        description="Whether to enable FINZ V1.0 module",
    )
    flag_exposure: FLAGExposure = Field(
        default=FLAGExposure.NONE,
        description="FLAG sector revenue exposure level",
    )
    flag_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of revenue from FLAG-related activities",
    )
    sbti_commitment_status: str = Field(
        default="not_committed",
        description="Current SBTi commitment status: not_committed, committed, targets_set, validated",
    )
    sbti_commitment_date: Optional[str] = Field(
        None,
        description="Date of SBTi commitment letter (ISO format)",
    )
    multi_entity: bool = Field(default=False)
    entity_count: int = Field(default=1, ge=1)
    parent_company: str = Field(
        default="",
        description="Parent company name if subsidiary",
    )

class BoundarySelection(BaseModel):
    """Boundary selection from step 2 -- SBTi target boundary rules."""

    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="GHG Protocol consolidation approach (SBTi C5)",
    )
    include_subsidiaries: bool = Field(default=True)
    subsidiary_count: int = Field(default=0, ge=0)
    joint_ventures: int = Field(default=0, ge=0)
    equity_share_threshold_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
    )
    countries_of_operation: List[str] = Field(
        default_factory=lambda: ["DE"],
    )

    # SBTi target boundary (must include S1+S2, S3 if material)
    include_scope_1_in_target: bool = Field(
        default=True,
        description="SBTi requires S1 in target boundary (C9)",
    )
    include_scope_2_in_target: bool = Field(
        default=True,
        description="SBTi requires S2 in target boundary (C10)",
    )
    include_scope_3_in_target: bool = Field(
        default=True,
        description="SBTi requires S3 if >40% of total (C12-C15)",
    )
    biogenic_emissions_separate: bool = Field(
        default=True,
        description="Report biogenic CO2 separately per GHG Protocol",
    )
    exclusion_justification: str = Field(
        default="",
        description="Justification if any sources excluded from boundary",
    )
    max_exclusion_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Max % of total emissions that can be excluded (SBTi <5%)",
    )

class ScopeConfiguration(BaseModel):
    """Scope configuration from step 3 -- SBTi materiality & coverage."""

    include_scope_1: bool = Field(default=True)
    include_scope_2: bool = Field(default=True)
    include_scope_3: bool = Field(default=True)
    scope2_methods: List[str] = Field(
        default_factory=lambda: ["location_based", "market_based"],
    )
    scope3_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7],
        description="Scope 3 category numbers (1-15)",
    )
    scope1_sources: List[str] = Field(
        default_factory=lambda: [
            "stationary_combustion", "mobile_combustion",
        ],
    )

    # SBTi Scope 3 materiality screening (C12-C15)
    scope3_screening_done: bool = Field(
        default=False,
        description="Whether Scope 3 screening has been completed",
    )
    scope3_total_pct_of_emissions: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Scope 3 as % of total emissions (40% trigger for targets)",
    )
    scope3_material_categories: List[int] = Field(
        default_factory=list,
        description="Categories identified as material in screening",
    )
    scope3_coverage_target_pct: float = Field(
        default=67.0, ge=0.0, le=100.0,
        description="Required Scope 3 coverage (67% for near-term, 90% for long-term)",
    )
    scope3_requires_target: bool = Field(
        default=True,
        description="Whether Scope 3 target required (>40% of total or material)",
    )

    # FLAG-specific scope items
    flag_in_scope: bool = Field(
        default=False,
        description="Whether FLAG emissions included in scope",
    )
    flag_commodities_in_scope: List[str] = Field(
        default_factory=list,
        description="FLAG commodity names included in scope",
    )

class DataSourceSetup(BaseModel):
    """Data source setup from step 4 -- SBTi data requirements."""

    erp_system: str = Field(default="none")
    erp_connected: bool = Field(default=False)
    file_formats: List[str] = Field(
        default_factory=lambda: ["excel", "csv"],
    )
    api_connections: List[str] = Field(default_factory=list)
    utility_provider_apis: bool = Field(default=False)
    travel_management_system: str = Field(default="none")
    procurement_system: str = Field(default="none")

    # SBTi-specific data fields
    base_year: int = Field(
        default=2019, ge=2015, le=2025,
        description="GHG inventory base year (SBTi C5-C6)",
    )
    base_year_data_available: bool = Field(
        default=False,
        description="Whether complete base year inventory is available",
    )
    base_year_total_tco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total base year emissions in tCO2e (if known)",
    )
    emission_factor_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.GHG_PROTOCOL,
        description="Primary emission factor database",
    )
    secondary_ef_sources: List[str] = Field(
        default_factory=list,
        description="Additional emission factor sources",
    )
    data_quality_score: float = Field(
        default=0.0, ge=0.0, le=5.0,
        description="Overall data quality score (1-5 scale, GHG Protocol)",
    )
    historical_years_available: int = Field(
        default=1, ge=1, le=10,
        description="Number of years of historical data available",
    )
    recalculation_policy_defined: bool = Field(
        default=False,
        description="Whether base year recalculation policy is in place (SBTi C28)",
    )
    recalculation_trigger_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Recalculation trigger threshold (% change, typically 5%)",
    )

    # FI-specific data
    portfolio_data_available: bool = Field(
        default=False,
        description="Whether portfolio-level emissions data is available (FI)",
    )
    asset_classes_with_data: List[str] = Field(
        default_factory=list,
        description="Asset classes with emissions data (FI)",
    )

class TargetPreferences(BaseModel):
    """Target preferences from step 5 -- SBTi pathway & target config."""

    # Core SBTi target parameters
    ambition_level: SBTiAmbitionLevel = Field(
        default=SBTiAmbitionLevel.AMBITIOUS_1_5C,
        description="SBTi ambition: 1.5C (min -4.2%/yr), WB2C (-2.5%/yr), 2C",
    )
    primary_pathway: SBTiPathwayType = Field(
        default=SBTiPathwayType.ACA,
        description="Primary target-setting pathway",
    )
    base_year: int = Field(
        default=2019, ge=2015, le=2025,
        description="Target base year",
    )
    near_term_target_year: int = Field(
        default=2030, ge=2025, le=2035,
        description="Near-term target year (5-10 years from submission)",
    )
    long_term_target_year: int = Field(
        default=2050, ge=2040, le=2060,
        description="Long-term target year (no later than 2050)",
    )
    net_zero_target_year: int = Field(
        default=2050, ge=2040, le=2060,
        description="Net-zero target year (no later than 2050)",
    )

    # Near-term reduction targets (SBTi C9-C15)
    scope1_scope2_near_term_pct: float = Field(
        default=42.0, ge=0.0, le=100.0,
        description="Near-term S1+S2 absolute reduction % (min 42% for 1.5C)",
    )
    scope3_near_term_pct: float = Field(
        default=25.0, ge=0.0, le=100.0,
        description="Near-term S3 reduction % (if material)",
    )

    # Long-term / Net-zero targets (SBTi C22-C24, NZ-C)
    scope1_scope2_long_term_pct: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="Long-term S1+S2 reduction % (min 90%)",
    )
    total_long_term_pct: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="Long-term total reduction % (min 90% for 1.5C, 97% all scopes)",
    )
    residual_emissions_max_pct: float = Field(
        default=10.0, ge=0.0, le=100.0,
        description="Max residual emissions at net-zero (<=10%)",
    )

    # SDA-specific preferences
    sda_sector: str = Field(
        default="",
        description="SDA sector key if using SDA pathway",
    )
    sda_intensity_metric: str = Field(
        default="",
        description="SDA intensity metric (e.g., tCO2e/MWh)",
    )
    sda_current_intensity: float = Field(
        default=0.0, ge=0.0,
        description="Current emission intensity",
    )

    # FLAG-specific preferences
    flag_target_required: bool = Field(
        default=False,
        description="Whether FLAG target is required (>=20% FLAG revenue)",
    )
    flag_commodities: List[str] = Field(
        default_factory=list,
        description="FLAG commodities for which targets are set",
    )
    flag_no_deforestation_commitment: bool = Field(
        default=False,
        description="Commitment to zero deforestation by 2025 (FLAG req)",
    )

    # FI-specific preferences (FINZ V1.0)
    fi_target_enabled: bool = Field(
        default=False,
        description="Whether FI portfolio targets are enabled",
    )
    fi_asset_classes: List[str] = Field(
        default_factory=list,
        description="FI asset classes for portfolio targets",
    )
    fi_portfolio_coverage_pct: float = Field(
        default=67.0, ge=0.0, le=100.0,
        description="FI portfolio coverage target %",
    )

    # Net-zero specific
    include_neutralization: bool = Field(
        default=True,
        description="Include neutralization via carbon removals for residual",
    )
    include_bvcm: bool = Field(
        default=False,
        description="Include Beyond Value Chain Mitigation (recommended)",
    )
    sbti_submission_planned: bool = Field(
        default=True,
        description="Planning to submit targets to SBTi for validation",
    )
    planned_submission_date: str = Field(
        default="",
        description="Planned SBTi submission date (ISO format)",
    )

class PresetSelection(BaseModel):
    """Preset selection from step 6 -- SBTi-tailored preset."""

    preset_name: str = Field(default="")
    preset_applied: bool = Field(default=False)
    engines_enabled: List[str] = Field(default_factory=list)
    scope3_priority: List[int] = Field(default_factory=list)
    recommended_levers: List[str] = Field(default_factory=list)
    pathway: str = Field(default="aca")
    sda_enabled: bool = Field(default=False)
    flag_enabled: bool = Field(default=False)
    fi_enabled: bool = Field(default=False)
    neutralization_enabled: bool = Field(default=True)
    near_term_min_reduction_pct: float = Field(default=42.0)
    sbti_criteria_applicable: List[str] = Field(
        default_factory=lambda: ["C1-C28"],
    )

# ---------------------------------------------------------------------------
# Wizard State Models
# ---------------------------------------------------------------------------

class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: SBTiWizardStep = Field(...)
    display_name: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    sbti_warnings: List[str] = Field(
        default_factory=list,
        description="SBTi-specific warnings (non-blocking)",
    )
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    execution_time_ms: float = Field(default=0.0)

class WizardState(BaseModel):
    """Complete state of the SBTi alignment setup wizard."""

    wizard_id: str = Field(default="")
    pack_id: str = Field(default="PACK-023")
    current_step: SBTiWizardStep = Field(
        default=SBTiWizardStep.ORGANIZATION_PROFILE,
    )
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    org_profile: Optional[OrganizationProfile] = Field(None)
    boundary: Optional[BoundarySelection] = Field(None)
    scope_config: Optional[ScopeConfiguration] = Field(None)
    data_sources: Optional[DataSourceSetup] = Field(None)
    target_prefs: Optional[TargetPreferences] = Field(None)
    preset: Optional[PresetSelection] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)

    # SBTi readiness indicators computed during wizard
    sbti_readiness_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Estimated SBTi submission readiness (0-100%)",
    )
    sbti_criteria_gaps: List[str] = Field(
        default_factory=list,
        description="SBTi criteria with identified gaps",
    )

class SetupResult(BaseModel):
    """Final setup result with SBTi-aligned configuration."""

    result_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-023")
    organization_name: str = Field(default="")
    sector: str = Field(default="")
    organization_type: str = Field(default="corporate")
    is_financial_institution: bool = Field(default=False)
    flag_exposure: str = Field(default="none")
    consolidation_approach: str = Field(default="")
    multi_entity: bool = Field(default=False)
    entity_count: int = Field(default=1)

    # Scopes
    scopes_included: List[str] = Field(default_factory=list)
    scope3_categories: List[int] = Field(default_factory=list)
    scope3_coverage_pct: float = Field(default=67.0)
    scope3_requires_target: bool = Field(default=True)

    # Pathway & targets
    primary_pathway: str = Field(default="aca")
    ambition_level: str = Field(default="1.5C")
    base_year: int = Field(default=2019)
    near_term_target_year: int = Field(default=2030)
    long_term_target_year: int = Field(default=2050)
    net_zero_target_year: int = Field(default=2050)
    scope1_scope2_near_term_pct: float = Field(default=42.0)
    scope3_near_term_pct: float = Field(default=25.0)
    scope1_scope2_long_term_pct: float = Field(default=90.0)

    # SDA
    is_sda_sector: bool = Field(default=False)
    sda_sector: str = Field(default="")
    sda_intensity_metric: str = Field(default="")

    # FLAG
    flag_target_required: bool = Field(default=False)
    flag_commodities: List[str] = Field(default_factory=list)

    # FI (FINZ)
    fi_target_enabled: bool = Field(default=False)
    fi_asset_classes: List[str] = Field(default_factory=list)

    # Engines & levers
    engines_enabled: List[str] = Field(default_factory=list)
    recommended_levers: List[str] = Field(default_factory=list)

    # SBTi criteria
    sbti_criteria_applicable: List[str] = Field(default_factory=list)
    sbti_readiness_score: float = Field(default=0.0)
    sbti_criteria_gaps: List[str] = Field(default_factory=list)

    # Metadata
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=6)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[SBTiWizardStep] = [
    SBTiWizardStep.ORGANIZATION_PROFILE,
    SBTiWizardStep.BOUNDARY_SELECTION,
    SBTiWizardStep.SCOPE_CONFIGURATION,
    SBTiWizardStep.DATA_SOURCE_SETUP,
    SBTiWizardStep.TARGET_PREFERENCES,
    SBTiWizardStep.PRESET_SELECTION,
]

STEP_DISPLAY_NAMES: Dict[SBTiWizardStep, str] = {
    SBTiWizardStep.ORGANIZATION_PROFILE: "Organization Profile & SBTi Classification",
    SBTiWizardStep.BOUNDARY_SELECTION: "GHG Boundary & SBTi Target Boundary",
    SBTiWizardStep.SCOPE_CONFIGURATION: "Scope Configuration & Materiality Screening",
    SBTiWizardStep.DATA_SOURCE_SETUP: "Data Sources & Base Year Setup",
    SBTiWizardStep.TARGET_PREFERENCES: "SBTi Pathway & Target Preferences",
    SBTiWizardStep.PRESET_SELECTION: "SBTi Sector Preset Selection",
}

# ---------------------------------------------------------------------------
# Sector Presets (10) -- SBTi-specific
# ---------------------------------------------------------------------------

SECTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "manufacturing_aca": {
        "description": "Manufacturing using Absolute Contraction Approach",
        "pathway": "aca",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "target_definition_engine", "criteria_validation_engine",
            "readiness_assessment_engine", "cross_framework_engine",
            "sda_sector_engine", "submission_readiness_engine",
        ],
        "scope3_priority": [1, 2, 3, 4, 5, 6, 7, 9, 12],
        "recommended_levers": [
            "energy_efficiency", "renewable_energy", "electrification",
            "fuel_switching", "process_innovation", "supplier_engagement",
        ],
        "sda_enabled": False,
        "flag_enabled": False,
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28"],
    },
    "manufacturing_sda": {
        "description": "Manufacturing in SDA-eligible sector (intensity pathway)",
        "pathway": "sda",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "sda_sector_engine", "target_definition_engine",
            "criteria_validation_engine", "readiness_assessment_engine",
            "cross_framework_engine", "submission_readiness_engine",
        ],
        "scope3_priority": [1, 2, 3, 4, 5, 6, 7, 9, 12],
        "recommended_levers": [
            "energy_efficiency", "fuel_switching", "process_innovation",
            "ccus", "waste_heat_recovery", "electrification",
            "supplier_engagement", "circular_economy",
        ],
        "sda_enabled": True,
        "sda_sectors": ["cement", "steel", "aluminium", "pulp_paper",
                        "chemicals", "glass"],
        "flag_enabled": False,
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28"],
    },
    "services_aca": {
        "description": "Services sector using Absolute Contraction Approach",
        "pathway": "aca",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "target_definition_engine", "criteria_validation_engine",
            "readiness_assessment_engine", "cross_framework_engine",
            "submission_readiness_engine",
        ],
        "scope3_priority": [1, 3, 5, 6, 7, 8],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "building_decarbonisation",
            "fleet_decarbonisation", "green_procurement", "remote_work",
        ],
        "sda_enabled": False,
        "flag_enabled": False,
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28"],
    },
    "technology_aca": {
        "description": "Technology sector using Absolute Contraction Approach",
        "pathway": "aca",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "target_definition_engine", "criteria_validation_engine",
            "readiness_assessment_engine", "cross_framework_engine",
            "submission_readiness_engine",
        ],
        "scope3_priority": [1, 2, 3, 6, 7, 11],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "supplier_engagement",
            "green_procurement", "data_center_efficiency", "product_lifecycle",
        ],
        "sda_enabled": False,
        "flag_enabled": False,
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28"],
    },
    "retail_aca": {
        "description": "Retail sector using Absolute Contraction Approach",
        "pathway": "aca",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "target_definition_engine", "criteria_validation_engine",
            "readiness_assessment_engine", "cross_framework_engine",
            "submission_readiness_engine",
        ],
        "scope3_priority": [1, 4, 5, 7, 9, 12],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "supplier_engagement",
            "fleet_decarbonisation", "green_procurement", "refrigerant_management",
        ],
        "sda_enabled": False,
        "flag_enabled": False,
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28"],
    },
    "financial_institution": {
        "description": "Financial institution using FINZ V1.0 portfolio targets",
        "pathway": "finz",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "fi_portfolio_engine", "target_definition_engine",
            "criteria_validation_engine", "readiness_assessment_engine",
            "cross_framework_engine", "submission_readiness_engine",
        ],
        "scope3_priority": [1, 6, 7, 15],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "building_decarbonisation",
            "green_procurement", "portfolio_decarbonisation",
            "engagement_escalation", "fossil_fuel_exclusion",
        ],
        "sda_enabled": False,
        "flag_enabled": False,
        "fi_enabled": True,
        "fi_asset_classes": ["listed_equity", "corporate_bonds",
                             "project_finance", "commercial_real_estate"],
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28", "FINZ-1 to FINZ-8"],
    },
    "energy_sda": {
        "description": "Energy/power sector using SDA intensity pathway",
        "pathway": "sda",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "sda_sector_engine", "target_definition_engine",
            "criteria_validation_engine", "readiness_assessment_engine",
            "cross_framework_engine", "submission_readiness_engine",
        ],
        "scope3_priority": [1, 3, 4, 9, 10, 11],
        "recommended_levers": [
            "renewable_energy", "ccus", "fuel_switching",
            "electrification", "process_innovation", "energy_efficiency",
            "methane_abatement", "hydrogen",
        ],
        "sda_enabled": True,
        "sda_sectors": ["power_generation"],
        "flag_enabled": False,
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28"],
    },
    "heavy_industry_sda": {
        "description": "Heavy industry (cement/steel/aluminium) using SDA",
        "pathway": "sda",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "sda_sector_engine", "target_definition_engine",
            "criteria_validation_engine", "readiness_assessment_engine",
            "cross_framework_engine", "submission_readiness_engine",
        ],
        "scope3_priority": [1, 2, 3, 4, 5, 9],
        "recommended_levers": [
            "fuel_switching", "electrification", "ccus",
            "process_innovation", "waste_heat_recovery", "energy_efficiency",
            "hydrogen", "circular_economy",
        ],
        "sda_enabled": True,
        "sda_sectors": ["cement", "steel", "aluminium"],
        "flag_enabled": False,
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28"],
    },
    "agriculture_flag": {
        "description": "Agriculture/forestry sector with FLAG targets required",
        "pathway": "flag",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "flag_assessment_engine", "target_definition_engine",
            "criteria_validation_engine", "readiness_assessment_engine",
            "cross_framework_engine", "submission_readiness_engine",
        ],
        "scope3_priority": [1, 3, 4, 5, 10, 12],
        "recommended_levers": [
            "no_deforestation", "sustainable_agriculture",
            "land_restoration", "livestock_management",
            "feed_optimisation", "fertilizer_reduction",
            "agroforestry", "soil_carbon_sequestration",
        ],
        "sda_enabled": False,
        "flag_enabled": True,
        "flag_commodities": ["cattle_beef", "cattle_dairy", "palm_oil",
                             "soy", "timber_forestry"],
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28", "FLAG-1 to FLAG-8"],
    },
    "transport_sda": {
        "description": "Transport sector using SDA intensity pathway",
        "pathway": "sda",
        "engines": [
            "commitment_engine", "inventory_compilation_engine",
            "scope3_screening_engine", "pathway_modelling_engine",
            "sda_sector_engine", "target_definition_engine",
            "criteria_validation_engine", "readiness_assessment_engine",
            "cross_framework_engine", "submission_readiness_engine",
        ],
        "scope3_priority": [1, 3, 4, 9, 11],
        "recommended_levers": [
            "fleet_decarbonisation", "electrification", "fuel_switching",
            "route_optimisation", "modal_shift", "energy_efficiency",
            "sustainable_aviation_fuel", "hydrogen",
        ],
        "sda_enabled": True,
        "sda_sectors": ["transport"],
        "flag_enabled": False,
        "fi_enabled": False,
        "near_term_min_reduction_pct": 42.0,
        "sbti_criteria": ["C1-C28"],
    },
}

# ---------------------------------------------------------------------------
# SBTi Criteria Reference (for readiness assessment)
# ---------------------------------------------------------------------------

SBTI_NEAR_TERM_CRITERIA: Dict[str, str] = {
    "C1": "Commitment letter submitted to SBTi",
    "C2": "Public announcement of commitment",
    "C3": "24-month target development timeline",
    "C4": "Commitment to all GHG Protocol scopes",
    "C5": "Base year within 2 years of most recent inventory",
    "C6": "Inventory quality meets GHG Protocol standards",
    "C7": "Scope 1+2 inventory is complete and verified",
    "C8": "Scope 3 screening completed for all 15 categories",
    "C9": "Scope 1+2 near-term target: min 4.2%/yr linear for 1.5C",
    "C10": "Scope 2 target may be separate if predominantly electricity",
    "C11": "Target boundary covers minimum 95% of S1+S2 emissions",
    "C12": "Scope 3 target required if S3 >40% of total emissions",
    "C13": "Scope 3 near-term covers minimum 67% of S3 emissions",
    "C14": "Scope 3 target uses approved methods (absolute or intensity)",
    "C15": "Scope 3 timeline: 5-10 years from submission date",
    "C16": "No use of offsets toward target achievement",
    "C17": "Renewable energy procurement covers S2 where applicable",
    "C18": "Target boundary exclusions justified and <5%",
    "C19": "Reporting commitment: annual GHG inventory disclosure",
    "C20": "Board-level governance of climate targets",
    "C21": "Progress reporting against targets annually",
    "C22": "Long-term target: min 90% S1+S2 reduction by 2050",
    "C23": "Long-term target: min 90% total reduction for 1.5C",
    "C24": "Net-zero by 2050 with neutralization of residual emissions",
    "C25": "Neutralization via high-quality carbon removals only",
    "C26": "Residual emissions <=10% of base year",
    "C27": "BVCM recommended for beyond value chain mitigation",
    "C28": "Base year recalculation policy with 5% significance trigger",
}

SBTI_NET_ZERO_CRITERIA: Dict[str, str] = {
    "NZ-C1": "Long-term science-based target set covering all scopes",
    "NZ-C2": "Near-term science-based target validated by SBTi",
    "NZ-C3": "Emissions reduced by minimum 90% from base year",
    "NZ-C4": "Residual emissions neutralized with carbon removals",
    "NZ-C5": "Removals are permanent (>100 year durability)",
    "NZ-C6": "Removals verified by independent third party",
    "NZ-C7": "BVCM investments recommended at scale",
    "NZ-C8": "No double counting of removal credits",
    "NZ-C9": "Annual progress reporting on all targets",
    "NZ-C10": "Target recalculation when significant changes occur",
    "NZ-C11": "Transition plan published and regularly updated",
    "NZ-C12": "Just transition considerations integrated",
    "NZ-C13": "Scope 3 engagement with key value chain partners",
    "NZ-C14": "Governance structure supports long-term targets",
}

# ---------------------------------------------------------------------------
# SBTiAlignmentSetupWizard
# ---------------------------------------------------------------------------

class SBTiAlignmentSetupWizard:
    """6-step guided configuration wizard for PACK-023 SBTi Alignment.

    Guides organisations through SBTi-specific setup including pathway
    selection (ACA/SDA/FLAG/FINZ), sector classification, FLAG commodity
    assessment, FI portfolio configuration, and 10 sector presets that
    auto-configure engines, criteria, and recommended levers.

    Example:
        >>> wizard = SBTiAlignmentSetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("organization_profile", {...})
        >>> state = wizard.complete_step("boundary_selection", {...})
        >>> state = wizard.complete_step("scope_configuration", {...})
        >>> state = wizard.complete_step("data_source_setup", {...})
        >>> state = wizard.complete_step("target_preferences", {...})
        >>> state = wizard.complete_step("preset_selection", {...})
        >>> result = wizard.generate_config()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the SBTi Alignment Setup Wizard.

        Args:
            config: Optional configuration overrides.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            SBTiWizardStep.ORGANIZATION_PROFILE: self._handle_org_profile,
            SBTiWizardStep.BOUNDARY_SELECTION: self._handle_boundary,
            SBTiWizardStep.SCOPE_CONFIGURATION: self._handle_scope_config,
            SBTiWizardStep.DATA_SOURCE_SETUP: self._handle_data_sources,
            SBTiWizardStep.TARGET_PREFERENCES: self._handle_target_prefs,
            SBTiWizardStep.PRESET_SELECTION: self._handle_preset,
        }
        self.logger.info("SBTiAlignmentSetupWizard initialized")

    # ----- Public Methods -----

    def start(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initial WizardState with all steps in PENDING status.
        """
        wizard_id = _compute_hash(
            f"sbti-wizard:{utcnow().isoformat()}"
        )[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(
            wizard_id=wizard_id,
            current_step=STEP_ORDER[0],
            steps=steps,
        )
        self.logger.info("SBTi alignment wizard started: %s", wizard_id)
        return self._state

    def complete_step(
        self, step_name: str, data: Dict[str, Any],
    ) -> WizardState:
        """Complete a wizard step with provided data.

        Args:
            step_name: Step name to complete (e.g., 'organization_profile').
            data: Step configuration data.

        Returns:
            Updated WizardState.

        Raises:
            RuntimeError: If wizard not started.
            ValueError: If step name invalid.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first -- call start()")

        try:
            step_enum = SBTiWizardStep(step_name)
        except ValueError:
            valid = [s.value for s in SBTiWizardStep]
            raise ValueError(f"Unknown step '{step_name}'. Valid: {valid}")

        step = self._state.steps.get(step_name)
        if step is None:
            raise ValueError(f"Step '{step_name}' not found in wizard state")

        step.status = StepStatus.IN_PROGRESS
        step.started_at = utcnow()
        start_time = time.monotonic()

        handler = self._step_handlers.get(step_enum)
        if handler is None:
            raise ValueError(f"No handler registered for step '{step_name}'")

        try:
            errors, warnings = handler(data)
            elapsed = (time.monotonic() - start_time) * 1000
            step.execution_time_ms = elapsed
            step.data = data

            if errors:
                step.status = StepStatus.FAILED
                step.validation_errors = errors
                step.sbti_warnings = warnings
            else:
                step.status = StepStatus.COMPLETED
                step.completed_at = utcnow()
                step.validation_errors = []
                step.sbti_warnings = warnings
                self._advance_step(step_enum)
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.validation_errors = [str(exc)]
            step.execution_time_ms = (time.monotonic() - start_time) * 1000

        return self._state

    def generate_config(self) -> SetupResult:
        """Generate the final PACK-023 configuration from wizard state.

        Returns:
            SetupResult with SBTi-aligned configuration.
        """
        return self._generate_result()

    def run_demo(self, demo_type: str = "corporate_aca") -> SetupResult:
        """Execute a pre-configured demo setup.

        Args:
            demo_type: Demo scenario to run. Options:
                - 'corporate_aca': Manufacturing using ACA (default)
                - 'corporate_sda': Cement company using SDA
                - 'financial_institution': Bank using FINZ
                - 'agriculture_flag': Agribusiness with FLAG targets

        Returns:
            SetupResult with demo configuration.
        """
        self.start()

        demos = {
            "corporate_aca": self._demo_corporate_aca,
            "corporate_sda": self._demo_corporate_sda,
            "financial_institution": self._demo_financial_institution,
            "agriculture_flag": self._demo_agriculture_flag,
        }

        demo_fn = demos.get(demo_type, self._demo_corporate_aca)
        demo_steps = demo_fn()

        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)

        return self._generate_result()

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    def get_sector_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get the preset configuration for a sector.

        Args:
            preset_name: Preset key (e.g., 'manufacturing_aca').

        Returns:
            Preset configuration dict, or None if not found.
        """
        return SECTOR_PRESETS.get(preset_name)

    def list_presets(self) -> Dict[str, str]:
        """List all available presets with descriptions.

        Returns:
            Dict mapping preset name to description.
        """
        return {
            k: v.get("description", k) for k, v in SECTOR_PRESETS.items()
        }

    def get_sda_sectors(self) -> Dict[str, Dict[str, Any]]:
        """Return all SDA sector definitions with benchmarks.

        Returns:
            Dict of SDA sectors with intensity metrics and benchmarks.
        """
        return dict(SDA_SECTORS)

    def get_flag_commodities(self) -> Dict[str, Dict[str, Any]]:
        """Return all FLAG commodity definitions.

        Returns:
            Dict of FLAG commodities with GHG types and sources.
        """
        return dict(FLAG_COMMODITIES)

    def get_finz_asset_classes(self) -> Dict[str, Dict[str, Any]]:
        """Return all FINZ asset class definitions.

        Returns:
            Dict of FI asset classes with target-setting methods.
        """
        return dict(FINZ_ASSET_CLASSES)

    def get_sbti_criteria(self) -> Dict[str, Dict[str, str]]:
        """Return SBTi criteria reference.

        Returns:
            Dict with 'near_term' and 'net_zero' criteria dictionaries.
        """
        return {
            "near_term": dict(SBTI_NEAR_TERM_CRITERIA),
            "net_zero": dict(SBTI_NET_ZERO_CRITERIA),
        }

    def recommend_preset(
        self,
        sector: str,
        is_fi: bool = False,
        flag_exposure: str = "none",
        is_sda: bool = False,
        size: str = "medium",
    ) -> Dict[str, Any]:
        """Recommend a preset based on organization characteristics.

        Args:
            sector: Organization sector.
            is_fi: Whether the organization is a financial institution.
            flag_exposure: FLAG exposure level.
            is_sda: Whether the organization is in an SDA-eligible sector.
            size: Organization size.

        Returns:
            Recommended preset with justification and SBTi criteria.
        """
        # Priority: FI > FLAG > SDA > ACA
        if is_fi:
            preset_name = "financial_institution"
        elif flag_exposure in ("above_threshold", "primarily_flag"):
            preset_name = "agriculture_flag"
        elif is_sda:
            sector_sda_map = {
                "energy": "energy_sda",
                "power": "energy_sda",
                "cement": "heavy_industry_sda",
                "steel": "heavy_industry_sda",
                "aluminium": "heavy_industry_sda",
                "transport": "transport_sda",
                "logistics": "transport_sda",
                "manufacturing": "manufacturing_sda",
                "chemicals": "manufacturing_sda",
            }
            preset_name = sector_sda_map.get(sector, "manufacturing_sda")
        else:
            sector_aca_map = {
                "manufacturing": "manufacturing_aca",
                "services": "services_aca",
                "technology": "technology_aca",
                "retail": "retail_aca",
                "financial_services": "financial_institution",
            }
            preset_name = sector_aca_map.get(sector, "services_aca")

        preset = SECTOR_PRESETS.get(
            preset_name, SECTOR_PRESETS["services_aca"],
        )

        recommendation: Dict[str, Any] = {
            "recommended_preset": preset_name,
            "preset": preset,
            "pathway": preset.get("pathway", "aca"),
            "justification": (
                f"Selected '{preset_name}' based on sector='{sector}', "
                f"FI={is_fi}, FLAG='{flag_exposure}', SDA={is_sda}, "
                f"size='{size}'"
            ),
            "sbti_criteria": preset.get("sbti_criteria", ["C1-C28"]),
        }

        # Size-based adjustments
        if size in ("small", "medium"):
            recommendation["scope3_adjustment"] = (
                "Consider fewer Scope 3 categories for initial assessment. "
                "SMEs may qualify for simplified SBTi pathway."
            )
            recommendation["simplified_scope3"] = (
                preset.get("scope3_priority", [])[:5]
            )

        return recommendation

    def assess_sbti_readiness(self) -> Dict[str, Any]:
        """Assess SBTi submission readiness based on wizard state.

        Returns:
            Readiness assessment with score, gaps, and recommendations.
        """
        if self._state is None:
            return {
                "readiness_score": 0.0,
                "status": "not_started",
                "gaps": ["Wizard not started"],
                "recommendations": ["Start the setup wizard"],
            }

        score = 0.0
        gaps: List[str] = []
        recs: List[str] = []
        max_score = 100.0

        # Check commitment (C1-C2) -- 10 points
        if self._state.org_profile:
            status = self._state.org_profile.sbti_commitment_status
            if status == "validated":
                score += 10.0
            elif status == "targets_set":
                score += 8.0
            elif status == "committed":
                score += 5.0
            else:
                gaps.append("C1-C2: SBTi commitment not yet made")
                recs.append("Submit commitment letter to SBTi")

        # Check boundary (C5, C11, C18) -- 10 points
        if self._state.boundary:
            score += 5.0
            if self._state.boundary.max_exclusion_pct <= 5.0:
                score += 5.0
            else:
                gaps.append("C18: Boundary exclusions exceed 5%")
                recs.append("Reduce emission source exclusions to <5%")
        else:
            gaps.append("C5: Boundary not configured")

        # Check scope configuration (C7-C8, C12-C15) -- 20 points
        if self._state.scope_config:
            if self._state.scope_config.include_scope_1:
                score += 5.0
            else:
                gaps.append("C7: Scope 1 not included")
            if self._state.scope_config.include_scope_2:
                score += 5.0
            else:
                gaps.append("C7: Scope 2 not included")
            if self._state.scope_config.scope3_screening_done:
                score += 5.0
            else:
                gaps.append("C8: Scope 3 screening not completed")
                recs.append("Complete Scope 3 15-category screening")
            if (self._state.scope_config.scope3_requires_target and
                    self._state.scope_config.include_scope_3):
                score += 5.0
            elif self._state.scope_config.scope3_requires_target:
                gaps.append("C12: Scope 3 target required but S3 not in scope")
        else:
            gaps.append("C7-C8: Scope configuration missing")

        # Check data sources (C5-C6) -- 15 points
        if self._state.data_sources:
            if self._state.data_sources.base_year_data_available:
                score += 8.0
            else:
                gaps.append("C5-C6: Base year inventory data not available")
                recs.append("Complete base year GHG inventory")
            if self._state.data_sources.recalculation_policy_defined:
                score += 4.0
            else:
                gaps.append("C28: Recalculation policy not defined")
                recs.append("Define base year recalculation policy (5% trigger)")
            if self._state.data_sources.data_quality_score >= 3.0:
                score += 3.0
            else:
                gaps.append("C6: Data quality below acceptable threshold")
                recs.append("Improve data quality to GHG Protocol standard")
        else:
            gaps.append("C5-C6: Data source configuration missing")

        # Check target preferences (C9-C15, C22-C24) -- 30 points
        if self._state.target_prefs:
            # Near-term S1+S2 check
            if self._state.target_prefs.scope1_scope2_near_term_pct >= 42.0:
                score += 10.0
            else:
                gaps.append(
                    f"C9: S1+S2 near-term reduction {self._state.target_prefs.scope1_scope2_near_term_pct}% "
                    f"< 42% minimum for 1.5C"
                )
                recs.append("Increase S1+S2 near-term target to at least 42%")

            # Long-term check
            if self._state.target_prefs.scope1_scope2_long_term_pct >= 90.0:
                score += 10.0
            else:
                gaps.append("C22: Long-term S1+S2 target < 90%")

            # Target timeline
            base = self._state.target_prefs.base_year
            nt = self._state.target_prefs.near_term_target_year
            lt = self._state.target_prefs.long_term_target_year
            if 5 <= (nt - base) <= 15 and lt <= 2050:
                score += 5.0
            else:
                gaps.append("C15: Target timeline does not meet SBTi requirements")

            # Neutralization / net-zero
            if self._state.target_prefs.include_neutralization:
                score += 5.0
            else:
                gaps.append("NZ-C4: Neutralization strategy not configured")
                recs.append("Configure carbon removal neutralization for residual emissions")
        else:
            gaps.append("C9-C24: Target preferences not configured")

        # Check preset selection -- 15 points
        if self._state.preset:
            score += 10.0
            if self._state.preset.preset_applied:
                score += 5.0
        else:
            gaps.append("Sector preset not applied")
            recs.append("Select and apply a sector preset")

        # Clamp score
        readiness_pct = min(score, max_score)

        # Store in state
        self._state.sbti_readiness_score = readiness_pct
        self._state.sbti_criteria_gaps = gaps

        status = "not_ready"
        if readiness_pct >= 90.0:
            status = "ready"
        elif readiness_pct >= 70.0:
            status = "near_ready"
        elif readiness_pct >= 40.0:
            status = "in_progress"

        return {
            "readiness_score": readiness_pct,
            "max_score": max_score,
            "status": status,
            "gaps": gaps,
            "recommendations": recs,
            "criteria_checked": len(SBTI_NEAR_TERM_CRITERIA) + len(SBTI_NET_ZERO_CRITERIA),
            "provenance_hash": _compute_hash({
                "score": readiness_pct,
                "gaps": gaps,
            }),
        }

    # ----- Step Handlers -----
    # Each returns (errors: List[str], warnings: List[str])

    def _handle_org_profile(
        self, data: Dict[str, Any],
    ) -> tuple:
        """Handle organization profile step with SBTi classification."""
        errors: List[str] = []
        warnings: List[str] = []
        try:
            profile = OrganizationProfile(**data)

            # SBTi-specific validation
            if profile.flag_revenue_pct >= 20.0:
                if profile.flag_exposure not in (
                    FLAGExposure.ABOVE_THRESHOLD.value,
                    FLAGExposure.PRIMARILY_FLAG.value,
                    FLAGExposure.ABOVE_THRESHOLD,
                    FLAGExposure.PRIMARILY_FLAG,
                ):
                    warnings.append(
                        f"FLAG revenue is {profile.flag_revenue_pct}% (>=20%) but "
                        f"flag_exposure is '{profile.flag_exposure}'. "
                        f"FLAG targets will be required."
                    )

            if profile.is_financial_institution and profile.organization_type != OrganizationType.FINANCIAL_INSTITUTION:
                warnings.append(
                    "is_financial_institution=True but organization_type is not "
                    "'financial_institution'. FINZ module will be enabled."
                )

            if profile.sbti_commitment_status == "not_committed":
                warnings.append(
                    "SBTi commitment not yet made. Consider submitting "
                    "commitment letter before target development (C1-C2)."
                )

            if not errors and self._state:
                self._state.org_profile = profile

        except Exception as exc:
            errors.append(f"Invalid organization profile: {exc}")
        return errors, warnings

    def _handle_boundary(
        self, data: Dict[str, Any],
    ) -> tuple:
        """Handle boundary selection step with SBTi target boundary rules."""
        errors: List[str] = []
        warnings: List[str] = []
        try:
            boundary = BoundarySelection(**data)

            # SBTi requires S1+S2 in target boundary
            if not boundary.include_scope_1_in_target:
                errors.append(
                    "SBTi C9: Scope 1 must be included in target boundary"
                )
            if not boundary.include_scope_2_in_target:
                errors.append(
                    "SBTi C10: Scope 2 must be included in target boundary"
                )

            # SBTi max 5% exclusion
            if boundary.max_exclusion_pct > 5.0:
                warnings.append(
                    f"SBTi C18: Boundary exclusions at {boundary.max_exclusion_pct}% "
                    f"exceed 5% maximum. Justify or reduce exclusions."
                )

            if boundary.exclusion_justification == "" and boundary.max_exclusion_pct > 0:
                warnings.append(
                    "SBTi C18: Exclusion justification required for any "
                    "excluded emission sources."
                )

            if not errors and self._state:
                self._state.boundary = boundary

        except Exception as exc:
            errors.append(f"Invalid boundary selection: {exc}")
        return errors, warnings

    def _handle_scope_config(
        self, data: Dict[str, Any],
    ) -> tuple:
        """Handle scope configuration with SBTi materiality screening."""
        errors: List[str] = []
        warnings: List[str] = []
        try:
            scope_config = ScopeConfiguration(**data)

            # Validate Scope 3 category numbers
            for cat in scope_config.scope3_categories:
                if cat < 1 or cat > 15:
                    errors.append(
                        f"Invalid Scope 3 category: {cat} (must be 1-15)"
                    )

            # SBTi: if S3 >40% of total, S3 target required
            if scope_config.scope3_total_pct_of_emissions >= 40.0:
                if not scope_config.scope3_requires_target:
                    warnings.append(
                        f"SBTi C12: Scope 3 is {scope_config.scope3_total_pct_of_emissions}% "
                        f"of total emissions (>=40%). Scope 3 target is required."
                    )
                    scope_config.scope3_requires_target = True

            # SBTi: S3 coverage requirements
            if scope_config.scope3_requires_target:
                if scope_config.scope3_coverage_target_pct < 67.0:
                    warnings.append(
                        f"SBTi C13: Scope 3 coverage target is "
                        f"{scope_config.scope3_coverage_target_pct}% "
                        f"but minimum 67% required for near-term targets."
                    )

            # SBTi: screening must be done
            if not scope_config.scope3_screening_done:
                warnings.append(
                    "SBTi C8: Scope 3 screening not yet completed. "
                    "All 15 categories must be screened."
                )

            # FLAG commodities check
            if scope_config.flag_in_scope:
                for commodity in scope_config.flag_commodities_in_scope:
                    if commodity not in FLAG_COMMODITIES:
                        warnings.append(
                            f"FLAG commodity '{commodity}' not in standard list. "
                            f"Valid: {sorted(FLAG_COMMODITIES.keys())}"
                        )

            if not errors and self._state:
                self._state.scope_config = scope_config

        except Exception as exc:
            errors.append(f"Invalid scope configuration: {exc}")
        return errors, warnings

    def _handle_data_sources(
        self, data: Dict[str, Any],
    ) -> tuple:
        """Handle data source setup with SBTi base year requirements."""
        errors: List[str] = []
        warnings: List[str] = []
        try:
            data_setup = DataSourceSetup(**data)

            # SBTi C5: base year validation
            current_year = utcnow().year
            if current_year - data_setup.base_year > 3:
                warnings.append(
                    f"SBTi C5: Base year {data_setup.base_year} is "
                    f"{current_year - data_setup.base_year} years old. "
                    f"Should be within 2 years of most recent inventory."
                )

            if not data_setup.base_year_data_available:
                warnings.append(
                    "SBTi C5-C6: Base year inventory data not yet available. "
                    "Complete GHG inventory before target submission."
                )

            # SBTi C28: recalculation policy
            if not data_setup.recalculation_policy_defined:
                warnings.append(
                    "SBTi C28: Base year recalculation policy not defined. "
                    "Required for SBTi validation."
                )

            # Data quality check
            if data_setup.data_quality_score > 0 and data_setup.data_quality_score < 3.0:
                warnings.append(
                    f"Data quality score is {data_setup.data_quality_score}/5. "
                    f"Consider improving to meet GHG Protocol standards."
                )

            # FI-specific data check
            if (self._state and self._state.org_profile and
                    self._state.org_profile.is_financial_institution):
                if not data_setup.portfolio_data_available:
                    warnings.append(
                        "FINZ: Portfolio-level emissions data not available. "
                        "Required for FI portfolio target setting."
                    )
                if not data_setup.asset_classes_with_data:
                    warnings.append(
                        "FINZ: No asset classes with emissions data. "
                        "At least one asset class required."
                    )

            if not errors and self._state:
                self._state.data_sources = data_setup

        except Exception as exc:
            errors.append(f"Invalid data source setup: {exc}")
        return errors, warnings

    def _handle_target_prefs(
        self, data: Dict[str, Any],
    ) -> tuple:
        """Handle target preferences with SBTi pathway validation."""
        errors: List[str] = []
        warnings: List[str] = []
        try:
            prefs = TargetPreferences(**data)

            # Validate timeline (C15)
            if prefs.near_term_target_year <= prefs.base_year:
                errors.append(
                    "Near-term target year must be after base year"
                )
            if prefs.long_term_target_year <= prefs.near_term_target_year:
                errors.append(
                    "Long-term target year must be after near-term target year"
                )
            nt_gap = prefs.near_term_target_year - prefs.base_year
            if nt_gap < 5 or nt_gap > 15:
                warnings.append(
                    f"SBTi C15: Near-term target should be 5-10 years from "
                    f"submission. Current gap: {nt_gap} years from base year."
                )

            # SBTi C24: net-zero no later than 2050
            if prefs.net_zero_target_year > 2050:
                warnings.append(
                    f"SBTi C24/NZ-C1: Net-zero target year {prefs.net_zero_target_year} "
                    f"exceeds 2050 maximum."
                )

            # Near-term reduction check (C9)
            if prefs.ambition_level == SBTiAmbitionLevel.AMBITIOUS_1_5C:
                min_reduction = 42.0
            elif prefs.ambition_level == SBTiAmbitionLevel.WELL_BELOW_2C:
                min_reduction = 25.0
            else:
                min_reduction = 15.0

            if prefs.scope1_scope2_near_term_pct < min_reduction:
                warnings.append(
                    f"SBTi C9: S1+S2 near-term reduction "
                    f"{prefs.scope1_scope2_near_term_pct}% is below "
                    f"{min_reduction}% minimum for {prefs.ambition_level.value} pathway."
                )

            # Long-term check (C22-C23)
            if prefs.scope1_scope2_long_term_pct < 90.0:
                warnings.append(
                    f"SBTi C22: Long-term S1+S2 reduction "
                    f"{prefs.scope1_scope2_long_term_pct}% is below 90% minimum."
                )

            # Residual check (NZ-C3, C26)
            if prefs.residual_emissions_max_pct > 10.0:
                warnings.append(
                    f"SBTi C26: Residual emissions at {prefs.residual_emissions_max_pct}% "
                    f"exceed 10% maximum."
                )

            # SDA pathway validation
            if prefs.primary_pathway == SBTiPathwayType.SDA:
                if not prefs.sda_sector:
                    errors.append(
                        "SDA pathway selected but no sector specified. "
                        "Choose from: " + ", ".join(sorted(SDA_SECTORS.keys()))
                    )
                elif prefs.sda_sector not in SDA_SECTORS:
                    errors.append(
                        f"Invalid SDA sector '{prefs.sda_sector}'. "
                        f"Valid: {sorted(SDA_SECTORS.keys())}"
                    )

            # FLAG pathway validation
            if prefs.primary_pathway == SBTiPathwayType.FLAG:
                if not prefs.flag_commodities:
                    warnings.append(
                        "FLAG pathway selected but no commodities specified. "
                        "At least one FLAG commodity required."
                    )
                if not prefs.flag_no_deforestation_commitment:
                    warnings.append(
                        "FLAG: Zero deforestation commitment by 2025 "
                        "is required for FLAG targets."
                    )

            # FINZ pathway validation
            if prefs.primary_pathway == SBTiPathwayType.FINZ:
                if not prefs.fi_target_enabled:
                    errors.append(
                        "FINZ pathway selected but fi_target_enabled=False"
                    )
                if not prefs.fi_asset_classes:
                    warnings.append(
                        "FINZ: No asset classes selected for portfolio targets. "
                        "At least one required."
                    )

            # Offsets warning (C16)
            if not prefs.include_neutralization:
                warnings.append(
                    "SBTi C16: Offsets cannot count toward near-term or "
                    "long-term target achievement. Neutralization is only "
                    "for residual emissions at net-zero."
                )

            if not errors and self._state:
                self._state.target_prefs = prefs

        except Exception as exc:
            errors.append(f"Invalid target preferences: {exc}")
        return errors, warnings

    def _handle_preset(
        self, data: Dict[str, Any],
    ) -> tuple:
        """Handle preset selection with auto-detection."""
        errors: List[str] = []
        warnings: List[str] = []
        preset_name = data.get("preset_name", "")

        if not preset_name:
            # Auto-detect from org profile and target preferences
            if self._state and self._state.org_profile:
                is_fi = self._state.org_profile.is_financial_institution
                flag_exp = self._state.org_profile.flag_exposure
                if isinstance(flag_exp, FLAGExposure):
                    flag_exp = flag_exp.value
                is_sda = (
                    self._state.target_prefs.primary_pathway == SBTiPathwayType.SDA
                    if self._state.target_prefs else False
                )
                rec = self.recommend_preset(
                    sector=self._state.org_profile.sector,
                    is_fi=is_fi,
                    flag_exposure=flag_exp,
                    is_sda=is_sda,
                    size=self._state.org_profile.size.value,
                )
                preset_name = rec["recommended_preset"]
                warnings.append(
                    f"Auto-selected preset '{preset_name}' based on "
                    f"organization profile."
                )

        preset = SECTOR_PRESETS.get(preset_name)
        if preset is None:
            errors.append(
                f"Unknown preset '{preset_name}'. "
                f"Valid: {sorted(SECTOR_PRESETS.keys())}"
            )
            return errors, warnings

        # Build selection from preset
        selection = PresetSelection(
            preset_name=preset_name,
            preset_applied=True,
            engines_enabled=preset.get("engines", []),
            scope3_priority=preset.get("scope3_priority", []),
            recommended_levers=preset.get("recommended_levers", []),
            pathway=preset.get("pathway", "aca"),
            sda_enabled=preset.get("sda_enabled", False),
            flag_enabled=preset.get("flag_enabled", False),
            fi_enabled=preset.get("fi_enabled", False),
            neutralization_enabled=True,
            near_term_min_reduction_pct=preset.get(
                "near_term_min_reduction_pct", 42.0,
            ),
            sbti_criteria_applicable=preset.get(
                "sbti_criteria", ["C1-C28"],
            ),
        )

        if self._state:
            self._state.preset = selection

        self.logger.info("SBTi preset applied: %s", preset_name)
        return errors, warnings

    # ----- Navigation -----

    def _advance_step(self, current: SBTiWizardStep) -> None:
        """Advance to the next step in the wizard."""
        if self._state is None:
            return
        try:
            idx = STEP_ORDER.index(current)
            if idx < len(STEP_ORDER) - 1:
                self._state.current_step = STEP_ORDER[idx + 1]
            else:
                self._state.is_complete = True
                self._state.completed_at = utcnow()
                # Compute final readiness
                self.assess_sbti_readiness()
        except ValueError:
            pass

    # ----- Demo Scenarios -----

    def _demo_corporate_aca(self) -> Dict[str, Dict[str, Any]]:
        """Demo: Manufacturing company using ACA pathway."""
        return {
            "organization_profile": {
                "organization_name": "Demo Manufacturing AG",
                "sector": "manufacturing",
                "sub_sector": "machinery",
                "region": "EU",
                "country": "DE",
                "employee_count": 5000,
                "annual_revenue_eur": 500_000_000.0,
                "size": "large",
                "is_listed": True,
                "nace_code": "C28.1",
                "organization_type": "corporate",
                "is_financial_institution": False,
                "flag_exposure": "none",
                "flag_revenue_pct": 0.0,
                "sbti_commitment_status": "committed",
                "multi_entity": True,
                "entity_count": 3,
            },
            "boundary_selection": {
                "consolidation_approach": "operational_control",
                "include_subsidiaries": True,
                "subsidiary_count": 2,
                "countries_of_operation": ["DE", "US", "CN"],
                "include_scope_1_in_target": True,
                "include_scope_2_in_target": True,
                "include_scope_3_in_target": True,
                "biogenic_emissions_separate": True,
                "max_exclusion_pct": 3.0,
            },
            "scope_configuration": {
                "include_scope_1": True,
                "include_scope_2": True,
                "include_scope_3": True,
                "scope2_methods": ["location_based", "market_based"],
                "scope3_categories": [1, 2, 3, 4, 5, 6, 7, 9, 12],
                "scope1_sources": [
                    "stationary_combustion", "mobile_combustion",
                    "process_emissions", "fugitive_emissions",
                ],
                "scope3_screening_done": True,
                "scope3_total_pct_of_emissions": 65.0,
                "scope3_material_categories": [1, 2, 4, 5, 9],
                "scope3_coverage_target_pct": 67.0,
                "scope3_requires_target": True,
            },
            "data_source_setup": {
                "erp_system": "sap",
                "erp_connected": True,
                "file_formats": ["excel", "csv"],
                "utility_provider_apis": True,
                "travel_management_system": "concur",
                "base_year": 2019,
                "base_year_data_available": True,
                "base_year_total_tco2e": 150_000.0,
                "emission_factor_source": "ghg_protocol",
                "data_quality_score": 3.5,
                "historical_years_available": 5,
                "recalculation_policy_defined": True,
                "recalculation_trigger_pct": 5.0,
            },
            "target_preferences": {
                "ambition_level": "1.5C",
                "primary_pathway": "aca",
                "base_year": 2019,
                "near_term_target_year": 2030,
                "long_term_target_year": 2050,
                "net_zero_target_year": 2050,
                "scope1_scope2_near_term_pct": 46.2,
                "scope3_near_term_pct": 27.5,
                "scope1_scope2_long_term_pct": 95.0,
                "total_long_term_pct": 90.0,
                "residual_emissions_max_pct": 10.0,
                "include_neutralization": True,
                "include_bvcm": True,
                "sbti_submission_planned": True,
            },
            "preset_selection": {
                "preset_name": "manufacturing_aca",
            },
        }

    def _demo_corporate_sda(self) -> Dict[str, Dict[str, Any]]:
        """Demo: Cement company using SDA pathway."""
        return {
            "organization_profile": {
                "organization_name": "Demo Cement Corp",
                "sector": "cement",
                "sub_sector": "clinker_production",
                "region": "EU",
                "country": "DE",
                "employee_count": 8000,
                "annual_revenue_eur": 2_000_000_000.0,
                "size": "enterprise",
                "is_listed": True,
                "nace_code": "C23.5",
                "organization_type": "corporate",
                "is_financial_institution": False,
                "flag_exposure": "none",
                "flag_revenue_pct": 0.0,
                "sbti_commitment_status": "targets_set",
                "multi_entity": True,
                "entity_count": 5,
            },
            "boundary_selection": {
                "consolidation_approach": "operational_control",
                "include_subsidiaries": True,
                "subsidiary_count": 4,
                "countries_of_operation": ["DE", "US", "IN", "BR"],
                "include_scope_1_in_target": True,
                "include_scope_2_in_target": True,
                "include_scope_3_in_target": True,
                "biogenic_emissions_separate": True,
                "max_exclusion_pct": 2.0,
            },
            "scope_configuration": {
                "include_scope_1": True,
                "include_scope_2": True,
                "include_scope_3": True,
                "scope2_methods": ["location_based", "market_based"],
                "scope3_categories": [1, 2, 3, 4, 5, 6, 7, 9],
                "scope1_sources": [
                    "stationary_combustion", "process_emissions",
                    "fugitive_emissions",
                ],
                "scope3_screening_done": True,
                "scope3_total_pct_of_emissions": 45.0,
                "scope3_material_categories": [1, 3, 4, 5],
                "scope3_coverage_target_pct": 67.0,
                "scope3_requires_target": True,
            },
            "data_source_setup": {
                "erp_system": "sap",
                "erp_connected": True,
                "file_formats": ["excel", "csv"],
                "base_year": 2019,
                "base_year_data_available": True,
                "base_year_total_tco2e": 850_000.0,
                "emission_factor_source": "ghg_protocol",
                "data_quality_score": 4.0,
                "historical_years_available": 6,
                "recalculation_policy_defined": True,
                "recalculation_trigger_pct": 5.0,
            },
            "target_preferences": {
                "ambition_level": "1.5C",
                "primary_pathway": "sda",
                "base_year": 2019,
                "near_term_target_year": 2030,
                "long_term_target_year": 2050,
                "net_zero_target_year": 2050,
                "scope1_scope2_near_term_pct": 42.0,
                "scope3_near_term_pct": 25.0,
                "scope1_scope2_long_term_pct": 90.0,
                "total_long_term_pct": 90.0,
                "residual_emissions_max_pct": 10.0,
                "sda_sector": "cement",
                "sda_intensity_metric": "tCO2e/t_cementitious",
                "sda_current_intensity": 0.85,
                "include_neutralization": True,
                "sbti_submission_planned": True,
            },
            "preset_selection": {
                "preset_name": "heavy_industry_sda",
            },
        }

    def _demo_financial_institution(self) -> Dict[str, Dict[str, Any]]:
        """Demo: Bank using FINZ V1.0 portfolio targets."""
        return {
            "organization_profile": {
                "organization_name": "Demo Bank AG",
                "sector": "financial_services",
                "sub_sector": "commercial_banking",
                "region": "EU",
                "country": "DE",
                "employee_count": 12000,
                "annual_revenue_eur": 5_000_000_000.0,
                "size": "enterprise",
                "is_listed": True,
                "nace_code": "K64.1",
                "organization_type": "financial_institution",
                "is_financial_institution": True,
                "flag_exposure": "none",
                "flag_revenue_pct": 0.0,
                "sbti_commitment_status": "committed",
                "multi_entity": True,
                "entity_count": 8,
            },
            "boundary_selection": {
                "consolidation_approach": "operational_control",
                "include_subsidiaries": True,
                "subsidiary_count": 7,
                "countries_of_operation": ["DE", "UK", "US", "SG", "HK"],
                "include_scope_1_in_target": True,
                "include_scope_2_in_target": True,
                "include_scope_3_in_target": True,
                "biogenic_emissions_separate": True,
                "max_exclusion_pct": 2.0,
            },
            "scope_configuration": {
                "include_scope_1": True,
                "include_scope_2": True,
                "include_scope_3": True,
                "scope2_methods": ["location_based", "market_based"],
                "scope3_categories": [1, 6, 7, 15],
                "scope1_sources": ["stationary_combustion"],
                "scope3_screening_done": True,
                "scope3_total_pct_of_emissions": 92.0,
                "scope3_material_categories": [15],
                "scope3_coverage_target_pct": 67.0,
                "scope3_requires_target": True,
            },
            "data_source_setup": {
                "erp_system": "sap",
                "erp_connected": True,
                "file_formats": ["excel", "csv"],
                "base_year": 2020,
                "base_year_data_available": True,
                "base_year_total_tco2e": 25_000.0,
                "emission_factor_source": "ghg_protocol",
                "data_quality_score": 3.5,
                "historical_years_available": 4,
                "recalculation_policy_defined": True,
                "recalculation_trigger_pct": 5.0,
                "portfolio_data_available": True,
                "asset_classes_with_data": [
                    "listed_equity", "corporate_bonds",
                    "commercial_real_estate", "mortgages",
                ],
            },
            "target_preferences": {
                "ambition_level": "1.5C",
                "primary_pathway": "finz",
                "base_year": 2020,
                "near_term_target_year": 2030,
                "long_term_target_year": 2050,
                "net_zero_target_year": 2050,
                "scope1_scope2_near_term_pct": 46.2,
                "scope3_near_term_pct": 25.0,
                "scope1_scope2_long_term_pct": 90.0,
                "total_long_term_pct": 90.0,
                "residual_emissions_max_pct": 10.0,
                "fi_target_enabled": True,
                "fi_asset_classes": [
                    "listed_equity", "corporate_bonds",
                    "commercial_real_estate", "mortgages",
                ],
                "fi_portfolio_coverage_pct": 67.0,
                "include_neutralization": True,
                "include_bvcm": True,
                "sbti_submission_planned": True,
            },
            "preset_selection": {
                "preset_name": "financial_institution",
            },
        }

    def _demo_agriculture_flag(self) -> Dict[str, Dict[str, Any]]:
        """Demo: Agribusiness with FLAG targets required."""
        return {
            "organization_profile": {
                "organization_name": "Demo Agri Holdings",
                "sector": "agriculture",
                "sub_sector": "livestock_and_crops",
                "region": "SA",
                "country": "BR",
                "employee_count": 3000,
                "annual_revenue_eur": 800_000_000.0,
                "size": "large",
                "is_listed": True,
                "nace_code": "A01.1",
                "organization_type": "corporate",
                "is_financial_institution": False,
                "flag_exposure": "primarily_flag",
                "flag_revenue_pct": 75.0,
                "sbti_commitment_status": "committed",
                "multi_entity": True,
                "entity_count": 4,
            },
            "boundary_selection": {
                "consolidation_approach": "operational_control",
                "include_subsidiaries": True,
                "subsidiary_count": 3,
                "countries_of_operation": ["BR", "AR", "UY"],
                "include_scope_1_in_target": True,
                "include_scope_2_in_target": True,
                "include_scope_3_in_target": True,
                "biogenic_emissions_separate": True,
                "max_exclusion_pct": 3.0,
            },
            "scope_configuration": {
                "include_scope_1": True,
                "include_scope_2": True,
                "include_scope_3": True,
                "scope2_methods": ["location_based", "market_based"],
                "scope3_categories": [1, 3, 4, 5, 10, 12],
                "scope1_sources": [
                    "stationary_combustion", "mobile_combustion",
                    "agricultural_emissions", "land_use_emissions",
                ],
                "scope3_screening_done": True,
                "scope3_total_pct_of_emissions": 55.0,
                "scope3_material_categories": [1, 4, 10],
                "scope3_coverage_target_pct": 67.0,
                "scope3_requires_target": True,
                "flag_in_scope": True,
                "flag_commodities_in_scope": [
                    "cattle_beef", "cattle_dairy", "soy",
                ],
            },
            "data_source_setup": {
                "erp_system": "none",
                "erp_connected": False,
                "file_formats": ["excel", "csv"],
                "base_year": 2020,
                "base_year_data_available": True,
                "base_year_total_tco2e": 420_000.0,
                "emission_factor_source": "ipcc_ar6",
                "data_quality_score": 3.0,
                "historical_years_available": 3,
                "recalculation_policy_defined": True,
                "recalculation_trigger_pct": 5.0,
            },
            "target_preferences": {
                "ambition_level": "1.5C",
                "primary_pathway": "flag",
                "base_year": 2020,
                "near_term_target_year": 2030,
                "long_term_target_year": 2050,
                "net_zero_target_year": 2050,
                "scope1_scope2_near_term_pct": 42.0,
                "scope3_near_term_pct": 30.0,
                "scope1_scope2_long_term_pct": 90.0,
                "total_long_term_pct": 90.0,
                "residual_emissions_max_pct": 10.0,
                "flag_target_required": True,
                "flag_commodities": [
                    "cattle_beef", "cattle_dairy", "soy",
                ],
                "flag_no_deforestation_commitment": True,
                "include_neutralization": True,
                "include_bvcm": True,
                "sbti_submission_planned": True,
            },
            "preset_selection": {
                "preset_name": "agriculture_flag",
            },
        }

    # ----- Result Generation -----

    def _generate_result(self) -> SetupResult:
        """Generate the final SBTi-aligned setup result from wizard state."""
        if self._state is None:
            return SetupResult()

        completed_count = sum(
            1 for s in self._state.steps.values()
            if s.status == StepStatus.COMPLETED
        )

        # Build scopes list
        scopes: List[str] = []
        scope3_cats: List[int] = []
        scope3_coverage = 67.0
        scope3_required = True
        if self._state.scope_config:
            if self._state.scope_config.include_scope_1:
                scopes.append("scope_1")
            if self._state.scope_config.include_scope_2:
                scopes.append("scope_2")
            if self._state.scope_config.include_scope_3:
                scopes.append("scope_3")
            scope3_cats = list(self._state.scope_config.scope3_categories)
            scope3_coverage = self._state.scope_config.scope3_coverage_target_pct
            scope3_required = self._state.scope_config.scope3_requires_target

        # Get engines and levers from preset
        engines: List[str] = []
        levers: List[str] = []
        sbti_criteria: List[str] = ["C1-C28"]
        if self._state.preset:
            engines = list(self._state.preset.engines_enabled)
            levers = list(self._state.preset.recommended_levers)
            sbti_criteria = list(self._state.preset.sbti_criteria_applicable)

        # Compute config hash
        config_hash = _compute_hash({
            "org": (
                self._state.org_profile.organization_name
                if self._state.org_profile else ""
            ),
            "sector": (
                self._state.org_profile.sector
                if self._state.org_profile else ""
            ),
            "pathway": (
                self._state.target_prefs.primary_pathway.value
                if self._state.target_prefs else ""
            ),
            "base_year": (
                self._state.target_prefs.base_year
                if self._state.target_prefs else 2019
            ),
            "scopes": scopes,
            "scope3": scope3_cats,
        })

        result = SetupResult(
            organization_name=(
                self._state.org_profile.organization_name
                if self._state.org_profile else ""
            ),
            sector=(
                self._state.org_profile.sector
                if self._state.org_profile else ""
            ),
            organization_type=(
                self._state.org_profile.organization_type.value
                if self._state.org_profile else "corporate"
            ),
            is_financial_institution=(
                self._state.org_profile.is_financial_institution
                if self._state.org_profile else False
            ),
            flag_exposure=(
                self._state.org_profile.flag_exposure.value
                if self._state.org_profile and isinstance(
                    self._state.org_profile.flag_exposure, FLAGExposure
                )
                else str(self._state.org_profile.flag_exposure)
                if self._state.org_profile else "none"
            ),
            consolidation_approach=(
                self._state.boundary.consolidation_approach.value
                if self._state.boundary else ""
            ),
            multi_entity=(
                self._state.org_profile.multi_entity
                if self._state.org_profile else False
            ),
            entity_count=(
                self._state.org_profile.entity_count
                if self._state.org_profile else 1
            ),
            scopes_included=scopes,
            scope3_categories=scope3_cats,
            scope3_coverage_pct=scope3_coverage,
            scope3_requires_target=scope3_required,
            primary_pathway=(
                self._state.target_prefs.primary_pathway.value
                if self._state.target_prefs else "aca"
            ),
            ambition_level=(
                self._state.target_prefs.ambition_level.value
                if self._state.target_prefs else "1.5C"
            ),
            base_year=(
                self._state.target_prefs.base_year
                if self._state.target_prefs else 2019
            ),
            near_term_target_year=(
                self._state.target_prefs.near_term_target_year
                if self._state.target_prefs else 2030
            ),
            long_term_target_year=(
                self._state.target_prefs.long_term_target_year
                if self._state.target_prefs else 2050
            ),
            net_zero_target_year=(
                self._state.target_prefs.net_zero_target_year
                if self._state.target_prefs else 2050
            ),
            scope1_scope2_near_term_pct=(
                self._state.target_prefs.scope1_scope2_near_term_pct
                if self._state.target_prefs else 42.0
            ),
            scope3_near_term_pct=(
                self._state.target_prefs.scope3_near_term_pct
                if self._state.target_prefs else 25.0
            ),
            scope1_scope2_long_term_pct=(
                self._state.target_prefs.scope1_scope2_long_term_pct
                if self._state.target_prefs else 90.0
            ),
            is_sda_sector=(
                self._state.target_prefs.primary_pathway == SBTiPathwayType.SDA
                if self._state.target_prefs else False
            ),
            sda_sector=(
                self._state.target_prefs.sda_sector
                if self._state.target_prefs else ""
            ),
            sda_intensity_metric=(
                self._state.target_prefs.sda_intensity_metric
                if self._state.target_prefs else ""
            ),
            flag_target_required=(
                self._state.target_prefs.flag_target_required
                if self._state.target_prefs else False
            ),
            flag_commodities=(
                list(self._state.target_prefs.flag_commodities)
                if self._state.target_prefs else []
            ),
            fi_target_enabled=(
                self._state.target_prefs.fi_target_enabled
                if self._state.target_prefs else False
            ),
            fi_asset_classes=(
                list(self._state.target_prefs.fi_asset_classes)
                if self._state.target_prefs else []
            ),
            engines_enabled=engines,
            recommended_levers=levers,
            sbti_criteria_applicable=sbti_criteria,
            sbti_readiness_score=self._state.sbti_readiness_score,
            sbti_criteria_gaps=list(self._state.sbti_criteria_gaps),
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
