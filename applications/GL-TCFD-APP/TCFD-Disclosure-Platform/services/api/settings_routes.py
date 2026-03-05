"""
GL-TCFD-APP Settings API

Manages organization-level configuration for the TCFD Disclosure & Scenario
Analysis Platform. Provides endpoints for retrieving and updating reporting
preferences, scenario defaults, and jurisdictional settings.  Also exposes
reference-data catalogues for currencies, sectors, physical hazard types,
scenario types, and supported jurisdictions.

Settings Categories:
    - Reporting: fiscal year, reporting period, currency, units
    - Scenarios: default scenario families, temperature pathways, time horizons
    - Jurisdictions: regulatory requirements by geography
    - Thresholds: materiality thresholds, risk appetite, target ambition

Reference Data Catalogues:
    - Currencies: 20 ISO 4217 currencies supported for financial modelling
    - Sectors: 8 industry sectors with sub-sectors, TCFD/ISSB guidance flags
    - Hazard Types: 12 physical hazards (7 acute, 5 chronic) with RCP coverage
    - Scenario Types: 10 pre-built IEA/NGFS/IPCC climate scenarios
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/tcfd/settings", tags=["Settings"])


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


class ReportingPeriod(str, Enum):
    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"


class DisclosureFramework(str, Enum):
    TCFD = "tcfd"
    ISSB_IFRS_S2 = "issb_ifrs_s2"
    CSRD_ESRS_E1 = "csrd_esrs_e1"
    SEC_CLIMATE = "sec_climate"


class RiskScoringMethod(str, Enum):
    QUALITATIVE_5X5 = "qualitative_5x5"
    SEMI_QUANTITATIVE = "semi_quantitative"
    QUANTITATIVE_MONTE_CARLO = "quantitative_monte_carlo"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class UpdateSettingsRequest(BaseModel):
    """Update organization settings."""
    organization_name: Optional[str] = Field(None, max_length=300, description="Organization name")
    industry: Optional[str] = Field(None, max_length=200, description="Industry classification")
    sector: Optional[str] = Field(None, max_length=200, description="Sector (for benchmarking)")
    fiscal_year_end: Optional[FiscalYearEnd] = Field(None, description="Fiscal year end month")
    reporting_period: Optional[ReportingPeriod] = Field(None, description="Reporting frequency")
    reporting_currency: Optional[Currency] = Field(None, description="Reporting currency")
    emission_unit: Optional[EmissionUnit] = Field(None, description="Preferred emission unit")
    base_year: Optional[int] = Field(None, ge=2000, le=2030, description="Emissions base year")
    materiality_threshold_usd: Optional[float] = Field(None, ge=0, description="Materiality threshold (USD)")
    internal_carbon_price_usd: Optional[float] = Field(None, ge=0, description="Internal carbon price (USD/tCO2e)")
    discount_rate_pct: Optional[float] = Field(None, ge=0, le=50, description="Default discount rate (%)")
    primary_regulation: Optional[str] = Field(None, description="Primary regulation: tcfd, issb, csrd, sec")
    risk_scoring_method: Optional[RiskScoringMethod] = Field(None, description="Default risk scoring method")
    auto_generate_disclosures: Optional[bool] = Field(None, description="Auto-generate disclosure sections")
    enable_issb_crosswalk: Optional[bool] = Field(None, description="Enable ISSB cross-walk features")
    enable_peer_benchmarking: Optional[bool] = Field(None, description="Enable peer benchmarking features")
    notification_email: Optional[str] = Field(None, description="Notification email address")

    class Config:
        json_schema_extra = {
            "example": {
                "organization_name": "Acme Corporation",
                "industry": "Manufacturing",
                "sector": "Materials & Buildings",
                "fiscal_year_end": "december",
                "reporting_period": "annual",
                "reporting_currency": "USD",
                "emission_unit": "tCO2e",
                "base_year": 2020,
                "materiality_threshold_usd": 1000000,
                "internal_carbon_price_usd": 75,
                "discount_rate_pct": 8.0,
                "primary_regulation": "issb",
                "risk_scoring_method": "qualitative_5x5",
                "auto_generate_disclosures": True,
                "enable_issb_crosswalk": True,
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SettingsResponse(BaseModel):
    """Organization settings."""
    org_id: str
    organization_name: str
    industry: str
    sector: str
    fiscal_year_end: str
    reporting_period: str
    reporting_currency: str
    emission_unit: str
    base_year: int
    materiality_threshold_usd: float
    internal_carbon_price_usd: float
    discount_rate_pct: float
    primary_regulation: str
    risk_scoring_method: str
    auto_generate_disclosures: bool
    enable_issb_crosswalk: bool
    enable_peer_benchmarking: bool
    notification_email: Optional[str]
    updated_at: datetime


class CurrencyInfo(BaseModel):
    """Supported currency reference."""
    code: str = Field(..., description="ISO 4217 currency code")
    name: str = Field(..., description="Currency name")
    symbol: str = Field(..., description="Currency symbol")
    supported_for_financial_impact: bool = Field(..., description="Supported for financial impact modelling")


class SectorInfo(BaseModel):
    """Supported sector reference."""
    sector_id: str = Field(..., description="Sector identifier")
    name: str = Field(..., description="Sector display name")
    sub_sectors: List[Dict[str, str]] = Field(..., description="Available sub-sectors")
    tcfd_supplemental_guidance: bool = Field(..., description="Has TCFD supplemental guidance")
    issb_industry_metrics: bool = Field(..., description="Has ISSB industry-specific metrics")


class HazardTypeInfo(BaseModel):
    """Physical hazard type reference."""
    hazard_id: str = Field(..., description="Hazard identifier")
    name: str = Field(..., description="Hazard display name")
    category: str = Field(..., description="acute or chronic")
    description: str = Field(..., description="Hazard description")
    typical_metrics: List[str] = Field(..., description="Common measurement metrics")
    rcp_scenarios_available: List[str] = Field(..., description="Available RCP scenarios")


class ScenarioTypeInfo(BaseModel):
    """Available scenario type reference."""
    scenario_id: str = Field(..., description="Scenario identifier")
    name: str = Field(..., description="Scenario display name")
    provider: str = Field(..., description="Scenario provider (IEA, NGFS, IPCC)")
    warming_target: str = Field(..., description="Temperature pathway")
    description: str = Field(..., description="Scenario description")
    time_horizon: str = Field(..., description="Analysis time horizon")
    transition_risk_level: str = Field(..., description="Relative transition risk level")
    physical_risk_level: str = Field(..., description="Relative physical risk level")
    recommended_for: List[str] = Field(..., description="Recommended use cases")


class JurisdictionResponse(BaseModel):
    """Supported jurisdiction details."""
    jurisdiction_code: str
    jurisdiction_name: str
    region: str
    regulations: List[Dict[str, Any]]
    mandatory_tcfd: bool
    issb_adopted: bool
    additional_requirements: List[str]


class JurisdictionListResponse(BaseModel):
    """List of supported jurisdictions."""
    total_jurisdictions: int
    jurisdictions: List[JurisdictionResponse]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Static Reference Data -- Currencies
# ---------------------------------------------------------------------------

SUPPORTED_CURRENCIES: List[Dict[str, Any]] = [
    {"code": "USD", "name": "US Dollar", "symbol": "$", "supported_for_financial_impact": True},
    {"code": "EUR", "name": "Euro", "symbol": "\u20ac", "supported_for_financial_impact": True},
    {"code": "GBP", "name": "British Pound", "symbol": "\u00a3", "supported_for_financial_impact": True},
    {"code": "JPY", "name": "Japanese Yen", "symbol": "\u00a5", "supported_for_financial_impact": True},
    {"code": "CHF", "name": "Swiss Franc", "symbol": "CHF", "supported_for_financial_impact": True},
    {"code": "CAD", "name": "Canadian Dollar", "symbol": "C$", "supported_for_financial_impact": True},
    {"code": "AUD", "name": "Australian Dollar", "symbol": "A$", "supported_for_financial_impact": True},
    {"code": "CNY", "name": "Chinese Yuan", "symbol": "\u00a5", "supported_for_financial_impact": True},
    {"code": "INR", "name": "Indian Rupee", "symbol": "\u20b9", "supported_for_financial_impact": True},
    {"code": "BRL", "name": "Brazilian Real", "symbol": "R$", "supported_for_financial_impact": True},
    {"code": "SGD", "name": "Singapore Dollar", "symbol": "S$", "supported_for_financial_impact": True},
    {"code": "HKD", "name": "Hong Kong Dollar", "symbol": "HK$", "supported_for_financial_impact": True},
    {"code": "KRW", "name": "South Korean Won", "symbol": "\u20a9", "supported_for_financial_impact": True},
    {"code": "ZAR", "name": "South African Rand", "symbol": "R", "supported_for_financial_impact": True},
    {"code": "SEK", "name": "Swedish Krona", "symbol": "kr", "supported_for_financial_impact": True},
    {"code": "NOK", "name": "Norwegian Krone", "symbol": "kr", "supported_for_financial_impact": True},
    {"code": "DKK", "name": "Danish Krone", "symbol": "kr", "supported_for_financial_impact": True},
    {"code": "MXN", "name": "Mexican Peso", "symbol": "$", "supported_for_financial_impact": True},
    {"code": "AED", "name": "UAE Dirham", "symbol": "AED", "supported_for_financial_impact": True},
    {"code": "SAR", "name": "Saudi Riyal", "symbol": "SAR", "supported_for_financial_impact": True},
]


# ---------------------------------------------------------------------------
# Static Reference Data -- Sectors
# ---------------------------------------------------------------------------

SUPPORTED_SECTORS: List[Dict[str, Any]] = [
    {
        "sector_id": "energy",
        "name": "Energy",
        "sub_sectors": [
            {"id": "oil_gas_upstream", "name": "Oil & Gas -- Upstream"},
            {"id": "oil_gas_midstream", "name": "Oil & Gas -- Midstream"},
            {"id": "oil_gas_downstream", "name": "Oil & Gas -- Downstream"},
            {"id": "oil_gas_integrated", "name": "Oil & Gas -- Integrated"},
            {"id": "coal_mining", "name": "Coal Mining"},
            {"id": "renewable_energy", "name": "Renewable Energy"},
            {"id": "electric_utilities", "name": "Electric Utilities"},
            {"id": "gas_utilities", "name": "Gas Utilities"},
        ],
        "tcfd_supplemental_guidance": True,
        "issb_industry_metrics": True,
    },
    {
        "sector_id": "materials_buildings",
        "name": "Materials & Buildings",
        "sub_sectors": [
            {"id": "metals_mining", "name": "Metals & Mining"},
            {"id": "chemicals", "name": "Chemicals"},
            {"id": "construction_materials", "name": "Construction Materials"},
            {"id": "capital_goods", "name": "Capital Goods"},
            {"id": "real_estate", "name": "Real Estate Management"},
        ],
        "tcfd_supplemental_guidance": True,
        "issb_industry_metrics": True,
    },
    {
        "sector_id": "transportation",
        "name": "Transportation",
        "sub_sectors": [
            {"id": "air_freight", "name": "Air Freight & Logistics"},
            {"id": "passenger_air", "name": "Passenger Air Transport"},
            {"id": "maritime", "name": "Maritime Transport"},
            {"id": "rail", "name": "Rail Transport"},
            {"id": "road_freight", "name": "Road Freight"},
            {"id": "automobiles", "name": "Automobiles & Components"},
        ],
        "tcfd_supplemental_guidance": True,
        "issb_industry_metrics": True,
    },
    {
        "sector_id": "agriculture_food_forest",
        "name": "Agriculture, Food & Forest Products",
        "sub_sectors": [
            {"id": "agricultural_products", "name": "Agricultural Products"},
            {"id": "meat_poultry_dairy", "name": "Meat, Poultry & Dairy"},
            {"id": "processed_foods", "name": "Processed Foods"},
            {"id": "beverages", "name": "Beverages"},
            {"id": "forestry", "name": "Forestry Management"},
            {"id": "paper_packaging", "name": "Paper & Packaging"},
        ],
        "tcfd_supplemental_guidance": True,
        "issb_industry_metrics": True,
    },
    {
        "sector_id": "financial",
        "name": "Financial Services",
        "sub_sectors": [
            {"id": "commercial_banking", "name": "Commercial Banking"},
            {"id": "investment_banking", "name": "Investment Banking"},
            {"id": "asset_management", "name": "Asset Management"},
            {"id": "insurance", "name": "Insurance"},
            {"id": "asset_owners", "name": "Asset Owners (Pension Funds)"},
        ],
        "tcfd_supplemental_guidance": True,
        "issb_industry_metrics": True,
    },
    {
        "sector_id": "technology_media",
        "name": "Technology & Media",
        "sub_sectors": [
            {"id": "software_services", "name": "Software & IT Services"},
            {"id": "hardware", "name": "Hardware & Semiconductors"},
            {"id": "telecommunications", "name": "Telecommunications"},
            {"id": "media_entertainment", "name": "Media & Entertainment"},
        ],
        "tcfd_supplemental_guidance": False,
        "issb_industry_metrics": True,
    },
    {
        "sector_id": "consumer_goods",
        "name": "Consumer Goods",
        "sub_sectors": [
            {"id": "apparel", "name": "Apparel & Textiles"},
            {"id": "household_products", "name": "Household Products"},
            {"id": "retail", "name": "Retail & Distribution"},
        ],
        "tcfd_supplemental_guidance": False,
        "issb_industry_metrics": True,
    },
    {
        "sector_id": "healthcare",
        "name": "Healthcare",
        "sub_sectors": [
            {"id": "pharmaceuticals", "name": "Pharmaceuticals"},
            {"id": "medical_devices", "name": "Medical Devices"},
            {"id": "healthcare_providers", "name": "Healthcare Providers"},
        ],
        "tcfd_supplemental_guidance": False,
        "issb_industry_metrics": True,
    },
]


# ---------------------------------------------------------------------------
# Static Reference Data -- Physical Hazard Types
# ---------------------------------------------------------------------------

HAZARD_TYPES: List[Dict[str, Any]] = [
    # Acute hazards
    {
        "hazard_id": "cyclone",
        "name": "Tropical Cyclone / Hurricane / Typhoon",
        "category": "acute",
        "description": "Intense rotating storm systems with sustained high winds, storm surge, and heavy rainfall.",
        "typical_metrics": ["wind_speed_kmh", "storm_surge_m", "return_period_years", "insured_loss_usd"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "flood_riverine",
        "name": "Riverine Flood",
        "category": "acute",
        "description": "Flooding caused by rivers or streams exceeding bank capacity due to prolonged or intense rainfall.",
        "typical_metrics": ["flood_depth_m", "return_period_years", "duration_days", "annualized_loss_usd"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "flood_coastal",
        "name": "Coastal Flood",
        "category": "acute",
        "description": "Inundation of coastal areas from storm surge, high tides, and sea level rise interaction.",
        "typical_metrics": ["inundation_depth_m", "return_period_years", "annualized_loss_usd"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "flood_pluvial",
        "name": "Pluvial (Flash) Flood",
        "category": "acute",
        "description": "Surface water flooding from intense precipitation exceeding local drainage capacity.",
        "typical_metrics": ["rainfall_intensity_mm_hr", "ponding_depth_m", "annualized_loss_usd"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "wildfire",
        "name": "Wildfire",
        "category": "acute",
        "description": "Uncontrolled fires in natural vegetation, exacerbated by drought and heat conditions.",
        "typical_metrics": ["fire_weather_index", "burn_probability_pct", "defensible_space_m", "annualized_loss_usd"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "extreme_heat",
        "name": "Extreme Heat Event",
        "category": "acute",
        "description": "Prolonged periods of abnormally high temperatures exceeding regional thresholds.",
        "typical_metrics": ["max_temperature_c", "heat_days_above_35c", "wet_bulb_temperature_c", "productivity_loss_pct"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "extreme_cold",
        "name": "Extreme Cold / Ice Storm",
        "category": "acute",
        "description": "Severe cold snaps, ice storms, and freezing events causing infrastructure damage.",
        "typical_metrics": ["min_temperature_c", "frost_days", "ice_thickness_mm", "annualized_loss_usd"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    # Chronic hazards
    {
        "hazard_id": "sea_level_rise",
        "name": "Sea Level Rise",
        "category": "chronic",
        "description": "Gradual increase in mean sea level due to thermal expansion and ice sheet melt.",
        "typical_metrics": ["slr_cm_by_2050", "slr_cm_by_2100", "permanent_inundation_risk", "asset_value_at_risk_usd"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "temperature_increase",
        "name": "Mean Temperature Increase",
        "category": "chronic",
        "description": "Long-term shift in average temperatures affecting ecosystems, agriculture, and energy demand.",
        "typical_metrics": ["delta_mean_temp_c", "cooling_degree_days", "heating_degree_days", "crop_yield_change_pct"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "precipitation_change",
        "name": "Precipitation Pattern Change",
        "category": "chronic",
        "description": "Shifts in rainfall patterns including increased drought frequency or intensified monsoons.",
        "typical_metrics": ["annual_precip_change_pct", "drought_frequency", "consecutive_dry_days", "water_stress_index"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "water_stress",
        "name": "Water Stress",
        "category": "chronic",
        "description": "Chronic shortage of freshwater supply relative to demand in a region.",
        "typical_metrics": ["baseline_water_stress", "future_water_stress", "water_depletion_pct", "cost_increase_pct"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
    {
        "hazard_id": "permafrost_thaw",
        "name": "Permafrost Thaw",
        "category": "chronic",
        "description": "Degradation of permanently frozen ground, causing subsidence and infrastructure instability.",
        "typical_metrics": ["active_layer_depth_m", "subsidence_risk", "foundation_damage_pct", "annualized_loss_usd"],
        "rcp_scenarios_available": ["rcp2.6", "rcp4.5", "rcp8.5"],
    },
]


# ---------------------------------------------------------------------------
# Static Reference Data -- Scenario Types
# ---------------------------------------------------------------------------

SCENARIO_TYPES: List[Dict[str, Any]] = [
    {
        "scenario_id": "iea_nze_2050",
        "name": "IEA Net Zero Emissions by 2050",
        "provider": "IEA",
        "warming_target": "1.5C",
        "description": "Aggressive pathway achieving net-zero CO2 emissions globally by 2050, aligned with limiting warming to 1.5C.",
        "time_horizon": "2050",
        "transition_risk_level": "very_high",
        "physical_risk_level": "low",
        "recommended_for": ["transition_risk_assessment", "carbon_price_projection", "stranded_asset_analysis"],
    },
    {
        "scenario_id": "iea_aps",
        "name": "IEA Announced Pledges Scenario",
        "provider": "IEA",
        "warming_target": "1.7C",
        "description": "Assumes all national climate pledges (NDCs) are met in full and on time.",
        "time_horizon": "2050",
        "transition_risk_level": "high",
        "physical_risk_level": "moderate",
        "recommended_for": ["policy_analysis", "regulatory_compliance", "investment_planning"],
    },
    {
        "scenario_id": "iea_steps",
        "name": "IEA Stated Policies Scenario",
        "provider": "IEA",
        "warming_target": "2.5C",
        "description": "Reflects existing policies and measures as of mid-2024, without assuming additional pledges are met.",
        "time_horizon": "2050",
        "transition_risk_level": "moderate",
        "physical_risk_level": "high",
        "recommended_for": ["baseline_projection", "current_trajectory_analysis"],
    },
    {
        "scenario_id": "ngfs_net_zero_2050",
        "name": "NGFS Net Zero 2050",
        "provider": "NGFS",
        "warming_target": "1.5C",
        "description": "Orderly transition with immediate, coordinated global action to limit warming to 1.5C by 2100.",
        "time_horizon": "2050",
        "transition_risk_level": "high",
        "physical_risk_level": "low",
        "recommended_for": ["financial_sector_stress_test", "portfolio_alignment", "carbon_pricing"],
    },
    {
        "scenario_id": "ngfs_below_2c",
        "name": "NGFS Below 2C",
        "provider": "NGFS",
        "warming_target": "1.7C",
        "description": "Orderly transition with gradual strengthening of climate policies to keep warming below 2C.",
        "time_horizon": "2050",
        "transition_risk_level": "moderate",
        "physical_risk_level": "moderate",
        "recommended_for": ["central_scenario_analysis", "regulatory_stress_test"],
    },
    {
        "scenario_id": "ngfs_delayed_transition",
        "name": "NGFS Delayed Transition",
        "provider": "NGFS",
        "warming_target": "1.8C",
        "description": "Disorderly transition -- policies delayed until 2030, then aggressive action with high short-term costs.",
        "time_horizon": "2050",
        "transition_risk_level": "very_high",
        "physical_risk_level": "moderate",
        "recommended_for": ["disorderly_scenario_stress_test", "abrupt_policy_impact"],
    },
    {
        "scenario_id": "ngfs_divergent_net_zero",
        "name": "NGFS Divergent Net Zero",
        "provider": "NGFS",
        "warming_target": "1.5C",
        "description": "Disorderly transition with uncoordinated sector-level policies leading to higher overall costs.",
        "time_horizon": "2050",
        "transition_risk_level": "very_high",
        "physical_risk_level": "low",
        "recommended_for": ["sector_policy_divergence", "cost_comparison"],
    },
    {
        "scenario_id": "ngfs_current_policies",
        "name": "NGFS Current Policies",
        "provider": "NGFS",
        "warming_target": "3C+",
        "description": "Hot-house world -- only currently implemented policies, leading to severe physical risks.",
        "time_horizon": "2100",
        "transition_risk_level": "low",
        "physical_risk_level": "very_high",
        "recommended_for": ["physical_risk_assessment", "worst_case_physical_scenario", "adaptation_planning"],
    },
    {
        "scenario_id": "ipcc_ssp1_1_9",
        "name": "IPCC SSP1-1.9",
        "provider": "IPCC",
        "warming_target": "1.5C",
        "description": "Sustainability-focused pathway (SSP1) with very low GHG forcing (1.9 W/m2) limiting warming to 1.5C.",
        "time_horizon": "2100",
        "transition_risk_level": "high",
        "physical_risk_level": "low",
        "recommended_for": ["long_term_physical_risk", "adaptation_assessment", "scientific_alignment"],
    },
    {
        "scenario_id": "ipcc_ssp5_8_5",
        "name": "IPCC SSP5-8.5",
        "provider": "IPCC",
        "warming_target": "4.4C",
        "description": "Fossil-fuelled development pathway (SSP5) with very high forcing (8.5 W/m2) -- worst-case warming.",
        "time_horizon": "2100",
        "transition_risk_level": "low",
        "physical_risk_level": "very_high",
        "recommended_for": ["extreme_physical_risk_stress_test", "upper_bound_adaptation"],
    },
]


# ---------------------------------------------------------------------------
# Static Reference Data -- Jurisdictions
# ---------------------------------------------------------------------------

JURISDICTION_CATALOG: List[Dict[str, Any]] = [
    {
        "jurisdiction_code": "UK",
        "jurisdiction_name": "United Kingdom",
        "region": "Europe",
        "regulations": [
            {"name": "FCA TCFD Rules", "effective_date": "2022-01-01", "mandatory": True},
            {"name": "Companies Act S414CB", "effective_date": "2022-04-06", "mandatory": True},
            {"name": "UK Sustainability Disclosure Standards (UK SDS)", "effective_date": "2025-01-01", "mandatory": True},
        ],
        "mandatory_tcfd": True,
        "issb_adopted": True,
        "additional_requirements": [
            "Premium listed companies",
            "Large private companies (500+ employees)",
            "Asset managers, owners, insurers",
        ],
    },
    {
        "jurisdiction_code": "EU",
        "jurisdiction_name": "European Union",
        "region": "Europe",
        "regulations": [
            {"name": "CSRD/ESRS E1", "effective_date": "2024-01-01", "mandatory": True},
            {"name": "EU Taxonomy Regulation", "effective_date": "2022-01-01", "mandatory": True},
            {"name": "SFDR", "effective_date": "2021-03-10", "mandatory": True},
        ],
        "mandatory_tcfd": False,
        "issb_adopted": False,
        "additional_requirements": [
            "CSRD applies to large companies and listed SMEs",
            "Double materiality assessment required",
            "EU Taxonomy alignment reporting",
        ],
    },
    {
        "jurisdiction_code": "US",
        "jurisdiction_name": "United States",
        "region": "Americas",
        "regulations": [
            {"name": "SEC Climate Disclosure Rule", "effective_date": "2024-03-06", "mandatory": True},
            {"name": "California SB 253 (Climate Corp Data Accountability)", "effective_date": "2026-01-01", "mandatory": True},
            {"name": "California SB 261 (Climate-Related Financial Risk)", "effective_date": "2026-01-01", "mandatory": True},
        ],
        "mandatory_tcfd": False,
        "issb_adopted": False,
        "additional_requirements": [
            "SEC rule applies to public companies",
            "California laws apply to companies doing business in CA above revenue thresholds",
            "Scope 3 phased in",
        ],
    },
    {
        "jurisdiction_code": "JP",
        "jurisdiction_name": "Japan",
        "region": "Asia Pacific",
        "regulations": [
            {"name": "SSBJ Standards (Japan ISSB equivalent)", "effective_date": "2025-04-01", "mandatory": True},
            {"name": "TSE Prime Market TCFD Disclosure", "effective_date": "2022-04-01", "mandatory": True},
        ],
        "mandatory_tcfd": True,
        "issb_adopted": True,
        "additional_requirements": [
            "TSE Prime Market listed companies",
            "SSBJ aligned with ISSB/IFRS S2",
        ],
    },
    {
        "jurisdiction_code": "AU",
        "jurisdiction_name": "Australia",
        "region": "Asia Pacific",
        "regulations": [
            {"name": "AASB Climate-related Financial Disclosures", "effective_date": "2025-01-01", "mandatory": True},
        ],
        "mandatory_tcfd": True,
        "issb_adopted": True,
        "additional_requirements": [
            "Phased implementation by entity size",
            "Large entities from 2025, medium from 2026, small from 2027",
        ],
    },
    {
        "jurisdiction_code": "SG",
        "jurisdiction_name": "Singapore",
        "region": "Asia Pacific",
        "regulations": [
            {"name": "SGX Climate Reporting", "effective_date": "2024-01-01", "mandatory": True},
        ],
        "mandatory_tcfd": True,
        "issb_adopted": True,
        "additional_requirements": [
            "SGX-listed companies on a comply-or-explain basis",
            "ISSB adoption planned",
        ],
    },
    {
        "jurisdiction_code": "CA",
        "jurisdiction_name": "Canada",
        "region": "Americas",
        "regulations": [
            {"name": "CSSB Standards (Canadian ISSB equivalent)", "effective_date": "2025-01-01", "mandatory": True},
            {"name": "CSA NI 51-107", "effective_date": "2024-01-01", "mandatory": True},
        ],
        "mandatory_tcfd": True,
        "issb_adopted": True,
        "additional_requirements": [
            "CSSB adopted ISSB standards with Canadian modifications",
            "Federally regulated financial institutions required",
        ],
    },
    {
        "jurisdiction_code": "HK",
        "jurisdiction_name": "Hong Kong",
        "region": "Asia Pacific",
        "regulations": [
            {"name": "HKEX ESG Reporting", "effective_date": "2025-01-01", "mandatory": True},
        ],
        "mandatory_tcfd": True,
        "issb_adopted": True,
        "additional_requirements": [
            "All HKEX-listed companies",
            "ISSB-aligned climate reporting mandatory from 2025",
        ],
    },
    {
        "jurisdiction_code": "BR",
        "jurisdiction_name": "Brazil",
        "region": "Americas",
        "regulations": [
            {"name": "CVM Resolution 193 (ISSB adoption)", "effective_date": "2026-01-01", "mandatory": True},
        ],
        "mandatory_tcfd": False,
        "issb_adopted": True,
        "additional_requirements": [
            "Publicly traded companies",
            "Voluntary from 2024, mandatory from 2026",
        ],
    },
    {
        "jurisdiction_code": "NZ",
        "jurisdiction_name": "New Zealand",
        "region": "Asia Pacific",
        "regulations": [
            {"name": "Financial Sector (Climate-related Disclosures) Amendment Act", "effective_date": "2023-01-01", "mandatory": True},
        ],
        "mandatory_tcfd": True,
        "issb_adopted": True,
        "additional_requirements": [
            "First country to mandate climate disclosures for financial entities",
            "Covers banks, insurers, fund managers, large listed companies",
        ],
    },
]


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_org_settings: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.utcnow()


def _default_settings(org_id: str) -> Dict[str, Any]:
    """Return default settings for an organization."""
    return {
        "org_id": org_id,
        "organization_name": "Default Organization",
        "industry": "General",
        "sector": "Diversified",
        "fiscal_year_end": FiscalYearEnd.DECEMBER.value,
        "reporting_period": ReportingPeriod.ANNUAL.value,
        "reporting_currency": Currency.USD.value,
        "emission_unit": EmissionUnit.TCO2E.value,
        "base_year": 2020,
        "materiality_threshold_usd": 1000000,
        "internal_carbon_price_usd": 50,
        "discount_rate_pct": 8.0,
        "primary_regulation": "tcfd",
        "risk_scoring_method": RiskScoringMethod.QUALITATIVE_5X5.value,
        "auto_generate_disclosures": True,
        "enable_issb_crosswalk": True,
        "enable_peer_benchmarking": True,
        "notification_email": None,
        "updated_at": _now(),
    }


# ---------------------------------------------------------------------------
# Endpoints -- Organisation Settings (spec endpoints 1 & 2)
# ---------------------------------------------------------------------------

@router.get(
    "/{org_id}",
    response_model=SettingsResponse,
    summary="Get organisation settings",
    description=(
        "Retrieve the current TCFD platform settings for an organisation "
        "including framework preferences, risk scoring method, scenario "
        "defaults, financial parameters, and notification preferences. "
        "Returns default settings if the organisation has not yet customised."
    ),
)
async def get_settings(org_id: str) -> SettingsResponse:
    """Get organization settings, creating defaults on first access."""
    if org_id not in _org_settings:
        _org_settings[org_id] = _default_settings(org_id)
    return SettingsResponse(**_org_settings[org_id])


@router.put(
    "/{org_id}",
    response_model=SettingsResponse,
    summary="Update organisation settings",
    description=(
        "Partially update TCFD platform settings for an organisation.  Only "
        "fields included in the request body are modified; all other settings "
        "retain their current values.  Validates currency and sector against "
        "supported catalogues."
    ),
)
async def update_settings(
    org_id: str,
    request: UpdateSettingsRequest,
) -> SettingsResponse:
    """Apply partial updates to organization settings."""
    if org_id not in _org_settings:
        _org_settings[org_id] = _default_settings(org_id)

    settings = _org_settings[org_id]
    updates = request.model_dump(exclude_unset=True)

    # Convert enums to string values for storage
    for key, value in updates.items():
        if hasattr(value, "value"):
            updates[key] = value.value

    # Validate currency if provided
    if "reporting_currency" in updates:
        valid_currencies = {c["code"] for c in SUPPORTED_CURRENCIES}
        if updates["reporting_currency"] not in valid_currencies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid currency '{updates['reporting_currency']}'. Use GET /currencies for valid codes.",
            )

    # Validate sector if provided
    if "sector" in updates:
        valid_sectors = {s["sector_id"] for s in SUPPORTED_SECTORS}
        valid_names = {s["name"] for s in SUPPORTED_SECTORS}
        if updates["sector"] not in valid_sectors and updates["sector"] not in valid_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sector '{updates['sector']}'. Use GET /sectors for valid values.",
            )

    settings.update(updates)
    settings["updated_at"] = _now()

    return SettingsResponse(**settings)


# ---------------------------------------------------------------------------
# Endpoints -- Reference Data: Currencies (spec endpoint 3)
# ---------------------------------------------------------------------------

@router.get(
    "/currencies",
    response_model=List[CurrencyInfo],
    summary="Supported currencies",
    description=(
        "List all ISO 4217 currencies supported by the TCFD platform for "
        "financial impact modelling, scenario analysis, and disclosure "
        "reporting.  Returns 20 major currencies."
    ),
)
async def get_supported_currencies() -> List[CurrencyInfo]:
    """Return the full catalogue of supported currencies."""
    return [CurrencyInfo(**c) for c in SUPPORTED_CURRENCIES]


# ---------------------------------------------------------------------------
# Endpoints -- Reference Data: Sectors (spec endpoint 4)
# ---------------------------------------------------------------------------

@router.get(
    "/sectors",
    response_model=List[SectorInfo],
    summary="Supported sectors",
    description=(
        "List all industry sectors supported by the TCFD platform, including "
        "sub-sector breakdowns, TCFD supplemental guidance availability, and "
        "ISSB industry-specific metric support.  Eight sector groups covering "
        "energy, materials, transportation, agriculture, financial services, "
        "technology, consumer goods, and healthcare."
    ),
)
async def get_supported_sectors() -> List[SectorInfo]:
    """Return the full sector taxonomy with sub-sectors and guidance flags."""
    return [SectorInfo(**s) for s in SUPPORTED_SECTORS]


# ---------------------------------------------------------------------------
# Endpoints -- Reference Data: Hazard Types (spec endpoint 5)
# ---------------------------------------------------------------------------

@router.get(
    "/hazard-types",
    response_model=List[HazardTypeInfo],
    summary="Physical hazard types",
    description=(
        "List all physical climate hazard types catalogued in the platform, "
        "categorised as acute (event-driven) or chronic (long-term shift). "
        "Includes 7 acute hazards (cyclone, riverine/coastal/pluvial flood, "
        "wildfire, extreme heat, extreme cold) and 5 chronic hazards (sea "
        "level rise, temperature increase, precipitation change, water "
        "stress, permafrost thaw).  Each entry lists typical measurement "
        "metrics and available RCP scenarios."
    ),
)
async def get_hazard_types(
    category: Optional[str] = Query(
        None,
        description="Filter by category: 'acute' or 'chronic'",
    ),
) -> List[HazardTypeInfo]:
    """Return hazard type catalogue, optionally filtered by category."""
    hazards = HAZARD_TYPES

    if category:
        cat_lower = category.lower()
        if cat_lower not in ("acute", "chronic"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid category '{category}'. Must be 'acute' or 'chronic'.",
            )
        hazards = [h for h in hazards if h["category"] == cat_lower]

    return [HazardTypeInfo(**h) for h in hazards]


# ---------------------------------------------------------------------------
# Endpoints -- Reference Data: Scenario Types (spec endpoint 6)
# ---------------------------------------------------------------------------

@router.get(
    "/scenario-types",
    response_model=List[ScenarioTypeInfo],
    summary="Available scenario types",
    description=(
        "List all pre-built climate scenario types available for scenario "
        "analysis.  Includes IEA (NZE 2050, APS, STEPS), NGFS (Net Zero "
        "2050, Below 2C, Delayed Transition, Divergent Net Zero, Current "
        "Policies), and IPCC (SSP1-1.9, SSP5-8.5) pathways.  Each entry "
        "describes the warming target, transition and physical risk levels, "
        "and recommended use cases."
    ),
)
async def get_scenario_types(
    provider: Optional[str] = Query(
        None,
        description="Filter by provider: 'IEA', 'NGFS', or 'IPCC'",
    ),
) -> List[ScenarioTypeInfo]:
    """Return scenario type catalogue, optionally filtered by provider."""
    scenarios = SCENARIO_TYPES

    if provider:
        provider_upper = provider.upper()
        valid_providers = sorted({s["provider"] for s in SCENARIO_TYPES})
        if provider_upper not in valid_providers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider '{provider}'. Valid providers: {valid_providers}",
            )
        scenarios = [s for s in scenarios if s["provider"] == provider_upper]

    return [ScenarioTypeInfo(**s) for s in scenarios]


# ---------------------------------------------------------------------------
# Endpoints -- Reference Data: Jurisdictions (bonus endpoint)
# ---------------------------------------------------------------------------

@router.get(
    "/jurisdictions",
    response_model=JurisdictionListResponse,
    summary="Supported jurisdictions",
    description=(
        "List all jurisdictions supported by the TCFD platform with their "
        "applicable regulations, TCFD mandatory status, and ISSB adoption "
        "status.  Covers 10 major jurisdictions: UK, EU, US, Japan, "
        "Australia, Singapore, Canada, Hong Kong, Brazil, New Zealand."
    ),
)
async def get_supported_jurisdictions(
    region: Optional[str] = Query(
        None,
        description="Filter by region: 'Europe', 'Americas', 'Asia Pacific'",
    ),
    mandatory_tcfd_only: bool = Query(
        False,
        description="Return only jurisdictions with mandatory TCFD requirements",
    ),
) -> JurisdictionListResponse:
    """List supported jurisdictions with optional filters."""
    jurisdictions = JURISDICTION_CATALOG

    if region:
        jurisdictions = [j for j in jurisdictions if j["region"].lower() == region.lower()]

    if mandatory_tcfd_only:
        jurisdictions = [j for j in jurisdictions if j["mandatory_tcfd"]]

    items = [JurisdictionResponse(**j) for j in jurisdictions]
    return JurisdictionListResponse(
        total_jurisdictions=len(items),
        jurisdictions=items,
        generated_at=_now(),
    )
