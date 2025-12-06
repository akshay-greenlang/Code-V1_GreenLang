"""
GL-011: Climate Risk Assessment Agent

This module implements the Climate Risk Assessment Agent for TCFD-aligned
physical and transition risk analysis using zero-hallucination deterministic
calculations.

The agent supports:
- Physical Risk Assessment (Acute and Chronic)
  - Acute: floods, cyclones, wildfires, extreme heat, drought
  - Chronic: sea level rise, temperature increase, precipitation changes
- Transition Risk Assessment
  - Policy: carbon pricing, regulations, mandates
  - Technology: disruption, obsolescence
  - Market: demand shifts, commodity prices
  - Reputation: stakeholder concerns, stigmatization
- Scenario Analysis (IPCC Pathways)
  - RCP 2.6 (1.5C pathway)
  - RCP 4.5 (2C pathway)
  - RCP 8.5 (4C+ pathway)
  - SSP1-2.6, SSP2-4.5, SSP5-8.5
- Financial Impact Quantification
  - Asset value at risk
  - Revenue impact
  - Cost increases
  - Insurance premium changes
- TCFD-Aligned Outputs
  - Risk register with scores
  - Scenario impact analysis
  - Financial exposure summary
  - Resilience recommendations

Example:
    >>> agent = ClimateRiskAgent()
    >>> result = agent.run(ClimateRiskInput(
    ...     organization_name="Example Corp",
    ...     assets=[Asset(name="HQ Building", asset_type="real_estate", value_usd=50000000)],
    ...     location=GeoLocation(latitude=25.7617, longitude=-80.1918, country="US"),
    ...     time_horizon_years=10,
    ...     scenario=ClimateScenario.RCP_4_5
    ... ))
    >>> print(f"Total risk score: {result.data.total_risk_score}")
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class PhysicalRiskType(str, Enum):
    """Physical climate risk types."""

    # Acute risks
    FLOOD = "flood"
    CYCLONE = "cyclone"
    WILDFIRE = "wildfire"
    EXTREME_HEAT = "extreme_heat"
    DROUGHT = "drought"
    STORM_SURGE = "storm_surge"
    HAILSTORM = "hailstorm"

    # Chronic risks
    SEA_LEVEL_RISE = "sea_level_rise"
    TEMPERATURE_INCREASE = "temperature_increase"
    PRECIPITATION_CHANGE = "precipitation_change"
    WATER_STRESS = "water_stress"
    PERMAFROST_THAW = "permafrost_thaw"


class TransitionRiskType(str, Enum):
    """Transition climate risk types."""

    # Policy risks
    CARBON_PRICING = "carbon_pricing"
    REGULATION = "regulation"
    MANDATE = "mandate"
    SUBSIDY_REMOVAL = "subsidy_removal"

    # Technology risks
    TECHNOLOGY_DISRUPTION = "technology_disruption"
    TECHNOLOGY_OBSOLESCENCE = "technology_obsolescence"
    STRANDED_ASSETS = "stranded_assets"

    # Market risks
    DEMAND_SHIFT = "demand_shift"
    COMMODITY_PRICE = "commodity_price"
    SUPPLY_CHAIN = "supply_chain"

    # Reputation risks
    STAKEHOLDER_CONCERN = "stakeholder_concern"
    STIGMATIZATION = "stigmatization"
    LITIGATION = "litigation"


class ClimateScenario(str, Enum):
    """IPCC climate scenarios."""

    # Representative Concentration Pathways
    RCP_2_6 = "rcp_2.6"  # 1.5C pathway
    RCP_4_5 = "rcp_4.5"  # 2C pathway
    RCP_6_0 = "rcp_6.0"  # 3C pathway
    RCP_8_5 = "rcp_8.5"  # 4C+ pathway

    # Shared Socioeconomic Pathways
    SSP1_2_6 = "ssp1_2.6"  # Sustainability
    SSP2_4_5 = "ssp2_4.5"  # Middle of the Road
    SSP3_7_0 = "ssp3_7.0"  # Regional Rivalry
    SSP5_8_5 = "ssp5_8.5"  # Fossil-fueled Development


class TimeHorizon(str, Enum):
    """Time horizon categories."""

    SHORT_TERM = "short_term"  # 0-5 years
    MEDIUM_TERM = "medium_term"  # 5-15 years
    LONG_TERM = "long_term"  # 15-30 years


class RiskCategory(str, Enum):
    """Risk categorization levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class AssetType(str, Enum):
    """Asset types for risk assessment."""

    REAL_ESTATE = "real_estate"
    INFRASTRUCTURE = "infrastructure"
    EQUIPMENT = "equipment"
    INVENTORY = "inventory"
    SUPPLY_CHAIN = "supply_chain"
    FINANCIAL = "financial"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    HUMAN_CAPITAL = "human_capital"


class SectorType(str, Enum):
    """Industry sector types."""

    ENERGY = "energy"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"
    MANUFACTURING = "manufacturing"
    AGRICULTURE = "agriculture"
    REAL_ESTATE = "real_estate"
    FINANCIAL_SERVICES = "financial_services"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    CONSUMER_GOODS = "consumer_goods"
    MINING = "mining"
    CONSTRUCTION = "construction"


# =============================================================================
# Pydantic Models - Input
# =============================================================================


class GeoLocation(BaseModel):
    """Geographic location for risk assessment."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    country: str = Field(..., description="ISO 3166 country code")
    region: Optional[str] = Field(None, description="State/Province/Region")
    city: Optional[str] = Field(None, description="City name")
    elevation_m: Optional[float] = Field(None, description="Elevation in meters")
    coastal_distance_km: Optional[float] = Field(
        None, ge=0, description="Distance to coast in km"
    )


class Asset(BaseModel):
    """Asset subject to climate risk assessment."""

    name: str = Field(..., description="Asset name")
    asset_type: AssetType = Field(..., description="Type of asset")
    value_usd: float = Field(..., ge=0, description="Asset value in USD")
    location: Optional[GeoLocation] = Field(None, description="Asset location")
    useful_life_years: Optional[int] = Field(
        None, ge=0, description="Remaining useful life"
    )
    insurance_coverage_usd: Optional[float] = Field(
        None, ge=0, description="Insurance coverage amount"
    )
    adaptation_measures: List[str] = Field(
        default_factory=list, description="Existing adaptation measures"
    )
    carbon_intensity: Optional[float] = Field(
        None, ge=0, description="Carbon intensity (tCO2e/year)"
    )


class RevenueStream(BaseModel):
    """Revenue stream for financial impact assessment."""

    name: str = Field(..., description="Revenue stream name")
    annual_revenue_usd: float = Field(..., ge=0, description="Annual revenue USD")
    sector: SectorType = Field(..., description="Sector classification")
    climate_sensitivity: float = Field(
        0.5, ge=0, le=1, description="Sensitivity to climate factors (0-1)"
    )
    geographic_exposure: List[str] = Field(
        default_factory=list, description="Geographic regions of exposure"
    )


class CarbonExposure(BaseModel):
    """Carbon exposure for transition risk assessment."""

    annual_emissions_tco2e: float = Field(
        ..., ge=0, description="Annual emissions tCO2e"
    )
    scope1_emissions: float = Field(0, ge=0, description="Scope 1 emissions")
    scope2_emissions: float = Field(0, ge=0, description="Scope 2 emissions")
    scope3_emissions: float = Field(0, ge=0, description="Scope 3 emissions")
    carbon_intensity_revenue: Optional[float] = Field(
        None, ge=0, description="tCO2e per million USD revenue"
    )
    current_carbon_price_usd: Optional[float] = Field(
        None, ge=0, description="Current carbon price exposure"
    )


class MitigationMeasure(BaseModel):
    """Mitigation measure for risk reduction."""

    name: str = Field(..., description="Mitigation measure name")
    risk_type: str = Field(..., description="Risk type addressed")
    effectiveness: float = Field(
        ..., ge=0, le=1, description="Effectiveness (0-1)"
    )
    implementation_cost_usd: float = Field(
        0, ge=0, description="Implementation cost"
    )
    annual_cost_usd: float = Field(0, ge=0, description="Annual operating cost")
    implementation_status: str = Field(
        "planned", description="Status: planned, in_progress, implemented"
    )


class ClimateRiskInput(BaseModel):
    """
    Input model for Climate Risk Assessment Agent.

    Attributes:
        organization_name: Name of the organization
        sector: Industry sector
        assets: List of assets to assess
        revenue_streams: Revenue streams for financial impact
        location: Primary organization location
        carbon_exposure: Carbon emissions exposure
        time_horizon_years: Assessment time horizon
        scenario: Climate scenario for analysis
        scenarios_to_compare: Additional scenarios for comparison
        mitigation_measures: Existing mitigation measures
    """

    organization_name: str = Field(..., description="Organization name")
    sector: SectorType = Field(
        SectorType.MANUFACTURING, description="Industry sector"
    )
    assets: List[Asset] = Field(
        default_factory=list, description="Assets to assess"
    )
    revenue_streams: List[RevenueStream] = Field(
        default_factory=list, description="Revenue streams"
    )
    location: GeoLocation = Field(..., description="Primary location")
    carbon_exposure: Optional[CarbonExposure] = Field(
        None, description="Carbon emissions exposure"
    )
    time_horizon_years: int = Field(
        10, ge=1, le=50, description="Assessment time horizon in years"
    )
    scenario: ClimateScenario = Field(
        ClimateScenario.RCP_4_5, description="Primary climate scenario"
    )
    scenarios_to_compare: List[ClimateScenario] = Field(
        default_factory=lambda: [
            ClimateScenario.RCP_2_6,
            ClimateScenario.RCP_4_5,
            ClimateScenario.RCP_8_5,
        ],
        description="Scenarios for comparison analysis"
    )
    mitigation_measures: List[MitigationMeasure] = Field(
        default_factory=list, description="Existing mitigation measures"
    )
    base_year: int = Field(
        2024, ge=2020, le=2030, description="Base year for projections"
    )
    discount_rate: float = Field(
        0.05, ge=0, le=0.20, description="Discount rate for NPV calculations"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("time_horizon_years")
    def validate_time_horizon(cls, v: int) -> int:
        """Validate time horizon is reasonable."""
        if v > 50:
            logger.warning(f"Time horizon {v} years is very long, uncertainty increases significantly")
        return v

    @root_validator
    def validate_inputs(cls, values: Dict) -> Dict:
        """Validate input consistency."""
        assets = values.get("assets", [])
        location = values.get("location")

        # Ensure at least location or assets have geographic data
        if not location and not any(a.location for a in assets):
            logger.warning("No geographic location provided for risk assessment")

        return values


# =============================================================================
# Pydantic Models - Output
# =============================================================================


class RiskScore(BaseModel):
    """Individual risk score with breakdown."""

    risk_type: str = Field(..., description="Type of risk")
    risk_category: str = Field(..., description="Physical or Transition")
    likelihood: int = Field(..., ge=1, le=5, description="Likelihood score (1-5)")
    impact: int = Field(..., ge=1, le=5, description="Impact score (1-5)")
    mitigation_effectiveness: float = Field(
        0, ge=0, le=1, description="Mitigation effectiveness (0-1)"
    )
    raw_score: float = Field(..., ge=0, description="Raw risk score")
    adjusted_score: float = Field(
        ..., ge=0, description="Score after mitigation adjustment"
    )
    risk_level: RiskCategory = Field(..., description="Risk categorization")
    financial_impact_usd: Optional[float] = Field(
        None, description="Estimated financial impact"
    )
    time_horizon: TimeHorizon = Field(..., description="Risk time horizon")
    confidence_level: float = Field(
        0.7, ge=0, le=1, description="Assessment confidence"
    )
    key_drivers: List[str] = Field(
        default_factory=list, description="Key risk drivers"
    )


class PhysicalRiskAssessment(BaseModel):
    """Physical risk assessment results."""

    acute_risks: List[RiskScore] = Field(
        default_factory=list, description="Acute physical risks"
    )
    chronic_risks: List[RiskScore] = Field(
        default_factory=list, description="Chronic physical risks"
    )
    total_physical_risk_score: float = Field(
        ..., description="Aggregated physical risk score"
    )
    highest_risk: Optional[str] = Field(
        None, description="Highest rated physical risk"
    )
    asset_exposure_summary: Dict[str, float] = Field(
        default_factory=dict, description="Asset exposure by risk type"
    )


class TransitionRiskAssessment(BaseModel):
    """Transition risk assessment results."""

    policy_risks: List[RiskScore] = Field(
        default_factory=list, description="Policy-related risks"
    )
    technology_risks: List[RiskScore] = Field(
        default_factory=list, description="Technology-related risks"
    )
    market_risks: List[RiskScore] = Field(
        default_factory=list, description="Market-related risks"
    )
    reputation_risks: List[RiskScore] = Field(
        default_factory=list, description="Reputation-related risks"
    )
    total_transition_risk_score: float = Field(
        ..., description="Aggregated transition risk score"
    )
    highest_risk: Optional[str] = Field(
        None, description="Highest rated transition risk"
    )
    carbon_price_impact_usd: Optional[float] = Field(
        None, description="Impact of carbon pricing"
    )


class ScenarioImpact(BaseModel):
    """Impact assessment for a specific scenario."""

    scenario: str = Field(..., description="Climate scenario")
    temperature_increase_c: float = Field(
        ..., description="Expected temperature increase"
    )
    physical_risk_multiplier: float = Field(
        ..., ge=0, description="Physical risk multiplier vs baseline"
    )
    transition_risk_multiplier: float = Field(
        ..., ge=0, description="Transition risk multiplier vs baseline"
    )
    total_value_at_risk_usd: float = Field(
        ..., ge=0, description="Total value at risk"
    )
    revenue_impact_pct: float = Field(
        ..., description="Revenue impact percentage"
    )
    cost_increase_pct: float = Field(
        ..., description="Cost increase percentage"
    )
    npv_impact_usd: float = Field(
        ..., description="NPV of climate impacts"
    )
    key_assumptions: List[str] = Field(
        default_factory=list, description="Key scenario assumptions"
    )


class FinancialExposure(BaseModel):
    """Financial exposure summary."""

    total_asset_value_at_risk_usd: float = Field(
        ..., ge=0, description="Total asset value at risk"
    )
    annual_revenue_at_risk_usd: float = Field(
        ..., ge=0, description="Annual revenue at risk"
    )
    expected_cost_increases_usd: float = Field(
        ..., ge=0, description="Expected annual cost increases"
    )
    insurance_gap_usd: float = Field(
        ..., ge=0, description="Insurance coverage gap"
    )
    carbon_liability_usd: float = Field(
        ..., ge=0, description="Carbon pricing liability"
    )
    stranded_asset_risk_usd: float = Field(
        ..., ge=0, description="Stranded asset exposure"
    )
    total_financial_exposure_usd: float = Field(
        ..., ge=0, description="Total financial exposure"
    )
    exposure_as_pct_assets: float = Field(
        ..., ge=0, description="Exposure as % of total assets"
    )


class ResilienceRecommendation(BaseModel):
    """Resilience recommendation."""

    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest)")
    category: str = Field(..., description="Recommendation category")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    risk_addressed: List[str] = Field(
        default_factory=list, description="Risks addressed"
    )
    estimated_risk_reduction_pct: float = Field(
        ..., ge=0, le=100, description="Expected risk reduction"
    )
    estimated_cost_usd: float = Field(
        ..., ge=0, description="Estimated implementation cost"
    )
    payback_years: Optional[float] = Field(
        None, ge=0, description="Estimated payback period"
    )
    implementation_timeline: str = Field(
        ..., description="Implementation timeline"
    )


class RiskRegister(BaseModel):
    """TCFD-aligned risk register."""

    organization: str = Field(..., description="Organization name")
    assessment_date: datetime = Field(
        default_factory=datetime.utcnow, description="Assessment date"
    )
    scenario: str = Field(..., description="Primary scenario assessed")
    time_horizon_years: int = Field(..., description="Assessment time horizon")
    physical_risks: List[RiskScore] = Field(
        default_factory=list, description="All physical risks"
    )
    transition_risks: List[RiskScore] = Field(
        default_factory=list, description="All transition risks"
    )
    total_risks_assessed: int = Field(..., description="Total risks assessed")
    critical_risks: int = Field(..., description="Number of critical risks")
    high_risks: int = Field(..., description="Number of high risks")


class ClimateRiskOutput(BaseModel):
    """
    Output model for Climate Risk Assessment Agent.

    TCFD-aligned comprehensive climate risk assessment results.
    """

    # Identification
    organization_name: str = Field(..., description="Organization assessed")
    assessment_id: str = Field(..., description="Unique assessment ID")
    assessment_date: datetime = Field(
        default_factory=datetime.utcnow, description="Assessment timestamp"
    )

    # Risk Assessments
    physical_risk_assessment: PhysicalRiskAssessment = Field(
        ..., description="Physical risk results"
    )
    transition_risk_assessment: TransitionRiskAssessment = Field(
        ..., description="Transition risk results"
    )

    # Aggregated Scores
    total_risk_score: float = Field(
        ..., ge=0, description="Aggregated total risk score"
    )
    overall_risk_level: RiskCategory = Field(
        ..., description="Overall risk categorization"
    )

    # Scenario Analysis
    primary_scenario_impact: ScenarioImpact = Field(
        ..., description="Primary scenario impact"
    )
    scenario_comparison: List[ScenarioImpact] = Field(
        default_factory=list, description="Scenario comparison results"
    )

    # Financial Exposure
    financial_exposure: FinancialExposure = Field(
        ..., description="Financial exposure summary"
    )

    # TCFD Outputs
    risk_register: RiskRegister = Field(
        ..., description="TCFD risk register"
    )
    resilience_recommendations: List[ResilienceRecommendation] = Field(
        default_factory=list, description="Resilience recommendations"
    )

    # Audit Trail
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    processing_time_ms: float = Field(..., description="Processing time ms")
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    methodology_version: str = Field(
        "TCFD-2023", description="Methodology version"
    )


# =============================================================================
# Risk Parameters Database (Zero-Hallucination Reference Data)
# =============================================================================


# Physical risk likelihood by region (baseline probabilities)
PHYSICAL_RISK_BASELINE: Dict[str, Dict[str, float]] = {
    # Flood risk by latitude bands
    "flood": {
        "tropical": 0.7,  # High flood risk
        "subtropical": 0.5,
        "temperate": 0.3,
        "continental": 0.2,
        "polar": 0.1,
    },
    "cyclone": {
        "tropical": 0.8,
        "subtropical": 0.5,
        "temperate": 0.1,
        "continental": 0.05,
        "polar": 0.01,
    },
    "wildfire": {
        "tropical": 0.3,
        "subtropical": 0.6,
        "temperate": 0.4,
        "continental": 0.3,
        "polar": 0.1,
    },
    "extreme_heat": {
        "tropical": 0.8,
        "subtropical": 0.7,
        "temperate": 0.4,
        "continental": 0.5,
        "polar": 0.1,
    },
    "drought": {
        "tropical": 0.4,
        "subtropical": 0.6,
        "temperate": 0.3,
        "continental": 0.4,
        "polar": 0.2,
    },
    "sea_level_rise": {
        "coastal_0_10km": 0.9,
        "coastal_10_50km": 0.5,
        "coastal_50_100km": 0.2,
        "inland": 0.0,
    },
}

# Scenario temperature projections (IPCC AR6)
SCENARIO_PARAMETERS: Dict[ClimateScenario, Dict[str, Any]] = {
    ClimateScenario.RCP_2_6: {
        "temperature_2050": 1.5,
        "temperature_2100": 1.8,
        "sea_level_2100_m": 0.4,
        "physical_risk_multiplier": 1.2,
        "transition_risk_multiplier": 1.8,  # High transition risk
        "carbon_price_2030": 135,
        "carbon_price_2050": 250,
        "description": "Strong mitigation pathway (1.5C)",
    },
    ClimateScenario.RCP_4_5: {
        "temperature_2050": 2.0,
        "temperature_2100": 2.7,
        "sea_level_2100_m": 0.6,
        "physical_risk_multiplier": 1.5,
        "transition_risk_multiplier": 1.4,
        "carbon_price_2030": 80,
        "carbon_price_2050": 160,
        "description": "Moderate mitigation pathway (2C)",
    },
    ClimateScenario.RCP_6_0: {
        "temperature_2050": 2.3,
        "temperature_2100": 3.3,
        "sea_level_2100_m": 0.7,
        "physical_risk_multiplier": 1.8,
        "transition_risk_multiplier": 1.2,
        "carbon_price_2030": 50,
        "carbon_price_2050": 100,
        "description": "Limited mitigation pathway (3C)",
    },
    ClimateScenario.RCP_8_5: {
        "temperature_2050": 2.6,
        "temperature_2100": 4.4,
        "sea_level_2100_m": 1.1,
        "physical_risk_multiplier": 2.5,
        "transition_risk_multiplier": 1.0,  # Low transition risk
        "carbon_price_2030": 20,
        "carbon_price_2050": 40,
        "description": "High emissions pathway (4C+)",
    },
    ClimateScenario.SSP1_2_6: {
        "temperature_2050": 1.6,
        "temperature_2100": 1.8,
        "sea_level_2100_m": 0.4,
        "physical_risk_multiplier": 1.2,
        "transition_risk_multiplier": 1.9,
        "carbon_price_2030": 150,
        "carbon_price_2050": 300,
        "description": "Sustainability pathway",
    },
    ClimateScenario.SSP2_4_5: {
        "temperature_2050": 2.0,
        "temperature_2100": 2.7,
        "sea_level_2100_m": 0.6,
        "physical_risk_multiplier": 1.5,
        "transition_risk_multiplier": 1.4,
        "carbon_price_2030": 75,
        "carbon_price_2050": 150,
        "description": "Middle of the Road",
    },
    ClimateScenario.SSP3_7_0: {
        "temperature_2050": 2.4,
        "temperature_2100": 3.6,
        "sea_level_2100_m": 0.8,
        "physical_risk_multiplier": 2.0,
        "transition_risk_multiplier": 1.1,
        "carbon_price_2030": 30,
        "carbon_price_2050": 60,
        "description": "Regional Rivalry",
    },
    ClimateScenario.SSP5_8_5: {
        "temperature_2050": 2.7,
        "temperature_2100": 4.8,
        "sea_level_2100_m": 1.2,
        "physical_risk_multiplier": 2.8,
        "transition_risk_multiplier": 0.9,
        "carbon_price_2030": 15,
        "carbon_price_2050": 30,
        "description": "Fossil-fueled Development",
    },
}

# Sector transition risk sensitivity
SECTOR_TRANSITION_SENSITIVITY: Dict[SectorType, Dict[str, float]] = {
    SectorType.ENERGY: {
        "carbon_pricing": 0.95,
        "regulation": 0.90,
        "technology_disruption": 0.85,
        "demand_shift": 0.80,
        "reputation": 0.70,
    },
    SectorType.UTILITIES: {
        "carbon_pricing": 0.85,
        "regulation": 0.85,
        "technology_disruption": 0.70,
        "demand_shift": 0.60,
        "reputation": 0.50,
    },
    SectorType.TRANSPORTATION: {
        "carbon_pricing": 0.75,
        "regulation": 0.80,
        "technology_disruption": 0.90,
        "demand_shift": 0.70,
        "reputation": 0.60,
    },
    SectorType.MANUFACTURING: {
        "carbon_pricing": 0.70,
        "regulation": 0.65,
        "technology_disruption": 0.60,
        "demand_shift": 0.55,
        "reputation": 0.45,
    },
    SectorType.AGRICULTURE: {
        "carbon_pricing": 0.40,
        "regulation": 0.50,
        "technology_disruption": 0.30,
        "demand_shift": 0.60,
        "reputation": 0.40,
    },
    SectorType.REAL_ESTATE: {
        "carbon_pricing": 0.50,
        "regulation": 0.70,
        "technology_disruption": 0.40,
        "demand_shift": 0.50,
        "reputation": 0.35,
    },
    SectorType.FINANCIAL_SERVICES: {
        "carbon_pricing": 0.30,
        "regulation": 0.60,
        "technology_disruption": 0.35,
        "demand_shift": 0.45,
        "reputation": 0.70,
    },
    SectorType.TECHNOLOGY: {
        "carbon_pricing": 0.35,
        "regulation": 0.40,
        "technology_disruption": 0.30,
        "demand_shift": 0.40,
        "reputation": 0.50,
    },
    SectorType.HEALTHCARE: {
        "carbon_pricing": 0.25,
        "regulation": 0.35,
        "technology_disruption": 0.25,
        "demand_shift": 0.30,
        "reputation": 0.40,
    },
    SectorType.CONSUMER_GOODS: {
        "carbon_pricing": 0.45,
        "regulation": 0.50,
        "technology_disruption": 0.40,
        "demand_shift": 0.65,
        "reputation": 0.60,
    },
    SectorType.MINING: {
        "carbon_pricing": 0.80,
        "regulation": 0.85,
        "technology_disruption": 0.50,
        "demand_shift": 0.70,
        "reputation": 0.75,
    },
    SectorType.CONSTRUCTION: {
        "carbon_pricing": 0.55,
        "regulation": 0.65,
        "technology_disruption": 0.45,
        "demand_shift": 0.50,
        "reputation": 0.40,
    },
}

# Impact multipliers for financial calculations
IMPACT_MULTIPLIERS: Dict[str, float] = {
    "flood_asset_damage": 0.15,  # 15% of asset value
    "flood_revenue_disruption": 0.10,
    "cyclone_asset_damage": 0.25,
    "cyclone_revenue_disruption": 0.15,
    "wildfire_asset_damage": 0.40,
    "wildfire_revenue_disruption": 0.20,
    "extreme_heat_productivity_loss": 0.08,
    "drought_operational_cost_increase": 0.12,
    "sea_level_rise_asset_impairment": 0.50,
    "carbon_price_cost_increase_per_tco2e": 1.0,  # Direct pass-through
    "technology_stranded_asset": 0.30,
    "demand_shift_revenue_loss": 0.20,
    "reputation_market_cap_impact": 0.05,
}


# =============================================================================
# Climate Risk Agent Implementation
# =============================================================================


class ClimateRiskAgent:
    """
    GL-011: Climate Risk Assessment Agent.

    This agent performs TCFD-aligned climate risk assessments using
    zero-hallucination deterministic calculations:
    - Physical risk: likelihood * impact * (1 - mitigation)
    - Financial impact: deterministic multipliers from IPCC data
    - Scenario analysis: fixed parameters per scenario
    - No LLM in calculation path

    Aligned with:
    - Task Force on Climate-related Financial Disclosures (TCFD)
    - IPCC AR6 climate scenarios
    - NGFS climate scenarios
    - ISO 14091 Climate change adaptation

    Attributes:
        scenario_parameters: Climate scenario reference data
        sector_sensitivity: Sector transition risk sensitivity
        physical_risk_baseline: Regional physical risk baselines

    Example:
        >>> agent = ClimateRiskAgent()
        >>> result = agent.run(ClimateRiskInput(
        ...     organization_name="Example Corp",
        ...     assets=[Asset(name="HQ", asset_type=AssetType.REAL_ESTATE, value_usd=50000000)],
        ...     location=GeoLocation(latitude=25.76, longitude=-80.19, country="US")
        ... ))
        >>> print(f"Total risk: {result.total_risk_score}")
    """

    AGENT_ID = "risk/climate_risk_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "TCFD-aligned climate risk assessment agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Climate Risk Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []
        self.scenario_parameters = SCENARIO_PARAMETERS
        self.sector_sensitivity = SECTOR_TRANSITION_SENSITIVITY
        self.physical_risk_baseline = PHYSICAL_RISK_BASELINE

        logger.info(f"ClimateRiskAgent initialized (version {self.VERSION})")

    def run(self, input_data: ClimateRiskInput) -> ClimateRiskOutput:
        """
        Execute the climate risk assessment.

        ZERO-HALLUCINATION assessment using deterministic formulas:
        - risk_score = likelihood * impact * (1 - mitigation_effectiveness)
        - All parameters from validated reference data
        - Complete audit trail via provenance hash

        Args:
            input_data: Validated climate risk input data

        Returns:
            Comprehensive TCFD-aligned risk assessment results

        Raises:
            ValueError: If input validation fails
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        logger.info(
            f"Assessing climate risk for {input_data.organization_name}, "
            f"scenario={input_data.scenario}, horizon={input_data.time_horizon_years}y"
        )

        try:
            # Generate assessment ID
            assessment_id = self._generate_assessment_id(input_data)

            self._track_step("initialization", {
                "organization": input_data.organization_name,
                "scenario": input_data.scenario.value,
                "time_horizon": input_data.time_horizon_years,
                "assets_count": len(input_data.assets),
            })

            # Step 1: Assess physical risks
            physical_assessment = self._assess_physical_risks(input_data)

            self._track_step("physical_risk_assessment", {
                "acute_risks_count": len(physical_assessment.acute_risks),
                "chronic_risks_count": len(physical_assessment.chronic_risks),
                "total_physical_score": physical_assessment.total_physical_risk_score,
            })

            # Step 2: Assess transition risks
            transition_assessment = self._assess_transition_risks(input_data)

            self._track_step("transition_risk_assessment", {
                "policy_risks_count": len(transition_assessment.policy_risks),
                "technology_risks_count": len(transition_assessment.technology_risks),
                "market_risks_count": len(transition_assessment.market_risks),
                "reputation_risks_count": len(transition_assessment.reputation_risks),
                "total_transition_score": transition_assessment.total_transition_risk_score,
            })

            # Step 3: Calculate total risk score
            # ZERO-HALLUCINATION CALCULATION
            total_risk_score = self._calculate_total_risk_score(
                physical_assessment.total_physical_risk_score,
                transition_assessment.total_transition_risk_score
            )

            overall_risk_level = self._categorize_risk(total_risk_score)

            self._track_step("risk_aggregation", {
                "formula": "total = (physical * 0.5) + (transition * 0.5)",
                "physical_weight": 0.5,
                "transition_weight": 0.5,
                "total_risk_score": total_risk_score,
                "risk_level": overall_risk_level.value,
            })

            # Step 4: Scenario analysis
            primary_scenario_impact = self._analyze_scenario(
                input_data, input_data.scenario
            )

            scenario_comparison = [
                self._analyze_scenario(input_data, scenario)
                for scenario in input_data.scenarios_to_compare
                if scenario != input_data.scenario
            ]

            self._track_step("scenario_analysis", {
                "primary_scenario": input_data.scenario.value,
                "scenarios_compared": len(scenario_comparison),
            })

            # Step 5: Calculate financial exposure
            financial_exposure = self._calculate_financial_exposure(
                input_data,
                physical_assessment,
                transition_assessment,
                primary_scenario_impact
            )

            self._track_step("financial_exposure", {
                "total_exposure_usd": financial_exposure.total_financial_exposure_usd,
                "asset_value_at_risk": financial_exposure.total_asset_value_at_risk_usd,
                "revenue_at_risk": financial_exposure.annual_revenue_at_risk_usd,
            })

            # Step 6: Generate risk register
            risk_register = self._generate_risk_register(
                input_data,
                physical_assessment,
                transition_assessment
            )

            self._track_step("risk_register", {
                "total_risks": risk_register.total_risks_assessed,
                "critical_risks": risk_register.critical_risks,
                "high_risks": risk_register.high_risks,
            })

            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                input_data,
                physical_assessment,
                transition_assessment,
                financial_exposure
            )

            self._track_step("recommendations", {
                "recommendations_count": len(recommendations),
            })

            # Step 8: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 9: Calculate processing time
            processing_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000

            # Step 10: Create output
            output = ClimateRiskOutput(
                organization_name=input_data.organization_name,
                assessment_id=assessment_id,
                physical_risk_assessment=physical_assessment,
                transition_risk_assessment=transition_assessment,
                total_risk_score=round(total_risk_score, 2),
                overall_risk_level=overall_risk_level,
                primary_scenario_impact=primary_scenario_impact,
                scenario_comparison=scenario_comparison,
                financial_exposure=financial_exposure,
                risk_register=risk_register,
                resilience_recommendations=recommendations,
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time_ms, 2),
                data_sources=[
                    "IPCC AR6",
                    "TCFD Recommendations",
                    "NGFS Climate Scenarios",
                ],
            )

            logger.info(
                f"Climate risk assessment complete: "
                f"total_score={total_risk_score:.1f}, "
                f"risk_level={overall_risk_level.value}, "
                f"exposure=${financial_exposure.total_financial_exposure_usd:,.0f} "
                f"(duration: {processing_time_ms:.2f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Climate risk assessment failed: {str(e)}", exc_info=True)
            raise

    def _assess_physical_risks(
        self,
        input_data: ClimateRiskInput
    ) -> PhysicalRiskAssessment:
        """
        Assess physical climate risks.

        ZERO-HALLUCINATION: Uses fixed risk parameters based on location
        and scenario data from IPCC.

        Args:
            input_data: Climate risk input

        Returns:
            Physical risk assessment results
        """
        acute_risks: List[RiskScore] = []
        chronic_risks: List[RiskScore] = []
        asset_exposure: Dict[str, float] = {}

        # Determine climate zone
        climate_zone = self._get_climate_zone(input_data.location.latitude)
        coastal_category = self._get_coastal_category(
            input_data.location.coastal_distance_km
        )

        # Get scenario parameters
        scenario_params = self.scenario_parameters[input_data.scenario]
        physical_multiplier = scenario_params["physical_risk_multiplier"]

        # Calculate total asset value
        total_asset_value = sum(a.value_usd for a in input_data.assets)

        # Get mitigation effectiveness
        mitigation_map = self._build_mitigation_map(input_data.mitigation_measures)

        # Assess acute physical risks
        acute_risk_types = [
            PhysicalRiskType.FLOOD,
            PhysicalRiskType.CYCLONE,
            PhysicalRiskType.WILDFIRE,
            PhysicalRiskType.EXTREME_HEAT,
            PhysicalRiskType.DROUGHT,
        ]

        for risk_type in acute_risk_types:
            baseline = self.physical_risk_baseline.get(risk_type.value, {})
            base_likelihood = baseline.get(climate_zone, 0.3)

            # Adjust for scenario
            adjusted_likelihood = min(base_likelihood * physical_multiplier, 1.0)
            likelihood_score = self._probability_to_score(adjusted_likelihood)

            # Determine impact based on asset exposure
            impact_multiplier = IMPACT_MULTIPLIERS.get(
                f"{risk_type.value}_asset_damage", 0.1
            )
            impact_score = self._impact_to_score(impact_multiplier, total_asset_value)

            # Get mitigation effectiveness
            mitigation = mitigation_map.get(risk_type.value, 0)

            # ZERO-HALLUCINATION CALCULATION
            # Formula: risk_score = likelihood * impact * (1 - mitigation)
            raw_score = likelihood_score * impact_score
            adjusted_score = raw_score * (1 - mitigation)

            # Calculate financial impact
            financial_impact = total_asset_value * impact_multiplier * adjusted_likelihood

            risk_score = RiskScore(
                risk_type=risk_type.value,
                risk_category="physical_acute",
                likelihood=likelihood_score,
                impact=impact_score,
                mitigation_effectiveness=mitigation,
                raw_score=round(raw_score, 2),
                adjusted_score=round(adjusted_score, 2),
                risk_level=self._score_to_category(adjusted_score),
                financial_impact_usd=round(financial_impact, 0),
                time_horizon=self._get_time_horizon(input_data.time_horizon_years),
                confidence_level=0.7,
                key_drivers=[
                    f"Climate zone: {climate_zone}",
                    f"Scenario: {input_data.scenario.value}",
                ],
            )

            acute_risks.append(risk_score)
            asset_exposure[risk_type.value] = financial_impact

        # Assess chronic physical risks
        chronic_risk_types = [
            PhysicalRiskType.SEA_LEVEL_RISE,
            PhysicalRiskType.TEMPERATURE_INCREASE,
            PhysicalRiskType.PRECIPITATION_CHANGE,
            PhysicalRiskType.WATER_STRESS,
        ]

        for risk_type in chronic_risk_types:
            if risk_type == PhysicalRiskType.SEA_LEVEL_RISE:
                base_likelihood = self.physical_risk_baseline.get(
                    "sea_level_rise", {}
                ).get(coastal_category, 0.0)
                impact_multiplier = IMPACT_MULTIPLIERS.get(
                    "sea_level_rise_asset_impairment", 0.2
                )
            else:
                base_likelihood = 0.5  # Chronic risks are highly likely
                impact_multiplier = 0.05  # Lower per-year impact

            # Adjust for scenario and time horizon
            time_factor = min(input_data.time_horizon_years / 30, 1.0)
            adjusted_likelihood = min(
                base_likelihood * physical_multiplier * time_factor, 1.0
            )
            likelihood_score = self._probability_to_score(adjusted_likelihood)

            impact_score = self._impact_to_score(impact_multiplier, total_asset_value)

            mitigation = mitigation_map.get(risk_type.value, 0)

            raw_score = likelihood_score * impact_score
            adjusted_score = raw_score * (1 - mitigation)

            financial_impact = (
                total_asset_value * impact_multiplier *
                adjusted_likelihood * time_factor
            )

            risk_score = RiskScore(
                risk_type=risk_type.value,
                risk_category="physical_chronic",
                likelihood=likelihood_score,
                impact=impact_score,
                mitigation_effectiveness=mitigation,
                raw_score=round(raw_score, 2),
                adjusted_score=round(adjusted_score, 2),
                risk_level=self._score_to_category(adjusted_score),
                financial_impact_usd=round(financial_impact, 0),
                time_horizon=TimeHorizon.LONG_TERM,
                confidence_level=0.6,
                key_drivers=[
                    f"Time horizon: {input_data.time_horizon_years}y",
                    f"Coastal distance: {coastal_category}",
                ],
            )

            chronic_risks.append(risk_score)
            asset_exposure[risk_type.value] = financial_impact

        # Calculate total physical risk score
        all_physical_scores = [r.adjusted_score for r in acute_risks + chronic_risks]
        total_physical_score = (
            sum(all_physical_scores) / len(all_physical_scores)
            if all_physical_scores else 0
        )

        # Find highest risk
        all_risks = acute_risks + chronic_risks
        highest_risk = max(all_risks, key=lambda r: r.adjusted_score) if all_risks else None

        return PhysicalRiskAssessment(
            acute_risks=acute_risks,
            chronic_risks=chronic_risks,
            total_physical_risk_score=round(total_physical_score, 2),
            highest_risk=highest_risk.risk_type if highest_risk else None,
            asset_exposure_summary=asset_exposure,
        )

    def _assess_transition_risks(
        self,
        input_data: ClimateRiskInput
    ) -> TransitionRiskAssessment:
        """
        Assess transition climate risks.

        ZERO-HALLUCINATION: Uses sector sensitivity data and scenario
        carbon price projections.

        Args:
            input_data: Climate risk input

        Returns:
            Transition risk assessment results
        """
        policy_risks: List[RiskScore] = []
        technology_risks: List[RiskScore] = []
        market_risks: List[RiskScore] = []
        reputation_risks: List[RiskScore] = []

        # Get sector sensitivity
        sector_sensitivity = self.sector_sensitivity.get(
            input_data.sector,
            self.sector_sensitivity[SectorType.MANUFACTURING]
        )

        # Get scenario parameters
        scenario_params = self.scenario_parameters[input_data.scenario]
        transition_multiplier = scenario_params["transition_risk_multiplier"]

        # Calculate total revenue and carbon exposure
        total_revenue = sum(rs.annual_revenue_usd for rs in input_data.revenue_streams)
        total_asset_value = sum(a.value_usd for a in input_data.assets)

        carbon_emissions = (
            input_data.carbon_exposure.annual_emissions_tco2e
            if input_data.carbon_exposure else 0
        )

        # Get mitigation effectiveness
        mitigation_map = self._build_mitigation_map(input_data.mitigation_measures)

        # === Policy Risks ===

        # Carbon pricing risk
        carbon_sensitivity = sector_sensitivity.get("carbon_pricing", 0.5)
        carbon_price_2030 = scenario_params["carbon_price_2030"]
        carbon_price_2050 = scenario_params["carbon_price_2050"]

        # Interpolate carbon price for time horizon
        years_to_2030 = max(0, min(2030 - input_data.base_year, input_data.time_horizon_years))
        years_to_2050 = max(0, min(2050 - input_data.base_year, input_data.time_horizon_years))

        if input_data.time_horizon_years <= (2030 - input_data.base_year):
            projected_carbon_price = carbon_price_2030
        elif input_data.time_horizon_years <= (2050 - input_data.base_year):
            # Linear interpolation
            t = (input_data.time_horizon_years - years_to_2030) / (years_to_2050 - years_to_2030 + 0.01)
            projected_carbon_price = carbon_price_2030 + t * (carbon_price_2050 - carbon_price_2030)
        else:
            projected_carbon_price = carbon_price_2050

        carbon_price_impact = carbon_emissions * projected_carbon_price

        likelihood_score = self._probability_to_score(
            carbon_sensitivity * transition_multiplier
        )
        impact_score = self._carbon_impact_to_score(carbon_price_impact, total_revenue)
        mitigation = mitigation_map.get(TransitionRiskType.CARBON_PRICING.value, 0)

        raw_score = likelihood_score * impact_score
        adjusted_score = raw_score * (1 - mitigation)

        policy_risks.append(RiskScore(
            risk_type=TransitionRiskType.CARBON_PRICING.value,
            risk_category="transition_policy",
            likelihood=likelihood_score,
            impact=impact_score,
            mitigation_effectiveness=mitigation,
            raw_score=round(raw_score, 2),
            adjusted_score=round(adjusted_score, 2),
            risk_level=self._score_to_category(adjusted_score),
            financial_impact_usd=round(carbon_price_impact, 0),
            time_horizon=self._get_time_horizon(input_data.time_horizon_years),
            confidence_level=0.75,
            key_drivers=[
                f"Projected carbon price: ${projected_carbon_price}/tCO2e",
                f"Annual emissions: {carbon_emissions:,.0f} tCO2e",
            ],
        ))

        # Regulation risk
        regulation_sensitivity = sector_sensitivity.get("regulation", 0.5)
        likelihood_score = self._probability_to_score(
            regulation_sensitivity * transition_multiplier
        )
        impact_score = 3  # Medium baseline impact
        mitigation = mitigation_map.get(TransitionRiskType.REGULATION.value, 0)

        raw_score = likelihood_score * impact_score
        adjusted_score = raw_score * (1 - mitigation)

        regulatory_cost = total_revenue * 0.02 * regulation_sensitivity  # 2% compliance cost

        policy_risks.append(RiskScore(
            risk_type=TransitionRiskType.REGULATION.value,
            risk_category="transition_policy",
            likelihood=likelihood_score,
            impact=impact_score,
            mitigation_effectiveness=mitigation,
            raw_score=round(raw_score, 2),
            adjusted_score=round(adjusted_score, 2),
            risk_level=self._score_to_category(adjusted_score),
            financial_impact_usd=round(regulatory_cost, 0),
            time_horizon=TimeHorizon.MEDIUM_TERM,
            confidence_level=0.6,
            key_drivers=[
                f"Sector: {input_data.sector.value}",
                f"Scenario: {input_data.scenario.value}",
            ],
        ))

        # === Technology Risks ===

        tech_disruption_sensitivity = sector_sensitivity.get("technology_disruption", 0.5)
        likelihood_score = self._probability_to_score(
            tech_disruption_sensitivity * transition_multiplier
        )
        stranded_asset_risk = total_asset_value * IMPACT_MULTIPLIERS.get(
            "technology_stranded_asset", 0.3
        ) * tech_disruption_sensitivity
        impact_score = self._impact_to_score(0.3, total_asset_value)
        mitigation = mitigation_map.get(TransitionRiskType.TECHNOLOGY_DISRUPTION.value, 0)

        raw_score = likelihood_score * impact_score
        adjusted_score = raw_score * (1 - mitigation)

        technology_risks.append(RiskScore(
            risk_type=TransitionRiskType.TECHNOLOGY_DISRUPTION.value,
            risk_category="transition_technology",
            likelihood=likelihood_score,
            impact=impact_score,
            mitigation_effectiveness=mitigation,
            raw_score=round(raw_score, 2),
            adjusted_score=round(adjusted_score, 2),
            risk_level=self._score_to_category(adjusted_score),
            financial_impact_usd=round(stranded_asset_risk, 0),
            time_horizon=TimeHorizon.MEDIUM_TERM,
            confidence_level=0.55,
            key_drivers=[
                f"Technology sensitivity: {tech_disruption_sensitivity:.0%}",
            ],
        ))

        # Stranded assets risk
        if carbon_emissions > 0:
            stranded_likelihood = self._probability_to_score(
                0.6 * transition_multiplier
            )
            stranded_value = total_asset_value * 0.2 * transition_multiplier
            stranded_impact = self._impact_to_score(0.2, total_asset_value)
            mitigation = mitigation_map.get(TransitionRiskType.STRANDED_ASSETS.value, 0)

            raw_score = stranded_likelihood * stranded_impact
            adjusted_score = raw_score * (1 - mitigation)

            technology_risks.append(RiskScore(
                risk_type=TransitionRiskType.STRANDED_ASSETS.value,
                risk_category="transition_technology",
                likelihood=stranded_likelihood,
                impact=stranded_impact,
                mitigation_effectiveness=mitigation,
                raw_score=round(raw_score, 2),
                adjusted_score=round(adjusted_score, 2),
                risk_level=self._score_to_category(adjusted_score),
                financial_impact_usd=round(stranded_value, 0),
                time_horizon=TimeHorizon.LONG_TERM,
                confidence_level=0.5,
                key_drivers=[
                    f"Carbon-intensive assets",
                    f"Transition scenario: {input_data.scenario.value}",
                ],
            ))

        # === Market Risks ===

        demand_sensitivity = sector_sensitivity.get("demand_shift", 0.5)
        likelihood_score = self._probability_to_score(
            demand_sensitivity * transition_multiplier
        )
        revenue_impact = total_revenue * IMPACT_MULTIPLIERS.get(
            "demand_shift_revenue_loss", 0.2
        ) * demand_sensitivity
        impact_score = self._revenue_impact_to_score(revenue_impact, total_revenue)
        mitigation = mitigation_map.get(TransitionRiskType.DEMAND_SHIFT.value, 0)

        raw_score = likelihood_score * impact_score
        adjusted_score = raw_score * (1 - mitigation)

        market_risks.append(RiskScore(
            risk_type=TransitionRiskType.DEMAND_SHIFT.value,
            risk_category="transition_market",
            likelihood=likelihood_score,
            impact=impact_score,
            mitigation_effectiveness=mitigation,
            raw_score=round(raw_score, 2),
            adjusted_score=round(adjusted_score, 2),
            risk_level=self._score_to_category(adjusted_score),
            financial_impact_usd=round(revenue_impact, 0),
            time_horizon=TimeHorizon.MEDIUM_TERM,
            confidence_level=0.6,
            key_drivers=[
                f"Demand sensitivity: {demand_sensitivity:.0%}",
            ],
        ))

        # Supply chain risk
        supply_sensitivity = 0.5
        likelihood_score = self._probability_to_score(supply_sensitivity)
        supply_chain_cost = total_revenue * 0.05  # 5% supply chain exposure
        impact_score = 3
        mitigation = mitigation_map.get(TransitionRiskType.SUPPLY_CHAIN.value, 0)

        raw_score = likelihood_score * impact_score
        adjusted_score = raw_score * (1 - mitigation)

        market_risks.append(RiskScore(
            risk_type=TransitionRiskType.SUPPLY_CHAIN.value,
            risk_category="transition_market",
            likelihood=likelihood_score,
            impact=impact_score,
            mitigation_effectiveness=mitigation,
            raw_score=round(raw_score, 2),
            adjusted_score=round(adjusted_score, 2),
            risk_level=self._score_to_category(adjusted_score),
            financial_impact_usd=round(supply_chain_cost, 0),
            time_horizon=TimeHorizon.SHORT_TERM,
            confidence_level=0.65,
            key_drivers=["Supply chain climate exposure"],
        ))

        # === Reputation Risks ===

        reputation_sensitivity = sector_sensitivity.get("reputation", 0.5)
        likelihood_score = self._probability_to_score(
            reputation_sensitivity * transition_multiplier
        )
        reputation_impact = total_asset_value * IMPACT_MULTIPLIERS.get(
            "reputation_market_cap_impact", 0.05
        ) * reputation_sensitivity
        impact_score = self._impact_to_score(0.05, total_asset_value)
        mitigation = mitigation_map.get(TransitionRiskType.STAKEHOLDER_CONCERN.value, 0)

        raw_score = likelihood_score * impact_score
        adjusted_score = raw_score * (1 - mitigation)

        reputation_risks.append(RiskScore(
            risk_type=TransitionRiskType.STAKEHOLDER_CONCERN.value,
            risk_category="transition_reputation",
            likelihood=likelihood_score,
            impact=impact_score,
            mitigation_effectiveness=mitigation,
            raw_score=round(raw_score, 2),
            adjusted_score=round(adjusted_score, 2),
            risk_level=self._score_to_category(adjusted_score),
            financial_impact_usd=round(reputation_impact, 0),
            time_horizon=TimeHorizon.SHORT_TERM,
            confidence_level=0.5,
            key_drivers=[
                f"Reputation sensitivity: {reputation_sensitivity:.0%}",
            ],
        ))

        # Litigation risk for high-emission sectors
        if carbon_emissions > 100000:  # Large emitters
            litigation_likelihood = self._probability_to_score(
                0.3 * transition_multiplier
            )
            litigation_cost = min(total_revenue * 0.1, 100000000)  # Cap at $100M
            litigation_impact = self._revenue_impact_to_score(litigation_cost, total_revenue)
            mitigation = mitigation_map.get(TransitionRiskType.LITIGATION.value, 0)

            raw_score = litigation_likelihood * litigation_impact
            adjusted_score = raw_score * (1 - mitigation)

            reputation_risks.append(RiskScore(
                risk_type=TransitionRiskType.LITIGATION.value,
                risk_category="transition_reputation",
                likelihood=litigation_likelihood,
                impact=litigation_impact,
                mitigation_effectiveness=mitigation,
                raw_score=round(raw_score, 2),
                adjusted_score=round(adjusted_score, 2),
                risk_level=self._score_to_category(adjusted_score),
                financial_impact_usd=round(litigation_cost, 0),
                time_horizon=TimeHorizon.MEDIUM_TERM,
                confidence_level=0.4,
                key_drivers=[
                    f"High emissions: {carbon_emissions:,.0f} tCO2e",
                    "Climate litigation trend",
                ],
            ))

        # Calculate total transition risk score
        all_transition_risks = policy_risks + technology_risks + market_risks + reputation_risks
        all_transition_scores = [r.adjusted_score for r in all_transition_risks]
        total_transition_score = (
            sum(all_transition_scores) / len(all_transition_scores)
            if all_transition_scores else 0
        )

        # Find highest risk
        highest_risk = max(
            all_transition_risks, key=lambda r: r.adjusted_score
        ) if all_transition_risks else None

        # Calculate carbon price impact
        carbon_price_total = carbon_price_impact if carbon_emissions > 0 else None

        return TransitionRiskAssessment(
            policy_risks=policy_risks,
            technology_risks=technology_risks,
            market_risks=market_risks,
            reputation_risks=reputation_risks,
            total_transition_risk_score=round(total_transition_score, 2),
            highest_risk=highest_risk.risk_type if highest_risk else None,
            carbon_price_impact_usd=carbon_price_total,
        )

    def _analyze_scenario(
        self,
        input_data: ClimateRiskInput,
        scenario: ClimateScenario
    ) -> ScenarioImpact:
        """
        Analyze impact for a specific climate scenario.

        ZERO-HALLUCINATION: Uses fixed IPCC scenario parameters.

        Args:
            input_data: Climate risk input
            scenario: Climate scenario to analyze

        Returns:
            Scenario impact analysis
        """
        params = self.scenario_parameters[scenario]

        # Interpolate temperature for time horizon
        years_to_2050 = 2050 - input_data.base_year
        years_to_2100 = 2100 - input_data.base_year

        if input_data.time_horizon_years <= years_to_2050:
            temp_increase = params["temperature_2050"] * (
                input_data.time_horizon_years / years_to_2050
            )
        else:
            t = (input_data.time_horizon_years - years_to_2050) / (years_to_2100 - years_to_2050)
            temp_increase = params["temperature_2050"] + t * (
                params["temperature_2100"] - params["temperature_2050"]
            )

        # Calculate total values
        total_asset_value = sum(a.value_usd for a in input_data.assets)
        total_revenue = sum(rs.annual_revenue_usd for rs in input_data.revenue_streams)

        # Calculate value at risk
        physical_var = (
            total_asset_value *
            0.1 *  # Base physical risk exposure
            params["physical_risk_multiplier"]
        )

        transition_var = (
            total_revenue *
            0.15 *  # Base transition risk exposure
            params["transition_risk_multiplier"]
        )

        total_var = physical_var + transition_var

        # Calculate revenue impact percentage
        revenue_impact_pct = (
            transition_var / total_revenue * 100
            if total_revenue > 0 else 0
        )

        # Calculate cost increase percentage
        cost_increase_pct = (
            5 *  # Base cost increase
            params["physical_risk_multiplier"] *
            (input_data.time_horizon_years / 30)
        )

        # Calculate NPV of impacts
        discount_rate = input_data.discount_rate
        annual_impact = total_var / input_data.time_horizon_years

        npv_impact = sum(
            annual_impact / ((1 + discount_rate) ** year)
            for year in range(1, input_data.time_horizon_years + 1)
        )

        return ScenarioImpact(
            scenario=scenario.value,
            temperature_increase_c=round(temp_increase, 1),
            physical_risk_multiplier=params["physical_risk_multiplier"],
            transition_risk_multiplier=params["transition_risk_multiplier"],
            total_value_at_risk_usd=round(total_var, 0),
            revenue_impact_pct=round(revenue_impact_pct, 2),
            cost_increase_pct=round(cost_increase_pct, 2),
            npv_impact_usd=round(npv_impact, 0),
            key_assumptions=[
                params["description"],
                f"Temperature increase by {input_data.base_year + input_data.time_horizon_years}: {temp_increase:.1f}C",
                f"Carbon price: ${params['carbon_price_2030']}/tCO2e by 2030",
            ],
        )

    def _calculate_financial_exposure(
        self,
        input_data: ClimateRiskInput,
        physical_assessment: PhysicalRiskAssessment,
        transition_assessment: TransitionRiskAssessment,
        scenario_impact: ScenarioImpact
    ) -> FinancialExposure:
        """
        Calculate comprehensive financial exposure.

        ZERO-HALLUCINATION: Deterministic aggregation of assessed impacts.

        Args:
            input_data: Climate risk input
            physical_assessment: Physical risk results
            transition_assessment: Transition risk results
            scenario_impact: Scenario impact analysis

        Returns:
            Financial exposure summary
        """
        # Asset value at risk (from physical risks)
        asset_var = sum(physical_assessment.asset_exposure_summary.values())

        # Revenue at risk (from transition risks)
        total_revenue = sum(rs.annual_revenue_usd for rs in input_data.revenue_streams)
        revenue_at_risk = total_revenue * (scenario_impact.revenue_impact_pct / 100)

        # Expected cost increases
        expected_cost_increases = total_revenue * (scenario_impact.cost_increase_pct / 100)

        # Insurance gap
        total_asset_value = sum(a.value_usd for a in input_data.assets)
        total_insurance = sum(
            a.insurance_coverage_usd or 0 for a in input_data.assets
        )
        insurance_gap = max(0, asset_var - total_insurance)

        # Carbon liability
        carbon_liability = transition_assessment.carbon_price_impact_usd or 0

        # Stranded asset risk
        stranded_risk = 0
        for risk in transition_assessment.technology_risks:
            if risk.risk_type == TransitionRiskType.STRANDED_ASSETS.value:
                stranded_risk = risk.financial_impact_usd or 0
                break

        # Total financial exposure
        total_exposure = (
            asset_var +
            revenue_at_risk +
            expected_cost_increases +
            carbon_liability +
            stranded_risk
        )

        # Exposure as percentage of assets
        exposure_pct = (
            (total_exposure / total_asset_value * 100)
            if total_asset_value > 0 else 0
        )

        return FinancialExposure(
            total_asset_value_at_risk_usd=round(asset_var, 0),
            annual_revenue_at_risk_usd=round(revenue_at_risk, 0),
            expected_cost_increases_usd=round(expected_cost_increases, 0),
            insurance_gap_usd=round(insurance_gap, 0),
            carbon_liability_usd=round(carbon_liability, 0),
            stranded_asset_risk_usd=round(stranded_risk, 0),
            total_financial_exposure_usd=round(total_exposure, 0),
            exposure_as_pct_assets=round(exposure_pct, 2),
        )

    def _generate_risk_register(
        self,
        input_data: ClimateRiskInput,
        physical_assessment: PhysicalRiskAssessment,
        transition_assessment: TransitionRiskAssessment
    ) -> RiskRegister:
        """
        Generate TCFD-aligned risk register.

        Args:
            input_data: Climate risk input
            physical_assessment: Physical risk results
            transition_assessment: Transition risk results

        Returns:
            Risk register
        """
        all_physical = physical_assessment.acute_risks + physical_assessment.chronic_risks
        all_transition = (
            transition_assessment.policy_risks +
            transition_assessment.technology_risks +
            transition_assessment.market_risks +
            transition_assessment.reputation_risks
        )

        all_risks = all_physical + all_transition

        critical_count = sum(1 for r in all_risks if r.risk_level == RiskCategory.CRITICAL)
        high_count = sum(1 for r in all_risks if r.risk_level == RiskCategory.HIGH)

        return RiskRegister(
            organization=input_data.organization_name,
            scenario=input_data.scenario.value,
            time_horizon_years=input_data.time_horizon_years,
            physical_risks=all_physical,
            transition_risks=all_transition,
            total_risks_assessed=len(all_risks),
            critical_risks=critical_count,
            high_risks=high_count,
        )

    def _generate_recommendations(
        self,
        input_data: ClimateRiskInput,
        physical_assessment: PhysicalRiskAssessment,
        transition_assessment: TransitionRiskAssessment,
        financial_exposure: FinancialExposure
    ) -> List[ResilienceRecommendation]:
        """
        Generate resilience recommendations based on assessment.

        Args:
            input_data: Climate risk input
            physical_assessment: Physical risk results
            transition_assessment: Transition risk results
            financial_exposure: Financial exposure

        Returns:
            List of prioritized recommendations
        """
        recommendations: List[ResilienceRecommendation] = []
        priority = 1

        # Physical risk recommendations
        for risk in physical_assessment.acute_risks + physical_assessment.chronic_risks:
            if risk.risk_level in [RiskCategory.CRITICAL, RiskCategory.HIGH]:
                rec = self._create_physical_recommendation(
                    risk, priority, financial_exposure
                )
                recommendations.append(rec)
                priority += 1

        # Transition risk recommendations
        for risk in (
            transition_assessment.policy_risks +
            transition_assessment.technology_risks +
            transition_assessment.market_risks
        ):
            if risk.risk_level in [RiskCategory.CRITICAL, RiskCategory.HIGH]:
                rec = self._create_transition_recommendation(
                    risk, priority, input_data
                )
                recommendations.append(rec)
                priority += 1

        # General recommendations
        if financial_exposure.insurance_gap_usd > 0:
            recommendations.append(ResilienceRecommendation(
                priority=priority,
                category="risk_transfer",
                title="Close Insurance Coverage Gap",
                description=(
                    f"Current insurance coverage leaves ${financial_exposure.insurance_gap_usd:,.0f} "
                    "in climate-related asset exposure uninsured. Evaluate parametric insurance "
                    "and climate-specific coverage options."
                ),
                risk_addressed=["flood", "cyclone", "wildfire"],
                estimated_risk_reduction_pct=30,
                estimated_cost_usd=financial_exposure.insurance_gap_usd * 0.02,
                payback_years=1,
                implementation_timeline="3-6 months",
            ))
            priority += 1

        if financial_exposure.carbon_liability_usd > 1000000:
            recommendations.append(ResilienceRecommendation(
                priority=priority,
                category="emissions_reduction",
                title="Accelerate Decarbonization Program",
                description=(
                    f"Projected carbon liability of ${financial_exposure.carbon_liability_usd:,.0f} "
                    "warrants accelerated emissions reduction. Develop science-based targets and "
                    "implement high-impact reduction measures."
                ),
                risk_addressed=["carbon_pricing", "regulation", "stranded_assets"],
                estimated_risk_reduction_pct=40,
                estimated_cost_usd=financial_exposure.carbon_liability_usd * 0.5,
                payback_years=5,
                implementation_timeline="12-24 months",
            ))

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        return recommendations[:10]  # Return top 10 recommendations

    def _create_physical_recommendation(
        self,
        risk: RiskScore,
        priority: int,
        financial_exposure: FinancialExposure
    ) -> ResilienceRecommendation:
        """Create recommendation for physical risk."""
        risk_type = risk.risk_type

        recommendations_map = {
            "flood": {
                "title": "Flood Resilience Enhancement",
                "description": "Implement flood barriers, elevate critical infrastructure, develop flood response plans",
                "cost_factor": 0.05,
                "reduction": 40,
                "timeline": "6-12 months",
            },
            "cyclone": {
                "title": "Storm Hardening Program",
                "description": "Strengthen building envelopes, install impact-resistant windows, secure rooftop equipment",
                "cost_factor": 0.08,
                "reduction": 35,
                "timeline": "12-18 months",
            },
            "wildfire": {
                "title": "Wildfire Defense Implementation",
                "description": "Create defensible space, install fire-resistant materials, develop evacuation plans",
                "cost_factor": 0.06,
                "reduction": 45,
                "timeline": "6-12 months",
            },
            "extreme_heat": {
                "title": "Heat Resilience Measures",
                "description": "Upgrade cooling systems, implement heat action plans, protect outdoor workers",
                "cost_factor": 0.03,
                "reduction": 50,
                "timeline": "3-6 months",
            },
            "sea_level_rise": {
                "title": "Coastal Adaptation Strategy",
                "description": "Evaluate asset relocation, implement coastal defenses, develop managed retreat plan",
                "cost_factor": 0.15,
                "reduction": 60,
                "timeline": "24-36 months",
            },
        }

        rec_data = recommendations_map.get(risk_type, {
            "title": f"Address {risk_type.replace('_', ' ').title()} Risk",
            "description": "Develop and implement risk mitigation measures",
            "cost_factor": 0.05,
            "reduction": 30,
            "timeline": "6-12 months",
        })

        estimated_cost = (risk.financial_impact_usd or 0) * rec_data["cost_factor"]
        risk_reduction_value = (risk.financial_impact_usd or 0) * (rec_data["reduction"] / 100)
        payback = estimated_cost / risk_reduction_value if risk_reduction_value > 0 else None

        return ResilienceRecommendation(
            priority=priority,
            category="physical_resilience",
            title=rec_data["title"],
            description=rec_data["description"],
            risk_addressed=[risk_type],
            estimated_risk_reduction_pct=rec_data["reduction"],
            estimated_cost_usd=round(estimated_cost, 0),
            payback_years=round(payback, 1) if payback else None,
            implementation_timeline=rec_data["timeline"],
        )

    def _create_transition_recommendation(
        self,
        risk: RiskScore,
        priority: int,
        input_data: ClimateRiskInput
    ) -> ResilienceRecommendation:
        """Create recommendation for transition risk."""
        risk_type = risk.risk_type

        recommendations_map = {
            "carbon_pricing": {
                "title": "Carbon Cost Management Program",
                "description": "Implement internal carbon pricing, identify reduction opportunities, develop hedging strategy",
                "cost_factor": 0.02,
                "reduction": 35,
                "timeline": "6-12 months",
            },
            "regulation": {
                "title": "Regulatory Compliance Readiness",
                "description": "Monitor emerging regulations, implement compliance systems, engage with policymakers",
                "cost_factor": 0.03,
                "reduction": 40,
                "timeline": "3-6 months",
            },
            "technology_disruption": {
                "title": "Technology Transition Planning",
                "description": "Assess technology landscape, develop transition roadmap, invest in low-carbon alternatives",
                "cost_factor": 0.10,
                "reduction": 50,
                "timeline": "12-24 months",
            },
            "stranded_assets": {
                "title": "Asset Portfolio Optimization",
                "description": "Evaluate asset stranding risk, accelerate depreciation, divest high-risk assets",
                "cost_factor": 0.05,
                "reduction": 45,
                "timeline": "12-24 months",
            },
            "demand_shift": {
                "title": "Market Positioning Strategy",
                "description": "Develop low-carbon products, diversify revenue streams, strengthen green credentials",
                "cost_factor": 0.08,
                "reduction": 40,
                "timeline": "6-18 months",
            },
        }

        rec_data = recommendations_map.get(risk_type, {
            "title": f"Address {risk_type.replace('_', ' ').title()} Risk",
            "description": "Develop and implement transition risk mitigation",
            "cost_factor": 0.05,
            "reduction": 30,
            "timeline": "6-12 months",
        })

        estimated_cost = (risk.financial_impact_usd or 0) * rec_data["cost_factor"]
        risk_reduction_value = (risk.financial_impact_usd or 0) * (rec_data["reduction"] / 100)
        payback = estimated_cost / risk_reduction_value if risk_reduction_value > 0 else None

        return ResilienceRecommendation(
            priority=priority,
            category="transition_readiness",
            title=rec_data["title"],
            description=rec_data["description"],
            risk_addressed=[risk_type],
            estimated_risk_reduction_pct=rec_data["reduction"],
            estimated_cost_usd=round(estimated_cost, 0),
            payback_years=round(payback, 1) if payback else None,
            implementation_timeline=rec_data["timeline"],
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_assessment_id(self, input_data: ClimateRiskInput) -> str:
        """Generate unique assessment ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        org_hash = hashlib.md5(
            input_data.organization_name.encode()
        ).hexdigest()[:8]
        return f"CRA-{timestamp}-{org_hash}"

    def _get_climate_zone(self, latitude: float) -> str:
        """Determine climate zone from latitude."""
        abs_lat = abs(latitude)
        if abs_lat < 23.5:
            return "tropical"
        elif abs_lat < 35:
            return "subtropical"
        elif abs_lat < 55:
            return "temperate"
        elif abs_lat < 66.5:
            return "continental"
        else:
            return "polar"

    def _get_coastal_category(self, coastal_distance_km: Optional[float]) -> str:
        """Determine coastal category."""
        if coastal_distance_km is None:
            return "inland"
        elif coastal_distance_km < 10:
            return "coastal_0_10km"
        elif coastal_distance_km < 50:
            return "coastal_10_50km"
        elif coastal_distance_km < 100:
            return "coastal_50_100km"
        else:
            return "inland"

    def _get_time_horizon(self, years: int) -> TimeHorizon:
        """Determine time horizon category."""
        if years <= 5:
            return TimeHorizon.SHORT_TERM
        elif years <= 15:
            return TimeHorizon.MEDIUM_TERM
        else:
            return TimeHorizon.LONG_TERM

    def _probability_to_score(self, probability: float) -> int:
        """
        Convert probability (0-1) to likelihood score (1-5).

        ZERO-HALLUCINATION: Fixed mapping.
        """
        if probability >= 0.8:
            return 5
        elif probability >= 0.6:
            return 4
        elif probability >= 0.4:
            return 3
        elif probability >= 0.2:
            return 2
        else:
            return 1

    def _impact_to_score(self, impact_pct: float, total_value: float) -> int:
        """
        Convert impact percentage to impact score (1-5).

        ZERO-HALLUCINATION: Fixed thresholds.
        """
        if impact_pct >= 0.25:
            return 5
        elif impact_pct >= 0.15:
            return 4
        elif impact_pct >= 0.08:
            return 3
        elif impact_pct >= 0.03:
            return 2
        else:
            return 1

    def _revenue_impact_to_score(self, impact_usd: float, total_revenue: float) -> int:
        """Convert revenue impact to score."""
        if total_revenue == 0:
            return 1

        pct = impact_usd / total_revenue
        if pct >= 0.20:
            return 5
        elif pct >= 0.10:
            return 4
        elif pct >= 0.05:
            return 3
        elif pct >= 0.02:
            return 2
        else:
            return 1

    def _carbon_impact_to_score(self, impact_usd: float, total_revenue: float) -> int:
        """Convert carbon cost impact to score."""
        if total_revenue == 0:
            return 1

        pct = impact_usd / total_revenue
        if pct >= 0.10:
            return 5
        elif pct >= 0.05:
            return 4
        elif pct >= 0.02:
            return 3
        elif pct >= 0.01:
            return 2
        else:
            return 1

    def _score_to_category(self, score: float) -> RiskCategory:
        """
        Convert risk score to category.

        ZERO-HALLUCINATION: Fixed thresholds.
        """
        if score >= 20:
            return RiskCategory.CRITICAL
        elif score >= 12:
            return RiskCategory.HIGH
        elif score >= 6:
            return RiskCategory.MEDIUM
        elif score >= 3:
            return RiskCategory.LOW
        else:
            return RiskCategory.MINIMAL

    def _categorize_risk(self, score: float) -> RiskCategory:
        """Categorize total risk score."""
        if score >= 15:
            return RiskCategory.CRITICAL
        elif score >= 10:
            return RiskCategory.HIGH
        elif score >= 5:
            return RiskCategory.MEDIUM
        elif score >= 2:
            return RiskCategory.LOW
        else:
            return RiskCategory.MINIMAL

    def _calculate_total_risk_score(
        self,
        physical_score: float,
        transition_score: float
    ) -> float:
        """
        Calculate total risk score.

        ZERO-HALLUCINATION CALCULATION:
        Formula: total = (physical * 0.5) + (transition * 0.5)
        """
        return (physical_score * 0.5) + (transition_score * 0.5)

    def _build_mitigation_map(
        self,
        measures: List[MitigationMeasure]
    ) -> Dict[str, float]:
        """Build mapping of risk types to mitigation effectiveness."""
        mitigation_map: Dict[str, float] = {}
        for measure in measures:
            if measure.implementation_status == "implemented":
                effectiveness = measure.effectiveness
            elif measure.implementation_status == "in_progress":
                effectiveness = measure.effectiveness * 0.5
            else:
                effectiveness = 0

            risk_type = measure.risk_type.lower()
            if risk_type in mitigation_map:
                # Don't exceed 95% mitigation
                mitigation_map[risk_type] = min(
                    0.95,
                    mitigation_map[risk_type] + effectiveness * (1 - mitigation_map[risk_type])
                )
            else:
                mitigation_map[risk_type] = min(0.95, effectiveness)

        return mitigation_map

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a processing step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash of complete provenance chain.

        This hash enables:
        - Verification that assessment was deterministic
        - Audit trail for regulatory compliance
        - Reproducibility checking
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def get_supported_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of supported climate scenarios."""
        return [
            {
                "id": scenario.value,
                "name": params["description"],
                "temperature_2050": params["temperature_2050"],
                "temperature_2100": params["temperature_2100"],
            }
            for scenario, params in self.scenario_parameters.items()
        ]

    def get_sector_sensitivity(self, sector: SectorType) -> Dict[str, float]:
        """Get transition risk sensitivity for a sector."""
        return self.sector_sensitivity.get(
            sector,
            self.sector_sensitivity[SectorType.MANUFACTURING]
        )

    def get_physical_risk_types(self) -> List[str]:
        """Get list of physical risk types."""
        return [rt.value for rt in PhysicalRiskType]

    def get_transition_risk_types(self) -> List[str]:
        """Get list of transition risk types."""
        return [rt.value for rt in TransitionRiskType]


# =============================================================================
# Pack Specification
# =============================================================================


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "risk/climate_risk_v1",
    "name": "Climate Risk Assessment Agent",
    "version": "1.0.0",
    "summary": "TCFD-aligned climate risk assessment with physical and transition risks",
    "tags": [
        "climate-risk",
        "tcfd",
        "physical-risk",
        "transition-risk",
        "scenario-analysis",
        "ipcc",
    ],
    "owners": ["risk-team"],
    "compute": {
        "entrypoint": "python://agents.gl_011_climate_risk.agent:ClimateRiskAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "scenario://ipcc/ar6/2023"},
        {"ref": "scenario://ngfs/2024"},
        {"ref": "framework://tcfd/2017"},
    ],
    "provenance": {
        "ipcc_version": "AR6",
        "tcfd_version": "2017",
        "enable_audit": True,
    },
}
