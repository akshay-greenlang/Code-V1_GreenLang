"""
GL-TCFD-APP v1.0 -- TCFD Disclosure & Scenario Analysis Platform Domain Models

This module defines all Pydantic v2 domain models for the GL-TCFD-APP v1.0
platform.  Models cover the full TCFD four-pillar lifecycle: governance
assessment, strategy analysis (risks & opportunities), scenario analysis
(IEA/NGFS), physical risk assessment, transition risk assessment, financial
impact quantification, risk management integration, metrics & targets tracking,
disclosure generation, ISSB/IFRS S2 cross-walk, gap analysis, and
recommendation generation.

All monetary values are in USD unless otherwise noted.  All emissions are in
metric tonnes CO2e.  Timestamps are UTC.

Reference:
    - TCFD Final Report (June 2017)
    - TCFD Annex: Implementing the Recommendations (June 2017)
    - TCFD Guidance on Scenario Analysis (October 2020)
    - IFRS S2 Climate-related Disclosures (June 2023)

Example:
    >>> from .config import RiskType, TimeHorizon
    >>> risk = ClimateRisk(
    ...     tenant_id="tenant-1", org_id="org-1",
    ...     risk_type=RiskType.TRANSITION_POLICY,
    ...     name="Carbon tax", description="Regulatory carbon pricing")
    >>> print(risk.id)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import (
    AssetType,
    DataQualityTier,
    DisclosureCode,
    DisclosureStatus,
    FinancialImpactCategory,
    FinancialStatementType,
    GovernanceMaturityLevel,
    ISSBMetricType,
    ISSBS2Paragraph,
    MetricCategory,
    OpportunityCategory,
    PhysicalHazard,
    ReportFormat,
    RiskImpact,
    RiskLikelihood,
    RiskResponse,
    RiskType,
    SBTiAlignment,
    ScenarioType,
    SectorType,
    TargetType,
    TCFDPillar,
    TemperatureOutcome,
    TimeHorizon,
    TransitionRiskSubType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Generate a deterministic-safe UUID4 string."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """UTC now truncated to seconds."""
    return datetime.utcnow().replace(microsecond=0)


def _sha256(payload: str) -> str:
    """SHA-256 hex digest for provenance tracking."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class Organization(BaseModel):
    """
    Organization registered for TCFD climate disclosure.

    Holds the organizational profile, sector classification,
    and reporting boundary metadata.  Multi-tenant via tenant_id.
    """

    id: str = Field(default_factory=_new_id, description="Unique organization ID")
    tenant_id: str = Field(..., description="Tenant ID for multi-tenancy isolation")
    name: str = Field(..., min_length=1, max_length=500, description="Legal entity name")
    sector: SectorType = Field(
        default=SectorType.ENERGY, description="TCFD sector classification",
    )
    country: str = Field(..., min_length=2, max_length=3, description="HQ country ISO 3166")
    region: Optional[str] = Field(None, max_length=100, description="Geographic region")
    description: Optional[str] = Field(None, max_length=2000)
    employee_count: Optional[int] = Field(None, ge=0, description="Full-time equivalents")
    annual_revenue_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    total_assets_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    total_emissions_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class AssetLocation(BaseModel):
    """
    A physical asset subject to climate risk assessment.

    Tracks geographic coordinates (lat/lon), asset type, replacement value,
    and building characteristics for physical risk scoring.
    """

    id: str = Field(default_factory=_new_id, description="Asset ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    asset_name: str = Field(..., min_length=1, max_length=255, description="Asset name")
    asset_type: AssetType = Field(default=AssetType.BUILDING)
    latitude: Decimal = Field(
        ..., ge=Decimal("-90"), le=Decimal("90"), description="Latitude (WGS84)",
    )
    longitude: Decimal = Field(
        ..., ge=Decimal("-180"), le=Decimal("180"), description="Longitude (WGS84)",
    )
    country: str = Field(..., min_length=2, max_length=3, description="ISO country code")
    region: Optional[str] = Field(None, max_length=100)
    elevation_m: Decimal = Field(
        default=Decimal("0"), description="Elevation in meters above sea level",
    )
    building_type: Optional[str] = Field(None, max_length=100, description="Building classification")
    replacement_value_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Asset replacement value (USD)",
    )
    book_value_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    year_built: Optional[int] = Field(None, ge=1800, le=2100)
    useful_life_years: Optional[int] = Field(None, ge=1, le=200)
    insurance_coverage_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Current insurance coverage (USD)",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ReportingPeriod(BaseModel):
    """
    Reporting period definition for TCFD disclosures.

    Tracks the calendar or fiscal year boundaries and reporting status.
    """

    id: str = Field(default_factory=_new_id, description="Reporting period ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    start_date: date = Field(..., description="Period start date")
    end_date: date = Field(..., description="Period end date")
    is_fiscal_year: bool = Field(default=False, description="True if fiscal year differs from calendar")
    status: DisclosureStatus = Field(default=DisclosureStatus.DRAFT)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        """End date must be after start date."""
        start = info.data.get("start_date")
        if start is not None and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


# ---------------------------------------------------------------------------
# Governance Models (Pillar 1)
# ---------------------------------------------------------------------------

class BoardOversight(BaseModel):
    """Board-level climate oversight details per TCFD Governance (a)."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    committee_name: str = Field(..., min_length=1, max_length=255)
    charter_includes_climate: bool = Field(default=False)
    meeting_frequency_annual: int = Field(default=0, ge=0)
    climate_agenda_items_annual: int = Field(default=0, ge=0)
    has_dedicated_climate_committee: bool = Field(default=False)
    board_climate_training: bool = Field(default=False)
    training_hours_annual: int = Field(default=0, ge=0)
    oversight_scope: List[str] = Field(
        default_factory=list,
        description="Areas overseen (strategy, risk, targets, capex, reporting)",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ManagementRole(BaseModel):
    """
    A climate-related management role per TCFD Governance (b).

    Tracks roles with climate accountability, reporting lines,
    and competency information.
    """

    id: str = Field(default_factory=_new_id, description="Unique role ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    role_title: str = Field(..., min_length=1, max_length=255, description="Role title")
    person_name: Optional[str] = Field(None, max_length=255, description="Incumbent name")
    responsibility_description: str = Field(
        default="", max_length=2000, description="Climate-related responsibilities",
    )
    is_c_suite: bool = Field(default=False, description="Whether this is a C-suite role")
    climate_accountability: bool = Field(
        default=False, description="Explicit climate accountability",
    )
    reporting_line: Optional[str] = Field(
        None, max_length=255, description="Reports to (role or committee)",
    )
    decision_authority: List[str] = Field(
        default_factory=list,
        description="Decision areas (budget, strategy, operations)",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ClimateCompetency(BaseModel):
    """Board and management climate competency assessment."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    role_id: Optional[str] = Field(None, description="Role ID if individual assessment")
    competency_area: str = Field(..., max_length=255, description="Competency domain")
    skill_level: int = Field(default=1, ge=1, le=5, description="Skill level (1-5)")
    evidence: Optional[str] = Field(None, max_length=2000, description="Evidence of competency")
    development_plan: Optional[str] = Field(None, max_length=2000)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class IncentiveLinkage(BaseModel):
    """Remuneration and incentive linkage to climate targets."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    role_title: str = Field(..., max_length=255)
    incentive_type: str = Field(
        default="bonus",
        description="bonus, long_term_incentive, equity, kpi_scorecard",
    )
    climate_metric_linked: str = Field(
        default="", max_length=255, description="Which climate metric is linked",
    )
    percentage_of_compensation: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    threshold_value: Optional[Decimal] = Field(None)
    target_value: Optional[Decimal] = Field(None)
    stretch_value: Optional[Decimal] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class GovernanceMaturity(BaseModel):
    """Governance maturity scoring across 8 dimensions."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    dimension: str = Field(..., max_length=100, description="Maturity dimension name")
    score: int = Field(default=1, ge=1, le=5, description="Dimension score (1-5)")
    weight: Decimal = Field(
        default=Decimal("0.125"), ge=Decimal("0"), le=Decimal("1"),
        description="Weight in overall maturity calculation",
    )
    evidence: Optional[str] = Field(None, max_length=2000)
    recommendations: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class GovernanceAssessment(BaseModel):
    """
    TCFD Governance pillar assessment per Governance (a) and (b) disclosures.

    Evaluates board oversight, management roles, climate competency,
    meeting frequency, reporting structures, and incentive alignment
    across 8 maturity dimensions, each scored 1-5.
    """

    id: str = Field(default_factory=_new_id, description="Assessment ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    assessment_date: date = Field(default_factory=lambda: date.today())
    board_oversight_score: int = Field(
        default=1, ge=1, le=5, description="Board climate oversight maturity (1-5)",
    )
    board_oversight: Optional[BoardOversight] = Field(None)
    board_committees: List[str] = Field(
        default_factory=list,
        description="Board committees with climate responsibility",
    )
    meeting_frequency: int = Field(
        default=0, ge=0, description="Annual climate-focused board meetings",
    )
    management_roles: List[ManagementRole] = Field(
        default_factory=list, description="Management roles with climate accountability",
    )
    competencies: List[ClimateCompetency] = Field(default_factory=list)
    climate_competency_score: int = Field(
        default=1, ge=1, le=5, description="Board/mgmt climate competency (1-5)",
    )
    incentive_linkages: List[IncentiveLinkage] = Field(default_factory=list)
    incentive_linkage: bool = Field(
        default=False, description="Whether remuneration is linked to climate targets",
    )
    incentive_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of remuneration linked to climate performance",
    )
    maturity_dimensions: List[GovernanceMaturity] = Field(default_factory=list)
    maturity_scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Maturity scores per dimension (8 dimensions, each 1-5)",
    )
    overall_maturity: GovernanceMaturityLevel = Field(
        default=GovernanceMaturityLevel.INITIAL, description="Overall governance maturity",
    )
    notes: Optional[str] = Field(None, max_length=5000)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.assessment_date}:"
                f"{self.board_oversight_score}:{self.climate_competency_score}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Strategy Models (Pillar 2) -- Climate Risks
# ---------------------------------------------------------------------------

class ClimateRisk(BaseModel):
    """
    A climate-related risk identified per TCFD Strategy (a).

    Captures risk type (physical/transition), time horizon, likelihood,
    impact, financial impact estimate, and response strategy.
    """

    id: str = Field(default_factory=_new_id, description="Risk ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    risk_type: RiskType = Field(..., description="Physical or transition risk type")
    name: str = Field(..., min_length=1, max_length=255, description="Risk name")
    description: str = Field(default="", max_length=5000, description="Risk description")
    category: str = Field(default="", description="Sub-category or grouping")
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.MEDIUM_TERM, description="Time horizon for the risk",
    )
    likelihood: RiskLikelihood = Field(
        default=RiskLikelihood.MEDIUM, description="Likelihood of occurrence",
    )
    impact: RiskImpact = Field(
        default=RiskImpact.MODERATE, description="Impact severity",
    )
    risk_score: int = Field(
        default=0, ge=0, le=25, description="Combined score = likelihood_num x impact_num",
    )
    financial_impact_low_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Financial impact lower bound (USD)",
    )
    financial_impact_mid_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Financial impact midpoint (USD)",
    )
    financial_impact_high_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Financial impact upper bound (USD)",
    )
    affected_assets: List[str] = Field(
        default_factory=list, description="Asset IDs or descriptions affected",
    )
    affected_value_chain_stages: List[str] = Field(
        default_factory=list, description="Value chain stages affected",
    )
    response_strategy: RiskResponse = Field(
        default=RiskResponse.MITIGATE, description="Risk response strategy",
    )
    response_description: str = Field(
        default="", max_length=2000, description="Detailed response plan",
    )
    owner: Optional[str] = Field(None, max_length=255, description="Risk owner")
    status: str = Field(
        default="active", description="Risk status (active, mitigated, closed)",
    )
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.risk_type}:{self.name}:"
                f"{self.financial_impact_mid_usd}"
            )
            self.provenance_hash = _sha256(payload)


class ClimateOpportunity(BaseModel):
    """
    A climate-related opportunity per TCFD Strategy (a).

    Captures opportunity category, revenue potential, cost savings,
    investment required, ROI, and feasibility scoring.
    """

    id: str = Field(default_factory=_new_id, description="Opportunity ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    category: OpportunityCategory = Field(..., description="Opportunity category")
    name: str = Field(..., min_length=1, max_length=255, description="Opportunity name")
    description: str = Field(default="", max_length=5000)
    revenue_potential_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Potential annual revenue (USD)",
    )
    cost_savings_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Potential annual cost savings (USD)",
    )
    investment_required_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Investment needed (USD)",
    )
    roi_estimate_pct: Decimal = Field(
        default=Decimal("0"), description="Estimated ROI percentage",
    )
    payback_period_years: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Payback period in years",
    )
    timeline: TimeHorizon = Field(
        default=TimeHorizon.MEDIUM_TERM, description="Realization timeline",
    )
    feasibility_score: int = Field(
        default=3, ge=1, le=5, description="Feasibility score (1-5)",
    )
    priority_score: int = Field(
        default=3, ge=1, le=5, description="Priority score (1-5)",
    )
    status: str = Field(
        default="identified",
        description="Pipeline status (identified, evaluated, approved, in_progress, realized)",
    )
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.category}:{self.name}:"
                f"{self.revenue_potential_usd}"
            )
            self.provenance_hash = _sha256(payload)


class BusinessModelImpact(BaseModel):
    """Impact of climate risks/opportunities on business model per TCFD Strategy (b)."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    impact_area: str = Field(
        ..., max_length=255,
        description="Area (revenue_mix, cost_structure, supply_chain, customer_base, operations)",
    )
    description: str = Field(default="", max_length=5000)
    current_state: str = Field(default="", max_length=2000)
    projected_state: str = Field(default="", max_length=2000)
    financial_impact_pct: Decimal = Field(
        default=Decimal("0"), description="Impact as percentage of relevant base",
    )
    time_horizon: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    confidence_level: Decimal = Field(
        default=Decimal("0.5"), ge=Decimal("0"), le=Decimal("1"),
    )
    related_risk_ids: List[str] = Field(default_factory=list)
    related_opportunity_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ValueChainImpact(BaseModel):
    """Climate impact on value chain stages per TCFD Strategy (b)."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    value_chain_stage: str = Field(
        ..., max_length=255,
        description="Stage (upstream_suppliers, operations, downstream_customers, end_of_life)",
    )
    description: str = Field(default="", max_length=5000)
    exposure_level: str = Field(
        default="medium", description="Exposure level (low, medium, high, very_high)",
    )
    financial_impact_usd: Decimal = Field(
        default=Decimal("0"), description="Estimated impact (USD)",
    )
    adaptation_actions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class StrategicResponse(BaseModel):
    """Organizational strategic response to climate risks/opportunities."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    response_name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=5000)
    response_type: str = Field(
        default="adaptation",
        description="adaptation, mitigation, diversification, transformation",
    )
    investment_required_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    expected_benefit_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    implementation_timeline: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    status: str = Field(default="planned", description="planned, in_progress, completed")
    related_risk_ids: List[str] = Field(default_factory=list)
    related_opportunity_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Scenario Models (Pillar 2 - Strategy (c))
# ---------------------------------------------------------------------------

class ScenarioParameters(BaseModel):
    """
    Detailed parameters for a climate scenario.

    Contains carbon price trajectory, energy mix, temperature pathway,
    and sector-specific assumptions.
    """

    id: str = Field(default_factory=_new_id)
    scenario_id: str = Field(default="", description="Parent scenario ID")
    carbon_price_trajectory: Dict[int, Decimal] = Field(
        default_factory=dict, description="Carbon price (USD/tCO2e) by year",
    )
    energy_mix_trajectory: Dict[int, Dict[str, int]] = Field(
        default_factory=dict, description="Energy mix percentages by year",
    )
    temperature_pathway: Dict[int, Decimal] = Field(
        default_factory=dict, description="Temperature projection (degrees C) by year",
    )
    gdp_growth_rate: Optional[Decimal] = Field(None, description="Annual GDP growth assumption (%)")
    population_growth_rate: Optional[Decimal] = Field(None, description="Annual population growth (%)")
    technology_assumptions: Dict[str, str] = Field(
        default_factory=dict, description="Technology deployment assumptions",
    )
    regulatory_assumptions: Dict[str, str] = Field(
        default_factory=dict, description="Regulatory policy assumptions",
    )
    physical_assumptions: Dict[str, str] = Field(
        default_factory=dict, description="Physical climate assumptions",
    )
    sector_specific_params: Dict[str, Any] = Field(
        default_factory=dict, description="Sector-specific parameters",
    )
    created_at: datetime = Field(default_factory=_now)


class ScenarioDefinition(BaseModel):
    """
    A climate scenario definition for TCFD Strategy (c) analysis.

    Contains scenario type, temperature outcome, time horizons,
    and full parameter set.
    """

    id: str = Field(default_factory=_new_id, description="Scenario ID")
    tenant_id: str = Field(..., description="Tenant ID")
    name: str = Field(..., min_length=1, max_length=255, description="Scenario name")
    scenario_type: ScenarioType = Field(..., description="Scenario archetype")
    temperature_outcome: TemperatureOutcome = Field(
        default=TemperatureOutcome.AROUND_2C, description="End-of-century temperature",
    )
    description: str = Field(default="", max_length=5000)
    time_horizons: List[TimeHorizon] = Field(
        default_factory=lambda: [
            TimeHorizon.SHORT_TERM,
            TimeHorizon.MEDIUM_TERM,
            TimeHorizon.LONG_TERM,
        ],
    )
    parameters: Optional[ScenarioParameters] = Field(None)
    source: str = Field(
        default="", description="Source reference (e.g. IEA WEO 2023, NGFS Phase IV)",
    )
    is_custom: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ScenarioResult(BaseModel):
    """
    Results of running a scenario analysis for an organization.

    Contains revenue/cost/asset impacts, NPV, capex requirements,
    confidence intervals, key assumptions, and narrative summary.
    """

    id: str = Field(default_factory=_new_id, description="Result ID")
    tenant_id: str = Field(..., description="Tenant ID")
    scenario_id: str = Field(..., description="Scenario definition ID")
    org_id: str = Field(..., description="Organization ID")
    analysis_date: date = Field(default_factory=lambda: date.today())
    revenue_impact_pct: Decimal = Field(
        default=Decimal("0"), description="Revenue impact as percentage",
    )
    cost_impact_pct: Decimal = Field(
        default=Decimal("0"), description="Cost impact as percentage",
    )
    asset_impairment_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Asset impairment as percentage of total assets",
    )
    capex_required_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Capital expenditure required (USD)",
    )
    npv_usd: Decimal = Field(
        default=Decimal("0"), description="Net present value of impacts (USD)",
    )
    carbon_cost_annual_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Annual carbon cost (USD)",
    )
    stranded_asset_value_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Stranded asset exposure (USD)",
    )
    confidence_interval_lower: Decimal = Field(
        default=Decimal("0"), description="95% CI lower bound",
    )
    confidence_interval_upper: Decimal = Field(
        default=Decimal("0"), description="95% CI upper bound",
    )
    key_assumptions: List[str] = Field(
        default_factory=list, description="Key assumptions used in analysis",
    )
    narrative: str = Field(
        default="", max_length=10000, description="Scenario narrative summary",
    )
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.scenario_id}:{self.org_id}:"
                f"{self.revenue_impact_pct}:{self.npv_usd}"
            )
            self.provenance_hash = _sha256(payload)


class ScenarioComparison(BaseModel):
    """Comparison of multiple scenario results side-by-side."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    scenario_results: List[ScenarioResult] = Field(default_factory=list)
    comparison_date: date = Field(default_factory=lambda: date.today())
    most_resilient_scenario: Optional[str] = Field(
        None, description="Scenario ID with best outcome",
    )
    highest_risk_scenario: Optional[str] = Field(
        None, description="Scenario ID with worst outcome",
    )
    summary: str = Field(default="", max_length=5000)
    created_at: datetime = Field(default_factory=_now)


class SensitivityResult(BaseModel):
    """Result of a single sensitivity analysis run on a specific variable."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    scenario_id: str = Field(..., description="Base scenario ID")
    variable_name: str = Field(..., max_length=255, description="Variable tested")
    base_value: Decimal = Field(..., description="Base case value")
    tested_value: Decimal = Field(..., description="Sensitivity case value")
    change_pct: Decimal = Field(default=Decimal("0"), description="Percentage change from base")
    npv_impact_usd: Decimal = Field(default=Decimal("0"), description="NPV impact (USD)")
    revenue_impact_pct: Decimal = Field(default=Decimal("0"))
    cost_impact_pct: Decimal = Field(default=Decimal("0"))
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Physical Risk Models
# ---------------------------------------------------------------------------

class HazardExposure(BaseModel):
    """Exposure of a specific asset to a specific physical hazard."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    asset_id: str = Field(..., description="Asset ID")
    hazard_type: PhysicalHazard = Field(..., description="Hazard type")
    exposure_score: int = Field(default=1, ge=1, le=5, description="Exposure score (1-5)")
    return_period_years: Optional[int] = Field(
        None, ge=1, description="Return period in years (e.g. 1-in-100)",
    )
    historical_events: int = Field(
        default=0, ge=0, description="Number of historical events in past 30 years",
    )
    projected_change_pct: Decimal = Field(
        default=Decimal("0"), description="Projected change in hazard intensity (%)",
    )
    data_source: str = Field(default="", max_length=255)
    created_at: datetime = Field(default_factory=_now)


class DamageEstimate(BaseModel):
    """Estimated physical damage for a specific asset-hazard combination."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    asset_id: str = Field(..., description="Asset ID")
    hazard_type: PhysicalHazard = Field(...)
    damage_low_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    damage_mid_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    damage_high_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    downtime_days: int = Field(default=0, ge=0, description="Estimated business downtime")
    business_interruption_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    repair_time_months: Optional[int] = Field(None, ge=0)
    created_at: datetime = Field(default_factory=_now)


class InsuranceImpact(BaseModel):
    """Insurance cost impact from physical climate risks."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    asset_id: str = Field(..., description="Asset ID")
    current_premium_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    projected_premium_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    premium_increase_pct: Decimal = Field(default=Decimal("0"))
    deductible_change_usd: Decimal = Field(default=Decimal("0"))
    coverage_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    insurability_risk: str = Field(
        default="low", description="Insurability risk level (low, medium, high, uninsurable)",
    )
    created_at: datetime = Field(default_factory=_now)


class PhysicalRiskAssessment(BaseModel):
    """
    Physical climate risk assessment for an individual asset.

    Computes exposure, vulnerability, and adaptive capacity scores
    to derive a composite risk score and financial damage estimate.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    asset_id: str = Field(..., description="Asset ID")
    hazard_type: PhysicalHazard = Field(..., description="Climate hazard assessed")
    rcp_scenario: str = Field(
        default="ssp2_45", description="RCP/SSP scenario used",
    )
    exposure_score: int = Field(
        default=1, ge=1, le=5, description="Exposure score (1-5)",
    )
    vulnerability_score: int = Field(
        default=1, ge=1, le=5, description="Vulnerability score (1-5)",
    )
    adaptive_capacity_score: int = Field(
        default=3, ge=1, le=5, description="Adaptive capacity score (1-5, higher=better)",
    )
    composite_risk_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Composite risk score (0-100)",
    )
    hazard_exposures: List[HazardExposure] = Field(default_factory=list)
    damage_estimate: Optional[DamageEstimate] = Field(None)
    insurance_impact: Optional[InsuranceImpact] = Field(None)
    financial_damage_estimate_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Estimated damage (USD)",
    )
    adaptation_cost_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Cost of adaptation measures (USD)",
    )
    assessment_date: date = Field(default_factory=lambda: date.today())
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.asset_id}:{self.hazard_type}:"
                f"{self.composite_risk_score}:{self.financial_damage_estimate_usd}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Transition Risk Models
# ---------------------------------------------------------------------------

class PolicyRisk(BaseModel):
    """Policy and regulatory transition risk details."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    policy_name: str = Field(..., max_length=255)
    jurisdiction: str = Field(default="", max_length=100)
    effective_date: Optional[date] = Field(None)
    compliance_cost_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    penalty_risk_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    carbon_price_exposure_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    description: str = Field(default="", max_length=2000)
    created_at: datetime = Field(default_factory=_now)


class TechnologyRisk(BaseModel):
    """Technology-related transition risk details."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    technology_area: str = Field(..., max_length=255)
    disruption_timeline: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    current_technology_value_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    write_off_risk_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    replacement_cost_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    description: str = Field(default="", max_length=2000)
    created_at: datetime = Field(default_factory=_now)


class MarketRisk(BaseModel):
    """Market-related transition risk details."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    market_segment: str = Field(..., max_length=255)
    demand_shift_pct: Decimal = Field(
        default=Decimal("0"), description="Projected demand change (%)",
    )
    revenue_at_risk_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    commodity_price_impact_usd: Decimal = Field(default=Decimal("0"))
    supply_chain_disruption_cost_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    description: str = Field(default="", max_length=2000)
    created_at: datetime = Field(default_factory=_now)


class ReputationRisk(BaseModel):
    """Reputation-related transition risk details."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    risk_driver: str = Field(
        ..., max_length=255,
        description="Driver (greenwashing, emissions_performance, litigation, activism)",
    )
    brand_value_at_risk_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    customer_attrition_risk_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    talent_attrition_risk_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    estimated_revenue_impact_usd: Decimal = Field(default=Decimal("0"))
    description: str = Field(default="", max_length=2000)
    created_at: datetime = Field(default_factory=_now)


class StrandedAssetAnalysis(BaseModel):
    """Analysis of assets at risk of stranding under transition scenarios."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    asset_id: str = Field(..., description="Asset ID")
    scenario_id: str = Field(..., description="Scenario ID")
    current_book_value_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    projected_value_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    impairment_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    stranding_probability_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    stranding_year: Optional[int] = Field(None, ge=2020, le=2100)
    remaining_useful_life_years: Optional[int] = Field(None, ge=0)
    created_at: datetime = Field(default_factory=_now)


class TransitionRiskAssessment(BaseModel):
    """
    Transition risk assessment covering policy, technology, market, reputation.

    Tracks current and projected exposure across TCFD transition risk
    categories with financial impact quantification.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    risk_type: RiskType = Field(..., description="Transition risk sub-type")
    sub_type: TransitionRiskSubType = Field(
        default=TransitionRiskSubType.CARBON_PRICING,
        description="Specific transition driver",
    )
    sector: SectorType = Field(
        default=SectorType.ENERGY, description="Sector classification",
    )
    current_exposure_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Current exposure (USD)",
    )
    projected_exposure_2030_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Projected 2030 exposure (USD)",
    )
    projected_exposure_2050_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Projected 2050 exposure (USD)",
    )
    financial_impact_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Total financial impact (USD)",
    )
    policy_risks: List[PolicyRisk] = Field(default_factory=list)
    technology_risks: List[TechnologyRisk] = Field(default_factory=list)
    market_risks: List[MarketRisk] = Field(default_factory=list)
    reputation_risks: List[ReputationRisk] = Field(default_factory=list)
    stranded_assets: List[StrandedAssetAnalysis] = Field(default_factory=list)
    mitigation_actions: List[str] = Field(
        default_factory=list, description="Planned mitigation actions",
    )
    assessment_date: date = Field(default_factory=lambda: date.today())
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.risk_type}:{self.sub_type}:"
                f"{self.financial_impact_usd}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Opportunity Models
# ---------------------------------------------------------------------------

class RevenueSizing(BaseModel):
    """Revenue sizing for a climate opportunity."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    opportunity_id: str = Field(..., description="Opportunity ID")
    addressable_market_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    market_share_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    revenue_year_1_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    revenue_year_3_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    revenue_year_5_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    growth_rate_pct: Decimal = Field(default=Decimal("0"))
    created_at: datetime = Field(default_factory=_now)


class CostSavingsEstimate(BaseModel):
    """Cost savings estimate for a climate opportunity."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    opportunity_id: str = Field(..., description="Opportunity ID")
    savings_category: str = Field(
        ..., max_length=255,
        description="Category (energy, water, waste, materials, carbon)",
    )
    annual_savings_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    implementation_cost_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    payback_period_years: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    created_at: datetime = Field(default_factory=_now)


class InvestmentAnalysis(BaseModel):
    """Investment analysis for pursuing a climate opportunity."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    opportunity_id: str = Field(..., description="Opportunity ID")
    total_investment_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    npv_usd: Decimal = Field(default=Decimal("0"))
    irr_pct: Decimal = Field(default=Decimal("0"))
    payback_period_years: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    risk_adjusted_return_pct: Decimal = Field(default=Decimal("0"))
    discount_rate_used: Decimal = Field(default=Decimal("0.08"))
    created_at: datetime = Field(default_factory=_now)


class OpportunityAssessment(BaseModel):
    """Full opportunity assessment combining sizing, savings, and investment."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    opportunity_id: str = Field(..., description="Climate opportunity ID")
    revenue_sizing: Optional[RevenueSizing] = Field(None)
    cost_savings: List[CostSavingsEstimate] = Field(default_factory=list)
    investment_analysis: Optional[InvestmentAnalysis] = Field(None)
    total_npv_usd: Decimal = Field(default=Decimal("0"))
    recommendation: str = Field(
        default="", description="proceed, defer, reject",
    )
    created_at: datetime = Field(default_factory=_now)


class OpportunityPipeline(BaseModel):
    """Pipeline of all identified climate opportunities."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    opportunities: List[ClimateOpportunity] = Field(default_factory=list)
    total_revenue_potential_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    total_cost_savings_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    total_investment_required_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    pipeline_npv_usd: Decimal = Field(default=Decimal("0"))
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Financial Impact Models
# ---------------------------------------------------------------------------

class FinancialImpact(BaseModel):
    """
    Climate financial impact on a specific line item in financial statements.

    Per TCFD/IFRS S2, organizations must quantify the impact on income
    statement, balance sheet, and cash flow statement.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    scenario_id: str = Field(default="", description="Scenario ID (if scenario-specific)")
    statement_type: FinancialStatementType = Field(
        ..., description="Financial statement type",
    )
    impact_category: FinancialImpactCategory = Field(
        ..., description="Impact category",
    )
    line_item: str = Field(..., min_length=1, max_length=255, description="Line item name")
    current_value_usd: Decimal = Field(
        default=Decimal("0"), description="Current value (USD)",
    )
    projected_value_usd: Decimal = Field(
        default=Decimal("0"), description="Projected value under scenario (USD)",
    )
    impact_amount_usd: Decimal = Field(
        default=Decimal("0"), description="Impact = projected - current (USD)",
    )
    impact_pct: Decimal = Field(
        default=Decimal("0"), description="Impact as percentage of current value",
    )
    time_horizon: TimeHorizon = Field(
        default=TimeHorizon.MEDIUM_TERM, description="Time horizon",
    )
    confidence_level: Decimal = Field(
        default=Decimal("0.5"), ge=Decimal("0"), le=Decimal("1"),
    )
    assumptions: List[str] = Field(default_factory=list)
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute impact amount, percentage, and provenance."""
        if self.impact_amount_usd == Decimal("0") and self.current_value_usd != Decimal("0"):
            object.__setattr__(
                self, "impact_amount_usd",
                self.projected_value_usd - self.current_value_usd,
            )
        if self.impact_pct == Decimal("0") and self.current_value_usd != Decimal("0"):
            object.__setattr__(
                self, "impact_pct",
                (self.projected_value_usd - self.current_value_usd)
                / self.current_value_usd * 100,
            )
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.statement_type}:{self.line_item}:"
                f"{self.impact_amount_usd}"
            )
            self.provenance_hash = _sha256(payload)


class IncomeStatementImpact(BaseModel):
    """Aggregate income statement impacts from climate risks/opportunities."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    scenario_id: str = Field(default="")
    revenue_impact_usd: Decimal = Field(default=Decimal("0"))
    operating_cost_impact_usd: Decimal = Field(default=Decimal("0"))
    carbon_cost_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    insurance_cost_increase_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    compliance_cost_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    net_income_impact_usd: Decimal = Field(default=Decimal("0"))
    ebitda_impact_pct: Decimal = Field(default=Decimal("0"))
    created_at: datetime = Field(default_factory=_now)


class BalanceSheetImpact(BaseModel):
    """Aggregate balance sheet impacts from climate risks."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    scenario_id: str = Field(default="")
    asset_impairment_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    stranded_assets_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    new_capex_required_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    provisions_increase_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    net_asset_impact_usd: Decimal = Field(default=Decimal("0"))
    total_assets_impact_pct: Decimal = Field(default=Decimal("0"))
    created_at: datetime = Field(default_factory=_now)


class CashFlowImpact(BaseModel):
    """Aggregate cash flow impacts from climate risks/opportunities."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    scenario_id: str = Field(default="")
    operating_cash_flow_impact_usd: Decimal = Field(default=Decimal("0"))
    investing_cash_flow_impact_usd: Decimal = Field(default=Decimal("0"))
    financing_cash_flow_impact_usd: Decimal = Field(default=Decimal("0"))
    net_cash_flow_impact_usd: Decimal = Field(default=Decimal("0"))
    free_cash_flow_impact_pct: Decimal = Field(default=Decimal("0"))
    created_at: datetime = Field(default_factory=_now)


class NPVAnalysis(BaseModel):
    """Net Present Value analysis for climate-related investments."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    analysis_name: str = Field(..., max_length=255)
    discount_rate: Decimal = Field(default=Decimal("0.08"))
    projection_years: int = Field(default=30, ge=1, le=100)
    cash_flows: Dict[int, Decimal] = Field(
        default_factory=dict, description="Year -> cash flow (USD)",
    )
    npv_usd: Decimal = Field(default=Decimal("0"))
    irr_pct: Optional[Decimal] = Field(None)
    payback_period_years: Optional[Decimal] = Field(None, ge=Decimal("0"))
    created_at: datetime = Field(default_factory=_now)


class MACCEntry(BaseModel):
    """Marginal Abatement Cost Curve entry."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    measure_name: str = Field(..., max_length=255)
    abatement_potential_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    cost_per_tco2e_usd: Decimal = Field(default=Decimal("0"))
    capex_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    annual_opex_usd: Decimal = Field(default=Decimal("0"))
    implementation_year: int = Field(default=2025, ge=2020, le=2060)
    category: str = Field(default="", max_length=255)
    created_at: datetime = Field(default_factory=_now)


class CarbonPriceSensitivity(BaseModel):
    """Carbon price sensitivity analysis result."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    carbon_price_usd: Decimal = Field(..., description="Carbon price tested (USD/tCO2e)")
    annual_cost_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    emissions_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    emissions_scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    ebitda_impact_pct: Decimal = Field(default=Decimal("0"))
    breakeven_carbon_price_usd: Optional[Decimal] = Field(None)
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Risk Management Models (Pillar 3)
# ---------------------------------------------------------------------------

class RiskAssessment(BaseModel):
    """Structured risk assessment with likelihood x impact scoring."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    risk_id: str = Field(..., description="Climate risk ID")
    likelihood: RiskLikelihood = Field(default=RiskLikelihood.MEDIUM)
    likelihood_score: int = Field(default=3, ge=1, le=5)
    impact: RiskImpact = Field(default=RiskImpact.MODERATE)
    impact_score: int = Field(default=3, ge=1, le=5)
    risk_score: int = Field(default=9, ge=1, le=25, description="likelihood x impact")
    risk_rating: str = Field(default="medium", description="low, medium, high, critical")
    velocity: str = Field(
        default="medium",
        description="Speed of onset (immediate, fast, medium, slow, gradual)",
    )
    persistence: str = Field(
        default="medium",
        description="Duration (temporary, short_term, medium_term, long_term, permanent)",
    )
    assessed_by: Optional[str] = Field(None, max_length=255)
    assessed_at: datetime = Field(default_factory=_now)
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute risk score and rating."""
        computed = self.likelihood_score * self.impact_score
        object.__setattr__(self, "risk_score", computed)
        if computed <= 5:
            object.__setattr__(self, "risk_rating", "low")
        elif computed <= 12:
            object.__setattr__(self, "risk_rating", "medium")
        elif computed <= 19:
            object.__setattr__(self, "risk_rating", "high")
        else:
            object.__setattr__(self, "risk_rating", "critical")


class RiskRegisterEntry(BaseModel):
    """Entry in the climate risk register."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    risk_id: str = Field(..., description="Climate risk ID")
    risk_name: str = Field(default="", max_length=255)
    risk_type: RiskType = Field(default=RiskType.TRANSITION_POLICY)
    assessment: Optional[RiskAssessment] = Field(None)
    response: RiskResponse = Field(default=RiskResponse.MITIGATE)
    response_actions: List[str] = Field(default_factory=list)
    owner: Optional[str] = Field(None, max_length=255)
    review_frequency: str = Field(
        default="quarterly", description="monthly, quarterly, semi_annual, annual",
    )
    next_review_date: Optional[date] = Field(None)
    status: str = Field(default="active", description="active, mitigated, closed, accepted")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class RiskManagementRecord(BaseModel):
    """
    Risk management record per TCFD Pillar 3 (RM a/b/c).

    Tracks risk identification process, assessment methodology,
    scoring, response actions, and ERM integration status.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    risk_id: str = Field(..., description="Climate risk ID")
    identification_process: str = Field(
        default="", max_length=2000, description="How the risk was identified",
    )
    assessment_methodology: str = Field(
        default="", max_length=2000, description="Methodology used for assessment",
    )
    assessment: Optional[RiskAssessment] = Field(None)
    response_type: RiskResponse = Field(default=RiskResponse.ACCEPT)
    response_actions: List[str] = Field(
        default_factory=list, description="Planned response actions",
    )
    response_cost_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Cost of response actions",
    )
    residual_risk_score: Optional[int] = Field(
        None, ge=1, le=25, description="Risk score after mitigation",
    )
    owner: Optional[str] = Field(None, max_length=255, description="Risk owner")
    review_date: Optional[date] = Field(None, description="Next review date")
    erm_integrated: bool = Field(
        default=False, description="Whether integrated into enterprise risk management",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ERMIntegration(BaseModel):
    """Enterprise Risk Management integration status."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    erm_framework: str = Field(
        default="COSO", max_length=100, description="ERM framework (COSO, ISO 31000)",
    )
    climate_risk_in_erm: bool = Field(default=False)
    integration_level: str = Field(
        default="partial", description="none, partial, full",
    )
    risk_appetite_defined: bool = Field(default=False)
    climate_risk_appetite_statement: Optional[str] = Field(None, max_length=2000)
    board_risk_committee_oversight: bool = Field(default=False)
    reporting_frequency: str = Field(
        default="quarterly", description="monthly, quarterly, semi_annual, annual",
    )
    last_review_date: Optional[date] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class RiskIndicator(BaseModel):
    """Key Risk Indicator (KRI) for climate risk monitoring."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    indicator_name: str = Field(..., max_length=255)
    current_value: Decimal = Field(default=Decimal("0"))
    threshold_amber: Decimal = Field(default=Decimal("0"))
    threshold_red: Decimal = Field(default=Decimal("0"))
    unit: str = Field(default="", max_length=50)
    status: str = Field(default="green", description="green, amber, red")
    trend: str = Field(default="stable", description="improving, stable, deteriorating")
    related_risk_id: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Metrics & Targets Models (Pillar 4)
# ---------------------------------------------------------------------------

class MetricValue(BaseModel):
    """A single value for a climate metric at a point in time."""

    id: str = Field(default_factory=_new_id)
    metric_id: str = Field(..., description="Parent metric ID")
    reporting_year: int = Field(..., ge=1990, le=2100)
    value: Decimal = Field(default=Decimal("0"))
    unit: str = Field(default="", max_length=50)
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    source: str = Field(default="", max_length=255)
    verified: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_now)


class ClimateMetric(BaseModel):
    """
    A climate-related metric per TCFD MT(a) and ISSB para 29.

    Tracks metric type, historical values, data quality,
    and industry benchmark comparison.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    metric_category: MetricCategory = Field(
        default=MetricCategory.CROSS_INDUSTRY,
    )
    issb_metric_type: Optional[ISSBMetricType] = Field(None)
    metric_name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    unit: str = Field(default="", max_length=50)
    values: List[MetricValue] = Field(default_factory=list)
    current_value: Decimal = Field(default=Decimal("0"))
    reporting_year: int = Field(default=2025, ge=1990, le=2100)
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    industry_benchmark: Optional[Decimal] = Field(None)
    peer_percentile: Optional[int] = Field(None, ge=0, le=100)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.metric_name}:{self.current_value}:"
                f"{self.reporting_year}"
            )
            self.provenance_hash = _sha256(payload)


class EmissionsMetric(BaseModel):
    """GHG emissions metric with Scope 1/2/3 breakdown per TCFD MT(b)."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    reporting_year: int = Field(..., ge=1990, le=2100)
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_by_category: Dict[int, Decimal] = Field(
        default_factory=dict, description="Scope 3 emissions by category (1-15)",
    )
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    biogenic_co2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    removals_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    methodology: str = Field(
        default="GHG Protocol", max_length=255,
    )
    verification_status: str = Field(
        default="not_verified",
        description="not_verified, limited, reasonable",
    )
    data_quality: DataQualityTier = Field(default=DataQualityTier.CALCULATED)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute total and provenance hash."""
        if self.total_tco2e == Decimal("0"):
            computed = (
                self.scope1_tco2e
                + self.scope2_market_tco2e
                + self.scope3_tco2e
            )
            if computed > 0:
                object.__setattr__(self, "total_tco2e", computed)
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.reporting_year}:{self.scope1_tco2e}:"
                f"{self.scope2_market_tco2e}:{self.scope3_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


class IntensityMetric(BaseModel):
    """Emissions intensity metric (e.g. tCO2e per million USD revenue)."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    metric_name: str = Field(..., max_length=255)
    numerator_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    denominator_value: Decimal = Field(default=Decimal("1"), ge=Decimal("0"))
    denominator_unit: str = Field(
        default="million_usd_revenue", max_length=100,
    )
    intensity_value: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    reporting_year: int = Field(default=2025, ge=1990, le=2100)
    scope_coverage: str = Field(
        default="scope_1_2", description="scope_1, scope_2, scope_1_2, scope_1_2_3",
    )
    industry_benchmark: Optional[Decimal] = Field(None)
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute intensity."""
        if (
            self.intensity_value == Decimal("0")
            and self.denominator_value > 0
        ):
            object.__setattr__(
                self, "intensity_value",
                self.numerator_tco2e / self.denominator_value,
            )


class CrossIndustryMetric(BaseModel):
    """ISSB cross-industry metric value (one of 7 required metrics)."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    issb_metric_type: ISSBMetricType = Field(...)
    metric_name: str = Field(default="", max_length=255)
    value: Decimal = Field(default=Decimal("0"))
    unit: str = Field(default="", max_length=50)
    reporting_year: int = Field(default=2025, ge=1990, le=2100)
    ifrs_s2_paragraph: str = Field(default="", max_length=20)
    disclosed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Target Models (Pillar 4 - MT(c))
# ---------------------------------------------------------------------------

class ClimateTarget(BaseModel):
    """
    A climate-related target per TCFD MT(c).

    Tracks target type, base/target year values, interim milestones,
    SBTi alignment, and current status.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    target_type: TargetType = Field(..., description="Target type")
    target_name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    scope_coverage: str = Field(
        default="scope_1_2", description="Scope coverage (scope_1, scope_2, scope_1_2, all)",
    )
    base_year: int = Field(..., ge=1990, le=2100)
    base_value: Decimal = Field(..., description="Base year value")
    target_year: int = Field(..., ge=1990, le=2100)
    target_value: Decimal = Field(..., description="Target value")
    reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Reduction target percentage",
    )
    unit: str = Field(default="tCO2e", max_length=50)
    interim_milestones: Dict[int, Decimal] = Field(
        default_factory=dict, description="Year -> milestone value",
    )
    sbti_alignment: SBTiAlignment = Field(
        default=SBTiAlignment.NOT_ALIGNED,
    )
    sbti_validated: bool = Field(default=False)
    sbti_validation_date: Optional[date] = Field(None)
    status: str = Field(
        default="active", description="active, achieved, revised, expired",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("target_year")
    @classmethod
    def target_after_base(cls, v: int, info) -> int:
        """Target year must be after base year."""
        base = info.data.get("base_year")
        if base is not None and v <= base:
            raise ValueError("target_year must be after base_year")
        return v


class TargetProgress(BaseModel):
    """Annual progress record against a climate target."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    target_id: str = Field(..., description="Target ID")
    reporting_year: int = Field(..., ge=1990, le=2100)
    current_value: Decimal = Field(..., description="Current year value")
    progress_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("200"),
    )
    gap_to_target: Decimal = Field(default=Decimal("0"))
    annual_change_pct: Decimal = Field(
        default=Decimal("0"), description="Year-over-year change percentage",
    )
    on_track: bool = Field(default=False, description="Whether on track")
    required_annual_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Required annual reduction to meet target",
    )
    notes: Optional[str] = Field(None, max_length=2000)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.target_id}:{self.reporting_year}:"
                f"{self.current_value}:{self.progress_pct}"
            )
            self.provenance_hash = _sha256(payload)


class SBTiAssessment(BaseModel):
    """Science Based Targets initiative alignment assessment."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    commitment_status: str = Field(
        default="not_committed",
        description="not_committed, committed, targets_set, validated",
    )
    near_term_target_id: Optional[str] = Field(None)
    long_term_target_id: Optional[str] = Field(None)
    alignment: SBTiAlignment = Field(default=SBTiAlignment.NOT_ALIGNED)
    validation_date: Optional[date] = Field(None)
    annual_reduction_rate_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    required_reduction_rate_pct: Decimal = Field(
        default=Decimal("4.2"), ge=Decimal("0"),
        description="SBTi minimum annual reduction rate",
    )
    gap_pct: Decimal = Field(
        default=Decimal("0"),
        description="Gap between actual and required reduction rate",
    )
    recommendations: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class GapToTarget(BaseModel):
    """Gap analysis between current performance and target."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    target_id: str = Field(..., description="Target ID")
    current_value: Decimal = Field(default=Decimal("0"))
    target_value: Decimal = Field(default=Decimal("0"))
    gap_absolute: Decimal = Field(default=Decimal("0"))
    gap_pct: Decimal = Field(default=Decimal("0"))
    years_remaining: int = Field(default=0, ge=0)
    required_annual_reduction: Decimal = Field(default=Decimal("0"))
    trajectory_status: str = Field(
        default="off_track", description="on_track, at_risk, off_track",
    )
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Disclosure Models
# ---------------------------------------------------------------------------

class DisclosureEvidence(BaseModel):
    """Evidence document supporting a disclosure section."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    section_id: str = Field(..., description="Disclosure section ID")
    file_name: str = Field(..., min_length=1, max_length=255)
    file_type: str = Field(default="pdf", max_length=10)
    file_size_bytes: int = Field(default=0, ge=0)
    file_path: Optional[str] = Field(None)
    description: Optional[str] = Field(None, max_length=500)
    uploaded_by: Optional[str] = Field(None, max_length=255)
    uploaded_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = f"{self.file_name}:{self.file_size_bytes}:{self.uploaded_at}"
            self.provenance_hash = _sha256(payload)


class ComplianceScore(BaseModel):
    """Compliance scoring for a TCFD disclosure pillar or section."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    disclosure_id: str = Field(..., description="Parent disclosure ID")
    pillar: TCFDPillar = Field(...)
    disclosure_code: Optional[DisclosureCode] = Field(None)
    score: int = Field(default=0, ge=0, le=100, description="Compliance score (0-100)")
    max_score: int = Field(default=100, ge=0, le=100)
    completeness_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    quality_rating: str = Field(
        default="partial", description="none, partial, substantial, full",
    )
    gaps_identified: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_now)


class DisclosureSection(BaseModel):
    """
    A section of a TCFD disclosure document.

    Maps to one of the 11 recommended disclosures and tracks
    content, evidence references, and compliance scoring.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    disclosure_id: str = Field(default="", description="Parent disclosure ID")
    pillar: TCFDPillar = Field(..., description="TCFD pillar")
    disclosure_code: DisclosureCode = Field(
        ..., description="Disclosure code (e.g. GOV_A, STRAT_C)",
    )
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(default="", max_length=50000, description="Disclosure content")
    evidence: List[DisclosureEvidence] = Field(default_factory=list)
    compliance_score: int = Field(
        default=0, ge=0, le=100, description="Section compliance score (0-100)",
    )
    reviewer_notes: Optional[str] = Field(None, max_length=5000)
    status: DisclosureStatus = Field(default=DisclosureStatus.DRAFT)
    assigned_to: Optional[str] = Field(None, max_length=255)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class TCFDDisclosure(BaseModel):
    """
    A complete TCFD disclosure document for an organization-year.

    Contains all 11 recommended disclosure sections, completeness
    scoring, approval tracking, and lifecycle status.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    reporting_year: int = Field(..., ge=1990, le=2100)
    version: int = Field(default=1, ge=1, description="Disclosure version")
    status: DisclosureStatus = Field(
        default=DisclosureStatus.DRAFT, description="Disclosure lifecycle status",
    )
    sections: List[DisclosureSection] = Field(
        default_factory=list, description="Disclosure sections",
    )
    pillar_scores: Dict[str, int] = Field(
        default_factory=dict, description="Per-pillar compliance scores",
    )
    completeness_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    approved_by: Optional[str] = Field(None, max_length=255)
    approved_at: Optional[datetime] = Field(None)
    published_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute completeness and provenance."""
        if self.sections:
            total_score = sum(s.compliance_score for s in self.sections)
            avg = Decimal(str(total_score)) / Decimal(str(len(self.sections)))
            object.__setattr__(self, "completeness_score", avg)
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.reporting_year}:{self.version}:"
                f"{self.completeness_score}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# ISSB Cross-Walk Models
# ---------------------------------------------------------------------------

class ISSBMapping(BaseModel):
    """
    Mapping between a TCFD disclosure reference and IFRS S2 paragraph.

    Tracks mapping status and identifies gaps requiring additional
    disclosures to achieve IFRS S2 compliance.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    tcfd_disclosure_code: DisclosureCode = Field(...)
    ifrs_s2_paragraph: str = Field(
        ..., max_length=20, description="IFRS S2 paragraph reference",
    )
    mapping_status: str = Field(
        default="fully_mapped",
        description="fully_mapped, enhanced, partial, gap",
    )
    gap_description: Optional[str] = Field(None, max_length=2000)
    action_required: Optional[str] = Field(None, max_length=2000)
    created_at: datetime = Field(default_factory=_now)


class ISSBGap(BaseModel):
    """Identified gap between TCFD disclosure and IFRS S2 requirements."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    ifrs_s2_paragraph: str = Field(..., max_length=20)
    gap_type: str = Field(
        default="missing", description="missing, incomplete, quality",
    )
    severity: str = Field(
        default="medium", description="low, medium, high, critical",
    )
    description: str = Field(default="", max_length=2000)
    remediation_action: str = Field(default="", max_length=2000)
    effort_estimate: str = Field(
        default="medium", description="low, medium, high",
    )
    target_completion_date: Optional[date] = Field(None)
    status: str = Field(default="open", description="open, in_progress, closed")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class MigrationPathway(BaseModel):
    """Migration pathway from TCFD to IFRS S2 compliance."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    current_tcfd_score: int = Field(default=0, ge=0, le=100)
    target_ifrs_s2_readiness: int = Field(default=100, ge=0, le=100)
    gaps: List[ISSBGap] = Field(default_factory=list)
    total_gaps: int = Field(default=0, ge=0)
    critical_gaps: int = Field(default=0, ge=0)
    estimated_completion_months: int = Field(default=0, ge=0)
    migration_status: str = Field(
        default="not_started",
        description="not_started, in_progress, substantially_complete, complete",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Gap Analysis Models
# ---------------------------------------------------------------------------

class MaturityScore(BaseModel):
    """Maturity score for a specific TCFD pillar or dimension."""

    pillar: TCFDPillar = Field(...)
    score: int = Field(default=1, ge=1, le=5)
    level: GovernanceMaturityLevel = Field(default=GovernanceMaturityLevel.INITIAL)
    description: str = Field(default="", max_length=500)


class PeerBenchmark(BaseModel):
    """Peer benchmarking data for a TCFD pillar."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    sector: SectorType = Field(default=SectorType.ENERGY)
    pillar: TCFDPillar = Field(...)
    org_score: int = Field(default=0, ge=0, le=100)
    peer_average: int = Field(default=0, ge=0, le=100)
    peer_median: int = Field(default=0, ge=0, le=100)
    peer_p25: int = Field(default=0, ge=0, le=100)
    peer_p75: int = Field(default=0, ge=0, le=100)
    percentile: int = Field(default=0, ge=0, le=100)
    peer_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_now)


class ActionItem(BaseModel):
    """A specific action item for closing a TCFD gap."""

    id: str = Field(default_factory=_new_id)
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    pillar: TCFDPillar = Field(...)
    disclosure_code: Optional[DisclosureCode] = Field(None)
    priority: int = Field(default=3, ge=1, le=5, description="1=highest")
    effort: str = Field(default="medium", description="low, medium, high")
    expected_impact: str = Field(default="medium", description="low, medium, high")
    owner: Optional[str] = Field(None, max_length=255)
    due_date: Optional[date] = Field(None)
    status: str = Field(default="planned", description="planned, in_progress, completed")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ActionPlan(BaseModel):
    """Complete action plan for improving TCFD disclosure quality."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    plan_name: str = Field(default="TCFD Improvement Plan", max_length=255)
    actions: List[ActionItem] = Field(default_factory=list)
    total_actions: int = Field(default=0, ge=0)
    completed_actions: int = Field(default=0, ge=0)
    completion_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    target_maturity: GovernanceMaturityLevel = Field(
        default=GovernanceMaturityLevel.MANAGED,
    )
    target_date: Optional[date] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute counts."""
        if self.actions and self.total_actions == 0:
            object.__setattr__(self, "total_actions", len(self.actions))
            object.__setattr__(
                self, "completed_actions",
                sum(1 for a in self.actions if a.status == "completed"),
            )
            if len(self.actions) > 0:
                pct = Decimal(str(self.completed_actions)) / Decimal(str(len(self.actions))) * 100
                object.__setattr__(self, "completion_pct", pct)


class GapAssessment(BaseModel):
    """
    Comprehensive disclosure gap analysis and maturity assessment.

    Evaluates organizational readiness across all four TCFD pillars
    and identifies gaps with recommended actions.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    assessment_date: date = Field(default_factory=lambda: date.today())
    pillar_scores: List[MaturityScore] = Field(default_factory=list)
    overall_score: int = Field(default=0, ge=0, le=100)
    overall_maturity: GovernanceMaturityLevel = Field(
        default=GovernanceMaturityLevel.INITIAL,
    )
    gaps: List[Dict[str, str]] = Field(
        default_factory=list, description="Identified gaps",
    )
    total_gaps: int = Field(default=0, ge=0)
    critical_gaps: int = Field(default=0, ge=0)
    action_plan: Optional[ActionPlan] = Field(None)
    peer_benchmarks: List[PeerBenchmark] = Field(default_factory=list)
    issb_migration: Optional[MigrationPathway] = Field(None)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.assessment_date}:"
                f"{self.overall_maturity}:{self.total_gaps}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Recommendation Models
# ---------------------------------------------------------------------------

class ImprovementAction(BaseModel):
    """A specific step within a recommendation."""

    step_number: int = Field(..., ge=1)
    action: str = Field(..., max_length=500)
    estimated_effort_hours: Optional[int] = Field(None, ge=0)
    responsible_role: Optional[str] = Field(None, max_length=255)


class Recommendation(BaseModel):
    """
    An improvement recommendation for TCFD disclosure quality.

    Generated from gap analysis, peer benchmarking, and best practice
    comparison to prioritize organizational improvements.
    """

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    pillar: TCFDPillar = Field(...)
    disclosure_code: Optional[DisclosureCode] = Field(None)
    priority: int = Field(default=3, ge=1, le=5, description="1=highest, 5=lowest")
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=5000)
    estimated_impact: str = Field(
        default="medium", description="low, medium, high",
    )
    estimated_effort: str = Field(
        default="medium", description="low, medium, high",
    )
    implementation_steps: List[ImprovementAction] = Field(default_factory=list)
    reference_standard: Optional[str] = Field(
        None, max_length=255, description="e.g. TCFD Guidance, IFRS S2 para X",
    )
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# API Request Models
# ---------------------------------------------------------------------------

class CreateOrganizationRequest(BaseModel):
    """Request to create a new organization."""

    name: str = Field(..., min_length=1, max_length=500)
    sector: SectorType = Field(default=SectorType.ENERGY)
    country: str = Field(..., min_length=2, max_length=3)
    description: Optional[str] = Field(None, max_length=2000)
    employee_count: Optional[int] = Field(None, ge=0)
    annual_revenue_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    total_assets_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)


class CreateGovernanceAssessmentRequest(BaseModel):
    """Request to create a governance assessment."""

    board_oversight_score: int = Field(default=1, ge=1, le=5)
    board_committees: List[str] = Field(default_factory=list)
    meeting_frequency: int = Field(default=0, ge=0)
    climate_competency_score: int = Field(default=1, ge=1, le=5)
    incentive_linkage: bool = Field(default=False)
    incentive_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    notes: Optional[str] = Field(None, max_length=5000)


class CreateClimateRiskRequest(BaseModel):
    """Request to create a climate risk."""

    risk_type: RiskType = Field(...)
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=5000)
    category: str = Field(default="")
    time_horizon: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    likelihood: RiskLikelihood = Field(default=RiskLikelihood.MEDIUM)
    impact: RiskImpact = Field(default=RiskImpact.MODERATE)
    financial_impact_mid_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    response_strategy: RiskResponse = Field(default=RiskResponse.MITIGATE)
    response_description: str = Field(default="", max_length=2000)
    owner: Optional[str] = Field(None, max_length=255)


class CreateClimateOpportunityRequest(BaseModel):
    """Request to create a climate opportunity."""

    category: OpportunityCategory = Field(...)
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=5000)
    revenue_potential_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    cost_savings_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    investment_required_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    timeline: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    feasibility_score: int = Field(default=3, ge=1, le=5)
    priority_score: int = Field(default=3, ge=1, le=5)


class CreateScenarioRequest(BaseModel):
    """Request to create or customize a scenario."""

    name: str = Field(..., min_length=1, max_length=255)
    scenario_type: ScenarioType = Field(default=ScenarioType.CUSTOM)
    temperature_outcome: TemperatureOutcome = Field(
        default=TemperatureOutcome.AROUND_2C,
    )
    description: str = Field(default="", max_length=5000)
    carbon_price_trajectory: Optional[Dict[int, Decimal]] = Field(None)
    energy_mix_trajectory: Optional[Dict[int, Dict[str, int]]] = Field(None)
    technology_assumptions: Optional[Dict[str, str]] = Field(None)
    regulatory_assumptions: Optional[Dict[str, str]] = Field(None)
    physical_assumptions: Optional[Dict[str, str]] = Field(None)


class RunScenarioAnalysisRequest(BaseModel):
    """Request to run a scenario analysis."""

    scenario_id: str = Field(...)
    revenue_base_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    cost_base_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total_assets_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    emissions_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    emissions_scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    emissions_scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    discount_rate: Optional[Decimal] = Field(None)
    enable_monte_carlo: bool = Field(default=True)


class RegisterAssetRequest(BaseModel):
    """Request to register a physical asset."""

    asset_name: str = Field(..., min_length=1, max_length=255)
    asset_type: AssetType = Field(default=AssetType.BUILDING)
    latitude: Decimal = Field(..., ge=Decimal("-90"), le=Decimal("90"))
    longitude: Decimal = Field(..., ge=Decimal("-180"), le=Decimal("180"))
    country: str = Field(..., min_length=2, max_length=3)
    region: Optional[str] = Field(None, max_length=100)
    elevation_m: Decimal = Field(default=Decimal("0"))
    building_type: Optional[str] = Field(None, max_length=100)
    replacement_value_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    year_built: Optional[int] = Field(None, ge=1800, le=2100)


class CreateTargetRequest(BaseModel):
    """Request to create a climate target."""

    target_type: TargetType = Field(...)
    target_name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    scope_coverage: str = Field(default="scope_1_2")
    base_year: int = Field(..., ge=1990, le=2100)
    base_value: Decimal = Field(...)
    target_year: int = Field(..., ge=1990, le=2100)
    target_value: Decimal = Field(...)
    unit: str = Field(default="tCO2e")
    interim_milestones: Optional[Dict[int, Decimal]] = Field(None)
    sbti_alignment: SBTiAlignment = Field(default=SBTiAlignment.NOT_ALIGNED)

    @field_validator("target_year")
    @classmethod
    def target_after_base(cls, v: int, info) -> int:
        """Target year must be after base year."""
        base = info.data.get("base_year")
        if base is not None and v <= base:
            raise ValueError("target_year must be after base_year")
        return v


class RecordMetricRequest(BaseModel):
    """Request to record a climate metric."""

    metric_category: MetricCategory = Field(
        default=MetricCategory.CROSS_INDUSTRY,
    )
    issb_metric_type: Optional[ISSBMetricType] = Field(None)
    metric_name: str = Field(..., min_length=1, max_length=255)
    value: Decimal = Field(...)
    unit: str = Field(default="")
    reporting_year: int = Field(..., ge=1990, le=2100)
    data_quality: DataQualityTier = Field(default=DataQualityTier.ESTIMATED)
    source: str = Field(default="")


class RecordEmissionsRequest(BaseModel):
    """Request to record GHG emissions data."""

    reporting_year: int = Field(..., ge=1990, le=2100)
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_by_category: Optional[Dict[int, Decimal]] = Field(None)
    methodology: str = Field(default="GHG Protocol")


class CreateDisclosureRequest(BaseModel):
    """Request to create a TCFD disclosure."""

    reporting_year: int = Field(..., ge=1990, le=2100)
    version: int = Field(default=1, ge=1)


class UpdateDisclosureSectionRequest(BaseModel):
    """Request to update a disclosure section."""

    content: str = Field(default="", max_length=50000)
    reviewer_notes: Optional[str] = Field(None, max_length=5000)
    status: Optional[DisclosureStatus] = Field(None)


class UpdateDisclosureStatusRequest(BaseModel):
    """Request to update disclosure status."""

    status: DisclosureStatus = Field(...)
    approved_by: Optional[str] = Field(None, max_length=255)


class CreateRiskManagementRecordRequest(BaseModel):
    """Request to create a risk management record."""

    risk_id: str = Field(...)
    identification_process: str = Field(default="", max_length=2000)
    assessment_methodology: str = Field(default="", max_length=2000)
    likelihood: RiskLikelihood = Field(default=RiskLikelihood.MEDIUM)
    impact: RiskImpact = Field(default=RiskImpact.MODERATE)
    response_type: RiskResponse = Field(default=RiskResponse.MITIGATE)
    response_actions: List[str] = Field(default_factory=list)
    owner: Optional[str] = Field(None, max_length=255)
    erm_integrated: bool = Field(default=False)


class RunGapAnalysisRequest(BaseModel):
    """Request to run a TCFD gap analysis."""

    target_maturity: GovernanceMaturityLevel = Field(
        default=GovernanceMaturityLevel.MANAGED,
    )
    include_issb_crosswalk: bool = Field(default=True)
    include_peer_benchmark: bool = Field(default=True)


class GenerateReportRequest(BaseModel):
    """Request to generate a TCFD report."""

    format: ReportFormat = Field(default=ReportFormat.PDF)
    include_scenario_analysis: bool = Field(default=True)
    include_financial_impact: bool = Field(default=True)
    include_gap_analysis: bool = Field(default=True)
    include_issb_crosswalk: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)


class UpdateSettingsRequest(BaseModel):
    """Request to update platform configuration."""

    default_scenario: Optional[ScenarioType] = Field(None)
    physical_risk_data_source: Optional[str] = Field(None)
    enable_monte_carlo: Optional[bool] = Field(None)
    monte_carlo_iterations: Optional[int] = Field(None, ge=1000, le=1000000)
    default_discount_rate: Optional[Decimal] = Field(None)
    currency: Optional[str] = Field(None)
    financial_projection_years: Optional[int] = Field(None, ge=5, le=100)
    sbti_validation_enabled: Optional[bool] = Field(None)
    default_report_format: Optional[ReportFormat] = Field(None)
    log_level: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Generic API Response Models
# ---------------------------------------------------------------------------

class ApiError(BaseModel):
    """Standard API error response."""

    code: str = Field(..., description="Error code (e.g. VALIDATION_ERROR)")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None)
    timestamp: datetime = Field(default_factory=_now)


class ApiResponse(BaseModel):
    """Standard API success response wrapper."""

    success: bool = Field(default=True)
    data: Optional[Any] = Field(None, description="Response payload")
    message: str = Field(default="OK")
    errors: List[ApiError] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)
    provenance_hash: Optional[str] = Field(None)


class PaginatedResponse(BaseModel):
    """Paginated list response for collection endpoints."""

    items: List[Any] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)
    total_pages: int = Field(default=0, ge=0)
    has_next: bool = Field(default=False)
    has_previous: bool = Field(default=False)

    def model_post_init(self, __context: Any) -> None:
        """Compute pagination metadata."""
        if self.page_size > 0 and self.total > 0:
            computed_pages = (self.total + self.page_size - 1) // self.page_size
            object.__setattr__(self, "total_pages", computed_pages)
            object.__setattr__(self, "has_next", self.page < computed_pages)
            object.__setattr__(self, "has_previous", self.page > 1)
