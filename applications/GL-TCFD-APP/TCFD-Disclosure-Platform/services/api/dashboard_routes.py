"""
GL-TCFD-APP Dashboard API

Provides pre-aggregated metrics and KPIs for the TCFD disclosure dashboard.
Returns executive summary, risk/opportunity balance, scenario comparison,
physical risk map data, transition risk radar, disclosure progress, metrics
summary, and year-over-year trends in a set of efficient endpoints for
frontend rendering.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/api/v1/tcfd/dashboard", tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ExecutiveSummaryResponse(BaseModel):
    """Executive KPI summary."""
    org_id: str
    total_risks: int
    total_opportunities: int
    net_climate_impact_usd: float
    governance_maturity: str
    governance_score: float
    disclosure_completeness_pct: float
    scope1_tco2e: float
    scope2_tco2e: float
    scope3_tco2e: float
    target_progress_pct: float
    sbti_status: str
    physical_risk_score: float
    transition_risk_score: float
    resilience_score: float
    top_risk: Dict[str, Any]
    top_opportunity: Dict[str, Any]
    generated_at: datetime


class RiskOpportunityBalanceResponse(BaseModel):
    """Risk/opportunity balance."""
    org_id: str
    total_risk_exposure_usd: float
    total_opportunity_value_usd: float
    net_balance_usd: float
    risk_count: int
    opportunity_count: int
    by_time_horizon: Dict[str, Dict[str, float]]
    by_category: Dict[str, float]
    generated_at: datetime


class ScenarioComparisonDashResponse(BaseModel):
    """Scenario comparison dashboard data."""
    org_id: str
    scenarios: List[Dict[str, Any]]
    carbon_cost_comparison: Dict[str, float]
    asset_impairment_comparison: Dict[str, float]
    resilience_comparison: Dict[str, float]
    generated_at: datetime


class PhysicalRiskMapDashResponse(BaseModel):
    """Physical risk map dashboard data."""
    org_id: str
    total_assets: int
    high_risk_assets: int
    total_annualized_loss_usd: float
    risk_distribution: Dict[str, int]
    top_hazards: List[Dict[str, Any]]
    generated_at: datetime


class TransitionRadarResponse(BaseModel):
    """Transition risk radar chart data."""
    org_id: str
    policy_score: float
    technology_score: float
    market_score: float
    reputation_score: float
    composite_score: float
    peer_average: Dict[str, float]
    generated_at: datetime


class DisclosureProgressResponse(BaseModel):
    """Disclosure completeness progress."""
    org_id: str
    overall_completeness_pct: float
    governance_pct: float
    strategy_pct: float
    risk_management_pct: float
    metrics_pct: float
    sections_completed: int
    sections_total: int
    sections_reviewed: int
    status: str
    generated_at: datetime


class MetricsSummaryDashResponse(BaseModel):
    """Cross-industry metrics summary."""
    org_id: str
    ghg_emissions_tco2e: float
    yoy_change_pct: float
    intensity_revenue: float
    internal_carbon_price: float
    capital_deployed_usd: float
    remuneration_linked_pct: float
    targets_on_track: int
    targets_total: int
    generated_at: datetime


class TrendsDashResponse(BaseModel):
    """Year-over-year trends."""
    org_id: str
    emissions_trend: Dict[str, float]
    risk_trend: Dict[str, float]
    disclosure_trend: Dict[str, float]
    target_progress_trend: Dict[str, float]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/summary/{org_id}",
    response_model=ExecutiveSummaryResponse,
    summary="Executive KPI summary",
    description="Get a comprehensive executive-level summary of all TCFD KPIs for dashboard rendering.",
)
async def get_executive_summary(org_id: str) -> ExecutiveSummaryResponse:
    """Get executive KPI summary."""
    return ExecutiveSummaryResponse(
        org_id=org_id,
        total_risks=18,
        total_opportunities=12,
        net_climate_impact_usd=-15000000,
        governance_maturity="level_4_managed",
        governance_score=78.5,
        disclosure_completeness_pct=72.7,
        scope1_tco2e=25000,
        scope2_tco2e=15000,
        scope3_tco2e=120000,
        target_progress_pct=45.0,
        sbti_status="validated",
        physical_risk_score=0.52,
        transition_risk_score=58.0,
        resilience_score=72.0,
        top_risk={"name": "Carbon pricing regulation", "score": 85, "type": "transition_policy"},
        top_opportunity={"name": "Green product line", "value_usd": 25000000, "type": "products_services"},
        generated_at=_now(),
    )


@router.get(
    "/risk-opportunity/{org_id}",
    response_model=RiskOpportunityBalanceResponse,
    summary="Risk/opportunity balance",
    description="Get the balance between climate-related risks and opportunities.",
)
async def get_risk_opportunity_balance(org_id: str) -> RiskOpportunityBalanceResponse:
    """Get risk/opportunity balance."""
    return RiskOpportunityBalanceResponse(
        org_id=org_id,
        total_risk_exposure_usd=85000000,
        total_opportunity_value_usd=60000000,
        net_balance_usd=-25000000,
        risk_count=18,
        opportunity_count=12,
        by_time_horizon={
            "short_term": {"risk_usd": 15000000, "opportunity_usd": 10000000},
            "medium_term": {"risk_usd": 35000000, "opportunity_usd": 25000000},
            "long_term": {"risk_usd": 35000000, "opportunity_usd": 25000000},
        },
        by_category={
            "physical_risk": 35000000,
            "transition_risk": 50000000,
            "opportunities": -60000000,
        },
        generated_at=_now(),
    )


@router.get(
    "/scenario-comparison/{org_id}",
    response_model=ScenarioComparisonDashResponse,
    summary="Scenario comparison data",
    description="Get scenario comparison data for dashboard visualization.",
)
async def get_scenario_comparison(org_id: str) -> ScenarioComparisonDashResponse:
    """Get scenario comparison dashboard data."""
    return ScenarioComparisonDashResponse(
        org_id=org_id,
        scenarios=[
            {"name": "IEA NZE 2050", "warming": "1.5C", "carbon_cost_2050_usd": 5000000, "resilience": 72},
            {"name": "IEA APS", "warming": "1.7C", "carbon_cost_2050_usd": 3500000, "resilience": 78},
            {"name": "NGFS Delayed", "warming": "1.8C", "carbon_cost_2050_usd": 7000000, "resilience": 55},
            {"name": "NGFS Current", "warming": "3C+", "carbon_cost_2050_usd": 400000, "resilience": 40},
        ],
        carbon_cost_comparison={"nze": 5000000, "aps": 3500000, "delayed": 7000000, "current": 400000},
        asset_impairment_comparison={"nze": 45000000, "aps": 30000000, "delayed": 55000000, "current": 10000000},
        resilience_comparison={"nze": 72, "aps": 78, "delayed": 55, "current": 40},
        generated_at=_now(),
    )


@router.get(
    "/physical-risk-map/{org_id}",
    response_model=PhysicalRiskMapDashResponse,
    summary="Physical risk map data",
    description="Get aggregated physical risk data for map visualization.",
)
async def get_physical_risk_map(org_id: str) -> PhysicalRiskMapDashResponse:
    """Get physical risk map data."""
    return PhysicalRiskMapDashResponse(
        org_id=org_id,
        total_assets=25,
        high_risk_assets=5,
        total_annualized_loss_usd=3200000,
        risk_distribution={"negligible": 3, "low": 7, "medium": 10, "high": 4, "very_high": 1},
        top_hazards=[
            {"hazard": "flood_riverine", "affected_assets": 8, "total_loss_usd": 950000},
            {"hazard": "extreme_heat", "affected_assets": 12, "total_loss_usd": 780000},
            {"hazard": "cyclone", "affected_assets": 4, "total_loss_usd": 620000},
        ],
        generated_at=_now(),
    )


@router.get(
    "/transition-radar/{org_id}",
    response_model=TransitionRadarResponse,
    summary="Transition risk radar data",
    description="Get transition risk scores for radar chart visualization.",
)
async def get_transition_radar(org_id: str) -> TransitionRadarResponse:
    """Get transition risk radar data."""
    return TransitionRadarResponse(
        org_id=org_id,
        policy_score=65,
        technology_score=45,
        market_score=55,
        reputation_score=35,
        composite_score=52,
        peer_average={"policy": 58, "technology": 50, "market": 52, "reputation": 40},
        generated_at=_now(),
    )


@router.get(
    "/disclosure-progress/{org_id}",
    response_model=DisclosureProgressResponse,
    summary="Disclosure completeness",
    description="Get disclosure completeness progress by pillar.",
)
async def get_disclosure_progress(org_id: str) -> DisclosureProgressResponse:
    """Get disclosure progress."""
    return DisclosureProgressResponse(
        org_id=org_id,
        overall_completeness_pct=72.7,
        governance_pct=100.0,
        strategy_pct=66.7,
        risk_management_pct=66.7,
        metrics_pct=66.7,
        sections_completed=8,
        sections_total=11,
        sections_reviewed=5,
        status="in_progress",
        generated_at=_now(),
    )


@router.get(
    "/metrics-summary/{org_id}",
    response_model=MetricsSummaryDashResponse,
    summary="Cross-industry metrics summary",
    description="Get a summary of ISSB cross-industry metrics for dashboard display.",
)
async def get_metrics_summary(org_id: str) -> MetricsSummaryDashResponse:
    """Get metrics summary."""
    return MetricsSummaryDashResponse(
        org_id=org_id,
        ghg_emissions_tco2e=160000,
        yoy_change_pct=-4.2,
        intensity_revenue=80.0,
        internal_carbon_price=75.0,
        capital_deployed_usd=35000000,
        remuneration_linked_pct=15.0,
        targets_on_track=3,
        targets_total=5,
        generated_at=_now(),
    )


@router.get(
    "/trends/{org_id}",
    response_model=TrendsDashResponse,
    summary="Year-over-year trends",
    description="Get multi-year trend data for key TCFD metrics.",
)
async def get_trends(org_id: str) -> TrendsDashResponse:
    """Get year-over-year trends."""
    return TrendsDashResponse(
        org_id=org_id,
        emissions_trend={"2020": 50000, "2021": 48500, "2022": 46800, "2023": 44200, "2024": 42100, "2025": 40000},
        risk_trend={"2020": 45, "2021": 48, "2022": 52, "2023": 55, "2024": 56, "2025": 58},
        disclosure_trend={"2020": 20, "2021": 35, "2022": 50, "2023": 60, "2024": 68, "2025": 73},
        target_progress_trend={"2020": 0, "2021": 8, "2022": 18, "2023": 28, "2024": 38, "2025": 45},
        generated_at=_now(),
    )
