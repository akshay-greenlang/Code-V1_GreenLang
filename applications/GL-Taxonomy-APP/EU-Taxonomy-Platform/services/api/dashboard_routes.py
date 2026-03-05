"""
GL-Taxonomy-APP Dashboard API

Provides pre-aggregated metrics and KPIs for the EU Taxonomy alignment
executive dashboard.  Returns overview cards, alignment summaries, KPI
status, sector breakdowns, trend data, and eligibility funnel charts
in endpoints optimized for frontend rendering.
"""

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/api/v1/taxonomy/dashboard", tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class OverviewResponse(BaseModel):
    """Executive dashboard overview."""
    org_id: str
    reporting_year: int
    overall_alignment_pct: float
    turnover_alignment_pct: float
    capex_alignment_pct: float
    opex_alignment_pct: float
    total_activities: int
    eligible_activities: int
    aligned_activities: int
    capex_plan_active: bool
    regulatory_status: str
    last_report_date: Optional[str]
    generated_at: datetime


class AlignmentSummaryResponse(BaseModel):
    """Alignment summary with step breakdown."""
    org_id: str
    total_activities: int
    eligible: int
    sc_passed: int
    dnsh_passed: int
    safeguards_passed: int
    fully_aligned: int
    by_objective: Dict[str, int]
    by_sector: Dict[str, int]
    conversion_rate_pct: float
    generated_at: datetime


class KPICardsResponse(BaseModel):
    """KPI summary cards for dashboard."""
    org_id: str
    cards: List[Dict[str, Any]]
    reporting_year: int
    trend_vs_prior_year: Dict[str, float]
    generated_at: datetime


class SectorBreakdownResponse(BaseModel):
    """Sector pie chart data."""
    org_id: str
    sectors: List[Dict[str, Any]]
    total_aligned_turnover_eur: float
    generated_at: datetime


class TrendsResponse(BaseModel):
    """Alignment trends over time."""
    org_id: str
    periods: List[Dict[str, Any]]
    trend: str
    projected_next_year_pct: float
    generated_at: datetime


class EligibilityFunnelResponse(BaseModel):
    """Eligibility funnel data."""
    org_id: str
    funnel_stages: List[Dict[str, Any]]
    conversion_rate_pct: float
    biggest_dropout: str
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
    "/{org_id}/overview",
    response_model=OverviewResponse,
    summary="Executive dashboard",
    description="Get the executive dashboard overview with key taxonomy metrics.",
)
async def get_overview(
    org_id: str,
    reporting_year: int = Query(2025, ge=2022, le=2030),
) -> OverviewResponse:
    """Get executive dashboard overview."""
    return OverviewResponse(
        org_id=org_id,
        reporting_year=reporting_year,
        overall_alignment_pct=42.0,
        turnover_alignment_pct=42.0,
        capex_alignment_pct=56.0,
        opex_alignment_pct=46.7,
        total_activities=30,
        eligible_activities=20,
        aligned_activities=12,
        capex_plan_active=True,
        regulatory_status="csrd_compliant",
        last_report_date="2025-03-31",
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/alignment-summary",
    response_model=AlignmentSummaryResponse,
    summary="Alignment summary",
    description="Get alignment summary with step-by-step breakdown.",
)
async def get_alignment_summary(org_id: str) -> AlignmentSummaryResponse:
    """Get alignment summary."""
    return AlignmentSummaryResponse(
        org_id=org_id,
        total_activities=30,
        eligible=20,
        sc_passed=16,
        dnsh_passed=14,
        safeguards_passed=12,
        fully_aligned=12,
        by_objective={
            "climate_change_mitigation": 10,
            "climate_change_adaptation": 1,
            "water": 0,
            "circular_economy": 1,
            "pollution_prevention": 0,
            "biodiversity": 0,
        },
        by_sector={
            "Energy": 5,
            "Construction & Real Estate": 3,
            "Transport": 2,
            "Manufacturing": 1,
            "ICT": 1,
        },
        conversion_rate_pct=40.0,
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/kpi-cards",
    response_model=KPICardsResponse,
    summary="KPI summary cards",
    description="Get KPI summary cards for dashboard display.",
)
async def get_kpi_cards(
    org_id: str,
    reporting_year: int = Query(2025, ge=2022, le=2030),
) -> KPICardsResponse:
    """Get KPI summary cards."""
    return KPICardsResponse(
        org_id=org_id,
        cards=[
            {"kpi": "Turnover", "eligible_pct": 65.0, "aligned_pct": 42.0, "total_eur": 100000000, "aligned_eur": 42000000, "trend": "up", "change_pct": 6.5},
            {"kpi": "CapEx", "eligible_pct": 72.0, "aligned_pct": 56.0, "total_eur": 25000000, "aligned_eur": 14000000, "trend": "up", "change_pct": 11.0},
            {"kpi": "OpEx", "eligible_pct": 66.7, "aligned_pct": 46.7, "total_eur": 15000000, "aligned_eur": 7000000, "trend": "up", "change_pct": 8.7},
        ],
        reporting_year=reporting_year,
        trend_vs_prior_year={
            "turnover_change_pp": 6.5,
            "capex_change_pp": 11.0,
            "opex_change_pp": 8.7,
        },
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/sector-breakdown",
    response_model=SectorBreakdownResponse,
    summary="Sector pie chart data",
    description="Get sector breakdown of aligned turnover for pie chart visualization.",
)
async def get_sector_breakdown(org_id: str) -> SectorBreakdownResponse:
    """Get sector breakdown."""
    sectors = [
        {"sector": "Energy", "aligned_turnover_eur": 15000000, "pct_of_aligned": 35.7, "activity_count": 5, "color": "#22c55e"},
        {"sector": "Construction & Real Estate", "aligned_turnover_eur": 12000000, "pct_of_aligned": 28.6, "activity_count": 4, "color": "#3b82f6"},
        {"sector": "Transport", "aligned_turnover_eur": 8000000, "pct_of_aligned": 19.0, "activity_count": 3, "color": "#f59e0b"},
        {"sector": "Manufacturing", "aligned_turnover_eur": 5000000, "pct_of_aligned": 11.9, "activity_count": 2, "color": "#8b5cf6"},
        {"sector": "ICT", "aligned_turnover_eur": 2000000, "pct_of_aligned": 4.8, "activity_count": 1, "color": "#06b6d4"},
    ]

    return SectorBreakdownResponse(
        org_id=org_id,
        sectors=sectors,
        total_aligned_turnover_eur=sum(s["aligned_turnover_eur"] for s in sectors),
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/trends",
    response_model=TrendsResponse,
    summary="Alignment trends over time",
    description="Get alignment trends across reporting periods for line chart.",
)
async def get_trends(org_id: str) -> TrendsResponse:
    """Get alignment trends."""
    periods = [
        {"year": 2022, "turnover_alignment_pct": 15.0, "capex_alignment_pct": 22.0, "opex_alignment_pct": 18.0},
        {"year": 2023, "turnover_alignment_pct": 25.0, "capex_alignment_pct": 35.0, "opex_alignment_pct": 28.0},
        {"year": 2024, "turnover_alignment_pct": 35.5, "capex_alignment_pct": 45.0, "opex_alignment_pct": 38.0},
        {"year": 2025, "turnover_alignment_pct": 42.0, "capex_alignment_pct": 56.0, "opex_alignment_pct": 46.7},
    ]

    return TrendsResponse(
        org_id=org_id,
        periods=periods,
        trend="improving",
        projected_next_year_pct=48.5,
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/eligible-funnel",
    response_model=EligibilityFunnelResponse,
    summary="Eligibility funnel data",
    description="Get eligibility-to-alignment funnel data for bar/funnel chart.",
)
async def get_eligibility_funnel(org_id: str) -> EligibilityFunnelResponse:
    """Get eligibility funnel data."""
    stages = [
        {"stage": "Total Activities", "count": 30, "pct_of_total": 100.0},
        {"stage": "Taxonomy Eligible", "count": 20, "pct_of_total": 66.7, "dropout": 10},
        {"stage": "SC Passed", "count": 16, "pct_of_total": 53.3, "dropout": 4},
        {"stage": "DNSH Passed", "count": 14, "pct_of_total": 46.7, "dropout": 2},
        {"stage": "Safeguards Passed", "count": 12, "pct_of_total": 40.0, "dropout": 2},
        {"stage": "Fully Aligned", "count": 12, "pct_of_total": 40.0, "dropout": 0},
    ]

    return EligibilityFunnelResponse(
        org_id=org_id,
        funnel_stages=stages,
        conversion_rate_pct=40.0,
        biggest_dropout="Not Eligible (10 activities)",
        generated_at=_now(),
    )
