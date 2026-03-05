"""
GL-SBTi-APP Dashboard API

Provides pre-aggregated metrics and KPIs for the SBTi target-setting
dashboard.  Returns readiness scores, target status cards, pathway
overlay data, temperature gauge, review countdown, and key milestones
in a set of efficient endpoints optimized for frontend rendering.
"""

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/api/v1/sbti/dashboard", tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ReadinessScoreResponse(BaseModel):
    """Overall SBTi readiness score."""
    org_id: str
    overall_readiness_pct: float
    readiness_level: str
    category_scores: Dict[str, float]
    sbti_status: str
    validation_date: Optional[str]
    next_review_date: Optional[str]
    generated_at: datetime


class TargetSummaryDashResponse(BaseModel):
    """Target status cards for dashboard."""
    org_id: str
    total_targets: int
    active_targets: int
    draft_targets: int
    expired_targets: int
    targets: List[Dict[str, Any]]
    generated_at: datetime


class PathwayOverlayResponse(BaseModel):
    """Pathway with actuals overlay for chart rendering."""
    org_id: str
    pathway_years: Dict[str, float]
    actual_years: Dict[str, float]
    base_year: int
    target_year: int
    on_track: bool
    variance_current_pct: float
    generated_at: datetime


class TemperatureGaugeResponse(BaseModel):
    """Temperature gauge data."""
    org_id: str
    temperature_c: float
    alignment_status: str
    scope1_2_c: float
    scope3_c: float
    target_c: float
    improvement_trend_c: float
    generated_at: datetime


class ReviewCountdownResponse(BaseModel):
    """Review countdown data."""
    org_id: str
    next_review_date: str
    days_until_review: int
    review_status: str
    readiness_pct: float
    review_target_id: str
    generated_at: datetime


class MilestonesResponse(BaseModel):
    """Key milestones timeline."""
    org_id: str
    milestones: List[Dict[str, Any]]
    next_milestone: Dict[str, Any]
    completed_count: int
    total_count: int
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
    "/org/{org_id}/readiness",
    response_model=ReadinessScoreResponse,
    summary="Overall readiness score",
    description="Get the overall SBTi readiness score with category breakdowns.",
)
async def get_readiness_score(org_id: str) -> ReadinessScoreResponse:
    """Get overall readiness score."""
    return ReadinessScoreResponse(
        org_id=org_id,
        overall_readiness_pct=82.0,
        readiness_level="nearly_ready",
        category_scores={
            "emissions_inventory": 95.0,
            "target_definition": 90.0,
            "scope3_screening": 65.0,
            "pathway_calculation": 85.0,
            "reporting_disclosure": 70.0,
            "governance_approval": 80.0,
        },
        sbti_status="validated",
        validation_date="2023-06-15",
        next_review_date="2028-06-15",
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/target-summary",
    response_model=TargetSummaryDashResponse,
    summary="Target status cards",
    description="Get target status cards for dashboard display.",
)
async def get_target_summary(org_id: str) -> TargetSummaryDashResponse:
    """Get target status cards."""
    return TargetSummaryDashResponse(
        org_id=org_id,
        total_targets=4,
        active_targets=3,
        draft_targets=1,
        expired_targets=0,
        targets=[
            {
                "target_name": "Scope 1+2 Near-Term 1.5C",
                "target_type": "near_term", "scope": "scope_1_2",
                "ambition": "1.5C", "reduction_pct": 42.0,
                "base_year": 2020, "target_year": 2030,
                "status": "active", "on_track": True,
                "progress_pct": 37.6,
            },
            {
                "target_name": "Scope 3 Supplier Engagement",
                "target_type": "near_term", "scope": "scope_3",
                "ambition": "well_below_2C", "reduction_pct": 25.0,
                "base_year": 2020, "target_year": 2030,
                "status": "active", "on_track": True,
                "progress_pct": 32.8,
            },
            {
                "target_name": "FLAG Target (Land Use)",
                "target_type": "near_term", "scope": "scope_1_2",
                "ambition": "1.5C", "reduction_pct": 30.0,
                "base_year": 2020, "target_year": 2030,
                "status": "active", "on_track": False,
                "progress_pct": 17.0,
            },
            {
                "target_name": "Net-Zero All Scopes",
                "target_type": "net_zero", "scope": "all_scopes",
                "ambition": "1.5C", "reduction_pct": 90.0,
                "base_year": 2020, "target_year": 2050,
                "status": "draft", "on_track": None,
                "progress_pct": 0,
            },
        ],
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/pathway-overview",
    response_model=PathwayOverlayResponse,
    summary="Pathway with actuals overlay",
    description="Get pathway budget data overlaid with actual emissions for chart rendering.",
)
async def get_pathway_overlay(org_id: str) -> PathwayOverlayResponse:
    """Get pathway overlay data."""
    return PathwayOverlayResponse(
        org_id=org_id,
        pathway_years={
            "2020": 50000, "2021": 47900, "2022": 45800,
            "2023": 43700, "2024": 41600, "2025": 39500,
            "2026": 37400, "2027": 35300, "2028": 33200,
            "2029": 31100, "2030": 29000,
        },
        actual_years={
            "2020": 50000, "2021": 48500, "2022": 46800,
            "2023": 44200, "2024": 42100,
        },
        base_year=2020,
        target_year=2030,
        on_track=True,
        variance_current_pct=-1.2,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/temperature",
    response_model=TemperatureGaugeResponse,
    summary="Temperature gauge",
    description="Get temperature gauge data for dashboard visualization.",
)
async def get_temperature_gauge(org_id: str) -> TemperatureGaugeResponse:
    """Get temperature gauge data."""
    return TemperatureGaugeResponse(
        org_id=org_id,
        temperature_c=1.95,
        alignment_status="below_2C",
        scope1_2_c=1.65,
        scope3_c=2.15,
        target_c=1.5,
        improvement_trend_c=0.17,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/review-countdown",
    response_model=ReviewCountdownResponse,
    summary="Review countdown",
    description="Get countdown to next five-year review deadline.",
)
async def get_review_countdown(org_id: str) -> ReviewCountdownResponse:
    """Get review countdown."""
    return ReviewCountdownResponse(
        org_id=org_id,
        next_review_date="2028-06-15",
        days_until_review=834,
        review_status="scheduled",
        readiness_pct=45.0,
        review_target_id="tgt_main_001",
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/milestones",
    response_model=MilestonesResponse,
    summary="Key milestones",
    description="Get key SBTi milestones timeline for dashboard display.",
)
async def get_milestones(org_id: str) -> MilestonesResponse:
    """Get key milestones."""
    milestones = [
        {"milestone": "SBTi Commitment Letter Signed", "date": "2022-01-15", "status": "completed", "type": "commitment"},
        {"milestone": "Base Year Inventory Completed", "date": "2022-06-30", "status": "completed", "type": "data"},
        {"milestone": "Near-Term Target Validated", "date": "2023-06-15", "status": "completed", "type": "validation"},
        {"milestone": "First Annual Progress Report", "date": "2024-03-31", "status": "completed", "type": "reporting"},
        {"milestone": "25% Reduction Milestone", "date": "2025-12-31", "status": "upcoming", "type": "target"},
        {"milestone": "Scope 3 Supplier Engagement 50%", "date": "2026-12-31", "status": "upcoming", "type": "target"},
        {"milestone": "Five-Year Review", "date": "2028-06-15", "status": "upcoming", "type": "review"},
        {"milestone": "Near-Term Target Year", "date": "2030-12-31", "status": "upcoming", "type": "target"},
        {"milestone": "Net-Zero Target Year", "date": "2050-12-31", "status": "upcoming", "type": "target"},
    ]

    completed = sum(1 for m in milestones if m["status"] == "completed")
    upcoming = [m for m in milestones if m["status"] == "upcoming"]
    next_ms = upcoming[0] if upcoming else milestones[-1]

    return MilestonesResponse(
        org_id=org_id,
        milestones=milestones,
        next_milestone=next_ms,
        completed_count=completed,
        total_count=len(milestones),
        generated_at=_now(),
    )
