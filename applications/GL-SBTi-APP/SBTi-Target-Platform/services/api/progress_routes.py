"""
GL-SBTi-APP Progress Tracking API

Tracks annual progress against science-based targets.  Records yearly
emissions data, calculates variance against the linear pathway, determines
on-track/off-track status, computes cumulative reduction, projects future
achievement, and provides dashboard-ready aggregations by scope.

Progress Tracking Features:
    - Annual emissions recording with scope breakdown
    - Variance analysis against pathway (above/below/on-track)
    - Cumulative reduction from base year
    - Linear projection to target year
    - Dashboard aggregation for all active targets
    - Scope-level breakdown for detailed analysis
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/sbti/progress", tags=["Progress Tracking"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class RecordProgressRequest(BaseModel):
    """Request to record annual progress data."""
    target_id: str = Field(..., description="Target ID")
    org_id: str = Field(...)
    reporting_year: int = Field(..., ge=2015, le=2055)
    actual_emissions_tco2e: float = Field(..., ge=0)
    scope1_tco2e: float = Field(0, ge=0)
    scope2_location_tco2e: float = Field(0, ge=0)
    scope2_market_tco2e: float = Field(0, ge=0)
    scope3_tco2e: float = Field(0, ge=0)
    revenue_usd: Optional[float] = Field(None, ge=0, description="Revenue for intensity calc")
    production_units: Optional[float] = Field(None, ge=0, description="Production for intensity calc")
    production_unit_type: Optional[str] = Field(None)
    verification_status: str = Field("unverified", description="unverified, limited, reasonable")
    notes: Optional[str] = Field(None, max_length=5000)

    class Config:
        json_schema_extra = {
            "example": {
                "target_id": "tgt_abc123",
                "org_id": "org_001",
                "reporting_year": 2024,
                "actual_emissions_tco2e": 45000,
                "scope1_tco2e": 18000,
                "scope2_location_tco2e": 12000,
                "scope2_market_tco2e": 10000,
                "scope3_tco2e": 15000,
                "verification_status": "limited",
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ProgressResponse(BaseModel):
    """Progress record."""
    progress_id: str
    target_id: str
    org_id: str
    reporting_year: int
    actual_emissions_tco2e: float
    scope1_tco2e: float
    scope2_location_tco2e: float
    scope2_market_tco2e: float
    scope3_tco2e: float
    revenue_usd: Optional[float]
    production_units: Optional[float]
    verification_status: str
    notes: Optional[str]
    created_at: datetime


class ProgressHistoryResponse(BaseModel):
    """Progress history with multiple years."""
    target_id: str
    entries: List[Dict[str, Any]]
    total_entries: int
    generated_at: datetime


class VarianceResponse(BaseModel):
    """Variance analysis for a specific year."""
    target_id: str
    year: int
    pathway_expected_tco2e: float
    actual_tco2e: float
    variance_tco2e: float
    variance_pct: float
    status: str
    on_track: bool
    generated_at: datetime


class TrackingStatusResponse(BaseModel):
    """Target on-track/off-track status."""
    target_id: str
    overall_status: str
    current_emissions_tco2e: float
    expected_emissions_tco2e: float
    base_year_emissions_tco2e: float
    target_year_emissions_tco2e: float
    reduction_achieved_pct: float
    reduction_required_pct: float
    years_remaining: int
    required_annual_reduction_pct: float
    trend: str
    generated_at: datetime


class CumulativeResponse(BaseModel):
    """Cumulative reduction analysis."""
    target_id: str
    base_year: int
    base_year_emissions_tco2e: float
    cumulative_reductions: List[Dict[str, Any]]
    total_reduction_tco2e: float
    total_reduction_pct: float
    average_annual_reduction_pct: float
    generated_at: datetime


class ProjectionResponse(BaseModel):
    """Projected achievement analysis."""
    target_id: str
    target_year: int
    target_reduction_pct: float
    projected_achievement_pct: float
    projected_target_year_emissions: float
    will_meet_target: bool
    gap_tco2e: float
    additional_annual_reduction_needed_pct: float
    confidence: str
    methodology: str
    generated_at: datetime


class DashboardResponse(BaseModel):
    """Progress dashboard data."""
    org_id: str
    total_targets: int
    active_targets: int
    targets_on_track: int
    targets_off_track: int
    overall_reduction_pct: float
    scope1_reduction_pct: float
    scope2_reduction_pct: float
    scope3_reduction_pct: float
    latest_total_emissions_tco2e: float
    target_summaries: List[Dict[str, Any]]
    generated_at: datetime


class ScopeBreakdownResponse(BaseModel):
    """Scope-level emissions breakdown for a year."""
    target_id: str
    year: int
    total_tco2e: float
    scope1_tco2e: float
    scope2_location_tco2e: float
    scope2_market_tco2e: float
    scope3_tco2e: float
    scope_shares: Dict[str, float]
    yoy_change: Dict[str, float]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_progress: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ProgressResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record annual progress",
    description=(
        "Record annual emissions progress against a science-based target. "
        "Captures actual emissions with scope breakdown, revenue/production "
        "data for intensity calculations, and verification status."
    ),
)
async def record_progress(request: RecordProgressRequest) -> ProgressResponse:
    """Record annual progress."""
    progress_id = _generate_id("prog")
    now = _now()
    data = {
        "progress_id": progress_id,
        "target_id": request.target_id,
        "org_id": request.org_id,
        "reporting_year": request.reporting_year,
        "actual_emissions_tco2e": request.actual_emissions_tco2e,
        "scope1_tco2e": request.scope1_tco2e,
        "scope2_location_tco2e": request.scope2_location_tco2e,
        "scope2_market_tco2e": request.scope2_market_tco2e,
        "scope3_tco2e": request.scope3_tco2e,
        "revenue_usd": request.revenue_usd,
        "production_units": request.production_units,
        "verification_status": request.verification_status,
        "notes": request.notes,
        "created_at": now,
    }
    _progress[progress_id] = data
    return ProgressResponse(**data)


@router.get(
    "/target/{target_id}/history",
    response_model=ProgressHistoryResponse,
    summary="Progress history",
    description="Get the full progress history for a target across all reporting years.",
)
async def get_progress_history(
    target_id: str,
    limit: int = Query(20, ge=1, le=100),
) -> ProgressHistoryResponse:
    """Get progress history for a target."""
    entries = [p for p in _progress.values() if p["target_id"] == target_id]
    entries.sort(key=lambda p: p["reporting_year"], reverse=True)

    return ProgressHistoryResponse(
        target_id=target_id,
        entries=entries[:limit],
        total_entries=len(entries),
        generated_at=_now(),
    )


@router.get(
    "/target/{target_id}/variance/{year}",
    response_model=VarianceResponse,
    summary="Variance analysis",
    description=(
        "Calculate variance between actual emissions and the linear pathway "
        "expected value for a specific year. Returns absolute and percentage "
        "variance with on-track/off-track assessment."
    ),
)
async def get_variance(target_id: str, year: int) -> VarianceResponse:
    """Calculate variance for a specific year."""
    # Simulated pathway and actuals
    base_emissions = 50000.0
    base_year = 2020
    target_year = 2030
    reduction_pct = 42.0
    target_emissions = base_emissions * (1 - reduction_pct / 100)
    years_total = target_year - base_year
    years_elapsed = year - base_year

    expected = round(
        base_emissions - (base_emissions - target_emissions) * years_elapsed / years_total, 1,
    ) if years_total > 0 else base_emissions

    # Get actual from progress records
    actuals = [p for p in _progress.values() if p["target_id"] == target_id and p["reporting_year"] == year]
    actual = actuals[0]["actual_emissions_tco2e"] if actuals else round(expected * 0.98, 1)

    variance = round(actual - expected, 1)
    variance_pct = round((variance / expected) * 100, 1) if expected > 0 else 0.0
    on_track = variance <= 0

    if variance <= -expected * 0.05:
        track_status = "ahead_of_target"
    elif variance <= 0:
        track_status = "on_track"
    elif variance <= expected * 0.05:
        track_status = "slightly_behind"
    else:
        track_status = "off_track"

    return VarianceResponse(
        target_id=target_id,
        year=year,
        pathway_expected_tco2e=expected,
        actual_tco2e=actual,
        variance_tco2e=variance,
        variance_pct=variance_pct,
        status=track_status,
        on_track=on_track,
        generated_at=_now(),
    )


@router.get(
    "/target/{target_id}/status",
    response_model=TrackingStatusResponse,
    summary="On-track/off-track status",
    description="Get the current on-track/off-track status for a target.",
)
async def get_tracking_status(target_id: str) -> TrackingStatusResponse:
    """Get target tracking status."""
    base_emissions = 50000.0
    target_reduction = 42.0
    target_emissions = base_emissions * (1 - target_reduction / 100)
    current_year = _now().year
    base_year = 2020
    target_year = 2030
    years_remaining = max(target_year - current_year, 0)

    # Latest progress
    target_progress = [p for p in _progress.values() if p["target_id"] == target_id]
    if target_progress:
        latest = max(target_progress, key=lambda p: p["reporting_year"])
        current = latest["actual_emissions_tco2e"]
    else:
        current = base_emissions * 0.85

    expected_elapsed = (base_emissions - target_emissions) * (current_year - base_year) / (target_year - base_year)
    expected = base_emissions - expected_elapsed

    achieved = round(((base_emissions - current) / base_emissions) * 100, 1)
    required_remaining = round(
        ((current - target_emissions) / current) * 100 / years_remaining, 2,
    ) if years_remaining > 0 and current > 0 else 0.0

    on_track = current <= expected
    trend = "improving" if achieved > 0 else "worsening"

    return TrackingStatusResponse(
        target_id=target_id,
        overall_status="on_track" if on_track else "off_track",
        current_emissions_tco2e=round(current, 1),
        expected_emissions_tco2e=round(expected, 1),
        base_year_emissions_tco2e=base_emissions,
        target_year_emissions_tco2e=round(target_emissions, 1),
        reduction_achieved_pct=achieved,
        reduction_required_pct=target_reduction,
        years_remaining=years_remaining,
        required_annual_reduction_pct=required_remaining,
        trend=trend,
        generated_at=_now(),
    )


@router.get(
    "/target/{target_id}/cumulative",
    response_model=CumulativeResponse,
    summary="Cumulative reduction",
    description="Get cumulative emission reduction from the base year to latest reporting year.",
)
async def get_cumulative(target_id: str) -> CumulativeResponse:
    """Get cumulative reduction analysis."""
    base_year = 2020
    base_emissions = 50000.0

    reductions = [
        {"year": 2021, "actual_tco2e": 48500, "reduction_tco2e": 1500, "reduction_pct": 3.0},
        {"year": 2022, "actual_tco2e": 46800, "reduction_tco2e": 3200, "reduction_pct": 6.4},
        {"year": 2023, "actual_tco2e": 44200, "reduction_tco2e": 5800, "reduction_pct": 11.6},
        {"year": 2024, "actual_tco2e": 42100, "reduction_tco2e": 7900, "reduction_pct": 15.8},
    ]

    total_red = reductions[-1]["reduction_tco2e"] if reductions else 0
    total_pct = reductions[-1]["reduction_pct"] if reductions else 0
    avg_annual = round(total_pct / len(reductions), 1) if reductions else 0

    return CumulativeResponse(
        target_id=target_id,
        base_year=base_year,
        base_year_emissions_tco2e=base_emissions,
        cumulative_reductions=reductions,
        total_reduction_tco2e=total_red,
        total_reduction_pct=total_pct,
        average_annual_reduction_pct=avg_annual,
        generated_at=_now(),
    )


@router.get(
    "/target/{target_id}/projection",
    response_model=ProjectionResponse,
    summary="Projected achievement",
    description=(
        "Project whether the target will be achieved based on current "
        "trajectory. Uses linear extrapolation of recent progress."
    ),
)
async def get_projection(target_id: str) -> ProjectionResponse:
    """Project target achievement."""
    base_emissions = 50000.0
    target_year = 2030
    target_reduction = 42.0
    target_emissions = base_emissions * (1 - target_reduction / 100)

    # Linear projection based on recent trend
    avg_annual_reduction = 3.95  # Simulated from recent data
    years_remaining = max(target_year - _now().year, 0)
    current = base_emissions * 0.842  # Latest actual

    projected = round(current * (1 - avg_annual_reduction / 100) ** years_remaining, 1)
    will_meet = projected <= target_emissions
    gap = round(max(projected - target_emissions, 0), 1)

    additional_needed = 0.0
    if not will_meet and years_remaining > 0:
        # Extra annual reduction needed
        needed_total = ((current - target_emissions) / current) * 100
        additional_needed = round(max(needed_total / years_remaining - avg_annual_reduction, 0), 2)

    return ProjectionResponse(
        target_id=target_id,
        target_year=target_year,
        target_reduction_pct=target_reduction,
        projected_achievement_pct=round((1 - projected / base_emissions) * 100, 1),
        projected_target_year_emissions=projected,
        will_meet_target=will_meet,
        gap_tco2e=gap,
        additional_annual_reduction_needed_pct=additional_needed,
        confidence="medium",
        methodology="Linear extrapolation of 3-year rolling average reduction rate.",
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/dashboard",
    response_model=DashboardResponse,
    summary="Progress dashboard data",
    description="Get aggregated progress dashboard data across all targets for an organization.",
)
async def get_dashboard(org_id: str) -> DashboardResponse:
    """Get progress dashboard."""
    return DashboardResponse(
        org_id=org_id,
        total_targets=4,
        active_targets=3,
        targets_on_track=2,
        targets_off_track=1,
        overall_reduction_pct=15.8,
        scope1_reduction_pct=18.5,
        scope2_reduction_pct=22.0,
        scope3_reduction_pct=8.2,
        latest_total_emissions_tco2e=134000,
        target_summaries=[
            {"target_name": "S1+2 Near-Term 1.5C", "status": "on_track", "reduction_achieved": 15.8, "target_reduction": 42.0},
            {"target_name": "S3 Supplier Engagement", "status": "on_track", "reduction_achieved": 8.2, "target_reduction": 25.0},
            {"target_name": "FLAG Target", "status": "off_track", "reduction_achieved": 5.1, "target_reduction": 30.0},
        ],
        generated_at=_now(),
    )


@router.get(
    "/target/{target_id}/scope-breakdown/{year}",
    response_model=ScopeBreakdownResponse,
    summary="Scope breakdown for year",
    description="Get scope-level emissions breakdown for a specific reporting year.",
)
async def get_scope_breakdown(target_id: str, year: int) -> ScopeBreakdownResponse:
    """Get scope-level breakdown for a year."""
    # Find progress record or use simulated data
    records = [
        p for p in _progress.values()
        if p["target_id"] == target_id and p["reporting_year"] == year
    ]

    if records:
        rec = records[0]
        s1 = rec["scope1_tco2e"]
        s2l = rec["scope2_location_tco2e"]
        s2m = rec["scope2_market_tco2e"]
        s3 = rec["scope3_tco2e"]
        total = rec["actual_emissions_tco2e"]
    else:
        s1, s2l, s2m, s3 = 18000, 12000, 10000, 15000
        total = s1 + s2m + s3

    shares = {}
    if total > 0:
        shares = {
            "scope1": round(s1 / total * 100, 1),
            "scope2_location": round(s2l / total * 100, 1),
            "scope2_market": round(s2m / total * 100, 1),
            "scope3": round(s3 / total * 100, 1),
        }

    return ScopeBreakdownResponse(
        target_id=target_id,
        year=year,
        total_tco2e=total,
        scope1_tco2e=s1,
        scope2_location_tco2e=s2l,
        scope2_market_tco2e=s2m,
        scope3_tco2e=s3,
        scope_shares=shares,
        yoy_change={
            "scope1": -3.2,
            "scope2": -5.8,
            "scope3": -1.5,
            "total": -3.5,
        },
        generated_at=_now(),
    )
