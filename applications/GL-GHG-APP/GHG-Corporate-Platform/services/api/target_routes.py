"""
GL-GHG-APP Targets API

Manages GHG emission reduction targets, SBTi alignment checks,
gap-to-target analysis, and progress tracking.

Target types:
    - Absolute: reduce total emissions by X% from base year
    - Intensity: reduce emissions per unit of output by X%

SBTi alignment:
    - Well-below 2 degrees C: >= 2.5% annual linear reduction
    - 1.5 degrees C: >= 4.2% annual linear reduction
    - Net-zero by 2050: long-term targets with near-term milestones
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/targets", tags=["Targets"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetType(str, Enum):
    """Types of reduction targets."""
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"


class TargetScope(str, Enum):
    """Which scopes the target covers."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_3 = "scope_3"
    ALL_SCOPES = "all_scopes"


class SBTiPathway(str, Enum):
    """SBTi temperature alignment pathways."""
    WELL_BELOW_2C = "well_below_2c"
    ONE_POINT_FIVE_C = "1.5c"
    NET_ZERO_2050 = "net_zero_2050"


class TargetStatus(str, Enum):
    """Target tracking status."""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    ACHIEVED = "achieved"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateTargetRequest(BaseModel):
    """Request to create a GHG reduction target."""
    org_id: str = Field(..., description="Organization ID")
    type: TargetType = Field(..., description="Target type: absolute or intensity")
    scope: TargetScope = Field(..., description="Scope coverage of the target")
    base_year: int = Field(..., ge=1990, le=2100, description="Base year for reduction calculation")
    target_year: int = Field(..., ge=2025, le=2100, description="Target year")
    reduction_pct: float = Field(..., gt=0, le=100, description="Reduction percentage from base year")
    sbti_aligned: bool = Field(False, description="Whether target is SBTi-aligned")
    sbti_pathway: Optional[SBTiPathway] = Field(None, description="SBTi pathway if aligned")
    intensity_metric: Optional[str] = Field(
        None, description="Intensity denominator (e.g. revenue_usd, employee, sqft)"
    )
    interim_milestones: Optional[List[Dict[str, Any]]] = Field(
        None, description="Interim milestones: [{year, reduction_pct}]"
    )
    description: Optional[str] = Field(None, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "type": "absolute",
                "scope": "scope_1_2",
                "base_year": 2019,
                "target_year": 2030,
                "reduction_pct": 42.0,
                "sbti_aligned": True,
                "sbti_pathway": "1.5c",
                "interim_milestones": [
                    {"year": 2025, "reduction_pct": 20.0},
                    {"year": 2027, "reduction_pct": 30.0}
                ],
                "description": "Near-term SBTi 1.5C aligned target for Scope 1+2"
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TargetResponse(BaseModel):
    """A GHG reduction target."""
    target_id: str
    org_id: str
    type: str
    scope: str
    base_year: int
    target_year: int
    reduction_pct: float
    base_year_emissions_tco2e: float
    target_emissions_tco2e: float
    current_emissions_tco2e: float
    current_reduction_pct: float
    status: str
    sbti_aligned: bool
    sbti_pathway: Optional[str]
    intensity_metric: Optional[str]
    interim_milestones: Optional[List[Dict[str, Any]]]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class TargetProgressResponse(BaseModel):
    """Target progress with linear forecast."""
    target_id: str
    target_type: str
    scope: str
    base_year: int
    target_year: int
    reduction_target_pct: float
    base_year_emissions: float
    target_emissions: float
    current_year: int
    current_emissions: float
    current_reduction_pct: float
    required_annual_reduction_pct: float
    actual_annual_reduction_pct: float
    status: str
    on_track: bool
    years_remaining: int
    remaining_reduction_needed_tco2e: float
    linear_forecast: List[Dict[str, Any]]
    historical: List[Dict[str, Any]]


class SBTiAlignmentResponse(BaseModel):
    """SBTi alignment assessment."""
    target_id: str
    is_sbti_eligible: bool
    pathway: Optional[str]
    required_annual_reduction_pct: float
    actual_annual_reduction_pct: float
    alignment_status: str
    alignment_gap_pct: float
    criteria_checks: List[Dict[str, Any]]
    recommendations: List[str]


class GapAnalysisResponse(BaseModel):
    """Gap-to-target analysis."""
    target_id: str
    target_year: int
    target_emissions_tco2e: float
    current_emissions_tco2e: float
    gap_tco2e: float
    gap_pct: float
    years_remaining: int
    annual_reduction_needed_tco2e: float
    reduction_levers: List[Dict[str, Any]]
    scenario_projections: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_targets: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=TargetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Set reduction target",
    description=(
        "Create a GHG emission reduction target. Supports absolute and "
        "intensity targets, with optional SBTi alignment and interim milestones."
    ),
)
async def create_target(request: CreateTargetRequest) -> TargetResponse:
    if request.target_year <= request.base_year:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target year must be after base year.",
        )
    if request.type == TargetType.INTENSITY and not request.intensity_metric:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Intensity metric is required for intensity targets.",
        )
    if request.sbti_aligned and not request.sbti_pathway:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SBTi pathway is required when sbti_aligned is true.",
        )

    target_id = _generate_id("tgt")
    now = _now()

    # Simulated base year emissions based on scope
    base_emissions_map = {
        "scope_1": 14200.0,
        "scope_2": 9800.0,
        "scope_1_2": 24000.0,
        "scope_3": 48000.0,
        "all_scopes": 72000.0,
    }
    current_emissions_map = {
        "scope_1": 12450.8,
        "scope_2": 8320.5,
        "scope_1_2": 20771.3,
        "scope_3": 45230.2,
        "all_scopes": 66001.5,
    }

    base_emissions = base_emissions_map.get(request.scope.value, 72000.0)
    current_emissions = current_emissions_map.get(request.scope.value, 66001.5)
    target_emissions = round(base_emissions * (1 - request.reduction_pct / 100.0), 2)
    current_reduction = round((1 - current_emissions / base_emissions) * 100, 2)

    if current_reduction >= request.reduction_pct:
        target_status = TargetStatus.ACHIEVED.value
    elif current_reduction >= request.reduction_pct * 0.7:
        target_status = TargetStatus.ON_TRACK.value
    elif current_reduction >= request.reduction_pct * 0.4:
        target_status = TargetStatus.AT_RISK.value
    else:
        target_status = TargetStatus.OFF_TRACK.value

    target = {
        "target_id": target_id,
        "org_id": request.org_id,
        "type": request.type.value,
        "scope": request.scope.value,
        "base_year": request.base_year,
        "target_year": request.target_year,
        "reduction_pct": request.reduction_pct,
        "base_year_emissions_tco2e": base_emissions,
        "target_emissions_tco2e": target_emissions,
        "current_emissions_tco2e": current_emissions,
        "current_reduction_pct": current_reduction,
        "status": target_status,
        "sbti_aligned": request.sbti_aligned,
        "sbti_pathway": request.sbti_pathway.value if request.sbti_pathway else None,
        "intensity_metric": request.intensity_metric,
        "interim_milestones": request.interim_milestones,
        "description": request.description,
        "created_at": now,
        "updated_at": now,
    }
    _targets[target_id] = target
    return TargetResponse(**target)


@router.get(
    "/{org_id}",
    response_model=List[TargetResponse],
    summary="List targets for organization",
    description="Retrieve all GHG reduction targets for an organization.",
)
async def list_targets(
    org_id: str,
    scope: Optional[str] = Query(None, description="Filter by scope"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
) -> List[TargetResponse]:
    targets = [t for t in _targets.values() if t["org_id"] == org_id]
    if scope:
        targets = [t for t in targets if t["scope"] == scope]
    if status_filter:
        targets = [t for t in targets if t["status"] == status_filter]
    targets.sort(key=lambda t: t["created_at"], reverse=True)

    # If no targets exist, return demo targets
    if not targets:
        now = _now()
        targets = [
            {
                "target_id": _generate_id("tgt"),
                "org_id": org_id,
                "type": "absolute",
                "scope": "scope_1_2",
                "base_year": 2019,
                "target_year": 2030,
                "reduction_pct": 42.0,
                "base_year_emissions_tco2e": 24000.0,
                "target_emissions_tco2e": 13920.0,
                "current_emissions_tco2e": 20771.3,
                "current_reduction_pct": 13.5,
                "status": "on_track",
                "sbti_aligned": True,
                "sbti_pathway": "1.5c",
                "intensity_metric": None,
                "interim_milestones": [
                    {"year": 2025, "reduction_pct": 20.0},
                    {"year": 2027, "reduction_pct": 30.0},
                ],
                "description": "Near-term SBTi 1.5C target for Scope 1+2",
                "created_at": now,
                "updated_at": now,
            },
            {
                "target_id": _generate_id("tgt"),
                "org_id": org_id,
                "type": "absolute",
                "scope": "scope_3",
                "base_year": 2019,
                "target_year": 2030,
                "reduction_pct": 25.0,
                "base_year_emissions_tco2e": 48000.0,
                "target_emissions_tco2e": 36000.0,
                "current_emissions_tco2e": 45230.2,
                "current_reduction_pct": 5.8,
                "status": "at_risk",
                "sbti_aligned": True,
                "sbti_pathway": "well_below_2c",
                "intensity_metric": None,
                "interim_milestones": None,
                "description": "SBTi Scope 3 target (WB2C pathway)",
                "created_at": now,
                "updated_at": now,
            },
        ]

    return [TargetResponse(**t) for t in targets]


@router.get(
    "/{target_id}/progress",
    response_model=TargetProgressResponse,
    summary="Target progress with forecast",
    description=(
        "Detailed progress tracking with historical emissions, linear "
        "forecast, required vs. actual annual reduction rates, and on-track status."
    ),
)
async def get_target_progress(target_id: str) -> TargetProgressResponse:
    target = _targets.get(target_id)
    base_year = target["base_year"] if target else 2019
    target_year = target["target_year"] if target else 2030
    reduction_pct = target["reduction_pct"] if target else 42.0
    scope = target["scope"] if target else "scope_1_2"
    target_type = target["type"] if target else "absolute"
    base_emissions = target["base_year_emissions_tco2e"] if target else 24000.0

    target_emissions = round(base_emissions * (1 - reduction_pct / 100.0), 2)
    current_year = 2025
    current_emissions = 20771.3
    current_reduction = round((1 - current_emissions / base_emissions) * 100, 2)
    years_total = target_year - base_year
    years_elapsed = current_year - base_year
    years_remaining = target_year - current_year
    required_annual = round(reduction_pct / years_total, 2)
    actual_annual = round(current_reduction / max(years_elapsed, 1), 2)
    on_track = actual_annual >= required_annual * 0.85
    remaining_reduction = round(current_emissions - target_emissions, 2)

    historical = [
        {"year": 2019, "emissions_tco2e": 24000.0, "reduction_pct": 0.0},
        {"year": 2020, "emissions_tco2e": 22800.0, "reduction_pct": 5.0},
        {"year": 2021, "emissions_tco2e": 22200.0, "reduction_pct": 7.5},
        {"year": 2022, "emissions_tco2e": 21800.0, "reduction_pct": 9.2},
        {"year": 2023, "emissions_tco2e": 21400.0, "reduction_pct": 10.8},
        {"year": 2024, "emissions_tco2e": 21050.0, "reduction_pct": 12.3},
        {"year": 2025, "emissions_tco2e": 20771.3, "reduction_pct": 13.5},
    ]

    # Linear forecast from current to target
    annual_decrease = remaining_reduction / max(years_remaining, 1)
    forecast = []
    for yr_offset in range(1, years_remaining + 1):
        yr = current_year + yr_offset
        projected = round(current_emissions - annual_decrease * yr_offset, 2)
        forecast.append({
            "year": yr,
            "projected_tco2e": max(projected, target_emissions),
            "target_pathway_tco2e": round(
                base_emissions * (1 - (reduction_pct / years_total * (years_elapsed + yr_offset)) / 100), 2
            ),
        })

    return TargetProgressResponse(
        target_id=target_id,
        target_type=target_type,
        scope=scope,
        base_year=base_year,
        target_year=target_year,
        reduction_target_pct=reduction_pct,
        base_year_emissions=base_emissions,
        target_emissions=target_emissions,
        current_year=current_year,
        current_emissions=current_emissions,
        current_reduction_pct=current_reduction,
        required_annual_reduction_pct=required_annual,
        actual_annual_reduction_pct=actual_annual,
        status="on_track" if on_track else "at_risk",
        on_track=on_track,
        years_remaining=years_remaining,
        remaining_reduction_needed_tco2e=remaining_reduction,
        linear_forecast=forecast,
        historical=historical,
    )


@router.get(
    "/{target_id}/sbti",
    response_model=SBTiAlignmentResponse,
    summary="SBTi alignment check",
    description=(
        "Assess whether the target meets Science Based Targets initiative "
        "criteria for the specified pathway (1.5C, WB2C, or Net Zero)."
    ),
)
async def check_sbti_alignment(target_id: str) -> SBTiAlignmentResponse:
    target = _targets.get(target_id)
    pathway = target["sbti_pathway"] if target else "1.5c"

    required_annual = {"1.5c": 4.2, "well_below_2c": 2.5, "net_zero_2050": 4.2}
    required = required_annual.get(pathway, 4.2)
    actual = 2.25  # simulated

    is_aligned = actual >= required
    gap = round(required - actual, 2)

    criteria = [
        {"criterion": "Scope coverage", "description": "Target covers Scope 1+2", "met": True},
        {"criterion": "Timeframe", "description": "Near-term target: 5-10 years", "met": True},
        {"criterion": "Ambition level", "description": f"Required >= {required}%/yr, actual {actual}%/yr", "met": is_aligned},
        {"criterion": "Base year", "description": "Base year within last 5 years of submission", "met": True},
        {"criterion": "Scope 3 target", "description": "Scope 3 target set if >40% of total", "met": True},
        {"criterion": "Method", "description": "Absolute contraction approach used", "met": True},
    ]

    recommendations = []
    if not is_aligned:
        recommendations.append(
            f"Increase annual reduction rate from {actual}% to >= {required}% per year"
        )
        recommendations.append("Consider additional energy efficiency investments")
        recommendations.append("Accelerate renewable energy procurement strategy")
    else:
        recommendations.append("Target meets SBTi criteria. Submit for official validation.")
        recommendations.append("Consider upgrading to 1.5C pathway if currently on WB2C.")

    return SBTiAlignmentResponse(
        target_id=target_id,
        is_sbti_eligible=True,
        pathway=pathway,
        required_annual_reduction_pct=required,
        actual_annual_reduction_pct=actual,
        alignment_status="aligned" if is_aligned else "not_aligned",
        alignment_gap_pct=max(0, gap),
        criteria_checks=criteria,
        recommendations=recommendations,
    )


@router.get(
    "/{target_id}/gap",
    response_model=GapAnalysisResponse,
    summary="Gap-to-target analysis",
    description=(
        "Analyze the remaining gap to reach the target, including required "
        "annual reductions and potential reduction levers with estimated "
        "abatement potential."
    ),
)
async def get_gap_analysis(target_id: str) -> GapAnalysisResponse:
    target = _targets.get(target_id)
    target_year = target["target_year"] if target else 2030
    target_emissions = target["target_emissions_tco2e"] if target else 13920.0
    current_emissions = target["current_emissions_tco2e"] if target else 20771.3

    gap = round(current_emissions - target_emissions, 2)
    gap_pct = round(gap / current_emissions * 100, 2)
    years_remaining = target_year - 2025
    annual_needed = round(gap / max(years_remaining, 1), 2)

    levers = [
        {
            "lever": "Energy efficiency upgrades",
            "scope": "scope_1",
            "estimated_reduction_tco2e": 1200.0,
            "cost_usd": 850000,
            "marginal_abatement_cost_usd_per_tco2e": 708.0,
            "implementation_timeline_years": 2,
            "confidence": "high",
        },
        {
            "lever": "Fleet electrification (50% of fleet)",
            "scope": "scope_1",
            "estimated_reduction_tco2e": 1170.0,
            "cost_usd": 2400000,
            "marginal_abatement_cost_usd_per_tco2e": 2051.0,
            "implementation_timeline_years": 3,
            "confidence": "medium",
        },
        {
            "lever": "Additional REC procurement (5,000 MWh)",
            "scope": "scope_2",
            "estimated_reduction_tco2e": 2475.0,
            "cost_usd": 125000,
            "marginal_abatement_cost_usd_per_tco2e": 50.5,
            "implementation_timeline_years": 1,
            "confidence": "high",
        },
        {
            "lever": "On-site solar installation (2 MW)",
            "scope": "scope_2",
            "estimated_reduction_tco2e": 1650.0,
            "cost_usd": 3200000,
            "marginal_abatement_cost_usd_per_tco2e": 1939.0,
            "implementation_timeline_years": 2,
            "confidence": "high",
        },
        {
            "lever": "Supplier engagement (top 20 suppliers)",
            "scope": "scope_3",
            "estimated_reduction_tco2e": 2500.0,
            "cost_usd": 300000,
            "marginal_abatement_cost_usd_per_tco2e": 120.0,
            "implementation_timeline_years": 3,
            "confidence": "low",
        },
    ]

    scenarios = [
        {
            "scenario": "Business as usual",
            "annual_reduction_pct": 1.5,
            "emissions_at_target_year": round(current_emissions * (1 - 0.015) ** years_remaining, 2),
            "meets_target": False,
        },
        {
            "scenario": "Moderate action",
            "annual_reduction_pct": 3.5,
            "emissions_at_target_year": round(current_emissions * (1 - 0.035) ** years_remaining, 2),
            "meets_target": False,
        },
        {
            "scenario": "Aggressive action",
            "annual_reduction_pct": 5.5,
            "emissions_at_target_year": round(current_emissions * (1 - 0.055) ** years_remaining, 2),
            "meets_target": True,
        },
    ]

    return GapAnalysisResponse(
        target_id=target_id,
        target_year=target_year,
        target_emissions_tco2e=target_emissions,
        current_emissions_tco2e=current_emissions,
        gap_tco2e=gap,
        gap_pct=gap_pct,
        years_remaining=years_remaining,
        annual_reduction_needed_tco2e=annual_needed,
        reduction_levers=levers,
        scenario_projections=scenarios,
    )


@router.delete(
    "/{target_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete target",
    description="Delete a GHG reduction target. This action is irreversible.",
)
async def delete_target(target_id: str):
    if target_id in _targets:
        del _targets[target_id]
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target {target_id} not found",
        )
    return None
