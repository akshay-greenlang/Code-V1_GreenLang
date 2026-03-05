"""
GL-SBTi-APP Target Management API

Manages science-based emissions reduction targets across all target types
supported by the SBTi framework: near-term (5-10 year), long-term (by 2050),
and net-zero commitments.  Provides full CRUD operations, target status
lifecycle management, submission form generation, Scope 3 coverage
requirement checks, and target-level coverage validation.

Target Types:
    - Near-term: 5-10 year absolute or intensity reduction targets
    - Long-term: Targets aligned with reaching net-zero by 2050 or sooner
    - Net-zero: Commitment to achieve net-zero across all material scopes

SBTi v2.1 Criteria Referenced:
    - C1-C5: Target boundary and timeframe
    - C6-C8: Level of ambition (1.5C, well-below 2C)
    - C13-C15: Scope 3 requirements (40% threshold)
    - C20-C23: Net-zero target requirements
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/sbti/targets", tags=["Targets"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetType(str, Enum):
    """SBTi target type classification."""
    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"
    NET_ZERO = "net_zero"


class TargetScope(str, Enum):
    """Emission scope covered by the target."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_3 = "scope_3"
    ALL_SCOPES = "all_scopes"


class TargetMethod(str, Enum):
    """Target-setting methodology."""
    ABSOLUTE = "absolute"
    INTENSITY_PHYSICAL = "intensity_physical"
    INTENSITY_ECONOMIC = "intensity_economic"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"


class AmbitionLevel(str, Enum):
    """Level of climate ambition."""
    C_1_5 = "1.5C"
    WELL_BELOW_2C = "well_below_2C"


class TargetStatus(str, Enum):
    """Target lifecycle status."""
    DRAFT = "draft"
    PENDING_VALIDATION = "pending_validation"
    SUBMITTED = "submitted"
    VALIDATED = "validated"
    APPROVED = "approved"
    ACTIVE = "active"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateTargetRequest(BaseModel):
    """Request to create a science-based target."""
    target_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Descriptive name for the target",
    )
    target_type: TargetType = Field(..., description="near_term, long_term, or net_zero")
    scope: TargetScope = Field(..., description="Emission scope(s) covered")
    method: TargetMethod = Field(..., description="Target-setting methodology")
    ambition_level: AmbitionLevel = Field(..., description="Temperature alignment")
    base_year: int = Field(..., ge=2015, le=2025, description="Emissions base year")
    base_year_emissions_tco2e: float = Field(
        ..., gt=0, description="Base year emissions (tCO2e)",
    )
    target_year: int = Field(..., ge=2025, le=2055, description="Target achievement year")
    reduction_pct: float = Field(
        ..., gt=0, le=100, description="Targeted percentage reduction from base year",
    )
    intensity_metric: Optional[str] = Field(
        None, max_length=200,
        description="Intensity denominator (e.g. per USD revenue, per tonne product)",
    )
    boundary_coverage_pct: float = Field(
        95.0, ge=0, le=100,
        description="Percentage of boundary emissions covered by target",
    )
    scope3_categories_included: Optional[List[int]] = Field(
        None,
        description="List of Scope 3 category numbers (1-15) included in target",
    )
    pathway_id: Optional[str] = Field(
        None, description="Associated decarbonization pathway ID",
    )
    notes: Optional[str] = Field(None, max_length=5000)

    class Config:
        json_schema_extra = {
            "example": {
                "target_name": "Scope 1+2 Near-Term 1.5C Aligned",
                "target_type": "near_term",
                "scope": "scope_1_2",
                "method": "absolute",
                "ambition_level": "1.5C",
                "base_year": 2020,
                "base_year_emissions_tco2e": 50000,
                "target_year": 2030,
                "reduction_pct": 42.0,
                "boundary_coverage_pct": 95.0,
            }
        }


class UpdateTargetRequest(BaseModel):
    """Request to update a science-based target."""
    target_name: Optional[str] = Field(None, max_length=300)
    reduction_pct: Optional[float] = Field(None, gt=0, le=100)
    target_year: Optional[int] = Field(None, ge=2025, le=2055)
    intensity_metric: Optional[str] = Field(None, max_length=200)
    boundary_coverage_pct: Optional[float] = Field(None, ge=0, le=100)
    scope3_categories_included: Optional[List[int]] = None
    pathway_id: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=5000)


class UpdateTargetStatusRequest(BaseModel):
    """Request to update target lifecycle status."""
    new_status: TargetStatus = Field(..., description="New target status")
    reason: Optional[str] = Field(None, max_length=2000, description="Reason for status change")


class CoverageCheckRequest(BaseModel):
    """Request to validate target coverage meets SBTi minimum thresholds."""
    scope1_covered_pct: float = Field(..., ge=0, le=100)
    scope2_covered_pct: float = Field(..., ge=0, le=100)
    scope3_covered_pct: float = Field(0, ge=0, le=100)
    total_scope3_pct_of_total: float = Field(
        0, ge=0, le=100,
        description="Scope 3 as percentage of total Scope 1+2+3 emissions",
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TargetResponse(BaseModel):
    """Science-based target record."""
    target_id: str
    org_id: str
    target_name: str
    target_type: str
    scope: str
    method: str
    ambition_level: str
    base_year: int
    base_year_emissions_tco2e: float
    target_year: int
    reduction_pct: float
    intensity_metric: Optional[str]
    boundary_coverage_pct: float
    scope3_categories_included: Optional[List[int]]
    pathway_id: Optional[str]
    status: str
    linear_annual_reduction_pct: float
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class TargetSummaryResponse(BaseModel):
    """Summary view of a target with key metrics."""
    target_id: str
    org_id: str
    target_name: str
    target_type: str
    scope: str
    ambition_level: str
    base_year: int
    target_year: int
    reduction_pct: float
    status: str
    years_remaining: int
    linear_annual_reduction_pct: float
    required_annual_tco2e: float
    on_track: Optional[bool]
    generated_at: datetime


class SubmissionFormResponse(BaseModel):
    """Pre-populated SBTi submission form data."""
    target_id: str
    org_id: str
    form_version: str
    target_details: Dict[str, Any]
    boundary_details: Dict[str, Any]
    methodology_details: Dict[str, Any]
    ambition_assessment: Dict[str, Any]
    supporting_data: Dict[str, Any]
    completeness_pct: float
    missing_fields: List[str]
    generated_at: datetime


class Scope3RequirementResponse(BaseModel):
    """Scope 3 target requirement assessment."""
    org_id: str
    scope3_pct_of_total: float
    threshold_pct: float
    scope3_target_required: bool
    minimum_coverage_pct: float
    recommendation: str
    generated_at: datetime


class CoverageCheckResponse(BaseModel):
    """Target coverage validation result."""
    org_id: str
    scope1_covered_pct: float
    scope2_covered_pct: float
    scope3_covered_pct: float
    scope1_2_meets_minimum: bool
    scope3_meets_minimum: bool
    scope3_target_required: bool
    overall_valid: bool
    issues: List[str]
    recommendations: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_targets: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _linear_annual_reduction(reduction_pct: float, base_year: int, target_year: int) -> float:
    """Calculate linear annual reduction percentage."""
    years = target_year - base_year
    if years <= 0:
        return 0.0
    return round(reduction_pct / years, 2)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=List[TargetResponse],
    summary="List targets for organization",
    description=(
        "Retrieve all science-based targets for an organization. Supports "
        "filtering by target type, scope, status, and ambition level."
    ),
)
async def list_targets(
    org_id: str = Query(..., description="Organization ID"),
    target_type: Optional[str] = Query(None, description="Filter by target type"),
    scope: Optional[str] = Query(None, description="Filter by scope"),
    target_status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> List[TargetResponse]:
    """List all targets for an organization."""
    results = [t for t in _targets.values() if t["org_id"] == org_id]
    if target_type:
        results = [t for t in results if t["target_type"] == target_type]
    if scope:
        results = [t for t in results if t["scope"] == scope]
    if target_status:
        results = [t for t in results if t["status"] == target_status]
    results.sort(key=lambda t: t["created_at"], reverse=True)
    return [TargetResponse(**t) for t in results[:limit]]


@router.get(
    "/{target_id}",
    response_model=TargetResponse,
    summary="Get target details",
    description="Retrieve a single science-based target by its ID.",
)
async def get_target(target_id: str) -> TargetResponse:
    """Get target by ID."""
    target = _targets.get(target_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target {target_id} not found",
        )
    return TargetResponse(**target)


@router.post(
    "/",
    response_model=TargetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create science-based target",
    description=(
        "Create a new science-based target for an organization. Validates "
        "timeframe constraints (5-10 years for near-term, by 2050 for "
        "long-term) and calculates linear annual reduction rate."
    ),
)
async def create_target(
    org_id: str = Query(..., description="Organization ID"),
    request: CreateTargetRequest = ...,
) -> TargetResponse:
    """Create a science-based target."""
    # Validate timeframe for near-term targets (5-10 years)
    timeframe = request.target_year - request.base_year
    if request.target_type == TargetType.NEAR_TERM:
        if timeframe < 5 or timeframe > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Near-term targets must have 5-10 year timeframe, got {timeframe} years",
            )
    # Validate long-term/net-zero targets are by 2050
    if request.target_type in (TargetType.LONG_TERM, TargetType.NET_ZERO):
        if request.target_year > 2050:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Long-term and net-zero targets must be by 2050 or sooner",
            )

    target_id = _generate_id("tgt")
    now = _now()
    annual_rate = _linear_annual_reduction(
        request.reduction_pct, request.base_year, request.target_year,
    )

    data = {
        "target_id": target_id,
        "org_id": org_id,
        "target_name": request.target_name,
        "target_type": request.target_type.value,
        "scope": request.scope.value,
        "method": request.method.value,
        "ambition_level": request.ambition_level.value,
        "base_year": request.base_year,
        "base_year_emissions_tco2e": request.base_year_emissions_tco2e,
        "target_year": request.target_year,
        "reduction_pct": request.reduction_pct,
        "intensity_metric": request.intensity_metric,
        "boundary_coverage_pct": request.boundary_coverage_pct,
        "scope3_categories_included": request.scope3_categories_included,
        "pathway_id": request.pathway_id,
        "status": TargetStatus.DRAFT.value,
        "linear_annual_reduction_pct": annual_rate,
        "notes": request.notes,
        "created_at": now,
        "updated_at": now,
    }
    _targets[target_id] = data
    return TargetResponse(**data)


@router.put(
    "/{target_id}",
    response_model=TargetResponse,
    summary="Update target",
    description="Update an existing science-based target and recalculate derived metrics.",
)
async def update_target(
    target_id: str,
    request: UpdateTargetRequest,
) -> TargetResponse:
    """Update an existing target."""
    target = _targets.get(target_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target {target_id} not found",
        )
    if target["status"] in ("validated", "approved", "active"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update target in '{target['status']}' status. Withdraw or create new target.",
        )

    updates = request.model_dump(exclude_unset=True)
    target.update(updates)

    # Recalculate linear annual reduction if relevant fields changed
    if "reduction_pct" in updates or "target_year" in updates:
        target["linear_annual_reduction_pct"] = _linear_annual_reduction(
            target["reduction_pct"], target["base_year"], target["target_year"],
        )

    target["updated_at"] = _now()
    return TargetResponse(**target)


@router.delete(
    "/{target_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete target",
    description="Delete a science-based target. Only draft targets may be deleted.",
)
async def delete_target(target_id: str) -> None:
    """Delete a target (draft only)."""
    target = _targets.get(target_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target {target_id} not found",
        )
    if target["status"] != "draft":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only draft targets may be deleted. Withdraw active targets instead.",
        )
    del _targets[target_id]
    return None


@router.put(
    "/{target_id}/status",
    response_model=TargetResponse,
    summary="Update target status",
    description=(
        "Transition a target through its lifecycle: draft -> pending_validation "
        "-> submitted -> validated -> approved -> active. Also supports "
        "expired and withdrawn terminal states."
    ),
)
async def update_target_status(
    target_id: str,
    request: UpdateTargetStatusRequest,
) -> TargetResponse:
    """Update target lifecycle status."""
    target = _targets.get(target_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target {target_id} not found",
        )

    valid_transitions = {
        "draft": ["pending_validation", "withdrawn"],
        "pending_validation": ["submitted", "draft", "withdrawn"],
        "submitted": ["validated", "pending_validation", "withdrawn"],
        "validated": ["approved", "withdrawn"],
        "approved": ["active", "withdrawn"],
        "active": ["expired", "withdrawn"],
        "expired": [],
        "withdrawn": [],
    }

    current = target["status"]
    new = request.new_status.value
    if new not in valid_transitions.get(current, []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transition from '{current}' to '{new}'. Valid: {valid_transitions.get(current, [])}",
        )

    target["status"] = new
    target["updated_at"] = _now()
    return TargetResponse(**target)


@router.get(
    "/{target_id}/summary",
    response_model=TargetSummaryResponse,
    summary="Target summary",
    description=(
        "Get a summary view of a target with calculated metrics including "
        "years remaining, linear annual reduction rate, and required annual "
        "absolute reduction in tCO2e."
    ),
)
async def get_target_summary(target_id: str) -> TargetSummaryResponse:
    """Get target summary with derived metrics."""
    target = _targets.get(target_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target {target_id} not found",
        )

    now = _now()
    years_remaining = max(target["target_year"] - now.year, 0)
    total_reduction = target["base_year_emissions_tco2e"] * (target["reduction_pct"] / 100.0)
    timeframe = target["target_year"] - target["base_year"]
    annual_tco2e = round(total_reduction / timeframe, 1) if timeframe > 0 else 0.0

    return TargetSummaryResponse(
        target_id=target["target_id"],
        org_id=target["org_id"],
        target_name=target["target_name"],
        target_type=target["target_type"],
        scope=target["scope"],
        ambition_level=target["ambition_level"],
        base_year=target["base_year"],
        target_year=target["target_year"],
        reduction_pct=target["reduction_pct"],
        status=target["status"],
        years_remaining=years_remaining,
        linear_annual_reduction_pct=target["linear_annual_reduction_pct"],
        required_annual_tco2e=annual_tco2e,
        on_track=None,
        generated_at=now,
    )


@router.post(
    "/{target_id}/submission",
    response_model=SubmissionFormResponse,
    summary="Generate submission form data",
    description=(
        "Generate pre-populated SBTi target submission form data. Evaluates "
        "completeness and identifies missing fields required for formal "
        "SBTi submission."
    ),
)
async def generate_submission_form(target_id: str) -> SubmissionFormResponse:
    """Generate SBTi submission form data."""
    target = _targets.get(target_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target {target_id} not found",
        )

    missing = []
    if not target.get("pathway_id"):
        missing.append("pathway_id")
    if target["scope"] in ("scope_3", "all_scopes") and not target.get("scope3_categories_included"):
        missing.append("scope3_categories_included")

    completeness = round(max(0, 100 - len(missing) * 15), 1)

    return SubmissionFormResponse(
        target_id=target["target_id"],
        org_id=target["org_id"],
        form_version="SBTi-v2.1",
        target_details={
            "target_name": target["target_name"],
            "target_type": target["target_type"],
            "scope": target["scope"],
            "base_year": target["base_year"],
            "target_year": target["target_year"],
            "reduction_pct": target["reduction_pct"],
        },
        boundary_details={
            "coverage_pct": target["boundary_coverage_pct"],
            "scope3_categories": target.get("scope3_categories_included", []),
        },
        methodology_details={
            "method": target["method"],
            "ambition_level": target["ambition_level"],
            "linear_annual_reduction": target["linear_annual_reduction_pct"],
        },
        ambition_assessment={
            "aligns_with_1_5C": target["ambition_level"] == "1.5C",
            "aligns_with_well_below_2C": True,
            "minimum_ambition_met": target["reduction_pct"] >= 25.0,
        },
        supporting_data={
            "base_year_emissions_tco2e": target["base_year_emissions_tco2e"],
            "verification_status": "pending",
        },
        completeness_pct=completeness,
        missing_fields=missing,
        generated_at=_now(),
    )


@router.get(
    "/org/{org_id}/scope3-requirement",
    response_model=Scope3RequirementResponse,
    summary="Check Scope 3 target requirement",
    description=(
        "Determine whether an organization is required to set a Scope 3 target "
        "based on the SBTi 40% threshold rule (C13). If Scope 3 emissions "
        "represent 40% or more of total Scope 1+2+3, a Scope 3 target is mandatory."
    ),
)
async def check_scope3_requirement(
    org_id: str,
    scope1_tco2e: float = Query(..., ge=0, description="Scope 1 emissions (tCO2e)"),
    scope2_tco2e: float = Query(..., ge=0, description="Scope 2 emissions (tCO2e)"),
    scope3_tco2e: float = Query(..., ge=0, description="Scope 3 emissions (tCO2e)"),
) -> Scope3RequirementResponse:
    """Check whether Scope 3 target is required."""
    total = scope1_tco2e + scope2_tco2e + scope3_tco2e
    scope3_pct = round((scope3_tco2e / total) * 100, 1) if total > 0 else 0.0
    threshold = 40.0
    required = scope3_pct >= threshold

    if required:
        recommendation = (
            f"Scope 3 emissions represent {scope3_pct}% of total emissions, exceeding "
            f"the {threshold}% threshold. A Scope 3 target covering at least 67% of "
            f"Scope 3 emissions is required per SBTi criteria C13."
        )
    else:
        recommendation = (
            f"Scope 3 emissions represent {scope3_pct}% of total emissions, below "
            f"the {threshold}% threshold. A Scope 3 target is encouraged but not mandatory."
        )

    return Scope3RequirementResponse(
        org_id=org_id,
        scope3_pct_of_total=scope3_pct,
        threshold_pct=threshold,
        scope3_target_required=required,
        minimum_coverage_pct=67.0 if required else 0.0,
        recommendation=recommendation,
        generated_at=_now(),
    )


@router.post(
    "/org/{org_id}/coverage-check",
    response_model=CoverageCheckResponse,
    summary="Validate target coverage",
    description=(
        "Validate that an organization's target coverage meets SBTi minimum "
        "requirements: 95% of Scope 1+2 and 67% of Scope 3 (if required). "
        "Returns pass/fail with specific issues and recommendations."
    ),
)
async def coverage_check(
    org_id: str,
    request: CoverageCheckRequest,
) -> CoverageCheckResponse:
    """Validate target coverage against SBTi minimums."""
    issues: List[str] = []
    recommendations: List[str] = []

    scope1_2_min = 95.0
    scope3_min = 67.0

    s12_ok = request.scope1_covered_pct >= scope1_2_min and request.scope2_covered_pct >= scope1_2_min
    if request.scope1_covered_pct < scope1_2_min:
        issues.append(
            f"Scope 1 coverage ({request.scope1_covered_pct}%) below {scope1_2_min}% minimum"
        )
        recommendations.append("Expand Scope 1 boundary to cover at least 95% of emissions")
    if request.scope2_covered_pct < scope1_2_min:
        issues.append(
            f"Scope 2 coverage ({request.scope2_covered_pct}%) below {scope1_2_min}% minimum"
        )
        recommendations.append("Expand Scope 2 boundary to cover at least 95% of emissions")

    scope3_required = request.total_scope3_pct_of_total >= 40.0
    s3_ok = True
    if scope3_required:
        s3_ok = request.scope3_covered_pct >= scope3_min
        if not s3_ok:
            issues.append(
                f"Scope 3 coverage ({request.scope3_covered_pct}%) below {scope3_min}% minimum"
            )
            recommendations.append(
                "Expand Scope 3 target boundary to cover at least 67% of Scope 3 emissions"
            )

    overall = s12_ok and s3_ok

    return CoverageCheckResponse(
        org_id=org_id,
        scope1_covered_pct=request.scope1_covered_pct,
        scope2_covered_pct=request.scope2_covered_pct,
        scope3_covered_pct=request.scope3_covered_pct,
        scope1_2_meets_minimum=s12_ok,
        scope3_meets_minimum=s3_ok,
        scope3_target_required=scope3_required,
        overall_valid=overall,
        issues=issues,
        recommendations=recommendations,
        generated_at=_now(),
    )
