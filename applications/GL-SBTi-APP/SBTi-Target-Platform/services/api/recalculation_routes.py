"""
GL-SBTi-APP Recalculation API

Manages base year recalculation triggers and processes per SBTi criterion
C19.  When significant changes exceed the 5% threshold (mergers,
acquisitions, divestitures, methodology changes, outsourcing/insourcing),
the base year emissions must be recalculated and the target revalidated.

Recalculation Triggers (SBTi C19):
    - Structural changes (M&A, divestitures) >5% of base year emissions
    - Methodology changes affecting >5% of inventory
    - Discovery of significant errors (>5%)
    - Insourcing/outsourcing of emitting activities
    - Organic growth does NOT trigger recalculation
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/sbti/recalculation", tags=["Recalculation"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ThresholdCheckRequest(BaseModel):
    """Request to check the 5% recalculation threshold."""
    org_id: str = Field(...)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    change_type: str = Field(
        ..., description="merger, acquisition, divestiture, methodology_change, error_correction, insourcing, outsourcing",
    )
    change_emissions_impact_tco2e: float = Field(
        ..., description="Absolute emission impact of the change",
    )
    description: str = Field(..., max_length=2000)


class CreateRecalculationRequest(BaseModel):
    """Request to create a recalculation record."""
    org_id: str = Field(...)
    target_id: str = Field(...)
    trigger_type: str = Field(...)
    original_base_year_emissions_tco2e: float = Field(..., gt=0)
    recalculated_base_year_emissions_tco2e: float = Field(..., gt=0)
    change_description: str = Field(..., max_length=2000)
    adjustment_methodology: str = Field(..., max_length=1000)
    supporting_evidence: Optional[str] = Field(None, max_length=2000)


class MAImpactRequest(BaseModel):
    """Request to model M&A impact on targets."""
    org_id: str = Field(...)
    transaction_type: str = Field(..., description="merger, acquisition, divestiture")
    target_entity_emissions_tco2e: float = Field(..., gt=0)
    acquiring_entity_emissions_tco2e: float = Field(..., gt=0)
    transaction_date: str = Field(...)
    expected_synergies_pct: float = Field(0, ge=0, le=100)
    notes: Optional[str] = Field(None, max_length=2000)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ThresholdCheckResponse(BaseModel):
    """Recalculation threshold check result."""
    org_id: str
    base_year_emissions_tco2e: float
    change_emissions_impact_tco2e: float
    change_pct: float
    threshold_pct: float
    recalculation_required: bool
    change_type: str
    recommendation: str
    generated_at: datetime


class RecalculationResponse(BaseModel):
    """Recalculation record."""
    recalculation_id: str
    org_id: str
    target_id: str
    trigger_type: str
    original_base_year_emissions_tco2e: float
    recalculated_base_year_emissions_tco2e: float
    change_pct: float
    change_description: str
    adjustment_methodology: str
    revalidation_required: bool
    status: str
    created_at: datetime


class RecalculationHistoryResponse(BaseModel):
    """Recalculation history."""
    org_id: str
    recalculations: List[Dict[str, Any]]
    total_count: int
    generated_at: datetime


class RevalidationResponse(BaseModel):
    """Revalidation assessment after recalculation."""
    recalculation_id: str
    revalidation_required: bool
    original_target_reduction_pct: float
    adjusted_target_reduction_pct: float
    target_still_valid: bool
    ambition_impact: str
    action_required: str
    generated_at: datetime


class MAImpactResponse(BaseModel):
    """M&A impact assessment."""
    org_id: str
    transaction_type: str
    pre_transaction_emissions_tco2e: float
    post_transaction_emissions_tco2e: float
    change_pct: float
    threshold_exceeded: bool
    recalculation_required: bool
    impact_on_target: Dict[str, Any]
    timeline: Dict[str, str]
    recommendation: str
    generated_at: datetime


class AuditTrailResponse(BaseModel):
    """Recalculation audit trail."""
    recalculation_id: str
    audit_entries: List[Dict[str, Any]]
    provenance_hash: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_recalculations: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/threshold-check",
    response_model=ThresholdCheckResponse,
    summary="Check 5% threshold",
    description=(
        "Check whether a structural change or event exceeds the SBTi 5% "
        "recalculation threshold (C19). If the change impacts more than "
        "5% of base year emissions, recalculation is required."
    ),
)
async def check_threshold(request: ThresholdCheckRequest) -> ThresholdCheckResponse:
    """Check recalculation threshold."""
    change_pct = round(
        abs(request.change_emissions_impact_tco2e) / request.base_year_emissions_tco2e * 100, 1,
    )
    required = change_pct >= 5.0

    if required:
        recommendation = (
            f"Change of {change_pct}% exceeds the 5% threshold. Base year "
            f"recalculation is required. Recalculate base year emissions, "
            f"revalidate target alignment, and document methodology."
        )
    else:
        recommendation = (
            f"Change of {change_pct}% is below the 5% threshold. "
            f"Recalculation is not required, but document the change "
            f"in the emissions inventory."
        )

    return ThresholdCheckResponse(
        org_id=request.org_id,
        base_year_emissions_tco2e=request.base_year_emissions_tco2e,
        change_emissions_impact_tco2e=request.change_emissions_impact_tco2e,
        change_pct=change_pct,
        threshold_pct=5.0,
        recalculation_required=required,
        change_type=request.change_type,
        recommendation=recommendation,
        generated_at=_now(),
    )


@router.post(
    "/",
    response_model=RecalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create recalculation record",
    description=(
        "Create a formal recalculation record documenting the base year "
        "adjustment, trigger type, methodology, and impact assessment."
    ),
)
async def create_recalculation(request: CreateRecalculationRequest) -> RecalculationResponse:
    """Create a recalculation record."""
    recalc_id = _generate_id("recalc")
    change_pct = round(
        abs(request.recalculated_base_year_emissions_tco2e - request.original_base_year_emissions_tco2e)
        / request.original_base_year_emissions_tco2e * 100, 1,
    )
    reval_required = change_pct >= 5.0

    data = {
        "recalculation_id": recalc_id,
        "org_id": request.org_id,
        "target_id": request.target_id,
        "trigger_type": request.trigger_type,
        "original_base_year_emissions_tco2e": request.original_base_year_emissions_tco2e,
        "recalculated_base_year_emissions_tco2e": request.recalculated_base_year_emissions_tco2e,
        "change_pct": change_pct,
        "change_description": request.change_description,
        "adjustment_methodology": request.adjustment_methodology,
        "revalidation_required": reval_required,
        "status": "pending_review",
        "created_at": _now(),
    }
    _recalculations[recalc_id] = data
    return RecalculationResponse(**data)


@router.get(
    "/org/{org_id}/history",
    response_model=RecalculationHistoryResponse,
    summary="Recalculation history",
    description="Get the history of all base year recalculations for an organization.",
)
async def get_recalculation_history(
    org_id: str,
    limit: int = Query(20, ge=1, le=100),
) -> RecalculationHistoryResponse:
    """Get recalculation history."""
    records = [r for r in _recalculations.values() if r["org_id"] == org_id]
    records.sort(key=lambda r: r["created_at"], reverse=True)

    return RecalculationHistoryResponse(
        org_id=org_id,
        recalculations=records[:limit],
        total_count=len(records),
        generated_at=_now(),
    )


@router.get(
    "/{recalculation_id}/revalidation",
    response_model=RevalidationResponse,
    summary="Revalidation assessment",
    description=(
        "Assess whether a recalculation requires target revalidation and "
        "evaluate the impact on target ambition."
    ),
)
async def get_revalidation(recalculation_id: str) -> RevalidationResponse:
    """Get revalidation assessment."""
    recalc = _recalculations.get(recalculation_id)
    if not recalc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recalculation {recalculation_id} not found",
        )

    original_pct = 42.0
    # If base year changed, the percentage reduction needs recalculating
    if recalc["recalculated_base_year_emissions_tco2e"] > recalc["original_base_year_emissions_tco2e"]:
        # Higher base year = easier to meet percentage target
        adjusted = round(original_pct * recalc["original_base_year_emissions_tco2e"] / recalc["recalculated_base_year_emissions_tco2e"], 1)
        still_valid = adjusted >= 25.0  # Minimum ambition
        impact = "reduced" if adjusted < original_pct else "unchanged"
    else:
        adjusted = round(original_pct * recalc["original_base_year_emissions_tco2e"] / recalc["recalculated_base_year_emissions_tco2e"], 1)
        still_valid = True
        impact = "increased"

    action = "No action required" if still_valid else "Target must be strengthened to maintain minimum ambition"

    return RevalidationResponse(
        recalculation_id=recalculation_id,
        revalidation_required=recalc["revalidation_required"],
        original_target_reduction_pct=original_pct,
        adjusted_target_reduction_pct=adjusted,
        target_still_valid=still_valid,
        ambition_impact=impact,
        action_required=action,
        generated_at=_now(),
    )


@router.post(
    "/ma-impact",
    response_model=MAImpactResponse,
    summary="Model M&A impact",
    description=(
        "Model the impact of a merger, acquisition, or divestiture on "
        "existing targets and determine recalculation requirements."
    ),
)
async def model_ma_impact(request: MAImpactRequest) -> MAImpactResponse:
    """Model M&A impact on targets."""
    if request.transaction_type in ("merger", "acquisition"):
        post = request.acquiring_entity_emissions_tco2e + request.target_entity_emissions_tco2e
    else:  # divestiture
        post = request.acquiring_entity_emissions_tco2e - request.target_entity_emissions_tco2e

    change = abs(post - request.acquiring_entity_emissions_tco2e)
    change_pct = round(change / request.acquiring_entity_emissions_tco2e * 100, 1) if request.acquiring_entity_emissions_tco2e > 0 else 0.0
    threshold_exceeded = change_pct >= 5.0

    synergy_savings = round(
        post * request.expected_synergies_pct / 100, 1,
    ) if request.expected_synergies_pct > 0 else 0.0

    return MAImpactResponse(
        org_id=request.org_id,
        transaction_type=request.transaction_type,
        pre_transaction_emissions_tco2e=request.acquiring_entity_emissions_tco2e,
        post_transaction_emissions_tco2e=round(post, 1),
        change_pct=change_pct,
        threshold_exceeded=threshold_exceeded,
        recalculation_required=threshold_exceeded,
        impact_on_target={
            "base_year_adjustment": "required" if threshold_exceeded else "not_required",
            "target_absolute_change_tco2e": round(change, 1),
            "expected_synergy_savings_tco2e": synergy_savings,
            "post_synergy_emissions": round(post - synergy_savings, 1),
        },
        timeline={
            "recalculation_deadline": "Within 6 months of transaction close",
            "revalidation_deadline": "Within 12 months of recalculation",
            "reporting_deadline": "Next annual disclosure cycle",
        },
        recommendation=(
            f"Transaction results in {change_pct}% change. "
            + ("Base year recalculation and target revalidation required. "
               "Engage SBTi for guidance on adjusted target." if threshold_exceeded
               else "Change below threshold; document in inventory notes.")
        ),
        generated_at=_now(),
    )


@router.get(
    "/{recalculation_id}/audit",
    response_model=AuditTrailResponse,
    summary="Recalculation audit trail",
    description="Get the full audit trail for a recalculation including provenance hash.",
)
async def get_audit_trail(recalculation_id: str) -> AuditTrailResponse:
    """Get recalculation audit trail."""
    recalc = _recalculations.get(recalculation_id)
    if not recalc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recalculation {recalculation_id} not found",
        )

    import hashlib
    provenance = hashlib.sha256(
        f"{recalculation_id}{recalc['original_base_year_emissions_tco2e']}"
        f"{recalc['recalculated_base_year_emissions_tco2e']}".encode()
    ).hexdigest()

    entries = [
        {"timestamp": recalc["created_at"].isoformat(), "event": "recalculation_created",
         "actor": "system", "details": recalc["change_description"]},
        {"timestamp": recalc["created_at"].isoformat(), "event": "threshold_check",
         "actor": "system", "details": f"Change of {recalc['change_pct']}% {'exceeds' if recalc['revalidation_required'] else 'below'} 5% threshold"},
        {"timestamp": recalc["created_at"].isoformat(), "event": "methodology_recorded",
         "actor": "system", "details": recalc["adjustment_methodology"]},
    ]

    return AuditTrailResponse(
        recalculation_id=recalculation_id,
        audit_entries=entries,
        provenance_hash=provenance,
        generated_at=_now(),
    )
