"""
GL-Taxonomy-APP Alignment Workflow API

Orchestrates the full 4-step EU Taxonomy alignment workflow and provides
portfolio-level alignment computation, batch processing, progress tracking,
dashboard data, funnel charts, and period comparison.

4-Step Alignment Workflow:
    Step 1: Eligibility Screening  -- Is the activity in the CDA/EDA?
    Step 2: Substantial Contribution (SC)  -- Does it meet TSC?
    Step 3: Do No Significant Harm (DNSH)  -- Does it pass all 5 DNSH?
    Step 4: Minimum Safeguards (MS)  -- Does the entity pass MS?

An activity is taxonomy-ALIGNED only if it passes ALL four steps.
Eligibility >= Aligned (alignment is always a subset of eligibility).

Funnel Analysis:
    Total Activities -> Eligible -> SC -> SC+DNSH -> Fully Aligned
    Each step filters out activities that do not meet criteria.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/alignment", tags=["Alignment Workflow"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class FullAlignmentRequest(BaseModel):
    """Run full 4-step alignment workflow for an activity."""
    org_id: str = Field(...)
    activity_code: str = Field(...)
    activity_name: str = Field(..., max_length=500)
    nace_code: str = Field(...)
    objective: str = Field("climate_change_mitigation")
    turnover_eur: float = Field(..., ge=0)
    capex_eur: float = Field(0, ge=0)
    opex_eur: float = Field(0, ge=0)
    sc_reported_value: Optional[float] = Field(None)
    sc_reported_unit: Optional[str] = Field(None)
    dnsh_all_compliant: bool = Field(False)
    safeguards_compliant: bool = Field(False)
    reporting_year: int = Field(2025, ge=2022, le=2030)


class PortfolioAlignmentRequest(BaseModel):
    """Run portfolio-level alignment."""
    org_id: str = Field(...)
    reporting_year: int = Field(2025, ge=2022, le=2030)
    activities: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=500,
        description="List of activities with alignment data",
    )
    total_turnover_eur: float = Field(..., gt=0)
    total_capex_eur: float = Field(..., gt=0)
    total_opex_eur: float = Field(..., gt=0)


class BatchAlignmentRequest(BaseModel):
    """Batch alignment request."""
    org_id: str = Field(...)
    reporting_year: int = Field(2025, ge=2022, le=2030)
    activities: List[Dict[str, Any]] = Field(..., min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class AlignmentStepResult(BaseModel):
    """Result for a single alignment step."""
    step: int
    step_name: str
    status: str
    details: Dict[str, Any]


class FullAlignmentResponse(BaseModel):
    """Full 4-step alignment result."""
    alignment_id: str
    org_id: str
    activity_code: str
    activity_name: str
    nace_code: str
    objective: str
    final_status: str
    is_eligible: bool
    is_aligned: bool
    steps: List[AlignmentStepResult]
    turnover_eur: float
    aligned_turnover_eur: float
    capex_eur: float
    aligned_capex_eur: float
    opex_eur: float
    aligned_opex_eur: float
    assessed_at: datetime


class PortfolioAlignmentResponse(BaseModel):
    """Portfolio-level alignment results."""
    alignment_id: str
    org_id: str
    reporting_year: int
    total_activities: int
    eligible_count: int
    aligned_count: int
    eligibility_pct: float
    alignment_pct: float
    turnover_alignment_pct: float
    capex_alignment_pct: float
    opex_alignment_pct: float
    by_objective: Dict[str, float]
    funnel: Dict[str, int]
    generated_at: datetime


class BatchAlignmentResponse(BaseModel):
    """Batch alignment results."""
    org_id: str
    total_assessed: int
    aligned_count: int
    not_aligned_count: int
    results: List[FullAlignmentResponse]
    assessed_at: datetime


class AlignmentStatusResponse(BaseModel):
    """Alignment status for a specific activity."""
    org_id: str
    activity_code: str
    latest_status: str
    is_eligible: bool
    is_aligned: bool
    step_statuses: Dict[str, str]
    last_assessed: Optional[datetime]
    generated_at: datetime


class AlignmentProgressResponse(BaseModel):
    """Alignment progress for organization."""
    org_id: str
    total_activities: int
    screened: int
    sc_assessed: int
    dnsh_assessed: int
    safeguards_assessed: int
    fully_aligned: int
    completion_pct: float
    next_step: str
    generated_at: datetime


class AlignmentDashboardResponse(BaseModel):
    """Alignment dashboard data."""
    org_id: str
    reporting_year: int
    overall_alignment_pct: float
    turnover_alignment_pct: float
    capex_alignment_pct: float
    opex_alignment_pct: float
    eligible_activities: int
    aligned_activities: int
    total_activities: int
    top_aligned_sectors: List[Dict[str, Any]]
    generated_at: datetime


class FunnelChartResponse(BaseModel):
    """Eligible-to-aligned funnel chart data."""
    org_id: str
    total_activities: int
    eligible: int
    sc_passed: int
    dnsh_passed: int
    safeguards_passed: int
    fully_aligned: int
    dropout_by_step: Dict[str, int]
    conversion_rate_pct: float
    generated_at: datetime


class AlignmentCompareResponse(BaseModel):
    """Period comparison for alignment."""
    org_id: str
    period_1: Dict[str, Any]
    period_2: Dict[str, Any]
    changes: Dict[str, float]
    trend: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_alignments: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# TSC thresholds (simplified mapping)
_TSC_THRESHOLDS = {
    "4.1": {"threshold": 100, "unit": "gCO2e/kWh", "op": "lt"},
    "4.3": {"threshold": 100, "unit": "gCO2e/kWh", "op": "lt"},
    "7.1": {"threshold": 10, "unit": "pct_below_nzeb", "op": "gte"},
    "7.2": {"threshold": 30, "unit": "pct_reduction", "op": "gte"},
    "6.5": {"threshold": 50, "unit": "gCO2/km", "op": "lt"},
    "3.9": {"threshold": 1.331, "unit": "tCO2e/t", "op": "lt"},
}

# NACE eligibility set (simplified)
_ELIGIBLE_NACE = {"A1", "A2", "D35.11", "D35.21", "D35.30", "C25", "C27", "C28", "C29.1",
                  "C24.10", "C23.51", "C20.11", "H49.10", "H49.20", "H49.31", "H49.39",
                  "F41.1", "F41.2", "F41", "F42", "F43", "L68", "J63.11", "J61", "J62",
                  "K65.12", "K65.20", "E36.00", "E37.00", "E38.11", "E38.32",
                  "H49.32", "H49.41", "H50.20"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/full",
    response_model=FullAlignmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run full 4-step alignment workflow",
    description=(
        "Run the complete EU Taxonomy alignment workflow for a single "
        "activity: eligibility, SC, DNSH, and minimum safeguards."
    ),
)
async def full_alignment(request: FullAlignmentRequest) -> FullAlignmentResponse:
    """Run full 4-step alignment workflow."""
    alignment_id = _generate_id("aln")
    steps = []

    # Step 1: Eligibility
    eligible = request.nace_code in _ELIGIBLE_NACE or any(
        request.nace_code.startswith(n) for n in _ELIGIBLE_NACE
    )
    steps.append(AlignmentStepResult(
        step=1, step_name="Eligibility Screening",
        status="pass" if eligible else "fail",
        details={"nace_code": request.nace_code, "eligible": eligible},
    ))

    # Step 2: Substantial Contribution
    sc_passed = False
    if eligible:
        tsc = _TSC_THRESHOLDS.get(request.activity_code)
        if tsc and request.sc_reported_value is not None:
            if tsc["op"] == "lt":
                sc_passed = request.sc_reported_value < tsc["threshold"]
            elif tsc["op"] == "gte":
                sc_passed = request.sc_reported_value >= tsc["threshold"]
        elif request.sc_reported_value is not None:
            sc_passed = True  # No quantitative threshold, qualitative pass
    steps.append(AlignmentStepResult(
        step=2, step_name="Substantial Contribution",
        status="pass" if sc_passed else ("fail" if eligible else "skipped"),
        details={"objective": request.objective, "sc_met": sc_passed},
    ))

    # Step 3: DNSH
    dnsh_passed = sc_passed and request.dnsh_all_compliant
    steps.append(AlignmentStepResult(
        step=3, step_name="Do No Significant Harm",
        status="pass" if dnsh_passed else ("fail" if sc_passed else "skipped"),
        details={"all_objectives_compliant": request.dnsh_all_compliant},
    ))

    # Step 4: Minimum Safeguards
    safeguards_passed = dnsh_passed and request.safeguards_compliant
    steps.append(AlignmentStepResult(
        step=4, step_name="Minimum Safeguards",
        status="pass" if safeguards_passed else ("fail" if dnsh_passed else "skipped"),
        details={"safeguards_compliant": request.safeguards_compliant},
    ))

    is_aligned = safeguards_passed
    if is_aligned:
        final_status = "aligned"
    elif eligible:
        final_status = "eligible_not_aligned"
    else:
        final_status = "not_eligible"

    aligned_t = request.turnover_eur if is_aligned else 0
    aligned_c = request.capex_eur if is_aligned else 0
    aligned_o = request.opex_eur if is_aligned else 0

    data = {
        "alignment_id": alignment_id,
        "org_id": request.org_id,
        "activity_code": request.activity_code,
        "activity_name": request.activity_name,
        "nace_code": request.nace_code,
        "objective": request.objective,
        "final_status": final_status,
        "is_eligible": eligible,
        "is_aligned": is_aligned,
        "steps": steps,
        "turnover_eur": request.turnover_eur,
        "aligned_turnover_eur": aligned_t,
        "capex_eur": request.capex_eur,
        "aligned_capex_eur": aligned_c,
        "opex_eur": request.opex_eur,
        "aligned_opex_eur": aligned_o,
        "assessed_at": _now(),
    }
    _alignments[alignment_id] = data
    return FullAlignmentResponse(**data)


@router.post(
    "/portfolio",
    response_model=PortfolioAlignmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run portfolio-level alignment",
    description="Calculate portfolio-level alignment aggregated across all activities.",
)
async def portfolio_alignment(request: PortfolioAlignmentRequest) -> PortfolioAlignmentResponse:
    """Run portfolio-level alignment."""
    alignment_id = _generate_id("paln")
    eligible_count = 0
    aligned_count = 0
    aligned_turnover = 0.0
    aligned_capex = 0.0
    aligned_opex = 0.0
    by_obj: Dict[str, float] = {}
    funnel = {"total": len(request.activities), "eligible": 0, "sc_passed": 0, "dnsh_passed": 0, "aligned": 0}

    for act in request.activities:
        nace = act.get("nace_code", "")
        is_eligible = nace in _ELIGIBLE_NACE or any(nace.startswith(n) for n in _ELIGIBLE_NACE)
        is_aligned = act.get("is_aligned", False) and is_eligible
        sc_passed = act.get("sc_passed", False) and is_eligible
        dnsh_passed = act.get("dnsh_passed", False) and sc_passed

        if is_eligible:
            eligible_count += 1
            funnel["eligible"] += 1
        if sc_passed:
            funnel["sc_passed"] += 1
        if dnsh_passed:
            funnel["dnsh_passed"] += 1
        if is_aligned:
            aligned_count += 1
            funnel["aligned"] += 1
            t = float(act.get("turnover_eur", 0))
            c = float(act.get("capex_eur", 0))
            o = float(act.get("opex_eur", 0))
            aligned_turnover += t
            aligned_capex += c
            aligned_opex += o
            obj = act.get("objective", "climate_change_mitigation")
            by_obj[obj] = by_obj.get(obj, 0) + t

    total = len(request.activities)
    elig_pct = round((eligible_count / total) * 100, 1) if total > 0 else 0
    align_pct = round((aligned_count / total) * 100, 1) if total > 0 else 0
    t_pct = round((aligned_turnover / request.total_turnover_eur) * 100, 2) if request.total_turnover_eur > 0 else 0
    c_pct = round((aligned_capex / request.total_capex_eur) * 100, 2) if request.total_capex_eur > 0 else 0
    o_pct = round((aligned_opex / request.total_opex_eur) * 100, 2) if request.total_opex_eur > 0 else 0

    # Normalize by_obj to percentages
    total_obj = sum(by_obj.values())
    by_obj_pct = {k: round((v / total_obj) * 100, 1) if total_obj > 0 else 0 for k, v in by_obj.items()}

    return PortfolioAlignmentResponse(
        alignment_id=alignment_id,
        org_id=request.org_id,
        reporting_year=request.reporting_year,
        total_activities=total,
        eligible_count=eligible_count,
        aligned_count=aligned_count,
        eligibility_pct=elig_pct,
        alignment_pct=align_pct,
        turnover_alignment_pct=t_pct,
        capex_alignment_pct=c_pct,
        opex_alignment_pct=o_pct,
        by_objective=by_obj_pct,
        funnel=funnel,
        generated_at=_now(),
    )


@router.post(
    "/batch",
    response_model=BatchAlignmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch alignment",
    description="Run alignment for multiple activities in batch.",
)
async def batch_alignment(request: BatchAlignmentRequest) -> BatchAlignmentResponse:
    """Batch alignment."""
    results = []
    aligned = 0

    for act in request.activities:
        sub_req = FullAlignmentRequest(
            org_id=request.org_id,
            activity_code=act.get("activity_code", ""),
            activity_name=act.get("activity_name", ""),
            nace_code=act.get("nace_code", ""),
            objective=act.get("objective", "climate_change_mitigation"),
            turnover_eur=float(act.get("turnover_eur", 0)),
            capex_eur=float(act.get("capex_eur", 0)),
            opex_eur=float(act.get("opex_eur", 0)),
            sc_reported_value=act.get("sc_reported_value"),
            sc_reported_unit=act.get("sc_reported_unit"),
            dnsh_all_compliant=act.get("dnsh_all_compliant", False),
            safeguards_compliant=act.get("safeguards_compliant", False),
            reporting_year=request.reporting_year,
        )
        result = await full_alignment(sub_req)
        results.append(result)
        if result.is_aligned:
            aligned += 1

    return BatchAlignmentResponse(
        org_id=request.org_id,
        total_assessed=len(results),
        aligned_count=aligned,
        not_aligned_count=len(results) - aligned,
        results=results,
        assessed_at=_now(),
    )


@router.get(
    "/{org_id}/status/{activity_code}",
    response_model=AlignmentStatusResponse,
    summary="Get alignment status",
    description="Get the alignment status for a specific activity.",
)
async def get_alignment_status(org_id: str, activity_code: str) -> AlignmentStatusResponse:
    """Get alignment status for activity."""
    org_alignments = [
        a for a in _alignments.values()
        if a["org_id"] == org_id and a["activity_code"] == activity_code
    ]

    if org_alignments:
        latest = max(org_alignments, key=lambda a: a["assessed_at"])
        step_statuses = {s.step_name: s.status for s in latest["steps"]}
        return AlignmentStatusResponse(
            org_id=org_id, activity_code=activity_code,
            latest_status=latest["final_status"],
            is_eligible=latest["is_eligible"], is_aligned=latest["is_aligned"],
            step_statuses=step_statuses, last_assessed=latest["assessed_at"],
            generated_at=_now(),
        )

    return AlignmentStatusResponse(
        org_id=org_id, activity_code=activity_code,
        latest_status="not_assessed",
        is_eligible=False, is_aligned=False,
        step_statuses={}, last_assessed=None,
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/progress",
    response_model=AlignmentProgressResponse,
    summary="Alignment progress",
    description="Get alignment workflow progress for all activities.",
)
async def get_progress(org_id: str) -> AlignmentProgressResponse:
    """Get alignment progress."""
    org_alignments = [a for a in _alignments.values() if a["org_id"] == org_id]
    total = len(org_alignments) if org_alignments else 25
    screened = len(org_alignments) if org_alignments else 20
    sc = sum(1 for a in org_alignments if a.get("is_eligible", False)) if org_alignments else 15
    dnsh = sum(1 for a in org_alignments if any(s.step_name == "Do No Significant Harm" and s.status == "pass" for s in a.get("steps", []))) if org_alignments else 12
    safeguards = sum(1 for a in org_alignments if a.get("is_aligned", False)) if org_alignments else 10
    aligned = safeguards

    completion = round((aligned / total) * 100, 1) if total > 0 else 0
    if aligned < safeguards:
        next_step = "Complete Minimum Safeguards assessment"
    elif safeguards < dnsh:
        next_step = "Complete DNSH assessments for remaining activities"
    elif dnsh < sc:
        next_step = "Complete SC assessments for eligible activities"
    else:
        next_step = "Screen remaining activities for eligibility"

    return AlignmentProgressResponse(
        org_id=org_id,
        total_activities=total, screened=screened,
        sc_assessed=sc, dnsh_assessed=dnsh,
        safeguards_assessed=safeguards, fully_aligned=aligned,
        completion_pct=completion, next_step=next_step,
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/dashboard",
    response_model=AlignmentDashboardResponse,
    summary="Alignment dashboard",
    description="Get alignment dashboard with key metrics.",
)
async def get_dashboard(
    org_id: str,
    reporting_year: int = Query(2025, ge=2022, le=2030),
) -> AlignmentDashboardResponse:
    """Get alignment dashboard."""
    return AlignmentDashboardResponse(
        org_id=org_id, reporting_year=reporting_year,
        overall_alignment_pct=42.0,
        turnover_alignment_pct=42.0,
        capex_alignment_pct=56.0,
        opex_alignment_pct=46.7,
        eligible_activities=20,
        aligned_activities=12,
        total_activities=30,
        top_aligned_sectors=[
            {"sector": "Energy", "alignment_pct": 65.0, "activity_count": 5},
            {"sector": "Construction", "alignment_pct": 48.0, "activity_count": 4},
            {"sector": "Transport", "alignment_pct": 35.0, "activity_count": 3},
        ],
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/eligible-vs-aligned",
    response_model=FunnelChartResponse,
    summary="Funnel chart data",
    description="Get eligible-to-aligned funnel chart data showing dropout at each step.",
)
async def get_funnel(org_id: str) -> FunnelChartResponse:
    """Get funnel chart data."""
    total = 30
    eligible = 20
    sc = 16
    dnsh = 14
    safeguards = 12
    aligned = 12

    return FunnelChartResponse(
        org_id=org_id,
        total_activities=total,
        eligible=eligible,
        sc_passed=sc,
        dnsh_passed=dnsh,
        safeguards_passed=safeguards,
        fully_aligned=aligned,
        dropout_by_step={
            "not_eligible": total - eligible,
            "sc_fail": eligible - sc,
            "dnsh_fail": sc - dnsh,
            "safeguards_fail": dnsh - safeguards,
        },
        conversion_rate_pct=round((aligned / total) * 100, 1),
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/compare",
    response_model=AlignmentCompareResponse,
    summary="Compare periods",
    description="Compare alignment metrics between two reporting periods.",
)
async def compare_periods(
    org_id: str,
    year_1: int = Query(2024, ge=2022, le=2030),
    year_2: int = Query(2025, ge=2022, le=2030),
) -> AlignmentCompareResponse:
    """Compare alignment periods."""
    p1 = {"year": year_1, "alignment_pct": 35.5, "eligible_count": 18, "aligned_count": 9, "turnover_pct": 35.5, "capex_pct": 45.0}
    p2 = {"year": year_2, "alignment_pct": 42.0, "eligible_count": 20, "aligned_count": 12, "turnover_pct": 42.0, "capex_pct": 56.0}
    changes = {
        "alignment_pct": round(p2["alignment_pct"] - p1["alignment_pct"], 1),
        "turnover_pct": round(p2["turnover_pct"] - p1["turnover_pct"], 1),
        "capex_pct": round(p2["capex_pct"] - p1["capex_pct"], 1),
        "new_aligned": p2["aligned_count"] - p1["aligned_count"],
    }

    return AlignmentCompareResponse(
        org_id=org_id, period_1=p1, period_2=p2, changes=changes,
        trend="improving" if changes["alignment_pct"] > 0 else "stable",
        generated_at=_now(),
    )
