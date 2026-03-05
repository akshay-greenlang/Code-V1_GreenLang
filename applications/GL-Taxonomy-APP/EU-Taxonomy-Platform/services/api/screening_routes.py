"""
GL-Taxonomy-APP Eligibility Screening API

Screens economic activities against the EU Taxonomy activity catalog to
determine taxonomy eligibility.  Supports individual and batch screening
by NACE code, de minimis threshold application, and sector-level breakdown
of eligible vs. non-eligible turnover.

Eligibility is the first gate of the 4-step EU Taxonomy alignment test:
    Step 1: Eligibility Screening  <-- this router
    Step 2: Substantial Contribution (SC)
    Step 3: Do No Significant Harm (DNSH)
    Step 4: Minimum Safeguards (MS)

An activity is taxonomy-eligible if it appears in the Climate Delegated
Act (CDA), Complementary CDA, or Environmental Delegated Act (EDA)
annexes, regardless of whether it meets the TSC criteria.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/screening", tags=["Eligibility Screening"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EligibilityStatus(str, Enum):
    """Screening eligibility status."""
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    PARTIALLY_ELIGIBLE = "partially_eligible"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class EligibilityScreenRequest(BaseModel):
    """Request to screen a single activity for taxonomy eligibility."""
    org_id: str = Field(..., description="Organization ID")
    activity_name: str = Field(..., min_length=1, max_length=500)
    nace_code: str = Field(..., description="NACE code of the activity (e.g. D35.11)")
    turnover_eur: float = Field(..., ge=0, description="Annual turnover in EUR")
    capex_eur: float = Field(0, ge=0, description="Annual CapEx in EUR")
    opex_eur: float = Field(0, ge=0, description="Annual OpEx in EUR")
    description: Optional[str] = Field(None, max_length=2000)

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_001",
                "activity_name": "Solar PV Installation",
                "nace_code": "D35.11",
                "turnover_eur": 5000000,
                "capex_eur": 1200000,
                "opex_eur": 300000,
            }
        }


class BatchScreenRequest(BaseModel):
    """Request to batch-screen multiple NACE codes."""
    org_id: str = Field(...)
    activities: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=500,
        description="List of {nace_code, activity_name, turnover_eur, capex_eur, opex_eur}",
    )


class DeMinimisRequest(BaseModel):
    """Apply de minimis threshold to screening results."""
    threshold_pct: float = Field(
        5.0, ge=0, le=100,
        description="De minimis threshold percentage (default 5%)",
    )
    apply_to: str = Field("turnover", description="turnover, capex, or opex")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class EligibilityResultResponse(BaseModel):
    """Single activity eligibility result."""
    screening_id: str
    org_id: str
    activity_name: str
    nace_code: str
    eligibility_status: str
    matched_activity_codes: List[str]
    matched_activity_names: List[str]
    objectives_available: List[str]
    delegated_act: Optional[str]
    turnover_eur: float
    capex_eur: float
    opex_eur: float
    is_transitional: bool
    is_enabling: bool
    confidence: float
    screened_at: datetime


class BatchScreenResponse(BaseModel):
    """Batch screening results."""
    org_id: str
    total_screened: int
    eligible_count: int
    not_eligible_count: int
    eligible_turnover_eur: float
    total_turnover_eur: float
    eligibility_ratio_pct: float
    results: List[EligibilityResultResponse]
    screened_at: datetime


class ScreeningSummaryResponse(BaseModel):
    """Screening summary for organization."""
    org_id: str
    total_activities: int
    eligible_count: int
    not_eligible_count: int
    eligible_turnover_eur: float
    eligible_capex_eur: float
    eligible_opex_eur: float
    total_turnover_eur: float
    total_capex_eur: float
    total_opex_eur: float
    turnover_eligibility_pct: float
    capex_eligibility_pct: float
    opex_eligibility_pct: float
    generated_at: datetime


class SectorBreakdownResponse(BaseModel):
    """Sector breakdown of screening results."""
    org_id: str
    sectors: List[Dict[str, Any]]
    total_eligible_turnover_eur: float
    total_turnover_eur: float
    generated_at: datetime


class DeMinimisResponse(BaseModel):
    """De minimis threshold application result."""
    org_id: str
    threshold_pct: float
    applied_to: str
    activities_below_threshold: int
    activities_above_threshold: int
    reclassified_count: int
    summary: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_screenings: Dict[str, Dict[str, Any]] = {}

# NACE-to-activity mapping (simplified from activity catalog)
NACE_ACTIVITY_MAP: Dict[str, List[Dict[str, Any]]] = {
    "A1": [{"code": "1.1", "name": "Afforestation", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "A2": [{"code": "1.1", "name": "Afforestation", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "D35.11": [
        {"code": "4.1", "name": "Electricity generation using solar PV", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False},
        {"code": "4.3", "name": "Electricity generation from wind power", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False},
        {"code": "4.29", "name": "Electricity generation from fossil gaseous fuels", "objectives": ["climate_change_mitigation"], "da": "CDA_COMP", "transitional": True, "enabling": False},
    ],
    "D35.21": [{"code": "4.13", "name": "Manufacture of biogas and biofuels", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "D35.30": [{"code": "4.15", "name": "District heating/cooling distribution", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": True, "enabling": False}],
    "C25": [{"code": "3.1", "name": "Manufacture of renewable energy technologies", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": True}],
    "C27": [{"code": "3.1", "name": "Manufacture of renewable energy technologies", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": True}],
    "C28": [{"code": "3.1", "name": "Manufacture of renewable energy technologies", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": True}],
    "C29.1": [{"code": "3.3", "name": "Manufacture of low carbon transport technologies", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": True}],
    "C24.10": [{"code": "3.9", "name": "Manufacture of iron and steel", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": True, "enabling": False}],
    "C23.51": [{"code": "3.12", "name": "Manufacture of cement", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": True, "enabling": False}],
    "C20.11": [{"code": "3.10", "name": "Manufacture of hydrogen", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "H49.10": [{"code": "6.1", "name": "Passenger interurban rail transport", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "H49.20": [{"code": "6.2", "name": "Freight rail transport", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "F41.1": [{"code": "7.1", "name": "Construction of new buildings", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "F41.2": [{"code": "7.1", "name": "Construction of new buildings", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "L68": [{"code": "7.7", "name": "Acquisition and ownership of buildings", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": False}],
    "J63.11": [{"code": "8.1", "name": "Data processing, hosting and related activities", "objectives": ["climate_change_mitigation"], "da": "CDA", "transitional": False, "enabling": True}],
    "K65.12": [{"code": "10.1", "name": "Non-life insurance: climate-related perils", "objectives": ["climate_change_adaptation"], "da": "CDA", "transitional": False, "enabling": True}],
}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _screen_nace(nace_code: str) -> List[Dict[str, Any]]:
    """Screen a NACE code against the activity catalog."""
    # Direct match
    if nace_code in NACE_ACTIVITY_MAP:
        return NACE_ACTIVITY_MAP[nace_code]
    # Prefix match (e.g. C24 matches C24.10)
    matches = []
    for key, activities in NACE_ACTIVITY_MAP.items():
        if key.startswith(nace_code) or nace_code.startswith(key):
            matches.extend(activities)
    return matches


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/eligibility",
    response_model=EligibilityResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Screen activity eligibility",
    description=(
        "Screen a single economic activity against the EU Taxonomy activity "
        "catalog to determine taxonomy eligibility. Matches by NACE code "
        "and returns matched taxonomy activities and applicable objectives."
    ),
)
async def screen_eligibility(request: EligibilityScreenRequest) -> EligibilityResultResponse:
    """Screen a single activity for taxonomy eligibility."""
    matched = _screen_nace(request.nace_code)
    screening_id = _generate_id("scr")

    if matched:
        eligibility = EligibilityStatus.ELIGIBLE.value
        codes = list({m["code"] for m in matched})
        names = list({m["name"] for m in matched})
        objectives = list({obj for m in matched for obj in m["objectives"]})
        da = matched[0]["da"]
        transitional = any(m["transitional"] for m in matched)
        enabling = any(m["enabling"] for m in matched)
        confidence = 0.95
    else:
        eligibility = EligibilityStatus.NOT_ELIGIBLE.value
        codes = []
        names = []
        objectives = []
        da = None
        transitional = False
        enabling = False
        confidence = 0.90

    data = {
        "screening_id": screening_id,
        "org_id": request.org_id,
        "activity_name": request.activity_name,
        "nace_code": request.nace_code,
        "eligibility_status": eligibility,
        "matched_activity_codes": codes,
        "matched_activity_names": names,
        "objectives_available": objectives,
        "delegated_act": da,
        "turnover_eur": request.turnover_eur,
        "capex_eur": request.capex_eur,
        "opex_eur": request.opex_eur,
        "is_transitional": transitional,
        "is_enabling": enabling,
        "confidence": confidence,
        "screened_at": _now(),
    }
    _screenings[screening_id] = data
    return EligibilityResultResponse(**data)


@router.post(
    "/batch",
    response_model=BatchScreenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch screen multiple NACE codes",
    description=(
        "Screen multiple economic activities in a single request. Returns "
        "individual eligibility results and aggregate eligibility ratio."
    ),
)
async def batch_screen(request: BatchScreenRequest) -> BatchScreenResponse:
    """Batch screen multiple activities."""
    results = []
    eligible_count = 0
    eligible_turnover = 0.0
    total_turnover = 0.0

    for activity in request.activities:
        nace = activity.get("nace_code", "")
        name = activity.get("activity_name", "Unknown")
        turnover = float(activity.get("turnover_eur", 0))
        capex = float(activity.get("capex_eur", 0))
        opex = float(activity.get("opex_eur", 0))

        matched = _screen_nace(nace)
        screening_id = _generate_id("scr")
        total_turnover += turnover

        if matched:
            eligible_count += 1
            eligible_turnover += turnover
            eligibility = EligibilityStatus.ELIGIBLE.value
            codes = list({m["code"] for m in matched})
            names = list({m["name"] for m in matched})
            objectives = list({obj for m in matched for obj in m["objectives"]})
            da = matched[0]["da"]
            transitional = any(m["transitional"] for m in matched)
            enabling = any(m["enabling"] for m in matched)
            confidence = 0.95
        else:
            eligibility = EligibilityStatus.NOT_ELIGIBLE.value
            codes = []
            names = []
            objectives = []
            da = None
            transitional = False
            enabling = False
            confidence = 0.90

        entry = {
            "screening_id": screening_id,
            "org_id": request.org_id,
            "activity_name": name,
            "nace_code": nace,
            "eligibility_status": eligibility,
            "matched_activity_codes": codes,
            "matched_activity_names": names,
            "objectives_available": objectives,
            "delegated_act": da,
            "turnover_eur": turnover,
            "capex_eur": capex,
            "opex_eur": opex,
            "is_transitional": transitional,
            "is_enabling": enabling,
            "confidence": confidence,
            "screened_at": _now(),
        }
        _screenings[screening_id] = entry
        results.append(EligibilityResultResponse(**entry))

    ratio = round((eligible_turnover / total_turnover) * 100, 1) if total_turnover > 0 else 0.0

    return BatchScreenResponse(
        org_id=request.org_id,
        total_screened=len(request.activities),
        eligible_count=eligible_count,
        not_eligible_count=len(request.activities) - eligible_count,
        eligible_turnover_eur=round(eligible_turnover, 2),
        total_turnover_eur=round(total_turnover, 2),
        eligibility_ratio_pct=ratio,
        results=results,
        screened_at=_now(),
    )


@router.get(
    "/{org_id}/results",
    response_model=List[EligibilityResultResponse],
    summary="Get screening results",
    description="Retrieve all screening results for an organization.",
)
async def get_results(
    org_id: str,
    eligibility_status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
) -> List[EligibilityResultResponse]:
    """Get screening results for organization."""
    results = [s for s in _screenings.values() if s["org_id"] == org_id]
    if eligibility_status:
        results = [s for s in results if s["eligibility_status"] == eligibility_status]
    results.sort(key=lambda s: s["screened_at"], reverse=True)
    return [EligibilityResultResponse(**s) for s in results[:limit]]


@router.get(
    "/{org_id}/summary",
    response_model=ScreeningSummaryResponse,
    summary="Get eligibility summary",
    description="Get aggregated eligibility summary with KPI ratios.",
)
async def get_summary(org_id: str) -> ScreeningSummaryResponse:
    """Get eligibility summary for organization."""
    org_screenings = [s for s in _screenings.values() if s["org_id"] == org_id]

    if not org_screenings:
        return ScreeningSummaryResponse(
            org_id=org_id,
            total_activities=0,
            eligible_count=0,
            not_eligible_count=0,
            eligible_turnover_eur=0,
            eligible_capex_eur=0,
            eligible_opex_eur=0,
            total_turnover_eur=0,
            total_capex_eur=0,
            total_opex_eur=0,
            turnover_eligibility_pct=0,
            capex_eligibility_pct=0,
            opex_eligibility_pct=0,
            generated_at=_now(),
        )

    eligible = [s for s in org_screenings if s["eligibility_status"] == "eligible"]
    total_t = sum(s["turnover_eur"] for s in org_screenings)
    total_c = sum(s["capex_eur"] for s in org_screenings)
    total_o = sum(s["opex_eur"] for s in org_screenings)
    elig_t = sum(s["turnover_eur"] for s in eligible)
    elig_c = sum(s["capex_eur"] for s in eligible)
    elig_o = sum(s["opex_eur"] for s in eligible)

    return ScreeningSummaryResponse(
        org_id=org_id,
        total_activities=len(org_screenings),
        eligible_count=len(eligible),
        not_eligible_count=len(org_screenings) - len(eligible),
        eligible_turnover_eur=round(elig_t, 2),
        eligible_capex_eur=round(elig_c, 2),
        eligible_opex_eur=round(elig_o, 2),
        total_turnover_eur=round(total_t, 2),
        total_capex_eur=round(total_c, 2),
        total_opex_eur=round(total_o, 2),
        turnover_eligibility_pct=round((elig_t / total_t) * 100, 1) if total_t > 0 else 0,
        capex_eligibility_pct=round((elig_c / total_c) * 100, 1) if total_c > 0 else 0,
        opex_eligibility_pct=round((elig_o / total_o) * 100, 1) if total_o > 0 else 0,
        generated_at=_now(),
    )


@router.post(
    "/{org_id}/de-minimis",
    response_model=DeMinimisResponse,
    summary="Apply de minimis threshold",
    description=(
        "Apply a de minimis threshold to screening results. Activities "
        "contributing less than the threshold percentage are reclassified "
        "as immaterial for reporting purposes."
    ),
)
async def apply_de_minimis(
    org_id: str,
    request: DeMinimisRequest,
) -> DeMinimisResponse:
    """Apply de minimis threshold to screening results."""
    org_screenings = [s for s in _screenings.values() if s["org_id"] == org_id]
    if not org_screenings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No screening results found for org {org_id}.",
        )

    kpi_field = f"{request.apply_to}_eur"
    total = sum(s.get(kpi_field, 0) for s in org_screenings)
    if total <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Total {request.apply_to} is zero. Cannot apply de minimis.",
        )

    below = 0
    above = 0
    reclassified = 0
    for s in org_screenings:
        pct = (s.get(kpi_field, 0) / total) * 100
        if pct < request.threshold_pct:
            below += 1
            if s["eligibility_status"] == "eligible":
                reclassified += 1
        else:
            above += 1

    return DeMinimisResponse(
        org_id=org_id,
        threshold_pct=request.threshold_pct,
        applied_to=request.apply_to,
        activities_below_threshold=below,
        activities_above_threshold=above,
        reclassified_count=reclassified,
        summary=(
            f"Applied {request.threshold_pct}% de minimis threshold to {request.apply_to}. "
            f"{below} activities below threshold, {reclassified} reclassified as immaterial."
        ),
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/sector-breakdown",
    response_model=SectorBreakdownResponse,
    summary="Sector breakdown",
    description="Get sector-level breakdown of eligible vs. non-eligible activities.",
)
async def get_sector_breakdown(org_id: str) -> SectorBreakdownResponse:
    """Get sector breakdown of screening results."""
    org_screenings = [s for s in _screenings.values() if s["org_id"] == org_id]

    # Aggregate by matched sector (from activity codes)
    sector_map: Dict[str, Dict[str, float]] = {}
    total_eligible = 0.0
    total_all = 0.0

    for s in org_screenings:
        total_all += s["turnover_eur"]
        # Derive sector from first digit of activity code
        for code in s.get("matched_activity_codes", []):
            sector_num = code.split(".")[0] if "." in code else "0"
            sector_label = {
                "1": "Forestry", "2": "Environmental Protection",
                "3": "Manufacturing", "4": "Energy",
                "5": "Water Supply & Waste", "6": "Transport",
                "7": "Construction & Real Estate", "8": "ICT",
                "9": "Professional/Scientific", "10": "Financial & Insurance",
            }.get(sector_num, "Other")

            if sector_label not in sector_map:
                sector_map[sector_label] = {"eligible_turnover": 0, "total_turnover": 0, "count": 0}
            sector_map[sector_label]["eligible_turnover"] += s["turnover_eur"]
            sector_map[sector_label]["count"] += 1
            total_eligible += s["turnover_eur"]
            break

    sectors = [
        {
            "sector": name,
            "eligible_turnover_eur": round(data["eligible_turnover"], 2),
            "activity_count": int(data["count"]),
            "pct_of_eligible": round((data["eligible_turnover"] / total_eligible) * 100, 1) if total_eligible > 0 else 0,
        }
        for name, data in sorted(sector_map.items(), key=lambda x: x[1]["eligible_turnover"], reverse=True)
    ]

    return SectorBreakdownResponse(
        org_id=org_id,
        sectors=sectors,
        total_eligible_turnover_eur=round(total_eligible, 2),
        total_turnover_eur=round(total_all, 2),
        generated_at=_now(),
    )


@router.delete(
    "/{screening_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete screening",
    description="Delete a specific screening result.",
)
async def delete_screening(screening_id: str) -> None:
    """Delete a screening result."""
    if screening_id not in _screenings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screening {screening_id} not found.",
        )
    del _screenings[screening_id]
    return None
