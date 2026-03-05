"""
GL-TCFD-APP Risk Management API

TCFD Pillar 3 -- Risk Management.  Manages climate risk records, risk
responses, heat map generation, ERM integration status, risk indicators,
risk review scheduling, and risk management disclosure text.

TCFD Recommended Disclosures (Risk Management):
    a) Processes for identifying and assessing climate-related risks
    b) Processes for managing climate-related risks
    c) Integration into overall risk management

ISSB/IFRS S2 references: paragraphs 25(a)-(c) (Risk Management).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/risk-management", tags=["Risk Management"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskCategory(str, Enum):
    PHYSICAL_ACUTE = "physical_acute"
    PHYSICAL_CHRONIC = "physical_chronic"
    TRANSITION_POLICY = "transition_policy"
    TRANSITION_TECHNOLOGY = "transition_technology"
    TRANSITION_MARKET = "transition_market"
    TRANSITION_REPUTATION = "transition_reputation"


class RiskStatus(str, Enum):
    IDENTIFIED = "identified"
    ASSESSED = "assessed"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    TRANSFERRED = "transferred"
    CLOSED = "closed"


class ResponseType(str, Enum):
    AVOID = "avoid"
    REDUCE = "reduce"
    TRANSFER = "transfer"
    ACCEPT = "accept"
    EXPLOIT = "exploit"


class ReviewFrequency(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateRiskRecordRequest(BaseModel):
    """Request to create a risk management record."""
    risk_name: str = Field(..., min_length=1, max_length=500, description="Risk name")
    category: RiskCategory = Field(..., description="Risk category")
    description: str = Field(..., min_length=1, max_length=5000, description="Risk description")
    impact_score: int = Field(..., ge=1, le=5, description="Impact score (1-5)")
    likelihood_score: int = Field(..., ge=1, le=5, description="Likelihood score (1-5)")
    velocity: str = Field("medium", description="Speed of impact: slow, medium, fast")
    risk_owner: str = Field(..., min_length=1, max_length=200, description="Risk owner name/role")
    status: RiskStatus = Field(RiskStatus.IDENTIFIED, description="Risk status")
    financial_exposure_usd: Optional[float] = Field(None, ge=0, description="Financial exposure (USD)")
    residual_impact_score: Optional[int] = Field(None, ge=1, le=5, description="Post-mitigation impact")
    residual_likelihood_score: Optional[int] = Field(None, ge=1, le=5, description="Post-mitigation likelihood")

    class Config:
        json_schema_extra = {
            "example": {
                "risk_name": "Carbon pricing regulation impact",
                "category": "transition_policy",
                "description": "Increasing carbon prices under EU ETS and potential CBAM expansion",
                "impact_score": 4,
                "likelihood_score": 5,
                "velocity": "medium",
                "risk_owner": "Chief Risk Officer",
                "status": "assessed",
                "financial_exposure_usd": 12000000,
                "residual_impact_score": 3,
                "residual_likelihood_score": 4,
            }
        }


class UpdateRiskRecordRequest(BaseModel):
    """Request to update a risk record."""
    risk_name: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)
    impact_score: Optional[int] = Field(None, ge=1, le=5)
    likelihood_score: Optional[int] = Field(None, ge=1, le=5)
    status: Optional[RiskStatus] = None
    risk_owner: Optional[str] = Field(None, max_length=200)
    financial_exposure_usd: Optional[float] = Field(None, ge=0)
    residual_impact_score: Optional[int] = Field(None, ge=1, le=5)
    residual_likelihood_score: Optional[int] = Field(None, ge=1, le=5)


class CreateRiskResponseRequest(BaseModel):
    """Request to create a risk response action."""
    response_type: ResponseType = Field(..., description="Response strategy")
    description: str = Field(..., min_length=1, max_length=3000, description="Response description")
    responsible_party: str = Field(..., min_length=1, max_length=200, description="Responsible party")
    target_date: str = Field(..., description="Target completion date (YYYY-MM-DD)")
    estimated_cost_usd: Optional[float] = Field(None, ge=0, description="Estimated cost")
    expected_risk_reduction_pct: float = Field(0.0, ge=0, le=100, description="Expected risk reduction (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "response_type": "reduce",
                "description": "Implement internal carbon pricing at $75/tCO2e to drive emissions reduction",
                "responsible_party": "VP Sustainability",
                "target_date": "2026-06-30",
                "estimated_cost_usd": 500000,
                "expected_risk_reduction_pct": 30,
            }
        }


class ScheduleReviewRequest(BaseModel):
    """Request to schedule a risk review."""
    review_date: str = Field(..., description="Review date (YYYY-MM-DD)")
    frequency: ReviewFrequency = Field(ReviewFrequency.QUARTERLY, description="Review frequency")
    participants: List[str] = Field(default_factory=list, description="Review participants")
    scope: str = Field("all", description="Review scope: all, physical, transition, specific_risks")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class RiskRecordResponse(BaseModel):
    """A risk management record."""
    record_id: str
    org_id: str
    risk_name: str
    category: str
    description: str
    impact_score: int
    likelihood_score: int
    inherent_risk_score: int
    velocity: str
    risk_owner: str
    status: str
    financial_exposure_usd: Optional[float]
    residual_impact_score: Optional[int]
    residual_likelihood_score: Optional[int]
    residual_risk_score: Optional[int]
    risk_responses: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class RiskResponseRecord(BaseModel):
    """A risk response action."""
    response_id: str
    risk_id: str
    response_type: str
    description: str
    responsible_party: str
    target_date: str
    estimated_cost_usd: Optional[float]
    expected_risk_reduction_pct: float
    status: str
    created_at: datetime


class HeatMapResponse(BaseModel):
    """Risk heat map data."""
    org_id: str
    cells: List[Dict[str, Any]]
    risk_count_by_category: Dict[str, int]
    high_risk_count: int
    total_risks: int
    generated_at: datetime


class ERMIntegrationResponse(BaseModel):
    """Enterprise Risk Management integration status."""
    org_id: str
    erm_framework: str
    climate_integrated: bool
    integration_score: float
    climate_risk_register_linked: bool
    board_reporting_frequency: str
    risk_appetite_defined: bool
    key_risk_indicators_count: int
    maturity_assessment: str
    gaps: List[str]
    recommendations: List[str]
    assessed_at: datetime


class RiskIndicatorsResponse(BaseModel):
    """Climate risk indicators (KRIs)."""
    org_id: str
    indicators: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    generated_at: datetime


class ReviewScheduleResponse(BaseModel):
    """Risk review schedule."""
    review_id: str
    org_id: str
    review_date: str
    frequency: str
    participants: List[str]
    scope: str
    status: str
    created_at: datetime


class RiskManagementDisclosureResponse(BaseModel):
    """Risk management disclosure text."""
    org_id: str
    pillar: str
    disclosure_a: str
    disclosure_b: str
    disclosure_c: str
    word_count: int
    compliance_score: float
    issb_references: List[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_records: Dict[str, Dict[str, Any]] = {}
_responses: Dict[str, Dict[str, Any]] = {}
_reviews: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/records",
    response_model=RiskRecordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create risk management record",
    description=(
        "Create a climate risk management record with impact and likelihood "
        "scores, financial exposure, and risk ownership."
    ),
)
async def create_risk_record(
    org_id: str = Query(..., description="Organization ID"),
    request: CreateRiskRecordRequest = ...,
) -> RiskRecordResponse:
    """Create a risk management record."""
    record_id = _generate_id("rmrec")
    now = _now()
    inherent = request.impact_score * request.likelihood_score
    residual = None
    if request.residual_impact_score and request.residual_likelihood_score:
        residual = request.residual_impact_score * request.residual_likelihood_score

    record = {
        "record_id": record_id,
        "org_id": org_id,
        "risk_name": request.risk_name,
        "category": request.category.value,
        "description": request.description,
        "impact_score": request.impact_score,
        "likelihood_score": request.likelihood_score,
        "inherent_risk_score": inherent,
        "velocity": request.velocity,
        "risk_owner": request.risk_owner,
        "status": request.status.value,
        "financial_exposure_usd": request.financial_exposure_usd,
        "residual_impact_score": request.residual_impact_score,
        "residual_likelihood_score": request.residual_likelihood_score,
        "residual_risk_score": residual,
        "risk_responses": [],
        "created_at": now,
        "updated_at": now,
    }
    _records[record_id] = record
    return RiskRecordResponse(**record)


@router.get(
    "/records/{org_id}",
    response_model=List[RiskRecordResponse],
    summary="List risk records",
    description="Retrieve all risk management records for an organization.",
)
async def list_risk_records(
    org_id: str,
    category: Optional[str] = Query(None, description="Filter by category"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
) -> List[RiskRecordResponse]:
    """List risk records."""
    results = [r for r in _records.values() if r["org_id"] == org_id]
    if category:
        results = [r for r in results if r["category"] == category]
    if status_filter:
        results = [r for r in results if r["status"] == status_filter]
    results.sort(key=lambda r: r["inherent_risk_score"], reverse=True)
    return [RiskRecordResponse(**r) for r in results[:limit]]


@router.get(
    "/records/{org_id}/{record_id}",
    response_model=RiskRecordResponse,
    summary="Get risk record detail",
    description="Retrieve a single risk management record by ID.",
)
async def get_risk_record(org_id: str, record_id: str) -> RiskRecordResponse:
    """Get a risk record."""
    record = _records.get(record_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Risk record {record_id} not found")
    if record["org_id"] != org_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Record does not belong to org {org_id}")
    return RiskRecordResponse(**record)


@router.put(
    "/records/{record_id}",
    response_model=RiskRecordResponse,
    summary="Update risk record",
    description="Update an existing risk management record.",
)
async def update_risk_record(record_id: str, request: UpdateRiskRecordRequest) -> RiskRecordResponse:
    """Update a risk record."""
    record = _records.get(record_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Risk record {record_id} not found")
    updates = request.model_dump(exclude_unset=True)
    if "status" in updates and hasattr(updates["status"], "value"):
        updates["status"] = updates["status"].value
    record.update(updates)
    record["inherent_risk_score"] = record["impact_score"] * record["likelihood_score"]
    if record.get("residual_impact_score") and record.get("residual_likelihood_score"):
        record["residual_risk_score"] = record["residual_impact_score"] * record["residual_likelihood_score"]
    record["updated_at"] = _now()
    return RiskRecordResponse(**record)


@router.post(
    "/responses/{risk_id}",
    response_model=RiskResponseRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Create risk response",
    description="Add a risk response action to a risk record.",
)
async def create_risk_response(risk_id: str, request: CreateRiskResponseRequest) -> RiskResponseRecord:
    """Create a risk response action."""
    record = _records.get(risk_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Risk record {risk_id} not found")

    response_id = _generate_id("rresp")
    now = _now()
    response = {
        "response_id": response_id,
        "risk_id": risk_id,
        "response_type": request.response_type.value,
        "description": request.description,
        "responsible_party": request.responsible_party,
        "target_date": request.target_date,
        "estimated_cost_usd": request.estimated_cost_usd,
        "expected_risk_reduction_pct": request.expected_risk_reduction_pct,
        "status": "planned",
        "created_at": now,
    }
    _responses[response_id] = response
    record["risk_responses"].append(response)
    record["updated_at"] = now
    return RiskResponseRecord(**response)


@router.get(
    "/heat-map/{org_id}",
    response_model=HeatMapResponse,
    summary="Risk heat map data",
    description="Generate risk heat map data (5x5 impact vs likelihood grid) for visualization.",
)
async def get_heat_map(org_id: str) -> HeatMapResponse:
    """Generate risk heat map data."""
    org_records = [r for r in _records.values() if r["org_id"] == org_id]
    cells = []
    for impact in range(1, 6):
        for likelihood in range(1, 6):
            matching = [
                r for r in org_records
                if r["impact_score"] == impact and r["likelihood_score"] == likelihood
            ]
            cells.append({
                "impact": impact,
                "likelihood": likelihood,
                "risk_score": impact * likelihood,
                "count": len(matching),
                "risks": [{"record_id": r["record_id"], "name": r["risk_name"]} for r in matching],
            })

    by_cat: Dict[str, int] = {}
    for r in org_records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1

    high_risk = sum(1 for r in org_records if r["inherent_risk_score"] >= 15)

    return HeatMapResponse(
        org_id=org_id,
        cells=cells,
        risk_count_by_category=by_cat,
        high_risk_count=high_risk,
        total_risks=len(org_records),
        generated_at=_now(),
    )


@router.get(
    "/erm/{org_id}",
    response_model=ERMIntegrationResponse,
    summary="ERM integration status",
    description="Assess the level of climate risk integration into the enterprise risk management framework.",
)
async def get_erm_integration(org_id: str) -> ERMIntegrationResponse:
    """Assess ERM integration status."""
    org_records = [r for r in _records.values() if r["org_id"] == org_id]
    has_records = len(org_records) > 0
    has_owners = all(r.get("risk_owner") for r in org_records) if org_records else False
    has_responses = any(r.get("risk_responses") for r in org_records) if org_records else False

    score = 0.0
    if has_records:
        score += 25
    if has_owners:
        score += 20
    if has_responses:
        score += 20
    if len(org_records) >= 5:
        score += 15
    score += min(len(org_records) * 2, 20)

    gaps = []
    if not has_records:
        gaps.append("No climate risks registered in the risk management framework")
    if not has_owners:
        gaps.append("Not all climate risks have designated owners")
    if not has_responses:
        gaps.append("No risk response actions defined")
    if score < 70:
        gaps.append("Climate risk integration into ERM is below maturity threshold")

    maturity = "leading" if score >= 85 else "managed" if score >= 70 else "defined" if score >= 50 else "developing" if score >= 25 else "initial"

    return ERMIntegrationResponse(
        org_id=org_id,
        erm_framework="COSO ERM 2017" if has_records else "Not defined",
        climate_integrated=score >= 50,
        integration_score=round(score, 1),
        climate_risk_register_linked=has_records,
        board_reporting_frequency="quarterly" if score >= 70 else "annually",
        risk_appetite_defined=score >= 60,
        key_risk_indicators_count=min(len(org_records) * 2, 15),
        maturity_assessment=maturity,
        gaps=gaps,
        recommendations=[
            "Integrate climate risks into the corporate risk register",
            "Define risk appetite for climate-related risks",
            "Establish regular board reporting on climate risk profile",
            "Link climate KRIs to early warning systems",
            "Conduct annual climate risk workshops with business units",
        ],
        assessed_at=_now(),
    )


@router.get(
    "/indicators/{org_id}",
    response_model=RiskIndicatorsResponse,
    summary="Risk indicators",
    description="Get key risk indicators (KRIs) for monitoring climate risk exposure.",
)
async def get_risk_indicators(org_id: str) -> RiskIndicatorsResponse:
    """Get climate risk indicators."""
    indicators = [
        {"name": "Carbon price (EU ETS)", "current_value": 85.0, "unit": "EUR/tCO2e", "trend": "increasing", "threshold": 100, "status": "amber"},
        {"name": "Scope 1+2 emissions", "current_value": 40000, "unit": "tCO2e", "trend": "decreasing", "threshold": 45000, "status": "green"},
        {"name": "Physical risk score (portfolio)", "current_value": 0.52, "unit": "score 0-1", "trend": "stable", "threshold": 0.6, "status": "amber"},
        {"name": "Transition readiness index", "current_value": 62, "unit": "%", "trend": "increasing", "threshold": 70, "status": "amber"},
        {"name": "Climate litigation cases", "current_value": 0, "unit": "count", "trend": "stable", "threshold": 1, "status": "green"},
        {"name": "Stranded asset ratio", "current_value": 15.2, "unit": "%", "trend": "stable", "threshold": 20, "status": "amber"},
        {"name": "ESG rating", "current_value": 68, "unit": "score 0-100", "trend": "increasing", "threshold": 60, "status": "green"},
        {"name": "Insurance premium change", "current_value": 12, "unit": "%", "trend": "increasing", "threshold": 15, "status": "amber"},
    ]

    alerts = [
        ind for ind in indicators if ind["status"] == "red"
        or (ind["trend"] == "increasing" and ind["status"] == "amber")
    ]

    return RiskIndicatorsResponse(
        org_id=org_id,
        indicators=indicators,
        alerts=alerts,
        generated_at=_now(),
    )


@router.post(
    "/review/{org_id}",
    response_model=ReviewScheduleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Schedule risk review",
    description="Schedule a periodic climate risk review meeting.",
)
async def schedule_review(org_id: str, request: ScheduleReviewRequest) -> ReviewScheduleResponse:
    """Schedule a risk review."""
    review_id = _generate_id("rev")
    now = _now()
    review = {
        "review_id": review_id,
        "org_id": org_id,
        "review_date": request.review_date,
        "frequency": request.frequency.value,
        "participants": request.participants,
        "scope": request.scope,
        "status": "scheduled",
        "created_at": now,
    }
    _reviews[review_id] = review
    return ReviewScheduleResponse(**review)


@router.get(
    "/disclosure/{org_id}",
    response_model=RiskManagementDisclosureResponse,
    summary="Risk management disclosure",
    description="Generate TCFD-aligned risk management disclosure text covering all three recommended disclosures.",
)
async def get_risk_management_disclosure(org_id: str) -> RiskManagementDisclosureResponse:
    """Generate risk management disclosure text."""
    org_records = [r for r in _records.values() if r["org_id"] == org_id]
    record_count = len(org_records)
    phys_count = sum(1 for r in org_records if r["category"].startswith("physical"))
    trans_count = sum(1 for r in org_records if r["category"].startswith("transition"))

    disclosure_a = (
        f"The organization has implemented a structured process for identifying and "
        f"assessing climate-related risks, covering both physical risks (acute and chronic) "
        f"and transition risks (policy, technology, market, and reputation). "
        f"{record_count} climate risks have been identified and assessed: "
        f"{phys_count} physical and {trans_count} transition risks. "
        f"Risks are assessed using a 5x5 impact-likelihood matrix with financial "
        f"quantification where feasible."
    )

    responses_count = sum(len(r.get("risk_responses", [])) for r in org_records)
    disclosure_b = (
        f"Climate risks are managed through a combination of avoidance, reduction, "
        f"transfer, and acceptance strategies. {responses_count} risk response actions "
        f"have been defined and are being tracked. The organization prioritizes risks "
        f"based on inherent risk score and financial exposure, with high-priority risks "
        f"subject to quarterly review by the risk management committee."
    )

    disclosure_c = (
        f"Climate-related risks are integrated into the organization's enterprise "
        f"risk management (ERM) framework. The climate risk register is linked to "
        f"the corporate risk register, and climate key risk indicators are monitored "
        f"alongside traditional business risks. The board receives regular updates "
        f"on the climate risk profile as part of its risk oversight responsibilities."
    )

    word_count = sum(len(d.split()) for d in [disclosure_a, disclosure_b, disclosure_c])
    score = min(record_count * 10 + responses_count * 5 + 20, 100.0)

    return RiskManagementDisclosureResponse(
        org_id=org_id,
        pillar="risk_management",
        disclosure_a=disclosure_a,
        disclosure_b=disclosure_b,
        disclosure_c=disclosure_c,
        word_count=word_count,
        compliance_score=round(score, 1),
        issb_references=["IFRS S2 para 25(a)", "IFRS S2 para 25(b)", "IFRS S2 para 25(c)"],
        generated_at=_now(),
    )
