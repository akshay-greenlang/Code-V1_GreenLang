"""
GL-TCFD-APP Governance API

TCFD Pillar 1 -- Governance.  Manages board and management-level governance
assessments, role assignments, maturity scoring, board climate competency
evaluation, and governance disclosure text generation.

TCFD Recommended Disclosures (Governance):
    a) Board oversight of climate-related risks and opportunities
    b) Management's role in assessing and managing climate-related risks
       and opportunities

ISSB/IFRS S2 references: paragraphs 26-27 (Governance).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/tcfd/governance", tags=["Governance"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OversightFrequency(str, Enum):
    """How often the board reviews climate topics."""
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    AD_HOC = "ad_hoc"


class MaturityLevel(str, Enum):
    """Governance maturity levels (1-5)."""
    LEVEL_1_INITIAL = "level_1_initial"
    LEVEL_2_DEVELOPING = "level_2_developing"
    LEVEL_3_DEFINED = "level_3_defined"
    LEVEL_4_MANAGED = "level_4_managed"
    LEVEL_5_LEADING = "level_5_leading"


class GovernanceRoleType(str, Enum):
    """Types of governance roles for climate oversight."""
    BOARD_CHAIR = "board_chair"
    BOARD_MEMBER = "board_member"
    COMMITTEE_CHAIR = "committee_chair"
    CEO = "ceo"
    CFO = "cfo"
    CSO = "cso"
    CRO = "cro"
    SUSTAINABILITY_DIRECTOR = "sustainability_director"
    RISK_MANAGER = "risk_manager"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateGovernanceAssessmentRequest(BaseModel):
    """Request to create a governance assessment."""
    reporting_year: int = Field(..., ge=2015, le=2100, description="Reporting year")
    board_oversight: bool = Field(..., description="Board has climate oversight responsibility")
    oversight_frequency: OversightFrequency = Field(
        ..., description="Frequency of board climate review"
    )
    dedicated_committee: bool = Field(
        False, description="Dedicated sustainability/climate committee exists"
    )
    committee_name: Optional[str] = Field(
        None, max_length=200, description="Name of the committee"
    )
    management_accountability: bool = Field(
        ..., description="Management has explicit climate accountability"
    )
    climate_in_strategy: bool = Field(
        False, description="Climate integrated into strategic planning"
    )
    climate_in_risk: bool = Field(
        False, description="Climate integrated into ERM framework"
    )
    climate_in_remuneration: bool = Field(
        False, description="Climate KPIs linked to executive remuneration"
    )
    remuneration_details: Optional[str] = Field(
        None, max_length=2000, description="Details on climate-linked remuneration"
    )
    notes: Optional[str] = Field(None, max_length=5000)

    class Config:
        json_schema_extra = {
            "example": {
                "reporting_year": 2025,
                "board_oversight": True,
                "oversight_frequency": "quarterly",
                "dedicated_committee": True,
                "committee_name": "Sustainability & Climate Risk Committee",
                "management_accountability": True,
                "climate_in_strategy": True,
                "climate_in_risk": True,
                "climate_in_remuneration": True,
                "remuneration_details": "15% of annual bonus linked to Scope 1+2 reduction targets",
            }
        }


class UpdateGovernanceAssessmentRequest(BaseModel):
    """Request to update an existing governance assessment."""
    board_oversight: Optional[bool] = None
    oversight_frequency: Optional[OversightFrequency] = None
    dedicated_committee: Optional[bool] = None
    committee_name: Optional[str] = Field(None, max_length=200)
    management_accountability: Optional[bool] = None
    climate_in_strategy: Optional[bool] = None
    climate_in_risk: Optional[bool] = None
    climate_in_remuneration: Optional[bool] = None
    remuneration_details: Optional[str] = Field(None, max_length=2000)
    notes: Optional[str] = Field(None, max_length=5000)


class AddGovernanceRoleRequest(BaseModel):
    """Request to add a governance role."""
    role_type: GovernanceRoleType = Field(..., description="Type of governance role")
    person_name: str = Field(..., min_length=1, max_length=200, description="Name of the person")
    title: str = Field(..., min_length=1, max_length=200, description="Job title")
    climate_responsibilities: str = Field(
        ..., min_length=1, max_length=2000,
        description="Description of climate-related responsibilities"
    )
    has_climate_expertise: bool = Field(
        False, description="Person has formal climate/sustainability expertise"
    )
    expertise_description: Optional[str] = Field(
        None, max_length=1000, description="Description of climate expertise"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "role_type": "committee_chair",
                "person_name": "Dr. Elena Torres",
                "title": "Chair, Sustainability & Climate Risk Committee",
                "climate_responsibilities": "Oversees climate risk assessment, reviews scenario analysis, approves TCFD disclosures",
                "has_climate_expertise": True,
                "expertise_description": "PhD in Environmental Science, 10 years in climate risk consulting",
            }
        }


class UpdateGovernanceRoleRequest(BaseModel):
    """Request to update a governance role."""
    person_name: Optional[str] = Field(None, min_length=1, max_length=200)
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    climate_responsibilities: Optional[str] = Field(None, max_length=2000)
    has_climate_expertise: Optional[bool] = None
    expertise_description: Optional[str] = Field(None, max_length=1000)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class GovernanceAssessmentResponse(BaseModel):
    """Governance assessment result."""
    assessment_id: str
    org_id: str
    reporting_year: int
    board_oversight: bool
    oversight_frequency: str
    dedicated_committee: bool
    committee_name: Optional[str]
    management_accountability: bool
    climate_in_strategy: bool
    climate_in_risk: bool
    climate_in_remuneration: bool
    remuneration_details: Optional[str]
    governance_score: float
    maturity_level: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class GovernanceRoleResponse(BaseModel):
    """A governance role assignment."""
    role_id: str
    org_id: str
    role_type: str
    person_name: str
    title: str
    climate_responsibilities: str
    has_climate_expertise: bool
    expertise_description: Optional[str]
    created_at: datetime
    updated_at: datetime


class MaturityScoreResponse(BaseModel):
    """Governance maturity score breakdown."""
    org_id: str
    overall_score: float
    maturity_level: str
    board_oversight_score: float
    management_accountability_score: float
    integration_score: float
    remuneration_score: float
    competency_score: float
    dimensions: Dict[str, float]
    recommendations: List[str]
    assessed_at: datetime


class CompetencyAssessmentResponse(BaseModel):
    """Board climate competency assessment."""
    org_id: str
    total_board_members: int
    members_with_climate_expertise: int
    competency_ratio: float
    competency_areas: Dict[str, int]
    training_programs: List[str]
    gaps: List[str]
    recommendations: List[str]
    assessed_at: datetime


class GovernanceDisclosureResponse(BaseModel):
    """Generated governance disclosure text."""
    org_id: str
    reporting_year: int
    pillar: str
    disclosure_a: str
    disclosure_b: str
    word_count: int
    compliance_score: float
    issb_references: List[str]
    generated_at: datetime


class GovernanceBenchmarkResponse(BaseModel):
    """Peer governance benchmarking result."""
    org_id: str
    org_score: float
    org_maturity: str
    peer_average_score: float
    peer_median_score: float
    sector_average_score: float
    percentile_rank: int
    peer_count: int
    comparison: Dict[str, Dict[str, float]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_assessments: Dict[str, Dict[str, Any]] = {}
_roles: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _calculate_governance_score(data: Dict[str, Any]) -> float:
    """Calculate a governance score 0-100 from assessment data."""
    score = 0.0
    if data.get("board_oversight"):
        score += 20.0
    freq_scores = {"quarterly": 15.0, "semi_annually": 10.0, "annually": 5.0, "ad_hoc": 2.0}
    score += freq_scores.get(data.get("oversight_frequency", ""), 0.0)
    if data.get("dedicated_committee"):
        score += 15.0
    if data.get("management_accountability"):
        score += 15.0
    if data.get("climate_in_strategy"):
        score += 12.0
    if data.get("climate_in_risk"):
        score += 12.0
    if data.get("climate_in_remuneration"):
        score += 11.0
    return min(round(score, 1), 100.0)


def _maturity_from_score(score: float) -> str:
    """Map governance score to maturity level."""
    if score >= 85:
        return MaturityLevel.LEVEL_5_LEADING.value
    if score >= 70:
        return MaturityLevel.LEVEL_4_MANAGED.value
    if score >= 50:
        return MaturityLevel.LEVEL_3_DEFINED.value
    if score >= 30:
        return MaturityLevel.LEVEL_2_DEVELOPING.value
    return MaturityLevel.LEVEL_1_INITIAL.value


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/assessments",
    response_model=GovernanceAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create governance assessment",
    description=(
        "Create a new TCFD Governance pillar assessment for an organization. "
        "Captures board oversight, management accountability, committee structure, "
        "and climate integration into strategy, risk, and remuneration."
    ),
)
async def create_governance_assessment(
    org_id: str = Query(..., description="Organization ID"),
    request: CreateGovernanceAssessmentRequest = ...,
) -> GovernanceAssessmentResponse:
    """Create a new governance assessment."""
    assessment_id = _generate_id("gov")
    now = _now()
    data = {
        "assessment_id": assessment_id,
        "org_id": org_id,
        "reporting_year": request.reporting_year,
        "board_oversight": request.board_oversight,
        "oversight_frequency": request.oversight_frequency.value,
        "dedicated_committee": request.dedicated_committee,
        "committee_name": request.committee_name,
        "management_accountability": request.management_accountability,
        "climate_in_strategy": request.climate_in_strategy,
        "climate_in_risk": request.climate_in_risk,
        "climate_in_remuneration": request.climate_in_remuneration,
        "remuneration_details": request.remuneration_details,
        "notes": request.notes,
        "created_at": now,
        "updated_at": now,
    }
    data["governance_score"] = _calculate_governance_score(data)
    data["maturity_level"] = _maturity_from_score(data["governance_score"])
    _assessments[assessment_id] = data
    return GovernanceAssessmentResponse(**data)


@router.get(
    "/assessments/{org_id}",
    response_model=List[GovernanceAssessmentResponse],
    summary="List governance assessments for organization",
    description="Retrieve all governance assessments for an organization, ordered by reporting year descending.",
)
async def list_governance_assessments(
    org_id: str,
    reporting_year: Optional[int] = Query(None, ge=2015, le=2100, description="Filter by year"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> List[GovernanceAssessmentResponse]:
    """List governance assessments for an organization."""
    results = [a for a in _assessments.values() if a["org_id"] == org_id]
    if reporting_year is not None:
        results = [a for a in results if a["reporting_year"] == reporting_year]
    results.sort(key=lambda a: a["reporting_year"], reverse=True)
    return [GovernanceAssessmentResponse(**a) for a in results[:limit]]


@router.get(
    "/assessments/{org_id}/{assessment_id}",
    response_model=GovernanceAssessmentResponse,
    summary="Get governance assessment detail",
    description="Retrieve a single governance assessment by ID.",
)
async def get_governance_assessment(
    org_id: str,
    assessment_id: str,
) -> GovernanceAssessmentResponse:
    """Retrieve a governance assessment by ID."""
    assessment = _assessments.get(assessment_id)
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Governance assessment {assessment_id} not found",
        )
    if assessment["org_id"] != org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Assessment {assessment_id} does not belong to organization {org_id}",
        )
    return GovernanceAssessmentResponse(**assessment)


@router.put(
    "/assessments/{assessment_id}",
    response_model=GovernanceAssessmentResponse,
    summary="Update governance assessment",
    description="Update an existing governance assessment and recalculate scores.",
)
async def update_governance_assessment(
    assessment_id: str,
    request: UpdateGovernanceAssessmentRequest,
) -> GovernanceAssessmentResponse:
    """Update an existing governance assessment."""
    assessment = _assessments.get(assessment_id)
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Governance assessment {assessment_id} not found",
        )
    updates = request.model_dump(exclude_unset=True)
    if "oversight_frequency" in updates and hasattr(updates["oversight_frequency"], "value"):
        updates["oversight_frequency"] = updates["oversight_frequency"].value
    assessment.update(updates)
    assessment["governance_score"] = _calculate_governance_score(assessment)
    assessment["maturity_level"] = _maturity_from_score(assessment["governance_score"])
    assessment["updated_at"] = _now()
    return GovernanceAssessmentResponse(**assessment)


@router.post(
    "/roles",
    response_model=GovernanceRoleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add governance role",
    description=(
        "Add a governance role with climate-related responsibilities. "
        "Covers board members, committee chairs, and management positions "
        "with explicit climate accountability."
    ),
)
async def add_governance_role(
    org_id: str = Query(..., description="Organization ID"),
    request: AddGovernanceRoleRequest = ...,
) -> GovernanceRoleResponse:
    """Add a governance role."""
    role_id = _generate_id("grole")
    now = _now()
    role = {
        "role_id": role_id,
        "org_id": org_id,
        "role_type": request.role_type.value,
        "person_name": request.person_name,
        "title": request.title,
        "climate_responsibilities": request.climate_responsibilities,
        "has_climate_expertise": request.has_climate_expertise,
        "expertise_description": request.expertise_description,
        "created_at": now,
        "updated_at": now,
    }
    _roles[role_id] = role
    return GovernanceRoleResponse(**role)


@router.get(
    "/roles/{org_id}",
    response_model=List[GovernanceRoleResponse],
    summary="List governance roles",
    description="List all governance roles with climate responsibilities for an organization.",
)
async def list_governance_roles(
    org_id: str,
    role_type: Optional[str] = Query(None, description="Filter by role type"),
) -> List[GovernanceRoleResponse]:
    """List governance roles for an organization."""
    results = [r for r in _roles.values() if r["org_id"] == org_id]
    if role_type:
        results = [r for r in results if r["role_type"] == role_type]
    results.sort(key=lambda r: r["created_at"])
    return [GovernanceRoleResponse(**r) for r in results]


@router.put(
    "/roles/{role_id}",
    response_model=GovernanceRoleResponse,
    summary="Update governance role",
    description="Update an existing governance role assignment.",
)
async def update_governance_role(
    role_id: str,
    request: UpdateGovernanceRoleRequest,
) -> GovernanceRoleResponse:
    """Update a governance role."""
    role = _roles.get(role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Governance role {role_id} not found",
        )
    updates = request.model_dump(exclude_unset=True)
    role.update(updates)
    role["updated_at"] = _now()
    return GovernanceRoleResponse(**role)


@router.delete(
    "/roles/{role_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove governance role",
    description="Remove a governance role assignment.",
)
async def delete_governance_role(role_id: str) -> None:
    """Delete a governance role."""
    if role_id not in _roles:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Governance role {role_id} not found",
        )
    del _roles[role_id]
    return None


@router.get(
    "/maturity/{org_id}",
    response_model=MaturityScoreResponse,
    summary="Get governance maturity scores",
    description=(
        "Calculate multi-dimensional governance maturity scores across board "
        "oversight, management accountability, integration, remuneration, and "
        "competency.  Returns overall maturity level and dimension-level scores."
    ),
)
async def get_maturity_scores(org_id: str) -> MaturityScoreResponse:
    """Calculate governance maturity scores."""
    org_assessments = [a for a in _assessments.values() if a["org_id"] == org_id]
    if not org_assessments:
        # Return default maturity for orgs with no assessments
        return MaturityScoreResponse(
            org_id=org_id,
            overall_score=0.0,
            maturity_level=MaturityLevel.LEVEL_1_INITIAL.value,
            board_oversight_score=0.0,
            management_accountability_score=0.0,
            integration_score=0.0,
            remuneration_score=0.0,
            competency_score=0.0,
            dimensions={},
            recommendations=[
                "Complete initial governance assessment",
                "Establish board-level climate oversight",
                "Define management climate accountability",
            ],
            assessed_at=_now(),
        )

    latest = max(org_assessments, key=lambda a: a["reporting_year"])
    board_score = 0.0
    if latest.get("board_oversight"):
        board_score += 50.0
    freq_map = {"quarterly": 30.0, "semi_annually": 20.0, "annually": 10.0, "ad_hoc": 5.0}
    board_score += freq_map.get(latest.get("oversight_frequency", ""), 0.0)
    if latest.get("dedicated_committee"):
        board_score += 20.0

    mgmt_score = 0.0
    if latest.get("management_accountability"):
        mgmt_score += 60.0
    org_roles = [r for r in _roles.values() if r["org_id"] == org_id]
    mgmt_score += min(len(org_roles) * 10.0, 40.0)

    integration_score = 0.0
    if latest.get("climate_in_strategy"):
        integration_score += 50.0
    if latest.get("climate_in_risk"):
        integration_score += 50.0

    remuneration_score = 0.0
    if latest.get("climate_in_remuneration"):
        remuneration_score = 80.0
        if latest.get("remuneration_details"):
            remuneration_score = 100.0

    experts = [r for r in org_roles if r.get("has_climate_expertise")]
    competency_score = min(len(experts) * 25.0, 100.0)

    overall = round(
        (board_score * 0.25 + mgmt_score * 0.20 + integration_score * 0.25
         + remuneration_score * 0.15 + competency_score * 0.15), 1
    )

    dimensions = {
        "board_oversight": round(board_score, 1),
        "management_accountability": round(mgmt_score, 1),
        "strategic_integration": round(integration_score, 1),
        "remuneration_linkage": round(remuneration_score, 1),
        "climate_competency": round(competency_score, 1),
    }

    recommendations = []
    if board_score < 70:
        recommendations.append("Increase frequency of board climate reviews to quarterly")
    if not latest.get("dedicated_committee"):
        recommendations.append("Establish a dedicated sustainability/climate committee")
    if mgmt_score < 70:
        recommendations.append("Define explicit management roles for climate risk assessment")
    if integration_score < 70:
        recommendations.append("Integrate climate considerations into strategic planning and ERM")
    if remuneration_score < 70:
        recommendations.append("Link executive remuneration to climate KPIs")
    if competency_score < 50:
        recommendations.append("Develop board climate competency through training programs")

    return MaturityScoreResponse(
        org_id=org_id,
        overall_score=overall,
        maturity_level=_maturity_from_score(overall),
        board_oversight_score=round(board_score, 1),
        management_accountability_score=round(mgmt_score, 1),
        integration_score=round(integration_score, 1),
        remuneration_score=round(remuneration_score, 1),
        competency_score=round(competency_score, 1),
        dimensions=dimensions,
        recommendations=recommendations,
        assessed_at=_now(),
    )


@router.get(
    "/competency/{org_id}",
    response_model=CompetencyAssessmentResponse,
    summary="Get board competency assessment",
    description=(
        "Assess the climate competency of board and management governance "
        "members.  Returns expertise ratio, competency area coverage, and "
        "training recommendations."
    ),
)
async def get_competency_assessment(org_id: str) -> CompetencyAssessmentResponse:
    """Assess board climate competency."""
    org_roles = [r for r in _roles.values() if r["org_id"] == org_id]
    board_roles = [
        r for r in org_roles
        if r["role_type"] in ("board_chair", "board_member", "committee_chair")
    ]
    total = len(board_roles)
    experts = [r for r in board_roles if r.get("has_climate_expertise")]
    ratio = round(len(experts) / total, 2) if total > 0 else 0.0

    competency_areas: Dict[str, int] = {
        "climate_science": 0,
        "risk_management": 0,
        "regulatory_compliance": 0,
        "sustainability_strategy": 0,
        "financial_analysis": 0,
    }
    for expert in experts:
        desc = (expert.get("expertise_description") or "").lower()
        if "climate" in desc or "environmental" in desc:
            competency_areas["climate_science"] += 1
        if "risk" in desc:
            competency_areas["risk_management"] += 1
        if "regulat" in desc or "compliance" in desc:
            competency_areas["regulatory_compliance"] += 1
        if "sustainab" in desc or "strategy" in desc:
            competency_areas["sustainability_strategy"] += 1
        if "financ" in desc or "accounting" in desc:
            competency_areas["financial_analysis"] += 1

    training_programs = [
        "TCFD Implementation Masterclass",
        "Climate Scenario Analysis for Directors",
        "Physical and Transition Risk Assessment",
    ]

    gaps = []
    if competency_areas["climate_science"] == 0:
        gaps.append("No board members with climate science expertise")
    if competency_areas["risk_management"] == 0:
        gaps.append("No board members with climate risk management expertise")
    if ratio < 0.3:
        gaps.append(f"Climate competency ratio ({ratio:.0%}) below 30% threshold")

    recommendations = []
    if ratio < 0.3:
        recommendations.append("Recruit board members with climate expertise")
    if competency_areas["climate_science"] == 0:
        recommendations.append("Provide climate science training to at least 2 board members")
    recommendations.append("Schedule annual board climate education sessions")

    return CompetencyAssessmentResponse(
        org_id=org_id,
        total_board_members=total,
        members_with_climate_expertise=len(experts),
        competency_ratio=ratio,
        competency_areas=competency_areas,
        training_programs=training_programs,
        gaps=gaps,
        recommendations=recommendations,
        assessed_at=_now(),
    )


@router.get(
    "/disclosure/{org_id}/{year}",
    response_model=GovernanceDisclosureResponse,
    summary="Generate governance disclosure text",
    description=(
        "Generate TCFD-aligned governance disclosure text for the specified "
        "organization and reporting year.  Produces Disclosure (a) on board "
        "oversight and Disclosure (b) on management role."
    ),
)
async def generate_governance_disclosure(
    org_id: str,
    year: int,
) -> GovernanceDisclosureResponse:
    """Generate governance disclosure text."""
    org_assessments = [
        a for a in _assessments.values()
        if a["org_id"] == org_id and a["reporting_year"] == year
    ]
    latest = org_assessments[0] if org_assessments else None

    if latest:
        freq_text = latest.get("oversight_frequency", "periodically").replace("_", " ")
        committee_text = ""
        if latest.get("dedicated_committee"):
            committee_text = (
                f" The {latest.get('committee_name', 'Sustainability Committee')} is "
                f"responsible for overseeing climate-related risks and opportunities."
            )
        disclosure_a = (
            f"The Board of Directors maintains oversight of climate-related risks "
            f"and opportunities through {freq_text} reviews of the organization's "
            f"climate strategy, risk profile, and progress against targets.{committee_text} "
            f"Climate considerations are {'integrated into' if latest.get('climate_in_strategy') else 'being integrated into'} "
            f"the organization's strategic planning process."
        )
        remun_text = ""
        if latest.get("climate_in_remuneration"):
            remun_text = (
                f" Climate-related performance metrics are incorporated into "
                f"executive remuneration. {latest.get('remuneration_details', '')}"
            )
        disclosure_b = (
            f"Management {'has' if latest.get('management_accountability') else 'is developing'} "
            f"explicit accountability for assessing and managing climate-related risks "
            f"and opportunities. Climate risk assessment is "
            f"{'integrated into' if latest.get('climate_in_risk') else 'being integrated into'} "
            f"the enterprise risk management framework.{remun_text}"
        )
    else:
        disclosure_a = (
            "The organization is in the process of establishing board-level "
            "oversight of climate-related risks and opportunities in line with "
            "TCFD recommendations."
        )
        disclosure_b = (
            "The organization is developing management processes for assessing "
            "and managing climate-related risks and opportunities."
        )

    word_count = len(disclosure_a.split()) + len(disclosure_b.split())
    compliance_score = 0.0
    if latest:
        compliance_score = min(latest.get("governance_score", 0.0) * 1.1, 100.0)

    return GovernanceDisclosureResponse(
        org_id=org_id,
        reporting_year=year,
        pillar="governance",
        disclosure_a=disclosure_a,
        disclosure_b=disclosure_b,
        word_count=word_count,
        compliance_score=round(compliance_score, 1),
        issb_references=["IFRS S2 para 26(a)", "IFRS S2 para 26(b)", "IFRS S2 para 27"],
        generated_at=_now(),
    )


@router.get(
    "/benchmark/{org_id}",
    response_model=GovernanceBenchmarkResponse,
    summary="Peer governance benchmarking",
    description=(
        "Compare the organization's governance maturity against sector peers. "
        "Returns percentile rank, peer average, and dimension-level comparison."
    ),
)
async def get_governance_benchmark(org_id: str) -> GovernanceBenchmarkResponse:
    """Benchmark governance maturity against peers."""
    org_assessments = [a for a in _assessments.values() if a["org_id"] == org_id]
    org_score = 0.0
    if org_assessments:
        latest = max(org_assessments, key=lambda a: a["reporting_year"])
        org_score = latest.get("governance_score", 0.0)

    # Simulated peer data
    peer_scores = [42.0, 48.5, 55.0, 58.3, 62.0, 65.5, 70.0, 72.8, 78.0, 85.0]
    peer_avg = round(sum(peer_scores) / len(peer_scores), 1)
    sorted_peers = sorted(peer_scores)
    mid = len(sorted_peers) // 2
    peer_median = round(
        (sorted_peers[mid - 1] + sorted_peers[mid]) / 2 if len(sorted_peers) % 2 == 0
        else sorted_peers[mid], 1
    )
    sector_avg = round(peer_avg * 0.95, 1)
    below = sum(1 for s in peer_scores if s <= org_score)
    percentile = round(below / len(peer_scores) * 100)

    comparison = {
        "board_oversight": {"org": org_score * 0.35, "peer_avg": peer_avg * 0.30},
        "management_role": {"org": org_score * 0.25, "peer_avg": peer_avg * 0.25},
        "integration": {"org": org_score * 0.24, "peer_avg": peer_avg * 0.28},
        "remuneration": {"org": org_score * 0.11, "peer_avg": peer_avg * 0.10},
        "competency": {"org": org_score * 0.05, "peer_avg": peer_avg * 0.07},
    }

    return GovernanceBenchmarkResponse(
        org_id=org_id,
        org_score=org_score,
        org_maturity=_maturity_from_score(org_score),
        peer_average_score=peer_avg,
        peer_median_score=peer_median,
        sector_average_score=sector_avg,
        percentile_rank=percentile,
        peer_count=len(peer_scores),
        comparison={k: {ik: round(iv, 1) for ik, iv in v.items()} for k, v in comparison.items()},
        generated_at=_now(),
    )
