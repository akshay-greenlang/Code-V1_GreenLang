"""
GL-Taxonomy-APP Minimum Safeguards API

Assesses compliance with the EU Taxonomy minimum safeguards (Article 18)
across four mandatory topics derived from the OECD Guidelines for
Multinational Enterprises and the UN Guiding Principles on Business
and Human Rights.

Minimum Safeguards is Step 4 of the 4-step alignment test:
    Step 1: Eligibility Screening
    Step 2: Substantial Contribution (SC)
    Step 3: Do No Significant Harm (DNSH)
    Step 4: Minimum Safeguards (MS)  <-- this router

Four Safeguard Topics:
    1. Human Rights (UNGP, ILO core conventions)
    2. Anti-Corruption (OECD Anti-Bribery Convention)
    3. Taxation (EU tax governance / BEPS)
    4. Fair Competition (EU competition law)

Assessment Dimensions:
    - Procedural: Policies, due diligence processes in place
    - Outcome: No adverse findings (convictions, sanctions, violations)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/safeguards", tags=["Minimum Safeguards"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SafeguardTopic(str, Enum):
    """Minimum safeguard topics per Article 18."""
    HUMAN_RIGHTS = "human_rights"
    ANTI_CORRUPTION = "anti_corruption"
    TAXATION = "taxation"
    FAIR_COMPETITION = "fair_competition"


class SafeguardStatus(str, Enum):
    """Safeguard assessment status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_DATA = "insufficient_data"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class FullSafeguardRequest(BaseModel):
    """Full 4-topic safeguard assessment."""
    org_id: str = Field(...)
    human_rights_policy: bool = Field(False, description="UNGP-aligned HR policy exists")
    human_rights_due_diligence: bool = Field(False, description="HR due diligence process active")
    grievance_mechanism: bool = Field(False, description="Accessible grievance mechanism exists")
    anti_corruption_policy: bool = Field(False, description="Anti-bribery/corruption policy exists")
    anti_corruption_training: bool = Field(False, description="Employee anti-corruption training")
    whistleblower_channel: bool = Field(False, description="Whistleblower reporting channel")
    tax_governance_policy: bool = Field(False, description="Tax governance/strategy in place")
    country_by_country_reporting: bool = Field(False, description="CBCR compliance")
    no_aggressive_tax_planning: bool = Field(False, description="No aggressive tax planning")
    competition_compliance: bool = Field(False, description="Competition law compliance program")
    no_cartel_involvement: bool = Field(False, description="No cartel/anti-competitive activity")
    evidence_notes: Optional[str] = Field(None, max_length=5000)

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_001",
                "human_rights_policy": True,
                "human_rights_due_diligence": True,
                "grievance_mechanism": True,
                "anti_corruption_policy": True,
                "anti_corruption_training": True,
                "whistleblower_channel": True,
                "tax_governance_policy": True,
                "country_by_country_reporting": True,
                "no_aggressive_tax_planning": True,
                "competition_compliance": True,
                "no_cartel_involvement": True,
            }
        }


class SingleTopicRequest(BaseModel):
    """Single topic safeguard assessment."""
    org_id: str = Field(...)
    checks: Dict[str, bool] = Field(
        ..., description="Topic-specific compliance checks",
    )
    evidence: Optional[str] = Field(None, max_length=5000)


class ProceduralCheckRequest(BaseModel):
    """Run procedural safeguard checks (policies and processes)."""
    has_hr_policy: bool = Field(False)
    has_hr_due_diligence: bool = Field(False)
    has_grievance_mechanism: bool = Field(False)
    has_anti_corruption_policy: bool = Field(False)
    has_whistleblower_channel: bool = Field(False)
    has_tax_governance: bool = Field(False)
    has_competition_program: bool = Field(False)


class OutcomeCheckRequest(BaseModel):
    """Run outcome safeguard checks (no adverse findings)."""
    no_hr_violations: bool = Field(True)
    no_ilo_violations: bool = Field(True)
    no_bribery_convictions: bool = Field(True)
    no_tax_evasion: bool = Field(True)
    no_competition_sanctions: bool = Field(True)
    no_eu_sanctions_list: bool = Field(True)


class AdverseFindingRequest(BaseModel):
    """Record an adverse finding."""
    topic: SafeguardTopic = Field(...)
    finding_type: str = Field(..., description="conviction, sanction, violation, investigation")
    description: str = Field(..., max_length=5000)
    date_identified: str = Field(..., description="ISO date")
    severity: str = Field("medium", description="low, medium, high, critical")
    remediation_plan: Optional[str] = Field(None, max_length=5000)
    resolved: bool = Field(False)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SafeguardAssessmentResponse(BaseModel):
    """Full safeguard assessment result."""
    assessment_id: str
    org_id: str
    overall_status: str
    topic_results: Dict[str, Dict[str, Any]]
    procedural_score_pct: float
    outcome_score_pct: float
    compliant_topics: int
    non_compliant_topics: int
    total_topics: int
    recommendations: List[str]
    assessed_at: datetime


class SingleTopicResponse(BaseModel):
    """Single topic assessment result."""
    assessment_id: str
    org_id: str
    topic: str
    topic_status: str
    checks: List[Dict[str, Any]]
    checks_passed: int
    checks_total: int
    score_pct: float
    requirements: List[str]
    assessed_at: datetime


class ProceduralCheckResponse(BaseModel):
    """Procedural check result."""
    org_id: str
    procedural_score_pct: float
    checks: List[Dict[str, Any]]
    passed: int
    total: int
    missing_policies: List[str]
    assessed_at: datetime


class OutcomeCheckResponse(BaseModel):
    """Outcome check result."""
    org_id: str
    outcome_score_pct: float
    checks: List[Dict[str, Any]]
    passed: int
    total: int
    adverse_findings: List[str]
    assessed_at: datetime


class AdverseFindingResponse(BaseModel):
    """Adverse finding record."""
    finding_id: str
    org_id: str
    topic: str
    finding_type: str
    description: str
    severity: str
    date_identified: str
    remediation_plan: Optional[str]
    resolved: bool
    recorded_at: datetime


class SafeguardSummaryResponse(BaseModel):
    """Safeguard summary for organization."""
    org_id: str
    overall_status: str
    topic_statuses: Dict[str, str]
    procedural_score_pct: float
    outcome_score_pct: float
    adverse_findings_count: int
    unresolved_findings: int
    last_assessed: Optional[datetime]
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_safeguard_assessments: Dict[str, Dict[str, Any]] = {}
_adverse_findings: Dict[str, List[Dict[str, Any]]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/assess",
    response_model=SafeguardAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Full 4-topic assessment",
    description=(
        "Run a comprehensive minimum safeguards assessment across all four "
        "topics: human rights, anti-corruption, taxation, and fair competition."
    ),
)
async def full_assessment(request: FullSafeguardRequest) -> SafeguardAssessmentResponse:
    """Full 4-topic safeguard assessment."""
    assessment_id = _generate_id("msg")
    recommendations = []

    # Human rights
    hr_checks = [request.human_rights_policy, request.human_rights_due_diligence, request.grievance_mechanism]
    hr_passed = sum(hr_checks)
    hr_status = "compliant" if hr_passed == 3 else ("non_compliant" if hr_passed < 2 else "non_compliant")
    if not request.human_rights_policy:
        recommendations.append("Adopt a UNGP-aligned human rights policy")
    if not request.human_rights_due_diligence:
        recommendations.append("Implement human rights due diligence process per UNGP")
    if not request.grievance_mechanism:
        recommendations.append("Establish accessible grievance mechanism")

    # Anti-corruption
    ac_checks = [request.anti_corruption_policy, request.anti_corruption_training, request.whistleblower_channel]
    ac_passed = sum(ac_checks)
    ac_status = "compliant" if ac_passed == 3 else "non_compliant"
    if not request.anti_corruption_policy:
        recommendations.append("Adopt anti-bribery/corruption policy per OECD Convention")
    if not request.whistleblower_channel:
        recommendations.append("Establish confidential whistleblower reporting channel")

    # Taxation
    tax_checks = [request.tax_governance_policy, request.country_by_country_reporting, request.no_aggressive_tax_planning]
    tax_passed = sum(tax_checks)
    tax_status = "compliant" if tax_passed == 3 else "non_compliant"
    if not request.tax_governance_policy:
        recommendations.append("Publish tax governance strategy")
    if not request.country_by_country_reporting:
        recommendations.append("Implement country-by-country reporting per BEPS Action 13")

    # Fair competition
    comp_checks = [request.competition_compliance, request.no_cartel_involvement]
    comp_passed = sum(comp_checks)
    comp_status = "compliant" if comp_passed == 2 else "non_compliant"
    if not request.competition_compliance:
        recommendations.append("Implement competition law compliance program")

    topic_results = {
        "human_rights": {"status": hr_status, "passed": hr_passed, "total": 3, "score_pct": round(hr_passed / 3 * 100, 1)},
        "anti_corruption": {"status": ac_status, "passed": ac_passed, "total": 3, "score_pct": round(ac_passed / 3 * 100, 1)},
        "taxation": {"status": tax_status, "passed": tax_passed, "total": 3, "score_pct": round(tax_passed / 3 * 100, 1)},
        "fair_competition": {"status": comp_status, "passed": comp_passed, "total": 2, "score_pct": round(comp_passed / 2 * 100, 1)},
    }

    compliant_topics = sum(1 for t in topic_results.values() if t["status"] == "compliant")
    total_procedural = sum(hr_checks + ac_checks + tax_checks + comp_checks)
    total_checks = 11
    procedural_pct = round(total_procedural / total_checks * 100, 1)

    overall = "compliant" if compliant_topics == 4 else "non_compliant"

    data = {
        "assessment_id": assessment_id,
        "org_id": request.org_id,
        "overall_status": overall,
        "topic_results": topic_results,
        "procedural_score_pct": procedural_pct,
        "outcome_score_pct": 100.0,
        "compliant_topics": compliant_topics,
        "non_compliant_topics": 4 - compliant_topics,
        "total_topics": 4,
        "recommendations": recommendations if recommendations else ["All minimum safeguard requirements met"],
        "assessed_at": _now(),
    }
    _safeguard_assessments[assessment_id] = data
    return SafeguardAssessmentResponse(**data)


@router.post(
    "/assess/{topic}",
    response_model=SingleTopicResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assess single topic",
    description="Assess a single safeguard topic (human_rights, anti_corruption, taxation, fair_competition).",
)
async def assess_single_topic(
    topic: str,
    request: SingleTopicRequest,
) -> SingleTopicResponse:
    """Assess single safeguard topic."""
    valid_topics = [t.value for t in SafeguardTopic]
    if topic not in valid_topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid topic '{topic}'. Valid: {valid_topics}",
        )

    assessment_id = _generate_id("msg_s")
    checks_list = []
    requirements = []

    topic_criteria = {
        "human_rights": [
            ("hr_policy", "UNGP-aligned human rights policy"),
            ("hr_due_diligence", "Human rights due diligence process"),
            ("grievance_mechanism", "Accessible grievance mechanism"),
            ("ilo_conventions", "ILO core conventions compliance"),
        ],
        "anti_corruption": [
            ("anti_corruption_policy", "Anti-bribery/corruption policy"),
            ("training", "Employee anti-corruption training"),
            ("whistleblower", "Whistleblower reporting channel"),
            ("risk_assessment", "Corruption risk assessment"),
        ],
        "taxation": [
            ("tax_governance", "Tax governance strategy published"),
            ("cbcr", "Country-by-country reporting"),
            ("no_aggressive_planning", "No aggressive tax planning"),
            ("transfer_pricing", "Arm's-length transfer pricing"),
        ],
        "fair_competition": [
            ("compliance_program", "Competition law compliance program"),
            ("no_cartel", "No cartel or anti-competitive activity"),
            ("fair_dealing", "Fair dealing with suppliers and customers"),
        ],
    }

    criteria = topic_criteria.get(topic, [])
    passed = 0
    for key, label in criteria:
        met = request.checks.get(key, False)
        checks_list.append({"check": label, "key": key, "met": met})
        if met:
            passed += 1
        else:
            requirements.append(f"Address: {label}")

    total = len(criteria)
    score = round(passed / total * 100, 1) if total > 0 else 0
    topic_status = "compliant" if passed == total else "non_compliant"

    return SingleTopicResponse(
        assessment_id=assessment_id,
        org_id=request.org_id,
        topic=topic,
        topic_status=topic_status,
        checks=checks_list,
        checks_passed=passed,
        checks_total=total,
        score_pct=score,
        requirements=requirements if requirements else [f"All {topic} safeguard requirements met"],
        assessed_at=_now(),
    )


@router.post(
    "/{org_id}/procedural",
    response_model=ProceduralCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run procedural checks",
    description="Run procedural safeguard checks (policies and due diligence processes).",
)
async def procedural_checks(
    org_id: str,
    request: ProceduralCheckRequest,
) -> ProceduralCheckResponse:
    """Run procedural safeguard checks."""
    checks = [
        {"check": "Human rights policy", "met": request.has_hr_policy},
        {"check": "HR due diligence process", "met": request.has_hr_due_diligence},
        {"check": "Grievance mechanism", "met": request.has_grievance_mechanism},
        {"check": "Anti-corruption policy", "met": request.has_anti_corruption_policy},
        {"check": "Whistleblower channel", "met": request.has_whistleblower_channel},
        {"check": "Tax governance policy", "met": request.has_tax_governance},
        {"check": "Competition compliance program", "met": request.has_competition_program},
    ]

    passed = sum(1 for c in checks if c["met"])
    total = len(checks)
    missing = [c["check"] for c in checks if not c["met"]]

    return ProceduralCheckResponse(
        org_id=org_id,
        procedural_score_pct=round(passed / total * 100, 1),
        checks=checks,
        passed=passed,
        total=total,
        missing_policies=missing,
        assessed_at=_now(),
    )


@router.post(
    "/{org_id}/outcome",
    response_model=OutcomeCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run outcome checks",
    description="Run outcome safeguard checks (no convictions, sanctions, or violations).",
)
async def outcome_checks(
    org_id: str,
    request: OutcomeCheckRequest,
) -> OutcomeCheckResponse:
    """Run outcome safeguard checks."""
    checks = [
        {"check": "No human rights violations", "met": request.no_hr_violations},
        {"check": "No ILO convention violations", "met": request.no_ilo_violations},
        {"check": "No bribery convictions", "met": request.no_bribery_convictions},
        {"check": "No tax evasion findings", "met": request.no_tax_evasion},
        {"check": "No competition sanctions", "met": request.no_competition_sanctions},
        {"check": "Not on EU sanctions list", "met": request.no_eu_sanctions_list},
    ]

    passed = sum(1 for c in checks if c["met"])
    total = len(checks)
    adverse = [c["check"].replace("No ", "") for c in checks if not c["met"]]

    return OutcomeCheckResponse(
        org_id=org_id,
        outcome_score_pct=round(passed / total * 100, 1),
        checks=checks,
        passed=passed,
        total=total,
        adverse_findings=adverse,
        assessed_at=_now(),
    )


@router.post(
    "/{org_id}/adverse-finding",
    response_model=AdverseFindingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record adverse finding",
    description="Record an adverse finding that may disqualify the organization from alignment.",
)
async def record_adverse_finding(
    org_id: str,
    request: AdverseFindingRequest,
) -> AdverseFindingResponse:
    """Record adverse finding."""
    finding_id = _generate_id("af")
    entry = {
        "finding_id": finding_id,
        "org_id": org_id,
        "topic": request.topic.value,
        "finding_type": request.finding_type,
        "description": request.description,
        "severity": request.severity,
        "date_identified": request.date_identified,
        "remediation_plan": request.remediation_plan,
        "resolved": request.resolved,
        "recorded_at": _now(),
    }

    if org_id not in _adverse_findings:
        _adverse_findings[org_id] = []
    _adverse_findings[org_id].append(entry)

    return AdverseFindingResponse(**entry)


@router.get(
    "/{org_id}/results",
    response_model=List[SafeguardAssessmentResponse],
    summary="Get safeguard results",
    description="Retrieve all safeguard assessment results for an organization.",
)
async def get_results(
    org_id: str,
    limit: int = Query(20, ge=1, le=100),
) -> List[SafeguardAssessmentResponse]:
    """Get safeguard results."""
    results = [a for a in _safeguard_assessments.values() if a["org_id"] == org_id]
    results.sort(key=lambda a: a["assessed_at"], reverse=True)
    return [SafeguardAssessmentResponse(**a) for a in results[:limit]]


@router.get(
    "/{org_id}/summary",
    response_model=SafeguardSummaryResponse,
    summary="Get safeguard summary",
    description="Get aggregated safeguard assessment summary for an organization.",
)
async def get_summary(org_id: str) -> SafeguardSummaryResponse:
    """Get safeguard summary."""
    org_assessments = [a for a in _safeguard_assessments.values() if a["org_id"] == org_id]
    findings = _adverse_findings.get(org_id, [])

    if org_assessments:
        latest = max(org_assessments, key=lambda a: a["assessed_at"])
        overall = latest["overall_status"]
        topic_statuses = {t: d["status"] for t, d in latest["topic_results"].items()}
        proc_score = latest["procedural_score_pct"]
        out_score = latest["outcome_score_pct"]
        last_assessed = latest["assessed_at"]
    else:
        overall = "pending"
        topic_statuses = {t.value: "pending" for t in SafeguardTopic}
        proc_score = 0
        out_score = 0
        last_assessed = None

    unresolved = sum(1 for f in findings if not f["resolved"])

    return SafeguardSummaryResponse(
        org_id=org_id,
        overall_status=overall,
        topic_statuses=topic_statuses,
        procedural_score_pct=proc_score,
        outcome_score_pct=out_score,
        adverse_findings_count=len(findings),
        unresolved_findings=unresolved,
        last_assessed=last_assessed,
        generated_at=_now(),
    )
