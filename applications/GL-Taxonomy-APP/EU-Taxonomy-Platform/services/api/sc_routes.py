"""
GL-Taxonomy-APP Substantial Contribution API

Assesses whether taxonomy-eligible activities make a substantial
contribution (SC) to one or more of the six EU Taxonomy environmental
objectives by evaluating activity-specific Technical Screening Criteria
(TSC) from the Delegated Acts.

Substantial Contribution is Step 2 of the 4-step alignment test:
    Step 1: Eligibility Screening
    Step 2: Substantial Contribution (SC)  <-- this router
    Step 3: Do No Significant Harm (DNSH)
    Step 4: Minimum Safeguards (MS)

TSC are activity-specific and quantitative where possible.  Examples:
    - Solar PV: Life-cycle GHG < 100 gCO2e/kWh
    - New buildings: Primary energy demand >= 10% below NZEB
    - Iron & steel: GHG intensity < EU ETS benchmark
    - Vehicles: Direct CO2 emissions = 0 g/km (or < 50 g/km transitional)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/substantial-contribution", tags=["Substantial Contribution"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SCStatus(str, Enum):
    """Substantial contribution assessment status."""
    MEETS_TSC = "meets_tsc"
    DOES_NOT_MEET = "does_not_meet"
    INSUFFICIENT_DATA = "insufficient_data"
    PENDING = "pending"


class SCObjective(str, Enum):
    """Environmental objective for SC assessment."""
    CCM = "climate_change_mitigation"
    CCA = "climate_change_adaptation"
    WTR = "water"
    CE = "circular_economy"
    PPC = "pollution_prevention"
    BIO = "biodiversity"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class SCAssessRequest(BaseModel):
    """Request to assess substantial contribution."""
    org_id: str = Field(...)
    activity_code: str = Field(..., description="Taxonomy activity code (e.g. 4.1)")
    objective: str = Field(..., description="Environmental objective for SC")
    reported_value: Optional[float] = Field(None, description="Reported metric value for quantitative TSC")
    reported_unit: Optional[str] = Field(None, description="Unit of reported value")
    evidence_description: Optional[str] = Field(None, max_length=5000)
    reporting_year: int = Field(2025, ge=2022, le=2030)

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_001",
                "activity_code": "4.1",
                "objective": "climate_change_mitigation",
                "reported_value": 35.0,
                "reported_unit": "gCO2e/kWh",
                "reporting_year": 2025,
            }
        }


class BatchSCRequest(BaseModel):
    """Batch SC assessment request."""
    org_id: str = Field(...)
    assessments: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=200,
        description="List of {activity_code, objective, reported_value, reported_unit}",
    )
    reporting_year: int = Field(2025, ge=2022, le=2030)


class ThresholdCheckRequest(BaseModel):
    """Check a quantitative threshold."""
    activity_code: str = Field(...)
    objective: str = Field(...)
    reported_value: float = Field(...)
    reported_unit: str = Field(...)


class EvidenceRequest(BaseModel):
    """Record evidence for SC assessment."""
    evidence_type: str = Field(..., description="certificate, measurement, calculation, third_party_audit")
    description: str = Field(..., max_length=5000)
    source: Optional[str] = Field(None, max_length=500)
    document_ref: Optional[str] = Field(None, max_length=300)
    verified: bool = Field(False)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SCAssessmentResponse(BaseModel):
    """Substantial contribution assessment result."""
    assessment_id: str
    org_id: str
    activity_code: str
    activity_name: str
    objective: str
    sc_status: str
    tsc_criteria: List[Dict[str, Any]]
    reported_value: Optional[float]
    threshold_value: Optional[float]
    threshold_unit: Optional[str]
    threshold_met: Optional[bool]
    is_transitional: bool
    is_enabling: bool
    confidence: float
    evidence_count: int
    assessed_at: datetime


class BatchSCResponse(BaseModel):
    """Batch SC assessment results."""
    org_id: str
    total_assessed: int
    meets_tsc_count: int
    does_not_meet_count: int
    insufficient_data_count: int
    results: List[SCAssessmentResponse]
    assessed_at: datetime


class TSCCriteriaResponse(BaseModel):
    """Technical screening criteria for an activity."""
    activity_code: str
    activity_name: str
    objective: str
    criteria: List[Dict[str, Any]]
    quantitative_threshold: Optional[Dict[str, Any]]
    qualitative_requirements: List[str]
    delegated_act_reference: str
    last_amended: Optional[str]


class SCProfileResponse(BaseModel):
    """SC profile across all 6 objectives."""
    activity_code: str
    activity_name: str
    objectives: Dict[str, Dict[str, Any]]
    primary_objective: str
    sc_objectives_count: int
    generated_at: datetime


class ThresholdCheckResponse(BaseModel):
    """Quantitative threshold check result."""
    activity_code: str
    objective: str
    reported_value: float
    threshold_value: float
    threshold_unit: str
    threshold_met: bool
    margin_pct: float
    assessment: str


class EvidenceResponse(BaseModel):
    """Evidence recording result."""
    evidence_id: str
    assessment_id: str
    evidence_type: str
    description: str
    source: Optional[str]
    document_ref: Optional[str]
    verified: bool
    recorded_at: datetime


class SCSummaryResponse(BaseModel):
    """SC assessment summary for organization."""
    org_id: str
    total_assessments: int
    meets_tsc: int
    does_not_meet: int
    insufficient_data: int
    sc_rate_pct: float
    by_objective: Dict[str, Dict[str, int]]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reference Data -- TSC Thresholds
# ---------------------------------------------------------------------------

TSC_THRESHOLDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "4.1": {
        "climate_change_mitigation": {
            "activity_name": "Electricity generation using solar PV",
            "threshold_value": 100,
            "threshold_unit": "gCO2e/kWh",
            "operator": "lt",
            "description": "Life-cycle GHG emissions < 100 gCO2e/kWh",
            "da_ref": "CDA Annex I, Section 4.1",
        }
    },
    "4.3": {
        "climate_change_mitigation": {
            "activity_name": "Electricity generation from wind power",
            "threshold_value": 100,
            "threshold_unit": "gCO2e/kWh",
            "operator": "lt",
            "description": "Life-cycle GHG emissions < 100 gCO2e/kWh",
            "da_ref": "CDA Annex I, Section 4.3",
        }
    },
    "4.29": {
        "climate_change_mitigation": {
            "activity_name": "Electricity generation from fossil gaseous fuels",
            "threshold_value": 270,
            "threshold_unit": "gCO2e/kWh",
            "operator": "lt",
            "description": "Direct GHG emissions < 270 gCO2e/kWh or 550 kgCO2e/kW annual average (transitional)",
            "da_ref": "Complementary CDA, Section 4.29",
        }
    },
    "3.9": {
        "climate_change_mitigation": {
            "activity_name": "Manufacture of iron and steel",
            "threshold_value": 1.331,
            "threshold_unit": "tCO2e/tonne_product",
            "operator": "lt",
            "description": "GHG emissions intensity below EU ETS benchmark",
            "da_ref": "CDA Annex I, Section 3.9",
        }
    },
    "3.12": {
        "climate_change_mitigation": {
            "activity_name": "Manufacture of cement",
            "threshold_value": 0.469,
            "threshold_unit": "tCO2e/tonne_clinker",
            "operator": "lt",
            "description": "GHG emissions below EU ETS clinker benchmark",
            "da_ref": "CDA Annex I, Section 3.12",
        }
    },
    "6.5": {
        "climate_change_mitigation": {
            "activity_name": "Transport by passenger cars",
            "threshold_value": 50,
            "threshold_unit": "gCO2/km",
            "operator": "lt",
            "description": "Direct CO2 emissions < 50 gCO2/km (until 2025: 0 gCO2/km preferred)",
            "da_ref": "CDA Annex I, Section 6.5",
        }
    },
    "7.1": {
        "climate_change_mitigation": {
            "activity_name": "Construction of new buildings",
            "threshold_value": 10,
            "threshold_unit": "pct_below_nzeb",
            "operator": "gte",
            "description": "Primary energy demand at least 10% below NZEB threshold",
            "da_ref": "CDA Annex I, Section 7.1",
        }
    },
    "7.2": {
        "climate_change_mitigation": {
            "activity_name": "Renovation of existing buildings",
            "threshold_value": 30,
            "threshold_unit": "pct_reduction",
            "operator": "gte",
            "description": "Achieves 30% reduction in primary energy demand",
            "da_ref": "CDA Annex I, Section 7.2",
        }
    },
    "7.7": {
        "climate_change_mitigation": {
            "activity_name": "Acquisition and ownership of buildings",
            "threshold_value": 15,
            "threshold_unit": "pct_top_energy_performance",
            "operator": "within",
            "description": "Building within top 15% of national/regional stock or EPC A",
            "da_ref": "CDA Annex I, Section 7.7",
        }
    },
}


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_assessments: Dict[str, Dict[str, Any]] = {}
_evidence: Dict[str, List[Dict[str, Any]]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


def _check_threshold(activity_code: str, objective: str, value: float) -> Dict[str, Any]:
    """Check a reported value against TSC threshold."""
    tsc = TSC_THRESHOLDS.get(activity_code, {}).get(objective)
    if not tsc:
        return {"met": None, "threshold": None, "unit": None, "margin": 0}

    threshold = tsc["threshold_value"]
    operator = tsc["operator"]

    if operator == "lt":
        met = value < threshold
    elif operator == "gte":
        met = value >= threshold
    elif operator == "within":
        met = value <= threshold
    else:
        met = value <= threshold

    margin = round(((threshold - value) / threshold) * 100, 1) if threshold != 0 else 0

    return {
        "met": met,
        "threshold": threshold,
        "unit": tsc["threshold_unit"],
        "margin": margin,
    }


def _get_activity_name(activity_code: str) -> str:
    """Get activity name from TSC data."""
    for obj_data in TSC_THRESHOLDS.get(activity_code, {}).values():
        return obj_data.get("activity_name", f"Activity {activity_code}")
    return f"Activity {activity_code}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/assess",
    response_model=SCAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assess substantial contribution",
    description=(
        "Assess whether an activity makes a substantial contribution to "
        "the specified environmental objective by evaluating its reported "
        "metrics against the TSC criteria from the Delegated Acts."
    ),
)
async def assess_sc(request: SCAssessRequest) -> SCAssessmentResponse:
    """Assess substantial contribution for activity + objective."""
    assessment_id = _generate_id("sc")
    tsc_data = TSC_THRESHOLDS.get(request.activity_code, {}).get(request.objective)
    activity_name = _get_activity_name(request.activity_code)

    if not tsc_data:
        sc_status = SCStatus.INSUFFICIENT_DATA.value
        criteria = [{"criterion": "TSC not found", "detail": f"No TSC defined for {request.activity_code}/{request.objective}"}]
        threshold_value = None
        threshold_unit = None
        threshold_met = None
        confidence = 0.3
    elif request.reported_value is not None:
        check = _check_threshold(request.activity_code, request.objective, request.reported_value)
        threshold_met = check["met"]
        threshold_value = check["threshold"]
        threshold_unit = check["unit"]
        sc_status = SCStatus.MEETS_TSC.value if threshold_met else SCStatus.DOES_NOT_MEET.value
        criteria = [{"criterion": tsc_data["description"], "threshold": threshold_value, "unit": threshold_unit, "reported": request.reported_value, "met": threshold_met}]
        confidence = 0.90 if threshold_met is not None else 0.5
    else:
        sc_status = SCStatus.INSUFFICIENT_DATA.value
        criteria = [{"criterion": tsc_data["description"], "detail": "No reported value provided"}]
        threshold_value = tsc_data["threshold_value"]
        threshold_unit = tsc_data["threshold_unit"]
        threshold_met = None
        confidence = 0.4

    transitional_codes = {"3.9", "3.12", "3.14", "4.15", "4.29", "5.5", "6.5", "6.6", "6.10"}
    enabling_codes = {"3.1", "3.2", "3.3", "3.5", "3.6", "6.14", "6.15", "7.3", "7.4", "7.5", "7.6", "8.1", "8.2", "10.1", "10.2"}

    data = {
        "assessment_id": assessment_id,
        "org_id": request.org_id,
        "activity_code": request.activity_code,
        "activity_name": activity_name,
        "objective": request.objective,
        "sc_status": sc_status,
        "tsc_criteria": criteria,
        "reported_value": request.reported_value,
        "threshold_value": threshold_value,
        "threshold_unit": threshold_unit,
        "threshold_met": threshold_met,
        "is_transitional": request.activity_code in transitional_codes,
        "is_enabling": request.activity_code in enabling_codes,
        "confidence": confidence,
        "evidence_count": 0,
        "assessed_at": _now(),
    }
    _assessments[assessment_id] = data
    return SCAssessmentResponse(**data)


@router.post(
    "/batch",
    response_model=BatchSCResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch SC assessment",
    description="Assess substantial contribution for multiple activities in one request.",
)
async def batch_assess(request: BatchSCRequest) -> BatchSCResponse:
    """Batch SC assessment."""
    results = []
    meets = 0
    does_not = 0
    insufficient = 0

    for item in request.assessments:
        sub_req = SCAssessRequest(
            org_id=request.org_id,
            activity_code=item.get("activity_code", ""),
            objective=item.get("objective", "climate_change_mitigation"),
            reported_value=item.get("reported_value"),
            reported_unit=item.get("reported_unit"),
            reporting_year=request.reporting_year,
        )
        result = await assess_sc(sub_req)
        results.append(result)

        if result.sc_status == "meets_tsc":
            meets += 1
        elif result.sc_status == "does_not_meet":
            does_not += 1
        else:
            insufficient += 1

    return BatchSCResponse(
        org_id=request.org_id,
        total_assessed=len(results),
        meets_tsc_count=meets,
        does_not_meet_count=does_not,
        insufficient_data_count=insufficient,
        results=results,
        assessed_at=_now(),
    )


@router.get(
    "/{org_id}/results",
    response_model=List[SCAssessmentResponse],
    summary="Get SC results",
    description="Retrieve all SC assessment results for an organization.",
)
async def get_results(
    org_id: str,
    sc_status: Optional[str] = Query(None, description="Filter by SC status"),
    objective: Optional[str] = Query(None, description="Filter by objective"),
    limit: int = Query(50, ge=1, le=200),
) -> List[SCAssessmentResponse]:
    """Get SC assessment results for organization."""
    results = [a for a in _assessments.values() if a["org_id"] == org_id]
    if sc_status:
        results = [a for a in results if a["sc_status"] == sc_status]
    if objective:
        results = [a for a in results if a["objective"] == objective]
    results.sort(key=lambda a: a["assessed_at"], reverse=True)
    return [SCAssessmentResponse(**a) for a in results[:limit]]


@router.get(
    "/{activity_code}/criteria",
    response_model=TSCCriteriaResponse,
    summary="Get TSC criteria for activity",
    description=(
        "Retrieve the Technical Screening Criteria for a specific activity "
        "and environmental objective from the Delegated Acts."
    ),
)
async def get_tsc_criteria(
    activity_code: str,
    objective: str = Query("climate_change_mitigation", description="Environmental objective"),
) -> TSCCriteriaResponse:
    """Get TSC criteria for an activity."""
    tsc = TSC_THRESHOLDS.get(activity_code, {}).get(objective)
    activity_name = _get_activity_name(activity_code)

    if tsc:
        criteria = [{"criterion": tsc["description"], "threshold": tsc["threshold_value"], "unit": tsc["threshold_unit"], "operator": tsc["operator"]}]
        quantitative = {"value": tsc["threshold_value"], "unit": tsc["threshold_unit"], "operator": tsc["operator"]}
        qualitative = [f"Comply with {tsc['da_ref']} requirements"]
        da_ref = tsc["da_ref"]
    else:
        criteria = [{"criterion": "No specific quantitative TSC defined", "detail": "Qualitative assessment required"}]
        quantitative = None
        qualitative = ["Activity must demonstrate substantial contribution through qualitative evidence"]
        da_ref = f"CDA Annex I, Section {activity_code}"

    return TSCCriteriaResponse(
        activity_code=activity_code,
        activity_name=activity_name,
        objective=objective,
        criteria=criteria,
        quantitative_threshold=quantitative,
        qualitative_requirements=qualitative,
        delegated_act_reference=da_ref,
        last_amended="2023-06-27",
    )


@router.get(
    "/{activity_code}/profile",
    response_model=SCProfileResponse,
    summary="Get SC profile (all objectives)",
    description="Get the SC profile for an activity across all 6 environmental objectives.",
)
async def get_sc_profile(activity_code: str) -> SCProfileResponse:
    """Get SC profile across all objectives."""
    activity_name = _get_activity_name(activity_code)
    all_objectives = ["climate_change_mitigation", "climate_change_adaptation", "water", "circular_economy", "pollution_prevention", "biodiversity"]

    objectives_data: Dict[str, Dict[str, Any]] = {}
    sc_count = 0
    primary = None

    for obj in all_objectives:
        tsc = TSC_THRESHOLDS.get(activity_code, {}).get(obj)
        if tsc:
            objectives_data[obj] = {
                "sc_possible": True,
                "has_quantitative_tsc": True,
                "threshold": tsc["threshold_value"],
                "unit": tsc["threshold_unit"],
                "da_ref": tsc["da_ref"],
            }
            sc_count += 1
            if primary is None:
                primary = obj
        else:
            objectives_data[obj] = {
                "sc_possible": False,
                "has_quantitative_tsc": False,
                "threshold": None,
                "unit": None,
                "da_ref": None,
            }

    return SCProfileResponse(
        activity_code=activity_code,
        activity_name=activity_name,
        objectives=objectives_data,
        primary_objective=primary or "climate_change_mitigation",
        sc_objectives_count=sc_count,
        generated_at=_now(),
    )


@router.post(
    "/threshold-check",
    response_model=ThresholdCheckResponse,
    summary="Check quantitative threshold",
    description="Check a reported value against a TSC quantitative threshold.",
)
async def threshold_check(request: ThresholdCheckRequest) -> ThresholdCheckResponse:
    """Check quantitative TSC threshold."""
    check = _check_threshold(request.activity_code, request.objective, request.reported_value)

    if check["threshold"] is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No quantitative TSC found for {request.activity_code}/{request.objective}.",
        )

    if check["met"]:
        assessment = f"PASS: Reported value ({request.reported_value} {request.reported_unit}) meets the threshold ({check['threshold']} {check['unit']}) with {abs(check['margin'])}% margin."
    else:
        assessment = f"FAIL: Reported value ({request.reported_value} {request.reported_unit}) exceeds the threshold ({check['threshold']} {check['unit']}) by {abs(check['margin'])}%."

    return ThresholdCheckResponse(
        activity_code=request.activity_code,
        objective=request.objective,
        reported_value=request.reported_value,
        threshold_value=check["threshold"],
        threshold_unit=check["unit"],
        threshold_met=check["met"],
        margin_pct=check["margin"],
        assessment=assessment,
    )


@router.post(
    "/{assessment_id}/evidence",
    response_model=EvidenceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record evidence",
    description="Record supporting evidence for an SC assessment.",
)
async def record_evidence(
    assessment_id: str,
    request: EvidenceRequest,
) -> EvidenceResponse:
    """Record evidence for SC assessment."""
    if assessment_id not in _assessments:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found.",
        )

    evidence_id = _generate_id("evi")
    entry = {
        "evidence_id": evidence_id,
        "assessment_id": assessment_id,
        "evidence_type": request.evidence_type,
        "description": request.description,
        "source": request.source,
        "document_ref": request.document_ref,
        "verified": request.verified,
        "recorded_at": _now(),
    }

    if assessment_id not in _evidence:
        _evidence[assessment_id] = []
    _evidence[assessment_id].append(entry)

    # Update evidence count
    _assessments[assessment_id]["evidence_count"] = len(_evidence[assessment_id])

    return EvidenceResponse(**entry)


@router.get(
    "/{org_id}/summary",
    response_model=SCSummaryResponse,
    summary="SC summary",
    description="Get SC assessment summary for an organization.",
)
async def get_sc_summary(org_id: str) -> SCSummaryResponse:
    """Get SC assessment summary."""
    org_assessments = [a for a in _assessments.values() if a["org_id"] == org_id]

    meets = sum(1 for a in org_assessments if a["sc_status"] == "meets_tsc")
    does_not = sum(1 for a in org_assessments if a["sc_status"] == "does_not_meet")
    insufficient = sum(1 for a in org_assessments if a["sc_status"] == "insufficient_data")
    total = len(org_assessments)
    rate = round((meets / total) * 100, 1) if total > 0 else 0

    by_obj: Dict[str, Dict[str, int]] = {}
    for a in org_assessments:
        obj = a["objective"]
        if obj not in by_obj:
            by_obj[obj] = {"meets_tsc": 0, "does_not_meet": 0, "insufficient_data": 0}
        if a["sc_status"] == "meets_tsc":
            by_obj[obj]["meets_tsc"] += 1
        elif a["sc_status"] == "does_not_meet":
            by_obj[obj]["does_not_meet"] += 1
        else:
            by_obj[obj]["insufficient_data"] += 1

    return SCSummaryResponse(
        org_id=org_id,
        total_assessments=total,
        meets_tsc=meets,
        does_not_meet=does_not,
        insufficient_data=insufficient,
        sc_rate_pct=rate,
        by_objective=by_obj,
        generated_at=_now(),
    )
