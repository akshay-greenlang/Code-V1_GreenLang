"""
GL-Taxonomy-APP Do No Significant Harm (DNSH) API

Assesses whether taxonomy-eligible activities comply with the DNSH
criteria for all five environmental objectives to which the activity
does NOT make a substantial contribution.  Implements activity-specific
DNSH criteria from the CDA/EDA Delegated Act annexes.

DNSH is Step 3 of the 4-step EU Taxonomy alignment test:
    Step 1: Eligibility Screening
    Step 2: Substantial Contribution (SC)
    Step 3: Do No Significant Harm (DNSH)  <-- this router
    Step 4: Minimum Safeguards (MS)

DNSH Objectives (for an activity with CCM as SC):
    - CCA: Climate risk assessment (Appendix A physical hazards)
    - WTR: Water Framework Directive compliance
    - CE:  Waste hierarchy, recyclability, durability
    - PPC: IED BAT / pollution thresholds (RoHS, REACH)
    - BIO: EIA, Natura 2000, no degradation of high-biodiversity areas

Special DNSH criteria:
    - CCA requires a robust climate risk assessment per Appendix A
    - WTR requires Water Use Efficiency and Environmental Quality Standards
    - PPC requires compliance with specific EU pollution Directives
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/dnsh", tags=["DNSH Assessment"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DNSHStatus(str, Enum):
    """DNSH assessment status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


class DNSHObjective(str, Enum):
    """Environmental objective for DNSH."""
    CCA = "climate_change_adaptation"
    CCM = "climate_change_mitigation"
    WTR = "water"
    CE = "circular_economy"
    PPC = "pollution_prevention"
    BIO = "biodiversity"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class FullDNSHRequest(BaseModel):
    """Request for full DNSH assessment (5 non-SC objectives)."""
    org_id: str = Field(...)
    activity_code: str = Field(..., description="Taxonomy activity code")
    sc_objective: str = Field(
        "climate_change_mitigation",
        description="The objective for which SC is claimed (excluded from DNSH)",
    )
    climate_risk_assessed: bool = Field(False, description="Has climate risk assessment been performed")
    water_compliance: Optional[bool] = Field(None)
    waste_hierarchy_applied: Optional[bool] = Field(None)
    pollution_within_limits: Optional[bool] = Field(None)
    biodiversity_eia_done: Optional[bool] = Field(None)
    evidence_notes: Optional[str] = Field(None, max_length=5000)

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_001",
                "activity_code": "4.1",
                "sc_objective": "climate_change_mitigation",
                "climate_risk_assessed": True,
                "water_compliance": True,
                "waste_hierarchy_applied": True,
                "pollution_within_limits": True,
                "biodiversity_eia_done": True,
            }
        }


class SingleDNSHRequest(BaseModel):
    """Request for single-objective DNSH assessment."""
    org_id: str = Field(...)
    activity_code: str = Field(...)
    evidence: Optional[str] = Field(None, max_length=5000)


class ClimateRiskRequest(BaseModel):
    """Climate risk assessment for DNSH CCA."""
    org_id: str = Field(...)
    activity_code: str = Field(...)
    location_lat: Optional[float] = Field(None, description="Latitude")
    location_lon: Optional[float] = Field(None, description="Longitude")
    asset_lifetime_years: int = Field(30, ge=1, le=100)
    physical_hazards_assessed: List[str] = Field(
        default_factory=list,
        description="Appendix A hazards assessed (heat_wave, flood, wildfire, etc.)",
    )
    adaptation_measures: List[str] = Field(default_factory=list)
    rcp_scenario: str = Field("rcp8.5", description="Climate scenario (rcp2.6, rcp4.5, rcp8.5)")


class WaterDNSHRequest(BaseModel):
    """Water DNSH assessment."""
    org_id: str = Field(...)
    activity_code: str = Field(...)
    water_use_efficiency_plan: bool = Field(False)
    eqs_compliance: bool = Field(False, description="Environmental Quality Standards met")
    water_stress_area: bool = Field(False)
    wfd_compliance: bool = Field(False, description="Water Framework Directive compliance")


class CircularEconomyDNSHRequest(BaseModel):
    """Circular economy DNSH assessment."""
    org_id: str = Field(...)
    activity_code: str = Field(...)
    waste_hierarchy_applied: bool = Field(False)
    recyclability_pct: Optional[float] = Field(None, ge=0, le=100)
    durability_assessment: bool = Field(False)
    hazardous_substances_managed: bool = Field(False)


class PollutionDNSHRequest(BaseModel):
    """Pollution prevention DNSH assessment."""
    org_id: str = Field(...)
    activity_code: str = Field(...)
    ied_compliance: bool = Field(False, description="Industrial Emissions Directive BAT")
    reach_compliance: bool = Field(False, description="REACH regulation compliance")
    rohs_compliance: bool = Field(False, description="RoHS Directive compliance")
    emission_limits_met: bool = Field(False)
    svhc_absence: bool = Field(False, description="No Substances of Very High Concern")


class BiodiversityDNSHRequest(BaseModel):
    """Biodiversity DNSH assessment."""
    org_id: str = Field(...)
    activity_code: str = Field(...)
    eia_completed: bool = Field(False, description="Environmental Impact Assessment done")
    natura_2000_clear: bool = Field(False, description="No impact on Natura 2000 sites")
    high_biodiversity_areas_clear: bool = Field(False)
    no_deforestation: bool = Field(False)
    habitat_management_plan: bool = Field(False)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class DNSHAssessmentResponse(BaseModel):
    """Full DNSH assessment result."""
    assessment_id: str
    org_id: str
    activity_code: str
    sc_objective: str
    overall_status: str
    objective_results: Dict[str, Dict[str, Any]]
    compliant_count: int
    non_compliant_count: int
    insufficient_data_count: int
    not_applicable_count: int
    assessed_at: datetime


class SingleDNSHResponse(BaseModel):
    """Single-objective DNSH result."""
    assessment_id: str
    org_id: str
    activity_code: str
    objective: str
    dnsh_status: str
    criteria: List[Dict[str, Any]]
    requirements: List[str]
    evidence_status: str
    confidence: float
    assessed_at: datetime


class ClimateRiskResponse(BaseModel):
    """Climate risk assessment result."""
    assessment_id: str
    org_id: str
    activity_code: str
    overall_risk_level: str
    hazards_assessed: int
    hazards_required: int
    material_risks: List[Dict[str, Any]]
    adaptation_adequate: bool
    rcp_scenario: str
    asset_lifetime_years: int
    dnsh_cca_status: str
    recommendations: List[str]
    assessed_at: datetime


class DNSHMatrixResponse(BaseModel):
    """DNSH criteria matrix for an activity."""
    activity_code: str
    activity_name: str
    sc_objective: str
    matrix: Dict[str, Dict[str, Any]]
    total_criteria: int
    delegated_act_ref: str
    generated_at: datetime


class DNSHSummaryResponse(BaseModel):
    """DNSH assessment summary for organization."""
    org_id: str
    total_assessments: int
    fully_compliant: int
    partially_compliant: int
    non_compliant: int
    by_objective: Dict[str, Dict[str, int]]
    compliance_rate_pct: float
    generated_at: datetime


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

APPENDIX_A_HAZARDS = [
    "heat_wave", "cold_wave", "wildfire", "cyclone", "storm", "tornado",
    "flood_coastal", "flood_fluvial", "flood_pluvial", "sea_level_rise",
    "drought", "water_stress", "precipitation_change", "ocean_acidification",
    "permafrost_thaw", "glacier_retreat", "soil_degradation", "soil_erosion",
    "landslide", "subsidence",
]


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_dnsh_assessments: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/assess",
    response_model=DNSHAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Full DNSH assessment (5 non-SC objectives)",
    description=(
        "Run a full DNSH assessment evaluating the activity against "
        "all five non-SC environmental objectives per the Delegated Acts."
    ),
)
async def full_dnsh_assessment(request: FullDNSHRequest) -> DNSHAssessmentResponse:
    """Run full DNSH assessment."""
    assessment_id = _generate_id("dnsh")
    all_objectives = ["climate_change_mitigation", "climate_change_adaptation", "water", "circular_economy", "pollution_prevention", "biodiversity"]
    dnsh_objectives = [obj for obj in all_objectives if obj != request.sc_objective]

    objective_results: Dict[str, Dict[str, Any]] = {}
    compliant = 0
    non_compliant = 0
    insufficient = 0
    na_count = 0

    for obj in dnsh_objectives:
        if obj == "climate_change_adaptation":
            if request.climate_risk_assessed:
                obj_status = "compliant"
                compliant += 1
            else:
                obj_status = "insufficient_data"
                insufficient += 1
        elif obj == "water":
            if request.water_compliance is True:
                obj_status = "compliant"
                compliant += 1
            elif request.water_compliance is False:
                obj_status = "non_compliant"
                non_compliant += 1
            else:
                obj_status = "insufficient_data"
                insufficient += 1
        elif obj == "circular_economy":
            if request.waste_hierarchy_applied is True:
                obj_status = "compliant"
                compliant += 1
            elif request.waste_hierarchy_applied is False:
                obj_status = "non_compliant"
                non_compliant += 1
            else:
                obj_status = "insufficient_data"
                insufficient += 1
        elif obj == "pollution_prevention":
            if request.pollution_within_limits is True:
                obj_status = "compliant"
                compliant += 1
            elif request.pollution_within_limits is False:
                obj_status = "non_compliant"
                non_compliant += 1
            else:
                obj_status = "insufficient_data"
                insufficient += 1
        elif obj == "biodiversity":
            if request.biodiversity_eia_done is True:
                obj_status = "compliant"
                compliant += 1
            elif request.biodiversity_eia_done is False:
                obj_status = "non_compliant"
                non_compliant += 1
            else:
                obj_status = "insufficient_data"
                insufficient += 1
        else:
            obj_status = "not_applicable"
            na_count += 1

        objective_results[obj] = {"status": obj_status, "criteria_count": 3}

    overall = "compliant" if non_compliant == 0 and insufficient == 0 else (
        "non_compliant" if non_compliant > 0 else "insufficient_data"
    )

    data = {
        "assessment_id": assessment_id,
        "org_id": request.org_id,
        "activity_code": request.activity_code,
        "sc_objective": request.sc_objective,
        "overall_status": overall,
        "objective_results": objective_results,
        "compliant_count": compliant,
        "non_compliant_count": non_compliant,
        "insufficient_data_count": insufficient,
        "not_applicable_count": na_count,
        "assessed_at": _now(),
    }
    _dnsh_assessments[assessment_id] = data
    return DNSHAssessmentResponse(**data)


@router.post(
    "/assess/{objective}",
    response_model=SingleDNSHResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assess single DNSH objective",
    description="Assess DNSH for a single environmental objective.",
)
async def assess_single_dnsh(
    objective: str,
    request: SingleDNSHRequest,
) -> SingleDNSHResponse:
    """Assess single DNSH objective."""
    valid_objectives = [e.value for e in DNSHObjective]
    if objective not in valid_objectives:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid objective '{objective}'. Valid: {valid_objectives}",
        )

    assessment_id = _generate_id("dnsh_s")
    criteria = []
    requirements = []

    if objective == "climate_change_adaptation":
        criteria = [
            {"criterion": "Climate risk assessment per Appendix A", "type": "mandatory"},
            {"criterion": "Adaptation solutions for material physical risks", "type": "mandatory"},
            {"criterion": "No adverse impact on adaptation efforts of others", "type": "mandatory"},
        ]
        requirements = [
            "Perform robust climate risk assessment covering Appendix A hazards",
            "Implement adaptation measures for identified material risks",
            "Ensure adaptation solutions do not adversely affect others",
        ]
    elif objective == "water":
        criteria = [
            {"criterion": "Water use efficiency plan", "type": "mandatory"},
            {"criterion": "Environmental Quality Standards compliance", "type": "mandatory"},
            {"criterion": "Water Framework Directive compliance", "type": "mandatory"},
        ]
        requirements = [
            "Establish site-level water use and protection plan",
            "Meet Environmental Quality Standards for water bodies",
            "Comply with EU Water Framework Directive requirements",
        ]
    elif objective == "circular_economy":
        criteria = [
            {"criterion": "Waste hierarchy application", "type": "mandatory"},
            {"criterion": "Recyclability and durability", "type": "recommended"},
            {"criterion": "Hazardous substance management", "type": "mandatory"},
        ]
        requirements = [
            "Apply EU waste hierarchy (prevention, reuse, recycling, recovery, disposal)",
            "Maximize recyclability and durability of products/assets",
            "Manage hazardous substances per Waste Framework Directive",
        ]
    elif objective == "pollution_prevention":
        criteria = [
            {"criterion": "IED BAT compliance", "type": "mandatory"},
            {"criterion": "REACH regulation compliance", "type": "mandatory"},
            {"criterion": "RoHS Directive compliance", "type": "conditional"},
            {"criterion": "No SVHC usage", "type": "mandatory"},
        ]
        requirements = [
            "Operate within Industrial Emissions Directive BAT-AELs",
            "Comply with REACH registration and restrictions",
            "No substances of very high concern (SVHC) above threshold",
        ]
    elif objective == "biodiversity":
        criteria = [
            {"criterion": "Environmental Impact Assessment", "type": "mandatory"},
            {"criterion": "Natura 2000 compatibility", "type": "mandatory"},
            {"criterion": "No deforestation", "type": "mandatory"},
            {"criterion": "High biodiversity area protection", "type": "mandatory"},
        ]
        requirements = [
            "Complete EIA per EU EIA Directive",
            "No adverse impact on Natura 2000 sites",
            "Zero deforestation commitment",
            "No degradation of high conservation value areas",
        ]
    else:
        criteria = [{"criterion": "General DNSH assessment required", "type": "mandatory"}]
        requirements = ["Meet delegated act DNSH criteria for this objective"]

    has_evidence = request.evidence is not None and len(request.evidence) > 10
    dnsh_status = "compliant" if has_evidence else "insufficient_data"
    confidence = 0.7 if has_evidence else 0.3

    data = {
        "assessment_id": assessment_id,
        "org_id": request.org_id,
        "activity_code": request.activity_code,
        "objective": objective,
        "dnsh_status": dnsh_status,
        "criteria": criteria,
        "requirements": requirements,
        "evidence_status": "provided" if has_evidence else "missing",
        "confidence": confidence,
        "assessed_at": _now(),
    }
    return SingleDNSHResponse(**data)


@router.post(
    "/climate-risk",
    response_model=ClimateRiskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Perform climate risk assessment",
    description=(
        "Perform a climate risk assessment for DNSH CCA per Appendix A "
        "of the Climate Delegated Act. Evaluates physical climate hazards "
        "over the asset lifetime under the specified RCP scenario."
    ),
)
async def climate_risk_assessment(request: ClimateRiskRequest) -> ClimateRiskResponse:
    """Perform climate risk assessment for DNSH CCA."""
    assessment_id = _generate_id("cra")

    # Minimum hazards required = 10 for comprehensive assessment
    hazards_required = 10
    hazards_assessed = len(request.physical_hazards_assessed)

    # Simulated material risks
    material_risks = []
    high_risk_hazards = {"flood_coastal", "flood_fluvial", "heat_wave", "drought", "wildfire"}
    for hazard in request.physical_hazards_assessed:
        if hazard in high_risk_hazards:
            material_risks.append({
                "hazard": hazard,
                "risk_level": "high",
                "time_horizon": "2030-2050",
                "scenario": request.rcp_scenario,
            })

    adaptation_adequate = (
        len(request.adaptation_measures) >= len(material_risks)
        and hazards_assessed >= hazards_required
    )

    if adaptation_adequate and hazards_assessed >= hazards_required:
        risk_level = "managed"
        dnsh_status = "compliant"
    elif hazards_assessed >= hazards_required:
        risk_level = "partially_managed"
        dnsh_status = "non_compliant"
    else:
        risk_level = "insufficient_assessment"
        dnsh_status = "insufficient_data"

    recommendations = []
    if hazards_assessed < hazards_required:
        recommendations.append(f"Assess at least {hazards_required} Appendix A physical hazards (currently {hazards_assessed}).")
    if len(material_risks) > len(request.adaptation_measures):
        recommendations.append(f"Implement {len(material_risks) - len(request.adaptation_measures)} additional adaptation measures for material risks.")
    if request.rcp_scenario != "rcp8.5":
        recommendations.append("Consider using RCP 8.5 (worst-case) scenario as required by CDA Appendix A.")
    if not recommendations:
        recommendations.append("Climate risk assessment meets DNSH CCA requirements.")

    return ClimateRiskResponse(
        assessment_id=assessment_id,
        org_id=request.org_id,
        activity_code=request.activity_code,
        overall_risk_level=risk_level,
        hazards_assessed=hazards_assessed,
        hazards_required=hazards_required,
        material_risks=material_risks,
        adaptation_adequate=adaptation_adequate,
        rcp_scenario=request.rcp_scenario,
        asset_lifetime_years=request.asset_lifetime_years,
        dnsh_cca_status=dnsh_status,
        recommendations=recommendations,
        assessed_at=_now(),
    )


@router.post(
    "/water",
    response_model=SingleDNSHResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Water DNSH assessment",
    description="Assess DNSH for water and marine resources objective.",
)
async def water_dnsh(request: WaterDNSHRequest) -> SingleDNSHResponse:
    """Water DNSH assessment."""
    assessment_id = _generate_id("dnsh_w")
    checks_passed = sum([
        request.water_use_efficiency_plan,
        request.eqs_compliance,
        request.wfd_compliance,
    ])
    total_checks = 3

    if checks_passed == total_checks:
        dnsh_status = "compliant"
        confidence = 0.90
    elif checks_passed >= 2:
        dnsh_status = "non_compliant"
        confidence = 0.75
    else:
        dnsh_status = "non_compliant"
        confidence = 0.60

    criteria = [
        {"criterion": "Water use efficiency plan", "met": request.water_use_efficiency_plan},
        {"criterion": "Environmental Quality Standards", "met": request.eqs_compliance},
        {"criterion": "Water Framework Directive compliance", "met": request.wfd_compliance},
    ]

    requirements = []
    if not request.water_use_efficiency_plan:
        requirements.append("Establish site-level water use and protection plan")
    if not request.eqs_compliance:
        requirements.append("Ensure discharges meet Environmental Quality Standards")
    if not request.wfd_compliance:
        requirements.append("Achieve compliance with Water Framework Directive")
    if request.water_stress_area:
        requirements.append("Additional water stewardship measures required for water-stressed areas")

    return SingleDNSHResponse(
        assessment_id=assessment_id,
        org_id=request.org_id,
        activity_code=request.activity_code,
        objective="water",
        dnsh_status=dnsh_status,
        criteria=criteria,
        requirements=requirements if requirements else ["All water DNSH criteria met"],
        evidence_status="provided",
        confidence=confidence,
        assessed_at=_now(),
    )


@router.post(
    "/circular-economy",
    response_model=SingleDNSHResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Circular economy DNSH assessment",
    description="Assess DNSH for transition to a circular economy objective.",
)
async def circular_economy_dnsh(request: CircularEconomyDNSHRequest) -> SingleDNSHResponse:
    """Circular economy DNSH assessment."""
    assessment_id = _generate_id("dnsh_ce")
    checks_passed = sum([
        request.waste_hierarchy_applied,
        request.durability_assessment,
        request.hazardous_substances_managed,
    ])

    if checks_passed == 3:
        dnsh_status = "compliant"
        confidence = 0.90
    elif checks_passed >= 2:
        dnsh_status = "non_compliant"
        confidence = 0.70
    else:
        dnsh_status = "non_compliant"
        confidence = 0.50

    criteria = [
        {"criterion": "Waste hierarchy applied", "met": request.waste_hierarchy_applied},
        {"criterion": "Durability assessment done", "met": request.durability_assessment},
        {"criterion": "Hazardous substances managed", "met": request.hazardous_substances_managed},
    ]
    if request.recyclability_pct is not None:
        criteria.append({"criterion": f"Recyclability: {request.recyclability_pct}%", "met": request.recyclability_pct >= 50})

    requirements = []
    if not request.waste_hierarchy_applied:
        requirements.append("Apply EU waste hierarchy (prevention > reuse > recycling > recovery > disposal)")
    if not request.durability_assessment:
        requirements.append("Perform durability and reparability assessment")
    if not request.hazardous_substances_managed:
        requirements.append("Manage hazardous substances per Waste Framework Directive")

    return SingleDNSHResponse(
        assessment_id=assessment_id,
        org_id=request.org_id,
        activity_code=request.activity_code,
        objective="circular_economy",
        dnsh_status=dnsh_status,
        criteria=criteria,
        requirements=requirements if requirements else ["All circular economy DNSH criteria met"],
        evidence_status="provided",
        confidence=confidence,
        assessed_at=_now(),
    )


@router.post(
    "/pollution",
    response_model=SingleDNSHResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Pollution DNSH assessment",
    description="Assess DNSH for pollution prevention and control objective.",
)
async def pollution_dnsh(request: PollutionDNSHRequest) -> SingleDNSHResponse:
    """Pollution prevention DNSH assessment."""
    assessment_id = _generate_id("dnsh_p")
    checks_passed = sum([
        request.ied_compliance,
        request.reach_compliance,
        request.emission_limits_met,
        request.svhc_absence,
    ])

    if checks_passed == 4:
        dnsh_status = "compliant"
        confidence = 0.92
    elif checks_passed >= 3:
        dnsh_status = "non_compliant"
        confidence = 0.70
    else:
        dnsh_status = "non_compliant"
        confidence = 0.50

    criteria = [
        {"criterion": "IED BAT compliance", "met": request.ied_compliance},
        {"criterion": "REACH regulation compliance", "met": request.reach_compliance},
        {"criterion": "RoHS Directive compliance", "met": request.rohs_compliance},
        {"criterion": "Emission limits within BAT-AEL", "met": request.emission_limits_met},
        {"criterion": "No SVHC above threshold", "met": request.svhc_absence},
    ]

    requirements = []
    if not request.ied_compliance:
        requirements.append("Achieve compliance with Industrial Emissions Directive BAT-AELs")
    if not request.reach_compliance:
        requirements.append("Complete REACH registration for all substances > 1 tonne/year")
    if not request.svhc_absence:
        requirements.append("Eliminate or substitute Substances of Very High Concern")
    if not request.emission_limits_met:
        requirements.append("Reduce emissions to within BAT-AEL ranges")

    return SingleDNSHResponse(
        assessment_id=assessment_id,
        org_id=request.org_id,
        activity_code=request.activity_code,
        objective="pollution_prevention",
        dnsh_status=dnsh_status,
        criteria=criteria,
        requirements=requirements if requirements else ["All pollution DNSH criteria met"],
        evidence_status="provided",
        confidence=confidence,
        assessed_at=_now(),
    )


@router.post(
    "/biodiversity",
    response_model=SingleDNSHResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Biodiversity DNSH assessment",
    description="Assess DNSH for biodiversity and ecosystems objective.",
)
async def biodiversity_dnsh(request: BiodiversityDNSHRequest) -> SingleDNSHResponse:
    """Biodiversity DNSH assessment."""
    assessment_id = _generate_id("dnsh_b")
    checks_passed = sum([
        request.eia_completed,
        request.natura_2000_clear,
        request.high_biodiversity_areas_clear,
        request.no_deforestation,
    ])

    if checks_passed == 4:
        dnsh_status = "compliant"
        confidence = 0.90
    elif checks_passed >= 3:
        dnsh_status = "non_compliant"
        confidence = 0.70
    else:
        dnsh_status = "non_compliant"
        confidence = 0.50

    criteria = [
        {"criterion": "EIA completed per EU EIA Directive", "met": request.eia_completed},
        {"criterion": "No adverse impact on Natura 2000 sites", "met": request.natura_2000_clear},
        {"criterion": "High biodiversity areas protected", "met": request.high_biodiversity_areas_clear},
        {"criterion": "Zero deforestation commitment", "met": request.no_deforestation},
        {"criterion": "Habitat management plan in place", "met": request.habitat_management_plan},
    ]

    requirements = []
    if not request.eia_completed:
        requirements.append("Complete Environmental Impact Assessment per EU EIA Directive 2014/52/EU")
    if not request.natura_2000_clear:
        requirements.append("Obtain Appropriate Assessment for Natura 2000 sites")
    if not request.no_deforestation:
        requirements.append("Implement zero deforestation commitment per EUDR")
    if not request.high_biodiversity_areas_clear:
        requirements.append("Ensure no degradation of high conservation value / key biodiversity areas")

    return SingleDNSHResponse(
        assessment_id=assessment_id,
        org_id=request.org_id,
        activity_code=request.activity_code,
        objective="biodiversity",
        dnsh_status=dnsh_status,
        criteria=criteria,
        requirements=requirements if requirements else ["All biodiversity DNSH criteria met"],
        evidence_status="provided",
        confidence=confidence,
        assessed_at=_now(),
    )


@router.get(
    "/{activity_code}/matrix",
    response_model=DNSHMatrixResponse,
    summary="Get DNSH matrix for activity",
    description=(
        "Retrieve the complete DNSH criteria matrix showing requirements "
        "for each non-SC environmental objective."
    ),
)
async def get_dnsh_matrix(
    activity_code: str,
    sc_objective: str = Query("climate_change_mitigation", description="SC objective to exclude"),
) -> DNSHMatrixResponse:
    """Get DNSH criteria matrix for an activity."""
    all_objectives = ["climate_change_mitigation", "climate_change_adaptation", "water", "circular_economy", "pollution_prevention", "biodiversity"]
    dnsh_objectives = [obj for obj in all_objectives if obj != sc_objective]

    matrix: Dict[str, Dict[str, Any]] = {}
    total_criteria = 0

    criteria_map = {
        "climate_change_mitigation": {
            "criteria": ["No significant increase in GHG emissions"],
            "count": 1,
        },
        "climate_change_adaptation": {
            "criteria": ["Climate risk assessment per Appendix A", "Adaptation solutions implemented", "No adverse impact on others' adaptation"],
            "count": 3,
        },
        "water": {
            "criteria": ["Water use efficiency plan", "Environmental Quality Standards", "Water Framework Directive compliance"],
            "count": 3,
        },
        "circular_economy": {
            "criteria": ["Waste hierarchy applied", "Recyclability/durability assessment", "Hazardous substance management"],
            "count": 3,
        },
        "pollution_prevention": {
            "criteria": ["IED BAT compliance", "REACH compliance", "RoHS compliance", "No SVHC above threshold"],
            "count": 4,
        },
        "biodiversity": {
            "criteria": ["EIA completed", "Natura 2000 compatibility", "No deforestation", "High biodiversity area protection"],
            "count": 4,
        },
    }

    for obj in dnsh_objectives:
        cdata = criteria_map.get(obj, {"criteria": ["General DNSH criteria"], "count": 1})
        matrix[obj] = {
            "applicable": True,
            "criteria": cdata["criteria"],
            "criteria_count": cdata["count"],
            "delegated_act_section": f"CDA Annex I, Section {activity_code} - DNSH {obj}",
        }
        total_criteria += cdata["count"]

    return DNSHMatrixResponse(
        activity_code=activity_code,
        activity_name=f"Activity {activity_code}",
        sc_objective=sc_objective,
        matrix=matrix,
        total_criteria=total_criteria,
        delegated_act_ref=f"CDA Annex I, Section {activity_code}",
        generated_at=_now(),
    )


@router.get(
    "/{org_id}/summary",
    response_model=DNSHSummaryResponse,
    summary="DNSH assessment summary",
    description="Get aggregated DNSH assessment summary for an organization.",
)
async def get_dnsh_summary(org_id: str) -> DNSHSummaryResponse:
    """Get DNSH assessment summary."""
    org_assessments = [a for a in _dnsh_assessments.values() if a["org_id"] == org_id]

    fully = sum(1 for a in org_assessments if a["overall_status"] == "compliant")
    partial = sum(1 for a in org_assessments if a["overall_status"] == "insufficient_data")
    non_comp = sum(1 for a in org_assessments if a["overall_status"] == "non_compliant")
    total = len(org_assessments)

    by_obj: Dict[str, Dict[str, int]] = {}
    for a in org_assessments:
        for obj, result in a.get("objective_results", {}).items():
            if obj not in by_obj:
                by_obj[obj] = {"compliant": 0, "non_compliant": 0, "insufficient_data": 0}
            obj_status = result.get("status", "insufficient_data")
            if obj_status in by_obj[obj]:
                by_obj[obj][obj_status] += 1

    rate = round((fully / total) * 100, 1) if total > 0 else 0

    return DNSHSummaryResponse(
        org_id=org_id,
        total_assessments=total,
        fully_compliant=fully,
        partially_compliant=partial,
        non_compliant=non_comp,
        by_objective=by_obj,
        compliance_rate_pct=rate,
        generated_at=_now(),
    )
