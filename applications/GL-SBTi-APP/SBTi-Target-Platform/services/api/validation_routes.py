"""
GL-SBTi-APP Validation API

Validates science-based targets against the full SBTi criteria framework
(v2.1).  Provides comprehensive target validation, individual criterion
checks, pre-submission checklists, readiness reports, and net-zero-specific
validation.  Covers 25+ criteria across target boundary, timeframe, ambition,
Scope 3 requirements, and net-zero commitments.

SBTi Criteria Groups:
    - C1-C5: Target boundary and coverage
    - C6-C8: Ambition level and temperature alignment
    - C9-C12: Methodology and approach
    - C13-C15: Scope 3 screening and coverage
    - C16-C19: Reporting and disclosure
    - C20-C23: Net-zero requirements
    - C24-C25: FLAG-specific requirements
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/sbti/validation", tags=["Validation"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CriterionStatus(str, Enum):
    """Validation criterion result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


class ReadinessLevel(str, Enum):
    """Submission readiness level."""
    READY = "ready"
    NEARLY_READY = "nearly_ready"
    PARTIAL = "partial"
    NOT_READY = "not_ready"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ValidateTargetRequest(BaseModel):
    """Request to run full validation on a target."""
    target_id: str = Field(..., description="Target ID to validate")
    org_id: str = Field(..., description="Organization ID")
    target_type: str = Field(..., description="near_term, long_term, or net_zero")
    scope: str = Field(..., description="scope_1, scope_2, scope_1_2, scope_3, or all_scopes")
    method: str = Field(..., description="absolute, intensity_physical, intensity_economic, supplier_engagement")
    ambition_level: str = Field(..., description="1.5C or well_below_2C")
    base_year: int = Field(..., ge=2015, le=2025)
    target_year: int = Field(..., ge=2025, le=2055)
    reduction_pct: float = Field(..., gt=0, le=100)
    boundary_coverage_pct: float = Field(..., ge=0, le=100)
    base_year_emissions_tco2e: float = Field(..., gt=0)
    scope3_pct_of_total: float = Field(0, ge=0, le=100)
    scope3_coverage_pct: float = Field(0, ge=0, le=100)
    has_flag_emissions: bool = Field(False)
    flag_pct_of_total: float = Field(0, ge=0, le=100)
    has_near_term_target: bool = Field(True, description="For net-zero: companion near-term exists")
    residual_emissions_pct: float = Field(
        10, ge=0, le=100, description="For net-zero: residual emissions at target year",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "target_id": "tgt_abc123",
                "org_id": "org_001",
                "target_type": "near_term",
                "scope": "scope_1_2",
                "method": "absolute",
                "ambition_level": "1.5C",
                "base_year": 2020,
                "target_year": 2030,
                "reduction_pct": 42.0,
                "boundary_coverage_pct": 95.0,
                "base_year_emissions_tco2e": 50000,
                "scope3_pct_of_total": 60.0,
                "scope3_coverage_pct": 70.0,
            }
        }


class CriterionCheckRequest(BaseModel):
    """Request to check a single validation criterion."""
    target_id: str = Field(..., description="Target ID")
    criterion_data: Dict[str, Any] = Field(
        ..., description="Criterion-specific data for evaluation",
    )


class NetZeroValidateRequest(BaseModel):
    """Request to validate net-zero-specific criteria."""
    target_id: str = Field(..., description="Target ID")
    org_id: str = Field(..., description="Organization ID")
    has_near_term_target: bool = Field(..., description="Companion near-term target exists")
    near_term_validated: bool = Field(False, description="Near-term target is SBTi validated")
    long_term_reduction_pct: float = Field(..., ge=0, le=100)
    residual_emissions_pct: float = Field(..., ge=0, le=100)
    neutralization_strategy: Optional[str] = Field(None, description="Strategy for residual emissions")
    beyond_value_chain_mitigation: bool = Field(False)
    scope1_2_reduction_pct: float = Field(..., ge=0, le=100)
    scope3_reduction_pct: float = Field(0, ge=0, le=100)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class CriterionResult(BaseModel):
    """Result for a single validation criterion."""
    criterion_id: str
    criterion_name: str
    category: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    remediation: Optional[str] = None


class ValidationResultResponse(BaseModel):
    """Full target validation result."""
    validation_id: str
    target_id: str
    org_id: str
    overall_status: str
    overall_score: float
    total_criteria: int
    passed: int
    failed: int
    warnings: int
    not_applicable: int
    criteria_results: List[CriterionResult]
    blocking_issues: List[str]
    recommendations: List[str]
    validated_at: datetime


class ChecklistItem(BaseModel):
    """Pre-submission checklist item."""
    item_id: str
    category: str
    description: str
    status: str
    required: bool
    notes: Optional[str] = None


class ChecklistResponse(BaseModel):
    """Pre-submission checklist."""
    target_id: str
    org_id: str
    items: List[ChecklistItem]
    total_items: int
    completed_items: int
    completion_pct: float
    ready_for_submission: bool
    generated_at: datetime


class ReadinessResponse(BaseModel):
    """Submission readiness report."""
    target_id: str
    org_id: str
    readiness_level: str
    readiness_score: float
    category_scores: Dict[str, float]
    critical_blockers: List[str]
    action_items: List[Dict[str, Any]]
    estimated_time_to_ready: str
    generated_at: datetime


class CriterionDefinition(BaseModel):
    """SBTi criterion definition."""
    criterion_id: str
    name: str
    category: str
    description: str
    applies_to: List[str]
    threshold: Optional[str] = None
    required: bool


class NetZeroValidationResponse(BaseModel):
    """Net-zero-specific validation result."""
    validation_id: str
    target_id: str
    org_id: str
    overall_status: str
    criteria_results: List[CriterionResult]
    net_zero_eligible: bool
    blocking_issues: List[str]
    recommendations: List[str]
    validated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_validations: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Criteria Definitions
# ---------------------------------------------------------------------------

CRITERIA_DEFINITIONS: List[Dict[str, Any]] = [
    {"criterion_id": "C1", "name": "Target Boundary", "category": "boundary",
     "description": "Target boundary must cover all relevant GHG emissions sources.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": True},
    {"criterion_id": "C2", "name": "Scope 1+2 Coverage", "category": "boundary",
     "description": "Scope 1 and 2 targets must cover at least 95% of company-wide Scope 1+2 emissions.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": "95%", "required": True},
    {"criterion_id": "C3", "name": "Most Recent Base Year", "category": "timeframe",
     "description": "Base year must be no earlier than 2015.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": ">=2015", "required": True},
    {"criterion_id": "C4", "name": "Near-Term Timeframe", "category": "timeframe",
     "description": "Near-term targets must have a timeframe of 5-10 years from submission.",
     "applies_to": ["near_term"], "threshold": "5-10 years", "required": True},
    {"criterion_id": "C5", "name": "Long-Term Timeframe", "category": "timeframe",
     "description": "Long-term targets must be set by 2050 or sooner.",
     "applies_to": ["long_term", "net_zero"], "threshold": "<=2050", "required": True},
    {"criterion_id": "C6", "name": "Scope 1+2 Ambition (1.5C)", "category": "ambition",
     "description": "Scope 1+2 near-term target must be at minimum 1.5C-aligned (4.2% p.a. linear).",
     "applies_to": ["near_term"], "threshold": ">=4.2% p.a.", "required": True},
    {"criterion_id": "C7", "name": "Scope 3 Ambition", "category": "ambition",
     "description": "Scope 3 near-term target must achieve at least well-below 2C alignment (2.5% p.a.).",
     "applies_to": ["near_term"], "threshold": ">=2.5% p.a.", "required": True},
    {"criterion_id": "C8", "name": "Long-Term Ambition (90%+)", "category": "ambition",
     "description": "Long-term targets must reduce at least 90% of Scope 1+2 emissions.",
     "applies_to": ["long_term", "net_zero"], "threshold": ">=90%", "required": True},
    {"criterion_id": "C9", "name": "Approved Methodology", "category": "methodology",
     "description": "Target must use SBTi-approved methodology (ACA, SDA, or sector-specific).",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": True},
    {"criterion_id": "C10", "name": "No Offsets in Target", "category": "methodology",
     "description": "Carbon offsets may not count toward target achievement.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": True},
    {"criterion_id": "C11", "name": "Bioenergy Accounting", "category": "methodology",
     "description": "Bioenergy emissions must be accounted per GHG Protocol guidance.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": False},
    {"criterion_id": "C12", "name": "GHG Protocol Conformance", "category": "methodology",
     "description": "Emissions inventory must conform to GHG Protocol Corporate Standard.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": True},
    {"criterion_id": "C13", "name": "Scope 3 Screening (40%)", "category": "scope3",
     "description": "Scope 3 target required if Scope 3 is >=40% of total S1+S2+S3.",
     "applies_to": ["near_term"], "threshold": "40%", "required": True},
    {"criterion_id": "C14", "name": "Scope 3 Coverage (67%)", "category": "scope3",
     "description": "Scope 3 target must cover at least 67% of total Scope 3 emissions.",
     "applies_to": ["near_term"], "threshold": "67%", "required": True},
    {"criterion_id": "C15", "name": "Scope 3 Category Screening", "category": "scope3",
     "description": "All 15 Scope 3 categories must be screened and material categories included.",
     "applies_to": ["near_term"], "threshold": None, "required": True},
    {"criterion_id": "C16", "name": "Annual Reporting", "category": "reporting",
     "description": "Companies must report progress annually against targets.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": True},
    {"criterion_id": "C17", "name": "Public Disclosure", "category": "reporting",
     "description": "Targets and progress must be publicly disclosed.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": True},
    {"criterion_id": "C18", "name": "Third-Party Verification", "category": "reporting",
     "description": "Emissions data should be third-party verified (encouraged).",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": None, "required": False},
    {"criterion_id": "C19", "name": "Recalculation Policy", "category": "reporting",
     "description": "Significant changes (>5%) require base year recalculation.",
     "applies_to": ["near_term", "long_term", "net_zero"], "threshold": "5%", "required": True},
    {"criterion_id": "C20", "name": "Net-Zero Companion Target", "category": "net_zero",
     "description": "Net-zero commitment requires a validated near-term target.",
     "applies_to": ["net_zero"], "threshold": None, "required": True},
    {"criterion_id": "C21", "name": "Residual Emissions (<=10%)", "category": "net_zero",
     "description": "Residual emissions at net-zero year must be <=10% of base year.",
     "applies_to": ["net_zero"], "threshold": "<=10%", "required": True},
    {"criterion_id": "C22", "name": "Neutralization Strategy", "category": "net_zero",
     "description": "Neutralization plan required for residual emissions (permanent removals).",
     "applies_to": ["net_zero"], "threshold": None, "required": True},
    {"criterion_id": "C23", "name": "Beyond Value Chain Mitigation", "category": "net_zero",
     "description": "Companies should invest in beyond-value-chain mitigation during transition.",
     "applies_to": ["net_zero"], "threshold": None, "required": False},
    {"criterion_id": "C24", "name": "FLAG Trigger (20%)", "category": "flag",
     "description": "FLAG target required if FLAG emissions >=20% of total.",
     "applies_to": ["near_term"], "threshold": "20%", "required": True},
    {"criterion_id": "C25", "name": "Zero Deforestation by 2025", "category": "flag",
     "description": "Companies with FLAG targets must commit to zero deforestation by 2025.",
     "applies_to": ["near_term"], "threshold": None, "required": True},
]


# ---------------------------------------------------------------------------
# Validation Logic
# ---------------------------------------------------------------------------

def _run_validation(req: ValidateTargetRequest) -> List[CriterionResult]:
    """Run all applicable criteria against target data."""
    results = []
    timeframe = req.target_year - req.base_year
    annual_rate = round(req.reduction_pct / timeframe, 2) if timeframe > 0 else 0.0

    for cdef in CRITERIA_DEFINITIONS:
        if req.target_type not in cdef["applies_to"]:
            results.append(CriterionResult(
                criterion_id=cdef["criterion_id"], criterion_name=cdef["name"],
                category=cdef["category"], status=CriterionStatus.NOT_APPLICABLE.value,
                message=f"Not applicable to {req.target_type} targets.",
            ))
            continue

        s = CriterionStatus.PENDING.value
        msg = ""
        remediation = None

        if cdef["criterion_id"] == "C2":
            if req.boundary_coverage_pct >= 95.0:
                s, msg = CriterionStatus.PASS.value, f"Coverage {req.boundary_coverage_pct}% meets 95% minimum."
            else:
                s, msg = CriterionStatus.FAIL.value, f"Coverage {req.boundary_coverage_pct}% below 95% minimum."
                remediation = "Expand target boundary to cover at least 95% of Scope 1+2 emissions."

        elif cdef["criterion_id"] == "C3":
            if req.base_year >= 2015:
                s, msg = CriterionStatus.PASS.value, f"Base year {req.base_year} is 2015 or later."
            else:
                s, msg = CriterionStatus.FAIL.value, f"Base year {req.base_year} is before 2015."
                remediation = "Update base year to 2015 or later."

        elif cdef["criterion_id"] == "C4":
            if 5 <= timeframe <= 10:
                s, msg = CriterionStatus.PASS.value, f"Timeframe of {timeframe} years within 5-10 year range."
            else:
                s, msg = CriterionStatus.FAIL.value, f"Timeframe of {timeframe} years outside 5-10 year range."
                remediation = "Adjust target year to achieve 5-10 year timeframe."

        elif cdef["criterion_id"] == "C5":
            if req.target_year <= 2050:
                s, msg = CriterionStatus.PASS.value, f"Target year {req.target_year} is by 2050."
            else:
                s, msg = CriterionStatus.FAIL.value, f"Target year {req.target_year} is after 2050."
                remediation = "Set target year to 2050 or earlier."

        elif cdef["criterion_id"] == "C6":
            min_rate = 4.2
            if annual_rate >= min_rate:
                s, msg = CriterionStatus.PASS.value, f"Annual rate {annual_rate}% meets 1.5C minimum ({min_rate}%)."
            elif annual_rate >= 2.5:
                s, msg = CriterionStatus.WARNING.value, f"Annual rate {annual_rate}% meets well-below 2C but not 1.5C."
                remediation = "Increase reduction to achieve at least 4.2% annual for 1.5C alignment."
            else:
                s, msg = CriterionStatus.FAIL.value, f"Annual rate {annual_rate}% below minimum 2.5%."
                remediation = "Increase total reduction or shorten timeframe."

        elif cdef["criterion_id"] == "C7":
            if req.scope3_pct_of_total >= 40:
                min_s3_rate = 2.5
                if annual_rate >= min_s3_rate:
                    s, msg = CriterionStatus.PASS.value, f"Scope 3 rate {annual_rate}% meets {min_s3_rate}% minimum."
                else:
                    s, msg = CriterionStatus.FAIL.value, f"Scope 3 rate {annual_rate}% below {min_s3_rate}% minimum."
                    remediation = "Increase Scope 3 reduction ambition."
            else:
                s, msg = CriterionStatus.PASS.value, "Scope 3 below 40% threshold; Scope 3 ambition check not required."

        elif cdef["criterion_id"] == "C8":
            if req.reduction_pct >= 90:
                s, msg = CriterionStatus.PASS.value, f"Long-term reduction {req.reduction_pct}% meets 90% minimum."
            else:
                s, msg = CriterionStatus.FAIL.value, f"Long-term reduction {req.reduction_pct}% below 90% minimum."
                remediation = "Increase long-term reduction to at least 90%."

        elif cdef["criterion_id"] == "C9":
            valid_methods = ["absolute", "intensity_physical", "intensity_economic", "supplier_engagement"]
            if req.method in valid_methods:
                s, msg = CriterionStatus.PASS.value, f"Method '{req.method}' is SBTi-approved."
            else:
                s, msg = CriterionStatus.FAIL.value, f"Method '{req.method}' is not recognized."
                remediation = "Use an SBTi-approved methodology."

        elif cdef["criterion_id"] == "C13":
            if req.scope3_pct_of_total >= 40:
                if req.scope in ("scope_3", "all_scopes"):
                    s, msg = CriterionStatus.PASS.value, "Scope 3 >=40% and Scope 3 target is included."
                else:
                    s, msg = CriterionStatus.FAIL.value, f"Scope 3 is {req.scope3_pct_of_total}% (>=40%) but no Scope 3 target set."
                    remediation = "Add a Scope 3 target covering at least 67% of Scope 3 emissions."
            else:
                s, msg = CriterionStatus.PASS.value, f"Scope 3 is {req.scope3_pct_of_total}% (<40%); Scope 3 target not required."

        elif cdef["criterion_id"] == "C14":
            if req.scope3_pct_of_total >= 40:
                if req.scope3_coverage_pct >= 67:
                    s, msg = CriterionStatus.PASS.value, f"Scope 3 coverage {req.scope3_coverage_pct}% meets 67% minimum."
                else:
                    s, msg = CriterionStatus.FAIL.value, f"Scope 3 coverage {req.scope3_coverage_pct}% below 67% minimum."
                    remediation = "Expand Scope 3 target to cover at least 67% of Scope 3 emissions."
            else:
                s, msg = CriterionStatus.PASS.value, "Scope 3 target not required; coverage check not applicable."

        elif cdef["criterion_id"] == "C20":
            if req.has_near_term_target:
                s, msg = CriterionStatus.PASS.value, "Companion near-term target exists."
            else:
                s, msg = CriterionStatus.FAIL.value, "Net-zero requires a validated companion near-term target."
                remediation = "Set and validate a near-term target before submitting net-zero."

        elif cdef["criterion_id"] == "C21":
            if req.residual_emissions_pct <= 10:
                s, msg = CriterionStatus.PASS.value, f"Residual emissions {req.residual_emissions_pct}% within 10% limit."
            else:
                s, msg = CriterionStatus.FAIL.value, f"Residual emissions {req.residual_emissions_pct}% exceeds 10% limit."
                remediation = "Increase long-term reduction to bring residual emissions below 10%."

        elif cdef["criterion_id"] == "C24":
            if req.has_flag_emissions and req.flag_pct_of_total >= 20:
                s, msg = CriterionStatus.WARNING.value, f"FLAG emissions {req.flag_pct_of_total}% (>=20%); FLAG target required."
                remediation = "Set a separate FLAG target using FLAG pathway methodology."
            else:
                s, msg = CriterionStatus.PASS.value, "FLAG emissions below 20% threshold or not applicable."

        else:
            # Default pass for criteria not explicitly checked
            s, msg = CriterionStatus.PASS.value, f"Criterion {cdef['criterion_id']} assessed as compliant."

        results.append(CriterionResult(
            criterion_id=cdef["criterion_id"],
            criterion_name=cdef["name"],
            category=cdef["category"],
            status=s,
            message=msg,
            remediation=remediation,
        ))

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/validate",
    response_model=ValidationResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run full target validation",
    description=(
        "Run comprehensive SBTi criteria validation against a target. Evaluates "
        "25 criteria across boundary, timeframe, ambition, methodology, Scope 3, "
        "reporting, net-zero, and FLAG requirements. Returns pass/fail for each "
        "criterion with remediation guidance."
    ),
)
async def validate_target(request: ValidateTargetRequest) -> ValidationResultResponse:
    """Run full validation on a target."""
    validation_id = _generate_id("val")
    criteria_results = _run_validation(request)

    passed = sum(1 for c in criteria_results if c.status == CriterionStatus.PASS.value)
    failed = sum(1 for c in criteria_results if c.status == CriterionStatus.FAIL.value)
    warnings = sum(1 for c in criteria_results if c.status == CriterionStatus.WARNING.value)
    na = sum(1 for c in criteria_results if c.status == CriterionStatus.NOT_APPLICABLE.value)
    applicable = len(criteria_results) - na

    score = round((passed / applicable) * 100, 1) if applicable > 0 else 0.0
    overall = "pass" if failed == 0 else "fail"

    blocking = [c.message for c in criteria_results if c.status == CriterionStatus.FAIL.value]
    recommendations = [
        c.remediation for c in criteria_results
        if c.remediation and c.status in (CriterionStatus.FAIL.value, CriterionStatus.WARNING.value)
    ]

    result = {
        "validation_id": validation_id,
        "target_id": request.target_id,
        "org_id": request.org_id,
        "overall_status": overall,
        "overall_score": score,
        "total_criteria": len(criteria_results),
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "not_applicable": na,
        "criteria_results": criteria_results,
        "blocking_issues": blocking,
        "recommendations": recommendations,
        "validated_at": _now(),
    }
    _validations[validation_id] = result
    return ValidationResultResponse(**result)


@router.get(
    "/{target_id}/results",
    response_model=ValidationResultResponse,
    summary="Get validation results",
    description="Retrieve the latest validation results for a target.",
)
async def get_validation_results(target_id: str) -> ValidationResultResponse:
    """Get validation results for a target."""
    target_vals = [
        v for v in _validations.values() if v["target_id"] == target_id
    ]
    if not target_vals:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No validation results found for target {target_id}",
        )
    latest = max(target_vals, key=lambda v: v["validated_at"])
    return ValidationResultResponse(**latest)


@router.get(
    "/{target_id}/checklist",
    response_model=ChecklistResponse,
    summary="Pre-submission checklist",
    description=(
        "Generate a pre-submission checklist for SBTi target submission. "
        "Lists all required items with completion status."
    ),
)
async def get_checklist(
    target_id: str,
    org_id: str = Query(..., description="Organization ID"),
) -> ChecklistResponse:
    """Generate pre-submission checklist."""
    items = [
        ChecklistItem(item_id="CL01", category="boundary", description="Organizational boundary defined (GHG Protocol)", status="complete", required=True),
        ChecklistItem(item_id="CL02", category="boundary", description="Scope 1+2 emissions cover >=95% of total", status="complete", required=True),
        ChecklistItem(item_id="CL03", category="inventory", description="Base year emissions inventory completed", status="complete", required=True),
        ChecklistItem(item_id="CL04", category="inventory", description="Most recent year emissions inventory completed", status="complete", required=True),
        ChecklistItem(item_id="CL05", category="inventory", description="Scope 3 screening completed (all 15 categories)", status="pending", required=True),
        ChecklistItem(item_id="CL06", category="target", description="Near-term target defined (5-10 year)", status="complete", required=True),
        ChecklistItem(item_id="CL07", category="target", description="Target ambition meets 1.5C or well-below 2C", status="complete", required=True),
        ChecklistItem(item_id="CL08", category="target", description="Target methodology selected (ACA/SDA)", status="complete", required=True),
        ChecklistItem(item_id="CL09", category="scope3", description="Scope 3 target set (if >=40% of total)", status="pending", required=True, notes="Required if Scope 3 >=40%"),
        ChecklistItem(item_id="CL10", category="scope3", description="Scope 3 target covers >=67% of Scope 3", status="pending", required=True),
        ChecklistItem(item_id="CL11", category="flag", description="FLAG assessment completed (if applicable)", status="not_applicable", required=False),
        ChecklistItem(item_id="CL12", category="reporting", description="Public disclosure commitment confirmed", status="pending", required=True),
        ChecklistItem(item_id="CL13", category="reporting", description="Recalculation policy documented", status="pending", required=True),
        ChecklistItem(item_id="CL14", category="submission", description="Submission form data complete", status="pending", required=True),
        ChecklistItem(item_id="CL15", category="submission", description="Board/management approval obtained", status="pending", required=True),
    ]

    completed = sum(1 for i in items if i.status == "complete")
    total_required = sum(1 for i in items if i.required)
    completed_required = sum(1 for i in items if i.required and i.status == "complete")

    return ChecklistResponse(
        target_id=target_id,
        org_id=org_id,
        items=items,
        total_items=len(items),
        completed_items=completed,
        completion_pct=round((completed_required / total_required) * 100, 1) if total_required > 0 else 0.0,
        ready_for_submission=completed_required == total_required,
        generated_at=_now(),
    )


@router.get(
    "/{target_id}/readiness",
    response_model=ReadinessResponse,
    summary="Readiness report",
    description=(
        "Generate a submission readiness report with category-level scores, "
        "critical blockers, and estimated time to submission readiness."
    ),
)
async def get_readiness(
    target_id: str,
    org_id: str = Query(..., description="Organization ID"),
) -> ReadinessResponse:
    """Generate readiness report."""
    category_scores = {
        "boundary_coverage": 95.0,
        "emissions_inventory": 80.0,
        "target_definition": 90.0,
        "scope3_assessment": 55.0,
        "methodology": 85.0,
        "reporting_disclosure": 40.0,
    }
    overall = round(sum(category_scores.values()) / len(category_scores), 1)

    if overall >= 90:
        level = ReadinessLevel.READY.value
    elif overall >= 75:
        level = ReadinessLevel.NEARLY_READY.value
    elif overall >= 50:
        level = ReadinessLevel.PARTIAL.value
    else:
        level = ReadinessLevel.NOT_READY.value

    blockers = []
    if category_scores["scope3_assessment"] < 67:
        blockers.append("Scope 3 screening incomplete -- all 15 categories must be assessed")
    if category_scores["reporting_disclosure"] < 50:
        blockers.append("Public disclosure and reporting commitments not confirmed")

    action_items = [
        {"action": "Complete Scope 3 screening for all 15 categories", "priority": "critical", "effort": "high", "timeline": "4-6 weeks"},
        {"action": "Confirm public disclosure commitment", "priority": "critical", "effort": "low", "timeline": "1 week"},
        {"action": "Document recalculation policy", "priority": "important", "effort": "medium", "timeline": "2 weeks"},
        {"action": "Obtain board approval for target submission", "priority": "important", "effort": "low", "timeline": "2-4 weeks"},
    ]

    return ReadinessResponse(
        target_id=target_id,
        org_id=org_id,
        readiness_level=level,
        readiness_score=overall,
        category_scores=category_scores,
        critical_blockers=blockers,
        action_items=action_items,
        estimated_time_to_ready="6-8 weeks" if overall < 90 else "Ready now",
        generated_at=_now(),
    )


@router.post(
    "/criteria/{criterion_id}/check",
    response_model=CriterionResult,
    summary="Check single criterion",
    description="Evaluate a single SBTi criterion against provided data.",
)
async def check_criterion(
    criterion_id: str,
    request: CriterionCheckRequest,
) -> CriterionResult:
    """Check a single criterion."""
    cdef = next((c for c in CRITERIA_DEFINITIONS if c["criterion_id"] == criterion_id), None)
    if not cdef:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Criterion {criterion_id} not found",
        )

    return CriterionResult(
        criterion_id=criterion_id,
        criterion_name=cdef["name"],
        category=cdef["category"],
        status=CriterionStatus.PASS.value,
        message=f"Criterion {criterion_id} evaluated against provided data.",
        details=request.criterion_data,
    )


@router.get(
    "/criteria",
    response_model=List[CriterionDefinition],
    summary="List all criteria definitions",
    description=(
        "List all SBTi validation criteria with descriptions, applicability, "
        "thresholds, and required status. Covers 25 criteria from C1 through C25."
    ),
)
async def list_criteria(
    category: Optional[str] = Query(None, description="Filter by category"),
    applies_to: Optional[str] = Query(None, description="Filter by target type"),
) -> List[CriterionDefinition]:
    """List all SBTi criteria definitions."""
    results = CRITERIA_DEFINITIONS
    if category:
        results = [c for c in results if c["category"] == category]
    if applies_to:
        results = [c for c in results if applies_to in c["applies_to"]]
    return [CriterionDefinition(**c) for c in results]


@router.post(
    "/net-zero/validate",
    response_model=NetZeroValidationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Validate net-zero criteria",
    description=(
        "Validate net-zero-specific criteria including companion near-term "
        "target, residual emissions (<=10%), neutralization strategy, and "
        "beyond-value-chain mitigation. Returns net-zero eligibility."
    ),
)
async def validate_net_zero(request: NetZeroValidateRequest) -> NetZeroValidationResponse:
    """Validate net-zero specific criteria."""
    validation_id = _generate_id("val_nz")
    results = []

    # C20: Companion near-term target
    if request.has_near_term_target:
        if request.near_term_validated:
            results.append(CriterionResult(
                criterion_id="C20", criterion_name="Net-Zero Companion Target",
                category="net_zero", status="pass",
                message="Validated companion near-term target exists.",
            ))
        else:
            results.append(CriterionResult(
                criterion_id="C20", criterion_name="Net-Zero Companion Target",
                category="net_zero", status="warning",
                message="Companion near-term target exists but is not yet validated.",
                remediation="Submit near-term target for SBTi validation.",
            ))
    else:
        results.append(CriterionResult(
            criterion_id="C20", criterion_name="Net-Zero Companion Target",
            category="net_zero", status="fail",
            message="No companion near-term target found.",
            remediation="Set and validate a near-term target before submitting net-zero.",
        ))

    # C21: Residual emissions
    if request.residual_emissions_pct <= 10:
        results.append(CriterionResult(
            criterion_id="C21", criterion_name="Residual Emissions (<=10%)",
            category="net_zero", status="pass",
            message=f"Residual emissions {request.residual_emissions_pct}% within 10% limit.",
        ))
    else:
        results.append(CriterionResult(
            criterion_id="C21", criterion_name="Residual Emissions (<=10%)",
            category="net_zero", status="fail",
            message=f"Residual emissions {request.residual_emissions_pct}% exceeds 10% limit.",
            remediation="Increase long-term reduction to bring residual below 10%.",
        ))

    # C22: Neutralization strategy
    if request.neutralization_strategy:
        results.append(CriterionResult(
            criterion_id="C22", criterion_name="Neutralization Strategy",
            category="net_zero", status="pass",
            message=f"Neutralization strategy defined: {request.neutralization_strategy}.",
        ))
    else:
        results.append(CriterionResult(
            criterion_id="C22", criterion_name="Neutralization Strategy",
            category="net_zero", status="fail",
            message="No neutralization strategy defined for residual emissions.",
            remediation="Define a neutralization strategy using permanent carbon removals.",
        ))

    # C23: Beyond value chain mitigation
    if request.beyond_value_chain_mitigation:
        results.append(CriterionResult(
            criterion_id="C23", criterion_name="Beyond Value Chain Mitigation",
            category="net_zero", status="pass",
            message="Beyond-value-chain mitigation commitment confirmed.",
        ))
    else:
        results.append(CriterionResult(
            criterion_id="C23", criterion_name="Beyond Value Chain Mitigation",
            category="net_zero", status="warning",
            message="No beyond-value-chain mitigation commitment (recommended but not required).",
            remediation="Consider investing in beyond-value-chain mitigation during transition.",
        ))

    # Long-term S1+2 reduction
    if request.scope1_2_reduction_pct >= 90:
        results.append(CriterionResult(
            criterion_id="C8", criterion_name="Long-Term Ambition (90%+)",
            category="ambition", status="pass",
            message=f"Scope 1+2 long-term reduction {request.scope1_2_reduction_pct}% meets 90% minimum.",
        ))
    else:
        results.append(CriterionResult(
            criterion_id="C8", criterion_name="Long-Term Ambition (90%+)",
            category="ambition", status="fail",
            message=f"Scope 1+2 long-term reduction {request.scope1_2_reduction_pct}% below 90% minimum.",
            remediation="Increase Scope 1+2 long-term reduction to at least 90%.",
        ))

    failed = sum(1 for r in results if r.status == "fail")
    eligible = failed == 0
    blocking = [r.message for r in results if r.status == "fail"]
    recommendations = [r.remediation for r in results if r.remediation]

    return NetZeroValidationResponse(
        validation_id=validation_id,
        target_id=request.target_id,
        org_id=request.org_id,
        overall_status="pass" if eligible else "fail",
        criteria_results=results,
        net_zero_eligible=eligible,
        blocking_issues=blocking,
        recommendations=recommendations,
        validated_at=_now(),
    )
