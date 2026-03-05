"""
GL-ISO14064-APP Significance Assessment API

Manages multi-criteria significance assessments for indirect emission
categories (3-6) per ISO 14064-1:2018 Clause 5.2.2.

Organizations must assess the significance of indirect categories using
defined criteria such as magnitude, influence, risk, stakeholder concern,
and sector norms.  Categories assessed as significant must be quantified
and reported; non-significant categories may be excluded with justification.

Categories 1 and 2 are always mandatory and do not require significance
assessment.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/iso14064/significance", tags=["Significance Assessment"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ISOCategory(str, Enum):
    """ISO categories eligible for significance assessment (3-6 only)."""
    CATEGORY_3_TRANSPORT = "category_3_transport"
    CATEGORY_4_PRODUCTS_USED = "category_4_products_used"
    CATEGORY_5_PRODUCTS_FROM_ORG = "category_5_products_from_org"
    CATEGORY_6_OTHER = "category_6_other"


class SignificanceLevel(str, Enum):
    """Significance assessment result."""
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    UNDER_REVIEW = "under_review"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class SignificanceCriterionRequest(BaseModel):
    """A single significance criterion and score."""
    criterion: str = Field(..., description="Criterion name (e.g. magnitude, influence, risk)")
    weight: float = Field(1.0, ge=0, le=1, description="Criterion weight (0-1)")
    score: float = Field(0.0, ge=0, le=10, description="Score (0-10)")
    rationale: str = Field("", max_length=1000, description="Justification for the score")


class CreateAssessmentRequest(BaseModel):
    """Request to create a significance assessment for an indirect category."""
    category: ISOCategory = Field(..., description="ISO category (3-6)")
    criteria: List[SignificanceCriterionRequest] = Field(
        ..., min_length=1, description="Assessment criteria and scores"
    )
    threshold: float = Field(5.0, ge=0, le=10, description="Significance threshold score")
    estimated_magnitude_tco2e: Optional[float] = Field(None, ge=0, description="Estimated emissions")
    assessed_by: str = Field("", description="Name of assessor")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "category_3_transport",
                "criteria": [
                    {"criterion": "magnitude", "weight": 0.3, "score": 7.5, "rationale": "Estimated 8,500 tCO2e from transportation"},
                    {"criterion": "influence", "weight": 0.2, "score": 6.0, "rationale": "Moderate ability to influence carrier selection"},
                    {"criterion": "risk", "weight": 0.2, "score": 5.0, "rationale": "Regulatory risk from supply chain disclosure requirements"},
                    {"criterion": "stakeholder_concern", "weight": 0.15, "score": 8.0, "rationale": "Key concern from investors and customers"},
                    {"criterion": "sector_norms", "weight": 0.15, "score": 7.0, "rationale": "Industry peers report this category"},
                ],
                "threshold": 5.0,
                "estimated_magnitude_tco2e": 8500.0,
                "assessed_by": "Jane Smith, Sustainability Manager",
            }
        }


class UpdateAssessmentRequest(BaseModel):
    """Request to update a significance assessment."""
    criteria: Optional[List[SignificanceCriterionRequest]] = None
    threshold: Optional[float] = Field(None, ge=0, le=10)
    estimated_magnitude_tco2e: Optional[float] = Field(None, ge=0)
    assessed_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class SignificanceCriterionResponse(BaseModel):
    """A criterion in the assessment."""
    criterion: str
    weight: float
    score: float
    weighted_score: float
    rationale: str


class AssessmentResponse(BaseModel):
    """Significance assessment for an indirect category."""
    assessment_id: str
    inventory_id: str
    category: str
    category_name: str
    criteria: List[SignificanceCriterionResponse]
    total_weighted_score: float
    threshold: float
    result: str
    estimated_magnitude_tco2e: Optional[float]
    magnitude_pct_of_total: Optional[float]
    assessed_by: str
    assessed_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_assessments: Dict[str, Dict[str, Any]] = {}

CATEGORY_NAMES = {
    "category_3_transport": "Category 3 - Indirect GHG emissions from transportation",
    "category_4_products_used": "Category 4 - Indirect GHG emissions from products used by the organization",
    "category_5_products_from_org": "Category 5 - Indirect GHG emissions from the use of products from the organization",
    "category_6_other": "Category 6 - Indirect GHG emissions from other sources",
}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _compute_assessment(criteria_data: List[Dict[str, Any]], threshold: float) -> tuple:
    """Compute total weighted score and significance result."""
    total_weight = sum(c["weight"] for c in criteria_data)
    if total_weight == 0:
        return 0.0, SignificanceLevel.UNDER_REVIEW.value
    total_weighted = sum(c["score"] * c["weight"] for c in criteria_data)
    normalized = round(total_weighted / total_weight, 2)
    if normalized >= threshold:
        result = SignificanceLevel.SIGNIFICANT.value
    else:
        result = SignificanceLevel.NOT_SIGNIFICANT.value
    return normalized, result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{inventory_id}",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create significance assessment",
    description=(
        "Create a multi-criteria significance assessment for an indirect "
        "ISO category (3-6).  The system computes a total weighted score "
        "and determines whether the category is significant based on the "
        "threshold.  Per ISO 14064-1 Clause 5.2.2."
    ),
)
async def create_assessment(
    inventory_id: str,
    request: CreateAssessmentRequest,
) -> AssessmentResponse:
    """Create a significance assessment for an indirect category."""
    assessment_id = _generate_id("sig")
    now = _now()
    criteria_list = []
    for c in request.criteria:
        weighted = round(c.score * c.weight, 4)
        criteria_list.append({
            "criterion": c.criterion,
            "weight": c.weight,
            "score": c.score,
            "weighted_score": weighted,
            "rationale": c.rationale,
        })
    total_weighted, result = _compute_assessment(criteria_list, request.threshold)
    assessment = {
        "assessment_id": assessment_id,
        "inventory_id": inventory_id,
        "category": request.category.value,
        "category_name": CATEGORY_NAMES.get(request.category.value, ""),
        "criteria": criteria_list,
        "total_weighted_score": total_weighted,
        "threshold": request.threshold,
        "result": result,
        "estimated_magnitude_tco2e": request.estimated_magnitude_tco2e,
        "magnitude_pct_of_total": None,
        "assessed_by": request.assessed_by,
        "assessed_at": now,
    }
    _assessments[assessment_id] = assessment
    return AssessmentResponse(**assessment)


@router.get(
    "/{inventory_id}",
    response_model=List[AssessmentResponse],
    summary="List significance assessments",
    description="Retrieve all significance assessments for an inventory.",
)
async def list_assessments(
    inventory_id: str,
    result_filter: Optional[str] = Query(None, alias="result", description="Filter by result: significant, not_significant, under_review"),
) -> List[AssessmentResponse]:
    """List significance assessments for an inventory."""
    assessments = [a for a in _assessments.values() if a["inventory_id"] == inventory_id]
    if result_filter:
        assessments = [a for a in assessments if a["result"] == result_filter]
    assessments.sort(key=lambda a: a["total_weighted_score"], reverse=True)
    return [AssessmentResponse(**a) for a in assessments]


@router.get(
    "/{inventory_id}/{assessment_id}",
    response_model=AssessmentResponse,
    summary="Get significance assessment",
    description="Retrieve a single significance assessment by ID.",
)
async def get_assessment(
    inventory_id: str,
    assessment_id: str,
) -> AssessmentResponse:
    """Retrieve a specific significance assessment."""
    assessment = _assessments.get(assessment_id)
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found",
        )
    if assessment["inventory_id"] != inventory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Assessment {assessment_id} does not belong to inventory {inventory_id}",
        )
    return AssessmentResponse(**assessment)


@router.put(
    "/{inventory_id}/{assessment_id}",
    response_model=AssessmentResponse,
    summary="Update significance assessment",
    description="Update criteria, threshold, or magnitude in a significance assessment.",
)
async def update_assessment(
    inventory_id: str,
    assessment_id: str,
    request: UpdateAssessmentRequest,
) -> AssessmentResponse:
    """Update a significance assessment and recompute the result."""
    assessment = _assessments.get(assessment_id)
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found",
        )
    if assessment["inventory_id"] != inventory_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Assessment {assessment_id} does not belong to inventory {inventory_id}",
        )
    updates = request.model_dump(exclude_unset=True)
    if "criteria" in updates and updates["criteria"] is not None:
        criteria_list = []
        for c in updates["criteria"]:
            if isinstance(c, dict):
                weighted = round(c["score"] * c["weight"], 4)
                criteria_list.append({
                    "criterion": c["criterion"],
                    "weight": c["weight"],
                    "score": c["score"],
                    "weighted_score": weighted,
                    "rationale": c.get("rationale", ""),
                })
            else:
                weighted = round(c.score * c.weight, 4)
                criteria_list.append({
                    "criterion": c.criterion,
                    "weight": c.weight,
                    "score": c.score,
                    "weighted_score": weighted,
                    "rationale": c.rationale,
                })
        assessment["criteria"] = criteria_list
    if "threshold" in updates:
        assessment["threshold"] = updates["threshold"]
    if "estimated_magnitude_tco2e" in updates:
        assessment["estimated_magnitude_tco2e"] = updates["estimated_magnitude_tco2e"]
    if "assessed_by" in updates:
        assessment["assessed_by"] = updates["assessed_by"]
    total_weighted, result = _compute_assessment(assessment["criteria"], assessment["threshold"])
    assessment["total_weighted_score"] = total_weighted
    assessment["result"] = result
    assessment["assessed_at"] = _now()
    return AssessmentResponse(**assessment)
