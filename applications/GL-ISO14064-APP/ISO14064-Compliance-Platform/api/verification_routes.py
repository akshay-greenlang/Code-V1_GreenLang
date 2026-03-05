"""
GL-ISO14064-APP Verification API

Manages the verification workflow per ISO 14064-3:2019 for GHG inventories.

Verification levels:
    - Limited assurance (review-level procedures)
    - Reasonable assurance (audit-level procedures)

Workflow stages: draft -> internal_review -> approved -> external_verification -> verified

Findings management: create, list, and resolve verification findings with
severity classification and management response tracking.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/iso14064/verification", tags=["Verification"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerificationLevel(str, Enum):
    """ISO 14064-3:2019 assurance levels."""
    LIMITED = "limited"
    REASONABLE = "reasonable"
    NOT_VERIFIED = "not_verified"


class VerificationStage(str, Enum):
    """Verification workflow stages."""
    DRAFT = "draft"
    INTERNAL_REVIEW = "internal_review"
    APPROVED = "approved"
    EXTERNAL_VERIFICATION = "external_verification"
    VERIFIED = "verified"


class FindingSeverity(str, Enum):
    """Severity classification for findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingStatus(str, Enum):
    """Status of a verification finding."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateVerificationRequest(BaseModel):
    """Request to create a verification engagement."""
    inventory_id: str = Field(..., description="Inventory to verify")
    verifier_name: str = Field("", description="Verification body name")
    verifier_accreditation: str = Field("", description="Accreditation number")
    verification_level: VerificationLevel = Field(
        VerificationLevel.LIMITED, description="Assurance level"
    )
    scope_of_verification: str = Field(
        "Full ISO 14064-1:2018 inventory", description="Scope description"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_id": "inv_abc123",
                "verifier_name": "EY Climate Change & Sustainability Services",
                "verifier_accreditation": "ANAB-VB-001234",
                "verification_level": "limited",
                "scope_of_verification": "Full ISO 14064-1:2018 inventory",
            }
        }


class UpdateVerificationRequest(BaseModel):
    """Request to update a verification record."""
    stage: Optional[VerificationStage] = None
    opinion: Optional[str] = Field(None, max_length=2000)
    verifier_name: Optional[str] = None
    verifier_accreditation: Optional[str] = None


class AddFindingRequest(BaseModel):
    """Request to add a verification finding."""
    category: str = Field(..., description="Finding category")
    severity: FindingSeverity = Field(FindingSeverity.MEDIUM, description="Severity")
    description: str = Field(..., min_length=10, max_length=2000, description="Finding description")
    affected_category: Optional[str] = Field(None, description="ISO category affected")
    emissions_impact_tco2e: Optional[float] = Field(None, description="Estimated emission impact")
    recommendation: str = Field("", max_length=1000, description="Recommended corrective action")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "data_quality",
                "severity": "medium",
                "description": "Category 3 transportation emissions use spend-based factors without source documentation.",
                "affected_category": "category_3_transport",
                "emissions_impact_tco2e": 450.0,
                "recommendation": "Obtain supplier-specific emission factors for top 5 logistics providers.",
            }
        }


class ResolveFindingRequest(BaseModel):
    """Request to resolve a verification finding."""
    management_response: str = Field(
        ..., min_length=10, max_length=2000, description="Management response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "management_response": "Obtained supplier-specific EFs for top 5 providers and updated calculations. Impact revised from 450 to 380 tCO2e."
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class FindingResponse(BaseModel):
    """A verification finding."""
    finding_id: str
    verification_id: str
    category: str
    severity: str
    description: str
    affected_category: Optional[str]
    emissions_impact_tco2e: Optional[float]
    recommendation: str
    management_response: Optional[str]
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]


class VerificationResponse(BaseModel):
    """A verification record."""
    verification_id: str
    inventory_id: str
    verifier_name: str
    verifier_accreditation: str
    verification_level: str
    scope_of_verification: str
    stage: str
    opinion: Optional[str]
    opinion_date: Optional[datetime]
    findings_count: int
    open_findings_count: int
    created_at: datetime
    updated_at: datetime


class VerificationHistoryEntry(BaseModel):
    """Summary entry for verification history."""
    verification_id: str
    verification_level: str
    stage: str
    verifier_name: str
    findings_count: int
    created_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_verifications: Dict[str, Dict[str, Any]] = {}
_findings: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints -- Verification Records
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=VerificationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create verification engagement",
    description=(
        "Initiate a verification engagement for an ISO 14064-1 inventory. "
        "Specify the assurance level (limited or reasonable) and the verifier."
    ),
)
async def create_verification(
    request: CreateVerificationRequest,
) -> VerificationResponse:
    """Create a verification engagement."""
    verification_id = _generate_id("ver")
    now = _now()
    record = {
        "verification_id": verification_id,
        "inventory_id": request.inventory_id,
        "verifier_name": request.verifier_name,
        "verifier_accreditation": request.verifier_accreditation,
        "verification_level": request.verification_level.value,
        "scope_of_verification": request.scope_of_verification,
        "stage": VerificationStage.DRAFT.value,
        "opinion": None,
        "opinion_date": None,
        "findings_count": 0,
        "open_findings_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    _verifications[verification_id] = record
    return VerificationResponse(**record)


@router.get(
    "/{verification_id}",
    response_model=VerificationResponse,
    summary="Get verification record",
    description="Retrieve the current status and details of a verification record.",
)
async def get_verification(verification_id: str) -> VerificationResponse:
    """Retrieve a verification record by ID."""
    record = _verifications.get(verification_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {verification_id} not found",
        )
    return VerificationResponse(**record)


@router.get(
    "/history/{inventory_id}",
    response_model=List[VerificationHistoryEntry],
    summary="Verification history",
    description="List all verification engagements for an inventory.",
)
async def get_verification_history(
    inventory_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
) -> List[VerificationHistoryEntry]:
    """List verification history for an inventory."""
    records = [v for v in _verifications.values() if v["inventory_id"] == inventory_id]
    records.sort(key=lambda r: r["created_at"], reverse=True)
    return [
        VerificationHistoryEntry(
            verification_id=r["verification_id"],
            verification_level=r["verification_level"],
            stage=r["stage"],
            verifier_name=r["verifier_name"],
            findings_count=r["findings_count"],
            created_at=r["created_at"],
        )
        for r in records[:limit]
    ]


@router.put(
    "/{verification_id}",
    response_model=VerificationResponse,
    summary="Update verification record",
    description="Update stage, opinion, or verifier details.",
)
async def update_verification(
    verification_id: str,
    request: UpdateVerificationRequest,
) -> VerificationResponse:
    """Update a verification record."""
    record = _verifications.get(verification_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {verification_id} not found",
        )
    updates = request.model_dump(exclude_unset=True)
    if "stage" in updates:
        updates["stage"] = updates["stage"].value if hasattr(updates["stage"], "value") else updates["stage"]
    if "opinion" in updates and updates["opinion"]:
        record["opinion_date"] = _now()
    record.update(updates)
    record["updated_at"] = _now()
    return VerificationResponse(**record)


# ---------------------------------------------------------------------------
# Endpoints -- Findings
# ---------------------------------------------------------------------------

@router.post(
    "/{verification_id}/findings",
    response_model=FindingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add verification finding",
    description=(
        "Add a finding discovered during the verification engagement. "
        "The verification record finding counts are updated automatically."
    ),
)
async def add_finding(
    verification_id: str,
    request: AddFindingRequest,
) -> FindingResponse:
    """Add a finding to a verification record."""
    record = _verifications.get(verification_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {verification_id} not found",
        )
    if record["stage"] == VerificationStage.VERIFIED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot add findings to a completed verification.",
        )
    finding_id = _generate_id("fnd")
    now = _now()
    finding = {
        "finding_id": finding_id,
        "verification_id": verification_id,
        "category": request.category,
        "severity": request.severity.value,
        "description": request.description,
        "affected_category": request.affected_category,
        "emissions_impact_tco2e": request.emissions_impact_tco2e,
        "recommendation": request.recommendation,
        "management_response": None,
        "status": FindingStatus.OPEN.value,
        "created_at": now,
        "resolved_at": None,
    }
    _findings[finding_id] = finding
    record["findings_count"] = record.get("findings_count", 0) + 1
    record["open_findings_count"] = record.get("open_findings_count", 0) + 1
    record["updated_at"] = now
    return FindingResponse(**finding)


@router.get(
    "/{verification_id}/findings",
    response_model=List[FindingResponse],
    summary="List findings",
    description="Retrieve all findings for a verification engagement.",
)
async def list_findings(
    verification_id: str,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
) -> List[FindingResponse]:
    """List findings for a verification record."""
    findings = [f for f in _findings.values() if f["verification_id"] == verification_id]
    if status_filter:
        findings = [f for f in findings if f["status"] == status_filter]
    findings.sort(key=lambda f: f["created_at"], reverse=True)
    return [FindingResponse(**f) for f in findings]


@router.post(
    "/{verification_id}/findings/{finding_id}/resolve",
    response_model=FindingResponse,
    summary="Resolve finding",
    description=(
        "Resolve a verification finding by providing the management response "
        "and corrective action taken."
    ),
)
async def resolve_finding(
    verification_id: str,
    finding_id: str,
    request: ResolveFindingRequest,
) -> FindingResponse:
    """Resolve a verification finding."""
    record = _verifications.get(verification_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {verification_id} not found",
        )
    finding = _findings.get(finding_id)
    if not finding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Finding {finding_id} not found",
        )
    if finding["verification_id"] != verification_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Finding {finding_id} does not belong to verification {verification_id}",
        )
    if finding["status"] == FindingStatus.RESOLVED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Finding {finding_id} is already resolved.",
        )
    finding["status"] = FindingStatus.RESOLVED.value
    finding["management_response"] = request.management_response
    finding["resolved_at"] = _now()
    record["open_findings_count"] = max(0, record.get("open_findings_count", 1) - 1)
    record["updated_at"] = _now()
    return FindingResponse(**finding)
