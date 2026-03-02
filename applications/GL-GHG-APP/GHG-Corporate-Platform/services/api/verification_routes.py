"""
GL-GHG-APP Verification API

Manages the third-party verification workflow for GHG inventories
per ISO 14064-3 and GHG Protocol requirements.

Verification levels:
    - Limited assurance (review-level)
    - Reasonable assurance (audit-level)

Workflow: start -> in_progress -> findings -> resolve -> approve/reject
"""

from fastapi import APIRouter, HTTPException, Query, Path, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/verification", tags=["Verification"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerificationLevel(str, Enum):
    """Assurance levels per ISO 14064-3."""
    LIMITED = "limited"
    REASONABLE = "reasonable"


class VerificationStatus(str, Enum):
    """Verification workflow statuses."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    FINDINGS_OPEN = "findings_open"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class FindingType(str, Enum):
    """Types of verification findings."""
    OBSERVATION = "observation"
    MINOR_NONCONFORMITY = "minor_nonconformity"
    MAJOR_NONCONFORMITY = "major_nonconformity"
    OPPORTUNITY_FOR_IMPROVEMENT = "opportunity_for_improvement"


class FindingSeverity(str, Enum):
    """Materiality/severity of a finding."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingStatus(str, Enum):
    """Status of a verification finding."""
    OPEN = "open"
    IN_REMEDIATION = "in_remediation"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class StartVerificationRequest(BaseModel):
    """Request to initiate verification of a GHG inventory."""
    inventory_id: str = Field(..., description="Inventory to verify")
    level: VerificationLevel = Field(..., description="Assurance level")
    verifier_name: Optional[str] = Field(None, description="Verification body name")
    verifier_accreditation: Optional[str] = Field(None, description="Accreditation number")
    scope_of_verification: Optional[List[int]] = Field(
        None, description="Scopes to verify (null = all scopes in boundary)"
    )
    target_completion_date: Optional[str] = Field(None, description="Target date (YYYY-MM-DD)")
    notes: Optional[str] = Field(None, max_length=2000)

    class Config:
        json_schema_extra = {
            "example": {
                "inventory_id": "inv_abc123",
                "level": "limited",
                "verifier_name": "EY Climate Change & Sustainability Services",
                "verifier_accreditation": "ANAB-VB-001234",
                "scope_of_verification": [1, 2, 3],
                "target_completion_date": "2026-03-31",
                "notes": "Annual limited assurance engagement"
            }
        }


class ApproveVerificationRequest(BaseModel):
    """Request to approve (issue positive opinion on) a verification."""
    opinion: str = Field(
        ..., description="Verification opinion statement"
    )
    opinion_type: str = Field(
        "unmodified", description="Opinion type: unmodified, modified, adverse, disclaimer"
    )
    material_misstatement_found: bool = Field(
        False, description="Whether a material misstatement was identified"
    )
    assurance_statement_url: Optional[str] = Field(
        None, description="URL to signed assurance statement"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "opinion": "Based on our limited assurance procedures, nothing has come to our attention that causes us to believe that the GHG statement is not fairly stated.",
                "opinion_type": "unmodified",
                "material_misstatement_found": False,
                "assurance_statement_url": "https://storage.greenlang.io/verification/stmt_2025.pdf"
            }
        }


class RejectVerificationRequest(BaseModel):
    """Request to reject (issue adverse opinion on) a verification."""
    reason: str = Field(..., min_length=10, max_length=2000, description="Reason for rejection")
    unresolved_findings: Optional[List[str]] = Field(
        None, description="IDs of unresolved findings that led to rejection"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reason": "Material misstatement identified in Scope 1 process emissions. Reported 1,890 tCO2e but supporting documentation indicates 3,200 tCO2e. Difference exceeds 5% materiality threshold.",
                "unresolved_findings": ["fnd_abc123"]
            }
        }


class AddFindingRequest(BaseModel):
    """Request to add a verification finding."""
    type: FindingType = Field(..., description="Finding type")
    severity: FindingSeverity = Field(..., description="Finding severity/materiality")
    scope: Optional[int] = Field(None, ge=1, le=3, description="Related scope")
    category: Optional[str] = Field(None, description="Related category")
    description: str = Field(..., min_length=10, max_length=2000, description="Finding description")
    evidence: Optional[str] = Field(None, max_length=2000, description="Supporting evidence")
    recommendation: Optional[str] = Field(None, max_length=1000, description="Recommended corrective action")
    response_deadline: Optional[str] = Field(None, description="Deadline for response (YYYY-MM-DD)")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "minor_nonconformity",
                "severity": "medium",
                "scope": 1,
                "category": "Refrigerants",
                "description": "Refrigerant inventory records do not reconcile with HVAC maintenance logs. Discrepancy of 0.15 kg HFC-134a.",
                "evidence": "Reviewed maintenance log entries vs. annual inventory. Log shows 3 recharge events totaling 0.95 kg; inventory shows 0.80 kg consumed.",
                "recommendation": "Reconcile refrigerant inventory with all maintenance records and recharge invoices.",
                "response_deadline": "2026-02-15"
            }
        }


class ResolveFindingRequest(BaseModel):
    """Request to resolve a verification finding."""
    resolution: str = Field(..., min_length=10, max_length=2000, description="How the finding was resolved")
    corrective_action_taken: str = Field(..., min_length=10, max_length=2000, description="Corrective action description")
    evidence_url: Optional[str] = Field(None, description="URL to supporting evidence")

    class Config:
        json_schema_extra = {
            "example": {
                "resolution": "Reconciled refrigerant inventory with all maintenance logs. Updated inventory to reflect 0.95 kg HFC-134a consumed.",
                "corrective_action_taken": "Implemented quarterly reconciliation process between maintenance logs and refrigerant inventory. Updated SOP-ENV-012.",
                "evidence_url": "https://storage.greenlang.io/evidence/reconciliation_report.pdf"
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class FindingResponse(BaseModel):
    """A verification finding."""
    finding_id: str
    record_id: str
    type: str
    severity: str
    scope: Optional[int]
    category: Optional[str]
    description: str
    evidence: Optional[str]
    recommendation: Optional[str]
    status: str
    resolution: Optional[str]
    corrective_action: Optional[str]
    response_deadline: Optional[str]
    resolved_at: Optional[datetime]
    created_at: datetime


class VerificationResponse(BaseModel):
    """Verification record."""
    record_id: str
    inventory_id: str
    level: str
    status: str
    verifier_name: Optional[str]
    verifier_accreditation: Optional[str]
    scope_of_verification: List[int]
    findings_count: int
    open_findings_count: int
    opinion: Optional[str]
    opinion_type: Optional[str]
    material_misstatement_found: Optional[bool]
    assurance_statement_url: Optional[str]
    rejection_reason: Optional[str]
    target_completion_date: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]


class VerificationHistoryEntry(BaseModel):
    """An entry in verification history."""
    record_id: str
    level: str
    status: str
    verifier_name: Optional[str]
    findings_count: int
    opinion_type: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_verifications: Dict[str, Dict[str, Any]] = {}
_findings: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/start",
    response_model=VerificationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start verification",
    description=(
        "Initiate a third-party verification engagement for a GHG inventory. "
        "Specify the assurance level (limited or reasonable) and the verifier."
    ),
)
async def start_verification(request: StartVerificationRequest) -> VerificationResponse:
    record_id = _generate_id("ver")
    now = _now()
    scopes = request.scope_of_verification or [1, 2, 3]
    record = {
        "record_id": record_id,
        "inventory_id": request.inventory_id,
        "level": request.level.value,
        "status": VerificationStatus.INITIATED.value,
        "verifier_name": request.verifier_name,
        "verifier_accreditation": request.verifier_accreditation,
        "scope_of_verification": scopes,
        "findings_count": 0,
        "open_findings_count": 0,
        "opinion": None,
        "opinion_type": None,
        "material_misstatement_found": None,
        "assurance_statement_url": None,
        "rejection_reason": None,
        "target_completion_date": request.target_completion_date,
        "started_at": now,
        "completed_at": None,
    }
    _verifications[record_id] = record
    return VerificationResponse(**record)


@router.get(
    "/{record_id}",
    response_model=VerificationResponse,
    summary="Get verification status",
    description="Retrieve the current status and details of a verification record.",
)
async def get_verification(record_id: str) -> VerificationResponse:
    record = _verifications.get(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {record_id} not found",
        )
    return VerificationResponse(**record)


@router.post(
    "/{record_id}/approve",
    response_model=VerificationResponse,
    summary="Approve verification",
    description=(
        "Issue a positive verification opinion. The verifier confirms the "
        "GHG statement is fairly stated (limited) or materially correct (reasonable)."
    ),
)
async def approve_verification(
    record_id: str,
    request: ApproveVerificationRequest,
) -> VerificationResponse:
    record = _verifications.get(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {record_id} not found",
        )
    open_findings = [
        f for f in _findings.values()
        if f["record_id"] == record_id and f["status"] in ("open", "in_remediation")
    ]
    if open_findings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot approve with {len(open_findings)} unresolved findings. Resolve all findings first.",
        )
    record["status"] = VerificationStatus.APPROVED.value
    record["opinion"] = request.opinion
    record["opinion_type"] = request.opinion_type
    record["material_misstatement_found"] = request.material_misstatement_found
    record["assurance_statement_url"] = request.assurance_statement_url
    record["completed_at"] = _now()
    return VerificationResponse(**record)


@router.post(
    "/{record_id}/reject",
    response_model=VerificationResponse,
    summary="Reject verification",
    description=(
        "Issue an adverse verification opinion. The verifier determines the "
        "GHG statement contains material misstatements."
    ),
)
async def reject_verification(
    record_id: str,
    request: RejectVerificationRequest,
) -> VerificationResponse:
    record = _verifications.get(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {record_id} not found",
        )
    record["status"] = VerificationStatus.REJECTED.value
    record["rejection_reason"] = request.reason
    record["opinion_type"] = "adverse"
    record["material_misstatement_found"] = True
    record["completed_at"] = _now()
    return VerificationResponse(**record)


@router.post(
    "/{record_id}/findings",
    response_model=FindingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add verification finding",
    description=(
        "Add a finding discovered during the verification engagement. "
        "Findings can be observations, minor nonconformities, major "
        "nonconformities, or opportunities for improvement."
    ),
)
async def add_finding(
    record_id: str,
    request: AddFindingRequest,
) -> FindingResponse:
    record = _verifications.get(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {record_id} not found",
        )
    if record["status"] in (VerificationStatus.APPROVED.value, VerificationStatus.REJECTED.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot add findings to a completed verification.",
        )

    finding_id = _generate_id("fnd")
    now = _now()
    finding = {
        "finding_id": finding_id,
        "record_id": record_id,
        "type": request.type.value,
        "severity": request.severity.value,
        "scope": request.scope,
        "category": request.category,
        "description": request.description,
        "evidence": request.evidence,
        "recommendation": request.recommendation,
        "status": FindingStatus.OPEN.value,
        "resolution": None,
        "corrective_action": None,
        "response_deadline": request.response_deadline,
        "resolved_at": None,
        "created_at": now,
    }
    _findings[finding_id] = finding

    # Update verification record counts
    record["findings_count"] = record.get("findings_count", 0) + 1
    record["open_findings_count"] = record.get("open_findings_count", 0) + 1
    if record["status"] == VerificationStatus.INITIATED.value:
        record["status"] = VerificationStatus.IN_PROGRESS.value
    if record["open_findings_count"] > 0:
        record["status"] = VerificationStatus.FINDINGS_OPEN.value

    return FindingResponse(**finding)


@router.post(
    "/{record_id}/findings/{finding_id}/resolve",
    response_model=FindingResponse,
    summary="Resolve verification finding",
    description=(
        "Resolve a finding by providing the corrective action taken and "
        "supporting evidence. Resolved findings can be accepted by the verifier."
    ),
)
async def resolve_finding(
    record_id: str,
    finding_id: str,
    request: ResolveFindingRequest,
) -> FindingResponse:
    record = _verifications.get(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification record {record_id} not found",
        )
    finding = _findings.get(finding_id)
    if not finding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Finding {finding_id} not found",
        )
    if finding["record_id"] != record_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Finding {finding_id} does not belong to verification {record_id}",
        )
    if finding["status"] == FindingStatus.RESOLVED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Finding {finding_id} is already resolved.",
        )

    finding["status"] = FindingStatus.RESOLVED.value
    finding["resolution"] = request.resolution
    finding["corrective_action"] = request.corrective_action_taken
    finding["resolved_at"] = _now()

    # Update verification record
    record["open_findings_count"] = max(0, record.get("open_findings_count", 1) - 1)
    if record["open_findings_count"] == 0:
        record["status"] = VerificationStatus.UNDER_REVIEW.value

    return FindingResponse(**finding)


@router.get(
    "/history/{inventory_id}",
    response_model=List[VerificationHistoryEntry],
    summary="Verification history",
    description="List all verification engagements for an inventory.",
)
async def get_verification_history(
    inventory_id: str,
    limit: int = Query(20, ge=1, le=100),
) -> List[VerificationHistoryEntry]:
    records = [
        v for v in _verifications.values()
        if v["inventory_id"] == inventory_id
    ]
    records.sort(key=lambda r: r["started_at"], reverse=True)
    return [
        VerificationHistoryEntry(
            record_id=r["record_id"],
            level=r["level"],
            status=r["status"],
            verifier_name=r.get("verifier_name"),
            findings_count=r.get("findings_count", 0),
            opinion_type=r.get("opinion_type"),
            started_at=r["started_at"],
            completed_at=r.get("completed_at"),
        )
        for r in records[:limit]
    ]
