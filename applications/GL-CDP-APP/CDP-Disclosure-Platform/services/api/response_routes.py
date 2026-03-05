"""
GL-CDP-APP Response Management API

Manages CDP questionnaire responses through their full lifecycle:
create, update, review, approve, version, attach evidence, assign to
team members, and perform bulk operations.

Response lifecycle: draft -> in_review -> approved -> submitted
Each response is tied to a specific question within a questionnaire instance.

Supports:
    - Rich-text responses with markdown
    - Evidence attachment (documents, data tables, links)
    - Multi-user collaboration and assignment
    - Review workflow with approve/reject and comments
    - Version control for all response edits
    - Bulk import from previous year responses
    - Bulk approval of reviewed responses
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/responses", tags=["Responses"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResponseStatus(str, Enum):
    """Lifecycle status of a CDP response."""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    REJECTED = "rejected"


class EvidenceType(str, Enum):
    """Type of evidence attachment."""
    DOCUMENT = "document"
    DATA_TABLE = "data_table"
    LINK = "link"
    SCREENSHOT = "screenshot"
    CALCULATION = "calculation"
    VERIFICATION_STATEMENT = "verification_statement"


class ReviewDecision(str, Enum):
    """Review decision options."""
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


# ---------------------------------------------------------------------------
# Valid Status Transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Dict[str, List[str]] = {
    "draft": ["in_review"],
    "in_review": ["draft", "approved", "rejected"],
    "approved": ["in_review", "submitted"],
    "rejected": ["draft"],
    "submitted": [],
}


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateResponseRequest(BaseModel):
    """Request to create a response for a question."""
    questionnaire_id: str = Field(..., description="Questionnaire instance ID")
    question_id: str = Field(..., description="Question ID being answered")
    content: str = Field(..., min_length=1, max_length=50000, description="Response content (markdown supported)")
    content_structured: Optional[Dict[str, Any]] = Field(
        None, description="Structured content for table-type questions"
    )
    auto_populated: bool = Field(False, description="Whether this response was auto-populated from MRV data")
    data_source: Optional[str] = Field(None, description="Data source identifier for auto-populated responses")

    class Config:
        json_schema_extra = {
            "example": {
                "questionnaire_id": "cdpq_abc123",
                "question_id": "q_m7_001",
                "content": "Total Scope 1 emissions for the reporting year were 12,450.8 tCO2e.",
                "auto_populated": True,
                "data_source": "MRV-001 Stationary Combustion + MRV-003 Mobile Combustion",
            }
        }


class UpdateResponseRequest(BaseModel):
    """Request to update response content."""
    content: Optional[str] = Field(None, min_length=1, max_length=50000, description="Updated response content")
    content_structured: Optional[Dict[str, Any]] = Field(
        None, description="Updated structured content"
    )
    change_reason: Optional[str] = Field(None, max_length=1000, description="Reason for the change")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Updated Scope 1 emissions: 12,580.3 tCO2e after verification adjustment.",
                "change_reason": "Verification finding resolution - updated mobile combustion figures.",
            }
        }


class TransitionStatusRequest(BaseModel):
    """Request to transition response status."""
    target_status: ResponseStatus = Field(
        ..., description="Target status for the response"
    )
    notes: Optional[str] = Field(None, max_length=2000, description="Transition notes")

    class Config:
        json_schema_extra = {
            "example": {
                "target_status": "in_review",
                "notes": "Ready for sustainability manager review",
            }
        }


class AttachEvidenceRequest(BaseModel):
    """Request to attach evidence to a response."""
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    title: str = Field(..., min_length=1, max_length=300, description="Evidence title")
    description: Optional[str] = Field(None, max_length=2000, description="Evidence description")
    file_url: Optional[str] = Field(None, description="URL to evidence file")
    file_data: Optional[str] = Field(None, description="Base64-encoded file data")
    file_name: Optional[str] = Field(None, description="Original file name")
    file_size_bytes: Optional[int] = Field(None, ge=0, description="File size in bytes")
    link_url: Optional[str] = Field(None, description="External link URL (for link type)")

    class Config:
        json_schema_extra = {
            "example": {
                "evidence_type": "document",
                "title": "Scope 1 Verification Statement 2025",
                "description": "Limited assurance verification from EY covering all Scope 1 sources.",
                "file_url": "https://storage.greenlang.io/evidence/verification_2025.pdf",
                "file_name": "verification_statement_2025.pdf",
                "file_size_bytes": 245000,
            }
        }


class AssignResponseRequest(BaseModel):
    """Request to assign a response to a team member."""
    assignee_id: str = Field(..., description="Team member user ID")
    assignee_email: str = Field(..., description="Team member email")
    assignee_name: str = Field("", description="Team member display name")
    message: Optional[str] = Field(None, max_length=1000, description="Assignment message")

    class Config:
        json_schema_extra = {
            "example": {
                "assignee_id": "usr_def456",
                "assignee_email": "j.smith@example.com",
                "assignee_name": "Jane Smith",
                "message": "Please review and update the Scope 1 emissions data for M7.",
            }
        }


class ReviewResponseRequest(BaseModel):
    """Request to submit a review decision."""
    decision: ReviewDecision = Field(..., description="Review decision")
    reviewer_id: str = Field(..., description="Reviewer user ID")
    reviewer_name: str = Field("", description="Reviewer display name")
    comments: str = Field("", max_length=5000, description="Review comments")
    score_suggestion: Optional[str] = Field(
        None, description="Suggested scoring level (disclosure/awareness/management/leadership)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "decision": "approve",
                "reviewer_id": "usr_ghi789",
                "reviewer_name": "Dr. Alice Chen",
                "comments": "Scope 1 data verified and consistent with MRV output.",
                "score_suggestion": "management",
            }
        }


class BulkImportRequest(BaseModel):
    """Request to import responses from a previous year questionnaire."""
    source_questionnaire_id: str = Field(
        ..., description="Questionnaire to import responses from"
    )
    target_questionnaire_id: str = Field(
        ..., description="Questionnaire to import responses into"
    )
    module_codes: Optional[List[str]] = Field(
        None, description="Specific modules to import (null = all)"
    )
    overwrite_existing: bool = Field(
        False, description="Overwrite responses that already exist"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "source_questionnaire_id": "cdpq_prev123",
                "target_questionnaire_id": "cdpq_curr456",
                "module_codes": ["M0", "M1", "M2"],
                "overwrite_existing": False,
            }
        }


class BulkApproveRequest(BaseModel):
    """Request to bulk approve reviewed responses."""
    response_ids: List[str] = Field(
        ..., min_length=1, max_length=500, description="Response IDs to approve"
    )
    reviewer_id: str = Field(..., description="Approver user ID")
    reviewer_name: str = Field("", description="Approver display name")
    comments: Optional[str] = Field(None, max_length=2000, description="Bulk approval comments")

    class Config:
        json_schema_extra = {
            "example": {
                "response_ids": ["resp_001", "resp_002", "resp_003"],
                "reviewer_id": "usr_ghi789",
                "reviewer_name": "Dr. Alice Chen",
                "comments": "Batch approval of all M7 responses after verification review.",
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class CDPResponseModel(BaseModel):
    """A CDP questionnaire response."""
    response_id: str
    questionnaire_id: str
    question_id: str
    module_code: str
    question_number: str
    content: str
    content_structured: Optional[Dict[str, Any]]
    status: str
    auto_populated: bool
    data_source: Optional[str]
    assignee_id: Optional[str]
    assignee_name: Optional[str]
    version: int
    evidence_count: int
    review_count: int
    created_by: str
    created_at: datetime
    updated_at: datetime


class CDPResponseListEntry(BaseModel):
    """Summary entry in a response list."""
    response_id: str
    questionnaire_id: str
    question_id: str
    module_code: str
    question_number: str
    status: str
    assignee_name: Optional[str]
    version: int
    evidence_count: int
    updated_at: datetime


class EvidenceResponse(BaseModel):
    """Evidence attachment on a response."""
    evidence_id: str
    response_id: str
    evidence_type: str
    title: str
    description: Optional[str]
    file_url: Optional[str]
    file_name: Optional[str]
    file_size_bytes: Optional[int]
    link_url: Optional[str]
    uploaded_by: str
    created_at: datetime


class VersionResponse(BaseModel):
    """A version snapshot of a response."""
    version_id: str
    response_id: str
    version_number: int
    content: str
    content_structured: Optional[Dict[str, Any]]
    change_reason: Optional[str]
    changed_by: str
    created_at: datetime


class ReviewRecord(BaseModel):
    """Review record for a response."""
    review_id: str
    response_id: str
    decision: str
    reviewer_id: str
    reviewer_name: str
    comments: str
    score_suggestion: Optional[str]
    created_at: datetime


class BulkImportResponse(BaseModel):
    """Result of a bulk import operation."""
    import_id: str
    source_questionnaire_id: str
    target_questionnaire_id: str
    total_imported: int
    total_skipped: int
    total_errors: int
    imported_modules: List[str]
    created_at: datetime


class BulkApproveResponse(BaseModel):
    """Result of a bulk approve operation."""
    total_approved: int
    total_skipped: int
    total_errors: int
    approved_ids: List[str]
    skipped_ids: List[str]
    error_details: List[Dict[str, str]]


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_responses: Dict[str, Dict[str, Any]] = {}
_evidence: Dict[str, Dict[str, Any]] = {}
_versions: Dict[str, Dict[str, Any]] = {}
_reviews: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints -- Response CRUD
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=List[CDPResponseListEntry],
    summary="List responses",
    description=(
        "Retrieve all responses for a questionnaire with optional filtering "
        "by module code, response status, assignee, or auto-populated flag."
    ),
)
async def list_responses(
    questionnaire_id: str = Query(..., description="Questionnaire instance ID"),
    module_code: Optional[str] = Query(None, description="Filter by module code"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    assignee_id: Optional[str] = Query(None, description="Filter by assignee user ID"),
    auto_populated: Optional[bool] = Query(None, description="Filter by auto-populated flag"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
) -> List[CDPResponseListEntry]:
    """List responses for a questionnaire."""
    responses = [
        r for r in _responses.values()
        if r["questionnaire_id"] == questionnaire_id
    ]
    if module_code is not None:
        responses = [r for r in responses if r["module_code"] == module_code]
    if status_filter is not None:
        responses = [r for r in responses if r["status"] == status_filter]
    if assignee_id is not None:
        responses = [r for r in responses if r.get("assignee_id") == assignee_id]
    if auto_populated is not None:
        responses = [r for r in responses if r["auto_populated"] == auto_populated]

    responses.sort(key=lambda r: r["updated_at"], reverse=True)
    page = responses[offset: offset + limit]
    return [
        CDPResponseListEntry(
            response_id=r["response_id"],
            questionnaire_id=r["questionnaire_id"],
            question_id=r["question_id"],
            module_code=r["module_code"],
            question_number=r["question_number"],
            status=r["status"],
            assignee_name=r.get("assignee_name"),
            version=r["version"],
            evidence_count=r["evidence_count"],
            updated_at=r["updated_at"],
        )
        for r in page
    ]


@router.get(
    "/{response_id}",
    response_model=CDPResponseModel,
    summary="Get response detail",
    description="Retrieve full details of a single CDP response including content and metadata.",
)
async def get_response(response_id: str) -> CDPResponseModel:
    """Retrieve a response by ID."""
    response = _responses.get(response_id)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response {response_id} not found",
        )
    return CDPResponseModel(**response)


@router.post(
    "",
    response_model=CDPResponseModel,
    status_code=status.HTTP_201_CREATED,
    summary="Create response",
    description=(
        "Create a new response for a question in a CDP questionnaire. "
        "Supports rich-text markdown content and structured data for table questions. "
        "Optionally flags auto-populated responses with MRV data source attribution."
    ),
)
async def create_response(request: CreateResponseRequest) -> CDPResponseModel:
    """Create a new response for a CDP question."""
    existing = [
        r for r in _responses.values()
        if r["questionnaire_id"] == request.questionnaire_id
        and r["question_id"] == request.question_id
    ]
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Response for question {request.question_id} already exists: "
                f"{existing[0]['response_id']}. Use PUT to update."
            ),
        )

    response_id = _generate_id("resp")
    now = _now()

    # Derive module_code and question_number from question_id pattern
    parts = request.question_id.split("_")
    module_code = parts[1].upper() if len(parts) >= 2 else "M0"
    question_number = f"{module_code}.{parts[2]}" if len(parts) >= 3 else request.question_id

    response = {
        "response_id": response_id,
        "questionnaire_id": request.questionnaire_id,
        "question_id": request.question_id,
        "module_code": module_code,
        "question_number": question_number,
        "content": request.content,
        "content_structured": request.content_structured,
        "status": ResponseStatus.DRAFT.value,
        "auto_populated": request.auto_populated,
        "data_source": request.data_source,
        "assignee_id": None,
        "assignee_name": None,
        "version": 1,
        "evidence_count": 0,
        "review_count": 0,
        "created_by": "current_user",
        "created_at": now,
        "updated_at": now,
    }
    _responses[response_id] = response

    # Create initial version
    version_id = _generate_id("ver")
    _versions[version_id] = {
        "version_id": version_id,
        "response_id": response_id,
        "version_number": 1,
        "content": request.content,
        "content_structured": request.content_structured,
        "change_reason": "Initial creation",
        "changed_by": "current_user",
        "created_at": now,
    }

    return CDPResponseModel(**response)


@router.put(
    "/{response_id}",
    response_model=CDPResponseModel,
    summary="Update response content",
    description=(
        "Update the content of a response. Creates a new version snapshot. "
        "Only allowed when status is draft or rejected."
    ),
)
async def update_response(
    response_id: str,
    request: UpdateResponseRequest,
) -> CDPResponseModel:
    """Update response content."""
    response = _responses.get(response_id)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response {response_id} not found",
        )
    if response["status"] not in (ResponseStatus.DRAFT.value, ResponseStatus.REJECTED.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot edit response in '{response['status']}' status. Must be draft or rejected.",
        )

    now = _now()
    updates = request.model_dump(exclude_unset=True)

    if "content" in updates:
        response["content"] = updates["content"]
    if "content_structured" in updates:
        response["content_structured"] = updates["content_structured"]

    response["version"] += 1
    response["updated_at"] = now

    # If previously rejected, move back to draft
    if response["status"] == ResponseStatus.REJECTED.value:
        response["status"] = ResponseStatus.DRAFT.value

    # Create version snapshot
    version_id = _generate_id("ver")
    _versions[version_id] = {
        "version_id": version_id,
        "response_id": response_id,
        "version_number": response["version"],
        "content": response["content"],
        "content_structured": response["content_structured"],
        "change_reason": updates.get("change_reason", "Content update"),
        "changed_by": "current_user",
        "created_at": now,
    }

    return CDPResponseModel(**response)


@router.patch(
    "/{response_id}/status",
    response_model=CDPResponseModel,
    summary="Transition response status",
    description=(
        "Transition a response to a new lifecycle status. "
        "Valid transitions: draft->in_review, in_review->draft|approved|rejected, "
        "approved->in_review|submitted, rejected->draft."
    ),
)
async def transition_status(
    response_id: str,
    request: TransitionStatusRequest,
) -> CDPResponseModel:
    """Transition response status through the lifecycle."""
    response = _responses.get(response_id)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response {response_id} not found",
        )
    current = response["status"]
    target = request.target_status.value
    allowed = VALID_TRANSITIONS.get(current, [])
    if target not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot transition from '{current}' to '{target}'. "
                f"Allowed transitions: {allowed}"
            ),
        )
    response["status"] = target
    response["updated_at"] = _now()
    return CDPResponseModel(**response)


# ---------------------------------------------------------------------------
# Endpoints -- Version History
# ---------------------------------------------------------------------------

@router.get(
    "/{response_id}/versions",
    response_model=List[VersionResponse],
    summary="List version history",
    description="Retrieve the full version history for a response, ordered newest first.",
)
async def list_versions(
    response_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> List[VersionResponse]:
    """List version history for a response."""
    response = _responses.get(response_id)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response {response_id} not found",
        )
    versions = [
        v for v in _versions.values() if v["response_id"] == response_id
    ]
    versions.sort(key=lambda v: v["version_number"], reverse=True)
    return [VersionResponse(**v) for v in versions[:limit]]


# ---------------------------------------------------------------------------
# Endpoints -- Evidence
# ---------------------------------------------------------------------------

@router.post(
    "/{response_id}/evidence",
    response_model=EvidenceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Attach evidence",
    description=(
        "Attach supporting evidence to a response. Supports documents, "
        "data tables, external links, screenshots, calculations, and "
        "verification statements."
    ),
)
async def attach_evidence(
    response_id: str,
    request: AttachEvidenceRequest,
) -> EvidenceResponse:
    """Attach evidence to a response."""
    response = _responses.get(response_id)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response {response_id} not found",
        )
    if response["status"] == ResponseStatus.SUBMITTED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot attach evidence to a submitted response.",
        )

    evidence_id = _generate_id("evd")
    now = _now()
    evidence = {
        "evidence_id": evidence_id,
        "response_id": response_id,
        "evidence_type": request.evidence_type.value,
        "title": request.title,
        "description": request.description,
        "file_url": request.file_url,
        "file_name": request.file_name,
        "file_size_bytes": request.file_size_bytes,
        "link_url": request.link_url,
        "uploaded_by": "current_user",
        "created_at": now,
    }
    _evidence[evidence_id] = evidence
    response["evidence_count"] = response.get("evidence_count", 0) + 1
    response["updated_at"] = now
    return EvidenceResponse(**evidence)


@router.delete(
    "/{response_id}/evidence/{evidence_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove evidence",
    description="Remove an evidence attachment from a response.",
)
async def remove_evidence(response_id: str, evidence_id: str) -> None:
    """Remove evidence from a response."""
    response = _responses.get(response_id)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response {response_id} not found",
        )
    evidence = _evidence.get(evidence_id)
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence {evidence_id} not found",
        )
    if evidence["response_id"] != response_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evidence {evidence_id} does not belong to response {response_id}",
        )
    if response["status"] == ResponseStatus.SUBMITTED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove evidence from a submitted response.",
        )
    del _evidence[evidence_id]
    response["evidence_count"] = max(0, response.get("evidence_count", 1) - 1)
    response["updated_at"] = _now()


# ---------------------------------------------------------------------------
# Endpoints -- Assignment
# ---------------------------------------------------------------------------

@router.post(
    "/{response_id}/assign",
    response_model=CDPResponseModel,
    summary="Assign to team member",
    description=(
        "Assign a response to a team member for drafting or review. "
        "Sends a notification to the assignee with an optional message."
    ),
)
async def assign_response(
    response_id: str,
    request: AssignResponseRequest,
) -> CDPResponseModel:
    """Assign a response to a team member."""
    response = _responses.get(response_id)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response {response_id} not found",
        )
    if response["status"] == ResponseStatus.SUBMITTED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot reassign a submitted response.",
        )
    response["assignee_id"] = request.assignee_id
    response["assignee_name"] = request.assignee_name
    response["updated_at"] = _now()
    return CDPResponseModel(**response)


# ---------------------------------------------------------------------------
# Endpoints -- Review
# ---------------------------------------------------------------------------

@router.post(
    "/{response_id}/review",
    response_model=CDPResponseModel,
    summary="Submit review",
    description=(
        "Submit a review decision for a response. Can approve, reject, "
        "or request changes. Automatically transitions the response status "
        "based on the decision."
    ),
)
async def submit_review(
    response_id: str,
    request: ReviewResponseRequest,
) -> CDPResponseModel:
    """Submit a review decision for a response."""
    response = _responses.get(response_id)
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response {response_id} not found",
        )
    if response["status"] != ResponseStatus.IN_REVIEW.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Response must be in 'in_review' status to review. Current: '{response['status']}'",
        )

    now = _now()
    review_id = _generate_id("rev")
    _reviews[review_id] = {
        "review_id": review_id,
        "response_id": response_id,
        "decision": request.decision.value,
        "reviewer_id": request.reviewer_id,
        "reviewer_name": request.reviewer_name,
        "comments": request.comments,
        "score_suggestion": request.score_suggestion,
        "created_at": now,
    }
    response["review_count"] = response.get("review_count", 0) + 1
    response["updated_at"] = now

    # Transition status based on decision
    if request.decision == ReviewDecision.APPROVE:
        response["status"] = ResponseStatus.APPROVED.value
    elif request.decision == ReviewDecision.REJECT:
        response["status"] = ResponseStatus.REJECTED.value
    elif request.decision == ReviewDecision.REQUEST_CHANGES:
        response["status"] = ResponseStatus.DRAFT.value

    return CDPResponseModel(**response)


# ---------------------------------------------------------------------------
# Endpoints -- Bulk Operations
# ---------------------------------------------------------------------------

@router.post(
    "/bulk-import",
    response_model=BulkImportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Bulk import previous year responses",
    description=(
        "Import responses from a previous year questionnaire into the current "
        "one. Maps questions by question_id. Optionally filters by module codes "
        "and controls overwrite behavior."
    ),
)
async def bulk_import(request: BulkImportRequest) -> BulkImportResponse:
    """Bulk import responses from a previous year questionnaire."""
    import_id = _generate_id("imp")
    now = _now()

    source_responses = [
        r for r in _responses.values()
        if r["questionnaire_id"] == request.source_questionnaire_id
    ]
    if request.module_codes:
        source_responses = [
            r for r in source_responses
            if r["module_code"] in request.module_codes
        ]

    imported = 0
    skipped = 0
    errors = 0
    imported_modules = set()

    for src in source_responses:
        existing = [
            r for r in _responses.values()
            if r["questionnaire_id"] == request.target_questionnaire_id
            and r["question_id"] == src["question_id"]
        ]
        if existing and not request.overwrite_existing:
            skipped += 1
            continue

        response_id = _generate_id("resp")
        new_response = {
            "response_id": response_id,
            "questionnaire_id": request.target_questionnaire_id,
            "question_id": src["question_id"],
            "module_code": src["module_code"],
            "question_number": src["question_number"],
            "content": src["content"],
            "content_structured": src.get("content_structured"),
            "status": ResponseStatus.DRAFT.value,
            "auto_populated": False,
            "data_source": f"Imported from {request.source_questionnaire_id}",
            "assignee_id": None,
            "assignee_name": None,
            "version": 1,
            "evidence_count": 0,
            "review_count": 0,
            "created_by": "bulk_import",
            "created_at": now,
            "updated_at": now,
        }
        _responses[response_id] = new_response
        imported += 1
        imported_modules.add(src["module_code"])

    return BulkImportResponse(
        import_id=import_id,
        source_questionnaire_id=request.source_questionnaire_id,
        target_questionnaire_id=request.target_questionnaire_id,
        total_imported=imported,
        total_skipped=skipped,
        total_errors=errors,
        imported_modules=sorted(imported_modules),
        created_at=now,
    )


@router.post(
    "/bulk-approve",
    response_model=BulkApproveResponse,
    summary="Bulk approve reviewed responses",
    description=(
        "Approve multiple responses that are currently in 'in_review' status. "
        "Responses not in 'in_review' status are skipped."
    ),
)
async def bulk_approve(request: BulkApproveRequest) -> BulkApproveResponse:
    """Bulk approve reviewed responses."""
    now = _now()
    approved_ids = []
    skipped_ids = []
    error_details = []

    for rid in request.response_ids:
        response = _responses.get(rid)
        if not response:
            error_details.append({"response_id": rid, "error": f"Response {rid} not found"})
            continue
        if response["status"] != ResponseStatus.IN_REVIEW.value:
            skipped_ids.append(rid)
            continue

        response["status"] = ResponseStatus.APPROVED.value
        response["updated_at"] = now
        response["review_count"] = response.get("review_count", 0) + 1

        review_id = _generate_id("rev")
        _reviews[review_id] = {
            "review_id": review_id,
            "response_id": rid,
            "decision": ReviewDecision.APPROVE.value,
            "reviewer_id": request.reviewer_id,
            "reviewer_name": request.reviewer_name,
            "comments": request.comments or "Bulk approval",
            "score_suggestion": None,
            "created_at": now,
        }
        approved_ids.append(rid)

    return BulkApproveResponse(
        total_approved=len(approved_ids),
        total_skipped=len(skipped_ids),
        total_errors=len(error_details),
        approved_ids=approved_ids,
        skipped_ids=skipped_ids,
        error_details=error_details,
    )
