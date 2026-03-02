"""
Document Management API Routes for GL-EUDR-APP v1.0

Handles upload, retrieval, verification, and gap analysis for
supporting documents required under EUDR compliance. Document types
include certificates, permits, land titles, invoices, transport
documents, and other evidence.

Prefix: /api/v1/documents
Tags: Documents
"""

import uuid
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    CERTIFICATE = "CERTIFICATE"
    PERMIT = "PERMIT"
    LAND_TITLE = "LAND_TITLE"
    INVOICE = "INVOICE"
    TRANSPORT = "TRANSPORT"
    OTHER = "OTHER"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class DocumentUploadRequest(BaseModel):
    """Request body for uploading a document.

    Example::

        {
            "name": "FSC Certificate 2024",
            "doc_type": "CERTIFICATE",
            "supplier_id": "sup_abc123",
            "plot_id": "plot_xyz",
            "file_name": "fsc_cert_2024.pdf",
            "file_size_bytes": 245760,
            "mime_type": "application/pdf",
            "description": "Forest Stewardship Council certificate for Plot Alpha",
            "expiry_date": "2025-12-31T00:00:00Z"
        }
    """

    name: str = Field(..., min_length=1, max_length=255, description="Document display name")
    doc_type: DocumentType = Field(..., description="Document classification")
    supplier_id: Optional[str] = Field(None, description="Associated supplier ID")
    plot_id: Optional[str] = Field(None, description="Associated plot ID")
    dds_id: Optional[str] = Field(None, description="Associated DDS ID")
    file_name: str = Field(..., min_length=1, max_length=255, description="Original file name")
    file_size_bytes: Optional[int] = Field(None, ge=0, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type (e.g. application/pdf)")
    description: Optional[str] = Field(None, max_length=2000, description="Document description")
    expiry_date: Optional[datetime] = Field(None, description="Document expiry date")


class DocumentLinkRequest(BaseModel):
    """Request to link a document to an entity.

    Example::

        {
            "entity_type": "supplier",
            "entity_id": "sup_abc123"
        }
    """

    entity_type: str = Field(
        ..., description="Entity type: supplier, plot, or dds"
    )
    entity_id: str = Field(..., description="Entity identifier")


class DocumentVerifyRequest(BaseModel):
    """Request to trigger document verification."""

    verification_notes: Optional[str] = Field(
        None, max_length=2000, description="Notes for the verification process"
    )


class DocumentResponse(BaseModel):
    """Response model for a single document."""

    doc_id: str = Field(..., description="Unique document identifier")
    name: str
    doc_type: DocumentType
    supplier_id: Optional[str] = None
    plot_id: Optional[str] = None
    dds_id: Optional[str] = None
    file_name: str
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    description: Optional[str] = None
    verification_status: VerificationStatus
    verification_notes: Optional[str] = None
    verified_at: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    linked_entities: List[Dict] = Field(
        default_factory=list,
        description="Entities linked to this document",
    )
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    items: List[DocumentResponse]
    page: int
    limit: int
    total: int
    total_pages: int


class DocumentVerificationResponse(BaseModel):
    """Result of a document verification request."""

    doc_id: str
    name: str
    verification_status: VerificationStatus
    verified_at: Optional[datetime]
    verification_notes: Optional[str]
    message: str


class DocumentGap(BaseModel):
    """A single gap identified in document coverage."""

    doc_type: DocumentType
    required: bool
    status: str = Field(..., description="missing | expired | pending_verification")
    description: str


class DocumentGapAnalysisResponse(BaseModel):
    """Document gap analysis for a supplier."""

    supplier_id: str
    total_required: int
    total_present: int
    total_verified: int
    total_gaps: int
    completeness_percent: float
    gaps: List[DocumentGap]


# ---------------------------------------------------------------------------
# In-Memory Storage (v1.0)
# ---------------------------------------------------------------------------

_documents: Dict[str, dict] = {}


def _build_doc_response(data: dict) -> DocumentResponse:
    return DocumentResponse(**data)


# ---------------------------------------------------------------------------
# Required Document Types per EUDR
# ---------------------------------------------------------------------------

REQUIRED_DOC_TYPES = [
    DocumentType.CERTIFICATE,
    DocumentType.PERMIT,
    DocumentType.LAND_TITLE,
    DocumentType.INVOICE,
    DocumentType.TRANSPORT,
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document",
    description="Upload a document with metadata for EUDR compliance evidence.",
)
async def upload_document(body: DocumentUploadRequest) -> DocumentResponse:
    """
    Upload document metadata.

    In v1.0, stores metadata only (file content upload would be handled
    by a separate object storage integration in production).

    Returns:
        201 with created document record.
    """
    now = datetime.now(timezone.utc)
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    linked_entities: List[Dict] = []
    if body.supplier_id:
        linked_entities.append({"entity_type": "supplier", "entity_id": body.supplier_id})
    if body.plot_id:
        linked_entities.append({"entity_type": "plot", "entity_id": body.plot_id})
    if body.dds_id:
        linked_entities.append({"entity_type": "dds", "entity_id": body.dds_id})

    record = {
        "doc_id": doc_id,
        "name": body.name,
        "doc_type": body.doc_type,
        "supplier_id": body.supplier_id,
        "plot_id": body.plot_id,
        "dds_id": body.dds_id,
        "file_name": body.file_name,
        "file_size_bytes": body.file_size_bytes,
        "mime_type": body.mime_type,
        "description": body.description,
        "verification_status": VerificationStatus.PENDING,
        "verification_notes": None,
        "verified_at": None,
        "expiry_date": body.expiry_date,
        "linked_entities": linked_entities,
        "created_at": now,
        "updated_at": now,
    }
    _documents[doc_id] = record
    logger.info("Document uploaded: %s (%s)", doc_id, body.name)
    return _build_doc_response(record)


@router.get(
    "/{doc_id}",
    response_model=DocumentResponse,
    summary="Get document",
    description="Retrieve document details by identifier.",
)
async def get_document(doc_id: str) -> DocumentResponse:
    """
    Fetch document details by ID.

    Returns:
        200 with document record.

    Raises:
        404 if document not found.
    """
    record = _documents.get(doc_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )
    return _build_doc_response(record)


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List documents",
    description="List documents with filtering and pagination.",
)
async def list_documents(
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    doc_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    verification_status: Optional[VerificationStatus] = Query(
        None, description="Filter by verification status"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
) -> DocumentListResponse:
    """
    Retrieve a paginated list of documents.

    Returns:
        200 with paginated document list.
    """
    results = list(_documents.values())

    if supplier_id:
        results = [d for d in results if d.get("supplier_id") == supplier_id]
    if doc_type:
        results = [
            d for d in results
            if (d["doc_type"] == doc_type
                or (isinstance(d["doc_type"], str) and d["doc_type"] == doc_type.value))
        ]
    if verification_status:
        results = [
            d for d in results
            if (d["verification_status"] == verification_status
                or (isinstance(d["verification_status"], str)
                    and d["verification_status"] == verification_status.value))
        ]

    results.sort(key=lambda d: d["created_at"], reverse=True)

    total = len(results)
    total_pages = max(1, math.ceil(total / limit))
    start = (page - 1) * limit
    page_items = results[start : start + limit]

    return DocumentListResponse(
        items=[_build_doc_response(d) for d in page_items],
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages,
    )


@router.post(
    "/{doc_id}/verify",
    response_model=DocumentVerificationResponse,
    summary="Verify document",
    description="Trigger verification of a document's authenticity and validity.",
)
async def verify_document(
    doc_id: str,
    body: Optional[DocumentVerifyRequest] = None,
) -> DocumentVerificationResponse:
    """
    Run document verification.

    In v1.0, this simulates a verification process that checks:
    - Document is not expired
    - Document type is recognized
    - Basic metadata completeness

    Production would integrate OCR, blockchain verification, and
    third-party certificate validation.

    Returns:
        200 with verification result.

    Raises:
        404 if document not found.
    """
    record = _documents.get(doc_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )

    now = datetime.now(timezone.utc)
    notes = body.verification_notes if body else None

    # Simulated verification logic
    is_expired = False
    if record.get("expiry_date"):
        expiry = record["expiry_date"]
        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        if expiry < now:
            is_expired = True

    if is_expired:
        new_status = VerificationStatus.EXPIRED
        message = "Document has expired and cannot be verified"
    else:
        # Simulate successful verification for valid documents
        new_status = VerificationStatus.VERIFIED
        message = "Document verified successfully (simulated)"

    record["verification_status"] = new_status
    record["verified_at"] = now
    record["verification_notes"] = notes
    record["updated_at"] = now

    logger.info("Document %s verification: %s", doc_id, new_status.value)

    return DocumentVerificationResponse(
        doc_id=doc_id,
        name=record["name"],
        verification_status=new_status,
        verified_at=now,
        verification_notes=notes,
        message=message,
    )


@router.post(
    "/{doc_id}/link",
    response_model=DocumentResponse,
    summary="Link document to entity",
    description="Link a document to a supplier, plot, or DDS.",
)
async def link_document(doc_id: str, body: DocumentLinkRequest) -> DocumentResponse:
    """
    Link a document to an entity (supplier, plot, or DDS).

    A document can be linked to multiple entities simultaneously.

    Returns:
        200 with updated document record.

    Raises:
        400 if entity_type is invalid.
        404 if document not found.
    """
    record = _documents.get(doc_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )

    allowed_types = {"supplier", "plot", "dds"}
    if body.entity_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid entity_type '{body.entity_type}'. Allowed: {sorted(allowed_types)}",
        )

    link = {"entity_type": body.entity_type, "entity_id": body.entity_id}

    # Avoid duplicates
    existing = record.get("linked_entities", [])
    if link not in existing:
        existing.append(link)
    record["linked_entities"] = existing

    # Also update convenience fields
    if body.entity_type == "supplier" and not record.get("supplier_id"):
        record["supplier_id"] = body.entity_id
    elif body.entity_type == "plot" and not record.get("plot_id"):
        record["plot_id"] = body.entity_id
    elif body.entity_type == "dds" and not record.get("dds_id"):
        record["dds_id"] = body.entity_id

    record["updated_at"] = datetime.now(timezone.utc)
    logger.info("Document %s linked to %s:%s", doc_id, body.entity_type, body.entity_id)
    return _build_doc_response(record)


@router.get(
    "/gaps/{supplier_id}",
    response_model=DocumentGapAnalysisResponse,
    summary="Document gap analysis",
    description="Analyze document coverage gaps for a supplier's EUDR compliance.",
)
async def document_gap_analysis(supplier_id: str) -> DocumentGapAnalysisResponse:
    """
    Perform document gap analysis for a supplier.

    Checks which required document types are present, verified, missing,
    or expired. Returns a completeness percentage and list of gaps.

    Returns:
        200 with gap analysis results.
    """
    supplier_docs = [
        d for d in _documents.values() if d.get("supplier_id") == supplier_id
    ]

    # Build a set of doc types the supplier has
    present_types: Dict[str, dict] = {}
    for doc in supplier_docs:
        dt = doc["doc_type"]
        dt_value = dt.value if isinstance(dt, DocumentType) else dt
        if dt_value not in present_types:
            present_types[dt_value] = doc
        else:
            # Keep the most recently verified one
            existing = present_types[dt_value]
            vs = doc["verification_status"]
            vs_value = vs.value if isinstance(vs, VerificationStatus) else vs
            if vs_value == "verified":
                present_types[dt_value] = doc

    gaps: List[DocumentGap] = []
    total_verified = 0
    now = datetime.now(timezone.utc)

    for req_type in REQUIRED_DOC_TYPES:
        if req_type.value not in present_types:
            gaps.append(DocumentGap(
                doc_type=req_type,
                required=True,
                status="missing",
                description=f"{req_type.value} document is required but not uploaded",
            ))
        else:
            doc = present_types[req_type.value]
            vs = doc["verification_status"]
            vs_value = vs.value if isinstance(vs, VerificationStatus) else vs

            # Check expiry
            if doc.get("expiry_date"):
                expiry = doc["expiry_date"]
                if isinstance(expiry, str):
                    expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
                if expiry < now:
                    gaps.append(DocumentGap(
                        doc_type=req_type,
                        required=True,
                        status="expired",
                        description=f"{req_type.value} document has expired",
                    ))
                    continue

            if vs_value == "verified":
                total_verified += 1
            elif vs_value == "pending":
                gaps.append(DocumentGap(
                    doc_type=req_type,
                    required=True,
                    status="pending_verification",
                    description=f"{req_type.value} document uploaded but not yet verified",
                ))
            elif vs_value == "rejected":
                gaps.append(DocumentGap(
                    doc_type=req_type,
                    required=True,
                    status="missing",
                    description=f"{req_type.value} document was rejected; re-upload required",
                ))

    total_required = len(REQUIRED_DOC_TYPES)
    total_present = len([t for t in REQUIRED_DOC_TYPES if t.value in present_types])
    completeness = round((total_verified / total_required) * 100, 1) if total_required > 0 else 0.0

    return DocumentGapAnalysisResponse(
        supplier_id=supplier_id,
        total_required=total_required,
        total_present=total_present,
        total_verified=total_verified,
        total_gaps=len(gaps),
        completeness_percent=completeness,
        gaps=gaps,
    )
