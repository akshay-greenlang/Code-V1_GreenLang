# -*- coding: utf-8 -*-
"""
Audit Execution Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for checklist management, evidence collection, and real-time
audit progress tracking per EUDR Articles 9-11 and ISO 19011:2018.

Endpoints (5):
    GET  /audits/{audit_id}/checklist                      - Get audit checklist
    PUT  /audits/{audit_id}/checklist/{criterion_id}       - Update criterion result
    POST /audits/{audit_id}/evidence                       - Upload audit evidence
    GET  /audits/{audit_id}/evidence                       - List evidence items
    GET  /audits/{audit_id}/progress                       - Get audit progress

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024, AuditExecutionEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_execution_engine,
    get_pagination,
    rate_limit_evidence,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    AuditStatusEnum,
    ChecklistResponse,
    CriterionUpdateRequest,
    CriterionUpdateResponse,
    CriterionEntry,
    ErrorResponse,
    EvidenceEntry,
    EvidenceListResponse,
    EvidenceTypeEnum,
    EvidenceUploadRequest,
    EvidenceUploadResponse,
    MetadataSchema,
    PaginatedMeta,
    ProgressResponse,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Audit Execution"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# GET /audits/{audit_id}/checklist
# ---------------------------------------------------------------------------


@router.get(
    "/audits/{audit_id}/checklist",
    response_model=ChecklistResponse,
    summary="Get audit checklist",
    description=(
        "Retrieve the audit checklist with EUDR and scheme-specific criteria, "
        "completion status, and evidence linkages."
    ),
    responses={
        200: {"description": "Checklist retrieved"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def get_checklist(
    audit_id: str,
    user: AuthUser = Depends(
        require_permission("eudr-tam:execution:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ChecklistResponse:
    """Retrieve audit checklist with completion status.

    Args:
        audit_id: UUID of the audit.
        user: Authenticated user with execution:read permission.

    Returns:
        ChecklistResponse with checklists and completion data.
    """
    start = time.monotonic()

    try:
        engine = get_execution_engine()
        result = engine.get_checklist(audit_id=audit_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit not found: {audit_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        return ChecklistResponse(
            checklists=result.get("checklists", []),
            overall_completion=Decimal(str(result.get("overall_completion", 0))),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance(
                    f"checklist:{audit_id}", str(result.get("overall_completion", 0))
                ),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Checklist retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve checklist",
        )


# ---------------------------------------------------------------------------
# PUT /audits/{audit_id}/checklist/{criterion_id}
# ---------------------------------------------------------------------------


@router.put(
    "/audits/{audit_id}/checklist/{criterion_id}",
    response_model=CriterionUpdateResponse,
    summary="Update checklist criterion",
    description="Update the result of a specific checklist criterion (pass/fail/NA).",
    responses={
        200: {"description": "Criterion updated"},
        404: {"model": ErrorResponse, "description": "Criterion not found"},
    },
)
async def update_criterion(
    audit_id: str,
    criterion_id: str,
    body: CriterionUpdateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-tam:execution:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CriterionUpdateResponse:
    """Update a checklist criterion result.

    Args:
        audit_id: UUID of the audit.
        criterion_id: ID of the criterion.
        body: Update request with result and notes.
        user: Authenticated user with execution:write permission.

    Returns:
        CriterionUpdateResponse with updated criterion.
    """
    start = time.monotonic()

    try:
        engine = get_execution_engine()
        result = engine.update_criterion(
            audit_id=audit_id,
            criterion_id=criterion_id,
            result=body.result.value,
            evidence_ids=body.evidence_ids,
            auditor_notes=body.auditor_notes,
            assessed_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Criterion not found: {criterion_id}",
            )

        criterion = CriterionEntry(
            criterion_id=criterion_id,
            category=result.get("category", ""),
            reference=result.get("reference", ""),
            description=result.get("description", ""),
            result=body.result,
            evidence_ids=body.evidence_ids or [],
            auditor_notes=body.auditor_notes,
            assessed_by=user.user_id,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        logger.info(
            "Criterion updated: audit=%s criterion=%s result=%s user=%s",
            audit_id, criterion_id, body.result.value, user.user_id,
        )

        return CriterionUpdateResponse(
            criterion=criterion,
            checklist_completion=Decimal(str(result.get("checklist_completion", 0))),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance(
                    f"criterion:{audit_id}:{criterion_id}", body.result.value
                ),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Criterion update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Criterion update failed",
        )


# ---------------------------------------------------------------------------
# POST /audits/{audit_id}/evidence
# ---------------------------------------------------------------------------


@router.post(
    "/audits/{audit_id}/evidence",
    response_model=EvidenceUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload audit evidence",
    description=(
        "Register an audit evidence item with metadata, SHA-256 integrity "
        "hash, geolocation, and type classification."
    ),
    responses={
        201: {"description": "Evidence uploaded"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def upload_evidence(
    audit_id: str,
    body: EvidenceUploadRequest,
    user: AuthUser = Depends(
        require_permission("eudr-tam:execution:write")
    ),
    _rate: None = Depends(rate_limit_evidence),
) -> EvidenceUploadResponse:
    """Upload audit evidence.

    Args:
        audit_id: UUID of the audit.
        body: Evidence upload request with metadata and hash.
        user: Authenticated user with execution:write permission.

    Returns:
        EvidenceUploadResponse with the registered evidence record.
    """
    start = time.monotonic()

    try:
        engine = get_execution_engine()
        result = engine.upload_evidence(
            audit_id=audit_id,
            evidence_type=body.evidence_type.value,
            file_name=body.file_name,
            file_path=body.file_path,
            file_size_bytes=body.file_size_bytes,
            mime_type=body.mime_type,
            description=body.description,
            tags=body.tags,
            location_latitude=body.location_latitude,
            location_longitude=body.location_longitude,
            captured_date=str(body.captured_date) if body.captured_date else None,
            sha256_hash=body.sha256_hash,
            uploaded_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit not found: {audit_id}",
            )

        evidence = EvidenceEntry(
            evidence_id=result.get("evidence_id", ""),
            audit_id=audit_id,
            evidence_type=body.evidence_type,
            file_name=body.file_name,
            file_path=body.file_path,
            file_size_bytes=body.file_size_bytes,
            sha256_hash=body.sha256_hash,
            uploaded_by=user.user_id,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        logger.info(
            "Evidence uploaded: audit=%s evidence=%s type=%s user=%s",
            audit_id, evidence.evidence_id, body.evidence_type.value, user.user_id,
        )

        return EvidenceUploadResponse(
            evidence=evidence,
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance(
                    f"evidence:{audit_id}:{body.sha256_hash[:16]}", evidence.evidence_id
                ),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Evidence upload failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence upload failed",
        )


# ---------------------------------------------------------------------------
# GET /audits/{audit_id}/evidence
# ---------------------------------------------------------------------------


@router.get(
    "/audits/{audit_id}/evidence",
    response_model=EvidenceListResponse,
    summary="List audit evidence",
    description="Retrieve paginated list of evidence items for an audit.",
    responses={
        200: {"description": "Evidence list retrieved"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def list_evidence(
    audit_id: str,
    evidence_type: Optional[str] = Query(None, description="Filter by type"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-tam:execution:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> EvidenceListResponse:
    """Retrieve paginated evidence list for an audit.

    Args:
        audit_id: UUID of the audit.
        evidence_type: Optional evidence type filter.
        pagination: Pagination parameters.
        user: Authenticated user with execution:read permission.

    Returns:
        EvidenceListResponse with evidence items and pagination.
    """
    start = time.monotonic()

    try:
        engine = get_execution_engine()
        result = engine.list_evidence(
            audit_id=audit_id,
            evidence_type=evidence_type,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        items = []
        for e in result.get("evidence_items", []):
            items.append(EvidenceEntry(
                evidence_id=e.get("evidence_id", ""),
                audit_id=audit_id,
                evidence_type=EvidenceTypeEnum(e.get("evidence_type", "other")),
                file_name=e.get("file_name", ""),
                sha256_hash=e.get("sha256_hash", ""),
            ))

        total = result.get("total", len(items))
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        return EvidenceListResponse(
            evidence_items=items,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance(f"evidence_list:{audit_id}", f"total:{total}"),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Evidence list failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evidence list",
        )


# ---------------------------------------------------------------------------
# GET /audits/{audit_id}/progress
# ---------------------------------------------------------------------------


@router.get(
    "/audits/{audit_id}/progress",
    response_model=ProgressResponse,
    summary="Get audit progress",
    description="Retrieve real-time audit progress including checklist completion and findings.",
    responses={
        200: {"description": "Progress retrieved"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def get_progress(
    audit_id: str,
    user: AuthUser = Depends(
        require_permission("eudr-tam:execution:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ProgressResponse:
    """Retrieve real-time audit progress.

    Args:
        audit_id: UUID of the audit.
        user: Authenticated user with execution:read permission.

    Returns:
        ProgressResponse with completion status and metrics.
    """
    start = time.monotonic()

    try:
        engine = get_execution_engine()
        result = engine.get_progress(audit_id=audit_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit not found: {audit_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

        return ProgressResponse(
            audit_id=audit_id,
            status=AuditStatusEnum(result.get("status", "planned")),
            checklist_completion=Decimal(str(result.get("checklist_completion", 0))),
            evidence_count=result.get("evidence_count", 0),
            findings_count=result.get("findings_count", {}),
            team_members=result.get("team_members", 0),
            days_elapsed=result.get("days_elapsed", 0),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance(f"progress:{audit_id}", ""),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Progress retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit progress",
        )
