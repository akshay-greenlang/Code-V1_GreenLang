# -*- coding: utf-8 -*-
"""
Audit Integration Routes - AGENT-EUDR-023 Legal Compliance Verifier API

Endpoints for audit report ingestion, listing, findings retrieval, and
corrective action management per EUDR Articles 9, 10, 11, 14, 15.

Endpoints:
    POST /audits/ingest                         - Ingest an audit report
    GET  /audits                                - List audit reports (paginated)
    GET  /audits/{audit_id}/findings            - Get audit findings
    PUT  /audits/{audit_id}/corrective-actions  - Update corrective actions

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023, AuditIntegrationEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.legal_compliance_verifier.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_audit_engine,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.schemas import (
    AuditEntry,
    AuditFindingsResponse,
    AuditIngestRequest,
    AuditIngestResponse,
    AuditListResponse,
    AuditStatusEnum,
    AuditTypeEnum,
    ComplianceCategoryEnum,
    CorrectiveActionEntry,
    CorrectiveActionStatusEnum,
    CorrectiveActionsRequest,
    CorrectiveActionsResponse,
    EUDRCommodityEnum,
    ErrorResponse,
    FindingEntry,
    FindingSeverityEnum,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audits", tags=["Audit Integration"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /audits/ingest
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=AuditIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest an audit report",
    description=(
        "Ingest an internal, external, regulatory, or third-party audit "
        "report into the compliance system. Automatically extracts findings "
        "and creates corrective action tracking."
    ),
    responses={
        201: {"description": "Audit report ingested"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def ingest_audit(
    request: Request,
    body: AuditIngestRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:audit:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> AuditIngestResponse:
    """Ingest an audit report for compliance tracking.

    Args:
        body: Audit ingestion request.
        user: Authenticated user with audit:create permission.

    Returns:
        AuditIngestResponse with ingested audit record.
    """
    start = time.monotonic()

    try:
        engine = get_audit_engine()
        result = engine.ingest(
            audit_type=body.audit_type.value,
            auditor_name=body.auditor_name,
            auditor_accreditation=body.auditor_accreditation,
            audit_date=body.audit_date,
            operator_id=body.operator_id or user.operator_id,
            supplier_id=body.supplier_id,
            scope=body.scope,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            country_code=body.country_code,
            findings_summary=body.findings_summary,
            file_url=body.file_url,
            file_hash=body.file_hash,
            ingested_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audit ingestion failed: invalid data",
            )

        audit = AuditEntry(
            audit_id=result.get("audit_id", ""),
            audit_type=AuditTypeEnum(result.get("audit_type", body.audit_type.value)),
            auditor_name=result.get("auditor_name", body.auditor_name),
            audit_date=result.get("audit_date", body.audit_date),
            status=AuditStatusEnum(result.get("status", "ingested")),
            operator_id=result.get("operator_id", body.operator_id),
            supplier_id=result.get("supplier_id", body.supplier_id),
            country_code=result.get("country_code", body.country_code),
            total_findings=result.get("total_findings", 0),
            major_findings=result.get("major_findings", 0),
            minor_findings=result.get("minor_findings", 0),
            corrective_actions_pending=result.get("corrective_actions_pending", 0),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"audit_ingest:{body.auditor_name}:{body.audit_date}",
            audit.audit_id,
        )

        logger.info(
            "Audit ingested: id=%s type=%s auditor=%s findings=%d user=%s",
            audit.audit_id,
            body.audit_type.value,
            body.auditor_name,
            audit.total_findings,
            user.user_id,
        )

        return AuditIngestResponse(
            audit=audit,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AuditIntegrationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit ingestion failed",
        )


# ---------------------------------------------------------------------------
# GET /audits
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=AuditListResponse,
    summary="List audit reports",
    description=(
        "Retrieve a paginated list of ingested audit reports with optional "
        "filtering by type, status, operator, supplier, and commodity."
    ),
    responses={
        200: {"description": "Audits retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_audits(
    request: Request,
    audit_type: Optional[AuditTypeEnum] = Query(
        None, description="Filter by audit type"
    ),
    audit_status: Optional[AuditStatusEnum] = Query(
        None, alias="status", description="Filter by audit status"
    ),
    operator_id: Optional[str] = Query(
        None, description="Filter by operator ID"
    ),
    supplier_id: Optional[str] = Query(
        None, description="Filter by supplier ID"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-lcv:audit:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AuditListResponse:
    """List audit reports with pagination.

    Args:
        audit_type: Optional type filter.
        audit_status: Optional status filter.
        operator_id: Optional operator filter.
        supplier_id: Optional supplier filter.
        pagination: Pagination parameters.
        user: Authenticated user with audit:read permission.

    Returns:
        AuditListResponse with paginated audit records.
    """
    start = time.monotonic()

    try:
        engine = get_audit_engine()
        result = engine.list_audits(
            audit_type=audit_type.value if audit_type else None,
            status=audit_status.value if audit_status else None,
            operator_id=operator_id,
            supplier_id=supplier_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        audits = []
        for a in result.get("audits", []):
            audits.append(
                AuditEntry(
                    audit_id=a.get("audit_id", ""),
                    audit_type=AuditTypeEnum(a.get("audit_type", "internal")),
                    auditor_name=a.get("auditor_name", ""),
                    audit_date=a.get("audit_date"),
                    status=AuditStatusEnum(a.get("status", "ingested")),
                    operator_id=a.get("operator_id"),
                    supplier_id=a.get("supplier_id"),
                    country_code=a.get("country_code"),
                    total_findings=a.get("total_findings", 0),
                    major_findings=a.get("major_findings", 0),
                    minor_findings=a.get("minor_findings", 0),
                    corrective_actions_pending=a.get("corrective_actions_pending", 0),
                )
            )

        total = result.get("total", len(audits))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"audit_list:{audit_type}:{audit_status}",
            str(total),
        )

        logger.info(
            "Audits listed: total=%d user=%s",
            total,
            user.user_id,
        )

        return AuditListResponse(
            audits=audits,
            total_audits=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AuditIntegrationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit listing failed",
        )


# ---------------------------------------------------------------------------
# GET /audits/{audit_id}/findings
# ---------------------------------------------------------------------------


@router.get(
    "/{audit_id}/findings",
    response_model=AuditFindingsResponse,
    summary="Get audit findings",
    description=(
        "Retrieve all findings from an audit report including severity, "
        "category, corrective action requirements, and deadlines."
    ),
    responses={
        200: {"description": "Findings retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def get_audit_findings(
    audit_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:audit:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AuditFindingsResponse:
    """Get findings from an audit report.

    Args:
        audit_id: Unique audit record identifier.
        user: Authenticated user with audit:read permission.

    Returns:
        AuditFindingsResponse with audit findings.
    """
    start = time.monotonic()

    try:
        engine = get_audit_engine()
        result = engine.get_findings(audit_id=audit_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit not found: {audit_id}",
            )

        findings = []
        for f in result.get("findings", []):
            findings.append(
                FindingEntry(
                    finding_id=f.get("finding_id", ""),
                    severity=FindingSeverityEnum(
                        f.get("severity", "observation")
                    ),
                    category=ComplianceCategoryEnum(f["category"])
                    if f.get("category") else None,
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    regulatory_reference=f.get("regulatory_reference"),
                    corrective_action_required=f.get("corrective_action_required", False),
                    corrective_action_deadline=f.get("corrective_action_deadline"),
                    corrective_action_status=CorrectiveActionStatusEnum(
                        f["corrective_action_status"]
                    ) if f.get("corrective_action_status") else None,
                )
            )

        major_count = sum(
            1 for f in findings
            if f.severity == FindingSeverityEnum.MAJOR_NON_CONFORMITY
        )
        minor_count = sum(
            1 for f in findings
            if f.severity == FindingSeverityEnum.MINOR_NON_CONFORMITY
        )
        observations_count = sum(
            1 for f in findings
            if f.severity == FindingSeverityEnum.OBSERVATION
        )
        ca_required = sum(1 for f in findings if f.corrective_action_required)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"audit_findings:{audit_id}",
            str(len(findings)),
        )

        logger.info(
            "Audit findings retrieved: audit_id=%s total=%d major=%d user=%s",
            audit_id,
            len(findings),
            major_count,
            user.user_id,
        )

        return AuditFindingsResponse(
            audit_id=audit_id,
            findings=findings,
            total_findings=len(findings),
            major_count=major_count,
            minor_count=minor_count,
            observations_count=observations_count,
            corrective_actions_required=ca_required,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AuditIntegrationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audit findings retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit findings retrieval failed",
        )


# ---------------------------------------------------------------------------
# PUT /audits/{audit_id}/corrective-actions
# ---------------------------------------------------------------------------


@router.put(
    "/{audit_id}/corrective-actions",
    response_model=CorrectiveActionsResponse,
    summary="Update corrective actions for an audit",
    description=(
        "Create or update corrective actions for audit findings. Tracks "
        "status, deadlines, responsible parties, and completion evidence."
    ),
    responses={
        200: {"description": "Corrective actions updated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def update_corrective_actions(
    audit_id: str,
    request: Request,
    body: CorrectiveActionsRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:audit:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CorrectiveActionsResponse:
    """Update corrective actions for audit findings.

    Args:
        audit_id: Audit to update corrective actions for.
        body: Corrective actions request.
        user: Authenticated user with audit:update permission.

    Returns:
        CorrectiveActionsResponse with updated actions.
    """
    start = time.monotonic()

    try:
        engine = get_audit_engine()
        result = engine.update_corrective_actions(
            audit_id=audit_id,
            actions=[a.model_dump() for a in body.actions],
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit not found: {audit_id}",
            )

        actions = []
        for a in result.get("actions", []):
            actions.append(
                CorrectiveActionEntry(
                    action_id=a.get("action_id", ""),
                    finding_id=a.get("finding_id", ""),
                    description=a.get("description", ""),
                    responsible_party=a.get("responsible_party", ""),
                    deadline=a.get("deadline"),
                    status=CorrectiveActionStatusEnum(a.get("status", "open")),
                    completion_date=a.get("completion_date"),
                    evidence_urls=a.get("evidence_urls", []),
                    notes=a.get("notes"),
                )
            )

        open_count = sum(
            1 for a in actions if a.status == CorrectiveActionStatusEnum.OPEN
        )
        in_progress = sum(
            1 for a in actions if a.status == CorrectiveActionStatusEnum.IN_PROGRESS
        )
        completed = sum(
            1 for a in actions if a.status in (
                CorrectiveActionStatusEnum.COMPLETED,
                CorrectiveActionStatusEnum.VERIFIED,
            )
        )
        overdue = sum(
            1 for a in actions if a.status == CorrectiveActionStatusEnum.OVERDUE
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"corrective_actions:{audit_id}",
            str(len(actions)),
        )

        logger.info(
            "Corrective actions updated: audit_id=%s total=%d open=%d user=%s",
            audit_id,
            len(actions),
            open_count,
            user.user_id,
        )

        return CorrectiveActionsResponse(
            audit_id=audit_id,
            actions=actions,
            total_actions=len(actions),
            open_count=open_count,
            in_progress_count=in_progress,
            completed_count=completed,
            overdue_count=overdue,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["AuditIntegrationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Corrective actions update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Corrective actions update failed",
        )
