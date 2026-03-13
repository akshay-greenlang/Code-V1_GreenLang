# -*- coding: utf-8 -*-
"""
Checklist Management Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for audit checklist management including EUDR-specific and
certification scheme checklists with criterion pass/fail/NA tracking.

Endpoints (3):
    GET  /checklists                         - List available checklist templates
    POST /checklists/custom                  - Create a custom checklist
    POST /audits/{audit_id}/checklist-progress - Update checklist progress

RBAC Permissions:
    eudr-tam:checklist:read   - View checklist templates and progress
    eudr-tam:checklist:create - Create custom checklists
    eudr-tam:checklist:update - Update checklist criterion results

Checklist types:
    - EUDR base (17 criteria covering Articles 3, 4, 9, 10, 11)
    - FSC (12 criteria for P1-P10)
    - PEFC (8 criteria for C1-C7)
    - RSPO (8 criteria for P1-P7)
    - Rainforest Alliance (7 criteria for Ch 1-6)
    - ISCC (8 criteria for SR 1-6)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    get_execution_engine,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    CertSchemeEnum,
    ChecklistResponse,
    CriterionUpdateRequest,
    ErrorResponse,
    MetadataSchema,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Checklist Management"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# GET /checklists
# ---------------------------------------------------------------------------


@router.get(
    "/checklists",
    response_model=ChecklistResponse,
    summary="List available checklist templates",
    description=(
        "Retrieve available audit checklist templates for EUDR base criteria "
        "and certification scheme-specific criteria. Templates are versioned "
        "and map scheme criteria to EUDR articles for unified compliance view."
    ),
    responses={
        200: {"description": "Checklist templates retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_checklists(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-tam:checklist:read")),
    _rl: None = Depends(rate_limit_standard),
    execution_engine: object = Depends(get_execution_engine),
    scheme: Optional[CertSchemeEnum] = Query(
        None, description="Filter by certification scheme"
    ),
    checklist_type: Optional[str] = Query(
        None, description="Filter by checklist type (eudr, fsc, pefc, rspo, ra, iscc)"
    ),
) -> ChecklistResponse:
    """List available audit checklist templates.

    Returns versioned checklist templates including EUDR base criteria
    and scheme-specific criteria with EUDR article mappings.

    Args:
        user: Authenticated user with checklist:read permission.
        execution_engine: AuditExecutionEngine singleton.
        scheme: Optional certification scheme filter.
        checklist_type: Optional checklist type filter.

    Returns:
        Available checklist templates with criteria and version info.
    """
    start = time.monotonic()
    try:
        filters: Dict[str, Any] = {}
        if scheme:
            filters["scheme"] = scheme.value
        if checklist_type:
            filters["checklist_type"] = checklist_type

        checklists: List[Dict[str, Any]] = []
        if hasattr(execution_engine, "list_checklists"):
            checklists = await execution_engine.list_checklists(filters=filters)

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(filters, len(checklists))

        return ChecklistResponse(
            checklists=checklists,
            total=len(checklists),
            provenance=ProvenanceInfo(
                provenance_hash=prov_hash,
                processing_time_ms=Decimal(str(round(elapsed, 2))),
            ),
            metadata=MetadataSchema(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list checklists: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve checklist templates",
        )


# ---------------------------------------------------------------------------
# POST /checklists/custom
# ---------------------------------------------------------------------------


@router.post(
    "/checklists/custom",
    status_code=status.HTTP_201_CREATED,
    summary="Create a custom checklist",
    description=(
        "Create a custom audit checklist by combining criteria from "
        "multiple sources (EUDR base + one or more certification schemes). "
        "Custom checklists are versioned and linked to specific audit IDs."
    ),
    responses={
        201: {"description": "Custom checklist created"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def create_custom_checklist(
    request: Request,
    body: Dict[str, Any],
    user: AuthUser = Depends(require_permission("eudr-tam:checklist:create")),
    _rl: None = Depends(rate_limit_write),
    execution_engine: object = Depends(get_execution_engine),
) -> dict:
    """Create a custom audit checklist.

    Allows combining EUDR base criteria with scheme-specific criteria
    into a unified checklist for a specific audit.

    Args:
        body: Custom checklist specification with criteria selection.
        user: Authenticated user with checklist:create permission.
        execution_engine: AuditExecutionEngine singleton.

    Returns:
        Created checklist with ID, criteria count, and version.
    """
    start = time.monotonic()
    try:
        audit_id = body.get("audit_id")
        if not audit_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audit_id is required",
            )

        result: Dict[str, Any] = {}
        if hasattr(execution_engine, "create_custom_checklist"):
            result = await execution_engine.create_custom_checklist(
                audit_id=audit_id,
                criteria_selection=body.get("criteria", []),
                created_by=user.user_id,
            )
        else:
            result = {
                "checklist_id": hashlib.sha256(
                    f"{audit_id}{time.time()}".encode()
                ).hexdigest()[:36],
                "audit_id": audit_id,
                "total_criteria": len(body.get("criteria", [])),
            }

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(audit_id, result.get("checklist_id", ""))

        return {
            "checklist_id": result.get("checklist_id", ""),
            "audit_id": audit_id,
            "total_criteria": result.get("total_criteria", 0),
            "version": result.get("version", "1.0"),
            "provenance_hash": prov_hash,
            "processing_time_ms": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create custom checklist: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create custom checklist",
        )


# ---------------------------------------------------------------------------
# POST /audits/{audit_id}/checklist-progress
# ---------------------------------------------------------------------------


@router.post(
    "/audits/{audit_id}/checklist-progress",
    summary="Update checklist progress",
    description=(
        "Update the progress of an audit checklist by submitting criterion "
        "results (pass/fail/NA) with optional evidence IDs and auditor notes. "
        "Recalculates completion percentage and updates audit findings count."
    ),
    responses={
        200: {"description": "Checklist progress updated"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
        400: {"model": ErrorResponse, "description": "Invalid criterion data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def update_checklist_progress(
    audit_id: str,
    request: Request,
    body: CriterionUpdateRequest,
    user: AuthUser = Depends(require_permission("eudr-tam:checklist:update")),
    _rl: None = Depends(rate_limit_write),
    execution_engine: object = Depends(get_execution_engine),
) -> dict:
    """Update audit checklist criterion results.

    Submits pass/fail/NA results for individual checklist criteria,
    recalculates completion percentage, and updates findings counts.

    Args:
        audit_id: Unique audit identifier.
        body: Criterion update with result, evidence IDs, and notes.
        user: Authenticated user with checklist:update permission.
        execution_engine: AuditExecutionEngine singleton.

    Returns:
        Updated checklist completion status and findings summary.

    Raises:
        HTTPException: 404 if audit not found.
    """
    start = time.monotonic()
    try:
        result: Optional[Dict[str, Any]] = None
        if hasattr(execution_engine, "update_criterion"):
            result = await execution_engine.update_criterion(
                audit_id=audit_id,
                criterion_id=body.criterion_id,
                criterion_result=body.result.value if body.result else None,
                evidence_ids=body.evidence_ids if hasattr(body, "evidence_ids") else [],
                auditor_notes=body.auditor_notes if hasattr(body, "auditor_notes") else None,
                assessed_by=user.user_id,
            )
        else:
            result = {
                "audit_id": audit_id,
                "criterion_id": body.criterion_id,
                "result": body.result.value if body.result else "pass",
                "completion_percentage": "0.00",
            }

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit {audit_id} not found",
            )

        elapsed = (time.monotonic() - start) * 1000
        prov_hash = _compute_provenance(
            f"{audit_id}:{body.criterion_id}", body.result
        )

        return {
            "audit_id": audit_id,
            "criterion_id": result.get("criterion_id", body.criterion_id),
            "result": result.get("result", ""),
            "completion_percentage": result.get("completion_percentage", "0.00"),
            "findings_count": result.get("findings_count", {}),
            "provenance_hash": prov_hash,
            "processing_time_ms": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to update checklist for %s: %s", audit_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update checklist progress",
        )
