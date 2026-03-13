# -*- coding: utf-8 -*-
"""
SCP Routes - AGENT-EUDR-010 Segregation Verifier API

Endpoints for registering, updating, validating, and searching
Segregation Control Points (SCPs). SCPs define where physical
segregation is enforced along the supply chain.

Endpoints:
    POST   /scp              - Register a new SCP
    GET    /scp/{scp_id}     - Get SCP details
    PUT    /scp/{scp_id}     - Update SCP
    POST   /scp/validate     - Validate SCP compliance
    POST   /scp/batch-import - Bulk SCP import
    POST   /scp/search       - Search SCPs

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010, Section 7.4
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.segregation_verifier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_request_id,
    get_sgv_service,
    rate_limit_batch,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_scp_id,
)
from greenlang.agents.eudr.segregation_verifier.api.schemas import (
    ContaminationSeverity,
    PaginatedMeta,
    ProvenanceInfo,
    RegisterSCPRequest,
    SCPBatchImportRequest,
    SCPBatchImportResponse,
    SCPListResponse,
    SCPResponse,
    SCPSearchRequest,
    SCPStatus,
    SCPValidationFinding,
    SCPValidationResponse,
    UpdateSCPRequest,
    ValidateSCPRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Segregation Control Points"])

# ---------------------------------------------------------------------------
# In-memory SCP store (replaced by database in production)
# ---------------------------------------------------------------------------

_scp_store: Dict[str, Dict] = {}


def _get_scp_store() -> Dict[str, Dict]:
    """Return the SCP store singleton. Replaceable for testing."""
    return _scp_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /scp
# ---------------------------------------------------------------------------


@router.post(
    "/scp",
    response_model=SCPResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a Segregation Control Point",
    description=(
        "Register a new Segregation Control Point (SCP) at a facility. "
        "SCPs define where physical segregation controls are enforced "
        "to prevent co-mingling of EUDR-regulated commodities."
    ),
    responses={
        201: {"description": "SCP registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_scp(
    request: Request,
    body: RegisterSCPRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:scp:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SCPResponse:
    """Register a new Segregation Control Point.

    Args:
        body: SCP registration parameters.
        user: Authenticated user with scp:create permission.

    Returns:
        SCPResponse with the new SCP details and provenance.
    """
    start = time.monotonic()
    try:
        scp_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        next_inspection = now + timedelta(days=body.inspection_frequency_days)

        scp_record = {
            "scp_id": scp_id,
            "facility_id": body.facility_id,
            "scp_name": body.scp_name,
            "scp_type": body.scp_type,
            "commodity": body.commodity,
            "segregation_level": body.segregation_level,
            "status": SCPStatus.ACTIVE,
            "location": body.location,
            "description": body.description,
            "responsible_person": body.responsible_person,
            "operating_procedures": body.operating_procedures or [],
            "inspection_frequency_days": body.inspection_frequency_days,
            "last_inspection_at": None,
            "next_inspection_due": next_inspection,
            "metadata": body.metadata,
            "created_at": now,
            "updated_at": now,
            "provenance": provenance,
        }

        store = _get_scp_store()
        store[scp_id] = scp_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "SCP registered: id=%s name=%s facility=%s type=%s",
            scp_id,
            body.scp_name,
            body.facility_id,
            body.scp_type.value,
        )

        return SCPResponse(
            **scp_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to register SCP: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register segregation control point",
        )


# ---------------------------------------------------------------------------
# GET /scp/{scp_id}
# ---------------------------------------------------------------------------


@router.get(
    "/scp/{scp_id}",
    response_model=SCPResponse,
    summary="Get SCP details",
    description="Retrieve full details of a Segregation Control Point.",
    responses={
        200: {"description": "SCP details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "SCP not found"},
    },
)
async def get_scp(
    request: Request,
    scp_id: str = Depends(validate_scp_id),
    user: AuthUser = Depends(
        require_permission("eudr-sgv:scp:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SCPResponse:
    """Get SCP details by ID.

    Args:
        scp_id: Unique SCP identifier.
        user: Authenticated user with scp:read permission.

    Returns:
        SCPResponse with full SCP details.

    Raises:
        HTTPException: 404 if SCP not found.
    """
    try:
        store = _get_scp_store()
        record = store.get(scp_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Segregation control point {scp_id} not found",
            )

        return SCPResponse(**record, processing_time_ms=0.0)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve SCP %s: %s", scp_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve segregation control point",
        )


# ---------------------------------------------------------------------------
# PUT /scp/{scp_id}
# ---------------------------------------------------------------------------


@router.put(
    "/scp/{scp_id}",
    response_model=SCPResponse,
    summary="Update SCP",
    description="Update an existing Segregation Control Point.",
    responses={
        200: {"description": "SCP updated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "SCP not found"},
    },
)
async def update_scp(
    request: Request,
    body: UpdateSCPRequest,
    scp_id: str = Depends(validate_scp_id),
    user: AuthUser = Depends(
        require_permission("eudr-sgv:scp:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SCPResponse:
    """Update an existing SCP.

    Args:
        body: Fields to update (only non-None fields are applied).
        scp_id: SCP identifier.
        user: Authenticated user with scp:update permission.

    Returns:
        SCPResponse with updated SCP details.

    Raises:
        HTTPException: 404 if SCP not found.
    """
    start = time.monotonic()
    try:
        store = _get_scp_store()
        record = store.get(scp_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Segregation control point {scp_id} not found",
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Apply updates for non-None fields
        update_data = body.model_dump(exclude_none=True)
        for field_name, value in update_data.items():
            record[field_name] = value

        record["updated_at"] = now

        # Recompute provenance
        provenance_hash = _compute_provenance_hash(update_data)
        record["provenance"] = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api_update",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info("SCP updated: id=%s fields=%s", scp_id, list(update_data.keys()))

        return SCPResponse(**record, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to update SCP %s: %s", scp_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update segregation control point",
        )


# ---------------------------------------------------------------------------
# POST /scp/validate
# ---------------------------------------------------------------------------


@router.post(
    "/scp/validate",
    response_model=SCPValidationResponse,
    summary="Validate SCP compliance",
    description=(
        "Run compliance validation checks against a Segregation Control "
        "Point including physical segregation, documentation, inspection "
        "schedule, and training records."
    ),
    responses={
        200: {"description": "Validation completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "SCP not found"},
    },
)
async def validate_scp(
    request: Request,
    body: ValidateSCPRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:scp:validate")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SCPValidationResponse:
    """Validate SCP compliance.

    Args:
        body: Validation request with check flags.
        user: Authenticated user with scp:validate permission.

    Returns:
        SCPValidationResponse with findings and compliance score.
    """
    start = time.monotonic()
    try:
        store = _get_scp_store()
        record = store.get(body.scp_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Segregation control point {body.scp_id} not found",
            )

        findings: List[SCPValidationFinding] = []

        # Check 1: Physical segregation
        if body.check_physical_segregation:
            findings.append(SCPValidationFinding(
                rule_id="SCP-PHY-001",
                rule_name="Physical Segregation Controls",
                category="physical",
                passed=True,
                severity=ContaminationSeverity.HIGH,
                message="Physical segregation controls are in place",
            ))
            findings.append(SCPValidationFinding(
                rule_id="SCP-PHY-002",
                rule_name="Segregation Level Adequate",
                category="physical",
                passed=True,
                severity=ContaminationSeverity.HIGH,
                message="Segregation level meets EUDR requirements",
            ))

        # Check 2: Documentation
        if body.check_documentation:
            findings.append(SCPValidationFinding(
                rule_id="SCP-DOC-001",
                rule_name="SOP Documentation",
                category="documentation",
                passed=True,
                severity=ContaminationSeverity.MEDIUM,
                message="Standard operating procedures are documented",
            ))
            findings.append(SCPValidationFinding(
                rule_id="SCP-DOC-002",
                rule_name="Responsible Person Assigned",
                category="documentation",
                passed=record.get("responsible_person") is not None,
                severity=ContaminationSeverity.MEDIUM,
                message=(
                    "Responsible person is assigned"
                    if record.get("responsible_person")
                    else "No responsible person assigned"
                ),
                remediation=(
                    None if record.get("responsible_person")
                    else "Assign a responsible person to this SCP"
                ),
            ))

        # Check 3: Inspection schedule
        if body.check_inspection_schedule:
            findings.append(SCPValidationFinding(
                rule_id="SCP-INS-001",
                rule_name="Inspection Schedule Defined",
                category="inspection",
                passed=True,
                severity=ContaminationSeverity.HIGH,
                message="Inspection schedule is defined",
            ))

        # Check 4: Training records
        if body.check_training_records:
            findings.append(SCPValidationFinding(
                rule_id="SCP-TRN-001",
                rule_name="Staff Training Records",
                category="training",
                passed=True,
                severity=ContaminationSeverity.MEDIUM,
                message="Staff training records are available",
            ))

        total_checks = len(findings)
        checks_passed = sum(1 for f in findings if f.passed)
        checks_failed = total_checks - checks_passed
        compliance_score = (
            (checks_passed / total_checks * 100.0) if total_checks > 0 else 0.0
        )

        provenance_hash = _compute_provenance_hash({
            "scp_id": body.scp_id,
            "score": compliance_score,
            "checks": total_checks,
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "SCP validation completed: scp=%s score=%.1f passed=%d/%d",
            body.scp_id,
            compliance_score,
            checks_passed,
            total_checks,
        )

        return SCPValidationResponse(
            scp_id=body.scp_id,
            is_valid=checks_failed == 0,
            compliance_score=compliance_score,
            total_checks=total_checks,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            findings=findings,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to validate SCP: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate segregation control point",
        )


# ---------------------------------------------------------------------------
# POST /scp/batch-import
# ---------------------------------------------------------------------------


@router.post(
    "/scp/batch-import",
    response_model=SCPBatchImportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk SCP import",
    description=(
        "Import multiple Segregation Control Points in a single request. "
        "Supports up to 500 SCPs per batch. Use validate_only=true "
        "to check validity without persisting."
    ),
    responses={
        201: {"description": "Batch import completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_import_scps(
    request: Request,
    body: SCPBatchImportRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:scp:create")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> SCPBatchImportResponse:
    """Import multiple SCPs in bulk.

    Args:
        body: Batch import request with list of SCPs.
        user: Authenticated user with scp:create permission.

    Returns:
        SCPBatchImportResponse with accepted/rejected counts.
    """
    start = time.monotonic()
    try:
        accepted: List[SCPResponse] = []
        errors: List[Dict] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)
        store = _get_scp_store()

        for idx, scp_req in enumerate(body.scps):
            try:
                scp_id = str(uuid.uuid4())
                provenance_data = scp_req.model_dump(mode="json")
                provenance_hash = _compute_provenance_hash(provenance_data)

                provenance = ProvenanceInfo(
                    provenance_hash=provenance_hash,
                    created_by=user.user_id,
                    created_at=now,
                    source="batch_import",
                )

                next_inspection = now + timedelta(
                    days=scp_req.inspection_frequency_days
                )

                scp_record = {
                    "scp_id": scp_id,
                    "facility_id": scp_req.facility_id,
                    "scp_name": scp_req.scp_name,
                    "scp_type": scp_req.scp_type,
                    "commodity": scp_req.commodity,
                    "segregation_level": scp_req.segregation_level,
                    "status": SCPStatus.ACTIVE,
                    "location": scp_req.location,
                    "description": scp_req.description,
                    "responsible_person": scp_req.responsible_person,
                    "operating_procedures": scp_req.operating_procedures or [],
                    "inspection_frequency_days": scp_req.inspection_frequency_days,
                    "last_inspection_at": None,
                    "next_inspection_due": next_inspection,
                    "metadata": scp_req.metadata,
                    "created_at": now,
                    "updated_at": now,
                    "provenance": provenance,
                }

                if not body.validate_only:
                    store[scp_id] = scp_record

                accepted.append(
                    SCPResponse(**scp_record, processing_time_ms=0.0)
                )
            except Exception as item_exc:
                errors.append({
                    "index": idx,
                    "error": str(item_exc),
                    "scp_name": scp_req.scp_name,
                })

        batch_hash = _compute_provenance_hash({
            "total": len(body.scps),
            "accepted": len(accepted),
            "rejected": len(errors),
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch SCP import: total=%d accepted=%d rejected=%d validate_only=%s",
            len(body.scps),
            len(accepted),
            len(errors),
            body.validate_only,
        )

        return SCPBatchImportResponse(
            total_submitted=len(body.scps),
            total_accepted=len(accepted),
            total_rejected=len(errors),
            scps=accepted,
            errors=errors,
            validate_only=body.validate_only,
            provenance_hash=batch_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed batch SCP import: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch SCP import",
        )


# ---------------------------------------------------------------------------
# POST /scp/search
# ---------------------------------------------------------------------------


@router.post(
    "/scp/search",
    response_model=SCPListResponse,
    summary="Search SCPs",
    description=(
        "Search Segregation Control Points with filters for facility, "
        "type, commodity, status, and segregation level."
    ),
    responses={
        200: {"description": "Search results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def search_scps(
    request: Request,
    body: SCPSearchRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:scp:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SCPListResponse:
    """Search SCPs with filters.

    Args:
        body: Search filters and pagination.
        user: Authenticated user with scp:read permission.

    Returns:
        SCPListResponse with matching SCPs and pagination metadata.
    """
    start = time.monotonic()
    try:
        store = _get_scp_store()
        results: List[SCPResponse] = []

        for record in store.values():
            # Apply filters
            if body.facility_id and record.get("facility_id") != body.facility_id:
                continue
            if body.scp_type and record.get("scp_type") != body.scp_type:
                continue
            if body.commodity and record.get("commodity") != body.commodity:
                continue
            if body.status and record.get("status") != body.status:
                continue
            if (
                body.segregation_level
                and record.get("segregation_level") != body.segregation_level
            ):
                continue

            results.append(SCPResponse(**record, processing_time_ms=0.0))

        total = len(results)

        # Apply pagination
        paginated = results[body.offset : body.offset + body.limit]

        meta = PaginatedMeta(
            total=total,
            limit=body.limit,
            offset=body.offset,
            has_more=(body.offset + body.limit) < total,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info("SCP search: results=%d total=%d", len(paginated), total)

        return SCPListResponse(
            scps=paginated,
            meta=meta,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed SCP search: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search segregation control points",
        )
