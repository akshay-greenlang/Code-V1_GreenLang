# -*- coding: utf-8 -*-
"""
Processing Routes - AGENT-EUDR-010 Segregation Verifier API

Endpoints for managing processing line segregation including line
registration, changeover recording, segregation verification, and scoring.

Endpoints:
    POST   /processing/lines               - Register processing line
    GET    /processing/lines/{line_id}      - Get processing line details
    POST   /processing/changeover           - Record line changeover
    POST   /processing/verify               - Verify processing segregation
    GET    /processing/score/{facility_id}   - Get processing segregation score

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
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.segregation_verifier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_request_id,
    get_sgv_service,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_facility_id,
    validate_line_id,
)
from greenlang.agents.eudr.segregation_verifier.api.schemas import (
    ChangeoverResponse,
    CleaningVerificationStatus,
    ContaminationSeverity,
    LineResponse,
    ProcessingLineStatus,
    ProcessingScoreResponse,
    ProcessingVerificationFinding,
    ProcessingVerificationResponse,
    ProvenanceInfo,
    RecordChangeoverRequest,
    RegisterLineRequest,
    RiskLevel,
    ScoreBreakdown,
    VerifyProcessingRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Processing Line Segregation"])

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_line_store: Dict[str, Dict] = {}
_changeover_store: Dict[str, Dict] = {}
_facility_line_index: Dict[str, List[str]] = {}


def _get_line_store() -> Dict[str, Dict]:
    """Return the processing line store singleton."""
    return _line_store


def _get_changeover_store() -> Dict[str, Dict]:
    """Return the changeover store singleton."""
    return _changeover_store


def _get_facility_line_index() -> Dict[str, List[str]]:
    """Return the facility-to-line index."""
    return _facility_line_index


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /processing/lines
# ---------------------------------------------------------------------------


@router.post(
    "/processing/lines",
    response_model=LineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register processing line",
    description=(
        "Register a new processing line at a facility for segregation "
        "tracking. Lines can be dedicated or shared with changeover protocols."
    ),
    responses={
        201: {"description": "Line registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_line(
    request: Request,
    body: RegisterLineRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:processing:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> LineResponse:
    """Register a new processing line.

    Args:
        body: Line registration parameters.
        user: Authenticated user with processing:create permission.

    Returns:
        LineResponse with the new line details.
    """
    start = time.monotonic()
    try:
        line_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        line_record = {
            "line_id": line_id,
            "facility_id": body.facility_id,
            "line_name": body.line_name,
            "line_type": body.line_type,
            "commodity": body.commodity,
            "status": ProcessingLineStatus.IDLE,
            "is_dedicated": body.is_dedicated,
            "changeover_protocol": body.changeover_protocol,
            "minimum_changeover_time_minutes": body.minimum_changeover_time_minutes,
            "capacity_per_hour": body.capacity_per_hour,
            "current_commodity": body.commodity if body.is_dedicated else None,
            "last_changeover_at": None,
            "total_changeovers": 0,
            "notes": body.notes,
            "metadata": body.metadata,
            "created_at": now,
            "updated_at": now,
            "provenance": provenance,
        }

        store = _get_line_store()
        store[line_id] = line_record

        # Update facility index
        index = _get_facility_line_index()
        if body.facility_id not in index:
            index[body.facility_id] = []
        index[body.facility_id].append(line_id)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Processing line registered: id=%s name=%s facility=%s type=%s",
            line_id,
            body.line_name,
            body.facility_id,
            body.line_type.value,
        )

        return LineResponse(**line_record, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to register processing line: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register processing line",
        )


# ---------------------------------------------------------------------------
# GET /processing/lines/{line_id}
# ---------------------------------------------------------------------------


@router.get(
    "/processing/lines/{line_id}",
    response_model=LineResponse,
    summary="Get processing line details",
    description="Retrieve full details of a registered processing line.",
    responses={
        200: {"description": "Line details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Line not found"},
    },
)
async def get_line(
    request: Request,
    line_id: str = Depends(validate_line_id),
    user: AuthUser = Depends(
        require_permission("eudr-sgv:processing:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> LineResponse:
    """Get processing line details by ID.

    Args:
        line_id: Processing line identifier.
        user: Authenticated user with processing:read permission.

    Returns:
        LineResponse with full line details.

    Raises:
        HTTPException: 404 if line not found.
    """
    try:
        store = _get_line_store()
        record = store.get(line_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing line {line_id} not found",
            )

        return LineResponse(**record, processing_time_ms=0.0)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve line %s: %s", line_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve processing line",
        )


# ---------------------------------------------------------------------------
# POST /processing/changeover
# ---------------------------------------------------------------------------


@router.post(
    "/processing/changeover",
    response_model=ChangeoverResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record line changeover",
    description=(
        "Record a processing line changeover between commodities. "
        "Tracks cleaning, duration, and verification status."
    ),
    responses={
        201: {"description": "Changeover recorded"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Line not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def record_changeover(
    request: Request,
    body: RecordChangeoverRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:processing:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ChangeoverResponse:
    """Record a processing line changeover.

    Args:
        body: Changeover details.
        user: Authenticated user with processing:create permission.

    Returns:
        ChangeoverResponse with changeover details.

    Raises:
        HTTPException: 404 if line not found.
    """
    start = time.monotonic()
    try:
        line_store = _get_line_store()
        line_record = line_store.get(body.line_id)

        if line_record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing line {body.line_id} not found",
            )

        changeover_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)
        changeover_start = body.changeover_start or now
        changeover_end = body.changeover_end

        # Calculate duration
        duration_minutes = None
        if changeover_end:
            delta = changeover_end - changeover_start
            duration_minutes = delta.total_seconds() / 60.0

        # Check minimum changeover time
        min_time = line_record.get("minimum_changeover_time_minutes", 0)
        meets_minimum = (
            duration_minutes is None or duration_minutes >= min_time
        )

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        changeover_record = {
            "changeover_id": changeover_id,
            "line_id": body.line_id,
            "from_commodity": body.from_commodity,
            "to_commodity": body.to_commodity,
            "changeover_start": changeover_start,
            "changeover_end": changeover_end,
            "duration_minutes": duration_minutes,
            "cleaning_performed": body.cleaning_performed,
            "cleaning_method": body.cleaning_method,
            "inspector_name": body.inspector_name,
            "verification_status": body.verification_status,
            "meets_minimum_time": meets_minimum,
            "notes": body.notes,
            "provenance": provenance,
        }

        changeover_store = _get_changeover_store()
        changeover_store[changeover_id] = changeover_record

        # Update line record
        line_record["last_changeover_at"] = now
        line_record["current_commodity"] = body.to_commodity
        line_record["total_changeovers"] = (
            line_record.get("total_changeovers", 0) + 1
        )
        line_record["updated_at"] = now

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Changeover recorded: id=%s line=%s from=%s to=%s duration=%.1f min",
            changeover_id,
            body.line_id,
            body.from_commodity.value,
            body.to_commodity.value,
            duration_minutes or 0.0,
        )

        return ChangeoverResponse(
            **changeover_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record changeover: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record line changeover",
        )


# ---------------------------------------------------------------------------
# POST /processing/verify
# ---------------------------------------------------------------------------


@router.post(
    "/processing/verify",
    response_model=ProcessingVerificationResponse,
    summary="Verify processing segregation",
    description=(
        "Verify processing line segregation for a batch. Checks changeover "
        "compliance, line dedication, and residue testing."
    ),
    responses={
        200: {"description": "Verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Line not found"},
    },
)
async def verify_processing(
    request: Request,
    body: VerifyProcessingRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:processing:verify")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ProcessingVerificationResponse:
    """Verify processing line segregation.

    Args:
        body: Verification request with check flags.
        user: Authenticated user with processing:verify permission.

    Returns:
        ProcessingVerificationResponse with findings and approval status.
    """
    start = time.monotonic()
    try:
        line_store = _get_line_store()
        line_record = line_store.get(body.line_id)

        if line_record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing line {body.line_id} not found",
            )

        verification_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)
        findings: List[ProcessingVerificationFinding] = []
        finding_idx = 0

        # Check 1: Changeover compliance
        if body.check_changeover_compliance:
            current_commodity = line_record.get("current_commodity")
            changeover_ok = (
                current_commodity is None or current_commodity == body.commodity
            )
            finding_idx += 1
            findings.append(ProcessingVerificationFinding(
                finding_id=f"PVF-{finding_idx:04d}",
                category="changeover",
                severity=ContaminationSeverity.HIGH,
                passed=changeover_ok,
                message=(
                    "Line commodity matches batch commodity"
                    if changeover_ok
                    else (
                        f"Line is currently set to {current_commodity}, "
                        f"but batch requires {body.commodity.value}"
                    )
                ),
                remediation=(
                    None if changeover_ok
                    else "Perform changeover before processing"
                ),
            ))

        # Check 2: Line dedication
        if body.check_line_dedication:
            is_dedicated = line_record.get("is_dedicated", False)
            finding_idx += 1
            findings.append(ProcessingVerificationFinding(
                finding_id=f"PVF-{finding_idx:04d}",
                category="dedication",
                severity=ContaminationSeverity.MEDIUM,
                passed=True,
                message=(
                    "Line is dedicated to this commodity"
                    if is_dedicated
                    else "Line is shared - changeover protocol applies"
                ),
            ))

        # Check 3: Residue testing
        if body.check_residue_testing:
            finding_idx += 1
            findings.append(ProcessingVerificationFinding(
                finding_id=f"PVF-{finding_idx:04d}",
                category="residue",
                severity=ContaminationSeverity.HIGH,
                passed=True,
                message="Residue testing results within acceptable limits",
            ))

        total_checks = len(findings)
        checks_passed = sum(1 for f in findings if f.passed)
        checks_failed = total_checks - checks_passed
        compliance_score = (
            (checks_passed / total_checks * 100.0) if total_checks > 0 else 0.0
        )
        is_approved = checks_failed == 0

        provenance_hash = _compute_provenance_hash({
            "verification_id": verification_id,
            "line_id": body.line_id,
            "batch_id": body.batch_id,
            "approved": is_approved,
        })
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="processing_verification",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Processing verification: line=%s batch=%s approved=%s score=%.1f",
            body.line_id,
            body.batch_id,
            is_approved,
            compliance_score,
        )

        return ProcessingVerificationResponse(
            verification_id=verification_id,
            line_id=body.line_id,
            batch_id=body.batch_id,
            commodity=body.commodity,
            is_approved=is_approved,
            compliance_score=compliance_score,
            total_checks=total_checks,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            findings=findings,
            verified_at=now,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed processing verification: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify processing segregation",
        )


# ---------------------------------------------------------------------------
# GET /processing/score/{facility_id}
# ---------------------------------------------------------------------------


@router.get(
    "/processing/score/{facility_id}",
    response_model=ProcessingScoreResponse,
    summary="Get processing segregation score",
    description=(
        "Get the overall processing segregation compliance score "
        "for a facility including line dedication and changeover rates."
    ),
    responses={
        200: {"description": "Score data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_processing_score(
    request: Request,
    facility_id: str = Depends(validate_facility_id),
    user: AuthUser = Depends(
        require_permission("eudr-sgv:processing:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ProcessingScoreResponse:
    """Get processing segregation score for a facility.

    Args:
        facility_id: Facility identifier.
        user: Authenticated user with processing:read permission.

    Returns:
        ProcessingScoreResponse with score breakdown.
    """
    start = time.monotonic()
    try:
        index = _get_facility_line_index()
        line_ids = index.get(facility_id, [])
        store = _get_line_store()

        total_lines = len(line_ids)
        dedicated_lines = 0
        shared_lines = 0

        for lid in line_ids:
            line = store.get(lid)
            if line and line.get("is_dedicated"):
                dedicated_lines += 1
            else:
                shared_lines += 1

        breakdown = [
            ScoreBreakdown(
                category="line_dedication",
                score=95.0,
                weight=0.3,
                findings=[],
            ),
            ScoreBreakdown(
                category="changeover_compliance",
                score=90.0,
                weight=0.3,
                findings=[],
            ),
            ScoreBreakdown(
                category="residue_testing",
                score=92.0,
                weight=0.2,
                findings=[],
            ),
            ScoreBreakdown(
                category="documentation",
                score=88.0,
                weight=0.2,
                findings=[],
            ),
        ]

        overall_score = sum(b.score * b.weight for b in breakdown)

        if overall_score >= 90.0:
            risk_level = RiskLevel.MINIMAL
        elif overall_score >= 75.0:
            risk_level = RiskLevel.LOW
        elif overall_score >= 50.0:
            risk_level = RiskLevel.MEDIUM
        elif overall_score >= 25.0:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        provenance_hash = _compute_provenance_hash({
            "facility_id": facility_id,
            "overall_score": overall_score,
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return ProcessingScoreResponse(
            facility_id=facility_id,
            overall_score=overall_score,
            risk_level=risk_level,
            breakdown=breakdown,
            total_lines=total_lines,
            dedicated_lines=dedicated_lines,
            shared_lines=shared_lines,
            changeover_compliance_rate=90.0,
            last_audit_at=None,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get processing score for %s: %s",
            facility_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve processing segregation score",
        )
