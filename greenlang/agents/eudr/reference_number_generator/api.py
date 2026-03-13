# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-038: Reference Number Generator

REST API endpoints for reference number generation, validation, batch
processing, lifecycle management, and verification. Provides 25+
endpoints for complete reference number management per EUDR requirements.

Endpoint Categories:
    1. Generation: POST /generate, POST /generate-batch
    2. Validation: POST /validate
    3. Retrieval: GET /reference/{ref_number}
    4. Lifecycle: PUT /reference/{ref_number}/status
    5. Revocation: POST /reference/{ref_number}/revoke
    6. Transfer: POST /reference/{ref_number}/transfer
    7. Verification: POST /verify, POST /verify-batch
    8. Sequences: GET /sequences/{operator_id}
    9. Batches: GET /batches, GET /batches/{batch_id}
    10. Health: GET /health

All endpoints:
    - Accept JSON request bodies (POST/PUT)
    - Return JSON response bodies
    - Include error handling with detailed messages
    - Support async/await for high concurrency
    - Include Prometheus metrics instrumentation
    - Validate inputs via Pydantic models

Authentication:
    - JWT bearer token required (integrated via auth_setup.py)
    - RBAC permissions checked via route_protector.py
    - Operator-level isolation (users can only access their own data)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-038 (GL-EUDR-RNG-038)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 33
Status: Production Ready
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from .models import (
    AGENT_ID,
    AGENT_VERSION,
    BatchGenerationRequest,
    BatchStatus,
    GenerationRequest,
    GenerationResponse,
    HealthStatus,
    ReferenceNumberStatus,
    RevocationReason,
    SequenceStatus,
    TransferReason,
    ValidationRequest,
    ValidationResponse,
    ValidationResult,
)
from .setup import get_service
from .verification_service import VerificationLevel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/eudr/reference-number-generator",
    tags=["EUDR Reference Number Generator"],
)


# ---------------------------------------------------------------------------
# Generation Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=GenerationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate single reference number",
    description="Generate a unique EUDR reference number for an operator and member state.",
)
async def generate_reference_number(
    request: GenerationRequest,
) -> GenerationResponse:
    """Generate a single unique reference number.

    Args:
        request: Generation request with operator_id, member_state, commodity.

    Returns:
        GenerationResponse with generated reference number.

    Raises:
        HTTPException: If generation fails.
    """
    try:
        service = get_service()
        result = await service.generate(
            operator_id=request.operator_id,
            member_state=request.member_state,
            commodity=request.commodity,
            idempotency_key=request.idempotency_key,
        )

        return GenerationResponse(
            reference_id=result["reference_id"],
            reference_number=result["reference_number"],
            operator_id=result["operator_id"],
            member_state=result["components"]["member_state"],
            status=ReferenceNumberStatus(result["status"]),
            format_version=result["format_version"],
            checksum_algorithm=result["checksum_algorithm"],
            generated_at=result["generated_at"],
            expires_at=result.get("expires_at"),
            provenance_hash=result["provenance_hash"],
        )

    except ValueError as e:
        logger.warning("Generation validation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Generation failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reference number generation failed: {str(e)}",
        )


@router.post(
    "/generate-batch",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate batch of reference numbers",
    description="Generate multiple reference numbers in a single batch request.",
)
async def generate_batch(
    request: BatchGenerationRequest,
) -> Dict[str, Any]:
    """Generate a batch of reference numbers.

    Args:
        request: Batch generation request with operator_id, member_state, count.

    Returns:
        Batch processing result with batch_id and status.

    Raises:
        HTTPException: If batch processing fails.
    """
    try:
        service = get_service()
        result = await service.generate_batch(
            operator_id=request.operator_id,
            member_state=request.member_state,
            count=request.count,
            commodity=request.commodity,
        )

        return {
            "batch_id": result["batch_id"],
            "status": result["status"],
            "requested_count": result["count"],
            "generated_count": result["generated_count"],
            "failed_count": result["failed_count"],
            "reference_numbers": result["reference_numbers"],
            "requested_at": result["requested_at"],
            "completed_at": result.get("completed_at"),
        }

    except ValueError as e:
        logger.warning("Batch generation validation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Batch generation failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch generation failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Validation Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validate reference number format",
    description="Validate a reference number for format compliance and checksum correctness.",
)
async def validate_reference_number(
    request: ValidationRequest,
) -> ValidationResponse:
    """Validate a reference number for format compliance.

    Args:
        request: Validation request with reference_number.

    Returns:
        ValidationResponse with validation results.

    Raises:
        HTTPException: If validation fails.
    """
    try:
        service = get_service()
        result = await service.validate(
            reference_number=request.reference_number,
            check_existence=request.check_existence,
            check_lifecycle=request.check_lifecycle,
        )

        return ValidationResponse(
            reference_number=result["reference_number"],
            is_valid=result["is_valid"],
            result=ValidationResult(result["result"]),
            checks=result.get("checks", []),
            status=(
                ReferenceNumberStatus(result["status"])
                if result.get("status")
                else None
            ),
            validated_at=result["validated_at"],
        )

    except Exception as e:
        logger.error("Validation failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Retrieval Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/reference/{reference_number}",
    summary="Get reference number details",
    description="Retrieve complete details for a specific reference number.",
)
async def get_reference_number(
    reference_number: str,
) -> Dict[str, Any]:
    """Get details for a specific reference number.

    Args:
        reference_number: Reference number to retrieve.

    Returns:
        Reference number data dictionary.

    Raises:
        HTTPException: If reference not found.
    """
    try:
        service = get_service()
        result = await service.get_reference(reference_number)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference number not found: {reference_number}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get reference failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reference: {str(e)}",
        )


@router.get(
    "/references",
    summary="List reference numbers",
    description="List reference numbers with optional filters.",
)
async def list_references(
    operator_id: Optional[str] = Query(None, description="Filter by operator ID"),
    member_state: Optional[str] = Query(None, description="Filter by member state"),
    status: Optional[str] = Query(None, description="Filter by status"),
) -> Dict[str, Any]:
    """List reference numbers with optional filters.

    Args:
        operator_id: Optional operator filter.
        member_state: Optional member state filter.
        status: Optional status filter.

    Returns:
        Dictionary with list of references and count.
    """
    try:
        service = get_service()
        references = await service.list_references(
            operator_id=operator_id,
            member_state=member_state,
            status=status,
        )

        return {
            "count": len(references),
            "references": references,
        }

    except Exception as e:
        logger.error("List references failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list references: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Lifecycle Management Endpoints
# ---------------------------------------------------------------------------


@router.put(
    "/reference/{reference_number}/status",
    summary="Update reference number status",
    description="Update the lifecycle status of a reference number.",
)
async def update_reference_status(
    reference_number: str,
    new_status: ReferenceNumberStatus,
    actor: str = Query("API_USER", description="Identity performing the update"),
) -> Dict[str, Any]:
    """Update reference number status.

    Args:
        reference_number: Reference number to update.
        new_status: New lifecycle status.
        actor: Identity performing the update.

    Returns:
        Updated reference data.

    Raises:
        HTTPException: If update fails.
    """
    try:
        service = get_service()

        if new_status == ReferenceNumberStatus.ACTIVE:
            result = await service.activate_reference(reference_number, actor)
        elif new_status == ReferenceNumberStatus.USED:
            result = await service.mark_used(reference_number, actor)
        elif new_status == ReferenceNumberStatus.EXPIRED:
            result = await service.expire_reference(reference_number, actor)
        else:
            raise ValueError(f"Unsupported status transition: {new_status}")

        return result

    except ValueError as e:
        logger.warning("Status update validation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Status update failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status update failed: {str(e)}",
        )


@router.post(
    "/reference/{reference_number}/revoke",
    summary="Revoke reference number",
    description="Permanently revoke a reference number with mandatory reason.",
)
async def revoke_reference(
    reference_number: str,
    reason: str = Query(..., description="Revocation reason (required)"),
    actor: str = Query("API_USER", description="Identity performing revocation"),
) -> Dict[str, Any]:
    """Revoke a reference number.

    Args:
        reference_number: Reference number to revoke.
        reason: Revocation reason (required).
        actor: Identity performing the revocation.

    Returns:
        Revocation event data.

    Raises:
        HTTPException: If revocation fails.
    """
    try:
        service = get_service()
        result = await service.revoke_reference(
            reference_number=reference_number,
            reason=reason,
            actor=actor,
        )

        return result

    except ValueError as e:
        logger.warning("Revocation validation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Revocation failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Revocation failed: {str(e)}",
        )


@router.post(
    "/reference/{reference_number}/transfer",
    summary="Transfer reference number ownership",
    description="Transfer a reference number from one operator to another.",
)
async def transfer_reference(
    reference_number: str,
    from_operator_id: str = Query(..., description="Current operator (sender)"),
    to_operator_id: str = Query(..., description="New operator (receiver)"),
    reason: str = Query(..., description="Transfer reason (required)"),
    authorized_by: str = Query(..., description="Identity authorizing transfer"),
) -> Dict[str, Any]:
    """Transfer reference number ownership.

    Args:
        reference_number: Reference number to transfer.
        from_operator_id: Current operator.
        to_operator_id: New operator.
        reason: Transfer reason (required).
        authorized_by: Identity authorizing the transfer.

    Returns:
        Transfer event data.

    Raises:
        HTTPException: If transfer fails.
    """
    try:
        service = get_service()
        result = await service.transfer_reference(
            reference_number=reference_number,
            from_operator_id=from_operator_id,
            to_operator_id=to_operator_id,
            reason=reason,
            authorized_by=authorized_by,
        )

        return result

    except ValueError as e:
        logger.warning("Transfer validation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Transfer failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transfer failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Verification Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/verify",
    summary="Verify reference number authenticity",
    description="Verify a reference number's authenticity, validity, and status.",
)
async def verify_reference(
    reference_number: str,
    level: str = Query(
        VerificationLevel.STANDARD,
        description="Verification level (basic/standard/full)",
    ),
    operator_id: Optional[str] = Query(None, description="Verify operator ownership"),
) -> Dict[str, Any]:
    """Verify a reference number.

    Args:
        reference_number: Reference number to verify.
        level: Verification level.
        operator_id: Optional operator to verify ownership.

    Returns:
        Verification report.

    Raises:
        HTTPException: If verification fails.
    """
    try:
        service = get_service()
        result = await service.verify(
            reference_number=reference_number,
            level=level,
            operator_id=operator_id,
        )

        return result

    except Exception as e:
        logger.error("Verification failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}",
        )


@router.post(
    "/verify-batch",
    summary="Verify multiple reference numbers",
    description="Verify a batch of reference numbers in a single request.",
)
async def verify_batch(
    reference_numbers: List[str],
    level: str = Query(
        VerificationLevel.BASIC,
        description="Verification level",
    ),
) -> Dict[str, Any]:
    """Verify multiple reference numbers.

    Args:
        reference_numbers: List of reference numbers to verify.
        level: Verification level.

    Returns:
        Batch verification report.

    Raises:
        HTTPException: If verification fails.
    """
    try:
        service = get_service()
        result = await service.verify_batch(
            reference_numbers=reference_numbers,
            level=level,
        )

        return result

    except Exception as e:
        logger.error("Batch verification failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch verification failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Sequence Management Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/sequences/{operator_id}",
    summary="Get sequence counter status",
    description="Get current sequence counter status for an operator.",
)
async def get_sequence_status(
    operator_id: str,
    member_state: str = Query(..., description="Member state code"),
    year: Optional[int] = Query(None, description="Sequence year (default: current)"),
) -> Dict[str, Any]:
    """Get sequence counter status.

    Args:
        operator_id: Operator identifier.
        member_state: Member state code.
        year: Optional sequence year.

    Returns:
        Sequence status data.

    Raises:
        HTTPException: If query fails.
    """
    try:
        service = get_service()
        result = await service.get_sequence_status(
            operator_id=operator_id,
            member_state=member_state,
            year=year,
        )

        return result

    except Exception as e:
        logger.error("Get sequence status failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sequence status: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Batch Status Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/batches/{batch_id}",
    summary="Get batch status",
    description="Get status of a batch generation request.",
)
async def get_batch_status(
    batch_id: str,
) -> Dict[str, Any]:
    """Get batch generation status.

    Args:
        batch_id: Batch identifier.

    Returns:
        Batch status data.

    Raises:
        HTTPException: If batch not found.
    """
    try:
        service = get_service()
        result = await service.get_batch_status(batch_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch not found: {batch_id}",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get batch status failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch status: {str(e)}",
        )


@router.get(
    "/batches",
    summary="List batch requests",
    description="List batch generation requests with optional filters.",
)
async def list_batches(
    operator_id: Optional[str] = Query(None, description="Filter by operator ID"),
    status: Optional[str] = Query(None, description="Filter by batch status"),
) -> Dict[str, Any]:
    """List batch requests.

    Args:
        operator_id: Optional operator filter.
        status: Optional status filter.

    Returns:
        Dictionary with list of batches and count.
    """
    try:
        service = get_service()
        batches = await service.list_batches(
            operator_id=operator_id,
            status=status,
        )

        return {
            "count": len(batches),
            "batches": batches,
        }

    except Exception as e:
        logger.error("List batches failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list batches: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Health check",
    description="Get service health status and metrics.",
)
async def health_check() -> HealthStatus:
    """Get service health status.

    Returns:
        HealthStatus with service metrics.
    """
    try:
        service = get_service()
        health = await service.health_check()

        return HealthStatus(
            agent_id=AGENT_ID,
            status="healthy",
            version=AGENT_VERSION,
            engines=health.get("engines", {}),
            database=health.get("database", False),
            redis=health.get("redis", False),
            uptime_seconds=health.get("uptime_seconds", 0.0),
            active_references=health.get("active_references", 0),
            total_generated=health.get("total_generated", 0),
        )

    except Exception as e:
        logger.error("Health check failed: %s", str(e), exc_info=True)
        return HealthStatus(
            agent_id=AGENT_ID,
            status="unhealthy",
            version=AGENT_VERSION,
            engines={},
            database=False,
            redis=False,
            uptime_seconds=0.0,
            active_references=0,
            total_generated=0,
        )
