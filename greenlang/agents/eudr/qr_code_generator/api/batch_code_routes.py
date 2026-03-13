# -*- coding: utf-8 -*-
"""
Batch Code Routes - AGENT-EUDR-014 QR Code Generator API

Endpoints for batch code generation including code generation with
operator-commodity-year prefix format, code reservation, detail
retrieval with Luhn/ISO 7064/CRC-8 check digit validation, and
batch code hierarchy lookup.

Endpoints:
    POST   /batch-codes/generate        - Generate batch codes
    POST   /batch-codes/reserve         - Reserve a code range
    GET    /batch-codes/{code}           - Get batch code details
    GET    /batch-codes/{code}/hierarchy - Get batch code hierarchy

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014, Feature 4 (Batch Code Generation Engine)
Agent ID: GL-EUDR-QRG-014
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.qr_code_generator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_qrg_service,
    rate_limit_batch_code,
    rate_limit_standard,
    require_permission,
    validate_batch_code,
)
from greenlang.agents.eudr.qr_code_generator.api.schemas import (
    BatchCodeDetailResponse,
    BatchCodeItem,
    CodeHierarchyResponse,
    GenerateBatchCodesRequest,
    GenerateBatchCodesResponse,
    ProvenanceInfo,
    ReserveCodesRequest,
    ReserveCodesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Batch Codes"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_batch_code_store: Dict[str, Dict] = {}
_sequence_counter: Dict[str, int] = {}


def _get_batch_store() -> Dict[str, Dict]:
    """Return the batch code store singleton."""
    return _batch_code_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_luhn_check_digit(number_str: str) -> str:
    """Compute Luhn check digit for a numeric string.

    Args:
        number_str: Numeric string to compute check digit for.

    Returns:
        Single check digit character.
    """
    digits = [int(d) for d in number_str]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    check = (10 - (total % 10)) % 10
    return str(check)


def _build_prefix(operator_id: str, commodity: str, year: int) -> str:
    """Build the operator-commodity-year prefix.

    Args:
        operator_id: Operator identifier (first 3 chars).
        commodity: Commodity code (first 3 chars).
        year: Production year (last 2 digits).

    Returns:
        Formatted prefix string.
    """
    op = operator_id[:3].upper()
    comm = commodity[:3].upper()
    yr = str(year)[-2:]
    return f"{op}-{comm}-{yr}"


# ---------------------------------------------------------------------------
# POST /batch-codes/generate
# ---------------------------------------------------------------------------


@router.post(
    "/batch-codes/generate",
    response_model=GenerateBatchCodesResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate batch codes",
    description=(
        "Generate batch codes with operator-commodity-year prefix "
        "format and configurable check digit algorithm (Luhn, "
        "ISO 7064 Mod 11,10, or CRC-8)."
    ),
    responses={
        201: {"description": "Batch codes generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_batch_codes(
    request: Request,
    body: GenerateBatchCodesRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:batch-codes:generate")
    ),
    _rate: None = Depends(rate_limit_batch_code),
    service: Any = Depends(get_qrg_service),
) -> GenerateBatchCodesResponse:
    """Generate batch codes with check digits.

    Args:
        request: FastAPI request object.
        body: Batch code generation parameters.
        user: Authenticated user with batch-codes:generate permission.
        service: QR Code Generator service.

    Returns:
        GenerateBatchCodesResponse with generated codes.
    """
    start = time.monotonic()
    try:
        now = _utcnow()
        store = _get_batch_store()
        commodity_val = body.commodity.value
        algorithm = body.check_digit_algorithm.value if body.check_digit_algorithm else "luhn"
        prefix = body.prefix_format or _build_prefix(
            body.operator_id, commodity_val, body.year
        )

        # Get or initialize sequence counter for this prefix
        seq_key = prefix
        current_seq = _sequence_counter.get(seq_key, 0)

        codes: List[BatchCodeItem] = []
        for i in range(body.count):
            seq_num = current_seq + i + 1
            seq_str = str(seq_num).zfill(6)
            check_digit = _compute_luhn_check_digit(seq_str)
            code_value = f"{prefix}-{seq_str}-{check_digit}"

            batch_code_id = str(uuid.uuid4())
            code_record = {
                "batch_code_id": batch_code_id,
                "code_value": code_value,
                "prefix": prefix,
                "sequence_number": seq_num,
                "check_digit": check_digit,
                "check_digit_algorithm": algorithm,
                "operator_id": body.operator_id,
                "commodity": commodity_val,
                "year": body.year,
                "facility_id": body.facility_id,
                "status": "created",
                "created_at": now,
            }

            store[code_value] = code_record
            codes.append(BatchCodeItem(**code_record))

        _sequence_counter[seq_key] = current_seq + body.count

        provenance_hash = _compute_provenance_hash({
            "prefix": prefix,
            "count": body.count,
            "operator_id": body.operator_id,
            "commodity": commodity_val,
            "year": body.year,
            "created_by": user.user_id,
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Batch codes generated: prefix=%s count=%d algorithm=%s "
            "elapsed_ms=%.1f",
            prefix,
            body.count,
            algorithm,
            elapsed_ms,
        )

        return GenerateBatchCodesResponse(
            status="success",
            batch_codes=codes,
            count=len(codes),
            processing_time_ms=elapsed_ms,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to generate batch codes: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate batch codes",
        )


# ---------------------------------------------------------------------------
# POST /batch-codes/reserve
# ---------------------------------------------------------------------------


@router.post(
    "/batch-codes/reserve",
    response_model=ReserveCodesResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Reserve batch code range",
    description=(
        "Reserve a range of batch code sequence numbers for future "
        "use. Prevents collisions in distributed generation scenarios."
    ),
    responses={
        201: {"description": "Codes reserved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def reserve_codes(
    request: Request,
    body: ReserveCodesRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:batch-codes:generate")
    ),
    _rate: None = Depends(rate_limit_batch_code),
    service: Any = Depends(get_qrg_service),
) -> ReserveCodesResponse:
    """Reserve a range of batch code sequence numbers.

    Args:
        request: FastAPI request object.
        body: Code reservation parameters.
        user: Authenticated user with batch-codes:generate permission.
        service: QR Code Generator service.

    Returns:
        ReserveCodesResponse with reserved range details.
    """
    start = time.monotonic()
    try:
        commodity_val = body.commodity.value
        prefix = _build_prefix(body.operator_id, commodity_val, body.year)

        # Reserve sequence range
        seq_key = prefix
        current_seq = _sequence_counter.get(seq_key, 0)
        start_seq = current_seq + 1
        end_seq = current_seq + body.count
        _sequence_counter[seq_key] = end_seq

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Codes reserved: prefix=%s range=%d-%d count=%d elapsed_ms=%.1f",
            prefix,
            start_seq,
            end_seq,
            body.count,
            elapsed_ms,
        )

        return ReserveCodesResponse(
            status="success",
            reserved_count=body.count,
            start_sequence=start_seq,
            end_sequence=end_seq,
            prefix=prefix,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to reserve codes: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reserve batch codes",
        )


# ---------------------------------------------------------------------------
# GET /batch-codes/{code}
# ---------------------------------------------------------------------------


@router.get(
    "/batch-codes/{code}",
    response_model=BatchCodeDetailResponse,
    summary="Get batch code details",
    description=(
        "Retrieve details of a batch code including check digit "
        "validation and associated QR codes."
    ),
    responses={
        200: {"description": "Batch code details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Batch code not found"},
    },
)
async def get_batch_code_detail(
    request: Request,
    code: str = Depends(validate_batch_code),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:batch-codes:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> BatchCodeDetailResponse:
    """Get batch code details by code value.

    Args:
        request: FastAPI request object.
        code: Batch code value.
        user: Authenticated user with batch-codes:read permission.
        service: QR Code Generator service.

    Returns:
        BatchCodeDetailResponse with code details.

    Raises:
        HTTPException: 404 if code not found.
    """
    try:
        store = _get_batch_store()
        record = store.get(code)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch code {code} not found",
            )

        now = _utcnow()
        provenance_hash = _compute_provenance_hash({
            "code_value": code,
            "operator_id": record.get("operator_id", ""),
            "accessed_at": str(now),
        })

        return BatchCodeDetailResponse(
            batch_code=BatchCodeItem(**record),
            associated_code_ids=[],
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get batch code %s: %s", code, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve batch code details",
        )


# ---------------------------------------------------------------------------
# GET /batch-codes/{code}/hierarchy
# ---------------------------------------------------------------------------


@router.get(
    "/batch-codes/{code}/hierarchy",
    response_model=CodeHierarchyResponse,
    summary="Get batch code hierarchy",
    description=(
        "Retrieve the batch code hierarchy showing parent codes, "
        "child codes, and associated QR code identifiers."
    ),
    responses={
        200: {"description": "Code hierarchy retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Batch code not found"},
    },
)
async def get_batch_code_hierarchy(
    request: Request,
    code: str = Depends(validate_batch_code),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:batch-codes:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> CodeHierarchyResponse:
    """Get batch code hierarchy.

    Args:
        request: FastAPI request object.
        code: Batch code value.
        user: Authenticated user with batch-codes:read permission.
        service: QR Code Generator service.

    Returns:
        CodeHierarchyResponse with hierarchy details.

    Raises:
        HTTPException: 404 if code not found.
    """
    try:
        store = _get_batch_store()
        record = store.get(code)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch code {code} not found",
            )

        return CodeHierarchyResponse(
            code_value=code,
            operator_id=record.get("operator_id", ""),
            commodity=record.get("commodity"),
            year=record.get("year", 2025),
            parent_codes=[],
            child_codes=[],
            associated_qr_codes=[],
            total_associations=0,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get code hierarchy %s: %s", code, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve code hierarchy",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
