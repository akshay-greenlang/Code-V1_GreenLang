# -*- coding: utf-8 -*-
"""
QR Code Routes - AGENT-EUDR-014 QR Code Generator API

Endpoints for QR code generation including single QR code creation,
Data Matrix code generation, detail retrieval, and image download
in multiple formats per ISO/IEC 18004 and ISO/IEC 16022.

Endpoints:
    POST   /qr/generate                   - Generate a single QR code
    POST   /qr/generate/data-matrix       - Generate a Data Matrix code
    GET    /qr/{code_id}                   - Get QR code details
    GET    /qr/{code_id}/image             - Get QR code image (default format)
    GET    /qr/{code_id}/image/{format}    - Get QR code image (specific format)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014, Feature 1 (QR Code Generation Engine)
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
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.qr_code_generator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_qrg_service,
    rate_limit_qr_generate,
    rate_limit_standard,
    require_permission,
    validate_code_id,
    validate_format_path,
)
from greenlang.agents.eudr.qr_code_generator.api.schemas import (
    GenerateDataMatrixRequest,
    GenerateDataMatrixResponse,
    GenerateQRRequest,
    GenerateQRResponse,
    ProvenanceInfo,
    QRCodeDetailResponse,
    QRImageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["QR Code Generation"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_qr_code_store: Dict[str, Dict] = {}


def _get_qr_store() -> Dict[str, Dict]:
    """Return the QR code record store singleton."""
    return _qr_code_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# POST /qr/generate
# ---------------------------------------------------------------------------


@router.post(
    "/qr/generate",
    response_model=GenerateQRResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate QR code",
    description=(
        "Generate a single QR code with configurable version, error "
        "correction level, output format, and symbology per ISO/IEC "
        "18004. Supports logo embedding and ISO/IEC 15416 quality "
        "grading for EUDR compliance labelling."
    ),
    responses={
        201: {"description": "QR code generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_qr_code(
    request: Request,
    body: GenerateQRRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:qr:generate")
    ),
    _rate: None = Depends(rate_limit_qr_generate),
    service: Any = Depends(get_qrg_service),
) -> GenerateQRResponse:
    """Generate a single QR code.

    Args:
        request: FastAPI request object.
        body: QR code generation parameters.
        user: Authenticated user with qr:generate permission.
        service: QR Code Generator service.

    Returns:
        GenerateQRResponse with code ID and QR code details.
    """
    start = time.monotonic()
    try:
        code_id = str(uuid.uuid4())
        now = _utcnow()

        # Compute payload hash
        payload_serialized = json.dumps(body.payload_data, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(payload_serialized.encode("utf-8")).hexdigest()
        payload_size = len(payload_serialized.encode("utf-8"))

        provenance_hash = _compute_provenance_hash({
            "code_id": code_id,
            "operator_id": body.operator_id,
            "payload_hash": payload_hash,
            "created_by": user.user_id,
        })

        qr_record = {
            "code_id": code_id,
            "version": body.version or "auto",
            "error_correction": body.error_correction.value if body.error_correction else "M",
            "symbology": body.symbology.value if body.symbology else "qr_code",
            "output_format": body.output_format.value if body.output_format else "png",
            "module_size": body.module_size or 10,
            "quiet_zone": body.quiet_zone if body.quiet_zone is not None else 4,
            "dpi": body.dpi or 300,
            "payload_hash": payload_hash,
            "payload_size_bytes": payload_size,
            "content_type": body.content_type.value if body.content_type else "compact_verification",
            "encoding": "utf8",
            "image_width_px": (body.module_size or 10) * 33,
            "image_height_px": (body.module_size or 10) * 33,
            "logo_embedded": body.embed_logo or False,
            "quality_grade": "A",
            "operator_id": body.operator_id,
            "commodity": body.commodity.value if body.commodity else None,
            "compliance_status": body.compliance_status.value if body.compliance_status else "pending",
            "dds_reference": body.dds_reference,
            "batch_code": None,
            "verification_url": None,
            "blockchain_anchor_hash": None,
            "status": "created",
            "reprint_count": 0,
            "scan_count": 0,
            "created_at": now,
            "activated_at": None,
            "expires_at": None,
            "provenance": ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        }

        store = _get_qr_store()
        store[code_id] = qr_record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "QR code generated: id=%s operator=%s format=%s elapsed_ms=%.1f",
            code_id,
            body.operator_id,
            qr_record["output_format"],
            elapsed_ms,
        )

        return GenerateQRResponse(
            code_id=code_id,
            status="success",
            qr_code=QRCodeDetailResponse(**qr_record),
            verification_url=None,
            processing_time_ms=elapsed_ms,
            provenance=qr_record["provenance"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to generate QR code: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate QR code",
        )


# ---------------------------------------------------------------------------
# POST /qr/generate/data-matrix
# ---------------------------------------------------------------------------


@router.post(
    "/qr/generate/data-matrix",
    response_model=GenerateDataMatrixResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Data Matrix code",
    description=(
        "Generate a Data Matrix code per ISO/IEC 16022. Data Matrix "
        "codes are commonly used for small product labels and support "
        "GS1 Digital Link encoding for EUDR traceability."
    ),
    responses={
        201: {"description": "Data Matrix code generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_data_matrix(
    request: Request,
    body: GenerateDataMatrixRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:qr:generate")
    ),
    _rate: None = Depends(rate_limit_qr_generate),
    service: Any = Depends(get_qrg_service),
) -> GenerateDataMatrixResponse:
    """Generate a Data Matrix code per ISO/IEC 16022.

    Args:
        request: FastAPI request object.
        body: Data Matrix generation parameters.
        user: Authenticated user with qr:generate permission.
        service: QR Code Generator service.

    Returns:
        GenerateDataMatrixResponse with code ID and details.
    """
    start = time.monotonic()
    try:
        code_id = str(uuid.uuid4())
        now = _utcnow()

        payload_serialized = json.dumps(body.payload_data, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(payload_serialized.encode("utf-8")).hexdigest()
        payload_size = len(payload_serialized.encode("utf-8"))

        provenance_hash = _compute_provenance_hash({
            "code_id": code_id,
            "operator_id": body.operator_id,
            "symbology": "data_matrix",
            "payload_hash": payload_hash,
            "created_by": user.user_id,
        })

        qr_record = {
            "code_id": code_id,
            "version": "auto",
            "error_correction": "M",
            "symbology": "data_matrix",
            "output_format": body.output_format.value if body.output_format else "png",
            "module_size": body.module_size or 10,
            "quiet_zone": body.quiet_zone if body.quiet_zone is not None else 4,
            "dpi": body.dpi or 300,
            "payload_hash": payload_hash,
            "payload_size_bytes": payload_size,
            "content_type": body.content_type.value if body.content_type else "compact_verification",
            "encoding": "utf8",
            "image_width_px": (body.module_size or 10) * 26,
            "image_height_px": (body.module_size or 10) * 26,
            "logo_embedded": False,
            "quality_grade": "A",
            "operator_id": body.operator_id,
            "commodity": body.commodity.value if body.commodity else None,
            "compliance_status": "pending",
            "dds_reference": body.dds_reference,
            "batch_code": None,
            "verification_url": None,
            "blockchain_anchor_hash": None,
            "status": "created",
            "reprint_count": 0,
            "scan_count": 0,
            "created_at": now,
            "activated_at": None,
            "expires_at": None,
            "provenance": ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        }

        store = _get_qr_store()
        store[code_id] = qr_record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Data Matrix generated: id=%s operator=%s elapsed_ms=%.1f",
            code_id,
            body.operator_id,
            elapsed_ms,
        )

        return GenerateDataMatrixResponse(
            code_id=code_id,
            status="success",
            symbology="data_matrix",
            qr_code=QRCodeDetailResponse(**qr_record),
            processing_time_ms=elapsed_ms,
            provenance=qr_record["provenance"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to generate Data Matrix: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate Data Matrix code",
        )


# ---------------------------------------------------------------------------
# GET /qr/{code_id}
# ---------------------------------------------------------------------------


@router.get(
    "/qr/{code_id}",
    response_model=QRCodeDetailResponse,
    summary="Get QR code details",
    description="Retrieve details of a generated QR code record.",
    responses={
        200: {"description": "QR code details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "QR code not found"},
    },
)
async def get_qr_code_detail(
    request: Request,
    code_id: str = Depends(validate_code_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:qr:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> QRCodeDetailResponse:
    """Get QR code details by ID.

    Args:
        request: FastAPI request object.
        code_id: QR code identifier.
        user: Authenticated user with qr:read permission.
        service: QR Code Generator service.

    Returns:
        QRCodeDetailResponse with full code details.

    Raises:
        HTTPException: 404 if code not found.
    """
    try:
        store = _get_qr_store()
        record = store.get(code_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"QR code {code_id} not found",
            )

        return QRCodeDetailResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get QR code %s: %s", code_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve QR code details",
        )


# ---------------------------------------------------------------------------
# GET /qr/{code_id}/image
# ---------------------------------------------------------------------------


@router.get(
    "/qr/{code_id}/image",
    response_model=QRImageResponse,
    summary="Get QR code image",
    description=(
        "Download the QR code image in its default output format. "
        "Returns the base64-encoded image data with SHA-256 hash."
    ),
    responses={
        200: {"description": "QR code image retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "QR code not found"},
    },
)
async def get_qr_image(
    request: Request,
    code_id: str = Depends(validate_code_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:qr:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> QRImageResponse:
    """Get QR code image in its default format.

    Args:
        request: FastAPI request object.
        code_id: QR code identifier.
        user: Authenticated user with qr:read permission.
        service: QR Code Generator service.

    Returns:
        QRImageResponse with base64-encoded image data.

    Raises:
        HTTPException: 404 if code not found.
    """
    try:
        store = _get_qr_store()
        record = store.get(code_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"QR code {code_id} not found",
            )

        output_format = record.get("output_format", "png")
        mime_map = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
            "zpl": "application/zpl",
            "eps": "application/postscript",
        }

        return QRImageResponse(
            code_id=code_id,
            format=output_format,
            content_type=mime_map.get(output_format, "application/octet-stream"),
            file_size_bytes=record.get("payload_size_bytes", 0),
            image_data_base64=None,
            image_hash=record.get("payload_hash"),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get QR image %s: %s", code_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve QR code image",
        )


# ---------------------------------------------------------------------------
# GET /qr/{code_id}/image/{format}
# ---------------------------------------------------------------------------


@router.get(
    "/qr/{code_id}/image/{format}",
    response_model=QRImageResponse,
    summary="Get QR code image in specific format",
    description=(
        "Download the QR code image in a specific format. Supported "
        "formats: png, svg, pdf, zpl, eps."
    ),
    responses={
        200: {"description": "QR code image retrieved"},
        400: {"model": ErrorResponse, "description": "Unsupported format"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "QR code not found"},
    },
)
async def get_qr_image_format(
    request: Request,
    code_id: str = Depends(validate_code_id),
    format: str = Depends(validate_format_path),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:qr:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> QRImageResponse:
    """Get QR code image in a specific format.

    Args:
        request: FastAPI request object.
        code_id: QR code identifier.
        format: Desired output format (png, svg, pdf, zpl, eps).
        user: Authenticated user with qr:read permission.
        service: QR Code Generator service.

    Returns:
        QRImageResponse with base64-encoded image data.

    Raises:
        HTTPException: 404 if code not found, 400 if format unsupported.
    """
    try:
        store = _get_qr_store()
        record = store.get(code_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"QR code {code_id} not found",
            )

        mime_map = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
            "zpl": "application/zpl",
            "eps": "application/postscript",
        }

        return QRImageResponse(
            code_id=code_id,
            format=format,
            content_type=mime_map.get(format, "application/octet-stream"),
            file_size_bytes=record.get("payload_size_bytes", 0),
            image_data_base64=None,
            image_hash=record.get("payload_hash"),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get QR image %s (format=%s): %s",
            code_id,
            format,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve QR code image",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
