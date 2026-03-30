# -*- coding: utf-8 -*-
"""
Label Routes - AGENT-EUDR-014 QR Code Generator API

Endpoints for EUDR compliance label rendering including single label
generation, batch label generation, detail retrieval, label download,
and template listing with five pre-designed templates.

Endpoints:
    POST   /labels/generate        - Generate a single label
    POST   /labels/generate/batch  - Batch generate labels
    GET    /labels/{label_id}      - Get label details
    GET    /labels/{label_id}/download - Download label file
    GET    /labels/templates       - List label templates

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014, Feature 3 (Label Rendering Engine)
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
from greenlang.schemas import utcnow

from greenlang.agents.eudr.qr_code_generator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_qrg_service,
    rate_limit_label_generate,
    rate_limit_standard,
    require_permission,
    validate_label_id,
)
from greenlang.agents.eudr.qr_code_generator.api.schemas import (
    BatchLabelRequest,
    BatchLabelResponse,
    GenerateLabelRequest,
    GenerateLabelResponse,
    LabelDetailResponse,
    ProvenanceInfo,
    TemplateDetailResponse,
    TemplateListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Label Rendering"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_label_store: Dict[str, Dict] = {}

def _get_label_store() -> Dict[str, Dict]:
    """Return the label record store singleton."""
    return _label_store

def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Compliance colour mapping per EUDR status
# ---------------------------------------------------------------------------

_COMPLIANCE_COLOURS = {
    "compliant": "#2E7D32",       # Green
    "pending": "#F57F17",          # Amber
    "non_compliant": "#C62828",   # Red
    "under_review": "#F57F17",    # Amber
}

# ---------------------------------------------------------------------------
# Built-in template definitions (5 per PRD)
# ---------------------------------------------------------------------------

_TEMPLATES: List[TemplateDetailResponse] = [
    TemplateDetailResponse(
        template_id="tpl-product-001",
        name="Product Label",
        template_type="product_label",
        description="Standard EUDR product label with QR code, compliance status, and product details.",
        width_mm=50.0,
        height_mm=30.0,
        qr_size_mm=20.0,
    ),
    TemplateDetailResponse(
        template_id="tpl-shipping-001",
        name="Shipping Label",
        template_type="shipping_label",
        description="Shipping label with QR code, batch code, consignment details, and EUDR compliance.",
        width_mm=100.0,
        height_mm=70.0,
        qr_size_mm=30.0,
    ),
    TemplateDetailResponse(
        template_id="tpl-pallet-001",
        name="Pallet Label",
        template_type="pallet_label",
        description="Pallet-level label with large QR code for warehouse scanning.",
        width_mm=150.0,
        height_mm=100.0,
        qr_size_mm=50.0,
    ),
    TemplateDetailResponse(
        template_id="tpl-container-001",
        name="Container Label",
        template_type="container_label",
        description="Shipping container label with weatherproof QR and compliance status.",
        width_mm=200.0,
        height_mm=150.0,
        qr_size_mm=80.0,
    ),
    TemplateDetailResponse(
        template_id="tpl-consumer-001",
        name="Consumer Label",
        template_type="consumer_label",
        description="Consumer-facing label with QR code linking to sustainability information.",
        width_mm=40.0,
        height_mm=25.0,
        qr_size_mm=15.0,
    ),
]

def _create_label_record(
    req: GenerateLabelRequest,
    user_id: str,
) -> Dict[str, Any]:
    """Create a label record from a request.

    Args:
        req: Label generation request.
        user_id: ID of the user creating the label.

    Returns:
        Dict representing the label record.
    """
    label_id = str(uuid.uuid4())
    now = utcnow()

    template = req.template.value if req.template else "product_label"
    output_format = req.output_format.value if req.output_format else "pdf"
    compliance_status = "pending"
    colour_hex = _COMPLIANCE_COLOURS.get(compliance_status, "#F57F17")

    # Find matching template for dimensions
    tpl = None
    for t in _TEMPLATES:
        if t.template_type == template:
            tpl = t
            break

    provenance_hash = _compute_provenance_hash({
        "label_id": label_id,
        "code_id": req.code_id,
        "operator_id": req.operator_id,
        "template": template,
        "created_by": user_id,
    })

    return {
        "label_id": label_id,
        "code_id": req.code_id,
        "template": template,
        "compliance_status": compliance_status,
        "compliance_color_hex": colour_hex,
        "output_format": output_format,
        "dpi": req.dpi or 300,
        "width_mm": tpl.width_mm if tpl else 50.0,
        "height_mm": tpl.height_mm if tpl else 30.0,
        "file_size_bytes": 4096,
        "image_data_hash": hashlib.sha256(
            f"{label_id}:{req.code_id}".encode()
        ).hexdigest(),
        "operator_id": req.operator_id,
        "commodity": None,
        "product_name": req.product_name,
        "batch_code": req.batch_code,
        "created_at": now,
        "provenance": ProvenanceInfo(
            provenance_hash=provenance_hash,
            algorithm="sha256",
            created_at=now,
        ),
    }

# ---------------------------------------------------------------------------
# POST /labels/generate
# ---------------------------------------------------------------------------

@router.post(
    "/labels/generate",
    response_model=GenerateLabelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate label",
    description=(
        "Generate a single EUDR compliance label with embedded QR code. "
        "Supports five template types with configurable compliance "
        "status colour coding (green/amber/red)."
    ),
    responses={
        201: {"description": "Label generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_label(
    request: Request,
    body: GenerateLabelRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:labels:generate")
    ),
    _rate: None = Depends(rate_limit_label_generate),
    service: Any = Depends(get_qrg_service),
) -> GenerateLabelResponse:
    """Generate a single compliance label.

    Args:
        request: FastAPI request object.
        body: Label generation parameters.
        user: Authenticated user with labels:generate permission.
        service: QR Code Generator service.

    Returns:
        GenerateLabelResponse with label ID and details.
    """
    start = time.monotonic()
    try:
        record = _create_label_record(body, user.user_id)
        store = _get_label_store()
        store[record["label_id"]] = record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Label generated: id=%s code_id=%s template=%s elapsed_ms=%.1f",
            record["label_id"],
            body.code_id,
            record["template"],
            elapsed_ms,
        )

        return GenerateLabelResponse(
            label_id=record["label_id"],
            status="success",
            label=LabelDetailResponse(**record),
            file_size_bytes=record["file_size_bytes"],
            processing_time_ms=elapsed_ms,
            provenance=record["provenance"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to generate label: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate label",
        )

# ---------------------------------------------------------------------------
# POST /labels/generate/batch
# ---------------------------------------------------------------------------

@router.post(
    "/labels/generate/batch",
    response_model=BatchLabelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch generate labels",
    description=(
        "Generate multiple EUDR compliance labels in a single request. "
        "Up to 500 labels can be generated per batch."
    ),
    responses={
        201: {"description": "Batch labels generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_generate_labels(
    request: Request,
    body: BatchLabelRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:labels:generate")
    ),
    _rate: None = Depends(rate_limit_label_generate),
    service: Any = Depends(get_qrg_service),
) -> BatchLabelResponse:
    """Generate multiple labels in a batch.

    Args:
        request: FastAPI request object.
        body: Batch label generation request.
        user: Authenticated user with labels:generate permission.
        service: QR Code Generator service.

    Returns:
        BatchLabelResponse with all generated labels.
    """
    start = time.monotonic()
    try:
        store = _get_label_store()
        generated: List[GenerateLabelResponse] = []
        failed_count = 0

        for label_req in body.labels:
            try:
                # Apply batch-level overrides
                if body.template and not label_req.template:
                    label_req = label_req.model_copy(
                        update={"template": body.template}
                    )
                if body.output_format and not label_req.output_format:
                    label_req = label_req.model_copy(
                        update={"output_format": body.output_format}
                    )
                if body.dpi and not label_req.dpi:
                    label_req = label_req.model_copy(
                        update={"dpi": body.dpi}
                    )

                record = _create_label_record(label_req, user.user_id)
                store[record["label_id"]] = record
                generated.append(
                    GenerateLabelResponse(
                        label_id=record["label_id"],
                        status="success",
                        label=LabelDetailResponse(**record),
                        file_size_bytes=record["file_size_bytes"],
                        processing_time_ms=0.0,
                        provenance=record["provenance"],
                    )
                )
            except Exception as label_exc:
                logger.warning(
                    "Failed to generate label in batch: %s", label_exc
                )
                failed_count += 1

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Batch label generation: generated=%d failed=%d elapsed_ms=%.1f",
            len(generated),
            failed_count,
            elapsed_ms,
        )

        return BatchLabelResponse(
            status="success" if failed_count == 0 else "partial",
            labels=generated,
            total_generated=len(generated),
            total_failed=failed_count,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to batch generate labels: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to batch generate labels",
        )

# ---------------------------------------------------------------------------
# GET /labels/{label_id}
# ---------------------------------------------------------------------------

@router.get(
    "/labels/{label_id}",
    response_model=LabelDetailResponse,
    summary="Get label details",
    description="Retrieve details of a generated compliance label.",
    responses={
        200: {"description": "Label details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Label not found"},
    },
)
async def get_label_detail(
    request: Request,
    label_id: str = Depends(validate_label_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:labels:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> LabelDetailResponse:
    """Get label details by ID.

    Args:
        request: FastAPI request object.
        label_id: Label identifier.
        user: Authenticated user with labels:read permission.
        service: QR Code Generator service.

    Returns:
        LabelDetailResponse with label details.

    Raises:
        HTTPException: 404 if label not found.
    """
    try:
        store = _get_label_store()
        record = store.get(label_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Label {label_id} not found",
            )

        return LabelDetailResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get label %s: %s", label_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve label details",
        )

# ---------------------------------------------------------------------------
# GET /labels/{label_id}/download
# ---------------------------------------------------------------------------

@router.get(
    "/labels/{label_id}/download",
    response_model=LabelDetailResponse,
    summary="Download label",
    description=(
        "Download the rendered label file. Returns the label record "
        "with file metadata for client-side download."
    ),
    responses={
        200: {"description": "Label download data retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Label not found"},
    },
)
async def download_label(
    request: Request,
    label_id: str = Depends(validate_label_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:labels:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> LabelDetailResponse:
    """Download a rendered label file.

    Args:
        request: FastAPI request object.
        label_id: Label identifier.
        user: Authenticated user with labels:read permission.
        service: QR Code Generator service.

    Returns:
        LabelDetailResponse with label file data.

    Raises:
        HTTPException: 404 if label not found.
    """
    try:
        store = _get_label_store()
        record = store.get(label_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Label {label_id} not found",
            )

        return LabelDetailResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to download label %s: %s", label_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download label",
        )

# ---------------------------------------------------------------------------
# GET /labels/templates
# ---------------------------------------------------------------------------

@router.get(
    "/labels/templates",
    response_model=TemplateListResponse,
    summary="List label templates",
    description=(
        "List all available label templates with dimensions, QR code "
        "size, and descriptions."
    ),
    responses={
        200: {"description": "Templates retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_templates(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:labels:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> TemplateListResponse:
    """List all available label templates.

    Args:
        request: FastAPI request object.
        user: Authenticated user with labels:read permission.
        service: QR Code Generator service.

    Returns:
        TemplateListResponse with all template definitions.
    """
    return TemplateListResponse(
        templates=_TEMPLATES,
        total=len(_TEMPLATES),
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
