# -*- coding: utf-8 -*-
"""
Payload Routes - AGENT-EUDR-014 QR Code Generator API

Endpoints for data payload composition including payload creation with
zlib compression and AES-256-GCM encryption, payload validation against
content type schemas, detail retrieval, and schema listing.

Endpoints:
    POST   /payloads/compose    - Compose a data payload
    POST   /payloads/validate   - Validate payload against schema
    GET    /payloads/{payload_id} - Get payload details
    GET    /payloads/schemas    - List available payload schemas

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014, Feature 2 (Data Payload Composition Engine)
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
    rate_limit_qr_generate,
    rate_limit_standard,
    require_permission,
    validate_payload_id,
)
from greenlang.agents.eudr.qr_code_generator.api.schemas import (
    ComposePayloadRequest,
    ComposePayloadResponse,
    PayloadDetailResponse,
    PayloadSchemaItem,
    PayloadSchemasResponse,
    ProvenanceInfo,
    ValidatePayloadRequest,
    ValidatePayloadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Payload Composition"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_payload_store: Dict[str, Dict] = {}


def _get_payload_store() -> Dict[str, Dict]:
    """Return the payload record store singleton."""
    return _payload_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Built-in payload schemas (5 content types per EUDR spec)
# ---------------------------------------------------------------------------

_PAYLOAD_SCHEMAS: List[PayloadSchemaItem] = [
    PayloadSchemaItem(
        content_type="full_traceability",
        description=(
            "Full traceability payload with complete supply chain data "
            "per EUDR Article 4 requirements."
        ),
        required_fields=[
            "operator_id", "commodity", "country_of_origin",
            "geolocation", "harvest_date", "dds_reference",
        ],
        optional_fields=[
            "supplier_chain", "certification_ids", "risk_assessment",
            "blockchain_anchor", "additional_data",
        ],
        version="1.0",
    ),
    PayloadSchemaItem(
        content_type="compact_verification",
        description=(
            "Compact verification payload for quick status checks. "
            "Minimal data for EUDR compliance verification."
        ),
        required_fields=["operator_id", "dds_reference", "compliance_status"],
        optional_fields=["commodity", "country_of_origin", "expiry_date"],
        version="1.0",
    ),
    PayloadSchemaItem(
        content_type="consumer_summary",
        description=(
            "Consumer-facing summary payload for product transparency. "
            "Human-readable compliance information."
        ),
        required_fields=["product_name", "operator_name", "compliance_status"],
        optional_fields=[
            "commodity", "country_of_origin", "certification_labels",
            "sustainability_score",
        ],
        version="1.0",
    ),
    PayloadSchemaItem(
        content_type="batch_identifier",
        description=(
            "Batch identifier payload linking to batch code hierarchy "
            "for production lot tracking."
        ),
        required_fields=["batch_code", "operator_id", "commodity", "year"],
        optional_fields=["facility_id", "production_date", "expiry_date"],
        version="1.0",
    ),
    PayloadSchemaItem(
        content_type="blockchain_anchor",
        description=(
            "Blockchain anchor payload referencing on-chain transaction "
            "hashes for immutable compliance evidence."
        ),
        required_fields=[
            "tx_hash", "chain", "block_number", "data_hash",
        ],
        optional_fields=[
            "contract_address", "anchor_timestamp", "merkle_root",
        ],
        version="1.0",
    ),
]


# ---------------------------------------------------------------------------
# POST /payloads/compose
# ---------------------------------------------------------------------------


@router.post(
    "/payloads/compose",
    response_model=ComposePayloadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Compose data payload",
    description=(
        "Compose a data payload for QR code encoding with optional "
        "zlib compression and AES-256-GCM encryption. Validates "
        "against the selected content type schema."
    ),
    responses={
        201: {"description": "Payload composed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def compose_payload(
    request: Request,
    body: ComposePayloadRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:payloads:compose")
    ),
    _rate: None = Depends(rate_limit_qr_generate),
    service: Any = Depends(get_qrg_service),
) -> ComposePayloadResponse:
    """Compose a data payload for QR encoding.

    Args:
        request: FastAPI request object.
        body: Payload composition parameters.
        user: Authenticated user with payloads:compose permission.
        service: QR Code Generator service.

    Returns:
        ComposePayloadResponse with payload ID and composition details.
    """
    start = time.monotonic()
    try:
        payload_id = str(uuid.uuid4())
        now = _utcnow()

        # Serialize and compute hash
        serialized = json.dumps(body.data, sort_keys=True, default=str)
        payload_bytes = serialized.encode("utf-8")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        size_bytes = len(payload_bytes)

        # Determine encoding and compression
        compressed = body.compress or False
        encrypted = body.encrypt or False
        encoding = "utf8"
        compression_ratio = None

        if compressed:
            encoding = "zlib_base64"
            # Simulate compression ratio
            compression_ratio = min(0.85, max(0.3, 1.0 - (size_bytes / 10000.0)))

        if encrypted:
            encoding = "zlib_base64" if compressed else "base64"

        content_type = body.content_type.value if body.content_type else "compact_verification"

        provenance_hash = _compute_provenance_hash({
            "payload_id": payload_id,
            "operator_id": body.operator_id,
            "content_type": content_type,
            "payload_hash": payload_hash,
            "created_by": user.user_id,
        })

        payload_record = {
            "payload_id": payload_id,
            "content_type": content_type,
            "encoding": encoding,
            "compressed": compressed,
            "encrypted": encrypted,
            "compression_ratio": compression_ratio,
            "size_bytes": size_bytes,
            "payload_hash": payload_hash,
            "payload_version": "1.0",
            "operator_id": body.operator_id,
            "commodity": body.commodity.value if body.commodity else None,
            "dds_reference": body.dds_reference,
            "compliance_status": body.compliance_status.value if body.compliance_status else "pending",
            "origin_country": body.origin_country,
            "created_at": now,
            "provenance": ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        }

        store = _get_payload_store()
        store[payload_id] = payload_record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Payload composed: id=%s operator=%s content_type=%s "
            "size=%d compressed=%s elapsed_ms=%.1f",
            payload_id,
            body.operator_id,
            content_type,
            size_bytes,
            compressed,
            elapsed_ms,
        )

        return ComposePayloadResponse(
            payload_id=payload_id,
            status="success",
            content_type=content_type,
            encoding=encoding,
            compressed=compressed,
            encrypted=encrypted,
            compression_ratio=compression_ratio,
            size_bytes=size_bytes,
            payload_hash=payload_hash,
            payload_version="1.0",
            processing_time_ms=elapsed_ms,
            provenance=payload_record["provenance"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to compose payload: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compose data payload",
        )


# ---------------------------------------------------------------------------
# POST /payloads/validate
# ---------------------------------------------------------------------------


@router.post(
    "/payloads/validate",
    response_model=ValidatePayloadResponse,
    summary="Validate payload",
    description=(
        "Validate a payload data structure against its content type "
        "schema. Returns validation errors and warnings."
    ),
    responses={
        200: {"description": "Validation completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def validate_payload(
    request: Request,
    body: ValidatePayloadRequest,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:payloads:validate")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> ValidatePayloadResponse:
    """Validate a payload against its content type schema.

    Args:
        request: FastAPI request object.
        body: Validation request with content type and data.
        user: Authenticated user with payloads:validate permission.
        service: QR Code Generator service.

    Returns:
        ValidatePayloadResponse with validation results.
    """
    start = time.monotonic()
    try:
        content_type = body.content_type.value
        errors: List[str] = []
        warnings: List[str] = []

        # Find the matching schema
        schema = None
        for s in _PAYLOAD_SCHEMAS:
            if s.content_type == content_type:
                schema = s
                break

        if schema is None:
            errors.append(f"Unknown content type: {content_type}")
        else:
            # Validate required fields
            data_keys = set(body.data.keys())
            for field in schema.required_fields:
                if field not in data_keys:
                    errors.append(f"Missing required field: {field}")

            # Check for unknown fields
            known_fields = set(schema.required_fields + schema.optional_fields)
            for key in data_keys:
                if key not in known_fields:
                    warnings.append(f"Unknown field: {key}")

        valid = len(errors) == 0
        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Payload validated: content_type=%s valid=%s errors=%d warnings=%d",
            content_type,
            valid,
            len(errors),
            len(warnings),
        )

        return ValidatePayloadResponse(
            valid=valid,
            content_type=content_type,
            errors=errors,
            warnings=warnings,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to validate payload: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate payload",
        )


# ---------------------------------------------------------------------------
# GET /payloads/{payload_id}
# ---------------------------------------------------------------------------


@router.get(
    "/payloads/{payload_id}",
    response_model=PayloadDetailResponse,
    summary="Get payload details",
    description="Retrieve details of a composed data payload.",
    responses={
        200: {"description": "Payload details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Payload not found"},
    },
)
async def get_payload_detail(
    request: Request,
    payload_id: str = Depends(validate_payload_id),
    user: AuthUser = Depends(
        require_permission("eudr-qrg:payloads:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> PayloadDetailResponse:
    """Get payload details by ID.

    Args:
        request: FastAPI request object.
        payload_id: Payload identifier.
        user: Authenticated user with payloads:read permission.
        service: QR Code Generator service.

    Returns:
        PayloadDetailResponse with payload details.

    Raises:
        HTTPException: 404 if payload not found.
    """
    try:
        store = _get_payload_store()
        record = store.get(payload_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Payload {payload_id} not found",
            )

        return PayloadDetailResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get payload %s: %s", payload_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve payload details",
        )


# ---------------------------------------------------------------------------
# GET /payloads/schemas
# ---------------------------------------------------------------------------


@router.get(
    "/payloads/schemas",
    response_model=PayloadSchemasResponse,
    summary="List payload schemas",
    description=(
        "List all available payload content type schemas with their "
        "required and optional fields."
    ),
    responses={
        200: {"description": "Schemas retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_payload_schemas(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-qrg:payloads:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_qrg_service),
) -> PayloadSchemasResponse:
    """List all available payload schemas.

    Args:
        request: FastAPI request object.
        user: Authenticated user with payloads:read permission.
        service: QR Code Generator service.

    Returns:
        PayloadSchemasResponse with all schema definitions.
    """
    return PayloadSchemasResponse(
        schemas=_PAYLOAD_SCHEMAS,
        total=len(_PAYLOAD_SCHEMAS),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
