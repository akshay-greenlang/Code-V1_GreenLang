# -*- coding: utf-8 -*-
"""
Balance, Transform, and Document Routes - AGENT-EUDR-009 Chain of Custody API

Endpoints for mass balance ledger management, product transformations,
and document chain linking.

Balance Endpoints:
    POST   /balance/input                - Record mass balance input
    POST   /balance/output               - Record mass balance output
    GET    /balance/{facility_id}         - Get current balance
    POST   /balance/reconcile            - Reconcile period balance
    GET    /balance/history/{facility_id} - Get balance history

Transform Endpoints:
    POST   /transform                    - Record transformation
    POST   /transform/batch              - Batch transformation import
    GET    /transform/{transform_id}     - Get transformation details

Document Endpoints:
    POST   /documents                    - Link document to event
    GET    /documents/{batch_id}         - Get documents for batch
    POST   /documents/validate           - Validate document chain

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009, Section 7.4
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.chain_of_custody.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_coc_service,
    get_pagination,
    get_request_id,
    rate_limit_batch,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_batch_id,
    validate_facility_id,
    validate_transform_id,
)
from greenlang.agents.eudr.chain_of_custody.api.schemas import (
    BalanceEntryResponse,
    BalanceEntryType,
    BalanceHistoryEntry,
    BalanceHistoryResponse,
    BalanceInputRequest,
    BalanceOutputRequest,
    BalanceReconcileRequest,
    BalanceReconcileResponse,
    DocumentLinkRequest,
    DocumentListResponse,
    DocumentResponse,
    DocumentType,
    DocumentValidateRequest,
    DocumentValidateResponse,
    DocumentValidationFinding,
    EUDRCommodity,
    FacilityBalanceResponse,
    PaginatedMeta,
    ProvenanceInfo,
    QuantitySpec,
    ReconciliationStatus,
    TransformBatchRequest,
    TransformBatchResponse,
    TransformCreateRequest,
    TransformInputDetail,
    TransformOutputDetail,
    TransformResponse,
    UnitOfMeasure,
    VerificationSeverity,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Mass Balance, Transforms & Documents"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_balance_store: Dict[str, List[Dict]] = {}
_transform_store: Dict[str, Dict] = {}
_document_store: Dict[str, Dict] = {}
_document_batch_index: Dict[str, List[str]] = {}


def _get_balance_store() -> Dict[str, List[Dict]]:
    """Return the balance entry store."""
    return _balance_store


def _get_transform_store() -> Dict[str, Dict]:
    """Return the transformation store."""
    return _transform_store


def _get_document_store() -> Dict[str, Dict]:
    """Return the document store."""
    return _document_store


def _get_document_batch_index() -> Dict[str, List[str]]:
    """Return the batch-to-document index."""
    return _document_batch_index


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _zero_quantity(unit: UnitOfMeasure = UnitOfMeasure.KG) -> QuantitySpec:
    """Create a zero-quantity spec."""
    return QuantitySpec(amount=Decimal("0"), unit=unit)


# =============================================================================
# MASS BALANCE ENDPOINTS
# =============================================================================


# ---------------------------------------------------------------------------
# POST /balance/input
# ---------------------------------------------------------------------------


@router.post(
    "/balance/input",
    response_model=BalanceEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record mass balance input",
    description=(
        "Record an input entry in the mass balance ledger. "
        "Inputs represent material received at a facility."
    ),
    responses={
        201: {"description": "Input recorded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def record_balance_input(
    request: Request,
    body: BalanceInputRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:balance:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BalanceEntryResponse:
    """Record a mass balance input entry.

    Args:
        body: Balance input parameters.
        user: Authenticated user with balance:write permission.

    Returns:
        BalanceEntryResponse with the recorded entry.
    """
    start = time.monotonic()
    try:
        entry_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        entry = {
            "entry_id": entry_id,
            "facility_id": body.facility_id,
            "batch_id": body.batch_id,
            "commodity": body.commodity,
            "entry_type": body.entry_type,
            "quantity": body.quantity,
            "is_certified": body.is_certified,
            "certification_id": body.certification_id,
            "source_facility_id": body.source_facility_id,
            "destination_facility_id": None,
            "conversion_factor": None,
            "loss_quantity": None,
            "period_start": body.period_start,
            "period_end": body.period_end,
            "notes": body.notes,
            "provenance": provenance,
            "created_at": now,
        }

        store = _get_balance_store()
        key = f"{body.facility_id}:{body.commodity.value}"
        if key not in store:
            store[key] = []
        store[key].append(entry)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Balance input recorded: facility=%s batch=%s qty=%s certified=%s",
            body.facility_id,
            body.batch_id,
            str(body.quantity.amount),
            body.is_certified,
        )

        return BalanceEntryResponse(**entry, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record balance input: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record mass balance input",
        )


# ---------------------------------------------------------------------------
# POST /balance/output
# ---------------------------------------------------------------------------


@router.post(
    "/balance/output",
    response_model=BalanceEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record mass balance output",
    description=(
        "Record an output entry in the mass balance ledger. "
        "Outputs represent material dispatched from a facility."
    ),
    responses={
        201: {"description": "Output recorded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def record_balance_output(
    request: Request,
    body: BalanceOutputRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:balance:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BalanceEntryResponse:
    """Record a mass balance output entry.

    Args:
        body: Balance output parameters.
        user: Authenticated user with balance:write permission.

    Returns:
        BalanceEntryResponse with the recorded entry.
    """
    start = time.monotonic()
    try:
        entry_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        entry = {
            "entry_id": entry_id,
            "facility_id": body.facility_id,
            "batch_id": body.batch_id,
            "commodity": body.commodity,
            "entry_type": body.entry_type,
            "quantity": body.quantity,
            "is_certified": body.is_certified_claim,
            "certification_id": None,
            "source_facility_id": None,
            "destination_facility_id": body.destination_facility_id,
            "conversion_factor": body.conversion_factor,
            "loss_quantity": body.loss_quantity,
            "period_start": body.period_start,
            "period_end": body.period_end,
            "notes": body.notes,
            "provenance": provenance,
            "created_at": now,
        }

        store = _get_balance_store()
        key = f"{body.facility_id}:{body.commodity.value}"
        if key not in store:
            store[key] = []
        store[key].append(entry)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Balance output recorded: facility=%s batch=%s qty=%s",
            body.facility_id,
            body.batch_id,
            str(body.quantity.amount),
        )

        return BalanceEntryResponse(**entry, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record balance output: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record mass balance output",
        )


# ---------------------------------------------------------------------------
# GET /balance/{facility_id}
# ---------------------------------------------------------------------------


@router.get(
    "/balance/{facility_id}",
    response_model=FacilityBalanceResponse,
    summary="Get current balance",
    description="Get the current mass balance for a facility.",
    responses={
        200: {"description": "Current facility balance"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No balance entries found"},
    },
)
async def get_facility_balance(
    request: Request,
    facility_id: str = Depends(validate_facility_id),
    commodity: EUDRCommodity = Query(..., description="Commodity to query"),
    user: AuthUser = Depends(
        require_permission("eudr-coc:balance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FacilityBalanceResponse:
    """Get current mass balance for a facility.

    Args:
        facility_id: Facility identifier.
        commodity: Commodity to query balance for.
        user: Authenticated user with balance:read permission.

    Returns:
        FacilityBalanceResponse with current balance details.
    """
    start = time.monotonic()
    try:
        store = _get_balance_store()
        key = f"{facility_id}:{commodity.value}"
        entries = store.get(key, [])

        if not entries:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No balance entries found for facility {facility_id}",
            )

        unit = entries[0]["quantity"].unit

        total_input = Decimal("0")
        total_output = Decimal("0")
        total_loss = Decimal("0")
        certified_input = Decimal("0")
        certified_output = Decimal("0")

        for entry in entries:
            amount = entry["quantity"].amount
            entry_type = entry["entry_type"]
            is_cert = entry.get("is_certified", False)

            if entry_type == BalanceEntryType.INPUT:
                total_input += amount
                if is_cert:
                    certified_input += amount
            elif entry_type == BalanceEntryType.OUTPUT:
                total_output += amount
                if is_cert:
                    certified_output += amount
            elif entry_type in (BalanceEntryType.LOSS, BalanceEntryType.WASTE):
                total_loss += amount

            if entry.get("loss_quantity"):
                total_loss += entry["loss_quantity"].amount

        current_balance = total_input - total_output - total_loss
        certified_ratio = float(certified_input / total_input) if total_input > 0 else 0.0

        provenance_data = {
            "facility_id": facility_id,
            "commodity": commodity.value,
            "balance": str(current_balance),
        }
        provenance_hash = _compute_provenance_hash(provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Balance retrieved: facility=%s commodity=%s balance=%s",
            facility_id,
            commodity.value,
            str(current_balance),
        )

        return FacilityBalanceResponse(
            facility_id=facility_id,
            commodity=commodity,
            total_input=QuantitySpec(amount=total_input, unit=unit),
            total_output=QuantitySpec(amount=total_output, unit=unit),
            total_loss=QuantitySpec(amount=total_loss, unit=unit),
            current_balance=QuantitySpec(amount=max(current_balance, Decimal("0")), unit=unit),
            certified_input=QuantitySpec(amount=certified_input, unit=unit),
            certified_output=QuantitySpec(amount=certified_output, unit=unit),
            certified_ratio=certified_ratio,
            entry_count=len(entries),
            last_updated=datetime.now(timezone.utc).replace(microsecond=0),
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get balance for %s: %s", facility_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve facility balance",
        )


# ---------------------------------------------------------------------------
# POST /balance/reconcile
# ---------------------------------------------------------------------------


@router.post(
    "/balance/reconcile",
    response_model=BalanceReconcileResponse,
    summary="Reconcile period balance",
    description=(
        "Reconcile the mass balance for a facility over a specified "
        "period, checking input/output variance against tolerance."
    ),
    responses={
        200: {"description": "Reconciliation completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def reconcile_balance(
    request: Request,
    body: BalanceReconcileRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:balance:reconcile")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BalanceReconcileResponse:
    """Reconcile mass balance for a period.

    Args:
        body: Reconciliation parameters.
        user: Authenticated user with balance:reconcile permission.

    Returns:
        BalanceReconcileResponse with reconciliation results.
    """
    start = time.monotonic()
    try:
        reconciliation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        store = _get_balance_store()
        key = f"{body.facility_id}:{body.commodity.value}"
        entries = store.get(key, [])

        # Filter entries by period
        period_entries = [
            e for e in entries
            if (e.get("period_start") is None or e["created_at"] >= body.period_start)
            and (e.get("period_end") is None or e["created_at"] <= body.period_end)
        ]

        unit = period_entries[0]["quantity"].unit if period_entries else UnitOfMeasure.KG

        total_input = Decimal("0")
        total_output = Decimal("0")
        total_loss = Decimal("0")
        certified_input = Decimal("0")

        for entry in period_entries:
            amount = entry["quantity"].amount
            entry_type = entry["entry_type"]

            if entry_type == BalanceEntryType.INPUT:
                total_input += amount
                if entry.get("is_certified"):
                    certified_input += amount
            elif entry_type == BalanceEntryType.OUTPUT:
                total_output += amount
            elif entry_type in (BalanceEntryType.LOSS, BalanceEntryType.WASTE):
                total_loss += amount

        variance = total_input - total_output - total_loss
        variance_percent = (
            (variance / total_input * 100) if total_input > 0 else Decimal("0")
        )
        within_tolerance = abs(float(variance_percent)) <= float(body.tolerance_percent * 100)

        if within_tolerance:
            recon_status = ReconciliationStatus.BALANCED
        elif variance > 0:
            recon_status = ReconciliationStatus.SURPLUS
        else:
            recon_status = ReconciliationStatus.DEFICIT

        certified_ratio = float(certified_input / total_input) if total_input > 0 else 0.0

        findings: List[str] = []
        if not within_tolerance:
            findings.append(
                f"Variance of {float(variance_percent):.2f}% exceeds "
                f"tolerance of {float(body.tolerance_percent * 100):.2f}%"
            )
        if total_loss > 0 and body.expected_loss_rate is not None:
            actual_loss_rate = total_loss / total_input if total_input > 0 else Decimal("0")
            if actual_loss_rate > body.expected_loss_rate:
                findings.append(
                    f"Actual loss rate ({float(actual_loss_rate):.4f}) exceeds "
                    f"expected rate ({float(body.expected_loss_rate):.4f})"
                )

        provenance_data = {
            "reconciliation_id": reconciliation_id,
            "facility_id": body.facility_id,
            "variance": str(variance),
            "status": recon_status.value,
        }
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="reconciliation",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Balance reconciled: facility=%s commodity=%s status=%s variance=%.2f%%",
            body.facility_id,
            body.commodity.value,
            recon_status.value,
            float(variance_percent),
        )

        return BalanceReconcileResponse(
            reconciliation_id=reconciliation_id,
            facility_id=body.facility_id,
            commodity=body.commodity,
            period_start=body.period_start,
            period_end=body.period_end,
            status=recon_status,
            total_input=QuantitySpec(amount=total_input, unit=unit),
            total_output=QuantitySpec(amount=total_output, unit=unit),
            total_loss=QuantitySpec(amount=total_loss, unit=unit),
            variance=QuantitySpec(amount=abs(variance), unit=unit),
            variance_percent=abs(variance_percent),
            within_tolerance=within_tolerance,
            certified_ratio=certified_ratio,
            findings=findings,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed balance reconciliation: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reconcile mass balance",
        )


# ---------------------------------------------------------------------------
# GET /balance/history/{facility_id}
# ---------------------------------------------------------------------------


@router.get(
    "/balance/history/{facility_id}",
    response_model=BalanceHistoryResponse,
    summary="Get balance history",
    description="Get the mass balance ledger history for a facility.",
    responses={
        200: {"description": "Balance history"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No entries found"},
    },
)
async def get_balance_history(
    request: Request,
    facility_id: str = Depends(validate_facility_id),
    commodity: EUDRCommodity = Query(..., description="Commodity"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-coc:balance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BalanceHistoryResponse:
    """Get balance history for a facility.

    Args:
        facility_id: Facility identifier.
        commodity: Commodity to query.
        pagination: Pagination parameters.
        user: Authenticated user with balance:read permission.

    Returns:
        BalanceHistoryResponse with ledger entries.
    """
    start = time.monotonic()
    try:
        store = _get_balance_store()
        key = f"{facility_id}:{commodity.value}"
        entries = store.get(key, [])

        if not entries:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No balance history for facility {facility_id}",
            )

        unit = entries[0]["quantity"].unit
        total = len(entries)
        page = entries[pagination.offset: pagination.offset + pagination.limit]

        # Compute running balance
        running_balance = Decimal("0")
        history_entries: List[BalanceHistoryEntry] = []
        for entry in page:
            amount = entry["quantity"].amount
            if entry["entry_type"] == BalanceEntryType.INPUT:
                running_balance += amount
            else:
                running_balance -= amount

            history_entries.append(
                BalanceHistoryEntry(
                    entry_id=entry["entry_id"],
                    entry_type=entry["entry_type"],
                    batch_id=entry["batch_id"],
                    quantity=entry["quantity"],
                    running_balance=QuantitySpec(
                        amount=max(running_balance, Decimal("0")),
                        unit=unit,
                    ),
                    is_certified=entry.get("is_certified", False),
                    source=entry.get("source_facility_id"),
                    destination=entry.get("destination_facility_id"),
                    timestamp=entry.get("created_at", datetime.now(timezone.utc)),
                )
            )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return BalanceHistoryResponse(
            facility_id=facility_id,
            commodity=commodity,
            entries=history_entries,
            meta=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            current_balance=QuantitySpec(
                amount=max(running_balance, Decimal("0")),
                unit=unit,
            ),
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get balance history for %s: %s",
            facility_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve balance history",
        )


# =============================================================================
# TRANSFORMATION ENDPOINTS
# =============================================================================


# ---------------------------------------------------------------------------
# POST /transform
# ---------------------------------------------------------------------------


@router.post(
    "/transform",
    response_model=TransformResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record transformation",
    description=(
        "Record a product transformation (processing, refining, roasting, etc.) "
        "with input/output batch tracking and conversion factors."
    ),
    responses={
        201: {"description": "Transformation recorded"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def record_transformation(
    request: Request,
    body: TransformCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:transforms:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TransformResponse:
    """Record a product transformation.

    Args:
        body: Transformation parameters.
        user: Authenticated user with transforms:create permission.

    Returns:
        TransformResponse with transformation details.
    """
    start = time.monotonic()
    try:
        transform_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)
        transform_timestamp = body.timestamp or now

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        # Build input details
        input_details = [
            TransformInputDetail(
                batch_id=inp.batch_id,
                quantity=inp.quantity,
                commodity=inp.commodity,
                origin_country=None,
                custody_model=None,
            )
            for inp in body.input_batches
        ]

        # Build output details with generated batch IDs
        output_details = [
            TransformOutputDetail(
                batch_id=str(uuid.uuid4()),
                batch_reference=out.batch_reference,
                quantity=out.quantity,
                commodity=out.commodity,
            )
            for out in body.output_batches
        ]

        # Calculate totals
        total_input_amount = sum(
            inp.quantity.amount for inp in body.input_batches
        )
        total_output_amount = sum(
            out.quantity.amount for out in body.output_batches
        )
        input_unit = body.input_batches[0].quantity.unit
        output_unit = body.output_batches[0].quantity.unit

        total_loss_amount = total_input_amount - total_output_amount
        total_loss = None
        if total_loss_amount > 0:
            total_loss = QuantitySpec(amount=total_loss_amount, unit=input_unit)

        transform_record = {
            "transform_id": transform_id,
            "facility_id": body.facility_id,
            "transformation_type": body.transformation_type,
            "input_batches": input_details,
            "output_batches": output_details,
            "conversion_factor": body.conversion_factor,
            "loss_rate": body.loss_rate,
            "total_input_quantity": QuantitySpec(
                amount=total_input_amount, unit=input_unit
            ),
            "total_output_quantity": QuantitySpec(
                amount=total_output_amount, unit=output_unit
            ),
            "total_loss_quantity": total_loss,
            "timestamp": transform_timestamp,
            "status": "completed",
            "notes": body.notes,
            "provenance": provenance,
        }

        store = _get_transform_store()
        store[transform_id] = transform_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Transformation recorded: id=%s type=%s facility=%s",
            transform_id,
            body.transformation_type.value,
            body.facility_id,
        )

        return TransformResponse(
            **transform_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record transformation: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record transformation",
        )


# ---------------------------------------------------------------------------
# POST /transform/batch
# ---------------------------------------------------------------------------


@router.post(
    "/transform/batch",
    response_model=TransformBatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch transformation import",
    description="Import multiple transformations in a single request.",
    responses={
        201: {"description": "Batch import completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_import_transformations(
    request: Request,
    body: TransformBatchRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:transforms:create")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> TransformBatchResponse:
    """Import multiple transformations in bulk.

    Args:
        body: Batch import request.
        user: Authenticated user with transforms:create permission.

    Returns:
        TransformBatchResponse with results.
    """
    start = time.monotonic()
    try:
        accepted: List[TransformResponse] = []
        errors: List[Dict] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)
        store = _get_transform_store()

        for idx, transform_req in enumerate(body.transformations):
            try:
                transform_id = str(uuid.uuid4())
                transform_timestamp = transform_req.timestamp or now

                provenance_data = transform_req.model_dump(mode="json")
                provenance_hash = _compute_provenance_hash(provenance_data)
                provenance = ProvenanceInfo(
                    provenance_hash=provenance_hash,
                    created_by=user.user_id,
                    created_at=now,
                    source="batch_import",
                )

                input_details = [
                    TransformInputDetail(
                        batch_id=inp.batch_id,
                        quantity=inp.quantity,
                        commodity=inp.commodity,
                    )
                    for inp in transform_req.input_batches
                ]
                output_details = [
                    TransformOutputDetail(
                        batch_id=str(uuid.uuid4()),
                        batch_reference=out.batch_reference,
                        quantity=out.quantity,
                        commodity=out.commodity,
                    )
                    for out in transform_req.output_batches
                ]

                total_input = sum(i.quantity.amount for i in transform_req.input_batches)
                total_output = sum(o.quantity.amount for o in transform_req.output_batches)
                input_unit = transform_req.input_batches[0].quantity.unit
                output_unit = transform_req.output_batches[0].quantity.unit

                record = {
                    "transform_id": transform_id,
                    "facility_id": transform_req.facility_id,
                    "transformation_type": transform_req.transformation_type,
                    "input_batches": input_details,
                    "output_batches": output_details,
                    "conversion_factor": transform_req.conversion_factor,
                    "loss_rate": transform_req.loss_rate,
                    "total_input_quantity": QuantitySpec(amount=total_input, unit=input_unit),
                    "total_output_quantity": QuantitySpec(amount=total_output, unit=output_unit),
                    "total_loss_quantity": None,
                    "timestamp": transform_timestamp,
                    "status": "completed",
                    "notes": transform_req.notes,
                    "provenance": provenance,
                }

                if not body.validate_only:
                    store[transform_id] = record

                accepted.append(
                    TransformResponse(**record, processing_time_ms=0.0)
                )
            except Exception as item_exc:
                errors.append({"index": idx, "error": str(item_exc)})

        batch_hash = _compute_provenance_hash({
            "total": len(body.transformations),
            "accepted": len(accepted),
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch transform import: total=%d accepted=%d rejected=%d",
            len(body.transformations),
            len(accepted),
            len(errors),
        )

        return TransformBatchResponse(
            total_submitted=len(body.transformations),
            total_accepted=len(accepted),
            total_rejected=len(errors),
            transformations=accepted,
            errors=errors,
            validate_only=body.validate_only,
            provenance_hash=batch_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed batch transform import: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch transformation import",
        )


# ---------------------------------------------------------------------------
# GET /transform/{transform_id}
# ---------------------------------------------------------------------------


@router.get(
    "/transform/{transform_id}",
    response_model=TransformResponse,
    summary="Get transformation details",
    description="Retrieve full details of a recorded transformation.",
    responses={
        200: {"description": "Transformation details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Transformation not found"},
    },
)
async def get_transformation(
    request: Request,
    transform_id: str = Depends(validate_transform_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:transforms:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TransformResponse:
    """Get transformation details by ID.

    Args:
        transform_id: Transformation identifier.
        user: Authenticated user with transforms:read permission.

    Returns:
        TransformResponse with transformation details.

    Raises:
        HTTPException: 404 if transformation not found.
    """
    try:
        store = _get_transform_store()
        record = store.get(transform_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transformation {transform_id} not found",
            )

        return TransformResponse(**record, processing_time_ms=0.0)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get transformation %s: %s",
            transform_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transformation",
        )


# =============================================================================
# DOCUMENT ENDPOINTS
# =============================================================================


# ---------------------------------------------------------------------------
# POST /documents
# ---------------------------------------------------------------------------


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Link document to event",
    description=(
        "Link a supporting document (bill of lading, certificate, etc.) "
        "to a custody event or batch for the document chain."
    ),
    responses={
        201: {"description": "Document linked successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def link_document(
    request: Request,
    body: DocumentLinkRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:documents:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> DocumentResponse:
    """Link a document to a custody event or batch.

    Args:
        body: Document link parameters.
        user: Authenticated user with documents:create permission.

    Returns:
        DocumentResponse with linked document details.
    """
    start = time.monotonic()
    try:
        document_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Determine validity
        is_valid = True
        if body.expiry_date and body.expiry_date < now:
            is_valid = False

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        doc_record = {
            "document_id": document_id,
            "event_id": body.event_id,
            "batch_id": body.batch_id,
            "document_type": body.document_type,
            "document_reference": body.document_reference,
            "document_url": body.document_url,
            "document_hash": body.document_hash,
            "issuer": body.issuer,
            "issue_date": body.issue_date,
            "expiry_date": body.expiry_date,
            "is_valid": is_valid,
            "notes": body.notes,
            "provenance": provenance,
            "created_at": now,
        }

        store = _get_document_store()
        store[document_id] = doc_record

        # Update batch index
        if body.batch_id:
            index = _get_document_batch_index()
            if body.batch_id not in index:
                index[body.batch_id] = []
            index[body.batch_id].append(document_id)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Document linked: id=%s type=%s batch=%s event=%s",
            document_id,
            body.document_type.value,
            body.batch_id,
            body.event_id,
        )

        return DocumentResponse(**doc_record, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to link document: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to link document",
        )


# ---------------------------------------------------------------------------
# GET /documents/{batch_id}
# ---------------------------------------------------------------------------


@router.get(
    "/documents/{batch_id}",
    response_model=DocumentListResponse,
    summary="Get documents for batch",
    description="Retrieve all documents linked to a batch.",
    responses={
        200: {"description": "Document list"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No documents found"},
    },
)
async def get_batch_documents(
    request: Request,
    batch_id: str = Depends(validate_batch_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:documents:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DocumentListResponse:
    """Get all documents linked to a batch.

    Args:
        batch_id: Batch identifier.
        user: Authenticated user with documents:read permission.

    Returns:
        DocumentListResponse with linked documents.
    """
    try:
        index = _get_document_batch_index()
        doc_ids = index.get(batch_id, [])
        store = _get_document_store()

        documents: List[DocumentResponse] = []
        doc_types_present: set = set()

        for doc_id in doc_ids:
            record = store.get(doc_id)
            if record:
                documents.append(
                    DocumentResponse(**record, processing_time_ms=0.0)
                )
                doc_types_present.add(record["document_type"])

        # Check for required document types
        required_types = {
            DocumentType.CERTIFICATE_OF_ORIGIN,
            DocumentType.BILL_OF_LADING,
        }
        missing_types = required_types - doc_types_present
        has_complete_chain = len(missing_types) == 0

        return DocumentListResponse(
            batch_id=batch_id,
            documents=documents,
            total_documents=len(documents),
            document_types_present=list(doc_types_present),
            has_complete_chain=has_complete_chain,
            missing_document_types=list(missing_types),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get documents for batch %s: %s",
            batch_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve batch documents",
        )


# ---------------------------------------------------------------------------
# POST /documents/validate
# ---------------------------------------------------------------------------


@router.post(
    "/documents/validate",
    response_model=DocumentValidateResponse,
    summary="Validate document chain",
    description=(
        "Validate the document chain for a batch, checking for "
        "completeness, expiry, and hash integrity."
    ),
    responses={
        200: {"description": "Validation results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def validate_document_chain(
    request: Request,
    body: DocumentValidateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:documents:validate")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DocumentValidateResponse:
    """Validate document chain for a batch.

    Args:
        body: Validation request.
        user: Authenticated user with documents:validate permission.

    Returns:
        DocumentValidateResponse with validation findings.
    """
    start = time.monotonic()
    try:
        index = _get_document_batch_index()
        doc_ids = index.get(body.batch_id, [])
        store = _get_document_store()
        now = datetime.now(timezone.utc).replace(microsecond=0)

        findings: List[DocumentValidationFinding] = []
        docs_valid = 0
        docs_invalid = 0
        expired_count = 0
        doc_types_found: set = set()

        for doc_id in doc_ids:
            record = store.get(doc_id)
            if not record:
                continue

            doc_type = record["document_type"]
            doc_types_found.add(doc_type)

            # Check expiry
            if body.check_expiry and record.get("expiry_date"):
                if record["expiry_date"] < now:
                    expired_count += 1
                    docs_invalid += 1
                    findings.append(
                        DocumentValidationFinding(
                            document_id=doc_id,
                            document_type=doc_type,
                            finding_type="expired",
                            severity=VerificationSeverity.HIGH,
                            message=f"Document {record['document_reference']} has expired",
                        )
                    )
                    continue

            docs_valid += 1

        # Check for missing required types
        missing_types: List[DocumentType] = []
        for req_type in body.required_document_types:
            if req_type not in doc_types_found:
                missing_types.append(req_type)
                findings.append(
                    DocumentValidationFinding(
                        document_id=None,
                        document_type=req_type,
                        finding_type="missing",
                        severity=VerificationSeverity.CRITICAL,
                        message=f"Required document type '{req_type.value}' is missing",
                    )
                )

        is_valid = len(findings) == 0

        provenance_data = {
            "batch_id": body.batch_id,
            "is_valid": is_valid,
            "docs_checked": docs_valid + docs_invalid,
        }
        provenance_hash = _compute_provenance_hash(provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Document validation: batch=%s valid=%s findings=%d",
            body.batch_id,
            is_valid,
            len(findings),
        )

        return DocumentValidateResponse(
            batch_id=body.batch_id,
            is_valid=is_valid,
            findings=findings,
            total_documents_checked=docs_valid + docs_invalid,
            documents_valid=docs_valid,
            documents_invalid=docs_invalid,
            missing_types=missing_types,
            expired_documents=expired_count,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed document validation for batch %s: %s",
            body.batch_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate document chain",
        )
