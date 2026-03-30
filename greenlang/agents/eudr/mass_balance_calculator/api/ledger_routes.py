# -*- coding: utf-8 -*-
"""
Ledger Routes - AGENT-EUDR-011 Mass Balance Calculator API

Endpoints for double-entry ledger management including creation,
entry recording, bulk import, balance queries, history retrieval,
and multi-criteria search. All quantity fields use Decimal for
bit-perfect arithmetic required by EUDR Article 14 and ISO 22095:2020.

Endpoints:
    POST   /ledgers                     - Create a new mass balance ledger
    GET    /ledgers/{ledger_id}         - Get ledger details
    POST   /ledgers/entries             - Record a single ledger entry
    POST   /ledgers/entries/bulk        - Bulk entry import (max 500)
    GET    /ledgers/{ledger_id}/balance - Get current balance details
    GET    /ledgers/{ledger_id}/history - Get entry history with filters
    POST   /ledgers/search             - Search ledgers by criteria

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011, Feature 1 (Double-Entry Ledger Management)
Agent ID: GL-EUDR-MBC-011
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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.mass_balance_calculator.api.dependencies import (
    AuthUser,
    DateRangeParams,
    ErrorResponse,
    PaginationParams,
    get_date_range,
    get_mbc_service,
    get_pagination,
    get_request_id,
    rate_limit_batch,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_ledger_id,
)
from greenlang.agents.eudr.mass_balance_calculator.api.schemas import (
    BulkEntryResultSchema,
    BulkEntrySchema,
    ComplianceStatusSchema,
    CreateLedgerSchema,
    EntryHistorySchema,
    LedgerBalanceSchema,
    LedgerDetailSchema,
    LedgerEntryDetailSchema,
    LedgerEntryTypeSchema,
    LedgerListSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RecordEntrySchema,
    SearchLedgerSchema,
    StandardTypeSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ledgers"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_ledger_store: Dict[str, Dict] = {}
_entry_store: Dict[str, Dict] = {}
_ledger_entry_index: Dict[str, List[str]] = {}

def _get_ledger_store() -> Dict[str, Dict]:
    """Return the ledger store singleton."""
    return _ledger_store

def _get_entry_store() -> Dict[str, Dict]:
    """Return the entry store singleton."""
    return _entry_store

def _get_ledger_entry_index() -> Dict[str, List[str]]:
    """Return the ledger-to-entries index."""
    return _ledger_entry_index

def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# POST /ledgers
# ---------------------------------------------------------------------------

@router.post(
    "/ledgers",
    response_model=LedgerDetailSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new mass balance ledger",
    description=(
        "Create a new double-entry mass balance ledger for a facility and "
        "commodity combination. Each ledger tracks inputs, outputs, losses, "
        "and waste with SHA-256 provenance hashing per EUDR Article 14."
    ),
    responses={
        201: {"description": "Ledger created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_ledger(
    request: Request,
    body: CreateLedgerSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:ledgers:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> LedgerDetailSchema:
    """Create a new mass balance ledger.

    Args:
        body: Ledger creation parameters including facility_id, commodity,
            standard, and optional initial balance.
        user: Authenticated user with ledgers:create permission.

    Returns:
        LedgerDetailSchema with the newly created ledger details.
    """
    start = time.monotonic()
    try:
        ledger_id = str(uuid.uuid4())
        now = utcnow()

        provenance_data = body.model_dump(mode="json")
        provenance_data["created_by"] = user.user_id
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        ledger_record = {
            "ledger_id": ledger_id,
            "facility_id": body.facility_id,
            "commodity": body.commodity,
            "standard": body.standard,
            "period_id": None,
            "current_balance": body.initial_balance,
            "total_inputs": Decimal("0"),
            "total_outputs": Decimal("0"),
            "total_losses": Decimal("0"),
            "total_waste": Decimal("0"),
            "utilization_rate": 0.0,
            "compliance_status": ComplianceStatusSchema.COMPLIANT,
            "metadata": body.metadata,
            "provenance": provenance,
            "created_at": now,
            "updated_at": now,
        }

        store = _get_ledger_store()
        store[ledger_id] = ledger_record

        # Initialize entry index
        index = _get_ledger_entry_index()
        index[ledger_id] = []

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Ledger created: id=%s facility=%s commodity=%s standard=%s",
            ledger_id,
            body.facility_id,
            body.commodity,
            body.standard.value,
        )

        return LedgerDetailSchema(
            **ledger_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create ledger: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create ledger",
        )

# ---------------------------------------------------------------------------
# GET /ledgers/{ledger_id}
# ---------------------------------------------------------------------------

@router.get(
    "/ledgers/{ledger_id}",
    response_model=LedgerDetailSchema,
    summary="Get ledger details",
    description=(
        "Retrieve full details of a mass balance ledger including "
        "current balance, cumulative totals, utilization rate, "
        "compliance status, and provenance information."
    ),
    responses={
        200: {"description": "Ledger details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Ledger not found"},
    },
)
async def get_ledger(
    request: Request,
    ledger_id: str = Depends(validate_ledger_id),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:ledgers:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> LedgerDetailSchema:
    """Get details of a specific ledger.

    Args:
        ledger_id: Unique ledger identifier.
        user: Authenticated user with ledgers:read permission.

    Returns:
        LedgerDetailSchema with full ledger details.

    Raises:
        HTTPException: 404 if ledger not found.
    """
    start = time.monotonic()
    try:
        store = _get_ledger_store()
        record = store.get(ledger_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ledger {ledger_id} not found",
            )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return LedgerDetailSchema(
            **record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get ledger %s: %s", ledger_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve ledger",
        )

# ---------------------------------------------------------------------------
# POST /ledgers/entries
# ---------------------------------------------------------------------------

@router.post(
    "/ledgers/entries",
    response_model=LedgerEntryDetailSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Record a ledger entry",
    description=(
        "Record a single double-entry ledger entry (input, output, "
        "adjustment, loss, waste, carry-forward, or expiry). Entries "
        "are immutable once created. Balance is updated atomically."
    ),
    responses={
        201: {"description": "Entry recorded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Ledger not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def record_entry(
    request: Request,
    body: RecordEntrySchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:ledgers:entries:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> LedgerEntryDetailSchema:
    """Record a single ledger entry.

    Args:
        body: Entry parameters including ledger_id, entry_type,
            quantity_kg, and optional batch_id.
        user: Authenticated user with entries:create permission.

    Returns:
        LedgerEntryDetailSchema with the recorded entry details.

    Raises:
        HTTPException: 404 if target ledger not found.
    """
    start = time.monotonic()
    try:
        # Verify ledger exists
        ledger_store = _get_ledger_store()
        ledger = ledger_store.get(body.ledger_id)
        if ledger is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ledger {body.ledger_id} not found",
            )

        entry_id = str(uuid.uuid4())
        now = utcnow()

        provenance_data = body.model_dump(mode="json")
        provenance_data["entry_id"] = entry_id
        provenance_data["recorded_by"] = user.user_id

        entry_record = {
            "entry_id": entry_id,
            "ledger_id": body.ledger_id,
            "entry_type": body.entry_type,
            "quantity_kg": body.quantity_kg,
            "batch_id": body.batch_id,
            "source_destination": body.source_destination,
            "conversion_factor_applied": body.conversion_factor,
            "compliance_status": ComplianceStatusSchema.COMPLIANT,
            "operator_id": body.operator_id or user.user_id,
            "notes": body.notes,
            "voided": False,
            "voided_at": None,
            "voided_by": None,
            "void_reason": None,
            "metadata": body.metadata,
            "timestamp": now,
            "created_at": now,
        }

        # Update ledger balance based on entry type
        _update_ledger_balance(ledger, body.entry_type, body.quantity_kg)
        ledger["updated_at"] = now

        # Store entry and update index
        entry_store = _get_entry_store()
        entry_store[entry_id] = entry_record

        index = _get_ledger_entry_index()
        if body.ledger_id not in index:
            index[body.ledger_id] = []
        index[body.ledger_id].append(entry_id)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Ledger entry recorded: id=%s ledger=%s type=%s qty=%s",
            entry_id,
            body.ledger_id,
            body.entry_type.value,
            body.quantity_kg,
        )

        return LedgerEntryDetailSchema(**entry_record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record entry: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record ledger entry",
        )

def _update_ledger_balance(
    ledger: Dict[str, Any],
    entry_type: LedgerEntryTypeSchema,
    quantity_kg: Decimal,
) -> None:
    """Update ledger balance based on entry type.

    Deterministic Decimal arithmetic only -- zero hallucination.

    Args:
        ledger: Mutable ledger record dict.
        entry_type: Type of entry being recorded.
        quantity_kg: Quantity in kilograms.
    """
    if entry_type in (
        LedgerEntryTypeSchema.INPUT,
        LedgerEntryTypeSchema.CARRY_FORWARD_IN,
    ):
        ledger["current_balance"] += quantity_kg
        ledger["total_inputs"] += quantity_kg
    elif entry_type in (
        LedgerEntryTypeSchema.OUTPUT,
        LedgerEntryTypeSchema.CARRY_FORWARD_OUT,
    ):
        ledger["current_balance"] -= quantity_kg
        ledger["total_outputs"] += quantity_kg
    elif entry_type == LedgerEntryTypeSchema.LOSS:
        ledger["current_balance"] -= quantity_kg
        ledger["total_losses"] += quantity_kg
    elif entry_type == LedgerEntryTypeSchema.WASTE:
        ledger["current_balance"] -= quantity_kg
        ledger["total_waste"] += quantity_kg
    elif entry_type == LedgerEntryTypeSchema.EXPIRY:
        ledger["current_balance"] -= quantity_kg
    elif entry_type == LedgerEntryTypeSchema.ADJUSTMENT:
        ledger["current_balance"] += quantity_kg

    # Recalculate utilization rate
    total_in = ledger["total_inputs"]
    if total_in > Decimal("0"):
        total_out = ledger["total_outputs"]
        ledger["utilization_rate"] = float(total_out / total_in)

# ---------------------------------------------------------------------------
# POST /ledgers/entries/bulk
# ---------------------------------------------------------------------------

@router.post(
    "/ledgers/entries/bulk",
    response_model=BulkEntryResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk import ledger entries",
    description=(
        "Import up to 500 ledger entries in a single request. Supports "
        "validate-only mode for dry-run validation without persistence. "
        "Each entry is validated independently; partial success is allowed."
    ),
    responses={
        201: {"description": "Bulk import processed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def bulk_import_entries(
    request: Request,
    body: BulkEntrySchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:ledgers:entries:bulk")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BulkEntryResultSchema:
    """Bulk import multiple ledger entries.

    Args:
        body: Bulk entry request with list of entries and options.
        user: Authenticated user with entries:bulk permission.

    Returns:
        BulkEntryResultSchema with accepted/rejected counts and details.
    """
    start = time.monotonic()
    try:
        now = utcnow()
        accepted_entries: List[LedgerEntryDetailSchema] = []
        errors: List[Dict[str, Any]] = []
        ledger_store = _get_ledger_store()
        entry_store = _get_entry_store()
        index = _get_ledger_entry_index()

        for idx, entry_req in enumerate(body.entries):
            try:
                # Verify ledger exists
                ledger = ledger_store.get(entry_req.ledger_id)
                if ledger is None:
                    errors.append({
                        "index": idx,
                        "ledger_id": entry_req.ledger_id,
                        "error": f"Ledger {entry_req.ledger_id} not found",
                    })
                    continue

                entry_id = str(uuid.uuid4())
                operator = body.operator_id or entry_req.operator_id or user.user_id

                entry_record = {
                    "entry_id": entry_id,
                    "ledger_id": entry_req.ledger_id,
                    "entry_type": entry_req.entry_type,
                    "quantity_kg": entry_req.quantity_kg,
                    "batch_id": entry_req.batch_id,
                    "source_destination": entry_req.source_destination,
                    "conversion_factor_applied": entry_req.conversion_factor,
                    "compliance_status": ComplianceStatusSchema.COMPLIANT,
                    "operator_id": operator,
                    "notes": entry_req.notes,
                    "voided": False,
                    "voided_at": None,
                    "voided_by": None,
                    "void_reason": None,
                    "metadata": entry_req.metadata,
                    "timestamp": now,
                    "created_at": now,
                }

                if not body.validate_only:
                    _update_ledger_balance(
                        ledger, entry_req.entry_type, entry_req.quantity_kg
                    )
                    ledger["updated_at"] = now
                    entry_store[entry_id] = entry_record
                    if entry_req.ledger_id not in index:
                        index[entry_req.ledger_id] = []
                    index[entry_req.ledger_id].append(entry_id)

                accepted_entries.append(LedgerEntryDetailSchema(**entry_record))

            except Exception as entry_exc:
                errors.append({
                    "index": idx,
                    "ledger_id": entry_req.ledger_id,
                    "error": str(entry_exc),
                })

        provenance_hash = _compute_provenance_hash({
            "total": len(body.entries),
            "accepted": len(accepted_entries),
            "rejected": len(errors),
            "operator": user.user_id,
        })
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Bulk import processed: total=%d accepted=%d rejected=%d validate_only=%s",
            len(body.entries),
            len(accepted_entries),
            len(errors),
            body.validate_only,
        )

        return BulkEntryResultSchema(
            total_submitted=len(body.entries),
            total_accepted=len(accepted_entries),
            total_rejected=len(errors),
            entries=accepted_entries,
            errors=errors,
            validate_only=body.validate_only,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to process bulk import: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process bulk entry import",
        )

# ---------------------------------------------------------------------------
# GET /ledgers/{ledger_id}/balance
# ---------------------------------------------------------------------------

@router.get(
    "/ledgers/{ledger_id}/balance",
    response_model=LedgerBalanceSchema,
    summary="Get current ledger balance",
    description=(
        "Retrieve the current balance of a ledger including cumulative "
        "inputs, outputs, losses, waste, utilization rate, carry-forward "
        "availability, and overdraft status."
    ),
    responses={
        200: {"description": "Balance retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Ledger not found"},
    },
)
async def get_balance(
    request: Request,
    ledger_id: str = Depends(validate_ledger_id),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:ledgers:balance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> LedgerBalanceSchema:
    """Get the current balance of a ledger.

    Args:
        ledger_id: Unique ledger identifier.
        user: Authenticated user with balance:read permission.

    Returns:
        LedgerBalanceSchema with detailed balance information.

    Raises:
        HTTPException: 404 if ledger not found.
    """
    start = time.monotonic()
    try:
        store = _get_ledger_store()
        record = store.get(ledger_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ledger {ledger_id} not found",
            )

        now = utcnow()
        provenance_hash = _compute_provenance_hash({
            "ledger_id": ledger_id,
            "balance": str(record["current_balance"]),
            "timestamp": str(now),
        })
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        overdraft_status = record["current_balance"] < Decimal("0")
        elapsed_ms = (time.monotonic() - start) * 1000.0

        return LedgerBalanceSchema(
            ledger_id=ledger_id,
            current_balance=record["current_balance"],
            total_inputs=record["total_inputs"],
            total_outputs=record["total_outputs"],
            total_losses=record["total_losses"],
            total_waste=record["total_waste"],
            utilization_rate=record["utilization_rate"],
            carry_forward_available=Decimal("0"),
            overdraft_status=overdraft_status,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
            timestamp=now,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get balance for %s: %s", ledger_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve ledger balance",
        )

# ---------------------------------------------------------------------------
# GET /ledgers/{ledger_id}/history
# ---------------------------------------------------------------------------

@router.get(
    "/ledgers/{ledger_id}/history",
    response_model=EntryHistorySchema,
    summary="Get entry history",
    description=(
        "Retrieve the entry history for a ledger with optional filters "
        "by entry type, date range, and pagination."
    ),
    responses={
        200: {"description": "History retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Ledger not found"},
    },
)
async def get_history(
    request: Request,
    ledger_id: str = Depends(validate_ledger_id),
    entry_type: Optional[LedgerEntryTypeSchema] = Query(
        None, description="Filter by entry type"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    date_range: DateRangeParams = Depends(get_date_range),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:ledgers:history:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> EntryHistorySchema:
    """Get entry history for a ledger.

    Args:
        ledger_id: Unique ledger identifier.
        entry_type: Optional filter by entry type.
        pagination: Pagination parameters.
        date_range: Optional date range filter.
        user: Authenticated user with history:read permission.

    Returns:
        EntryHistorySchema with matching entries and pagination metadata.

    Raises:
        HTTPException: 404 if ledger not found.
    """
    start = time.monotonic()
    try:
        ledger_store = _get_ledger_store()
        if ledger_id not in ledger_store:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ledger {ledger_id} not found",
            )

        index = _get_ledger_entry_index()
        entry_ids = index.get(ledger_id, [])
        entry_store = _get_entry_store()

        # Collect and filter entries
        all_entries = []
        for eid in entry_ids:
            entry = entry_store.get(eid)
            if entry is None:
                continue
            # Filter by entry type
            if entry_type is not None and entry["entry_type"] != entry_type:
                continue
            # Filter by date range
            if date_range.start_date and entry["timestamp"] < date_range.start_date:
                continue
            if date_range.end_date and entry["timestamp"] > date_range.end_date:
                continue
            all_entries.append(entry)

        # Sort by timestamp descending
        all_entries.sort(key=lambda e: e["timestamp"], reverse=True)

        total = len(all_entries)
        paginated = all_entries[pagination.offset: pagination.offset + pagination.limit]
        has_more = (pagination.offset + pagination.limit) < total

        entries = [LedgerEntryDetailSchema(**e) for e in paginated]
        meta = PaginatedMeta(
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
            has_more=has_more,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return EntryHistorySchema(
            ledger_id=ledger_id,
            entries=entries,
            pagination=meta,
            processing_time_ms=elapsed_ms,
            timestamp=utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get history for %s: %s", ledger_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve entry history",
        )

# ---------------------------------------------------------------------------
# POST /ledgers/search
# ---------------------------------------------------------------------------

@router.post(
    "/ledgers/search",
    response_model=LedgerListSchema,
    summary="Search ledgers",
    description=(
        "Search ledgers by facility, commodity, standard, balance range, "
        "compliance status, and other criteria. Supports sorting and "
        "pagination."
    ),
    responses={
        200: {"description": "Search results returned"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def search_ledgers(
    request: Request,
    body: SearchLedgerSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:ledgers:search")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> LedgerListSchema:
    """Search ledgers by criteria.

    Args:
        body: Search criteria including facility_id, commodity, standard,
            balance range, compliance status, and pagination.
        user: Authenticated user with ledgers:search permission.

    Returns:
        LedgerListSchema with matching ledgers and pagination metadata.
    """
    start = time.monotonic()
    try:
        store = _get_ledger_store()
        results = []

        for ledger in store.values():
            # Apply filters
            if body.facility_id and ledger["facility_id"] != body.facility_id:
                continue
            if body.commodity and ledger["commodity"] != body.commodity.lower():
                continue
            if body.standard and ledger["standard"] != body.standard:
                continue
            if body.min_balance is not None and ledger["current_balance"] < body.min_balance:
                continue
            if body.max_balance is not None and ledger["current_balance"] > body.max_balance:
                continue
            if body.compliance_status and ledger["compliance_status"] != body.compliance_status:
                continue
            results.append(ledger)

        # Sort results
        reverse = body.sort_order.value == "desc"
        sort_key = body.sort_by
        results.sort(
            key=lambda r: r.get(sort_key, ""),
            reverse=reverse,
        )

        total = len(results)
        paginated = results[body.offset: body.offset + body.limit]
        has_more = (body.offset + body.limit) < total

        ledger_schemas = [LedgerDetailSchema(**r) for r in paginated]
        meta = PaginatedMeta(
            total=total,
            limit=body.limit,
            offset=body.offset,
            has_more=has_more,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Ledger search: filters=%d results=%d",
            sum(1 for v in [
                body.facility_id, body.commodity, body.standard,
                body.min_balance, body.max_balance, body.compliance_status,
            ] if v is not None),
            total,
        )

        return LedgerListSchema(
            ledgers=ledger_schemas,
            pagination=meta,
            processing_time_ms=elapsed_ms,
            timestamp=utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to search ledgers: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search ledgers",
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
]
