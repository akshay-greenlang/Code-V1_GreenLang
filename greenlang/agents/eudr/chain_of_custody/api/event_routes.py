# -*- coding: utf-8 -*-
"""
Event Routes - AGENT-EUDR-009 Chain of Custody API

Endpoints for recording, querying, and amending custody events.
Events form the immutable audit trail of every custody transfer,
receipt, inspection, and transformation along the supply chain.

Endpoints:
    POST   /events                  - Record a new custody event
    POST   /events/batch            - Bulk event import
    GET    /events/{event_id}       - Get event details
    GET    /events/chain/{batch_id} - Get event chain for a batch
    POST   /events/amend/{event_id} - Amend event (immutable append)

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
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.chain_of_custody.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_coc_service,
    get_request_id,
    rate_limit_batch,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_batch_id,
    validate_event_id,
)
from greenlang.agents.eudr.chain_of_custody.api.schemas import (
    CustodyEventAmendRequest,
    CustodyEventAmendResponse,
    CustodyEventBatchRequest,
    CustodyEventBatchResponse,
    CustodyEventChainResponse,
    CustodyEventCreateRequest,
    CustodyEventResponse,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Custody Events"])

# ---------------------------------------------------------------------------
# In-memory event store (replaced by database in production)
# ---------------------------------------------------------------------------

_event_store: Dict[str, Dict] = {}
_batch_event_index: Dict[str, List[str]] = {}


def _get_event_store() -> Dict[str, Dict]:
    """Return the event store singleton. Replaceable for testing."""
    return _event_store


def _get_batch_event_index() -> Dict[str, List[str]]:
    """Return the batch-to-event index. Replaceable for testing."""
    return _batch_event_index


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /events
# ---------------------------------------------------------------------------


@router.post(
    "/events",
    response_model=CustodyEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a custody event",
    description=(
        "Record a new chain-of-custody event such as receipt, transfer, "
        "inspection, or transformation. Events are immutable once created."
    ),
    responses={
        201: {"description": "Event recorded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_event(
    request: Request,
    body: CustodyEventCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:events:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CustodyEventResponse:
    """Record a new custody event for a batch.

    Args:
        body: Event creation parameters.
        user: Authenticated user with events:create permission.

    Returns:
        CustodyEventResponse with the new event details and provenance.
    """
    start = time.monotonic()
    try:
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)
        event_timestamp = body.timestamp or now

        # Build provenance hash from input data
        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        # Store event
        event_record = {
            "event_id": event_id,
            "event_type": body.event_type,
            "batch_id": body.batch_id,
            "facility_id": body.facility_id,
            "commodity": body.commodity,
            "quantity": body.quantity,
            "timestamp": event_timestamp,
            "source_facility_id": body.source_facility_id,
            "destination_facility_id": body.destination_facility_id,
            "transport_mode": body.transport_mode,
            "transport_document_ref": body.transport_document_ref,
            "location": body.location,
            "custody_model": body.custody_model,
            "notes": body.notes,
            "metadata": body.metadata,
            "is_amendment": False,
            "amends_event_id": None,
            "amendment_reason": None,
            "provenance": provenance,
        }

        store = _get_event_store()
        store[event_id] = event_record

        # Update batch-to-event index
        index = _get_batch_event_index()
        if body.batch_id not in index:
            index[body.batch_id] = []
        index[body.batch_id].append(event_id)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Custody event created: id=%s type=%s batch=%s facility=%s",
            event_id,
            body.event_type.value,
            body.batch_id,
            body.facility_id,
        )

        return CustodyEventResponse(
            event_id=event_id,
            event_type=body.event_type,
            batch_id=body.batch_id,
            facility_id=body.facility_id,
            commodity=body.commodity,
            quantity=body.quantity,
            timestamp=event_timestamp,
            source_facility_id=body.source_facility_id,
            destination_facility_id=body.destination_facility_id,
            transport_mode=body.transport_mode,
            transport_document_ref=body.transport_document_ref,
            location=body.location,
            custody_model=body.custody_model,
            notes=body.notes,
            metadata=body.metadata,
            is_amendment=False,
            amends_event_id=None,
            amendment_reason=None,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create custody event: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record custody event",
        )


# ---------------------------------------------------------------------------
# POST /events/batch
# ---------------------------------------------------------------------------


@router.post(
    "/events/batch",
    response_model=CustodyEventBatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk event import",
    description=(
        "Import multiple custody events in a single request. "
        "Supports up to 1000 events per batch. Use validate_only=true "
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
async def batch_import_events(
    request: Request,
    body: CustodyEventBatchRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:events:create")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> CustodyEventBatchResponse:
    """Import multiple custody events in bulk.

    Args:
        body: Batch import request with list of events.
        user: Authenticated user with events:create permission.

    Returns:
        CustodyEventBatchResponse with accepted/rejected counts.
    """
    start = time.monotonic()
    try:
        accepted: List[CustodyEventResponse] = []
        errors: List[Dict] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)
        store = _get_event_store()
        index = _get_batch_event_index()

        for idx, event_req in enumerate(body.events):
            try:
                event_id = str(uuid.uuid4())
                event_timestamp = event_req.timestamp or now

                provenance_data = event_req.model_dump(mode="json")
                provenance_hash = _compute_provenance_hash(provenance_data)

                provenance = ProvenanceInfo(
                    provenance_hash=provenance_hash,
                    created_by=user.user_id,
                    created_at=now,
                    source="batch_import",
                )

                if not body.validate_only:
                    event_record = {
                        "event_id": event_id,
                        "event_type": event_req.event_type,
                        "batch_id": event_req.batch_id,
                        "facility_id": event_req.facility_id,
                        "commodity": event_req.commodity,
                        "quantity": event_req.quantity,
                        "timestamp": event_timestamp,
                        "source_facility_id": event_req.source_facility_id,
                        "destination_facility_id": event_req.destination_facility_id,
                        "transport_mode": event_req.transport_mode,
                        "transport_document_ref": event_req.transport_document_ref,
                        "location": event_req.location,
                        "custody_model": event_req.custody_model,
                        "notes": event_req.notes,
                        "metadata": event_req.metadata,
                        "is_amendment": False,
                        "amends_event_id": None,
                        "amendment_reason": None,
                        "provenance": provenance,
                    }
                    store[event_id] = event_record

                    if event_req.batch_id not in index:
                        index[event_req.batch_id] = []
                    index[event_req.batch_id].append(event_id)

                accepted.append(
                    CustodyEventResponse(
                        event_id=event_id,
                        event_type=event_req.event_type,
                        batch_id=event_req.batch_id,
                        facility_id=event_req.facility_id,
                        commodity=event_req.commodity,
                        quantity=event_req.quantity,
                        timestamp=event_timestamp,
                        source_facility_id=event_req.source_facility_id,
                        destination_facility_id=event_req.destination_facility_id,
                        transport_mode=event_req.transport_mode,
                        transport_document_ref=event_req.transport_document_ref,
                        location=event_req.location,
                        custody_model=event_req.custody_model,
                        notes=event_req.notes,
                        metadata=event_req.metadata,
                        is_amendment=False,
                        provenance=provenance,
                        processing_time_ms=0.0,
                    )
                )
            except Exception as item_exc:
                errors.append({
                    "index": idx,
                    "error": str(item_exc),
                    "batch_id": event_req.batch_id,
                })

        # Compute batch-level provenance
        batch_provenance_data = {
            "total": len(body.events),
            "accepted": len(accepted),
            "rejected": len(errors),
        }
        batch_hash = _compute_provenance_hash(batch_provenance_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch event import: total=%d accepted=%d rejected=%d validate_only=%s",
            len(body.events),
            len(accepted),
            len(errors),
            body.validate_only,
        )

        return CustodyEventBatchResponse(
            total_submitted=len(body.events),
            total_accepted=len(accepted),
            total_rejected=len(errors),
            events=accepted,
            errors=errors,
            validate_only=body.validate_only,
            provenance_hash=batch_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed batch event import: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch event import",
        )


# ---------------------------------------------------------------------------
# GET /events/{event_id}
# ---------------------------------------------------------------------------


@router.get(
    "/events/{event_id}",
    response_model=CustodyEventResponse,
    summary="Get event details",
    description="Retrieve full details of a custody event by its ID.",
    responses={
        200: {"description": "Event details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Event not found"},
    },
)
async def get_event(
    request: Request,
    event_id: str = Depends(validate_event_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:events:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CustodyEventResponse:
    """Get custody event details by ID.

    Args:
        event_id: Unique event identifier.
        user: Authenticated user with events:read permission.

    Returns:
        CustodyEventResponse with full event details.

    Raises:
        HTTPException: 404 if event not found.
    """
    try:
        store = _get_event_store()
        event_record = store.get(event_id)

        if event_record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Custody event {event_id} not found",
            )

        return CustodyEventResponse(**event_record, processing_time_ms=0.0)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve event %s: %s", event_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve custody event",
        )


# ---------------------------------------------------------------------------
# GET /events/chain/{batch_id}
# ---------------------------------------------------------------------------


@router.get(
    "/events/chain/{batch_id}",
    response_model=CustodyEventChainResponse,
    summary="Get event chain for batch",
    description=(
        "Retrieve the complete chain of custody events for a batch, "
        "ordered chronologically."
    ),
    responses={
        200: {"description": "Event chain"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No events found for batch"},
    },
)
async def get_event_chain(
    request: Request,
    batch_id: str = Depends(validate_batch_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:events:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CustodyEventChainResponse:
    """Get the complete event chain for a batch.

    Args:
        batch_id: Batch/lot identifier.
        user: Authenticated user with events:read permission.

    Returns:
        CustodyEventChainResponse with ordered events and chain metadata.

    Raises:
        HTTPException: 404 if no events found for the batch.
    """
    start = time.monotonic()
    try:
        store = _get_event_store()
        index = _get_batch_event_index()
        event_ids = index.get(batch_id, [])

        if not event_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No custody events found for batch {batch_id}",
            )

        events: List[CustodyEventResponse] = []
        custody_models_set: set = set()
        facilities_set: set = set()
        has_amendments = False

        for eid in event_ids:
            record = store.get(eid)
            if record is not None:
                events.append(
                    CustodyEventResponse(**record, processing_time_ms=0.0)
                )
                if record.get("custody_model"):
                    custody_models_set.add(record["custody_model"])
                facilities_set.add(record["facility_id"])
                if record.get("is_amendment"):
                    has_amendments = True

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        chain_start = events[0].timestamp if events else None
        chain_end = events[-1].timestamp if events else None
        chain_duration = None
        if chain_start and chain_end:
            delta = chain_end - chain_start
            chain_duration = delta.total_seconds() / 3600.0

        chain_data = {
            "batch_id": batch_id,
            "total_events": len(events),
            "facilities": list(facilities_set),
        }
        provenance_hash = _compute_provenance_hash(chain_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Event chain retrieved: batch=%s events=%d",
            batch_id,
            len(events),
        )

        return CustodyEventChainResponse(
            batch_id=batch_id,
            events=events,
            total_events=len(events),
            chain_start=chain_start,
            chain_end=chain_end,
            chain_duration_hours=chain_duration,
            custody_models_used=list(custody_models_set),
            facilities_involved=list(facilities_set),
            has_amendments=has_amendments,
            provenance_hash=provenance_hash,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve event chain for batch %s: %s",
            batch_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve event chain",
        )


# ---------------------------------------------------------------------------
# POST /events/amend/{event_id}
# ---------------------------------------------------------------------------


@router.post(
    "/events/amend/{event_id}",
    response_model=CustodyEventAmendResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Amend a custody event (immutable)",
    description=(
        "Create an amendment to an existing custody event. The original "
        "event is preserved unchanged; the amendment is appended as a "
        "new immutable record linked to the original."
    ),
    responses={
        201: {"description": "Amendment created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Original event not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def amend_event(
    request: Request,
    body: CustodyEventAmendRequest,
    event_id: str = Depends(validate_event_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:events:amend")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CustodyEventAmendResponse:
    """Amend an existing custody event with an immutable correction.

    Args:
        body: Amendment details (reason, corrected values).
        event_id: Original event ID to amend.
        user: Authenticated user with events:amend permission.

    Returns:
        CustodyEventAmendResponse with the new amendment event.

    Raises:
        HTTPException: 404 if original event not found.
    """
    start = time.monotonic()
    try:
        store = _get_event_store()
        original = store.get(event_id)

        if original is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Original event {event_id} not found",
            )

        amendment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Build provenance hash
        amend_data = {
            "original_event_id": event_id,
            "amendment": body.model_dump(mode="json"),
        }
        provenance_hash = _compute_provenance_hash(amend_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api_amendment",
        )

        # Create amendment event record
        amendment_record = {
            **original,
            "event_id": amendment_id,
            "is_amendment": True,
            "amends_event_id": event_id,
            "amendment_reason": body.reason,
            "timestamp": now,
            "provenance": provenance,
        }

        # Apply corrections
        if body.corrected_quantity is not None:
            amendment_record["quantity"] = body.corrected_quantity
        if body.corrected_timestamp is not None:
            amendment_record["timestamp"] = body.corrected_timestamp
        if body.corrected_facility_id is not None:
            amendment_record["facility_id"] = body.corrected_facility_id
        if body.corrected_notes is not None:
            amendment_record["notes"] = body.corrected_notes
        if body.corrected_metadata is not None:
            amendment_record["metadata"] = body.corrected_metadata

        store[amendment_id] = amendment_record

        # Add to batch index
        batch_id = original.get("batch_id", "")
        if batch_id:
            index = _get_batch_event_index()
            if batch_id not in index:
                index[batch_id] = []
            index[batch_id].append(amendment_id)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Event amended: original=%s amendment=%s reason=%s",
            event_id,
            amendment_id,
            body.reason[:100],
        )

        return CustodyEventAmendResponse(
            amendment_event_id=amendment_id,
            original_event_id=event_id,
            reason=body.reason,
            status="amended",
            amended_at=now,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to amend event %s: %s", event_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to amend custody event",
        )
