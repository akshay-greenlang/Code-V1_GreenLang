# -*- coding: utf-8 -*-
"""
Batch Routes - AGENT-EUDR-009 Chain of Custody API

Endpoints for batch/lot lifecycle management including creation,
splitting, merging, blending, genealogy queries, and search.

Endpoints:
    POST   /batches                     - Create a new batch
    GET    /batches/{batch_id}           - Get batch details
    POST   /batches/split               - Split a batch
    POST   /batches/merge               - Merge batches
    POST   /batches/blend               - Blend batches
    GET    /batches/{batch_id}/genealogy - Get batch genealogy
    POST   /batches/search              - Search batches

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

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.chain_of_custody.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_coc_service,
    get_request_id,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_batch_id,
)
from greenlang.agents.eudr.chain_of_custody.api.schemas import (
    BatchBlendRequest,
    BatchBlendResponse,
    BatchCreateRequest,
    BatchGenealogyResponse,
    BatchMergeRequest,
    BatchMergeResponse,
    BatchResponse,
    BatchSearchRequest,
    BatchSearchResponse,
    BatchSplitRequest,
    BatchSplitResponse,
    BatchStatus,
    BatchOriginType,
    GenealogyNode,
    PaginatedMeta,
    ProvenanceInfo,
    QuantitySpec,
    UnitOfMeasure,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Batch Management"])

# ---------------------------------------------------------------------------
# In-memory batch store (replaced by database in production)
# ---------------------------------------------------------------------------

_batch_store: Dict[str, Dict] = {}


def _get_batch_store() -> Dict[str, Dict]:
    """Return the batch store singleton. Replaceable for testing."""
    return _batch_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _zero_quantity(unit: UnitOfMeasure) -> QuantitySpec:
    """Create a zero-quantity spec with the given unit."""
    return QuantitySpec(amount=Decimal("0"), unit=unit)


# ---------------------------------------------------------------------------
# POST /batches
# ---------------------------------------------------------------------------


@router.post(
    "/batches",
    response_model=BatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new batch",
    description=(
        "Create a new batch/lot representing a discrete quantity of "
        "an EUDR commodity with full origin traceability."
    ),
    responses={
        201: {"description": "Batch created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_batch(
    request: Request,
    body: BatchCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:batches:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BatchResponse:
    """Create a new batch with origin information and CoC model.

    Args:
        body: Batch creation parameters.
        user: Authenticated user with batches:create permission.

    Returns:
        BatchResponse with the new batch details and provenance.
    """
    start = time.monotonic()
    try:
        batch_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        batch_record = {
            "batch_id": batch_id,
            "batch_reference": body.batch_reference,
            "commodity": body.commodity,
            "facility_id": body.facility_id,
            "status": BatchStatus.CREATED,
            "origin_type": body.origin_type,
            "quantity": body.quantity,
            "original_quantity": body.quantity,
            "origin_country": body.origin_country,
            "origin_region": body.origin_region,
            "origin_plot_ids": body.origin_plot_ids,
            "harvest_date": body.harvest_date,
            "custody_model": body.custody_model,
            "certifications": body.certifications,
            "supplier_id": body.supplier_id,
            "parent_batch_ids": [],
            "child_batch_ids": [],
            "event_count": 0,
            "notes": body.notes,
            "metadata": body.metadata,
            "created_at": now,
            "updated_at": now,
            "provenance": provenance,
        }

        store = _get_batch_store()
        store[batch_id] = batch_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch created: id=%s commodity=%s facility=%s origin=%s model=%s",
            batch_id,
            body.commodity.value,
            body.facility_id,
            body.origin_country,
            body.custody_model.value,
        )

        return BatchResponse(
            **batch_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create batch: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch",
        )


# ---------------------------------------------------------------------------
# GET /batches/{batch_id}
# ---------------------------------------------------------------------------


@router.get(
    "/batches/{batch_id}",
    response_model=BatchResponse,
    summary="Get batch details",
    description="Retrieve full batch details including origin and genealogy links.",
    responses={
        200: {"description": "Batch details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Batch not found"},
    },
)
async def get_batch(
    request: Request,
    batch_id: str = Depends(validate_batch_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:batches:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchResponse:
    """Get batch details by ID.

    Args:
        batch_id: Unique batch identifier.
        user: Authenticated user with batches:read permission.

    Returns:
        BatchResponse with full batch details.

    Raises:
        HTTPException: 404 if batch not found.
    """
    try:
        store = _get_batch_store()
        record = store.get(batch_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found",
            )

        return BatchResponse(**record, processing_time_ms=0.0)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve batch %s: %s", batch_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve batch",
        )


# ---------------------------------------------------------------------------
# POST /batches/split
# ---------------------------------------------------------------------------


@router.post(
    "/batches/split",
    response_model=BatchSplitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Split a batch",
    description=(
        "Split a batch into multiple child batches. The source batch "
        "is marked as SPLIT and child batches inherit origin traceability."
    ),
    responses={
        201: {"description": "Batch split successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Source batch not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def split_batch(
    request: Request,
    body: BatchSplitRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:batches:split")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BatchSplitResponse:
    """Split a batch into multiple child batches.

    Args:
        body: Split parameters with source batch and allocations.
        user: Authenticated user with batches:split permission.

    Returns:
        BatchSplitResponse with child batches created.

    Raises:
        HTTPException: 404 if source batch not found, 400 if invalid.
    """
    start = time.monotonic()
    try:
        store = _get_batch_store()
        source = store.get(body.source_batch_id)

        if source is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source batch {body.source_batch_id} not found",
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        child_batches: List[BatchResponse] = []
        child_ids: List[str] = []

        for split_part in body.splits:
            child_id = str(uuid.uuid4())
            child_ids.append(child_id)

            child_provenance_data = {
                "source_batch_id": body.source_batch_id,
                "child_id": child_id,
                "quantity": str(split_part.quantity.amount),
            }
            provenance_hash = _compute_provenance_hash(child_provenance_data)
            provenance = ProvenanceInfo(
                provenance_hash=provenance_hash,
                created_by=user.user_id,
                created_at=now,
                source="batch_split",
            )

            facility = split_part.destination_facility_id or source["facility_id"]

            child_record = {
                "batch_id": child_id,
                "batch_reference": split_part.batch_reference,
                "commodity": source["commodity"],
                "facility_id": facility,
                "status": BatchStatus.ACTIVE,
                "origin_type": BatchOriginType.SPLIT,
                "quantity": split_part.quantity,
                "original_quantity": split_part.quantity,
                "origin_country": source["origin_country"],
                "origin_region": source.get("origin_region"),
                "origin_plot_ids": source.get("origin_plot_ids", []),
                "harvest_date": source.get("harvest_date"),
                "custody_model": source["custody_model"],
                "certifications": source.get("certifications", []),
                "supplier_id": source.get("supplier_id"),
                "parent_batch_ids": [body.source_batch_id],
                "child_batch_ids": [],
                "event_count": 0,
                "notes": split_part.notes,
                "metadata": None,
                "created_at": now,
                "updated_at": now,
                "provenance": provenance,
            }

            store[child_id] = child_record
            child_batches.append(
                BatchResponse(**child_record, processing_time_ms=0.0)
            )

        # Update source batch
        source["status"] = BatchStatus.SPLIT
        source["child_batch_ids"] = child_ids
        source["updated_at"] = now

        split_provenance = _compute_provenance_hash({
            "source_batch_id": body.source_batch_id,
            "child_ids": child_ids,
            "reason": body.reason,
        })

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch split: source=%s children=%d reason=%s",
            body.source_batch_id,
            len(child_batches),
            body.reason[:100],
        )

        return BatchSplitResponse(
            source_batch_id=body.source_batch_id,
            child_batches=child_batches,
            total_splits=len(child_batches),
            source_status=BatchStatus.SPLIT,
            provenance_hash=split_provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to split batch: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to split batch",
        )


# ---------------------------------------------------------------------------
# POST /batches/merge
# ---------------------------------------------------------------------------


@router.post(
    "/batches/merge",
    response_model=BatchMergeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Merge batches",
    description=(
        "Merge multiple batches into a single batch. Source batches are "
        "marked as MERGED. All must share the same commodity and CoC model."
    ),
    responses={
        201: {"description": "Batches merged successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Source batch not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def merge_batches(
    request: Request,
    body: BatchMergeRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:batches:merge")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BatchMergeResponse:
    """Merge multiple batches into one.

    Args:
        body: Merge parameters with source batch IDs.
        user: Authenticated user with batches:merge permission.

    Returns:
        BatchMergeResponse with the merged batch.

    Raises:
        HTTPException: 404 if source batch not found, 400 if invalid.
    """
    start = time.monotonic()
    try:
        store = _get_batch_store()
        source_records: List[Dict] = []

        for src_id in body.source_batch_ids:
            record = store.get(src_id)
            if record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Source batch {src_id} not found",
                )
            source_records.append(record)

        # Validate all sources share commodity
        commodities = {r["commodity"] for r in source_records}
        if len(commodities) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All source batches must share the same commodity",
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        merged_id = str(uuid.uuid4())

        # Sum quantities (use unit from first batch)
        first_unit = source_records[0]["quantity"].unit
        total_amount = sum(
            r["quantity"].amount for r in source_records
        )
        total_quantity = QuantitySpec(amount=total_amount, unit=first_unit)

        # Collect origin info
        all_plot_ids = []
        all_certs = []
        origin_countries = set()
        for r in source_records:
            all_plot_ids.extend(r.get("origin_plot_ids", []))
            all_certs.extend(r.get("certifications", []))
            origin_countries.add(r["origin_country"])

        provenance_data = {
            "source_batch_ids": body.source_batch_ids,
            "merged_id": merged_id,
            "total_quantity": str(total_amount),
        }
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="batch_merge",
        )

        merged_record = {
            "batch_id": merged_id,
            "batch_reference": body.batch_reference,
            "commodity": source_records[0]["commodity"],
            "facility_id": body.destination_facility_id,
            "status": BatchStatus.ACTIVE,
            "origin_type": BatchOriginType.MERGE,
            "quantity": total_quantity,
            "original_quantity": total_quantity,
            "origin_country": list(origin_countries)[0] if len(origin_countries) == 1 else source_records[0]["origin_country"],
            "origin_region": source_records[0].get("origin_region"),
            "origin_plot_ids": list(set(all_plot_ids)),
            "harvest_date": source_records[0].get("harvest_date"),
            "custody_model": source_records[0]["custody_model"],
            "certifications": list(set(all_certs)),
            "supplier_id": source_records[0].get("supplier_id"),
            "parent_batch_ids": body.source_batch_ids,
            "child_batch_ids": [],
            "event_count": 0,
            "notes": body.notes,
            "metadata": None,
            "created_at": now,
            "updated_at": now,
            "provenance": provenance,
        }

        store[merged_id] = merged_record

        # Mark source batches as merged
        for src_id in body.source_batch_ids:
            store[src_id]["status"] = BatchStatus.MERGED
            store[src_id]["updated_at"] = now

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batches merged: sources=%s merged_id=%s quantity=%s",
            body.source_batch_ids,
            merged_id,
            str(total_amount),
        )

        return BatchMergeResponse(
            merged_batch=BatchResponse(**merged_record, processing_time_ms=0.0),
            source_batch_ids=body.source_batch_ids,
            total_quantity=total_quantity,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to merge batches: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to merge batches",
        )


# ---------------------------------------------------------------------------
# POST /batches/blend
# ---------------------------------------------------------------------------


@router.post(
    "/batches/blend",
    response_model=BatchBlendResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Blend batches",
    description=(
        "Blend batches using the Controlled Blending (CB) model. "
        "Blend ratios must sum to 1.0. Source batches are marked BLENDED."
    ),
    responses={
        201: {"description": "Batches blended successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Source batch not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def blend_batches(
    request: Request,
    body: BatchBlendRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:batches:blend")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BatchBlendResponse:
    """Blend batches using controlled blending model.

    Args:
        body: Blend parameters with source batches and ratios.
        user: Authenticated user with batches:blend permission.

    Returns:
        BatchBlendResponse with the blended batch.

    Raises:
        HTTPException: 404 if source batch not found, 400 if invalid.
    """
    start = time.monotonic()
    try:
        store = _get_batch_store()

        if len(body.source_batch_ids) != len(body.blend_ratios):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of source_batch_ids must match blend_ratios",
            )

        source_records: List[Dict] = []
        for src_id in body.source_batch_ids:
            record = store.get(src_id)
            if record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Source batch {src_id} not found",
                )
            source_records.append(record)

        now = datetime.now(timezone.utc).replace(microsecond=0)
        blended_id = str(uuid.uuid4())

        # Collect origin info
        all_plot_ids = []
        all_certs = []
        for r in source_records:
            all_plot_ids.extend(r.get("origin_plot_ids", []))
            all_certs.extend(r.get("certifications", []))

        provenance_data = {
            "source_batch_ids": body.source_batch_ids,
            "blend_ratios": [str(r) for r in body.blend_ratios],
            "blended_id": blended_id,
        }
        provenance_hash = _compute_provenance_hash(provenance_data)
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="batch_blend",
        )

        from greenlang.agents.eudr.chain_of_custody.api.schemas import (
            CustodyModelType,
        )

        blended_record = {
            "batch_id": blended_id,
            "batch_reference": body.batch_reference,
            "commodity": source_records[0]["commodity"],
            "facility_id": body.destination_facility_id,
            "status": BatchStatus.ACTIVE,
            "origin_type": BatchOriginType.BLEND,
            "quantity": body.total_output_quantity,
            "original_quantity": body.total_output_quantity,
            "origin_country": source_records[0]["origin_country"],
            "origin_region": source_records[0].get("origin_region"),
            "origin_plot_ids": list(set(all_plot_ids)),
            "harvest_date": None,
            "custody_model": CustodyModelType.CONTROLLED_BLENDING,
            "certifications": list(set(all_certs)),
            "supplier_id": None,
            "parent_batch_ids": body.source_batch_ids,
            "child_batch_ids": [],
            "event_count": 0,
            "notes": body.notes,
            "metadata": None,
            "created_at": now,
            "updated_at": now,
            "provenance": provenance,
        }

        store[blended_id] = blended_record

        # Mark source batches as blended
        for src_id in body.source_batch_ids:
            store[src_id]["status"] = BatchStatus.BLENDED
            store[src_id]["updated_at"] = now

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batches blended: sources=%s blended_id=%s ratios=%s",
            body.source_batch_ids,
            blended_id,
            [str(r) for r in body.blend_ratios],
        )

        return BatchBlendResponse(
            blended_batch=BatchResponse(**blended_record, processing_time_ms=0.0),
            source_batch_ids=body.source_batch_ids,
            blend_ratios=body.blend_ratios,
            total_output_quantity=body.total_output_quantity,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to blend batches: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to blend batches",
        )


# ---------------------------------------------------------------------------
# GET /batches/{batch_id}/genealogy
# ---------------------------------------------------------------------------


@router.get(
    "/batches/{batch_id}/genealogy",
    response_model=BatchGenealogyResponse,
    summary="Get batch genealogy",
    description=(
        "Retrieve the full genealogy tree for a batch, including all "
        "parent and child relationships from splits, merges, and blends."
    ),
    responses={
        200: {"description": "Batch genealogy tree"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Batch not found"},
    },
)
async def get_batch_genealogy(
    request: Request,
    batch_id: str = Depends(validate_batch_id),
    user: AuthUser = Depends(
        require_permission("eudr-coc:batches:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchGenealogyResponse:
    """Get the genealogy tree for a batch.

    Args:
        batch_id: Root batch identifier.
        user: Authenticated user with batches:read permission.

    Returns:
        BatchGenealogyResponse with the genealogy tree.

    Raises:
        HTTPException: 404 if batch not found.
    """
    start = time.monotonic()
    try:
        store = _get_batch_store()
        root = store.get(batch_id)

        if root is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found",
            )

        # Build genealogy tree (BFS traversal)
        nodes: List[GenealogyNode] = []
        visited: set = set()
        queue = [(batch_id, 0)]
        all_countries: set = set()
        all_plots: set = set()
        all_models: set = set()
        max_depth = 0

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            record = store.get(current_id)
            if record is None:
                continue

            max_depth = max(max_depth, depth)
            all_countries.add(record["origin_country"])
            all_plots.update(record.get("origin_plot_ids", []))
            all_models.add(record["custody_model"])

            nodes.append(
                GenealogyNode(
                    batch_id=current_id,
                    batch_reference=record.get("batch_reference"),
                    status=record["status"],
                    origin_type=record["origin_type"],
                    commodity=record["commodity"],
                    quantity=record["quantity"],
                    custody_model=record["custody_model"],
                    facility_id=record["facility_id"],
                    origin_country=record["origin_country"],
                    depth=depth,
                    parent_batch_ids=record.get("parent_batch_ids", []),
                    child_batch_ids=record.get("child_batch_ids", []),
                    created_at=record.get("created_at", datetime.now(timezone.utc)),
                )
            )

            # Traverse parents and children
            for parent_id in record.get("parent_batch_ids", []):
                if parent_id not in visited:
                    queue.append((parent_id, depth + 1))
            for child_id in record.get("child_batch_ids", []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        genealogy_data = {
            "batch_id": batch_id,
            "total_nodes": len(nodes),
            "max_depth": max_depth,
        }
        provenance_hash = _compute_provenance_hash(genealogy_data)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Genealogy retrieved: batch=%s nodes=%d depth=%d",
            batch_id,
            len(nodes),
            max_depth,
        )

        return BatchGenealogyResponse(
            batch_id=batch_id,
            nodes=nodes,
            total_nodes=len(nodes),
            max_depth=max_depth,
            origin_countries=list(all_countries),
            origin_plot_ids=list(all_plots),
            custody_models_used=list(all_models),
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve genealogy for batch %s: %s",
            batch_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve batch genealogy",
        )


# ---------------------------------------------------------------------------
# POST /batches/search
# ---------------------------------------------------------------------------


@router.post(
    "/batches/search",
    response_model=BatchSearchResponse,
    summary="Search batches",
    description=(
        "Search batches with flexible filters including commodity, "
        "facility, status, CoC model, origin, date range, and quantity."
    ),
    responses={
        200: {"description": "Search results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def search_batches(
    request: Request,
    body: BatchSearchRequest,
    user: AuthUser = Depends(
        require_permission("eudr-coc:batches:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchSearchResponse:
    """Search batches with filters.

    Args:
        body: Search criteria with filters and pagination.
        user: Authenticated user with batches:read permission.

    Returns:
        BatchSearchResponse with matching batches and pagination.
    """
    start = time.monotonic()
    try:
        store = _get_batch_store()
        results: List[Dict] = []

        for record in store.values():
            # Apply filters
            if body.commodity and record["commodity"] != body.commodity:
                continue
            if body.facility_id and record["facility_id"] != body.facility_id:
                continue
            if body.status and record["status"] != body.status:
                continue
            if body.custody_model and record["custody_model"] != body.custody_model:
                continue
            if body.origin_country and record["origin_country"] != body.origin_country:
                continue
            if body.supplier_id and record.get("supplier_id") != body.supplier_id:
                continue
            if body.date_from and record.get("created_at", datetime.min) < body.date_from:
                continue
            if body.date_to and record.get("created_at", datetime.max) > body.date_to:
                continue
            if body.batch_reference and body.batch_reference.lower() not in (record.get("batch_reference") or "").lower():
                continue
            if body.min_quantity and record["quantity"].amount < body.min_quantity:
                continue
            if body.max_quantity and record["quantity"].amount > body.max_quantity:
                continue
            results.append(record)

        # Sort
        reverse = body.sort_order == "desc"
        if body.sort_by == "quantity":
            results.sort(key=lambda r: r["quantity"].amount, reverse=reverse)
        elif body.sort_by == "batch_id":
            results.sort(key=lambda r: r["batch_id"], reverse=reverse)
        else:
            results.sort(
                key=lambda r: r.get("created_at", datetime.min),
                reverse=reverse,
            )

        total = len(results)
        page = results[body.offset: body.offset + body.limit]

        batch_responses = [
            BatchResponse(**r, processing_time_ms=0.0) for r in page
        ]

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch search: total=%d returned=%d", total, len(page)
        )

        return BatchSearchResponse(
            batches=batch_responses,
            meta=PaginatedMeta(
                total=total,
                limit=body.limit,
                offset=body.offset,
                has_more=(body.offset + body.limit) < total,
            ),
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to search batches: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search batches",
        )
