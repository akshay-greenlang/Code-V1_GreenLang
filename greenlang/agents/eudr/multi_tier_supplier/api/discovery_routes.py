# -*- coding: utf-8 -*-
"""
Discovery Routes - AGENT-EUDR-008 Multi-Tier Supplier Tracker API

Endpoints for discovering sub-tier suppliers from declarations,
questionnaires, certification databases, and batch bulk data.

Endpoints:
    POST /discover                  - Discover sub-tier suppliers
    POST /discover/batch            - Batch discovery
    POST /discover/from-declaration - Discovery from supplier declaration
    POST /discover/from-questionnaire - Discovery from questionnaire

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.multi_tier_supplier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_supplier_service,
    rate_limit_batch,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.multi_tier_supplier.api.schemas import (
    BatchDiscoverRequestSchema,
    BatchDiscoverResponseSchema,
    DeclarationDiscoverRequestSchema,
    DiscoverRequestSchema,
    DiscoverResponseSchema,
    QuestionnaireDiscoverRequestSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Supplier Discovery"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /discover
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=DiscoverResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Discover sub-tier suppliers",
    description=(
        "Discover sub-tier suppliers for a given Tier 1 supplier and commodity. "
        "Searches declarations, certification databases, shipping documents, "
        "and ERP data to build the supplier hierarchy recursively up to the "
        "configured maximum depth (default 15 tiers)."
    ),
    responses={
        200: {"description": "Suppliers discovered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def discover_suppliers(
    body: DiscoverRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:discover:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DiscoverResponseSchema:
    """Discover sub-tier suppliers for a given root supplier.

    Uses multiple data sources to recursively build the supplier
    hierarchy from Tier 1 down to the origin farm/plot level.

    Args:
        body: Discovery request with supplier ID, commodity, and options.
        request: FastAPI request object.
        user: Authenticated user with discover:write permission.

    Returns:
        DiscoverResponseSchema with discovered suppliers and metadata.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Discover request: user=%s supplier_id=%s commodity=%s max_depth=%d",
        user.user_id,
        body.supplier_id,
        body.commodity,
        body.max_depth,
    )

    try:
        service = get_supplier_service()

        result = service.discover_suppliers(
            supplier_id=body.supplier_id,
            commodity=body.commodity,
            max_depth=body.max_depth,
            sources=body.sources,
            include_inferred=body.include_inferred,
            country_filter=body.country_filter,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"{body.supplier_id}|{body.commodity}|"
            f"{result.get('total_discovered', 0)}|{elapsed}"
        )

        logger.info(
            "Discovery completed: user=%s supplier_id=%s total=%d "
            "max_depth_reached=%d elapsed_ms=%.1f",
            user.user_id,
            body.supplier_id,
            result.get("total_discovered", 0),
            result.get("max_depth_reached", 0),
            elapsed * 1000,
        )

        return DiscoverResponseSchema(
            supplier_id=body.supplier_id,
            commodity=body.commodity,
            total_discovered=result.get("total_discovered", 0),
            max_depth_reached=result.get("max_depth_reached", 0),
            discovered_suppliers=result.get("discovered_suppliers", []),
            discovery_sources_used=result.get("discovery_sources_used", body.sources),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Discover validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Discovery validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Discover failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supplier discovery failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /discover/batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=BatchDiscoverResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Batch discovery of sub-tier suppliers",
    description=(
        "Submit multiple supplier discovery requests in a single batch. "
        "Each request is processed independently with individual error "
        "handling. Maximum 100 discovery requests per batch."
    ),
    responses={
        200: {"description": "Batch discovery completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_discover_suppliers(
    body: BatchDiscoverRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:discover:write")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchDiscoverResponseSchema:
    """Batch discover sub-tier suppliers for multiple root suppliers.

    Processes each discovery request independently. Errors on individual
    requests do not block the entire batch.

    Args:
        body: Batch discovery request with list of individual requests.
        request: FastAPI request object.
        user: Authenticated user with discover:write permission.

    Returns:
        BatchDiscoverResponseSchema with results and per-request errors.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Batch discover request: user=%s count=%d",
        user.user_id,
        len(body.discoveries),
    )

    try:
        service = get_supplier_service()
        results: List[DiscoverResponseSchema] = []
        errors: List[Dict[str, Any]] = []
        total_discovered = 0

        for idx, disc_req in enumerate(body.discoveries):
            try:
                result = service.discover_suppliers(
                    supplier_id=disc_req.supplier_id,
                    commodity=disc_req.commodity,
                    max_depth=disc_req.max_depth,
                    sources=disc_req.sources,
                    include_inferred=disc_req.include_inferred,
                    country_filter=disc_req.country_filter,
                )

                item_elapsed = time.monotonic() - start
                provenance = _compute_provenance(
                    f"{disc_req.supplier_id}|{disc_req.commodity}|"
                    f"{result.get('total_discovered', 0)}"
                )

                resp = DiscoverResponseSchema(
                    supplier_id=disc_req.supplier_id,
                    commodity=disc_req.commodity,
                    total_discovered=result.get("total_discovered", 0),
                    max_depth_reached=result.get("max_depth_reached", 0),
                    discovered_suppliers=result.get("discovered_suppliers", []),
                    discovery_sources_used=result.get(
                        "discovery_sources_used", disc_req.sources
                    ),
                    elapsed_ms=item_elapsed * 1000,
                    provenance_hash=provenance,
                )
                results.append(resp)
                total_discovered += result.get("total_discovered", 0)

            except Exception as item_exc:
                errors.append({
                    "index": idx,
                    "supplier_id": disc_req.supplier_id,
                    "error": str(item_exc),
                })

        elapsed = time.monotonic() - start
        batch_provenance = _compute_provenance(
            f"batch|{len(results)}|{total_discovered}|{elapsed}"
        )

        logger.info(
            "Batch discover completed: user=%s total_requests=%d "
            "total_discovered=%d errors=%d elapsed_ms=%.1f",
            user.user_id,
            len(body.discoveries),
            total_discovered,
            len(errors),
            elapsed * 1000,
        )

        return BatchDiscoverResponseSchema(
            total_requests=len(body.discoveries),
            total_discovered=total_discovered,
            results=results,
            errors=errors,
            elapsed_ms=elapsed * 1000,
            provenance_hash=batch_provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Batch discover validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch discovery validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Batch discover failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch supplier discovery failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /discover/from-declaration
# ---------------------------------------------------------------------------


@router.post(
    "/from-declaration",
    response_model=DiscoverResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Discover suppliers from declaration document",
    description=(
        "Parse a supplier declaration document to extract sub-tier "
        "supplier information and build the supplier hierarchy. "
        "Supports structured and semi-structured declaration formats."
    ),
    responses={
        200: {"description": "Discovery from declaration completed"},
        400: {"model": ErrorResponse, "description": "Invalid input or parse error"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def discover_from_declaration(
    body: DeclarationDiscoverRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:discover:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DiscoverResponseSchema:
    """Discover suppliers from a supplier declaration document.

    Parses the declaration text to identify sub-tier suppliers,
    their relationships, and associated metadata.

    Args:
        body: Declaration discovery request with text and context.
        request: FastAPI request object.
        user: Authenticated user with discover:write permission.

    Returns:
        DiscoverResponseSchema with discovered suppliers.

    Raises:
        HTTPException: 400 on parse error, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Declaration discover request: user=%s declaring_supplier=%s "
        "commodity=%s text_length=%d",
        user.user_id,
        body.declaring_supplier_id,
        body.commodity,
        len(body.declaration_text),
    )

    try:
        service = get_supplier_service()

        result = service.discover_from_declaration(
            declaration_text=body.declaration_text,
            declaring_supplier_id=body.declaring_supplier_id,
            commodity=body.commodity,
            max_depth=body.max_depth,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"declaration|{body.declaring_supplier_id}|{body.commodity}|"
            f"{result.get('total_discovered', 0)}"
        )

        logger.info(
            "Declaration discover completed: user=%s supplier=%s "
            "total=%d elapsed_ms=%.1f",
            user.user_id,
            body.declaring_supplier_id,
            result.get("total_discovered", 0),
            elapsed * 1000,
        )

        return DiscoverResponseSchema(
            supplier_id=body.declaring_supplier_id,
            commodity=body.commodity,
            total_discovered=result.get("total_discovered", 0),
            max_depth_reached=result.get("max_depth_reached", 0),
            discovered_suppliers=result.get("discovered_suppliers", []),
            discovery_sources_used=["declaration"],
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Declaration discover error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Declaration parsing failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Declaration discover failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Declaration discovery failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /discover/from-questionnaire
# ---------------------------------------------------------------------------


@router.post(
    "/from-questionnaire",
    response_model=DiscoverResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Discover suppliers from questionnaire responses",
    description=(
        "Parse supplier questionnaire response data to extract sub-tier "
        "supplier information. Supports structured questionnaire formats "
        "with nested supplier declarations."
    ),
    responses={
        200: {"description": "Discovery from questionnaire completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def discover_from_questionnaire(
    body: QuestionnaireDiscoverRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:discover:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DiscoverResponseSchema:
    """Discover suppliers from questionnaire response data.

    Parses structured questionnaire data to identify sub-tier
    suppliers and build the hierarchy.

    Args:
        body: Questionnaire discovery request with response data.
        request: FastAPI request object.
        user: Authenticated user with discover:write permission.

    Returns:
        DiscoverResponseSchema with discovered suppliers.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Questionnaire discover request: user=%s responding_supplier=%s "
        "commodity=%s",
        user.user_id,
        body.responding_supplier_id,
        body.commodity,
    )

    try:
        service = get_supplier_service()

        result = service.discover_from_questionnaire(
            questionnaire_data=body.questionnaire_data,
            responding_supplier_id=body.responding_supplier_id,
            commodity=body.commodity,
            max_depth=body.max_depth,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"questionnaire|{body.responding_supplier_id}|{body.commodity}|"
            f"{result.get('total_discovered', 0)}"
        )

        logger.info(
            "Questionnaire discover completed: user=%s supplier=%s "
            "total=%d elapsed_ms=%.1f",
            user.user_id,
            body.responding_supplier_id,
            result.get("total_discovered", 0),
            elapsed * 1000,
        )

        return DiscoverResponseSchema(
            supplier_id=body.responding_supplier_id,
            commodity=body.commodity,
            total_discovered=result.get("total_discovered", 0),
            max_depth_reached=result.get("max_depth_reached", 0),
            discovered_suppliers=result.get("discovered_suppliers", []),
            discovery_sources_used=["questionnaire"],
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Questionnaire discover error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Questionnaire parsing failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Questionnaire discover failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Questionnaire discovery failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
